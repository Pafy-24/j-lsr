from __future__ import annotations

from typing import Any

from .config import (
    SEARCH_EXACT_MAX,
    SEARCH_L1_TOP,
    SEARCH_NEXT_TOP,
    DEFAULT_CACHE_POLICY,
    CONFIDENCE_DECAY_ENABLED,
    CONFIDENCE_DECAY_MAX_DELTA,
    CONFIDENCE_DECAY_PENALTY,
    INDEX_HNSW_THRESHOLD,
    SCORING_ANCHOR_WEIGHT,
    SCORING_TOP_AVERAGE_WEIGHT,
    SCORING_LAYER_WEIGHT,
    SCORING_LEXICAL_WEIGHT,
    SCORING_GENERIC_PENALTY,
    SCORING_REPETITION_PENALTY,
    SEPARATION_ENABLED,
    SEPARATION_MIN_TOP_DELTA,
    SEPARATION_RATIO_THRESHOLD,
    SEPARATION_PENALTY,
    SEPARATION_REFUSE_DELTA,
    all_layer_thresholds,
    layer_thresholds,
)
from .dependencies import ensure_numpy
from .errors import RouterError
from .indexes import build_index
from .memory import current_footprint_bytes, format_bytes, release_unused_memory
from .models import LayeredSignature
from .routing_policy import child_ids, top_indices
from .scoring import confidence_from_score, generic_penalty, lexical_score, repetition_penalty, weighted_layer_score


class LSRRouter:
    def __init__(
        self,
        signatures: dict[str, LayeredSignature],
        top_k: int = 3,
        index_policy: str = "auto",
        model_name: str = "",
        memory_limit_mb: float = 0.0,
        descend_threshold: float | None = None,
        accept_threshold: float | None = None,
        full_depth: bool = False,
        cache_policy: str = DEFAULT_CACHE_POLICY,
    ):
        if top_k < 1:
            raise RouterError("--top-k must be at least 1.")
        if accept_threshold is not None and descend_threshold is not None and accept_threshold < descend_threshold:
            raise RouterError("--accept-threshold must be greater than or equal to --descend-threshold.")
        self.signatures = signatures
        self.top_k = top_k
        self.index_policy = index_policy
        self.model_name = model_name
        self.memory_limit_mb = max(0.0, float(memory_limit_mb or 0.0))
        self.memory_limit_bytes = int(self.memory_limit_mb * 1024 * 1024)
        self.descend_threshold = None if descend_threshold is None else float(descend_threshold)
        self.accept_threshold = None if accept_threshold is None else float(accept_threshold)
        self.full_depth = full_depth
        self.cache_policy = cache_policy
        self.layer_indexes: dict[tuple[str, int], Any] = {}
        self.global_l1_entries: list[tuple[str, int]] = []
        self.global_l1_index: Any = None
        self.index_backends: set[str] = set()
        self.last_l1_scanned = 0
        self.build_indexes()

    def route(
        self,
        prompt: str,
        prompt_embedding: Any,
        top_k: int | None = None,
        full_depth: bool | None = None,
    ) -> dict[str, Any]:
        original_full_depth = self.full_depth
        if full_depth is not None:
            self.full_depth = bool(full_depth)
        try:
            return self._route(prompt, prompt_embedding, self.top_k if top_k is None else int(top_k))
        finally:
            self.full_depth = original_full_depth

    def _route(self, prompt: str, prompt_embedding: Any, display_limit: int) -> dict[str, Any]:
        coarse_results = self.global_l1_search(prompt_embedding)

        results: list[dict[str, Any]] = []
        coarse_by_facet = {str(result["facet"]): result for result in coarse_results}
        for facet, signature in self.signatures.items():
            result = dict(coarse_by_facet[facet])
            decision = str(result.get("decision", "refused"))
            if decision == "passed" or self.full_depth:
                results.append(self.search_signature(signature, prompt, prompt_embedding))
            else:
                self.apply_scoring_diagnostics(
                    result,
                    prompt,
                    float(result.get("semanticScore", result.get("score", 0.0))),
                    float(result.get("anchorSimilarity", 0.0)),
                )
                results.append(result)

        if not results:
            raise RouterError("No usable vectors found in signature file.")

        for result in results:
            self.apply_confidence_decay(result)
        self.apply_separation(results)
        results.sort(key=lambda item: item["score"], reverse=True)
        global_l1_scanned = self.last_l1_scanned
        deep_scanned = sum(
            int(result.get("vectors_scanned", 0))
            for result in results
            if not str(result.get("mode", "")).lower().startswith(("global l1", "chunked l1", "l1 "))
        )
        route = {
            "input": prompt,
            "results": results[:display_limit],
            "compared_facets": len(results),
            "deep_facets": sorted(
                str(result["facet"])
                for result in results
                if any(int(layer["level"]) > 1 for layer in result.get("layers", []))
            ),
            "modelName": self.model_name,
            "indexBackend": self.index_backend_label(),
            "vectorsScanned": global_l1_scanned + deep_scanned,
            "activeFacets": self.facets_by_decision(results),
            "loadingPolicy": self.loading_policy_label(),
            "thresholds": self.thresholds_label(),
            "fullDepth": self.full_depth,
        }
        self.release_after_route()
        return route

    def build_indexes(self) -> None:
        self.layer_indexes.clear()
        self.global_l1_entries.clear()
        self.global_l1_index = None
        self.index_backends.clear()
        self.index_backends.add("streaming-flat")

    def index_backend_label(self) -> str:
        if not self.index_backends:
            return "none"
        return "+".join(sorted(self.index_backends))

    def loading_policy_label(self) -> str:
        if not self.memory_limit_bytes:
            return "streaming-release-layer"
        return f"streaming-release-layer:{self.memory_limit_mb:g}MB-total"

    def cache_policy_label(self) -> str:
        return "release_after_layer"

    def thresholds_for(self, level: int) -> dict[str, float]:
        thresholds = layer_thresholds(level)
        if self.descend_threshold is not None:
            thresholds["pass"] = self.descend_threshold
        if self.accept_threshold is not None:
            thresholds["accept"] = self.accept_threshold
        return thresholds

    def thresholds_label(self) -> dict[str, Any]:
        if self.descend_threshold is None and self.accept_threshold is None:
            return all_layer_thresholds()
        return {
            "override": {
                "pass": self.descend_threshold,
                "accept": self.accept_threshold,
            }
        }

    def ensure_memory_available(self, extra_bytes: int, context: str) -> None:
        if not self.memory_limit_bytes:
            return
        current = current_footprint_bytes()
        required = current + max(0, int(extra_bytes))
        if required > self.memory_limit_bytes:
            raise RouterError(
                f"Memory limit exceeded before loading {context}: "
                f"current RAM/VRAM footprint {format_bytes(current)}, "
                f"needed chunk {format_bytes(extra_bytes)}, "
                f"limit {format_bytes(self.memory_limit_bytes)}. "
                "Increase --memory-limit-mb or use a smaller model/dataset."
            )

    def load_layer(self, signature: LayeredSignature, level: int) -> Any:
        if level not in signature.layers:
            self.ensure_memory_available(
                signature.estimated_layer_bytes(level),
                f"{signature.facet} L{level}",
            )
        return signature.get_layer(level)

    def release_layer(self, signature: LayeredSignature, level: int) -> None:
        signature.release_layer(level)

    def release_memory(self) -> None:
        release_unused_memory()

    def layer_decision(self, level: int, score: float) -> str:
        thresholds = self.thresholds_for(level)
        if score >= thresholds["accept"]:
            return "accepted"
        if score >= thresholds["pass"]:
            return "passed"
        return "refused"

    def layer_trace(
        self,
        level: int,
        score: float,
        vector_index: int,
        vectors_scanned: int,
        candidates: int,
    ) -> dict[str, Any]:
        thresholds = self.thresholds_for(level)
        delta = 0.0
        decay = 0.0
        return {
            "level": level,
            "score": score,
            "passThreshold": thresholds["pass"],
            "acceptThreshold": thresholds["accept"],
            "decision": self.layer_decision(level, score),
            "deltaFromPrevious": delta,
            "confidenceDecay": decay,
            "entropy": 0.0,
            "vector_index": vector_index,
            "vectors_scanned": vectors_scanned,
            "candidates": candidates,
        }

    def enrich_trace_metrics(self, traces: list[dict[str, Any]]) -> None:
        previous_score: float | None = None
        for trace in traces:
            score = float(trace["score"])
            if previous_score is None:
                trace["deltaFromPrevious"] = 0.0
                trace["confidenceDecay"] = 0.0
            else:
                delta = score - previous_score
                trace["deltaFromPrevious"] = delta
                trace["confidenceDecay"] = max(0.0, -delta)
            trace["entropy"] = confidence_entropy(score)
            previous_score = score

    def apply_confidence_decay(self, result: dict[str, Any]) -> None:
        traces = result.get("layers", [])
        if not traces:
            return
        self.enrich_trace_metrics(traces)
        max_decay = max(float(trace.get("confidenceDecay", 0.0)) for trace in traces)
        result["confidence_decay"] = max_decay
        result["entropy"] = float(traces[-1].get("entropy", 0.0))
        if CONFIDENCE_DECAY_ENABLED and max_decay > CONFIDENCE_DECAY_MAX_DELTA:
            result["score"] = float(result["score"]) - (max_decay * CONFIDENCE_DECAY_PENALTY)

    def apply_scoring_diagnostics(
        self,
        result: dict[str, Any],
        prompt: str,
        semantic_score: float,
        anchor_similarity: float = 0.0,
    ) -> dict[str, Any]:
        has_prompt = bool(prompt.strip())
        result["semanticScore"] = float(semantic_score)
        result["anchorSimilarity"] = float(anchor_similarity)
        result["lexicalScore"] = lexical_score(prompt, str(result["facet"])) if has_prompt else 0.0
        result["genericPenalty"] = generic_penalty(prompt) if has_prompt else 0.0
        result["repetitionPenalty"] = repetition_penalty(prompt) if has_prompt else 0.0
        last_layer = result.get("layers", [{}])[-1]
        result["confidence"] = confidence_from_score(
            float(result["semanticScore"]),
            float(last_layer.get("passThreshold", 0.0)),
            float(last_layer.get("acceptThreshold", 1.0)),
        )
        result["separation"] = 0.0
        result["score"] = (
            float(result["semanticScore"])
            + (SCORING_LEXICAL_WEIGHT * float(result["lexicalScore"]))
            - (SCORING_GENERIC_PENALTY * float(result["genericPenalty"]))
            - (SCORING_REPETITION_PENALTY * float(result["repetitionPenalty"]))
        )
        if (
            has_prompt
            and
            float(result["genericPenalty"]) >= 0.5
            and float(result["confidence"]) < 0.12
            and str(result.get("decision")) in {"accepted", "passed"}
        ):
            result["decision"] = "refused"
            result["mode"] = f"{result.get('mode', '')} generic-refused".strip()
        return result

    def apply_separation(self, results: list[dict[str, Any]]) -> None:
        if not results:
            return
        ordered = sorted(results, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        top_score = max(float(ordered[0].get("score", 0.0)), 1e-9)
        for rank, result in enumerate(ordered):
            score = float(result.get("score", 0.0))
            delta = max(0.0, top_score - score)
            ratio = score / top_score if top_score > 0 else 0.0
            result["separation"] = delta
            result["separationRatio"] = ratio
            if rank == 0 or not SEPARATION_ENABLED:
                continue
            weak_delta = delta >= SEPARATION_REFUSE_DELTA
            weak_ratio = ratio < SEPARATION_RATIO_THRESHOLD and delta >= SEPARATION_MIN_TOP_DELTA
            if weak_delta or weak_ratio:
                result["score"] = score - SEPARATION_PENALTY
                if str(result.get("decision")) in {"accepted", "passed"}:
                    result["decision"] = "refused"
                    result["mode"] = f"{result.get('mode', '')} separation-refused".strip()

    def facets_by_decision(self, results: list[dict[str, Any]]) -> dict[str, list[str]]:
        grouped = {"accepted": [], "passed": [], "refused": []}
        for result in results:
            decision = str(result.get("decision", "refused"))
            if decision not in grouped:
                decision = "refused"
            grouped[decision].append(str(result["facet"]))
        return {key: sorted(values) for key, values in grouped.items()}

    def global_l1_search(self, prompt_embedding: Any) -> list[dict[str, Any]]:
        return self.chunked_l1_search(prompt_embedding)

    def chunked_l1_search(self, prompt_embedding: Any) -> list[dict[str, Any]]:
        np = ensure_numpy()
        scanned = 0
        results: list[dict[str, Any]] = []

        for facet, signature in sorted(self.signatures.items()):
            levels = signature.levels()
            if not levels:
                continue
            first_level = levels[0]
            layer = self.load_layer(signature, first_level)
            try:
                scanned += len(layer.vectors)
                top, top_scores = self.search_layer(layer.vectors, prompt_embedding, min(5, len(layer.vectors)), first_level)
                best_index = int(top[0])
                best_score = float(top_scores[0])
                avg_top_score = float(np.mean(top_scores))
                score = 0.7 * best_score + 0.3 * avg_top_score
                trace = self.layer_trace(
                    first_level,
                    score,
                    best_index,
                    len(layer.vectors),
                    len(layer.vectors),
                )
                result = (
                    {
                        "facet": facet,
                        "score": score,
                        "best_vector_score": best_score,
                        "cluster_score": best_score,
                        "vector_index": best_index,
                        "exact_count": 0,
                        "mode": f"L{first_level} {trace['decision']}",
                        "vectors_scanned": len(layer.vectors),
                        "decision": trace["decision"],
                        "layers": [trace],
                    }
                )
                self.apply_scoring_diagnostics(result, "", score, 0.0)
                results.append(result)
            finally:
                self.release_layer(signature, first_level)
                del layer
                self.release_memory()

        self.last_l1_scanned = scanned
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def coarse_search_signature(self, signature: LayeredSignature, prompt_embedding: Any) -> dict[str, Any]:
        np = ensure_numpy()
        first_level = signature.levels()[0]
        layer = self.load_layer(signature, first_level)
        try:
            top, top_scores = self.search_layer(layer.vectors, prompt_embedding, min(5, len(layer.vectors)), first_level)
            best_index = top[0]
            best_score = float(top_scores[0])
            avg_top_score = float(np.mean(top_scores))
            score = 0.7 * best_score + 0.3 * avg_top_score
            trace = self.layer_trace(first_level, score, int(best_index), len(layer.vectors), len(layer.vectors))
            result = {
                "facet": signature.facet,
                "score": score,
                "best_vector_score": best_score,
                "cluster_score": best_score,
                "vector_index": int(best_index),
                "exact_count": 0,
                "mode": f"L{first_level} {trace['decision']}",
                "vectors_scanned": len(layer.vectors),
                "decision": trace["decision"],
                "layers": [trace],
            }
            return self.apply_scoring_diagnostics(result, "", score, 0.0)
        finally:
            self.release_layer(signature, first_level)
            del layer
            self.release_memory()

    def search_signature(self, signature: LayeredSignature, prompt: str, prompt_embedding: Any) -> dict[str, Any]:
        np = ensure_numpy()
        levels = signature.levels()
        last_level = levels[-1]

        if len(levels) == 1:
            last_layer = self.load_layer(signature, last_level)
            try:
                top, top_scores = self.search_layer(last_layer.vectors, prompt_embedding, min(5, len(last_layer.vectors)), last_level)
                best_index = int(top[0])
                best_score = float(top_scores[0])
                avg_top_score = float(np.mean(top_scores))
                score = 0.7 * best_score + 0.3 * avg_top_score
                trace = self.layer_trace(last_level, score, best_index, len(last_layer.vectors), len(last_layer.vectors))
                result = {
                    "facet": signature.facet,
                    "score": score,
                    "best_vector_score": best_score,
                    "cluster_score": 0.0,
                    "vector_index": best_index,
                    "exact_count": len(last_layer.vectors),
                    "mode": f"L{last_level} {trace['decision']}",
                    "vectors_scanned": len(last_layer.vectors),
                    "decision": trace["decision"],
                    "layers": [trace],
                }
                return self.apply_scoring_diagnostics(result, prompt, score, best_score)
            finally:
                self.release_layer(signature, last_level)
                del last_layer
                self.release_memory()

        candidate_ids: list[int] | None = None
        best_cluster_score = 0.0
        vectors_scanned = 0
        traces: list[dict[str, Any]] = []
        for position, level in enumerate(levels[:-1]):
            layer = self.load_layer(signature, level)
            try:
                if candidate_ids is None:
                    source_ids = list(range(len(layer.vectors)))
                    limit = SEARCH_L1_TOP if position == 0 else SEARCH_NEXT_TOP
                    top_local, top_scores = self.search_layer(layer.vectors, prompt_embedding, limit, level)
                    top_source_ids = [int(index) for index in top_local]
                    vectors_scanned += len(layer.vectors)
                else:
                    source_ids = candidate_ids
                    self.ensure_memory_available(
                        len(candidate_ids) * signature.dimension * 4,
                        f"{signature.facet} L{level} candidate view",
                    )
                    scores = layer.vectors[candidate_ids] @ prompt_embedding
                    limit = SEARCH_L1_TOP if position == 0 else SEARCH_NEXT_TOP
                    top_local = top_indices(scores, limit)
                    top_scores = [float(scores[index]) for index in top_local]
                    top_source_ids = [int(source_ids[index]) for index in top_local]
                    vectors_scanned += len(candidate_ids)
                if len(top_local):
                    best_cluster_score = float(top_scores[0])

                vector_index = int(top_source_ids[0]) if top_source_ids else 0
                trace = self.layer_trace(
                    level,
                    best_cluster_score,
                    vector_index,
                    vectors_scanned,
                    len(source_ids),
                )
                traces.append(trace)

                if trace["decision"] == "accepted" and not self.full_depth:
                    result = {
                        "facet": signature.facet,
                        "score": best_cluster_score,
                        "best_vector_score": best_cluster_score,
                        "cluster_score": best_cluster_score,
                        "vector_index": vector_index,
                        "exact_count": 0,
                        "mode": f"L{level} accepted",
                        "vectors_scanned": vectors_scanned,
                        "decision": "accepted",
                        "layers": traces,
                    }
                    return self.apply_scoring_diagnostics(
                        result,
                        prompt,
                        weighted_layer_score(traces),
                        best_cluster_score,
                    )

                if trace["decision"] == "refused" and not self.full_depth:
                    result = {
                        "facet": signature.facet,
                        "score": best_cluster_score,
                        "best_vector_score": best_cluster_score,
                        "cluster_score": best_cluster_score,
                        "vector_index": vector_index,
                        "exact_count": 0,
                        "mode": f"L{level} stopped",
                        "vectors_scanned": vectors_scanned,
                        "decision": "refused",
                        "layers": traces,
                    }
                    return self.apply_scoring_diagnostics(
                        result,
                        prompt,
                        weighted_layer_score(traces),
                        best_cluster_score,
                    )

                next_limit = SEARCH_EXACT_MAX if levels[position + 1] == last_level else None
                candidate_ids = child_ids(layer.links, top_source_ids, next_limit)
            finally:
                self.release_layer(signature, level)
                del layer
                self.release_memory()

            if not candidate_ids:
                next_level = levels[position + 1]
                next_layer = self.load_layer(signature, next_level)
                try:
                    fallback_limit = next_limit or len(next_layer.vectors)
                    candidate_ids = list(range(min(len(next_layer.vectors), fallback_limit)))
                finally:
                    self.release_layer(signature, next_level)
                    del next_layer
                    self.release_memory()

        last_layer = self.load_layer(signature, last_level)
        try:
            candidates = candidate_ids or list(range(min(len(last_layer.vectors), SEARCH_EXACT_MAX)))
            self.ensure_memory_available(
                len(candidates) * signature.dimension * 4,
                f"{signature.facet} L{last_level} exact candidate view",
            )
            candidate_scores = last_layer.vectors[candidates] @ prompt_embedding
            top_candidate_local = top_indices(candidate_scores, min(5, len(candidate_scores)))
            best_local = top_candidate_local[0]
            best_index = int(candidates[best_local])
            best_anchor = float(candidate_scores[best_local])
            avg_top_score = float(np.mean(candidate_scores[top_candidate_local]))
            layer_score = weighted_layer_score(traces)
            final_score = (
                (SCORING_ANCHOR_WEIGHT * best_anchor)
                + (SCORING_TOP_AVERAGE_WEIGHT * avg_top_score)
                + (SCORING_LAYER_WEIGHT * layer_score)
            )
            final_trace = self.layer_trace(
                last_level,
                final_score,
                best_index,
                len(candidates),
                len(candidates),
            )
            traces.append(final_trace)

            result = {
                "facet": signature.facet,
                "score": final_score,
                "best_vector_score": best_anchor,
                "cluster_score": best_cluster_score,
                "vector_index": best_index,
                "exact_count": len(candidates),
                "mode": "deep",
                "vectors_scanned": vectors_scanned + len(candidates),
                "decision": final_trace["decision"],
                "layers": traces,
            }
            return self.apply_scoring_diagnostics(result, prompt, final_score, best_anchor)
        finally:
            self.release_layer(signature, last_level)
            del last_layer
            self.release_memory()

    def release_loaded_layers(self) -> None:
        for signature in self.signatures.values():
            signature.release_layers_except(set())

    def release_after_route(self) -> None:
        self.release_loaded_layers()
        self.release_memory()

    def search_layer(self, vectors: Any, query: Any, limit: int, level: int) -> tuple[list[int], list[float]]:
        np = ensure_numpy()
        if len(vectors) == 0:
            return [], []
        active_limit = min(max(1, int(limit)), len(vectors))
        use_index = (
            self.index_policy in {"auto", "hnsw", "flat"}
            and not self.memory_limit_bytes
            and (self.index_policy == "hnsw" or len(vectors) > INDEX_HNSW_THRESHOLD)
        )
        if use_index:
            index = build_index(vectors, self.index_policy, level)
            self.index_backends.add(index.backend)
            try:
                matches = index.search(query, active_limit)
                return [match.index for match in matches], [match.score for match in matches]
            finally:
                del index
                self.release_memory()

        scores = vectors @ query
        top = top_indices(scores, active_limit)
        self.index_backends.add("flat")
        return [int(index) for index in top], [float(scores[index]) for index in top]


def coarse_search_signature(signature: LayeredSignature, prompt_embedding: Any) -> dict[str, Any]:
    return LSRRouter({signature.facet: signature}).coarse_search_signature(
        signature, prompt_embedding
    )


def search_signature(signature: LayeredSignature, prompt_embedding: Any, args: Any) -> dict[str, Any]:
    return LSRRouter(
        {signature.facet: signature},
        getattr(args, "top_k", 3),
        getattr(args, "index", "auto"),
        memory_limit_mb=getattr(args, "memory_limit_mb", 0.0),
        descend_threshold=getattr(args, "descend_threshold", None),
        accept_threshold=getattr(args, "accept_threshold", None),
        full_depth=getattr(args, "full_depth", False),
    ).search_signature(
        signature, "", prompt_embedding
    )


def route_embedding(
    prompt: str,
    prompt_embedding: Any,
    signatures: dict[str, LayeredSignature],
    args: Any,
) -> dict[str, Any]:
    return LSRRouter(
        signatures,
        getattr(args, "top_k", 3),
        getattr(args, "index", "auto"),
        memory_limit_mb=getattr(args, "memory_limit_mb", 0.0),
        descend_threshold=getattr(args, "descend_threshold", None),
        accept_threshold=getattr(args, "accept_threshold", None),
        full_depth=getattr(args, "full_depth", False),
    ).route(
        prompt, prompt_embedding
    )


def confidence_entropy(score: float) -> float:
    probability = min(1.0, max(0.0, (float(score) + 1.0) / 2.0))
    if probability in {0.0, 1.0}:
        return 0.0
    import math

    return -(
        probability * math.log2(probability)
        + (1.0 - probability) * math.log2(1.0 - probability)
    )


def json_ready_route(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": route["input"],
        "compared_facets": route["compared_facets"],
        "deep_facets": route["deep_facets"],
        "modelName": route.get("modelName", ""),
        "indexBackend": route.get("indexBackend", ""),
        "vectorsScanned": route.get("vectorsScanned", 0),
        "activeFacets": route.get("activeFacets", {}),
        "loadingPolicy": route.get("loadingPolicy", ""),
        "thresholds": route.get("thresholds", {}),
        "fullDepth": route.get("fullDepth", False),
        "inputMode": route.get("inputMode", "raw"),
        "inputChars": route.get("inputChars", len(str(route.get("input", "")))),
        "routingTextChars": route.get("routingTextChars", len(str(route.get("input", "")))),
        "chunksUsed": route.get("chunksUsed", 1 if route.get("input") else 0),
        "results": [
            {
                "rank": index,
                "facet": result["facet"],
                "score": result["score"],
                "best_vector_score": result["best_vector_score"],
                "cluster_score": result["cluster_score"],
                "exact_count": result["exact_count"],
                "mode": result["mode"],
                "vector_index": result["vector_index"],
                "vectors_scanned": result.get("vectors_scanned", 0),
                "decision": result.get("decision", ""),
                "semanticScore": result.get("semanticScore", result["score"]),
                "confidence": result.get("confidence", 0.0),
                "separation": result.get("separation", 0.0),
                "separationRatio": result.get("separationRatio", 0.0),
                "anchorSimilarity": result.get("anchorSimilarity", 0.0),
                "lexicalScore": result.get("lexicalScore", 0.0),
                "genericPenalty": result.get("genericPenalty", 0.0),
                "repetitionPenalty": result.get("repetitionPenalty", 0.0),
                "confidence_decay": result.get("confidence_decay", 0.0),
                "entropy": result.get("entropy", 0.0),
                "layers": result.get("layers", []),
            }
            for index, result in enumerate(route["results"], start=1)
        ],
    }
