from __future__ import annotations

from typing import Any

from .config import (
    GLOBAL_L1_TOP,
    SEARCH_EXACT_MAX,
    SEARCH_L1_TOP,
    SEARCH_NEXT_TOP,
    DEFAULT_CACHE_POLICY,
    CONFIDENCE_DECAY_ENABLED,
    CONFIDENCE_DECAY_MAX_DELTA,
    CONFIDENCE_DECAY_PENALTY,
    all_layer_thresholds,
    layer_thresholds,
)
from .dependencies import ensure_numpy
from .errors import RouterError
from .indexes import VectorIndex, build_index
from .models import LayeredSignature
from .routing_policy import child_ids, top_indices


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
        self.layer_indexes: dict[tuple[str, int], VectorIndex] = {}
        self.global_l1_entries: list[tuple[str, int]] = []
        self.global_l1_index: VectorIndex | None = None
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
                results.append(self.search_signature(signature, prompt_embedding))
            else:
                results.append(result)

        if not results:
            raise RouterError("No usable vectors found in signature file.")

        for result in results:
            self.apply_confidence_decay(result)
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
        np = ensure_numpy()
        self.layer_indexes.clear()
        self.global_l1_entries.clear()
        self.global_l1_index = None
        self.index_backends.clear()
        if self.memory_limit_bytes:
            self.index_backends.add("chunked-flat")
            return
        l1_vectors: list[Any] = []
        for facet, signature in sorted(self.signatures.items()):
            levels = signature.levels()
            if not levels:
                continue
            first_level = levels[0]
            first_layer = signature.get_layer(first_level)
            for vector_index, vector in enumerate(first_layer.vectors):
                self.global_l1_entries.append((facet, int(vector_index)))
                l1_vectors.append(vector)
        if l1_vectors:
            matrix = np.asarray(l1_vectors, dtype=np.float32)
            self.global_l1_index = build_index(matrix, "flat", layer=1)
            self.index_backends.add(self.global_l1_index.backend)

    def index_backend_label(self) -> str:
        if not self.index_backends:
            return "none"
        return "+".join(sorted(self.index_backends))

    def loading_policy_label(self) -> str:
        if not self.memory_limit_bytes:
            return "warm-l1"
        return f"lazy-chunked:{self.memory_limit_mb:g}MB"

    def cache_policy_label(self) -> str:
        if self.memory_limit_bytes:
            return "release_all"
        return self.cache_policy

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

    def facets_by_decision(self, results: list[dict[str, Any]]) -> dict[str, list[str]]:
        grouped = {"accepted": [], "passed": [], "refused": []}
        for result in results:
            decision = str(result.get("decision", "refused"))
            if decision not in grouped:
                decision = "refused"
            grouped[decision].append(str(result["facet"]))
        return {key: sorted(values) for key, values in grouped.items()}

    def global_l1_search(self, prompt_embedding: Any) -> list[dict[str, Any]]:
        if self.global_l1_index is None:
            return self.chunked_l1_search(prompt_embedding)
        top_k = min(len(self.global_l1_entries), max(GLOBAL_L1_TOP, len(self.global_l1_entries)))
        matches = self.global_l1_index.search(prompt_embedding, top_k)
        self.last_l1_scanned = len(self.global_l1_entries)
        grouped: dict[str, list[tuple[int, float]]] = {}
        for match in matches:
            facet, vector_index = self.global_l1_entries[match.index]
            grouped.setdefault(facet, []).append((vector_index, match.score))

        results: list[dict[str, Any]] = []
        for facet, signature in self.signatures.items():
            facet_matches = grouped.get(facet, [])
            first_level = signature.levels()[0]
            if facet_matches:
                best_index, best_score = max(facet_matches, key=lambda item: item[1])
                top_scores = [score for _, score in sorted(facet_matches, key=lambda item: item[1], reverse=True)[:5]]
                avg_top_score = float(sum(top_scores) / len(top_scores))
            else:
                layer = signature.get_layer(first_level)
                index = self.layer_indexes.get((facet, first_level))
                local_match = (index or build_index(layer.vectors, "flat", first_level)).search(prompt_embedding, 1)[0]
                best_index, best_score = local_match.index, local_match.score
                avg_top_score = best_score
            score = 0.7 * best_score + 0.3 * avg_top_score
            trace = self.layer_trace(
                first_level,
                score,
                int(best_index),
                len(self.global_l1_entries),
                len(facet_matches) or 1,
            )
            results.append(
                {
                    "facet": facet,
                    "score": score,
                    "best_vector_score": best_score,
                    "cluster_score": best_score,
                    "vector_index": int(best_index),
                    "exact_count": 0,
                    "mode": f"L{first_level} {trace['decision']}",
                    "vectors_scanned": len(self.global_l1_entries),
                    "decision": trace["decision"],
                    "layers": [trace],
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def chunked_l1_search(self, prompt_embedding: Any) -> list[dict[str, Any]]:
        np = ensure_numpy()
        grouped: dict[str, list[tuple[int, float]]] = {}
        vectors: list[Any] = []
        entries: list[tuple[str, int]] = []
        loaded_facets: set[str] = set()
        bytes_used = 0
        scanned = 0
        budget = self.memory_limit_bytes or 0

        def flush() -> None:
            nonlocal vectors, entries, loaded_facets, bytes_used, scanned
            if not vectors:
                return
            matrix = np.asarray(vectors, dtype=np.float32)
            scores = matrix @ prompt_embedding
            for (facet, vector_index), score in zip(entries, scores.tolist()):
                grouped.setdefault(facet, []).append((vector_index, float(score)))
            scanned += len(entries)
            for facet in loaded_facets:
                signature = self.signatures.get(facet)
                if signature is not None and signature.levels():
                    signature.release_layer(signature.levels()[0])
            vectors = []
            entries = []
            loaded_facets = set()
            bytes_used = 0

        for facet, signature in sorted(self.signatures.items()):
            levels = signature.levels()
            if not levels:
                continue
            first_level = levels[0]
            layer = signature.get_layer(first_level)
            layer_bytes = int(getattr(layer.vectors, "nbytes", 0))
            if budget and vectors and bytes_used + layer_bytes > budget:
                flush()
            loaded_facets.add(facet)
            for vector_index, vector in enumerate(layer.vectors):
                entries.append((facet, int(vector_index)))
                vectors.append(vector)
            bytes_used += layer_bytes
            if budget and bytes_used >= budget:
                flush()
        flush()
        self.last_l1_scanned = scanned

        results: list[dict[str, Any]] = []
        for facet in self.signatures:
            facet_matches = grouped.get(facet, [])
            if not facet_matches:
                continue
            best_index, best_score = max(facet_matches, key=lambda item: item[1])
            top_scores = [score for _, score in sorted(facet_matches, key=lambda item: item[1], reverse=True)[:5]]
            avg_top_score = float(sum(top_scores) / len(top_scores))
            first_level = self.signatures[facet].levels()[0]
            score = 0.7 * best_score + 0.3 * avg_top_score
            trace = self.layer_trace(
                first_level,
                score,
                int(best_index),
                scanned,
                len(facet_matches),
            )
            results.append(
                {
                    "facet": facet,
                    "score": score,
                    "best_vector_score": best_score,
                    "cluster_score": best_score,
                    "vector_index": int(best_index),
                    "exact_count": 0,
                    "mode": f"L{first_level} {trace['decision']}",
                    "vectors_scanned": scanned,
                    "decision": trace["decision"],
                    "layers": [trace],
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def coarse_search_signature(self, signature: LayeredSignature, prompt_embedding: Any) -> dict[str, Any]:
        np = ensure_numpy()
        first_level = signature.levels()[0]
        layer = signature.get_layer(first_level)
        similarities = layer.vectors @ prompt_embedding
        top = top_indices(similarities, min(5, len(similarities)))
        best_index = top[0]
        best_score = float(similarities[best_index])
        avg_top_score = float(np.mean(similarities[top]))
        score = 0.7 * best_score + 0.3 * avg_top_score
        trace = self.layer_trace(first_level, score, int(best_index), len(layer.vectors), len(layer.vectors))
        return {
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

    def search_signature(self, signature: LayeredSignature, prompt_embedding: Any) -> dict[str, Any]:
        np = ensure_numpy()
        levels = signature.levels()
        last_level = levels[-1]

        if len(levels) == 1:
            last_layer = signature.get_layer(last_level)
            index = build_index(last_layer.vectors, self.index_policy, last_level)
            self.index_backends.add(index.backend)
            matches = index.search(prompt_embedding, min(5, len(last_layer.vectors)))
            best_index = matches[0].index
            best_score = matches[0].score
            avg_top_score = float(np.mean([match.score for match in matches]))
            score = 0.7 * best_score + 0.3 * avg_top_score
            trace = self.layer_trace(last_level, score, int(best_index), len(last_layer.vectors), len(last_layer.vectors))
            return {
                "facet": signature.facet,
                "score": score,
                "best_vector_score": best_score,
                "cluster_score": 0.0,
                "vector_index": int(best_index),
                "exact_count": len(last_layer.vectors),
                "mode": f"L{last_level} {trace['decision']}",
                "vectors_scanned": len(last_layer.vectors),
                "decision": trace["decision"],
                "layers": [trace],
            }

        candidate_ids: list[int] | None = None
        best_cluster_score = 0.0
        vectors_scanned = 0
        traces: list[dict[str, Any]] = []
        for position, level in enumerate(levels[:-1]):
            layer = signature.get_layer(level)
            if candidate_ids is None:
                local_vectors = layer.vectors
                source_ids = list(range(len(layer.vectors)))
                index = build_index(layer.vectors, self.index_policy, level)
                self.index_backends.add(index.backend)
                matches = index.search(
                    prompt_embedding,
                    SEARCH_L1_TOP if position == 0 else SEARCH_NEXT_TOP,
                )
                top_source_ids = [match.index for match in matches]
                vectors_scanned += len(layer.vectors)
                if matches:
                    best_cluster_score = matches[0].score
            else:
                local_vectors = layer.vectors[candidate_ids]
                source_ids = candidate_ids
                scores = local_vectors @ prompt_embedding
                limit = SEARCH_L1_TOP if position == 0 else SEARCH_NEXT_TOP
                top_local = top_indices(scores, limit)
                top_source_ids = [source_ids[index] for index in top_local]
                vectors_scanned += len(candidate_ids)
                if top_local:
                    best_cluster_score = float(scores[top_local[0]])

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
                if self.memory_limit_bytes:
                    signature.release_layers_except(set())
                return result

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
                if self.memory_limit_bytes:
                    signature.release_layers_except(set())
                return result

            next_limit = SEARCH_EXACT_MAX if levels[position + 1] == last_level else None
            candidate_ids = child_ids(layer.links, top_source_ids, next_limit)
            if not candidate_ids:
                next_layer = signature.get_layer(levels[position + 1])
                fallback_limit = next_limit or len(next_layer.vectors)
                candidate_ids = list(range(min(len(next_layer.vectors), fallback_limit)))
                if self.memory_limit_bytes:
                    signature.release_layer(levels[position + 1])
            if self.memory_limit_bytes:
                signature.release_layer(level)

        last_layer = signature.get_layer(last_level)
        candidates = candidate_ids or list(range(min(len(last_layer.vectors), SEARCH_EXACT_MAX)))
        candidate_matrix = last_layer.vectors[candidates]
        candidate_scores = candidate_matrix @ prompt_embedding
        top_candidate_local = top_indices(candidate_scores, min(5, len(candidate_scores)))
        best_local = top_candidate_local[0]
        best_index = int(candidates[best_local])
        best_anchor = float(candidate_scores[best_local])
        avg_top_score = float(np.mean(candidate_scores[top_candidate_local]))
        final_score = 0.6 * best_anchor + 0.25 * avg_top_score + 0.15 * best_cluster_score
        final_trace = self.layer_trace(
            last_level,
            final_score,
            best_index,
            len(candidates),
            len(candidates),
        )
        traces.append(final_trace)

        return {
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

    def release_loaded_layers(self) -> None:
        for signature in self.signatures.values():
            signature.release_layers_except(set())

    def release_after_route(self) -> None:
        if self.memory_limit_bytes:
            self.release_loaded_layers()
            return
        if self.cache_policy == "keep_l1_only":
            for signature in self.signatures.values():
                levels = signature.levels()
                keep = {levels[0]} if levels else set()
                signature.release_layers_except(keep)


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
        signature, prompt_embedding
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
                "confidence_decay": result.get("confidence_decay", 0.0),
                "entropy": result.get("entropy", 0.0),
                "layers": result.get("layers", []),
            }
            for index, result in enumerate(route["results"], start=1)
        ],
    }
