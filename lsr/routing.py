from __future__ import annotations

from typing import Any

from .config import (
    ACTIVE_FACETS_DEEP,
    ACTIVE_FACETS_L2,
    ACTIVE_FACETS_L3,
    COARSE_DESCEND_MARGIN,
    COARSE_DESCEND_MIN_SCORE,
    DEEP_FACET_MULTIPLIER,
    GLOBAL_L1_TOP,
    LAYER_ACCEPT_THRESHOLD,
    LAYER_DESCEND_THRESHOLD,
    MAX_DEEP_FACETS,
    SEARCH_EXACT_MAX,
    SEARCH_L1_TOP,
    SEARCH_NEXT_TOP,
)
from .dependencies import ensure_numpy
from .errors import RouterError
from .indexes import FlatIPIndex, VectorIndex, build_index
from .models import LayeredSignature


class LSRRouter:
    def __init__(
        self,
        signatures: dict[str, LayeredSignature],
        top_k: int = 3,
        index_policy: str = "auto",
        model_name: str = "",
        memory_limit_mb: float = 0.0,
        descend_threshold: float = LAYER_DESCEND_THRESHOLD,
        accept_threshold: float = LAYER_ACCEPT_THRESHOLD,
    ):
        if top_k < 1:
            raise RouterError("--top-k must be at least 1.")
        if accept_threshold < descend_threshold:
            raise RouterError("--accept-threshold must be greater than or equal to --descend-threshold.")
        self.signatures = signatures
        self.top_k = top_k
        self.index_policy = index_policy
        self.model_name = model_name
        self.memory_limit_mb = max(0.0, float(memory_limit_mb or 0.0))
        self.memory_limit_bytes = int(self.memory_limit_mb * 1024 * 1024)
        self.descend_threshold = float(descend_threshold)
        self.accept_threshold = float(accept_threshold)
        self.layer_indexes: dict[tuple[str, int], VectorIndex] = {}
        self.global_l1_entries: list[tuple[str, int]] = []
        self.global_l1_index: VectorIndex | None = None
        self.index_backends: set[str] = set()
        self.last_l1_scanned = 0
        self.build_indexes()

    def route(self, prompt: str, prompt_embedding: Any, top_k: int | None = None) -> dict[str, Any]:
        active_top_k = self.top_k if top_k is None else int(top_k)
        coarse_results = self.global_l1_search(prompt_embedding)
        deep_facets = self.select_deep_facets(coarse_results, active_top_k)
        accepted_facets = {
            str(result["facet"])
            for result in coarse_results
            if float(result["score"]) >= self.accept_threshold
        }

        results: list[dict[str, Any]] = []
        coarse_by_facet = {str(result["facet"]): result for result in coarse_results}
        for facet, signature in self.signatures.items():
            if facet in accepted_facets:
                result = dict(coarse_by_facet[facet])
                result["mode"] = "L1 accepted"
                result["decision"] = "accepted"
                results.append(result)
            elif facet in deep_facets:
                results.append(self.search_signature(signature, prompt_embedding))
            else:
                result = dict(coarse_by_facet[facet])
                result["decision"] = "pruned"
                results.append(result)

        if not results:
            raise RouterError("No usable vectors found in signature file.")

        results.sort(key=lambda item: item["score"], reverse=True)
        global_l1_scanned = self.last_l1_scanned
        deep_scanned = sum(
            int(result.get("vectors_scanned", 0))
            for result in results
            if not str(result.get("mode", "")).lower().startswith(("global l1", "chunked l1", "l1 "))
        )
        route = {
            "input": prompt,
            "results": results[: active_top_k],
            "compared_facets": len(results),
            "deep_facets": sorted(deep_facets),
            "modelName": self.model_name,
            "indexBackend": self.index_backend_label(),
            "vectorsScanned": global_l1_scanned + deep_scanned,
            "activeFacets": {
                "L1": sorted(self.signatures),
                "L2": sorted(deep_facets),
                "L3": [str(result["facet"]) for result in results[: min(ACTIVE_FACETS_L3, len(results))]],
                "final": [str(result["facet"]) for result in results[: min(ACTIVE_FACETS_DEEP, len(results))]],
            },
            "loadingPolicy": self.loading_policy_label(),
            "thresholds": {
                "descend": self.descend_threshold,
                "accept": self.accept_threshold,
            },
        }
        if self.memory_limit_bytes:
            self.release_loaded_layers()
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
            if facet_matches:
                best_index, best_score = max(facet_matches, key=lambda item: item[1])
                top_scores = [score for _, score in sorted(facet_matches, key=lambda item: item[1], reverse=True)[:5]]
                avg_top_score = float(sum(top_scores) / len(top_scores))
            else:
                first_level = sorted(signature.layers)[0]
                layer = signature.layers[first_level]
                index = self.layer_indexes.get((facet, first_level))
                local_match = (index or build_index(layer.vectors, "flat", first_level)).search(prompt_embedding, 1)[0]
                best_index, best_score = local_match.index, local_match.score
                avg_top_score = best_score
            results.append(
                {
                    "facet": facet,
                    "score": 0.7 * best_score + 0.3 * avg_top_score,
                    "best_vector_score": best_score,
                    "cluster_score": best_score,
                    "vector_index": int(best_index),
                    "exact_count": 0,
                    "mode": "global L1 coarse",
                    "vectors_scanned": len(self.global_l1_entries),
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
            results.append(
                {
                    "facet": facet,
                    "score": 0.7 * best_score + 0.3 * avg_top_score,
                    "best_vector_score": best_score,
                    "cluster_score": best_score,
                    "vector_index": int(best_index),
                    "exact_count": 0,
                    "mode": "chunked L1 coarse",
                    "vectors_scanned": scanned,
                    "decision": "descend"
                    if best_score >= self.descend_threshold
                    else "below-threshold",
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
        return {
            "facet": signature.facet,
            "score": 0.7 * best_score + 0.3 * avg_top_score,
            "best_vector_score": best_score,
            "cluster_score": best_score,
            "vector_index": int(best_index),
            "exact_count": 0,
            "mode": f"L{first_level} coarse",
            "vectors_scanned": len(layer.vectors),
            "decision": "descend" if best_score >= self.descend_threshold else "below-threshold",
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
            return {
                "facet": signature.facet,
                "score": 0.7 * best_score + 0.3 * avg_top_score,
                "best_vector_score": best_score,
                "cluster_score": 0.0,
                "vector_index": int(best_index),
                "exact_count": len(last_layer.vectors),
                "mode": "exact",
                "vectors_scanned": len(last_layer.vectors),
                "decision": "accepted",
            }

        candidate_ids: list[int] | None = None
        best_cluster_score = 0.0
        vectors_scanned = 0
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

            if best_cluster_score >= self.accept_threshold:
                result = {
                    "facet": signature.facet,
                    "score": best_cluster_score,
                    "best_vector_score": best_cluster_score,
                    "cluster_score": best_cluster_score,
                    "vector_index": int(top_source_ids[0]) if top_source_ids else 0,
                    "exact_count": 0,
                    "mode": f"L{level} accepted",
                    "vectors_scanned": vectors_scanned,
                    "decision": "accepted",
                }
                if self.memory_limit_bytes:
                    signature.release_layers_except(set())
                return result

            if best_cluster_score < self.descend_threshold:
                result = {
                    "facet": signature.facet,
                    "score": best_cluster_score,
                    "best_vector_score": best_cluster_score,
                    "cluster_score": best_cluster_score,
                    "vector_index": int(top_source_ids[0]) if top_source_ids else 0,
                    "exact_count": 0,
                    "mode": f"L{level} stopped",
                    "vectors_scanned": vectors_scanned,
                    "decision": "below-threshold",
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

        return {
            "facet": signature.facet,
            "score": final_score,
            "best_vector_score": best_anchor,
            "cluster_score": best_cluster_score,
            "vector_index": best_index,
            "exact_count": len(candidates),
            "mode": "deep",
            "vectors_scanned": vectors_scanned + len(candidates),
            "decision": "accepted" if final_score >= self.accept_threshold else "ranked",
        }

    def select_deep_facets(self, coarse_results: list[dict[str, Any]], top_k: int | None = None) -> set[str]:
        return select_deep_facets(
            coarse_results,
            self.top_k if top_k is None else int(top_k),
            self.descend_threshold,
            self.accept_threshold,
        )

    def release_loaded_layers(self) -> None:
        for signature in self.signatures.values():
            signature.release_layers_except(set())


def child_ids(links: list[list[int]], parent_ids: list[int], max_items: int | None = None) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for parent_id in parent_ids:
        if 0 <= parent_id < len(links):
            children = links[parent_id]
        else:
            children = []
        for child_id in children:
            if child_id not in seen:
                seen.add(child_id)
                ids.append(child_id)
                if max_items is not None and len(ids) >= max_items:
                    return ids
    return ids


def top_indices(scores: Any, limit: int) -> list[int]:
    np = ensure_numpy()
    if len(scores) == 0:
        return []
    limit = min(max(1, limit), len(scores))
    if limit == len(scores):
        return np.argsort(scores)[::-1].astype(int).tolist()
    partial = np.argpartition(scores, -limit)[-limit:]
    ordered = partial[np.argsort(scores[partial])[::-1]]
    return ordered.astype(int).tolist()


def should_descend(
    coarse_result: dict[str, Any],
    best_coarse_score: float,
    forced: bool,
    descend_threshold: float = COARSE_DESCEND_MIN_SCORE,
    accept_threshold: float = LAYER_ACCEPT_THRESHOLD,
) -> bool:
    score = float(coarse_result["score"])
    if score >= accept_threshold:
        return False
    if forced:
        return True
    return (
        score >= descend_threshold
        and score >= best_coarse_score - COARSE_DESCEND_MARGIN
    )


def select_deep_facets(
    coarse_results: list[dict[str, Any]],
    top_k: int,
    descend_threshold: float = COARSE_DESCEND_MIN_SCORE,
    accept_threshold: float = LAYER_ACCEPT_THRESHOLD,
) -> set[str]:
    ordered = sorted(coarse_results, key=lambda item: item["score"], reverse=True)
    if not ordered:
        return set()

    max_deep = min(MAX_DEEP_FACETS, ACTIVE_FACETS_L2, len(ordered))
    forced_count = min(max(1, top_k), max_deep)
    soft_limit = min(max_deep, max(forced_count, top_k * DEEP_FACET_MULTIPLIER))
    best_score = float(ordered[0]["score"])

    selected: set[str] = set()
    for index, result in enumerate(ordered):
        if len(selected) >= soft_limit:
            break
        forced = index < forced_count
        if should_descend(result, best_score, forced, descend_threshold, accept_threshold):
            selected.add(str(result["facet"]))

    return selected


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
        descend_threshold=getattr(args, "descend_threshold", LAYER_DESCEND_THRESHOLD),
        accept_threshold=getattr(args, "accept_threshold", LAYER_ACCEPT_THRESHOLD),
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
        descend_threshold=getattr(args, "descend_threshold", LAYER_DESCEND_THRESHOLD),
        accept_threshold=getattr(args, "accept_threshold", LAYER_ACCEPT_THRESHOLD),
    ).route(
        prompt, prompt_embedding
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
            }
            for index, result in enumerate(route["results"], start=1)
        ],
    }
