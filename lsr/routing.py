from __future__ import annotations

from typing import Any

from .config import (
    COARSE_DESCEND_MARGIN,
    COARSE_DESCEND_MIN_SCORE,
    DEEP_FACET_MULTIPLIER,
    MAX_DEEP_FACETS,
    SEARCH_EXACT_MAX,
    SEARCH_L1_TOP,
    SEARCH_NEXT_TOP,
)
from .dependencies import ensure_numpy
from .errors import RouterError
from .models import LayeredSignature


class LSRRouter:
    def __init__(self, signatures: dict[str, LayeredSignature], top_k: int = 3):
        if top_k < 1:
            raise RouterError("--top-k must be at least 1.")
        self.signatures = signatures
        self.top_k = top_k

    def route(self, prompt: str, prompt_embedding: Any) -> dict[str, Any]:
        coarse_results = [
            self.coarse_search_signature(signature, prompt_embedding)
            for signature in self.signatures.values()
        ]
        deep_facets = self.select_deep_facets(coarse_results)

        results: list[dict[str, Any]] = []
        coarse_by_facet = {str(result["facet"]): result for result in coarse_results}
        for facet, signature in self.signatures.items():
            if facet in deep_facets:
                results.append(self.search_signature(signature, prompt_embedding))
            else:
                results.append(coarse_by_facet[facet])

        if not results:
            raise RouterError("No usable vectors found in signature file.")

        results.sort(key=lambda item: item["score"], reverse=True)
        return {
            "input": prompt,
            "results": results[: self.top_k],
            "compared_facets": len(results),
            "deep_facets": sorted(deep_facets),
        }

    def coarse_search_signature(self, signature: LayeredSignature, prompt_embedding: Any) -> dict[str, Any]:
        np = ensure_numpy()
        first_level = sorted(signature.layers)[0]
        layer = signature.layers[first_level]
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
        }

    def search_signature(self, signature: LayeredSignature, prompt_embedding: Any) -> dict[str, Any]:
        np = ensure_numpy()
        levels = sorted(signature.layers)
        last_level = levels[-1]
        last_layer = signature.layers[last_level]

        if len(levels) == 1:
            similarities = last_layer.vectors @ prompt_embedding
            top = top_indices(similarities, min(5, len(similarities)))
            best_index = top[0]
            best_score = float(similarities[best_index])
            avg_top_score = float(np.mean(similarities[top]))
            return {
                "facet": signature.facet,
                "score": 0.7 * best_score + 0.3 * avg_top_score,
                "best_vector_score": best_score,
                "cluster_score": 0.0,
                "vector_index": int(best_index),
                "exact_count": len(similarities),
                "mode": "exact",
            }

        candidate_ids: list[int] | None = None
        best_cluster_score = 0.0
        for position, level in enumerate(levels[:-1]):
            layer = signature.layers[level]
            if candidate_ids is None:
                local_vectors = layer.vectors
                source_ids = list(range(len(layer.vectors)))
            else:
                local_vectors = layer.vectors[candidate_ids]
                source_ids = candidate_ids

            scores = local_vectors @ prompt_embedding
            limit = SEARCH_L1_TOP if position == 0 else SEARCH_NEXT_TOP
            top_local = top_indices(scores, limit)
            top_source_ids = [source_ids[index] for index in top_local]
            if top_local:
                best_cluster_score = float(scores[top_local[0]])

            next_limit = SEARCH_EXACT_MAX if levels[position + 1] == last_level else None
            candidate_ids = child_ids(layer.links, top_source_ids, next_limit)
            if not candidate_ids:
                next_layer = signature.layers[levels[position + 1]]
                fallback_limit = next_limit or len(next_layer.vectors)
                candidate_ids = list(range(min(len(next_layer.vectors), fallback_limit)))

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
        }

    def select_deep_facets(self, coarse_results: list[dict[str, Any]]) -> set[str]:
        return select_deep_facets(coarse_results, self.top_k)


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


def should_descend(coarse_result: dict[str, Any], best_coarse_score: float, forced: bool) -> bool:
    if forced:
        return True
    score = float(coarse_result["score"])
    return (
        score >= COARSE_DESCEND_MIN_SCORE
        and score >= best_coarse_score - COARSE_DESCEND_MARGIN
    )


def select_deep_facets(coarse_results: list[dict[str, Any]], top_k: int) -> set[str]:
    ordered = sorted(coarse_results, key=lambda item: item["score"], reverse=True)
    if not ordered:
        return set()

    max_deep = min(MAX_DEEP_FACETS, len(ordered))
    forced_count = min(max(1, top_k), max_deep)
    soft_limit = min(max_deep, max(forced_count, top_k * DEEP_FACET_MULTIPLIER))
    best_score = float(ordered[0]["score"])

    selected: set[str] = set()
    for index, result in enumerate(ordered):
        if len(selected) >= soft_limit:
            break
        forced = index < forced_count
        if should_descend(result, best_score, forced):
            selected.add(str(result["facet"]))

    return selected


def coarse_search_signature(signature: LayeredSignature, prompt_embedding: Any) -> dict[str, Any]:
    return LSRRouter({signature.facet: signature}).coarse_search_signature(
        signature, prompt_embedding
    )


def search_signature(signature: LayeredSignature, prompt_embedding: Any, args: Any) -> dict[str, Any]:
    return LSRRouter({signature.facet: signature}, getattr(args, "top_k", 3)).search_signature(
        signature, prompt_embedding
    )


def route_embedding(
    prompt: str,
    prompt_embedding: Any,
    signatures: dict[str, LayeredSignature],
    args: Any,
) -> dict[str, Any]:
    return LSRRouter(signatures, getattr(args, "top_k", 3)).route(
        prompt, prompt_embedding
    )


def json_ready_route(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": route["input"],
        "compared_facets": route["compared_facets"],
        "deep_facets": route["deep_facets"],
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
            }
            for index, result in enumerate(route["results"], start=1)
        ],
    }
