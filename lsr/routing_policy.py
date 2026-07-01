from __future__ import annotations

from typing import Any

from .dependencies import ensure_numpy


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
