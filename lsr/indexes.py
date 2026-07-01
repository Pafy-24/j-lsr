from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import INDEX_HNSW_EF_CONSTRUCTION, INDEX_HNSW_EF_SEARCH, INDEX_HNSW_M, INDEX_HNSW_THRESHOLD
from .dependencies import ensure_faiss, ensure_numpy
from .errors import RouterError


@dataclass
class SearchResult:
    index: int
    score: float


class VectorIndex:
    backend = "base"

    def __init__(self, vectors: Any):
        self.vectors = vectors

    def search(self, query: Any, top_k: int) -> list[SearchResult]:
        raise NotImplementedError


class FlatIPIndex(VectorIndex):
    backend = "flat"

    def search(self, query: Any, top_k: int) -> list[SearchResult]:
        np = ensure_numpy()
        if len(self.vectors) == 0:
            return []
        limit = min(max(1, int(top_k)), len(self.vectors))
        scores = self.vectors @ query
        if limit == len(scores):
            indexes = np.argsort(scores)[::-1]
        else:
            partial = np.argpartition(scores, -limit)[-limit:]
            indexes = partial[np.argsort(scores[partial])[::-1]]
        return [SearchResult(int(index), float(scores[index])) for index in indexes]


class FaissHNSWIndex(VectorIndex):
    backend = "hnsw"

    def __init__(self, vectors: Any):
        super().__init__(vectors)
        np = ensure_numpy()
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise RouterError("HNSW index expects a 2D vector matrix.")
        faiss = ensure_faiss()
        self.index = faiss.IndexHNSWFlat(int(matrix.shape[1]), INDEX_HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = INDEX_HNSW_EF_CONSTRUCTION
        self.index.hnsw.efSearch = INDEX_HNSW_EF_SEARCH
        self.index.add(matrix)

    def search(self, query: Any, top_k: int) -> list[SearchResult]:
        np = ensure_numpy()
        if len(self.vectors) == 0:
            return []
        limit = min(max(1, int(top_k)), len(self.vectors))
        query_matrix = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores, indexes = self.index.search(query_matrix, limit)
        return [
            SearchResult(int(index), float(score))
            for index, score in zip(indexes[0].tolist(), scores[0].tolist())
            if int(index) >= 0
        ]


def choose_index_backend(n_vectors: int, policy: str = "auto", layer: int | None = None) -> str:
    if policy not in {"auto", "flat", "hnsw"}:
        raise RouterError(f"Unsupported --index value: {policy}")
    if policy in {"flat", "hnsw"}:
        return policy
    if layer == 1:
        return "flat"
    return "hnsw" if n_vectors > INDEX_HNSW_THRESHOLD else "flat"


def build_index(vectors: Any, policy: str = "auto", layer: int | None = None) -> VectorIndex:
    backend = choose_index_backend(len(vectors), policy, layer)
    if backend == "flat":
        return FlatIPIndex(vectors)
    return FaissHNSWIndex(vectors)
