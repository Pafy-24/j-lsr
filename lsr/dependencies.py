from __future__ import annotations

from typing import Any

from .errors import RouterError

np = None
SentenceTransformer = None
MiniBatchKMeans = None
faiss = None


def ensure_numpy() -> Any:
    global np
    if np is None:
        import numpy as numpy_module

        np = numpy_module
    return np


def get_sentence_transformer_class() -> Any:
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as sentence_transformer_class
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise RouterError(
                "Missing dependency: sentence-transformers. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        SentenceTransformer = sentence_transformer_class
    return SentenceTransformer


def get_kmeans_class() -> Any:
    global MiniBatchKMeans
    if MiniBatchKMeans is None:
        try:
            from sklearn.cluster import MiniBatchKMeans as minibatch_kmeans_class
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise RouterError(
                "Missing dependency: scikit-learn. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        MiniBatchKMeans = minibatch_kmeans_class
    return MiniBatchKMeans


def ensure_faiss() -> Any:
    global faiss
    if faiss is None:
        try:
            import faiss as faiss_module
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise RouterError(
                "Missing dependency: faiss-cpu. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        faiss = faiss_module
    return faiss


def make_table(*args: Any, **kwargs: Any) -> Any:
    from rich.table import Table

    return Table(*args, **kwargs)


def make_ascii_table(*args: Any, **kwargs: Any) -> Any:
    from rich import box
    from rich.table import Table

    kwargs.setdefault("box", box.ASCII)
    return Table(*args, **kwargs)
