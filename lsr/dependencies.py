from __future__ import annotations

from typing import Any

from .errors import RouterError

np = None
SentenceTransformer = None
MiniBatchKMeans = None


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


def make_table(*args: Any, **kwargs: Any) -> Any:
    from rich.table import Table

    return Table(*args, **kwargs)
