from __future__ import annotations

import os
from typing import Any

from .dependencies import ensure_numpy, get_sentence_transformer_class
from .devices import select_processing_unit
from .errors import RouterError


def normalize_matrix(matrix: Any) -> Any:
    np = ensure_numpy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        pu_index: int | None = None,
        online: bool = False,
        model: Any | None = None,
    ):
        self.model_name = model_name
        self.pu_index = pu_index
        self.online = online
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def embed(self, texts: list[str]) -> Any:
        return embed_texts(self.model, texts)

    def _load_model(self) -> Any:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        unit = select_processing_unit(self.pu_index)
        sentence_transformer_class = get_sentence_transformer_class()
        kwargs = {"local_files_only": not self.online}
        try:
            if unit is None:
                return sentence_transformer_class(self.model_name, **kwargs)
            return sentence_transformer_class(self.model_name, device=unit.device, **kwargs)
        except OSError as exc:
            if not self.online:
                raise RouterError(
                    f"Model not found in local cache: {self.model_name}. "
                    "Run once with --online to download/cache it, or pass --model with a local model path."
                ) from exc
            raise RouterError(f"Could not load model {self.model_name}: {exc}") from exc


def embed_texts(model: Any, texts: list[str]) -> Any:
    np = ensure_numpy()
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_model(model_name: str, pu_index: int | None = None, online: bool = False) -> Any:
    return EmbeddingService(model_name, pu_index, online).model


def model_tensor_bytes_by_device(model: Any) -> dict[str, int]:
    """Count owned model parameters/buffers by their actual tensor device."""
    totals: dict[str, int] = {}
    seen: set[int] = set()
    for accessor in ("parameters", "buffers"):
        values = getattr(model, accessor, None)
        if not callable(values):
            continue
        for tensor in values():
            identity = id(tensor)
            if identity in seen:
                continue
            seen.add(identity)
            try:
                size = int(tensor.nelement()) * int(tensor.element_size())
                device = str(getattr(tensor, "device", "cpu"))
            except (AttributeError, TypeError, ValueError):
                continue
            totals[device] = totals.get(device, 0) + size
    return totals
