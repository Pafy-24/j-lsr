from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from .dependencies import ensure_numpy, get_sentence_transformer_class
from .devices import select_processing_unit
from .errors import RouterError
from .rm_integration import managed_processing_unit


def normalize_matrix(matrix: Any) -> Any:
    np = ensure_numpy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def normalize_matrix_inplace(matrix: Any) -> Any:
    """Normalize a writable floating-point matrix without a second matrix copy."""
    np = ensure_numpy()
    if not getattr(matrix, "flags", None) or not matrix.flags.writeable:
        raise ValueError("normalize_matrix_inplace expects a writable matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    np.divide(matrix, norms, out=matrix)
    return matrix


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        pu_index: int | str | None = None,
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
        unit = managed_processing_unit()
        if unit is None:
            unit = select_processing_unit(self.pu_index)
        if unit is not None and unit.backend.startswith("windowsml."):
            from .windows_ml import load_windows_ml_model

            return load_windows_ml_model(self.model_name, unit, self.online)
        sentence_transformer_class = get_sentence_transformer_class()
        kwargs = {"local_files_only": not self.online}
        try:
            if unit is None:
                return sentence_transformer_class(self.model_name, **kwargs)
            model = sentence_transformer_class(self.model_name, device=unit.device, **kwargs)
            model.processing_unit = unit
            model.backend = unit.backend
            return model
        except OSError as exc:
            if not self.online:
                raise RouterError(
                    f"Model not found in local cache: {self.model_name}. "
                    "Run once with --online to download/cache it, or pass --model with a local model path."
                ) from exc
            raise RouterError(f"Could not load model {self.model_name}: {exc}") from exc


@dataclass
class _EmbeddingJob:
    texts: tuple[str, ...]
    future: Future[Any]


class EmbeddingBatcher:
    """Coalesce concurrent synchronous callers into one model invocation."""

    def __init__(
        self,
        model: Any,
        max_batch_units: int = 256,
        max_wait_ms: float = 5.0,
    ):
        if max_batch_units < 1:
            raise ValueError("max_batch_units must be at least 1.")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative.")
        self.model = model
        self.max_batch_units = int(max_batch_units)
        self.max_wait_seconds = float(max_wait_ms) / 1000.0
        self._condition = threading.Condition()
        self._jobs: deque[_EmbeddingJob] = deque()
        self._closed = False
        self._batches = 0
        self._requests = 0
        self._units = 0
        self._last_batch_units = 0
        self._thread = threading.Thread(
            target=self._run,
            name="lsr-embedding-batcher",
            daemon=True,
        )
        self._thread.start()

    def embed(self, texts: list[str]) -> Any:
        if not texts:
            return embed_texts(self.model, texts)
        future: Future[Any] = Future()
        job = _EmbeddingJob(tuple(texts), future)
        with self._condition:
            if self._closed:
                raise RouterError("Embedding batcher is closed.")
            self._jobs.append(job)
            self._condition.notify()
        return future.result()

    def stats(self) -> dict[str, int | float]:
        with self._condition:
            return {
                "maxUnits": self.max_batch_units,
                "maxWaitMs": self.max_wait_seconds * 1000.0,
                "pendingRequests": len(self._jobs),
                "batches": self._batches,
                "requests": self._requests,
                "units": self._units,
                "lastBatchUnits": self._last_batch_units,
            }

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._thread.join()

    def _run(self) -> None:
        while True:
            jobs = self._next_batch()
            if not jobs:
                return
            texts = [text for job in jobs for text in job.texts]
            try:
                embeddings = embed_texts(self.model, texts)
                offset = 0
                for job in jobs:
                    end = offset + len(job.texts)
                    job.future.set_result(embeddings[offset:end])
                    offset = end
            except BaseException as exc:
                for job in jobs:
                    job.future.set_exception(exc)
            finally:
                with self._condition:
                    self._batches += 1
                    self._requests += len(jobs)
                    self._units += len(texts)
                    self._last_batch_units = len(texts)

    def _next_batch(self) -> list[_EmbeddingJob]:
        with self._condition:
            while not self._jobs and not self._closed:
                self._condition.wait()
            if not self._jobs:
                return []

            jobs = [self._jobs.popleft()]
            unit_count = len(jobs[0].texts)
            deadline = time.monotonic() + self.max_wait_seconds
            while unit_count < self.max_batch_units:
                if self._jobs:
                    next_units = len(self._jobs[0].texts)
                    if unit_count + next_units > self.max_batch_units:
                        break
                    job = self._jobs.popleft()
                    jobs.append(job)
                    unit_count += next_units
                    continue
                if self._closed:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return jobs


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


def load_model(
    model_name: str,
    pu_index: int | str | None = None,
    online: bool = False,
) -> Any:
    return EmbeddingService(model_name, pu_index, online).model


def model_tensor_bytes_by_device(model: Any) -> dict[str, int]:
    """Count owned model parameters/buffers by their actual tensor device."""
    model_bytes = getattr(model, "model_bytes", None)
    processing_unit = getattr(model, "processing_unit", None)
    if callable(model_bytes) and processing_unit is not None:
        kind = str(getattr(processing_unit, "kind", "device")).lower()
        return {kind: int(model_bytes())}

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
