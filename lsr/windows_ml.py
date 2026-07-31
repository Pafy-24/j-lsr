from __future__ import annotations

import asyncio
import atexit
import gc
import hashlib
import os
import site
import sys
import threading
from pathlib import Path
from typing import Any

from .errors import RouterError
from .models import ProcessingUnit


VITIS_PROVIDER = "VitisAIExecutionProvider"
DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
NPU_MODEL_FILENAME = "onnx/model_npu_a16w8.onnx"
DML_MODEL_FILENAME = "onnx/model.onnx"
DEFAULT_MAX_SEQUENCE_LENGTH = 256

_RUNTIME_LOCK = threading.Lock()
_APP_SDK_CONTEXT: Any = None
_VITIS_REGISTERED = False
_VITIS_ERROR = ""


def _shutdown_windows_app_sdk() -> None:
    global _APP_SDK_CONTEXT
    if _APP_SDK_CONTEXT is not None:
        try:
            _APP_SDK_CONTEXT()
        except Exception:
            pass
        _APP_SDK_CONTEXT = None


def _add_winui3_to_path() -> None:
    for package_root in site.getsitepackages():
        winui3_path = Path(package_root) / "winui3"
        if winui3_path.is_dir() and str(winui3_path) not in sys.path:
            sys.path.insert(0, str(winui3_path))
            return


def _await_winrt(operation: Any) -> Any:
    async def wait_for_operation() -> Any:
        return await operation

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(wait_for_operation())

    result: list[Any] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(wait_for_operation()))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=runner, name="lsr-winrt-await")
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def register_windows_ml_providers(include_npu: bool = True) -> tuple[Any, str]:
    """Initialize Windows ML and register AMD's Vitis AI provider for this process."""
    global _APP_SDK_CONTEXT, _VITIS_ERROR, _VITIS_REGISTERED

    if os.name != "nt":
        raise RouterError("Windows ML is available only on Windows.")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RouterError(
            "Windows ML requires onnxruntime-windowsml in the active Python environment."
        ) from exc

    if not include_npu:
        return ort, ""

    with _RUNTIME_LOCK:
        if _VITIS_REGISTERED:
            return ort, ""
        if VITIS_PROVIDER in ort.get_available_providers():
            _VITIS_REGISTERED = True
            return ort, ""
        if _VITIS_ERROR:
            return ort, _VITIS_ERROR
        try:
            _add_winui3_to_path()
            from microsoft.windows.ai.machinelearning import (
                ExecutionProviderCatalog,
                ExecutionProviderReadyResultState,
            )
            from microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
                initialize,
            )

            if _APP_SDK_CONTEXT is None:
                _APP_SDK_CONTEXT = initialize()
                atexit.register(_shutdown_windows_app_sdk)

            provider = next(
                (
                    item
                    for item in ExecutionProviderCatalog.get_default().find_all_providers()
                    if item.name == VITIS_PROVIDER
                ),
                None,
            )
            if provider is None:
                raise RuntimeError(
                    "the AMD Vitis AI Windows ML execution-provider package is not installed"
                )
            ready_result = _await_winrt(provider.ensure_ready_async())
            if ready_result.status == ExecutionProviderReadyResultState.FAILURE:
                diagnostic = str(getattr(ready_result, "diagnostic_text", "")).strip()
                raise RuntimeError(diagnostic or "Vitis AI provider could not be prepared")
            ort.register_execution_provider_library(provider.name, provider.library_path)
            _VITIS_REGISTERED = VITIS_PROVIDER in ort.get_available_providers()
            if not _VITIS_REGISTERED:
                raise RuntimeError("Vitis AI provider registration did not become visible")
        except Exception as exc:
            _VITIS_ERROR = str(exc)
            return ort, _VITIS_ERROR
    return ort, ""


def _device_payload(ep_device: Any) -> dict[str, Any]:
    hardware = ep_device.device
    metadata = {str(key): str(value).strip() for key, value in dict(hardware.metadata).items()}
    ep_metadata = {
        str(key): str(value).strip()
        for key, value in dict(getattr(ep_device, "ep_metadata", {})).items()
    }
    if ep_metadata.get("version"):
        metadata["EpVersion"] = ep_metadata["version"]
    description = metadata.get("Description") or f"{hardware.vendor} {hardware.type}"
    luid_decimal = metadata.get("LUID", "")
    luid = f"{int(luid_decimal):X}" if luid_decimal.isdigit() else ""
    return {
        "provider": str(ep_device.ep_name),
        "name": description.strip(),
        "vendor": str(hardware.vendor).strip(),
        "type": str(hardware.type).rsplit(".", 1)[-1],
        "luid": luid,
        "metadata": metadata,
    }


def windows_ml_devices(include_npu: bool = True) -> tuple[list[dict[str, Any]], str]:
    if os.name != "nt":
        return [], ""
    try:
        ort, npu_error = register_windows_ml_providers(include_npu=include_npu)
        devices = [_device_payload(device) for device in ort.get_ep_devices()]
        return devices, npu_error
    except Exception as exc:
        return [], str(exc)


def _matching_ep_device(ort: Any, unit: ProcessingUnit) -> Any:
    for ep_device in ort.get_ep_devices():
        payload = _device_payload(ep_device)
        if payload["provider"] != unit.provider:
            continue
        if unit.luid and payload["luid"] != unit.luid:
            continue
        if not unit.luid and payload["name"] != unit.name:
            continue
        return ep_device
    raise RouterError(
        f"Windows ML device '{unit.name}' ({unit.provider}, LUID {unit.luid or '-'}) disappeared."
    )


def resolve_model_root(model_name: str, online: bool) -> Path:
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return local_path if local_path.is_dir() else local_path.parent
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=model_name,
                local_files_only=not online,
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.safetensors",
                    DML_MODEL_FILENAME,
                    NPU_MODEL_FILENAME,
                ],
            )
        )
    except Exception as exc:
        mode = "download" if online else "local cache"
        raise RouterError(
            f"Could not resolve {model_name} from the Hugging Face {mode}: {exc}"
        ) from exc


def _model_path(model_root: Path, unit: ProcessingUnit) -> Path:
    filename = NPU_MODEL_FILENAME if unit.provider == VITIS_PROVIDER else DML_MODEL_FILENAME
    path = model_root / Path(filename)
    if path.exists():
        return path
    if unit.provider == VITIS_PROVIDER:
        raise RouterError(
            f"NPU model not found: {path}. Prepare it once from a Visual Studio "
            "Developer PowerShell with: python scripts/prepare_windows_ml_model.py"
        )
    raise RouterError(
        f"ONNX model not found: {path}. Run with --online once to cache the ONNX export."
    )


class WindowsMLEmbeddingModel:
    """SentenceTransformer-compatible encode facade backed by Windows ML/ORT."""

    def __init__(self, model_name: str, unit: ProcessingUnit, online: bool = False):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.processing_unit = unit
        self.backend = unit.backend
        self.device = f"{unit.provider}:{unit.name}"
        self.model_root = resolve_model_root(model_name, online)
        self.model_path = _model_path(self.model_root, unit)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_root,
            local_files_only=not online,
        )
        self.session = self._create_session(unit)
        self._atexit_close = self.close
        atexit.register(self._atexit_close)
        first_input_shape = self.session.get_inputs()[0].shape
        self.fixed_batch_size = (
            int(first_input_shape[0])
            if first_input_shape and isinstance(first_input_shape[0], int)
            else None
        )
        self.fixed_sequence_length = (
            int(first_input_shape[1])
            if len(first_input_shape) > 1 and isinstance(first_input_shape[1], int)
            else None
        )
        if unit.provider == VITIS_PROVIDER and self.fixed_batch_size != 1:
            self.close()
            raise RouterError(
                "The AMD NPU model must have a static batch size of 1. "
                "Rebuild it with scripts/prepare_windows_ml_model.py --force."
            )

    def _create_session(self, unit: ProcessingUnit) -> Any:
        ort, npu_error = register_windows_ml_providers(
            include_npu=unit.provider == VITIS_PROVIDER
        )
        if unit.provider == VITIS_PROVIDER and npu_error:
            raise RouterError(f"AMD NPU provider is unavailable: {npu_error}")
        options = ort.SessionOptions()
        provider_options: dict[str, str] = {}
        if unit.provider == VITIS_PROVIDER:
            provider_options = self._vitis_provider_options(unit)
        options.add_provider_for_devices(
            [_matching_ep_device(ort, unit)],
            provider_options,
        )
        try:
            return ort.InferenceSession(str(self.model_path), sess_options=options)
        except Exception as exc:
            target = "AMD NPU" if unit.provider == VITIS_PROVIDER else "AMD iGPU"
            raise RouterError(
                f"{target} could not compile {self.model_path.name}: {exc}"
            ) from exc

    def _vitis_provider_options(self, unit: ProcessingUnit) -> dict[str, str]:
        cache_dir = self.model_root / ".lsr_vitis_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        ep_version = "".join(
            character if character.isalnum() else "_"
            for character in unit.metadata.get("EpVersion", "unknown")
        )
        return {
            "cache_dir": str(cache_dir),
            "cache_key": f"minilm_{digest.hexdigest()[:16]}_ep_{ep_version}",
            "enable_cache_file_io_in_mem": "0",
            "log_level": "error",
        }

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> Any:
        import numpy as np

        del show_progress_bar, convert_to_numpy
        batches: list[Any] = []
        effective_batch_size = max(1, int(batch_size))
        if self.fixed_batch_size:
            effective_batch_size = min(effective_batch_size, self.fixed_batch_size)
        for start in range(0, len(texts), effective_batch_size):
            batch = texts[start : start + effective_batch_size]
            padding = "max_length" if self.fixed_sequence_length else True
            encoded = self.tokenizer(
                batch,
                return_tensors="np",
                padding=padding,
                truncation=True,
                max_length=self.fixed_sequence_length or DEFAULT_MAX_SEQUENCE_LENGTH,
            )
            feeds = {
                item.name: np.asarray(encoded[item.name], dtype=np.int64)
                for item in self.session.get_inputs()
            }
            token_embeddings = np.asarray(self.session.run(None, feeds)[0], dtype=np.float32)
            attention_mask = np.asarray(encoded["attention_mask"], dtype=np.float32)
            mask = attention_mask[:, :, None]
            pooled = (token_embeddings * mask).sum(axis=1)
            pooled /= np.clip(mask.sum(axis=1), 1e-9, None)
            if normalize_embeddings:
                pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
            batches.append(np.asarray(pooled, dtype=np.float32))
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(batches, axis=0)

    def model_bytes(self) -> int:
        total = self.model_path.stat().st_size
        external_data = self.model_path.with_name(f"{self.model_path.name}.data")
        if external_data.exists():
            total += external_data.stat().st_size
        return total

    def close(self) -> None:
        if getattr(self, "session", None) is not None:
            self.session = None
            gc.collect()
        callback = getattr(self, "_atexit_close", None)
        if callback is not None:
            atexit.unregister(callback)
            self._atexit_close = None


def load_windows_ml_model(
    model_name: str,
    unit: ProcessingUnit,
    online: bool = False,
) -> WindowsMLEmbeddingModel:
    return WindowsMLEmbeddingModel(model_name, unit, online)
