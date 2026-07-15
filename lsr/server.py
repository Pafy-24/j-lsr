from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .console import console
from .embedding import embed_texts, load_model, model_tensor_bytes_by_device
from .errors import RouterError
from .input_digest import prepare_routing_input
from .memory import current_process_ram_bytes, current_vram_bytes, estimate_object_bytes
from .model_selection import resolve_model_name, validate_signature_model_compatibility
from .models import LayeredSignature
from .monitor import ServerMonitor
from .profiling import RankProfiler
from .registry import SignatureRegistry
from .routing import LSRRouter, json_ready_route


class LSRRouteService:
    def __init__(
        self,
        signatures_path: str,
        model_name: str,
        pu_index: int | None,
        online: bool,
        default_top_k: int,
        index_policy: str = "auto",
        memory_limit_mb: float = 0.0,
        descend_threshold: float | None = None,
        accept_threshold: float | None = None,
        load_initial_signatures: bool = True,
        monitor_enabled: bool = True,
    ):
        if default_top_k < 1:
            raise RouterError("--top-k must be at least 1.")
        self.signatures_path = signatures_path
        self.requested_model_name = model_name
        self.default_top_k = default_top_k
        self.index_policy = index_policy
        self.memory_limit_mb = memory_limit_mb
        self.descend_threshold = descend_threshold
        self.accept_threshold = accept_threshold
        self.request_lock = threading.RLock()
        self.registry = SignatureRegistry(signatures_path)
        self.signatures: dict[str, LayeredSignature] = self.registry.signatures
        self.model_name = model_name
        if signatures_path and load_initial_signatures:
            self.load_signatures(signatures_path, required=True)
        self.model_name = resolve_model_name(model_name, self.signatures)
        validate_signature_model_compatibility(self.model_name, self.signatures)
        ram_before_model = current_process_ram_bytes()
        self.model = load_model(self.model_name, pu_index, online)
        model_tensor_bytes = model_tensor_bytes_by_device(self.model)
        ram_after_model = current_process_ram_bytes()
        self.monitor: ServerMonitor | None = None
        if monitor_enabled:
            self.monitor = ServerMonitor(self)
            cpu_parameter_bytes = sum(
                size for device, size in model_tensor_bytes.items() if device.startswith("cpu")
            )
            if cpu_parameter_bytes:
                self.monitor.set_component(
                    "RAM",
                    f"model-parameters:{self.model_name}",
                    cpu_parameter_bytes,
                    "CPU tensors",
                    persistent=True,
                )
            self.monitor.set_component(
                "RAM",
                "embedding-framework/tokenizer/driver",
                max(0, ram_after_model - ram_before_model - cpu_parameter_bytes),
                "native/Python working set delta",
                persistent=True,
            )
            self._sync_signature_components()
            cuda_parameter_bytes = sum(
                size for device, size in model_tensor_bytes.items() if device.startswith("cuda")
            )
            if cuda_parameter_bytes:
                self.monitor.set_component(
                    "VRAM",
                    f"model-parameters:{self.model_name}",
                    cuda_parameter_bytes,
                    ", ".join(device for device in model_tensor_bytes if device.startswith("cuda")),
                    persistent=True,
                )
        self.router = self.build_router(default_top_k)

    def build_router(self, top_k: int) -> LSRRouter:
        return LSRRouter(
            self.signatures,
            top_k,
            index_policy=self.index_policy,
            model_name=self.model_name,
            memory_limit_mb=self.memory_limit_mb,
            descend_threshold=self.descend_threshold,
            accept_threshold=self.accept_threshold,
            cache_policy="release_after_layer",
            status_callback=self._status,
            memory_callback=self._memory_event,
        )

    def _status(self, operation: str, detail: str = "") -> None:
        if self.monitor is not None:
            self.monitor.update(operation, detail)

    def _memory_event(
        self, action: str, memory: str, name: str, size: int, location: str
    ) -> None:
        if self.monitor is None:
            return
        if action == "set":
            self.monitor.set_component(memory, name, size, location)
        else:
            self.monitor.remove_component(memory, name)

    def _sync_signature_components(self) -> None:
        if getattr(self, "monitor", None) is None:
            return
        self.monitor.remove_components("RAM", "signature-metadata:")
        for facet, signature in sorted(self.signatures.items()):
            self.monitor.set_component(
                "RAM",
                f"signature-metadata:{facet}",
                estimate_object_bytes(signature),
                "DER metadata/index",
                persistent=True,
            )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "LSR",
            "model": self.model_name,
            "requested_model": self.requested_model_name,
            "index_policy": self.index_policy,
            "index_backend": self.router.index_backend_label(),
            "loading_policy": self.router.loading_policy_label(),
            "memory_limit_mb": self.memory_limit_mb,
            "processRamBytes": current_process_ram_bytes(),
            "processVramBytes": current_vram_bytes(),
            "thresholds": self.router.thresholds_label(),
            "cachePolicy": self.router.cache_policy_label(),
            "loadedLayers": self.registry.loaded_layers(),
            "loadedVectorBytes": self.registry.loaded_vector_bytes(),
            "memoryComponents": self.monitor.components_payload() if self.monitor is not None else [],
            "memoryStats": self.monitor.memory_stats() if self.monitor is not None else {},
            "lastRankPerformance": self.monitor.last_profile() if self.monitor is not None else {},
            "signatureFiles": self.registry.signature_files_payload(),
            "staleSignatures": self.registry.stale_signatures,
            "signatures": self.signatures_path,
            "facets": sorted(self.signatures),
            "facet_metadata": {
                facet: {
                    "modelName": signature.model_name,
                    "modelHash": signature.model_hash.hex(),
                    "dimension": signature.dimension,
                    "normalize": signature.normalize,
                    "distanceMetric": signature.distance_metric,
                }
                for facet, signature in sorted(self.signatures.items())
            },
            "default_top_k": self.default_top_k,
        }

    def load_signatures(self, path: str, required: bool = True) -> dict[str, Any]:
        self.registry = SignatureRegistry(path)
        loaded = self.registry.load(required=required)
        if getattr(self, "model_name", "auto") != "auto":
            validate_signature_model_compatibility(self.model_name, loaded)
        self.signatures = self.registry.signatures
        self.signatures_path = path
        if hasattr(self, "router"):
            self.router = self.build_router(self.default_top_k)
        self._sync_signature_components()
        return {
            "loaded": sorted(loaded),
            "facets": sorted(self.signatures),
        }

    def unload_signatures(
        self, facets: list[str] | None = None, unload_all: bool = False
    ) -> dict[str, Any]:
        if unload_all:
            removed = sorted(self.signatures)
            self.signatures.clear()
            self.registry.signatures.clear()
            self.registry.files.clear()
            self.registry.file_stats.clear()
            self.router = self.build_router(self.default_top_k)
            self._sync_signature_components()
            return {"unloaded": removed, "facets": []}
        if not facets:
            raise RouterError("Provide facets or all=true.")

        removed: list[str] = []
        missing: list[str] = []
        for facet in facets:
            key = facet.strip().lower()
            if key in self.signatures:
                self.registry.unload(key)
                removed.append(key)
            else:
                missing.append(key)
        self.router = self.build_router(self.default_top_k)
        self._sync_signature_components()
        return {
            "unloaded": removed,
            "missing": missing,
            "facets": sorted(self.signatures),
        }

    def route_one(
        self,
        prompt: str,
        top_k: int | None = None,
        full_depth: bool = False,
        large_input: str = "auto",
    ) -> dict[str, Any]:
        profiler = RankProfiler()
        profiler.start()
        prompt = prompt.strip()
        if not prompt:
            raise RouterError("Missing prompt.")
        self.refresh_signatures()
        if not self.signatures:
            raise RouterError("No signatures loaded. Use loadsig first.")
        active_top_k = self._top_k(top_k)
        routing_input = prepare_routing_input(prompt, large_input)
        self._status("embedding", f"1 input on {getattr(self.model, 'device', 'selected PU')}")
        embedding_started = time.perf_counter()
        prompt_embedding = embed_texts(self.model, routing_input.routing_units)
        profiler.stage("embedding", embedding_started)
        routing_started = time.perf_counter()
        route = self.router.route(routing_input.routing_text, prompt_embedding, active_top_k, full_depth=full_depth)
        profiler.stage("routing", routing_started)
        attach_input_metadata(route, routing_input)
        response = json_ready_route(route)
        response["performance"] = profiler.finish()
        if self.monitor is not None:
            self.monitor.request_finished(
                float(response["performance"]["timing"]["rankingMs"]),
                response["performance"],
            )
        return response

    def route_batch(
        self,
        prompts: list[str],
        top_k: int | None = None,
        full_depth: bool = False,
        large_input: str = "auto",
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        cleaned = [prompt.strip() for prompt in prompts]
        if not cleaned:
            raise RouterError("Missing prompts.")
        self.refresh_signatures()
        if not self.signatures:
            raise RouterError("No signatures loaded. Use loadsig first.")
        for index, prompt in enumerate(cleaned, start=1):
            if not prompt:
                raise RouterError(f"Batch item {index} has no prompt text.")
        active_top_k = self._top_k(top_k)
        routing_inputs = [prepare_routing_input(prompt, large_input) for prompt in cleaned]
        self._status("embedding", f"batch of {len(routing_inputs)} inputs")
        flat_units = [unit for item in routing_inputs for unit in item.routing_units]
        embeddings = embed_texts(self.model, flat_units)
        routes = []
        offset = 0
        for routing_input in routing_inputs:
            unit_count = len(routing_input.routing_units)
            route = self.router.route(
                routing_input.routing_text,
                embeddings[offset : offset + unit_count],
                active_top_k,
                full_depth=full_depth,
            )
            offset += unit_count
            attach_input_metadata(route, routing_input)
            routes.append(json_ready_route(route))
        if self.monitor is not None:
            self.monitor.request_finished((time.perf_counter() - started) * 1000.0)
        return routes

    def _top_k(self, top_k: int | None) -> int:
        active_top_k = self.default_top_k if top_k is None else int(top_k)
        if active_top_k < 1:
            raise RouterError("top_k must be at least 1.")
        return active_top_k

    def refresh_signatures(self) -> None:
        if self.registry.refresh():
            validate_signature_model_compatibility(self.model_name, self.registry.signatures)
            self.signatures = self.registry.signatures
            self.router = self.build_router(self.default_top_k)
            self._sync_signature_components()


class LSRHTTPServer(HTTPServer):
    """HTTP server with a bounded worker pool instead of one thread per request."""

    def __init__(self, address: tuple[str, int], service: LSRRouteService, workers: int = 4):
        super().__init__(address, LSRRequestHandler)
        self.service = service
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="lsr-http")

    def process_request(self, request: Any, client_address: Any) -> None:
        self.executor.submit(self._process_request, request, client_address)

    def _process_request(self, request: Any, client_address: Any) -> None:
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def server_close(self) -> None:
        super().server_close()
        self.executor.shutdown(wait=True, cancel_futures=True)


class LSRRequestHandler(BaseHTTPRequestHandler):
    server: LSRHTTPServer

    def do_GET(self) -> None:
        if self.path in {"/", "/health"}:
            self._write_json(HTTPStatus.OK, self.server.service.health())
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            with self.server.service.request_lock:
                if self.path in {"/route", "/rank"}:
                    response = self.server.service.route_one(
                        extract_prompt(payload),
                        extract_top_k(payload),
                        extract_full_depth(payload),
                        extract_large_input(payload),
                    )
                elif self.path == "/batch":
                    response = self.server.service.route_batch(
                        extract_prompts(payload),
                        extract_top_k(payload),
                        extract_full_depth(payload),
                        extract_large_input(payload),
                    )
                elif self.path == "/loadsig":
                    response = self.server.service.load_signatures(extract_signature_path(payload))
                elif self.path == "/unloadsig":
                    facets, unload_all = extract_unload_request(payload)
                    response = self.server.service.unload_signatures(facets, unload_all)
                else:
                    self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                    return
            self._write_json(HTTPStatus.OK, response)
        except RouterError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except json.JSONDecodeError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body."})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise RouterError("Missing JSON body.")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def extract_prompt(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        raise RouterError("Request body must be an object or prompt string.")
    prompt = payload.get("input") or payload.get("prompt") or payload.get("text")
    if not isinstance(prompt, str):
        raise RouterError("Request body must include input, prompt, or text.")
    return prompt


def extract_prompts(payload: Any) -> list[str]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("inputs") or payload.get("prompts") or payload.get("texts")
    else:
        items = None
    if not isinstance(items, list):
        raise RouterError("Batch body must be a list or include inputs/prompts/texts.")
    prompts: list[str] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict):
            prompt = item.get("input") or item.get("prompt") or item.get("text")
            if not isinstance(prompt, str):
                raise RouterError(f"Batch item {index} has no prompt text.")
            prompts.append(prompt)
        else:
            raise RouterError(f"Batch item {index} must be a string or object.")
    return prompts


def extract_top_k(payload: Any) -> int | None:
    if not isinstance(payload, dict) or payload.get("top_k") is None:
        return None
    try:
        return int(payload["top_k"])
    except (TypeError, ValueError) as exc:
        raise RouterError("top_k must be an integer.") from exc


def extract_full_depth(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("full_depth") or payload.get("fullDepth"))


def extract_large_input(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "auto"
    mode = payload.get("large_input") or payload.get("largeInput") or "auto"
    if mode not in {"auto", "off", "force"}:
        raise RouterError("large_input must be auto, off, or force.")
    return str(mode)


def attach_input_metadata(route: dict[str, Any], routing_input: Any) -> None:
    route["inputMode"] = routing_input.input_mode
    route["inputChars"] = routing_input.input_chars
    route["routingTextChars"] = routing_input.routing_text_chars
    route["chunksUsed"] = routing_input.chunks_used
    route["routingUnits"] = len(routing_input.routing_units)


def extract_signature_path(payload: Any) -> str:
    if isinstance(payload, str):
        path = payload
    elif isinstance(payload, dict):
        path = payload.get("signatures") or payload.get("path")
    else:
        path = None
    if not isinstance(path, str) or not path.strip():
        raise RouterError("Request body must include signatures or path.")
    return path.strip()


def extract_unload_request(payload: Any) -> tuple[list[str] | None, bool]:
    if isinstance(payload, str):
        return [payload], False
    if not isinstance(payload, dict):
        raise RouterError("Unload body must be an object, facet string, or all=true.")
    unload_all = bool(payload.get("all"))
    raw_facets = payload.get("facets")
    if raw_facets is None and payload.get("facet") is not None:
        raw_facets = [payload["facet"]]
    if raw_facets is None:
        return None, unload_all
    if not isinstance(raw_facets, list) or not all(isinstance(item, str) for item in raw_facets):
        raise RouterError("facets must be a list of strings.")
    return raw_facets, unload_all


def serve(
    host: str,
    port: int,
    signatures_path: str,
    model_name: str,
    pu_index: int | None,
    online: bool,
    default_top_k: int,
    index_policy: str = "auto",
    memory_limit_mb: float = 0.0,
    descend_threshold: float | None = None,
    accept_threshold: float | None = None,
    load_initial_signatures: bool = True,
    monitor_enabled: bool = True,
    workers: int = 4,
) -> None:
    service = LSRRouteService(
        signatures_path=signatures_path,
        model_name=model_name,
        pu_index=pu_index,
        online=online,
        default_top_k=default_top_k,
        index_policy=index_policy,
        memory_limit_mb=memory_limit_mb,
        descend_threshold=descend_threshold,
        accept_threshold=accept_threshold,
        load_initial_signatures=load_initial_signatures,
        monitor_enabled=monitor_enabled,
    )
    server = LSRHTTPServer((host, port), service, workers=workers)
    console.print(
        f"LSR server ready on http://{host}:{port} "
        f"with facets: {', '.join(service.health()['facets']) or 'none'}"
    )
    if service.monitor is not None:
        service.monitor.start()
    try:
        server.serve_forever()
    finally:
        if service.monitor is not None:
            service.monitor.stop()
        server.server_close()
