from __future__ import annotations

import json
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .config import (
    DEFAULT_CACHE_POLICY,
    DEFAULT_EMBEDDING_BATCH_MAX_UNITS,
    DEFAULT_EMBEDDING_BATCH_WAIT_MS,
    DEFAULT_SERVER_MAX_BATCH_SIZE,
    DEFAULT_SERVER_MAX_PENDING_REQUESTS,
    DEFAULT_SERVER_MAX_REQUEST_BYTES,
    DEFAULT_SIGNATURE_REFRESH_INTERVAL_SECONDS,
)
from .console import console
from .embedding import EmbeddingBatcher, load_model, model_tensor_bytes_by_device
from .errors import RouterError
from .channel_sessions import ChannelSessionDenied, ManagedChannelGuard
from .input_digest import prepare_routing_input
from .memory import current_process_ram_bytes, current_vram_bytes, estimate_object_bytes
from .model_selection import resolve_model_binding, validate_signature_model_compatibility
from .models import LayeredSignature
from .monitor import ServerMonitor
from .profiling import RankProfiler
from .registry import SignatureRegistry
from .routing import LSRRouter, json_ready_route
from .routing_context import agent_event_to_text


class LSRRouteService:
    def __init__(
        self,
        signatures_path: str,
        model_name: str,
        pu_index: int | str | None,
        online: bool,
        default_top_k: int,
        index_policy: str = "auto",
        memory_limit_mb: float = 0.0,
        descend_threshold: float | None = None,
        accept_threshold: float | None = None,
        load_initial_signatures: bool = True,
        monitor_enabled: bool = True,
        model_identity: str | None = None,
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
        self._state_lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._route_ids = itertools.count(1)
        self.signature_refresh_interval_seconds = DEFAULT_SIGNATURE_REFRESH_INTERVAL_SECONDS
        self._last_signature_refresh = 0.0
        self.registry = SignatureRegistry(signatures_path)
        self.signatures: dict[str, LayeredSignature] = self.registry.signatures
        self.model_name = model_identity or model_name
        if signatures_path and load_initial_signatures:
            self.load_signatures(signatures_path, required=True)
        self.model_load_path, self.model_name = resolve_model_binding(
            model_name,
            self.signatures,
            model_identity=model_identity,
        )
        ram_before_model = current_process_ram_bytes()
        self.model = load_model(self.model_load_path, pu_index, online)
        self.embedding_batcher = EmbeddingBatcher(
            self.model,
            max_batch_units=DEFAULT_EMBEDDING_BATCH_MAX_UNITS,
            max_wait_ms=DEFAULT_EMBEDDING_BATCH_WAIT_MS,
        )
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

    def build_router(
        self,
        top_k: int,
        signatures: dict[str, LayeredSignature] | None = None,
        memory_callback: Any = None,
    ) -> LSRRouter:
        return LSRRouter(
            self.signatures if signatures is None else signatures,
            top_k,
            index_policy=self.index_policy,
            model_name=self.model_name,
            memory_limit_mb=self.memory_limit_mb,
            descend_threshold=self.descend_threshold,
            accept_threshold=self.accept_threshold,
            cache_policy=DEFAULT_CACHE_POLICY,
            status_callback=self._status,
            memory_callback=self._memory_event if memory_callback is None else memory_callback,
        )

    def routing_router(
        self, top_k: int, allowed_facets: list[str] | None = None
    ) -> LSRRouter:
        """Build a request-local router from an atomic signature snapshot."""
        allowed = set(allowed_facets) if allowed_facets is not None else None
        with self._state_lock:
            signatures = {
                facet: signature.routing_copy()
                for facet, signature in self.signatures.items()
                if allowed is None or facet in allowed
            }
        if not signatures:
            if allowed is not None:
                raise RouterError("None of the allowed_facets are loaded.")
            raise RouterError("No signatures loaded. Use loadsig first.")
        route_id = next(self._route_ids)

        def memory_event(
            action: str,
            memory: str,
            name: str,
            size: int,
            location: str,
        ) -> None:
            self._memory_event(
                action,
                memory,
                f"route-{route_id}:{name}",
                size,
                location,
            )

        return self.build_router(top_k, signatures, memory_callback=memory_event)

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
        with self._state_lock:
            router = self.router
            signatures = dict(self.signatures)
            signatures_path = self.signatures_path
            loaded_layers = self.registry.loaded_layers()
            loaded_vector_bytes = self.registry.loaded_vector_bytes()
            signature_files = self.registry.signature_files_payload()
            stale_signatures = list(self.registry.stale_signatures)
            disabled_facets = sorted(self.registry.disabled_facets)
        processing_units = (
            self.monitor.processing_units_payload() if self.monitor is not None else []
        )
        selected_processing_unit = next(
            (unit for unit in processing_units if unit.get("selected")),
            None,
        )
        if selected_processing_unit is None:
            unit = getattr(self.model, "processing_unit", None)
            if unit is not None:
                selected_processing_unit = {
                    "index": unit.index,
                    "kind": unit.kind,
                    "name": unit.name,
                    "backend": unit.backend,
                    "luid": unit.luid or None,
                }
        return {
            "status": "ok",
            "service": "LSR",
            "model": self.model_name,
            "requested_model": self.requested_model_name,
            "model_load_path": self.model_load_path,
            "index_policy": self.index_policy,
            "index_backend": router.index_backend_label(),
            "loading_policy": router.loading_policy_label(),
            "memory_limit_mb": self.memory_limit_mb,
            "processRamBytes": current_process_ram_bytes(),
            "processVramBytes": current_vram_bytes(),
            "thresholds": router.thresholds_label(),
            "cachePolicy": router.cache_policy_label(),
            "embeddingBatcher": self.embedding_batcher.stats(),
            "embeddingBackend": str(
                getattr(self.model, "backend", type(self.model).__name__)
            ),
            "processingUnits": processing_units,
            "selectedProcessingUnit": selected_processing_unit,
            "loadedLayers": loaded_layers,
            "loadedVectorBytes": loaded_vector_bytes,
            "memoryComponents": self.monitor.components_payload() if self.monitor is not None else [],
            "memoryStats": self.monitor.memory_stats() if self.monitor is not None else {},
            "lastRankPerformance": self.monitor.last_profile() if self.monitor is not None else {},
            "signatureFiles": signature_files,
            "staleSignatures": stale_signatures,
            "disabledFacets": disabled_facets,
            "signatures": signatures_path,
            "facets": sorted(signatures),
            "facet_metadata": {
                facet: {
                    "modelName": signature.model_name,
                    "modelHash": signature.model_hash.hex(),
                    "dimension": signature.dimension,
                    "normalize": signature.normalize,
                    "distanceMetric": signature.distance_metric,
                }
                for facet, signature in sorted(signatures.items())
            },
            "default_top_k": self.default_top_k,
        }

    def load_signatures(self, path: str, required: bool = True) -> dict[str, Any]:
        with self._state_lock:
            registry = SignatureRegistry(path)
            loaded = registry.load(required=required)
            if getattr(self, "model_name", "auto") != "auto":
                validate_signature_model_compatibility(self.model_name, loaded)
            self.registry = registry
            self.signatures = registry.signatures
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
        with self._state_lock:
            if unload_all:
                removed = sorted(self.signatures)
                for facet in removed:
                    self.registry.unload(facet, disable=True)
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
                    self.registry.unload(key, disable=True)
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
        allowed_facets: list[str] | None = None,
    ) -> dict[str, Any]:
        profiler = RankProfiler()
        profiler.start()
        prompt = prompt.strip()
        if not prompt:
            raise RouterError("Missing prompt.")
        self.refresh_signatures()
        active_top_k = self._top_k(top_k)
        router = self.routing_router(active_top_k, allowed_facets)
        routing_input = prepare_routing_input(prompt, large_input)
        self._status("embedding", f"1 input on {getattr(self.model, 'device', 'selected PU')}")
        embedding_started = time.perf_counter()
        prompt_embedding = self.embedding_batcher.embed(routing_input.routing_units)
        profiler.stage("embedding", embedding_started)
        routing_started = time.perf_counter()
        route = router.route(
            routing_input.routing_text,
            prompt_embedding,
            active_top_k,
            full_depth=full_depth,
        )
        profiler.stage("routing", routing_started)
        attach_input_metadata(route, routing_input)
        response = json_ready_route(route)
        response["candidateFacetFilter"] = candidate_facet_filter(
            allowed_facets, router.signatures
        )
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
        allowed_facets: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        cleaned = [prompt.strip() for prompt in prompts]
        if not cleaned:
            raise RouterError("Missing prompts.")
        if len(cleaned) > DEFAULT_SERVER_MAX_BATCH_SIZE:
            raise RouterError(
                f"Batch exceeds the {DEFAULT_SERVER_MAX_BATCH_SIZE}-item limit."
            )
        self.refresh_signatures()
        for index, prompt in enumerate(cleaned, start=1):
            if not prompt:
                raise RouterError(f"Batch item {index} has no prompt text.")
        active_top_k = self._top_k(top_k)
        router = self.routing_router(active_top_k, allowed_facets)
        routing_inputs = [prepare_routing_input(prompt, large_input) for prompt in cleaned]
        self._status("embedding", f"batch of {len(routing_inputs)} inputs")
        flat_units = [unit for item in routing_inputs for unit in item.routing_units]
        embeddings = self.embedding_batcher.embed(flat_units)
        routes = []
        offset = 0
        for routing_input in routing_inputs:
            unit_count = len(routing_input.routing_units)
            route = router.route(
                routing_input.routing_text,
                embeddings[offset : offset + unit_count],
                active_top_k,
                full_depth=full_depth,
            )
            offset += unit_count
            attach_input_metadata(route, routing_input)
            response = json_ready_route(route)
            response["candidateFacetFilter"] = candidate_facet_filter(
                allowed_facets, router.signatures
            )
            routes.append(response)
        if self.monitor is not None:
            self.monitor.request_finished((time.perf_counter() - started) * 1000.0)
        return routes

    def _top_k(self, top_k: int | None) -> int:
        active_top_k = self.default_top_k if top_k is None else int(top_k)
        if active_top_k < 1:
            raise RouterError("top_k must be at least 1.")
        return active_top_k

    def refresh_signatures(self, force: bool = False) -> None:
        if not self._refresh_lock.acquire(blocking=force):
            return
        try:
            with self._state_lock:
                now = time.monotonic()
                if (
                    not force
                    and now - self._last_signature_refresh
                    < self.signature_refresh_interval_seconds
                ):
                    return
                self._last_signature_refresh = now
                if self.registry.refresh():
                    validate_signature_model_compatibility(
                        self.model_name,
                        self.registry.signatures,
                    )
                    self.signatures = self.registry.signatures
                    self.router = self.build_router(self.default_top_k)
                    self._sync_signature_components()
        finally:
            self._refresh_lock.release()

    def close(self) -> None:
        self.embedding_batcher.close()
        close_model = getattr(self.model, "close", None)
        if callable(close_model):
            close_model()


class LSRHTTPServer(HTTPServer):
    """HTTP server with a bounded worker pool instead of one thread per request."""

    def __init__(self, address: tuple[str, int], service: LSRRouteService, workers: int = 4, *, managed_channel_guard: ManagedChannelGuard | None = None, require_managed_session: bool = False):
        super().__init__(address, LSRRequestHandler)
        self.service = service
        self.managed_channel_guard = managed_channel_guard
        self.require_managed_session = require_managed_session
        active_workers = max(1, workers)
        self.executor = ThreadPoolExecutor(
            max_workers=active_workers,
            thread_name_prefix="lsr-http",
        )
        self.request_slots = threading.BoundedSemaphore(
            active_workers + max(0, DEFAULT_SERVER_MAX_PENDING_REQUESTS)
        )

    def process_request(self, request: Any, client_address: Any) -> None:
        self.request_slots.acquire()
        try:
            self.executor.submit(self._process_request, request, client_address)
        except Exception:
            self.request_slots.release()
            self.shutdown_request(request)
            raise

    def _process_request(self, request: Any, client_address: Any) -> None:
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)
            self.request_slots.release()

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
            if self.server.require_managed_session:
                if self.path != "/route" or self.server.managed_channel_guard is None:
                    raise ChannelSessionDenied("managed channel request is not authorized")
                self.server.managed_channel_guard.authorize(
                    self.headers, operation="lsr.route", method="POST"
                )
            payload = self._read_json()
            if self.path in {"/route", "/rank"}:
                response = self.server.service.route_one(
                    extract_prompt(payload),
                    extract_top_k(payload),
                    extract_full_depth(payload),
                    extract_large_input(payload),
                    extract_allowed_facets(payload),
                )
            elif self.path == "/batch":
                response = self.server.service.route_batch(
                    extract_prompts(payload),
                    extract_top_k(payload),
                    extract_full_depth(payload),
                    extract_large_input(payload),
                    extract_allowed_facets(payload),
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
        except ChannelSessionDenied:
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "managed channel session is not authorized"})
        except RouterError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except json.JSONDecodeError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body."})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> Any:
        raw_length = self.headers.get("Content-Length", "0") or "0"
        try:
            length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise RouterError("Content-Length must be an integer.") from exc
        if length <= 0:
            raise RouterError("Missing JSON body.")
        if length > DEFAULT_SERVER_MAX_REQUEST_BYTES:
            raise RouterError(
                f"JSON body exceeds the {DEFAULT_SERVER_MAX_REQUEST_BYTES}-byte limit."
            )
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise RouterError("JSON body must be UTF-8.") from exc

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
    event = payload.get("agentEvent")
    if event is None:
        event = payload.get("agent_event")
    if event is not None:
        task_context = payload.get("taskContext") or payload.get("task_context")
        return agent_event_to_text(event, task_context)
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
    if len(items) > DEFAULT_SERVER_MAX_BATCH_SIZE:
        raise RouterError(
            f"Batch exceeds the {DEFAULT_SERVER_MAX_BATCH_SIZE}-item limit."
        )
    prompts: list[str] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict):
            try:
                prompts.append(extract_prompt(item))
            except RouterError as exc:
                raise RouterError(f"Batch item {index}: {exc}") from exc
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


def extract_allowed_facets(payload: Any) -> list[str] | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("allowed_facets")
    if value is None:
        value = payload.get("allowedFacets")
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise RouterError("allowed_facets must be a non-empty array of facet IDs.")
    if len(value) > 1024:
        raise RouterError("allowed_facets exceeds the 1024-item limit.")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise RouterError("Every allowed_facets item must be a string.")
        facet = item.strip().lower()
        if not facet or len(facet) > 256:
            raise RouterError("Every allowed_facets item must contain 1-256 characters.")
        if facet not in seen:
            result.append(facet)
            seen.add(facet)
    return result


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


def candidate_facet_filter(
    requested: list[str] | None, effective_signatures: dict[str, Any]
) -> dict[str, Any]:
    effective = sorted(str(facet) for facet in effective_signatures)
    if requested is None:
        return {
            "applied": False,
            "requested": None,
            "effective": effective,
            "missing": [],
        }
    requested_set = set(requested)
    return {
        "applied": True,
        "requested": list(requested),
        "effective": effective,
        "missing": sorted(requested_set.difference(effective)),
    }


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
    pu_index: int | str | None,
    online: bool,
    default_top_k: int,
    index_policy: str = "auto",
    memory_limit_mb: float = 0.0,
    descend_threshold: float | None = None,
    accept_threshold: float | None = None,
    load_initial_signatures: bool = True,
    monitor_enabled: bool = True,
    workers: int = 4,
    model_identity: str | None = None,
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
        model_identity=model_identity,
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
        service.close()
