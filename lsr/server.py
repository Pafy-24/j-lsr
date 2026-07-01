from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .console import console
from .embedding import embed_texts, load_model
from .errors import RouterError
from .model_selection import resolve_model_name, validate_signature_model_compatibility
from .models import LayeredSignature
from .routing import LSRRouter, json_ready_route
from .signatures import SignatureStore


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
        descend_threshold: float = 0.18,
        accept_threshold: float = 0.82,
        load_initial_signatures: bool = True,
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
        self.signatures: dict[str, LayeredSignature] = {}
        self.model_name = model_name
        if signatures_path and load_initial_signatures:
            self.load_signatures(signatures_path, required=True)
        self.model_name = resolve_model_name(model_name, self.signatures)
        validate_signature_model_compatibility(self.model_name, self.signatures)
        self.model = load_model(self.model_name, pu_index, online)
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
            "thresholds": {
                "descend": self.descend_threshold,
                "accept": self.accept_threshold,
            },
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
        loaded = SignatureStore(path).load_all(required=required)
        if getattr(self, "model_name", "auto") != "auto":
            validate_signature_model_compatibility(self.model_name, loaded)
        self.signatures.update(loaded)
        self.signatures_path = path
        if hasattr(self, "router"):
            self.router = self.build_router(self.default_top_k)
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
            self.router = self.build_router(self.default_top_k)
            return {"unloaded": removed, "facets": []}
        if not facets:
            raise RouterError("Provide facets or all=true.")

        removed: list[str] = []
        missing: list[str] = []
        for facet in facets:
            key = facet.strip().lower()
            if key in self.signatures:
                del self.signatures[key]
                removed.append(key)
            else:
                missing.append(key)
        self.router = self.build_router(self.default_top_k)
        return {
            "unloaded": removed,
            "missing": missing,
            "facets": sorted(self.signatures),
        }

    def route_one(self, prompt: str, top_k: int | None = None) -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise RouterError("Missing prompt.")
        if not self.signatures:
            raise RouterError("No signatures loaded. Use loadsig first.")
        active_top_k = self._top_k(top_k)
        prompt_embedding = embed_texts(self.model, [prompt])[0]
        route = self.router.route(prompt, prompt_embedding, active_top_k)
        return json_ready_route(route)

    def route_batch(self, prompts: list[str], top_k: int | None = None) -> list[dict[str, Any]]:
        cleaned = [prompt.strip() for prompt in prompts]
        if not cleaned:
            raise RouterError("Missing prompts.")
        if not self.signatures:
            raise RouterError("No signatures loaded. Use loadsig first.")
        for index, prompt in enumerate(cleaned, start=1):
            if not prompt:
                raise RouterError(f"Batch item {index} has no prompt text.")
        active_top_k = self._top_k(top_k)
        embeddings = embed_texts(self.model, cleaned)
        return [
            json_ready_route(self.router.route(prompt, embeddings[index], active_top_k))
            for index, prompt in enumerate(cleaned)
        ]

    def _top_k(self, top_k: int | None) -> int:
        active_top_k = self.default_top_k if top_k is None else int(top_k)
        if active_top_k < 1:
            raise RouterError("top_k must be at least 1.")
        return active_top_k


class LSRHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], service: LSRRouteService):
        super().__init__(address, LSRRequestHandler)
        self.service = service


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
            if self.path in {"/route", "/rank"}:
                response = self.server.service.route_one(
                    extract_prompt(payload), extract_top_k(payload)
                )
            elif self.path == "/batch":
                response = self.server.service.route_batch(
                    extract_prompts(payload), extract_top_k(payload)
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
    descend_threshold: float = 0.18,
    accept_threshold: float = 0.82,
    load_initial_signatures: bool = True,
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
    )
    server = LSRHTTPServer((host, port), service)
    console.print(
        f"LSR server ready on http://{host}:{port} "
        f"with facets: {', '.join(service.health()['facets']) or 'none'}"
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
