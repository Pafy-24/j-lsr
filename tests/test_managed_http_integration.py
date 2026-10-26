from __future__ import annotations

import os
import pytest

if os.environ.get("JESTER_CROSS_REPO_TESTS") != "1":
    pytest.skip("cross-repo RM/component integration; set JESTER_CROSS_REPO_TESTS=1", allow_module_level=True)

import json
import os
import socket
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import jester_rm
from jester_rm.config import Settings
from jester_rm.controller.service import ControllerService
from jester_rm.models import Deployment, EntityMetadata, HostSpec, Lease

from lsr.channel_sessions import ManagedChannelContext, ManagedChannelGuard
from lsr.server import (
    DENIED_REQUEST_BODY_DISCARD_MAX_BYTES,
    LSRHTTPServer,
    LSRRequestHandler,
)


def _assert_canonical_rm_origin() -> None:
    # This is source integration coverage, not native deployment qualification.
    module_file = Path(jester_rm.__file__).resolve()
    engine_root = Path(__file__).resolve().parents[4]
    expected = engine_root / "jester-rm/src/jester_rm/__init__.py"
    assert module_file == expected.resolve(), module_file


_assert_canonical_rm_origin()


class _RouteService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def health(self):
        return {"status": "ok"}

    def route_one(self, prompt, *_args):
        self.calls.append(prompt)
        return {"accepted": prompt}


def _post(port: int, credential: str, body: bytes = b'{"input":"deterministic"}') -> tuple[int, dict]:
    request = Request(f"http://127.0.0.1:{port}/route", data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("X-Jester-Component-Session", credential)
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def _raw_denied_post(
    port: int,
    *,
    content_length: str | None,
    body: bytes = b"",
    transfer_encoding: str | None = None,
    additional_content_length: str | None = None,
) -> tuple[int, float]:
    headers = [
        "POST /route HTTP/1.1",
        f"Host: 127.0.0.1:{port}",
        "X-Jester-Component-Session: invalid-session",
        "Content-Type: application/json",
        "Connection: close",
    ]
    if content_length is not None:
        headers.append(f"Content-Length: {content_length}")
    if additional_content_length is not None:
        headers.append(f"Content-Length: {additional_content_length}")
    if transfer_encoding is not None:
        headers.append(f"Transfer-Encoding: {transfer_encoding}")
    request = ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + body

    started = time.monotonic()
    with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
        client.settimeout(2)
        client.sendall(request)
        response = bytearray()
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
    elapsed = time.monotonic() - started
    status_line = bytes(response).split(b"\r\n", 1)[0]
    return int(status_line.split(b" ", 2)[1]), elapsed


def _slow_trickle_denied_post(
    port: int,
    discard_started: threading.Event,
) -> tuple[int, float]:
    request = (
        "\r\n".join(
            (
                "POST /route HTTP/1.1",
                f"Host: 127.0.0.1:{port}",
                "X-Jester-Component-Session: invalid-session",
                "Content-Type: application/json",
                "Content-Length: 4",
                "Connection: close",
            )
        )
        + "\r\n\r\n"
    ).encode("ascii") + b"{"

    started = time.monotonic()
    with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
        client.settimeout(2)
        client.sendall(request)

        def trickle() -> None:
            if not discard_started.wait(timeout=1):
                return
            for byte in (b"x", b"y", b"z"):
                time.sleep(0.18)
                try:
                    client.sendall(byte)
                except OSError:
                    return

        sender = threading.Thread(target=trickle, daemon=True)
        sender.start()
        response = bytearray()
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
        response_finished = time.monotonic()
        sender.join(timeout=1)
    elapsed = response_finished - started
    status_line = bytes(response).split(b"\r\n", 1)[0]
    return int(status_line.split(b" ", 2)[1]), elapsed


def test_rm_issued_session_controls_actual_lsr_route_handler(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JESTER_RM_HOME", str(tmp_path / "rm-home"))
    monkeypatch.setenv("JESTER_RM_AGENT_HOME", str(tmp_path / "rm-agent"))
    controller = ControllerService(Settings.load(create=True))
    server = None
    thread = None
    try:
        host = HostSpec(metadata=EntityMetadata(id="host-one"), transport="local", status="online")
        controller.put_host(host)
        from jester_rm.models import ComponentSpec, InstanceBody, InstanceSpec, ResourceRequirements
        def instance(identifier: str, adapter: str, port: int) -> InstanceSpec:
            return InstanceSpec(metadata=EntityMetadata(id=identifier), spec=InstanceBody(
                hostId="host-one", component=ComponentSpec(adapter=adapter), resources=ResourceRequirements.model_validate({"cpu":{},"memory":{"ramBytes":1},"storage":{"poolId":"pool-one","quotaBytes":1}}), network={"port": port},
            ))
        for item in (instance("engine-one", "engine.serve", 8780), instance("lsr-one", "lsr.serve", 9123)):
            controller.store.put_entity("instance", item.metadata.id, item.api_dict(), expected_generation=0)
        deployment = Deployment.model_validate({"metadata":{"id":"deployment-one"}, "spec":{
            "projectId":"project-one", "engineInstanceId":"engine-one", "members":[{"memberId":"lsr","instanceId":"lsr-one"}],
            "controllerChannels":[{"channelId":"engine-lsr","destinationMemberId":"lsr","destinationPort":"http","purpose":"inference","allowedOperations":["lsr.route"]}],
        }})
        stored = controller.put_deployment(deployment)
        running = Deployment.model_validate(stored).model_copy(update={"state":"running"})
        running = Deployment.model_validate(controller.store.put_entity("deployment", "deployment-one", running.api_dict(), expected_generation=1))
        for lease_id, instance_id, port in (("engine-lease", "engine-one", 8780), ("lsr-lease", "lsr-one", 9123)):
            controller.store.put_lease(Lease(leaseId=lease_id, instanceId=instance_id, instanceGeneration=1, hostId="host-one", hostEpoch="epoch", state="active", configSha256="hash", telemetryToken="token", paths={"root":str(tmp_path / instance_id)}, ports={"http":port}).api_dict())
        grant = controller._issue_engine_grant(running)
        first = controller.controller_channel_sessions(grant["credential"], project_id="project-one", deployment_id="deployment-one")["items"][0]
        def local(operation: str, method: str):
            return {"destinationInstanceId":"lsr-one","destinationInstanceGeneration":1,"destinationLeaseId":"lsr-lease","operation":operation,"method":method}
        guard = ManagedChannelGuard(ManagedChannelContext(rm_url="http://unused"), introspect=lambda credential, expected: controller.introspect_controller_channel_session(credential, expectation=expected) or {}, expectation_provider=local)
        service = _RouteService()
        discarded: list[int] = []
        parsed_requests: list[str] = []
        discard_started = threading.Event()
        original_discard = LSRRequestHandler._discard_denied_request_body
        original_read_json = LSRRequestHandler._read_json
        def tracked_discard(handler: LSRRequestHandler) -> int:
            discard_started.set()
            count = original_discard(handler)
            discarded.append(count)
            return count
        def tracked_read_json(handler: LSRRequestHandler):
            parsed_requests.append(handler.path)
            return original_read_json(handler)
        monkeypatch.setattr(LSRRequestHandler, "_discard_denied_request_body", tracked_discard)
        monkeypatch.setattr(LSRRequestHandler, "_read_json", tracked_read_json)
        server = LSRHTTPServer(("127.0.0.1", 0), service, managed_channel_guard=guard, require_managed_session=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        assert _post(server.server_port, first["credential"]) == (200, {"accepted":"deterministic"})
        renewed = controller.controller_channel_sessions(grant["credential"], project_id="project-one", deployment_id="deployment-one")["items"][0]
        assert _post(server.server_port, first["credential"])[0] == 401
        # An invalid session is rejected before body parsing, even for malformed JSON.
        parsed_before_denial = len(parsed_requests)
        assert _post(server.server_port, "invalid-session", b"{")[0] == 401
        assert len(parsed_requests) == parsed_before_denial
        assert discarded[-1] == 1
        # Denied bodies are only discarded when their framing is trusted and
        # their complete size fits the strict cap. Missing bytes wait only for
        # the short discard deadline before the handler returns its 401.
        bounded_denials = (
            _raw_denied_post(
                server.server_port,
                content_length=str(DENIED_REQUEST_BODY_DISCARD_MAX_BYTES + 1),
            ),
            _raw_denied_post(server.server_port, content_length="2", body=b"{"),
            _raw_denied_post(server.server_port, content_length="1"),
            _raw_denied_post(server.server_port, content_length="invalid"),
            _raw_denied_post(server.server_port, content_length="9" * 5000),
            _raw_denied_post(
                server.server_port,
                content_length="1",
                additional_content_length="1",
            ),
            _raw_denied_post(
                server.server_port,
                content_length=None,
                transfer_encoding="chunked",
            ),
        )
        assert all(status == 401 and elapsed < 1.0 for status, elapsed in bounded_denials)
        assert discarded[-7:] == [0, 1, 0, 0, 0, 0, 0]
        assert len(parsed_requests) == parsed_before_denial
        discard_started.clear()
        trickle_status, trickle_elapsed = _slow_trickle_denied_post(
            server.server_port,
            discard_started,
        )
        assert trickle_status == 401
        assert trickle_elapsed < 0.45
        assert 0 < discarded[-1] < 4
        assert len(parsed_requests) == parsed_before_denial
        assert _post(server.server_port, renewed["credential"])[0] == 200
        # The managed channel only grants lsr.route; explain/profile remain local-only.
        assert _post(server.server_port, renewed["credential"], b'{"input":"deterministic","strategy_operation":"explain"}')[0] == 401
        assert service.calls == ["deterministic", "deterministic"]
        guard._expectation_provider = lambda operation, method: {**local(operation, method), "destinationLeaseId":"wrong-lease"}
        assert _post(server.server_port, renewed["credential"])[0] == 401
    finally:
        if server is not None:
            server.shutdown(); server.server_close()
        if thread is not None:
            thread.join(timeout=2)
        controller.close()
        if thread is not None:
            assert not thread.is_alive()



def test_explicit_local_session_controls_native_http_handler(tmp_path, monkeypatch):
    from jester_sdk_runtime.standalone_launch import local_guard
    monkeypatch.delenv("JESTER_RM_REQUIRED", raising=False)
    server = LSRHTTPServer(("127.0.0.1", 0), _RouteService(), require_managed_session=True)
    context = {
        "schemaVersion": "jester-local-launch-v1", "host": "127.0.0.1", "port": server.server_port,
        "credential": "local-test-credential-longer-than-32-bytes", "operations": ["lsr.route"],
    }
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(context), encoding="utf-8")
    monkeypatch.setenv("JESTER_STANDALONE_CONTEXT", str(path))
    server.managed_channel_guard = local_guard(host="127.0.0.1", port=server.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import hmac
        for challenge in ["cd" * 32, "bad-challenge", "a" * 129]:
            request = Request(f"http://127.0.0.1:{server.server_port}/health",
                              headers={"X-Jester-Readiness-Challenge": challenge})
            with urlopen(request, timeout=3) as response:
                proof = response.headers.get("X-Jester-Readiness-Proof")
                if challenge == "cd" * 32:
                    assert proof == hmac.new(context["credential"].encode(), challenge.encode(), "sha256").hexdigest()
                else:
                    assert proof is None
        assert _post(server.server_port, "invalid-local-credential")[0] == 401
        assert _post(server.server_port, context["credential"]) == (200, {"accepted": "deterministic"})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert not thread.is_alive()



def test_native_serve_refuses_wrong_local_endpoint_before_loading(tmp_path, monkeypatch):
    from lsr.server import serve
    from lsr.errors import RouterError
    monkeypatch.delenv("JESTER_RM_REQUIRED", raising=False)
    context = {
        "schemaVersion": "jester-local-launch-v1", "host": "127.0.0.1", "port": 18451,
        "credential": "local-test-credential-longer-than-32-bytes", "operations": ["unused"],
    }
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(context), encoding="utf-8")
    monkeypatch.setenv("JESTER_STANDALONE_CONTEXT", str(path))
    with pytest.raises(RouterError, match="component launch context is not authorized"):
        serve(host="0.0.0.0", port=18451, signatures_path="", model_name="must-not-be-loaded", pu_index=None, online=False, default_top_k=1)
