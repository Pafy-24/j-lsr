from __future__ import annotations

import os
import pytest

if os.environ.get("JESTER_CROSS_REPO_TESTS") != "1":
    pytest.skip("cross-repo RM/component integration; set JESTER_CROSS_REPO_TESTS=1", allow_module_level=True)

import json
import os
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve()
PACKAGES = HERE.parents[1].parent
WORKSPACE = PACKAGES.parents[1]
for source in (PACKAGES / "rm" / "src", WORKSPACE / "jester-core" / "packages" / "python" / "core" / "src"):
    sys.path.insert(0, str(source))

from jester_rm.config import Settings
from jester_rm.controller.service import ControllerService
from jester_rm.models import Deployment, EntityMetadata, HostSpec, Lease

from lsr.channel_sessions import ManagedChannelContext, ManagedChannelGuard
from lsr.server import LSRHTTPServer


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


def test_rm_issued_session_controls_actual_lsr_route_handler(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JESTER_RM_HOME", str(tmp_path / "rm-home"))
    monkeypatch.setenv("JESTER_RM_AGENT_HOME", str(tmp_path / "rm-agent"))
    controller = ControllerService(Settings.load(create=True))
    server = None
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
        server = LSRHTTPServer(("127.0.0.1", 0), service, managed_channel_guard=guard, require_managed_session=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        assert _post(server.server_port, first["credential"]) == (200, {"accepted":"deterministic"})
        renewed = controller.controller_channel_sessions(grant["credential"], project_id="project-one", deployment_id="deployment-one")["items"][0]
        assert _post(server.server_port, first["credential"])[0] == 401
        # An invalid session is rejected before body parsing, even for malformed JSON.
        assert _post(server.server_port, "invalid-session", b"{")[0] == 401
        assert _post(server.server_port, renewed["credential"])[0] == 200
        # The managed channel only grants lsr.route; explain/profile remain local-only.
        assert _post(server.server_port, renewed["credential"], b'{"input":"deterministic","strategy_operation":"explain"}')[0] == 401
        assert service.calls == ["deterministic", "deterministic"]
        guard._expectation_provider = lambda operation, method: {**local(operation, method), "destinationLeaseId":"wrong-lease"}
        assert _post(server.server_port, renewed["credential"])[0] == 401
    finally:
        if server is not None:
            server.shutdown(); server.server_close()
        controller.close()
