from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from lsr.errors import RouterError
from lsr.rm_integration import command_requires_rm, managed_processing_unit, require_managed_launch


def test_hardware_inventory_cli_is_owned_by_resource_manager() -> None:
    args = SimpleNamespace(PUs=True, command=None)
    assert command_requires_rm(args) is True
    with pytest.raises(RouterError, match="jester-rm resources list"):
        require_managed_launch(args)


def test_local_embedding_service_requires_an_rm_lease(monkeypatch) -> None:
    monkeypatch.delenv("JESTER_RM_LEASE_FILE", raising=False)
    monkeypatch.delenv("JESTER_RM_REQUIRED", raising=False)
    args = SimpleNamespace(
        PUs=False,
        command="serve",
        PU=None,
        memory_limit_mb=0,
    )
    with pytest.raises(RouterError, match="RM_LEASE_REQUIRED"):
        require_managed_launch(args)


def test_remote_rank_does_not_allocate_local_hardware() -> None:
    args = SimpleNamespace(PUs=False, command="rank", local=False)
    assert command_requires_rm(args) is False


def test_managed_cpu_selection_uses_validated_lease_without_gpu(monkeypatch) -> None:
    """LSR maps RM's CPU choice without importing Torch or probing a GPU."""
    client = SimpleNamespace(current_lease=lambda *, required: object())
    cpu = SimpleNamespace(kind="CPU", backend="torch", luid="", device="cpu")
    monkeypatch.setitem(sys.modules, "jester_rm_client", client)
    monkeypatch.setenv("JESTER_RM_REQUIRED", "1")
    monkeypatch.setenv("JESTER_RM_DEVICE_BACKEND", "cpu")
    monkeypatch.delenv("JESTER_RM_DEVICE_LUID", raising=False)
    monkeypatch.setattr("lsr.devices.detect_processing_units", lambda: [cpu])

    assert managed_processing_unit() is cpu
