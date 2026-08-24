from __future__ import annotations

from types import SimpleNamespace

import pytest

from lsr.errors import RouterError
from lsr.rm_integration import command_requires_rm, require_managed_launch


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
