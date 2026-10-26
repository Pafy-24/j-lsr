from __future__ import annotations

import os
from typing import Any

from jester_sdk_runtime.launch import ManagedLaunchError, require_execution_launch

from .errors import RouterError


MANAGED_COMMANDS = {"serve", "sign", "benchmark-models", "evaluate"}


def command_requires_rm(args: Any) -> bool:
    if getattr(args, "PUs", False):
        return True
    command = getattr(args, "command", None)
    if command in MANAGED_COMMANDS:
        return True
    return command == "rank" and bool(getattr(args, "local", False))


def require_managed_launch(args: Any) -> Any:
    if not command_requires_rm(args):
        return None
    if getattr(args, "PUs", False):
        raise RouterError(
            "RM_LEASE_REQUIRED: --PUs was replaced by `jester-rm resources list`; "
            "hardware inventory is owned by Resource Manager."
        )
    try:
        lease = require_execution_launch()
    except ManagedLaunchError as exc:
        message = "install jester-rm-client and launch this LSR task through jester-rm." if exc.client_unavailable else str(exc)
        raise RouterError(f"{exc.code}: {message}") from exc
    if lease is None:
        return None
    if getattr(args, "PU", None) is not None:
        raise RouterError("--PU is owned by Resource Manager for managed LSR processes.")
    if float(getattr(args, "memory_limit_mb", 0.0) or 0.0) != 0.0:
        raise RouterError("--memory-limit-mb is owned by Resource Manager for managed LSR processes.")
    return lease


def managed_processing_unit() -> Any:
    if os.environ.get("JESTER_RM_REQUIRED", "").lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        from jester_rm_client import current_lease

        from .devices import detect_processing_units

        # The client validates that RM issued this lease.  LSR only maps the
        # RM-selected backend/LUID onto its local, already-detected adapters;
        # it neither inventories nor reserves hardware itself.
        current_lease(required=True)
        backend = os.environ.get("JESTER_RM_DEVICE_BACKEND", "cpu")
        luid = os.environ.get("JESTER_RM_DEVICE_LUID", "").upper().removeprefix("0X")
        units = detect_processing_units()
        if backend == "cpu":
            return next(
                (unit for unit in units if str(getattr(unit, "kind", "")).upper() == "CPU"),
                None,
            )
        for unit in units:
            if str(getattr(unit, "backend", "")) != backend:
                continue
            unit_luid = str(getattr(unit, "luid", "")).upper().removeprefix("0X")
            if luid and unit_luid != luid:
                continue
            return unit
        resource_id = os.environ.get("JESTER_RM_DEVICE_ID", "unknown")
        raise RouterError(
            f"Managed resource {resource_id} ({backend}, LUID {luid or '-'}) "
            "is not available in this process."
        )
    except Exception as exc:
        code = getattr(exc, "code", "invalid_rm_lease")
        raise RouterError(f"{code}: {exc}") from exc
        code = getattr(exc, "code", "invalid_rm_lease")
        raise RouterError(f"{code}: {exc}") from exc
