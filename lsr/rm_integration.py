from __future__ import annotations

import os
from typing import Any

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
        from jester_rm_client import require_lease

        lease = require_lease()
    except ImportError as exc:
        raise RouterError(
            "RM_LEASE_REQUIRED: install jester-resource-manager and launch this LSR task through jester-rm."
        ) from exc
    except Exception as exc:
        code = getattr(exc, "code", "RM_LEASE_REQUIRED")
        raise RouterError(f"{code}: {exc}") from exc
    if getattr(args, "PU", None) is not None:
        raise RouterError("--PU is owned by Resource Manager for managed LSR processes.")
    if float(getattr(args, "memory_limit_mb", 0.0) or 0.0) != 0.0:
        raise RouterError("--memory-limit-mb is owned by Resource Manager for managed LSR processes.")
    return lease


def managed_processing_unit() -> Any:
    if os.environ.get("JESTER_RM_REQUIRED", "").lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        from jester_rm_client import managed_processing_unit as select_managed

        from .devices import detect_processing_units

        return select_managed(detect_processing_units())
    except Exception as exc:
        code = getattr(exc, "code", "invalid_rm_lease")
        raise RouterError(f"{code}: {exc}") from exc


def start_telemetry() -> Any:
    try:
        from jester_rm_client import start_default_telemetry

        return start_default_telemetry()
    except Exception:
        return None
