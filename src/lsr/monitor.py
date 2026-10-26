"""Compatibility name for SDK functional diagnostics, without hardware probes."""
from typing import Any

from jester_sdk_runtime.diagnostics import RequestDiagnostics


class ServerMonitor(RequestDiagnostics):
    def __init__(self, service: Any, refresh_hz: float = 1.0):
        super().__init__()

    def start(self) -> None:
        """No sampler is started; RMA observes the process externally."""

    def stop(self) -> None:
        """No background monitor is owned by the component."""
