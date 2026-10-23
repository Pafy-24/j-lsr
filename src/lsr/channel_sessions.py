"""Compatibility imports for the shared managed-channel runtime."""
from jester_sdk_runtime.channel_sessions import (
    ChannelSessionDenied,
    ManagedChannelContext,
    ManagedChannelGuard as _ClaimsGuard,
    rm_introspector,
)

__all__ = [
    "ChannelSessionDenied", "ManagedChannelContext", "ManagedChannelGuard",
    "rm_introspector", "guard_from_managed_environment",
]


class ManagedChannelGuard(_ClaimsGuard):
    """Preserve this component's historical authorization-only return value."""

    def authorize(self, headers, *, operation: str, method: str) -> None:
        super().authorize(headers, operation=operation, method=method)


def guard_from_managed_environment() -> ManagedChannelGuard:
    return ManagedChannelGuard(ManagedChannelContext.from_environment())
