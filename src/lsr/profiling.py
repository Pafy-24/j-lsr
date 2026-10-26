"""Compatibility name for SDK wall-clock request timing."""
from jester_sdk_runtime.diagnostics import RequestTimer


class RankProfiler(RequestTimer):
    def __init__(self, sample_interval_seconds: float = 0.002):
        # Keep the argument for callers; no resource sampling occurs in LSR.
        super().__init__(total_name="rankingMs")
