from __future__ import annotations

import sys
import threading
import time
from typing import Any

from .memory import cuda_memory_by_device, current_process_private_bytes, current_process_ram_bytes


class RankProfiler:
    """Low-overhead per-request latency and memory profiler."""

    def __init__(self, sample_interval_seconds: float = 0.002):
        self.sample_interval = max(0.001, float(sample_interval_seconds))
        self.started = 0.0
        self.cpu_started = 0.0
        self.ram_before = 0
        self.ram_peak = 0
        self.private_before = 0
        self.private_peak = 0
        self.vram_before: list[dict[str, int | str]] = []
        self.stages: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.ram_before = current_process_ram_bytes()
        self.ram_peak = self.ram_before
        self.private_before = current_process_private_bytes()
        self.private_peak = self.private_before
        self.vram_before = cuda_memory_by_device()
        self._reset_cuda_peaks()
        self.cpu_started = time.process_time()
        self.started = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_ram, name="lsr-rank-profiler", daemon=True)
        self._thread.start()

    def stage(self, name: str, started: float) -> None:
        self.stages[name] = (time.perf_counter() - started) * 1000.0

    def finish(self) -> dict[str, Any]:
        total_ms = (time.perf_counter() - self.started) * 1000.0
        cpu_ms = (time.process_time() - self.cpu_started) * 1000.0
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.05)
        ram_after = current_process_ram_bytes()
        private_after = current_process_private_bytes()
        self.ram_peak = max(self.ram_peak, ram_after)
        self.private_peak = max(self.private_peak, private_after)
        cuda = self._cuda_result()
        timing = {f"{name}Ms": round(value, 3) for name, value in self.stages.items()}
        timing.update({"rankingMs": round(total_ms, 3), "cpuTimeMs": round(cpu_ms, 3)})
        return {
            "timing": timing,
            "resources": {
                "ramBeforeBytes": self.ram_before,
                "ramAfterBytes": ram_after,
                "ramPeakBytes": self.ram_peak,
                "ramPeakDeltaBytes": max(0, self.ram_peak - self.ram_before),
                "ramRetainedDeltaBytes": ram_after - self.ram_before,
                "privateBeforeBytes": self.private_before,
                "privateAfterBytes": private_after,
                "privatePeakBytes": self.private_peak,
                "privatePeakDeltaBytes": max(0, self.private_peak - self.private_before),
                "privateRetainedDeltaBytes": private_after - self.private_before,
                "cuda": cuda,
                "ramSampleIntervalMs": self.sample_interval * 1000.0,
            },
        }

    def _sample_ram(self) -> None:
        while not self._stop.wait(self.sample_interval):
            self.ram_peak = max(self.ram_peak, current_process_ram_bytes())
            self.private_peak = max(self.private_peak, current_process_private_bytes())

    @staticmethod
    def _reset_cuda_peaks() -> None:
        try:
            torch = sys.modules.get("torch")
            if torch is None or not torch.cuda.is_available():
                return
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)
        except Exception:
            return

    def _cuda_result(self) -> list[dict[str, int | str]]:
        try:
            torch = sys.modules.get("torch")
            if torch is None or not torch.cuda.is_available():
                return []
            result = []
            before = {int(item["index"]): item for item in self.vram_before}
            for index in range(torch.cuda.device_count()):
                allocated = int(torch.cuda.memory_allocated(index))
                reserved = int(torch.cuda.memory_reserved(index))
                peak_allocated = int(torch.cuda.max_memory_allocated(index))
                peak_reserved = int(torch.cuda.max_memory_reserved(index))
                prior = before.get(index, {})
                before_allocated = int(prior.get("allocated", 0))
                before_reserved = int(prior.get("reserved", 0))
                result.append(
                    {
                        "index": index,
                        "name": str(torch.cuda.get_device_name(index)),
                        "allocatedBeforeBytes": before_allocated,
                        "allocatedBytes": allocated,
                        "reservedBeforeBytes": before_reserved,
                        "reservedBytes": reserved,
                        "peakAllocatedBytes": peak_allocated,
                        "peakReservedBytes": peak_reserved,
                        "peakAllocatedDeltaBytes": max(0, peak_allocated - before_allocated),
                        "peakReservedDeltaBytes": max(0, peak_reserved - before_reserved),
                        "allocatedRetainedDeltaBytes": allocated - before_allocated,
                        "reservedRetainedDeltaBytes": reserved - before_reserved,
                    }
                )
            return result
        except Exception:
            return []
