from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from typing import Any

from .devices import detect_processing_units
from .memory import cuda_memory_by_device, current_process_ram_bytes, current_vram_bytes, format_bytes


class ServerMonitor:
    """Rich live dashboard for a running LSR service."""

    def __init__(self, service: Any, refresh_hz: float = 1.0):
        self.service = service
        self.refresh_hz = max(0.25, float(refresh_hz))
        self._operation = "idle"
        self._detail = "waiting for requests"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._dirty = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = time.monotonic()
        self._units = detect_processing_units()
        self._requests = 0
        self._last_ms = 0.0
        self._last_profile: dict[str, Any] = {}
        initial_ram = current_process_ram_bytes()
        self._ram_samples: deque[int] = deque([initial_ram], maxlen=32)
        self._peak_ram = initial_ram
        self._components: dict[tuple[str, str], dict[str, Any]] = {}
        self._process = None
        self._gpu_samples: dict[int, tuple[float, str, str]] = {}
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(None)
        except (ImportError, OSError):
            pass

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="lsr-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def update(self, operation: str, detail: str = "") -> None:
        with self._lock:
            self._operation = operation
            self._detail = detail
        self._dirty.set()

    def set_component(
        self,
        memory: str,
        name: str,
        size: int,
        location: str,
        persistent: bool = False,
    ) -> None:
        with self._lock:
            self._components[(memory, name)] = {
                "memory": memory,
                "name": name,
                "size": max(0, int(size)),
                "location": location,
                "persistent": persistent,
            }
        self._dirty.set()

    def remove_component(self, memory: str, name: str) -> None:
        with self._lock:
            self._components.pop((memory, name), None)
        self._dirty.set()

    def remove_components(self, memory: str, prefix: str) -> None:
        with self._lock:
            for key in list(self._components):
                if key[0] == memory and key[1].startswith(prefix):
                    self._components.pop(key, None)
        self._dirty.set()

    def components_payload(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._components.values()]

    def request_finished(self, elapsed_ms: float, profile: dict[str, Any] | None = None) -> None:
        ram = current_process_ram_bytes()
        with self._lock:
            self._requests += 1
            self._last_ms = elapsed_ms
            self._last_profile = dict(profile or {})
            self._operation = "idle"
            self._detail = "waiting for requests"
            self._ram_samples.append(ram)
            self._peak_ram = max(self._peak_ram, ram)
        self._dirty.set()

    def memory_stats(self) -> dict[str, int]:
        with self._lock:
            samples = list(self._ram_samples)
            peak = self._peak_ram
        return {
            "currentRamBytes": current_process_ram_bytes(),
            "peakObservedRamBytes": peak,
            "windowGrowthBytes": (samples[-1] - samples[0]) if len(samples) > 1 else 0,
            "windowSamples": len(samples),
        }

    def last_profile(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_profile)

    def _run(self) -> None:
        from rich.console import Console
        from rich.live import Live

        console = Console()
        with Live(self.render(), console=console, refresh_per_second=self.refresh_hz, screen=False) as live:
            last_idle_refresh = time.monotonic()
            while not self._stop.wait(1.0 / self.refresh_hz):
                with self._lock:
                    idle = self._operation == "idle"
                now = time.monotonic()
                if not self._dirty.is_set() and idle and now - last_idle_refresh < 10.0:
                    continue
                self._dirty.clear()
                live.update(self.render())
                if idle:
                    last_idle_refresh = now

    def render(self) -> Any:
        from rich.columns import Columns
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        with self._lock:
            operation, detail = self._operation, self._detail
            requests, last_ms = self._requests, self._last_ms
            components = [dict(item) for item in self._components.values()]
            ram_samples = list(self._ram_samples)
            peak_ram = self._peak_ram
            last_profile = dict(self._last_profile)

        pu_table = Table(title="Processing units", expand=True)
        pu_table.add_column("PU")
        pu_table.add_column("Backend")
        pu_table.add_column("Util", justify="right")
        pu_table.add_column("Memory", justify="right")
        cpu_total, cpu_process = self._cpu_usage()
        for unit in self._units:
            utilization = "unavailable"
            memory = "-"
            if unit.kind == "CPU":
                utilization = f"system {cpu_total:.1f}% / LSR {cpu_process:.1f}%"
                memory = format_bytes(current_process_ram_bytes())
            elif unit.backend == "torch.cuda":
                utilization, memory = self._cuda_usage(unit.device)
            elif unit.usable:
                utilization = "active" if unit.backend in detail else "idle"
            pu_table.add_row(f"{unit.index} {unit.kind} {unit.name}", unit.backend, utilization, memory)

        ram_table = self._component_table("RAM components", "RAM", components)
        vram_table = self._component_table("VRAM components", "VRAM", components)

        summary = (
            f"[bold]{operation}[/bold]  {detail}\n"
            f"requests={requests}  last={last_ms:.1f} ms  "
            f"RAM={format_bytes(current_process_ram_bytes())}  "
            f"VRAM={format_bytes(current_vram_bytes())}  "
            f"RAM peak={format_bytes(peak_ram)}  "
            f"trend={format_bytes(max(0, ram_samples[-1] - ram_samples[0])) if ram_samples else '0 B'}  "
            f"uptime={time.monotonic() - self._started_at:.0f}s"
        )
        resources = last_profile.get("resources", {}) if isinstance(last_profile, dict) else {}
        if resources:
            cuda_peak = sum(
                int(item.get("peakAllocatedDeltaBytes", 0))
                for item in resources.get("cuda", [])
            )
            cuda_retained = sum(
                int(item.get("allocatedRetainedDeltaBytes", 0))
                for item in resources.get("cuda", [])
            )
            summary += (
                f"\nlast rank peak: RAM +{format_bytes(int(resources.get('ramPeakDeltaBytes', 0)))}  "
                f"CUDA +{format_bytes(cuda_peak)}  retained: "
                f"working-set {resources.get('ramRetainedDeltaBytes', 0) / 1024:+.1f} KB  "
                f"private {resources.get('privateRetainedDeltaBytes', 0) / 1024:+.1f} KB  "
                f"CUDA {cuda_retained / 1024:+.1f} KB"
            )
        return Panel(Group(pu_table, Columns([ram_table, vram_table], equal=True)), title=summary)

    def _component_table(self, title: str, memory: str, components: list[dict[str, Any]]) -> Any:
        from rich.table import Table

        table = Table(title=title, expand=True)
        table.add_column("Component")
        table.add_column("Location")
        table.add_column("Size", justify="right")
        matching = sorted(
            (item for item in components if item["memory"] == memory),
            key=lambda item: (-int(item["size"]), str(item["name"])),
        )
        tracked = sum(int(item["size"]) for item in matching)
        total = current_process_ram_bytes() if memory == "RAM" else current_vram_bytes()
        for item in matching:
            table.add_row(str(item["name"]), str(item["location"]), format_bytes(int(item["size"])))
        if memory == "VRAM":
            cuda = cuda_memory_by_device()
            allocated = sum(int(item["allocated"]) for item in cuda)
            cache = sum(int(item["cache"]) for item in cuda)
            table.add_row("CUDA allocator cache", "all CUDA devices", format_bytes(cache))
            table.add_row("other allocated tensors", "all CUDA devices", format_bytes(max(0, allocated - tracked)))
        else:
            table.add_row("runtime/unattributed", "Python/native process", format_bytes(max(0, total - tracked)))
        table.add_row("[bold]TOTAL[/bold]", memory, f"[bold]{format_bytes(total)}[/bold]")
        return table

    def _cpu_usage(self) -> tuple[float, float]:
        try:
            import psutil

            total = float(psutil.cpu_percent(None))
            process = float(self._process.cpu_percent(None)) if self._process is not None else 0.0
            return total, process
        except (ImportError, OSError):
            return 0.0, 0.0

    def _cuda_usage(self, device: Any) -> tuple[str, str]:
        try:
            import torch

            index = int(str(device).split(":")[-1])
            now = time.monotonic()
            cached = self._gpu_samples.get(index)
            if cached is not None and now - cached[0] < 2.0:
                return cached[1], cached[2]
            # torch.cuda.utilization() initializes persistent NVML state and
            # adds roughly 100 MB to the Python process on Windows. A short
            # nvidia-smi child process keeps that monitoring overhead outside
            # the long-lived routing server.
            utilization = self._nvidia_utilization(index)
            free, total = torch.cuda.mem_get_info(index)
            used = int(total - free)
            reserved = int(torch.cuda.memory_reserved(index))
            result = (f"{utilization:.1f}%", f"device {format_bytes(used)} / LSR {format_bytes(reserved)}")
            self._gpu_samples[index] = (now, result[0], result[1])
            return result
        except Exception:
            return "unavailable", format_bytes(current_vram_bytes())

    @staticmethod
    def _nvidia_utilization(index: int) -> float:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=True,
            creationflags=creationflags,
        )
        return float(completed.stdout.strip().splitlines()[0])
