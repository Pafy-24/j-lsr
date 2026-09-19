from __future__ import annotations

import ctypes
import gc
import os
import sys
from collections.abc import Mapping
from typing import Any


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def current_process_memory_bytes() -> tuple[int, int]:
    """Return current working-set/RSS and private bytes from one OS sample."""
    if os.name == "nt":
        counters = process_memory_counters_windows()
        if counters is None:
            return 0, 0
        return int(counters.WorkingSetSize), int(counters.PrivateUsage)
    resident = current_process_ram_bytes_posix()
    return resident, resident


def current_process_ram_bytes() -> int:
    return current_process_memory_bytes()[0]


def current_process_ram_bytes_windows() -> int:
    return current_process_memory_bytes()[0]


def current_process_private_bytes() -> int:
    return current_process_memory_bytes()[1]


def process_memory_counters_windows() -> Any:
    from ctypes import wintypes

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    for function in (
        getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None),
        getattr(ctypes.WinDLL("psapi.dll"), "GetProcessMemoryInfo", None),
    ):
        if function is None:
            continue
        function.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        function.restype = wintypes.BOOL
        if function(handle, ctypes.byref(counters), counters.cb):
            return counters
    return None


def current_process_ram_bytes_posix() -> int:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm", "r", encoding="ascii") as handle:
                resident_pages = int(handle.read().split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, ValueError):
            pass
    try:
        import resource

        # Fallback for POSIX platforms without /proc. ru_maxrss is a peak,
        # so Linux uses /proc above to keep memory-limit checks current.
        usage = resource.getrusage(resource.RUSAGE_SELF)
        factor = 1024 if sys.platform != "darwin" else 1
        return int(usage.ru_maxrss * factor)
    except Exception:
        return 0


def current_vram_bytes() -> int:
    """CUDA memory reserved by this process (allocated is included in it)."""
    try:
        torch = sys.modules.get("torch")
        if torch is None:
            return 0

        if not torch.cuda.is_available():
            return 0
        total = 0
        for index in range(torch.cuda.device_count()):
            total += int(torch.cuda.memory_reserved(index))
        return total
    except Exception:
        return 0


def cuda_memory_by_device() -> list[dict[str, int | str]]:
    """Return non-overlapping CUDA allocator counters for every device."""
    try:
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_available():
            return []
        result: list[dict[str, int | str]] = []
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            allocated = int(torch.cuda.memory_allocated(index))
            reserved = int(torch.cuda.memory_reserved(index))
            result.append(
                {
                    "index": index,
                    "name": str(torch.cuda.get_device_name(index)),
                    "allocated": allocated,
                    "reserved": reserved,
                    "cache": max(0, reserved - allocated),
                    "deviceUsed": int(total - free),
                    "deviceTotal": int(total),
                }
            )
        return result
    except Exception:
        return []


def current_footprint_bytes() -> int:
    return current_process_ram_bytes() + current_vram_bytes()


def estimate_object_bytes(value: object, seen: set[int] | None = None) -> int:
    """Best-effort owned Python/NumPy size, excluding callables and modules."""
    if value is None or callable(value):
        return 0
    active_seen = seen if seen is not None else set()
    identity = id(value)
    if identity in active_seen:
        return 0
    active_seen.add(identity)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return int(nbytes)
    size = sys.getsizeof(value, 0)
    if isinstance(value, Mapping):
        return size + sum(
            estimate_object_bytes(key, active_seen) + estimate_object_bytes(item, active_seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(estimate_object_bytes(item, active_seen) for item in value)
    namespace = getattr(value, "__dict__", None)
    if isinstance(namespace, dict):
        return size + estimate_object_bytes(namespace, active_seen)
    return size


def release_unused_memory() -> None:
    gc.collect()
    try:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    trim_process_working_set()


def trim_process_working_set() -> None:
    if os.name == "nt":
        trim_process_working_set_windows()
    else:
        trim_process_working_set_posix()


def trim_process_working_set_windows() -> None:
    from ctypes import wintypes

    handle = ctypes.windll.kernel32.GetCurrentProcess()
    for function_name in ("K32EmptyWorkingSet", "EmptyWorkingSet"):
        try:
            library = ctypes.windll.kernel32 if function_name.startswith("K32") else ctypes.WinDLL("psapi.dll")
            function = getattr(library, function_name)
            function.argtypes = [wintypes.HANDLE]
            function.restype = wintypes.BOOL
            if function(handle):
                return
        except Exception:
            continue

    try:
        function = ctypes.windll.kernel32.SetProcessWorkingSetSize
        function.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t]
        function.restype = wintypes.BOOL
        function(handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    except Exception:
        pass


def trim_process_working_set_posix() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def format_bytes(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
