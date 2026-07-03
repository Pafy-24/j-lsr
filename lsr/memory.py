from __future__ import annotations

import ctypes
import gc
import os
import sys


def current_process_ram_bytes() -> int:
    if os.name == "nt":
        return current_process_ram_bytes_windows()
    return current_process_ram_bytes_posix()


def current_process_ram_bytes_windows() -> int:
    from ctypes import wintypes

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
        ]

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
            return int(counters.WorkingSetSize)
    return 0


def current_process_ram_bytes_posix() -> int:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        factor = 1024 if sys.platform != "darwin" else 1
        return int(usage.ru_maxrss * factor)
    except Exception:
        return 0


def current_vram_bytes() -> int:
    try:
        torch = sys.modules.get("torch")
        if torch is None:
            return 0

        if not torch.cuda.is_available():
            return 0
        total = 0
        for index in range(torch.cuda.device_count()):
            total += int(torch.cuda.memory_allocated(index))
            total += int(torch.cuda.memory_reserved(index))
        return total
    except Exception:
        return 0


def current_footprint_bytes() -> int:
    return current_process_ram_bytes() + current_vram_bytes()


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
