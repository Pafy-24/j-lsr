from __future__ import annotations

import ctypes
import gc
import os
import sys
from collections.abc import Mapping
from typing import Any


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
