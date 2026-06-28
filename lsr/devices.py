from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Any

from .console import console
from .dependencies import make_table
from .errors import RouterError
from .models import ProcessingUnit


def powershell_json(script: str) -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    try:
        output = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"{script} | ConvertTo-Json -Compress",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=8,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return []
    if not output:
        return []
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def windows_gpu_devices() -> list[dict[str, Any]]:
    return powershell_json(
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterCompatibility,Status,PNPDeviceID"
    )


def windows_compute_accelerators() -> list[dict[str, Any]]:
    return powershell_json(
        "Get-PnpDevice -Class ComputeAccelerator | "
        "Select-Object FriendlyName,Status,InstanceId"
    )


class ProcessingUnitDetector:
    def detect(self) -> list[ProcessingUnit]:
        units: list[ProcessingUnit] = []
        cpu_name = platform.processor() or platform.machine() or "CPU"
        units.append(
            ProcessingUnit(
                index=0,
                kind="CPU",
                name=f"{cpu_name} ({os.cpu_count() or '?'} logical cores)",
                backend="torch",
                device="cpu",
                usable=True,
            )
        )

        try:
            import torch
        except ModuleNotFoundError:
            return units

        if torch.cuda.is_available():
            for cuda_index in range(torch.cuda.device_count()):
                units.append(
                    ProcessingUnit(
                        index=len(units),
                        kind="GPU",
                        name=torch.cuda.get_device_name(cuda_index),
                        backend="torch.cuda",
                        device=f"cuda:{cuda_index}",
                        usable=True,
                    )
                )

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            units.append(
                ProcessingUnit(
                    index=len(units),
                    kind="GPU",
                    name="Apple Metal Performance Shaders",
                    backend="torch.mps",
                    device="mps",
                    usable=True,
                )
            )

        self._add_directml_units(units)
        self._add_xla_units(units)
        self._add_windows_visible_units(units)
        return units

    def select(self, index: int | None) -> ProcessingUnit | None:
        if index is None:
            return None
        units = self.detect()
        for unit in units:
            if unit.index == index:
                if not unit.usable:
                    raise RouterError(
                        f"--PU {index} selects '{unit.name}', but it is not usable by "
                        f"the current embedding backend. {unit.note}"
                    )
                return unit
        available = ", ".join(str(unit.index) for unit in units)
        raise RouterError(f"Invalid --PU index {index}. Available indexes: {available}")

    def _add_directml_units(self, units: list[ProcessingUnit]) -> None:
        try:
            import torch_directml
        except ModuleNotFoundError:
            return
        count_fn = getattr(torch_directml, "device_count", None)
        count = int(count_fn()) if callable(count_fn) else 1
        for directml_index in range(max(0, count)):
            device = torch_directml.device(directml_index)
            units.append(
                ProcessingUnit(
                    index=len(units),
                    kind="GPU/NPU",
                    name=f"DirectML device {directml_index}",
                    backend="torch_directml",
                    device=device,
                    usable=True,
                )
            )

    def _add_xla_units(self, units: list[ProcessingUnit]) -> None:
        try:
            import torch_xla.core.xla_model as xm
        except ModuleNotFoundError:
            return
        try:
            units.append(
                ProcessingUnit(
                    index=len(units),
                    kind="TPU",
                    name="XLA device",
                    backend="torch_xla",
                    device=xm.xla_device(),
                    usable=True,
                )
            )
        except Exception:
            pass

    def _add_windows_visible_units(self, units: list[ProcessingUnit]) -> None:
        for gpu in windows_gpu_devices():
            name = str(gpu.get("Name") or "").strip()
            if not name or self._unit_exists(units, name):
                continue
            vendor = str(gpu.get("AdapterCompatibility") or "").strip()
            backend = "windows.display"
            note = "detected by Windows; not usable by current PyTorch backend"
            if "nvidia" in f"{name} {vendor}".lower():
                note = "detected by NVIDIA driver; install CUDA-enabled PyTorch to use"
            units.append(
                ProcessingUnit(
                    index=len(units),
                    kind="GPU",
                    name=name,
                    backend=backend,
                    device="unavailable",
                    usable=False,
                    note=note,
                )
            )

        for accelerator in windows_compute_accelerators():
            name = str(accelerator.get("FriendlyName") or "").strip()
            if not name or self._unit_exists(units, name):
                continue
            units.append(
                ProcessingUnit(
                    index=len(units),
                    kind="NPU",
                    name=name,
                    backend="windows.compute",
                    device="unavailable",
                    usable=False,
                    note="detected by Windows; no SentenceTransformer/PyTorch backend available",
                )
            )

    @staticmethod
    def _unit_exists(units: list[ProcessingUnit], name: str) -> bool:
        needle = name.lower()
        return any(needle and needle in unit.name.lower() for unit in units)


def detect_processing_units() -> list[ProcessingUnit]:
    return ProcessingUnitDetector().detect()


def select_processing_unit(index: int | None) -> ProcessingUnit | None:
    return ProcessingUnitDetector().select(index)


def unit_exists(units: list[ProcessingUnit], name: str) -> bool:
    return ProcessingUnitDetector._unit_exists(units, name)


def print_processing_units() -> None:
    table = make_table(title="Available Processing Units")
    table.add_column("Index", justify="right")
    table.add_column("Type")
    table.add_column("Usable")
    table.add_column("Backend")
    table.add_column("Device")
    table.add_column("Name")
    table.add_column("Note")

    for unit in detect_processing_units():
        table.add_row(
            str(unit.index),
            unit.kind,
            "yes" if unit.usable else "no",
            unit.backend,
            str(unit.device),
            unit.name,
            unit.note,
        )
    console.print(table)
