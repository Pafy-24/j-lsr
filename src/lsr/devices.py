from __future__ import annotations

import json
import os
import platform
import re
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


def windows_accelerator_counters(luids: list[str]) -> dict[str, dict[str, float]]:
    """Read physical GPU/NPU engine and adapter-memory counters by Windows LUID."""
    normalized = [
        luid.upper()
        for luid in luids
        if re.fullmatch(r"[0-9A-Fa-f]{1,16}", str(luid or ""))
    ]
    if os.name != "nt" or not normalized:
        return {}
    ids = ",".join(f"'{luid}'" for luid in normalized)
    rows = powershell_json(
        "& { "
        f"$ids=@({ids}); "
        "foreach($id in $ids){ "
        "$padded=$id.PadLeft(8,'0'); "
        "$util=0.0; $dedicated=0.0; $shared=0.0; "
        "try { "
        "$samples=(Get-Counter \"\\GPU Engine(*luid_0x00000000_0x$padded*)"
        "\\Utilization Percentage\" -ErrorAction Stop).CounterSamples.CookedValue; "
        "if($samples){$util=($samples|Measure-Object -Maximum).Maximum} "
        "} catch {} "
        "try { "
        "$memory=(Get-Counter "
        "\"\\GPU Adapter Memory(*luid_0x00000000_0x$padded*)\\Dedicated Usage\","
        "\"\\GPU Adapter Memory(*luid_0x00000000_0x$padded*)\\Shared Usage\" "
        "-ErrorAction Stop).CounterSamples; "
        "$dedicated=($memory|Where-Object {$_.Path -like '*dedicated usage'}|"
        "Measure-Object CookedValue -Sum).Sum; "
        "$shared=($memory|Where-Object {$_.Path -like '*shared usage'}|"
        "Measure-Object CookedValue -Sum).Sum "
        "} catch {} "
        "[pscustomobject]@{Luid=$id;UtilizationPercent=[double]$util;"
        "DedicatedBytes=[double]$dedicated;SharedBytes=[double]$shared} "
        "} }"
    )
    counters: dict[str, dict[str, float]] = {}
    for row in rows:
        luid = str(row.get("Luid") or "").upper()
        if not luid:
            continue
        counters[luid] = {
            "utilizationPercent": float(row.get("UtilizationPercent") or 0.0),
            "dedicatedBytes": float(row.get("DedicatedBytes") or 0.0),
            "sharedBytes": float(row.get("SharedBytes") or 0.0),
        }
    return counters


class ProcessingUnitDetector:
    def detect(
        self,
        include_windows_devices: bool = True,
        include_npu: bool = True,
    ) -> list[ProcessingUnit]:
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
        if include_windows_devices:
            self._add_windows_ml_units(units, include_npu=include_npu)
            self._add_windows_visible_units(units)
        return units

    def select(self, index: int | str | None) -> ProcessingUnit | None:
        if index is None:
            return None
        if isinstance(index, str):
            if index.lower() != "auto":
                try:
                    index = int(index)
                except ValueError as exc:
                    raise RouterError("--PU must be an integer index or auto.") from exc
            else:
                units = self.detect()
                return self.select_auto(units)
        basic_units = self.detect(include_windows_devices=False)
        for unit in basic_units:
            if unit.index == index and unit.usable:
                return unit
        dml_units = self.detect(include_npu=False)
        for unit in dml_units:
            if unit.index == index and unit.usable:
                return unit
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

    def select_auto(self, units: list[ProcessingUnit] | None = None) -> ProcessingUnit | None:
        active_units = units or self.detect()
        usable = [unit for unit in active_units if unit.usable]
        if not usable:
            return None
        for preferred in (
            "torch.cuda",
            "windowsml.vitisai",
            "windowsml.dml",
            "torch.mps",
            "torch_directml",
            "torch_xla",
        ):
            for unit in usable:
                if unit.backend == preferred:
                    return unit
        return usable[0]

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

    def _add_windows_ml_units(
        self,
        units: list[ProcessingUnit],
        include_npu: bool = True,
    ) -> None:
        if os.name != "nt":
            return
        try:
            from .windows_ml import DML_PROVIDER, VITIS_PROVIDER, windows_ml_devices

            devices, npu_error = windows_ml_devices(include_npu=include_npu)
        except Exception:
            return

        for item in devices:
            provider = str(item.get("provider") or "")
            name = str(item.get("name") or "").strip()
            vendor = str(item.get("vendor") or "").lower()
            metadata = {
                str(key): str(value)
                for key, value in dict(item.get("metadata") or {}).items()
            }
            if not name:
                continue
            if provider == DML_PROVIDER:
                integrated = metadata.get("Discrete") == "0"
                amd = "amd" in vendor or "advanced micro devices" in vendor
                if not integrated or not amd or self._unit_exists(units, name):
                    continue
                units.append(
                    ProcessingUnit(
                        index=len(units),
                        kind="iGPU",
                        name=name,
                        backend="windowsml.dml",
                        device=f"LUID 0x{item.get('luid') or '?'}",
                        usable=True,
                        note="Windows ML / DirectML; explicitly bound to the integrated Radeon",
                        provider=provider,
                        luid=str(item.get("luid") or ""),
                        metadata=metadata,
                    )
                )
            elif include_npu and provider == VITIS_PROVIDER and not any(
                unit.backend == "windowsml.vitisai" for unit in units
            ):
                units.append(
                    ProcessingUnit(
                        index=len(units),
                        kind="NPU",
                        name=name,
                        backend="windowsml.vitisai",
                        device=f"LUID 0x{item.get('luid') or '?'}",
                        usable=True,
                        note=(
                            "Windows ML / Vitis AI; MiniLM requires "
                            "onnx/model_npu_a16w8.onnx"
                        ),
                        provider=provider,
                        luid=str(item.get("luid") or ""),
                        metadata=metadata,
                    )
                )

        if (
            include_npu
            and npu_error
            and not any(unit.backend == "windowsml.vitisai" for unit in units)
        ):
            for accelerator in windows_compute_accelerators():
                name = str(accelerator.get("FriendlyName") or "").strip()
                if name and not self._unit_exists(units, name):
                    units.append(
                        ProcessingUnit(
                            index=len(units),
                            kind="NPU",
                            name=name,
                            backend="windowsml.vitisai",
                            device="unavailable",
                            usable=False,
                            note=f"Windows ML Vitis AI registration failed: {npu_error}",
                        )
                    )

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
            if (
                not name
                or self._unit_exists(units, name)
                or any(unit.backend == "windowsml.vitisai" for unit in units)
            ):
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


def select_processing_unit(index: int | str | None) -> ProcessingUnit | None:
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
