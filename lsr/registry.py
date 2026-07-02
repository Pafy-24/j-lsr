from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import LayeredSignature
from .signatures import SignatureCodec


@dataclass
class SignatureFileState:
    facet: str
    path: str
    mtime: float
    size: int
    model_name: str
    model_hash: str
    dimension: int
    normalize: bool
    distance_metric: str
    layer_sizes: dict[int, int]
    loaded_layers: list[int]
    loaded_vector_bytes: int


class SignatureRegistry:
    def __init__(self, path: str | Path, codec: SignatureCodec | None = None):
        self.path = Path(path)
        self.codec = codec or SignatureCodec()
        self.signatures: dict[str, LayeredSignature] = {}
        self.files: dict[str, Path] = {}
        self.file_stats: dict[str, tuple[float, int]] = {}
        self.stale_signatures: list[str] = []

    def load(self, required: bool = True) -> dict[str, LayeredSignature]:
        self.signatures.clear()
        self.files.clear()
        self.file_stats.clear()
        for file_path in self.signature_files(required):
            self.load_file(file_path)
        return self.signatures

    def refresh(self) -> bool:
        changed = False
        self.stale_signatures = []
        current_files = {file_path.resolve(): file_path for file_path in self.signature_files(False)}
        known_paths = {path.resolve(): facet for facet, path in self.files.items()}

        for resolved_path, facet in list(known_paths.items()):
            file_path = self.files.get(facet)
            if resolved_path not in current_files or file_path is None or not file_path.exists():
                self.stale_signatures.append(facet)
                self.unload(facet)
                changed = True
                continue
            stat = file_path.stat()
            current = (stat.st_mtime, stat.st_size)
            if self.file_stats.get(facet) != current:
                self.load_file(file_path)
                changed = True

        for resolved_path, file_path in current_files.items():
            if resolved_path not in known_paths:
                self.load_file(file_path)
                changed = True

        return changed

    def unload(self, facet: str) -> None:
        key = facet.strip().lower()
        self.signatures.pop(key, None)
        self.files.pop(key, None)
        self.file_stats.pop(key, None)

    def load_file(self, file_path: Path) -> LayeredSignature:
        signature = self.codec.load(file_path)
        stat = file_path.stat()
        self.signatures[signature.facet] = signature
        self.files[signature.facet] = file_path
        self.file_stats[signature.facet] = (stat.st_mtime, stat.st_size)
        return signature

    def signature_files(self, required: bool) -> list[Path]:
        if not self.path.exists():
            if required:
                from .errors import RouterError

                raise RouterError(f"Signature path not found: {self.path}. Run --sign first.")
            return []
        if self.path.is_dir():
            return sorted(self.path.glob("*.sig"))
        if self.path.suffix.lower() == ".sig":
            return [self.path]
        from .errors import RouterError

        raise RouterError("Signature path must be a .sig file or a directory of .sig files.")

    def state(self) -> list[SignatureFileState]:
        states: list[SignatureFileState] = []
        for facet, signature in sorted(self.signatures.items()):
            path = self.files.get(facet)
            mtime, size = self.file_stats.get(facet, (0.0, 0))
            states.append(
                SignatureFileState(
                    facet=facet,
                    path=str(path or ""),
                    mtime=mtime,
                    size=size,
                    model_name=signature.model_name,
                    model_hash=signature.model_hash.hex(),
                    dimension=signature.dimension,
                    normalize=signature.normalize,
                    distance_metric=signature.distance_metric,
                    layer_sizes=dict(signature.layer_sizes),
                    loaded_layers=sorted(signature.layers),
                    loaded_vector_bytes=signature.loaded_vector_bytes(),
                )
            )
        return states

    def loaded_layers(self) -> dict[str, list[int]]:
        return {facet: sorted(signature.layers) for facet, signature in sorted(self.signatures.items())}

    def loaded_vector_bytes(self) -> int:
        return sum(signature.loaded_vector_bytes() for signature in self.signatures.values())

    def signature_files_payload(self) -> dict[str, dict[str, Any]]:
        return {
            state.facet: {
                "path": state.path,
                "mtime": state.mtime,
                "size": state.size,
                "layerSizes": state.layer_sizes,
            }
            for state in self.state()
        }
