from __future__ import annotations

import hashlib
import json
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


@dataclass(frozen=True)
class SignatureSnapshot:
    """Portable identity metadata for one loaded signature."""

    facet: str
    path: str
    content_sha256: str
    loaded_sha256: str
    size: int
    model_name: str
    model_hash: str
    dimension: int
    normalize: bool
    distance_metric: str
    layer_sizes: tuple[tuple[int, int], ...]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "facet": self.facet,
            "contentSha256": self.content_sha256,
            "loadedSha256": self.loaded_sha256,
            "size": self.size,
            "modelName": self.model_name,
            "modelHash": self.model_hash,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "distanceMetric": self.distance_metric,
            "layerSizes": [[level, count] for level, count in self.layer_sizes],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["path"] = self.path
        return value


@dataclass(frozen=True)
class RegistrySnapshot:
    """Immutable, content-addressed view of a loaded signature registry."""

    schema_version: str
    entries: tuple[SignatureSnapshot, ...]

    @property
    def facets(self) -> tuple[str, ...]:
        return tuple(entry.facet for entry in self.entries)

    @property
    def identity(self) -> str:
        payload = {
            "schemaVersion": self.schema_version,
            "entries": [entry.identity_payload() for entry in self.entries],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def candidate_facets(self, requested: Any = None) -> tuple[str, ...]:
        """Return a request-local intersection without changing the snapshot."""
        if requested is None:
            return self.facets
        values = [requested] if isinstance(requested, str) else requested
        allowed = frozenset(str(value).strip().lower() for value in values if str(value).strip())
        return tuple(facet for facet in self.facets if facet in allowed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "snapshotId": self.identity,
            "facets": [entry.to_dict() for entry in self.entries],
        }


class SignatureRegistry:
    def __init__(self, path: str | Path, codec: SignatureCodec | None = None):
        self.path = Path(path)
        self.codec = codec or SignatureCodec()
        self.signatures: dict[str, LayeredSignature] = {}
        self.files: dict[str, Path] = {}
        self.file_stats: dict[str, tuple[float, int]] = {}
        self.file_digests: dict[str, str] = {}
        self.stale_signatures: list[str] = []
        self.disabled_facets: set[str] = set()

    def load(self, required: bool = True) -> dict[str, LayeredSignature]:
        self.signatures.clear()
        self.files.clear()
        self.file_stats.clear()
        self.file_digests.clear()
        self.disabled_facets.clear()
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
            if (
                resolved_path not in known_paths
                and file_path.stem.lower() not in self.disabled_facets
            ):
                self.load_file(file_path)
                changed = True

        return changed

    def unload(self, facet: str, disable: bool = False) -> None:
        key = facet.strip().lower()
        self.signatures.pop(key, None)
        self.files.pop(key, None)
        self.file_stats.pop(key, None)
        self.file_digests.pop(key, None)
        if disable:
            self.disabled_facets.add(key)

    def load_file(self, file_path: Path) -> LayeredSignature:
        signature = self.codec.load(file_path)
        stat = file_path.stat()
        self.signatures[signature.facet] = signature
        self.files[signature.facet] = file_path
        self.file_stats[signature.facet] = (stat.st_mtime, stat.st_size)
        self.file_digests[signature.facet] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        self.disabled_facets.discard(signature.facet)
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

    def snapshot(self) -> RegistrySnapshot:
        """Capture loaded signatures without exposing mutable registry state."""
        entries: list[SignatureSnapshot] = []
        for facet, signature in sorted(self.signatures.items()):
            path = self.files.get(facet)
            resolved_path = str(path.resolve()) if path is not None else ""
            if path is not None and path.exists() and path.is_file():
                if facet not in self.file_digests:
                    self.file_digests[facet] = hashlib.sha256(path.read_bytes()).hexdigest()
                content_sha256 = self.file_digests[facet]
                size = int(self.file_stats.get(facet, (0.0, path.stat().st_size))[1])
            else:
                metadata = {
                    "facet": signature.facet,
                    "dimension": signature.dimension,
                    "modelName": signature.model_name,
                    "modelHash": signature.model_hash.hex(),
                    "normalize": signature.normalize,
                    "distanceMetric": signature.distance_metric,
                    "layerSizes": sorted((int(level), int(count)) for level, count in signature.layer_sizes.items()),
                }
                encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
                content_sha256 = hashlib.sha256(encoded).hexdigest()
                size = len(encoded)
            loaded_sha256 = _loaded_signature_digest(signature)
            entries.append(
                SignatureSnapshot(
                    facet=signature.facet,
                    path=resolved_path,
                    content_sha256=content_sha256,
                    loaded_sha256=loaded_sha256,
                    size=size,
                    model_name=signature.model_name,
                    model_hash=signature.model_hash.hex(),
                    dimension=int(signature.dimension),
                    normalize=bool(signature.normalize),
                    distance_metric=signature.distance_metric,
                    layer_sizes=tuple(sorted((int(level), int(count)) for level, count in signature.layer_sizes.items())),
                )
            )
        return RegistrySnapshot(schema_version="lsr-registry-snapshot-v1", entries=tuple(entries))


def _loaded_signature_digest(signature: LayeredSignature) -> str:
    """Digest immutable signature content, independent of lazy cache state."""
    digest = hashlib.sha256()
    digest.update(str(signature.facet).encode("utf-8"))
    digest.update(str(int(signature.dimension)).encode("ascii"))
    if signature.layer_digests:
        for level, layer_digest in sorted(signature.layer_digests.items()):
            digest.update(f"layer:{int(level)}:{layer_digest}".encode("ascii"))
        return digest.hexdigest()
    for level in signature.levels():
        layer = signature.layers.get(level)
        if layer is None:
            digest.update(f"unloaded:{level}".encode("ascii"))
            continue
        digest.update(f"level:{level}".encode("ascii"))
        try:
            import numpy as np

            vectors = np.asarray(layer.vectors, dtype=np.float32)
            digest.update(repr(vectors.shape).encode("ascii"))
            digest.update(vectors.tobytes(order="C"))
        except Exception:
            digest.update(repr(layer.vectors).encode("utf-8"))
        for links in layer.links:
            digest.update(json.dumps([int(value) for value in links], separators=(",", ":")).encode("ascii"))
    return digest.hexdigest()
