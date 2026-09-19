from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FingerprintSpec:
    algorithm: str = "signed-feature-hash-jaccard-v2"
    bits: int = 128
    seed: str = "jester-lsr-j5"

    def __post_init__(self) -> None:
        if self.algorithm != "signed-feature-hash-jaccard-v2":
            raise ValueError("unsupported fingerprint algorithm")
        if isinstance(self.bits, bool) or self.bits < 8 or self.bits > 4096 or self.bits % 8:
            raise ValueError("fingerprint bits must be between 8 and 4096")
        if not self.seed:
            raise ValueError("fingerprint seed is required")

    def to_dict(self) -> dict[str, Any]:
        return {"algorithm": self.algorithm, "bits": self.bits, "seed": self.seed}


def _indices(seed: str, bit: int, dimension: int) -> tuple[int, int, int, int]:
    digest = hashlib.sha256(f"{seed}:{bit}".encode("utf-8")).digest()
    return tuple(int.from_bytes(digest[offset : offset + 4], "big") % dimension for offset in (0, 4, 8, 12))  # type: ignore[return-value]


def fingerprint(values: Any, spec: FingerprintSpec, *, expected_dimension: int | None = None) -> int:
    array = np.asarray(values, dtype=np.float32)
    vector = np.mean(array, axis=0).reshape(-1) if array.ndim > 1 else array.reshape(-1)
    if vector.size == 0:
        raise ValueError("cannot fingerprint an empty vector")
    if not np.isfinite(vector).all():
        raise ValueError("cannot fingerprint non-finite values")
    if expected_dimension is not None and vector.size != int(expected_dimension):
        raise ValueError(f"fingerprint dimension mismatch: expected {expected_dimension}, got {vector.size}")
    if float(np.linalg.norm(vector)) <= 0.0:
        raise ValueError("cannot fingerprint a zero-norm vector")
    result = 0
    for bit in range(spec.bits):
        digest = hashlib.sha256(f"{spec.seed}:{bit}:signed".encode("utf-8")).digest()
        indices = _indices(spec.seed, bit, vector.size)
        signs = tuple(1.0 if digest[offset] & 1 else -1.0 for offset in (16, 17, 18, 19))
        feature = float(np.mean([vector[index] * sign for index, sign in zip(indices, signs)]))
        if feature >= 0.0:
            result |= 1 << bit
    return result


def fingerprint_hex(values: Any, spec: FingerprintSpec, *, expected_dimension: int | None = None) -> str:
    width = (spec.bits + 3) // 4
    return f"{fingerprint(values, spec, expected_dimension=expected_dimension):0{width}x}"


def jaccard_hex(left: str, right: str, bits: int) -> float:
    if isinstance(bits, bool) or bits < 8 or bits > 4096 or bits % 8:
        raise ValueError("fingerprint bits must be between 8 and 4096")
    width = bits // 4
    if not isinstance(left, str) or not isinstance(right, str):
        raise ValueError("fingerprints must be hexadecimal strings")
    if len(left) != width or len(right) != width:
        raise ValueError("fingerprint length does not match configured bits")
    try:
        left_value = int(left, 16)
        right_value = int(right, 16)
    except ValueError as error:
        raise ValueError("fingerprints must be hexadecimal strings") from error
    union = (left_value | right_value).bit_count()
    if union == 0:
        return 1.0 if left_value == right_value == 0 else 0.0
    return (left_value & right_value).bit_count() / union


def signature_fingerprint(signature: Any, spec: FingerprintSpec) -> str:
    local = signature.routing_copy()
    levels = local.levels()
    if not levels:
        raise ValueError(f"signature {signature.facet!r} has no layers")
    layer = local.get_layer(levels[0])
    vectors = np.asarray(layer.vectors, dtype=np.float32)
    prototype = np.mean(vectors, axis=0)
    return fingerprint_hex(prototype, spec, expected_dimension=int(signature.dimension))
