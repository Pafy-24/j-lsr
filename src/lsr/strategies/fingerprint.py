from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FingerprintSpec:
    algorithm: str = "signed-feature-hash-jaccard-v1"
    bits: int = 128
    seed: str = "jester-lsr-j5"

    def __post_init__(self) -> None:
        if self.bits < 8 or self.bits > 4096:
            raise ValueError("fingerprint bits must be between 8 and 4096")

    def to_dict(self) -> dict[str, Any]:
        return {"algorithm": self.algorithm, "bits": self.bits, "seed": self.seed}


def _indices(seed: str, bit: int, dimension: int) -> tuple[int, int, int, int]:
    digest = hashlib.sha256(f"{seed}:{bit}".encode("utf-8")).digest()
    return tuple(int.from_bytes(digest[offset : offset + 4], "big") % dimension for offset in (0, 4, 8, 12))  # type: ignore[return-value]


def fingerprint(values: Any, spec: FingerprintSpec) -> int:
    array = np.asarray(values, dtype=np.float32)
    vector = np.mean(array, axis=0).reshape(-1) if array.ndim > 1 else array.reshape(-1)
    if vector.size == 0:
        raise ValueError("cannot fingerprint an empty vector")
    result = 0
    for bit in range(spec.bits):
        indices = _indices(spec.seed, bit, vector.size)
        feature = float(np.mean(vector[list(indices)]))
        if feature >= 0.0:
            result |= 1 << bit
    return result


def fingerprint_hex(values: Any, spec: FingerprintSpec) -> str:
    width = (spec.bits + 3) // 4
    return f"{fingerprint(values, spec):0{width}x}"


def jaccard_hex(left: str, right: str, bits: int) -> float:
    left_value = int(left, 16)
    right_value = int(right, 16)
    union = (left_value | right_value).bit_count()
    if union == 0:
        return 1.0
    return (left_value & right_value).bit_count() / union


def signature_fingerprint(signature: Any, spec: FingerprintSpec) -> str:
    local = signature.routing_copy()
    levels = local.levels()
    if not levels:
        raise ValueError(f"signature {signature.facet!r} has no layers")
    layer = local.get_layer(levels[0])
    vectors = np.asarray(layer.vectors, dtype=np.float32)
    prototype = np.mean(vectors, axis=0)
    return fingerprint_hex(prototype, spec)
