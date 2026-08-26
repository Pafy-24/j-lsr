from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RouteExample:
    facet: str
    text: str


@dataclass
class FacetSignature:
    facet: str
    num_route_texts: int
    layer_sizes: dict[int, int]


@dataclass
class SignatureLayer:
    level: int
    vectors: Any
    links: Sequence[Sequence[int]]


@dataclass(frozen=True)
class CompactLinks(Sequence[Sequence[int]]):
    """Immutable CSR-style topology with four-byte parent/child offsets."""

    offsets: array
    children: array

    @classmethod
    def from_nested(cls, links: Sequence[Sequence[int]]) -> CompactLinks:
        offsets = array("I", [0])
        children = array("I")
        for child_ids in links:
            children.extend(int(child_id) for child_id in child_ids)
            offsets.append(len(children))
        return cls(offsets=offsets, children=children)

    def __len__(self) -> int:
        return max(0, len(self.offsets) - 1)

    def __getitem__(self, index: int | slice) -> Sequence[int]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        start = int(self.offsets[position])
        end = int(self.offsets[position + 1])
        return memoryview(self.children)[start:end]

    @property
    def nbytes(self) -> int:
        return (
            len(self.offsets) * self.offsets.itemsize
            + len(self.children) * self.children.itemsize
        )


@dataclass
class LayeredSignature:
    facet: str
    dimension: int
    layers: dict[int, SignatureLayer]
    model_name: str = ""
    model_hash: bytes = b""
    normalize: bool = True
    distance_metric: str = "cosine"
    layer_sizes: dict[int, int] = field(default_factory=dict)
    layer_byte_sizes: dict[int, int] = field(default_factory=dict)
    layer_loader: Any = None
    topology_cache: dict[int, CompactLinks] = field(default_factory=dict)

    def levels(self) -> list[int]:
        return sorted(self.layer_sizes or self.layers)

    def get_layer(self, level: int) -> SignatureLayer:
        if level not in self.layers:
            if self.layer_loader is None:
                raise KeyError(level)
            self.layers[level] = self.layer_loader(level)
        return self.layers[level]

    def release_layer(self, level: int) -> None:
        if self.layer_loader is not None:
            self.layers.pop(level, None)

    def release_layers_except(self, keep: set[int] | None = None) -> None:
        if self.layer_loader is None:
            return
        keep = keep or set()
        for level in list(self.layers):
            if level not in keep:
                self.layers.pop(level, None)

    def loaded_vector_bytes(self) -> int:
        total = 0
        for layer in self.layers.values():
            total += int(getattr(layer.vectors, "nbytes", 0))
        return total

    def estimated_layer_bytes(self, level: int) -> int:
        if level in self.layer_byte_sizes:
            return int(self.layer_byte_sizes[level])
        vector_count = int(self.layer_sizes.get(level, 0))
        return vector_count * int(self.dimension) * 4

    def cached_topology_bytes(self) -> int:
        return sum(topology.nbytes for topology in self.topology_cache.values())

    def routing_copy(self) -> LayeredSignature:
        """Create request-local layer storage over immutable signature metadata."""
        return LayeredSignature(
            facet=self.facet,
            dimension=self.dimension,
            # Eager/test signatures are immutable and can share their layers.
            # Lazy production signatures get an independent layer cache so
            # concurrent routes cannot release another request's vectors.
            layers=dict(self.layers) if self.layer_loader is None else {},
            model_name=self.model_name,
            model_hash=self.model_hash,
            normalize=self.normalize,
            distance_metric=self.distance_metric,
            layer_sizes=dict(self.layer_sizes),
            layer_byte_sizes=dict(self.layer_byte_sizes),
            layer_loader=self.layer_loader,
            topology_cache=self.topology_cache,
        )


@dataclass
class ProcessingUnit:
    index: int
    kind: str
    name: str
    backend: str
    device: Any
    usable: bool
    note: str = ""
    provider: str = ""
    luid: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
