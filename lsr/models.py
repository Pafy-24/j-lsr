from __future__ import annotations

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
    links: list[list[int]]


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
    layer_loader: Any = None

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


@dataclass
class ProcessingUnit:
    index: int
    kind: str
    name: str
    backend: str
    device: Any
    usable: bool
    note: str = ""
