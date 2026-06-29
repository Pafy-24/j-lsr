from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class ProcessingUnit:
    index: int
    kind: str
    name: str
    backend: str
    device: Any
    usable: bool
    note: str = ""
