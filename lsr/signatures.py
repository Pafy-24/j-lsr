from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .config import LAYER_FANOUT, SIGNATURE_VERSION, TOP_LAYER_CENTROIDS
from .dependencies import ensure_numpy, get_kmeans_class
from .der import DerReader, der_integer, der_octet_string, der_sequence
from .embedding import embed_texts, normalize_matrix
from .errors import RouterError
from .models import FacetSignature, LayeredSignature, RouteExample, SignatureLayer


class SignatureCodec:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    def save(self, path: str | Path, signature: LayeredSignature, model_name: str | None = None) -> Path:
        signature_path = signature_output_path(path, signature.facet)
        active_model_name = model_name or self.model_name or ""
        layers = [signature.layers[level] for level in sorted(signature.layers)]
        payload = der_sequence(
            der_integer(SIGNATURE_VERSION),
            der_octet_string(model_hash(active_model_name)),
            der_integer(signature.dimension),
            der_sequence(*(self.encode_layer(layer) for layer in layers)),
        )
        signature_path.parent.mkdir(parents=True, exist_ok=True)
        signature_path.write_bytes(payload)
        return signature_path

    def load(self, path: Path) -> LayeredSignature:
        raw = path.read_bytes()
        if not raw:
            raise RouterError(f"Empty signature file: {path}")

        if raw[:1] != b"\x30":
            return self._load_legacy_json_signature(path, raw)

        root = DerReader(raw).read_sequence()
        version = root.read_integer()
        if version != SIGNATURE_VERSION:
            raise RouterError(f"Unsupported signature version {version} in {path}")
        root.read_octet_string()
        dimension = root.read_integer()
        layers_reader = root.read_sequence()

        layers: dict[int, SignatureLayer] = {}
        while layers_reader.remaining():
            layer = self.decode_layer(layers_reader, dimension)
            layers[layer.level] = layer

        if not layers:
            raise RouterError(f"Signature does not contain any layers: {path}")
        return LayeredSignature(facet=path.stem.lower(), dimension=dimension, layers=layers)

    def encode_layer(self, layer: SignatureLayer) -> bytes:
        items = [
            der_integer(layer.level),
            der_octet_string(matrix_to_octets(layer.vectors)),
        ]
        if layer.links:
            items.append(encode_links(layer.links))
        return der_sequence(*items)

    def decode_layer(self, reader: DerReader, dimension: int) -> SignatureLayer:
        layer_reader = reader.read_sequence()
        level = layer_reader.read_integer()
        vectors = octets_to_matrix(layer_reader.read_octet_string(), dimension)
        links: list[list[int]] = []
        if layer_reader.remaining():
            links = decode_links(layer_reader)
        return SignatureLayer(level=level, vectors=vectors, links=links)

    def _load_legacy_json_signature(self, path: Path, raw: bytes) -> LayeredSignature:
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RouterError(f"Invalid signature file: {path}") from exc
        np = ensure_numpy()
        matrix = normalize_matrix(np.asarray(data, dtype=np.float32))
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise RouterError(f"JSON signature must be a non-empty 2D vector list: {path}")
        return LayeredSignature(
            facet=path.stem.lower(),
            dimension=int(matrix.shape[1]),
            layers={1: SignatureLayer(level=1, vectors=matrix, links=[])},
        )


class SignatureStore:
    def __init__(self, path: str | Path, codec: SignatureCodec | None = None):
        self.path = Path(path)
        self.codec = codec or SignatureCodec()

    def output_path(self, facet: str) -> Path:
        return signature_output_path(self.path, facet)

    def save(self, signature: LayeredSignature, model_name: str) -> Path:
        return self.codec.save(self.path, signature, model_name)

    def load_all(self, required: bool = True) -> dict[str, LayeredSignature]:
        if not self.path.exists():
            if required:
                raise RouterError(
                    f"Signature path not found: {self.path}. Run --sign first."
                )
            return {}

        if self.path.is_dir():
            files = sorted(self.path.glob("*.sig"))
        elif self.path.suffix.lower() == ".sig":
            files = [self.path]
        else:
            raise RouterError("Signature path must be a .sig file or a directory of .sig files.")

        if required and not files:
            raise RouterError(f"No .sig files found in: {self.path}")

        signatures: dict[str, LayeredSignature] = {}
        for file_path in files:
            signature = self.codec.load(file_path)
            signatures[signature.facet] = signature
        return signatures


class SignatureBuilder:
    def __init__(self, max_vectors: int):
        self.max_vectors = max_vectors

    def build_from_texts(self, facet: str, texts: list[str], model: Any) -> tuple[FacetSignature, LayeredSignature]:
        embeddings = embed_texts(model, texts)
        layered = self.build(facet, embeddings)
        summary = FacetSignature(
            facet=facet,
            num_route_texts=len(texts),
            layer_sizes={level: len(layer.vectors) for level, layer in sorted(layered.layers.items())},
        )
        return summary, layered

    def build(self, facet: str, embeddings: Any) -> LayeredSignature:
        anchors = cap_anchor_vectors(embeddings, self.max_vectors)
        n_anchors, dimension = anchors.shape
        sizes = planned_layer_sizes(n_anchors)

        layers: dict[int, SignatureLayer] = {
            len(sizes): SignatureLayer(level=len(sizes), vectors=anchors, links=[])
        }

        for level_index in range(len(sizes) - 2, -1, -1):
            level = level_index + 1
            child_layer = layers[level + 1]
            parent_count = sizes[level_index]
            parent_vectors, child_labels = cluster_embeddings(child_layer.vectors, parent_count)
            parent_links = links_from_labels(child_labels, len(parent_vectors))
            layers[level] = SignatureLayer(
                level=level,
                vectors=parent_vectors,
                links=parent_links,
            )

        return LayeredSignature(facet=facet, dimension=int(dimension), layers=layers)


def matrix_to_octets(matrix: Any) -> bytes:
    np = ensure_numpy()
    normalized = normalize_matrix(np.asarray(matrix, dtype=np.float32))
    return normalized.astype("<f4", copy=False).tobytes(order="C")


def octets_to_matrix(payload: bytes, dimension: int) -> Any:
    np = ensure_numpy()
    if dimension <= 0:
        raise RouterError("Invalid signature dimension.")
    item_size = np.dtype("<f4").itemsize
    row_size = dimension * item_size
    if len(payload) % row_size != 0:
        raise RouterError("Vector byte payload is not divisible by signature dimension.")
    matrix = np.frombuffer(payload, dtype="<f4").reshape((-1, dimension)).astype(np.float32)
    return normalize_matrix(matrix)


def encode_links(links: list[list[int]]) -> bytes:
    return der_sequence(
        *[
            der_sequence(
                der_integer(parent_id),
                der_sequence(*(der_integer(child_id) for child_id in child_ids)),
            )
            for parent_id, child_ids in enumerate(links)
        ]
    )


def decode_links(reader: DerReader) -> list[list[int]]:
    links_reader = reader.read_sequence()
    links: list[list[int]] = []
    while links_reader.remaining():
        link_reader = links_reader.read_sequence()
        parent_id = link_reader.read_integer()
        child_reader = link_reader.read_sequence()
        child_ids: list[int] = []
        while child_reader.remaining():
            child_ids.append(child_reader.read_integer())
        while len(links) <= parent_id:
            links.append([])
        links[parent_id] = child_ids
    return links


def model_hash(model_name: str) -> bytes:
    return hashlib.sha256(model_name.encode("utf-8")).digest()[:16]


def choose_centroids(embeddings: Any) -> tuple[Any, Any]:
    np = ensure_numpy()
    n_examples = len(embeddings)
    if n_examples <= 20:
        centroid = normalize_matrix(np.mean(embeddings, axis=0, keepdims=True))
        labels = np.zeros(n_examples, dtype=int)
        return centroid, labels

    kmeans_class = get_kmeans_class()

    k = min(32, max(1, int(math.sqrt(n_examples))))
    kmeans = kmeans_class(
        n_clusters=k,
        random_state=42,
        batch_size=min(1024, max(100, n_examples)),
        n_init=10,
    )
    labels = kmeans.fit_predict(embeddings)
    centroids = normalize_matrix(kmeans.cluster_centers_.astype(np.float32))
    return centroids, labels


def cluster_embeddings(embeddings: Any, n_clusters: int) -> tuple[Any, Any]:
    np = ensure_numpy()
    n_clusters = min(max(1, n_clusters), len(embeddings))
    if n_clusters == len(embeddings):
        labels = np.arange(len(embeddings), dtype=int)
        return embeddings, labels

    kmeans_class = get_kmeans_class()

    kmeans = kmeans_class(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(2048, max(256, len(embeddings))),
        n_init=10,
    )
    labels = kmeans.fit_predict(embeddings)
    return normalize_matrix(kmeans.cluster_centers_.astype(np.float32)), labels


def links_from_labels(labels: Any, parent_count: int) -> list[list[int]]:
    links: list[list[int]] = [[] for _ in range(parent_count)]
    for child_id, parent_id in enumerate(labels.tolist()):
        links[int(parent_id)].append(int(child_id))
    return links


def cap_anchor_vectors(embeddings: Any, max_vectors: int) -> Any:
    np = ensure_numpy()
    if max_vectors < 1:
        raise RouterError("--max-vectors must be at least 1.")
    if len(embeddings) <= max_vectors:
        return embeddings

    indexes = np.linspace(0, len(embeddings) - 1, num=max_vectors, dtype=int)
    return embeddings[indexes]


def planned_layer_sizes(anchor_count: int) -> list[int]:
    if anchor_count < 1:
        raise RouterError("Cannot build a signature without route vectors.")

    sizes = [min(TOP_LAYER_CENTROIDS, anchor_count)]
    while sizes[-1] < anchor_count:
        next_size = min(anchor_count, sizes[-1] * LAYER_FANOUT)
        if next_size == sizes[-1]:
            break
        sizes.append(next_size)
    return sizes


def build_layered_signature(facet: str, embeddings: Any, max_vectors: int) -> LayeredSignature:
    return SignatureBuilder(max_vectors).build(facet, embeddings)


def build_facet_signature(
    facet: str,
    texts: list[str],
    model: Any,
    max_vectors: int,
) -> tuple[FacetSignature, LayeredSignature]:
    return SignatureBuilder(max_vectors).build_from_texts(facet, texts, model)


def signature_output_path(path: str | Path, facet: str) -> Path:
    signature_path = Path(path)
    if signature_path.suffix.lower() == ".sig":
        return signature_path
    return signature_path / f"{facet.lower()}.sig"


def save_layered_signature(path: str | Path, signature: LayeredSignature, model_name: str) -> Path:
    return SignatureCodec(model_name).save(path, signature, model_name)


def load_layered_signature(path: Path) -> LayeredSignature:
    return SignatureCodec().load(path)


def load_signatures(path: str | Path, required: bool) -> dict[str, LayeredSignature]:
    return SignatureStore(path).load_all(required)


def grouped_examples(examples: list[RouteExample]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for example in examples:
        grouped.setdefault(example.facet, []).append(example.text)
    return grouped
