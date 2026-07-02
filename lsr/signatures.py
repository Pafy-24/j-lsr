from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .config import LAYER_FANOUT, SIGNATURE_VERSION, TOP_LAYER_CENTROIDS
from .dependencies import ensure_numpy, get_kmeans_class
from .der import DerReader, der_integer, der_octet_string, der_sequence
from .embedding import embed_texts, normalize_matrix
from .errors import RouterError
from .models import FacetSignature, LayeredSignature, RouteExample, SignatureLayer


@dataclass(frozen=True)
class SignatureLayerRef:
    path: Path
    start: int
    end: int
    level: int
    vector_count: int


def read_tlv_view(
    data: bytes,
    offset: int,
    expected_tag: int | None = None,
) -> tuple[int, int, int, int]:
    if offset >= len(data):
        raise RouterError("Unexpected end of DER data.")
    start = offset
    tag = data[offset]
    offset += 1
    if expected_tag is not None and tag != expected_tag:
        raise RouterError(f"Unexpected DER tag 0x{tag:02x}; expected 0x{expected_tag:02x}.")
    if offset >= len(data):
        raise RouterError("Missing DER length.")
    first = data[offset]
    offset += 1
    if first < 0x80:
        length = first
    else:
        n_bytes = first & 0x7F
        if n_bytes == 0 or offset + n_bytes > len(data):
            raise RouterError("Invalid DER length.")
        length = int.from_bytes(data[offset : offset + n_bytes], "big")
        offset += n_bytes
    payload_start = offset
    payload_end = payload_start + length
    if payload_end > len(data):
        raise RouterError("DER payload length exceeds file size.")
    return tag, payload_start, payload_end, payload_end


def read_exact(handle: BinaryIO, size: int) -> bytes:
    payload = handle.read(size)
    if len(payload) != size:
        raise RouterError("Unexpected end of DER data.")
    return payload


def read_tlv_header(
    handle: BinaryIO,
    offset: int,
    file_size: int,
    expected_tag: int | None = None,
) -> tuple[int, int, int, int]:
    if offset >= file_size:
        raise RouterError("Unexpected end of DER data.")
    handle.seek(offset)
    tag = read_exact(handle, 1)[0]
    if expected_tag is not None and tag != expected_tag:
        raise RouterError(f"Unexpected DER tag 0x{tag:02x}; expected 0x{expected_tag:02x}.")

    if handle.tell() >= file_size:
        raise RouterError("Missing DER length.")
    first = read_exact(handle, 1)[0]
    if first < 0x80:
        length = first
    else:
        n_bytes = first & 0x7F
        if n_bytes == 0 or handle.tell() + n_bytes > file_size:
            raise RouterError("Invalid DER length.")
        length = int.from_bytes(read_exact(handle, n_bytes), "big")

    payload_start = handle.tell()
    payload_end = payload_start + length
    if payload_end > file_size:
        raise RouterError("DER payload length exceeds file size.")
    return tag, payload_start, payload_end, payload_end


def read_tlv_payload(
    handle: BinaryIO,
    offset: int,
    file_size: int,
    expected_tag: int | None = None,
) -> tuple[int, bytes, int]:
    tag, payload_start, payload_end, next_offset = read_tlv_header(
        handle,
        offset,
        file_size,
        expected_tag,
    )
    handle.seek(payload_start)
    return tag, read_exact(handle, payload_end - payload_start), next_offset


class SignatureCodec:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    def save(self, path: str | Path, signature: LayeredSignature, model_name: str | None = None) -> Path:
        signature_path = signature_output_path(path, signature.facet)
        active_model_name = model_name or signature.model_name or self.model_name or ""
        active_model_hash = signature.model_hash or model_hash(active_model_name)
        active_normalize = signature.normalize
        active_distance_metric = signature.distance_metric or "cosine"
        layers = [signature.layers[level] for level in sorted(signature.layers)]
        payload = der_sequence(
            der_integer(SIGNATURE_VERSION),
            encode_metadata(
                model_name=active_model_name,
                model_hash_value=active_model_hash,
                dimension=signature.dimension,
                normalize=active_normalize,
                distance_metric=active_distance_metric,
            ),
            der_sequence(*(self.encode_layer(layer) for layer in layers)),
        )
        signature_path.parent.mkdir(parents=True, exist_ok=True)
        signature_path.write_bytes(payload)
        return signature_path

    def load(self, path: Path) -> LayeredSignature:
        return self.load_lazy(path)

    def load_lazy(self, path: Path) -> LayeredSignature:
        file_size = path.stat().st_size
        if file_size == 0:
            raise RouterError(f"Empty signature file: {path}")

        with path.open("rb") as handle:
            first = read_exact(handle, 1)
            if first != b"\x30":
                handle.seek(0)
                return self._load_legacy_json_signature(path, handle.read())

            _, root_start, root_end, _ = read_tlv_header(handle, 0, file_size, 0x30)
            offset = root_start
            _, version_payload, offset = read_tlv_payload(handle, offset, file_size, 0x02)
            version = int.from_bytes(version_payload, "big", signed=False)

            if version == SIGNATURE_VERSION:
                _, metadata_payload, offset = read_tlv_payload(handle, offset, file_size, 0x30)
                metadata = decode_metadata(DerReader(metadata_payload))
            elif version == 2:
                _, model_hash_value, offset = read_tlv_payload(handle, offset, file_size, 0x04)
                _, dimension_payload, offset = read_tlv_payload(handle, offset, file_size, 0x02)
                dimension_value = int.from_bytes(dimension_payload, "big", signed=False)
                metadata = {
                    "model_name": self.model_name or "",
                    "model_hash": model_hash_value,
                    "dimension": dimension_value,
                    "normalize": True,
                    "distance_metric": "cosine",
                }
            else:
                raise RouterError(f"Unsupported signature version {version} in {path}")

            dimension = int(metadata["dimension"])
            _, layers_start, layers_end, offset = read_tlv_header(handle, offset, file_size, 0x30)
            if offset != root_end:
                raise RouterError("Unexpected extra fields in signature root.")

            refs: dict[int, SignatureLayerRef] = {}
            layer_offset = layers_start
            while layer_offset < layers_end:
                layer_start = layer_offset
                _, layer_payload_start, layer_payload_end, layer_offset = read_tlv_header(
                    handle,
                    layer_offset,
                    file_size,
                    0x30,
                )
                _, level_payload, after_level = read_tlv_payload(
                    handle,
                    layer_payload_start,
                    file_size,
                    0x02,
                )
                level = int.from_bytes(level_payload, "big", signed=False)
                _, vectors_start, vectors_end, _ = read_tlv_header(
                    handle,
                    after_level,
                    file_size,
                    0x04,
                )
                vector_bytes = vectors_end - vectors_start
                row_size = dimension * 4
                if row_size <= 0 or vector_bytes % row_size != 0:
                    raise RouterError(f"Invalid vector payload in signature: {path}")
                refs[level] = SignatureLayerRef(
                    path=path,
                    start=layer_start,
                    end=layer_offset,
                    level=level,
                    vector_count=vector_bytes // row_size,
                )
                if layer_payload_end > layer_offset:
                    raise RouterError(f"Invalid layer bounds in signature: {path}")

        if not refs:
            raise RouterError(f"Signature does not contain any layers: {path}")

        def load_layer(level: int) -> SignatureLayer:
            ref = refs.get(level)
            if ref is None:
                raise RouterError(f"Signature {path} does not contain layer {level}.")
            with ref.path.open("rb") as handle:
                handle.seek(ref.start)
                payload = handle.read(ref.end - ref.start)
            return self.decode_layer(DerReader(payload), dimension)

        return LayeredSignature(
            facet=path.stem.lower(),
            dimension=dimension,
            layers={},
            model_name=str(metadata["model_name"]),
            model_hash=metadata["model_hash"],
            normalize=bool(metadata["normalize"]),
            distance_metric=str(metadata["distance_metric"]),
            layer_sizes={level: ref.vector_count for level, ref in refs.items()},
            layer_loader=load_layer,
        )

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

    def decode_legacy_metadata(self, root: DerReader) -> dict[str, Any]:
        model_hash_value = root.read_octet_string()
        dimension = root.read_integer()
        return {
            "model_name": self.model_name or "",
            "model_hash": model_hash_value,
            "dimension": dimension,
            "normalize": True,
            "distance_metric": "cosine",
        }

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
            model_name=self.model_name or "",
            model_hash=model_hash(self.model_name or ""),
            normalize=True,
            distance_metric="cosine",
            layer_sizes={1: len(matrix)},
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

        return LayeredSignature(
            facet=facet,
            dimension=int(dimension),
            layers=layers,
            normalize=True,
            distance_metric="cosine",
        )


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


def encode_metadata(
    model_name: str,
    model_hash_value: bytes,
    dimension: int,
    normalize: bool,
    distance_metric: str,
) -> bytes:
    return der_sequence(
        der_octet_string(model_name.encode("utf-8")),
        der_octet_string(model_hash_value),
        der_integer(dimension),
        der_integer(1 if normalize else 0),
        der_octet_string(distance_metric.encode("utf-8")),
    )


def decode_metadata(reader: DerReader) -> dict[str, Any]:
    model_name = reader.read_octet_string().decode("utf-8")
    model_hash_value = reader.read_octet_string()
    dimension = reader.read_integer()
    normalize = bool(reader.read_integer())
    distance_metric = reader.read_octet_string().decode("utf-8")
    if reader.remaining():
        raise RouterError("Unexpected extra fields in signature metadata.")
    if dimension <= 0:
        raise RouterError("Invalid signature metadata dimension.")
    if not distance_metric:
        raise RouterError("Signature metadata is missing distanceMetric.")
    return {
        "model_name": model_name,
        "model_hash": model_hash_value,
        "dimension": dimension,
        "normalize": normalize,
        "distance_metric": distance_metric,
    }


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
