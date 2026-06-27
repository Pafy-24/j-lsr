from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIGNATURES = "signatures"
DEFAULT_MAX_VECTORS = 10000
SIGNATURE_VERSION = 2
TOP_LAYER_CENTROIDS = 128
LAYER_FANOUT = 10
SEARCH_L1_TOP = 3
SEARCH_NEXT_TOP = 12
SEARCH_EXACT_MAX = 512
DEEP_FACET_MULTIPLIER = 3
MAX_DEEP_FACETS = 8
COARSE_DESCEND_MIN_SCORE = 0.18
COARSE_DESCEND_MARGIN = 0.08

np = None
SentenceTransformer = None
MiniBatchKMeans = None


class LazyConsole:
    def __init__(self, stderr: bool = False):
        self.stderr = stderr
        self._console = None

    def print(self, *args: Any, **kwargs: Any) -> None:
        if self._console is None:
            from rich.console import Console

            self._console = Console(stderr=self.stderr)
        self._console.print(*args, **kwargs)


console = LazyConsole()
error_console = LazyConsole(stderr=True)


class RouterError(Exception):
    """User-facing CLI error."""


def ensure_numpy() -> Any:
    global np
    if np is None:
        import numpy as numpy_module

        np = numpy_module
    return np


def get_sentence_transformer_class() -> Any:
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as sentence_transformer_class
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise RouterError(
                "Missing dependency: sentence-transformers. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        SentenceTransformer = sentence_transformer_class
    return SentenceTransformer


def get_kmeans_class() -> Any:
    global MiniBatchKMeans
    if MiniBatchKMeans is None:
        try:
            from sklearn.cluster import MiniBatchKMeans as minibatch_kmeans_class
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise RouterError(
                "Missing dependency: scikit-learn. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        MiniBatchKMeans = minibatch_kmeans_class
    return MiniBatchKMeans


def make_table(*args: Any, **kwargs: Any) -> Any:
    from rich.table import Table

    return Table(*args, **kwargs)


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


def unit_exists(units: list[ProcessingUnit], name: str) -> bool:
    needle = name.lower()
    return any(needle and needle in unit.name.lower() for unit in units)


def detect_processing_units() -> list[ProcessingUnit]:
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

    try:
        import torch_directml
    except ModuleNotFoundError:
        torch_directml = None
    if torch_directml is not None:
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

    try:
        import torch_xla.core.xla_model as xm
    except ModuleNotFoundError:
        xm = None
    if xm is not None:
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

    for gpu in windows_gpu_devices():
        name = str(gpu.get("Name") or "").strip()
        if not name or unit_exists(units, name):
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
        if not name or unit_exists(units, name):
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

    return units


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


def select_processing_unit(index: int | None) -> ProcessingUnit | None:
    if index is None:
        return None
    units = detect_processing_units()
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
    vectors: np.ndarray
    links: list[list[int]]


@dataclass
class LayeredSignature:
    facet: str
    dimension: int
    layers: dict[int, SignatureLayer]


@dataclass
class ProcessingUnit:
    index: int
    kind: str
    name: str
    backend: str
    device: Any
    usable: bool
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate semantic facet signatures and route prompts to facets."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Raw text to sign or route. Quote multi-word prompts.",
    )
    parser.add_argument(
        "--sign",
        action="store_true",
        help="Generate or update signatures instead of routing.",
    )
    parser.add_argument(
        "-in",
        "--input",
        dest="input_path",
        help="Input file path. In sign mode this must be a JSON dataset.",
    )
    parser.add_argument(
        "--facet",
        help="Facet name. Optional when it can be inferred from a faucet path.",
    )
    parser.add_argument(
        "--signatures",
        default=DEFAULT_SIGNATURES,
        help=(
            "Signature file or directory. Facet signatures are stored as "
            "[facet].sig ASN.1 DER files. Default: signatures"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of facets to return in routing mode. Default: 3",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Route multiple prompts from -in JSON/JSONL or stdin in one process.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a Rich table.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow model downloads/checks. By default the router uses local model files only.",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=DEFAULT_MAX_VECTORS,
        help=(
            "Maximum vectors to store per faucet signature. "
            f"Default: {DEFAULT_MAX_VECTORS}"
        ),
    )
    parser.add_argument(
        "--PUs",
        action="store_true",
        help="List available processing units and exit.",
    )
    parser.add_argument(
        "--PU",
        type=int,
        help="Processing unit index to use for embedding operations. See --PUs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra details.",
    )
    return parser.parse_args()


def read_text_arg(parts: list[str]) -> str:
    return " ".join(parts).strip()


def read_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise RouterError(f"Input file not found: {file_path}")
    if not file_path.is_file():
        raise RouterError(f"Input path is not a file: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


def read_stdin_if_available() -> str:
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read().strip()


def extract_prompt_from_item(item: Any, index: int) -> str:
    if isinstance(item, str):
        prompt = item.strip()
    elif isinstance(item, dict):
        prompt = str(
            item.get("input")
            or item.get("prompt")
            or item.get("text")
            or ""
        ).strip()
    else:
        prompt = ""

    if not prompt:
        raise RouterError(f"Batch item {index} has no prompt text.")
    return prompt


def load_batch_prompts(raw: str) -> list[str]:
    if not raw.strip():
        raise RouterError("Missing batch input.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        prompts: list[str] = []
        for index, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = line
            prompts.append(extract_prompt_from_item(item, index))
        if not prompts:
            raise RouterError("Batch input must be JSON, JSONL, or non-empty lines.")
        return prompts

    if not isinstance(data, list):
        raise RouterError("Batch input must be a JSON list or JSONL stream.")
    return [extract_prompt_from_item(item, index) for index, item in enumerate(data, start=1)]


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    raw = read_file(path)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RouterError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise RouterError("Dataset must be a JSON list of objects.")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise RouterError(f"Dataset item {index} must be an object.")
    return data


def tags_to_text(tags: Any) -> str:
    if tags is None:
        return ""
    if isinstance(tags, list):
        return " ".join(str(tag) for tag in tags if str(tag).strip())
    return str(tags)


def compact_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def truncate_text(text: str, max_chars: int) -> str:
    compact = compact_whitespace(text)
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rsplit(" ", 1)[0].strip()


TECH_TERM_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Z0-9]{2,}|[a-zA-Z]+-[a-zA-Z0-9-]+|nonce|signature|"
    r"private\s+key|public\s+key|elliptic\s+curve|hash|cipher|oracle|"
    r"lattice|quantum|POVM|RSA|DSA|ECDSA|AES|CBC|CTR|GCM|MAC|HMAC|PRF|PRG)\b",
    re.IGNORECASE,
)


def technical_terms(*texts: str, limit: int = 32) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for match in TECH_TERM_PATTERN.finditer(text or ""):
            term = compact_whitespace(match.group(0))
            key = term.lower()
            if key not in seen:
                seen.add(key)
                terms.append(term)
            if len(terms) >= limit:
                return " ".join(terms)
    return " ".join(terms)


def extract_route_text(item: dict[str, Any]) -> str:
    # Prefer explicit routing text. Fall back to prompt-side fields and only use
    # output as a last resort so signatures learn the request, not the answer.
    text = str(item.get("text") or "").strip()
    if text:
        return text

    prompt_parts = [
        str(item.get("instruction") or "").strip(),
        str(item.get("input") or "").strip(),
        tags_to_text(item.get("tags")).strip(),
    ]
    prompt_text = " ".join(part for part in prompt_parts if part).strip()
    if prompt_text:
        return prompt_text

    return str(item.get("output") or "").strip()


def infer_facet_from_paths(
    input_path: str | None, signatures_path: str | None, explicit_facet: str | None
) -> str:
    if explicit_facet and explicit_facet.strip():
        return explicit_facet.strip().lower()

    for raw_path in (input_path, signatures_path):
        if not raw_path:
            continue
        path = Path(raw_path)
        parts = list(path.parts)
        lowered = [part.lower() for part in parts]
        if "faucets" in lowered:
            index = lowered.index("faucets")
            if index + 1 < len(parts):
                return parts[index + 1].strip().lower()
        if path.suffix.lower() == ".sig" and path.stem:
            return path.stem.strip().lower()

    raise RouterError(
        "Dataset items do not define a facet and no facet could be inferred. "
        "Pass --facet crypto or use a path under faucets\\Crypto."
    )


def route_texts_from_exam_item(item: dict[str, Any]) -> list[str]:
    topic = str(item.get("topic") or "").strip()
    describe = str(item.get("describe") or "").strip()
    item_tags = tags_to_text(item.get("tags")).strip()

    questions = item.get("questions")
    if not isinstance(questions, list):
        return []

    route_texts: list[str] = []
    for question_item in questions:
        if not isinstance(question_item, dict):
            continue
        question = str(question_item.get("question") or "").strip()
        if not question:
            continue

        q_tags = tags_to_text(question_item.get("tags")).strip()
        prereq = tags_to_text(question_item.get("prerequisite_knowledge")).strip()
        difficulty = str(question_item.get("difficulty") or "").strip()
        problem_type = str(question_item.get("problem_type") or "").strip()
        tags = " ".join(part for part in [item_tags, q_tags, prereq] if part)
        solution_terms = technical_terms(
            question,
            str(question_item.get("solution") or ""),
            tags,
            limit=28,
        )

        # Keep the actual user-like request first. Long descriptions can drown out
        # the discriminative terms once the transformer tokenizer truncates input.
        focused = " ".join(
            part
            for part in [
                f"Question: {truncate_text(question, 900)}",
                f"Topic: {truncate_text(topic, 180)}" if topic else "",
                f"Tags: {truncate_text(tags, 240)}" if tags else "",
                f"Keywords: {solution_terms}" if solution_terms else "",
                difficulty,
                problem_type,
            ]
            if part
        )
        route_texts.append(focused)

        if topic or tags:
            route_texts.append(
                " ".join(
                    part
                    for part in [
                        f"Topic: {truncate_text(topic, 220)}" if topic else "",
                        f"Tags: {truncate_text(tags, 320)}" if tags else "",
                        f"Keywords: {solution_terms}" if solution_terms else "",
                        f"Question: {truncate_text(question, 360)}",
                    ]
                    if part
                )
            )

        if describe:
            route_texts.append(
                " ".join(
                    part
                    for part in [
                        f"Question: {truncate_text(question, 500)}",
                        f"Context: {truncate_text(describe, 420)}",
                        f"Keywords: {solution_terms}" if solution_terms else "",
                    ]
                    if part
                )
            )

    return route_texts


def examples_from_dataset(
    dataset: list[dict[str, Any]], default_facet: str | None = None
) -> list[RouteExample]:
    examples: list[RouteExample] = []
    for index, item in enumerate(dataset):
        facet = str(item.get("facet") or default_facet or "").strip().lower()
        if not facet:
            raise RouterError(f"Dataset item {index} is missing required field: facet")

        route_texts = route_texts_from_exam_item(item) or [extract_route_text(item)]
        route_texts = list(dict.fromkeys(text for text in route_texts if text))
        if not route_texts:
            raise RouterError(
                f"Dataset item {index} for facet '{facet}' has no routable text."
            )
        examples.extend(RouteExample(facet=facet, text=text) for text in route_texts)
    return examples


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    ensure_numpy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    ensure_numpy()
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_model(
    model_name: str, pu_index: int | None = None, online: bool = False
) -> SentenceTransformer:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    unit = select_processing_unit(pu_index)
    sentence_transformer_class = get_sentence_transformer_class()
    kwargs = {"local_files_only": not online}
    if unit is None:
        return sentence_transformer_class(model_name, **kwargs)
    return sentence_transformer_class(model_name, device=unit.device, **kwargs)


def der_length(length: int) -> bytes:
    if length < 0x80:
        return bytes([length])
    raw = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(raw)]) + raw


def der_tlv(tag: int, payload: bytes) -> bytes:
    return bytes([tag]) + der_length(len(payload)) + payload


def der_sequence(*items: bytes) -> bytes:
    return der_tlv(0x30, b"".join(items))


def der_integer(value: int) -> bytes:
    if value < 0:
        raise RouterError("DER encoder only supports non-negative integers.")
    raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
    if raw[0] & 0x80:
        raw = b"\x00" + raw
    return der_tlv(0x02, raw)


def der_octet_string(payload: bytes) -> bytes:
    return der_tlv(0x04, payload)


class DerReader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def remaining(self) -> int:
        return len(self.data) - self.offset

    def read_tlv(self, expected_tag: int | None = None) -> tuple[int, bytes]:
        if self.offset >= len(self.data):
            raise RouterError("Unexpected end of DER data.")
        tag = self.data[self.offset]
        self.offset += 1
        if expected_tag is not None and tag != expected_tag:
            raise RouterError(f"Unexpected DER tag 0x{tag:02x}; expected 0x{expected_tag:02x}.")

        if self.offset >= len(self.data):
            raise RouterError("Missing DER length.")
        first = self.data[self.offset]
        self.offset += 1
        if first < 0x80:
            length = first
        else:
            n_bytes = first & 0x7F
            if n_bytes == 0 or self.offset + n_bytes > len(self.data):
                raise RouterError("Invalid DER length.")
            length = int.from_bytes(self.data[self.offset : self.offset + n_bytes], "big")
            self.offset += n_bytes

        if self.offset + length > len(self.data):
            raise RouterError("DER payload length exceeds file size.")
        payload = self.data[self.offset : self.offset + length]
        self.offset += length
        return tag, payload

    def read_sequence(self) -> "DerReader":
        _, payload = self.read_tlv(0x30)
        return DerReader(payload)

    def read_integer(self) -> int:
        _, payload = self.read_tlv(0x02)
        if not payload:
            raise RouterError("Empty DER INTEGER.")
        return int.from_bytes(payload, "big", signed=False)

    def read_octet_string(self) -> bytes:
        _, payload = self.read_tlv(0x04)
        return payload


def matrix_to_octets(matrix: np.ndarray) -> bytes:
    ensure_numpy()
    normalized = normalize_matrix(np.asarray(matrix, dtype=np.float32))
    return normalized.astype("<f4", copy=False).tobytes(order="C")


def octets_to_matrix(payload: bytes, dimension: int) -> np.ndarray:
    ensure_numpy()
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


def encode_layer(layer: SignatureLayer) -> bytes:
    items = [
        der_integer(layer.level),
        der_octet_string(matrix_to_octets(layer.vectors)),
    ]
    if layer.links:
        items.append(encode_links(layer.links))
    return der_sequence(*items)


def save_layered_signature(path: str | Path, signature: LayeredSignature, model_name: str) -> Path:
    signature_path = signature_output_path(path, signature.facet)
    layers = [signature.layers[level] for level in sorted(signature.layers)]
    payload = der_sequence(
        der_integer(SIGNATURE_VERSION),
        der_octet_string(model_hash(model_name)),
        der_integer(signature.dimension),
        der_sequence(*(encode_layer(layer) for layer in layers)),
    )
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.write_bytes(payload)
    return signature_path


def decode_layer(reader: DerReader, dimension: int) -> SignatureLayer:
    layer_reader = reader.read_sequence()
    level = layer_reader.read_integer()
    vectors = octets_to_matrix(layer_reader.read_octet_string(), dimension)
    links: list[list[int]] = []
    if layer_reader.remaining():
        links = decode_links(layer_reader)
    return SignatureLayer(level=level, vectors=vectors, links=links)


def load_layered_signature(path: Path) -> LayeredSignature:
    raw = path.read_bytes()
    if not raw:
        raise RouterError(f"Empty signature file: {path}")

    # Backward-compatible fallback for old prototype JSON .sig files.
    if raw[:1] != b"\x30":
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RouterError(f"Invalid signature file: {path}") from exc
        ensure_numpy()
        matrix = normalize_matrix(np.asarray(data, dtype=np.float32))
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise RouterError(f"JSON signature must be a non-empty 2D vector list: {path}")
        return LayeredSignature(
            facet=path.stem.lower(),
            dimension=int(matrix.shape[1]),
            layers={1: SignatureLayer(level=1, vectors=matrix, links=[])},
        )

    root = DerReader(raw).read_sequence()
    version = root.read_integer()
    if version != SIGNATURE_VERSION:
        raise RouterError(f"Unsupported signature version {version} in {path}")
    root.read_octet_string()  # model hash; currently informational.
    dimension = root.read_integer()
    layers_reader = root.read_sequence()

    layers: dict[int, SignatureLayer] = {}
    while layers_reader.remaining():
        layer = decode_layer(layers_reader, dimension)
        layers[layer.level] = layer

    if not layers:
        raise RouterError(f"Signature does not contain any layers: {path}")
    return LayeredSignature(facet=path.stem.lower(), dimension=dimension, layers=layers)


def choose_centroids(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ensure_numpy()
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


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    ensure_numpy()
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


def links_from_labels(labels: np.ndarray, parent_count: int) -> list[list[int]]:
    ensure_numpy()
    links: list[list[int]] = [[] for _ in range(parent_count)]
    for child_id, parent_id in enumerate(labels.tolist()):
        links[int(parent_id)].append(int(child_id))
    return links


def cap_anchor_vectors(embeddings: np.ndarray, max_vectors: int) -> np.ndarray:
    ensure_numpy()
    if max_vectors < 1:
        raise RouterError("--max-vectors must be at least 1.")
    if len(embeddings) <= max_vectors:
        return embeddings

    # Deterministic downsample for very large faucets. This keeps .sig bounded
    # without adding another expensive clustering pass before the real index.
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


def build_layered_signature(
    facet: str,
    embeddings: np.ndarray,
    max_vectors: int,
) -> LayeredSignature:
    anchors = cap_anchor_vectors(embeddings, max_vectors)
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


def build_facet_signature(
    facet: str,
    texts: list[str],
    model: SentenceTransformer,
    max_vectors: int,
) -> tuple[FacetSignature, LayeredSignature]:
    embeddings = embed_texts(model, texts)
    layered = build_layered_signature(facet, embeddings, max_vectors)
    summary = FacetSignature(
        facet=facet,
        num_route_texts=len(texts),
        layer_sizes={level: len(layer.vectors) for level, layer in sorted(layered.layers.items())},
    )
    return summary, layered


def signature_output_path(path: str | Path, facet: str) -> Path:
    signature_path = Path(path)
    if signature_path.suffix.lower() == ".sig":
        return signature_path
    return signature_path / f"{facet.lower()}.sig"


def load_signatures(path: str | Path, required: bool) -> dict[str, LayeredSignature]:
    signature_path = Path(path)
    if not signature_path.exists():
        if required:
            raise RouterError(
                f"Signature path not found: {signature_path}. Run --sign first."
            )
        return {}

    files: list[Path]
    if signature_path.is_dir():
        files = sorted(signature_path.glob("*.sig"))
    elif signature_path.suffix.lower() == ".sig":
        files = [signature_path]
    else:
        raise RouterError("Signature path must be a .sig file or a directory of .sig files.")

    if required and not files:
        raise RouterError(f"No .sig files found in: {signature_path}")

    signatures: dict[str, LayeredSignature] = {}
    for file_path in files:
        signature = load_layered_signature(file_path)
        signatures[signature.facet] = signature
    return signatures


def grouped_examples(examples: list[RouteExample]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for example in examples:
        grouped.setdefault(example.facet, []).append(example.text)
    return grouped


def print_signature_summary(signatures: list[FacetSignature]) -> None:
    table = make_table(title="Signature Summary")
    table.add_column("Facet")
    table.add_column("Route Texts", justify="right")
    table.add_column("Layers")
    table.add_column("Output")

    for signature in sorted(signatures, key=lambda item: item.facet):
        layer_text = ", ".join(
            f"L{level}={size}" for level, size in sorted(signature.layer_sizes.items())
        )
        table.add_row(
            signature.facet,
            str(signature.num_route_texts),
            layer_text,
            f"{signature.facet}.sig",
        )
    console.print(table)


def sign_dataset(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.input_path)
    default_facet = infer_facet_from_paths(args.input_path, args.signatures, args.facet)
    grouped = grouped_examples(examples_from_dataset(dataset, default_facet))
    model = load_model(args.model, args.PU, args.online)

    signatures: list[FacetSignature] = []
    for facet, texts in sorted(grouped.items()):
        facet_signature, layered_signature = build_facet_signature(
            facet, texts, model, args.max_vectors
        )
        output_path = save_layered_signature(args.signatures, layered_signature, args.model)
        signatures.append(facet_signature)
        if args.debug:
            layer_text = ", ".join(
                f"L{level}={size}" for level, size in sorted(facet_signature.layer_sizes.items())
            )
            console.print(
                f"[dim]{facet}: {len(texts)} route texts, "
                f"{layer_text} -> {output_path}[/dim]"
            )

    print_signature_summary(signatures)
    console.print(f"Saved ASN.1 DER signatures under [bold]{args.signatures}[/bold]")


def sign_raw_text(args: argparse.Namespace) -> None:
    text = read_text_arg(args.text)
    if not text:
        raise RouterError("Missing text to sign. Provide raw text or use -in dataset.json.")
    facet = infer_facet_from_paths(None, args.signatures, args.facet)
    model = load_model(args.model, args.PU, args.online)
    facet_signature, layered_signature = build_facet_signature(
        facet, [text], model, args.max_vectors
    )
    output_path = save_layered_signature(args.signatures, layered_signature, args.model)
    print_signature_summary([facet_signature])
    console.print(f"Saved vector signature to [bold]{output_path}[/bold]")


QUERY_EXPANSIONS = {
    "ecdsa": "elliptic curve digital signature algorithm repeated nonce k private key recovery r s signature",
    "dsa": "digital signature algorithm nonce k private key recovery signature",
    "rsa": "rsa public key private key signature verification encryption modulus exponent",
    "nonce": "reused repeated nonce ephemeral secret k signature private key recovery",
    "private key": "secret key recovery cryptographic attack",
    "signature": "digital signature verification signing public key private key",
    "aes": "advanced encryption standard block cipher symmetric encryption",
}


def expand_query(prompt: str) -> str:
    lowered = prompt.lower()
    expansions = [
        expansion
        for trigger, expansion in QUERY_EXPANSIONS.items()
        if trigger in lowered
    ]
    if not expansions:
        return prompt
    return f"{prompt}\n\nExpanded technical context: {' '.join(expansions)}"


def child_ids(links: list[list[int]], parent_ids: list[int], max_items: int | None = None) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for parent_id in parent_ids:
        if 0 <= parent_id < len(links):
            children = links[parent_id]
        else:
            children = []
        for child_id in children:
            if child_id not in seen:
                seen.add(child_id)
                ids.append(child_id)
                if max_items is not None and len(ids) >= max_items:
                    return ids
    return ids


def top_indices(scores: np.ndarray, limit: int) -> list[int]:
    ensure_numpy()
    if len(scores) == 0:
        return []
    limit = min(max(1, limit), len(scores))
    if limit == len(scores):
        return np.argsort(scores)[::-1].astype(int).tolist()
    partial = np.argpartition(scores, -limit)[-limit:]
    ordered = partial[np.argsort(scores[partial])[::-1]]
    return ordered.astype(int).tolist()


def coarse_search_signature(
    signature: LayeredSignature, prompt_embedding: np.ndarray
) -> dict[str, Any]:
    first_level = sorted(signature.layers)[0]
    layer = signature.layers[first_level]
    similarities = layer.vectors @ prompt_embedding
    top = top_indices(similarities, min(5, len(similarities)))
    best_index = top[0]
    best_score = float(similarities[best_index])
    avg_top_score = float(np.mean(similarities[top]))
    return {
        "facet": signature.facet,
        "score": 0.7 * best_score + 0.3 * avg_top_score,
        "best_vector_score": best_score,
        "cluster_score": best_score,
        "vector_index": int(best_index),
        "exact_count": 0,
        "mode": f"L{first_level} coarse",
    }


def should_descend(
    coarse_result: dict[str, Any], best_coarse_score: float, forced: bool
) -> bool:
    if forced:
        return True
    score = float(coarse_result["score"])
    return (
        score >= COARSE_DESCEND_MIN_SCORE
        and score >= best_coarse_score - COARSE_DESCEND_MARGIN
    )


def select_deep_facets(
    coarse_results: list[dict[str, Any]], top_k: int
) -> set[str]:
    ordered = sorted(coarse_results, key=lambda item: item["score"], reverse=True)
    if not ordered:
        return set()

    max_deep = min(MAX_DEEP_FACETS, len(ordered))
    forced_count = min(max(1, top_k), max_deep)
    soft_limit = min(max_deep, max(forced_count, top_k * DEEP_FACET_MULTIPLIER))
    best_score = float(ordered[0]["score"])

    selected: set[str] = set()
    for index, result in enumerate(ordered):
        if len(selected) >= soft_limit:
            break
        forced = index < forced_count
        if should_descend(result, best_score, forced):
            selected.add(str(result["facet"]))

    return selected


def search_signature(
    signature: LayeredSignature, prompt_embedding: np.ndarray, args: argparse.Namespace
) -> dict[str, Any]:
    levels = sorted(signature.layers)
    last_level = levels[-1]
    last_layer = signature.layers[last_level]

    if len(levels) == 1:
        similarities = last_layer.vectors @ prompt_embedding
        top = top_indices(similarities, min(5, len(similarities)))
        best_index = top[0]
        best_score = float(similarities[best_index])
        avg_top_score = float(np.mean(similarities[top]))
        return {
            "facet": signature.facet,
            "score": 0.7 * best_score + 0.3 * avg_top_score,
            "best_vector_score": best_score,
            "cluster_score": 0.0,
            "vector_index": int(best_index),
            "exact_count": len(similarities),
            "mode": "exact",
        }

    candidate_ids: list[int] | None = None
    best_cluster_score = 0.0
    for position, level in enumerate(levels[:-1]):
        layer = signature.layers[level]
        if candidate_ids is None:
            local_vectors = layer.vectors
            source_ids = list(range(len(layer.vectors)))
        else:
            local_vectors = layer.vectors[candidate_ids]
            source_ids = candidate_ids

        scores = local_vectors @ prompt_embedding
        limit = SEARCH_L1_TOP if position == 0 else SEARCH_NEXT_TOP
        top_local = top_indices(scores, limit)
        top_source_ids = [source_ids[index] for index in top_local]
        if top_local:
            best_cluster_score = float(scores[top_local[0]])

        next_limit = SEARCH_EXACT_MAX if levels[position + 1] == last_level else None
        candidate_ids = child_ids(layer.links, top_source_ids, next_limit)
        if not candidate_ids:
            next_layer = signature.layers[levels[position + 1]]
            fallback_limit = next_limit or len(next_layer.vectors)
            candidate_ids = list(range(min(len(next_layer.vectors), fallback_limit)))

    l3_candidates = candidate_ids or list(range(min(len(last_layer.vectors), SEARCH_EXACT_MAX)))
    l3_matrix = last_layer.vectors[l3_candidates]
    l3_scores = l3_matrix @ prompt_embedding
    top_l3_local = top_indices(l3_scores, min(5, len(l3_scores)))
    best_local = top_l3_local[0]
    best_index = int(l3_candidates[best_local])
    best_anchor = float(l3_scores[best_local])
    avg_top_score = float(np.mean(l3_scores[top_l3_local]))
    final_score = 0.6 * best_anchor + 0.25 * avg_top_score + 0.15 * best_cluster_score

    return {
        "facet": signature.facet,
        "score": final_score,
        "best_vector_score": best_anchor,
        "cluster_score": best_cluster_score,
        "vector_index": best_index,
        "exact_count": len(l3_candidates),
        "mode": "deep",
    }


def route_embedding(
    prompt: str,
    routed_prompt: str,
    prompt_embedding: Any,
    signatures: dict[str, LayeredSignature],
    args: argparse.Namespace,
) -> dict[str, Any]:
    coarse_results = [
        coarse_search_signature(signature, prompt_embedding)
        for signature in signatures.values()
    ]
    deep_facets = select_deep_facets(coarse_results, args.top_k)

    results: list[dict[str, Any]] = []
    coarse_by_facet = {str(result["facet"]): result for result in coarse_results}
    for facet, signature in signatures.items():
        if facet in deep_facets:
            results.append(search_signature(signature, prompt_embedding, args))
        else:
            results.append(coarse_by_facet[facet])

    if not results:
        raise RouterError("No usable vectors found in signature file.")

    results.sort(key=lambda item: item["score"], reverse=True)
    return {
        "input": prompt,
        "expanded": routed_prompt if routed_prompt != prompt else None,
        "results": results[: args.top_k],
        "compared_facets": len(results),
        "deep_facets": sorted(deep_facets),
    }


def json_ready_route(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": route["input"],
        "expanded": route["expanded"],
        "compared_facets": route["compared_facets"],
        "deep_facets": route["deep_facets"],
        "results": [
            {
                "rank": index,
                "facet": result["facet"],
                "score": result["score"],
                "best_vector_score": result["best_vector_score"],
                "cluster_score": result["cluster_score"],
                "exact_count": result["exact_count"],
                "mode": result["mode"],
                "vector_index": result["vector_index"],
            }
            for index, result in enumerate(route["results"], start=1)
        ],
    }


def print_route_table(route: dict[str, Any], args: argparse.Namespace) -> None:
    top_results = route["results"]

    table = make_table(title="Routing Results")
    table.add_column("Rank", justify="right")
    table.add_column("Facet")
    table.add_column("Score", justify="right")
    table.add_column("Best Vector", justify="right")
    table.add_column("Cluster", justify="right")
    table.add_column("Exact", justify="right")
    table.add_column("Mode")
    table.add_column("Best Match")
    for rank, result in enumerate(top_results, start=1):
        table.add_row(
            str(rank),
            result["facet"],
            f"{result['score']:.4f}",
            f"{result['best_vector_score']:.4f}",
            f"{result['cluster_score']:.4f}",
            str(result["exact_count"]),
            str(result["mode"]),
            f"vector #{result['vector_index']}",
        )
    console.print(table)

    if args.debug:
        prompt = route["input"]
        routed_prompt = route["expanded"] or prompt
        console.print(f"[dim]Prompt: {prompt}[/dim]")
        if routed_prompt != prompt:
            console.print(f"[dim]Expanded: {routed_prompt}[/dim]")
        console.print(f"[dim]Model: {args.model}[/dim]")
        if args.PU is not None:
            unit = select_processing_unit(args.PU)
            console.print(
                f"[dim]PU: #{unit.index} {unit.kind} "
                f"{unit.backend} {unit.device}[/dim]"
            )
        console.print(f"[dim]Compared {route['compared_facets']} facets.[/dim]")
        console.print(
            f"[dim]Deep searched {len(route['deep_facets'])} facets: "
            f"{', '.join(route['deep_facets'])}[/dim]"
        )


def route_prompt(args: argparse.Namespace) -> None:
    signatures = load_signatures(args.signatures, required=True)
    if not signatures:
        raise RouterError(f"No facets found in signature path: {args.signatures}")

    if args.input_path:
        prompt = read_file(args.input_path)
    else:
        prompt = read_text_arg(args.text) or read_stdin_if_available()

    if not prompt:
        raise RouterError("Missing prompt. Provide -in input.txt, raw text, or stdin.")
    if args.top_k < 1:
        raise RouterError("--top-k must be at least 1.")

    model = load_model(args.model, args.PU, args.online)
    routed_prompt = expand_query(prompt)
    prompt_embedding = embed_texts(model, [routed_prompt])[0]
    route = route_embedding(prompt, routed_prompt, prompt_embedding, signatures, args)

    if args.json:
        print(json.dumps(json_ready_route(route), ensure_ascii=False))
    else:
        print_route_table(route, args)


def route_batch(args: argparse.Namespace) -> None:
    signatures = load_signatures(args.signatures, required=True)
    if not signatures:
        raise RouterError(f"No facets found in signature path: {args.signatures}")
    if args.top_k < 1:
        raise RouterError("--top-k must be at least 1.")

    raw = read_file(args.input_path) if args.input_path else read_stdin_if_available()
    prompts = load_batch_prompts(raw)
    routed_prompts = [expand_query(prompt) for prompt in prompts]

    model = load_model(args.model, args.PU, args.online)
    embeddings = embed_texts(model, routed_prompts)
    routes = [
        route_embedding(prompt, routed_prompt, embeddings[index], signatures, args)
        for index, (prompt, routed_prompt) in enumerate(zip(prompts, routed_prompts))
    ]
    print(json.dumps([json_ready_route(route) for route in routes], ensure_ascii=False))


def main() -> int:
    args = parse_args()
    try:
        if args.PUs:
            print_processing_units()
            return 0
        if args.sign:
            if args.input_path:
                sign_dataset(args)
            else:
                sign_raw_text(args)
        elif args.batch:
            route_batch(args)
        else:
            route_prompt(args)
        return 0
    except RouterError as exc:
        error_console.print(f"[red]Error:[/red] {exc}")
        return 2
    except KeyboardInterrupt:
        error_console.print("\n[red]Interrupted.[/red]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
