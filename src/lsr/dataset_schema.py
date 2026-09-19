from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset import examples_from_dataset, infer_facet_from_paths
from .errors import RouterError
from .io_utils import load_dataset, read_file
from .models import RouteExample
from .routing_context import facet_record_to_routing_text, messages_to_routing_text

DATASET_STAGES = ("pretrain", "core", "finetune")


@dataclass
class LSRDatasetItem:
    id: str
    facet: str
    texts: list[str]
    expected_facets: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    split: str = "train"
    stage: str = "pretrain"


def load_lsr_dataset(path: str | Path, default_facet: str | None = None) -> list[LSRDatasetItem]:
    items: list[LSRDatasetItem] = []
    for dataset_path in expand_dataset_paths(path):
        items.extend(load_lsr_dataset_file(dataset_path, default_facet))
    if not items:
        raise RouterError(f"No dataset items found in: {path}")
    return items


def expand_dataset_paths(path: str | Path) -> list[Path]:
    raw = str(path)
    if "," in raw:
        paths: list[Path] = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                paths.extend(expand_dataset_paths(part))
        return paths

    dataset_path = Path(path)
    if dataset_path.is_dir():
        canonical_train_files = [
            dataset_path / filename
            for filename in ("pretraining.jsonl", "training.jsonl")
            if (dataset_path / filename).is_file()
        ]
        if canonical_train_files:
            return canonical_train_files
        files = [
            file_path
            for file_path in sorted(dataset_path.glob("*.jsonl"))
            if dataset_stage_from_path(file_path) in DATASET_STAGES
        ]
        if files:
            return files
        return sorted(dataset_path.glob("*.jsonl"))
    return [dataset_path]


def load_lsr_dataset_file(path: Path, default_facet: str | None = None) -> list[LSRDatasetItem]:
    if not path.exists():
        raise RouterError(f"Dataset path not found: {path}")
    if not path.is_file():
        raise RouterError(f"Dataset path is not a file: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        raw_items = load_dataset(path)
    except RouterError:
        raise

    if raw_items and all(is_canonical_facet_record(item) for item in raw_items):
        return [
            decode_canonical_facet_record(item, index, path, default_facet)
            for index, item in enumerate(raw_items)
        ]

    stage = dataset_stage_from_path(path)
    if stage in DATASET_STAGES and all(is_stage_item(item, stage) for item in raw_items):
        facet = default_facet or infer_dataset_facet(path)
        split = infer_dataset_split(path)
        return [
            decode_stage_item(item, index, facet, split, stage)
            for index, item in enumerate(raw_items)
        ]

    if raw_items and all(is_legacy_lsr_item(item) for item in raw_items):
        return [decode_legacy_lsr_item(item, index, path, default_facet) for index, item in enumerate(raw_items)]

    facet = default_facet
    if facet is None and not all(str(item.get("facet") or "").strip() for item in raw_items):
        facet = infer_facet_from_paths(str(path), None, None)
    return adapt_raw_dataset(raw_items, facet, infer_dataset_split(path), stage or "pretrain")


def is_stage_item(item: dict[str, Any], stage: str) -> bool:
    if not isinstance(item.get("id"), str):
        return False
    if stage == "pretrain":
        return isinstance(item.get("text"), str)
    return isinstance(item.get("messages"), list)


def is_canonical_facet_record(item: dict[str, Any]) -> bool:
    return item.get("schemaVersion") == "facet-record-v3"


def decode_canonical_facet_record(
    item: dict[str, Any],
    index: int,
    path: Path,
    default_facet: str | None = None,
) -> LSRDatasetItem:
    facet = str(item.get("facet") or default_facet or infer_dataset_facet(path)).strip().lower()
    if not facet:
        raise RouterError(f"Canonical facet record {index} is missing facet.")
    item_id = str(item.get("id") or f"{facet}-{index}").strip()
    activation = item.get("activation") if isinstance(item.get("activation"), dict) else {}
    expected = (
        activation.get("directSelectedFacets")
        or activation.get("direct_selected_facets")
        or activation.get("directFacets")
        or activation.get("direct_facets")
    )
    if not expected:
        primary = activation.get("primaryFacet") or activation.get("primary_facet")
        expected = [primary] if isinstance(primary, str) and primary.strip() else [facet]
    expected_facets = normalize_expected_facets(expected, facet)
    metadata: dict[str, Any] = {}
    for field in ("coverage", "provenance", "supervision"):
        value = item.get(field)
        if isinstance(value, dict):
            metadata[field] = value
    metadata["activation"] = activation
    return LSRDatasetItem(
        id=item_id,
        facet=facet,
        texts=[facet_record_to_routing_text(item, index)],
        expected_facets=expected_facets,
        metadata=metadata,
        split=str(item.get("split") or infer_dataset_split(path)),
        stage=str(item.get("stage") or "sft"),
    )


def decode_stage_item(
    item: dict[str, Any],
    index: int,
    facet: str,
    split: str,
    stage: str,
) -> LSRDatasetItem:
    item_id = str(item.get("id") or f"{facet}_{stage}_{index:04d}").strip()
    if stage == "pretrain":
        text = str(item.get("text") or "").strip()
        texts = [text] if text else []
    else:
        texts = [messages_to_routing_text(item.get("messages"), index)]
    if not texts or not any(text.strip() for text in texts):
        raise RouterError(f"{stage} dataset item {index} has no routable text.")
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return LSRDatasetItem(
        id=item_id,
        facet=facet.lower(),
        texts=[text for text in texts if text.strip()],
        expected_facets=[facet.lower()],
        metadata=metadata,
        split=split,
        stage=stage,
    )


def messages_to_text(messages: Any, index: int) -> str:
    return messages_to_routing_text(messages, index)


def is_legacy_lsr_item(item: dict[str, Any]) -> bool:
    return (
        isinstance(item.get("texts"), list)
        and isinstance(item.get("expectedFacets"), list)
    )


def decode_legacy_lsr_item(
    item: dict[str, Any],
    index: int,
    path: Path,
    default_facet: str | None = None,
) -> LSRDatasetItem:
    facet = str(item.get("facet") or default_facet or infer_dataset_facet(path)).strip().lower()
    texts = [str(text).strip() for text in item.get("texts", []) if str(text).strip()]
    if not texts:
        raise RouterError(f"Legacy LSR dataset item {index} has no texts.")
    expected = item.get("expectedFacets") or item.get("expected_facets") or [facet]
    expected_facets = normalize_expected_facets(expected, facet)
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return LSRDatasetItem(
        id=str(item.get("id") or f"{facet}-{index}"),
        facet=facet,
        texts=texts,
        expected_facets=expected_facets,
        metadata=metadata,
        split=str(item.get("split") or infer_dataset_split(path)),
        stage=str(item.get("stage") or dataset_stage_from_path(path) or "pretrain"),
    )


def normalize_expected_facets(value: Any, fallback: str) -> list[str]:
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else [fallback]
    if isinstance(value, list):
        facets = [str(facet).strip().lower() for facet in value if str(facet).strip()]
        return facets or [fallback]
    return [fallback]


def adapt_raw_dataset(
    dataset: list[dict[str, Any]],
    default_facet: str | None = None,
    split: str = "train",
    stage: str = "pretrain",
) -> list[LSRDatasetItem]:
    examples = examples_from_dataset(dataset, default_facet)
    items: list[LSRDatasetItem] = []
    for index, example in enumerate(examples):
        items.append(
            LSRDatasetItem(
                id=f"{example.facet}_{stage}_{index:04d}",
                facet=example.facet,
                texts=[example.text],
                expected_facets=[example.facet],
                metadata={},
                split=split,
                stage=stage,
            )
        )
    return items


def items_to_examples(items: list[LSRDatasetItem]) -> list[RouteExample]:
    examples: list[RouteExample] = []
    for item in items:
        examples.extend(RouteExample(facet=item.facet, text=text) for text in item.texts)
    return examples


def load_labeled_prompts(path: str | Path) -> tuple[list[str], list[list[str]]]:
    if should_use_training_dataset_loader(path):
        items = load_lsr_dataset(path)
        return dataset_items_to_labeled_prompts(items)

    raw = read_file(path)
    try:
        return load_legacy_benchmark_items(raw)
    except RouterError:
        items = load_lsr_dataset(path)
        return dataset_items_to_labeled_prompts(items)


def dataset_items_to_labeled_prompts(items: list[LSRDatasetItem]) -> tuple[list[str], list[list[str]]]:
    prompts = ["\n".join(item.texts) for item in items]
    expected = [item.expected_facets for item in items]
    return prompts, expected


def should_use_training_dataset_loader(path: str | Path) -> bool:
    raw = str(path)
    if "," in raw:
        return True
    dataset_path = Path(path)
    return dataset_path.is_dir() or dataset_stage_from_path(dataset_path) in DATASET_STAGES


def load_legacy_benchmark_items(raw: str) -> tuple[list[str], list[list[str]]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        data = []
        for line_index, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as line_exc:
                raise RouterError(f"Benchmark JSONL line {line_index} is invalid: {line_exc}") from exc
    if not isinstance(data, list):
        raise RouterError("Benchmark input must be a JSON list.")
    prompts: list[str] = []
    expected: list[list[str]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise RouterError(f"Benchmark item {index} must be an object.")
        prompt = item.get("input") or item.get("prompt") or item.get("text")
        if prompt is None and isinstance(item.get("texts"), list):
            prompt = "\n".join(str(text) for text in item["texts"])
        expected_facets = (
            item.get("expectedFacets")
            or item.get("expected_facets")
            or item.get("expected")
            or item.get("facet")
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise RouterError(f"Benchmark item {index} has no prompt text.")
        prompts.append(prompt.strip())
        expected.append(normalize_expected_facets(expected_facets, ""))
    return prompts, expected


def write_stage_jsonl(path: str | Path, items: list[LSRDatasetItem]) -> None:
    output_path = Path(path)
    stage = dataset_stage_from_path(output_path)
    if stage not in DATASET_STAGES:
        raise RouterError("Stage dataset filename must end in _pretrain.jsonl, _core.jsonl, or _finetune.jsonl.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            if stage == "pretrain":
                for text_index, text in enumerate(item.texts):
                    payload = {
                        "id": item.id if len(item.texts) == 1 else f"{item.id}_{text_index:04d}",
                        "text": text,
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            else:
                payload = {
                    "id": item.id,
                    "messages": [{"role": "user", "content": "\n".join(item.texts)}],
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def dataset_stage_from_path(path: Path) -> str:
    stem = path.stem.lower()
    for stage in DATASET_STAGES:
        if stem.endswith(f"_{stage}"):
            return stage
    return ""


def infer_dataset_split(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    if "test" in lowered:
        return "test"
    if "train" in lowered:
        return "train"
    return "train"


def infer_dataset_facet(path: Path) -> str:
    stem = path.stem.lower()
    for stage in DATASET_STAGES:
        suffix = f"_{stage}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)].lower()
    return infer_facet_from_paths(str(path), None, None)
