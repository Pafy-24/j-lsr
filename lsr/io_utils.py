from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .errors import RouterError


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
        data = load_jsonl_dataset(raw, path, exc)

    if not isinstance(data, list):
        raise RouterError("Dataset must be a JSON list of objects.")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise RouterError(f"Dataset item {index} must be an object.")
    return data


def load_jsonl_dataset(raw: str, path: str | Path, original_error: json.JSONDecodeError) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RouterError(f"Invalid JSON/JSONL in {path} at line {index}: {exc}") from exc
        if not isinstance(item, dict):
            raise RouterError(f"JSONL dataset item {index} must be an object.")
        items.append(item)
    if not items:
        raise RouterError(f"Invalid JSON in {path}: {original_error}")
    return items
