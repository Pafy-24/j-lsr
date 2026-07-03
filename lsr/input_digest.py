from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import (
    LARGE_INPUT_CHUNK_CHARS,
    LARGE_INPUT_FIRST_CHUNKS,
    LARGE_INPUT_KEYWORD_CHUNKS,
    LARGE_INPUT_LAST_CHUNKS,
    LARGE_INPUT_MAX_DIGEST_CHARS,
    LARGE_INPUT_THRESHOLD_CHARS,
)
from .dataset import compact_whitespace, technical_terms


HEADING_PATTERN = re.compile(r"^\s{0,3}(#{1,6}\s+.+|<h[1-6][^>]*>.*?</h[1-6]>)\s*$", re.IGNORECASE)
TITLE_PATTERN = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


@dataclass
class RoutingInput:
    original: str
    routing_text: str
    input_mode: str
    input_chars: int
    routing_text_chars: int
    chunks_used: int


def prepare_routing_input(text: str, mode: str = "auto") -> RoutingInput:
    if mode not in {"auto", "off", "force"}:
        mode = "auto"
    input_chars = len(text)
    should_digest = mode == "force" or (mode == "auto" and input_chars > LARGE_INPUT_THRESHOLD_CHARS)
    if mode == "off" or not should_digest:
        return RoutingInput(
            original=text,
            routing_text=text,
            input_mode="raw",
            input_chars=input_chars,
            routing_text_chars=len(text),
            chunks_used=1 if text else 0,
        )

    digest, chunks_used = build_digest(text)
    return RoutingInput(
        original=text,
        routing_text=digest,
        input_mode="digest",
        input_chars=input_chars,
        routing_text_chars=len(digest),
        chunks_used=chunks_used,
    )


def prepare_routing_input_from_path(path: str | Path, mode: str = "auto") -> RoutingInput:
    input_path = Path(path)
    if mode not in {"auto", "off", "force"}:
        mode = "auto"
    if not input_path.exists():
        from .errors import RouterError

        raise RouterError(f"Input path not found: {input_path}")
    if mode == "off":
        return prepare_routing_input(input_path.read_text(encoding="utf-8"), "off")

    size_hint = input_path.stat().st_size
    should_stream_digest = mode == "force" or size_hint > LARGE_INPUT_THRESHOLD_CHARS
    if not should_stream_digest:
        return prepare_routing_input(input_path.read_text(encoding="utf-8"), "auto")

    digest, chunks_used, input_chars = build_digest_from_path(input_path)
    return RoutingInput(
        original="",
        routing_text=digest,
        input_mode="digest",
        input_chars=input_chars,
        routing_text_chars=len(digest),
        chunks_used=chunks_used,
    )


def build_digest(text: str) -> tuple[str, int]:
    chunks = split_chunks(text, LARGE_INPUT_CHUNK_CHARS)
    selected: list[str] = []

    title = extract_title(text)
    headings = extract_headings(text)
    if title:
        selected.append(f"Title: {title}")
    if headings:
        selected.append("Headings: " + " | ".join(headings[:16]))

    selected.extend(chunks[:LARGE_INPUT_FIRST_CHUNKS])
    if LARGE_INPUT_LAST_CHUNKS:
        selected.extend(chunks[-LARGE_INPUT_LAST_CHUNKS:])
    selected.extend(keyword_heavy_chunks(chunks, LARGE_INPUT_KEYWORD_CHUNKS))

    compacted: list[str] = []
    seen: set[str] = set()
    for part in selected:
        clean = compact_whitespace(part)
        if clean and clean not in seen:
            seen.add(clean)
            compacted.append(clean)

    digest = "\n\n".join(compacted)
    if len(digest) > LARGE_INPUT_MAX_DIGEST_CHARS:
        digest = digest[:LARGE_INPUT_MAX_DIGEST_CHARS].rsplit(" ", 1)[0].strip()
    return digest, len(compacted)


def build_digest_from_path(path: Path) -> tuple[str, int, int]:
    first_chunks: list[str] = []
    last_chunks: list[str] = []
    keyword_chunks: list[tuple[int, int, str]] = []
    headings: list[str] = []
    title = ""
    input_chars = 0
    chunk_index = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            chunk = handle.read(LARGE_INPUT_CHUNK_CHARS)
            if not chunk:
                break
            input_chars += len(chunk)
            if len(first_chunks) < LARGE_INPUT_FIRST_CHUNKS:
                first_chunks.append(chunk)
            if LARGE_INPUT_LAST_CHUNKS:
                last_chunks.append(chunk)
                if len(last_chunks) > LARGE_INPUT_LAST_CHUNKS:
                    last_chunks.pop(0)
            if not title:
                title = extract_title(chunk)
            if len(headings) < 16:
                headings.extend(extract_headings(chunk)[: 16 - len(headings)])
            keyword_count = len(technical_terms(chunk, limit=64).split())
            if keyword_count:
                keyword_chunks.append((keyword_count, chunk_index, chunk))
                keyword_chunks.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                if len(keyword_chunks) > LARGE_INPUT_KEYWORD_CHUNKS:
                    keyword_chunks.pop()
            chunk_index += 1

    selected: list[str] = []
    if title:
        selected.append(f"Title: {title}")
    if headings:
        selected.append("Headings: " + " | ".join(headings[:16]))
    selected.extend(first_chunks)
    selected.extend(last_chunks)
    selected.extend(chunk for _, _, chunk in keyword_chunks)
    return compact_digest_parts(selected), len({compact_whitespace(part) for part in selected if compact_whitespace(part)}), input_chars


def compact_digest_parts(parts: list[str]) -> str:
    compacted: list[str] = []
    seen: set[str] = set()
    for part in parts:
        clean = compact_whitespace(part)
        if clean and clean not in seen:
            seen.add(clean)
            compacted.append(clean)

    digest = "\n\n".join(compacted)
    if len(digest) > LARGE_INPUT_MAX_DIGEST_CHARS:
        digest = digest[:LARGE_INPUT_MAX_DIGEST_CHARS].rsplit(" ", 1)[0].strip()
    return digest


def split_chunks(text: str, chunk_chars: int) -> list[str]:
    if chunk_chars <= 0:
        return [text]
    return [text[index : index + chunk_chars] for index in range(0, len(text), chunk_chars)]


def extract_title(text: str) -> str:
    match = TITLE_PATTERN.search(text)
    if match:
        return compact_whitespace(strip_tags(match.group(1)))[:240]
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:240]
    return ""


def extract_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        if HEADING_PATTERN.match(line):
            headings.append(compact_whitespace(strip_tags(line.strip().lstrip("#"))))
    return [heading for heading in headings if heading]


def keyword_heavy_chunks(chunks: list[str], limit: int) -> list[str]:
    scored = []
    for index, chunk in enumerate(chunks):
        keyword_count = len(technical_terms(chunk, limit=64).split())
        if keyword_count:
            scored.append((keyword_count, index, chunk))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [chunk for _, _, chunk in scored[:limit]]


def strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)
