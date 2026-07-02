from __future__ import annotations

import re
from dataclasses import dataclass

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
