from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .errors import RouterError
from .models import RouteExample

TECH_TERM_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Z0-9]{2,}|[a-zA-Z]+-[a-zA-Z0-9-]+|nonce|signature|"
    r"private\s+key|public\s+key|elliptic\s+curve|hash|cipher|oracle|"
    r"lattice|quantum|POVM|RSA|DSA|ECDSA|AES|CBC|CTR|GCM|MAC|HMAC|PRF|PRG)\b",
    re.IGNORECASE,
)


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


class DatasetRouteExtractor:
    def examples(self, dataset: list[dict[str, Any]], default_facet: str | None = None) -> list[RouteExample]:
        examples: list[RouteExample] = []
        for index, item in enumerate(dataset):
            facet = str(item.get("facet") or default_facet or "").strip().lower()
            if not facet:
                raise RouterError(f"Dataset item {index} is missing required field: facet")

            route_texts = self.route_texts_from_exam_item(item) or [extract_route_text(item)]
            route_texts = list(dict.fromkeys(text for text in route_texts if text))
            if not route_texts:
                raise RouterError(
                    f"Dataset item {index} for facet '{facet}' has no routable text."
                )
            examples.extend(RouteExample(facet=facet, text=text) for text in route_texts)
        return examples

    def route_texts_from_exam_item(self, item: dict[str, Any]) -> list[str]:
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


def route_texts_from_exam_item(item: dict[str, Any]) -> list[str]:
    return DatasetRouteExtractor().route_texts_from_exam_item(item)


def examples_from_dataset(
    dataset: list[dict[str, Any]], default_facet: str | None = None
) -> list[RouteExample]:
    return DatasetRouteExtractor().examples(dataset, default_facet)


def grouped_examples(examples: list[RouteExample]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for example in examples:
        grouped.setdefault(example.facet, []).append(example.text)
    return grouped
