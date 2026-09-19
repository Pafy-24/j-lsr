from __future__ import annotations

import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_+#.-]*", re.IGNORECASE)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "with",
}

GENERIC_TOKENS = {
    "help",
    "question",
    "example",
    "info",
    "information",
    "thing",
    "stuff",
    "topic",
    "something",
    "anything",
    "explain",
}


def tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def lexical_score(text: str, facet: str) -> float:
    """Compatibility diagnostic; routing never uses facet-specific vocabulary."""
    return 0.0


def generic_penalty(text: str) -> float:
    items = tokens(text)
    if not items:
        return 1.0
    unique = set(items)
    generic_hits = len(unique & GENERIC_TOKENS)
    if len(unique) <= 2:
        return 0.7
    if generic_hits and generic_hits / len(unique) >= 0.5:
        return 0.5
    return 0.0


def repetition_penalty(text: str) -> float:
    items = tokens(text)
    if len(items) < 6:
        return 0.0
    counts = Counter(items)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    token_penalty = repeated / len(items)
    bigrams = list(zip(items, items[1:]))
    if not bigrams:
        return min(1.0, token_penalty)
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(count - 1 for count in bigram_counts.values() if count > 1)
    bigram_penalty = repeated_bigrams / len(bigrams)
    return min(1.0, max(token_penalty, bigram_penalty))


def weighted_layer_score(layers: list[dict[str, object]]) -> float:
    if not layers:
        return 0.0
    weights = list(range(1, len(layers) + 1))
    total_weight = sum(weights)
    return sum(float(layer.get("score", 0.0)) * weight for layer, weight in zip(layers, weights)) / total_weight


def confidence_from_score(score: float, pass_threshold: float, accept_threshold: float) -> float:
    if accept_threshold <= pass_threshold:
        return 1.0 if score >= accept_threshold else 0.0
    return max(0.0, min(1.0, (score - pass_threshold) / (accept_threshold - pass_threshold)))


def facet_weight_from_score(score: float, pass_threshold: float, decision: str) -> float:
    """Return an independent multi-label activation weight in ``[0, 1]``.

    The weight is intentionally not normalized across facets: two relevant
    facets may both receive strong weights.  Crossing the accept threshold
    also does not force the weight to 1; full activation is reserved for the
    maximum cosine score.
    """
    if decision == "refused" or score <= pass_threshold:
        return 0.0
    if pass_threshold >= 1.0:
        return 1.0
    return max(0.0, min(1.0, (score - pass_threshold) / (1.0 - pass_threshold)))
