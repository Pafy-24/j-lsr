from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class LabeledRoute:
    case_id: str
    expected_facets: frozenset[str]


def evaluate_multilabel(cases: Iterable[LabeledRoute], routes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(zip(cases, routes))
    if not rows:
        return {"cases": 0, "precision": 0.0, "recall": 0.0, "falseActivation": 0.0, "abstention": 0.0, "calibration": {"ece": 0.0, "brier": 0.0}}
    precision_values: list[float] = []
    recall_values: list[float] = []
    false_activations = 0
    abstention_cases = 0
    abstention_correct = 0
    calibration_error = 0.0
    brier = 0.0
    calibration_count = 0
    for case, route in rows:
        predicted = {str(item.get("facet")) for item in route.get("selectedFacets", []) if str(item.get("facet") or "")}
        expected = set(case.expected_facets)
        true_positive = len(predicted & expected)
        precision_values.append(true_positive / len(predicted) if predicted else (1.0 if not expected else 0.0))
        recall_values.append(true_positive / len(expected) if expected else (1.0 if not predicted else 0.0))
        false_activations += len(predicted - expected)
        if not expected:
            abstention_cases += 1
            abstention_correct += int(not predicted)
        for item in route.get("results", []):
            facet = str(item.get("facet") or "")
            score = max(0.0, min(1.0, float(item.get("weight", 0.0))))
            label = 1.0 if facet in expected else 0.0
            calibration_error += abs(score - label)
            brier += (score - label) ** 2
            calibration_count += 1
    return {"cases": len(rows), "precision": sum(precision_values) / len(rows), "recall": sum(recall_values) / len(rows), "falseActivation": false_activations / len(rows), "abstention": abstention_correct / abstention_cases if abstention_cases else 0.0, "calibration": {"ece": calibration_error / calibration_count if calibration_count else 0.0, "brier": brier / calibration_count if calibration_count else 0.0}}


def timing_report(*, embedding_ms: Iterable[float] = (), search_ms: Iterable[float] = ()) -> dict[str, Any]:
    def summarize(values: Iterable[float]) -> dict[str, float]:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return {"count": 0, "p50Ms": 0.0, "p95Ms": 0.0, "meanMs": 0.0}
        def percentile(percent: float) -> float:
            index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percent)))
            return ordered[index]
        return {"count": len(ordered), "p50Ms": percentile(0.50), "p95Ms": percentile(0.95), "meanMs": sum(ordered) / len(ordered)}
    return {"embedding": summarize(embedding_ms), "search": summarize(search_ms)}
