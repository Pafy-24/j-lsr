from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class LabeledRoute:
    case_id: str
    expected_facets: frozenset[str]


def _calibration_bins(scores: list[tuple[float, float]], count: int) -> tuple[dict[str, Any], ...]:
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(count)]
    for score, label in scores:
        index = min(count - 1, int(score * count))
        buckets[index].append((score, label))
    result = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        confidence = sum(score for score, _ in bucket) / len(bucket)
        accuracy = sum(label for _, label in bucket) / len(bucket)
        result.append({"bin": index, "count": len(bucket), "confidence": confidence, "accuracy": accuracy, "gap": abs(confidence - accuracy)})
    return tuple(result)


def evaluate_multilabel(cases: Iterable[LabeledRoute], routes: Iterable[Mapping[str, Any]], *, label_universe: Iterable[str] | None = None, missing_score_policy: str = "error", calibration_bin_count: int = 10) -> dict[str, Any]:
    case_rows = list(cases)
    route_rows = list(routes)
    if len(case_rows) != len(route_rows):
        case_ids = [case.case_id for case in case_rows]
        raise ValueError(f"case/route length mismatch: cases={len(case_rows)} routes={len(route_rows)} caseIds={case_ids}")
    if calibration_bin_count < 1:
        raise ValueError("calibration_bin_count must be positive")
    if missing_score_policy not in {"error", "unavailable", "no-candidate-zero"}:
        raise ValueError("unknown missing_score_policy")
    universe = set(str(label) for label in label_universe) if label_universe is not None else None
    if universe is None:
        universe = set()
        for case in case_rows:
            universe.update(case.expected_facets)
        for route in route_rows:
            universe.update(str(item.get("facet")) for item in route.get("results", []) if str(item.get("facet") or ""))
    precision_values: list[float] = []
    recall_values: list[float] = []
    false_activations = 0
    abstention_cases = 0
    abstention_correct = 0
    calibration_scores: list[tuple[float, float]] = []
    unavailable: list[dict[str, Any]] = []
    for case, route in zip(case_rows, route_rows):
        predicted = {str(item.get("facet")) for item in route.get("selectedFacets", []) if str(item.get("facet") or "")}
        expected = set(case.expected_facets)
        true_positive = len(predicted & expected)
        precision_values.append(true_positive / len(predicted) if predicted else (1.0 if not expected else 0.0))
        recall_values.append(true_positive / len(expected) if expected else (1.0 if not predicted else 0.0))
        false_activations += len(predicted - expected)
        if not expected:
            abstention_cases += 1
            abstention_correct += int(not predicted)
        score_map = {str(item.get("facet")): item for item in route.get("results", []) if str(item.get("facet") or "")}
        missing = sorted(universe - set(score_map))
        if missing and missing_score_policy == "error":
            raise ValueError(f"missing calibration scores for case {case.case_id}: {missing}")
        for facet in sorted(universe):
            item = score_map.get(facet)
            if item is None:
                if missing_score_policy == "unavailable":
                    unavailable.append({"caseId": case.case_id, "facet": facet})
                    continue
                score = 0.0
            else:
                if "weight" not in item:
                    raise ValueError(f"calibration score unavailable for case {case.case_id} facet {facet}")
                score = float(item["weight"])
                if not 0.0 <= score <= 1.0:
                    raise ValueError(f"calibration score out of range for case {case.case_id} facet {facet}")
            calibration_scores.append((score, 1.0 if facet in expected else 0.0))
    bins = _calibration_bins(calibration_scores, calibration_bin_count)
    ece = sum(float(bucket["count"]) * float(bucket["gap"]) for bucket in bins) / len(calibration_scores) if calibration_scores else 0.0
    brier = sum((score - label) ** 2 for score, label in calibration_scores) / len(calibration_scores) if calibration_scores else 0.0
    return {"cases": len(case_rows), "labelUniverse": sorted(universe), "labelUniverseDeclared": label_universe is not None, "precision": sum(precision_values) / len(case_rows) if case_rows else 0.0, "recall": sum(recall_values) / len(case_rows) if case_rows else 0.0, "falseActivation": false_activations / len(case_rows) if case_rows else 0.0, "abstention": abstention_correct / abstention_cases if abstention_cases else 0.0, "calibration": {"population": len(calibration_scores), "unavailable": unavailable, "bins": list(bins), "ece": ece, "brier": brier}}


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
