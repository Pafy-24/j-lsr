from __future__ import annotations

from lsr.strategies import LabeledRoute, evaluate_multilabel, timing_report


def test_multilabel_evaluation_accepts_empty_expected_labels() -> None:
    report = evaluate_multilabel(
        [LabeledRoute("base", frozenset()), LabeledRoute("coder", frozenset({"coder", "coder.python"}))],
        [
            {"selectedFacets": [], "results": []},
            {"selectedFacets": [{"facet": "coder", "weight": 0.8}, {"facet": "coder.python", "weight": 0.6}], "results": [{"facet": "coder", "weight": 0.8}, {"facet": "coder.python", "weight": 0.6}]},
        ],
    )
    assert report["cases"] == 2
    assert report["abstention"] == 1.0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0


def test_timing_report_keeps_embedding_and_search_separate() -> None:
    report = timing_report(embedding_ms=[2.0, 4.0], search_ms=[1.0, 3.0, 5.0])
    assert report["embedding"]["count"] == 2
    assert report["search"]["p50Ms"] == 3.0
    assert report["search"]["p95Ms"] == 5.0
