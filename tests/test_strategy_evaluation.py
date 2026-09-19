from __future__ import annotations

from lsr.strategies import LabeledRoute, evaluate_multilabel, timing_report


def test_multilabel_evaluation_accepts_empty_expected_labels() -> None:
    report = evaluate_multilabel(
        [LabeledRoute("base", frozenset()), LabeledRoute("coder", frozenset({"coder", "coder.python"}))],
        [
            {"selectedFacets": [], "results": []},
            {"selectedFacets": [{"facet": "coder", "weight": 0.8}, {"facet": "coder.python", "weight": 0.6}], "results": [{"facet": "coder", "weight": 0.8}, {"facet": "coder.python", "weight": 0.6}]},
        ],
        label_universe=["coder", "coder.python"],
        missing_score_policy="no-candidate-zero",
    )
    assert report["cases"] == 2
    assert report["abstention"] == 1.0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0


def test_binned_ece_uses_confidence_and_accuracy_not_mean_absolute_error() -> None:
    report = evaluate_multilabel(
        [LabeledRoute("positive", frozenset({"coder"})), LabeledRoute("negative", frozenset())],
        [
            {"selectedFacets": [{"facet": "coder"}], "results": [{"facet": "coder", "weight": 0.5}]},
            {"selectedFacets": [], "results": [{"facet": "coder", "weight": 0.5}]},
        ],
        label_universe=["coder"],
        calibration_bin_count=2,
    )
    assert report["calibration"]["population"] == 2
    assert report["calibration"]["ece"] == 0.0


def test_evaluation_rejects_length_mismatch_and_requires_missing_score_policy() -> None:
    cases = [LabeledRoute("one", frozenset({"coder"}))]
    try:
        evaluate_multilabel(cases, [], label_universe=["coder"])
    except ValueError as error:
        assert "length mismatch" in str(error)
    else:
        raise AssertionError("length mismatch must fail closed")

    try:
        evaluate_multilabel(cases, [{"selectedFacets": [], "results": []}], label_universe=["coder"])
    except ValueError as error:
        assert "missing calibration scores" in str(error)
    else:
        raise AssertionError("missing calibration scores must be explicit")


def test_declared_empty_label_universe_is_preserved() -> None:
    report = evaluate_multilabel(
        [LabeledRoute("empty", frozenset())],
        [{"selectedFacets": [], "results": []}],
        label_universe=[],
    )
    assert report["labelUniverseDeclared"] is True
    assert report["labelUniverse"] == []
    assert report["calibration"]["population"] == 0


def test_timing_report_keeps_embedding_and_search_separate() -> None:
    report = timing_report(embedding_ms=[2.0, 4.0], search_ms=[1.0, 3.0, 5.0])
    assert report["embedding"]["count"] == 2
    assert report["search"]["p50Ms"] == 3.0
    assert report["search"]["p95Ms"] == 5.0
