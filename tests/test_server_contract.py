from __future__ import annotations

import pytest

from lsr.errors import RouterError
from lsr.server import candidate_facet_filter, extract_allowed_facets, extract_strategy_operation


def test_allowed_facets_are_normalized_and_deduplicated() -> None:
    assert extract_allowed_facets(
        {"allowedFacets": [" Coder ", "coder.python", "CODER"]}
    ) == ["coder", "coder.python"]


@pytest.mark.parametrize("value", [[], [""], [42], "coder"])
def test_allowed_facets_reject_invalid_contracts(value: object) -> None:
    with pytest.raises(RouterError, match="allowed_facets"):
        extract_allowed_facets({"allowed_facets": value})


def test_candidate_filter_reports_effective_and_missing_facets() -> None:
    result = candidate_facet_filter(
        ["coder", "coder.python", "missing"],
        {"coder": object(), "coder.python": object()},
    )

    assert result == {
        "applied": True,
        "requested": ["coder", "coder.python", "missing"],
        "effective": ["coder", "coder.python"],
        "missing": ["missing"],
    }


@pytest.mark.parametrize("value", [["route"], {"operation": "route"}, 42, True])
def test_strategy_operation_rejects_non_string_values(value: object) -> None:
    with pytest.raises(RouterError, match="strategy_operation"):
        extract_strategy_operation({"strategy_operation": value})
