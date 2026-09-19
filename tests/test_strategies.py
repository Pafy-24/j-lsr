from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lsr.models import LayeredSignature, SignatureLayer
from lsr.registry import SignatureRegistry
from lsr.routing import LSRRouter
from lsr.strategies import (
    FingerprintSpec,
    FingerprintGateSemanticStrategy,
    FingerprintOnlyStrategy,
    SemanticBaselineStrategy,
    StrategyAsset,
    StrategyError,
    build_strategy,
    fingerprint_hex,
    jaccard_hex,
    load_strategy,
)


def _signature(facet: str, vector: list[float]) -> LayeredSignature:
    matrix = np.asarray([vector, vector], dtype=np.float32)
    return LayeredSignature(facet=facet, dimension=len(vector), layers={1: SignatureLayer(1, matrix, [])}, layer_sizes={1: len(matrix)})


def _registry() -> tuple[SignatureRegistry, LSRRouter]:
    registry = SignatureRegistry("unused")
    registry.signatures = {
        "first": _signature("first", [1.0, 0.0]),
        "second": _signature("second", [-1.0, 0.0]),
    }
    router = LSRRouter(registry.signatures, top_k=2)
    return registry, router


def test_semantic_baseline_preserves_default_route_and_filters_request_locally() -> None:
    registry, router = _registry()
    strategy = SemanticBaselineStrategy.build(registry, router)
    embedding = np.asarray([1.0, 0.0], dtype=np.float32)
    assert strategy.route("same", embedding) == router.route("same", embedding)
    filtered = strategy.route("same", embedding, candidate_facets=["first"])
    assert [item["facet"] for item in filtered["results"]] == ["first"]
    empty = strategy.route("same", embedding, candidate_facets=[])
    assert empty["selectedFacets"] == []
    assert set(registry.signatures) == {"first", "second"}


def test_asset_roundtrip_identity_and_defensive_config(tmp_path) -> None:
    registry, router = _registry()
    strategy = FingerprintOnlyStrategy.build(registry, router, minimum_jaccard=0.7)
    path = tmp_path / "strategy.json"
    strategy.asset.save(path)
    loaded_asset = StrategyAsset.load(path)
    assert loaded_asset.identity == strategy.asset.identity
    config = loaded_asset.config
    config["minimumJaccard"] = 0.0
    assert StrategyAsset.load(path).config["minimumJaccard"] == 0.7
    loaded = load_strategy(str(path), registry, router)
    assert loaded.asset.identity == strategy.asset.identity


def test_fingerprint_gate_cannot_reactivate_refused_or_filtered_facets() -> None:
    registry, router = _registry()
    strategy = FingerprintGateSemanticStrategy.build(registry, router, minimum_jaccard=0.7)
    embedding = np.asarray([1.0, 0.0], dtype=np.float32)
    first = strategy.route("first", embedding)
    assert [item["facet"] for item in first["selectedFacets"]] == ["first"]
    refused = strategy.route("first", embedding, candidate_facets=["second"])
    assert refused["selectedFacets"] == []
    assert refused["results"] == []


def test_request_local_filters_are_isolated_under_concurrency() -> None:
    registry, router = _registry()
    strategy = SemanticBaselineStrategy.build(registry, router)
    embedding = np.asarray([1.0, 0.0], dtype=np.float32)

    def route(name: str) -> list[str]:
        return [item["facet"] for item in strategy.route(name, embedding, candidate_facets=[name])["results"]]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(route, ["first", "second"]))
    assert results == [["first"], ["second"]]
    assert set(registry.signatures) == {"first", "second"}


def test_strategy_factory_builds_all_versioned_variants() -> None:
    registry, router = _registry()
    for strategy_id in ("semantic-baseline-v1", "fingerprint-only-v1", "fingerprint-gate-semantic-v1"):
        strategy = build_strategy(strategy_id, registry, router)
        assert strategy.asset.strategy_id == strategy_id


def test_snapshot_identity_includes_loaded_vector_content() -> None:
    registry, _ = _registry()
    before = registry.snapshot().identity
    registry.signatures["first"].layers[1].vectors[0, 0] = 7.0
    after = registry.snapshot().identity
    assert after != before


def test_signed_fingerprint_distinguishes_opposite_vectors() -> None:
    spec = FingerprintSpec(bits=128, seed="counterexample")
    positive = fingerprint_hex([1.0, 1.0], spec)
    negative = fingerprint_hex([-1.0, -1.0], spec)
    assert positive != negative
    assert jaccard_hex(positive, negative, spec.bits) < 1.0


def test_fingerprint_rejects_nonfinite_values() -> None:
    spec = FingerprintSpec()
    try:
        fingerprint_hex([float("nan"), 1.0], spec)
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite vectors must fail closed")


def test_asset_identity_includes_complete_router_settings() -> None:
    registry, _ = _registry()
    first = SemanticBaselineStrategy.build(registry, LSRRouter(registry.signatures, top_k=1))
    second = SemanticBaselineStrategy.build(registry, LSRRouter(registry.signatures, top_k=9))
    assert first.asset.identity != second.asset.identity


def test_asset_loader_rejects_unknown_fields_and_algorithm() -> None:
    registry, router = _registry()
    asset = FingerprintOnlyStrategy.build(registry, router).asset.to_dict()
    unknown = dict(asset)
    unknown["unexpected"] = True
    try:
        StrategyAsset.from_dict(unknown)
    except StrategyError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown asset fields must fail closed")

    invalid_algorithm = dict(asset)
    invalid_algorithm["config"] = dict(asset["config"])
    invalid_algorithm["config"]["algorithm"] = "toy-unsupported"
    try:
        StrategyAsset.from_dict(invalid_algorithm)
    except StrategyError as error:
        assert "algorithm" in str(error)
    else:
        raise AssertionError("unknown fingerprint algorithms must fail closed")
