from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import json
import threading
from types import SimpleNamespace
from urllib.request import Request, urlopen
from unittest.mock import patch

import numpy as np
import pytest

from lsr.errors import RouterError
from lsr.models import LayeredSignature, SignatureLayer
from lsr.registry import SignatureRegistry
from lsr.routing import LSRRouter
from lsr.server import LSRHTTPServer, LSRRouteService
from lsr.signatures import SignatureCodec
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


class _TinyCpuEncoder:
    """Deliberately tiny test double: no model download or accelerator path."""

    backend = "tiny-cpu"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts, **_kwargs):
        self.calls += 1
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


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


def test_fingerprint_rejects_zero_norm_and_dimension_mismatch() -> None:
    spec = FingerprintSpec()
    with pytest.raises(ValueError, match="zero-norm"):
        fingerprint_hex([0.0, 0.0], spec)
    with pytest.raises(ValueError, match="dimension mismatch"):
        fingerprint_hex([1.0, 0.0, 0.0], spec, expected_dimension=2)


def test_asset_identity_includes_complete_router_settings() -> None:
    registry, _ = _registry()
    first = SemanticBaselineStrategy.build(registry, LSRRouter(registry.signatures, top_k=1))
    second = SemanticBaselineStrategy.build(registry, LSRRouter(registry.signatures, top_k=9))
    assert first.asset.identity != second.asset.identity


def test_strategy_rejects_router_mutation_after_asset_binding() -> None:
    registry, router = _registry()
    strategy = SemanticBaselineStrategy.build(registry, router)
    router.top_k = 1
    with pytest.raises(StrategyError, match="router settings changed"):
        strategy.route("same", np.asarray([1.0, 0.0], dtype=np.float32))


def test_real_lazy_signature_owns_bytes_and_identity_survives_release_and_overwrite(tmp_path) -> None:
    codec = SignatureCodec("test-model")
    signature = _signature("facet", [1.0, 0.0])
    path = codec.save(tmp_path / "facet.sig", signature)
    registry = SignatureRegistry(path)
    registry.load()
    router = LSRRouter(registry.signatures, top_k=1)
    strategy = SemanticBaselineStrategy.build(registry, router)
    snapshot_id = strategy.asset.snapshot_id
    loaded_layer = registry.signatures["facet"].get_layer(1)
    with pytest.raises(ValueError):
        loaded_layer.vectors[0, 0] = -1.0
    registry.signatures["facet"].release_layer(1)
    first = strategy.route("same", np.asarray([1.0, 0.0], dtype=np.float32))
    original_stat = path.stat()
    codec.save(path, _signature("facet", [-1.0, 0.0]))
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(RouterError, match="backing bytes changed"):
        strategy.route("same", np.asarray([1.0, 0.0], dtype=np.float32))
    assert strategy.snapshot().identity == snapshot_id


def test_lazy_codec_does_not_read_or_retain_the_whole_signature_file(tmp_path) -> None:
    codec = SignatureCodec("test-model")
    path = codec.save(tmp_path / "facet.sig", _signature("facet", [1.0, 0.0]))
    from unittest.mock import patch

    with patch.object(path.__class__, "read_bytes", side_effect=AssertionError("whole-file read")):
        loaded = codec.load(path)
    assert loaded.layer_loader is not None
    assert loaded.layer_digests


def test_registry_streams_source_digest_during_load_and_snapshot(tmp_path) -> None:
    codec = SignatureCodec("test-model")
    path = codec.save(tmp_path / "facet.sig", _signature("facet", [1.0, 0.0]))
    registry = SignatureRegistry(path)

    with patch.object(path.__class__, "read_bytes", side_effect=AssertionError("whole-file read")):
        registry.load()
        # Exercise the fallback path as well, which is used for registries
        # reconstructed from already-loaded signatures.
        registry.file_digests.clear()
        snapshot = registry.snapshot()

    assert snapshot.entries[0].content_sha256


def test_strategy_loader_rejects_version_that_does_not_match_implementation(tmp_path) -> None:
    registry, router = _registry()
    baseline = SemanticBaselineStrategy.build(registry, router).asset
    incompatible = StrategyAsset.create(
        baseline.strategy_id,
        "2",
        baseline.snapshot_id,
        baseline.config,
        fingerprints=baseline.fingerprint_map,
        created_at=baseline.created_at,
    )
    path = tmp_path / "incompatible-version.json"
    incompatible.save(path)

    with pytest.raises(StrategyError, match="version differs"):
        load_strategy(str(path), registry, router)


def test_strategy_loader_rejects_router_threshold_override(tmp_path) -> None:
    registry, router = _registry()
    asset_path = tmp_path / "semantic-baseline.json"
    SemanticBaselineStrategy.build(registry, router).asset.save(asset_path)
    overridden = LSRRouter(registry.signatures, top_k=2, accept_threshold=0.9)

    with pytest.raises(StrategyError, match="router settings differ"):
        load_strategy(str(asset_path), registry, overridden)


def test_invalid_asset_fails_before_service_encoder_load(tmp_path) -> None:
    codec = SignatureCodec("test-model")
    path = codec.save(tmp_path / "facet.sig", _signature("facet", [1.0, 0.0]))
    registry = SignatureRegistry(path)
    registry.load()
    baseline = SemanticBaselineStrategy.build(
        registry, LSRRouter(registry.signatures, top_k=1, model_name="test-model")
    ).asset
    invalid_path = tmp_path / "wrong-version.json"
    StrategyAsset.create(
        baseline.strategy_id,
        "2",
        baseline.snapshot_id,
        baseline.config,
        fingerprints=baseline.fingerprint_map,
        created_at=baseline.created_at,
    ).save(invalid_path)

    with patch("lsr.server.load_model", side_effect=AssertionError("encoder must not load")) as load_model:
        with pytest.raises(StrategyError, match="version differs"):
            LSRRouteService(
                str(path), "test-model", None, False, 1,
                monitor_enabled=False, strategy_asset_path=str(invalid_path),
            )
    assert load_model.call_count == 0


def test_invalid_local_rank_asset_fails_before_encoder_load(tmp_path) -> None:
    codec = SignatureCodec("test-model")
    signature_path = codec.save(tmp_path / "facet.sig", _signature("facet", [1.0, 0.0]))
    registry = SignatureRegistry(signature_path)
    registry.load()
    baseline = SemanticBaselineStrategy.build(
        registry, LSRRouter(registry.signatures, top_k=1, model_name="test-model")
    ).asset
    invalid_path = tmp_path / "wrong-version.json"
    StrategyAsset.create(
        baseline.strategy_id,
        "2",
        baseline.snapshot_id,
        baseline.config,
        fingerprints=baseline.fingerprint_map,
        created_at=baseline.created_at,
    ).save(invalid_path)

    from lsr.commands import RouterCLI

    args = SimpleNamespace(
        signatures=str(signature_path),
        model="test-model",
        index="auto",
        memory_limit_mb=0.0,
        descend_threshold=None,
        accept_threshold=None,
        full_depth=False,
        top_k=1,
        strategy_asset=str(invalid_path),
        strategy_operation="route",
        input_path=None,
        text="facet",
        large_input="off",
        PU=None,
        online=False,
        json=True,
    )
    with patch("lsr.commands.load_model", side_effect=AssertionError("encoder must not load")) as load_model:
        with pytest.raises(StrategyError, match="version differs"):
            RouterCLI(args).route_prompt()
    assert load_model.call_count == 0


def test_search_only_profile_does_not_invent_embedding_total() -> None:
    registry, router = _registry()
    profile = SemanticBaselineStrategy.build(registry, router).profile(
        "first", np.asarray([1.0, 0.0], dtype=np.float32)
    )
    assert profile["timing"]["scope"] == "search-only"
    assert profile["timing"]["embeddingMs"] is None
    assert profile["timing"]["totalMs"] is None


def test_service_http_and_cli_bind_the_same_validated_asset_without_model_download(tmp_path, monkeypatch, capsys) -> None:
    codec = SignatureCodec("test-model")
    codec.save(tmp_path / "first.sig", _signature("first", [1.0, 0.0]))
    codec.save(tmp_path / "second.sig", _signature("second", [-1.0, 0.0]))
    registry = SignatureRegistry(tmp_path)
    registry.load()
    router = LSRRouter(registry.signatures, top_k=1, model_name="test-model")
    asset_path = tmp_path / "semantic-baseline.json"
    SemanticBaselineStrategy.build(registry, router).asset.save(asset_path)

    # The CLI load operation uses only signatures and the asset; it has no
    # encoder construction path. Its printed asset is the exact service asset.
    from lsr.commands import RouterCLI

    cli_args = SimpleNamespace(
        PUs=False,
        command="strategy",
        action="load",
        signatures=str(tmp_path),
        asset=str(asset_path),
        output=None,
        strategy_id="semantic-baseline-v1",
        model="test-model",
        index="auto",
        memory_limit_mb=0.0,
        descend_threshold=None,
        accept_threshold=None,
        top_k=1,
        fingerprint_bits=128,
        fingerprint_seed="jester-lsr-j5",
        minimum_jaccard=0.5,
    )
    assert RouterCLI(cli_args).run() == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["asset"]["assetId"] == StrategyAsset.load(asset_path).identity

    encoder = _TinyCpuEncoder()
    with patch("lsr.server.load_model", return_value=encoder) as load_model:
        service = LSRRouteService(
            signatures_path=str(tmp_path),
            model_name="test-model",
            pu_index=None,
            online=False,
            default_top_k=1,
            monitor_enabled=False,
            strategy_asset_path=str(asset_path),
        )
    try:
        assert load_model.call_count == 1
        assert service.health()["strategy"]["asset"]["assetId"] == cli_payload["asset"]["assetId"]
        direct = service.route_one("first", allowed_facets=["first"])
        assert [item["facet"] for item in direct["results"]] == ["first"]

        http = LSRHTTPServer(("127.0.0.1", 0), service)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{http.server_port}/route",
                data=json.dumps({"input": "first", "allowedFacets": ["first"]}).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            assert [item["facet"] for item in payload["results"]] == ["first"]
        finally:
            http.shutdown()
            http.server_close()

        with pytest.raises(RouterError, match="default top_k"):
            service.route_one("first", top_k=2)
        with pytest.raises(RouterError, match="full_depth=false"):
            service.route_one("first", full_depth=True)
    finally:
        service.close()


def test_service_rejects_changed_registry_before_embedding_after_refresh(tmp_path) -> None:
    codec = SignatureCodec("test-model")
    path = codec.save(tmp_path / "facet.sig", _signature("facet", [1.0, 0.0]))
    registry = SignatureRegistry(path)
    registry.load()
    asset_path = tmp_path / "semantic-baseline.json"
    SemanticBaselineStrategy.build(
        registry, LSRRouter(registry.signatures, top_k=1, model_name="test-model")
    ).asset.save(asset_path)
    encoder = _TinyCpuEncoder()
    with patch("lsr.server.load_model", return_value=encoder):
        service = LSRRouteService(
            str(path), "test-model", None, False, 1,
            monitor_enabled=False, strategy_asset_path=str(asset_path),
        )
    try:
        codec.save(path, _signature("facet", [-1.0, 0.0]))
        service.refresh_signatures(force=True)
        calls_before = encoder.calls
        with pytest.raises(StrategyError, match="registry snapshot differs"):
            service.route_one("facet")
        assert encoder.calls == calls_before
    finally:
        service.close()


def test_service_preserves_large_input_metadata_before_route_serialization(tmp_path) -> None:
    codec = SignatureCodec("test-model")
    codec.save(tmp_path / "facet.sig", _signature("facet", [1.0, 0.0]))
    encoder = _TinyCpuEncoder()
    with patch("lsr.server.load_model", return_value=encoder):
        service = LSRRouteService(
            str(tmp_path), "test-model", None, False, 1, monitor_enabled=False
        )
    try:
        source = "# routing heading\n" + ("semantic routing payload. " * 800)
        route = service.route_one(source, large_input="force")
        assert route["inputMode"] == "digest"
        assert route["inputChars"] == len(source.strip())
        assert route["routingTextChars"] < len(source.strip())
        assert route["chunksUsed"] >= 1
        assert route["routingUnits"] >= 1
    finally:
        service.close()


def test_signature_reload_rejects_asset_mismatch_without_rebinding_old_registry(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    codec = SignatureCodec("test-model")
    codec.save(first_dir / "facet.sig", _signature("facet", [1.0, 0.0]))
    codec.save(second_dir / "facet.sig", _signature("facet", [-1.0, 0.0]))
    first_registry = SignatureRegistry(first_dir)
    first_registry.load()
    asset_path = tmp_path / "semantic-baseline.json"
    SemanticBaselineStrategy.build(
        first_registry, LSRRouter(first_registry.signatures, top_k=1, model_name="test-model")
    ).asset.save(asset_path)
    encoder = _TinyCpuEncoder()
    with patch("lsr.server.load_model", return_value=encoder):
        service = LSRRouteService(
            str(first_dir), "test-model", None, False, 1,
            monitor_enabled=False, strategy_asset_path=str(asset_path),
        )
    try:
        old_snapshot = service.health()["strategy"]["snapshotId"]
        with pytest.raises(StrategyError, match="registry snapshot differs"):
            service.load_signatures(str(second_dir))
        assert service.health()["strategy"]["snapshotId"] == old_snapshot
        assert service.health()["signatures"] == str(first_dir)
        route = service.route_one("facet")
        assert route["selectedFacets"]
    finally:
        service.close()


def test_real_lazy_request_filters_are_isolated_concurrently(tmp_path) -> None:
    codec = SignatureCodec("test-model")
    codec.save(tmp_path / "first.sig", _signature("first", [1.0, 0.0]))
    codec.save(tmp_path / "second.sig", _signature("second", [-1.0, 0.0]))
    registry = SignatureRegistry(tmp_path)
    registry.load()
    strategy = SemanticBaselineStrategy.build(registry, LSRRouter(registry.signatures, top_k=1))
    embedding = np.asarray([1.0, 0.0], dtype=np.float32)

    def route(name: str) -> list[str]:
        return [item["facet"] for item in strategy.route(name, embedding, candidate_facets=[name])["results"]]

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(route, ["first", "second"])) == [["first"], ["second"]]


def test_real_lazy_unfiltered_and_filtered_routes_are_request_local_concurrently(tmp_path) -> None:
    codec = SignatureCodec("test-model")
    codec.save(tmp_path / "first.sig", _signature("first", [1.0, 0.0]))
    codec.save(tmp_path / "second.sig", _signature("second", [-1.0, 0.0]))
    registry = SignatureRegistry(tmp_path)
    registry.load()
    strategy = SemanticBaselineStrategy.build(registry, LSRRouter(registry.signatures, top_k=2))
    embedding = np.asarray([1.0, 0.0], dtype=np.float32)

    def route(candidate_facets):
        return [
            item["facet"]
            for item in strategy.route("prompt", embedding, candidate_facets=candidate_facets)["results"]
        ]

    # The bound router owns mutable request metrics and lazy-layer caches; no
    # strategy request may call it directly, including the default baseline.
    with patch.object(strategy.router, "route", side_effect=AssertionError("shared router used")):
        with ThreadPoolExecutor(max_workers=2) as executor:
            unfiltered = executor.submit(route, None)
            filtered = executor.submit(route, ["first"])
            assert unfiltered.result() == ["first", "second"]
            assert filtered.result() == ["first"]

    assert all(not signature.layers for signature in registry.signatures.values())


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
