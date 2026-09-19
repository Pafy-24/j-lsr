from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterable

from ..models import LayeredSignature
from ..registry import RegistrySnapshot, SignatureRegistry
from ..routing import FACET_WEIGHT_SEMANTICS, ROUTE_SCHEMA_VERSION, LSRRouter
from .assets import StrategyAsset, StrategyError


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class Strategy:
    strategy_id = ""
    version = "1"

    def __init__(self, registry: SignatureRegistry, router: LSRRouter, asset: StrategyAsset):
        self.registry = registry
        self.router = router
        self.asset = asset
        self._check_snapshot()

    @classmethod
    def _make_asset(cls, registry: SignatureRegistry, config: dict[str, Any], *, fingerprints: dict[str, str] | None = None, created_at: str = "") -> StrategyAsset:
        snapshot = registry.snapshot()
        return StrategyAsset.create(cls.strategy_id, cls.version, snapshot.identity, config, fingerprints=fingerprints, created_at=created_at or utc_now())

    def _check_snapshot(self) -> RegistrySnapshot:
        snapshot = self.registry.snapshot()
        if snapshot.identity != self.asset.snapshot_id:
            raise StrategyError("registry snapshot differs from strategy asset")
        return snapshot

    def snapshot(self) -> RegistrySnapshot:
        return self._check_snapshot()

    def _candidate_names(self, candidate_facets: Iterable[str] | None) -> tuple[str, ...]:
        return self.snapshot().candidate_facets(candidate_facets)

    def _empty_route(self, prompt: str) -> dict[str, Any]:
        return {
            "schemaVersion": ROUTE_SCHEMA_VERSION,
            "input": prompt,
            "results": [],
            "compared_facets": 0,
            "deep_facets": [],
            "modelName": self.router.model_name,
            "indexBackend": "none",
            "vectorsScanned": 0,
            "activeFacets": {"accepted": [], "passed": [], "refused": []},
            "selectedFacets": [],
            "weightSemantics": FACET_WEIGHT_SEMANTICS,
            "loadingPolicy": self.router.loading_policy_label(),
            "thresholds": self.router.thresholds_label(),
            "fullDepth": self.router.full_depth,
            "queryUnits": 0,
            "signatureMemory": {"facetCount": 0, "peakResidentVectorBytes": 0},
        }

    def _route_semantic_subset(self, prompt: str, prompt_embedding: Any, names: tuple[str, ...]) -> dict[str, Any]:
        if not names:
            return self._empty_route(prompt)
        signatures = {name: self.registry.signatures[name].routing_copy() for name in names}
        return self.router.request_router(signatures).route(prompt, prompt_embedding)

    def route(self, prompt: str, prompt_embedding: Any, *, candidate_facets: Iterable[str] | None = None) -> dict[str, Any]:
        names = self._candidate_names(candidate_facets)
        if candidate_facets is None:
            return self.router.route(prompt, prompt_embedding)
        return self._route_semantic_subset(prompt, prompt_embedding, names)

    def explain(self, prompt: str, prompt_embedding: Any, *, candidate_facets: Iterable[str] | None = None) -> dict[str, Any]:
        names = self._candidate_names(candidate_facets)
        return {
            "strategy": self.asset.to_dict(),
            "registrySnapshot": self.snapshot().to_dict(),
            "candidateFacets": list(names),
            "route": self.route(prompt, prompt_embedding, candidate_facets=candidate_facets),
        }

    def profile(self, prompt: str, prompt_embedding: Any, *, candidate_facets: Iterable[str] | None = None, embedding_ms: float | None = None) -> dict[str, Any]:
        started = perf_counter()
        route = self.route(prompt, prompt_embedding, candidate_facets=candidate_facets)
        search_ms = (perf_counter() - started) * 1000.0
        return {"strategyId": self.asset.strategy_id, "strategyVersion": self.asset.version, "timing": {"embeddingMs": embedding_ms, "searchMs": search_ms, "totalMs": search_ms + float(embedding_ms or 0.0)}, "route": route}
