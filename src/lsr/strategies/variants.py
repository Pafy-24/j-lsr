from __future__ import annotations

from typing import Any, Iterable

from ..registry import SignatureRegistry
from ..routing import LSRRouter
from .assets import StrategyAsset, StrategyError
from .base import Strategy
from .fingerprint import FingerprintSpec, fingerprint_hex, jaccard_hex, signature_fingerprint


class SemanticBaselineStrategy(Strategy):
    strategy_id = "semantic-baseline-v1"
    version = "1"

    @classmethod
    def build(cls, registry: SignatureRegistry, router: LSRRouter) -> "SemanticBaselineStrategy":
        return cls(registry, router, cls._make_asset(registry, {"mode": "existing-semantic-router"}))


class FingerprintOnlyStrategy(Strategy):
    strategy_id = "fingerprint-only-v1"
    version = "1"

    def __init__(self, registry: SignatureRegistry, router: LSRRouter, asset: StrategyAsset):
        super().__init__(registry, router, asset)
        config = asset.config
        self.spec = FingerprintSpec(str(config.get("algorithm", "signed-feature-hash-jaccard-v1")), int(config.get("bits", 128)), str(config.get("seed", "jester-lsr-j5")))
        self.minimum_jaccard = float(config.get("minimumJaccard", 0.5))

    @classmethod
    def build(cls, registry: SignatureRegistry, router: LSRRouter, *, bits: int = 128, seed: str = "jester-lsr-j5", minimum_jaccard: float = 0.5) -> "FingerprintOnlyStrategy":
        spec = FingerprintSpec(bits=bits, seed=seed)
        fingerprints = {facet: signature_fingerprint(registry.signatures[facet], spec) for facet in registry.snapshot().facets}
        asset = cls._make_asset(registry, {"algorithm": spec.algorithm, "bits": bits, "seed": seed, "minimumJaccard": minimum_jaccard}, fingerprints=fingerprints)
        return cls(registry, router, asset)

    def _scores(self, prompt_embedding: Any, names: tuple[str, ...]) -> dict[str, float]:
        query = fingerprint_hex(prompt_embedding, self.spec)
        return {name: jaccard_hex(query, self.asset.fingerprint_map[name], self.spec.bits) for name in names}

    def route(self, prompt: str, prompt_embedding: Any, *, candidate_facets: Iterable[str] | None = None) -> dict[str, Any]:
        names = self._candidate_names(candidate_facets)
        if not names:
            return self._empty_route(prompt)
        scores = self._scores(prompt_embedding, names)
        results = []
        for facet in names:
            score = scores[facet]
            decision = "passed" if score >= self.minimum_jaccard else "refused"
            results.append({"facet": facet, "score": score, "best_vector_score": score, "cluster_score": score, "vector_index": 0, "exact_count": 0, "mode": "fingerprint", "vectors_scanned": 0, "decision": decision, "layers": [], "semanticScore": score, "anchorSimilarity": score, "confidence": score, "separation": 0.0, "weight": score if decision != "refused" else 0.0})
        results.sort(key=lambda item: item["score"], reverse=True)
        selected = [{"facet": item["facet"], "weight": item["weight"], "decision": item["decision"]} for item in results if item["decision"] != "refused"][: self.router.top_k]
        grouped = {"accepted": [], "passed": [], "refused": []}
        for item in results:
            grouped[item["decision"]].append(item["facet"])
        return {"schemaVersion": "lsr-route-v2", "input": prompt, "results": results[: self.router.top_k], "compared_facets": len(results), "deep_facets": [], "modelName": self.router.model_name, "indexBackend": "fingerprint", "vectorsScanned": 0, "activeFacets": {key: sorted(value) for key, value in grouped.items()}, "selectedFacets": selected, "weightSemantics": "independent-pass-to-max-v1", "loadingPolicy": "fingerprint-only", "thresholds": {"minimumJaccard": self.minimum_jaccard}, "fullDepth": False, "queryUnits": 1, "signatureMemory": {"facetCount": len(results), "peakResidentVectorBytes": 0}}

    def explain(self, prompt: str, prompt_embedding: Any, *, candidate_facets: Iterable[str] | None = None) -> dict[str, Any]:
        names = self._candidate_names(candidate_facets)
        result = super().explain(prompt, prompt_embedding, candidate_facets=candidate_facets)
        result["fingerprint"] = {"algorithm": self.spec.algorithm, "bits": self.spec.bits, "seed": self.spec.seed, "minimumJaccard": self.minimum_jaccard, "scores": self._scores(prompt_embedding, names) if names else {}}
        return result


class FingerprintGateSemanticStrategy(FingerprintOnlyStrategy):
    strategy_id = "fingerprint-gate-semantic-v1"
    version = "1"

    @classmethod
    def build(cls, registry: SignatureRegistry, router: LSRRouter, *, bits: int = 128, seed: str = "jester-lsr-j5", minimum_jaccard: float = 0.5) -> "FingerprintGateSemanticStrategy":
        spec = FingerprintSpec(bits=bits, seed=seed)
        fingerprints = {facet: signature_fingerprint(registry.signatures[facet], spec) for facet in registry.snapshot().facets}
        asset = cls._make_asset(registry, {"algorithm": spec.algorithm, "bits": bits, "seed": seed, "minimumJaccard": minimum_jaccard, "mode": "fingerprint-gate-then-semantic"}, fingerprints=fingerprints)
        return cls(registry, router, asset)

    def route(self, prompt: str, prompt_embedding: Any, *, candidate_facets: Iterable[str] | None = None) -> dict[str, Any]:
        names = self._candidate_names(candidate_facets)
        scores = self._scores(prompt_embedding, names) if names else {}
        gated = tuple(name for name in names if scores[name] >= self.minimum_jaccard)
        return self._route_semantic_subset(prompt, prompt_embedding, gated)


def build_strategy(strategy_id: str, registry: SignatureRegistry, router: LSRRouter, **config: Any) -> Strategy:
    normalized = str(strategy_id).strip().lower()
    if normalized == SemanticBaselineStrategy.strategy_id:
        return SemanticBaselineStrategy.build(registry, router)
    if normalized == FingerprintOnlyStrategy.strategy_id:
        return FingerprintOnlyStrategy.build(registry, router, **config)
    if normalized == FingerprintGateSemanticStrategy.strategy_id:
        return FingerprintGateSemanticStrategy.build(registry, router, **config)
    raise StrategyError(f"unknown strategy: {strategy_id}")


def load_strategy(path: str, registry: SignatureRegistry, router: LSRRouter) -> Strategy:
    asset = StrategyAsset.load(path)
    classes = {cls.strategy_id: cls for cls in (SemanticBaselineStrategy, FingerprintOnlyStrategy, FingerprintGateSemanticStrategy)}
    strategy_class = classes.get(asset.strategy_id)
    if strategy_class is None:
        raise StrategyError(f"unknown strategy asset: {asset.strategy_id}")
    return strategy_class(registry, router, asset)
