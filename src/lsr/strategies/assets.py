from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ASSET_SCHEMA_VERSION = "lsr-strategy-asset-v1"
KNOWN_STRATEGIES = {"semantic-baseline-v1", "fingerprint-only-v1", "fingerprint-gate-semantic-v1"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class StrategyError(ValueError):
    """A strategy asset or registry snapshot cannot be used safely."""


def canonical_json(value: Any) -> str:
    _validate_json_value(value, "config")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_json_value(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise StrategyError(f"non-finite asset value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise StrategyError(f"asset object key at {path} is not a string")
            _validate_json_value(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, bool, float)):
        raise StrategyError(f"unsupported asset value at {path}")


@dataclass(frozen=True)
class StrategyAsset:
    schema_version: str
    strategy_id: str
    version: str
    snapshot_id: str
    config_json: str
    fingerprints: tuple[tuple[str, str], ...] = ()
    created_at: str = ""

    @classmethod
    def create(
        cls,
        strategy_id: str,
        version: str,
        snapshot_id: str,
        config: Mapping[str, Any],
        *,
        fingerprints: Mapping[str, str] | None = None,
        created_at: str = "",
    ) -> "StrategyAsset":
        if strategy_id not in KNOWN_STRATEGIES:
            raise StrategyError(f"unknown strategy asset: {strategy_id}")
        if not version:
            raise StrategyError("strategy asset version is required")
        if not HEX64.fullmatch(str(snapshot_id)):
            raise StrategyError("strategy asset snapshot_id must be a lowercase SHA-256")
        if fingerprints is not None:
            for facet, value in fingerprints.items():
                if not str(facet).strip() or not re.fullmatch(r"[0-9a-f]+", str(value)):
                    raise StrategyError("malformed facet fingerprint")
        return cls(
            schema_version=ASSET_SCHEMA_VERSION,
            strategy_id=str(strategy_id),
            version=str(version),
            snapshot_id=str(snapshot_id),
            config_json=canonical_json(dict(config)),
            fingerprints=tuple(sorted((str(key), str(value)) for key, value in (fingerprints or {}).items())),
            created_at=str(created_at),
        )

    @property
    def config(self) -> dict[str, Any]:
        """Return a defensive copy so callers cannot mutate the asset."""
        return json.loads(self.config_json)

    @property
    def fingerprint_map(self) -> dict[str, str]:
        return dict(self.fingerprints)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "strategyId": self.strategy_id,
            "version": self.version,
            "snapshotId": self.snapshot_id,
            "config": json.loads(self.config_json),
            "fingerprints": [[key, value] for key, value in self.fingerprints],
            "createdAt": self.created_at,
        }

    @property
    def identity(self) -> str:
        return hashlib.sha256(canonical_json(self.identity_payload()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["assetId"] = self.identity
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyAsset":
        required = {"schemaVersion", "strategyId", "version", "snapshotId", "config", "fingerprints", "createdAt", "assetId"}
        unknown = set(value) - required
        if unknown:
            raise StrategyError(f"unknown strategy asset fields: {sorted(unknown)}")
        if set(value) != required:
            raise StrategyError("strategy asset has missing required fields")
        if str(value.get("schemaVersion") or "") != ASSET_SCHEMA_VERSION:
            raise StrategyError("unsupported strategy asset schema")
        if not isinstance(value.get("config"), dict) or not isinstance(value.get("fingerprints"), list):
            raise StrategyError("strategy asset config/fingerprints have invalid types")
        config = value["config"]
        algorithm = config.get("algorithm")
        if algorithm is not None and algorithm != "signed-feature-hash-jaccard-v2":
            raise StrategyError("unsupported fingerprint algorithm")
        fingerprints = value["fingerprints"]
        parsed_fingerprints: dict[str, str] = {}
        for item in fingerprints:
            if not isinstance(item, list) or len(item) != 2 or not all(isinstance(part, str) for part in item):
                raise StrategyError("strategy asset fingerprint entry is malformed")
            if item[0] in parsed_fingerprints:
                raise StrategyError("strategy asset contains duplicate fingerprint facet")
            parsed_fingerprints[item[0]] = item[1]
        if not isinstance(value.get("createdAt"), str) or not isinstance(value.get("assetId"), str):
            raise StrategyError("strategy asset createdAt/assetId must be strings")
        asset = cls.create(
            value["strategyId"], value["version"], value["snapshotId"], config,
            fingerprints=parsed_fingerprints, created_at=value["createdAt"],
        )
        expected = value["assetId"]
        if expected != asset.identity:
            raise StrategyError("strategy asset identity digest mismatch")
        return asset

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "StrategyAsset":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise StrategyError("strategy asset must be a JSON object")
        return cls.from_dict(value)
