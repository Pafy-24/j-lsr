from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ASSET_SCHEMA_VERSION = "lsr-strategy-asset-v1"


class StrategyError(ValueError):
    """A strategy asset or registry snapshot cannot be used safely."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
        if str(value.get("schemaVersion") or "") != ASSET_SCHEMA_VERSION:
            raise StrategyError("unsupported strategy asset schema")
        fingerprints = value.get("fingerprints") or []
        asset = cls.create(
            str(value.get("strategyId") or ""),
            str(value.get("version") or ""),
            str(value.get("snapshotId") or ""),
            value.get("config") if isinstance(value.get("config"), dict) else {},
            fingerprints={str(item[0]): str(item[1]) for item in fingerprints if isinstance(item, list) and len(item) == 2},
            created_at=str(value.get("createdAt") or ""),
        )
        expected = str(value.get("assetId") or "")
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
