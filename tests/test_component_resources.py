from __future__ import annotations

import hashlib
import ast
import json
import os
import sys
from importlib import metadata
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from jester_core import ComponentManifest, component_identity


RESOURCE_ROOT = Path(__file__).parents[1] / "src" / "lsr" / "resources"
MANIFEST_NAME = "component-manifest.json"
HASH = "a" * 64


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload() -> dict[str, object]:
    return {
        "encoder": {
            "asset": {"id": "encoder.example", "version": "1", "sha256": HASH},
            "modelIdentity": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "signatureSet": {
            "asset": {"id": "signatures.example", "version": "1", "sha256": "b" * 64}
        },
        "topK": 3,
    }


def test_manifest_binds_exact_source_resource_bytes_and_core_identity() -> None:
    manifest = ComponentManifest.model_validate(_json(RESOURCE_ROOT / MANIFEST_NAME))
    assert manifest.id == "lsr"
    assert manifest.version == "0.1.0"
    assert component_identity(manifest).manifest_sha256 == manifest.semantic_sha256
    assert manifest.capabilities[0].capability.concurrency.sharing == "dedicated-only"
    assert manifest.capabilities[0].capability.concurrency.max_concurrent_requests == 1

    for resource in manifest.resources:
        assert resource.installed_path.startswith("resources/")
        path = RESOURCE_ROOT.parent / resource.installed_path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == resource.sha256


def test_configuration_schema_accepts_only_complete_portable_payload() -> None:
    schema = _json(RESOURCE_ROOT / "lsr-serve-configuration-v1.schema.json")
    validator = Draft202012Validator(schema)
    validator.validate(_payload())

    defaults = _json(RESOURCE_ROOT / "lsr-serve-configuration-defaults-v1.json")
    assert defaults == {"topK": 3}
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(defaults)

    invalid = [
        _payload() | {"host": "127.0.0.1"},
        _payload() | {"online": True},
        _payload() | {"strategy": "new"},
        _payload() | {"topK": 0},
        _payload() | {"topK": 1025},
    ]
    for model_identity in (
        "C:/models/encoder",
        "../encoder",
        "./encoder",
        "/encoder",
        "org/../../encoder",
        "org//encoder",
        "org/./encoder",
        r"models\encoder",
        "model/base\n",
    ):
        payload = _payload()
        payload["encoder"] = {**payload["encoder"], "modelIdentity": model_identity}  # type: ignore[arg-type]
        invalid.append(payload)
    for field in ("id", "version", "sha256"):
        payload = _payload()
        asset = dict(payload["encoder"]["asset"])  # type: ignore[index]
        asset[field] = f"{asset[field]}\n"
        payload["encoder"] = {**payload["encoder"], "asset": asset}  # type: ignore[arg-type]
        invalid.append(payload)
    for payload in invalid:
        with pytest.raises(JsonSchemaValidationError):
            validator.validate(payload)


def test_mapping_is_closed_and_uses_explicit_asset_roles() -> None:
    mapping = _json(RESOURCE_ROOT / "lsr-serve-adapter-mapping-v1.json")
    assert mapping["adapter"] == "lsr.serve"
    assert mapping["environment"] == {}
    assert mapping["additionalArguments"] is False
    assert mapping["assetRoles"] == {
        "encoder": {
            "payloadRef": "/encoder/asset",
            "materialization": "directory",
            "argument": "--model",
        },
        "signatureSet": {
            "payloadRef": "/signatureSet/asset",
            "materialization": "directory",
            "argument": "--signatures",
        },
    }
    assert [item["flag"] for item in mapping["arguments"]] == [
        "--model",
        "--model-identity",
        "--signatures",
        "--top-k",
    ]


def test_capability_matches_managed_server_authorization_source() -> None:
    contract = _json(RESOURCE_ROOT / "lsr-route-capability-v1.json")
    assert contract["channelOperation"] == "lsr.route"
    assert contract["method"] == "POST"
    assert contract["allowedManagedStrategyOperations"] == ["route"]

    server_path = Path(__file__).parents[1] / "src" / "lsr" / "server.py"
    tree = ast.parse(server_path.read_text(encoding="utf-8"))
    do_post = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "do_POST"
    )
    authorize_bindings = []
    managed_strategy_guards = []
    for node in ast.walk(do_post):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "authorize":
                keywords = {item.arg: item.value for item in node.keywords if item.arg}
                operation = keywords.get("operation")
                method = keywords.get("method")
                if isinstance(operation, ast.Constant) and isinstance(method, ast.Constant):
                    authorize_bindings.append((operation.value, method.value))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq):
            if isinstance(node.left, ast.Name) and node.left.id == "strategy_operation":
                compared = node.comparators[0]
                if isinstance(compared, ast.Constant):
                    managed_strategy_guards.append(compared.value)
    assert (contract["channelOperation"], contract["method"]) in authorize_bindings
    assert contract["allowedManagedStrategyOperations"] == managed_strategy_guards


@pytest.mark.skipif(
    os.environ.get("JESTER_LSR_INSTALLED_RESOURCE_PROOF") != "1",
    reason="runs only in the isolated installed-wheel acceptance environment",
)
def test_installed_distribution_resources_match_manifest_without_importing_lsr() -> None:
    assert "lsr" not in sys.modules
    distribution = metadata.distribution("jester-lsr")
    package_root = Path(distribution.locate_file("lsr")).resolve()
    manifest_path = package_root / "resources" / MANIFEST_NAME
    manifest = ComponentManifest.model_validate(_json(manifest_path))
    assert "lsr" not in sys.modules

    installed_files = {str(item).replace("\\", "/") for item in distribution.files or ()}
    for resource in manifest.resources:
        relative = f"lsr/{resource.installed_path}"
        assert relative in installed_files
        path = Path(distribution.locate_file(relative)).resolve()
        assert path.is_relative_to(package_root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == resource.sha256
    assert "lsr" not in sys.modules
