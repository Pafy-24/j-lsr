from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lsr.args import parse_args
from lsr.commands import RouterCLI
from lsr.errors import RouterError
from lsr.model_selection import (
    resolve_model_binding,
    validate_signature_model_compatibility,
)
from lsr.models import LayeredSignature, SignatureLayer
from lsr.signatures import SignatureCodec, model_hash
from lsr.strategies.base import router_settings


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class TinyEncoder:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


def signatures() -> dict[str, LayeredSignature]:
    return {
        "coder": LayeredSignature(
            facet="coder",
            dimension=384,
            layers={},
            model_name=MODEL_ID,
            model_hash=model_hash(MODEL_ID),
        )
    }


def parse_cli(monkeypatch: pytest.MonkeyPatch, *arguments: object):
    monkeypatch.setattr(sys, "argv", ["jester-lsr", *(str(value) for value in arguments)])
    return parse_args()


def sign_tiny_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_identity: str | None = MODEL_ID,
) -> tuple[Path, Path]:
    checkout = tmp_path / "checkout-a" / "encoder"
    signature_dir = tmp_path / "signatures"
    arguments: list[object] = [
        "sign",
        "synthetic route example",
        "--facet",
        "coder",
        "--signatures",
        signature_dir,
        "--model",
        checkout,
    ]
    if model_identity is not None:
        arguments.extend(("--model-identity", model_identity))
    args = parse_cli(monkeypatch, *arguments)

    with patch("lsr.commands.load_model", return_value=TinyEncoder()) as load_model:
        assert RouterCLI(args).run() == 0

    load_model.assert_called_once_with(str(checkout), None, False)
    signature_path = signature_dir / "coder.sig"
    assert signature_path.is_file()
    return checkout, signature_path


def test_cli_parses_portable_identity_for_sign_rank_and_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sign = parse_cli(
        monkeypatch,
        "sign",
        "text",
        "--model",
        "E:/checkout-a/model",
        "--model-identity",
        MODEL_ID,
    )
    rank = parse_cli(
        monkeypatch,
        "rank",
        "text",
        "--local",
        "--model",
        "E:/checkout-b/model",
        "--model-identity",
        MODEL_ID,
    )
    serve = parse_cli(
        monkeypatch,
        "serve",
        "--model",
        "E:/checkout-c/model",
        "--model-identity",
        MODEL_ID,
    )
    sign_without_identity = parse_cli(monkeypatch, "sign", "text", "--model", "test-model")

    assert sign.model_identity == MODEL_ID
    assert rank.local is True
    assert rank.model_identity == MODEL_ID
    assert serve.model_identity == MODEL_ID
    assert sign_without_identity.model_identity is None


def test_signing_checkout_a_writes_portable_identity_in_real_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, signature_path = sign_tiny_signature(tmp_path, monkeypatch)

    loaded = SignatureCodec().load(signature_path)
    assert loaded.model_name == MODEL_ID
    assert loaded.model_hash == hashlib.sha256(MODEL_ID.encode("utf-8")).digest()[:16]
    assert loaded.model_hash != hashlib.sha256(str(checkout).encode("utf-8")).digest()[:16]
    assert loaded.dimension == 3
    assert loaded.get_layer(1).vectors.shape == (1, 3)


def test_sign_without_identity_preserves_existing_model_identity_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, signature_path = sign_tiny_signature(
        tmp_path,
        monkeypatch,
        model_identity=None,
    )

    loaded = SignatureCodec().load(signature_path)
    assert loaded.model_name == str(checkout)
    assert loaded.model_hash == model_hash(str(checkout))


def test_rank_checkout_b_loads_physical_path_and_routes_with_portable_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, signature_path = sign_tiny_signature(tmp_path, monkeypatch)
    capsys.readouterr()
    checkout_b = tmp_path / "checkout-b" / "encoder"
    args = parse_cli(
        monkeypatch,
        "rank",
        "synthetic rank request",
        "--local",
        "--json",
        "--signatures",
        signature_path,
        "--model",
        checkout_b,
        "--model-identity",
        MODEL_ID,
    )
    cli = RouterCLI(args)
    _, router, model_load_path = cli._local_registry_router()
    assert model_load_path == str(checkout_b)
    assert router.model_name == MODEL_ID
    assert router_settings(router)["modelName"] == MODEL_ID

    with patch("lsr.commands.load_model", return_value=TinyEncoder()) as load_model:
        cli.route_prompt()

    load_model.assert_called_once_with(str(checkout_b), None, False)
    route = json.loads(capsys.readouterr().out)
    assert route["modelName"] == MODEL_ID
    assert route["selectedFacets"][0]["facet"] == "coder"


def test_batch_rank_uses_physical_checkout_with_portable_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, signature_path = sign_tiny_signature(tmp_path, monkeypatch)
    capsys.readouterr()
    checkout_b = tmp_path / "checkout-b" / "encoder"
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first request\nsecond request\n", encoding="utf-8")
    args = parse_cli(
        monkeypatch,
        "rank",
        "--local",
        "--batch",
        "--input",
        prompts,
        "--signatures",
        signature_path,
        "--model",
        checkout_b,
        "--model-identity",
        MODEL_ID,
    )

    with patch("lsr.commands.load_model", return_value=TinyEncoder()) as load_model:
        RouterCLI(args).route_batch()

    load_model.assert_called_once_with(str(checkout_b), None, False)
    routes = json.loads(capsys.readouterr().out)
    assert [route["modelName"] for route in routes] == [MODEL_ID, MODEL_ID]


def test_wrong_portable_identity_is_rejected_before_encoder_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, signature_path = sign_tiny_signature(tmp_path, monkeypatch)
    args = parse_cli(
        monkeypatch,
        "rank",
        "text",
        "--local",
        "--signatures",
        signature_path,
        "--model",
        tmp_path / "checkout-b" / "encoder",
        "--model-identity",
        "different/model",
    )

    with patch("lsr.commands.load_model", side_effect=AssertionError("encoder loaded")) as load_model:
        with pytest.raises(RouterError, match="different model"):
            RouterCLI(args).route_prompt()
    assert load_model.call_count == 0


def test_corrupt_model_hash_is_rejected_before_encoder_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = LayeredSignature(
        facet="coder",
        dimension=3,
        layers={
            1: SignatureLayer(
                level=1,
                vectors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                links=[],
            )
        },
        model_name=MODEL_ID,
        model_hash=b"corrupt-hash-000",
    )
    signature_path = SignatureCodec().save(tmp_path / "coder.sig", signature)
    assert SignatureCodec().load(signature_path).model_hash == b"corrupt-hash-000"
    args = parse_cli(
        monkeypatch,
        "rank",
        "text",
        "--local",
        "--signatures",
        signature_path,
        "--model",
        tmp_path / "checkout-b" / "encoder",
        "--model-identity",
        MODEL_ID,
    )

    with patch("lsr.commands.load_model", side_effect=AssertionError("encoder loaded")) as load_model:
        with pytest.raises(RouterError, match="invalid modelHash metadata"):
            RouterCLI(args).route_prompt()
    assert load_model.call_count == 0


def test_legacy_signature_without_model_metadata_remains_compatible() -> None:
    legacy = LayeredSignature(
        facet="legacy",
        dimension=3,
        layers={},
        model_name="",
        model_hash=b"historical-hash",
    )
    validate_signature_model_compatibility("explicit/model", {"legacy": legacy})


def test_rm_artifact_load_path_keeps_portable_signature_identity() -> None:
    load_path, identity = resolve_model_binding(
        "E:/instance/artifacts/cas/model",
        signatures(),
        model_identity=MODEL_ID,
    )
    assert load_path == "E:/instance/artifacts/cas/model"
    assert identity == MODEL_ID


def test_auto_model_uses_signature_identity_as_load_name() -> None:
    assert resolve_model_binding("auto", signatures()) == (MODEL_ID, MODEL_ID)


def test_model_identity_mismatch_is_rejected() -> None:
    with pytest.raises(RouterError):
        resolve_model_binding(
            "E:/instance/artifacts/cas/model",
            signatures(),
            model_identity="different/model",
        )


def test_sign_auto_is_rejected_before_encoder_load(monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_cli(monkeypatch, "sign", "text", "--model", "auto")
    with patch("lsr.commands.load_model", side_effect=AssertionError("encoder loaded")) as load_model:
        with pytest.raises(RouterError, match="does not support --model auto"):
            RouterCLI(args).run()
    assert load_model.call_count == 0
