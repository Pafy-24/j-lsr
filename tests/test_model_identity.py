from __future__ import annotations

import pytest

from lsr.errors import RouterError
from lsr.model_selection import resolve_model_binding
from lsr.models import LayeredSignature


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def signatures() -> dict[str, LayeredSignature]:
    return {
        "coder": LayeredSignature(
            facet="coder",
            dimension=384,
            layers={},
            model_name=MODEL_ID,
        )
    }


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
