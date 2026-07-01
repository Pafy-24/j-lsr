from __future__ import annotations

from .errors import RouterError
from .models import LayeredSignature


def resolve_model_name(requested_model: str, signatures: dict[str, LayeredSignature]) -> str:
    if requested_model != "auto":
        return requested_model
    if not signatures:
        raise RouterError(
            "--model auto requires loaded v3 signatures with modelName metadata. "
            "Pass --model explicitly when starting without signatures."
        )
    model_names = sorted(
        {
            signature.model_name.strip()
            for signature in signatures.values()
            if signature.model_name and signature.model_name.strip()
        }
    )
    if len(model_names) == 1:
        return model_names[0]
    if not model_names:
        raise RouterError(
            "--model auto requires v3 signatures with modelName metadata. "
            "Re-sign with the current CLI or pass --model explicitly."
        )
    raise RouterError(
        "--model auto found signatures created with multiple models: "
        f"{', '.join(model_names)}. Multi-model routing is not supported in v1."
    )


def validate_signature_model_compatibility(
    model_name: str,
    signatures: dict[str, LayeredSignature],
) -> None:
    if model_name == "auto":
        return
    mismatched = sorted(
        signature.facet
        for signature in signatures.values()
        if signature.model_name and signature.model_name != model_name
    )
    if mismatched:
        raise RouterError(
            f"Loaded signatures for {', '.join(mismatched)} were created with a different model. "
            "Use --model auto or re-sign with the requested model."
        )
