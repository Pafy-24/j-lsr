from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEQUENCE_LENGTH = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and quantize a SentenceTransformer encoder for the AMD Ryzen AI NPU."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--calibration", type=Path, action="append", default=[])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument(
        "--config",
        default="A16W8",
        help="AMD Quark configuration. Default: A16W8.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--online", action="store_true")
    return parser.parse_args()


def resolve_model_root(model_name: str, online: bool) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_name,
            local_files_only=not online,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.safetensors",
                "onnx/model.onnx",
            ],
        )
    )


def export_encoder(
    model_name: str,
    model_root: Path,
    output_path: Path,
    sequence_length: int,
    opset: int,
    online: bool,
) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_root,
        local_files_only=not online,
    )
    encoder = AutoModel.from_pretrained(
        model_root,
        local_files_only=not online,
    )
    encoder.eval()
    inputs = tokenizer(
        ["LSR calibration input"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
    )

    class Encoder(torch.nn.Module):
        def __init__(self, wrapped: Any):
            super().__init__()
            self.wrapped = wrapped

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Any,
        ) -> Any:
            return self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped_encoder = Encoder(encoder).eval()
    with torch.inference_mode():
        torch.onnx.export(
            wrapped_encoder,
            (
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["token_type_ids"],
            ),
            output_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"Exported static-sequence encoder: {output_path}")


def calibration_texts(paths: list[Path], limit: int) -> list[str]:
    import json

    texts: list[str] = []
    for raw_path in paths:
        candidates = sorted(raw_path.glob("*.jsonl")) if raw_path.is_dir() else [raw_path]
        for path in candidates:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                text = str(item.get("text") or "").strip()
                if not text and isinstance(item.get("messages"), list):
                    text = "\n".join(
                        str(message.get("content") or "").strip()
                        for message in item["messages"]
                        if isinstance(message, dict)
                    ).strip()
                if text:
                    texts.append(text)
                if len(texts) >= limit:
                    return texts
    if texts:
        return texts
    return [
        "Verify an ECDSA signature and explain nonce reuse.",
        "Configure an nginx reverse proxy for a FastAPI service.",
        "Explain AES-GCM authenticated encryption.",
        "Extract the main content and title from an HTML page.",
        "How does RSA public key verification work?",
        "Build a responsive web application with semantic HTML and CSS.",
        "Derive a shared secret using elliptic curve cryptography.",
        "Implement an HTTP API with authentication and structured JSON responses.",
    ]


class TokenCalibrationReader:
    def __init__(
        self,
        tokenizer: Any,
        texts: list[str],
        sequence_length: int,
    ):
        self._feeds = []
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=sequence_length,
            )
            self._feeds.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "token_type_ids": encoded["token_type_ids"],
                }
            )
        self._iterator = iter(self._feeds)

    def get_next(self) -> dict[str, Any] | None:
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = iter(self._feeds)


def main() -> int:
    args = parse_args()
    if args.sequence_length < 1:
        raise ValueError("--sequence-length must be at least 1")
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")

    # Quark's first import compiles optional kernels. Hiding CUDA makes this a
    # small CPU-only build and avoids coupling NPU preparation to nvcc.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    model_root = resolve_model_root(args.model, args.online)
    source_path = args.source or model_root / "onnx" / "model_npu_source_static.onnx"
    output_path = args.output or model_root / "onnx" / "model_npu_a16w8.onnx"
    if args.force or not source_path.exists():
        export_encoder(
            args.model,
            model_root,
            source_path,
            args.sequence_length,
            args.opset,
            args.online,
        )

    from quark.onnx import ModelQuantizer, QConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=not args.online)
    reader = TokenCalibrationReader(
        tokenizer,
        calibration_texts(args.calibration, args.samples)[: args.samples],
        args.sequence_length,
    )
    config = QConfig.get_default_config(args.config)
    global_config = getattr(config, "global_quant_config", config)
    global_config.enable_npu_transformer = True
    global_config.extra_options["CalibMovingAverage"] = True
    quantizer = ModelQuantizer(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_directory = Path.cwd()
    try:
        os.chdir(output_path.parent)
        quantizer.quantize_model(source_path, output_path, reader)
    finally:
        os.chdir(previous_directory)
    print(f"Prepared AMD NPU model: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
