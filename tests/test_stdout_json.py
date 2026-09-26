from __future__ import annotations

import io
import json
from types import SimpleNamespace
import sys

import pytest

import lsr.commands as commands


def _cp1250_stdout_bytes(monkeypatch: pytest.MonkeyPatch, action) -> bytes:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1250", errors="strict", newline="\n")
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "stdout", stream)
        action()
        stream.flush()
        return raw.getvalue()


class _Profiler:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    def start(self) -> None:
        pass

    def stage(self, name: str, started: float) -> None:
        self.stages[name] = started

    def finish(self) -> dict[str, object]:
        return {}


class _Router:
    def route(self, prompt: str, embedding: object) -> dict[str, object]:
        return {
            "input": prompt,
            "compared_facets": 1,
            "deep_facets": [],
            "results": [
                {
                    "facet": "verifier",
                    "score": 0.5,
                    "best_vector_score": 0.5,
                    "cluster_score": 0.0,
                    "exact_count": 1,
                    "mode": "L1 passed",
                    "vector_index": 0,
                }
            ],
        }


def test_local_rank_json_roundtrips_romanian_on_cp1250_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = "Verifică ș și ț"
    routing_input = SimpleNamespace(
        routing_text=prompt,
        routing_units=[prompt],
        input_mode="raw",
        input_chars=len(prompt),
        routing_text_chars=len(prompt),
        chunks_used=1,
    )
    args = SimpleNamespace(
        input_path=None,
        text=prompt,
        large_input="off",
        PU=0,
        online=False,
        strategy_operation="route",
        json=True,
    )
    cli = commands.RouterCLI(args)
    monkeypatch.setattr(cli, "_local_registry_router", lambda: (object(), _Router(), "physical-model"))
    monkeypatch.setattr(cli, "_local_strategy", lambda registry, router: None)
    monkeypatch.setattr(commands, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(commands, "prepare_routing_input", lambda *args, **kwargs: routing_input)
    monkeypatch.setattr(commands, "embed_texts", lambda *args, **kwargs: object())
    monkeypatch.setattr(commands, "RankProfiler", _Profiler)

    encoded = _cp1250_stdout_bytes(monkeypatch, cli.route_prompt)

    assert encoded.isascii()
    decoded = json.loads(encoded.decode("utf-8"))
    assert decoded["input"] == prompt


def test_json_stdout_avoids_cp1250_bytes_that_are_not_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"message": "plan A–B"}
    legacy_raw = io.BytesIO()
    legacy_stream = io.TextIOWrapper(legacy_raw, encoding="cp1250", errors="strict", newline="\n")
    legacy_stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    legacy_stream.flush()
    legacy_bytes = legacy_raw.getvalue()
    assert b"\x96" in legacy_bytes
    with pytest.raises(UnicodeDecodeError):
        legacy_bytes.decode("utf-8")

    encoded = _cp1250_stdout_bytes(monkeypatch, lambda: commands._print_json_stdout(payload))

    assert encoded.isascii()
    assert b"\x96" not in encoded
    assert json.loads(encoded.decode("utf-8")) == payload
