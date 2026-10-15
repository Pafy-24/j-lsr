from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import router


def test_native_qualification_routes_two_distinct_inputs_with_one_model_load(monkeypatch, tmp_path, capsys):
    model = tmp_path / "model"; signatures = tmp_path / "signatures"
    model.mkdir(); signatures.mkdir()
    args = SimpleNamespace(PU="0", large_input="off")
    monkeypatch.setattr(router, "parse_args", lambda: args)
    calls = {"model": [], "route": 0}

    class FakeRouter:
        def route(self, text, embedding):
            calls["route"] += 1
            return {"modelName": "sentence-transformers/all-MiniLM-L6-v2", "selectedFacets": [{"facet": "smoke.code", "weight": 1.0}], "results": [{"facet": "smoke.code"}], "text": text}

    class FakeCLI:
        def __init__(self, _args): pass
        def _local_registry_router(self): return object(), FakeRouter(), str(model)
        def _local_strategy(self, registry, route): return None

    def load(*load_args): calls["model"].append(load_args); return object()
    monkeypatch.setattr(router, "RouterCLI", FakeCLI)
    monkeypatch.setattr(router, "load_model", load)
    monkeypatch.setattr(router, "embed_texts", lambda _model, units: [[1.0]])
    monkeypatch.setattr(router, "prepare_routing_input", lambda prompt, mode: SimpleNamespace(routing_text=prompt, routing_units=[prompt]))
    monkeypatch.setattr(router, "json_ready_route", lambda value: value)

    prompts = [
        "Write a Python function that returns the sum of two integers.",
        "Write a Python function that reverses a list.",
    ]
    rc = router.native_qualification(["--model", str(model), "--model-identity", "sentence-transformers/all-MiniLM-L6-v2",
                                      "--signatures", str(signatures), "--prompt", prompts[0], "--prompt", prompts[1]])
    receipt = json.loads(capsys.readouterr().out)
    assert rc == 0 and calls["route"] == 2 and calls["model"] == [(str(model.resolve()), "0", False)]
    assert len(receipt["requests"]) == 2
    assert [(row["index"], row["input"]) for row in receipt["requests"]] == list(enumerate(prompts))
    assert receipt["startupMs"] <= receipt["totalMs"]
    assert receipt["pid"] > 0
    assert all(row["route"]["results"][0]["facet"] == "smoke.code" for row in receipt["requests"])
    assert all(row["route"]["selectedFacets"][0]["facet"] == "smoke.code" for row in receipt["requests"])
    assert all(row["route"]["modelName"] == "sentence-transformers/all-MiniLM-L6-v2" for row in receipt["requests"])


@pytest.mark.parametrize("prompts", [
    [],
    ["one"],
    ["one", "two", "three"],
    ["", "two"],
    ["   ", "two"],
    ["same", "same"],
])
def test_native_qualification_rejects_invalid_prompt_pairs(tmp_path, prompts):
    model = tmp_path / "model"; signatures = tmp_path / "signatures"
    model.mkdir(); signatures.mkdir()
    argv = ["--model", str(model), "--model-identity", "model", "--signatures", str(signatures)]
    for prompt in prompts:
        argv.extend(["--prompt", prompt])
    with pytest.raises(SystemExit, match="2"):
        router.native_qualification(argv)
