from __future__ import annotations

import json
from types import SimpleNamespace

import router


def test_native_qualification_routes_twice_with_one_model_load(monkeypatch, tmp_path, capsys):
    model = tmp_path / "model"; signatures = tmp_path / "signatures"
    model.mkdir(); signatures.mkdir()
    args = SimpleNamespace(PU="0", large_input="off")
    monkeypatch.setattr(router, "parse_args", lambda: args)
    calls = {"model": 0, "route": 0}

    class FakeRouter:
        def route(self, text, embedding):
            calls["route"] += 1
            return {"selectedFacets": ["smoke.code"], "results": [{"facet": "smoke.code"}], "text": text}

    class FakeCLI:
        def __init__(self, _args): pass
        def _local_registry_router(self): return object(), FakeRouter(), str(model)
        def _local_strategy(self, registry, route): return None

    def load(*_args): calls["model"] += 1; return object()
    monkeypatch.setattr(router, "RouterCLI", FakeCLI)
    monkeypatch.setattr(router, "load_model", load)
    monkeypatch.setattr(router, "embed_texts", lambda _model, units: [[1.0]])
    monkeypatch.setattr(router, "prepare_routing_input", lambda prompt, mode: SimpleNamespace(routing_text=prompt, routing_units=[prompt]))
    monkeypatch.setattr(router, "json_ready_route", lambda value: value)

    rc = router.native_qualification(["--model", str(model), "--model-identity", "sentence-transformers/all-MiniLM-L6-v2",
                                      "--signatures", str(signatures), "--prompt", "Write a Python function that returns the sum of two integers."])
    receipt = json.loads(capsys.readouterr().out)
    assert rc == 0 and calls == {"model": 1, "route": 2}
    assert len(receipt["requests"]) == 2
    assert receipt["startupMs"] <= receipt["totalMs"]
    assert all(row["route"]["results"][0]["facet"] == "smoke.code" for row in receipt["requests"])
