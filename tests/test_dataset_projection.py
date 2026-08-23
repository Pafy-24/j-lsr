from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lsr.dataset_schema import load_lsr_dataset
from lsr.routing_context import agent_event_to_text, facet_record_to_routing_text
from lsr.server import extract_prompt


class DatasetProjectionTests(unittest.TestCase):
    def test_canonical_record_uses_training_input_without_target_or_labels(self) -> None:
        record = {
            "schemaVersion": "facet-record-v3",
            "id": "coder_python_train_0001",
            "facet": "coder.python",
            "split": "train",
            "stage": "sft",
            "input": {
                "agentEvent": {
                    "scope": "step",
                    "phase": "construct",
                    "objective": "Implement the approved parser design.",
                    "requestedOperation": "Produce a Python implementation and tests.",
                    "artifact": {"type": "source-code", "technology": "python"},
                },
                "messages": [{"role": "user", "content": "Support Python 3.11."}],
                "conditioningMessages": [
                    {"role": "system", "content": "SECRET coder.python CONDITIONING"}
                ],
            },
            "target": {
                "messages": [
                    {"role": "assistant", "content": "SECRET MODEL TARGET"}
                ]
            },
            "activation": {
                "scope": "step",
                "primaryFacet": "coder.python",
                "activeFacets": ["coder", "coder.python"],
                "routeFrom": "input.agentEvent",
            },
            "coverage": {"language": "en", "domainContext": "software"},
            "provenance": {"source": "synthetic"},
        }

        route_text = facet_record_to_routing_text(record)

        self.assertIn("Python implementation", route_text)
        self.assertNotIn("Support Python 3.11", route_text)
        self.assertNotIn("SECRET MODEL TARGET", route_text)
        self.assertNotIn("SECRET coder.python CONDITIONING", route_text)
        self.assertNotIn("coder.python", route_text)

    def test_canonical_directory_signing_excludes_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, split, marker in (
                ("pretraining.jsonl", "train", "TRAIN PRETRAIN"),
                ("training.jsonl", "train", "TRAIN SFT"),
                ("test.jsonl", "test", "HELD OUT TEST"),
            ):
                stage = "continued-pretraining" if filename == "pretraining.jsonl" else "sft"
                row = {
                    "schemaVersion": "facet-record-v3",
                    "id": marker.lower().replace(" ", "_"),
                    "facet": "planner",
                    "split": split,
                    "stage": stage,
                    "input": {"content": marker},
                    "target": {"objective": "causal-lm"},
                    "activation": {
                        "scope": "step",
                        "primaryFacet": "planner",
                        "activeFacets": ["planner"],
                        "routeFrom": "input.content",
                    },
                    "coverage": {"language": "en", "domainContext": "general"},
                    "provenance": {"source": "test"},
                }
                (root / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")

            items = load_lsr_dataset(root)

        self.assertEqual([item.texts[0] for item in items], ["TRAIN PRETRAIN", "TRAIN SFT"])

    def test_legacy_sft_projection_excludes_assistant_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planner_core.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "planner_core_0001",
                        "messages": [
                            {"role": "user", "content": "Build a safe plan."},
                            {"role": "assistant", "content": "LEAKED TARGET"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            item = load_lsr_dataset(path)[0]

        self.assertIn("Build a safe plan", item.texts[0])
        self.assertNotIn("LEAKED TARGET", item.texts[0])

    def test_server_accepts_agent_event_as_the_routing_request(self) -> None:
        payload = {
            "agentEvent": {
                "scope": "step",
                "phase": "verify",
                "objective": "Decide whether the patch is ready.",
                "observation": {"kind": "test-run", "summary": "Two tests failed."},
            },
            "taskContext": {"language": "ro", "domainContext": "software"},
        }

        projected = extract_prompt(payload)

        self.assertEqual(projected, agent_event_to_text(payload["agentEvent"], payload["taskContext"]))
        self.assertIn("Agent phase: verify", projected)
        self.assertIn("Two tests failed", projected)


if __name__ == "__main__":
    unittest.main()
