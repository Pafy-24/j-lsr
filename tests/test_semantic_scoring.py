from __future__ import annotations

import unittest

import numpy as np

from lsr.models import LayeredSignature, SignatureLayer
from lsr.embedding import model_tensor_bytes_by_device
from lsr.input_digest import prepare_routing_input
from lsr.routing import LSRRouter
from lsr.scoring import lexical_score
from lsr.profiling import RankProfiler


class SemanticScoringTests(unittest.TestCase):
    def test_short_mixed_prompt_produces_bounded_semantic_units(self) -> None:
        routing_input = prepare_routing_input(
            "Build a React website about RSA, ECDSA and post-quantum cryptography."
        )
        self.assertGreaterEqual(len(routing_input.routing_units), 2)
        self.assertLessEqual(len(routing_input.routing_units), 6)
        self.assertEqual(routing_input.routing_units[0], routing_input.routing_text)

    def test_unit_max_can_activate_two_facets_without_extra_layer_loads(self) -> None:
        queries = np.asarray([[0.7, 0.7], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        signatures = {
            "first": self._signature("first", [[1.0, 0.0], [0.9, 0.1]]),
            "second": self._signature("second", [[0.0, 1.0], [0.1, 0.9]]),
        }
        route = LSRRouter(signatures, top_k=2).route("mixed", queries)
        self.assertEqual(route["queryUnits"], 3)
        self.assertTrue(all(result["decision"] != "refused" for result in route["results"]))
        self.assertEqual(route["signatureMemory"]["byLevel"]["1"]["loads"], 2)

    def test_rank_profiler_reports_time_without_process_resource_sampling(self) -> None:
        profiler = RankProfiler(sample_interval_seconds=0.001)
        profiler.start()
        started = __import__("time").perf_counter()
        __import__("time").sleep(0.003)
        profiler.stage("routing", started)
        result = profiler.finish()
        self.assertGreaterEqual(result["timing"]["rankingMs"], 3.0)
        self.assertNotIn("resources", result)
        self.assertNotIn("cpuTimeMs", result["timing"])

    def test_model_memory_is_counted_by_actual_tensor_device(self) -> None:
        class Tensor:
            def __init__(self, count: int, width: int, device: str):
                self.count, self.width, self.device = count, width, device

            def nelement(self) -> int:
                return self.count

            def element_size(self) -> int:
                return self.width

        class Model:
            def parameters(self):
                return iter([Tensor(10, 4, "cpu"), Tensor(20, 2, "cuda:0")])

            def buffers(self):
                return iter([Tensor(3, 4, "cuda:0")])

        self.assertEqual(model_tensor_bytes_by_device(Model()), {"cpu": 40, "cuda:0": 52})

    def test_lexical_vocabulary_does_not_change_score(self) -> None:
        self.assertEqual(lexical_score("RSA ECDSA React HTML", "crypto"), 0.0)
        self.assertEqual(lexical_score("RSA ECDSA React HTML", "web"), 0.0)

    def test_mixed_input_can_keep_two_facets_active(self) -> None:
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        signatures = {
            "crypto": self._signature("crypto", [[0.90, 0.10], [0.85, 0.15]]),
            "web": self._signature("web", [[0.88, 0.12], [0.82, 0.18]]),
        }
        route = LSRRouter(signatures, top_k=2).route("mixed prompt", query)
        decisions = {result["facet"]: result["decision"] for result in route["results"]}
        self.assertNotEqual(decisions["crypto"], "refused")
        self.assertNotEqual(decisions["web"], "refused")
        self.assertEqual(route["signatureMemory"]["byLevel"]["1"]["loads"], 2)
        self.assertEqual(route["signatureMemory"]["residentVectorBytesAfterRank"], 0)

    def test_selected_facets_have_independent_fractional_weights(self) -> None:
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        signatures = {
            "crypto": self._signature("crypto", [[0.90, 0.10], [0.86, 0.14]]),
            "web": self._signature("web", [[0.88, 0.12], [0.84, 0.16]]),
        }

        route = LSRRouter(signatures, top_k=2).route("mixed", query)
        weights = [item["weight"] for item in route["selectedFacets"]]

        self.assertEqual(route["schemaVersion"], "lsr-route-v2")
        self.assertEqual(route["weightSemantics"], "independent-pass-to-max-v1")
        self.assertEqual({item["facet"] for item in route["selectedFacets"]}, {"crypto", "web"})
        self.assertTrue(all(0.0 < weight < 1.0 for weight in weights))
        self.assertGreater(sum(weights), 1.0)
        self.assertTrue(all(0.0 <= result["weight"] <= 1.0 for result in route["results"]))

    def test_all_refused_facets_produce_an_empty_selection(self) -> None:
        route = LSRRouter(
            {"crypto": self._signature("crypto", [[1.0, 0.0], [0.9, 0.1]])},
            top_k=1,
        ).route("unrelated", np.asarray([-1.0, 0.0], dtype=np.float32))

        self.assertEqual(route["selectedFacets"], [])
        self.assertEqual(route["results"][0]["decision"], "refused")
        self.assertEqual(route["results"][0]["weight"], 0.0)

    def test_vector_memory_is_registered_and_released(self) -> None:
        events: list[tuple[object, ...]] = []
        signature = self._signature("crypto", [[0.90, 0.10], [0.85, 0.15]])
        router = LSRRouter(
            {"crypto": signature},
            memory_callback=lambda *event: events.append(event),
        )
        router.route("prompt", np.asarray([1.0, 0.0], dtype=np.float32))
        sets = [event for event in events if event[0] == "set"]
        removals = [event for event in events if event[0] == "remove"]
        self.assertTrue(any(event[2] == "signature:crypto:L1" and event[3] == 16 for event in sets))
        self.assertTrue(any(event[2] == "signature:crypto:L1" for event in removals))

    @staticmethod
    def _signature(facet: str, vectors: list[list[float]]) -> LayeredSignature:
        matrix = np.asarray(vectors, dtype=np.float32)
        return LayeredSignature(
            facet=facet,
            dimension=2,
            layers={1: SignatureLayer(1, matrix, [])},
            layer_sizes={1: len(matrix)},
        )


if __name__ == "__main__":
    unittest.main()
