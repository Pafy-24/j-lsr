from __future__ import annotations

import io
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from lsr.config import DEFAULT_SERVER_MAX_BATCH_SIZE, DEFAULT_SERVER_MAX_REQUEST_BYTES
from lsr.embedding import EmbeddingBatcher
from lsr.errors import RouterError
from lsr.memory import current_process_memory_bytes
from lsr.models import LayeredSignature, SignatureLayer
from lsr.routing import LSRRouter
from lsr.registry import SignatureRegistry
from lsr.server import LSRRequestHandler, LSRRouteService, extract_prompts
from lsr.signatures import matrix_to_octets, octets_to_matrix


class ResourceEfficiencyTests(unittest.TestCase):
    def test_process_memory_snapshot_is_non_negative(self) -> None:
        working_set, private = current_process_memory_bytes()
        self.assertGreaterEqual(working_set, 0)
        self.assertGreaterEqual(private, 0)

    def test_signature_decode_returns_normalized_writable_matrix(self) -> None:
        source = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        decoded = octets_to_matrix(matrix_to_octets(source), dimension=2)

        self.assertEqual(decoded.dtype, np.float32)
        self.assertTrue(decoded.flags.c_contiguous)
        self.assertTrue(decoded.flags.writeable)
        np.testing.assert_allclose(decoded[0], [0.6, 0.8], rtol=1e-6)
        np.testing.assert_allclose(decoded[1], [0.0, 0.0], rtol=1e-6)

    def test_separation_is_linear_and_does_not_reorder_results(self) -> None:
        router = LSRRouter({"facet": self._signature()})
        results = [
            {"facet": "low", "score": 0.25},
            {"facet": "high", "score": 0.75},
            {"facet": "middle", "score": 0.50},
        ]

        router.apply_separation(results)

        self.assertEqual([item["facet"] for item in results], ["low", "high", "middle"])
        self.assertAlmostEqual(results[0]["separation"], 0.50)
        self.assertAlmostEqual(results[1]["separationRatio"], 1.0)

    def test_full_depth_override_does_not_mutate_router_default(self) -> None:
        router = LSRRouter({"facet": self._signature()}, full_depth=False)
        route = router.route(
            "prompt",
            np.asarray([1.0, 0.0], dtype=np.float32),
            full_depth=True,
        )

        self.assertTrue(route["fullDepth"])
        self.assertFalse(router.full_depth)

    def test_batch_limit_is_enforced_before_prompt_processing(self) -> None:
        with self.assertRaisesRegex(RouterError, "Batch exceeds"):
            extract_prompts(["prompt"] * (DEFAULT_SERVER_MAX_BATCH_SIZE + 1))

    def test_request_body_size_limit_is_enforced_before_read(self) -> None:
        handler = object.__new__(LSRRequestHandler)
        handler.headers = {
            "Content-Length": str(DEFAULT_SERVER_MAX_REQUEST_BYTES + 1)
        }
        handler.rfile = io.BytesIO(b"")

        with self.assertRaisesRegex(RouterError, "exceeds"):
            handler._read_json()

    def test_request_content_length_must_be_an_integer(self) -> None:
        handler = object.__new__(LSRRequestHandler)
        handler.headers = {"Content-Length": "invalid"}
        handler.rfile = io.BytesIO(b"")

        with self.assertRaisesRegex(RouterError, "must be an integer"):
            handler._read_json()

    def test_request_body_must_be_utf8(self) -> None:
        handler = object.__new__(LSRRequestHandler)
        handler.headers = {"Content-Length": "1"}
        handler.rfile = io.BytesIO(b"\xff")

        with self.assertRaisesRegex(RouterError, "must be UTF-8"):
            handler._read_json()

    def test_signature_refresh_checks_are_throttled(self) -> None:
        service = object.__new__(LSRRouteService)
        service.signature_refresh_interval_seconds = 1.0
        service._last_signature_refresh = 0.0
        service._refresh_lock = threading.Lock()
        service._state_lock = threading.RLock()
        service.registry = Mock()
        service.registry.refresh.return_value = False

        with patch("lsr.server.time.monotonic", side_effect=[10.0, 10.5, 11.1]):
            service.refresh_signatures()
            service.refresh_signatures()
            service.refresh_signatures()

        self.assertEqual(service.registry.refresh.call_count, 2)

    def test_embedding_batcher_coalesces_concurrent_requests(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []
                self.lock = threading.Lock()

            def encode(self, texts, **kwargs):
                with self.lock:
                    self.calls.append(list(texts))
                return np.asarray(
                    [[float(len(text)), 1.0] for text in texts],
                    dtype=np.float32,
                )

        model = Model()
        batcher = EmbeddingBatcher(model, max_batch_units=8, max_wait_ms=50.0)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(batcher.embed, ["one"])
                second = executor.submit(batcher.embed, ["two", "three"])
                self.assertEqual(first.result().shape, (1, 2))
                self.assertEqual(second.result().shape, (2, 2))
        finally:
            batcher.close()

        self.assertEqual(len(model.calls), 1)
        self.assertCountEqual(model.calls[0], ["one", "two", "three"])

    def test_lazy_signature_routing_copies_have_independent_layer_caches(self) -> None:
        source = self._signature()
        source.layer_loader = lambda level: source.layers[level]
        first = source.routing_copy()
        second = source.routing_copy()

        first.get_layer(1)

        self.assertIn(1, first.layers)
        self.assertNotIn(1, second.layers)

    def test_explicitly_unloaded_facet_stays_disabled_during_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            signature_path = Path(directory) / "web.sig"
            signature_path.write_bytes(b"placeholder")
            registry = SignatureRegistry(directory)
            registry.signatures["web"] = self._signature()
            registry.files["web"] = signature_path
            registry.file_stats["web"] = (
                signature_path.stat().st_mtime,
                signature_path.stat().st_size,
            )
            registry.unload("web", disable=True)
            registry.codec.load = Mock(side_effect=AssertionError("must stay unloaded"))

            changed = registry.refresh()

            self.assertFalse(changed)
            self.assertEqual(registry.signatures, {})
            self.assertEqual(registry.disabled_facets, {"web"})

    @staticmethod
    def _signature() -> LayeredSignature:
        matrix = np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
        return LayeredSignature(
            facet="facet",
            dimension=2,
            layers={1: SignatureLayer(1, matrix, [])},
            layer_sizes={1: len(matrix)},
        )


if __name__ == "__main__":
    unittest.main()
