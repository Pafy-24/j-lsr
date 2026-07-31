from __future__ import annotations

import unittest

import numpy as np

from lsr.windows_ml import WindowsMLEmbeddingModel, _device_payload


class _Input:
    def __init__(self, name: str):
        self.name = name


class _Tokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, np.ndarray]:
        batch = len(texts)
        return {
            "input_ids": np.ones((batch, 4), dtype=np.int64),
            "attention_mask": np.asarray([[1, 1, 0, 0]] * batch, dtype=np.int64),
            "token_type_ids": np.zeros((batch, 4), dtype=np.int64),
        }


class _Session:
    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self) -> list[_Input]:
        return [_Input("input_ids"), _Input("attention_mask"), _Input("token_type_ids")]

    def run(self, _: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls += 1
        batch = feeds["input_ids"].shape[0]
        tokens = np.zeros((batch, 4, 2), dtype=np.float32)
        tokens[:, 0, :] = [3.0, 0.0]
        tokens[:, 1, :] = [1.0, 0.0]
        return [tokens]


class WindowsMLTests(unittest.TestCase):
    def test_fixed_batch_model_serializes_inputs_and_mean_pools(self) -> None:
        model = WindowsMLEmbeddingModel.__new__(WindowsMLEmbeddingModel)
        model.fixed_batch_size = 1
        model.fixed_sequence_length = 4
        model.tokenizer = _Tokenizer()
        model.session = _Session()

        embeddings = model.encode(["first", "second"], batch_size=64)

        self.assertEqual(model.session.calls, 2)
        self.assertEqual(embeddings.shape, (2, 2))
        np.testing.assert_allclose(embeddings, [[1.0, 0.0], [1.0, 0.0]])

    def test_ort_device_payload_converts_decimal_luid_to_hex(self) -> None:
        hardware = type(
            "Hardware",
            (),
            {
                "metadata": {
                    "Description": "AMD Radeon(TM) 880M Graphics",
                    "LUID": "99036",
                },
                "vendor": "Advanced Micro Devices, Inc.",
                "type": "OrtHardwareDeviceType.GPU",
            },
        )()
        ep_device = type(
            "EpDevice",
            (),
            {"device": hardware, "ep_name": "DmlExecutionProvider"},
        )()

        payload = _device_payload(ep_device)

        self.assertEqual(payload["luid"], "182DC")
        self.assertEqual(payload["provider"], "DmlExecutionProvider")


if __name__ == "__main__":
    unittest.main()
