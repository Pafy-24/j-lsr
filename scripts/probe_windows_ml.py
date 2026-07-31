from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lsr.embedding import load_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Sustained Windows ML embedding probe.")
    parser.add_argument("--PU", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = parser.parse_args()

    model = load_model(args.model, args.PU, online=False)
    started = time.perf_counter()
    for _ in range(max(1, args.iterations)):
        model.encode(
            ["ECDSA nonce reuse and nginx reverse proxy configuration"],
            batch_size=1,
        )
    elapsed = time.perf_counter() - started
    print(
        f"{model.device}: {max(1, args.iterations)} embeddings in "
        f"{elapsed:.3f}s ({elapsed * 1000 / max(1, args.iterations):.2f} ms/item)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
