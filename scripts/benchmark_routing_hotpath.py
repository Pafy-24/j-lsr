from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


LSR_ROOT = Path(__file__).resolve().parents[1]
if str(LSR_ROOT) not in sys.path:
    sys.path.insert(0, str(LSR_ROOT))

from lsr.routing import LSRRouter
from lsr.signatures import SignatureStore


def percentile(samples: list[float], value: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(value * len(ordered)) - 1))
    return ordered[index]


def route_once(
    signatures: dict[str, Any],
    query: Any,
    *,
    top_k: int,
    index_policy: str,
    full_depth: bool,
) -> tuple[float, dict[str, Any]]:
    request_signatures = {
        facet: signature.routing_copy() for facet, signature in signatures.items()
    }
    router = LSRRouter(
        request_signatures,
        top_k=top_k,
        index_policy=index_policy,
        full_depth=full_depth,
    )
    started = time.perf_counter()
    route = router.route(
        "route-only anchor benchmark",
        query,
        full_depth=full_depth,
    )
    return (time.perf_counter() - started) * 1000.0, route


def benchmark_mode(args: argparse.Namespace, *, full_depth: bool) -> dict[str, Any]:
    signatures = SignatureStore(args.signatures).load_all(required=True)
    if args.anchor_facet not in signatures:
        available = ", ".join(sorted(signatures))
        raise ValueError(
            f"Unknown --anchor-facet {args.anchor_facet!r}; available: {available}"
        )
    source = signatures[args.anchor_facet]
    deepest_level = max(source.levels())
    deepest = source.get_layer(deepest_level)
    try:
        query = deepest.vectors[:1].copy()
    finally:
        source.release_layer(deepest_level)

    cold_ms, _ = route_once(
        signatures,
        query,
        top_k=args.top_k,
        index_policy=args.index,
        full_depth=full_depth,
    )
    for _ in range(args.warmup):
        route_once(
            signatures,
            query,
            top_k=args.top_k,
            index_policy=args.index,
            full_depth=full_depth,
        )

    samples: list[float] = []
    route: dict[str, Any] | None = None
    for _ in range(args.iterations):
        elapsed, route = route_once(
            signatures,
            query,
            top_k=args.top_k,
            index_policy=args.index,
            full_depth=full_depth,
        )
        samples.append(elapsed)
    assert route is not None
    memory = route["signatureMemory"]
    return {
        "fullDepth": full_depth,
        "coldMs": round(cold_ms, 3),
        "steadyState": {
            "iterations": len(samples),
            "minimumMs": round(min(samples), 3),
            "medianMs": round(statistics.median(samples), 3),
            "meanMs": round(statistics.mean(samples), 3),
            "p95Ms": round(percentile(samples, 0.95), 3),
            "maximumMs": round(max(samples), 3),
        },
        "route": {
            "indexBackend": route["indexBackend"],
            "vectorsScanned": route["vectorsScanned"],
            "layerLoadCount": memory["layerLoadCount"],
            "totalLayerBytesRead": memory["totalLayerBytesRead"],
            "peakResidentVectorBytes": memory["peakResidentVectorBytes"],
            "persistentTopologyCacheBytes": memory["persistentTopologyCacheBytes"],
            "persistentTopologyCacheLevels": memory["persistentTopologyCacheLevels"],
            "peakResidentSignatureBytes": memory["peakResidentSignatureBytes"],
            "selectedFacets": route["selectedFacets"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark only the LSR routing hot path using an anchor vector from an "
            "existing signature; embedding latency is deliberately excluded."
        )
    )
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--anchor-facet", default="coder")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--index", choices=("auto", "flat", "hnsw"), default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    payload = {
        "schemaVersion": "lsr-routing-hotpath-benchmark-v1",
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "configuration": {
            "signatures": str(Path(args.signatures).resolve()),
            "anchorFacet": args.anchor_facet,
            "topK": args.top_k,
            "indexPolicy": args.index,
            "warmupIterations": args.warmup,
            "measuredIterations": args.iterations,
            "embeddingIncluded": False,
        },
        "modes": {
            "normal": benchmark_mode(args, full_depth=False),
            "fullDepth": benchmark_mode(args, full_depth=True),
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
