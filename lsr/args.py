from __future__ import annotations

import argparse

from .config import (
    DEFAULT_BENCHMARK_DATASET,
    DEFAULT_BENCHMARK_INPUT,
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_LARGE_INPUT_MODE,
    DEFAULT_MAX_VECTORS,
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_MODEL,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_SERVER_URL,
    DEFAULT_SIGNATURES,
    DEFAULT_TOP_K,
    LAYER_ACCEPT_THRESHOLD,
    LAYER_DESCEND_THRESHOLD,
)


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow model downloads/checks. By default LSR uses local model files only.",
    )
    parser.add_argument(
        "--PU",
        help="Processing unit index to use for embedding operations. See --PUs.",
    )


def add_index_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index",
        choices=("auto", "flat", "hnsw"),
        default="auto",
        help="Vector index backend. Default: auto",
    )


def add_loading_policy_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memory-limit-mb",
        type=float,
        default=DEFAULT_MEMORY_LIMIT_MB,
        help=(
            "Maximum total RAM/VRAM footprint allowed before loading a routing layer. "
            f"0 means unlimited. Default: {DEFAULT_MEMORY_LIMIT_MB:g}"
        ),
    )
    parser.add_argument(
        "--descend-threshold",
        type=float,
        default=None,
        help=(
            "Override all per-layer pass thresholds from defaults.json. "
            f"Default config fallback: {LAYER_DESCEND_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=None,
        help=(
            "Override all per-layer accept thresholds from defaults.json. "
            f"Default config fallback: {LAYER_ACCEPT_THRESHOLD}"
        ),
    )


def add_input_policy_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--large-input",
        choices=("auto", "off", "force"),
        default=DEFAULT_LARGE_INPUT_MODE,
        help=f"Large input routing mode. Default: {DEFAULT_LARGE_INPUT_MODE}",
    )


def add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ascii-table",
        action="store_true",
        help="Render CLI tables with ASCII-safe characters.",
    )


def add_signature_path_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--signatures",
        "-s",
        default=DEFAULT_SIGNATURES,
        help=(
            "Signature file or directory. Facet signatures are stored as "
            "[facet].sig ASN.1 DER files. Default: signatures"
        ),
    )


def add_server_target_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_URL,
        help=f"Running LSR server URL. Default: {DEFAULT_SERVER_URL}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LSR (Layered Signatures Router) command line interface."
    )
    parser.add_argument(
        "--PUs",
        action="store_true",
        help="List available processing units and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser(
        "serve",
        help="Start an HTTP routing server that keeps model and signatures loaded.",
    )
    add_signature_path_option(serve)
    add_runtime_options(serve)
    add_index_option(serve)
    add_loading_policy_options(serve)
    serve.add_argument("--host", default=DEFAULT_SERVER_HOST, help=f"Host address. Default: {DEFAULT_SERVER_HOST}")
    serve.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT, help=f"Port. Default: {DEFAULT_SERVER_PORT}")
    serve.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Default result count. Default: {DEFAULT_TOP_K}")
    serve.add_argument(
        "--no-load",
        action="store_true",
        help="Start without loading initial signatures. Use loadsig later.",
    )
    serve.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable the real-time processing-unit and memory dashboard.",
    )
    serve.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Bounded HTTP worker threads. Requests are serialized around the shared router. Default: 4",
    )

    loadsig = subparsers.add_parser(
        "loadsig",
        help="Load .sig files into a running LSR server.",
    )
    add_signature_path_option(loadsig)
    add_server_target_option(loadsig)

    unloadsig = subparsers.add_parser(
        "unloadsig",
        help="Unload signatures from a running LSR server.",
    )
    add_server_target_option(unloadsig)
    unloadsig.add_argument("--facet", "-f", action="append", help="Facet to unload. Repeatable.")
    unloadsig.add_argument("--all", action="store_true", help="Unload all loaded signatures.")

    sign = subparsers.add_parser("sign", help="Generate or update LSR signatures.")
    sign.add_argument(
        "text",
        nargs="*",
        help="Raw text to sign. Quote multi-word prompts.",
    )
    sign.add_argument(
        "-in",
        "--input",
        dest="input_path",
        help="Input file path. For dataset signing this must be JSON.",
    )
    sign.add_argument(
        "--facet",
        help="Facet name. Optional when it can be inferred from a faucet path.",
    )
    add_signature_path_option(sign)
    add_runtime_options(sign)
    sign.add_argument(
        "--max-vectors",
        type=int,
        default=DEFAULT_MAX_VECTORS,
        help=f"Maximum vectors to store per faucet signature. Default: {DEFAULT_MAX_VECTORS}",
    )
    sign.add_argument("--debug", action="store_true", help="Print extra details.")

    rank = subparsers.add_parser("rank", help="Rank loaded signatures against prompt text.")
    rank.add_argument(
        "text",
        nargs="*",
        help="Raw prompt text. Quote multi-word prompts.",
    )
    rank.add_argument(
        "-in",
        "--input",
        dest="input_path",
        help="Input file path. With --batch this must be JSON/JSONL/text prompts.",
    )
    add_signature_path_option(rank)
    add_runtime_options(rank)
    add_index_option(rank)
    add_loading_policy_options(rank)
    add_input_policy_options(rank)
    add_server_target_option(rank)
    rank.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Compatibility/display limit. Routing depth is controlled by per-layer "
            f"thresholds. Default: {DEFAULT_TOP_K}"
        ),
    )
    rank.add_argument(
        "--full-depth",
        action="store_true",
        help="Evaluate every layer even if an earlier layer accepted the facet.",
    )
    rank.add_argument("--batch", action="store_true", help="Rank multiple prompts.")
    rank.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    add_output_options(rank)
    rank.add_argument(
        "--remote",
        action="store_true",
        help="Compatibility option; ranking uses --server by default.",
    )
    rank.add_argument(
        "--local",
        action="store_true",
        help="Load model/signatures locally instead of using the running server.",
    )
    rank.add_argument("--debug", action="store_true", help="Print extra details.")

    benchmark = subparsers.add_parser(
        "benchmark-models",
        help="Benchmark candidate embedding models on a labeled prompt set.",
    )
    benchmark.add_argument(
        "-in",
        "--input",
        dest="input_path",
        default=DEFAULT_BENCHMARK_INPUT,
        help=f"Benchmark input JSON/JSONL/text file. Default: {DEFAULT_BENCHMARK_INPUT}",
    )
    add_signature_path_option(benchmark)
    benchmark.add_argument(
        "--models",
        default=DEFAULT_BENCHMARK_MODELS,
        help="Comma-separated model names to benchmark.",
    )
    benchmark.add_argument(
        "--output",
        help="Optional JSON output path for benchmark results.",
    )
    benchmark.add_argument(
        "--dataset",
        default=DEFAULT_BENCHMARK_DATASET,
        help=f"Training dataset used to build temporary signatures per model. Default: {DEFAULT_BENCHMARK_DATASET}",
    )
    benchmark.add_argument("--facet", help="Facet name for benchmark dataset if it cannot be inferred.")
    benchmark.add_argument(
        "--max-vectors",
        type=int,
        default=DEFAULT_MAX_VECTORS,
        help=f"Maximum anchor vectors per temporary benchmark signature. Default: {DEFAULT_MAX_VECTORS}",
    )
    benchmark.add_argument("--online", action="store_true", help="Allow model downloads/checks.")
    benchmark.add_argument("--PU", help="Processing unit index or auto. See --PUs.")
    add_index_option(benchmark)
    add_loading_policy_options(benchmark)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate loaded signatures on a labeled LSR JSONL benchmark.",
    )
    evaluate.add_argument(
        "-in",
        "--input",
        dest="input_path",
        default=DEFAULT_BENCHMARK_INPUT,
        help=f"Evaluation JSON/JSONL input. Default: {DEFAULT_BENCHMARK_INPUT}",
    )
    add_signature_path_option(evaluate)
    add_runtime_options(evaluate)
    add_index_option(evaluate)
    add_loading_policy_options(evaluate)
    add_input_policy_options(evaluate)
    evaluate.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Returned result limit. Default: {DEFAULT_TOP_K}")
    evaluate.add_argument("--output", help="Optional JSON report output path.")
    evaluate.add_argument("--sweep-thresholds", action="store_true", help="Evaluate a small pass/accept threshold grid.")

    return parser.parse_args()
