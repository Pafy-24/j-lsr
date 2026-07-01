from __future__ import annotations

import argparse

from .config import (
    DEFAULT_MAX_VECTORS,
    DEFAULT_MODEL,
    DEFAULT_SIGNATURES,
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
        default=0.0,
        help="Maximum signature vector memory kept warm. 0 means unlimited. Default: 0",
    )
    parser.add_argument(
        "--descend-threshold",
        type=float,
        default=LAYER_DESCEND_THRESHOLD,
        help=f"Minimum layer score needed to continue deeper. Default: {LAYER_DESCEND_THRESHOLD}",
    )
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=LAYER_ACCEPT_THRESHOLD,
        help=f"Layer score that accepts a facet without deeper comparison. Default: {LAYER_ACCEPT_THRESHOLD}",
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
        default="http://127.0.0.1:8765",
        help="Running LSR server URL. Default: http://127.0.0.1:8765",
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
    serve.add_argument("--host", default="127.0.0.1", help="Host address. Default: 127.0.0.1")
    serve.add_argument("--port", type=int, default=8765, help="Port. Default: 8765")
    serve.add_argument("--top-k", type=int, default=3, help="Default result count. Default: 3")
    serve.add_argument(
        "--no-load",
        action="store_true",
        help="Start without loading initial signatures. Use loadsig later.",
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
    add_server_target_option(rank)
    rank.add_argument("--top-k", type=int, default=3, help="Number of facets to return. Default: 3")
    rank.add_argument("--batch", action="store_true", help="Rank multiple prompts.")
    rank.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
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
        default="../router_test_crypto_web_100.json",
        help="Benchmark input JSON/JSONL/text file. Default: ../router_test_crypto_web_100.json",
    )
    add_signature_path_option(benchmark)
    benchmark.add_argument(
        "--models",
        default="sentence-transformers/all-MiniLM-L6-v2,thenlper/gte-small",
        help="Comma-separated model names to benchmark.",
    )
    benchmark.add_argument(
        "--output",
        help="Optional JSON output path for benchmark results.",
    )
    benchmark.add_argument(
        "--dataset",
        default="../facets/Crypto/datasets/train/crypto_dataset.json",
        help="Training dataset used to build temporary signatures per model.",
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

    return parser.parse_args()
