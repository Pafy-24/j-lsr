from __future__ import annotations

"""Compatibility wrapper; installed users should invoke `jester-lsr`."""
import argparse
import json
import os
import sys
import time

from lsr.args import parse_args
from lsr.commands import (
    RouterCLI,
    main,
    print_route_table,
    print_signature_summary,
    route_batch,
    route_prompt,
    sign_dataset,
    sign_raw_text,
)
from lsr.config import *
from lsr.console import LazyConsole, console, error_console
from lsr.dataset import *
from lsr import dependencies as _dependencies
from lsr.dependencies import (
    ensure_faiss,
    ensure_numpy,
    get_kmeans_class,
    get_sentence_transformer_class,
    make_table,
)
from lsr.der import *
from lsr.devices import *
from lsr.embedding import *
from lsr.errors import RouterError
from lsr.io_utils import *
from lsr.models import *
from lsr.routing import *
from lsr.routing import json_ready_route
from lsr.input_digest import prepare_routing_input
from lsr.signatures import *


def __getattr__(name: str):
    if name in {"np", "SentenceTransformer", "MiniBatchKMeans", "faiss"}:
        return getattr(_dependencies, name)
    raise AttributeError(name)


def native_qualification(argv: list[str]) -> int:
    """Finite packaging qualification using the normal local CPU routing path."""
    parser = argparse.ArgumentParser(prog="LSR.exe native-qualification")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--prompt", required=True)
    values = parser.parse_args(argv)
    if not values.prompt.strip() or len(values.prompt) > 4096:
        parser.error("--prompt must contain 1..4096 characters")
    values.model = os.path.abspath(values.model)
    values.signatures = os.path.abspath(values.signatures)
    if not os.path.isdir(values.model):
        parser.error(f"local model directory does not exist: {values.model}")
    if not os.path.exists(values.signatures):
        parser.error(f"local signatures file or directory does not exist: {values.signatures}")
    started = time.perf_counter()
    saved = sys.argv
    try:
        sys.argv = [saved[0], "rank", values.prompt, "--local", "--json", "--top-k", "2",
                    "--PU", "0", "--model", values.model, "--model-identity", values.model_identity,
                    "--signatures", values.signatures]
        args = parse_args()
    finally:
        sys.argv = saved
    cli = RouterCLI(args)
    registry, router, model_load_path = cli._local_registry_router()
    strategy = cli._local_strategy(registry, router)
    model_started = time.perf_counter()
    model = load_model(model_load_path, args.PU, False)
    model_load_ms = (time.perf_counter() - model_started) * 1000.0
    startup_ms = (time.perf_counter() - started) * 1000.0
    requests = []
    for index in range(2):
        request_started = time.perf_counter()
        routing_input = prepare_routing_input(values.prompt, args.large_input)
        embedding = embed_texts(model, routing_input.routing_units)
        route = router.route(routing_input.routing_text, embedding) if strategy is None else strategy.route(routing_input.routing_text, embedding)
        requests.append({"index": index, "durationMs": (time.perf_counter() - request_started) * 1000.0,
                         "route": json_ready_route(route)})
    print(json.dumps({"schemaVersion": "jester-lsr-native-qualification-v1", "pid": os.getpid(),
                      "modelIdentity": values.model_identity, "startupMs": startup_ms,
                      "modelLoadMs": model_load_ms, "requests": requests,
                      "totalMs": (time.perf_counter() - started) * 1000.0}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "native-qualification":
        raise SystemExit(native_qualification(sys.argv[2:]))
    raise SystemExit(main())
