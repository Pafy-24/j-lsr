from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any

from .args import parse_args
from .console import console, error_console
from .dataset import examples_from_dataset, grouped_examples, infer_facet_from_paths
from .dependencies import make_table
from .devices import print_processing_units, select_processing_unit
from .embedding import embed_texts, load_model
from .errors import RouterError
from .io_utils import load_batch_prompts, load_dataset, read_file, read_stdin_if_available, read_text_arg
from .model_selection import resolve_model_name, validate_signature_model_compatibility
from .models import FacetSignature
from .routing import LSRRouter, json_ready_route
from .server import serve
from .signatures import SignatureBuilder, SignatureStore


class RouterCLI:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run(self) -> int:
        if self.args.PUs:
            print_processing_units()
            return 0
        if self.args.command == "sign":
            if self.args.model == "auto":
                raise RouterError("sign does not support --model auto. Pass the model used to create embeddings.")
            if self.args.input_path:
                self.sign_dataset()
            else:
                self.sign_raw_text()
        elif self.args.command == "serve":
            self.serve()
        elif self.args.command == "loadsig":
            self.loadsig()
        elif self.args.command == "unloadsig":
            self.unloadsig()
        elif self.args.command == "rank":
            if self.args.local and self.args.batch:
                self.route_batch()
            elif self.args.local:
                self.route_prompt()
            else:
                self.rank_remote()
        elif self.args.command == "benchmark-models":
            self.benchmark_models()
        else:
            raise RouterError("Missing command. Use serve, loadsig, unloadsig, sign, or rank.")
        return 0

    def sign_dataset(self) -> None:
        dataset = load_dataset(self.args.input_path)
        default_facet = self._default_facet_for_dataset(dataset)
        grouped = grouped_examples(examples_from_dataset(dataset, default_facet))
        model = load_model(self.args.model, self.args.PU, self.args.online)
        builder = SignatureBuilder(self.args.max_vectors)
        store = SignatureStore(self.args.signatures)

        signatures: list[FacetSignature] = []
        for facet, texts in sorted(grouped.items()):
            facet_signature, layered_signature = builder.build_from_texts(facet, texts, model)
            output_path = store.save(layered_signature, self.args.model)
            signatures.append(facet_signature)
            if self.args.debug:
                layer_text = ", ".join(
                    f"L{level}={size}" for level, size in sorted(facet_signature.layer_sizes.items())
                )
                console.print(
                    f"[dim]{facet}: {len(texts)} route texts, "
                    f"{layer_text} -> {output_path}[/dim]"
                )

        print_signature_summary(signatures)
        console.print(f"Saved ASN.1 DER signatures under [bold]{self.args.signatures}[/bold]")

    def _default_facet_for_dataset(self, dataset: list[dict[str, Any]]) -> str | None:
        if self.args.facet:
            return self.args.facet.strip().lower()
        if all(str(item.get("facet") or "").strip() for item in dataset):
            return None
        return infer_facet_from_paths(self.args.input_path, self.args.signatures, None)

    def sign_raw_text(self) -> None:
        text = read_text_arg(self.args.text)
        if not text:
            raise RouterError("Missing text to sign. Provide raw text or use -in dataset.json.")
        facet = infer_facet_from_paths(None, self.args.signatures, self.args.facet)
        model = load_model(self.args.model, self.args.PU, self.args.online)
        builder = SignatureBuilder(self.args.max_vectors)
        facet_signature, layered_signature = builder.build_from_texts(facet, [text], model)
        output_path = SignatureStore(self.args.signatures).save(layered_signature, self.args.model)
        print_signature_summary([facet_signature])
        console.print(f"Saved vector signature to [bold]{output_path}[/bold]")

    def route_prompt(self) -> None:
        signatures = SignatureStore(self.args.signatures).load_all(required=True)
        if not signatures:
            raise RouterError(f"No facets found in signature path: {self.args.signatures}")
        model_name = resolve_model_name(self.args.model, signatures)
        validate_signature_model_compatibility(model_name, signatures)
        if self.args.top_k < 1:
            raise RouterError("--top-k must be at least 1.")

        if self.args.input_path:
            prompt = read_file(self.args.input_path)
        else:
            prompt = read_text_arg(self.args.text) or read_stdin_if_available()

        if not prompt:
            raise RouterError("Missing prompt. Provide -in input.txt, raw text, or stdin.")

        model = load_model(model_name, self.args.PU, self.args.online)
        prompt_embedding = embed_texts(model, [prompt])[0]
        route = LSRRouter(
            signatures,
            self.args.top_k,
            index_policy=self.args.index,
            model_name=model_name,
            memory_limit_mb=self.args.memory_limit_mb,
            descend_threshold=self.args.descend_threshold,
            accept_threshold=self.args.accept_threshold,
            full_depth=self.args.full_depth,
        ).route(prompt, prompt_embedding)

        if self.args.json:
            print(json.dumps(json_ready_route(route), ensure_ascii=False))
        else:
            print_route_table(route, self.args)

    def route_batch(self) -> None:
        signatures = SignatureStore(self.args.signatures).load_all(required=True)
        if not signatures:
            raise RouterError(f"No facets found in signature path: {self.args.signatures}")
        model_name = resolve_model_name(self.args.model, signatures)
        validate_signature_model_compatibility(model_name, signatures)
        if self.args.top_k < 1:
            raise RouterError("--top-k must be at least 1.")

        raw = read_file(self.args.input_path) if self.args.input_path else read_stdin_if_available()
        prompts = load_batch_prompts(raw)
        model = load_model(model_name, self.args.PU, self.args.online)
        embeddings = embed_texts(model, prompts)
        router = LSRRouter(
            signatures,
            self.args.top_k,
            index_policy=self.args.index,
            model_name=model_name,
            memory_limit_mb=self.args.memory_limit_mb,
            descend_threshold=self.args.descend_threshold,
            accept_threshold=self.args.accept_threshold,
            full_depth=self.args.full_depth,
        )
        routes = [
            router.route(prompt, embeddings[index])
            for index, prompt in enumerate(prompts)
        ]
        print(json.dumps([json_ready_route(route) for route in routes], ensure_ascii=False))

    def serve(self) -> None:
        console.print(
            f"Starting LSR server on {self.args.host}:{self.args.port} "
            f"with signatures {self.args.signatures}"
        )
        serve(
            host=self.args.host,
            port=self.args.port,
            signatures_path=self.args.signatures,
            model_name=self.args.model,
            pu_index=self.args.PU,
            online=self.args.online,
            default_top_k=self.args.top_k,
            index_policy=self.args.index,
            memory_limit_mb=self.args.memory_limit_mb,
            descend_threshold=self.args.descend_threshold,
            accept_threshold=self.args.accept_threshold,
            load_initial_signatures=not self.args.no_load,
        )

    def loadsig(self) -> None:
        response = post_json(
            self.args.server,
            "/loadsig",
            {"signatures": self.args.signatures},
        )
        print(json.dumps(response, ensure_ascii=False))

    def unloadsig(self) -> None:
        if not self.args.all and not self.args.facet:
            raise RouterError("unloadsig requires --facet or --all.")
        response = post_json(
            self.args.server,
            "/unloadsig",
            {"facets": self.args.facet or [], "all": self.args.all},
        )
        print(json.dumps(response, ensure_ascii=False))

    def rank_remote(self) -> None:
        if self.args.batch:
            raw = read_file(self.args.input_path) if self.args.input_path else read_stdin_if_available()
            prompts = load_batch_prompts(raw)
            response = post_json(
                self.args.server,
                "/batch",
                {"inputs": prompts, "top_k": self.args.top_k, "full_depth": self.args.full_depth},
            )
        else:
            if self.args.input_path:
                prompt = read_file(self.args.input_path)
            else:
                prompt = read_text_arg(self.args.text) or read_stdin_if_available()
            if not prompt:
                raise RouterError("Missing prompt. Provide -in input.txt, raw text, or stdin.")
            response = post_json(
                self.args.server,
                "/rank",
                {"input": prompt, "top_k": self.args.top_k, "full_depth": self.args.full_depth},
            )
        if self.args.batch or self.args.json:
            print(json.dumps(response, ensure_ascii=False))
        else:
            print_route_table(response, self.args)

    def benchmark_models(self) -> None:
        raw = read_file(self.args.input_path)
        prompts, expected = load_benchmark_items(raw)
        models = [item.strip() for item in self.args.models.split(",") if item.strip()]
        if not models:
            raise RouterError("--models must include at least one model.")
        benchmark_dataset = load_dataset(self.args.dataset)
        default_facet = self.args.facet
        if not default_facet and not all(str(item.get("facet") or "").strip() for item in benchmark_dataset):
            default_facet = infer_facet_from_paths(self.args.dataset, self.args.signatures, None)
        grouped = grouped_examples(examples_from_dataset(benchmark_dataset, default_facet))

        results = []
        for model_name in models:
            started = time.perf_counter()
            model = load_model(model_name, self.args.PU, self.args.online)
            model_load_seconds = time.perf_counter() - started

            build_started = time.perf_counter()
            builder = SignatureBuilder(self.args.max_vectors)
            signatures = {
                facet: builder.build_from_texts(facet, texts, model)[1]
                for facet, texts in sorted(grouped.items())
            }
            signature_build_seconds = time.perf_counter() - build_started

            index_started = time.perf_counter()
            router = LSRRouter(
                signatures,
                top_k=3,
                index_policy=self.args.index,
                model_name=model_name,
                memory_limit_mb=self.args.memory_limit_mb,
                descend_threshold=self.args.descend_threshold,
                accept_threshold=self.args.accept_threshold,
            )
            index_build_seconds = time.perf_counter() - index_started

            latencies_ms: list[float] = []
            ranks: list[int | None] = []
            for prompt, expected_facets in zip(prompts, expected):
                route_started = time.perf_counter()
                embedding = embed_texts(model, [prompt])[0]
                route = router.route(prompt, embedding)
                latencies_ms.append((time.perf_counter() - route_started) * 1000)
                facets = [str(result["facet"]).lower() for result in route["results"]]
                expected_set = {facet.lower() for facet in expected_facets}
                rank = next(
                    (index + 1 for index, facet in enumerate(facets) if facet in expected_set),
                    None,
                )
                ranks.append(rank)

            summary = benchmark_summary(
                model_name,
                model_load_seconds,
                signature_build_seconds,
                index_build_seconds,
                latencies_ms,
                ranks,
                signatures,
            )
            results.append(summary)

        recommendation = recommend_model(results)
        payload = {"recommendation": recommendation, "results": results}
        if self.args.output:
            from pathlib import Path

            Path(self.args.output).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def print_signature_summary(signatures: list[FacetSignature]) -> None:
    table = make_table(title="Signature Summary")
    table.add_column("Facet")
    table.add_column("Route Texts", justify="right")
    table.add_column("Layers")
    table.add_column("Output")

    for signature in sorted(signatures, key=lambda item: item.facet):
        layer_text = ", ".join(
            f"L{level}={size}" for level, size in sorted(signature.layer_sizes.items())
        )
        table.add_row(
            signature.facet,
            str(signature.num_route_texts),
            layer_text,
            f"{signature.facet}.sig",
        )
    console.print(table)


def print_route_table(route: dict[str, Any], args: argparse.Namespace) -> None:
    top_results = route["results"]

    table = make_table(title="Routing Results")
    table.add_column("Rank", justify="right")
    table.add_column("Facet")
    table.add_column("Score", justify="right")
    table.add_column("Decision")
    table.add_column("Exact", justify="right")
    table.add_column("Layers")
    for rank, result in enumerate(top_results, start=1):
        table.add_row(
            str(rank),
            result["facet"],
            f"{result['score']:.4f}",
            str(result.get("decision", "")),
            str(result["exact_count"]),
            format_layer_trace(result.get("layers", [])),
        )
    console.print(table)

    if args.debug:
        prompt = route["input"]
        console.print(f"[dim]Prompt: {prompt}[/dim]")
        console.print(f"[dim]Model: {args.model}[/dim]")
        if args.PU is not None:
            unit = select_processing_unit(args.PU)
            console.print(
                f"[dim]PU: #{unit.index} {unit.kind} "
                f"{unit.backend} {unit.device}[/dim]"
            )
        console.print(f"[dim]Compared {route['compared_facets']} facets.[/dim]")
        console.print(
            f"[dim]Deep searched {len(route['deep_facets'])} facets: "
            f"{', '.join(route['deep_facets'])}[/dim]"
        )
        console.print(f"[dim]Model: {route.get('modelName') or args.model}[/dim]")
        console.print(f"[dim]Index: {route.get('indexBackend', 'unknown')}[/dim]")
        console.print(f"[dim]Vectors scanned: {route.get('vectorsScanned', 0)}[/dim]")
        console.print(f"[dim]Loading: {route.get('loadingPolicy', 'unknown')}[/dim]")


def format_layer_trace(layers: list[dict[str, Any]]) -> str:
    if not layers:
        return ""
    parts: list[str] = []
    for layer in layers:
        level = layer.get("level", "?")
        score = float(layer.get("score", 0.0))
        decision = str(layer.get("decision", ""))
        pass_threshold = float(layer.get("passThreshold", 0.0))
        accept_threshold = float(layer.get("acceptThreshold", 0.0))
        parts.append(
            f"L{level} {score:.3f} {decision} "
            f"(p>{pass_threshold:.2f}, a>{accept_threshold:.2f})"
        )
    return " | ".join(parts)


def load_benchmark_items(raw: str) -> tuple[list[str], list[list[str]]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RouterError("Benchmark input must be a JSON list of labeled prompts.") from exc
    if not isinstance(data, list):
        raise RouterError("Benchmark input must be a JSON list.")
    prompts: list[str] = []
    expected: list[list[str]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise RouterError(f"Benchmark item {index} must be an object.")
        prompt = item.get("input") or item.get("prompt") or item.get("text")
        expected_facets = item.get("expected_facets") or item.get("expected") or item.get("facet")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RouterError(f"Benchmark item {index} has no prompt text.")
        if isinstance(expected_facets, str):
            expected_list = [expected_facets]
        elif isinstance(expected_facets, list) and all(isinstance(facet, str) for facet in expected_facets):
            expected_list = expected_facets
        else:
            raise RouterError(f"Benchmark item {index} has no expected facet.")
        prompts.append(prompt.strip())
        expected.append([facet.strip().lower() for facet in expected_list if facet.strip()])
    return prompts, expected


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[index])


def estimate_signature_ram_mb(signatures: dict[str, Any]) -> float:
    total_bytes = 0
    for signature in signatures.values():
        for layer in signature.layers.values():
            total_bytes += int(layer.vectors.nbytes)
    return total_bytes / (1024 * 1024)


def benchmark_summary(
    model_name: str,
    model_load_seconds: float,
    signature_build_seconds: float,
    index_build_seconds: float,
    latencies_ms: list[float],
    ranks: list[int | None],
    signatures: dict[str, Any],
) -> dict[str, Any]:
    total = len(ranks) or 1
    recall_at_1 = sum(1 for rank in ranks if rank == 1) / total
    recall_at_3 = sum(1 for rank in ranks if rank is not None and rank <= 3) / total
    mrr = sum((1 / rank) for rank in ranks if rank) / total
    return {
        "model": model_name,
        "recall@1": recall_at_1,
        "recall@3": recall_at_3,
        "mrr": mrr,
        "latency_ms_p50": percentile(latencies_ms, 50),
        "latency_ms_p95": percentile(latencies_ms, 95),
        "model_load_seconds": model_load_seconds,
        "signature_build_seconds": signature_build_seconds,
        "index_build_seconds": index_build_seconds,
        "signature_ram_mb": estimate_signature_ram_mb(signatures),
    }


def recommend_model(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    best_recall = max(float(result["recall@1"]) for result in results)
    close = [
        result
        for result in results
        if best_recall - float(result["recall@1"]) <= 0.02
    ]
    selected = min(close, key=lambda item: float(item["latency_ms_p50"]))
    return str(selected["model"])


def sign_dataset(args: argparse.Namespace) -> None:
    RouterCLI(args).sign_dataset()


def sign_raw_text(args: argparse.Namespace) -> None:
    RouterCLI(args).sign_raw_text()


def route_prompt(args: argparse.Namespace) -> None:
    RouterCLI(args).route_prompt()


def route_batch(args: argparse.Namespace) -> None:
    RouterCLI(args).route_batch()


def post_json(server_url: str, path: str, payload: Any, timeout: int = 60) -> Any:
    base_url = server_url.rstrip("/")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {"error": body or exc.reason}
        raise RouterError(str(data.get("error") or data)) from exc
    except urllib.error.URLError as exc:
        raise RouterError(f"Cannot reach LSR server at {server_url}: {exc.reason}") from exc


def main() -> int:
    args = parse_args()
    try:
        return RouterCLI(args).run()
    except RouterError as exc:
        error_console.print(f"[red]Error:[/red] {exc}")
        return 2
    except KeyboardInterrupt:
        error_console.print("\n[red]Interrupted.[/red]")
        return 130
