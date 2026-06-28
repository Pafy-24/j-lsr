from __future__ import annotations

import argparse
import json
from typing import Any

from .args import parse_args
from .console import console, error_console
from .dataset import examples_from_dataset, grouped_examples, infer_facet_from_paths
from .dependencies import make_table
from .devices import print_processing_units, select_processing_unit
from .embedding import embed_texts, load_model
from .errors import RouterError
from .io_utils import load_batch_prompts, load_dataset, read_file, read_stdin_if_available, read_text_arg
from .models import FacetSignature
from .routing import LSRRouter, expand_query, json_ready_route
from .server import serve
from .signatures import SignatureBuilder, SignatureStore


class RouterCLI:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run(self) -> int:
        if self.args.PUs:
            print_processing_units()
            return 0
        if self.args.sign:
            if self.args.input_path:
                self.sign_dataset()
            else:
                self.sign_raw_text()
        elif self.args.serve:
            self.serve()
        elif self.args.batch:
            self.route_batch()
        else:
            self.route_prompt()
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
        if self.args.top_k < 1:
            raise RouterError("--top-k must be at least 1.")

        if self.args.input_path:
            prompt = read_file(self.args.input_path)
        else:
            prompt = read_text_arg(self.args.text) or read_stdin_if_available()

        if not prompt:
            raise RouterError("Missing prompt. Provide -in input.txt, raw text, or stdin.")

        model = load_model(self.args.model, self.args.PU, self.args.online)
        routed_prompt = expand_query(prompt)
        prompt_embedding = embed_texts(model, [routed_prompt])[0]
        route = LSRRouter(signatures, self.args.top_k).route(
            prompt, routed_prompt, prompt_embedding
        )

        if self.args.json:
            print(json.dumps(json_ready_route(route), ensure_ascii=False))
        else:
            print_route_table(route, self.args)

    def route_batch(self) -> None:
        signatures = SignatureStore(self.args.signatures).load_all(required=True)
        if not signatures:
            raise RouterError(f"No facets found in signature path: {self.args.signatures}")
        if self.args.top_k < 1:
            raise RouterError("--top-k must be at least 1.")

        raw = read_file(self.args.input_path) if self.args.input_path else read_stdin_if_available()
        prompts = load_batch_prompts(raw)
        routed_prompts = [expand_query(prompt) for prompt in prompts]

        model = load_model(self.args.model, self.args.PU, self.args.online)
        embeddings = embed_texts(model, routed_prompts)
        router = LSRRouter(signatures, self.args.top_k)
        routes = [
            router.route(prompt, routed_prompt, embeddings[index])
            for index, (prompt, routed_prompt) in enumerate(zip(prompts, routed_prompts))
        ]
        print(json.dumps([json_ready_route(route) for route in routes], ensure_ascii=False))

    def serve(self) -> None:
        console.print(
            f"Starting LSR server on [bold]{self.args.host}:{self.args.port}[/bold] "
            f"with signatures [bold]{self.args.signatures}[/bold]"
        )
        serve(
            host=self.args.host,
            port=self.args.port,
            signatures_path=self.args.signatures,
            model_name=self.args.model,
            pu_index=self.args.PU,
            online=self.args.online,
            default_top_k=self.args.top_k,
        )


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
    table.add_column("Best Vector", justify="right")
    table.add_column("Cluster", justify="right")
    table.add_column("Exact", justify="right")
    table.add_column("Mode")
    table.add_column("Best Match")
    for rank, result in enumerate(top_results, start=1):
        table.add_row(
            str(rank),
            result["facet"],
            f"{result['score']:.4f}",
            f"{result['best_vector_score']:.4f}",
            f"{result['cluster_score']:.4f}",
            str(result["exact_count"]),
            str(result["mode"]),
            f"vector #{result['vector_index']}",
        )
    console.print(table)

    if args.debug:
        prompt = route["input"]
        routed_prompt = route["expanded"] or prompt
        console.print(f"[dim]Prompt: {prompt}[/dim]")
        if routed_prompt != prompt:
            console.print(f"[dim]Expanded: {routed_prompt}[/dim]")
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


def sign_dataset(args: argparse.Namespace) -> None:
    RouterCLI(args).sign_dataset()


def sign_raw_text(args: argparse.Namespace) -> None:
    RouterCLI(args).sign_raw_text()


def route_prompt(args: argparse.Namespace) -> None:
    RouterCLI(args).route_prompt()


def route_batch(args: argparse.Namespace) -> None:
    RouterCLI(args).route_batch()


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
