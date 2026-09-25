from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any

from .args import parse_args
from .console import console, error_console
from .dataset import grouped_examples, infer_facet_from_paths
from .dataset_schema import benchmark_expected_facets, items_to_examples, load_labeled_prompts, load_lsr_dataset
from .dependencies import make_ascii_table, make_table
from .devices import print_processing_units, select_processing_unit
from .embedding import embed_texts, load_model
from .errors import RouterError
from .input_digest import prepare_routing_input, prepare_routing_input_from_path
from .io_utils import load_batch_prompts, read_file, read_stdin_if_available, read_text_arg
from .memory import current_footprint_bytes, format_bytes
from .model_selection import resolve_model_name, validate_signature_model_compatibility
from .models import FacetSignature
from .profiling import RankProfiler
from .routing import LSRRouter, json_ready_route
from .rm_integration import require_managed_launch, start_telemetry
from .server import serve
from .signatures import SignatureBuilder, SignatureStore
from .registry import SignatureRegistry
from .strategies import build_strategy, load_strategy


class RouterCLI:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run(self) -> int:
        if getattr(self.args, "PUs", False):
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
        elif self.args.command == "strategy":
            self.strategy_asset()
        elif self.args.command == "benchmark-models":
            self.benchmark_models()
        elif self.args.command == "evaluate":
            self.evaluate()
        else:
            raise RouterError("Missing command. Use serve, loadsig, unloadsig, sign, rank, or strategy.")
        return 0

    def sign_dataset(self) -> None:
        default_facet = self.args.facet.strip().lower() if self.args.facet else None
        items = load_lsr_dataset(self.args.input_path, default_facet)
        grouped = grouped_examples(items_to_examples(items))
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
        registry, router, model_name = self._local_registry_router()
        # Bind the asset before constructing an encoder so an incompatible
        # local strategy fails without allocating or downloading a model.
        strategy = self._local_strategy(registry, router)

        routing_input = None
        if self.args.input_path:
            routing_input = prepare_routing_input_from_path(self.args.input_path, self.args.large_input)
            prompt = routing_input.routing_text
        else:
            prompt = read_text_arg(self.args.text) or read_stdin_if_available()

        if not prompt:
            raise RouterError("Missing prompt. Provide -in input.txt, raw text, or stdin.")

        model = load_model(model_name, self.args.PU, self.args.online)
        if routing_input is None:
            routing_input = prepare_routing_input(prompt, self.args.large_input)
        profiler = RankProfiler()
        profiler.start()
        embedding_started = time.perf_counter()
        prompt_embedding = embed_texts(model, routing_input.routing_units)
        profiler.stage("embedding", embedding_started)
        routing_started = time.perf_counter()
        if strategy is None:
            route = router.route(routing_input.routing_text, prompt_embedding)
            result: dict[str, Any] = route
        elif self.args.strategy_operation == "explain":
            result = strategy.explain(routing_input.routing_text, prompt_embedding)
            route = result["route"]
        elif self.args.strategy_operation == "profile":
            result = strategy.profile(
                routing_input.routing_text,
                prompt_embedding,
                embedding_ms=profiler.stages.get("embedding"),
            )
            route = result["route"]
        else:
            route = strategy.route(routing_input.routing_text, prompt_embedding)
            result = route
        profiler.stage("routing", routing_started)
        attach_input_metadata(route, routing_input)
        if result is route:
            route["performance"] = profiler.finish()
        else:
            result["performance"] = profiler.finish()

        if self.args.json or result is not route:
            if result is not route:
                result = dict(result)
                result["route"] = json_ready_route(route)
            else:
                result = json_ready_route(route)
            print(json.dumps(result, ensure_ascii=False))
        else:
            print_route_table(route, self.args)

    def route_batch(self) -> None:
        registry, router, model_name = self._local_registry_router()
        strategy = self._local_strategy(registry, router)
        if self.args.strategy_operation != "route":
            raise RouterError("strategy explain/profile are supported for one local rank request only.")

        raw = read_file(self.args.input_path) if self.args.input_path else read_stdin_if_available()
        prompts = load_batch_prompts(raw)
        model = load_model(model_name, self.args.PU, self.args.online)
        routing_inputs = [prepare_routing_input(prompt, self.args.large_input) for prompt in prompts]
        flat_units = [unit for item in routing_inputs for unit in item.routing_units]
        embeddings = embed_texts(model, flat_units)
        routes = []
        offset = 0
        for routing_input in routing_inputs:
            unit_count = len(routing_input.routing_units)
            routes.append(
                with_input_metadata(
                    (
                        strategy.route(routing_input.routing_text, embeddings[offset : offset + unit_count])
                        if strategy is not None
                        else router.route(routing_input.routing_text, embeddings[offset : offset + unit_count])
                    ),
                    routing_input,
                )
            )
            offset += unit_count
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
            monitor_enabled=not self.args.no_monitor,
            workers=self.args.workers,
            model_identity=self.args.model_identity,
            strategy_asset_path=self.args.strategy_asset,
        )

    def _local_registry_router(self) -> tuple[SignatureRegistry, LSRRouter, str]:
        registry = SignatureRegistry(self.args.signatures)
        signatures = registry.load(required=True)
        if not signatures:
            raise RouterError(f"No facets found in signature path: {self.args.signatures}")
        model_name = resolve_model_name(self.args.model, signatures)
        validate_signature_model_compatibility(model_name, signatures)
        if self.args.top_k < 1:
            raise RouterError("--top-k must be at least 1.")
        return registry, LSRRouter(
            signatures,
            self.args.top_k,
            index_policy=self.args.index,
            model_name=model_name,
            memory_limit_mb=self.args.memory_limit_mb,
            descend_threshold=self.args.descend_threshold,
            accept_threshold=self.args.accept_threshold,
            full_depth=self.args.full_depth,
        ), model_name

    def _local_strategy(self, registry: SignatureRegistry, router: LSRRouter):
        asset_path = self.args.strategy_asset
        if not asset_path:
            if self.args.strategy_operation != "route":
                raise RouterError("--strategy-operation requires --strategy-asset.")
            return None
        return load_strategy(asset_path, registry, router)

    def strategy_asset(self) -> None:
        registry = SignatureRegistry(self.args.signatures)
        signatures = registry.load(required=True)
        if not signatures:
            raise RouterError(f"No facets found in signature path: {self.args.signatures}")
        model_name = resolve_model_name(self.args.model, signatures)
        validate_signature_model_compatibility(model_name, signatures)
        if self.args.top_k < 1:
            raise RouterError("--top-k must be at least 1.")
        router = LSRRouter(
            signatures,
            self.args.top_k,
            index_policy=self.args.index,
            model_name=model_name,
            memory_limit_mb=self.args.memory_limit_mb,
            descend_threshold=self.args.descend_threshold,
            accept_threshold=self.args.accept_threshold,
        )
        if self.args.action == "build":
            if not self.args.output:
                raise RouterError("strategy build requires --output.")
            config = {}
            if self.args.strategy_id != "semantic-baseline-v1":
                config = {
                    "bits": self.args.fingerprint_bits,
                    "seed": self.args.fingerprint_seed,
                    "minimum_jaccard": self.args.minimum_jaccard,
                }
            strategy = build_strategy(self.args.strategy_id, registry, router, **config)
            strategy.asset.save(self.args.output)
            print(json.dumps(strategy.asset.to_dict(), ensure_ascii=False))
            return
        if not self.args.asset:
            raise RouterError("strategy load requires --asset.")
        strategy = load_strategy(self.args.asset, registry, router)
        print(json.dumps({"asset": strategy.asset.to_dict(), "snapshot": strategy.snapshot().to_dict()}, ensure_ascii=False))

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
        if self.args.strategy_asset:
            raise RouterError(
                "--strategy-asset is local-only for rank; configure it when starting the target service."
            )
        if self.args.batch:
            raw = read_file(self.args.input_path) if self.args.input_path else read_stdin_if_available()
            prompts = load_batch_prompts(raw)
            response = post_json(
                self.args.server,
                "/batch",
                {
                    "inputs": prompts,
                    "top_k": self.args.top_k,
                    "full_depth": self.args.full_depth,
                    "large_input": self.args.large_input,
                    "strategy_operation": self.args.strategy_operation,
                },
            )
        else:
            routing_input = None
            if self.args.input_path:
                routing_input = prepare_routing_input_from_path(self.args.input_path, self.args.large_input)
                prompt = routing_input.routing_text
            else:
                prompt = read_text_arg(self.args.text) or read_stdin_if_available()
            if not prompt:
                raise RouterError("Missing prompt. Provide -in input.txt, raw text, or stdin.")
            response = post_json(
                self.args.server,
                "/rank",
                {
                    "input": prompt,
                    "top_k": self.args.top_k,
                    "full_depth": self.args.full_depth,
                    "large_input": "off" if routing_input is not None else self.args.large_input,
                    "strategy_operation": self.args.strategy_operation,
                },
            )
            if routing_input is not None:
                response["inputMode"] = routing_input.input_mode
                response["inputChars"] = routing_input.input_chars
                response["routingTextChars"] = routing_input.routing_text_chars
                response["chunksUsed"] = routing_input.chunks_used
        if self.args.batch or self.args.json:
            print(json.dumps(response, ensure_ascii=False))
        else:
            print_route_table(response, self.args)

    def benchmark_models(self) -> None:
        prompts, expected = load_labeled_prompts(self.args.input_path)
        models = [item.strip() for item in self.args.models.split(",") if item.strip()]
        if not models:
            raise RouterError("--models must include at least one model.")
        default_facet = self.args.facet.strip().lower() if self.args.facet else None
        benchmark_items = load_lsr_dataset(self.args.dataset, default_facet)
        grouped = grouped_examples(items_to_examples(benchmark_items))

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
                routing_input = prepare_routing_input(prompt, "auto")
                embedding = embed_texts(model, routing_input.routing_units)
                route = router.route(routing_input.routing_text, embedding)
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

    def evaluate(self) -> None:
        signatures = SignatureStore(self.args.signatures).load_all(required=True)
        if not signatures:
            raise RouterError(f"No facets found in signature path: {self.args.signatures}")
        model_name = resolve_model_name(self.args.model, signatures)
        validate_signature_model_compatibility(model_name, signatures)
        items = load_lsr_dataset(self.args.input_path)
        prompts = ["\n".join(item.texts) for item in items]
        expected = [item.expected_facets for item in items]
        model = load_model(model_name, self.args.PU, self.args.online)

        if self.args.sweep_thresholds:
            payload = {
                "sweeps": [
                    self.evaluate_once(
                        signatures,
                        model,
                        model_name,
                        prompts,
                        expected,
                        pass_threshold=pass_threshold,
                        accept_threshold=accept_threshold,
                    )
                    for pass_threshold, accept_threshold in threshold_grid()
                ]
            }
        else:
            payload = self.evaluate_once(signatures, model, model_name, prompts, expected)

        if self.args.output:
            from pathlib import Path

            Path(self.args.output).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    def evaluate_once(
        self,
        signatures: dict[str, Any],
        model: Any,
        model_name: str,
        prompts: list[str],
        expected: list[list[str]],
        pass_threshold: float | None = None,
        accept_threshold: float | None = None,
    ) -> dict[str, Any]:
        router = LSRRouter(
            signatures,
            self.args.top_k,
            index_policy=self.args.index,
            model_name=model_name,
            memory_limit_mb=self.args.memory_limit_mb,
            descend_threshold=pass_threshold if pass_threshold is not None else self.args.descend_threshold,
            accept_threshold=accept_threshold if accept_threshold is not None else self.args.accept_threshold,
        )
        latencies_ms: list[float] = []
        ranks: list[int | None] = []
        false_positives = 0
        refused = 0
        decisions_by_layer: dict[str, dict[str, int]] = {}
        decays: list[float] = []
        entropies: list[float] = []
        confidences: list[float] = []
        weights: list[float] = []
        selected_counts: list[float] = []
        selection_precisions: list[float] = []
        selection_recalls: list[float] = []
        selection_f1_scores: list[float] = []
        extraneous_counts: list[float] = []
        missing_counts: list[float] = []
        exact_set_matches = 0
        over_selections = 0
        under_selections = 0
        separations: list[float] = []
        separation_ratios: list[float] = []
        lexical_scores: list[float] = []
        generic_penalties: list[float] = []
        repetition_penalties: list[float] = []
        # Warm the embedding/CUDA kernels and one routing path before taking
        # the memory baseline; otherwise lazy backend initialization is
        # incorrectly reported as per-evaluation signature growth.
        if prompts:
            warm_input = prepare_routing_input(prompts[0], self.args.large_input)
            warm_embedding = embed_texts(model, warm_input.routing_units)
            router.route(warm_input.routing_text, warm_embedding)
        before_memory = process_memory_mb()

        for prompt, expected_facets in zip(prompts, expected):
            routing_input = prepare_routing_input(prompt, self.args.large_input)
            started = time.perf_counter()
            embedding = embed_texts(model, routing_input.routing_units)
            route = router.route(routing_input.routing_text, embedding)
            latencies_ms.append((time.perf_counter() - started) * 1000)
            results = route["results"]
            selected = route.get("selectedFacets", [])
            facets = [str(result["facet"]).lower() for result in selected]
            selected_counts.append(float(len(selected)))
            expected_set = {facet.lower() for facet in expected_facets}
            selected_set = set(facets)
            true_positives = len(selected_set & expected_set)
            precision = true_positives / len(selected_set) if selected_set else 0.0
            recall = true_positives / len(expected_set) if expected_set else 1.0
            selection_precisions.append(precision)
            selection_recalls.append(recall)
            selection_f1_scores.append(
                0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
            )
            extraneous = len(selected_set - expected_set)
            missing = len(expected_set - selected_set)
            extraneous_counts.append(float(extraneous))
            missing_counts.append(float(missing))
            if selected_set == expected_set:
                exact_set_matches += 1
            if extraneous:
                over_selections += 1
            if missing:
                under_selections += 1
            rank = next((index + 1 for index, facet in enumerate(facets) if facet in expected_set), None)
            ranks.append(rank)
            if facets and facets[0] not in expected_set:
                false_positives += 1
            if not selected:
                refused += 1
            for result in results:
                decays.append(float(result.get("confidence_decay", 0.0)))
                entropies.append(float(result.get("entropy", 0.0)))
                confidences.append(float(result.get("confidence", 0.0)))
                weights.append(float(result.get("weight", 0.0)))
                separations.append(float(result.get("separation", 0.0)))
                separation_ratios.append(float(result.get("separationRatio", 0.0)))
                lexical_scores.append(float(result.get("lexicalScore", 0.0)))
                generic_penalties.append(float(result.get("genericPenalty", 0.0)))
                repetition_penalties.append(float(result.get("repetitionPenalty", 0.0)))
                for layer in result.get("layers", []):
                    key = str(layer.get("level"))
                    decision = str(layer.get("decision"))
                    decisions_by_layer.setdefault(key, {"accepted": 0, "passed": 0, "refused": 0})
                    if decision in decisions_by_layer[key]:
                        decisions_by_layer[key][decision] += 1

        after_memory = process_memory_mb()
        total = len(ranks) or 1
        return {
            "items": len(ranks),
            "recall@1": sum(1 for rank in ranks if rank == 1) / total,
            "recall@3": sum(1 for rank in ranks if rank is not None and rank <= 3) / total,
            "mrr": sum((1 / rank) for rank in ranks if rank) / total,
            "falsePositiveRate": false_positives / total,
            "falsePositiveRateSemantics": "top-1-selected-facet-not-in-expected-set",
            "refusedRate": refused / total,
            "selectionSet": {
                "precisionMacro": sum(selection_precisions) / total,
                "recallMacro": sum(selection_recalls) / total,
                "f1Macro": sum(selection_f1_scores) / total,
                "exactMatchRate": exact_set_matches / total,
                "overSelectionRate": over_selections / total,
                "underSelectionRate": under_selections / total,
                "extraneousFacetCount": metric_summary(extraneous_counts),
                "missingExpectedFacetCount": metric_summary(missing_counts),
            },
            "decisionsByLayer": decisions_by_layer,
            "confidenceDecay": metric_summary(decays),
            "confidence": metric_summary(confidences),
            "weight": metric_summary(weights),
            "selectedFacetCount": metric_summary(selected_counts),
            "separation": metric_summary(separations),
            "separationRatio": metric_summary(separation_ratios),
            "lexicalScore": metric_summary(lexical_scores),
            "genericPenalty": metric_summary(generic_penalties),
            "repetitionPenalty": metric_summary(repetition_penalties),
            "entropy": metric_summary(entropies),
            "latencyMs": {
                "p50": percentile(latencies_ms, 50),
                "p95": percentile(latencies_ms, 95),
                "p99": percentile(latencies_ms, 99),
            },
            "memoryMb": {
                "before": before_memory,
                "after": after_memory,
            },
            "loadedVectorBytes": sum(signature.loaded_vector_bytes() for signature in signatures.values()),
            "thresholds": {
                "pass": pass_threshold,
                "accept": accept_threshold,
            },
        }


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

    table_factory = make_ascii_table if getattr(args, "ascii_table", False) else make_table
    table = table_factory(title="Routing Results")
    table.add_column("Rank", justify="right")
    table.add_column("Facet")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Decision")
    table.add_column("Exact", justify="right")
    table.add_column("Layers")
    for rank, result in enumerate(top_results, start=1):
        table.add_row(
            str(rank),
            result["facet"],
            f"{result['score']:.4f}",
            f"{float(result.get('weight', 0.0)):.4f}",
            str(result.get("decision", "")),
            str(result["exact_count"]),
            format_layer_trace(result.get("layers", [])),
        )
    console.print(table)
    performance = route.get("performance", {})
    timing = performance.get("timing", {}) if isinstance(performance, dict) else {}
    if timing:
        console.print(
            "[dim]Timing: "
            f"rank {float(timing.get('rankingMs', 0.0)):.3f} ms | "
            f"embedding {float(timing.get('embeddingMs', 0.0)):.3f} ms | "
            f"routing {float(timing.get('routingMs', 0.0)):.3f} ms | "
            f"CPU {float(timing.get('cpuTimeMs', 0.0)):.3f} ms[/dim]"
        )
        resources = performance.get("resources", {})
        cuda = resources.get("cuda", []) if isinstance(resources, dict) else []
        cuda_peak = sum(int(item.get("peakAllocatedDeltaBytes", 0)) for item in cuda)
        cuda_retained = sum(int(item.get("allocatedRetainedDeltaBytes", 0)) for item in cuda)
        console.print(
            "[dim]Peak during rank: "
            f"RAM +{format_bytes(int(resources.get('ramPeakDeltaBytes', 0)))} | "
            f"CUDA allocated +{format_bytes(cuda_peak)}[/dim]"
        )
        console.print(
            "[dim]Retained after rank: "
            f"working set {format_signed_bytes(int(resources.get('ramRetainedDeltaBytes', 0)))} | "
            f"private {format_signed_bytes(int(resources.get('privateRetainedDeltaBytes', 0)))} | "
            f"CUDA allocated {format_signed_bytes(cuda_retained)}[/dim]"
        )
    signature_memory = route.get("signatureMemory", {})
    if isinstance(signature_memory, dict) and signature_memory:
        peak = int(signature_memory.get("peakResidentVectorBytes", 0))
        eager = int(signature_memory.get("eagerAllLayersBytes", 0))
        l1 = int(signature_memory.get("allFacetL1Bytes", 0))
        read = int(signature_memory.get("totalLayerBytesRead", 0))
        fraction = float(signature_memory.get("peakFractionOfEager", 0.0)) * 100.0
        console.print(
            "[dim]Signature memory: "
            f"peak {format_bytes(peak)} ({fraction:.2f}% of eager) | "
            f"all L1 {format_bytes(l1)} | all layers {format_bytes(eager)} | "
            f"read {format_bytes(read)}[/dim]"
        )

    if args.debug:
        prompt = route["input"]
        console.print(f"[dim]Prompt: {prompt}[/dim]")
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


def format_signed_bytes(size: int) -> str:
    sign = "+" if size > 0 else "-" if size < 0 else ""
    return f"{sign}{format_bytes(abs(size))}"


def attach_input_metadata(route: dict[str, Any], routing_input: Any) -> None:
    route["inputMode"] = routing_input.input_mode
    route["inputChars"] = routing_input.input_chars
    route["routingTextChars"] = routing_input.routing_text_chars
    route["chunksUsed"] = routing_input.chunks_used
    route["routingUnits"] = len(routing_input.routing_units)


def with_input_metadata(route: dict[str, Any], routing_input: Any) -> dict[str, Any]:
    attach_input_metadata(route, routing_input)
    return route


def load_benchmark_items(raw: str) -> tuple[list[str], list[list[str]]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        data = []
        for line_index, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as line_exc:
                raise RouterError(f"Benchmark JSONL line {line_index} is invalid: {line_exc}") from exc
    if not isinstance(data, list):
        raise RouterError("Benchmark input must be a JSON list.")
    prompts: list[str] = []
    expected: list[list[str]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise RouterError(f"Benchmark item {index} must be an object.")
        prompt = item.get("input") or item.get("prompt") or item.get("text")
        if prompt is None and isinstance(item.get("texts"), list):
            prompt = "\n".join(str(text) for text in item["texts"])
        if not isinstance(prompt, str) or not prompt.strip():
            raise RouterError(f"Benchmark item {index} has no prompt text.")
        prompts.append(prompt.strip())
        expected.append(benchmark_expected_facets(item, index))
    return prompts, expected


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[index])


def metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": float(max(values)),
    }


def threshold_grid() -> list[tuple[float, float]]:
    return [
        (pass_threshold, accept_threshold)
        for pass_threshold in (0.16, 0.18, 0.20, 0.22)
        for accept_threshold in (0.78, 0.82, 0.86)
        if accept_threshold >= pass_threshold
    ]


def process_memory_mb() -> float:
    return current_footprint_bytes() / (1024 * 1024)


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
    telemetry = None
    try:
        require_managed_launch(args)
        telemetry = start_telemetry()
        return RouterCLI(args).run()
    except RouterError as exc:
        error_console.print(f"[red]Error:[/red] {exc}")
        return 2
    except KeyboardInterrupt:
        error_console.print("\n[red]Interrupted.[/red]")
        return 130
    finally:
        if telemetry is not None:
            telemetry.close()
