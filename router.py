from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    SentenceTransformer = None

try:
    from sklearn.cluster import MiniBatchKMeans
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    MiniBatchKMeans = None


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIGNATURES = "signatures.json"

console = Console()
error_console = Console(stderr=True)


class RouterError(Exception):
    """User-facing CLI error."""


@dataclass
class RouteExample:
    facet: str
    text: str


@dataclass
class FacetSignature:
    facet: str
    num_examples: int
    centroids: np.ndarray
    representatives: list[dict[str, Any]]
    examples: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate semantic facet signatures and route prompts to facets."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Raw text to sign or route. Quote multi-word prompts.",
    )
    parser.add_argument(
        "--sign",
        action="store_true",
        help="Generate or update signatures instead of routing.",
    )
    parser.add_argument(
        "-in",
        "--input",
        dest="input_path",
        help="Input file path. In sign mode this must be a JSON dataset.",
    )
    parser.add_argument(
        "--facet",
        help="Facet name. Required when signing raw text.",
    )
    parser.add_argument(
        "--signatures",
        default=DEFAULT_SIGNATURES,
        help=f"Path to signatures JSON file. Default: {DEFAULT_SIGNATURES}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of facets to return in routing mode. Default: 3",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra details.",
    )
    return parser.parse_args()


def read_text_arg(parts: list[str]) -> str:
    return " ".join(parts).strip()


def read_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise RouterError(f"Input file not found: {file_path}")
    if not file_path.is_file():
        raise RouterError(f"Input path is not a file: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


def read_stdin_if_available() -> str:
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read().strip()


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    raw = read_file(path)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RouterError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise RouterError("Dataset must be a JSON list of objects.")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise RouterError(f"Dataset item {index} must be an object.")
    return data


def tags_to_text(tags: Any) -> str:
    if tags is None:
        return ""
    if isinstance(tags, list):
        return " ".join(str(tag) for tag in tags if str(tag).strip())
    return str(tags)


def extract_route_text(item: dict[str, Any]) -> str:
    # Prefer explicit routing text. Fall back to prompt-side fields and only use
    # output as a last resort so signatures learn the request, not the answer.
    text = str(item.get("text") or "").strip()
    if text:
        return text

    prompt_parts = [
        str(item.get("instruction") or "").strip(),
        str(item.get("input") or "").strip(),
        tags_to_text(item.get("tags")).strip(),
    ]
    prompt_text = " ".join(part for part in prompt_parts if part).strip()
    if prompt_text:
        return prompt_text

    return str(item.get("output") or "").strip()


def examples_from_dataset(dataset: list[dict[str, Any]]) -> list[RouteExample]:
    examples: list[RouteExample] = []
    for index, item in enumerate(dataset):
        facet = str(item.get("facet") or "").strip()
        if not facet:
            raise RouterError(f"Dataset item {index} is missing required field: facet")

        route_text = extract_route_text(item)
        if not route_text:
            raise RouterError(
                f"Dataset item {index} for facet '{facet}' has no routable text."
            )
        examples.append(RouteExample(facet=facet, text=route_text))
    return examples


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_model(model_name: str) -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RouterError(
            "Missing dependency: sentence-transformers. "
            "Install dependencies with: pip install -r requirements.txt"
        )
    return SentenceTransformer(model_name)


def choose_centroids(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_examples = len(embeddings)
    if n_examples <= 20:
        centroid = normalize_matrix(np.mean(embeddings, axis=0, keepdims=True))
        labels = np.zeros(n_examples, dtype=int)
        return centroid, labels

    if MiniBatchKMeans is None:
        raise RouterError(
            "Missing dependency: scikit-learn. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    k = min(32, max(1, int(math.sqrt(n_examples))))
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=min(1024, max(100, n_examples)),
        n_init=10,
    )
    labels = kmeans.fit_predict(embeddings)
    centroids = normalize_matrix(kmeans.cluster_centers_.astype(np.float32))
    return centroids, labels


def representatives_for_centroids(
    texts: list[str], embeddings: np.ndarray, centroids: np.ndarray, labels: np.ndarray
) -> list[dict[str, Any]]:
    representatives: list[dict[str, Any]] = []
    for centroid_index, centroid in enumerate(centroids):
        member_indexes = np.where(labels == centroid_index)[0]
        if len(member_indexes) == 0:
            member_indexes = np.arange(len(texts))

        member_embeddings = embeddings[member_indexes]
        similarities = member_embeddings @ centroid
        best_member_position = int(np.argmax(similarities))
        best_index = int(member_indexes[best_member_position])
        representatives.append(
            {
                "text": texts[best_index],
                "centroid_index": centroid_index,
            }
        )
    return representatives


def build_facet_signature(
    facet: str, texts: list[str], model: SentenceTransformer
) -> FacetSignature:
    embeddings = embed_texts(model, texts)
    centroids, labels = choose_centroids(embeddings)
    representatives = representatives_for_centroids(texts, embeddings, centroids, labels)
    return FacetSignature(
        facet=facet,
        num_examples=len(texts),
        centroids=centroids,
        representatives=representatives,
        examples=texts,
    )


def signature_to_json(signature: FacetSignature) -> dict[str, Any]:
    return {
        "num_examples": signature.num_examples,
        "centroids": signature.centroids.tolist(),
        "representatives": signature.representatives,
        "examples": [{"text": text} for text in signature.examples],
    }


def load_signatures(path: str | Path, required: bool) -> dict[str, Any]:
    signature_path = Path(path)
    if not signature_path.exists():
        if required:
            raise RouterError(
                f"Signatures file not found: {signature_path}. Run --sign first."
            )
        return {"model": DEFAULT_MODEL, "facets": {}}

    try:
        data = json.loads(signature_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RouterError(f"Invalid signatures JSON in {signature_path}: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("facets"), dict):
        raise RouterError("Signatures file must contain an object with a facets object.")
    return data


def save_signatures(path: str | Path, payload: dict[str, Any]) -> None:
    signature_path = Path(path)
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def grouped_examples(examples: list[RouteExample]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for example in examples:
        grouped.setdefault(example.facet, []).append(example.text)
    return grouped


def existing_facet_examples(signatures: dict[str, Any], facet: str) -> list[str]:
    facet_data = signatures.get("facets", {}).get(facet, {})
    examples = facet_data.get("examples", [])
    texts: list[str] = []
    for item in examples:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            texts.append(str(item["text"]).strip())
    if texts:
        return texts

    # Backward-compatible fallback for signature files without retained examples.
    reps = facet_data.get("representatives", [])
    for item in reps:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            texts.append(str(item["text"]).strip())
    return texts


def print_signature_summary(signatures: dict[str, Any]) -> None:
    table = Table(title="Signature Summary")
    table.add_column("Facet")
    table.add_column("Examples", justify="right")
    table.add_column("Centroids", justify="right")
    table.add_column("Representative Examples")

    for facet, data in sorted(signatures["facets"].items()):
        reps = data.get("representatives", [])
        rep_text = "\n".join(rep.get("text", "") for rep in reps[:3])
        table.add_row(
            facet,
            str(data.get("num_examples", 0)),
            str(len(data.get("centroids", []))),
            rep_text,
        )
    console.print(table)


def sign_dataset(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.input_path)
    grouped = grouped_examples(examples_from_dataset(dataset))
    model = load_model(args.model)

    signatures = {"model": args.model, "facets": {}}
    for facet, texts in sorted(grouped.items()):
        facet_signature = build_facet_signature(facet, texts, model)
        signatures["facets"][facet] = signature_to_json(facet_signature)
        if args.debug:
            console.print(
                f"[dim]{facet}: {len(texts)} examples, "
                f"{len(facet_signature.centroids)} centroids[/dim]"
            )

    save_signatures(args.signatures, signatures)
    print_signature_summary(signatures)
    console.print(f"Saved signatures to [bold]{args.signatures}[/bold]")


def sign_raw_text(args: argparse.Namespace) -> None:
    text = read_text_arg(args.text)
    if not text:
        raise RouterError("Missing text to sign. Provide raw text or use -in dataset.json.")
    if not args.facet:
        raise RouterError("--facet is required when signing raw text.")

    signatures = load_signatures(args.signatures, required=False)
    existing_facets = signatures.get("facets") or {}
    model_name = (signatures.get("model") if existing_facets else args.model) or args.model
    if model_name != args.model and args.debug:
        console.print(
            f"[yellow]Using model from existing signatures: {model_name}[/yellow]"
        )

    facet = args.facet.strip()
    if not facet:
        raise RouterError("--facet must not be empty.")

    previous_texts = existing_facet_examples(signatures, facet)
    all_texts = previous_texts + [text]
    model = load_model(model_name)
    facet_signature = build_facet_signature(facet, all_texts, model)

    signatures["model"] = model_name
    signatures.setdefault("facets", {})[facet] = signature_to_json(facet_signature)
    save_signatures(args.signatures, signatures)
    print_signature_summary(signatures)
    console.print(f"Updated facet [bold]{facet}[/bold] in [bold]{args.signatures}[/bold]")


def best_representative_for_centroid(
    representatives: list[dict[str, Any]], centroid_index: int
) -> str:
    for representative in representatives:
        if representative.get("centroid_index") == centroid_index:
            return str(representative.get("text") or "")
    if representatives:
        return str(representatives[0].get("text") or "")
    return ""


def route_prompt(args: argparse.Namespace) -> None:
    signatures = load_signatures(args.signatures, required=True)
    if not signatures["facets"]:
        raise RouterError(f"No facets found in signatures file: {args.signatures}")

    if args.input_path:
        prompt = read_file(args.input_path)
    else:
        prompt = read_text_arg(args.text) or read_stdin_if_available()

    if not prompt:
        raise RouterError("Missing prompt. Provide -in input.txt, raw text, or stdin.")
    if args.top_k < 1:
        raise RouterError("--top-k must be at least 1.")

    model_name = signatures.get("model") or args.model
    model = load_model(model_name)
    prompt_embedding = embed_texts(model, [prompt])[0]

    results: list[dict[str, Any]] = []
    for facet, data in signatures["facets"].items():
        centroids = np.asarray(data.get("centroids") or [], dtype=np.float32)
        if centroids.size == 0:
            continue
        centroids = normalize_matrix(centroids)
        similarities = centroids @ prompt_embedding
        top_n = min(3, len(similarities))
        top_indexes = np.argsort(similarities)[::-1][:top_n]
        best_index = int(top_indexes[0])
        max_score = float(similarities[best_index])
        avg_top_score = float(np.mean(similarities[top_indexes]))
        final_score = 0.7 * max_score + 0.3 * avg_top_score
        results.append(
            {
                "facet": facet,
                "score": final_score,
                "best_centroid_score": max_score,
                "representative": best_representative_for_centroid(
                    data.get("representatives", []), best_index
                ),
                "centroid_index": best_index,
            }
        )

    if not results:
        raise RouterError("No usable centroids found in signatures file.")

    results.sort(key=lambda item: item["score"], reverse=True)
    top_results = results[: args.top_k]

    table = Table(title="Routing Results")
    table.add_column("Rank", justify="right")
    table.add_column("Facet")
    table.add_column("Score", justify="right")
    table.add_column("Best Centroid", justify="right")
    table.add_column("Best Match")
    for rank, result in enumerate(top_results, start=1):
        table.add_row(
            str(rank),
            result["facet"],
            f"{result['score']:.4f}",
            f"{result['best_centroid_score']:.4f}",
            result["representative"],
        )
    console.print(table)

    if args.debug:
        console.print(f"[dim]Prompt: {prompt}[/dim]")
        console.print(f"[dim]Model: {model_name}[/dim]")
        console.print(f"[dim]Compared {len(results)} facets.[/dim]")


def main() -> int:
    args = parse_args()
    try:
        if args.sign:
            if args.input_path:
                sign_dataset(args)
            else:
                sign_raw_text(args)
        else:
            route_prompt(args)
        return 0
    except RouterError as exc:
        error_console.print(f"[red]Error:[/red] {exc}")
        return 2
    except KeyboardInterrupt:
        error_console.print("\n[red]Interrupted.[/red]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
