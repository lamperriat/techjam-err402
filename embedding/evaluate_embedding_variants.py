from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.v2_embedding import AgentV2Embedding
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from retrieval.embeddings import QwenQueryEmbedder


VARIANTS = {
    "union-only": ("minmax", {"buying": 0.0, "browsing": 0.0}, False),
    "minmax-low": ("minmax", {"buying": 0.03, "browsing": 0.05}, False),
    "margin-initial": ("margin", {"buying": 0.15, "browsing": 0.25}, False),
    "margin-low": ("margin", {"buying": 0.05, "browsing": 0.08}, False),
    "margin-buying-only": ("margin", {"buying": 0.15, "browsing": 0.0}, False),
    "margin-browsing-only": ("margin", {"buying": 0.0, "browsing": 0.25}, False),
    "margin-constrained": ("margin", {"buying": 0.15, "browsing": 0.25}, True),
    "margin-low-constrained": ("margin", {"buying": 0.05, "browsing": 0.08}, True),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate reproducible V2 embedding score-calibration variants."
    )
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--attributes",
        type=Path,
        default=PROJECT_ROOT / "results/catalog_attributes_processed.jsonl",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=PROJECT_ROOT / "results/embeddings/qwen3_embedding_0_6b",
    )
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/public_set.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.embeddings / "manifest.json").read_text(encoding="utf-8"))
    configuration = manifest["configuration"]
    query_embedder = QwenQueryEmbedder(
        str(configuration["model"]),
        int(configuration["dimensions"]),
        int(configuration["max_length"]),
    )
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.variants:
        mode, weights, requires_constraints = VARIANTS[name]
        if not args.quiet:
            print(
                f"Evaluating {name}: mode={mode}, weights={weights}, "
                f"requires_constraints={requires_constraints}",
                file=sys.stderr,
            )
        agent = AgentV2Embedding(
            args.catalog,
            args.attributes,
            args.embeddings,
            query_embedder,
            weights,
            mode,
            requires_constraints,
        )
        try:
            result = evaluate(
                agent,
                samples,
                catalog_ids,
                categories,
                products,
                show_progress=not args.quiet,
            )
        finally:
            agent.close()
        output = args.output_dir / f"v2_embedding_{name}_{args.dataset.stem}.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        summary = {key: value for key, value in result.items() if key != "sessions"}
        print(json.dumps({"variant": name, "output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
