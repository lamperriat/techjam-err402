from __future__ import annotations

"""Thin, reproducible entry point around the released local evaluator.

The JSON written to ``--output`` is the evaluator result itself, so it remains
compatible with existing result-comparison tooling.  Reproducibility metadata is
written separately to a manifest next to the result.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.attributes import SCHEMA_VERSION as ATTRIBUTE_SCHEMA_VERSION  # noqa: E402
from starter.clarification import SCHEMA_VERSION as QUESTION_VALUE_SCHEMA_VERSION  # noqa: E402
from starter.reranker import SCORER_VERSION  # noqa: E402
from starter.slot_ledger import SCHEMA_VERSION as SLOT_LEDGER_SCHEMA_VERSION  # noqa: E402


SCHEMA_VERSION = "p2.evaluate-agent.v1"
QUESTION_POLICIES = ("fast", "boundary", "conservative")
RERANK_MODES = ("off", "shadow", "active")
METRIC_KEYS = (
    "sample_count",
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
    "reported_token_usage",
    "scenario_metrics",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in METRIC_KEYS}


def _default_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.manifest.json")


def run_evaluation(
    catalog_path: Path,
    dataset_path: Path,
    *,
    question_policy: str = "fast",
    rerank_mode: str = "off",
) -> tuple[dict[str, Any], float]:
    """Execute the released evaluator without changing its interaction flow."""

    if question_policy not in QUESTION_POLICIES:
        raise ValueError(
            f"question policy must be one of: {', '.join(QUESTION_POLICIES)}"
        )
    if rerank_mode not in RERANK_MODES:
        raise ValueError(f"rerank mode must be one of: {', '.join(RERANK_MODES)}")

    started = time.perf_counter()
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(
        catalog_path,
        llm_client=None,
        question_policy=question_policy,
        rerank_mode=rerank_mode,
    )
    try:
        result = evaluate(agent, samples, catalog_ids, categories, products)
    finally:
        agent.connection.close()
    return result, round(time.perf_counter() - started, 6)


def build_manifest(
    catalog_path: Path,
    dataset_path: Path,
    result: dict[str, Any],
    *,
    question_policy: str,
    rerank_mode: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "catalog_path": str(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
            "dataset_path": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "question_policy": question_policy,
            "rerank_mode": rerank_mode,
            "network_required": False,
            "llm_client": None,
        },
        "implementation": {
            "runner_source_sha256": _sha256(Path(__file__)),
            "agent_source_sha256": _sha256(PROJECT_ROOT / "starter" / "agent.py"),
            "attribute_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "attributes.py"
            ),
            "reranker_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "reranker.py"
            ),
            "slot_ledger_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "slot_ledger.py"
            ),
            "clarification_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "clarification.py"
            ),
            "evaluator_source_sha256": _sha256(
                PROJECT_ROOT / "evaluator" / "local_evaluator.py"
            ),
            "attribute_schema_version": ATTRIBUTE_SCHEMA_VERSION,
            "reranker_scorer_version": SCORER_VERSION,
            "slot_ledger_schema_version": SLOT_LEDGER_SCHEMA_VERSION,
            "question_value_schema_version": QUESTION_VALUE_SCHEMA_VERSION,
        },
        "run": {
            "elapsed_seconds": elapsed_seconds,
            "official_result_sha256": _stable_sha256(result),
            "metrics": _metrics(result),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the released evaluator against the configurable local Agent."
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Provenance manifest path (default: <output stem>.manifest.json).",
    )
    parser.add_argument(
        "--question-policy",
        choices=QUESTION_POLICIES,
        default="fast",
    )
    parser.add_argument(
        "--rerank-mode",
        choices=RERANK_MODES,
        default="off",
        help=(
            "off preserves P1; shadow computes reranking without serving it; active serves "
            "reranked output (default: off)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result, elapsed_seconds = run_evaluation(
        args.catalog,
        args.dataset,
        question_policy=args.question_policy,
        rerank_mode=args.rerank_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest_path = args.manifest or _default_manifest_path(args.output)
    manifest = build_manifest(
        args.catalog,
        args.dataset,
        result,
        question_policy=args.question_policy,
        rerank_mode=args.rerank_mode,
        elapsed_seconds=elapsed_seconds,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = _metrics(result)
    print(
        f"[evaluate] mode={args.rerank_mode} samples={metrics['sample_count']} "
        f"HR@10={metrics['hit_rate_at_10']:.6f} MRR={metrics['mrr']:.6f} "
        f"MTTC={metrics['mttc']:.6f} "
        f"score={metrics['recommended_technical_score']:.6f}",
        flush=True,
    )
    print(f"[evaluate] wrote {args.output} and {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
