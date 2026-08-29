from __future__ import annotations

"""Thin, reproducible entry point around the released local evaluator.

The JSON written to ``--output`` is the evaluator result itself, so it remains
compatible with existing result-comparison tooling.  Reproducibility metadata is
written separately to a manifest next to the result.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import (  # noqa: E402
    DEFAULT_P11_MODE,
    P11_MODES,
    RETRIEVAL_MODES,
    Agent,
    resolve_retrieval_mode,
)
from starter.attributes import SCHEMA_VERSION as ATTRIBUTE_SCHEMA_VERSION  # noqa: E402
from starter.clarification import SCHEMA_VERSION as QUESTION_VALUE_SCHEMA_VERSION  # noqa: E402
from starter.reranker import SCORER_VERSION  # noqa: E402
from starter.slot_ledger import SCHEMA_VERSION as SLOT_LEDGER_SCHEMA_VERSION  # noqa: E402


SCHEMA_VERSION = "p11.evaluate-agent.v3"
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


def _close_agent(agent: Any) -> None:
    close = getattr(agent, "close", None)
    if callable(close):
        close()
        return
    agent.connection.close()


def _resolve_p11_sidecar(
    p11_mode: str, p11_sidecar_path: Path | None
) -> Path | None:
    if p11_mode not in {"shadow", "active"}:
        return None
    configured = p11_sidecar_path
    if configured is None:
        environment_path = os.getenv("TECHJAM_P11_SIDECAR_PATH")
        configured = (
            Path(environment_path)
            if environment_path
            else PROJECT_ROOT / "starter" / "assets" / "p11_features.sqlite"
        )
    return configured.resolve()


def run_evaluation(
    catalog_path: Path,
    dataset_path: Path,
    *,
    question_policy: str | None = None,
    rerank_mode: str | None = None,
    retrieval_mode: str | None = None,
    p11_mode: str | None = None,
    p11_sidecar_path: Path | None = None,
    _include_p11_status: bool = False,
) -> (
    tuple[dict[str, Any], float]
    | tuple[dict[str, Any], float, dict[str, Any]]
):
    """Execute the released evaluator without changing its interaction flow."""

    legacy_configuration_requested = any(
        value is not None for value in (question_policy, rerank_mode, retrieval_mode)
    )
    resolved_question_policy = question_policy or "fast"
    resolved_rerank_mode = rerank_mode or "off"
    resolved_p11_mode = (
        p11_mode
        if p11_mode is not None
        else ("off" if legacy_configuration_requested else DEFAULT_P11_MODE)
    )
    if resolved_question_policy not in QUESTION_POLICIES:
        raise ValueError(
            f"question policy must be one of: {', '.join(QUESTION_POLICIES)}"
        )
    if resolved_rerank_mode not in RERANK_MODES:
        raise ValueError(f"rerank mode must be one of: {', '.join(RERANK_MODES)}")
    if resolved_p11_mode not in P11_MODES:
        raise ValueError(f"P11 mode must be one of: {', '.join(P11_MODES)}")
    resolved_retrieval_mode = resolve_retrieval_mode(
        retrieval_mode, resolved_rerank_mode
    )
    resolved_sidecar = _resolve_p11_sidecar(
        resolved_p11_mode, p11_sidecar_path
    )
    if resolved_retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(
            f"retrieval mode must be one of: {', '.join(RETRIEVAL_MODES)}"
        )

    started = time.perf_counter()
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(
        catalog_path,
        llm_client=None,
        question_policy=resolved_question_policy,
        rerank_mode=resolved_rerank_mode,
        retrieval_mode=resolved_retrieval_mode,
        p11_mode=resolved_p11_mode,
        p11_sidecar_path=resolved_sidecar,
    )
    try:
        result = evaluate(agent, samples, catalog_ids, categories, products)
        raw_status = agent._p11_status()
        p11_status = {
            key: raw_status.get(key)
            for key in (
                "configured_mode",
                "effective_mode",
                "fallback",
                "reason_code",
                "identity_verified",
                "schema_version",
                "feature_schema_version",
                "scorer_version",
                "feature_registry_sha256",
                "feature_semantics_sha256",
            )
        }
        p11_status["stats"] = json.loads(
            json.dumps(raw_status.get("stats"), ensure_ascii=False)
        )
        p11_status["sidecar_path"] = (
            str(resolved_sidecar) if resolved_sidecar is not None else None
        )
    finally:
        _close_agent(agent)
    elapsed_seconds = round(time.perf_counter() - started, 6)
    if _include_p11_status:
        return result, elapsed_seconds, p11_status
    return result, elapsed_seconds


def build_manifest(
    catalog_path: Path,
    dataset_path: Path,
    result: dict[str, Any],
    *,
    question_policy: str,
    rerank_mode: str,
    retrieval_mode: str,
    p11_mode: str,
    p11_sidecar_path: Path | None,
    p11_status: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    status_sidecar = p11_status.get("sidecar_path")
    sidecar = Path(status_sidecar) if isinstance(status_sidecar, str) else None
    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "catalog_path": str(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
            "dataset_path": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "question_policy": question_policy,
            "rerank_mode": rerank_mode,
            "retrieval_mode": retrieval_mode,
            "p11_mode": p11_mode,
            "p11_sidecar_path": str(sidecar) if sidecar is not None else None,
            "p11_sidecar_sha256": (
                _sha256(sidecar) if sidecar is not None and sidecar.is_file() else None
            ),
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
            "coverage_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "coverage.py"
            ),
            "p11_bridge_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "p11_bridge.py"
            ),
            "p11_features_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "p11_features.py"
            ),
            "p11_bridge_config_sha256": _sha256(
                PROJECT_ROOT / "configs" / "p11_production_bridge.json"
            ),
            "p11_sidecar_manifest_sha256": _sha256(
                PROJECT_ROOT / "starter" / "assets" / "p11_features.manifest.json"
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
            "p11": p11_status,
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
        "--retrieval-mode",
        choices=RETRIEVAL_MODES,
        help=(
            "control serves weighted RRF; coverage serves the promoted R08 cascade. "
            "Default: coverage when rerank is off, otherwise control."
        ),
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
    parser.add_argument(
        "--p11-mode",
        choices=P11_MODES,
        default=DEFAULT_P11_MODE,
        help=(
            "Frozen P11 bridge mode (default: active). Use off for exact R08 output."
        ),
    )
    parser.add_argument(
        "--p11-sidecar",
        type=Path,
        help="Optional frozen P11 sidecar path for shadow/active mode.",
    )
    return parser


def _resolve_cli_p11_mode(
    args: argparse.Namespace,
    raw_argv: list[str],
) -> str:
    explicit_p11 = any(
        token == "--p11-mode" or token.startswith("--p11-mode=")
        for token in raw_argv
    )
    legacy_flag = any(
        token == name or token.startswith(f"{name}=")
        for token in raw_argv
        for name in ("--question-policy", "--rerank-mode", "--retrieval-mode")
    )
    if not explicit_p11 and legacy_flag:
        return "off"
    return str(args.p11_mode)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    retrieval_mode = resolve_retrieval_mode(args.retrieval_mode, args.rerank_mode)
    p11_mode = _resolve_cli_p11_mode(args, raw_argv)
    result, elapsed_seconds, p11_status = run_evaluation(
        args.catalog,
        args.dataset,
        question_policy=args.question_policy,
        rerank_mode=args.rerank_mode,
        retrieval_mode=retrieval_mode,
        p11_mode=p11_mode,
        p11_sidecar_path=args.p11_sidecar,
        _include_p11_status=True,
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
        retrieval_mode=retrieval_mode,
        p11_mode=p11_mode,
        p11_sidecar_path=args.p11_sidecar,
        p11_status=p11_status,
        elapsed_seconds=elapsed_seconds,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = _metrics(result)
    print(
        f"[evaluate] retrieval={retrieval_mode} rerank={args.rerank_mode} "
        f"p11={p11_status['effective_mode']} "
        f"samples={metrics['sample_count']} "
        f"HR@10={metrics['hit_rate_at_10']:.6f} MRR={metrics['mrr']:.6f} "
        f"MTTC={metrics['mttc']:.6f} "
        f"score={metrics['recommended_technical_score']:.6f}",
        flush=True,
    )
    print(f"[evaluate] wrote {args.output} and {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
