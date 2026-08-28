from __future__ import annotations

"""Re-run the promoted R08 reference wrapper as a post-promotion semantic bridge."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_resources import build_benchmark  # noqa: E402
from scripts.gate_provenance import git_snapshot, sha256  # noqa: E402
from starter.frozen_winner import (  # noqa: E402
    FROZEN_WINNER_ID,
    SELECTION_COMMIT,
    SELECTION_RESULT_SHA256,
)


SCHEMA_VERSION = "p4.promotion-reference-bridge.v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the R08 ArchitectureAgent only as a frozen-to-promoted bridge."
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument(
        "--frozen-resource",
        type=Path,
        default=Path("experiments/p4_r08_resources.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/p4_reference_bridge_resources.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    git_before = git_snapshot(PROJECT_ROOT)
    if git_before["dirty"]:
        raise RuntimeError("promotion reference bridge requires a clean Git worktree")
    frozen = _load(args.frozen_resource)
    frozen_run = frozen["runs"][0]
    benchmark = build_benchmark(
        args.catalog,
        args.dataset,
        runs=2,
        question_policy="fast",
        rerank_mode="off",
        architecture_variant=FROZEN_WINNER_ID,
        rss_sample_ms=10.0,
        verbose=True,
    )
    git_after = git_snapshot(PROJECT_ROOT)
    if git_before != git_after:
        raise RuntimeError("Git state changed during promotion reference bridge")
    exact_official_result = all(
        run["official_result_sha256"] == frozen_run["official_result_sha256"]
        for run in benchmark["runs"]
    )
    checks = {
        "selection_artifact_hash_recorded": bool(SELECTION_RESULT_SHA256),
        "frozen_resource_gate_passed": bool(
            frozen.get("frozen_winner_gate", {}).get("passed")
        ),
        "reference_resource_gate_passed": bool(
            benchmark.get("frozen_winner_gate", {}).get("passed")
        ),
        "exact_official_result_equal_to_pre_promotion_reference": exact_official_result,
        "reference_complete_trace_deterministic": (
            benchmark.get("determinism", {}).get("status") == "passed"
        ),
    }
    artifact = {
        **benchmark,
        "schema_version": SCHEMA_VERSION,
        "evaluation_role": "post_promotion_semantic_reference_bridge",
        "selection_commit": SELECTION_COMMIT,
        "selection_result_sha256": SELECTION_RESULT_SHA256,
        "inputs": {
            "catalog_sha256": sha256(args.catalog),
            "dataset_sha256": sha256(args.dataset),
            "frozen_resource_sha256": sha256(args.frozen_resource),
        },
        "bridge_provenance": {
            "git_preflight": git_before,
            "git_postflight": git_after,
            "current_architecture_source_sha256": sha256(
                PROJECT_ROOT / "starter" / "architecture_lab.py"
            ),
            "current_coverage_source_sha256": sha256(
                PROJECT_ROOT / "starter" / "coverage.py"
            ),
            "boundary": (
                "The immutable pre-promotion artifact anchors selection. This current "
                "ArchitectureAgent rerun is used only to compare exact served responses and "
                "final routes after extracting the shared coverage helper; it does not repeat "
                "architecture selection."
            ),
        },
        "bridge_gate": {
            "checks": checks,
            "passed": all(checks.values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[promotion-reference] wrote {args.output}", flush=True)
    return 0 if artifact["bridge_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
