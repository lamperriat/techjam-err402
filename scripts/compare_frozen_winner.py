from __future__ import annotations

"""Compare the sole frozen winner with the current served control after selection."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_generalization import METRIC_KEYS, _session_changes
from starter.frozen_winner import FROZEN_WINNER_ID


SCHEMA_VERSION = "p4.frozen-winner-comparison.v1"
EPSILON = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _public_suites(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        return artifact["corpora"]["released_public"]["suites"]
    except (KeyError, TypeError) as exc:
        raise ValueError("artifact has no released-public suite results") from exc


def _metric_delta(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, float]:
    return {
        key: round(float(candidate[key]) - float(control[key]), 9)
        for key in METRIC_KEYS
    }


def _scenario_hit_regressions(
    candidate: dict[str, Any], control: dict[str, Any]
) -> list[str]:
    regressions: list[str] = []
    control_scenarios = control.get("scenario_metrics") or {}
    candidate_scenarios = candidate.get("scenario_metrics") or {}
    for scenario, baseline in control_scenarios.items():
        current = candidate_scenarios.get(scenario) or {}
        if float(current.get("hit_rate_at_10", 0.0)) + EPSILON < float(
            baseline["hit_rate_at_10"]
        ):
            regressions.append(str(scenario))
    return regressions


def compare(
    control: dict[str, Any],
    candidate: dict[str, Any],
    resources: dict[str, Any],
) -> dict[str, Any]:
    if candidate.get("configuration", {}).get("architecture_variant") != FROZEN_WINNER_ID:
        raise ValueError("candidate is not the sole frozen winner")
    if control.get("configuration", {}).get("architecture_variant") is not None:
        raise ValueError("control artifact is not the default served Agent")

    control_suites = _public_suites(control)
    candidate_suites = _public_suites(candidate)
    if list(control_suites) != list(candidate_suites):
        raise ValueError("control and candidate suite registries differ")

    suite_comparisons: dict[str, Any] = {}
    for name in control_suites:
        baseline = control_suites[name]
        current = candidate_suites[name]
        baseline_metrics = baseline["metrics"]
        current_metrics = current["metrics"]
        baseline_ids = [
            str(value["sample_id"]) for value in baseline["result"].get("sessions", [])
        ]
        current_ids = [
            str(value["sample_id"]) for value in current["result"].get("sessions", [])
        ]
        changes = _session_changes(current["result"], baseline["result"])
        scenario_regressions = _scenario_hit_regressions(
            current_metrics,
            baseline_metrics,
        )
        gates = {
            "paired_session_order_equal": baseline_ids == current_ids,
            "sample_count_equal": (
                current_metrics["sample_count"] == baseline_metrics["sample_count"]
            ),
            "hit_rate_non_regression": (
                float(current_metrics["hit_rate_at_10"]) + EPSILON
                >= float(baseline_metrics["hit_rate_at_10"])
            ),
            "mrr_non_regression": (
                float(current_metrics["mrr"]) + EPSILON
                >= float(baseline_metrics["mrr"])
            ),
            "mttc_non_regression": (
                float(current_metrics["mttc"])
                <= float(baseline_metrics["mttc"]) + EPSILON
            ),
            "technical_score_non_regression": (
                float(current_metrics["recommended_technical_score"]) + EPSILON
                >= float(baseline_metrics["recommended_technical_score"])
            ),
            "zero_control_hit_to_candidate_miss": changes["hit_to_miss_count"] == 0,
            "scenario_hit_rate_non_regression": not scenario_regressions,
            "strict_response_contract_clean": not current.get("contract_errors"),
            "phrase_transform_coverage_valid": bool(
                current.get("transform", {}).get("coverage_valid")
            ),
        }
        suite_comparisons[name] = {
            "control_metrics": baseline_metrics,
            "candidate_metrics": current_metrics,
            "delta_candidate_minus_control": _metric_delta(
                current_metrics,
                baseline_metrics,
            ),
            "session_changes": changes,
            "scenario_hit_rate_regressions": scenario_regressions,
            "gates": gates,
            "passed": all(gates.values()),
        }

    control_config = control.get("configuration", {})
    candidate_config = candidate.get("configuration", {})
    control_git = control.get("provenance", {}).get("preflight", {}).get("git", {})
    candidate_git = candidate.get("provenance", {}).get("preflight", {}).get("git", {})
    resource_gate = resources.get("frozen_winner_gate") or {}
    global_checks = {
        "all_suite_gates_passed": all(
            value["passed"] for value in suite_comparisons.values()
        ),
        "candidate_generalization_gate_passed": bool(
            candidate.get("frozen_winner_gate", {}).get("passed")
        ),
        "candidate_resource_gate_passed": bool(resource_gate.get("passed")),
        "candidate_selection_was_not_repeated_on_public": (
            candidate_config.get("selection_performed") is False
            and candidate_config.get("public_used_for_selection") is False
            and candidate_config.get("candidate_count") == 1
        ),
        "same_underlying_agent_source": (
            control_config.get("agent_source_sha256")
            == candidate_config.get("agent_source_sha256")
        ),
        "same_public_gate_commit": (
            bool(control_git.get("commit"))
            and control_git.get("commit") == candidate_git.get("commit")
        ),
        "candidate_snapshots_stable": bool(
            candidate.get("provenance", {}).get("snapshot_stable")
            and resources.get("provenance", {}).get("snapshot_stable")
        ),
        "complete_trace_determinism_passed": (
            resources.get("determinism", {}).get("status") == "passed"
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "winner": FROZEN_WINNER_ID,
        "decision": "promote" if all(global_checks.values()) else "reject",
        "passed": all(global_checks.values()),
        "global_checks": global_checks,
        "suite_comparisons": suite_comparisons,
        "resource_evidence": {
            "gate": resource_gate,
            "determinism": resources.get("determinism"),
            "runs": [
                {
                    "run_number": run.get("run_number"),
                    "respond_call_count": run.get("respond_call_count"),
                    "respond_latency": run.get("respond_latency"),
                    "memory": run.get("memory"),
                    "timing_seconds": run.get("timing_seconds"),
                    "architecture_stats": run.get("architecture_stats"),
                    "no_key_default": run.get("no_key_default"),
                    "contract_errors": run.get("contract_errors"),
                }
                for run in resources.get("runs", [])
            ],
            "boundary": (
                "Latency and RSS are observed local Windows measurements. The official rules "
                "publish no numeric limit, so these values are not a threshold claim."
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pair the frozen R08 winner with the current control and resource gates."
    )
    parser.add_argument(
        "--control", type=Path, default=Path("experiments/p4_control_public_all.json")
    )
    parser.add_argument(
        "--candidate", type=Path, default=Path("experiments/p4_r08_public_all.json")
    )
    parser.add_argument(
        "--resources", type=Path, default=Path("experiments/p4_r08_resources.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/p4_r08_promotion.json")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = compare(
        _load(args.control),
        _load(args.candidate),
        _load(args.resources),
    )
    artifact["inputs"] = {
        "control": str(args.control.resolve()),
        "control_sha256": _sha256(args.control),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": _sha256(args.candidate),
        "resources": str(args.resources.resolve()),
        "resources_sha256": _sha256(args.resources),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[promotion] {artifact['decision']} {FROZEN_WINNER_ID}; wrote {args.output}",
        flush=True,
    )
    return 0 if artifact["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
