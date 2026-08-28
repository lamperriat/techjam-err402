from __future__ import annotations

"""Evaluate the P5 guarded-PRF experiment on its frozen selection corpus.

The released-public corpus is loaded only to prove target exclusion.  It is never
passed to the evaluator by this runner, so P5 selection cannot tune on public scores.
"""

import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from scripts.evaluate_generalization import _delta, _session_changes  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.p5_lab import (  # noqa: E402
    ACTIVE_ID,
    CONTROL_ID,
    SCHEMA_VERSION as LAB_SCHEMA_VERSION,
    SHADOW_ID,
    SPECS,
    SPEC_BY_ID,
    P5Agent,
)
from starter.response_contract import (  # noqa: E402
    SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION,
    ContractRecorder,
)


SCHEMA_VERSION = "p5.prf-selection.v1"
DEFAULT_SELECTION = PROJECT_ROOT / "experiments" / "p5_selection_product_disjoint.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "p5_prf_selection.json"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "catalog.jsonl"
DEFAULT_PUBLIC = PROJECT_ROOT / "data" / "public_set.jsonl"
DEFAULT_PRIOR_DERIVED = (
    PROJECT_ROOT / "experiments" / "p1_derived_product_disjoint.jsonl"
)
EXPECTED_SELECTION_SHA256 = (
    "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c"
)
EXPECTED_SAMPLE_COUNT = 200
EXPECTED_CATALOG_COUNT = 50_000
EXPECTED_EXCLUSION_COUNT = 200
EXPECTED_SCENARIOS = {
    "boundary": 10,
    "browsing": 80,
    "buying": 80,
    "intent_override": 30,
}
VARIANT_ORDER = (CONTROL_ID, SHADOW_ID, ACTIVE_ID)
SERVED_REFERENCE_ID = "served.Agent.coverage_off"
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


class ResponseCapture:
    """Capture ordered response payloads without retaining opaque session IDs."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.sessions: list[list[dict[str, Any]]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions.append([])
        self.delegate.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        response = self.delegate.respond(session_id, user_message, turn, top_k)
        self.sessions[-1].append(
            {
                "turn": turn,
                "response": json.loads(json.dumps(response, ensure_ascii=False)),
            }
        )
        return response

    def hashes(self) -> dict[str, Any]:
        return {
            "response_trace_sha256": _stable_sha256(self.sessions),
            "session_response_sha256": [
                _stable_sha256(session) for session in self.sessions
            ],
        }


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


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in METRIC_KEYS}


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        index = max(1, min(len(ordered), int(len(ordered) * fraction + 0.999999))) - 1
        return round(ordered[index], 6)

    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 6) if values else None,
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "max_ms": round(max(values), 6) if values else None,
    }


def _source_paths() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "p5_lab": PROJECT_ROOT / "starter" / "p5_lab.py",
        "prf": PROJECT_ROOT / "starter" / "prf.py",
        "agent": PROJECT_ROOT / "starter" / "agent.py",
        "coverage": PROJECT_ROOT / "starter" / "coverage.py",
        "reranker": PROJECT_ROOT / "starter" / "reranker.py",
        "attributes": PROJECT_ROOT / "starter" / "attributes.py",
        "clarification": PROJECT_ROOT / "starter" / "clarification.py",
        "slot_ledger": PROJECT_ROOT / "starter" / "slot_ledger.py",
        "response_contract": PROJECT_ROOT / "starter" / "response_contract.py",
        "evaluator": PROJECT_ROOT / "evaluator" / "local_evaluator.py",
        "generalization_helpers": PROJECT_ROOT / "scripts" / "evaluate_generalization.py",
        "selection_builder": PROJECT_ROOT / "scripts" / "build_p5_selection_corpus.py",
    }


def _git_snapshot() -> dict[str, Any]:
    dirty = _git("status", "--porcelain").splitlines()
    return {
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(dirty),
        "status_porcelain": dirty,
    }


def _capture_snapshot(
    source_paths: dict[str, Path], input_paths: dict[str, Path]
) -> dict[str, Any]:
    missing = [str(path) for path in [*source_paths.values(), *input_paths.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("snapshot inputs are missing: " + ", ".join(missing))
    return {
        "git": _git_snapshot(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "input_sha256": {name: _sha256(path) for name, path in input_paths.items()},
    }


def _assert_snapshot_stable(before: dict[str, Any], after: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "git_branch_unchanged": before["git"]["branch"] == after["git"]["branch"],
        "git_commit_unchanged": before["git"]["commit"] == after["git"]["commit"],
        "git_status_unchanged": (
            before["git"]["status_porcelain"] == after["git"]["status_porcelain"]
        ),
        "source_hashes_unchanged": before["source_sha256"] == after["source_sha256"],
        "input_hashes_unchanged": before["input_sha256"] == after["input_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(
            "source, input, branch, or commit changed during P5 evaluation; discard this run"
        )
    return checks


def _target_ids(samples: list[dict[str, Any]], label: str) -> list[str]:
    values = [
        str(sample.get("ground_truth", {}).get("parent_asin") or "").strip()
        for sample in samples
    ]
    if not all(values):
        raise ValueError(f"{label} contains an empty ground-truth parent_asin")
    return values


def validate_selection_samples(
    samples: list[dict[str, Any]],
    public_samples: list[dict[str, Any]],
    prior_derived_samples: list[dict[str, Any]],
    catalog_ids: set[str],
    *,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
    expected_exclusion_count: int = EXPECTED_EXCLUSION_COUNT,
) -> dict[str, Any]:
    """Validate identity, mix, membership, and both target-exclusion boundaries."""

    if len(samples) != expected_count:
        raise ValueError(f"P5 sample count is {len(samples)}, expected {expected_count}")
    sample_ids = [str(sample.get("sample_id") or "") for sample in samples]
    if len(set(sample_ids)) != expected_count:
        raise ValueError("P5 sample IDs must be unique")
    invalid_ids = [value for value in sample_ids if not re.fullmatch(r"derived_p5_\d{4}", value)]
    if invalid_ids:
        raise ValueError("P5 sample IDs must use the derived_p5_#### namespace")

    targets = _target_ids(samples, "P5 selection corpus")
    if len(set(targets)) != expected_count:
        raise ValueError("P5 selection targets must be unique")
    missing_targets = sorted(set(targets) - catalog_ids)
    if missing_targets:
        raise ValueError("P5 selection targets must all belong to the frozen catalog")

    scenario_counts = dict(
        sorted(Counter(str(sample.get("scenario_type") or "") for sample in samples).items())
    )
    if expected_count == EXPECTED_SAMPLE_COUNT and scenario_counts != EXPECTED_SCENARIOS:
        raise ValueError(
            f"P5 scenario mix is {scenario_counts}, expected {EXPECTED_SCENARIOS}"
        )

    if len(public_samples) != expected_exclusion_count:
        raise ValueError(
            f"released-public exclusion count is {len(public_samples)}, expected {expected_exclusion_count}"
        )
    if len(prior_derived_samples) != expected_exclusion_count:
        raise ValueError(
            "prior P1-derived exclusion count is "
            f"{len(prior_derived_samples)}, expected {expected_exclusion_count}"
        )
    public_targets = _target_ids(public_samples, "released-public corpus")
    prior_targets = _target_ids(prior_derived_samples, "prior P1-derived corpus")
    if len(set(public_targets)) != expected_exclusion_count:
        raise ValueError("released-public exclusion targets must be unique")
    if len(set(prior_targets)) != expected_exclusion_count:
        raise ValueError("prior P1-derived exclusion targets must be unique")
    public_overlap = set(targets) & set(public_targets)
    prior_overlap = set(targets) & set(prior_targets)
    if public_overlap:
        raise ValueError("P5 selection targets overlap released-public targets")
    if prior_overlap:
        raise ValueError("P5 selection targets overlap prior P1-derived targets")

    return {
        "sample_count": len(samples),
        "unique_sample_id_count": len(set(sample_ids)),
        "unique_target_count": len(set(targets)),
        "scenario_counts": scenario_counts,
        "all_targets_in_catalog": True,
        "released_public_target_overlap": 0,
        "prior_p1_derived_target_overlap": 0,
        "released_public_role": "target_exclusion_only",
        "released_public_evaluated": False,
    }


def load_frozen_inputs(
    catalog_path: Path,
    selection_path: Path,
    public_path: Path,
    prior_derived_path: Path,
) -> tuple[
    list[dict[str, Any]],
    set[str],
    dict[str, list[str]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    selection_hash = _sha256(selection_path)
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise ValueError(
            "frozen P5 selection corpus SHA-256 mismatch: "
            f"{selection_hash} != {EXPECTED_SELECTION_SHA256}"
        )
    catalog_ids, categories, products = catalog_index(catalog_path)
    if len(catalog_ids) != EXPECTED_CATALOG_COUNT:
        raise ValueError(
            f"catalog contains {len(catalog_ids)} unique IDs, expected {EXPECTED_CATALOG_COUNT}"
        )
    samples = load_jsonl(selection_path)
    public_samples = load_jsonl(public_path)
    prior_samples = load_jsonl(prior_derived_path)
    validation = validate_selection_samples(
        samples,
        public_samples,
        prior_samples,
        catalog_ids,
    )
    validation.update(
        {
            "path": str(selection_path),
            "sha256": selection_hash,
            "expected_sha256": EXPECTED_SELECTION_SHA256,
            "hash_verified": True,
            "catalog_unique_id_count": len(catalog_ids),
        }
    )
    return samples, catalog_ids, categories, products, validation


def _spec_payload(spec: Any) -> dict[str, Any]:
    if hasattr(spec, "as_dict"):
        return dict(spec.as_dict())
    return {
        "variant_id": str(spec.variant_id),
        "family": str(getattr(spec, "family", "")),
        "mechanism": str(getattr(spec, "mechanism", "")),
        "stage_graph": list(getattr(spec, "stage_graph", ())),
        "description": str(getattr(spec, "description", "")),
        "parameters": dict(getattr(spec, "parameters", {})),
    }


def run_variant(
    spec: Any,
    catalog_path: Path,
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    variant_id = str(spec.variant_id)
    print(f"[p5] {variant_id}: building index", flush=True)
    build_started = time.perf_counter()
    agent = P5Agent(catalog_path, variant_id, question_policy="fast")
    build_seconds = time.perf_counter() - build_started
    capture = ResponseCapture(agent)
    recorder = ContractRecorder(capture, catalog_ids)
    try:
        evaluation_started = time.perf_counter()
        result = evaluate(recorder, samples, catalog_ids, categories, products)
        evaluation_seconds = time.perf_counter() - evaluation_started
        stats = agent.experiment_stats()
    finally:
        agent.connection.close()

    metrics = _metrics(result)
    functional_hash = _stable_sha256(result)
    print(
        f"[p5] {variant_id}: score={metrics['recommended_technical_score']:.6f} "
        f"HR={metrics['hit_rate_at_10']:.6f} MRR={metrics['mrr']:.6f} "
        f"MTTC={metrics['mttc']:.6f} hash={functional_hash[:12]}",
        flush=True,
    )
    return {
        "variant_id": variant_id,
        "spec": _spec_payload(spec),
        "stats": stats,
        "timing": {
            "index_build_seconds": round(build_seconds, 6),
            "evaluation_seconds": round(evaluation_seconds, 6),
            "total_seconds": round(build_seconds + evaluation_seconds, 6),
            "respond_latency": _latency_summary(recorder.latencies_ms),
        },
        "contract_errors": list(recorder.errors),
        "metrics": metrics,
        "functional_result_sha256": functional_hash,
        **capture.hashes(),
        "sessions": result.get("sessions", []),
    }


def run_served_reference(
    catalog_path: Path,
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run the exact served R08 configuration used as the P5 control reference."""

    print(f"[p5] {SERVED_REFERENCE_ID}: building index", flush=True)
    build_started = time.perf_counter()
    agent = Agent(
        catalog_path,
        question_policy="fast",
        rerank_mode="off",
        retrieval_mode="coverage",
    )
    build_seconds = time.perf_counter() - build_started
    capture = ResponseCapture(agent)
    recorder = ContractRecorder(capture, catalog_ids)
    try:
        evaluation_started = time.perf_counter()
        result = evaluate(recorder, samples, catalog_ids, categories, products)
        evaluation_seconds = time.perf_counter() - evaluation_started
    finally:
        agent.connection.close()
    metrics = _metrics(result)
    functional_hash = _stable_sha256(result)
    print(
        f"[p5] {SERVED_REFERENCE_ID}: "
        f"score={metrics['recommended_technical_score']:.6f} "
        f"HR={metrics['hit_rate_at_10']:.6f} MRR={metrics['mrr']:.6f} "
        f"MTTC={metrics['mttc']:.6f} hash={functional_hash[:12]}",
        flush=True,
    )
    return {
        "variant_id": SERVED_REFERENCE_ID,
        "configuration": {
            "class": "starter.agent.Agent",
            "question_policy": "fast",
            "rerank_mode": "off",
            "retrieval_mode": "coverage",
        },
        "timing": {
            "index_build_seconds": round(build_seconds, 6),
            "evaluation_seconds": round(evaluation_seconds, 6),
            "total_seconds": round(build_seconds + evaluation_seconds, 6),
            "respond_latency": _latency_summary(recorder.latencies_ms),
        },
        "contract_errors": list(recorder.errors),
        "metrics": metrics,
        "functional_result_sha256": functional_hash,
        **capture.hashes(),
        "sessions": result.get("sessions", []),
    }


def _is_complete(run: dict[str, Any], expected_sample_ids: set[str]) -> bool:
    sessions = run.get("sessions", [])
    session_ids = [str(item.get("sample_id") or "") for item in sessions]
    return bool(
        run.get("metrics", {}).get("sample_count") == len(expected_sample_ids)
        and len(sessions) == len(expected_sample_ids)
        and len(set(session_ids)) == len(expected_sample_ids)
        and set(session_ids) == expected_sample_ids
    )


def served_reference_bridge(
    control: dict[str, Any],
    served_reference: dict[str, Any],
    expected_sample_ids: set[str],
) -> dict[str, Any]:
    checks = {
        "served_reference_contract": not served_reference.get("contract_errors"),
        "served_reference_complete": _is_complete(
            served_reference, expected_sample_ids
        ),
        "control_full_functional_hash_equals_served_reference": (
            control.get("functional_result_sha256")
            == served_reference.get("functional_result_sha256")
        ),
        "control_response_trace_equals_served_reference": (
            control.get("response_trace_sha256")
            == served_reference.get("response_trace_sha256")
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "served_reference_functional_result_sha256": served_reference.get(
            "functional_result_sha256"
        ),
        "control_functional_result_sha256": control.get(
            "functional_result_sha256"
        ),
        "served_reference_response_trace_sha256": served_reference.get(
            "response_trace_sha256"
        ),
        "control_response_trace_sha256": control.get("response_trace_sha256"),
    }


def assert_control_integrity(
    run: dict[str, Any],
    served_reference: dict[str, Any],
    expected_sample_ids: set[str],
) -> None:
    failures: list[str] = []
    if run.get("variant_id") != CONTROL_ID:
        failures.append("wrong control variant ID")
    if run.get("contract_errors"):
        failures.append("strict response contract violations")
    if not _is_complete(run, expected_sample_ids):
        failures.append("incomplete or misaligned session set")
    bridge = served_reference_bridge(run, served_reference, expected_sample_ids)
    if not bridge["passed"]:
        failures.append("control is not functionally identical to served Agent coverage/off")
    if failures:
        raise RuntimeError("P5 control integrity failure: " + "; ".join(failures))


def gate_variant(
    run: dict[str, Any],
    control: dict[str, Any],
    served_reference: dict[str, Any],
    expected_sample_ids: set[str],
) -> dict[str, Any]:
    variant_id = str(run["variant_id"])
    common_gates = {
        "contract": not run.get("contract_errors"),
        "complete_evaluation": _is_complete(run, expected_sample_ids),
    }
    changes = _session_changes(
        {"sessions": run.get("sessions", [])},
        {"sessions": control.get("sessions", [])},
    )
    metrics = run["metrics"]
    baseline = control["metrics"]
    scenario_regressions: list[str] = []
    for scenario, base_values in baseline.get("scenario_metrics", {}).items():
        current_values = metrics.get("scenario_metrics", {}).get(scenario, {})
        if float(current_values.get("hit_rate_at_10", -1.0)) < float(
            base_values.get("hit_rate_at_10", 0.0)
        ):
            scenario_regressions.append(str(scenario))

    if variant_id == CONTROL_ID:
        gates = {
            **common_gates,
            "served_reference_exact": served_reference_bridge(
                run, served_reference, expected_sample_ids
            )["passed"],
        }
        decision = "control" if all(gates.values()) else "invalid_control"
    elif variant_id == SHADOW_ID:
        gates = {
            **common_gates,
            "functional_output_equals_control": (
                run.get("functional_result_sha256")
                == control.get("functional_result_sha256")
            ),
            "response_trace_equals_control": (
                run.get("response_trace_sha256")
                == control.get("response_trace_sha256")
            ),
        }
        decision = "shadow_only" if all(gates.values()) else "invalid_shadow"
    elif variant_id == ACTIVE_ID:
        stats = run.get("stats", {})
        gates = {
            **common_gates,
            "effective": (
                int(stats.get("activations", 0)) > 0
                and int(stats.get("output_changes", 0)) > 0
            ),
            "hit_rate_non_decrease": (
                float(metrics["hit_rate_at_10"])
                >= float(baseline["hit_rate_at_10"])
            ),
            "mrr_non_decrease": float(metrics["mrr"]) >= float(baseline["mrr"]),
            "mttc_non_increase": float(metrics["mttc"]) <= float(baseline["mttc"]),
            "technical_score_strict_improvement": (
                float(metrics["recommended_technical_score"])
                > float(baseline["recommended_technical_score"])
            ),
            "zero_hit_to_miss": changes["hit_to_miss_count"] == 0,
            "scenario_hit_rate_non_decrease": not scenario_regressions,
            "evaluation_time_within_1_30x": (
                float(run["timing"]["evaluation_seconds"])
                <= 1.30 * float(control["timing"]["evaluation_seconds"])
            ),
        }
        decision = "eligible" if all(gates.values()) else "reject"
    else:
        raise ValueError(f"unknown P5 variant: {variant_id}")

    return {
        "decision": decision,
        "gates": gates,
        "delta_vs_control": _delta(metrics, baseline),
        "session_changes_vs_control": changes,
        "scenario_hit_rate_regressions": scenario_regressions,
    }


def build_confirmation(
    initial: dict[str, Any], repeated: dict[str, Any], expected_sample_ids: set[str]
) -> dict[str, Any]:
    checks = {
        "strict_full_functional_hash_equal": (
            repeated.get("functional_result_sha256")
            == initial.get("functional_result_sha256")
        ),
        "strict_response_trace_hash_equal": (
            repeated.get("response_trace_sha256")
            == initial.get("response_trace_sha256")
        ),
        "strict_response_contract_clean": not repeated.get("contract_errors"),
        "complete_evaluation": _is_complete(repeated, expected_sample_ids),
    }
    return {
        "attempted": True,
        "variant_id": ACTIVE_ID,
        "first_functional_result_sha256": initial.get("functional_result_sha256"),
        "repeat_functional_result_sha256": repeated.get("functional_result_sha256"),
        "first_response_trace_sha256": initial.get("response_trace_sha256"),
        "repeat_response_trace_sha256": repeated.get("response_trace_sha256"),
        "checks": checks,
        "passed": all(checks.values()),
        "repeat_run": repeated,
    }


def select_winner(
    gates: dict[str, dict[str, Any]], confirmation: dict[str, Any]
) -> dict[str, Any]:
    control_valid = gates.get(CONTROL_ID, {}).get("decision") == "control"
    shadow_valid = gates.get(SHADOW_ID, {}).get("decision") == "shadow_only"
    active_eligible = gates.get(ACTIVE_ID, {}).get("decision") == "eligible"
    active_confirmed = bool(
        active_eligible and confirmation.get("attempted") and confirmation.get("passed")
    )
    experiment_valid = bool(control_valid and shadow_valid)
    if not experiment_valid:
        winner = None
        decision = "invalid_experiment"
    elif active_confirmed:
        winner = ACTIVE_ID
        decision = "promote_active"
    elif active_eligible:
        winner = CONTROL_ID
        decision = "retain_control_confirmation_failed"
    else:
        winner = CONTROL_ID
        decision = "retain_control_active_rejected"
    return {
        "decision": decision,
        "winner_id": winner,
        "experiment_valid": experiment_valid,
        "active_eligible_before_confirmation": active_eligible,
        "active_confirmed": active_confirmed,
        "shadow_can_win": False,
        "selectable_variant_ids": [CONTROL_ID, ACTIVE_ID],
        "tie_policy": (
            "R01 must strictly improve technical score while satisfying every safety gate; "
            "metric ties retain C00. S00 is diagnostic-only."
        ),
    }


def run_selection(
    catalog_path: Path,
    selection_path: Path,
    public_path: Path,
    prior_derived_path: Path,
) -> dict[str, Any]:
    paths = [catalog_path, selection_path, public_path, prior_derived_path]
    catalog_path, selection_path, public_path, prior_derived_path = (
        path.resolve() for path in paths
    )
    source_paths = _source_paths()
    input_paths = {
        "catalog": catalog_path,
        "p5_selection": selection_path,
        "released_public_exclusion_only": public_path,
        "prior_p1_derived_exclusion_only": prior_derived_path,
    }
    preflight = _capture_snapshot(source_paths, input_paths)
    samples, catalog_ids, categories, products, corpus = load_frozen_inputs(
        catalog_path, selection_path, public_path, prior_derived_path
    )
    expected_sample_ids = {str(sample["sample_id"]) for sample in samples}

    missing_specs = [variant_id for variant_id in VARIANT_ORDER if variant_id not in SPEC_BY_ID]
    if missing_specs:
        raise RuntimeError("P5 spec registry is incomplete: " + ", ".join(missing_specs))
    served_reference = run_served_reference(
        catalog_path,
        samples,
        catalog_ids,
        categories,
        products,
    )
    runs = {
        variant_id: run_variant(
            SPEC_BY_ID[variant_id],
            catalog_path,
            samples,
            catalog_ids,
            categories,
            products,
        )
        for variant_id in VARIANT_ORDER
    }
    control = runs[CONTROL_ID]
    bridge = served_reference_bridge(control, served_reference, expected_sample_ids)
    assert_control_integrity(control, served_reference, expected_sample_ids)
    gates = {
        variant_id: gate_variant(
            run, control, served_reference, expected_sample_ids
        )
        for variant_id, run in runs.items()
    }
    for variant_id, gate in gates.items():
        runs[variant_id].update(gate)

    confirmation: dict[str, Any] = {
        "attempted": False,
        "variant_id": ACTIVE_ID,
        "reason": "active candidate did not pass every first-run gate",
        "passed": False,
    }
    if gates[ACTIVE_ID]["decision"] == "eligible":
        repeated = run_variant(
            SPEC_BY_ID[ACTIVE_ID],
            catalog_path,
            samples,
            catalog_ids,
            categories,
            products,
        )
        confirmation = build_confirmation(
            runs[ACTIVE_ID], repeated, expected_sample_ids
        )

    selection = select_winner(gates, confirmation)
    postflight = _capture_snapshot(source_paths, input_paths)
    stability_checks = _assert_snapshot_stable(preflight, postflight)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "evaluation_role": "fresh_product_disjoint_p5_selection",
            "question_policy": "fast",
            "variant_order": list(VARIANT_ORDER),
            "spec_registry": [_spec_payload(spec) for spec in SPECS],
            "p5_lab_schema_version": LAB_SCHEMA_VERSION,
            "response_contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "network_required": False,
            "target_blind": True,
            "released_public_used_for_target_exclusion_only": True,
            "released_public_evaluated": False,
            "released_public_metrics_recorded": False,
        },
        "corpus": corpus,
        "served_reference": served_reference,
        "served_reference_bridge": bridge,
        "runs": runs,
        "confirmation": confirmation,
        "selection": selection,
        "provenance": {
            "preflight": preflight,
            "postflight": postflight,
            "stability_checks": stability_checks,
            "snapshot_stable": all(stability_checks.values()),
        },
        "boundary": (
            "Selection uses only the frozen catalog-derived P5 corpus. Released-public and "
            "prior P1-derived rows are read only to prove target exclusion. No private data, "
            "public metric, online service, or target-aware feature is used."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--public-set", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--prior-derived", type=Path, default=DEFAULT_PRIOR_DERIVED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = run_selection(
        args.catalog,
        args.selection,
        args.public_set,
        args.prior_derived,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    selection = artifact["selection"]
    print(
        f"[p5] decision={selection['decision']} winner={selection['winner_id']} "
        f"corpus_sha256={artifact['corpus']['sha256']}",
        flush=True,
    )
    print(f"[p5] wrote {args.output}", flush=True)
    return 0 if selection["experiment_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
