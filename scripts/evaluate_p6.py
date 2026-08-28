from __future__ import annotations

"""Evaluate P6 adaptive-depth retrieval on its frozen selection corpus.

Released-public, P1, and P5 rows are loaded only to prove product exclusion.  The
P6 agent receives the catalog, visible conversation messages, and profile only.
Ground truth is joined to completed shadow-route records solely for post-hoc pool
diagnostics, after all responses and route captures have been produced.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    evaluate,
    load_jsonl,
    materialize_hidden_fields,
)
from scripts.benchmark_resources import PeakRssSampler  # noqa: E402
from scripts.verify_official_assets import (  # noqa: E402
    EXPECTED_CATALOG_SHA256,
    EXPECTED_PUBLIC_BLOB,
    git_blob_sha1,
)
from starter.agent import Agent  # noqa: E402
from starter.p6_lab import (  # noqa: E402
    ACTIVE_ID,
    CONTROL_ID,
    SCHEMA_VERSION as LAB_SCHEMA_VERSION,
    SHADOW_ID,
    SPECS,
    SPEC_BY_ID,
    P6Agent,
)
from starter.response_contract import (  # noqa: E402
    SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION,
    ContractRecorder,
)


SCHEMA_VERSION = "p6.adaptive-depth-selection.v1"
DEFAULT_SELECTION = PROJECT_ROOT / "experiments" / "p6_selection_product_disjoint.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "p6_adaptive_depth_selection.json"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "catalog.jsonl"
DEFAULT_PUBLIC = PROJECT_ROOT / "data" / "public_set.jsonl"
DEFAULT_P1 = PROJECT_ROOT / "experiments" / "p1_derived_product_disjoint.jsonl"
DEFAULT_P5 = PROJECT_ROOT / "experiments" / "p5_selection_product_disjoint.jsonl"
EXPECTED_SELECTION_SHA256 = (
    "27544cdb6ed9495808c35bbab09b4dbadcb88a1d75d162f17bb4fba6ee8841c7"
)
EXPECTED_P1_SAMPLES_SHA256 = (
    "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae"
)
EXPECTED_P5_SHA256 = (
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
RR_SCALE = 2520
CONTRIBUTION_SCALE = 25_200
RESOURCE_RATIO_LIMIT = 1.30
RSS_RATIO_LIMIT = 1.20
RSS_SAMPLE_INTERVAL_MS = 10.0
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
_PROHIBITED_AUDIT_KEYS = {"ground_truth", "target", "target_id", "target_asin"}


class ConfirmationWorkerFailure(RuntimeError):
    """Carry safe partial-process accounting without discarding the selection run."""

    def __init__(
        self,
        worker_id: str,
        attempt_count: int,
        completed_runs: dict[str, dict[str, Any]],
        cause: Exception,
    ) -> None:
        super().__init__(f"confirmation worker {worker_id} failed: {type(cause).__name__}")
        self.worker_id = worker_id
        self.attempt_count = attempt_count
        self.completed_runs = completed_runs
        self.cause = cause


class ResponseCapture:
    """Capture responses in evaluator order without retaining opaque session IDs."""

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


def _samples_sha256(samples: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            json.dumps(
                sample,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


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
    """Every direct implementation dependency whose drift invalidates a run."""

    return {
        "runner": Path(__file__).resolve(),
        "p6_lab": PROJECT_ROOT / "starter" / "p6_lab.py",
        "adaptive_depth": PROJECT_ROOT / "starter" / "adaptive_depth.py",
        "agent": PROJECT_ROOT / "starter" / "agent.py",
        "coverage": PROJECT_ROOT / "starter" / "coverage.py",
        "reranker": PROJECT_ROOT / "starter" / "reranker.py",
        "attributes": PROJECT_ROOT / "starter" / "attributes.py",
        "clarification": PROJECT_ROOT / "starter" / "clarification.py",
        "slot_ledger": PROJECT_ROOT / "starter" / "slot_ledger.py",
        "response_contract": PROJECT_ROOT / "starter" / "response_contract.py",
        "evaluator": PROJECT_ROOT / "evaluator" / "local_evaluator.py",
        "selection_builder": PROJECT_ROOT / "scripts" / "build_p6_selection_corpus.py",
        "generalization_helpers": PROJECT_ROOT / "scripts" / "evaluate_generalization.py",
        "resource_measurement": PROJECT_ROOT / "scripts" / "benchmark_resources.py",
        "official_asset_verifier": PROJECT_ROOT / "scripts" / "verify_official_assets.py",
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
    all_paths = [*source_paths.values(), *input_paths.values()]
    missing = [str(path) for path in all_paths if not path.is_file()]
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
            "source, input, branch, or commit changed during P6 evaluation; discard this run"
        )
    return checks


def assert_clean_preregistered_snapshot(snapshot: dict[str, Any]) -> None:
    """Selection is invalid unless it starts from a named, committed clean tree."""

    git = snapshot.get("git", {})
    failures: list[str] = []
    if not str(git.get("branch") or "").strip():
        failures.append("missing git branch")
    if not str(git.get("commit") or "").strip():
        failures.append("missing git commit")
    if git.get("dirty") or git.get("status_porcelain"):
        failures.append("dirty preregistration tree")
    if failures:
        raise RuntimeError("P6 preregistration snapshot is invalid: " + "; ".join(failures))


def _target_ids(samples: list[dict[str, Any]], label: str) -> list[str]:
    values = [
        str(sample.get("ground_truth", {}).get("parent_asin") or "").strip()
        for sample in samples
    ]
    if not all(values):
        raise ValueError(f"{label} contains an empty ground-truth parent_asin")
    return values


def validate_official_asset_hashes(catalog_path: Path, public_path: Path) -> dict[str, Any]:
    catalog_hash = _sha256(catalog_path)
    public_blob = git_blob_sha1(public_path)
    if catalog_hash != EXPECTED_CATALOG_SHA256:
        raise ValueError(
            "official catalog SHA-256 mismatch: "
            f"{catalog_hash} != {EXPECTED_CATALOG_SHA256}"
        )
    if public_blob != EXPECTED_PUBLIC_BLOB:
        raise ValueError(
            "official released-public normalized Git blob mismatch: "
            f"{public_blob} != {EXPECTED_PUBLIC_BLOB}"
        )
    return {
        "catalog_sha256": catalog_hash,
        "catalog_hash_verified": True,
        "released_public_git_blob_sha1_lf": public_blob,
        "released_public_blob_verified": True,
    }


def validate_selection_samples(
    samples: list[dict[str, Any]],
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    catalog_ids: set[str],
    *,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
    expected_exclusion_count: int = EXPECTED_EXCLUSION_COUNT,
) -> dict[str, Any]:
    """Validate P6 identity, mix, catalog membership, and three exclusions."""

    if len(samples) != expected_count:
        raise ValueError(f"P6 sample count is {len(samples)}, expected {expected_count}")
    sample_ids = [str(sample.get("sample_id") or "") for sample in samples]
    if len(set(sample_ids)) != expected_count:
        raise ValueError("P6 sample IDs must be unique")
    if any(not re.fullmatch(r"derived_p6_\d{4}", value) for value in sample_ids):
        raise ValueError("P6 sample IDs must use the derived_p6_#### namespace")

    selected_targets = _target_ids(samples, "P6 selection corpus")
    if len(set(selected_targets)) != expected_count:
        raise ValueError("P6 selection targets must be unique")
    if set(selected_targets) - catalog_ids:
        raise ValueError("P6 selection targets must all belong to the frozen catalog")
    scenario_counts = dict(
        sorted(Counter(str(sample.get("scenario_type") or "") for sample in samples).items())
    )
    if expected_count == EXPECTED_SAMPLE_COUNT and scenario_counts != EXPECTED_SCENARIOS:
        raise ValueError(
            f"P6 scenario mix is {scenario_counts}, expected {EXPECTED_SCENARIOS}"
        )

    exclusions = {
        "released_public": public_samples,
        "prior_p1_derived": p1_samples,
        "prior_p5_derived": p5_samples,
    }
    exclusion_targets: dict[str, set[str]] = {}
    for name, rows in exclusions.items():
        if len(rows) != expected_exclusion_count:
            raise ValueError(
                f"{name} exclusion count is {len(rows)}, expected {expected_exclusion_count}"
            )
        values = _target_ids(rows, name)
        if len(set(values)) != expected_exclusion_count:
            raise ValueError(f"{name} exclusion targets must be unique")
        exclusion_targets[name] = set(values)
        if set(selected_targets) & exclusion_targets[name]:
            raise ValueError(f"P6 selection targets overlap {name} targets")

    names = list(exclusion_targets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if exclusion_targets[left] & exclusion_targets[right]:
                raise ValueError(f"{left} and {right} exclusion targets overlap")

    return {
        "sample_count": len(samples),
        "unique_sample_id_count": len(set(sample_ids)),
        "unique_target_count": len(set(selected_targets)),
        "scenario_counts": scenario_counts,
        "all_targets_in_catalog": True,
        "released_public_target_overlap": 0,
        "prior_p1_derived_target_overlap": 0,
        "prior_p5_derived_target_overlap": 0,
        "released_public_role": "target_exclusion_only",
        "released_public_evaluated": False,
    }


def load_frozen_inputs(
    catalog_path: Path,
    selection_path: Path,
    public_path: Path,
    p1_path: Path,
    p5_path: Path,
) -> tuple[
    list[dict[str, Any]],
    set[str],
    dict[str, list[str]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    official_assets = validate_official_asset_hashes(catalog_path, public_path)
    selection_hash = _sha256(selection_path)
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise ValueError(
            "frozen P6 selection corpus SHA-256 mismatch: "
            f"{selection_hash} != {EXPECTED_SELECTION_SHA256}"
        )
    if _sha256(p5_path) != EXPECTED_P5_SHA256:
        raise ValueError("frozen P5 exclusion corpus SHA-256 mismatch")

    samples = load_jsonl(selection_path)
    public_samples = load_jsonl(public_path)
    p1_samples = load_jsonl(p1_path)
    p5_samples = load_jsonl(p5_path)
    p1_canonical_hash = _samples_sha256(p1_samples)
    if p1_canonical_hash != EXPECTED_P1_SAMPLES_SHA256:
        raise ValueError("frozen P1 exclusion corpus canonical SHA-256 mismatch")

    catalog_ids, categories, products = catalog_index(catalog_path)
    if len(catalog_ids) != EXPECTED_CATALOG_COUNT:
        raise ValueError(
            f"catalog contains {len(catalog_ids)} unique IDs, expected {EXPECTED_CATALOG_COUNT}"
        )
    validation = validate_selection_samples(
        samples,
        public_samples,
        p1_samples,
        p5_samples,
        catalog_ids,
    )
    validation.update(
        {
            "path": str(selection_path),
            "sha256": selection_hash,
            "expected_sha256": EXPECTED_SELECTION_SHA256,
            "hash_verified": True,
            "catalog_unique_id_count": len(catalog_ids),
            "p1_canonical_samples_sha256": p1_canonical_hash,
            "p1_hash_verified": True,
            "p5_sha256": EXPECTED_P5_SHA256,
            "p5_hash_verified": True,
            "official_assets": official_assets,
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


def _agent_turn_audit(agent: Any) -> list[dict[str, Any]]:
    reader = getattr(agent, "experiment_audit", None)
    if reader is None:
        raise RuntimeError("P6Agent must expose experiment_audit()")
    records = reader()
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise RuntimeError("P6Agent experiment_audit() must return a list of objects")
    return json.loads(json.dumps(records, ensure_ascii=False))


def _memory_summary(
    sampler: PeakRssSampler,
    baseline_rss: int | None,
    peak_rss: int | None,
) -> dict[str, Any]:
    increment = (
        max(0, peak_rss - baseline_rss)
        if peak_rss is not None and baseline_rss is not None
        else None
    )
    return {
        "backend": sampler.backend,
        "sampling_interval_ms": RSS_SAMPLE_INTERVAL_MS,
        "baseline_rss_bytes": baseline_rss,
        "peak_rss_bytes": peak_rss,
        "peak_rss_increment_bytes": increment,
        "available": bool(
            baseline_rss is not None
            and peak_rss is not None
            and increment is not None
            and sampler.backend != "unavailable"
        ),
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
    print(f"[p6] {variant_id}: building index", flush=True)
    gc.collect()
    sampler = PeakRssSampler(RSS_SAMPLE_INTERVAL_MS)
    baseline_rss = sampler.start()
    agent: P6Agent | None = None
    try:
        build_started = time.perf_counter()
        agent = P6Agent(catalog_path, variant_id, question_policy="fast")
        build_seconds = time.perf_counter() - build_started
        capture = ResponseCapture(agent)
        recorder = ContractRecorder(capture, catalog_ids)
        evaluation_started = time.perf_counter()
        result = evaluate(recorder, samples, catalog_ids, categories, products)
        evaluation_seconds = time.perf_counter() - evaluation_started
        stats = agent.experiment_stats()
        turn_audit = _agent_turn_audit(agent)
    finally:
        if agent is not None:
            agent.connection.close()
        peak_rss = sampler.stop()

    metrics = _metrics(result)
    exact_totals = build_exact_totals(result.get("sessions", []))
    functional_hash = _stable_sha256(result)
    print(
        f"[p6] {variant_id}: score={metrics['recommended_technical_score']:.6f} "
        f"HR={metrics['hit_rate_at_10']:.6f} MRR={metrics['mrr']:.6f} "
        f"MTTC={metrics['mttc']:.6f} hash={functional_hash[:12]}",
        flush=True,
    )
    return {
        "variant_id": variant_id,
        "spec": _spec_payload(spec),
        "stats": stats,
        "turn_audit": turn_audit,
        "turn_audit_sha256": _stable_sha256(turn_audit),
        "timing": {
            "index_build_seconds": round(build_seconds, 6),
            "evaluation_seconds": round(evaluation_seconds, 6),
            "total_seconds": round(build_seconds + evaluation_seconds, 6),
            "respond_latency": _latency_summary(recorder.latencies_ms),
        },
        "memory": _memory_summary(sampler, baseline_rss, peak_rss),
        "contract_errors": list(recorder.errors),
        "metrics": metrics,
        "exact_totals": exact_totals,
        "functional_result_sha256": functional_hash,
        **capture.hashes(),
        "response_sessions": capture.sessions,
        "sessions": result.get("sessions", []),
    }


def run_served_reference(
    catalog_path: Path,
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run the exact served R08 configuration used as the P6 reference."""

    print(f"[p6] {SERVED_REFERENCE_ID}: building index", flush=True)
    gc.collect()
    sampler = PeakRssSampler(RSS_SAMPLE_INTERVAL_MS)
    baseline_rss = sampler.start()
    agent: Agent | None = None
    try:
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
        evaluation_started = time.perf_counter()
        result = evaluate(recorder, samples, catalog_ids, categories, products)
        evaluation_seconds = time.perf_counter() - evaluation_started
    finally:
        if agent is not None:
            agent.connection.close()
        peak_rss = sampler.stop()
    metrics = _metrics(result)
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
        "memory": _memory_summary(sampler, baseline_rss, peak_rss),
        "contract_errors": list(recorder.errors),
        "metrics": metrics,
        "exact_totals": build_exact_totals(result.get("sessions", [])),
        "functional_result_sha256": _stable_sha256(result),
        **capture.hashes(),
        "response_sessions": capture.sessions,
        "sessions": result.get("sessions", []),
    }


def _session_exact_values(session: dict[str, Any]) -> dict[str, int]:
    hit = bool(session.get("hit"))
    rank = session.get("best_rank")
    turn = session.get("first_hit_turn")
    if hit:
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= 10:
            raise ValueError("hit session has an invalid Top-10 rank")
        if not isinstance(turn, int) or isinstance(turn, bool) or not 1 <= turn <= 10:
            raise ValueError("hit session has an invalid first-hit turn")
        rr_units = RR_SCALE // rank
        mttc_turn = turn
        contribution_units = (
            CONTRIBUTION_SCALE // 2
            + 3 * rr_units
            + CONTRIBUTION_SCALE // 50 * (11 - turn)
        )
    else:
        if rank is not None or turn is not None:
            raise ValueError("miss session must not report a rank or first-hit turn")
        rr_units = 0
        mttc_turn = 11
        contribution_units = 0
    return {
        "hit": int(hit),
        "rr_x2520": rr_units,
        "mttc_turn": mttc_turn,
        "official_contribution_x25200": contribution_units,
    }


def build_exact_totals(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return integer sufficient statistics for every official score component."""

    scenario_sample_counts: Counter[str] = Counter()
    scenario_hit_counts: Counter[str] = Counter()
    hit_count = rr_sum = mttc_sum = contribution_sum = 0
    for session in sessions:
        values = _session_exact_values(session)
        scenario = str(session.get("scenario_type") or "")
        scenario_sample_counts[scenario] += 1
        scenario_hit_counts[scenario] += values["hit"]
        hit_count += values["hit"]
        rr_sum += values["rr_x2520"]
        mttc_sum += values["mttc_turn"]
        contribution_sum += values["official_contribution_x25200"]
    return {
        "sample_count": len(sessions),
        "hit_count": hit_count,
        "rr_sum_x2520": rr_sum,
        "mttc_turn_sum": mttc_sum,
        "official_contribution_sum_x25200": contribution_sum,
        "scenario_sample_counts": dict(sorted(scenario_sample_counts.items())),
        "scenario_hit_counts": dict(sorted(scenario_hit_counts.items())),
    }


def exact_totals_match_metrics(run: dict[str, Any]) -> bool:
    totals = run.get("exact_totals", {})
    metrics = run.get("metrics", {})
    count = int(totals.get("sample_count", 0))
    if count <= 0 or int(metrics.get("sample_count", -1)) != count:
        return False
    expected = {
        "hit_rate_at_10": round(int(totals["hit_count"]) / count, 6),
        "mrr": round(int(totals["rr_sum_x2520"]) / (RR_SCALE * count), 6),
        "mttc": round(int(totals["mttc_turn_sum"]) / count, 6),
    }
    return all(float(metrics.get(key, -1.0)) == value for key, value in expected.items())


def _is_complete(run: dict[str, Any], expected_sample_ids: set[str]) -> bool:
    sessions = run.get("sessions", [])
    session_ids = [str(item.get("sample_id") or "") for item in sessions]
    return bool(
        run.get("metrics", {}).get("sample_count") == len(expected_sample_ids)
        and len(sessions) == len(expected_sample_ids)
        and len(set(session_ids)) == len(expected_sample_ids)
        and set(session_ids) == expected_sample_ids
        and len(run.get("response_sessions", [])) == len(expected_sample_ids)
    )


def served_reference_bridge(
    control: dict[str, Any],
    served_reference: dict[str, Any],
    expected_sample_ids: set[str],
) -> dict[str, Any]:
    checks = {
        "served_reference_contract": not served_reference.get("contract_errors"),
        "served_reference_complete": _is_complete(served_reference, expected_sample_ids),
        "control_full_functional_hash_equals_served_reference": (
            control.get("functional_result_sha256")
            == served_reference.get("functional_result_sha256")
        ),
        "control_response_trace_equals_served_reference": (
            control.get("response_trace_sha256")
            == served_reference.get("response_trace_sha256")
        ),
        "control_exact_totals_equal_served_reference": (
            control.get("exact_totals") == served_reference.get("exact_totals")
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "served_reference_functional_result_sha256": served_reference.get(
            "functional_result_sha256"
        ),
        "control_functional_result_sha256": control.get("functional_result_sha256"),
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
    if not exact_totals_match_metrics(run):
        failures.append("official metrics disagree with exact session totals")
    if not served_reference_bridge(run, served_reference, expected_sample_ids)["passed"]:
        failures.append("control is not exactly identical to served Agent coverage/off")
    if failures:
        raise RuntimeError("P6 control integrity failure: " + "; ".join(failures))


def _prohibited_audit_key(value: object) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _PROHIBITED_AUDIT_KEYS:
                return normalized
            nested = _prohibited_audit_key(item)
            if nested:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _prohibited_audit_key(item)
            if nested:
                return nested
    return None


def _record_pool(record: dict[str, Any], key: str) -> list[str]:
    values = record.get(key, [])
    if not isinstance(values, list):
        raise ValueError(f"turn audit {key} must be an array")
    result = [str(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"turn audit {key} contains duplicate IDs")
    return result


def validate_turn_audit_alignment(run: dict[str, Any]) -> dict[str, Any]:
    records = run.get("turn_audit", [])
    response_sessions = run.get("response_sessions", [])
    violations: list[str] = []
    if _prohibited_audit_key(records):
        violations.append("target_or_ground_truth_key_present")
    expected_pairs = [
        (session_index, int(item["turn"]))
        for session_index, session in enumerate(response_sessions)
        for item in session
    ]
    actual_pairs: list[tuple[int, int]] = []
    for record in records:
        try:
            actual_pairs.append((int(record["session_index"]), int(record["turn"])))
        except (KeyError, TypeError, ValueError):
            violations.append("invalid_session_turn_coordinate")
            break
    if actual_pairs != expected_pairs:
        violations.append("turn_records_do_not_align_with_ordered_responses")
    return {
        "record_count": len(records),
        "response_turn_count": len(expected_pairs),
        "target_blind": "target_or_ground_truth_key_present" not in violations,
        "ordered_alignment": actual_pairs == expected_pairs,
        "violations": violations,
        "passed": not violations,
    }


def validate_active_invariants(run: dict[str, Any]) -> dict[str, Any]:
    """Independently verify the frozen deep-tail promotion contract."""

    alignment = validate_turn_audit_alignment(run)
    violations = list(alignment["violations"])
    changed_count = active_count = newcomer_count = 0
    for index, record in enumerate(run.get("turn_audit", [])):
        try:
            base_pool = _record_pool(record, "base_pool")
            deep_pool = _record_pool(record, "deep_pool")
            base_union_pool = _record_pool(record, "base_union_pool")
            baseline = _record_pool(record, "baseline_top10")
            proposal = _record_pool(record, "proposal_top10")
            served = _record_pool(record, "served_top10")
        except ValueError as exc:
            violations.append(f"record_{index}:{exc}")
            continue
        deep_query_executed = bool(record.get("deep_query_executed"))
        if len(base_pool) > 120 or len(deep_pool) > 240:
            violations.append(f"record_{index}:pool_depth_exceeded")
        if deep_query_executed and len(base_pool) != 120:
            violations.append(f"record_{index}:deep_query_without_exact_base120")
        if deep_query_executed and deep_pool[: len(base_pool)] != base_pool:
            violations.append(f"record_{index}:base_not_exact_deep_prefix")
        if not deep_query_executed and deep_pool:
            violations.append(f"record_{index}:deep_pool_without_query")
        active = bool(record.get("active"))
        changed = proposal != baseline
        active_count += int(active)
        changed_count += int(changed)
        if run.get("variant_id") == ACTIVE_ID and served != proposal:
            violations.append(f"record_{index}:active_did_not_serve_proposal")
        if run.get("variant_id") != ACTIVE_ID and served != baseline:
            violations.append(f"record_{index}:nonactive_changed_output")
        if not changed:
            continue
        if not active:
            violations.append(f"record_{index}:changed_without_activation")
        if proposal[:9] != baseline[:9]:
            violations.append(f"record_{index}:top9_not_preserved")
        if not (len(baseline) == len(proposal) == len(served) == 10):
            violations.append(f"record_{index}:changed_top10_length_not_exact")
        newcomers = [value for value in proposal if value not in baseline]
        newcomer_count += len(newcomers)
        if len(newcomers) > 1:
            violations.append(f"record_{index}:more_than_one_newcomer")
        if not newcomers:
            violations.append(f"record_{index}:change_without_newcomer")
            continue
        newcomer = newcomers[0]
        if len(proposal) != 10 or proposal[9] != newcomer:
            violations.append(f"record_{index}:newcomer_not_exactly_rank10")
        if newcomer in base_union_pool:
            violations.append(f"record_{index}:newcomer_overlaps_base_union")
        if newcomer not in deep_pool[120:]:
            violations.append(f"record_{index}:newcomer_not_from_deep_tail")
        coverage = record.get("coverage_by_parent_asin", {})
        if not isinstance(coverage, dict) or not baseline:
            violations.append(f"record_{index}:missing_coverage_evidence")
        elif int(coverage.get(newcomer, -1)) <= int(coverage.get(baseline[-1], -1)):
            violations.append(f"record_{index}:coverage_not_strictly_better")
        excluded_matches = record.get("matched_excluded_terms_by_parent_asin", {})
        if not isinstance(excluded_matches, dict) or excluded_matches.get(newcomer):
            violations.append(f"record_{index}:excluded_term_match")
    unique_violations = list(dict.fromkeys(violations))
    return {
        "record_count": len(run.get("turn_audit", [])),
        "activation_count": active_count,
        "changed_turn_count": changed_count,
        "newcomer_count": newcomer_count,
        "top9_preserved": not any("top9_not_preserved" in item for item in unique_violations),
        "at_most_one_newcomer": not any(
            "more_than_one_newcomer" in item for item in unique_violations
        ),
        "strict_coverage_advantage": not any(
            "coverage_not_strictly_better" in item or "missing_coverage" in item
            for item in unique_violations
        ),
        "excluded_terms_respected": not any(
            "excluded_term_match" in item for item in unique_violations
        ),
        "base_is_deep_prefix": not any(
            "base_not_exact_deep_prefix" in item for item in unique_violations
        ),
        "target_blind_and_aligned": alignment["passed"],
        "violations": unique_violations,
        "passed": not unique_violations,
    }


def build_posthoc_pool_audit(
    shadow_run: dict[str, Any],
    samples: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Join labels only after a complete target-blind shadow capture."""

    records = json.loads(json.dumps(shadow_run.get("turn_audit", [])))
    alignment = validate_turn_audit_alignment({**shadow_run, "turn_audit": records})
    if not alignment["passed"]:
        raise RuntimeError("cannot join post-hoc labels to invalid shadow turn records")
    if len(shadow_run.get("sessions", [])) != len(samples):
        raise RuntimeError("shadow sessions and P6 samples are not aligned")

    # Ground truth is intentionally first read after the immutable route copy above.
    targets = _target_ids(samples, "P6 post-hoc pool audit")
    effective_behaviors = [
        materialize_hidden_fields(sample, products)[1] for sample in samples
    ]
    base_hits = deep_hits = deep_only = eligible = excluded_pre_override = 0
    base_recalled_sessions: set[int] = set()
    deep_recalled_sessions: set[int] = set()
    for record in records:
        session_index = int(record["session_index"])
        turn = int(record["turn"])
        sample = samples[session_index]
        if str(sample.get("scenario_type")) == "intent_override":
            override_turn = int(
                effective_behaviors[session_index]
                .get("override", {})
                .get("turn", 3)
            )
            if turn < override_turn:
                excluded_pre_override += 1
                continue
        if not bool(record.get("deep_query_executed")):
            continue
        eligible += 1
        base_pool = _record_pool(record, "base_union_pool")
        deep_pool = _record_pool(record, "deep_union_pool")
        target = targets[session_index]
        in_base = target in base_pool
        in_deep = target in deep_pool
        base_hits += int(in_base)
        deep_hits += int(in_deep)
        deep_only += int(in_deep and not in_base)
        if in_base:
            base_recalled_sessions.add(session_index)
        if in_deep:
            deep_recalled_sessions.add(session_index)
    rescued_sessions = deep_recalled_sessions - base_recalled_sessions
    rescued_sample_ids = sorted(str(samples[index].get("sample_id") or "") for index in rescued_sessions)
    return {
        "source_variant_id": shadow_run.get("variant_id"),
        "target_blind_turn_record_sha256": _stable_sha256(records),
        "captured_turn_count": len(records),
        "eligible_posthoc_turn_count": eligible,
        "intent_override_pre_switch_turns_excluded": excluded_pre_override,
        "base120_target_present_turn_count": base_hits,
        "deep240_target_present_turn_count": deep_hits,
        "deep_only_target_recovery_turn_count": deep_only,
        "base_union_recalled_session_count": len(base_recalled_sessions),
        "deep_union_recalled_session_count": len(deep_recalled_sessions),
        "rescued_session_count": len(rescued_sessions),
        "rescued_sample_ids_sha256": _stable_sha256(rescued_sample_ids),
        "rescued_sample_ids_recorded": False,
        "label_join_phase": "after_all_responses_and_route_captures",
        "labels_exposed_to_agent": False,
        "per_target_identifiers_recorded": False,
        "alignment": alignment,
    }


def _session_change_audit(
    run: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    current = {str(item["sample_id"]): item for item in run.get("sessions", [])}
    baseline = {str(item["sample_id"]): item for item in control.get("sessions", [])}
    if set(current) != set(baseline):
        raise ValueError("cannot compare runs with different session IDs")
    hit_to_miss = miss_to_hit = contribution_regressions = 0
    contribution_improvements = rank_improvements = earlier_hits = 0
    for sample_id in sorted(current):
        now = _session_exact_values(current[sample_id])
        before = _session_exact_values(baseline[sample_id])
        hit_to_miss += int(before["hit"] == 1 and now["hit"] == 0)
        miss_to_hit += int(before["hit"] == 0 and now["hit"] == 1)
        delta = (
            now["official_contribution_x25200"]
            - before["official_contribution_x25200"]
        )
        contribution_regressions += int(delta < 0)
        contribution_improvements += int(delta > 0)
        rank_improvements += int(now["rr_x2520"] > before["rr_x2520"])
        earlier_hits += int(now["mttc_turn"] < before["mttc_turn"])
    return {
        "hit_to_miss_count": hit_to_miss,
        "miss_to_hit_count": miss_to_hit,
        "official_contribution_regression_count": contribution_regressions,
        "official_contribution_improvement_count": contribution_improvements,
        "rank_improvement_count": rank_improvements,
        "earlier_hit_count": earlier_hits,
    }


def _policy_payload(turn: dict[str, Any]) -> dict[str, Any]:
    response = turn.get("response", {})
    if not isinstance(response, dict):
        response = {}
    return {
        "turn": turn.get("turn"),
        "message": response.get("message"),
        "ask_attribute": response.get("ask_attribute"),
        "usage": response.get("usage"),
    }


def policy_common_turn_bridge(
    run: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    """Prove the policy is identical on every turn reached by both variants."""

    current_sessions = run.get("response_sessions", [])
    control_sessions = control.get("response_sessions", [])
    if len(current_sessions) != len(control_sessions):
        return {
            "passed": False,
            "session_count_aligned": False,
            "common_turn_count": 0,
            "mismatch_count": 0,
        }
    current_common: list[dict[str, Any]] = []
    control_common: list[dict[str, Any]] = []
    mismatch_count = 0
    for session_index, (current, baseline) in enumerate(
        zip(current_sessions, control_sessions)
    ):
        for current_turn, baseline_turn in zip(current, baseline):
            current_payload = _policy_payload(current_turn)
            baseline_payload = _policy_payload(baseline_turn)
            current_common.append(
                {"session_index": session_index, **current_payload}
            )
            control_common.append(
                {"session_index": session_index, **baseline_payload}
            )
            mismatch_count += int(current_payload != baseline_payload)
    return {
        "passed": bool(current_common) and mismatch_count == 0,
        "session_count_aligned": True,
        "common_turn_count": len(current_common),
        "mismatch_count": mismatch_count,
        "current_common_policy_sha256": _stable_sha256(current_common),
        "control_common_policy_sha256": _stable_sha256(control_common),
        "recommendations_ignored": True,
        "unequal_session_lengths_allowed": True,
    }


def _runtime_checks(run: dict[str, Any], control: dict[str, Any]) -> dict[str, bool]:
    evaluation = float(run.get("timing", {}).get("evaluation_seconds") or 0.0)
    base_evaluation = float(control.get("timing", {}).get("evaluation_seconds") or 0.0)
    p95 = float(
        run.get("timing", {}).get("respond_latency", {}).get("p95_ms") or 0.0
    )
    base_p95 = float(
        control.get("timing", {}).get("respond_latency", {}).get("p95_ms") or 0.0
    )
    memory = run.get("memory", {})
    base_memory = control.get("memory", {})
    peak_rss = memory.get("peak_rss_bytes")
    base_peak_rss = base_memory.get("peak_rss_bytes")
    rss_increment = memory.get("peak_rss_increment_bytes")
    base_rss_increment = base_memory.get("peak_rss_increment_bytes")
    return {
        "evaluation_time_within_1_30x": bool(
            base_evaluation > 0 and evaluation <= RESOURCE_RATIO_LIMIT * base_evaluation
        ),
        "response_p95_within_1_30x": bool(
            base_p95 > 0 and p95 <= RESOURCE_RATIO_LIMIT * base_p95
        ),
        "peak_rss_increment_within_1_20x": bool(
            memory.get("available") is True
            and base_memory.get("available") is True
            and isinstance(rss_increment, int)
            and isinstance(base_rss_increment, int)
            and base_rss_increment > 0
            and rss_increment <= RSS_RATIO_LIMIT * base_rss_increment
        ),
        "absolute_peak_rss_within_1_20x": bool(
            memory.get("available") is True
            and base_memory.get("available") is True
            and isinstance(peak_rss, int)
            and isinstance(base_peak_rss, int)
            and base_peak_rss > 0
            and peak_rss <= RSS_RATIO_LIMIT * base_peak_rss
        ),
    }


def gate_variant(
    run: dict[str, Any],
    control: dict[str, Any],
    served_reference: dict[str, Any],
    expected_sample_ids: set[str],
    posthoc_pool_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant_id = str(run["variant_id"])
    common = {
        "contract": not run.get("contract_errors"),
        "complete_evaluation": _is_complete(run, expected_sample_ids),
        "exact_totals_match_official_metrics": exact_totals_match_metrics(run),
    }
    totals = run["exact_totals"]
    baseline = control["exact_totals"]
    changes = _session_change_audit(run, control)
    scenario_regressions = [
        scenario
        for scenario, hit_count in baseline["scenario_hit_counts"].items()
        if int(totals["scenario_hit_counts"].get(scenario, -1)) < int(hit_count)
    ]
    invariants: dict[str, Any] | None = None
    if variant_id == CONTROL_ID:
        gates = {
            **common,
            "served_reference_exact": served_reference_bridge(
                run, served_reference, expected_sample_ids
            )["passed"],
        }
        decision = "control" if all(gates.values()) else "invalid_control"
    elif variant_id == SHADOW_ID:
        gates = {
            **common,
            "functional_output_equals_control": (
                run.get("functional_result_sha256")
                == control.get("functional_result_sha256")
            ),
            "response_trace_equals_control": (
                run.get("response_trace_sha256") == control.get("response_trace_sha256")
            ),
            "exact_totals_equal_control": totals == baseline,
            "turn_audit_target_blind_and_aligned": validate_turn_audit_alignment(run)[
                "passed"
            ],
        }
        decision = "shadow_only" if all(gates.values()) else "invalid_shadow"
    elif variant_id == ACTIVE_ID:
        invariants = validate_active_invariants(run)
        policy_bridge = policy_common_turn_bridge(run, control)
        pool_audit = posthoc_pool_audit or {}
        served_resource_all = _runtime_checks(run, served_reference)
        stats = run.get("stats", {})
        gates = {
            **common,
            "effective": (
                int(stats.get("activations", 0)) > 0
                and int(stats.get("output_changes", 0)) > 0
            ),
            "top9_preserved": invariants["top9_preserved"],
            "at_most_one_newcomer": invariants["at_most_one_newcomer"],
            "strict_coverage_advantage": invariants["strict_coverage_advantage"],
            "excluded_terms_respected": invariants["excluded_terms_respected"],
            "base120_is_deep240_prefix": invariants["base_is_deep_prefix"],
            "turn_audit_target_blind_and_aligned": invariants[
                "target_blind_and_aligned"
            ],
            "all_active_invariants": invariants["passed"],
            "policy_common_turns_exact": policy_bridge["passed"],
            "posthoc_pool_alignment": pool_audit.get("alignment", {}).get("passed") is True,
            "deep_union_session_recall_strict_improvement": (
                int(pool_audit.get("deep_union_recalled_session_count", 0))
                > int(pool_audit.get("base_union_recalled_session_count", 0))
            ),
            "deep_union_rescued_session_count_positive": (
                int(pool_audit.get("rescued_session_count", 0)) > 0
            ),
            "evaluation_time_vs_served_within_1_30x": served_resource_all[
                "evaluation_time_within_1_30x"
            ],
            "response_p95_vs_served_within_1_30x": served_resource_all[
                "response_p95_within_1_30x"
            ],
            "absolute_peak_rss_vs_served_within_1_20x": served_resource_all[
                "absolute_peak_rss_within_1_20x"
            ],
            "hit_count_non_decrease": int(totals["hit_count"]) >= int(baseline["hit_count"]),
            "rr_sum_x2520_non_decrease": (
                int(totals["rr_sum_x2520"]) >= int(baseline["rr_sum_x2520"])
            ),
            "mttc_turn_sum_non_increase": (
                int(totals["mttc_turn_sum"]) <= int(baseline["mttc_turn_sum"])
            ),
            "technical_score_strict_exact_improvement": (
                int(totals["official_contribution_sum_x25200"])
                > int(baseline["official_contribution_sum_x25200"])
            ),
            "zero_hit_to_miss": changes["hit_to_miss_count"] == 0,
            "zero_official_contribution_regression": (
                changes["official_contribution_regression_count"] == 0
            ),
            "scenario_hit_counts_non_decrease": not scenario_regressions,
            **_runtime_checks(run, control),
        }
        decision = "eligible" if all(gates.values()) else "reject"
    else:
        raise ValueError(f"unknown P6 variant: {variant_id}")
    return {
        "decision": decision,
        "gates": gates,
        "exact_delta_vs_control": {
            key: int(totals[key]) - int(baseline[key])
            for key in (
                "hit_count",
                "rr_sum_x2520",
                "mttc_turn_sum",
                "official_contribution_sum_x25200",
            )
        },
        "session_changes_vs_control": changes,
        "scenario_hit_count_regressions": scenario_regressions,
        "active_invariants": invariants,
        "policy_common_turn_bridge": (
            policy_common_turn_bridge(run, control)
            if variant_id == ACTIVE_ID
            else None
        ),
    }


def run_single_worker(
    worker_id: str,
    phase: str,
    worker_nonce: str,
    catalog_path: Path,
    selection_path: Path,
    public_path: Path,
    p1_path: Path,
    p5_path: Path,
) -> dict[str, Any]:
    """Run exactly one served reference or P6 variant in this interpreter."""

    allowed = {SERVED_REFERENCE_ID, *VARIANT_ORDER}
    if worker_id not in allowed:
        raise ValueError(f"unknown isolated P6 worker ID: {worker_id}")
    if not re.fullmatch(r"[a-f0-9]{32}", worker_nonce):
        raise ValueError("isolated P6 worker nonce must be 32 lowercase hex characters")
    samples, catalog_ids, categories, products, corpus = load_frozen_inputs(
        catalog_path, selection_path, public_path, p1_path, p5_path
    )
    run = (
        run_served_reference(
            catalog_path,
            samples,
            catalog_ids,
            categories,
            products,
        )
        if worker_id == SERVED_REFERENCE_ID
        else run_variant(
            SPEC_BY_ID[worker_id],
            catalog_path,
            samples,
            catalog_ids,
            categories,
            products,
        )
    )
    return {
        "worker_schema_version": SCHEMA_VERSION,
        "isolated_process": True,
        "worker_id": worker_id,
        "phase": phase,
        "worker_nonce": worker_nonce,
        "pid": os.getpid(),
        "corpus_sha256": corpus["sha256"],
        "run": run,
    }


def run_isolated_worker(
    worker_id: str,
    phase: str,
    catalog_path: Path,
    selection_path: Path,
    public_path: Path,
    p1_path: Path,
    p5_path: Path,
) -> dict[str, Any]:
    """Launch one variant measurement in a fresh interpreter and return its run."""

    worker_nonce = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="track4-p6-worker-") as directory:
        output = Path(directory) / "worker.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--isolated-worker-output",
            str(output),
            "--isolated-worker-id",
            worker_id,
            "--isolated-worker-phase",
            phase,
            "--isolated-worker-nonce",
            worker_nonce,
            "--catalog",
            str(catalog_path),
            "--selection",
            str(selection_path),
            "--public-set",
            str(public_path),
            "--prior-p1",
            str(p1_path),
            "--prior-p5",
            str(p5_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=900)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            raise RuntimeError(
                f"P6 isolated worker {worker_id} ({phase}) timed out"
            ) from error
        if process.returncode != 0 or not output.is_file():
            raise RuntimeError(
                f"P6 isolated worker {worker_id} ({phase}) failed: "
                + (stderr.strip() or stdout.strip())[-1000:]
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
    pid = payload.get("pid")
    checks = {
        "isolated_process": payload.get("isolated_process") is True,
        "worker_id": payload.get("worker_id") == worker_id,
        "phase": payload.get("phase") == phase,
        "worker_nonce": payload.get("worker_nonce") == worker_nonce,
        "child_pid": (
            isinstance(pid, int)
            and pid > 0
            and pid == process.pid
            and pid != os.getpid()
        ),
        "run_variant_id": payload.get("run", {}).get("variant_id") == worker_id,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"P6 isolated worker {worker_id} returned invalid metadata: {checks}"
        )
    run = payload["run"]
    run["worker_process"] = {
        "isolated": True,
        "pid": pid,
        "phase": phase,
        "worker_nonce": worker_nonce,
    }
    return run


def run_clean_confirmation(
    catalog_path: Path,
    selection_path: Path,
    public_path: Path,
    p1_path: Path,
    p5_path: Path,
) -> dict[str, Any]:
    """Repeat served, C00, and R01 in three separate fresh interpreters."""

    worker_ids = (SERVED_REFERENCE_ID, CONTROL_ID, ACTIVE_ID)
    runs: dict[str, dict[str, Any]] = {}
    for attempt_count, worker_id in enumerate(worker_ids, start=1):
        try:
            runs[worker_id] = run_isolated_worker(
                worker_id,
                "confirmation",
                catalog_path,
                selection_path,
                public_path,
                p1_path,
                p5_path,
            )
        except (
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            raise ConfirmationWorkerFailure(
                worker_id,
                attempt_count,
                dict(runs),
                error,
            ) from error
    pids = [int(runs[value]["worker_process"]["pid"]) for value in worker_ids]
    nonces = [str(runs[value]["worker_process"]["worker_nonce"]) for value in worker_ids]
    if len(set(nonces)) != len(worker_ids):
        raise RuntimeError("P6 confirmation workers must have unique parent-issued nonces")
    return {
        "isolated_processes": True,
        "confirmation_worker_process_count": 3,
        "confirmation_worker_attempt_count": 3,
        "worker_pids": pids,
        "distinct_worker_pid_count": len(set(pids)),
        "worker_nonces": nonces,
        "distinct_worker_nonce_count": len(set(nonces)),
        "runs_per_variant_in_confirmation": {
            SERVED_REFERENCE_ID: 1,
            CONTROL_ID: 1,
            ACTIVE_ID: 1,
        },
        "runs": runs,
        "completed_variant_ids": list(worker_ids),
    }


def run_selection_workers(
    catalog_path: Path,
    selection_path: Path,
    public_path: Path,
    p1_path: Path,
    p5_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[int]]:
    """Run served/C00/S00/R01 once each in four isolated interpreters."""

    worker_ids = (SERVED_REFERENCE_ID, *VARIANT_ORDER)
    measured = {
        worker_id: run_isolated_worker(
            worker_id,
            "selection",
            catalog_path,
            selection_path,
            public_path,
            p1_path,
            p5_path,
        )
        for worker_id in worker_ids
    }
    pids = [int(measured[value]["worker_process"]["pid"]) for value in worker_ids]
    nonces = [
        str(measured[value]["worker_process"]["worker_nonce"])
        for value in worker_ids
    ]
    if len(set(nonces)) != len(worker_ids):
        raise RuntimeError("P6 selection workers must have unique parent-issued nonces")
    return (
        measured[SERVED_REFERENCE_ID],
        {variant_id: measured[variant_id] for variant_id in VARIANT_ORDER},
        pids,
    )


def build_confirmation(
    initial_active: dict[str, Any],
    initial_control: dict[str, Any],
    initial_served_reference: dict[str, Any],
    repeated_pair: dict[str, Any],
    expected_sample_ids: set[str],
) -> dict[str, Any]:
    repeated = repeated_pair.get("runs", {})
    repeat_control = repeated.get(CONTROL_ID, {})
    repeat_active = repeated.get(ACTIVE_ID, {})
    repeat_served = repeated.get(SERVED_REFERENCE_ID, {})
    resource = _runtime_checks(repeat_active, repeat_control)
    served_resource_all = _runtime_checks(repeat_active, repeat_served)
    served_resource = {
        "evaluation_time_vs_served_within_1_30x": served_resource_all[
            "evaluation_time_within_1_30x"
        ],
        "response_p95_vs_served_within_1_30x": served_resource_all[
            "response_p95_within_1_30x"
        ],
        "absolute_peak_rss_vs_served_within_1_20x": served_resource_all[
            "absolute_peak_rss_within_1_20x"
        ],
    }
    repeat_invariants = validate_active_invariants(repeat_active)
    repeat_policy_bridge = policy_common_turn_bridge(repeat_active, repeat_control)
    all_worker_pids = [
        initial_served_reference.get("worker_process", {}).get("pid"),
        initial_control.get("worker_process", {}).get("pid"),
        initial_active.get("worker_process", {}).get("pid"),
        repeat_served.get("worker_process", {}).get("pid"),
        repeat_control.get("worker_process", {}).get("pid"),
        repeat_active.get("worker_process", {}).get("pid"),
    ]
    all_worker_nonces = [
        run.get("worker_process", {}).get("worker_nonce")
        for run in (
            initial_served_reference,
            initial_control,
            initial_active,
            repeat_served,
            repeat_control,
            repeat_active,
        )
    ]
    worker_metadata_valid = all(
        isinstance(pid, int) and pid > 0 for pid in all_worker_pids
    ) and all(
        isinstance(nonce, str) and re.fullmatch(r"[a-f0-9]{32}", nonce)
        for nonce in all_worker_nonces
    ) and all(
        run.get("worker_process", {}).get("isolated") is True
        for run in (
            initial_served_reference,
            initial_control,
            initial_active,
            repeat_served,
            repeat_control,
            repeat_active,
        )
    )
    checks = {
        "confirmation_uses_three_isolated_processes": (
            repeated_pair.get("isolated_processes") is True
            and repeated_pair.get("confirmation_worker_process_count") == 3
            and repeated_pair.get("distinct_worker_nonce_count") == 3
        ),
        "worker_metadata_valid": worker_metadata_valid,
        "served_control_active_use_six_unique_worker_nonces_across_two_runs": (
            worker_metadata_valid and len(set(all_worker_nonces)) == 6
        ),
        "served_functional_hash_equal": (
            repeat_served.get("functional_result_sha256")
            == initial_served_reference.get("functional_result_sha256")
        ),
        "served_response_trace_hash_equal": (
            repeat_served.get("response_trace_sha256")
            == initial_served_reference.get("response_trace_sha256")
        ),
        "control_functional_hash_equal": (
            repeat_control.get("functional_result_sha256")
            == initial_control.get("functional_result_sha256")
        ),
        "control_response_trace_hash_equal": (
            repeat_control.get("response_trace_sha256")
            == initial_control.get("response_trace_sha256")
        ),
        "active_functional_hash_equal": (
            repeat_active.get("functional_result_sha256")
            == initial_active.get("functional_result_sha256")
        ),
        "active_response_trace_hash_equal": (
            repeat_active.get("response_trace_sha256")
            == initial_active.get("response_trace_sha256")
        ),
        "control_turn_audit_hash_equal": bool(
            initial_control.get("turn_audit_sha256")
            and repeat_control.get("turn_audit_sha256")
            == initial_control.get("turn_audit_sha256")
        ),
        "active_turn_audit_hash_equal": bool(
            initial_active.get("turn_audit_sha256")
            and repeat_active.get("turn_audit_sha256")
            == initial_active.get("turn_audit_sha256")
        ),
        "active_exact_totals_equal": (
            repeat_active.get("exact_totals") == initial_active.get("exact_totals")
        ),
        "strict_response_contract_clean": (
            not repeat_served.get("contract_errors")
            and not repeat_control.get("contract_errors")
            and not repeat_active.get("contract_errors")
        ),
        "complete_control": _is_complete(repeat_control, expected_sample_ids),
        "complete_active": _is_complete(repeat_active, expected_sample_ids),
        "complete_served_reference": _is_complete(repeat_served, expected_sample_ids),
        "active_invariants_repeat": repeat_invariants["passed"],
        "policy_common_turns_exact_repeat": repeat_policy_bridge["passed"],
        **resource,
        **served_resource,
    }
    return {
        "attempted": True,
        "variant_id": ACTIVE_ID,
        "confirmation_worker_process_count": 3,
        "confirmation_worker_attempt_count": 3,
        "served_control_active_total_worker_process_count": 6,
        "runs_per_variant": {
            SERVED_REFERENCE_ID: 2,
            CONTROL_ID: 2,
            ACTIVE_ID: 2,
        },
        "worker_pids": all_worker_pids,
        "worker_nonces": all_worker_nonces,
        "confirmation_worker_pids": list(repeated_pair.get("worker_pids", [])),
        "confirmation_worker_nonces": list(repeated_pair.get("worker_nonces", [])),
        "completed_variant_ids": list(repeated_pair.get("completed_variant_ids", [])),
        "distinct_worker_pid_count": len(set(all_worker_pids)),
        "distinct_worker_nonce_count": len(set(all_worker_nonces)),
        "checks": checks,
        "passed": all(checks.values()),
        "independent_resource_confirmation": resource,
        "resource_confirmation_vs_served_reference": served_resource,
        "repeat_policy_common_turn_bridge": repeat_policy_bridge,
        "repeat_hashes": {
            "control_functional": repeat_control.get("functional_result_sha256"),
            "control_response": repeat_control.get("response_trace_sha256"),
            "active_functional": repeat_active.get("functional_result_sha256"),
            "active_response": repeat_active.get("response_trace_sha256"),
            "control_turn_audit": repeat_control.get("turn_audit_sha256"),
            "active_turn_audit": repeat_active.get("turn_audit_sha256"),
            "served_functional": repeat_served.get("functional_result_sha256"),
            "served_response": repeat_served.get("response_trace_sha256"),
        },
        "repeat_timings": {
            CONTROL_ID: repeat_control.get("timing"),
            ACTIVE_ID: repeat_active.get("timing"),
            SERVED_REFERENCE_ID: repeat_served.get("timing"),
        },
    }


def attempt_confirmation(
    initial_active: dict[str, Any],
    initial_control: dict[str, Any],
    initial_served_reference: dict[str, Any],
    expected_sample_ids: set[str],
    catalog_path: Path,
    selection_path: Path,
    public_path: Path,
    p1_path: Path,
    p5_path: Path,
) -> dict[str, Any]:
    """Convert confirmation-only worker failures into a retain-control result."""

    repeated_pair: dict[str, Any] | None = None
    try:
        repeated_pair = run_clean_confirmation(
            catalog_path, selection_path, public_path, p1_path, p5_path
        )
        return build_confirmation(
            initial_active,
            initial_control,
            initial_served_reference,
            repeated_pair,
            expected_sample_ids,
        )
    except ConfirmationWorkerFailure as error:
        completed_runs = error.completed_runs
        completed_variant_ids = list(completed_runs)
        return {
            "attempted": True,
            "variant_id": ACTIVE_ID,
            "passed": False,
            "reason": "confirmation_worker_failure",
            "error_class": type(error.cause).__name__,
            "error_summary": str(error)[-300:],
            "failed_worker_id": error.worker_id,
            "confirmation_worker_attempt_count": error.attempt_count,
            "confirmation_worker_process_count": len(completed_runs),
            "completed_variant_ids": completed_variant_ids,
            "confirmation_worker_pids": [
                run.get("worker_process", {}).get("pid")
                for run in completed_runs.values()
            ],
            "confirmation_worker_nonces": [
                run.get("worker_process", {}).get("worker_nonce")
                for run in completed_runs.values()
            ],
            "runs_per_variant": {
                SERVED_REFERENCE_ID: 1 + int(SERVED_REFERENCE_ID in completed_runs),
                CONTROL_ID: 1 + int(CONTROL_ID in completed_runs),
                ACTIVE_ID: 1 + int(ACTIVE_ID in completed_runs),
            },
        }
    except (RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        completed_runs = (repeated_pair or {}).get("runs", {})
        return {
            "attempted": True,
            "variant_id": ACTIVE_ID,
            "passed": False,
            "reason": "confirmation_validation_failure",
            "error_class": type(error).__name__,
            "error_summary": str(error)[-300:],
            "confirmation_worker_attempt_count": int(
                (repeated_pair or {}).get("confirmation_worker_attempt_count", 0)
            ),
            "confirmation_worker_process_count": len(completed_runs),
            "completed_variant_ids": list(completed_runs),
            "confirmation_worker_pids": list(
                (repeated_pair or {}).get("worker_pids", [])
            ),
            "confirmation_worker_nonces": list(
                (repeated_pair or {}).get("worker_nonces", [])
            ),
            "runs_per_variant": {
                SERVED_REFERENCE_ID: 1 + int(SERVED_REFERENCE_ID in completed_runs),
                CONTROL_ID: 1 + int(CONTROL_ID in completed_runs),
                ACTIVE_ID: 1 + int(ACTIVE_ID in completed_runs),
            },
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
        "public_confirmation_allowed": decision == "promote_active",
        "selectable_variant_ids": [CONTROL_ID, ACTIVE_ID],
        "tie_policy": (
            "R01 must strictly improve exact official contribution while satisfying every "
            "safety and two-part resource gate; metric ties retain C00. S00 is diagnostic-only."
        ),
    }


def _artifact_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    """Keep the result auditable without persisting labels or per-turn catalog pools."""

    keys = (
        "variant_id",
        "configuration",
        "spec",
        "stats",
        "timing",
        "memory",
        "contract_errors",
        "metrics",
        "exact_totals",
        "functional_result_sha256",
        "response_trace_sha256",
        "turn_audit_sha256",
        "decision",
        "gates",
        "exact_delta_vs_control",
        "session_changes_vs_control",
        "scenario_hit_count_regressions",
        "active_invariants",
        "policy_common_turn_bridge",
        "worker_process",
    )
    return {key: run[key] for key in keys if key in run}


def run_selection(
    catalog_path: Path,
    selection_path: Path,
    public_path: Path,
    p1_path: Path,
    p5_path: Path,
) -> dict[str, Any]:
    paths = [catalog_path, selection_path, public_path, p1_path, p5_path]
    catalog_path, selection_path, public_path, p1_path, p5_path = (
        path.resolve() for path in paths
    )
    source_paths = _source_paths()
    input_paths = {
        "catalog": catalog_path,
        "p6_selection": selection_path,
        "released_public_exclusion_only": public_path,
        "prior_p1_derived_exclusion_only": p1_path,
        "prior_p5_derived_exclusion_only": p5_path,
    }
    preflight = _capture_snapshot(source_paths, input_paths)
    assert_clean_preregistered_snapshot(preflight)
    samples, _, _, products, corpus = load_frozen_inputs(
        catalog_path, selection_path, public_path, p1_path, p5_path
    )
    expected_sample_ids = {str(sample["sample_id"]) for sample in samples}
    missing_specs = [value for value in VARIANT_ORDER if value not in SPEC_BY_ID]
    if missing_specs:
        raise RuntimeError("P6 spec registry is incomplete: " + ", ".join(missing_specs))

    served_reference, runs, selection_worker_pids = run_selection_workers(
        catalog_path, selection_path, public_path, p1_path, p5_path
    )
    control = runs[CONTROL_ID]
    bridge = served_reference_bridge(control, served_reference, expected_sample_ids)
    assert_control_integrity(control, served_reference, expected_sample_ids)
    posthoc_pool_audit = build_posthoc_pool_audit(
        runs[SHADOW_ID], samples, products
    )
    gates = {
        variant_id: gate_variant(
            run,
            control,
            served_reference,
            expected_sample_ids,
            posthoc_pool_audit,
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
        "confirmation_worker_process_count": 0,
        "confirmation_worker_attempt_count": 0,
        "completed_variant_ids": [],
        "runs_per_variant": {
            SERVED_REFERENCE_ID: 1,
            CONTROL_ID: 1,
            ACTIVE_ID: 1,
        },
    }
    if gates[ACTIVE_ID]["decision"] == "eligible":
        confirmation = attempt_confirmation(
            runs[ACTIVE_ID],
            control,
            served_reference,
            expected_sample_ids,
            catalog_path,
            selection_path,
            public_path,
            p1_path,
            p5_path,
        )

    selection = select_winner(gates, confirmation)
    confirmation_pids = [
        int(value)
        for value in confirmation.get("confirmation_worker_pids", [])
        if isinstance(value, int)
    ]
    all_worker_pids = [*selection_worker_pids, *confirmation_pids]
    selection_worker_nonces = [
        str(run["worker_process"]["worker_nonce"])
        for run in (served_reference, *(runs[value] for value in VARIANT_ORDER))
    ]
    confirmation_nonces = [
        str(value)
        for value in confirmation.get("confirmation_worker_nonces", [])
    ]
    all_worker_nonces = [*selection_worker_nonces, *confirmation_nonces]
    if len(set(all_worker_nonces)) != len(all_worker_nonces):
        raise RuntimeError("P6 experiment requires a unique parent-issued nonce for every run")
    completed_confirmation_variants = set(
        confirmation.get("completed_variant_ids", [])
    )
    runs_per_variant = {
        SERVED_REFERENCE_ID: 1 + int(
            SERVED_REFERENCE_ID in completed_confirmation_variants
        ),
        CONTROL_ID: 1 + int(CONTROL_ID in completed_confirmation_variants),
        SHADOW_ID: 1,
        ACTIVE_ID: 1 + int(ACTIVE_ID in completed_confirmation_variants),
    }
    postflight = _capture_snapshot(source_paths, input_paths)
    stability_checks = _assert_snapshot_stable(preflight, postflight)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "evaluation_role": "fresh_triple_product_disjoint_p6_selection",
            "question_policy": "fast",
            "variant_order": list(VARIANT_ORDER),
            "spec_registry": [_spec_payload(spec) for spec in SPECS],
            "p6_lab_schema_version": LAB_SCHEMA_VERSION,
            "response_contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "resource_ratio_limit": RESOURCE_RATIO_LIMIT,
            "rss_increment_ratio_limit": RSS_RATIO_LIMIT,
            "rss_sample_interval_ms": RSS_SAMPLE_INTERVAL_MS,
            "variant_resource_process_isolation": "one fresh interpreter per run",
            "network_required": False,
            "target_blind": True,
            "released_public_used_for_target_exclusion_only": True,
            "released_public_evaluated": False,
            "released_public_metrics_recorded": False,
        },
        "corpus": corpus,
        "served_reference": _artifact_run_summary(served_reference),
        "served_reference_bridge": bridge,
        "runs": {
            variant_id: _artifact_run_summary(run) for variant_id, run in runs.items()
        },
        "posthoc_pool_audit": posthoc_pool_audit,
        "confirmation": confirmation,
        "selection": selection,
        "worker_execution": {
            "selection_worker_process_count": 4,
            "confirmation_worker_process_count": int(
                confirmation.get("confirmation_worker_process_count", 0)
            ),
            "confirmation_worker_attempt_count": int(
                confirmation.get("confirmation_worker_attempt_count", 0)
            ),
            "completed_worker_process_count": len(all_worker_pids),
            "total_worker_attempt_count": (
                4
                + int(confirmation.get("confirmation_worker_attempt_count", 0))
            ),
            "distinct_worker_pid_count": len(set(all_worker_pids)),
            "historical_worker_pids_all_distinct": (
                len(all_worker_pids) == len(set(all_worker_pids))
            ),
            "worker_pid_reuse_is_allowed": True,
            "distinct_worker_nonce_count": len(set(all_worker_nonces)),
            "all_worker_nonces_distinct": (
                len(all_worker_nonces) == len(set(all_worker_nonces))
            ),
            "runs_per_variant": runs_per_variant,
            "parent_process_measures_no_variant_resources": True,
        },
        "provenance": {
            "preflight": preflight,
            "postflight": postflight,
            "stability_checks": stability_checks,
            "snapshot_stable": all(stability_checks.values()),
        },
        "boundary": (
            "Selection evaluates only the frozen P6 corpus. Released-public, P1, and P5 "
            "rows prove target exclusion only. Labels are joined after shadow responses and "
            "route captures solely for aggregate pool diagnostics; the Agent never receives "
            "targets, public metrics, private data, or online features."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--public-set", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--prior-p1", type=Path, default=DEFAULT_P1)
    parser.add_argument("--prior-p5", type=Path, default=DEFAULT_P5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--isolated-worker-output",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--isolated-worker-id", help=argparse.SUPPRESS)
    parser.add_argument("--isolated-worker-phase", help=argparse.SUPPRESS)
    parser.add_argument("--isolated-worker-nonce", help=argparse.SUPPRESS)
    return parser


def validate_output_path(
    output: Path,
    frozen_inputs: Iterable[Path],
    *,
    must_not_exist: bool = False,
) -> Path:
    resolved = output.resolve()
    protected = {
        *(path.resolve() for path in frozen_inputs),
        *(path.resolve() for path in _source_paths().values()),
    }
    if resolved in protected:
        raise ValueError("P6 output must not overwrite a frozen input or source file")
    if resolved.exists() and resolved.is_dir():
        raise ValueError("P6 output path must be a file, not a directory")
    if must_not_exist and resolved.exists():
        raise ValueError("isolated worker output must be a new temporary file")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    frozen_inputs = (
        args.catalog,
        args.selection,
        args.public_set,
        args.prior_p1,
        args.prior_p5,
    )
    worker_arguments = (
        args.isolated_worker_output,
        args.isolated_worker_id,
        args.isolated_worker_phase,
        args.isolated_worker_nonce,
    )
    if any(value is not None for value in worker_arguments):
        if not all(value is not None for value in worker_arguments):
            raise ValueError(
                "isolated worker output, ID, phase, and nonce must be provided together"
            )
        args.isolated_worker_output = validate_output_path(
            args.isolated_worker_output,
            frozen_inputs,
            must_not_exist=True,
        )
        payload = run_single_worker(
            str(args.isolated_worker_id),
            str(args.isolated_worker_phase),
            str(args.isolated_worker_nonce),
            args.catalog.resolve(),
            args.selection.resolve(),
            args.public_set.resolve(),
            args.prior_p1.resolve(),
            args.prior_p5.resolve(),
        )
        args.isolated_worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.isolated_worker_output.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return 0

    args.output = validate_output_path(
        args.output,
        frozen_inputs,
    )
    artifact = run_selection(
        args.catalog,
        args.selection,
        args.public_set,
        args.prior_p1,
        args.prior_p5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    selection = artifact["selection"]
    print(
        f"[p6] decision={selection['decision']} winner={selection['winner_id']} "
        f"corpus_sha256={artifact['corpus']['sha256']}",
        flush=True,
    )
    print(f"[p6] wrote {args.output}", flush=True)
    return 0 if selection["experiment_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
