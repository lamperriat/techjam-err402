from __future__ import annotations

"""Run target-blind architecture ablations on the frozen product-disjoint corpus."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from scripts.evaluate_generalization import (  # noqa: E402
    _delta,
    _samples_sha256,
    _scenario_counts,
    _session_changes,
    build_product_disjoint_samples,
)
from starter.agent import ALLOWED_ATTRIBUTES  # noqa: E402
from starter.architecture_lab import (  # noqa: E402
    CONTROL_ID,
    SCHEMA_VERSION as LAB_SCHEMA_VERSION,
    SPECS,
    ArchitectureAgent,
    ArchitectureSpec,
)


SCHEMA_VERSION = "p4.architecture-search.v1"
DEFAULT_SEED = "track4-p1-product-disjoint-v1"
DEFAULT_DERIVED = PROJECT_ROOT / "experiments" / "p1_derived_product_disjoint.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "p4_architecture_search.json"
EXPECTED_DEFAULT_DERIVED_SHA256 = "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae"
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


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _source_paths() -> dict[str, Path]:
    return {
        "agent": PROJECT_ROOT / "starter" / "agent.py",
        "architecture_lab": PROJECT_ROOT / "starter" / "architecture_lab.py",
        "runner": Path(__file__).resolve(),
        "evaluator": PROJECT_ROOT / "evaluator" / "local_evaluator.py",
        "attributes": PROJECT_ROOT / "starter" / "attributes.py",
        "clarification": PROJECT_ROOT / "starter" / "clarification.py",
        "reranker": PROJECT_ROOT / "starter" / "reranker.py",
        "slot_ledger": PROJECT_ROOT / "starter" / "slot_ledger.py",
        "generalization_helpers": PROJECT_ROOT / "scripts" / "evaluate_generalization.py",
    }


def _git_snapshot() -> dict[str, str | bool]:
    return {
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def _hash_snapshot(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in paths.items()}


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


@dataclass(slots=True)
class ContractRecorder:
    delegate: ArchitectureAgent
    catalog_ids: set[str]
    errors: list[str] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.delegate.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        started = time.perf_counter()
        try:
            response = self.delegate.respond(session_id, user_message, turn, top_k)
            violations = validate_response(response, self.catalog_ids)
            if violations:
                self.errors.extend(f"turn {turn}: {value}" for value in violations)
                raise ValueError("; ".join(violations))
            return response
        except Exception as exc:
            marker = f"turn {turn}: {type(exc).__name__}: {exc}"
            if marker not in self.errors:
                self.errors.append(marker)
            raise
        finally:
            self.latencies_ms.append((time.perf_counter() - started) * 1000.0)


def validate_response(response: object, catalog_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(response, dict):
        return ["response is not an object"]
    allowed_keys = {"message", "ask_attribute", "recommendations", "usage"}
    required_keys = {"message", "ask_attribute", "recommendations"}
    if not required_keys <= set(response):
        errors.append("missing required response keys")
    if set(response) - allowed_keys:
        errors.append("response contains undeclared keys")
    if not isinstance(response.get("message"), str):
        errors.append("message is not a string")
    ask = response.get("ask_attribute")
    if ask is not None and (
        not isinstance(ask, str) or ask not in ALLOWED_ATTRIBUTES
    ):
        errors.append("ask_attribute is outside the official enum")
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list):
        errors.append("recommendations is not an array")
        return errors
    if len(recommendations) > 10:
        errors.append("recommendations exceeds the repository Top-10 invariant")
    identifiers: list[str] = []
    for item in recommendations:
        if not isinstance(item, dict) or set(item) - {"parent_asin", "score"}:
            errors.append("recommendation is not a contract object")
            continue
        identifier = item.get("parent_asin")
        if not isinstance(identifier, str) or identifier not in catalog_ids:
            errors.append("recommendation is outside the frozen catalog")
            continue
        identifiers.append(identifier)
        score = item.get("score")
        if score is not None and (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            errors.append("recommendation score is not a finite number")
    if len(identifiers) != len(set(identifiers)):
        errors.append("recommendations contain duplicate IDs")
    usage = response.get("usage")
    if usage is not None:
        if not isinstance(usage, dict) or set(usage) != {
            "prompt_tokens",
            "completion_tokens",
        }:
            errors.append("usage does not match the contract")
        elif any(
            not isinstance(usage[key], int) or isinstance(usage[key], bool) or usage[key] < 0
            for key in usage
        ):
            errors.append("usage token counts are invalid")
    return errors


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in METRIC_KEYS}


def _functional_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        **_metrics(result),
        "sessions": result.get("sessions", []),
    }


def run_variant(
    spec: ArchitectureSpec,
    catalog_path: Path,
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    print(f"[architecture] {spec.variant_id}: building index", flush=True)
    build_started = time.perf_counter()
    agent = ArchitectureAgent(catalog_path, spec.variant_id, question_policy="fast")
    build_seconds = time.perf_counter() - build_started
    recorder = ContractRecorder(agent, catalog_ids)
    try:
        evaluation_started = time.perf_counter()
        result = evaluate(recorder, samples, catalog_ids, categories, products)
        evaluation_seconds = time.perf_counter() - evaluation_started
        stats = agent.experiment_stats()
    finally:
        agent.connection.close()
    functional = _functional_result(result)
    print(
        f"[architecture] {spec.variant_id}: "
        f"score={result['recommended_technical_score']:.6f} "
        f"HR={result['hit_rate_at_10']:.6f} MRR={result['mrr']:.6f} "
        f"MTTC={result['mttc']:.6f} activations={stats['activations']} "
        f"changes={stats['output_changes']}",
        flush=True,
    )
    return {
        "variant_id": spec.variant_id,
        "family": spec.family,
        "mechanism": spec.mechanism,
        "stage_graph": list(spec.stage_graph),
        "description": spec.description,
        "parameters": dict(spec.parameters),
        "mechanism_fingerprint": _stable_sha256({
            "mechanism": spec.mechanism,
            "stage_graph": spec.stage_graph,
        }),
        "stats": stats,
        "timing": {
            "index_build_seconds": round(build_seconds, 6),
            "evaluation_seconds": round(evaluation_seconds, 6),
            "total_seconds": round(build_seconds + evaluation_seconds, 6),
            "respond_latency": _latency_summary(recorder.latencies_ms),
        },
        "contract_errors": list(recorder.errors),
        "metrics": _metrics(result),
        "functional_result_sha256": _stable_sha256(functional),
        "sessions": result.get("sessions", []),
    }


def gate_variant(run: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    stats = run["stats"]
    effective = stats["activations"] > 0 and stats["output_changes"] > 0
    metrics = run["metrics"]
    baseline = control["metrics"]
    changes = _session_changes(
        {"sessions": run["sessions"]},
        {"sessions": control["sessions"]},
    )
    scenario_regressions = []
    for scenario, values in baseline["scenario_metrics"].items():
        current = metrics["scenario_metrics"].get(scenario, {})
        if float(current.get("hit_rate_at_10", 0.0)) < float(values["hit_rate_at_10"]):
            scenario_regressions.append(scenario)
    gates = {
        "effective_architecture": effective,
        "contract": not run["contract_errors"],
        "complete_evaluation": (
            metrics["sample_count"] == baseline["sample_count"]
            and len(run["sessions"]) == metrics["sample_count"]
        ),
        "hit_rate": metrics["hit_rate_at_10"] >= baseline["hit_rate_at_10"],
        "mrr": metrics["mrr"] >= baseline["mrr"],
        "technical_score": (
            metrics["recommended_technical_score"]
            >= baseline["recommended_technical_score"]
        ),
        "mttc": metrics["mttc"] <= baseline["mttc"],
        "zero_hit_to_miss": changes["hit_to_miss_count"] == 0,
        "scenario_hit_rate": not scenario_regressions,
    }
    if run["variant_id"] == CONTROL_ID:
        decision = (
            "control"
            if gates["contract"] and gates["complete_evaluation"]
            else "invalid_control"
        )
    elif not effective:
        decision = "not_counted"
    elif all(gates.values()):
        decision = "eligible"
    else:
        decision = "reject"
    return {
        "decision": decision,
        "gates": gates,
        "scenario_hit_rate_regressions": scenario_regressions,
        "delta_vs_control": _delta(metrics, baseline),
        "session_changes_vs_control": changes,
    }


def assert_control_integrity(control: dict[str, Any], expected_count: int) -> None:
    errors = []
    if control.get("contract_errors"):
        errors.append("official response contract violations")
    metrics = control.get("metrics", {})
    if metrics.get("sample_count") != expected_count:
        errors.append(
            f"metric sample_count={metrics.get('sample_count')!r}, expected {expected_count}"
        )
    if len(control.get("sessions", [])) != expected_count:
        errors.append(
            f"session_count={len(control.get('sessions', []))}, expected {expected_count}"
        )
    if errors:
        raise RuntimeError("control integrity failure: " + "; ".join(errors))


def _contract_complete(run: dict[str, Any]) -> bool:
    gates = run.get("gates", {})
    return bool(
        gates.get("contract", not run.get("contract_errors"))
        and gates.get("complete_evaluation", True)
    )


def count_effective_non_control(runs: Iterable[dict[str, Any]]) -> int:
    return sum(
        run.get("variant_id") != CONTROL_ID
        and bool(run.get("gates", {}).get("effective_architecture"))
        and _contract_complete(run)
        for run in runs
    )


def _winner_key(run: dict[str, Any]) -> tuple[float, float, float, float, str]:
    metrics = run["metrics"]
    return (
        float(metrics["recommended_technical_score"]),
        float(metrics["hit_rate_at_10"]),
        float(metrics["mrr"]),
        -float(metrics["mttc"]),
        str(run["variant_id"]),
    )


def select_candidates(
    runs: list[dict[str, Any]],
    *,
    confirm_top: int,
) -> dict[str, Any]:
    counted = [
        run
        for run in runs
        if run["decision"] != "not_counted" and _contract_complete(run)
    ]
    if not counted:
        raise RuntimeError("no contract-clean architecture run is available")
    raw = max(counted, key=_winner_key)
    eligible = [
        run
        for run in runs
        if run["decision"] in {"control", "eligible"}
        and _contract_complete(run)
    ]
    if not eligible:
        raise RuntimeError("no contract-clean eligible control or candidate is available")
    eligible_winner = max(eligible, key=_winner_key)
    confirmation = sorted(counted, key=_winner_key, reverse=True)[: max(0, confirm_top)]
    required = [
        eligible_winner,
        next(run for run in runs if run["variant_id"] == CONTROL_ID),
    ]
    confirmation = list({
        run["variant_id"]: run for run in [*confirmation, *required]
    }.values())
    return {
        "raw_score_winner": raw["variant_id"],
        "eligible_winner_before_confirmation": eligible_winner["variant_id"],
        "confirmation_candidates": [run["variant_id"] for run in confirmation],
        "tie_break": [
            "technical_score",
            "hit_rate_at_10",
            "mrr",
            "lower_mttc",
            "variant_id",
        ],
    }


def _parse_variants(value: str) -> list[ArchitectureSpec]:
    if value.strip().lower() == "all":
        return list(SPECS)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    known = {spec.variant_id: spec for spec in SPECS}
    missing = [value for value in requested if value not in known]
    if missing:
        raise ValueError(f"unknown architecture variants: {', '.join(missing)}")
    if CONTROL_ID not in requested:
        requested.insert(0, CONTROL_ID)
    return [known[value] for value in dict.fromkeys(requested)]


def validate_selection_samples(
    samples: list[dict[str, Any]],
    public_samples: list[dict[str, Any]],
    catalog_ids: set[str],
    *,
    expected_count: int,
) -> dict[str, Any]:
    if len(samples) != expected_count:
        raise ValueError(
            f"derived sample count is {len(samples)}, expected {expected_count}"
        )
    sample_ids = [str(sample.get("sample_id") or "") for sample in samples]
    targets = [
        str(sample.get("ground_truth", {}).get("parent_asin") or "")
        for sample in samples
    ]
    scenarios = [str(sample.get("scenario_type") or "") for sample in samples]
    if not all(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("derived sample IDs must be non-empty and unique")
    if not all(targets) or len(targets) != len(set(targets)):
        raise ValueError("derived targets must be non-empty and unique")
    missing = sorted(set(targets) - catalog_ids)
    if missing:
        raise ValueError("derived targets must all exist in the frozen catalog")
    scenario_counts = dict(sorted(Counter(scenarios).items()))
    expected_scenarios = dict(sorted(_scenario_counts(expected_count).items()))
    if scenario_counts != expected_scenarios:
        raise ValueError(
            f"derived scenario mix is {scenario_counts}, expected {expected_scenarios}"
        )
    public_targets = {
        str(sample.get("ground_truth", {}).get("parent_asin") or "")
        for sample in public_samples
    }
    overlap = public_targets & set(targets)
    if overlap:
        raise ValueError("selection corpus overlaps released-public targets")
    return {
        "sample_count": len(samples),
        "samples_sha256": _samples_sha256(samples),
        "unique_sample_count": len(set(sample_ids)),
        "unique_target_count": len(set(targets)),
        "public_target_overlap": 0,
        "scenario_counts": scenario_counts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--public-set", type=Path, default=PROJECT_ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--derived-set", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--derived-count", type=int, default=200)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--confirm-top", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    specs = _parse_variants(args.variants)
    catalog_path = args.catalog.resolve()
    public_path = args.public_set.resolve()
    derived_path = args.derived_set.resolve()
    derived_set_existed = args.derived_set.exists()
    catalog_ids, categories, products = catalog_index(catalog_path)
    public_samples = load_jsonl(public_path)
    if args.derived_set.exists():
        samples = load_jsonl(args.derived_set)
        derived_metadata = validate_selection_samples(
            samples,
            public_samples,
            catalog_ids,
            expected_count=args.derived_count,
        )
        derived_metadata.update({
            "seed": None,
            "seed_status": "not inferred from a pre-existing file",
            "source": str(args.derived_set.resolve()),
        })
        if args.derived_set.resolve() == DEFAULT_DERIVED.resolve():
            if derived_metadata["samples_sha256"] != EXPECTED_DEFAULT_DERIVED_SHA256:
                raise ValueError("default frozen derived corpus hash does not match the registry")
            derived_metadata.update({
                "seed": DEFAULT_SEED,
                "seed_status": "verified by the frozen sample-registry SHA-256",
            })
    else:
        samples, derived_metadata = build_product_disjoint_samples(
            public_samples,
            products,
            args.derived_count,
            args.seed,
        )
        derived_metadata.update(validate_selection_samples(
            samples,
            public_samples,
            catalog_ids,
            expected_count=args.derived_count,
        ))
        derived_metadata["source"] = "deterministically generated in memory"

    source_paths = _source_paths()
    input_paths = {
        "catalog": catalog_path,
        "public_set": public_path,
    }
    if args.derived_set.exists():
        input_paths["derived_set"] = args.derived_set.resolve()
    preflight = {
        "git": _git_snapshot(),
        "source_sha256": _hash_snapshot(source_paths),
        "input_sha256": _hash_snapshot(input_paths),
        "derived_set_path_exists": derived_set_existed,
    }

    runs = [
        run_variant(spec, catalog_path, samples, catalog_ids, categories, products)
        for spec in specs
    ]
    control = next(run for run in runs if run["variant_id"] == CONTROL_ID)
    assert_control_integrity(control, len(samples))
    for run in runs:
        run.update(gate_variant(run, control))
    selection = select_candidates(runs, confirm_top=args.confirm_top)

    confirmations: dict[str, Any] = {}
    spec_by_id = {spec.variant_id: spec for spec in specs}
    for variant_id in selection["confirmation_candidates"]:
        repeated = run_variant(
            spec_by_id[variant_id], catalog_path, samples, catalog_ids, categories, products
        )
        original = next(run for run in runs if run["variant_id"] == variant_id)
        confirmations[variant_id] = {
            "functional_result_sha256": repeated["functional_result_sha256"],
            "first_functional_result_sha256": original["functional_result_sha256"],
            "strict_functional_equal": (
                repeated["functional_result_sha256"]
                == original["functional_result_sha256"]
            ),
            "metrics": repeated["metrics"],
            "timing": repeated["timing"],
            "contract_errors": repeated["contract_errors"],
            "contract_clean": not repeated["contract_errors"],
            "sample_count": repeated["metrics"]["sample_count"],
            "session_count": len(repeated["sessions"]),
        }

    eligible_confirmed = [
        run
        for run in runs
        if run["decision"] in {"control", "eligible"}
        and confirmations.get(run["variant_id"], {}).get("strict_functional_equal")
        and confirmations.get(run["variant_id"], {}).get("contract_clean")
        and confirmations.get(run["variant_id"], {}).get("sample_count") == len(samples)
        and confirmations.get(run["variant_id"], {}).get("session_count") == len(samples)
    ]
    selection["eligible_winner"] = (
        max(eligible_confirmed, key=_winner_key)["variant_id"]
        if eligible_confirmed
        else None
    )
    selection["effective_non_control_count"] = count_effective_non_control(runs)
    selection["requirement_met"] = selection["effective_non_control_count"] >= 10

    postflight = {
        "git": _git_snapshot(),
        "source_sha256": _hash_snapshot(source_paths),
        "input_sha256": _hash_snapshot(input_paths),
        "derived_set_path_exists": args.derived_set.exists(),
    }
    if postflight != preflight:
        raise RuntimeError(
            "source, input, or Git state changed during the architecture matrix; "
            "discard this run and retry from a stable tree"
        )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "architecture_lab_schema_version": LAB_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "selection_corpus": "derived_product_disjoint",
            "public_used_for_selection": False,
            "public_used_only_for_target_exclusion": True,
            "question_policy": "fast",
            "top_k": 10,
            "max_turns": 10,
            "network_required": False,
        },
        "invocation": {
            "catalog": str(catalog_path),
            "public_set": str(public_path),
            "derived_set": str(derived_path),
            "derived_set_existed": derived_set_existed,
            "output": str(args.output.resolve()),
            "derived_count": args.derived_count,
            "seed": args.seed,
            "requested_variants": args.variants,
            "selected_variant_ids": [spec.variant_id for spec in specs],
            "confirm_top": args.confirm_top,
        },
        "provenance": {
            "git_branch": preflight["git"]["branch"],
            "git_commit": preflight["git"]["commit"],
            "git_dirty": preflight["git"]["dirty"],
            "python": sys.version,
            "platform": platform.platform(),
            "sqlite": __import__("sqlite3").sqlite_version,
            "catalog_sha256": preflight["input_sha256"]["catalog"],
            "public_set_sha256": preflight["input_sha256"]["public_set"],
            "derived": derived_metadata,
            "source_sha256": preflight["source_sha256"],
            "postflight_snapshot_equal": True,
        },
        "control": CONTROL_ID,
        "runs": runs,
        "confirmations": confirmations,
        "selection": selection,
        "winner_boundary": (
            "The eligible winner is the best deterministic design under this frozen local "
            "selection protocol; it is not evidence of superiority on the private 800 sessions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[architecture] wrote {args.output.resolve()}", flush=True)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    return 0 if selection["requirement_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
