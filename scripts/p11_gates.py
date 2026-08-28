"""Deterministic, fail-closed promotion gates for the P11 reranker.

The input contains three independent split bundles named ``primary``,
``confirmation``, and ``uniform_tail``.  Each bundle has ``served``,
``control``, and ``candidate`` runs with this schema::

    {
        "sessions": [... official evaluator session rows ...],
        "metrics": {... official overall metrics ...},
        "scenario_metrics": {... official scenario summaries ...},
        "resources": {
            "wall_seconds": 1.0,
            "p95_latency_ms": 2.0,
            "peak_rss_bytes": 100,
        },
    }

``validation_flags`` is an upstream audit summary.  Every required flag must
be exactly ``True``.  This module does not infer audit facts it cannot observe.

Promotion semantics are intentionally strict:

* primary and confirmation must each improve official TechnicalScore and have
  a paired 95% bootstrap lower bound strictly above zero;
* primary must improve TechnicalScore by at least 0.005 absolute;
* uniform-tail quality must not regress;
* all splits require nondecreasing HR/MRR, nonincreasing MTTC, zero hit-to-miss
  transitions, nondecreasing HR in every scenario, and candidate resources
  bounded against both the served B00 run and control C00 run.

Bootstrap samples the paired per-session, unrounded TechnicalScore
contribution deltas.  The interval is the deterministic nearest-rank
equal-tailed percentile interval.  Official aggregate metrics are separately
validated with :func:`scripts.official_metric_bridge.validate_official_metrics`.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from scripts.official_metric_bridge import (
    rebuild_official_metrics,
    validate_official_metrics,
)


SPLIT_NAMES = ("primary", "confirmation", "uniform_tail")
FRESH_SPLITS = ("primary", "confirmation")
REQUIRED_FLAGS = (
    "exact_repeat",
    "contract_clean",
    "target_blind",
    "network_attempts_zero",
    "token_usage_zero",
    "exceptions_zero",
)
RESOURCE_LIMITS = {
    "wall_seconds": Decimal("1.15"),
    "p95_latency_ms": Decimal("1.20"),
    "peak_rss_bytes": Decimal("1.10"),
}
DEFAULT_BOOTSTRAP_SEED = 20260829
MIN_BOOTSTRAP_RESAMPLES = 10_000
PRIMARY_MIN_SCORE_DELTA = Decimal("0.005")

_OVERALL_FIELDS = (
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "recommended_technical_score",
)
_SCENARIO_FIELDS = ("sample_count", "hit_rate_at_10", "mrr", "mttc")


class P11GateError(ValueError):
    """Raised internally when an input cannot be evaluated safely."""


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise P11GateError(f"{label} must be numeric")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise P11GateError(f"{label} must be numeric") from exc
    if not number.is_finite():
        raise P11GateError(f"{label} must be finite")
    return number


def _session_map(sessions: object, label: str) -> dict[str, Mapping[str, object]]:
    if isinstance(sessions, (str, bytes, bytearray)) or not isinstance(
        sessions, Sequence
    ):
        raise P11GateError(f"{label} sessions must be a sequence")
    indexed: dict[str, Mapping[str, object]] = {}
    for row in sessions:
        if not isinstance(row, Mapping):
            raise P11GateError(f"{label} session rows must be mappings")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise P11GateError(f"{label} sessions require non-empty sample_id")
        if sample_id in indexed:
            raise P11GateError(f"{label} sessions contain duplicate sample_id")
        scenario = row.get("scenario_type")
        if not isinstance(scenario, str) or not scenario:
            raise P11GateError(f"{label} sessions require scenario_type")
        indexed[sample_id] = row
    if not indexed:
        raise P11GateError(f"{label} sessions must be non-empty")
    return indexed


def _expected_scenario_metrics(
    sessions: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in sessions:
        scenario = row.get("scenario_type")
        if not isinstance(scenario, str) or not scenario:
            raise P11GateError("sessions require scenario_type")
        grouped[scenario].append(row)
    expected: dict[str, dict[str, int | float]] = {}
    for scenario in sorted(grouped):
        rebuilt = rebuild_official_metrics(grouped[scenario])
        expected[scenario] = {field: rebuilt[field] for field in _SCENARIO_FIELDS}
    return expected


def _scenario_metrics_valid(sessions: object, observed: object) -> bool:
    if not isinstance(observed, Mapping):
        return False
    try:
        indexed = _session_map(sessions, "scenario metrics")
        expected = _expected_scenario_metrics(list(indexed.values()))
    except Exception:
        return False
    if set(observed) != set(expected):
        return False
    for scenario, values in expected.items():
        actual = observed.get(scenario)
        if not isinstance(actual, Mapping) or set(actual) != set(_SCENARIO_FIELDS):
            return False
        if any(actual[field] != value for field, value in values.items()):
            return False
    return True


def _session_contribution(row: Mapping[str, object]) -> float:
    hit = row.get("hit")
    if not isinstance(hit, bool):
        raise P11GateError("session hit must be boolean")
    if not hit:
        return 0.0
    turn_value = row.get("first_hit_turn")
    if (
        not isinstance(turn_value, int)
        or isinstance(turn_value, bool)
        or not 1 <= turn_value <= 10
    ):
        raise P11GateError("hit session first_hit_turn must be an integer from 1 to 10")
    reciprocal_rank = _decimal(
        row.get("reciprocal_rank"), "session reciprocal_rank"
    )
    if reciprocal_rank <= 0 or reciprocal_rank > 1:
        raise P11GateError("hit session reciprocal_rank must be in (0, 1]")
    return (
        0.50
        + 0.30 * float(reciprocal_rank)
        + 0.20 * ((11.0 - turn_value) / 10.0)
    )


def _paired_rows(
    control_sessions: object,
    candidate_sessions: object,
    label: str,
) -> list[tuple[Mapping[str, object], Mapping[str, object]]]:
    control = _session_map(control_sessions, f"{label}.control")
    candidate = _session_map(candidate_sessions, f"{label}.candidate")
    if set(control) != set(candidate):
        raise P11GateError(f"{label} session identifiers differ")
    pairs: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    for sample_id in sorted(control):
        before = control[sample_id]
        after = candidate[sample_id]
        if before["scenario_type"] != after["scenario_type"]:
            raise P11GateError(f"{label} paired scenario_type differs")
        pairs.append((before, after))
    return pairs


def _split_seed(seed: int, split_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{split_name}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _paired_bootstrap_ci(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, int | float | str]:
    deltas = [
        _session_contribution(after) - _session_contribution(before)
        for before, after in pairs
    ]
    count = len(deltas)
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        total = math.fsum(deltas[rng.randrange(count)] for _ in range(count))
        estimates.append(total / count)
    estimates.sort()
    lower_index = math.ceil(0.025 * resamples) - 1
    upper_index = math.ceil(0.975 * resamples) - 1
    return {
        "method": "paired_nearest_rank_percentile",
        "confidence": 0.95,
        "seed": seed,
        "resamples": resamples,
        "observed_mean": round(math.fsum(deltas) / count, 12),
        "lower": round(estimates[lower_index], 12),
        "upper": round(estimates[upper_index], 12),
    }


def _metric_delta(candidate: Mapping[str, object], control: Mapping[str, object], field: str) -> Decimal:
    return _decimal(candidate[field], f"candidate.{field}") - _decimal(
        control[field], f"control.{field}"
    )


def _ratio(candidate: object, control: object, label: str) -> Decimal:
    numerator = _decimal(candidate, f"candidate {label}")
    denominator = _decimal(control, f"control {label}")
    if numerator < 0 or denominator <= 0:
        raise P11GateError(f"{label} observations must have a positive control")
    return numerator / denominator


def evaluate_p11_gates(
    splits: Mapping[str, object],
    validation_flags: Mapping[str, object],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = MIN_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Return a stable promotion decision and all auditable gate evidence.

    Malformed or unverifiable evidence fails closed instead of promoting.  The
    result always has exactly ``passed``, ``checks``, ``reasons``, ``deltas``,
    and ``ci`` top-level keys.
    """

    checks: dict[str, bool] = {}
    reasons: list[str] = []
    deltas: dict[str, Any] = {}
    confidence_intervals: dict[str, Any] = {}

    def record(name: str, passed: bool, reason: str | None = None) -> None:
        checks[name] = bool(passed)
        if not passed:
            reasons.append(reason or name)

    resample_count_valid = (
        isinstance(bootstrap_resamples, int)
        and not isinstance(bootstrap_resamples, bool)
        and bootstrap_resamples >= MIN_BOOTSTRAP_RESAMPLES
    )
    seed_valid = isinstance(bootstrap_seed, int) and not isinstance(bootstrap_seed, bool)
    record("bootstrap.resamples_at_least_10000", resample_count_valid)
    record("bootstrap.seed_is_integer", seed_valid)

    flags = validation_flags if isinstance(validation_flags, Mapping) else {}
    for flag in REQUIRED_FLAGS:
        record(f"audit.{flag}", flags.get(flag) is True)

    bundles = splits if isinstance(splits, Mapping) else {}
    for split_name in SPLIT_NAMES:
        bundle = bundles.get(split_name)
        bundle_valid = isinstance(bundle, Mapping)
        record(f"{split_name}.bundle_valid", bundle_valid)
        if not bundle_valid:
            continue

        runs: dict[str, Mapping[str, object]] = {}
        run_evidence_valid: dict[str, bool] = {}
        for role in ("served", "control", "candidate"):
            run = bundle.get(role)
            run_valid = isinstance(run, Mapping)
            record(f"{split_name}.{role}.run_valid", run_valid)
            if not run_valid:
                continue
            runs[role] = run
            sessions = run.get("sessions")
            metrics = run.get("metrics")
            bridge_sessions = (
                sessions
                if isinstance(sessions, Sequence)
                and not isinstance(sessions, (str, bytes, bytearray))
                else []
            )
            try:
                bridge = validate_official_metrics(  # required exact official bridge
                    bridge_sessions,
                    metrics if isinstance(metrics, Mapping) else {},
                )
            except Exception as exc:
                bridge = {
                    "passed": False,
                    "failure_reasons": [f"bridge_exception:{type(exc).__name__}"],
                }
            official_valid = bridge.get("passed") is True
            record(
                f"{split_name}.{role}.official_metrics_valid",
                official_valid,
                ";".join(
                    f"{split_name}.{role}.{item}"
                    for item in bridge.get("failure_reasons", [])
                )
                or None,
            )
            scenario_valid = _scenario_metrics_valid(
                sessions, run.get("scenario_metrics")
            )
            record(f"{split_name}.{role}.scenario_metrics_valid", scenario_valid)
            run_evidence_valid[role] = official_valid and scenario_valid

        if set(runs) != {"served", "control", "candidate"}:
            continue
        if not all(
            run_evidence_valid.get(role, False)
            for role in ("served", "control", "candidate")
        ):
            record(f"{split_name}.verified_evidence_available", False)
            continue
        record(f"{split_name}.verified_evidence_available", True)
        control = runs["control"]
        candidate = runs["candidate"]
        try:
            pairs = _paired_rows(
                control.get("sessions"), candidate.get("sessions"), split_name
            )
        except (P11GateError, KeyError, TypeError, ValueError) as exc:
            record(f"{split_name}.paired_sessions_valid", False, str(exc))
            continue
        record(f"{split_name}.paired_sessions_valid", True)

        control_metrics = control.get("metrics")
        candidate_metrics = candidate.get("metrics")
        if not isinstance(control_metrics, Mapping) or not isinstance(
            candidate_metrics, Mapping
        ):
            record(f"{split_name}.quality_metrics_available", False)
            continue

        try:
            split_deltas = {
                field: _metric_delta(candidate_metrics, control_metrics, field)
                for field in _OVERALL_FIELDS
            }
            hit_to_miss = sum(
                int(before["hit"] is True and after["hit"] is False)
                for before, after in pairs
            )
            control_scenarios = control.get("scenario_metrics")
            candidate_scenarios = candidate.get("scenario_metrics")
            if not isinstance(control_scenarios, Mapping) or not isinstance(
                candidate_scenarios, Mapping
            ) or set(control_scenarios) != set(candidate_scenarios):
                raise P11GateError("paired scenario metric registries differ")
            scenario_deltas = {
                name: _decimal(
                    candidate_scenarios[name]["hit_rate_at_10"],
                    f"candidate scenario {name} HR",
                )
                - _decimal(
                    control_scenarios[name]["hit_rate_at_10"],
                    f"control scenario {name} HR",
                )
                for name in sorted(control_scenarios)
            }
        except (P11GateError, KeyError, TypeError, ValueError) as exc:
            record(f"{split_name}.quality_metrics_available", False, str(exc))
            continue
        record(f"{split_name}.quality_metrics_available", True)

        deltas[split_name] = {
            **{field: float(value) for field, value in split_deltas.items()},
            "hit_to_miss_count": hit_to_miss,
            "scenario_hit_rate_at_10": {
                name: float(value) for name, value in scenario_deltas.items()
            },
        }

        score_delta = split_deltas["recommended_technical_score"]
        if split_name in FRESH_SPLITS:
            record(
                f"{split_name}.technical_score_strict_increase", score_delta > 0
            )
        else:
            record(
                f"{split_name}.technical_score_non_decrease", score_delta >= 0
            )
        if split_name == "primary":
            record(
                "primary.technical_score_delta_at_least_0_005",
                score_delta >= PRIMARY_MIN_SCORE_DELTA,
            )
        record(
            f"{split_name}.hit_rate_non_decrease",
            split_deltas["hit_rate_at_10"] >= 0,
        )
        record(f"{split_name}.mrr_non_decrease", split_deltas["mrr"] >= 0)
        record(f"{split_name}.mttc_no_worse", split_deltas["mttc"] <= 0)
        record(f"{split_name}.zero_hit_to_miss", hit_to_miss == 0)
        for scenario, delta in scenario_deltas.items():
            record(
                f"{split_name}.scenario.{scenario}.hit_rate_non_decrease",
                delta >= 0,
            )

        if split_name in FRESH_SPLITS and resample_count_valid and seed_valid:
            try:
                ci = _paired_bootstrap_ci(
                    pairs,
                    seed=_split_seed(bootstrap_seed, split_name),
                    resamples=bootstrap_resamples,
                )
            except Exception as exc:
                record(
                    f"{split_name}.bootstrap_ci_lower_above_zero",
                    False,
                    f"{split_name} bootstrap evidence invalid: {type(exc).__name__}",
                )
            else:
                confidence_intervals[split_name] = ci
                record(
                    f"{split_name}.bootstrap_ci_lower_above_zero", ci["lower"] > 0
                )
        elif split_name in FRESH_SPLITS:
            record(f"{split_name}.bootstrap_ci_lower_above_zero", False)

        served_resources = runs["served"].get("resources")
        control_resources = control.get("resources")
        candidate_resources = candidate.get("resources")
        resource_deltas: dict[str, float | None] = {}
        served_resource_deltas: dict[str, float | None] = {}
        for resource, limit in RESOURCE_LIMITS.items():
            for baseline, baseline_resources, output, suffix in (
                ("control", control_resources, resource_deltas, ""),
                ("served", served_resources, served_resource_deltas, "_vs_served"),
            ):
                try:
                    if not isinstance(baseline_resources, Mapping) or not isinstance(
                        candidate_resources, Mapping
                    ):
                        raise P11GateError("resource observations must be mappings")
                    ratio = _ratio(
                        candidate_resources.get(resource),
                        baseline_resources.get(resource),
                        f"{resource} vs {baseline}",
                    )
                except (P11GateError, KeyError, TypeError, ValueError):
                    output[resource] = None
                    record(
                        f"{split_name}.resource.{resource}{suffix}_within_limit",
                        False,
                    )
                else:
                    output[resource] = round(float(ratio), 12)
                    record(
                        f"{split_name}.resource.{resource}{suffix}_within_limit",
                        ratio <= limit,
                    )
        deltas[split_name]["resource_ratios"] = resource_deltas
        deltas[split_name]["resource_ratios_vs_served"] = served_resource_deltas

    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "reasons": reasons,
        "deltas": deltas,
        "ci": confidence_intervals,
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "MIN_BOOTSTRAP_RESAMPLES",
    "PRIMARY_MIN_SCORE_DELTA",
    "REQUIRED_FLAGS",
    "RESOURCE_LIMITS",
    "SPLIT_NAMES",
    "evaluate_p11_gates",
]
