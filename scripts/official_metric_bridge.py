"""Exact metric bridge for experiments after the frozen P9 protocol.

The official evaluator rounds the aggregate HR, MRR, and MTTC before it
computes efficiency and TechnicalScore.  Future experiment gates must preserve
that ordering; reconstructing the score directly from per-session terms can
differ by one unit at six decimal places.

This module intentionally does not import or modify the P9 runner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from evaluator.local_evaluator import MAX_TURNS, metric_summary


METRIC_FIELDS = (
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
)


class OfficialMetricBridgeError(ValueError):
    """Raised when a session ledger cannot represent official evaluator output."""


def _validate_sessions(
    sessions: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    if isinstance(sessions, (str, bytes, bytearray)) or not isinstance(
        sessions, Sequence
    ):
        raise OfficialMetricBridgeError("sessions must be a non-empty sequence")
    rows = list(sessions)
    if not rows:
        raise OfficialMetricBridgeError("sessions must be non-empty")
    for row in rows:
        if not isinstance(row, Mapping):
            raise OfficialMetricBridgeError("every session must be a mapping")
        hit = row.get("hit")
        best_rank = row.get("best_rank")
        reciprocal_rank = row.get("reciprocal_rank")
        first_hit_turn = row.get("first_hit_turn")
        if not isinstance(hit, bool):
            raise OfficialMetricBridgeError("session hit must be boolean")
        if (
            not isinstance(reciprocal_rank, (int, float))
            or isinstance(reciprocal_rank, bool)
            or not math.isfinite(float(reciprocal_rank))
        ):
            raise OfficialMetricBridgeError("session reciprocal rank must be finite")
        reciprocal_rank = float(reciprocal_rank)
        if hit:
            if (
                not isinstance(first_hit_turn, int)
                or isinstance(first_hit_turn, bool)
                or not 1 <= first_hit_turn <= MAX_TURNS
                or not isinstance(best_rank, int)
                or isinstance(best_rank, bool)
                or not 1 <= best_rank <= 10
                or reciprocal_rank != 1.0 / best_rank
            ):
                raise OfficialMetricBridgeError("hit session fields are inconsistent")
        elif (
            first_hit_turn is not None
            or best_rank is not None
            or reciprocal_rank != 0.0
        ):
            raise OfficialMetricBridgeError("miss session fields are inconsistent")
    return rows


def rebuild_official_metrics(
    sessions: Sequence[Mapping[str, object]],
) -> dict[str, int | float]:
    """Rebuild the official overall metrics with the evaluator's exact ordering."""

    rows = _validate_sessions(sessions)
    overall = metric_summary(rows)
    mttc = overall["mttc"]
    if not isinstance(mttc, (int, float)):
        raise OfficialMetricBridgeError("official MTTC is unavailable")
    efficiency = max(0.0, min(1.0, (11.0 - float(mttc)) / 10.0))
    technical_score = (
        0.50 * float(overall["hit_rate_at_10"])
        + 0.30 * float(overall["mrr"])
        + 0.20 * efficiency
    )
    return {
        "sample_count": int(overall["sample_count"]),
        "hit_rate_at_10": float(overall["hit_rate_at_10"]),
        "mrr": float(overall["mrr"]),
        "mttc": float(mttc),
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
    }


def _valid_observed(field: str, value: object) -> bool:
    if field == "sample_count":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_official_metrics(
    sessions: Sequence[Mapping[str, object]],
    observed: Mapping[str, object],
) -> dict[str, Any]:
    """Fail-closed exact comparison against official aggregate semantics."""

    try:
        expected = rebuild_official_metrics(sessions)
    except (OfficialMetricBridgeError, KeyError, TypeError, ValueError):
        return {
            "passed": False,
            "checks": {"session_ledger_valid": False},
            "expected": None,
            "failure_reasons": ["invalid_session_ledger"],
        }

    fields = ("sample_count", *METRIC_FIELDS)
    checks: dict[str, bool] = {"session_ledger_valid": True}
    reasons: list[str] = []
    if not isinstance(observed, Mapping):
        observed = {}
    for field in fields:
        if field not in observed:
            checks[field] = False
            reasons.append(f"missing_metric:{field}")
            continue
        value = observed[field]
        if not _valid_observed(field, value):
            checks[field] = False
            reasons.append(f"invalid_metric:{field}")
            continue
        checks[field] = value == expected[field]
        if not checks[field]:
            reasons.append(f"official_rounding_mismatch:{field}")
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected": expected,
        "failure_reasons": reasons,
    }


__all__ = [
    "METRIC_FIELDS",
    "OfficialMetricBridgeError",
    "rebuild_official_metrics",
    "validate_official_metrics",
]
