"""Pure aggregation for the P12 validation-only action oracle.

The caller must join targets only after the blind worker has exited.  This
module consumes that joined, in-memory ledger and returns aggregates only; no
session or target identifier is copied into its result.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from scripts.official_metric_bridge import rebuild_official_metrics


MAX_TURNS = 10


class P12OracleMetricsError(ValueError):
    """Raised when a joined oracle ledger is malformed."""


def _finite_number(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise P12OracleMetricsError(f"{label} must be finite")
    return float(value)


def _outcome(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise P12OracleMetricsError(f"{label} must be a mapping")
    turn = value.get("first_hit_turn")
    rank = value.get("first_rank")
    if turn is None and rank is None:
        return {
            "hit": False,
            "first_hit_turn": None,
            "best_rank": None,
            "reciprocal_rank": 0.0,
        }
    if (
        not isinstance(turn, int)
        or isinstance(turn, bool)
        or not 1 <= turn <= MAX_TURNS
        or not isinstance(rank, int)
        or isinstance(rank, bool)
        or not 1 <= rank <= 10
    ):
        raise P12OracleMetricsError(
            f"{label} hit requires first_hit_turn in 1..10 and first_rank in 1..10"
        )
    return {
        "hit": True,
        "first_hit_turn": turn,
        "best_rank": rank,
        "reciprocal_rank": 1.0 / rank,
    }


def _utility(outcome: Mapping[str, object]) -> float:
    if not outcome["hit"]:
        return 0.0
    return (
        0.50
        + 0.30 * float(outcome["reciprocal_rank"])
        + 0.02 * (11 - int(outcome["first_hit_turn"]))
    )


def _weighted_metrics(
    outcomes: Sequence[Mapping[str, object]], weights: Sequence[float]
) -> dict[str, int | float]:
    total = math.fsum(weights)
    if total <= 0.0:
        raise P12OracleMetricsError("metric weights must have a positive sum")
    hit_rate = round(
        math.fsum(weight * int(outcome["hit"]) for outcome, weight in zip(outcomes, weights))
        / total,
        6,
    )
    mrr = round(
        math.fsum(
            weight * float(outcome["reciprocal_rank"])
            for outcome, weight in zip(outcomes, weights)
        )
        / total,
        6,
    )
    mttc = round(
        math.fsum(
            weight
            * (
                int(outcome["first_hit_turn"])
                if outcome["first_hit_turn"] is not None
                else MAX_TURNS + 1
            )
            for outcome, weight in zip(outcomes, weights)
        )
        / total,
        6,
    )
    # Match the official evaluator: aggregate metrics are rounded first, while
    # the unrounded efficiency derived from rounded MTTC enters the score.
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": len(outcomes),
        "weight_sum": round(total, 12),
        "hit_rate_at_10": hit_rate,
        "mrr": mrr,
        "mttc": mttc,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
    }


def _balanced_weights(labels: Sequence[str]) -> list[float]:
    counts = Counter(labels)
    return [1.0 / counts[label] for label in labels]


def _metric_views(
    outcomes: Sequence[Mapping[str, object]], rows: Sequence[Mapping[str, object]]
) -> dict[str, dict[str, int | float]]:
    official = rebuild_official_metrics(outcomes)
    source_weights = [float(row["source_weight"]) for row in rows]
    target_labels = [str(row["target_id"]) for row in rows]
    taxonomy_labels = [str(row["taxonomy"]) for row in rows]
    return {
        "row_uniform_official": official,
        "source_weighted": _weighted_metrics(outcomes, source_weights),
        "target_uniform": _weighted_metrics(
            outcomes, _balanced_weights(target_labels)
        ),
        "taxonomy_balanced": _weighted_metrics(
            outcomes, _balanced_weights(taxonomy_labels)
        ),
    }


def _relative(
    candidate: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int | float]:
    miss_to_hit = sum(
        int(not before["hit"] and after["hit"])
        for before, after in zip(baseline, candidate)
    )
    hit_to_miss = sum(
        int(before["hit"] and not after["hit"])
        for before, after in zip(baseline, candidate)
    )
    count = len(candidate)
    rescue_scenarios: set[str] = set()
    rescue_taxonomies: set[str] = set()
    scenario_net: Counter[str] = Counter()
    taxonomy_net: Counter[str] = Counter()
    for row, before, after in zip(rows, baseline, candidate):
        transition = int(not before["hit"] and after["hit"]) - int(
            before["hit"] and not after["hit"]
        )
        scenario = str(row["scenario"])
        taxonomy = str(row["taxonomy"])
        scenario_net[scenario] += transition
        taxonomy_net[taxonomy] += transition
        if transition == 1:
            rescue_scenarios.add(scenario)
            rescue_taxonomies.add(taxonomy)
    candidate_hit_rate = math.fsum(int(row["hit"]) for row in candidate) / count
    baseline_hit_rate = math.fsum(int(row["hit"]) for row in baseline) / count
    return {
        "hit_count_delta": miss_to_hit - hit_to_miss,
        "hit_rate_delta": round(candidate_hit_rate - baseline_hit_rate, 12),
        "miss_to_hit": miss_to_hit,
        "hit_to_miss": hit_to_miss,
        "net_rescues": miss_to_hit - hit_to_miss,
        "miss_to_hit_rate": round(miss_to_hit / count, 12),
        "hit_to_miss_rate": round(hit_to_miss / count, 12),
        "net_rescue_rate": round((miss_to_hit - hit_to_miss) / count, 12),
        "rescue_scenario_span": len(rescue_scenarios),
        "rescue_taxonomy_span": len(rescue_taxonomies),
        "positive_net_scenario_span": sum(value > 0 for value in scenario_net.values()),
        "positive_net_taxonomy_span": sum(value > 0 for value in taxonomy_net.values()),
    }


def _normalize_seed(seed: int | str) -> int:
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    if isinstance(seed, str) and seed:
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")
    raise P12OracleMetricsError("bootstrap_seed must be an integer or non-empty string")


def _target_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, int | float | str]:
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples <= 0:
        raise P12OracleMetricsError("bootstrap_resamples must be a positive integer")
    clusters: dict[str, list[float]] = defaultdict(list)
    for row, after, before in zip(rows, candidate, baseline):
        clusters[str(row["target_id"])].append(_utility(after) - _utility(before))
    ordered = [clusters[target] for target in sorted(clusters)]
    observed = math.fsum(math.fsum(cluster) for cluster in ordered) / sum(
        len(cluster) for cluster in ordered
    )
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled = [ordered[rng.randrange(len(ordered))] for _ in ordered]
        estimates.append(
            math.fsum(math.fsum(cluster) for cluster in sampled)
            / sum(len(cluster) for cluster in sampled)
        )
    estimates.sort()
    lower_index = math.ceil(0.025 * resamples) - 1
    upper_index = math.ceil(0.975 * resamples) - 1
    return {
        "method": "paired_target_cluster_nearest_rank_percentile",
        "statistic": "row_uniform_mean_utility_delta",
        "confidence": 0.95,
        "seed": seed,
        "resamples": resamples,
        "cluster_count": len(ordered),
        "observed_mean": round(observed, 12),
        "lower": round(estimates[lower_index], 12),
        "upper": round(estimates[upper_index], 12),
    }


def aggregate_action_oracle(
    records: Sequence[Mapping[str, object]],
    *,
    action_ids: Sequence[str],
    oracle_eligible_actions: Sequence[str],
    baseline_action: str = "KEEP_P11",
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int | str = 12012,
) -> dict[str, Any]:
    """Aggregate a post-worker joined ledger without returning identifier rows."""

    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise P12OracleMetricsError("records must be a non-empty sequence")
    rows = list(records)
    if not rows:
        raise P12OracleMetricsError("records must be non-empty")
    if isinstance(action_ids, (str, bytes, bytearray)) or isinstance(
        oracle_eligible_actions, (str, bytes, bytearray)
    ):
        raise P12OracleMetricsError("action lists must be sequences of action IDs")
    actions = list(action_ids)
    eligible = list(oracle_eligible_actions)
    if (
        not actions
        or len(actions) != len(set(actions))
        or any(not isinstance(action, str) or not action for action in actions)
    ):
        raise P12OracleMetricsError("action_ids must be unique non-empty strings")
    if (
        not eligible
        or len(eligible) != len(set(eligible))
        or any(action not in actions for action in eligible)
    ):
        raise P12OracleMetricsError("oracle_eligible_actions must be a non-empty action subset")
    if baseline_action not in actions:
        raise P12OracleMetricsError("baseline_action must be present in action_ids")
    normalized_seed = _normalize_seed(bootstrap_seed)

    ordinals: set[int] = set()
    normalized: dict[str, list[dict[str, object]]] = {action: [] for action in actions}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise P12OracleMetricsError("every record must be a mapping")
        ordinal = row.get("ordinal")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal < 0
            or ordinal in ordinals
        ):
            raise P12OracleMetricsError("record ordinals must be unique non-negative integers")
        ordinals.add(ordinal)
        for field in ("target_id", "scenario", "taxonomy"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise P12OracleMetricsError(f"record {field} must be a non-empty string")
        weight = _finite_number(row.get("source_weight"), "record source_weight")
        if weight < 0.0:
            raise P12OracleMetricsError("record source_weight must be non-negative")
        action_values = row.get("actions")
        if not isinstance(action_values, Mapping) or set(action_values) != set(actions):
            raise P12OracleMetricsError("record actions must exactly match action_ids")
        for action in actions:
            normalized[action].append(
                _outcome(action_values[action], f"record action {action}")
            )

    baseline = normalized[baseline_action]
    action_aggregates: dict[str, Any] = {}
    for action in actions:
        action_aggregates[action] = {
            "metrics": _metric_views(normalized[action], rows),
            "relative_to_baseline": _relative(normalized[action], baseline, rows),
        }

    oracle_outcomes: list[dict[str, object]] = []
    chosen = Counter()
    for index in range(len(rows)):
        best_action = max(
            eligible,
            key=lambda action: (_utility(normalized[action][index]), -eligible.index(action)),
        )
        chosen[best_action] += 1
        oracle_outcomes.append(normalized[best_action][index])

    return {
        "schema_version": "track4.p12-oracle-metrics.v1",
        "session_count": len(rows),
        "target_count": len({str(row["target_id"]) for row in rows}),
        "taxonomy_count": len({str(row["taxonomy"]) for row in rows}),
        "baseline_action": baseline_action,
        "actions": action_aggregates,
        "oracle": {
            "eligible_actions": eligible,
            "selection_counts": {action: chosen[action] for action in eligible},
            "metrics": _metric_views(oracle_outcomes, rows),
            "relative_to_baseline": _relative(oracle_outcomes, baseline, rows),
            "paired_utility_bootstrap_ci": _target_cluster_bootstrap(
                rows,
                oracle_outcomes,
                baseline,
                seed=normalized_seed,
                resamples=bootstrap_resamples,
            ),
        },
    }


__all__ = [
    "P12OracleMetricsError",
    "aggregate_action_oracle",
]
