"""Attribute the 57 misses left by the frozen semantic-off nested-OOF policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_small_ranker_metric_gate as metric  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_p12_counterfactual_router as trace_source  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-remaining-miss-attribution.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_0.miss_attribution_preregistration.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
EXPECTED_ACTIVATION_SHA256 = (
    "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
)
EXPECTED_CHOSEN_SHA256 = (
    "229952c9ced7f6eec1ff1938480adc85ba5093ad865336465749029576e47051"
)
ASIN_SHAPE_RE = re.compile(
    rb"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE
)
TAXONOMY_NAMES = {
    0: "accessories-other",
    1: "clothing",
    2: "jewelry",
    3: "shoes",
}
ROUTES = (
    "coverage",
    "p11",
    "broad",
    "strict",
    "fused",
    "structured",
    "semantic",
)
ATTRIBUTE_SLOTS = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "price",
    "feature",
    "use_case",
)
CLASS_ORDER = (
    "A_candidate_absent",
    "C_admission_failure",
    "D_irrecoverable_ambiguous",
    "B_ranker_failure",
)
NONPOSITIONAL_EVIDENCE_NAMES = tuple(
    name
    for name in base.FEATURE_NAMES
    if (
        name.startswith(("turn_", "goal_", "active_"))
        and name != "turn_fraction"
    )
    or any(name.startswith(f"{slot}_") for slot in ATTRIBUTE_SLOTS)
    or name
    in {
        "hard_clause_coverage",
        "explicit_negative_violation",
        "missing_positive_evidence_fraction",
        "current_turn_override",
        "retired_goal_evidence_conflict",
        "price_missing",
        "active_constraint_count_fraction",
        "hard_constraint_count_fraction",
        "negative_constraint_count_fraction",
        "query_specificity_fraction",
        "goal_age_fraction",
        "goal_version_fraction",
        "override_count_fraction",
        "candidate_bayesian_rating_percentile",
        "candidate_popularity_percentile",
        "title_category_length_log",
        "features_details_length_log",
        "description_store_length_log",
    }
)


class MissAttributionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_class(
    *, candidate_present: bool, correct_proposal: bool, ambiguous: bool
) -> str:
    if not candidate_present:
        return "A_candidate_absent"
    if correct_proposal:
        return "C_admission_failure"
    if ambiguous:
        return "D_irrecoverable_ambiguous"
    return "B_ranker_failure"


def _candidate_recall_flags(
    turns: Sequence[Mapping[str, Any]], target: str, eligible_turn: int
) -> dict[int, bool]:
    flags = {depth: False for depth in (10, 20, 50, 100)}
    for turn_index, row in enumerate(turns, 1):
        if turn_index < eligible_turn:
            continue
        pool = row["c100"]
        for depth in flags:
            flags[depth] = flags[depth] or target in pool[:depth]
    return flags


def _reproduce_nested_activation(
    surface: frozen.ActionSurface,
    labels: Mapping[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    outer = np.asarray(labels["outer_fold"])
    inner = np.asarray(labels["inner_fold"])
    flat_x = surface.gate_features.reshape(-1, len(base.GATE_FEATURE_NAMES))
    flat_action = surface.action.reshape(-1)
    flat_session = np.repeat(np.arange(len(outer)), base.TURN_COUNT)
    targets = (surface.rescue.reshape(-1), surface.regret.reshape(-1))
    weights = (
        surface.rescue_weights.reshape(-1),
        surface.regret_weights.reshape(-1),
    )
    activation = np.zeros_like(surface.action, dtype=bool)
    selections: list[dict[str, Any]] = []

    for outer_fold in range(base.OUTER_FOLDS):
        train_sessions = outer != outer_fold
        held_sessions = outer == outer_fold
        inner_probability = [
            np.zeros_like(surface.action, dtype=np.float32) for _ in range(2)
        ]
        for inner_index in range(base.OUTER_FOLDS):
            model_train = train_sessions & (inner != inner_index)
            model_valid = train_sessions & (inner == inner_index)
            train_rows = flat_action & model_train[flat_session]
            valid_rows = flat_action & model_valid[flat_session]
            if not np.any(train_rows) or not np.any(valid_rows):
                raise MissAttributionError("nested admission partition is empty")
            for head in range(2):
                inner_probability[head].reshape(-1)[valid_rows] = (
                    frozen._fit_predict(
                        flat_x,
                        train_rows,
                        valid_rows,
                        targets[head],
                        weights[head],
                        seed
                        + head * 10_000
                        + outer_fold * 31
                        + inner_index,
                    )
                )
        inner_utility = inner_probability[0] - inner_probability[1]
        selected = frozen._select_inner_quantile(
            inner_utility,
            surface,
            labels,
            train_sessions,
            inner,
        )

        train_rows = flat_action & train_sessions[flat_session]
        held_rows = flat_action & held_sessions[flat_session]
        train_probability = [
            np.zeros_like(surface.action, dtype=np.float32) for _ in range(2)
        ]
        held_probability = [
            np.zeros_like(surface.action, dtype=np.float32) for _ in range(2)
        ]
        for head in range(2):
            model, mean, scale = base._fit_gate_model(
                flat_x[train_rows],
                targets[head][train_rows],
                weights[head][train_rows],
                seed + head * 10_000 + outer_fold * 101,
            )
            train_probability[head].reshape(-1)[train_rows] = (
                base._predict_gate(
                    model, mean, scale, flat_x[train_rows]
                ).astype(np.float32)
            )
            held_probability[head].reshape(-1)[held_rows] = (
                base._predict_gate(
                    model, mean, scale, flat_x[held_rows]
                ).astype(np.float32)
            )
        train_utility = train_probability[0] - train_probability[1]
        threshold = frozen._threshold_at_quantile(
            train_utility[surface.action & train_sessions[:, None]],
            float(selected["quantile"]),
        )
        held_utility = held_probability[0] - held_probability[1]
        activation[held_sessions] = surface.action[held_sessions] & (
            held_utility[held_sessions] >= threshold
        )
        selections.append(
            {
                "fold": outer_fold,
                "quantile": float(selected["quantile"]),
                "threshold": threshold,
            }
        )
    return activation, selections


def _load_proxy_rows(source_root: Path) -> list[dict[str, Any]]:
    path = source_root / (
        "experiments/fast_track/proxy_v1/proxy_train_explore.jsonl"
    )
    if not path.is_file() or _sha256(path) != trace_source.PROXY_SHA256:
        raise MissAttributionError("train_explore proxy identity mismatch")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise MissAttributionError("proxy row is invalid")
            rows.append(value)
    if len(rows) != base.SESSION_COUNT or _sha256(path) != trace_source.PROXY_SHA256:
        raise MissAttributionError("proxy row count changed")
    return rows


def _load_traces(source_root: Path) -> tuple[tuple[dict[str, Any], ...], ...]:
    original = trace_source.AGGREGATE
    try:
        trace_source.AGGREGATE = source_root / (
            "experiments/fast_track/action_oracle_v1/"
            "train_explore-full-aggregate.json"
        )
        sessions, _identifiers = trace_source._load_traces()
    finally:
        trace_source.AGGREGATE = original
    return sessions


def _counter_rows(
    class_by_session: Mapping[int, str], values: Sequence[str]
) -> list[dict[str, Any]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for session, class_name in class_by_session.items():
        table[str(values[session])][class_name] += 1
    return [
        {
            "value": value,
            "total": sum(counts.values()),
            "classes": {
                name: int(counts.get(name, 0)) for name in CLASS_ORDER
            },
        }
        for value, counts in sorted(table.items())
    ]


def _strict_ambiguity(
    session: int,
    eligible_index: int,
    positive: np.ndarray,
    incumbent: np.ndarray,
    source_features: np.ndarray,
    scores: np.ndarray,
) -> bool:
    columns = [base.FEATURE_INDEX[name] for name in NONPOSITIONAL_EVIDENCE_NAMES]
    compared_turns = 0
    for turn in range(eligible_index, base.TURN_COUNT):
        target_index = int(positive[session, turn])
        if target_index < 0:
            continue
        compared_turns += 1
        target_evidence = np.asarray(
            source_features[session, turn, target_index, columns]
        )
        target_score = float(scores[session, turn, target_index])
        collision = False
        for candidate in range(base.CANDIDATE_COUNT):
            if (
                candidate == target_index
                or (candidate < 10 and candidate != int(incumbent[session, turn]))
            ):
                continue
            if (
                abs(float(scores[session, turn, candidate]) - target_score)
                <= 1e-7
                and np.array_equal(
                    np.asarray(
                        source_features[session, turn, candidate, columns]
                    ),
                    target_evidence,
                )
            ):
                collision = True
                break
        if not collision:
            return False
    return compared_turns > 0


def run(
    source_root: Path,
    projection_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = output_path.resolve()
    if output_path.exists() or ROOT not in output_path.parents:
        raise MissAttributionError("output must be a new path below the worktree")
    inputs = frozen._load_inputs(source_root, projection_root)
    surface = frozen._action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    activation, selections = _reproduce_nested_activation(
        surface, inputs.labels, seed=40220260830
    )
    activation_sha = hashlib.sha256(activation.tobytes()).hexdigest()
    chosen_sha = hashlib.sha256(surface.chosen.tobytes()).hexdigest()
    if (
        activation_sha != EXPECTED_ACTIVATION_SHA256
        or chosen_sha != EXPECTED_CHOSEN_SHA256
    ):
        raise MissAttributionError("frozen policy decision identity mismatch")

    zero = np.zeros_like(activation, dtype=bool)
    baseline_state = metric.policy_session_state(
        inputs.labels, surface.chosen, zero
    )
    policy_state = metric.policy_session_state(
        inputs.labels, surface.chosen, activation
    )
    policy_hit = np.asarray(policy_state["hit"], dtype=bool)
    remaining = np.flatnonzero(~policy_hit)
    if int(policy_hit.sum()) != 1943 or len(remaining) != 57:
        raise MissAttributionError("remaining-miss count is not frozen 1943/57")

    traces = _load_traces(source_root.resolve())
    proxy_rows = _load_proxy_rows(source_root.resolve())
    positive = np.asarray(inputs.labels["positive_index"])
    eligible_from = np.asarray(inputs.labels["eligible_from"])
    targets = [str(row["ground_truth"]["parent_asin"]) for row in proxy_rows]
    for session in range(base.SESSION_COUNT):
        expected_eligible = trace_source._eligible_turn(proxy_rows[session])
        if int(eligible_from[session]) != expected_eligible:
            raise MissAttributionError("eligible-turn binding mismatch")
        for turn in range(base.TURN_COUNT):
            try:
                expected_positive = traces[session][turn]["c100"].index(
                    targets[session]
                )
            except ValueError:
                expected_positive = -1
            if int(positive[session, turn]) != expected_positive:
                raise MissAttributionError("trace/label candidate binding mismatch")

    recall_counts = Counter({10: 0, 20: 0, 50: 0, 100: 0})
    recall_by_session: list[dict[int, bool]] = []
    for session in range(base.SESSION_COUNT):
        flags = _candidate_recall_flags(
            traces[session], targets[session], int(eligible_from[session])
        )
        recall_by_session.append(flags)
        for depth, present in flags.items():
            recall_counts[depth] += int(present)

    class_by_session: dict[int, str] = {}
    earliest_reachable: dict[int, int | None] = {}
    earliest_correct: dict[int, int | None] = {}
    candidate_depth_bucket: dict[int, str] = {}
    source_support: dict[str, Counter[str]] = {
        class_name: Counter() for class_name in CLASS_ORDER
    }
    attribute_unknown: dict[str, Counter[str]] = {
        class_name: Counter() for class_name in CLASS_ORDER
    }
    attribute_conflict: dict[str, Counter[str]] = {
        class_name: Counter() for class_name in CLASS_ORDER
    }

    for session in remaining:
        session = int(session)
        eligible_index = int(eligible_from[session]) - 1
        present_turns = [
            turn
            for turn in range(eligible_index, base.TURN_COUNT)
            if int(positive[session, turn]) >= 0
        ]
        correct_turns = [
            turn
            for turn in present_turns
            if int(surface.chosen[session, turn])
            == int(positive[session, turn])
        ]
        if any(activation[session, turn] for turn in correct_turns):
            raise MissAttributionError("remaining miss has an activated rescue")
        ambiguous = bool(
            present_turns
            and not correct_turns
            and _strict_ambiguity(
                session,
                eligible_index,
                positive,
                surface.incumbent,
                inputs.source_features,
                inputs.oof_scores,
            )
        )
        class_name = _exclusive_class(
            candidate_present=bool(present_turns),
            correct_proposal=bool(correct_turns),
            ambiguous=ambiguous,
        )
        class_by_session[session] = class_name
        earliest_reachable[session] = (
            min(present_turns) + 1 if present_turns else None
        )
        earliest_correct[session] = (
            min(correct_turns) + 1 if correct_turns else None
        )
        flags = recall_by_session[session]
        if flags[10]:
            candidate_depth_bucket[session] = "C10"
        elif flags[20]:
            candidate_depth_bucket[session] = "C11-20"
        elif flags[50]:
            candidate_depth_bucket[session] = "C21-50"
        elif flags[100]:
            candidate_depth_bucket[session] = "C51-100"
        else:
            candidate_depth_bucket[session] = "absent-C100"

        for route in ROUTES:
            column = base.FEATURE_INDEX[f"{route}_presence"]
            supported = any(
                float(
                    inputs.source_features[
                        session, turn, int(positive[session, turn]), column
                    ]
                )
                > 0.5
                for turn in present_turns
            )
            source_support[class_name][route] += int(supported)
        if present_turns:
            turn = min(present_turns)
            target_index = int(positive[session, turn])
            for slot in ATTRIBUTE_SLOTS:
                attribute_unknown[class_name][slot] += int(
                    float(
                        inputs.source_features[
                            session,
                            turn,
                            target_index,
                            base.FEATURE_INDEX[f"{slot}_unknown"],
                        ]
                    )
                    > 0.5
                )
                attribute_conflict[class_name][slot] += int(
                    float(
                        inputs.source_features[
                            session,
                            turn,
                            target_index,
                            base.FEATURE_INDEX[f"{slot}_conflict"],
                        ]
                    )
                    > 0.5
                )

    class_counts = Counter(class_by_session.values())
    if sum(class_counts.values()) != 57:
        raise MissAttributionError("exclusive class assignment is incomplete")
    current_hits = int(policy_hit.sum())
    actionable = int(
        class_counts["B_ranker_failure"]
        + class_counts["C_admission_failure"]
    )
    c100_reachable = int(recall_counts[100])
    required_for_099 = max(0, 1980 - current_hits)

    outer = np.asarray(inputs.labels["outer_fold"])
    taxonomy = np.asarray(inputs.labels["taxonomy_code"])
    scenario_values = [str(row.get("scenario_type", "missing")) for row in proxy_rows]
    difficulty_values = [
        str(row.get("difficulty_bucket", "missing")) for row in proxy_rows
    ]
    fold_values = [str(int(value)) for value in outer]
    taxonomy_values = [
        TAXONOMY_NAMES.get(int(value), f"unknown-{int(value)}")
        for value in taxonomy
    ]
    strata = Counter(
        (
            class_by_session[session],
            taxonomy_values[session],
            scenario_values[session],
            difficulty_values[session],
            candidate_depth_bucket[session],
        )
        for session in class_by_session
    )
    typical = [
        {
            "class": key[0],
            "taxonomy": key[1],
            "scenario_type": key[2],
            "difficulty": key[3],
            "candidate_depth": key[4],
            "sessions": count,
        }
        for key, count in sorted(
            strata.items(), key=lambda item: (-item[1], item[0])
        )[:20]
    ]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.0-MISS-ATTRIBUTION",
        "scope": {
            "split": "train_explore",
            "posthoc_target_access": True,
            "target_or_identity_runtime_features": False,
            "individual_identifiers_serialized": False,
            "held_out_splits_opened": False,
            "agent_or_evaluator_started": False,
        },
        "sources": {
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "feature_cache_sha256": frozen.EXPECTED_HASHES["features"],
            "label_cache_sha256": frozen.EXPECTED_HASHES["labels"],
            "projected_feature_sha256": frozen.EXPECTED_HASHES[
                "projected_features"
            ],
            "projected_oof_score_sha256": frozen.EXPECTED_HASHES[
                "projected_oof_scores"
            ],
            "proxy_sha256": trace_source.PROXY_SHA256,
            "combined_trace_sha256": trace_source.COMBINED_TRACE_SHA256,
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
        },
        "policy_reproduction": {
            "baseline_hits": int(np.asarray(baseline_state["hit"]).sum()),
            "candidate_hits": current_hits,
            "remaining_misses": len(remaining),
            "activation_sha256": activation_sha,
            "chosen_sha256": chosen_sha,
            "fold_quantiles": [row["quantile"] for row in selections],
        },
        "classes": {
            name: {
                "sessions": int(class_counts.get(name, 0)),
                "fraction_of_57": round(
                    int(class_counts.get(name, 0)) / 57.0, 6
                ),
            }
            for name in CLASS_ORDER
        },
        "candidate_recall": {
            f"C{depth}": {
                "sessions": int(recall_counts[depth]),
                "recall": round(recall_counts[depth] / 2000.0, 6),
            }
            for depth in (10, 20, 50, 100)
        },
        "candidate_recall_C200": {
            "available": False,
            "reason": "the frozen blind trace records exact C20/C50/C100 pools only; C200 is not inferred",
        },
        "oracle_bounds": {
            "current_hr_at_10": round(current_hits / 2000.0, 6),
            "rescues_needed_for_0_99": required_for_099,
            "mechanical_c100_oracle_hits": c100_reachable,
            "mechanical_c100_oracle_hr_at_10": round(
                c100_reachable / 2000.0, 6
            ),
            "remaining_c100_reachable_misses": int(
                57 - class_counts["A_candidate_absent"]
            ),
            "strict_ambiguity_lower_bound_sessions": int(
                class_counts["D_irrecoverable_ambiguous"]
            ),
            "conservative_distinguishable_repair_hits": current_hits
            + actionable,
            "conservative_distinguishable_repair_hr_at_10": round(
                (current_hits + actionable) / 2000.0, 6
            ),
            "c100_oracle_reaches_0_99": c100_reachable >= 1980,
            "distinguishable_upper_reaches_0_99": current_hits
            + actionable
            >= 1980,
        },
        "breakdowns": {
            "outer_fold": _counter_rows(class_by_session, fold_values),
            "taxonomy": _counter_rows(class_by_session, taxonomy_values),
            "scenario_type": _counter_rows(
                class_by_session, scenario_values
            ),
            "difficulty": _counter_rows(
                class_by_session, difficulty_values
            ),
            "candidate_depth": _counter_rows(
                class_by_session,
                [candidate_depth_bucket.get(index, "not-a-miss") for index in range(2000)],
            ),
            "earliest_reachable_turn": _counter_rows(
                class_by_session,
                [
                    str(earliest_reachable.get(index, "not-a-miss"))
                    for index in range(2000)
                ],
            ),
            "earliest_correct_proposal_turn": _counter_rows(
                class_by_session,
                [
                    str(earliest_correct.get(index, "not-a-miss"))
                    for index in range(2000)
                ],
            ),
        },
        "candidate_source_support": {
            class_name: {
                route: int(source_support[class_name][route])
                for route in ROUTES
            }
            for class_name in CLASS_ORDER
        },
        "first_reachable_target_attribute_evidence": {
            class_name: {
                "unknown": {
                    slot: int(attribute_unknown[class_name][slot])
                    for slot in ATTRIBUTE_SLOTS
                },
                "conflict": {
                    slot: int(attribute_conflict[class_name][slot])
                    for slot in ATTRIBUTE_SLOTS
                },
            }
            for class_name in CLASS_ORDER
        },
        "typical_aggregate_error_strata": typical,
        "decision": {
            "largest_class": max(
                CLASS_ORDER, key=lambda name: int(class_counts.get(name, 0))
            ),
            "oracle_bound_stop": bool(
                c100_reachable < 1980
                or current_hits + actionable < 1980
            ),
            "next": "preregister one mechanism for the largest actionable class",
        },
        "timing_seconds": {
            "total": round(time.perf_counter() - started, 6)
        },
    }
    raw = (
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if ASIN_SHAPE_RE.search(raw):
        raise MissAttributionError("aggregate result contains an identity token")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(raw)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.source_root, args.projection_root, args.output)
    print(
        json.dumps(
            {
                "classes": result["classes"],
                "candidate_recall": result["candidate_recall"],
                "oracle_bounds": result["oracle_bounds"],
                "decision": result["decision"],
                "timing_seconds": result["timing_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
