"""Evaluate a pairwise proposal only as a supplement to the frozen policy.

The current semantic-off nested-OOF policy is the immutable comparator and the
default decision surface.  Pairwise may replace its served slot-10 choice only
through an independently nested, target-blind rescue-versus-regret gate.  This
script reads the frozen train_explore numeric caches; it never starts Agent or
the official evaluator and never opens a held-out split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_small_ranker_metric_gate as metric  # noqa: E402
from scripts import analyze_small_ranker_remaining_misses as attribution  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-supplemental-pairwise-evaluation.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_3.supplemental_pairwise_preregistration.json"
)
IMPLEMENTATION_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_3.supplemental_pairwise_implementation_amendment.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
DEFAULT_PAIRWISE_PROJECTION_RESULT = ROOT / (
    "experiments/fast_track/small_ranker_v2_1/"
    "pairwise_projection_20260830T1530/projection_result.json"
)
EXPECTED_PAIRWISE_SCORE_SHA256 = (
    "1765f60c3f111f751e8d0c133bbbd93d2a1e174db24b2c4c64c80aea66a4778b"
)
EXPECTED_CURRENT_ACTIVATION_SHA256 = (
    "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
)
EXPECTED_CURRENT_CHOSEN_SHA256 = (
    "229952c9ced7f6eec1ff1938480adc85ba5093ad865336465749029576e47051"
)
CURRENT_HR = 0.9715
UTILITY_REGRET_MULTIPLIER = 1.0
SUPPLEMENTAL_FEATURE_NAMES = (
    "current_policy_active",
    "current_choice_rank_fraction_under_pairwise_ranker",
    "pairwise_choice_rank_fraction_under_current_ranker",
    "pairwise_choice_coverage_rank_fraction",
    "pairwise_minus_current_top10_route_agreement",
    "pairwise_minus_current_active_token_recall",
    "pairwise_minus_current_hard_clause_coverage",
    "pairwise_minus_current_constraint_conflict_sum",
)


class SupplementalEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SupplementalSurface:
    current_chosen: np.ndarray
    current_activation: np.ndarray
    current_choice: np.ndarray
    pairwise_chosen: np.ndarray
    action: np.ndarray
    gate_features: np.ndarray
    rescue: np.ndarray
    rescue_weights: np.ndarray
    regret: np.ndarray
    regret_weights: np.ndarray
    rr_loss: np.ndarray
    mttc_loss: np.ndarray


@dataclass(frozen=True)
class CrossFitOutput:
    supplement: np.ndarray
    final_chosen: np.ndarray
    final_activation: np.ndarray
    rescue_probability: np.ndarray
    regret_probability: np.ndarray
    selections: list[dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _serialized_threshold(value: float) -> float | str:
    return "KEEP" if math.isinf(float(value)) else float(value)


def _compose_policy(
    current_chosen: np.ndarray,
    current_activation: np.ndarray,
    pairwise_chosen: np.ndarray,
    supplement: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not (
        current_chosen.shape
        == current_activation.shape
        == pairwise_chosen.shape
        == supplement.shape
    ):
        raise SupplementalEvaluationError("policy arrays have different shapes")
    final_chosen = np.where(
        supplement, pairwise_chosen, current_chosen
    ).astype(np.uint8)
    final_activation = np.asarray(current_activation, dtype=bool) | np.asarray(
        supplement, dtype=bool
    )
    return final_chosen, final_activation


def _allowed_rank_fraction(
    scores: np.ndarray, choice: np.ndarray, incumbent: np.ndarray
) -> np.ndarray:
    """Return stable ascending rank / allowed-count for the slot-10 action set."""

    if scores.ndim != 3 or choice.shape != scores.shape[:2]:
        raise SupplementalEvaluationError("rank-fraction input shape mismatch")
    if incumbent.shape != choice.shape or scores.shape[2] < 11:
        raise SupplementalEvaluationError("rank-fraction incumbent shape mismatch")
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1, scores.shape[2])
    flat_choice = np.asarray(choice, dtype=np.int64).reshape(-1)
    flat_incumbent = np.asarray(incumbent, dtype=np.int64).reshape(-1)
    rows = np.arange(len(flat_choice))
    candidate_indices = np.arange(scores.shape[2], dtype=np.int64)[None, :]
    allowed = np.ones(flat_scores.shape, dtype=bool)
    allowed[:, :10] = False
    allowed[rows, flat_incumbent] = True
    if not np.all(allowed[rows, flat_choice]):
        raise SupplementalEvaluationError("ranked choice is outside slot-10 action set")
    selected = flat_scores[rows, flat_choice]
    better = allowed & (
        (flat_scores > selected[:, None])
        | (
            (flat_scores == selected[:, None])
            & (candidate_indices < flat_choice[:, None])
        )
    )
    rank = 1 + better.sum(axis=1)
    allowed_count = allowed.sum(axis=1)
    if scores.shape[2] == base.CANDIDATE_COUNT and not np.all(
        allowed_count == base.CANDIDATE_COUNT - 9
    ):
        raise SupplementalEvaluationError("slot-10 action set is not 91 candidates")
    return (rank / allowed_count).astype(np.float32).reshape(choice.shape)


def _isolated_labels(
    labels: Mapping[str, np.ndarray],
    current_chosen: np.ndarray,
    current_activation: np.ndarray,
    pairwise_chosen: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build action labels relative to the complete frozen current policy."""

    current_state = metric.policy_session_state(
        labels, current_chosen, current_activation
    )
    current_hit = np.asarray(current_state["hit"], dtype=bool)
    current_rr = np.where(
        current_hit,
        1.0 / np.maximum(np.asarray(current_state["first_rank"]), 1),
        0.0,
    )
    current_turn = np.asarray(current_state["first_turn"], dtype=np.int16)
    rescue = np.zeros_like(action, dtype=np.uint8)
    regret = np.zeros_like(action, dtype=np.uint8)
    rr_loss = np.zeros_like(action, dtype=np.float32)
    mttc_loss = np.zeros_like(action, dtype=np.float32)
    for turn in range(action.shape[1]):
        isolated = np.zeros_like(action, dtype=bool)
        isolated[:, turn] = action[:, turn]
        chosen, activation = _compose_policy(
            current_chosen, current_activation, pairwise_chosen, isolated
        )
        state = metric.policy_session_state(labels, chosen, activation)
        hit = np.asarray(state["hit"], dtype=bool)
        policy_rr = np.where(
            hit,
            1.0 / np.maximum(np.asarray(state["first_rank"]), 1),
            0.0,
        )
        loss_rr = np.maximum(0.0, current_rr - policy_rr)
        loss_turn = np.maximum(
            0, np.asarray(state["first_turn"], dtype=np.int16) - current_turn
        )
        rescue[:, turn] = (
            action[:, turn] & ~current_hit & hit
        ).astype(np.uint8)
        rr_loss[:, turn] = np.where(action[:, turn], loss_rr, 0.0)
        mttc_loss[:, turn] = np.where(action[:, turn], loss_turn, 0.0)
        regret[:, turn] = (
            action[:, turn] & ((loss_rr > 0.0) | (loss_turn > 0))
        ).astype(np.uint8)
    return rescue, regret, rr_loss, mttc_loss


def _session_normalize_weights(
    raw_weights: np.ndarray, action: np.ndarray
) -> np.ndarray:
    if raw_weights.shape != action.shape:
        raise SupplementalEvaluationError("weight/action shape mismatch")
    selected = np.where(action, raw_weights, 0.0).astype(np.float64)
    totals = selected.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(selected)
    np.divide(selected, totals, out=normalized, where=totals > 0.0)
    return normalized


def _surface(
    features: np.ndarray,
    current_scores: np.ndarray,
    pairwise_scores: np.ndarray,
    labels: Mapping[str, np.ndarray],
    current_activation: np.ndarray,
) -> SupplementalSurface:
    incumbent = base._incumbent_indices(features)
    current_chosen, _current_margin, _current_gap = base.choose_slot10(
        current_scores, incumbent
    )
    pairwise_chosen, _pairwise_margin_incumbent, _pairwise_gap = (
        base.choose_slot10(pairwise_scores, incumbent)
    )
    current_choice = np.where(
        current_activation, current_chosen, incumbent
    ).astype(np.uint8)
    action = (pairwise_chosen != current_choice) & (
        pairwise_chosen != incumbent
    )

    flat_features = np.asarray(features).reshape(
        -1, base.CANDIDATE_COUNT, base.FEATURE_COUNT
    )
    flat_pairwise = pairwise_chosen.reshape(-1).astype(np.int64)
    flat_current = current_choice.reshape(-1).astype(np.int64)
    rows = np.arange(len(flat_pairwise))
    pairwise_features = flat_features[rows, flat_pairwise]
    current_features = flat_features[rows, flat_current]
    conflict_columns = [
        base.FEATURE_INDEX[f"{slot}_conflict"]
        for slot in base.CONSTRAINT_SLOTS
    ]
    pairwise_conflict = pairwise_features[:, conflict_columns].sum(axis=1)
    current_conflict = current_features[:, conflict_columns].sum(axis=1)
    gate_features = np.column_stack(
        (
            np.asarray(current_activation, dtype=np.float32).reshape(-1),
            _allowed_rank_fraction(
                pairwise_scores, current_choice, incumbent
            ).reshape(-1),
            _allowed_rank_fraction(
                current_scores, pairwise_chosen, incumbent
            ).reshape(-1),
            pairwise_features[:, base.FEATURE_INDEX["coverage_rank_fraction"]],
            pairwise_features[
                :, base.FEATURE_INDEX["top10_route_agreement_fraction"]
            ]
            - current_features[
                :, base.FEATURE_INDEX["top10_route_agreement_fraction"]
            ],
            pairwise_features[:, base.FEATURE_INDEX["active_token_recall"]]
            - current_features[:, base.FEATURE_INDEX["active_token_recall"]],
            pairwise_features[:, base.FEATURE_INDEX["hard_clause_coverage"]]
            - current_features[:, base.FEATURE_INDEX["hard_clause_coverage"]],
            pairwise_conflict - current_conflict,
        )
    ).astype(np.float32).reshape(
        base.SESSION_COUNT, base.TURN_COUNT, len(SUPPLEMENTAL_FEATURE_NAMES)
    )
    if gate_features.shape != (
        base.SESSION_COUNT,
        base.TURN_COUNT,
        len(SUPPLEMENTAL_FEATURE_NAMES),
    ) or not np.isfinite(gate_features).all():
        raise SupplementalEvaluationError("supplemental feature schema mismatch")

    rescue, regret, rr_loss, mttc_loss = _isolated_labels(
        labels,
        current_chosen,
        current_activation,
        pairwise_chosen,
        action,
    )
    rescue_weights = _session_normalize_weights(
        np.where(
            rescue > 0, 1.0, np.where(regret > 0, 5.0, 0.05)
        ).astype(np.float64),
        action,
    )
    regret_weights = _session_normalize_weights(
        np.where(
            regret > 0,
            5.0 + 20.0 * rr_loss + 0.2 * mttc_loss,
            np.where(rescue > 0, 0.2, 0.05),
        ).astype(np.float64),
        action,
    )
    return SupplementalSurface(
        current_chosen=current_chosen,
        current_activation=np.asarray(current_activation, dtype=bool),
        current_choice=current_choice,
        pairwise_chosen=pairwise_chosen,
        action=action,
        gate_features=gate_features,
        rescue=rescue,
        rescue_weights=rescue_weights,
        regret=regret,
        regret_weights=regret_weights,
        rr_loss=rr_loss,
        mttc_loss=mttc_loss,
    )


def _fit_predict(
    x: np.ndarray,
    train_rows: np.ndarray,
    predict_rows: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> np.ndarray:
    if not np.any(train_rows) or not np.any(predict_rows):
        raise SupplementalEvaluationError("supplemental head partition is empty")
    model, mean, scale = _fit_gate_model(
        x[train_rows], target[train_rows], weights[train_rows], seed
    )
    return base._predict_gate(
        model, mean, scale, x[predict_rows]
    ).astype(np.float32)


def _fit_gate_model(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit the frozen logistic head with session-weighted preprocessing."""

    from sklearn.linear_model import LogisticRegression

    if not len(x) or float(np.sum(weights)) <= 0.0:
        raise SupplementalEvaluationError("supplemental head training set is empty")
    normalized_weight = np.asarray(weights, dtype=np.float64)
    total = float(normalized_weight.sum())
    mean = np.sum(
        np.asarray(x, dtype=np.float64) * normalized_weight[:, None], axis=0
    ) / total
    centered = np.asarray(x, dtype=np.float64) - mean
    scale = np.sqrt(
        np.sum(centered * centered * normalized_weight[:, None], axis=0)
        / total
    )
    scale[scale < 1e-8] = 1.0
    classes = np.unique(y)
    if len(classes) != 2:
        return base._ConstantGate(float(classes[0])), mean, scale
    model = LogisticRegression(
        C=0.2,
        solver="liblinear",
        max_iter=300,
        random_state=base._library_seed(seed),
    )
    model.fit(centered / scale, y, sample_weight=normalized_weight)
    return model, mean, scale


def _compose_supplement(
    surface: SupplementalSurface, supplement: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if np.any(supplement & ~surface.action):
        raise SupplementalEvaluationError("unavailable supplemental action activated")
    return _compose_policy(
        surface.current_chosen,
        surface.current_activation,
        surface.pairwise_chosen,
        supplement,
    )


def _selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["technical_score_delta"]),
        int(row["net_hits"]),
        sum(int(value > 0) for value in row["inner_fold_net_hits"]),
        -int(row["activation_turns"]),
        float(row["quantile"]),
    )


def _select_inner_quantile(
    utility: np.ndarray,
    surface: SupplementalSurface,
    labels: Mapping[str, np.ndarray],
    session_mask: np.ndarray,
    inner_fold: np.ndarray,
) -> dict[str, Any]:
    values = utility[surface.action & session_mask[:, None]]
    if not len(values):
        raise SupplementalEvaluationError("inner supplemental action set is empty")
    current_state = metric.policy_session_state(
        labels, surface.current_chosen, surface.current_activation
    )
    candidates: list[dict[str, Any]] = []
    for quantile in (*frozen.QUANTILES, frozen.KEEP_QUANTILE):
        threshold = frozen._threshold_at_quantile(values, quantile)
        supplement = (
            surface.action
            & session_mask[:, None]
            & (utility >= threshold)
        )
        chosen, activation = _compose_supplement(surface, supplement)
        policy_state = metric.policy_session_state(labels, chosen, activation)
        aggregate = metric.transition_metrics(
            current_state, policy_state, supplement, session_mask
        )
        inner_metrics: list[dict[str, Any]] = []
        for fold in sorted(set(int(value) for value in inner_fold[session_mask])):
            fold_mask = session_mask & (inner_fold == fold)
            inner_metrics.append(
                metric.transition_metrics(
                    current_state, policy_state, supplement, fold_mask
                )
            )
        if all(
            int(row["hit_to_miss"]) == 0
            and float(row["mrr_delta"]) >= 0.0
            and float(row["mttc_delta"]) <= 0.0
            for row in (aggregate, *inner_metrics)
        ):
            candidates.append(
                {
                    "quantile": float(quantile),
                    "inner_threshold": _serialized_threshold(threshold),
                    **aggregate,
                    "inner_fold_net_hits": [
                        int(row["net_hits"]) for row in inner_metrics
                    ],
                    "inner_fold_mrr_delta": [
                        float(row["mrr_delta"]) for row in inner_metrics
                    ],
                    "inner_fold_mttc_delta": [
                        float(row["mttc_delta"]) for row in inner_metrics
                    ],
                }
            )
    if not candidates:
        raise SupplementalEvaluationError("KEEP failed inner safety")
    return max(candidates, key=_selection_key)


def _nested_oof(
    surface: SupplementalSurface,
    labels: Mapping[str, np.ndarray],
    seed: int,
) -> CrossFitOutput:
    outer = np.asarray(labels["outer_fold"])
    inner = np.asarray(labels["inner_fold"])
    flat_x = surface.gate_features.reshape(
        -1, len(SUPPLEMENTAL_FEATURE_NAMES)
    )
    flat_action = surface.action.reshape(-1)
    flat_session = np.repeat(np.arange(len(outer)), base.TURN_COUNT)
    targets = (surface.rescue.reshape(-1), surface.regret.reshape(-1))
    weights = (
        surface.rescue_weights.reshape(-1),
        surface.regret_weights.reshape(-1),
    )
    supplement = np.zeros_like(surface.action, dtype=bool)
    rescue_probability = np.zeros_like(surface.action, dtype=np.float32)
    regret_probability = np.zeros_like(surface.action, dtype=np.float32)
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
            for head in range(2):
                inner_probability[head].reshape(-1)[valid_rows] = _fit_predict(
                    flat_x,
                    train_rows,
                    valid_rows,
                    targets[head],
                    weights[head],
                    seed + head * 10_000 + outer_fold * 31 + inner_index,
                )
        inner_utility = (
            inner_probability[0]
            - UTILITY_REGRET_MULTIPLIER * inner_probability[1]
        )
        selected = _select_inner_quantile(
            inner_utility, surface, labels, train_sessions, inner
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
            model, mean, scale = _fit_gate_model(
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
        train_utility = (
            train_probability[0]
            - UTILITY_REGRET_MULTIPLIER * train_probability[1]
        )
        threshold = frozen._threshold_at_quantile(
            train_utility[surface.action & train_sessions[:, None]],
            float(selected["quantile"]),
        )
        held_utility = (
            held_probability[0]
            - UTILITY_REGRET_MULTIPLIER * held_probability[1]
        )
        supplement[held_sessions] = surface.action[held_sessions] & (
            held_utility[held_sessions] >= threshold
        )
        rescue_probability[held_sessions] = held_probability[0][held_sessions]
        regret_probability[held_sessions] = held_probability[1][held_sessions]
        selections.append(
            {
                "fold": outer_fold,
                "selected_quantile": float(selected["quantile"]),
                "mapped_outer_train_threshold": _serialized_threshold(
                    threshold
                ),
                "inner_selection": selected,
                "train_action_rows": int(train_rows.sum()),
                "train_rescue_rows": int(targets[0][train_rows].sum()),
                "train_regret_rows": int(targets[1][train_rows].sum()),
            }
        )
    final_chosen, final_activation = _compose_supplement(surface, supplement)
    return CrossFitOutput(
        supplement=supplement,
        final_chosen=final_chosen,
        final_activation=final_activation,
        rescue_probability=rescue_probability,
        regret_probability=regret_probability,
        selections=selections,
    )


def _metrics(
    labels: Mapping[str, np.ndarray],
    current_state: Mapping[str, np.ndarray],
    output: CrossFitOutput,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    final_state = metric.policy_session_state(
        labels, output.final_chosen, output.final_activation
    )
    outer = np.asarray(labels["outer_fold"])
    all_sessions = np.ones(len(outer), dtype=bool)
    global_metrics = metric.transition_metrics(
        current_state, final_state, output.supplement, all_sessions
    )
    folds = [
        {
            "fold": fold,
            **metric.transition_metrics(
                current_state,
                final_state,
                output.supplement,
                outer == fold,
            ),
        }
        for fold in range(base.OUTER_FOLDS)
    ]
    return global_metrics, folds, final_state


def _promotion_gate(
    global_metrics: Mapping[str, Any], folds: Sequence[Mapping[str, Any]]
) -> bool:
    return bool(
        float(global_metrics["policy"]["hit_rate_at_10"]) > CURRENT_HR
        and int(global_metrics["hit_to_miss"]) == 0
        and float(global_metrics["mrr_delta"]) >= 0.0
        and float(global_metrics["mttc_delta"]) <= 0.0
        and float(global_metrics["technical_score_delta"]) > 0.0
        and all(
            int(row["net_hits"]) >= 0
            and int(row["hit_to_miss"]) == 0
            and float(row["mrr_delta"]) >= 0.0
            and float(row["mttc_delta"]) <= 0.0
            for row in folds
        )
    )


def _supplemental_oracle(
    surface: SupplementalSurface,
    labels: Mapping[str, np.ndarray],
    current_state: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Report a target-informed ceiling; never return decisions or identities."""

    positive = np.asarray(labels["positive_index"])
    eligible_from = np.asarray(labels["eligible_from"])
    eligible = (
        np.arange(1, base.TURN_COUNT + 1)[None, :]
        >= eligible_from[:, None]
    )
    current_miss = ~np.asarray(current_state["hit"], dtype=bool)
    correct = (
        surface.action
        & eligible
        & (positive >= 0)
        & (surface.pairwise_chosen == positive)
    )
    reachable = current_miss & np.any(correct, axis=1)
    outer = np.asarray(labels["outer_fold"])
    return {
        "current_miss_sessions": int(current_miss.sum()),
        "correct_action_turns_on_current_misses": int(
            correct[current_miss].sum()
        ),
        "reachable_current_miss_sessions": int(reachable.sum()),
        "maximum_zero_harm_hr_at_10": round(
            float((np.asarray(current_state["hit"]) | reachable).mean()), 6
        ),
        "reachable_by_fold": [
            int(np.sum(reachable & (outer == fold)))
            for fold in range(base.OUTER_FOLDS)
        ],
        "posthoc_target_informed_not_runtime": True,
    }


def run(
    source_root: Path,
    projection_root: Path,
    pairwise_projection_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = output_path.resolve()
    if output_path.exists() or ROOT not in output_path.parents:
        raise SupplementalEvaluationError("output must be new and local")
    preregistration = json.loads(
        PREREGISTRATION.read_text(encoding="utf-8")
    )
    amendment = json.loads(
        IMPLEMENTATION_AMENDMENT.read_text(encoding="utf-8")
    )
    if (
        preregistration.get("schema_version")
        != "small-ranker-supplemental-pairwise-preregistration.v1"
        or amendment.get("schema_version")
        != "small-ranker-supplemental-pairwise-implementation-amendment.v1"
        or tuple(
            amendment["runtime_feature_contract"]["feature_names"]
        )
        != SUPPLEMENTAL_FEATURE_NAMES
    ):
        raise SupplementalEvaluationError("preregistration contract mismatch")

    projection_result_path = pairwise_projection_result_path.resolve()
    projection_result = json.loads(
        projection_result_path.read_text(encoding="utf-8")
    )
    if (
        projection_result.get("schema_version")
        != "small-ranker-pairwise-semantic-off-projection.v1"
        or projection_result.get("scores", {}).get("byte_identical") is not True
        or projection_result.get("scope", {}).get("target_label_read") is not False
    ):
        raise SupplementalEvaluationError("pairwise projection contract mismatch")
    pairwise_score_path = ROOT / str(projection_result["scores"]["path"])
    pairwise_repeat_path = ROOT / str(
        projection_result["scores"]["repeat_path"]
    )
    if not pairwise_score_path.is_file() or not pairwise_repeat_path.is_file():
        raise SupplementalEvaluationError("pairwise OOF score file unavailable")
    if (
        _sha256(pairwise_score_path) != EXPECTED_PAIRWISE_SCORE_SHA256
        or _sha256(pairwise_repeat_path) != EXPECTED_PAIRWISE_SCORE_SHA256
    ):
        raise SupplementalEvaluationError("pairwise OOF score identity mismatch")

    inputs = frozen._load_inputs(source_root, projection_root)
    pairwise_scores = np.load(pairwise_score_path, mmap_mode="r")
    if pairwise_scores.shape != inputs.oof_scores.shape:
        raise SupplementalEvaluationError("pairwise OOF score schema mismatch")
    current_surface = frozen._action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    current_activation, current_selections = (
        attribution._reproduce_nested_activation(
            current_surface, inputs.labels, seed=40220260830
        )
    )
    if (
        hashlib.sha256(current_activation.tobytes()).hexdigest()
        != EXPECTED_CURRENT_ACTIVATION_SHA256
        or hashlib.sha256(current_surface.chosen.tobytes()).hexdigest()
        != EXPECTED_CURRENT_CHOSEN_SHA256
    ):
        raise SupplementalEvaluationError("current policy did not reproduce")

    build_started = time.perf_counter()
    surface = _surface(
        inputs.projected_features,
        inputs.oof_scores,
        pairwise_scores,
        inputs.labels,
        current_activation,
    )
    build_seconds = time.perf_counter() - build_started
    current_state = metric.policy_session_state(
        inputs.labels, surface.current_chosen, surface.current_activation
    )
    zero = np.zeros_like(surface.current_activation, dtype=bool)
    p11_state = metric.policy_session_state(
        inputs.labels, surface.current_chosen, zero
    )
    all_sessions = np.ones(base.SESSION_COUNT, dtype=bool)
    current_vs_p11 = metric.transition_metrics(
        p11_state,
        current_state,
        surface.current_activation,
        all_sessions,
    )
    if not (
        float(current_vs_p11["policy"]["hit_rate_at_10"]) == CURRENT_HR
        and int(current_vs_p11["miss_to_hit"]) == 48
        and int(current_vs_p11["hit_to_miss"]) == 0
    ):
        raise SupplementalEvaluationError("current comparator metric drifted")

    first_started = time.perf_counter()
    first = _nested_oof(surface, inputs.labels, seed=40220260830)
    first_seconds = time.perf_counter() - first_started
    repeat_started = time.perf_counter()
    repeat = _nested_oof(surface, inputs.labels, seed=40220260830)
    repeat_seconds = time.perf_counter() - repeat_started
    exact = bool(
        np.array_equal(first.supplement, repeat.supplement)
        and np.array_equal(first.final_chosen, repeat.final_chosen)
        and np.array_equal(first.final_activation, repeat.final_activation)
        and _canonical_sha256(first.selections)
        == _canonical_sha256(repeat.selections)
    )
    if not exact:
        raise SupplementalEvaluationError("nested supplemental repeat differs")

    global_metrics, folds, final_state = _metrics(
        inputs.labels, current_state, first
    )
    passed = _promotion_gate(global_metrics, folds)
    final_vs_p11 = metric.transition_metrics(
        p11_state,
        final_state,
        first.final_activation,
        all_sessions,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.3-SUPPLEMENTAL-PAIRWISE",
        "scope": {
            "split": "train_explore",
            "cached_inputs_only": True,
            "agent_or_evaluator_started": False,
            "held_out_splits_opened": False,
            "external_data_downloaded": False,
            "full_model_or_artifact_trained": False,
            "target_is_training_or_posthoc_label_only": True,
            "runtime_features_target_blind": True,
        },
        "sources": {
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "implementation_amendment_sha256": _sha256(
                IMPLEMENTATION_AMENDMENT
            ),
            "pairwise_projection_result_sha256": _sha256(
                projection_result_path
            ),
            "pairwise_score_sha256": EXPECTED_PAIRWISE_SCORE_SHA256,
            "current_oof_score_sha256": frozen.EXPECTED_HASHES[
                "projected_oof_scores"
            ],
            "feature_cache_sha256": frozen.EXPECTED_HASHES["features"],
            "projected_features_sha256": frozen.EXPECTED_HASHES[
                "projected_features"
            ],
            "label_cache_sha256": frozen.EXPECTED_HASHES["labels"],
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
        },
        "feature_contract": {
            "names": list(SUPPLEMENTAL_FEATURE_NAMES),
            "count": len(SUPPLEMENTAL_FEATURE_NAMES),
            "schema_sha256": _canonical_sha256(
                list(SUPPLEMENTAL_FEATURE_NAMES)
            ),
        },
        "surface": {
            "action_rows": int(surface.action.sum()),
            "action_sessions": int(np.any(surface.action, axis=1).sum()),
            "rescue_label_rows": int(surface.rescue.sum()),
            "regret_label_rows": int(surface.regret.sum()),
            "rr_loss_sum": round(float(surface.rr_loss.sum()), 6),
            "mttc_loss_sum": round(float(surface.mttc_loss.sum()), 6),
        },
        "current": {
            "activation_sha256": hashlib.sha256(
                surface.current_activation.tobytes()
            ).hexdigest(),
            "chosen_sha256": hashlib.sha256(
                surface.current_chosen.tobytes()
            ).hexdigest(),
            "selections_sha256": _canonical_sha256(current_selections),
            "versus_p11": current_vs_p11,
        },
        "challenger": {
            "relative_to_current": global_metrics,
            "folds_relative_to_current": folds,
            "versus_p11": final_vs_p11,
            "outer_selections": first.selections,
            "supplemental_activation_sha256": hashlib.sha256(
                first.supplement.tobytes()
            ).hexdigest(),
            "final_chosen_sha256": hashlib.sha256(
                first.final_chosen.tobytes()
            ).hexdigest(),
            "final_activation_sha256": hashlib.sha256(
                first.final_activation.tobytes()
            ).hexdigest(),
        },
        "repeat": {
            "exact": exact,
            "selection_canonical_sha256": _canonical_sha256(
                first.selections
            ),
        },
        "pairwise_supplemental_oracle": _supplemental_oracle(
            surface, inputs.labels, current_state
        ),
        "decision": {
            "promotion_gate_passed": passed,
            "status": "PROMOTE" if passed else "NO_GO",
            "full_artifact_authorized": passed,
            "next": (
                "freeze and benchmark the two-ranker supplemental artifact"
                if passed
                else "close this fixed supplemental gate and use the recorded oracle bound to select a different mechanism"
            ),
        },
        "timing_seconds": {
            "surface_and_labels": round(build_seconds, 6),
            "first_nested_oof": round(first_seconds, 6),
            "repeat_nested_oof": round(repeat_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT
    )
    parser.add_argument(
        "--pairwise-projection-result",
        type=Path,
        default=DEFAULT_PAIRWISE_PROJECTION_RESULT,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        args.source_root,
        args.projection_root,
        args.pairwise_projection_result,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
