"""Python-3.9-compatible runtime subset of the frozen v2.7 selector.

The implementation preserves valid-input numeric and decision parity with the
helpers in ``evaluate_small_ranker_portfolio_selector.py`` and its supplemental
helper.  Report/evaluator imports are omitted and ``int.bit_count`` is expressed
as ``bin(value).count('1')`` for the pinned Python 3.9 training environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from scripts import analyze_small_ranker_metric_gate as metric
from scripts import export_small_ranker_fold_safe_artifact as frozen
from scripts import train_small_ranker as base


MODEL_SEED = 20260830
MAX_ACTIONS = 3
MIN_RESCUE_SESSIONS = 5
MIN_RESCUE_FAMILIES = 5
REGRET_MULTIPLIER = 1.0
FAMILY_NAMES = ("pairwise", "rrf3", "focused_lambdamart")
FEATURE_NAMES = (
    "current_policy_active",
    "turn_fraction",
    "prior_current_activation_count_fraction",
    "actionable_unique_count_fraction",
    "pairwise_support",
    "rrf3_support",
    "focused_support",
    "action_rank_fraction_under_current",
    "action_rank_fraction_under_pairwise",
    "action_rank_fraction_under_rrf3",
    "action_rank_fraction_under_focused",
    "current_choice_rank_fraction_under_pairwise",
    "current_choice_rank_fraction_under_rrf3",
    "current_choice_rank_fraction_under_focused",
    "action_coverage_rank_fraction",
    "action_minus_current_top10_route_agreement",
    "action_minus_current_active_token_recall",
    "action_minus_current_hard_clause_coverage",
    "action_minus_current_constraint_conflict_sum",
)
BIT_COUNT = np.asarray(
    [bin(value).count("1") for value in range(256)], dtype=np.uint8
)


class PortfolioSelectorError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimePortfolioSurface:
    current_chosen: np.ndarray
    current_activation: np.ndarray
    current_choice: np.ndarray
    incumbent: np.ndarray
    family_choices: np.ndarray
    candidates: np.ndarray
    source_mask: np.ndarray
    available: np.ndarray
    features: np.ndarray


@dataclass(frozen=True)
class PortfolioSurface(RuntimePortfolioSurface):
    rescue: np.ndarray
    rescue_weights: np.ndarray
    regret: np.ndarray
    regret_weights: np.ndarray
    rr_loss: np.ndarray
    mttc_loss: np.ndarray


def _serialized_threshold(value: float) -> Union[float, str]:
    return "KEEP" if math.isinf(float(value)) else float(value)


def _deduplicate_actions(
    family_choices: np.ndarray,
    current_choice: np.ndarray,
    incumbent: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    choices = np.asarray(family_choices)
    if (
        choices.ndim != 3
        or choices.shape[2] != len(FAMILY_NAMES)
        or current_choice.shape != choices.shape[:2]
        or incumbent.shape != choices.shape[:2]
    ):
        raise PortfolioSelectorError("proposal action shape mismatch")
    if np.any((choices < 0) | (choices >= base.CANDIDATE_COUNT)):
        raise PortfolioSelectorError("proposal candidate is out of range")
    candidates = np.full((*choices.shape[:2], MAX_ACTIONS), -1, dtype=np.int16)
    support = np.zeros(candidates.shape, dtype=np.uint8)
    available = np.zeros(candidates.shape, dtype=bool)
    for session in range(choices.shape[0]):
        for turn in range(choices.shape[1]):
            merged = {}
            for family in range(choices.shape[2]):
                candidate = int(choices[session, turn, family])
                if candidate in {
                    int(current_choice[session, turn]),
                    int(incumbent[session, turn]),
                }:
                    continue
                merged[candidate] = merged.get(candidate, 0) | (1 << family)
            if len(merged) > MAX_ACTIONS:
                raise PortfolioSelectorError("too many unique proposal actions")
            for slot, candidate in enumerate(sorted(merged)):
                candidates[session, turn, slot] = candidate
                support[session, turn, slot] = merged[candidate]
                available[session, turn, slot] = True
    return candidates, support, available


def _within_turn_winner(
    candidates: np.ndarray,
    source_mask: np.ndarray,
    available: np.ndarray,
    utility: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (
        candidates.shape
        == source_mask.shape
        == available.shape
        == utility.shape
        and candidates.ndim >= 1
    ):
        raise PortfolioSelectorError("within-turn input shape mismatch")
    if np.asarray(available).dtype != np.bool_:
        raise PortfolioSelectorError("within-turn availability must be boolean")
    if np.any(
        available
        & (
            (candidates < 0)
            | (candidates >= base.CANDIDATE_COUNT)
            | (source_mask < 1)
            | (source_mask > 0b111)
        )
    ):
        raise PortfolioSelectorError("available action metadata is invalid")
    if not np.isfinite(np.asarray(utility)[available]).all():
        raise PortfolioSelectorError("available utility is non-finite")
    leading = candidates.shape[:-1]
    best_slot = np.full(leading, -1, dtype=np.int8)
    best_candidate = np.full(leading, -1, dtype=np.int16)
    best_utility = np.full(leading, -np.inf, dtype=np.float32)
    best_support = np.full(leading, -1, dtype=np.int8)
    for slot in range(candidates.shape[-1]):
        valid = np.asarray(available[..., slot], dtype=bool)
        candidate = np.asarray(candidates[..., slot], dtype=np.int16)
        value = np.asarray(utility[..., slot], dtype=np.float32)
        support_count = BIT_COUNT[np.asarray(source_mask[..., slot], dtype=np.uint8)]
        better = valid & (
            (best_slot < 0)
            | (value > best_utility)
            | (
                (value == best_utility)
                & (
                    (support_count > best_support)
                    | (
                        (support_count == best_support)
                        & (candidate < best_candidate)
                    )
                )
            )
        )
        best_slot = np.where(better, slot, best_slot).astype(np.int8)
        best_candidate = np.where(better, candidate, best_candidate).astype(
            np.int16
        )
        best_utility = np.where(better, value, best_utility).astype(np.float32)
        best_support = np.where(better, support_count, best_support).astype(
            np.int8
        )
    return best_slot, best_candidate, best_utility, best_slot >= 0


def _per_turn_winner_utilities(
    candidates: np.ndarray,
    source_mask: np.ndarray,
    available: np.ndarray,
    utility: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _within_turn_winner(candidates, source_mask, available, utility)


def _causal_latch(
    winner_candidate: np.ndarray,
    winner_utility: np.ndarray,
    winner_available: np.ndarray,
    threshold: float,
    session_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (
        winner_candidate.shape
        == winner_utility.shape
        == winner_available.shape
        and winner_candidate.ndim == 2
    ):
        raise PortfolioSelectorError("causal latch input shape mismatch")
    if np.asarray(winner_available).dtype != np.bool_:
        raise PortfolioSelectorError("winner availability must be boolean")
    if math.isnan(float(threshold)) or float(threshold) == -math.inf:
        raise PortfolioSelectorError("causal threshold is invalid")
    if not np.isfinite(np.asarray(winner_utility)[winner_available]).all():
        raise PortfolioSelectorError("winner utility is non-finite")
    if np.any(
        winner_available
        & ((winner_candidate < 0) | (winner_candidate >= base.CANDIDATE_COUNT))
    ):
        raise PortfolioSelectorError("winner candidate is out of range")
    sessions, turns = winner_candidate.shape
    selected_sessions = (
        np.ones(sessions, dtype=bool)
        if session_mask is None
        else np.asarray(session_mask, dtype=bool)
    )
    if selected_sessions.shape != (sessions,):
        raise PortfolioSelectorError("causal latch session mask mismatch")
    supplement = np.zeros((sessions, turns), dtype=bool)
    supplemental_choice = np.full((sessions, turns), -1, dtype=np.int16)
    used = np.zeros(sessions, dtype=bool)
    for turn in range(turns):
        activate = (
            selected_sessions
            & ~used
            & winner_available[:, turn]
            & (winner_utility[:, turn] >= threshold)
        )
        supplement[:, turn] = activate
        supplemental_choice[activate, turn] = winner_candidate[activate, turn]
        used |= activate
    if np.any(supplement.sum(axis=1) > 1):
        raise PortfolioSelectorError("causal latch selected multiple actions")
    return supplement, supplemental_choice


def _map_outer_quantile(
    winner_utility: np.ndarray,
    winner_available: np.ndarray,
    train_sessions: np.ndarray,
    quantile: float,
) -> float:
    if winner_utility.shape != winner_available.shape or winner_utility.ndim != 2:
        raise PortfolioSelectorError("quantile winner shape mismatch")
    if (
        np.asarray(winner_available).dtype != np.bool_
        or np.asarray(train_sessions).dtype != np.bool_
    ):
        raise PortfolioSelectorError("quantile masks must be boolean")
    train = np.asarray(train_sessions, dtype=bool)
    if train.shape != (winner_utility.shape[0],):
        raise PortfolioSelectorError("quantile session mask mismatch")
    values = winner_utility[train[:, None] & winner_available]
    return frozen._threshold_at_quantile(values, float(quantile))


def _causal_policy(
    candidates: np.ndarray,
    source_mask: np.ndarray,
    available: np.ndarray,
    utility: np.ndarray,
    threshold: float,
    session_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    _slot, candidate, winner_utility, winner_available = (
        _per_turn_winner_utilities(
            candidates, source_mask, available, utility
        )
    )
    return _causal_latch(
        candidate,
        winner_utility,
        winner_available,
        threshold,
        session_mask,
    )


def _compose_policy(
    current_chosen: np.ndarray,
    current_activation: np.ndarray,
    candidates: np.ndarray,
    available: np.ndarray,
    supplement: np.ndarray,
    supplemental_choice: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if (
        supplement.shape != current_chosen.shape
        or current_activation.shape != current_chosen.shape
        or supplemental_choice.shape != supplement.shape
        or candidates.shape != available.shape
        or candidates.shape[:2] != supplement.shape
        or np.any(supplement.sum(axis=1) > 1)
        or np.any(supplement & (supplemental_choice < 0))
    ):
        raise PortfolioSelectorError("supplemental policy shape mismatch")
    member = np.any(
        available & (candidates == supplemental_choice[..., None]), axis=2
    )
    if np.any(supplement & ~member):
        raise PortfolioSelectorError("supplemental choice is not an available action")
    final_chosen = np.asarray(current_chosen, dtype=np.uint8).copy()
    final_chosen[supplement] = supplemental_choice[supplement].astype(np.uint8)
    final_activation = np.asarray(current_activation, dtype=bool) | supplement
    return final_chosen, final_activation


def _allowed_rank_fraction(
    scores: np.ndarray, choice: np.ndarray, incumbent: np.ndarray
) -> np.ndarray:
    if scores.ndim != 3 or choice.shape != scores.shape[:2]:
        raise PortfolioSelectorError("rank-fraction input shape mismatch")
    if incumbent.shape != choice.shape or scores.shape[2] < 11:
        raise PortfolioSelectorError("rank-fraction incumbent shape mismatch")
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1, scores.shape[2])
    flat_choice = np.asarray(choice, dtype=np.int64).reshape(-1)
    flat_incumbent = np.asarray(incumbent, dtype=np.int64).reshape(-1)
    rows = np.arange(len(flat_choice))
    candidate_indices = np.arange(scores.shape[2], dtype=np.int64)[None, :]
    allowed = np.ones(flat_scores.shape, dtype=bool)
    allowed[:, :10] = False
    allowed[rows, flat_incumbent] = True
    if not np.all(allowed[rows, flat_choice]):
        raise PortfolioSelectorError("ranked choice is outside slot-10 action set")
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
        raise PortfolioSelectorError("slot-10 action set is not 91 candidates")
    return (rank / allowed_count).astype(np.float32).reshape(choice.shape)


def _rank_fraction_for_actions(
    scores: np.ndarray,
    candidates: np.ndarray,
    available: np.ndarray,
    incumbent: np.ndarray,
) -> np.ndarray:
    result = np.zeros(candidates.shape, dtype=np.float32)
    for slot in range(candidates.shape[2]):
        safe_choice = np.where(
            available[..., slot], candidates[..., slot], incumbent
        ).astype(np.uint8)
        rank = _allowed_rank_fraction(scores, safe_choice, incumbent)
        result[..., slot] = np.where(available[..., slot], rank, 0.0)
    return result


def _compose_isolated_policy(
    current_chosen: np.ndarray,
    current_activation: np.ndarray,
    proposal_chosen: np.ndarray,
    supplement: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (
        current_chosen.shape
        == current_activation.shape
        == proposal_chosen.shape
        == supplement.shape
    ):
        raise PortfolioSelectorError("policy arrays have different shapes")
    final_chosen = np.where(
        supplement, proposal_chosen, current_chosen
    ).astype(np.uint8)
    final_activation = np.asarray(current_activation, dtype=bool) | np.asarray(
        supplement, dtype=bool
    )
    return final_chosen, final_activation


def _isolated_action_labels(
    labels: Mapping[str, np.ndarray],
    current_chosen: np.ndarray,
    current_activation: np.ndarray,
    proposal_chosen: np.ndarray,
    action: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        chosen, activation = _compose_isolated_policy(
            current_chosen, current_activation, proposal_chosen, isolated
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
        rescue[:, turn] = (action[:, turn] & ~current_hit & hit).astype(
            np.uint8
        )
        rr_loss[:, turn] = np.where(action[:, turn], loss_rr, 0.0)
        mttc_loss[:, turn] = np.where(action[:, turn], loss_turn, 0.0)
        regret[:, turn] = (
            action[:, turn] & ((loss_rr > 0.0) | (loss_turn > 0))
        ).astype(np.uint8)
    return rescue, regret, rr_loss, mttc_loss


def _session_normalize_weights(
    raw_weights: np.ndarray, available: np.ndarray
) -> np.ndarray:
    if raw_weights.shape != available.shape or raw_weights.ndim != 3:
        raise PortfolioSelectorError("portfolio weight shape mismatch")
    if (
        not np.isfinite(raw_weights).all()
        or np.any(np.asarray(raw_weights) < 0.0)
        or np.asarray(available).dtype != np.bool_
    ):
        raise PortfolioSelectorError("portfolio weights are invalid")
    selected = np.where(available, raw_weights, 0.0).astype(np.float64)
    totals = selected.sum(axis=(1, 2), keepdims=True)
    normalized = np.zeros_like(selected)
    np.divide(selected, totals, out=normalized, where=totals > 0.0)
    active_sessions = np.any(available, axis=(1, 2))
    if not np.allclose(
        normalized.sum(axis=(1, 2))[active_sessions], 1.0, atol=1e-12
    ):
        raise PortfolioSelectorError("portfolio session weights do not sum to one")
    if np.any(normalized[~active_sessions]):
        raise PortfolioSelectorError("session without actions has nonzero weight")
    return normalized


def _attach_isolated_labels(
    runtime: RuntimePortfolioSurface, labels: Mapping[str, np.ndarray]
) -> PortfolioSurface:
    available = runtime.available
    candidates = runtime.candidates
    rescue = np.zeros(available.shape, dtype=np.uint8)
    regret = np.zeros(available.shape, dtype=np.uint8)
    rr_loss = np.zeros(available.shape, dtype=np.float32)
    mttc_loss = np.zeros(available.shape, dtype=np.float32)
    for slot in range(MAX_ACTIONS):
        safe_candidate = np.where(
            available[..., slot], candidates[..., slot], runtime.incumbent
        ).astype(np.uint8)
        slot_labels = _isolated_action_labels(
            labels,
            runtime.current_chosen,
            runtime.current_activation,
            safe_candidate,
            available[..., slot],
        )
        (
            rescue[..., slot],
            regret[..., slot],
            rr_loss[..., slot],
            mttc_loss[..., slot],
        ) = slot_labels
    rescue_weights = _session_normalize_weights(
        np.where(rescue > 0, 1.0, np.where(regret > 0, 5.0, 0.05)),
        available,
    )
    regret_weights = _session_normalize_weights(
        np.where(
            regret > 0,
            5.0 + 20.0 * rr_loss + 0.2 * mttc_loss,
            np.where(rescue > 0, 0.2, 0.05),
        ),
        available,
    )
    return PortfolioSurface(
        current_chosen=runtime.current_chosen,
        current_activation=runtime.current_activation,
        current_choice=runtime.current_choice,
        incumbent=runtime.incumbent,
        family_choices=runtime.family_choices,
        candidates=runtime.candidates,
        source_mask=runtime.source_mask,
        available=runtime.available,
        features=runtime.features,
        rescue=rescue,
        rescue_weights=rescue_weights,
        regret=regret,
        regret_weights=regret_weights,
        rr_loss=rr_loss,
        mttc_loss=mttc_loss,
    )


def _fit_gate_model(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int
) -> Tuple[Any, np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    if not len(x) or float(np.sum(weights)) <= 0.0:
        raise PortfolioSelectorError("selector head training set is empty")
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


def _validate_fitted_model(model: Any) -> None:
    if not hasattr(model, "get_params") or not hasattr(model, "n_iter_"):
        raise PortfolioSelectorError("selector head unexpectedly became constant")
    params = model.get_params()
    if not (
        float(params.get("C")) == 0.2
        and params.get("penalty") == "l2"
        and params.get("solver") == "liblinear"
        and params.get("fit_intercept") is True
        and int(params.get("max_iter")) == 300
        and int(params.get("random_state")) == MODEL_SEED
    ):
        raise PortfolioSelectorError("selector model parameter drifted")
    if int(np.max(model.n_iter_)) >= 300:
        raise PortfolioSelectorError("selector model did not converge")


def _fit_readiness(
    rows: np.ndarray,
    rescue: np.ndarray,
    regret: np.ndarray,
    row_session: np.ndarray,
    row_family: np.ndarray,
) -> dict:
    selected = np.asarray(rows, dtype=bool)
    rescue_values = np.asarray(rescue)[selected]
    regret_values = np.asarray(regret)[selected]
    rescue_sessions = np.unique(
        np.asarray(row_session)[selected & (np.asarray(rescue) > 0)]
    )
    rescue_families = np.unique(
        np.asarray(row_family)[selected & (np.asarray(rescue) > 0)]
    )
    return {
        "action_rows": int(selected.sum()),
        "rescue_rows": int(rescue_values.sum()),
        "regret_rows": int(regret_values.sum()),
        "distinct_rescue_sessions": int(len(rescue_sessions)),
        "distinct_rescue_families": int(len(rescue_families)),
        "rescue_two_class": bool(len(np.unique(rescue_values)) == 2),
        "regret_two_class": bool(len(np.unique(regret_values)) == 2),
        "ready": bool(
            selected.any()
            and len(rescue_sessions) >= MIN_RESCUE_SESSIONS
            and len(rescue_families) >= MIN_RESCUE_FAMILIES
            and len(np.unique(rescue_values)) == 2
            and len(np.unique(regret_values)) == 2
        ),
    }


def _unrounded_deltas(
    baseline_state: Mapping[str, np.ndarray],
    policy_state: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> dict:
    selected = np.asarray(mask, dtype=bool)
    baseline_hit = np.asarray(baseline_state["hit"], dtype=bool)[selected]
    policy_hit = np.asarray(policy_state["hit"], dtype=bool)[selected]
    baseline_rank = np.asarray(baseline_state["first_rank"])[selected]
    policy_rank = np.asarray(policy_state["first_rank"])[selected]
    baseline_turn = np.asarray(baseline_state["first_turn"])[selected]
    policy_turn = np.asarray(policy_state["first_turn"])[selected]
    baseline_rr = np.where(
        baseline_hit, 1.0 / np.maximum(baseline_rank, 1), 0.0
    )
    policy_rr = np.where(
        policy_hit, 1.0 / np.maximum(policy_rank, 1), 0.0
    )
    baseline_mttc = np.where(baseline_hit, baseline_turn, 11)
    policy_mttc = np.where(policy_hit, policy_turn, 11)
    return {
        "unrounded_mrr_delta": float(policy_rr.mean() - baseline_rr.mean()),
        "unrounded_mttc_delta": float(
            policy_mttc.mean() - baseline_mttc.mean()
        ),
    }


def _transition(
    baseline_state: Mapping[str, np.ndarray],
    policy_state: Mapping[str, np.ndarray],
    supplement: np.ndarray,
    mask: np.ndarray,
) -> dict:
    return {
        **metric.transition_metrics(
            baseline_state, policy_state, supplement, mask
        ),
        **_unrounded_deltas(baseline_state, policy_state, mask),
    }


def _safe_transition(row: Mapping[str, Any]) -> bool:
    return bool(
        int(row["hit_to_miss"]) == 0
        and int(row["net_hits"]) >= 0
        and float(row["mrr_delta"]) >= 0.0
        and float(row["mttc_delta"]) <= 0.0
        and float(row["unrounded_mrr_delta"]) >= -1e-12
        and float(row["unrounded_mttc_delta"]) <= 1e-12
    )


def _selection_key(row: Mapping[str, Any]) -> Tuple[float, ...]:
    return (
        float(row["technical_score_delta"]),
        int(row["net_hits"]),
        int(row["positive_inner_folds"]),
        -int(row["activation_turns"]),
        float(row["quantile"]),
    )


def _select_inner_quantile(
    surface: PortfolioSurface,
    utility: np.ndarray,
    labels: Mapping[str, np.ndarray],
    current_state: Mapping[str, np.ndarray],
    train_sessions: np.ndarray,
    inner_fold: np.ndarray,
) -> dict:
    _slot, winner_candidate, winner_utility, winner_available = (
        _per_turn_winner_utilities(
            surface.candidates,
            surface.source_mask,
            surface.available,
            utility,
        )
    )
    candidates = []
    for quantile in (*frozen.QUANTILES, frozen.KEEP_QUANTILE):
        threshold = _map_outer_quantile(
            winner_utility, winner_available, train_sessions, quantile
        )
        supplement, supplemental_choice = _causal_latch(
            winner_candidate,
            winner_utility,
            winner_available,
            threshold,
            train_sessions,
        )
        final_chosen, final_activation = _compose_policy(
            surface.current_chosen,
            surface.current_activation,
            surface.candidates,
            surface.available,
            supplement,
            supplemental_choice,
        )
        policy_state = metric.policy_session_state(
            labels, final_chosen, final_activation
        )
        aggregate = _transition(
            current_state, policy_state, supplement, train_sessions
        )
        folds = []
        for fold in range(base.OUTER_FOLDS):
            fold_mask = train_sessions & (inner_fold == fold)
            folds.append(
                {
                    "fold": fold,
                    **_transition(
                        current_state, policy_state, supplement, fold_mask
                    ),
                }
            )
        positive_folds = sum(int(row["miss_to_hit"]) > 0 for row in folds)
        keep = float(quantile) >= frozen.KEEP_QUANTILE
        safe = _safe_transition(aggregate) and all(
            _safe_transition(row) for row in folds
        )
        evidence = int(aggregate["miss_to_hit"]) >= 1 and positive_folds >= 2
        if keep or (safe and evidence):
            candidates.append(
                {
                    "quantile": float(quantile),
                    "inner_threshold": _serialized_threshold(threshold),
                    **aggregate,
                    "positive_inner_folds": int(positive_folds),
                    "inner_fold_net_hits": [
                        int(row["net_hits"]) for row in folds
                    ],
                    "inner_fold_hit_to_miss": [
                        int(row["hit_to_miss"]) for row in folds
                    ],
                    "inner_fold_mrr_delta": [
                        float(row["mrr_delta"]) for row in folds
                    ],
                    "inner_fold_mttc_delta": [
                        float(row["mttc_delta"]) for row in folds
                    ],
                    "safe": bool(safe),
                    "evidence_minimum_passed": bool(evidence),
                }
            )
    if not candidates:
        raise PortfolioSelectorError("KEEP was not available")
    return max(candidates, key=_selection_key)
