"""Evaluate a causal selector over three frozen slot-10 proposal surfaces.

This is cached train_explore research only.  Runtime helpers receive no labels:
each turn deduplicates the three proposals, selects that turn's best action, and
uses the first threshold crossing in chronological order.  The current policy
is the immutable default and at most one supplemental action is allowed per
session.
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
from scripts import analyze_small_ranker_proposal_overlap as overlap  # noqa: E402
from scripts import analyze_small_ranker_remaining_misses as attribution  # noqa: E402
from scripts import evaluate_small_ranker_rrf3 as rrf3  # noqa: E402
from scripts import evaluate_small_ranker_supplemental_pairwise as supplemental  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-portfolio-selector-evaluation.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_7.portfolio_selector_preregistration.json"
)
IMPLEMENTATION_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_7.portfolio_selector_implementation_amendment.json"
)
PROPOSAL_PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_6.proposal_overlap_preregistration.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
CURRENT_HR = 0.9715
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
EXPECTED_CURRENT_ACTIVATION_SHA256 = (
    "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
)
EXPECTED_CURRENT_CHOSEN_SHA256 = (
    "229952c9ced7f6eec1ff1938480adc85ba5093ad865336465749029576e47051"
)
BIT_COUNT = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


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


@dataclass(frozen=True)
class SelectorOutput:
    supplement: np.ndarray
    supplemental_choice: np.ndarray
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


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _serialized_threshold(value: float) -> float | str:
    return "KEEP" if math.isinf(float(value)) else float(value)


def _deduplicate_actions(
    family_choices: np.ndarray,
    current_choice: np.ndarray,
    incumbent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build target-blind unique (session, turn, candidate) actions."""

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
            merged: dict[int, int] = {}
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose by utility, family-support count, then lower candidate ordinal."""

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
    winner_available = best_slot >= 0
    return best_slot, best_candidate, best_utility, winner_available


def _per_turn_winner_utilities(
    candidates: np.ndarray,
    source_mask: np.ndarray,
    available: np.ndarray,
    utility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _within_turn_winner(candidates, source_mask, available, utility)


def _causal_latch(
    winner_candidate: np.ndarray,
    winner_utility: np.ndarray,
    winner_available: np.ndarray,
    threshold: float,
    session_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Activate the earliest passing turn without consulting future turns."""

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
        & (
            (winner_candidate < 0)
            | (winner_candidate >= base.CANDIDATE_COUNT)
        )
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
) -> tuple[np.ndarray, np.ndarray]:
    _slot, candidate, winner_utility, winner_available = (
        _per_turn_winner_utilities(
            candidates,
            source_mask,
            available,
            utility,
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
) -> tuple[np.ndarray, np.ndarray]:
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
        rank = supplemental._allowed_rank_fraction(
            scores, safe_choice, incumbent
        )
        result[..., slot] = np.where(available[..., slot], rank, 0.0)
    return result


def _candidate_feature(
    features: np.ndarray,
    candidates: np.ndarray,
    available: np.ndarray,
    feature_name: str,
) -> np.ndarray:
    sessions = np.arange(candidates.shape[0])[:, None]
    turns = np.arange(candidates.shape[1])[None, :]
    index = base.FEATURE_INDEX[feature_name]
    result = np.zeros(candidates.shape, dtype=np.float32)
    for slot in range(candidates.shape[2]):
        safe_candidate = np.where(available[..., slot], candidates[..., slot], 0)
        value = features[sessions, turns, safe_candidate, index]
        result[..., slot] = np.where(available[..., slot], value, 0.0)
    return result


def _served_feature(
    features: np.ndarray, current_choice: np.ndarray, feature_name: str
) -> np.ndarray:
    sessions = np.arange(current_choice.shape[0])[:, None]
    turns = np.arange(current_choice.shape[1])[None, :]
    return np.asarray(
        features[
            sessions,
            turns,
            current_choice,
            base.FEATURE_INDEX[feature_name],
        ],
        dtype=np.float32,
    )


def _conflict_sum_for_actions(
    features: np.ndarray, candidates: np.ndarray, available: np.ndarray
) -> np.ndarray:
    result = np.zeros(candidates.shape, dtype=np.float32)
    for slot_name in base.CONSTRAINT_SLOTS:
        result += _candidate_feature(
            features, candidates, available, f"{slot_name}_conflict"
        )
    return result


def _served_conflict_sum(
    features: np.ndarray, current_choice: np.ndarray
) -> np.ndarray:
    result = np.zeros(current_choice.shape, dtype=np.float32)
    for slot_name in base.CONSTRAINT_SLOTS:
        result += _served_feature(
            features, current_choice, f"{slot_name}_conflict"
        )
    return result


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


def _build_runtime_surface(
    features: np.ndarray,
    current_scores: np.ndarray,
    family_scores: Sequence[np.ndarray],
    current_chosen: np.ndarray,
    current_activation: np.ndarray,
    incumbent: np.ndarray,
) -> RuntimePortfolioSurface:
    if len(family_scores) != len(FAMILY_NAMES):
        raise PortfolioSelectorError("portfolio requires three score surfaces")
    family_choices = np.stack(
        [base.choose_slot10(scores, incumbent)[0] for scores in family_scores],
        axis=2,
    ).astype(np.uint8)
    current_choice = np.where(
        current_activation, current_chosen, incumbent
    ).astype(np.uint8)
    candidates, source_mask, available = _deduplicate_actions(
        family_choices, current_choice, incumbent
    )

    shape = (*available.shape, len(FEATURE_NAMES))
    gate_features = np.zeros(shape, dtype=np.float32)
    gate_features[..., 0] = current_activation[..., None]
    gate_features[..., 1] = (
        (np.arange(current_choice.shape[1], dtype=np.float32) + 1.0)
        / float(current_choice.shape[1])
    )[None, :, None]
    prior_activation = (
        np.cumsum(current_activation, axis=1, dtype=np.int16)
        - current_activation.astype(np.int16)
    ) / float(current_choice.shape[1])
    gate_features[..., 2] = prior_activation[..., None]
    gate_features[..., 3] = (
        available.sum(axis=2, dtype=np.float32) / float(MAX_ACTIONS)
    )[..., None]
    for family in range(len(FAMILY_NAMES)):
        gate_features[..., 4 + family] = (
            source_mask & (1 << family) != 0
        ).astype(np.float32)

    all_scores = (current_scores, *family_scores)
    for index, scores in enumerate(all_scores):
        gate_features[..., 7 + index] = _rank_fraction_for_actions(
            scores, candidates, available, incumbent
        )
    for index, scores in enumerate(family_scores):
        current_rank = supplemental._allowed_rank_fraction(
            scores, current_choice, incumbent
        )
        gate_features[..., 11 + index] = current_rank[..., None]

    gate_features[..., 14] = _candidate_feature(
        features, candidates, available, "coverage_rank_fraction"
    )
    for offset, feature_name in enumerate(
        (
            "top10_route_agreement_fraction",
            "active_token_recall",
            "hard_clause_coverage",
        ),
        start=15,
    ):
        candidate_value = _candidate_feature(
            features, candidates, available, feature_name
        )
        served_value = _served_feature(features, current_choice, feature_name)
        gate_features[..., offset] = np.where(
            available, candidate_value - served_value[..., None], 0.0
        )
    candidate_conflict = _conflict_sum_for_actions(
        features, candidates, available
    )
    served_conflict = _served_conflict_sum(features, current_choice)
    gate_features[..., 18] = np.where(
        available, candidate_conflict - served_conflict[..., None], 0.0
    )
    gate_features = np.where(
        available[..., None], gate_features, 0.0
    ).astype(np.float32)
    if gate_features.shape != shape or not np.isfinite(gate_features).all():
        raise PortfolioSelectorError("portfolio feature schema mismatch")
    return RuntimePortfolioSurface(
        current_chosen=np.asarray(current_chosen, dtype=np.uint8),
        current_activation=np.asarray(current_activation, dtype=bool),
        current_choice=current_choice,
        incumbent=np.asarray(incumbent, dtype=np.uint8),
        family_choices=family_choices,
        candidates=candidates,
        source_mask=source_mask,
        available=available,
        features=gate_features,
    )


def _attach_isolated_labels(
    runtime: RuntimePortfolioSurface,
    labels: Mapping[str, np.ndarray],
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
        slot_labels = supplemental._isolated_labels(
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


def _fit_readiness(
    rows: np.ndarray,
    rescue: np.ndarray,
    regret: np.ndarray,
    row_session: np.ndarray,
    row_family: np.ndarray,
) -> dict[str, Any]:
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


def _fit_predict(
    x: np.ndarray,
    train_rows: np.ndarray,
    predict_rows: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    model, mean, scale = supplemental._fit_gate_model(
        x[train_rows], target[train_rows], weights[train_rows], MODEL_SEED
    )
    _validate_fitted_model(model)
    return base._predict_gate(
        model, mean, scale, x[predict_rows]
    ).astype(np.float32)


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


def _unrounded_deltas(
    baseline_state: Mapping[str, np.ndarray],
    policy_state: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float]:
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
    policy_rr = np.where(policy_hit, 1.0 / np.maximum(policy_rank, 1), 0.0)
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
) -> dict[str, Any]:
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


def _selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
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
) -> dict[str, Any]:
    _slot, winner_candidate, winner_utility, winner_available = (
        _per_turn_winner_utilities(
            surface.candidates,
            surface.source_mask,
            surface.available,
            utility,
        )
    )
    candidates: list[dict[str, Any]] = []
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


def _nested_oof(
    surface: PortfolioSurface,
    labels: Mapping[str, np.ndarray],
) -> SelectorOutput:
    outer = np.asarray(labels["outer_fold"], dtype=np.int16)
    inner = np.asarray(labels["inner_fold"], dtype=np.int16)
    flat_x = surface.features.reshape(-1, len(FEATURE_NAMES))
    flat_available = surface.available.reshape(-1)
    flat_session = np.repeat(
        np.arange(len(outer)), surface.available.shape[1] * MAX_ACTIONS
    )
    flat_family = np.repeat(
        np.asarray(labels["family_index"]),
        surface.available.shape[1] * MAX_ACTIONS,
    )
    targets = (surface.rescue.reshape(-1), surface.regret.reshape(-1))
    weights = (
        surface.rescue_weights.reshape(-1),
        surface.regret_weights.reshape(-1),
    )
    supplement = np.zeros(surface.current_chosen.shape, dtype=bool)
    supplemental_choice = np.full(surface.current_chosen.shape, -1, dtype=np.int16)
    rescue_probability = np.zeros(surface.available.shape, dtype=np.float32)
    regret_probability = np.zeros(surface.available.shape, dtype=np.float32)
    current_state = metric.policy_session_state(
        labels, surface.current_chosen, surface.current_activation
    )
    selections: list[dict[str, Any]] = []

    for outer_fold in range(base.OUTER_FOLDS):
        train_sessions = outer != outer_fold
        held_sessions = outer == outer_fold
        inner_probability = [
            np.zeros(surface.available.shape, dtype=np.float32) for _ in range(2)
        ]
        inner_readiness: list[dict[str, Any]] = []
        blocked = False
        for inner_index in range(base.OUTER_FOLDS):
            model_train = train_sessions & (inner != inner_index)
            model_valid = train_sessions & (inner == inner_index)
            train_rows = flat_available & model_train[flat_session]
            valid_rows = flat_available & model_valid[flat_session]
            readiness = _fit_readiness(
                train_rows,
                targets[0],
                targets[1],
                flat_session,
                flat_family,
            )
            readiness["inner_fold"] = inner_index
            readiness["valid_action_rows"] = int(valid_rows.sum())
            inner_readiness.append(readiness)
            if not readiness["ready"] or not valid_rows.any():
                blocked = True
                continue
            for head in range(2):
                prediction = _fit_predict(
                    flat_x,
                    train_rows,
                    valid_rows,
                    targets[head],
                    weights[head],
                )
                inner_probability[head].reshape(-1)[valid_rows] = prediction

        selected: dict[str, Any]
        outer_readiness: dict[str, Any] | None = None
        mapped_threshold = math.inf
        if blocked:
            selected = {
                "quantile": frozen.KEEP_QUANTILE,
                "status": "KEEP_INSUFFICIENT_INNER_FIT",
            }
        else:
            inner_utility = (
                inner_probability[0]
                - REGRET_MULTIPLIER * inner_probability[1]
            )
            selected = _select_inner_quantile(
                surface,
                inner_utility,
                labels,
                current_state,
                train_sessions,
                inner,
            )
            if float(selected["quantile"]) < frozen.KEEP_QUANTILE:
                train_rows = flat_available & train_sessions[flat_session]
                held_rows = flat_available & held_sessions[flat_session]
                outer_readiness = _fit_readiness(
                    train_rows,
                    targets[0],
                    targets[1],
                    flat_session,
                    flat_family,
                )
                if outer_readiness["ready"] and held_rows.any():
                    train_probability = [
                        np.zeros(surface.available.shape, dtype=np.float32)
                        for _ in range(2)
                    ]
                    held_probability = [
                        np.zeros(surface.available.shape, dtype=np.float32)
                        for _ in range(2)
                    ]
                    for head in range(2):
                        model, mean, scale = supplemental._fit_gate_model(
                            flat_x[train_rows],
                            targets[head][train_rows],
                            weights[head][train_rows],
                            MODEL_SEED,
                        )
                        _validate_fitted_model(model)
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
                        - REGRET_MULTIPLIER * train_probability[1]
                    )
                    held_utility = (
                        held_probability[0]
                        - REGRET_MULTIPLIER * held_probability[1]
                    )
                    _slot, _candidate, train_winner, train_winner_available = (
                        _per_turn_winner_utilities(
                            surface.candidates,
                            surface.source_mask,
                            surface.available,
                            train_utility,
                        )
                    )
                    mapped_threshold = _map_outer_quantile(
                        train_winner,
                        train_winner_available,
                        train_sessions,
                        float(selected["quantile"]),
                    )
                    held_supplement, held_choice = _causal_policy(
                        surface.candidates,
                        surface.source_mask,
                        surface.available,
                        held_utility,
                        mapped_threshold,
                        held_sessions,
                    )
                    supplement[held_sessions] = held_supplement[held_sessions]
                    supplemental_choice[held_sessions] = held_choice[held_sessions]
                    rescue_probability[held_sessions] = held_probability[0][
                        held_sessions
                    ]
                    regret_probability[held_sessions] = held_probability[1][
                        held_sessions
                    ]
                else:
                    selected = {
                        **selected,
                        "proposed_quantile": float(selected["quantile"]),
                        "quantile": frozen.KEEP_QUANTILE,
                        "status": "KEEP_INSUFFICIENT_OUTER_FIT",
                    }
            else:
                selected = {**selected, "status": "KEEP_SELECTED"}
        selections.append(
            {
                "fold": outer_fold,
                "selected_quantile": float(selected["quantile"]),
                "mapped_outer_train_threshold": _serialized_threshold(
                    mapped_threshold
                ),
                "inner_selection": selected,
                "inner_fit_readiness": inner_readiness,
                "outer_fit_readiness": outer_readiness,
            }
        )
    final_chosen, final_activation = _compose_policy(
        surface.current_chosen,
        surface.current_activation,
        surface.candidates,
        surface.available,
        supplement,
        supplemental_choice,
    )
    return SelectorOutput(
        supplement=supplement,
        supplemental_choice=supplemental_choice,
        final_chosen=final_chosen,
        final_activation=final_activation,
        rescue_probability=rescue_probability,
        regret_probability=regret_probability,
        selections=selections,
    )


def _metrics(
    labels: Mapping[str, np.ndarray],
    current_state: Mapping[str, np.ndarray],
    output: SelectorOutput,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    final_state = metric.policy_session_state(
        labels, output.final_chosen, output.final_activation
    )
    outer = np.asarray(labels["outer_fold"])
    aggregate = _transition(
        current_state,
        final_state,
        output.supplement,
        np.ones(len(outer), dtype=bool),
    )
    folds = [
        {
            "fold": fold,
            **_transition(
                current_state,
                final_state,
                output.supplement,
                outer == fold,
            ),
        }
        for fold in range(base.OUTER_FOLDS)
    ]
    return aggregate, folds, final_state


def _promotion_gate(
    aggregate: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    exact_repeat: bool,
) -> bool:
    return bool(
        exact_repeat
        and float(aggregate["policy"]["hit_rate_at_10"]) > CURRENT_HR
        and float(aggregate["technical_score_delta"]) > 0.0
        and _safe_transition(aggregate)
        and all(_safe_transition(row) for row in folds)
    )


def _portfolio_oracle(
    surface: PortfolioSurface,
    labels: Mapping[str, np.ndarray],
    current_state: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    positive = np.asarray(labels["positive_index"], dtype=np.int16)
    eligible_from = np.asarray(labels["eligible_from"], dtype=np.int16)
    eligible = (
        np.arange(1, surface.current_chosen.shape[1] + 1)[None, :]
        >= eligible_from[:, None]
    )
    correct = (
        surface.available
        & eligible[..., None]
        & (positive[..., None] >= 0)
        & (surface.candidates == positive[..., None])
    )
    current_miss = ~np.asarray(current_state["hit"], dtype=bool)
    reachable = current_miss & np.any(correct, axis=(1, 2))
    outer = np.asarray(labels["outer_fold"])
    return {
        "current_miss_sessions": int(current_miss.sum()),
        "correct_action_rows_on_current_misses": int(correct[current_miss].sum()),
        "reachable_current_miss_sessions": int(reachable.sum()),
        "reachable_by_outer_fold": [
            int(np.sum(reachable & (outer == fold)))
            for fold in range(base.OUTER_FOLDS)
        ],
        "maximum_zero_harm_hr_at_10": round(
            float((np.asarray(current_state["hit"]) | reachable).mean()), 6
        ),
        "posthoc_target_informed_not_runtime": True,
    }


def _validate_preregistration(
    preregistration: Mapping[str, Any], amendment: Mapping[str, Any]
) -> None:
    model = preregistration.get("model", {})
    runtime = preregistration.get("runtime_feature_contract", {})
    action = preregistration.get("action_contract", {})
    frozen_inputs = preregistration.get("frozen_inputs", {})
    constants = amendment.get("implementation_constants", {})
    if not (
        preregistration.get("schema_version")
        == "small-ranker-portfolio-selector-preregistration.v1"
        and amendment.get("schema_version")
        == "small-ranker-portfolio-selector-implementation-amendment.v1"
        and tuple(runtime.get("feature_names", [])) == FEATURE_NAMES
        and int(runtime.get("feature_count", -1)) == len(FEATURE_NAMES)
        and int(action.get("maximum_actions_per_turn", -1)) == MAX_ACTIONS
        and int(action.get("maximum_supplemental_actions_per_session", -1)) == 1
        and float(model.get("C", -1.0)) == 0.2
        and model.get("solver") == "liblinear"
        and model.get("fit_intercept") is True
        and int(model.get("max_iter", -1)) == 300
        and int(model.get("random_state", -1)) == MODEL_SEED
        and int(model.get("minimum_distinct_rescue_sessions_per_fit", -1))
        == MIN_RESCUE_SESSIONS
        and model.get("utility")
        == "P(rescue) - 1.0*P(composite_regret)"
        and frozen_inputs.get("candidate_universe") == "C100"
        and frozen_inputs.get("current_activation_sha256")
        == EXPECTED_CURRENT_ACTIVATION_SHA256
        and frozen_inputs.get("current_semantic_off_oof_score_sha256")
        == frozen.EXPECTED_HASHES["projected_oof_scores"]
        and int(constants.get("max_actions", -1)) == MAX_ACTIONS
        and int(constants.get("model_seed", -1)) == MODEL_SEED
        and float(constants.get("regret_multiplier", -1.0))
        == REGRET_MULTIPLIER
        and int(constants.get("minimum_rescue_sessions", -1))
        == MIN_RESCUE_SESSIONS
        and int(constants.get("minimum_rescue_families", -1))
        == MIN_RESCUE_FAMILIES
        and constants.get("quantiles")
        == "0/64 through 63/64 plus KEEP=1.0"
        and frozen.QUANTILES
        == tuple(float(value) / 64.0 for value in range(64))
        and frozen.KEEP_QUANTILE == 1.0
    ):
        raise PortfolioSelectorError("portfolio preregistration binding drifted")


def _validate_fold_integrity(labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    family = np.asarray(labels["family_index"])
    outer = np.asarray(labels["outer_fold"])
    inner = np.asarray(labels["inner_fold"])
    expected_shape = (base.SESSION_COUNT,)
    if family.shape != expected_shape or outer.shape != expected_shape or inner.shape != expected_shape:
        raise PortfolioSelectorError("fold metadata shape mismatch")
    expected_folds = set(range(base.OUTER_FOLDS))
    if set(int(value) for value in np.unique(outer)) != expected_folds or set(
        int(value) for value in np.unique(inner)
    ) != expected_folds:
        raise PortfolioSelectorError("fold metadata range mismatch")
    for value in np.unique(family):
        selected = family == value
        if len(np.unique(outer[selected])) != 1 or len(np.unique(inner[selected])) != 1:
            raise PortfolioSelectorError("product family crosses a fixed fold")
    joint = [
        [int(np.sum((outer == outer_fold) & (inner == inner_fold))) for inner_fold in range(base.OUTER_FOLDS)]
        for outer_fold in range(base.OUTER_FOLDS)
    ]
    if min(min(row) for row in joint) <= 0:
        raise PortfolioSelectorError("outer-by-inner fold cell is empty")
    return {
        "product_families": int(len(np.unique(family))),
        "outer_session_counts": [
            int(np.sum(outer == fold)) for fold in range(base.OUTER_FOLDS)
        ],
        "inner_session_counts": [
            int(np.sum(inner == fold)) for fold in range(base.OUTER_FOLDS)
        ],
        "outer_by_inner_session_counts": joint,
        "every_family_outer_and_inner_unique": True,
    }


def _load_proposal_scores(
    inputs: frozen.FreezeInputs,
    preregistration: Mapping[str, Any],
) -> tuple[list[np.ndarray], Mapping[str, Any]]:
    proposal_prereg = json.loads(
        PROPOSAL_PREREGISTRATION.read_text(encoding="utf-8")
    )
    if (
        proposal_prereg.get("schema_version")
        != "small-ranker-proposal-overlap-preregistration.v1"
    ):
        raise PortfolioSelectorError("proposal preregistration mismatch")
    fixed = proposal_prereg["fixed_surfaces"]
    expected = preregistration["frozen_inputs"]
    pairwise = overlap._local_score(
        fixed["pairwise"]["path"], expected["pairwise_semantic_off_oof_score_sha256"]
    )
    focused = overlap._local_score(
        fixed["focused_lambdamart"]["path"],
        expected["focused_lambdamart_oof_score_sha256"],
    )
    members = [
        overlap._local_score(path, expected_hash)
        for path, expected_hash in zip(
            fixed["rrf3"]["members"][1:],
            fixed["rrf3"]["member_sha256"][1:],
            strict=True,
        )
    ]
    rrf_scores = rrf3.rrf_scores([inputs.oof_scores, *members])
    if _array_sha256(rrf_scores) != expected["rrf3_combined_score_sha256"]:
        raise PortfolioSelectorError("RRF-3 score identity mismatch")
    return [pairwise, rrf_scores, focused], proposal_prereg


def run(
    source_root: Path,
    projection_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = output_path.resolve()
    experiments_root = (ROOT / "experiments").resolve()
    if (
        output_path.exists()
        or output_path.is_symlink()
        or experiments_root not in output_path.parents
    ):
        raise PortfolioSelectorError("output must be new and below experiments")
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    amendment = json.loads(
        IMPLEMENTATION_AMENDMENT.read_text(encoding="utf-8")
    )
    _validate_preregistration(preregistration, amendment)

    inputs = frozen._load_inputs(source_root, projection_root)
    fold_integrity = _validate_fold_integrity(inputs.labels)
    proposal_scores, proposal_prereg = _load_proposal_scores(
        inputs, preregistration
    )
    current_surface = frozen._action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    current_activation, current_selections = (
        attribution._reproduce_nested_activation(
            current_surface, inputs.labels, seed=40220260830
        )
    )
    if (
        _array_sha256(current_activation) != EXPECTED_CURRENT_ACTIVATION_SHA256
        or _array_sha256(current_surface.chosen) != EXPECTED_CURRENT_CHOSEN_SHA256
    ):
        raise PortfolioSelectorError("frozen current policy did not reproduce")

    build_started = time.perf_counter()
    runtime_surface = _build_runtime_surface(
        inputs.projected_features,
        inputs.oof_scores,
        proposal_scores,
        current_surface.chosen,
        current_activation,
        current_surface.incumbent,
    )
    runtime_feature_sha256 = _array_sha256(runtime_surface.features)
    surface = _attach_isolated_labels(runtime_surface, inputs.labels)
    if _array_sha256(surface.features) != runtime_feature_sha256:
        raise PortfolioSelectorError("attaching labels changed runtime features")
    build_seconds = time.perf_counter() - build_started
    current_state = metric.policy_session_state(
        inputs.labels, surface.current_chosen, surface.current_activation
    )
    zero = np.zeros_like(surface.current_activation, dtype=bool)
    p11_state = metric.policy_session_state(
        inputs.labels, surface.current_chosen, zero
    )
    all_sessions = np.ones(base.SESSION_COUNT, dtype=bool)
    current_vs_p11 = _transition(
        p11_state, current_state, surface.current_activation, all_sessions
    )
    if not (
        float(current_vs_p11["policy"]["hit_rate_at_10"]) == CURRENT_HR
        and int(current_vs_p11["miss_to_hit"]) == 48
        and int(current_vs_p11["hit_to_miss"]) == 0
    ):
        raise PortfolioSelectorError("current comparator metric drifted")

    first_started = time.perf_counter()
    first = _nested_oof(surface, inputs.labels)
    first_seconds = time.perf_counter() - first_started
    repeat_started = time.perf_counter()
    repeat = _nested_oof(surface, inputs.labels)
    repeat_seconds = time.perf_counter() - repeat_started
    exact = bool(
        np.array_equal(first.supplement, repeat.supplement)
        and np.array_equal(first.supplemental_choice, repeat.supplemental_choice)
        and np.array_equal(first.final_chosen, repeat.final_chosen)
        and np.array_equal(first.final_activation, repeat.final_activation)
        and np.array_equal(first.rescue_probability, repeat.rescue_probability)
        and np.array_equal(first.regret_probability, repeat.regret_probability)
        and _canonical_sha256(first.selections)
        == _canonical_sha256(repeat.selections)
    )
    if not exact:
        raise PortfolioSelectorError("portfolio selector exact repeat failed")

    relative, folds, final_state = _metrics(inputs.labels, current_state, first)
    direction_passed = _promotion_gate(relative, folds, exact)
    final_vs_p11 = _transition(
        p11_state, final_state, first.final_activation, all_sessions
    )
    oracle = _portfolio_oracle(surface, inputs.labels, current_state)
    if not (
        int(oracle["reachable_current_miss_sessions"]) == 20
        and oracle["reachable_by_outer_fold"] == [9, 2, 7, 1, 1]
        and float(oracle["maximum_zero_harm_hr_at_10"]) == 0.9815
    ):
        raise PortfolioSelectorError("frozen portfolio oracle invariant drifted")
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.7-CAUSAL-PORTFOLIO-SELECTOR",
        "scope": {
            "split": "train_explore",
            "cached_inputs_only": True,
            "agent_or_full_evaluator_started": False,
            "held_out_splits_opened": False,
            "calibration_selection_confirmation_public_opened": False,
            "external_data_downloaded": False,
            "runtime_features_target_blind": True,
            "runtime_surface_contains_training_labels": False,
            "eligible_from_not_in_runtime_api": True,
            "target_is_training_or_posthoc_label_only": True,
            "claim": "nested admission conditional on frozen OOF proposal surfaces, not end-to-end nested stacking",
            "full_model_or_runtime_artifact_trained": False,
        },
        "sources": {
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "implementation_amendment_sha256": _sha256(
                IMPLEMENTATION_AMENDMENT
            ),
            "proposal_preregistration_sha256": _sha256(
                PROPOSAL_PREREGISTRATION
            ),
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
            "current_oof_score_sha256": frozen.EXPECTED_HASHES[
                "projected_oof_scores"
            ],
            "pairwise_score_sha256": preregistration["frozen_inputs"][
                "pairwise_semantic_off_oof_score_sha256"
            ],
            "rrf3_score_sha256": preregistration["frozen_inputs"][
                "rrf3_combined_score_sha256"
            ],
            "focused_score_sha256": preregistration["frozen_inputs"][
                "focused_lambdamart_oof_score_sha256"
            ],
            "feature_cache_sha256": frozen.EXPECTED_HASHES["features"],
            "projected_features_sha256": frozen.EXPECTED_HASHES[
                "projected_features"
            ],
            "label_cache_sha256": frozen.EXPECTED_HASHES["labels"],
            "proposal_schema_sha256": _canonical_sha256(proposal_prereg),
        },
        "protocol": {
            "fold_integrity": fold_integrity,
            "max_actions_per_turn": MAX_ACTIONS,
            "maximum_supplements_per_session": 1,
            "minimum_rescue_sessions_per_fit": MIN_RESCUE_SESSIONS,
            "minimum_rescue_families_per_fit": MIN_RESCUE_FAMILIES,
            "model_seed": MODEL_SEED,
            "regret_multiplier": REGRET_MULTIPLIER,
            "numpy_version": np.__version__,
            "sklearn_version": __import__("sklearn").__version__,
        },
        "feature_contract": {
            "names": list(FEATURE_NAMES),
            "count": len(FEATURE_NAMES),
            "schema_sha256": _canonical_sha256(list(FEATURE_NAMES)),
        },
        "surface": {
            "raw_family_action_rows": int(
                sum(
                    np.sum(
                        (surface.family_choices[..., family] != surface.current_choice)
                        & (surface.family_choices[..., family] != surface.incumbent)
                    )
                    for family in range(len(FAMILY_NAMES))
                )
            ),
            "deduplicated_action_rows": int(surface.available.sum()),
            "action_sessions": int(
                np.any(surface.available, axis=(1, 2)).sum()
            ),
            "action_session_turns": int(
                np.any(surface.available, axis=2).sum()
            ),
            "rescue_label_rows": int(surface.rescue.sum()),
            "rescue_label_sessions": int(
                np.any(surface.rescue > 0, axis=(1, 2)).sum()
            ),
            "regret_label_rows": int(surface.regret.sum()),
            "regret_label_sessions": int(
                np.any(surface.regret > 0, axis=(1, 2)).sum()
            ),
            "candidate_sha256": _array_sha256(surface.candidates),
            "source_mask_sha256": _array_sha256(surface.source_mask),
            "feature_sha256": _array_sha256(surface.features),
        },
        "current": {
            "activation_sha256": _array_sha256(surface.current_activation),
            "chosen_sha256": _array_sha256(surface.current_chosen),
            "selections_sha256": _canonical_sha256(current_selections),
            "versus_p11": current_vs_p11,
        },
        "challenger": {
            "relative_to_current": relative,
            "folds_relative_to_current": folds,
            "versus_p11": final_vs_p11,
            "outer_selections": first.selections,
            "supplemental_activation_sha256": _array_sha256(first.supplement),
            "supplemental_choice_sha256": _array_sha256(
                first.supplemental_choice
            ),
            "final_chosen_sha256": _array_sha256(first.final_chosen),
            "final_activation_sha256": _array_sha256(first.final_activation),
        },
        "repeat": {
            "exact": exact,
            "selection_canonical_sha256": _canonical_sha256(first.selections),
            "rescue_probability_sha256": _array_sha256(
                first.rescue_probability
            ),
            "regret_probability_sha256": _array_sha256(
                first.regret_probability
            ),
        },
        "portfolio_oracle": oracle,
        "decision": {
            "direction_gate_passed": direction_passed,
            "status": (
                "DIRECTION_PASS_STRICT_RESTACK_REQUIRED"
                if direction_passed
                else "NO_GO_ONE_ACTION_SELECTOR"
            ),
            "full_artifact_authorized": False,
            "strict_restack_required_before_deployment_claim": direction_passed,
            "next": (
                "regenerate all three upstream proposals inside every meta fold"
                if direction_passed
                else "close this selector without tuning and register a materially new proposal or explicitly posthoc multi-action validation route"
            ),
        },
        "timing_seconds": {
            "surface_and_labels": round(build_seconds, 6),
            "first_nested_oof": round(first_seconds, 6),
            "repeat_nested_oof": round(repeat_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
        "in_memory_array_bytes": {
            "runtime_surface": int(
                sum(
                    value.nbytes
                    for value in (
                        surface.current_chosen,
                        surface.current_activation,
                        surface.current_choice,
                        surface.incumbent,
                        surface.family_choices,
                        surface.candidates,
                        surface.source_mask,
                        surface.available,
                        surface.features,
                    )
                )
            ),
            "training_outcomes_and_weights": int(
                sum(
                    value.nbytes
                    for value in (
                        surface.rescue,
                        surface.rescue_weights,
                        surface.regret,
                        surface.regret_weights,
                        surface.rr_loss,
                        surface.mttc_loss,
                    )
                )
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
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
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
