"""Test an explicit first-hit reciprocal-rank harm head in nested OOF.

The frozen ranker and rich target-blind features are unchanged.  A second
logistic head predicts whether replacing P11 slot 10 at a turn removes the
baseline session's earliest hit.  Inner OOF selects a fixed harm multiplier
and utility threshold under HR, MRR, and MTTC constraints; outer OOF is the
only reported evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_small_ranker_metric_gate as metric  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-mrr-harm-gate-diagnostic.v1"
HARM_MULTIPLIERS = (0.5, 1.0, 2.0, 5.0)


def first_hit_harm_labels(
    labels: Mapping[str, np.ndarray], chosen: np.ndarray, incumbent: np.ndarray
) -> np.ndarray:
    """Label an action that removes the earliest eligible baseline rank-10 hit."""

    baseline_rank = np.asarray(labels["baseline_rank"])
    positive = np.asarray(labels["positive_index"])
    eligible_from = np.asarray(labels["eligible_from"])
    turns = np.arange(1, baseline_rank.shape[1] + 1)[None, :]
    eligible = turns >= eligible_from[:, None]
    baseline_turn_hit = eligible & (baseline_rank > 0)
    baseline_hit = baseline_turn_hit.any(axis=1)
    first_turn = np.where(baseline_hit, np.argmax(baseline_turn_hit, axis=1) + 1, 11)
    action = chosen != incumbent
    return (
        action
        & eligible
        & (turns == first_turn[:, None])
        & (baseline_rank == 10)
        & ((positive < 0) | (chosen != positive))
    ).astype(np.uint8)


def _choice_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["technical_score_delta"]),
        int(row["net_hits"]),
        int(row["miss_to_hit"]),
        -int(row["activation_turns"]),
        float(row["threshold"]),
        float(row["harm_multiplier"]),
    )


def compare_harm_head(
    features: np.ndarray,
    scores: np.ndarray,
    labels: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    incumbent = base._incumbent_indices(features)
    chosen, margin, top_gap = base.choose_slot10(scores, incumbent)
    gate_features = base.gate_feature_matrix(features, scores, chosen, incumbent, margin, top_gap)
    rescue, direct_risk, rescue_weights = base.action_training_labels(labels, chosen, incumbent)
    first_hit_harm = first_hit_harm_labels(labels, chosen, incumbent)
    harm_weights = np.where(
        first_hit_harm > 0,
        10.0,
        np.where(rescue > 0, 0.2, 0.05),
    ).astype(np.float64)
    action = chosen != incumbent
    outer = np.asarray(labels["outer_fold"])
    inner = np.asarray(labels["inner_fold"])
    flat_x = gate_features.reshape(-1, gate_features.shape[-1])
    flat_rescue = rescue.reshape(-1)
    flat_harm = first_hit_harm.reshape(-1)
    flat_rescue_weights = rescue_weights.reshape(-1)
    flat_harm_weights = harm_weights.reshape(-1)
    flat_action = action.reshape(-1)
    flat_session = np.repeat(np.arange(len(outer)), base.TURN_COUNT)
    rescue_probability = np.zeros(action.shape, dtype=np.float32)
    harm_probability = np.zeros(action.shape, dtype=np.float32)
    thresholds = np.full(base.OUTER_FOLDS, math.inf, dtype=np.float64)
    multipliers = np.zeros(base.OUTER_FOLDS, dtype=np.float64)
    selections: list[dict[str, Any]] = []
    for outer_fold in range(base.OUTER_FOLDS):
        train_sessions = outer != outer_fold
        held_sessions = outer == outer_fold
        inner_rescue = np.zeros(action.shape, dtype=np.float32)
        inner_harm = np.zeros(action.shape, dtype=np.float32)
        for inner_fold in range(base.OUTER_FOLDS):
            model_train = train_sessions & (inner != inner_fold)
            model_valid = train_sessions & (inner == inner_fold)
            train_rows = flat_action & model_train[flat_session]
            valid_rows = flat_action & model_valid[flat_session]
            if not np.any(valid_rows):
                continue
            rescue_model, rescue_mean, rescue_scale = base._fit_gate_model(
                flat_x[train_rows],
                flat_rescue[train_rows],
                flat_rescue_weights[train_rows],
                seed + outer_fold * 31 + inner_fold,
            )
            harm_model, harm_mean, harm_scale = base._fit_gate_model(
                flat_x[train_rows],
                flat_harm[train_rows],
                flat_harm_weights[train_rows],
                seed + 10_000 + outer_fold * 31 + inner_fold,
            )
            inner_rescue.reshape(-1)[valid_rows] = base._predict_gate(
                rescue_model, rescue_mean, rescue_scale, flat_x[valid_rows]
            ).astype(np.float32)
            inner_harm.reshape(-1)[valid_rows] = base._predict_gate(
                harm_model, harm_mean, harm_scale, flat_x[valid_rows]
            ).astype(np.float32)
        choices: list[dict[str, Any]] = []
        for multiplier in HARM_MULTIPLIERS:
            utility = inner_rescue - multiplier * inner_harm
            selected = metric.select_metric_safe_threshold(
                utility, action, chosen, labels, train_sessions
            )
            choices.append({"harm_multiplier": multiplier, **selected})
        choice = max(choices, key=_choice_key)
        thresholds[outer_fold] = float(choice["threshold"])
        multipliers[outer_fold] = float(choice["harm_multiplier"])
        train_rows = flat_action & train_sessions[flat_session]
        held_rows = flat_action & held_sessions[flat_session]
        rescue_model, rescue_mean, rescue_scale = base._fit_gate_model(
            flat_x[train_rows],
            flat_rescue[train_rows],
            flat_rescue_weights[train_rows],
            seed + outer_fold * 101,
        )
        harm_model, harm_mean, harm_scale = base._fit_gate_model(
            flat_x[train_rows],
            flat_harm[train_rows],
            flat_harm_weights[train_rows],
            seed + 10_000 + outer_fold * 101,
        )
        rescue_probability.reshape(-1)[held_rows] = base._predict_gate(
            rescue_model, rescue_mean, rescue_scale, flat_x[held_rows]
        ).astype(np.float32)
        harm_probability.reshape(-1)[held_rows] = base._predict_gate(
            harm_model, harm_mean, harm_scale, flat_x[held_rows]
        ).astype(np.float32)
        selections.append(
            {
                "fold": outer_fold,
                "selected": choice,
                "multiplier_comparison": choices,
                "train_rescue_rows": int(flat_rescue[train_rows].sum()),
                "train_first_hit_harm_rows": int(flat_harm[train_rows].sum()),
                "train_direct_risk_rows": int(direct_risk.reshape(-1)[train_rows].sum()),
            }
        )
    utility = rescue_probability - multipliers[outer][:, None] * harm_probability
    activation = action & (utility >= thresholds[outer][:, None])
    zero = np.zeros_like(action, dtype=bool)
    baseline_state = metric.policy_session_state(labels, chosen, zero)
    policy_state = metric.policy_session_state(labels, chosen, activation)
    all_sessions = np.ones(len(outer), dtype=bool)
    result: dict[str, Any] = {
        "harm_label_rows": int(first_hit_harm.sum()),
        "direct_risk_rows": int(direct_risk.sum()),
        "rescue_label_rows": int(rescue.sum()),
        "global": metric.transition_metrics(
            baseline_state, policy_state, activation, all_sessions
        ),
        "folds": [],
        "inner_selections": selections,
        "activated_first_hit_harm_rows": int((activation & (first_hit_harm > 0)).sum()),
        "probability_ranges": {
            "rescue_min": round(float(rescue_probability[action].min()), 8),
            "rescue_max": round(float(rescue_probability[action].max()), 8),
            "harm_min": round(float(harm_probability[action].min()), 8),
            "harm_max": round(float(harm_probability[action].max()), 8),
        },
    }
    for fold in range(base.OUTER_FOLDS):
        mask = outer == fold
        result["folds"].append(
            {
                "fold": fold,
                "harm_multiplier": float(multipliers[fold]),
                "threshold": float(thresholds[fold]),
                **metric.transition_metrics(
                    baseline_state, policy_state, activation, mask
                ),
            }
        )
    return result


def run(source_root: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    source_root = source_root.resolve()
    feature_path = source_root / "experiments/fast_track/small_ranker_v1/features.npy"
    label_path = source_root / "experiments/fast_track/small_ranker_v1/labels_v2.npz"
    score_path = source_root / "experiments/fast_track/small_ranker_v1/oof_batch_v1/oof_scores_ndcg_d4_lr003.npy"
    for path in (feature_path, label_path, score_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    features = np.load(feature_path, mmap_mode="r")
    scores = np.load(score_path, mmap_mode="r")
    with np.load(label_path, allow_pickle=False) as archive:
        labels = {name: archive[name] for name in archive.files}
    comparison = compare_harm_head(features, scores, labels, seed=40220260830)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "TRAIN_EXPLORE_NESTED_OOF_DIAGNOSTIC_ONLY",
        "source": {
            "feature_cache_sha256": metric._sha256(feature_path),
            "label_cache_sha256": metric._sha256(label_path),
            "oof_score_sha256": metric._sha256(score_path),
            "analyzer_sha256": metric._sha256(Path(__file__).resolve()),
        },
        "protocol": {
            "ranker_retrained": False,
            "target_blind_runtime_features": True,
            "harm_label": "action removes earliest eligible baseline rank-10 hit",
            "harm_multipliers": list(HARM_MULTIPLIERS),
            "outer_folds": 5,
            "inner_folds": 5,
            "held_out_splits_opened": False,
            "runtime_or_agent_started": False,
        },
        "comparison": comparison,
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "decision": {"promote": False},
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"D:\tiktok\techjam-err402-fast-track"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run(args.source_root, args.output.resolve())
    print(
        json.dumps(
            {
                "global": result["comparison"]["global"],
                "folds": result["comparison"]["folds"],
                "label_counts": {
                    "rescue": result["comparison"]["rescue_label_rows"],
                    "first_hit_harm": result["comparison"]["harm_label_rows"],
                    "direct_risk": result["comparison"]["direct_risk_rows"],
                },
                "timing_seconds": result["timing_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
