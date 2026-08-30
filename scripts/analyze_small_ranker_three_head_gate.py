"""Test rescue, reciprocal-rank-regret, and hit-loss heads in nested OOF."""

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
from scripts import analyze_small_ranker_rr_regret_gate as rr  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-three-head-gate-diagnostic.v1"
RR_MULTIPLIERS = (0.5, 1.0, 2.0, 5.0)
LOSS_MULTIPLIERS = (1.0, 2.0, 5.0, 10.0)


def single_action_hit_loss(
    labels: Mapping[str, np.ndarray], chosen: np.ndarray, incumbent: np.ndarray
) -> np.ndarray:
    """Label an isolated available action that changes a baseline hit to miss."""

    action = chosen != incumbent
    zero = np.zeros_like(action, dtype=bool)
    baseline = metric.policy_session_state(labels, chosen, zero)
    loss = np.zeros_like(action, dtype=np.uint8)
    for turn in range(chosen.shape[1]):
        activation = np.zeros_like(action, dtype=bool)
        activation[:, turn] = action[:, turn]
        state = metric.policy_session_state(labels, chosen, activation)
        loss[:, turn] = (baseline["hit"] & ~state["hit"]).astype(np.uint8)
    return loss


def select_inner_fold_safe_threshold(
    utility: np.ndarray,
    action: np.ndarray,
    chosen: np.ndarray,
    labels: Mapping[str, np.ndarray],
    session_mask: np.ndarray,
    inner_fold: np.ndarray,
) -> dict[str, Any]:
    """Select only thresholds safe in aggregate and in every inner fold."""

    values = utility[action & session_mask[:, None]]
    if not len(values):
        raise ValueError("empty action set")
    thresholds = np.unique(
        np.quantile(
            values,
            np.linspace(0.0, 1.0, metric.QUANTILE_COUNT - 1),
            method="higher",
        )
    )
    thresholds = np.concatenate((thresholds, np.asarray([math.inf])))
    zero = np.zeros_like(action, dtype=bool)
    baseline_state = metric.policy_session_state(labels, chosen, zero)
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        activation = action & session_mask[:, None] & (utility >= threshold)
        policy_state = metric.policy_session_state(labels, chosen, activation)
        aggregate = metric.transition_metrics(
            baseline_state, policy_state, activation, session_mask
        )
        fold_metrics: list[dict[str, Any]] = []
        for fold in sorted(set(int(value) for value in inner_fold[session_mask])):
            fold_mask = session_mask & (inner_fold == fold)
            fold_metrics.append(
                metric.transition_metrics(
                    baseline_state, policy_state, activation, fold_mask
                )
            )
        safe = [aggregate, *fold_metrics]
        if all(
            row["hit_to_miss"] == 0
            and row["mrr_delta"] >= 0.0
            and row["mttc_delta"] <= 0.0
            for row in safe
        ):
            candidates.append(
                {
                    "threshold": float(threshold),
                    **aggregate,
                    "inner_fold_net_hits": [
                        int(row["net_hits"]) for row in fold_metrics
                    ],
                    "inner_fold_mrr_delta": [
                        float(row["mrr_delta"]) for row in fold_metrics
                    ],
                }
            )
    if not candidates:
        raise RuntimeError("KEEP fallback unexpectedly failed fold safety")
    return max(
        candidates,
        key=lambda row: (
            float(row["technical_score_delta"]),
            int(row["net_hits"]),
            sum(int(value > 0) for value in row["inner_fold_net_hits"]),
            -int(row["activation_turns"]),
            float(row["threshold"]),
        ),
    )


def _choice_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["technical_score_delta"]),
        int(row["net_hits"]),
        sum(int(value > 0) for value in row["inner_fold_net_hits"]),
        -int(row["activation_turns"]),
        float(row["threshold"]),
        float(row["loss_multiplier"]),
        float(row["rr_multiplier"]),
    )


def _fit_predict(
    flat_x: np.ndarray,
    train_rows: np.ndarray,
    predict_rows: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> np.ndarray:
    model, mean, scale = base._fit_gate_model(
        flat_x[train_rows], target[train_rows], weights[train_rows], seed
    )
    return base._predict_gate(
        model, mean, scale, flat_x[predict_rows]
    ).astype(np.float32)


def compare_three_heads(
    features: np.ndarray,
    scores: np.ndarray,
    labels: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    incumbent = base._incumbent_indices(features)
    chosen, margin, top_gap = base.choose_slot10(scores, incumbent)
    gate_features = base.gate_feature_matrix(
        features, scores, chosen, incumbent, margin, top_gap
    )
    rescue, direct_risk, rescue_weights = base.action_training_labels(
        labels, chosen, incumbent
    )
    rr_regret = rr.single_action_rr_regret(labels, chosen, incumbent)
    regret_label = (rr_regret > 0).astype(np.uint8)
    hit_loss = single_action_hit_loss(labels, chosen, incumbent)
    regret_weights = np.where(
        regret_label > 0,
        5.0 + 20.0 * rr_regret,
        np.where(rescue > 0, 0.2, 0.05),
    ).astype(np.float64)
    loss_weights = np.where(
        hit_loss > 0, 25.0, np.where(rescue > 0, 0.2, 0.05)
    ).astype(np.float64)
    action = chosen != incumbent
    outer = np.asarray(labels["outer_fold"])
    inner = np.asarray(labels["inner_fold"])
    flat_x = gate_features.reshape(-1, gate_features.shape[-1])
    flat_action = action.reshape(-1)
    flat_session = np.repeat(np.arange(len(outer)), base.TURN_COUNT)
    targets = [rescue.reshape(-1), regret_label.reshape(-1), hit_loss.reshape(-1)]
    weights = [
        rescue_weights.reshape(-1),
        regret_weights.reshape(-1),
        loss_weights.reshape(-1),
    ]
    probabilities = [
        np.zeros(action.shape, dtype=np.float32) for _ in range(3)
    ]
    thresholds = np.full(base.OUTER_FOLDS, math.inf, dtype=np.float64)
    rr_selected = np.zeros(base.OUTER_FOLDS, dtype=np.float64)
    loss_selected = np.zeros(base.OUTER_FOLDS, dtype=np.float64)
    selections: list[dict[str, Any]] = []

    for outer_fold in range(base.OUTER_FOLDS):
        train_sessions = outer != outer_fold
        held_sessions = outer == outer_fold
        inner_probability = [
            np.zeros(action.shape, dtype=np.float32) for _ in range(3)
        ]
        for inner_index in range(base.OUTER_FOLDS):
            model_train = train_sessions & (inner != inner_index)
            model_valid = train_sessions & (inner == inner_index)
            train_rows = flat_action & model_train[flat_session]
            valid_rows = flat_action & model_valid[flat_session]
            if not np.any(valid_rows):
                continue
            for head in range(3):
                inner_probability[head].reshape(-1)[valid_rows] = _fit_predict(
                    flat_x,
                    train_rows,
                    valid_rows,
                    targets[head],
                    weights[head],
                    seed + head * 10_000 + outer_fold * 31 + inner_index,
                )

        choices: list[dict[str, Any]] = []
        for rr_multiplier in RR_MULTIPLIERS:
            for loss_multiplier in LOSS_MULTIPLIERS:
                utility = (
                    inner_probability[0]
                    - rr_multiplier * inner_probability[1]
                    - loss_multiplier * inner_probability[2]
                )
                selected = select_inner_fold_safe_threshold(
                    utility,
                    action,
                    chosen,
                    labels,
                    train_sessions,
                    inner,
                )
                choices.append(
                    {
                        "rr_multiplier": rr_multiplier,
                        "loss_multiplier": loss_multiplier,
                        **selected,
                    }
                )
        choice = max(choices, key=_choice_key)
        thresholds[outer_fold] = float(choice["threshold"])
        rr_selected[outer_fold] = float(choice["rr_multiplier"])
        loss_selected[outer_fold] = float(choice["loss_multiplier"])

        train_rows = flat_action & train_sessions[flat_session]
        held_rows = flat_action & held_sessions[flat_session]
        for head in range(3):
            probabilities[head].reshape(-1)[held_rows] = _fit_predict(
                flat_x,
                train_rows,
                held_rows,
                targets[head],
                weights[head],
                seed + head * 10_000 + outer_fold * 101,
            )
        selections.append(
            {
                "fold": outer_fold,
                "selected": choice,
                "combination_count": len(choices),
                "train_rescue_rows": int(targets[0][train_rows].sum()),
                "train_rr_regret_rows": int(targets[1][train_rows].sum()),
                "train_hit_loss_rows": int(targets[2][train_rows].sum()),
            }
        )

    utility = (
        probabilities[0]
        - rr_selected[outer][:, None] * probabilities[1]
        - loss_selected[outer][:, None] * probabilities[2]
    )
    activation = action & (utility >= thresholds[outer][:, None])
    zero = np.zeros_like(action, dtype=bool)
    baseline_state = metric.policy_session_state(labels, chosen, zero)
    policy_state = metric.policy_session_state(labels, chosen, activation)
    all_sessions = np.ones(len(outer), dtype=bool)
    result: dict[str, Any] = {
        "rescue_label_rows": int(rescue.sum()),
        "rr_regret_label_rows": int(regret_label.sum()),
        "hit_loss_label_rows": int(hit_loss.sum()),
        "direct_risk_rows": int(direct_risk.sum()),
        "activated_rr_regret_rows": int(
            (activation & (regret_label > 0)).sum()
        ),
        "activated_hit_loss_rows": int((activation & (hit_loss > 0)).sum()),
        "global": metric.transition_metrics(
            baseline_state, policy_state, activation, all_sessions
        ),
        "folds": [],
        "inner_selections": selections,
    }
    for fold in range(base.OUTER_FOLDS):
        mask = outer == fold
        result["folds"].append(
            {
                "fold": fold,
                "rr_multiplier": float(rr_selected[fold]),
                "loss_multiplier": float(loss_selected[fold]),
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
    score_path = source_root / (
        "experiments/fast_track/small_ranker_v1/oof_batch_v1/"
        "oof_scores_ndcg_d4_lr003.npy"
    )
    features = np.load(feature_path, mmap_mode="r")
    scores = np.load(score_path, mmap_mode="r")
    with np.load(label_path, allow_pickle=False) as archive:
        labels = {name: archive[name] for name in archive.files}
    comparison = compare_three_heads(
        features, scores, labels, seed=40220260830
    )
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
            "heads": ["rescue", "reciprocal_rank_regret", "direct_hit_loss"],
            "inner_fold_safety_required": True,
            "rr_multipliers": list(RR_MULTIPLIERS),
            "loss_multipliers": list(LOSS_MULTIPLIERS),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"D:\tiktok\techjam-err402-fast-track"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_root, args.output.resolve())
    comparison = result["comparison"]
    print(
        json.dumps(
            {
                "global": comparison["global"],
                "folds": comparison["folds"],
                "label_counts": {
                    "rescue": comparison["rescue_label_rows"],
                    "rr_regret": comparison["rr_regret_label_rows"],
                    "hit_loss": comparison["hit_loss_label_rows"],
                    "activated_hit_loss": comparison["activated_hit_loss_rows"],
                },
                "timing_seconds": result["timing_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
