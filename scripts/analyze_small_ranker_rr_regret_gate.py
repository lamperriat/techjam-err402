"""Test a reciprocal-rank-regret admission head in nested OOF.

The prior first-hit-removal label missed the dominant MRR failure: inserting a
rank-10 target before a later, better-ranked baseline hit.  This diagnostic
labels every single-turn action whose isolated application lowers session
reciprocal rank, then learns a target-blind regret head.  Ranker scores, rich
features, and all outer folds remain frozen.
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
from scripts import analyze_small_ranker_mrr_harm_gate as prior  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-rr-regret-gate-diagnostic.v1"
REGRET_MULTIPLIERS = (0.5, 1.0, 2.0, 5.0, 10.0)


def single_action_rr_regret(
    labels: Mapping[str, np.ndarray], chosen: np.ndarray, incumbent: np.ndarray
) -> np.ndarray:
    """Return isolated reciprocal-rank loss for each available slot-10 action."""

    action = chosen != incumbent
    zero = np.zeros_like(action, dtype=bool)
    baseline = metric.policy_session_state(labels, chosen, zero)
    baseline_rr = np.where(
        baseline["hit"], 1.0 / np.maximum(baseline["first_rank"], 1), 0.0
    )
    regret = np.zeros_like(chosen, dtype=np.float32)
    for turn in range(chosen.shape[1]):
        activation = np.zeros_like(action, dtype=bool)
        activation[:, turn] = action[:, turn]
        state = metric.policy_session_state(labels, chosen, activation)
        policy_rr = np.where(
            state["hit"], 1.0 / np.maximum(state["first_rank"], 1), 0.0
        )
        regret[:, turn] = np.maximum(0.0, baseline_rr - policy_rr)
    return regret


def _choice_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["technical_score_delta"]),
        int(row["net_hits"]),
        int(row["miss_to_hit"]),
        -int(row["activation_turns"]),
        float(row["threshold"]),
        float(row["regret_multiplier"]),
    )


def compare_regret_head(
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
    rr_regret = single_action_rr_regret(labels, chosen, incumbent)
    regret_label = (rr_regret > 0).astype(np.uint8)
    regret_weights = np.where(
        regret_label > 0,
        5.0 + 20.0 * rr_regret,
        np.where(rescue > 0, 0.2, 0.05),
    ).astype(np.float64)
    action = chosen != incumbent
    outer = np.asarray(labels["outer_fold"])
    inner = np.asarray(labels["inner_fold"])
    flat_x = gate_features.reshape(-1, gate_features.shape[-1])
    flat_rescue = rescue.reshape(-1)
    flat_regret = regret_label.reshape(-1)
    flat_rescue_weights = rescue_weights.reshape(-1)
    flat_regret_weights = regret_weights.reshape(-1)
    flat_action = action.reshape(-1)
    flat_session = np.repeat(np.arange(len(outer)), base.TURN_COUNT)
    rescue_probability = np.zeros(action.shape, dtype=np.float32)
    regret_probability = np.zeros(action.shape, dtype=np.float32)
    thresholds = np.full(base.OUTER_FOLDS, math.inf, dtype=np.float64)
    multipliers = np.zeros(base.OUTER_FOLDS, dtype=np.float64)
    selections: list[dict[str, Any]] = []

    for outer_fold in range(base.OUTER_FOLDS):
        train_sessions = outer != outer_fold
        held_sessions = outer == outer_fold
        inner_rescue = np.zeros(action.shape, dtype=np.float32)
        inner_regret = np.zeros(action.shape, dtype=np.float32)
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
            regret_model, regret_mean, regret_scale = base._fit_gate_model(
                flat_x[train_rows],
                flat_regret[train_rows],
                flat_regret_weights[train_rows],
                seed + 10_000 + outer_fold * 31 + inner_fold,
            )
            inner_rescue.reshape(-1)[valid_rows] = base._predict_gate(
                rescue_model, rescue_mean, rescue_scale, flat_x[valid_rows]
            ).astype(np.float32)
            inner_regret.reshape(-1)[valid_rows] = base._predict_gate(
                regret_model, regret_mean, regret_scale, flat_x[valid_rows]
            ).astype(np.float32)

        choices: list[dict[str, Any]] = []
        for multiplier in REGRET_MULTIPLIERS:
            utility = inner_rescue - multiplier * inner_regret
            selected = metric.select_metric_safe_threshold(
                utility, action, chosen, labels, train_sessions
            )
            choices.append({"regret_multiplier": multiplier, **selected})
        choice = max(choices, key=_choice_key)
        thresholds[outer_fold] = float(choice["threshold"])
        multipliers[outer_fold] = float(choice["regret_multiplier"])

        train_rows = flat_action & train_sessions[flat_session]
        held_rows = flat_action & held_sessions[flat_session]
        rescue_model, rescue_mean, rescue_scale = base._fit_gate_model(
            flat_x[train_rows],
            flat_rescue[train_rows],
            flat_rescue_weights[train_rows],
            seed + outer_fold * 101,
        )
        regret_model, regret_mean, regret_scale = base._fit_gate_model(
            flat_x[train_rows],
            flat_regret[train_rows],
            flat_regret_weights[train_rows],
            seed + 10_000 + outer_fold * 101,
        )
        rescue_probability.reshape(-1)[held_rows] = base._predict_gate(
            rescue_model, rescue_mean, rescue_scale, flat_x[held_rows]
        ).astype(np.float32)
        regret_probability.reshape(-1)[held_rows] = base._predict_gate(
            regret_model, regret_mean, regret_scale, flat_x[held_rows]
        ).astype(np.float32)
        selections.append(
            {
                "fold": outer_fold,
                "selected": choice,
                "multiplier_comparison": choices,
                "train_rescue_rows": int(flat_rescue[train_rows].sum()),
                "train_rr_regret_rows": int(flat_regret[train_rows].sum()),
                "train_direct_risk_rows": int(
                    direct_risk.reshape(-1)[train_rows].sum()
                ),
            }
        )

    utility = rescue_probability - multipliers[outer][:, None] * regret_probability
    activation = action & (utility >= thresholds[outer][:, None])
    zero = np.zeros_like(action, dtype=bool)
    baseline_state = metric.policy_session_state(labels, chosen, zero)
    policy_state = metric.policy_session_state(labels, chosen, activation)
    all_sessions = np.ones(len(outer), dtype=bool)
    result: dict[str, Any] = {
        "rr_regret_label_rows": int(regret_label.sum()),
        "rr_regret_magnitude_sum": round(float(rr_regret.sum()), 6),
        "direct_risk_rows": int(direct_risk.sum()),
        "rescue_label_rows": int(rescue.sum()),
        "global": metric.transition_metrics(
            baseline_state, policy_state, activation, all_sessions
        ),
        "folds": [],
        "inner_selections": selections,
        "activated_rr_regret_rows": int(
            (activation & (regret_label > 0)).sum()
        ),
        "probability_ranges": {
            "rescue_min": round(float(rescue_probability[action].min()), 8),
            "rescue_max": round(float(rescue_probability[action].max()), 8),
            "regret_min": round(float(regret_probability[action].min()), 8),
            "regret_max": round(float(regret_probability[action].max()), 8),
        },
    }
    for fold in range(base.OUTER_FOLDS):
        mask = outer == fold
        result["folds"].append(
            {
                "fold": fold,
                "regret_multiplier": float(multipliers[fold]),
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
    for path in (feature_path, label_path, score_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    features = np.load(feature_path, mmap_mode="r")
    scores = np.load(score_path, mmap_mode="r")
    with np.load(label_path, allow_pickle=False) as archive:
        labels = {name: archive[name] for name in archive.files}
    comparison = compare_regret_head(
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
            "regret_label": "isolated action lowers session reciprocal rank",
            "regret_multipliers": list(REGRET_MULTIPLIERS),
            "outer_folds": 5,
            "inner_folds": 5,
            "held_out_splits_opened": False,
            "runtime_or_agent_started": False,
            "known_calibration_reused": False,
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
                    "direct_risk": comparison["direct_risk_rows"],
                    "activated_rr_regret": comparison["activated_rr_regret_rows"],
                },
                "timing_seconds": result["timing_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
