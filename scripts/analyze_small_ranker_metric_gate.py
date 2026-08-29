"""Bounded nested-OOF diagnostic for an MRR/MTTC-safe admission gate.

The v1 gate selected thresholds using session hit-to-miss only.  This script
reuses the frozen ranker scores and target-blind gate features, but selects each
outer fold's threshold on inner OOF with additional MRR and MTTC constraints.
It opens train_explore numeric caches only and writes a research-only result.
"""

from __future__ import annotations

import argparse
import hashlib
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

from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-metric-gate-diagnostic.v1"
QUANTILE_COUNT = 65


class MetricGateError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def policy_session_state(
    labels: Mapping[str, np.ndarray],
    chosen: np.ndarray,
    activation: np.ndarray,
) -> dict[str, np.ndarray]:
    baseline_rank = np.asarray(labels["baseline_rank"])
    positive = np.asarray(labels["positive_index"])
    eligible_from = np.asarray(labels["eligible_from"])
    if chosen.shape != baseline_rank.shape or activation.shape != baseline_rank.shape:
        raise MetricGateError("policy arrays do not match label shape")
    eligible = np.arange(1, baseline_rank.shape[1] + 1)[None, :] >= eligible_from[:, None]
    protected = (baseline_rank >= 1) & (baseline_rank <= 9)
    rank = np.where(
        protected,
        baseline_rank,
        np.where(
            ~activation,
            baseline_rank,
            np.where((positive >= 0) & (chosen == positive), 10, 0),
        ),
    )
    rank = np.where(eligible, rank, 0)
    hit_turn = rank > 0
    hit = hit_turn.any(axis=1)
    first_index = np.argmax(hit_turn, axis=1)
    first_turn = np.where(hit, first_index + 1, 11)
    first_rank = np.take_along_axis(rank, first_index[:, None], axis=1)[:, 0]
    return {
        "hit": hit,
        "first_rank": first_rank.astype(np.int16),
        "first_turn": first_turn.astype(np.int16),
    }


def official_metrics(state: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(mask, dtype=bool)
    hit = np.asarray(state["hit"])[selected]
    rank = np.asarray(state["first_rank"])[selected]
    turn = np.asarray(state["first_turn"])[selected]
    count = int(hit.size)
    if not count:
        raise MetricGateError("official metric mask is empty")
    hr = float(hit.mean())
    mrr = float(np.where(hit, 1.0 / np.maximum(rank, 1), 0.0).mean())
    mttc = float(np.where(hit, turn, 11).mean())
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "sample_count": count,
        "hit_rate_at_10": round(hr, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(0.5 * hr + 0.3 * mrr + 0.2 * efficiency, 6),
    }


def transition_metrics(
    baseline_state: Mapping[str, np.ndarray],
    policy_state: Mapping[str, np.ndarray],
    activation: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    baseline_hit = np.asarray(baseline_state["hit"])
    policy_hit = np.asarray(policy_state["hit"])
    miss_to_hit = int(np.sum(selected & ~baseline_hit & policy_hit))
    hit_to_miss = int(np.sum(selected & baseline_hit & ~policy_hit))
    baseline = official_metrics(baseline_state, selected)
    policy = official_metrics(policy_state, selected)
    return {
        "miss_to_hit": miss_to_hit,
        "hit_to_miss": hit_to_miss,
        "net_hits": miss_to_hit - hit_to_miss,
        "activation_turns": int(activation[selected].sum()),
        "activation_sessions": int(np.any(activation[selected], axis=1).sum()),
        "hr_delta": round(float(policy["hit_rate_at_10"]) - float(baseline["hit_rate_at_10"]), 6),
        "mrr_delta": round(float(policy["mrr"]) - float(baseline["mrr"]), 6),
        "mttc_delta": round(float(policy["mttc"]) - float(baseline["mttc"]), 6),
        "technical_score_delta": round(
            float(policy["technical_score"]) - float(baseline["technical_score"]), 6
        ),
        "baseline": baseline,
        "policy": policy,
    }


def select_metric_safe_threshold(
    probabilities: np.ndarray,
    action_available: np.ndarray,
    chosen: np.ndarray,
    labels: Mapping[str, np.ndarray],
    session_mask: np.ndarray,
) -> dict[str, Any]:
    values = probabilities[action_available & session_mask[:, None]]
    if not len(values):
        return {"threshold": math.inf, "miss_to_hit": 0, "hit_to_miss": 0}
    thresholds = np.unique(
        np.quantile(values, np.linspace(0.0, 1.0, QUANTILE_COUNT - 1), method="higher")
    )
    thresholds = np.concatenate((thresholds, np.asarray([math.inf])))
    zero = np.zeros_like(action_available, dtype=bool)
    baseline_state = policy_session_state(labels, chosen, zero)
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        activation = action_available & session_mask[:, None] & (probabilities >= threshold)
        policy_state = policy_session_state(labels, chosen, activation)
        metrics = transition_metrics(
            baseline_state, policy_state, activation, session_mask
        )
        if (
            metrics["hit_to_miss"] == 0
            and metrics["mrr_delta"] >= 0.0
            and metrics["mttc_delta"] <= 0.0
        ):
            candidates.append({"threshold": float(threshold), **metrics})
    if not candidates:
        raise MetricGateError("KEEP fallback unexpectedly failed metric constraints")
    return max(
        candidates,
        key=lambda row: (
            float(row["technical_score_delta"]),
            int(row["net_hits"]),
            int(row["miss_to_hit"]),
            -int(row["activation_turns"]),
            float(row["threshold"]),
        ),
    )


def compare_nested_gates(
    features: np.ndarray,
    scores: np.ndarray,
    labels: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    incumbent = base._incumbent_indices(features)
    chosen, margin, top_gap = base.choose_slot10(scores, incumbent)
    gate_features = base.gate_feature_matrix(features, scores, chosen, incumbent, margin, top_gap)
    rescue, direct_risk, weights = base.action_training_labels(labels, chosen, incumbent)
    action = chosen != incumbent
    outer = np.asarray(labels["outer_fold"])
    inner = np.asarray(labels["inner_fold"])
    flat_x = gate_features.reshape(-1, gate_features.shape[-1])
    flat_rescue = rescue.reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_action = action.reshape(-1)
    flat_session = np.repeat(np.arange(len(outer)), base.TURN_COUNT)
    probabilities = np.zeros(action.shape, dtype=np.float32)
    current_thresholds = np.full(base.OUTER_FOLDS, math.inf, dtype=np.float64)
    metric_thresholds = np.full(base.OUTER_FOLDS, math.inf, dtype=np.float64)
    selections: list[dict[str, Any]] = []
    for outer_fold in range(base.OUTER_FOLDS):
        train_sessions = outer != outer_fold
        held_sessions = outer == outer_fold
        inner_probabilities = np.zeros(action.shape, dtype=np.float32)
        for inner_fold in range(base.OUTER_FOLDS):
            model_train = train_sessions & (inner != inner_fold)
            model_valid = train_sessions & (inner == inner_fold)
            train_rows = flat_action & model_train[flat_session]
            valid_rows = flat_action & model_valid[flat_session]
            if not np.any(valid_rows):
                continue
            model, mean, scale = base._fit_gate_model(
                flat_x[train_rows],
                flat_rescue[train_rows],
                flat_weights[train_rows],
                seed + outer_fold * 17 + inner_fold,
            )
            inner_probabilities.reshape(-1)[valid_rows] = base._predict_gate(
                model, mean, scale, flat_x[valid_rows]
            ).astype(np.float32)
        current = base.select_zero_harm_threshold(
            inner_probabilities, action, chosen, labels, train_sessions
        )
        metric = select_metric_safe_threshold(
            inner_probabilities, action, chosen, labels, train_sessions
        )
        current_thresholds[outer_fold] = float(current["threshold"])
        metric_thresholds[outer_fold] = float(metric["threshold"])
        train_rows = flat_action & train_sessions[flat_session]
        held_rows = flat_action & held_sessions[flat_session]
        model, mean, scale = base._fit_gate_model(
            flat_x[train_rows],
            flat_rescue[train_rows],
            flat_weights[train_rows],
            seed + outer_fold * 101,
        )
        probabilities.reshape(-1)[held_rows] = base._predict_gate(
            model, mean, scale, flat_x[held_rows]
        ).astype(np.float32)
        selections.append(
            {
                "fold": outer_fold,
                "current_inner_selection": current,
                "metric_safe_inner_selection": metric,
                "gate_train_rescue_rows": int(flat_rescue[train_rows].sum()),
                "gate_train_direct_risk_rows": int(direct_risk.reshape(-1)[train_rows].sum()),
            }
        )
    current_activation = action & (probabilities >= current_thresholds[outer][:, None])
    metric_activation = action & (probabilities >= metric_thresholds[outer][:, None])
    zero = np.zeros_like(action, dtype=bool)
    baseline_state = policy_session_state(labels, chosen, zero)
    all_sessions = np.ones(len(outer), dtype=bool)
    result: dict[str, Any] = {
        "current_gate": transition_metrics(
            baseline_state,
            policy_session_state(labels, chosen, current_activation),
            current_activation,
            all_sessions,
        ),
        "metric_safe_gate": transition_metrics(
            baseline_state,
            policy_session_state(labels, chosen, metric_activation),
            metric_activation,
            all_sessions,
        ),
        "folds": [],
        "inner_selections": selections,
    }
    for fold in range(base.OUTER_FOLDS):
        mask = outer == fold
        result["folds"].append(
            {
                "fold": fold,
                "current_gate": transition_metrics(
                    baseline_state,
                    policy_session_state(labels, chosen, current_activation),
                    current_activation,
                    mask,
                ),
                "metric_safe_gate": transition_metrics(
                    baseline_state,
                    policy_session_state(labels, chosen, metric_activation),
                    metric_activation,
                    mask,
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
    comparison = compare_nested_gates(
        features, scores, labels, seed=40220260830
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "TRAIN_EXPLORE_NESTED_OOF_DIAGNOSTIC_ONLY",
        "source": {
            "feature_cache_sha256": _sha256(feature_path),
            "label_cache_sha256": _sha256(label_path),
            "oof_score_sha256": _sha256(score_path),
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
        },
        "protocol": {
            "ranker": "frozen ndcg_d4_lr003 outer-OOF scores",
            "gate_models": "same nested target-blind logistic gate",
            "current_threshold_rule": "zero hit-to-miss then max miss-to-hit",
            "challenger_threshold_rule": "zero hit-to-miss, nonnegative MRR delta, nonpositive MTTC delta, then max TechnicalScore",
            "outer_folds": 5,
            "inner_folds": 5,
            "threshold_quantiles": QUANTILE_COUNT,
            "held_out_splits_opened": False,
            "runtime_or_agent_started": False,
        },
        "comparison": comparison,
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "decision": {
            "promote": False,
            "next": "retain only if nested OOF preserves positive net rescue while removing MRR and MTTC regression",
        },
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
                "current_gate": result["comparison"]["current_gate"],
                "metric_safe_gate": result["comparison"]["metric_safe_gate"],
                "timing_seconds": result["timing_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
