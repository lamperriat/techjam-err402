"""Freeze the reproduced fold-safe OOF policy as one deployable artifact.

This script reads only frozen train_explore caches and model outputs.  Targets
are used only to train the two admission heads and to audit nested OOF; they are
never serialized into the deployable artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_small_ranker_metric_gate as metric  # noqa: E402
from scripts import analyze_small_ranker_rr_regret_gate as rr  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-fold-safe-artifact.v1"
RESULT_SCHEMA_VERSION = "small-ranker-fold-safe-artifact-freeze-result.v1"
PREREGISTRATION = ROOT / "configs/small_ranker_v1_9.artifact_export_preregistration.json"
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
QUANTILES = tuple(float(value) / 64.0 for value in range(64))
KEEP_QUANTILE = 1.0
RR_MULTIPLIER = 1.0
EXPECTED_HASHES = {
    "features": "2b19835a1bced7f21322610296c712e3d06d915274719e11c268d31f7f596089",
    "labels": "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb",
    "projected_features": "cd9b075b923c31afe10b9a9c6d720de22e221c93de961595ff484ea4f532b90a",
    "projected_oof_scores": "5000deb9b77b3e7b326ccab6455222b291d2ec859ddab2043fe67d23a3217c5e",
    "full_ranker": "6730ad952c5a0fa8e4bdb33864ec84690cb3ac342a8c899edaae16c8481eff40",
}
FORBIDDEN_ARTIFACT_KEYS = {
    "asin",
    "best_rank",
    "eligible_from",
    "evaluator_label",
    "future_interaction",
    "positive_index",
    "target",
    "target_id",
    "user_id",
}
ASIN_SHAPE_RE = re.compile(
    rb"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE
)
SEMANTIC_OFF_FEATURE_NAMES = (
    "route_rank_mean",
    "route_rank_min",
    "route_rank_max",
    "route_rank_dispersion",
    "top10_route_agreement_fraction",
    "mean_top10_route_jaccard",
    "top10_vote_entropy",
    "p11_semantic_top10_jaccard",
    "semantic_presence",
    "semantic_rank_fraction",
    "semantic_reciprocal_rank",
    "semantic_incumbent_rr_margin",
)


class ArtifactFreezeError(RuntimeError):
    pass


@dataclass(frozen=True)
class FreezeInputs:
    source_features: np.ndarray
    projected_features: np.ndarray
    oof_scores: np.ndarray
    labels: Mapping[str, np.ndarray]
    feature_path: Path
    label_path: Path
    projected_path: Path
    score_path: Path
    ranker_path: Path


@dataclass(frozen=True)
class ActionSurface:
    incumbent: np.ndarray
    chosen: np.ndarray
    action: np.ndarray
    gate_features: np.ndarray
    rescue: np.ndarray
    rescue_weights: np.ndarray
    regret: np.ndarray
    regret_weights: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()


def _require_output_below_root(path: Path) -> None:
    if path == ROOT or ROOT not in path.parents:
        raise ArtifactFreezeError("output directory must be below this worktree")


def _feature_order_hash(names: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_bytes(list(names))).hexdigest()


def _load_inputs(source_root: Path, projection_root: Path) -> FreezeInputs:
    source_root = source_root.resolve()
    projection_root = projection_root.resolve()
    paths = {
        "features": source_root / "experiments/fast_track/small_ranker_v1/features.npy",
        "labels": source_root / "experiments/fast_track/small_ranker_v1/labels_v2.npz",
        "projected_features": projection_root
        / "experiments/fast_track/small_ranker_fold_safe_projected_features.npy",
        "projected_oof_scores": source_root
        / "experiments/fast_track/small_ranker_v1/oof_batch_v1/oof_scores_runtime_projection_no_semantic.npy",
        "full_ranker": source_root
        / "experiments/fast_track/small_ranker_v1/oof_batch_v1/research_runtime_v1.xgb.json",
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ArtifactFreezeError(f"frozen input unavailable: {name}")
        if _sha256(path) != EXPECTED_HASHES[name]:
            raise ArtifactFreezeError(f"frozen input hash mismatch: {name}")
    source_features = np.load(paths["features"], mmap_mode="r")
    projected = np.load(paths["projected_features"], mmap_mode="r")
    scores = np.load(paths["projected_oof_scores"], mmap_mode="r")
    with np.load(paths["labels"], allow_pickle=False) as archive:
        labels = {name: archive[name] for name in archive.files}
    expected_feature_shape = (
        base.SESSION_COUNT,
        base.TURN_COUNT,
        base.CANDIDATE_COUNT,
        base.FEATURE_COUNT,
    )
    if (
        source_features.shape != expected_feature_shape
        or source_features.dtype != np.float32
    ):
        raise ArtifactFreezeError("source feature tensor schema mismatch")
    if projected.shape != expected_feature_shape or projected.dtype != np.float32:
        raise ArtifactFreezeError("projected feature tensor schema mismatch")
    if scores.shape != (
        base.SESSION_COUNT,
        base.TURN_COUNT,
        base.CANDIDATE_COUNT,
    ) or scores.dtype != np.float32:
        raise ArtifactFreezeError("projected OOF score schema mismatch")
    return FreezeInputs(
        source_features,
        projected,
        scores,
        labels,
        paths["features"],
        paths["labels"],
        paths["projected_features"],
        paths["projected_oof_scores"],
        paths["full_ranker"],
    )


def _action_surface(
    features: np.ndarray,
    scores: np.ndarray,
    labels: Mapping[str, np.ndarray],
) -> ActionSurface:
    incumbent = base._incumbent_indices(features)
    chosen, margin, top_gap = base.choose_slot10(scores, incumbent)
    gate_features = base.gate_feature_matrix(
        features, scores, chosen, incumbent, margin, top_gap
    )
    rescue, _direct_risk, rescue_weights = base.action_training_labels(
        labels, chosen, incumbent
    )
    rr_regret = rr.single_action_rr_regret(labels, chosen, incumbent)
    regret = (rr_regret > 0).astype(np.uint8)
    regret_weights = np.where(
        regret > 0,
        5.0 + 20.0 * rr_regret,
        np.where(rescue > 0, 0.2, 0.05),
    ).astype(np.float64)
    return ActionSurface(
        incumbent,
        chosen,
        chosen != incumbent,
        gate_features,
        rescue,
        rescue_weights,
        regret,
        regret_weights,
    )


def _fit_predict(
    x: np.ndarray,
    train_rows: np.ndarray,
    predict_rows: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> np.ndarray:
    model, mean, scale = base._fit_gate_model(
        x[train_rows], target[train_rows], weights[train_rows], seed
    )
    return base._predict_gate(model, mean, scale, x[predict_rows]).astype(
        np.float32
    )


def _threshold_at_quantile(values: np.ndarray, quantile: float) -> float:
    if quantile >= KEEP_QUANTILE:
        return math.inf
    if not len(values):
        raise ArtifactFreezeError("cannot map a quantile over an empty action set")
    return float(np.quantile(values, quantile, method="higher"))


def _deployable_threshold_at_quantile(
    values: np.ndarray, quantile: float
) -> tuple[float, float]:
    """Place the serialized threshold inside the quantile decision gap."""

    raw_threshold = _threshold_at_quantile(values, quantile)
    if math.isinf(raw_threshold):
        return raw_threshold, raw_threshold
    intended = values >= raw_threshold
    if not np.any(intended):
        raise ArtifactFreezeError("finite quantile selected no action rows")
    lowest_active = float(np.min(values[intended]))
    if np.all(intended):
        threshold = float(np.nextafter(lowest_active, -math.inf))
    else:
        highest_inactive = float(np.max(values[~intended]))
        if highest_inactive >= lowest_active:
            raise ArtifactFreezeError("quantile decision gap is not separable")
        threshold = highest_inactive + (
            lowest_active - highest_inactive
        ) / 2.0
    if not np.array_equal(intended, values >= threshold):
        raise ArtifactFreezeError("stable quantile threshold changed membership")
    return raw_threshold, threshold


def _quantile_choice_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["technical_score_delta"]),
        int(row["net_hits"]),
        sum(int(value > 0) for value in row["inner_fold_net_hits"]),
        -int(row["activation_turns"]),
        float(row["quantile"]),
    )


def _select_inner_quantile(
    utility: np.ndarray,
    surface: ActionSurface,
    labels: Mapping[str, np.ndarray],
    session_mask: np.ndarray,
    inner_fold: np.ndarray,
) -> dict[str, Any]:
    values = utility[surface.action & session_mask[:, None]]
    if not len(values):
        raise ArtifactFreezeError("nested quantile action set is empty")
    zero = np.zeros_like(surface.action, dtype=bool)
    baseline_state = metric.policy_session_state(labels, surface.chosen, zero)
    candidates: list[dict[str, Any]] = []
    for quantile in (*QUANTILES, KEEP_QUANTILE):
        threshold = _threshold_at_quantile(values, quantile)
        activation = (
            surface.action & session_mask[:, None] & (utility >= threshold)
        )
        policy_state = metric.policy_session_state(
            labels, surface.chosen, activation
        )
        aggregate = metric.transition_metrics(
            baseline_state, policy_state, activation, session_mask
        )
        inner_metrics: list[dict[str, Any]] = []
        for fold in sorted(set(int(value) for value in inner_fold[session_mask])):
            fold_mask = session_mask & (inner_fold == fold)
            inner_metrics.append(
                metric.transition_metrics(
                    baseline_state, policy_state, activation, fold_mask
                )
            )
        if all(
            row["hit_to_miss"] == 0
            and row["mrr_delta"] >= 0.0
            and row["mttc_delta"] <= 0.0
            for row in (aggregate, *inner_metrics)
        ):
            candidates.append(
                {
                    "quantile": quantile,
                    "inner_threshold": threshold,
                    **aggregate,
                    "inner_fold_net_hits": [
                        int(row["net_hits"]) for row in inner_metrics
                    ],
                    "inner_fold_mrr_delta": [
                        float(row["mrr_delta"]) for row in inner_metrics
                    ],
                }
            )
    if not candidates:
        raise ArtifactFreezeError("KEEP unexpectedly failed inner-fold safety")
    return max(candidates, key=_quantile_choice_key)


def _nested_oof_policy(
    surface: ActionSurface,
    labels: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
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
            if not np.any(train_rows) or not np.any(valid_rows):
                raise ArtifactFreezeError("nested head partition is empty")
            for head in range(2):
                inner_probability[head].reshape(-1)[valid_rows] = _fit_predict(
                    flat_x,
                    train_rows,
                    valid_rows,
                    targets[head],
                    weights[head],
                    seed + head * 10_000 + outer_fold * 31 + inner_index,
                )
        inner_utility = inner_probability[0] - RR_MULTIPLIER * inner_probability[1]
        selected = _select_inner_quantile(
            inner_utility, surface, labels, train_sessions, inner
        )

        train_rows = flat_action & train_sessions[flat_session]
        held_rows = flat_action & held_sessions[flat_session]
        train_probability = [
            np.zeros_like(surface.action, dtype=np.float32) for _ in range(2)
        ]
        for head in range(2):
            model, mean, scale = base._fit_gate_model(
                flat_x[train_rows],
                targets[head][train_rows],
                weights[head][train_rows],
                seed + head * 10_000 + outer_fold * 101,
            )
            train_probability[head].reshape(-1)[train_rows] = base._predict_gate(
                model, mean, scale, flat_x[train_rows]
            ).astype(np.float32)
            held_prediction = base._predict_gate(
                model, mean, scale, flat_x[held_rows]
            ).astype(np.float32)
            if head == 0:
                rescue_probability.reshape(-1)[held_rows] = held_prediction
            else:
                regret_probability.reshape(-1)[held_rows] = held_prediction
        train_utility = train_probability[0] - RR_MULTIPLIER * train_probability[1]
        threshold = _threshold_at_quantile(
            train_utility[surface.action & train_sessions[:, None]],
            float(selected["quantile"]),
        )
        held_utility = rescue_probability - RR_MULTIPLIER * regret_probability
        activation[held_sessions] = (
            surface.action[held_sessions]
            & (held_utility[held_sessions] >= threshold)
        )
        selections.append(
            {
                "fold": outer_fold,
                "selected_quantile": float(selected["quantile"]),
                "mapped_outer_train_threshold": threshold,
                "inner_selection": selected,
                "train_rescue_rows": int(targets[0][train_rows].sum()),
                "train_rr_regret_rows": int(targets[1][train_rows].sum()),
            }
        )

    zero = np.zeros_like(surface.action, dtype=bool)
    baseline_state = metric.policy_session_state(labels, surface.chosen, zero)
    policy_state = metric.policy_session_state(labels, surface.chosen, activation)
    all_sessions = np.ones(len(outer), dtype=bool)
    folds: list[dict[str, Any]] = []
    for fold in range(base.OUTER_FOLDS):
        fold_mask = outer == fold
        folds.append(
            {
                "fold": fold,
                **metric.transition_metrics(
                    baseline_state, policy_state, activation, fold_mask
                ),
            }
        )
    return {
        "global": metric.transition_metrics(
            baseline_state, policy_state, activation, all_sessions
        ),
        "folds": folds,
        "outer_selections": selections,
        "fold_quantiles": [
            float(row["selected_quantile"]) for row in selections
        ],
        "activation_sha256": hashlib.sha256(activation.tobytes()).hexdigest(),
        "chosen_sha256": hashlib.sha256(surface.chosen.tobytes()).hexdigest(),
    }


def _oof_gate_passed(result: Mapping[str, Any]) -> bool:
    global_metrics = result["global"]
    return bool(
        float(global_metrics["policy"]["hit_rate_at_10"]) >= 0.9715
        and int(global_metrics["hit_to_miss"]) == 0
        and float(global_metrics["mrr_delta"]) >= 0.0
        and float(global_metrics["mttc_delta"]) <= 0.0
        and float(global_metrics["technical_score_delta"]) > 0.0
        and all(
            int(row["net_hits"]) >= 0
            and int(row["hit_to_miss"]) == 0
            and float(row["mrr_delta"]) >= 0.0
            and float(row["mttc_delta"]) <= 0.0
            for row in result["folds"]
        )
    )


def _export_head(
    x: np.ndarray,
    rows: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray]:
    model, mean, scale = base._fit_gate_model(
        x[rows], target[rows], weights[rows], seed
    )
    if not hasattr(model, "coef_"):
        raise ArtifactFreezeError("final admission head collapsed to a constant")
    payload = {
        "mean": [float(value) for value in mean],
        "scale": [float(value) for value in scale],
        "coef": [float(value) for value in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
    }
    return payload, model, mean, scale


def _head_probability(head: Mapping[str, Any], row: Sequence[float]) -> float:
    lengths = (
        len(row),
        len(head["mean"]),
        len(head["scale"]),
        len(head["coef"]),
    )
    if len(set(lengths)) != 1:
        raise ArtifactFreezeError("serialized admission head shape mismatch")
    logit = float(head["intercept"])
    for value, mean, scale, coefficient in zip(
        row, head["mean"], head["scale"], head["coef"]
    ):
        logit += float(coefficient) * (
            (float(value) - float(mean)) / float(scale)
        )
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _serialized_head_probabilities(
    head: Mapping[str, Any], matrix: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [_head_probability(head, row) for row in matrix], dtype=np.float64
    )


def _semantic_projection_audit(
    source_features: np.ndarray, projected_features: np.ndarray
) -> dict[str, Any]:
    sessions = np.asarray(
        [0, 1, 97, 399, 400, 799, 1200, 1599, 1998, 1999],
        dtype=np.int64,
    )
    source = np.asarray(source_features[sessions], dtype=np.float32)
    expected = base.project_semantic_route_off(source)
    observed = np.asarray(projected_features[sessions], dtype=np.float32)
    exact = bool(np.array_equal(expected, observed))
    if not exact:
        raise ArtifactFreezeError("semantic-off cached features failed projection parity")
    direct_names = (
        "p11_semantic_top10_jaccard",
        "semantic_presence",
        "semantic_rank_fraction",
        "semantic_reciprocal_rank",
        "semantic_incumbent_rr_margin",
    )
    direct_indices = [base.FEATURE_INDEX[name] for name in direct_names]
    direct = observed[..., direct_indices]
    expected_constants = np.asarray(
        [0.0, 0.0, 1.25, 0.0, 0.0], dtype=np.float32
    )
    constants_exact = bool(np.all(direct == expected_constants))
    if not constants_exact:
        raise ArtifactFreezeError("semantic-off direct feature constants mismatch")
    changed = source != observed
    changed_columns = np.flatnonzero(np.any(changed, axis=(0, 1, 2)))
    return {
        "sessions": [int(value) for value in sessions],
        "rows": int(source.shape[0] * source.shape[1] * source.shape[2]),
        "projected_cache_exact": exact,
        "direct_semantic_constants_exact": constants_exact,
        "changed_values": int(changed.sum()),
        "changed_feature_names": [
            str(base.FEATURE_NAMES[int(index)]) for index in changed_columns
        ],
    }


def _tree_semantic_dependency_audit(
    tree_model: Mapping[str, Any], projection: Mapping[str, Any]
) -> dict[str, Any]:
    changed_names = set(str(value) for value in projection["changed_feature_names"])
    changed_indices = {
        int(base.FEATURE_INDEX[name]) for name in SEMANTIC_OFF_FEATURE_NAMES
    }
    split_indices: list[int] = []
    for tree in tree_model["trees"]:
        if len(tree["l"]) != len(tree["f"]):
            raise ArtifactFreezeError("serialized tree shape mismatch")
        split_indices.extend(
            int(feature)
            for left, feature in zip(tree["l"], tree["f"])
            if int(left) >= 0
        )
    dependent = [index for index in split_indices if index in changed_indices]
    return {
        "semantic_route_required": False,
        "semantic_off_projection_required": bool(dependent),
        "changed_feature_split_count": len(dependent),
        "changed_feature_split_names": sorted(
            {str(base.FEATURE_NAMES[index]) for index in dependent}
        ),
        "audit_sample_observed_changed_features": sorted(changed_names),
    }


def _artifact_key_scan(value: object) -> list[str]:
    matches: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in FORBIDDEN_ARTIFACT_KEYS:
                matches.append(lowered)
            matches.extend(_artifact_key_scan(item))
    elif isinstance(value, list):
        for item in value:
            matches.extend(_artifact_key_scan(item))
    return matches


def _deep_size(value: object, seen: set[int] | None = None) -> int:
    seen = seen if seen is not None else set()
    identifier = id(value)
    if identifier in seen:
        return 0
    seen.add(identifier)
    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        size += sum(
            _deep_size(key, seen) + _deep_size(item, seen)
            for key, item in value.items()
        )
    elif isinstance(value, (list, tuple)):
        size += sum(_deep_size(item, seen) for item in value)
    return size


def _predict_full_ranker(
    booster: Any,
    features: np.ndarray,
    output_path: Path,
) -> tuple[np.ndarray, float]:
    import xgboost as xgb

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    scores = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(base.SESSION_COUNT, base.TURN_COUNT, base.CANDIDATE_COUNT),
    )
    started = time.perf_counter()
    for offset in range(0, base.SESSION_COUNT, 25):
        block = np.asarray(features[offset : offset + 25], dtype=np.float32)
        prediction = booster.predict(
            xgb.DMatrix(block.reshape(-1, base.FEATURE_COUNT)),
            output_margin=True,
        )
        scores[offset : offset + len(block)] = np.asarray(
            prediction, dtype=np.float32
        ).reshape(len(block), base.TURN_COUNT, base.CANDIDATE_COUNT)
    scores.flush()
    return scores, time.perf_counter() - started


def _parity_audit(
    booster: Any,
    tree_model: Mapping[str, Any],
    projected_features: np.ndarray,
    full_gate_features: np.ndarray,
    heads: Sequence[tuple[Mapping[str, Any], Any, np.ndarray, np.ndarray]],
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    import xgboost as xgb

    rng = np.random.default_rng(base._library_seed(seed))
    group_indices = rng.choice(
        base.SESSION_COUNT * base.TURN_COUNT, size=10, replace=False
    )
    sample_rows = projected_features.reshape(
        -1, base.CANDIDATE_COUNT, base.FEATURE_COUNT
    )[group_indices]
    sample_matrix = sample_rows.reshape(-1, base.FEATURE_COUNT)
    reference_scores = np.asarray(
        booster.predict(xgb.DMatrix(sample_matrix), output_margin=True),
        dtype=np.float64,
    )
    lightweight_scores = np.asarray(
        [base.lightweight_tree_score(tree_model, row) for row in sample_matrix],
        dtype=np.float64,
    )
    reference_order = np.argsort(
        -reference_scores.reshape(10, base.CANDIDATE_COUNT),
        axis=1,
        kind="stable",
    )
    lightweight_order = np.argsort(
        -lightweight_scores.reshape(10, base.CANDIDATE_COUNT),
        axis=1,
        kind="stable",
    )
    gate_rows = rng.choice(
        base.SESSION_COUNT * base.TURN_COUNT, size=1000, replace=False
    )
    gate_matrix = full_gate_features.reshape(-1, len(base.GATE_FEATURE_NAMES))[
        gate_rows
    ]
    head_errors: list[float] = []
    pure_probabilities: list[np.ndarray] = []
    reference_probabilities: list[np.ndarray] = []
    for payload, model, mean, scale in heads:
        reference = base._predict_gate(model, mean, scale, gate_matrix)
        pure = np.asarray(
            [_head_probability(payload, row) for row in gate_matrix],
            dtype=np.float64,
        )
        reference_probabilities.append(np.asarray(reference, dtype=np.float64))
        pure_probabilities.append(pure)
        head_errors.append(float(np.max(np.abs(reference - pure))))
    reference_activation = (
        reference_probabilities[0]
        - RR_MULTIPLIER * reference_probabilities[1]
        >= threshold
    )
    pure_activation = (
        pure_probabilities[0] - RR_MULTIPLIER * pure_probabilities[1]
        >= threshold
    )
    return {
        "ranker_rows": len(sample_matrix),
        "ranker_maximum_absolute_score_error": float(
            np.max(np.abs(reference_scores - lightweight_scores))
        ),
        "ranker_full_c100_order_exact": bool(
            np.array_equal(reference_order, lightweight_order)
        ),
        "head_rows": len(gate_matrix),
        "head_maximum_absolute_probability_error": max(head_errors),
        "activation_exact": bool(
            np.array_equal(reference_activation, pure_activation)
        ),
    }


def _benchmark_tree(
    tree_model: Mapping[str, Any], projected_features: np.ndarray
) -> dict[str, Any]:
    turn_times: list[float] = []
    flat = projected_features.reshape(
        -1, base.CANDIDATE_COUNT, base.FEATURE_COUNT
    )
    for group in range(20):
        started = time.perf_counter()
        for row in flat[group]:
            base.lightweight_tree_score(tree_model, row)
        turn_times.append((time.perf_counter() - started) * 1000.0)
    session_times: list[float] = []
    for session in range(3):
        started = time.perf_counter()
        for row in projected_features[session].reshape(-1, base.FEATURE_COUNT):
            base.lightweight_tree_score(tree_model, row)
        session_times.append((time.perf_counter() - started) * 1000.0)
    batch_started = time.perf_counter()
    for row in projected_features[:10].reshape(-1, base.FEATURE_COUNT):
        base.lightweight_tree_score(tree_model, row)
    batch_seconds = time.perf_counter() - batch_started
    return {
        "scope": "pure exported tree scoring on cached feature rows; excludes feature construction and SQLite",
        "turn_rows": base.CANDIDATE_COUNT,
        "turn_p50_ms": float(np.quantile(turn_times, 0.5, method="higher")),
        "turn_p95_ms": float(np.quantile(turn_times, 0.95, method="higher")),
        "session_rows": base.TURN_COUNT * base.CANDIDATE_COUNT,
        "session_p50_ms": float(np.quantile(session_times, 0.5, method="higher")),
        "session_p95_ms": float(np.quantile(session_times, 0.95, method="higher")),
        "batch_sessions": 10,
        "batch_seconds": batch_seconds,
        "batch_rows_per_second": (
            10 * base.TURN_COUNT * base.CANDIDATE_COUNT / batch_seconds
        ),
    }


def run(
    source_root: Path, projection_root: Path, output_dir: Path
) -> dict[str, Any]:
    import sklearn
    import xgboost as xgb

    started = time.perf_counter()
    output_dir = output_dir.resolve()
    _require_output_below_root(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    result_path = output_dir / "freeze_result.json"
    artifact_path = output_dir / "small_ranker_fold_safe_v1.json"
    repeat_path = output_dir / "small_ranker_fold_safe_v1.repeat.json"
    score_path = output_dir / "full_ranker_scores.npy"
    inputs = _load_inputs(source_root, projection_root)
    projection_audit = _semantic_projection_audit(
        inputs.source_features, inputs.projected_features
    )

    oof_started = time.perf_counter()
    oof_surface = _action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    nested = _nested_oof_policy(
        oof_surface, inputs.labels, seed=40220260830
    )
    oof_seconds = time.perf_counter() - oof_started
    passed = _oof_gate_passed(nested)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": "SR-V1.9-ARTIFACT-FREEZE",
        "scope": {
            "split": "train_explore",
            "held_out_splits_opened": False,
            "agent_or_runtime_started": False,
            "ranker_retrained": False,
        },
        "sources": {
            **EXPECTED_HASHES,
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "exporter_sha256": _sha256(Path(__file__).resolve()),
        },
        "nested_oof": nested,
        "semantic_projection": projection_audit,
        "decision": {
            "oof_gate_passed": passed,
            "artifact_frozen": False,
            "served_default": "off",
            "status": "OOF_GATE_PASSED" if passed else "ARTIFACT_FREEZE_NO_GO",
        },
        "timing_seconds": {"nested_oof": round(oof_seconds, 6)},
    }
    if not passed:
        result["timing_seconds"]["total"] = round(
            time.perf_counter() - started, 6
        )
        _write_exclusive(result_path, result)
        return result

    flat_gate = oof_surface.gate_features.reshape(
        -1, len(base.GATE_FEATURE_NAMES)
    )
    flat_action = oof_surface.action.reshape(-1)
    head_started = time.perf_counter()
    rescue_head = _export_head(
        flat_gate,
        flat_action,
        oof_surface.rescue.reshape(-1),
        oof_surface.rescue_weights.reshape(-1),
        40221250830,
    )
    regret_head = _export_head(
        flat_gate,
        flat_action,
        oof_surface.regret.reshape(-1),
        oof_surface.regret_weights.reshape(-1),
        40221260830,
    )
    head_seconds = time.perf_counter() - head_started

    booster = xgb.Booster()
    booster.load_model(inputs.ranker_path)
    if int(booster.num_features()) != base.FEATURE_COUNT:
        raise ArtifactFreezeError("full ranker feature count mismatch")
    full_scores, full_score_seconds = _predict_full_ranker(
        booster, inputs.projected_features, score_path
    )
    full_incumbent = base._incumbent_indices(inputs.projected_features)
    full_chosen, full_margin, full_gap = base.choose_slot10(
        np.asarray(full_scores), full_incumbent
    )
    full_gate = base.gate_feature_matrix(
        inputs.projected_features,
        np.asarray(full_scores),
        full_chosen,
        full_incumbent,
        full_margin,
        full_gap,
    )
    full_action = full_chosen != full_incumbent
    full_gate_flat = full_gate.reshape(-1, len(base.GATE_FEATURE_NAMES))
    full_rescue_probability = base._predict_gate(
        rescue_head[1], rescue_head[2], rescue_head[3], full_gate_flat
    ).reshape(full_action.shape)
    full_regret_probability = base._predict_gate(
        regret_head[1], regret_head[2], regret_head[3], full_gate_flat
    ).reshape(full_action.shape)
    full_reference_utility = (
        full_rescue_probability - RR_MULTIPLIER * full_regret_probability
    )
    serialized_rescue_probability = _serialized_head_probabilities(
        rescue_head[0], full_gate_flat
    ).reshape(full_action.shape)
    serialized_regret_probability = _serialized_head_probabilities(
        regret_head[0], full_gate_flat
    ).reshape(full_action.shape)
    serialized_utility = (
        serialized_rescue_probability
        - RR_MULTIPLIER * serialized_regret_probability
    )
    final_quantile = float(np.median(np.asarray(nested["fold_quantiles"])))
    quantile_reference_value, final_threshold = (
        _deployable_threshold_at_quantile(
            serialized_utility[full_action], final_quantile
        )
    )
    full_activation = full_action & (serialized_utility >= final_threshold)
    reference_activation = full_action & (
        full_reference_utility >= final_threshold
    )
    full_decision_exact = bool(
        np.array_equal(full_activation, reference_activation)
    )

    tree_model, _base_score = base._export_tree_model(inputs.ranker_path)
    if len(tree_model["trees"]) != 273:
        raise ArtifactFreezeError("full ranker tree count mismatch")
    semantic_dependency = _tree_semantic_dependency_audit(
        tree_model, projection_audit
    )
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode_default": "off",
        "feature_names": list(base.FEATURE_NAMES),
        "feature_order_sha256": _feature_order_hash(base.FEATURE_NAMES),
        "feature_schema_sha256": "92795134cc11cbe496cb63b3921b585d9f71028f75b39d9c6716a13c7e6608f8",
        "runtime_projection": {"semantic_route": "missing"},
        "ranker": {
            "config_id": "ndcg_d4_lr003",
            "rounds": 273,
            "model": tree_model,
        },
        "admission": {
            "feature_names": list(base.GATE_FEATURE_NAMES),
            "feature_order_sha256": _feature_order_hash(base.GATE_FEATURE_NAMES),
            "rescue_head": rescue_head[0],
            "rr_regret_head": regret_head[0],
            "rr_multiplier": RR_MULTIPLIER,
            "activation_quantile": final_quantile,
            "quantile_reference_value": quantile_reference_value,
            "threshold": final_threshold,
        },
        "sources": {
            "target_blind_feature_cache_sha256": EXPECTED_HASHES["features"],
            "semantic_off_projection_sha256": EXPECTED_HASHES[
                "projected_features"
            ],
            "full_ranker_sha256": EXPECTED_HASHES["full_ranker"],
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "exporter_sha256": _sha256(Path(__file__).resolve()),
        },
        "fallback": {
            "policy": "P11/R08",
            "on_missing_corrupt_exception_or_budget": True,
        },
        "privacy": {
            "target_features": False,
            "future_features": False,
            "identity_features": False,
        },
    }
    forbidden_keys = sorted(set(_artifact_key_scan(artifact)))
    raw_artifact = _canonical_bytes(artifact)
    asin_matches = len(ASIN_SHAPE_RE.findall(raw_artifact))
    if forbidden_keys or asin_matches:
        raise ArtifactFreezeError("deployable artifact failed privacy scan")
    _write_exclusive(artifact_path, artifact)
    _write_exclusive(repeat_path, json.loads(artifact_path.read_text("utf-8")))
    repeat_equal = artifact_path.read_bytes() == repeat_path.read_bytes()

    parity = _parity_audit(
        booster,
        tree_model,
        inputs.projected_features,
        full_gate,
        (rescue_head, regret_head),
        final_threshold,
        seed=40221360830,
    )
    artifact_record = {
        "path": artifact_path.relative_to(ROOT).as_posix(),
        "bytes": artifact_path.stat().st_size,
        "sha256": _sha256(artifact_path),
        "repeat_path": repeat_path.relative_to(ROOT).as_posix(),
        "repeat_sha256": _sha256(repeat_path),
        "byte_identical_repeat": repeat_equal,
        "deep_size_bytes": _deep_size(artifact),
        "forbidden_key_matches": forbidden_keys,
        "asin_shape_matches": asin_matches,
    }
    full_policy_record = {
        "activation_quantile": final_quantile,
        "quantile_reference_value": quantile_reference_value,
        "mapped_threshold": final_threshold,
        "available_action_turns": int(full_action.sum()),
        "activated_turns": int(full_activation.sum()),
        "activated_sessions": int(np.any(full_activation, axis=1).sum()),
        "decision_sha256": hashlib.sha256(
            full_activation.tobytes()
        ).hexdigest(),
        "reference_decision_sha256": hashlib.sha256(
            reference_activation.tobytes()
        ).hexdigest(),
        "serialized_decision_exact": full_decision_exact,
        "head_training_action_rows": int(flat_action.sum()),
        "rescue_positive_rows": int(oof_surface.rescue.sum()),
        "rr_regret_positive_rows": int(oof_surface.regret.sum()),
    }
    parity_failures = {
        "ranker_score_error": parity[
            "ranker_maximum_absolute_score_error"
        ]
        > 2e-5,
        "ranker_order": not parity["ranker_full_c100_order_exact"],
        "head_probability_error": parity[
            "head_maximum_absolute_probability_error"
        ]
        > 1e-10,
        "sample_activation": not parity["activation_exact"],
        "full_activation": not full_decision_exact,
        "artifact_repeat": not repeat_equal,
        "artifact_size": artifact_path.stat().st_size > 8 * 1024 * 1024,
    }
    if any(parity_failures.values()):
        result.update(
            {
                "full_policy": full_policy_record,
                "artifact": artifact_record,
                "parity": parity,
                "parity_failures": parity_failures,
                "semantic_dependency": semantic_dependency,
            }
        )
        result["decision"] = {
            "oof_gate_passed": True,
            "artifact_frozen": False,
            "served_default": "off",
            "status": "ARTIFACT_PARITY_NO_GO",
        }
        result["timing_seconds"]["total"] = round(
            time.perf_counter() - started, 6
        )
        _write_exclusive(result_path, result)
        raise ArtifactFreezeError("deployable artifact parity/resource gate failed")
    benchmark = _benchmark_tree(tree_model, inputs.projected_features)
    result.update(
        {
            "full_policy": full_policy_record,
            "artifact": artifact_record,
            "full_ranker_scores": {
                "path": score_path.relative_to(ROOT).as_posix(),
                "bytes": score_path.stat().st_size,
                "sha256": _sha256(score_path),
            },
            "parity": parity,
            "semantic_dependency": semantic_dependency,
            "benchmark": benchmark,
            "versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "xgboost": xgb.__version__,
            },
        }
    )
    result["decision"] = {
        "oof_gate_passed": True,
        "artifact_frozen": True,
        "served_default": "off",
        "status": "ARTIFACT_FREEZE_PASSED",
        "open_held_out_data": False,
    }
    result["timing_seconds"].update(
        {
            "full_head_training": round(head_seconds, 6),
            "full_ranker_scoring": round(full_score_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        }
    )
    _write_exclusive(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.source_root, args.projection_root, args.output_dir)
    print(
        json.dumps(
            {
                "nested_oof": result["nested_oof"]["global"],
                "fold_quantiles": result["nested_oof"]["fold_quantiles"],
                "artifact": result.get("artifact"),
                "parity": result.get("parity"),
                "benchmark": result.get("benchmark"),
                "decision": result["decision"],
                "timing_seconds": result["timing_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["decision"]["artifact_frozen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
