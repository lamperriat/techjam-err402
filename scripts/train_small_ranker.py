"""Train the single preregistered grouped-OOF small-ranker v1 batch.

This script consumes only the sealed numeric cache and its tracked manifest.
It never imports the evaluator or Agent, and it refuses every non-train split.
XGBoost and scikit-learn are research-only dependencies; no artifact is copied
to ``starter/`` unless the separate OOF and calibration gates later pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/small_ranker_v1.json"
_FROZEN_CACHE_MANIFEST = json.loads(
    (ROOT / "configs/small_ranker_v1.cache.manifest.json").read_text(encoding="utf-8")
)
FEATURE_NAMES = tuple(str(name) for name in _FROZEN_CACHE_MANIFEST["feature_cache"]["feature_names"])
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
SESSION_COUNT = 2_000
TURN_COUNT = 10
CANDIDATE_COUNT = 100
FEATURE_COUNT = len(FEATURE_NAMES)
OUTER_FOLDS = 5
SCHEMA_VERSION = "small-ranker-oof-results.v1"
ASIN_SHAPE_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE)
GATE_FEATURE_NAMES = (
    "ranker_margin",
    "ranker_top_gap",
    "challenger_score",
    "incumbent_score",
    "coverage_rank_fraction",
    "broad_presence",
    "strict_presence",
    "structured_presence",
    "semantic_presence",
    "top10_route_agreement_fraction",
    "route_rank_dispersion",
    "active_title_category_idf_coverage",
    "active_features_details_idf_coverage",
    "active_description_store_idf_coverage",
    "active_token_recall",
    "active_rare_term_coverage",
    "active_bigram_coverage",
    "hard_clause_coverage",
    "explicit_negative_violation",
    "missing_positive_evidence_fraction",
    "query_specificity_fraction",
    "goal_age_fraction",
    "challenger_minus_incumbent_active_recall",
    "challenger_minus_incumbent_hard_coverage",
    "challenger_minus_incumbent_conflict_sum",
)
GATE_STATIC_FEATURES = (
    "coverage_rank_fraction",
    "broad_presence",
    "strict_presence",
    "structured_presence",
    "semantic_presence",
    "top10_route_agreement_fraction",
    "route_rank_dispersion",
    "active_title_category_idf_coverage",
    "active_features_details_idf_coverage",
    "active_description_store_idf_coverage",
    "active_token_recall",
    "active_rare_term_coverage",
    "active_bigram_coverage",
    "hard_clause_coverage",
    "explicit_negative_violation",
    "missing_positive_evidence_fraction",
    "query_specificity_fraction",
    "goal_age_fraction",
)
CONSTRAINT_SLOTS = (
    "category", "material", "color", "size", "style", "brand", "price", "feature", "use_case"
)


class SmallRankerTrainingError(RuntimeError):
    pass


def _library_seed(value: int) -> int:
    return int(value) % (2**32 - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _identity_shape_scan(path: Path) -> int:
    pattern = re.compile(rb"B0[A-Z0-9]{8}", re.IGNORECASE)
    matches = 0
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            payload = overlap + chunk
            matches += len(pattern.findall(payload))
            overlap = payload[-9:]
    return matches


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SmallRankerTrainingError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class CacheInputs:
    config: Mapping[str, Any]
    manifest: Mapping[str, Any]
    features: np.ndarray
    labels: Mapping[str, np.ndarray]
    output_dir: Path


def load_cache(config_path: Path) -> CacheInputs:
    config_path = config_path.resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != "small-ranker.v1" or config.get("split") != "train_explore":
        raise SmallRankerTrainingError("only the frozen train_explore config is permitted")
    model_configs = config.get("training", {}).get("configs")
    if not isinstance(model_configs, list) or not 1 <= len(model_configs) <= 6:
        raise SmallRankerTrainingError("one to six preregistered model configs required")
    manifest_path = (ROOT / str(config["cache"]["manifest"])).resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != "small-ranker-cache-manifest.v1" or manifest.get("split") != "train_explore":
        raise SmallRankerTrainingError("cache manifest schema/split mismatch")
    if manifest.get("config", {}).get("sha256") != _sha256(config_path):
        raise SmallRankerTrainingError("config changed after cache construction")
    feature_path = (ROOT / str(manifest["feature_cache"]["path"])).resolve()
    label_path = (ROOT / str(manifest["label_cache"]["path"])).resolve()
    for path, expected in (
        (feature_path, manifest["feature_cache"]["sha256"]),
        (label_path, manifest["label_cache"]["sha256"]),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise SmallRankerTrainingError(f"sealed cache hash mismatch: {path.name}")
    features = np.load(feature_path, mmap_mode="r")
    labels_npz = np.load(label_path)
    labels = {name: labels_npz[name] for name in labels_npz.files}
    expected_shape = (SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT, FEATURE_COUNT)
    if features.shape != expected_shape or features.dtype != np.dtype("float32"):
        raise SmallRankerTrainingError("rich feature cache shape/dtype mismatch")
    if labels["positive_index"].shape != (SESSION_COUNT, TURN_COUNT):
        raise SmallRankerTrainingError("label cache query shape mismatch")
    for family in np.unique(labels["family_index"]):
        mask = labels["family_index"] == family
        if len(np.unique(labels["outer_fold"][mask])) != 1:
            raise SmallRankerTrainingError("product family crosses outer folds")
    output_dir = (ROOT / config["cache"]["directory"] / "oof_batch_v1").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return CacheInputs(config, manifest, features, labels, output_dir)


@dataclass(frozen=True)
class SelectedTrainingRows:
    x: np.ndarray
    y: np.ndarray
    qid: np.ndarray
    session: np.ndarray
    turn: np.ndarray
    group_count: int


def validate_grouped_qid(qid: np.ndarray, y: np.ndarray) -> int:
    if qid.ndim != 1 or y.ndim != 1 or len(qid) != len(y) or not len(qid):
        raise SmallRankerTrainingError("qid/label vectors are invalid")
    if np.any(np.diff(qid) < 0):
        raise SmallRankerTrainingError("qid rows are not monotonic")
    boundaries = np.r_[0, np.flatnonzero(np.diff(qid) != 0) + 1, len(qid)]
    sums = np.add.reduceat(y, boundaries[:-1])
    if not np.allclose(sums, 1.0):
        raise SmallRankerTrainingError("each qid must contain exactly one positive")
    return len(boundaries) - 1


def build_selected_training_rows(features: np.ndarray, labels: Mapping[str, np.ndarray]) -> SelectedTrainingRows:
    lengths = np.asarray(labels["training_length"], dtype=np.int64)
    indices = np.asarray(labels["training_indices"], dtype=np.int64)
    positives = np.asarray(labels["positive_index"], dtype=np.int64)
    row_count = int(lengths.sum())
    x = np.empty((row_count, FEATURE_COUNT), dtype=np.float32)
    y = np.zeros(row_count, dtype=np.float32)
    qid = np.empty(row_count, dtype=np.int32)
    session_rows = np.empty(row_count, dtype=np.int16)
    turn_rows = np.empty(row_count, dtype=np.uint8)
    cursor = 0
    trainable_groups = 0
    for session in range(SESSION_COUNT):
        for turn in range(TURN_COUNT):
            length = int(lengths[session, turn])
            if not length:
                continue
            selected = indices[session, turn, :length]
            if np.any(selected < 0) or len(np.unique(selected)) != length:
                raise SmallRankerTrainingError("hard-negative index group is invalid")
            end = cursor + length
            x[cursor:end] = features[session, turn, selected]
            y[cursor:end] = (selected == positives[session, turn]).astype(np.float32)
            if int(y[cursor:end].sum()) != 1:
                raise SmallRankerTrainingError("query group must have exactly one positive")
            group = session * TURN_COUNT + turn
            qid[cursor:end] = group
            session_rows[cursor:end] = session
            turn_rows[cursor:end] = turn
            cursor = end
            trainable_groups += 1
    if cursor != row_count:
        raise SmallRankerTrainingError("selected rows are not contiguous by qid")
    validated_groups = validate_grouped_qid(qid, y)
    if validated_groups != trainable_groups:
        raise SmallRankerTrainingError("qid group count drifted")
    return SelectedTrainingRows(x, y, qid, session_rows, turn_rows, trainable_groups)


def _incumbent_indices(features: np.ndarray) -> np.ndarray:
    presence = np.asarray(features[..., FEATURE_INDEX["p11_presence"]]) > 0.5
    rank = np.asarray(features[..., FEATURE_INDEX["p11_rank_fraction"]])
    candidates = presence & np.isclose(rank, 1.0, atol=1e-6)
    counts = candidates.sum(axis=2)
    if not np.all(counts == 1):
        raise SmallRankerTrainingError("each query must expose one P11 slot10 incumbent")
    return np.argmax(candidates, axis=2).astype(np.uint8)


def choose_slot10(scores: np.ndarray, incumbent_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Protect coverage Top10 except the true P11 rank10 incumbent."""

    if scores.shape != (SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT):
        raise SmallRankerTrainingError("OOF score shape mismatch")
    allowed = np.asarray(scores, dtype=np.float32).copy()
    allowed[:, :, :10] = -np.inf
    sessions = np.arange(SESSION_COUNT)[:, None]
    turns = np.arange(TURN_COUNT)[None, :]
    incumbent_scores = scores[sessions, turns, incumbent_index]
    allowed[sessions, turns, incumbent_index] = incumbent_scores
    chosen = np.argmax(allowed, axis=2).astype(np.uint8)
    chosen_scores = allowed[sessions, turns, chosen]
    second = np.partition(allowed, -2, axis=2)[:, :, -2]
    margin = chosen_scores - incumbent_scores
    top_gap = chosen_scores - second
    return chosen, margin.astype(np.float32), top_gap.astype(np.float32)


def policy_session_hits(
    baseline_rank: np.ndarray,
    positive_index: np.ndarray,
    eligible_from: np.ndarray,
    chosen_index: np.ndarray,
    activate: np.ndarray,
) -> np.ndarray:
    turns = np.arange(1, TURN_COUNT + 1, dtype=np.uint8)[None, :]
    eligible = turns >= eligible_from[:, None]
    protected_hit = (baseline_rank >= 1) & (baseline_rank <= 9)
    inactive_hit = (~activate) & (baseline_rank > 0)
    selected_hit = activate & (positive_index >= 0) & (chosen_index == positive_index)
    return np.any(eligible & (protected_hit | inactive_hit | selected_hit), axis=1).astype(np.uint8)


def transition_metrics(baseline_hit: np.ndarray, policy_hit: np.ndarray, activate: np.ndarray) -> dict[str, Any]:
    miss_to_hit = int(np.sum((baseline_hit == 0) & (policy_hit == 1)))
    hit_to_miss = int(np.sum((baseline_hit == 1) & (policy_hit == 0)))
    active_sessions = np.any(activate, axis=1)
    return {
        "baseline_hits": int(baseline_hit.sum()),
        "policy_hits": int(policy_hit.sum()),
        "baseline_hr_at_10": round(float(baseline_hit.mean()), 6),
        "policy_hr_at_10": round(float(policy_hit.mean()), 6),
        "hr_delta": round(float((policy_hit - baseline_hit.astype(np.int16)).mean()), 6),
        "miss_to_hit": miss_to_hit,
        "hit_to_miss": hit_to_miss,
        "net_hits": miss_to_hit - hit_to_miss,
        "activation_turns": int(activate.sum()),
        "activation_sessions": int(active_sessions.sum()),
    }


def gate_feature_matrix(
    features: np.ndarray,
    scores: np.ndarray,
    chosen: np.ndarray,
    incumbent: np.ndarray,
    margin: np.ndarray,
    top_gap: np.ndarray,
) -> np.ndarray:
    flat_features = features.reshape(-1, CANDIDATE_COUNT, FEATURE_COUNT)
    flat_scores = scores.reshape(-1, CANDIDATE_COUNT)
    flat_chosen = chosen.reshape(-1).astype(np.int64)
    flat_incumbent = incumbent.reshape(-1).astype(np.int64)
    rows = np.arange(len(flat_chosen))
    chosen_all = np.asarray(flat_features[rows, flat_chosen])
    incumbent_all = np.asarray(flat_features[rows, flat_incumbent])
    static_columns = [FEATURE_INDEX[name] for name in GATE_STATIC_FEATURES]
    chosen_static = chosen_all[:, static_columns]
    conflict_columns = [
        FEATURE_INDEX[f"{slot}_conflict"]
        for slot in CONSTRAINT_SLOTS
    ]
    chosen_conflict = chosen_all[:, conflict_columns].sum(axis=1)
    incumbent_conflict = incumbent_all[:, conflict_columns].sum(axis=1)
    active_recall = FEATURE_INDEX["active_token_recall"]
    hard_coverage = FEATURE_INDEX["hard_clause_coverage"]
    matrix = np.column_stack(
        (
            margin.reshape(-1),
            top_gap.reshape(-1),
            flat_scores[rows, flat_chosen],
            flat_scores[rows, flat_incumbent],
            chosen_static,
            chosen_all[:, active_recall] - incumbent_all[:, active_recall],
            chosen_all[:, hard_coverage] - incumbent_all[:, hard_coverage],
            chosen_conflict - incumbent_conflict,
        )
    ).astype(np.float32)
    if matrix.shape != (SESSION_COUNT * TURN_COUNT, len(GATE_FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise SmallRankerTrainingError("safe-gate feature matrix is invalid")
    return matrix.reshape(SESSION_COUNT, TURN_COUNT, -1)


def action_training_labels(
    labels: Mapping[str, np.ndarray], chosen: np.ndarray, incumbent: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = np.asarray(labels["positive_index"])
    baseline_rank = np.asarray(labels["baseline_rank"])
    baseline_hit = np.asarray(labels["baseline_session_hit"])
    eligible_from = np.asarray(labels["eligible_from"])
    eligible = np.arange(1, TURN_COUNT + 1)[None, :] >= eligible_from[:, None]
    action = chosen != incumbent
    rescue = action & eligible & (baseline_hit[:, None] == 0) & (chosen == positive) & (positive >= 0)
    direct_risk = action & eligible & (baseline_hit[:, None] == 1) & (baseline_rank == 10) & (chosen != positive)
    weights = np.where(rescue, 1.0, np.where(direct_risk, 5.0, 0.05)).astype(np.float64)
    return rescue.astype(np.uint8), direct_risk.astype(np.uint8), weights


class _ConstantGate:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        positive = np.full(len(x), self.probability, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def _fit_gate_model(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    if not len(x):
        return _ConstantGate(0.0), np.zeros(len(GATE_FEATURE_NAMES)), np.ones(len(GATE_FEATURE_NAMES))
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    classes = np.unique(y)
    if len(classes) != 2:
        return _ConstantGate(float(classes[0]) if len(classes) else 0.0), mean, scale
    model = LogisticRegression(
        C=0.2,
        solver="liblinear",
        max_iter=300,
        random_state=_library_seed(seed),
    )
    model.fit(standardized, y, sample_weight=weights)
    return model, mean, scale


def _predict_gate(model: Any, mean: np.ndarray, scale: np.ndarray, x: np.ndarray) -> np.ndarray:
    return model.predict_proba((x - mean) / scale)[:, 1]


def select_zero_harm_threshold(
    probabilities: np.ndarray,
    action_available: np.ndarray,
    chosen: np.ndarray,
    labels: Mapping[str, np.ndarray],
    session_mask: np.ndarray,
    *,
    maximum_thresholds: int = 257,
) -> dict[str, Any]:
    values = probabilities[action_available & session_mask[:, None]]
    if not len(values):
        return {"threshold": math.inf, "miss_to_hit": 0, "hit_to_miss": 0, "activation_turns": 0}
    quantiles = np.linspace(0.0, 1.0, maximum_thresholds - 1)
    thresholds = np.unique(np.quantile(values, quantiles, method="higher"))
    thresholds = np.concatenate((thresholds, np.asarray([math.inf])))
    activation = (
        action_available[None, :, :]
        & session_mask[None, :, None]
        & (probabilities[None, :, :] >= thresholds[:, None, None])
    )
    baseline_rank = np.asarray(labels["baseline_rank"])
    positive = np.asarray(labels["positive_index"])
    eligible_from = np.asarray(labels["eligible_from"])
    eligible = np.arange(1, TURN_COUNT + 1)[None, :] >= eligible_from[:, None]
    protected = (baseline_rank >= 1) & (baseline_rank <= 9)
    turn_hit = eligible[None, :, :] & (
        protected[None, :, :]
        | ((~activation) & (baseline_rank[None, :, :] > 0))
        | (activation & (positive[None, :, :] >= 0) & (chosen[None, :, :] == positive[None, :, :]))
    )
    policy_hit = np.any(turn_hit, axis=2)
    baseline_hit = np.asarray(labels["baseline_session_hit"]).astype(bool)
    relevant = session_mask[None, :]
    miss_to_hit = np.sum(relevant & (~baseline_hit[None, :]) & policy_hit, axis=1)
    hit_to_miss = np.sum(relevant & baseline_hit[None, :] & (~policy_hit), axis=1)
    activation_turns = activation.sum(axis=(1, 2))
    valid = np.flatnonzero(hit_to_miss == 0)
    if not len(valid):
        raise SmallRankerTrainingError("threshold sweep unexpectedly lacks KEEP fallback")
    best = min(
        valid.tolist(),
        key=lambda index: (
            -int(miss_to_hit[index]),
            int(activation_turns[index]),
            -float(thresholds[index]),
        ),
    )
    return {
        "threshold": float(thresholds[best]),
        "miss_to_hit": int(miss_to_hit[best]),
        "hit_to_miss": int(hit_to_miss[best]),
        "activation_turns": int(activation_turns[best]),
        "candidate_threshold_count": int(len(thresholds)),
    }


def cross_fit_safe_gate(
    gate_features: np.ndarray,
    chosen: np.ndarray,
    incumbent: np.ndarray,
    labels: Mapping[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    outer = np.asarray(labels["outer_fold"])
    inner = np.asarray(labels["inner_fold"])
    rescue, direct_risk, weights = action_training_labels(labels, chosen, incumbent)
    action = chosen != incumbent
    flat_x = gate_features.reshape(-1, gate_features.shape[-1])
    flat_rescue = rescue.reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_action = action.reshape(-1)
    flat_session = np.repeat(np.arange(SESSION_COUNT), TURN_COUNT)
    probabilities = np.zeros((SESSION_COUNT, TURN_COUNT), dtype=np.float32)
    thresholds = np.full(OUTER_FOLDS, math.inf, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for outer_fold in range(OUTER_FOLDS):
        train_sessions = outer != outer_fold
        held_sessions = outer == outer_fold
        inner_probabilities = np.zeros((SESSION_COUNT, TURN_COUNT), dtype=np.float32)
        for inner_fold in range(OUTER_FOLDS):
            gate_train_sessions = train_sessions & (inner != inner_fold)
            gate_valid_sessions = train_sessions & (inner == inner_fold)
            train_rows = flat_action & gate_train_sessions[flat_session]
            valid_rows = flat_action & gate_valid_sessions[flat_session]
            if not np.any(valid_rows):
                continue
            model, mean, scale = _fit_gate_model(
                flat_x[train_rows], flat_rescue[train_rows], flat_weights[train_rows], seed + outer_fold * 17 + inner_fold
            )
            inner_probabilities.reshape(-1)[valid_rows] = _predict_gate(
                model, mean, scale, flat_x[valid_rows]
            ).astype(np.float32)
        selection = select_zero_harm_threshold(
            inner_probabilities,
            action,
            chosen,
            labels,
            train_sessions,
        )
        thresholds[outer_fold] = selection["threshold"]
        train_rows = flat_action & train_sessions[flat_session]
        held_rows = flat_action & held_sessions[flat_session]
        model, mean, scale = _fit_gate_model(
            flat_x[train_rows], flat_rescue[train_rows], flat_weights[train_rows], seed + outer_fold * 101
        )
        probabilities.reshape(-1)[held_rows] = _predict_gate(
            model, mean, scale, flat_x[held_rows]
        ).astype(np.float32)
        fold_records.append(
            {
                "fold": outer_fold,
                "threshold": selection["threshold"],
                "inner_oof_miss_to_hit": selection["miss_to_hit"],
                "inner_oof_hit_to_miss": selection["hit_to_miss"],
                "inner_oof_activation_turns": selection["activation_turns"],
                "gate_train_rescue_rows": int(flat_rescue[train_rows].sum()),
                "gate_train_direct_risk_rows": int(direct_risk.reshape(-1)[train_rows].sum()),
            }
        )
    activation = action & (probabilities >= thresholds[outer][:, None])
    audit = {
        "gate_feature_names": list(GATE_FEATURE_NAMES),
        "gate_feature_schema_sha256": _canonical_sha256(list(GATE_FEATURE_NAMES)),
        "rescue_training_rows": int(rescue.sum()),
        "direct_risk_training_rows": int(direct_risk.sum()),
        "action_rows": int(action.sum()),
        "nested_inner_folds": OUTER_FOLDS,
        "threshold_rule": "inner-OOF zero hit-to-miss, then max miss-to-hit, then min activation",
    }
    return probabilities, activation, fold_records, audit


def _paired_bootstrap(delta: np.ndarray, seed: int, draws: int = 2_000) -> dict[str, float]:
    rng = np.random.default_rng(_library_seed(seed))
    values = np.asarray(delta, dtype=np.float32)
    means = np.empty(draws, dtype=np.float32)
    for offset in range(0, draws, 100):
        count = min(100, draws - offset)
        indices = rng.integers(0, len(values), size=(count, len(values)), endpoint=False)
        means[offset : offset + count] = values[indices].mean(axis=1)
    return {
        "draws": draws,
        "lower_2_5": round(float(np.quantile(means, 0.025)), 6),
        "median": round(float(np.quantile(means, 0.5)), 6),
        "upper_97_5": round(float(np.quantile(means, 0.975)), 6),
    }


def _stratified_transitions(
    codes: np.ndarray,
    codebook: Mapping[str, int],
    baseline: np.ndarray,
    policy: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, code in sorted(codebook.items(), key=lambda item: item[1]):
        mask = codes == int(code)
        result[name] = {
            "sessions": int(mask.sum()),
            "miss_to_hit": int(np.sum(mask & (baseline == 0) & (policy == 1))),
            "hit_to_miss": int(np.sum(mask & (baseline == 1) & (policy == 0))),
            "net": int(np.sum(mask & (policy.astype(np.int16) - baseline.astype(np.int16)))),
        }
    return result


def evaluate_ranker(
    scores: np.ndarray,
    features: np.ndarray,
    labels: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    incumbent = _incumbent_indices(features)
    chosen, margin, top_gap = choose_slot10(scores, incumbent)
    baseline_hit = np.asarray(labels["baseline_session_hit"])
    positive = np.asarray(labels["positive_index"])
    baseline_rank = np.asarray(labels["baseline_rank"])
    eligible_from = np.asarray(labels["eligible_from"])
    ungated_activate = chosen != incumbent
    ungated_hit = policy_session_hits(baseline_rank, positive, eligible_from, chosen, ungated_activate)
    ungated_metrics = transition_metrics(baseline_hit, ungated_hit, ungated_activate)
    gate_features = gate_feature_matrix(features, scores, chosen, incumbent, margin, top_gap)
    gate_probabilities, activation, gate_folds, gate_audit = cross_fit_safe_gate(
        gate_features, chosen, incumbent, labels, seed
    )
    policy_hit = policy_session_hits(baseline_rank, positive, eligible_from, chosen, activation)
    metrics = transition_metrics(baseline_hit, policy_hit, activation)
    outer = np.asarray(labels["outer_fold"])
    family = np.asarray(labels["family_index"])
    fold_transitions = []
    for fold in range(OUTER_FOLDS):
        mask = outer == fold
        fold_transitions.append(
            {
                "fold": fold,
                "sessions": int(mask.sum()),
                "miss_to_hit": int(np.sum(mask & (baseline_hit == 0) & (policy_hit == 1))),
                "hit_to_miss": int(np.sum(mask & (baseline_hit == 1) & (policy_hit == 0))),
                "net": int(np.sum(mask & (policy_hit.astype(np.int16) - baseline_hit.astype(np.int16)))),
                "activation_sessions": int(np.sum(mask & np.any(activation, axis=1))),
            }
        )
    family_delta = np.asarray(
        [
            np.mean(policy_hit[family == value].astype(np.int16) - baseline_hit[family == value].astype(np.int16))
            for value in np.unique(family)
        ],
        dtype=np.float64,
    )
    flat_scores = scores.reshape(-1, CANDIDATE_COUNT)
    flat_positive = positive.reshape(-1)
    positive_mask = flat_positive >= 0
    positive_rows = np.flatnonzero(positive_mask)
    target_scores = flat_scores[positive_rows, flat_positive[positive_mask]]
    indices = np.arange(CANDIDATE_COUNT)[None, :]
    model_rank = 1 + np.sum(
        (flat_scores[positive_rows] > target_scores[:, None])
        | ((flat_scores[positive_rows] == target_scores[:, None]) & (indices < flat_positive[positive_mask, None])),
        axis=1,
    )
    taxonomy_codebook = manifest["label_join"]["taxonomy_codebook"]
    popularity_codebook = manifest["label_join"]["popularity_codebook"]
    delta = policy_hit.astype(np.int16) - baseline_hit.astype(np.int16)
    taxonomy = _stratified_transitions(
        np.asarray(labels["taxonomy_code"]), taxonomy_codebook, baseline_hit, policy_hit
    )
    popularity = _stratified_transitions(
        np.asarray(labels["popularity_code"]), popularity_codebook, baseline_hit, policy_hit
    )
    rescue_family_count = len(set(family[(baseline_hit == 0) & (policy_hit == 1)].tolist()))
    rescue_taxonomies = sum(int(value["miss_to_hit"] > 0) for value in taxonomy.values())
    bootstrap = _paired_bootstrap(delta, seed)
    fold_rescue_coverage = sum(int(value["miss_to_hit"] > 0) for value in fold_transitions)
    gate_requirement = manifest.get("config", {})
    # Threshold constants are read by the caller from the frozen config; the
    # fixed values are duplicated here only for a self-contained result row.
    passes = bool(
        metrics["net_hits"] >= 10
        and metrics["hr_delta"] >= 0.005
        and metrics["hit_to_miss"] == 0
        and fold_rescue_coverage >= 3
        and float(family_delta.mean()) > 0.0
        and bootstrap["lower_2_5"] >= 0.0
        and metrics["activation_sessions"] > 0
        and rescue_family_count >= 2
        and rescue_taxonomies >= 2
    )
    return {
        "ungated_policy": ungated_metrics,
        "gated_policy": metrics,
        "gate": {
            **gate_audit,
            "outer_fold_thresholds": gate_folds,
            "probability_min": round(float(gate_probabilities.min()), 8),
            "probability_max": round(float(gate_probabilities.max()), 8),
        },
        "candidate_metrics": {
            "positive_query_groups": int(positive_mask.sum()),
            "top1_accuracy": round(float(np.mean(model_rank == 1)), 6),
            "top10_accuracy": round(float(np.mean(model_rank <= 10)), 6),
            "mean_c100_rank_before": round(float(np.mean(flat_positive[positive_mask] + 1)), 6),
            "mean_model_rank_after": round(float(np.mean(model_rank)), 6),
            "median_model_rank_after": round(float(np.median(model_rank)), 6),
        },
        "fold_transitions": fold_transitions,
        "taxonomy_transitions": taxonomy,
        "popularity_transitions": popularity,
        "paired_session_bootstrap": bootstrap,
        "family_uniform_delta": round(float(family_delta.mean()), 8),
        "rescue_family_count": rescue_family_count,
        "rescue_taxonomy_count": rescue_taxonomies,
        "fold_rescue_coverage": fold_rescue_coverage,
        "passes_oof_runtime_gate": passes,
    }


def _model_params(spec: Mapping[str, Any], seed: int, xgboost_version: str) -> dict[str, Any]:
    major = int(xgboost_version.split(".", 1)[0])
    params: dict[str, Any] = {
        "objective": str(spec["objective"]),
        "n_estimators": int(spec["max_rounds"]),
        "max_depth": int(spec["max_depth"]),
        "learning_rate": float(spec["eta"]),
        "min_child_weight": float(spec["min_child_weight"]),
        "subsample": float(spec["subsample"]),
        "colsample_bytree": float(spec["colsample_bytree"]),
        "reg_alpha": float(spec["reg_alpha"]),
        "reg_lambda": float(spec["reg_lambda"]),
        "tree_method": "hist",
        "max_bin": 256,
        "eval_metric": "ndcg@10",
        "random_state": _library_seed(seed),
        "n_jobs": max(1, (os.cpu_count() or 2) - 1),
        "verbosity": 0,
    }
    if major >= 2:
        params["lambdarank_pair_method"] = str(spec["lambdarank_pair_method"])
        if spec["lambdarank_pair_method"] == "topk":
            params["lambdarank_num_pair_per_sample"] = 10
    return params


def _predict_session_batches(model: Any, features: np.ndarray, sessions: np.ndarray, batch_size: int = 50) -> tuple[np.ndarray, float]:
    output = np.empty((len(sessions), TURN_COUNT, CANDIDATE_COUNT), dtype=np.float32)
    started = time.perf_counter()
    for offset in range(0, len(sessions), batch_size):
        selected = sessions[offset : offset + batch_size]
        block = np.asarray(features[selected], dtype=np.float32).reshape(-1, FEATURE_COUNT)
        prediction = np.asarray(model.predict(block), dtype=np.float32)
        output[offset : offset + len(selected)] = prediction.reshape(
            len(selected), TURN_COUNT, CANDIDATE_COUNT
        )
    return output, time.perf_counter() - started


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    if ASIN_SHAPE_RE.search(raw.decode("ascii")):
        raise SmallRankerTrainingError("OOF result contains an identity-shaped token")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _global_upper_bound(labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    positive = np.asarray(labels["positive_index"])
    eligible_from = np.asarray(labels["eligible_from"])
    eligible = np.arange(1, TURN_COUNT + 1)[None, :] >= eligible_from[:, None]
    c100_hit = np.any(eligible & (positive >= 0), axis=1)
    baseline = np.asarray(labels["baseline_session_hit"])
    return {
        "baseline_hits": int(baseline.sum()),
        "c100_oracle_hits": int(c100_hit.sum()),
        "c100_oracle_hr_at_10": round(float(c100_hit.mean()), 6),
        "reachable_extra_sessions": int(np.sum((baseline == 0) & c100_hit)),
        "unreachable_baseline_misses": int(np.sum((baseline == 0) & (~c100_hit))),
    }


def train_batch(config_path: Path) -> dict[str, Any]:
    import sklearn
    import xgboost as xgb

    started = time.perf_counter()
    inputs = load_cache(config_path)
    result_path = inputs.output_dir / "oof_results.json"
    leaderboard_path = inputs.output_dir / "leaderboard.json"
    if result_path.exists() or leaderboard_path.exists():
        raise FileExistsError("OOF batch output already exists")
    selected_started = time.perf_counter()
    selected = build_selected_training_rows(inputs.features, inputs.labels)
    selected_seconds = time.perf_counter() - selected_started
    outer_by_row = np.asarray(inputs.labels["outer_fold"])[selected.session]
    inner_by_row = np.asarray(inputs.labels["inner_fold"])[selected.session]
    config_results: list[dict[str, Any]] = []
    model_specs = list(inputs.config["training"]["configs"])
    xgboost_version = str(xgb.__version__)
    compatibility = {
        "requested": "XGBoost 3.2.x CPU",
        "used": xgboost_version,
        "offline_resolution": "only locally installed XGBoost was used; no network/install occurred",
        "pair_method_supported": int(xgboost_version.split(".", 1)[0]) >= 2,
        "limitation": (
            None
            if int(xgboost_version.split(".", 1)[0]) >= 2
            else "1.7.6 does not expose lambdarank_pair_method; objective/depth/lr/reg configs remain frozen"
        ),
    }
    for config_index, spec in enumerate(model_specs):
        config_id = str(spec["id"])
        score_path = inputs.output_dir / f"oof_scores_{config_id}.npy"
        if score_path.exists() or score_path.is_symlink():
            raise FileExistsError(score_path)
        oof_scores = np.lib.format.open_memmap(
            score_path,
            mode="w+",
            dtype=np.float32,
            shape=(SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT),
        )
        fold_models: list[dict[str, Any]] = []
        train_seconds = 0.0
        predict_seconds = 0.0
        prediction_rows = 0
        deterministic = True
        model_dir = inputs.output_dir / "models" / config_id
        model_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(OUTER_FOLDS):
            train_mask = (outer_by_row != fold) & (inner_by_row != fold)
            eval_mask = (outer_by_row != fold) & (inner_by_row == fold)
            if not np.any(train_mask) or not np.any(eval_mask):
                raise SmallRankerTrainingError("outer/inner training partition is empty")
            params = _model_params(spec, int(inputs.config["seed"]) + config_index * 1_000 + fold, xgboost_version)
            model = xgb.XGBRanker(**params)
            tick = time.perf_counter()
            model.fit(
                selected.x[train_mask],
                selected.y[train_mask],
                qid=selected.qid[train_mask],
                eval_set=[(selected.x[eval_mask], selected.y[eval_mask])],
                eval_qid=[selected.qid[eval_mask]],
                early_stopping_rounds=40,
                verbose=False,
            )
            fold_train_seconds = time.perf_counter() - tick
            train_seconds += fold_train_seconds
            held_sessions = np.flatnonzero(np.asarray(inputs.labels["outer_fold"]) == fold)
            held_scores, fold_predict_seconds = _predict_session_batches(
                model, inputs.features, held_sessions
            )
            predict_seconds += fold_predict_seconds
            prediction_rows += int(held_scores.size)
            oof_scores[held_sessions] = held_scores
            parity_block = np.asarray(inputs.features[held_sessions[:1]], dtype=np.float32).reshape(-1, FEATURE_COUNT)[:1_000]
            first = np.asarray(model.predict(parity_block), dtype=np.float32)
            second = np.asarray(model.predict(parity_block), dtype=np.float32)
            deterministic = deterministic and first.tobytes() == second.tobytes()
            model_path = model_dir / f"fold_{fold}.json"
            model.get_booster().save_model(model_path)
            if _identity_shape_scan(model_path):
                raise SmallRankerTrainingError("research model contains an identity-shaped token")
            fold_record = {
                "fold": fold,
                "train_rows": int(train_mask.sum()),
                "eval_rows": int(eval_mask.sum()),
                "train_query_groups": int(len(np.unique(selected.qid[train_mask]))),
                "eval_query_groups": int(len(np.unique(selected.qid[eval_mask]))),
                "best_iteration": int(getattr(model, "best_iteration", int(spec["max_rounds"]) - 1)),
                "training_seconds": round(fold_train_seconds, 6),
                "prediction_seconds": round(fold_predict_seconds, 6),
                "model_bytes": model_path.stat().st_size,
                "model_sha256": _sha256(model_path),
            }
            fold_models.append(fold_record)
            print(
                json.dumps(
                    {
                        "config": config_id,
                        "fold": fold,
                        "best_iteration": fold_record["best_iteration"],
                        "train_seconds": fold_record["training_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        oof_scores.flush()
        evaluation_started = time.perf_counter()
        evaluation = evaluate_ranker(
            np.asarray(oof_scores),
            inputs.features,
            inputs.labels,
            inputs.manifest,
            int(inputs.config["seed"]) + config_index * 10_000,
        )
        evaluation_seconds = time.perf_counter() - evaluation_started
        score_sha256 = _sha256(score_path)
        config_result = {
            "id": config_id,
            "spec": spec,
            "spec_sha256": _canonical_sha256(spec),
            "fold_models": fold_models,
            "model_bytes_total": sum(int(item["model_bytes"]) for item in fold_models),
            "model_bytes_max_fold": max(int(item["model_bytes"]) for item in fold_models),
            "training_seconds": round(train_seconds, 6),
            "prediction_seconds": round(predict_seconds, 6),
            "prediction_rows_per_second": round(prediction_rows / predict_seconds, 3) if predict_seconds else None,
            "evaluation_seconds": round(evaluation_seconds, 6),
            "exact_repeat_predictions_byte_identical": deterministic,
            "oof_scores": {
                "path": score_path.relative_to(ROOT).as_posix(),
                "bytes": score_path.stat().st_size,
                "sha256": score_sha256,
            },
            **evaluation,
        }
        config_results.append(config_result)
        print(
            json.dumps(
                {
                    "config": config_id,
                    "net_hits": evaluation["gated_policy"]["net_hits"],
                    "miss_to_hit": evaluation["gated_policy"]["miss_to_hit"],
                    "hit_to_miss": evaluation["gated_policy"]["hit_to_miss"],
                    "passes": evaluation["passes_oof_runtime_gate"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del oof_scores
    ordered = sorted(
        config_results,
        key=lambda item: (
            -int(item["passes_oof_runtime_gate"]),
            -int(item["gated_policy"]["net_hits"]),
            int(item["gated_policy"]["hit_to_miss"]),
            -float(item["candidate_metrics"]["top1_accuracy"]),
            str(item["id"]),
        ),
    )
    leaderboard = [
        {
            "rank": rank,
            "id": item["id"],
            "candidate_top1": item["candidate_metrics"]["top1_accuracy"],
            "ungated_net": item["ungated_policy"]["net_hits"],
            "gated_hr_delta": item["gated_policy"]["hr_delta"],
            "miss_to_hit": item["gated_policy"]["miss_to_hit"],
            "hit_to_miss": item["gated_policy"]["hit_to_miss"],
            "activation_sessions": item["gated_policy"]["activation_sessions"],
            "fold_rescue_coverage": item["fold_rescue_coverage"],
            "bootstrap_lower": item["paired_session_bootstrap"]["lower_2_5"],
            "passes_oof_runtime_gate": item["passes_oof_runtime_gate"],
        }
        for rank, item in enumerate(ordered, 1)
    ]
    promoted = [item for item in ordered if item["passes_oof_runtime_gate"]]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split": "train_explore",
        "cache": {
            "feature_sha256": inputs.manifest["feature_cache"]["sha256"],
            "label_sha256": inputs.manifest["label_cache"]["sha256"],
            "feature_schema_sha256": inputs.manifest["feature_cache"]["feature_schema_sha256"],
        },
        "dependency": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "xgboost": xgboost_version,
            "sklearn": sklearn.__version__,
            "compatibility": compatibility,
            "runtime_dependencies_added": [],
        },
        "selected_training_rows": {
            "rows": len(selected.y),
            "query_groups": selected.group_count,
            "build_seconds": round(selected_seconds, 6),
            "qid_monotonic": True,
            "one_positive_per_group": True,
        },
        "c100_upper_bound": _global_upper_bound(inputs.labels),
        "configs": config_results,
        "leaderboard": leaderboard,
        "decision": {
            "runtime_candidate": promoted[0]["id"] if promoted else None,
            "run_runtime_smoke": bool(promoted),
            "run_calibration": False,
            "reason": (
                "at least one config passed every OOF runtime gate"
                if promoted
                else "no preregistered rich LambdaMART config passed +10 net/zero-harm/fold/bootstrap gates"
            ),
        },
        "timing_seconds": {"total_batch": round(time.perf_counter() - started, 6)},
        "trainer": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "privacy": {
            "identity_features": False,
            "labels_used_only_for_offline_objective_and_metrics": True,
            "model_asin_shape_matches": 0,
            "calibration_opened": False,
            "selection_opened": False,
            "confirmation_opened": False,
            "public_opened": False,
        },
    }
    result["canonical_sha256"] = _canonical_sha256(result)
    _write_json_exclusive(result_path, result)
    _write_json_exclusive(
        leaderboard_path,
        {
            "schema_version": "small-ranker-leaderboard.v1",
            "rows": leaderboard,
            "decision": result["decision"],
            "source_result_sha256": _sha256(result_path),
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train_batch(args.config)
    print(
        json.dumps(
            {
                "leader": result["leaderboard"][0],
                "decision": result["decision"],
                "total_seconds": result["timing_seconds"]["total_batch"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SmallRankerTrainingError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[small-ranker-training] {error}", file=sys.stderr)
        raise SystemExit(1)
