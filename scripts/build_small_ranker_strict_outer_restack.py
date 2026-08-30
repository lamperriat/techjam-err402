"""Build one label-isolated outer shard for the v2.8 strict restack cache.

Each invocation rebuilds all 30 rankers for exactly one ``(pass, outer)``
domain.  Only T_o outcome slices are retained.  H_o receives target-free
ranker, current-gate, and portfolio-selector inference; this module deliberately
contains no held-outcome evaluation entry point.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_small_ranker_metric_gate as metric  # noqa: E402
from scripts import analyze_small_ranker_rr_regret_gate as rr  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import probe_small_ranker_strict_outer_restack as probe  # noqa: E402
from scripts import small_ranker_portfolio_selector_py39 as portfolio  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402
from scripts import train_small_ranker_focused_outer_oof as focused  # noqa: E402


SCHEMA_VERSION = "small-ranker-strict-outer-restack-outer-cache.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_8.strict_outer_restack_preregistration.json"
)
STAGE0_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_8.strict_outer_restack_implementation_amendment.json"
)
STAGE1_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_8.strict_outer_restack_stage1_implementation_amendment.json"
)
STAGE0_MANIFEST = ROOT / (
    "configs/small_ranker_v2_8.strict_outer_restack_stage0.manifest.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
EXPECTED_HASHES = {
    "raw_features": "2b19835a1bced7f21322610296c712e3d06d915274719e11c268d31f7f596089",
    "labels": "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb",
    "projected_features": "cd9b075b923c31afe10b9a9c6d720de22e221c93de961595ff484ea4f532b90a",
    "feature_schema_manifest": "a930d184672bc29d9dd4bc1c2e908da035712ab061f2127a9771b2f3ed6a5c1a",
    "preregistration": "67d93823f2016dc4fdbcfa6687c236692f2e554cfa678e0878b5f1294c7a3ef9",
    "stage0_amendment": "a4a44d28d4e88ed75c92b61723b0aa7b5627211faeb08aaa0a9c8ba2a9bc3938",
    "stage1_amendment": "4f613340292dc0fead811fd7d19c1e8125b98fbf7ab4b48b089e4b80826d069a",
    "stage0_manifest": "1f79d728add447be4a010dddc75e31660b0c3c6161a40b6cb3fc1b409a2a2a18",
    "stage0_result": "b32f00462bd8206cae229063f5d6f94a0d5ad50d5a109fa42010e71651a90a0d",
}
HELPER_HASHES = {
    "scripts/probe_small_ranker_strict_outer_restack.py": "a1f158c4ceffe692ba1acd8f11763dab5243efea07cc4c2c9aa54e614256159d",
    "scripts/evaluate_small_ranker_portfolio_selector.py": "b50ce1f78a4a8cc209d051eae3610fae2b2c3298f417db8ee094a05bd4f7468e",
    "scripts/train_small_ranker.py": "db7f4a3e19da118abb7d37fc1530babd6928894e51e85010b11d9dcdc1d7e583",
    "scripts/export_small_ranker_fold_safe_artifact.py": "5115026c53b21d4d5930cb9af7783c0988b049a0e259f5a0a588901ad44f5e8b",
    "scripts/analyze_small_ranker_metric_gate.py": "8c0cbffa6cd3dc62ddee3bb386c16bd60592a6324ecf6fcf4bcd4cf37951ca83",
    "scripts/analyze_small_ranker_rr_regret_gate.py": "793e3615df38cd995f55e57decaeea35b549e40ad50ee3bf8a6dbf1055ca7e80",
    "scripts/train_small_ranker_focused_outer_oof.py": "8fbf05d6225e04e3f76db7e65a91160394f11316ac90b416b6cf53b0b2ba497c",
    "scripts/evaluate_small_ranker_supplemental_pairwise.py": "e393aa5cfe71de20640fca28b675c3f28cb73578156fdf9a607871f1799b31ce",
    "scripts/small_ranker_portfolio_selector_py39.py": "35b7b68af7c52b7ecf0fe37ee686ed2e737ff2f6643622abd26dbc97e192cba8",
}
CONFIG_HASHES = {
    "configs/small_ranker_v2_7.portfolio_selector_preregistration.json": "15179b29c76f4075a2635d893fca0e62922b196808df7ad0e1d9a59d02da3e43",
    "configs/small_ranker_v2_7.portfolio_selector_implementation_amendment.json": "e27ba5f668fffe5dbcca7b4a2ca44a2c8f719e17f472134aaaaa305d8e362ef9",
}
OUTCOME_FIELDS = (
    "baseline_rank",
    "baseline_session_hit",
    "eligible_from",
    "positive_index",
    "training_indices",
    "training_length",
)
GENERIC_MODEL_IDS = (
    "current_ndcg_d4_lr003",
    "pairwise_d4_control",
    "ndcg_d6_lr006",
    "ndcg_d4_regularized",
)
ALL_SESSIONS = np.arange(base.SESSION_COUNT, dtype=np.int16)


class StrictRestackBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class OuterInputs:
    raw_features: np.ndarray
    projected_features: np.ndarray
    outer_fold: np.ndarray
    inner_fold: np.ndarray
    family_index: np.ndarray
    t_sessions: np.ndarray
    h_sessions: np.ndarray
    labels_t: Mapping[str, np.ndarray]
    paths: Mapping[str, Path]


@dataclass(frozen=True)
class InferenceSurface:
    incumbent: np.ndarray
    chosen: np.ndarray
    action: np.ndarray
    gate_features: np.ndarray


@dataclass(frozen=True)
class FocusedGroups:
    local_session: np.ndarray
    global_session: np.ndarray
    turn: np.ndarray
    hard: np.ndarray
    hard_session: np.ndarray
    control_session: np.ndarray


def _sha256(path: Path) -> str:
    return probe._sha256(path)


def _array_sha256(value: np.ndarray) -> str:
    return probe._array_sha256(np.asarray(value))


def _canonical_sha256(value: object) -> str:
    return probe._canonical_sha256(value)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StrictRestackBuildError("expected a JSON object")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    probe._write_json_exclusive(path, value)


def _write_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    return probe._write_npy_exclusive(path, np.asarray(value))


def _validate_environment() -> dict[str, Any]:
    import sklearn
    import xgboost

    actual = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "xgboost": xgboost.__version__,
        "sklearn": sklearn.__version__,
        "workers": 1,
    }
    expected = {
        "python": "3.9.19",
        "numpy": "1.26.4",
        "xgboost": "1.7.6",
        "sklearn": "1.1.3",
        "workers": 1,
    }
    if actual != expected:
        raise StrictRestackBuildError("Stage 1 dependency identity mismatch")
    return actual


def _validate_protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    schema_manifest = ROOT / "configs/small_ranker_v1.cache.manifest.json"
    required = {
        "preregistration": PREREGISTRATION,
        "stage0_amendment": STAGE0_AMENDMENT,
        "stage1_amendment": STAGE1_AMENDMENT,
        "stage0_manifest": STAGE0_MANIFEST,
        "feature_schema_manifest": schema_manifest,
    }
    for name, path in required.items():
        if not path.is_file() or path.is_symlink() or _sha256(path) != EXPECTED_HASHES[name]:
            raise StrictRestackBuildError("protocol identity mismatch: %s" % name)
    for relative_path, expected in {**HELPER_HASHES, **CONFIG_HASHES}.items():
        path = ROOT / relative_path
        if not path.is_file() or path.is_symlink():
            raise StrictRestackBuildError("frozen source unavailable: %s" % relative_path)
        if expected is not None and _sha256(path) != expected:
            raise StrictRestackBuildError("frozen source drifted: %s" % relative_path)
    manifest = _load_json(STAGE0_MANIFEST)
    result_path = ROOT / str(manifest.get("result", {}).get("path", ""))
    if (
        manifest.get("status") != "IMPLEMENTATION_PASS_STAGE0"
        or not result_path.is_file()
        or result_path.is_symlink()
        or _sha256(result_path) != EXPECTED_HASHES["stage0_result"]
        or manifest.get("result", {}).get("sha256") != EXPECTED_HASHES["stage0_result"]
        or manifest.get("exact_repeat", {}).get("equal") is not True
        or manifest.get("decision", {}).get("stage_1_authorized") is not True
    ):
        raise StrictRestackBuildError("Stage-0 authorization is invalid")
    prereg = _load_json(PREREGISTRATION)
    amendment = _load_json(STAGE1_AMENDMENT)
    models = prereg.get("generic_training", {}).get("models", [])
    if (
        prereg.get("schema_version")
        != "small-ranker-strict-outer-restack-preregistration.v1"
        or amendment.get("schema_version")
        != "small-ranker-strict-outer-restack-stage1-implementation-amendment.v1"
        or {str(row.get("id")) for row in models} != set(GENERIC_MODEL_IDS)
        or int(amendment.get("full_retrain_rule", {}).get("fits_per_pass", -1)) != 150
        or int(amendment.get("full_retrain_rule", {}).get("required_total_xgboost_fits", -1)) != 300
    ):
        raise StrictRestackBuildError("Stage-1 mechanics drifted")
    return prereg, amendment, manifest


def _source_snapshot() -> dict[str, Any]:
    stage0_manifest = _load_json(STAGE0_MANIFEST)
    result_path = ROOT / str(stage0_manifest["result"]["path"])
    return {
        "builder_sha256": _sha256(Path(__file__).resolve()),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "stage0_amendment_sha256": _sha256(STAGE0_AMENDMENT),
        "stage1_amendment_sha256": _sha256(STAGE1_AMENDMENT),
        "stage0_manifest_sha256": _sha256(STAGE0_MANIFEST),
        "stage0_result_sha256": _sha256(result_path),
        "helper_sha256": {
            path: _sha256(ROOT / path) for path in HELPER_HASHES
        },
        "config_sha256": {
            path: _sha256(ROOT / path) for path in CONFIG_HASHES
        },
    }


def _load_outer_inputs(
    source_root: Path, projection_root: Path, outer_fold: int
) -> OuterInputs:
    if outer_fold not in range(base.OUTER_FOLDS):
        raise StrictRestackBuildError("outer fold is out of range")
    paths = {
        "raw_features": source_root.resolve()
        / "experiments/fast_track/small_ranker_v1/features.npy",
        "labels": source_root.resolve()
        / "experiments/fast_track/small_ranker_v1/labels_v2.npz",
        "projected_features": projection_root.resolve()
        / "experiments/fast_track/small_ranker_fold_safe_projected_features.npy",
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink() or _sha256(path) != EXPECTED_HASHES[name]:
            raise StrictRestackBuildError("sealed input mismatch: %s" % name)
    raw = np.load(paths["raw_features"], mmap_mode="r", allow_pickle=False)
    projected = np.load(paths["projected_features"], mmap_mode="r", allow_pickle=False)
    expected_shape = (
        base.SESSION_COUNT,
        base.TURN_COUNT,
        base.CANDIDATE_COUNT,
        base.FEATURE_COUNT,
    )
    if (
        raw.shape != expected_shape
        or projected.shape != expected_shape
        or raw.dtype != np.float32
        or projected.dtype != np.float32
    ):
        raise StrictRestackBuildError("feature tensor schema mismatch")
    with np.load(paths["labels"], allow_pickle=False) as archive:
        outer = np.asarray(archive["outer_fold"], dtype=np.uint8).copy()
        inner = np.asarray(archive["inner_fold"], dtype=np.uint8).copy()
        family = np.asarray(archive["family_index"], dtype=np.int32).copy()
        t_sessions = np.flatnonzero(outer != outer_fold).astype(np.int16)
        h_sessions = np.flatnonzero(outer == outer_fold).astype(np.int16)
        labels_t = {
            name: np.asarray(archive[name][t_sessions]).copy()
            for name in OUTCOME_FIELDS
        }
    labels_t = {
        **labels_t,
        "outer_fold": outer[t_sessions].copy(),
        "inner_fold": inner[t_sessions].copy(),
        "family_index": family[t_sessions].copy(),
    }
    if len(t_sessions) != 1600 or len(h_sessions) != 400:
        raise StrictRestackBuildError("outer domain cardinality mismatch")
    if set(np.unique(inner[t_sessions]).tolist()) != set(range(base.OUTER_FOLDS)):
        raise StrictRestackBuildError("T_o inner folds are incomplete")
    for family_id in np.unique(family):
        family_mask = family == family_id
        if len(np.unique(outer[family_mask])) != 1 or len(np.unique(inner[family_mask])) != 1:
            raise StrictRestackBuildError("product family crosses a fold boundary")
    if any(len(np.asarray(labels_t[name])) != len(t_sessions) for name in OUTCOME_FIELDS):
        raise StrictRestackBuildError("T_o outcome slice scope mismatch")
    return OuterInputs(
        raw,
        projected,
        outer,
        inner,
        family,
        t_sessions,
        h_sessions,
        labels_t,
        paths,
    )


def _mask_record(mask: np.ndarray, family: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    return {
        "sessions": int(selected.sum()),
        "families": int(len(np.unique(family[selected]))),
        "mask_sha256": _array_sha256(selected.astype(np.uint8)),
        "session_sha256": _array_sha256(
            np.flatnonzero(selected).astype(np.int16)
        ),
    }


def _domain_records(inputs: OuterInputs, outer_fold: int) -> dict[str, Any]:
    for family_id in np.unique(inputs.family_index):
        family_mask = inputs.family_index == family_id
        if (
            len(np.unique(inputs.outer_fold[family_mask])) != 1
            or len(np.unique(inputs.inner_fold[family_mask])) != 1
        ):
            raise StrictRestackBuildError("product family crosses a fold boundary")
    held = inputs.outer_fold == outer_fold
    train = ~held
    coverage = np.zeros(base.SESSION_COUNT, dtype=np.uint8)
    records = {
        "H_%d" % outer_fold: _mask_record(held, inputs.family_index),
        "T_%d" % outer_fold: _mask_record(train, inputs.family_index),
    }
    for inner_fold in range(base.OUTER_FOLDS):
        valid = train & (inputs.inner_fold == inner_fold)
        actual_train = train & (inputs.inner_fold != inner_fold)
        if np.any(valid & actual_train) or np.any(held & (valid | actual_train)):
            raise StrictRestackBuildError("outer/inner domain masks overlap")
        coverage[valid] += 1
        records["V_%d%d" % (outer_fold, inner_fold)] = _mask_record(
            valid, inputs.family_index
        )
        records["A_%d%d" % (outer_fold, inner_fold)] = _mask_record(
            actual_train, inputs.family_index
        )
    if not np.all(coverage[train] == 1) or np.any(coverage[held]):
        raise StrictRestackBuildError("inner validation coverage is invalid")
    records["inner_validation_coverage_sha256"] = _array_sha256(coverage)
    return records


def _build_selected_domain(inputs: OuterInputs) -> base.SelectedTrainingRows:
    lengths = np.asarray(inputs.labels_t["training_length"], dtype=np.int64)
    indices = np.asarray(inputs.labels_t["training_indices"], dtype=np.int64)
    positive = np.asarray(inputs.labels_t["positive_index"], dtype=np.int64)
    row_count = int(lengths.sum())
    x = np.empty((row_count, base.FEATURE_COUNT), dtype=np.float32)
    y = np.empty(row_count, dtype=np.float32)
    qid = np.empty(row_count, dtype=np.int32)
    sessions = np.empty(row_count, dtype=np.int16)
    turns = np.empty(row_count, dtype=np.uint8)
    cursor = 0
    group_count = 0
    for local_session, global_session in enumerate(inputs.t_sessions):
        for turn in range(base.TURN_COUNT):
            length = int(lengths[local_session, turn])
            if not length:
                continue
            chosen = indices[local_session, turn, :length]
            if np.any(chosen < 0) or len(np.unique(chosen)) != length:
                raise StrictRestackBuildError("training candidate group is invalid")
            end = cursor + length
            x[cursor:end] = inputs.raw_features[int(global_session), turn, chosen]
            y[cursor:end] = (chosen == positive[local_session, turn]).astype(
                np.float32
            )
            qid[cursor:end] = int(global_session) * base.TURN_COUNT + turn
            sessions[cursor:end] = int(global_session)
            turns[cursor:end] = turn
            cursor = end
            group_count += 1
    if (
        cursor != row_count
        or base.validate_grouped_qid(qid, y) != group_count
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise StrictRestackBuildError("T_o grouped rows failed validation")
    return base.SelectedTrainingRows(x, y, qid, sessions, turns, group_count)


def _training_surface(
    inputs: OuterInputs, scores: np.ndarray
) -> frozen.ActionSurface:
    incumbent = probe._incumbent_for_sessions(
        inputs.projected_features, inputs.t_sessions
    )
    chosen, margin, top_gap = probe.choose_slot10_any(scores, incumbent)
    gate_features = probe.gate_feature_matrix_any(
        inputs.projected_features,
        inputs.t_sessions,
        scores,
        chosen,
        incumbent,
        margin,
        top_gap,
    )
    rescue, _direct_risk, rescue_weights = base.action_training_labels(
        inputs.labels_t, chosen, incumbent
    )
    rr_regret = rr.single_action_rr_regret(inputs.labels_t, chosen, incumbent)
    regret = (rr_regret > 0).astype(np.uint8)
    regret_weights = np.where(
        regret > 0,
        5.0 + 20.0 * rr_regret,
        np.where(rescue > 0, 0.2, 0.05),
    ).astype(np.float64)
    return frozen.ActionSurface(
        incumbent,
        chosen,
        chosen != incumbent,
        gate_features,
        rescue,
        rescue_weights,
        regret,
        regret_weights,
    )


def _inference_surface(
    projected: np.ndarray, sessions: np.ndarray, scores: np.ndarray
) -> InferenceSurface:
    incumbent = probe._incumbent_for_sessions(projected, sessions)
    chosen, margin, top_gap = probe.choose_slot10_any(scores, incumbent)
    gate_features = probe.gate_feature_matrix_any(
        projected,
        sessions,
        scores,
        chosen,
        incumbent,
        margin,
        top_gap,
    )
    return InferenceSurface(
        incumbent=incumbent,
        chosen=chosen,
        action=chosen != incumbent,
        gate_features=gate_features,
    )


def _crossfit_current(
    surface: frozen.ActionSurface,
    labels_t: Mapping[str, np.ndarray],
    outer_fold: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    inner = np.asarray(labels_t["inner_fold"], dtype=np.uint8)
    flat_x = surface.gate_features.reshape(-1, len(base.GATE_FEATURE_NAMES))
    flat_action = surface.action.reshape(-1)
    flat_session = np.repeat(np.arange(len(inner)), base.TURN_COUNT)
    targets = (surface.rescue.reshape(-1), surface.regret.reshape(-1))
    weights = (
        surface.rescue_weights.reshape(-1),
        surface.regret_weights.reshape(-1),
    )
    probabilities = [
        np.zeros_like(surface.action, dtype=np.float32) for _ in range(2)
    ]
    coverage = np.zeros_like(flat_action, dtype=np.uint8)
    fold_records = []
    for inner_fold in range(base.OUTER_FOLDS):
        train_session = inner != inner_fold
        valid_session = inner == inner_fold
        train_rows = flat_action & train_session[flat_session]
        valid_rows = flat_action & valid_session[flat_session]
        if not np.any(train_rows) or not np.any(valid_rows):
            raise StrictRestackBuildError("current gate partition is empty")
        coverage[valid_rows] += 1
        head_hashes = []
        for head in range(2):
            seed = probe.BASE_SEED + head * 10_000 + outer_fold * 31 + inner_fold
            model, mean, scale = base._fit_gate_model(
                flat_x[train_rows],
                targets[head][train_rows],
                weights[head][train_rows],
                seed,
            )
            probabilities[head].reshape(-1)[valid_rows] = base._predict_gate(
                model, mean, scale, flat_x[valid_rows]
            ).astype(np.float32)
            head_hashes.append(probe._gate_model_sha256(model, mean, scale))
        fold_records.append(
            {
                "inner_fold": inner_fold,
                "train_action_rows": int(train_rows.sum()),
                "valid_action_rows": int(valid_rows.sum()),
                "head_model_sha256": head_hashes,
            }
        )
    if not np.all(coverage[flat_action] == 1) or np.any(coverage[~flat_action]):
        raise StrictRestackBuildError("current gate OOF coverage is invalid")
    utility = probabilities[0] - frozen.RR_MULTIPLIER * probabilities[1]
    selected = frozen._select_inner_quantile(
        utility,
        surface,
        labels_t,
        np.ones(len(inner), dtype=bool),
        inner,
    )
    threshold = float(selected["inner_threshold"])
    activation = surface.action & (utility >= threshold)
    state = metric.policy_session_state(labels_t, surface.chosen, activation)
    record = {
        "selected_quantile": float(selected["quantile"]),
        "inner_threshold": threshold if np.isfinite(threshold) else "inf",
        "folds": fold_records,
        "head_0_probability_sha256": _array_sha256(probabilities[0]),
        "head_1_probability_sha256": _array_sha256(probabilities[1]),
        "utility_sha256": _array_sha256(utility),
        "activation_sha256": _array_sha256(activation.astype(np.uint8)),
        "action_oof_coverage_sha256": _array_sha256(coverage),
        "training_surface_sha256": {
            "head_0_label": _array_sha256(surface.rescue),
            "head_0_weight": _array_sha256(surface.rescue_weights),
            "head_1_label": _array_sha256(surface.regret),
            "head_1_weight": _array_sha256(surface.regret_weights),
        },
        "current_state_sha256": {
            name: _array_sha256(np.asarray(state[name]))
            for name in ("hit", "first_rank", "first_turn")
        },
        "probabilities": probabilities,
        "utility": utility,
    }
    return state, activation, record


def _fit_full_current_gate(
    training: frozen.ActionSurface,
    inference: InferenceSurface,
    outer_fold: int,
    quantile: float,
    t_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if t_mask.shape != (base.SESSION_COUNT,) or t_mask.dtype != np.bool_:
        raise StrictRestackBuildError("current reference mask mismatch")
    train_x = training.gate_features.reshape(-1, len(base.GATE_FEATURE_NAMES))
    train_action = training.action.reshape(-1)
    infer_x = inference.gate_features.reshape(-1, len(base.GATE_FEATURE_NAMES))
    infer_action = inference.action.reshape(-1)
    targets = (training.rescue.reshape(-1), training.regret.reshape(-1))
    weights = (
        training.rescue_weights.reshape(-1),
        training.regret_weights.reshape(-1),
    )
    probabilities = [
        np.zeros(inference.action.shape, dtype=np.float32) for _ in range(2)
    ]
    head_hashes = []
    for head in range(2):
        seed = probe.BASE_SEED + head * 10_000 + outer_fold * 101
        model, mean, scale = base._fit_gate_model(
            train_x[train_action],
            targets[head][train_action],
            weights[head][train_action],
            seed,
        )
        probabilities[head].reshape(-1)[infer_action] = base._predict_gate(
            model, mean, scale, infer_x[infer_action]
        ).astype(np.float32)
        head_hashes.append(
            {
                "seed": int(seed),
                "model_sha256": probe._gate_model_sha256(model, mean, scale),
            }
        )
    utility = probabilities[0] - frozen.RR_MULTIPLIER * probabilities[1]
    reference_values = utility[t_mask[:, None] & inference.action]
    mapped_threshold = frozen._threshold_at_quantile(reference_values, quantile)
    activation = inference.action & (utility >= mapped_threshold)
    return activation, {
        "head_models": head_hashes,
        "train_action_rows": int(train_action.sum()),
        "reference_action_rows": int((t_mask[:, None] & inference.action).sum()),
        "selected_quantile": float(quantile),
        "mapped_reference_threshold": (
            float(mapped_threshold) if np.isfinite(mapped_threshold) else "inf"
        ),
        "head_0_probability_sha256": _array_sha256(probabilities[0]),
        "head_1_probability_sha256": _array_sha256(probabilities[1]),
        "utility_sha256": _array_sha256(utility),
        "reference_utility_sha256": _array_sha256(utility[t_mask]),
        "held_utility_sha256": _array_sha256(utility[~t_mask]),
        "activation_sha256": _array_sha256(activation.astype(np.uint8)),
        "probabilities": probabilities,
        "utility": utility,
    }


def _select_focused_groups(
    inputs: OuterInputs,
    current_state: Mapping[str, np.ndarray],
    incumbent: np.ndarray,
) -> FocusedGroups:
    positive = np.asarray(inputs.labels_t["positive_index"], dtype=np.int16)
    eligible_from = np.asarray(inputs.labels_t["eligible_from"], dtype=np.int16)
    hit = np.asarray(current_state["hit"], dtype=bool)
    first_rank = np.asarray(current_state["first_rank"], dtype=np.int16)
    first_turn = np.asarray(current_state["first_turn"], dtype=np.int16)
    hard_session = np.zeros(len(inputs.t_sessions), dtype=bool)
    control_session = np.zeros(len(inputs.t_sessions), dtype=bool)
    allowed_turn = np.zeros_like(positive, dtype=bool)
    for local_session in range(len(inputs.t_sessions)):
        eligible_index = int(eligible_from[local_session]) - 1
        if not 0 <= eligible_index < base.TURN_COUNT:
            raise StrictRestackBuildError("eligible turn is outside the horizon")
        for turn in range(eligible_index, base.TURN_COUNT):
            target = int(positive[local_session, turn])
            allowed_turn[local_session, turn] = target >= 0 and (
                target == int(incumbent[local_session, turn])
                or 10 <= target < base.CANDIDATE_COUNT
            )
        if not hit[local_session]:
            hard_session[local_session] = bool(np.any(allowed_turn[local_session]))
        elif int(first_rank[local_session]) == 10:
            stop = min(int(first_turn[local_session]), base.TURN_COUNT)
            control_session[local_session] = bool(
                np.any(allowed_turn[local_session, eligible_index:stop])
            )
    local_sessions = []
    turns = []
    hard_flags = []
    for local_session in range(len(inputs.t_sessions)):
        if hard_session[local_session]:
            stop = base.TURN_COUNT
            is_hard = True
        elif control_session[local_session]:
            stop = min(int(first_turn[local_session]), base.TURN_COUNT)
            is_hard = False
        else:
            continue
        eligible_index = int(eligible_from[local_session]) - 1
        for turn in range(eligible_index, stop):
            if allowed_turn[local_session, turn]:
                local_sessions.append(local_session)
                turns.append(turn)
                hard_flags.append(is_hard)
    local = np.asarray(local_sessions, dtype=np.int16)
    turn_array = np.asarray(turns, dtype=np.uint8)
    hard = np.asarray(hard_flags, dtype=np.uint8)
    if (
        not len(local)
        or not np.any(hard)
        or not np.any(~hard.astype(bool))
        or np.any(hard_session & control_session)
    ):
        raise StrictRestackBuildError("T_o focused cohorts are invalid")
    return FocusedGroups(
        local,
        inputs.t_sessions[local].astype(np.int16),
        turn_array,
        hard,
        hard_session,
        control_session,
    )


def _materialize_focused(
    inputs: OuterInputs,
    groups: FocusedGroups,
    incumbent: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    count = len(groups.local_session)
    x = np.empty(
        (count, focused.ROWS_PER_GROUP, base.FEATURE_COUNT), dtype=np.float32
    )
    y = np.empty((count, focused.ROWS_PER_GROUP), dtype=np.uint8)
    positive = np.asarray(inputs.labels_t["positive_index"], dtype=np.int16)
    for group in range(count):
        local_session = int(groups.local_session[group])
        global_session = int(groups.global_session[group])
        turn = int(groups.turn[group])
        current = int(incumbent[local_session, turn])
        allowed = np.asarray(
            [current, *range(10, base.CANDIDATE_COUNT)], dtype=np.int16
        )
        x[group] = inputs.projected_features[global_session, turn, allowed]
        y[group] = (allowed == int(positive[local_session, turn])).astype(np.uint8)
    if not np.isfinite(x).all() or not np.all(y.sum(axis=1) == 1):
        raise StrictRestackBuildError("focused cache parity failed")
    metadata = {
        "session_ordinal": groups.global_session,
        "turn_index": groups.turn,
        "hard_cohort": groups.hard,
        "outer_fold": inputs.outer_fold[groups.global_session],
        "inner_fold": inputs.inner_fold[groups.global_session],
    }
    files = {
        "features": _write_array(output_dir / "focused_features.npy", x),
        "relevance": _write_array(output_dir / "focused_relevance.npy", y),
    }
    for name, value in metadata.items():
        files[name] = _write_array(output_dir / (name + ".npy"), value)
    return x, y, metadata, {
        "query_groups": count,
        "rows": int(count * focused.ROWS_PER_GROUP),
        "hard_sessions": int(groups.hard_session.sum()),
        "control_sessions": int(groups.control_session.sum()),
        "hard_groups": int(groups.hard.sum()),
        "control_groups": int((~groups.hard.astype(bool)).sum()),
        "files": files,
    }


def _train_generic_domain(
    model_specs: Mapping[str, Mapping[str, Any]],
    selected: base.SelectedTrainingRows,
    train_sessions: np.ndarray,
    score_sessions: np.ndarray,
    inputs: OuterInputs,
    domain: str,
    domain_code: int,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    import xgboost as xgb

    train_mask = np.asarray(train_sessions, dtype=bool)
    if train_mask.shape != (base.SESSION_COUNT,):
        raise StrictRestackBuildError("generic train-session mask mismatch")
    train_rows = train_mask[selected.session]
    row_indices = np.flatnonzero(train_rows)
    qid = selected.qid[row_indices]
    boundaries = np.r_[0, np.flatnonzero(np.diff(qid) != 0) + 1, len(qid)]
    group_sizes = np.diff(boundaries).astype(np.uint32)
    if (
        not len(row_indices)
        or base.validate_grouped_qid(qid, selected.y[row_indices])
        != len(group_sizes)
        or np.any(~train_mask[selected.session[row_indices]])
    ):
        raise StrictRestackBuildError("generic grouped partition is invalid")
    matrix = xgb.DMatrix(
        np.asarray(selected.x[row_indices], dtype=np.float32),
        label=np.asarray(selected.y[row_indices], dtype=np.float32),
        nthread=1,
    )
    matrix.set_group(group_sizes)
    incumbent = probe._incumbent_for_sessions(
        inputs.projected_features, score_sessions
    )
    scores_by_model: dict[str, np.ndarray] = {}
    records = []
    for model_id in GENERIC_MODEL_IDS:
        spec = model_specs[model_id]
        seed = probe._model_seed(model_id, domain_code)
        rounds = int(spec["rounds"])
        started = time.perf_counter()
        booster = xgb.train(
            probe._generic_params(spec, seed), matrix, num_boost_round=rounds
        )
        training_seconds = time.perf_counter() - started
        if (
            int(booster.num_features()) != base.FEATURE_COUNT
            or int(booster.num_boosted_rounds()) != rounds
        ):
            raise StrictRestackBuildError("generic model contract failed")
        model_path = output_dir / "models" / (model_id + "__" + domain + ".json")
        model_record = probe._save_booster_exclusive(booster, model_path)
        scores, prediction = probe._score_booster(
            booster, inputs.projected_features, score_sessions
        )
        choice, _margin, _gap = probe.choose_slot10_any(scores, incumbent)
        score_record = _write_array(
            output_dir / "scores" / (model_id + "__" + domain + ".npy"),
            scores,
        )
        choice_record = _write_array(
            output_dir / "choices" / (model_id + "__" + domain + ".npy"),
            choice,
        )
        parity = probe._serialized_model_parity(
            model_path,
            inputs.projected_features,
            score_sessions,
            scores,
        )
        rss, peak = focused._rss()
        record = {
            "model_id": model_id,
            "domain": domain,
            "seed": int(seed),
            "rounds": rounds,
            "train_sessions": int(train_mask.sum()),
            "train_session_mask_sha256": _array_sha256(
                train_mask.astype(np.uint8)
            ),
            "train_rows": int(len(row_indices)),
            "train_query_groups": int(len(group_sizes)),
            "score_sessions": int(len(score_sessions)),
            "score_session_sha256": _array_sha256(score_sessions),
            "training_seconds": round(training_seconds, 6),
            "prediction": prediction,
            "rss_bytes_after_fit": int(rss),
            "peak_working_set_bytes": int(peak),
            "model": model_record,
            "score": score_record,
            "choice": choice_record,
            "serialized_model_parity": parity,
        }
        scores_by_model[model_id] = scores
        records.append(record)
        del booster
        gc.collect()
    del matrix
    gc.collect()
    return scores_by_model, records


def _train_focused_domain(
    x: np.ndarray,
    y: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    selected_groups: np.ndarray,
    parent_sessions: np.ndarray,
    inputs: OuterInputs,
    score_sessions: np.ndarray,
    domain: str,
    domain_code: int,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    selected_groups = np.asarray(selected_groups, dtype=bool)
    parent_sessions = np.asarray(parent_sessions, dtype=bool)
    if selected_groups.shape != (len(x),) or parent_sessions.shape != (
        base.SESSION_COUNT,
    ):
        raise StrictRestackBuildError("focused partition shape mismatch")
    group_sessions = np.asarray(metadata["session_ordinal"], dtype=np.int16)
    if np.any(selected_groups & ~parent_sessions[group_sessions]):
        raise StrictRestackBuildError("focused cohort escaped parent domain")
    weights = focused.partition_group_weights(
        group_sessions,
        np.asarray(metadata["hard_cohort"], dtype=np.uint8),
        selected_groups,
    )
    selected_hard = np.asarray(metadata["hard_cohort"], dtype=bool)[selected_groups]
    seed = probe._model_seed("focused_ndcg_d3", domain_code)
    model_path = output_dir / "models" / ("focused_ndcg_d3__" + domain + ".json")
    booster, audit = focused._train_model(
        x,
        y,
        {
            "session_ordinal": metadata["session_ordinal"],
            "hard_cohort": metadata["hard_cohort"],
        },
        selected_groups,
        seed,
        model_path,
    )
    scores, prediction = probe._score_booster(
        booster, inputs.projected_features, score_sessions
    )
    incumbent = probe._incumbent_for_sessions(
        inputs.projected_features, score_sessions
    )
    choice, _margin, _gap = probe.choose_slot10_any(scores, incumbent)
    score_record = _write_array(
        output_dir / "scores" / ("focused_ndcg_d3__" + domain + ".npy"),
        scores,
    )
    choice_record = _write_array(
        output_dir / "choices" / ("focused_ndcg_d3__" + domain + ".npy"),
        choice,
    )
    parity = probe._serialized_model_parity(
        model_path,
        inputs.projected_features,
        score_sessions,
        scores,
    )
    actual_sessions = np.unique(group_sessions[selected_groups])
    actual_mask = np.isin(ALL_SESSIONS, actual_sessions)
    model_record = {
        "path": model_path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(model_path),
        "bytes": model_path.stat().st_size,
        "asin_shape_matches": base._identity_shape_scan(model_path),
    }
    record = {
        "model_id": "focused_ndcg_d3",
        "domain": domain,
        "seed": int(seed),
        "rounds": focused.ROUNDS,
        "train_sessions": int(len(actual_sessions)),
        "train_session_mask_sha256": _array_sha256(actual_mask.astype(np.uint8)),
        "parent_domain_mask_sha256": _array_sha256(
            parent_sessions.astype(np.uint8)
        ),
        "train_rows": int(selected_groups.sum() * focused.ROWS_PER_GROUP),
        "train_query_groups": int(selected_groups.sum()),
        "score_sessions": int(len(score_sessions)),
        "score_session_sha256": _array_sha256(score_sessions),
        "training_seconds": float(audit["train_seconds"]),
        "prediction": prediction,
        "rss_bytes_after_fit": int(audit["rss_bytes_after_fit"]),
        "peak_working_set_bytes": int(audit["peak_working_set_bytes"]),
        "cohort_weight": {
            "hard_query_groups": int(selected_hard.sum()),
            "control_query_groups": int((~selected_hard).sum()),
            "hard_weight_sum": float(audit["hard_weight_sum"]),
            "control_weight_sum": float(audit["control_weight_sum"]),
            "mean": float(audit["group_weight_mean"]),
        },
        "model": model_record,
        "score": score_record,
        "choice": choice_record,
        "serialized_model_parity": parity,
    }
    subset = {
        "query_groups": int(selected_groups.sum()),
        "hard_sessions": int(
            len(np.unique(group_sessions[selected_groups & np.asarray(metadata["hard_cohort"], dtype=bool)]))
        ),
        "control_sessions": int(
            len(np.unique(group_sessions[selected_groups & ~np.asarray(metadata["hard_cohort"], dtype=bool)]))
        ),
        "hard_groups": int(selected_hard.sum()),
        "control_groups": int((~selected_hard).sum()),
        "group_weight_sha256": _array_sha256(weights),
        "arrays": {
            "features": _array_sha256(np.asarray(x[selected_groups])),
            "relevance": _array_sha256(np.asarray(y[selected_groups])),
            **{
                name: _array_sha256(np.asarray(value)[selected_groups])
                for name, value in metadata.items()
            },
        },
    }
    del booster
    gc.collect()
    return scores, record, subset


def _candidate_feature_for_sessions(
    features: np.ndarray,
    feature_sessions: np.ndarray,
    candidates: np.ndarray,
    available: np.ndarray,
    feature_name: str,
) -> np.ndarray:
    sessions = np.asarray(feature_sessions, dtype=np.int64)[:, None, None]
    turns = np.arange(candidates.shape[1], dtype=np.int64)[None, :, None]
    safe = np.where(available, candidates, 0).astype(np.int64)
    values = features[
        sessions,
        turns,
        safe,
        base.FEATURE_INDEX[feature_name],
    ]
    return np.where(available, values, 0.0).astype(np.float32)


def _served_feature_for_sessions(
    features: np.ndarray,
    feature_sessions: np.ndarray,
    current_choice: np.ndarray,
    feature_name: str,
) -> np.ndarray:
    sessions = np.asarray(feature_sessions, dtype=np.int64)[:, None]
    turns = np.arange(current_choice.shape[1], dtype=np.int64)[None, :]
    return np.asarray(
        features[
            sessions,
            turns,
            current_choice,
            base.FEATURE_INDEX[feature_name],
        ],
        dtype=np.float32,
    )


def _conflict_sum_for_sessions(
    features: np.ndarray,
    feature_sessions: np.ndarray,
    candidates: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    result = np.zeros(candidates.shape, dtype=np.float32)
    for slot_name in base.CONSTRAINT_SLOTS:
        result += _candidate_feature_for_sessions(
            features,
            feature_sessions,
            candidates,
            available,
            "%s_conflict" % slot_name,
        )
    return result


def _served_conflict_for_sessions(
    features: np.ndarray,
    feature_sessions: np.ndarray,
    current_choice: np.ndarray,
) -> np.ndarray:
    result = np.zeros(current_choice.shape, dtype=np.float32)
    for slot_name in base.CONSTRAINT_SLOTS:
        result += _served_feature_for_sessions(
            features,
            feature_sessions,
            current_choice,
            "%s_conflict" % slot_name,
        )
    return result


def _build_runtime_surface_any(
    features: np.ndarray,
    feature_sessions: np.ndarray,
    current_scores: np.ndarray,
    family_scores: Sequence[np.ndarray],
    current_chosen: np.ndarray,
    current_activation: np.ndarray,
    incumbent: np.ndarray,
) -> portfolio.RuntimePortfolioSurface:
    if len(family_scores) != len(portfolio.FAMILY_NAMES):
        raise StrictRestackBuildError("portfolio requires three score surfaces")
    expected = current_scores.shape
    if (
        expected != (*current_chosen.shape, base.CANDIDATE_COUNT)
        or current_activation.shape != current_chosen.shape
        or incumbent.shape != current_chosen.shape
        or np.asarray(feature_sessions).shape != (current_chosen.shape[0],)
        or features.shape
        != (
            base.SESSION_COUNT,
            base.TURN_COUNT,
            base.CANDIDATE_COUNT,
            base.FEATURE_COUNT,
        )
        or any(np.asarray(scores).shape != expected for scores in family_scores)
    ):
        raise StrictRestackBuildError("runtime portfolio input shape mismatch")
    family_choices = np.stack(
        [
            probe.choose_slot10_any(np.asarray(scores), incumbent)[0]
            for scores in family_scores
        ],
        axis=2,
    ).astype(np.uint8)
    current_choice = np.where(
        current_activation, current_chosen, incumbent
    ).astype(np.uint8)
    candidates, source_mask, available = portfolio._deduplicate_actions(
        family_choices, current_choice, incumbent
    )
    shape = (*available.shape, len(portfolio.FEATURE_NAMES))
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
        available.sum(axis=2, dtype=np.float32) / float(portfolio.MAX_ACTIONS)
    )[..., None]
    for family in range(len(portfolio.FAMILY_NAMES)):
        gate_features[..., 4 + family] = (
            source_mask & (1 << family) != 0
        ).astype(np.float32)
    all_scores = (current_scores, *family_scores)
    for index, scores in enumerate(all_scores):
        gate_features[..., 7 + index] = portfolio._rank_fraction_for_actions(
            scores, candidates, available, incumbent
        )
    for index, scores in enumerate(family_scores):
        current_rank = portfolio._allowed_rank_fraction(
            scores, current_choice, incumbent
        )
        gate_features[..., 11 + index] = current_rank[..., None]
    gate_features[..., 14] = _candidate_feature_for_sessions(
        features,
        feature_sessions,
        candidates,
        available,
        "coverage_rank_fraction",
    )
    for offset, feature_name in enumerate(
        (
            "top10_route_agreement_fraction",
            "active_token_recall",
            "hard_clause_coverage",
        ),
        start=15,
    ):
        candidate_value = _candidate_feature_for_sessions(
            features, feature_sessions, candidates, available, feature_name
        )
        served_value = _served_feature_for_sessions(
            features, feature_sessions, current_choice, feature_name
        )
        gate_features[..., offset] = np.where(
            available, candidate_value - served_value[..., None], 0.0
        )
    candidate_conflict = _conflict_sum_for_sessions(
        features, feature_sessions, candidates, available
    )
    served_conflict = _served_conflict_for_sessions(
        features, feature_sessions, current_choice
    )
    gate_features[..., 18] = np.where(
        available, candidate_conflict - served_conflict[..., None], 0.0
    )
    gate_features = np.where(
        available[..., None], gate_features, 0.0
    ).astype(np.float32)
    if gate_features.shape != shape or not np.isfinite(gate_features).all():
        raise StrictRestackBuildError("portfolio feature schema mismatch")
    return portfolio.RuntimePortfolioSurface(
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


RUNTIME_FIELDS = (
    "current_chosen",
    "current_activation",
    "current_choice",
    "incumbent",
    "family_choices",
    "candidates",
    "source_mask",
    "available",
    "features",
)


def _slice_runtime(
    runtime: portfolio.RuntimePortfolioSurface, positions: np.ndarray
) -> portfolio.RuntimePortfolioSurface:
    values = {
        name: np.asarray(getattr(runtime, name))[positions]
        for name in RUNTIME_FIELDS
    }
    return portfolio.RuntimePortfolioSurface(**values)


def _write_runtime(
    output_dir: Path, runtime: portfolio.RuntimePortfolioSurface
) -> dict[str, Any]:
    files = {
        name: _write_array(output_dir / (name + ".npy"), getattr(runtime, name))
        for name in RUNTIME_FIELDS
    }
    return {
        "sessions": int(runtime.current_chosen.shape[0]),
        "available_action_rows": int(runtime.available.sum()),
        "feature_order_sha256": _canonical_sha256(
            list(portfolio.FEATURE_NAMES)
        ),
        "files": files,
    }


def _write_portfolio_labels(
    output_dir: Path, surface: portfolio.PortfolioSurface
) -> dict[str, Any]:
    fields = (
        "rescue",
        "rescue_weights",
        "regret",
        "regret_weights",
        "rr_loss",
        "mttc_loss",
    )
    return {
        "files": {
            name: _write_array(output_dir / (name + ".npy"), getattr(surface, name))
            for name in fields
        },
        "rescue_rows": int(surface.rescue.sum()),
        "regret_rows": int(surface.regret.sum()),
    }


def _strict_selector(
    training: portfolio.PortfolioSurface,
    labels_t: Mapping[str, np.ndarray],
    current_state: Mapping[str, np.ndarray],
    reference: portfolio.RuntimePortfolioSurface,
    held: portfolio.RuntimePortfolioSurface,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    inner = np.asarray(labels_t["inner_fold"], dtype=np.uint8)
    family = np.asarray(labels_t["family_index"], dtype=np.int32)
    flat_x = training.features.reshape(-1, len(portfolio.FEATURE_NAMES))
    flat_available = training.available.reshape(-1)
    flat_session = np.repeat(
        np.arange(len(inner)), base.TURN_COUNT * portfolio.MAX_ACTIONS
    )
    flat_family = np.repeat(
        family, base.TURN_COUNT * portfolio.MAX_ACTIONS
    )
    targets = (training.rescue.reshape(-1), training.regret.reshape(-1))
    weights = (
        training.rescue_weights.reshape(-1),
        training.regret_weights.reshape(-1),
    )
    inner_probability = [
        np.zeros(training.available.shape, dtype=np.float32) for _ in range(2)
    ]
    coverage = np.zeros_like(flat_available, dtype=np.uint8)
    readiness_records = []
    blocked = False
    for inner_fold in range(base.OUTER_FOLDS):
        train_sessions = inner != inner_fold
        valid_sessions = inner == inner_fold
        train_rows = flat_available & train_sessions[flat_session]
        valid_rows = flat_available & valid_sessions[flat_session]
        readiness = portfolio._fit_readiness(
            train_rows,
            targets[0],
            targets[1],
            flat_session,
            flat_family,
        )
        readiness["inner_fold"] = inner_fold
        readiness["valid_action_rows"] = int(valid_rows.sum())
        head_hashes = []
        if not readiness["ready"] or not np.any(valid_rows):
            blocked = True
        else:
            coverage[valid_rows] += 1
            for head in range(2):
                model, mean, scale = portfolio._fit_gate_model(
                    flat_x[train_rows],
                    targets[head][train_rows],
                    weights[head][train_rows],
                    portfolio.MODEL_SEED,
                )
                portfolio._validate_fitted_model(model)
                inner_probability[head].reshape(-1)[valid_rows] = (
                    base._predict_gate(
                        model, mean, scale, flat_x[valid_rows]
                    ).astype(np.float32)
                )
                head_hashes.append(
                    probe._gate_model_sha256(model, mean, scale)
                )
        readiness["head_model_sha256"] = head_hashes
        readiness_records.append(readiness)
    if not blocked and (
        not np.all(coverage[flat_available] == 1)
        or np.any(coverage[~flat_available])
    ):
        raise StrictRestackBuildError("selector inner coverage is invalid")
    selected: dict[str, Any]
    outer_readiness: Optional[dict[str, Any]] = None
    outer_heads: list[dict[str, Any]] = []
    mapped_threshold = math.inf
    reference_probability = [
        np.zeros(reference.available.shape, dtype=np.float32) for _ in range(2)
    ]
    held_probability = [
        np.zeros(held.available.shape, dtype=np.float32) for _ in range(2)
    ]
    supplement = np.zeros(held.current_chosen.shape, dtype=bool)
    supplemental_choice = np.full(
        held.current_chosen.shape, -1, dtype=np.int16
    )
    if blocked:
        selected = {
            "quantile": frozen.KEEP_QUANTILE,
            "status": "KEEP_INSUFFICIENT_INNER_FIT",
        }
    else:
        inner_utility = (
            inner_probability[0]
            - portfolio.REGRET_MULTIPLIER * inner_probability[1]
        )
        selected = portfolio._select_inner_quantile(
            training,
            inner_utility,
            labels_t,
            current_state,
            np.ones(len(inner), dtype=bool),
            inner,
        )
        if float(selected["quantile"]) < frozen.KEEP_QUANTILE:
            outer_readiness = portfolio._fit_readiness(
                flat_available,
                targets[0],
                targets[1],
                flat_session,
                flat_family,
            )
            if outer_readiness["ready"]:
                reference_flat = reference.features.reshape(
                    -1, len(portfolio.FEATURE_NAMES)
                )
                held_flat = held.features.reshape(
                    -1, len(portfolio.FEATURE_NAMES)
                )
                reference_available = reference.available.reshape(-1)
                held_available = held.available.reshape(-1)
                if not np.any(reference_available):
                    raise StrictRestackBuildError(
                        "finite selector quantile has no reference actions"
                    )
                for head in range(2):
                    model, mean, scale = portfolio._fit_gate_model(
                        flat_x[flat_available],
                        targets[head][flat_available],
                        weights[head][flat_available],
                        portfolio.MODEL_SEED,
                    )
                    portfolio._validate_fitted_model(model)
                    reference_probability[head].reshape(-1)[
                        reference_available
                    ] = base._predict_gate(
                        model, mean, scale, reference_flat[reference_available]
                    ).astype(np.float32)
                    if np.any(held_available):
                        held_probability[head].reshape(-1)[held_available] = (
                            base._predict_gate(
                                model, mean, scale, held_flat[held_available]
                            ).astype(np.float32)
                        )
                    outer_heads.append(
                        {
                            "head": head,
                            "seed": portfolio.MODEL_SEED,
                            "model_sha256": probe._gate_model_sha256(
                                model, mean, scale
                            ),
                        }
                    )
                reference_utility = (
                    reference_probability[0]
                    - portfolio.REGRET_MULTIPLIER * reference_probability[1]
                )
                held_utility = (
                    held_probability[0]
                    - portfolio.REGRET_MULTIPLIER * held_probability[1]
                )
                (
                    _slot,
                    _candidate,
                    reference_winner,
                    reference_winner_available,
                ) = portfolio._per_turn_winner_utilities(
                    reference.candidates,
                    reference.source_mask,
                    reference.available,
                    reference_utility,
                )
                mapped_threshold = portfolio._map_outer_quantile(
                    reference_winner,
                    reference_winner_available,
                    np.ones(len(reference.current_chosen), dtype=bool),
                    float(selected["quantile"]),
                )
                supplement, supplemental_choice = portfolio._causal_policy(
                    held.candidates,
                    held.source_mask,
                    held.available,
                    held_utility,
                    mapped_threshold,
                    np.ones(len(held.current_chosen), dtype=bool),
                )
                selected = {**selected, "status": "FINITE_SELECTED"}
            else:
                selected = {
                    **selected,
                    "proposed_quantile": float(selected["quantile"]),
                    "quantile": frozen.KEEP_QUANTILE,
                    "status": "KEEP_INSUFFICIENT_OUTER_FIT",
                }
        else:
            selected = {**selected, "status": "KEEP_SELECTED"}
    reference_utility = (
        reference_probability[0]
        - portfolio.REGRET_MULTIPLIER * reference_probability[1]
    )
    held_utility = (
        held_probability[0]
        - portfolio.REGRET_MULTIPLIER * held_probability[1]
    )
    final_chosen, final_activation = portfolio._compose_policy(
        held.current_chosen,
        held.current_activation,
        held.candidates,
        held.available,
        supplement,
        supplemental_choice,
    )
    files = {
        "inner_rescue_probability": _write_array(
            output_dir / "inner_rescue_probability.npy", inner_probability[0]
        ),
        "inner_regret_probability": _write_array(
            output_dir / "inner_regret_probability.npy", inner_probability[1]
        ),
        "inner_coverage": _write_array(output_dir / "inner_coverage.npy", coverage),
        "reference_rescue_probability": _write_array(
            output_dir / "reference_rescue_probability.npy",
            reference_probability[0],
        ),
        "reference_regret_probability": _write_array(
            output_dir / "reference_regret_probability.npy",
            reference_probability[1],
        ),
        "reference_utility": _write_array(
            output_dir / "reference_utility.npy", reference_utility
        ),
        "held_rescue_probability": _write_array(
            output_dir / "held_rescue_probability.npy", held_probability[0]
        ),
        "held_regret_probability": _write_array(
            output_dir / "held_regret_probability.npy", held_probability[1]
        ),
        "held_utility": _write_array(
            output_dir / "held_utility.npy", held_utility
        ),
        "supplement": _write_array(output_dir / "supplement.npy", supplement),
        "supplemental_choice": _write_array(
            output_dir / "supplemental_choice.npy", supplemental_choice
        ),
        "final_chosen": _write_array(
            output_dir / "final_chosen.npy", final_chosen
        ),
        "final_activation": _write_array(
            output_dir / "final_activation.npy", final_activation
        ),
    }
    record = {
        "selected_quantile": float(selected["quantile"]),
        "mapped_reference_threshold": (
            float(mapped_threshold) if np.isfinite(mapped_threshold) else "KEEP"
        ),
        "inner_selection": selected,
        "inner_fit_readiness": readiness_records,
        "inner_oof_coverage_sha256": _array_sha256(coverage),
        "outer_fit_readiness": outer_readiness,
        "outer_head_models": outer_heads,
        "files": files,
        "supplement_turns": int(supplement.sum()),
        "supplement_sessions": int(np.any(supplement, axis=1).sum()),
    }
    return final_chosen, final_activation, record


def _first_difference(left: object, right: object, path: str = "root") -> str:
    if type(left) is not type(right):
        return "%s:type(%s!=%s)" % (path, type(left).__name__, type(right).__name__)
    if isinstance(left, Mapping):
        left_keys = set(left)
        right_keys = set(right)  # type: ignore[arg-type]
        if left_keys != right_keys:
            return "%s:keys(%s!=%s)" % (
                path,
                sorted(left_keys),
                sorted(right_keys),
            )
        for key in sorted(left_keys):
            child = _first_difference(
                left[key], right[key], "%s.%s" % (path, key)  # type: ignore[index]
            )
            if child:
                return child
        return ""
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):  # type: ignore[arg-type]
            return "%s:length" % path
        for index, (left_value, right_value) in enumerate(zip(left, right)):  # type: ignore[arg-type]
            child = _first_difference(
                left_value, right_value, "%s[%d]" % (path, index)
            )
            if child:
                return child
        return ""
    return "" if left == right else "%s:value(%r!=%r)" % (path, left, right)


def _stage0_prefix_parity(
    pass_name: str,
    generic_records: Sequence[Mapping[str, Any]],
    focused_records: Sequence[Mapping[str, Any]],
    current_files: Mapping[str, Mapping[str, Any]],
    current_gate: Mapping[str, Any],
    focused_subsets: Mapping[str, Mapping[str, Any]],
    rrf_score: Mapping[str, Any],
    rrf_choice: Mapping[str, Any],
) -> dict[str, Any]:
    stage0_manifest = _load_json(STAGE0_MANIFEST)
    result_path = ROOT / str(stage0_manifest["result"]["path"])
    stage0 = _load_json(result_path)
    expected_pass = next(
        row for row in stage0["passes"] if row["name"] == pass_name
    )
    expected = probe._repeat_identity(expected_pass)
    generic_map = {
        (str(row["model_id"]), str(row["domain"])): row
        for row in generic_records
    }
    focused_map = {str(row["domain"]): row for row in focused_records}
    model_records = [
        generic_map[("current_ndcg_d4_lr003", "A_0%d" % inner)]
        for inner in range(base.OUTER_FOLDS)
    ]
    model_records.extend(
        generic_map[(model_id, "A_00")]
        for model_id in (
            "pairwise_d4_control",
            "ndcg_d6_lr006",
            "ndcg_d4_regularized",
        )
    )
    model_records.append(focused_map["A_00"])
    subset = focused_subsets["A_00"]
    virtual = {
        "models": model_records,
        "current": {"files": dict(current_files), "gate": dict(current_gate)},
        "focused_cache": {
            "query_groups": subset["query_groups"],
            "hard_sessions": subset["hard_sessions"],
            "control_sessions": subset["control_sessions"],
            "hard_groups": subset["hard_groups"],
            "control_groups": subset["control_groups"],
            "group_weight_sha256": subset["group_weight_sha256"],
            "files": {
                key: {"array_sha256": value}
                for key, value in subset["arrays"].items()
            },
        },
        "rrf3": {"score": dict(rrf_score), "choice": dict(rrf_choice)},
    }
    actual = probe._repeat_identity(virtual)
    expected_sha = _canonical_sha256(expected)
    actual_sha = _canonical_sha256(actual)
    if (
        expected_sha
        != "f3f1c23b10d3126e7cc31b26d384e40a4d4d7e36c0681bc48a455db538655bf2"
        or actual_sha != expected_sha
        or actual != expected
    ):
        raise StrictRestackBuildError(
            "Stage-0 prefix parity failed: %s"
            % _first_difference(actual, expected)
        )
    return {
        "equal": True,
        "expected_identity_sha256": expected_sha,
        "actual_identity_sha256": actual_sha,
        "models_compared": len(actual["models"]),
        "stage0_pass": pass_name,
    }


def _model_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": record["model_id"],
        "domain": record["domain"],
        "seed": record["seed"],
        "rounds": record["rounds"],
        "train_sessions": record["train_sessions"],
        "train_session_mask_sha256": record["train_session_mask_sha256"],
        "parent_domain_mask_sha256": record.get("parent_domain_mask_sha256"),
        "train_rows": record["train_rows"],
        "train_query_groups": record["train_query_groups"],
        "score_sessions": record["score_sessions"],
        "score_session_sha256": record["score_session_sha256"],
        "model_sha256": record["model"]["sha256"],
        "score_array_sha256": record["score"]["array_sha256"],
        "choice_array_sha256": record["choice"]["array_sha256"],
        "serialized_model_parity": record["serialized_model_parity"],
        "cohort_weight": record.get("cohort_weight"),
    }


def _file_array_hashes(files: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {key: str(value["array_sha256"]) for key, value in files.items()}


def _record_without_files(
    record: Mapping[str, Any], *excluded: str
) -> dict[str, Any]:
    omitted = {"files", *excluded}
    return {key: value for key, value in record.items() if key not in omitted}


def run_outer(
    source_root: Path,
    projection_root: Path,
    output_dir: Path,
    pass_name: str,
    outer_fold: int,
) -> dict[str, Any]:
    if pass_name not in {"first", "repeat"}:
        raise StrictRestackBuildError("pass name must be first or repeat")
    output_dir = output_dir.resolve()
    experiments_root = (ROOT / "experiments").resolve()
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or experiments_root not in output_dir.parents
    ):
        raise StrictRestackBuildError(
            "output directory must be new and below this worktree's experiments"
        )
    started = time.perf_counter()
    environment = _validate_environment()
    prereg, amendment, stage0_manifest = _validate_protocol()
    source_snapshot = _source_snapshot()
    inputs = _load_outer_inputs(source_root, projection_root, outer_fold)
    domains = _domain_records(inputs, outer_fold)
    output_dir.mkdir(parents=True)
    selected_started = time.perf_counter()
    selected = _build_selected_domain(inputs)
    selected_record = {
        "rows": int(len(selected.y)),
        "query_groups": int(selected.group_count),
        "build_seconds": round(time.perf_counter() - selected_started, 6),
        "feature_array_sha256": _array_sha256(selected.x),
        "label_array_sha256": _array_sha256(selected.y),
        "qid_array_sha256": _array_sha256(selected.qid),
        "session_array_sha256": _array_sha256(selected.session),
        "held_outer_rows": int(
            np.sum(inputs.outer_fold[selected.session] == outer_fold)
        ),
    }
    if selected_record["held_outer_rows"] != 0:
        raise StrictRestackBuildError("H_o entered selected training rows")
    model_specs = {
        str(row["id"]): row for row in prereg["generic_training"]["models"]
    }
    t_mask = inputs.outer_fold != outer_fold
    oof_scores = {
        model_id: np.empty(
            (len(inputs.t_sessions), base.TURN_COUNT, base.CANDIDATE_COUNT),
            dtype=np.float32,
        )
        for model_id in GENERIC_MODEL_IDS
    }
    oof_coverage = np.zeros(len(inputs.t_sessions), dtype=np.uint8)
    generic_records: list[dict[str, Any]] = []
    for inner_fold in range(base.OUTER_FOLDS):
        train_mask = t_mask & (inputs.inner_fold != inner_fold)
        valid_sessions = np.flatnonzero(
            t_mask & (inputs.inner_fold == inner_fold)
        ).astype(np.int16)
        domain = "A_%d%d" % (outer_fold, inner_fold)
        scores, records = _train_generic_domain(
            model_specs,
            selected,
            train_mask,
            valid_sessions,
            inputs,
            domain,
            5 * outer_fold + inner_fold,
            output_dir / "domains" / domain,
        )
        positions = np.searchsorted(inputs.t_sessions, valid_sessions)
        if not np.array_equal(inputs.t_sessions[positions], valid_sessions):
            raise StrictRestackBuildError("generic OOF score lineage is invalid")
        for model_id in GENERIC_MODEL_IDS:
            oof_scores[model_id][positions] = scores[model_id]
        oof_coverage[positions] += 1
        generic_records.extend(records)
        print(
            json.dumps(
                {
                    "phase": "generic_A",
                    "pass": pass_name,
                    "outer": outer_fold,
                    "inner": inner_fold,
                    "models_complete": len(generic_records),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not np.all(oof_coverage == 1) or any(
        not np.isfinite(scores).all() for scores in oof_scores.values()
    ):
        raise StrictRestackBuildError("generic OOF assembly failed")
    oof_score_files = {
        model_id: _write_array(
            output_dir / "portfolio" / "oof_scores" / (model_id + ".npy"),
            scores,
        )
        for model_id, scores in oof_scores.items()
    }
    current_surface = _training_surface(
        inputs, oof_scores["current_ndcg_d4_lr003"]
    )
    current_state, current_activation, current_inner = _crossfit_current(
        current_surface, inputs.labels_t, outer_fold
    )
    current_inner_probability = current_inner.pop("probabilities")
    current_inner_utility = current_inner.pop("utility")
    current_files = {
        "scores": _write_array(
            output_dir / "current" / "current_oof_scores.npy",
            oof_scores["current_ndcg_d4_lr003"],
        ),
        "chosen": _write_array(
            output_dir / "current" / "current_oof_chosen.npy",
            current_surface.chosen,
        ),
        "action": _write_array(
            output_dir / "current" / "current_oof_action.npy",
            current_surface.action.astype(np.uint8),
        ),
        "gate_features": _write_array(
            output_dir / "current" / "current_oof_gate_features.npy",
            current_surface.gate_features,
        ),
        "activation": _write_array(
            output_dir / "current" / "current_oof_activation.npy",
            current_activation.astype(np.uint8),
        ),
    }
    current_aux_files = {
        "rescue_probability": _write_array(
            output_dir / "current" / "current_oof_rescue_probability.npy",
            current_inner_probability[0],
        ),
        "regret_probability": _write_array(
            output_dir / "current" / "current_oof_regret_probability.npy",
            current_inner_probability[1],
        ),
        "utility": _write_array(
            output_dir / "current" / "current_oof_utility.npy",
            current_inner_utility,
        ),
        **{
            "state_" + name: _write_array(
                output_dir / "current" / ("current_oof_state_" + name + ".npy"),
                value,
            )
            for name, value in current_state.items()
        },
    }
    groups = _select_focused_groups(
        inputs, current_state, current_surface.incumbent
    )
    focused_x, focused_y, focused_metadata, focused_cache = _materialize_focused(
        inputs,
        groups,
        current_surface.incumbent,
        output_dir / "focused" / "cohort_T_o",
    )
    focused_oof = np.empty_like(
        oof_scores["current_ndcg_d4_lr003"], dtype=np.float32
    )
    focused_coverage = np.zeros(len(inputs.t_sessions), dtype=np.uint8)
    focused_records: list[dict[str, Any]] = []
    focused_subsets: dict[str, dict[str, Any]] = {}
    for inner_fold in range(base.OUTER_FOLDS):
        parent_mask = t_mask & (inputs.inner_fold != inner_fold)
        selected_groups = (
            np.asarray(focused_metadata["inner_fold"], dtype=np.uint8)
            != inner_fold
        )
        valid_sessions = np.flatnonzero(
            t_mask & (inputs.inner_fold == inner_fold)
        ).astype(np.int16)
        domain = "A_%d%d" % (outer_fold, inner_fold)
        scores, record, subset = _train_focused_domain(
            focused_x,
            focused_y,
            focused_metadata,
            selected_groups,
            parent_mask,
            inputs,
            valid_sessions,
            domain,
            5 * outer_fold + inner_fold,
            output_dir / "domains" / domain,
        )
        positions = np.searchsorted(inputs.t_sessions, valid_sessions)
        focused_oof[positions] = scores
        focused_coverage[positions] += 1
        focused_records.append(record)
        focused_subsets[domain] = subset
        print(
            json.dumps(
                {
                    "phase": "focused_A",
                    "pass": pass_name,
                    "outer": outer_fold,
                    "inner": inner_fold,
                    "models_complete": len(generic_records) + len(focused_records),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not np.all(focused_coverage == 1) or not np.isfinite(focused_oof).all():
        raise StrictRestackBuildError("focused OOF assembly failed")
    focused_oof_file = _write_array(
        output_dir / "portfolio" / "oof_scores" / "focused_ndcg_d3.npy",
        focused_oof,
    )
    rrf_oof = probe.rrf3_scores_any(
        [
            oof_scores["current_ndcg_d4_lr003"],
            oof_scores["ndcg_d6_lr006"],
            oof_scores["ndcg_d4_regularized"],
        ]
    )
    rrf_oof_file = _write_array(
        output_dir / "portfolio" / "oof_scores" / "rrf3.npy", rrf_oof
    )
    if outer_fold == 0:
        v00_positions = np.flatnonzero(
            np.asarray(inputs.labels_t["inner_fold"], dtype=np.uint8) == 0
        )
        v00_rrf = rrf_oof[v00_positions]
        v00_incumbent = current_surface.incumbent[v00_positions]
        v00_choice, _margin, _gap = probe.choose_slot10_any(
            v00_rrf, v00_incumbent
        )
        v00_rrf_record = _write_array(
            output_dir / "stage0_parity" / "rrf3_v00.npy", v00_rrf
        )
        v00_choice_record = _write_array(
            output_dir / "stage0_parity" / "rrf3_v00_choice.npy", v00_choice
        )
        stage0_parity = _stage0_prefix_parity(
            pass_name,
            generic_records,
            focused_records,
            current_files,
            current_inner,
            focused_subsets,
            v00_rrf_record,
            v00_choice_record,
        )
        stage0_parity_files = {
            "rrf3_score": v00_rrf_record,
            "rrf3_choice": v00_choice_record,
        }
    else:
        stage0_parity = {
            "applicable": False,
            "reason": "Stage-0 oracle covers outer 0 only",
        }
        stage0_parity_files = {}
    domain_t = "T_%d" % outer_fold
    full_generic, full_generic_records = _train_generic_domain(
        model_specs,
        selected,
        t_mask,
        ALL_SESSIONS,
        inputs,
        domain_t,
        25 + outer_fold,
        output_dir / "domains" / domain_t,
    )
    generic_records.extend(full_generic_records)
    full_focused, full_focused_record, full_focused_subset = (
        _train_focused_domain(
            focused_x,
            focused_y,
            focused_metadata,
            np.ones(len(focused_x), dtype=bool),
            t_mask,
            inputs,
            ALL_SESSIONS,
            domain_t,
            25 + outer_fold,
            output_dir / "domains" / domain_t,
        )
    )
    focused_records.append(full_focused_record)
    focused_subsets[domain_t] = full_focused_subset
    print(
        json.dumps(
            {
                "phase": "full_T",
                "pass": pass_name,
                "outer": outer_fold,
                "models_complete": len(generic_records) + len(focused_records),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if len(generic_records) != 24 or len(focused_records) != 6:
        raise StrictRestackBuildError("outer worker did not rebuild all 30 models")
    full_current_surface = _inference_surface(
        inputs.projected_features,
        ALL_SESSIONS,
        full_generic["current_ndcg_d4_lr003"],
    )
    full_current_activation, current_full = _fit_full_current_gate(
        current_surface,
        full_current_surface,
        outer_fold,
        float(current_inner["selected_quantile"]),
        t_mask,
    )
    current_full_probability = current_full.pop("probabilities")
    current_full_utility = current_full.pop("utility")
    current_full_files = {
        "chosen": _write_array(
            output_dir / "current" / "current_full_chosen.npy",
            full_current_surface.chosen,
        ),
        "action": _write_array(
            output_dir / "current" / "current_full_action.npy",
            full_current_surface.action,
        ),
        "gate_features": _write_array(
            output_dir / "current" / "current_full_gate_features.npy",
            full_current_surface.gate_features,
        ),
        "activation": _write_array(
            output_dir / "current" / "current_full_activation.npy",
            full_current_activation,
        ),
        "rescue_probability": _write_array(
            output_dir / "current" / "current_full_rescue_probability.npy",
            current_full_probability[0],
        ),
        "regret_probability": _write_array(
            output_dir / "current" / "current_full_regret_probability.npy",
            current_full_probability[1],
        ),
        "utility": _write_array(
            output_dir / "current" / "current_full_utility.npy",
            current_full_utility,
        ),
    }
    rrf_full = probe.rrf3_scores_any(
        [
            full_generic["current_ndcg_d4_lr003"],
            full_generic["ndcg_d6_lr006"],
            full_generic["ndcg_d4_regularized"],
        ]
    )
    rrf_full_file = _write_array(
        output_dir / "portfolio" / "full_scores" / "rrf3.npy", rrf_full
    )
    runtime_oof = _build_runtime_surface_any(
        inputs.projected_features,
        inputs.t_sessions,
        oof_scores["current_ndcg_d4_lr003"],
        [oof_scores["pairwise_d4_control"], rrf_oof, focused_oof],
        current_surface.chosen,
        current_activation,
        current_surface.incumbent,
    )
    runtime_full = _build_runtime_surface_any(
        inputs.projected_features,
        ALL_SESSIONS,
        full_generic["current_ndcg_d4_lr003"],
        [full_generic["pairwise_d4_control"], rrf_full, full_focused],
        full_current_surface.chosen,
        full_current_activation,
        full_current_surface.incumbent,
    )
    runtime_reference = _slice_runtime(runtime_full, inputs.t_sessions)
    runtime_held = _slice_runtime(runtime_full, inputs.h_sessions)
    runtime_records = {
        "oof_T": _write_runtime(
            output_dir / "portfolio" / "runtime" / "oof_T", runtime_oof
        ),
        "reference_T": _write_runtime(
            output_dir / "portfolio" / "runtime" / "reference_T",
            runtime_reference,
        ),
        "held_H": _write_runtime(
            output_dir / "portfolio" / "runtime" / "held_H", runtime_held
        ),
    }
    training_portfolio = portfolio._attach_isolated_labels(
        runtime_oof, inputs.labels_t
    )
    portfolio_label_record = _write_portfolio_labels(
        output_dir / "portfolio" / "training_labels", training_portfolio
    )
    final_chosen, final_activation, selector_record = _strict_selector(
        training_portfolio,
        inputs.labels_t,
        current_state,
        runtime_reference,
        runtime_held,
        output_dir / "selector",
    )
    held_files = {
        "session_ordinal": _write_array(
            output_dir / "held" / "session_ordinal.npy", inputs.h_sessions
        ),
        "current_chosen": _write_array(
            output_dir / "held" / "domain_local_current_chosen.npy",
            runtime_held.current_chosen,
        ),
        "current_activation": _write_array(
            output_dir / "held" / "domain_local_current_activation.npy",
            runtime_held.current_activation,
        ),
        "final_chosen": _write_array(
            output_dir / "held" / "final_chosen.npy", final_chosen
        ),
        "final_activation": _write_array(
            output_dir / "held" / "final_activation.npy", final_activation
        ),
    }
    model_identity = sorted(
        [_model_identity(row) for row in (*generic_records, *focused_records)],
        key=lambda row: (str(row["domain"]), str(row["model_id"])),
    )
    identity = {
        "outer_fold": outer_fold,
        "source_snapshot": source_snapshot,
        "dependencies": environment,
        "domains": domains,
        "selected_training_rows": {
            key: value
            for key, value in selected_record.items()
            if key != "build_seconds"
        },
        "models": model_identity,
        "oof_score_arrays": _file_array_hashes(oof_score_files),
        "focused_oof_array": focused_oof_file["array_sha256"],
        "oof_coverage_sha256": _array_sha256(oof_coverage),
        "focused_coverage_sha256": _array_sha256(focused_coverage),
        "current": {
            "oof_files": _file_array_hashes(current_files),
            "oof_aux_files": _file_array_hashes(current_aux_files),
            "inner_gate": current_inner,
            "full_files": _file_array_hashes(current_full_files),
            "full_gate": current_full,
        },
        "focused_cache": {
            "counts": {
                key: value
                for key, value in focused_cache.items()
                if key != "files"
            },
            "files": _file_array_hashes(focused_cache["files"]),
            "partitions": focused_subsets,
        },
        "rrf": {
            "oof": rrf_oof_file["array_sha256"],
            "full": rrf_full_file["array_sha256"],
        },
        "runtime": {
            key: {
                **_record_without_files(value),
                "files": _file_array_hashes(value["files"]),
            }
            for key, value in runtime_records.items()
        },
        "portfolio_training_labels": {
            **_record_without_files(portfolio_label_record),
            "files": _file_array_hashes(portfolio_label_record["files"]),
        },
        "selector": {
            **_record_without_files(selector_record),
            "files": _file_array_hashes(selector_record["files"]),
        },
        "held": _file_array_hashes(held_files),
        "stage0_prefix_parity": {
            key: value
            for key, value in stage0_parity.items()
            if key != "stage0_pass"
        },
    }
    peak = max(
        int(row["peak_working_set_bytes"])
        for row in (*generic_records, *focused_records)
    )
    final_source_snapshot = _source_snapshot()
    if final_source_snapshot != source_snapshot:
        raise StrictRestackBuildError(
            "source/config identity changed while outer worker was running"
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.8-STRICT-OUTER-RESTACK-STAGE1",
        "status": "OUTER_CACHE_COMPLETE",
        "pass_name": pass_name,
        "outer_fold": outer_fold,
        "evidence_boundary": "target-free held cache only; no H_o outcome computed",
        "sources": {
            "raw_features": {
                "path": str(inputs.paths["raw_features"]),
                "sha256": EXPECTED_HASHES["raw_features"],
            },
            "labels": {
                "path": str(inputs.paths["labels"]),
                "sha256": EXPECTED_HASHES["labels"],
            },
            "projected_features": {
                "path": str(inputs.paths["projected_features"]),
                "sha256": EXPECTED_HASHES["projected_features"],
            },
            **source_snapshot,
            "stage0_result_manifest_sha256": stage0_manifest["result"]["sha256"],
            "source_config_stable_for_entire_worker": True,
            "sealed_inputs_validated_at_worker_start": True,
        },
        "dependencies": environment,
        "domains": domains,
        "selected_training_rows": selected_record,
        "models": {
            "generic": generic_records,
            "focused": focused_records,
            "generic_count": len(generic_records),
            "focused_count": len(focused_records),
            "total_count": len(generic_records) + len(focused_records),
        },
        "current": {
            "oof_files": current_files,
            "oof_aux_files": current_aux_files,
            "inner_gate": current_inner,
            "full_files": current_full_files,
            "full_gate": current_full,
        },
        "focused_cache": focused_cache,
        "focused_partitions": focused_subsets,
        "portfolio_score_files": {
            "oof_generic": oof_score_files,
            "oof_focused": focused_oof_file,
        },
        "coverage": {
            "oof_sha256": _array_sha256(oof_coverage),
            "focused_sha256": _array_sha256(focused_coverage),
        },
        "rrf": {"oof": rrf_oof_file, "full": rrf_full_file},
        "runtime": runtime_records,
        "portfolio_training_labels": portfolio_label_record,
        "selector": selector_record,
        "held": held_files,
        "stage0_prefix_parity": stage0_parity,
        "stage0_parity_files": stage0_parity_files,
        "identity": identity,
        "identity_sha256": _canonical_sha256(identity),
        "privacy": {
            "retained_outcome_scope": "T_%d only" % outer_fold,
            "compressed_npz_member_decompression_may_include_H_o": True,
            "held_outcome_rows_retained_or_supplied_to_fit_selection_or_metric": 0,
            "held_surface_builder_accepts_labels": False,
            "held_state_or_outcome_metric_computed": False,
            "frozen_current_comparator_opened": False,
            "agent_or_official_evaluator_started": False,
            "calibration_selection_confirmation_public_or_external_opened": False,
        },
        "resource": {
            "total_wall_seconds": round(time.perf_counter() - started, 6),
            "peak_working_set_bytes": peak,
            "workers": 1,
        },
    }
    probe._assert_no_identity_matches(result)
    complete_path = output_dir / "outer_complete.json"
    _write_json_exclusive(complete_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "pass": pass_name,
                "outer": outer_fold,
                "models": result["models"]["total_count"],
                "identity_sha256": result["identity_sha256"],
                "result": complete_path.relative_to(ROOT).as_posix(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pass-name", choices=("first", "repeat"), required=True)
    parser.add_argument(
        "--outer-fold", type=int, choices=range(base.OUTER_FOLDS), required=True
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    run_outer(
        args.source_root,
        args.projection_root,
        args.output_dir,
        args.pass_name,
        args.outer_fold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
