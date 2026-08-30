"""Run the held/proposal-outcome-blind Stage-0 probe for v2.8 strict restacking.

The probe rebuilds the five outer-0 current inner models and the four A00
proposal models twice.  T0 labels are training-side supervision only.  No H0
outcome is retained, supplied, or allowed to influence a fit, and no proposal
or selector outcome is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from scripts import train_small_ranker as base  # noqa: E402
from scripts import train_small_ranker_focused_outer_oof as focused  # noqa: E402


SCHEMA_VERSION = "small-ranker-strict-outer-restack-stage0.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_8.strict_outer_restack_preregistration.json"
)
IMPLEMENTATION_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_8.strict_outer_restack_implementation_amendment.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
EXPECTED_HASHES = {
    "raw_features": "2b19835a1bced7f21322610296c712e3d06d915274719e11c268d31f7f596089",
    "labels": "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb",
    "projected_features": "cd9b075b923c31afe10b9a9c6d720de22e221c93de961595ff484ea4f532b90a",
    "feature_schema_manifest": "a930d184672bc29d9dd4bc1c2e908da035712ab061f2127a9771b2f3ed6a5c1a",
    "preregistration": "67d93823f2016dc4fdbcfa6687c236692f2e554cfa678e0878b5f1294c7a3ef9",
    "implementation_amendment": "a4a44d28d4e88ed75c92b61723b0aa7b5627211faeb08aaa0a9c8ba2a9bc3938",
}
HELPER_HASHES = {
    "scripts/train_small_ranker.py": "db7f4a3e19da118abb7d37fc1530babd6928894e51e85010b11d9dcdc1d7e583",
    "scripts/export_small_ranker_fold_safe_artifact.py": "5115026c53b21d4d5930cb9af7783c0988b049a0e259f5a0a588901ad44f5e8b",
    "scripts/analyze_small_ranker_metric_gate.py": "8c0cbffa6cd3dc62ddee3bb386c16bd60592a6324ecf6fcf4bcd4cf37951ca83",
    "scripts/analyze_small_ranker_rr_regret_gate.py": "793e3615df38cd995f55e57decaeea35b549e40ad50ee3bf8a6dbf1055ca7e80",
    "scripts/train_small_ranker_focused_outer_oof.py": "8fbf05d6225e04e3f76db7e65a91160394f11316ac90b416b6cf53b0b2ba497c",
}
BASE_SEED = 40220260830
OUTER_FOLD = 0
MODEL_OFFSETS = {
    "current_ndcg_d4_lr003": 0,
    "pairwise_d4_control": 10_000,
    "ndcg_d6_lr006": 20_000,
    "ndcg_d4_regularized": 30_000,
    "focused_ndcg_d3": 40_000,
}
T0_OUTCOME_FIELDS = (
    "baseline_rank",
    "baseline_session_hit",
    "eligible_from",
    "positive_index",
    "training_indices",
    "training_length",
)


class StrictRestackProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeInputs:
    raw_features: np.ndarray
    projected_features: np.ndarray
    outer_fold: np.ndarray
    inner_fold: np.ndarray
    family_index: np.ndarray
    t0_sessions: np.ndarray
    labels_t0: Mapping[str, np.ndarray]
    paths: Mapping[str, Path]


@dataclass(frozen=True)
class FocusedGroups:
    local_session: np.ndarray
    global_session: np.ndarray
    turn: np.ndarray
    hard: np.ndarray
    hard_session: np.ndarray
    control_session: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StrictRestackProbeError("expected a JSON object")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npy_exclusive(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "array_sha256": _array_sha256(np.asarray(value)),
        "bytes": path.stat().st_size,
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "asin_shape_matches": base._identity_shape_scan(path),
    }


def _assert_no_identity_matches(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "asin_shape_matches" and int(child) != 0:
                raise StrictRestackProbeError("output identity-shape scan failed")
            _assert_no_identity_matches(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_identity_matches(child)


def _validate_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    schema_manifest = ROOT / "configs/small_ranker_v1.cache.manifest.json"
    if (
        not PREREGISTRATION.is_file()
        or _sha256(PREREGISTRATION) != EXPECTED_HASHES["preregistration"]
        or not IMPLEMENTATION_AMENDMENT.is_file()
        or _sha256(IMPLEMENTATION_AMENDMENT)
        != EXPECTED_HASHES["implementation_amendment"]
        or not schema_manifest.is_file()
        or _sha256(schema_manifest) != EXPECTED_HASHES["feature_schema_manifest"]
    ):
        raise StrictRestackProbeError("v2.8 protocol identity mismatch")
    for relative_path, expected_hash in HELPER_HASHES.items():
        helper_path = ROOT / relative_path
        if not helper_path.is_file() or _sha256(helper_path) != expected_hash:
            raise StrictRestackProbeError(
                "frozen helper identity mismatch: %s" % relative_path
            )
    prereg = _load_json(PREREGISTRATION)
    amendment = _load_json(IMPLEMENTATION_AMENDMENT)
    models = prereg.get("generic_training", {}).get("models", [])
    offsets = prereg.get("generic_training", {}).get("seed", {}).get(
        "model_offsets", {}
    )
    if (
        prereg.get("schema_version")
        != "small-ranker-strict-outer-restack-preregistration.v1"
        or amendment.get("schema_version")
        != "small-ranker-strict-outer-restack-implementation-amendment.v1"
        or {str(item.get("id")) for item in models}
        != set(MODEL_OFFSETS).difference({"focused_ndcg_d3"})
        or {str(key): int(value) for key, value in offsets.items()}
        != MODEL_OFFSETS
        or int(prereg.get("frozen_inputs", {}).get("sessions", -1))
        != base.SESSION_COUNT
        or int(prereg.get("frozen_inputs", {}).get("feature_count", -1))
        != base.FEATURE_COUNT
    ):
        raise StrictRestackProbeError("v2.8 protocol mechanics drifted")
    return prereg, amendment


def _load_inputs(source_root: Path, projection_root: Path) -> ProbeInputs:
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
            raise StrictRestackProbeError("sealed input mismatch: %s" % name)
    raw = np.load(paths["raw_features"], mmap_mode="r")
    projected = np.load(paths["projected_features"], mmap_mode="r")
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
        raise StrictRestackProbeError("feature tensor schema mismatch")
    with np.load(paths["labels"], allow_pickle=False) as archive:
        outer = np.asarray(archive["outer_fold"], dtype=np.uint8).copy()
        inner = np.asarray(archive["inner_fold"], dtype=np.uint8).copy()
        family = np.asarray(archive["family_index"], dtype=np.int32).copy()
        t0_sessions = np.flatnonzero(outer != OUTER_FOLD).astype(np.int16)
        labels_t0 = {
            name: np.asarray(archive[name][t0_sessions]).copy()
            for name in T0_OUTCOME_FIELDS
        }
    labels_t0 = {
        **labels_t0,
        "outer_fold": outer[t0_sessions].copy(),
        "inner_fold": inner[t0_sessions].copy(),
        "family_index": family[t0_sessions].copy(),
    }
    if len(t0_sessions) != 1600 or set(np.unique(inner[t0_sessions]).tolist()) != set(range(5)):
        raise StrictRestackProbeError("T0 fold schema mismatch")
    for family_id in np.unique(family):
        family_mask = family == family_id
        if (
            len(np.unique(outer[family_mask])) != 1
            or len(np.unique(inner[family_mask])) != 1
        ):
            raise StrictRestackProbeError("product family crosses a fold boundary")
    return ProbeInputs(
        raw, projected, outer, inner, family, t0_sessions, labels_t0, paths
    )


def _mask_record(mask: np.ndarray, family: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    return {
        "sessions": int(selected.sum()),
        "families": int(len(np.unique(family[selected]))),
        "mask_sha256": _array_sha256(selected.astype(np.uint8)),
    }


def _domain_records(inputs: ProbeInputs) -> dict[str, Any]:
    h0 = inputs.outer_fold == OUTER_FOLD
    t0 = ~h0
    records: dict[str, Any] = {
        "H_0": _mask_record(h0, inputs.family_index),
        "T_0": _mask_record(t0, inputs.family_index),
    }
    coverage = np.zeros(base.SESSION_COUNT, dtype=np.uint8)
    for inner_fold in range(base.OUTER_FOLDS):
        valid = t0 & (inputs.inner_fold == inner_fold)
        train = t0 & (inputs.inner_fold != inner_fold)
        if np.any(valid & train) or np.any(h0 & (valid | train)):
            raise StrictRestackProbeError("domain masks overlap")
        coverage[valid] += 1
        records["V_0%d" % inner_fold] = _mask_record(valid, inputs.family_index)
        records["A_0%d" % inner_fold] = _mask_record(train, inputs.family_index)
    if not np.all(coverage[t0] == 1) or np.any(coverage[h0]):
        raise StrictRestackProbeError("inner validation coverage is invalid")
    return records


def _build_selected_t0(inputs: ProbeInputs) -> base.SelectedTrainingRows:
    lengths = np.asarray(inputs.labels_t0["training_length"], dtype=np.int64)
    indices = np.asarray(inputs.labels_t0["training_indices"], dtype=np.int64)
    positive = np.asarray(inputs.labels_t0["positive_index"], dtype=np.int64)
    row_count = int(lengths.sum())
    x = np.empty((row_count, base.FEATURE_COUNT), dtype=np.float32)
    y = np.empty(row_count, dtype=np.float32)
    qid = np.empty(row_count, dtype=np.int32)
    sessions = np.empty(row_count, dtype=np.int16)
    turns = np.empty(row_count, dtype=np.uint8)
    cursor = 0
    group_count = 0
    for local_session, global_session in enumerate(inputs.t0_sessions):
        for turn in range(base.TURN_COUNT):
            length = int(lengths[local_session, turn])
            if not length:
                continue
            chosen = indices[local_session, turn, :length]
            if np.any(chosen < 0) or len(np.unique(chosen)) != length:
                raise StrictRestackProbeError("training candidate group is invalid")
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
        raise StrictRestackProbeError("T0 grouped rows failed validation")
    return base.SelectedTrainingRows(x, y, qid, sessions, turns, group_count)


def choose_slot10_any(
    scores: np.ndarray, incumbent: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    incumbent = np.asarray(incumbent, dtype=np.int64)
    if (
        scores.ndim != 3
        or scores.shape[1:] != (base.TURN_COUNT, base.CANDIDATE_COUNT)
        or incumbent.shape != scores.shape[:2]
        or np.any(incumbent < 0)
        or np.any(incumbent >= 10)
    ):
        raise StrictRestackProbeError("slot-10 choice input shape mismatch")
    allowed = scores.copy()
    allowed[:, :, :10] = -np.inf
    sessions = np.arange(len(scores))[:, None]
    turns = np.arange(base.TURN_COUNT)[None, :]
    incumbent_score = scores[sessions, turns, incumbent]
    allowed[sessions, turns, incumbent] = incumbent_score
    chosen = np.argmax(allowed, axis=2).astype(np.uint8)
    chosen_score = allowed[sessions, turns, chosen]
    second = np.partition(allowed, -2, axis=2)[:, :, -2]
    return (
        chosen,
        (chosen_score - incumbent_score).astype(np.float32),
        (chosen_score - second).astype(np.float32),
    )


def _incumbent_for_sessions(
    features: np.ndarray, sessions: np.ndarray, batch_size: int = 20
) -> np.ndarray:
    output = np.empty((len(sessions), base.TURN_COUNT), dtype=np.uint8)
    p11_presence = base.FEATURE_INDEX["p11_presence"]
    p11_rank = base.FEATURE_INDEX["p11_rank_fraction"]
    for offset in range(0, len(sessions), batch_size):
        selected = sessions[offset : offset + batch_size]
        block = np.asarray(features[selected], dtype=np.float32)
        candidates = (block[..., p11_presence] > 0.5) & np.isclose(
            block[..., p11_rank], 1.0, atol=1e-6
        )
        if not np.all(candidates.sum(axis=2) == 1):
            raise StrictRestackProbeError("incumbent identity is invalid")
        output[offset : offset + len(selected)] = np.argmax(
            candidates, axis=2
        ).astype(np.uint8)
    return output


def gate_feature_matrix_any(
    features: np.ndarray,
    sessions: np.ndarray,
    scores: np.ndarray,
    chosen: np.ndarray,
    incumbent: np.ndarray,
    margin: np.ndarray,
    top_gap: np.ndarray,
    batch_size: int = 20,
) -> np.ndarray:
    expected = (len(sessions), base.TURN_COUNT)
    if (
        scores.shape != (*expected, base.CANDIDATE_COUNT)
        or chosen.shape != expected
        or incumbent.shape != expected
        or margin.shape != expected
        or top_gap.shape != expected
    ):
        raise StrictRestackProbeError("gate surface input shape mismatch")
    output = np.empty((*expected, len(base.GATE_FEATURE_NAMES)), dtype=np.float32)
    static_columns = [base.FEATURE_INDEX[name] for name in base.GATE_STATIC_FEATURES]
    conflict_columns = [
        base.FEATURE_INDEX["%s_conflict" % slot] for slot in base.CONSTRAINT_SLOTS
    ]
    active_recall = base.FEATURE_INDEX["active_token_recall"]
    hard_coverage = base.FEATURE_INDEX["hard_clause_coverage"]
    for offset in range(0, len(sessions), batch_size):
        selected_sessions = sessions[offset : offset + batch_size]
        count = len(selected_sessions)
        block = np.asarray(features[selected_sessions], dtype=np.float32).reshape(
            -1, base.CANDIDATE_COUNT, base.FEATURE_COUNT
        )
        block_scores = scores[offset : offset + count].reshape(
            -1, base.CANDIDATE_COUNT
        )
        block_chosen = chosen[offset : offset + count].reshape(-1).astype(np.int64)
        block_incumbent = incumbent[offset : offset + count].reshape(-1).astype(
            np.int64
        )
        rows = np.arange(len(block_chosen))
        chosen_features = block[rows, block_chosen]
        incumbent_features = block[rows, block_incumbent]
        chosen_conflict = chosen_features[:, conflict_columns].sum(axis=1)
        incumbent_conflict = incumbent_features[:, conflict_columns].sum(axis=1)
        matrix = np.column_stack(
            (
                margin[offset : offset + count].reshape(-1),
                top_gap[offset : offset + count].reshape(-1),
                block_scores[rows, block_chosen],
                block_scores[rows, block_incumbent],
                chosen_features[:, static_columns],
                chosen_features[:, active_recall]
                - incumbent_features[:, active_recall],
                chosen_features[:, hard_coverage]
                - incumbent_features[:, hard_coverage],
                chosen_conflict - incumbent_conflict,
            )
        ).astype(np.float32)
        output[offset : offset + count] = matrix.reshape(
            count, base.TURN_COUNT, -1
        )
    if not np.isfinite(output).all():
        raise StrictRestackProbeError("gate features are non-finite")
    return output


def rrf3_scores_any(members: Sequence[np.ndarray]) -> np.ndarray:
    if len(members) != 3:
        raise StrictRestackProbeError("RRF-3 requires exactly three members")
    result = np.zeros(np.asarray(members[0]).shape, dtype=np.float32)
    for member in members:
        scores = np.asarray(member)
        if (
            scores.shape != result.shape
            or scores.ndim != 3
            or scores.shape[2] != base.CANDIDATE_COUNT
            or scores.dtype != np.float32
            or not np.isfinite(scores).all()
        ):
            raise StrictRestackProbeError("RRF-3 member schema mismatch")
        order = np.argsort(-scores, axis=2, kind="stable")
        ranks = np.empty(order.shape, dtype=np.uint8)
        values = np.broadcast_to(
            np.arange(1, base.CANDIDATE_COUNT + 1, dtype=np.uint8),
            order.shape,
        )
        np.put_along_axis(ranks, order, values, axis=2)
        result += 1.0 / (np.float32(60.0) + ranks)
    if not np.isfinite(result).all():
        raise StrictRestackProbeError("RRF-3 scores are non-finite")
    return result


def _training_surface_t0(
    inputs: ProbeInputs, scores: np.ndarray
) -> frozen.ActionSurface:
    incumbent = _incumbent_for_sessions(
        inputs.projected_features, inputs.t0_sessions
    )
    chosen, margin, top_gap = choose_slot10_any(scores, incumbent)
    gate_features = gate_feature_matrix_any(
        inputs.projected_features,
        inputs.t0_sessions,
        scores,
        chosen,
        incumbent,
        margin,
        top_gap,
    )
    rescue, _direct_risk, rescue_weights = base.action_training_labels(
        inputs.labels_t0, chosen, incumbent
    )
    rr_regret = rr.single_action_rr_regret(inputs.labels_t0, chosen, incumbent)
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


def _gate_model_sha256(
    model: Any, mean: np.ndarray, scale: np.ndarray
) -> str:
    payload: dict[str, Any] = {
        "type": type(model).__name__,
        "mean": _array_sha256(np.asarray(mean)),
        "scale": _array_sha256(np.asarray(scale)),
    }
    if hasattr(model, "coef_"):
        payload.update(
            {
                "coef": _array_sha256(np.asarray(model.coef_)),
                "intercept": _array_sha256(np.asarray(model.intercept_)),
                "classes": _array_sha256(np.asarray(model.classes_)),
            }
        )
    else:
        payload["probability"] = float(model.probability)
    return _canonical_sha256(payload)


def _crossfit_current_t0(
    surface: frozen.ActionSurface, labels_t0: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    inner = np.asarray(labels_t0["inner_fold"], dtype=np.uint8)
    if set(np.unique(inner).tolist()) != set(range(base.OUTER_FOLDS)):
        raise StrictRestackProbeError("T0 inner folds are incomplete")
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
            raise StrictRestackProbeError("current gate partition is empty")
        coverage[valid_rows] += 1
        head_hashes = []
        for head in range(2):
            model, mean, scale = base._fit_gate_model(
                flat_x[train_rows],
                targets[head][train_rows],
                weights[head][train_rows],
                BASE_SEED + head * 10_000 + OUTER_FOLD * 31 + inner_fold,
            )
            probabilities[head].reshape(-1)[valid_rows] = base._predict_gate(
                model, mean, scale, flat_x[valid_rows]
            ).astype(np.float32)
            head_hashes.append(_gate_model_sha256(model, mean, scale))
        fold_records.append(
            {
                "inner_fold": inner_fold,
                "train_action_rows": int(train_rows.sum()),
                "valid_action_rows": int(valid_rows.sum()),
                "head_model_sha256": head_hashes,
            }
        )
    if not np.all(coverage[flat_action] == 1) or np.any(coverage[~flat_action]):
        raise StrictRestackProbeError("current gate OOF coverage is invalid")
    utility = probabilities[0] - frozen.RR_MULTIPLIER * probabilities[1]
    selected = frozen._select_inner_quantile(
        utility,
        surface,
        labels_t0,
        np.ones(len(inner), dtype=bool),
        inner,
    )
    threshold = float(selected["inner_threshold"])
    activation = surface.action & (utility >= threshold)
    state = metric.policy_session_state(labels_t0, surface.chosen, activation)
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
    }
    return state, activation, record


def _select_focused_a00_groups(
    inputs: ProbeInputs,
    current_state: Mapping[str, np.ndarray],
    incumbent: np.ndarray,
) -> FocusedGroups:
    positive = np.asarray(inputs.labels_t0["positive_index"], dtype=np.int16)
    eligible_from = np.asarray(inputs.labels_t0["eligible_from"], dtype=np.int16)
    inner = np.asarray(inputs.labels_t0["inner_fold"], dtype=np.uint8)
    domain = inner != 0
    hit = np.asarray(current_state["hit"], dtype=bool)
    first_rank = np.asarray(current_state["first_rank"], dtype=np.int16)
    first_turn = np.asarray(current_state["first_turn"], dtype=np.int16)
    if positive.shape != incumbent.shape or len(domain) != len(inputs.t0_sessions):
        raise StrictRestackProbeError("focused cohort input shape mismatch")
    hard_session = np.zeros(len(domain), dtype=bool)
    control_session = np.zeros(len(domain), dtype=bool)
    allowed_turn = np.zeros_like(positive, dtype=bool)
    for local_session in np.flatnonzero(domain):
        eligible_index = int(eligible_from[local_session]) - 1
        if not 0 <= eligible_index < base.TURN_COUNT:
            raise StrictRestackProbeError("eligible turn is outside the horizon")
        for turn in range(eligible_index, base.TURN_COUNT):
            target = int(positive[local_session, turn])
            allowed_turn[local_session, turn] = target >= 0 and (
                target == int(incumbent[local_session, turn])
                or 10 <= target < base.CANDIDATE_COUNT
            )
        if not hit[local_session]:
            hard_session[local_session] = bool(
                np.any(allowed_turn[local_session])
            )
        elif int(first_rank[local_session]) == 10:
            stop = min(int(first_turn[local_session]), base.TURN_COUNT)
            control_session[local_session] = bool(
                np.any(allowed_turn[local_session, eligible_index:stop])
            )
    local_sessions = []
    turns = []
    hard_flags = []
    for local_session in np.flatnonzero(domain):
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
                local_sessions.append(int(local_session))
                turns.append(turn)
                hard_flags.append(is_hard)
    local_array = np.asarray(local_sessions, dtype=np.int16)
    turn_array = np.asarray(turns, dtype=np.uint8)
    hard_array = np.asarray(hard_flags, dtype=np.uint8)
    if (
        not len(local_array)
        or not np.any(hard_array)
        or not np.any(~hard_array.astype(bool))
        or np.any(~domain[local_array])
        or np.any(hard_session & control_session)
    ):
        raise StrictRestackProbeError("A00 focused cohorts are invalid")
    return FocusedGroups(
        local_array,
        inputs.t0_sessions[local_array].astype(np.int16),
        turn_array,
        hard_array,
        hard_session,
        control_session,
    )


def _materialize_focused(
    inputs: ProbeInputs,
    groups: FocusedGroups,
    incumbent: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    group_count = len(groups.local_session)
    x = np.empty(
        (group_count, focused.ROWS_PER_GROUP, base.FEATURE_COUNT),
        dtype=np.float32,
    )
    y = np.empty((group_count, focused.ROWS_PER_GROUP), dtype=np.uint8)
    positive = np.asarray(inputs.labels_t0["positive_index"], dtype=np.int16)
    for group in range(group_count):
        local_session = int(groups.local_session[group])
        global_session = int(groups.global_session[group])
        turn = int(groups.turn[group])
        current = int(incumbent[local_session, turn])
        allowed = np.asarray(
            [current, *range(10, base.CANDIDATE_COUNT)], dtype=np.int16
        )
        x[group] = inputs.projected_features[global_session, turn, allowed]
        y[group] = (
            allowed == int(positive[local_session, turn])
        ).astype(np.uint8)
    if not np.isfinite(x).all() or not np.all(y.sum(axis=1) == 1):
        raise StrictRestackProbeError("focused cache parity failed")
    metadata = {
        "session_ordinal": groups.global_session,
        "turn_index": groups.turn,
        "hard_cohort": groups.hard,
        "outer_fold": inputs.outer_fold[groups.global_session],
        "inner_fold": inputs.inner_fold[groups.global_session],
    }
    files = {
        "features": _write_npy_exclusive(output_dir / "focused_features.npy", x),
        "relevance": _write_npy_exclusive(output_dir / "focused_relevance.npy", y),
    }
    for name, value in metadata.items():
        files[name] = _write_npy_exclusive(output_dir / (name + ".npy"), value)
    if any(int(item["asin_shape_matches"]) for item in files.values()):
        raise StrictRestackProbeError("focused cache identity scan failed")
    weights = focused.partition_group_weights(
        metadata["session_ordinal"],
        metadata["hard_cohort"],
        np.ones(group_count, dtype=bool),
    )
    record = {
        "query_groups": group_count,
        "rows": int(group_count * focused.ROWS_PER_GROUP),
        "hard_sessions": int(groups.hard_session.sum()),
        "control_sessions": int(groups.control_session.sum()),
        "hard_groups": int(groups.hard.sum()),
        "control_groups": int((~groups.hard.astype(bool)).sum()),
        "group_weight_sha256": _array_sha256(weights),
        "files": files,
    }
    return x, y, metadata, record


def _model_seed(model_id: str, domain_code: int) -> int:
    return int((BASE_SEED + MODEL_OFFSETS[model_id] + domain_code) % (2**32 - 1))


def _generic_params(spec: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        "objective": str(spec["objective"]),
        "eval_metric": "ndcg@10",
        "tree_method": "hist",
        "max_bin": 256,
        "max_depth": int(spec["max_depth"]),
        "eta": float(spec["eta"]),
        "min_child_weight": float(spec["min_child_weight"]),
        "subsample": float(spec["subsample"]),
        "colsample_bytree": float(spec["colsample_bytree"]),
        "alpha": float(spec["reg_alpha"]),
        "lambda": float(spec["reg_lambda"]),
        "nthread": 1,
        "verbosity": 0,
        "seed": int(seed),
    }


def _save_booster_exclusive(booster: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(bytes(booster.save_raw(raw_format="json")))
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "asin_shape_matches": base._identity_shape_scan(path),
    }


def _score_booster(
    booster: Any,
    projected: np.ndarray,
    sessions: np.ndarray,
    batch_size: int = 10,
) -> tuple[np.ndarray, dict[str, float]]:
    import xgboost as xgb

    scores = np.empty(
        (len(sessions), base.TURN_COUNT, base.CANDIDATE_COUNT), dtype=np.float32
    )
    per_session_ms = []
    started = time.perf_counter()
    for offset in range(0, len(sessions), batch_size):
        selected = sessions[offset : offset + batch_size]
        block = np.asarray(projected[selected], dtype=np.float32)
        if not np.isfinite(block).all():
            raise StrictRestackProbeError("projected scoring block is non-finite")
        tick = time.perf_counter()
        prediction = booster.predict(
            xgb.DMatrix(block.reshape(-1, base.FEATURE_COUNT), nthread=1),
            output_margin=True,
        )
        elapsed = time.perf_counter() - tick
        scores[offset : offset + len(selected)] = np.asarray(
            prediction, dtype=np.float32
        ).reshape(len(selected), base.TURN_COUNT, base.CANDIDATE_COUNT)
        per_session_ms.append(1000.0 * elapsed / len(selected))
    if not np.isfinite(scores).all():
        raise StrictRestackProbeError("model scores are non-finite")
    values = np.asarray(per_session_ms, dtype=np.float64)
    return scores, {
        "total_seconds": round(time.perf_counter() - started, 6),
        "p50_ms_per_session_batch": round(float(np.quantile(values, 0.5)), 6),
        "p95_ms_per_session_batch": round(float(np.quantile(values, 0.95)), 6),
    }


def _serialized_model_parity(
    model_path: Path,
    projected: np.ndarray,
    score_sessions: np.ndarray,
    expected_scores: np.ndarray,
) -> dict[str, Any]:
    import xgboost as xgb

    sample_positions = np.unique(
        np.asarray([0, len(score_sessions) - 1], dtype=np.int64)
    )
    sampled_sessions = score_sessions[sample_positions]
    block = np.asarray(projected[sampled_sessions], dtype=np.float32)
    if not np.isfinite(block).all():
        raise StrictRestackProbeError("serialized parity block is non-finite")
    reloaded = xgb.Booster()
    reloaded.load_model(model_path)
    actual = np.asarray(
        reloaded.predict(
            xgb.DMatrix(block.reshape(-1, base.FEATURE_COUNT), nthread=1),
            output_margin=True,
        ),
        dtype=np.float32,
    ).reshape(len(sampled_sessions), base.TURN_COUNT, base.CANDIDATE_COUNT)
    expected = np.asarray(expected_scores[sample_positions], dtype=np.float32)
    maximum_error = float(np.max(np.abs(actual - expected)))
    order_exact = bool(
        np.array_equal(
            np.argsort(-actual, axis=2, kind="stable"),
            np.argsort(-expected, axis=2, kind="stable"),
        )
    )
    if maximum_error != 0.0 or not order_exact:
        raise StrictRestackProbeError("serialized model score parity failed")
    return {
        "sample_session_sha256": _array_sha256(sampled_sessions),
        "rows": int(actual.size),
        "maximum_absolute_error": maximum_error,
        "c100_order_exact": order_exact,
    }


def _train_generic(
    model_id: str,
    spec: Mapping[str, Any],
    selected: base.SelectedTrainingRows,
    train_sessions: np.ndarray,
    score_sessions: np.ndarray,
    inputs: ProbeInputs,
    domain: str,
    domain_code: int,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import xgboost as xgb

    train_rows = np.asarray(train_sessions, dtype=bool)[selected.session]
    row_indices = np.flatnonzero(train_rows)
    qid = selected.qid[row_indices]
    boundaries = np.r_[0, np.flatnonzero(np.diff(qid) != 0) + 1, len(qid)]
    group_sizes = np.diff(boundaries).astype(np.uint32)
    if not len(row_indices) or base.validate_grouped_qid(qid, selected.y[row_indices]) != len(group_sizes):
        raise StrictRestackProbeError("generic grouped partition is invalid")
    matrix = xgb.DMatrix(
        np.asarray(selected.x[row_indices], dtype=np.float32),
        label=np.asarray(selected.y[row_indices], dtype=np.float32),
        nthread=1,
    )
    matrix.set_group(group_sizes)
    seed = _model_seed(model_id, domain_code)
    rounds = int(spec["rounds"])
    started = time.perf_counter()
    booster = xgb.train(
        _generic_params(spec, seed), matrix, num_boost_round=rounds
    )
    train_seconds = time.perf_counter() - started
    if (
        int(booster.num_features()) != base.FEATURE_COUNT
        or int(booster.num_boosted_rounds()) != rounds
    ):
        raise StrictRestackProbeError("generic model contract failed")
    del matrix
    model_path = output_dir / "models" / (model_id + "__" + domain + ".json")
    model_record = _save_booster_exclusive(booster, model_path)
    scores, prediction_timing = _score_booster(
        booster, inputs.projected_features, score_sessions
    )
    incumbent = _incumbent_for_sessions(inputs.projected_features, score_sessions)
    choice, _margin, _gap = choose_slot10_any(scores, incumbent)
    score_record = _write_npy_exclusive(
        output_dir / "scores" / (model_id + "__" + domain + ".npy"), scores
    )
    choice_record = _write_npy_exclusive(
        output_dir / "choices" / (model_id + "__" + domain + ".npy"),
        choice,
    )
    serialized_parity = _serialized_model_parity(
        model_path,
        inputs.projected_features,
        score_sessions,
        scores,
    )
    rss, peak = focused._rss()
    return scores, choice, {
        "model_id": model_id,
        "domain": domain,
        "seed": seed,
        "rounds": rounds,
        "train_sessions": int(np.asarray(train_sessions, dtype=bool).sum()),
        "train_session_mask_sha256": _array_sha256(
            np.asarray(train_sessions, dtype=np.uint8)
        ),
        "train_rows": int(len(row_indices)),
        "train_query_groups": int(len(group_sizes)),
        "score_sessions": int(len(score_sessions)),
        "score_session_sha256": _array_sha256(score_sessions),
        "training_seconds": round(train_seconds, 6),
        "prediction": prediction_timing,
        "rss_bytes_after_fit": rss,
        "peak_working_set_bytes": peak,
        "model": model_record,
        "score": score_record,
        "choice": choice_record,
        "serialized_model_parity": serialized_parity,
    }


def _train_focused_a00(
    x: np.ndarray,
    y: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    inputs: ProbeInputs,
    score_sessions: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    groups = {
        "session_ordinal": metadata["session_ordinal"],
        "hard_cohort": metadata["hard_cohort"],
    }
    seed = _model_seed("focused_ndcg_d3", 0)
    model_path = output_dir / "models/focused_ndcg_d3__A_00.json"
    booster, audit = focused._train_model(
        x,
        y,
        groups,
        np.ones(len(x), dtype=bool),
        seed,
        model_path,
    )
    scores, prediction_timing = _score_booster(
        booster, inputs.projected_features, score_sessions
    )
    incumbent = _incumbent_for_sessions(inputs.projected_features, score_sessions)
    choice, _margin, _gap = choose_slot10_any(scores, incumbent)
    score_record = _write_npy_exclusive(
        output_dir / "scores/focused_ndcg_d3__A_00.npy", scores
    )
    choice_record = _write_npy_exclusive(
        output_dir / "choices/focused_ndcg_d3__A_00.npy", choice
    )
    model_record = {
        "path": model_path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(model_path),
        "bytes": model_path.stat().st_size,
        "asin_shape_matches": base._identity_shape_scan(model_path),
    }
    serialized_parity = _serialized_model_parity(
        model_path,
        inputs.projected_features,
        score_sessions,
        scores,
    )
    actual_train_sessions = np.unique(metadata["session_ordinal"])
    actual_train_mask = np.isin(
        np.arange(base.SESSION_COUNT), actual_train_sessions
    )
    parent_a00_mask = (inputs.outer_fold != 0) & (inputs.inner_fold != 0)
    if np.any(actual_train_mask & ~parent_a00_mask):
        raise StrictRestackProbeError("focused cohort escaped A00")
    return scores, choice, {
        "model_id": "focused_ndcg_d3",
        "domain": "A_00",
        "seed": seed,
        "rounds": focused.ROUNDS,
        "train_sessions": int(len(actual_train_sessions)),
        "train_session_mask_sha256": _array_sha256(
            actual_train_mask.astype(np.uint8)
        ),
        "parent_domain_mask_sha256": _array_sha256(
            parent_a00_mask.astype(np.uint8)
        ),
        "train_rows": int(len(x) * focused.ROWS_PER_GROUP),
        "train_query_groups": int(len(x)),
        "score_sessions": int(len(score_sessions)),
        "score_session_sha256": _array_sha256(score_sessions),
        "training_seconds": float(audit["train_seconds"]),
        "prediction": prediction_timing,
        "rss_bytes_after_fit": int(audit["rss_bytes_after_fit"]),
        "peak_working_set_bytes": int(audit["peak_working_set_bytes"]),
        "cohort_weight": {
            "hard_query_groups": int(audit["hard_query_groups"]),
            "control_query_groups": int(audit["control_query_groups"]),
            "hard_weight_sum": float(audit["hard_weight_sum"]),
            "control_weight_sum": float(audit["control_weight_sum"]),
            "mean": float(audit["group_weight_mean"]),
        },
        "model": model_record,
        "score": score_record,
        "choice": choice_record,
        "serialized_model_parity": serialized_parity,
    }


def _run_pass(
    name: str,
    prereg: Mapping[str, Any],
    inputs: ProbeInputs,
    selected: base.SelectedTrainingRows,
    output_root: Path,
) -> dict[str, Any]:
    pass_started = time.perf_counter()
    output_dir = output_root / name
    model_specs = {
        str(item["id"]): item
        for item in prereg["generic_training"]["models"]
    }
    t0 = inputs.outer_fold != OUTER_FOLD
    current_scores = np.empty(
        (len(inputs.t0_sessions), base.TURN_COUNT, base.CANDIDATE_COUNT),
        dtype=np.float32,
    )
    current_covered = np.zeros(len(inputs.t0_sessions), dtype=np.uint8)
    records = []
    current_v00: Optional[np.ndarray] = None
    for inner_fold in range(base.OUTER_FOLDS):
        train = t0 & (inputs.inner_fold != inner_fold)
        valid_sessions = np.flatnonzero(
            t0 & (inputs.inner_fold == inner_fold)
        ).astype(np.int16)
        scores, _choice, record = _train_generic(
            "current_ndcg_d4_lr003",
            model_specs["current_ndcg_d4_lr003"],
            selected,
            train,
            valid_sessions,
            inputs,
            "A_0%d" % inner_fold,
            inner_fold,
            output_dir,
        )
        positions = np.searchsorted(inputs.t0_sessions, valid_sessions)
        if not np.array_equal(inputs.t0_sessions[positions], valid_sessions):
            raise StrictRestackProbeError("current score lineage is invalid")
        current_scores[positions] = scores
        current_covered[positions] += 1
        if inner_fold == 0:
            current_v00 = scores
        records.append(record)
    if (
        current_v00 is None
        or not np.all(current_covered == 1)
        or not np.isfinite(current_scores).all()
    ):
        raise StrictRestackProbeError("T0 current score assembly failed")
    surface = _training_surface_t0(inputs, current_scores)
    current_state, current_activation, current_gate = _crossfit_current_t0(
        surface, inputs.labels_t0
    )
    groups = _select_focused_a00_groups(
        inputs, current_state, surface.incumbent
    )
    x, y, metadata, focused_cache = _materialize_focused(
        inputs, groups, surface.incumbent, output_dir / "focused_cache"
    )
    v00_sessions = np.flatnonzero(
        t0 & (inputs.inner_fold == 0)
    ).astype(np.int16)
    a00 = t0 & (inputs.inner_fold != 0)
    proposal_scores: dict[str, np.ndarray] = {}
    for model_id in (
        "pairwise_d4_control",
        "ndcg_d6_lr006",
        "ndcg_d4_regularized",
    ):
        scores, _choice, record = _train_generic(
            model_id,
            model_specs[model_id],
            selected,
            a00,
            v00_sessions,
            inputs,
            "A_00",
            0,
            output_dir,
        )
        proposal_scores[model_id] = scores
        records.append(record)
    focused_scores, _focused_choice, focused_record = _train_focused_a00(
        x, y, metadata, inputs, v00_sessions, output_dir
    )
    proposal_scores["focused_ndcg_d3"] = focused_scores
    records.append(focused_record)
    rrf_scores = rrf3_scores_any(
        [
            current_v00,
            proposal_scores["ndcg_d6_lr006"],
            proposal_scores["ndcg_d4_regularized"],
        ]
    )
    v00_incumbent = _incumbent_for_sessions(
        inputs.projected_features, v00_sessions
    )
    rrf_choice, _margin, _gap = choose_slot10_any(rrf_scores, v00_incumbent)
    current_files = {
        "scores": _write_npy_exclusive(
            output_dir / "current_t0_scores.npy", current_scores
        ),
        "chosen": _write_npy_exclusive(
            output_dir / "current_t0_chosen.npy", surface.chosen
        ),
        "action": _write_npy_exclusive(
            output_dir / "current_t0_action.npy", surface.action.astype(np.uint8)
        ),
        "gate_features": _write_npy_exclusive(
            output_dir / "current_t0_gate_features.npy", surface.gate_features
        ),
        "activation": _write_npy_exclusive(
            output_dir / "current_t0_activation.npy",
            current_activation.astype(np.uint8),
        ),
    }
    rrf_files = {
        "score": _write_npy_exclusive(output_dir / "rrf3_v00.npy", rrf_scores),
        "choice": _write_npy_exclusive(
            output_dir / "rrf3_v00_choice.npy", rrf_choice
        ),
    }
    del x, y
    peak = max(int(record["peak_working_set_bytes"]) for record in records)
    result = {
        "name": name,
        "models": records,
        "current": {"files": current_files, "gate": current_gate},
        "focused_cache": focused_cache,
        "rrf3": rrf_files,
        "timing_seconds": {"total": round(time.perf_counter() - pass_started, 6)},
        "peak_working_set_bytes": peak,
    }
    _assert_no_identity_matches(result)
    return result


def _repeat_identity(pass_record: Mapping[str, Any]) -> dict[str, Any]:
    models = []
    for record in pass_record["models"]:
        models.append(
            {
                "model_id": record["model_id"],
                "domain": record["domain"],
                "seed": record["seed"],
                "rounds": record["rounds"],
                "train_sessions": record["train_sessions"],
                "train_session_mask_sha256": record[
                    "train_session_mask_sha256"
                ],
                "parent_domain_mask_sha256": record.get(
                    "parent_domain_mask_sha256"
                ),
                "train_rows": record["train_rows"],
                "train_query_groups": record["train_query_groups"],
                "score_session_sha256": record["score_session_sha256"],
                "model_sha256": record["model"]["sha256"],
                "score_array_sha256": record["score"]["array_sha256"],
                "choice_array_sha256": record["choice"]["array_sha256"],
                "serialized_model_parity": record[
                    "serialized_model_parity"
                ],
            }
        )
    current_files = pass_record["current"]["files"]
    focused_files = pass_record["focused_cache"]["files"]
    rrf_files = pass_record["rrf3"]
    return {
        "models": models,
        "current_arrays": {
            key: value["array_sha256"] for key, value in current_files.items()
        },
        "current_gate": pass_record["current"]["gate"],
        "focused_cache": {
            "query_groups": pass_record["focused_cache"]["query_groups"],
            "hard_sessions": pass_record["focused_cache"]["hard_sessions"],
            "control_sessions": pass_record["focused_cache"]["control_sessions"],
            "hard_groups": pass_record["focused_cache"]["hard_groups"],
            "control_groups": pass_record["focused_cache"]["control_groups"],
            "group_weight_sha256": pass_record["focused_cache"][
                "group_weight_sha256"
            ],
            "arrays": {
                key: value["array_sha256"] for key, value in focused_files.items()
            },
        },
        "rrf3": {
            key: value["array_sha256"] for key, value in rrf_files.items()
        },
    }


def run(
    source_root: Path, projection_root: Path, output_dir: Path
) -> dict[str, Any]:
    import sklearn
    import xgboost

    started = time.perf_counter()
    output_dir = output_dir.resolve()
    experiments_root = (ROOT / "experiments").resolve()
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or experiments_root not in output_dir.parents
    ):
        raise StrictRestackProbeError(
            "output directory must be new and below this worktree's experiments"
        )
    if (
        sys.version.split()[0] != "3.9.19"
        or np.__version__ != "1.26.4"
        or xgboost.__version__ != "1.7.6"
        or sklearn.__version__ != "1.1.3"
    ):
        raise StrictRestackProbeError(
            "Stage 0 requires Python 3.9.19, NumPy 1.26.4, XGBoost 1.7.6, and sklearn 1.1.3"
        )
    prereg, amendment = _validate_protocol()
    inputs = _load_inputs(source_root, projection_root)
    domains = _domain_records(inputs)
    output_dir.mkdir(parents=True)
    selected_started = time.perf_counter()
    selected = _build_selected_t0(inputs)
    selected_seconds = time.perf_counter() - selected_started
    selected_record = {
        "rows": int(len(selected.y)),
        "query_groups": int(selected.group_count),
        "build_seconds": round(selected_seconds, 6),
        "feature_array_sha256": _array_sha256(selected.x),
        "label_array_sha256": _array_sha256(selected.y),
        "qid_array_sha256": _array_sha256(selected.qid),
        "session_array_sha256": _array_sha256(selected.session),
        "held_outer_rows": int(
            np.sum(inputs.outer_fold[selected.session] == OUTER_FOLD)
        ),
    }
    if selected_record["held_outer_rows"] != 0:
        raise StrictRestackProbeError("H0 entered the selected training rows")
    held_scope_overlap = int(
        np.sum(inputs.outer_fold[inputs.t0_sessions] == OUTER_FOLD)
    )
    if held_scope_overlap != 0 or any(
        len(np.asarray(inputs.labels_t0[name])) != len(inputs.t0_sessions)
        for name in T0_OUTCOME_FIELDS
    ):
        raise StrictRestackProbeError("T0 supervision scope guard failed")
    first = _run_pass("first", prereg, inputs, selected, output_dir)
    repeat = _run_pass("repeat", prereg, inputs, selected, output_dir)
    first_identity = _repeat_identity(first)
    repeat_identity = _repeat_identity(repeat)
    if len(first_identity["models"]) != 9 or len(repeat_identity["models"]) != 9:
        raise StrictRestackProbeError("Stage 0 did not rebuild all nine models")
    exact = first_identity == repeat_identity
    if not exact:
        raise StrictRestackProbeError("Stage-0 exact repeat differs")
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.8-STRICT-OUTER-RESTACK-STAGE0",
        "status": "IMPLEMENTATION_PASS_STAGE0",
        "evidence_boundary": "mechanics only; no algorithmic promotion evidence",
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
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "implementation_amendment_sha256": _sha256(
                IMPLEMENTATION_AMENDMENT
            ),
            "feature_schema_manifest_sha256": EXPECTED_HASHES[
                "feature_schema_manifest"
            ],
            "helper_sha256": dict(HELPER_HASHES),
            "probe_sha256": _sha256(Path(__file__).resolve()),
        },
        "dependencies": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "xgboost": xgboost.__version__,
            "sklearn": sklearn.__version__,
            "workers": 1,
        },
        "domains": domains,
        "selected_training_rows": selected_record,
        "passes": [first, repeat],
        "exact_repeat": {
            "equal": exact,
            "identity_sha256": _canonical_sha256(first_identity),
            "models_compared": len(first_identity["models"]),
        },
        "privacy": {
            "t0_outcome_slices_materialized": list(T0_OUTCOME_FIELDS),
            "t0_supervision_session_sha256": _array_sha256(inputs.t0_sessions),
            "t0_supervision_scope_guard_passed": True,
            "compressed_npz_member_decompression_may_include_h0": True,
            "held_h0_outcome_rows_retained_or_supplied_to_training": held_scope_overlap,
            "v00_proposal_or_selector_outcome_metrics_computed": False,
            "agent_or_official_evaluator_started": False,
            "calibration_selection_confirmation_public_or_external_opened": False,
            "runtime_features_target_blind": True,
        },
        "resource": {
            "total_wall_seconds": round(time.perf_counter() - started, 6),
            "peak_working_set_bytes": max(
                int(first["peak_working_set_bytes"]),
                int(repeat["peak_working_set_bytes"]),
            ),
        },
    }
    result_path = output_dir / "stage0_result.json"
    _write_json_exclusive(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "exact_repeat": exact,
                "models_compared": result["exact_repeat"]["models_compared"],
                "result": result_path.relative_to(ROOT).as_posix(),
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    run(args.source_root, args.projection_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
