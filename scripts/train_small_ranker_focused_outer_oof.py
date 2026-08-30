"""Train and exactly repeat the v2.5 focused outer-OOF LambdaMART scores.

This module is intentionally Python-3.9 compatible: the XGBoost environment
must not import the Python-3.11-only attribution chain.  It consumes the
already-built numeric focused cache and only the frozen fold/projected-feature
artifacts needed for held-fold projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-focused-outer-oof-training.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_5.focused_lambdamart_preregistration.json"
)
IMPLEMENTATION_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_5.focused_lambdamart_implementation_amendment.json"
)
FOCUSED_CACHE_MANIFEST = ROOT / (
    "configs/small_ranker_v2_5.focused_cache.manifest.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
FEATURE_SHA256 = "2b19835a1bced7f21322610296c712e3d06d915274719e11c268d31f7f596089"
LABEL_SHA256 = "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb"
PROJECTED_SHA256 = "cd9b075b923c31afe10b9a9c6d720de22e221c93de961595ff484ea4f532b90a"
SEED = 40220260830
ROWS_PER_GROUP = base.CANDIDATE_COUNT - 9
ROUNDS = 300
PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",
    "tree_method": "hist",
    "max_bin": 256,
    "max_depth": 3,
    "eta": 0.03,
    "min_child_weight": 8.0,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "alpha": 0.1,
    "lambda": 4.0,
    "nthread": 1,
    "verbosity": 0,
}


class FocusedTrainingError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _local_regular_file(value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise FocusedTrainingError("focused cache path must be repo-relative")
    unresolved = ROOT / relative
    path = unresolved.resolve()
    if ROOT not in path.parents or not path.is_file() or unresolved.is_symlink():
        raise FocusedTrainingError("focused cache path is not a local regular file")
    return path


def _exclusive_npy_memmap(
    path: Path, dtype: Any, shape: Sequence[int]
) -> np.memmap:
    """Create a standard C-order ``.npy`` memmap without overwrite semantics."""

    resolved_dtype = np.dtype(dtype)
    resolved_shape = tuple(int(value) for value in shape)
    with path.open("xb") as handle:
        np.lib.format.write_array_header_2_0(
            handle,
            {
                "descr": np.lib.format.dtype_to_descr(resolved_dtype),
                "fortran_order": False,
                "shape": resolved_shape,
            },
        )
        offset = handle.tell()
        payload_bytes = int(np.prod(resolved_shape, dtype=np.int64)) * resolved_dtype.itemsize
        if payload_bytes <= 0:
            raise FocusedTrainingError("exclusive memmap shape is empty")
        handle.seek(offset + payload_bytes - 1)
        handle.write(b"\0")
    return np.memmap(
        path,
        dtype=resolved_dtype,
        mode="r+",
        offset=offset,
        shape=resolved_shape,
        order="C",
    )


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FocusedTrainingError("expected a JSON object")
    return value


def partition_group_weights(
    session: np.ndarray,
    hard: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """Return one session-normalized, cohort-balanced weight per selected qid."""

    session = np.asarray(session)
    hard = np.asarray(hard, dtype=bool)
    selected = np.asarray(selected, dtype=bool)
    if session.ndim != 1 or session.shape != hard.shape or session.shape != selected.shape:
        raise FocusedTrainingError("group-weight input shape mismatch")
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise FocusedTrainingError("training partition has no query groups")
    selected_session = session[indices]
    selected_hard = hard[indices]
    if not np.any(selected_hard) or not np.any(~selected_hard):
        raise FocusedTrainingError("training partition must contain both cohorts")
    unique, inverse, counts = np.unique(
        selected_session, return_inverse=True, return_counts=True
    )
    if not len(unique) or np.any(counts <= 0):
        raise FocusedTrainingError("session group counts are invalid")
    weight = 1.0 / counts[inverse].astype(np.float64)
    for cohort in (False, True):
        cohort_mask = selected_hard == cohort
        total = float(weight[cohort_mask].sum())
        if not total > 0.0:
            raise FocusedTrainingError("cohort weight is empty")
        weight[cohort_mask] *= 0.5 / total
    weight *= len(weight) / float(weight.sum())
    if (
        not np.isfinite(weight).all()
        or not np.isclose(weight.mean(), 1.0, atol=1e-12)
        or not np.isclose(weight[selected_hard].sum(), len(weight) / 2.0, atol=1e-9)
        or not np.isclose(weight[~selected_hard].sum(), len(weight) / 2.0, atol=1e-9)
    ):
        raise FocusedTrainingError("query-group weighting contract failed")
    return weight


def _rss() -> Tuple[int, int]:
    try:
        import psutil

        info = psutil.Process().memory_info()
        rss = int(info.rss)
        peak = int(getattr(info, "peak_wset", rss))
        return rss, max(rss, peak)
    except Exception:
        return 0, 0


def _file_contract(path: Path, record: Mapping[str, Any]) -> None:
    if (
        _sha256(path) != record.get("sha256")
        or path.stat().st_size != int(record.get("bytes", -1))
        or int(record.get("asin_shape_matches", -1)) != 0
        or base._identity_shape_scan(path) != 0
    ):
        raise FocusedTrainingError("focused cache file contract mismatch")


def _load_focused_cache(
    cache_result_path: Path,
) -> Tuple[dict, dict, np.ndarray, np.ndarray, dict]:
    unresolved = cache_result_path
    cache_result_path = unresolved.resolve()
    if (
        ROOT not in cache_result_path.parents
        or not cache_result_path.is_file()
        or unresolved.is_symlink()
    ):
        raise FocusedTrainingError("cache result must be a local regular file")
    result = _load_json(cache_result_path)
    if not FOCUSED_CACHE_MANIFEST.is_file():
        raise FocusedTrainingError("tracked focused-cache manifest is unavailable")
    manifest = _load_json(FOCUSED_CACHE_MANIFEST)
    relative_result = cache_result_path.relative_to(ROOT).as_posix()
    if (
        result.get("schema_version") != "small-ranker-focused-cache-build.v1"
        or result.get("scope", {}).get("split") != "train_explore"
        or result.get("scope", {}).get("agent_or_evaluator_started") is not False
        or result.get("sources", {}).get("preregistration_sha256")
        != _sha256(PREREGISTRATION)
        or result.get("sources", {}).get("implementation_amendment_sha256")
        != _sha256(IMPLEMENTATION_AMENDMENT)
        or result.get("sources", {}).get("builder_sha256")
        != _sha256(ROOT / "scripts/build_small_ranker_focused_cache.py")
        or result.get("sources", {}).get("projected_features_sha256")
        != PROJECTED_SHA256
        or result.get("sources", {}).get("label_cache_sha256") != LABEL_SHA256
        or manifest.get("schema_version")
        != "small-ranker-focused-cache-manifest.v1"
        or manifest.get("experiment_id")
        != "SR-V2.5-FOCUSED-LAMBDAMART-CACHE"
        or manifest.get("result", {}).get("path") != relative_result
        or manifest.get("result", {}).get("sha256") != _sha256(cache_result_path)
        or manifest.get("sources", {}).get("preregistration_sha256")
        != _sha256(PREREGISTRATION)
        or manifest.get("sources", {}).get("implementation_amendment_sha256")
        != _sha256(IMPLEMENTATION_AMENDMENT)
        or manifest.get("sources", {}).get("builder_sha256")
        != _sha256(ROOT / "scripts/build_small_ranker_focused_cache.py")
    ):
        raise FocusedTrainingError("focused cache protocol mismatch")
    cache = result.get("cache", {})
    feature_path = _local_regular_file(cache.get("features", {}).get("path"))
    relevance_path = _local_regular_file(cache.get("relevance", {}).get("path"))
    group_path = _local_regular_file(cache.get("groups", {}).get("path"))
    _file_contract(feature_path, cache["features"])
    _file_contract(relevance_path, cache["relevance"])
    _file_contract(group_path, cache["groups"])
    if manifest.get("files") != {
        "features": cache["features"],
        "relevance": cache["relevance"],
        "groups": cache["groups"],
    }:
        raise FocusedTrainingError("focused cache manifest file records drifted")
    x = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    y = np.load(relevance_path, mmap_mode="r", allow_pickle=False)
    with np.load(group_path, allow_pickle=False) as archive:
        expected_names = {
            "session_ordinal",
            "turn_index",
            "hard_cohort",
            "outer_fold",
            "inner_fold",
        }
        if set(archive.files) != expected_names:
            raise FocusedTrainingError("focused group metadata schema mismatch")
        groups = {name: np.asarray(archive[name]) for name in archive.files}
    group_count = int(cache.get("query_groups", -1))
    if (
        x.shape != (group_count, ROWS_PER_GROUP, base.FEATURE_COUNT)
        or x.dtype != np.float32
        or y.shape != (group_count, ROWS_PER_GROUP)
        or y.dtype != np.uint8
        or not np.isfinite(np.asarray(x)).all()
        or not np.all(np.asarray(y).sum(axis=1) == 1)
        or any(value.shape != (group_count,) for value in groups.values())
        or groups["session_ordinal"].dtype != np.int16
        or groups["turn_index"].dtype != np.uint8
        or groups["hard_cohort"].dtype != np.uint8
        or groups["outer_fold"].dtype != np.uint8
        or groups["inner_fold"].dtype != np.uint8
        or np.any(groups["session_ordinal"] < 0)
        or np.any(groups["session_ordinal"] >= base.SESSION_COUNT)
        or np.any(groups["turn_index"] >= base.TURN_COUNT)
        or np.any(groups["outer_fold"] >= base.OUTER_FOLDS)
        or np.any(groups["inner_fold"] >= base.OUTER_FOLDS)
        or not set(np.unique(groups["hard_cohort"])).issubset({0, 1})
    ):
        raise FocusedTrainingError("focused cache array schema mismatch")
    keys = (
        groups["session_ordinal"].astype(np.int32) * base.TURN_COUNT
        + groups["turn_index"].astype(np.int32)
    )
    if np.any(np.diff(keys) <= 0) or len(np.unique(keys)) != group_count:
        raise FocusedTrainingError("focused query groups are duplicated or unordered")
    hard = groups["hard_cohort"].astype(bool)
    for session_value in np.unique(groups["session_ordinal"]):
        session_rows = groups["session_ordinal"] == session_value
        if len(np.unique(groups["hard_cohort"][session_rows])) != 1:
            raise FocusedTrainingError("one session crosses focused cohorts")
    if (
        int(hard.sum()) != int(manifest.get("cohorts", {}).get("hard", {}).get("query_groups", -1))
        or int((~hard).sum())
        != int(manifest.get("cohorts", {}).get("control", {}).get("query_groups", -1))
        or len(np.unique(groups["session_ordinal"][hard]))
        != int(manifest.get("cohorts", {}).get("hard", {}).get("sessions", -1))
        or len(np.unique(groups["session_ordinal"][~hard]))
        != int(manifest.get("cohorts", {}).get("control", {}).get("sessions", -1))
    ):
        raise FocusedTrainingError("focused cohort manifest counts drifted")
    return result, manifest, x, y, groups


def _validate_feature_parity(
    x: np.ndarray,
    y: np.ndarray,
    groups: Mapping[str, np.ndarray],
    projected: np.ndarray,
    positive: np.ndarray,
    full_outer: np.ndarray,
    full_inner: np.ndarray,
) -> None:
    incumbent = base._incumbent_indices(projected)
    for group in range(len(x)):
        session = int(groups["session_ordinal"][group])
        turn = int(groups["turn_index"][group])
        current = int(incumbent[session, turn])
        if not 0 <= current < 10:
            raise FocusedTrainingError("focused incumbent escaped protected C10")
        allowed = np.asarray(
            [current, *range(10, base.CANDIDATE_COUNT)], dtype=np.int64
        )
        if not np.array_equal(
            np.asarray(x[group]), np.asarray(projected[session, turn, allowed])
        ):
            raise FocusedTrainingError("focused feature row parity failed")
        expected_relevance = (allowed == int(positive[session, turn])).astype(
            np.uint8
        )
        if not np.array_equal(np.asarray(y[group]), expected_relevance):
            raise FocusedTrainingError("focused relevance label parity failed")
        if (
            int(groups["outer_fold"][group]) != int(full_outer[session])
            or int(groups["inner_fold"][group]) != int(full_inner[session])
        ):
            raise FocusedTrainingError("focused fold metadata parity failed")


def _train_model(
    x: np.ndarray,
    y: np.ndarray,
    groups: Mapping[str, np.ndarray],
    selected: np.ndarray,
    seed: int,
    model_path: Path,
) -> Tuple[Any, dict]:
    import xgboost as xgb

    indices = np.flatnonzero(selected)
    weights = partition_group_weights(
        groups["session_ordinal"], groups["hard_cohort"], selected
    )
    train_x = np.asarray(x[indices], dtype=np.float32).reshape(
        -1, base.FEATURE_COUNT
    )
    train_y = np.asarray(y[indices], dtype=np.float32).reshape(-1)
    matrix = xgb.DMatrix(train_x, label=train_y, nthread=1)
    matrix.set_group(np.full(len(indices), ROWS_PER_GROUP, dtype=np.uint32))
    matrix.set_weight(weights.astype(np.float32))
    params = dict(PARAMS)
    params["seed"] = int(seed)
    started = time.perf_counter()
    booster = xgb.train(params, matrix, num_boost_round=ROUNDS)
    train_seconds = time.perf_counter() - started
    if int(booster.num_features()) != base.FEATURE_COUNT:
        raise FocusedTrainingError("focused model feature count mismatch")
    if model_path.exists() or model_path.is_symlink():
        raise FileExistsError(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("xb") as handle:
        handle.write(bytes(booster.save_raw(raw_format="json")))
    hard = np.asarray(groups["hard_cohort"], dtype=bool)[indices]
    rss, peak = _rss()
    return booster, {
        "train_query_groups": int(len(indices)),
        "train_rows": int(len(train_y)),
        "train_sessions": int(
            len(np.unique(np.asarray(groups["session_ordinal"])[indices]))
        ),
        "hard_query_groups": int(hard.sum()),
        "control_query_groups": int((~hard).sum()),
        "group_weight_mean": round(float(weights.mean()), 12),
        "hard_weight_sum": round(float(weights[hard].sum()), 9),
        "control_weight_sum": round(float(weights[~hard].sum()), 9),
        "weight_unit": "query_group",
        "train_seconds": round(train_seconds, 6),
        "model_sha256": _sha256(model_path),
        "rss_bytes_after_fit": rss,
        "peak_working_set_bytes": peak,
    }


def _run_pass(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: Mapping[str, np.ndarray],
    projected: np.ndarray,
    full_outer: np.ndarray,
    output_dir: Path,
) -> dict:
    import xgboost as xgb

    score_path = output_dir / ("focused_outer_oof.npy" if name == "first" else "focused_outer_oof.repeat.npy")
    if score_path.exists() or score_path.is_symlink():
        raise FileExistsError(score_path)
    scores = _exclusive_npy_memmap(
        score_path,
        np.float32,
        (base.SESSION_COUNT, base.TURN_COUNT, base.CANDIDATE_COUNT),
    )
    covered = np.zeros(base.SESSION_COUNT, dtype=bool)
    fold_records = []
    per_session_ms = []
    pass_started = time.perf_counter()
    peak_working_set = 0
    for fold in range(base.OUTER_FOLDS):
        train_groups = np.asarray(groups["outer_fold"]) != fold
        held_sessions = np.flatnonzero(full_outer == fold)
        if (
            len(held_sessions) != 400
            or np.any(np.asarray(groups["outer_fold"])[train_groups] == fold)
        ):
            raise FocusedTrainingError("outer-fold isolation failed")
        model_path = output_dir / "models" / name / ("fold_%d.json" % fold)
        booster, audit = _train_model(
            x, y, groups, train_groups, SEED + fold, model_path
        )
        peak_working_set = max(
            peak_working_set, int(audit["peak_working_set_bytes"])
        )
        predict_started = time.perf_counter()
        for offset in range(0, len(held_sessions), 10):
            selected = held_sessions[offset : offset + 10]
            block = np.asarray(projected[selected], dtype=np.float32)
            tick = time.perf_counter()
            prediction = booster.predict(
                xgb.DMatrix(block.reshape(-1, base.FEATURE_COUNT), nthread=1),
                output_margin=True,
            )
            elapsed = time.perf_counter() - tick
            scores[selected] = np.asarray(prediction, dtype=np.float32).reshape(
                len(selected), base.TURN_COUNT, base.CANDIDATE_COUNT
            )
            per_session_ms.append(1000.0 * elapsed / len(selected))
            _rss_now, peak_now = _rss()
            peak_working_set = max(peak_working_set, peak_now)
        covered[held_sessions] = True
        fold_records.append(
            {
                "fold": fold,
                "held_sessions": int(len(held_sessions)),
                "held_fold_overlap": 0,
                "prediction_seconds": round(
                    time.perf_counter() - predict_started, 6
                ),
                "model_path": model_path.relative_to(ROOT).as_posix(),
                "seed": SEED + fold,
                **audit,
            }
        )
    scores.flush()
    if not np.all(covered) or not np.isfinite(np.asarray(scores)).all():
        raise FocusedTrainingError("focused OOF score coverage failed")
    serialized_parity = []
    for row in fold_records:
        fold = int(row["fold"])
        held_sessions = np.flatnonzero(full_outer == fold)
        sampled = np.asarray([held_sessions[0], held_sessions[-1]], dtype=np.int64)
        booster = xgb.Booster()
        booster.load_model(ROOT / str(row["model_path"]))
        block = np.asarray(projected[sampled], dtype=np.float32)
        actual = np.asarray(
            booster.predict(
                xgb.DMatrix(block.reshape(-1, base.FEATURE_COUNT), nthread=1),
                output_margin=True,
            ),
            dtype=np.float32,
        ).reshape(len(sampled), base.TURN_COUNT, base.CANDIDATE_COUNT)
        expected = np.asarray(scores[sampled], dtype=np.float32)
        maximum_error = float(np.max(np.abs(actual - expected)))
        order_exact = bool(
            np.array_equal(
                np.argsort(-actual, axis=2, kind="stable"),
                np.argsort(-expected, axis=2, kind="stable"),
            )
        )
        if maximum_error != 0.0 or not order_exact:
            raise FocusedTrainingError("serialized fold model parity failed")
        serialized_parity.append(
            {
                "fold": fold,
                "sampled_sessions": sampled.astype(int).tolist(),
                "rows": int(actual.size),
                "maximum_absolute_error": maximum_error,
                "c100_order_exact": order_exact,
            }
        )
    values = np.asarray(per_session_ms, dtype=np.float64)
    return {
        "name": name,
        "score_path": score_path.relative_to(ROOT).as_posix(),
        "score_sha256": _sha256(score_path),
        "score_bytes": score_path.stat().st_size,
        "folds": fold_records,
        "serialized_model_parity": serialized_parity,
        "timing_seconds": {"total": round(time.perf_counter() - pass_started, 6)},
        "prediction_ms_per_session": {
            "p50": round(float(np.quantile(values, 0.50)), 6),
            "p95": round(float(np.quantile(values, 0.95)), 6),
        },
        "peak_working_set_bytes": int(peak_working_set),
    }


def run(
    cache_result_path: Path,
    source_root: Path,
    projection_root: Path,
    output_dir: Path,
) -> dict:
    import xgboost as xgb

    started = time.perf_counter()
    if xgb.__version__ != "1.7.6":
        raise FocusedTrainingError("v2.5 requires xgboost 1.7.6")
    output_dir = output_dir.resolve()
    experiments_root = (ROOT / "experiments").resolve()
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or experiments_root not in output_dir.parents
    ):
        raise FocusedTrainingError("output directory must be new and below experiments")
    cache_result, cache_manifest, x, y, groups = _load_focused_cache(
        cache_result_path
    )
    projected_path = projection_root.resolve() / (
        "experiments/fast_track/small_ranker_fold_safe_projected_features.npy"
    )
    label_path = source_root.resolve() / (
        "experiments/fast_track/small_ranker_v1/labels_v2.npz"
    )
    if (
        not projected_path.is_file()
        or projected_path.is_symlink()
        or _sha256(projected_path) != PROJECTED_SHA256
        or not label_path.is_file()
        or label_path.is_symlink()
        or _sha256(label_path) != LABEL_SHA256
    ):
        raise FocusedTrainingError("frozen projection/fold input mismatch")
    projected = np.load(projected_path, mmap_mode="r", allow_pickle=False)
    with np.load(label_path, allow_pickle=False) as archive:
        full_outer = np.asarray(archive["outer_fold"], dtype=np.uint8)
        full_inner = np.asarray(archive["inner_fold"], dtype=np.uint8)
        positive = np.asarray(archive["positive_index"], dtype=np.int16)
    if projected.shape != (
        base.SESSION_COUNT,
        base.TURN_COUNT,
        base.CANDIDATE_COUNT,
        base.FEATURE_COUNT,
    ) or projected.dtype != np.float32:
        raise FocusedTrainingError("projected feature tensor schema mismatch")
    session = np.asarray(groups["session_ordinal"], dtype=np.int64)
    if not np.array_equal(np.asarray(groups["outer_fold"]), full_outer[session]):
        raise FocusedTrainingError("focused metadata outer-fold mismatch")
    _validate_feature_parity(
        x, y, groups, projected, positive, full_outer, full_inner
    )

    output_dir.mkdir(parents=True)
    first = _run_pass("first", x, y, groups, projected, full_outer, output_dir)
    repeat = _run_pass("repeat", x, y, groups, projected, full_outer, output_dir)
    first_path = ROOT / first["score_path"]
    repeat_path = ROOT / repeat["score_path"]
    first_scores = np.load(first_path, mmap_mode="r", allow_pickle=False)
    repeat_scores = np.load(repeat_path, mmap_mode="r", allow_pickle=False)
    incumbent = base._incumbent_indices(projected)
    first_chosen = base.choose_slot10(first_scores, incumbent)[0]
    repeat_chosen = base.choose_slot10(repeat_scores, incumbent)[0]
    score_exact = first["score_sha256"] == repeat["score_sha256"]
    decision_exact = np.array_equal(first_chosen, repeat_chosen)
    if not score_exact or not decision_exact:
        raise FocusedTrainingError("focused OOF exact repeat failed")
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.5-FOCUSED-LAMBDAMART-STAGE-A-TRAIN",
        "scope": {
            "split": "train_explore",
            "focused_numeric_cache_only": True,
            "target_used_as_training_relevance_only": True,
            "score_projection_target_free": True,
            "agent_or_evaluator_started": False,
            "held_out_splits_opened": False,
            "external_data_downloaded": False,
        },
        "sources": {
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "implementation_amendment_sha256": _sha256(
                IMPLEMENTATION_AMENDMENT
            ),
            "cache_result_path": cache_result_path.resolve().relative_to(ROOT).as_posix(),
            "cache_result_sha256": _sha256(cache_result_path.resolve()),
            "cache_manifest_sha256": _sha256(FOCUSED_CACHE_MANIFEST),
            "builder_sha256": cache_result["sources"]["builder_sha256"],
            "trainer_sha256": _sha256(Path(__file__).resolve()),
            "projected_features_sha256": PROJECTED_SHA256,
            "label_cache_sha256": LABEL_SHA256,
        },
        "cache": {
            "query_groups": int(len(x)),
            "rows": int(len(x) * ROWS_PER_GROUP),
            "hard_query_groups": int(np.asarray(groups["hard_cohort"]).sum()),
            "control_query_groups": int(
                len(x) - np.asarray(groups["hard_cohort"]).sum()
            ),
            "feature_parity_exact": True,
        },
        "model": {
            "parameters": PARAMS,
            "parameters_sha256": _canonical_sha256(PARAMS),
            "rounds": ROUNDS,
            "seed": SEED,
            "xgboost_version": xgb.__version__,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
        },
        "first": first,
        "repeat": repeat,
        "exact_repeat": {
            "score_bytes_identical": score_exact,
            "proposal_decisions_identical": decision_exact,
            "proposal_decision_sha256": hashlib.sha256(
                first_chosen.tobytes()
            ).hexdigest(),
            "repeat_proposal_decision_sha256": hashlib.sha256(
                repeat_chosen.tobytes()
            ).hexdigest(),
        },
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
    }
    result_path = output_dir / "training_result.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-result", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        args.cache_result,
        args.source_root,
        args.projection_root,
        args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
