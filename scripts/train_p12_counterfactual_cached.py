"""Train the P12 counterfactual router from a sealed numeric cache.

The cache builder is the only component allowed to join product labels.  This
consumer accepts numeric, identifier-free arrays and precomputed sufficient
statistics, performs nested target-cluster cross-fitting, and writes an
OOF-research-only artifact.  It never opens catalog, proxy, trace, selection,
confirmation, or held-out files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "p12.counterfactual-cache-manifest.v1"
ARTIFACT_SCHEMA = "p12.counterfactual-cached-router.v1"
BANDS = ("c50_tail", "c100_only")
STAGES = ("triage", "full")
ALPHAS = (1.0, 10.0, 100.0)
LAMBDAS = (2.0, 5.0, 10.0)
SELECTION_HARM_MULTIPLIER = 10.0
GAPS = (0.0, 0.001, 0.005, 0.01)
QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.975, 0.99)
OUTER_FOLDS = 5
INNER_FOLDS = 5
SESSIONS = 2_000
TURNS = 10
FORBIDDEN_PATH_WORDS = ("calibration", "selection", "confirmation", "sealed")
REQUIRED_SHARED = (
    "baseline_label_rank", "eligible_from", "outer_fold", "inner_fold",
    "group_index", "full_session_mask", "triage_session_mask",
    "atomic_session_harm",
)
REQUIRED_BAND = (
    "x", "rescue_label", "harm_label", "proposal_is_label",
)
STAT_FIELDS = (
    "row_count", "sum_w", "sum_wx", "sum_wxx", "sum_wy_rescue",
    "sum_wy_harm", "sum_wxy_rescue", "sum_wxy_harm",
)


class CacheTrainingError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise CacheTrainingError(f"{label} must be a regular non-symlink file")
    attrs = getattr(resolved.stat(), "st_file_attributes", 0)
    if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise CacheTrainingError(f"{label} must not be a reparse point")
    return resolved


def _reject_held_out_path(path: Path) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(word in lowered for word in FORBIDDEN_PATH_WORDS):
        raise CacheTrainingError("held-out or sealed path is forbidden")


def _json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheTrainingError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise CacheTrainingError(f"{label} must be a JSON object")
    return value


def _mapped_key(mapping: Mapping[str, Any], logical: str, section: str) -> str:
    key = mapping.get(logical)
    if not isinstance(key, str) or not key:
        raise CacheTrainingError(f"manifest lacks {section}.{logical} NPZ mapping")
    return key


def _numeric_array(npz: Any, key: str) -> np.ndarray:
    if key not in npz.files:
        raise CacheTrainingError(f"NPZ member is missing: {key}")
    value = npz[key]
    if value.dtype.kind in "OUSV" or value.dtype.hasobject:
        raise CacheTrainingError(f"NPZ member is not identifier-free numeric data: {key}")
    if value.dtype.kind not in "biuf":
        raise CacheTrainingError(f"unsupported NPZ dtype for {key}: {value.dtype}")
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise CacheTrainingError(f"NPZ member contains non-finite values: {key}")
    return value


def _validate_manifest(path: Path, band: str, stage: str) -> tuple[dict[str, Any], Path]:
    _reject_held_out_path(path)
    manifest_path = _regular_file(path, "cache manifest")
    manifest = _json_file(manifest_path, "cache manifest")
    if manifest.get("schema_version") != SCHEMA:
        raise CacheTrainingError("cache manifest schema mismatch")
    declared_manifest_hash = manifest.get("manifest_canonical_sha256")
    if not isinstance(declared_manifest_hash, str) or len(declared_manifest_hash) != 64:
        raise CacheTrainingError("cache manifest canonical SHA-256 is missing")
    canonical_manifest = dict(manifest)
    del canonical_manifest["manifest_canonical_sha256"]
    computed_manifest_hash = hashlib.sha256(json.dumps(
        canonical_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")).hexdigest()
    if computed_manifest_hash != declared_manifest_hash.lower():
        raise CacheTrainingError("cache manifest canonical SHA-256 mismatch")
    privacy = manifest.get("privacy")
    if (manifest.get("identity_free_numeric_cache") is not True
            or manifest.get("target_used_only_as_label") is not True
            or not isinstance(privacy, dict)
            or privacy.get("string_or_object_arrays") != 0
            or privacy.get("identity_values_serialized") != 0
            or privacy.get("reversible_group_mapping") is not False):
        raise CacheTrainingError("cache manifest does not close the identifier boundary")
    cache = manifest.get("cache")
    if not isinstance(cache, dict):
        raise CacheTrainingError("cache manifest lacks cache identity")
    relative = cache.get("path")
    expected_hash = cache.get("sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise CacheTrainingError("cache path must be a manifest-relative path")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise CacheTrainingError("cache SHA-256 is invalid")
    cache_path = manifest_path.parent / relative
    _reject_held_out_path(cache_path)
    cache_path = _regular_file(cache_path, "numeric cache")
    if _sha256(cache_path) != expected_hash.lower():
        raise CacheTrainingError("numeric cache SHA-256 mismatch")
    bands = manifest.get("bands")
    stats = manifest.get("sufficient_stats")
    if not isinstance(bands, dict) or not isinstance(bands.get(band), dict):
        raise CacheTrainingError(f"manifest does not declare band {band}")
    if not isinstance(stats, dict) or not isinstance(stats.get(stage), dict):
        raise CacheTrainingError(f"manifest does not declare stage {stage} statistics")
    return manifest, cache_path


def _load_cache(manifest: Mapping[str, Any], cache_path: Path, band: str, stage: str) -> dict[str, Any]:
    shared_map = manifest.get("shared_arrays")
    band_map = manifest["bands"][band].get("arrays")
    stat_map = manifest["sufficient_stats"][stage]
    if isinstance(stat_map, dict) and isinstance(stat_map.get(band), dict):
        stat_map = stat_map[band]
    if not all(isinstance(item, dict) for item in (shared_map, band_map, stat_map)):
        raise CacheTrainingError("manifest array mappings are malformed")
    with np.load(cache_path, allow_pickle=False) as npz:
        # Refuse a mixed cache even when a string/object member is not selected
        # by this band: such a member would violate the cache trust boundary.
        loaded: dict[str, np.ndarray] = {}
        for key in npz.files:
            try:
                loaded[key] = _numeric_array(npz, key)
            except ValueError as exc:
                raise CacheTrainingError(f"NPZ contains an object array: {key}") from exc
        shared = {name: loaded[_mapped_key(shared_map, name, "shared_arrays")] for name in REQUIRED_SHARED}
        arrays = {name: loaded[_mapped_key(band_map, name, f"bands.{band}.arrays")] for name in REQUIRED_BAND}
        stats = {name: loaded[_mapped_key(stat_map, name, f"sufficient_stats.{stage}")] for name in STAT_FIELDS}
    if _sha256(cache_path) != str(manifest["cache"]["sha256"]).lower():
        raise CacheTrainingError("numeric cache changed while loading")
    x = arrays["x"]
    if x.ndim != 4 or x.shape[:2] != (SESSIONS, TURNS) or x.shape[2] < 2 or x.shape[3] < 1:
        raise CacheTrainingError("x must have shape [2000,10,proposal_count,feature_count]")
    proposal_count, feature_count = x.shape[2], x.shape[3]
    row_shape = (SESSIONS, TURNS, proposal_count)
    for name in ("rescue_label", "harm_label", "proposal_is_label"):
        if arrays[name].shape != row_shape:
            raise CacheTrainingError(f"{name} shape mismatch")
        if not np.isin(arrays[name], (0, 1)).all():
            raise CacheTrainingError(f"{name} must be binary")
    if shared["baseline_label_rank"].shape != (SESSIONS, TURNS):
        raise CacheTrainingError("baseline_label_rank shape mismatch")
    if not np.isin(shared["baseline_label_rank"], np.arange(0, 11)).all():
        raise CacheTrainingError("baseline_label_rank must be in 0..10")
    if shared["eligible_from"].shape != (SESSIONS,) or not np.isin(shared["eligible_from"], np.arange(1, 11)).all():
        raise CacheTrainingError("eligible_from must be in 1..10")
    if (shared["atomic_session_harm"].shape != (SESSIONS, TURNS)
            or not np.isin(shared["atomic_session_harm"], (0, 1)).all()):
        raise CacheTrainingError("atomic_session_harm must be a binary session/turn matrix")
    for name in ("outer_fold", "inner_fold"):
        if shared[name].shape != (SESSIONS,) or not np.isin(shared[name], np.arange(5)).all():
            raise CacheTrainingError(f"{name} must contain folds 0..4")
    if shared["group_index"].shape != (SESSIONS,) or shared["group_index"].dtype.kind not in "iu":
        raise CacheTrainingError("group_index must be an integer session vector")
    for name in ("full_session_mask", "triage_session_mask"):
        if shared[name].shape != (SESSIONS,) or not np.isin(shared[name], (0, 1)).all():
            raise CacheTrainingError(f"{name} must be a binary session vector")
    # The builder must keep every product group wholly inside one outer and one inner fold.
    groups = shared["group_index"].astype(np.int64, copy=False)
    for group in np.unique(groups):
        members = groups == group
        if np.unique(shared["outer_fold"][members]).size != 1 or np.unique(shared["inner_fold"][members]).size != 1:
            raise CacheTrainingError("target-cluster fold leakage detected")
    expected_prefix = (OUTER_FOLDS, INNER_FOLDS)
    shapes = {
        "row_count": expected_prefix, "sum_w": expected_prefix,
        "sum_wx": expected_prefix + (feature_count,),
        "sum_wxx": expected_prefix + (feature_count, feature_count),
        "sum_wy_rescue": expected_prefix, "sum_wy_harm": expected_prefix,
        "sum_wxy_rescue": expected_prefix + (feature_count,),
        "sum_wxy_harm": expected_prefix + (feature_count,),
    }
    for name, shape in shapes.items():
        if stats[name].shape != shape:
            raise CacheTrainingError(f"sufficient statistic {name} shape mismatch")
    if (stats["row_count"] < 0).any() or (stats["sum_w"] < 0).any():
        raise CacheTrainingError("negative sufficient statistic count/weight")
    mask_name = f"{stage}_session_mask"
    session_mask = shared[mask_name].astype(bool, copy=False)
    if not session_mask.any():
        raise CacheTrainingError(f"{stage} session mask is empty")
    return {
        "x": x.astype(np.float32, copy=False),
        "rescue": arrays["rescue_label"].astype(np.uint8, copy=False),
        "harm": arrays["harm_label"].astype(np.uint8, copy=False),
        "atomic_harm": shared["atomic_session_harm"].astype(np.uint8, copy=False),
        "proposal_is_label": arrays["proposal_is_label"].astype(bool, copy=False),
        **shared,
        "stats": stats,
        "session_mask": session_mask,
        "proposal_count": proposal_count,
        "feature_count": feature_count,
    }


def _sum_cells(stats: Mapping[str, np.ndarray], cells: np.ndarray) -> dict[str, np.ndarray | float]:
    if cells.shape != (5, 5) or not cells.any():
        raise CacheTrainingError("empty or invalid sufficient-statistic cell mask")
    return {name: np.asarray(value[cells]).sum(axis=0, dtype=np.float64) for name, value in stats.items()}


def _ridge_from_stats(stats: Mapping[str, np.ndarray], cells: np.ndarray, alpha: float, head: str) -> dict[str, np.ndarray]:
    sums = _sum_cells(stats, cells)
    sw = float(sums["sum_w"])
    row_count = float(sums["row_count"])
    if sw <= 0 or row_count <= 0:
        raise CacheTrainingError("ridge sufficient statistics have zero weight")
    swx = np.asarray(sums["sum_wx"], dtype=np.float64)
    swxx = np.asarray(sums["sum_wxx"], dtype=np.float64)
    swy = float(sums[f"sum_wy_{head}"])
    swxy = np.asarray(sums[f"sum_wxy_{head}"], dtype=np.float64)
    mean = swx / sw
    variance = np.diag(swxx) / sw - mean * mean
    variance = np.maximum(variance, 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    centered_xx = swxx - np.outer(swx, mean) - np.outer(mean, swx) + sw * np.outer(mean, mean)
    zz = centered_xx / np.outer(scale, scale)
    zsum = (swx - sw * mean) / scale
    zy = (swxy - mean * swy) / scale
    feature_count = len(mean)
    gram = np.empty((feature_count + 1, feature_count + 1), dtype=np.float64)
    gram[0, 0], gram[0, 1:], gram[1:, 0], gram[1:, 1:] = sw, zsum, zsum, zz
    rhs = np.concatenate(([swy], zy))
    # The historical trainer divides row weights by their mean before ridge.
    # Restoring that common factor preserves alpha's exact effective strength.
    weight_mean_normalizer = row_count / sw
    gram *= weight_mean_normalizer
    rhs *= weight_mean_normalizer
    penalty = np.eye(feature_count + 1, dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    try:
        coefficient = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError as exc:
        raise CacheTrainingError("ridge solve failed") from exc
    if not all(np.isfinite(item).all() for item in (mean, scale, coefficient)):
        raise CacheTrainingError("ridge model is non-finite")
    return {"mean": mean, "scale": scale, "coefficient": coefficient}


def _predict(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64, copy=False)
    values = model["coefficient"][0] + ((flat - model["mean"]) / model["scale"]) @ model["coefficient"][1:]
    return np.clip(values, 0.0, 1.0).reshape(x.shape[:-1])


def _official(hit: np.ndarray, rank: np.ndarray, first_turn: np.ndarray) -> dict[str, float | int]:
    count = int(hit.size)
    hr = round(float(hit.mean()), 6)
    mrr = round(float(np.where(hit, 1.0 / np.maximum(rank, 1), 0.0).mean()), 6)
    mttc = round(float(np.where(hit, first_turn, 11).mean()), 6)
    efficiency = round(max(0.0, min(1.0, (11.0 - mttc) / 10.0)), 6)
    score = round(0.50 * hr + 0.30 * mrr + 0.20 * efficiency, 6)
    return {"sample_count": count, "hit_rate_at_10": hr, "mrr": mrr, "mttc": mttc,
            "efficiency": efficiency, "recommended_technical_score": score}


def _gate_sweep(
    cache: Mapping[str, Any],
    scores: np.ndarray,
    session_mask: np.ndarray,
    frozen_thresholds: np.ndarray | None = None,
    frozen_gaps: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    # All parameter combinations are evaluated as broadcast tensors; there is no
    # Python loop over sessions, turns, proposals, thresholds, or gaps.
    best_index = np.argmax(scores, axis=2)
    best = np.take_along_axis(scores, best_index[..., None], axis=2)[..., 0]
    runner = np.partition(scores, -2, axis=2)[..., -2]
    proposal_hit = np.take_along_axis(cache["proposal_is_label"], best_index[..., None], axis=2)[..., 0]
    if (frozen_thresholds is None) != (frozen_gaps is None):
        raise CacheTrainingError("frozen threshold and gap must be supplied together")
    if frozen_thresholds is None:
        thresholds = np.unique(np.quantile(best[session_mask], QUANTILES))
        gate_threshold = np.repeat(thresholds, len(GAPS))
        gate_gap = np.tile(np.asarray(GAPS, dtype=np.float64), len(thresholds))
    else:
        gate_threshold = np.asarray(frozen_thresholds, dtype=np.float64).reshape(-1)
        gate_gap = np.asarray(frozen_gaps, dtype=np.float64).reshape(-1)
        if gate_threshold.shape != gate_gap.shape or not np.isfinite(gate_threshold).all() or not np.isfinite(gate_gap).all():
            raise CacheTrainingError("invalid frozen gate arrays")
    active = (best[None, :, :] >= gate_threshold[:, None, None]) & (
        best[None, :, :] - runner[None, :, :] >= gate_gap[:, None, None]
    )
    turns = np.arange(1, TURNS + 1)[None, None, :]
    eligible = turns >= np.asarray(cache["eligible_from"])[None, :, None]
    baseline_rank = np.asarray(cache["baseline_label_rank"])[None, :, :]
    hit_rank = np.where(
        baseline_rank > 0,
        np.where((baseline_rank <= 9) | ~active, baseline_rank, 0),
        0,
    )
    # A slot-10 admission can only create a rank-10 hit when Top1-9 did not
    # already hit; it must never degrade an existing Top1-9 label hit.
    hit_rank = np.where(active & proposal_hit[None, :, :] & (hit_rank == 0), 10, hit_rank)
    hit_rank = np.where(eligible, hit_rank, 0)
    hit_turn = hit_rank > 0
    any_hit = hit_turn.any(axis=2)
    first_index = np.argmax(hit_turn, axis=2)
    first_turn = np.where(any_hit, first_index + 1, 11)
    first_rank = np.take_along_axis(hit_rank, first_index[..., None], axis=2)[..., 0]
    reachable = turns <= first_turn[:, :, None]
    activation = (active & reachable).sum(axis=2)
    atomic_harm_count = (
        active & reachable & np.asarray(cache["atomic_harm"])[None, :, :].astype(bool)
    ).sum(axis=2)
    baseline_hit_turn = (baseline_rank > 0) & eligible
    baseline_hit = baseline_hit_turn.any(axis=2)[0]
    baseline_first_index = np.argmax(baseline_hit_turn, axis=2)[0]
    baseline_first_turn = np.where(baseline_hit, baseline_first_index + 1, 11)
    baseline_first_rank = np.take_along_axis(
        baseline_rank[0], baseline_first_index[:, None], axis=1
    )[:, 0]
    indices = np.flatnonzero(session_mask)
    rows: list[dict[str, Any]] = []
    for gate in range(len(gate_threshold)):
        policy_hit = any_hit[gate, indices]
        base = baseline_hit[indices]
        m2h = int((~base & policy_hit).sum())
        h2m = int((base & ~policy_hit).sum())
        rows.append({
            "threshold": float(gate_threshold[gate]), "runner_gap": float(gate_gap[gate]),
            "miss_to_hit": m2h, "hit_to_miss": h2m, "net_rescue": m2h - h2m,
            "activation_turns": int(activation[gate, indices].sum()),
            "activation_sessions": int((activation[gate, indices] > 0).sum()),
            "atomic_harm_count": int(atomic_harm_count[gate, indices].sum()),
            "policy_hits": int(policy_hit.sum()),
            "policy_reciprocal_rank_sum": float(np.where(
                policy_hit, 1.0 / np.maximum(first_rank[gate, indices], 1), 0.0
            ).sum()),
            "policy_first_turn_sum": int(np.where(policy_hit, first_turn[gate, indices], 11).sum()),
            "baseline_hits": int(base.sum()),
            "baseline_reciprocal_rank_sum": float(np.where(
                base, 1.0 / np.maximum(baseline_first_rank[indices], 1), 0.0
            ).sum()),
            "baseline_first_turn_sum": int(np.where(base, baseline_first_turn[indices], 11).sum()),
            "policy": _official(policy_hit, first_rank[gate, indices], first_turn[gate, indices]),
            "baseline": _official(base, baseline_first_rank[indices], baseline_first_turn[indices]),
        })
    tensors = {"best_index": best_index, "best": best, "runner": runner}
    return rows, tensors


def _choice_key(row: Mapping[str, Any], lam: float, alpha: float) -> tuple[float, ...]:
    m2h, h2m = int(row["miss_to_hit"]), int(row["hit_to_miss"])
    net = m2h - h2m
    if h2m == 0 and m2h > 0:
        risk_tier = 3
    elif h2m <= 1 and net > 0:
        risk_tier = 2
    elif net > 0:
        risk_tier = 1
    else:
        risk_tier = 0
    objective = m2h - SELECTION_HARM_MULTIPLIER * h2m
    return (risk_tier, -h2m, objective, net, m2h,
            -int(row["atomic_harm_count"]), -int(row["activation_turns"]),
            float(row["threshold"]), float(row["runner_gap"]), lam, alpha)


def _select(
    cache: Mapping[str, Any], outer_held: int | None
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    outer = np.asarray(cache["outer_fold"])
    inner = np.asarray(cache["inner_fold"])
    stage_mask = np.asarray(cache["session_mask"])
    train_sessions = stage_mask & (True if outer_held is None else outer != outer_held)
    best_choice: tuple[tuple[float, ...], dict[str, Any]] | None = None
    best_by_lambda: dict[float, tuple[tuple[float, ...], dict[str, Any]]] = {}
    timing = {"fit": 0.0, "predict": 0.0, "sweep": 0.0}
    for alpha in ALPHAS:
        rescue_oof = np.full(cache["x"].shape[:-1], np.nan, dtype=np.float32)
        harm_oof = np.full_like(rescue_oof, np.nan)
        for valid_fold in range(INNER_FOLDS):
            cells = np.ones((5, 5), dtype=bool)
            if outer_held is not None:
                cells[outer_held, :] = False
            cells[:, valid_fold] = False
            tick = time.perf_counter()
            rescue_model = _ridge_from_stats(cache["stats"], cells, alpha, "rescue")
            harm_model = _ridge_from_stats(cache["stats"], cells, alpha, "harm")
            timing["fit"] += time.perf_counter() - tick
            valid = train_sessions & (inner == valid_fold)
            if valid.any():
                tick = time.perf_counter()
                rescue_oof[valid] = _predict(rescue_model, cache["x"][valid])
                harm_oof[valid] = _predict(harm_model, cache["x"][valid])
                timing["predict"] += time.perf_counter() - tick
        if not np.isfinite(rescue_oof[train_sessions]).all() or not np.isfinite(harm_oof[train_sessions]).all():
            raise CacheTrainingError("inner OOF prediction coverage is incomplete")
        for lam in LAMBDAS:
            scores = rescue_oof - lam * harm_oof
            tick = time.perf_counter()
            rows, _ = _gate_sweep(cache, scores, train_sessions)
            timing["sweep"] += time.perf_counter() - tick
            for row in rows:
                choice = {**row, "alpha": alpha, "harm_multiplier": lam}
                key = _choice_key(row, lam, alpha)
                if best_choice is None or key > best_choice[0]:
                    best_choice = (key, choice)
                if lam not in best_by_lambda or key > best_by_lambda[lam][0]:
                    best_by_lambda[lam] = (key, choice)
    if best_choice is None:
        raise CacheTrainingError("model selection produced no choice")
    comparison = [
        {key: value for key, value in best_by_lambda[lam][1].items()
         if key not in ("policy", "baseline")}
        for lam in LAMBDAS
    ]
    return best_choice[1], comparison, timing


def _model_for_outer(cache: Mapping[str, Any], outer_held: int, alpha: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    cells = np.ones((5, 5), dtype=bool)
    cells[outer_held, :] = False
    return (_ridge_from_stats(cache["stats"], cells, alpha, "rescue"),
            _ridge_from_stats(cache["stats"], cells, alpha, "harm"))


def _serialize_model(model: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {name: [round(float(item), 15) for item in value] for name, value in model.items()}


def _positive_triage(
    path: Path | None, band: str, manifest_canonical_sha256: str
) -> tuple[str, dict[str, Any]]:
    if path is None:
        raise CacheTrainingError("full stage requires --triage-artifact")
    _reject_held_out_path(path)
    resolved = _regular_file(path, "triage artifact")
    value = _json_file(resolved, "triage artifact")
    if (value.get("schema_version") != ARTIFACT_SCHEMA or value.get("stage") != "triage"
            or value.get("band") != band
            or value.get("source", {}).get("manifest_canonical_sha256")
            != manifest_canonical_sha256):
        raise CacheTrainingError("full stage requires a compatible triage artifact")
    outcome = value.get("oof_outcome")
    if not isinstance(outcome, dict) or not isinstance(outcome.get("net_rescue"), int) or outcome["net_rescue"] <= 0:
        raise CacheTrainingError("full stage rejected: triage net_rescue is not positive")
    return _sha256(resolved), value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _reject_held_out_path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def train(manifest_path: Path, band: str, stage: str, output: Path, triage_artifact: Path | None) -> dict[str, Any]:
    started = time.perf_counter()
    load_started = time.perf_counter()
    manifest, cache_path = _validate_manifest(manifest_path, band, stage)
    cache = _load_cache(manifest, cache_path, band, stage)
    manifest_file_sha256 = _sha256(_regular_file(manifest_path, "cache manifest"))
    manifest_canonical_sha256 = str(manifest["manifest_canonical_sha256"])
    triage_sha = None
    if stage == "full":
        triage_sha, _ = _positive_triage(
            triage_artifact, band, manifest_canonical_sha256
        )
    load_seconds = time.perf_counter() - load_started
    fit_seconds = predict_seconds = sweep_seconds = 0.0
    outer_choices: list[dict[str, Any]] = []
    oof_scores = np.full(cache["x"].shape[:-1], np.nan, dtype=np.float32)
    for outer_fold in range(OUTER_FOLDS):
        choice, lambda_comparison, select_timing = _select(cache, outer_fold)
        fit_seconds += select_timing["fit"]
        predict_seconds += select_timing["predict"]
        sweep_seconds += select_timing["sweep"]
        fit_started = time.perf_counter()
        models = _model_for_outer(cache, outer_fold, float(choice["alpha"]))
        fit_seconds += time.perf_counter() - fit_started
        held = np.asarray(cache["session_mask"]) & (np.asarray(cache["outer_fold"]) == outer_fold)
        predict_started = time.perf_counter()
        oof_scores[held] = _predict(models[0], cache["x"][held]) - float(choice["harm_multiplier"]) * _predict(models[1], cache["x"][held])
        predict_seconds += time.perf_counter() - predict_started
        outer_choice = {
            key: value for key, value in choice.items() if key not in ("policy", "baseline")
        }
        outer_choice["lambda_comparison"] = lambda_comparison
        outer_choices.append(outer_choice)
    active_sessions = np.asarray(cache["session_mask"])
    if not np.isfinite(oof_scores[active_sessions]).all():
        raise CacheTrainingError("outer OOF prediction coverage is incomplete")
    # Each outer fold uses only its independently selected threshold and gap.
    aggregate = {"miss_to_hit": 0, "hit_to_miss": 0, "net_rescue": 0,
                 "atomic_harm_count": 0,
                 "activation_turns": 0, "activation_sessions": 0}
    official_sums = {"policy_hits": 0, "policy_reciprocal_rank_sum": 0.0,
                     "policy_first_turn_sum": 0, "baseline_hits": 0,
                     "baseline_reciprocal_rank_sum": 0.0,
                     "baseline_first_turn_sum": 0, "sessions": 0}
    for fold, choice in enumerate(outer_choices):
        held = active_sessions & (np.asarray(cache["outer_fold"]) == fold)
        rows, _ = _gate_sweep(
            cache, oof_scores, held,
            np.asarray([choice["threshold"]]), np.asarray([choice["runner_gap"]]),
        )
        row = rows[0]
        for key in aggregate:
            aggregate[key] += int(row[key])
        n = int(row["policy"]["sample_count"])
        official_sums["sessions"] += n
        for key in tuple(official_sums):
            if key != "sessions":
                official_sums[key] += row[key]
    # Fit the deliverable research model after nested OOF evaluation.
    final_choice, final_lambda_comparison, select_timing = _select(cache, None)
    fit_seconds += select_timing["fit"]
    predict_seconds += select_timing["predict"]
    sweep_seconds += select_timing["sweep"]
    cells = np.ones((5, 5), dtype=bool)
    fit_started = time.perf_counter()
    final_models = (_ridge_from_stats(cache["stats"], cells, float(final_choice["alpha"]), "rescue"),
                    _ridge_from_stats(cache["stats"], cells, float(final_choice["alpha"]), "harm"))
    fit_seconds += time.perf_counter() - fit_started
    n = official_sums["sessions"]
    def official_from_sums(prefix: str) -> dict[str, float | int]:
        hr = round(int(official_sums[f"{prefix}_hits"]) / n, 6)
        mrr = round(float(official_sums[f"{prefix}_reciprocal_rank_sum"]) / n, 6)
        mttc = round(int(official_sums[f"{prefix}_first_turn_sum"]) / n, 6)
        efficiency = round(max(0.0, min(1.0, (11.0 - mttc) / 10.0)), 6)
        return {"sample_count": n, "hit_rate_at_10": hr, "mrr": mrr, "mttc": mttc,
                "efficiency": efficiency,
                "recommended_technical_score": round(.5 * hr + .3 * mrr + .2 * efficiency, 6)}
    policy = official_from_sums("policy")
    baseline = official_from_sums("baseline")
    elapsed = time.perf_counter() - started
    artifact = {
        "schema_version": ARTIFACT_SCHEMA, "stage": stage, "band": band,
        "promotion_status": "OOF_RESEARCH_ONLY_NOT_RUNTIME_DEPLOYABLE",
        "source": {
            "manifest_canonical_sha256": manifest_canonical_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "cache_sha256": _sha256(cache_path),
            "triage_artifact_sha256": triage_sha,
            "trainer_sha256": _sha256(Path(__file__).resolve()),
        },
        "protocol": {"nested_target_cluster_outer_folds": 5, "inner_folds": 5,
                     "target_only_label": True, "identifier_free_runtime_features": True,
                     "feature_count": int(cache["feature_count"]), "proposal_count": int(cache["proposal_count"]),
                     "selection_outcome_harm_multiplier": SELECTION_HARM_MULTIPLIER,
                     "selection_order": "positive-zero-h2m,positive-near-zero-h2m,positive-net,frozen-objective,less-activation,conservative-gate",
                     "near_zero_hit_to_miss": "at most one inner-OOF hit-to-miss"},
        "oof_outcome": {**aggregate, "baseline_official": baseline, "policy_official": policy},
        "outer_choices": outer_choices,
        "final_lambda_comparison": final_lambda_comparison,
        "model": {"alpha": final_choice["alpha"], "harm_multiplier": final_choice["harm_multiplier"],
                  "threshold": final_choice["threshold"], "runner_gap": final_choice["runner_gap"],
                  "rescue_head": _serialize_model(final_models[0]), "harm_head": _serialize_model(final_models[1])},
        "timing_seconds": {"load": round(load_seconds, 6), "fit": round(fit_seconds, 6),
                           "predict": round(predict_seconds, 6), "sweep": round(sweep_seconds, 6),
                           "total": round(elapsed, 6), "target_under_60_seconds": elapsed < 60.0},
    }
    _write_exclusive(output, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--band", choices=BANDS, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--triage-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    result = train(args.manifest, args.band, args.stage, output, args.triage_artifact)
    print(json.dumps({"output": str(output.resolve()), "oof_outcome": result["oof_outcome"],
                      "timing_seconds": result["timing_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CacheTrainingError, FileExistsError, OSError, ValueError) as exc:
        print(f"[p12-counterfactual-cached] {exc}", file=sys.stderr)
        raise SystemExit(1)
