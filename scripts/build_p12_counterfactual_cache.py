"""Build the numeric-only cache for the P12 counterfactual router.

The builder is intentionally bound to the already-closed train/explore full
artifact.  It never imports the Agent or evaluator, and it cannot select a
held-out split.  Product labels are used transiently for numeric labels and
grouping only; neither the NPZ nor its tracked manifest contains an identifier,
text feature, reversible group mapping, or string/object ndarray.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from scripts import train_p12_counterfactual_router as router


ROOT = Path(__file__).resolve().parents[1]
SPLIT = "train_explore"
SESSION_COUNT = router.SESSION_COUNT
TURN_COUNT = router.TURN_COUNT
FEATURE_NAMES = router.FEATURE_NAMES
FEATURE_COUNT = len(FEATURE_NAMES)
SUBSET_NAMES = ("full", "triage")
BAND_NAMES = ("c50_tail", "c100_only")
OUTER_FOLDS = 5
INNER_FOLDS = 5
SCHEMA_VERSION = "p12.counterfactual-cache-manifest.v1"


class CacheBuildError(RuntimeError):
    pass


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_output(path: Path, required_root: Path, label: str) -> Path:
    router._reject_forbidden_path(path)
    absolute = path if path.is_absolute() else Path.cwd() / path
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"{label} already exists: {absolute}")
    parent = absolute.parent.resolve(strict=True)
    expected_root = required_root.resolve(strict=True)
    if not parent.is_dir() or not _inside(parent, expected_root):
        raise CacheBuildError(f"{label} parent must stay under {required_root}")
    attrs = getattr(parent.stat(), "st_file_attributes", 0)
    if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise CacheBuildError(f"{label} parent must not be a reparse point")
    resolved = parent / absolute.name
    if resolved.suffix.lower() != (".npz" if label == "cache" else ".json"):
        raise CacheBuildError(f"{label} has the wrong suffix")
    return resolved


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _group_indices(labels: Sequence[str]) -> tuple[np.ndarray, int]:
    # First-observation numbering is deterministic for the frozen proxy but is
    # deliberately nonreversible: the mapping is discarded immediately.
    mapping: dict[str, int] = {}
    values = np.empty(len(labels), dtype=np.int32)
    for index, label in enumerate(labels):
        if label not in mapping:
            mapping[label] = len(mapping)
        values[index] = mapping[label]
    return values, len(mapping)


def _baseline_metadata(
    trace: Sequence[Sequence[Mapping[str, Any]]],
    labels: Sequence[str],
    eligible_from: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks = np.zeros((SESSION_COUNT, TURN_COUNT), dtype=np.uint8)
    session_hit = np.zeros(SESSION_COUNT, dtype=np.uint8)
    atomic_session_harm = np.zeros((SESSION_COUNT, TURN_COUNT), dtype=np.uint8)
    for session_index, turns in enumerate(trace):
        label = labels[session_index]
        eligible = int(eligible_from[session_index])
        eligible_hit_turns: set[int] = set()
        for turn_index, turn in enumerate(turns):
            p11 = turn["actions"]["KEEP_P11"]
            try:
                rank = p11.index(label) + 1
            except ValueError:
                rank = 0
            ranks[session_index, turn_index] = rank
            if turn_index + 1 >= eligible and rank:
                eligible_hit_turns.add(turn_index)
        session_hit[session_index] = int(bool(eligible_hit_turns))
        for turn_index in eligible_hit_turns:
            atomic_session_harm[session_index, turn_index] = int(
                ranks[session_index, turn_index] == 10
                and eligible_hit_turns == {turn_index}
            )
    return ranks, session_hit, atomic_session_harm


def _band_features(
    trace: Sequence[Sequence[Mapping[str, Any]]],
    spec: router.ProposalSpec,
) -> np.ndarray:
    proposal_count = spec.proposal_count
    x = np.empty(
        (SESSION_COUNT, TURN_COUNT, proposal_count, FEATURE_COUNT),
        dtype=np.float32,
    )
    for session_index, turns in enumerate(trace):
        for turn_index, turn in enumerate(turns):
            pool = turn[spec.pool_key]
            if len(pool) < spec.rank_stop:
                raise CacheBuildError("candidate pool is shorter than its frozen band")
            proposals = pool[spec.rank_start - 1 : spec.rank_stop]
            if len(proposals) != proposal_count:
                raise CacheBuildError("proposal band reconstruction failed")
            p11 = turn["actions"]["KEEP_P11"]
            if len(p11) != 10:
                raise CacheBuildError("P11 Top10 is incomplete")
            incumbent = p11[9]
            structured = turn["actions"]["CANDIDATE_RERANK"]
            semantic = turn["actions"]["FROZEN_SEMANTIC_RERANK"]
            previous = turns[turn_index - 1] if turn_index else None
            for proposal_index, proposal in enumerate(proposals):
                values = router._features(
                    turn_index,
                    spec.rank_start + proposal_index,
                    proposal,
                    incumbent,
                    p11,
                    structured,
                    semantic,
                    previous,
                    spec,
                )
                if len(values) != FEATURE_COUNT or not all(
                    math.isfinite(float(value)) for value in values
                ):
                    raise CacheBuildError("non-finite runtime feature row")
                x[session_index, turn_index, proposal_index] = values
    return x


def _band_labels(
    trace: Sequence[Sequence[Mapping[str, Any]]],
    labels: Sequence[str],
    eligible_from: Sequence[int],
    baseline_ranks: np.ndarray,
    baseline_hit: np.ndarray,
    spec: router.ProposalSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proposal_count = spec.proposal_count
    proposal_is_label = np.zeros(
        (SESSION_COUNT, TURN_COUNT, proposal_count), dtype=np.uint8
    )
    rescue = np.zeros_like(proposal_is_label)
    harm = np.zeros_like(proposal_is_label)
    for session_index, turns in enumerate(trace):
        label = labels[session_index]
        eligible = int(eligible_from[session_index])
        for turn_index, turn in enumerate(turns):
            pool = turn[spec.pool_key]
            proposals = pool[spec.rank_start - 1 : spec.rank_stop]
            if len(proposals) != proposal_count:
                raise CacheBuildError("label proposal band reconstruction failed")
            boundary_harm = bool(
                turn_index + 1 >= eligible
                and baseline_ranks[session_index, turn_index] == 10
            )
            for proposal_index, proposal in enumerate(proposals):
                is_label = proposal == label
                proposal_is_label[session_index, turn_index, proposal_index] = int(
                    is_label
                )
                rescue[session_index, turn_index, proposal_index] = int(
                    not baseline_hit[session_index]
                    and turn_index + 1 >= eligible
                    and is_label
                )
                harm[session_index, turn_index, proposal_index] = int(boundary_harm)
    return proposal_is_label, rescue, harm


def _feature_table_sha256(x_by_band: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, array in zip(BAND_NAMES, x_by_band, strict=True):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _triage_mask(group_index: np.ndarray, baseline_hit: np.ndarray) -> np.ndarray:
    miss_groups = set(group_index[baseline_hit == 0].tolist())
    return np.asarray(
        [int(group in miss_groups) for group in group_index], dtype=np.uint8
    )


def _assert_group_fold_integrity(
    labels: Sequence[str],
    group_index: np.ndarray,
    outer_fold: np.ndarray,
    inner_fold: np.ndarray,
) -> None:
    """Validate folds with transient labels; no label value leaves this function."""
    for group in np.unique(group_index):
        members = group_index == group
        if len(np.unique(outer_fold[members])) != 1:
            raise CacheBuildError("one group spans multiple outer folds")
        if len(np.unique(inner_fold[members])) != 1:
            raise CacheBuildError("one group spans multiple inner folds")
    outer_label_sets = [
        {label for label, fold in zip(labels, outer_fold, strict=True) if fold == index}
        for index in range(OUTER_FOLDS)
    ]
    for left in range(OUTER_FOLDS):
        for right in range(left + 1, OUTER_FOLDS):
            if outer_label_sets[left] & outer_label_sets[right]:
                raise CacheBuildError("outer fold label partitions overlap")
    if set().union(*outer_label_sets) != set(labels):
        raise CacheBuildError("outer fold label partitions are incomplete")


def _weighted_stats(
    x_by_band: Sequence[np.ndarray],
    rescue_by_band: Sequence[np.ndarray],
    harm_by_band: Sequence[np.ndarray],
    group_index: np.ndarray,
    outer_fold: np.ndarray,
    inner_fold: np.ndarray,
    full_mask: np.ndarray,
    triage_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    prefix = (len(SUBSET_NAMES), len(BAND_NAMES), OUTER_FOLDS, INNER_FOLDS)
    row_count = np.zeros(prefix, dtype=np.int64)
    sum_w = np.zeros(prefix, dtype=np.float64)
    sum_wx = np.zeros((*prefix, FEATURE_COUNT), dtype=np.float64)
    sum_wxx = np.zeros((*prefix, FEATURE_COUNT, FEATURE_COUNT), dtype=np.float64)
    sum_wy_rescue = np.zeros(prefix, dtype=np.float64)
    sum_wy_harm = np.zeros(prefix, dtype=np.float64)
    sum_wxy_rescue = np.zeros((*prefix, FEATURE_COUNT), dtype=np.float64)
    sum_wxy_harm = np.zeros((*prefix, FEATURE_COUNT), dtype=np.float64)
    subset_masks = (full_mask.astype(bool), triage_mask.astype(bool))
    for subset_index, subset in enumerate(subset_masks):
        for band_index, (x, rescue, harm) in enumerate(
            zip(x_by_band, rescue_by_band, harm_by_band, strict=True)
        ):
            rows_per_session = TURN_COUNT * x.shape[2]
            for outer in range(OUTER_FOLDS):
                for inner in range(INNER_FOLDS):
                    sessions = np.flatnonzero(
                        subset & (outer_fold == outer) & (inner_fold == inner)
                    )
                    if not len(sessions):
                        continue
                    groups, counts = np.unique(group_index[sessions], return_counts=True)
                    count_by_group = {
                        int(group): int(count)
                        for group, count in zip(groups, counts, strict=True)
                    }
                    session_weights = np.asarray(
                        [
                            1.0
                            / (count_by_group[int(group_index[session])] * rows_per_session)
                            for session in sessions
                        ],
                        dtype=np.float64,
                    )
                    flat_x = x[sessions].reshape(-1, FEATURE_COUNT).astype(
                        np.float64, copy=False
                    )
                    flat_rescue = rescue[sessions].reshape(-1).astype(
                        np.float64, copy=False
                    )
                    flat_harm = harm[sessions].reshape(-1).astype(
                        np.float64, copy=False
                    )
                    weights = np.repeat(session_weights, rows_per_session)
                    key = (subset_index, band_index, outer, inner)
                    row_count[key] = len(flat_x)
                    sum_w[key] = math.fsum(float(value) for value in weights)
                    weighted_x = flat_x * weights[:, None]
                    sum_wx[key] = weighted_x.sum(axis=0)
                    sum_wxx[key] = flat_x.T @ weighted_x
                    sum_wy_rescue[key] = float(weights @ flat_rescue)
                    sum_wy_harm[key] = float(weights @ flat_harm)
                    sum_wxy_rescue[key] = flat_x.T @ (weights * flat_rescue)
                    sum_wxy_harm[key] = flat_x.T @ (weights * flat_harm)
    return {
        "row_count": row_count,
        "sum_w": sum_w,
        "sum_wx": sum_wx,
        "sum_wxx": sum_wxx,
        "sum_wy_rescue": sum_wy_rescue,
        "sum_wy_harm": sum_wy_harm,
        "sum_wxy_rescue": sum_wxy_rescue,
        "sum_wxy_harm": sum_wxy_harm,
    }


def _numeric_array_audit(arrays: Mapping[str, np.ndarray]) -> None:
    if not arrays:
        raise CacheBuildError("cache array registry is empty")
    for name, array in arrays.items():
        if not isinstance(name, str) or not name:
            raise CacheBuildError("cache array key is invalid")
        if not isinstance(array, np.ndarray):
            raise CacheBuildError(f"cache value is not ndarray: {name}")
        if array.dtype.kind not in "biuf":
            raise CacheBuildError(f"string/object array forbidden: {name}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise CacheBuildError(f"non-finite cache array: {name}")


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _array_registry(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {"dtype": array.dtype.str, "shape": list(array.shape)}
        for name, array in sorted(arrays.items())
    }


def _fold_counts(
    mask: np.ndarray, outer_fold: np.ndarray, inner_fold: np.ndarray
) -> list[list[int]]:
    return [
        [
            int(np.sum(mask.astype(bool) & (outer_fold == outer) & (inner_fold == inner)))
            for inner in range(INNER_FOLDS)
        ]
        for outer in range(OUTER_FOLDS)
    ]


def build(cache_path: Path, manifest_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    timing: dict[str, float] = {}
    builder_path = Path(__file__).resolve()
    builder_sha256 = router._sha256(builder_path)
    cache = _safe_output(
        cache_path, ROOT / "experiments/fast_track", "cache"
    )
    manifest = _safe_output(manifest_path, ROOT / "configs", "manifest")
    # Construct, hash, and write-close the complete label-free feature table
    # before loading any label-bearing source.
    tick = time.perf_counter()
    router._validate_aggregate()
    trace, trace_identifiers = router._load_traces()
    timing["validate_and_load_trace"] = time.perf_counter() - tick
    specs = [router.PROPOSAL_SPECS[name] for name in BAND_NAMES]
    tick = time.perf_counter()
    x_by_band = [
        _band_features(trace, spec)
        for spec in specs
    ]
    feature_table_sha256 = _feature_table_sha256(x_by_band)
    for feature_table in x_by_band:
        feature_table.setflags(write=False)
    timing["feature_build_and_close"] = time.perf_counter() - tick

    # The proxy join begins only after the feature table is closed above.
    tick = time.perf_counter()
    labels, eligible_from_values = router._load_proxy()
    group_index, group_count = _group_indices(labels)
    eligible_from = np.asarray(eligible_from_values, dtype=np.uint8)
    baseline_rank, baseline_hit, atomic_session_harm = _baseline_metadata(
        trace, labels, eligible_from_values
    )
    outer_fold = router._folds(labels, OUTER_FOLDS, "outer").astype(
        np.uint8
    )
    inner_fold = router._folds(labels, INNER_FOLDS, "cache-inner").astype(
        np.uint8
    )
    _assert_group_fold_integrity(labels, group_index, outer_fold, inner_fold)
    full_mask = np.ones(SESSION_COUNT, dtype=np.uint8)
    triage_mask = _triage_mask(group_index, baseline_hit)

    proposal_label_by_band: list[np.ndarray] = []
    rescue_by_band: list[np.ndarray] = []
    harm_by_band: list[np.ndarray] = []
    for spec in specs:
        proposal_is_label, rescue, harm = _band_labels(
            trace,
            labels,
            eligible_from_values,
            baseline_rank,
            baseline_hit,
            spec,
        )
        proposal_label_by_band.append(proposal_is_label)
        rescue_by_band.append(rescue)
        harm_by_band.append(harm)
    timing["label_fold_and_triage_join"] = time.perf_counter() - tick

    arrays: dict[str, np.ndarray] = {
        "x_c50_tail": x_by_band[0],
        "x_c100_only": x_by_band[1],
        "proposal_is_label_c50_tail": proposal_label_by_band[0],
        "proposal_is_label_c100_only": proposal_label_by_band[1],
        "rescue_c50_tail": rescue_by_band[0],
        "rescue_c100_only": rescue_by_band[1],
        "harm_c50_tail": harm_by_band[0],
        "harm_c100_only": harm_by_band[1],
        "atomic_session_harm": atomic_session_harm,
        "baseline_label_rank": baseline_rank,
        "baseline_session_hit": baseline_hit,
        "eligible_from": eligible_from,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "group_index": group_index,
        "full_session_mask": full_mask,
        "triage_session_mask": triage_mask,
    }
    tick = time.perf_counter()
    all_stats = _weighted_stats(
        x_by_band,
        rescue_by_band,
        harm_by_band,
        group_index,
        outer_fold,
        inner_fold,
        full_mask,
        triage_mask,
    )
    stat_mappings: dict[str, dict[str, dict[str, str]]] = {}
    for subset_index, subset_name in enumerate(SUBSET_NAMES):
        stat_mappings[subset_name] = {}
        for band_index, band_name in enumerate(BAND_NAMES):
            mapping: dict[str, str] = {}
            for field, values in all_stats.items():
                key = f"stats_{subset_name}_{band_name}_{field}"
                arrays[key] = values[subset_index, band_index]
                mapping[field] = key
            stat_mappings[subset_name][band_name] = mapping
    timing["sufficient_statistics"] = time.perf_counter() - tick
    _numeric_array_audit(arrays)
    tick = time.perf_counter()
    _write_npz_exclusive(cache, arrays)
    cache_bytes = cache.stat().st_size
    cache_sha256 = router._sha256(cache)
    timing["cache_write_and_hash"] = time.perf_counter() - tick
    timing["through_cache_write"] = time.perf_counter() - started

    feature_schema = {
        "names": list(FEATURE_NAMES),
        "formulas_in_name_order": list(router.FEATURE_FORMULAS),
        "dtype": "float32",
        "bands": {
            "c50_tail": {"rank_min": 11, "rank_max": 50},
            "c100_only": {"rank_min": 51, "rank_max": 100},
        },
        "future_features": False,
        "identity_features": False,
    }
    label_counts = {
        "baseline_hit_sessions": int(baseline_hit.sum()),
        "baseline_miss_sessions": int(SESSION_COUNT - baseline_hit.sum()),
        "direct_boundary_harm_turns": int(
            np.sum(
                (baseline_rank == 10)
                & (
                    np.arange(1, TURN_COUNT + 1)[None, :]
                    >= eligible_from[:, None]
                )
            )
        ),
        "atomic_session_harm_turns": int(atomic_session_harm.sum()),
        "c50_rescue_rows": int(rescue_by_band[0].sum()),
        "c100_rescue_rows": int(rescue_by_band[1].sum()),
        "c50_proposal_label_rows": int(proposal_label_by_band[0].sum()),
        "c100_proposal_label_rows": int(proposal_label_by_band[1].sum()),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split": SPLIT,
        "identity_free_numeric_cache": True,
        "target_used_only_as_label": True,
        "cache": {
            "path": Path(os.path.relpath(cache, manifest.parent)).as_posix(),
            "bytes": cache_bytes,
            "sha256": cache_sha256,
            "format": "numpy-npz-numeric-only",
        },
        "arrays": _array_registry(arrays),
        "shared_arrays": {
            "baseline_label_rank": "baseline_label_rank",
            "baseline_session_hit": "baseline_session_hit",
            "eligible_from": "eligible_from",
            "outer_fold": "outer_fold",
            "inner_fold": "inner_fold",
            "group_index": "group_index",
            "full_session_mask": "full_session_mask",
            "triage_session_mask": "triage_session_mask",
            "atomic_session_harm": "atomic_session_harm",
        },
        "bands": {
            name: {
                "arrays": {
                    "x": f"x_{name}",
                    "rescue_label": f"rescue_{name}",
                    "harm_label": f"harm_{name}",
                    "proposal_is_label": f"proposal_is_label_{name}",
                }
            }
            for name in BAND_NAMES
        },
        "sources": {
            "aggregate_sha256": router.AGGREGATE_SHA256,
            "proxy_sha256": router.PROXY_SHA256,
            "trace_registry_sha256": router.TRACE_REGISTRY_SHA256,
            "combined_trace_sha256": router.COMBINED_TRACE_SHA256,
            "trace_shard_sha256": [digest for _, digest in router.TRACE_SPECS],
        },
        "feature_schema": feature_schema,
        "feature_schema_sha256": _canonical_sha256(feature_schema),
        "feature_table": {
            "sha256": feature_table_sha256,
            "phase_order": (
                "feature arrays were constructed, raw-hashed, and write-closed "
                "before proxy labels were loaded or joined"
            ),
            "write_closed_before_label_join": True,
        },
        "labels": {
            "rescue": (
                "baseline session miss and eligible-turn proposal equals the offline label"
            ),
            "harm": (
                "direct boundary removal: eligible-turn P11 rank10 equals the offline label; "
                "repeated across proposals"
            ),
            "atomic_session_harm_audit": (
                "direct boundary removal also eliminates the only eligible baseline hit"
            ),
            "counts": label_counts,
        },
        "triage": {
            "definition": (
                "all baseline-miss groups plus baseline-hit controls in the same group"
            ),
            "full_sessions": int(full_mask.sum()),
            "triage_sessions": int(triage_mask.sum()),
            "triage_groups": int(len(set(group_index[triage_mask.astype(bool)].tolist()))),
        },
        "folds": {
            "outer": OUTER_FOLDS,
            "inner": INNER_FOLDS,
            "group_count": group_count,
            "full_joint_session_counts": _fold_counts(
                full_mask, outer_fold, inner_fold
            ),
            "triage_joint_session_counts": _fold_counts(
                triage_mask, outer_fold, inner_fold
            ),
            "group_weighting": (
                "within each subset/band/joint cell every group has total raw weight one"
            ),
            "group_fold_consistency_verified": True,
            "outer_group_partitions_disjoint_verified": True,
        },
        "sufficient_stats": {
            **stat_mappings,
            "metadata": {
                "axes": ["outer_fold", "inner_fold"],
                "raw": True,
                "fields": list(all_stats),
                "weight_normalization": (
                    "each target cluster has equal raw total weight inside a subset; "
                    "the trainer restores row-weight mean one before applying alpha"
                ),
            },
        },
        "builder": {
            "path": "scripts/build_p12_counterfactual_cache.py",
            "raw_sha256": builder_sha256,
        },
        "privacy": {
            "string_or_object_arrays": 0,
            "reversible_group_mapping": False,
            "identity_values_serialized": 0,
            "text_features": 0,
        },
        "timing_seconds": {
            key: round(value, 6) for key, value in timing.items()
        },
        "iteration_target_seconds": 60,
    }
    router._recursive_audit(
        result, frozenset(trace_identifiers) | frozenset(labels)
    )
    if router._sha256(builder_path) != builder_sha256:
        raise CacheBuildError("builder changed while cache was being constructed")
    result["manifest_canonical_sha256"] = _canonical_sha256(result)
    _write_json_exclusive(manifest, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.split != SPLIT:
        raise CacheBuildError("only train_explore is permitted")
    result = build(args.cache, args.manifest)
    print(
        json.dumps(
            {
                "cache_bytes": result["cache"]["bytes"],
                "cache_sha256": result["cache"]["sha256"],
                "manifest": str(args.manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CacheBuildError, OSError, ValueError, router.RouterTrainingError) as exc:
        print(f"[p12-counterfactual-cache] {exc}", file=sys.stderr)
        raise SystemExit(1)
