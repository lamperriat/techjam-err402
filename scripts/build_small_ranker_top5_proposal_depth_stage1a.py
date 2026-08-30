"""Freeze all five target-free v2.9 top-5 runtime surfaces twice.

This thin orchestrator hash-checks and reuses the standalone Stage-0 mechanics.
It never opens the label archive, fits a selector, computes an outcome, or runs
the Agent/evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE0_PROBE = ROOT / "scripts/probe_small_ranker_top5_proposal_depth.py"
EXPECTED_STAGE0_PROBE_SHA256 = (
    "6ca765d0df519da789ebbeecf82b9629ac32298b41180e1ca951d769f1a94e64"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


if (
    not STAGE0_PROBE.is_file()
    or STAGE0_PROBE.is_symlink()
    or _sha256(STAGE0_PROBE) != EXPECTED_STAGE0_PROBE_SHA256
):
    raise RuntimeError("hash-pinned Stage-0 mechanics are unavailable")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import probe_small_ranker_top5_proposal_depth as mechanics  # noqa: E402

if (
    Path(mechanics.__file__).resolve() != STAGE0_PROBE.resolve()
    or _sha256(Path(mechanics.__file__).resolve()) != EXPECTED_STAGE0_PROBE_SHA256
):
    raise RuntimeError("imported Stage-0 mechanics do not match the hash-pinned file")


SCHEMA_VERSION = "small-ranker-top5-proposal-depth-stage1a.v1"
STAGE1A_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_9.top5_proposal_depth_"
    "stage1a_implementation_amendment.json"
)
STAGE0_MANIFEST = ROOT / (
    "configs/small_ranker_v2_9.top5_proposal_depth_stage0.manifest.json"
)
STAGE0_RESULT = ROOT / (
    "experiments/fast_track/small_ranker_v2_9/"
    "stage0_20260830T222836/stage0_result.json"
)
EXPECTED_HASHES = {
    "preregistration": "51c0a9d909e7e8d21604ff29981c8a35ca217b94e0ec9d6f8c98ca12d700cebb",
    "stage0_implementation_amendment": "3058bd3b009bc8079478d0a0c95df5d02b2691b66b40ef322d80bc3a61c3b505",
    "stage0_probe": EXPECTED_STAGE0_PROBE_SHA256,
    "stage0_manifest": "88b3f59d0580e8e3ded22159fcd7367c33c3313d0534dafce20115593b932886",
    "stage0_result": "1ccb96454936c58f28cf096f2e864b4e5dd874e46afe4de3bec15a0704e113e3",
    "stage1a_amendment": "b0c7e1cfe6ef9a56657f9898dd2f7358628471fe9380de5bae4ac564cf3324d3",
    "v28_manifest": "c57c39d8220914dcab91717b9505f7734837aeca936178167f464ec33cbbcceb",
    "v28_result": "a683f50f6acdb2ee3cc0c88507e6f5ac4f46b2a6b0599acf2e6a4abdc3d17c97",
}
STAGE0_RESULT_BYTES = 38_796
STAGE0_CACHE_BYTES = 87_813_708
STAGE0_IDENTITY = "41e5cd8563ec38e78c3c279c27937721c8e2b0c8cfa960139e28af278bc1a27a"
V28_FREEZE_IDENTITY = "78e70b6211804e677c38db550bcf6c032834d7abcfd312ac0508f8db0db3a0b2"
V28_HELD_COVERAGE_SHA256 = (
    "bbdbdfa1aad6975399baa2db9d5f554000b79024d1d04581b3f6ccff2dfc4334"
)
OUTER_FOLDS = tuple(range(5))
PASSES = ("first", "repeat")
EXPECTED_ARRAY_FILES = len(PASSES) * len(OUTER_FOLDS) * len(mechanics.PHASES) * len(
    mechanics.SURFACE_FIELDS
)


class Stage1AError(RuntimeError):
    pass


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage1AError("expected a JSON object")
    return value


def _path_below(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    parent = root.resolve()
    return resolved != parent and parent in resolved.parents


def _source_snapshot() -> Dict[str, str]:
    paths = {
        "orchestrator": Path(__file__).resolve(),
        "preregistration": mechanics.PREREGISTRATION,
        "stage0_implementation_amendment": mechanics.IMPLEMENTATION_AMENDMENT,
        "stage0_probe": STAGE0_PROBE,
        "stage0_manifest": STAGE0_MANIFEST,
        "stage0_result": STAGE0_RESULT,
        "stage1a_amendment": STAGE1A_AMENDMENT,
        "v28_manifest": mechanics.V28_MANIFEST,
        "v28_result": mechanics.V28_RESULT,
    }
    snapshot = {}
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise Stage1AError("source is missing or symlinked: %s" % name)
        snapshot[name] = _sha256(path)
    return snapshot


def _validate_protocol() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    expected_paths = {
        "preregistration": mechanics.PREREGISTRATION,
        "stage0_implementation_amendment": mechanics.IMPLEMENTATION_AMENDMENT,
        "stage0_probe": STAGE0_PROBE,
        "stage0_manifest": STAGE0_MANIFEST,
        "stage0_result": STAGE0_RESULT,
        "stage1a_amendment": STAGE1A_AMENDMENT,
        "v28_manifest": mechanics.V28_MANIFEST,
        "v28_result": mechanics.V28_RESULT,
    }
    for name, path in expected_paths.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256(path) != EXPECTED_HASHES[name]
        ):
            raise Stage1AError("Stage-1a source mismatch: %s" % name)

    prereg, _stage0_amendment, frozen_result = mechanics._validate_protocol()
    stage0_manifest = _load_json(STAGE0_MANIFEST)
    stage0_result = _load_json(STAGE0_RESULT)
    amendment = _load_json(STAGE1A_AMENDMENT)
    expected_binding = {
        "preregistration_sha256": EXPECTED_HASHES["preregistration"],
        "stage0_implementation_amendment_sha256": EXPECTED_HASHES[
            "stage0_implementation_amendment"
        ],
        "stage0_probe_sha256": EXPECTED_HASHES["stage0_probe"],
        "stage0_manifest_sha256": EXPECTED_HASHES["stage0_manifest"],
        "stage0_result_sha256": EXPECTED_HASHES["stage0_result"],
        "stage0_identity_sha256": STAGE0_IDENTITY,
        "v2_8_stage1_result_sha256": EXPECTED_HASHES["v28_result"],
    }
    stage0_budget = stage0_result.get("resource", {}).get("budget", {})
    if not (
        amendment.get("schema_version")
        == "small-ranker-top5-proposal-depth-stage1a-implementation-amendment.v1"
        and amendment.get("parent_stage0_evidence_commit")
        == "6f3765f4671deba04bde792b35f7df7c8db3536b"
        and amendment.get("source_binding") == expected_binding
        and amendment.get("stage1a_scope", {}).get("outer_folds")
        == list(OUTER_FOLDS)
        and amendment.get("stage1a_scope", {}).get("passes") == list(PASSES)
        and amendment.get("stage1a_scope", {}).get("phases")
        == list(mechanics.PHASES)
        and amendment.get("stage1a_scope", {}).get("surface_files_per_outer_pass")
        == len(mechanics.PHASES) * len(mechanics.SURFACE_FIELDS)
        and stage0_manifest.get("status") == "IMPLEMENTATION_PASS_STAGE0"
        and stage0_manifest.get("result", {}).get("sha256")
        == EXPECTED_HASHES["stage0_result"]
        and stage0_manifest.get("result", {}).get("bytes") == STAGE0_RESULT_BYTES
        and stage0_manifest.get("result", {}).get("identity_sha256")
        == STAGE0_IDENTITY
        and stage0_manifest.get("decision", {}).get("stage1a_authorized") is True
        and stage0_manifest.get("decision", {}).get("selector_or_outcome_authorized")
        is False
        and stage0_manifest.get("decision", {}).get("runtime_artifact_authorized")
        is False
        and stage0_manifest.get("privacy", {}).get("label_archive_opened") is False
        and stage0_manifest.get("privacy", {}).get("outcome_member_accesses") == 0
        and stage0_manifest.get("resource", {}).get("all_budget_gates_passed") is True
        and stage0_manifest.get("resource", {}).get("new_cache_bytes")
        == STAGE0_CACHE_BYTES
        and STAGE0_RESULT.stat().st_size == STAGE0_RESULT_BYTES
        and stage0_result.get("status") == "IMPLEMENTATION_PASS_STAGE0"
        and stage0_result.get("exact_repeat", {}).get("equal") is True
        and stage0_result.get("exact_repeat", {}).get("identity_sha256")
        == STAGE0_IDENTITY
        and stage0_result.get("privacy", {}).get("label_archive_opened") is False
        and stage0_result.get("privacy", {}).get("outcome_member_accesses") == 0
        and stage0_result.get("privacy", {}).get("held_state_or_metric_computed")
        is False
        and stage0_result.get("privacy", {}).get("agent_or_full_evaluator_started")
        is False
        and stage0_result.get("decision", {}).get("stage1a_authorized") is True
        and stage0_result.get("decision", {}).get("selector_or_outcome_authorized")
        is False
        and stage0_result.get("decision", {}).get("runtime_artifact_authorized")
        is False
        and len(stage0_budget) == 3
        and all(item.get("pass") is True for item in stage0_budget.values())
        and frozen_result.get("exact_repeat", {}).get("identity_sha256")
        == V28_FREEZE_IDENTITY
    ):
        raise Stage1AError("Stage-0 prerequisite or Stage-1a protocol drifted")
    return prereg, frozen_result


def _validate_output_root(output_root: Path) -> Path:
    unresolved = output_root
    output_root = output_root.resolve()
    allowed_root = (
        ROOT / "experiments" / "fast_track" / "small_ranker_v2_9"
    ).resolve()
    if (
        output_root.exists()
        or unresolved.is_symlink()
        or output_root.parent != allowed_root
        or not output_root.name.startswith("stage1a_")
    ):
        raise Stage1AError("output must be a new direct stage1a_* child of v2.9")
    return output_root


def _file_identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        name: record[name]
        for name in (
            "sha256",
            "array_sha256",
            "bytes",
            "shape",
            "dtype",
            "asin_shape_matches",
        )
    }


def _outer_pair_summary(
    first: Mapping[str, Any],
    repeat: Mapping[str, Any],
    outer_fold: int,
    expected_outer_identity: str,
) -> Dict[str, Any]:
    if not (
        first.get("outer_fold") == repeat.get("outer_fold") == outer_fold
        and first.get("status") == repeat.get("status")
        == "TARGET_FREE_SURFACE_COMPLETE"
        and first.get("identity") == repeat.get("identity")
        and first.get("identity_sha256") == repeat.get("identity_sha256")
        and first.get("identity", {}).get("source_outer_identity_sha256")
        == expected_outer_identity
        and repeat.get("identity", {}).get("source_outer_identity_sha256")
        == expected_outer_identity
    ):
        raise Stage1AError("outer first/repeat identity failed: %d" % outer_fold)
    if outer_fold == 0 and first.get("identity_sha256") != STAGE0_IDENTITY:
        raise Stage1AError("outer-0 Stage-0 prefix identity failed")
    for record in (first, repeat):
        privacy = record.get("privacy", {})
        if not (
            privacy.get("label_archive_opened") is False
            and privacy.get("outcome_member_accesses") == 0
            and privacy.get("held_state_or_metric_computed") is False
            and privacy.get("agent_or_evaluator_started") is False
        ):
            raise Stage1AError("outer privacy boundary failed")
    common_sources = set(first["sources"]) - {"outer_result"}
    if common_sources != set(repeat["sources"]) - {"outer_result"} or any(
        first["sources"][name] != repeat["sources"][name]
        for name in common_sources
    ):
        raise Stage1AError("outer first/repeat source snapshot failed")

    phase_summary = {}
    physical_files = 0
    for phase in mechanics.PHASES:
        left = first["phases"][phase]
        right = repeat["phases"][phase]
        expected_sessions = 400 if phase == "held_H" else 1600
        if not (
            left["sessions"] == right["sessions"] == expected_sessions
            and left["turns"] == right["turns"] == expected_sessions * 10
            and left["k1_full_surface_parity"]
            and right["k1_full_surface_parity"]
            and left["old_top1_subset"]
            and right["old_top1_subset"]
            and left["causal_latch_at_most_one"]
            and right["causal_latch_at_most_one"]
            and left["keep_composition_exact"]
            and right["keep_composition_exact"]
            and left["width_max"] <= mechanics.MAX_ACTIONS
            and right["width_max"] <= mechanics.MAX_ACTIONS
            and set(left["files"]) == set(right["files"])
            == set(mechanics.SURFACE_FIELDS)
        ):
            raise Stage1AError("outer phase invariant failed: %d/%s" % (outer_fold, phase))
        for field in mechanics.SURFACE_FIELDS:
            if _file_identity(left["files"][field]) != _file_identity(
                right["files"][field]
            ):
                raise Stage1AError(
                    "outer physical repeat failed: %d/%s/%s"
                    % (outer_fold, phase, field)
                )
            physical_files += 2
        phase_summary[phase] = {
            "sessions": left["sessions"],
            "turns": left["turns"],
            "available_action_rows": left["available_action_rows"],
            "action_turns": left["action_turns"],
            "width_mean": left["width_mean"],
            "width_p50_higher": left["width_p50_higher"],
            "width_p95_higher": left["width_p95_higher"],
            "width_max": left["width_max"],
            "session_order_sha256": left["session_order_sha256"],
            "feature_order_sha256": left["feature_order_sha256"],
            "old_top1_action_rows": left["old_top1_action_rows"],
            "k1_full_surface_parity": True,
            "old_top1_subset": True,
            "causal_latch_at_most_one": True,
            "keep_composition_exact": True,
        }
    if physical_files != 2 * len(mechanics.PHASES) * len(mechanics.SURFACE_FIELDS):
        raise Stage1AError("outer physical file count failed")
    return {
        "outer_fold": outer_fold,
        "equal": True,
        "identity_sha256": first["identity_sha256"],
        "first_identity_sha256": first["identity_sha256"],
        "repeat_identity_sha256": repeat["identity_sha256"],
        "source_outer_identity_sha256": expected_outer_identity,
        "phases": phase_summary,
        "physical_repeat_equal": True,
        "physical_files_per_pass": physical_files // 2,
        "physical_files_across_passes": physical_files,
    }


def _load_partition_orders(
    frozen_result: Mapping[str, Any], pass_name: str
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    training_orders = []
    held_orders = []
    for outer_fold in OUTER_FOLDS:
        outer_root, outer, _expected_sha = mechanics._validate_outer_result(
            frozen_result, pass_name, outer_fold
        )
        arrays = mechanics._FrozenArrays(
            outer_root, mechanics._stage0_array_records(outer, outer_fold)
        )
        training, held = mechanics._session_orders(outer, arrays)
        arrays.verify_unchanged()
        training_orders.append(np.asarray(training, dtype=np.int16).copy())
        held_orders.append(np.asarray(held, dtype=np.int16).copy())
    return tuple(training_orders), tuple(held_orders)


def _held_partition(
    first_training: Sequence[np.ndarray],
    first_held: Sequence[np.ndarray],
    repeat_training: Sequence[np.ndarray],
    repeat_held: Sequence[np.ndarray],
) -> Dict[str, Any]:
    if not (
        len(first_training)
        == len(first_held)
        == len(repeat_training)
        == len(repeat_held)
        == len(OUTER_FOLDS)
    ):
        raise Stage1AError("held partition fold count failed")
    owner = np.full(mechanics.SESSION_COUNT, 255, dtype=np.uint8)
    coverage = np.zeros(mechanics.SESSION_COUNT, dtype=np.uint8)
    per_outer = []
    for outer_fold in OUTER_FOLDS:
        t_first = np.asarray(first_training[outer_fold], dtype=np.int16)
        h_first = np.asarray(first_held[outer_fold], dtype=np.int16)
        t_repeat = np.asarray(repeat_training[outer_fold], dtype=np.int16)
        h_repeat = np.asarray(repeat_held[outer_fold], dtype=np.int16)
        mask = np.ones(mechanics.SESSION_COUNT, dtype=bool)
        if np.any((h_first < 0) | (h_first >= mechanics.SESSION_COUNT)):
            raise Stage1AError("held partition ordinal is out of range")
        mask[h_first] = False
        expected_training = np.flatnonzero(mask).astype(np.int16)
        if not (
            np.array_equal(t_first, t_repeat)
            and np.array_equal(h_first, h_repeat)
            and t_first.shape == (1600,)
            and h_first.shape == (400,)
            and len(np.unique(h_first)) == 400
            and np.array_equal(t_first, expected_training)
            and not np.intersect1d(t_first, h_first).size
        ):
            raise Stage1AError("held partition repeat or complement failed")
        if np.any(owner[h_first] != 255):
            raise Stage1AError("held partition overlap detected")
        owner[h_first] = outer_fold
        coverage[h_first] += 1
        per_outer.append(
            {
                "outer_fold": outer_fold,
                "training_sessions": 1600,
                "held_sessions": 400,
                "training_order_sha256": mechanics._array_sha256(t_first),
                "held_order_sha256": mechanics._array_sha256(h_first),
            }
        )
    if (
        np.any(owner == 255)
        or not np.all(coverage == 1)
        or mechanics._array_sha256(coverage) != V28_HELD_COVERAGE_SHA256
    ):
        raise Stage1AError("held partition does not cover every session exactly once")
    return {
        "sessions": mechanics.SESSION_COUNT,
        "per_outer_counts": [400] * len(OUTER_FOLDS),
        "unique_sessions": mechanics.SESSION_COUNT,
        "missing_sessions": 0,
        "overlap_sessions": 0,
        "coverage_count_array_sha256": mechanics._array_sha256(coverage),
        "outer_fold_by_session_sha256": mechanics._array_sha256(owner),
        "first_repeat_orders_exact": True,
        "per_outer": per_outer,
    }


def _aggregate_identity(
    records: Sequence[Mapping[str, Any]], held_partition: Mapping[str, Any]
) -> Dict[str, Any]:
    if len(records) != len(OUTER_FOLDS):
        raise Stage1AError("aggregate outer count failed")
    return {
        "schema_version": SCHEMA_VERSION,
        "proposal_depth": mechanics.TOP_K,
        "maximum_actions_per_turn": mechanics.MAX_ACTIONS,
        "stage0_probe_sha256": EXPECTED_STAGE0_PROBE_SHA256,
        "v28_freeze_identity_sha256": V28_FREEZE_IDENTITY,
        "outer_folds": [record["identity"] for record in records],
        "held_partition": dict(held_partition),
    }


def _directory_stats(path: Path) -> Tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def _verify_output_records(
    output_root: Path, pass_records: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    seen = set()
    output_bytes = 0
    for pass_name in PASSES:
        records = pass_records.get(pass_name, ())
        if len(records) != len(OUTER_FOLDS):
            raise Stage1AError("output audit outer count failed")
        for outer_fold, outer in enumerate(records):
            if int(outer.get("outer_fold", -1)) != outer_fold:
                raise Stage1AError("output audit outer order failed")
            for phase in mechanics.PHASES:
                files = outer["phases"][phase]["files"]
                if set(files) != set(mechanics.SURFACE_FIELDS):
                    raise Stage1AError("output audit surface registry failed")
                for field in mechanics.SURFACE_FIELDS:
                    record = files[field]
                    raw = record.get("path")
                    if not isinstance(raw, str) or not raw:
                        raise Stage1AError("output audit record has no path")
                    unresolved = ROOT / raw if not Path(raw).is_absolute() else Path(raw)
                    if unresolved.is_symlink():
                        raise Stage1AError("output audit rejects symlinks")
                    path = unresolved.resolve()
                    key = str(path).lower()
                    if (
                        path.suffix.lower() != ".npy"
                        or not _path_below(path, output_root)
                        or not path.is_file()
                        or key in seen
                    ):
                        raise Stage1AError("output audit path boundary failed")
                    if (
                        path.stat().st_size != int(record.get("bytes", -1))
                        or _sha256(path) != str(record.get("sha256"))
                        or int(record.get("asin_shape_matches", -1)) != 0
                        or mechanics._identity_shape_scan(path) != 0
                    ):
                        raise Stage1AError("output audit physical file failed")
                    value = mechanics._load_npy_mmap(path)
                    expected_shape = tuple(
                        int(item) for item in record.get("shape", [])
                    )
                    if (
                        not isinstance(value, np.ndarray)
                        or value.shape != expected_shape
                        or str(value.dtype) != str(record.get("dtype"))
                        or mechanics._array_sha256(value)
                        != str(record.get("array_sha256"))
                        or (
                            np.issubdtype(value.dtype, np.floating)
                            and not np.isfinite(value).all()
                        )
                    ):
                        raise Stage1AError("output audit array value failed")
                    seen.add(key)
                    output_bytes += path.stat().st_size
    if len(seen) != EXPECTED_ARRAY_FILES:
        raise Stage1AError("output audit file count failed")
    return {
        "registered_files": len(seen),
        "registered_bytes": output_bytes,
        "all_paths_unique_npy_below_output_root": True,
        "symlink_count": 0,
        "all_file_and_array_hashes_verified": True,
        "all_shapes_and_dtypes_verified": True,
        "all_float_arrays_finite": True,
        "identity_shape_matches": 0,
    }


def run(output_root: Path) -> Dict[str, Any]:
    output_root = _validate_output_root(output_root)
    started = time.perf_counter()
    prereg, frozen_result = _validate_protocol()
    source_start = _source_snapshot()
    for name, expected in EXPECTED_HASHES.items():
        if name in source_start and source_start[name] != expected:
            raise Stage1AError("source drifted before Stage-1a: %s" % name)
    output_root.mkdir(parents=True)

    pass_records: Dict[str, list] = {name: [] for name in PASSES}
    for pass_name in PASSES:
        for outer_fold in OUTER_FOLDS:
            pass_records[pass_name].append(
                mechanics._build_pass(
                    output_root / pass_name / ("outer_%d" % outer_fold),
                    pass_name,
                    outer_fold,
                    frozen_result,
                )
            )

    if len(frozen_result["outer_pairs"]) != len(OUTER_FOLDS):
        raise Stage1AError("v2.8 outer identity registry count failed")
    expected_outer_identities = {
        int(record["outer_fold"]): str(record["identity_sha256"])
        for record in frozen_result["outer_pairs"]
    }
    if set(expected_outer_identities) != set(OUTER_FOLDS):
        raise Stage1AError("v2.8 outer identity registry failed")
    pair_summaries = [
        _outer_pair_summary(
            pass_records["first"][outer_fold],
            pass_records["repeat"][outer_fold],
            outer_fold,
            expected_outer_identities[outer_fold],
        )
        for outer_fold in OUTER_FOLDS
    ]

    first_training, first_held = _load_partition_orders(frozen_result, "first")
    repeat_training, repeat_held = _load_partition_orders(frozen_result, "repeat")
    held_partition = _held_partition(
        first_training, first_held, repeat_training, repeat_held
    )
    first_identity = _aggregate_identity(pass_records["first"], held_partition)
    repeat_identity = _aggregate_identity(pass_records["repeat"], held_partition)
    if first_identity != repeat_identity:
        raise Stage1AError("Stage-1a aggregate identity differs")
    identity_sha256 = mechanics._canonical_sha256(first_identity)
    physical_output_audit = _verify_output_records(output_root, pass_records)

    source_end = _source_snapshot()
    if source_end != source_start:
        raise Stage1AError("Stage-1a source changed during execution")
    file_count, output_bytes = _directory_stats(output_root)
    if (
        file_count != EXPECTED_ARRAY_FILES
        or file_count != physical_output_audit["registered_files"]
        or output_bytes != physical_output_audit["registered_bytes"]
    ):
        raise Stage1AError("Stage-1a output array count failed")
    rss, peak = mechanics._process_memory()
    build_wall = sum(
        float(record["resource"]["wall_seconds"])
        for pass_name in PASSES
        for record in pass_records[pass_name]
    )
    total_wall = time.perf_counter() - started
    budget = prereg["resource_budget"]
    stage0_cache_bytes = STAGE0_CACHE_BYTES
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.9-STRICT-TOP5-PROPOSAL-DEPTH-STAGE1A",
        "status": "PENDING_RESOURCE_GATE",
        "evidence_boundary": "all-five-outer target-free surface freeze; no labels, selector, outcome, Agent, or evaluator",
        "first": pass_records["first"],
        "repeat": pass_records["repeat"],
        "outer_pairs": pair_summaries,
        "held_partition": held_partition,
        "physical_output_audit": physical_output_audit,
        "exact_repeat": {
            "equal": True,
            "outer_shards": len(OUTER_FOLDS),
            "identity": first_identity,
            "identity_sha256": identity_sha256,
            "per_outer_identity_sha256": [
                record["identity_sha256"] for record in pass_records["first"]
            ],
        },
        "sources": source_end,
        "privacy": {
            "label_archive_opened": False,
            "t_label_rows_copied_or_retained": 0,
            "held_label_rows_copied_retained_or_supplied": 0,
            "outcome_member_accesses": 0,
            "held_state_or_metric_computed": False,
            "agent_or_full_evaluator_started": False,
            "forbidden_split_or_external_data_opened": False,
        },
        "resource": {
            "wall_seconds": round(total_wall, 6),
            "build_pass_wall_seconds": round(build_wall, 6),
            "rss_bytes": int(rss),
            "peak_working_set_bytes": int(peak),
            "output_files_before_result": file_count,
            "output_bytes_before_result": output_bytes,
            "stage1a_result_bytes": 0,
            "output_files_after_result": file_count + 1,
            "incremental_cache_bytes": output_bytes,
            "stage0_plus_stage1a_cache_bytes": stage0_cache_bytes + output_bytes,
            "budget": {},
            "xgboost_fits": 0,
            "selector_fits": 0,
            "retrieval_queries": 0,
            "workers": 1,
        },
        "decision": {
            "stage1b_t_only_selector_implementation_authorized": False,
            "stage1b_t_only_label_attach_authorized": False,
            "tracked_stage1a_manifest_commit_required_before_label_attach": True,
            "label_archive_open_authorized_in_this_process": False,
            "held_outcome_attach_authorized": False,
            "runtime_artifact_authorized": False,
            "hr_mrr_mttc_or_technical_score_claimed": False,
        },
    }

    previous_state = None
    for _iteration in range(10):
        result_bytes = len(mechanics._serialized_json(result))
        incremental = output_bytes + result_bytes
        cumulative = stage0_cache_bytes + incremental
        checks = {
            "wall_seconds": {
                "actual": round(total_wall, 6),
                "maximum": int(budget["first_plus_repeat_wall_seconds_maximum"]),
                "pass": total_wall
                <= int(budget["first_plus_repeat_wall_seconds_maximum"]),
            },
            "peak_working_set_bytes": {
                "actual": int(peak),
                "maximum": int(budget["peak_working_set_bytes_maximum"]),
                "pass": 0 < peak <= int(budget["peak_working_set_bytes_maximum"]),
            },
            "incremental_cache_bytes": {
                "actual": incremental,
                "maximum": int(budget["new_cache_bytes_maximum"]),
                "pass": incremental <= int(budget["new_cache_bytes_maximum"]),
            },
            "stage0_plus_stage1a_cache_bytes": {
                "actual": cumulative,
                "maximum": int(budget["new_cache_bytes_maximum"]),
                "pass": cumulative <= int(budget["new_cache_bytes_maximum"]),
            },
        }
        budget_pass = all(item["pass"] for item in checks.values())
        result["resource"]["stage1a_result_bytes"] = result_bytes
        result["resource"]["incremental_cache_bytes"] = incremental
        result["resource"]["stage0_plus_stage1a_cache_bytes"] = cumulative
        result["resource"]["budget"] = checks
        result["status"] = (
            "TARGET_FREE_ALL_OUTER_SURFACES_FROZEN"
            if budget_pass
            else "IMPLEMENTATION_FAIL_STAGE1A_RESOURCE_BUDGET"
        )
        result["decision"][
            "stage1b_t_only_selector_implementation_authorized"
        ] = budget_pass
        state = (result_bytes, incremental, cumulative, budget_pass, result["status"])
        if state == previous_state:
            break
        previous_state = state
    else:
        raise Stage1AError("Stage-1a result-size accounting did not converge")
    if len(mechanics._serialized_json(result)) != result["resource"][
        "stage1a_result_bytes"
    ]:
        raise Stage1AError("Stage-1a result-size accounting drifted")
    mechanics._assert_no_identity_matches(result)
    mechanics._write_json_exclusive(output_root / "stage1a_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.output_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "identity_sha256": result["exact_repeat"]["identity_sha256"],
                "wall_seconds": result["resource"]["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return (
        0
        if result["status"] == "TARGET_FREE_ALL_OUTER_SURFACES_FROZEN"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
