"""Build the label-free outer-0 mechanics probe for v2.9 proposal depth.

The probe reuses only physically frozen v2.8 score/runtime arrays.  It never
opens the label archive, fits a model, or computes an outcome.  K=1 must first
reproduce the complete v2.8 runtime surface before the fixed K=5 surface is
written for both independently rebuilt score passes.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "small-ranker-top5-proposal-depth-stage0.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_9.top5_proposal_depth_preregistration.json"
)
IMPLEMENTATION_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_9.top5_proposal_depth_implementation_amendment.json"
)
V28_MANIFEST = ROOT / (
    "configs/small_ranker_v2_8.strict_outer_restack_stage1.manifest.json"
)
V28_RESULT = ROOT / (
    "experiments/fast_track/small_ranker_v2_8/"
    "stage1_20260830T1834/stage1_cache_result.json"
)
EXPECTED_HASHES = {
    "preregistration": "51c0a9d909e7e8d21604ff29981c8a35ca217b94e0ec9d6f8c98ca12d700cebb",
    "implementation_amendment": "3058bd3b009bc8079478d0a0c95df5d02b2691b66b40ef322d80bc3a61c3b505",
    "v28_manifest": "c57c39d8220914dcab91717b9505f7734837aeca936178167f464ec33cbbcceb",
    "v28_result": "a683f50f6acdb2ee3cc0c88507e6f5ac4f46b2a6b0599acf2e6a4abdc3d17c97",
    "projected_features": "cd9b075b923c31afe10b9a9c6d720de22e221c93de961595ff484ea4f532b90a",
}
V28_IDENTITY = "78e70b6211804e677c38db550bcf6c032834d7abcfd312ac0508f8db0db3a0b2"
SESSION_COUNT = 2_000
TURN_COUNT = 10
CANDIDATE_COUNT = 100
FEATURE_COUNT = 133
OUTER_FOLDS = 5
TOP_K = 5
FAMILY_NAMES = ("pairwise", "rrf3", "focused_lambdamart")
FAMILY_COUNT = len(FAMILY_NAMES)
MAX_ACTIONS = TOP_K * FAMILY_COUNT
EXPECTED_STAGE0_ARRAY_FILES = 36
PHASES = ("oof_T", "reference_T", "held_H")
FEATURE_NAMES = (
    "current_policy_active",
    "turn_fraction",
    "prior_current_activation_count_fraction",
    "actionable_unique_count_fraction",
    "pairwise_support",
    "rrf3_support",
    "focused_support",
    "action_rank_fraction_under_current",
    "action_rank_fraction_under_pairwise",
    "action_rank_fraction_under_rrf3",
    "action_rank_fraction_under_focused",
    "current_choice_rank_fraction_under_pairwise",
    "current_choice_rank_fraction_under_rrf3",
    "current_choice_rank_fraction_under_focused",
    "action_coverage_rank_fraction",
    "action_minus_current_top10_route_agreement",
    "action_minus_current_active_token_recall",
    "action_minus_current_hard_clause_coverage",
    "action_minus_current_constraint_conflict_sum",
)
PROJECTED_FEATURE_INDEX = {
    "coverage_rank_fraction": 2,
    "top10_route_agreement_fraction": 22,
    "active_token_recall": 71,
    "category_conflict": 82,
    "material_conflict": 86,
    "color_conflict": 90,
    "size_conflict": 94,
    "style_conflict": 98,
    "brand_conflict": 102,
    "price_conflict": 106,
    "feature_conflict": 110,
    "use_case_conflict": 114,
    "hard_clause_coverage": 115,
}
CONSTRAINT_SLOTS = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "price",
    "feature",
    "use_case",
)
BIT_COUNT = np.asarray(
    [bin(value).count("1") for value in range(256)], dtype=np.uint8
)
SURFACE_FIELDS = (
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


class Top5ProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExpandedRuntimeSurface:
    current_chosen: np.ndarray
    current_activation: np.ndarray
    current_choice: np.ndarray
    incumbent: np.ndarray
    family_choices: np.ndarray
    candidates: np.ndarray
    source_mask: np.ndarray
    available: np.ndarray
    features: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Top5ProbeError("expected a JSON object")
    return value


def _path_below(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    parent = root.resolve()
    return resolved != parent and parent in resolved.parents


def _record_path(
    record: Mapping[str, Any], source_root: Path, required_suffix: str
) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise Top5ProbeError("file record has no path")
    unresolved = ROOT / raw if not Path(raw).is_absolute() else Path(raw)
    if unresolved.is_symlink():
        raise Top5ProbeError("symlinked frozen input is forbidden")
    path = unresolved.resolve()
    if (
        path.suffix.lower() != required_suffix
        or not _path_below(path, source_root)
        or not path.is_file()
    ):
        raise Top5ProbeError("file record escapes the frozen source root")
    return path


def _load_npy_mmap(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".npy" or path.is_symlink() or not path.is_file():
        raise Top5ProbeError("only regular .npy inputs may be loaded")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _identity_shape_scan(path: Path) -> int:
    pattern = re.compile(
        rb"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE
    )
    matches = 0
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            payload = overlap + chunk
            matches += len(pattern.findall(payload))
            overlap = payload[-9:]
    return matches


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = _serialized_json(value)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _serialized_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _write_npy_exclusive(path: Path, value: np.ndarray) -> Dict[str, Any]:
    if path.suffix.lower() != ".npy":
        raise Top5ProbeError("Stage 0 outputs must be .npy files")
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(value)
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "array_sha256": _array_sha256(array),
        "bytes": path.stat().st_size,
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "asin_shape_matches": _identity_shape_scan(path),
    }


def _assert_no_identity_matches(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "asin_shape_matches" and int(child) != 0:
                raise Top5ProbeError("output identity-shape scan failed")
            _assert_no_identity_matches(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_identity_matches(child)


def _process_memory() -> Tuple[int, int]:
    if os.name != "nt":
        try:
            import resource

            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            peak = peak if sys.platform == "darwin" else peak * 1024
            return peak, peak
        except Exception:
            return 0, 0

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = (
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        )

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    get_process_memory_info.restype = wintypes.BOOL
    if not get_process_memory_info(
        get_current_process(), ctypes.byref(counters), counters.cb
    ):
        return 0, 0
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


class _FrozenArrays:
    def __init__(
        self,
        source_root: Path,
        allowed_records: Sequence[Mapping[str, Any]],
    ) -> None:
        self.source_root = source_root.resolve()
        self.cache: Dict[str, np.ndarray] = {}
        self.allowed: Dict[str, str] = {}
        self.allowed_paths: Dict[str, Path] = {}
        self.allowed_records: Dict[str, Dict[str, Any]] = {}
        for record in allowed_records:
            path = _record_path(record, self.source_root, ".npy")
            key = str(path).lower()
            fingerprint = _canonical_sha256(dict(record))
            if key in self.allowed and self.allowed[key] != fingerprint:
                raise Top5ProbeError("conflicting records name one frozen array")
            self.allowed[key] = fingerprint
            self.allowed_paths[key] = path
            self.allowed_records[key] = dict(record)
        if len(self.allowed) != EXPECTED_STAGE0_ARRAY_FILES:
            raise Top5ProbeError("Stage 0 array allow-list is incomplete")

    def load(self, record: Mapping[str, Any]) -> np.ndarray:
        path = _record_path(record, self.source_root, ".npy")
        key = str(path).lower()
        if self.allowed.get(key) != _canonical_sha256(dict(record)):
            raise Top5ProbeError("array is outside the Stage 0 allow-list")
        if int(record.get("asin_shape_matches", -1)) != 0:
            raise Top5ProbeError("frozen array identity scan is not clean")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise Top5ProbeError("frozen array byte count drifted")
        if _sha256(path) != str(record.get("sha256")):
            raise Top5ProbeError("frozen array file hash drifted")
        if key not in self.cache:
            value = _load_npy_mmap(path)
            expected_shape = tuple(int(item) for item in record.get("shape", []))
            if value.shape != expected_shape or str(value.dtype) != str(record.get("dtype")):
                raise Top5ProbeError("frozen array schema drifted")
            if _array_sha256(value) != str(record.get("array_sha256")):
                raise Top5ProbeError("frozen array value hash drifted")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise Top5ProbeError("frozen score or feature array is non-finite")
            self.cache[key] = value
        return self.cache[key]

    def verify_unchanged(self) -> None:
        for key, path in self.allowed_paths.items():
            record = self.allowed_records[key]
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != int(record.get("bytes", -1))
                or _sha256(path) != str(record.get("sha256"))
            ):
                raise Top5ProbeError("frozen array changed during Stage 0")


def _validate_protocol() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for name, path in (
        ("preregistration", PREREGISTRATION),
        ("implementation_amendment", IMPLEMENTATION_AMENDMENT),
        ("v28_manifest", V28_MANIFEST),
        ("v28_result", V28_RESULT),
    ):
        if not path.is_file() or path.is_symlink() or _sha256(path) != EXPECTED_HASHES[name]:
            raise Top5ProbeError("protocol source mismatch: %s" % name)
    prereg = _load_json(PREREGISTRATION)
    amendment = _load_json(IMPLEMENTATION_AMENDMENT)
    manifest = _load_json(V28_MANIFEST)
    result = _load_json(V28_RESULT)
    expected_binding = {
        "preregistration_sha256": EXPECTED_HASHES["preregistration"],
        "v2_8_stage1_result_sha256": EXPECTED_HASHES["v28_result"],
        "v2_8_stage1_manifest_sha256": EXPECTED_HASHES["v28_manifest"],
        "projected_feature_sha256": EXPECTED_HASHES["projected_features"],
    }
    if not (
        prereg.get("schema_version")
        == "small-ranker-top5-proposal-depth-preregistration.v1"
        and prereg.get("status") == "PREREGISTERED_NO_V2_9_OUTCOME_OPENED"
        and prereg.get("single_algorithmic_variable", {}).get("new") == TOP_K
        and prereg.get("proposal_contract", {}).get("maximum_unique_actions_per_turn")
        == MAX_ACTIONS
        and amendment.get("schema_version")
        == "small-ranker-top5-proposal-depth-implementation-amendment.v1"
        and amendment.get("experiment_id")
        == "SR-V2.9-STRICT-TOP5-PROPOSAL-DEPTH"
        and amendment.get("parent_preregistration_commit") == "6ad3518"
        and amendment.get("source_binding") == expected_binding
        and amendment.get("stage0_scope", {}).get("outer_fold") == 0
        and amendment.get("stage0_scope", {}).get("passes")
        == ["first", "repeat"]
        and amendment.get("stage0_scope", {}).get("phases") == list(PHASES)
        and manifest.get("status") == "NO_GO_TARGET_FREE_POLICY_IDENTITY"
        and result.get("schema_version")
        == "small-ranker-strict-outer-restack-stage1-freeze.v1"
        and result.get("status") == "CACHE_REPEAT_FROZEN"
        and result.get("exact_repeat", {}).get("equal") is True
        and result.get("exact_repeat", {}).get("identity_sha256") == V28_IDENTITY
        and result.get("privacy", {}).get("outcome_label_archive_opened") is False
        and result.get("privacy", {}).get("held_state_or_metric_computed") is False
    ):
        raise Top5ProbeError("frozen protocol semantics drifted")
    return prereg, amendment, result


def _validate_outer_result(
    frozen_result: Mapping[str, Any], pass_name: str, outer_fold: int
) -> Tuple[Path, Dict[str, Any], str]:
    records = frozen_result.get("audited_worker_result_files", {}).get(pass_name, [])
    if not isinstance(records, list) or len(records) != OUTER_FOLDS:
        raise Top5ProbeError("frozen worker result registry is incomplete")
    suffix = "/%s/outer_%d/outer_complete.json" % (pass_name, outer_fold)
    matches = [
        row
        for row in records
        if str(row.get("path", "")).replace("\\", "/").endswith(suffix)
    ]
    if len(matches) != 1:
        raise Top5ProbeError("outer result path registry is ambiguous")
    record = matches[0]
    stage_root = (ROOT / str(frozen_result.get("stage_root"))).resolve()
    path = _record_path(record, stage_root, ".json")
    if path.stat().st_size != int(record.get("bytes", -1)) or _sha256(path) != record.get(
        "sha256"
    ):
        raise Top5ProbeError("outer result record drifted")
    result = _load_json(path)
    pair_matches = [
        row
        for row in frozen_result.get("outer_pairs", [])
        if int(row.get("outer_fold", -1)) == outer_fold
    ]
    if len(pair_matches) != 1:
        raise Top5ProbeError("outer identity registry is ambiguous")
    expected_identity = pair_matches[0].get("identity_sha256")
    if not (
        result.get("schema_version")
        == "small-ranker-strict-outer-restack-outer-cache.v1"
        and result.get("status") == "OUTER_CACHE_COMPLETE"
        and result.get("pass_name") == pass_name
        and int(result.get("outer_fold", -1)) == outer_fold
        and result.get("identity_sha256") == expected_identity
        and result.get("privacy", {}).get(
            "held_outcome_rows_retained_or_supplied_to_fit_selection_or_metric"
        )
        == 0
        and result.get("privacy", {}).get("held_state_or_outcome_metric_computed")
        is False
    ):
        raise Top5ProbeError("outer result provenance drifted")
    return path.parent, result, str(record["sha256"])


def _model_score_record(
    outer: Mapping[str, Any], domain: str, model_id: str
) -> Mapping[str, Any]:
    groups = outer.get("models", {})
    rows = list(groups.get("generic", [])) + list(groups.get("focused", []))
    matches = [
        row
        for row in rows
        if row.get("domain") == domain and row.get("model_id") == model_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("score"), Mapping):
        raise Top5ProbeError("full score model record is ambiguous")
    return matches[0]["score"]


def _session_orders(
    outer: Mapping[str, Any], arrays: _FrozenArrays
) -> Tuple[np.ndarray, np.ndarray]:
    held = np.asarray(arrays.load(outer["held"]["session_ordinal"]), dtype=np.int16)
    if (
        held.shape != (400,)
        or np.any((held < 0) | (held >= SESSION_COUNT))
        or len(np.unique(held)) != len(held)
    ):
        raise Top5ProbeError("held session order is invalid")
    mask = np.ones(SESSION_COUNT, dtype=bool)
    mask[held] = False
    training = np.flatnonzero(mask).astype(np.int16)
    if training.shape != (1600,) or np.intersect1d(training, held).size:
        raise Top5ProbeError("training session complement is invalid")
    domain = outer.get("domains", {})
    held_record = domain.get("H_%d" % int(outer.get("outer_fold", -1)), {})
    train_record = domain.get("T_%d" % int(outer.get("outer_fold", -1)), {})
    if (
        _array_sha256(held) != held_record.get("session_sha256")
        or _array_sha256(training) != train_record.get("session_sha256")
    ):
        raise Top5ProbeError("recovered session order disagrees with v2.8 domains")
    return training, held


def _stable_raw_topk(
    scores: np.ndarray, incumbent: np.ndarray, k: int
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    served = np.asarray(incumbent, dtype=np.int64)
    if (
        values.ndim != 3
        or values.shape[1:] != (TURN_COUNT, CANDIDATE_COUNT)
        or served.shape != values.shape[:2]
        or k < 1
        or k > CANDIDATE_COUNT - 9
        or np.any((served < 0) | (served >= 10))
        or not np.isfinite(values).all()
    ):
        raise Top5ProbeError("top-k score input is invalid")
    allowed = values.copy()
    allowed[:, :, :10] = -np.inf
    sessions = np.arange(len(values))[:, None]
    turns = np.arange(TURN_COUNT)[None, :]
    allowed[sessions, turns, served] = values[sessions, turns, served]
    order = np.argsort(-allowed, axis=2, kind="stable")[:, :, :k]
    if np.any((order < 0) | (order >= CANDIDATE_COUNT)):
        raise Top5ProbeError("top-k choice is outside C100")
    return order.astype(np.uint8)


def _deduplicate_topk(
    family_choices: np.ndarray,
    current_choice: np.ndarray,
    incumbent: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(family_choices)
    current = np.asarray(current_choice)
    served = np.asarray(incumbent)
    if (
        raw.ndim != 4
        or raw.shape[2] != FAMILY_COUNT
        or raw.shape[3] < 1
        or current.shape != raw.shape[:2]
        or served.shape != raw.shape[:2]
        or np.any((raw < 0) | (raw >= CANDIDATE_COUNT))
    ):
        raise Top5ProbeError("top-k proposal shape is invalid")
    width = FAMILY_COUNT * raw.shape[3]
    candidates = np.full((*raw.shape[:2], width), -1, dtype=np.int16)
    source_mask = np.zeros(candidates.shape, dtype=np.uint8)
    available = np.zeros(candidates.shape, dtype=bool)
    for session in range(raw.shape[0]):
        for turn in range(raw.shape[1]):
            merged: Dict[int, int] = {}
            excluded = {int(current[session, turn]), int(served[session, turn])}
            for family in range(FAMILY_COUNT):
                for rank in range(raw.shape[3]):
                    candidate = int(raw[session, turn, family, rank])
                    if candidate in excluded:
                        continue
                    merged[candidate] = merged.get(candidate, 0) | (1 << family)
            if len(merged) > width:
                raise Top5ProbeError("top-k action width overflowed")
            for slot, candidate in enumerate(sorted(merged)):
                candidates[session, turn, slot] = candidate
                source_mask[session, turn, slot] = merged[candidate]
                available[session, turn, slot] = True
    return candidates, source_mask, available


def _allowed_rank_fraction(
    scores: np.ndarray, choice: np.ndarray, incumbent: np.ndarray
) -> np.ndarray:
    if scores.ndim != 3 or choice.shape != scores.shape[:2]:
        raise Top5ProbeError("rank-fraction input shape mismatch")
    if incumbent.shape != choice.shape or scores.shape[2] < 11:
        raise Top5ProbeError("rank-fraction incumbent shape mismatch")
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1, scores.shape[2])
    flat_choice = np.asarray(choice, dtype=np.int64).reshape(-1)
    flat_incumbent = np.asarray(incumbent, dtype=np.int64).reshape(-1)
    rows = np.arange(len(flat_choice))
    candidate_indices = np.arange(scores.shape[2], dtype=np.int64)[None, :]
    allowed = np.ones(flat_scores.shape, dtype=bool)
    allowed[:, :10] = False
    allowed[rows, flat_incumbent] = True
    if not np.all(allowed[rows, flat_choice]):
        raise Top5ProbeError("ranked choice is outside slot-10 action set")
    selected = flat_scores[rows, flat_choice]
    better = allowed & (
        (flat_scores > selected[:, None])
        | (
            (flat_scores == selected[:, None])
            & (candidate_indices < flat_choice[:, None])
        )
    )
    rank = 1 + better.sum(axis=1)
    allowed_count = allowed.sum(axis=1)
    if scores.shape[2] == CANDIDATE_COUNT and not np.all(
        allowed_count == CANDIDATE_COUNT - 9
    ):
        raise Top5ProbeError("slot-10 action set is not 91 candidates")
    return (rank / allowed_count).astype(np.float32).reshape(choice.shape)


def _rank_fraction_for_actions(
    scores: np.ndarray,
    candidates: np.ndarray,
    available: np.ndarray,
    incumbent: np.ndarray,
) -> np.ndarray:
    result = np.zeros(candidates.shape, dtype=np.float32)
    for slot in range(candidates.shape[2]):
        safe_choice = np.where(
            available[..., slot], candidates[..., slot], incumbent
        ).astype(np.uint8)
        rank = _allowed_rank_fraction(scores, safe_choice, incumbent)
        result[..., slot] = np.where(available[..., slot], rank, 0.0)
    return result


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
        PROJECTED_FEATURE_INDEX[feature_name],
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
            PROJECTED_FEATURE_INDEX[feature_name],
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
    for slot_name in CONSTRAINT_SLOTS:
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
    for slot_name in CONSTRAINT_SLOTS:
        result += _served_feature_for_sessions(
            features,
            feature_sessions,
            current_choice,
            "%s_conflict" % slot_name,
        )
    return result


def _within_turn_winner(
    candidates: np.ndarray,
    source_mask: np.ndarray,
    available: np.ndarray,
    utility: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (
        candidates.shape
        == source_mask.shape
        == available.shape
        == utility.shape
        and candidates.ndim >= 1
    ):
        raise Top5ProbeError("within-turn input shape mismatch")
    if np.asarray(available).dtype != np.bool_:
        raise Top5ProbeError("within-turn availability must be boolean")
    if np.any(
        available
        & (
            (candidates < 0)
            | (candidates >= CANDIDATE_COUNT)
            | (source_mask < 1)
            | (source_mask > 0b111)
        )
    ):
        raise Top5ProbeError("available action metadata is invalid")
    if not np.isfinite(np.asarray(utility)[available]).all():
        raise Top5ProbeError("available utility is non-finite")
    leading = candidates.shape[:-1]
    best_slot = np.full(leading, -1, dtype=np.int8)
    best_candidate = np.full(leading, -1, dtype=np.int16)
    best_utility = np.full(leading, -np.inf, dtype=np.float32)
    best_support = np.full(leading, -1, dtype=np.int8)
    for slot in range(candidates.shape[-1]):
        valid = np.asarray(available[..., slot], dtype=bool)
        candidate = np.asarray(candidates[..., slot], dtype=np.int16)
        value = np.asarray(utility[..., slot], dtype=np.float32)
        support_count = BIT_COUNT[np.asarray(source_mask[..., slot], dtype=np.uint8)]
        better = valid & (
            (best_slot < 0)
            | (value > best_utility)
            | (
                (value == best_utility)
                & (
                    (support_count > best_support)
                    | (
                        (support_count == best_support)
                        & (candidate < best_candidate)
                    )
                )
            )
        )
        best_slot = np.where(better, slot, best_slot).astype(np.int8)
        best_candidate = np.where(better, candidate, best_candidate).astype(
            np.int16
        )
        best_utility = np.where(better, value, best_utility).astype(np.float32)
        best_support = np.where(better, support_count, best_support).astype(np.int8)
    return best_slot, best_candidate, best_utility, best_slot >= 0


def _causal_latch(
    winner_candidate: np.ndarray,
    winner_utility: np.ndarray,
    winner_available: np.ndarray,
    threshold: float,
    session_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (
        winner_candidate.shape
        == winner_utility.shape
        == winner_available.shape
        and winner_candidate.ndim == 2
    ):
        raise Top5ProbeError("causal latch input shape mismatch")
    if np.asarray(winner_available).dtype != np.bool_:
        raise Top5ProbeError("winner availability must be boolean")
    if math.isnan(float(threshold)) or float(threshold) == -math.inf:
        raise Top5ProbeError("causal threshold is invalid")
    if not np.isfinite(np.asarray(winner_utility)[winner_available]).all():
        raise Top5ProbeError("winner utility is non-finite")
    if np.any(
        winner_available
        & ((winner_candidate < 0) | (winner_candidate >= CANDIDATE_COUNT))
    ):
        raise Top5ProbeError("winner candidate is out of range")
    sessions, turns = winner_candidate.shape
    selected_sessions = (
        np.ones(sessions, dtype=bool)
        if session_mask is None
        else np.asarray(session_mask, dtype=bool)
    )
    if selected_sessions.shape != (sessions,):
        raise Top5ProbeError("causal latch session mask mismatch")
    supplement = np.zeros((sessions, turns), dtype=bool)
    supplemental_choice = np.full((sessions, turns), -1, dtype=np.int16)
    used = np.zeros(sessions, dtype=bool)
    for turn in range(turns):
        activate = (
            selected_sessions
            & ~used
            & winner_available[:, turn]
            & (winner_utility[:, turn] >= threshold)
        )
        supplement[:, turn] = activate
        supplemental_choice[activate, turn] = winner_candidate[activate, turn]
        used |= activate
    if np.any(supplement.sum(axis=1) > 1):
        raise Top5ProbeError("causal latch selected multiple actions")
    return supplement, supplemental_choice


def _compose_policy(
    current_chosen: np.ndarray,
    current_activation: np.ndarray,
    candidates: np.ndarray,
    available: np.ndarray,
    supplement: np.ndarray,
    supplemental_choice: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if (
        supplement.shape != current_chosen.shape
        or current_activation.shape != current_chosen.shape
        or supplemental_choice.shape != supplement.shape
        or candidates.shape != available.shape
        or candidates.shape[:2] != supplement.shape
        or np.any(supplement.sum(axis=1) > 1)
        or np.any(supplement & (supplemental_choice < 0))
    ):
        raise Top5ProbeError("supplemental policy shape mismatch")
    member = np.any(
        available & (candidates == supplemental_choice[..., None]), axis=2
    )
    if np.any(supplement & ~member):
        raise Top5ProbeError("supplemental choice is not an available action")
    final_chosen = np.asarray(current_chosen, dtype=np.uint8).copy()
    final_chosen[supplement] = supplemental_choice[supplement].astype(np.uint8)
    final_activation = np.asarray(current_activation, dtype=bool) | supplement
    return final_chosen, final_activation


def _build_surface(
    features: np.ndarray,
    feature_sessions: np.ndarray,
    current_scores: np.ndarray,
    family_scores: Sequence[np.ndarray],
    current_chosen: np.ndarray,
    current_activation: np.ndarray,
    incumbent: np.ndarray,
    k: int,
) -> ExpandedRuntimeSurface:
    if len(family_scores) != FAMILY_COUNT:
        raise Top5ProbeError("exactly three proposal families are required")
    current_score_values = np.asarray(current_scores, dtype=np.float32)
    chosen = np.asarray(current_chosen, dtype=np.uint8)
    activation = np.asarray(current_activation, dtype=bool)
    served = np.asarray(incumbent, dtype=np.uint8)
    sessions = np.asarray(feature_sessions, dtype=np.int64)
    expected = (*chosen.shape, CANDIDATE_COUNT)
    if (
        current_score_values.shape != expected
        or activation.shape != chosen.shape
        or served.shape != chosen.shape
        or sessions.shape != (len(chosen),)
        or features.shape
        != (
            SESSION_COUNT,
            TURN_COUNT,
            CANDIDATE_COUNT,
            FEATURE_COUNT,
        )
        or any(np.asarray(value).shape != expected for value in family_scores)
    ):
        raise Top5ProbeError("runtime surface input shape mismatch")
    current_choice = np.where(activation, chosen, served).astype(np.uint8)
    raw_choices = np.stack(
        [_stable_raw_topk(value, served, k) for value in family_scores], axis=2
    ).astype(np.uint8)
    candidates, source_mask, available = _deduplicate_topk(
        raw_choices, current_choice, served
    )
    width = FAMILY_COUNT * k
    shape = (*available.shape, len(FEATURE_NAMES))
    gate = np.zeros(shape, dtype=np.float32)
    gate[..., 0] = activation[..., None]
    gate[..., 1] = (
        (np.arange(chosen.shape[1], dtype=np.float32) + 1.0)
        / float(chosen.shape[1])
    )[None, :, None]
    prior = (
        np.cumsum(activation, axis=1, dtype=np.int16) - activation.astype(np.int16)
    ) / float(chosen.shape[1])
    gate[..., 2] = prior[..., None]
    gate[..., 3] = (
        available.sum(axis=2, dtype=np.float32) / float(width)
    )[..., None]
    for family in range(FAMILY_COUNT):
        gate[..., 4 + family] = (
            (source_mask & (1 << family)) != 0
        ).astype(np.float32)
    score_values = (current_score_values, *[np.asarray(v) for v in family_scores])
    for index, values in enumerate(score_values):
        gate[..., 7 + index] = _rank_fraction_for_actions(
            values, candidates, available, served
        )
    for index, values in enumerate(family_scores):
        current_rank = _allowed_rank_fraction(
            np.asarray(values), current_choice, served
        )
        gate[..., 11 + index] = current_rank[..., None]
    gate[..., 14] = _candidate_feature_for_sessions(
        features, sessions, candidates, available, "coverage_rank_fraction"
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
            features, sessions, candidates, available, feature_name
        )
        current_value = _served_feature_for_sessions(
            features, sessions, current_choice, feature_name
        )
        gate[..., offset] = np.where(
            available, candidate_value - current_value[..., None], 0.0
        )
    candidate_conflict = _conflict_sum_for_sessions(
        features, sessions, candidates, available
    )
    current_conflict = _served_conflict_for_sessions(
        features, sessions, current_choice
    )
    gate[..., 18] = np.where(
        available, candidate_conflict - current_conflict[..., None], 0.0
    )
    gate = np.where(available[..., None], gate, 0.0).astype(np.float32)
    if (
        candidates.shape != (*chosen.shape, width)
        or raw_choices.shape != (*chosen.shape, FAMILY_COUNT, k)
        or gate.shape != shape
        or not np.isfinite(gate).all()
        or np.any(candidates[available] < 10)
        or np.any(source_mask[available] < 1)
        or np.any(source_mask[available] > 0b111)
        or np.any(candidates[available] == np.repeat(current_choice[..., None], width, axis=2)[available])
        or np.any(candidates[available] == np.repeat(served[..., None], width, axis=2)[available])
        or np.any(candidates[~available] != -1)
        or np.any(source_mask[~available] != 0)
        or np.any(gate[~available] != 0.0)
    ):
        raise Top5ProbeError("expanded runtime surface invariant failed")
    return ExpandedRuntimeSurface(
        current_chosen=chosen,
        current_activation=activation,
        current_choice=current_choice,
        incumbent=served,
        family_choices=raw_choices,
        candidates=candidates,
        source_mask=source_mask,
        available=available,
        features=gate,
    )


def _assert_k1_parity(
    rebuilt: ExpandedRuntimeSurface,
    old_files: Mapping[str, Mapping[str, Any]],
    arrays: _FrozenArrays,
) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for name in SURFACE_FIELDS:
        actual = np.asarray(getattr(rebuilt, name))
        if name == "family_choices":
            actual = actual[..., 0]
        expected = np.asarray(arrays.load(old_files[name]))
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise Top5ProbeError("K=1 runtime parity failed: %s" % name)
        hashes[name] = _array_sha256(actual)
    return hashes


def _assert_top1_subset(
    old_files: Mapping[str, Mapping[str, Any]],
    expanded: ExpandedRuntimeSurface,
    arrays: _FrozenArrays,
) -> int:
    old_candidates = np.asarray(arrays.load(old_files["candidates"]), dtype=np.int16)
    old_available = np.asarray(arrays.load(old_files["available"]), dtype=bool)
    old_source_mask = np.asarray(
        arrays.load(old_files["source_mask"]), dtype=np.uint8
    )
    if not (
        old_candidates.shape == old_available.shape == old_source_mask.shape
        and old_candidates.shape[:2] == expanded.candidates.shape[:2]
    ):
        raise Top5ProbeError("old top-1 surface shape is invalid")
    matches = old_candidates[..., None] == expanded.candidates[:, :, None, :]
    support_superset = (
        expanded.source_mask[:, :, None, :] & old_source_mask[..., None]
    ) == old_source_mask[..., None]
    present = np.any(
        matches & expanded.available[:, :, None, :] & support_superset, axis=3
    )
    if not np.all(present[old_available]):
        raise Top5ProbeError("an old top-1 action or support bit is absent from top-5")
    return int(old_available.sum())


def _assert_policy_mechanics(value: ExpandedRuntimeSurface) -> None:
    utility = np.where(value.available, 0.0, 0.0).astype(np.float32)
    _slot, candidate, winner, winner_available = _within_turn_winner(
        value.candidates, value.source_mask, value.available, utility
    )
    supplement, supplemental_choice = _causal_latch(
        candidate, winner, winner_available, 0.0
    )
    if np.any(supplement.sum(axis=1) > 1):
        raise Top5ProbeError("causal latch selected more than one action")
    zero = np.zeros_like(value.current_activation, dtype=bool)
    no_choice = np.full(value.current_chosen.shape, -1, dtype=np.int16)
    final_chosen, final_activation = _compose_policy(
        value.current_chosen,
        value.current_activation,
        value.candidates,
        value.available,
        zero,
        no_choice,
    )
    if not (
        np.array_equal(final_chosen, value.current_chosen)
        and np.array_equal(final_activation, value.current_activation)
        and supplemental_choice.shape == value.current_chosen.shape
    ):
        raise Top5ProbeError("KEEP composition changed current policy")


def _write_surface(output_dir: Path, value: ExpandedRuntimeSurface) -> Dict[str, Any]:
    files = {
        name: _write_npy_exclusive(
            output_dir / (name + ".npy"), np.asarray(getattr(value, name))
        )
        for name in SURFACE_FIELDS
    }
    widths = value.available.sum(axis=2).astype(np.int16)
    unique, counts = np.unique(widths, return_counts=True)
    return {
        "files": files,
        "sessions": int(value.current_chosen.shape[0]),
        "turns": int(value.current_chosen.size),
        "available_action_rows": int(value.available.sum()),
        "action_turns": int(np.any(value.available, axis=2).sum()),
        "width_mean": float(widths.mean()),
        "width_p50_higher": int(np.quantile(widths, 0.50, method="higher")),
        "width_p95_higher": int(np.quantile(widths, 0.95, method="higher")),
        "width_max": int(widths.max()),
        "width_histogram": {
            str(int(item)): int(count)
            for item, count in zip(unique.tolist(), counts.tolist())
        },
        "feature_order_sha256": _canonical_sha256(list(FEATURE_NAMES)),
    }


def _phase_identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "array_sha256": {
            name: record["files"][name]["array_sha256"] for name in SURFACE_FIELDS
        },
        "available_action_rows": record["available_action_rows"],
        "action_turns": record["action_turns"],
        "width_histogram": record["width_histogram"],
        "feature_order_sha256": record["feature_order_sha256"],
    }


def _load_projected_features(outer: Mapping[str, Any]) -> Tuple[Path, np.ndarray]:
    source = outer.get("sources", {}).get("projected_features", {})
    unresolved = Path(str(source.get("path", "")))
    if (
        unresolved.suffix.lower() != ".npy"
        or unresolved.is_symlink()
    ):
        raise Top5ProbeError("projected feature source is not an allowed .npy")
    path = unresolved.resolve()
    if (
        str(source.get("sha256")) != EXPECTED_HASHES["projected_features"]
        or not _path_below(path, ROOT.parent)
        or not path.is_file()
        or _sha256(path) != EXPECTED_HASHES["projected_features"]
    ):
        raise Top5ProbeError("projected feature source drifted")
    features = _load_npy_mmap(path)
    if features.shape != (
        SESSION_COUNT,
        TURN_COUNT,
        CANDIDATE_COUNT,
        FEATURE_COUNT,
    ) or features.dtype != np.float32:
        raise Top5ProbeError("projected feature schema drifted")
    return path, features


def _score_records(
    outer: Mapping[str, Any], outer_fold: int, phase: str
) -> Tuple[Mapping[str, Any], Tuple[Mapping[str, Any], ...], bool]:
    if phase == "oof_T":
        generic = outer["portfolio_score_files"]["oof_generic"]
        return (
            generic["current_ndcg_d4_lr003"],
            (
                generic["pairwise_d4_control"],
                outer["rrf"]["oof"],
                outer["portfolio_score_files"]["oof_focused"],
            ),
            False,
        )
    domain = "T_%d" % outer_fold
    return (
        _model_score_record(outer, domain, "current_ndcg_d4_lr003"),
        (
            _model_score_record(outer, domain, "pairwise_d4_control"),
            outer["rrf"]["full"],
            _model_score_record(outer, domain, "focused_ndcg_d3"),
        ),
        True,
    )


def _same_array_record(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "sha256",
        "array_sha256",
        "bytes",
        "shape",
        "dtype",
        "asin_shape_matches",
    )
    return all(left.get(name) == right.get(name) for name in fields)


def _stage0_array_records(
    outer: Mapping[str, Any], outer_fold: int
) -> Tuple[Mapping[str, Any], ...]:
    records = [outer["held"]["session_ordinal"]]
    for phase in PHASES:
        files = outer["runtime"][phase]["files"]
        if set(files) != set(SURFACE_FIELDS):
            raise Top5ProbeError("runtime surface file registry drifted")
        records.extend(files[name] for name in SURFACE_FIELDS)

    oof_current, oof_families, oof_full = _score_records(
        outer, outer_fold, "oof_T"
    )
    full_current, full_families, full = _score_records(
        outer, outer_fold, "reference_T"
    )
    if oof_full or not full:
        raise Top5ProbeError("score phase registry drifted")
    duplicate_current = outer.get("current", {}).get("oof_files", {}).get("scores")
    if not isinstance(duplicate_current, Mapping) or not _same_array_record(
        oof_current, duplicate_current
    ):
        raise Top5ProbeError("canonical current OOF score record drifted")
    records.extend((oof_current, *oof_families, full_current, *full_families))
    return tuple(records)


def _build_pass(
    output_dir: Path,
    pass_name: str,
    outer_fold: int,
    frozen_result: Mapping[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    outer_root, outer, expected_outer_result_sha256 = _validate_outer_result(
        frozen_result, pass_name, outer_fold
    )
    arrays = _FrozenArrays(
        outer_root, _stage0_array_records(outer, outer_fold)
    )
    t_sessions, h_sessions = _session_orders(outer, arrays)
    projected_path, projected = _load_projected_features(outer)
    source_hash_start = {
        "preregistration": _sha256(PREREGISTRATION),
        "implementation_amendment": _sha256(IMPLEMENTATION_AMENDMENT),
        "probe": _sha256(Path(__file__).resolve()),
        "v28_manifest": _sha256(V28_MANIFEST),
        "v28_result": _sha256(V28_RESULT),
        "outer_result": _sha256(outer_root / "outer_complete.json"),
        "projected_features": _sha256(projected_path),
    }
    for name in (
        "preregistration",
        "implementation_amendment",
        "v28_manifest",
        "v28_result",
    ):
        if source_hash_start[name] != EXPECTED_HASHES[name]:
            raise Top5ProbeError("frozen source drifted before Stage 0: %s" % name)
    if (
        source_hash_start["projected_features"]
        != EXPECTED_HASHES["projected_features"]
        or source_hash_start["outer_result"] != expected_outer_result_sha256
    ):
        raise Top5ProbeError("pass-specific source drifted before Stage 0")
    phase_records: Dict[str, Any] = {}
    for phase in PHASES:
        session_order = h_sessions if phase == "held_H" else t_sessions
        runtime_record = outer["runtime"][phase]
        if runtime_record.get("feature_order_sha256") != _canonical_sha256(
            list(FEATURE_NAMES)
        ):
            raise Top5ProbeError("v2.8 runtime feature order drifted")
        old_files = runtime_record["files"]
        current_record, family_records, full = _score_records(
            outer, outer_fold, phase
        )
        current_scores_source = arrays.load(current_record)
        family_score_sources = [arrays.load(record) for record in family_records]
        current_scores = (
            np.asarray(current_scores_source[session_order], dtype=np.float32)
            if full
            else np.asarray(current_scores_source, dtype=np.float32)
        )
        family_scores = [
            (
                np.asarray(value[session_order], dtype=np.float32)
                if full
                else np.asarray(value, dtype=np.float32)
            )
            for value in family_score_sources
        ]
        current_chosen = np.asarray(arrays.load(old_files["current_chosen"]))
        current_activation = np.asarray(arrays.load(old_files["current_activation"]))
        incumbent = np.asarray(arrays.load(old_files["incumbent"]))
        k1 = _build_surface(
            projected,
            session_order,
            current_scores,
            family_scores,
            current_chosen,
            current_activation,
            incumbent,
            1,
        )
        k1_hashes = _assert_k1_parity(k1, old_files, arrays)
        expanded = _build_surface(
            projected,
            session_order,
            current_scores,
            family_scores,
            current_chosen,
            current_activation,
            incumbent,
            TOP_K,
        )
        _assert_policy_mechanics(expanded)
        old_action_rows = _assert_top1_subset(old_files, expanded, arrays)
        record = _write_surface(output_dir / phase, expanded)
        record["k1_full_surface_parity"] = True
        record["k1_array_sha256"] = k1_hashes
        record["old_top1_action_rows"] = old_action_rows
        record["old_top1_subset"] = True
        record["causal_latch_at_most_one"] = True
        record["keep_composition_exact"] = True
        record["session_order_sha256"] = _array_sha256(session_order)
        phase_records[phase] = record
    keep_checks = []
    for phase in PHASES:
        files = phase_records[phase]["files"]
        chosen = arrays.load(outer["runtime"][phase]["files"]["current_chosen"])
        activation = arrays.load(
            outer["runtime"][phase]["files"]["current_activation"]
        )
        if (
            files["current_chosen"]["array_sha256"] != _array_sha256(chosen)
            or files["current_activation"]["array_sha256"]
            != _array_sha256(activation)
        ):
            raise Top5ProbeError("KEEP did not preserve current policy")
        keep_checks.append(phase)
    arrays.verify_unchanged()
    source_hash_end = {
        "preregistration": _sha256(PREREGISTRATION),
        "implementation_amendment": _sha256(IMPLEMENTATION_AMENDMENT),
        "probe": _sha256(Path(__file__).resolve()),
        "v28_manifest": _sha256(V28_MANIFEST),
        "v28_result": _sha256(V28_RESULT),
        "outer_result": _sha256(outer_root / "outer_complete.json"),
        "projected_features": _sha256(projected_path),
    }
    if source_hash_end != source_hash_start:
        raise Top5ProbeError("source changed during Stage 0")
    identity = {
        "outer_fold": outer_fold,
        "source_outer_identity_sha256": outer["identity_sha256"],
        "v28_freeze_identity_sha256": V28_IDENTITY,
        "training_session_order_sha256": _array_sha256(t_sessions),
        "held_session_order_sha256": _array_sha256(h_sessions),
        "phases": {
            phase: _phase_identity(phase_records[phase]) for phase in PHASES
        },
        "k1_full_surface_parity": True,
        "old_top1_subset": True,
        "keep_preserves_current_phases": keep_checks,
    }
    return {
        "pass_name": pass_name,
        "outer_fold": outer_fold,
        "status": "TARGET_FREE_SURFACE_COMPLETE",
        "identity": identity,
        "identity_sha256": _canonical_sha256(identity),
        "phases": phase_records,
        "sources": source_hash_end,
        "privacy": {
            "label_archive_opened": False,
            "outcome_member_accesses": 0,
            "held_state_or_metric_computed": False,
            "agent_or_evaluator_started": False,
        },
        "resource": {
            "wall_seconds": round(time.perf_counter() - started, 6),
        },
    }


def _directory_stats(path: Path) -> Tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def run(output_root: Path, outer_fold: int = 0) -> Dict[str, Any]:
    if outer_fold != 0:
        raise Top5ProbeError("Stage 0 is frozen to outer fold 0")
    unresolved_output = output_root
    output_root = output_root.resolve()
    experiments_root = (
        ROOT / "experiments" / "fast_track" / "small_ranker_v2_9"
    ).resolve()
    if (
        output_root.exists()
        or unresolved_output.is_symlink()
        or not _path_below(output_root, experiments_root)
    ):
        raise Top5ProbeError(
            "output root must be new and below experiments/fast_track/small_ranker_v2_9"
        )
    started = time.perf_counter()
    _prereg, _amendment, frozen_result = _validate_protocol()
    output_root.mkdir(parents=True)
    first = _build_pass(output_root / "first" / "outer_0", "first", 0, frozen_result)
    repeat = _build_pass(
        output_root / "repeat" / "outer_0", "repeat", 0, frozen_result
    )
    exact = first["identity"] == repeat["identity"]
    if not exact or first["identity_sha256"] != repeat["identity_sha256"]:
        raise Top5ProbeError("first/repeat target-free surface identity differs")
    common_source_names = set(first["sources"]) - {"outer_result"}
    if common_source_names != set(repeat["sources"]) - {"outer_result"} or any(
        first["sources"][name] != repeat["sources"][name]
        for name in common_source_names
    ):
        raise Top5ProbeError("first/repeat frozen source snapshot differs")
    final_source_hashes = {
        "preregistration": _sha256(PREREGISTRATION),
        "implementation_amendment": _sha256(IMPLEMENTATION_AMENDMENT),
        "probe": _sha256(Path(__file__).resolve()),
        "v28_manifest": _sha256(V28_MANIFEST),
        "v28_result": _sha256(V28_RESULT),
    }
    if any(
        first["sources"][name] != value
        for name, value in final_source_hashes.items()
    ):
        raise Top5ProbeError("frozen source changed before Stage 0 result freeze")
    file_count, output_bytes = _directory_stats(output_root)
    rss, peak = _process_memory()
    budget = _prereg["resource_budget"]
    first_repeat_wall = float(first["resource"]["wall_seconds"]) + float(
        repeat["resource"]["wall_seconds"]
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.9-STRICT-TOP5-PROPOSAL-DEPTH-STAGE0",
        "status": "PENDING_RESOURCE_GATE",
        "evidence_boundary": "target-free mechanics only; no label archive, selector fit, proposal oracle, or outcome metric",
        "outer_fold": outer_fold,
        "proposal_depth": TOP_K,
        "maximum_actions_per_turn": MAX_ACTIONS,
        "first": first,
        "repeat": repeat,
        "exact_repeat": {
            "equal": exact,
            "identity_sha256": first["identity_sha256"],
        },
        "sources": {
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "implementation_amendment_sha256": _sha256(IMPLEMENTATION_AMENDMENT),
            "probe_sha256": _sha256(Path(__file__).resolve()),
            "v28_manifest_sha256": _sha256(V28_MANIFEST),
            "v28_result_sha256": _sha256(V28_RESULT),
            "v28_freeze_identity_sha256": V28_IDENTITY,
        },
        "privacy": {
            "label_archive_opened": False,
            "outcome_member_accesses": 0,
            "held_state_or_metric_computed": False,
            "agent_or_full_evaluator_started": False,
            "forbidden_split_or_external_data_opened": False,
        },
        "resource": {
            "wall_seconds": round(time.perf_counter() - started, 6),
            "rss_bytes": int(rss),
            "peak_working_set_bytes": int(peak),
            "output_files_before_result": file_count,
            "output_bytes_before_result": output_bytes,
            "stage0_result_bytes": 0,
            "output_files_after_result": file_count + 1,
            "output_bytes_after_result": output_bytes,
            "first_plus_repeat_wall_seconds": round(first_repeat_wall, 6),
            "budget": {},
            "xgboost_fits": 0,
            "selector_fits": 0,
            "workers": 1,
        },
        "decision": {
            "stage1a_authorized": False,
            "selector_or_outcome_authorized": False,
            "runtime_artifact_authorized": False,
        },
    }
    previous_resource_state = None
    for _iteration in range(10):
        result_bytes = len(_serialized_json(result))
        output_bytes_after_result = output_bytes + result_bytes
        budget_checks = {
            "first_plus_repeat_wall_seconds": {
                "actual": round(first_repeat_wall, 6),
                "maximum": int(budget["first_plus_repeat_wall_seconds_maximum"]),
                "pass": first_repeat_wall
                <= int(budget["first_plus_repeat_wall_seconds_maximum"]),
            },
            "peak_working_set_bytes": {
                "actual": int(peak),
                "maximum": int(budget["peak_working_set_bytes_maximum"]),
                "pass": 0 < peak <= int(budget["peak_working_set_bytes_maximum"]),
            },
            "new_cache_bytes": {
                "actual": output_bytes_after_result,
                "maximum": int(budget["new_cache_bytes_maximum"]),
                "pass": output_bytes_after_result
                <= int(budget["new_cache_bytes_maximum"]),
            },
        }
        budget_pass = all(item["pass"] for item in budget_checks.values())
        result["resource"]["stage0_result_bytes"] = result_bytes
        result["resource"]["output_bytes_after_result"] = output_bytes_after_result
        result["resource"]["budget"] = budget_checks
        result["status"] = (
            "IMPLEMENTATION_PASS_STAGE0"
            if budget_pass
            else "IMPLEMENTATION_FAIL_STAGE0_RESOURCE_BUDGET"
        )
        result["decision"]["stage1a_authorized"] = budget_pass
        resource_state = (
            result_bytes,
            output_bytes_after_result,
            budget_pass,
            result["status"],
        )
        if resource_state == previous_resource_state:
            break
        previous_resource_state = resource_state
    else:
        raise Top5ProbeError("Stage 0 result-size accounting did not converge")
    if len(_serialized_json(result)) != result["resource"]["stage0_result_bytes"]:
        raise Top5ProbeError("Stage 0 result-size accounting drifted")
    _assert_no_identity_matches(result)
    _write_json_exclusive(output_root / "stage0_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.output_root, args.outer_fold)
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
    return 0 if result["status"] == "IMPLEMENTATION_PASS_STAGE0" else 2


if __name__ == "__main__":
    raise SystemExit(main())
