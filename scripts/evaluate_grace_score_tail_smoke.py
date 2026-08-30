"""One-shot 100-session smoke for fixed grace plus frozen score-tail priority.

All work before the durable exclusive receipt is target-free.  The sealed
numeric label archive is opened once after receipt creation, solely to
reproduce the frozen comparator and compute anonymous aggregate metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat as stat_module
import subprocess
import sys
import time
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_rank1_score_priority_replacement as score_v13  # noqa: E402
from scripts import evaluate_rank1_seen_replacement as replay_v12  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


Rank1ReplayError = replay_v12.Rank1ReplayError
Rank1ReplayConsumedError = replay_v12.Rank1ReplayConsumedError
ReplayBundle = replay_v12.ReplayBundle
Transition = replay_v12.Transition
MetricValues = replay_v12.MetricValues
score_priority_ordinals = score_v13.score_priority_ordinals
decode_intent_versions = replay_v12.decode_intent_versions
reconstruct_v19_order = replay_v12.reconstruct_v19_order
load_outcomes_from_open_handle = replay_v12.load_outcomes_from_open_handle
derive_baseline_session_hit = replay_v12.derive_baseline_session_hit
state_from_positive_index = replay_v12.state_from_positive_index
metric_values = replay_v12.metric_values
transition_metrics = replay_v12.transition_metrics


SCHEMA_VERSION = "small-ranker-grace-score-tail-smoke-outcome.v1"
EXPERIMENT_ID = "SR-V2.15-GRACE-SCORE-TAIL-SMOKE"
BRANCH = "small-ranker-v2.15-grace-score-combo"
REMOTE = "origin"
REMOTE_URL = "https://github.com/lamperriat/techjam-err402.git"
REMOTE_REF = "refs/remotes/origin/" + BRANCH
BASE_COMMIT = "11d248ca6ff32b602bfad661711da94cb5e5235c"
PREREG_COMMIT = "a4ca533a5b3710f0116181c9059fedb5e1b48139"
PREREG_BLOB = "5e4c88d671622a78124403ca8637adcfd2fedcd3"
PREREG_CANONICAL_SHA256 = "569deefd88ffd28b5a1b6b620d5c60e4c612aaef0ee3e2a599d820acd6796d3e"
PREREG_PATH = ROOT / "configs/small_ranker_v2_15.grace_score_tail_smoke_preregistration.json"
PREREG_PATHS = {
    "configs/small_ranker_v2_15.grace_score_tail_smoke_preregistration.json"
}
IMPLEMENTATION_PATHS = {
    "scripts/evaluate_grace_score_tail_smoke.py",
    "tests/test_grace_score_tail_smoke.py",
}
PINNED_BLOBS = {
    "scripts/evaluate_rank1_score_priority_replacement.py": "efe5646d5164f8fe2f952d308d489b957c7a4bb6",
    "scripts/evaluate_rank1_seen_replacement.py": "f15c54aae4a3760d95afd366f07bdefd1ef34665",
    "configs/small_ranker_global_benchmark_comparability_v1.manifest.json": "344cc82b1ce06144a0a29c6754cfdb1c4859cfdb",
    "starter/assets/small_ranker_fold_safe_v1.json": "c62922bc7478c3a5b0d6df67c6372923630d8fb9",
}

FULL_SESSION_COUNT = 2_000
SMOKE_SESSION_COUNT = 100
TURN_COUNT = 10
CANDIDATE_COUNT = 100
FEATURE_COUNT = 133
OUTER_FOLDS = 5
BASE_SEED = score_v13.BASE_SEED
RESOURCE_BYTES_MAXIMUM = 2_147_483_648
RESOURCE_SECONDS_MAXIMUM = 120.0
POLICY_P95_MICROSECONDS_MAXIMUM = 5_000.0

SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
PROJECTED_FEATURES_PATH = PROJECTION_ROOT / (
    "experiments/fast_track/small_ranker_fold_safe_projected_features.npy"
)
PROJECTED_FEATURES_SHA256 = "cd9b075b923c31afe10b9a9c6d720de22e221c93de961595ff484ea4f532b90a"
PROJECTED_FEATURES_BYTES = 1_064_000_128
PROJECTED_PREFIX_BYTES = 53_200_000
PROJECTED_PREFIX_SHA256 = "91a1f6c839e008a589e2e2abd0111e07929b5361348c5edb519056ba7d884159"
OOF_SCORES_PATH = SOURCE_ROOT / (
    "experiments/fast_track/small_ranker_v1/oof_batch_v1/"
    "oof_scores_runtime_projection_no_semantic.npy"
)
OOF_SCORES_SHA256 = "5000deb9b77b3e7b326ccab6455222b291d2ec859ddab2043fe67d23a3217c5e"
OOF_SCORES_BYTES = 8_000_128
OOF_PREFIX_BYTES = 400_000
OOF_PREFIX_SHA256 = "48bc950685e7c09db97507ae411c22aaff2537be748580c0b197f0df2222f90d"
LABEL_PATH = SOURCE_ROOT / "experiments/fast_track/small_ranker_v1/labels_v2.npz"
LABEL_SHA256 = "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb"
LABEL_BYTES = 1_702_876
TRACE_PATH = SOURCE_ROOT / (
    "experiments/fast_track/action_oracle_v1/"
    "train_explore-full-blind-shard-01-of-04.jsonl"
)
TRACE_SHA256 = "fac3bc71e6210d1a449de706d335cc5bb945d4d3daf01e8cbecbe15c0600bf1a"
TRACE_BYTES = 15_939_420
TRACE_ROWS = 5_000
PREFIX_TRACE_ROWS = 1_000
PREFIX_TRACE_CANONICAL_BYTES = 3_187_020
PREFIX_TRACE_CANONICAL_SHA256 = "96e2976eb40e8bf693e433d457d1dbc9b6115439f7a52876c374121523960554"
EXPECTED_PRIORITY_PREFIX_SHA256 = "cd33b8d6c124f41c6d493b4d99caed9f7b7c10a3048cd2efaa3fbc0b31bac155"
EXPECTED_VERSION_PREFIX_SHA256 = "f2e5477907877660a61b910376a7f8140852265619762bdc1626056aee5474c6"
EXPECTED_CHOSEN_PREFIX_SHA256 = "b5b8e0e2fcadacb96e873bf724f03e957785ee3b8f7957481d4aa0ef219c1bfe"
EXPECTED_GRACE_MASK_SHA256 = "8c92cd2e82968e40382e26c27d497b259c681b08060afc8f2ab590dba10fe1df"
EXPECTED_CHOSEN_SHA256 = score_v13.EXPECTED_CHOSEN_SHA256
EXPECTED_ACTIVATION_SHA256 = score_v13.EXPECTED_ACTIVATION_SHA256
EXPECTED_BASELINE_HITS = 1_895
EXPECTED_GRACE_TURNS = 230
EXPECTED_POST_GRACE_TURNS = 770
EXPECTED_RESETS = 115
OUTPUT_PATH = ROOT / (
    "experiments/fast_track/small_ranker_v2_15/"
    "grace_score_tail_smoke_20260831/result.json"
)

ASIN_SHAPE_RE = re.compile(
    rb"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE
)


@dataclass(frozen=True)
class SmokeInputs:
    projected_features: np.ndarray
    oof_scores: np.ndarray
    traces: tuple[tuple[dict[str, Any], ...], ...]
    versions: np.ndarray
    reset_mask: np.ndarray
    ages: np.ndarray
    grace_mask: np.ndarray
    chosen: np.ndarray
    priority: np.ndarray
    source_snapshots: Mapping[str, tuple[int, int, int]]


@dataclass(frozen=True)
class Preflight:
    environment: Mapping[str, Any]
    protocol: Mapping[str, Any]
    git: Mapping[str, Any]
    inputs: SmokeInputs
    memory_before_receipt: tuple[int, int]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "raw_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _load_json_no_duplicates(path: Path) -> dict[str, Any]:
    def hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Rank1ReplayError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    if not isinstance(value, dict):
        raise Rank1ReplayError("expected JSON object")
    return value


def _validate_protocol() -> dict[str, Any]:
    if not PREREG_PATH.is_file() or _is_link_or_reparse(PREREG_PATH):
        raise Rank1ReplayError("preregistration is unavailable")
    prereg = _load_json_no_duplicates(PREREG_PATH)
    if not (
        _canonical_sha256(prereg) == PREREG_CANONICAL_SHA256
        and prereg.get("schema_version")
        == "small-ranker-grace-score-tail-smoke-preregistration.v1"
        and prereg.get("status")
        == "PREREGISTERED_BEFORE_IMPLEMENTATION_AND_OUTCOME"
        and prereg.get("experiment_id") == EXPERIMENT_ID
        and prereg.get("parent_commit") == BASE_COMMIT
        and prereg.get("outcome_protocol", {}).get("attach_count") == 1
        and prereg.get("outcome_protocol", {}).get("fixed_output")
        == OUTPUT_PATH.relative_to(ROOT).as_posix()
        and prereg.get("decision", {}).get("served_default") == "off"
    ):
        raise Rank1ReplayError("preregistration binding drifted")
    return {
        "commit": PREREG_COMMIT,
        "git_blob_oid": PREREG_BLOB,
        "canonical_sha256": PREREG_CANONICAL_SHA256,
    }


def _git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=" + ROOT.as_posix(), *args],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode:
        raise Rank1ReplayError("Git checkpoint command failed")
    return completed.stdout.strip()


def _validate_git_checkpoint(implementation_commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise Rank1ReplayError("implementation commit is not a full object name")
    status = _git(("status", "--porcelain=v1", "--untracked-files=all"))
    if status:
        raise Rank1ReplayError("implementation worktree is not clean")
    head = _git(("rev-parse", "HEAD"))
    parent = _git(("rev-parse", "HEAD^"))
    prereg_parent = _git(("rev-parse", PREREG_COMMIT + "^"))
    branch = _git(("branch", "--show-current"))
    remote_url = _git(("remote", "get-url", REMOTE))
    remote_head = _git(("rev-parse", REMOTE_REF))
    paths = set(
        _git(("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")).splitlines()
    )
    prereg_paths = set(
        _git(
            (
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                PREREG_COMMIT,
            )
        ).splitlines()
    )
    prereg_blob = _git(("rev-parse", "HEAD:" + PREREG_PATH.relative_to(ROOT).as_posix()))
    pinned = {
        path: _git(("rev-parse", "HEAD:" + path)) for path in PINNED_BLOBS
    }
    normalized_remote = remote_url.rstrip("/").removesuffix(".git")
    expected_remote = REMOTE_URL.rstrip("/").removesuffix(".git")
    if not (
        head == implementation_commit
        and parent == PREREG_COMMIT
        and prereg_parent == BASE_COMMIT
        and branch == BRANCH
        and normalized_remote == expected_remote
        and remote_head == head
        and paths == IMPLEMENTATION_PATHS
        and prereg_paths == PREREG_PATHS
        and prereg_blob == PREREG_BLOB
        and pinned == PINNED_BLOBS
    ):
        raise Rank1ReplayError("implementation Git checkpoint drifted")
    implementation_blobs = {
        path: _git(("rev-parse", "HEAD:" + path)) for path in sorted(IMPLEMENTATION_PATHS)
    }
    return {
        "commit": head,
        "parent": parent,
        "preregistration_commit": PREREG_COMMIT,
        "branch": branch,
        "remote_equal": True,
        "clean_including_untracked": True,
        "paths_exact": True,
        "implementation_blobs": implementation_blobs,
        "pinned_blobs": pinned,
    }


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return False
    attributes = int(getattr(stat, "st_file_attributes", 0))
    marker = int(getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & marker)


def _check_output_components(path: Path, root: Path) -> None:
    root_resolved = root.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise Rank1ReplayError("one-shot output escapes the worktree") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise Rank1ReplayError("one-shot output path is unsafe")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if not current.exists() or not current.is_dir():
            raise Rank1ReplayError("one-shot output parent is not prepared")
        if _is_link_or_reparse(current):
            raise Rank1ReplayError("one-shot output has a link or reparse component")
    parent_resolved = path.parent.resolve(strict=True)
    if parent_resolved != root_resolved and root_resolved not in parent_resolved.parents:
        raise Rank1ReplayError("one-shot output escapes the worktree")
    if path.exists() or path.is_symlink():
        raise Rank1ReplayError("one-shot output is already consumed")


def _write_descriptor(descriptor: int, value: Mapping[str, Any]) -> tuple[int, str]:
    payload = _canonical_bytes(value) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short receipt write")
        view = view[written:]
    os.fsync(descriptor)
    return len(payload), hashlib.sha256(payload).hexdigest()


def _invalid_receipt(
    descriptor: int, implementation_commit: str, error: BaseException
) -> None:
    try:
        _write_descriptor(
            descriptor,
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "status": "INVALID_ONE_SHOT_CONSUMED",
                "implementation_commit": implementation_commit,
                "error_class": type(error).__name__,
                "rerun_forbidden": True,
            },
        )
    except BaseException:
        pass


def _open_receipt(path: Path, root: Path, implementation_commit: str) -> int:
    _check_output_components(path, root)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor: int | None = None
    try:
        descriptor = os.open(str(path), flags, 0o600)
        _write_descriptor(
            descriptor,
            {
                "schema_version": "small-ranker-grace-score-tail-smoke-marker.v1",
                "experiment_id": EXPERIMENT_ID,
                "implementation_commit": implementation_commit,
                "status": "CONSUMED_PENDING_RERUN_FORBIDDEN",
            },
        )
    except BaseException as error:
        if descriptor is not None:
            _invalid_receipt(descriptor, implementation_commit, error)
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise Rank1ReplayConsumedError(
                "receipt failed after path consumption; rerun is forbidden"
            ) from error
        if isinstance(error, FileExistsError):
            raise Rank1ReplayError("one-shot output is already consumed") from error
        if path.exists() or path.is_symlink():
            raise Rank1ReplayConsumedError(
                "receipt path may have been consumed; rerun is forbidden"
            ) from error
        raise
    if descriptor is None:
        raise Rank1ReplayError("receipt descriptor was not created")
    return descriptor


def canonical_trace_prefix(
    records: Iterable[Mapping[str, Any]], shard_number: int = 1
) -> bytes:
    if not isinstance(shard_number, int) or isinstance(shard_number, bool) or shard_number < 1:
        raise Rank1ReplayError("invalid trace shard number")
    output = bytearray()
    for record in records:
        if not isinstance(record, Mapping):
            raise Rank1ReplayError("trace record is not an object")
        raw = dict(record)
        ordinal = raw.get("ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise Rank1ReplayError("trace ordinal is invalid")
        raw["ordinal"] = (shard_number - 1) * 500 + ordinal
        output.extend(_canonical_bytes(raw))
        output.extend(b"\n")
    return bytes(output)


def _clean_trace_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    if set(raw) != {"ordinal", "turn", "actions", "candidate_pools"}:
        raise Rank1ReplayError("blind trace row schema drifted")
    actions = raw["actions"]
    pools = raw["candidate_pools"]
    if (
        not isinstance(actions, dict)
        or set(actions) != set(replay_v12.TRACE_ACTIONS)
        or not isinstance(pools, dict)
        or set(pools) != {"c20", "c50", "c100"}
    ):
        raise Rank1ReplayError("blind trace registry drifted")
    clean_actions = {
        name: replay_v12._trace_ranking(actions[name], 10, name)
        for name in replay_v12.TRACE_ACTIONS
    }
    c20 = replay_v12._trace_ranking(pools["c20"], 20, "c20")
    c50 = replay_v12._trace_ranking(pools["c50"], 50, "c50")
    c100 = replay_v12._trace_ranking(pools["c100"], 100, "c100")
    if not (
        len(c20) == 20
        and len(c50) == 50
        and len(c100) == 100
        and c20 == c50[:20]
        and c50 == c100[:50]
        and clean_actions["KEEP_R08"] == c20[:10]
        and set(clean_actions["KEEP_R08"]) == set(clean_actions["KEEP_P11"])
        and clean_actions["ASK"] == clean_actions["KEEP_P11"]
        and all(
            set(clean_actions[name]).issubset(set(c50))
            for name in ("CANDIDATE_RERANK", "FROZEN_SEMANTIC_RERANK")
        )
    ):
        raise Rank1ReplayError("blind trace membership invariant failed")
    return {"actions": clean_actions, "c20": c20, "c50": c50, "c100": c100}


def _path_identity(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(getattr(stat, "st_ino", 0)),
    )


def _validate_target_free_file(
    path: Path, expected_sha256: str, expected_bytes: int
) -> tuple[int, int, int]:
    if not path.is_file() or _is_link_or_reparse(path):
        raise Rank1ReplayError("frozen target-free source is unavailable")
    digest, size = score_v13._sha256_path(path)
    if digest != expected_sha256 or size != expected_bytes:
        raise Rank1ReplayError("frozen target-free source identity drifted")
    return _path_identity(path)


def _load_trace_prefix() -> tuple[tuple[tuple[dict[str, Any], ...], ...], tuple[int, int, int]]:
    snapshot = _validate_target_free_file(TRACE_PATH, TRACE_SHA256, TRACE_BYTES)
    prefix_raw: list[dict[str, Any]] = []
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    line_count = 0
    with TRACE_PATH.open("r", encoding="utf-8", newline="") as handle:
        for line_count, line in enumerate(handle, start=1):
            if not line.strip():
                raise Rank1ReplayError("blind trace has a blank row")
            if line_count > PREFIX_TRACE_ROWS:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise Rank1ReplayError("blind trace row is not an object")
            ordinal = raw.get("ordinal")
            turn = raw.get("turn")
            expected_ordinal = (line_count - 1) // TURN_COUNT + 1
            expected_turn = (line_count - 1) % TURN_COUNT + 1
            if ordinal != expected_ordinal or turn != expected_turn:
                raise Rank1ReplayError("blind trace prefix coordinate drifted")
            prefix_raw.append(raw)
            rows[(ordinal, turn)] = _clean_trace_record(raw)
    if line_count != TRACE_ROWS or len(prefix_raw) != PREFIX_TRACE_ROWS:
        raise Rank1ReplayError("blind trace row count drifted")
    canonical = canonical_trace_prefix(prefix_raw, shard_number=1)
    if (
        len(canonical) != PREFIX_TRACE_CANONICAL_BYTES
        or hashlib.sha256(canonical).hexdigest() != PREFIX_TRACE_CANONICAL_SHA256
    ):
        raise Rank1ReplayError("blind trace prefix identity drifted")
    traces = tuple(
        tuple(rows[(ordinal, turn)] for turn in range(1, TURN_COUNT + 1))
        for ordinal in range(1, SMOKE_SESSION_COUNT + 1)
    )
    return traces, snapshot


def intent_age_and_grace_mask(versions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(versions)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise Rank1ReplayError("decoded version surface is invalid")
    ages = np.ones(values.shape, dtype=np.int16)
    for turn in range(1, values.shape[1]):
        same = values[:, turn] == values[:, turn - 1]
        ages[:, turn] = np.where(same, ages[:, turn - 1] + 1, 1)
    mask = ages <= 2
    ages.setflags(write=False)
    mask.setflags(write=False)
    return ages, mask


def _load_target_free_inputs() -> SmokeInputs:
    snapshots = {
        "projected_features": _validate_target_free_file(
            PROJECTED_FEATURES_PATH, PROJECTED_FEATURES_SHA256, PROJECTED_FEATURES_BYTES
        ),
        "oof_scores": _validate_target_free_file(
            OOF_SCORES_PATH, OOF_SCORES_SHA256, OOF_SCORES_BYTES
        ),
    }
    projected = np.load(PROJECTED_FEATURES_PATH, mmap_mode="r")
    scores = np.load(OOF_SCORES_PATH, mmap_mode="r")
    if (
        projected.shape != (FULL_SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT, FEATURE_COUNT)
        or projected.dtype != np.float32
        or scores.shape != (FULL_SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT)
        or scores.dtype != np.float32
        or projected.flags.writeable
        or scores.flags.writeable
    ):
        raise Rank1ReplayError("target-free array schema drifted")
    projected_prefix = np.ascontiguousarray(projected[:SMOKE_SESSION_COUNT])
    score_prefix = np.ascontiguousarray(scores[:SMOKE_SESSION_COUNT])
    if not (
        projected_prefix.nbytes == PROJECTED_PREFIX_BYTES
        and _array_sha256(projected_prefix) == PROJECTED_PREFIX_SHA256
        and score_prefix.nbytes == OOF_PREFIX_BYTES
        and _array_sha256(score_prefix) == OOF_PREFIX_SHA256
        and np.isfinite(score_prefix).all()
    ):
        raise Rank1ReplayError("target-free prefix identity drifted")
    versions, reset_mask, _reset_audit = decode_intent_versions(projected_prefix)
    incumbent = base._incumbent_indices(projected_prefix)
    chosen, _margin, _gap = base.choose_slot10(score_prefix, incumbent)
    priority = score_priority_ordinals(score_prefix)
    ages, grace_mask = intent_age_and_grace_mask(versions)
    if not (
        _array_sha256(versions) == EXPECTED_VERSION_PREFIX_SHA256
        and _array_sha256(chosen) == EXPECTED_CHOSEN_PREFIX_SHA256
        and _array_sha256(priority) == EXPECTED_PRIORITY_PREFIX_SHA256
        and _array_sha256(grace_mask) == EXPECTED_GRACE_MASK_SHA256
        and int(grace_mask.sum()) == EXPECTED_GRACE_TURNS
        and int((~grace_mask).sum()) == EXPECTED_POST_GRACE_TURNS
        and int(reset_mask.sum()) == EXPECTED_RESETS
    ):
        raise Rank1ReplayError("target-free state identity drifted")
    traces, trace_snapshot = _load_trace_prefix()
    snapshots["trace_shard_1"] = trace_snapshot
    for value in (versions, reset_mask, chosen, priority):
        value.setflags(write=False)
    return SmokeInputs(
        projected,
        scores,
        traces,
        versions,
        reset_mask,
        ages,
        grace_mask,
        chosen,
        priority,
        snapshots,
    )


def _source_snapshots_equal(snapshots: Mapping[str, tuple[int, int, int]]) -> bool:
    paths = {
        "projected_features": PROJECTED_FEATURES_PATH,
        "oof_scores": OOF_SCORES_PATH,
        "trace_shard_1": TRACE_PATH,
    }
    return all(_path_identity(paths[name]) == identity for name, identity in snapshots.items())


def preflight_only(implementation_commit: str) -> Preflight:
    environment = score_v13._validate_environment()
    protocol = _validate_protocol()
    git = _validate_git_checkpoint(implementation_commit)
    _check_output_components(OUTPUT_PATH, ROOT)
    inputs = _load_target_free_inputs()
    if not _source_snapshots_equal(inputs.source_snapshots):
        raise Rank1ReplayError("target-free source changed during preflight")
    working, peak = score_v13._process_memory()
    if not (0 < working <= RESOURCE_BYTES_MAXIMUM and 0 < peak <= RESOURCE_BYTES_MAXIMUM):
        raise Rank1ReplayError("pre-receipt memory gate failed")
    return Preflight(environment, protocol, git, inputs, (working, peak))


def compose_grace_score_tail_page(
    order: Sequence[str],
    raw_c100: Sequence[str],
    priority: Sequence[int],
    served: set[str],
    age: int,
) -> tuple[str, ...]:
    ranked = tuple(str(value) for value in order)
    raw = tuple(str(value) for value in raw_c100)
    priority_values = tuple(int(value) for value in priority)
    if (
        len(ranked) != CANDIDATE_COUNT
        or len(raw) != CANDIDATE_COUNT
        or len(set(ranked)) != CANDIDATE_COUNT
        or len(set(raw)) != CANDIDATE_COUNT
        or set(ranked) != set(raw)
        or len(priority_values) != CANDIDATE_COUNT
        or set(priority_values) != set(range(CANDIDATE_COUNT))
        or not isinstance(age, int)
        or isinstance(age, bool)
        or age < 1
    ):
        raise Rank1ReplayError("grace score-tail row schema failed")
    if age <= 2:
        return ranked[:10]
    head_unseen = [value for value in ranked[:10] if value not in served]
    legal_tail = set(ranked[10:])
    tail_unseen = [
        raw[index]
        for index in priority_values
        if raw[index] in legal_tail and raw[index] not in served
    ]
    seen_fallback = [value for value in ranked if value in served]
    page = tuple((head_unseen + tail_unseen + seen_fallback)[:10])
    if len(page) != 10 or len(set(page)) != 10 or not set(page).issubset(set(raw)):
        raise Rank1ReplayError("grace score-tail page is invalid")
    return page


def replay_grace_score_tail_pages(
    traces: Sequence[Sequence[Mapping[str, Any]]],
    scores: np.ndarray,
    chosen: np.ndarray,
    activation: np.ndarray,
    versions: np.ndarray,
    measure_timing: bool = False,
) -> ReplayBundle:
    values = np.asarray(scores)
    chosen_values = np.asarray(chosen)
    activation_values = np.asarray(activation, dtype=bool)
    version_values = np.asarray(versions)
    session_count, turn_count = version_values.shape
    if (
        values.shape != (session_count, turn_count, CANDIDATE_COUNT)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
        or chosen_values.shape != (session_count, turn_count)
        or activation_values.shape != (session_count, turn_count)
        or len(traces) != session_count
        or any(len(turns) != turn_count for turns in traces)
    ):
        raise Rank1ReplayError("grace score-tail replay surface failed")
    priorities = score_priority_ordinals(values)
    baseline_pages = np.empty((session_count, turn_count, 10), dtype=np.int16)
    candidate_pages = np.empty_like(baseline_pages)
    changed = np.zeros((session_count, turn_count), dtype=bool)
    last_reset_turn = np.ones(session_count, dtype=np.int16)
    baseline_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    latency_ns: list[int] = []
    reset_count = changed_turns = changed_sessions = grace_turns = 0
    for session, turns in enumerate(traces):
        grace_served: set[str] = set()
        candidate_served: set[str] = set()
        last_version: int | None = None
        age = 0
        session_changed = False
        for turn_index, turn in enumerate(turns):
            version = int(version_values[session, turn_index])
            is_reset = last_version is None or version != last_version
            if is_reset:
                grace_served.clear()
                candidate_served.clear()
                age = 1
                reset_count += 1
                last_reset_turn[session] = turn_index + 1
            else:
                age += 1
            order = reconstruct_v19_order(
                turn,
                int(chosen_values[session, turn_index]),
                bool(activation_values[session, turn_index]),
            )
            raw = tuple(str(value) for value in turn["c100"])
            if len(raw) != CANDIDATE_COUNT or set(raw) != set(order):
                raise Rank1ReplayError("grace score-tail raw/order identity failed")
            grace_page = (
                order[:10]
                if age <= 2
                else score_v13._stable_unseen_first(order, grace_served)
            )
            tick = time.perf_counter_ns() if measure_timing else 0
            candidate_page = compose_grace_score_tail_page(
                order,
                raw,
                priorities[session, turn_index],
                candidate_served,
                age,
            )
            if measure_timing:
                latency_ns.append(time.perf_counter_ns() - tick)
            if age <= 2:
                grace_turns += 1
                if candidate_page != grace_page or candidate_page != order[:10]:
                    raise Rank1ReplayError("first-two-page identity failed")
            raw_index = {value: index for index, value in enumerate(raw)}
            try:
                baseline_pages[session, turn_index] = [raw_index[value] for value in grace_page]
                candidate_pages[session, turn_index] = [raw_index[value] for value in candidate_page]
            except KeyError as error:
                raise Rank1ReplayError("grace score-tail page escaped raw C100") from error
            page_changed = candidate_page != grace_page
            changed[session, turn_index] = page_changed
            changed_turns += int(page_changed)
            session_changed |= page_changed
            score_v13._digest_page(baseline_digest, grace_page)
            score_v13._digest_page(candidate_digest, candidate_page)
            grace_served.update(grace_page)
            candidate_served.update(candidate_page)
            last_version = version
        changed_sessions += int(session_changed)
    structural = {
        "reset_count": reset_count,
        "grace_turns": grace_turns,
        "post_grace_turns": session_count * turn_count - grace_turns,
        "changed_turns": changed_turns,
        "changed_sessions": changed_sessions,
        "first_two_pages_byte_identical": True,
        "grace_mask_pages_array_equal": True,
        "grace_mask_pages_raw_bytes_equal": True,
        "unseen_head_order_preserved": True,
        "unseen_legal_tail_score_priority": True,
        "seen_final_fallback": True,
        "actual_output_drives_separate_served_state": True,
        "reset_pages_identity": True,
    }
    return score_v13._finish_replay_bundle(
        baseline_pages,
        candidate_pages,
        changed,
        last_reset_turn,
        baseline_digest.hexdigest(),
        candidate_digest.hexdigest(),
        structural,
        latency_ns,
    )


def _bundle_exact_repeat(first: ReplayBundle, repeat: ReplayBundle) -> bool:
    return score_v13._bundle_exact_repeat(first, repeat)


def passes_smoke_gates(
    transition: Transition,
    exact_repeat: bool,
    identity_and_resource_gates: bool,
) -> bool:
    return bool(
        exact_repeat
        and identity_and_resource_gates
        and transition.miss_to_hit >= 1
        and transition.hit_to_miss == 0
        and transition.net_hits >= 1
        and transition.policy.hit_rate_at_10 > transition.baseline.hit_rate_at_10
        and transition.policy.official()["hit_rate_at_10"]
        > transition.baseline.official()["hit_rate_at_10"]
        and replay_v12._dual_nonnegative(
            transition.exact_delta("mrr"), transition.official_delta("mrr")
        )
        and replay_v12._dual_nonpositive(
            transition.exact_delta("mttc"), transition.official_delta("mttc")
        )
        and replay_v12._dual_strict_positive(
            transition.exact_delta("technical_score"),
            transition.official_delta("technical_score"),
        )
    )


def _candidate_recall(
    positive_index: np.ndarray, eligible_from: np.ndarray
) -> dict[str, dict[str, float | int]]:
    positive = np.asarray(positive_index)
    eligible = np.asarray(eligible_from)
    if positive.shape != (SMOKE_SESSION_COUNT, TURN_COUNT) or eligible.shape != (
        SMOKE_SESSION_COUNT,
    ):
        raise Rank1ReplayError("candidate recall prefix schema failed")
    visible = np.arange(1, TURN_COUNT + 1)[None, :] >= eligible[:, None]
    result: dict[str, dict[str, float | int]] = {}
    for depth in (10, 20, 50, 100):
        hit = np.any(visible & (positive >= 0) & (positive < depth), axis=1)
        count = int(hit.sum())
        result["c" + str(depth)] = {
            "count": count,
            "fraction": round(count / SMOKE_SESSION_COUNT, 6),
        }
    return result


def _result_privacy_scan(result: object) -> None:
    forbidden_keys = {
        "session_id",
        "sample_id",
        "product_id",
        "target",
        "target_id",
        "ground_truth",
        "positive_index",
        "eligible_from",
        "per_session",
        "membership_vector",
    }
    forbidden_values = {"positive_index", "eligible_from", "ground_truth", "target"}

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            keys = {str(key) for key in value}
            if keys & forbidden_keys:
                raise Rank1ReplayError("result contains a forbidden outcome key")
            for child in value.values():
                walk(child)
        elif isinstance(value, np.ndarray):
            raise Rank1ReplayError("result contains an ndarray")
        elif isinstance(value, (list, tuple)):
            if len(value) >= SMOKE_SESSION_COUNT:
                raise Rank1ReplayError("result contains a per-session vector")
            for child in value:
                walk(child)
        elif isinstance(value, str) and value in forbidden_values:
            raise Rank1ReplayError("result contains a forbidden outcome schema value")

    walk(result)
    if ASIN_SHAPE_RE.search(_canonical_bytes(result)):
        raise Rank1ReplayError("result contains a product identifier")


def _label_stat_identity(handle: BinaryIO) -> tuple[int, int, int]:
    stat = os.fstat(handle.fileno())
    return int(stat.st_size), int(stat.st_mtime_ns), int(getattr(stat, "st_ino", 0))


def _safe_close_descriptor(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except BaseException:
        pass


def _safe_close_handle(handle: BinaryIO | None) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except BaseException:
        pass


def _fold_reports(
    baseline_state: Mapping[str, np.ndarray],
    candidate_state: Mapping[str, np.ndarray],
    changed: np.ndarray,
    outer_fold: np.ndarray,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for fold in range(OUTER_FOLDS):
        mask = np.asarray(outer_fold) == fold
        if not np.any(mask):
            raise Rank1ReplayError("smoke prefix is missing an outer fold")
        reports.append(
            {"fold": fold, **transition_metrics(baseline_state, candidate_state, changed, mask).report()}
        )
    return reports


def run(implementation_commit: str) -> dict[str, Any]:
    started = time.perf_counter()
    descriptor: int | None = None
    label_handle: BinaryIO | None = None
    consumed = False
    final_written = False
    try:
        preflight = preflight_only(implementation_commit)
        inputs = preflight.inputs
        descriptor = _open_receipt(OUTPUT_PATH, ROOT, implementation_commit)
        consumed = True

        # This is deliberately the first label-path operation in the formal run.
        if not LABEL_PATH.is_file() or _is_link_or_reparse(LABEL_PATH):
            raise Rank1ReplayError("sealed label archive is unavailable")
        label_handle = LABEL_PATH.open("rb")
        label_identity_start = _label_stat_identity(label_handle)
        label_start_sha, label_start_bytes = score_v13._sha256_handle(label_handle)
        if label_start_sha != LABEL_SHA256 or label_start_bytes != LABEL_BYTES:
            raise Rank1ReplayError("sealed label archive identity drifted")
        label_handle.seek(0)
        outcomes = load_outcomes_from_open_handle(label_handle)
        baseline_session_hit = derive_baseline_session_hit(
            outcomes.baseline_rank, outcomes.eligible_from
        )
        if int(baseline_session_hit.sum()) != EXPECTED_BASELINE_HITS:
            raise Rank1ReplayError("frozen baseline hit identity drifted")
        labels = {
            "baseline_rank": outcomes.baseline_rank,
            "positive_index": outcomes.positive_index,
            "eligible_from": outcomes.eligible_from,
            "outer_fold": outcomes.outer_fold,
            "inner_fold": outcomes.inner_fold,
            "baseline_session_hit": baseline_session_hit,
        }
        surface = frozen._action_surface(inputs.projected_features, inputs.oof_scores, labels)
        if (
            _array_sha256(surface.chosen) != EXPECTED_CHOSEN_SHA256
            or not np.array_equal(surface.chosen[:SMOKE_SESSION_COUNT], inputs.chosen)
        ):
            raise Rank1ReplayError("post-receipt chosen surface drifted")
        partition = score_v13.audit_nested_partition(
            outcomes.outer_fold, outcomes.inner_fold, surface.action
        )
        activation, selections, comparator_reproduction = score_v13._reproduce_nested_activation(
            surface, labels, seed=BASE_SEED
        )
        if _array_sha256(activation) != EXPECTED_ACTIVATION_SHA256:
            raise Rank1ReplayError("frozen activation identity drifted")

        prefix = slice(0, SMOKE_SESSION_COUNT)
        prefix_scores = np.asarray(inputs.oof_scores[prefix])
        prefix_activation = np.asarray(activation[prefix])
        first = replay_grace_score_tail_pages(
            inputs.traces,
            prefix_scores,
            surface.chosen[prefix],
            prefix_activation,
            inputs.versions,
            measure_timing=True,
        )
        repeat = replay_grace_score_tail_pages(
            inputs.traces,
            prefix_scores,
            surface.chosen[prefix],
            prefix_activation,
            inputs.versions,
        )
        grace = score_v13.replay_grace_pages(
            inputs.traces, surface.chosen[prefix], prefix_activation, inputs.versions
        )
        grace_repeat = score_v13.replay_grace_pages(
            inputs.traces, surface.chosen[prefix], prefix_activation, inputs.versions
        )
        repeat_equal = _bundle_exact_repeat(first, repeat) and _bundle_exact_repeat(
            grace, grace_repeat
        )
        grace_mask = np.asarray(inputs.grace_mask)
        grace_identity = bool(
            np.array_equal(first.baseline_pages, grace.candidate_pages)
            and first.identity["baseline_ascii_page_sha256"]
            == grace.identity["candidate_ascii_page_sha256"]
            and np.array_equal(
                np.ascontiguousarray(first.candidate_pages[grace_mask]),
                np.ascontiguousarray(grace.candidate_pages[grace_mask]),
            )
            and np.ascontiguousarray(first.candidate_pages[grace_mask]).tobytes()
            == np.ascontiguousarray(grace.candidate_pages[grace_mask]).tobytes()
            and first.structural.get("first_two_pages_byte_identical") is True
            and first.structural.get("unseen_head_order_preserved") is True
            and first.structural.get("unseen_legal_tail_score_priority") is True
            and first.structural.get("seen_final_fallback") is True
            and first.structural.get("actual_output_drives_separate_served_state") is True
            and int(first.structural.get("grace_turns", -1)) == EXPECTED_GRACE_TURNS
            and int(first.structural.get("reset_count", -1)) == EXPECTED_RESETS
        )
        reset_mismatches = int(
            np.sum(first.last_reset_turn != outcomes.eligible_from[prefix])
        )
        trace_rank_mismatches = replay_v12._trace_baseline_rank_mismatches(
            inputs.traces, outcomes
        )
        if reset_mismatches or trace_rank_mismatches:
            raise Rank1ReplayError("smoke prefix label alignment drifted")

        positive = outcomes.positive_index[prefix]
        eligible = outcomes.eligible_from[prefix]
        baseline_state = state_from_positive_index(first.baseline_pages, positive, eligible)
        candidate_state = state_from_positive_index(first.candidate_pages, positive, eligible)
        repeat_state = state_from_positive_index(repeat.candidate_pages, positive, eligible)
        state_repeat = all(
            np.array_equal(candidate_state[name], repeat_state[name])
            for name in ("hit", "first_rank", "first_turn")
        )
        all_mask = np.ones(SMOKE_SESSION_COUNT, dtype=bool)
        transition = transition_metrics(
            baseline_state, candidate_state, first.changed, all_mask
        )
        fold_reports = _fold_reports(
            baseline_state,
            candidate_state,
            first.changed,
            outcomes.outer_fold[prefix],
        )
        candidate_recall = _candidate_recall(positive, eligible)

        label_end_sha, label_end_bytes = score_v13._sha256_handle(label_handle)
        label_identity_end = _label_stat_identity(label_handle)
        source_hashes_end = {
            "projected_features": score_v13._sha256_path(PROJECTED_FEATURES_PATH)[0],
            "oof_scores": score_v13._sha256_path(OOF_SCORES_PATH)[0],
            "trace_shard_1": score_v13._sha256_path(TRACE_PATH)[0],
        }
        final_git = _validate_git_checkpoint(implementation_commit)
        working, peak = score_v13._process_memory()
        wall_seconds = time.perf_counter() - started
        p95 = first.timing.get("p95_microseconds")
        resource_ok = bool(
            0 < working <= RESOURCE_BYTES_MAXIMUM
            and 0 < peak <= RESOURCE_BYTES_MAXIMUM
            and wall_seconds <= RESOURCE_SECONDS_MAXIMUM
            and isinstance(p95, (int, float))
            and 0 <= float(p95) <= POLICY_P95_MICROSECONDS_MAXIMUM
        )
        source_ok = bool(
            label_start_sha == label_end_sha == LABEL_SHA256
            and label_start_bytes == label_end_bytes == LABEL_BYTES
            and label_identity_start == label_identity_end
            and source_hashes_end
            == {
                "projected_features": PROJECTED_FEATURES_SHA256,
                "oof_scores": OOF_SCORES_SHA256,
                "trace_shard_1": TRACE_SHA256,
            }
            and _source_snapshots_equal(inputs.source_snapshots)
            and final_git == preflight.git
        )
        exact_repeat = bool(repeat_equal and state_repeat)
        identity_and_resource = bool(grace_identity and source_ok and resource_ok)
        gate_pass = passes_smoke_gates(
            transition, exact_repeat, identity_and_resource
        )
        status = (
            "SMOKE_GO_TO_SEPARATE_2K_PREREGISTRATION"
            if gate_pass
            else "SMOKE_NO_GO_CLOSE_COMBINATION"
        )
        label_order_sha = hashlib.sha256(
            b"baseline_rank\0positive_index\0eligible_from\0outer_fold\0inner_fold"
        ).hexdigest()
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "evidence_scope": "fixed shared-cohort train_explore rows 0:100 cached smoke; not private, independent, or 2000-session promotion evidence",
            "environment": dict(preflight.environment),
            "git": dict(preflight.git),
            "protocol": dict(preflight.protocol),
            "sources": {
                "projected_features_sha256": PROJECTED_FEATURES_SHA256,
                "oof_scores_sha256": OOF_SCORES_SHA256,
                "trace_shard_1_sha256": TRACE_SHA256,
                "trace_prefix_canonical_sha256": PREFIX_TRACE_CANONICAL_SHA256,
                "label_archive_sha256": LABEL_SHA256,
                "chosen_full_sha256": EXPECTED_CHOSEN_SHA256,
                "activation_full_sha256": EXPECTED_ACTIVATION_SHA256,
                "priority_prefix_sha256": EXPECTED_PRIORITY_PREFIX_SHA256,
            },
            "cohort": {
                "sessions": SMOKE_SESSION_COUNT,
                "turns": SMOKE_SESSION_COUNT * TURN_COUNT,
                "canonical_rows": "0:100",
                "trace_prefix_canonical_bytes": PREFIX_TRACE_CANONICAL_BYTES,
            },
            "policy": {
                "name": "FIXED_TWO_PAGE_GRACE_PLUS_FROZEN_SCORE_PRIORITY_UNSEEN_TAIL",
                "baseline_identity": dict(first.identity["baseline_pages"]),
                "candidate_identity": dict(first.identity["candidate_pages"]),
                "changed_identity": dict(first.identity["changed"]),
                "structural": dict(first.structural),
                "grace_mask_identity": _array_identity(inputs.grace_mask),
            },
            "identity_gates": {
                "grace_prefix_exact": grace_identity,
                "exact_repeat": exact_repeat,
                "reset_eligibility_mismatches": reset_mismatches,
                "trace_baseline_rank_mismatches": trace_rank_mismatches,
                "source_same_handle_and_rehash": source_ok,
                "resource": resource_ok,
            },
            "metrics": {
                "grace": transition.baseline.report(),
                "candidate": transition.policy.report(),
                "comparison": transition.report(),
                "outer_fold_diagnostics": fold_reports,
            },
            "candidate_recall": candidate_recall,
            "activation": {
                "changed_sessions": transition.activation_sessions,
                "changed_turns": transition.activation_turns,
            },
            "partition": {
                "outer_counts_full": partition["outer_counts"],
                "inner_counts_full": partition["inner_counts"],
                "frozen_comparator": comparator_reproduction,
                "selection_count": len(selections),
            },
            "access_audit": {
                "receipt_durable_before_label": True,
                "label_archive_open_count": 1,
                "label_member_access_count": 5,
                "label_member_order_sha256": label_order_sha,
                "proxy_open_count": 0,
                "agent_or_full_evaluator_started": False,
                "target_runtime_features": 0,
            },
            "resources": {
                "wall_seconds": round(wall_seconds, 6),
                "working_set_bytes": working,
                "peak_working_set_bytes": peak,
                "pre_receipt_working_set_bytes": preflight.memory_before_receipt[0],
                "pre_receipt_peak_working_set_bytes": preflight.memory_before_receipt[1],
                "policy_latency": dict(first.timing),
                "workers": 1,
                "gpu_used": False,
                "gpu_peak_bytes": 0,
            },
            "decision": {
                "gate_pass": gate_pass,
                "next_stage": (
                    "separately preregister one cached 2000-session nested-OOF exact-repeat"
                    if gate_pass
                    else "do not open 2000-session outcome; continue to clause-isolated C200 then C400 candidate-recall probes"
                ),
                "runtime_changed": False,
                "served_default": "off",
                "fallback_order": [
                    "SR-V2.12-FIXED-TWO-PAGE-GRACE",
                    "v1.9",
                    "P11",
                    "R08",
                ],
            },
            "receipt": {
                "path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
                "durable": True,
                "self_hash_omitted": True,
                "rerun_forbidden": True,
            },
        }
        _result_privacy_scan(result)
        _write_descriptor(descriptor, result)
        final_written = True
        _safe_close_descriptor(descriptor)
        descriptor = None
        _safe_close_handle(label_handle)
        label_handle = None
        return result
    except BaseException as error:
        _safe_close_handle(label_handle)
        label_handle = None
        if consumed and descriptor is not None and not final_written:
            _invalid_receipt(descriptor, implementation_commit, error)
            _safe_close_descriptor(descriptor)
            descriptor = None
            raise Rank1ReplayConsumedError(
                "v2.15 smoke was consumed; inspect the durable invalid receipt"
            ) from error
        raise
    finally:
        _safe_close_handle(label_handle)
        _safe_close_descriptor(descriptor)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight_only:
        preflight = preflight_only(args.implementation_commit)
        print(
            json.dumps(
                {
                    "status": "TARGET_FREE_PREFLIGHT_PASS",
                    "environment": preflight.environment,
                    "git": preflight.git,
                    "memory": list(preflight.memory_before_receipt),
                    "cohort_sessions": SMOKE_SESSION_COUNT,
                },
                sort_keys=True,
            )
        )
        return 0
    result = run(args.implementation_commit)
    print(
        json.dumps(
            {
                "status": result["status"],
                "comparison": result["metrics"]["comparison"],
                "activation": result["activation"],
                "candidate_recall": result["candidate_recall"],
                "decision": result["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
