"""One-shot cached evaluation for the preregistered v2.12 rank-1 transform.

The policy is target-blind.  Before the irreversible receipt, this module may
read only frozen feature/score arrays and blind traces.  It never loads the
proxy split and never calls the legacy evaluator-derived eligible-turn helper.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, BinaryIO, Mapping, Sequence


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

from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-rank1-seen-replacement-outcome.v1"
EXPERIMENT_ID = "SR-V2.12-RANK1-SEEN-REPLACEMENT"
BRANCH = "small-ranker-v2.12-rank1-seen-replacement"
REMOTE = "origin"
REMOTE_URL = "https://github.com/lamperriat/techjam-err402.git"
REMOTE_REF = "refs/remotes/origin/" + BRANCH
BASE_COMMIT = "39f1618f59f89052c347438164a1d17f11355558"
ORIGINAL_PREREG_COMMIT = "ed7aec80aa55d601c1ad2395e59ea4681da43fd7"
PREREG_COMMIT = "f557a0c7def1bb2a14310699764b75a71bfb5348"
PREREG_PATH = ROOT / (
    "configs/small_ranker_v2_12.rank1_seen_replacement_preregistration.json"
)
PREREG_RAW_SHA256 = (
    "871ff1a81f675788030ec8633a74c58328803bc1100aeab5f74731b3f5d5534c"
)
PREREG_CANONICAL_SHA256 = (
    "3d05aec6485a5d566b18bc1bcedf6d15fadc2f26cd2b7f62701e9809eef10c08"
)
PREREG_BLOB_OID = "6fdbd714c6e99697895896c2f56a2a989f725e34"
ORIGINAL_PREREG_BLOB_OID = "f65eedefea08327c1a7221ee4877dce21f1ccb97"
SELECTOR_SOURCE_PATH = ROOT / "scripts/export_small_ranker_fold_safe_artifact.py"
SELECTOR_SOURCE_SHA256 = (
    "5115026c53b21d4d5930cb9af7783c0988b049a0e259f5a0a588901ad44f5e8b"
)
SELECTOR_SOURCE_BLOB_OID = "5a714ca8c6d2cf6403be89ae6e107a4d0e0b2512"

SESSION_COUNT = 2_000
TURN_COUNT = 10
CANDIDATE_COUNT = 100
FEATURE_COUNT = 133
OUTER_FOLDS = 5
VERSION_FEATURE_INDEX = 126
GOAL_AGE_FEATURE_INDEX = 125
CURRENT_OVERRIDE_FEATURE_INDEX = 118
OVERRIDE_COUNT_FEATURE_INDEX = 127
BASE_SEED = 40_220_260_830
EXPECTED_INNER_FITS = 50
EXPECTED_OUTER_FITS = 10
EXPECTED_FIT_INVOCATIONS = 60
EXPECTED_SELECTION_INVOCATIONS = 5
EXPECTED_FOLD_QUANTILES = (0.390625, 0.6875, 0.40625, 0.859375, 0.5)

SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
PROJECTED_FEATURES_PATH = PROJECTION_ROOT / (
    "experiments/fast_track/small_ranker_fold_safe_projected_features.npy"
)
PROJECTED_FEATURES_SHA256 = (
    "cd9b075b923c31afe10b9a9c6d720de22e221c93de961595ff484ea4f532b90a"
)
PROJECTED_FEATURES_BYTES = 1_064_000_128
OOF_SCORES_PATH = SOURCE_ROOT / (
    "experiments/fast_track/small_ranker_v1/oof_batch_v1/"
    "oof_scores_runtime_projection_no_semantic.npy"
)
OOF_SCORES_SHA256 = (
    "5000deb9b77b3e7b326ccab6455222b291d2ec859ddab2043fe67d23a3217c5e"
)
OOF_SCORES_BYTES = 8_000_128
LABEL_PATH = SOURCE_ROOT / "experiments/fast_track/small_ranker_v1/labels_v2.npz"
LABEL_SHA256 = "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb"
LABEL_BYTES = 1_702_876
OUTPUT_PATH = ROOT / (
    "experiments/fast_track/small_ranker_v2_12/"
    "rank1_seen_replacement_one_shot_20260831/"
    "rank1_seen_replacement_result.json"
)

EXPECTED_CHOSEN_SHA256 = (
    "229952c9ced7f6eec1ff1938480adc85ba5093ad865336465749029576e47051"
)
EXPECTED_ACTIVATION_SHA256 = (
    "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
)
FEATURE_ORDER_SHA256 = (
    "92795134cc11cbe496cb63b3921b585d9f71028f75b39d9c6716a13c7e6608f8"
)
EXPECTED_BASELINE_OFFICIAL = {
    "hit_rate_at_10": 0.9715,
    "mrr": 0.676861,
    "mttc": 3.056,
    "technical_score": 0.847688,
}
LABEL_MEMBER_SPECS = (
    ("baseline_rank", (SESSION_COUNT, TURN_COUNT), "uint8"),
    ("positive_index", (SESSION_COUNT, TURN_COUNT), "int16"),
    ("eligible_from", (SESSION_COUNT,), "uint8"),
    ("outer_fold", (SESSION_COUNT,), "uint8"),
    ("inner_fold", (SESSION_COUNT,), "uint8"),
)
TRACE_AGGREGATE_PATH = SOURCE_ROOT / (
    "experiments/fast_track/action_oracle_v1/train_explore-full-aggregate.json"
)
TRACE_AGGREGATE_SHA256 = (
    "11ad3e24aec412f6cb3b146d248aa7e2335a12dafccc20241eeb3301af97ca24"
)
COMBINED_TRACE_SHA256 = (
    "f9a441220926aebf49f4b4d54a0f50df99f72ad4f8c0342e5528517503473e7b"
)
TRACE_SPECS = (
    (
        "train_explore-full-blind-shard-01-of-04.jsonl",
        "fac3bc71e6210d1a449de706d335cc5bb945d4d3daf01e8cbecbe15c0600bf1a",
    ),
    (
        "train_explore-full-blind-shard-02-of-04.jsonl",
        "63812776b374fc0041871600a5781fbf1ea6046a3219334e7263338abbab6657",
    ),
    (
        "train_explore-full-blind-shard-03-of-04.jsonl",
        "36a8706a2f8c51635e4feb4cde905a9789c7953ffeab25ae036ef824061f36b3",
    ),
    (
        "train_explore-full-blind-shard-04-of-04.jsonl",
        "1f9968795ab5490968badcf82c39ec11bedd00f22797569dfec8c2ff3fb7ed99",
    ),
)
TRACE_ACTIONS = (
    "KEEP_R08",
    "KEEP_P11",
    "CANDIDATE_RERANK",
    "FROZEN_SEMANTIC_RERANK",
    "RESULT_AWARE_REWRITE_RETRIEVE",
    "ASK",
)
IMPLEMENTATION_PATHS = {
    "scripts/evaluate_rank1_seen_replacement.py",
    "tests/test_rank1_seen_replacement.py",
}
PREREG_PATH_SET = {
    "configs/small_ranker_v2_12.rank1_seen_replacement_preregistration.json"
}
ASIN_SHAPE_RE = re.compile(
    rb"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE
)


class Rank1ReplayError(RuntimeError):
    """A mechanical failure before the one-shot receipt exists."""


class Rank1ReplayConsumedError(RuntimeError):
    """A post-receipt failure; the fixed one-shot path is permanently consumed."""


@dataclass(frozen=True)
class OutcomeBundle:
    baseline_rank: np.ndarray
    positive_index: np.ndarray
    eligible_from: np.ndarray
    outer_fold: np.ndarray
    inner_fold: np.ndarray


@dataclass(frozen=True)
class TargetFreeInputs:
    projected_features: np.ndarray
    oof_scores: np.ndarray
    traces: tuple[tuple[dict[str, Any], ...], ...]
    versions: np.ndarray
    reset_mask: np.ndarray
    chosen: np.ndarray
    incumbent: np.ndarray
    reset_audit: Mapping[str, Any]
    source_snapshots: Mapping[str, tuple[int, int]]


@dataclass(frozen=True)
class ReplayBundle:
    baseline_pages: np.ndarray
    candidate_pages: np.ndarray
    changed: np.ndarray
    last_reset_turn: np.ndarray
    identity: Mapping[str, Any]
    structural: Mapping[str, Any]
    timing: Mapping[str, Any]


@dataclass(frozen=True)
class MetricValues:
    sample_count: int
    hit_rate_at_10: Fraction
    mrr: Fraction
    mttc: Fraction
    efficiency: Fraction
    technical_score: Fraction

    def official(self) -> dict[str, float | int]:
        hit_rate = round(float(self.hit_rate_at_10), 6)
        mrr = round(float(self.mrr), 6)
        mttc = round(float(self.mttc), 6)
        efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
        technical_score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
        return {
            "sample_count": self.sample_count,
            "hit_rate_at_10": hit_rate,
            "mrr": mrr,
            "mttc": mttc,
            "efficiency": round(efficiency, 6),
            "technical_score": round(technical_score, 6),
        }

    def report(self) -> dict[str, Any]:
        values = {
            "hit_rate_at_10": self.hit_rate_at_10,
            "mrr": self.mrr,
            "mttc": self.mttc,
            "efficiency": self.efficiency,
            "technical_score": self.technical_score,
        }
        return {
            "sample_count": self.sample_count,
            "exact": {name: _fraction_report(value) for name, value in values.items()},
            "official_6dp": self.official(),
        }


@dataclass(frozen=True)
class Transition:
    baseline: MetricValues
    policy: MetricValues
    miss_to_hit: int
    hit_to_miss: int
    activation_turns: int
    activation_sessions: int

    @property
    def net_hits(self) -> int:
        return self.miss_to_hit - self.hit_to_miss

    def exact_delta(self, name: str) -> Fraction:
        return getattr(self.policy, name) - getattr(self.baseline, name)

    def official_delta(self, name: str) -> float:
        before = float(self.baseline.official()[name])
        after = float(self.policy.official()[name])
        return round(after - before, 6)

    def report(self) -> dict[str, Any]:
        names = (
            "hit_rate_at_10",
            "mrr",
            "mttc",
            "technical_score",
        )
        return {
            "miss_to_hit": self.miss_to_hit,
            "hit_to_miss": self.hit_to_miss,
            "net_hits": self.net_hits,
            "activation_turns": self.activation_turns,
            "activation_sessions": self.activation_sessions,
            "exact_delta": {
                name: _fraction_report(self.exact_delta(name)) for name in names
            },
            "official_6dp_delta": {
                name: self.official_delta(name) for name in names
            },
            "baseline": self.baseline.report(),
            "policy": self.policy.report(),
        }


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


def _fraction_report(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": round(float(value), 12),
    }


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "raw_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count


def _sha256_handle(handle: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    handle.seek(0)
    while True:
        block = handle.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
        count += len(block)
    return digest.hexdigest(), count


def _load_json_no_duplicates(path: Path) -> dict[str, Any]:
    def object_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise Rank1ReplayError("duplicate JSON key")
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_hook)
    if not isinstance(value, dict):
        raise Rank1ReplayError("expected JSON object")
    return value


def _validate_preregistration() -> dict[str, Any]:
    if not PREREG_PATH.is_file() or PREREG_PATH.is_symlink():
        raise Rank1ReplayError("preregistration file is unavailable")
    if not SELECTOR_SOURCE_PATH.is_file() or SELECTOR_SOURCE_PATH.is_symlink():
        raise Rank1ReplayError("frozen selector source is unavailable")
    raw = PREREG_PATH.read_bytes()
    value = _load_json_no_duplicates(PREREG_PATH)
    selector_sha, selector_bytes = _sha256_path(SELECTOR_SOURCE_PATH)
    relative_output = OUTPUT_PATH.relative_to(ROOT).as_posix()
    choreography = value.get("checkpoint_choreography", {})
    offline = value.get("causal_reset_contract", {}).get("offline_signal", {})
    amendment = value.get("protocol_amendment", {})
    reproduction = value.get("outcome_protocol", {}).get(
        "frozen_comparator_reproduction", {}
    )
    if not (
        hashlib.sha256(raw).hexdigest() == PREREG_RAW_SHA256
        and _canonical_sha256(value) == PREREG_CANONICAL_SHA256
        and value.get("status")
        == "PREREGISTERED_BEFORE_OUTCOME_WITH_MECHANICAL_AMENDMENT"
        and value.get("experiment_id") == EXPERIMENT_ID
        and value.get("parent_commit") == BASE_COMMIT
        and amendment.get("original_preregistration_commit")
        == ORIGINAL_PREREG_COMMIT
        and value.get("outcome_protocol", {}).get("one_shot_output")
        == relative_output
        and value.get("outcome_protocol", {}).get("attach_count") == 1
        and reproduction.get("fit_invocations_exact")
        == EXPECTED_FIT_INVOCATIONS
        and reproduction.get("inner_fit_invocations") == EXPECTED_INNER_FITS
        and reproduction.get("outer_fit_invocations") == EXPECTED_OUTER_FITS
        and reproduction.get("inner_quantile_selection_invocations_exact")
        == EXPECTED_SELECTION_INVOCATIONS
        and reproduction.get("base_seed") == BASE_SEED
        and tuple(reproduction.get("expected_fold_quantiles", ()))
        == EXPECTED_FOLD_QUANTILES
        and reproduction.get("expected_activation_raw_sha256")
        == EXPECTED_ACTIVATION_SHA256
        and offline.get("zero_based_feature_index") == VERSION_FEATURE_INDEX
        and choreography.get("branch") == BRANCH
        and choreography.get("remote") == REMOTE
        and choreography.get("remote_url") == REMOTE_URL
        and choreography.get("remote_tracking_ref") == REMOTE_REF
        and selector_sha == SELECTOR_SOURCE_SHA256
        and frozen.RR_MULTIPLIER == 1.0
        and tuple(frozen.QUANTILES)
        == tuple(float(value) / 64.0 for value in range(64))
        and frozen.KEEP_QUANTILE == 1.0
    ):
        raise Rank1ReplayError("preregistration binding drifted")
    return {
        "original_commit": ORIGINAL_PREREG_COMMIT,
        "protocol_amendment_commit": PREREG_COMMIT,
        "git_blob_oid": PREREG_BLOB_OID,
        "raw_sha256": PREREG_RAW_SHA256,
        "canonical_sha256": PREREG_CANONICAL_SHA256,
        "frozen_selector_source": {
            "path": SELECTOR_SOURCE_PATH.relative_to(ROOT).as_posix(),
            "git_blob_oid": SELECTOR_SOURCE_BLOB_OID,
            "sha256": selector_sha,
            "bytes": selector_bytes,
        },
    }


def _git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=" + ROOT.as_posix(), *args],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise Rank1ReplayError("Git checkpoint validation failed")
    return completed.stdout.strip()


def _commit_parent(commit: str) -> str:
    values = _git(("rev-list", "--parents", "-n", "1", commit)).split()
    if len(values) != 2 or values[0] != commit:
        raise Rank1ReplayError("commit must have exactly one parent")
    return values[1]


def _changed_paths(commit: str) -> set[str]:
    return {
        line.strip().replace("\\", "/")
        for line in _git(
            ("diff-tree", "--no-commit-id", "--name-only", "-r", commit)
        ).splitlines()
        if line.strip()
    }


def _validate_git_checkpoint(implementation_commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise Rank1ReplayError("implementation commit must be a full SHA")
    if _git(("status", "--porcelain=v1", "--untracked-files=all")):
        raise Rank1ReplayError("tracked or untracked worktree state is not clean")
    head = _git(("rev-parse", "HEAD"))
    branch = _git(("symbolic-ref", "--short", "HEAD"))
    remote_url = _git(("remote", "get-url", REMOTE))
    remote_head = _git(("rev-parse", REMOTE_REF))
    prereg_relative = PREREG_PATH.relative_to(ROOT).as_posix()
    selector_relative = SELECTOR_SOURCE_PATH.relative_to(ROOT).as_posix()
    prereg_blob = _git(("rev-parse", PREREG_COMMIT + ":" + prereg_relative))
    original_prereg_blob = _git(
        ("rev-parse", ORIGINAL_PREREG_COMMIT + ":" + prereg_relative)
    )
    selector_blob = _git(("rev-parse", BASE_COMMIT + ":" + selector_relative))
    if not (
        head == implementation_commit
        and branch == BRANCH
        and remote_url == REMOTE_URL
        and remote_head == head
        and _commit_parent(head) == PREREG_COMMIT
        and _commit_parent(PREREG_COMMIT) == ORIGINAL_PREREG_COMMIT
        and _commit_parent(ORIGINAL_PREREG_COMMIT) == BASE_COMMIT
        and _changed_paths(head) == IMPLEMENTATION_PATHS
        and _changed_paths(PREREG_COMMIT) == PREREG_PATH_SET
        and _changed_paths(ORIGINAL_PREREG_COMMIT) == PREREG_PATH_SET
        and prereg_blob == PREREG_BLOB_OID
        and original_prereg_blob == ORIGINAL_PREREG_BLOB_OID
        and selector_blob == SELECTOR_SOURCE_BLOB_OID
    ):
        raise Rank1ReplayError("commit choreography drifted")
    implementation_files: dict[str, dict[str, Any]] = {}
    for relative in sorted(IMPLEMENTATION_PATHS):
        path = ROOT / relative
        digest, size = _sha256_path(path)
        implementation_files[relative] = {
            "git_blob_oid": _git(("rev-parse", head + ":" + relative)),
            "sha256": digest,
            "bytes": size,
        }
    return {
        "implementation_commit": head,
        "protocol_amendment_commit": PREREG_COMMIT,
        "original_preregistration_commit": ORIGINAL_PREREG_COMMIT,
        "base_commit": BASE_COMMIT,
        "branch": branch,
        "remote": REMOTE,
        "remote_url": remote_url,
        "remote_tracking_ref": REMOTE_REF,
        "remote_equal": True,
        "clean_including_untracked": True,
        "implementation_paths_exact": True,
        "protocol_amendment_paths_exact": True,
        "original_preregistration_paths_exact": True,
        "frozen_selector_source_blob_exact": True,
        "implementation_files": implementation_files,
    }


def _validate_environment() -> dict[str, Any]:
    import sklearn
    import xgboost

    actual = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "workers": 1,
    }
    expected = {
        "python": "3.9.19",
        "numpy": "1.26.4",
        "sklearn": "1.1.3",
        "xgboost": "1.7.6",
        "workers": 1,
    }
    if actual != expected:
        raise Rank1ReplayError("dependency identity mismatch")
    return actual


def _process_memory() -> tuple[int, int]:
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
    current_process = ctypes.windll.kernel32.GetCurrentProcess
    current_process.restype = wintypes.HANDLE
    memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    memory_info.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    memory_info.restype = wintypes.BOOL
    if not memory_info(current_process(), ctypes.byref(counters), counters.cb):
        return 0, 0
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def decode_intent_versions(
    projected_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features = np.asarray(projected_features)
    if (
        features.ndim != 4
        or features.shape[-2:] != (CANDIDATE_COUNT, FEATURE_COUNT)
        or features.dtype != np.float32
        or features.shape[1] < 1
    ):
        raise Rank1ReplayError("projected feature schema is invalid")

    version_column = np.asarray(features[..., VERSION_FEATURE_INDEX])
    first_version = version_column[:, :, 0]
    if not np.array_equal(
        version_column,
        np.broadcast_to(first_version[:, :, None], version_column.shape),
    ):
        raise Rank1ReplayError("version feature is candidate-dependent")
    if not np.isfinite(first_version).all():
        raise Rank1ReplayError("version feature is non-finite")
    decoded = np.rint(first_version * np.float32(10.0)).astype(np.int16)
    encoded = np.minimum(decoded, 10).astype(np.float32) / np.float32(10.0)
    if not np.array_equal(first_version, encoded):
        raise Rank1ReplayError("version feature does not exactly re-encode")
    if not np.all(decoded[:, 0] == 1):
        raise Rank1ReplayError("first-turn version is not one")
    deltas = np.diff(decoded, axis=1)
    if np.any((deltas < 0) | (deltas > 1)):
        raise Rank1ReplayError("version is not causal monotone unit-step state")
    if decoded.shape[1] > 1 and np.any(first_version[:, :-1] >= np.float32(1.0)):
        raise Rank1ReplayError("version feature saturates before the final turn")

    reset = np.zeros(decoded.shape, dtype=bool)
    reset[:, 0] = True
    if decoded.shape[1] > 1:
        reset[:, 1:] = deltas == 1

    age_column = np.asarray(features[..., GOAL_AGE_FEATURE_INDEX])
    first_age = age_column[:, :, 0]
    if not np.array_equal(
        age_column, np.broadcast_to(first_age[:, :, None], age_column.shape)
    ):
        raise Rank1ReplayError("goal-age feature is candidate-dependent")
    if np.any(first_age[reset] != np.float32(0.1)):
        raise Rank1ReplayError("reset does not have exact goal age 0.1")

    override_column = np.asarray(features[..., CURRENT_OVERRIDE_FEATURE_INDEX])
    first_override = override_column[:, :, 0]
    if not np.array_equal(
        override_column,
        np.broadcast_to(first_override[:, :, None], override_column.shape),
    ) or np.any((first_override != 0.0) & (first_override != 1.0)):
        raise Rank1ReplayError("current-turn override audit column is invalid")
    if np.any((first_override > 0.5) & ~reset):
        raise Rank1ReplayError("visible override does not imply a version reset")

    count_column = np.asarray(features[..., OVERRIDE_COUNT_FEATURE_INDEX])
    first_count = count_column[:, :, 0]
    if not np.array_equal(
        count_column, np.broadcast_to(first_count[:, :, None], count_column.shape)
    ):
        raise Rank1ReplayError("override-count feature is candidate-dependent")
    decoded_count = np.rint(first_count * np.float32(5.0)).astype(np.int16)
    reencoded_count = np.minimum(decoded_count, 5).astype(np.float32) / np.float32(5.0)
    if not np.array_equal(first_count, reencoded_count):
        raise Rank1ReplayError("override-count feature does not exactly re-encode")
    count_delta = np.diff(decoded_count, axis=1)
    if np.any((count_delta < 0) | (count_delta > 1)):
        raise Rank1ReplayError("override count is not monotone unit-step state")

    decoded.setflags(write=False)
    reset.setflags(write=False)
    return decoded, reset, {
        "feature_index": VERSION_FEATURE_INDEX,
        "candidate_invariant_max_spread": 0.0,
        "unique_versions": sorted(int(value) for value in np.unique(decoded)),
        "version_increment_boundaries": int(reset[:, 1:].sum()),
        "version_decreases": int(np.sum(deltas < 0)),
        "maximum_goal_version_fraction": float(first_version.max()),
        "non_final_saturation_count": int(
            np.sum(first_version[:, :-1] >= np.float32(1.0))
        )
        if first_version.shape[1] > 1
        else 0,
        "explicit_override_turns": int(np.sum(first_override > 0.5)),
        "reset_goal_age_mismatches": 0,
        "reset_source": "goal_version_fraction_delta_only",
        "proxy_or_eligible_input_used": False,
    }


def rank1_seen_replacement(
    order: Sequence[str], served: set[str]
) -> tuple[str, ...]:
    """Replace only a previously served rank 1 with the best unseen tail item."""

    ranked = tuple(str(identifier) for identifier in order)
    if len(ranked) != CANDIDATE_COUNT or len(set(ranked)) != CANDIDATE_COUNT:
        raise Rank1ReplayError("ranked C100 must contain 100 unique products")
    baseline = ranked[:10]
    if baseline[0] not in served:
        return baseline
    replacement = next(
        (identifier for identifier in ranked[10:] if identifier not in served),
        None,
    )
    if replacement is None:
        return baseline
    page = (replacement, *baseline[1:])
    if len(page) != 10 or len(set(page)) != 10:
        raise Rank1ReplayError("rank-1 replacement produced an invalid page")
    return page


def reconstruct_v19_order(
    turn: Mapping[str, Any], chosen_index: int, activated: bool
) -> tuple[str, ...]:
    c100 = tuple(str(value) for value in turn.get("c100", ()))
    actions = turn.get("actions", {})
    p11 = tuple(str(value) for value in actions.get("KEEP_P11", ())) if isinstance(actions, Mapping) else ()
    if (
        len(c100) != CANDIDATE_COUNT
        or len(set(c100)) != CANDIDATE_COUNT
        or len(p11) != 10
        or len(set(p11)) != 10
        or set(p11) != set(c100[:10])
    ):
        raise Rank1ReplayError("blind trace P11/C100 invariant failed")
    order = list(p11 + c100[10:])
    if activated:
        if not 10 <= int(chosen_index) < CANDIDATE_COUNT:
            raise Rank1ReplayError("activated challenger is not a C100 tail item")
        challenger = c100[int(chosen_index)]
        position = order.index(challenger)
        order[9], order[position] = order[position], order[9]
    if len(order) != CANDIDATE_COUNT or len(set(order)) != CANDIDATE_COUNT:
        raise Rank1ReplayError("v1.9 full order reconstruction failed")
    return tuple(order)


def replay_pages(
    traces: Sequence[Sequence[Mapping[str, Any]]],
    chosen: np.ndarray,
    activation: np.ndarray,
    versions: np.ndarray,
    measure_timing: bool = False,
) -> ReplayBundle:
    chosen = np.asarray(chosen)
    activation = np.asarray(activation, dtype=bool)
    versions = np.asarray(versions)
    session_count, turn_count = versions.shape
    if (
        len(traces) != session_count
        or chosen.shape != (session_count, turn_count)
        or activation.shape != (session_count, turn_count)
    ):
        raise Rank1ReplayError("replay surface shape mismatch")

    baseline_pages = np.empty((session_count, turn_count, 10), dtype=np.int16)
    candidate_pages = np.empty_like(baseline_pages)
    changed = np.zeros((session_count, turn_count), dtype=bool)
    last_reset_turn = np.ones(session_count, dtype=np.int16)
    reset_count = 0
    changed_sessions = 0
    changed_turns = 0
    baseline_distinct_total = 0
    candidate_distinct_total = 0
    latency_ns: list[int] = []

    for session, session_turns in enumerate(traces):
        if len(session_turns) != turn_count:
            raise Rank1ReplayError("blind trace turn count mismatch")
        served: set[str] = set()
        last_version: int | None = None
        session_changed = False
        all_baseline: set[str] = set()
        all_candidate: set[str] = set()
        for turn_index, turn in enumerate(session_turns):
            version = int(versions[session, turn_index])
            is_reset = last_version is None or version != last_version
            if is_reset:
                served.clear()
                reset_count += 1
                last_reset_turn[session] = turn_index + 1
            order = reconstruct_v19_order(
                turn,
                int(chosen[session, turn_index]),
                bool(activation[session, turn_index]),
            )
            baseline = order[:10]
            policy_started = time.perf_counter_ns() if measure_timing else 0
            candidate = rank1_seen_replacement(order, served)
            if measure_timing:
                latency_ns.append(time.perf_counter_ns() - policy_started)
            page_changed = candidate != baseline
            if is_reset and page_changed:
                raise Rank1ReplayError("reset page is not comparator identity")
            if page_changed:
                if not (
                    candidate[1:] == baseline[1:]
                    and baseline[0] in served
                    and candidate[0] in order[10:]
                    and candidate[0] not in served
                ):
                    raise Rank1ReplayError("rank-1 structural invariant failed")
                changed_turns += 1
                session_changed = True
            elif baseline[0] not in served and candidate != baseline:
                raise Rank1ReplayError("unseen rank-1 identity invariant failed")

            c100 = tuple(str(value) for value in turn["c100"])
            index = {identifier: position for position, identifier in enumerate(c100)}
            try:
                baseline_pages[session, turn_index] = [
                    index[identifier] for identifier in baseline
                ]
                candidate_pages[session, turn_index] = [
                    index[identifier] for identifier in candidate
                ]
            except KeyError as error:
                raise Rank1ReplayError("served page escaped current C100") from error
            changed[session, turn_index] = page_changed
            served.update(candidate)
            all_baseline.update(baseline)
            all_candidate.update(candidate)
            last_version = version

        changed_sessions += int(session_changed)
        baseline_distinct_total += len(all_baseline)
        candidate_distinct_total += len(all_candidate)

    for value in (baseline_pages, candidate_pages, changed, last_reset_turn):
        value.setflags(write=False)
    identity = {
        "baseline_pages": _array_identity(baseline_pages),
        "candidate_pages": _array_identity(candidate_pages),
        "changed": _array_identity(changed),
        "last_reset_turn": _array_identity(last_reset_turn),
    }
    structural = {
        "reset_count": reset_count,
        "changed_turns": changed_turns,
        "changed_sessions": changed_sessions,
        "changed_slots_per_changed_turn": 1 if changed_turns else 0,
        "ranks_2_to_10_byte_identical": True,
        "removed_rank1_already_served": True,
        "inserted_from_unseen_c100_tail": True,
        "reset_pages_identity": True,
        "baseline_mean_distinct_products": round(
            baseline_distinct_total / session_count, 6
        ),
        "candidate_mean_distinct_products": round(
            candidate_distinct_total / session_count, 6
        ),
    }
    timing = {
        "sample_count": len(latency_ns),
        "p50_microseconds": round(float(np.percentile(latency_ns, 50)) / 1_000.0, 6)
        if latency_ns
        else None,
        "p95_microseconds": round(float(np.percentile(latency_ns, 95)) / 1_000.0, 6)
        if latency_ns
        else None,
        "maximum_microseconds": round(max(latency_ns) / 1_000.0, 6)
        if latency_ns
        else None,
    }
    return ReplayBundle(
        baseline_pages,
        candidate_pages,
        changed,
        last_reset_turn,
        identity,
        structural,
        timing,
    )


def derive_baseline_session_hit(
    baseline_rank: np.ndarray, eligible_from: np.ndarray
) -> np.ndarray:
    baseline_rank = np.asarray(baseline_rank)
    eligible_from = np.asarray(eligible_from)
    if (
        baseline_rank.ndim != 2
        or eligible_from.shape != (baseline_rank.shape[0],)
        or np.any((baseline_rank < 0) | (baseline_rank > 10))
        or np.any((eligible_from < 1) | (eligible_from > baseline_rank.shape[1]))
    ):
        raise Rank1ReplayError("baseline hit derivation schema failed")
    eligible = (
        np.arange(1, baseline_rank.shape[1] + 1)[None, :]
        >= eligible_from[:, None]
    )
    result = np.any(eligible & (baseline_rank > 0), axis=1).astype(np.uint8)
    result.setflags(write=False)
    return result


def _validate_label_member(name: str, value: np.ndarray) -> None:
    if name == "baseline_rank" and np.any(value > 10):
        raise Rank1ReplayError("baseline rank is outside 0..10")
    if name == "positive_index" and np.any((value < -1) | (value >= 100)):
        raise Rank1ReplayError("positive index is outside -1..99")
    if name == "eligible_from" and np.any((value < 1) | (value > 10)):
        raise Rank1ReplayError("eligible_from is outside 1..10")
    if name in {"outer_fold", "inner_fold"} and (
        np.any(value > 4) or set(np.unique(value).tolist()) != set(range(5))
    ):
        raise Rank1ReplayError("fold member is invalid")


def load_outcomes_from_open_handle(
    handle: BinaryIO, np_load: Any = np.load
) -> OutcomeBundle:
    """Access exactly five frozen members in their preregistered order."""

    values: dict[str, np.ndarray] = {}
    archive = np_load(handle, allow_pickle=False)
    try:
        for name, shape, dtype in LABEL_MEMBER_SPECS:
            member = archive[name]
            if not (
                isinstance(member, np.ndarray)
                and member.shape == shape
                and str(member.dtype) == dtype
            ):
                raise Rank1ReplayError("sealed label member schema failed")
            copied = np.asarray(member).copy()
            _validate_label_member(name, copied)
            copied.setflags(write=False)
            values[name] = copied
            del member
    finally:
        archive.close()
    if tuple(values) != tuple(name for name, _shape, _dtype in LABEL_MEMBER_SPECS):
        raise Rank1ReplayError("sealed label member order drifted")
    return OutcomeBundle(**values)


def _trace_baseline_rank_mismatches(
    traces: Sequence[Sequence[Mapping[str, Any]]],
    outcomes: OutcomeBundle,
) -> int:
    mismatches = 0
    for session, turns in enumerate(traces):
        for turn_index, turn in enumerate(turns):
            positive = int(outcomes.positive_index[session, turn_index])
            expected = int(outcomes.baseline_rank[session, turn_index])
            if positive < 0:
                actual = 0
            else:
                c100 = tuple(str(value) for value in turn["c100"])
                p11 = tuple(str(value) for value in turn["actions"]["KEEP_P11"])
                target = c100[positive]
                actual = p11.index(target) + 1 if target in p11 else 0
            mismatches += int(actual != expected)
    return mismatches


def state_from_positive_index(
    pages: np.ndarray,
    positive_index: np.ndarray,
    eligible_from: np.ndarray,
) -> dict[str, np.ndarray]:
    pages = np.asarray(pages)
    positive = np.asarray(positive_index)
    eligible_from = np.asarray(eligible_from)
    if (
        pages.ndim != 3
        or pages.shape[2] != 10
        or positive.shape != pages.shape[:2]
        or eligible_from.shape != (pages.shape[0],)
        or np.any((positive < -1) | (positive >= CANDIDATE_COUNT))
    ):
        raise Rank1ReplayError("policy-state schema failed")
    eligible = (
        np.arange(1, pages.shape[1] + 1)[None, :] >= eligible_from[:, None]
    )
    matches = pages == positive[:, :, None]
    hit_turn = matches.any(axis=2) & (positive >= 0) & eligible
    hit = hit_turn.any(axis=1)
    first_index = np.argmax(hit_turn, axis=1)
    ranks = np.where(
        matches,
        np.arange(1, 11, dtype=np.int16)[None, None, :],
        0,
    ).max(axis=2)
    first_rank = np.take_along_axis(ranks, first_index[:, None], axis=1)[:, 0]
    first_rank = np.where(hit, first_rank, 0).astype(np.int16)
    first_turn = np.where(hit, first_index + 1, pages.shape[1] + 1).astype(np.int16)
    for value in (hit, first_rank, first_turn):
        value.setflags(write=False)
    return {"hit": hit, "first_rank": first_rank, "first_turn": first_turn}


def dominance_audit(
    baseline_state: Mapping[str, np.ndarray],
    candidate_state: Mapping[str, np.ndarray],
) -> dict[str, int]:
    baseline_hit = np.asarray(baseline_state["hit"], dtype=bool)
    candidate_hit = np.asarray(candidate_state["hit"], dtype=bool)
    baseline_turn = np.asarray(baseline_state["first_turn"])
    candidate_turn = np.asarray(candidate_state["first_turn"])
    baseline_rank = np.asarray(baseline_state["first_rank"])
    candidate_rank = np.asarray(candidate_state["first_rank"])
    both = baseline_hit & candidate_hit
    earlier = candidate_hit & (candidate_turn < baseline_turn)
    same = both & (candidate_turn == baseline_turn)
    new_hit = ~baseline_hit & candidate_hit
    return {
        "hit_to_miss": int(np.sum(baseline_hit & ~candidate_hit)),
        "later_first_hit": int(np.sum(both & (candidate_turn > baseline_turn))),
        "same_turn_worse_rank": int(
            np.sum(same & (candidate_rank > baseline_rank))
        ),
        "earlier_hit_not_rank1": int(np.sum(earlier & (candidate_rank != 1))),
        "new_hit_not_rank1": int(np.sum(new_hit & (candidate_rank != 1))),
    }


def metric_values(
    state: Mapping[str, np.ndarray], mask: np.ndarray
) -> MetricValues:
    selected = np.asarray(mask, dtype=bool)
    hit = np.asarray(state["hit"], dtype=bool)[selected]
    rank = np.asarray(state["first_rank"])[selected]
    turn = np.asarray(state["first_turn"])[selected]
    count = int(hit.size)
    if count <= 0:
        raise Rank1ReplayError("metric mask is empty")
    hits = int(hit.sum())
    hr = Fraction(hits, count)
    rr_sum = sum(
        (Fraction(1, int(value)) if bool(is_hit) else Fraction(0, 1))
        for value, is_hit in zip(rank.tolist(), hit.tolist())
    )
    mrr = rr_sum / count
    mttc = Fraction(
        sum(int(value) if bool(is_hit) else 11 for value, is_hit in zip(turn.tolist(), hit.tolist())),
        count,
    )
    efficiency = (Fraction(11, 1) - mttc) / 10
    efficiency = min(Fraction(1, 1), max(Fraction(0, 1), efficiency))
    technical = Fraction(1, 2) * hr + Fraction(3, 10) * mrr + Fraction(1, 5) * efficiency
    return MetricValues(count, hr, mrr, mttc, efficiency, technical)


def transition_metrics(
    baseline_state: Mapping[str, np.ndarray],
    candidate_state: Mapping[str, np.ndarray],
    changed: np.ndarray,
    mask: np.ndarray,
) -> Transition:
    selected = np.asarray(mask, dtype=bool)
    baseline_hit = np.asarray(baseline_state["hit"], dtype=bool)
    candidate_hit = np.asarray(candidate_state["hit"], dtype=bool)
    changed = np.asarray(changed, dtype=bool)
    return Transition(
        baseline=metric_values(baseline_state, selected),
        policy=metric_values(candidate_state, selected),
        miss_to_hit=int(np.sum(selected & ~baseline_hit & candidate_hit)),
        hit_to_miss=int(np.sum(selected & baseline_hit & ~candidate_hit)),
        activation_turns=int(changed[selected].sum()),
        activation_sessions=int(np.any(changed[selected], axis=1).sum()),
    )


def _dual_nonnegative(exact: Fraction, official: float) -> bool:
    return exact >= 0 and official >= 0.0


def _dual_nonpositive(exact: Fraction, official: float) -> bool:
    return exact <= 0 and official <= 0.0


def _dual_strict_positive(exact: Fraction, official: float) -> bool:
    return exact > 0 and official > 0.0


def passes_promotion_gates(
    aggregate: Transition,
    folds: Sequence[Transition],
    structural_ok: bool,
    exact_repeat: bool,
) -> bool:
    aggregate_official = aggregate.policy.official()
    return bool(
        structural_ok
        and exact_repeat
        and aggregate.policy.hit_rate_at_10 > aggregate.baseline.hit_rate_at_10
        and float(aggregate_official["hit_rate_at_10"]) > 0.9715
        and aggregate.miss_to_hit >= 1
        and aggregate.hit_to_miss == 0
        and _dual_nonnegative(
            aggregate.exact_delta("mrr"), aggregate.official_delta("mrr")
        )
        and _dual_nonpositive(
            aggregate.exact_delta("mttc"), aggregate.official_delta("mttc")
        )
        and _dual_strict_positive(
            aggregate.exact_delta("technical_score"),
            aggregate.official_delta("technical_score"),
        )
        and len(folds) == OUTER_FOLDS
        and all(
            row.hit_to_miss == 0
            and row.net_hits >= 0
            and _dual_nonnegative(
                row.exact_delta("hit_rate_at_10"),
                row.official_delta("hit_rate_at_10"),
            )
            and _dual_nonnegative(
                row.exact_delta("mrr"), row.official_delta("mrr")
            )
            and _dual_nonpositive(
                row.exact_delta("mttc"), row.official_delta("mttc")
            )
            for row in folds
        )
    )


def _trace_ranking(value: object, limit: int, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not 0 < len(value) <= limit
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise Rank1ReplayError("invalid blind trace ranking: " + label)
    return tuple(value)


def _load_blind_traces() -> tuple[tuple[dict[str, Any], ...], ...]:
    if not TRACE_AGGREGATE_PATH.is_file() or TRACE_AGGREGATE_PATH.is_symlink():
        raise Rank1ReplayError("blind trace aggregate binding drifted")
    aggregate_sha, _aggregate_bytes = _sha256_path(TRACE_AGGREGATE_PATH)
    if aggregate_sha != TRACE_AGGREGATE_SHA256:
        raise Rank1ReplayError("blind trace aggregate binding drifted")
    trace_root = TRACE_AGGREGATE_PATH.parent
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    combined = hashlib.sha256()
    for shard_number, (filename, expected_sha) in enumerate(
        TRACE_SPECS, start=1
    ):
        path = trace_root / filename
        if not path.is_file() or path.is_symlink():
            raise Rank1ReplayError("blind trace shard is unavailable")
        digest, _size = _sha256_path(path)
        if digest != expected_sha:
            raise Rank1ReplayError("blind trace shard hash drifted")
        count = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.strip():
                    raise Rank1ReplayError("blind trace has a blank row")
                raw = json.loads(line)
                if not isinstance(raw, dict) or set(raw) != {
                    "ordinal",
                    "turn",
                    "actions",
                    "candidate_pools",
                }:
                    raise Rank1ReplayError("blind trace row schema drifted")
                local_ordinal = raw["ordinal"]
                turn = raw["turn"]
                if (
                    not isinstance(local_ordinal, int)
                    or isinstance(local_ordinal, bool)
                    or not 1 <= local_ordinal <= 500
                    or not isinstance(turn, int)
                    or isinstance(turn, bool)
                    or not 1 <= turn <= TURN_COUNT
                ):
                    raise Rank1ReplayError("blind trace coordinate is invalid")
                ordinal = (shard_number - 1) * 500 + local_ordinal
                coordinate = (ordinal, turn)
                if coordinate in rows:
                    raise Rank1ReplayError("blind trace coordinate is duplicated")
                actions = raw["actions"]
                pools = raw["candidate_pools"]
                if (
                    not isinstance(actions, dict)
                    or set(actions) != set(TRACE_ACTIONS)
                    or not isinstance(pools, dict)
                    or set(pools) != {"c20", "c50", "c100"}
                ):
                    raise Rank1ReplayError("blind trace registry drifted")
                clean_actions = {
                    name: _trace_ranking(actions[name], 10, name)
                    for name in TRACE_ACTIONS
                }
                c20 = _trace_ranking(pools["c20"], 20, "c20")
                c50 = _trace_ranking(pools["c50"], 50, "c50")
                c100 = _trace_ranking(pools["c100"], 100, "c100")
                if not (
                    len(c20) == 20
                    and len(c50) == 50
                    and len(c100) == 100
                    and c20 == c50[:20]
                    and c50 == c100[:50]
                    and clean_actions["KEEP_R08"] == c20[:10]
                    and set(clean_actions["KEEP_R08"])
                    == set(clean_actions["KEEP_P11"])
                    and clean_actions["ASK"] == clean_actions["KEEP_P11"]
                    and all(
                        set(clean_actions[name]).issubset(set(c50))
                        for name in (
                            "CANDIDATE_RERANK",
                            "FROZEN_SEMANTIC_RERANK",
                        )
                    )
                ):
                    raise Rank1ReplayError("blind trace membership invariant failed")
                rows[coordinate] = {
                    "actions": clean_actions,
                    "c20": c20,
                    "c50": c50,
                    "c100": c100,
                }
                normalized = dict(raw)
                normalized["ordinal"] = ordinal
                combined.update(_canonical_bytes(normalized) + b"\n")
                count += 1
        post_digest, _post_size = _sha256_path(path)
        if count != 5_000 or post_digest != expected_sha:
            raise Rank1ReplayError("blind trace shard row count drifted")
    if (
        len(rows) != SESSION_COUNT * TURN_COUNT
        or combined.hexdigest() != COMBINED_TRACE_SHA256
    ):
        raise Rank1ReplayError("combined blind trace identity drifted")
    return tuple(
        tuple(rows[(ordinal, turn)] for turn in range(1, TURN_COUNT + 1))
        for ordinal in range(1, SESSION_COUNT + 1)
    )


def _reproduce_nested_activation(
    surface: frozen.ActionSurface,
    labels: Mapping[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Pinned v1.9 nested OOF activation without importing the proxy module."""

    outer = np.asarray(labels["outer_fold"])
    inner = np.asarray(labels["inner_fold"])
    flat_x = surface.gate_features.reshape(-1, len(base.GATE_FEATURE_NAMES))
    flat_action = surface.action.reshape(-1)
    flat_session = np.repeat(np.arange(len(outer)), TURN_COUNT)
    targets = (surface.rescue.reshape(-1), surface.regret.reshape(-1))
    weights = (
        surface.rescue_weights.reshape(-1),
        surface.regret_weights.reshape(-1),
    )
    activation = np.zeros_like(surface.action, dtype=bool)
    selections: list[dict[str, Any]] = []
    inner_fit_invocations = 0
    outer_fit_invocations = 0
    selection_invocations = 0
    for outer_fold in range(OUTER_FOLDS):
        train_sessions = outer != outer_fold
        held_sessions = outer == outer_fold
        inner_probability = [
            np.zeros_like(surface.action, dtype=np.float32) for _ in range(2)
        ]
        for inner_index in range(OUTER_FOLDS):
            model_train = train_sessions & (inner != inner_index)
            model_valid = train_sessions & (inner == inner_index)
            train_rows = flat_action & model_train[flat_session]
            valid_rows = flat_action & model_valid[flat_session]
            if not np.any(train_rows) or not np.any(valid_rows):
                raise Rank1ReplayError("nested admission partition is empty")
            for head in range(2):
                inner_fit_invocations += 1
                inner_probability[head].reshape(-1)[valid_rows] = (
                    frozen._fit_predict(
                        flat_x,
                        train_rows,
                        valid_rows,
                        targets[head],
                        weights[head],
                        seed
                        + head * 10_000
                        + outer_fold * 31
                        + inner_index,
                    )
                )
        inner_utility = inner_probability[0] - inner_probability[1]
        selected = frozen._select_inner_quantile(
            inner_utility,
            surface,
            labels,
            train_sessions,
            inner,
        )
        selection_invocations += 1
        train_rows = flat_action & train_sessions[flat_session]
        held_rows = flat_action & held_sessions[flat_session]
        train_probability = [
            np.zeros_like(surface.action, dtype=np.float32) for _ in range(2)
        ]
        held_probability = [
            np.zeros_like(surface.action, dtype=np.float32) for _ in range(2)
        ]
        for head in range(2):
            outer_fit_invocations += 1
            model, mean, scale = base._fit_gate_model(
                flat_x[train_rows],
                targets[head][train_rows],
                weights[head][train_rows],
                seed + head * 10_000 + outer_fold * 101,
            )
            train_probability[head].reshape(-1)[train_rows] = (
                base._predict_gate(
                    model, mean, scale, flat_x[train_rows]
                ).astype(np.float32)
            )
            held_probability[head].reshape(-1)[held_rows] = (
                base._predict_gate(
                    model, mean, scale, flat_x[held_rows]
                ).astype(np.float32)
            )
        train_utility = train_probability[0] - train_probability[1]
        threshold = frozen._threshold_at_quantile(
            train_utility[surface.action & train_sessions[:, None]],
            float(selected["quantile"]),
        )
        held_utility = held_probability[0] - held_probability[1]
        activation[held_sessions] = surface.action[held_sessions] & (
            held_utility[held_sessions] >= threshold
        )
        selections.append(
            {
                "fold": outer_fold,
                "quantile": float(selected["quantile"]),
                "threshold": float(threshold),
            }
        )
    ordered_quantiles = tuple(float(row["quantile"]) for row in selections)
    fit_invocations = inner_fit_invocations + outer_fit_invocations
    if not (
        seed == BASE_SEED
        and inner_fit_invocations == EXPECTED_INNER_FITS
        and outer_fit_invocations == EXPECTED_OUTER_FITS
        and fit_invocations == EXPECTED_FIT_INVOCATIONS
        and selection_invocations == EXPECTED_SELECTION_INVOCATIONS
        and ordered_quantiles == EXPECTED_FOLD_QUANTILES
    ):
        raise Rank1ReplayError("frozen comparator reproduction contract drifted")
    audit = {
        "fit_invocations": fit_invocations,
        "inner_fit_invocations": inner_fit_invocations,
        "outer_fit_invocations": outer_fit_invocations,
        "inner_quantile_selection_invocations": selection_invocations,
        "ordered_fold_quantiles": list(ordered_quantiles),
        "held_outer_rows_used_for_fit_or_quantile_selection": False,
        "new_v2_12_fit_or_selection": False,
    }
    return activation, selections, audit


def _validate_target_free_file(
    path: Path, expected_sha256: str, expected_bytes: int
) -> tuple[int, int]:
    if not path.is_file() or path.is_symlink():
        raise Rank1ReplayError("frozen target-free input is unavailable")
    digest, size = _sha256_path(path)
    if digest != expected_sha256 or size != expected_bytes:
        raise Rank1ReplayError("frozen target-free input hash drifted")
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _load_target_free_inputs() -> TargetFreeInputs:
    snapshots = {
        "projected_features": _validate_target_free_file(
            PROJECTED_FEATURES_PATH,
            PROJECTED_FEATURES_SHA256,
            PROJECTED_FEATURES_BYTES,
        ),
        "oof_scores": _validate_target_free_file(
            OOF_SCORES_PATH, OOF_SCORES_SHA256, OOF_SCORES_BYTES
        ),
    }
    projected = np.load(PROJECTED_FEATURES_PATH, mmap_mode="r")
    scores = np.load(OOF_SCORES_PATH, mmap_mode="r")
    if (
        projected.shape
        != (SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT, FEATURE_COUNT)
        or projected.dtype != np.float32
        or scores.shape != (SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT)
        or scores.dtype != np.float32
    ):
        raise Rank1ReplayError("frozen target-free array schema drifted")
    versions, reset_mask, reset_audit = decode_intent_versions(projected)
    incumbent = base._incumbent_indices(projected)
    chosen, _margin, _top_gap = base.choose_slot10(scores, incumbent)
    if hashlib.sha256(chosen.tobytes()).hexdigest() != EXPECTED_CHOSEN_SHA256:
        raise Rank1ReplayError("frozen chosen identity drifted")
    traces = _load_blind_traces()
    if len(traces) != SESSION_COUNT or any(len(turns) != TURN_COUNT for turns in traces):
        raise Rank1ReplayError("blind trace schema drifted")
    for value in (incumbent, chosen):
        value.setflags(write=False)
    return TargetFreeInputs(
        projected,
        scores,
        traces,
        versions,
        reset_mask,
        chosen,
        incumbent,
        reset_audit,
        snapshots,
    )


def _validate_source_snapshots(
    snapshots: Mapping[str, tuple[int, int]]
) -> bool:
    current = {
        "projected_features": (
            int(PROJECTED_FEATURES_PATH.stat().st_size),
            int(PROJECTED_FEATURES_PATH.stat().st_mtime_ns),
        ),
        "oof_scores": (
            int(OOF_SCORES_PATH.stat().st_size),
            int(OOF_SCORES_PATH.stat().st_mtime_ns),
        ),
    }
    return dict(snapshots) == current


def _check_output_components(path: Path, root: Path) -> None:
    root_resolved = root.resolve(strict=True)
    if path.is_absolute():
        candidate = path
    else:
        candidate = root / path
    parent_resolved = candidate.parent.resolve(strict=False)
    if parent_resolved != root_resolved and root_resolved not in parent_resolved.parents:
        raise Rank1ReplayError("one-shot output escapes the worktree")
    relative = candidate.relative_to(root)
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise Rank1ReplayError("one-shot output has a symlink component")
    if candidate.exists() or candidate.is_symlink():
        raise Rank1ReplayError("one-shot output is already consumed")


def _write_receipt_payload(handle: BinaryIO, value: Mapping[str, Any]) -> tuple[int, str]:
    payload = _canonical_bytes(value) + b"\n"
    handle.seek(0)
    handle.truncate(0)
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
    return len(payload), hashlib.sha256(payload).hexdigest()


def _invalid_receipt_payload(
    implementation_commit: str, error: BaseException
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "INVALID_ONE_SHOT_CONSUMED",
        "implementation_commit": implementation_commit,
        "error_class": type(error).__name__,
        "rerun_forbidden": True,
    }


def _write_descriptor_payload(
    descriptor: int, value: Mapping[str, Any]
) -> tuple[int, str]:
    payload = _canonical_bytes(value) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    view = memoryview(payload)
    while view:
        count = os.write(descriptor, view)
        if count <= 0:
            raise OSError("short receipt descriptor write")
        view = view[count:]
    os.fsync(descriptor)
    return len(payload), hashlib.sha256(payload).hexdigest()


def open_one_shot_receipt(
    path: Path,
    root: Path,
    implementation_commit: str,
) -> BinaryIO:
    _check_output_components(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_output_components(path, root)
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except FileExistsError as error:
        raise Rank1ReplayError("one-shot output is already consumed") from error
    marker = {
        "schema_version": "small-ranker-one-shot-consumed-marker.v1",
        "experiment_id": EXPERIMENT_ID,
        "implementation_commit": implementation_commit,
        "status": "CONSUMED_PENDING_RERUN_FORBIDDEN",
    }
    try:
        handle = os.fdopen(descriptor, "r+b")
    except Exception as error:
        try:
            _write_descriptor_payload(
                descriptor, _invalid_receipt_payload(implementation_commit, error)
            )
        except Exception:
            pass
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise Rank1ReplayConsumedError(
            "one-shot path was created but receipt wrapping failed; rerun is forbidden"
        ) from error
    try:
        _write_receipt_payload(handle, marker)
    except Exception as error:
        try:
            _write_receipt_payload(
                handle, _invalid_receipt_payload(implementation_commit, error)
            )
        except Exception:
            pass
        finally:
            _safe_close(handle)
        raise Rank1ReplayConsumedError(
            "one-shot path was created but marker durability failed; rerun is forbidden"
        ) from error
    return handle


def _label_stat_identity(handle: BinaryIO) -> tuple[int, int, int]:
    stat = os.fstat(handle.fileno())
    return int(stat.st_size), int(stat.st_mtime_ns), int(getattr(stat, "st_ino", 0))


def _result_privacy_scan(result: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(result)
    if ASIN_SHAPE_RE.search(payload):
        raise Rank1ReplayError("result contains a product identifier")

    forbidden_keys = {
        "session_id",
        "sample_id",
        "target_id",
        "target_asin",
        "ground_truth",
        "per_session",
        "membership_vector",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden_keys & {str(key) for key in value}:
                raise Rank1ReplayError("result contains a forbidden identity key")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            if len(value) >= SESSION_COUNT:
                raise Rank1ReplayError("result contains a per-session vector")
            for child in value:
                walk(child)
        elif isinstance(value, np.ndarray):
            raise Rank1ReplayError("result contains an ndarray")

    walk(result)


def _safe_close(handle: BinaryIO | None) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except Exception:
        pass


def run(implementation_commit: str) -> dict[str, Any]:
    started = time.perf_counter()
    receipt: BinaryIO | None = None
    label_handle: BinaryIO | None = None
    consumed = False
    final_written = False
    try:
        environment = _validate_environment()
        preregistration = _validate_preregistration()
        git = _validate_git_checkpoint(implementation_commit)
        if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
            raise Rank1ReplayError("one-shot result already exists")

        target_free = _load_target_free_inputs()
        working_set_before, peak_before = _process_memory()
        if working_set_before <= 0 or peak_before <= 0:
            raise Rank1ReplayError("process memory measurement is unavailable")

        receipt = open_one_shot_receipt(OUTPUT_PATH, ROOT, implementation_commit)
        consumed = True

        if not LABEL_PATH.is_file() or LABEL_PATH.is_symlink():
            raise Rank1ReplayError("sealed label archive is unavailable")
        label_handle = LABEL_PATH.open("rb")
        label_identity_start = _label_stat_identity(label_handle)
        label_start_sha, label_start_bytes = _sha256_handle(label_handle)
        if label_start_sha != LABEL_SHA256 or label_start_bytes != LABEL_BYTES:
            raise Rank1ReplayError("sealed label archive hash drifted")
        label_handle.seek(0)
        outcomes = load_outcomes_from_open_handle(label_handle)

        baseline_session_hit = derive_baseline_session_hit(
            outcomes.baseline_rank, outcomes.eligible_from
        )
        labels = {
            "baseline_rank": outcomes.baseline_rank,
            "positive_index": outcomes.positive_index,
            "eligible_from": outcomes.eligible_from,
            "outer_fold": outcomes.outer_fold,
            "inner_fold": outcomes.inner_fold,
            "baseline_session_hit": baseline_session_hit,
        }
        surface = frozen._action_surface(
            target_free.projected_features,
            target_free.oof_scores,
            labels,
        )
        if not np.array_equal(surface.chosen, target_free.chosen):
            raise Rank1ReplayError("post-attach chosen surface drifted")
        activation, selections, comparator_reproduction = _reproduce_nested_activation(
            surface, labels, seed=BASE_SEED
        )
        activation_sha = hashlib.sha256(activation.tobytes()).hexdigest()
        if activation_sha != EXPECTED_ACTIVATION_SHA256:
            raise Rank1ReplayError("frozen activation identity drifted")

        first = replay_pages(
            target_free.traces,
            surface.chosen,
            activation,
            target_free.versions,
            measure_timing=True,
        )
        repeat = replay_pages(
            target_free.traces,
            surface.chosen,
            activation,
            target_free.versions,
        )
        first_identity_sha = _canonical_sha256(first.identity)
        repeat_identity_sha = _canonical_sha256(repeat.identity)
        exact_repeat = (
            first_identity_sha == repeat_identity_sha
            and first.identity == repeat.identity
            and first.structural == repeat.structural
        )

        baseline_state = state_from_positive_index(
            first.baseline_pages,
            outcomes.positive_index,
            outcomes.eligible_from,
        )
        candidate_state = state_from_positive_index(
            first.candidate_pages,
            outcomes.positive_index,
            outcomes.eligible_from,
        )
        all_sessions = np.ones(SESSION_COUNT, dtype=bool)
        aggregate = transition_metrics(
            baseline_state, candidate_state, first.changed, all_sessions
        )
        folds = [
            transition_metrics(
                baseline_state,
                candidate_state,
                first.changed,
                outcomes.outer_fold == fold,
            )
            for fold in range(OUTER_FOLDS)
        ]
        dominance = dominance_audit(baseline_state, candidate_state)
        trace_rank_mismatches = _trace_baseline_rank_mismatches(
            target_free.traces, outcomes
        )
        reset_eligibility_mismatches = int(
            np.sum(first.last_reset_turn != outcomes.eligible_from)
        )
        label_partition_ok = bool(
            int(baseline_session_hit.sum()) == 1_895
            and all(
                int(np.sum(outcomes.outer_fold == fold)) == 400
                and int(np.sum(outcomes.inner_fold == fold)) == 400
                for fold in range(OUTER_FOLDS)
            )
        )
        baseline_official = aggregate.baseline.official()
        baseline_identity_ok = all(
            baseline_official[name] == expected
            for name, expected in EXPECTED_BASELINE_OFFICIAL.items()
        )
        structural_ok = bool(
            exact_repeat
            and label_partition_ok
            and baseline_identity_ok
            and trace_rank_mismatches == 0
            and reset_eligibility_mismatches == 0
            and not any(dominance.values())
            and first.structural["ranks_2_to_10_byte_identical"] is True
            and first.structural["removed_rank1_already_served"] is True
            and first.structural["inserted_from_unseen_c100_tail"] is True
            and first.structural["reset_pages_identity"] is True
        )
        promote = passes_promotion_gates(
            aggregate, folds, structural_ok, exact_repeat
        )
        if not structural_ok:
            status = "INVALID_IMPLEMENTATION_NO_ALGORITHM_CONCLUSION"
        elif promote:
            status = "GO_RUNTIME_INTEGRATION"
        else:
            status = "NO_GO_CLOSE_RANK1_SEEN_REPLACEMENT"

        label_end_sha, label_end_bytes = _sha256_handle(label_handle)
        label_identity_end = _label_stat_identity(label_handle)
        if not (
            label_end_sha == label_start_sha
            and label_end_bytes == label_start_bytes
            and label_identity_end == label_identity_start
        ):
            raise Rank1ReplayError("sealed label archive changed during attach")
        label_handle.close()
        label_handle = None
        if not _validate_source_snapshots(target_free.source_snapshots):
            raise Rank1ReplayError("target-free input changed during run")

        working_set_after, peak_after = _process_memory()
        wall_seconds = time.perf_counter() - started
        if working_set_after <= 0 or peak_after <= 0:
            raise Rank1ReplayError("final process memory measurement is unavailable")
        if wall_seconds > 120:
            raise Rank1ReplayError("cached replay exceeded wall-time gate")

        fold_reports = [
            {"fold": fold, **transition.report()}
            for fold, transition in enumerate(folds)
        ]
        decision_payload = {
            "status": status,
            "promote_to_default_off_runtime_patch": promote,
            "structural_integrity": structural_ok,
            "promotion_gates_pass": promote,
        }
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "environment": environment,
            "git": git,
            "preregistration": preregistration,
            "sources": {
                "projected_features": {
                    "path": str(PROJECTED_FEATURES_PATH),
                    "bytes": PROJECTED_FEATURES_BYTES,
                    "sha256": PROJECTED_FEATURES_SHA256,
                },
                "oof_scores": {
                    "path": str(OOF_SCORES_PATH),
                    "bytes": OOF_SCORES_BYTES,
                    "sha256": OOF_SCORES_SHA256,
                },
                "blind_trace_aggregate_sha256": (
                    "11ad3e24aec412f6cb3b146d248aa7e2335a12dafccc20241eeb3301af97ca24"
                ),
                "combined_blind_trace_sha256": (
                    "f9a441220926aebf49f4b4d54a0f50df99f72ad4f8c0342e5528517503473e7b"
                ),
                "chosen_raw_sha256": EXPECTED_CHOSEN_SHA256,
                "activation_raw_sha256": activation_sha,
                "frozen_v1_9_activation_turns": int(activation.sum()),
                "frozen_v1_9_activation_sessions": int(
                    np.any(activation, axis=1).sum()
                ),
            },
            "target_free_reset": dict(target_free.reset_audit),
            "replay": {
                "policy": "RANK1_SEEN_REPLACEMENT",
                "identity": dict(first.identity),
                "identity_sha256": first_identity_sha,
                "structural": dict(first.structural),
            },
            "exact_repeat": {
                "equal": exact_repeat,
                "first_identity_sha256": first_identity_sha,
                "repeat_identity_sha256": repeat_identity_sha,
            },
            "label_attach": {
                "archive_open_count": 1,
                "member_access_count": len(LABEL_MEMBER_SPECS),
                "member_access_order": [
                    name for name, _shape, _dtype in LABEL_MEMBER_SPECS
                ],
                "archive_sha256": label_start_sha,
                "archive_bytes": label_start_bytes,
                "same_handle_start_end": True,
                "baseline_session_hit_derived": True,
                "baseline_hit_count": int(baseline_session_hit.sum()),
                "trace_baseline_rank_mismatches": trace_rank_mismatches,
                "last_reset_eligibility_mismatches": reset_eligibility_mismatches,
                "partition_valid": label_partition_ok,
            },
            "dominance_audit": dominance,
            "comparison": {
                "global": aggregate.report(),
                "folds": fold_reports,
            },
            "candidate_recall_frozen_context": {
                "not_recomputed_in_this_outcome": True,
                "C10": 0.9475,
                "C20": 0.9715,
                "C50": 0.991,
                "C100": 0.993,
                "C200": "unavailable_in_frozen_blind_trace",
            },
            "outer_selections": selections,
            "frozen_comparator_reproduction": comparator_reproduction,
            "privacy": {
                "split": "train_explore",
                "receipt_preceded_label_open_and_hash": True,
                "target_runtime_features": 0,
                "proxy_opened": False,
                "legacy_eligible_turn_helper_called": False,
                "per_session_outcomes_serialized": False,
                "product_identifiers_serialized": False,
                "agent_or_full_evaluator_started": False,
                "held_out_split_opened": False,
            },
            "resource": {
                "wall_seconds": round(wall_seconds, 6),
                "working_set_before_receipt_bytes": working_set_before,
                "peak_before_receipt_bytes": peak_before,
                "working_set_final_bytes": working_set_after,
                "peak_final_bytes": peak_after,
                "policy_transform_latency": dict(first.timing),
                "workers": 1,
            },
            "decision": {
                **decision_payload,
                "strict_decision_sha256": _canonical_sha256(decision_payload),
                "served_default": "off",
                "fallback": "complete frozen v1.9, P11, and R08 paths",
                "project_goal_hr_at_10_report_only": 0.99,
            },
        }
        result["receipt"] = {
            "path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "durable": True,
            "self_hash_omitted": True,
        }
        _result_privacy_scan(result)
        _write_receipt_payload(receipt, result)
        final_written = True
        _safe_close(receipt)
        receipt = None
        return result
    except Exception as error:
        _safe_close(label_handle)
        label_handle = None
        if consumed and receipt is not None and not final_written:
            invalid = _invalid_receipt_payload(implementation_commit, error)
            try:
                _write_receipt_payload(receipt, invalid)
            except Exception:
                pass
            finally:
                _safe_close(receipt)
                receipt = None
            raise Rank1ReplayConsumedError(
                "v2.12 one-shot was consumed; inspect the durable invalid receipt"
            ) from error
        raise
    finally:
        _safe_close(label_handle)
        _safe_close(receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation-commit",
        required=True,
        help="Exact clean pushed implementation commit to bind before label access.",
    )
    args = parser.parse_args()
    result = run(str(args.implementation_commit))
    print(
        json.dumps(
            {
                "status": result["status"],
                "global": result["comparison"]["global"],
                "folds": result["comparison"]["folds"],
                "dominance_audit": result["dominance_audit"],
                "structural": result["replay"]["structural"],
                "resource": result["resource"],
                "decision": result["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
