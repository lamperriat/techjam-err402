"""One-shot Stage-2 outcome attachment for the frozen v2.9 top-5 policy.

The module deliberately has no import-time filesystem activity.  Production
execution is allowed only from the clean, pushed implementation checkpoint
whose parent is the frozen protocol checkpoint.  Before the one permitted
label archive open, every policy, proposal, source, Git, and physical-input
identity is validated and the complete target-free comparator surface is
constructed.  Held outcomes are never supplied to a fit or selection step.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, BinaryIO, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/small_ranker_v2_9.top5_proposal_depth_stage2_outcome_protocol.json"
PROTOCOL_SHA256 = "dcbcfc7b04b1b5700f9913e24dcdfdfad575fb638ffb3bf70184dfc1c7a40ba9"
PROTOCOL_COMMIT = "a92fc83ed37b4697ae492d958e3ebd852b4d08fb"
STAGE1B_EVIDENCE_COMMIT = "85b378186fa3fefb2b34ae38ac1437e873c2b7c7"
SCHEMA_VERSION = "small-ranker-top5-proposal-depth-stage2-outcome.v1"

SESSION_COUNT = 2_000
TURN_COUNT = 10
CANDIDATE_COUNT = 100
FEATURE_COUNT = 133
GATE_FEATURE_COUNT = 26
OUTER_FOLDS = 5
BASE_SEED = 40_220_260_830
KEEP_QUANTILE = 1.0
QUANTILES = tuple(float(value) / 64.0 for value in range(64))

EXPECTED_GLOBAL_CHOSEN_RAW_SHA256 = (
    "229952c9ced7f6eec1ff1938480adc85ba5093ad865336465749029576e47051"
)
EXPECTED_GLOBAL_ACTIVATION_RAW_SHA256 = (
    "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
)
EXPECTED_GLOBAL_QUANTILES = (0.390625, 0.6875, 0.40625, 0.859375, 0.5)
EXPECTED_LABEL_SHA256 = "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb"
EXPECTED_LABEL_BYTES = 1_702_876
LABEL_PATH = Path(
    r"D:\tiktok\techjam-err402-fast-track\experiments\fast_track\small_ranker_v1\labels_v2.npz"
)
PROJECTED_FEATURES_PATH = Path(
    r"D:\tiktok\techjam-v1-2-metric-gate\experiments\fast_track\small_ranker_fold_safe_projected_features.npy"
)
PROJECTED_FEATURES_SHA256 = "cd9b075b923c31afe10b9a9c6d720de22e221c93de961595ff484ea4f532b90a"
OOF_SCORES_PATH = Path(
    r"D:\tiktok\techjam-err402-fast-track\experiments\fast_track\small_ranker_v1\oof_batch_v1\oof_scores_runtime_projection_no_semantic.npy"
)
OOF_SCORES_SHA256 = "5000deb9b77b3e7b326ccab6455222b291d2ec859ddab2043fe67d23a3217c5e"
OUTPUT_PATH = ROOT / (
    "experiments/fast_track/small_ranker_v2_9/"
    "stage2_one_shot_20260831/stage2_outcome_result.json"
)

V28_MANIFEST = ROOT / "configs/small_ranker_v2_8.strict_outer_restack_stage1.manifest.json"
V28_RESULT = ROOT / (
    "experiments/fast_track/small_ranker_v2_8/"
    "stage1_20260830T1834/stage1_cache_result.json"
)
STAGE1A_MANIFEST = ROOT / "configs/small_ranker_v2_9.top5_proposal_depth_stage1a.manifest.json"
STAGE1A_RESULT = ROOT / (
    "experiments/fast_track/small_ranker_v2_9/"
    "stage1a_20260830T225937/stage1a_result.json"
)
STAGE1B_MANIFEST = ROOT / "configs/small_ranker_v2_9.top5_proposal_depth_stage1b.manifest.json"
STAGE1B_RESULT = ROOT / (
    "experiments/fast_track/small_ranker_v2_9/"
    "stage1b_20260831T000352/stage1b_cache_result.json"
)

PINNED_REPO_FILES = {
    "protocol": (PROTOCOL, PROTOCOL_SHA256),
    "v28_manifest": (V28_MANIFEST, "c57c39d8220914dcab91717b9505f7734837aeca936178167f464ec33cbbcceb"),
    "v28_result": (V28_RESULT, "a683f50f6acdb2ee3cc0c88507e6f5ac4f46b2a6b0599acf2e6a4abdc3d17c97"),
    "stage1a_manifest": (STAGE1A_MANIFEST, "d85f75a2e09ee9ad7a39e12e5b5cf858acd5c134ce122978bcab40c5d4081704"),
    "stage1a_result": (STAGE1A_RESULT, "54577998a25dbf3054f2f393e837b0d49ea1fb5a87d95103d860595c97b067b1"),
    "stage1b_manifest": (STAGE1B_MANIFEST, "dfda329ad820d4e064591031c3b9e644648139cd99d194fc8239228fdd04bcf4"),
    "stage1b_result": (STAGE1B_RESULT, "04fc8551ccc90d56546c8525df943cc6a7fc9497f9bcda5351b08205f525f642"),
    "preregistration": (ROOT / "configs/small_ranker_v2_9.top5_proposal_depth_preregistration.json", "51c0a9d909e7e8d21604ff29981c8a35ca217b94e0ec9d6f8c98ca12d700cebb"),
    "legacy_reproducer": (ROOT / "scripts/analyze_small_ranker_remaining_misses.py", "c080b66e94757d1865596af262ba456611490864c5169f024a1ecb1bbe0fca79"),
    "stage1b_orchestrator": (ROOT / "scripts/build_small_ranker_top5_proposal_depth_stage1b.py", "090cc4c383b7287cbcf373126211444063bcee4126f7dc0106843bf8aa247586"),
    "stage1b_amendment": (ROOT / "configs/small_ranker_v2_9.top5_proposal_depth_stage1b_implementation_amendment.json", "5c1e6a6e78b56c3c22a8329daac941be0ab46bc5c84c989141e30a402fdf7d7c"),
    "selector": (ROOT / "scripts/small_ranker_portfolio_selector_py39.py", "35b7b68af7c52b7ecf0fe37ee686ed2e737ff2f6643622abd26dbc97e192cba8"),
    "metric": (ROOT / "scripts/analyze_small_ranker_metric_gate.py", "8c0cbffa6cd3dc62ddee3bb386c16bd60592a6324ecf6fcf4bcd4cf37951ca83"),
    "frozen": (ROOT / "scripts/export_small_ranker_fold_safe_artifact.py", "5115026c53b21d4d5930cb9af7783c0988b049a0e259f5a0a588901ad44f5e8b"),
    "base": (ROOT / "scripts/train_small_ranker.py", "db7f4a3e19da118abb7d37fc1530babd6928894e51e85010b11d9dcdc1d7e583"),
    "rr": (ROOT / "scripts/analyze_small_ranker_rr_regret_gate.py", "793e3615df38cd995f55e57decaeea35b549e40ad50ee3bf8a6dbf1055ca7e80"),
    "runtime_projection_repeat": (ROOT / "configs/small_ranker_v1_7.runtime_projection_repeat.manifest.json", "dcedd70bc8d905342e805c84b08ad3f375a28774fd44414b9a31025e343ceada"),
    "remaining_miss": (ROOT / "configs/small_ranker_v2_0.remaining_miss_attribution.manifest.json", "a0f7041b9deea488e2a2b527308449146f1193fbbf055c748bdea72224add089"),
    "deployable": (ROOT / "configs/small_ranker_v1_9.deployable_artifact.manifest.json", "0bd44c26b47b999e26fda7a7b6ee8f003b43c02dbe53b9328fbd75507eb7e574"),
}

LABEL_MEMBER_SPECS = (
    ("baseline_rank", (SESSION_COUNT, TURN_COUNT), "uint8"),
    ("positive_index", (SESSION_COUNT, TURN_COUNT), "int16"),
    ("eligible_from", (SESSION_COUNT,), "uint8"),
    ("outer_fold", (SESSION_COUNT,), "uint8"),
    ("inner_fold", (SESSION_COUNT,), "uint8"),
)
ARRAY_RECORD_KEYS = {
    "array_sha256", "asin_shape_matches", "bytes", "dtype", "path", "sha256", "shape"
}
ASIN_SHAPE_RE = re.compile(rb"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE)


class Stage2Error(RuntimeError):
    """Pre-receipt mechanical failure; no outcome has been consumed."""


class Stage2ConsumedError(RuntimeError):
    """Failure after the irreversible one-shot receipt was created."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    sha256: str
    size: int
    stat_identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class RuntimeSurface:
    incumbent: np.ndarray
    chosen: np.ndarray
    action: np.ndarray
    gate_features: np.ndarray


@dataclass(frozen=True)
class OutcomeBundle:
    baseline_rank: np.ndarray
    positive_index: np.ndarray
    eligible_from: np.ndarray
    outer_fold: np.ndarray
    inner_fold: np.ndarray


@dataclass
class FitCounters:
    helper_invocations: int = 0
    liblinear_fit_calls: int = 0
    constant_gate_returns: int = 0


@dataclass(frozen=True)
class FrozenInputs:
    owner: np.ndarray
    held_orders: tuple[np.ndarray, ...]
    candidates: np.ndarray
    available: np.ndarray
    supplement: np.ndarray
    domain_chosen: np.ndarray
    domain_activation: np.ndarray
    final_chosen: np.ndarray
    final_activation: np.ndarray
    global_surface: RuntimeSurface
    stage1a_result: Mapping[str, Any]
    stage1b_result: Mapping[str, Any]


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + suffix
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_stream(handle: BinaryIO) -> tuple[str, int]:
    handle.seek(0)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)[0]


def _logical_array_sha256(value: np.ndarray) -> str:
    """Stage1a/Stage1b registry hash: dtype + JSON shape + C bytes."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _raw_array_sha256(value: np.ndarray) -> str:
    """Historical comparator identity: raw contiguous C-order bytes only."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return (int(value.st_dev), int(value.st_ino), int(value.st_size), int(value.st_mtime_ns))


def _snapshot_regular(
    path: Path, expected_sha256: str, expected_bytes: int | None = None
) -> FileSnapshot:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Stage2Error("pinned input is not a regular non-symlink file")
    resolved = path.resolve()
    size = int(resolved.stat().st_size)
    if expected_bytes is not None and size != int(expected_bytes):
        raise Stage2Error("pinned input byte count drifted")
    actual = _sha256(resolved)
    if actual != expected_sha256:
        raise Stage2Error("pinned input hash drifted")
    return FileSnapshot(resolved, actual, size, _stat_identity(resolved))


def _revalidate_snapshots(snapshots: Mapping[str, FileSnapshot]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in sorted(snapshots):
        start = snapshots[name]
        end = _snapshot_regular(start.path, start.sha256, start.size)
        if end.stat_identity != start.stat_identity:
            raise Stage2Error("pinned input stat identity drifted")
        rows.append({"name": name, "sha256": end.sha256, "bytes": end.size})
    return {
        "input_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "start_end_equal": True,
        "identity_sha256": _canonical_sha256(rows),
    }


def _identity_shape_scan(path: Path) -> int:
    count = 0
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            payload = overlap + chunk
            count += len(ASIN_SHAPE_RE.findall(payload))
            overlap = payload[-9:]
    return count


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage2Error("expected a JSON object")
    return value


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


def _require_process_memory(
    stage: str, reader: Any = _process_memory
) -> tuple[int, int]:
    working_set, peak = reader()
    if int(working_set) <= 0 or int(peak) <= 0:
        raise Stage2Error("working-set measurement is unavailable: %s" % stage)
    return int(working_set), int(peak)


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
        raise Stage2Error("Stage2 dependency identity mismatch")
    return actual


def _git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=%s" % ROOT.as_posix(), *args],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise Stage2Error("Stage2 Git checkpoint validation failed")
    return completed.stdout.strip()


def _changed_paths(commit: str) -> set[str]:
    return {
        line.strip().replace("\\", "/")
        for line in _git(("diff-tree", "--no-commit-id", "--name-only", "-r", commit)).splitlines()
        if line.strip()
    }


def _validate_git_checkpoint(implementation_commit: str) -> dict[str, Any]:
    if _git(("status", "--porcelain")):
        raise Stage2Error("Stage2 requires a clean tracked and untracked worktree")
    head = _git(("rev-parse", "HEAD"))
    upstream = _git(("rev-parse", "@{upstream}"))
    parent = _git(("rev-parse", "HEAD^"))
    protocol_parent = _git(("rev-parse", "%s^" % PROTOCOL_COMMIT))
    if not (
        head == implementation_commit
        and upstream == head
        and parent == PROTOCOL_COMMIT
        and protocol_parent == STAGE1B_EVIDENCE_COMMIT
        and _changed_paths(PROTOCOL_COMMIT)
        == {
            ".gitattributes",
            "configs/small_ranker_v2_9.top5_proposal_depth_stage2_outcome_protocol.json",
        }
        and _changed_paths(head)
        == {
            "scripts/evaluate_small_ranker_top5_proposal_depth_stage2.py",
            "tests/test_small_ranker_top5_proposal_depth_stage2.py",
        }
    ):
        raise Stage2Error("Stage2 commit choreography drifted")
    return {
        "implementation_commit": head,
        "protocol_commit": parent,
        "stage1b_evidence_commit": protocol_parent,
        "upstream_equal": True,
        "clean": True,
        "implementation_paths_exact": True,
        "protocol_paths_exact": True,
    }


def derive_baseline_session_hit(
    baseline_rank: np.ndarray, eligible_from: np.ndarray
) -> np.ndarray:
    baseline_rank = np.asarray(baseline_rank)
    eligible_from = np.asarray(eligible_from)
    if (
        baseline_rank.ndim != 2
        or baseline_rank.shape[1] != TURN_COUNT
        or eligible_from.shape != (baseline_rank.shape[0],)
        or np.any((baseline_rank < 0) | (baseline_rank > 10))
        or np.any((eligible_from < 1) | (eligible_from > TURN_COUNT))
    ):
        raise Stage2Error("baseline-hit derivation schema failed")
    eligible = np.arange(1, baseline_rank.shape[1] + 1)[None, :] >= eligible_from[:, None]
    return np.any(eligible & (baseline_rank > 0), axis=1).astype(np.uint8)


def policy_session_state(
    labels: Mapping[str, np.ndarray], chosen: np.ndarray, activation: np.ndarray
) -> dict[str, np.ndarray]:
    baseline_rank = np.asarray(labels["baseline_rank"])
    positive = np.asarray(labels["positive_index"])
    eligible_from = np.asarray(labels["eligible_from"])
    chosen = np.asarray(chosen)
    activation = np.asarray(activation, dtype=bool)
    if not (
        baseline_rank.shape == positive.shape == chosen.shape == activation.shape
        and eligible_from.shape == (baseline_rank.shape[0],)
    ):
        raise Stage2Error("policy-state schema failed")
    eligible = np.arange(1, baseline_rank.shape[1] + 1)[None, :] >= eligible_from[:, None]
    protected = (baseline_rank >= 1) & (baseline_rank <= 9)
    rank = np.where(
        protected,
        baseline_rank,
        np.where(
            ~activation,
            baseline_rank,
            np.where((positive >= 0) & (chosen == positive), 10, 0),
        ),
    )
    rank = np.where(eligible, rank, 0)
    hit_turn = rank > 0
    hit = hit_turn.any(axis=1)
    first_index = np.argmax(hit_turn, axis=1)
    first_turn = np.where(hit, first_index + 1, 11).astype(np.int16)
    first_rank = np.take_along_axis(rank, first_index[:, None], axis=1)[:, 0].astype(np.int16)
    return {"hit": hit, "first_rank": first_rank, "first_turn": first_turn}


def _fraction_payload(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": int(value.numerator),
        "denominator": int(value.denominator),
        "decimal": float(value),
    }


def exact_metrics(state: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    hit = np.asarray(state["hit"], dtype=bool)[selected]
    rank = np.asarray(state["first_rank"], dtype=np.int64)[selected]
    turn = np.asarray(state["first_turn"], dtype=np.int64)[selected]
    count = int(hit.size)
    if count <= 0:
        raise Stage2Error("metric mask is empty")
    hits = int(hit.sum())
    hr = Fraction(hits, count)
    reciprocal_sum = sum(
        (Fraction(1, int(value)) for value in rank[hit]), start=Fraction(0, 1)
    )
    mrr = reciprocal_sum / count
    mttc = Fraction(sum(int(value) for value in turn), count)
    efficiency = min(Fraction(1, 1), max(Fraction(0, 1), (Fraction(11, 1) - mttc) / 10))
    technical = Fraction(1, 2) * hr + Fraction(3, 10) * mrr + Fraction(1, 5) * efficiency
    hr_rounded = round(float(hr), 6)
    mrr_rounded = round(float(mrr), 6)
    mttc_rounded = round(float(mttc), 6)
    official_efficiency = max(0.0, min(1.0, (11.0 - mttc_rounded) / 10.0))
    official_technical = round(
        0.5 * hr_rounded + 0.3 * mrr_rounded + 0.2 * official_efficiency,
        6,
    )
    return {
        "sample_count": count,
        "hits": hits,
        "misses": count - hits,
        "exact": {
            "hit_rate_at_10": _fraction_payload(hr),
            "mrr": _fraction_payload(mrr),
            "mttc": _fraction_payload(mttc),
            "efficiency": _fraction_payload(efficiency),
            "technical_score": _fraction_payload(technical),
        },
        "rounded": {
            "hit_rate_at_10": hr_rounded,
            "mrr": mrr_rounded,
            "mttc": mttc_rounded,
            "efficiency": round(official_efficiency, 6),
            "technical_score": official_technical,
        },
    }


def _exact_value(metrics: Mapping[str, Any], name: str) -> Fraction:
    row = metrics["exact"][name]
    return Fraction(int(row["numerator"]), int(row["denominator"]))


def comparison_payload(
    baseline_state: Mapping[str, np.ndarray],
    policy_state: Mapping[str, np.ndarray],
    activation: np.ndarray,
    owner: np.ndarray,
) -> dict[str, Any]:
    def one(mask: np.ndarray, fold: int | None) -> dict[str, Any]:
        baseline = exact_metrics(baseline_state, mask)
        policy = exact_metrics(policy_state, mask)
        baseline_hit = np.asarray(baseline_state["hit"], dtype=bool)
        policy_hit = np.asarray(policy_state["hit"], dtype=bool)
        miss_to_hit = int(np.sum(mask & ~baseline_hit & policy_hit))
        hit_to_miss = int(np.sum(mask & baseline_hit & ~policy_hit))
        exact_delta = {
            name: _fraction_payload(_exact_value(policy, name) - _exact_value(baseline, name))
            for name in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "technical_score")
        }
        rounded_delta = {
            name: round(float(policy["rounded"][name]) - float(baseline["rounded"][name]), 6)
            for name in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "technical_score")
        }
        return {
            "fold": fold,
            "miss_to_hit": miss_to_hit,
            "hit_to_miss": hit_to_miss,
            "net_hits": miss_to_hit - hit_to_miss,
            "activation_sessions": int(np.any(np.asarray(activation)[mask], axis=1).sum()),
            "activation_turns": int(np.asarray(activation)[mask].sum()),
            "baseline": baseline,
            "policy": policy,
            "exact_delta": exact_delta,
            "rounded_delta": rounded_delta,
        }

    aggregate_mask = np.ones(SESSION_COUNT, dtype=bool)
    return {
        "aggregate": one(aggregate_mask, None),
        "folds": [one(np.asarray(owner) == fold, fold) for fold in range(OUTER_FOLDS)],
    }


def _comparison_gate(payload: Mapping[str, Any], *, secondary: bool) -> dict[str, Any]:
    aggregate = payload["aggregate"]
    folds = payload["folds"]

    def safe(row: Mapping[str, Any]) -> bool:
        return bool(
            int(row["hit_to_miss"]) == 0
            and int(row["net_hits"]) >= 0
            and Fraction(
                int(row["exact_delta"]["mrr"]["numerator"]),
                int(row["exact_delta"]["mrr"]["denominator"]),
            ) >= 0
            and Fraction(
                int(row["exact_delta"]["mttc"]["numerator"]),
                int(row["exact_delta"]["mttc"]["denominator"]),
            ) <= 0
            and float(row["rounded_delta"]["mrr"]) >= 0.0
            and float(row["rounded_delta"]["mttc"]) <= 0.0
        )

    checks = {
        "aggregate_miss_to_hit_positive": int(aggregate["miss_to_hit"]) > 0,
        "aggregate_hit_to_miss_zero": int(aggregate["hit_to_miss"]) == 0,
        "aggregate_rank_and_turn_safe": safe(aggregate),
        "all_outer_folds_safe": all(safe(row) for row in folds),
        "official_technical_score_delta_positive": float(
            aggregate["rounded_delta"]["technical_score"]
        ) > 0.0,
    }
    if secondary:
        policy = aggregate["policy"]
        checks.update(
            {
                "policy_hits_strictly_above_1943": int(policy["hits"]) > 1943,
                "official_hr_strictly_above_0_9715": float(
                    policy["rounded"]["hit_rate_at_10"]
                ) > 0.9715,
                "exact_hr_strictly_above_0_9715": _exact_value(
                    policy, "hit_rate_at_10"
                ) > Fraction(9715, 10000),
            }
        )
    return {"checks": checks, "pass": all(checks.values())}


def _derive_training_targets(
    runtime: RuntimeSurface, labels: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive = np.asarray(labels["positive_index"])
    baseline_rank = np.asarray(labels["baseline_rank"])
    eligible_from = np.asarray(labels["eligible_from"])
    baseline_hit = np.asarray(labels["baseline_session_hit"])
    eligible = np.arange(1, TURN_COUNT + 1)[None, :] >= eligible_from[:, None]
    rescue = (
        runtime.action
        & eligible
        & (baseline_hit[:, None] == 0)
        & (runtime.chosen == positive)
        & (positive >= 0)
    )
    direct_risk = (
        runtime.action
        & eligible
        & (baseline_hit[:, None] == 1)
        & (baseline_rank == 10)
        & (runtime.chosen != positive)
    )
    rescue_weights = np.where(rescue, 1.0, np.where(direct_risk, 5.0, 0.05)).astype(
        np.float64
    )
    zero = np.zeros_like(runtime.action, dtype=bool)
    baseline_state = policy_session_state(labels, runtime.chosen, zero)
    baseline_rr = np.where(
        baseline_state["hit"], 1.0 / np.maximum(baseline_state["first_rank"], 1), 0.0
    )
    rr_loss = np.zeros_like(runtime.chosen, dtype=np.float32)
    for turn in range(TURN_COUNT):
        isolated = np.zeros_like(runtime.action, dtype=bool)
        isolated[:, turn] = runtime.action[:, turn]
        state = policy_session_state(labels, runtime.chosen, isolated)
        policy_rr = np.where(state["hit"], 1.0 / np.maximum(state["first_rank"], 1), 0.0)
        rr_loss[:, turn] = np.maximum(0.0, baseline_rr - policy_rr)
    regret = (rr_loss > 0).astype(np.uint8)
    regret_weights = np.where(
        regret > 0,
        5.0 + 20.0 * rr_loss,
        np.where(rescue, 0.2, 0.05),
    ).astype(np.float64)
    return rescue.astype(np.uint8), rescue_weights, regret, regret_weights


def _fit_head(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    seed: int,
    counters: FitCounters,
    base_module: Any,
) -> tuple[Any, np.ndarray, np.ndarray]:
    counters.helper_invocations += 1
    model, mean, scale = base_module._fit_gate_model(x, y, weights, seed)
    if isinstance(model, base_module._ConstantGate):
        counters.constant_gate_returns += 1
    else:
        counters.liblinear_fit_calls += 1
    return model, mean, scale


def _fit_seed(head: int, outer_fold: int, inner_fold: int | None = None) -> int:
    if head not in (0, 1) or outer_fold not in range(OUTER_FOLDS):
        raise Stage2Error("fit seed coordinate is invalid")
    if inner_fold is None:
        return BASE_SEED + head * 10_000 + outer_fold * 101
    if inner_fold not in range(OUTER_FOLDS):
        raise Stage2Error("fit seed inner coordinate is invalid")
    return BASE_SEED + head * 10_000 + outer_fold * 31 + inner_fold


def _threshold_at_quantile(values: np.ndarray, quantile: float) -> float:
    if float(quantile) >= KEEP_QUANTILE:
        return math.inf
    if not len(values):
        raise Stage2Error("cannot map a quantile over an empty action set")
    return float(np.quantile(values, float(quantile), method="higher"))


def _quantile_choice_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["technical_score_delta"]),
        int(row["net_hits"]),
        sum(int(value > 0) for value in row["inner_fold_net_hits"]),
        -int(row["activation_turns"]),
        float(row["quantile"]),
    )


def _rounded_transition(
    baseline_state: Mapping[str, np.ndarray],
    policy_state: Mapping[str, np.ndarray],
    activation: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    baseline_hit = np.asarray(baseline_state["hit"], dtype=bool)
    policy_hit = np.asarray(policy_state["hit"], dtype=bool)
    baseline = _legacy_rounded_metrics(baseline_state, mask)
    policy = _legacy_rounded_metrics(policy_state, mask)
    miss_to_hit = int(np.sum(mask & ~baseline_hit & policy_hit))
    hit_to_miss = int(np.sum(mask & baseline_hit & ~policy_hit))
    return {
        "miss_to_hit": miss_to_hit,
        "hit_to_miss": hit_to_miss,
        "net_hits": miss_to_hit - hit_to_miss,
        "activation_turns": int(np.asarray(activation)[mask].sum()),
        "mrr_delta": round(
            float(policy["mrr"]) - float(baseline["mrr"]), 6
        ),
        "mttc_delta": round(
            float(policy["mttc"]) - float(baseline["mttc"]), 6
        ),
        "technical_score_delta": round(
            float(policy["technical_score"])
            - float(baseline["technical_score"]),
            6,
        ),
    }


def _legacy_rounded_metrics(
    state: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, float]:
    """Pinned historical helper semantics used only for quantile reproduction.

    The legacy comparator selected its quantile with the old metric helper,
    whose TechnicalScore was computed from raw aggregates.  Final Stage2
    reporting and promotion use :func:`exact_metrics`, which follows the
    official evaluator's aggregate-rounding order.
    """
    selected = np.asarray(mask, dtype=bool)
    hit = np.asarray(state["hit"], dtype=bool)[selected]
    rank = np.asarray(state["first_rank"], dtype=np.int64)[selected]
    turn = np.asarray(state["first_turn"], dtype=np.int64)[selected]
    if not len(hit):
        raise Stage2Error("legacy metric mask is empty")
    hr = float(hit.mean())
    mrr = float(np.where(hit, 1.0 / np.maximum(rank, 1), 0.0).mean())
    mttc = float(np.where(hit, turn, 11).mean())
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "hit_rate_at_10": round(hr, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(0.5 * hr + 0.3 * mrr + 0.2 * efficiency, 6),
    }


def _select_inner_quantile_t(
    utility: np.ndarray,
    runtime: RuntimeSurface,
    labels: Mapping[str, np.ndarray],
    inner_fold: np.ndarray,
) -> dict[str, Any]:
    values = utility[runtime.action]
    if not len(values):
        raise Stage2Error("nested quantile action set is empty")
    zero = np.zeros_like(runtime.action, dtype=bool)
    baseline_state = policy_session_state(labels, runtime.chosen, zero)
    all_sessions = np.ones(runtime.action.shape[0], dtype=bool)
    candidates: list[dict[str, Any]] = []
    for quantile in (*QUANTILES, KEEP_QUANTILE):
        threshold = _threshold_at_quantile(values, quantile)
        activation = runtime.action & (utility >= threshold)
        policy_state = policy_session_state(labels, runtime.chosen, activation)
        aggregate = _rounded_transition(
            baseline_state, policy_state, activation, all_sessions
        )
        folds = [
            _rounded_transition(
                baseline_state,
                policy_state,
                activation,
                np.asarray(inner_fold) == fold,
            )
            for fold in range(OUTER_FOLDS)
        ]
        if all(
            row["hit_to_miss"] == 0
            and row["mrr_delta"] >= 0.0
            and row["mttc_delta"] <= 0.0
            for row in (aggregate, *folds)
        ):
            candidates.append(
                {
                    "quantile": float(quantile),
                    "inner_threshold": (
                        float(threshold) if np.isfinite(threshold) else "KEEP"
                    ),
                    **aggregate,
                    "inner_fold_net_hits": [int(row["net_hits"]) for row in folds],
                    "inner_fold_mrr_delta": [float(row["mrr_delta"]) for row in folds],
                    "inner_fold_mttc_delta": [float(row["mttc_delta"]) for row in folds],
                }
            )
    if not candidates:
        raise Stage2Error("KEEP unexpectedly failed inner-fold safety")
    return max(candidates, key=_quantile_choice_key)


def _slice_runtime(surface: RuntimeSurface, rows: np.ndarray) -> RuntimeSurface:
    return RuntimeSurface(
        np.asarray(surface.incumbent)[rows],
        np.asarray(surface.chosen)[rows],
        np.asarray(surface.action)[rows],
        np.asarray(surface.gate_features)[rows],
    )


def _training_labels(outcomes: OutcomeBundle, rows: np.ndarray) -> dict[str, np.ndarray]:
    baseline_rank = np.asarray(outcomes.baseline_rank)[rows].copy()
    eligible_from = np.asarray(outcomes.eligible_from)[rows].copy()
    labels = {
        "baseline_rank": baseline_rank,
        "positive_index": np.asarray(outcomes.positive_index)[rows].copy(),
        "eligible_from": eligible_from,
        "inner_fold": np.asarray(outcomes.inner_fold)[rows].copy(),
        "baseline_session_hit": derive_baseline_session_hit(
            baseline_rank, eligible_from
        ),
    }
    for value in labels.values():
        value.setflags(write=False)
    return labels


def reproduce_frozen_global(
    surface: RuntimeSurface,
    outcomes: OutcomeBundle,
    owner: np.ndarray,
    base_module: Any,
) -> tuple[np.ndarray, list[dict[str, Any]], FitCounters]:
    """Reproduce the fixed legacy comparator with T-only target derivation."""
    owner = np.asarray(owner)
    activation = np.zeros((SESSION_COUNT, TURN_COUNT), dtype=bool)
    coverage = np.zeros(SESSION_COUNT, dtype=np.uint8)
    selections: list[dict[str, Any]] = []
    counters = FitCounters()

    for outer_fold in range(OUTER_FOLDS):
        held_rows = np.flatnonzero(owner == outer_fold)
        train_rows = np.flatnonzero(owner != outer_fold)
        if (
            held_rows.shape != (400,)
            or train_rows.shape != (1600,)
            or not np.array_equal(held_rows, np.sort(held_rows))
            or not np.array_equal(train_rows, np.sort(train_rows))
        ):
            raise Stage2Error("row-isolated outer partition failed")
        runtime_t = _slice_runtime(surface, train_rows)
        runtime_h = _slice_runtime(surface, held_rows)
        labels_t = _training_labels(outcomes, train_rows)
        inner_t = labels_t["inner_fold"]
        rescue, rescue_weights, regret, regret_weights = _derive_training_targets(
            runtime_t, labels_t
        )
        flat_x = runtime_t.gate_features.reshape(-1, GATE_FEATURE_COUNT)
        flat_action = runtime_t.action.reshape(-1)
        flat_session = np.repeat(np.arange(1600), TURN_COUNT)
        targets = (rescue.reshape(-1), regret.reshape(-1))
        weights = (rescue_weights.reshape(-1), regret_weights.reshape(-1))
        inner_probability = [
            np.zeros((1600, TURN_COUNT), dtype=np.float32) for _ in range(2)
        ]
        for inner_index in range(OUTER_FOLDS):
            model_train = inner_t != inner_index
            model_valid = inner_t == inner_index
            fit_rows = flat_action & model_train[flat_session]
            predict_rows = flat_action & model_valid[flat_session]
            if not np.any(fit_rows) or not np.any(predict_rows):
                raise Stage2Error("nested comparator partition is empty")
            for head in range(2):
                model, mean, scale = _fit_head(
                    flat_x[fit_rows],
                    targets[head][fit_rows],
                    weights[head][fit_rows],
                    _fit_seed(head, outer_fold, inner_index),
                    counters,
                    base_module,
                )
                inner_probability[head].reshape(-1)[predict_rows] = (
                    base_module._predict_gate(
                        model, mean, scale, flat_x[predict_rows]
                    ).astype(np.float32)
                )
        selected = _select_inner_quantile_t(
            inner_probability[0] - inner_probability[1],
            runtime_t,
            labels_t,
            inner_t,
        )

        full_train_rows = flat_action
        held_flat_x = runtime_h.gate_features.reshape(-1, GATE_FEATURE_COUNT)
        held_action = runtime_h.action.reshape(-1)
        train_probability = [
            np.zeros((1600, TURN_COUNT), dtype=np.float32) for _ in range(2)
        ]
        held_probability = [
            np.zeros((400, TURN_COUNT), dtype=np.float32) for _ in range(2)
        ]
        for head in range(2):
            model, mean, scale = _fit_head(
                flat_x[full_train_rows],
                targets[head][full_train_rows],
                weights[head][full_train_rows],
                _fit_seed(head, outer_fold),
                counters,
                base_module,
            )
            train_probability[head].reshape(-1)[full_train_rows] = (
                base_module._predict_gate(
                    model, mean, scale, flat_x[full_train_rows]
                ).astype(np.float32)
            )
            held_probability[head].reshape(-1)[held_action] = (
                base_module._predict_gate(
                    model, mean, scale, held_flat_x[held_action]
                ).astype(np.float32)
            )
        train_utility = train_probability[0] - train_probability[1]
        threshold = _threshold_at_quantile(
            train_utility[runtime_t.action], float(selected["quantile"])
        )
        held_utility = held_probability[0] - held_probability[1]
        held_activation = runtime_h.action & (held_utility >= threshold)
        activation[held_rows] = held_activation
        coverage[held_rows] += 1
        selections.append(
            {
                "fold": outer_fold,
                "selected_quantile": float(selected["quantile"]),
                "mapped_outer_train_threshold": (
                    float(threshold) if np.isfinite(threshold) else "KEEP"
                ),
                "inner_selection": selected,
                "train_rescue_rows": int(rescue.sum()),
                "train_rr_regret_rows": int(regret.sum()),
            }
        )

    if not np.all(coverage == 1):
        raise Stage2Error("global comparator held coverage failed")
    if not (
        counters.helper_invocations == 60
        and counters.liblinear_fit_calls == 60
        and counters.constant_gate_returns == 0
    ):
        raise Stage2Error("global comparator fit topology drifted")
    return activation, selections, counters


def _build_global_runtime(features: np.ndarray, scores: np.ndarray, base_module: Any) -> RuntimeSurface:
    if not (
        features.shape == (SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT, FEATURE_COUNT)
        and str(features.dtype) == "float32"
        and scores.shape == (SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT)
        and str(scores.dtype) == "float32"
        and np.isfinite(features).all()
        and np.isfinite(scores).all()
    ):
        raise Stage2Error("global comparator target-free tensor schema failed")
    incumbent = base_module._incumbent_indices(features)
    chosen, margin, top_gap = base_module.choose_slot10(scores, incumbent)
    gate_features = base_module.gate_feature_matrix(
        features, scores, chosen, incumbent, margin, top_gap
    )
    surface = RuntimeSurface(incumbent, chosen, chosen != incumbent, gate_features)
    if not (
        surface.gate_features.shape
        == (SESSION_COUNT, TURN_COUNT, GATE_FEATURE_COUNT)
        and str(surface.gate_features.dtype) == "float32"
        and np.isfinite(surface.gate_features).all()
        and _raw_array_sha256(surface.chosen) == EXPECTED_GLOBAL_CHOSEN_RAW_SHA256
    ):
        raise Stage2Error("global comparator target-free decision identity failed")
    return surface


def _outer_map(rows: object, name: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(rows, list) or len(rows) != OUTER_FOLDS:
        raise Stage2Error("%s outer registry is incomplete" % name)
    result: dict[int, Mapping[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise Stage2Error("%s outer registry row is invalid" % name)
        fold = int(raw.get("outer_fold", -1))
        if fold not in range(OUTER_FOLDS) or fold in result:
            raise Stage2Error("%s outer registry is duplicated" % name)
        result[fold] = raw
    if set(result) != set(range(OUTER_FOLDS)):
        raise Stage2Error("%s outer registry coverage failed" % name)
    return result


def _file_identity(record: Mapping[str, Any]) -> dict[str, Any]:
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


def _resolve_registered_path(record: Mapping[str, Any], expected_relative: str) -> Path:
    if set(record) != ARRAY_RECORD_KEYS:
        raise Stage2Error("registered array record schema drifted")
    raw = str(record.get("path", "")).replace("\\", "/")
    if raw != expected_relative or Path(raw).is_absolute():
        raise Stage2Error("registered array path drifted")
    unresolved = ROOT / Path(raw)
    if unresolved.is_symlink():
        raise Stage2Error("registered array symlink is forbidden")
    path = unresolved.resolve()
    if ROOT.resolve() not in path.parents or path.suffix.lower() != ".npy":
        raise Stage2Error("registered array escapes its frozen root")
    return path


def _load_registered_array(
    record: Mapping[str, Any],
    expected_relative: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    audit_name: str,
    snapshots: MutableMapping[str, FileSnapshot],
) -> np.ndarray:
    path = _resolve_registered_path(record, expected_relative)
    snapshot = _snapshot_regular(
        path, str(record["sha256"]), int(record["bytes"])
    )
    if int(record["asin_shape_matches"]) != 0 or _identity_shape_scan(path) != 0:
        raise Stage2Error("registered numeric array contains an identity-shaped value")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if not (
            tuple(int(item) for item in record["shape"]) == expected_shape
            and str(record["dtype"]) == expected_dtype
            and value.shape == expected_shape
            and str(value.dtype) == expected_dtype
            and _logical_array_sha256(value) == str(record["array_sha256"])
            and (
                not np.issubdtype(value.dtype, np.floating)
                or bool(np.isfinite(value).all())
            )
        ):
            raise Stage2Error("registered numeric array identity failed")
        copied = np.asarray(value).copy()
    finally:
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()
    copied.setflags(write=False)
    snapshots[audit_name] = snapshot
    return copied


def _expected_registered_array_paths() -> dict[str, Path]:
    expected: dict[str, Path] = {}
    for fold in range(OUTER_FOLDS):
        expected["array:v28:outer:%d:session_ordinal" % fold] = ROOT / (
            "experiments/fast_track/small_ranker_v2_8/stage1_20260830T1834/"
            "first/outer_%d/held/session_ordinal.npy" % fold
        )
        for field in (
            "candidates",
            "available",
            "current_chosen",
            "current_activation",
        ):
            expected["array:stage1a:outer:%d:%s" % (fold, field)] = ROOT / (
                "experiments/fast_track/small_ranker_v2_9/"
                "stage1a_20260830T225937/first/outer_%d/held_H/%s.npy"
                % (fold, field)
            )
        expected["array:stage1b:outer:%d:supplement" % fold] = ROOT / (
            "experiments/fast_track/small_ranker_v2_9/"
            "stage1b_20260831T000352/first/outer_%d/supplement.npy" % fold
        )
    for field in (
        "domain_local_current_chosen",
        "domain_local_current_activation",
        "final_chosen",
        "final_activation",
    ):
        expected["array:policy:%s" % field] = ROOT / (
            "experiments/fast_track/small_ranker_v2_9/"
            "stage1b_20260831T000352/frozen/%s.npy" % field
        )
    return {name: path.resolve() for name, path in expected.items()}


def _validate_final_composition(
    domain_chosen: np.ndarray,
    domain_activation: np.ndarray,
    final_chosen: np.ndarray,
    final_activation: np.ndarray,
    supplement: np.ndarray,
    candidates: np.ndarray,
    available: np.ndarray,
) -> None:
    member = np.any(
        available & (candidates == final_chosen[..., None]), axis=2
    )
    if not (
        np.array_equal(final_activation, np.asarray(domain_activation) | supplement)
        and np.array_equal(final_chosen[~supplement], domain_chosen[~supplement])
        and np.all(member[supplement])
        and int(supplement.sum(axis=1).max(initial=0)) <= 1
    ):
        raise Stage2Error("frozen final policy is not the registered supplement composition")


def _validate_protocol_documents(
    snapshots: MutableMapping[str, FileSnapshot]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for name, (path, expected_hash) in PINNED_REPO_FILES.items():
        snapshots["repo:%s" % name] = _snapshot_regular(path, expected_hash)
    protocol = _load_json(PROTOCOL)
    v28 = _load_json(V28_RESULT)
    stage1a = _load_json(STAGE1A_RESULT)
    stage1b = _load_json(STAGE1B_RESULT)
    v28_manifest = _load_json(V28_MANIFEST)
    stage1a_manifest = _load_json(STAGE1A_MANIFEST)
    stage1b_manifest = _load_json(STAGE1B_MANIFEST)
    if not (
        protocol.get("schema_version")
        == "small-ranker-top5-proposal-depth-stage2-outcome-protocol.v1"
        and protocol.get("status")
        == "PROTOCOL_FROZEN_BEFORE_ANY_STAGE2_HELD_OUTCOME_ACCESS"
        and protocol.get("parent_evidence", {}).get("commit")
        == STAGE1B_EVIDENCE_COMMIT
        and protocol.get("execution_and_output", {}).get("output")
        == OUTPUT_PATH.relative_to(ROOT).as_posix()
        and protocol.get("label_and_outcome_access", {}).get("member_access_order")
        == [name for name, _shape, _dtype in LABEL_MEMBER_SPECS]
        and protocol.get("label_and_outcome_access", {}).get("archive_open_count") == 1
        and protocol.get("label_and_outcome_access", {}).get("member_access_count") == 5
        and v28.get("schema_version") == "small-ranker-strict-outer-restack-stage1-freeze.v1"
        and v28.get("status") == "CACHE_REPEAT_FROZEN"
        and v28.get("exact_repeat", {}).get("equal") is True
        and v28_manifest.get("schema_version")
        == "small-ranker-strict-outer-restack-stage1-manifest.v1"
        and v28_manifest.get("result", {}).get("sha256")
        == PINNED_REPO_FILES["v28_result"][1]
        and stage1a.get("schema_version")
        == "small-ranker-top5-proposal-depth-stage1a.v1"
        and stage1a.get("status") == "TARGET_FREE_ALL_OUTER_SURFACES_FROZEN"
        and stage1a.get("exact_repeat", {}).get("identity_sha256")
        == "b0abefdd23b4364c336e10035baa8a62b5dd1b4dd17f435b53244aec72b9c1da"
        and stage1a.get("privacy", {}).get("label_archive_opened") is False
        and stage1a_manifest.get("status") == "TARGET_FREE_ALL_OUTER_SURFACES_FROZEN"
        and stage1a_manifest.get("result", {}).get("sha256")
        == PINNED_REPO_FILES["stage1a_result"][1]
        and stage1b.get("schema_version")
        == "small-ranker-top5-proposal-depth-stage1b.v1"
        and stage1b.get("status") == "TARGET_FREE_NON_IDENTITY_POLICY_FROZEN"
        and stage1b.get("exact_repeat", {}).get("aggregate_identity_sha256")
        == "91fedb662eef097fc76ff9dde6e9c4f5b8d46535a421aa14ba3c981e4f852cfa"
        and stage1b.get("decision", {}).get("status")
        == "TARGET_FREE_NON_IDENTITY_POLICY_FROZEN"
        and stage1b_manifest.get("status") == "TARGET_FREE_NON_IDENTITY_POLICY_FROZEN"
        and stage1b_manifest.get("result", {}).get("sha256")
        == PINNED_REPO_FILES["stage1b_result"][1]
    ):
        raise Stage2Error("frozen prerequisite document semantics drifted")
    return v28, stage1a, stage1b


def _validate_v28_worker_record(
    record: Mapping[str, Any], fold: int, snapshots: MutableMapping[str, FileSnapshot]
) -> tuple[dict[str, Any], str]:
    if set(record) != {"bytes", "path", "sha256"}:
        raise Stage2Error("v2.8 audited worker record schema drifted")
    expected_relative = (
        "experiments/fast_track/small_ranker_v2_8/stage1_20260830T1834/"
        "first/outer_%d/outer_complete.json" % fold
    )
    if str(record["path"]).replace("\\", "/") != expected_relative:
        raise Stage2Error("v2.8 audited worker path drifted")
    path = (ROOT / expected_relative).resolve()
    snapshot = _snapshot_regular(path, str(record["sha256"]), int(record["bytes"]))
    snapshots["v28:outer_complete:%d" % fold] = snapshot
    value = _load_json(path)
    if not (
        value.get("schema_version") == "small-ranker-strict-outer-restack-outer-cache.v1"
        and value.get("status") == "OUTER_CACHE_COMPLETE"
        and value.get("pass_name") == "first"
        and int(value.get("outer_fold", -1)) == fold
        and value.get("privacy", {}).get(
            "held_outcome_rows_retained_or_supplied_to_fit_selection_or_metric"
        )
        == 0
        and value.get("privacy", {}).get("held_state_or_outcome_metric_computed")
        is False
    ):
        raise Stage2Error("v2.8 outer cache semantics drifted")
    return value, str(record["sha256"])


def _load_frozen_inputs(
    snapshots: MutableMapping[str, FileSnapshot], base_module: Any
) -> FrozenInputs:
    v28, stage1a, stage1b = _validate_protocol_documents(snapshots)
    v28_pairs = _outer_map(v28.get("outer_pairs"), "v2.8 outer-pair")
    stage1a_first = _outer_map(stage1a.get("first"), "Stage1a first")
    stage1a_repeat = _outer_map(stage1a.get("repeat"), "Stage1a repeat")
    stage1a_pairs = _outer_map(stage1a.get("outer_pairs"), "Stage1a outer-pair")
    stage1b_first = _outer_map(stage1b.get("first"), "Stage1b first")
    stage1b_repeat = _outer_map(stage1b.get("repeat"), "Stage1b repeat")
    stage1b_pairs = _outer_map(stage1b.get("outer_pairs"), "Stage1b outer-pair")

    audited = v28.get("audited_worker_result_files", {}).get("first")
    if not isinstance(audited, list) or len(audited) != OUTER_FOLDS:
        raise Stage2Error("v2.8 audited first-pass registry is incomplete")
    audited_by_fold: dict[int, Mapping[str, Any]] = {}
    for record in audited:
        raw = str(record.get("path", "")).replace("\\", "/")
        matched = [
            fold
            for fold in range(OUTER_FOLDS)
            if raw.endswith("/first/outer_%d/outer_complete.json" % fold)
        ]
        if len(matched) != 1 or matched[0] in audited_by_fold:
            raise Stage2Error("v2.8 audited first-pass path is ambiguous")
        audited_by_fold[matched[0]] = record
    if set(audited_by_fold) != set(range(OUTER_FOLDS)):
        raise Stage2Error("v2.8 audited first-pass coverage failed")

    held_orders: list[np.ndarray] = []
    candidate_shards: list[np.ndarray] = []
    available_shards: list[np.ndarray] = []
    current_chosen_shards: list[np.ndarray] = []
    current_activation_shards: list[np.ndarray] = []
    supplement_shards: list[np.ndarray] = []
    seen_paths: set[str] = set()

    stage1a_partition = _outer_map(
        stage1a.get("held_partition", {}).get("per_outer"), "Stage1a partition"
    )
    stage1b_order_hashes = stage1b.get("held_partition", {}).get(
        "per_outer_held_order_sha256"
    )
    if not isinstance(stage1b_order_hashes, list) or len(stage1b_order_hashes) != 5:
        raise Stage2Error("Stage1b held-order registry is incomplete")

    for fold in range(OUTER_FOLDS):
        outer_complete, outer_file_sha = _validate_v28_worker_record(
            audited_by_fold[fold], fold, snapshots
        )
        v28_pair = v28_pairs[fold]
        if not (
            v28_pair.get("equal") is True
            and outer_complete.get("identity_sha256") == v28_pair.get("identity_sha256")
            and outer_complete.get("identity_sha256")
            == _canonical_sha256(outer_complete.get("identity"))
        ):
            raise Stage2Error("v2.8 first/repeat outer identity drifted")
        ordinal_record = outer_complete.get("held", {}).get("session_ordinal")
        ordinal_relative = (
            "experiments/fast_track/small_ranker_v2_8/stage1_20260830T1834/"
            "first/outer_%d/held/session_ordinal.npy" % fold
        )
        ordinal = _load_registered_array(
            ordinal_record,
            ordinal_relative,
            (400,),
            "int16",
            "array:v28:outer:%d:session_ordinal" % fold,
            snapshots,
        )
        ordinal_hash = _logical_array_sha256(ordinal)
        if not (
            np.array_equal(ordinal, np.sort(ordinal))
            and len(np.unique(ordinal)) == 400
            and np.all((ordinal >= 0) & (ordinal < SESSION_COUNT))
            and ordinal_hash
            == outer_complete.get("identity", {}).get("held", {}).get("session_ordinal")
            == outer_complete.get("domains", {}).get("H_%d" % fold, {}).get(
                "session_sha256"
            )
        ):
            raise Stage2Error("v2.8 held session ordinal identity failed")

        first_a = stage1a_first[fold]
        repeat_a = stage1a_repeat[fold]
        pair_a = stage1a_pairs[fold]
        partition_a = stage1a_partition[fold]
        if not (
            first_a.get("pass_name") == "first"
            and first_a.get("status") == "TARGET_FREE_SURFACE_COMPLETE"
            and repeat_a.get("pass_name") == "repeat"
            and first_a.get("identity_sha256") == repeat_a.get("identity_sha256")
            == pair_a.get("identity_sha256")
            and first_a.get("identity_sha256")
            == _canonical_sha256(first_a.get("identity"))
            and repeat_a.get("identity_sha256")
            == _canonical_sha256(repeat_a.get("identity"))
            and pair_a.get("equal") is True
            and pair_a.get("physical_repeat_equal") is True
            and first_a.get("identity", {}).get("source_outer_identity_sha256")
            == v28_pair.get("identity_sha256")
            and first_a.get("sources", {}).get("outer_result") == outer_file_sha
            and first_a.get("phases", {}).get("held_H", {}).get("session_order_sha256")
            == first_a.get("identity", {}).get("held_session_order_sha256")
            == partition_a.get("held_order_sha256")
            == ordinal_hash
        ):
            raise Stage2Error("Stage1a held lineage drifted")
        files_a = first_a.get("phases", {}).get("held_H", {}).get("files", {})
        if set(files_a) != {
            "available",
            "candidates",
            "current_activation",
            "current_choice",
            "current_chosen",
            "family_choices",
            "features",
            "incumbent",
            "source_mask",
        }:
            raise Stage2Error("Stage1a held surface registry drifted")
        loaded_a: dict[str, np.ndarray] = {}
        specs = {
            "candidates": ((400, 10, 15), "int16"),
            "available": ((400, 10, 15), "bool"),
            "current_chosen": ((400, 10), "uint8"),
            "current_activation": ((400, 10), "bool"),
        }
        for field, (shape, dtype) in specs.items():
            relative = (
                "experiments/fast_track/small_ranker_v2_9/"
                "stage1a_20260830T225937/first/outer_%d/held_H/%s.npy"
                % (fold, field)
            )
            record = files_a[field]
            if record.get("array_sha256") != first_a.get("identity", {}).get(
                "phases", {}
            ).get("held_H", {}).get("array_sha256", {}).get(field):
                raise Stage2Error("Stage1a held logical registry drifted")
            loaded_a[field] = _load_registered_array(
                record,
                relative,
                shape,
                dtype,
                "array:stage1a:outer:%d:%s" % (fold, field),
                snapshots,
            )

        first_b = stage1b_first[fold]
        repeat_b = stage1b_repeat[fold]
        pair_b = stage1b_pairs[fold]
        if not (
            first_b.get("pass_name") == "first"
            and first_b.get("status") == "T_ONLY_SELECTOR_COMPLETE"
            and repeat_b.get("pass_name") == "repeat"
            and first_b.get("identity_sha256") == repeat_b.get("identity_sha256")
            == pair_b.get("identity_sha256")
            and first_b.get("identity_sha256")
            == _canonical_sha256(first_b.get("identity"))
            and repeat_b.get("identity_sha256")
            == _canonical_sha256(repeat_b.get("identity"))
            and pair_b.get("equal") is True
            and first_b.get("identity", {}).get("held_order_sha256") == ordinal_hash
            and stage1b_order_hashes[fold] == ordinal_hash
            and first_b.get("identity", {}).get("stage1a_outer_identity")
            == first_a.get("identity")
        ):
            raise Stage2Error("Stage1b held lineage drifted")
        supplement_record = first_b.get("files", {}).get("supplement")
        repeat_supplement_record = repeat_b.get("files", {}).get("supplement")
        if not (
            _file_identity(supplement_record)
            == first_b.get("identity", {}).get("output_arrays", {}).get("supplement")
            and _file_identity(supplement_record) == _file_identity(repeat_supplement_record)
        ):
            raise Stage2Error("Stage1b supplement repeat identity drifted")
        supplement_relative = (
            "experiments/fast_track/small_ranker_v2_9/"
            "stage1b_20260831T000352/first/outer_%d/supplement.npy" % fold
        )
        supplement = _load_registered_array(
            supplement_record,
            supplement_relative,
            (400, 10),
            "bool",
            "array:stage1b:outer:%d:supplement" % fold,
            snapshots,
        )
        if not (
            int(supplement.sum()) == int(first_b.get("supplement_turns", -1))
            and int(np.any(supplement, axis=1).sum())
            == int(first_b.get("supplement_sessions", -1))
            and int(supplement.sum(axis=1).max(initial=0)) <= 1
        ):
            raise Stage2Error("Stage1b supplement count drifted")

        held_orders.append(ordinal)
        candidate_shards.append(loaded_a["candidates"])
        available_shards.append(loaded_a["available"])
        current_chosen_shards.append(loaded_a["current_chosen"])
        current_activation_shards.append(loaded_a["current_activation"])
        supplement_shards.append(supplement)

    owner = np.full(SESSION_COUNT, 255, dtype=np.uint8)
    coverage = np.zeros(SESSION_COUNT, dtype=np.uint8)
    candidates = np.full((SESSION_COUNT, TURN_COUNT, 15), -1, dtype=np.int16)
    available = np.zeros((SESSION_COUNT, TURN_COUNT, 15), dtype=bool)
    stitched_chosen = np.zeros((SESSION_COUNT, TURN_COUNT), dtype=np.uint8)
    stitched_activation = np.zeros((SESSION_COUNT, TURN_COUNT), dtype=bool)
    supplement = np.zeros((SESSION_COUNT, TURN_COUNT), dtype=bool)
    for fold, order in enumerate(held_orders):
        if np.any(owner[order] != 255):
            raise Stage2Error("held partitions overlap")
        owner[order] = fold
        coverage[order] += 1
        candidates[order] = candidate_shards[fold]
        available[order] = available_shards[fold]
        stitched_chosen[order] = current_chosen_shards[fold]
        stitched_activation[order] = current_activation_shards[fold]
        supplement[order] = supplement_shards[fold]
    if not (
        np.all(coverage == 1)
        and _logical_array_sha256(coverage)
        == "bbdbdfa1aad6975399baa2db9d5f554000b79024d1d04581b3f6ccff2dfc4334"
        and _logical_array_sha256(owner)
        == "0cbb3b977530d520717d5dbb3a58c6183539cf97366e78647c6fa15eaddecfe5"
        and np.all((candidates[available] >= 0) & (candidates[available] < 100))
        and np.all(candidates[~available] == -1)
        and all(
            len(np.unique(candidates[session, turn][available[session, turn]]))
            == int(available[session, turn].sum())
            for session in range(SESSION_COUNT)
            for turn in range(TURN_COUNT)
        )
        and int(supplement.sum()) == 698
        and int(np.any(supplement, axis=1).sum()) == 698
        and int(supplement.sum(axis=1).max(initial=0)) <= 1
    ):
        raise Stage2Error("stitched proposal or partition invariant failed")

    frozen_files = stage1b.get("frozen_policy", {}).get("files", {})
    if set(frozen_files) != {
        "domain_local_current_chosen",
        "domain_local_current_activation",
        "final_chosen",
        "final_activation",
    }:
        raise Stage2Error("frozen policy registry drifted")
    policy_specs = {
        "domain_local_current_chosen": "uint8",
        "domain_local_current_activation": "bool",
        "final_chosen": "uint8",
        "final_activation": "bool",
    }
    policies: dict[str, np.ndarray] = {}
    protocol = _load_json(PROTOCOL)
    protocol_policy = protocol["frozen_target_free_policy"]
    for field, dtype in policy_specs.items():
        record = frozen_files[field]
        binding = protocol_policy[field]
        if not (
            record.get("path") == binding.get("path")
            and record.get("sha256") == binding.get("file_sha256")
            and record.get("array_sha256") == binding.get("array_sha256")
            and record.get("dtype") == binding.get("dtype")
        ):
            raise Stage2Error("protocol frozen-policy binding drifted")
        policies[field] = _load_registered_array(
            record,
            str(binding["path"]),
            (SESSION_COUNT, TURN_COUNT),
            dtype,
            "array:policy:%s" % field,
            snapshots,
        )
    if not (
        np.array_equal(stitched_chosen, policies["domain_local_current_chosen"])
        and np.array_equal(
            stitched_activation, policies["domain_local_current_activation"]
        )
        and (
            not np.array_equal(policies["final_chosen"], stitched_chosen)
            or not np.array_equal(policies["final_activation"], stitched_activation)
        )
    ):
        raise Stage2Error("frozen target-free policy stitch failed")
    _validate_final_composition(
        policies["domain_local_current_chosen"],
        policies["domain_local_current_activation"],
        policies["final_chosen"],
        policies["final_activation"],
        supplement,
        candidates,
        available,
    )

    snapshots["external:projected_features"] = _snapshot_regular(
        PROJECTED_FEATURES_PATH, PROJECTED_FEATURES_SHA256
    )
    snapshots["external:oof_scores"] = _snapshot_regular(
        OOF_SCORES_PATH, OOF_SCORES_SHA256
    )
    features = np.load(PROJECTED_FEATURES_PATH, mmap_mode="r", allow_pickle=False)
    scores = np.load(OOF_SCORES_PATH, mmap_mode="r", allow_pickle=False)
    try:
        global_surface = _build_global_runtime(features, scores, base_module)
    finally:
        for value in (features, scores):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()

    expected_arrays = _expected_registered_array_paths()
    actual_arrays = {
        name: snapshot.path
        for name, snapshot in snapshots.items()
        if name.startswith("array:")
    }
    if actual_arrays != expected_arrays or len(
        {str(path).lower() for path in actual_arrays.values()}
    ) != 34:
        raise Stage2Error("Stage2 registered NPY allow-list is not the exact 34-file set")
    for value in (
        owner,
        candidates,
        available,
        supplement,
        *policies.values(),
    ):
        value.setflags(write=False)
    return FrozenInputs(
        owner,
        tuple(held_orders),
        candidates,
        available,
        supplement,
        policies["domain_local_current_chosen"],
        policies["domain_local_current_activation"],
        policies["final_chosen"],
        policies["final_activation"],
        global_surface,
        stage1a,
        stage1b,
    )


def _validate_label_member(name: str, value: np.ndarray) -> None:
    if name == "baseline_rank" and np.any(value > 10):
        raise Stage2Error("baseline rank is outside 0..10")
    if name == "positive_index" and np.any((value < -1) | (value >= 100)):
        raise Stage2Error("positive index is outside -1..99")
    if name == "eligible_from" and np.any((value < 1) | (value > 10)):
        raise Stage2Error("eligible_from is outside 1..10")
    if name in {"outer_fold", "inner_fold"} and (
        np.any(value > 4) or set(np.unique(value).tolist()) != set(range(5))
    ):
        raise Stage2Error("fold member is outside 0..4 or lacks coverage")


def _load_outcomes_from_open_handle(
    handle: BinaryIO, np_load: Any = np.load
) -> OutcomeBundle:
    """Access exactly the five protocol members in their frozen order."""
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
                raise Stage2Error("sealed label member schema failed: %s" % name)
            copied = np.asarray(member).copy()
            _validate_label_member(name, copied)
            copied.setflags(write=False)
            values[name] = copied
            del member
    finally:
        archive.close()
    if tuple(values) != tuple(name for name, _shape, _dtype in LABEL_MEMBER_SPECS):
        raise Stage2Error("sealed label member access order drifted")
    return OutcomeBundle(**values)


def _validate_label_binding(outcomes: OutcomeBundle, owner: np.ndarray) -> dict[str, Any]:
    baseline_hit = derive_baseline_session_hit(
        outcomes.baseline_rank, outcomes.eligible_from
    )
    if not (
        np.array_equal(outcomes.outer_fold, owner)
        and int(baseline_hit.sum()) == 1895
        and all(int(np.sum(outcomes.outer_fold == fold)) == 400 for fold in range(5))
    ):
        raise Stage2Error("sealed label partition or baseline binding failed")
    return {
        "member_access_order": [name for name, _shape, _dtype in LABEL_MEMBER_SPECS],
        "member_access_count": 5,
        "archive_open_count": 1,
        "baseline_session_hit_derived_from_allowed_members": True,
        "baseline_hit_sessions": int(baseline_hit.sum()),
        "outer_fold_matches_target_free_owner": True,
    }


def _official_global_identity(
    outcomes: OutcomeBundle,
    chosen: np.ndarray,
    activation: np.ndarray,
    selections: Sequence[Mapping[str, Any]],
    counters: FitCounters,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    quantiles = tuple(float(row["selected_quantile"]) for row in selections)
    if not (
        _raw_array_sha256(chosen) == EXPECTED_GLOBAL_CHOSEN_RAW_SHA256
        and _raw_array_sha256(activation) == EXPECTED_GLOBAL_ACTIVATION_RAW_SHA256
        and quantiles == EXPECTED_GLOBAL_QUANTILES
        and int(np.any(activation, axis=1).sum()) == 1251
        and int(activation.sum()) == 6573
        and counters.helper_invocations == 60
        and counters.liblinear_fit_calls == 60
        and counters.constant_gate_returns == 0
    ):
        raise Stage2Error("frozen global comparator decision identity drifted")

    # Only after the target-free decision identities above pass may complete
    # H_o outcome rows enter the comparator state/metric computation.
    labels = {
        "baseline_rank": outcomes.baseline_rank,
        "positive_index": outcomes.positive_index,
        "eligible_from": outcomes.eligible_from,
    }
    state = policy_session_state(labels, chosen, activation)
    metrics = exact_metrics(state, np.ones(SESSION_COUNT, dtype=bool))
    if not (
        int(metrics["hits"]) == 1943
        and int(metrics["misses"]) == 57
        and float(metrics["rounded"]["hit_rate_at_10"]) == 0.9715
        and float(metrics["rounded"]["mrr"]) == 0.676861
        and float(metrics["rounded"]["mttc"]) == 3.056
        and float(metrics["rounded"]["technical_score"]) == 0.847688
    ):
        raise Stage2Error("frozen global comparator metric drifted")
    return state, {
        "chosen_raw_sha256": _raw_array_sha256(chosen),
        "activation_raw_sha256": _raw_array_sha256(activation),
        "fold_quantiles": list(quantiles),
        "activation_sessions": int(np.any(activation, axis=1).sum()),
        "activation_turns": int(activation.sum()),
        "metrics": metrics,
        "fit_counters": {
            "head_fit_helper_invocations": counters.helper_invocations,
            "liblinear_fit_calls": counters.liblinear_fit_calls,
            "constant_gate_returns": counters.constant_gate_returns,
            "counter_sum_verified": True,
        },
        "t_only_target_derivation_verified": True,
        "held_rows_used_for_prediction_only": True,
    }


def _proposal_union_one(
    name: str,
    current_state: Mapping[str, np.ndarray],
    candidates: np.ndarray,
    available: np.ndarray,
    outcomes: OutcomeBundle,
    owner: np.ndarray,
) -> dict[str, Any]:
    miss = ~np.asarray(current_state["hit"], dtype=bool)
    eligible = (
        np.arange(1, TURN_COUNT + 1)[None, :, None]
        >= outcomes.eligible_from[:, None, None]
    )
    correct = (
        available
        & eligible
        & (outcomes.positive_index[:, :, None] >= 0)
        & (candidates == outcomes.positive_index[:, :, None])
    )
    correct_on_miss = correct & miss[:, None, None]
    reachable = miss & np.any(correct, axis=(1, 2))
    hits = int((~miss).sum())
    misses = int(miss.sum())
    reachable_count = int(reachable.sum())
    result = {
        "comparator": name,
        "current_hits": hits,
        "current_misses": misses,
        "hits_by_outer_fold": [int(np.sum((~miss) & (owner == fold))) for fold in range(5)],
        "misses_by_outer_fold": [int(np.sum(miss & (owner == fold))) for fold in range(5)],
        "correct_action_rows_on_current_misses": int(correct_on_miss.sum()),
        "reachable_sessions": reachable_count,
        "reachable_by_outer_fold": [
            int(np.sum(reachable & (owner == fold))) for fold in range(5)
        ],
        "maximum_zero_harm_ceiling_hits": hits + reachable_count,
        "maximum_zero_harm_ceiling_hr_at_10": round(
            float(hits + reachable_count) / SESSION_COUNT, 6
        ),
    }
    checks = [
        result["maximum_zero_harm_ceiling_hits"] == hits + reachable_count,
        sum(result["hits_by_outer_fold"]) == hits,
        sum(result["misses_by_outer_fold"]) == misses,
        sum(result["reachable_by_outer_fold"]) == reachable_count,
        reachable_count <= misses,
    ]
    if name == "frozen_global_current":
        checks.extend(
            [
                reachable_count <= 43,
                result["maximum_zero_harm_ceiling_hr_at_10"] <= 0.993,
                hits == 1943,
                misses == 57,
            ]
        )
        result["capacity_at_least_37"] = reachable_count >= 37
        result["capacity_strictly_above_0_99"] = reachable_count >= 38
    if not all(checks):
        raise Stage2Error("posthoc proposal-union invariant failed")
    result["invariants_pass"] = True
    return result


def _posthoc_proposal_union(
    domain_state: Mapping[str, np.ndarray],
    global_state: Mapping[str, np.ndarray],
    frozen: FrozenInputs,
    outcomes: OutcomeBundle,
) -> dict[str, Any]:
    try:
        return {
            "status": "REPORT_ONLY_AVAILABLE",
            "computed_after_strict_decision_freeze": True,
            "policy_selection_or_mutation": False,
            "comparators": [
                _proposal_union_one(
                    "domain_local_current",
                    domain_state,
                    frozen.candidates,
                    frozen.available,
                    outcomes,
                    frozen.owner,
                ),
                _proposal_union_one(
                    "frozen_global_current",
                    global_state,
                    frozen.candidates,
                    frozen.available,
                    outcomes,
                    frozen.owner,
                ),
            ],
            "frozen_candidate_recall_reused": {
                "C10": 0.9475,
                "C20": 0.9715,
                "C50": 0.991,
                "C100": 0.993,
                "C200": "unavailable in the frozen blind trace and not inferred",
            },
        }
    except Exception:
        return {
            "status": "REPORT_ONLY_UNAVAILABLE",
            "computed_after_strict_decision_freeze": True,
            "policy_selection_or_mutation": False,
        }


def _result_privacy_scan(value: object) -> None:
    forbidden = {
        "asin",
        "ground_truth",
        "identifier",
        "membership",
        "parent_asin",
        "positive_index",
        "sample_id",
        "target",
        "target_id",
        "user_id",
        "labels",
        "outcomes",
        "session_ordinal",
        "session_ordinals",
        "target_hashes",
        "memberships",
    }

    def walk(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                lowered = str(key).lower()
                if lowered in forbidden:
                    raise Stage2Error("result key crosses the aggregate-only privacy boundary")
                walk(child)
        elif isinstance(item, (list, tuple)):
            if len(item) > 64:
                raise Stage2Error("result contains a per-session-shaped sequence")
            for child in item:
                walk(child)
        elif isinstance(item, np.ndarray):
            raise Stage2Error("result contains a serialized numeric array")
        elif isinstance(item, str) and ASIN_SHAPE_RE.search(item.encode("utf-8", "ignore")):
            raise Stage2Error("result contains an identity-shaped value")

    walk(value)


def _write_receipt_payload(handle: BinaryIO, value: Mapping[str, Any]) -> int:
    raw = _canonical_bytes(value, newline=True)
    handle.seek(0)
    handle.write(raw)
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())
    return len(raw)


def _best_effort_close(handle: BinaryIO) -> bool:
    try:
        handle.close()
        return True
    except Exception:
        return False


def _open_one_shot_receipt(path: Path, root: Path) -> BinaryIO:
    path = Path(path)
    root = Path(root).resolve()
    if path.exists() or path.is_symlink():
        raise Stage2Error("Stage2 one-shot receipt already exists")
    resolved = path.resolve()
    if root not in resolved.parents:
        raise Stage2Error("Stage2 output path escapes its authorized root")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.open("xb")


def _physical_decision_binding(
    physical_end: Mapping[str, Any],
    label_sha256: str,
    label_bytes: int,
    label_stat_identity: Sequence[int],
) -> dict[str, Any]:
    return {
        "target_free_input_audit_identity_sha256": str(
            physical_end["identity_sha256"]
        ),
        "target_free_input_count": int(physical_end["input_count"]),
        "target_free_input_bytes": int(physical_end["total_bytes"]),
        "label_start_end_sha256": str(label_sha256),
        "label_start_end_bytes": int(label_bytes),
        "label_stat_identity_sha256": _canonical_sha256(
            [int(value) for value in label_stat_identity]
        ),
        "same_label_handle_start_end": True,
    }


def run(implementation_commit: str) -> dict[str, Any]:
    started = time.perf_counter()
    snapshots: dict[str, FileSnapshot] = {}
    receipt: BinaryIO | None = None
    label_handle: BinaryIO | None = None
    semantic_attach_runs = 0
    implementation_snapshot: FileSnapshot | None = None
    success_committed = False
    try:
        environment = _validate_environment()
        git = _validate_git_checkpoint(implementation_commit)
        implementation_sha = _sha256(Path(__file__).resolve())
        implementation_snapshot = _snapshot_regular(
            Path(__file__).resolve(), implementation_sha
        )
        snapshots["implementation"] = implementation_snapshot

        if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
            raise Stage2Error("Stage2 one-shot receipt already exists")
        if ROOT.resolve() not in OUTPUT_PATH.resolve().parents:
            raise Stage2Error("Stage2 output path escapes the worktree")

        # Source hashes are validated before imports so operation-equivalence is
        # bound to the pinned helper implementations.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts import train_small_ranker as base

        if Path(base.__file__).resolve() != PINNED_REPO_FILES["base"][0].resolve():
            raise Stage2Error("imported base helper path drifted")
        frozen = _load_frozen_inputs(snapshots, base)
        pre_attach_working_set, pre_attach_peak = _require_process_memory(
            "before_receipt"
        )

        # All preflight and target-free reconstruction is now complete.  The
        # label file is opened once, physically hashed through this handle, and
        # the irreversible receipt is created before the sole semantic np.load.
        if LABEL_PATH.is_symlink() or not LABEL_PATH.is_file():
            raise Stage2Error("sealed label archive is unavailable")
        label_handle = LABEL_PATH.open("rb")
        label_start_sha, label_start_bytes = _sha256_stream(label_handle)
        if not (
            label_start_sha == EXPECTED_LABEL_SHA256
            and label_start_bytes == EXPECTED_LABEL_BYTES
        ):
            raise Stage2Error("sealed label archive physical hash failed")
        label_start_stat = _stat_identity(LABEL_PATH.resolve())
        receipt = _open_one_shot_receipt(OUTPUT_PATH, ROOT)

        label_handle.seek(0)
        semantic_attach_runs = 1
        outcomes = _load_outcomes_from_open_handle(label_handle)
        label_binding = _validate_label_binding(outcomes, frozen.owner)

        global_activation, selections, counters = reproduce_frozen_global(
            frozen.global_surface, outcomes, frozen.owner, base
        )
        global_state, global_identity = _official_global_identity(
            outcomes,
            frozen.global_surface.chosen,
            global_activation,
            selections,
            counters,
        )

        labels = {
            "baseline_rank": outcomes.baseline_rank,
            "positive_index": outcomes.positive_index,
            "eligible_from": outcomes.eligible_from,
        }
        domain_state = policy_session_state(
            labels, frozen.domain_chosen, frozen.domain_activation
        )
        final_state = policy_session_state(
            labels, frozen.final_chosen, frozen.final_activation
        )
        primary = comparison_payload(
            domain_state, final_state, frozen.supplement, frozen.owner
        )
        secondary = comparison_payload(
            global_state, final_state, frozen.final_activation, frozen.owner
        )
        drift = comparison_payload(
            global_state, domain_state, frozen.domain_activation, frozen.owner
        )
        if primary["aggregate"]["policy"] != secondary["aggregate"]["policy"] or [
            row["policy"] for row in primary["folds"]
        ] != [row["policy"] for row in secondary["folds"]]:
            raise Stage2Error("final policy metrics differ across comparator payloads")

        label_end_sha, label_end_bytes = _sha256_stream(label_handle)
        if not (
            label_end_sha == label_start_sha
            and label_end_bytes == label_start_bytes
            and _stat_identity(LABEL_PATH.resolve()) == label_start_stat
        ):
            raise Stage2Error("sealed label archive changed during attach")
        physical_end = _revalidate_snapshots(snapshots)
        physical_decision_binding = _physical_decision_binding(
            physical_end,
            label_start_sha,
            label_start_bytes,
            label_start_stat,
        )

        primary_gate = _comparison_gate(primary, secondary=False)
        secondary_gate = _comparison_gate(secondary, secondary=True)
        current_working_set, peak = _require_process_memory("after_receipt")
        resource_checks = {
            "workers_exactly_one": environment["workers"] == 1,
            "stage1_first_repeat_identity": True,
            "stage1b_target_free_policy_identity_and_nonidentity": True,
            "all_git_source_path_file_array_partition_privacy_and_physical_checks": bool(
                physical_end["start_end_equal"]
            ),
            "new_xgboost_or_challenger_selector_fits_zero": True,
            "fixed_secondary_comparator_head_fit_helper_invocations_60": counters.helper_invocations
            == 60
            and counters.liblinear_fit_calls == 60
            and counters.constant_gate_returns == 0,
            "new_retrieval_queries_zero": True,
            "new_full_agent_or_official_evaluator_runs_zero": True,
            "semantic_outcome_attach_runs_one": semantic_attach_runs == 1,
            "new_feature_or_candidate_cache_builds_zero": True,
            "pre_stage_wall_seconds_within_600": 152.638274 <= 600.0,
            "observed_cumulative_peak_working_set_within_limit": max(
                773_652_480, int(peak)
            )
            <= 3_289_186_304,
            "stage2_peak_working_set_measurement_available": int(peak) > 0,
            "stage2_result_upper_bound_within_additional_cache_limit": 1_000_000
            <= 488_741_963,
            "cumulative_cache_upper_bound_within_1gib": 584_999_861 + 1_000_000
            <= 1_073_741_824,
        }
        required_pass = all(resource_checks.values())
        promote = bool(primary_gate["pass"] and secondary_gate["pass"] and required_pass)
        strict_decision: dict[str, Any] = {
            "primary_gate": primary_gate,
            "secondary_gate": secondary_gate,
            "required_for_both": {
                "checks": resource_checks,
                "pass": required_pass,
                "workers": 1,
                "semantic_outcome_attach_runs": semantic_attach_runs,
                "new_xgboost_or_challenger_selector_fits": 0,
                "new_retrieval_queries": 0,
                "new_full_agent_or_official_evaluator_runs": 0,
                "new_feature_or_candidate_cache_builds": 0,
                "fixed_head_fit_helper_invocations": counters.helper_invocations,
                "stage0_plus_stage1a_plus_stage1b_wall_seconds": 152.638274,
                "pre_stage2_peak_working_set_bytes": 773_652_480,
                "pre_attach_observed_working_set_bytes": int(pre_attach_working_set),
                "pre_attach_observed_peak_working_set_bytes": int(pre_attach_peak),
                "stage2_observed_working_set_bytes": int(current_working_set),
                "stage2_observed_peak_working_set_bytes": int(peak),
                "observed_cumulative_peak_working_set_bytes": max(
                    773_652_480, int(peak)
                ),
                "stage0_plus_stage1a_plus_stage1b_cache_bytes": 584_999_861,
                "stage2_result_bytes_fail_closed_upper_bound": 1_000_000,
                "observed_cumulative_cache_bytes_fail_closed_upper_bound": 585_999_861,
                "physical_input_audit_binding": physical_decision_binding,
            },
            "promote": promote,
            "status": (
                "PROMOTE_FULL_DATA_ARTIFACT_FREEZE_ONLY"
                if promote
                else "NO_GO_CLOSE_EXACT_K5_ROUTE"
            ),
            "strict_decision_frozen_before_posthoc": True,
        }
        strict_decision_sha = _canonical_sha256(strict_decision)

        # The proposal union is strictly report-only and occurs after the
        # decision object and its hash have been frozen in memory.
        posthoc = _posthoc_proposal_union(
            domain_state, global_state, frozen, outcomes
        )
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": "SR-V2.9-STRICT-TOP5-PROPOSAL-DEPTH-STAGE2",
            "status": strict_decision["status"],
            "evidence_boundary": {
                "split": "train_explore",
                "semantic_mode": "off",
                "outcome_attach_runs": 1,
                "full_agent_or_official_evaluator_runs": 0,
                "forbidden_splits_or_external_data_opened": 0,
                "target_runtime_features": 0,
                "held_rows_supplied_to_fit_or_selection": 0,
                "outcome_values_serialized_per_session": 0,
            },
            "git": git,
            "environment": environment,
            "sources": {
                "protocol_sha256": PROTOCOL_SHA256,
                "implementation_source_sha256": implementation_snapshot.sha256,
                "input_start_end_audit": physical_end,
                "label_archive": {
                    "sha256": label_start_sha,
                    "bytes": label_start_bytes,
                    "same_binary_handle_start_end": True,
                    "start_end_hash_and_stat_equal": True,
                },
            },
            "label_attach": label_binding,
            "frozen_global_comparator": global_identity,
            "comparisons": {
                "primary_final_vs_domain_local_current": primary,
                "secondary_final_vs_frozen_global_current": secondary,
                "descriptive_global_to_domain_local_drift": drift,
                "comparison_order_frozen": True,
                "same_final_metrics_exact_across_comparators": True,
            },
            "strict_decision": strict_decision,
            "strict_decision_canonical_sha256": strict_decision_sha,
            "posthoc_proposal_union": posthoc,
            "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
            "privacy": {
                "aggregate_numeric_evidence_only": True,
                "identifiers_or_target_strings": 0,
                "per_session_rows_or_membership_arrays": 0,
                "other_output_files": 0,
            },
        }
        _result_privacy_scan(result)
        raw = _canonical_bytes(result, newline=True)
        if len(raw) > 1_000_000:
            raise Stage2Error("Stage2 canonical result exceeds its fail-closed bound")
        result["output"] = {
            "canonical_json_bytes_without_output_record": len(raw),
            "within_fail_closed_upper_bound": True,
        }
        _result_privacy_scan(result)
        label_handle.close()
        label_handle = None
        final_bytes = _write_receipt_payload(receipt, result)
        if final_bytes > 1_000_000:
            raise Stage2Error("Stage2 final result exceeds its fail-closed bound")
        success_committed = True
        # The canonical payload has already been flushed and fsynced.  A close
        # error after that durable boundary cannot turn the receipt back into a
        # writable/tunable attempt.
        _best_effort_close(receipt)
        receipt = None
        return result
    except Exception as exc:
        if receipt is not None and not success_committed:
            invalid = {
                "schema_version": SCHEMA_VERSION,
                "status": "INVALID_STAGE2_ONE_SHOT_CONSUMED",
                "protocol_sha256": PROTOCOL_SHA256,
                "implementation_commit": implementation_commit,
                "semantic_outcome_attach_runs": semantic_attach_runs,
                "failure_class": type(exc).__name__,
                "rerun_authorized": False,
                "algorithm_no_go": False,
            }
            try:
                try:
                    _write_receipt_payload(receipt, invalid)
                except Exception:
                    pass
                _best_effort_close(receipt)
            finally:
                receipt = None
            raise Stage2ConsumedError(
                "Stage2 one-shot was consumed; the fixed receipt records INVALID_STAGE2_ONE_SHOT_CONSUMED"
            ) from exc
        raise
    finally:
        if label_handle is not None:
            _best_effort_close(label_handle)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation-commit",
        required=True,
        help="Exact clean pushed implementation commit to bind before outcome access.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(str(args.implementation_commit))
    print(json.dumps({
        "status": result["status"],
        "strict_decision_sha256": result["strict_decision_canonical_sha256"],
        "output": OUTPUT_PATH.relative_to(ROOT).as_posix(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
