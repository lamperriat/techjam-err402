"""One-shot sparse multi-view candidate-recall probe.

The runner stays target-free through source, entrypoint, smoke, receipt, and
two closed-trace gates.  It then attaches only the frozen shared-cohort proxy
and numeric fold labels through the pinned C200 evaluator helpers.  It never
runs ``Agent.respond`` or the full evaluator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat as stat_module
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


# This bootstrap precedes every repository import and is exercised from a
# non-repository cwd to guard the consumed C400/E0 missing-module failure.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True


SCHEMA_VERSION = "small-ranker-registry-ca-g0-outcome.v1"
WORKER_SCHEMA_VERSION = "small-ranker-registry-ca-g0-worker-summary.v1"
EXPERIMENT_ID = "SR-V2.19-REGISTRY-CA-G0"
BRANCH = "small-ranker-v2.19-sparse-multiview"
REMOTE = "origin"
REMOTE_URL = "https://github.com/lamperriat/techjam-err402.git"
REMOTE_REF = f"refs/remotes/{REMOTE}/{BRANCH}"

BASE_COMMIT = "f810642c1dd805df6543ce152904892207164c95"
PREREG_COMMIT = "c4e89db7a64b71e15aeab51049670de0bb8f8e6e"
PREREG_RELATIVE = "configs/small_ranker_v2_19.registry_ca_g0_preregistration.json"
PREREG_PATH = ROOT / PREREG_RELATIVE
PREREG_BLOB = "e480cb7efd7c3ed80f2751e843577052430ea599"
PREREG_BYTES = 20_271
PREREG_RAW_SHA256 = "c5363e51e7d6248e958ec9225bea827a062fe11fe73a7129ef26c8a539a70fc4"
PREREG_CANONICAL_SHA256 = "f51ba13ac437f57506b3bfc7d39ae1136d3a353ce3ded9ceea257ae31b7c3c35"
PREREG_PATHS = {PREREG_RELATIVE}
IMPLEMENTATION_PATHS = {
    "starter/sparse_multiview.py",
    "scripts/sparse_multiview_candidate_worker.py",
    "scripts/probe_sparse_multiview_candidate_recall.py",
    "tests/test_sparse_multiview_candidate_recall.py",
}
PINNED_BLOBS = {
    "starter/agent.py": "421c6d43c598102b8fefb181b72bab5da4bf1294",
    "starter/architecture_lab.py": "8d340d0dce3fc2f1bb987a5dd632444776a05667",
    "starter/attributes.py": "92260323f077c9861aa4edd5242aff772c875760",
    "starter/p8_negative.py": "719078234dba297ce59f68d8a2b1734ec53c9c63",
    "starter/slot_ledger.py": "72975cff12af59e4044e52911c58294cd74a785a",
    "scripts/c200_candidate_worker.py": "b94fddcf5a9b20ddde540f3f43ea9962982cb096",
    "scripts/probe_c200_candidate_recall.py": "0a57f63866683b476b9f49184673cf3154531911",
    "scripts/probe_e0_embedding_candidate_recall.py": "5bb9ec7f38f90d814d0c121c9f8992267d3491d5",
    "evaluator/local_evaluator.py": "7c808347b31ef3121a9cbc4810ac3eb325f950ba",
}

SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
CATALOG_PATH = SOURCE_ROOT / "data/catalog.jsonl"
CATALOG_BYTES = 60_546_327
CATALOG_ROWS = 50_000
CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
PROXY_PATH = SOURCE_ROOT / "experiments/fast_track/proxy_v1/proxy_train_explore.jsonl"
LABEL_PATH = SOURCE_ROOT / "experiments/fast_track/small_ranker_v1/labels_v2.npz"

C200_ROOT = Path(r"D:\tiktok\techjam-v2-16-c200-recall")
C200_CACHE_ROOT = C200_ROOT / "experiments/fast_track/c200_candidate_recall_cache_20260831"
CONTEXT_PATH = C200_CACHE_ROOT / "visible_context.jsonl"
CONTEXT_BYTES = 47_168_882
CONTEXT_ROWS = 2_000
CONTEXT_SHA256 = "f30a98700da5d480731fe7e82c87c40a22f06de290e069e20dc68f9fefecd20f"
C200_REFERENCE_PATHS = (
    C200_CACHE_ROOT / "replica_a.jsonl",
    C200_CACHE_ROOT / "replica_b.jsonl",
)
C200_TRACE_BYTES = 32_226_135
C200_TRACE_ROWS = 20_000
C200_TRACE_SHA256 = "a8589749376f48f019997a618481578dde36be4ca1fc723e8ed00056c23e40dc"
C200_CANDIDATE_CELLS = 2_425_785
C100_NORMALIZED_BYTES = 26_690_930
C100_NORMALIZED_SHA256 = "b22b035cb7789570f36db6c52256e5deb67f593f90cbbc5c334d48f2f0a01a67"

OUTPUT_PATH = ROOT / "experiments/fast_track/small_ranker_v2_19_registry_ca_g0_20260831.json"
CACHE_ROOT = ROOT / "experiments/fast_track/small_ranker_v2_19_registry_ca_g0_cache_20260831"
TRACE_PATHS = (CACHE_ROOT / "replica_a.jsonl", CACHE_ROOT / "replica_b.jsonl")
WORKER_PATH = ROOT / "scripts/sparse_multiview_candidate_worker.py"
RUNNER_PATH = Path(__file__).resolve()

EXPECTED_EXECUTABLE = Path(r"D:\450\conda\envs\tiktok\python.exe")
EXPECTED_PYTHON = "3.11.16"
EXPECTED_SQLITE = "3.53.4"
SESSION_COUNT = 2_000
TURN_COUNT = 10
RECORD_COUNT = SESSION_COUNT * TURN_COUNT
CUTOFFS = (10, 20, 50, 100, 200, 320)
EXPECTED_C200_RECALL = {10: 1_895, 20: 1_943, 50: 1_982, 100: 1_986, 200: 1_986}

TOTAL_WALL_MAXIMUM = 1_800.0
EXTRA_ROUTE_P95_MS_MAXIMUM = 100.0
PER_TURN_P95_MS_MAXIMUM = 400.0
WORKER_RSS_MAXIMUM = 1_610_612_736
CELL_RATIO_MAXIMUM = 2.0
TRACE_RATIO_MAXIMUM = 2.1
FREE_DISK_MINIMUM = 536_870_912
RECEIPT_BYTES_MAXIMUM = 24_000

COMMIT_RE = re.compile(r"[0-9a-f]{40}")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
CATALOG_ID_RE = re.compile(r"[A-Z0-9]{10}")
ASIN_SHAPE_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.I)
FORBIDDEN_RESULT_KEYS = frozenset({
    "asin", "parent_asin", "sample_id", "session_id", "product_id", "ground_truth",
    "ground_truth_parent_asin", "target", "target_id", "target_asin", "targets",
    "positive_index", "eligible_from", "eligibility", "message", "messages", "query",
    "query_terms", "ordinal", "turn", "outer_fold", "family_index",
    "per_session", "per_session_values", "membership", "membership_vector", "memberships",
    "candidates", "c200_candidates", "records", "trace_records", "raw_trace",
})
ERROR_ENUM_RE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
SAFE_NAME_RE = re.compile(r"[A-Za-z_<][A-Za-z0-9_.<>-]{0,127}")


class SparseProbeError(RuntimeError):
    """A sanitized, enumerated probe-contract failure."""

    def __init__(self, message: str, code: str = "CONTRACT_DRIFT") -> None:
        super().__init__(message)
        self.code = code


class _ProcessAuditGuard:
    """Block known protected path surfaces in Python-audited operations.

    CPython does not audit every filesystem stat.  This guard supplements,
    rather than replaces, the runner's exact manual allowlist checks.
    """

    _EVENTS = frozenset({
        "open", "import", "os.listdir", "os.scandir", "os.mkdir", "os.remove",
        "os.rename", "os.rmdir", "os.startfile", "pathlib.Path.glob",
        "pathlib.Path.rglob",
    })

    def __init__(self) -> None:
        self.active = True
        self.post_gate_target_access = False

    @staticmethod
    def _protected(path: str | os.PathLike[str]) -> bool:
        key = _lexical(Path(path))
        if key in {_lexical(PROXY_PATH), _lexical(LABEL_PATH)}:
            return True
        components = tuple(part.casefold() for part in Path(key).parts)
        markers = ("proxy", "label", "calibration", "selection", "confirmation",
                   "public", "heldout")
        return any(any(marker in component for marker in markers)
                   for component in components)

    def hook(self, event: str, arguments: tuple[Any, ...]) -> None:
        if not self.active or event not in self._EVENTS:
            return
        for argument in arguments:
            if not isinstance(argument, (str, os.PathLike)):
                continue
            try:
                protected = self._protected(argument)
                authorized = self.post_gate_target_access and _lexical(Path(argument)) in {
                    _lexical(PROXY_PATH), _lexical(LABEL_PATH)
                }
            except (OSError, TypeError, ValueError):
                continue
            if protected and not authorized:
                raise SparseProbeError("audited protected-path access denied",
                                       "AUDITED_PATH_DENIED")

    def allow_post_gate_targets(self) -> None:
        self.post_gate_target_access = True

    def close(self) -> None:
        self.active = False


def _install_process_audit_guard() -> _ProcessAuditGuard:
    guard = _ProcessAuditGuard()
    sys.addaudithook(guard.hook)
    return guard


@dataclass(frozen=True)
class FileIdentity:
    bytes: int
    rows: int
    sha256: str
    snapshot: tuple[int, int, int]

    def report(self) -> dict[str, int | str]:
        return {"bytes": self.bytes, "rows": self.rows, "sha256": self.sha256}


@dataclass(frozen=True)
class TraceValidation:
    records: tuple[dict[str, Any], ...]
    lengths: tuple[int, ...]
    c200_lengths: tuple[int, ...]
    canonical_trace_sha256: str
    canonical_trace_bytes: int
    record_count: int
    activation_turns: int
    activation_sessions: int


@dataclass(frozen=True)
class Preflight:
    environment: Mapping[str, Any]
    git: Mapping[str, Any]
    protocol: Mapping[str, Any]
    catalog_ids: frozenset[str]
    source_identities: Mapping[str, Any]
    entrypoints: Mapping[str, Any]
    smoke: Mapping[str, Any]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SparseProbeError("duplicate JSON key", "INVALID_JSON")
        result[key] = value
    return result


def _snapshot(value: os.stat_result) -> tuple[int, int, int]:
    return (int(value.st_size), int(value.st_mtime_ns), int(getattr(value, "st_ino", 0)))


def _lexical(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _is_reparse(path: Path) -> bool:
    observed = path.lstat()
    flag = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(getattr(observed, "st_file_attributes", 0) & flag)


def _require_plain(path: Path, *, directory: bool = False) -> Path:
    absolute = path.absolute()
    anchor = Path(absolute.anchor)
    current = anchor
    for part in absolute.parts[1:]:
        current /= part
        if current.exists() or current.is_symlink():
            if _is_reparse(current):
                raise SparseProbeError("path traverses a link or reparse point", "UNSAFE_PATH")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SparseProbeError("required path is unavailable", "PATH_UNAVAILABLE") from error
    if ((directory and not resolved.is_dir()) or (not directory and not resolved.is_file())):
        raise SparseProbeError("required path has unsafe type", "UNSAFE_PATH")
    return resolved


def _pre_receipt_paths() -> frozenset[str]:
    tracked = {PREREG_PATH, RUNNER_PATH, WORKER_PATH}
    tracked.update(ROOT / path for path in IMPLEMENTATION_PATHS | set(PINNED_BLOBS))
    return frozenset(map(_lexical, tracked | {CATALOG_PATH, CONTEXT_PATH, *C200_REFERENCE_PATHS}))


def _guard_experiment_data(path: Path, *, post_receipt: bool = False) -> None:
    """Check every runner-owned experiment-data operation against exact paths."""

    key = _lexical(path)
    allowed = set(_pre_receipt_paths())
    if post_receipt:
        allowed.update({_lexical(PROXY_PATH), _lexical(LABEL_PATH)})
    if key not in allowed:
        raise SparseProbeError("experiment-data path is outside the allowlist", "DATA_PATH_DENIED")


def _validate_path_policy() -> dict[str, Any]:
    for path in (CATALOG_PATH, CONTEXT_PATH, *C200_REFERENCE_PATHS, PREREG_PATH,
                 RUNNER_PATH, WORKER_PATH):
        _guard_experiment_data(path)
    denied = (
        PROXY_PATH, LABEL_PATH, SOURCE_ROOT / "data/public_set.jsonl",
        SOURCE_ROOT / "experiments/fast_track/calibration/result.json",
        SOURCE_ROOT / "experiments/selection/result.json",
        SOURCE_ROOT / "experiments/confirmation/result.json",
        SOURCE_ROOT / "experiments/heldout/result.json",
    )
    for path in denied:
        try:
            _guard_experiment_data(path)
        except SparseProbeError as error:
            if error.code != "DATA_PATH_DENIED":
                raise
        else:
            raise SparseProbeError("protected path was not denied", "DATA_POLICY_FAILURE")
    _guard_experiment_data(PROXY_PATH, post_receipt=True)
    _guard_experiment_data(LABEL_PATH, post_receipt=True)
    return {"runner_owned_exact_path_allowlist": True,
            "known_protected_paths_rejected_by_runner_guard": True,
            "python_audit_guard_for_open_import_and_os_events": True,
            "universal_os_stat_or_child_process_audit_proven": False,
            "recursive_external_scans": False}


def _file_identity(path: Path, label: str, *, post_receipt: bool = False) -> FileIdentity:
    _guard_experiment_data(path, post_receipt=post_receipt)
    resolved = _require_plain(path)
    digest = hashlib.sha256()
    byte_count = row_count = 0
    with resolved.open("rb") as handle:
        before = _snapshot(os.fstat(handle.fileno()))
        for raw in handle:
            digest.update(raw)
            byte_count += len(raw)
            row_count += 1
        after = _snapshot(os.fstat(handle.fileno()))
    if before != after or _snapshot(resolved.stat()) != after or byte_count != before[0]:
        raise SparseProbeError(f"{label} changed while hashed", "SOURCE_MUTATION")
    return FileIdentity(byte_count, row_count, digest.hexdigest(), after)


def _git(*args: str, binary: bool = False) -> Any:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(["git", *args], cwd=ROOT, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise SparseProbeError("Git identity command failed", "GIT_COMMAND_FAILED")
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _changed_paths(commitish: str) -> set[str]:
    return set(filter(None, _git("diff-tree", "--no-commit-id", "--name-only", "-r",
                                 commitish).splitlines()))


def _diff_paths(left: str, right: str) -> set[str]:
    return set(filter(None, _git("diff", "--name-only", left, right).splitlines()))


def _worktree_blob(path: Path) -> str:
    _guard_experiment_data(path)
    raw = _require_plain(path).read_bytes()
    # The frozen repository uses Git-LF as the authoritative text identity while
    # this Windows worktree has core.autocrlf=true.  Mirror Git's clean text
    # representation so an unchanged CRLF checkout binds to the tracked LF blob.
    clean = raw.replace(b"\r\n", b"\n")
    return hashlib.sha1(
        b"blob " + str(len(clean)).encode("ascii") + b"\0" + clean
    ).hexdigest()


def _validate_environment() -> dict[str, Any]:
    try:
        expected = EXPECTED_EXECUTABLE.resolve(strict=True)
        actual = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        raise SparseProbeError("formal executable is unavailable", "ENVIRONMENT_DRIFT") from error
    if not (actual.as_posix().casefold() == expected.as_posix().casefold()
            and sys.version.split()[0] == EXPECTED_PYTHON
            and sqlite3.sqlite_version == EXPECTED_SQLITE
            and os.getenv("PYTHONHASHSEED") == "0"
            and Path.cwd().resolve().as_posix().casefold() == ROOT.resolve().as_posix().casefold()):
        raise SparseProbeError("formal environment identity drifted", "ENVIRONMENT_DRIFT")
    return {"executable": actual.as_posix(), "python": EXPECTED_PYTHON,
            "sqlite": EXPECTED_SQLITE, "pythonhashseed": "0", "cpu_only": True,
            "network_attempt_count": 0, "gpu_peak_bytes": 0}


def _load_preregistration() -> dict[str, Any]:
    if _git("rev-parse", f"{PREREG_COMMIT}:{PREREG_RELATIVE}") != PREREG_BLOB:
        raise SparseProbeError("preregistration blob drifted", "PREREG_DRIFT")
    raw = _git("cat-file", "blob", PREREG_BLOB, binary=True)
    _guard_experiment_data(PREREG_PATH)
    worktree_raw = _require_plain(PREREG_PATH).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        worktree = json.loads(worktree_raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SparseProbeError("preregistration JSON is invalid", "PREREG_DRIFT") from error
    if not (len(raw) == PREREG_BYTES and hashlib.sha256(raw).hexdigest() == PREREG_RAW_SHA256
            and raw == worktree_raw and _canonical_sha256(value) == PREREG_CANONICAL_SHA256
            and _canonical_sha256(worktree) == PREREG_CANONICAL_SHA256
            and value.get("schema_version") == "small-ranker-registry-ca-g0-preregistration.v1"
            and value.get("status") == "PREREGISTERED_BEFORE_IMPLEMENTATION_AND_OUTCOME"
            and value.get("parent_commit") == BASE_COMMIT):
        raise SparseProbeError("preregistration identity drifted", "PREREG_DRIFT")
    return value


def _validate_git_checkpoint(implementation_commit: str) -> dict[str, Any]:
    if not COMMIT_RE.fullmatch(implementation_commit):
        raise SparseProbeError("implementation commit is invalid", "GIT_CHECKPOINT")
    head = _git("rev-parse", "HEAD")
    pinned = {path: _git("rev-parse", f"HEAD:{path}") for path in PINNED_BLOBS}
    implementation_blobs = {
        path: _git("rev-parse", f"HEAD:{path}") for path in sorted(IMPLEMENTATION_PATHS)
    }
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    worktree_paths = sorted(set(PINNED_BLOBS) | IMPLEMENTATION_PATHS | {PREREG_RELATIVE})
    worktree_equal = all(_worktree_blob(ROOT / path) == _git("rev-parse", f"HEAD:{path}")
                         for path in worktree_paths)
    if not (head == implementation_commit and _git("rev-parse", "HEAD^") == PREREG_COMMIT
            and _git("rev-parse", f"{PREREG_COMMIT}^") == BASE_COMMIT
            and _git("branch", "--show-current") == BRANCH
            and _git("remote", "get-url", REMOTE) == REMOTE_URL
            and _git("rev-parse", REMOTE_REF) == head
            and _changed_paths(PREREG_COMMIT) == PREREG_PATHS
            and _changed_paths(head) == IMPLEMENTATION_PATHS
            and _diff_paths(PREREG_COMMIT, head) == IMPLEMENTATION_PATHS
            and pinned == PINNED_BLOBS and worktree_equal and not status):
        raise SparseProbeError("Git checkpoint gate failed", "GIT_CHECKPOINT")
    return {"branch": BRANCH, "commit": head, "parent": PREREG_COMMIT,
            "remote_equal": True, "clean_including_untracked_nonignored": True,
            "relevant_worktree_equal": True,
            "exact_changed_paths": True, "implementation_blobs": implementation_blobs,
            "pinned_blobs_sha256": _canonical_sha256(pinned)}


def _load_catalog_ids() -> tuple[frozenset[str], FileIdentity]:
    identity = _file_identity(CATALOG_PATH, "catalog")
    if identity.report() != {"bytes": CATALOG_BYTES, "rows": CATALOG_ROWS,
                             "sha256": CATALOG_SHA256}:
        raise SparseProbeError("catalog identity drifted", "SOURCE_DRIFT")
    identifiers: set[str] = set()
    _guard_experiment_data(CATALOG_PATH)
    with CATALOG_PATH.open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            value = json.loads(raw, object_pairs_hook=_unique_object)
            identifier = value.get("parent_asin") if isinstance(value, dict) else None
            if (not isinstance(identifier, str) or not CATALOG_ID_RE.fullmatch(identifier)
                    or identifier != identifier.upper() or identifier in identifiers):
                raise SparseProbeError("catalog identifier surface drifted", "SOURCE_DRIFT")
            identifiers.add(identifier)
    return frozenset(identifiers), identity


def _canonical_reference_line(ordinal: int, turn: int, values: Sequence[str]) -> bytes:
    return _canonical_bytes({"c200": list(values), "ordinal": ordinal, "turn": turn}) + b"\n"


def _canonical_trace_line(ordinal: int, turn: int, values: Sequence[str]) -> bytes:
    return _canonical_bytes({"candidates": list(values), "ordinal": ordinal, "turn": turn}) + b"\n"


def _parse_line(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SparseProbeError(f"{label} JSONL is invalid", "INVALID_TRACE") from error
    if not isinstance(value, dict):
        raise SparseProbeError(f"{label} row is not an object", "INVALID_TRACE")
    return value


def _validate_candidate_row(row: Mapping[str, Any], reference: Sequence[str], index: int,
                            catalog_ids: frozenset[str]) -> tuple[tuple[str, ...], bytes]:
    ordinal, turn = index // TURN_COUNT + 1, index % TURN_COUNT + 1
    if (set(row) != {"candidates", "ordinal", "turn"} or row.get("ordinal") != ordinal
            or isinstance(row.get("ordinal"), bool) or row.get("turn") != turn
            or isinstance(row.get("turn"), bool)):
        raise SparseProbeError("trace schema or order drifted", "INVALID_TRACE")
    raw_values = row.get("candidates")
    if not isinstance(raw_values, list):
        raise SparseProbeError("trace candidates are invalid", "INVALID_TRACE")
    values = tuple(raw_values)
    prefix = tuple(reference)
    if (not 100 <= len(prefix) <= 200 or len(set(prefix)) != len(prefix)
            or len(values) < len(prefix) or len(values) > min(400, len(prefix) + 120)
            or len(values) > 320 or values[:len(prefix)] != prefix
            or len(set(values)) != len(values)
            or any(not isinstance(value, str) or value not in catalog_ids for value in values)):
        raise SparseProbeError("complete variable-C200 prefix gate failed", "PREFIX_GATE")
    return values, _canonical_trace_line(ordinal, turn, values)


def validate_trace_records(records: Sequence[Mapping[str, Any]],
                           c200_reference: Sequence[Sequence[str]],
                           catalog_ids: frozenset[str], *,
                           expected_records: int | None = None) -> TraceValidation:
    expected = len(c200_reference) if expected_records is None else expected_records
    if expected <= 0 or len(records) != expected or len(c200_reference) < expected:
        raise SparseProbeError("trace record count drifted", "INVALID_TRACE")
    digest = hashlib.sha256()
    byte_count = activation_turns = 0
    activation_sessions: set[int] = set()
    lengths: list[int] = []
    c200_lengths: list[int] = []
    normalized: list[dict[str, Any]] = []
    for index in range(expected):
        values, payload = _validate_candidate_row(records[index], c200_reference[index],
                                                  index, catalog_ids)
        digest.update(payload)
        byte_count += len(payload)
        lengths.append(len(values))
        c200_lengths.append(len(c200_reference[index]))
        if len(values) > len(c200_reference[index]):
            activation_turns += 1
            activation_sessions.add(index // TURN_COUNT)
        normalized.append({"candidates": list(values), "ordinal": index // TURN_COUNT + 1,
                           "turn": index % TURN_COUNT + 1})
    return TraceValidation(tuple(normalized), tuple(lengths), tuple(c200_lengths),
                           digest.hexdigest(), byte_count, expected, activation_turns,
                           len(activation_sessions))


def load_and_validate_trace(path: Path, c200_reference: Path | Sequence[Sequence[str]],
                            catalog_ids: frozenset[str], *, session_limit: int = SESSION_COUNT,
                            retain_records: bool = False) -> TraceValidation:
    expected = session_limit * TURN_COUNT
    trace_path = _require_plain(path)
    reference_handle: BinaryIO | None = None
    if isinstance(c200_reference, (str, os.PathLike, Path)):
        reference_path = Path(c200_reference)
        _guard_experiment_data(reference_path)
        reference_handle = _require_plain(reference_path).open("rb")
        reference_rows: Sequence[Sequence[str]] | None = None
    else:
        reference_rows = c200_reference
    digest = hashlib.sha256()
    byte_count = activation_turns = 0
    activation_sessions: set[int] = set()
    lengths: list[int] = []
    c200_lengths: list[int] = []
    retained: list[dict[str, Any]] = []
    try:
        with trace_path.open("rb") as trace_handle:
            before = _snapshot(os.fstat(trace_handle.fileno()))
            for index, raw in enumerate(trace_handle):
                if index >= expected or not raw.strip():
                    raise SparseProbeError("trace framing drifted", "INVALID_TRACE")
                if reference_handle is not None:
                    reference_raw = reference_handle.readline()
                    if not reference_raw:
                        raise SparseProbeError("C200 reference ended early", "SOURCE_DRIFT")
                    reference_row = _parse_line(reference_raw, "C200 reference")
                    ordinal, turn = index // TURN_COUNT + 1, index % TURN_COUNT + 1
                    prefix_value = reference_row.get("c200")
                    if (set(reference_row) != {"c200", "ordinal", "turn"}
                            or reference_row.get("ordinal") != ordinal
                            or reference_row.get("turn") != turn
                            or not isinstance(prefix_value, list)
                            or reference_raw != _canonical_reference_line(ordinal, turn,
                                                                          prefix_value)):
                        raise SparseProbeError("C200 reference schema drifted", "SOURCE_DRIFT")
                    reference = prefix_value
                else:
                    if reference_rows is None or index >= len(reference_rows):
                        raise SparseProbeError("C200 reference ended early", "SOURCE_DRIFT")
                    reference = reference_rows[index]
                row = _parse_line(raw, "candidate trace")
                values, canonical = _validate_candidate_row(row, reference, index, catalog_ids)
                if raw != canonical:
                    raise SparseProbeError("trace is not canonical LF JSON", "INVALID_TRACE")
                digest.update(raw)
                byte_count += len(raw)
                lengths.append(len(values))
                c200_lengths.append(len(reference))
                if len(values) > len(reference):
                    activation_turns += 1
                    activation_sessions.add(index // TURN_COUNT)
                if retain_records:
                    retained.append(dict(row))
            after = _snapshot(os.fstat(trace_handle.fileno()))
        if (len(lengths) != expected or before != after or byte_count != before[0]
                or _snapshot(trace_path.stat()) != after):
            raise SparseProbeError("trace identity or count drifted", "INVALID_TRACE")
    finally:
        if reference_handle is not None:
            reference_handle.close()
    return TraceValidation(tuple(retained), tuple(lengths), tuple(c200_lengths),
                           digest.hexdigest(), byte_count, expected, activation_turns,
                           len(activation_sessions))


def candidate_recall_flags(target: str, eligible_from: int,
                           turns: Sequence[Mapping[str, Any]],
                           cutoffs: Sequence[int] = CUTOFFS, *,
                           baseline_lengths: Sequence[int] | None = None) -> dict[int, bool]:
    if (not isinstance(target, str) or not target or not isinstance(eligible_from, int)
            or isinstance(eligible_from, bool) or not 1 <= eligible_from <= TURN_COUNT
            or tuple(cutoffs) != CUTOFFS or baseline_lengths is None
            or len(baseline_lengths) != len(turns)):
        raise SparseProbeError("candidate recall input is invalid", "AGGREGATE_INPUT")
    result = {cutoff: False for cutoff in CUTOFFS}
    for index, row in enumerate(turns):
        turn, candidates = row.get("turn"), row.get("candidates")
        baseline = baseline_lengths[index]
        if (not isinstance(turn, int) or isinstance(turn, bool)
                or not isinstance(candidates, (list, tuple)) or not 100 <= baseline <= 200):
            raise SparseProbeError("candidate recall row is invalid", "AGGREGATE_INPUT")
        if turn < eligible_from:
            continue
        for cutoff in CUTOFFS:
            limit = baseline if cutoff == 200 else cutoff
            result[cutoff] = result[cutoff] or target in candidates[:limit]
    return result


def _recall_view(flags: Sequence[Mapping[int, bool]], indices: Sequence[int]) -> dict[str, Any]:
    denominator = len(indices)
    return {f"c{cutoff}": {"count": sum(int(flags[index][cutoff]) for index in indices),
                            "fraction": round(sum(int(flags[index][cutoff]) for index in indices)
                                              / denominator, 6) if denominator else 0.0}
            for cutoff in CUTOFFS}


def _exact_target_uniform_raw(flags: Sequence[Mapping[int, bool]],
                              targets: Sequence[str]) -> tuple[float, float, float]:
    if not flags or len(flags) != len(targets):
        raise SparseProbeError("target-uniform dimensions drifted", "AGGREGATE_INPUT")
    target_members: dict[str, list[int]] = {}
    for index, target in enumerate(targets):
        if not isinstance(target, str) or not target:
            raise SparseProbeError("target cluster surface drifted", "AGGREGATE_INPUT")
        target_members.setdefault(target, []).append(index)
    baseline = statistics.fmean(
        statistics.fmean(int(flags[index][200]) for index in members)
        for members in target_members.values()
    )
    candidate = statistics.fmean(
        statistics.fmean(int(flags[index][320]) for index in members)
        for members in target_members.values()
    )
    return baseline, candidate, candidate - baseline


def aggregate_candidate_recall(flags: Sequence[Mapping[int, bool]], *,
                               outer_fold: Sequence[Any], family_index: Sequence[Any],
                               taxonomy: Sequence[str], targets: Sequence[str]) -> dict[str, Any]:
    count = len(flags)
    if not count or not (len(outer_fold) == len(family_index) == len(taxonomy)
                         == len(targets) == count):
        raise SparseProbeError("aggregate dimensions drifted", "AGGREGATE_INPUT")
    if any(set(row) != set(CUTOFFS) or any(type(row[key]) is not bool for key in CUTOFFS)
           for row in flags):
        raise SparseProbeError("aggregate flag schema drifted", "AGGREGATE_INPUT")
    folds = [int(value) for value in outer_fold]
    families = [int(value) for value in family_index]
    if any(not 0 <= value < 5 for value in folds) or any(value < 0 for value in families):
        raise SparseProbeError("fold labels drifted", "AGGREGATE_INPUT")
    family_fold: dict[int, int] = {}
    for family, fold in zip(families, folds, strict=True):
        if family in family_fold and family_fold[family] != fold:
            raise SparseProbeError("family crosses outer folds", "AGGREGATE_INPUT")
        family_fold[family] = fold
    indices = list(range(count))
    frontier = [index for index in indices if not flags[index][200]]
    increment = [index for index in frontier if flags[index][320]]
    by_fold = []
    for fold in sorted(set(folds)):
        members = [index for index, value in enumerate(folds) if value == fold]
        recall = _recall_view(flags, members)
        by_fold.append({"fold": fold, "sessions": len(members), "c200": recall["c200"],
                        "c320_complete_union": recall["c320"],
                        "increment": sum(index in increment for index in members)})
    by_taxonomy = {}
    for name in sorted(set(map(str, taxonomy))):
        members = [index for index, value in enumerate(taxonomy) if str(value) == name]
        recall = _recall_view(flags, members)
        by_taxonomy[name] = {"sessions": len(members), "c200": recall["c200"],
                             "c320_complete_union": recall["c320"],
                             "increment": sum(index in increment for index in members)}
    target_members = {target for target in targets}
    uniform_c200, uniform_c320, uniform_delta = _exact_target_uniform_raw(flags, targets)
    return {
        "all_sessions": _recall_view(flags, indices),
        "c200_absent_frontier": {"sessions": len(frontier),
                                  "c200": _recall_view(flags, frontier)["c200"],
                                  "c320_complete_union": _recall_view(flags, frontier)["c320"]},
        "increment": {"count": len(increment), "outer_fold_span": len({folds[i] for i in increment}),
                      "taxonomy_span": len({str(taxonomy[i]) for i in increment}),
                      "non_clothing_count": sum(str(taxonomy[i]) != "clothing" for i in increment),
                      "target_cluster_count": len({targets[i] for i in increment})},
        "by_outer_fold": by_fold, "by_taxonomy": by_taxonomy,
        "exact_target_cluster_uniform": {
            "cluster_count": len(target_members), "c200_fraction": round(uniform_c200, 9),
            "c320_complete_union_fraction": round(uniform_c320, 9),
            "delta": round(uniform_delta, 9),
        },
        "family_disjoint_audit": {"valid": True, "family_count": len(family_fold),
                                  "families_crossing_outer_folds": 0},
    }


def _offline_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if key.casefold() not in {"pythonpath", "pythonhome"}}
    environment.update({
        "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
    })
    return environment


def _run_subprocess(command: Sequence[str], *, timeout: float,
                    cwd: Path = ROOT) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(list(command), cwd=cwd, env=_offline_environment(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=False, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise SparseProbeError("isolated subprocess timed out", "WORKER_TIMEOUT") from error


def _entrypoint_self_check(required_module: str) -> dict[str, Any]:
    if not required_module or not isinstance(required_module, str):
        raise SparseProbeError("required module is invalid", "ENTRYPOINT_FAILURE")
    importlib.import_module(required_module)
    return {"status": "ENTRYPOINT_SELF_CHECK_PASS", "required_module": required_module,
            "project_root_bootstrapped": str(PROJECT_ROOT) in sys.path}


def _self_check_command(subject: str, mode: str, required_module: str) -> list[str]:
    if subject == "runner":
        direct_path, module_name = RUNNER_PATH, "scripts.probe_sparse_multiview_candidate_recall"
    elif subject == "worker":
        direct_path, module_name = WORKER_PATH, "scripts.sparse_multiview_candidate_worker"
    else:
        raise SparseProbeError("entrypoint subject is invalid", "ENTRYPOINT_FAILURE")
    prefix = ([str(EXPECTED_EXECUTABLE), "-B", str(direct_path)] if mode == "direct"
              else [str(EXPECTED_EXECUTABLE), "-B", "-m", module_name])
    return [*prefix, "--entrypoint-self-check", "--require-module", required_module]


def _validate_self_check(completed: subprocess.CompletedProcess[bytes], label: str) -> None:
    if completed.returncode != 0 or completed.stderr:
        raise SparseProbeError(f"{label} self-check failed", "ENTRYPOINT_FAILURE")
    value = _parse_line(completed.stdout, "entrypoint receipt")
    if (completed.stdout != _canonical_bytes(value) + b"\n"
            or value != {"project_root_bootstrapped": True,
                         "required_module": "evaluator.local_evaluator",
                         "status": "ENTRYPOINT_SELF_CHECK_PASS"}):
        raise SparseProbeError("entrypoint receipt drifted", "ENTRYPOINT_FAILURE")


def _verify_entrypoints_before_receipt() -> dict[str, Any]:
    required = "evaluator.local_evaluator"
    for subject in ("runner", "worker"):
        for mode in ("direct", "module"):
            _validate_self_check(
                _run_subprocess(_self_check_command(subject, mode, required), timeout=60.0),
                f"{subject}-{mode}",
            )
        _validate_self_check(
            _run_subprocess(_self_check_command(subject, "direct", required), timeout=60.0,
                            cwd=ROOT.parent),
            f"{subject}-direct-outside-cwd",
        )
    missing = "v219_intentionally_absent_required_module"
    for subject in ("runner", "worker"):
        for mode in ("direct", "module"):
            completed = _run_subprocess(_self_check_command(subject, mode, missing), timeout=60.0,
                                        cwd=ROOT.parent if mode == "direct" else ROOT)
            if completed.returncode == 0:
                raise SparseProbeError("missing-module regression failed open",
                                       "ENTRYPOINT_FAILURE")
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise SparseProbeError("entrypoint check created formal receipt", "RECEIPT_PREEXISTS")
    return {"runner_direct": True, "runner_module": True, "worker_direct": True,
            "worker_module": True, "direct_outside_repository_cwd": True,
            "missing_module_failed_nonzero": True, "receipt_created": False}


def _worker_command(*, mode: str, nonce: str, reference: Path, trace: Path,
                    session_limit: int) -> list[str]:
    if mode == "direct":
        prefix = [str(EXPECTED_EXECUTABLE), "-B", str(WORKER_PATH)]
    elif mode == "module":
        prefix = [str(EXPECTED_EXECUTABLE), "-B", "-m",
                  "scripts.sparse_multiview_candidate_worker"]
    else:
        raise SparseProbeError("worker invocation mode is invalid", "WORKER_CONTRACT")
    return [*prefix, "--catalog", str(CATALOG_PATH), "--context", str(CONTEXT_PATH),
            "--c200-reference", str(reference), "--trace-output", str(trace),
            "--session-limit", str(session_limit), "--nonce", nonce]


def _finite(value: object, label: str) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or float(value) < 0):
        raise SparseProbeError(f"worker {label} is invalid", "WORKER_CONTRACT")
    return float(value)


def _lookup_p95(latency: Mapping[str, Any], names: Sequence[str]) -> float:
    for name in names:
        value = latency.get(name)
        if isinstance(value, Mapping) and "p95_milliseconds" in value:
            return _finite(value["p95_milliseconds"], name)
        if name in latency and isinstance(value, (int, float)):
            return _finite(value, name)
    raise SparseProbeError("worker latency summary is incomplete", "WORKER_CONTRACT")


def _validate_worker_receipt(payload: bytes, *, nonce: str,
                             session_limit: int) -> dict[str, Any]:
    value = _parse_line(payload, "worker receipt")
    if payload != _canonical_bytes(value) + b"\n":
        raise SparseProbeError("worker receipt is not canonical LF JSON", "WORKER_CONTRACT")
    summary = value.get("summary")
    if not (set(value) == {"schema_version", "kind", "status", "phase", "error_code",
                           "nonce", "trace_bytes", "trace_sha256", "record_count",
                           "last_completed_session", "summary"}
            and value.get("schema_version") == WORKER_SCHEMA_VERSION
            and value.get("kind") == "receipt" and value.get("status") == "SUCCESS"
            and value.get("phase") == "COMPLETE" and value.get("error_code") == "NONE"
            and value.get("nonce") == nonce and value.get("last_completed_session") == session_limit
            and value.get("record_count") == session_limit * TURN_COUNT
            and isinstance(value.get("trace_bytes"), int) and value["trace_bytes"] > 0
            and isinstance(value.get("trace_sha256"), str)
            and DIGEST_RE.fullmatch(value["trace_sha256"])
            and isinstance(summary, Mapping) and summary.get("session_limit") == session_limit
            and summary.get("processed_sessions") == session_limit
            and summary.get("processed_turns") == session_limit * TURN_COUNT):
        raise SparseProbeError("worker receipt contract drifted", "WORKER_CONTRACT")
    activation, mask = summary.get("activation"), summary.get("mask")
    latency, resources = summary.get("latency"), summary.get("resources")
    lifecycle, inputs = summary.get("lifecycle"), summary.get("input_identities")
    if not (isinstance(activation, Mapping) and isinstance(mask, Mapping)
            and isinstance(latency, Mapping) and isinstance(resources, Mapping)
            and isinstance(lifecycle, Mapping) and lifecycle
            and all(flag is True for flag in lifecycle.values()) and isinstance(inputs, Mapping)):
        raise SparseProbeError("worker summary is incomplete", "WORKER_CONTRACT")
    activation_count = int(activation.get("activated_records", 0))
    conflicts = int(mask.get("tail_explicit_conflict_count", 0))
    duplicates = int(mask.get("tail_duplicate_count", 0))
    network = int(resources.get("network_attempt_count", 0))
    gpu = int(resources.get("gpu_peak_bytes", 0))
    rss = resources.get("peak_working_set_bytes")
    wall = _finite(resources.get("wall_seconds"), "wall")
    route_p95 = _lookup_p95(latency, ("extra_route_and_mask", "route_and_mask", "extra_route"))
    turn_p95 = _lookup_p95(latency, ("per_turn", "total_per_turn", "turn"))
    if (activation_count <= 0 or conflicts != 0 or duplicates != 0 or network != 0 or gpu != 0
            or not isinstance(rss, int) or isinstance(rss, bool) or not 0 < rss <= WORKER_RSS_MAXIMUM
            or wall > TOTAL_WALL_MAXIMUM or route_p95 > EXTRA_ROUTE_P95_MS_MAXIMUM
            or turn_p95 > PER_TURN_P95_MS_MAXIMUM):
        raise SparseProbeError("worker integrity or resource gate failed", "RESOURCE_GATE")
    return dict(value)


def _bounded_nonnegative_int(value: object, maximum: int, label: str) -> int:
    if (not isinstance(value, int) or isinstance(value, bool)
            or not 0 <= value <= maximum):
        raise SparseProbeError(f"worker {label} is invalid", "WORKER_ERROR_CONTRACT")
    return value


def _validate_identity_triplet(value: object, label: str, *, maximum_bytes: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"bytes", "rows", "sha256"}:
        raise SparseProbeError(f"worker {label} identity schema drifted",
                               "WORKER_ERROR_CONTRACT")
    byte_count = _bounded_nonnegative_int(value.get("bytes"), maximum_bytes,
                                          f"{label} bytes")
    row_count = _bounded_nonnegative_int(value.get("rows"), RECORD_COUNT,
                                         f"{label} rows")
    digest = value.get("sha256")
    if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
        raise SparseProbeError(f"worker {label} digest drifted", "WORKER_ERROR_CONTRACT")
    return {"bytes": byte_count, "rows": row_count, "sha256": digest}


def _validate_worker_error_receipt(payload: bytes, *, nonce: str, session_limit: int,
                                   captured_stderr: bytes) -> dict[str, Any]:
    value = _parse_line(payload, "worker error receipt")
    expected_keys = {
        "error_code", "kind", "last_completed_session", "nonce", "partial_trace",
        "phase", "resources", "schema_version", "source_identities", "status",
        "traceback",
    }
    if (payload != _canonical_bytes(value) + b"\n" or set(value) != expected_keys
            or value.get("schema_version") != WORKER_SCHEMA_VERSION
            or value.get("kind") != "receipt" or value.get("status") != "ERROR"
            or value.get("nonce") != nonce):
        raise SparseProbeError("worker ERROR receipt envelope drifted",
                               "WORKER_ERROR_CONTRACT")
    phase, error_code = value.get("phase"), value.get("error_code")
    if (not isinstance(phase, str) or not ERROR_ENUM_RE.fullmatch(phase)
            or not isinstance(error_code, str) or not ERROR_ENUM_RE.fullmatch(error_code)
            or error_code == "NONE"):
        raise SparseProbeError("worker ERROR enum drifted", "WORKER_ERROR_CONTRACT")
    last = _bounded_nonnegative_int(value.get("last_completed_session"), session_limit,
                                    "last completed session")
    partial = _validate_identity_triplet(
        value.get("partial_trace"), "partial trace",
        maximum_bytes=math.ceil(C200_TRACE_BYTES * TRACE_RATIO_MAXIMUM),
    )
    if partial["rows"] > min(session_limit * TURN_COUNT, (last + 1) * TURN_COUNT):
        raise SparseProbeError("worker partial trace progress drifted",
                               "WORKER_ERROR_CONTRACT")

    resources = value.get("resources")
    resource_keys = {"gpu_peak_bytes", "network_attempt_count", "peak_working_set_backend",
                     "peak_working_set_bytes", "wall_seconds"}
    if not isinstance(resources, Mapping) or set(resources) != resource_keys:
        raise SparseProbeError("worker ERROR resources drifted", "WORKER_ERROR_CONTRACT")
    gpu = _bounded_nonnegative_int(resources.get("gpu_peak_bytes"), 0, "GPU bytes")
    network = _bounded_nonnegative_int(resources.get("network_attempt_count"), 1_000_000,
                                       "network attempts")
    peak = _bounded_nonnegative_int(resources.get("peak_working_set_bytes"), 1 << 63,
                                    "peak working set")
    wall = _finite(resources.get("wall_seconds"), "ERROR wall")
    backend = resources.get("peak_working_set_backend")
    if wall > TOTAL_WALL_MAXIMUM * 2 or not isinstance(backend, str) or not backend \
            or len(backend) > 64 or not SAFE_NAME_RE.fullmatch(backend):
        raise SparseProbeError("worker ERROR resource bounds drifted",
                               "WORKER_ERROR_CONTRACT")

    actual_stderr = {"bytes": len(captured_stderr),
                     "sha256": hashlib.sha256(captured_stderr).hexdigest()}
    if actual_stderr["bytes"] > 1 << 20:
        raise SparseProbeError("worker ERROR stderr exceeds diagnostic bound",
                               "WORKER_ERROR_CONTRACT")

    traceback_value = value.get("traceback")
    if (not isinstance(traceback_value, Mapping)
            or set(traceback_value) != {"exception_type", "sha256", "top_frame"}):
        raise SparseProbeError("worker ERROR traceback schema drifted",
                               "WORKER_ERROR_CONTRACT")
    exception_type = traceback_value.get("exception_type")
    traceback_sha = traceback_value.get("sha256")
    top_frame = traceback_value.get("top_frame")
    if (not isinstance(exception_type, str) or not SAFE_NAME_RE.fullmatch(exception_type)
            or not isinstance(traceback_sha, str) or not DIGEST_RE.fullmatch(traceback_sha)
            or not isinstance(top_frame, Mapping)
            or set(top_frame) != {"file", "function", "line"}):
        raise SparseProbeError("worker ERROR traceback values drifted",
                               "WORKER_ERROR_CONTRACT")
    file_name, function_name = top_frame.get("file"), top_frame.get("function")
    line = _bounded_nonnegative_int(top_frame.get("line"), 10_000_000, "top-frame line")
    if (not isinstance(file_name, str) or Path(file_name).name != file_name
            or not SAFE_NAME_RE.fullmatch(file_name) or not isinstance(function_name, str)
            or not SAFE_NAME_RE.fullmatch(function_name)):
        raise SparseProbeError("worker ERROR top frame is unsafe", "WORKER_ERROR_CONTRACT")

    source_identities = value.get("source_identities")
    implementation_sources = {
        "preregistration", "scripts/sparse_multiview_candidate_worker.py",
        "starter/sparse_multiview.py",
    }
    allowed_sources = implementation_sources | set(PINNED_BLOBS)
    if (not isinstance(source_identities, Mapping)
            or not set(source_identities).issubset(allowed_sources)):
        raise SparseProbeError("worker ERROR source schema drifted", "WORKER_ERROR_CONTRACT")
    for name, identity in source_identities.items():
        expected_keys = {"bytes", "rows", "sha256"}
        if name in PINNED_BLOBS:
            expected_keys.add("raw_git_blob_sha1")
        if not isinstance(identity, Mapping) or set(identity) != expected_keys:
            raise SparseProbeError("worker ERROR source identity schema drifted",
                                   "WORKER_ERROR_CONTRACT")
        _validate_identity_triplet(
            {key: identity[key] for key in ("bytes", "rows", "sha256")},
            f"source {name}", maximum_bytes=100_000_000,
        )
        if (name in PINNED_BLOBS
                and identity.get("raw_git_blob_sha1") != PINNED_BLOBS[name]):
            raise SparseProbeError("worker ERROR pinned source blob drifted",
                                   "WORKER_ERROR_CONTRACT")

    return {
        "phase": phase, "error_code": error_code, "exception_class": exception_type,
        "traceback_sha256": traceback_sha,
        "top_frame": {"file": file_name, "function": function_name, "line": line},
        "last_completed_session": last, "partial_trace": partial,
        "stderr": actual_stderr,
        "resources": {"gpu_peak_bytes": gpu, "network_attempt_count": network,
                      "peak_working_set_bytes": peak, "wall_seconds": wall,
                      "peak_working_set_backend": backend},
    }


def _run_worker(*, mode: str, nonce: str, reference: Path, trace: Path,
                session_limit: int) -> dict[str, Any]:
    started = time.perf_counter()
    completed = _run_subprocess(
        _worker_command(mode=mode, nonce=nonce, reference=reference, trace=trace,
                        session_limit=session_limit),
        timeout=TOTAL_WALL_MAXIMUM,
    )
    stderr = {"bytes": len(completed.stderr), "sha256": hashlib.sha256(completed.stderr).hexdigest()}
    if completed.returncode != 0:
        # Never surface worker exception text, identifiers, messages, or partial rows.
        try:
            diagnostic = _validate_worker_error_receipt(
                completed.stdout, nonce=nonce, session_limit=session_limit,
                captured_stderr=completed.stderr,
            )
        except BaseException as validation_error:
            diagnostic = {"phase": "WORKER", "error_code": "INVALID_ERROR_RECEIPT",
                          "traceback_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                          "validation_error_code": validation_error.code
                          if isinstance(validation_error, SparseProbeError) else
                          "UNEXPECTED_EXCEPTION",
                          "last_completed_session": 0,
                          "partial_trace": {"bytes": 0, "rows": 0,
                                            "sha256": hashlib.sha256(b"").hexdigest()},
                          "stderr": stderr}
        failure = SparseProbeError("isolated worker failed", "WORKER_FAILURE")
        failure.diagnostic = diagnostic  # type: ignore[attr-defined]
        raise failure
    receipt = _validate_worker_receipt(completed.stdout, nonce=nonce,
                                       session_limit=session_limit)
    return {"receipt": receipt, "mode": mode,
            "wall_seconds": round(time.perf_counter() - started, 6), "stderr": stderr}


def _bind_worker_receipt(result: Mapping[str, Any], trace: TraceValidation) -> None:
    receipt = result.get("receipt")
    if not (isinstance(receipt, Mapping)
            and receipt.get("trace_sha256") == trace.canonical_trace_sha256
            and receipt.get("trace_bytes") == trace.canonical_trace_bytes
            and receipt.get("record_count") == trace.record_count):
        raise SparseProbeError("worker receipt does not bind to trace", "TRACE_BINDING")


def _same_trace_validation(left: TraceValidation, right: TraceValidation) -> bool:
    return (
        left.canonical_trace_sha256 == right.canonical_trace_sha256
        and left.canonical_trace_bytes == right.canonical_trace_bytes
        and left.record_count == right.record_count
        and left.lengths == right.lengths
        and left.c200_lengths == right.c200_lengths
        and left.activation_turns == right.activation_turns
        and left.activation_sessions == right.activation_sessions
    )


def _files_equal(left: Path, right: Path) -> bool:
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            a, b = first.read(1 << 20), second.read(1 << 20)
            if a != b:
                return False
            if not a:
                return True


def _run_pair(*, root: Path, session_limit: int) -> tuple[dict[str, dict[str, Any]],
                                                           tuple[Path, Path]]:
    traces = (root / "replica_a.jsonl", root / "replica_b.jsonl")
    results: dict[str, dict[str, Any]] = {}
    for index, (name, mode) in enumerate((("replica_a", "direct"),
                                          ("replica_b", "module"))):
        nonce = hashlib.sha256(os.urandom(32) + name.encode("ascii")
                               + str(session_limit).encode("ascii")).hexdigest()[:32]
        results[name] = _run_worker(mode=mode, nonce=nonce,
                                    reference=C200_REFERENCE_PATHS[index], trace=traces[index],
                                    session_limit=session_limit)
    return results, traces


def _trace_pair_gate(results: Mapping[str, Mapping[str, Any]], traces: Sequence[Path],
                     catalog_ids: frozenset[str], session_limit: int) -> TraceValidation:
    first = load_and_validate_trace(traces[0], C200_REFERENCE_PATHS[0], catalog_ids,
                                    session_limit=session_limit)
    second = load_and_validate_trace(traces[1], C200_REFERENCE_PATHS[1], catalog_ids,
                                     session_limit=session_limit)
    _bind_worker_receipt(results["replica_a"], first)
    _bind_worker_receipt(results["replica_b"], second)
    if not (first.canonical_trace_sha256 == second.canonical_trace_sha256
            and first.canonical_trace_bytes == second.canonical_trace_bytes
            and first.lengths == second.lengths and first.c200_lengths == second.c200_lengths
            and first.activation_turns > 0 and first.activation_sessions > 0
            and _files_equal(traces[0], traces[1])):
        raise SparseProbeError("direct/module traces are not exact repeats", "EXACT_REPEAT")
    return first


def _validate_reference_sources(catalog_ids: frozenset[str]) -> dict[str, Any]:
    reports = []
    for path in C200_REFERENCE_PATHS:
        identity = _file_identity(path, "sealed C200 reference")
        if identity.report() != {"bytes": C200_TRACE_BYTES, "rows": C200_TRACE_ROWS,
                                 "sha256": C200_TRACE_SHA256}:
            raise SparseProbeError("sealed C200 identity drifted", "SOURCE_DRIFT")
        reports.append(identity.report())
    _guard_experiment_data(C200_REFERENCE_PATHS[0])
    _guard_experiment_data(C200_REFERENCE_PATHS[1])
    if not _files_equal(*C200_REFERENCE_PATHS):
        raise SparseProbeError("sealed C200 replicas differ", "SOURCE_DRIFT")
    cells = c100_bytes = 0
    c100_digest = hashlib.sha256()
    with C200_REFERENCE_PATHS[0].open("rb") as handle:
        for index, raw in enumerate(handle):
            row = _parse_line(raw, "sealed C200")
            ordinal, turn = index // TURN_COUNT + 1, index % TURN_COUNT + 1
            values = row.get("c200")
            if (set(row) != {"c200", "ordinal", "turn"} or row.get("ordinal") != ordinal
                    or row.get("turn") != turn or not isinstance(values, list)
                    or not 100 <= len(values) <= 200 or len(set(values)) != len(values)
                    or any(value not in catalog_ids for value in values)
                    or raw != _canonical_reference_line(ordinal, turn, values)):
                raise SparseProbeError("sealed C200 schema drifted", "SOURCE_DRIFT")
            cells += len(values)
            normalized = _canonical_bytes({"c100": values[:100], "ordinal": ordinal,
                                           "turn": turn}) + b"\n"
            c100_digest.update(normalized)
            c100_bytes += len(normalized)
    if not (cells == C200_CANDIDATE_CELLS and c100_bytes == C100_NORMALIZED_BYTES
            and c100_digest.hexdigest() == C100_NORMALIZED_SHA256):
        raise SparseProbeError("sealed C200 normalized identity drifted", "SOURCE_DRIFT")
    return {"replicas": reports, "candidate_cells": cells,
            "normalized_c100_bytes": c100_bytes,
            "normalized_c100_sha256": c100_digest.hexdigest()}


def _source_checkpoint(catalog_ids: frozenset[str] | None = None) -> tuple[frozenset[str],
                                                                            dict[str, Any]]:
    if catalog_ids is None:
        catalog_ids, catalog = _load_catalog_ids()
    else:
        catalog = _file_identity(CATALOG_PATH, "catalog rehash")
    context = _file_identity(CONTEXT_PATH, "visible context")
    if context.report() != {"bytes": CONTEXT_BYTES, "rows": CONTEXT_ROWS,
                            "sha256": CONTEXT_SHA256}:
        raise SparseProbeError("visible context identity drifted", "SOURCE_DRIFT")
    references = _validate_reference_sources(catalog_ids)
    return catalog_ids, {"catalog": catalog.report(), "visible_context": context.report(),
                         "sealed_c200": references,
                         "worker_blob": _git("rev-parse", "HEAD:scripts/sparse_multiview_candidate_worker.py"),
                         "runner_blob": _git("rev-parse", "HEAD:scripts/probe_sparse_multiview_candidate_recall.py")}


def _smoke(catalog_ids: frozenset[str], implementation_commit: str) -> dict[str, Any]:
    parent = _require_plain(OUTPUT_PATH.parent, directory=True)
    stages = []
    for session_limit in (20, 100):
        with tempfile.TemporaryDirectory(prefix=f"v219_smoke_{session_limit}_", dir=parent) as name:
            results, traces = _run_pair(root=Path(name), session_limit=session_limit)
            validation = _trace_pair_gate(results, traces, catalog_ids, session_limit)
            wall = sum(float(results[key]["wall_seconds"]) for key in sorted(results))
            stages.append({"sessions": session_limit, "record_count": validation.record_count,
                           "trace_sha256": validation.canonical_trace_sha256,
                           "activation_turns": validation.activation_turns,
                           "direct_module_exact_repeat": True,
                           "pair_wall_seconds": round(wall, 6)})
        _source_checkpoint(catalog_ids)
    extrapolated = stages[-1]["pair_wall_seconds"] * (SESSION_COUNT / 100) * 1.5
    if extrapolated > TOTAL_WALL_MAXIMUM:
        raise SparseProbeError("100-session wall extrapolation failed", "RESOURCE_GATE")
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise SparseProbeError("smoke created formal receipt", "RECEIPT_PREEXISTS")
    return {"stages": stages, "linear_pair_wall_x1_5_seconds": round(extrapolated, 6),
            "receipt_created": False, "nonce_temp_paths": True,
            "implementation_commit_used_for_identity_only": bool(implementation_commit)}


def _assert_fresh_outputs() -> None:
    parent = _require_plain(OUTPUT_PATH.parent, directory=True)
    if OUTPUT_PATH.resolve(strict=False).parent != parent:
        raise SparseProbeError("receipt parent identity drifted", "UNSAFE_PATH")
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise SparseProbeError("formal receipt is already consumed", "RECEIPT_PREEXISTS")
    if CACHE_ROOT.exists() or CACHE_ROOT.is_symlink():
        raise SparseProbeError("formal cache is already consumed", "CACHE_PREEXISTS")
    if shutil.disk_usage(parent).free < FREE_DISK_MINIMUM:
        raise SparseProbeError("formal free-disk gate failed", "RESOURCE_GATE")


def preflight_only(implementation_commit: str) -> Preflight:
    """Run every target-free gate, including the 20 then 100 session smokes."""

    audit_guard = _install_process_audit_guard()
    try:
        _assert_fresh_outputs()
        policy = _validate_path_policy()
        environment = _validate_environment()
        protocol = _load_preregistration()
        git = _validate_git_checkpoint(implementation_commit)
        catalog_ids, sources = _source_checkpoint()
        entrypoints = _verify_entrypoints_before_receipt()
        smoke = _smoke(catalog_ids, implementation_commit)
        _, repeated_sources = _source_checkpoint(catalog_ids)
        if sources != repeated_sources:
            raise SparseProbeError("target-free sources changed during preflight", "SOURCE_MUTATION")
        if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
            raise SparseProbeError("preflight created formal receipt", "RECEIPT_PREEXISTS")
        return Preflight(environment, git,
                         {"schema_version": protocol["schema_version"], "commit": PREREG_COMMIT,
                          "canonical_sha256": PREREG_CANONICAL_SHA256, "path_policy": policy},
                         catalog_ids, sources, entrypoints, smoke)
    finally:
        audit_guard.close()


def _write_descriptor(descriptor: int, value: object) -> tuple[int, str]:
    payload = _canonical_bytes(value) + b"\n"
    if len(payload) > RECEIPT_BYTES_MAXIMUM:
        raise SparseProbeError("formal receipt is not compact", "RECEIPT_WRITE")
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


def _seal_receipt_size_estimate(value: dict[str, Any]) -> int:
    value["receipt_size"] = {"canonical_bytes_estimate": 0,
                             "maximum_bytes": RECEIPT_BYTES_MAXIMUM}
    for _ in range(4):
        estimate = len(_canonical_bytes(value)) + 1
        if value["receipt_size"]["canonical_bytes_estimate"] == estimate:
            break
        value["receipt_size"]["canonical_bytes_estimate"] = estimate
    final_estimate = len(_canonical_bytes(value)) + 1
    if final_estimate != value["receipt_size"]["canonical_bytes_estimate"]:
        value["receipt_size"]["canonical_bytes_estimate"] = final_estimate
        final_estimate = len(_canonical_bytes(value)) + 1
    if final_estimate > RECEIPT_BYTES_MAXIMUM:
        raise SparseProbeError("formal receipt size estimate exceeds maximum", "RECEIPT_WRITE")
    return final_estimate


def _pending_receipt(implementation_commit: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID,
            "status": "PENDING_ONE_SHOT_CONSUMED", "implementation_commit": implementation_commit,
            "preregistration_commit": PREREG_COMMIT, "recorded_on": "2026-08-31",
            "rerun_forbidden": True}


def _open_receipt(implementation_commit: str) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(str(OUTPUT_PATH), flags, 0o600)
    except OSError as error:
        raise SparseProbeError("exclusive receipt creation failed", "RECEIPT_PREEXISTS") from error
    try:
        _write_descriptor(descriptor, _pending_receipt(implementation_commit))
        return descriptor
    except BaseException as error:
        try:
            _write_invalid_receipt(descriptor, implementation_commit, error,
                                   phase="PENDING_RECEIPT_WRITE")
        finally:
            pass
        raise


def _invalid_value(implementation_commit: str, error: BaseException, phase: str) -> dict[str, Any]:
    frames = traceback.extract_tb(error.__traceback__)
    canonical_traceback = "|".join(f"{frame.name}:{frame.lineno}" for frame in frames)
    code = error.code if isinstance(error, SparseProbeError) else "UNEXPECTED_EXCEPTION"
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID,
        "status": "INVALID_ONE_SHOT_CONSUMED", "phase": phase, "error_code": code,
        "exception_class": type(error).__name__,
        "traceback_sha256": hashlib.sha256(canonical_traceback.encode()).hexdigest(),
        "implementation_commit": implementation_commit, "preregistration_commit": PREREG_COMMIT,
        "recorded_on": "2026-08-31", "rerun_forbidden": True,
        "algorithm_interpretation": "implementation_or_integrity_failure_not_algorithm_no_go",
        "compliance": {
            "overall_orchestration_clean": False,
            "pre_preregistration_boundary_event":
                "PRE_PREREG_PROTECTED_PATH_ACCESS_INCIDENT_RECORDED",
            "qualification_does_not_erase_prior_incident": True,
        },
    }
    diagnostic = getattr(error, "diagnostic", None)
    if isinstance(diagnostic, Mapping):
        encoded = _canonical_bytes(diagnostic)
        if len(encoded) <= 4_096:
            value["worker_diagnostic"] = dict(diagnostic)
    _seal_receipt_size_estimate(value)
    return value


def _write_invalid_receipt(descriptor: int, implementation_commit: str,
                           error: BaseException, *, phase: str) -> None:
    value = _invalid_value(implementation_commit, error, phase)
    try:
        for attempt in range(2):
            try:
                _write_descriptor(descriptor, value)
                return
            except BaseException:
                if attempt:
                    raise
    finally:
        os.close(descriptor)


def _prepare_cache_root() -> None:
    parent = _require_plain(CACHE_ROOT.parent, directory=True)
    try:
        os.mkdir(CACHE_ROOT)
    except OSError as error:
        raise SparseProbeError("exclusive formal cache creation failed", "CACHE_PREEXISTS") from error
    if _is_reparse(CACHE_ROOT) or CACHE_ROOT.resolve(strict=True).parent != parent:
        raise SparseProbeError("formal cache root is unsafe", "UNSAFE_PATH")


def _inflation(trace: TraceValidation) -> dict[str, Any]:
    cells, baseline = sum(trace.lengths), sum(trace.c200_lengths)
    value = {"candidate_cells": cells, "sealed_c200_candidate_cells": baseline,
             "candidate_cells_union_over_c200": round(cells / baseline, 9),
             "canonical_trace_bytes_union_over_c200": round(
                 trace.canonical_trace_bytes / C200_TRACE_BYTES, 9),
             "length_min": min(trace.lengths), "length_max": max(trace.lengths),
             "length_p95": int(sorted(trace.lengths)[math.ceil(.95 * len(trace.lengths)) - 1])}
    if (value["candidate_cells_union_over_c200"] > CELL_RATIO_MAXIMUM
            or value["canonical_trace_bytes_union_over_c200"] > TRACE_RATIO_MAXIMUM):
        raise SparseProbeError("candidate inflation gate failed", "RESOURCE_GATE")
    return value


def _flags_from_trace(path: Path, reference: Path, catalog_ids: frozenset[str],
                      targets: Sequence[str], eligibility: Sequence[int],
                      ) -> tuple[list[dict[int, bool]], TraceValidation]:
    validation = load_and_validate_trace(path, reference, catalog_ids,
                                         session_limit=SESSION_COUNT, retain_records=True)
    flags = []
    for index in range(SESSION_COUNT):
        start = index * TURN_COUNT
        flags.append(candidate_recall_flags(
            targets[index], int(eligibility[index]),
            validation.records[start:start + TURN_COUNT],
            baseline_lengths=validation.c200_lengths[start:start + TURN_COUNT],
        ))
    return flags, validation


def _result_privacy_scan(value: object, catalog_ids: Iterable[str]) -> None:
    catalog = frozenset(str(identifier).casefold() for identifier in catalog_ids)

    def walk(item: object) -> Iterable[object]:
        yield item
        if isinstance(item, Mapping):
            if {str(key).casefold() for key in item} & FORBIDDEN_RESULT_KEYS:
                raise SparseProbeError("result contains identity-bearing key", "PRIVACY_GATE")
            for child in item.values():
                yield from walk(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            if len(item) >= SESSION_COUNT:
                raise SparseProbeError("result contains session-length vector", "PRIVACY_GATE")
            for child in item:
                yield from walk(child)

    for item in walk(value):
        if isinstance(item, str) and (
            ASIN_SHAPE_RE.search(item) or item.casefold() in catalog
        ):
            raise SparseProbeError("result contains catalog identifier", "PRIVACY_GATE")
        if item.__class__.__module__.startswith("numpy"):
            raise SparseProbeError("result contains numeric array", "PRIVACY_GATE")


def _run_guarded(implementation_commit: str,
                 audit_guard: _ProcessAuditGuard) -> dict[str, Any]:
    """Consume the fixed formal receipt and never permit a rerun."""

    started = time.perf_counter()
    preflight = preflight_only(implementation_commit)
    # Keep this explicit so a substituted preflight cannot bypass the final
    # missing-module regression before the durable receipt.
    final_entrypoints = _verify_entrypoints_before_receipt()
    descriptor: int | None = None
    proxy_source: Any | None = None
    phase = "RECEIPT_INITIALIZATION"
    try:
        # No path-producing or experiment operation is permitted between this
        # repeated checkpoint and the fixed O_EXCL receipt creation.
        receipt_git = _validate_git_checkpoint(implementation_commit)
        _, receipt_sources = _source_checkpoint(preflight.catalog_ids)
        if receipt_git != preflight.git or receipt_sources != preflight.source_identities:
            raise SparseProbeError("pre-receipt checkpoint changed", "SOURCE_MUTATION")
        descriptor = _open_receipt(implementation_commit)
        phase = "SEQUENTIAL_FRESH_WORKERS"
        _prepare_cache_root()
        worker_results, traces = _run_pair(root=CACHE_ROOT, session_limit=SESSION_COUNT)

        phase = "CLOSED_TRACE_GATES"
        trace = _trace_pair_gate(worker_results, traces, preflight.catalog_ids, SESSION_COUNT)
        inflation = _inflation(trace)
        if (sum(int(worker_results[name]["receipt"]["summary"]["resources"]
                   ["peak_working_set_bytes"]) for name in worker_results)
                > 2 * WORKER_RSS_MAXIMUM):
            raise SparseProbeError("sequential worker resource gate failed", "RESOURCE_GATE")
        _, before_attach_sources = _source_checkpoint(preflight.catalog_ids)
        before_attach_git = _validate_git_checkpoint(implementation_commit)
        if before_attach_sources != preflight.source_identities:
            raise SparseProbeError("source changed before target attach", "SOURCE_MUTATION")
        if time.perf_counter() - started > TOTAL_WALL_MAXIMUM:
            raise SparseProbeError("formal wall gate failed before attach", "RESOURCE_GATE")

        # First target-side import and first target-side opens.  The pinned C200
        # helpers enforce same-handle proxy and label identities.
        phase = "POST_GATE_PROXY_AND_FOLD_ATTACH"
        audit_guard.allow_post_gate_targets()
        c200 = importlib.import_module("scripts.probe_c200_candidate_recall")
        catalog_ids, products, _categories, catalog_identity = c200._load_catalog_target_free()
        if (catalog_ids != preflight.catalog_ids
                or catalog_identity.report() != preflight.source_identities["catalog"]):
            raise SparseProbeError("post-receipt catalog identity drifted", "SOURCE_MUTATION")
        _guard_experiment_data(PROXY_PATH, post_receipt=True)
        proxy_source = c200._open_proxy_after_receipt(PROXY_PATH)
        targets, eligibility, taxonomy = c200._derive_target_membership_inputs(
            proxy_source.samples, products)
        proxy_identity = c200._reverify_and_close_proxy(proxy_source)
        proxy_source = None
        _guard_experiment_data(LABEL_PATH, post_receipt=True)
        outer_fold, family_index, label_identity = c200._load_fold_labels_after_traces(LABEL_PATH)

        phase = "ANONYMOUS_RECALL_AGGREGATION"
        flags, outcome_trace_a = _flags_from_trace(
            TRACE_PATHS[0], C200_REFERENCE_PATHS[0], preflight.catalog_ids,
            targets, eligibility,
        )
        _bind_worker_receipt(worker_results["replica_a"], outcome_trace_a)
        outcome_trace_b = load_and_validate_trace(
            TRACE_PATHS[1], C200_REFERENCE_PATHS[1], preflight.catalog_ids,
            session_limit=SESSION_COUNT,
        )
        _bind_worker_receipt(worker_results["replica_b"], outcome_trace_b)
        if (not _same_trace_validation(trace, outcome_trace_a)
                or not _same_trace_validation(trace, outcome_trace_b)
                or not _same_trace_validation(outcome_trace_a, outcome_trace_b)
                or not _files_equal(TRACE_PATHS[0], TRACE_PATHS[1])):
            raise SparseProbeError("post-attach traces changed or lost exact repeat",
                                   "POST_ATTACH_TRACE_MUTATION")
        uniform_raw_delta = _exact_target_uniform_raw(flags, targets)[2]
        aggregate = aggregate_candidate_recall(flags, outer_fold=outer_fold,
                                               family_index=family_index, taxonomy=taxonomy,
                                               targets=targets)
        del flags, targets, eligibility, products, catalog_ids
        sanity = aggregate["all_sessions"]
        if any(sanity[f"c{cutoff}"]["count"] != expected
               for cutoff, expected in EXPECTED_C200_RECALL.items()):
            raise SparseProbeError("sealed C200 recall sanity drifted", "BASELINE_SANITY")
        if sanity["c320"]["count"] < sanity["c200"]["count"]:
            raise SparseProbeError("candidate recall is not monotone", "PREFIX_GATE")

        phase = "FINAL_SOURCE_RESOURCE_PRIVACY_GATES"
        _, final_sources = _source_checkpoint(preflight.catalog_ids)
        final_git = _validate_git_checkpoint(implementation_commit)
        if final_sources != before_attach_sources or final_git != before_attach_git:
            raise SparseProbeError("source or Git identity changed", "SOURCE_MUTATION")
        total_wall = time.perf_counter() - started
        if total_wall > TOTAL_WALL_MAXIMUM:
            raise SparseProbeError("formal total wall gate failed", "RESOURCE_GATE")
        increment = aggregate["increment"]
        promoted = bool(
            sanity["c320"]["count"] >= 1_988
            and increment["outer_fold_span"] >= 2
            and increment["non_clothing_count"] >= 1
            and uniform_raw_delta > 0
        )
        status = ("REGISTRY_CA_G0_RECALL_GO_ALLOW_PREREGISTERED_POLICY_SMOKE"
                  if promoted else "REGISTRY_CA_G0_RECALL_NO_GO_FREEZE_ROUTE")
        worker_diagnostics = {}
        for name, worker in worker_results.items():
            summary = worker["receipt"]["summary"]
            worker_diagnostics[name] = {
                "mode": worker["mode"], "wall_seconds": worker["wall_seconds"],
                "stderr": worker["stderr"], "last_completed_session": SESSION_COUNT,
                "trace": {"bytes": worker["receipt"]["trace_bytes"],
                          "rows": worker["receipt"]["record_count"],
                          "sha256": worker["receipt"]["trace_sha256"]},
                "latency": summary["latency"],
                "resources": summary["resources"],
            }
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID,
            "status": status, "recorded_on": "2026-08-31", "rerun_forbidden": True,
            "evidence_scope": "shared 2000-session diagnostic candidate recall; not private validation",
            "implementation": {
                "commit": implementation_commit, "preregistration_commit": PREREG_COMMIT,
                "branch": BRANCH, "default": "off", "causal": True,
                "runtime_target_blind": True, "no_fit_or_selection": True,
                "full_agent_evaluator_started": False, "served_top10_unchanged": True,
                "protected_splits_opened_by_formal_runner": False,
            },
            "compliance": {
                "overall_orchestration_clean": False,
                "pre_preregistration_boundary_event":
                    "PRE_PREREG_PROTECTED_PATH_ACCESS_INCIDENT_RECORDED",
                "formal_runner_owned_pre_target_exact_path_allowlist_passed": True,
                "known_protected_paths_rejected_in_python_audited_events": True,
                "universal_os_stat_or_child_process_path_audit_proven": False,
                "qualification_does_not_erase_prior_incident": True,
            },
            "entrypoint_regression": {**preflight.entrypoints,
                                      "final_pre_receipt_check": bool(final_entrypoints)},
            "target_free_smoke": preflight.smoke,
            "candidate_recall": aggregate,
            "candidate_retention": {"complete_variable_c200_exact_ordered_prefix": True,
                                    "c200_loss_count": 0, "c200_reorder_count": 0,
                                    "c200_duplicate_count": 0, "tail_duplicate_count": 0,
                                    "tail_explicit_hard_conflict_count": 0,
                                    "served_c200_first10_unchanged": True},
            "top10_metrics": {"served_metrics_computed": False,
                              "reason": "candidate-recall-only; C200[:10] unchanged"},
            "activation": {"sessions": trace.activation_sessions,
                           "turns": trace.activation_turns},
            "inflation": inflation,
            "exact_repeat": {"passed": True, "trace_sha256": trace.canonical_trace_sha256,
                             "trace_bytes": trace.canonical_trace_bytes,
                             "record_count": trace.record_count},
            "resources": {"total_wall_seconds": round(total_wall, 6),
                          "workers": worker_diagnostics, "network_attempt_count": 0,
                          "gpu_peak_bytes": 0, "budgets_passed": True},
            "source_hashes": {**final_sources, "proxy": proxy_identity.report(),
                              "numeric_fold_archive": label_identity,
                              "fresh_trace": trace.canonical_trace_sha256},
            "git": final_git,
            "decision": {"promotion_gate_passed": promoted, "top10_global_promotion": False,
                         "threshold_c320_complete_union": 1_988,
                         "next_stage": "separately preregistered 100-session policy smoke"
                         if promoted else
                         "freeze this route and preregister the next independent multi-view sparse mechanism",
                         "fallback_order": ["SR-V2.12-FIXED-TWO-PAGE-GRACE", "v1.9", "P11", "R08"]},
        }
        _seal_receipt_size_estimate(result)
        _result_privacy_scan(result, preflight.catalog_ids)
        if descriptor is None:
            raise SparseProbeError("receipt descriptor closed early", "RECEIPT_WRITE")
        _write_descriptor(descriptor, result)
        os.close(descriptor)
        descriptor = None
        return result
    except BaseException as error:
        if proxy_source is not None:
            try:
                proxy_source.handle.close()
            except BaseException:
                pass
        if descriptor is not None:
            try:
                _write_invalid_receipt(descriptor, implementation_commit, error, phase=phase)
            finally:
                descriptor = None
        if isinstance(error, SparseProbeError):
            raise
        raise SparseProbeError("formal sparse one-shot failed", "UNEXPECTED_EXCEPTION") from error


def run(implementation_commit: str) -> dict[str, Any]:
    """Install the process audit supplement and consume the formal one-shot."""

    audit_guard = _install_process_audit_guard()
    try:
        return _run_guarded(implementation_commit, audit_guard)
    finally:
        audit_guard.close()


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise SparseProbeError("formal CLI arguments are invalid", "CLI_CONTRACT")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--implementation-commit")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight-only", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--entrypoint-self-check", action="store_true")
    parser.add_argument("--require-module", default="evaluator.local_evaluator")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    exit_code = 0
    try:
        arguments = _parser().parse_args(argv)
        if arguments.entrypoint_self_check:
            output = _entrypoint_self_check(arguments.require_module)
        else:
            if not arguments.implementation_commit:
                raise SparseProbeError("--implementation-commit is required", "CLI_CONTRACT")
            if arguments.preflight_only:
                checked = preflight_only(arguments.implementation_commit)
                output = {"status": "TARGET_FREE_PREFLIGHT_PASS",
                          "commit": checked.git["commit"], "entrypoints_passed": True,
                          "smoke_20_then_100_passed": True, "receipt_created": False}
            else:
                outcome = run(arguments.implementation_commit)
                output = {"status": outcome["status"],
                          "commit": outcome["implementation"]["commit"],
                          "c200_count": outcome["candidate_recall"]["all_sessions"]["c200"]["count"],
                          "c320_complete_union_count": outcome["candidate_recall"]["all_sessions"]
                          ["c320"]["count"], "promotion_gate_passed": outcome["decision"]
                          ["promotion_gate_passed"], "exact_repeat": True}
    except BaseException as error:
        code = error.code if isinstance(error, SparseProbeError) else "UNEXPECTED_EXCEPTION"
        output = {
            "status": "ERROR", "error_code": code,
            "exception_class": type(error).__name__,
            "formal_receipt_authoritative": OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink(),
            "raw_traceback_or_stderr_emitted": False,
        }
        exit_code = 2
    sys.stdout.buffer.write(_canonical_bytes(output) + b"\n")
    sys.stdout.buffer.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
