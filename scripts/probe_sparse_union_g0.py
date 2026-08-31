"""Protected v2.21 dual-view RRF G0 preflight and candidate-recall probe.

The runner has two mutually exclusive modes.  ``preflight`` opens only the
frozen target-free catalog/context/C200 inputs and compares one uncached direct
control with cached direct and literal-module workers at 20 and 100 sessions.
``candidate`` first requires the immutable preflight terminal, creates two
fresh full traces, closes and validates both traces, and only then attaches the
proxy target and the three explicitly allowed numeric label members.

This module never writes the fixed one-shot claim, outer envelope, or terminal
receipt.  ``run_v221_preflight.ps1`` owns those durable files so a child error
cannot discard the bootstrap envelope before it is recorded.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import sqlite3
import stat as stat_module
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

SCHEMA_VERSION = "small-ranker-v2.21-dual-view-rrf-g0-probe.v1"
WORKER_SCHEMA_VERSION = "small-ranker-v2.21-dual-view-rrf-g0-worker-summary.v1"
EXPERIMENT_ID = "SR-V2.21-CLAUSE-ISOLATED-DUAL-VIEW-RRF-G0"
BRANCH = "small-ranker-v2.21-gloss-g0"
REMOTE_URL = "https://github.com/lamperriat/techjam-err402.git"
REMOTE_REF = f"refs/remotes/origin/{BRANCH}"

PARENT_COMMIT = "d7bc963188e1ba357539c22e75a016611ad52ba2"
PREREG_COMMIT = "b81351a0657411ab04810bb4740b35b407d175cc"
PREREG_RELATIVE = "configs/small_ranker_v2_21.dual_view_rrf_g0_preregistration.json"
PREREG_PATH = ROOT / PREREG_RELATIVE
PREREG_BLOB = "5ee89ebda59c3dbf973fc3cd3f127ec34f47d1fa"
PREREG_BYTES = 19_911
PREREG_SHA256 = "4f454f736d27762723e4b395036a518babed0299677510ddd35b03c62abdfb53"

IMPLEMENTATION_PATHS = frozenset(
    {
        "starter/sparse_union_g0.py",
        "scripts/sparse_union_g0_worker.py",
        "scripts/probe_sparse_union_g0.py",
        "scripts/v221_safe_bootstrap.py",
        "scripts/run_v221_preflight.ps1",
        "tests/test_sparse_union_g0.py",
    }
)
RUNNER_RELATIVE = "scripts/probe_sparse_union_g0.py"
WORKER_RELATIVE = "scripts/sparse_union_g0_worker.py"
BOOTSTRAP_RELATIVE = "scripts/v221_safe_bootstrap.py"
CORE_RELATIVE = "starter/sparse_union_g0.py"
RUNNER_PATH = ROOT / RUNNER_RELATIVE
WORKER_PATH = ROOT / WORKER_RELATIVE
BOOTSTRAP_PATH = ROOT / BOOTSTRAP_RELATIVE

CATALOG_PATH = Path(r"D:\tiktok\techjam-err402-fast-track\data\catalog.jsonl")
CATALOG_IDENTITY = {
    "bytes": 60_546_327,
    "rows": 50_000,
    "sha256": "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67",
}
C200_CACHE_ROOT = Path(
    r"D:\tiktok\techjam-v2-16-c200-recall\experiments\fast_track"
    r"\c200_candidate_recall_cache_20260831"
)
CONTEXT_PATH = C200_CACHE_ROOT / "visible_context.jsonl"
CONTEXT_IDENTITY = {
    "bytes": 47_168_882,
    "rows": 2_000,
    "sha256": "f30a98700da5d480731fe7e82c87c40a22f06de290e069e20dc68f9fefecd20f",
}
C200_REFERENCE_PATHS = (
    C200_CACHE_ROOT / "replica_a.jsonl",
    C200_CACHE_ROOT / "replica_b.jsonl",
)
C200_IDENTITY = {
    "bytes": 32_226_135,
    "rows": 20_000,
    "sha256": "a8589749376f48f019997a618481578dde36be4ca1fc723e8ed00056c23e40dc",
}
PROXY_PATH = Path(
    r"D:\tiktok\techjam-err402-fast-track\experiments\fast_track"
    r"\proxy_v1\proxy_train_explore.jsonl"
)
PROXY_IDENTITY = {
    "bytes": 1_315_338,
    "rows": 2_000,
    "sha256": "2175696171c0d874fca4b9aa456ff5fd7d570f2184f59ade6781198f6443198e",
}
LABEL_PATH = Path(
    r"D:\tiktok\techjam-err402-fast-track\experiments\fast_track"
    r"\small_ranker_v1\labels_v2.npz"
)
LABEL_IDENTITY = {
    "bytes": 1_702_876,
    "sha256": "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb",
}

RUNTIME_BASE = Path(r"D:\tiktok\.v221_runtime")
EXPERIMENT_ROOT = ROOT / "experiments" / "fast_track"
PREFLIGHT_CLAIM_PATH = EXPERIMENT_ROOT / (
    "small_ranker_v2_21_dual_view_rrf_g0_preflight_claim_20260831.json"
)
PREFLIGHT_OUTER_PATH = EXPERIMENT_ROOT / (
    "small_ranker_v2_21_dual_view_rrf_g0_preflight_outer_20260831.json"
)
PREFLIGHT_RESULT_PATH = EXPERIMENT_ROOT / (
    "small_ranker_v2_21_dual_view_rrf_g0_preflight_20260831.json"
)
CANDIDATE_CLAIM_PATH = EXPERIMENT_ROOT / (
    "small_ranker_v2_21_dual_view_rrf_g0_candidate_recall_claim_20260831.json"
)

EXPECTED_EXECUTABLE = Path(r"D:\450\conda\envs\tiktok\python.exe")
EXPECTED_EXECUTABLE_BYTES = 93_184
EXPECTED_EXECUTABLE_SHA256 = (
    "7819c841b9a6457da034e567563de1283dbc0b86482fd83d62b5d982d2a83a63"
)
EXPECTED_PYTHON = "3.11.16"
EXPECTED_SQLITE = "3.53.4"
EXPECTED_GIT = Path(r"C:\Program Files\Git\mingw64\bin\git.exe")
EXPECTED_GIT_BYTES = 4_018_680
EXPECTED_GIT_SHA256 = (
    "3fe4878d8399f6fb7632b9325559d1bb38c3a17aac7a60f667c1e5f90b865248"
)
EXPECTED_GIT_VERSION = "git version 2.45.2.windows.1"
EXPECTED_GITDIR = Path(
    r"D:\tiktok\techjam-err402\.git\worktrees\techjam-v2-21-gloss-g0"
)
EXPECTED_COMMON_GITDIR = Path(r"D:\tiktok\techjam-err402\.git")
BOOTSTRAP_ATTESTATION = "_techjam_v221_bootstrap_attestation"

SESSION_COUNT = 2_000
TURN_COUNT = 10
ROUTE_LIMIT = 120
MAX_CANDIDATES = 400
ALLOWED_PREFLIGHT_LIMITS = (20, 100)
WORKER_RSS_MAXIMUM = 1_610_612_736
EXTRA_ROUTE_P95_MAXIMUM = 100.0
PER_TURN_P95_MAXIMUM = 400.0
PAIR_WALL_MAXIMUM = 60.0
FORMAL_WALL_MAXIMUM = 1_800.0
CELL_RATIO_MAXIMUM = 2.0
TRACE_RATIO_MAXIMUM = 2.1

BASELINE_SANITY = {
    "SEALED_PREFIX_K10": 1_895,
    "SEALED_PREFIX_K20": 1_943,
    "SEALED_PREFIX_K50": 1_982,
    "SEALED_PREFIX_K100": 1_986,
    "SEALED_VARIABLE_C200": 1_986,
}
TAXONOMY_CODEBOOK = {
    0: "accessories-other",
    1: "clothing",
    2: "jewelry",
    3: "shoes",
}

COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
CATALOG_ID_RE = re.compile(r"[A-Z0-9]{10}\Z")
ASIN_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.I)
FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "asin",
        "parent_asin",
        "sample_id",
        "session_id",
        "product_id",
        "ground_truth",
        "target",
        "target_id",
        "target_asin",
        "targets",
        "eligible_from",
        "positive_index",
        "identifier",
        "identifiers",
        "query",
        "queries",
        "messages",
        "per_session",
        "membership",
        "membership_vector",
        "candidates",
        "c200",
        "ordinal",
        "turn",
    }
)


class SparseUnionProbeError(RuntimeError):
    """A stable, non-disclosing formal probe failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class FileIdentity:
    bytes: int
    rows: int
    sha256: str
    snapshot: tuple[int, int, int]

    def report(self) -> dict[str, int | str]:
        return {"bytes": self.bytes, "rows": self.rows, "sha256": self.sha256}


@dataclass(frozen=True)
class BootstrapLaunch:
    mode: str
    command: tuple[str, ...]
    environment: Mapping[str, str]
    runtime_root: Path
    runtime_snapshot: tuple[int, int, int]
    pycache: Path
    trace_path: Path | None


@dataclass
class WorkerRun:
    mode: str
    cache_enabled: bool
    receipt: dict[str, Any]
    parent_wall_seconds: float
    launch: BootstrapLaunch


@dataclass(frozen=True)
class TraceAudit:
    bytes: int
    rows: int
    sha256: str
    c200_cells: int
    candidate_cells: int
    reference_prefix_bytes: int
    expansion_turns: int
    expansion_sessions: int
    min_candidates: int
    max_candidates: int
    records: tuple[tuple[tuple[str, ...], int], ...]

    def compact(self) -> dict[str, int | str | float]:
        return {
            "bytes": self.bytes,
            "rows": self.rows,
            "sha256": self.sha256,
            "candidate_cells": self.candidate_cells,
            "c200_cells": self.c200_cells,
            "candidate_cell_ratio_over_c200": round(
                self.candidate_cells / self.c200_cells, 6
            ),
            "trace_byte_ratio_over_c200": round(
                self.bytes / self.reference_prefix_bytes, 6
            ),
            "expansion_turns": self.expansion_turns,
            "expansion_sessions": self.expansion_sessions,
            "min_candidates": self.min_candidates,
            "max_candidates": self.max_candidates,
        }


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SparseUnionProbeError("NON_CANONICAL_JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SparseUnionProbeError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise SparseUnionProbeError("NONFINITE_JSON")


def _parse_canonical_json(raw: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise SparseUnionProbeError(code) from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise SparseUnionProbeError(code)
    return value


def _parse_json_object(raw: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise SparseUnionProbeError(code) from error
    if not isinstance(value, dict):
        raise SparseUnionProbeError(code)
    return value


def _snapshot(value: os.stat_result) -> tuple[int, int, int]:
    return int(value.st_size), int(value.st_mtime_ns), int(value.st_ino)


def _is_reparse(path: Path) -> bool:
    try:
        observed = path.lstat()
    except OSError:
        return True
    flag = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(
        getattr(observed, "st_file_attributes", 0) & flag
    )


def _require_plain(path: Path, *, directory: bool = False) -> Path:
    resolved = path.resolve(strict=True)
    current = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        current = current / part
        if _is_reparse(current):
            raise SparseUnionProbeError("REPARSE_PATH_DENIED")
    observed = resolved.stat()
    if directory:
        if not stat_module.S_ISDIR(observed.st_mode):
            raise SparseUnionProbeError("DIRECTORY_REQUIRED")
    elif not stat_module.S_ISREG(observed.st_mode):
        raise SparseUnionProbeError("REGULAR_FILE_REQUIRED")
    return resolved


def _identity(path: Path, expected: Mapping[str, Any] | None = None) -> FileIdentity:
    resolved = _require_plain(path)
    digest = hashlib.sha256()
    count = 0
    rows = 0
    with resolved.open("rb") as handle:
        before = _snapshot(os.fstat(handle.fileno()))
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            count += len(chunk)
            rows += chunk.count(b"\n")
        after = _snapshot(os.fstat(handle.fileno()))
    value = FileIdentity(count, rows, digest.hexdigest(), after)
    if before != after or _snapshot(resolved.stat()) != after or count != before[0]:
        raise SparseUnionProbeError("SOURCE_CHANGED_WHILE_HASHED")
    if expected is not None:
        report = value.report()
        if any(report.get(str(key)) != item for key, item in expected.items()):
            raise SparseUnionProbeError("SOURCE_IDENTITY_DRIFT")
    return value


def _git_blob_from_raw(raw: bytes) -> str:
    normalized = raw.replace(b"\r\n", b"\n")
    header = b"blob " + str(len(normalized)).encode("ascii") + b"\0"
    return hashlib.sha1(header + normalized).hexdigest()


def _minimal_environment() -> dict[str, str]:
    allowed = {
        "COMSPEC",
        "PATHEXT",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "WINDIR",
    }
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in allowed
    }
    system_root = environment.get("SYSTEMROOT", r"C:\Windows")
    environment["PATH"] = str(Path(system_root) / "System32")
    return environment


def _offline_environment() -> dict[str, str]:
    environment = _minimal_environment()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def _git_environment() -> dict[str, str]:
    environment = _minimal_environment()
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "NUL",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C",
            "LC_ALL": "C",
            "PAGER": "cat",
        }
    )
    return environment


def _run_process(
    command: Sequence[str],
    *,
    timeout: float,
    environment: Mapping[str, str],
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            close_fds=True,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SparseUnionProbeError("CHILD_PROCESS_FAILED") from error


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    allowed_first = {
        "cat-file",
        "config",
        "diff-tree",
        "merge-base",
        "rev-parse",
        "rev-list",
        "symbolic-ref",
    }
    if not arguments or arguments[0] not in allowed_first:
        raise SparseUnionProbeError("GIT_COMMAND_DENIED")
    completed = _run_process(
        (str(EXPECTED_GIT), *arguments),
        timeout=30.0,
        environment=_git_environment(),
    )
    if completed.returncode != 0 or completed.stderr:
        raise SparseUnionProbeError("GIT_COMMAND_FAILED")
    if binary:
        return completed.stdout
    try:
        return completed.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeError as error:
        raise SparseUnionProbeError("GIT_OUTPUT_ENCODING") from error


def _validate_runtime() -> dict[str, Any]:
    executable = _identity(EXPECTED_EXECUTABLE)
    git_executable = _identity(EXPECTED_GIT)
    if (
        executable.bytes != EXPECTED_EXECUTABLE_BYTES
        or executable.sha256 != EXPECTED_EXECUTABLE_SHA256
        or git_executable.bytes != EXPECTED_GIT_BYTES
        or git_executable.sha256 != EXPECTED_GIT_SHA256
        or Path(sys.executable).resolve(strict=True) != EXPECTED_EXECUTABLE.resolve(strict=True)
        or sys.version.split()[0] != EXPECTED_PYTHON
        or sqlite3.sqlite_version != EXPECTED_SQLITE
        or os.environ.get("PYTHONHASHSEED") != "0"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise SparseUnionProbeError("RUNTIME_IDENTITY")
    version = _run_process(
        (str(EXPECTED_GIT), "--version"),
        timeout=10.0,
        environment=_git_environment(),
        cwd=Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32",
    )
    if (
        version.returncode != 0
        or version.stderr
        or version.stdout.decode("utf-8", errors="strict").strip()
        != EXPECTED_GIT_VERSION
    ):
        raise SparseUnionProbeError("GIT_RUNTIME_IDENTITY")
    if any(name in sys.modules for name in ("torch", "tensorflow", "cupy")):
        raise SparseUnionProbeError("GPU_RUNTIME_PRESENT")
    return {
        "cpu_only": True,
        "gpu_peak_bytes": 0,
        "python": EXPECTED_PYTHON,
        "python_sha256": EXPECTED_EXECUTABLE_SHA256,
        "sqlite": EXPECTED_SQLITE,
        "git": EXPECTED_GIT_VERSION,
        "git_sha256": EXPECTED_GIT_SHA256,
    }


def _validate_preregistration() -> dict[str, Any]:
    identity = _identity(
        PREREG_PATH,
        {"bytes": PREREG_BYTES, "sha256": PREREG_SHA256},
    )
    if _git_blob_from_raw(PREREG_PATH.read_bytes()) != PREREG_BLOB:
        raise SparseUnionProbeError("PREREG_WORKTREE_BLOB")
    try:
        value = json.loads(
            PREREG_PATH.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise SparseUnionProbeError("PREREG_JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("status")
        != "PREREGISTERED_BEFORE_IMPLEMENTATION_AND_OUTCOME"
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("branch") != BRANCH
        or value.get("parent_commit") != PARENT_COMMIT
        or set(
            value.get("checkpoint_choreography", {}).get(
                "implementation_exact_changed_paths", []
            )
        )
        != set(IMPLEMENTATION_PATHS)
    ):
        raise SparseUnionProbeError("PREREG_SEMANTICS")
    return identity.report()


def _validate_git_checkpoint(implementation_commit: str) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(implementation_commit) is None:
        raise SparseUnionProbeError("IMPLEMENTATION_COMMIT_SHAPE")
    if _require_plain(ROOT / ".git").read_bytes().replace(b"\r\n", b"\n") != (
        b"gitdir: D:/tiktok/techjam-err402/.git/worktrees/"
        b"techjam-v2-21-gloss-g0\n"
    ):
        raise SparseUnionProbeError("WORKTREE_POINTER")
    if _require_plain(EXPECTED_GITDIR / "HEAD").read_bytes().replace(
        b"\r\n", b"\n"
    ) != b"ref: refs/heads/small-ranker-v2.21-gloss-g0\n":
        raise SparseUnionProbeError("WORKTREE_HEAD_CONTROL")
    if _git("rev-parse", "HEAD") != implementation_commit:
        raise SparseUnionProbeError("HEAD_NOT_IMPLEMENTATION")
    if _git("symbolic-ref", "--short", "HEAD") != BRANCH:
        raise SparseUnionProbeError("BRANCH_DRIFT")
    if _git("rev-parse", REMOTE_REF) != implementation_commit:
        raise SparseUnionProbeError("IMPLEMENTATION_NOT_PUSHED")
    if implementation_commit == PREREG_COMMIT:
        raise SparseUnionProbeError("IMPLEMENTATION_COMMIT_REQUIRED")
    _git("merge-base", "--is-ancestor", PREREG_COMMIT, implementation_commit)
    if _git("rev-list", "--min-parents=2", f"{PREREG_COMMIT}..{implementation_commit}"):
        raise SparseUnionProbeError("IMPLEMENTATION_MERGE_DENIED")
    if _git("rev-parse", f"{PREREG_COMMIT}^") != PARENT_COMMIT:
        raise SparseUnionProbeError("PREREG_PARENT")
    if _git("rev-parse", f"{PREREG_COMMIT}:{PREREG_RELATIVE}") != PREREG_BLOB:
        raise SparseUnionProbeError("PREREG_OBJECT_BLOB")
    remote = _git(
        "config", "--local", "--no-includes", "--get-all", "remote.origin.url"
    )
    if remote != REMOTE_URL:
        raise SparseUnionProbeError("REMOTE_DRIFT")
    changed_raw = _git(
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        PREREG_COMMIT,
        implementation_commit,
    )
    changed = frozenset(line for line in str(changed_raw).splitlines() if line)
    if changed != IMPLEMENTATION_PATHS:
        raise SparseUnionProbeError("IMPLEMENTATION_PATH_ALLOWLIST")
    blobs: dict[str, str] = {}
    for relative in sorted(IMPLEMENTATION_PATHS):
        blob = str(_git("rev-parse", f"{implementation_commit}:{relative}"))
        if COMMIT_RE.fullmatch(blob) is None:
            raise SparseUnionProbeError("IMPLEMENTATION_BLOB_SHAPE")
        if _git_blob_from_raw(_require_plain(ROOT / relative).read_bytes()) != blob:
            raise SparseUnionProbeError("WORKTREE_SOURCE_DRIFT")
        blobs[relative] = blob
    # The bootstrap manifest is the second independent binding of executable
    # runner/worker/core bytes.  Placeholders must be gone before formal use.
    bootstrap_text = _require_plain(BOOTSTRAP_PATH).read_text(
        encoding="utf-8", errors="strict"
    )
    for relative in (RUNNER_RELATIVE, WORKER_RELATIVE, CORE_RELATIVE):
        if blobs[relative] not in bootstrap_text:
            raise SparseUnionProbeError("BOOTSTRAP_MANIFEST_NOT_FINAL")
    return {
        "branch": BRANCH,
        "commit": implementation_commit,
        "pushed": True,
        "remote": REMOTE_URL,
        "preregistration_commit": PREREG_COMMIT,
        "implementation_blobs": blobs,
        "object_only_git": True,
    }


def _require_attestation(git_report: Mapping[str, Any]) -> dict[str, Any]:
    value = getattr(sys, BOOTSTRAP_ATTESTATION, None)
    if not isinstance(value, Mapping) or set(value) != {
        "mode",
        "bootstrap_blob",
        "target_blob",
        "source_only",
        "guarded_path",
        "pycache_prefix",
    }:
        raise SparseUnionProbeError("BOOTSTRAP_ATTESTATION_MISSING")
    blobs = git_report["implementation_blobs"]
    if (
        value.get("mode") != "direct"
        or value.get("bootstrap_blob") != blobs[BOOTSTRAP_RELATIVE]
        or value.get("target_blob") != blobs[RUNNER_RELATIVE]
        or value.get("source_only") is not True
        or value.get("guarded_path") is not True
        or not isinstance(value.get("pycache_prefix"), str)
    ):
        raise SparseUnionProbeError("BOOTSTRAP_ATTESTATION_DRIFT")
    return dict(value)


def _source_checkpoint(*, include_targets: bool = False) -> dict[str, Any]:
    sources: dict[str, Any] = {
        "catalog": _identity(CATALOG_PATH, CATALOG_IDENTITY).report(),
        "visible_context": _identity(CONTEXT_PATH, CONTEXT_IDENTITY).report(),
        "sealed_c200": [
            _identity(path, C200_IDENTITY).report() for path in C200_REFERENCE_PATHS
        ],
    }
    if sources["sealed_c200"][0] != sources["sealed_c200"][1]:
        raise SparseUnionProbeError("C200_REPLICAS_DIFFER")
    if include_targets:
        sources["proxy"] = _identity(PROXY_PATH, PROXY_IDENTITY).report()
        label = _identity(LABEL_PATH, LABEL_IDENTITY)
        sources["numeric_label_archive"] = {
            "bytes": label.bytes,
            "sha256": label.sha256,
        }
    return sources


def _load_catalog_ids() -> frozenset[str]:
    identifiers: set[str] = set()
    with _require_plain(CATALOG_PATH).open("rb") as handle:
        for raw in handle:
            try:
                row = json.loads(
                    raw.decode("utf-8", errors="strict"),
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (UnicodeError, ValueError, TypeError) as error:
                raise SparseUnionProbeError("CATALOG_JSON") from error
            identifier = row.get("parent_asin") if isinstance(row, dict) else None
            if (
                not isinstance(identifier, str)
                or CATALOG_ID_RE.fullmatch(identifier) is None
                or identifier in identifiers
            ):
                raise SparseUnionProbeError("CATALOG_IDENTIFIER_SCHEMA")
            identifiers.add(identifier)
    if len(identifiers) != CATALOG_IDENTITY["rows"]:
        raise SparseUnionProbeError("CATALOG_IDENTIFIER_COUNT")
    return frozenset(identifiers)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise SparseUnionProbeError("SHORT_WRITE")
        view = view[written:]


def _prepare_launch(
    *,
    mode: str,
    target_path: Path,
    target_module: str,
    target_blob: str,
    bootstrap_blob: str,
    target_arguments: Sequence[str],
    trace_filename: str | None = None,
) -> BootstrapLaunch:
    if (
        mode not in {"direct", "module"}
        or COMMIT_RE.fullmatch(target_blob) is None
        or COMMIT_RE.fullmatch(bootstrap_blob) is None
    ):
        raise SparseUnionProbeError("BOOTSTRAP_LAUNCH_CONTRACT")
    base = _require_plain(RUNTIME_BASE, directory=True)
    runtime_root = Path(tempfile.mkdtemp(prefix="v221-", dir=base))
    runtime_snapshot = _snapshot(runtime_root.stat())
    trace_path = runtime_root / trace_filename if trace_filename is not None else None
    try:
        if runtime_root.parent != base or _is_reparse(runtime_root):
            raise SparseUnionProbeError("RUNTIME_ROOT_UNSAFE")
        pycache = runtime_root / "pycache"
        os.mkdir(pycache, 0o700)
        _require_plain(pycache, directory=True)
        bootstrap_raw = _require_plain(BOOTSTRAP_PATH).read_bytes()
        if _git_blob_from_raw(bootstrap_raw) != bootstrap_blob:
            raise SparseUnionProbeError("BOOTSTRAP_WORKTREE_DRIFT")
        arguments = [
            str(trace_path) if value == "{TRACE}" and trace_path is not None else value
            for value in target_arguments
        ]
        if "{TRACE}" in arguments:
            raise SparseUnionProbeError("TRACE_ARGUMENT_UNBOUND")
        environment = _offline_environment()
        prefix = [
            str(EXPECTED_EXECUTABLE),
            "-P",
            "-S",
            "-s",
            "-B",
            "-X",
            f"pycache_prefix={pycache.as_posix()}",
        ]
        if mode == "direct":
            prefix.append(str(BOOTSTRAP_PATH))
        else:
            module_root = runtime_root / "module"
            os.mkdir(module_root, 0o700)
            module_path = module_root / "v221_safe_bootstrap.py"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            descriptor = os.open(module_path, flags, 0o600)
            try:
                _write_all(descriptor, bootstrap_raw)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if _git_blob_from_raw(_require_plain(module_path).read_bytes()) != bootstrap_blob:
                raise SparseUnionProbeError("BOOTSTRAP_MODULE_COPY_DRIFT")
            environment["PYTHONPATH"] = module_root.as_posix()
            prefix.extend(("-m", "v221_safe_bootstrap"))
        command = (
            *prefix,
            "--mode",
            mode,
            "--target-path",
            target_path.as_posix(),
            "--target-module",
            target_module,
            "--target-blob",
            target_blob,
            "--bootstrap-blob",
            bootstrap_blob,
            "--",
            *arguments,
        )
        return BootstrapLaunch(
            mode=mode,
            command=tuple(command),
            environment=environment,
            runtime_root=runtime_root,
            runtime_snapshot=runtime_snapshot,
            pycache=pycache,
            trace_path=trace_path,
        )
    except BaseException:
        _cleanup_runtime_path(runtime_root, runtime_snapshot)
        raise


def _cleanup_runtime_path(root: Path, expected_snapshot: tuple[int, int, int]) -> None:
    try:
        base = _require_plain(RUNTIME_BASE, directory=True)
        observed = root.lstat()
    except FileNotFoundError:
        return
    lexical_root = PureWindowsPath(str(root)).as_posix().casefold()
    lexical_base = PureWindowsPath(str(base)).as_posix().casefold().rstrip("/")
    if (
        root.parent != base
        or lexical_root == lexical_base
        or not lexical_root.startswith(lexical_base + "/")
        or _is_reparse(root)
        or _snapshot(observed) != expected_snapshot
        or not stat_module.S_ISDIR(observed.st_mode)
    ):
        raise SparseUnionProbeError("RUNTIME_CLEANUP_IDENTITY")
    shutil.rmtree(root)


def _cleanup_launch(launch: BootstrapLaunch) -> None:
    _cleanup_runtime_path(launch.runtime_root, launch.runtime_snapshot)


def _validate_bootstrap_envelope(
    completed: subprocess.CompletedProcess[bytes],
    *,
    launch: BootstrapLaunch,
    target_blob: str,
    bootstrap_blob: str,
) -> tuple[int, dict[str, Any] | None]:
    if completed.stderr or len(completed.stdout) > (1 << 20):
        raise SparseUnionProbeError("BOOTSTRAP_STDIO")
    envelope = _parse_canonical_json(completed.stdout, "BOOTSTRAP_ENVELOPE")
    if set(envelope) != {"bootstrap", "target_exit_code", "target_receipt"}:
        raise SparseUnionProbeError("BOOTSTRAP_ENVELOPE_SCHEMA")
    expected_attestation = {
        "mode": launch.mode,
        "bootstrap_blob": bootstrap_blob,
        "target_blob": target_blob,
        "source_only": True,
        "guarded_path": True,
        "pycache_prefix": launch.pycache.as_posix(),
    }
    exit_code = envelope.get("target_exit_code")
    receipt = envelope.get("target_receipt")
    if (
        not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or not 0 <= exit_code <= 255
        or completed.returncode != exit_code
        or envelope.get("bootstrap") != expected_attestation
        or (receipt is not None and not isinstance(receipt, dict))
        or (exit_code == 0 and not isinstance(receipt, dict))
    ):
        raise SparseUnionProbeError("BOOTSTRAP_ATTESTATION")
    return exit_code, receipt


def _invoke(
    *,
    mode: str,
    target_path: Path,
    target_module: str,
    target_blob: str,
    bootstrap_blob: str,
    target_arguments: Sequence[str],
    timeout: float,
    trace_filename: str | None = None,
) -> tuple[BootstrapLaunch, int, dict[str, Any] | None, float]:
    started = time.perf_counter()
    launch = _prepare_launch(
        mode=mode,
        target_path=target_path,
        target_module=target_module,
        target_blob=target_blob,
        bootstrap_blob=bootstrap_blob,
        target_arguments=target_arguments,
        trace_filename=trace_filename,
    )
    try:
        completed = _run_process(
            launch.command,
            timeout=timeout,
            environment=launch.environment,
        )
        exit_code, receipt = _validate_bootstrap_envelope(
            completed,
            launch=launch,
            target_blob=target_blob,
            bootstrap_blob=bootstrap_blob,
        )
        wall = time.perf_counter() - started
        return launch, exit_code, receipt, wall
    except BaseException:
        _cleanup_launch(launch)
        raise


def _entrypoint_receipt(value: object) -> None:
    if not isinstance(value, Mapping) or value != {
        "c200_contract_imported": True,
        "evaluator_imported": True,
        "legacy_runtime_absent": True,
        "project_root_bootstrapped": True,
        "required_module": "starter.sparse_union_g0",
        "status": "ENTRYPOINT_SELF_CHECK_PASS",
    }:
        raise SparseUnionProbeError("ENTRYPOINT_RECEIPT")


def _verify_entrypoints(blobs: Mapping[str, str]) -> dict[str, Any]:
    bootstrap_blob = blobs[BOOTSTRAP_RELATIVE]
    report: dict[str, Any] = {}
    for subject, target_path, target_module, relative in (
        ("runner", RUNNER_PATH, "scripts.probe_sparse_union_g0", RUNNER_RELATIVE),
        ("worker", WORKER_PATH, "scripts.sparse_union_g0_worker", WORKER_RELATIVE),
    ):
        for mode in ("direct", "module"):
            launch, exit_code, receipt, _wall = _invoke(
                mode=mode,
                target_path=target_path,
                target_module=target_module,
                target_blob=blobs[relative],
                bootstrap_blob=bootstrap_blob,
                target_arguments=(
                    "--entrypoint-self-check",
                    "--require-module",
                    "starter.sparse_union_g0",
                ),
                timeout=30.0,
            )
            try:
                if exit_code != 0:
                    raise SparseUnionProbeError("ENTRYPOINT_FAILED")
                _entrypoint_receipt(receipt)
            finally:
                _cleanup_launch(launch)
            report[f"{subject}_{mode}"] = True
    # The prohibited legacy runtime must fail in both literal invocation modes.
    for mode in ("direct", "module"):
        launch, exit_code, receipt, _wall = _invoke(
            mode=mode,
            target_path=WORKER_PATH,
            target_module="scripts.sparse_union_g0_worker",
            target_blob=blobs[WORKER_RELATIVE],
            bootstrap_blob=bootstrap_blob,
            target_arguments=(
                "--entrypoint-self-check",
                "--require-module",
                "starter.sparse_multiview",
            ),
            timeout=30.0,
        )
        try:
            if exit_code == 0 or receipt is not None:
                raise SparseUnionProbeError("LEGACY_ENTRYPOINT_NOT_DENIED")
        finally:
            _cleanup_launch(launch)
        report[f"legacy_module_denied_{mode}"] = True
    return report


def _worker_arguments(
    *,
    session_limit: int,
    reference: Path,
    nonce: str,
    cache_enabled: bool,
    blobs: Mapping[str, str],
) -> tuple[str, ...]:
    if session_limit not in (*ALLOWED_PREFLIGHT_LIMITS, SESSION_COUNT):
        raise SparseUnionProbeError("SESSION_LIMIT")
    result = [
        "--nonce",
        nonce,
        "--catalog",
        str(CATALOG_PATH),
        "--context",
        str(CONTEXT_PATH),
        "--c200-reference",
        str(reference),
        "--trace-output",
        "{TRACE}",
        "--session-limit",
        str(session_limit),
        "--semantic-audit",
        "--expected-worker-blob",
        blobs[WORKER_RELATIVE],
        "--expected-union-blob",
        blobs[CORE_RELATIVE],
    ]
    if cache_enabled:
        result.append("--semantic-cache")
    return tuple(result)


def _validate_nonnegative_int(value: object, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SparseUnionProbeError(code)
    return value


def _validate_latency(value: object, expected_count: int, code: str) -> float:
    if not isinstance(value, Mapping) or set(value) != {
        "count",
        "maximum_milliseconds",
        "p50_milliseconds",
        "p95_milliseconds",
    }:
        raise SparseUnionProbeError(code)
    if _validate_nonnegative_int(value.get("count"), code) != expected_count:
        raise SparseUnionProbeError(code)
    numbers = [value.get(key) for key in (
        "maximum_milliseconds", "p50_milliseconds", "p95_milliseconds"
    )]
    if any(
        not isinstance(number, (int, float))
        or isinstance(number, bool)
        or not math.isfinite(float(number))
        or float(number) < 0.0
        for number in numbers
    ):
        raise SparseUnionProbeError(code)
    return float(value["p95_milliseconds"])


def _validate_cache(value: object, *, expected_closed: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "enabled",
        "closed",
        "clears",
        "fts_route",
        "product_view",
        "mask_decision",
    }:
        raise SparseUnionProbeError("CACHE_SCHEMA")
    if value.get("enabled") is not True or value.get("closed") is not expected_closed:
        raise SparseUnionProbeError("CACHE_STATE")
    _validate_nonnegative_int(value.get("clears"), "CACHE_COUNTER")
    layer_keys = {
        "lookups", "hits", "misses", "evictions", "size", "capacity",
        "avoided_operations",
    }
    for name in ("fts_route", "product_view", "mask_decision"):
        layer = value.get(name)
        if not isinstance(layer, Mapping) or set(layer) != layer_keys:
            raise SparseUnionProbeError("CACHE_LAYER_SCHEMA")
        for item in layer.values():
            _validate_nonnegative_int(item, "CACHE_LAYER_COUNTER")
        if not expected_closed and (
            int(layer["lookups"]) <= 0 or int(layer["hits"]) <= 0
        ):
            raise SparseUnionProbeError("CACHE_LAYER_NO_HIT")
        if expected_closed and int(layer["size"]) != 0:
            raise SparseUnionProbeError("CACHE_NOT_CLEARED")
    return dict(value)


def _validate_worker_receipt(
    value: object,
    *,
    nonce: str,
    session_limit: int,
    cache_enabled: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "error_code",
        "kind",
        "last_completed_session",
        "nonce",
        "phase",
        "record_count",
        "schema_version",
        "status",
        "summary",
        "trace_bytes",
        "trace_sha256",
    }:
        raise SparseUnionProbeError("WORKER_RECEIPT_SCHEMA")
    expected_records = session_limit * TURN_COUNT
    if (
        value.get("schema_version") != WORKER_SCHEMA_VERSION
        or value.get("status") != "SUCCESS"
        or value.get("error_code") != "NONE"
        or value.get("kind") != "receipt"
        or value.get("phase") != "COMPLETE"
        or value.get("nonce") != nonce
        or value.get("last_completed_session") != session_limit
        or value.get("record_count") != expected_records
        or not isinstance(value.get("trace_bytes"), int)
        or isinstance(value.get("trace_bytes"), bool)
        or int(value["trace_bytes"]) <= 0
        or not isinstance(value.get("trace_sha256"), str)
        or DIGEST_RE.fullmatch(str(value["trace_sha256"])) is None
    ):
        raise SparseUnionProbeError("WORKER_RECEIPT_VALUES")
    summary = value.get("summary")
    required_summary = {
        "activation", "configuration", "environment", "input_identities",
        "latency", "lifecycle", "mask", "pool_lengths", "prefix_integrity",
        "processed_sessions", "processed_turns", "resources",
        "route_diagnostics", "session_limit", "source_identities",
        "semantic_trace",
    }
    if cache_enabled:
        required_summary.add("cache")
    if not isinstance(summary, Mapping) or set(summary) != required_summary:
        raise SparseUnionProbeError("WORKER_SUMMARY_SCHEMA")
    activation = summary.get("activation")
    activation_keys = {
        "category_route_activation_records",
        "positive_core_route_activation_records",
        "dual_route_activation_records",
        "dual_support_candidate_cells",
        "union_expansion_records",
        "union_novel_candidate_cells",
        "legacy_route_executions",
    }
    if not isinstance(activation, Mapping) or set(activation) != activation_keys:
        raise SparseUnionProbeError("ACTIVATION_SCHEMA")
    for item in activation.values():
        _validate_nonnegative_int(item, "ACTIVATION_COUNTER")
    if (
        int(activation["category_route_activation_records"]) <= 0
        or int(activation["positive_core_route_activation_records"]) <= 0
        or int(activation["union_expansion_records"]) <= 0
        or int(activation["union_novel_candidate_cells"]) <= 0
        or int(activation["legacy_route_executions"]) != 0
    ):
        raise SparseUnionProbeError("ACTIVATION_GATE")
    configuration = summary.get("configuration")
    if configuration != {
        "category_route_limit": ROUTE_LIMIT,
        "positive_core_route_limit": ROUTE_LIMIT,
        "default_off": True,
        "diagnostic_only": True,
        "exact_fraction_rrf": True,
        "served_top10_unchanged": True,
        "stable_append_after_complete_variable_c200": True,
    }:
        raise SparseUnionProbeError("WORKER_CONFIGURATION")
    environment = summary.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("python") != EXPECTED_PYTHON
        or environment.get("sqlite") != EXPECTED_SQLITE
        or environment.get("pythonhashseed") != "0"
        or environment.get("network_attempt_count") != 0
        or environment.get("gpu_used") is not False
        or environment.get("gpu_peak_bytes") != 0
    ):
        raise SparseUnionProbeError("WORKER_ENVIRONMENT")
    latency = summary.get("latency")
    if not isinstance(latency, Mapping) or set(latency) != {
        "context_container_parse", "extra_routes_fusion_and_mask", "per_turn"
    }:
        raise SparseUnionProbeError("LATENCY_SCHEMA")
    _validate_latency(latency["context_container_parse"], session_limit, "CONTEXT_LATENCY")
    extra_p95 = _validate_latency(
        latency["extra_routes_fusion_and_mask"], expected_records, "EXTRA_LATENCY"
    )
    turn_p95 = _validate_latency(latency["per_turn"], expected_records, "TURN_LATENCY")
    resources = summary.get("resources")
    if not isinstance(resources, Mapping):
        raise SparseUnionProbeError("RESOURCE_SCHEMA")
    rss = resources.get("peak_working_set_bytes")
    if (
        not isinstance(rss, int)
        or isinstance(rss, bool)
        or not 0 < rss <= WORKER_RSS_MAXIMUM
        or resources.get("network_attempt_count") != 0
        or resources.get("gpu_peak_bytes") != 0
        or extra_p95 > EXTRA_ROUTE_P95_MAXIMUM
        or turn_p95 > PER_TURN_P95_MAXIMUM
        or float(resources.get("candidate_cell_ratio_over_c200", math.inf))
        > CELL_RATIO_MAXIMUM
        or float(resources.get("trace_byte_ratio_over_c200", math.inf))
        > TRACE_RATIO_MAXIMUM
        or float(resources.get("wall_seconds", math.inf)) > FORMAL_WALL_MAXIMUM
    ):
        raise SparseUnionProbeError("WORKER_RESOURCE_GATE")
    prefix = summary.get("prefix_integrity")
    mask = summary.get("mask")
    if prefix != {
        "c200_duplicate_count": 0,
        "c200_loss_count": 0,
        "c200_reorder_count": 0,
        "top10_change_count": 0,
    } or not isinstance(mask, Mapping) or any(
        mask.get(key) != 0
        for key in ("tail_duplicate_count", "tail_explicit_conflict_count")
    ):
        raise SparseUnionProbeError("PREFIX_OR_MASK_GATE")
    semantic = summary.get("semantic_trace")
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("rows") != expected_records
        or not isinstance(semantic.get("sha256"), str)
        or DIGEST_RE.fullmatch(str(semantic["sha256"])) is None
    ):
        raise SparseUnionProbeError("SEMANTIC_TRACE")
    if cache_enabled:
        cache = summary.get("cache")
        if not isinstance(cache, Mapping) or set(cache) != {
            "before_close", "after_close"
        }:
            raise SparseUnionProbeError("CACHE_ENVELOPE")
        before = _validate_cache(cache["before_close"], expected_closed=False)
        after = _validate_cache(cache["after_close"], expected_closed=True)
        for layer in ("fts_route", "product_view", "mask_decision"):
            for key in ("lookups", "hits", "misses", "evictions", "capacity"):
                if before[layer][key] != after[layer][key]:
                    raise SparseUnionProbeError("CACHE_CLOSE_COUNTER_DRIFT")
    _privacy_scan(dict(value))
    return dict(value)


def _invoke_worker(
    *,
    mode: str,
    cache_enabled: bool,
    session_limit: int,
    reference: Path,
    blobs: Mapping[str, str],
) -> WorkerRun:
    nonce = os.urandom(16).hex()
    if NONCE_RE.fullmatch(nonce) is None:
        raise SparseUnionProbeError("NONCE_GENERATION")
    launch, exit_code, receipt, parent_wall = _invoke(
        mode=mode,
        target_path=WORKER_PATH,
        target_module="scripts.sparse_union_g0_worker",
        target_blob=blobs[WORKER_RELATIVE],
        bootstrap_blob=blobs[BOOTSTRAP_RELATIVE],
        target_arguments=_worker_arguments(
            session_limit=session_limit,
            reference=reference,
            nonce=nonce,
            cache_enabled=cache_enabled,
            blobs=blobs,
        ),
        timeout=FORMAL_WALL_MAXIMUM,
        trace_filename=f"trace-{nonce}.jsonl",
    )
    try:
        if exit_code != 0:
            raise SparseUnionProbeError("WORKER_EXIT")
        validated = _validate_worker_receipt(
            receipt,
            nonce=nonce,
            session_limit=session_limit,
            cache_enabled=cache_enabled,
        )
        if launch.trace_path is None or not launch.trace_path.is_file():
            raise SparseUnionProbeError("WORKER_TRACE_MISSING")
        return WorkerRun(mode, cache_enabled, validated, parent_wall, launch)
    except BaseException:
        _cleanup_launch(launch)
        raise


def _validated_identifiers(
    values: object,
    *,
    minimum: int,
    maximum: int,
    catalog_ids: frozenset[str],
    code: str,
) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise SparseUnionProbeError(code)
    result = tuple(values)
    if (
        not minimum <= len(result) <= maximum
        or len(result) != len(set(result))
        or any(
            not isinstance(item, str) or item not in catalog_ids for item in result
        )
    ):
        raise SparseUnionProbeError(code)
    return result


def _validate_trace(
    trace_path: Path,
    reference_path: Path,
    *,
    session_limit: int,
    catalog_ids: frozenset[str],
    retain_records: bool,
) -> TraceAudit:
    trace = _require_plain(trace_path)
    reference = _require_plain(reference_path)
    digest = hashlib.sha256()
    trace_bytes = 0
    rows = 0
    candidate_cells = 0
    c200_cells = 0
    reference_prefix_bytes = 0
    expansion_turns = 0
    expansion_sessions: set[int] = set()
    minimum = MAX_CANDIDATES
    maximum = 0
    retained: list[tuple[tuple[str, ...], int]] = []
    expected_rows = session_limit * TURN_COUNT
    with trace.open("rb") as trace_handle, reference.open("rb") as reference_handle:
        trace_before = _snapshot(os.fstat(trace_handle.fileno()))
        reference_before = _snapshot(os.fstat(reference_handle.fileno()))
        for index in range(expected_rows):
            raw_trace = trace_handle.readline()
            raw_reference = reference_handle.readline()
            if not raw_trace or not raw_reference:
                raise SparseUnionProbeError("TRACE_ENDED_EARLY")
            trace_row = _parse_canonical_json(raw_trace, "TRACE_JSON")
            reference_row = _parse_canonical_json(raw_reference, "REFERENCE_JSON")
            ordinal = index // TURN_COUNT + 1
            turn = index % TURN_COUNT + 1
            if (
                set(trace_row) != {"candidates", "ordinal", "turn"}
                or set(reference_row) != {"c200", "ordinal", "turn"}
                or trace_row.get("ordinal") != ordinal
                or trace_row.get("turn") != turn
                or reference_row.get("ordinal") != ordinal
                or reference_row.get("turn") != turn
            ):
                raise SparseUnionProbeError("TRACE_COORDINATE")
            prefix = _validated_identifiers(
                reference_row["c200"],
                minimum=100,
                maximum=200,
                catalog_ids=catalog_ids,
                code="REFERENCE_CANDIDATES",
            )
            candidates = _validated_identifiers(
                trace_row["candidates"],
                minimum=len(prefix),
                maximum=MAX_CANDIDATES,
                catalog_ids=catalog_ids,
                code="TRACE_CANDIDATES",
            )
            if candidates[: len(prefix)] != prefix:
                raise SparseUnionProbeError("ORDERED_PREFIX_GATE")
            if len(candidates) > len(prefix):
                expansion_turns += 1
                expansion_sessions.add(ordinal)
            digest.update(raw_trace)
            trace_bytes += len(raw_trace)
            rows += 1
            reference_prefix_bytes += len(raw_reference)
            candidate_cells += len(candidates)
            c200_cells += len(prefix)
            minimum = min(minimum, len(candidates))
            maximum = max(maximum, len(candidates))
            if retain_records:
                retained.append((candidates, len(prefix)))
        if trace_handle.read(1):
            raise SparseUnionProbeError("TRACE_EXCESS_ROWS")
        if session_limit == SESSION_COUNT and reference_handle.read(1):
            raise SparseUnionProbeError("REFERENCE_EXCESS_ROWS")
        trace_after = _snapshot(os.fstat(trace_handle.fileno()))
        reference_after = _snapshot(os.fstat(reference_handle.fileno()))
    if (
        trace_before != trace_after
        or _snapshot(trace.stat()) != trace_after
        or reference_before != reference_after
        or _snapshot(reference.stat()) != reference_after
        or rows != expected_rows
        or expansion_turns <= 0
    ):
        raise SparseUnionProbeError("TRACE_IDENTITY_GATE")
    result = TraceAudit(
        bytes=trace_bytes,
        rows=rows,
        sha256=digest.hexdigest(),
        c200_cells=c200_cells,
        candidate_cells=candidate_cells,
        reference_prefix_bytes=reference_prefix_bytes,
        expansion_turns=expansion_turns,
        expansion_sessions=len(expansion_sessions),
        min_candidates=minimum,
        max_candidates=maximum,
        records=tuple(retained),
    )
    if (
        result.candidate_cells / result.c200_cells > CELL_RATIO_MAXIMUM
        or result.bytes / result.reference_prefix_bytes > TRACE_RATIO_MAXIMUM
    ):
        raise SparseUnionProbeError("TRACE_INFLATION_GATE")
    return result


def _files_equal(left: Path, right: Path) -> bool:
    with _require_plain(left).open("rb") as first, _require_plain(right).open(
        "rb"
    ) as second:
        while True:
            left_chunk = first.read(1 << 20)
            right_chunk = second.read(1 << 20)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _bind_trace(run: WorkerRun, trace: TraceAudit) -> None:
    summary = run.receipt["summary"]
    pool = summary.get("pool_lengths", {}).get("expanded_union", {})
    if (
        run.receipt.get("trace_bytes") != trace.bytes
        or run.receipt.get("trace_sha256") != trace.sha256
        or run.receipt.get("record_count") != trace.rows
        or not isinstance(pool, Mapping)
        or pool.get("candidate_cells") != trace.candidate_cells
        or pool.get("min") != trace.min_candidates
        or pool.get("max") != trace.max_candidates
    ):
        raise SparseUnionProbeError("TRACE_RECEIPT_BINDING")


def _worker_compact(run: WorkerRun, trace: TraceAudit) -> dict[str, Any]:
    summary = run.receipt["summary"]
    result = {
        "mode": run.mode,
        "cache_enabled": run.cache_enabled,
        "parent_wall_seconds": round(run.parent_wall_seconds, 6),
        "child_wall_seconds": summary["resources"]["wall_seconds"],
        "peak_working_set_bytes": summary["resources"]["peak_working_set_bytes"],
        "activation": summary["activation"],
        "latency": summary["latency"],
        "semantic_trace": summary["semantic_trace"],
        "trace": trace.compact(),
        "network_attempt_count": 0,
        "gpu_peak_bytes": 0,
    }
    if run.cache_enabled:
        result["cache"] = summary["cache"]
    return result


def _semantic_triplet_gate(
    runs: Sequence[WorkerRun], audits: Sequence[TraceAudit]
) -> None:
    if len(runs) != 3 or len(audits) != 3:
        raise SparseUnionProbeError("TRIPLET_SHAPE")
    trace_identities = {(item.bytes, item.rows, item.sha256) for item in audits}
    semantic = {
        (
            run.receipt["summary"]["semantic_trace"]["rows"],
            run.receipt["summary"]["semantic_trace"]["sha256"],
        )
        for run in runs
    }
    activations = {
        _canonical_sha256(run.receipt["summary"]["activation"]) for run in runs
    }
    if len(trace_identities) != 1 or len(semantic) != 1 or len(activations) != 1:
        raise SparseUnionProbeError("TRIPLET_SEMANTIC_MISMATCH")
    if not (
        _files_equal(runs[0].launch.trace_path, runs[1].launch.trace_path)  # type: ignore[arg-type]
        and _files_equal(runs[1].launch.trace_path, runs[2].launch.trace_path)  # type: ignore[arg-type]
    ):
        raise SparseUnionProbeError("TRIPLET_BYTES_MISMATCH")


def _run_stage(
    session_limit: int,
    *,
    blobs: Mapping[str, str],
    catalog_ids: frozenset[str],
) -> dict[str, Any]:
    runs: list[WorkerRun] = []
    try:
        runs.append(
            _invoke_worker(
                mode="direct",
                cache_enabled=False,
                session_limit=session_limit,
                reference=C200_REFERENCE_PATHS[0],
                blobs=blobs,
            )
        )
        runs.append(
            _invoke_worker(
                mode="direct",
                cache_enabled=True,
                session_limit=session_limit,
                reference=C200_REFERENCE_PATHS[0],
                blobs=blobs,
            )
        )
        runs.append(
            _invoke_worker(
                mode="module",
                cache_enabled=True,
                session_limit=session_limit,
                reference=C200_REFERENCE_PATHS[0],
                blobs=blobs,
            )
        )
        audits = [
            _validate_trace(
                run.launch.trace_path,  # type: ignore[arg-type]
                C200_REFERENCE_PATHS[0],
                session_limit=session_limit,
                catalog_ids=catalog_ids,
                retain_records=False,
            )
            for run in runs
        ]
        for run, audit in zip(runs, audits, strict=True):
            _bind_trace(run, audit)
        _semantic_triplet_gate(runs, audits)
        cached_pair_wall = math.fsum(run.parent_wall_seconds for run in runs[1:])
        extrapolated = max(run.parent_wall_seconds for run in runs[1:]) * (
            SESSION_COUNT / session_limit
        ) * 1.5
        if session_limit == 100 and (
            cached_pair_wall > PAIR_WALL_MAXIMUM
            or extrapolated > FORMAL_WALL_MAXIMUM
        ):
            raise SparseUnionProbeError("PREFLIGHT_WALL_GATE")
        return {
            "session_limit": session_limit,
            "exact_triplet": True,
            "cached_pair_parent_wall_seconds": round(cached_pair_wall, 6),
            "linear_extrapolation_x1_5_seconds": round(extrapolated, 6),
            "workers": [_worker_compact(run, audit) for run, audit in zip(runs, audits, strict=True)],
        }
    finally:
        for run in reversed(runs):
            _cleanup_launch(run.launch)


def _read_stable_plain(path: Path, maximum: int, code: str) -> bytes:
    resolved = _require_plain(path)
    with resolved.open("rb") as handle:
        before = _snapshot(os.fstat(handle.fileno()))
        if not 0 < before[0] <= maximum:
            raise SparseUnionProbeError(code)
        raw = handle.read(maximum + 1)
        after = _snapshot(os.fstat(handle.fileno()))
    if (
        len(raw) != before[0]
        or len(raw) > maximum
        or before != after
        or _snapshot(resolved.stat()) != after
    ):
        raise SparseUnionProbeError(code)
    return raw


def _raw_identity(raw: bytes) -> dict[str, int | str]:
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _valid_identity(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"bytes", "sha256"}
        and isinstance(value.get("bytes"), int)
        and not isinstance(value.get("bytes"), bool)
        and int(value["bytes"]) > 0
        and isinstance(value.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"])) is not None
    )


def _parse_durable_claim(
    path: Path,
    mode: str,
    implementation_commit: str,
    expected_blobs: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, int | str]]:
    raw = _read_stable_plain(path, 65_536, "CLAIM_FILE")
    value = _parse_canonical_json(raw, "CLAIM_JSON")
    expected_names = {
        "attempt_consumed",
        "branch",
        "experiment_id",
        "implementation_commit",
        "mode",
        "one_shot",
        "preregistration",
        "preregistration_commit",
        "recorded_on",
        "schema_version",
        "target_source_blobs",
    }
    if mode == "candidate":
        expected_names.add("preflight_prerequisite")
    preregistration = value.get("preregistration")
    source_blobs = value.get("target_source_blobs")
    if (
        mode not in {"preflight", "candidate"}
        or set(value) != expected_names
        or value.get("schema_version")
        != "small-ranker-v2.21-durable-one-shot-claim.v1"
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("branch") != BRANCH
        or value.get("mode") != mode
        or value.get("implementation_commit") != implementation_commit
        or value.get("preregistration_commit") != PREREG_COMMIT
        or value.get("recorded_on") != "2026-08-31"
        or value.get("attempt_consumed") is not True
        or value.get("one_shot") is not True
        or preregistration != {"blob": PREREG_BLOB, "commit": PREREG_COMMIT}
        or not isinstance(source_blobs, Mapping)
        or set(source_blobs) != {"bootstrap", "runner", "worker"}
        or any(COMMIT_RE.fullmatch(str(item)) is None for item in source_blobs.values())
    ):
        raise SparseUnionProbeError("CLAIM_SEMANTICS")
    if expected_blobs is not None and dict(source_blobs) != {
        "bootstrap": expected_blobs[BOOTSTRAP_RELATIVE],
        "runner": expected_blobs[RUNNER_RELATIVE],
        "worker": expected_blobs[WORKER_RELATIVE],
    }:
        raise SparseUnionProbeError("CLAIM_SOURCE_BLOBS")
    if mode == "candidate":
        prerequisite = value.get("preflight_prerequisite")
        if (
            not isinstance(prerequisite, Mapping)
            or set(prerequisite) != {"claim", "outer", "terminal"}
            or any(not _valid_identity(prerequisite.get(name)) for name in prerequisite)
        ):
            raise SparseUnionProbeError("CLAIM_PREREQUISITE")
    return value, _raw_identity(raw)


def _read_claim(
    path: Path,
    mode: str,
    implementation_commit: str,
    expected_blobs: Mapping[str, str] | None = None,
) -> dict[str, int | str]:
    _value, identity = _parse_durable_claim(
        path, mode, implementation_commit, expected_blobs
    )
    return identity


def _load_context_eligibility() -> tuple[int, ...]:
    values: list[int] = []
    with _require_plain(CONTEXT_PATH).open("rb") as handle:
        for raw in handle:
            container = _parse_json_object(raw, "CONTEXT_JSON")
            turns = container.get("turns")
            if (
                set(container) != {"schema_version", "turns"}
                or container.get("schema_version") != "small-ranker-visible-context.v1"
                or not isinstance(turns, list)
                or len(turns) != TURN_COUNT
            ):
                raise SparseUnionProbeError("CONTEXT_SCHEMA")
            versions: list[int] = []
            override_flags: list[bool] = []
            for turn in turns:
                version = turn.get("version") if isinstance(turn, Mapping) else None
                override = (
                    turn.get("current_turn_override")
                    if isinstance(turn, Mapping)
                    else None
                )
                if (
                    not isinstance(version, int)
                    or isinstance(version, bool)
                    or version < 1
                    or type(override) is not bool
                ):
                    raise SparseUnionProbeError("CONTEXT_VERSION")
                versions.append(version)
                override_flags.append(override)
            initial = versions[0]
            version_changed = next(
                (index + 1 for index, version in enumerate(versions) if version != initial),
                1,
            )
            explicit_override = next(
                (index + 1 for index, active in enumerate(override_flags) if active),
                1,
            )
            if version_changed != explicit_override:
                raise SparseUnionProbeError("CONTEXT_OVERRIDE_VERSION_DRIFT")
            changed = explicit_override
            if changed != 1 and not 2 <= changed <= TURN_COUNT:
                raise SparseUnionProbeError("CONTEXT_ELIGIBILITY")
            values.append(changed)
    if len(values) != SESSION_COUNT:
        raise SparseUnionProbeError("CONTEXT_SESSION_COUNT")
    return tuple(values)


def _parse_npy_integer(raw: bytes, *, expected_count: int) -> tuple[int, ...]:
    if len(raw) < 10 or raw[:6] != b"\x93NUMPY":
        raise SparseUnionProbeError("NPY_MAGIC")
    major, minor = raw[6], raw[7]
    del minor
    if major == 1:
        header_size = 2
        header_length = struct.unpack("<H", raw[8:10])[0]
        offset = 10
    elif major in {2, 3}:
        header_size = 4
        if len(raw) < 12:
            raise SparseUnionProbeError("NPY_HEADER")
        header_length = struct.unpack("<I", raw[8:12])[0]
        offset = 12
    else:
        raise SparseUnionProbeError("NPY_VERSION")
    del header_size
    header_end = offset + header_length
    if header_length <= 0 or header_end > len(raw):
        raise SparseUnionProbeError("NPY_HEADER")
    try:
        header = ast.literal_eval(raw[offset:header_end].decode("latin-1").strip())
    except (UnicodeError, ValueError, SyntaxError) as error:
        raise SparseUnionProbeError("NPY_HEADER") from error
    if (
        not isinstance(header, dict)
        or set(header) != {"descr", "fortran_order", "shape"}
        or header.get("fortran_order") is not False
        or header.get("shape") != (expected_count,)
    ):
        raise SparseUnionProbeError("NPY_SCHEMA")
    descriptor = header.get("descr")
    formats = {
        "|u1": (1, False, "little"),
        "<u1": (1, False, "little"),
        "<u2": (2, False, "little"),
        "<u4": (4, False, "little"),
        "<u8": (8, False, "little"),
        "<i1": (1, True, "little"),
        "|i1": (1, True, "little"),
        "<i2": (2, True, "little"),
        "<i4": (4, True, "little"),
        "<i8": (8, True, "little"),
    }
    if descriptor not in formats:
        raise SparseUnionProbeError("NPY_DTYPE")
    width, signed, byteorder = formats[str(descriptor)]
    payload = raw[header_end:]
    if len(payload) != expected_count * width:
        raise SparseUnionProbeError("NPY_PAYLOAD")
    return tuple(
        int.from_bytes(
            payload[index : index + width], byteorder=byteorder, signed=signed
        )
        for index in range(0, len(payload), width)
    )


def _load_numeric_labels() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    _identity(LABEL_PATH, LABEL_IDENTITY)
    members = ("outer_fold", "family_index", "taxonomy_code")
    arrays: list[tuple[int, ...]] = []
    try:
        with zipfile.ZipFile(_require_plain(LABEL_PATH), "r") as archive:
            for member in members:
                info = archive.getinfo(member + ".npy")
                if info.file_size > 100_000 or info.is_dir():
                    raise SparseUnionProbeError("LABEL_MEMBER_SIZE")
                with archive.open(info, "r") as handle:
                    raw = handle.read(100_001)
                if len(raw) != info.file_size:
                    raise SparseUnionProbeError("LABEL_MEMBER_READ")
                arrays.append(_parse_npy_integer(raw, expected_count=SESSION_COUNT))
    except (KeyError, OSError, zipfile.BadZipFile) as error:
        raise SparseUnionProbeError("LABEL_ARCHIVE") from error
    outer_fold, family_index, taxonomy_code = arrays
    if (
        any(value not in range(5) for value in outer_fold)
        or any(value < 0 for value in family_index)
        or any(value not in TAXONOMY_CODEBOOK for value in taxonomy_code)
    ):
        raise SparseUnionProbeError("LABEL_VALUES")
    final = _identity(LABEL_PATH, LABEL_IDENTITY)
    return outer_fold, family_index, taxonomy_code, {
        "bytes": final.bytes,
        "sha256": final.sha256,
        "members_read_in_order": list(members),
    }


def _load_proxy_targets(
    *,
    catalog_ids: frozenset[str],
    context_eligibility: Sequence[int],
) -> tuple[tuple[str, ...], tuple[int, ...], dict[str, Any]]:
    _identity(PROXY_PATH, PROXY_IDENTITY)
    targets: list[str] = []
    eligibility: list[int] = []
    with _require_plain(PROXY_PATH).open("rb") as handle:
        for index, raw in enumerate(handle):
            try:
                sample = json.loads(
                    raw.decode("utf-8", errors="strict"),
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (UnicodeError, ValueError, TypeError) as error:
                raise SparseUnionProbeError("PROXY_JSON") from error
            ground = sample.get("ground_truth") if isinstance(sample, Mapping) else None
            target = ground.get("parent_asin") if isinstance(ground, Mapping) else None
            scenario = sample.get("scenario_type") if isinstance(sample, Mapping) else None
            if (
                index >= SESSION_COUNT
                or not isinstance(target, str)
                or target not in catalog_ids
                or not isinstance(scenario, str)
            ):
                raise SparseUnionProbeError("PROXY_MEMBERSHIP_SCHEMA")
            visible_eligible = int(context_eligibility[index])
            if scenario == "intent_override":
                if visible_eligible < 2:
                    raise SparseUnionProbeError("OVERRIDE_ELIGIBILITY")
                effective = visible_eligible
                behavior = sample.get("behavior")
                override = behavior.get("override") if isinstance(behavior, Mapping) else None
                declared = override.get("turn") if isinstance(override, Mapping) else None
                if declared is not None and declared != effective:
                    raise SparseUnionProbeError("OVERRIDE_ELIGIBILITY_DRIFT")
            else:
                if visible_eligible != 1:
                    raise SparseUnionProbeError("NON_OVERRIDE_VERSION_DRIFT")
                effective = 1
            targets.append(target)
            eligibility.append(effective)
    if len(targets) != SESSION_COUNT:
        raise SparseUnionProbeError("PROXY_ROW_COUNT")
    final = _identity(PROXY_PATH, PROXY_IDENTITY)
    return tuple(targets), tuple(eligibility), {
        "bytes": final.bytes,
        "rows": final.rows,
        "sha256": final.sha256,
    }


VIEW_ORDER = (
    "SEALED_PREFIX_K10",
    "SEALED_PREFIX_K20",
    "SEALED_PREFIX_K50",
    "SEALED_PREFIX_K100",
    "SEALED_VARIABLE_C200",
    "EXPANDED_FIXED_K200",
    "C400_COMPLETE_UNION",
)


def _membership_flags(
    trace: TraceAudit,
    targets: Sequence[str],
    eligibility: Sequence[int],
) -> tuple[dict[str, bool], ...]:
    if (
        len(trace.records) != SESSION_COUNT * TURN_COUNT
        or len(targets) != SESSION_COUNT
        or len(eligibility) != SESSION_COUNT
    ):
        raise SparseUnionProbeError("MEMBERSHIP_DIMENSIONS")
    result: list[dict[str, bool]] = []
    for session in range(SESSION_COUNT):
        target = targets[session]
        eligible = eligibility[session]
        flags = {name: False for name in VIEW_ORDER}
        for turn in range(1, TURN_COUNT + 1):
            candidates, prefix_length = trace.records[
                session * TURN_COUNT + turn - 1
            ]
            if turn < eligible:
                continue
            prefix = candidates[:prefix_length]
            flags["SEALED_PREFIX_K10"] |= target in prefix[:10]
            flags["SEALED_PREFIX_K20"] |= target in prefix[:20]
            flags["SEALED_PREFIX_K50"] |= target in prefix[:50]
            flags["SEALED_PREFIX_K100"] |= target in prefix[:100]
            flags["SEALED_VARIABLE_C200"] |= target in prefix
            flags["EXPANDED_FIXED_K200"] |= target in candidates[:200]
            flags["C400_COMPLETE_UNION"] |= target in candidates
        result.append(flags)
    return tuple(result)


def _view_counts(
    flags: Sequence[Mapping[str, bool]], indices: Sequence[int]
) -> dict[str, dict[str, int | float]]:
    denominator = len(indices)
    result: dict[str, dict[str, int | float]] = {}
    for view in VIEW_ORDER:
        count = sum(int(flags[index][view]) for index in indices)
        result[view] = {
            "count": count,
            "fraction": round(count / denominator, 6) if denominator else 0.0,
        }
    return result


def _aggregate_candidate_recall(
    flags: Sequence[Mapping[str, bool]],
    *,
    targets: Sequence[str],
    outer_fold: Sequence[int],
    family_index: Sequence[int],
    taxonomy_code: Sequence[int],
) -> tuple[dict[str, Any], float]:
    if not (
        len(flags)
        == len(targets)
        == len(outer_fold)
        == len(family_index)
        == len(taxonomy_code)
        == SESSION_COUNT
    ):
        raise SparseUnionProbeError("AGGREGATE_DIMENSIONS")
    if any(
        set(row) != set(VIEW_ORDER)
        or any(type(row[view]) is not bool for view in VIEW_ORDER)
        for row in flags
    ):
        raise SparseUnionProbeError("AGGREGATE_FLAG_SCHEMA")
    family_fold: dict[int, int] = {}
    for family, fold in zip(family_index, outer_fold, strict=True):
        if family in family_fold and family_fold[family] != fold:
            raise SparseUnionProbeError("FAMILY_CROSSES_FOLD")
        family_fold[family] = fold
    all_indices = list(range(SESSION_COUNT))
    frontier = [
        index for index in all_indices if not flags[index]["SEALED_VARIABLE_C200"]
    ]
    increment = [
        index
        for index in frontier
        if flags[index]["C400_COMPLETE_UNION"]
    ]
    target_members: dict[str, list[int]] = {}
    for index, target in enumerate(targets):
        target_members.setdefault(target, []).append(index)
    baseline_uniform = math.fsum(
        math.fsum(
            int(flags[index]["SEALED_VARIABLE_C200"]) for index in members
        )
        / len(members)
        for members in target_members.values()
    ) / len(target_members)
    candidate_uniform = math.fsum(
        math.fsum(
            int(flags[index]["C400_COMPLETE_UNION"]) for index in members
        )
        / len(members)
        for members in target_members.values()
    ) / len(target_members)
    uniform_delta = candidate_uniform - baseline_uniform
    by_fold: list[dict[str, Any]] = []
    for fold in range(5):
        members = [index for index, value in enumerate(outer_fold) if value == fold]
        by_fold.append(
            {
                "fold": fold,
                "sessions": len(members),
                "views": _view_counts(flags, members),
                "increment": sum(index in increment for index in members),
            }
        )
    by_taxonomy: dict[str, Any] = {}
    for code, name in sorted(TAXONOMY_CODEBOOK.items(), key=lambda item: item[1]):
        members = [index for index, value in enumerate(taxonomy_code) if value == code]
        by_taxonomy[name] = {
            "sessions": len(members),
            "views": _view_counts(flags, members),
            "increment": sum(index in increment for index in members),
        }
    aggregate = {
        "all_2000_sessions": _view_counts(flags, all_indices),
        "c200_absent_frontier": {
            "sessions": len(frontier),
            "views": _view_counts(flags, frontier),
        },
        "increment": {
            "count": len(increment),
            "outer_fold_span": len({outer_fold[index] for index in increment}),
            "taxonomy_span": len({taxonomy_code[index] for index in increment}),
            "non_clothing_count": sum(
                taxonomy_code[index] != 1 for index in increment
            ),
            "target_cluster_count": len({targets[index] for index in increment}),
        },
        "by_outer_fold": by_fold,
        "by_taxonomy": by_taxonomy,
        "exact_target_cluster_uniform": {
            "cluster_count": len(target_members),
            "sealed_variable_c200_fraction": round(baseline_uniform, 9),
            "c400_complete_union_fraction": round(candidate_uniform, 9),
            "delta": round(uniform_delta, 9),
        },
        "family_disjoint_audit": {
            "valid": True,
            "family_count": len(family_fold),
            "families_crossing_outer_folds": 0,
        },
    }
    return aggregate, uniform_delta


class _FormalAuditGuard:
    """Supplement the bootstrap with phase-aware target and network guards."""

    def __init__(self) -> None:
        self.targets_allowed = False
        self.network_attempt_count = 0

    @staticmethod
    def _key(value: object) -> str:
        try:
            raw = os.fspath(value)
        except TypeError:
            return ""
        if isinstance(raw, bytes):
            try:
                raw = os.fsdecode(raw)
            except UnicodeError:
                return ""
        if not isinstance(raw, str):
            return ""
        return str(PureWindowsPath(raw)).replace("/", "\\").casefold()

    def hook(self, event: str, arguments: tuple[object, ...]) -> None:
        if event.startswith("socket."):
            self.network_attempt_count += 1
            raise PermissionError("network disabled")
        if event in {"open", "os.stat", "os.lstat"} and arguments:
            key = self._key(arguments[0])
            if not key:
                return
            proxy = self._key(PROXY_PATH)
            labels = self._key(LABEL_PATH)
            if not self.targets_allowed and key in {proxy, labels}:
                raise PermissionError("target source denied before closed traces")
            marker = "\\experiments\\fast_track\\"
            if marker in key:
                tail = key.split(marker, 1)[1]
                if tail.startswith(
                    ("small_ranker_v2_19_", "small_ranker_v2_20_", "small_ranker_v2_20b_")
                ):
                    raise PermissionError("old experiment namespace denied")
        if event in {"os.listdir", "os.scandir"} and arguments:
            key = self._key(arguments[0]).rstrip("\\")
            if key == self._key(EXPERIMENT_ROOT).rstrip("\\"):
                raise PermissionError("experiment directory enumeration denied")

    def allow_targets(self) -> None:
        if self.targets_allowed:
            raise SparseUnionProbeError("TARGET_GATE_REOPEN")
        self.targets_allowed = True


def _privacy_scan(value: object, catalog_ids: Iterable[str] = ()) -> None:
    catalog = {str(item).casefold() for item in catalog_ids}

    def walk(item: object) -> None:
        if isinstance(item, Mapping):
            keys = {str(key).casefold() for key in item}
            if keys & FORBIDDEN_RESULT_KEYS:
                raise SparseUnionProbeError("RESULT_FORBIDDEN_KEY")
            for child in item.values():
                walk(child)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            if len(item) >= 1_000:
                raise SparseUnionProbeError("RESULT_LONG_ARRAY")
            for child in item:
                walk(child)
        elif isinstance(item, str):
            if ASIN_RE.search(item) or item.casefold() in catalog:
                raise SparseUnionProbeError("RESULT_IDENTIFIER")

    walk(value)


def _validate_prior_pycache_prefix(value: object) -> None:
    if not isinstance(value, str):
        raise SparseUnionProbeError("PREFLIGHT_BOOTSTRAP_ATTESTATION")
    path = PureWindowsPath(value)
    base = PureWindowsPath(str(RUNTIME_BASE))
    if (
        path.name.casefold() != "pycache"
        or re.fullmatch(r"v221-[0-9a-f]{32}", path.parent.name) is None
        or path.parent.parent.as_posix().casefold()
        != base.as_posix().casefold().rstrip("/")
    ):
        raise SparseUnionProbeError("PREFLIGHT_BOOTSTRAP_ATTESTATION")


def _validate_preflight_receipt(
    receipt: object,
    *,
    implementation_commit: str,
    blobs: Mapping[str, str],
    claim_identity: Mapping[str, int | str],
    bootstrap: Mapping[str, Any],
) -> None:
    expected_names = {
        "bootstrap",
        "claim",
        "device",
        "entrypoint_regression",
        "experiment_id",
        "git",
        "implementation",
        "integrity",
        "mode",
        "next",
        "preregistration",
        "recorded_on",
        "rerun_forbidden",
        "runtime",
        "schema_version",
        "sources",
        "stages",
        "status",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_names:
        raise SparseUnionProbeError("PREFLIGHT_RECEIPT_SCHEMA")
    implementation = receipt.get("implementation")
    integrity = receipt.get("integrity")
    device = receipt.get("device")
    git = receipt.get("git")
    stages = receipt.get("stages")
    entrypoints = receipt.get("entrypoint_regression")
    sources = receipt.get("sources")
    preregistration = receipt.get("preregistration")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("mode") != "preflight"
        or receipt.get("status") != "TARGET_FREE_PREFLIGHT_COMPLETE"
        or receipt.get("recorded_on") != "2026-08-31"
        or receipt.get("rerun_forbidden") is not True
        or not _valid_identity(receipt.get("claim"))
        or receipt.get("claim") != claim_identity
        or receipt.get("bootstrap") != bootstrap
        or not isinstance(implementation, Mapping)
        or set(implementation)
        != {
            "commit",
            "branch",
            "preregistration_commit",
            "default_off",
            "target_blind",
            "served_top10_unchanged",
        }
        or implementation.get("commit") != implementation_commit
        or implementation.get("branch") != BRANCH
        or implementation.get("preregistration_commit") != PREREG_COMMIT
        or implementation.get("default_off") is not True
        or implementation.get("target_blind") is not True
        or implementation.get("served_top10_unchanged") is not True
        or not isinstance(integrity, Mapping)
        or set(integrity)
        != {
            "exact_triplet_each_stage",
            "ordered_variable_c200_prefix",
            "legacy_route_executions",
            "target_sources_opened",
            "network_attempt_count",
        }
        or integrity.get("exact_triplet_each_stage") is not True
        or integrity.get("ordered_variable_c200_prefix") is not True
        or type(integrity.get("legacy_route_executions")) is not int
        or integrity.get("legacy_route_executions") != 0
        or integrity.get("target_sources_opened") is not False
        or type(integrity.get("network_attempt_count")) is not int
        or integrity.get("network_attempt_count") != 0
        or not isinstance(device, Mapping)
        or set(device) != {"selected", "reason", "gpu_peak_bytes"}
        or device.get("selected") != "CPU"
        or device.get("reason")
        != "frozen sparse FTS/mask/Fraction-RRF backend"
        or type(device.get("gpu_peak_bytes")) is not int
        or device.get("gpu_peak_bytes") != 0
        or not isinstance(entrypoints, Mapping)
        or set(entrypoints)
        != {
            "runner_direct",
            "runner_module",
            "worker_direct",
            "worker_module",
            "legacy_module_denied_direct",
            "legacy_module_denied_module",
        }
        or any(value is not True for value in entrypoints.values())
        or not isinstance(preregistration, Mapping)
        or set(preregistration) != {"bytes", "rows", "sha256"}
        or type(preregistration.get("bytes")) is not int
        or preregistration.get("bytes") != PREREG_BYTES
        or type(preregistration.get("rows")) is not int
        or preregistration.get("rows", 0) <= 0
        or preregistration.get("sha256") != PREREG_SHA256
        or not isinstance(stages, list)
        or len(stages) != 2
        or any(not isinstance(stage, Mapping) for stage in stages)
        or any(type(stage.get("session_limit")) is not int for stage in stages)
        or [stage.get("session_limit") for stage in stages] != [20, 100]
        or any(stage.get("exact_triplet") is not True for stage in stages)
        or not isinstance(sources, Mapping)
        or set(sources) != {"catalog", "visible_context", "sealed_c200"}
        or not isinstance(git, Mapping)
        or set(git)
        != {
            "branch",
            "commit",
            "pushed",
            "remote",
            "preregistration_commit",
            "implementation_blobs",
            "object_only_git",
        }
        or git.get("branch") != BRANCH
        or git.get("commit") != implementation_commit
        or git.get("pushed") is not True
        or git.get("remote") != REMOTE_URL
        or git.get("preregistration_commit") != PREREG_COMMIT
        or git.get("implementation_blobs") != blobs
        or git.get("object_only_git") is not True
    ):
        raise SparseUnionProbeError("PREFLIGHT_RECEIPT_NOT_ELIGIBLE")


def _validate_preflight_chain(
    implementation_commit: str, blobs: Mapping[str, str]
) -> dict[str, dict[str, int | str]]:
    _claim, claim_identity = _parse_durable_claim(
        PREFLIGHT_CLAIM_PATH,
        "preflight",
        implementation_commit,
        blobs,
    )
    outer_raw = _read_stable_plain(PREFLIGHT_OUTER_PATH, 1 << 20, "PREFLIGHT_OUTER_FILE")
    outer = _parse_canonical_json(outer_raw, "PREFLIGHT_OUTER_JSON")
    outer_identity = _raw_identity(outer_raw)
    if set(outer) != {"bootstrap", "target_exit_code", "target_receipt"}:
        raise SparseUnionProbeError("PREFLIGHT_OUTER_SCHEMA")
    bootstrap = outer.get("bootstrap")
    if (
        not isinstance(bootstrap, Mapping)
        or set(bootstrap)
        != {
            "bootstrap_blob",
            "guarded_path",
            "mode",
            "pycache_prefix",
            "source_only",
            "target_blob",
        }
        or bootstrap.get("bootstrap_blob") != blobs[BOOTSTRAP_RELATIVE]
        or bootstrap.get("target_blob") != blobs[RUNNER_RELATIVE]
        or bootstrap.get("mode") != "direct"
        or bootstrap.get("guarded_path") is not True
        or bootstrap.get("source_only") is not True
        or outer.get("target_exit_code") != 0
        or isinstance(outer.get("target_exit_code"), bool)
        or not isinstance(outer.get("target_receipt"), Mapping)
    ):
        raise SparseUnionProbeError("PREFLIGHT_OUTER_NOT_COMPLETE")
    _validate_prior_pycache_prefix(bootstrap.get("pycache_prefix"))
    receipt = outer["target_receipt"]
    _validate_preflight_receipt(
        receipt,
        implementation_commit=implementation_commit,
        blobs=blobs,
        claim_identity=claim_identity,
        bootstrap=bootstrap,
    )

    terminal_raw = _read_stable_plain(
        PREFLIGHT_RESULT_PATH, 1 << 20, "PREFLIGHT_TERMINAL_FILE"
    )
    terminal = _parse_canonical_json(terminal_raw, "PREFLIGHT_TERMINAL_JSON")
    terminal_identity = _raw_identity(terminal_raw)
    if (
        set(terminal)
        != {
            "implementation_commit",
            "mode",
            "outer",
            "preregistration",
            "process_exit_code",
            "raw_stderr_retained",
            "recorded_on",
            "schema_version",
            "status",
            "target_exit_code",
            "target_receipt",
        }
        or terminal.get("schema_version")
        != "small-ranker-v2.21-durable-terminal.v1"
        or terminal.get("status") != "COMPLETE"
        or terminal.get("mode") != "preflight"
        or terminal.get("implementation_commit") != implementation_commit
        or terminal.get("recorded_on") != "2026-08-31"
        or terminal.get("process_exit_code") != 0
        or isinstance(terminal.get("process_exit_code"), bool)
        or terminal.get("target_exit_code") != 0
        or isinstance(terminal.get("target_exit_code"), bool)
        or terminal.get("raw_stderr_retained") is not False
        or terminal.get("preregistration")
        != {"blob": PREREG_BLOB, "commit": PREREG_COMMIT}
        or terminal.get("outer") != outer_identity
        or terminal.get("target_receipt") != receipt
    ):
        raise SparseUnionProbeError("PREFLIGHT_TERMINAL_NOT_COMPLETE")
    _privacy_scan(terminal)
    return {
        "claim": dict(claim_identity),
        "outer": dict(outer_identity),
        "terminal": dict(terminal_identity),
    }


def _validate_preflight_terminal(
    implementation_commit: str, blobs: Mapping[str, str]
) -> dict[str, dict[str, int | str]]:
    """Compatibility name retained for the frozen runner's internal call site."""

    return _validate_preflight_chain(implementation_commit, blobs)


def _run_preflight(
    *,
    implementation_commit: str,
    git_report: Mapping[str, Any],
    attestation: Mapping[str, Any],
    audit_guard: _FormalAuditGuard,
) -> dict[str, Any]:
    claim = _read_claim(
        PREFLIGHT_CLAIM_PATH,
        "preflight",
        implementation_commit,
        git_report["implementation_blobs"],
    )
    entrypoints = _verify_entrypoints(git_report["implementation_blobs"])
    sources_before = _source_checkpoint()
    catalog_ids = _load_catalog_ids()
    stage20 = _run_stage(
        20,
        blobs=git_report["implementation_blobs"],
        catalog_ids=catalog_ids,
    )
    stage100 = _run_stage(
        100,
        blobs=git_report["implementation_blobs"],
        catalog_ids=catalog_ids,
    )
    sources_after = _source_checkpoint()
    final_git = _validate_git_checkpoint(implementation_commit)
    if sources_before != sources_after or final_git != git_report:
        raise SparseUnionProbeError("PREFLIGHT_SOURCE_MUTATION")
    if audit_guard.targets_allowed or audit_guard.network_attempt_count:
        raise SparseUnionProbeError("PREFLIGHT_AUDIT_GATE")
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "mode": "preflight",
        "status": "TARGET_FREE_PREFLIGHT_COMPLETE",
        "recorded_on": "2026-08-31",
        "rerun_forbidden": True,
        "claim": claim,
        "implementation": {
            "commit": implementation_commit,
            "branch": BRANCH,
            "preregistration_commit": PREREG_COMMIT,
            "default_off": True,
            "target_blind": True,
            "served_top10_unchanged": True,
        },
        "bootstrap": dict(attestation),
        "entrypoint_regression": entrypoints,
        "stages": [stage20, stage100],
        "sources": sources_after,
        "device": {
            "selected": "CPU",
            "reason": "frozen sparse FTS/mask/Fraction-RRF backend",
            "gpu_peak_bytes": 0,
        },
        "integrity": {
            "exact_triplet_each_stage": True,
            "ordered_variable_c200_prefix": True,
            "legacy_route_executions": 0,
            "target_sources_opened": False,
            "network_attempt_count": 0,
        },
        "git": final_git,
        "next": "one immutable full-2000 candidate-recall receipt",
    }
    _privacy_scan(result, catalog_ids)
    return result


def _pair_semantic_gate(
    first: WorkerRun,
    second: WorkerRun,
    first_trace: TraceAudit,
    second_trace: TraceAudit,
) -> None:
    _bind_trace(first, first_trace)
    _bind_trace(second, second_trace)
    left_summary = first.receipt["summary"]
    right_summary = second.receipt["summary"]
    if (
        (first_trace.bytes, first_trace.rows, first_trace.sha256)
        != (second_trace.bytes, second_trace.rows, second_trace.sha256)
        or left_summary["semantic_trace"] != right_summary["semantic_trace"]
        or left_summary["activation"] != right_summary["activation"]
        or left_summary["input_identities"] != right_summary["input_identities"]
        or left_summary["source_identities"] != right_summary["source_identities"]
        or not _files_equal(first.launch.trace_path, second.launch.trace_path)  # type: ignore[arg-type]
    ):
        raise SparseUnionProbeError("FULL_PAIR_EXACT_REPEAT")


def _run_candidate(
    *,
    implementation_commit: str,
    git_report: Mapping[str, Any],
    attestation: Mapping[str, Any],
    audit_guard: _FormalAuditGuard,
) -> dict[str, Any]:
    formal_started = time.perf_counter()
    claim_value, claim = _parse_durable_claim(
        CANDIDATE_CLAIM_PATH,
        "candidate",
        implementation_commit,
        git_report["implementation_blobs"],
    )
    preflight_prerequisite = _validate_preflight_chain(
        implementation_commit, git_report["implementation_blobs"]
    )
    if claim_value["preflight_prerequisite"] != preflight_prerequisite:
        raise SparseUnionProbeError("CANDIDATE_CLAIM_PREFLIGHT_BINDING")
    sources_before = _source_checkpoint()
    catalog_ids = _load_catalog_ids()
    context_eligibility = _load_context_eligibility()
    runs: list[WorkerRun] = []
    try:
        runs.append(
            _invoke_worker(
                mode="direct",
                cache_enabled=True,
                session_limit=SESSION_COUNT,
                reference=C200_REFERENCE_PATHS[0],
                blobs=git_report["implementation_blobs"],
            )
        )
        if time.perf_counter() - formal_started >= FORMAL_WALL_MAXIMUM:
            raise SparseUnionProbeError("FORMAL_WALL_BEFORE_REPLICA_B")
        runs.append(
            _invoke_worker(
                mode="module",
                cache_enabled=True,
                session_limit=SESSION_COUNT,
                reference=C200_REFERENCE_PATHS[1],
                blobs=git_report["implementation_blobs"],
            )
        )
        first_trace = _validate_trace(
            runs[0].launch.trace_path,  # type: ignore[arg-type]
            C200_REFERENCE_PATHS[0],
            session_limit=SESSION_COUNT,
            catalog_ids=catalog_ids,
            retain_records=True,
        )
        second_trace = _validate_trace(
            runs[1].launch.trace_path,  # type: ignore[arg-type]
            C200_REFERENCE_PATHS[1],
            session_limit=SESSION_COUNT,
            catalog_ids=catalog_ids,
            retain_records=False,
        )
        _pair_semantic_gate(runs[0], runs[1], first_trace, second_trace)
        sources_pre_attach = _source_checkpoint()
        git_pre_attach = _validate_git_checkpoint(implementation_commit)
        preflight_pre_attach = _validate_preflight_chain(
            implementation_commit, git_report["implementation_blobs"]
        )
        claim_value_pre_attach, claim_pre_attach = _parse_durable_claim(
            CANDIDATE_CLAIM_PATH,
            "candidate",
            implementation_commit,
            git_report["implementation_blobs"],
        )
        elapsed_pre_attach = time.perf_counter() - formal_started
        if (
            sources_pre_attach != sources_before
            or git_pre_attach != git_report
            or preflight_pre_attach != preflight_prerequisite
            or claim_value_pre_attach != claim_value
            or claim_pre_attach != claim
            or claim_value_pre_attach["preflight_prerequisite"]
            != preflight_pre_attach
            or elapsed_pre_attach > FORMAL_WALL_MAXIMUM
            or any(run.parent_wall_seconds > FORMAL_WALL_MAXIMUM for run in runs)
            or math.fsum(run.parent_wall_seconds for run in runs)
            > FORMAL_WALL_MAXIMUM
            or audit_guard.targets_allowed
            or audit_guard.network_attempt_count
        ):
            raise SparseUnionProbeError("PRE_TARGET_CLOSED_TRACE_GATE")

        # This is the first point at which either target-bearing source may be
        # opened.  Both traces are closed, published, byte-identical, and bound
        # to compact worker receipts before this irreversible phase change.
        audit_guard.allow_targets()
        targets, eligibility, proxy_identity = _load_proxy_targets(
            catalog_ids=catalog_ids,
            context_eligibility=context_eligibility,
        )
        outer_fold, family_index, taxonomy_code, label_identity = (
            _load_numeric_labels()
        )
        flags = _membership_flags(first_trace, targets, eligibility)
        aggregate, uniform_delta = _aggregate_candidate_recall(
            flags,
            targets=targets,
            outer_fold=outer_fold,
            family_index=family_index,
            taxonomy_code=taxonomy_code,
        )
        del flags, targets, eligibility, outer_fold, family_index, taxonomy_code
        all_views = aggregate["all_2000_sessions"]
        for view, expected in BASELINE_SANITY.items():
            if all_views[view]["count"] != expected:
                raise SparseUnionProbeError("BASELINE_SANITY")
        if (
            all_views["EXPANDED_FIXED_K200"]["count"]
            < all_views["SEALED_VARIABLE_C200"]["count"]
            or all_views["C400_COMPLETE_UNION"]["count"]
            < all_views["EXPANDED_FIXED_K200"]["count"]
        ):
            raise SparseUnionProbeError("MEMBERSHIP_MONOTONICITY")
        increment = aggregate["increment"]
        promoted = bool(
            all_views["C400_COMPLETE_UNION"]["count"] >= 1_988
            and increment["count"] >= 2
            and increment["outer_fold_span"] >= 2
            and increment["non_clothing_count"] >= 1
            and uniform_delta > 0.0
        )
        final_sources = _source_checkpoint(include_targets=True)
        final_sources["proxy"] = proxy_identity
        final_sources["numeric_label_archive"] = label_identity
        final_git = _validate_git_checkpoint(implementation_commit)
        total_wall = time.perf_counter() - formal_started
        if (
            final_git != git_report
            or final_sources["catalog"] != sources_before["catalog"]
            or final_sources["visible_context"] != sources_before["visible_context"]
            or final_sources["sealed_c200"] != sources_before["sealed_c200"]
            or total_wall > FORMAL_WALL_MAXIMUM
            or audit_guard.network_attempt_count
        ):
            raise SparseUnionProbeError("FINAL_SOURCE_OR_RESOURCE_GATE")
        status = (
            "CANDIDATE_RECALL_GO_ALLOW_SEPARATE_POLICY_PREREGISTRATION"
            if promoted
            else "CANDIDATE_RECALL_NO_GO_FREEZE_EXACT_G0"
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "mode": "candidate",
            "status": status,
            "recorded_on": "2026-08-31",
            "rerun_forbidden": True,
            "evidence_scope": (
                "shared 2000-session diagnostic candidate membership; "
                "not served HR@10 and not private validation"
            ),
            "claim": claim,
            "preflight_prerequisite": preflight_prerequisite,
            "implementation": {
                "commit": implementation_commit,
                "branch": BRANCH,
                "preregistration_commit": PREREG_COMMIT,
                "default_off": True,
                "runtime_target_blind": True,
                "served_top10_unchanged": True,
                "full_agent_evaluator_started": False,
            },
            "bootstrap": dict(attestation),
            "candidate_recall": aggregate,
            "baseline_sanity": BASELINE_SANITY,
            "candidate_retention": {
                "complete_variable_c200_exact_ordered_prefix": True,
                "c200_loss_count": 0,
                "c200_reorder_count": 0,
                "c200_duplicate_count": 0,
                "tail_duplicate_count": 0,
                "tail_explicit_hard_conflict_count": 0,
                "served_top10_unchanged": True,
            },
            "exact_repeat": {
                "passed": True,
                "trace_bytes": first_trace.bytes,
                "trace_rows": first_trace.rows,
                "trace_sha256": first_trace.sha256,
                "semantic_sha256": runs[0].receipt["summary"]["semantic_trace"]["sha256"],
            },
            "resources": {
                "total_formal_parent_wall_seconds": round(total_wall, 6),
                "workers": [
                    _worker_compact(runs[0], first_trace),
                    _worker_compact(runs[1], second_trace),
                ],
                "network_attempt_count": 0,
                "gpu_peak_bytes": 0,
                "budgets_passed": True,
            },
            "sources": final_sources,
            "git": final_git,
            "decision": {
                "promotion_gate_passed": promoted,
                "candidate_threshold": 1_988,
                "top10_global_promotion": False,
                "next_stage": (
                    "separate preregistration for the 100-session policy smoke"
                    if promoted
                    else "freeze this exact G0 and choose the next independent mechanism"
                ),
                "fallback_order": [
                    "SR-V2.12-FIXED-TWO-PAGE-GRACE",
                    "v1.9",
                    "P11",
                    "R08",
                ],
            },
        }
        _privacy_scan(result, catalog_ids)
        return result
    finally:
        for run in reversed(runs):
            _cleanup_launch(run.launch)


def run(mode: str, implementation_commit: str) -> dict[str, Any]:
    if mode not in {"preflight", "candidate"}:
        raise SparseUnionProbeError("MODE")
    audit_guard = _FormalAuditGuard()
    sys.addaudithook(audit_guard.hook)
    runtime = _validate_runtime()
    preregistration = _validate_preregistration()
    git_report = _validate_git_checkpoint(implementation_commit)
    attestation = _require_attestation(git_report)
    result = (
        _run_preflight(
            implementation_commit=implementation_commit,
            git_report=git_report,
            attestation=attestation,
            audit_guard=audit_guard,
        )
        if mode == "preflight"
        else _run_candidate(
            implementation_commit=implementation_commit,
            git_report=git_report,
            attestation=attestation,
            audit_guard=audit_guard,
        )
    )
    result["runtime"] = runtime
    result["preregistration"] = preregistration
    _privacy_scan(result)
    return result


def _entrypoint_self_check(arguments: Sequence[str]) -> int | None:
    if "--entrypoint-self-check" not in arguments:
        return None
    parser = _SafeArgumentParser(add_help=False)
    parser.add_argument("--entrypoint-self-check", action="store_true")
    parser.add_argument(
        "--require-module",
        default="starter.sparse_union_g0",
        choices=("starter.sparse_union_g0",),
    )
    parsed = parser.parse_args(arguments)
    if "starter.sparse_multiview" in sys.modules:
        raise SparseUnionProbeError("LEGACY_RUNTIME_PRELOADED")
    required = importlib.import_module(parsed.require_module)
    contract = importlib.import_module("scripts.c200_candidate_worker")
    evaluator = importlib.import_module("evaluator.local_evaluator")
    if (
        required.__name__ != "starter.sparse_union_g0"
        or contract.__name__ != "scripts.c200_candidate_worker"
        or evaluator.__name__ != "evaluator.local_evaluator"
        or Path(str(getattr(evaluator, "__file__", ""))).resolve(strict=True)
        != (ROOT / "evaluator" / "local_evaluator.py").resolve(strict=True)
    ):
        raise SparseUnionProbeError("ENTRYPOINT_IMPORT")
    payload = {
        "c200_contract_imported": True,
        "evaluator_imported": True,
        "legacy_runtime_absent": True,
        "project_root_bootstrapped": str(ROOT) in sys.path,
        "required_module": parsed.require_module,
        "status": "ENTRYPOINT_SELF_CHECK_PASS",
    }
    sys.stdout.buffer.write(_canonical_bytes(payload) + b"\n")
    sys.stdout.buffer.flush()
    return 0


def _preclaim_chain_self_check(arguments: Sequence[str]) -> int | None:
    if "--preclaim-chain-self-check" not in arguments:
        return None
    parser = _SafeArgumentParser(add_help=False)
    parser.add_argument("--preclaim-chain-self-check", action="store_true")
    parser.add_argument("--implementation-commit", required=True)
    parsed = parser.parse_args(arguments)
    commit = str(parsed.implementation_commit)
    if COMMIT_RE.fullmatch(commit) is None:
        raise SparseUnionProbeError("CLI_CONTRACT")
    git_report = _validate_git_checkpoint(commit)
    prerequisite = _validate_preflight_chain(
        commit, git_report["implementation_blobs"]
    )
    payload = {
        "implementation_commit": commit,
        "preflight_prerequisite": prerequisite,
        "status": "PRECLAIM_PREFLIGHT_CHAIN_PASS",
    }
    _privacy_scan(payload)
    sys.stdout.buffer.write(_canonical_bytes(payload) + b"\n")
    sys.stdout.buffer.flush()
    return 0


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise SparseUnionProbeError("CLI_CONTRACT")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--mode", choices=("preflight", "candidate"), required=True)
    parser.add_argument("--implementation-commit", required=True)
    return parser


def _error_receipt(error: BaseException, *, mode: str) -> dict[str, Any]:
    code = error.code if isinstance(error, SparseUnionProbeError) else "UNEXPECTED_EXCEPTION"
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "mode": mode if mode in {"preflight", "candidate"} else "unknown",
        "status": "INVALID_ONE_SHOT_CONSUMED",
        "error_code": code,
        "rerun_forbidden": True,
        "traceback_disclosed": False,
        "identifiers_disclosed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        self_check = _entrypoint_self_check(raw)
        if self_check is not None:
            return self_check
        chain_check = _preclaim_chain_self_check(raw)
        if chain_check is not None:
            return chain_check
    except BaseException:
        return 2
    mode = "unknown"
    try:
        parsed = _parser().parse_args(raw)
        mode = str(parsed.mode)
        if not parsed.run or COMMIT_RE.fullmatch(str(parsed.implementation_commit)) is None:
            raise SparseUnionProbeError("CLI_CONTRACT")
        receipt = run(mode, str(parsed.implementation_commit))
        exit_code = 0
    except BaseException as error:
        receipt = _error_receipt(error, mode=mode)
        exit_code = 1
    sys.stdout.buffer.write(_canonical_bytes(receipt) + b"\n")
    sys.stdout.buffer.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
