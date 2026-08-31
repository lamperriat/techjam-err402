"""Target-free v2.20 semantic-cache preflight.

This runner is intentionally independent of the consumed v2.19 one-shot
runner.  It opens only the preregistered catalog, visible-context cache, and
two C200 references after a clean, pushed implementation checkpoint passes.
For each 20/100-session stage it runs one cache-disabled semantic control and
two fresh cache-enabled workers, then requires exact trace and semantic-digest
parity.  It never imports an evaluator-side target helper and never opens a
proxy, label, fold, outcome, v2.19 receipt, or v2.19 cache namespace.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import ntpath
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat as stat_module
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True


SCHEMA_VERSION = "small-ranker-v2.20-sparse-cache-preflight.v1"
WORKER_SCHEMA_VERSION = "small-ranker-registry-ca-g0-worker-summary.v1"
EXPERIMENT_ID = "SR-V2.20-SPARSE-ROUTE-CACHE"
BRANCH = "small-ranker-v2.20-sparse-cache"
REMOTE = "origin"
REMOTE_URL = "https://github.com/lamperriat/techjam-err402.git"
REMOTE_REF = f"refs/remotes/{REMOTE}/{BRANCH}"

V219_BLOCKER_COMMIT = "c51e18de717b58aed7bbefc8db63a4b783178c43"
PREREG_COMMIT = "2a1aa5d51da4b9d621687b9101c26bafcac82ccb"
PREREG_RELATIVE = "configs/small_ranker_v2_20.sparse_route_cache_preregistration.json"
PREREG_PATH = ROOT / PREREG_RELATIVE
PREREG_BLOB = "3cad1aba4f92d1107dd5179c24861d370d8e321b"
PREREG_BYTES = 19_218
PREREG_RAW_SHA256 = "2c1099566164f7e43ce3d268d755b6cf0984b0e9f3f1ac8f6a2f9cc261154016"
PREREG_CANONICAL_SHA256 = (
    "e5d0a47606b3f5d3590f33630ec4d5f40e0e39a68443ccfc625ee3d16c5cc083"
)
PREREG_PATHS = {PREREG_RELATIVE}

IMPLEMENTATION_PATHS = {
    "starter/sparse_multiview.py",
    "scripts/sparse_multiview_candidate_worker.py",
    "scripts/probe_sparse_multiview_cache_preflight.py",
    "tests/test_sparse_multiview_cache.py",
}
FROZEN_PARENT_BLOBS = {
    "configs/small_ranker_v2_19.registry_ca_g0_preregistration.json": (
        "e480cb7efd7c3ed80f2751e843577052430ea599"
    ),
    "configs/small_ranker_v2_19.registry_ca_g0_preflight_blocker.json": (
        "0d2d7c2cd551e0150bc8cd6ec4e1aac399e44c39"
    ),
    "evaluator/local_evaluator.py": "7c808347b31ef3121a9cbc4810ac3eb325f950ba",
    "scripts/c200_candidate_worker.py": "b94fddcf5a9b20ddde540f3f43ea9962982cb096",
    "scripts/probe_c200_candidate_recall.py": "0a57f63866683b476b9f49184673cf3154531911",
    "scripts/probe_e0_embedding_candidate_recall.py": (
        "5bb9ec7f38f90d814d0c121c9f8992267d3491d5"
    ),
    "starter/agent.py": "421c6d43c598102b8fefb181b72bab5da4bf1294",
    "starter/architecture_lab.py": "8d340d0dce3fc2f1bb987a5dd632444776a05667",
    "starter/attributes.py": "92260323f077c9861aa4edd5242aff772c875760",
    "starter/p8_negative.py": "719078234dba297ce59f68d8a2b1734ec53c9c63",
    "starter/slot_ledger.py": "72975cff12af59e4044e52911c58294cd74a785a",
    "starter/sparse_multiview.py": "82cbd10399ac02533a20057d3523e09ad729811b",
    "scripts/sparse_multiview_candidate_worker.py": (
        "59769b480c0357d535aa10afcf740152027acdbf"
    ),
    "scripts/probe_sparse_multiview_candidate_recall.py": (
        "ae96a43611c0d0f835c0e0c2dda5f9e798fc7827"
    ),
    "tests/test_sparse_multiview_candidate_recall.py": (
        "7e1c10f6b45f1c296b3616dc014ec16fde555356"
    ),
}

SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
CATALOG_PATH = SOURCE_ROOT / "data/catalog.jsonl"
CATALOG_BYTES = 60_546_327
CATALOG_ROWS = 50_000
CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"

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

RESULT_PATH = ROOT / "experiments/fast_track/small_ranker_v2_20_sparse_cache_preflight_20260831.json"
WORKER_PATH = ROOT / "scripts/sparse_multiview_candidate_worker.py"
RUNNER_PATH = Path(__file__).resolve()

# These values are lexical sentinels only.  No code in this module resolves,
# stats, opens, hashes, creates, or removes either path.
V219_RESULT_DENIED_PATH = (
    ROOT / "experiments/fast_track/small_ranker_v2_19_registry_ca_g0_20260831.json"
)
V219_CACHE_DENIED_ROOT = (
    ROOT / "experiments/fast_track/small_ranker_v2_19_registry_ca_g0_cache_20260831"
)

# These lexical markers supplement exact allowlists.  They are deliberately
# broader than the current source names because v2.20 is now fail-closed before
# formal execution (see ``run``); a corrected preregistration may reuse them.
PROTECTED_PATH_MARKERS = (
    "train_explore", "proxy", "target", "identifier", "oracle", "label",
    "counterfactual", "fold", "taxonomy", "outcome", "calibration",
    "selection", "confirmation", "public", "amazon", "heldout",
)

EXPECTED_EXECUTABLE = Path(r"D:\450\conda\envs\tiktok\python.exe")
EXPECTED_PYTHON = "3.11.16"
EXPECTED_SQLITE = "3.53.4"
SESSION_COUNT = 2_000
TURN_COUNT = 10
ALLOWED_STAGE_LIMITS = (20, 100)
MIN_C200_CANDIDATES = 100
MAX_C200_CANDIDATES = 200
MAX_CANDIDATES = 400

FROZEN_20_TRACE_ROWS = 200
FROZEN_20_TRACE_BYTES = 441_241
FROZEN_20_TRACE_SHA256 = "e5177f1a69fe1e79d5d9d4729952c9dfcfac0325689aa13e31dc860fbf38e45a"
FROZEN_20_ACTIVATED_TURNS = 116

PAIR_WALL_SECONDS_MAXIMUM = 60.0
FORMAL_WALL_SECONDS_MAXIMUM = 1_800.0
EXTRA_ROUTE_P95_MS_MAXIMUM = 100.0
PER_TURN_P95_MS_MAXIMUM = 400.0
WORKER_RSS_MAXIMUM = 1_610_612_736
CELL_RATIO_MAXIMUM = 2.0
TRACE_RATIO_MAXIMUM = 2.1
FREE_DISK_MINIMUM = 536_870_912
RESULT_BYTES_MAXIMUM = 24_000
CACHE_CAPACITIES = {
    "fts_route": 256,
    "product_view": 4_096,
    "mask_decision": 16_384,
}

COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
CATALOG_ID_RE = re.compile(r"[A-Z0-9]{10}\Z")
ASIN_SHAPE_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.I)
FORBIDDEN_RESULT_KEYS = frozenset({
    "asin", "parent_asin", "sample_id", "session_id", "product_id",
    "ground_truth", "target", "target_id", "target_asin", "targets",
    "positive_index", "eligible_from", "eligibility", "message", "messages",
    "query", "queries", "query_terms", "rule", "rules", "identifier",
    "identifiers", "ordinal", "turn", "outer_fold", "family_index",
    "per_session", "membership", "membership_vector",
    "candidates", "c200", "trace_records", "raw_trace",
})

_REGISTERED_TEMP_ROOTS: set[str] = set()


class SparseCachePreflightError(RuntimeError):
    """Sanitized target-free preflight failure with a stable error code."""

    def __init__(self, message: str, code: str = "CONTRACT_DRIFT") -> None:
        super().__init__(message)
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
class TraceValidation:
    trace_sha256: str
    trace_bytes: int
    record_count: int
    activation_turns: int
    activation_sessions: int
    candidate_cells: int
    c200_cells: int
    reference_prefix_bytes: int
    min_candidates: int
    max_candidates: int

    def report(self) -> dict[str, int | str | float]:
        return {
            "candidate_expansion_sessions": self.activation_sessions,
            "candidate_expansion_turns": self.activation_turns,
            "bytes": self.trace_bytes,
            "candidate_cell_ratio_over_c200": round(
                self.candidate_cells / self.c200_cells, 6
            ),
            "max_candidates": self.max_candidates,
            "min_candidates": self.min_candidates,
            "record_count": self.record_count,
            "sha256": self.trace_sha256,
            "trace_byte_ratio_over_c200": round(
                self.trace_bytes / self.reference_prefix_bytes, 6
            ),
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
        raise SparseCachePreflightError("value is not canonical JSON", "INVALID_JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SparseCachePreflightError("duplicate JSON key", "INVALID_JSON")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise SparseCachePreflightError("non-finite JSON number", "INVALID_JSON")


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except SparseCachePreflightError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SparseCachePreflightError(f"{label} is invalid JSON", "INVALID_JSON") from error
    if not isinstance(value, dict):
        raise SparseCachePreflightError(f"{label} must be an object", "INVALID_JSON")
    return value


def _snapshot(value: os.stat_result) -> tuple[int, int, int]:
    return (
        int(value.st_size),
        int(value.st_mtime_ns),
        int(getattr(value, "st_ino", 0)),
    )


def _lexical_key(path: str | os.PathLike[str]) -> str:
    """Return a Windows lexical key without resolving or touching the path."""

    return ntpath.normcase(ntpath.abspath(os.fspath(path))).rstrip("\\/")


def _is_v219_denied_lexically(path: str | os.PathLike[str]) -> bool:
    key = _lexical_key(path)
    result_key = _lexical_key(V219_RESULT_DENIED_PATH)
    cache_key = _lexical_key(V219_CACHE_DENIED_ROOT)
    return key == result_key or key == cache_key or key.startswith(cache_key + "\\")


def _guard_v219_namespace(path: str | os.PathLike[str]) -> None:
    if _is_v219_denied_lexically(path):
        raise SparseCachePreflightError(
            "v2.19 receipt/cache namespace access denied",
            "V219_NAMESPACE_DENIED",
        )


class _ProcessAuditGuard:
    """Supplement exact allowlists with audited protected-path denial."""

    _EVENTS = frozenset({
        "open", "import", "os.listdir", "os.scandir", "os.mkdir", "os.remove",
        "os.rename", "os.rmdir", "os.startfile", "pathlib.Path.glob",
        "pathlib.Path.rglob",
    })
    _MARKERS = PROTECTED_PATH_MARKERS

    def __init__(self) -> None:
        self.active = True
        self.network_attempt_count = 0

    @classmethod
    def _protected(cls, path: str | os.PathLike[str]) -> bool:
        if _is_v219_denied_lexically(path):
            return True
        components = tuple(part.casefold() for part in ntpath.normpath(os.fspath(path)).split("\\"))
        return any(any(marker in component for marker in cls._MARKERS) for component in components)

    def hook(self, event: str, arguments: tuple[Any, ...]) -> None:
        if not self.active:
            return
        if event.startswith("socket."):
            self.network_attempt_count += 1
            raise SparseCachePreflightError(
                "parent network access denied", "NETWORK_ATTEMPT"
            )
        if event not in self._EVENTS:
            return
        for argument in arguments:
            if not isinstance(argument, (str, os.PathLike)):
                continue
            try:
                protected = self._protected(argument)
            except (TypeError, ValueError):
                continue
            if protected:
                code = (
                    "V219_NAMESPACE_DENIED"
                    if _is_v219_denied_lexically(argument)
                    else "AUDITED_PATH_DENIED"
                )
                raise SparseCachePreflightError("audited protected-path access denied", code)

    def close(self) -> None:
        self.active = False


def _install_process_audit_guard() -> _ProcessAuditGuard:
    guard = _ProcessAuditGuard()
    sys.addaudithook(guard.hook)
    return guard


def _is_reparse(path: Path) -> bool:
    _guard_v219_namespace(path)
    observed = path.lstat()
    flag = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(getattr(observed, "st_file_attributes", 0) & flag)


def _require_plain(path: Path, *, directory: bool = False) -> Path:
    _guard_v219_namespace(path)
    absolute = path.absolute()
    anchor = Path(absolute.anchor)
    current = anchor
    for part in absolute.parts[1:]:
        current /= part
        _guard_v219_namespace(current)
        if current.exists() or current.is_symlink():
            if _is_reparse(current):
                raise SparseCachePreflightError(
                    "path traverses a link or reparse point", "UNSAFE_PATH"
                )
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SparseCachePreflightError("required path unavailable", "PATH_UNAVAILABLE") from error
    _guard_v219_namespace(resolved)
    if (directory and not resolved.is_dir()) or (not directory and not resolved.is_file()):
        raise SparseCachePreflightError("required path has unsafe type", "UNSAFE_PATH")
    return resolved


def _static_allowed_paths() -> frozenset[str]:
    paths = {
        PREREG_PATH,
        RUNNER_PATH,
        WORKER_PATH,
        RESULT_PATH,
        CATALOG_PATH,
        CONTEXT_PATH,
        *C200_REFERENCE_PATHS,
    }
    paths.update(ROOT / path for path in IMPLEMENTATION_PATHS)
    paths.update(ROOT / path for path in FROZEN_PARENT_BLOBS)
    return frozenset(_lexical_key(path) for path in paths)


def _is_registered_temp_path(path: Path) -> bool:
    key = _lexical_key(path)
    return any(key == root or key.startswith(root + "\\") for root in _REGISTERED_TEMP_ROOTS)


def _guard_experiment_data(path: Path, *, allow_temp: bool = False) -> None:
    _guard_v219_namespace(path)
    key = _lexical_key(path)
    if key in _static_allowed_paths() or (allow_temp and _is_registered_temp_path(path)):
        return
    raise SparseCachePreflightError(
        "experiment-data path outside exact allowlist", "DATA_PATH_DENIED"
    )


def _validate_path_policy() -> dict[str, bool]:
    for path in (
        CATALOG_PATH,
        CONTEXT_PATH,
        *C200_REFERENCE_PATHS,
        PREREG_PATH,
        RUNNER_PATH,
        WORKER_PATH,
        RESULT_PATH,
    ):
        _guard_experiment_data(path)
    # This is a purely lexical assertion.  It deliberately performs no stat,
    # resolve, exists, open, or hash operation on either forbidden namespace.
    for denied in (V219_RESULT_DENIED_PATH, V219_CACHE_DENIED_ROOT):
        if not _is_v219_denied_lexically(denied):
            raise SparseCachePreflightError("v2.19 lexical deny drifted", "DATA_POLICY_FAILURE")
        try:
            _guard_v219_namespace(denied)
        except SparseCachePreflightError as error:
            if error.code != "V219_NAMESPACE_DENIED":
                raise
        else:
            raise SparseCachePreflightError("v2.19 path failed open", "DATA_POLICY_FAILURE")
    return {
        "exact_source_allowlist": True,
        "recursive_external_scans": False,
        "v2_19_lexical_deny_before_resolution": True,
        "python_audit_supplement_installed": True,
        "parent_socket_audit_installed": True,
    }


def _file_identity(path: Path, label: str) -> FileIdentity:
    _guard_experiment_data(path)
    resolved = _require_plain(path)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    with resolved.open("rb") as handle:
        before = _snapshot(os.fstat(handle.fileno()))
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            byte_count += len(chunk)
            row_count += chunk.count(b"\n")
        after = _snapshot(os.fstat(handle.fileno()))
    if before != after or _snapshot(resolved.stat()) != after or byte_count != before[0]:
        raise SparseCachePreflightError(f"{label} changed while hashed", "SOURCE_MUTATION")
    return FileIdentity(byte_count, row_count, digest.hexdigest(), after)


def _git(*args: str, binary: bool = False) -> Any:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise SparseCachePreflightError("Git identity command failed", "GIT_COMMAND_FAILED")
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _changed_paths(commitish: str) -> set[str]:
    return set(filter(None, _git(
        "diff-tree", "--no-commit-id", "--name-only", "-r", commitish
    ).splitlines()))


def _diff_paths(left: str, right: str) -> set[str]:
    return set(filter(None, _git("diff", "--name-only", left, right).splitlines()))


def _worktree_blob(path: Path) -> str:
    _guard_experiment_data(path)
    raw = _require_plain(path).read_bytes()
    clean = raw.replace(b"\r\n", b"\n")
    return hashlib.sha1(
        b"blob " + str(len(clean)).encode("ascii") + b"\0" + clean
    ).hexdigest()


def _load_preregistration() -> dict[str, Any]:
    if _git("rev-parse", f"{PREREG_COMMIT}:{PREREG_RELATIVE}") != PREREG_BLOB:
        raise SparseCachePreflightError("preregistration blob drifted", "PREREG_DRIFT")
    raw = _git("cat-file", "blob", PREREG_BLOB, binary=True)
    _guard_experiment_data(PREREG_PATH)
    worktree_raw = _require_plain(PREREG_PATH).read_bytes()
    value = _parse_json(raw, "preregistration")
    worktree = _parse_json(worktree_raw, "worktree preregistration")
    if not (
        len(raw) == PREREG_BYTES
        and hashlib.sha256(raw).hexdigest() == PREREG_RAW_SHA256
        and _worktree_blob(PREREG_PATH) == PREREG_BLOB
        and _canonical_sha256(value) == PREREG_CANONICAL_SHA256
        and _canonical_sha256(worktree) == PREREG_CANONICAL_SHA256
        and value.get("schema_version")
        == "small-ranker-v2.20-sparse-route-cache-preregistration.v1"
        and value.get("status") == "PREREGISTERED_TARGET_FREE_ONLY"
        and value.get("parent_evidence", {}).get("direct_parent") == V219_BLOCKER_COMMIT
    ):
        raise SparseCachePreflightError("preregistration identity drifted", "PREREG_DRIFT")
    return value


def _validate_git_checkpoint(implementation_commit: str) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(implementation_commit) is None:
        raise SparseCachePreflightError("implementation commit invalid", "GIT_CHECKPOINT")
    head = _git("rev-parse", "HEAD")
    parent_blobs = {
        path: _git("rev-parse", f"{PREREG_COMMIT}:{path}")
        for path in FROZEN_PARENT_BLOBS
    }
    implementation_blobs = {
        path: _git("rev-parse", f"HEAD:{path}")
        for path in sorted(IMPLEMENTATION_PATHS)
    }
    unchanged_head_blobs = {
        path: _git("rev-parse", f"HEAD:{path}")
        for path in FROZEN_PARENT_BLOBS
        if path not in IMPLEMENTATION_PATHS
    }
    expected_unchanged = {
        path: blob
        for path, blob in FROZEN_PARENT_BLOBS.items()
        if path not in IMPLEMENTATION_PATHS
    }
    worktree_paths = sorted(
        set(FROZEN_PARENT_BLOBS) | IMPLEMENTATION_PATHS | {PREREG_RELATIVE}
    )
    worktree_equal = all(
        _worktree_blob(ROOT / path) == _git("rev-parse", f"HEAD:{path}")
        for path in worktree_paths
    )
    # The frozen prereg's global-clean requirement conflicts with its stronger
    # zero-probe rule for a leaf under experiments/fast_track.  Do not perform
    # a worktree-wide Git traversal here.  These scoped checks are retained for
    # diagnostics only; ``run`` blocks before this function can be reached.
    scoped_index_diff = _git(
        "diff", "--cached", "--name-only", "HEAD", "--", *worktree_paths
    )
    scoped_worktree_diff = _git(
        "diff", "--name-only", "--", *worktree_paths
    )
    if not (
        head == implementation_commit
        and _git("rev-parse", "HEAD^") == PREREG_COMMIT
        and _git("rev-parse", f"{PREREG_COMMIT}^") == V219_BLOCKER_COMMIT
        and _git("branch", "--show-current") == BRANCH
        and _git("remote", "get-url", REMOTE) == REMOTE_URL
        and _git("rev-parse", REMOTE_REF) == head
        and _changed_paths(PREREG_COMMIT) == PREREG_PATHS
        and _changed_paths(head) == IMPLEMENTATION_PATHS
        and _diff_paths(PREREG_COMMIT, head) == IMPLEMENTATION_PATHS
        and parent_blobs == FROZEN_PARENT_BLOBS
        and unchanged_head_blobs == expected_unchanged
        and worktree_equal
        and not scoped_index_diff
        and not scoped_worktree_diff
    ):
        raise SparseCachePreflightError("Git checkpoint gate failed", "GIT_CHECKPOINT")
    return {
        "branch": BRANCH,
        "commit": head,
        "direct_parent": PREREG_COMMIT,
        "exact_changed_paths": True,
        "implementation_blobs": implementation_blobs,
        "parent_blobs_sha256": _canonical_sha256(parent_blobs),
        "remote_equal": True,
        "scoped_execution_inputs_clean": True,
        "ignored_artifact_tree_scanned": False,
    }


def _validate_environment() -> dict[str, Any]:
    try:
        expected = EXPECTED_EXECUTABLE.resolve(strict=True)
        actual = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        raise SparseCachePreflightError(
            "formal executable unavailable", "ENVIRONMENT_DRIFT"
        ) from error
    if not (
        _lexical_key(actual) == _lexical_key(expected)
        and sys.version.split()[0] == EXPECTED_PYTHON
        and sqlite3.sqlite_version == EXPECTED_SQLITE
        and os.getenv("PYTHONHASHSEED") == "0"
        and _lexical_key(Path.cwd()) == _lexical_key(ROOT)
        and not any(name in sys.modules for name in ("cupy", "tensorflow", "torch"))
    ):
        raise SparseCachePreflightError("runtime environment drifted", "ENVIRONMENT_DRIFT")
    return {
        "cpu_only": True,
        "executable": actual.as_posix(),
        "gpu_peak_bytes": 0,
        "network_attempt_count": 0,
        "python": EXPECTED_PYTHON,
        "pythonhashseed": "0",
        "sqlite": EXPECTED_SQLITE,
    }


def _catalog_identity_and_ids() -> tuple[FileIdentity, frozenset[str]]:
    _guard_experiment_data(CATALOG_PATH)
    resolved = _require_plain(CATALOG_PATH)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    identifiers: set[str] = set()
    with resolved.open("rb") as handle:
        before = _snapshot(os.fstat(handle.fileno()))
        for raw in handle:
            digest.update(raw)
            byte_count += len(raw)
            row_count += 1
            value = _parse_json(raw, "catalog row")
            identifier = value.get("parent_asin")
            if (
                not isinstance(identifier, str)
                or CATALOG_ID_RE.fullmatch(identifier) is None
                or identifier != identifier.upper()
                or identifier in identifiers
            ):
                raise SparseCachePreflightError("catalog identifier drifted", "SOURCE_SCHEMA")
            identifiers.add(identifier)
        after = _snapshot(os.fstat(handle.fileno()))
    identity = FileIdentity(byte_count, row_count, digest.hexdigest(), after)
    if (
        before != after
        or _snapshot(resolved.stat()) != after
        or identity.report()
        != {"bytes": CATALOG_BYTES, "rows": CATALOG_ROWS, "sha256": CATALOG_SHA256}
    ):
        raise SparseCachePreflightError("catalog identity drifted", "SOURCE_DRIFT")
    return identity, frozenset(identifiers)


def _source_checkpoint() -> tuple[frozenset[str], dict[str, Any]]:
    catalog, identifiers = _catalog_identity_and_ids()
    context = _file_identity(CONTEXT_PATH, "visible context")
    if context.report() != {
        "bytes": CONTEXT_BYTES,
        "rows": CONTEXT_ROWS,
        "sha256": CONTEXT_SHA256,
    }:
        raise SparseCachePreflightError("visible context identity drifted", "SOURCE_DRIFT")
    references = tuple(
        _file_identity(path, f"C200 reference {index}")
        for index, path in enumerate(C200_REFERENCE_PATHS, start=1)
    )
    expected_reference = {
        "bytes": C200_TRACE_BYTES,
        "rows": C200_TRACE_ROWS,
        "sha256": C200_TRACE_SHA256,
    }
    if any(identity.report() != expected_reference for identity in references):
        raise SparseCachePreflightError("C200 reference identity drifted", "SOURCE_DRIFT")
    if references[0].report() != references[1].report():
        raise SparseCachePreflightError("C200 references differ", "SOURCE_DRIFT")
    return identifiers, {
        "catalog": catalog.report(),
        "sealed_c200": [identity.report() for identity in references],
        "visible_context": context.report(),
    }


def _offline_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.casefold() not in {"pythonpath", "pythonhome"}
    }
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    return environment


def _run_subprocess(
    command: Sequence[str], *, timeout: float, cwd: Path = ROOT
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=_offline_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise SparseCachePreflightError("isolated subprocess timed out", "WORKER_TIMEOUT") from error


def _entrypoint_self_check(required_module: str) -> dict[str, Any]:
    if not isinstance(required_module, str) or not required_module:
        raise SparseCachePreflightError("required module invalid", "ENTRYPOINT_FAILURE")
    importlib.import_module(required_module)
    return {
        "project_root_bootstrapped": str(PROJECT_ROOT) in sys.path,
        "required_module": required_module,
        "status": "ENTRYPOINT_SELF_CHECK_PASS",
    }


def _self_check_command(subject: str, mode: str, required_module: str) -> list[str]:
    if subject == "runner":
        direct_path = RUNNER_PATH
        module_name = "scripts.probe_sparse_multiview_cache_preflight"
    elif subject == "worker":
        direct_path = WORKER_PATH
        module_name = "scripts.sparse_multiview_candidate_worker"
    else:
        raise SparseCachePreflightError("entrypoint subject invalid", "ENTRYPOINT_FAILURE")
    if mode == "direct":
        prefix = [str(EXPECTED_EXECUTABLE), "-B", str(direct_path)]
    elif mode == "module":
        prefix = [str(EXPECTED_EXECUTABLE), "-B", "-m", module_name]
    else:
        raise SparseCachePreflightError("entrypoint mode invalid", "ENTRYPOINT_FAILURE")
    return [*prefix, "--entrypoint-self-check", "--require-module", required_module]


def _validate_self_check(completed: subprocess.CompletedProcess[bytes], label: str) -> None:
    if completed.returncode != 0 or completed.stderr:
        raise SparseCachePreflightError(f"{label} self-check failed", "ENTRYPOINT_FAILURE")
    value = _parse_json(completed.stdout, "entrypoint receipt")
    if completed.stdout != _canonical_bytes(value) + b"\n" or value != {
        "project_root_bootstrapped": True,
        "required_module": "evaluator.local_evaluator",
        "status": "ENTRYPOINT_SELF_CHECK_PASS",
    }:
        raise SparseCachePreflightError("entrypoint receipt drifted", "ENTRYPOINT_FAILURE")


def _verify_entrypoints() -> dict[str, bool]:
    required = "evaluator.local_evaluator"
    for subject in ("runner", "worker"):
        for mode in ("direct", "module"):
            _validate_self_check(
                _run_subprocess(
                    _self_check_command(subject, mode, required), timeout=60.0
                ),
                f"{subject}-{mode}",
            )
        _validate_self_check(
            _run_subprocess(
                _self_check_command(subject, "direct", required),
                timeout=60.0,
                cwd=ROOT.parent,
            ),
            f"{subject}-direct-outside-cwd",
        )
    missing = "v220_intentionally_absent_required_module"
    for subject in ("runner", "worker"):
        for mode in ("direct", "module"):
            completed = _run_subprocess(
                _self_check_command(subject, mode, missing),
                timeout=60.0,
                cwd=ROOT.parent if mode == "direct" else ROOT,
            )
            if completed.returncode == 0 or completed.stderr:
                raise SparseCachePreflightError(
                    "missing-module check failed open", "ENTRYPOINT_FAILURE"
                )
            if subject == "runner":
                missing_receipt = _parse_json(
                    completed.stdout, "missing-module runner receipt"
                )
                if (
                    completed.stdout != _canonical_bytes(missing_receipt) + b"\n"
                    or missing_receipt
                    != {
                        "error_code": "UNEXPECTED_EXCEPTION",
                        "exception_class": "ModuleNotFoundError",
                        "raw_traceback_or_stderr_emitted": False,
                        "status": "ERROR",
                    }
                ):
                    raise SparseCachePreflightError(
                        "missing-module runner output drifted", "ENTRYPOINT_FAILURE"
                    )
            elif completed.stdout:
                raise SparseCachePreflightError(
                    "missing-module worker emitted output", "ENTRYPOINT_FAILURE"
                )
    _assert_fresh_result()
    return {
        "direct_outside_repository_cwd": True,
        "missing_module_failed_nonzero": True,
        "runner_direct_and_module": True,
        "worker_direct_and_module": True,
    }


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        result = {str(key).casefold() for key in value}
        for item in value.values():
            result.update(_walk_keys(item))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for item in value:
            result.update(_walk_keys(item))
        return result
    return set()


def _walk_values(value: object) -> Iterable[object]:
    yield value
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_values(item)


def _privacy_scan(value: object, catalog_ids: Iterable[str] = ()) -> None:
    if _walk_keys(value) & FORBIDDEN_RESULT_KEYS:
        raise SparseCachePreflightError("forbidden result key", "PRIVACY_GATE")
    catalog = {identifier.casefold() for identifier in catalog_ids}
    for item in _walk_values(value):
        if isinstance(item, (list, tuple)) and len(item) >= FROZEN_20_TRACE_ROWS:
            raise SparseCachePreflightError("result contains a long array", "PRIVACY_GATE")
        if isinstance(item, str) and (
            ASIN_SHAPE_RE.search(item) is not None or item.casefold() in catalog
        ):
            raise SparseCachePreflightError("result contains catalog identifier", "PRIVACY_GATE")


def _finite(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise SparseCachePreflightError(f"{label} invalid", "WORKER_CONTRACT")
    return float(value)


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SparseCachePreflightError(f"{label} invalid", "WORKER_CONTRACT")
    return value


def _latency_p95(value: object, label: str, expected_count: int) -> float:
    if not isinstance(value, Mapping):
        raise SparseCachePreflightError(f"{label} latency missing", "WORKER_CONTRACT")
    if _nonnegative_int(value.get("count"), f"{label} count") != expected_count:
        raise SparseCachePreflightError(f"{label} latency count drifted", "WORKER_CONTRACT")
    maximum = _finite(value.get("maximum_milliseconds"), f"{label} maximum")
    p50 = _finite(value.get("p50_milliseconds"), f"{label} p50")
    p95 = _finite(value.get("p95_milliseconds"), f"{label} p95")
    if not p50 <= p95 <= maximum:
        raise SparseCachePreflightError(f"{label} latency order drifted", "WORKER_CONTRACT")
    return p95


def _validate_cache_snapshot(value: object, *, closed: bool) -> dict[str, Any]:
    expected_keys = {"enabled", "closed", "clears", *CACHE_CAPACITIES}
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise SparseCachePreflightError("cache snapshot schema drifted", "CACHE_CONTRACT")
    if value.get("enabled") is not True or value.get("closed") is not closed:
        raise SparseCachePreflightError("cache state drifted", "CACHE_CONTRACT")
    clears = _nonnegative_int(value.get("clears"), "cache clears")
    result: dict[str, Any] = {"enabled": True, "closed": closed, "clears": clears}
    layer_keys = {
        "lookups", "hits", "misses", "evictions", "size", "capacity",
        "avoided_operations",
    }
    for name, capacity in CACHE_CAPACITIES.items():
        layer = value.get(name)
        if not isinstance(layer, Mapping) or set(layer) != layer_keys:
            raise SparseCachePreflightError("cache layer schema drifted", "CACHE_CONTRACT")
        normalized = {
            key: _nonnegative_int(layer.get(key), f"{name} {key}")
            for key in layer_keys
        }
        if (
            normalized["capacity"] != capacity
            or normalized["lookups"] != normalized["hits"] + normalized["misses"]
            or normalized["avoided_operations"] != normalized["hits"]
            or normalized["size"] > capacity
            or (closed and normalized["size"] != 0)
        ):
            raise SparseCachePreflightError("cache accounting drifted", "CACHE_CONTRACT")
        result[name] = normalized
    return result


def _validate_cache_diagnostics(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"before_close", "after_close"}:
        raise SparseCachePreflightError("cache diagnostics schema drifted", "CACHE_CONTRACT")
    before = _validate_cache_snapshot(value["before_close"], closed=False)
    after = _validate_cache_snapshot(value["after_close"], closed=True)
    if after["clears"] != before["clears"] + 1:
        raise SparseCachePreflightError("cache close clear count drifted", "CACHE_CONTRACT")
    for name in CACHE_CAPACITIES:
        for key in (
            "lookups", "hits", "misses", "evictions", "capacity",
            "avoided_operations",
        ):
            if after[name][key] != before[name][key]:
                raise SparseCachePreflightError(
                    "cache counters changed during close", "CACHE_CONTRACT"
                )
    return {"before_close": before, "after_close": after}


def _validate_worker_receipt(
    payload: bytes,
    *,
    nonce: str,
    session_limit: int,
    cached: bool,
    catalog_ids: Iterable[str],
) -> dict[str, Any]:
    value = _parse_json(payload, "worker receipt")
    if payload != _canonical_bytes(value) + b"\n":
        raise SparseCachePreflightError("worker receipt not canonical LF JSON", "WORKER_CONTRACT")
    expected_top = {
        "schema_version", "kind", "status", "phase", "error_code", "nonce",
        "trace_bytes", "trace_sha256", "record_count", "last_completed_session",
        "summary",
    }
    expected_records = session_limit * TURN_COUNT
    summary = value.get("summary")
    if not (
        set(value) == expected_top
        and value.get("schema_version") == WORKER_SCHEMA_VERSION
        and value.get("kind") == "receipt"
        and value.get("status") == "SUCCESS"
        and value.get("phase") == "COMPLETE"
        and value.get("error_code") == "NONE"
        and value.get("nonce") == nonce
        and value.get("record_count") == expected_records
        and value.get("last_completed_session") == session_limit
        and isinstance(value.get("trace_bytes"), int)
        and value["trace_bytes"] > 0
        and isinstance(value.get("trace_sha256"), str)
        and DIGEST_RE.fullmatch(value["trace_sha256"]) is not None
        and isinstance(summary, Mapping)
        and summary.get("session_limit") == session_limit
        and summary.get("processed_sessions") == session_limit
        and summary.get("processed_turns") == expected_records
    ):
        raise SparseCachePreflightError("worker receipt envelope drifted", "WORKER_CONTRACT")

    semantic = summary.get("semantic_trace")
    if not (
        isinstance(semantic, Mapping)
        and set(semantic) == {"rows", "sha256"}
        and semantic.get("rows") == expected_records
        and isinstance(semantic.get("sha256"), str)
        and DIGEST_RE.fullmatch(str(semantic["sha256"])) is not None
    ):
        raise SparseCachePreflightError("semantic digest contract drifted", "WORKER_CONTRACT")

    if cached:
        normalized_cache = _validate_cache_diagnostics(summary.get("cache"))
    else:
        if "cache" in summary:
            raise SparseCachePreflightError(
                "cache-disabled control emitted cache diagnostics", "WORKER_CONTRACT"
            )
        normalized_cache = None

    activation = summary.get("activation")
    mask = summary.get("mask")
    latency = summary.get("latency")
    lifecycle = summary.get("lifecycle")
    resources = summary.get("resources")
    inputs = summary.get("input_identities")
    if not (
        isinstance(activation, Mapping)
        and isinstance(mask, Mapping)
        and isinstance(latency, Mapping)
        and isinstance(lifecycle, Mapping)
        and lifecycle
        and all(flag is True for flag in lifecycle.values())
        and isinstance(resources, Mapping)
        and isinstance(inputs, Mapping)
    ):
        raise SparseCachePreflightError("worker summary incomplete", "WORKER_CONTRACT")
    activated = _nonnegative_int(activation.get("activated_records"), "activated records")
    inactive = _nonnegative_int(activation.get("inactive_records"), "inactive records")
    if activated + inactive != expected_records:
        raise SparseCachePreflightError("worker activation count drifted", "WORKER_CONTRACT")
    if _nonnegative_int(mask.get("tail_duplicate_count"), "tail duplicates") != 0:
        raise SparseCachePreflightError("tail duplicate gate failed", "WORKER_CONTRACT")
    if _nonnegative_int(
        mask.get("tail_explicit_conflict_count"), "tail conflicts"
    ) != 0:
        raise SparseCachePreflightError("tail conflict gate failed", "WORKER_CONTRACT")
    route_p95 = _latency_p95(
        latency.get("extra_route_and_mask"), "extra route", expected_records
    )
    turn_p95 = _latency_p95(latency.get("per_turn"), "per turn", expected_records)
    if route_p95 > EXTRA_ROUTE_P95_MS_MAXIMUM or turn_p95 > PER_TURN_P95_MS_MAXIMUM:
        raise SparseCachePreflightError("worker latency gate failed", "RESOURCE_GATE")

    peak = resources.get("peak_working_set_bytes")
    if (
        not isinstance(peak, int)
        or isinstance(peak, bool)
        or not 0 < peak <= WORKER_RSS_MAXIMUM
        or _nonnegative_int(resources.get("network_attempt_count"), "network attempts") != 0
        or _nonnegative_int(resources.get("gpu_peak_bytes"), "GPU bytes") != 0
        or _finite(resources.get("wall_seconds"), "worker wall")
        > FORMAL_WALL_SECONDS_MAXIMUM
        or _finite(
            resources.get("candidate_cell_ratio_over_c200"), "candidate cell ratio"
        )
        > CELL_RATIO_MAXIMUM
        or _finite(resources.get("trace_byte_ratio_over_c200"), "trace byte ratio")
        > TRACE_RATIO_MAXIMUM
    ):
        raise SparseCachePreflightError("worker resource gate failed", "RESOURCE_GATE")

    expected_inputs = {
        "catalog": {"bytes": CATALOG_BYTES, "rows": CATALOG_ROWS, "sha256": CATALOG_SHA256},
        "sealed_c200_reference": {
            "bytes": C200_TRACE_BYTES,
            "rows": C200_TRACE_ROWS,
            "sha256": C200_TRACE_SHA256,
        },
        "visible_context": {
            "bytes": CONTEXT_BYTES,
            "rows": CONTEXT_ROWS,
            "sha256": CONTEXT_SHA256,
        },
    }
    if dict(inputs) != expected_inputs:
        raise SparseCachePreflightError("worker input identity drifted", "WORKER_CONTRACT")

    _privacy_scan(value, catalog_ids)
    result = dict(value)
    if normalized_cache is not None:
        result["summary"] = {**dict(summary), "cache": normalized_cache}
    return result


def _worker_command(
    *,
    mode: str,
    nonce: str,
    reference: Path,
    trace: Path,
    session_limit: int,
    cached: bool,
    implementation_blobs: Mapping[str, str],
) -> list[str]:
    worker_blob = implementation_blobs.get(
        "scripts/sparse_multiview_candidate_worker.py"
    )
    sparse_blob = implementation_blobs.get("starter/sparse_multiview.py")
    if (
        not isinstance(worker_blob, str)
        or COMMIT_RE.fullmatch(worker_blob) is None
        or not isinstance(sparse_blob, str)
        or COMMIT_RE.fullmatch(sparse_blob) is None
    ):
        raise SparseCachePreflightError(
            "worker source blob handshake invalid", "GIT_CHECKPOINT"
        )
    if mode == "direct":
        prefix = [str(EXPECTED_EXECUTABLE), "-B", str(WORKER_PATH)]
    elif mode == "module":
        prefix = [
            str(EXPECTED_EXECUTABLE), "-B", "-m",
            "scripts.sparse_multiview_candidate_worker",
        ]
    else:
        raise SparseCachePreflightError("worker mode invalid", "WORKER_CONTRACT")
    command = [
        *prefix,
        "--catalog", str(CATALOG_PATH),
        "--context", str(CONTEXT_PATH),
        "--c200-reference", str(reference),
        "--trace-output", str(trace),
        "--session-limit", str(session_limit),
        "--nonce", nonce,
        "--semantic-audit",
        "--expected-worker-blob",
        worker_blob,
        "--expected-sparse-blob",
        sparse_blob,
    ]
    if cached:
        command.append("--semantic-cache")
    return command


def _run_worker(
    *,
    mode: str,
    nonce: str,
    reference: Path,
    trace: Path,
    session_limit: int,
    cached: bool,
    catalog_ids: Iterable[str],
    implementation_blobs: Mapping[str, str],
) -> dict[str, Any]:
    if NONCE_RE.fullmatch(nonce) is None:
        raise SparseCachePreflightError("worker nonce invalid", "WORKER_CONTRACT")
    _guard_experiment_data(reference)
    _guard_experiment_data(trace, allow_temp=True)
    command = _worker_command(
        mode=mode,
        nonce=nonce,
        reference=reference,
        trace=trace,
        session_limit=session_limit,
        cached=cached,
        implementation_blobs=implementation_blobs,
    )
    started = time.perf_counter()
    completed = _run_subprocess(command, timeout=FORMAL_WALL_SECONDS_MAXIMUM)
    if completed.returncode != 0 or completed.stderr:
        raise SparseCachePreflightError("isolated worker failed", "WORKER_FAILURE")
    receipt = _validate_worker_receipt(
        completed.stdout,
        nonce=nonce,
        session_limit=session_limit,
        cached=cached,
        catalog_ids=catalog_ids,
    )
    parent_wall_seconds = time.perf_counter() - started
    return {
        "receipt": receipt,
        "mode": mode,
        "cached": cached,
        "parent_wall_seconds": parent_wall_seconds,
        "stderr": {
            "bytes": len(completed.stderr),
            "sha256": hashlib.sha256(completed.stderr).hexdigest(),
        },
    }


def _parse_trace_row(raw: bytes, label: str, key: str) -> dict[str, Any]:
    value = _parse_json(raw, label)
    if raw != _canonical_bytes(value) + b"\n" or set(value) != {key, "ordinal", "turn"}:
        raise SparseCachePreflightError(f"{label} schema/canonical form drifted", "TRACE_CONTRACT")
    return value


def _validate_candidate_values(
    values: object,
    *,
    minimum: int,
    maximum: int,
    catalog_ids: frozenset[str],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise SparseCachePreflightError(f"{label} is not a list", "TRACE_CONTRACT")
    result = tuple(values)
    if (
        not minimum <= len(result) <= maximum
        or len(result) != len(set(result))
        or any(
            not isinstance(identifier, str) or identifier not in catalog_ids
            for identifier in result
        )
    ):
        raise SparseCachePreflightError(f"{label} shape drifted", "TRACE_CONTRACT")
    return result


def _load_and_validate_trace(
    trace_path: Path,
    reference_path: Path,
    catalog_ids: frozenset[str],
    *,
    session_limit: int,
) -> TraceValidation:
    _guard_experiment_data(trace_path, allow_temp=True)
    _guard_experiment_data(reference_path)
    trace = _require_plain(trace_path)
    reference = _require_plain(reference_path)
    digest = hashlib.sha256()
    trace_bytes = 0
    reference_prefix_bytes = 0
    candidate_cells = 0
    c200_cells = 0
    activation_turns = 0
    activation_sessions: set[int] = set()
    candidate_lengths: list[int] = []
    expected_records = session_limit * TURN_COUNT
    with trace.open("rb") as trace_handle, reference.open("rb") as reference_handle:
        trace_before = _snapshot(os.fstat(trace_handle.fileno()))
        reference_before = _snapshot(os.fstat(reference_handle.fileno()))
        for index in range(expected_records):
            ordinal = index // TURN_COUNT + 1
            turn = index % TURN_COUNT + 1
            trace_raw = trace_handle.readline()
            reference_raw = reference_handle.readline()
            if not trace_raw or not reference_raw:
                raise SparseCachePreflightError("trace/reference ended early", "TRACE_CONTRACT")
            trace_row = _parse_trace_row(trace_raw, "candidate trace row", "candidates")
            reference_row = _parse_trace_row(reference_raw, "C200 reference row", "c200")
            if (
                trace_row.get("ordinal") != ordinal
                or trace_row.get("turn") != turn
                or reference_row.get("ordinal") != ordinal
                or reference_row.get("turn") != turn
            ):
                raise SparseCachePreflightError("trace coordinate drifted", "TRACE_CONTRACT")
            c200 = _validate_candidate_values(
                reference_row["c200"],
                minimum=MIN_C200_CANDIDATES,
                maximum=MAX_C200_CANDIDATES,
                catalog_ids=catalog_ids,
                label="C200",
            )
            candidates = _validate_candidate_values(
                trace_row["candidates"],
                minimum=len(c200),
                maximum=MAX_CANDIDATES,
                catalog_ids=catalog_ids,
                label="expanded candidates",
            )
            if candidates[: len(c200)] != c200 or len(candidates) > len(c200) + 120:
                raise SparseCachePreflightError("stable prefix drifted", "TRACE_CONTRACT")
            if len(candidates) > len(c200):
                activation_turns += 1
                activation_sessions.add(ordinal)
            digest.update(trace_raw)
            trace_bytes += len(trace_raw)
            reference_prefix_bytes += len(reference_raw)
            candidate_cells += len(candidates)
            c200_cells += len(c200)
            candidate_lengths.append(len(candidates))
        if trace_handle.read(1):
            raise SparseCachePreflightError("candidate trace has excess rows", "TRACE_CONTRACT")
        trace_after = _snapshot(os.fstat(trace_handle.fileno()))
        reference_after = _snapshot(os.fstat(reference_handle.fileno()))
    if (
        trace_before != trace_after
        or _snapshot(trace.stat()) != trace_after
        or reference_before != reference_after
        or _snapshot(reference.stat()) != reference_after
        or trace_bytes != trace_after[0]
        or candidate_cells / c200_cells > CELL_RATIO_MAXIMUM
        or trace_bytes / reference_prefix_bytes > TRACE_RATIO_MAXIMUM
    ):
        raise SparseCachePreflightError("trace identity/resource drifted", "TRACE_CONTRACT")
    return TraceValidation(
        trace_sha256=digest.hexdigest(),
        trace_bytes=trace_bytes,
        record_count=expected_records,
        activation_turns=activation_turns,
        activation_sessions=len(activation_sessions),
        candidate_cells=candidate_cells,
        c200_cells=c200_cells,
        reference_prefix_bytes=reference_prefix_bytes,
        min_candidates=min(candidate_lengths),
        max_candidates=max(candidate_lengths),
    )


def _files_equal(left: Path, right: Path) -> bool:
    _guard_experiment_data(left, allow_temp=True)
    _guard_experiment_data(right, allow_temp=True)
    with _require_plain(left).open("rb") as first, _require_plain(right).open("rb") as second:
        while True:
            a = first.read(1 << 20)
            b = second.read(1 << 20)
            if a != b:
                return False
            if not a:
                return True


def _bind_receipt(result: Mapping[str, Any], trace: TraceValidation) -> None:
    receipt = result.get("receipt")
    if not (
        isinstance(receipt, Mapping)
        and receipt.get("trace_sha256") == trace.trace_sha256
        and receipt.get("trace_bytes") == trace.trace_bytes
        and receipt.get("record_count") == trace.record_count
    ):
        raise SparseCachePreflightError("worker receipt/trace binding failed", "TRACE_CONTRACT")


def _validate_stage_cache_gates(results: Mapping[str, Mapping[str, Any]]) -> None:
    normalized_by_worker: dict[str, dict[str, Any]] = {}
    for name in ("cached_direct", "cached_module"):
        worker = results.get(name)
        if not isinstance(worker, Mapping) or worker.get("cached") is not True:
            raise SparseCachePreflightError("cached worker missing", "CACHE_CONTRACT")
        receipt = worker.get("receipt")
        try:
            diagnostics = receipt["summary"]["cache"]  # type: ignore[index]
        except (KeyError, TypeError):
            raise SparseCachePreflightError("cache diagnostics missing", "CACHE_CONTRACT")
        normalized = _validate_cache_diagnostics(diagnostics)
        normalized_by_worker[name] = normalized
        before = normalized["before_close"]
        for layer in CACHE_CAPACITIES:
            if before[layer]["hits"] <= 0:
                raise SparseCachePreflightError("required cache hit absent", "CACHE_CONTRACT")
    if normalized_by_worker["cached_direct"] != normalized_by_worker["cached_module"]:
        raise SparseCachePreflightError(
            "direct/module cache diagnostics differ", "CACHE_CONTRACT"
        )


def _cached_pair_wall(results: Mapping[str, Mapping[str, Any]]) -> float:
    walls: list[float] = []
    for name in ("cached_direct", "cached_module"):
        worker = results.get(name)
        if not isinstance(worker, Mapping) or worker.get("cached") is not True:
            raise SparseCachePreflightError("cached pair incomplete", "RESOURCE_GATE")
        walls.append(_finite(worker.get("parent_wall_seconds"), f"{name} parent wall"))
    pair = math.fsum(walls)
    if pair > PAIR_WALL_SECONDS_MAXIMUM:
        raise SparseCachePreflightError("cached pair wall gate failed", "RESOURCE_GATE")
    return pair


def _validate_frozen_stage20(
    trace: TraceValidation,
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    receipt_activations = {
        _nonnegative_int(
            results[name]["receipt"]["summary"]["activation"].get(
                "activated_records"
            ),
            f"{name} activated records",
        )
        for name in ("control", "cached_direct", "cached_module")
    }
    if not (
        trace.record_count == FROZEN_20_TRACE_ROWS
        and trace.trace_bytes == FROZEN_20_TRACE_BYTES
        and trace.trace_sha256 == FROZEN_20_TRACE_SHA256
        and receipt_activations == {FROZEN_20_ACTIVATED_TURNS}
    ):
        raise SparseCachePreflightError(
            "20-session frozen trace identity failed", "FROZEN_20_IDENTITY"
        )


def _trace_triplet_gate(
    results: Mapping[str, Mapping[str, Any]],
    traces: Mapping[str, Path],
    catalog_ids: frozenset[str],
    session_limit: int,
) -> TraceValidation:
    expected_names = {"control", "cached_direct", "cached_module"}
    if set(results) != expected_names or set(traces) != expected_names:
        raise SparseCachePreflightError("worker triplet incomplete", "SEMANTIC_PARITY")
    validations = {
        "control": _load_and_validate_trace(
            traces["control"], C200_REFERENCE_PATHS[0], catalog_ids,
            session_limit=session_limit,
        ),
        "cached_direct": _load_and_validate_trace(
            traces["cached_direct"], C200_REFERENCE_PATHS[0], catalog_ids,
            session_limit=session_limit,
        ),
        "cached_module": _load_and_validate_trace(
            traces["cached_module"], C200_REFERENCE_PATHS[1], catalog_ids,
            session_limit=session_limit,
        ),
    }
    for name, validation in validations.items():
        _bind_receipt(results[name], validation)
    control = validations["control"]
    semantics = {
        str(results[name]["receipt"]["summary"]["semantic_trace"]["sha256"])
        for name in expected_names
    }
    receipt_activations = {
        _nonnegative_int(
            results[name]["receipt"]["summary"]["activation"].get(
                "activated_records"
            ),
            f"{name} activated records",
        )
        for name in expected_names
    }
    if (
        any(validation != control for validation in validations.values())
        or len(semantics) != 1
        or len(receipt_activations) != 1
        or not _files_equal(traces["control"], traces["cached_direct"])
        or not _files_equal(traces["control"], traces["cached_module"])
    ):
        raise SparseCachePreflightError(
            "cached/control trace or semantic digest differs", "SEMANTIC_PARITY"
        )
    _validate_stage_cache_gates(results)
    if session_limit == 20:
        _validate_frozen_stage20(control, results)
    return control


def _worker_report(result: Mapping[str, Any]) -> dict[str, Any]:
    receipt = result["receipt"]
    summary = receipt["summary"]
    report: dict[str, Any] = {
        "cached": result["cached"],
        "latency": summary["latency"],
        "mode": result["mode"],
        "parent_wall_seconds": round(float(result["parent_wall_seconds"]), 6),
        "resources": summary["resources"],
        "semantic_trace": summary["semantic_trace"],
        "stderr": result["stderr"],
    }
    if result["cached"]:
        report["cache"] = summary["cache"]
    return report


def _run_stage(
    session_limit: int,
    catalog_ids: frozenset[str],
    implementation_blobs: Mapping[str, str],
) -> dict[str, Any]:
    if session_limit not in ALLOWED_STAGE_LIMITS:
        raise SparseCachePreflightError("stage limit invalid", "CONTRACT_DRIFT")
    parent = _require_plain(RESULT_PATH.parent, directory=True)
    with tempfile.TemporaryDirectory(
        prefix=f"v220_cache_{session_limit}_", dir=parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        root_key = _lexical_key(temporary)
        _REGISTERED_TEMP_ROOTS.add(root_key)
        try:
            traces = {
                "control": temporary / "control.jsonl",
                "cached_direct": temporary / "cached_direct.jsonl",
                "cached_module": temporary / "cached_module.jsonl",
            }
            specs = (
                ("control", "direct", False, C200_REFERENCE_PATHS[0]),
                ("cached_direct", "direct", True, C200_REFERENCE_PATHS[0]),
                ("cached_module", "module", True, C200_REFERENCE_PATHS[1]),
            )
            results: dict[str, dict[str, Any]] = {}
            for name, mode, cached, reference in specs:
                nonce = hashlib.sha256(
                    os.urandom(32)
                    + name.encode("ascii")
                    + str(session_limit).encode("ascii")
                ).hexdigest()[:32]
                results[name] = _run_worker(
                    mode=mode,
                    nonce=nonce,
                    reference=reference,
                    trace=traces[name],
                    session_limit=session_limit,
                    cached=cached,
                    catalog_ids=catalog_ids,
                    implementation_blobs=implementation_blobs,
                )
            validation = _trace_triplet_gate(
                results, traces, catalog_ids, session_limit
            )
            cached_pair = math.fsum(
                float(results[name]["parent_wall_seconds"])
                for name in ("cached_direct", "cached_module")
            )
            if session_limit == 100:
                cached_pair = _cached_pair_wall(results)
                if cached_pair * (SESSION_COUNT / 100) * 1.5 > FORMAL_WALL_SECONDS_MAXIMUM:
                    raise SparseCachePreflightError(
                        "formal extrapolation gate failed", "RESOURCE_GATE"
                    )
            semantic_sha256 = str(
                results["control"]["receipt"]["summary"]["semantic_trace"]["sha256"]
            )
            return {
                "activated_turns": int(
                    results["control"]["receipt"]["summary"]["activation"][
                        "activated_records"
                    ]
                ),
                "cached_pair_parent_wall_seconds": round(cached_pair, 6),
                "direct_module_control_trace_exact": True,
                "semantic_trace_sha256": semantic_sha256,
                "sessions": session_limit,
                "trace": validation.report(),
                "workers": {
                    name: _worker_report(results[name])
                    for name in ("control", "cached_direct", "cached_module")
                },
            }
        finally:
            _REGISTERED_TEMP_ROOTS.discard(root_key)


def _assert_fresh_result() -> None:
    _guard_experiment_data(RESULT_PATH)
    parent = _require_plain(RESULT_PATH.parent, directory=True)
    if RESULT_PATH.absolute().parent != parent:
        raise SparseCachePreflightError("result parent identity drifted", "UNSAFE_PATH")
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
        raise SparseCachePreflightError("preflight result already exists", "RESULT_PREEXISTS")
    if shutil.disk_usage(parent).free < FREE_DISK_MINIMUM:
        raise SparseCachePreflightError("free disk gate failed", "RESOURCE_GATE")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise SparseCachePreflightError("short result write", "RESULT_WRITE")
        view = view[written:]


def _publish_result(value: Mapping[str, Any]) -> dict[str, int | str]:
    _guard_experiment_data(RESULT_PATH)
    _privacy_scan(value)
    payload = _canonical_bytes(value) + b"\n"
    if len(payload) > RESULT_BYTES_MAXIMUM:
        raise SparseCachePreflightError("compact result exceeds 24KB", "PRIVACY_GATE")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(RESULT_PATH, flags, 0o600)
    except FileExistsError as error:
        raise SparseCachePreflightError("preflight result already exists", "RESULT_PREEXISTS") from error
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if os.name != "nt":
        directory = os.open(str(RESULT_PATH.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    expected = {
        "bytes": len(payload),
        "rows": 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    observed = _file_identity(RESULT_PATH, "published preflight result").report()
    if observed != expected:
        raise SparseCachePreflightError(
            "published result identity drifted", "RESULT_WRITE"
        )
    return {"bytes": expected["bytes"], "sha256": expected["sha256"]}


def run(implementation_commit: str) -> dict[str, Any]:
    """Run the pushed, target-free 20/100 cache preflight exactly once."""

    del implementation_commit
    raise SparseCachePreflightError(
        "frozen clean-worktree and zero-probe requirements conflict",
        "PREREGISTRATION_CONSTRAINT_CONFLICT",
    )

    guard = _install_process_audit_guard()
    started = time.perf_counter()
    try:
        _assert_fresh_result()
        path_policy = _validate_path_policy()
        environment = _validate_environment()
        protocol = _load_preregistration()
        git = _validate_git_checkpoint(implementation_commit)
        entrypoints = _verify_entrypoints()

        catalog_ids, initial_sources = _source_checkpoint()
        implementation_blobs = git["implementation_blobs"]
        stage20 = _run_stage(20, catalog_ids, implementation_blobs)
        after20_ids, after20_sources = _source_checkpoint()
        after20_git = _validate_git_checkpoint(implementation_commit)
        if after20_ids != catalog_ids or after20_sources != initial_sources or after20_git != git:
            raise SparseCachePreflightError(
                "source/Git changed during stage 20", "SOURCE_MUTATION"
            )

        stage100 = _run_stage(100, catalog_ids, implementation_blobs)
        final_ids, final_sources = _source_checkpoint()
        final_git = _validate_git_checkpoint(implementation_commit)
        if final_ids != catalog_ids or final_sources != initial_sources or final_git != git:
            raise SparseCachePreflightError(
                "source/Git changed during stage 100", "SOURCE_MUTATION"
            )

        result: dict[str, Any] = {
            "claims": {
                "algorithm_go_or_no_go": False,
                "candidate_recall_computed": False,
                "outcome_accessed": False,
                "served_metric_claimed": False,
            },
            "environment": environment,
            "experiment_id": EXPERIMENT_ID,
            "git": git,
            "integrity": {
                "entrypoints": entrypoints,
                "exact_cached_uncached_semantics": True,
                "path_policy": path_policy,
                "source_identity_unchanged": True,
                "v2_19_namespace_touched": False,
            },
            "preregistration": {
                "canonical_sha256": PREREG_CANONICAL_SHA256,
                "commit": PREREG_COMMIT,
                "schema_version": protocol["schema_version"],
            },
            "recorded_on": "2026-08-31",
            "schema_version": SCHEMA_VERSION,
            "self_hash_omitted": True,
            "source_identities": initial_sources,
            "stages": [stage20, stage100],
            "status": "TARGET_FREE_CACHE_PREFLIGHT_PASS_ALLOW_V2_21_PREREGISTRATION",
            "total_parent_wall_seconds": round(time.perf_counter() - started, 6),
        }
        if guard.network_attempt_count:
            raise SparseCachePreflightError(
                "parent network attempt observed", "NETWORK_ATTEMPT"
            )
        _privacy_scan(result, catalog_ids)
        identity = _publish_result(result)
        return {**result, "published_result_identity": identity}
    finally:
        guard.close()


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise SparseCachePreflightError("CLI arguments invalid", "CLI_CONTRACT")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--implementation-commit")
    actions = parser.add_mutually_exclusive_group(required=True)
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
                raise SparseCachePreflightError(
                    "--implementation-commit required", "CLI_CONTRACT"
                )
            result = run(arguments.implementation_commit)
            stage100 = result["stages"][1]
            output = {
                "cached_pair_parent_wall_seconds": stage100[
                    "cached_pair_parent_wall_seconds"
                ],
                "commit": result["git"]["commit"],
                "result_sha256": result["published_result_identity"]["sha256"],
                "status": result["status"],
            }
    except BaseException as error:
        code = (
            error.code
            if isinstance(error, SparseCachePreflightError)
            else "UNEXPECTED_EXCEPTION"
        )
        output = {
            "error_code": code,
            "exception_class": type(error).__name__,
            "raw_traceback_or_stderr_emitted": False,
            "status": "ERROR",
        }
        exit_code = 2
    sys.stdout.buffer.write(_canonical_bytes(output) + b"\n")
    sys.stdout.buffer.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
