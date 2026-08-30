"""One-shot, target-blind C400 candidate-recall probe.

The formal path reuses the already sealed C200 visible-context and trace
artifacts.  Before a durable exclusive receipt exists it may inspect only
target-free sources.  After the receipt it runs two fresh offline C400
workers, proves complete C200-prefix, served-Top10, resource, and exact-repeat
identity, and only then joins targets and numeric folds.  Its result is
aggregate-only diagnostic evidence, never a private score.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat as stat_module
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True

import time
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "small-ranker-c400-candidate-recall-outcome.v1"
WORKER_SCHEMA_VERSION = "small-ranker-c400-candidate-worker-summary.v1"
EXPERIMENT_ID = "SR-V2.17-C400-CANDIDATE-RECALL"
BRANCH = "small-ranker-v2.17-c400-recall"
REMOTE = "origin"
REMOTE_URL = "https://github.com/lamperriat/techjam-err402.git"
REMOTE_REF = "refs/remotes/origin/" + BRANCH

BASE_COMMIT = "c94747edd890c56d4bf6a30edf86777f5868ded8"
PREREG_COMMIT = "59d09aa2a30566642af9bd1ac0c1999db795bc1c"
PREREG_BLOB = "ccf8dff2e2ff1236958b3224ba1370aefda86ccc"
PREREG_RAW_SHA256 = "35e555965fcdd8436da8e4fbcb0e256b6f6b967db6195d11b42c3bb0830c59a7"
PREREG_CANONICAL_SHA256 = (
    "0545f85d96f6b956968c55dacf93f262dbc34e81faff186cd104633e8549c79c"
)
PREREG_BYTES = 10_223
PREREG_RELATIVE = "configs/small_ranker_v2_17.c400_candidate_recall_preregistration.json"
PREREG_PATH = ROOT / PREREG_RELATIVE
PREREG_PATHS = {PREREG_RELATIVE}
IMPLEMENTATION_PATHS = {
    "scripts/c400_candidate_worker.py",
    "scripts/probe_c400_candidate_recall.py",
    "tests/test_c400_candidate_recall.py",
}
PINNED_BLOBS = {
    "starter/agent.py": "421c6d43c598102b8fefb181b72bab5da4bf1294",
    "starter/coverage.py": "59a6507fef63afa0d9761323f5771a52741c811a",
    "evaluator/local_evaluator.py": "7c808347b31ef3121a9cbc4810ac3eb325f950ba",
    "scripts/build_small_ranker_cache.py": "8724f052155ae7405ecd675bdebd83df18685895",
    "scripts/c200_candidate_worker.py": "b94fddcf5a9b20ddde540f3f43ea9962982cb096",
    "scripts/probe_c200_candidate_recall.py": "0a57f63866683b476b9f49184673cf3154531911",
}

SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
CATALOG_PATH = SOURCE_ROOT / "data/catalog.jsonl"
CATALOG_BYTES = 60_546_327
CATALOG_ROWS = 50_000
CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
PROXY_PATH = SOURCE_ROOT / "experiments/fast_track/proxy_v1/proxy_train_explore.jsonl"
PROXY_BYTES = 1_315_338
PROXY_ROWS = 2_000
PROXY_SHA256 = "2175696171c0d874fca4b9aa456ff5fd7d570f2184f59ade6781198f6443198e"
LABEL_PATH = SOURCE_ROOT / "experiments/fast_track/small_ranker_v1/labels_v2.npz"
LABEL_BYTES = 1_702_876
LABEL_SHA256 = "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb"

C200_ROOT = Path(r"D:\tiktok\techjam-v2-16-c200-recall")
C200_RESULT_PATH = C200_ROOT / "experiments/fast_track/small_ranker_v2_16_c200_candidate_recall_20260831.json"
C200_RESULT_RELATIVE = "experiments/fast_track/small_ranker_v2_16_c200_candidate_recall_20260831.json"
C200_RESULT_COMMIT = BASE_COMMIT
C200_IMPLEMENTATION_COMMIT = "b60e767314340c5e110c2509d00c112c755cea42"
C200_RESULT_BLOB = "b219f5baff7e94e615df0311ba5971ea35ecceee"
C200_RESULT_BYTES = 10_214
C200_RESULT_RAW_SHA256 = "e1925d8e962a89d666f2bd2c2cbf7b942cbd0058e0a01a83186a3585531561e4"
C200_RESULT_CANONICAL_SHA256 = (
    "5ae4cb43746cfb0063156ba1d1da3be94c4b6469762517c8fc1fd9f1dcbe1e7c"
)
C200_RECALL_CANONICAL_SHA256 = (
    "05a351b59c162f87d29b50cf572f5780d5f784209582128082af126203d104d4"
)
C200_RESOURCES_CANONICAL_SHA256 = (
    "85f319e545c98c54b6c58547570c278aac01ab10d697168df2e7cb6273abd2ea"
)
C200_SOURCES_CANONICAL_SHA256 = (
    "37ae2eb75e24abfd6383335d2f9c7e3445b6f6a07a31966f787d8d83c0fce4df"
)

C200_CACHE_ROOT = C200_ROOT / "experiments/fast_track/c200_candidate_recall_cache_20260831"
CONTEXT_PATH = C200_CACHE_ROOT / "visible_context.jsonl"
CONTEXT_BYTES = 47_168_882
CONTEXT_ROWS = 2_000
CONTEXT_TURNS = 20_000
CONTEXT_SHA256 = "f30a98700da5d480731fe7e82c87c40a22f06de290e069e20dc68f9fefecd20f"
CONTEXT_REDACTED_MESSAGES = 8
C200_TRACE_PATHS = (
    C200_CACHE_ROOT / "replica_a.jsonl",
    C200_CACHE_ROOT / "replica_b.jsonl",
)
C200_TRACE_ROWS = 20_000
C200_TRACE_BYTES = 32_226_135
C200_TRACE_SHA256 = "a8589749376f48f019997a618481578dde36be4ca1fc723e8ed00056c23e40dc"
C200_CANDIDATE_CELLS = 2_425_785
C100_NORMALIZED_BYTES = 26_690_930
C100_NORMALIZED_SHA256 = "b22b035cb7789570f36db6c52256e5deb67f593f90cbbc5c334d48f2f0a01a67"

OUTPUT_PATH = ROOT / "experiments/fast_track/small_ranker_v2_17_c400_candidate_recall_20260831.json"
CACHE_ROOT = ROOT / "experiments/fast_track/c400_candidate_recall_cache_20260831"
TRACE_PATHS = (CACHE_ROOT / "replica_a.jsonl", CACHE_ROOT / "replica_b.jsonl")
WORKER_PATH = ROOT / "scripts/c400_candidate_worker.py"

SESSION_COUNT = 2_000
TURN_COUNT = 10
RECORD_COUNT = SESSION_COUNT * TURN_COUNT
CUTOFFS = (10, 20, 50, 100, 200, 400)
EXPECTED_C200_RECALL = {10: 1_895, 20: 1_943, 50: 1_982, 100: 1_986, 200: 1_986}
EXPECTED_C200_FRONTIER = 14
EXPECTED_TAXONOMY = frozenset({"accessories-other", "clothing", "jewelry", "shoes"})

EXPECTED_EXECUTABLE = Path(r"D:\450\conda\envs\tiktok\python.exe")
EXPECTED_PYTHON = "3.11.16"
EXPECTED_SQLITE = "3.53.4"
EXPECTED_NUMPY = "2.4.6"
TOTAL_WALL_MAXIMUM = 3_600.0
RESPOND_P95_MS_MAXIMUM = 400.0
EXPANSION_P95_MS_MAXIMUM = 250.0
WORKER_RSS_MAXIMUM = 2_147_483_648
WORKER_RSS_SUM_MAXIMUM = 4_294_967_296
CELL_RATIO_C100_MAXIMUM = 4.0
CELL_RATIO_C200_MAXIMUM = 3.3
TRACE_RATIO_C100_MAXIMUM = 4.1
TRACE_RATIO_C200_MAXIMUM = 3.4

ASIN_SHAPE_RE = re.compile(rb"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE)
CATALOG_IDENTIFIER_RE = re.compile(rb"[A-Z0-9]{10}")
CATALOG_IDENTIFIER_TOKEN_RE = re.compile(
    rb"(?<![A-Z0-9])[A-Z0-9]{10}(?![A-Z0-9])", re.IGNORECASE
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
NONCE_RE = re.compile(r"[0-9a-f]{32}")
FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "asin", "parent_asin", "sample_id", "scenario_type", "ground_truth",
        "target", "target_id", "target_asin", "eligible_from", "outer_fold",
        "family_index", "future_turns", "evaluator_metadata", "user_id", "ordinal",
    }
)


class C400ProbeError(RuntimeError):
    """Raised when the C400 evidence boundary cannot be trusted."""


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
    normalized_c200_sha256: str
    normalized_c200_bytes: int
    record_count: int


@dataclass(frozen=True)
class C200Reference:
    candidates: tuple[tuple[str, ...], ...]
    lengths: tuple[int, ...]
    identity: FileIdentity
    candidate_cells: int
    normalized_c100_sha256: str
    normalized_c100_bytes: int


@dataclass(frozen=True)
class Preflight:
    environment: Mapping[str, Any]
    git: Mapping[str, Any]
    protocol: Mapping[str, Any]
    catalog_ids: frozenset[str]
    products: Mapping[str, Mapping[str, Any]]
    c200_reference: C200Reference
    source_identities: Mapping[str, Any]
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


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise C400ProbeError("duplicate JSON key")
        result[key] = value
    return result


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_link_or_reparse(path: Path) -> bool:
    try:
        observed = path.lstat()
    except OSError as error:
        raise C400ProbeError("path identity is unavailable") from error
    attributes = getattr(observed, "st_file_attributes", 0)
    reparse = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse)


def _snapshot_from_stat(observed: os.stat_result) -> tuple[int, int, int]:
    return (
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(getattr(observed, "st_ino", 0)),
    )


def _path_snapshot(path: Path) -> tuple[int, int, int]:
    return _snapshot_from_stat(path.stat())


def _handle_snapshot(handle: BinaryIO) -> tuple[int, int, int]:
    return _snapshot_from_stat(os.fstat(handle.fileno()))


def _require_lexical_ancestry(path: Path, anchor: Path, label: str) -> None:
    lexical_anchor = anchor.absolute()
    lexical_path = path.absolute()
    if not _inside(lexical_path, lexical_anchor):
        raise C400ProbeError(f"{label} escaped its lexical anchor")
    current = lexical_anchor
    components = (current,)
    for part in lexical_path.relative_to(lexical_anchor).parts:
        current = current / part
        components += (current,)
    for component in components:
        if component.exists() or component.is_symlink():
            if _is_link_or_reparse(component):
                raise C400ProbeError(f"{label} traverses a link or reparse point")


def _require_regular_file(path: Path, label: str) -> Path:
    workspace_path = Path(r"D:\tiktok")
    _require_lexical_ancestry(path, workspace_path, label)
    try:
        resolved = path.resolve(strict=True)
        workspace = workspace_path.resolve(strict=True)
    except OSError as error:
        raise C400ProbeError(f"{label} is unavailable") from error
    if not _inside(resolved, workspace) or not resolved.is_file():
        raise C400ProbeError(f"{label} escaped the workspace or is not a file")
    return resolved


def _require_plain_regular_file(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise C400ProbeError(f"{label} is unavailable") from error
    if not resolved.is_file() or _is_link_or_reparse(path):
        raise C400ProbeError(f"{label} is not a plain regular file")
    return resolved


def _hash_handle(handle: BinaryIO) -> tuple[str, int]:
    handle.seek(0)
    digest = hashlib.sha256()
    byte_count = 0
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
        byte_count += len(chunk)
    handle.seek(0)
    return digest.hexdigest(), byte_count


def _file_identity(path: Path, label: str) -> FileIdentity:
    resolved = _require_regular_file(path, label)
    before = _path_snapshot(resolved)
    digest = hashlib.sha256()
    byte_count = row_count = 0
    with resolved.open("rb") as handle:
        for line in handle:
            digest.update(line)
            byte_count += len(line)
            row_count += 1
    after = _path_snapshot(resolved)
    if before != after or byte_count != before[0]:
        raise C400ProbeError(f"{label} changed while hashed")
    return FileIdentity(byte_count, row_count, digest.hexdigest(), after)


def _git(*args: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise C400ProbeError("Git identity command failed")
    return completed.stdout.strip()


def _changed_paths(commitish: str) -> set[str]:
    output = _git("diff-tree", "--no-commit-id", "--name-only", "-r", commitish)
    return {line for line in output.splitlines() if line}


def _diff_paths(left: str, right: str) -> set[str]:
    output = _git("diff", "--name-only", left, right)
    return {line for line in output.splitlines() if line}


def _validate_environment() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    try:
        executable_ok = (
            executable.as_posix().casefold()
            == EXPECTED_EXECUTABLE.resolve(strict=True).as_posix().casefold()
        )
    except OSError:
        executable_ok = False
    observed = {
        "executable": executable.as_posix(),
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "numpy": np.__version__,
        "pythonhashseed": os.getenv("PYTHONHASHSEED"),
    }
    if not (
        executable_ok
        and observed["python"] == EXPECTED_PYTHON
        and observed["sqlite"] == EXPECTED_SQLITE
        and observed["numpy"] == EXPECTED_NUMPY
        and observed["pythonhashseed"] == "0"
    ):
        raise C400ProbeError("formal environment identity drifted")
    return observed


def _load_preregistration() -> dict[str, Any]:
    path = _require_regular_file(PREREG_PATH, "C400 preregistration")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise C400ProbeError("C400 preregistration JSON is invalid") from error
    if not isinstance(value, dict) or not (
        len(raw) == PREREG_BYTES
        and hashlib.sha256(raw).hexdigest() == PREREG_RAW_SHA256
        and _canonical_sha256(value) == PREREG_CANONICAL_SHA256
        and value.get("schema_version")
        == "small-ranker-c400-candidate-recall-preregistration.v1"
        and value.get("status") == "PREREGISTERED_BEFORE_C400_IMPLEMENTATION_AND_OUTCOME"
        and value.get("parent_commit") == BASE_COMMIT
        and _git("rev-parse", f"{PREREG_COMMIT}:{PREREG_RELATIVE}") == PREREG_BLOB
    ):
        raise C400ProbeError("C400 preregistration identity drifted")
    return value


def _validate_git_checkpoint(implementation_commit: str) -> dict[str, Any]:
    if not COMMIT_RE.fullmatch(implementation_commit):
        raise C400ProbeError("implementation commit is invalid")
    head = _git("rev-parse", "HEAD")
    parent = _git("rev-parse", "HEAD^")
    prereg_parent = _git("rev-parse", f"{PREREG_COMMIT}^")
    branch = _git("branch", "--show-current")
    remote_url = _git("remote", "get-url", REMOTE)
    remote_head = _git("rev-parse", REMOTE_REF)
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    pinned = {path: _git("rev-parse", f"HEAD:{path}") for path in PINNED_BLOBS}
    c200_result_blob = _git("rev-parse", f"{C200_RESULT_COMMIT}:{C200_RESULT_RELATIVE}")
    if not (
        head == implementation_commit
        and parent == PREREG_COMMIT
        and prereg_parent == BASE_COMMIT
        and branch == BRANCH
        and remote_url == REMOTE_URL
        and remote_head == head
        and not status
        and _changed_paths(PREREG_COMMIT) == PREREG_PATHS
        and _diff_paths(PREREG_COMMIT, head) == IMPLEMENTATION_PATHS
        and pinned == PINNED_BLOBS
        and c200_result_blob == C200_RESULT_BLOB
    ):
        raise C400ProbeError("Git checkpoint gate failed")
    implementation_blobs = {
        path: _git("rev-parse", f"HEAD:{path}") for path in sorted(IMPLEMENTATION_PATHS)
    }
    return {
        "branch": branch,
        "commit": head,
        "parent": parent,
        "preregistration_commit": PREREG_COMMIT,
        "base_result_commit": BASE_COMMIT,
        "remote_equal": True,
        "clean_including_untracked_nonignored": True,
        "exact_changed_paths": True,
        "pinned_blobs": pinned,
        "implementation_blobs": implementation_blobs,
        "sealed_c200_result_blob": c200_result_blob,
    }


def _load_catalog_target_free() -> tuple[frozenset[str], dict[str, Mapping[str, Any]], FileIdentity]:
    identity = _file_identity(CATALOG_PATH, "catalog")
    if identity.report() != {
        "bytes": CATALOG_BYTES,
        "rows": CATALOG_ROWS,
        "sha256": CATALOG_SHA256,
    }:
        raise C400ProbeError("catalog identity drifted")
    products: dict[str, Mapping[str, Any]] = {}
    with CATALOG_PATH.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            value = json.loads(line, object_pairs_hook=_unique_object)
            identifier = str(value.get("parent_asin", "")) if isinstance(value, dict) else ""
            if (
                not identifier
                or not identifier.isascii()
                or CATALOG_IDENTIFIER_RE.fullmatch(identifier.encode("ascii")) is None
                or identifier != identifier.upper()
                or identifier in products
            ):
                raise C400ProbeError("catalog identifier surface drifted")
            products[identifier] = value
    if len(products) != CATALOG_ROWS:
        raise C400ProbeError("catalog row count drifted")
    return frozenset(products), products, identity


def _walk_values(value: object) -> Iterable[object]:
    yield value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_values(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_values(child)


def _result_privacy_scan(value: object, *, catalog_ids: Iterable[str] = ()) -> None:
    catalog_membership = frozenset(str(item).upper() for item in catalog_ids)
    forbidden_keys = {
        "session_id", "sample_id", "product_id", "target", "target_id",
        "ground_truth", "positive_index", "eligible_from", "message",
        "per_session", "membership_vector",
    }
    for item in _walk_values(value):
        if isinstance(item, Mapping):
            if {str(key).casefold() for key in item} & forbidden_keys:
                raise C400ProbeError("result contains a forbidden identity-bearing key")
        elif isinstance(item, np.ndarray):
            raise C400ProbeError("result contains a numeric array")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            if len(item) >= SESSION_COUNT:
                raise C400ProbeError("result contains a session-length vector")
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            exact_tokens = {
                match.group(0).decode("ascii").upper()
                for match in CATALOG_IDENTIFIER_TOKEN_RE.finditer(encoded)
            }
            if ASIN_SHAPE_RE.search(encoded) or exact_tokens & catalog_membership:
                raise C400ProbeError("result contains an identifier-shaped token")


def _load_sealed_c200_result(catalog_ids: frozenset[str]) -> tuple[dict[str, Any], FileIdentity]:
    identity = _file_identity(C200_RESULT_PATH, "sealed C200 aggregate result")
    if identity.report() != {
        "bytes": C200_RESULT_BYTES,
        "rows": 1,
        "sha256": C200_RESULT_RAW_SHA256,
    }:
        raise C400ProbeError("sealed C200 result raw identity drifted")
    raw = C200_RESULT_PATH.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise C400ProbeError("sealed C200 result JSON is invalid") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise C400ProbeError("sealed C200 result is not canonical LF JSON")
    candidate_recall = value.get("candidate_recall")
    resources = value.get("resources")
    sources = value.get("source_identities")
    implementation = value.get("implementation")
    exact_repeat = value.get("exact_repeat")
    inflation = value.get("inflation")
    retention = value.get("candidate_retention")
    all_sessions = candidate_recall.get("all_sessions") if isinstance(candidate_recall, dict) else None
    frontier = candidate_recall.get("c100_absent_frontier") if isinstance(candidate_recall, dict) else None
    if not (
        _canonical_sha256(value) == C200_RESULT_CANONICAL_SHA256
        and value.get("schema_version") == "small-ranker-c200-candidate-recall-outcome.v1"
        and value.get("status") == "C200_RECALL_NO_GO_CONTINUE_C400"
        and isinstance(implementation, dict)
        and implementation.get("commit") == C200_IMPLEMENTATION_COMMIT
        and implementation.get("runtime_target_blind") is True
        and implementation.get("protected_splits_opened") is False
        and isinstance(candidate_recall, dict)
        and _canonical_sha256(candidate_recall) == C200_RECALL_CANONICAL_SHA256
        and isinstance(resources, dict)
        and _canonical_sha256(resources) == C200_RESOURCES_CANONICAL_SHA256
        and resources.get("budgets_passed") is True
        and resources.get("network_attempt_count") == 0
        and resources.get("gpu_peak_bytes") == 0
        and isinstance(sources, dict)
        and _canonical_sha256(sources) == C200_SOURCES_CANONICAL_SHA256
        and isinstance(exact_repeat, dict)
        and exact_repeat.get("passed") is True
        and exact_repeat.get("trace_sha256") == C200_TRACE_SHA256
        and exact_repeat.get("trace_bytes") == C200_TRACE_BYTES
        and exact_repeat.get("record_count") == C200_TRACE_ROWS
        and isinstance(inflation, dict)
        and inflation.get("candidate_cells") == C200_CANDIDATE_CELLS
        and isinstance(retention, dict)
        and retention.get("normalized_c100_sha256") == C100_NORMALIZED_SHA256
        and retention.get("normalized_c100_bytes") == C100_NORMALIZED_BYTES
        and isinstance(all_sessions, dict)
        and all_sessions.get("c200", {}).get("count") == EXPECTED_C200_RECALL[200]
        and isinstance(frontier, dict)
        and frontier.get("sessions") == EXPECTED_C200_FRONTIER
    ):
        raise C400ProbeError("sealed C200 result semantic identity drifted")
    for cutoff, expected in EXPECTED_C200_RECALL.items():
        if all_sessions.get(f"c{cutoff}", {}).get("count") != expected:
            raise C400ProbeError("sealed C200 recall sanity drifted")
    _result_privacy_scan(value, catalog_ids=catalog_ids)
    return value, identity


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).casefold())
            keys.update(_walk_keys(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def _load_visible_context_target_free(catalog_ids: frozenset[str]) -> FileIdentity:
    resolved = _require_regular_file(CONTEXT_PATH, "sealed visible context")
    before = _path_snapshot(resolved)
    digest = hashlib.sha256()
    byte_count = row_count = turn_count = 0
    with resolved.open("rb") as handle:
        for line in handle:
            if not line.strip():
                raise C400ProbeError("sealed visible context contains a blank row")
            digest.update(line)
            byte_count += len(line)
            row_count += 1
            try:
                value = json.loads(line.decode("utf-8", errors="strict"), object_pairs_hook=_unique_object)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise C400ProbeError("sealed visible context JSONL is invalid") from error
            turns = value.get("turns") if isinstance(value, dict) else None
            if (
                not isinstance(value, dict)
                or set(value) != {"schema_version", "turns"}
                or value.get("schema_version") != "small-ranker-visible-context.v1"
                or not isinstance(turns, list)
                or len(turns) != TURN_COUNT
                or _walk_keys(value) & FORBIDDEN_CONTEXT_KEYS
                or line != _canonical_bytes(value) + b"\n"
            ):
                raise C400ProbeError("sealed visible context schema drifted")
            exact_tokens = {
                match.group(0).decode("ascii").upper()
                for match in CATALOG_IDENTIFIER_TOKEN_RE.finditer(line)
            }
            if ASIN_SHAPE_RE.search(line) or exact_tokens & catalog_ids:
                raise C400ProbeError("sealed visible context contains a catalog identifier")
            turn_count += len(turns)
    after = _path_snapshot(resolved)
    identity = FileIdentity(byte_count, row_count, digest.hexdigest(), after)
    if not (
        before == after
        and byte_count == before[0]
        and turn_count == CONTEXT_TURNS
        and identity.report()
        == {"bytes": CONTEXT_BYTES, "rows": CONTEXT_ROWS, "sha256": CONTEXT_SHA256}
    ):
        raise C400ProbeError("sealed visible context identity drifted")
    return identity


def _canonical_c200_line(ordinal: int, turn: int, candidates: Sequence[str]) -> bytes:
    return _canonical_bytes({"c200": list(candidates), "ordinal": ordinal, "turn": turn}) + b"\n"


def _canonical_c400_line(ordinal: int, turn: int, candidates: Sequence[str]) -> bytes:
    return _canonical_bytes({"c400": list(candidates), "ordinal": ordinal, "turn": turn}) + b"\n"


def _catalog_intern_map(catalog_ids: Iterable[str]) -> dict[str, str]:
    return {identifier: identifier for identifier in catalog_ids}


def _validate_candidate_sequence(
    value: object,
    canonical_ids: Mapping[str, str],
    *,
    minimum: int,
    maximum: int,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise C400ProbeError(f"{label} is not an ordered sequence")
    if not minimum <= len(value) <= maximum:
        raise C400ProbeError(f"{label} length drifted")
    result: list[str] = []
    observed: set[str] = set()
    for identifier in value:
        if not isinstance(identifier, str) or identifier not in canonical_ids or identifier in observed:
            raise C400ProbeError(f"{label} candidate surface is invalid")
        observed.add(identifier)
        result.append(canonical_ids[identifier])
    return tuple(result)


def _load_c200_reference(
    path: Path,
    catalog_ids: frozenset[str],
    *,
    retain_candidates: bool,
) -> C200Reference:
    resolved = _require_regular_file(path, "sealed C200 reference")
    before = _path_snapshot(resolved)
    canonical_ids = _catalog_intern_map(catalog_ids)
    digest = hashlib.sha256()
    c100_digest = hashlib.sha256()
    byte_count = c100_bytes = row_count = candidate_cells = 0
    candidates_out: list[tuple[str, ...]] = []
    lengths: list[int] = []
    with resolved.open("rb") as handle:
        for line in handle:
            if row_count >= RECORD_COUNT or not line.strip():
                raise C400ProbeError("sealed C200 reference framing drifted")
            ordinal = row_count // TURN_COUNT + 1
            turn = row_count % TURN_COUNT + 1
            try:
                value = json.loads(line.decode("utf-8", errors="strict"), object_pairs_hook=_unique_object)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise C400ProbeError("sealed C200 reference JSONL is invalid") from error
            if (
                not isinstance(value, dict)
                or set(value) != {"c200", "ordinal", "turn"}
                or value.get("ordinal") != ordinal
                or isinstance(value.get("ordinal"), bool)
                or value.get("turn") != turn
                or isinstance(value.get("turn"), bool)
            ):
                raise C400ProbeError("sealed C200 reference schema or order drifted")
            candidates = _validate_candidate_sequence(
                value.get("c200"), canonical_ids, minimum=100, maximum=200, label="C200"
            )
            expected_line = _canonical_c200_line(ordinal, turn, candidates)
            if line != expected_line:
                raise C400ProbeError("sealed C200 row is not canonical LF JSON")
            c100_line = _canonical_bytes(
                {"c100": list(candidates[:100]), "ordinal": ordinal, "turn": turn}
            ) + b"\n"
            digest.update(line)
            c100_digest.update(c100_line)
            byte_count += len(line)
            c100_bytes += len(c100_line)
            candidate_cells += len(candidates)
            lengths.append(len(candidates))
            if retain_candidates:
                candidates_out.append(candidates)
            row_count += 1
    after = _path_snapshot(resolved)
    identity = FileIdentity(byte_count, row_count, digest.hexdigest(), after)
    if not (
        before == after
        and byte_count == before[0]
        and identity.report()
        == {"bytes": C200_TRACE_BYTES, "rows": C200_TRACE_ROWS, "sha256": C200_TRACE_SHA256}
        and candidate_cells == C200_CANDIDATE_CELLS
        and c100_bytes == C100_NORMALIZED_BYTES
        and c100_digest.hexdigest() == C100_NORMALIZED_SHA256
        and (not retain_candidates or len(candidates_out) == RECORD_COUNT)
    ):
        raise C400ProbeError("sealed C200 reference identity drifted")
    return C200Reference(
        candidates=tuple(candidates_out),
        lengths=tuple(lengths),
        identity=identity,
        candidate_cells=candidate_cells,
        normalized_c100_sha256=c100_digest.hexdigest(),
        normalized_c100_bytes=c100_bytes,
    )


def _files_equal(left: Path, right: Path) -> bool:
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            left_chunk = first.read(1 << 20)
            right_chunk = second.read(1 << 20)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _process_memory() -> tuple[int, int]:
    if os.name != "nt":
        return 0, 0
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return 0, 0


def preflight_only(implementation_commit: str) -> Preflight:
    """Validate only target-free sources; never inspect proxy or label paths."""

    environment = _validate_environment()
    protocol_value = _load_preregistration()
    git = _validate_git_checkpoint(implementation_commit)
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink() or CACHE_ROOT.exists() or CACHE_ROOT.is_symlink():
        raise C400ProbeError("formal C400 receipt or fresh cache path already exists")
    catalog_ids, products, catalog_identity = _load_catalog_target_free()
    sealed_result, sealed_result_identity = _load_sealed_c200_result(catalog_ids)
    context_identity = _load_visible_context_target_free(catalog_ids)
    reference_a = _load_c200_reference(C200_TRACE_PATHS[0], catalog_ids, retain_candidates=True)
    reference_b = _load_c200_reference(C200_TRACE_PATHS[1], catalog_ids, retain_candidates=False)
    if not (
        reference_a.identity.report() == reference_b.identity.report()
        and reference_a.lengths == reference_b.lengths
        and reference_a.candidate_cells == reference_b.candidate_cells
        and _files_equal(C200_TRACE_PATHS[0], C200_TRACE_PATHS[1])
    ):
        raise C400ProbeError("sealed C200 replicas are not exact repeats")
    visible_source = sealed_result.get("source_identities", {}).get("visible_context", {})
    if not (
        visible_source.get("bytes") == CONTEXT_BYTES
        and visible_source.get("rows") == CONTEXT_ROWS
        and visible_source.get("sha256") == CONTEXT_SHA256
        and visible_source.get("turns") == CONTEXT_TURNS
        and visible_source.get("redacted_message_count") == CONTEXT_REDACTED_MESSAGES
    ):
        raise C400ProbeError("sealed C200 result does not bind the reused context")
    worker_path = _require_regular_file(WORKER_PATH, "C400 worker")
    source_identities = {
        "catalog": catalog_identity.report(),
        "sealed_c200_result": sealed_result_identity.report(),
        "visible_context": {**context_identity.report(), "turns": CONTEXT_TURNS},
        "sealed_c200_traces": [reference_a.identity.report(), reference_b.identity.report()],
        "sealed_c200_candidate_cells": reference_a.candidate_cells,
        "normalized_c100": {
            "bytes": reference_a.normalized_c100_bytes,
            "sha256": reference_a.normalized_c100_sha256,
        },
        "worker_sha256": hashlib.sha256(worker_path.read_bytes()).hexdigest(),
    }
    memory_before_receipt = _process_memory()
    if not 0 < memory_before_receipt[0] <= memory_before_receipt[1]:
        raise C400ProbeError("parent working-set measurement is unavailable")
    return Preflight(
        environment=environment,
        git=git,
        protocol={
            "commit": PREREG_COMMIT,
            "git_blob_oid": PREREG_BLOB,
            "raw_sha256": PREREG_RAW_SHA256,
            "canonical_sha256": PREREG_CANONICAL_SHA256,
            "schema_version": protocol_value["schema_version"],
        },
        catalog_ids=catalog_ids,
        products=products,
        c200_reference=reference_a,
        source_identities=source_identities,
        memory_before_receipt=memory_before_receipt,
    )


def _reference_candidates(value: object) -> Sequence[str]:
    if isinstance(value, Mapping):
        candidate_value = value.get("c200")
        if isinstance(candidate_value, (list, tuple)):
            return candidate_value
    if isinstance(value, (list, tuple)):
        return value
    raise C400ProbeError("sealed C200 reference record is invalid")


def validate_trace_records(
    records: Sequence[Mapping[str, Any]],
    frozen_c200: Sequence[object],
    catalog_ids: Iterable[str],
    *,
    expected_records: int,
) -> TraceValidation:
    """Validate C400 rows and bind every row to its full variable C200 prefix."""

    if (
        not isinstance(expected_records, int)
        or isinstance(expected_records, bool)
        or expected_records <= 0
        or len(records) != expected_records
        or len(frozen_c200) != expected_records
    ):
        raise C400ProbeError("C400 trace record count drifted")
    canonical_ids = _catalog_intern_map(catalog_ids)
    canonical = hashlib.sha256()
    c200_digest = hashlib.sha256()
    canonical_bytes = c200_bytes = 0
    normalized_records: list[dict[str, Any]] = []
    lengths: list[int] = []
    c200_lengths: list[int] = []
    for index, (record, reference_value) in enumerate(
        zip(records, frozen_c200, strict=True)
    ):
        ordinal = index // TURN_COUNT + 1
        turn = index % TURN_COUNT + 1
        if (
            not isinstance(record, Mapping)
            or set(record) != {"c400", "ordinal", "turn"}
            or not isinstance(record.get("ordinal"), int)
            or isinstance(record.get("ordinal"), bool)
            or record.get("ordinal") != ordinal
            or not isinstance(record.get("turn"), int)
            or isinstance(record.get("turn"), bool)
            or record.get("turn") != turn
        ):
            raise C400ProbeError("C400 trace schema or order drifted")
        reference = _validate_candidate_sequence(
            _reference_candidates(reference_value),
            canonical_ids,
            minimum=100,
            maximum=200,
            label="sealed C200",
        )
        c400 = _validate_candidate_sequence(
            record.get("c400"),
            canonical_ids,
            minimum=len(reference),
            maximum=400,
            label="C400",
        )
        if c400[: len(reference)] != reference:
            raise C400ProbeError("C400 lost or reordered the full sealed C200 prefix")
        trace_line = _canonical_c400_line(ordinal, turn, c400)
        reference_line = _canonical_c200_line(ordinal, turn, reference)
        canonical.update(trace_line)
        c200_digest.update(reference_line)
        canonical_bytes += len(trace_line)
        c200_bytes += len(reference_line)
        normalized_records.append(
            {"c400": c400, "ordinal": ordinal, "turn": turn}
        )
        lengths.append(len(c400))
        c200_lengths.append(len(reference))
    return TraceValidation(
        records=tuple(normalized_records),
        lengths=tuple(lengths),
        c200_lengths=tuple(c200_lengths),
        canonical_trace_sha256=canonical.hexdigest(),
        canonical_trace_bytes=canonical_bytes,
        normalized_c200_sha256=c200_digest.hexdigest(),
        normalized_c200_bytes=c200_bytes,
        record_count=expected_records,
    )


def load_and_validate_c400_trace(
    path: Path,
    frozen_c200: Sequence[object],
    catalog_ids: Iterable[str],
    *,
    expected_records: int = RECORD_COUNT,
) -> TraceValidation:
    """Load a closed trace for public fixtures or the retained formal replica."""

    workspace = Path(r"D:\tiktok").absolute()
    resolved = (
        _require_regular_file(path, "C400 trace")
        if _inside(path.absolute(), workspace)
        else _require_plain_regular_file(path, "C400 trace")
    )
    records: list[dict[str, Any]] = []
    raw_digest = hashlib.sha256()
    raw_bytes = 0
    before = _path_snapshot(resolved)
    with resolved.open("rb") as handle:
        for line in handle:
            if not line.strip():
                raise C400ProbeError("C400 trace contains a blank row")
            try:
                value = json.loads(
                    line.decode("utf-8", errors="strict"),
                    object_pairs_hook=_unique_object,
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                raise C400ProbeError("C400 trace JSONL is invalid") from error
            if not isinstance(value, dict):
                raise C400ProbeError("C400 trace row is not an object")
            try:
                canonical_line = _canonical_c400_line(
                    value.get("ordinal"), value.get("turn"), value.get("c400", ())
                )
            except (TypeError, ValueError) as error:
                raise C400ProbeError("C400 trace row surface is invalid") from error
            if line != canonical_line:
                raise C400ProbeError("C400 trace row is not canonical LF JSON")
            raw_digest.update(line)
            raw_bytes += len(line)
            records.append(value)
    after = _path_snapshot(resolved)
    if before != after or raw_bytes != before[0]:
        raise C400ProbeError("C400 trace changed while loaded")
    validation = validate_trace_records(
        records,
        frozen_c200,
        catalog_ids,
        expected_records=expected_records,
    )
    if (
        raw_digest.hexdigest() != validation.canonical_trace_sha256
        or raw_bytes != validation.canonical_trace_bytes
    ):
        raise C400ProbeError("C400 trace raw/canonical identity differs")
    return validation


def _load_and_validate_c400_trace_streaming(
    path: Path,
    frozen_c200: Sequence[object],
    catalog_ids: Iterable[str],
    *,
    expected_records: int = RECORD_COUNT,
    retain_records: bool = False,
) -> TraceValidation:
    """Validate a formal trace while retaining at most one replica's cells."""

    resolved = _require_regular_file(path, "streamed C400 trace")
    if len(frozen_c200) != expected_records:
        raise C400ProbeError("streamed C400 reference dimension drifted")
    before = _path_snapshot(resolved)
    canonical_ids = _catalog_intern_map(catalog_ids)
    digest = hashlib.sha256()
    c200_digest = hashlib.sha256()
    byte_count = c200_bytes = row_count = 0
    lengths: list[int] = []
    c200_lengths: list[int] = []
    retained: list[dict[str, Any]] = []
    with resolved.open("rb") as handle:
        for line in handle:
            if row_count >= expected_records or not line.strip():
                raise C400ProbeError("streamed C400 trace framing drifted")
            ordinal = row_count // TURN_COUNT + 1
            turn = row_count % TURN_COUNT + 1
            try:
                value = json.loads(
                    line.decode("utf-8", errors="strict"),
                    object_pairs_hook=_unique_object,
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                raise C400ProbeError("streamed C400 trace JSONL is invalid") from error
            if (
                not isinstance(value, dict)
                or set(value) != {"c400", "ordinal", "turn"}
                or value.get("ordinal") != ordinal
                or isinstance(value.get("ordinal"), bool)
                or value.get("turn") != turn
                or isinstance(value.get("turn"), bool)
            ):
                raise C400ProbeError("streamed C400 trace schema or order drifted")
            reference = _validate_candidate_sequence(
                _reference_candidates(frozen_c200[row_count]),
                canonical_ids,
                minimum=100,
                maximum=200,
                label="sealed C200",
            )
            c400 = _validate_candidate_sequence(
                value.get("c400"),
                canonical_ids,
                minimum=len(reference),
                maximum=400,
                label="C400",
            )
            expected_line = _canonical_c400_line(ordinal, turn, c400)
            if line != expected_line or c400[: len(reference)] != reference:
                raise C400ProbeError("streamed C400 prefix or canonical gate failed")
            reference_line = _canonical_c200_line(ordinal, turn, reference)
            digest.update(line)
            c200_digest.update(reference_line)
            byte_count += len(line)
            c200_bytes += len(reference_line)
            lengths.append(len(c400))
            c200_lengths.append(len(reference))
            if retain_records:
                retained.append({"c400": c400, "ordinal": ordinal, "turn": turn})
            row_count += 1
    after = _path_snapshot(resolved)
    if not (
        before == after
        and byte_count == before[0]
        and row_count == expected_records
    ):
        raise C400ProbeError("streamed C400 trace identity drifted")
    return TraceValidation(
        records=tuple(retained),
        lengths=tuple(lengths),
        c200_lengths=tuple(c200_lengths),
        canonical_trace_sha256=digest.hexdigest(),
        canonical_trace_bytes=byte_count,
        normalized_c200_sha256=c200_digest.hexdigest(),
        normalized_c200_bytes=c200_bytes,
        record_count=row_count,
    )


def candidate_recall_flags(
    target: str,
    eligible_from: int,
    c400_turns: Sequence[Mapping[str, Any]],
    sealed_c200_turns: Sequence[object],
    cutoffs: Sequence[int] = CUTOFFS,
) -> dict[int, bool]:
    """Membership flags with C200 defined by the sealed variable-length pool.

    The numeric position 200 in C400 is deliberately not used as a surrogate
    for C200: a sealed pool may end at rank 100, making rank 101 diagnostic.
    """

    if (
        not isinstance(target, str)
        or not target
        or not isinstance(eligible_from, int)
        or isinstance(eligible_from, bool)
        or not 1 <= eligible_from <= TURN_COUNT
        or tuple(cutoffs) != CUTOFFS
        or len(c400_turns) != len(sealed_c200_turns)
    ):
        raise C400ProbeError("candidate recall input is invalid")
    result = {cutoff: False for cutoff in CUTOFFS}
    for c400_row, reference_value in zip(c400_turns, sealed_c200_turns, strict=True):
        turn = c400_row.get("turn") if isinstance(c400_row, Mapping) else None
        c400 = c400_row.get("c400") if isinstance(c400_row, Mapping) else None
        c200 = _reference_candidates(reference_value)
        if (
            not isinstance(turn, int)
            or isinstance(turn, bool)
            or not isinstance(c400, (list, tuple))
            or not isinstance(c200, (list, tuple))
            or tuple(c400[: len(c200)]) != tuple(c200)
        ):
            raise C400ProbeError("candidate recall trace/reference row is invalid")
        if turn < eligible_from:
            continue
        for cutoff in (10, 20, 50, 100):
            result[cutoff] = result[cutoff] or target in c200[:cutoff]
        result[200] = result[200] or target in c200
        result[400] = result[400] or target in c400
    return result


def _recall_view(
    flags: Sequence[Mapping[int, bool]], indices: Sequence[int]
) -> dict[str, Any]:
    denominator = len(indices)
    result: dict[str, Any] = {}
    for cutoff in CUTOFFS:
        count = sum(int(flags[index][cutoff]) for index in indices)
        result[f"c{cutoff}"] = {
            "count": count,
            "fraction": round(count / denominator, 6) if denominator else 0.0,
        }
    return result


def aggregate_candidate_recall(
    flags: Sequence[Mapping[int, bool]],
    *,
    outer_fold: Sequence[int],
    family_index: Sequence[int],
    taxonomy: Sequence[str],
    sealed_c200_lengths: Sequence[int],
) -> dict[str, Any]:
    count = len(flags)
    if not count or not (
        len(outer_fold) == len(family_index) == len(taxonomy) == count
    ) or not sealed_c200_lengths or len(sealed_c200_lengths) % count:
        raise C400ProbeError("candidate recall aggregate dimensions drifted")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 100 <= value <= 200
        for value in sealed_c200_lengths
    ):
        raise C400ProbeError("sealed C200 aggregate surface drifted")
    for row in flags:
        if set(row) != set(CUTOFFS) or any(
            not isinstance(row[cutoff], bool) for cutoff in CUTOFFS
        ):
            raise C400ProbeError("candidate recall flag schema drifted")
    family_fold: dict[int, int] = {}
    for family, fold in zip(family_index, outer_fold, strict=True):
        if (
            not isinstance(family, (int, np.integer))
            or isinstance(family, (bool, np.bool_))
            or not isinstance(fold, (int, np.integer))
            or isinstance(fold, (bool, np.bool_))
            or not 0 <= int(fold) < 5
        ):
            raise C400ProbeError("fold/family label is invalid")
        previous = family_fold.setdefault(int(family), int(fold))
        if previous != int(fold):
            raise C400ProbeError("one product family crosses outer folds")
    indices = list(range(count))
    frontier = [index for index in indices if not flags[index][200]]
    increment = [index for index in frontier if flags[index][400]]
    increment_set = set(increment)
    by_fold: list[dict[str, Any]] = []
    for fold in sorted({int(value) for value in outer_fold}):
        members = [index for index, value in enumerate(outer_fold) if int(value) == fold]
        by_fold.append(
            {
                "fold": fold,
                "sessions": len(members),
                "recall": _recall_view(flags, members),
                "increment": sum(int(index in increment_set) for index in members),
            }
        )
    by_taxonomy: dict[str, Any] = {}
    for name in sorted({str(value) for value in taxonomy}):
        members = [index for index, value in enumerate(taxonomy) if str(value) == name]
        by_taxonomy[name] = {
            "sessions": len(members),
            "recall": _recall_view(flags, members),
            "increment": sum(int(index in increment_set) for index in members),
        }
    family_members: dict[int, list[int]] = {}
    for index, family in enumerate(family_index):
        family_members.setdefault(int(family), []).append(index)
    target_uniform: dict[str, Any] = {"cluster_count": len(family_members)}
    for cutoff in CUTOFFS:
        cluster_rates = [
            statistics.fmean(int(flags[index][cutoff]) for index in members)
            for members in family_members.values()
        ]
        target_uniform[f"c{cutoff}"] = {
            "fraction": round(statistics.fmean(cluster_rates), 6)
        }
    return {
        "all_sessions": _recall_view(flags, indices),
        "c200_absent_frontier": {
            "sessions": len(frontier),
            **_recall_view(flags, frontier),
        },
        "increment": {
            "count": len(increment),
            "fraction": round(len(increment) / count, 6),
            "frontier_fraction": round(len(increment) / len(frontier), 6) if frontier else 0.0,
            "target_cluster_count": len({int(family_index[index]) for index in increment}),
            "outer_fold_span": len({int(outer_fold[index]) for index in increment}),
            "taxonomy_span": len({str(taxonomy[index]) for index in increment}),
            "first_frontier": "after_full_variable_c200_through_c400",
        },
        "by_outer_fold": by_fold,
        "target_uniform": target_uniform,
        "by_taxonomy": by_taxonomy,
        "family_disjoint_audit": {
            "valid": True,
            "family_count": len(family_members),
            "families_crossing_outer_folds": 0,
        },
        "sealed_c200_surface": {
            "records": len(sealed_c200_lengths),
            "turns_per_session": len(sealed_c200_lengths) // count,
            "candidate_cells": sum(sealed_c200_lengths),
            "length": {
                "minimum": min(sealed_c200_lengths),
                "p50": int(_nearest_rank(sealed_c200_lengths, 0.50)),
                "p95": int(_nearest_rank(sealed_c200_lengths, 0.95)),
                "maximum": max(sealed_c200_lengths),
                "mean": round(statistics.fmean(sealed_c200_lengths), 6),
            },
        },
    }


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise C400ProbeError("cannot summarize empty values")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def inflation_summary(
    lengths: Sequence[int],
    c200_lengths: Sequence[int],
    *,
    trace_bytes: int,
    c100_bytes: int,
    c200_bytes: int,
) -> dict[str, Any]:
    if (
        not lengths
        or len(lengths) != len(c200_lengths)
        or any(
            not isinstance(c400, int)
            or isinstance(c400, bool)
            or not isinstance(c200, int)
            or isinstance(c200, bool)
            or not 100 <= c200 <= 200
            or not c200 <= c400 <= 400
            for c400, c200 in zip(lengths, c200_lengths, strict=True)
        )
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (trace_bytes, c100_bytes, c200_bytes)
        )
    ):
        raise C400ProbeError("candidate inflation surface is invalid")
    added = [c400 - c200 for c400, c200 in zip(lengths, c200_lengths, strict=True)]

    def describe(values: Sequence[int]) -> dict[str, int | float]:
        return {
            "minimum": min(values),
            "p50": int(_nearest_rank(values, 0.50)),
            "p95": int(_nearest_rank(values, 0.95)),
            "maximum": max(values),
            "mean": round(statistics.fmean(values), 6),
        }

    candidate_cells = sum(lengths)
    c200_cells = sum(c200_lengths)
    return {
        "c400_length": describe(lengths),
        "added_over_c200": describe(added),
        "candidate_cells": candidate_cells,
        "sealed_c200_candidate_cells": c200_cells,
        "candidate_cell_ratio_over_c100": round(
            candidate_cells / (len(lengths) * 100), 12
        ),
        "candidate_cell_ratio_over_c200": round(candidate_cells / c200_cells, 12),
        "trace_bytes": trace_bytes,
        "trace_byte_ratio_over_c100": round(trace_bytes / c100_bytes, 12),
        "trace_byte_ratio_over_c200": round(trace_bytes / c200_bytes, 12),
    }


def _write_descriptor(descriptor: int, value: object) -> tuple[int, str]:
    payload = _canonical_bytes(value) + b"\n"
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short receipt write")
        view = view[written:]
    os.fsync(descriptor)
    return len(payload), hashlib.sha256(payload).hexdigest()


def _safe_close_descriptor(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _receipt_parent() -> Path:
    if ROOT.absolute() == Path(__file__).resolve().parents[1].absolute():
        _require_lexical_ancestry(ROOT, Path(r"D:\tiktok"), "receipt repository root")
    _require_lexical_ancestry(OUTPUT_PATH.parent, ROOT, "receipt parent")
    try:
        lexical_root = ROOT.absolute()
        lexical_output = OUTPUT_PATH.absolute()
        if not _inside(lexical_output, lexical_root):
            raise C400ProbeError("receipt path escapes the repository")
        root = ROOT.resolve(strict=True)
        parent = OUTPUT_PATH.parent.resolve(strict=True)
    except OSError as error:
        raise C400ProbeError("receipt parent must already exist") from error
    if not parent.is_dir() or not _inside(parent, root) or _is_link_or_reparse(parent):
        raise C400ProbeError("receipt parent escapes the repository or is unsafe")
    return parent


def _pending_receipt(implementation_commit: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "CONSUMED_PENDING_RERUN_FORBIDDEN",
        "implementation_commit": implementation_commit,
        "rerun_forbidden": True,
        "self_hash_omitted": True,
    }


def _invalid_value(
    implementation_commit: str, error: BaseException, *, phase: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "INVALID_ONE_SHOT_CONSUMED",
        "phase": phase,
        "error_class": type(error).__name__,
        "implementation_commit": implementation_commit,
        "rerun_forbidden": True,
        "self_hash_omitted": True,
    }


def _open_receipt(implementation_commit: str) -> int:
    if not COMMIT_RE.fullmatch(implementation_commit):
        raise C400ProbeError("implementation commit is invalid")
    _receipt_parent()
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise C400ProbeError("the one-shot receipt path is already consumed")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor: int | None = None
    try:
        descriptor = os.open(str(OUTPUT_PATH), flags, 0o600)
        _write_descriptor(descriptor, _pending_receipt(implementation_commit))
    except BaseException as error:
        if descriptor is None:
            if isinstance(error, OSError):
                raise C400ProbeError("exclusive one-shot receipt creation failed") from error
            raise
        try:
            _write_descriptor(
                descriptor,
                _invalid_value(
                    implementation_commit, error, phase="pending_receipt_write"
                ),
            )
        except BaseException:
            pass
        finally:
            _safe_close_descriptor(descriptor)
        raise C400ProbeError("one-shot receipt was consumed during initialization") from error
    return descriptor


def _write_invalid_receipt(
    descriptor: int,
    implementation_commit: str,
    error: BaseException,
    *,
    phase: str,
) -> None:
    value = _invalid_value(implementation_commit, error, phase=phase)
    try:
        try:
            _write_descriptor(descriptor, value)
        except BaseException:
            try:
                _write_descriptor(descriptor, value)
            except BaseException as second_error:
                raise C400ProbeError("durable invalid receipt seal failed twice") from second_error
    finally:
        _safe_close_descriptor(descriptor)


def _prepare_cache_root() -> None:
    _require_lexical_ancestry(CACHE_ROOT.parent, ROOT, "fresh C400 cache parent")
    try:
        parent = CACHE_ROOT.parent.resolve(strict=True)
        root = ROOT.resolve(strict=True)
    except OSError as error:
        raise C400ProbeError("fresh C400 cache parent is unavailable") from error
    if not _inside(parent, root) or _is_link_or_reparse(parent):
        raise C400ProbeError("fresh C400 cache parent is unsafe")
    try:
        os.mkdir(CACHE_ROOT)
    except OSError as error:
        raise C400ProbeError("exclusive C400 cache directory creation failed") from error
    if _is_link_or_reparse(CACHE_ROOT) or not CACHE_ROOT.is_dir():
        raise C400ProbeError("fresh C400 cache directory is unsafe")


def _offline_worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def _worker_command(nonce: str, trace_path: Path) -> list[str]:
    if not NONCE_RE.fullmatch(nonce):
        raise C400ProbeError("worker nonce is invalid")
    return [
        str(EXPECTED_EXECUTABLE),
        "-B",
        str(WORKER_PATH),
        "--nonce",
        nonce,
        "--catalog",
        str(CATALOG_PATH),
        "--catalog-bytes",
        str(CATALOG_BYTES),
        "--catalog-rows",
        str(CATALOG_ROWS),
        "--catalog-sha256",
        CATALOG_SHA256,
        "--context",
        str(CONTEXT_PATH),
        "--context-bytes",
        str(CONTEXT_BYTES),
        "--context-rows",
        str(CONTEXT_ROWS),
        "--context-turns",
        str(CONTEXT_TURNS),
        "--context-sha256",
        CONTEXT_SHA256,
        "--c200-reference",
        str(C200_TRACE_PATHS[0]),
        "--c200-reference-bytes",
        str(C200_TRACE_BYTES),
        "--c200-reference-rows",
        str(C200_TRACE_ROWS),
        "--c200-reference-sha256",
        C200_TRACE_SHA256,
        "--trace-output",
        str(trace_path),
    ]


def _finite_number(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise C400ProbeError(f"{label} is not finite numeric")
    return float(value)


def _validate_latency_summary(
    value: object, *, maximum_p95: float
) -> dict[str, Any]:
    keys = {
        "count",
        "p50_milliseconds",
        "p95_milliseconds",
        "maximum_milliseconds",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise C400ProbeError("worker latency summary schema drifted")
    if (
        not isinstance(value["count"], int)
        or isinstance(value["count"], bool)
        or value["count"] != RECORD_COUNT
    ):
        raise C400ProbeError("worker latency count drifted")
    p50 = _finite_number(value["p50_milliseconds"], "worker latency p50")
    p95 = _finite_number(value["p95_milliseconds"], "worker latency p95")
    maximum = _finite_number(value["maximum_milliseconds"], "worker maximum latency")
    if not (0.0 <= p50 <= p95 <= maximum and p95 <= maximum_p95):
        raise C400ProbeError("worker latency ordering or budget failed")
    return dict(value)


def _validate_pool_summary(
    value: object,
    *,
    minimum_allowed: int,
    maximum_allowed: int,
    minimum_cells: int,
    maximum_cells: int,
) -> dict[str, Any]:
    keys = {"min", "p50", "p95", "max", "mean", "records", "candidate_cells"}
    if not isinstance(value, dict) or set(value) != keys:
        raise C400ProbeError("worker pool summary schema drifted")
    if not all(
        isinstance(value[key], int) and not isinstance(value[key], bool)
        for key in ("min", "p50", "p95", "max", "records", "candidate_cells")
    ):
        raise C400ProbeError("worker pool summary types drifted")
    mean = _finite_number(value["mean"], "worker pool mean")
    if not (
        minimum_allowed
        <= value["min"]
        <= value["p50"]
        <= value["p95"]
        <= value["max"]
        <= maximum_allowed
        and value["records"] == RECORD_COUNT
        and minimum_cells <= value["candidate_cells"] <= maximum_cells
        and float(minimum_allowed) <= mean <= float(maximum_allowed)
    ):
        raise C400ProbeError("worker pool summary values drifted")
    return dict(value)


def _validate_worker_receipt(payload: bytes, *, nonce: str) -> dict[str, Any]:
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise C400ProbeError("worker stdout is not one canonical JSON line")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                C400ProbeError(f"worker emitted non-finite {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise C400ProbeError("worker receipt JSON is invalid") from error
    if not isinstance(value, dict) or payload != _canonical_bytes(value) + b"\n":
        raise C400ProbeError("worker receipt is not canonical")
    if set(value) != {
        "kind", "nonce", "trace_sha256", "trace_bytes", "record_count", "summary"
    }:
        raise C400ProbeError("worker receipt top-level schema drifted")
    if not (
        value["kind"] == "receipt"
        and value["nonce"] == nonce
        and isinstance(value["trace_sha256"], str)
        and SHA256_RE.fullmatch(value["trace_sha256"])
        and isinstance(value["trace_bytes"], int)
        and not isinstance(value["trace_bytes"], bool)
        and C200_TRACE_BYTES <= value["trace_bytes"]
        <= int(C100_NORMALIZED_BYTES * TRACE_RATIO_C100_MAXIMUM)
        and isinstance(value["record_count"], int)
        and not isinstance(value["record_count"], bool)
        and value["record_count"] == RECORD_COUNT
    ):
        raise C400ProbeError("worker receipt identity surface drifted")
    summary = value["summary"]
    if not isinstance(summary, dict) or set(summary) != {
        "schema_version",
        "environment",
        "configuration",
        "pool_lengths",
        "expanded_broad_lengths",
        "latency",
        "resources",
        "lifecycle",
    }:
        raise C400ProbeError("worker summary schema drifted")
    if summary["schema_version"] != WORKER_SCHEMA_VERSION:
        raise C400ProbeError("worker summary version drifted")
    environment = summary["environment"]
    if not isinstance(environment, dict) or set(environment) != {
        "executable", "python", "sqlite", "numpy", "pythonhashseed",
        "network_attempt_count", "gpu_used", "gpu_peak_bytes",
    }:
        raise C400ProbeError("worker environment schema drifted")
    try:
        executable_equal = (
            Path(str(environment["executable"])).resolve(strict=True).as_posix().casefold()
            == EXPECTED_EXECUTABLE.resolve(strict=True).as_posix().casefold()
        )
    except OSError:
        executable_equal = False
    if not (
        executable_equal
        and environment["python"] == EXPECTED_PYTHON
        and environment["sqlite"] == EXPECTED_SQLITE
        and environment["numpy"] == EXPECTED_NUMPY
        and environment["pythonhashseed"] == "0"
        and isinstance(environment["network_attempt_count"], int)
        and not isinstance(environment["network_attempt_count"], bool)
        and environment["network_attempt_count"] == 0
        and environment["gpu_used"] is False
        and isinstance(environment["gpu_peak_bytes"], int)
        and not isinstance(environment["gpu_peak_bytes"], bool)
        and environment["gpu_peak_bytes"] == 0
    ):
        raise C400ProbeError("worker environment identity drifted")
    if summary["configuration"] != {
        "p11_mode": "control",
        "small_ranker_mode": "off",
        "question_policy": "fast",
        "rerank_mode": "off",
        "retrieval_mode": "coverage",
        "production_broad_limit": 120,
        "diagnostic_broad_limit": 320,
        "strict_limit": 80,
        "stable_append": True,
    }:
        raise C400ProbeError("worker Agent configuration drifted")
    _validate_pool_summary(
        summary["pool_lengths"],
        minimum_allowed=100,
        maximum_allowed=400,
        minimum_cells=C200_CANDIDATE_CELLS,
        maximum_cells=RECORD_COUNT * 400,
    )
    _validate_pool_summary(
        summary["expanded_broad_lengths"],
        minimum_allowed=0,
        maximum_allowed=320,
        minimum_cells=0,
        maximum_cells=RECORD_COUNT * 320,
    )
    latency = summary["latency"]
    if not isinstance(latency, dict) or set(latency) != {"respond", "expansion"}:
        raise C400ProbeError("worker latency container drifted")
    _validate_latency_summary(
        latency["respond"], maximum_p95=RESPOND_P95_MS_MAXIMUM
    )
    _validate_latency_summary(
        latency["expansion"], maximum_p95=EXPANSION_P95_MS_MAXIMUM
    )
    resources = summary["resources"]
    if (
        not isinstance(resources, dict)
        or set(resources) != {"peak_working_set_bytes"}
        or not isinstance(resources["peak_working_set_bytes"], int)
        or isinstance(resources["peak_working_set_bytes"], bool)
        or not 0 < resources["peak_working_set_bytes"] <= WORKER_RSS_MAXIMUM
    ):
        raise C400ProbeError("worker RSS budget failed")
    lifecycle = summary["lifecycle"]
    if not isinstance(lifecycle, dict) or set(lifecycle) != {
        "agent_closed_before_trace_publish",
        "sqlite_closed_before_trace_publish",
        "catalog_unchanged_before_trace_publish",
        "context_unchanged_before_trace_publish",
        "c200_reference_unchanged_before_trace_publish",
    } or any(value is not True for value in lifecycle.values()):
        raise C400ProbeError("worker lifecycle/source gate failed")
    _result_privacy_scan(value)
    return value


def _run_one_worker(nonce: str, trace_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            _worker_command(nonce, trace_path),
            cwd=ROOT,
            env=_offline_worker_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            check=False,
            timeout=TOTAL_WALL_MAXIMUM,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise C400ProbeError("isolated C400 worker failed to complete") from error
    wall = time.perf_counter() - started
    if completed.returncode != 0 or completed.stderr != b"":
        if ASIN_SHAPE_RE.search(completed.stderr):
            raise C400ProbeError("worker failure output contained an identifier")
        raise C400ProbeError("isolated C400 worker exited uncleanly")
    receipt = _validate_worker_receipt(completed.stdout, nonce=nonce)
    return {"receipt": receipt, "wall_seconds": round(wall, 6)}


def _pool_summary_from_lengths(lengths: Sequence[int]) -> dict[str, Any]:
    return {
        "min": min(lengths),
        "p50": int(_nearest_rank(lengths, 0.50)),
        "p95": int(_nearest_rank(lengths, 0.95)),
        "max": max(lengths),
        "mean": round(statistics.fmean(lengths), 6),
        "records": len(lengths),
        "candidate_cells": sum(lengths),
    }


def _bind_worker_receipt_to_trace(
    worker_result: Mapping[str, Any], trace: TraceValidation
) -> None:
    receipt = worker_result.get("receipt")
    if not isinstance(receipt, Mapping) or not (
        receipt.get("trace_sha256") == trace.canonical_trace_sha256
        and receipt.get("trace_bytes") == trace.canonical_trace_bytes
        and receipt.get("record_count") == trace.record_count
        and receipt.get("summary", {}).get("pool_lengths")
        == _pool_summary_from_lengths(trace.lengths)
    ):
        raise C400ProbeError("worker receipt does not bind to its closed C400 trace")


def _open_proxy_after_gates(path: Path) -> tuple[list[dict[str, Any]], FileIdentity]:
    """First proxy touch: one handle, two hashes, stable fstat and path identity."""

    resolved = _require_regular_file(path, "train_explore proxy")
    rows: list[dict[str, Any]] = []
    first_digest = hashlib.sha256()
    first_bytes = 0
    with resolved.open("rb") as handle:
        if os.get_inheritable(handle.fileno()):
            raise C400ProbeError("proxy descriptor is unexpectedly inheritable")
        before = _handle_snapshot(handle)
        for raw in handle:
            if not raw.strip():
                raise C400ProbeError("proxy contains a blank physical row")
            first_digest.update(raw)
            first_bytes += len(raw)
            try:
                value = json.loads(
                    raw.decode("utf-8", errors="strict"),
                    object_pairs_hook=_unique_object,
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                raise C400ProbeError("proxy JSONL is invalid") from error
            if not isinstance(value, dict):
                raise C400ProbeError("proxy row is not an object")
            rows.append(value)
        second_sha, second_bytes = _hash_handle(handle)
        after = _handle_snapshot(handle)
    path_after = _path_snapshot(resolved)
    identity = FileIdentity(first_bytes, len(rows), first_digest.hexdigest(), after)
    if not (
        before == after == path_after
        and first_bytes == second_bytes == before[0]
        and first_digest.hexdigest() == second_sha
        and identity.report()
        == {"bytes": PROXY_BYTES, "rows": PROXY_ROWS, "sha256": PROXY_SHA256}
    ):
        raise C400ProbeError("proxy identity drifted during same-handle verification")
    return rows, identity


def _load_fold_labels_after_traces(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    resolved = _require_regular_file(path, "numeric fold archive")
    with resolved.open("rb") as handle:
        if os.get_inheritable(handle.fileno()):
            raise C400ProbeError("label descriptor is unexpectedly inheritable")
        before = _handle_snapshot(handle)
        first_sha, first_bytes = _hash_handle(handle)
        try:
            with np.load(handle, allow_pickle=False) as archive:
                outer_fold = np.asarray(archive["outer_fold"]).copy()
                family_index = np.asarray(archive["family_index"]).copy()
        except (KeyError, OSError, ValueError) as error:
            raise C400ProbeError("numeric fold archive is invalid") from error
        second_sha, second_bytes = _hash_handle(handle)
        after = _handle_snapshot(handle)
    path_after = _path_snapshot(resolved)
    if not (
        before == after == path_after
        and before[0] == LABEL_BYTES == first_bytes == second_bytes
        and first_sha == second_sha == LABEL_SHA256
    ):
        raise C400ProbeError("numeric fold archive identity drifted")
    if (
        outer_fold.shape != (SESSION_COUNT,)
        or family_index.shape != (SESSION_COUNT,)
        or outer_fold.dtype.kind not in "iu"
        or family_index.dtype.kind not in "iu"
        or np.any(outer_fold < 0)
        or np.any(outer_fold > 4)
        or np.any(family_index < 0)
    ):
        raise C400ProbeError("fold/family arrays drifted")
    return outer_fold, family_index, {
        "bytes": LABEL_BYTES,
        "sha256": LABEL_SHA256,
        "members_read_in_order": ["outer_fold", "family_index"],
    }


def _derive_target_membership_inputs(
    samples: Sequence[Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[int], list[str]]:
    if len(samples) != SESSION_COUNT:
        raise C400ProbeError("target-attach proxy row count drifted")
    from evaluator.local_evaluator import materialize_hidden_fields
    from scripts import evaluate_p12_action_oracle as blind_oracle

    targets: list[str] = []
    eligibility: list[int] = []
    taxonomy: list[str] = []
    for sample in samples:
        ground = sample.get("ground_truth")
        target = str(ground.get("parent_asin", "")) if isinstance(ground, Mapping) else ""
        if target not in products:
            raise C400ProbeError("target-attach catalog membership drifted")
        card, behavior = materialize_hidden_fields(dict(sample), products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        try:
            eligible_turn = int(blind_oracle._eligible_from_turn(effective))
        except (AttributeError, TypeError, ValueError, blind_oracle.OracleRunError) as error:
            raise C400ProbeError("target-attach eligibility drifted") from error
        taxonomy_value = sample.get("taxonomy")
        taxonomy_name = (
            str(taxonomy_value.get("group", "unknown"))
            if isinstance(taxonomy_value, Mapping)
            else "unknown"
        )
        if not 1 <= eligible_turn <= TURN_COUNT or taxonomy_name not in EXPECTED_TAXONOMY:
            raise C400ProbeError("target-attach aggregate stratum drifted")
        targets.append(target)
        eligibility.append(eligible_turn)
        taxonomy.append(taxonomy_name)
    return targets, eligibility, taxonomy


def _flags_from_trace(
    trace: TraceValidation,
    c200_reference: Sequence[Sequence[str]],
    targets: Sequence[str],
    eligibility: Sequence[int],
) -> list[dict[int, bool]]:
    if not (
        len(trace.records) == len(c200_reference) == RECORD_COUNT
        and len(targets) == len(eligibility) == SESSION_COUNT
    ):
        raise C400ProbeError("candidate-recall target dimensions drifted")
    return [
        candidate_recall_flags(
            targets[index],
            int(eligibility[index]),
            trace.records[index * TURN_COUNT : (index + 1) * TURN_COUNT],
            c200_reference[index * TURN_COUNT : (index + 1) * TURN_COUNT],
        )
        for index in range(SESSION_COUNT)
    ]


def _rehash_target_free_sources(
    preflight: Preflight,
    *,
    include_fresh_traces: bool,
) -> dict[str, Any]:
    catalog = _file_identity(CATALOG_PATH, "catalog rehash")
    sealed_result = _file_identity(C200_RESULT_PATH, "sealed C200 result rehash")
    context = _file_identity(CONTEXT_PATH, "sealed visible context rehash")
    old_traces = [
        _file_identity(path, f"sealed C200 trace rehash {index + 1}")
        for index, path in enumerate(C200_TRACE_PATHS)
    ]
    if not (
        catalog.report() == preflight.source_identities["catalog"]
        and sealed_result.report() == preflight.source_identities["sealed_c200_result"]
        and context.report()
        == {
            "bytes": CONTEXT_BYTES,
            "rows": CONTEXT_ROWS,
            "sha256": CONTEXT_SHA256,
        }
        and [identity.report() for identity in old_traces]
        == preflight.source_identities["sealed_c200_traces"]
        and old_traces[0].report() == old_traces[1].report()
        and _files_equal(C200_TRACE_PATHS[0], C200_TRACE_PATHS[1])
    ):
        raise C400ProbeError("sealed target-free input changed during formal probe")
    worker_sha = hashlib.sha256(
        _require_regular_file(WORKER_PATH, "C400 worker rehash").read_bytes()
    ).hexdigest()
    if worker_sha != preflight.source_identities["worker_sha256"]:
        raise C400ProbeError("C400 worker source changed during formal probe")
    if _canonical_sha256(_load_preregistration()) != PREREG_CANONICAL_SHA256:
        raise C400ProbeError("C400 preregistration changed during formal probe")
    result: dict[str, Any] = {
        "catalog_sha256": catalog.sha256,
        "sealed_c200_result_sha256": sealed_result.sha256,
        "visible_context_sha256": context.sha256,
        "sealed_c200_trace_sha256": old_traces[0].sha256,
        "worker_sha256": worker_sha,
        "preregistration_canonical_sha256": PREREG_CANONICAL_SHA256,
    }
    if include_fresh_traces:
        fresh = [
            _file_identity(path, f"fresh C400 trace rehash {index + 1}")
            for index, path in enumerate(TRACE_PATHS)
        ]
        if fresh[0].report() != fresh[1].report() or not _files_equal(
            TRACE_PATHS[0], TRACE_PATHS[1]
        ):
            raise C400ProbeError("fresh C400 traces changed or diverged")
        result["fresh_c400_traces"] = [identity.report() for identity in fresh]
    return result


def run(implementation_commit: str) -> dict[str, Any]:
    """Consume the unique formal C400 probe and write its aggregate receipt."""

    formal_started = time.perf_counter()
    preflight = preflight_only(implementation_commit)
    descriptor: int | None = None
    phase = "receipt_initialization"
    try:
        descriptor = _open_receipt(implementation_commit)
        phase = "fresh_cache_initialization"
        _prepare_cache_root()

        phase = "simultaneous_target_blind_c400_workers"
        nonces = {
            "replica_a": hashlib.sha256(
                f"{EXPERIMENT_ID}:{implementation_commit}:replica_a".encode("utf-8")
            ).hexdigest()[:32],
            "replica_b": hashlib.sha256(
                f"{EXPERIMENT_ID}:{implementation_commit}:replica_b".encode("utf-8")
            ).hexdigest()[:32],
        }
        worker_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(_run_one_worker, nonces[name], path): name
                for name, path in zip(
                    ("replica_a", "replica_b"), TRACE_PATHS, strict=True
                )
            }
            try:
                for future in as_completed(futures):
                    worker_results[futures[future]] = future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
        if set(worker_results) != {"replica_a", "replica_b"}:
            raise C400ProbeError("one isolated C400 worker result is missing")

        phase = "closed_c400_trace_and_full_c200_prefix_gates"
        trace_a = _load_and_validate_c400_trace_streaming(
            TRACE_PATHS[0],
            preflight.c200_reference.candidates,
            preflight.catalog_ids,
            retain_records=True,
        )
        trace_b = _load_and_validate_c400_trace_streaming(
            TRACE_PATHS[1],
            preflight.c200_reference.candidates,
            preflight.catalog_ids,
            retain_records=False,
        )
        _bind_worker_receipt_to_trace(worker_results["replica_a"], trace_a)
        _bind_worker_receipt_to_trace(worker_results["replica_b"], trace_b)
        for trace in (trace_a, trace_b):
            if not (
                trace.normalized_c200_sha256 == C200_TRACE_SHA256
                and trace.normalized_c200_bytes == C200_TRACE_BYTES
                and trace.c200_lengths == preflight.c200_reference.lengths
                and trace.record_count == RECORD_COUNT
            ):
                raise C400ProbeError("fresh trace does not preserve full sealed C200")
        if not (
            trace_a.canonical_trace_sha256 == trace_b.canonical_trace_sha256
            and trace_a.canonical_trace_bytes == trace_b.canonical_trace_bytes
            and trace_a.lengths == trace_b.lengths
            and trace_a.c200_lengths == trace_b.c200_lengths
            and _files_equal(TRACE_PATHS[0], TRACE_PATHS[1])
        ):
            raise C400ProbeError("two fresh C400 traces are not exact repeats")

        phase = "broad_top10_resource_and_exact_repeat_gates_before_target_attach"
        inflation_a = inflation_summary(
            trace_a.lengths,
            trace_a.c200_lengths,
            trace_bytes=trace_a.canonical_trace_bytes,
            c100_bytes=C100_NORMALIZED_BYTES,
            c200_bytes=C200_TRACE_BYTES,
        )
        inflation_b = inflation_summary(
            trace_b.lengths,
            trace_b.c200_lengths,
            trace_bytes=trace_b.canonical_trace_bytes,
            c100_bytes=C100_NORMALIZED_BYTES,
            c200_bytes=C200_TRACE_BYTES,
        )
        if inflation_a != inflation_b:
            raise C400ProbeError("C400 inflation is not an exact repeat")
        if not (
            inflation_a["sealed_c200_candidate_cells"] == C200_CANDIDATE_CELLS
            and inflation_a["candidate_cell_ratio_over_c100"]
            <= CELL_RATIO_C100_MAXIMUM
            and inflation_a["candidate_cell_ratio_over_c200"]
            <= CELL_RATIO_C200_MAXIMUM
            and inflation_a["trace_byte_ratio_over_c100"]
            <= TRACE_RATIO_C100_MAXIMUM
            and inflation_a["trace_byte_ratio_over_c200"]
            <= TRACE_RATIO_C200_MAXIMUM
        ):
            raise C400ProbeError("C400 candidate inflation budget failed")
        worker_rss = {
            name: int(value["receipt"]["summary"]["resources"]["peak_working_set_bytes"])
            for name, value in worker_results.items()
        }
        if sum(worker_rss.values()) > WORKER_RSS_SUM_MAXIMUM:
            raise C400ProbeError("conservative two-worker RSS sum exceeded budget")
        if time.perf_counter() - formal_started > TOTAL_WALL_MAXIMUM:
            raise C400ProbeError("formal C400 probe exceeded wall budget before target attach")
        pre_target_rehash = _rehash_target_free_sources(
            preflight, include_fresh_traces=True
        )
        expected_fresh_trace = {
            "bytes": trace_a.canonical_trace_bytes,
            "rows": trace_a.record_count,
            "sha256": trace_a.canonical_trace_sha256,
        }
        if any(
            identity != expected_fresh_trace
            for identity in pre_target_rehash["fresh_c400_traces"]
        ):
            raise C400ProbeError("fresh C400 trace changed before target attach")
        pre_target_git = _validate_git_checkpoint(implementation_commit)

        # This is deliberately the first reference to either protected join source
        # in the formal function: every trace, prefix, worker, Top10, broad-route,
        # resource, exact-repeat, source, and Git gate has already passed.
        phase = "evaluator_side_proxy_then_numeric_fold_attach"
        samples, proxy_identity = _open_proxy_after_gates(PROXY_PATH)
        outer_fold, family_index, label_identity = _load_fold_labels_after_traces(
            LABEL_PATH
        )
        targets, eligibility, taxonomy = _derive_target_membership_inputs(
            samples, preflight.products
        )
        flags = _flags_from_trace(
            trace_a,
            preflight.c200_reference.candidates,
            targets,
            eligibility,
        )
        del samples, targets, eligibility
        aggregate = aggregate_candidate_recall(
            flags,
            outer_fold=outer_fold,
            family_index=family_index,
            taxonomy=taxonomy,
            sealed_c200_lengths=preflight.c200_reference.lengths,
        )
        sanity = aggregate["all_sessions"]
        if any(
            sanity[f"c{cutoff}"]["count"] != expected
            for cutoff, expected in EXPECTED_C200_RECALL.items()
        ) or not (
            aggregate["c200_absent_frontier"]["sessions"] == EXPECTED_C200_FRONTIER
            and aggregate["sealed_c200_surface"]["records"] == RECORD_COUNT
            and aggregate["sealed_c200_surface"]["turns_per_session"] == TURN_COUNT
            and aggregate["sealed_c200_surface"]["candidate_cells"]
            == C200_CANDIDATE_CELLS
            and sanity["c400"]["count"] >= sanity["c200"]["count"]
            and aggregate["increment"]["count"]
            == sanity["c400"]["count"] - sanity["c200"]["count"]
        ):
            raise C400ProbeError("sealed C200 sanity or C400 monotonic recall drifted")

        phase = "final_source_git_resource_and_privacy_gates"
        del trace_b
        final_source_rehash = _rehash_target_free_sources(
            preflight, include_fresh_traces=True
        )
        if any(
            identity != expected_fresh_trace
            for identity in final_source_rehash["fresh_c400_traces"]
        ):
            raise C400ProbeError("fresh C400 trace changed before result seal")
        final_git = _validate_git_checkpoint(implementation_commit)
        if pre_target_git != final_git:
            raise C400ProbeError("Git identity changed after target attach")
        total_wall = time.perf_counter() - formal_started
        if total_wall > TOTAL_WALL_MAXIMUM:
            raise C400ProbeError("formal C400 probe exceeded total wall budget")
        parent_current_rss, parent_peak_rss = _process_memory()
        if not 0 < parent_current_rss <= parent_peak_rss:
            raise C400ProbeError("final parent working-set measurement is unavailable")

        activation_turns = sum(
            int(new > old)
            for new, old in zip(trace_a.lengths, trace_a.c200_lengths, strict=True)
        )
        activation_sessions = sum(
            int(
                any(
                    trace_a.lengths[index * TURN_COUNT + offset]
                    > trace_a.c200_lengths[index * TURN_COUNT + offset]
                    for offset in range(TURN_COUNT)
                )
            )
            for index in range(SESSION_COUNT)
        )
        increment = aggregate["increment"]
        recall_go = bool(increment["count"] > 0)
        family_members: dict[int, list[int]] = {}
        for session_index, family in enumerate(family_index):
            family_members.setdefault(int(family), []).append(session_index)
        exact_target_uniform_c200 = statistics.fmean(
            statistics.fmean(int(flags[index][200]) for index in members)
            for members in family_members.values()
        )
        exact_target_uniform_c400 = statistics.fmean(
            statistics.fmean(int(flags[index][400]) for index in members)
            for members in family_members.values()
        )
        target_uniform_gain = exact_target_uniform_c400 > exact_target_uniform_c200
        training_evidence_gate = bool(
            recall_go
            and target_uniform_gain
            and increment["target_cluster_count"] > 0
            and increment["outer_fold_span"] >= 2
        )
        status = (
            "C400_RECALL_GO_CONTINUE_FROZEN_EMBEDDING_E0"
            if recall_go
            else "C400_RECALL_NO_GO_CONTINUE_FROZEN_EMBEDDING_E0"
        )

        phase = "aggregate_result_seal"
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "recorded_on": "2026-08-31",
            "rerun_forbidden": True,
            "self_hash_omitted": True,
            "evidence_scope": (
                "shared-cohort 2000-session diagnostic candidate recall; "
                "outer-fold views are not private or independent validation"
            ),
            "implementation": {
                "commit": implementation_commit,
                "preregistration_commit": PREREG_COMMIT,
                "base_c200_result_commit": BASE_COMMIT,
                "branch": BRANCH,
                "default": "off",
                "causal": True,
                "runtime_target_blind": True,
                "full_agent_evaluator_started": False,
                "protected_splits_opened": False,
                "fit_or_selection_performed": False,
                "c200_rerun_performed": False,
                "visible_context_rebuilt": False,
            },
            "preregistration": dict(preflight.protocol),
            "environment": {
                "parent": dict(preflight.environment),
                "workers": {
                    name: value["receipt"]["summary"]["environment"]
                    for name, value in worker_results.items()
                },
            },
            "global_incumbent": {
                "commit": "7d08e144fe6f589d64fab21f1b68c3ec2feeb684",
                "hr_at_10": 0.991,
                "mrr": 0.695795,
                "mttc": 2.869,
                "technical_score": 0.866858,
                "comparable": True,
            },
            "top10_metrics": {
                "baseline_hr_at_10": None,
                "candidate_hr_at_10": None,
                "miss_to_hit": None,
                "hit_to_miss": None,
                "net": None,
                "mrr": None,
                "mttc": None,
                "technical_score": None,
                "outer_folds": None,
                "reason": "candidate-recall-only probe; served Top10 was unchanged",
            },
            "candidate_recall": aggregate,
            "candidate_retention": {
                "full_variable_c200_is_exact_ordered_prefix": True,
                "old_candidate_loss_count": 0,
                "candidate_level_hit_to_miss": 0,
                "sealed_c200_trace_sha256": C200_TRACE_SHA256,
                "sealed_c200_trace_bytes": C200_TRACE_BYTES,
                "sealed_c200_candidate_cells": C200_CANDIDATE_CELLS,
            },
            "mechanism_identity": {
                "single_changed_variable": "broad_bm25_limit_120_to_320",
                "expanded_broad_first_120_equals_production": True,
                "strict_limit_80_route_reused_without_requery": True,
                "production_c200_equals_sealed_reference": True,
                "served_top10_equals_sealed_c200_top10": True,
                "production_rankings_and_rowids_unmodified": True,
                "diagnostic_candidates_not_served": True,
            },
            "activation": {
                "sessions_with_candidates_after_c200": activation_sessions,
                "turns_with_candidates_after_c200": activation_turns,
            },
            "inflation": inflation_a,
            "exact_repeat": {
                "passed": True,
                "trace_sha256": trace_a.canonical_trace_sha256,
                "trace_bytes": trace_a.canonical_trace_bytes,
                "record_count": trace_a.record_count,
                "candidate_arrays_equal": True,
                "recall_aggregates_equal": True,
                "inflation_aggregates_equal": True,
            },
            "resources": {
                "total_wall_seconds": round(total_wall, 6),
                "worker_wall_seconds": {
                    name: value["wall_seconds"] for name, value in worker_results.items()
                },
                "worker_peak_working_set_bytes": worker_rss,
                "conservative_worker_peak_sum_bytes": sum(worker_rss.values()),
                "parent_current_working_set_bytes": parent_current_rss,
                "parent_lifetime_peak_working_set_bytes": parent_peak_rss,
                "parent_before_receipt_current_working_set_bytes": preflight.memory_before_receipt[0],
                "parent_before_receipt_peak_working_set_bytes": preflight.memory_before_receipt[1],
                "worker_latency": {
                    name: value["receipt"]["summary"]["latency"]
                    for name, value in worker_results.items()
                },
                "gpu_peak_bytes": 0,
                "network_attempt_count": 0,
                "budgets_passed": True,
            },
            "source_identities": {
                **preflight.source_identities,
                "proxy": proxy_identity.report(),
                "numeric_fold_archive": label_identity,
                "fresh_c400_trace_sha256": trace_a.canonical_trace_sha256,
                "pre_target_rehash": pre_target_rehash,
                "final_rehash": final_source_rehash,
            },
            "git": final_git,
            "decision": {
                "recall_go": recall_go,
                "training_evidence_gate_passed": training_evidence_gate,
                "top10_global_promotion": False,
                "next_stage": (
                    "local-model inventory and Frozen embedding E0, then deterministic GLoSS G0"
                ),
                "fallback_order": [
                    "SR-V2.12-FIXED-TWO-PAGE-GRACE", "v1.9", "P11", "R08"
                ],
            },
        }
        _result_privacy_scan(result, catalog_ids=preflight.catalog_ids)
        if descriptor is None:
            raise C400ProbeError("receipt descriptor closed before aggregate seal")
        _write_descriptor(descriptor, result)
        sealed_descriptor = descriptor
        descriptor = None
        _safe_close_descriptor(sealed_descriptor)
        return result
    except BaseException as error:
        invalid_seal_error: BaseException | None = None
        if descriptor is not None:
            try:
                _write_invalid_receipt(
                    descriptor, implementation_commit, error, phase=phase
                )
            except BaseException as seal_error:
                invalid_seal_error = seal_error
            finally:
                descriptor = None
        if invalid_seal_error is not None:
            raise C400ProbeError("formal failure receipt could not be sealed") from invalid_seal_error
        if isinstance(error, C400ProbeError):
            raise
        raise C400ProbeError("formal C400 one-shot failed") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.preflight_only:
        checked = preflight_only(arguments.implementation_commit)
        summary = {
            "status": "TARGET_FREE_PREFLIGHT_PASS",
            "commit": checked.git["commit"],
            "catalog_sha256": checked.source_identities["catalog"]["sha256"],
            "c200_trace_sha256": checked.source_identities["sealed_c200_traces"][0]["sha256"],
            "receipt_created": False,
        }
    else:
        outcome = run(arguments.implementation_commit)
        summary = {
            "status": outcome["status"],
            "commit": outcome["implementation"]["commit"],
            "c200_count": outcome["candidate_recall"]["all_sessions"]["c200"]["count"],
            "c400_count": outcome["candidate_recall"]["all_sessions"]["c400"]["count"],
            "increment_count": outcome["candidate_recall"]["increment"]["count"],
            "exact_repeat": outcome["exact_repeat"]["passed"],
        }
    sys.stdout.buffer.write(_canonical_bytes(summary) + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
