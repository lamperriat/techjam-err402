"""One-shot, target-blind C200 candidate-recall probe.

The formal run first creates a durable exclusive receipt.  It then rebuilds
the already-frozen identifier-free visible-context cache, runs two fresh
isolated workers, and proves that every generated C100 is byte-identical to
the historical blind C100 before any target or fold is joined.  The output is
aggregate-only and is diagnostic shared-cohort evidence, never a private score.
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


CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
)
from scripts import build_small_ranker_cache as context_builder  # noqa: E402
from scripts import c200_candidate_worker as worker_contract  # noqa: E402
from scripts import evaluate_p12_action_oracle as blind_oracle  # noqa: E402
from starter.agent import Agent, SessionState  # noqa: E402


SCHEMA_VERSION = "small-ranker-c200-candidate-recall-outcome.v1"
EXPERIMENT_ID = "SR-V2.16-C200-CANDIDATE-RECALL"
BRANCH = "small-ranker-v2.16-c200-recall"
REMOTE = "origin"
REMOTE_URL = "https://github.com/lamperriat/techjam-err402.git"
REMOTE_REF = "refs/remotes/origin/" + BRANCH
ORIGINAL_PREREG_COMMIT = "27decc46bc3752b5c7654c8c90a139ecac0ac78d"
BASE_COMMIT = "22ebdbe2016a46750cb82092279ab5055ab02252"
PREREG_COMMIT = "e412300fd12b36790eaee6ec81cab50fba60ba99"
PREREG_BLOB = "f911e483dcdc3d86e0def38e231260a832baa379"
PREREG_CANONICAL_SHA256 = (
    "1048c305cbef5a5c03671338862848c4a2addd6b29c1971e8ec8b45a3c8b9b09"
)
PREREG_PATH = ROOT / "configs/small_ranker_v2_16.c200_candidate_recall_preflight_erratum.json"
PREREG_PATHS = {
    "configs/small_ranker_v2_16.c200_candidate_recall_preflight_erratum.json"
}
IMPLEMENTATION_PATHS = {
    "scripts/c200_candidate_worker.py",
    "scripts/probe_c200_candidate_recall.py",
    "tests/test_c200_candidate_recall.py",
}
PINNED_BLOBS = {
    "starter/agent.py": "421c6d43c598102b8fefb181b72bab5da4bf1294",
    "starter/coverage.py": "59a6507fef63afa0d9761323f5771a52741c811a",
    "scripts/build_small_ranker_cache.py": "8724f052155ae7405ecd675bdebd83df18685895",
    "evaluator/local_evaluator.py": "7c808347b31ef3121a9cbc4810ac3eb325f950ba",
    "configs/small_ranker_v1.cache.manifest.json": "0da6d6268c2042d9854f95c4227a224bfa6342da",
}

SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
PROXY_PATH = SOURCE_ROOT / "experiments/fast_track/proxy_v1/proxy_train_explore.jsonl"
PROXY_BYTES = 1_315_338
PROXY_ROWS = 2_000
PROXY_SHA256 = "2175696171c0d874fca4b9aa456ff5fd7d570f2184f59ade6781198f6443198e"
CATALOG_PATH = SOURCE_ROOT / "data/catalog.jsonl"
CATALOG_BYTES = 60_546_327
CATALOG_ROWS = 50_000
CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
LABEL_PATH = SOURCE_ROOT / "experiments/fast_track/small_ranker_v1/labels_v2.npz"
LABEL_BYTES = 1_702_876
LABEL_SHA256 = "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb"
BLIND_ROOT = SOURCE_ROOT / "experiments/fast_track/action_oracle_v1"
BLIND_NAMES = tuple(
    f"train_explore-full-blind-shard-{index:02d}-of-04.jsonl"
    for index in range(1, 5)
)
BLIND_SHA256 = (
    "fac3bc71e6210d1a449de706d335cc5bb945d4d3daf01e8cbecbe15c0600bf1a",
    "63812776b374fc0041871600a5781fbf1ea6046a3219334e7263338abbab6657",
    "36a8706a2f8c51635e4feb4cde905a9789c7953ffeab25ae036ef824061f36b3",
    "1f9968795ab5490968badcf82c39ec11bedd00f22797569dfec8c2ff3fb7ed99",
)
BLIND_SHARD_ROWS = 5_000
REFERENCE_C100_RECORDS = 20_000
REFERENCE_C100_BYTES = 26_690_930
REFERENCE_C100_SHA256 = "b22b035cb7789570f36db6c52256e5deb67f593f90cbbc5c334d48f2f0a01a67"

CONTEXT_BYTES = 47_168_882
CONTEXT_ROWS = 2_000
CONTEXT_TURNS = 20_000
CONTEXT_SHA256 = "f30a98700da5d480731fe7e82c87c40a22f06de290e069e20dc68f9fefecd20f"
CONTEXT_REDACTED_MESSAGES = 8
SESSION_COUNT = 2_000
TURN_COUNT = 10
CUTOFFS = (10, 20, 50, 100, 200)
EXPECTED_CANDIDATE_RECALL = {10: 1_895, 20: 1_943, 50: 1_982, 100: 1_986}
EXPECTED_C100_MISSES = 14
EXPECTED_TAXONOMY = frozenset(
    {"accessories-other", "clothing", "jewelry", "shoes"}
)

EXPECTED_EXECUTABLE = Path(r"D:\450\conda\envs\tiktok\python.exe")
EXPECTED_PYTHON = "3.11.16"
EXPECTED_SQLITE = "3.53.4"
EXPECTED_NUMPY = "2.4.6"
CONTEXT_WALL_MAXIMUM = 120.0
TOTAL_WALL_MAXIMUM = 1_800.0
RESPOND_P95_MS_MAXIMUM = 250.0
CAPTURE_P95_US_MAXIMUM = 5_000.0
WORKER_RSS_MAXIMUM = 2_147_483_648
WORKER_RSS_SUM_MAXIMUM = 4_294_967_296
CELL_RATIO_MAXIMUM = 2.0
TRACE_RATIO_MAXIMUM = 2.1

OUTPUT_PATH = ROOT / "experiments/fast_track/small_ranker_v2_16_c200_candidate_recall_20260831.json"
CACHE_ROOT = ROOT / "experiments/fast_track/c200_candidate_recall_cache_20260831"
CONTEXT_PATH = CACHE_ROOT / "visible_context.jsonl"
TRACE_PATHS = (CACHE_ROOT / "replica_a.jsonl", CACHE_ROOT / "replica_b.jsonl")
WORKER_PATH = ROOT / "scripts/c200_candidate_worker.py"

ASIN_SHAPE_RE = re.compile(
    rb"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE
)
CATALOG_IDENTIFIER_RE = re.compile(rb"[A-Z0-9]{10}")
CATALOG_IDENTIFIER_TOKEN_RE = re.compile(
    rb"(?<![A-Z0-9])[A-Z0-9]{10}(?![A-Z0-9])", re.IGNORECASE
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class C200ProbeError(RuntimeError):
    """Raised when the C200 evidence boundary cannot be trusted."""


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
    canonical_trace_sha256: str
    canonical_trace_bytes: int
    normalized_c100_sha256: str
    normalized_c100_bytes: int


@dataclass(frozen=True)
class Preflight:
    environment: Mapping[str, Any]
    git: Mapping[str, Any]
    protocol: Mapping[str, Any]
    catalog_ids: frozenset[str]
    products: Mapping[str, Mapping[str, Any]]
    categories: Mapping[str, list[str]]
    frozen_c100: tuple[tuple[str, ...], ...]
    source_identities: Mapping[str, Any]
    memory_before_receipt: tuple[int, int]


@dataclass
class OpenProxySource:
    path: Path
    handle: BinaryIO
    samples: list[dict[str, Any]]
    identity: FileIdentity


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
            raise C200ProbeError("duplicate JSON key")
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
        raise C200ProbeError("path identity is unavailable") from error
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
        raise C200ProbeError(f"{label} escaped its lexical anchor")
    current = lexical_anchor
    components = (current,)
    for part in lexical_path.relative_to(lexical_anchor).parts:
        current = current / part
        components += (current,)
    for component in components:
        if component.exists() or component.is_symlink():
            if _is_link_or_reparse(component):
                raise C200ProbeError(f"{label} traverses a link or reparse point")


def _require_regular_file(path: Path, label: str) -> Path:
    workspace_path = Path(r"D:\tiktok")
    _require_lexical_ancestry(path, workspace_path, label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise C200ProbeError(f"{label} is unavailable") from error
    workspace = workspace_path.resolve(strict=True)
    if not _inside(resolved, workspace) or not resolved.is_file():
        raise C200ProbeError(f"{label} escaped the workspace or is not a file")
    return resolved


def _require_plain_regular_file(path: Path, label: str) -> Path:
    """Require a real regular file without imposing the formal workspace root.

    Public validation helpers are intentionally usable with synthetic pytest
    fixtures under the operating-system temporary directory.  Formal source
    paths still go through ``_require_regular_file`` and its D:\\tiktok gate.
    """

    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise C200ProbeError(f"{label} is unavailable") from error
    if not resolved.is_file() or _is_link_or_reparse(path):
        raise C200ProbeError(f"{label} is not a plain regular file")
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
        raise C200ProbeError(f"{label} changed while hashed")
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
        raise C200ProbeError("Git identity command failed")
    return completed.stdout.strip()


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
        raise C200ProbeError("formal environment identity drifted")
    return observed


def _load_preregistration() -> dict[str, Any]:
    path = _require_regular_file(PREREG_PATH, "preregistration")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or not (
        value.get("schema_version")
        == "small-ranker-c200-candidate-recall-preflight-erratum.v1"
        and value.get("status")
        == "PREREGISTERED_AFTER_TARGET_FREE_PREFLIGHT_FAILURE_BEFORE_CORRECTION_AND_OUTCOME"
        and value.get("parent_implementation_commit") == BASE_COMMIT
        and value.get("original_preregistration_commit") == ORIGINAL_PREREG_COMMIT
        and _canonical_sha256(value) == PREREG_CANONICAL_SHA256
        and _git("rev-parse", f"{PREREG_COMMIT}:configs/small_ranker_v2_16.c200_candidate_recall_preflight_erratum.json")
        == PREREG_BLOB
    ):
        raise C200ProbeError("preregistration identity drifted")
    return value


def _changed_paths(commitish: str) -> set[str]:
    output = _git("diff-tree", "--no-commit-id", "--name-only", "-r", commitish)
    return {line for line in output.splitlines() if line}


def _diff_paths(left: str, right: str) -> set[str]:
    output = _git("diff", "--name-only", left, right)
    return {line for line in output.splitlines() if line}


def _validate_git_checkpoint(implementation_commit: str) -> dict[str, Any]:
    if not COMMIT_RE.fullmatch(implementation_commit):
        raise C200ProbeError("implementation commit is invalid")
    head = _git("rev-parse", "HEAD")
    parent = _git("rev-parse", "HEAD^")
    prereg_parent = _git("rev-parse", f"{PREREG_COMMIT}^")
    branch = _git("branch", "--show-current")
    remote_url = _git("remote", "get-url", REMOTE)
    remote_head = _git("rev-parse", REMOTE_REF)
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    pinned = {path: _git("rev-parse", f"HEAD:{path}") for path in PINNED_BLOBS}
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
    ):
        raise C200ProbeError("Git checkpoint gate failed")
    implementation_blobs = {
        path: _git("rev-parse", f"HEAD:{path}") for path in sorted(IMPLEMENTATION_PATHS)
    }
    return {
        "branch": branch,
        "commit": head,
        "parent": parent,
        "preregistration_commit": PREREG_COMMIT,
        "remote_equal": True,
        "clean_including_untracked_nonignored": True,
        "exact_changed_paths": True,
        "pinned_blobs": pinned,
        "implementation_blobs": implementation_blobs,
    }


def _load_catalog_target_free() -> tuple[
    frozenset[str], dict[str, Mapping[str, Any]], dict[str, list[str]], FileIdentity
]:
    identity = _file_identity(CATALOG_PATH, "catalog")
    if identity.report() != {
        "bytes": CATALOG_BYTES,
        "rows": CATALOG_ROWS,
        "sha256": CATALOG_SHA256,
    }:
        raise C200ProbeError("catalog identity drifted")
    products: dict[str, Mapping[str, Any]] = {}
    categories: dict[str, list[str]] = {}
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
                raise C200ProbeError("catalog identifier surface drifted")
            products[identifier] = value
            categories[identifier] = [str(item) for item in value.get("categories") or ()]
    if len(products) != CATALOG_ROWS:
        raise C200ProbeError("catalog row count drifted")
    return frozenset(products), products, categories, identity


def _normalized_c100_line(ordinal: int, turn: int, c100: Sequence[str]) -> bytes:
    return _canonical_bytes(
        {"c100": list(c100), "ordinal": ordinal, "turn": turn}
    ) + b"\n"


def _load_frozen_c100() -> tuple[tuple[tuple[str, ...], ...], dict[str, Any]]:
    pools: list[tuple[str, ...]] = []
    normalized = hashlib.sha256()
    normalized_bytes = 0
    identities: list[dict[str, Any]] = []
    for shard_index, (name, expected_sha) in enumerate(
        zip(BLIND_NAMES, BLIND_SHA256, strict=True)
    ):
        path = BLIND_ROOT / name
        identity = _file_identity(path, f"frozen blind shard {shard_index + 1}")
        if identity.rows != BLIND_SHARD_ROWS or identity.sha256 != expected_sha:
            raise C200ProbeError("frozen blind shard identity drifted")
        identities.append({"name": name, **identity.report()})
        with path.open("r", encoding="utf-8", newline="") as handle:
            for local_index, line in enumerate(handle):
                row = json.loads(line, object_pairs_hook=_unique_object)
                local_ordinal = local_index // TURN_COUNT + 1
                turn = local_index % TURN_COUNT + 1
                if not isinstance(row, dict) or (
                    row.get("ordinal") != local_ordinal or row.get("turn") != turn
                ):
                    raise C200ProbeError("frozen C100 row order drifted")
                candidate_pools = row.get("candidate_pools")
                c100_raw = candidate_pools.get("c100") if isinstance(candidate_pools, dict) else None
                if not isinstance(c100_raw, list) or len(c100_raw) != 100:
                    raise C200ProbeError("frozen C100 pool shape drifted")
                c100 = tuple(c100_raw)
                if len(set(c100)) != 100 or any(not isinstance(item, str) or not item for item in c100):
                    raise C200ProbeError("frozen C100 pool is invalid")
                global_ordinal = shard_index * 500 + local_ordinal
                payload = _normalized_c100_line(global_ordinal, turn, c100)
                normalized.update(payload)
                normalized_bytes += len(payload)
                pools.append(c100)
    if not (
        len(pools) == REFERENCE_C100_RECORDS
        and normalized_bytes == REFERENCE_C100_BYTES
        and normalized.hexdigest() == REFERENCE_C100_SHA256
    ):
        raise C200ProbeError("normalized frozen C100 identity drifted")
    return tuple(pools), {
        "records": len(pools),
        "normalized_bytes": normalized_bytes,
        "normalized_sha256": normalized.hexdigest(),
        "shards": identities,
    }


def _process_memory() -> tuple[int, int]:
    if os.name != "nt":
        return 0, 0
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
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
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
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
    """Validate only target-free identities; outcome paths are not touched."""

    environment = _validate_environment()
    protocol_value = _load_preregistration()
    git = _validate_git_checkpoint(implementation_commit)
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink() or CACHE_ROOT.exists() or CACHE_ROOT.is_symlink():
        raise C200ProbeError("formal receipt or cache path already exists")
    catalog_ids, products, categories, catalog_identity = _load_catalog_target_free()
    frozen_c100, blind_identity = _load_frozen_c100()
    worker_path = _require_regular_file(WORKER_PATH, "C200 worker")
    source_identities = {
        "catalog": catalog_identity.report(),
        "blind_c100": blind_identity,
        "worker_sha256": hashlib.sha256(worker_path.read_bytes()).hexdigest(),
    }
    memory_before_receipt = _process_memory()
    if not 0 < memory_before_receipt[0] <= memory_before_receipt[1]:
        raise C200ProbeError("parent working-set measurement is unavailable")
    return Preflight(
        environment=environment,
        git=git,
        protocol={
            "commit": PREREG_COMMIT,
            "git_blob_oid": PREREG_BLOB,
            "canonical_sha256": PREREG_CANONICAL_SHA256,
            "schema_version": protocol_value["schema_version"],
        },
        catalog_ids=catalog_ids,
        products=products,
        categories=categories,
        frozen_c100=frozen_c100,
        source_identities=source_identities,
        memory_before_receipt=memory_before_receipt,
    )


def validate_trace_records(
    records: Sequence[Mapping[str, Any]],
    frozen_c100: Sequence[Sequence[str]],
    catalog_ids: Iterable[str],
    *,
    expected_records: int,
) -> TraceValidation:
    if (
        not isinstance(expected_records, int)
        or isinstance(expected_records, bool)
        or expected_records <= 0
        or len(records) != expected_records
        or len(frozen_c100) != expected_records
    ):
        raise C200ProbeError("C200 trace record count drifted")
    catalog = catalog_ids if isinstance(catalog_ids, (set, frozenset, dict)) else frozenset(catalog_ids)
    canonical = hashlib.sha256()
    c100_digest = hashlib.sha256()
    canonical_bytes = c100_bytes = 0
    normalized_records: list[dict[str, Any]] = []
    lengths: list[int] = []
    for index, (record, reference) in enumerate(zip(records, frozen_c100, strict=True)):
        ordinal = index // TURN_COUNT + 1
        turn = index % TURN_COUNT + 1
        if (
            not isinstance(record, Mapping)
            or set(record) != {"c200", "ordinal", "turn"}
            or not isinstance(record.get("ordinal"), int)
            or record.get("ordinal") != ordinal
            or isinstance(record.get("ordinal"), bool)
            or not isinstance(record.get("turn"), int)
            or record.get("turn") != turn
            or isinstance(record.get("turn"), bool)
        ):
            raise C200ProbeError("C200 trace schema or order drifted")
        try:
            c200 = worker_contract.validate_c200(record.get("c200"), catalog)
        except worker_contract.C200WorkerError as error:
            raise C200ProbeError("C200 trace candidate surface is invalid") from error
        reference_tuple = tuple(reference)
        if len(reference_tuple) != 100 or c200[:100] != reference_tuple:
            raise C200ProbeError("generated C100 differs from frozen C100")
        trace_line = worker_contract.canonical_trace_line(ordinal, turn, c200)
        c100_line = _normalized_c100_line(ordinal, turn, reference_tuple)
        canonical.update(trace_line)
        c100_digest.update(c100_line)
        canonical_bytes += len(trace_line)
        c100_bytes += len(c100_line)
        normalized_records.append({"c200": c200, "ordinal": ordinal, "turn": turn})
        lengths.append(len(c200))
    return TraceValidation(
        records=tuple(normalized_records),
        lengths=tuple(lengths),
        canonical_trace_sha256=canonical.hexdigest(),
        canonical_trace_bytes=canonical_bytes,
        normalized_c100_sha256=c100_digest.hexdigest(),
        normalized_c100_bytes=c100_bytes,
    )


def load_and_validate_c200_trace(
    path: Path,
    frozen_c100: Sequence[Sequence[str]],
    catalog_ids: Iterable[str],
    *,
    expected_records: int = REFERENCE_C100_RECORDS,
) -> TraceValidation:
    workspace = Path(r"D:\tiktok").absolute()
    resolved = (
        _require_regular_file(path, "C200 trace")
        if _inside(path.absolute(), workspace)
        else _require_plain_regular_file(path, "C200 trace")
    )
    records: list[dict[str, Any]] = []
    raw_digest = hashlib.sha256()
    raw_bytes = 0
    with resolved.open("rb") as handle:
        for line in handle:
            if not line.strip():
                raise C200ProbeError("C200 trace contains a blank row")
            try:
                value = json.loads(
                    line.decode("utf-8", errors="strict"),
                    object_pairs_hook=_unique_object,
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                raise C200ProbeError("C200 trace JSONL is invalid") from error
            if not isinstance(value, dict):
                raise C200ProbeError("C200 trace row is not an object")
            try:
                canonical_line = worker_contract.canonical_trace_line(
                    value.get("ordinal"), value.get("turn"), value.get("c200")
                )
            except worker_contract.C200WorkerError as error:
                raise C200ProbeError("C200 trace row failed canonical validation") from error
            if line != canonical_line:
                raise C200ProbeError("C200 trace row is not canonical LF JSON")
            raw_digest.update(line)
            raw_bytes += len(line)
            records.append(value)
    validation = validate_trace_records(
        records,
        frozen_c100,
        catalog_ids,
        expected_records=expected_records,
    )
    if (
        raw_digest.hexdigest() != validation.canonical_trace_sha256
        or raw_bytes != validation.canonical_trace_bytes
    ):
        raise C200ProbeError("C200 trace raw/canonical identity differs")
    return validation


def candidate_recall_flags(
    target: str,
    eligible_from: int,
    turns: Sequence[Mapping[str, Any]],
    cutoffs: Sequence[int] = CUTOFFS,
) -> dict[int, bool]:
    if (
        not isinstance(target, str)
        or not target
        or not isinstance(eligible_from, int)
        or isinstance(eligible_from, bool)
        or not 1 <= eligible_from <= TURN_COUNT
        or tuple(cutoffs) != CUTOFFS
    ):
        raise C200ProbeError("candidate recall input is invalid")
    result = {cutoff: False for cutoff in CUTOFFS}
    for row in turns:
        turn = row.get("turn") if isinstance(row, Mapping) else None
        c200 = row.get("c200") if isinstance(row, Mapping) else None
        if (
            not isinstance(turn, int)
            or isinstance(turn, bool)
            or not isinstance(c200, (list, tuple))
        ):
            raise C200ProbeError("candidate recall trace row is invalid")
        if turn < eligible_from:
            continue
        for cutoff in CUTOFFS:
            result[cutoff] = result[cutoff] or target in c200[:cutoff]
    return result


def _recall_view(flags: Sequence[Mapping[int, bool]], indices: Sequence[int]) -> dict[str, Any]:
    denominator = len(indices)
    return {
        f"c{cutoff}": {
            "count": sum(int(bool(flags[index][cutoff])) for index in indices),
            "fraction": round(
                sum(int(bool(flags[index][cutoff])) for index in indices) / denominator,
                6,
            )
            if denominator
            else 0.0,
        }
        for cutoff in CUTOFFS
    }


def aggregate_candidate_recall(
    flags: Sequence[Mapping[int, bool]],
    *,
    outer_fold: Sequence[int],
    family_index: Sequence[int],
    taxonomy: Sequence[str],
) -> dict[str, Any]:
    count = len(flags)
    if not count or not (len(outer_fold) == len(family_index) == len(taxonomy) == count):
        raise C200ProbeError("candidate recall aggregate dimensions drifted")
    for row in flags:
        if set(row) != set(CUTOFFS) or any(not isinstance(row[key], bool) for key in CUTOFFS):
            raise C200ProbeError("candidate recall flag schema drifted")
    family_fold: dict[int, int] = {}
    for family, fold in zip(family_index, outer_fold, strict=True):
        if (
            not isinstance(family, (int, np.integer))
            or isinstance(family, (bool, np.bool_))
            or not isinstance(fold, (int, np.integer))
            or isinstance(fold, (bool, np.bool_))
            or not 0 <= int(fold) < 5
        ):
            raise C200ProbeError("fold/family label is invalid")
        previous = family_fold.setdefault(int(family), int(fold))
        if previous != int(fold):
            raise C200ProbeError("one product family crosses outer folds")
    indices = list(range(count))
    frontier = [index for index in indices if not flags[index][100]]
    increment = [index for index in frontier if flags[index][200]]
    by_fold = []
    for fold in sorted({int(value) for value in outer_fold}):
        members = [index for index, value in enumerate(outer_fold) if int(value) == fold]
        by_fold.append(
            {
                "fold": fold,
                "sessions": len(members),
                "recall": _recall_view(flags, members),
                "increment": sum(int(index in increment) for index in members),
            }
        )
    taxonomies = sorted({str(value) for value in taxonomy})
    by_taxonomy = {
        name: {
            "sessions": len(members := [index for index, value in enumerate(taxonomy) if str(value) == name]),
            "recall": _recall_view(flags, members),
            "increment": sum(int(index in increment) for index in members),
        }
        for name in taxonomies
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
        "c100_absent_frontier": {
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
            "first_frontier": "ranks_101_to_200",
        },
        "by_outer_fold": by_fold,
        "target_uniform": target_uniform,
        "by_taxonomy": by_taxonomy,
        "family_disjoint_audit": {
            "valid": True,
            "family_count": len(family_members),
            "families_crossing_outer_folds": 0,
        },
    }


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise C200ProbeError("cannot summarize empty values")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def inflation_summary(
    lengths: Sequence[int],
    *,
    trace_bytes: int,
    reference_bytes: int,
) -> dict[str, Any]:
    if (
        not lengths
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 100 <= value <= 200
            for value in lengths
        )
        or not isinstance(trace_bytes, int)
        or not isinstance(reference_bytes, int)
        or trace_bytes <= 0
        or reference_bytes <= 0
    ):
        raise C200ProbeError("candidate inflation surface is invalid")
    added = [value - 100 for value in lengths]

    def describe(values: Sequence[int]) -> dict[str, int | float]:
        return {
            "minimum": min(values),
            "p50": int(_nearest_rank(values, 0.50)),
            "p95": int(_nearest_rank(values, 0.95)),
            "maximum": max(values),
            "mean": round(statistics.fmean(values), 6),
        }

    return {
        "c200_length": describe(lengths),
        "added_candidates": describe(added),
        "candidate_cells": sum(lengths),
        "candidate_cell_ratio": round(sum(lengths) / (len(lengths) * 100), 12),
        "trace_bytes": trace_bytes,
        "trace_byte_ratio": round(trace_bytes / reference_bytes, 12),
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
    if ROOT.absolute() == CODE_ROOT.absolute():
        _require_lexical_ancestry(
            ROOT,
            Path(r"D:\tiktok"),
            "receipt repository root",
        )
    _require_lexical_ancestry(
        OUTPUT_PATH.parent,
        ROOT,
        "receipt parent",
    )
    try:
        lexical_root = ROOT.absolute()
        lexical_output = OUTPUT_PATH.absolute()
        if not _inside(lexical_output, lexical_root):
            raise C200ProbeError("receipt path escapes the repository")
        root = ROOT.resolve(strict=True)
        parent = OUTPUT_PATH.parent.resolve(strict=True)
    except OSError as error:
        raise C200ProbeError("receipt parent must already exist") from error
    if not parent.is_dir() or not _inside(parent, root):
        raise C200ProbeError("receipt parent escapes the repository")
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
    implementation_commit: str,
    error: BaseException,
    *,
    phase: str,
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
        raise C200ProbeError("implementation commit is invalid")
    _receipt_parent()
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise C200ProbeError("the one-shot receipt path is already consumed")
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
                raise C200ProbeError("exclusive one-shot receipt creation failed") from error
            raise
        try:
            _write_descriptor(
                descriptor,
                _invalid_value(
                    implementation_commit,
                    error,
                    phase="pending_receipt_write",
                ),
            )
        except BaseException:
            pass
        finally:
            _safe_close_descriptor(descriptor)
        raise C200ProbeError("one-shot receipt was consumed during initialization") from error
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
        except BaseException as first_error:
            try:
                _write_descriptor(descriptor, value)
            except BaseException as second_error:
                raise C200ProbeError(
                    "durable invalid receipt seal failed twice"
                ) from second_error
    finally:
        _safe_close_descriptor(descriptor)


def _walk_values(value: object) -> Iterable[object]:
    yield value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_values(child)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            yield from _walk_values(child)


def _result_privacy_scan(
    value: object,
    *,
    catalog_ids: Iterable[str] = (),
) -> None:
    catalog_membership = frozenset(str(item).upper() for item in catalog_ids)
    forbidden_keys = {
        "session_id",
        "sample_id",
        "product_id",
        "target",
        "target_id",
        "ground_truth",
        "positive_index",
        "eligible_from",
        "message",
        "per_session",
        "membership_vector",
    }
    for item in _walk_values(value):
        if isinstance(item, Mapping):
            if {str(key).casefold() for key in item} & forbidden_keys:
                raise C200ProbeError("result contains a forbidden identity-bearing key")
        elif isinstance(item, np.ndarray):
            raise C200ProbeError("result contains a numeric array")
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            if len(item) >= SESSION_COUNT:
                raise C200ProbeError("result contains a session-length vector")
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            exact_catalog_tokens = {
                match.group(0).decode("ascii").upper()
                for match in CATALOG_IDENTIFIER_TOKEN_RE.finditer(encoded)
            }
            if ASIN_SHAPE_RE.search(encoded) or exact_catalog_tokens & catalog_membership:
                raise C200ProbeError("result contains an identifier-shaped token")


def _open_proxy_after_receipt(path: Path) -> OpenProxySource:
    resolved = _require_regular_file(path, "train_explore proxy")
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    byte_count = 0
    handle = resolved.open("rb")
    try:
        if os.get_inheritable(handle.fileno()):
            raise C200ProbeError("proxy descriptor is unexpectedly inheritable")
        before = _handle_snapshot(handle)
        for raw in handle:
            if not raw.strip():
                raise C200ProbeError("proxy contains a blank physical row")
            digest.update(raw)
            byte_count += len(raw)
            try:
                value = json.loads(
                    raw.decode("utf-8", errors="strict"),
                    object_pairs_hook=_unique_object,
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                raise C200ProbeError("proxy JSONL is invalid") from error
            if not isinstance(value, dict):
                raise C200ProbeError("proxy row is not an object")
            rows.append(value)
        after = _handle_snapshot(handle)
        identity = FileIdentity(byte_count, len(rows), digest.hexdigest(), after)
        if not (
            before == after
            and byte_count == before[0]
            and identity.report()
            == {"bytes": PROXY_BYTES, "rows": PROXY_ROWS, "sha256": PROXY_SHA256}
        ):
            raise C200ProbeError("proxy identity drifted")
        return OpenProxySource(resolved, handle, rows, identity)
    except BaseException:
        handle.close()
        raise


def _reverify_and_close_proxy(source: OpenProxySource) -> FileIdentity:
    try:
        final_sha, final_bytes = _hash_handle(source.handle)
        final_handle = _handle_snapshot(source.handle)
        final_path = _path_snapshot(source.path)
        try:
            same_resolved_path = (
                source.path.resolve(strict=True).as_posix().casefold()
                == source.path.as_posix().casefold()
            )
        except OSError:
            same_resolved_path = False
        if not (
            final_sha == source.identity.sha256
            and final_bytes == source.identity.bytes
            and final_handle == source.identity.snapshot == final_path
            and same_resolved_path
        ):
            raise C200ProbeError("proxy changed while its verified handle was retained")
        return source.identity
    finally:
        source.handle.close()


def _materialize_visible_context_from_samples(
    samples: Sequence[Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    categories: Mapping[str, list[str]],
    output_path: Path,
) -> dict[str, Any]:
    """Exact frozen simulator, writing only identifier-free visible context."""

    if len(samples) != SESSION_COUNT:
        raise C200ProbeError("proxy row count drifted before context build")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    parser = object.__new__(Agent)
    parser.question_policy = "fast"
    redacted_messages = 0
    context_digest = hashlib.sha256()
    with output_path.open("xb") as output:
        for sample in samples:
            sample_id = str(sample.get("sample_id", ""))
            ground = sample.get("ground_truth")
            label = str(ground.get("parent_asin", "")) if isinstance(ground, Mapping) else ""
            if label not in products or not sample_id:
                raise C200ProbeError("trusted simulator source row is invalid")
            card, behavior = materialize_hidden_fields(dict(sample), products)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = effective["scenario_type"] != "intent_override"
            message = initial_message(
                effective,
                coarse_category(categories[label]),
                disclosed,
            )
            state = SessionState(
                profile=blind_oracle.project_profile(sample.get("user_profile"))
            )
            turns: list[dict[str, Any]] = []
            for turn in range(1, TURN_COUNT + 1):
                visible, redacted = blind_oracle.sanitize_worker_visible_message(message)
                redacted_messages += int(redacted > 0)
                previous_version = state.version
                parsed = Agent._update_state(parser, state, visible, turn)
                state.slot_ledger.reconcile(
                    context_builder.build_conversation_constraint_view(
                        state.category_text,
                        state.active_terms,
                        state.excluded_terms,
                    ),
                    turn=turn,
                    version=state.version,
                    message=visible,
                    suppressed_slots=state.exhausted_attributes,
                    retired_status=(
                        context_builder.SUPERSEDED
                        if parsed.is_override or state.version != previous_version
                        else context_builder.DELETED
                    ),
                )
                active_records = [
                    context_builder._safe_record(record)
                    for record in state.slot_ledger.active_records()
                ]
                retired_records = [
                    context_builder._safe_record(record)
                    for record in state.slot_ledger.records
                    if record.status != "active"
                ]
                query_terms = Agent._query_terms(parser, state)
                goal_messages = state.messages[max(0, state.version_anchor_turn - 1) :]
                row = {
                    "message": visible,
                    "goal_messages": list(goal_messages),
                    "category_text": state.category_text,
                    "active_terms": list(state.active_terms),
                    "excluded_terms": sorted(state.excluded_terms),
                    "query_terms": list(query_terms),
                    "version": int(state.version),
                    "version_anchor_turn": int(state.version_anchor_turn),
                    "override_count": int(state.override_count),
                    "current_turn_override": bool(parsed.is_override),
                    "active_records": active_records,
                    "retired_records": retired_records,
                    "hard_clause_terms": list(
                        context_builder._latest_hard_clause_terms(state)
                    ),
                    "budget_upper": context_builder._budget_upper(goal_messages),
                }
                if context_builder._walk_keys(row) & context_builder.FORBIDDEN_ARTIFACT_KEYS:
                    raise C200ProbeError("visible context contains a forbidden key")
                serialized = json.dumps(row, sort_keys=True, ensure_ascii=False)
                if (
                    label.casefold() in serialized.casefold()
                    or sample_id in serialized
                    or context_builder.ASIN_SHAPE_RE.search(serialized)
                ):
                    raise C200ProbeError("identity token leaked into visible context")
                turns.append(row)
                ask_attribute = Agent._select_question(state, turn)
                if turn < TURN_COUNT:
                    override = effective.get("behavior", {}).get("override") or {}
                    if (
                        not override_applied
                        and turn + 1 == int(override.get("turn", 3))
                    ):
                        override_applied = True
                        new_value = str(override.get("new_value", ""))
                        if new_value:
                            disclosed.add(new_value)
                        message = str(
                            override.get(
                                "message",
                                "Actually, please ignore my earlier preference.",
                            )
                        )
                    else:
                        message, boundary_used = customer_reply(
                            effective,
                            ask_attribute,
                            disclosed,
                            boundary_used,
                        )
            container = {
                "schema_version": context_builder.CONTEXT_SCHEMA_VERSION,
                "turns": turns,
            }
            payload = _canonical_bytes(container) + b"\n"
            output.write(payload)
            context_digest.update(payload)
        output.flush()
        os.fsync(output.fileno())
    return {
        "rows": len(samples),
        "turns": len(samples) * TURN_COUNT,
        "bytes": output_path.stat().st_size,
        "sha256": context_digest.hexdigest(),
        "redacted_message_count": redacted_messages,
    }


def _prepare_cache_root() -> None:
    _require_lexical_ancestry(CACHE_ROOT.parent, ROOT, "cache parent")
    parent = CACHE_ROOT.parent.resolve(strict=True)
    root = ROOT.resolve(strict=True)
    if not _inside(parent, root) or _is_link_or_reparse(parent):
        raise C200ProbeError("cache parent is unsafe")
    try:
        os.mkdir(CACHE_ROOT)
    except OSError as error:
        raise C200ProbeError("exclusive cache directory creation failed") from error
    if _is_link_or_reparse(CACHE_ROOT) or not CACHE_ROOT.is_dir():
        raise C200ProbeError("cache directory is unsafe")


def _offline_worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def _worker_command(nonce: str, trace_path: Path) -> list[str]:
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
        "--trace-output",
        str(trace_path),
    ]


def _finite_number(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise C200ProbeError(f"{label} is not finite numeric")
    return float(value)


def _validate_latency_summary(
    value: object,
    *,
    unit: str,
    maximum_p95: float,
) -> dict[str, Any]:
    keys = {
        "count",
        f"p50_{unit}",
        f"p95_{unit}",
        f"maximum_{unit}",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise C200ProbeError("worker latency summary schema drifted")
    if (
        not isinstance(value["count"], int)
        or isinstance(value["count"], bool)
        or value["count"] != REFERENCE_C100_RECORDS
    ):
        raise C200ProbeError("worker latency count drifted")
    p50 = _finite_number(value[f"p50_{unit}"], "worker p50")
    p95 = _finite_number(value[f"p95_{unit}"], "worker p95")
    maximum = _finite_number(value[f"maximum_{unit}"], "worker maximum latency")
    if not (0.0 <= p50 <= p95 <= maximum and p95 <= maximum_p95):
        raise C200ProbeError("worker latency budget or ordering failed")
    return dict(value)


def _validate_worker_receipt(
    payload: bytes,
    *,
    nonce: str,
) -> dict[str, Any]:
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise C200ProbeError("worker stdout is not one canonical JSON line")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                C200ProbeError(f"worker emitted non-finite {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise C200ProbeError("worker receipt JSON is invalid") from error
    if not isinstance(value, dict) or payload != _canonical_bytes(value) + b"\n":
        raise C200ProbeError("worker receipt is not canonical")
    if set(value) != {
        "kind",
        "nonce",
        "trace_sha256",
        "trace_bytes",
        "record_count",
        "summary",
    }:
        raise C200ProbeError("worker receipt top-level schema drifted")
    if not (
        value["kind"] == "receipt"
        and value["nonce"] == nonce
        and isinstance(value["trace_sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["trace_sha256"])
        and isinstance(value["trace_bytes"], int)
        and not isinstance(value["trace_bytes"], bool)
        and value["trace_bytes"] > 0
        and isinstance(value["record_count"], int)
        and not isinstance(value["record_count"], bool)
        and value["record_count"] == REFERENCE_C100_RECORDS
    ):
        raise C200ProbeError("worker receipt identity surface drifted")
    summary = value["summary"]
    if not isinstance(summary, dict) or set(summary) != {
        "schema_version",
        "environment",
        "configuration",
        "pool_lengths",
        "latency",
        "resources",
        "lifecycle",
    }:
        raise C200ProbeError("worker summary schema drifted")
    if summary["schema_version"] != worker_contract.SCHEMA_VERSION:
        raise C200ProbeError("worker summary version drifted")
    environment = summary["environment"]
    expected_environment_keys = {
        "executable",
        "python",
        "sqlite",
        "pythonhashseed",
        "network_attempt_count",
        "gpu_used",
        "gpu_peak_bytes",
    }
    if not isinstance(environment, dict) or set(environment) != expected_environment_keys:
        raise C200ProbeError("worker environment schema drifted")
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
        and environment["pythonhashseed"] == "0"
        and isinstance(environment["network_attempt_count"], int)
        and not isinstance(environment["network_attempt_count"], bool)
        and environment["network_attempt_count"] == 0
        and environment["gpu_used"] is False
        and isinstance(environment["gpu_peak_bytes"], int)
        and not isinstance(environment["gpu_peak_bytes"], bool)
        and environment["gpu_peak_bytes"] == 0
    ):
        raise C200ProbeError("worker environment identity drifted")
    if summary["configuration"] != {
        "p11_mode": "control",
        "small_ranker_mode": "off",
        "question_policy": "fast",
        "rerank_mode": "off",
        "retrieval_mode": "coverage",
    }:
        raise C200ProbeError("worker Agent configuration drifted")
    pool = summary["pool_lengths"]
    if not isinstance(pool, dict) or set(pool) != {
        "min",
        "p50",
        "p95",
        "max",
        "mean",
        "records",
        "candidate_cells",
    }:
        raise C200ProbeError("worker pool summary schema drifted")
    if not (
        all(
            isinstance(pool[key], int) and not isinstance(pool[key], bool)
            for key in ("min", "p50", "p95", "max", "records", "candidate_cells")
        )
        and 100 <= pool["min"] <= pool["p50"] <= pool["p95"] <= pool["max"] <= 200
        and pool["records"] == REFERENCE_C100_RECORDS
        and 2_000_000 <= pool["candidate_cells"] <= 4_000_000
        and 100.0 <= _finite_number(pool["mean"], "worker pool mean") <= 200.0
    ):
        raise C200ProbeError("worker pool summary values drifted")
    latency = summary["latency"]
    if not isinstance(latency, dict) or set(latency) != {"respond", "capture"}:
        raise C200ProbeError("worker latency container drifted")
    _validate_latency_summary(
        latency["respond"],
        unit="milliseconds",
        maximum_p95=RESPOND_P95_MS_MAXIMUM,
    )
    _validate_latency_summary(
        latency["capture"],
        unit="microseconds",
        maximum_p95=CAPTURE_P95_US_MAXIMUM,
    )
    resources = summary["resources"]
    if (
        not isinstance(resources, dict)
        or set(resources) != {"peak_working_set_bytes"}
        or not isinstance(resources["peak_working_set_bytes"], int)
        or isinstance(resources["peak_working_set_bytes"], bool)
        or not 0 < resources["peak_working_set_bytes"] <= WORKER_RSS_MAXIMUM
    ):
        raise C200ProbeError("worker RSS budget failed")
    lifecycle = summary["lifecycle"]
    if (
        not isinstance(lifecycle, dict)
        or set(lifecycle)
        != {
            "agent_closed_before_trace_publish",
            "sqlite_closed_before_trace_publish",
        }
        or lifecycle["agent_closed_before_trace_publish"] is not True
        or lifecycle["sqlite_closed_before_trace_publish"] is not True
    ):
        raise C200ProbeError("worker lifecycle gate failed")
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
        raise C200ProbeError("isolated C200 worker failed to complete") from error
    wall = time.perf_counter() - started
    if completed.returncode != 0 or completed.stderr != b"":
        if ASIN_SHAPE_RE.search(completed.stderr):
            raise C200ProbeError("worker failure output contained an identifier")
        raise C200ProbeError("isolated C200 worker exited uncleanly")
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
    worker_result: Mapping[str, Any],
    trace: TraceValidation,
) -> None:
    receipt = worker_result["receipt"]
    if not (
        receipt["trace_sha256"] == trace.canonical_trace_sha256
        and receipt["trace_bytes"] == trace.canonical_trace_bytes
        and receipt["record_count"] == len(trace.records)
        and receipt["summary"]["pool_lengths"]
        == _pool_summary_from_lengths(trace.lengths)
    ):
        raise C200ProbeError("worker receipt does not bind to its closed trace")


def _files_equal(left: Path, right: Path) -> bool:
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            left_chunk = first.read(1 << 20)
            right_chunk = second.read(1 << 20)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _load_fold_labels_after_traces(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    resolved = _require_regular_file(path, "numeric fold archive")
    with resolved.open("rb") as handle:
        before = _handle_snapshot(handle)
        first_sha, first_bytes = _hash_handle(handle)
        try:
            with np.load(handle, allow_pickle=False) as archive:
                outer_fold = np.asarray(archive["outer_fold"]).copy()
                family_index = np.asarray(archive["family_index"]).copy()
        except (KeyError, OSError, ValueError) as error:
            raise C200ProbeError("numeric fold archive is invalid") from error
        second_sha, second_bytes = _hash_handle(handle)
        after = _handle_snapshot(handle)
    if not (
        before == after
        and before[0] == LABEL_BYTES == first_bytes == second_bytes
        and first_sha == second_sha == LABEL_SHA256
    ):
        raise C200ProbeError("numeric fold archive identity drifted")
    if (
        outer_fold.shape != (SESSION_COUNT,)
        or family_index.shape != (SESSION_COUNT,)
        or outer_fold.dtype.kind not in "iu"
        or family_index.dtype.kind not in "iu"
        or np.any(outer_fold < 0)
        or np.any(outer_fold > 4)
        or np.any(family_index < 0)
    ):
        raise C200ProbeError("fold/family arrays drifted")
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
        raise C200ProbeError("target-attach proxy row count drifted")
    targets: list[str] = []
    eligibility: list[int] = []
    taxonomy: list[str] = []
    for sample in samples:
        ground = sample.get("ground_truth")
        target = str(ground.get("parent_asin", "")) if isinstance(ground, Mapping) else ""
        if target not in products:
            raise C200ProbeError("target-attach catalog membership drifted")
        card, behavior = materialize_hidden_fields(dict(sample), products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        try:
            eligible_turn = int(blind_oracle._eligible_from_turn(effective))
        except (AttributeError, TypeError, ValueError, blind_oracle.OracleRunError) as error:
            raise C200ProbeError("target-attach eligibility drifted") from error
        taxonomy_value = sample.get("taxonomy")
        taxonomy_name = (
            str(taxonomy_value.get("group", "unknown"))
            if isinstance(taxonomy_value, Mapping)
            else "unknown"
        )
        if (
            not 1 <= eligible_turn <= TURN_COUNT
            or taxonomy_name not in EXPECTED_TAXONOMY
        ):
            raise C200ProbeError("target-attach aggregate stratum drifted")
        targets.append(target)
        eligibility.append(eligible_turn)
        taxonomy.append(taxonomy_name)
    return targets, eligibility, taxonomy


def _flags_from_trace(
    trace: TraceValidation,
    targets: Sequence[str],
    eligibility: Sequence[int],
) -> list[dict[int, bool]]:
    if len(targets) != SESSION_COUNT or len(eligibility) != SESSION_COUNT:
        raise C200ProbeError("candidate-recall target dimensions drifted")
    return [
        candidate_recall_flags(
            targets[index],
            int(eligibility[index]),
            trace.records[index * TURN_COUNT : (index + 1) * TURN_COUNT],
        )
        for index in range(SESSION_COUNT)
    ]


def _verify_context_identity(
    build_stats: Mapping[str, Any],
) -> FileIdentity:
    expected = {
        "rows": CONTEXT_ROWS,
        "turns": CONTEXT_TURNS,
        "bytes": CONTEXT_BYTES,
        "sha256": CONTEXT_SHA256,
        "redacted_message_count": CONTEXT_REDACTED_MESSAGES,
    }
    if dict(build_stats) != expected:
        raise C200ProbeError("rebuilt visible context differs from frozen reference")
    identity = _file_identity(CONTEXT_PATH, "rebuilt visible context")
    if identity.report() != {
        "bytes": CONTEXT_BYTES,
        "rows": CONTEXT_ROWS,
        "sha256": CONTEXT_SHA256,
    }:
        raise C200ProbeError("closed visible context identity drifted")
    return identity


def _final_target_free_rehash(preflight: Preflight) -> dict[str, Any]:
    catalog = _file_identity(CATALOG_PATH, "catalog final rehash")
    if catalog.report() != preflight.source_identities["catalog"]:
        raise C200ProbeError("catalog changed during the formal probe")
    shard_hashes: list[str] = []
    for index, (name, expected_sha) in enumerate(
        zip(BLIND_NAMES, BLIND_SHA256, strict=True)
    ):
        identity = _file_identity(BLIND_ROOT / name, f"blind shard final {index + 1}")
        if identity.rows != BLIND_SHARD_ROWS or identity.sha256 != expected_sha:
            raise C200ProbeError("frozen C100 changed during the formal probe")
        shard_hashes.append(identity.sha256)
    worker_sha = hashlib.sha256(
        _require_regular_file(WORKER_PATH, "worker final rehash").read_bytes()
    ).hexdigest()
    if worker_sha != preflight.source_identities["worker_sha256"]:
        raise C200ProbeError("worker source changed during the formal probe")
    if _canonical_sha256(_load_preregistration()) != PREREG_CANONICAL_SHA256:
        raise C200ProbeError("preregistration changed during the formal probe")
    return {
        "catalog_sha256": catalog.sha256,
        "blind_shard_sha256": shard_hashes,
        "worker_sha256": worker_sha,
        "preregistration_canonical_sha256": PREREG_CANONICAL_SHA256,
    }


def run(implementation_commit: str) -> dict[str, Any]:
    """Consume the unique formal C200 probe and write its aggregate receipt."""

    formal_started = time.perf_counter()
    preflight = preflight_only(implementation_commit)
    descriptor: int | None = None
    proxy_source: OpenProxySource | None = None
    phase = "receipt_initialization"
    try:
        descriptor = _open_receipt(implementation_commit)
        phase = "post_receipt_initialization"
        phase = "proxy_first_same_handle_verification"
        proxy_source = _open_proxy_after_receipt(PROXY_PATH)
        build_samples = proxy_source.samples
        proxy_first = proxy_source.identity

        phase = "visible_context_rebuild"
        _prepare_cache_root()
        context_started = time.perf_counter()
        build_stats = _materialize_visible_context_from_samples(
            build_samples,
            preflight.products,
            preflight.categories,
            CONTEXT_PATH,
        )
        context_wall = time.perf_counter() - context_started
        context_identity = _verify_context_identity(build_stats)
        if context_wall > CONTEXT_WALL_MAXIMUM:
            raise C200ProbeError("visible-context build exceeded its wall budget")

        phase = "simultaneous_target_blind_workers"
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
            futures: dict[Any, str] = {}
            try:
                futures = {
                    executor.submit(_run_one_worker, nonces[name], path): name
                    for name, path in zip(
                        ("replica_a", "replica_b"), TRACE_PATHS, strict=True
                    )
                }
                for future in as_completed(futures):
                    name = futures[future]
                    worker_results[name] = future.result()
            except BaseException as worker_error:
                for pending in futures:
                    pending.cancel()
                if descriptor is not None:
                    try:
                        _write_invalid_receipt(
                            descriptor,
                            implementation_commit,
                            worker_error,
                            phase=phase,
                        )
                    finally:
                        descriptor = None
                raise
        if set(worker_results) != {"replica_a", "replica_b"}:
            raise C200ProbeError("one isolated worker result is missing")

        phase = "closed_trace_validation_before_target_attach"
        trace_a = load_and_validate_c200_trace(
            TRACE_PATHS[0],
            preflight.frozen_c100,
            preflight.catalog_ids,
        )
        trace_b = load_and_validate_c200_trace(
            TRACE_PATHS[1],
            preflight.frozen_c100,
            preflight.catalog_ids,
        )
        _bind_worker_receipt_to_trace(worker_results["replica_a"], trace_a)
        _bind_worker_receipt_to_trace(worker_results["replica_b"], trace_b)
        for trace in (trace_a, trace_b):
            if not (
                trace.normalized_c100_sha256 == REFERENCE_C100_SHA256
                and trace.normalized_c100_bytes == REFERENCE_C100_BYTES
            ):
                raise C200ProbeError("fresh C100 reproduction identity failed")
        if not (
            trace_a.canonical_trace_sha256 == trace_b.canonical_trace_sha256
            and trace_a.canonical_trace_bytes == trace_b.canonical_trace_bytes
            and trace_a.records == trace_b.records
            and _files_equal(TRACE_PATHS[0], TRACE_PATHS[1])
        ):
            raise C200ProbeError("the two fresh C200 traces are not exact repeats")

        phase = "target_free_resource_gates_before_target_attach"
        inflation_a = inflation_summary(
            trace_a.lengths,
            trace_bytes=trace_a.canonical_trace_bytes,
            reference_bytes=REFERENCE_C100_BYTES,
        )
        inflation_b = inflation_summary(
            trace_b.lengths,
            trace_bytes=trace_b.canonical_trace_bytes,
            reference_bytes=REFERENCE_C100_BYTES,
        )
        if inflation_a != inflation_b:
            raise C200ProbeError("candidate inflation is not an exact repeat")
        if not (
            inflation_a["candidate_cell_ratio"] <= CELL_RATIO_MAXIMUM
            and inflation_a["trace_byte_ratio"] <= TRACE_RATIO_MAXIMUM
        ):
            raise C200ProbeError("candidate inflation budget failed")
        worker_rss = {
            name: int(value["receipt"]["summary"]["resources"]["peak_working_set_bytes"])
            for name, value in worker_results.items()
        }
        if sum(worker_rss.values()) > WORKER_RSS_SUM_MAXIMUM:
            raise C200ProbeError("conservative two-worker RSS sum exceeded budget")
        if time.perf_counter() - formal_started > TOTAL_WALL_MAXIMUM:
            raise C200ProbeError("formal C200 probe exceeded total wall before target attach")
        activation_sessions = sum(
            int(
                any(
                    length > 100
                    for length in trace_a.lengths[
                        index * TURN_COUNT : (index + 1) * TURN_COUNT
                    ]
                )
            )
            for index in range(SESSION_COUNT)
        )
        activation_turns = sum(int(length > 100) for length in trace_a.lengths)

        phase = "evaluator_side_target_and_fold_attach"
        proxy_final = _reverify_and_close_proxy(proxy_source)
        proxy_source = None
        if proxy_final.report() != proxy_first.report():
            raise C200ProbeError("proxy identity changed across the worker boundary")
        targets, eligibility, taxonomy = _derive_target_membership_inputs(
            build_samples,
            preflight.products,
        )
        outer_fold, family_index, label_identity = _load_fold_labels_after_traces(
            LABEL_PATH
        )
        flags_a = _flags_from_trace(trace_a, targets, eligibility)
        flags_b = _flags_from_trace(trace_b, targets, eligibility)
        del build_samples, targets, eligibility
        if flags_a != flags_b:
            raise C200ProbeError("derived candidate-recall membership is not repeatable")
        aggregate_a = aggregate_candidate_recall(
            flags_a,
            outer_fold=outer_fold,
            family_index=family_index,
            taxonomy=taxonomy,
        )
        aggregate_b = aggregate_candidate_recall(
            flags_b,
            outer_fold=outer_fold,
            family_index=family_index,
            taxonomy=taxonomy,
        )
        if aggregate_a != aggregate_b:
            raise C200ProbeError("derived candidate-recall aggregate is not repeatable")
        sanity = aggregate_a["all_sessions"]
        if any(
            sanity[f"c{cutoff}"]["count"] != expected
            for cutoff, expected in EXPECTED_CANDIDATE_RECALL.items()
        ) or aggregate_a["c100_absent_frontier"]["sessions"] != EXPECTED_C100_MISSES:
            raise C200ProbeError("frozen C100 recall sanity counts drifted")

        phase = "resource_repeat_and_source_gates"
        final_source_rehash = _final_target_free_rehash(preflight)
        final_git = _validate_git_checkpoint(implementation_commit)
        context_final = _file_identity(CONTEXT_PATH, "context final rehash")
        if context_final.report() != context_identity.report():
            raise C200ProbeError("visible context changed during worker execution")
        final_trace_identities = [
            _file_identity(path, f"fresh trace final {index + 1}")
            for index, path in enumerate(TRACE_PATHS)
        ]
        for identity, trace in zip(
            final_trace_identities, (trace_a, trace_b), strict=True
        ):
            if identity.report() != {
                "bytes": trace.canonical_trace_bytes,
                "rows": REFERENCE_C100_RECORDS,
                "sha256": trace.canonical_trace_sha256,
            }:
                raise C200ProbeError("fresh trace changed before final receipt")
        if final_trace_identities[0].report() != final_trace_identities[1].report():
            raise C200ProbeError("fresh trace final identities are not equal")
        final_source_rehash["fresh_traces"] = [
            identity.report() for identity in final_trace_identities
        ]
        total_wall = time.perf_counter() - formal_started
        if total_wall > TOTAL_WALL_MAXIMUM:
            raise C200ProbeError("formal C200 probe exceeded total wall budget")
        parent_current_rss, parent_peak_rss = _process_memory()
        if not 0 < parent_current_rss <= parent_peak_rss:
            raise C200ProbeError("final parent working-set measurement is unavailable")

        phase = "aggregate_result_seal"
        increment = aggregate_a["increment"]
        recall_go = bool(increment["count"] > 0)
        family_members: dict[int, list[int]] = {}
        for session_index, family in enumerate(family_index):
            family_members.setdefault(int(family), []).append(session_index)
        exact_target_uniform_c100 = statistics.fmean(
            statistics.fmean(int(flags_a[index][100]) for index in members)
            for members in family_members.values()
        )
        exact_target_uniform_c200 = statistics.fmean(
            statistics.fmean(int(flags_a[index][200]) for index in members)
            for members in family_members.values()
        )
        target_uniform_gain = exact_target_uniform_c200 > exact_target_uniform_c100
        training_evidence_gate = bool(
            recall_go
            and target_uniform_gain
            and increment["target_cluster_count"] > 0
            and increment["outer_fold_span"] >= 2
        )
        status = (
            "C200_RECALL_GO_CONTINUE_C400"
            if recall_go
            else "C200_RECALL_NO_GO_CONTINUE_C400"
        )
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
                "original_preregistration_commit": ORIGINAL_PREREG_COMMIT,
                "branch": BRANCH,
                "default": "off",
                "causal": True,
                "runtime_target_blind": True,
                "full_agent_evaluator_started": False,
                "protected_splits_opened": False,
                "fit_or_selection_performed": False,
            },
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
                "reason": "candidate-recall-only probe; Top10 was byte-behavior unchanged",
            },
            "candidate_recall": aggregate_a,
            "candidate_retention": {
                "old_c100_is_exact_ordered_prefix": True,
                "old_candidate_loss_count": 0,
                "candidate_level_hit_to_miss": 0,
                "normalized_c100_sha256": trace_a.normalized_c100_sha256,
                "normalized_c100_bytes": trace_a.normalized_c100_bytes,
            },
            "activation": {
                "sessions_with_more_than_c100": activation_sessions,
                "turns_with_more_than_c100": activation_turns,
            },
            "inflation": inflation_a,
            "exact_repeat": {
                "passed": True,
                "trace_sha256": trace_a.canonical_trace_sha256,
                "trace_bytes": trace_a.canonical_trace_bytes,
                "record_count": len(trace_a.records),
                "candidate_arrays_equal": True,
                "recall_aggregates_equal": True,
                "inflation_aggregates_equal": True,
            },
            "resources": {
                "context_build_wall_seconds": round(context_wall, 6),
                "total_wall_seconds": round(total_wall, 6),
                "worker_wall_seconds": {
                    name: value["wall_seconds"] for name, value in worker_results.items()
                },
                "worker_peak_working_set_bytes": worker_rss,
                "conservative_worker_peak_sum_bytes": sum(worker_rss.values()),
                "parent_current_working_set_bytes": parent_current_rss,
                "parent_lifetime_peak_working_set_bytes": parent_peak_rss,
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
                "proxy": proxy_final.report(),
                "numeric_fold_archive": label_identity,
                "visible_context": {
                    **context_identity.report(),
                    "turns": CONTEXT_TURNS,
                    "redacted_message_count": CONTEXT_REDACTED_MESSAGES,
                },
                "fresh_trace_sha256": trace_a.canonical_trace_sha256,
                "final_rehash": final_source_rehash,
            },
            "git": final_git,
            "decision": {
                "recall_go": recall_go,
                "training_evidence_gate_passed": training_evidence_gate,
                "top10_global_promotion": False,
                "next_stage": "independent one-variable C400 LIMIT-expansion preregistration",
                "fallback_order": [
                    "SR-V2.12-FIXED-TWO-PAGE-GRACE",
                    "v1.9",
                    "P11",
                    "R08",
                ],
            },
        }
        _result_privacy_scan(result, catalog_ids=preflight.catalog_ids)
        if descriptor is None:
            raise C200ProbeError("receipt descriptor was closed before result seal")
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
                    descriptor,
                    implementation_commit,
                    error,
                    phase=phase,
                )
            except BaseException as seal_error:
                invalid_seal_error = seal_error
            finally:
                descriptor = None
        if proxy_source is not None:
            try:
                proxy_source.handle.close()
            except BaseException:
                pass
            proxy_source = None
        if invalid_seal_error is not None:
            raise C200ProbeError("formal failure receipt could not be sealed") from invalid_seal_error
        if isinstance(error, C200ProbeError):
            raise
        raise C200ProbeError("formal C200 one-shot failed") from error


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
            "c100_sha256": checked.source_identities["blind_c100"]["normalized_sha256"],
            "receipt_created": False,
        }
    else:
        outcome = run(arguments.implementation_commit)
        summary = {
            "status": outcome["status"],
            "commit": outcome["implementation"]["commit"],
            "c100_count": outcome["candidate_recall"]["all_sessions"]["c100"]["count"],
            "c200_count": outcome["candidate_recall"]["all_sessions"]["c200"]["count"],
            "increment_count": outcome["candidate_recall"]["increment"]["count"],
            "exact_repeat": outcome["exact_repeat"]["passed"],
        }
    sys.stdout.buffer.write(_canonical_bytes(summary) + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
