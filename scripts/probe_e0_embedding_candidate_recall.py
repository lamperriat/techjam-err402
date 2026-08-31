"""Frozen BGE E0 candidate-recall one-shot runner.

This module is deliberately stdlib-only until every target-free trace,
resource, source and Git gate has passed.  The formal path preserves the
sealed variable-C200 as an ordered prefix, appends only a Dense-400 tail,
keeps the served Top10 unchanged, and joins outcomes only after both fresh
offline workers have closed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
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
import tempfile
import time
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


# This bootstrap is intentionally before any repository import.  It is the
# regression guard for the consumed C400 ``ModuleNotFoundError: evaluator``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True


SCHEMA_VERSION = "small-ranker-frozen-embedding-e0-outcome.v1"
WORKER_SCHEMA_VERSION = "small-ranker-frozen-embedding-e0-worker-summary.v1"
EXPERIMENT_ID = "SR-V2.18-FROZEN-EMBEDDING-E0-DENSE400"
BRANCH = "small-ranker-v2.18-frozen-embedding-e0"
REMOTE = "origin"
REMOTE_URL = "https://github.com/lamperriat/techjam-err402.git"
REMOTE_REF = f"refs/remotes/{REMOTE}/{BRANCH}"

BASE_COMMIT = "c94747edd890c56d4bf6a30edf86777f5868ded8"
PREREG_COMMIT = "63fbff0143eee14e6a86da5d4dbc759b7874243a"
INITIAL_IMPLEMENTATION_COMMIT = "8c79831f59b1e2f893e2529cdb3306b7dccaaf51"
PREREG_RELATIVE = "configs/small_ranker_v2_18.frozen_embedding_e0_preregistration.json"
PREREG_PATH = ROOT / PREREG_RELATIVE
PREREG_BLOB = "6732644bc9120ed147296a51458d6462a22ec60a"
PREREG_BYTES = 14_406
PREREG_RAW_SHA256 = "088aac976d3ae866e7f4a0f980904086dcf7ee89cd261ac66e74d38aa618fac0"
PREREG_CANONICAL_SHA256 = "9664cd1d07659bbcd087aadbd7cd2c6144112be8350aff2a212dafd69275ba08"
PREREG_PATHS = {PREREG_RELATIVE}
IMPLEMENTATION_PATHS = {
    "scripts/e0_embedding_candidate_worker.py",
    "scripts/probe_e0_embedding_candidate_recall.py",
    "tests/test_e0_embedding_candidate_recall.py",
}
PINNED_BLOBS = {
    "starter/agent.py": "421c6d43c598102b8fefb181b72bab5da4bf1294",
    "starter/coverage.py": "59a6507fef63afa0d9761323f5771a52741c811a",
    "starter/semantic.py": "926c3db3acecf433a4da0c4f83cb0b6d165511a3",
    "starter/p7_lab.py": "5a84ee82163314de28555aec0096f5abbb41f5ae",
    "scripts/c200_candidate_worker.py": "b94fddcf5a9b20ddde540f3f43ea9962982cb096",
    "scripts/probe_c200_candidate_recall.py": "0a57f63866683b476b9f49184673cf3154531911",
    "evaluator/local_evaluator.py": "7c808347b31ef3121a9cbc4810ac3eb325f950ba",
    "configs/p7_bge_small_en_v1_5.json": "d007e8a4dd5121109ea94dc250b778c29c7aa3ab",
    "third_party/bge-small-en-v1.5/LICENSE": "360931513aa6c02f933a403202afa99ac2c5bc88",
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
CONTEXT_TURNS = 20_000
CONTEXT_SHA256 = "f30a98700da5d480731fe7e82c87c40a22f06de290e069e20dc68f9fefecd20f"
C200_TRACE_PATHS = (
    C200_CACHE_ROOT / "replica_a.jsonl",
    C200_CACHE_ROOT / "replica_b.jsonl",
)
C200_TRACE_BYTES = 32_226_135
C200_TRACE_ROWS = 20_000
C200_TRACE_SHA256 = "a8589749376f48f019997a618481578dde36be4ca1fc723e8ed00056c23e40dc"
C200_CANDIDATE_CELLS = 2_425_785
C100_NORMALIZED_BYTES = 26_690_930
C100_NORMALIZED_SHA256 = "b22b035cb7789570f36db6c52256e5deb67f593f90cbbc5c334d48f2f0a01a67"

MODEL_SPEC_PATH = ROOT / "configs/p7_bge_small_en_v1_5.json"
MODEL_SPEC_BLOB = "d007e8a4dd5121109ea94dc250b778c29c7aa3ab"
MODEL_SPEC_BYTES = 11_976
MODEL_SPEC_RAW_SHA256 = "c27107edcab9f40f0aa8ba7b003434672e1c6c7fe7714b86f762768d7d3a4614"
MODEL_SPEC_CANONICAL_SHA256 = "e71d0cad480c89eac25ad2b276de9a4e7153e1ec2f3bdcc793682f183a592200"
LICENSE_BLOB = "360931513aa6c02f933a403202afa99ac2c5bc88"
LICENSE_BYTES = 1_065
LICENSE_SHA256 = "587a673933425dbc36ec61268d3b954051b2d3ef3c9b322ede357976055ffdd5"
MODEL_DIR = SOURCE_ROOT / "experiments/p7_assets/bge-small-en-v1.5"
INDEX_DIR = SOURCE_ROOT / "experiments/p7_index"
INDEX_MANIFEST_PATH = INDEX_DIR / "semantic-index.manifest.json"
INDEX_MANIFEST_BYTES = 9_474
INDEX_MANIFEST_SHA256 = "cca932a8b4d0a160e0a409ec6ce9cf3b68c99e3b95bddb911b9c7d83b67365ba"
INDEX_MATRIX_PATH = INDEX_DIR / "embeddings.npy"
INDEX_MATRIX_BYTES = 76_800_128
INDEX_MATRIX_SHA256 = "84897381c106b909b9e3d44229187d12f23796f108cfec97904db1cbeeb2d407"
INDEX_ASINS_PATH = INDEX_DIR / "parent_asins.txt"
INDEX_ASINS_BYTES = 550_000
INDEX_ASINS_SHA256 = "3af465b23ff2d33614501472edf02d2953ccfc170d2fe3348d55cd51c8ef0d54"
REQUIRED_ASSET_BYTES = 211_493_793

OUTPUT_PATH = ROOT / "experiments/fast_track/small_ranker_v2_18_frozen_embedding_e0_20260831.json"
CACHE_ROOT = ROOT / "experiments/fast_track/frozen_embedding_e0_cache_20260831"
TRACE_PATHS = (CACHE_ROOT / "replica_a.jsonl", CACHE_ROOT / "replica_b.jsonl")
WORKER_PATH = ROOT / "scripts/e0_embedding_candidate_worker.py"
RUNNER_PATH = Path(__file__).resolve()

EXPECTED_EXECUTABLE = Path(r"D:\450\conda\envs\tiktok\python.exe")
EXPECTED_PYTHON = "3.11.16"
EXPECTED_SQLITE = "3.53.4"
EXPECTED_PACKAGES = {"numpy": "2.4.6", "onnxruntime": "1.29.0", "tokenizers": "0.23.1"}
SESSION_COUNT = 2_000
TURN_COUNT = 10
RECORD_COUNT = SESSION_COUNT * TURN_COUNT
CUTOFFS = (10, 20, 50, 100, 200, 400)
EXPECTED_C200_RECALL = {10: 1_895, 20: 1_943, 50: 1_982, 100: 1_986, 200: 1_986}
EXPECTED_TAXONOMY = frozenset({"accessories-other", "clothing", "jewelry", "shoes"})

TOTAL_WALL_MAXIMUM = 3_600.0
COLD_INIT_MAXIMUM = 10.0
RESPOND_P95_MS_MAXIMUM = 400.0
DENSE_P95_MS_MAXIMUM = 100.0
WORKER_RSS_MAXIMUM = 1_610_612_736
WORKER_RSS_SUM_MAXIMUM = 3_221_225_472
ASSET_BYTES_MAXIMUM = 225_000_000
CELL_RATIO_C100_MAXIMUM = 4.0
CELL_RATIO_C200_MAXIMUM = 3.3
TRACE_RATIO_C100_MAXIMUM = 4.1
TRACE_RATIO_C200_MAXIMUM = 3.4

COMMIT_RE = re.compile(r"[0-9a-f]{40}")
ASIN_SHAPE_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE)
CATALOG_ID_RE = re.compile(r"[A-Z0-9]{10}")
FORBIDDEN_RESULT_KEYS = frozenset(
    {"asin", "parent_asin", "sample_id", "ground_truth", "target", "target_asin",
     "eligible_from", "per_session", "membership_vector", "candidates", "c200_candidates"}
)


class E0ProbeError(RuntimeError):
    """Raised when E0 evidence cannot be trusted."""


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


@dataclass(frozen=True)
class Preflight:
    environment: Mapping[str, Any]
    git: Mapping[str, Any]
    protocol: Mapping[str, Any]
    catalog_ids: frozenset[str]
    c200_reference: tuple[tuple[str, ...], ...]
    source_identities: Mapping[str, Any]
    asset_identities: Mapping[str, Any]
    entrypoint_checks: Mapping[str, Any]
    smoke: Mapping[str, Any]
    memory_before_receipt: tuple[int, int]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise E0ProbeError("duplicate JSON key")
        result[key] = value
    return result


def _snapshot(stat: os.stat_result) -> tuple[int, int, int]:
    return (int(stat.st_size), int(stat.st_mtime_ns), int(getattr(stat, "st_ino", 0)))


def _is_reparse(path: Path) -> bool:
    observed = path.lstat()
    flag = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(getattr(observed, "st_file_attributes", 0) & flag)


def _require_plain(path: Path, *, directory: bool = False) -> Path:
    anchor = Path(r"D:\tiktok").resolve(strict=True)
    absolute = path.absolute()
    try:
        absolute.relative_to(anchor)
    except ValueError as error:
        raise E0ProbeError("path escaped D:/tiktok") from error
    current = anchor
    for part in absolute.relative_to(anchor).parts:
        current /= part
        if current.exists() or current.is_symlink():
            if _is_reparse(current):
                raise E0ProbeError("path traverses a link or reparse point")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise E0ProbeError("required path is unavailable") from error
    if (directory and not resolved.is_dir()) or (not directory and not resolved.is_file()):
        raise E0ProbeError("required path has the wrong type")
    return resolved


def _file_identity(path: Path, label: str) -> FileIdentity:
    resolved = _require_plain(path)
    before = _snapshot(resolved.stat())
    digest = hashlib.sha256()
    byte_count = row_count = 0
    with resolved.open("rb") as handle:
        for line in handle:
            digest.update(line)
            byte_count += len(line)
            row_count += 1
    after = _snapshot(resolved.stat())
    if before != after or byte_count != before[0]:
        raise E0ProbeError(f"{label} changed while hashed")
    return FileIdentity(byte_count, row_count, digest.hexdigest(), after)


def _hash_file(path: Path, label: str) -> tuple[int, str]:
    identity = _file_identity(path, label)
    return identity.bytes, identity.sha256


def _git(*args: str, binary: bool = False) -> Any:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(["git", *args], cwd=ROOT, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               check=False)
    if completed.returncode:
        raise E0ProbeError("Git identity command failed")
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _changed_paths(commitish: str) -> set[str]:
    return set(filter(None, _git("diff-tree", "--no-commit-id", "--name-only", "-r", commitish).splitlines()))


def _diff_paths(left: str, right: str) -> set[str]:
    return set(filter(None, _git("diff", "--name-only", left, right).splitlines()))


def _validate_environment() -> dict[str, Any]:
    try:
        expected = EXPECTED_EXECUTABLE.resolve(strict=True)
        actual = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        raise E0ProbeError("formal executable is unavailable") from error
    packages = {}
    for name, version in EXPECTED_PACKAGES.items():
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise E0ProbeError("formal package inventory is incomplete") from error
        if packages[name] != version:
            raise E0ProbeError("formal package version drifted")
    if not (
        actual.as_posix().casefold() == expected.as_posix().casefold()
        and sys.version.split()[0] == EXPECTED_PYTHON
        and sqlite3.sqlite_version == EXPECTED_SQLITE
    ):
        raise E0ProbeError("formal interpreter identity drifted")
    return {"executable": actual.as_posix(), "python": EXPECTED_PYTHON,
            "sqlite": EXPECTED_SQLITE, "packages": packages, "top_level_stdlib_only": True,
            "network_attempt_count": 0, "gpu_peak_bytes": 0, "gpu_used": False}


def _load_preregistration() -> dict[str, Any]:
    if _git("rev-parse", f"{PREREG_COMMIT}:{PREREG_RELATIVE}") != PREREG_BLOB:
        raise E0ProbeError("preregistration blob drifted")
    raw = _git("cat-file", "blob", PREREG_BLOB, binary=True)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        worktree = json.loads(_require_plain(PREREG_PATH).read_text(encoding="utf-8"),
                              object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise E0ProbeError("preregistration JSON is invalid") from error
    if not (
        len(raw) == PREREG_BYTES
        and hashlib.sha256(raw).hexdigest() == PREREG_RAW_SHA256
        and _canonical_sha256(value) == PREREG_CANONICAL_SHA256
        and _canonical_sha256(worktree) == PREREG_CANONICAL_SHA256
        and value.get("schema_version") == "small-ranker-frozen-embedding-e0-preregistration.v1"
        and value.get("status") == "PREREGISTERED_BEFORE_E0_IMPLEMENTATION_AND_CURRENT_BENCHMARK_OUTCOME"
        and value.get("parent_commit") == BASE_COMMIT
    ):
        raise E0ProbeError("preregistration identity drifted")
    return value


def _validate_git_checkpoint(implementation_commit: str) -> dict[str, Any]:
    if not COMMIT_RE.fullmatch(implementation_commit):
        raise E0ProbeError("implementation commit is invalid")
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    pinned = {path: _git("rev-parse", f"HEAD:{path}") for path in PINNED_BLOBS}
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if not (
        head == implementation_commit
        and _git("rev-parse", "HEAD^") == INITIAL_IMPLEMENTATION_COMMIT
        and _git("rev-parse", f"{INITIAL_IMPLEMENTATION_COMMIT}^") == PREREG_COMMIT
        and _git("rev-parse", f"{PREREG_COMMIT}^") == BASE_COMMIT
        and branch == BRANCH
        and _git("remote", "get-url", REMOTE) == REMOTE_URL
        and _git("rev-parse", REMOTE_REF) == head
        and not status
        and _changed_paths(PREREG_COMMIT) == PREREG_PATHS
        and _changed_paths(INITIAL_IMPLEMENTATION_COMMIT) == IMPLEMENTATION_PATHS
        and _changed_paths(head) == IMPLEMENTATION_PATHS
        and _diff_paths(PREREG_COMMIT, head) == IMPLEMENTATION_PATHS
        and pinned == PINNED_BLOBS
    ):
        raise E0ProbeError("Git checkpoint gate failed")
    return {
        "branch": branch, "commit": head, "parent": INITIAL_IMPLEMENTATION_COMMIT,
        "implementation_chain": [INITIAL_IMPLEMENTATION_COMMIT, head],
        "preregistration_parent": PREREG_COMMIT,
        "remote_equal": True, "clean": True, "exact_changed_paths": True,
        "implementation_blobs": {
            path: _git("rev-parse", f"HEAD:{path}") for path in sorted(IMPLEMENTATION_PATHS)
        },
        "pinned_blobs_sha256": _canonical_sha256(pinned),
    }


def _load_catalog_ids() -> tuple[frozenset[str], FileIdentity]:
    identity = _file_identity(CATALOG_PATH, "catalog")
    if identity.report() != {"bytes": CATALOG_BYTES, "rows": CATALOG_ROWS, "sha256": CATALOG_SHA256}:
        raise E0ProbeError("catalog identity drifted")
    identifiers: set[str] = set()
    with CATALOG_PATH.open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            value = json.loads(raw, object_pairs_hook=_unique_object)
            identifier = value.get("parent_asin") if isinstance(value, dict) else None
            if (not isinstance(identifier, str) or not CATALOG_ID_RE.fullmatch(identifier)
                    or identifier != identifier.upper() or identifier in identifiers):
                raise E0ProbeError("catalog identifier surface drifted")
            identifiers.add(identifier)
    if len(identifiers) != CATALOG_ROWS:
        raise E0ProbeError("catalog row count drifted")
    return frozenset(identifiers), identity


def _canonical_c200_line(ordinal: int, turn: int, values: Sequence[str]) -> bytes:
    return _canonical_bytes({"c200": list(values), "ordinal": ordinal, "turn": turn}) + b"\n"


def _canonical_trace_line(ordinal: int, turn: int, values: Sequence[str]) -> bytes:
    return _canonical_bytes({"candidates": list(values), "ordinal": ordinal, "turn": turn}) + b"\n"


def _load_c200_reference(catalog_ids: frozenset[str]) -> tuple[tuple[tuple[str, ...], ...], dict[str, Any]]:
    canonical = {value: value for value in catalog_ids}
    references: list[tuple[str, ...]] = []
    reports = []
    for replica_path in C200_TRACE_PATHS:
        identity = _file_identity(replica_path, "sealed C200 trace")
        if identity.report() != {"bytes": C200_TRACE_BYTES, "rows": C200_TRACE_ROWS,
                                 "sha256": C200_TRACE_SHA256}:
            raise E0ProbeError("sealed C200 trace identity drifted")
        reports.append(identity.report())
    if not _files_equal(C200_TRACE_PATHS[0], C200_TRACE_PATHS[1]):
        raise E0ProbeError("sealed C200 replicas differ")
    cells = 0
    c100_digest = hashlib.sha256()
    c100_bytes = 0
    with C200_TRACE_PATHS[0].open("rb") as handle:
        for index, raw in enumerate(handle):
            ordinal, turn = index // TURN_COUNT + 1, index % TURN_COUNT + 1
            try:
                row = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise E0ProbeError("sealed C200 JSONL is invalid") from error
            values = row.get("c200") if isinstance(row, dict) else None
            if (not isinstance(values, list) or set(row) != {"c200", "ordinal", "turn"}
                    or row.get("ordinal") != ordinal or row.get("turn") != turn
                    or not 100 <= len(values) <= 200 or len(set(values)) != len(values)
                    or any(value not in canonical for value in values)):
                raise E0ProbeError("sealed C200 schema or candidates drifted")
            candidates = tuple(canonical[value] for value in values)
            if raw != _canonical_c200_line(ordinal, turn, candidates):
                raise E0ProbeError("sealed C200 row is not canonical LF JSON")
            references.append(candidates)
            cells += len(candidates)
            normalized = _canonical_bytes(
                {"c100": list(candidates[:100]), "ordinal": ordinal, "turn": turn}
            ) + b"\n"
            c100_digest.update(normalized)
            c100_bytes += len(normalized)
    if not (len(references) == RECORD_COUNT and cells == C200_CANDIDATE_CELLS
            and c100_bytes == C100_NORMALIZED_BYTES
            and c100_digest.hexdigest() == C100_NORMALIZED_SHA256):
        raise E0ProbeError("sealed C200 normalized identity drifted")
    return tuple(references), {
        "replicas": reports, "candidate_cells": cells,
        "normalized_c100_bytes": c100_bytes,
        "normalized_c100_sha256": c100_digest.hexdigest(),
    }


def _validate_assets() -> dict[str, Any]:
    model_root = _require_plain(MODEL_DIR, directory=True)
    index_root = _require_plain(INDEX_DIR, directory=True)
    if ".cache" in {part.casefold() for part in model_root.parts + index_root.parts}:
        raise E0ProbeError("frozen assets resolve through a cache alias")

    raw_spec = _git("cat-file", "blob", MODEL_SPEC_BLOB, binary=True)
    raw_license = _git("cat-file", "blob", LICENSE_BLOB, binary=True)
    try:
        spec = json.loads(raw_spec.decode("utf-8"), object_pairs_hook=_unique_object)
        worktree_spec = json.loads(_require_plain(MODEL_SPEC_PATH).read_text(encoding="utf-8"),
                                   object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise E0ProbeError("frozen model spec is invalid") from error
    if not (
        len(raw_spec) == MODEL_SPEC_BYTES
        and hashlib.sha256(raw_spec).hexdigest() == MODEL_SPEC_RAW_SHA256
        and _canonical_sha256(spec) == MODEL_SPEC_CANONICAL_SHA256
        and _canonical_sha256(worktree_spec) == MODEL_SPEC_CANONICAL_SHA256
        and len(raw_license) == LICENSE_BYTES
        and hashlib.sha256(raw_license).hexdigest() == LICENSE_SHA256
    ):
        raise E0ProbeError("Git-LF model specification or license drifted")

    required = spec.get("required_files") if isinstance(spec, dict) else None
    if not isinstance(required, list) or len(required) != 11:
        raise E0ProbeError("frozen model required-file manifest drifted")
    required_reports: list[dict[str, Any]] = []
    model_bytes = 0
    for item in required:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise E0ProbeError("frozen model required-file schema drifted")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise E0ProbeError("model asset escaped its frozen directory")
        byte_count, digest = _hash_file(model_root / relative, "model asset")
        if byte_count != item["bytes"] or digest != item["sha256"]:
            raise E0ProbeError("model asset identity drifted")
        model_bytes += byte_count
        required_reports.append({"path": relative.as_posix(), "bytes": byte_count, "sha256": digest})

    fixed_index = (
        (INDEX_MANIFEST_PATH, INDEX_MANIFEST_BYTES, INDEX_MANIFEST_SHA256, "manifest"),
        (INDEX_MATRIX_PATH, INDEX_MATRIX_BYTES, INDEX_MATRIX_SHA256, "matrix"),
        (INDEX_ASINS_PATH, INDEX_ASINS_BYTES, INDEX_ASINS_SHA256, "ordered_asins"),
    )
    index_reports: dict[str, Any] = {}
    for path, expected_bytes, expected_sha, label in fixed_index:
        byte_count, digest = _hash_file(path, f"index {label}")
        if byte_count != expected_bytes or digest != expected_sha:
            raise E0ProbeError("frozen semantic index identity drifted")
        index_reports[label] = {"bytes": byte_count, "sha256": digest}
    try:
        manifest = json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"),
                              object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise E0ProbeError("semantic index manifest is invalid") from error
    if not (
        manifest.get("schema_version") == "p7.semantic-index.v1"
        and manifest.get("catalog_sha256") == CATALOG_SHA256
        and manifest.get("rows") == CATALOG_ROWS
        and manifest.get("dimensions") == 384
        and manifest.get("model_spec_sha256") == MODEL_SPEC_CANONICAL_SHA256
        and manifest.get("asset_byte_scope", {}).get("required_asset_bytes") == REQUIRED_ASSET_BYTES
    ):
        raise E0ProbeError("semantic index manifest semantics drifted")
    total = model_bytes + INDEX_MANIFEST_BYTES + INDEX_MATRIX_BYTES + INDEX_ASINS_BYTES + LICENSE_BYTES
    if total != REQUIRED_ASSET_BYTES or total > ASSET_BYTES_MAXIMUM:
        raise E0ProbeError("frozen asset byte budget drifted")
    return {
        "model_spec_git_lf": {"blob": MODEL_SPEC_BLOB, "bytes": MODEL_SPEC_BYTES,
                              "sha256": MODEL_SPEC_RAW_SHA256,
                              "canonical_sha256": MODEL_SPEC_CANONICAL_SHA256},
        "license_git_lf": {"blob": LICENSE_BLOB, "bytes": LICENSE_BYTES,
                           "sha256": LICENSE_SHA256},
        "model_files_bundle_sha256": _canonical_sha256(required_reports),
        "model_required_file_count": len(required_reports),
        "index": index_reports,
        "required_asset_bytes": total,
        "path_policy_passed": True,
    }


def _offline_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if key.casefold() not in {"pythonpath", "pythonhome"}
    }
    environment.update({
        "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
    })
    return environment


def _entrypoint_check_commands(
    required_module: str = "evaluator.local_evaluator",
) -> tuple[list[str], list[str]]:
    common = ["--entrypoint-self-check", "--require-module", required_module]
    direct = [str(EXPECTED_EXECUTABLE), "-B", str(RUNNER_PATH), *common]
    module = [str(EXPECTED_EXECUTABLE), "-B", "-m",
              "scripts.probe_e0_embedding_candidate_recall", *common]
    return direct, module


def _run_subprocess(command: Sequence[str], *, timeout: float,
                    cwd: Path = ROOT, environment: Mapping[str, str] | None = None,
                    ) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(list(command), cwd=cwd,
                          env=dict(environment) if environment is not None else _offline_environment(),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False, timeout=timeout)


def _validate_self_check(completed: subprocess.CompletedProcess[bytes], mode: str) -> dict[str, Any]:
    if completed.returncode != 0:
        raise E0ProbeError(f"{mode} entrypoint self-check failed")
    try:
        value = json.loads(completed.stdout.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise E0ProbeError("entrypoint self-check receipt is invalid") from error
    if not (isinstance(value, dict) and value.get("status") == "ENTRYPOINT_SELF_CHECK_PASS"
            and value.get("required_module") == "evaluator.local_evaluator"
            and value.get("project_root_bootstrapped") is True):
        raise E0ProbeError("entrypoint self-check semantics drifted")
    return {"mode": mode, "module": value["required_module"], "passed": True}


def _verify_entrypoints_before_receipt() -> dict[str, Any]:
    """Prove both invocation forms import evaluator, while a missing import fails."""

    direct, module = _entrypoint_check_commands()
    completed: dict[str, subprocess.CompletedProcess[bytes]] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_run_subprocess, direct, timeout=60.0): "direct-script",
            executor.submit(_run_subprocess, module, timeout=60.0): "python-module",
        }
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
    checks = [_validate_self_check(completed[name], name)
              for name in ("direct-script", "python-module")]

    # An additional adversarial cwd proves the direct script does not succeed
    # merely because the repository root happened to be the current directory.
    adversarial = _run_subprocess(direct, timeout=60.0, cwd=ROOT.parent)
    adversarial_check = _validate_self_check(adversarial, "direct-script-outside-repo-cwd")

    missing_name = "e0_intentionally_missing_module_for_pre_receipt_gate"
    missing_direct, _ = _entrypoint_check_commands(missing_name)
    missing = _run_subprocess(missing_direct, timeout=60.0, cwd=ROOT.parent)
    if missing.returncode == 0:
        raise E0ProbeError("intentionally missing module did not fail closed")
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise E0ProbeError("entrypoint checks created the formal receipt")
    return {
        "direct_script": checks[0], "python_module": checks[1],
        "adversarial_direct_script": adversarial_check,
        "missing_module_failed_closed": True,
        "formal_executable": str(EXPECTED_EXECUTABLE), "cwd": ROOT.as_posix(),
        "adversarial_cwd_was_project_root": False,
        "receipt_created": False,
    }


def _entrypoint_self_check(required_module: str) -> dict[str, Any]:
    if not required_module or not isinstance(required_module, str):
        raise E0ProbeError("required module is invalid")
    importlib.import_module(required_module)
    return {
        "status": "ENTRYPOINT_SELF_CHECK_PASS",
        "required_module": required_module,
        "project_root_bootstrapped": str(PROJECT_ROOT) in sys.path,
    }


def _validate_candidates(value: object, catalog_ids: frozenset[str], *,
                         minimum: int, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not minimum <= len(value) <= maximum:
        raise E0ProbeError("candidate sequence length drifted")
    result = tuple(value)
    if len(set(result)) != len(result) or any(
        not isinstance(identifier, str) or identifier not in catalog_ids for identifier in result
    ):
        raise E0ProbeError("candidate sequence surface drifted")
    return result


def validate_trace_records(
    records: Sequence[Mapping[str, Any]],
    c200_reference: Sequence[Sequence[str]],
    catalog_ids: frozenset[str],
    *,
    expected_records: int | None = None,
) -> TraceValidation:
    expected = len(c200_reference) if expected_records is None else expected_records
    if expected <= 0 or len(records) != expected or len(c200_reference) < expected:
        raise E0ProbeError("E0 trace record count drifted")
    canonical = hashlib.sha256()
    byte_count = 0
    lengths: list[int] = []
    c200_lengths: list[int] = []
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        ordinal, turn = index // TURN_COUNT + 1, index % TURN_COUNT + 1
        if (not isinstance(row, Mapping) or set(row) != {"candidates", "ordinal", "turn"}
                or row.get("ordinal") != ordinal or isinstance(row.get("ordinal"), bool)
                or row.get("turn") != turn or isinstance(row.get("turn"), bool)):
            raise E0ProbeError("E0 trace schema or order drifted")
        candidates = _validate_candidates(row.get("candidates"), catalog_ids,
                                          minimum=100, maximum=400)
        prefix = tuple(c200_reference[index])
        if len(prefix) < 100 or len(prefix) > 200 or candidates[:len(prefix)] != prefix:
            raise E0ProbeError("sealed variable-C200 is not the exact ordered prefix")
        normalized_row = {"candidates": list(candidates), "ordinal": ordinal, "turn": turn}
        payload = _canonical_trace_line(ordinal, turn, candidates)
        canonical.update(payload)
        byte_count += len(payload)
        lengths.append(len(candidates))
        c200_lengths.append(len(prefix))
        normalized.append(normalized_row)
    return TraceValidation(tuple(normalized), tuple(lengths), tuple(c200_lengths),
                           canonical.hexdigest(), byte_count, expected)


def load_and_validate_e0_trace(
    path: Path,
    c200_reference: Sequence[Sequence[str]],
    catalog_ids: frozenset[str],
    *,
    expected_records: int | None = None,
    retain_records: bool = True,
) -> TraceValidation:
    expected = len(c200_reference) if expected_records is None else expected_records
    resolved = _require_plain(path)
    before = _snapshot(resolved.stat())
    raw_digest = hashlib.sha256()
    byte_count = 0
    records: list[dict[str, Any]] = []
    with resolved.open("rb") as handle:
        for raw in handle:
            if len(records) >= expected or not raw.strip():
                raise E0ProbeError("E0 trace framing drifted")
            try:
                row = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise E0ProbeError("E0 trace JSONL is invalid") from error
            index = len(records)
            ordinal, turn = index // TURN_COUNT + 1, index % TURN_COUNT + 1
            values = row.get("candidates") if isinstance(row, dict) else ()
            if raw != _canonical_trace_line(ordinal, turn, values if isinstance(values, list) else ()):
                raise E0ProbeError("E0 trace is not canonical LF JSON")
            records.append(row)
            raw_digest.update(raw)
            byte_count += len(raw)
    after = _snapshot(resolved.stat())
    if before != after or byte_count != before[0]:
        raise E0ProbeError("E0 trace changed while read")
    validated = validate_trace_records(records, c200_reference, catalog_ids,
                                       expected_records=expected)
    if (raw_digest.hexdigest() != validated.canonical_trace_sha256
            or byte_count != validated.canonical_trace_bytes):
        raise E0ProbeError("E0 raw and canonical trace identities differ")
    if retain_records:
        return validated
    return TraceValidation((), validated.lengths, validated.c200_lengths,
                           validated.canonical_trace_sha256,
                           validated.canonical_trace_bytes, validated.record_count)


def _files_equal(left: Path, right: Path) -> bool:
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            first, second = a.read(1 << 20), b.read(1 << 20)
            if first != second:
                return False
            if not first:
                return True


def candidate_recall_flags(
    target: str,
    eligible_from: int,
    turns: Sequence[Mapping[str, Any]],
    cutoffs: Sequence[int] = CUTOFFS,
    *,
    baseline_lengths: Sequence[int] | None = None,
) -> dict[int, bool]:
    if (not isinstance(target, str) or not target or not isinstance(eligible_from, int)
            or isinstance(eligible_from, bool) or not 1 <= eligible_from <= TURN_COUNT
            or tuple(cutoffs) != CUTOFFS):
        raise E0ProbeError("candidate recall input is invalid")
    if baseline_lengths is not None and (
        len(baseline_lengths) != len(turns)
        or any(not isinstance(value, int) or isinstance(value, bool) or not 100 <= value <= 200
               for value in baseline_lengths)
    ):
        raise E0ProbeError("variable-C200 baseline lengths are invalid")
    flags = {cutoff: False for cutoff in CUTOFFS}
    for index, row in enumerate(turns):
        turn = row.get("turn") if isinstance(row, Mapping) else None
        candidates = row.get("candidates") if isinstance(row, Mapping) else None
        if not isinstance(turn, int) or isinstance(turn, bool) or not isinstance(candidates, (list, tuple)):
            raise E0ProbeError("candidate recall trace row is invalid")
        if turn >= eligible_from:
            for cutoff in CUTOFFS:
                # C200 is the complete variable-length sealed sparse prefix,
                # not the first 200 positions of the expanded E0 union.  When
                # the prefix has length 100, Dense rank 101 must be an E0/C400
                # rescue rather than being misreported as baseline C200 recall.
                limit = baseline_lengths[index] if cutoff == 200 and baseline_lengths is not None else cutoff
                flags[cutoff] = flags[cutoff] or target in candidates[:limit]
    return flags


def _recall_view(flags: Sequence[Mapping[int, bool]], indices: Sequence[int]) -> dict[str, Any]:
    denominator = len(indices)
    return {
        f"c{cutoff}": {
            "count": sum(int(flags[index][cutoff]) for index in indices),
            "fraction": round(sum(int(flags[index][cutoff]) for index in indices) / denominator, 6)
            if denominator else 0.0,
        }
        for cutoff in CUTOFFS
    }


def aggregate_candidate_recall(
    flags: Sequence[Mapping[int, bool]], *, outer_fold: Sequence[Any],
    family_index: Sequence[Any], taxonomy: Sequence[str],
) -> dict[str, Any]:
    count = len(flags)
    if not count or not (len(outer_fold) == len(family_index) == len(taxonomy) == count):
        raise E0ProbeError("candidate recall aggregate dimensions drifted")
    if any(set(row) != set(CUTOFFS) or any(type(row[key]) is not bool for key in CUTOFFS)
           for row in flags):
        raise E0ProbeError("candidate recall flag schema drifted")
    folds = [int(value) for value in outer_fold]
    families = [int(value) for value in family_index]
    if any(not 0 <= value < 5 for value in folds) or any(value < 0 for value in families):
        raise E0ProbeError("fold or family labels drifted")
    family_fold: dict[int, int] = {}
    for family, fold in zip(families, folds, strict=True):
        if family in family_fold and family_fold[family] != fold:
            raise E0ProbeError("one family crosses outer folds")
        family_fold[family] = fold
    indices = list(range(count))
    frontier = [index for index in indices if not flags[index][200]]
    increment = [index for index in frontier if flags[index][400]]
    by_fold = []
    for fold in sorted(set(folds)):
        members = [index for index, value in enumerate(folds) if value == fold]
        by_fold.append({"fold": fold, "sessions": len(members),
                        "recall": _recall_view(flags, members),
                        "increment": sum(index in increment for index in members)})
    by_taxonomy = {}
    for name in sorted(set(map(str, taxonomy))):
        members = [index for index, value in enumerate(taxonomy) if str(value) == name]
        by_taxonomy[name] = {"sessions": len(members), "recall": _recall_view(flags, members),
                             "increment": sum(index in increment for index in members)}
    family_members: dict[int, list[int]] = {}
    for index, family in enumerate(families):
        family_members.setdefault(family, []).append(index)
    target_uniform = {"cluster_count": len(family_members)}
    for cutoff in CUTOFFS:
        rates = [statistics.fmean(int(flags[index][cutoff]) for index in members)
                 for members in family_members.values()]
        target_uniform[f"c{cutoff}"] = {"fraction": round(statistics.fmean(rates), 6)}
    return {
        "all_sessions": _recall_view(flags, indices),
        "c200_absent_frontier": {"sessions": len(frontier), **_recall_view(flags, frontier)},
        "increment": {
            "count": len(increment), "fraction": round(len(increment) / count, 6),
            "frontier_fraction": round(len(increment) / len(frontier), 6) if frontier else 0.0,
            "target_cluster_count": len({families[index] for index in increment}),
            "outer_fold_span": len({folds[index] for index in increment}),
            "taxonomy_span": len({str(taxonomy[index]) for index in increment}),
            "non_clothing_count": sum(str(taxonomy[index]) != "clothing" for index in increment),
            "first_frontier": "after_complete_variable_c200_prefix_to_union_cap400",
        },
        "by_outer_fold": by_fold, "target_uniform": target_uniform,
        "by_taxonomy": by_taxonomy,
        "family_disjoint_audit": {"valid": True, "family_count": len(family_members),
                                  "families_crossing_outer_folds": 0},
    }


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise E0ProbeError("cannot summarize an empty measurement")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def inflation_summary(lengths: Sequence[int], c200_lengths: Sequence[int], *,
                      trace_bytes: int) -> dict[str, Any]:
    if (not lengths or len(lengths) != len(c200_lengths)
            or any(not 100 <= value <= 400 for value in lengths)
            or any(not 100 <= value <= 200 for value in c200_lengths)
            or any(new < old for new, old in zip(lengths, c200_lengths, strict=True))):
        raise E0ProbeError("candidate inflation surface is invalid")
    cells = sum(lengths)
    return {
        "records": len(lengths), "candidate_cells": cells,
        "sealed_c200_candidate_cells": sum(c200_lengths),
        "length_min": min(lengths), "length_p50": int(_nearest_rank(lengths, .50)),
        "length_p95": int(_nearest_rank(lengths, .95)), "length_max": max(lengths),
        "candidate_cell_ratio_over_c100": round(cells / (len(lengths) * 100), 6),
        "candidate_cell_ratio_over_c200": round(cells / sum(c200_lengths), 6),
        "trace_byte_ratio_over_c100": round(trace_bytes / C100_NORMALIZED_BYTES, 6),
        "trace_byte_ratio_over_c200": round(trace_bytes / C200_TRACE_BYTES, 6),
    }


def _worker_command(*, mode: str, nonce: str, reference: Path, trace: Path,
                    session_limit: int) -> list[str]:
    if mode == "direct":
        prefix = [str(EXPECTED_EXECUTABLE), "-B", str(WORKER_PATH)]
    elif mode == "module":
        prefix = [str(EXPECTED_EXECUTABLE), "-B", "-m", "scripts.e0_embedding_candidate_worker"]
    else:
        raise E0ProbeError("worker invocation mode is invalid")
    return [
        *prefix,
        "--nonce", nonce,
        "--catalog", str(CATALOG_PATH),
        "--catalog-bytes", str(CATALOG_BYTES),
        "--catalog-rows", str(CATALOG_ROWS),
        "--catalog-sha256", CATALOG_SHA256,
        "--context", str(CONTEXT_PATH),
        "--context-bytes", str(CONTEXT_BYTES),
        "--context-rows", str(CONTEXT_ROWS),
        "--context-turns", str(CONTEXT_TURNS),
        "--context-sha256", CONTEXT_SHA256,
        "--c200-reference", str(reference),
        "--c200-reference-bytes", str(C200_TRACE_BYTES),
        "--c200-reference-rows", str(C200_TRACE_ROWS),
        "--c200-reference-sha256", C200_TRACE_SHA256,
        "--model-spec", str(MODEL_SPEC_PATH),
        "--model-spec-canonical-sha256", MODEL_SPEC_CANONICAL_SHA256,
        "--model-dir", str(MODEL_DIR),
        "--index-dir", str(INDEX_DIR),
        "--index-manifest-bytes", str(INDEX_MANIFEST_BYTES),
        "--index-manifest-sha256", INDEX_MANIFEST_SHA256,
        "--index-matrix-bytes", str(INDEX_MATRIX_BYTES),
        "--index-matrix-sha256", INDEX_MATRIX_SHA256,
        "--index-asins-bytes", str(INDEX_ASINS_BYTES),
        "--index-asins-sha256", INDEX_ASINS_SHA256,
        "--trace-output", str(trace),
        "--session-limit", str(session_limit),
    ]


def _finite(value: object, label: str) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or float(value) < 0):
        raise E0ProbeError(f"worker {label} is invalid")
    return float(value)


def _p95(summary: object, label: str) -> float:
    if not isinstance(summary, Mapping):
        raise E0ProbeError(f"worker {label} summary is absent")
    return _finite(summary.get("p95_milliseconds"), f"{label} p95")


def _validate_worker_receipt(payload: bytes, *, nonce: str,
                             session_limit: int) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise E0ProbeError("worker receipt is invalid JSON") from error
    if payload != _canonical_bytes(value) + b"\n" or not isinstance(value, dict):
        raise E0ProbeError("worker receipt is not canonical LF JSON")
    summary = value.get("summary")
    if not isinstance(summary, Mapping):
        raise E0ProbeError("worker summary is absent")
    environment = summary.get("environment")
    configuration = summary.get("configuration")
    lifecycle = summary.get("lifecycle")
    assets = summary.get("asset_identities")
    latency = summary.get("latency")
    resources = summary.get("resources")
    expected_records = session_limit * TURN_COUNT
    if not (
        set(value) == {"kind", "nonce", "trace_sha256", "trace_bytes", "record_count", "summary"}
        and value.get("kind") == "receipt" and value.get("nonce") == nonce
        and value.get("record_count") == expected_records
        and isinstance(value.get("trace_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["trace_sha256"])
        and isinstance(value.get("trace_bytes"), int) and value["trace_bytes"] > 0
        and summary.get("schema_version") == WORKER_SCHEMA_VERSION
        and summary.get("session_limit") == session_limit
        and summary.get("processed_sessions") == session_limit
        and summary.get("processed_turns") == expected_records
        and isinstance(environment, Mapping)
        and environment.get("provider") == "CPUExecutionProvider"
        and environment.get("network_attempt_count") == 0
        and environment.get("gpu_used") is False and environment.get("gpu_peak_bytes") == 0
        and isinstance(configuration, Mapping)
        and configuration.get("dense_depth") == 400
        and configuration.get("diagnostic_only") is True
        and configuration.get("served_top10_unchanged") is True
        and configuration.get("stable_append_after_complete_variable_c200") is True
        and isinstance(lifecycle, Mapping) and lifecycle
        and all(value is True for value in lifecycle.values())
        and isinstance(assets, Mapping)
        and assets.get("model_spec_canonical_sha256") == MODEL_SPEC_CANONICAL_SHA256
        and assets.get("required_asset_bytes") == REQUIRED_ASSET_BYTES
        and isinstance(latency, Mapping) and isinstance(resources, Mapping)
    ):
        raise E0ProbeError("worker receipt contract drifted")
    cold = _finite(latency.get("cold_semantic_initialization_seconds"), "cold init")
    respond_p95 = _p95(latency.get("respond"), "respond")
    dense_p95 = _p95(latency.get("dense_query_and_exact_search"), "dense")
    rss = resources.get("peak_working_set_bytes")
    wall = _finite(resources.get("wall_seconds"), "wall")
    if (not isinstance(rss, int) or isinstance(rss, bool) or not 0 < rss <= WORKER_RSS_MAXIMUM
            or cold > COLD_INIT_MAXIMUM or respond_p95 > RESPOND_P95_MS_MAXIMUM
            or dense_p95 > DENSE_P95_MS_MAXIMUM or wall > TOTAL_WALL_MAXIMUM):
        raise E0ProbeError("worker resource budget failed")
    return dict(value)


def _run_worker(*, mode: str, nonce: str, reference: Path, trace: Path,
                session_limit: int) -> dict[str, Any]:
    started = time.perf_counter()
    completed = _run_subprocess(
        _worker_command(mode=mode, nonce=nonce, reference=reference,
                        trace=trace, session_limit=session_limit),
        timeout=TOTAL_WALL_MAXIMUM,
    )
    if completed.returncode != 0:
        raise E0ProbeError(f"{mode} E0 worker failed")
    receipt = _validate_worker_receipt(completed.stdout, nonce=nonce,
                                       session_limit=session_limit)
    return {"receipt": receipt, "wall_seconds": round(time.perf_counter() - started, 6),
            "mode": mode}


def _bind_worker_receipt(result: Mapping[str, Any], trace: TraceValidation) -> None:
    receipt = result.get("receipt")
    summary = receipt.get("summary") if isinstance(receipt, Mapping) else None
    pool = summary.get("pool_lengths") if isinstance(summary, Mapping) else None
    control = summary.get("control_pool_lengths") if isinstance(summary, Mapping) else None
    if not (
        isinstance(receipt, Mapping) and receipt.get("trace_sha256") == trace.canonical_trace_sha256
        and receipt.get("trace_bytes") == trace.canonical_trace_bytes
        and receipt.get("record_count") == trace.record_count
        and isinstance(pool, Mapping) and pool.get("candidate_cells") == sum(trace.lengths)
        and isinstance(control, Mapping) and control.get("candidate_cells") == sum(trace.c200_lengths)
    ):
        raise E0ProbeError("worker receipt does not bind to its closed trace")


def _worker_pair(*, trace_paths: Sequence[Path], session_limit: int,
                 implementation_commit: str) -> dict[str, dict[str, Any]]:
    if len(trace_paths) != 2 or any(path.exists() or path.is_symlink() for path in trace_paths):
        raise E0ProbeError("worker trace outputs are not exclusive")
    specs = (
        ("replica_a", "direct", C200_TRACE_PATHS[0], trace_paths[0]),
        ("replica_b", "module", C200_TRACE_PATHS[1], trace_paths[1]),
    )
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for name, mode, reference, trace in specs:
            nonce = hashlib.sha256(
                f"{EXPERIMENT_ID}:{implementation_commit}:{session_limit}:{name}".encode()
            ).hexdigest()[:32]
            futures[executor.submit(_run_worker, mode=mode, nonce=nonce,
                                    reference=reference, trace=trace,
                                    session_limit=session_limit)] = name
        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if set(results) != {"replica_a", "replica_b"}:
        raise E0ProbeError("one fresh worker result is missing")
    return results


def _smoke_workers(catalog_ids: frozenset[str],
                   references: Sequence[Sequence[str]],
                   implementation_commit: str) -> dict[str, Any]:
    parent = _require_plain(ROOT / "experiments/fast_track", directory=True)
    with tempfile.TemporaryDirectory(prefix="e0_preflight_", dir=parent) as temporary:
        temp_root = Path(temporary)
        traces = (temp_root / "direct.jsonl", temp_root / "module.jsonl")
        results = _worker_pair(trace_paths=traces, session_limit=2,
                               implementation_commit=implementation_commit)
        reference = references[:2 * TURN_COUNT]
        first = load_and_validate_e0_trace(traces[0], reference, catalog_ids,
                                           expected_records=20)
        second = load_and_validate_e0_trace(traces[1], reference, catalog_ids,
                                            expected_records=20, retain_records=False)
        _bind_worker_receipt(results["replica_a"], first)
        _bind_worker_receipt(results["replica_b"], second)
        if not (
            first.canonical_trace_sha256 == second.canonical_trace_sha256
            and first.canonical_trace_bytes == second.canonical_trace_bytes
            and first.lengths == second.lengths
            and first.c200_lengths == second.c200_lengths
            and _files_equal(traces[0], traces[1])
        ):
            raise E0ProbeError("direct-script/module smoke traces differ")
        digest, byte_count = first.canonical_trace_sha256, first.canonical_trace_bytes
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise E0ProbeError("target-free smoke created the formal receipt")
    return {
        "session_limit": 2, "record_count": 20, "trace_sha256": digest,
        "trace_bytes": byte_count, "direct_module_exact_repeat": True,
        "sealed_c200_prefix": True, "served_top10_unchanged": True,
        "receipt_created": False,
    }


def _process_memory() -> tuple[int, int]:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                          ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return (1, 1)


def _assert_fresh_formal_outputs() -> None:
    _require_plain(OUTPUT_PATH.parent, directory=True)
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise E0ProbeError("formal E0 receipt already exists; rerun is forbidden")
    if CACHE_ROOT.exists() or CACHE_ROOT.is_symlink():
        raise E0ProbeError("formal E0 cache already exists; rerun is forbidden")


def preflight_only(implementation_commit: str) -> Preflight:
    """Run the complete target-free pre-receipt gate, including both entrypoints."""

    _assert_fresh_formal_outputs()
    environment = _validate_environment()
    protocol = _load_preregistration()
    git = _validate_git_checkpoint(implementation_commit)
    catalog_ids, catalog_identity = _load_catalog_ids()
    context_identity = _file_identity(CONTEXT_PATH, "visible context")
    if context_identity.report() != {"bytes": CONTEXT_BYTES, "rows": CONTEXT_ROWS,
                                     "sha256": CONTEXT_SHA256}:
        raise E0ProbeError("visible context identity drifted")
    references, reference_report = _load_c200_reference(catalog_ids)
    assets = _validate_assets()
    entrypoints = _verify_entrypoints_before_receipt()
    smoke = _smoke_workers(catalog_ids, references, implementation_commit)
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise E0ProbeError("preflight created a formal receipt")
    return Preflight(
        environment=environment,
        git=git,
        protocol={"schema_version": protocol["schema_version"],
                  "canonical_sha256": PREREG_CANONICAL_SHA256,
                  "commit": PREREG_COMMIT, "target_free": True},
        catalog_ids=catalog_ids,
        c200_reference=references,
        source_identities={"catalog": catalog_identity.report(),
                           "visible_context": context_identity.report(),
                           "sealed_c200": reference_report},
        asset_identities=assets,
        entrypoint_checks=entrypoints,
        smoke=smoke,
        memory_before_receipt=_process_memory(),
    )


def _write_descriptor(descriptor: int, value: object) -> tuple[int, str]:
    payload = _canonical_bytes(value) + b"\n"
    if len(payload) >= 10_000:
        raise E0ProbeError("compact receipt exceeds the preregistered size boundary")
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


def _pending_receipt(implementation_commit: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "PENDING_ONE_SHOT_CONSUMED",
        "implementation_commit": implementation_commit,
        "preregistration_commit": PREREG_COMMIT,
        "recorded_on": "2026-08-31",
        "rerun_forbidden": True,
    }


def _open_receipt(implementation_commit: str) -> int:
    parent = _require_plain(OUTPUT_PATH.parent, directory=True)
    if OUTPUT_PATH.parent.resolve(strict=True) != parent:
        raise E0ProbeError("receipt parent identity drifted")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(str(OUTPUT_PATH), flags, 0o600)
    except OSError as error:
        raise E0ProbeError("exclusive formal receipt creation failed") from error
    try:
        _write_descriptor(descriptor, _pending_receipt(implementation_commit))
        return descriptor
    except BaseException as error:
        try:
            _write_invalid_receipt(
                descriptor, implementation_commit, error, phase="pending_receipt_write"
            )
        except BaseException:
            pass
        raise


def _write_invalid_receipt(descriptor: int, implementation_commit: str,
                           error: BaseException, *, phase: str) -> None:
    invalid = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "INVALID_ONE_SHOT_CONSUMED",
        "phase": phase,
        "error_type": type(error).__name__,
        "implementation_commit": implementation_commit,
        "preregistration_commit": PREREG_COMMIT,
        "recorded_on": "2026-08-31",
        "rerun_forbidden": True,
        "algorithm_interpretation": "implementation_or_integrity_failure; not an algorithm No-Go",
    }
    try:
        for attempt in range(2):
            try:
                _write_descriptor(descriptor, invalid)
                return
            except BaseException as seal_error:
                if attempt == 0:
                    continue
                raise E0ProbeError("INVALID receipt could not be durably sealed") from seal_error
    finally:
        os.close(descriptor)


def _prepare_cache_root() -> None:
    parent = _require_plain(CACHE_ROOT.parent, directory=True)
    if CACHE_ROOT.exists() or CACHE_ROOT.is_symlink():
        raise E0ProbeError("exclusive formal cache already exists")
    try:
        os.mkdir(CACHE_ROOT)
    except OSError as error:
        raise E0ProbeError("exclusive formal cache creation failed") from error
    if _is_reparse(CACHE_ROOT) or CACHE_ROOT.resolve(strict=True).parent != parent:
        raise E0ProbeError("formal cache root is unsafe")


def _result_privacy_scan(value: object, *, catalog_ids: Iterable[str] = ()) -> None:
    catalog = {str(identifier).casefold() for identifier in catalog_ids}

    def walk(item: object) -> Iterable[object]:
        yield item
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).casefold() in FORBIDDEN_RESULT_KEYS:
                    raise E0ProbeError("result contains an identity-bearing key")
                yield from walk(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            if len(item) >= SESSION_COUNT:
                raise E0ProbeError("result contains a session-length vector")
            for child in item:
                yield from walk(child)

    for item in walk(value):
        if isinstance(item, str) and (ASIN_SHAPE_RE.search(item) or item.casefold() in catalog):
            raise E0ProbeError("result contains a catalog identifier")
        if item.__class__.__module__.startswith("numpy"):
            raise E0ProbeError("result contains a numeric array or scalar")


def _rehash_target_free(preflight: Preflight) -> dict[str, Any]:
    catalog = _file_identity(CATALOG_PATH, "catalog rehash")
    context = _file_identity(CONTEXT_PATH, "context rehash")
    references = [_file_identity(path, "C200 rehash").report() for path in C200_TRACE_PATHS]
    assets = _validate_assets()
    if not (
        catalog.report() == preflight.source_identities["catalog"]
        and context.report() == preflight.source_identities["visible_context"]
        and references == preflight.source_identities["sealed_c200"]["replicas"]
        and _canonical_sha256(assets) == _canonical_sha256(preflight.asset_identities)
    ):
        raise E0ProbeError("target-free source changed during the probe")
    return {
        "catalog_sha256": catalog.sha256, "context_sha256": context.sha256,
        "sealed_c200_sha256": references[0]["sha256"],
        "asset_manifest_sha256": _canonical_sha256(assets),
        "worker_blob": _git("rev-parse", "HEAD:scripts/e0_embedding_candidate_worker.py"),
        "runner_blob": _git("rev-parse", "HEAD:scripts/probe_e0_embedding_candidate_recall.py"),
    }


def _flags_from_trace(trace: TraceValidation, targets: Sequence[str],
                      eligibility: Sequence[int]) -> list[dict[int, bool]]:
    if len(targets) != SESSION_COUNT or len(eligibility) != SESSION_COUNT:
        raise E0ProbeError("target membership dimensions drifted")
    if len(trace.records) != RECORD_COUNT:
        raise E0ProbeError("retained E0 trace is incomplete")
    return [
        candidate_recall_flags(
            targets[index], int(eligibility[index]),
            trace.records[index * TURN_COUNT:(index + 1) * TURN_COUNT],
            baseline_lengths=trace.c200_lengths[
                index * TURN_COUNT:(index + 1) * TURN_COUNT
            ],
        )
        for index in range(SESSION_COUNT)
    ]


def _compact_aggregate(value: Mapping[str, Any]) -> dict[str, Any]:
    def two_cutoffs(view: Mapping[str, Any]) -> dict[str, Any]:
        return {"c200": view["c200"], "c400": view["c400"]}

    return {
        "all_sessions": value["all_sessions"],
        "c200_absent_frontier": {
            "sessions": value["c200_absent_frontier"]["sessions"],
            **two_cutoffs(value["c200_absent_frontier"]),
        },
        "increment": value["increment"],
        "by_outer_fold": [
            {"fold": row["fold"], "sessions": row["sessions"],
             **two_cutoffs(row["recall"]), "increment": row["increment"]}
            for row in value["by_outer_fold"]
        ],
        "by_taxonomy": {
            name: {"sessions": row["sessions"], **two_cutoffs(row["recall"]),
                   "increment": row["increment"]}
            for name, row in value["by_taxonomy"].items()
        },
        "target_uniform": {
            "cluster_count": value["target_uniform"]["cluster_count"],
            "c200": value["target_uniform"]["c200"],
            "c400": value["target_uniform"]["c400"],
        },
        "family_disjoint_audit": value["family_disjoint_audit"],
    }


def run(implementation_commit: str) -> dict[str, Any]:
    """Consume the unique formal E0 one-shot after every pre-receipt check."""

    formal_started = time.perf_counter()
    preflight = preflight_only(implementation_commit)
    # Kept explicit so tests can prove a failed direct/module import cannot call
    # _open_receipt even when preflight_only itself is substituted.
    final_entrypoint_check = _verify_entrypoints_before_receipt()
    descriptor: int | None = None
    proxy_source: Any | None = None
    phase = "receipt_initialization"
    try:
        descriptor = _open_receipt(implementation_commit)
        phase = "exclusive_cache_and_two_fresh_full_workers"
        _prepare_cache_root()
        worker_results = _worker_pair(trace_paths=TRACE_PATHS, session_limit=SESSION_COUNT,
                                      implementation_commit=implementation_commit)

        phase = "closed_trace_prefix_top10_exact_repeat_gates"
        trace_a = load_and_validate_e0_trace(
            TRACE_PATHS[0], preflight.c200_reference, preflight.catalog_ids,
            expected_records=RECORD_COUNT,
        )
        trace_b = load_and_validate_e0_trace(
            TRACE_PATHS[1], preflight.c200_reference, preflight.catalog_ids,
            expected_records=RECORD_COUNT, retain_records=False,
        )
        _bind_worker_receipt(worker_results["replica_a"], trace_a)
        _bind_worker_receipt(worker_results["replica_b"], trace_b)
        if not (
            trace_a.canonical_trace_sha256 == trace_b.canonical_trace_sha256
            and trace_a.canonical_trace_bytes == trace_b.canonical_trace_bytes
            and trace_a.lengths == trace_b.lengths
            and trace_a.c200_lengths == trace_b.c200_lengths
            and trace_a.c200_lengths == tuple(map(len, preflight.c200_reference))
            and _files_equal(TRACE_PATHS[0], TRACE_PATHS[1])
        ):
            raise E0ProbeError("two fresh full E0 workers are not exact repeats")

        inflation = inflation_summary(trace_a.lengths, trace_a.c200_lengths,
                                      trace_bytes=trace_a.canonical_trace_bytes)
        worker_rss = {
            name: int(result["receipt"]["summary"]["resources"]["peak_working_set_bytes"])
            for name, result in worker_results.items()
        }
        if not (
            inflation["sealed_c200_candidate_cells"] == C200_CANDIDATE_CELLS
            and inflation["candidate_cell_ratio_over_c100"] <= CELL_RATIO_C100_MAXIMUM
            and inflation["candidate_cell_ratio_over_c200"] <= CELL_RATIO_C200_MAXIMUM
            and inflation["trace_byte_ratio_over_c100"] <= TRACE_RATIO_C100_MAXIMUM
            and inflation["trace_byte_ratio_over_c200"] <= TRACE_RATIO_C200_MAXIMUM
            and sum(worker_rss.values()) <= WORKER_RSS_SUM_MAXIMUM
            and time.perf_counter() - formal_started <= TOTAL_WALL_MAXIMUM
        ):
            raise E0ProbeError("formal E0 resource gate failed")
        pre_target_sources = _rehash_target_free(preflight)
        pre_target_git = _validate_git_checkpoint(implementation_commit)

        # First outcome-side import and first outcome-file access.  All workers
        # are closed, both traces are immutable/exact, and all target-free gates
        # above have passed.
        phase = "evaluator_side_proxy_then_numeric_fold_attach"
        c200_probe = importlib.import_module("scripts.probe_c200_candidate_recall")
        catalog_ids, products, _categories, catalog_identity = c200_probe._load_catalog_target_free()
        if catalog_ids != preflight.catalog_ids or catalog_identity.report() != preflight.source_identities["catalog"]:
            raise E0ProbeError("evaluator-side catalog identity drifted")
        proxy_source = c200_probe._open_proxy_after_receipt(PROXY_PATH)
        targets, eligibility, taxonomy = c200_probe._derive_target_membership_inputs(
            proxy_source.samples, products
        )
        proxy_identity = c200_probe._reverify_and_close_proxy(proxy_source)
        proxy_source = None
        outer_fold, family_index, label_identity = c200_probe._load_fold_labels_after_traces(
            LABEL_PATH
        )
        flags = _flags_from_trace(trace_a, targets, eligibility)
        del targets, eligibility, products, catalog_ids
        aggregate_full = aggregate_candidate_recall(
            flags, outer_fold=outer_fold, family_index=family_index, taxonomy=taxonomy
        )
        sanity = aggregate_full["all_sessions"]
        if any(sanity[f"c{cutoff}"]["count"] != expected
               for cutoff, expected in EXPECTED_C200_RECALL.items()):
            raise E0ProbeError("sealed C200 recall sanity drifted")
        if sanity["c400"]["count"] < sanity["c200"]["count"]:
            raise E0ProbeError("E0 recall violates prefix monotonicity")

        phase = "final_source_git_resource_privacy_gates"
        final_sources = _rehash_target_free(preflight)
        final_git = _validate_git_checkpoint(implementation_commit)
        if pre_target_sources != final_sources or pre_target_git != final_git:
            raise E0ProbeError("source or Git identity changed after outcome attach")
        total_wall = time.perf_counter() - formal_started
        if total_wall > TOTAL_WALL_MAXIMUM:
            raise E0ProbeError("formal E0 probe exceeded total wall budget")

        increment = aggregate_full["increment"]
        c200_uniform = aggregate_full["target_uniform"]["c200"]["fraction"]
        c400_uniform = aggregate_full["target_uniform"]["c400"]["fraction"]
        recall_go = sanity["c400"]["count"] > EXPECTED_C200_RECALL[200]
        e1_gate = bool(
            recall_go and c400_uniform > c200_uniform
            and increment["outer_fold_span"] >= 2
            and increment["non_clothing_count"] >= 1
            and increment["target_cluster_count"] >= 1
        )
        status = (
            "E0_RECALL_GO_ALLOW_PREREGISTERED_100_SESSION_SAFETY_SMOKE"
            if recall_go else "E0_RECALL_NO_GO_CONTINUE_G0_MULTI_VIEW_SPARSE_UNION"
        )
        aggregate = _compact_aggregate(aggregate_full)
        activation_turns = sum(new > old for new, old in zip(
            trace_a.lengths, trace_a.c200_lengths, strict=True
        ))
        activation_sessions = sum(
            any(trace_a.lengths[index * TURN_COUNT + offset]
                > trace_a.c200_lengths[index * TURN_COUNT + offset]
                for offset in range(TURN_COUNT))
            for index in range(SESSION_COUNT)
        )
        worker_resources = {}
        for name, worker in worker_results.items():
            summary = worker["receipt"]["summary"]
            worker_resources[name] = {
                "mode": worker["mode"], "wall_seconds": worker["wall_seconds"],
                "peak_rss_bytes": summary["resources"]["peak_working_set_bytes"],
                "cold_init_seconds": summary["latency"]["cold_semantic_initialization_seconds"],
                "respond_p95_ms": summary["latency"]["respond"]["p95_milliseconds"],
                "dense_p95_ms": summary["latency"]["dense_query_and_exact_search"]["p95_milliseconds"],
            }

        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID,
            "status": status, "recorded_on": "2026-08-31", "rerun_forbidden": True,
            "evidence_scope": "shared 2000-session diagnostic candidate recall; not private validation",
            "implementation": {
                "commit": implementation_commit, "preregistration_commit": PREREG_COMMIT,
                "branch": BRANCH, "default": "off", "runtime_target_blind": True,
                "causal": True, "full_agent_evaluator_started": False,
                "protected_splits_opened": False, "fit_or_selection_performed": False,
                "c200_runner_invoked": False, "c400_runner_invoked": False,
            },
            "entrypoint_regression": {
                "direct_script": True, "python_module": True,
                "outside_repo_cwd_direct": True, "missing_module_failed_closed": True,
                "preflight_smoke_exact_repeat": preflight.smoke["direct_module_exact_repeat"],
                "final_pre_receipt_check": bool(final_entrypoint_check),
            },
            "candidate_recall": aggregate,
            "candidate_retention": {
                "full_variable_c200_exact_ordered_prefix": True,
                "old_candidate_loss_count": 0, "candidate_level_hit_to_miss": 0,
                "served_top10_unchanged": True,
            },
            "top10_metrics": {
                "baseline_hr_at_10": None, "candidate_hr_at_10": None,
                "miss_to_hit": None, "hit_to_miss": None, "net": None,
                "mrr": None, "mttc": None, "technical_score": None,
                "reason": "candidate-recall-only; served Top10 unchanged",
            },
            "activation": {"sessions": activation_sessions, "turns": activation_turns},
            "inflation": inflation,
            "exact_repeat": {
                "passed": True, "trace_sha256": trace_a.canonical_trace_sha256,
                "trace_bytes": trace_a.canonical_trace_bytes,
                "record_count": trace_a.record_count, "candidate_arrays_equal": True,
            },
            "resources": {
                "total_wall_seconds": round(total_wall, 6), "workers": worker_resources,
                "conservative_worker_peak_sum_bytes": sum(worker_rss.values()),
                "required_asset_bytes": REQUIRED_ASSET_BYTES,
                "network_attempt_count": 0, "gpu_peak_bytes": 0, "budgets_passed": True,
            },
            "source_hashes": {
                **final_sources,
                "proxy": proxy_identity.report(), "numeric_fold_archive": label_identity,
                "fresh_trace": trace_a.canonical_trace_sha256,
            },
            "git": final_git,
            "decision": {
                "recall_go": recall_go, "embedding_adapter_e1_gate_passed": e1_gate,
                "top10_global_promotion": False,
                "next_stage": "preregistered 100-session safety smoke then G0" if recall_go
                else "G0 deterministic multi-view sparse query union",
                "fallback_order": ["SR-V2.12-FIXED-TWO-PAGE-GRACE", "v1.9", "P11", "R08"],
            },
        }
        _result_privacy_scan(result, catalog_ids=preflight.catalog_ids)
        if descriptor is None:
            raise E0ProbeError("receipt descriptor closed before result seal")
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
        if isinstance(error, E0ProbeError):
            raise
        raise E0ProbeError("formal E0 one-shot failed") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-commit")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight-only", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--entrypoint-self-check", action="store_true")
    actions.add_argument("--self-check", dest="entrypoint_self_check", action="store_true",
                         help=argparse.SUPPRESS)
    parser.add_argument("--require-module", default="evaluator.local_evaluator")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.entrypoint_self_check:
        output = _entrypoint_self_check(arguments.require_module)
    else:
        if not arguments.implementation_commit:
            raise E0ProbeError("--implementation-commit is required")
        if arguments.preflight_only:
            checked = preflight_only(arguments.implementation_commit)
            output = {
                "status": "TARGET_FREE_PREFLIGHT_PASS",
                "commit": checked.git["commit"],
                "catalog_sha256": checked.source_identities["catalog"]["sha256"],
                "c200_trace_sha256": checked.source_identities["sealed_c200"]["replicas"][0]["sha256"],
                "asset_manifest_sha256": _canonical_sha256(checked.asset_identities),
                "direct_module_worker_smoke_exact": checked.smoke["direct_module_exact_repeat"],
                "entrypoints_passed": True, "receipt_created": False,
            }
        else:
            outcome = run(arguments.implementation_commit)
            output = {
                "status": outcome["status"], "commit": outcome["implementation"]["commit"],
                "c200_count": outcome["candidate_recall"]["all_sessions"]["c200"]["count"],
                "c400_count": outcome["candidate_recall"]["all_sessions"]["c400"]["count"],
                "increment": outcome["candidate_recall"]["increment"]["count"],
                "exact_repeat": outcome["exact_repeat"]["passed"],
            }
    sys.stdout.buffer.write(_canonical_bytes(output) + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
