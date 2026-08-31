"""Target-free worker for the frozen v2.23 OOV chargram bridge G0 probe.

This module is deliberately a diagnostic worker, not a serving entrypoint.  It
streams one complete candidate trace while preserving every sealed C200 row as
an ordered prefix.  Runtime imports are limited to the frozen C200 contract and
``starter.oov_chargram_bridge_g0``; prior sparse-union and multiview mechanisms
are never imported or executed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import stat as stat_module
import statistics
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = "small-ranker-v2.23-oov-chargram-worker-summary.v1"
FAILURE_SCHEMA_VERSION = "small-ranker-v2.23-oov-chargram-worker-failure.v1"
CONTEXT_SCHEMA_VERSION = "small-ranker-visible-context.v1"
SESSION_COUNT = 2_000
TURN_COUNT = 10
RECORD_COUNT = SESSION_COUNT * TURN_COUNT
ROUTE_LIMIT = 32
MAX_CANDIDATES = 400
TAIL_LIMIT = 192
MIN_C200_CANDIDATES = 100
MAX_C200_CANDIDATES = 200
ALLOWED_SESSION_LIMITS = (20, 100, SESSION_COUNT)

EXPECTED_CATALOG_BYTES = 60_546_327
EXPECTED_CATALOG_ROWS = 50_000
EXPECTED_CATALOG_SHA256 = (
    "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
)
EXPECTED_CONTEXT_BYTES = 47_168_882
EXPECTED_CONTEXT_ROWS = 2_000
EXPECTED_CONTEXT_TURNS = 20_000
EXPECTED_CONTEXT_SHA256 = (
    "f30a98700da5d480731fe7e82c87c40a22f06de290e069e20dc68f9fefecd20f"
)
EXPECTED_REDACTED_MESSAGE_COUNT = 8
EXPECTED_C200_REFERENCE_BYTES = 32_226_135
EXPECTED_C200_REFERENCE_ROWS = 20_000
EXPECTED_C200_REFERENCE_SHA256 = (
    "a8589749376f48f019997a618481578dde36be4ca1fc723e8ed00056c23e40dc"
)
EXPECTED_C200_CANDIDATE_CELLS = 2_425_785
EXPECTED_ATTRIBUTE_REGISTRY_SHA256 = (
    "1d85fc42f49fd9374238d98b8feaeab8d76269b0987740256fe60e666757d2ca"
)
EXPECTED_EXECUTABLE = Path(r"D:\450\conda\envs\tiktok\python.exe")
EXPECTED_PYTHON = "3.11.16"
EXPECTED_SQLITE = "3.53.4"
EXPECTED_PREREGISTRATION_BLOB_SHA1 = (
    "0f875b691f6a373433895daa5d62b594e0ceca2e"
)
EXPECTED_C200_CONTRACT_BLOB_SHA1 = (
    "b94fddcf5a9b20ddde540f3f43ea9962982cb096"
)
EXPECTED_CATALOG_PATH = PureWindowsPath(
    r"D:\tiktok\techjam-err402-fast-track\data\catalog.jsonl"
)
EXPECTED_CONTEXT_PATH = PureWindowsPath(
    r"D:\tiktok\techjam-v2-16-c200-recall\experiments\fast_track"
    r"\c200_candidate_recall_cache_20260831\visible_context.jsonl"
)
EXPECTED_C200_REFERENCE_PATHS = frozenset(
    {
        PureWindowsPath(
            r"D:\tiktok\techjam-v2-16-c200-recall\experiments\fast_track"
            r"\c200_candidate_recall_cache_20260831\replica_a.jsonl"
        ),
        PureWindowsPath(
            r"D:\tiktok\techjam-v2-16-c200-recall\experiments\fast_track"
            r"\c200_candidate_recall_cache_20260831\replica_b.jsonl"
        ),
    }
)
EXPECTED_RUNTIME_ROOT = PureWindowsPath(r"D:\tiktok\.v223_runtime")
PREREGISTRATION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "small_ranker_v2_23.oov_chargram_lexicon_bridge_g0_preregistration.json"
)
CORE_PATH = PROJECT_ROOT / "starter" / "oov_chargram_bridge_g0.py"
C200_CONTRACT_PATH = PROJECT_ROOT / "scripts" / "c200_candidate_worker.py"

MAX_ROUTE_P95_MILLISECONDS = 25.0
MAX_MASK_P95_MILLISECONDS = 50.0
MAX_EXTRA_P95_MILLISECONDS = 100.0
MAX_TURN_P95_MILLISECONDS = 400.0
MAX_WORKING_SET_BYTES = 1_610_612_736
MAX_WALL_SECONDS = 1_800.0
MIN_FULL_TURNS_PER_SECOND = 10.0
MAX_CANDIDATE_CELL_RATIO = 2.0
MAX_TRACE_BYTE_RATIO = 2.1
CACHE_CAPACITIES = {
    "oov_bridge": 4_096,
    "fts_route": 512,
    "product_view": 4_096,
    "mask_decision": 16_384,
}

NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
RUN_ID_RE = re.compile(r"[0-9a-z][0-9a-z._-]{7,127}\Z")
GIT_BLOB_RE = re.compile(r"[0-9a-f]{40}\Z")
OOV_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{3,23}\Z", re.ASCII)
ASIN_SHAPE_RE = re.compile(
    r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE
)
CATALOG_IDENTIFIER_TOKEN_RE = re.compile(
    r"(?<![A-Z0-9])[A-Z0-9]{10}(?![A-Z0-9])", re.IGNORECASE
)
FORBIDDEN_RECEIPT_KEYS = frozenset(
    {
        "asin",
        "c200",
        "candidates",
        "canonical_values",
        "eligible_from",
        "expression",
        "family_index",
        "ground_truth",
        "membership_vector",
        "message",
        "messages",
        "ordinal",
        "outer_fold",
        "parent_asin",
        "per_session",
        "query",
        "query_terms",
        "sample_id",
        "target",
        "target_asin",
        "target_id",
        "terms",
        "turn",
    }
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
PROHIBITED_RUNTIME_MODULES = frozenset(
    {
        "starter.sparse_multiview_g0",
        "starter.sparse_union_g0",
        "starter.sparse_multiview",
        "scripts.sparse_multiview_g0_worker",
        "scripts.sparse_union_g0_worker",
        "scripts.sparse_multiview_candidate_worker",
    }
)
GPU_RUNTIME_PREFIXES = (
    "cupy",
    "jax",
    "numba.cuda",
    "onnxruntime",
    "tensorflow",
    "torch",
)
SOURCE_FIELDS = {
    "unknown_category_token": ("title", "categories"),
    "exact_active_token": ("title", "features", "details", "store", "description"),
}

STAGE_ID_ALLOWLIST = frozenset(
    {
        "preflight_20_uncached_direct",
        "preflight_20_cached_direct",
        "preflight_20_cached_module",
        "preflight_100_uncached_direct",
        "preflight_100_cached_direct",
        "preflight_100_cached_module",
        "candidate_2000_cached_direct",
        "candidate_2000_cached_module",
        "preclaim_synthetic",
    }
)
WORKER_PHASE_ALLOWLIST = frozenset(
    {
        "ARGUMENT_VALIDATION",
        "ENVIRONMENT_AUDIT",
        "SOURCE_VALIDATION",
        "TRACE_PREPARATION",
        "SEALED_SOURCE_VALIDATION",
        "LEXICON_INITIALIZATION",
        "TRAJECTORY",
        "SQLITE_CLOSE",
        "SOURCE_REVALIDATION",
        "RESOURCE_VALIDATION",
        "TRACE_PUBLICATION",
        "UNKNOWN",
    }
)
WORKER_ERROR_CODE_ALLOWLIST = frozenset(
    {
        "ARGUMENT_INVALID",
        "ENVIRONMENT_INVALID",
        "SOURCE_IDENTITY",
        "SOURCE_IMPORT",
        "TRACE_PATH",
        "SEALED_SOURCE_SCHEMA",
        "SEALED_SOURCE_IDENTITY",
        "LEXICON_SCHEMA",
        "LEXICON_RESOURCE",
        "QUERY_ONLY_CONTRACT",
        "CONTEXT_SCHEMA",
        "C200_SCHEMA",
        "EXPANSION_CONTRACT",
        "CACHE_CONTRACT",
        "RESOURCE_GATE",
        "NETWORK_ATTEMPT",
        "GPU_RUNTIME_PRESENT",
        "SQLITE_CLOSE",
        "TRACE_PUBLICATION",
        "PRIVACY_SCAN",
        "INTERNAL_INVARIANT",
        "UNCLASSIFIED",
        "UNAVAILABLE",
    }
)
_FAILURE_SITE_ALLOWLIST = {
    ("worker", "_validate_arguments"): "SITE_0001",
    ("worker", "_freeze_cpu_environment"): "SITE_0002",
    ("worker", "_tracked_source_identities"): "SITE_0003",
    ("worker", "_load_runtime_after_audit"): "SITE_0004",
    ("worker", "_validate_trace_paths"): "SITE_0005",
    ("worker", "_validate_reference_identity"): "SITE_0006",
    ("worker", "parse_c200_reference_line"): "SITE_0007",
    ("worker", "validate_expansion_result"): "SITE_0008",
    ("worker", "expansion_timing_contract"): "SITE_0009",
    ("worker", "_cache_contract"): "SITE_0010",
    ("worker", "_bridge_contract"): "SITE_0011",
    ("worker", "run"): "SITE_0012",
    ("worker", "_publish_partial_exclusive"): "SITE_0013",
    ("core", "__init__"): "SITE_0020",
    ("core", "_initialize"): "SITE_0021",
    ("core", "_map_source"): "SITE_0022",
    ("core", "_query"): "SITE_0023",
    ("core", "_apply_mask"): "SITE_0024",
    ("core", "expand"): "SITE_0025",
    ("core", "validate"): "SITE_0026",
    ("core", "close"): "SITE_0027",
    ("c200", "_catalog_identity"): "SITE_0030",
    ("c200", "_validate_context_identity"): "SITE_0031",
    ("c200", "_parse_context_container"): "SITE_0032",
}


class OovChargramBridgeG0WorkerError(RuntimeError):
    """A sanitized, classified worker invariant failure."""

    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True)
class SourceIdentity:
    bytes: int
    rows: int
    sha256: str
    snapshot: tuple[int, int, int]

    def report(self) -> dict[str, int | str]:
        return {"bytes": self.bytes, "rows": self.rows, "sha256": self.sha256}


@dataclass
class WorkerProgress:
    phase: str = "ARGUMENT_VALIDATION"
    stage_id: str = "preclaim_synthetic"
    session_limit: int | None = None
    nonce: str | None = None
    wall_started: float = field(default_factory=time.perf_counter)
    last_completed_session: int = 0
    partial_path: Path | None = None
    partial_bytes: int = 0
    partial_rows: int = 0
    partial_sha256: str = EMPTY_SHA256
    network_audit: "OfflineNetworkAudit | None" = None
    input_identities: dict[str, dict[str, int | str]] = field(default_factory=dict)
    source_identities: dict[str, dict[str, int | str]] = field(default_factory=dict)
    sqlite_closed: bool = False


class OfflineNetworkAudit:
    """Fail closed on every Python-audited socket operation."""

    def __init__(self) -> None:
        self.attempt_count = 0
        self.event_counts: Counter[str] = Counter()

    def hook(self, event: str, _arguments: tuple[object, ...]) -> None:
        if not event.startswith("socket."):
            return
        self.attempt_count += 1
        self.event_counts[event] += 1
        raise PermissionError("network disabled")


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
        raise OovChargramBridgeG0WorkerError("NON_CANONICAL_JSON") from error


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OovChargramBridgeG0WorkerError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).casefold())
            keys.update(_walk_keys(child))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return False
    attributes = int(getattr(observed, "st_file_attributes", 0))
    marker = int(getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & marker)


def _require_real_ancestry(path: Path, error_code: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if _is_link_or_reparse(component):
            raise OovChargramBridgeG0WorkerError(error_code)


def _require_regular_file(path: Path, error_code: str) -> None:
    _require_real_ancestry(path, error_code)
    if not path.is_file():
        raise OovChargramBridgeG0WorkerError(error_code)


def _snapshot(path: Path) -> tuple[int, int, int]:
    observed = path.stat()
    return (
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(getattr(observed, "st_ino", 0)),
    )


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=True).as_posix().casefold() == right.resolve(
            strict=True
        ).as_posix().casefold()
    except (FileNotFoundError, OSError):
        return False


def _lexical_windows_path(path: Path | PureWindowsPath) -> PureWindowsPath:
    raw = os.fspath(path)
    raw_parts = raw.replace("/", "\\").split("\\")
    if (
        not raw
        or raw.startswith("\\\\")
        or raw.startswith("\\\\?\\")
        or any(part in {"", ".", ".."} for part in raw_parts[1:])
        or any(":" in part for part in raw_parts[1:])
    ):
        raise OovChargramBridgeG0WorkerError("PATH_NOT_LEXICALLY_CANONICAL")
    result = PureWindowsPath(raw)
    if not result.is_absolute():
        raise OovChargramBridgeG0WorkerError("PATH_NOT_ABSOLUTE")
    return result


def _lexical_path_key(path: PureWindowsPath) -> tuple[str, ...]:
    return tuple(part.casefold() for part in path.parts)


def _guard_legacy_namespaces(path: Path | PureWindowsPath) -> None:
    lexical = path if isinstance(path, PureWindowsPath) else _lexical_windows_path(path)
    parts = _lexical_path_key(lexical)
    joined = "/".join(parts)
    forbidden = (
        "small_ranker_v2_19_",
        "small_ranker_v2_20_",
        "small_ranker_v2_20b_",
        "small_ranker_v2_21_",
        "small_ranker_v2_22_",
        "small-ranker-v2.19",
        "small-ranker-v2.20",
        "small-ranker-v2.20b",
        "small-ranker-v2.21",
        "small-ranker-v2.22-",
    )
    if any(marker in joined for marker in forbidden):
        raise OovChargramBridgeG0WorkerError("LEGACY_NAMESPACE_DENIED")


def _raw_identity(path: Path, *, rows: bool = False) -> SourceIdentity:
    _require_regular_file(path, "SOURCE_UNAVAILABLE")
    before = _snapshot(path)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
            if rows:
                row_count += chunk.count(b"\n")
    after = _snapshot(path)
    if before != after or byte_count != before[0]:
        raise OovChargramBridgeG0WorkerError("SOURCE_CHANGED_DURING_READ")
    return SourceIdentity(byte_count, row_count, digest.hexdigest(), after)


def _raw_git_blob_sha1(path: Path) -> str:
    _require_regular_file(path, "PINNED_SOURCE_UNAVAILABLE")
    before = _snapshot(path)
    with path.open("rb") as handle:
        working_tree_bytes = handle.read()
    after = _snapshot(path)
    if before != after or len(working_tree_bytes) != before[0]:
        raise OovChargramBridgeG0WorkerError("PINNED_SOURCE_CHANGED_DURING_READ")
    blob_bytes = working_tree_bytes.replace(b"\r\n", b"\n")
    digest = hashlib.sha1()
    digest.update(f"blob {len(blob_bytes)}\0".encode("ascii"))
    digest.update(blob_bytes)
    return digest.hexdigest()


def _tracked_source_identities() -> dict[str, SourceIdentity]:
    paths = {
        "preregistration": PREREGISTRATION_PATH,
        "scripts/c200_candidate_worker.py": C200_CONTRACT_PATH,
        "scripts/oov_chargram_bridge_g0_worker.py": Path(__file__).resolve(),
        "starter/oov_chargram_bridge_g0.py": CORE_PATH,
    }
    identities = {name: _raw_identity(path, rows=True) for name, path in paths.items()}
    if (
        _raw_git_blob_sha1(PREREGISTRATION_PATH)
        != EXPECTED_PREREGISTRATION_BLOB_SHA1
        or _raw_git_blob_sha1(C200_CONTRACT_PATH)
        != EXPECTED_C200_CONTRACT_BLOB_SHA1
    ):
        raise OovChargramBridgeG0WorkerError("PINNED_GIT_BLOB_IDENTITY")
    return identities


def _source_identity_reports(
    identities: Mapping[str, SourceIdentity],
) -> dict[str, dict[str, int | str]]:
    reports = {name: identity.report() for name, identity in identities.items()}
    reports["preregistration"]["raw_git_blob_sha1"] = (
        EXPECTED_PREREGISTRATION_BLOB_SHA1
    )
    reports["scripts/c200_candidate_worker.py"]["raw_git_blob_sha1"] = (
        EXPECTED_C200_CONTRACT_BLOB_SHA1
    )
    return reports


def _validate_semantic_source_blobs(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "semantic_audit", False)):
        return
    expected_worker = getattr(args, "expected_worker_blob", None)
    expected_union = getattr(args, "expected_union_blob", None)
    if (
        not isinstance(expected_worker, str)
        or GIT_BLOB_RE.fullmatch(expected_worker) is None
        or not isinstance(expected_union, str)
        or GIT_BLOB_RE.fullmatch(expected_union) is None
        or _raw_git_blob_sha1(Path(__file__).resolve()) != expected_worker
        or _raw_git_blob_sha1(CORE_PATH) != expected_union
    ):
        raise OovChargramBridgeG0WorkerError("EXPECTED_SOURCE_BLOB_MISMATCH")


def _same_identities(
    before: Mapping[str, SourceIdentity], after: Mapping[str, SourceIdentity]
) -> bool:
    return all(
        name in after
        and identity.report() == after[name].report()
        and identity.snapshot == after[name].snapshot
        for name, identity in before.items()
    ) and set(before) == set(after)


def _assert_legacy_runtime_absent() -> None:
    if any(name in sys.modules for name in PROHIBITED_RUNTIME_MODULES):
        raise OovChargramBridgeG0WorkerError("LEGACY_RUNTIME_PRESENT")


def _load_runtime_after_audit() -> tuple[
    Any, Any, Callable[[], None], Callable[[], str]
]:
    _assert_legacy_runtime_absent()
    try:
        c200_contract = importlib.import_module("scripts.c200_candidate_worker")
        oov_chargram_bridge_g0 = importlib.import_module("starter.oov_chargram_bridge_g0")
        attributes = importlib.import_module("starter.attributes")
    except ImportError as error:
        raise OovChargramBridgeG0WorkerError("RUNTIME_IMPORT") from error
    validate_core = getattr(oov_chargram_bridge_g0, "validate", None)
    registry_hash = getattr(attributes, "attribute_registry_sha256", None)
    if not callable(validate_core) or not callable(registry_hash):
        raise OovChargramBridgeG0WorkerError("SOURCE_IMPORT")
    _assert_legacy_runtime_absent()
    return c200_contract, oov_chargram_bridge_g0, validate_core, registry_hash


def _verify_imported_module_origins(
    c200_contract: Any,
    oov_chargram_bridge_g0: Any,
    validate_core: Callable[[], None],
    registry_hash: Callable[[], str],
) -> None:
    if (
        getattr(c200_contract, "__name__", None)
        != "scripts.c200_candidate_worker"
        or getattr(oov_chargram_bridge_g0, "__name__", None) != "starter.oov_chargram_bridge_g0"
        or getattr(validate_core, "__module__", None)
        != "starter.oov_chargram_bridge_g0"
        or getattr(registry_hash, "__module__", None) != "starter.attributes"
    ):
        raise OovChargramBridgeG0WorkerError("RUNTIME_MODULE_IDENTITY")
    expected_origins = {
        "scripts.c200_candidate_worker": C200_CONTRACT_PATH,
        "starter.oov_chargram_bridge_g0": CORE_PATH,
        "starter.attributes": PROJECT_ROOT / "starter" / "attributes.py",
    }
    for name, expected in expected_origins.items():
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if (
            module is None
            or not isinstance(origin, str)
            or not isinstance(spec_origin, str)
            or not _same_path(Path(origin), expected)
            or not _same_path(Path(spec_origin), expected)
        ):
            raise OovChargramBridgeG0WorkerError("RUNTIME_MODULE_ORIGIN")
    _assert_legacy_runtime_absent()


def _freeze_cpu_environment() -> dict[str, Any]:
    actual = {
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "no_user_site_flag": bool(sys.flags.no_user_site),
        "python": sys.version.split()[0],
        "pythonhashseed": os.getenv("PYTHONHASHSEED"),
        "python_no_user_site": os.getenv("PYTHONNOUSERSITE"),
        "sqlite": sqlite3.sqlite_version,
    }
    if not (
        _same_path(Path(sys.executable), EXPECTED_EXECUTABLE)
        and _same_path(Path.cwd(), PROJECT_ROOT)
        and actual["python"] == EXPECTED_PYTHON
        and actual["sqlite"] == EXPECTED_SQLITE
        and actual["pythonhashseed"] == "0"
        and actual["cuda_visible_devices"] == ""
        and actual["python_no_user_site"] == "1"
        and actual["no_user_site_flag"] is True
    ):
        raise OovChargramBridgeG0WorkerError("RUNTIME_ENVIRONMENT")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    loaded_gpu_modules = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in GPU_RUNTIME_PREFIXES
        )
    )
    if loaded_gpu_modules:
        raise OovChargramBridgeG0WorkerError("GPU_RUNTIME_PRESENT")
    return {
        **actual,
        "cwd_is_project_root": True,
        "executable_is_frozen": True,
        "device": "CPU",
        "provider": "SQLite FTS5 + CPython",
        "gpu_peak_bytes": 0,
        "gpu_used": False,
        "network_attempt_count": 0,
    }


def _validate_c200_values(
    candidates: object, catalog_ids: Iterable[str]
) -> tuple[str, ...]:
    if not isinstance(candidates, (list, tuple)):
        raise OovChargramBridgeG0WorkerError("C200_SCHEMA")
    values = tuple(candidates)
    catalog = (
        catalog_ids
        if isinstance(catalog_ids, (set, frozenset, dict))
        else frozenset(catalog_ids)
    )
    if (
        not MIN_C200_CANDIDATES <= len(values) <= MAX_C200_CANDIDATES
        or len(values) != len(set(values))
        or any(
            not isinstance(identifier, str)
            or not identifier
            or identifier not in catalog
            for identifier in values
        )
    ):
        raise OovChargramBridgeG0WorkerError("C200_SCHEMA")
    return values


def parse_c200_reference_line(
    line: bytes,
    *,
    ordinal: int,
    turn: int,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    try:
        value = json.loads(
            line.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                OovChargramBridgeG0WorkerError("C200_NONFINITE")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OovChargramBridgeG0WorkerError("C200_JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"c200", "ordinal", "turn"}
        or value.get("ordinal") != ordinal
        or isinstance(value.get("ordinal"), bool)
        or value.get("turn") != turn
        or isinstance(value.get("turn"), bool)
    ):
        raise OovChargramBridgeG0WorkerError("C200_ORDER")
    candidates = _validate_c200_values(value.get("c200"), catalog_ids)
    expected = _canonical_bytes(
        {"c200": list(candidates), "ordinal": ordinal, "turn": turn}
    ) + b"\n"
    if line != expected:
        raise OovChargramBridgeG0WorkerError("C200_CANONICAL")
    return candidates


def _validate_reference_identity(
    path: Path, catalog_ids: frozenset[str], c200_contract: Any
) -> tuple[SourceIdentity, int]:
    del c200_contract
    _require_regular_file(path, "C200_REFERENCE_UNAVAILABLE")
    before = _snapshot(path)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    candidate_cells = 0
    with path.open("rb") as handle:
        for line in handle:
            row_count += 1
            if row_count > RECORD_COUNT or not line.strip():
                raise OovChargramBridgeG0WorkerError("C200_ROW_COUNT")
            candidates = parse_c200_reference_line(
                line,
                ordinal=(row_count - 1) // TURN_COUNT + 1,
                turn=(row_count - 1) % TURN_COUNT + 1,
                catalog_ids=catalog_ids,
            )
            digest.update(line)
            byte_count += len(line)
            candidate_cells += len(candidates)
    after = _snapshot(path)
    identity = SourceIdentity(byte_count, row_count, digest.hexdigest(), after)
    if (
        before != after
        or identity.report()
        != {
            "bytes": EXPECTED_C200_REFERENCE_BYTES,
            "rows": EXPECTED_C200_REFERENCE_ROWS,
            "sha256": EXPECTED_C200_REFERENCE_SHA256,
        }
        or candidate_cells != EXPECTED_C200_CANDIDATE_CELLS
    ):
        raise OovChargramBridgeG0WorkerError("C200_REFERENCE_IDENTITY")
    return identity, candidate_cells


def _ordered_identifiers(
    value: object, catalog: frozenset[str], error_code: str
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise OovChargramBridgeG0WorkerError(error_code)
    identifiers = tuple(value)
    if (
        len(identifiers) != len(set(identifiers))
        or any(
            not isinstance(identifier, str)
            or not identifier
            or identifier not in catalog
            for identifier in identifiers
        )
    ):
        raise OovChargramBridgeG0WorkerError(error_code)
    return identifiers


def _source_shape(source: object) -> tuple[str, str, tuple[str, ...]]:
    kind = getattr(source, "kind", None)
    token = getattr(source, "token", None)
    fields = getattr(source, "fields", None)
    if (
        kind not in {"unknown_category_token", "exact_active_token"}
        or not isinstance(token, str)
        or OOV_TOKEN_RE.fullmatch(token) is None
        or not isinstance(fields, (list, tuple))
        or tuple(fields) != SOURCE_FIELDS[str(kind)]
    ):
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    return str(kind), token, tuple(fields)


def _bridge_shape(bridge: object) -> tuple[tuple[str, str, tuple[str, ...]], tuple[tuple[object, ...], ...]]:
    source = _source_shape(getattr(bridge, "source", None))
    matches = getattr(bridge, "matches", None)
    if not isinstance(matches, (list, tuple)) or not 1 <= len(matches) <= 4:
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    shaped: list[tuple[object, ...]] = []
    for match in matches:
        token = getattr(match, "token", None)
        edit = getattr(match, "edit_distance", None)
        dice = getattr(match, "dice", None)
        doc_frequency = getattr(match, "global_doc_frequency", None)
        if (
            not isinstance(token, str)
            or OOV_TOKEN_RE.fullmatch(token) is None
            or not isinstance(edit, int)
            or isinstance(edit, bool)
            or edit < 0
            or not hasattr(dice, "numerator")
            or not hasattr(dice, "denominator")
            or int(dice.denominator) <= 0
            or 3 * int(dice.numerator) < 2 * int(dice.denominator)
            or not isinstance(doc_frequency, int)
            or isinstance(doc_frequency, bool)
            or doc_frequency < 0
        ):
            raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
        shaped.append(
            (
                token,
                edit,
                int(dice.numerator),
                int(dice.denominator),
                doc_frequency,
            )
        )
    maximum = 1 if len(source[1]) <= 7 else 2
    if any(
        int(item[1]) > maximum
        or abs(len(source[1]) - len(str(item[0]))) > maximum
        for item in shaped
    ) or shaped != sorted(
        shaped,
        key=lambda item: (
            int(item[1]),
            -Fraction(int(item[2]), int(item[3])),
            -int(item[4]),
            str(item[0]),
        ),
    ):
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    return source, tuple(shaped)


def validate_expansion_result(
    result: object,
    sealed_c200: object,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    """Validate the single bridge route, hard mask, and immutable C200 prefix."""

    prefix = _validate_c200_values(sealed_c200, catalog_ids)
    required = {
        "activated",
        "bridge_lookup_latency_ns",
        "bridge_sources",
        "candidates",
        "conflict_count",
        "controlled_write_rejected",
        "enabled",
        "fallback",
        "fallback_code",
        "filtered_identifiers",
        "fts_route_latency_ns",
        "hard_mask_latency_ns",
        "legacy_route_executions",
        "negative_violation_count",
        "novel_identifiers",
        "positive_conflict_count",
        "prefix",
        "query_only_readback_one",
        "routes",
        "source_records",
        "tail",
        "tail_conflict_count",
        "write_guard_unchanged",
    }
    if any(not hasattr(result, name) for name in required):
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    catalog = catalog_ids if isinstance(catalog_ids, frozenset) else frozenset(catalog_ids)
    candidates = _ordered_identifiers(result.candidates, catalog, "EXPANSION_CONTRACT")
    observed_prefix = _ordered_identifiers(result.prefix, catalog, "EXPANSION_CONTRACT")
    tail = _ordered_identifiers(result.tail, catalog, "EXPANSION_CONTRACT")
    novel = _ordered_identifiers(
        result.novel_identifiers, catalog, "EXPANSION_CONTRACT"
    )
    filtered = _ordered_identifiers(
        result.filtered_identifiers, catalog, "EXPANSION_CONTRACT"
    )
    if observed_prefix != prefix:
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")

    sources = getattr(result, "source_records")
    bridges = getattr(result, "bridge_sources")
    routes = getattr(result, "routes")
    if not all(isinstance(value, (list, tuple)) for value in (sources, bridges, routes)):
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    source_shapes = tuple(_source_shape(source) for source in sources)
    bridge_shapes = tuple(_bridge_shape(bridge) for bridge in bridges)
    kind_rank = {"unknown_category_token": 0, "exact_active_token": 1}
    if (
        len(source_shapes) != len(set(source_shapes))
        or len(bridge_shapes) > 6
        or len(bridge_shapes) != len(set(bridge_shapes))
        or bridge_shapes
        != tuple(
            sorted(
                bridge_shapes,
                key=lambda item: (
                    int(item[1][0][1]),
                    -Fraction(int(item[1][0][2]), int(item[1][0][3])),
                    -int(item[1][0][4]),
                    str(item[1][0][0]),
                    item[0][1],
                    kind_rank[item[0][0]],
                ),
            )
        )
    ):
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    if any(shape[0] not in source_shapes for shape in bridge_shapes):
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")

    raw_novel: list[str] = []
    raw_seen = set(prefix)
    route_shapes: list[tuple[object, ...]] = []
    for index, route in enumerate(routes):
        source_shape, match_shape = _bridge_shape(getattr(route, "source", None))
        expression = getattr(route, "expression", None)
        identifiers = _ordered_identifiers(
            getattr(route, "identifiers", None), catalog, "EXPANSION_CONTRACT"
        )
        route_novel = _ordered_identifiers(
            getattr(route, "novel_identifiers", None), catalog, "EXPANSION_CONTRACT"
        )
        route_filtered = _ordered_identifiers(
            getattr(route, "filtered_identifiers", None), catalog, "EXPANSION_CONTRACT"
        )
        latency = getattr(route, "latency_ns", None)
        expected_expression = (
            "{"
            + " ".join(source_shape[2])
            + "} : ("
            + " OR ".join('"' + str(match[0]) + '"' for match in match_shape)
            + ")"
        )
        if (
            index >= len(bridge_shapes)
            or (source_shape, match_shape) != bridge_shapes[index]
            or not isinstance(expression, str)
            or expression != expected_expression
            or len(identifiers) > ROUTE_LIMIT
            or route_novel != tuple(
                identifier for identifier in identifiers if identifier not in prefix
            )
            or any(identifier not in route_novel for identifier in route_filtered)
            or not isinstance(latency, int)
            or isinstance(latency, bool)
            or latency < 0
        ):
            raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
        for identifier in route_novel:
            if identifier not in raw_seen:
                raw_seen.add(identifier)
                raw_novel.append(identifier)
        route_shapes.append(
            (source_shape, match_shape, expression, identifiers, route_novel, route_filtered, latency)
        )
    filtered_set = frozenset(filtered)
    if (
        len(routes) != len(bridge_shapes)
        or novel != tuple(raw_novel)
        or filtered
        != tuple(identifier for identifier in novel if identifier in filtered_set)
        or any(
            route_shape[5]
            != tuple(
                identifier
                for identifier in route_shape[4]
                if identifier in filtered_set
            )
            for route_shape in route_shapes
        )
    ):
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    if any(identifier not in novel for identifier in filtered):
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")

    integer_fields = (
        "conflict_count",
        "negative_violation_count",
        "positive_conflict_count",
        "tail_conflict_count",
        "legacy_route_executions",
    )
    if any(
        not isinstance(getattr(result, name), int)
        or isinstance(getattr(result, name), bool)
        or int(getattr(result, name)) < 0
        for name in integer_fields
    ):
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    if (
        result.enabled is not True
        or not isinstance(result.activated, bool)
        or result.activated != bool(bridges)
        or result.fallback is not False
        or result.fallback_code != "NONE"
        or result.legacy_route_executions != 0
        or result.query_only_readback_one is not True
        or result.controlled_write_rejected is not True
        or result.write_guard_unchanged is not True
        or result.conflict_count != len(novel) - len(filtered)
        or result.tail_conflict_count != 0
        or candidates != prefix + tail
        or tail
        != filtered[: min(TAIL_LIMIT, max(0, MAX_CANDIDATES - len(prefix)))]
        or len(candidates) > MAX_CANDIDATES
        or len(candidates) != len(set(candidates))
    ):
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    return candidates


def expansion_timing_contract(result: object) -> tuple[int, int, int]:
    """Validate dynamic timing evidence without putting it in semantic hashes."""

    values = tuple(
        getattr(result, name, None)
        for name in (
            "bridge_lookup_latency_ns",
            "fts_route_latency_ns",
            "hard_mask_latency_ns",
        )
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ) or values[0] <= 0 or values[2] <= 0:
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    if bool(getattr(result, "routes", ())) and values[1] < 0:
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    return values


def canonical_trace_line(ordinal: int, turn: int, candidates: object) -> bytes:
    if (
        not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or not 1 <= ordinal <= SESSION_COUNT
        or not isinstance(turn, int)
        or isinstance(turn, bool)
        or not 1 <= turn <= TURN_COUNT
        or not isinstance(candidates, (list, tuple))
    ):
        raise OovChargramBridgeG0WorkerError("TRACE_COORDINATE")
    values = tuple(candidates)
    if (
        not MIN_C200_CANDIDATES <= len(values) <= MAX_CANDIDATES
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise OovChargramBridgeG0WorkerError("TRACE_CANDIDATES")
    return _canonical_bytes(
        {"candidates": list(values), "ordinal": ordinal, "turn": turn}
    ) + b"\n"


def canonical_semantic_line(ordinal: int, turn: int, result: object) -> bytes:
    if (
        not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or not 1 <= ordinal <= SESSION_COUNT
        or not isinstance(turn, int)
        or isinstance(turn, bool)
        or not 1 <= turn <= TURN_COUNT
    ):
        raise OovChargramBridgeG0WorkerError("SEMANTIC_AUDIT_ORDINAL")
    try:
        payload = {
            "activated": result.activated,
            "bridge_sources": [
                _bridge_shape(value) for value in result.bridge_sources
            ],
            "candidates": list(result.candidates),
            "conflict_count": result.conflict_count,
            "filtered_identifiers": list(result.filtered_identifiers),
            "negative_violation_count": result.negative_violation_count,
            "novel_identifiers": list(result.novel_identifiers),
            "positive_conflict_count": result.positive_conflict_count,
            "source_records": [
                _source_shape(value) for value in result.source_records
            ],
            "tail": list(result.tail),
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT") from error
    return _canonical_bytes(
        {"ordinal": ordinal, "result": dict(payload), "turn": turn}
    ) + b"\n"


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise OovChargramBridgeG0WorkerError("EMPTY_AGGREGATE")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _latency_summary(
    values: Sequence[int], *, allow_empty: bool = False
) -> dict[str, int | float]:
    numbers = list(values)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in numbers
    ):
        raise OovChargramBridgeG0WorkerError("LATENCY_AGGREGATE")
    if not numbers:
        if not allow_empty:
            raise OovChargramBridgeG0WorkerError("EMPTY_AGGREGATE")
        return {
            "count": 0,
            "maximum_milliseconds": 0.0,
            "p50_milliseconds": 0.0,
            "p95_milliseconds": 0.0,
        }

    def milliseconds(value: float) -> float:
        return round(value / 1_000_000.0, 6)

    ordered = sorted(numbers)
    return {
        "count": len(numbers),
        "maximum_milliseconds": milliseconds(max(numbers)),
        "p50_milliseconds": milliseconds(
            ordered[max(0, math.ceil(0.50 * len(ordered)) - 1)]
        ),
        "p95_milliseconds": milliseconds(
            ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
        ),
    }


def _p95_nanoseconds(values: Sequence[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _pool_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise OovChargramBridgeG0WorkerError("POOL_AGGREGATE")
    return {
        "candidate_cells": sum(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 6),
        "min": min(values),
        "p50": int(_nearest_rank(values, 0.50)),
        "p95": int(_nearest_rank(values, 0.95)),
        "records": len(values),
    }


def _peak_rss_bytes() -> tuple[int | None, str]:
    if os.name == "nt":
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
                return int(counters.PeakWorkingSetSize), "windows_peak_working_set"
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    else:
        try:
            import resource

            observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return (
                observed if sys.platform == "darwin" else observed * 1024,
                "resource_ru_maxrss",
            )
        except (ImportError, OSError, TypeError, ValueError):
            pass
    return None, "unavailable"


def _validate_trace_paths(output: Path, nonce: str) -> Path:
    lexical = _lexical_windows_path(output)
    _guard_legacy_namespaces(lexical)
    parts = _lexical_path_key(lexical)
    root_parts = _lexical_path_key(EXPECTED_RUNTIME_ROOT)
    if (
        len(parts) != len(root_parts) + 2
        or parts[: len(root_parts)] != root_parts
        or RUN_ID_RE.fullmatch(lexical.parts[-2].casefold()) is None
        or lexical.suffix.casefold() != ".jsonl"
    ):
        raise OovChargramBridgeG0WorkerError("OUTPUT_PATH_NOT_ALLOWLISTED")
    _require_real_ancestry(output.parent, "TRACE_PARENT_UNSAFE")
    if not output.parent.is_dir() or _is_link_or_reparse(output.parent):
        raise OovChargramBridgeG0WorkerError("TRACE_PARENT_UNAVAILABLE")
    resolved_parent = output.parent.resolve(strict=True)
    expected_parent = Path(EXPECTED_RUNTIME_ROOT) / lexical.parts[-2]
    if not _same_path(resolved_parent, expected_parent):
        raise OovChargramBridgeG0WorkerError("TRACE_PARENT_IDENTITY")
    partial = output.with_name(f".{output.name}.{nonce}.partial")
    for path in (output, partial):
        if path.exists() or path.is_symlink() or _is_link_or_reparse(path):
            raise OovChargramBridgeG0WorkerError("TRACE_ALREADY_EXISTS")
    return partial


def _open_exclusive(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    return os.open(str(path), flags, 0o600)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OovChargramBridgeG0WorkerError("TRACE_SHORT_WRITE")
        view = view[written:]


def _publish_partial_exclusive(partial: Path, output: Path) -> None:
    _require_regular_file(partial, "PARTIAL_TRACE_UNAVAILABLE")
    if output.exists() or output.is_symlink() or _is_link_or_reparse(output):
        raise OovChargramBridgeG0WorkerError("TRACE_ALREADY_EXISTS")
    try:
        os.link(partial, output, follow_symlinks=False)
    except FileExistsError as error:
        raise OovChargramBridgeG0WorkerError("TRACE_ALREADY_EXISTS") from error
    except OSError as error:
        raise OovChargramBridgeG0WorkerError("ATOMIC_TRACE_PUBLISH") from error
    if not os.path.samefile(partial, output):
        raise OovChargramBridgeG0WorkerError("ATOMIC_TRACE_IDENTITY")


def _receipt_privacy_scan(
    value: object, *, catalog_ids: Iterable[str] = ()
) -> None:
    if _walk_keys(value) & FORBIDDEN_RECEIPT_KEYS:
        raise OovChargramBridgeG0WorkerError("RECEIPT_FORBIDDEN_KEY")
    payload = _canonical_bytes(value).decode("utf-8")
    if ASIN_SHAPE_RE.search(payload):
        raise OovChargramBridgeG0WorkerError("RECEIPT_IDENTIFIER")
    catalog = {str(identifier).casefold() for identifier in catalog_ids}
    tokens = {
        match.group(0).casefold()
        for match in CATALOG_IDENTIFIER_TOKEN_RE.finditer(payload)
    }
    if tokens & catalog:
        raise OovChargramBridgeG0WorkerError("RECEIPT_IDENTIFIER")


def _validate_arguments(args: argparse.Namespace, progress: WorkerProgress) -> None:
    nonce = str(args.nonce)
    stage_id = str(getattr(args, "stage_id", ""))
    if NONCE_RE.fullmatch(nonce) is None:
        raise OovChargramBridgeG0WorkerError("NONCE_INVALID")
    if stage_id not in STAGE_ID_ALLOWLIST:
        raise OovChargramBridgeG0WorkerError("STAGE_ID_INVALID")
    if args.session_limit not in ALLOWED_SESSION_LIMITS:
        raise OovChargramBridgeG0WorkerError("SESSION_LIMIT_INVALID")
    if not isinstance(getattr(args, "semantic_audit", False), bool) or not isinstance(
        getattr(args, "semantic_cache", False), bool
    ):
        raise OovChargramBridgeG0WorkerError("SEMANTIC_MODE_INVALID")
    semantic_audit = bool(getattr(args, "semantic_audit", False))
    semantic_cache = bool(getattr(args, "semantic_cache", False))
    if not semantic_audit:
        raise OovChargramBridgeG0WorkerError("SEMANTIC_AUDIT_REQUIRED")
    if semantic_cache and not semantic_audit:
        raise OovChargramBridgeG0WorkerError("SEMANTIC_AUDIT_REQUIRED")
    expected_worker = getattr(args, "expected_worker_blob", None)
    expected_union = getattr(args, "expected_union_blob", None)
    if semantic_audit:
        if (
            not isinstance(expected_worker, str)
            or GIT_BLOB_RE.fullmatch(expected_worker) is None
            or not isinstance(expected_union, str)
            or GIT_BLOB_RE.fullmatch(expected_union) is None
        ):
            raise OovChargramBridgeG0WorkerError("EXPECTED_SOURCE_BLOB_INVALID")
    elif expected_worker is not None or expected_union is not None:
        raise OovChargramBridgeG0WorkerError("EXPECTED_SOURCE_BLOB_UNSCOPED")
    progress.nonce = nonce
    progress.stage_id = stage_id
    progress.session_limit = int(args.session_limit)
    catalog = _lexical_windows_path(args.catalog)
    context = _lexical_windows_path(args.context)
    reference = _lexical_windows_path(args.c200_reference)
    output = _lexical_windows_path(args.trace_output)
    _guard_legacy_namespaces(output)
    if (
        _lexical_path_key(catalog) != _lexical_path_key(EXPECTED_CATALOG_PATH)
        or _lexical_path_key(context) != _lexical_path_key(EXPECTED_CONTEXT_PATH)
        or _lexical_path_key(reference)
        not in {
            _lexical_path_key(path) for path in EXPECTED_C200_REFERENCE_PATHS
        }
    ):
        raise OovChargramBridgeG0WorkerError("INPUT_PATH_NOT_ALLOWLISTED")
    output_parts = _lexical_path_key(output)
    root_parts = _lexical_path_key(EXPECTED_RUNTIME_ROOT)
    if (
        len(output_parts) != len(root_parts) + 2
        or output_parts[: len(root_parts)] != root_parts
        or output.suffix.casefold() != ".jsonl"
    ):
        raise OovChargramBridgeG0WorkerError("OUTPUT_PATH_NOT_ALLOWLISTED")


def _validate_end_identity(
    path: Path, expected: SourceIdentity, c200_contract: Any
) -> None:
    observed = c200_contract._raw_jsonl_identity(path, "sealed source")
    if observed.report() != expected.report() or observed.snapshot != expected.snapshot:
        raise OovChargramBridgeG0WorkerError("SOURCE_CHANGED")


def _cache_contract(value: object, *, after_close: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(CACHE_CAPACITIES):
        raise OovChargramBridgeG0WorkerError("CACHE_CONTRACT")
    expected_layer = {
        "capacity",
        "size",
        "hits",
        "misses",
        "inserts",
        "evictions",
        "closed",
    }
    normalized: dict[str, Any] = {}
    for name in CACHE_CAPACITIES:
        layer = value.get(name)
        if not isinstance(layer, Mapping) or set(layer) != expected_layer:
            raise OovChargramBridgeG0WorkerError("CACHE_CONTRACT")
        if (
            any(
                not isinstance(layer[key], int)
                or isinstance(layer[key], bool)
                or int(layer[key]) < 0
                for key in expected_layer - {"closed"}
            )
            or layer["capacity"] != CACHE_CAPACITIES[name]
            or layer["size"] > layer["capacity"]
            or layer["closed"] is not after_close
            or (after_close and layer["size"] != 0)
        ):
            raise OovChargramBridgeG0WorkerError("CACHE_CONTRACT")
        normalized[name] = {str(key): layer[key] for key in sorted(layer)}
    return normalized


def _cache_pair_contract(
    before_value: object, after_value: object
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _cache_contract(before_value, after_close=False)
    after = _cache_contract(after_value, after_close=True)
    stable_fields = ("hits", "misses", "inserts", "evictions", "capacity")
    if any(
        before[layer][field] != after[layer][field]
        for layer in CACHE_CAPACITIES
        for field in stable_fields
    ):
        raise OovChargramBridgeG0WorkerError("CACHE_CONTRACT")
    return before, after


def _bridge_contract(value: object) -> dict[str, Any]:
    expected = {
        "bridge_lookups",
        "fts_route_executions",
        "legacy_route_executions",
        "query_only_readback_one",
        "controlled_write_rejected",
        "write_guard_unchanged",
        "cache_clears",
        "closed",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    integer_fields = {
        "bridge_lookups",
        "fts_route_executions",
        "legacy_route_executions",
        "cache_clears",
    }
    if (
        any(
            not isinstance(value[name], int)
            or isinstance(value[name], bool)
            or int(value[name]) < 0
            for name in integer_fields
        )
        or value["legacy_route_executions"] != 0
        or any(
            not isinstance(value[name], bool)
            for name in (
                "query_only_readback_one",
                "controlled_write_rejected",
                "write_guard_unchanged",
                "closed",
            )
        )
    ):
        raise OovChargramBridgeG0WorkerError("EXPANSION_CONTRACT")
    return dict(value)


def run(
    args: argparse.Namespace,
    *,
    network_audit: OfflineNetworkAudit | None = None,
    runtime_loader: Callable[
        [], tuple[Any, Any, Callable[[], None], Callable[[], str]]
    ] = (
        _load_runtime_after_audit
    ),
    expander_factory: Callable[..., Any] | None = None,
    progress: WorkerProgress | None = None,
) -> dict[str, Any]:
    """Run one fresh target-free single-route bridge trace worker."""

    state = progress or WorkerProgress()
    _validate_arguments(args, state)
    state.phase = "ENVIRONMENT_AUDIT"
    environment = _freeze_cpu_environment()
    audit = network_audit or OfflineNetworkAudit()
    state.network_audit = audit
    sys.addaudithook(audit.hook)

    state.phase = "SOURCE_VALIDATION"
    source_start = _tracked_source_identities()
    _validate_semantic_source_blobs(args)
    state.source_identities = _source_identity_reports(source_start)
    (
        c200_contract,
        oov_chargram_bridge_g0,
        module_validate,
        registry_hash,
    ) = runtime_loader()
    _verify_imported_module_origins(
        c200_contract, oov_chargram_bridge_g0, module_validate, registry_hash
    )
    module_validate()
    presealed_sources = _tracked_source_identities()
    _validate_semantic_source_blobs(args)
    if not _same_identities(source_start, presealed_sources):
        raise OovChargramBridgeG0WorkerError("TRACKED_SOURCE_CHANGED_BEFORE_SEALED_ACCESS")

    state.phase = "TRACE_PREPARATION"
    partial_path = _validate_trace_paths(args.trace_output, state.nonce or "")
    state.partial_path = partial_path

    state.phase = "SEALED_SOURCE_VALIDATION"
    try:
        catalog_identity, catalog_ids = c200_contract._catalog_identity(args.catalog)
        context_identity, context_turns, redacted_messages = (
            c200_contract._validate_context_identity(args.context, catalog_ids)
        )
    except BaseException as error:
        raise OovChargramBridgeG0WorkerError("SEALED_SOURCE_SCHEMA") from error
    reference_identity, reference_cells = _validate_reference_identity(
        args.c200_reference, catalog_ids, c200_contract
    )
    if catalog_identity.report() != {
        "bytes": EXPECTED_CATALOG_BYTES,
        "rows": EXPECTED_CATALOG_ROWS,
        "sha256": EXPECTED_CATALOG_SHA256,
    }:
        raise OovChargramBridgeG0WorkerError("CATALOG_IDENTITY")
    if (
        context_identity.report()
        != {
            "bytes": EXPECTED_CONTEXT_BYTES,
            "rows": EXPECTED_CONTEXT_ROWS,
            "sha256": EXPECTED_CONTEXT_SHA256,
        }
        or context_turns != EXPECTED_CONTEXT_TURNS
        or redacted_messages != EXPECTED_REDACTED_MESSAGE_COUNT
        or reference_cells != EXPECTED_C200_CANDIDATE_CELLS
        or registry_hash() != EXPECTED_ATTRIBUTE_REGISTRY_SHA256
    ):
        raise OovChargramBridgeG0WorkerError("SEALED_SOURCE_IDENTITY")
    state.input_identities = {
        "catalog": catalog_identity.report(),
        "sealed_c200_reference": reference_identity.report(),
        "visible_context": context_identity.report(),
    }

    pools: dict[str, list[int]] = {
        name: []
        for name in (
            "expanded_union",
            "sealed_c200",
            "source_records",
            "bridge_sources",
            "fts_route",
            "route_novel",
            "route_filtered",
            "tail",
        )
    }
    context_parse_nanoseconds: list[int] = []
    extra_nanoseconds: list[int] = []
    turn_nanoseconds: list[int] = []
    bridge_lookup_nanoseconds: list[int] = []
    fts_route_nanoseconds: list[int] = []
    hard_mask_nanoseconds: list[int] = []
    valid_oov_source_records = 0
    bridge_mapping_records = 0
    fts_route_records = 0
    novel_candidate_cells = 0
    conflict_count = 0
    negative_violation_count = 0
    positive_conflict_count = 0
    tail_conflict_count = 0
    legacy_route_executions = 0
    evaluated_unique_novel_candidate_cells = 0
    reference_prefix_bytes = 0
    reference_prefix_cells = 0
    trace_digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    semantic_audit = bool(args.semantic_audit)
    semantic_cache = bool(args.semantic_cache)
    cache_before_close: Mapping[str, Any] | None = None
    cache_after_close: Mapping[str, Any] | None = None
    bridge_before_close: Mapping[str, Any] | None = None
    bridge_after_close: Mapping[str, Any] | None = None
    processed_turns = 0

    state.phase = "LEXICON_INITIALIZATION"
    factory = expander_factory or oov_chargram_bridge_g0.OovChargramBridgeG0Expander
    expander = factory(args.catalog, enabled=True, cache_enabled=semantic_cache)
    try:
        expander.validate()
        state.phase = "TRAJECTORY"
        descriptor = _open_exclusive(partial_path)
        try:
            with args.context.open("rb") as context_handle, args.c200_reference.open(
                "rb"
            ) as reference_handle:
                for ordinal in range(1, args.session_limit + 1):
                    context_line = context_handle.readline()
                    if not context_line:
                        raise OovChargramBridgeG0WorkerError("CONTEXT_ENDED_EARLY")
                    context_parse_started = time.perf_counter_ns()
                    try:
                        contexts = c200_contract._parse_context_container(
                            context_line, catalog_ids
                        )
                    except BaseException as error:
                        raise OovChargramBridgeG0WorkerError("CONTEXT_SCHEMA") from error
                    context_elapsed = time.perf_counter_ns() - context_parse_started
                    context_parse_nanoseconds.append(context_elapsed)
                    context_share = context_elapsed // len(contexts)
                    for turn, context in enumerate(contexts, start=1):
                        turn_started = time.perf_counter_ns()
                        reference_line = reference_handle.readline()
                        if not reference_line:
                            raise OovChargramBridgeG0WorkerError("C200_ENDED_EARLY")
                        sealed_c200 = parse_c200_reference_line(
                            reference_line,
                            ordinal=ordinal,
                            turn=turn,
                            catalog_ids=catalog_ids,
                        )
                        extra_started = time.perf_counter_ns()
                        result = expander.expand(
                            sealed_c200,
                            category_text=context["category_text"],
                            active_terms=context["active_terms"],
                            excluded_terms=context["excluded_terms"],
                            current_version=context["version"],
                            records=context["active_records"],
                        )
                        extra_elapsed = time.perf_counter_ns() - extra_started
                        candidates = validate_expansion_result(
                            result, sealed_c200, catalog_ids
                        )
                        lookup_elapsed, route_elapsed, mask_elapsed = (
                            expansion_timing_contract(result)
                        )
                        if lookup_elapsed + route_elapsed + mask_elapsed > extra_elapsed:
                            raise OovChargramBridgeG0WorkerError(
                                "LATENCY_CONTAINMENT"
                            )
                        if semantic_audit:
                            semantic_digest.update(
                                canonical_semantic_line(ordinal, turn, result)
                            )
                        trace_line = canonical_trace_line(ordinal, turn, candidates)
                        _write_all(descriptor, trace_line)
                        trace_digest.update(trace_line)
                        state.partial_bytes += len(trace_line)
                        state.partial_rows += 1
                        state.partial_sha256 = trace_digest.hexdigest()
                        processed_turns += 1
                        reference_prefix_bytes += len(reference_line)
                        reference_prefix_cells += len(sealed_c200)
                        pools["expanded_union"].append(len(candidates))
                        pools["sealed_c200"].append(len(sealed_c200))
                        pools["source_records"].append(len(result.source_records))
                        pools["bridge_sources"].append(len(result.bridge_sources))
                        pools["fts_route"].append(
                            max((len(route.identifiers) for route in result.routes), default=0)
                        )
                        pools["route_novel"].append(len(result.novel_identifiers))
                        pools["route_filtered"].append(
                            len(result.filtered_identifiers)
                        )
                        pools["tail"].append(len(result.tail))
                        extra_nanoseconds.append(extra_elapsed)
                        turn_nanoseconds.append(
                            time.perf_counter_ns() - turn_started + context_share
                        )
                        bridge_lookup_nanoseconds.append(lookup_elapsed)
                        fts_route_nanoseconds.extend(
                            route.latency_ns
                            for route in result.routes
                            if route.latency_ns > 0
                        )
                        hard_mask_nanoseconds.append(mask_elapsed)
                        valid_oov_source_records += len(result.source_records)
                        bridge_mapping_records += len(result.bridge_sources)
                        fts_route_records += len(result.routes)
                        novel_candidate_cells += len(result.novel_identifiers)
                        conflict_count += int(result.conflict_count)
                        negative_violation_count += int(
                            result.negative_violation_count
                        )
                        positive_conflict_count += int(
                            result.positive_conflict_count
                        )
                        tail_conflict_count += int(result.tail_conflict_count)
                        legacy_route_executions += int(result.legacy_route_executions)
                        evaluated_unique_novel_candidate_cells += len(
                            result.novel_identifiers
                        )
                    state.last_completed_session = ordinal
                if args.session_limit == SESSION_COUNT:
                    if context_handle.read(1) or reference_handle.read(1):
                        raise OovChargramBridgeG0WorkerError("SEALED_SOURCE_EXCESS_ROWS")
            os.fsync(descriptor)
            bridge_before_close = expander.route_diagnostics()
            if semantic_cache:
                cache_before_close = expander.cache_diagnostics()
        finally:
            os.close(descriptor)
    finally:
        if sys.exc_info()[0] is None:
            state.phase = "SQLITE_CLOSE"
        try:
            expander.close()
        finally:
            state.sqlite_closed = bool(getattr(expander, "closed", False))
            route_diagnostics = getattr(expander, "route_diagnostics", None)
            if callable(route_diagnostics):
                bridge_after_close = route_diagnostics()
            if semantic_cache:
                cache_diagnostics = getattr(expander, "cache_diagnostics", None)
                if callable(cache_diagnostics):
                    cache_after_close = cache_diagnostics()

    if not state.sqlite_closed:
        raise OovChargramBridgeG0WorkerError("SQLITE_NOT_CLOSED")
    expected_records = args.session_limit * TURN_COUNT
    if (
        state.last_completed_session != args.session_limit
        or processed_turns != expected_records
        or state.partial_rows != expected_records
    ):
        raise OovChargramBridgeG0WorkerError("TRAJECTORY_INCOMPLETE")
    bridge_before_physical = _bridge_contract(bridge_before_close)
    bridge_after_physical = _bridge_contract(bridge_after_close)
    if (
        bridge_before_physical["closed"] is not False
        or bridge_after_physical["closed"] is not True
        or bridge_before_physical["bridge_lookups"] > valid_oov_source_records
        or bridge_before_physical["fts_route_executions"] > fts_route_records
        or bridge_before_physical["fts_route_executions"]
        != len(fts_route_nanoseconds)
        or bridge_after_physical["bridge_lookups"]
        != bridge_before_physical["bridge_lookups"]
        or bridge_after_physical["fts_route_executions"]
        != bridge_before_physical["fts_route_executions"]
        or bridge_after_physical["legacy_route_executions"]
        != bridge_before_physical["legacy_route_executions"]
        or bridge_before_physical["cache_clears"] != 0
        or bridge_after_physical["cache_clears"] != 1
        or any(
            bridge_before_physical[name] is not True
            or bridge_after_physical[name] is not True
            for name in (
                "query_only_readback_one",
                "controlled_write_rejected",
                "write_guard_unchanged",
            )
        )
        or len(hard_mask_nanoseconds) != expected_records
        or len(bridge_lookup_nanoseconds) != expected_records
        or legacy_route_executions != 0
    ):
        raise OovChargramBridgeG0WorkerError("BRIDGE_EXECUTION_AUDIT")
    if not semantic_cache and (
        bridge_before_physical["bridge_lookups"] != valid_oov_source_records
        or bridge_before_physical["fts_route_executions"] != fts_route_records
    ):
        raise OovChargramBridgeG0WorkerError("BRIDGE_EXECUTION_AUDIT")

    # These are deliberately physical counters. Cached and uncached traces must
    # be semantically identical, while their execution diagnostics may differ.
    bridge_diagnostics = bridge_before_physical

    state.phase = "SOURCE_REVALIDATION"
    _validate_end_identity(args.catalog, catalog_identity, c200_contract)
    _validate_end_identity(args.context, context_identity, c200_contract)
    _validate_end_identity(args.c200_reference, reference_identity, c200_contract)
    source_end = _tracked_source_identities()
    _validate_semantic_source_blobs(args)
    _assert_legacy_runtime_absent()
    if not _same_identities(source_start, source_end):
        raise OovChargramBridgeG0WorkerError("TRACKED_SOURCE_CHANGED")

    state.phase = "RESOURCE_VALIDATION"
    if audit.attempt_count:
        raise OovChargramBridgeG0WorkerError("NETWORK_ATTEMPT")
    if any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in GPU_RUNTIME_PREFIXES
    ):
        raise OovChargramBridgeG0WorkerError("GPU_RUNTIME_PRESENT")
    bridge_latency = _latency_summary(
        bridge_lookup_nanoseconds, allow_empty=True
    )
    route_latency = _latency_summary(fts_route_nanoseconds, allow_empty=True)
    mask_latency = _latency_summary(hard_mask_nanoseconds)
    extra_latency = _latency_summary(extra_nanoseconds)
    context_latency = _latency_summary(context_parse_nanoseconds)
    turn_latency = _latency_summary(turn_nanoseconds)
    pool_summaries = {name: _pool_summary(values) for name, values in pools.items()}
    peak_rss, rss_backend = _peak_rss_bytes()
    wall_seconds = time.perf_counter() - state.wall_started
    turns_per_second = processed_turns / wall_seconds if wall_seconds > 0.0 else 0.0
    candidate_ratio = (
        pool_summaries["expanded_union"]["candidate_cells"] / reference_prefix_cells
    )
    trace_ratio = state.partial_bytes / reference_prefix_bytes
    full_run = args.session_limit == SESSION_COUNT
    if (
        tail_conflict_count != 0
        or legacy_route_executions != 0
        or _p95_nanoseconds(bridge_lookup_nanoseconds)
        > int(MAX_ROUTE_P95_MILLISECONDS * 1_000_000)
        or _p95_nanoseconds(fts_route_nanoseconds)
        > int(MAX_ROUTE_P95_MILLISECONDS * 1_000_000)
        or _p95_nanoseconds(hard_mask_nanoseconds)
        > int(MAX_MASK_P95_MILLISECONDS * 1_000_000)
        or _p95_nanoseconds(extra_nanoseconds)
        > int(MAX_EXTRA_P95_MILLISECONDS * 1_000_000)
        or _p95_nanoseconds(turn_nanoseconds)
        > int(MAX_TURN_P95_MILLISECONDS * 1_000_000)
        or peak_rss is None
        or not 0 < peak_rss <= MAX_WORKING_SET_BYTES
        or wall_seconds > MAX_WALL_SECONDS
        or (full_run and turns_per_second < MIN_FULL_TURNS_PER_SECOND)
        or (full_run and candidate_ratio > MAX_CANDIDATE_CELL_RATIO)
        or (full_run and trace_ratio > MAX_TRACE_BYTE_RATIO)
    ):
        raise OovChargramBridgeG0WorkerError("RESOURCE_GATE")

    environment["network_attempt_count"] = audit.attempt_count
    summary: dict[str, Any] = {
        "activation": {
            "valid_oov_source_records": valid_oov_source_records,
            "bridge_mapping_records": bridge_mapping_records,
            "fts_route_records": fts_route_records,
            "novel_candidate_cells": novel_candidate_cells,
            "legacy_route_executions": legacy_route_executions,
            "v222_route_executions": 0,
        },
        "configuration": {
            "candidate_cap": MAX_CANDIDATES,
            "default_off": True,
            "diagnostic_only": True,
            "maximum_bridge_sources": 6,
            "maximum_bridge_tokens_per_source": 4,
            "route_limit": ROUTE_LIMIT,
            "served_top10_unchanged": True,
            "single_route_only": True,
            "stable_append_after_complete_variable_c200": True,
        },
        "environment": environment,
        "input_identities": state.input_identities,
        "latency": {
            "context_container_parse": context_latency,
            "bridge_lookup": bridge_latency,
            "fts_route": route_latency,
            "hard_conflict_mask": mask_latency,
            "extra_bridge_and_mask": extra_latency,
            "per_turn": turn_latency,
        },
        "lifecycle": {
            "atomic_exclusive_trace_publish": True,
            "inputs_unchanged_before_trace_publish": True,
            "legacy_runtime_absent": True,
            "partial_fsynced_and_closed_before_trace_publish": True,
            "runtime_module_origins_verified": True,
            "source_files_unchanged_before_trace_publish": True,
            "sqlite_closed_before_trace_publish": state.sqlite_closed,
        },
        "mask": {
            "evaluated_unique_novel_candidate_cells": (
                evaluated_unique_novel_candidate_cells
            ),
            "removed_explicit_conflicts": conflict_count,
            "tail_duplicate_count": 0,
            "tail_explicit_conflict_count": tail_conflict_count,
        },
        "pool_lengths": pool_summaries,
        "prefix_integrity": {
            "c200_duplicate_count": 0,
            "c200_loss_count": 0,
            "c200_reorder_count": 0,
            "top10_change_count": 0,
        },
        "query_only": {
            "controlled_write_rejected": True,
            "query_only_readback_one": True,
            "write_guard_unchanged": True,
        },
        "processed_sessions": args.session_limit,
        "processed_turns": expected_records,
        "resources": {
            "candidate_cell_ratio_over_c200": round(float(candidate_ratio), 6),
            "device": "CPU",
            "provider": "SQLite FTS5 + CPython",
            "gpu_peak_bytes": 0,
            "gpu_used": False,
            "network_attempt_count": audit.attempt_count,
            "peak_working_set_backend": rss_backend,
            "peak_working_set_bytes": peak_rss,
            "trace_byte_ratio_over_c200": round(float(trace_ratio), 6),
            "turns_per_second": round(float(turns_per_second), 6),
            "wall_seconds": round(wall_seconds, 6),
        },
        "bridge_diagnostics": bridge_diagnostics,
        "session_limit": args.session_limit,
        "source_identities": state.source_identities,
    }
    if semantic_audit:
        summary["semantic_trace"] = {
            "rows": expected_records,
            "sha256": semantic_digest.hexdigest(),
        }
    if semantic_cache:
        if cache_before_close is None or cache_after_close is None:
            raise OovChargramBridgeG0WorkerError("CACHE_DIAGNOSTICS_INCOMPLETE")
        cache_before, cache_after = _cache_pair_contract(
            cache_before_close, cache_after_close
        )
        if (
            cache_before["oov_bridge"]["hits"]
            + cache_before["oov_bridge"]["misses"]
            != valid_oov_source_records
            or cache_before["oov_bridge"]["misses"]
            != bridge_before_physical["bridge_lookups"]
            or cache_before["fts_route"]["hits"]
            + cache_before["fts_route"]["misses"]
            != fts_route_records
            or cache_before["fts_route"]["misses"]
            != bridge_before_physical["fts_route_executions"]
        ):
            raise OovChargramBridgeG0WorkerError("CACHE_CONTRACT")
        summary["cache"] = {
            "before_close": cache_before,
            "after_close": cache_after,
        }
    receipt: dict[str, Any] = {
        "error_code": "NONE",
        "kind": "receipt",
        "last_completed_session": state.last_completed_session,
        "nonce": state.nonce,
        "phase": "COMPLETE",
        "record_count": expected_records,
        "schema_version": SCHEMA_VERSION,
        "status": "SUCCESS",
        "summary": summary,
        "trace_bytes": state.partial_bytes,
        "trace_sha256": state.partial_sha256,
    }
    _receipt_privacy_scan(receipt, catalog_ids=catalog_ids)

    state.phase = "TRACE_PUBLICATION"
    _publish_partial_exclusive(partial_path, args.trace_output)
    published = _raw_identity(args.trace_output, rows=True)
    if (
        published.bytes != state.partial_bytes
        or published.rows != expected_records
        or published.sha256 != state.partial_sha256
    ):
        raise OovChargramBridgeG0WorkerError("PUBLISHED_TRACE_IDENTITY")
    return receipt


def _worker_error_code(error: BaseException, progress: WorkerProgress) -> str:
    """Collapse internal detail into the preregistered finite codebook."""

    detail = (
        error.error_code
        if isinstance(error, OovChargramBridgeG0WorkerError)
        else ""
    )
    phase = progress.phase if progress.phase in WORKER_PHASE_ALLOWLIST else "UNKNOWN"
    if progress.network_audit is not None and progress.network_audit.attempt_count:
        return "NETWORK_ATTEMPT"
    if detail == "GPU_RUNTIME_PRESENT":
        return "GPU_RUNTIME_PRESENT"
    if detail in {"RECEIPT_FORBIDDEN_KEY", "RECEIPT_IDENTIFIER"}:
        return "PRIVACY_SCAN"
    if "CACHE" in detail:
        return "CACHE_CONTRACT"
    if "QUERY_ONLY" in detail or "WRITE_GUARD" in detail:
        return "QUERY_ONLY_CONTRACT"
    if "LEXICON_RESOURCE" in detail or type(error).__name__ == (
        "OovChargramBridgeG0ResourceError"
    ):
        return "LEXICON_RESOURCE"
    if phase == "ARGUMENT_VALIDATION":
        return "ARGUMENT_INVALID"
    if phase == "ENVIRONMENT_AUDIT":
        return "ENVIRONMENT_INVALID"
    if phase == "SOURCE_VALIDATION":
        if isinstance(error, ImportError) or any(
            marker in detail for marker in ("IMPORT", "MODULE")
        ):
            return "SOURCE_IMPORT"
        return "SOURCE_IDENTITY"
    if phase == "TRACE_PREPARATION":
        return "TRACE_PATH"
    if phase == "SEALED_SOURCE_VALIDATION":
        if any(
            marker in detail
            for marker in ("IDENTITY", "ROW_COUNT", "CANDIDATE_CELLS")
        ):
            return "SEALED_SOURCE_IDENTITY"
        return "SEALED_SOURCE_SCHEMA"
    if phase == "LEXICON_INITIALIZATION":
        return "LEXICON_SCHEMA"
    if phase == "TRAJECTORY":
        if "CONTEXT" in detail:
            return "CONTEXT_SCHEMA"
        if "C200" in detail or "REFERENCE" in detail:
            return "C200_SCHEMA"
        return "EXPANSION_CONTRACT"
    if phase == "SQLITE_CLOSE":
        return "SQLITE_CLOSE"
    if phase == "SOURCE_REVALIDATION":
        return "SOURCE_IDENTITY"
    if phase == "RESOURCE_VALIDATION":
        return "RESOURCE_GATE"
    if phase == "TRACE_PUBLICATION":
        return "TRACE_PUBLICATION"
    if isinstance(error, (MemoryError, OSError)):
        return "RESOURCE_GATE"
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "UNCLASSIFIED"
    return "INTERNAL_INVARIANT"


def _failure_stack(error: BaseException) -> tuple[str, str]:
    """Return only a frozen site code and hash over allowlisted frame triples."""

    role_by_filename = {
        Path(__file__).name.casefold(): "worker",
        CORE_PATH.name.casefold(): "core",
        C200_CONTRACT_PATH.name.casefold(): "c200",
    }
    triples: list[tuple[str, str, int]] = []
    site = "SITE_0000"
    frames = traceback.extract_tb(error.__traceback__) if error.__traceback__ else []
    for frame in frames:
        role = role_by_filename.get(Path(frame.filename).name.casefold())
        key = (role, frame.name) if role is not None else None
        line = int(frame.lineno)
        if key in _FAILURE_SITE_ALLOWLIST and line > 0:
            triples.append((str(role), frame.name, line))
            site = _FAILURE_SITE_ALLOWLIST[key]  # type: ignore[index]
    if not triples:
        return "SITE_0000", "0" * 64
    return site, hashlib.sha256(_canonical_bytes(triples)).hexdigest()


def _progress_bucket(progress: WorkerProgress) -> str:
    completed = progress.last_completed_session
    limit = progress.session_limit
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
        return "UNKNOWN"
    if completed == 0:
        return "NONE"
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        return "UNKNOWN"
    if completed == limit:
        return "COMPLETE"
    if completed < limit:
        return "PARTIAL"
    return "UNKNOWN"


def _wall_time_bucket(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        return "UNKNOWN"
    if seconds < 1.0:
        return "LT_1S"
    if seconds < 10.0:
        return "1_TO_10S"
    if seconds < 60.0:
        return "10_TO_60S"
    if seconds < 300.0:
        return "60_TO_300S"
    return "GE_300S"


def _rss_bucket(value: int | None) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return "UNKNOWN"
    if value <= 268_435_456:
        return "LE_256M"
    if value <= 536_870_912:
        return "LE_512M"
    if value <= 1_073_741_824:
        return "LE_1G"
    if value <= 1_610_612_736:
        return "LE_1_5G"
    return "GT_1_5G"


def _error_receipt(error: BaseException, progress: WorkerProgress) -> dict[str, Any]:
    peak_rss, _backend = _peak_rss_bytes()
    wall_seconds = time.perf_counter() - progress.wall_started
    site, stack_hash = _failure_stack(error)
    phase = progress.phase if progress.phase in WORKER_PHASE_ALLOWLIST else "UNKNOWN"
    code = _worker_error_code(error, progress)
    if code not in WORKER_ERROR_CODE_ALLOWLIST:
        code = "UNCLASSIFIED"
    stage = (
        progress.stage_id
        if progress.stage_id in STAGE_ID_ALLOWLIST
        else "preclaim_synthetic"
    )
    receipt: dict[str, Any] = {
        "child_exit_code": 1,
        "failure_origin": "worker",
        "failure_site_id": site,
        "kind": "failure",
        "progress_bucket": _progress_bucket(progress),
        "rss_bucket": _rss_bucket(peak_rss),
        "schema_version": FAILURE_SCHEMA_VERSION,
        "stack_hash": stack_hash,
        "stage_id": stage,
        "status": "ERROR",
        "stderr_nonempty": False,
        "wall_time_bucket": _wall_time_bucket(wall_seconds),
        "worker_error_code": code,
        "worker_phase": phase,
    }
    _receipt_privacy_scan(receipt)
    return receipt


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise OovChargramBridgeG0WorkerError("ARGUMENT_INVALID")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-id", required=True, choices=tuple(sorted(STAGE_ID_ALLOWLIST))
    )
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--c200-reference", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument(
        "--session-limit", type=int, choices=ALLOWED_SESSION_LIMITS, required=True
    )
    parser.add_argument("--semantic-audit", action="store_true")
    parser.add_argument("--semantic-cache", action="store_true")
    parser.add_argument("--expected-worker-blob")
    parser.add_argument("--expected-union-blob")
    return parser


def _entrypoint_self_check(argv: Sequence[str]) -> int | None:
    if "--entrypoint-self-check" not in argv:
        return None
    parser = _SafeArgumentParser(add_help=False)
    parser.add_argument("--entrypoint-self-check", action="store_true")
    parser.add_argument(
        "--stage-id", required=True, choices=("preclaim_synthetic",)
    )
    parser.add_argument(
        "--require-module",
        default="starter.oov_chargram_bridge_g0",
        choices=("starter.oov_chargram_bridge_g0",),
    )
    arguments = parser.parse_args(argv)
    _assert_legacy_runtime_absent()
    required = importlib.import_module(arguments.require_module)
    contract = importlib.import_module("scripts.c200_candidate_worker")
    attributes = importlib.import_module("starter.attributes")
    evaluator = importlib.import_module("evaluator.local_evaluator")
    _verify_imported_module_origins(
        contract,
        required,
        getattr(required, "validate", None),
        getattr(attributes, "attribute_registry_sha256", None),
    )
    evaluator_origin = getattr(evaluator, "__file__", None)
    if (
        getattr(evaluator, "__name__", None) != "evaluator.local_evaluator"
        or not isinstance(evaluator_origin, str)
        or not _same_path(
            Path(evaluator_origin), PROJECT_ROOT / "evaluator" / "local_evaluator.py"
        )
    ):
        raise OovChargramBridgeG0WorkerError("EVALUATOR_MODULE_IDENTITY")
    payload = {
        "c200_contract_imported": True,
        "evaluator_imported": True,
        "legacy_runtime_absent": True,
        "project_root_bootstrapped": str(PROJECT_ROOT) in sys.path,
        "required_module": arguments.require_module,
        "status": "ENTRYPOINT_SELF_CHECK_PASS",
    }
    sys.stdout.buffer.write(_canonical_bytes(payload) + b"\n")
    sys.stdout.buffer.flush()
    return 0


def _prescan_stage_id(arguments: Sequence[str]) -> str:
    for index, value in enumerate(arguments):
        candidate: str | None = None
        if value == "--stage-id" and index + 1 < len(arguments):
            candidate = str(arguments[index + 1])
        elif value.startswith("--stage-id="):
            candidate = value.split("=", 1)[1]
        if candidate in STAGE_ID_ALLOWLIST:
            return str(candidate)
    return "preclaim_synthetic"


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        self_check = _entrypoint_self_check(raw_arguments)
        if self_check is not None:
            return self_check
    except BaseException:
        return 2
    progress = WorkerProgress(stage_id=_prescan_stage_id(raw_arguments))
    try:
        receipt = run(_parser().parse_args(raw_arguments), progress=progress)
        exit_code = 0
    except BaseException as error:
        receipt = _error_receipt(error, progress)
        exit_code = 1
    sys.stdout.buffer.write(_canonical_bytes(receipt) + b"\n")
    sys.stdout.buffer.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
