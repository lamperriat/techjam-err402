"""Target-free worker for the frozen v2.21 dual-view sparse G0 probe.

This module is deliberately a diagnostic worker, not a serving entrypoint.  It
streams one complete candidate trace while preserving every sealed C200 row as
an ordered prefix.  Runtime imports are limited to the frozen C200 contract and
``starter.sparse_union_g0``; the legacy sparse-multiview implementation is never
imported or executed.
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


SCHEMA_VERSION = "small-ranker-v2.21-dual-view-rrf-g0-worker-summary.v1"
CONTEXT_SCHEMA_VERSION = "small-ranker-visible-context.v1"
SESSION_COUNT = 2_000
TURN_COUNT = 10
RECORD_COUNT = SESSION_COUNT * TURN_COUNT
ROUTE_LIMIT = 120
MAX_CANDIDATES = 400
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
    "5ee89ebda59c3dbf973fc3cd3f127ec34f47d1fa"
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
EXPECTED_RUNTIME_ROOT = PureWindowsPath(r"D:\tiktok\.v221_runtime")
PREREGISTRATION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "small_ranker_v2_21.dual_view_rrf_g0_preregistration.json"
)
CORE_PATH = PROJECT_ROOT / "starter" / "sparse_union_g0.py"
C200_CONTRACT_PATH = PROJECT_ROOT / "scripts" / "c200_candidate_worker.py"

MAX_EXTRA_P95_MILLISECONDS = 100.0
MAX_TURN_P95_MILLISECONDS = 400.0
MAX_WORKING_SET_BYTES = 1_610_612_736
MAX_WALL_SECONDS = 1_800.0
MAX_CANDIDATE_CELL_RATIO = 2.0
MAX_TRACE_BYTE_RATIO = 2.1
CACHE_CAPACITIES = {
    "fts_route": 512,
    "product_view": 4_096,
    "mask_decision": 16_384,
}

NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
RUN_ID_RE = re.compile(r"[0-9a-z][0-9a-z._-]{7,127}\Z")
GIT_BLOB_RE = re.compile(r"[0-9a-f]{40}\Z")
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
        "starter.sparse_multiview",
        "scripts.sparse_multiview_candidate_worker",
    }
)


class SparseUnionG0WorkerError(RuntimeError):
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
        raise SparseUnionG0WorkerError("NON_CANONICAL_JSON") from error


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SparseUnionG0WorkerError("DUPLICATE_JSON_KEY")
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
            raise SparseUnionG0WorkerError(error_code)


def _require_regular_file(path: Path, error_code: str) -> None:
    _require_real_ancestry(path, error_code)
    if not path.is_file():
        raise SparseUnionG0WorkerError(error_code)


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
        raise SparseUnionG0WorkerError("PATH_NOT_LEXICALLY_CANONICAL")
    result = PureWindowsPath(raw)
    if not result.is_absolute():
        raise SparseUnionG0WorkerError("PATH_NOT_ABSOLUTE")
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
        "small-ranker-v2.19",
        "small-ranker-v2.20",
        "small-ranker-v2.20b",
    )
    if any(marker in joined for marker in forbidden):
        raise SparseUnionG0WorkerError("LEGACY_NAMESPACE_DENIED")


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
        raise SparseUnionG0WorkerError("SOURCE_CHANGED_DURING_READ")
    return SourceIdentity(byte_count, row_count, digest.hexdigest(), after)


def _raw_git_blob_sha1(path: Path) -> str:
    _require_regular_file(path, "PINNED_SOURCE_UNAVAILABLE")
    before = _snapshot(path)
    with path.open("rb") as handle:
        working_tree_bytes = handle.read()
    after = _snapshot(path)
    if before != after or len(working_tree_bytes) != before[0]:
        raise SparseUnionG0WorkerError("PINNED_SOURCE_CHANGED_DURING_READ")
    blob_bytes = working_tree_bytes.replace(b"\r\n", b"\n")
    digest = hashlib.sha1()
    digest.update(f"blob {len(blob_bytes)}\0".encode("ascii"))
    digest.update(blob_bytes)
    return digest.hexdigest()


def _tracked_source_identities() -> dict[str, SourceIdentity]:
    paths = {
        "preregistration": PREREGISTRATION_PATH,
        "scripts/c200_candidate_worker.py": C200_CONTRACT_PATH,
        "scripts/sparse_union_g0_worker.py": Path(__file__).resolve(),
        "starter/sparse_union_g0.py": CORE_PATH,
    }
    identities = {name: _raw_identity(path) for name, path in paths.items()}
    if (
        _raw_git_blob_sha1(PREREGISTRATION_PATH)
        != EXPECTED_PREREGISTRATION_BLOB_SHA1
        or _raw_git_blob_sha1(C200_CONTRACT_PATH)
        != EXPECTED_C200_CONTRACT_BLOB_SHA1
    ):
        raise SparseUnionG0WorkerError("PINNED_GIT_BLOB_IDENTITY")
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
        raise SparseUnionG0WorkerError("EXPECTED_SOURCE_BLOB_MISMATCH")


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
        raise SparseUnionG0WorkerError("LEGACY_RUNTIME_PRESENT")


def _load_runtime_after_audit() -> tuple[Any, Any, Callable[[], str]]:
    _assert_legacy_runtime_absent()
    try:
        c200_contract = importlib.import_module("scripts.c200_candidate_worker")
        sparse_union_g0 = importlib.import_module("starter.sparse_union_g0")
    except ImportError as error:
        raise SparseUnionG0WorkerError("RUNTIME_IMPORT") from error
    registry_hash = getattr(sparse_union_g0, "attribute_registry_sha256", None)
    if not callable(registry_hash):
        raise SparseUnionG0WorkerError("REGISTRY_API")
    _assert_legacy_runtime_absent()
    return c200_contract, sparse_union_g0, registry_hash


def _verify_imported_module_origins(
    c200_contract: Any,
    sparse_union_g0: Any,
    registry_hash: Callable[[], str],
) -> None:
    if (
        getattr(c200_contract, "__name__", None)
        != "scripts.c200_candidate_worker"
        or getattr(sparse_union_g0, "__name__", None) != "starter.sparse_union_g0"
        or getattr(registry_hash, "__module__", None)
        not in {"starter.sparse_union_g0", "starter.attributes"}
    ):
        raise SparseUnionG0WorkerError("RUNTIME_MODULE_IDENTITY")
    expected_origins = {
        "scripts.c200_candidate_worker": C200_CONTRACT_PATH,
        "starter.sparse_union_g0": CORE_PATH,
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
            raise SparseUnionG0WorkerError("RUNTIME_MODULE_ORIGIN")
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
        raise SparseUnionG0WorkerError("RUNTIME_ENVIRONMENT")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    loaded_gpu_modules = sorted(
        name for name in ("cupy", "tensorflow", "torch") if name in sys.modules
    )
    if loaded_gpu_modules:
        raise SparseUnionG0WorkerError("GPU_RUNTIME_PRESENT")
    return {
        **actual,
        "cwd_is_project_root": True,
        "executable_is_frozen": True,
        "gpu_peak_bytes": 0,
        "gpu_used": False,
        "network_attempt_count": 0,
    }


def _validate_c200_values(
    candidates: object, catalog_ids: Iterable[str]
) -> tuple[str, ...]:
    if not isinstance(candidates, (list, tuple)):
        raise SparseUnionG0WorkerError("C200_SCHEMA")
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
        raise SparseUnionG0WorkerError("C200_SCHEMA")
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
                SparseUnionG0WorkerError("C200_NONFINITE")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SparseUnionG0WorkerError("C200_JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"c200", "ordinal", "turn"}
        or value.get("ordinal") != ordinal
        or isinstance(value.get("ordinal"), bool)
        or value.get("turn") != turn
        or isinstance(value.get("turn"), bool)
    ):
        raise SparseUnionG0WorkerError("C200_ORDER")
    candidates = _validate_c200_values(value.get("c200"), catalog_ids)
    expected = _canonical_bytes(
        {"c200": list(candidates), "ordinal": ordinal, "turn": turn}
    ) + b"\n"
    if line != expected:
        raise SparseUnionG0WorkerError("C200_CANONICAL")
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
                raise SparseUnionG0WorkerError("C200_ROW_COUNT")
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
        raise SparseUnionG0WorkerError("C200_REFERENCE_IDENTITY")
    return identity, candidate_cells


def _route_tuple(value: object, catalog: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SparseUnionG0WorkerError("EXPANSION_SCHEMA")
    route = tuple(value)
    if (
        len(route) > ROUTE_LIMIT
        or len(route) != len(set(route))
        or any(not isinstance(identifier, str) or identifier not in catalog for identifier in route)
    ):
        raise SparseUnionG0WorkerError("ROUTE_CONTRACT")
    return route


def _subsequence(values: Sequence[str], container: Sequence[str]) -> bool:
    iterator = iter(container)
    return all(any(value == candidate for candidate in iterator) for value in values)


def _expected_fused_tail(
    prefix: Sequence[str],
    category_filtered: Sequence[str],
    positive_core_filtered: Sequence[str],
) -> tuple[str, ...]:
    prefix_set = set(prefix)
    category_rank = {
        identifier: rank
        for rank, identifier in enumerate(category_filtered, start=1)
        if identifier not in prefix_set
    }
    core_rank = {
        identifier: rank
        for rank, identifier in enumerate(positive_core_filtered, start=1)
        if identifier not in prefix_set
    }
    identifiers = set(category_rank) | set(core_rank)

    def key(identifier: str) -> tuple[object, ...]:
        category = category_rank.get(identifier)
        core = core_rank.get(identifier)
        score = sum(
            (
                Fraction(1, 60 + rank)
                for rank in (category, core)
                if rank is not None
            ),
            Fraction(0, 1),
        )
        support = int(category is not None) + int(core is not None)
        minimum = min(rank for rank in (category, core) if rank is not None)
        return (
            -score,
            -support,
            minimum,
            category if category is not None else ROUTE_LIMIT + 1,
            core if core is not None else ROUTE_LIMIT + 1,
            identifier,
        )

    capacity = max(0, MAX_CANDIDATES - len(prefix))
    return tuple(sorted(identifiers, key=key)[:capacity])


def validate_expansion_result(
    result: object,
    sealed_c200: object,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    """Validate both isolated routes, mask order, exact RRF, and C200 prefix."""

    prefix = _validate_c200_values(sealed_c200, catalog_ids)
    required = (
        "activated",
        "candidates",
        "category_filtered",
        "category_novel",
        "category_route",
        "conflict_count",
        "enabled",
        "legacy_route_executions",
        "positive_core_filtered",
        "positive_core_novel",
        "positive_core_route",
        "prefix",
        "queries",
        "tail",
        "tail_conflict_count",
    )
    if any(not hasattr(result, name) for name in required):
        raise SparseUnionG0WorkerError("EXPANSION_SCHEMA")
    catalog = (
        catalog_ids
        if isinstance(catalog_ids, frozenset)
        else frozenset(catalog_ids)
    )
    candidates = tuple(result.candidates)
    result_prefix = tuple(result.prefix)
    category_route = _route_tuple(result.category_route, catalog)
    core_route = _route_tuple(result.positive_core_route, catalog)
    category_novel = tuple(result.category_novel)
    core_novel = tuple(result.positive_core_novel)
    category_filtered = tuple(result.category_filtered)
    core_filtered = tuple(result.positive_core_filtered)
    tail = tuple(result.tail)
    queries = result.queries
    category_query = getattr(queries, "category", None)
    core_query = getattr(queries, "positive_core", None)
    category_active = getattr(category_query, "activated", None)
    core_active = getattr(core_query, "activated", None)
    if (
        result.enabled is not True
        or not isinstance(result.activated, bool)
        or not isinstance(category_active, bool)
        or not isinstance(core_active, bool)
        or result.activated != (category_active or core_active)
        or not isinstance(result.conflict_count, int)
        or isinstance(result.conflict_count, bool)
        or result.conflict_count < 0
        or not isinstance(result.tail_conflict_count, int)
        or isinstance(result.tail_conflict_count, bool)
        or result.tail_conflict_count < 0
        or result.legacy_route_executions != 0
        or isinstance(result.legacy_route_executions, bool)
        or result_prefix != prefix
    ):
        raise SparseUnionG0WorkerError("EXPANSION_CONTRACT")
    if (not category_active and category_route) or (not core_active and core_route):
        raise SparseUnionG0WorkerError("ROUTE_ACTIVATION_CONTRACT")
    seen = set(prefix)
    expected_category_novel = tuple(
        identifier for identifier in category_route if identifier not in seen
    )
    expected_core_novel = tuple(
        identifier for identifier in core_route if identifier not in seen
    )
    if (
        category_novel != expected_category_novel
        or core_novel != expected_core_novel
        or len(category_filtered) != len(set(category_filtered))
        or len(core_filtered) != len(set(core_filtered))
        or not _subsequence(category_filtered, category_novel)
        or not _subsequence(core_filtered, core_novel)
    ):
        raise SparseUnionG0WorkerError("NOVEL_OR_MASK_ORDER")
    expected_tail = _expected_fused_tail(prefix, category_filtered, core_filtered)
    if (
        result.conflict_count
        != (len(category_novel) - len(category_filtered))
        + (len(core_novel) - len(core_filtered))
        or tail != expected_tail
        or candidates != prefix + tail
        or len(candidates) > MAX_CANDIDATES
        or len(candidates) != len(set(candidates))
        or any(identifier not in catalog for identifier in candidates)
        or result.tail_conflict_count != 0
        or (not result.activated and (category_route or core_route or tail))
    ):
        raise SparseUnionG0WorkerError("PREFIX_MASK_OR_RRF_CONTRACT")
    return candidates


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
        raise SparseUnionG0WorkerError("TRACE_COORDINATE")
    values = tuple(candidates)
    if (
        not MIN_C200_CANDIDATES <= len(values) <= MAX_CANDIDATES
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise SparseUnionG0WorkerError("TRACE_CANDIDATES")
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
        raise SparseUnionG0WorkerError("SEMANTIC_AUDIT_ORDINAL")
    as_dict = getattr(result, "as_dict", None)
    if not callable(as_dict):
        raise SparseUnionG0WorkerError("SEMANTIC_AUDIT_SCHEMA")
    payload = as_dict()
    if not isinstance(payload, Mapping):
        raise SparseUnionG0WorkerError("SEMANTIC_AUDIT_SCHEMA")
    return _canonical_bytes(payload) + b"\n"


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise SparseUnionG0WorkerError("EMPTY_AGGREGATE")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _latency_summary(values: Sequence[float]) -> dict[str, int | float]:
    numbers = [float(value) for value in values]
    if not numbers or any(not math.isfinite(value) or value < 0.0 for value in numbers):
        raise SparseUnionG0WorkerError("LATENCY_AGGREGATE")
    return {
        "count": len(numbers),
        "maximum_milliseconds": round(max(numbers), 6),
        "p50_milliseconds": round(_nearest_rank(numbers, 0.50), 6),
        "p95_milliseconds": round(_nearest_rank(numbers, 0.95), 6),
    }


def _pool_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise SparseUnionG0WorkerError("POOL_AGGREGATE")
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
        raise SparseUnionG0WorkerError("OUTPUT_PATH_NOT_ALLOWLISTED")
    _require_real_ancestry(output.parent, "TRACE_PARENT_UNSAFE")
    if not output.parent.is_dir() or _is_link_or_reparse(output.parent):
        raise SparseUnionG0WorkerError("TRACE_PARENT_UNAVAILABLE")
    resolved_parent = output.parent.resolve(strict=True)
    expected_parent = Path(EXPECTED_RUNTIME_ROOT) / lexical.parts[-2]
    if not _same_path(resolved_parent, expected_parent):
        raise SparseUnionG0WorkerError("TRACE_PARENT_IDENTITY")
    partial = output.with_name(f".{output.name}.{nonce}.partial")
    for path in (output, partial):
        if path.exists() or path.is_symlink() or _is_link_or_reparse(path):
            raise SparseUnionG0WorkerError("TRACE_ALREADY_EXISTS")
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
            raise SparseUnionG0WorkerError("TRACE_SHORT_WRITE")
        view = view[written:]


def _publish_partial_exclusive(partial: Path, output: Path) -> None:
    _require_regular_file(partial, "PARTIAL_TRACE_UNAVAILABLE")
    if output.exists() or output.is_symlink() or _is_link_or_reparse(output):
        raise SparseUnionG0WorkerError("TRACE_ALREADY_EXISTS")
    try:
        os.link(partial, output, follow_symlinks=False)
    except FileExistsError as error:
        raise SparseUnionG0WorkerError("TRACE_ALREADY_EXISTS") from error
    except OSError as error:
        raise SparseUnionG0WorkerError("ATOMIC_TRACE_PUBLISH") from error
    if not os.path.samefile(partial, output):
        raise SparseUnionG0WorkerError("ATOMIC_TRACE_IDENTITY")


def _receipt_privacy_scan(
    value: object, *, catalog_ids: Iterable[str] = ()
) -> None:
    if _walk_keys(value) & FORBIDDEN_RECEIPT_KEYS:
        raise SparseUnionG0WorkerError("RECEIPT_FORBIDDEN_KEY")
    payload = _canonical_bytes(value).decode("utf-8")
    if ASIN_SHAPE_RE.search(payload):
        raise SparseUnionG0WorkerError("RECEIPT_IDENTIFIER")
    catalog = {str(identifier).casefold() for identifier in catalog_ids}
    tokens = {
        match.group(0).casefold()
        for match in CATALOG_IDENTIFIER_TOKEN_RE.finditer(payload)
    }
    if tokens & catalog:
        raise SparseUnionG0WorkerError("RECEIPT_IDENTIFIER")


def _validate_arguments(args: argparse.Namespace, progress: WorkerProgress) -> None:
    nonce = str(args.nonce)
    if NONCE_RE.fullmatch(nonce) is None:
        raise SparseUnionG0WorkerError("NONCE_INVALID")
    if args.session_limit not in ALLOWED_SESSION_LIMITS:
        raise SparseUnionG0WorkerError("SESSION_LIMIT_INVALID")
    if not isinstance(getattr(args, "semantic_audit", False), bool) or not isinstance(
        getattr(args, "semantic_cache", False), bool
    ):
        raise SparseUnionG0WorkerError("SEMANTIC_MODE_INVALID")
    semantic_audit = bool(getattr(args, "semantic_audit", False))
    semantic_cache = bool(getattr(args, "semantic_cache", False))
    if semantic_cache and not semantic_audit:
        raise SparseUnionG0WorkerError("SEMANTIC_AUDIT_REQUIRED")
    expected_worker = getattr(args, "expected_worker_blob", None)
    expected_union = getattr(args, "expected_union_blob", None)
    if semantic_audit:
        if (
            not isinstance(expected_worker, str)
            or GIT_BLOB_RE.fullmatch(expected_worker) is None
            or not isinstance(expected_union, str)
            or GIT_BLOB_RE.fullmatch(expected_union) is None
        ):
            raise SparseUnionG0WorkerError("EXPECTED_SOURCE_BLOB_INVALID")
    elif expected_worker is not None or expected_union is not None:
        raise SparseUnionG0WorkerError("EXPECTED_SOURCE_BLOB_UNSCOPED")
    progress.nonce = nonce
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
        raise SparseUnionG0WorkerError("INPUT_PATH_NOT_ALLOWLISTED")
    output_parts = _lexical_path_key(output)
    root_parts = _lexical_path_key(EXPECTED_RUNTIME_ROOT)
    if (
        len(output_parts) != len(root_parts) + 2
        or output_parts[: len(root_parts)] != root_parts
        or output.suffix.casefold() != ".jsonl"
    ):
        raise SparseUnionG0WorkerError("OUTPUT_PATH_NOT_ALLOWLISTED")


def _validate_end_identity(
    path: Path, expected: SourceIdentity, c200_contract: Any
) -> None:
    observed = c200_contract._raw_jsonl_identity(path, "sealed source")
    if observed.report() != expected.report() or observed.snapshot != expected.snapshot:
        raise SparseUnionG0WorkerError("SOURCE_CHANGED")


def _cache_contract(value: object, *, after_close: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SparseUnionG0WorkerError("CACHE_DIAGNOSTICS_SCHEMA")
    expected_top = {
        "enabled",
        "closed",
        "clears",
        "fts_route",
        "product_view",
        "mask_decision",
    }
    if set(value) != expected_top or value.get("enabled") is not True:
        raise SparseUnionG0WorkerError("CACHE_DIAGNOSTICS_SCHEMA")
    if value.get("closed") is not after_close:
        raise SparseUnionG0WorkerError("CACHE_CLOSE_STATE")
    expected_clears = 1 if after_close else 0
    if value.get("clears") != expected_clears or isinstance(
        value.get("clears"), bool
    ):
        raise SparseUnionG0WorkerError("CACHE_CLEAR_STATE")
    expected_layer = {
        "lookups",
        "hits",
        "misses",
        "evictions",
        "size",
        "capacity",
        "avoided_operations",
    }
    for name in ("fts_route", "product_view", "mask_decision"):
        layer = value.get(name)
        if not isinstance(layer, Mapping) or set(layer) != expected_layer:
            raise SparseUnionG0WorkerError("CACHE_DIAGNOSTICS_SCHEMA")
        if any(
            not isinstance(number, int) or isinstance(number, bool) or number < 0
            for number in layer.values()
        ):
            raise SparseUnionG0WorkerError("CACHE_DIAGNOSTICS_SCHEMA")
        if (
            layer["capacity"] != CACHE_CAPACITIES[name]
            or layer["size"] > layer["capacity"]
            or layer["lookups"] != layer["hits"] + layer["misses"]
            or layer["avoided_operations"] != layer["hits"]
            or (after_close and layer["size"] != 0)
        ):
            raise SparseUnionG0WorkerError("CACHE_DIAGNOSTICS_CONTRACT")
        if not after_close and (layer["lookups"] <= 0 or layer["hits"] <= 0):
            raise SparseUnionG0WorkerError("CACHE_LAYER_NO_HIT")
    return {str(key): value[key] for key in sorted(value)}


def _route_contract(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "category_route_executions",
        "positive_core_route_executions",
        "legacy_route_executions",
        "registry_sha256",
        "closed",
    }:
        raise SparseUnionG0WorkerError("ROUTE_DIAGNOSTICS_SCHEMA")
    if (
        not isinstance(value["category_route_executions"], int)
        or isinstance(value["category_route_executions"], bool)
        or value["category_route_executions"] < 0
        or not isinstance(value["positive_core_route_executions"], int)
        or isinstance(value["positive_core_route_executions"], bool)
        or value["positive_core_route_executions"] < 0
        or value["legacy_route_executions"] != 0
        or value["registry_sha256"] != EXPECTED_ATTRIBUTE_REGISTRY_SHA256
        or not isinstance(value["closed"], bool)
    ):
        raise SparseUnionG0WorkerError("ROUTE_DIAGNOSTICS_CONTRACT")
    return dict(value)


def run(
    args: argparse.Namespace,
    *,
    network_audit: OfflineNetworkAudit | None = None,
    runtime_loader: Callable[[], tuple[Any, Any, Callable[[], str]]] = (
        _load_runtime_after_audit
    ),
    expander_factory: Callable[..., Any] | None = None,
    progress: WorkerProgress | None = None,
) -> dict[str, Any]:
    """Run one fresh target-free dual-view trace worker."""

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
    c200_contract, sparse_union_g0, registry_hash = runtime_loader()
    _verify_imported_module_origins(c200_contract, sparse_union_g0, registry_hash)
    module_validate = getattr(sparse_union_g0, "validate", None)
    if not callable(module_validate):
        raise SparseUnionG0WorkerError("CORE_VALIDATE_API")
    module_validate()
    presealed_sources = _tracked_source_identities()
    _validate_semantic_source_blobs(args)
    if not _same_identities(source_start, presealed_sources):
        raise SparseUnionG0WorkerError("TRACKED_SOURCE_CHANGED_BEFORE_SEALED_ACCESS")

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
        raise SparseUnionG0WorkerError("SEALED_SOURCE_SCHEMA") from error
    reference_identity, reference_cells = _validate_reference_identity(
        args.c200_reference, catalog_ids, c200_contract
    )
    if catalog_identity.report() != {
        "bytes": EXPECTED_CATALOG_BYTES,
        "rows": EXPECTED_CATALOG_ROWS,
        "sha256": EXPECTED_CATALOG_SHA256,
    }:
        raise SparseUnionG0WorkerError("CATALOG_IDENTITY")
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
        raise SparseUnionG0WorkerError("SEALED_SOURCE_IDENTITY")
    state.input_identities = {
        "catalog": catalog_identity.report(),
        "sealed_c200_reference": reference_identity.report(),
        "visible_context": context_identity.report(),
    }

    pools: dict[str, list[int]] = {
        "expanded_union": [],
        "sealed_c200": [],
        "category_route": [],
        "positive_core_route": [],
        "category_filtered": [],
        "positive_core_filtered": [],
        "tail": [],
    }
    context_parse_milliseconds: list[float] = []
    extra_milliseconds: list[float] = []
    turn_milliseconds: list[float] = []
    category_activation_count = 0
    core_activation_count = 0
    dual_route_activation_count = 0
    dual_support_candidate_cells = 0
    union_novel_candidate_cells = 0
    union_expansion_count = 0
    conflict_count = 0
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
    route_before_close: Mapping[str, Any] | None = None
    route_after_close: Mapping[str, Any] | None = None
    processed_turns = 0

    state.phase = "EXPANDER_INITIALIZATION"
    factory = expander_factory or sparse_union_g0.SparseUnionG0Expander
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
                        raise SparseUnionG0WorkerError("CONTEXT_ENDED_EARLY")
                    context_parse_started = time.perf_counter_ns()
                    try:
                        contexts = c200_contract._parse_context_container(
                            context_line, catalog_ids
                        )
                    except BaseException as error:
                        raise SparseUnionG0WorkerError("CONTEXT_SCHEMA") from error
                    context_elapsed = (
                        time.perf_counter_ns() - context_parse_started
                    ) / 1_000_000.0
                    context_parse_milliseconds.append(context_elapsed)
                    context_share = context_elapsed / len(contexts)
                    for turn, context in enumerate(contexts, start=1):
                        turn_started = time.perf_counter_ns()
                        reference_line = reference_handle.readline()
                        if not reference_line:
                            raise SparseUnionG0WorkerError("C200_ENDED_EARLY")
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
                        extra_elapsed = (
                            time.perf_counter_ns() - extra_started
                        ) / 1_000_000.0
                        candidates = validate_expansion_result(
                            result, sealed_c200, catalog_ids
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
                        pools["category_route"].append(len(result.category_route))
                        pools["positive_core_route"].append(
                            len(result.positive_core_route)
                        )
                        pools["category_filtered"].append(
                            len(result.category_filtered)
                        )
                        pools["positive_core_filtered"].append(
                            len(result.positive_core_filtered)
                        )
                        pools["tail"].append(len(result.tail))
                        extra_milliseconds.append(extra_elapsed)
                        turn_milliseconds.append(
                            (time.perf_counter_ns() - turn_started) / 1_000_000.0
                            + context_share
                        )
                        category_active = bool(result.queries.category.activated)
                        core_active = bool(result.queries.positive_core.activated)
                        category_activation_count += int(category_active)
                        core_activation_count += int(core_active)
                        dual_route_activation_count += int(
                            category_active and core_active
                        )
                        dual_support_candidate_cells += int(
                            result.dual_support_count
                        )
                        union_novel_candidate_cells += int(
                            result.union_novel_count
                        )
                        union_expansion_count += int(bool(result.tail))
                        conflict_count += int(result.conflict_count)
                        tail_conflict_count += int(result.tail_conflict_count)
                        legacy_route_executions += int(result.legacy_route_executions)
                        evaluated_unique_novel_candidate_cells += len(
                            set(result.category_novel)
                            | set(result.positive_core_novel)
                        )
                    state.last_completed_session = ordinal
                if args.session_limit == SESSION_COUNT:
                    if context_handle.read(1) or reference_handle.read(1):
                        raise SparseUnionG0WorkerError("SEALED_SOURCE_EXCESS_ROWS")
            os.fsync(descriptor)
            route_before_close = expander.route_diagnostics()
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
                route_after_close = route_diagnostics()
            if semantic_cache:
                cache_diagnostics = getattr(expander, "cache_diagnostics", None)
                if callable(cache_diagnostics):
                    cache_after_close = cache_diagnostics()

    if not state.sqlite_closed:
        raise SparseUnionG0WorkerError("SQLITE_NOT_CLOSED")
    expected_records = args.session_limit * TURN_COUNT
    if (
        state.last_completed_session != args.session_limit
        or processed_turns != expected_records
        or state.partial_rows != expected_records
    ):
        raise SparseUnionG0WorkerError("TRAJECTORY_INCOMPLETE")
    route_before = _route_contract(route_before_close)
    route_after = _route_contract(route_after_close)
    execution_counts_valid = (
        0 < route_before["category_route_executions"]
        <= category_activation_count
        and 0 < route_before["positive_core_route_executions"]
        <= core_activation_count
    )
    if not semantic_cache:
        execution_counts_valid = execution_counts_valid and (
            route_before["category_route_executions"] == category_activation_count
            and route_before["positive_core_route_executions"]
            == core_activation_count
        )
    if (
        route_before["closed"] is not False
        or route_after["closed"] is not True
        or not execution_counts_valid
        or route_before["legacy_route_executions"] != 0
        or route_after["category_route_executions"]
        != route_before["category_route_executions"]
        or route_after["positive_core_route_executions"]
        != route_before["positive_core_route_executions"]
        or legacy_route_executions != 0
    ):
        raise SparseUnionG0WorkerError("ROUTE_EXECUTION_AUDIT")

    state.phase = "SOURCE_REVALIDATION"
    _validate_end_identity(args.catalog, catalog_identity, c200_contract)
    _validate_end_identity(args.context, context_identity, c200_contract)
    _validate_end_identity(args.c200_reference, reference_identity, c200_contract)
    source_end = _tracked_source_identities()
    _validate_semantic_source_blobs(args)
    _assert_legacy_runtime_absent()
    if not _same_identities(source_start, source_end):
        raise SparseUnionG0WorkerError("TRACKED_SOURCE_CHANGED")

    state.phase = "RESOURCE_VALIDATION"
    if audit.attempt_count:
        raise SparseUnionG0WorkerError("NETWORK_ATTEMPT")
    if any(name in sys.modules for name in ("cupy", "tensorflow", "torch")):
        raise SparseUnionG0WorkerError("GPU_RUNTIME_PRESENT")
    extra_latency = _latency_summary(extra_milliseconds)
    context_latency = _latency_summary(context_parse_milliseconds)
    turn_latency = _latency_summary(turn_milliseconds)
    pool_summaries = {name: _pool_summary(values) for name, values in pools.items()}
    peak_rss, rss_backend = _peak_rss_bytes()
    wall_seconds = time.perf_counter() - state.wall_started
    candidate_ratio = (
        pool_summaries["expanded_union"]["candidate_cells"] / reference_prefix_cells
    )
    trace_ratio = state.partial_bytes / reference_prefix_bytes
    if (
        category_activation_count <= 0
        or core_activation_count <= 0
        or union_expansion_count <= 0
        or pool_summaries["tail"]["candidate_cells"] <= 0
        or tail_conflict_count != 0
        or legacy_route_executions != 0
        or float(extra_latency["p95_milliseconds"])
        > MAX_EXTRA_P95_MILLISECONDS
        or float(turn_latency["p95_milliseconds"]) > MAX_TURN_P95_MILLISECONDS
        or peak_rss is None
        or not 0 < peak_rss <= MAX_WORKING_SET_BYTES
        or wall_seconds > MAX_WALL_SECONDS
        or candidate_ratio > MAX_CANDIDATE_CELL_RATIO
        or trace_ratio > MAX_TRACE_BYTE_RATIO
    ):
        raise SparseUnionG0WorkerError("RESOURCE_OR_ACTIVATION_GATE")

    environment["network_attempt_count"] = audit.attempt_count
    summary: dict[str, Any] = {
        "activation": {
            "category_route_activation_records": category_activation_count,
            "positive_core_route_activation_records": core_activation_count,
            "dual_route_activation_records": dual_route_activation_count,
            "dual_support_candidate_cells": dual_support_candidate_cells,
            "union_expansion_records": union_expansion_count,
            "union_novel_candidate_cells": union_novel_candidate_cells,
            "legacy_route_executions": legacy_route_executions,
        },
        "configuration": {
            "category_route_limit": ROUTE_LIMIT,
            "positive_core_route_limit": ROUTE_LIMIT,
            "default_off": True,
            "diagnostic_only": True,
            "exact_fraction_rrf": True,
            "served_top10_unchanged": True,
            "stable_append_after_complete_variable_c200": True,
        },
        "environment": environment,
        "input_identities": state.input_identities,
        "latency": {
            "context_container_parse": context_latency,
            "extra_routes_fusion_and_mask": extra_latency,
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
        "processed_sessions": args.session_limit,
        "processed_turns": expected_records,
        "resources": {
            "candidate_cell_ratio_over_c200": round(float(candidate_ratio), 6),
            "gpu_peak_bytes": 0,
            "network_attempt_count": audit.attempt_count,
            "peak_working_set_backend": rss_backend,
            "peak_working_set_bytes": peak_rss,
            "trace_byte_ratio_over_c200": round(float(trace_ratio), 6),
            "wall_seconds": round(wall_seconds, 6),
        },
        "route_diagnostics": route_before,
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
            raise SparseUnionG0WorkerError("CACHE_DIAGNOSTICS_INCOMPLETE")
        summary["cache"] = {
            "before_close": _cache_contract(cache_before_close, after_close=False),
            "after_close": _cache_contract(cache_after_close, after_close=True),
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
        raise SparseUnionG0WorkerError("PUBLISHED_TRACE_IDENTITY")
    return receipt


def _error_code(error: BaseException, progress: WorkerProgress) -> str:
    if isinstance(error, SparseUnionG0WorkerError):
        return error.error_code
    if isinstance(error, ImportError):
        return "RUNTIME_IMPORT"
    if isinstance(error, FileExistsError):
        return "TRACE_ALREADY_EXISTS"
    if isinstance(error, MemoryError):
        return "MEMORY_EXHAUSTED"
    if isinstance(error, PermissionError) and progress.network_audit is not None:
        if progress.network_audit.attempt_count:
            return "NETWORK_ATTEMPT"
    if isinstance(error, PermissionError):
        return "IO_PERMISSION"
    if isinstance(error, OSError):
        return "IO_ERROR"
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "INTERRUPTED"
    return "UNEXPECTED_EXCEPTION"


def _sanitized_traceback(error: BaseException) -> dict[str, Any]:
    frames = traceback.extract_tb(error.__traceback__) if error.__traceback__ else []
    sanitized = [
        {
            "file": Path(frame.filename).name,
            "function": frame.name,
            "line": int(frame.lineno),
        }
        for frame in frames[-8:]
    ]
    top_frame = sanitized[-1] if sanitized else {
        "file": "unknown",
        "function": "unknown",
        "line": 0,
    }
    surface = {"exception_type": type(error).__name__, "frames": sanitized}
    return {
        "exception_type": type(error).__name__,
        "sha256": hashlib.sha256(_canonical_bytes(surface)).hexdigest(),
        "top_frame": top_frame,
    }


def _partial_identity(progress: WorkerProgress) -> dict[str, int | str]:
    path = progress.partial_path
    if path is None or not path.is_file() or _is_link_or_reparse(path):
        return {"bytes": 0, "rows": 0, "sha256": EMPTY_SHA256}
    try:
        identity = _raw_identity(path, rows=True)
    except BaseException:
        return {
            "bytes": progress.partial_bytes,
            "rows": progress.partial_rows,
            "sha256": progress.partial_sha256,
        }
    return identity.report()


def _error_receipt(error: BaseException, progress: WorkerProgress) -> dict[str, Any]:
    peak_rss, rss_backend = _peak_rss_bytes()
    receipt: dict[str, Any] = {
        "error_code": _error_code(error, progress),
        "kind": "receipt",
        "last_completed_session": progress.last_completed_session,
        "partial_trace": _partial_identity(progress),
        "phase": progress.phase,
        "resources": {
            "gpu_peak_bytes": 0,
            "network_attempt_count": (
                progress.network_audit.attempt_count
                if progress.network_audit is not None
                else 0
            ),
            "peak_working_set_backend": rss_backend,
            "peak_working_set_bytes": peak_rss,
            "wall_seconds": round(time.perf_counter() - progress.wall_started, 6),
        },
        "schema_version": SCHEMA_VERSION,
        "source_identities": progress.source_identities,
        "status": "ERROR",
        "traceback": _sanitized_traceback(error),
    }
    if progress.nonce is not None:
        receipt["nonce"] = progress.nonce
    _receipt_privacy_scan(receipt)
    return receipt


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise SparseUnionG0WorkerError("ARGUMENT_INVALID")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
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
        "--require-module", default="starter.sparse_union_g0", choices=("starter.sparse_union_g0",)
    )
    arguments = parser.parse_args(argv)
    _assert_legacy_runtime_absent()
    required = importlib.import_module(arguments.require_module)
    contract = importlib.import_module("scripts.c200_candidate_worker")
    evaluator = importlib.import_module("evaluator.local_evaluator")
    _verify_imported_module_origins(
        contract,
        required,
        getattr(required, "attribute_registry_sha256", None),
    )
    evaluator_origin = getattr(evaluator, "__file__", None)
    if (
        getattr(evaluator, "__name__", None) != "evaluator.local_evaluator"
        or not isinstance(evaluator_origin, str)
        or not _same_path(
            Path(evaluator_origin), PROJECT_ROOT / "evaluator" / "local_evaluator.py"
        )
    ):
        raise SparseUnionG0WorkerError("EVALUATOR_MODULE_IDENTITY")
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


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        self_check = _entrypoint_self_check(raw_arguments)
        if self_check is not None:
            return self_check
    except BaseException:
        return 2
    progress = WorkerProgress()
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
