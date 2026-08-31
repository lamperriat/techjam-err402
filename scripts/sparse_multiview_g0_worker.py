"""Target-free worker for the frozen v2.22B four-view sparse RRF G0 probe.

This module is deliberately a diagnostic worker, not a serving entrypoint.  It
streams one complete candidate trace while preserving every sealed C200 row as
an ordered prefix.  Runtime imports are limited to the frozen C200 contract and
``starter.sparse_multiview_g0``; the legacy sparse-multiview implementation is never
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


SCHEMA_VERSION = "small-ranker-v2.22b-multiview-sparse-rrf-g0-worker-summary.v1"
CONTEXT_SCHEMA_VERSION = "small-ranker-visible-context.v1"
SESSION_COUNT = 2_000
TURN_COUNT = 10
RECORD_COUNT = SESSION_COUNT * TURN_COUNT
ROUTE_LIMIT = 120
MAX_CANDIDATES = 400
MIN_C200_CANDIDATES = 100
MAX_C200_CANDIDATES = 200
ALLOWED_SESSION_LIMITS = (20, 100, SESSION_COUNT)
ROUTE_ORDER = (
    "full_positive",
    "exact_active",
    "category_only",
    "title_store_exact",
)
ROUTE_FIELD_TRIPLES = (
    (
        "full_positive",
        "full_positive_route",
        "full_positive_novel",
        "full_positive_filtered",
    ),
    (
        "exact_active",
        "exact_active_route",
        "exact_active_novel",
        "exact_active_filtered",
    ),
    (
        "category_only",
        "category_only_route",
        "category_only_novel",
        "category_only_filtered",
    ),
    (
        "title_store_exact",
        "title_store_exact_route",
        "title_store_exact_novel",
        "title_store_exact_filtered",
    ),
)

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
    "e534bc7a9a304a03869e951f290fa2b96d51dee7"
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
EXPECTED_RUNTIME_ROOT = PureWindowsPath(r"D:\tiktok\.v222b_runtime")
PREREGISTRATION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "small_ranker_v2_22.multiview_sparse_rrf_g0_preregistration.json"
)
CORE_PATH = PROJECT_ROOT / "starter" / "sparse_multiview_g0.py"
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
        "starter.sparse_union_g0",
        "starter.sparse_multiview",
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


class SparseMultiviewG0WorkerError(RuntimeError):
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
        raise SparseMultiviewG0WorkerError("NON_CANONICAL_JSON") from error


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SparseMultiviewG0WorkerError("DUPLICATE_JSON_KEY")
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
            raise SparseMultiviewG0WorkerError(error_code)


def _require_regular_file(path: Path, error_code: str) -> None:
    _require_real_ancestry(path, error_code)
    if not path.is_file():
        raise SparseMultiviewG0WorkerError(error_code)


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
        raise SparseMultiviewG0WorkerError("PATH_NOT_LEXICALLY_CANONICAL")
    result = PureWindowsPath(raw)
    if not result.is_absolute():
        raise SparseMultiviewG0WorkerError("PATH_NOT_ABSOLUTE")
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
        raise SparseMultiviewG0WorkerError("LEGACY_NAMESPACE_DENIED")


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
        raise SparseMultiviewG0WorkerError("SOURCE_CHANGED_DURING_READ")
    return SourceIdentity(byte_count, row_count, digest.hexdigest(), after)


def _raw_git_blob_sha1(path: Path) -> str:
    _require_regular_file(path, "PINNED_SOURCE_UNAVAILABLE")
    before = _snapshot(path)
    with path.open("rb") as handle:
        working_tree_bytes = handle.read()
    after = _snapshot(path)
    if before != after or len(working_tree_bytes) != before[0]:
        raise SparseMultiviewG0WorkerError("PINNED_SOURCE_CHANGED_DURING_READ")
    blob_bytes = working_tree_bytes.replace(b"\r\n", b"\n")
    digest = hashlib.sha1()
    digest.update(f"blob {len(blob_bytes)}\0".encode("ascii"))
    digest.update(blob_bytes)
    return digest.hexdigest()


def _tracked_source_identities() -> dict[str, SourceIdentity]:
    paths = {
        "preregistration": PREREGISTRATION_PATH,
        "scripts/c200_candidate_worker.py": C200_CONTRACT_PATH,
        "scripts/sparse_multiview_g0_worker.py": Path(__file__).resolve(),
        "starter/sparse_multiview_g0.py": CORE_PATH,
    }
    identities = {name: _raw_identity(path) for name, path in paths.items()}
    if (
        _raw_git_blob_sha1(PREREGISTRATION_PATH)
        != EXPECTED_PREREGISTRATION_BLOB_SHA1
        or _raw_git_blob_sha1(C200_CONTRACT_PATH)
        != EXPECTED_C200_CONTRACT_BLOB_SHA1
    ):
        raise SparseMultiviewG0WorkerError("PINNED_GIT_BLOB_IDENTITY")
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
        raise SparseMultiviewG0WorkerError("EXPECTED_SOURCE_BLOB_MISMATCH")


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
        raise SparseMultiviewG0WorkerError("LEGACY_RUNTIME_PRESENT")


def _load_runtime_after_audit() -> tuple[Any, Any, Callable[[], str]]:
    _assert_legacy_runtime_absent()
    try:
        c200_contract = importlib.import_module("scripts.c200_candidate_worker")
        sparse_multiview_g0 = importlib.import_module("starter.sparse_multiview_g0")
    except ImportError as error:
        raise SparseMultiviewG0WorkerError("RUNTIME_IMPORT") from error
    registry_hash = getattr(sparse_multiview_g0, "attribute_registry_sha256", None)
    if not callable(registry_hash):
        raise SparseMultiviewG0WorkerError("REGISTRY_API")
    _assert_legacy_runtime_absent()
    return c200_contract, sparse_multiview_g0, registry_hash


def _verify_imported_module_origins(
    c200_contract: Any,
    sparse_multiview_g0: Any,
    registry_hash: Callable[[], str],
) -> None:
    if (
        getattr(c200_contract, "__name__", None)
        != "scripts.c200_candidate_worker"
        or getattr(sparse_multiview_g0, "__name__", None) != "starter.sparse_multiview_g0"
        or getattr(registry_hash, "__module__", None)
        not in {"starter.sparse_multiview_g0", "starter.attributes"}
    ):
        raise SparseMultiviewG0WorkerError("RUNTIME_MODULE_IDENTITY")
    expected_origins = {
        "scripts.c200_candidate_worker": C200_CONTRACT_PATH,
        "starter.sparse_multiview_g0": CORE_PATH,
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
            raise SparseMultiviewG0WorkerError("RUNTIME_MODULE_ORIGIN")
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
        raise SparseMultiviewG0WorkerError("RUNTIME_ENVIRONMENT")
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
        raise SparseMultiviewG0WorkerError("GPU_RUNTIME_PRESENT")
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
        raise SparseMultiviewG0WorkerError("C200_SCHEMA")
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
        raise SparseMultiviewG0WorkerError("C200_SCHEMA")
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
                SparseMultiviewG0WorkerError("C200_NONFINITE")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SparseMultiviewG0WorkerError("C200_JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"c200", "ordinal", "turn"}
        or value.get("ordinal") != ordinal
        or isinstance(value.get("ordinal"), bool)
        or value.get("turn") != turn
        or isinstance(value.get("turn"), bool)
    ):
        raise SparseMultiviewG0WorkerError("C200_ORDER")
    candidates = _validate_c200_values(value.get("c200"), catalog_ids)
    expected = _canonical_bytes(
        {"c200": list(candidates), "ordinal": ordinal, "turn": turn}
    ) + b"\n"
    if line != expected:
        raise SparseMultiviewG0WorkerError("C200_CANONICAL")
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
                raise SparseMultiviewG0WorkerError("C200_ROW_COUNT")
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
        raise SparseMultiviewG0WorkerError("C200_REFERENCE_IDENTITY")
    return identity, candidate_cells


def _route_tuple(value: object, catalog: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SparseMultiviewG0WorkerError("EXPANSION_SCHEMA")
    route = tuple(value)
    if (
        len(route) > ROUTE_LIMIT
        or len(route) != len(set(route))
        or any(not isinstance(identifier, str) or identifier not in catalog for identifier in route)
    ):
        raise SparseMultiviewG0WorkerError("ROUTE_CONTRACT")
    return route


def _subsequence(values: Sequence[str], container: Sequence[str]) -> bool:
    iterator = iter(container)
    return all(any(value == candidate for candidate in iterator) for value in values)


def _ranked_route(
    value: object,
    *,
    route_ids: Sequence[str],
    filtered_ids: Sequence[str],
) -> tuple[tuple[str, int], ...]:
    """Bind every mask survivor to its original one-based FTS route rank."""

    if not isinstance(value, (list, tuple)):
        raise SparseMultiviewG0WorkerError("FILTERED_RANK_SCHEMA")
    ranked: list[tuple[str, int]] = []
    for raw in value:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise SparseMultiviewG0WorkerError("FILTERED_RANK_SCHEMA")
        identifier, rank = raw
        if (
            not isinstance(identifier, str)
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 1 <= rank <= len(route_ids)
            or route_ids[rank - 1] != identifier
        ):
            raise SparseMultiviewG0WorkerError("FILTERED_RANK_CONTRACT")
        ranked.append((identifier, rank))
    if (
        tuple(identifier for identifier, _rank in ranked) != tuple(filtered_ids)
        or any(
            left_rank >= right_rank
            for (_left_id, left_rank), (_right_id, right_rank) in zip(
                ranked, ranked[1:]
            )
        )
    ):
        raise SparseMultiviewG0WorkerError("FILTERED_RANK_CONTRACT")
    return tuple(ranked)


def _fusion_item_tuple(item: object) -> tuple[object, ...]:
    fields = (
        "identifier",
        "score",
        "supporting_route_count",
        "minimum_route_rank",
        "full_positive_rank",
        "exact_active_rank",
        "category_only_rank",
        "title_store_exact_rank",
    )
    if any(not hasattr(item, field) for field in fields):
        raise SparseMultiviewG0WorkerError("FUSION_ITEM_SCHEMA")
    values = tuple(getattr(item, field) for field in fields)
    if (
        not isinstance(values[0], str)
        or not isinstance(values[1], Fraction)
        or any(
            not isinstance(number, int) or isinstance(number, bool)
            for number in values[2:]
        )
    ):
        raise SparseMultiviewG0WorkerError("FUSION_ITEM_SCHEMA")
    return values


def _expected_fused_tail(
    prefix: Sequence[str],
    full_positive_ranked: Sequence[tuple[str, int]],
    exact_active_ranked: Sequence[tuple[str, int]],
    category_only_ranked: Sequence[tuple[str, int]],
    title_store_exact_ranked: Sequence[tuple[str, int]],
) -> tuple[tuple[str, ...], tuple[tuple[object, ...], ...]]:
    prefix_set = set(prefix)
    rank_maps = {
        "full_positive": dict(full_positive_ranked),
        "exact_active": dict(exact_active_ranked),
        "category_only": dict(category_only_ranked),
        "title_store_exact": dict(title_store_exact_ranked),
    }
    if any(prefix_set.intersection(ranks) for ranks in rank_maps.values()):
        raise SparseMultiviewG0WorkerError("FILTERED_PREFIX_LEAK")
    identifiers = set().union(*rank_maps.values())

    def item(identifier: str) -> tuple[object, ...]:
        full_positive = rank_maps["full_positive"].get(identifier)
        exact_active = rank_maps["exact_active"].get(identifier)
        category_only = rank_maps["category_only"].get(identifier)
        title_store_exact = rank_maps["title_store_exact"].get(identifier)
        ranks = (
            full_positive,
            exact_active,
            category_only,
            title_store_exact,
        )
        score = sum(
            (
                Fraction(1, 60 + rank)
                for rank in ranks
                if rank is not None
            ),
            Fraction(0, 1),
        )
        support = sum(rank is not None for rank in ranks)
        minimum = min(rank for rank in ranks if rank is not None)
        return (
            identifier,
            score,
            support,
            minimum,
            full_positive if full_positive is not None else ROUTE_LIMIT + 1,
            exact_active if exact_active is not None else ROUTE_LIMIT + 1,
            category_only if category_only is not None else ROUTE_LIMIT + 1,
            title_store_exact if title_store_exact is not None else ROUTE_LIMIT + 1,
        )

    def key(values: tuple[object, ...]) -> tuple[object, ...]:
        return (
            -values[1],
            -values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            values[0],
        )

    ordered_items = tuple(sorted((item(identifier) for identifier in identifiers), key=key))
    capacity = max(0, MAX_CANDIDATES - len(prefix))
    return (
        tuple(str(values[0]) for values in ordered_items[:capacity]),
        ordered_items,
    )


def validate_expansion_result(
    result: object,
    sealed_c200: object,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    """Validate all four isolated routes, mask order, exact RRF, and C200 prefix."""

    prefix = _validate_c200_values(sealed_c200, catalog_ids)
    required = {
        "activated",
        "candidates",
        "conflict_count",
        "enabled",
        "fusion_items",
        "hard_mask_latency_ns",
        "legacy_route_executions",
        "multiroute_support_count",
        "prefix",
        "queries",
        "route_latency_ns",
        "tail",
        "tail_conflict_count",
        "union_novel_count",
    }
    for _route_name, route_field, novel_field, filtered_field in ROUTE_FIELD_TRIPLES:
        required.update(
            {
                route_field,
                novel_field,
                filtered_field,
                f"{_route_name}_filtered_ranked",
            }
        )
    if any(not hasattr(result, name) for name in required):
        raise SparseMultiviewG0WorkerError("EXPANSION_SCHEMA")
    ordered_fields = {"candidates", "fusion_items", "prefix", "tail"}
    for _route_name, route_field, novel_field, filtered_field in ROUTE_FIELD_TRIPLES:
        ordered_fields.update(
            {
                route_field,
                novel_field,
                filtered_field,
                f"{_route_name}_filtered_ranked",
            }
        )
    if any(
        not isinstance(getattr(result, name), (list, tuple))
        for name in ordered_fields
    ):
        raise SparseMultiviewG0WorkerError("EXPANSION_SCHEMA")
    catalog = (
        catalog_ids
        if isinstance(catalog_ids, frozenset)
        else frozenset(catalog_ids)
    )
    candidates = tuple(result.candidates)
    result_prefix = tuple(result.prefix)
    tail = tuple(result.tail)
    queries = result.queries
    route_values: dict[
        str,
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[str, int], ...],
            bool,
        ],
    ] = {}
    for route_name, route_field, novel_field, filtered_field in ROUTE_FIELD_TRIPLES:
        route_ids = _route_tuple(getattr(result, route_field), catalog)
        route_novel = tuple(getattr(result, novel_field))
        route_filtered = tuple(getattr(result, filtered_field))
        route_ranked = _ranked_route(
            getattr(result, f"{route_name}_filtered_ranked"),
            route_ids=route_ids,
            filtered_ids=route_filtered,
        )
        route_values[route_name] = (
            route_ids,
            route_novel,
            route_filtered,
            route_ranked,
            getattr(getattr(queries, route_name, None), "activated", None),
        )
    activation_flags = [value[4] for value in route_values.values()]
    if (
        result.enabled is not True
        or not isinstance(result.activated, bool)
        or any(not isinstance(flag, bool) for flag in activation_flags)
        or result.activated != any(bool(flag) for flag in activation_flags)
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
        raise SparseMultiviewG0WorkerError("EXPANSION_CONTRACT")
    seen = set(prefix)
    conflict_total = 0
    expected_tail, expected_fusion_items = _expected_fused_tail(
        prefix,
        route_values["full_positive"][3],
        route_values["exact_active"][3],
        route_values["category_only"][3],
        route_values["title_store_exact"][3],
    )
    for route_name, (
        route_ids,
        route_novel,
        route_filtered,
        _route_ranked,
        activated,
    ) in route_values.items():
        if (not activated and route_ids) or len(route_filtered) != len(set(route_filtered)):
            raise SparseMultiviewG0WorkerError("ROUTE_ACTIVATION_CONTRACT")
        expected_novel = tuple(
            identifier for identifier in route_ids if identifier not in seen
        )
        if route_novel != expected_novel or not _subsequence(route_filtered, route_novel):
            raise SparseMultiviewG0WorkerError("NOVEL_OR_MASK_ORDER")
        conflict_total += len(route_novel) - len(route_filtered)
    actual_fusion_items = tuple(
        _fusion_item_tuple(item) for item in result.fusion_items
    )
    expected_multiroute = sum(
        int(values[2] >= 2) for values in expected_fusion_items
    )
    if (
        result.conflict_count != conflict_total
        or actual_fusion_items != expected_fusion_items
        or tail != expected_tail
        or candidates != prefix + tail
        or len(candidates) > MAX_CANDIDATES
        or len(candidates) != len(set(candidates))
        or any(identifier not in catalog for identifier in candidates)
        or not isinstance(result.multiroute_support_count, int)
        or isinstance(result.multiroute_support_count, bool)
        or result.multiroute_support_count != expected_multiroute
        or not isinstance(result.union_novel_count, int)
        or isinstance(result.union_novel_count, bool)
        or result.union_novel_count != len(expected_fusion_items)
        or result.tail_conflict_count != 0
        or (
            not result.activated
            and any(route_values[route][0] for route in ROUTE_ORDER)
        )
        or (not result.activated and tail)
    ):
        raise SparseMultiviewG0WorkerError("PREFIX_MASK_OR_RRF_CONTRACT")
    return candidates


def expansion_timing_contract(
    result: object,
    *,
    cache_enabled: bool,
) -> tuple[dict[str, int], int]:
    """Validate dynamic timing evidence without putting it in semantic hashes."""

    raw_routes = getattr(result, "route_latency_ns", None)
    if not isinstance(raw_routes, (list, tuple)) or len(raw_routes) != len(
        ROUTE_ORDER
    ):
        raise SparseMultiviewG0WorkerError("ROUTE_LATENCY_SCHEMA")
    timings: dict[str, int] = {}
    for expected_route, raw in zip(ROUTE_ORDER, raw_routes):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise SparseMultiviewG0WorkerError("ROUTE_LATENCY_SCHEMA")
        route, elapsed = raw
        if (
            route != expected_route
            or not isinstance(elapsed, int)
            or isinstance(elapsed, bool)
            or elapsed < 0
        ):
            raise SparseMultiviewG0WorkerError("ROUTE_LATENCY_CONTRACT")
        query = getattr(getattr(result, "queries", None), expected_route, None)
        activated = getattr(query, "activated", None)
        if (
            not isinstance(activated, bool)
            or (not activated and elapsed != 0)
            or (activated and not cache_enabled and elapsed <= 0)
        ):
            raise SparseMultiviewG0WorkerError("ROUTE_LATENCY_CONTRACT")
        timings[expected_route] = elapsed
    hard_mask = getattr(result, "hard_mask_latency_ns", None)
    if (
        not isinstance(hard_mask, int)
        or isinstance(hard_mask, bool)
        or hard_mask <= 0
    ):
        raise SparseMultiviewG0WorkerError("MASK_LATENCY_CONTRACT")
    return timings, hard_mask


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
        raise SparseMultiviewG0WorkerError("TRACE_COORDINATE")
    values = tuple(candidates)
    if (
        not MIN_C200_CANDIDATES <= len(values) <= MAX_CANDIDATES
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise SparseMultiviewG0WorkerError("TRACE_CANDIDATES")
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
        raise SparseMultiviewG0WorkerError("SEMANTIC_AUDIT_ORDINAL")
    as_dict = getattr(result, "as_dict", None)
    if not callable(as_dict):
        raise SparseMultiviewG0WorkerError("SEMANTIC_AUDIT_SCHEMA")
    payload = as_dict()
    if not isinstance(payload, Mapping):
        raise SparseMultiviewG0WorkerError("SEMANTIC_AUDIT_SCHEMA")
    return _canonical_bytes(
        {"ordinal": ordinal, "result": dict(payload), "turn": turn}
    ) + b"\n"


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise SparseMultiviewG0WorkerError("EMPTY_AGGREGATE")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _latency_summary(
    values: Sequence[int], *, allow_empty: bool = False
) -> dict[str, int | float]:
    numbers = list(values)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in numbers
    ):
        raise SparseMultiviewG0WorkerError("LATENCY_AGGREGATE")
    if not numbers:
        if not allow_empty:
            raise SparseMultiviewG0WorkerError("EMPTY_AGGREGATE")
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
        raise SparseMultiviewG0WorkerError("POOL_AGGREGATE")
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
        raise SparseMultiviewG0WorkerError("OUTPUT_PATH_NOT_ALLOWLISTED")
    _require_real_ancestry(output.parent, "TRACE_PARENT_UNSAFE")
    if not output.parent.is_dir() or _is_link_or_reparse(output.parent):
        raise SparseMultiviewG0WorkerError("TRACE_PARENT_UNAVAILABLE")
    resolved_parent = output.parent.resolve(strict=True)
    expected_parent = Path(EXPECTED_RUNTIME_ROOT) / lexical.parts[-2]
    if not _same_path(resolved_parent, expected_parent):
        raise SparseMultiviewG0WorkerError("TRACE_PARENT_IDENTITY")
    partial = output.with_name(f".{output.name}.{nonce}.partial")
    for path in (output, partial):
        if path.exists() or path.is_symlink() or _is_link_or_reparse(path):
            raise SparseMultiviewG0WorkerError("TRACE_ALREADY_EXISTS")
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
            raise SparseMultiviewG0WorkerError("TRACE_SHORT_WRITE")
        view = view[written:]


def _publish_partial_exclusive(partial: Path, output: Path) -> None:
    _require_regular_file(partial, "PARTIAL_TRACE_UNAVAILABLE")
    if output.exists() or output.is_symlink() or _is_link_or_reparse(output):
        raise SparseMultiviewG0WorkerError("TRACE_ALREADY_EXISTS")
    try:
        os.link(partial, output, follow_symlinks=False)
    except FileExistsError as error:
        raise SparseMultiviewG0WorkerError("TRACE_ALREADY_EXISTS") from error
    except OSError as error:
        raise SparseMultiviewG0WorkerError("ATOMIC_TRACE_PUBLISH") from error
    if not os.path.samefile(partial, output):
        raise SparseMultiviewG0WorkerError("ATOMIC_TRACE_IDENTITY")


def _receipt_privacy_scan(
    value: object, *, catalog_ids: Iterable[str] = ()
) -> None:
    if _walk_keys(value) & FORBIDDEN_RECEIPT_KEYS:
        raise SparseMultiviewG0WorkerError("RECEIPT_FORBIDDEN_KEY")
    payload = _canonical_bytes(value).decode("utf-8")
    if ASIN_SHAPE_RE.search(payload):
        raise SparseMultiviewG0WorkerError("RECEIPT_IDENTIFIER")
    catalog = {str(identifier).casefold() for identifier in catalog_ids}
    tokens = {
        match.group(0).casefold()
        for match in CATALOG_IDENTIFIER_TOKEN_RE.finditer(payload)
    }
    if tokens & catalog:
        raise SparseMultiviewG0WorkerError("RECEIPT_IDENTIFIER")


def _validate_arguments(args: argparse.Namespace, progress: WorkerProgress) -> None:
    nonce = str(args.nonce)
    if NONCE_RE.fullmatch(nonce) is None:
        raise SparseMultiviewG0WorkerError("NONCE_INVALID")
    if args.session_limit not in ALLOWED_SESSION_LIMITS:
        raise SparseMultiviewG0WorkerError("SESSION_LIMIT_INVALID")
    if not isinstance(getattr(args, "semantic_audit", False), bool) or not isinstance(
        getattr(args, "semantic_cache", False), bool
    ):
        raise SparseMultiviewG0WorkerError("SEMANTIC_MODE_INVALID")
    semantic_audit = bool(getattr(args, "semantic_audit", False))
    semantic_cache = bool(getattr(args, "semantic_cache", False))
    if semantic_cache and not semantic_audit:
        raise SparseMultiviewG0WorkerError("SEMANTIC_AUDIT_REQUIRED")
    expected_worker = getattr(args, "expected_worker_blob", None)
    expected_union = getattr(args, "expected_union_blob", None)
    if semantic_audit:
        if (
            not isinstance(expected_worker, str)
            or GIT_BLOB_RE.fullmatch(expected_worker) is None
            or not isinstance(expected_union, str)
            or GIT_BLOB_RE.fullmatch(expected_union) is None
        ):
            raise SparseMultiviewG0WorkerError("EXPECTED_SOURCE_BLOB_INVALID")
    elif expected_worker is not None or expected_union is not None:
        raise SparseMultiviewG0WorkerError("EXPECTED_SOURCE_BLOB_UNSCOPED")
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
        raise SparseMultiviewG0WorkerError("INPUT_PATH_NOT_ALLOWLISTED")
    output_parts = _lexical_path_key(output)
    root_parts = _lexical_path_key(EXPECTED_RUNTIME_ROOT)
    if (
        len(output_parts) != len(root_parts) + 2
        or output_parts[: len(root_parts)] != root_parts
        or output.suffix.casefold() != ".jsonl"
    ):
        raise SparseMultiviewG0WorkerError("OUTPUT_PATH_NOT_ALLOWLISTED")


def _validate_end_identity(
    path: Path, expected: SourceIdentity, c200_contract: Any
) -> None:
    observed = c200_contract._raw_jsonl_identity(path, "sealed source")
    if observed.report() != expected.report() or observed.snapshot != expected.snapshot:
        raise SparseMultiviewG0WorkerError("SOURCE_CHANGED")


def _cache_contract(value: object, *, after_close: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SparseMultiviewG0WorkerError("CACHE_DIAGNOSTICS_SCHEMA")
    expected_top = {
        "enabled",
        "closed",
        "clears",
        "fts_route",
        "product_view",
        "mask_decision",
    }
    if set(value) != expected_top or value.get("enabled") is not True:
        raise SparseMultiviewG0WorkerError("CACHE_DIAGNOSTICS_SCHEMA")
    if value.get("closed") is not after_close:
        raise SparseMultiviewG0WorkerError("CACHE_CLOSE_STATE")
    expected_clears = 1 if after_close else 0
    if value.get("clears") != expected_clears or isinstance(
        value.get("clears"), bool
    ):
        raise SparseMultiviewG0WorkerError("CACHE_CLEAR_STATE")
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
            raise SparseMultiviewG0WorkerError("CACHE_DIAGNOSTICS_SCHEMA")
        if any(
            not isinstance(number, int) or isinstance(number, bool) or number < 0
            for number in layer.values()
        ):
            raise SparseMultiviewG0WorkerError("CACHE_DIAGNOSTICS_SCHEMA")
        if (
            layer["capacity"] != CACHE_CAPACITIES[name]
            or layer["size"] > layer["capacity"]
            or layer["lookups"] != layer["hits"] + layer["misses"]
            or layer["avoided_operations"] != layer["hits"]
            or (after_close and layer["size"] != 0)
        ):
            raise SparseMultiviewG0WorkerError("CACHE_DIAGNOSTICS_CONTRACT")
    return {str(key): value[key] for key in sorted(value)}


def _cache_pair_contract(
    before_value: object, after_value: object
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _cache_contract(before_value, after_close=False)
    after = _cache_contract(after_value, after_close=True)
    stable_fields = (
        "lookups",
        "hits",
        "misses",
        "evictions",
        "capacity",
        "avoided_operations",
    )
    if any(
        before[layer][field] != after[layer][field]
        for layer in CACHE_CAPACITIES
        for field in stable_fields
    ):
        raise SparseMultiviewG0WorkerError("CACHE_CLOSE_COUNTER_DRIFT")
    return before, after


def _route_contract(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "full_positive_route_executions",
        "exact_active_route_executions",
        "category_only_route_executions",
        "title_store_exact_route_executions",
        "legacy_route_executions",
        "registry_sha256",
        "closed",
    }:
        raise SparseMultiviewG0WorkerError("ROUTE_DIAGNOSTICS_SCHEMA")
    if (
        any(
            not isinstance(value[f"{route}_route_executions"], int)
            or isinstance(value[f"{route}_route_executions"], bool)
            or value[f"{route}_route_executions"] < 0
            for route in ROUTE_ORDER
        )
        or value["legacy_route_executions"] != 0
        or value["registry_sha256"] != EXPECTED_ATTRIBUTE_REGISTRY_SHA256
        or not isinstance(value["closed"], bool)
    ):
        raise SparseMultiviewG0WorkerError("ROUTE_DIAGNOSTICS_CONTRACT")
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
    """Run one fresh target-free four-view trace worker."""

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
    c200_contract, sparse_multiview_g0, registry_hash = runtime_loader()
    _verify_imported_module_origins(c200_contract, sparse_multiview_g0, registry_hash)
    module_validate = getattr(sparse_multiview_g0, "validate", None)
    if not callable(module_validate):
        raise SparseMultiviewG0WorkerError("CORE_VALIDATE_API")
    module_validate()
    presealed_sources = _tracked_source_identities()
    _validate_semantic_source_blobs(args)
    if not _same_identities(source_start, presealed_sources):
        raise SparseMultiviewG0WorkerError("TRACKED_SOURCE_CHANGED_BEFORE_SEALED_ACCESS")

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
        raise SparseMultiviewG0WorkerError("SEALED_SOURCE_SCHEMA") from error
    reference_identity, reference_cells = _validate_reference_identity(
        args.c200_reference, catalog_ids, c200_contract
    )
    if catalog_identity.report() != {
        "bytes": EXPECTED_CATALOG_BYTES,
        "rows": EXPECTED_CATALOG_ROWS,
        "sha256": EXPECTED_CATALOG_SHA256,
    }:
        raise SparseMultiviewG0WorkerError("CATALOG_IDENTITY")
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
        raise SparseMultiviewG0WorkerError("SEALED_SOURCE_IDENTITY")
    state.input_identities = {
        "catalog": catalog_identity.report(),
        "sealed_c200_reference": reference_identity.report(),
        "visible_context": context_identity.report(),
    }

    pools: dict[str, list[int]] = {
        "expanded_union": [],
        "sealed_c200": [],
        "full_positive_route": [],
        "exact_active_route": [],
        "category_only_route": [],
        "title_store_exact_route": [],
        "full_positive_filtered": [],
        "exact_active_filtered": [],
        "category_only_filtered": [],
        "title_store_exact_filtered": [],
        "tail": [],
    }
    context_parse_nanoseconds: list[int] = []
    extra_nanoseconds: list[int] = []
    turn_nanoseconds: list[int] = []
    route_nanoseconds: dict[str, list[int]] = {
        route: [] for route in ROUTE_ORDER
    }
    hard_mask_nanoseconds: list[int] = []
    route_activation_counts = {route: 0 for route in ROUTE_ORDER}
    multiroute_support_candidate_cells = 0
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
    factory = expander_factory or sparse_multiview_g0.SparseMultiviewG0Expander
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
                        raise SparseMultiviewG0WorkerError("CONTEXT_ENDED_EARLY")
                    context_parse_started = time.perf_counter_ns()
                    try:
                        contexts = c200_contract._parse_context_container(
                            context_line, catalog_ids
                        )
                    except BaseException as error:
                        raise SparseMultiviewG0WorkerError("CONTEXT_SCHEMA") from error
                    context_elapsed = time.perf_counter_ns() - context_parse_started
                    context_parse_nanoseconds.append(context_elapsed)
                    context_share = context_elapsed // len(contexts)
                    for turn, context in enumerate(contexts, start=1):
                        turn_started = time.perf_counter_ns()
                        reference_line = reference_handle.readline()
                        if not reference_line:
                            raise SparseMultiviewG0WorkerError("C200_ENDED_EARLY")
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
                        route_elapsed, mask_elapsed = expansion_timing_contract(
                            result, cache_enabled=semantic_cache
                        )
                        if sum(route_elapsed.values()) + mask_elapsed > extra_elapsed:
                            raise SparseMultiviewG0WorkerError(
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
                        for (
                            route_name,
                            route_field,
                            _novel_field,
                            filtered_field,
                        ) in ROUTE_FIELD_TRIPLES:
                            pools[route_field].append(len(getattr(result, route_field)))
                            pools[filtered_field].append(len(getattr(result, filtered_field)))
                        pools["tail"].append(len(result.tail))
                        extra_nanoseconds.append(extra_elapsed)
                        turn_nanoseconds.append(
                            time.perf_counter_ns() - turn_started + context_share
                        )
                        hard_mask_nanoseconds.append(mask_elapsed)
                        for route_name, elapsed in route_elapsed.items():
                            if elapsed:
                                route_nanoseconds[route_name].append(elapsed)
                        for route_name in ROUTE_ORDER:
                            route_activation_counts[route_name] += int(
                                bool(getattr(getattr(result.queries, route_name), "activated"))
                            )
                        multiroute_support_candidate_cells += int(
                            result.multiroute_support_count
                        )
                        union_novel_candidate_cells += int(
                            result.union_novel_count
                        )
                        union_expansion_count += int(bool(result.tail))
                        conflict_count += int(result.conflict_count)
                        tail_conflict_count += int(result.tail_conflict_count)
                        legacy_route_executions += int(result.legacy_route_executions)
                        evaluated_unique_novel_candidate_cells += len(
                            set(result.full_positive_novel)
                            | set(result.exact_active_novel)
                            | set(result.category_only_novel)
                            | set(result.title_store_exact_novel)
                        )
                    state.last_completed_session = ordinal
                if args.session_limit == SESSION_COUNT:
                    if context_handle.read(1) or reference_handle.read(1):
                        raise SparseMultiviewG0WorkerError("SEALED_SOURCE_EXCESS_ROWS")
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
        raise SparseMultiviewG0WorkerError("SQLITE_NOT_CLOSED")
    expected_records = args.session_limit * TURN_COUNT
    if (
        state.last_completed_session != args.session_limit
        or processed_turns != expected_records
        or state.partial_rows != expected_records
    ):
        raise SparseMultiviewG0WorkerError("TRAJECTORY_INCOMPLETE")
    route_before = _route_contract(route_before_close)
    route_after = _route_contract(route_after_close)
    execution_counts_valid = all(
        0 <= route_before[f"{route}_route_executions"]
        <= route_activation_counts[route]
        and route_before[f"{route}_route_executions"]
        == len(route_nanoseconds[route])
        for route in ROUTE_ORDER
    )
    if not semantic_cache:
        execution_counts_valid = execution_counts_valid and all(
            route_before[f"{route}_route_executions"] == route_activation_counts[route]
            for route in ROUTE_ORDER
        )
    if (
        route_before["closed"] is not False
        or route_after["closed"] is not True
        or not execution_counts_valid
        or route_before["legacy_route_executions"] != 0
        or any(
            route_after[f"{route}_route_executions"]
            != route_before[f"{route}_route_executions"]
            for route in ROUTE_ORDER
        )
        or len(hard_mask_nanoseconds) != expected_records
        or legacy_route_executions != 0
    ):
        raise SparseMultiviewG0WorkerError("ROUTE_EXECUTION_AUDIT")

    state.phase = "SOURCE_REVALIDATION"
    _validate_end_identity(args.catalog, catalog_identity, c200_contract)
    _validate_end_identity(args.context, context_identity, c200_contract)
    _validate_end_identity(args.c200_reference, reference_identity, c200_contract)
    source_end = _tracked_source_identities()
    _validate_semantic_source_blobs(args)
    _assert_legacy_runtime_absent()
    if not _same_identities(source_start, source_end):
        raise SparseMultiviewG0WorkerError("TRACKED_SOURCE_CHANGED")

    state.phase = "RESOURCE_VALIDATION"
    if audit.attempt_count:
        raise SparseMultiviewG0WorkerError("NETWORK_ATTEMPT")
    if any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in GPU_RUNTIME_PREFIXES
    ):
        raise SparseMultiviewG0WorkerError("GPU_RUNTIME_PRESENT")
    route_latency = {
        route: _latency_summary(route_nanoseconds[route], allow_empty=True)
        for route in ROUTE_ORDER
    }
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
        or any(
            _p95_nanoseconds(values)
            > int(MAX_ROUTE_P95_MILLISECONDS * 1_000_000)
            for values in route_nanoseconds.values()
        )
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
        raise SparseMultiviewG0WorkerError("RESOURCE_GATE")

    environment["network_attempt_count"] = audit.attempt_count
    summary: dict[str, Any] = {
        "activation": {
            "full_positive_route_activation_records": route_activation_counts["full_positive"],
            "exact_active_route_activation_records": route_activation_counts["exact_active"],
            "category_only_route_activation_records": route_activation_counts["category_only"],
            "title_store_exact_route_activation_records": route_activation_counts[
                "title_store_exact"
            ],
            "multiroute_support_candidate_cells": multiroute_support_candidate_cells,
            "union_expansion_records": union_expansion_count,
            "union_novel_candidate_cells": union_novel_candidate_cells,
            "legacy_route_executions": legacy_route_executions,
        },
        "configuration": {
            "full_positive_route_limit": ROUTE_LIMIT,
            "exact_active_route_limit": ROUTE_LIMIT,
            "category_only_route_limit": ROUTE_LIMIT,
            "title_store_exact_route_limit": ROUTE_LIMIT,
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
            "fts_routes": route_latency,
            "hard_conflict_mask": mask_latency,
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
            raise SparseMultiviewG0WorkerError("CACHE_DIAGNOSTICS_INCOMPLETE")
        cache_before, cache_after = _cache_pair_contract(
            cache_before_close, cache_after_close
        )
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
        raise SparseMultiviewG0WorkerError("PUBLISHED_TRACE_IDENTITY")
    return receipt


def _error_code(error: BaseException, progress: WorkerProgress) -> str:
    if isinstance(error, SparseMultiviewG0WorkerError):
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
    wall_seconds = time.perf_counter() - progress.wall_started
    receipt: dict[str, Any] = {
        "error_code": _error_code(error, progress),
        "kind": "receipt",
        "last_completed_session": progress.last_completed_session,
        "partial_trace": _partial_identity(progress),
        "phase": progress.phase,
        "resources": {
            "device": "CPU",
            "provider": "SQLite FTS5 + CPython",
            "gpu_peak_bytes": 0,
            "gpu_used": False,
            "network_attempt_count": (
                progress.network_audit.attempt_count
                if progress.network_audit is not None
                else 0
            ),
            "peak_working_set_backend": rss_backend,
            "peak_working_set_bytes": peak_rss,
            "turns_per_second": round(
                progress.partial_rows / wall_seconds if wall_seconds > 0.0 else 0.0,
                6,
            ),
            "wall_seconds": round(wall_seconds, 6),
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
        raise SparseMultiviewG0WorkerError("ARGUMENT_INVALID")


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
        "--require-module",
        default="starter.sparse_multiview_g0",
        choices=("starter.sparse_multiview_g0",),
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
        raise SparseMultiviewG0WorkerError("EVALUATOR_MODULE_IDENTITY")
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
