"""Isolated target-free worker for frozen sparse multiview trace generation.

The worker consumes only the sealed catalog, visible-context cache, and one
sealed variable-length C200 reference.  It calls the diagnostic sparse
expander directly; no production response path or evaluator is executed.
Candidate-bearing rows are streamed to a nonce-scoped partial file.  The
SQLite owner is closed and all source/resource gates pass before the trace is
published with exclusive-create semantics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
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


SCHEMA_VERSION = "small-ranker-registry-ca-g0-worker-summary.v1"
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
    "3cad1aba4f92d1107dd5179c24861d370d8e321b"
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
EXPECTED_TRACE_ROOT = PureWindowsPath(PROJECT_ROOT.as_posix()) / "experiments" / "fast_track"
V219_RESULT_DENIED_PATH = (
    EXPECTED_TRACE_ROOT / "small_ranker_v2_19_registry_ca_g0_20260831.json"
)
V219_CACHE_DENIED_ROOT = (
    EXPECTED_TRACE_ROOT / "small_ranker_v2_19_registry_ca_g0_cache_20260831"
)
PINNED_DEPENDENCY_BLOBS = {
    "evaluator/local_evaluator.py": "7c808347b31ef3121a9cbc4810ac3eb325f950ba",
    "scripts/c200_candidate_worker.py": "b94fddcf5a9b20ddde540f3f43ea9962982cb096",
    "scripts/probe_c200_candidate_recall.py": "0a57f63866683b476b9f49184673cf3154531911",
    "scripts/probe_e0_embedding_candidate_recall.py": "5bb9ec7f38f90d814d0c121c9f8992267d3491d5",
    "starter/agent.py": "421c6d43c598102b8fefb181b72bab5da4bf1294",
    "starter/architecture_lab.py": "8d340d0dce3fc2f1bb987a5dd632444776a05667",
    "starter/attributes.py": "92260323f077c9861aa4edd5242aff772c875760",
    "starter/p8_negative.py": "719078234dba297ce59f68d8a2b1734ec53c9c63",
    "starter/slot_ledger.py": "72975cff12af59e4044e52911c58294cd74a785a",
}

MAX_EXTRA_P95_MILLISECONDS = 100.0
MAX_TURN_P95_MILLISECONDS = 400.0
MAX_WORKING_SET_BYTES = 1_610_612_736
MAX_WALL_SECONDS = 1_800.0
MAX_CANDIDATE_CELL_RATIO = 2.0
MAX_TRACE_BYTE_RATIO = 2.1

NONCE_RE = re.compile(r"[0-9a-f]{32}")
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
        "eligible_from",
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
        "turn",
    }
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class SparseMultiviewWorkerError(RuntimeError):
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
        raise SparseMultiviewWorkerError("NON_CANONICAL_JSON") from error


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SparseMultiviewWorkerError("DUPLICATE_JSON_KEY")
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
            raise SparseMultiviewWorkerError(error_code)


def _require_regular_file(path: Path, error_code: str) -> None:
    _require_real_ancestry(path, error_code)
    if not path.is_file():
        raise SparseMultiviewWorkerError(error_code)


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


def _lexical_windows_path(path: Path) -> PureWindowsPath:
    raw = os.fspath(path)
    raw_parts = raw.replace("/", "\\").split("\\")
    if not raw or any(part in {".", ".."} for part in raw_parts):
        raise SparseMultiviewWorkerError("PATH_NOT_LEXICALLY_CANONICAL")
    result = PureWindowsPath(raw)
    if not result.is_absolute():
        raise SparseMultiviewWorkerError("PATH_NOT_ABSOLUTE")
    return result


def _lexical_path_key(path: PureWindowsPath) -> tuple[str, ...]:
    return tuple(part.casefold() for part in path.parts)


def _is_v219_trace_path_denied(path: Path | PureWindowsPath) -> bool:
    lexical = path if isinstance(path, PureWindowsPath) else _lexical_windows_path(path)
    key = _lexical_path_key(lexical)
    result_key = _lexical_path_key(V219_RESULT_DENIED_PATH)
    cache_key = _lexical_path_key(V219_CACHE_DENIED_ROOT)
    return key == result_key or key == cache_key or key[: len(cache_key)] == cache_key


def _guard_v219_trace_namespace(path: Path | PureWindowsPath) -> None:
    if _is_v219_trace_path_denied(path):
        raise SparseMultiviewWorkerError("V219_NAMESPACE_DENIED")


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
        raise SparseMultiviewWorkerError("SOURCE_CHANGED_DURING_READ")
    return SourceIdentity(byte_count, row_count, digest.hexdigest(), after)


def _raw_git_blob_sha1(path: Path) -> str:
    _require_regular_file(path, "PINNED_SOURCE_UNAVAILABLE")
    before = _snapshot(path)
    with path.open("rb") as handle:
        working_tree_bytes = handle.read()
    after = _snapshot(path)
    if before != after or len(working_tree_bytes) != before[0]:
        raise SparseMultiviewWorkerError("PINNED_SOURCE_CHANGED_DURING_READ")
    blob_bytes = working_tree_bytes.replace(b"\r\n", b"\n")
    digest = hashlib.sha1()
    digest.update(f"blob {len(blob_bytes)}\0".encode("ascii"))
    digest.update(blob_bytes)
    return digest.hexdigest()


def _tracked_source_identities() -> dict[str, SourceIdentity]:
    paths = {
        "preregistration": PROJECT_ROOT
        / "configs"
        / "small_ranker_v2_20.sparse_route_cache_preregistration.json",
        "scripts/sparse_multiview_candidate_worker.py": Path(__file__).resolve(),
        "starter/sparse_multiview.py": PROJECT_ROOT / "starter" / "sparse_multiview.py",
        **{
            relative: PROJECT_ROOT / Path(relative)
            for relative in PINNED_DEPENDENCY_BLOBS
        },
    }
    identities = {name: _raw_identity(path) for name, path in paths.items()}
    if (
        _raw_git_blob_sha1(paths["preregistration"])
        != EXPECTED_PREREGISTRATION_BLOB_SHA1
    ):
        raise SparseMultiviewWorkerError("PREREGISTRATION_IDENTITY")
    if any(
        _raw_git_blob_sha1(paths[relative]) != expected
        for relative, expected in PINNED_DEPENDENCY_BLOBS.items()
    ):
        raise SparseMultiviewWorkerError("PINNED_GIT_BLOB_IDENTITY")
    return identities


def _source_identity_reports(
    identities: Mapping[str, SourceIdentity],
) -> dict[str, dict[str, int | str]]:
    reports: dict[str, dict[str, int | str]] = {}
    for name, identity in identities.items():
        report = identity.report()
        if name in PINNED_DEPENDENCY_BLOBS:
            report["raw_git_blob_sha1"] = PINNED_DEPENDENCY_BLOBS[name]
        reports[name] = report
    return reports


def _validate_semantic_source_blobs(args: argparse.Namespace) -> None:
    """Bind the v2.20 worker/core bytes to the parent-verified Git tree."""

    if not bool(getattr(args, "semantic_audit", False)):
        return
    expected_worker = getattr(args, "expected_worker_blob", None)
    expected_sparse = getattr(args, "expected_sparse_blob", None)
    if (
        not isinstance(expected_worker, str)
        or GIT_BLOB_RE.fullmatch(expected_worker) is None
        or not isinstance(expected_sparse, str)
        or GIT_BLOB_RE.fullmatch(expected_sparse) is None
        or _raw_git_blob_sha1(Path(__file__).resolve()) != expected_worker
        or _raw_git_blob_sha1(PROJECT_ROOT / "starter" / "sparse_multiview.py")
        != expected_sparse
    ):
        raise SparseMultiviewWorkerError("EXPECTED_SOURCE_BLOB_MISMATCH")


def _same_identities(
    before: Mapping[str, SourceIdentity], after: Mapping[str, SourceIdentity]
) -> bool:
    return all(
        name in after
        and identity.report() == after[name].report()
        and identity.snapshot == after[name].snapshot
        for name, identity in before.items()
    ) and set(before) == set(after)


def _load_runtime_after_audit() -> tuple[Any, Any, Callable[[], str]]:
    try:
        c200_contract = importlib.import_module("scripts.c200_candidate_worker")
        sparse_multiview = importlib.import_module("starter.sparse_multiview")
        attributes = importlib.import_module("starter.attributes")
    except ImportError as error:
        raise SparseMultiviewWorkerError("RUNTIME_IMPORT") from error
    registry_hash = getattr(attributes, "attribute_registry_sha256", None)
    if not callable(registry_hash):
        raise SparseMultiviewWorkerError("REGISTRY_API")
    return c200_contract, sparse_multiview, registry_hash


def _verify_imported_module_origins(
    c200_contract: Any,
    sparse_multiview: Any,
    registry_hash: Callable[[], str],
) -> None:
    if (
        getattr(c200_contract, "__name__", None) != "scripts.c200_candidate_worker"
        or getattr(sparse_multiview, "__name__", None) != "starter.sparse_multiview"
        or getattr(registry_hash, "__module__", None) != "starter.attributes"
    ):
        raise SparseMultiviewWorkerError("RUNTIME_MODULE_IDENTITY")
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "starter"
            or name.startswith("starter.")
            or name == "scripts"
            or name.startswith("scripts.")
        ):
            continue
        origin = getattr(module, "__file__", None)
        spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
        relative = Path(*name.split("."))
        expected = PROJECT_ROOT / (
            relative / "__init__.py"
            if hasattr(module, "__path__")
            else relative.with_suffix(".py")
        )
        if (
            not isinstance(origin, str)
            or not isinstance(spec_origin, str)
            or not _same_path(Path(origin), expected)
            or not _same_path(Path(spec_origin), expected)
        ):
            raise SparseMultiviewWorkerError("RUNTIME_MODULE_ORIGIN")


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
        raise SparseMultiviewWorkerError("RUNTIME_ENVIRONMENT")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    loaded_gpu_modules = sorted(
        name
        for name in ("cupy", "tensorflow", "torch")
        if name in sys.modules
    )
    if loaded_gpu_modules:
        raise SparseMultiviewWorkerError("GPU_RUNTIME_PRESENT")
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
        raise SparseMultiviewWorkerError("C200_SCHEMA")
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
        raise SparseMultiviewWorkerError("C200_SCHEMA")
    return values


def parse_c200_reference_line(
    line: bytes,
    *,
    ordinal: int,
    turn: int,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    """Parse one exact canonical sealed reference row."""

    try:
        value = json.loads(
            line.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                SparseMultiviewWorkerError("C200_NONFINITE")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SparseMultiviewWorkerError("C200_JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"c200", "ordinal", "turn"}
        or value.get("ordinal") != ordinal
        or isinstance(value.get("ordinal"), bool)
        or value.get("turn") != turn
        or isinstance(value.get("turn"), bool)
    ):
        raise SparseMultiviewWorkerError("C200_ORDER")
    candidates = _validate_c200_values(value.get("c200"), catalog_ids)
    expected = _canonical_bytes(
        {"c200": list(candidates), "ordinal": ordinal, "turn": turn}
    ) + b"\n"
    if line != expected:
        raise SparseMultiviewWorkerError("C200_CANONICAL")
    return candidates


def _validate_reference_identity(
    path: Path, catalog_ids: frozenset[str], c200_contract: Any
) -> tuple[SourceIdentity, int]:
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
                raise SparseMultiviewWorkerError("C200_ROW_COUNT")
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
        raise SparseMultiviewWorkerError("C200_REFERENCE_IDENTITY")
    return identity, candidate_cells


def validate_expansion_result(
    result: object,
    sealed_c200: object,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    """Validate prefix preservation and the single-route append contract."""

    prefix = _validate_c200_values(sealed_c200, catalog_ids)
    required = (
        "activated",
        "candidates",
        "conflict_count",
        "novel_route",
        "prefix",
        "route",
        "tail",
        "tail_conflict_count",
    )
    if any(not hasattr(result, name) for name in required):
        raise SparseMultiviewWorkerError("EXPANSION_SCHEMA")
    candidates = tuple(result.candidates)
    result_prefix = tuple(result.prefix)
    route = tuple(result.route)
    novel_route = tuple(result.novel_route)
    tail = tuple(result.tail)
    activated = result.activated
    conflict_count = result.conflict_count
    tail_conflict_count = result.tail_conflict_count
    catalog = (
        catalog_ids
        if isinstance(catalog_ids, (set, frozenset))
        else frozenset(catalog_ids)
    )
    if (
        not isinstance(activated, bool)
        or not isinstance(conflict_count, int)
        or isinstance(conflict_count, bool)
        or conflict_count < 0
        or not isinstance(tail_conflict_count, int)
        or isinstance(tail_conflict_count, bool)
        or not 0 <= tail_conflict_count <= len(tail)
        or result_prefix != prefix
        or len(route) > ROUTE_LIMIT
        or len(route) != len(set(route))
        or any(not isinstance(value, str) or value not in catalog for value in route)
    ):
        raise SparseMultiviewWorkerError("EXPANSION_CONTRACT")
    seen = set(prefix)
    expected_novel: list[str] = []
    for identifier in route:
        if identifier not in seen:
            expected_novel.append(identifier)
            seen.add(identifier)
    if novel_route != tuple(expected_novel):
        raise SparseMultiviewWorkerError("NOVEL_ROUTE_ORDER")
    tail_iterator = iter(novel_route)
    if any(not any(value == candidate for candidate in tail_iterator) for value in tail):
        raise SparseMultiviewWorkerError("MASK_ORDER")
    if (
        conflict_count != len(novel_route) - len(tail)
        or candidates != prefix + tail
        or not len(prefix) <= len(candidates) <= min(MAX_CANDIDATES, len(prefix) + ROUTE_LIMIT)
        or len(candidates) != len(set(candidates))
        or any(value not in catalog for value in candidates)
        or (
            not activated
            and (
                route
                or novel_route
                or tail
                or conflict_count
                or tail_conflict_count
            )
        )
    ):
        raise SparseMultiviewWorkerError("PREFIX_OR_MASK_CONTRACT")
    return candidates


def canonical_trace_line(ordinal: int, turn: int, candidates: object) -> bytes:
    """Encode the sole candidate-bearing output schema as canonical LF JSON."""

    if (
        not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or not 1 <= ordinal <= SESSION_COUNT
        or not isinstance(turn, int)
        or isinstance(turn, bool)
        or not 1 <= turn <= TURN_COUNT
        or not isinstance(candidates, (list, tuple))
    ):
        raise SparseMultiviewWorkerError("TRACE_COORDINATE")
    values = tuple(candidates)
    if (
        not MIN_C200_CANDIDATES <= len(values) <= MAX_CANDIDATES
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise SparseMultiviewWorkerError("TRACE_CANDIDATES")
    return _canonical_bytes(
        {"candidates": list(values), "ordinal": ordinal, "turn": turn}
    ) + b"\n"


def canonical_semantic_line(ordinal: int, turn: int, result: object) -> bytes:
    """Hash-only audit encoding for a complete expansion result."""

    if (
        not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or not 1 <= ordinal <= SESSION_COUNT
        or not isinstance(turn, int)
        or isinstance(turn, bool)
        or not 1 <= turn <= TURN_COUNT
    ):
        raise SparseMultiviewWorkerError("SEMANTIC_AUDIT_ORDINAL")
    as_dict = getattr(result, "as_dict", None)
    if not callable(as_dict):
        raise SparseMultiviewWorkerError("SEMANTIC_AUDIT_SCHEMA")
    payload = as_dict()
    if not isinstance(payload, Mapping):
        raise SparseMultiviewWorkerError("SEMANTIC_AUDIT_SCHEMA")
    # Ordinal/turn validate the caller's frozen traversal; their order is encoded
    # by the sequence of LF-delimited canonical ExpansionResult payloads.
    return _canonical_bytes(payload) + b"\n"


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise SparseMultiviewWorkerError("EMPTY_AGGREGATE")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _latency_summary(values: Sequence[float]) -> dict[str, int | float]:
    numbers = [float(value) for value in values]
    if not numbers or any(not math.isfinite(value) or value < 0.0 for value in numbers):
        raise SparseMultiviewWorkerError("LATENCY_AGGREGATE")
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
        raise SparseMultiviewWorkerError("POOL_AGGREGATE")
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
    _guard_v219_trace_namespace(output)
    _require_real_ancestry(output.parent, "TRACE_PARENT_UNSAFE")
    if not output.parent.is_dir() or _is_link_or_reparse(output.parent):
        raise SparseMultiviewWorkerError("TRACE_PARENT_UNAVAILABLE")
    partial = output.with_name(f".{output.name}.{nonce}.partial")
    for path in (output, partial):
        if path.exists() or path.is_symlink() or _is_link_or_reparse(path):
            raise SparseMultiviewWorkerError("TRACE_ALREADY_EXISTS")
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
            raise SparseMultiviewWorkerError("TRACE_SHORT_WRITE")
        view = view[written:]


def _publish_partial_exclusive(partial: Path, output: Path) -> None:
    """Atomically publish a closed, fsynced partial as an exclusive hard link."""

    _require_regular_file(partial, "PARTIAL_TRACE_UNAVAILABLE")
    if output.exists() or output.is_symlink() or _is_link_or_reparse(output):
        raise SparseMultiviewWorkerError("TRACE_ALREADY_EXISTS")
    try:
        os.link(partial, output, follow_symlinks=False)
    except FileExistsError as error:
        raise SparseMultiviewWorkerError("TRACE_ALREADY_EXISTS") from error
    except OSError as error:
        raise SparseMultiviewWorkerError("ATOMIC_TRACE_PUBLISH") from error
    if not os.path.samefile(partial, output):
        raise SparseMultiviewWorkerError("ATOMIC_TRACE_IDENTITY")
    if os.name != "nt":
        directory = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _receipt_privacy_scan(
    value: object, *, catalog_ids: Iterable[str] = ()
) -> None:
    if _walk_keys(value) & FORBIDDEN_RECEIPT_KEYS:
        raise SparseMultiviewWorkerError("RECEIPT_FORBIDDEN_KEY")
    payload = _canonical_bytes(value).decode("utf-8")
    if ASIN_SHAPE_RE.search(payload):
        raise SparseMultiviewWorkerError("RECEIPT_IDENTIFIER")
    catalog = {str(identifier).casefold() for identifier in catalog_ids}
    tokens = {
        match.group(0).casefold()
        for match in CATALOG_IDENTIFIER_TOKEN_RE.finditer(payload)
    }
    if tokens & catalog:
        raise SparseMultiviewWorkerError("RECEIPT_IDENTIFIER")


def _validate_arguments(args: argparse.Namespace, progress: WorkerProgress) -> None:
    nonce = str(args.nonce)
    if NONCE_RE.fullmatch(nonce) is None:
        raise SparseMultiviewWorkerError("NONCE_INVALID")
    if args.session_limit not in ALLOWED_SESSION_LIMITS:
        raise SparseMultiviewWorkerError("SESSION_LIMIT_INVALID")
    if not isinstance(getattr(args, "semantic_audit", False), bool) or not isinstance(
        getattr(args, "semantic_cache", False), bool
    ):
        raise SparseMultiviewWorkerError("SEMANTIC_MODE_INVALID")
    if bool(getattr(args, "semantic_cache", False)) and not bool(
        getattr(args, "semantic_audit", False)
    ):
        raise SparseMultiviewWorkerError("SEMANTIC_AUDIT_REQUIRED")
    expected_worker_blob = getattr(args, "expected_worker_blob", None)
    expected_sparse_blob = getattr(args, "expected_sparse_blob", None)
    semantic_audit = bool(getattr(args, "semantic_audit", False))
    if semantic_audit:
        if (
            not isinstance(expected_worker_blob, str)
            or GIT_BLOB_RE.fullmatch(expected_worker_blob) is None
            or not isinstance(expected_sparse_blob, str)
            or GIT_BLOB_RE.fullmatch(expected_sparse_blob) is None
        ):
            raise SparseMultiviewWorkerError("EXPECTED_SOURCE_BLOB_INVALID")
    elif expected_worker_blob is not None or expected_sparse_blob is not None:
        raise SparseMultiviewWorkerError("EXPECTED_SOURCE_BLOB_UNSCOPED")
    progress.nonce = nonce
    catalog = _lexical_windows_path(args.catalog)
    context = _lexical_windows_path(args.context)
    reference = _lexical_windows_path(args.c200_reference)
    output = _lexical_windows_path(args.trace_output)
    _guard_v219_trace_namespace(output)
    if (
        _lexical_path_key(catalog) != _lexical_path_key(EXPECTED_CATALOG_PATH)
        or _lexical_path_key(context) != _lexical_path_key(EXPECTED_CONTEXT_PATH)
        or _lexical_path_key(reference)
        not in {
            _lexical_path_key(path) for path in EXPECTED_C200_REFERENCE_PATHS
        }
    ):
        raise SparseMultiviewWorkerError("INPUT_PATH_NOT_ALLOWLISTED")
    output_parts = _lexical_path_key(output)
    root_parts = _lexical_path_key(EXPECTED_TRACE_ROOT)
    if (
        len(output_parts) <= len(root_parts)
        or output_parts[: len(root_parts)] != root_parts
        or output.suffix.casefold() != ".jsonl"
    ):
        raise SparseMultiviewWorkerError("OUTPUT_PATH_NOT_ALLOWLISTED")


def _validate_end_identity(
    path: Path, expected: SourceIdentity, c200_contract: Any
) -> None:
    observed = c200_contract._raw_jsonl_identity(path, "sealed source")
    if observed.report() != expected.report() or observed.snapshot != expected.snapshot:
        raise SparseMultiviewWorkerError("SOURCE_CHANGED")


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
    """Run one fresh target-free sparse trace worker."""

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
    c200_contract, sparse_multiview, registry_hash = runtime_loader()
    _verify_imported_module_origins(
        c200_contract,
        sparse_multiview,
        registry_hash,
    )
    presealed_sources = _tracked_source_identities()
    _validate_semantic_source_blobs(args)
    if not _same_identities(source_start, presealed_sources):
        raise SparseMultiviewWorkerError("TRACKED_SOURCE_CHANGED_BEFORE_SEALED_ACCESS")

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
        raise SparseMultiviewWorkerError("SEALED_SOURCE_SCHEMA") from error
    reference_identity, reference_cells = _validate_reference_identity(
        args.c200_reference, catalog_ids, c200_contract
    )
    if catalog_identity.report() != {
        "bytes": EXPECTED_CATALOG_BYTES,
        "rows": EXPECTED_CATALOG_ROWS,
        "sha256": EXPECTED_CATALOG_SHA256,
    }:
        raise SparseMultiviewWorkerError("CATALOG_IDENTITY")
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
        raise SparseMultiviewWorkerError("SEALED_SOURCE_IDENTITY")
    state.input_identities = {
        "catalog": catalog_identity.report(),
        "sealed_c200_reference": reference_identity.report(),
        "visible_context": context_identity.report(),
    }

    candidate_lengths: list[int] = []
    c200_lengths: list[int] = []
    route_lengths: list[int] = []
    tail_lengths: list[int] = []
    context_parse_milliseconds: list[float] = []
    extra_milliseconds: list[float] = []
    turn_milliseconds: list[float] = []
    novel_route_lengths: list[int] = []
    activation_count = 0
    conflict_count = 0
    tail_conflict_count = 0
    reference_prefix_bytes = 0
    reference_prefix_cells = 0
    trace_digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    semantic_audit = bool(getattr(args, "semantic_audit", False))
    semantic_cache = bool(getattr(args, "semantic_cache", False))
    cache_before_close: Mapping[str, Any] | None = None
    cache_after_close: Mapping[str, Any] | None = None
    processed_turns = 0

    state.phase = "EXPANDER_INITIALIZATION"
    factory = expander_factory or sparse_multiview.SparseMultiviewExpander
    expander = (
        factory(args.catalog, enabled=True, cache_enabled=True)
        if semantic_cache
        else factory(args.catalog, enabled=True)
    )
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
                        raise SparseMultiviewWorkerError("CONTEXT_ENDED_EARLY")
                    context_parse_started = time.perf_counter_ns()
                    try:
                        contexts = c200_contract._parse_context_container(
                            context_line, catalog_ids
                        )
                    except BaseException as error:
                        raise SparseMultiviewWorkerError("CONTEXT_SCHEMA") from error
                    context_parse_elapsed = (
                        time.perf_counter_ns() - context_parse_started
                    ) / 1_000_000.0
                    context_parse_milliseconds.append(context_parse_elapsed)
                    context_parse_share = context_parse_elapsed / len(contexts)
                    for turn, context in enumerate(contexts, start=1):
                        turn_started = time.perf_counter_ns()
                        reference_line = reference_handle.readline()
                        if not reference_line:
                            raise SparseMultiviewWorkerError("C200_ENDED_EARLY")
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
                        candidate_lengths.append(len(candidates))
                        c200_lengths.append(len(sealed_c200))
                        route_lengths.append(len(result.route))
                        novel_route_lengths.append(len(result.novel_route))
                        tail_lengths.append(len(result.tail))
                        extra_milliseconds.append(extra_elapsed)
                        turn_milliseconds.append(
                            (time.perf_counter_ns() - turn_started) / 1_000_000.0
                            + context_parse_share
                        )
                        activation_count += int(result.activated)
                        conflict_count += int(result.conflict_count)
                        tail_conflict_count += int(result.tail_conflict_count)
                    state.last_completed_session = ordinal
                if args.session_limit == SESSION_COUNT:
                    if context_handle.read(1) or reference_handle.read(1):
                        raise SparseMultiviewWorkerError("SEALED_SOURCE_EXCESS_ROWS")
            os.fsync(descriptor)
            if semantic_cache:
                diagnostics = getattr(expander, "cache_diagnostics", None)
                if not callable(diagnostics):
                    raise SparseMultiviewWorkerError("CACHE_DIAGNOSTICS_UNAVAILABLE")
                cache_before_close = diagnostics()
        finally:
            os.close(descriptor)
    finally:
        if sys.exc_info()[0] is None:
            state.phase = "SQLITE_CLOSE"
        try:
            expander.close()
        finally:
            state.sqlite_closed = bool(getattr(expander, "closed", False))
            if semantic_cache:
                diagnostics = getattr(expander, "cache_diagnostics", None)
                if callable(diagnostics):
                    cache_after_close = diagnostics()

    if not state.sqlite_closed:
        raise SparseMultiviewWorkerError("SQLITE_NOT_CLOSED")
    expected_records = args.session_limit * TURN_COUNT
    if (
        state.last_completed_session != args.session_limit
        or processed_turns != expected_records
        or state.partial_rows != expected_records
    ):
        raise SparseMultiviewWorkerError("TRAJECTORY_INCOMPLETE")

    state.phase = "SOURCE_REVALIDATION"
    _validate_end_identity(args.catalog, catalog_identity, c200_contract)
    _validate_end_identity(args.context, context_identity, c200_contract)
    _validate_end_identity(args.c200_reference, reference_identity, c200_contract)
    source_end = _tracked_source_identities()
    _validate_semantic_source_blobs(args)
    if not _same_identities(source_start, source_end):
        raise SparseMultiviewWorkerError("TRACKED_SOURCE_CHANGED")

    state.phase = "RESOURCE_VALIDATION"
    if audit.attempt_count:
        raise SparseMultiviewWorkerError("NETWORK_ATTEMPT")
    if any(name in sys.modules for name in ("cupy", "tensorflow", "torch")):
        raise SparseMultiviewWorkerError("GPU_RUNTIME_PRESENT")
    extra_latency = _latency_summary(extra_milliseconds)
    context_parse_latency = _latency_summary(context_parse_milliseconds)
    turn_latency = _latency_summary(turn_milliseconds)
    candidate_pool = _pool_summary(candidate_lengths)
    c200_pool = _pool_summary(c200_lengths)
    route_pool = _pool_summary(route_lengths)
    tail_pool = _pool_summary(tail_lengths)
    peak_rss, rss_backend = _peak_rss_bytes()
    wall_seconds = time.perf_counter() - state.wall_started
    candidate_ratio = candidate_pool["candidate_cells"] / reference_prefix_cells
    trace_ratio = state.partial_bytes / reference_prefix_bytes
    if (
        activation_count <= 0
        or tail_conflict_count != 0
        or float(extra_latency["p95_milliseconds"])
        > MAX_EXTRA_P95_MILLISECONDS
        or float(turn_latency["p95_milliseconds"]) > MAX_TURN_P95_MILLISECONDS
        or peak_rss is None
        or not 0 < peak_rss <= MAX_WORKING_SET_BYTES
        or wall_seconds > MAX_WALL_SECONDS
        or candidate_ratio > MAX_CANDIDATE_CELL_RATIO
        or trace_ratio > MAX_TRACE_BYTE_RATIO
    ):
        raise SparseMultiviewWorkerError("RESOURCE_OR_ACTIVATION_GATE")

    environment["network_attempt_count"] = audit.attempt_count
    summary: dict[str, Any] = {
        "activation": {
            "activated_records": activation_count,
            "inactive_records": expected_records - activation_count,
        },
        "configuration": {
            "diagnostic_only": True,
            "route_limit": ROUTE_LIMIT,
            "served_top10_unchanged": True,
            "stable_append_after_complete_variable_c200": True,
        },
        "environment": environment,
        "input_identities": state.input_identities,
        "latency": {
            "context_container_parse": context_parse_latency,
            "extra_route_and_mask": extra_latency,
            "per_turn": turn_latency,
        },
        "lifecycle": {
            "atomic_exclusive_trace_publish": True,
            "catalog_unchanged_before_trace_publish": True,
            "c200_reference_unchanged_before_trace_publish": True,
            "context_unchanged_before_trace_publish": True,
            "partial_fsynced_and_closed_before_trace_publish": True,
            "source_files_unchanged_before_trace_publish": True,
            "sqlite_closed_before_trace_publish": state.sqlite_closed,
            "runtime_module_origins_verified": True,
        },
        "mask": {
            "evaluated_novel_candidates": sum(novel_route_lengths),
            "removed_explicit_conflicts": conflict_count,
            "tail_duplicate_count": 0,
            "tail_explicit_conflict_count": tail_conflict_count,
        },
        "pool_lengths": {
            "expanded_union": candidate_pool,
            "route": route_pool,
            "sealed_c200": c200_pool,
            "tail": tail_pool,
        },
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
        "session_limit": args.session_limit,
        "source_identities": state.source_identities,
    }
    if semantic_audit:
        summary["semantic_trace"] = {
            "rows": expected_records,
            "sha256": semantic_digest.hexdigest(),
        }
    if semantic_cache:
        if not isinstance(cache_before_close, Mapping) or not isinstance(
            cache_after_close, Mapping
        ):
            raise SparseMultiviewWorkerError("CACHE_DIAGNOSTICS_INCOMPLETE")
        summary["cache"] = {
            "after_close": dict(cache_after_close),
            "before_close": dict(cache_before_close),
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
        raise SparseMultiviewWorkerError("PUBLISHED_TRACE_IDENTITY")
    return receipt


def _error_code(error: BaseException, progress: WorkerProgress) -> str:
    if isinstance(error, SparseMultiviewWorkerError):
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
    digest_surface = {
        "exception_type": type(error).__name__,
        "frames": sanitized,
    }
    return {
        "exception_type": type(error).__name__,
        "sha256": hashlib.sha256(_canonical_bytes(digest_surface)).hexdigest(),
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
        raise SparseMultiviewWorkerError("ARGUMENT_INVALID")


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
    parser.add_argument("--expected-sparse-blob")
    return parser


def _entrypoint_self_check(argv: Sequence[str]) -> int | None:
    if "--entrypoint-self-check" not in argv:
        return None
    parser = _SafeArgumentParser(add_help=False)
    parser.add_argument("--entrypoint-self-check", action="store_true")
    parser.add_argument("--require-module", default="evaluator.local_evaluator")
    arguments = parser.parse_args(argv)
    importlib.import_module(arguments.require_module)
    payload = {
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
