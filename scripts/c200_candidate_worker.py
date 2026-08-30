"""Isolated target-blind worker for the frozen R08 C200 candidate probe.

The worker receives only the official catalog and a preregistered,
identifier-free visible-context cache.  It constructs one explicitly configured
control Agent, captures the coverage-ranked ``rankings["final"]`` at the
``_apply_p11`` entry boundary, and never receives targets, folds, eligibility,
sample identifiers, or future context as an Agent input.

Candidate-bearing rows stay private in memory until the Agent and its SQLite
connection are closed.  The worker then publishes exactly one exclusive JSONL
trace and emits one aggregate-only receipt on stdout.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.agent import Agent, SessionState  # noqa: E402


SCHEMA_VERSION = "small-ranker-c200-candidate-worker-summary.v1"
CONTEXT_SCHEMA_VERSION = "small-ranker-visible-context.v1"

SESSION_COUNT = 2_000
TURN_COUNT = 10
RECORD_COUNT = SESSION_COUNT * TURN_COUNT
TOP_K = 10
MIN_CANDIDATES = 100
MAX_CANDIDATES = 200

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
REDACTED_IDENTIFIER = "[identifier omitted]"

EXPECTED_EXECUTABLE = Path(r"D:\450\conda\envs\tiktok\python.exe")
EXPECTED_PYTHON = "3.11.16"
EXPECTED_SQLITE = "3.53.4"
REFERENCE_C100_CANONICAL_BYTES = 26_690_930
MAX_TRACE_BYTES = int(REFERENCE_C100_CANONICAL_BYTES * 2.1)
MAX_WORKING_SET_BYTES = 2_147_483_648

ASIN_SHAPE_RE = re.compile(
    r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE
)
CATALOG_IDENTIFIER_TOKEN_RE = re.compile(
    r"(?<![A-Z0-9])[A-Z0-9]{10}(?![A-Z0-9])", re.IGNORECASE
)
CATALOG_IDENTIFIER_RE = re.compile(r"[A-Z0-9]{10}")
NONCE_RE = re.compile(r"[0-9a-f]{32}")
CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "asin",
        "parent_asin",
        "sample_id",
        "scenario_type",
        "ground_truth",
        "target",
        "target_id",
        "target_asin",
        "eligible_from",
        "outer_fold",
        "family_index",
        "future_turns",
        "evaluator_metadata",
        "user_id",
        "ordinal",
    }
)
CONTEXT_TURN_KEYS = frozenset(
    {
        "message",
        "goal_messages",
        "category_text",
        "active_terms",
        "excluded_terms",
        "query_terms",
        "version",
        "version_anchor_turn",
        "override_count",
        "current_turn_override",
        "active_records",
        "retired_records",
        "hard_clause_terms",
        "budget_upper",
    }
)
SLOT_RECORD_KEYS = frozenset(
    {"slot", "value", "polarity", "hardness", "source_turn", "version", "status"}
)


class C200WorkerError(RuntimeError):
    """Raised when a frozen target-blind worker invariant is violated."""


@dataclass(frozen=True)
class SourceIdentity:
    bytes: int
    rows: int
    sha256: str
    snapshot: tuple[int, int, int]

    def report(self) -> dict[str, int | str]:
        return {"bytes": self.bytes, "rows": self.rows, "sha256": self.sha256}


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
        raise PermissionError("network activity is disabled in the C200 worker")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise C200WorkerError("JSON contains a duplicate object key")
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


def _walk_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            yield from _walk_strings(child)


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


def _snapshot(path: Path) -> tuple[int, int, int]:
    observed = path.stat()
    return (
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(getattr(observed, "st_ino", 0)),
    )


def _require_real_ancestry(path: Path, label: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if _is_link_or_reparse(component):
            raise C200WorkerError(f"{label} traverses a link or reparse point")


def _require_regular_input(path: Path, label: str) -> None:
    _require_real_ancestry(path, label)
    if not path.is_file():
        raise C200WorkerError(f"{label} is unavailable or unsafe")


def _raw_jsonl_identity(path: Path, label: str) -> SourceIdentity:
    _require_regular_input(path, label)
    before = _snapshot(path)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            byte_count += len(line)
            if not line.strip():
                raise C200WorkerError(f"{label} contains a blank physical row")
            row_count += 1
    after = _snapshot(path)
    if before != after or before[0] != byte_count:
        raise C200WorkerError(f"{label} changed while it was hashed")
    return SourceIdentity(byte_count, row_count, digest.hexdigest(), after)


def _catalog_identity(path: Path) -> tuple[SourceIdentity, frozenset[str]]:
    _require_regular_input(path, "catalog")
    before = _snapshot(path)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    identifiers: set[str] = set()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            byte_count += len(line)
            if not line.strip():
                raise C200WorkerError("catalog contains a blank physical row")
            row_count += 1
            try:
                product = json.loads(
                    line.decode("utf-8", errors="strict"),
                    object_pairs_hook=_unique_json_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise C200WorkerError("catalog JSONL is invalid") from error
            if not isinstance(product, dict):
                raise C200WorkerError("catalog row is not an object")
            identifier = product.get("parent_asin")
            if (
                not isinstance(identifier, str)
                or not identifier
                or not identifier.isascii()
                or CATALOG_IDENTIFIER_RE.fullmatch(identifier) is None
                or identifier != identifier.upper()
                or identifier in identifiers
            ):
                raise C200WorkerError("catalog identifier shape or uniqueness drifted")
            identifiers.add(identifier)
    after = _snapshot(path)
    if before != after or before[0] != byte_count:
        raise C200WorkerError("catalog changed while it was validated")
    return (
        SourceIdentity(byte_count, row_count, digest.hexdigest(), after),
        frozenset(identifiers),
    )


def _validate_string_list(value: object, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise C200WorkerError(f"visible context {label} is invalid")


def _validate_slot_records(value: object, label: str) -> None:
    if not isinstance(value, list):
        raise C200WorkerError(f"visible context {label} is invalid")
    for record in value:
        if not isinstance(record, dict) or set(record) != SLOT_RECORD_KEYS:
            raise C200WorkerError(f"visible context {label} record is invalid")
        if (
            not isinstance(record["slot"], str)
            or not isinstance(record["value"], str)
            or not isinstance(record["polarity"], int)
            or isinstance(record["polarity"], bool)
            or not isinstance(record["hardness"], str)
            or not isinstance(record["source_turn"], int)
            or isinstance(record["source_turn"], bool)
            or not isinstance(record["version"], int)
            or isinstance(record["version"], bool)
            or not isinstance(record["status"], str)
        ):
            raise C200WorkerError(f"visible context {label} record schema drifted")


def _validate_context_turn_payload(
    value: object,
    catalog_identifiers: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CONTEXT_TURN_KEYS:
        raise C200WorkerError("visible context turn schema drifted")
    if _walk_keys(value) & FORBIDDEN_CONTEXT_KEYS:
        raise C200WorkerError("visible context contains a forbidden key")
    message = value["message"]
    if (
        not isinstance(message, str)
        or not message
        or len(message.encode("utf-8")) > 8_192
        or CONTROL_CHARACTERS_RE.search(message)
    ):
        raise C200WorkerError("visible context message is invalid")
    for name in (
        "goal_messages",
        "active_terms",
        "excluded_terms",
        "query_terms",
        "hard_clause_terms",
    ):
        _validate_string_list(value[name], name)
    if not isinstance(value["category_text"], str):
        raise C200WorkerError("visible context category_text is invalid")
    for name in ("version", "version_anchor_turn", "override_count"):
        observed = value[name]
        if not isinstance(observed, int) or isinstance(observed, bool) or observed < 0:
            raise C200WorkerError(f"visible context {name} is invalid")
    if value["version"] < 1 or not 1 <= value["version_anchor_turn"] <= TURN_COUNT:
        raise C200WorkerError("visible context version boundary is invalid")
    if not isinstance(value["current_turn_override"], bool):
        raise C200WorkerError("visible context override flag is invalid")
    budget = value["budget_upper"]
    if budget is not None and (
        not isinstance(budget, (int, float))
        or isinstance(budget, bool)
        or not math.isfinite(float(budget))
        or float(budget) < 0.0
    ):
        raise C200WorkerError("visible context budget is invalid")
    _validate_slot_records(value["active_records"], "active_records")
    _validate_slot_records(value["retired_records"], "retired_records")
    for text in _walk_strings(value):
        catalog_tokens = {
            match.group(0).upper()
            for match in CATALOG_IDENTIFIER_TOKEN_RE.finditer(text)
        }
        if ASIN_SHAPE_RE.search(text) or catalog_tokens & catalog_identifiers:
            raise C200WorkerError("visible context contains an identifier-shaped token")
    return value


def validate_context_turn(turn: object) -> str:
    """Validate one target-free visible turn and return only its message.

    This deliberately narrow public boundary is used by synthetic structure
    tests: no cached state, future message, identifier, or evaluator field is
    returned to a ranking caller.
    """

    return str(_validate_context_turn_payload(turn, frozenset())["message"])


def _validate_c200_shape(candidates: object) -> tuple[str, ...]:
    if not isinstance(candidates, (list, tuple)):
        raise C200WorkerError("C200 candidates are not an ordered sequence")
    result = tuple(candidates)
    if (
        not MIN_CANDIDATES <= len(result) <= MAX_CANDIDATES
        or len(result) != len(set(result))
        or any(
            not isinstance(identifier, str) or not identifier
            for identifier in result
        )
    ):
        raise C200WorkerError("C200 candidate shape is invalid")
    return result


def validate_c200(
    candidates: object,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    """Return a frozen C200 tuple only when every member is in the catalog."""

    result = _validate_c200_shape(candidates)
    catalog_membership = (
        catalog_ids
        if isinstance(catalog_ids, (set, frozenset, dict))
        else frozenset(catalog_ids)
    )
    if any(identifier not in catalog_membership for identifier in result):
        raise C200WorkerError("C200 contains a non-catalog identifier")
    return result


def canonical_trace_line(
    ordinal: int,
    turn: int,
    c200: object,
) -> bytes:
    """Encode the only allowed candidate-bearing worker trace record."""

    if (
        not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or not 1 <= ordinal <= SESSION_COUNT
        or not isinstance(turn, int)
        or isinstance(turn, bool)
        or not 1 <= turn <= TURN_COUNT
    ):
        raise C200WorkerError("C200 trace coordinate is invalid")
    candidates = _validate_c200_shape(c200)
    return _canonical_bytes(
        {"c200": list(candidates), "ordinal": ordinal, "turn": turn}
    ) + b"\n"


def _parse_context_container(
    line: bytes,
    catalog_identifiers: frozenset[str],
) -> tuple[dict[str, Any], ...]:
    try:
        value = json.loads(
            line.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise C200WorkerError("visible context JSONL is invalid") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "turns"}
        or value.get("schema_version") != CONTEXT_SCHEMA_VERSION
    ):
        raise C200WorkerError("visible context container schema drifted")
    turns = value.get("turns")
    if not isinstance(turns, list) or len(turns) != TURN_COUNT:
        raise C200WorkerError("visible context turn count drifted")
    return tuple(
        _validate_context_turn_payload(turn, catalog_identifiers) for turn in turns
    )


def _validate_context_identity(
    path: Path,
    catalog_identifiers: frozenset[str],
) -> tuple[SourceIdentity, int, int]:
    _require_regular_input(path, "visible context")
    before = _snapshot(path)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    turn_count = 0
    redacted_messages = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            byte_count += len(line)
            if not line.strip():
                raise C200WorkerError("visible context contains a blank physical row")
            row_count += 1
            turns = _parse_context_container(line, catalog_identifiers)
            turn_count += len(turns)
            redacted_messages += sum(
                int(REDACTED_IDENTIFIER in str(turn["message"])) for turn in turns
            )
    after = _snapshot(path)
    if before != after or before[0] != byte_count:
        raise C200WorkerError("visible context changed while it was validated")
    return (
        SourceIdentity(byte_count, row_count, digest.hexdigest(), after),
        turn_count,
        redacted_messages,
    )


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=True).as_posix().casefold() == right.resolve(
            strict=True
        ).as_posix().casefold()
    except (FileNotFoundError, OSError):
        return False


def _validate_environment() -> dict[str, Any]:
    actual = {
        "executable": Path(sys.executable).resolve().as_posix(),
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "pythonhashseed": os.getenv("PYTHONHASHSEED"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    if not (
        _same_path(Path(sys.executable), EXPECTED_EXECUTABLE)
        and actual["python"] == EXPECTED_PYTHON
        and actual["sqlite"] == EXPECTED_SQLITE
        and actual["pythonhashseed"] == "0"
        and actual["cuda_visible_devices"] == ""
    ):
        raise C200WorkerError("frozen worker environment identity drifted")
    return {
        "executable": actual["executable"],
        "python": actual["python"],
        "sqlite": actual["sqlite"],
        "pythonhashseed": actual["pythonhashseed"],
        "network_attempt_count": 0,
        "gpu_used": False,
        "gpu_peak_bytes": 0,
    }


def _validate_declared_inputs(args: argparse.Namespace) -> None:
    expected = {
        "catalog_bytes": EXPECTED_CATALOG_BYTES,
        "catalog_rows": EXPECTED_CATALOG_ROWS,
        "catalog_sha256": EXPECTED_CATALOG_SHA256,
        "context_bytes": EXPECTED_CONTEXT_BYTES,
        "context_rows": EXPECTED_CONTEXT_ROWS,
        "context_turns": EXPECTED_CONTEXT_TURNS,
        "context_sha256": EXPECTED_CONTEXT_SHA256,
    }
    if any(getattr(args, name) != value for name, value in expected.items()):
        raise C200WorkerError("declared frozen input identity drifted")
    if not NONCE_RE.fullmatch(str(args.nonce)):
        raise C200WorkerError("nonce must be 32 lowercase hexadecimal characters")


class C200CaptureAgent(Agent):
    """Explicit control Agent capturing R08 at the frozen pre-P11 boundary."""

    def __init__(self, catalog_path: str | Path) -> None:
        self._c200_last_capture: dict[str, Any] | None = None
        self._c200_sqlite_closed = False
        super().__init__(
            catalog_path,
            llm_client=None,
            question_policy="fast",
            trace_sink=None,
            rerank_mode="off",
            retrieval_mode="coverage",
            p11_mode="control",
            small_ranker_mode="off",
        )
        if self._p11_bridge is not None or self._small_ranker is not None:
            try:
                super().close()
            finally:
                raise C200WorkerError("control Agent unexpectedly initialized an adapter")

    def close(self) -> None:
        super().close()
        try:
            self.connection.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            self._c200_sqlite_closed = True
        else:
            raise C200WorkerError("Agent SQLite remained open after close")

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._c200_last_capture = None
        super().reset(session_id, user_profile)

    def drop_session(self, session_id: str) -> None:
        self._c200_last_capture = None
        super().drop_session(session_id)

    def _apply_p11(
        self,
        state: SessionState,
        rankings: dict[str, list[str]],
        candidate_rowids: dict[str, int],
        query_terms: list[str],
    ) -> tuple[list[str], dict[str, Any]]:
        capture_started = time.perf_counter_ns()
        r08_full = tuple(str(value) for value in rankings.get("final", ()))
        c200 = validate_c200(r08_full, candidate_rowids)
        capture_microseconds = (time.perf_counter_ns() - capture_started) / 1_000.0

        served, diagnostics = super()._apply_p11(
            state, rankings, candidate_rowids, query_terms
        )
        if (
            tuple(served) != r08_full
            or diagnostics.get("configured_mode") != "control"
            or diagnostics.get("output_changed") is not False
        ):
            raise C200WorkerError("control P11 boundary changed the R08 ranking")
        self._c200_last_capture = {
            "state_identity": id(state),
            "c200": c200,
            "capture_microseconds": capture_microseconds,
        }
        return served, diagnostics

    def take_last_capture(self, session_id: str) -> dict[str, Any]:
        state = self._sessions.get(session_id)
        capture, self._c200_last_capture = self._c200_last_capture, None
        if (
            capture is None
            or state is None
            or capture.get("state_identity") != id(state)
        ):
            raise C200WorkerError("C200 capture is missing or misbound")
        return capture


def _served_identifiers(response: object) -> tuple[str, ...]:
    if not isinstance(response, Mapping):
        raise C200WorkerError("Agent response is not an object")
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list) or len(recommendations) != TOP_K:
        raise C200WorkerError("Agent response recommendation count drifted")
    identifiers: list[str] = []
    for recommendation in recommendations:
        if (
            not isinstance(recommendation, Mapping)
            or set(recommendation) != {"parent_asin"}
            or not isinstance(recommendation["parent_asin"], str)
            or not recommendation["parent_asin"]
        ):
            raise C200WorkerError("Agent response recommendation schema drifted")
        identifiers.append(str(recommendation["parent_asin"]))
    return tuple(identifiers)


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise C200WorkerError("cannot summarize an empty measurement")
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _latency_summary(
    values: Sequence[float],
    *,
    unit: str,
) -> dict[str, int | float]:
    if not values or unit not in {"milliseconds", "microseconds"}:
        raise C200WorkerError("latency surface is invalid")
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        f"p50_{unit}": round(_nearest_rank(numbers, 0.50), 6),
        f"p95_{unit}": round(_nearest_rank(numbers, 0.95), 6),
        f"maximum_{unit}": round(max(numbers), 6),
    }


def _pool_length_summary(
    values: Sequence[int],
    *,
    candidate_cells: int,
) -> dict[str, int | float]:
    if not values:
        raise C200WorkerError("candidate pool surface is empty")
    return {
        "min": min(values),
        "p50": int(_nearest_rank(values, 0.50)),
        "p95": int(_nearest_rank(values, 0.95)),
        "max": max(values),
        "mean": round(statistics.fmean(values), 6),
        "records": len(values),
        "candidate_cells": candidate_cells,
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
                return int(counters.PeakWorkingSetSize), "Windows PeakWorkingSetSize"
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    else:
        try:
            import resource

            observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return (
                observed if sys.platform == "darwin" else observed * 1024,
                "resource.getrusage ru_maxrss",
            )
        except (ImportError, OSError, TypeError, ValueError):
            pass
    return None, "unavailable"


def _check_trace_output(path: Path) -> None:
    _require_real_ancestry(path.parent, "C200 trace parent")
    if path.exists() or path.is_symlink() or _is_link_or_reparse(path):
        raise FileExistsError(f"C200 trace already exists: {path}")
    parent = path.parent
    if not parent.is_dir() or _is_link_or_reparse(parent):
        raise C200WorkerError("C200 trace parent must be a prepared real directory")
    resolved_parent = parent.resolve(strict=True)
    if not resolved_parent.is_dir():
        raise C200WorkerError("C200 trace parent is not a directory")


def _publish_trace_exclusive(path: Path, lines: Sequence[bytes]) -> None:
    _check_trace_output(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(str(path), flags, 0o600)
    try:
        for line in lines:
            view = memoryview(line)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short C200 trace write")
                view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _receipt_privacy_scan(value: object) -> None:
    forbidden = FORBIDDEN_CONTEXT_KEYS | frozenset(
        {"c200", "message", "messages", "per_session", "membership_vector"}
    )
    if _walk_keys(value) & forbidden:
        raise C200WorkerError("aggregate receipt contains a forbidden key")
    payload = _canonical_bytes(value).decode("utf-8")
    if ASIN_SHAPE_RE.search(payload):
        raise C200WorkerError("aggregate receipt contains an identifier-shaped token")


def _agent_factory(catalog_path: Path) -> C200CaptureAgent:
    return C200CaptureAgent(catalog_path)


def run(
    args: argparse.Namespace,
    *,
    agent_factory: Callable[[Path], Any] = _agent_factory,
    network_audit: OfflineNetworkAudit | None = None,
) -> dict[str, Any]:
    _validate_declared_inputs(args)
    environment = _validate_environment()
    _check_trace_output(args.trace_output)

    audit = network_audit or OfflineNetworkAudit()
    sys.addaudithook(audit.hook)

    catalog_identity, catalog_identifiers = _catalog_identity(args.catalog)
    context_identity, context_turns, redacted_messages = _validate_context_identity(
        args.context, catalog_identifiers
    )
    if catalog_identity.report() != {
        "bytes": EXPECTED_CATALOG_BYTES,
        "rows": EXPECTED_CATALOG_ROWS,
        "sha256": EXPECTED_CATALOG_SHA256,
    }:
        raise C200WorkerError("official catalog identity drifted")
    if (
        context_identity.report()
        != {
            "bytes": EXPECTED_CONTEXT_BYTES,
            "rows": EXPECTED_CONTEXT_ROWS,
            "sha256": EXPECTED_CONTEXT_SHA256,
        }
        or context_turns != EXPECTED_CONTEXT_TURNS
        or redacted_messages != EXPECTED_REDACTED_MESSAGE_COUNT
    ):
        raise C200WorkerError("visible context identity drifted")

    trace_lines: list[bytes] = []
    trace_digest = hashlib.sha256()
    trace_bytes = 0
    pool_lengths: list[int] = []
    respond_seconds: list[float] = []
    capture_microseconds: list[float] = []
    agent: Any | None = None
    agent_closed = False
    sqlite_closed = False
    processed_sessions = 0
    try:
        agent = agent_factory(args.catalog)
        with args.context.open("rb") as context_handle:
            for ordinal, line in enumerate(context_handle, start=1):
                if ordinal > SESSION_COUNT:
                    raise C200WorkerError("visible context has excess sessions")
                turns = _parse_context_container(line, catalog_identifiers)
                session_id = f"conversation_{ordinal}"
                agent.reset(session_id, {})
                try:
                    for turn, context in enumerate(turns, start=1):
                        # Deliberately pass only the currently visible message.
                        # All other cached fields are validation evidence and never
                        # seed or replace the Agent's own causal state.
                        respond_started = time.perf_counter()
                        response = agent.respond(
                            session_id, validate_context_turn(context), turn, TOP_K
                        )
                        respond_seconds.append(time.perf_counter() - respond_started)
                        capture = agent.take_last_capture(session_id)
                        c200 = validate_c200(
                            capture.get("c200", ()), catalog_identifiers
                        )
                        capture_latency = capture.get("capture_microseconds")
                        if (
                            not isinstance(capture_latency, (int, float))
                            or isinstance(capture_latency, bool)
                            or not math.isfinite(float(capture_latency))
                            or float(capture_latency) < 0.0
                        ):
                            raise C200WorkerError("captured C200 surface is invalid")
                        if _served_identifiers(response) != c200[:TOP_K]:
                            raise C200WorkerError("served Top10 differs from frozen R08")
                        line_bytes = canonical_trace_line(ordinal, turn, c200)
                        trace_bytes += len(line_bytes)
                        if trace_bytes > MAX_TRACE_BYTES:
                            raise C200WorkerError("C200 trace exceeds preregistered bytes")
                        trace_lines.append(line_bytes)
                        trace_digest.update(line_bytes)
                        pool_lengths.append(len(c200))
                        capture_microseconds.append(float(capture_latency))
                finally:
                    agent.drop_session(session_id)
                processed_sessions += 1
        if processed_sessions != SESSION_COUNT or len(trace_lines) != RECORD_COUNT:
            raise C200WorkerError("C200 trajectory is incomplete")
    finally:
        if agent is not None:
            agent.close()
            agent_closed = bool(getattr(agent, "_closed", True))
            sqlite_closed = bool(getattr(agent, "_c200_sqlite_closed", True))

    if not agent_closed or not sqlite_closed:
        raise C200WorkerError("Agent or SQLite was not closed before trace publication")
    if audit.attempt_count != 0:
        raise C200WorkerError("network activity was attempted")

    catalog_end = _raw_jsonl_identity(args.catalog, "catalog")
    context_end = _raw_jsonl_identity(args.context, "visible context")
    if (
        catalog_end.report() != catalog_identity.report()
        or catalog_end.snapshot != catalog_identity.snapshot
        or context_end.report() != context_identity.report()
        or context_end.snapshot != context_identity.snapshot
    ):
        raise C200WorkerError("target-free input changed during the worker run")

    peak_rss, _rss_backend = _peak_rss_bytes()
    if peak_rss is None or not 0 < peak_rss <= MAX_WORKING_SET_BYTES:
        raise C200WorkerError("worker lifetime peak working set is outside budget")
    _publish_trace_exclusive(args.trace_output, trace_lines)
    published = _raw_jsonl_identity(args.trace_output, "published C200 trace")
    if (
        published.bytes != trace_bytes
        or published.rows != RECORD_COUNT
        or published.sha256 != trace_digest.hexdigest()
    ):
        raise C200WorkerError("published C200 trace identity drifted")

    candidate_cells = sum(pool_lengths)
    pool_summary = _pool_length_summary(
        pool_lengths, candidate_cells=candidate_cells
    )
    respond_summary = _latency_summary(
        [value * 1_000.0 for value in respond_seconds], unit="milliseconds"
    )
    capture_summary = _latency_summary(
        capture_microseconds, unit="microseconds"
    )
    environment["network_attempt_count"] = audit.attempt_count
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "configuration": {
            "p11_mode": "control",
            "small_ranker_mode": "off",
            "question_policy": "fast",
            "rerank_mode": "off",
            "retrieval_mode": "coverage",
        },
        "pool_lengths": pool_summary,
        "latency": {
            "respond": respond_summary,
            "capture": capture_summary,
        },
        "resources": {
            "peak_working_set_bytes": peak_rss,
        },
        "lifecycle": {
            "agent_closed_before_trace_publish": agent_closed,
            "sqlite_closed_before_trace_publish": sqlite_closed,
        },
    }
    receipt: dict[str, Any] = {
        "kind": "receipt",
        "nonce": args.nonce,
        "trace_sha256": trace_digest.hexdigest(),
        "trace_bytes": trace_bytes,
        "record_count": len(trace_lines),
        "summary": summary,
    }
    _receipt_privacy_scan(receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-bytes", type=int, required=True)
    parser.add_argument("--catalog-rows", type=int, required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--context-bytes", type=int, required=True)
    parser.add_argument("--context-rows", type=int, required=True)
    parser.add_argument("--context-turns", type=int, required=True)
    parser.add_argument("--context-sha256", required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    receipt = run(_parser().parse_args(argv))
    payload = _canonical_bytes(receipt) + b"\n"
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
