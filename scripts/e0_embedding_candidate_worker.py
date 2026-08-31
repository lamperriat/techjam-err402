"""Target-blind worker for the frozen E0 C200-plus-Dense400 recall probe.

The worker replays the frozen control Agent only to recover each causal
pre-P11 query.  It requires the recomputed variable-length C200 to equal the
sealed C200 reference at every turn, then appends previously unseen identifiers
from the frozen exact Dense-400 route.  The diagnostic union is never served.

Only Python's standard library is imported at module load.  The offline CPU
environment and a fail-closed socket audit are installed before the Agent,
NumPy, tokenizers, ONNX Runtime, or semantic runtime can be imported.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
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


SCHEMA_VERSION = "small-ranker-frozen-embedding-e0-worker-summary.v1"
SESSION_COUNT = 2_000
TURN_COUNT = 10
RECORD_COUNT = SESSION_COUNT * TURN_COUNT
TOP_K = 10
MIN_C200_CANDIDATES = 100
MAX_C200_CANDIDATES = 200
DENSE_DEPTH = 400
MAX_E0_CANDIDATES = 400

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

EXPECTED_MODEL_SPEC_CANONICAL_SHA256 = (
    "e71d0cad480c89eac25ad2b276de9a4e7153e1ec2f3bdcc793682f183a592200"
)
EXPECTED_INDEX_MANIFEST_BYTES = 9_474
EXPECTED_INDEX_MANIFEST_SHA256 = (
    "cca932a8b4d0a160e0a409ec6ce9cf3b68c99e3b95bddb911b9c7d83b67365ba"
)
EXPECTED_INDEX_MATRIX_BYTES = 76_800_128
EXPECTED_INDEX_MATRIX_SHA256 = (
    "84897381c106b909b9e3d44229187d12f23796f108cfec97904db1cbeeb2d407"
)
EXPECTED_INDEX_ASINS_BYTES = 550_000
EXPECTED_INDEX_ASINS_SHA256 = (
    "3af465b23ff2d33614501472edf02d2953ccfc170d2fe3348d55cd51c8ef0d54"
)
EXPECTED_REQUIRED_ASSET_BYTES = 211_493_793
EXPECTED_MODEL_DIR = Path(
    r"D:\tiktok\techjam-err402-fast-track\experiments\p7_assets\bge-small-en-v1.5"
)
EXPECTED_INDEX_DIR = Path(
    r"D:\tiktok\techjam-err402-fast-track\experiments\p7_index"
)
EXPECTED_MODEL_SPEC = PROJECT_ROOT / "configs" / "p7_bge_small_en_v1_5.json"

EXPECTED_EXECUTABLE = Path(r"D:\450\conda\envs\tiktok\python.exe")
EXPECTED_PYTHON = "3.11.16"
EXPECTED_SQLITE = "3.53.4"
EXPECTED_NUMPY = "2.4.6"
EXPECTED_ONNXRUNTIME = "1.29.0"
EXPECTED_TOKENIZERS = "0.23.1"

REFERENCE_C100_CANONICAL_BYTES = 26_690_930
MAX_TRACE_BYTES_OVER_C100 = int(REFERENCE_C100_CANONICAL_BYTES * 4.1)
MAX_TRACE_BYTES_OVER_C200 = int(EXPECTED_C200_REFERENCE_BYTES * 3.4)
MAX_TRACE_BYTES = min(MAX_TRACE_BYTES_OVER_C100, MAX_TRACE_BYTES_OVER_C200)
MAX_CANDIDATE_CELLS = 8_000_000
MAX_WORKING_SET_BYTES = 1_610_612_736
MAX_TOTAL_WALL_SECONDS = 3_600.0
MAX_COLD_INITIALIZATION_SECONDS = 10.0
MAX_RESPOND_P95_MILLISECONDS = 400.0
MAX_DENSE_P95_MILLISECONDS = 100.0

NONCE_RE = re.compile(r"[0-9a-f]{32}")
ASIN_SHAPE_RE = re.compile(
    r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE
)
CATALOG_IDENTIFIER_RE = re.compile(r"[A-Z0-9]{10}")
CATALOG_IDENTIFIER_TOKEN_RE = re.compile(
    r"(?<![A-Z0-9])[A-Z0-9]{10}(?![A-Z0-9])", re.IGNORECASE
)
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


class E0WorkerError(RuntimeError):
    """Raised when a frozen target-blind E0 invariant is violated."""


class OfflineNetworkAudit:
    """Fail closed on all Python-audited socket activity."""

    def __init__(self) -> None:
        self.attempt_count = 0
        self.event_counts: Counter[str] = Counter()

    def hook(self, event: str, _arguments: tuple[object, ...]) -> None:
        if not event.startswith("socket."):
            return
        self.attempt_count += 1
        self.event_counts[event] += 1
        raise PermissionError("network activity is disabled in the E0 worker")


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
        raise E0WorkerError("value is not canonical JSON") from error


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise E0WorkerError("JSON contains a duplicate object key")
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


def _require_real_ancestry(path: Path, label: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if _is_link_or_reparse(component):
            raise E0WorkerError(f"{label} traverses a link or reparse point")


def _require_regular_file(path: Path, label: str) -> None:
    _require_real_ancestry(path, label)
    if not path.is_file():
        raise E0WorkerError(f"{label} is unavailable or unsafe")


def _require_real_directory(path: Path, label: str) -> None:
    _require_real_ancestry(path, label)
    if not path.is_dir() or _is_link_or_reparse(path):
        raise E0WorkerError(f"{label} must be an ordinary local directory")


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=True).as_posix().casefold() == right.resolve(
            strict=True
        ).as_posix().casefold()
    except (FileNotFoundError, OSError):
        return False


def _snapshot(path: Path) -> tuple[int, int, int]:
    observed = path.stat()
    return (
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(getattr(observed, "st_ino", 0)),
    )


def _file_identity(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    before = _snapshot(path)
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    after = _snapshot(path)
    if before != after or byte_count != before[0]:
        raise E0WorkerError(f"{label} changed while it was hashed")
    return {
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "snapshot": after,
    }


def _validate_exact_file(
    path: Path,
    label: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    identity = _file_identity(path, label)
    if (
        identity["bytes"] != expected_bytes
        or identity["sha256"] != expected_sha256
    ):
        raise E0WorkerError(f"{label} identity drifted")
    return identity


def _load_model_spec(path: Path) -> tuple[dict[str, Any], str, tuple[int, int, int]]:
    _require_regular_file(path, "semantic model spec")
    before = _snapshot(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                E0WorkerError(f"semantic model spec contains non-finite {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise E0WorkerError("semantic model spec is invalid") from error
    after = _snapshot(path)
    if before != after or not isinstance(value, dict):
        raise E0WorkerError("semantic model spec changed or is not an object")
    canonical_sha256 = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    if canonical_sha256 != EXPECTED_MODEL_SPEC_CANONICAL_SHA256:
        raise E0WorkerError("semantic model spec canonical identity drifted")
    return value, canonical_sha256, after


def _snapshot_model_assets(
    model_dir: Path,
    spec: Mapping[str, Any],
) -> tuple[tuple[Path, tuple[int, int, int]], ...]:
    required = spec.get("required_files")
    if not isinstance(required, list) or not required:
        raise E0WorkerError("semantic model required-file list is invalid")
    observed: list[tuple[Path, tuple[int, int, int]]] = []
    seen: set[Path] = set()
    for entry in required:
        if not isinstance(entry, Mapping):
            raise E0WorkerError("semantic model required-file entry is invalid")
        relative = entry.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
        ):
            raise E0WorkerError("semantic model required-file path is unsafe")
        path = model_dir / relative
        _require_regular_file(path, "semantic model asset")
        if path in seen:
            raise E0WorkerError("semantic model required-file path is duplicated")
        seen.add(path)
        snapshot = _snapshot(path)
        if snapshot[0] != entry.get("bytes"):
            raise E0WorkerError("semantic model asset size drifted")
        observed.append((path, snapshot))
    return tuple(observed)


def _catalog_membership(catalog_ids: Iterable[str]) -> set[str] | frozenset[str]:
    if isinstance(catalog_ids, (set, frozenset)):
        return catalog_ids
    if isinstance(catalog_ids, Mapping):
        return set(str(value) for value in catalog_ids)
    return frozenset(catalog_ids)


def _validate_c200_values(
    candidates: object,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    if not isinstance(candidates, (list, tuple)):
        raise E0WorkerError("C200 candidates are not an ordered sequence")
    values = tuple(candidates)
    catalog = _catalog_membership(catalog_ids)
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
        raise E0WorkerError("C200 candidate surface is invalid")
    return values


def append_dense_unseen_tail(
    c200: object,
    dense: object,
    catalog_ids: Iterable[str],
    *,
    limit: int = MAX_E0_CANDIDATES,
) -> tuple[str, ...]:
    """Append the Dense-400 order after the complete variable C200 prefix."""

    if not isinstance(limit, int) or isinstance(limit, bool) or limit != 400:
        raise E0WorkerError("E0 union limit must remain frozen at 400")
    prefix = _validate_c200_values(c200, catalog_ids)
    if not isinstance(dense, (list, tuple)):
        raise E0WorkerError("Dense-400 route is not an ordered sequence")
    dense_values = tuple(dense)
    catalog = _catalog_membership(catalog_ids)
    if (
        len(dense_values) > DENSE_DEPTH
        or any(
            not isinstance(identifier, str)
            or not identifier
            or identifier not in catalog
            for identifier in dense_values
        )
    ):
        raise E0WorkerError("Dense-400 candidate surface is invalid")

    result = list(prefix)
    seen = set(prefix)
    for identifier in dense_values:
        if identifier in seen:
            continue
        result.append(identifier)
        seen.add(identifier)
        if len(result) == limit:
            break
    return tuple(result)


def validate_e0_candidates(
    candidates: object,
    sealed_c200: object,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    """Validate catalog membership and the exact ordered variable-C200 prefix."""

    reference = _validate_c200_values(sealed_c200, catalog_ids)
    if not isinstance(candidates, (list, tuple)):
        raise E0WorkerError("E0 candidates are not an ordered sequence")
    values = tuple(candidates)
    catalog = _catalog_membership(catalog_ids)
    if (
        not len(reference) <= len(values) <= MAX_E0_CANDIDATES
        or len(values) != len(set(values))
        or values[: len(reference)] != reference
        or any(
            not isinstance(identifier, str)
            or not identifier
            or identifier not in catalog
            for identifier in values
        )
    ):
        raise E0WorkerError("E0 lost, reordered, duplicated, or invented a candidate")
    return values


def canonical_trace_line(ordinal: int, turn: int, candidates: object) -> bytes:
    """Encode the sole candidate-bearing E0 trace schema."""

    if (
        not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or not 1 <= ordinal <= SESSION_COUNT
        or not isinstance(turn, int)
        or isinstance(turn, bool)
        or not 1 <= turn <= TURN_COUNT
        or not isinstance(candidates, (list, tuple))
    ):
        raise E0WorkerError("E0 trace coordinate or candidates are invalid")
    values = tuple(candidates)
    if (
        not MIN_C200_CANDIDATES <= len(values) <= MAX_E0_CANDIDATES
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise E0WorkerError("E0 trace candidate shape is invalid")
    return _canonical_bytes(
        {"candidates": list(values), "ordinal": ordinal, "turn": turn}
    ) + b"\n"


def validate_context_turn(turn: object) -> str:
    """Expose only a currently visible message, never cached state or labels."""

    if not isinstance(turn, Mapping):
        raise E0WorkerError("visible context turn is not an object")
    keys = {str(key).casefold() for key in turn}
    if keys & FORBIDDEN_CONTEXT_KEYS:
        raise E0WorkerError("visible context contains a forbidden field")
    message = turn.get("message")
    if (
        not isinstance(message, str)
        or not message.strip()
        or ASIN_SHAPE_RE.search(message) is not None
    ):
        raise E0WorkerError("visible context message is invalid")
    return message


def parse_c200_reference_line(
    line: bytes,
    *,
    ordinal: int,
    turn: int,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    """Parse a canonical sealed C200 row at its exact global coordinate."""

    try:
        value = json.loads(
            line.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                E0WorkerError(f"C200 reference contains non-finite {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise E0WorkerError("C200 reference JSONL is invalid") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"c200", "ordinal", "turn"}
        or value.get("ordinal") != ordinal
        or isinstance(value.get("ordinal"), bool)
        or value.get("turn") != turn
        or isinstance(value.get("turn"), bool)
    ):
        raise E0WorkerError("C200 reference schema or order drifted")
    candidates = _validate_c200_values(value.get("c200"), catalog_ids)
    expected = _canonical_bytes(
        {"c200": list(candidates), "ordinal": ordinal, "turn": turn}
    ) + b"\n"
    if line != expected:
        raise E0WorkerError("C200 reference row is not canonical LF JSON")
    return candidates


def _validate_c200_reference_identity(
    path: Path,
    catalog_ids: frozenset[str],
    c200_contract: Any,
) -> tuple[Any, int]:
    c200_contract._require_regular_input(path, "sealed C200 reference")
    before = c200_contract._snapshot(path)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    candidate_cells = 0
    with path.open("rb") as handle:
        for line in handle:
            row_count += 1
            ordinal = (row_count - 1) // TURN_COUNT + 1
            turn = (row_count - 1) % TURN_COUNT + 1
            if row_count > RECORD_COUNT or not line.strip():
                raise E0WorkerError("C200 reference row count or framing drifted")
            candidates = parse_c200_reference_line(
                line,
                ordinal=ordinal,
                turn=turn,
                catalog_ids=catalog_ids,
            )
            digest.update(line)
            byte_count += len(line)
            candidate_cells += len(candidates)
    after = c200_contract._snapshot(path)
    identity = c200_contract.SourceIdentity(
        byte_count, row_count, digest.hexdigest(), after
    )
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
        raise E0WorkerError("sealed C200 reference identity drifted")
    return identity, candidate_cells


def _freeze_runtime_environment(spec: Mapping[str, Any]) -> dict[str, str]:
    runtime = spec.get("runtime")
    if not isinstance(runtime, Mapping):
        raise E0WorkerError("semantic runtime specification is missing")
    frozen = runtime.get("environment_before_numpy_or_onnxruntime_import")
    expected = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
    }
    if not isinstance(frozen, Mapping) or dict(frozen) != expected:
        raise E0WorkerError("semantic CPU/offline environment contract drifted")
    already_loaded = sorted(
        module for module in ("numpy", "tokenizers", "onnxruntime") if module in sys.modules
    )
    if already_loaded:
        raise E0WorkerError(
            "optional semantic modules loaded before E0 environment/audit: "
            + ",".join(already_loaded)
        )
    for key, value in expected.items():
        os.environ[key] = value
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.environ["PYTHONPATH"] = ""
    os.environ.pop("PYTHONHOME", None)
    if os.getenv("PYTHONHASHSEED") != "0":
        raise E0WorkerError("worker must start with PYTHONHASHSEED=0")
    return {
        **expected,
        "CUDA_VISIBLE_DEVICES": "",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def _load_runtime_after_audit() -> tuple[Any, Any]:
    """Import target-free Agent and semantic modules only after run installs audit."""

    try:
        c200_contract = importlib.import_module("scripts.c200_candidate_worker")
        semantic = importlib.import_module("starter.semantic")
    except ImportError as error:
        raise E0WorkerError("cannot import the frozen Agent/semantic runtime") from error
    return c200_contract, semantic


class E0CaptureAgent:
    """Import-free mixin adding a read-only Dense-400 shadow to C200CaptureAgent."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        encoder: Any,
        semantic_index: Any,
        catalog_ids: frozenset[str],
    ) -> None:
        self._e0_encoder = encoder
        self._e0_index = semantic_index
        self._e0_catalog_ids = catalog_ids
        super().__init__(catalog_path)  # type: ignore[misc]

    def _apply_p11(
        self,
        state: Any,
        rankings: dict[str, list[str]],
        candidate_rowids: dict[str, int],
        query_terms: list[str],
    ) -> tuple[list[str], dict[str, Any]]:
        if not isinstance(query_terms, list) or any(
            not isinstance(term, str) for term in query_terms
        ):
            raise E0WorkerError("pre-P11 query terms drifted")
        ranking_snapshot = {
            route: tuple(values) for route, values in rankings.items()
        }
        rowid_snapshot = dict(candidate_rowids)
        production_c200 = _validate_c200_values(
            tuple(str(value) for value in rankings.get("final", ())),
            self._e0_catalog_ids,
        )

        dense_started = time.perf_counter_ns()
        query = " ".join(query_terms)
        if query.strip():
            hits = self._e0_index.search_query(
                query,
                self._e0_encoder,
                top_k=DENSE_DEPTH,
            )
            dense = tuple(str(hit.parent_asin) for hit in hits)
            if len(dense) != DENSE_DEPTH or len(dense) != len(set(dense)):
                raise E0WorkerError("non-empty query did not produce exact Dense-400")
        else:
            dense = ()
        dense_milliseconds = (
            time.perf_counter_ns() - dense_started
        ) / 1_000_000.0
        candidates = append_dense_unseen_tail(
            production_c200,
            dense,
            self._e0_catalog_ids,
        )

        served, diagnostics = super()._apply_p11(  # type: ignore[misc]
            state,
            rankings,
            candidate_rowids,
            query_terms,
        )
        if (
            {route: tuple(values) for route, values in rankings.items()}
            != ranking_snapshot
            or candidate_rowids != rowid_snapshot
            or tuple(served) != production_c200
        ):
            raise E0WorkerError("diagnostic Dense-400 mutated production ranking")
        capture = getattr(self, "_c200_last_capture", None)
        if (
            not isinstance(capture, dict)
            or tuple(capture.get("c200", ())) != production_c200
        ):
            raise E0WorkerError("production C200 capture drifted")
        capture.update(
            {
                "e0_candidates": candidates,
                "dense_count": len(dense),
                "dense_milliseconds": dense_milliseconds,
                "query_empty": not bool(query.strip()),
            }
        )
        return served, diagnostics


def _runtime_agent_class(c200_contract: Any) -> type:
    return type(
        "RuntimeE0CaptureAgent",
        (E0CaptureAgent, c200_contract.C200CaptureAgent),
        {"__module__": __name__},
    )


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise E0WorkerError("cannot summarize an empty measurement")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _latency_summary(values: Sequence[float]) -> dict[str, int | float]:
    numbers = [float(value) for value in values]
    if not numbers or any(
        not math.isfinite(value) or value < 0.0 for value in numbers
    ):
        raise E0WorkerError("latency surface is invalid")
    return {
        "count": len(numbers),
        "p50_milliseconds": round(_nearest_rank(numbers, 0.50), 6),
        "p95_milliseconds": round(_nearest_rank(numbers, 0.95), 6),
        "maximum_milliseconds": round(max(numbers), 6),
    }


def _pool_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise E0WorkerError("candidate pool surface is invalid")
    return {
        "min": min(values),
        "p50": int(_nearest_rank(values, 0.50)),
        "p95": int(_nearest_rank(values, 0.95)),
        "max": max(values),
        "mean": round(statistics.fmean(values), 6),
        "records": len(values),
        "candidate_cells": sum(values),
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
    _require_real_ancestry(path.parent, "E0 trace parent")
    if path.exists() or path.is_symlink() or _is_link_or_reparse(path):
        raise FileExistsError(f"E0 trace already exists: {path}")
    if not path.parent.is_dir() or _is_link_or_reparse(path.parent):
        raise E0WorkerError("E0 trace parent must be a prepared real directory")


def _publish_trace_exclusive(path: Path, lines: Sequence[bytes]) -> None:
    """Durably publish one trace without overwrite or rename replacement."""

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
                    raise OSError("short E0 trace write")
                view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _receipt_privacy_scan(
    value: object,
    *,
    catalog_ids: Iterable[str] = (),
) -> None:
    forbidden = FORBIDDEN_CONTEXT_KEYS | frozenset(
        {
            "candidates",
            "c200",
            "e0_candidates",
            "message",
            "messages",
            "per_session",
            "membership_vector",
            "query",
            "query_terms",
        }
    )
    if _walk_keys(value) & forbidden:
        raise E0WorkerError("aggregate receipt contains a forbidden key")
    payload = _canonical_bytes(value).decode("utf-8")
    if ASIN_SHAPE_RE.search(payload) is not None:
        raise E0WorkerError("aggregate receipt contains an identifier-shaped token")
    catalog = {str(identifier).casefold() for identifier in catalog_ids}
    tokens = {
        match.group(0).casefold()
        for match in CATALOG_IDENTIFIER_TOKEN_RE.finditer(payload)
    }
    if tokens & catalog:
        raise E0WorkerError("aggregate receipt contains a catalog identifier")


def _validate_declared_inputs(args: argparse.Namespace) -> None:
    expected = {
        "catalog_bytes": EXPECTED_CATALOG_BYTES,
        "catalog_rows": EXPECTED_CATALOG_ROWS,
        "catalog_sha256": EXPECTED_CATALOG_SHA256,
        "context_bytes": EXPECTED_CONTEXT_BYTES,
        "context_rows": EXPECTED_CONTEXT_ROWS,
        "context_turns": EXPECTED_CONTEXT_TURNS,
        "context_sha256": EXPECTED_CONTEXT_SHA256,
        "c200_reference_bytes": EXPECTED_C200_REFERENCE_BYTES,
        "c200_reference_rows": EXPECTED_C200_REFERENCE_ROWS,
        "c200_reference_sha256": EXPECTED_C200_REFERENCE_SHA256,
        "model_spec_canonical_sha256": EXPECTED_MODEL_SPEC_CANONICAL_SHA256,
        "index_manifest_bytes": EXPECTED_INDEX_MANIFEST_BYTES,
        "index_manifest_sha256": EXPECTED_INDEX_MANIFEST_SHA256,
        "index_matrix_bytes": EXPECTED_INDEX_MATRIX_BYTES,
        "index_matrix_sha256": EXPECTED_INDEX_MATRIX_SHA256,
        "index_asins_bytes": EXPECTED_INDEX_ASINS_BYTES,
        "index_asins_sha256": EXPECTED_INDEX_ASINS_SHA256,
    }
    if any(getattr(args, name) != value for name, value in expected.items()):
        raise E0WorkerError("declared frozen input or asset identity drifted")
    if not NONCE_RE.fullmatch(str(args.nonce)):
        raise E0WorkerError("nonce must be 32 lowercase hexadecimal characters")
    if args.session_limit not in {2, SESSION_COUNT}:
        raise E0WorkerError("session limit must be target-free preflight 2 or formal 2000")
    if not _same_path(args.model_spec, EXPECTED_MODEL_SPEC):
        raise E0WorkerError("semantic model spec path drifted")
    if not _same_path(args.model_dir, EXPECTED_MODEL_DIR):
        raise E0WorkerError("frozen model directory path drifted")
    if not _same_path(args.index_dir, EXPECTED_INDEX_DIR):
        raise E0WorkerError("frozen index directory path drifted")


def _validate_environment(runtime_spec: Mapping[str, Any]) -> dict[str, Any]:
    actual = {
        "executable": Path(sys.executable).resolve().as_posix(),
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "numpy": str(getattr(sys.modules.get("numpy"), "__version__", "")),
        "onnxruntime": str(
            getattr(sys.modules.get("onnxruntime"), "__version__", "")
        ),
        "tokenizers": str(
            getattr(sys.modules.get("tokenizers"), "__version__", "")
        ),
        "provider": runtime_spec.get("execution_provider"),
        "pythonhashseed": os.getenv("PYTHONHASHSEED"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    if not (
        _same_path(Path(sys.executable), EXPECTED_EXECUTABLE)
        and actual["python"] == EXPECTED_PYTHON
        and actual["sqlite"] == EXPECTED_SQLITE
        and actual["numpy"] == EXPECTED_NUMPY
        and actual["onnxruntime"] == EXPECTED_ONNXRUNTIME
        and actual["tokenizers"] == EXPECTED_TOKENIZERS
        and actual["provider"] == "CPUExecutionProvider"
        and actual["pythonhashseed"] == "0"
        and actual["cuda_visible_devices"] == ""
    ):
        raise E0WorkerError("frozen E0 runtime environment identity drifted")
    return {
        **actual,
        "network_attempt_count": 0,
        "gpu_used": False,
        "gpu_peak_bytes": 0,
    }


def _validate_end_identity(path: Path, label: str, expected: Any, contract: Any) -> None:
    observed = contract._raw_jsonl_identity(path, label)
    if observed.report() != expected.report() or observed.snapshot != expected.snapshot:
        raise E0WorkerError(f"{label} changed during the worker run")


def run(
    args: argparse.Namespace,
    *,
    network_audit: OfflineNetworkAudit | None = None,
    runtime_loader: Callable[[], tuple[Any, Any]] = _load_runtime_after_audit,
    agent_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one fresh target-free E0 worker and return an aggregate receipt."""

    wall_started = time.perf_counter()
    _validate_declared_inputs(args)
    _check_trace_output(args.trace_output)
    _require_real_directory(args.model_dir, "frozen model root")
    _require_real_directory(args.index_dir, "frozen index root")

    spec, spec_canonical_sha256, spec_snapshot = _load_model_spec(args.model_spec)
    _freeze_runtime_environment(spec)
    audit = network_audit or OfflineNetworkAudit()
    sys.addaudithook(audit.hook)

    c200_contract, semantic = runtime_loader()
    semantic.validate_semantic_spec(spec)

    catalog_identity, catalog_ids = c200_contract._catalog_identity(args.catalog)
    context_identity, context_turns, redacted_messages = (
        c200_contract._validate_context_identity(args.context, catalog_ids)
    )
    reference_identity, reference_cells = _validate_c200_reference_identity(
        args.c200_reference,
        catalog_ids,
        c200_contract,
    )
    if catalog_identity.report() != {
        "bytes": EXPECTED_CATALOG_BYTES,
        "rows": EXPECTED_CATALOG_ROWS,
        "sha256": EXPECTED_CATALOG_SHA256,
    }:
        raise E0WorkerError("official catalog identity drifted")
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
    ):
        raise E0WorkerError("sealed target-free source identity drifted")

    cold_started = time.perf_counter()
    model_asset_snapshots = _snapshot_model_assets(args.model_dir, spec)
    index_manifest = _validate_exact_file(
        args.index_dir / "semantic-index.manifest.json",
        "semantic index manifest",
        expected_bytes=EXPECTED_INDEX_MANIFEST_BYTES,
        expected_sha256=EXPECTED_INDEX_MANIFEST_SHA256,
    )
    index_matrix = _validate_exact_file(
        args.index_dir / "embeddings.npy",
        "semantic index matrix",
        expected_bytes=EXPECTED_INDEX_MATRIX_BYTES,
        expected_sha256=EXPECTED_INDEX_MATRIX_SHA256,
    )
    index_asins = _validate_exact_file(
        args.index_dir / "parent_asins.txt",
        "semantic index ordered identifiers",
        expected_bytes=EXPECTED_INDEX_ASINS_BYTES,
        expected_sha256=EXPECTED_INDEX_ASINS_SHA256,
    )
    model_asset_bytes = sum(
        int(entry["bytes"]) for entry in spec.get("required_files", ())
    )
    computed_required_assets = (
        model_asset_bytes
        + EXPECTED_INDEX_MANIFEST_BYTES
        + EXPECTED_INDEX_MATRIX_BYTES
        + EXPECTED_INDEX_ASINS_BYTES
        + 1_065
    )
    if computed_required_assets != EXPECTED_REQUIRED_ASSET_BYTES:
        raise E0WorkerError("frozen required-asset byte accounting drifted")

    encoder: Any | None = None
    semantic_index: Any | None = None
    agent: Any | None = None
    agent_closed = False
    sqlite_closed = False
    encoder_closed = False
    index_closed = False
    trace_lines: list[bytes] = []
    trace_digest = hashlib.sha256()
    trace_bytes = 0
    pool_lengths: list[int] = []
    dense_lengths: list[int] = []
    c200_lengths: list[int] = []
    respond_milliseconds: list[float] = []
    dense_milliseconds: list[float] = []
    empty_query_count = 0
    processed_sessions = 0
    reference_rows = 0
    cold_initialization_seconds = 0.0

    try:
        encoder = semantic.OfflineSemanticEncoder.from_frozen_assets(
            spec,
            args.model_dir,
        )
        semantic_index = semantic.SemanticIndex.load(
            spec,
            args.index_dir,
            expected_catalog_sha256=EXPECTED_CATALOG_SHA256,
            numpy_module=getattr(encoder, "_np", None),
        )
        cold_initialization_seconds = time.perf_counter() - cold_started

        runtime_class = _runtime_agent_class(c200_contract)
        if agent_factory is None:
            agent = runtime_class(
                args.catalog,
                encoder=encoder,
                semantic_index=semantic_index,
                catalog_ids=catalog_ids,
            )
        else:
            agent = agent_factory(
                args.catalog,
                encoder=encoder,
                semantic_index=semantic_index,
                catalog_ids=catalog_ids,
            )

        with args.context.open("rb") as context_handle, args.c200_reference.open(
            "rb"
        ) as reference_handle:
            for ordinal in range(1, args.session_limit + 1):
                context_line = context_handle.readline()
                if not context_line:
                    raise E0WorkerError("visible context ended before session limit")
                contexts = c200_contract._parse_context_container(
                    context_line,
                    catalog_ids,
                )
                session_id = f"conversation_{ordinal}"
                agent.reset(session_id, {})
                try:
                    for turn, context in enumerate(contexts, start=1):
                        reference_line = reference_handle.readline()
                        if not reference_line:
                            raise E0WorkerError("sealed C200 reference ended early")
                        reference_rows += 1
                        sealed_c200 = parse_c200_reference_line(
                            reference_line,
                            ordinal=ordinal,
                            turn=turn,
                            catalog_ids=catalog_ids,
                        )

                        respond_started = time.perf_counter_ns()
                        response = agent.respond(
                            session_id,
                            validate_context_turn(context),
                            turn,
                            TOP_K,
                        )
                        respond_milliseconds.append(
                            (time.perf_counter_ns() - respond_started) / 1_000_000.0
                        )
                        capture = agent.take_last_capture(session_id)
                        production_c200 = tuple(capture.get("c200", ()))
                        candidates = validate_e0_candidates(
                            capture.get("e0_candidates", ()),
                            sealed_c200,
                            catalog_ids,
                        )
                        dense_count = capture.get("dense_count")
                        dense_latency = capture.get("dense_milliseconds")
                        query_empty = capture.get("query_empty")
                        if (
                            production_c200 != sealed_c200
                            or not isinstance(dense_count, int)
                            or isinstance(dense_count, bool)
                            or dense_count not in {0, DENSE_DEPTH}
                            or not isinstance(dense_latency, (int, float))
                            or isinstance(dense_latency, bool)
                            or not math.isfinite(float(dense_latency))
                            or float(dense_latency) < 0.0
                            or not isinstance(query_empty, bool)
                            or query_empty != (dense_count == 0)
                        ):
                            raise E0WorkerError(
                                "sealed C200 identity or Dense-400 capture drifted"
                            )
                        if c200_contract._served_identifiers(response) != sealed_c200[:TOP_K]:
                            raise E0WorkerError("served Top10 differs from sealed C200")

                        trace_line = canonical_trace_line(ordinal, turn, candidates)
                        trace_bytes += len(trace_line)
                        if trace_bytes > MAX_TRACE_BYTES:
                            raise E0WorkerError("E0 trace exceeds frozen byte budget")
                        trace_lines.append(trace_line)
                        trace_digest.update(trace_line)
                        pool_lengths.append(len(candidates))
                        c200_lengths.append(len(sealed_c200))
                        dense_lengths.append(dense_count)
                        dense_milliseconds.append(float(dense_latency))
                        empty_query_count += int(query_empty)
                finally:
                    agent.drop_session(session_id)
                processed_sessions += 1

            if args.session_limit == SESSION_COUNT:
                if context_handle.read(1) != b"":
                    raise E0WorkerError("visible context has excess sessions")
                if reference_handle.read(1) != b"":
                    raise E0WorkerError("sealed C200 reference has excess rows")

        expected_records = args.session_limit * TURN_COUNT
        if (
            processed_sessions != args.session_limit
            or reference_rows != expected_records
            or len(trace_lines) != expected_records
        ):
            raise E0WorkerError("E0 trajectory is incomplete")
    finally:
        try:
            if agent is not None:
                agent.close()
                agent_closed = bool(getattr(agent, "_closed", True))
                sqlite_closed = bool(getattr(agent, "_c200_sqlite_closed", True))
        finally:
            try:
                if semantic_index is not None:
                    semantic_index.close()
                    index_closed = bool(getattr(semantic_index, "_closed", True))
            finally:
                if encoder is not None:
                    encoder.close()
                    encoder_closed = bool(getattr(encoder, "_closed", True))

    if not (agent_closed and sqlite_closed and encoder_closed and index_closed):
        raise E0WorkerError("Agent, SQLite, encoder, or index was not closed")
    if audit.attempt_count != 0:
        raise E0WorkerError("network activity was attempted")

    _validate_end_identity(args.catalog, "catalog", catalog_identity, c200_contract)
    _validate_end_identity(
        args.context,
        "visible context",
        context_identity,
        c200_contract,
    )
    _validate_end_identity(
        args.c200_reference,
        "sealed C200 reference",
        reference_identity,
        c200_contract,
    )
    if _snapshot(args.model_spec) != spec_snapshot:
        raise E0WorkerError("semantic model spec changed during the worker run")
    if any(_snapshot(path) != snapshot for path, snapshot in model_asset_snapshots):
        raise E0WorkerError("semantic model asset changed during the worker run")
    for path, label, identity in (
        (
            args.index_dir / "semantic-index.manifest.json",
            "semantic index manifest",
            index_manifest,
        ),
        (args.index_dir / "embeddings.npy", "semantic index matrix", index_matrix),
        (
            args.index_dir / "parent_asins.txt",
            "semantic index ordered identifiers",
            index_asins,
        ),
    ):
        if _snapshot(path) != identity["snapshot"]:
            raise E0WorkerError(f"{label} changed during the worker run")

    environment = _validate_environment(spec["runtime"])
    environment["network_attempt_count"] = audit.attempt_count
    pool = _pool_summary(pool_lengths)
    c200_pool = _pool_summary(c200_lengths)
    dense_pool = _pool_summary(dense_lengths)
    respond_latency = _latency_summary(respond_milliseconds)
    dense_latency = _latency_summary(dense_milliseconds)
    peak_rss, rss_backend = _peak_rss_bytes()
    wall_seconds = time.perf_counter() - wall_started

    if pool["candidate_cells"] > MAX_CANDIDATE_CELLS:
        raise E0WorkerError("E0 candidate cells exceed the frozen budget")
    if args.session_limit == SESSION_COUNT:
        if (
            float(pool["candidate_cells"]) / (SESSION_COUNT * 100) > 4.0
            or float(pool["candidate_cells"]) / EXPECTED_C200_CANDIDATE_CELLS > 3.3
            or trace_bytes / REFERENCE_C100_CANONICAL_BYTES > 4.1
            or trace_bytes / EXPECTED_C200_REFERENCE_BYTES > 3.4
        ):
            raise E0WorkerError("E0 candidate or trace inflation exceeds budget")
    if (
        peak_rss is None
        or not 0 < peak_rss <= MAX_WORKING_SET_BYTES
        or cold_initialization_seconds > MAX_COLD_INITIALIZATION_SECONDS
        or float(respond_latency["p95_milliseconds"])
        > MAX_RESPOND_P95_MILLISECONDS
        or float(dense_latency["p95_milliseconds"]) > MAX_DENSE_P95_MILLISECONDS
        or wall_seconds > MAX_TOTAL_WALL_SECONDS
    ):
        raise E0WorkerError("E0 runtime resource budget failed")

    _publish_trace_exclusive(args.trace_output, trace_lines)
    published = c200_contract._raw_jsonl_identity(
        args.trace_output,
        "published E0 trace",
    )
    expected_records = args.session_limit * TURN_COUNT
    if (
        published.bytes != trace_bytes
        or published.rows != expected_records
        or published.sha256 != trace_digest.hexdigest()
    ):
        raise E0WorkerError("published E0 trace identity drifted")
    wall_seconds = time.perf_counter() - wall_started
    if wall_seconds > MAX_TOTAL_WALL_SECONDS:
        raise E0WorkerError("E0 worker exceeded wall budget during trace publication")

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "configuration": {
            "dense_depth": DENSE_DEPTH,
            "diagnostic_only": True,
            "p11_mode": "control",
            "question_policy": "fast",
            "rerank_mode": "off",
            "retrieval_mode": "coverage",
            "served_top10_unchanged": True,
            "small_ranker_mode": "off",
            "stable_append_after_complete_variable_c200": True,
        },
        "input_identities": {
            "catalog": catalog_identity.report(),
            "visible_context": context_identity.report(),
            "sealed_c200_reference": reference_identity.report(),
        },
        "asset_identities": {
            "model_spec_canonical_sha256": spec_canonical_sha256,
            "model_required_file_count": len(spec.get("required_files", ())),
            "model_required_file_bytes": model_asset_bytes,
            "model_files_bundle_sha256": hashlib.sha256(
                _canonical_bytes(spec.get("required_files", ()))
            ).hexdigest(),
            "index_manifest": {
                "bytes": index_manifest["bytes"],
                "sha256": index_manifest["sha256"],
            },
            "index_matrix": {
                "bytes": index_matrix["bytes"],
                "sha256": index_matrix["sha256"],
            },
            "index_order": {
                "bytes": index_asins["bytes"],
                "sha256": index_asins["sha256"],
            },
            "required_asset_bytes": computed_required_assets,
        },
        "pool_lengths": pool,
        "control_pool_lengths": c200_pool,
        "dense_lengths": dense_pool,
        "empty_query_count": empty_query_count,
        "latency": {
            "cold_semantic_initialization_seconds": round(
                cold_initialization_seconds, 6
            ),
            "respond": respond_latency,
            "dense_query_and_exact_search": dense_latency,
        },
        "resources": {
            "peak_working_set_bytes": peak_rss,
            "peak_working_set_backend": rss_backend,
            "wall_seconds": round(wall_seconds, 6),
        },
        "lifecycle": {
            "agent_closed_before_trace_publish": agent_closed,
            "sqlite_closed_before_trace_publish": sqlite_closed,
            "encoder_closed_before_trace_publish": encoder_closed,
            "index_closed_before_trace_publish": index_closed,
            "catalog_unchanged_before_trace_publish": True,
            "context_unchanged_before_trace_publish": True,
            "c200_reference_unchanged_before_trace_publish": True,
            "model_spec_unchanged_before_trace_publish": True,
            "model_files_unchanged_before_trace_publish": True,
            "index_files_unchanged_before_trace_publish": True,
        },
        "session_limit": args.session_limit,
        "processed_sessions": processed_sessions,
        "processed_turns": expected_records,
    }
    receipt: dict[str, Any] = {
        "kind": "receipt",
        "nonce": args.nonce,
        "trace_sha256": trace_digest.hexdigest(),
        "trace_bytes": trace_bytes,
        "record_count": expected_records,
        "summary": summary,
    }
    _receipt_privacy_scan(receipt, catalog_ids=catalog_ids)
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
    parser.add_argument("--c200-reference", type=Path, required=True)
    parser.add_argument("--c200-reference-bytes", type=int, required=True)
    parser.add_argument("--c200-reference-rows", type=int, required=True)
    parser.add_argument("--c200-reference-sha256", required=True)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--model-spec-canonical-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--index-manifest-bytes", type=int, required=True)
    parser.add_argument("--index-manifest-sha256", required=True)
    parser.add_argument("--index-matrix-bytes", type=int, required=True)
    parser.add_argument("--index-matrix-sha256", required=True)
    parser.add_argument("--index-asins-bytes", type=int, required=True)
    parser.add_argument("--index-asins-sha256", required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument(
        "--session-limit",
        type=int,
        choices=(2, SESSION_COUNT),
        default=SESSION_COUNT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    receipt = run(_parser().parse_args(argv))
    sys.stdout.buffer.write(_canonical_bytes(receipt) + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
