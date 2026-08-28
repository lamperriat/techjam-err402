from __future__ import annotations

"""Label-free C00/S00 capture wrapper for the P7 dense-shadow study."""

import copy
import hashlib
import importlib
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from starter.agent import Agent, SessionState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "p7.target-blind-capture.v1"
INDEX_LOCK_SCHEMA_VERSION = "p7.semantic-index-lock.v1"
DEFAULT_INDEX_LOCK_PATH = PROJECT_ROOT / "configs" / "p7_semantic_index_lock.json"
C00 = "P7.C00.r08_coverage"
S00 = "P7.S00.bge_dense_shadow"
CONTROL_ID = C00
SHADOW_ID = S00
DENSE_DEPTH = 120
MAX_CAPTURE_RECORDS = 2_000
FROZEN_CATALOG_SHA256 = (
    "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
)
FROZEN_CATALOG_ROWS = 50_000

DenseSearch = Callable[[str, int], Iterable[Any]]


@dataclass(slots=True)
class _RespondContext:
    ordinal: int
    turn: int
    state_identity: int
    route_started: bool = False
    route_captured: bool = False


def _canonical_copy(value: Any) -> Any:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"capture is not canonical JSON-safe: {exc}") from exc
    return json.loads(payload)


def canonical_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root must be an object")
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _safe_path(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path must be a non-empty string")
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} path must be a safe relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes its root") from exc
    return resolved


def _validate_locked_file(
    path: Path, entry: Mapping[str, Any], label: str
) -> None:
    expected_bytes = _positive_int(entry.get("bytes"), f"{label}.bytes")
    expected_sha256 = _sha256_text(entry.get("sha256"), f"{label}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"locked {label} file is missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"locked {label} bytes mismatch: {actual_bytes} != {expected_bytes}"
        )
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"locked {label} SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )


def _same_path(actual: str | Path, expected: Path, label: str) -> None:
    if Path(actual).resolve() != expected.resolve():
        raise ValueError(f"{label} does not match the frozen lock path")


def validate_p7_index_lock(
    catalog_path: str | Path,
    spec_path: str | Path,
    index_dir: str | Path,
    *,
    lock_path: str | Path = DEFAULT_INDEX_LOCK_PATH,
    project_root: str | Path = PROJECT_ROOT,
    expected_catalog_sha256: str = FROZEN_CATALOG_SHA256,
    expected_catalog_rows: int = FROZEN_CATALOG_ROWS,
) -> dict[str, Any]:
    """Hard-gate the tracked semantic index with only standard-library code."""

    project = Path(project_root).resolve()
    lock_file = Path(lock_path).resolve()
    if not lock_file.is_file():
        raise FileNotFoundError(f"tracked P7 semantic index lock is missing: {lock_file}")
    lock = _read_json(lock_file, "semantic-index lock")
    _exact_keys(
        lock,
        {
            "schema_version",
            "source",
            "model_spec",
            "catalog",
            "index",
            "asset_scope",
            "build_observation",
        },
        "semantic-index lock",
    )
    if lock["schema_version"] != INDEX_LOCK_SCHEMA_VERSION:
        raise ValueError("wrong semantic-index lock schema_version")

    source = _exact_keys(
        lock["source"], {"git_commit", "git_branch", "builder", "semantic"}, "source"
    )
    commit = source["git_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(
        char not in "0123456789abcdef" for char in commit
    ):
        raise ValueError("source.git_commit must be a full lowercase Git commit")
    if not isinstance(source["git_branch"], str) or not source["git_branch"]:
        raise ValueError("source.git_branch must be non-empty")
    for name in ("builder", "semantic"):
        entry = _exact_keys(source[name], {"path", "bytes", "sha256"}, f"source.{name}")
        path = _safe_path(project, entry["path"], f"source.{name}")
        _validate_locked_file(path, entry, f"source.{name}")

    model_spec = _exact_keys(
        lock["model_spec"],
        {"path", "raw_bytes", "raw_sha256", "canonical_sha256"},
        "model_spec",
    )
    locked_spec_path = _safe_path(project, model_spec["path"], "model_spec")
    _same_path(spec_path, locked_spec_path, "model spec path")
    spec_file_entry = {
        "bytes": model_spec["raw_bytes"],
        "sha256": model_spec["raw_sha256"],
    }
    _validate_locked_file(locked_spec_path, spec_file_entry, "model_spec")
    parsed_spec = _read_json(locked_spec_path, "model spec")
    canonical_spec_hash = hashlib.sha256(
        json.dumps(
            parsed_spec,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if canonical_spec_hash != _sha256_text(
        model_spec["canonical_sha256"], "model_spec.canonical_sha256"
    ):
        raise ValueError("model spec canonical SHA-256 mismatch")

    catalog = _exact_keys(
        lock["catalog"], {"path", "bytes", "sha256", "rows"}, "catalog"
    )
    locked_catalog_path = _safe_path(project, catalog["path"], "catalog")
    _same_path(catalog_path, locked_catalog_path, "catalog path")
    _validate_locked_file(locked_catalog_path, catalog, "catalog")
    locked_catalog_rows = _positive_int(catalog["rows"], "catalog.rows")
    if catalog["sha256"] != _sha256_text(
        expected_catalog_sha256, "expected_catalog_sha256"
    ):
        raise ValueError("locked catalog SHA-256 is not the frozen official catalog")
    frozen_expected_rows = _positive_int(
        expected_catalog_rows, "expected_catalog_rows"
    )
    if locked_catalog_rows != frozen_expected_rows:
        raise ValueError("locked catalog rows are not the frozen expected count")
    with locked_catalog_path.open("rb") as handle:
        actual_catalog_rows = 0
        for row in handle:
            if not row.strip():
                raise ValueError("locked catalog contains a blank row")
            actual_catalog_rows += 1
    if actual_catalog_rows != locked_catalog_rows:
        raise ValueError(
            f"locked catalog rows mismatch: {actual_catalog_rows} != {locked_catalog_rows}"
        )

    index = _exact_keys(
        lock["index"],
        {
            "directory",
            "manifest",
            "matrix",
            "ordered_asins",
            "canonical_documents_sha256",
        },
        "index",
    )
    locked_index_dir = _safe_path(project, index["directory"], "index.directory")
    _same_path(index_dir, locked_index_dir, "index directory")
    manifest_entry = _exact_keys(
        index["manifest"], {"path", "bytes", "sha256", "schema_version"}, "index.manifest"
    )
    matrix_entry = _exact_keys(
        index["matrix"], {"path", "bytes", "sha256", "dtype", "shape"}, "index.matrix"
    )
    asins_entry = _exact_keys(
        index["ordered_asins"],
        {"path", "bytes", "sha256", "count", "encoding", "line_ending"},
        "index.ordered_asins",
    )
    manifest_path = _safe_path(locked_index_dir, manifest_entry["path"], "index.manifest")
    matrix_path = _safe_path(locked_index_dir, matrix_entry["path"], "index.matrix")
    asins_path = _safe_path(locked_index_dir, asins_entry["path"], "index.ordered_asins")
    _validate_locked_file(manifest_path, manifest_entry, "index.manifest")
    _validate_locked_file(matrix_path, matrix_entry, "index.matrix")
    _validate_locked_file(asins_path, asins_entry, "index.ordered_asins")
    if manifest_entry["schema_version"] != "p7.semantic-index.v1":
        raise ValueError("locked manifest schema_version is invalid")
    if matrix_entry["dtype"] != "float32":
        raise ValueError("locked matrix dtype must be float32")
    shape = matrix_entry["shape"]
    if not isinstance(shape, list) or len(shape) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in shape
    ):
        raise ValueError("locked matrix shape is invalid")
    if asins_entry["count"] != shape[0]:
        raise ValueError("locked ordered-ASIN count differs from matrix rows")
    if asins_entry["encoding"] != "utf-8-lf" or asins_entry["line_ending"] != "LF":
        raise ValueError("locked ordered-ASIN encoding must be UTF-8 LF")
    _sha256_text(index["canonical_documents_sha256"], "canonical_documents_sha256")

    manifest = _read_json(manifest_path, "semantic-index manifest")
    if manifest.get("schema_version") != manifest_entry["schema_version"]:
        raise ValueError("lock and manifest schema_version differ")
    if manifest.get("model_spec_sha256") != canonical_spec_hash:
        raise ValueError("lock and manifest model-spec identity differ")
    if manifest.get("model_spec_serialization") != (
        "UTF-8 canonical JSON; object keys sorted; compact separators; "
        "ensure_ascii=false"
    ):
        raise ValueError("manifest model-spec serialization declaration is invalid")
    if manifest.get("catalog_sha256") != catalog["sha256"]:
        raise ValueError("lock and manifest catalog identity differ")
    if manifest.get("rows") != shape[0] or manifest.get("dimensions") != shape[1]:
        raise ValueError("lock and manifest index shape differ")
    manifest_matrix = manifest.get("matrix", {})
    for key in ("path", "bytes", "sha256", "dtype", "shape"):
        if manifest_matrix.get(key) != matrix_entry[key]:
            raise ValueError(f"lock and manifest matrix.{key} differ")
    if manifest_matrix.get("format") != "NumPy .npy":
        raise ValueError("manifest matrix format must be NumPy .npy")
    manifest_asins = manifest.get("ordered_asins", {})
    for key in ("path", "bytes", "sha256", "count", "encoding", "line_ending"):
        if manifest_asins.get(key) != asins_entry[key]:
            raise ValueError(f"lock and manifest ordered_asins.{key} differ")
    if (
        manifest.get("preprocessing", {}).get("canonical_documents_sha256")
        != index["canonical_documents_sha256"]
    ):
        raise ValueError("lock and manifest canonical-document identity differ")

    try:
        spec_license_relative = parsed_spec["model"]["license_notice"]
    except (KeyError, TypeError) as exc:
        raise ValueError("model spec is missing the bundled license path") from exc
    manifest_model = manifest.get("model")
    if not isinstance(manifest_model, Mapping):
        raise ValueError("semantic-index manifest model section is missing")
    license_entry = _exact_keys(
        manifest_model.get("license_notice"),
        {"path", "bytes", "sha256"},
        "manifest.model.license_notice",
    )
    if license_entry["path"] != spec_license_relative:
        raise ValueError("manifest license path differs from the frozen model spec")
    license_path = _safe_path(
        project, license_entry["path"], "manifest.model.license_notice"
    )
    _validate_locked_file(
        license_path, license_entry, "manifest.model.license_notice"
    )

    required_files = parsed_spec.get("required_files")
    if not isinstance(required_files, list) or not required_files:
        raise ValueError("model spec required_files are missing")
    model_required_file_bytes = 0
    for position, entry in enumerate(required_files):
        if not isinstance(entry, Mapping):
            raise ValueError(f"model spec required_files[{position}] is invalid")
        model_required_file_bytes += _positive_int(
            entry.get("bytes"), f"model spec required_files[{position}].bytes"
        )

    asset_scope = _exact_keys(
        lock["asset_scope"],
        {"required_asset_bytes", "required_asset_bytes_max"},
        "asset_scope",
    )
    asset_bytes = _positive_int(
        asset_scope["required_asset_bytes"], "asset_scope.required_asset_bytes"
    )
    asset_max = _positive_int(
        asset_scope["required_asset_bytes_max"], "asset_scope.required_asset_bytes_max"
    )
    try:
        spec_asset_max = parsed_spec["evaluation"]["resource_gates"][
            "required_asset_bytes_max"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("model spec is missing the frozen asset byte gate") from exc
    spec_asset_max = _positive_int(
        spec_asset_max, "model_spec.required_asset_bytes_max"
    )
    if asset_max != spec_asset_max:
        raise ValueError("lock asset byte maximum differs from the frozen model spec")
    if asset_bytes > asset_max:
        raise ValueError("locked semantic assets exceed the frozen byte gate")
    manifest_asset_scope = manifest.get("asset_byte_scope")
    if not isinstance(manifest_asset_scope, Mapping):
        raise ValueError("semantic-index manifest asset_byte_scope is missing")
    recomputed_excluding_manifest = (
        model_required_file_bytes
        + matrix_path.stat().st_size
        + asins_path.stat().st_size
        + license_path.stat().st_size
    )
    recomputed_asset_bytes = recomputed_excluding_manifest + manifest_path.stat().st_size
    manifest_components = {
        "model_required_files_bytes": model_required_file_bytes,
        "matrix_bytes": matrix_path.stat().st_size,
        "ordered_asins_bytes": asins_path.stat().st_size,
        "license_notice_bytes": license_path.stat().st_size,
        "required_asset_bytes_excluding_manifest": recomputed_excluding_manifest,
        "manifest_bytes": manifest_path.stat().st_size,
        "required_asset_bytes": recomputed_asset_bytes,
    }
    for key, expected in manifest_components.items():
        if manifest_asset_scope.get(key) != expected:
            raise ValueError(
                f"manifest asset_byte_scope.{key} does not match recomputed bytes"
            )
    if manifest_asset_scope.get("manifest_path") != manifest_entry["path"]:
        raise ValueError("manifest asset_byte_scope.manifest_path differs from lock")
    if asset_bytes != recomputed_asset_bytes:
        raise ValueError("lock required asset bytes do not match recomputed bytes")
    if manifest_asset_scope.get("required_asset_bytes") != asset_bytes:
        raise ValueError("lock and manifest required asset bytes differ")

    observation = _exact_keys(
        lock["build_observation"],
        {
            "wall_seconds",
            "rss_backend",
            "baseline_rss_bytes",
            "peak_rss_bytes",
            "peak_delta_from_baseline_bytes",
        },
        "build_observation",
    )
    if not isinstance(observation["wall_seconds"], (int, float)) or isinstance(
        observation["wall_seconds"], bool
    ) or observation["wall_seconds"] < 0:
        raise ValueError("build_observation.wall_seconds is invalid")
    if not isinstance(observation["rss_backend"], str) or not observation["rss_backend"]:
        raise ValueError("build_observation.rss_backend is invalid")
    for key in (
        "baseline_rss_bytes",
        "peak_rss_bytes",
        "peak_delta_from_baseline_bytes",
    ):
        if not isinstance(observation[key], int) or isinstance(observation[key], bool) or observation[key] < 0:
            raise ValueError(f"build_observation.{key} is invalid")
    manifest_resources = manifest.get("build_resources")
    if not isinstance(manifest_resources, Mapping):
        raise ValueError("semantic-index manifest build_resources are missing")
    for key in (
        "wall_seconds",
        "rss_backend",
        "baseline_rss_bytes",
        "peak_rss_bytes",
        "peak_delta_from_baseline_bytes",
    ):
        if observation[key] != manifest_resources.get(key):
            raise ValueError(f"lock and manifest build_resources.{key} differ")
    return _canonical_copy(lock)


def _dense_entry(value: Any) -> tuple[str, float]:
    if isinstance(value, Mapping):
        parent_asin = value.get("parent_asin")
        score = value.get("score")
    elif hasattr(value, "parent_asin") and hasattr(value, "score"):
        parent_asin = value.parent_asin
        score = value.score
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ) and len(value) == 2:
        parent_asin, score = value
    else:
        raise ValueError("dense result must be a two-item ASIN/score entry")
    if not isinstance(parent_asin, str) or not parent_asin.strip():
        raise ValueError("dense result parent_asin must be a non-empty string")
    if parent_asin != parent_asin.strip():
        raise ValueError("dense result parent_asin has outer whitespace")
    if isinstance(score, bool):
        raise ValueError("dense result score must be numeric")
    try:
        numeric_score = float(score)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dense result score must be numeric") from exc
    if not math.isfinite(numeric_score):
        raise ValueError("dense result score must be finite")
    return parent_asin, numeric_score


def _normalize_dense_route(values: Iterable[Any], top_k: int) -> list[dict[str, str]]:
    materialized = list(values)
    if len(materialized) > top_k:
        raise ValueError(f"dense result length {len(materialized)} exceeds {top_k}")
    route: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in materialized:
        parent_asin, score = _dense_entry(value)
        if parent_asin in seen:
            raise ValueError("dense result contains duplicate parent_asin values")
        seen.add(parent_asin)
        route.append({"parent_asin": parent_asin, "score": float(score).hex()})
    return route


class P7CaptureAgent(Agent):
    """One sparse Agent subclass used unchanged for both control and shadow roles."""

    def __init__(
        self,
        catalog_path: str | Path,
        role: str = C00,
        *,
        dense_search: DenseSearch | None = None,
        dense_close: Callable[[], None] | None = None,
        cold_initialization_seconds: float | None = None,
        required_asset_bytes: int = 0,
    ) -> None:
        if role not in {C00, S00}:
            raise ValueError(f"unknown P7 capture role: {role}")
        if role == C00 and dense_search is not None:
            raise ValueError("P7 C00 must not receive a dense search callable")
        if role == S00 and not callable(dense_search):
            raise ValueError("P7 S00 requires a dense search callable")
        if dense_close is not None and not callable(dense_close):
            raise ValueError("dense_close must be callable")
        if cold_initialization_seconds is not None and (
            isinstance(cold_initialization_seconds, bool)
            or not isinstance(cold_initialization_seconds, (int, float))
            or cold_initialization_seconds < 0
        ):
            raise ValueError("cold_initialization_seconds must be non-negative")
        if (
            not isinstance(required_asset_bytes, int)
            or isinstance(required_asset_bytes, bool)
            or required_asset_bytes < 0
        ):
            raise ValueError("required_asset_bytes must be a non-negative integer")
        self.p7_role = role
        self._dense_search = dense_search
        self._dense_close = dense_close
        self._thread_context = threading.local()
        self._state_ordinals: dict[int, int] = {}
        self._used_ordinals: set[int] = set()
        self._next_ordinal = 1
        self._route_records: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()
        self._response_records: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()
        self._integrity_errors: list[dict[str, Any]] = []
        self._dense_call_count = 0
        self._empty_query_count = 0
        self._semantic_exception_count = 0
        self._capture_exception_count = 0
        self._cold_initialization_seconds = (
            float(cold_initialization_seconds)
            if cold_initialization_seconds is not None
            else None
        )
        self._required_asset_bytes = required_asset_bytes
        self._closed = False
        super().__init__(
            catalog_path,
            llm_client=None,
            question_policy="fast",
            trace_sink=None,
            rerank_mode="off",
            retrieval_mode="coverage",
        )

    @staticmethod
    def _session_key(session_id: Any) -> str:
        return str(session_id)

    def _append_integrity_error(
        self, ordinal: int | None, turn: int | None, reason: str
    ) -> None:
        self._integrity_errors.append(
            {"ordinal": ordinal, "turn": turn, "reason": reason}
        )

    def reset(self, session_id: Any, user_profile: dict) -> None:
        key = self._session_key(session_id)
        with self._lock:
            if isinstance(session_id, int) and not isinstance(session_id, bool):
                ordinal = session_id
                if ordinal <= 0:
                    raise ValueError("explicit corpus ordinal must be positive")
                if ordinal in self._used_ordinals:
                    raise ValueError(f"duplicate corpus ordinal: {ordinal}")
            else:
                while self._next_ordinal in self._used_ordinals:
                    self._next_ordinal += 1
                ordinal = self._next_ordinal
            previous = self._sessions.get(key)
            super().reset(key, user_profile)
            if previous is not None:
                self._state_ordinals.pop(id(previous), None)
            self._used_ordinals.add(ordinal)
            self._next_ordinal = max(self._next_ordinal, ordinal + 1)
            self._state_ordinals[id(self._sessions[key])] = ordinal

    def drop_session(self, session_id: Any) -> None:
        key = self._session_key(session_id)
        with self._lock:
            state = self._sessions.get(key)
            super().drop_session(key)
            if state is not None:
                self._state_ordinals.pop(id(state), None)

    def _rank_candidates(self, state: SessionState) -> dict[str, list[str]]:
        rankings = super()._rank_candidates(state)
        context = getattr(self._thread_context, "active", None)
        if context is None or context.state_identity != id(state):
            return rankings
        if context.route_started:
            self._append_integrity_error(
                context.ordinal, context.turn, "multiple_rank_calls_in_one_response"
            )
            return rankings
        context.route_started = True
        started_ns = time.perf_counter_ns()
        actual_turn = len(state.messages)
        if actual_turn != context.turn:
            self._append_integrity_error(
                context.ordinal, context.turn, "declared_turn_does_not_match_visible_state"
            )
            return rankings

        query = " ".join(self._query_terms(state))
        dense_route: list[dict[str, str]] = []
        exception_class: str | None = None
        if not query:
            self._empty_query_count += 1
        elif self._dense_search is not None:
            self._dense_call_count += 1
            try:
                dense_route = _normalize_dense_route(
                    self._dense_search(query, DENSE_DEPTH), DENSE_DEPTH
                )
            except Exception as exc:  # shadow failure must never alter sparse output
                self._semantic_exception_count += 1
                exception_class = type(exc).__name__
                dense_route = []
        elapsed_ns = time.perf_counter_ns() - started_ns
        coordinate = (context.ordinal, context.turn)
        record: dict[str, Any] = {
            "ordinal": context.ordinal,
            "turn": context.turn,
            "query": query,
            "empty_query": not bool(query),
            "broad": list(rankings.get("broad", []))[:120],
            "strict": list(rankings.get("strict", []))[:80],
            "dense": dense_route,
            "query_search_ns": elapsed_ns,
        }
        if exception_class is not None:
            record["semantic_exception_class"] = exception_class
        if coordinate in self._route_records:
            self._append_integrity_error(
                context.ordinal, context.turn, "duplicate_route_coordinate"
            )
        elif len(self._route_records) >= MAX_CAPTURE_RECORDS:
            self._append_integrity_error(
                context.ordinal, context.turn, "route_capture_limit_exceeded"
            )
        else:
            try:
                self._route_records[coordinate] = _canonical_copy(record)
                context.route_captured = True
            except Exception as exc:
                self._capture_exception_count += 1
                self._append_integrity_error(
                    context.ordinal,
                    context.turn,
                    f"route_capture_{type(exc).__name__}",
                )
        return rankings

    def respond(
        self,
        session_id: Any,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        key = self._session_key(session_id)
        with self._lock:
            state = self._sessions.get(key)
            ordinal = self._state_ordinals.get(id(state)) if state is not None else None
            if state is None or ordinal is None:
                return super().respond(key, user_message, turn, top_k)
            if getattr(self._thread_context, "active", None) is not None:
                raise RuntimeError("nested P7 respond context is not allowed")
            context = _RespondContext(ordinal, turn, id(state))
            self._thread_context.active = context
            try:
                response = super().respond(key, user_message, turn, top_k)
                coordinate = (ordinal, turn)
                if not context.route_captured:
                    self._append_integrity_error(
                        ordinal, turn, "response_completed_without_route_capture"
                    )
                if coordinate in self._response_records:
                    self._append_integrity_error(
                        ordinal, turn, "duplicate_response_coordinate"
                    )
                elif len(self._response_records) >= MAX_CAPTURE_RECORDS:
                    self._append_integrity_error(
                        ordinal, turn, "response_capture_limit_exceeded"
                    )
                else:
                    try:
                        self._response_records[coordinate] = {
                            "ordinal": ordinal,
                            "turn": turn,
                            "response": copy.deepcopy(response),
                        }
                    except Exception as exc:
                        self._capture_exception_count += 1
                        self._append_integrity_error(
                            ordinal,
                            turn,
                            f"response_capture_{type(exc).__name__}",
                        )
                return response
            finally:
                del self._thread_context.active

    def route_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return _canonical_copy(list(self._route_records.values()))

    def response_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return _canonical_copy(list(self._response_records.values()))

    def semantic_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "role": self.p7_role,
                "route_record_count": len(self._route_records),
                "response_record_count": len(self._response_records),
                "dense_call_count": self._dense_call_count,
                "empty_query_count": self._empty_query_count,
                "semantic_exception_count": self._semantic_exception_count,
                "capture_exception_count": self._capture_exception_count,
                "integrity_error_count": len(self._integrity_errors),
                "cold_initialization_seconds": self._cold_initialization_seconds,
                "required_asset_bytes": self._required_asset_bytes,
            }

    def export_target_blind_capture(self) -> dict[str, Any]:
        with self._lock:
            routes = list(self._route_records.values())
            responses = list(self._response_records.values())
            stable_routes = [
                {
                    key: value
                    for key, value in record.items()
                    if key != "query_search_ns"
                }
                for record in routes
            ]
            dense_routes = [
                {
                    "ordinal": record["ordinal"],
                    "turn": record["turn"],
                    "query": record["query"],
                    "dense": record["dense"],
                }
                for record in routes
            ]
            payload = {
                "schema_version": SCHEMA_VERSION,
                "role": self.p7_role,
                "configuration": {
                    "retrieval_mode": "coverage",
                    "rerank_mode": "off",
                    "question_policy": "fast",
                    "dense_depth": DENSE_DEPTH,
                },
                "target_blind": True,
                "label_free": True,
                "route_records": routes,
                "response_records": responses,
                "integrity_errors": list(self._integrity_errors),
                "stats": self.semantic_stats(),
                "hashes": {
                    "routes_sha256": hashlib.sha256(
                        canonical_jsonl_bytes(stable_routes)
                    ).hexdigest(),
                    "dense_routes_sha256": hashlib.sha256(
                        canonical_jsonl_bytes(dense_routes)
                    ).hexdigest(),
                    "responses_sha256": hashlib.sha256(
                        canonical_jsonl_bytes(responses)
                    ).hexdigest(),
                },
            }
            return _canonical_copy(payload)

    def close(self) -> None:
        if self._closed:
            return
        error: Exception | None = None
        if self._dense_close is not None:
            try:
                self._dense_close()
            except Exception as exc:
                error = exc
        self.connection.close()
        self._closed = True
        if error is not None:
            raise error


def _normalize_role(role: str) -> str:
    aliases = {"C00": C00, "S00": S00, C00: C00, S00: S00}
    try:
        return aliases[role]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown P7 capture role: {role}") from exc


def create_p7_agent(
    role: str,
    catalog_path: str | Path,
    spec_path: str | Path,
    model_dir: str | Path,
    index_dir: str | Path,
    *,
    lock_path: str | Path = DEFAULT_INDEX_LOCK_PATH,
) -> P7CaptureAgent:
    """Construct C00 without optional imports, or a fully local S00 shadow."""

    normalized_role = _normalize_role(role)
    if normalized_role == C00:
        return P7CaptureAgent(catalog_path, C00)

    cold_started = time.perf_counter()
    lock = validate_p7_index_lock(
        catalog_path, spec_path, index_dir, lock_path=lock_path
    )
    semantic = importlib.import_module("starter.semantic")
    spec = semantic.load_semantic_spec(Path(spec_path))
    encoder = semantic.OfflineSemanticEncoder.from_frozen_assets(
        spec, Path(model_dir)
    )
    index = None
    try:
        index = semantic.SemanticIndex.load(
            spec,
            Path(index_dir),
            expected_catalog_sha256=lock["catalog"]["sha256"],
            numpy_module=encoder._np,
        )
        cold_initialization_seconds = time.perf_counter() - cold_started

        def dense_search(query: str, top_k: int) -> Iterable[Any]:
            return index.search_query(query, encoder, top_k=top_k)

        def dense_close() -> None:
            index.close()
            encoder.close()

        return P7CaptureAgent(
            catalog_path,
            S00,
            dense_search=dense_search,
            dense_close=dense_close,
            cold_initialization_seconds=cold_initialization_seconds,
            required_asset_bytes=lock["asset_scope"]["required_asset_bytes"],
        )
    except Exception:
        if index is not None:
            index.close()
        encoder.close()
        raise


__all__ = [
    "C00",
    "CONTROL_ID",
    "DEFAULT_INDEX_LOCK_PATH",
    "DENSE_DEPTH",
    "INDEX_LOCK_SCHEMA_VERSION",
    "MAX_CAPTURE_RECORDS",
    "P7CaptureAgent",
    "S00",
    "SCHEMA_VERSION",
    "SHADOW_ID",
    "canonical_jsonl_bytes",
    "create_p7_agent",
    "validate_p7_index_lock",
]
