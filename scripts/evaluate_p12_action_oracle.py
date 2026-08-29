"""Run the label-blind P12 action oracle on an unsealed proxy split.

The parent process owns evaluator labels and deterministic customer replies.  A
separate isolated worker receives only a projected profile and the visible
message history.  The worker is closed and its trace digest is verified before
this process joins any target to an action ranking.

This runner can never open the sealed confirmation split.  A limited run is a
smoke test only and is never eligible for a P12 promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
)
from scripts.p12_actions import (  # noqa: E402
    ACTION_IDS,
    ASK,
    CANDIDATE_RERANK,
    COMPACT_NEGATIVE_C50,
    FROZEN_SEMANTIC_RERANK,
    GUARDED_COMPACT_SLOT10,
    GUARDED_COMPACT_SLOT10_STRICT,
    HARD_CLAUSE_NOVEL_SLOT10,
    KEEP_P11,
    KEEP_R08,
    P11_EVIDENCE_NOVEL_SLOT10,
    RESULT_AWARE_REWRITE_RETRIEVE,
    TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10,
)
from scripts.p12_oracle_metrics import aggregate_action_oracle  # noqa: E402


SCHEMA_VERSION = "track4.p12-action-oracle-result.v1"
CONFIG_SCHEMA = "track4.p12-action-oracle.v1"
DEFAULT_CONFIG = Path("configs/p12_action_oracle_v1.json")
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "69da9c40aa6ec32448490e8c454508c3f1d1aa4fa45139d47f49b22e4d327bda"
)
ALLOWED_SPLITS = {
    "train_explore": {
        "path": "experiments/fast_track/proxy_v1/proxy_train_explore.jsonl",
        "rows": 2_000,
        "sha256": "2175696171c0d874fca4b9aa456ff5fd7d570f2184f59ade6781198f6443198e",
    },
    "calibration": {
        "path": "experiments/fast_track/proxy_v1/proxy_calibration.jsonl",
        "rows": 2_000,
        "sha256": "2ffd23092f0c30d33feef6d99b18262d03d856c6f5b6813e2e4e6e28dfeee8ed",
    },
    "selection": {
        "path": "experiments/fast_track/proxy_v1/proxy_selection.jsonl",
        "rows": 2_000,
        "sha256": "41ed8938ce3caa95df98155d10bb6cac01d5d279270ea8b6764979ca17a6e322",
    },
}
SEALED_CONFIRMATION_PATH = (
    "experiments/fast_track/proxy_v1/proxy_confirmation.sealed.jsonl"
)
EXPECTED_MANIFEST = {
    "path": "experiments/fast_track/proxy_v1/manifest.json",
    "sha256": "8058973426bbc76ea856a5c48a61e91ed9e35ae44988a21a6d7b2195e88a7193",
}
EXPECTED_CATALOG = {
    "path": "data/catalog.jsonl",
    "rows": 50_000,
    "sha256": "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67",
}
EXPECTED_WORKER = "scripts/p12_action_worker.py"
EXPECTED_OUTPUT_ROOT = "experiments/fast_track/action_oracle_v1"
EXPECTED_PARALLEL_WORKERS = 4
FAMILY2_NOVEL_SLOT10_ACTIONS = (
    P11_EVIDENCE_NOVEL_SLOT10,
    HARD_CLAUSE_NOVEL_SLOT10,
    TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10,
)
SAFE_PROFILE_KEYS = {
    "purchase_frequency",
    "average_prior_rating",
    "rating_style",
    "preference_tags",
    "summary",
}
FORBIDDEN_RPC_KEYS = {
    "sample_id",
    "ground_truth",
    "target",
    "target_id",
    "parent_asin",
    "scenario",
    "scenario_type",
    "taxonomy",
    "difficulty",
    "popularity",
    "source_weight",
    "evaluation_strata",
    "category_bucket",
    "difficulty_bucket",
    "intent_card",
    "behavior",
}
ASIN_TOKEN = re.compile(r"(?<![A-Z0-9])[A-Z0-9]{10}(?![A-Z0-9])")
ASIN_SHAPE_TOKEN = re.compile(
    r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE
)
VISIBLE_IDENTIFIER_REPLACEMENT = "[identifier omitted]"


class OracleRunError(RuntimeError):
    """Raised when the action-oracle protocol cannot be trusted."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lf(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _is_reparse_point(path: Path) -> bool:
    stat = path.lstat()
    attributes = getattr(stat, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse)


def _resolve_regular_file(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise OracleRunError("repository file path must be a non-empty relative path")
    lexical = REPO_ROOT.joinpath(relative)
    current = REPO_ROOT
    for part in Path(relative).parts:
        if part in {"", ".", ".."}:
            raise OracleRunError("repository file path contains an unsafe component")
        current = current / part
        if current.exists() and _is_reparse_point(current):
            raise OracleRunError(f"repository file path crosses a reparse point: {relative}")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise OracleRunError(f"repository file is missing or escaped: {relative}") from exc
    if not resolved.is_file():
        raise OracleRunError(f"repository path is not a regular file: {relative}")
    return resolved


def _resolve_repository_directory(relative: str, *, create: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise OracleRunError("repository directory must be a non-empty relative path")
    current = REPO_ROOT
    for part in Path(relative).parts:
        if part in {"", ".", ".."}:
            raise OracleRunError("repository directory contains an unsafe component")
        current = current / part
        if current.exists():
            if _is_reparse_point(current) or not current.is_dir():
                raise OracleRunError(f"repository directory is unsafe: {relative}")
        elif create:
            current.mkdir()
        else:
            raise OracleRunError(f"repository directory is missing: {relative}")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise OracleRunError(f"repository directory escaped: {relative}") from exc
    return resolved


def _resolve_asset_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise OracleRunError("asset path must be a non-empty relative path")
    current = root
    for part in Path(relative).parts:
        if part in {"", ".", ".."}:
            raise OracleRunError("asset path contains an unsafe component")
        current = current / part
        if not current.exists() or _is_reparse_point(current):
            raise OracleRunError(f"asset is missing or unsafe: {relative}")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise OracleRunError(f"asset escaped its frozen root: {relative}") from exc
    if not resolved.is_file():
        raise OracleRunError(f"asset is not a regular file: {relative}")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OracleRunError(f"invalid JSON file: {path.name}") from exc
    if not isinstance(value, dict):
        raise OracleRunError(f"JSON root must be an object: {path.name}")
    return value


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise OracleRunError(f"frozen configuration mismatch: {label}")


def load_frozen_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the tracked production config without touching a split."""

    expected = (REPO_ROOT / DEFAULT_CONFIG).resolve()
    requested = config_path if config_path.is_absolute() else REPO_ROOT / config_path
    if requested.resolve() != expected:
        raise OracleRunError("only the tracked P12 action-oracle config is accepted")
    config = _load_json(_resolve_regular_file(DEFAULT_CONFIG.as_posix()))
    if hashlib.sha256(_canonical_bytes(config)).hexdigest() != EXPECTED_CONFIG_CANONICAL_SHA256:
        raise OracleRunError("frozen P12 action-oracle config identity mismatch")
    _require_equal(config.get("schema_version"), CONFIG_SCHEMA, "schema_version")
    proxy = config.get("proxy")
    if not isinstance(proxy, dict):
        raise OracleRunError("proxy config must be an object")
    _require_equal(proxy.get("manifest_path"), EXPECTED_MANIFEST["path"], "manifest_path")
    _require_equal(
        proxy.get("manifest_sha256"), EXPECTED_MANIFEST["sha256"], "manifest_sha256"
    )
    _require_equal(proxy.get("allowed_splits"), ALLOWED_SPLITS, "allowed_splits")
    sealed = proxy.get("sealed_confirmation")
    if not isinstance(sealed, dict):
        raise OracleRunError("sealed confirmation declaration is missing")
    _require_equal(sealed.get("path"), SEALED_CONFIRMATION_PATH, "confirmation_path")
    _require_equal(sealed.get("generic_runner_access"), False, "confirmation_access")
    _require_equal(config.get("catalog"), EXPECTED_CATALOG, "catalog")
    _require_equal(config.get("actions", {}).get("ids"), list(ACTION_IDS), "action_ids")
    _require_equal(config.get("p11", {}).get("mode"), "active", "p11.mode")
    _require_equal(
        config.get("actions", {}).get("structured_existing_score_weight"),
        0.85,
        "actions.structured_existing_score_weight",
    )
    _require_equal(
        config.get("actions", {}).get("structured_numeric_budget_weight"),
        0.15,
        "actions.structured_numeric_budget_weight",
    )
    _require_equal(
        config.get("semantic", {}).get("sparse_prior_weight"),
        0.65,
        "semantic.sparse_prior_weight",
    )
    _require_equal(
        config.get("semantic", {}).get("cosine_weight"),
        0.35,
        "semantic.cosine_weight",
    )
    _require_equal(
        config.get("semantic", {}).get("full_catalog_search_allowed"),
        False,
        "semantic.full_catalog_search_allowed",
    )
    _require_equal(config.get("runtime", {}).get("worker_path"), EXPECTED_WORKER, "worker_path")
    _require_equal(
        config.get("runtime", {}).get("output_root"),
        EXPECTED_OUTPUT_ROOT,
        "output_root",
    )
    _require_equal(
        config.get("runtime", {}).get("parallel_workers"),
        EXPECTED_PARALLEL_WORKERS,
        "parallel_workers",
    )
    return config


def select_split(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Return an exact allowlisted split before any data-path operation occurs."""

    if split not in ALLOWED_SPLITS:
        raise OracleRunError(f"split is not available to the generic runner: {split}")
    spec = config.get("proxy", {}).get("allowed_splits", {}).get(split)
    if spec != ALLOWED_SPLITS[split]:
        raise OracleRunError(f"split declaration drifted: {split}")
    if spec.get("path") == SEALED_CONFIRMATION_PATH:
        raise OracleRunError("sealed confirmation cannot be selected")
    return dict(spec)


def _verified_jsonl(relative: str, expected_sha: str, expected_rows: int) -> list[dict[str, Any]]:
    path = _resolve_regular_file(relative)
    if _sha256(path) != expected_sha:
        raise OracleRunError(f"SHA-256 mismatch: {relative}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise OracleRunError(f"blank JSONL row: {relative}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise OracleRunError(f"JSONL row is not an object: {relative}:{line_number}")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OracleRunError(f"invalid JSONL file: {relative}") from exc
    if len(rows) != expected_rows:
        raise OracleRunError(f"row-count mismatch: {relative}")
    if _sha256(path) != expected_sha:
        raise OracleRunError(f"input changed while it was parsed: {relative}")
    return rows


def validate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    proxy = config["proxy"]
    path = _resolve_regular_file(str(proxy["manifest_path"]))
    if _sha256(path) != proxy["manifest_sha256"]:
        raise OracleRunError("proxy manifest SHA-256 mismatch")
    manifest = _load_json(path)
    if manifest.get("schema_version") != "track4.amazon-validation-proxy.v1":
        raise OracleRunError("proxy manifest schema mismatch")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise OracleRunError("proxy manifest splits are missing")
    for name, expected in ALLOWED_SPLITS.items():
        observed = splits.get(name)
        if not isinstance(observed, dict):
            raise OracleRunError(f"proxy manifest split is missing: {name}")
        if (
            observed.get("filename") != Path(expected["path"]).name
            or observed.get("rows") != expected["rows"]
            or observed.get("sha256") != expected["sha256"]
            or observed.get("sealed") is not False
        ):
            raise OracleRunError(f"proxy manifest split mismatch: {name}")
    confirmation = splits.get("confirmation")
    if not isinstance(confirmation, dict) or confirmation.get("sealed") is not True:
        raise OracleRunError("proxy confirmation is not declared sealed")
    return manifest


def load_catalog(config: Mapping[str, Any]) -> tuple[set[str], dict[str, list[str]], dict[str, dict[str, Any]]]:
    spec = config["catalog"]
    rows = _verified_jsonl(str(spec["path"]), str(spec["sha256"]), int(spec["rows"]))
    identifiers: set[str] = set()
    categories: dict[str, list[str]] = {}
    products: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get("parent_asin", "")).strip()
        if not identifier or identifier in identifiers:
            raise OracleRunError("catalog contains a blank or duplicate parent_asin")
        identifiers.add(identifier)
        categories[identifier] = [str(value) for value in row.get("categories") or []]
        products[identifier] = row
    return identifiers, categories, products


def project_profile(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != SAFE_PROFILE_KEYS:
        raise OracleRunError("proxy profile does not match the safe projection")
    tags = raw.get("preference_tags")
    if (
        not isinstance(raw.get("purchase_frequency"), str)
        or raw.get("average_prior_rating") is not None
        or raw.get("rating_style") != "unknown"
        or not isinstance(tags, list)
        or len(tags) > 3
        or any(not isinstance(value, str) or len(value) > 120 for value in tags)
        or not isinstance(raw.get("summary"), str)
        or len(str(raw["summary"])) > 512
    ):
        raise OracleRunError("proxy profile contains an unsafe value")
    return {key: raw[key] for key in sorted(SAFE_PROFILE_KEYS)}


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            keys.update(_walk_keys(item))
    return keys


def assert_blind_rpc(
    payload: Mapping[str, Any],
    *,
    current_target: str,
    sample_id: str,
    catalog_ids: set[str],
) -> None:
    forbidden = _walk_keys(payload) & FORBIDDEN_RPC_KEYS
    if forbidden:
        raise OracleRunError(f"label-shaped key in worker request: {sorted(forbidden)}")
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    serialized_upper = serialized.upper()
    if current_target and current_target.upper() in serialized_upper:
        raise OracleRunError("current target leaked into worker request")
    if sample_id and sample_id in serialized:
        raise OracleRunError("sample_id leaked into worker request")
    leaked_ids = set(ASIN_TOKEN.findall(serialized_upper)) & catalog_ids
    if leaked_ids:
        raise OracleRunError("catalog identifier leaked into worker request")


def sanitize_worker_visible_message(value: str) -> tuple[str, int]:
    """Redact non-catalog ASIN-shaped metadata artifacts before worker RPC.

    The caller must run ``assert_blind_rpc`` on the unmodified payload first,
    so this cannot conceal a target or frozen-catalog identifier leak.
    """

    matches = list(ASIN_SHAPE_TOKEN.finditer(value))
    if not matches:
        return value, 0
    return ASIN_SHAPE_TOKEN.sub(VISIBLE_IDENTIFIER_REPLACEMENT, value), len(matches)


def assert_identifier_free_artifact(
    artifact: Mapping[str, Any], catalog_ids: set[str]
) -> None:
    """Reject any row identifier key or catalog identifier in an aggregate."""

    forbidden = _walk_keys(artifact) & {
        "ordinal",
        "sample_id",
        "ground_truth",
        "parent_asin",
        "target_id",
    }
    if forbidden:
        raise OracleRunError(f"aggregate artifact contains identifier keys: {sorted(forbidden)}")
    serialized = json.dumps(artifact, sort_keys=True, ensure_ascii=False)
    if set(ASIN_TOKEN.findall(serialized)) & catalog_ids:
        raise OracleRunError("aggregate artifact contains a catalog identifier")


def _offline_environment() -> dict[str, str]:
    keep = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH")
    environment = {key: os.environ[key] for key in keep if key in os.environ}
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )
    return environment


@dataclass(slots=True)
class WorkerReceipt:
    trace_sha256: str
    record_count: int
    worker_summary: dict[str, Any]


@dataclass(slots=True)
class ShardResult:
    shard_index: int
    global_start: int
    samples: Sequence[Mapping[str, Any]]
    trace_path: Path
    receipt: WorkerReceipt
    ledger: list[dict[str, Any]]


def _raise_on_worker_error(
    response: Mapping[str, Any], request_id: int, operation: str
) -> None:
    if response.get("kind") != "error":
        return
    if set(response) != {"kind", "request_id", "error_class"}:
        raise OracleRunError("worker error reply shape mismatch")
    error_class = response.get("error_class")
    if not isinstance(error_class, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_class
    ):
        raise OracleRunError("worker error class is invalid")
    if response.get("request_id") is None:
        raise OracleRunError(
            f"worker {operation} rejected request before identity with {error_class}"
        )
    if response.get("request_id") != request_id:
        raise OracleRunError("worker error request identity mismatch")
    raise OracleRunError(
        f"worker {operation} failed closed with {error_class}"
    )


class WorkerClient:
    """Strict JSONL client for the isolated target-blind worker."""

    def __init__(self, config: Mapping[str, Any], trace_path: Path, nonce: str):
        runtime = config["runtime"]
        worker = _resolve_regular_file(str(runtime["worker_path"]))
        catalog = config["catalog"]
        p11 = config["p11"]
        semantic = config["semantic"]
        args = [
            sys.executable,
            "-I",
            str(worker),
            "--nonce",
            nonce,
            "--catalog",
            str(_resolve_regular_file(str(catalog["path"]))),
            "--catalog-rows",
            str(catalog["rows"]),
            "--catalog-sha256",
            str(catalog["sha256"]),
            "--sidecar",
            str(_resolve_regular_file(str(p11["sidecar_path"]))),
            "--sidecar-bytes",
            str(p11["sidecar_bytes"]),
            "--sidecar-sha256",
            str(p11["sidecar_sha256"]),
            "--semantic-spec",
            str(_resolve_regular_file(str(semantic["spec_path"]))),
            "--semantic-lock",
            str(_resolve_regular_file(str(semantic["lock_path"]))),
            "--semantic-model-dir",
            str(_resolve_repository_directory(str(semantic["model_dir"]))),
            "--semantic-index-dir",
            str(_resolve_repository_directory(str(semantic["index_dir"]))),
            "--trace-output",
            str(trace_path),
        ]
        self._timeout = float(runtime["request_timeout_seconds"])
        self._exit_timeout = float(runtime["process_exit_timeout_seconds"])
        self._queue: queue.Queue[object] = queue.Queue()
        self._stderr: list[str] = []
        self._request_id = 0
        self._closed = False
        self._process = subprocess.Popen(
            args,
            cwd=REPO_ROOT,
            env=_offline_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            ready = self._receive()
            if ready != {"kind": "ready", "nonce": nonce}:
                raise OracleRunError("worker ready handshake mismatch")
        except Exception:
            self.abort()
            raise

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        try:
            for line in self._process.stdout:
                self._queue.put(line)
        except Exception as exc:  # pragma: no cover - platform pipe failure
            self._queue.put(exc)
        finally:
            self._queue.put(None)

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        self._stderr.extend(line.rstrip("\r\n") for line in self._process.stderr)

    def _receive(self) -> dict[str, Any]:
        try:
            item = self._queue.get(timeout=self._timeout)
        except queue.Empty as exc:
            raise OracleRunError("worker request timed out") from exc
        if item is None:
            raise OracleRunError(f"worker stdout closed early: {' | '.join(self._stderr[-8:])}")
        if isinstance(item, BaseException):
            raise OracleRunError("worker stdout reader failed") from item
        try:
            value = json.loads(str(item))
        except json.JSONDecodeError as exc:
            raise OracleRunError("worker emitted non-JSON stdout") from exc
        if not isinstance(value, dict):
            raise OracleRunError("worker response is not an object")
        return value

    def _send(self, operation: str, fields: Mapping[str, Any]) -> tuple[int, Any]:
        if self._closed:
            raise OracleRunError("worker client is closed")
        self._request_id += 1
        request_id = self._request_id
        payload = {"operation": operation, "request_id": request_id, **fields}
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(_canonical_bytes(payload).decode("utf-8"))
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise OracleRunError("worker stdin closed early") from exc
        response = self._receive()
        _raise_on_worker_error(response, request_id, operation)
        return request_id, response

    def reset(self, ordinal: int, user_profile: Mapping[str, Any]) -> None:
        request_id, response = self._send(
            "reset", {"ordinal": ordinal, "user_profile": dict(user_profile)}
        )
        expected = {"kind": "reply", "request_id": request_id, "value": None}
        if response != expected:
            raise OracleRunError("worker reset reply mismatch")

    def respond(self, ordinal: int, user_message: str, turn: int) -> str | None:
        request_id, response = self._send(
            "respond",
            {
                "ordinal": ordinal,
                "user_message": user_message,
                "turn": turn,
                "top_k": 10,
            },
        )
        if set(response) != {"kind", "request_id", "value"}:
            raise OracleRunError("worker respond reply shape mismatch")
        if response.get("kind") != "reply" or response.get("request_id") != request_id:
            raise OracleRunError("worker respond reply identity mismatch")
        value = response.get("value")
        if not isinstance(value, dict) or set(value) != {"ask_attribute"}:
            raise OracleRunError("worker respond value shape mismatch")
        ask = value["ask_attribute"]
        if ask is not None and not isinstance(ask, str):
            raise OracleRunError("worker ask_attribute is invalid")
        return ask

    def drop(self, ordinal: int) -> None:
        request_id, response = self._send("drop", {"ordinal": ordinal})
        expected = {"kind": "reply", "request_id": request_id, "value": None}
        if response != expected:
            raise OracleRunError("worker drop reply mismatch")

    def finalize(self) -> WorkerReceipt:
        request_id, response = self._send("finalize", {})
        if set(response) != {
            "kind",
            "request_id",
            "trace_sha256",
            "record_count",
            "worker_summary",
        }:
            raise OracleRunError("worker receipt shape mismatch")
        if response.get("kind") != "receipt" or response.get("request_id") != request_id:
            raise OracleRunError("worker receipt identity mismatch")
        digest = response.get("trace_sha256")
        count = response.get("record_count")
        summary = response.get("worker_summary")
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or not isinstance(summary, dict)
        ):
            raise OracleRunError("worker receipt fields are invalid")
        self._close_cleanly()
        return WorkerReceipt(digest, count, summary)

    def _close_cleanly(self) -> None:
        if self._closed:
            return
        self._closed = True
        assert self._process.stdin is not None
        self._process.stdin.close()
        try:
            return_code = self._process.wait(timeout=self._exit_timeout)
        except subprocess.TimeoutExpired as exc:
            self._process.kill()
            self._process.wait()
            raise OracleRunError("worker did not exit after finalize") from exc
        self._stdout_thread.join(timeout=5)
        self._stderr_thread.join(timeout=5)
        if return_code != 0:
            raise OracleRunError(
                f"worker exited with {return_code}: {' | '.join(self._stderr[-8:])}"
            )
        try:
            trailing = self._queue.get_nowait()
        except queue.Empty:
            trailing = None
        if trailing not in (None,):
            raise OracleRunError("worker emitted output after its final receipt")

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait(timeout=10)


def _read_trace(path: Path, receipt: WorkerReceipt, expected_records: int) -> list[dict[str, Any]]:
    if not path.exists() or _is_reparse_point(path) or not path.is_file():
        raise OracleRunError("worker trace is missing or unsafe")
    if _sha256(path) != receipt.trace_sha256:
        raise OracleRunError("worker trace SHA-256 does not match its closed receipt")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.strip():
                    raise OracleRunError("worker trace contains a blank row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise OracleRunError("worker trace row is not an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OracleRunError("worker trace is invalid JSONL") from exc
    if len(rows) != receipt.record_count or len(rows) != expected_records:
        raise OracleRunError("worker trace record count mismatch")
    if _sha256(path) != receipt.trace_sha256:
        raise OracleRunError("worker trace changed while it was parsed")
    return rows


def _validate_ranking(value: object, catalog_ids: set[str], label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > 100
        or any(not isinstance(item, str) or item not in catalog_ids for item in value)
        or len(value) != len(set(value))
    ):
        raise OracleRunError(f"invalid worker ranking: {label}")
    return value


def validate_trace(
    rows: Sequence[Mapping[str, Any]], sample_count: int, catalog_ids: set[str]
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    expected_keys = {"ordinal", "turn", "actions", "candidate_pools"}
    for raw in rows:
        if set(raw) != expected_keys:
            raise OracleRunError("worker trace row shape mismatch")
        ordinal = raw.get("ordinal")
        turn = raw.get("turn")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or not 1 <= ordinal <= sample_count
            or not isinstance(turn, int)
            or isinstance(turn, bool)
            or not 1 <= turn <= 10
        ):
            raise OracleRunError("worker trace coordinate is invalid")
        actions = raw.get("actions")
        pools = raw.get("candidate_pools")
        if not isinstance(actions, dict) or set(actions) != set(ACTION_IDS):
            raise OracleRunError("worker trace action set mismatch")
        if not isinstance(pools, dict) or set(pools) != {"c20", "c50", "c100"}:
            raise OracleRunError("worker trace candidate-pool set mismatch")
        clean_actions = {
            action: _validate_ranking(actions[action], catalog_ids, action)
            for action in ACTION_IDS
        }
        clean_pools = {
            name: _validate_ranking(pools[name], catalog_ids, name)
            for name in ("c20", "c50", "c100")
        }
        r08 = clean_actions[KEEP_R08]
        p11 = clean_actions[KEEP_P11]
        if clean_actions[ASK] != p11:
            raise OracleRunError("ASK trace must reuse observed KEEP_P11 recommendations")
        if any(len(clean_actions[action]) > 10 for action in ACTION_IDS):
            raise OracleRunError("worker action ranking exceeds Top10")
        if (
            len(clean_pools["c20"]) > 20
            or len(clean_pools["c50"]) > 50
            or len(clean_pools["c100"]) > 100
        ):
            raise OracleRunError("worker candidate pool exceeds its fixed cutoff")
        if r08[:10] != clean_pools["c20"][:10]:
            raise OracleRunError("C10 is not the KEEP_R08 candidate prefix")
        if clean_pools["c20"] != clean_pools["c50"][: len(clean_pools["c20"])]:
            raise OracleRunError("C20 is not a prefix of C50")
        if clean_pools["c50"] != clean_pools["c100"][: len(clean_pools["c50"])]:
            raise OracleRunError("C50 is not a prefix of C100")
        if len(r08) != len(p11) or set(r08[:10]) != set(p11[:10]):
            raise OracleRunError("P11 changed frozen R08 Top10 membership")
        for action in (
            CANDIDATE_RERANK,
            FROZEN_SEMANTIC_RERANK,
            COMPACT_NEGATIVE_C50,
            GUARDED_COMPACT_SLOT10,
            GUARDED_COMPACT_SLOT10_STRICT,
            *FAMILY2_NOVEL_SLOT10_ACTIONS,
        ):
            if (
                len(clean_actions[action]) != min(10, len(clean_pools["c50"]))
                or not set(clean_actions[action]).issubset(set(clean_pools["c50"]))
            ):
                raise OracleRunError(f"{action} Top10 is not drawn from exact C50")
        guarded = clean_actions[GUARDED_COMPACT_SLOT10]
        if guarded != p11 and (
            len(p11) != 10
            or guarded[:9] != p11[:9]
            or len(set(guarded) ^ set(p11)) != 2
        ):
            raise OracleRunError("guarded compact action violates its single-slot guard")
        strict_guarded = clean_actions[GUARDED_COMPACT_SLOT10_STRICT]
        if strict_guarded != p11 and (
            len(p11) != 10
            or len(clean_pools["c50"]) <= 10
            or strict_guarded[:9] != p11[:9]
            or strict_guarded[9] != clean_pools["c50"][10]
            or len(set(strict_guarded) ^ set(p11)) != 2
        ):
            raise OracleRunError(
                "strict guarded compact action violates its adjacent single-slot guard"
            )
        structured_top10 = frozenset(clean_actions[CANDIDATE_RERANK])
        semantic_top10 = frozenset(clean_actions[FROZEN_SEMANTIC_RERANK])
        p11_tail_c50 = frozenset(clean_pools["c50"][10:])
        for action in FAMILY2_NOVEL_SLOT10_ACTIONS:
            novel = clean_actions[action]
            if novel == p11:
                continue
            added = set(novel) - set(p11)
            if (
                len(p11) != 10
                or novel[:9] != p11[:9]
                or len(set(novel) ^ set(p11)) != 2
                or len(added) != 1
            ):
                raise OracleRunError(
                    f"{action} violates its Top1-9-preserving single-slot guard"
                )
            added_identifier = next(iter(added))
            if added_identifier not in p11_tail_c50:
                raise OracleRunError(
                    f"{action} challenger is not drawn from P11 ranks 11-to-50"
                )
            if (
                added_identifier in structured_top10
                or added_identifier in semantic_top10
            ):
                raise OracleRunError(
                    f"{action} challenger is not novel to structured and semantic Top10"
                )
        row = {
            "ordinal": ordinal,
            "turn": turn,
            "actions": clean_actions,
            "candidate_pools": clean_pools,
        }
        grouped.setdefault(ordinal, []).append(row)
    if set(grouped) != set(range(1, sample_count + 1)):
        raise OracleRunError("worker trace ordinals are incomplete")
    for ordinal, turns in grouped.items():
        turns.sort(key=lambda value: value["turn"])
        if [value["turn"] for value in turns] != list(range(1, 11)):
            raise OracleRunError(f"worker trace turns are incomplete: {ordinal}")
    return grouped


def _outcome(target: str, eligible_from: int, turns: Sequence[Mapping[str, Any]], action: str) -> dict[str, Any]:
    for row in turns:
        turn = int(row["turn"])
        if turn < eligible_from:
            continue
        top10 = list(row["actions"][action])[:10]
        if target in top10:
            rank = top10.index(target) + 1
            return {
                "hit": True,
                "first_hit_turn": turn,
                "first_rank": rank,
                "best_rank": rank,
                "reciprocal_rank": 1.0 / rank,
            }
    return {
        "hit": False,
        "first_hit_turn": None,
        "first_rank": None,
        "best_rank": None,
        "reciprocal_rank": 0.0,
    }


def _candidate_recall(
    target: str, eligible_from: int, turns: Sequence[Mapping[str, Any]], cutoff: int
) -> bool:
    for row in turns:
        if int(row["turn"]) < eligible_from:
            continue
        pool = (
            list(row["actions"][KEEP_R08])[:10]
            if cutoff == 10
            else list(row["candidate_pools"][f"c{cutoff}"])
        )
        if target in pool[:cutoff]:
            return True
    return False


def join_labels_after_close(
    samples: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    trace: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    include_counts: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(samples) != len(ledger) or len(trace) != len(samples):
        raise OracleRunError("parent ledger and closed worker trace do not align")
    joined: list[dict[str, Any]] = []
    recall_counts = {10: 0, 20: 0, 50: 0, 100: 0}
    for ordinal, (sample, meta) in enumerate(zip(samples, ledger, strict=True), start=1):
        target = str(sample.get("ground_truth", {}).get("parent_asin", ""))
        if not target or meta.get("target_id") != target:
            raise OracleRunError("parent target ledger mismatch")
        turns = trace[ordinal]
        eligible_from = int(meta["eligible_from_turn"])
        actions = {
            action: _outcome(target, eligible_from, turns, action)
            for action in ACTION_IDS
        }
        for cutoff in recall_counts:
            recall_counts[cutoff] += int(
                _candidate_recall(target, eligible_from, turns, cutoff)
            )
        joined.append(
            {
                "ordinal": ordinal,
                "target_id": target,
                "scenario": meta["scenario"],
                "taxonomy": meta["taxonomy"],
                "difficulty": meta["difficulty"],
                "popularity": meta["popularity"],
                "source_weight": meta["source_weight"],
                "actions": actions,
            }
        )
    recalls = {
        f"recall_at_{cutoff}": round(count / len(samples), 6)
        for cutoff, count in recall_counts.items()
    }
    if include_counts:
        recalls = {
            "sample_count": len(samples),
            "cutoffs": {
                str(cutoff): {
                    "hit_count": count,
                    "rate": round(count / len(samples), 6),
                }
                for cutoff, count in recall_counts.items()
            },
        }
    return joined, recalls


def _eligible_from_turn(effective_sample: Mapping[str, Any]) -> int:
    if effective_sample.get("scenario_type") != "intent_override":
        return 1
    override = effective_sample.get("behavior", {}).get("override") or {}
    value = override.get("turn")
    if not isinstance(value, int) or isinstance(value, bool) or not 2 <= value <= 10:
        raise OracleRunError("intent-override turn is invalid")
    return value


def replay_blind_trajectories(
    worker: WorkerClient,
    samples: Sequence[Mapping[str, Any]],
    categories: Mapping[str, list[str]],
    products: Mapping[str, dict[str, Any]],
    catalog_ids: set[str],
    abort_event: threading.Event | None = None,
) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for ordinal, sample in enumerate(samples, start=1):
        if abort_event is not None and abort_event.is_set():
            raise OracleRunError("parallel shard aborted after a peer failure")
        sample_id = str(sample.get("sample_id", ""))
        target = str(sample.get("ground_truth", {}).get("parent_asin", ""))
        if target not in products or not sample_id:
            raise OracleRunError("proxy sample has an invalid target or sample_id")
        card, behavior = materialize_hidden_fields(dict(sample), products)  # type: ignore[arg-type]
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = effective["scenario_type"] != "intent_override"
        message = initial_message(
            effective, coarse_category(categories.get(target, [])), disclosed
        )
        sanitized_message_count = 0
        sanitized_token_count = 0
        profile = project_profile(sample.get("user_profile"))
        reset_payload = {
            "operation": "reset",
            "ordinal": ordinal,
            "user_profile": profile,
        }
        assert_blind_rpc(
            reset_payload,
            current_target=target,
            sample_id=sample_id,
            catalog_ids=catalog_ids,
        )
        worker.reset(ordinal, profile)
        for turn in range(1, 11):
            if abort_event is not None and abort_event.is_set():
                raise OracleRunError("parallel shard aborted after a peer failure")
            raw_respond_payload = {
                "operation": "respond",
                "ordinal": ordinal,
                "user_message": message,
                "turn": turn,
                "top_k": 10,
            }
            assert_blind_rpc(
                raw_respond_payload,
                current_target=target,
                sample_id=sample_id,
                catalog_ids=catalog_ids,
            )
            visible_message, redacted_tokens = sanitize_worker_visible_message(message)
            sanitized_message_count += int(redacted_tokens > 0)
            sanitized_token_count += redacted_tokens
            respond_payload = {
                **raw_respond_payload,
                "user_message": visible_message,
            }
            assert_blind_rpc(
                respond_payload,
                current_target=target,
                sample_id=sample_id,
                catalog_ids=catalog_ids,
            )
            ask_attribute = worker.respond(ordinal, visible_message, turn)
            if turn < 10:
                override = effective.get("behavior", {}).get("override") or {}
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        disclosed.add(new_value)
                    message = str(
                        override.get(
                            "message", "Actually, please ignore my earlier preference."
                        )
                    )
                else:
                    message, boundary_used = customer_reply(
                        effective, ask_attribute, disclosed, boundary_used
                    )
        worker.drop(ordinal)
        strata = sample.get("evaluation_strata")
        taxonomy = sample.get("taxonomy")
        if not isinstance(strata, Mapping) or not isinstance(taxonomy, Mapping):
            raise OracleRunError("proxy evaluation strata are missing")
        ledger.append(
            {
                "target_id": target,
                "eligible_from_turn": _eligible_from_turn(effective),
                "scenario": str(sample.get("scenario_type")),
                "taxonomy": str(taxonomy.get("group")),
                "difficulty": str(sample.get("difficulty_bucket")),
                "popularity": str(strata.get("popularity")),
                "source_weight": float(strata.get("source_weight")),
                "sanitized_visible_message_count": sanitized_message_count,
                "sanitized_visible_token_count": sanitized_token_count,
            }
        )
    return ledger


def balanced_contiguous_chunks(
    samples: Sequence[Mapping[str, Any]], requested_workers: int
) -> list[tuple[int, int, Sequence[Mapping[str, Any]]]]:
    """Return (shard index, one-based global start, contiguous rows)."""

    if not samples or requested_workers < 1:
        raise OracleRunError("parallel shard request is invalid")
    worker_count = min(requested_workers, len(samples))
    base, extra = divmod(len(samples), worker_count)
    chunks = []
    offset = 0
    for shard_index in range(worker_count):
        size = base + int(shard_index < extra)
        chunks.append((shard_index, offset + 1, samples[offset : offset + size]))
        offset += size
    return chunks


def _run_blind_shards(
    config: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    categories: Mapping[str, list[str]],
    products: Mapping[str, dict[str, Any]],
    catalog_ids: set[str],
    trace_paths: Sequence[Path],
    base_nonce: str,
) -> list[ShardResult]:
    chunks = balanced_contiguous_chunks(
        samples, int(config["runtime"]["parallel_workers"])
    )
    if len(trace_paths) != len(chunks):
        raise OracleRunError("parallel trace registry does not match shard count")
    stop = threading.Event()
    lock = threading.Lock()
    active: dict[int, WorkerClient] = {}

    def execute(
        shard_index: int,
        global_start: int,
        shard_samples: Sequence[Mapping[str, Any]],
    ) -> ShardResult:
        nonce = hashlib.sha256(
            f"{base_nonce}\0shard\0{shard_index}".encode("utf-8")
        ).hexdigest()[:32]
        client = WorkerClient(config, trace_paths[shard_index], nonce)
        with lock:
            active[shard_index] = client
        try:
            ledger = replay_blind_trajectories(
                client,
                shard_samples,
                categories,
                products,
                catalog_ids,
                stop,
            )
            if stop.is_set():
                raise OracleRunError("parallel shard aborted after a peer failure")
            receipt = client.finalize()  # also waits for a clean process exit
            return ShardResult(
                shard_index,
                global_start,
                shard_samples,
                trace_paths[shard_index],
                receipt,
                ledger,
            )
        finally:
            client.abort()
            with lock:
                active.pop(shard_index, None)

    results: list[ShardResult] = []
    executor = ThreadPoolExecutor(max_workers=len(chunks))
    futures = {
        executor.submit(execute, index, start, chunk): index
        for index, start, chunk in chunks
    }
    try:
        for future in as_completed(futures):
            results.append(future.result())
    except Exception:
        stop.set()
        with lock:
            clients = list(active.values())
        for client in clients:
            client.abort()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return sorted(results, key=lambda item: item.shard_index)


def _merge_worker_summaries(shards: Sequence[ShardResult]) -> dict[str, Any]:
    gate_keys = (
        "network_attempt_count",
        "semantic_failure_count",
        "rewrite_failure_count",
        "family2_score_failure_count",
        "full_catalog_search_calls",
        "p11_invariant_failure_count",
    )
    summaries = [shard.receipt.worker_summary for shard in shards]

    def part(summary: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        value = summary.get(key)
        if not isinstance(value, Mapping):
            raise OracleRunError(f"worker {key} summary is invalid")
        return value

    def count(summary: Mapping[str, Any], section: str | None, key: str) -> int:
        source = summary if section is None else part(summary, section)
        value = source.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise OracleRunError(f"worker summary count is invalid: {key}")
        return value

    first = summaries[0]
    expected_identity = (
        part(first, "trajectory").get("fixed_turns"),
        part(first, "trajectory").get("top_k"),
        part(first, "actions").get("ids"),
        part(first, "semantic").get("mode"),
    )
    for shard, summary in zip(shards, summaries, strict=True):
        identity = (
            part(summary, "trajectory").get("fixed_turns"),
            part(summary, "trajectory").get("top_k"),
            part(summary, "actions").get("ids"),
            part(summary, "semantic").get("mode"),
        )
        frozen_identity = (10, 10, list(ACTION_IDS), "candidate_only_c50")
        if identity != expected_identity or identity != frozen_identity:
            raise OracleRunError("worker summary identity mismatch across shards")
        if count(summary, "trajectory", "completed_sessions") != len(shard.samples):
            raise OracleRunError("worker completed-session count mismatch")

    def total(section: str | None, key: str) -> int:
        return sum(count(summary, section, key) for summary in summaries)

    worker_peak_rss_raw = [
        summary.get("memory", {}).get("peak_rss_bytes") for summary in summaries
    ]
    worker_peak_rss_available = all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in worker_peak_rss_raw
    )
    worker_peak_rss = (
        [int(value) for value in worker_peak_rss_raw]
        if worker_peak_rss_available
        else []
    )

    merged: dict[str, Any] = {
        "parallel_workers": len(shards),
        "per_shard": [
            {
                "shard_index": shard.shard_index,
                "sample_count": len(shard.samples),
                "summary": shard.receipt.worker_summary,
            }
            for shard in shards
        ],
        "trajectory": {
            "fixed_turns": 10,
            "top_k": 10,
            "completed_sessions": total("trajectory", "completed_sessions"),
            "respond_count": total("trajectory", "respond_count"),
        },
        "actions": {
            "ids": list(ACTION_IDS),
            "result_aware_computation_count": total("actions", "result_aware_computation_count"),
        },
        "semantic": {
            "query_count": total("semantic", "query_count"),
            "candidate_matrix_rows_read": total("semantic", "candidate_matrix_rows_read"),
            "maximum_candidate_rows_read": max(
                count(summary, "semantic", "maximum_candidate_rows_read") for summary in summaries
            ),
            "full_catalog_search_calls": total("semantic", "full_catalog_search_calls"),
            "failure_count": total("semantic", "failure_count"),
        },
        "memory": {
            "all_worker_peak_rss_available": worker_peak_rss_available,
            "peak_rss_bytes_max": (
                max(worker_peak_rss) if worker_peak_rss_available else None
            ),
            "sum_of_worker_peak_rss_bytes_upper_bound": (
                sum(worker_peak_rss) if worker_peak_rss_available else None
            ),
            "parent_process_rss_included": False,
        },
    }
    merged.update({key: total(None, key) for key in gate_keys})
    return merged


def _join_closed_shards(
    shards: Sequence[ShardResult], catalog_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    joined: list[dict[str, Any]] = []
    counts = {10: 0, 20: 0, 50: 0, 100: 0}
    combined_trace = hashlib.sha256()
    validated = []
    for shard in shards:
        raw = _read_trace(
            shard.trace_path, shard.receipt, len(shard.samples) * 10
        )
        trace = validate_trace(raw, len(shard.samples), catalog_ids)
        validated.append(trace)
        for raw_row in raw:
            normalized = dict(raw_row)
            normalized["ordinal"] = shard.global_start + int(raw_row["ordinal"]) - 1
            combined_trace.update(_canonical_bytes(normalized))
    # Validate every closed trace before the first label-bearing join.
    for shard, trace in zip(shards, validated, strict=True):
        local, recall = join_labels_after_close(
            shard.samples, shard.ledger, trace, include_counts=True
        )
        for row in local:
            row["ordinal"] = shard.global_start + int(row["ordinal"]) - 1
        joined.extend(local)
        for cutoff in counts:
            counts[cutoff] += int(recall["cutoffs"][str(cutoff)]["hit_count"])
    sample_count = len(joined)
    return joined, {
        "sample_count": sample_count,
        "cutoffs": {
            str(cutoff): {
                "hit_count": count,
                "rate": round(count / sample_count, 6),
            }
            for cutoff, count in counts.items()
        },
    }, combined_trace.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = {
        DEFAULT_CONFIG.as_posix(),
        EXPECTED_WORKER,
        "scripts/p12_actions.py",
        "scripts/p12_oracle_metrics.py",
        "scripts/official_metric_bridge.py",
        "scripts/evaluate_p12_action_oracle.py",
        *(
            path.relative_to(REPO_ROOT).as_posix()
            for package in (REPO_ROOT / "starter", REPO_ROOT / "evaluator")
            for path in package.glob("*.py")
        ),
    }
    return {
        relative: _sha256_lf(_resolve_regular_file(relative))
        for relative in sorted(paths)
    }


def _p11_asset_snapshot(config: Mapping[str, Any]) -> dict[str, int | str]:
    p11 = config["p11"]
    path = _resolve_regular_file(str(p11["sidecar_path"]))
    observed = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    expected = {
        "bytes": int(p11["sidecar_bytes"]),
        "sha256": str(p11["sidecar_sha256"]),
    }
    if observed != expected:
        raise OracleRunError("P11 sidecar identity mismatch")
    return observed


def _semantic_asset_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    semantic = config["semantic"]
    model_root = _resolve_repository_directory(str(semantic["model_dir"]))
    index_root = _resolve_repository_directory(str(semantic["index_dir"]))
    spec = _load_json(_resolve_regular_file(str(semantic["spec_path"])))
    lock = _load_json(_resolve_regular_file(str(semantic["lock_path"])))
    required = spec.get("required_files")
    index = lock.get("index")
    if not isinstance(required, list) or not isinstance(index, Mapping):
        raise OracleRunError("semantic asset registry is invalid")
    registry: dict[str, dict[str, int | str]] = {}
    for item in required:
        if not isinstance(item, Mapping):
            raise OracleRunError("semantic model asset entry is invalid")
        relative = str(item.get("path", ""))
        path = _resolve_asset_file(model_root, relative)
        observed = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        expected = {"bytes": item.get("bytes"), "sha256": item.get("sha256")}
        if observed != expected:
            raise OracleRunError(f"semantic model asset identity mismatch: {relative}")
        registry[f"model/{relative}"] = observed
    for name in ("manifest", "matrix", "ordered_asins"):
        item = index.get(name)
        if not isinstance(item, Mapping):
            raise OracleRunError(f"semantic index asset entry is invalid: {name}")
        relative = str(item.get("path", ""))
        path = _resolve_asset_file(index_root, relative)
        observed = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        expected = {"bytes": item.get("bytes"), "sha256": item.get("sha256")}
        if observed != expected:
            raise OracleRunError(f"semantic index asset identity mismatch: {name}")
        registry[f"index/{relative}"] = observed
    return {
        "files": registry,
        "file_count": len(registry),
        "total_bytes": sum(int(item["bytes"]) for item in registry.values()),
        "registry_sha256": hashlib.sha256(_canonical_bytes(registry)).hexdigest(),
    }


def build_go_no_go(
    aggregate: Mapping[str, Any],
    worker_summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    decision_eligible: bool,
) -> dict[str, Any]:
    """Apply the frozen P12 gate to aggregate-only evidence."""

    gate = config["go_no_go"]
    oracle = aggregate["oracle"]
    baseline = aggregate["actions"][KEEP_P11]
    oracle_metrics = oracle["metrics"]["row_uniform_official"]
    baseline_metrics = baseline["metrics"]["row_uniform_official"]
    oracle_hr_delta = float(oracle["relative_to_baseline"]["hit_rate_delta"])
    oracle_score_delta = round(
        float(oracle_metrics["recommended_technical_score"])
        - float(baseline_metrics["recommended_technical_score"]),
        12,
    )
    ci_lower = float(oracle["paired_utility_bootstrap_ci"]["lower"])
    signal_threshold_passed = (
        oracle_hr_delta >= float(gate["oracle_hr_delta_min"])
        or oracle_score_delta >= float(gate["oracle_score_delta_min"])
    )
    ci_passed = (
        ci_lower > 0.0
        if gate["require_positive_score_delta_ci_lower"] is True
        else True
    )

    deployable_actions = (
        CANDIDATE_RERANK,
        FROZEN_SEMANTIC_RERANK,
        RESULT_AWARE_REWRITE_RETRIEVE,
        COMPACT_NEGATIVE_C50,
        GUARDED_COMPACT_SLOT10,
        GUARDED_COMPACT_SLOT10_STRICT,
        *FAMILY2_NOVEL_SLOT10_ACTIONS,
    )
    stable_actions: list[str] = []
    for action in deployable_actions:
        relative = aggregate["actions"][action]["relative_to_baseline"]
        if (
            int(relative["net_rescues"]) >= int(gate["deployable_net_rescue_min"])
            and int(relative["positive_net_scenario_span"])
            >= int(gate["deployable_scenario_span_min"])
            and int(relative["positive_net_taxonomy_span"])
            >= int(gate["deployable_taxonomy_span_min"])
        ):
            stable_actions.append(action)

    required_worker_counts = (
        "network_attempt_count",
        "semantic_failure_count",
        "rewrite_failure_count",
        "family2_score_failure_count",
        "full_catalog_search_calls",
        "p11_invariant_failure_count",
    )
    counts: dict[str, int] = {}
    for field in required_worker_counts:
        value = worker_summary.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise OracleRunError(f"worker summary count is invalid: {field}")
        counts[field] = value
    action_failures = counts["semantic_failure_count"] + counts["rewrite_failure_count"]
    worker_passed = (
        counts["network_attempt_count"] <= int(gate["network_attempt_count_max"])
        and action_failures <= int(gate["semantic_or_rewrite_failure_count_max"])
        and counts["family2_score_failure_count"]
        <= int(gate["family2_score_failure_count_max"])
        and counts["full_catalog_search_calls"] == 0
        and counts["p11_invariant_failure_count"] == 0
    )
    evidence_passed = bool(
        signal_threshold_passed and ci_passed and stable_actions and worker_passed
    )
    return {
        "status": (
            "GO"
            if decision_eligible and evidence_passed
            else "NO_GO"
            if decision_eligible
            else "NON_DECISION_SIGNAL_ONLY"
        ),
        "decision_eligible": decision_eligible,
        "evidence_passed": evidence_passed,
        "oracle_hr_delta": oracle_hr_delta,
        "oracle_technical_score_delta": oracle_score_delta,
        "paired_utility_ci_lower": ci_lower,
        "signal_threshold_passed": signal_threshold_passed,
        "positive_ci_passed": ci_passed,
        "stable_deployable_actions": stable_actions,
        "worker_integrity_passed": worker_passed,
        "worker_counts": counts,
        "confirmation_authorized_by_generic_runner": False,
        "cage_r10_implementation_authorized": bool(decision_eligible and evidence_passed),
    }


def _safe_output_path(split: str, limit: int | None) -> Path:
    root = _resolve_repository_directory(EXPECTED_OUTPUT_ROOT, create=True)
    suffix = "full" if limit is None else f"smoke-{limit}"
    return root / f"{split}-{suffix}-aggregate.json"


def _write_exclusive(path: Path, value: object) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(_canonical_bytes(value))
    except FileExistsError as exc:
        raise OracleRunError(f"refusing to overwrite existing artifact: {path.name}") from exc


def run(split: str, *, limit: int | None = None) -> tuple[Path, dict[str, Any]]:
    # This in-memory gate must precede config/path/hash/process operations so a
    # programmatic caller cannot use the generic runner to probe confirmation.
    if split not in ALLOWED_SPLITS:
        raise OracleRunError(f"split is not available to the generic runner: {split}")
    config = load_frozen_config()
    split_spec = select_split(config, split)
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit < int(split_spec["rows"])
    ):
        raise OracleRunError("limit must be between 1 and rows-1; full runs omit it")

    manifest_snapshot = validate_manifest(config)
    samples = _verified_jsonl(
        str(split_spec["path"]), str(split_spec["sha256"]), int(split_spec["rows"])
    )
    if limit is not None:
        samples = samples[:limit]
    catalog_ids, categories, products = load_catalog(config)
    source_snapshot = _source_hashes()
    p11_snapshot = _p11_asset_snapshot(config)
    semantic_snapshot = _semantic_asset_snapshot(config)

    output_path = _safe_output_path(split, limit)
    if output_path.exists():
        raise OracleRunError(f"refusing to overwrite existing artifact: {output_path.name}")
    nonce = hashlib.sha256(
        f"p12-action-oracle-v1\0{split}\0{limit or 'full'}".encode("utf-8")
    ).hexdigest()[:32]
    shard_count = min(int(config["runtime"]["parallel_workers"]), len(samples))
    trace_paths = [
        output_path.with_name(
            output_path.name.replace(
                "-aggregate.json",
                f"-blind-shard-{index + 1:02d}-of-{shard_count:02d}.jsonl",
            )
        )
        for index in range(shard_count)
    ]
    for trace_path in trace_paths:
        if trace_path.exists():
            raise OracleRunError(
                f"refusing to overwrite existing trace: {trace_path.name}"
            )

    started = time.perf_counter()
    shards = _run_blind_shards(
        config,
        samples,
        categories,
        products,
        catalog_ids,
        trace_paths,
        nonce,
    )

    # Every worker has finalized and cleanly exited above.  No trace may be
    # opened and no target may be joined until this global revalidation passes.
    _require_equal(load_frozen_config(), config, "post_run_config")
    _require_equal(_source_hashes(), source_snapshot, "post_run_source_snapshot")
    _require_equal(_p11_asset_snapshot(config), p11_snapshot, "post_run_p11_sidecar")
    _require_equal(
        _semantic_asset_snapshot(config),
        semantic_snapshot,
        "post_run_semantic_assets",
    )
    _require_equal(validate_manifest(config), manifest_snapshot, "post_run_manifest")
    _require_equal(_sha256(_resolve_regular_file(str(split_spec["path"]))), split_spec["sha256"], "post_run_split_sha256")
    _require_equal(_sha256(_resolve_regular_file(str(config["catalog"]["path"]))), config["catalog"]["sha256"], "post_run_catalog_sha256")
    _require_equal(_sha256(_resolve_regular_file(str(config["proxy"]["manifest_path"]))), config["proxy"]["manifest_sha256"], "post_run_manifest_sha256")

    worker_summary = _merge_worker_summaries(shards)
    sanitized_session_count = sum(
        int(int(meta.get("sanitized_visible_token_count", 0)) > 0)
        for shard in shards
        for meta in shard.ledger
    )
    sanitized_message_count = sum(
        int(meta.get("sanitized_visible_message_count", 0))
        for shard in shards
        for meta in shard.ledger
    )
    sanitized_token_count = sum(
        int(meta.get("sanitized_visible_token_count", 0))
        for shard in shards
        for meta in shard.ledger
    )
    joined, candidate_recall, combined_trace_sha256 = _join_closed_shards(
        shards, catalog_ids
    )
    aggregate = aggregate_action_oracle(
        joined,
        action_ids=list(ACTION_IDS),
        oracle_eligible_actions=list(config["actions"]["oracle_eligible"]),
        baseline_action=KEEP_P11,
        bootstrap_resamples=int(config["evaluation"]["bootstrap"]["resamples"]),
        bootstrap_seed=int.from_bytes(
            hashlib.sha256(
                str(config["evaluation"]["bootstrap"]["seed"]).encode("utf-8")
            ).digest()[:8],
            "big",
        ),
    )

    # Detect source/config replacement during the label join or aggregation too.
    _require_equal(load_frozen_config(), config, "post_aggregate_config")
    _require_equal(_source_hashes(), source_snapshot, "post_aggregate_sources")

    decision_eligible = split == "selection" and limit is None
    decision = build_go_no_go(
        aggregate,
        worker_summary,
        config,
        decision_eligible=decision_eligible,
    )
    trace_registry = [
        {
            "shard_index": shard.shard_index,
            "trace_sha256": shard.receipt.trace_sha256,
            "record_count": shard.receipt.record_count,
        }
        for shard in shards
    ]
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "split": split,
            "sample_count": len(samples),
            "full_split": limit is None,
            "decision_eligible": decision_eligible,
            "fixed_turns_per_session": 10,
            "trajectory": config["evaluation"]["trajectory"],
            "label_join": config["evaluation"]["label_join"],
            "confirmation_accessed": False,
            "amazon_test_rows_read": 0,
            "released_public_rows_read": 0,
            "ask_is_independent_counterfactual": False,
            "visible_identifier_sanitization": {
                "policy": "reject_target_or_catalog_then_redact_non_catalog_asin_shape",
                "replacement": VISIBLE_IDENTIFIER_REPLACEMENT,
                "session_count": sanitized_session_count,
                "message_count": sanitized_message_count,
                "token_count": sanitized_token_count,
            },
        },
        "provenance": {
            "config_canonical_sha256": hashlib.sha256(
                _canonical_bytes(config)
            ).hexdigest(),
            "manifest_sha256": config["proxy"]["manifest_sha256"],
            "split_sha256": split_spec["sha256"],
            "catalog_sha256": config["catalog"]["sha256"],
            "blind_trace_registry": trace_registry,
            "blind_trace_registry_sha256": hashlib.sha256(
                _canonical_bytes(trace_registry)
            ).hexdigest(),
            "combined_blind_trace_sha256": combined_trace_sha256,
            "p11_sidecar": p11_snapshot,
            "semantic_assets": {
                "file_count": semantic_snapshot["file_count"],
                "total_bytes": semantic_snapshot["total_bytes"],
                "registry_sha256": semantic_snapshot["registry_sha256"],
            },
            "source_sha256_lf": source_snapshot,
            "python": {
                "version": sys.version,
                "executable_name": Path(sys.executable).name,
            },
        },
        "candidate_recall": candidate_recall,
        "action_oracle": aggregate,
        "go_no_go": decision,
        "worker": worker_summary,
        "runtime": {"wall_seconds": round(time.perf_counter() - started, 6)},
        "limitations": [
            "The proxy is Amazon validation-derived and is not organizer-private evaluation.",
            "ASK follows the observed KEEP_P11 trajectory and is not an independent counterfactual.",
            "RESULT_AWARE_REWRITE_RETRIEVE is an R08-based diagnostic action, not a P11 composition.",
            "CANDIDATE_RERANK and FROZEN_SEMANTIC_RERANK are fixed P12-v1 diagnostic policies, not promoted production routes.",
            "The three compact-negative actions are target-blind diagnostics, not promoted production routes.",
            "The three Family2 novel-slot actions are target-blind C50 diagnostics, not promoted production routes.",
            "BUDGET_AROUND_NOVEL_SLOT10 was rejected before execution because source-only catalog inspection found price coverage in only 3 of 50,000 rows.",
        ],
    }
    assert_identifier_free_artifact(artifact, catalog_ids)
    _write_exclusive(output_path, artifact)
    return output_path, artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=tuple(ALLOWED_SPLITS))
    parser.add_argument(
        "--limit",
        type=int,
        help="run a non-decision smoke prefix; omit for the complete split",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        path, artifact = run(args.split, limit=args.limit)
    except OracleRunError as exc:
        print(f"P12 action oracle failed closed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact": str(path.relative_to(REPO_ROOT)),
                "sample_count": artifact["protocol"]["sample_count"],
                "decision_eligible": artifact["protocol"]["decision_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
