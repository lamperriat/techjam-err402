from __future__ import annotations

"""Target-blind, process-isolated evaluator runner for the frozen P7 study.

The official simulator stays in the parent process.  A worker receives only the
catalog/model paths and the ordinary Agent reset/respond arguments; it never
receives a sample object, target, scenario label, or evaluator result.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SPEC = PROJECT_ROOT / "configs" / "p7_bge_small_en_v1_5.json"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "catalog.jsonl"
DEFAULT_SELECTION = PROJECT_ROOT / "experiments" / "p7_selection_product_disjoint.jsonl"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "experiments" / "p7_assets" / "bge-small-en-v1.5"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "experiments" / "p7_index"
DEFAULT_INDEX_LOCK = PROJECT_ROOT / "configs" / "p7_semantic_index_lock.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "p7_semantic_feasibility.json"
SCHEMA_VERSION = "p7.target-blind-evaluation.v1"
C00 = "P7.C00.r08_coverage"
S00 = "P7.S00.bge_dense_shadow"
WORKER_FACTORY = "starter.p7_lab:create_p7_agent"
FROZEN_CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
FROZEN_CATALOG_ROWS = 50_000
FROZEN_SELECTION_SHA256 = "bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546"
FROZEN_SELECTION_COUNT = 200
FROZEN_SCENARIOS = {"boundary": 10, "browsing": 80, "buying": 80, "intent_override": 30}
FROZEN_SPEC_CANONICAL_SHA256 = "e71d0cad480c89eac25ad2b276de9a4e7153e1ec2f3bdcc793682f183a592200"
PROHIBITED_CANONICAL_KEYS = {
    "duration",
    "duration_ns",
    "ground_truth",
    "latency",
    "pid",
    "sample_id",
    "scenario",
    "scenario_type",
    "session_id",
    "target",
    "target_asin",
    "target_id",
    "timestamp",
    "uuid",
}


class P7RunnerError(RuntimeError):
    """The frozen runner or worker contract was violated."""


def canonical_json_line(record: Mapping[str, Any]) -> bytes:
    _reject_prohibited_keys(record)
    try:
        payload = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise P7RunnerError(f"record is not canonical JSON: {exc}") from exc
    return payload + b"\n"


def canonical_jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_line(record) for record in records)


def canonical_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_jsonl(records)).hexdigest()


def _reject_prohibited_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            if normalized in PROHIBITED_CANONICAL_KEYS:
                raise P7RunnerError(f"prohibited canonical key: {key}")
            _reject_prohibited_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_prohibited_keys(nested)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P7RunnerError(f"JSON root must be an object: {path}")
    return value


def _safe_project_path(project_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise P7RunnerError(f"{label} path must be a non-empty project-relative string")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise P7RunnerError(f"{label} path is not safely project-relative")
    root = project_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise P7RunnerError(f"{label} path escapes project root") from exc
    return candidate


def _validate_locked_file(
    project_root: Path, entry: Mapping[str, Any], label: str
) -> tuple[Path, int, str]:
    path = _safe_project_path(project_root, entry.get("path"), label)
    expected_bytes = entry.get("bytes")
    expected_sha = entry.get("sha256")
    if (
        not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha)
    ):
        raise P7RunnerError(f"{label} has invalid frozen bytes/SHA-256")
    if not path.is_file():
        raise P7RunnerError(f"{label} is missing: {path}")
    actual_bytes, actual_sha = path.stat().st_size, _sha256_file(path)
    if actual_bytes != expected_bytes or actual_sha != expected_sha:
        raise P7RunnerError(f"{label} identity does not match semantic index lock")
    return path, actual_bytes, actual_sha


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_samples_sha256(samples: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            json.dumps(
                sample,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _git(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={project_root.resolve().as_posix()}", *arguments],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise P7RunnerError(f"Git command failed: {' '.join(arguments)}")
    return completed.stdout.strip()


def capture_source_snapshot(project_root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    branch = _git(project_root, "branch", "--show-current")
    if not branch:
        raise P7RunnerError("P7 evaluation requires a named Git branch")
    head = _git(project_root, "rev-parse", "HEAD")
    origin_head = _git(project_root, "rev-parse", f"origin/{branch}")
    status = _git(project_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise P7RunnerError("P7 evaluation requires a completely clean worktree")
    if origin_head != head:
        raise P7RunnerError("P7 evaluation requires origin/<branch> to equal HEAD")
    identities = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise P7RunnerError(f"snapshot input is missing: {resolved}")
        identities[resolved.relative_to(project_root.resolve()).as_posix()] = {
            "bytes": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
        }
    return {
        "branch": branch,
        "head": head,
        "origin_head": origin_head,
        "clean": True,
        "files": dict(sorted(identities.items())),
    }


def validate_frozen_inputs(
    *,
    catalog: Path,
    selection: Path,
    spec_path: Path,
    index_lock: Path,
    model_dir: Path,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_paths = {
        "catalog": DEFAULT_CATALOG,
        "selection": DEFAULT_SELECTION,
        "spec": DEFAULT_SPEC,
        "index_lock": DEFAULT_INDEX_LOCK,
        "model_dir": DEFAULT_MODEL_DIR,
    }
    observed_paths = {
        "catalog": catalog,
        "selection": selection,
        "spec": spec_path,
        "index_lock": index_lock,
        "model_dir": model_dir,
    }
    for name, expected in expected_paths.items():
        if observed_paths[name].resolve() != expected.resolve():
            raise P7RunnerError(f"formal P7 evaluation requires default frozen {name}")
    if _sha256_file(catalog) != FROZEN_CATALOG_SHA256:
        raise P7RunnerError("official catalog SHA-256 mismatch")
    with catalog.open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != FROZEN_CATALOG_ROWS:
        raise P7RunnerError("official catalog row count mismatch")
    sample_hash = _canonical_samples_sha256(samples)
    if sample_hash != FROZEN_SELECTION_SHA256 or len(samples) != FROZEN_SELECTION_COUNT:
        raise P7RunnerError("P7 selection corpus identity mismatch")
    scenario_counts: dict[str, int] = {}
    for sample in samples:
        scenario = str(sample.get("scenario_type"))
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    if scenario_counts != FROZEN_SCENARIOS:
        raise P7RunnerError("P7 selection scenario mix mismatch")
    spec = _load_json_object(spec_path)
    if _canonical_json_sha256(spec) != FROZEN_SPEC_CANONICAL_SHA256:
        raise P7RunnerError("P7 semantic model spec canonical SHA-256 mismatch")
    configured = spec.get("evaluation", {}).get("selection_corpus", {})
    if (
        configured.get("path") != "experiments/p7_selection_product_disjoint.jsonl"
        or configured.get("samples_sha256") != FROZEN_SELECTION_SHA256
        or configured.get("sample_count") != FROZEN_SELECTION_COUNT
        or configured.get("scenario_counts") != FROZEN_SCENARIOS
        or configured.get("public_evaluation_authorized") is not False
    ):
        raise P7RunnerError("P7 spec does not authorize only the frozen P7 corpus")
    return {
        "catalog_sha256": FROZEN_CATALOG_SHA256,
        "catalog_rows": rows,
        "selection_canonical_sha256": sample_hash,
        "selection_count": len(samples),
        "scenario_counts": scenario_counts,
        "spec_canonical_sha256": FROZEN_SPEC_CANONICAL_SHA256,
    }


def validate_index_lock(
    lock_path: Path,
    *,
    project_root: Path,
    spec_path: Path,
    catalog_path: Path,
    index_dir: Path,
    enforce_git: bool = True,
) -> dict[str, Any]:
    """Hard-gate the tracked post-build lock and every referenced artifact."""

    lock = _load_json_object(lock_path)
    required_roots = {
        "schema_version",
        "source",
        "model_spec",
        "catalog",
        "index",
        "asset_scope",
        "build_observation",
    }
    if set(lock) != required_roots or lock.get("schema_version") != "p7.semantic-index-lock.v1":
        raise P7RunnerError("semantic index lock has the wrong strict root schema")
    source = lock["source"]
    model_spec = lock["model_spec"]
    catalog = lock["catalog"]
    index = lock["index"]
    asset_scope = lock["asset_scope"]
    observation = lock["build_observation"]
    if any(
        not isinstance(value, Mapping)
        for value in (source, model_spec, catalog, index, asset_scope, observation)
    ):
        raise P7RunnerError("semantic index lock sections must be objects")
    if set(source) != {"git_commit", "git_branch", "builder", "semantic"}:
        raise P7RunnerError("semantic index lock source schema is incomplete")
    commit, branch = source.get("git_commit"), source.get("git_branch")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
        or not isinstance(branch, str)
        or not branch
    ):
        raise P7RunnerError("semantic index lock source revision is invalid")
    _validate_locked_file(project_root, source["builder"], "locked builder source")
    _validate_locked_file(project_root, source["semantic"], "locked semantic source")
    if enforce_git:
        commit_exists = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.resolve().as_posix()}", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        ancestor = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.resolve().as_posix()}", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        branch_run = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.resolve().as_posix()}", "branch", "--show-current"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_exists.returncode or ancestor.returncode:
            raise P7RunnerError("locked build commit is not an ancestor of current HEAD")
        if branch_run.returncode or branch_run.stdout.strip() != branch:
            raise P7RunnerError("current Git branch does not match semantic index lock")
        tracked = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={project_root.resolve().as_posix()}",
                "ls-files",
                "--error-unmatch",
                lock_path.resolve().relative_to(project_root.resolve()).as_posix(),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if tracked.returncode:
            raise P7RunnerError("semantic index lock must be tracked by Git")

    if set(model_spec) != {"path", "raw_bytes", "raw_sha256", "canonical_sha256"}:
        raise P7RunnerError("semantic index lock model_spec schema is incomplete")
    locked_spec = _safe_project_path(project_root, model_spec["path"], "locked model spec")
    if locked_spec.resolve() != spec_path.resolve():
        raise P7RunnerError("requested model spec differs from semantic index lock")
    if (
        spec_path.stat().st_size != model_spec["raw_bytes"]
        or _sha256_file(spec_path) != model_spec["raw_sha256"]
    ):
        raise P7RunnerError("model spec raw identity differs from semantic index lock")
    parsed_spec = _load_json_object(spec_path)
    if _canonical_json_sha256(parsed_spec) != model_spec["canonical_sha256"]:
        raise P7RunnerError("model spec canonical identity differs from semantic index lock")
    if enforce_git:
        def verify_build_blob(entry: Mapping[str, Any], label: str) -> None:
            relative = entry.get("path")
            if not isinstance(relative, str):
                raise P7RunnerError(f"{label} has no project-relative path")
            shown = subprocess.run(
                [
                    "git", "-c", f"safe.directory={project_root.resolve().as_posix()}",
                    "show", f"{commit}:{relative}",
                ],
                cwd=project_root,
                capture_output=True,
                check=False,
            )
            if shown.returncode:
                raise P7RunnerError(f"{label} is absent from locked build commit")
            blob = shown.stdout
            if len(blob) != entry.get("bytes") or hashlib.sha256(blob).hexdigest() != entry.get("sha256"):
                raise P7RunnerError(f"{label} differs from locked build commit blob")

        verify_build_blob(source["builder"], "locked builder source")
        verify_build_blob(source["semantic"], "locked semantic source")
        verify_build_blob(
            {"path": model_spec["path"], "bytes": model_spec["raw_bytes"], "sha256": model_spec["raw_sha256"]},
            "locked model spec",
        )

    if set(catalog) != {"path", "bytes", "sha256", "rows"}:
        raise P7RunnerError("semantic index lock catalog schema is incomplete")
    locked_catalog = _safe_project_path(project_root, catalog["path"], "locked catalog")
    if locked_catalog.resolve() != catalog_path.resolve():
        raise P7RunnerError("requested catalog differs from semantic index lock")
    if (
        not isinstance(catalog["rows"], int)
        or catalog["rows"] <= 0
        or catalog_path.stat().st_size != catalog["bytes"]
        or _sha256_file(catalog_path) != catalog["sha256"]
    ):
        raise P7RunnerError("catalog identity differs from semantic index lock")
    with catalog_path.open("rb") as handle:
        actual_rows = 0
        for line in handle:
            if not line.strip():
                raise P7RunnerError("locked catalog contains a blank row")
            actual_rows += 1
    if actual_rows != catalog["rows"]:
        raise P7RunnerError("catalog row count differs from semantic index lock")

    expected_index_keys = {
        "directory",
        "manifest",
        "matrix",
        "ordered_asins",
        "canonical_documents_sha256",
    }
    if set(index) != expected_index_keys:
        raise P7RunnerError("semantic index lock index schema is incomplete")
    locked_index_dir = _safe_project_path(project_root, index["directory"], "locked index directory")
    if locked_index_dir.resolve() != index_dir.resolve() or not locked_index_dir.is_dir():
        raise P7RunnerError("requested semantic index directory differs from lock")
    manifest_path, _, manifest_sha = _validate_locked_file(
        locked_index_dir, index["manifest"], "locked semantic manifest"
    )
    matrix_path, _, matrix_sha = _validate_locked_file(
        locked_index_dir, index["matrix"], "locked semantic matrix"
    )
    asins_path, _, asins_sha = _validate_locked_file(
        locked_index_dir, index["ordered_asins"], "locked ordered ASINs"
    )
    manifest = _load_json_object(manifest_path)
    manifest_lock = index["manifest"]
    if set(manifest_lock) != {"path", "bytes", "sha256", "schema_version"}:
        raise P7RunnerError("locked manifest schema is incomplete")
    if manifest.get("schema_version") != manifest_lock["schema_version"]:
        raise P7RunnerError("manifest schema differs from semantic index lock")
    matrix_lock = index["matrix"]
    if set(matrix_lock) != {"path", "bytes", "sha256", "dtype", "shape"}:
        raise P7RunnerError("locked matrix schema is incomplete")
    if matrix_lock["dtype"] != "float32" or matrix_lock["shape"] != parsed_spec["index"]["shape"]:
        raise P7RunnerError("locked matrix dtype/shape differs from frozen spec")
    asin_lock = index["ordered_asins"]
    if set(asin_lock) != {"path", "bytes", "sha256", "count", "encoding", "line_ending"}:
        raise P7RunnerError("locked ordered-ASIN schema is incomplete")
    if (
        asin_lock["count"] != catalog["rows"]
        or asin_lock["encoding"] != "utf-8-lf"
        or asin_lock["line_ending"] != "LF"
    ):
        raise P7RunnerError("locked ordered-ASIN metadata is invalid")
    if (
        manifest.get("matrix", {}).get("path") != matrix_lock["path"]
        or manifest.get("matrix", {}).get("sha256") != matrix_sha
        or manifest.get("matrix", {}).get("bytes") != matrix_lock["bytes"]
        or manifest.get("matrix", {}).get("dtype") != matrix_lock["dtype"]
        or manifest.get("matrix", {}).get("shape") != matrix_lock["shape"]
        or manifest.get("ordered_asins", {}).get("path") != asin_lock["path"]
        or manifest.get("ordered_asins", {}).get("sha256") != asins_sha
        or manifest.get("ordered_asins", {}).get("bytes") != asin_lock["bytes"]
        or manifest.get("ordered_asins", {}).get("count") != asin_lock["count"]
        or manifest.get("model_spec_sha256") != model_spec["canonical_sha256"]
        or manifest.get("catalog_sha256") != catalog["sha256"]
    ):
        raise P7RunnerError("semantic manifest and tracked lock disagree")
    documents_sha = index.get("canonical_documents_sha256")
    if (
        not isinstance(documents_sha, str)
        or len(documents_sha) != 64
        or manifest.get("preprocessing", {}).get("canonical_documents_sha256") != documents_sha
    ):
        raise P7RunnerError("canonical-document digest differs from semantic index lock")

    if set(asset_scope) != {"required_asset_bytes", "required_asset_bytes_max"}:
        raise P7RunnerError("semantic index lock asset_scope schema is incomplete")
    frozen_max = parsed_spec["evaluation"]["resource_gates"]["required_asset_bytes_max"]
    if (
        not isinstance(asset_scope["required_asset_bytes"], int)
        or asset_scope["required_asset_bytes"] <= 0
        or asset_scope["required_asset_bytes_max"] != frozen_max
        or asset_scope["required_asset_bytes"] > frozen_max
        or manifest.get("asset_byte_scope", {}).get("required_asset_bytes")
        != asset_scope["required_asset_bytes"]
    ):
        raise P7RunnerError("semantic asset-byte scope fails the tracked lock")
    if set(observation) != {
        "wall_seconds",
        "rss_backend",
        "baseline_rss_bytes",
        "peak_rss_bytes",
        "peak_delta_from_baseline_bytes",
    }:
        raise P7RunnerError("semantic index lock build_observation schema is incomplete")
    if (
        not isinstance(observation["wall_seconds"], (int, float))
        or observation["wall_seconds"] < 0
        or not isinstance(observation["rss_backend"], str)
        or not observation["rss_backend"]
        or not isinstance(observation["peak_rss_bytes"], int)
        or observation["peak_rss_bytes"] <= 0
        or not isinstance(observation["baseline_rss_bytes"], int)
        or observation["baseline_rss_bytes"] <= 0
        or not isinstance(observation["peak_delta_from_baseline_bytes"], int)
        or observation["peak_delta_from_baseline_bytes"] < 0
        or observation["peak_rss_bytes"] - observation["baseline_rss_bytes"]
        != observation["peak_delta_from_baseline_bytes"]
    ):
        raise P7RunnerError("semantic index lock build observation is invalid")
    manifest_resources = manifest.get("build_resources")
    if not isinstance(manifest_resources, Mapping) or any(
        manifest_resources.get(key) != observation[key]
        for key in observation
    ):
        raise P7RunnerError("semantic manifest build observation differs from lock")
    return {
        "lock_sha256": _sha256_file(lock_path),
        "manifest_sha256": manifest_sha,
        "matrix_sha256": matrix_sha,
        "ordered_asins_sha256": asins_sha,
        "required_asset_bytes": asset_scope["required_asset_bytes"],
        "canonical_documents_sha256": documents_sha,
    }


def _percentile95_ns(records: list[dict[str, Any]], eligible: set[tuple[int, int]]) -> float | None:
    values = sorted(
        int(record["nanoseconds"])
        for record in records
        if (int(record["ordinal"]), int(record["turn"])) in eligible
    )
    if not values:
        return None
    return values[math.ceil(0.95 * len(values)) - 1] / 1_000_000.0


@dataclass
class WorkerClient:
    role: str
    process: subprocess.Popen[str]
    nonce: str
    stderr_path: Path
    _next_request_id: int = 1
    _next_ordinal: int = 1
    _ordinal_by_opaque_id: dict[str, int] | None = None

    @classmethod
    def start(
        cls,
        role: str,
        *,
        catalog: Path,
        spec: Path,
        model_dir: Path,
        index_dir: Path,
        index_lock: Path,
        worker_factory: str = WORKER_FACTORY,
        rss_sample_ms: float = 10.0,
        stderr_path: Path,
    ) -> "WorkerClient":
        if role not in {C00, S00}:
            raise P7RunnerError(f"unknown P7 role: {role}")
        nonce = uuid.uuid4().hex
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "p7_worker.py"),
            "--role",
            role,
            "--nonce",
            nonce,
            "--factory",
            worker_factory,
            "--catalog",
            str(catalog),
            "--spec",
            str(spec),
            "--model",
            str(model_dir),
            "--index",
            str(index_dir),
            "--lock",
            str(index_lock),
            "--rss-ms",
            str(rss_sample_ms),
        ]
        error_handle = stderr_path.open("w+", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=error_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        error_handle.close()
        client = cls(role, process, nonce, stderr_path, _ordinal_by_opaque_id={})
        ready = client._read_message()
        if (
            ready.get("kind") != "ready"
            or ready.get("nonce") != nonce
            or ready.get("role") != role
        ):
            client.abort()
            raise P7RunnerError(f"invalid P7 worker ready message: {ready}")
        return client

    def _read_message(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise P7RunnerError("worker stdout is unavailable")
        line = self.process.stdout.readline()
        if not line:
            self.process.wait(timeout=5)
            error = self.stderr_path.read_text(encoding="utf-8", errors="replace")
            raise P7RunnerError(
                f"P7 worker {self.role} exited unexpectedly: {error[-1000:]}"
            )
        value = json.loads(line)
        if not isinstance(value, dict):
            raise P7RunnerError("worker protocol message must be an object")
        return value

    def _request(self, operation: str, **payload: Any) -> Any:
        if self.process.stdin is None:
            raise P7RunnerError("worker stdin is unavailable")
        request_id = self._next_request_id
        self._next_request_id += 1
        request = {"request_id": request_id, "operation": operation, **payload}
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        reply = self._read_message()
        if reply.get("request_id") != request_id:
            raise P7RunnerError("worker reply request ID mismatch")
        if reply.get("kind") == "error":
            raise P7RunnerError(
                f"worker {operation} failed: {reply.get('error_class', 'unknown')}"
            )
        if operation == "finalize":
            if reply.get("kind") != "result":
                raise P7RunnerError("worker finalize did not return a result")
            return reply["bundle"]
        if reply.get("kind") != "reply":
            raise P7RunnerError("worker returned an invalid reply")
        return reply.get("result")

    def reset(self, session_id: str, user_profile: dict) -> None:
        assert self._ordinal_by_opaque_id is not None
        if session_id in self._ordinal_by_opaque_id:
            raise P7RunnerError("official evaluator reused an opaque session ID")
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        self._ordinal_by_opaque_id[session_id] = ordinal
        self._request(
            "reset", ordinal=ordinal, user_profile=dict(user_profile)
        )

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        assert self._ordinal_by_opaque_id is not None
        ordinal = self._ordinal_by_opaque_id.get(session_id)
        if ordinal is None:
            raise P7RunnerError("respond received an unknown opaque session ID")
        result = self._request(
            "respond",
            ordinal=ordinal,
            user_message=user_message,
            turn=turn,
            top_k=top_k,
        )
        response = result.get("response") if isinstance(result, dict) else None
        if not isinstance(response, dict):
            raise P7RunnerError("worker response is not an object")
        return response

    def finalize(self) -> dict[str, Any]:
        bundle = self._request("finalize")
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=30)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if return_code != 0:
            error = self.stderr_path.read_text(encoding="utf-8", errors="replace")
            raise P7RunnerError(f"P7 worker failed after finalize: {error[-1000:]}")
        if not isinstance(bundle, dict):
            raise P7RunnerError("worker bundle is not an object")
        bundle["worker_process"] = {
            "isolated": True,
            "role": self.role,
            "nonce": self.nonce,
        }
        return bundle

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()


def normalize_route_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise P7RunnerError("route capture must be an array")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise P7RunnerError("route record must be an object")
        ordinal = int(raw["ordinal"])
        turn = int(raw["turn"])
        coordinate = (ordinal, turn)
        if ordinal <= 0 or not 1 <= turn <= 10 or coordinate in seen:
            raise P7RunnerError(f"invalid or duplicate route coordinate: {coordinate}")
        seen.add(coordinate)
        query = raw.get("query")
        broad = raw.get("broad")
        strict = raw.get("strict")
        dense = raw.get("dense", [])
        empty_query = raw.get("empty_query")
        query_search_ns = raw.get("query_search_ns")
        if not isinstance(query, str):
            raise P7RunnerError("normalized query must be a string")
        if not isinstance(empty_query, bool) or empty_query != (not bool(query)):
            raise P7RunnerError("empty_query flag disagrees with normalized query")
        if not isinstance(query_search_ns, int) or isinstance(query_search_ns, bool) or query_search_ns < 0:
            raise P7RunnerError("route requires non-negative query_search_ns")
        if not isinstance(broad, list) or not isinstance(strict, list) or not isinstance(dense, list):
            raise P7RunnerError("route pools must be arrays")
        if len(broad) > 120 or len(strict) > 80 or len(dense) > 120:
            raise P7RunnerError("route capture exceeds a frozen depth")
        broad_ids = [str(value) for value in broad]
        strict_ids = [str(value) for value in strict]
        if len(broad_ids) != len(set(broad_ids)) or len(strict_ids) != len(set(strict_ids)):
            raise P7RunnerError("sparse route contains duplicate IDs")
        dense_entries: list[dict[str, str]] = []
        dense_ids: set[str] = set()
        for hit in dense:
            if isinstance(hit, Mapping):
                parent_asin = str(hit["parent_asin"])
                score = hit.get("score")
            elif isinstance(hit, (list, tuple)) and len(hit) == 2:
                parent_asin = str(hit[0])
                score = hit[1]
            else:
                raise P7RunnerError("dense hit must contain parent_asin and score")
            if parent_asin in dense_ids:
                raise P7RunnerError("dense route contains duplicate IDs")
            dense_ids.add(parent_asin)
            if isinstance(score, str) and score.startswith(("0x", "-0x")):
                score_hex = score
                numeric_score = float.fromhex(score_hex)
                if not math.isfinite(numeric_score):
                    raise P7RunnerError("dense score must be finite")
            else:
                raise P7RunnerError("dense score must be a canonical hexadecimal float")
            dense_entries.append({"parent_asin": parent_asin, "score": score_hex})
        normalized.append(
            {
                "ordinal": ordinal,
                "turn": turn,
                "query": query,
                "broad": broad_ids,
                "strict": strict_ids,
                "dense": dense_entries,
                "empty_query": bool(empty_query),
                "query_search_ns": query_search_ns,
            }
        )
    expected = sorted(normalized, key=lambda item: (item["ordinal"], item["turn"]))
    if normalized != expected:
        raise P7RunnerError("route records are not in canonical ordinal/turn order")
    return normalized


def normalize_response_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise P7RunnerError("response capture must be an array")
    normalized = []
    seen: set[tuple[int, int]] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise P7RunnerError("response record must be an object")
        ordinal, turn = int(raw["ordinal"]), int(raw["turn"])
        coordinate = (ordinal, turn)
        if ordinal <= 0 or not 1 <= turn <= 10 or coordinate in seen:
            raise P7RunnerError(f"invalid or duplicate response coordinate: {coordinate}")
        seen.add(coordinate)
        response = raw.get("response")
        if not isinstance(response, dict):
            raise P7RunnerError("captured response must be an object")
        normalized.append(
            {"ordinal": ordinal, "turn": turn, "response": response}
        )
    expected = sorted(normalized, key=lambda item: (item["ordinal"], item["turn"]))
    if normalized != expected:
        raise P7RunnerError("response records are not in canonical ordinal/turn order")
    return normalized


def target_blind_alignment(control: dict[str, Any], shadow: dict[str, Any]) -> dict[str, Any]:
    control_responses = normalize_response_records(control.get("response_records"))
    shadow_responses = normalize_response_records(shadow.get("response_records"))
    control_routes = normalize_route_records(control.get("route_records"))
    shadow_routes = normalize_route_records(shadow.get("route_records"))
    control_response_hash = canonical_sha256(control_responses)
    shadow_response_hash = canonical_sha256(shadow_responses)
    control_coordinates = [
        (item["ordinal"], item["turn"]) for item in control_routes
    ]
    shadow_coordinates = [
        (item["ordinal"], item["turn"]) for item in shadow_routes
    ]
    sparse_equal = (
        control_coordinates == shadow_coordinates
        and all(
            left["query"] == right["query"]
            and left["broad"] == right["broad"]
            and left["strict"] == right["strict"]
            for left, right in zip(control_routes, shadow_routes)
        )
    )
    checks = {
        "response_coordinates_equal": [
            (item["ordinal"], item["turn"]) for item in control_responses
        ]
        == [
            (item["ordinal"], item["turn"]) for item in shadow_responses
        ],
        "response_hash_equal": control_response_hash == shadow_response_hash,
        "route_coordinates_equal": control_coordinates == shadow_coordinates,
        "query_broad_strict_equal": sparse_equal,
        "route_and_response_coordinates_equal": control_coordinates
        == [(item["ordinal"], item["turn"]) for item in control_responses],
        "official_result_hash_equal": (
            control.get("official_result_sha256")
            == shadow.get("official_result_sha256")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "control_response_sha256": control_response_hash,
        "shadow_response_sha256": shadow_response_hash,
        "control_routes": control_routes,
        "shadow_routes": shadow_routes,
    }


def integrity_gates(
    control: dict[str, Any], shadow: dict[str, Any], catalog_ids: set[str]
) -> dict[str, Any]:
    from starter.response_contract import validate_response

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for role, bundle in ((C00, control), (S00, shadow)):
        routes = normalize_route_records(bundle.get("route_records"))
        responses = normalize_response_records(bundle.get("response_records"))
        stats = bundle.get("semantic_stats")
        if not isinstance(stats, Mapping):
            raise P7RunnerError("P7 semantic stats are missing")
        empty_count = sum(bool(record["empty_query"]) for record in routes)
        nonempty_count = len(routes) - empty_count
        prefix = "control" if role == C00 else "shadow"
        contract_errors = [
            error
            for record in responses
            for error in validate_response(record["response"], catalog_ids)
        ]
        checks[f"{prefix}_strict_response_contract"] = not contract_errors
        checks[f"{prefix}_lab_integrity_errors_empty"] = bundle.get("lab_integrity_errors") == []
        checks[f"{prefix}_capture_exception_count_zero"] = stats.get("capture_exception_count") == 0
        checks[f"{prefix}_integrity_error_count_zero"] = stats.get("integrity_error_count") == 0
        checks[f"{prefix}_semantic_exception_count_zero"] = stats.get("semantic_exception_count") == 0
        checks[f"{prefix}_generic_exception_count_zero"] = bundle.get("generic_exception_count") == 0
        checks[f"{prefix}_network_attempt_count_zero"] = bundle.get("network_attempt_count") == 0
        checks[f"{prefix}_route_count_exact"] = stats.get("route_record_count") == len(routes)
        checks[f"{prefix}_response_count_exact"] = stats.get("response_record_count") == len(responses)
        checks[f"{prefix}_empty_count_exact"] = stats.get("empty_query_count") == empty_count
        checks[f"{prefix}_coordinates_exact"] = [
            (record["ordinal"], record["turn"]) for record in routes
        ] == [(record["ordinal"], record["turn"]) for record in responses]
        stable_routes = [
            {key: value for key, value in record.items() if key != "query_search_ns"}
            for record in routes
        ]
        dense_routes = [
            {
                "ordinal": record["ordinal"], "turn": record["turn"],
                "query": record["query"], "dense": record["dense"],
            }
            for record in routes
        ]
        expected_lab_hashes = {
            "routes_sha256": canonical_sha256(stable_routes),
            "dense_routes_sha256": canonical_sha256(dense_routes),
            "responses_sha256": canonical_sha256(responses),
        }
        checks[f"{prefix}_lab_hashes_exact"] = bundle.get("lab_hashes") == expected_lab_hashes
        checks[f"{prefix}_all_route_ids_in_catalog"] = all(
            parent_asin in catalog_ids
            for record in routes
            for parent_asin in [
                *record["broad"],
                *record["strict"],
                *(entry["parent_asin"] for entry in record["dense"]),
            ]
        )
        checks[f"{prefix}_dense_order_exact"] = all(
            all(
                float.fromhex(left["score"]) > float.fromhex(right["score"])
                or (
                    float.fromhex(left["score"]) == float.fromhex(right["score"])
                    and left["parent_asin"] < right["parent_asin"]
                )
                for left, right in zip(record["dense"], record["dense"][1:])
            )
            for record in routes
        )
        if role == C00:
            checks["control_dense_call_count_zero"] = stats.get("dense_call_count") == 0
            checks["control_all_dense_routes_empty"] = all(not record["dense"] for record in routes)
        else:
            checks["shadow_dense_call_count_exact"] = stats.get("dense_call_count") == nonempty_count
            checks["shadow_empty_routes_dense_zero"] = all(
                not record["dense"] for record in routes if record["empty_query"]
            )
            checks["shadow_nonempty_routes_dense120"] = all(
                len(record["dense"]) == 120
                for record in routes
                if not record["empty_query"]
            )
        details[prefix] = {
            "route_count": len(routes),
            "response_count": len(responses),
            "empty_query_count": empty_count,
            "nonempty_query_count": nonempty_count,
            "response_contract_error_count": len(contract_errors),
        }
    return {"passed": all(checks.values()), "checks": checks, "details": details}


def dense_canonical_records(
    records: list[dict[str, Any]], eligible: set[tuple[int, int]]
) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": int(record["ordinal"]),
            "turn": int(record["turn"]),
            "query": str(record["query"]),
            "dense": list(record["dense"]),
        }
        for record in records
        if (int(record["ordinal"]), int(record["turn"])) in eligible
    ]


def _eligible_coordinates(
    samples: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    routes: list[dict[str, Any]],
) -> set[tuple[int, int]]:
    from evaluator.local_evaluator import materialize_hidden_fields

    by_ordinal = {index + 1: sample for index, sample in enumerate(samples)}
    eligible: set[tuple[int, int]] = set()
    for record in routes:
        ordinal = int(record["ordinal"])
        turn = int(record["turn"])
        query = str(record["query"])
        if not query:
            continue
        sample = by_ordinal.get(ordinal)
        if sample is None:
            raise P7RunnerError("route ordinal is outside the frozen corpus")
        if str(sample.get("scenario_type")) == "intent_override":
            behavior = materialize_hidden_fields(sample, products)[1]
            override = behavior.get("override")
            if not isinstance(override, dict) or not isinstance(override.get("turn"), int):
                raise P7RunnerError("intent-override sample lacks a frozen override turn")
            if turn < int(override["turn"]):
                continue
        eligible.add((ordinal, turn))
    return eligible


def posthoc_recall(
    samples: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    eligible: set[tuple[int, int]],
) -> dict[str, Any]:
    targets = {
        index + 1: str(sample["ground_truth"]["parent_asin"])
        for index, sample in enumerate(samples)
    }
    scenario_by_ordinal = {
        index + 1: str(sample["scenario_type"])
        for index, sample in enumerate(samples)
    }
    sparse_recalled: set[int] = set()
    dense_recalled_at = {10: set(), 40: set(), 120: set()}
    union_recalled: set[int] = set()
    for record in routes:
        coordinate = (int(record["ordinal"]), int(record["turn"]))
        if coordinate not in eligible:
            continue
        ordinal = coordinate[0]
        target = targets[ordinal]
        sparse = list(dict.fromkeys([*record["broad"], *record["strict"]]))
        dense = [entry["parent_asin"] for entry in record["dense"]]
        union = list(dict.fromkeys([*sparse, *dense]))
        if target in sparse:
            sparse_recalled.add(ordinal)
        for cutoff in dense_recalled_at:
            if target in dense[:cutoff]:
                dense_recalled_at[cutoff].add(ordinal)
        if target in union:
            union_recalled.add(ordinal)
    rescued = dense_recalled_at[120] - sparse_recalled
    scenarios = {scenario_by_ordinal[value] for value in rescued}
    return {
        "session_count": len(samples),
        "eligible_turn_count": len(eligible),
        "sparse_recalled_session_count": len(sparse_recalled),
        "sparse_union_broad120_strict80_recalled_session_count": len(sparse_recalled),
        "dense_recalled_session_count_at_k": {
            str(cutoff): len(dense_recalled_at[cutoff])
            for cutoff in (10, 40, 120)
        },
        "dense_union_recalled_session_count": len(union_recalled),
        "rescued_session_count": len(rescued),
        "rescued_scenario_type_count": len(scenarios),
        "strict_improvement": len(union_recalled) > len(sparse_recalled),
        "per_target_identifiers_recorded": False,
        "labels_joined_after_workers_exited": True,
    }


def resource_gates(
    spec: Mapping[str, Any], control: dict[str, Any], shadow: dict[str, Any], eligible: set[tuple[int, int]]
) -> dict[str, Any]:
    limits = spec["evaluation"]["resource_gates"]
    stats = shadow.get("semantic_stats", {})
    shadow_routes = normalize_route_records(shadow.get("route_records"))
    durations = [
        {
            "ordinal": record["ordinal"],
            "turn": record["turn"],
            "nanoseconds": record["query_search_ns"],
        }
        for record in shadow_routes
        if record["query_search_ns"] is not None
    ]
    p95_ms = _percentile95_ns(durations, eligible)
    control_wall = float(control.get("evaluation_wall_seconds", 0.0))
    shadow_wall = float(shadow.get("evaluation_wall_seconds", 0.0))
    control_rss = control.get("memory", {}).get("absolute_peak_rss_bytes")
    shadow_rss = shadow.get("memory", {}).get("absolute_peak_rss_bytes")
    wall_ratio = shadow_wall / control_wall if control_wall > 0 else None
    rss_ratio = (
        float(shadow_rss) / float(control_rss)
        if isinstance(control_rss, int) and control_rss > 0 and isinstance(shadow_rss, int)
        else None
    )
    observed = {
        "required_asset_bytes": stats.get("required_asset_bytes"),
        "cold_initialization_seconds": stats.get("cold_initialization_seconds"),
        "query_search_p95_milliseconds": p95_ms,
        "shadow_to_control_evaluation_wall_ratio": wall_ratio,
        "shadow_to_control_absolute_peak_rss_ratio": rss_ratio,
        "semantic_exception_count": stats.get("semantic_exception_count"),
        "network_attempt_count": shadow.get("network_attempt_count"),
    }
    checks = {
        "required_asset_bytes": isinstance(observed["required_asset_bytes"], int)
        and observed["required_asset_bytes"] <= limits["required_asset_bytes_max"],
        "cold_initialization_seconds": isinstance(observed["cold_initialization_seconds"], (int, float))
        and observed["cold_initialization_seconds"] <= limits["cold_initialization_seconds_max"],
        "query_search_p95_milliseconds": p95_ms is not None
        and p95_ms <= limits["query_search_p95_milliseconds_max"],
        "evaluation_wall_ratio": wall_ratio is not None
        and wall_ratio <= limits["shadow_to_control_evaluation_wall_ratio_max"],
        "absolute_peak_rss_ratio": rss_ratio is not None
        and rss_ratio <= limits["shadow_to_control_absolute_peak_rss_ratio_max"],
        "semantic_exceptions": observed["semantic_exception_count"]
        == limits["semantic_exception_count_max"],
        "network_attempts": observed["network_attempt_count"]
        == limits["network_attempt_count_max"],
        "rss_available": bool(control.get("memory", {}).get("available"))
        and bool(shadow.get("memory", {}).get("available")),
        "generic_worker_exceptions": control.get("generic_exception_count") == 0
        and shadow.get("generic_exception_count") == 0,
    }
    return {"passed": all(checks.values()), "observed": observed, "checks": checks}


def recall_gates(spec: Mapping[str, Any], recall: Mapping[str, Any]) -> dict[str, Any]:
    limits = spec["evaluation"]["recall_gates"]
    checks = {
        "minimum_rescued_sessions": recall["rescued_session_count"]
        >= limits["minimum_rescued_sessions"],
        "minimum_rescued_scenario_types": recall["rescued_scenario_type_count"]
        >= limits["minimum_rescued_scenario_types"],
        "strict_improvement": bool(recall["strict_improvement"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def initial_gates_pass(
    alignment: Mapping[str, Any],
    integrity: Mapping[str, Any],
    recall: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> bool:
    return all(
        gate.get("passed") is True
        for gate in (alignment, integrity, recall, resources)
    )


def validate_worker_processes(*bundles: Mapping[str, Any]) -> dict[str, Any]:
    nonces: list[str] = []
    checks: dict[str, bool] = {}
    for index, bundle in enumerate(bundles):
        process = bundle.get("worker_process")
        role = bundle.get("role")
        valid_object = isinstance(process, Mapping)
        checks[f"worker_{index + 1}_metadata"] = bool(
            valid_object
            and process.get("isolated") is True
            and process.get("role") == role
            and isinstance(process.get("nonce"), str)
            and re.fullmatch(r"[a-f0-9]{32}", str(process.get("nonce")))
        )
        if valid_object and isinstance(process.get("nonce"), str):
            nonces.append(str(process["nonce"]))
    checks["worker_nonces_distinct"] = len(nonces) == len(bundles) == len(set(nonces))
    return {"passed": all(checks.values()), "checks": checks, "worker_count": len(bundles)}


def repeatability_gates(
    control: dict[str, Any],
    initial_shadow: dict[str, Any],
    repeat_shadow: dict[str, Any],
    eligible: set[tuple[int, int]],
) -> dict[str, Any]:
    repeated_alignment = target_blind_alignment(control, repeat_shadow)
    repeat_routes = dense_canonical_records(
        repeated_alignment["shadow_routes"], eligible
    )
    initial_routes = dense_canonical_records(
        normalize_route_records(initial_shadow["route_records"]), eligible
    )
    checks = {
        "response_equals_control": repeated_alignment["passed"],
        "shadow_response_hash_equal": canonical_sha256(
            normalize_response_records(repeat_shadow["response_records"])
        )
        == canonical_sha256(
            normalize_response_records(initial_shadow["response_records"])
        ),
        "dense_route_hash_equal": canonical_sha256(repeat_routes)
        == canonical_sha256(initial_routes),
    }
    return {
        "run": True,
        "passed": all(checks.values()),
        "checks": checks,
        "hashes": {
            "control_response": canonical_sha256(
                normalize_response_records(control["response_records"])
            ),
            "initial_shadow_response": canonical_sha256(
                normalize_response_records(initial_shadow["response_records"])
            ),
            "repeat_shadow_response": canonical_sha256(
                normalize_response_records(repeat_shadow["response_records"])
            ),
            "initial_dense_routes": canonical_sha256(initial_routes),
            "repeat_dense_routes": canonical_sha256(repeat_routes),
        },
    }


def run_variant(
    role: str,
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
    *,
    catalog: Path,
    spec: Path,
    model_dir: Path,
    index_dir: Path,
    index_lock: Path,
    worker_factory: str = WORKER_FACTORY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from evaluator.local_evaluator import evaluate

    with tempfile.TemporaryDirectory(prefix="p7-worker-log-") as directory:
        stderr_path = Path(directory) / "stderr.log"
        worker = WorkerClient.start(
            role,
            catalog=catalog,
            spec=spec,
            model_dir=model_dir,
            index_dir=index_dir,
            index_lock=index_lock,
            worker_factory=worker_factory,
            stderr_path=stderr_path,
        )
        try:
            started = time.perf_counter()
            result = evaluate(worker, samples, catalog_ids, categories, products)
            wall = time.perf_counter() - started
            bundle = worker.finalize()
        except BaseException:
            worker.abort()
            raise
    bundle["evaluation_wall_seconds"] = wall
    bundle["official_result_sha256"] = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return bundle, result


def run_evaluation(
    *,
    catalog: Path,
    selection: Path,
    spec_path: Path,
    model_dir: Path,
    index_dir: Path,
    index_lock: Path,
    worker_factory: str = WORKER_FACTORY,
    enforce_frozen: bool = True,
) -> dict[str, Any]:
    from evaluator.local_evaluator import catalog_index, load_jsonl

    if enforce_frozen and worker_factory != WORKER_FACTORY:
        raise P7RunnerError("frozen P7 evaluation requires the registered worker factory")

    samples = load_jsonl(selection)
    frozen_inputs = (
        validate_frozen_inputs(
            catalog=catalog,
            selection=selection,
            spec_path=spec_path,
            index_lock=index_lock,
            model_dir=model_dir,
            samples=samples,
        )
        if enforce_frozen
        else {
            "catalog_sha256": _sha256_file(catalog),
            "selection_canonical_sha256": _canonical_samples_sha256(samples),
            "selection_count": len(samples),
        }
    )
    lock_summary = validate_index_lock(
        index_lock,
        project_root=PROJECT_ROOT,
        spec_path=spec_path,
        catalog_path=catalog,
        index_dir=index_dir,
        enforce_git=enforce_frozen,
    )
    spec = _load_json_object(spec_path)
    if enforce_frozen:
        from starter.p7_lab import validate_p7_index_lock
        from starter.semantic import validate_model_assets

        lab_validated_lock = validate_p7_index_lock(
            catalog,
            spec_path,
            index_dir,
            lock_path=index_lock,
            project_root=PROJECT_ROOT,
        )
        if lab_validated_lock != _load_json_object(index_lock):
            raise P7RunnerError("runner and p7_lab lock validation disagree")
        validate_model_assets(model_dir, spec)
    lock_object = _load_json_object(index_lock)
    locked_index_directory = _safe_project_path(
        PROJECT_ROOT, lock_object["index"]["directory"], "snapshot index directory"
    )
    direct_sources = [
        PROJECT_ROOT / "evaluator" / "__init__.py",
        PROJECT_ROOT / "evaluator" / "local_evaluator.py",
        PROJECT_ROOT / "starter" / "__init__.py",
        PROJECT_ROOT / "starter" / "agent.py",
        PROJECT_ROOT / "starter" / "attributes.py",
        PROJECT_ROOT / "starter" / "clarification.py",
        PROJECT_ROOT / "starter" / "coverage.py",
        PROJECT_ROOT / "starter" / "reranker.py",
        PROJECT_ROOT / "starter" / "slot_ledger.py",
        PROJECT_ROOT / "starter" / "response_contract.py",
    ]
    model_assets = [
        _safe_project_path(model_dir, entry["path"], "snapshot model asset")
        for entry in spec["required_files"]
    ]
    license_asset = _safe_project_path(
        PROJECT_ROOT, spec["model"]["license_notice"], "snapshot model license"
    )
    snapshot_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "p7_worker.py",
        PROJECT_ROOT / "starter" / "p7_lab.py",
        *direct_sources,
        *model_assets,
        license_asset,
        selection.resolve(),
        *(
            _safe_project_path(PROJECT_ROOT, entry["path"], "snapshot lock entry")
            for entry in (
                lock_object["source"]["builder"],
                lock_object["source"]["semantic"],
                lock_object["model_spec"],
                lock_object["catalog"],
            )
        ),
        *(
            _safe_project_path(locked_index_directory, entry["path"], "snapshot index entry")
            for entry in (
                lock_object["index"]["manifest"],
                lock_object["index"]["matrix"],
                lock_object["index"]["ordered_asins"],
            )
        ),
        index_lock.resolve(),
    ]
    before_snapshot = (
        capture_source_snapshot(PROJECT_ROOT, snapshot_paths)
        if enforce_frozen
        else None
    )
    catalog_ids, categories, products = catalog_index(catalog)
    control, _ = run_variant(
        C00, samples, catalog_ids, categories, products,
        catalog=catalog, spec=spec_path, model_dir=model_dir, index_dir=index_dir,
        index_lock=index_lock,
        worker_factory=worker_factory,
    )
    shadow, _ = run_variant(
        S00, samples, catalog_ids, categories, products,
        catalog=catalog, spec=spec_path, model_dir=model_dir, index_dir=index_dir,
        index_lock=index_lock,
        worker_factory=worker_factory,
    )
    initial_workers = validate_worker_processes(control, shadow)
    alignment = target_blind_alignment(control, shadow)
    eligible = _eligible_coordinates(samples, products, alignment["shadow_routes"])
    recall = posthoc_recall(samples, alignment["shadow_routes"], eligible)
    recall_gate = recall_gates(spec, recall)
    resource_gate = resource_gates(spec, control, shadow, eligible)
    integrity_gate = integrity_gates(control, shadow, catalog_ids)
    initial_passed = initial_gates_pass(
        alignment, integrity_gate, recall_gate, resource_gate
    ) and initial_workers["passed"]
    repeat: dict[str, Any] | None = None
    repeat_gate: dict[str, Any] = {"run": False, "passed": False, "checks": {}}
    if initial_passed:
        repeat, _ = run_variant(
            S00, samples, catalog_ids, categories, products,
            catalog=catalog, spec=spec_path, model_dir=model_dir, index_dir=index_dir,
            index_lock=index_lock,
            worker_factory=worker_factory,
        )
        repeat_gate = repeatability_gates(control, shadow, repeat, eligible)
        repeat_integrity = integrity_gates(control, repeat, catalog_ids)
        repeat_gate["integrity_gates"] = repeat_integrity
        repeat_gate["passed"] = repeat_gate["passed"] and repeat_integrity["passed"]
        repeat_workers = validate_worker_processes(control, shadow, repeat)
        repeat_gate["worker_processes"] = repeat_workers
        repeat_gate["passed"] = repeat_gate["passed"] and repeat_workers["passed"]
    dense_records = dense_canonical_records(alignment["shadow_routes"], eligible)
    after_snapshot = (
        capture_source_snapshot(PROJECT_ROOT, snapshot_paths)
        if enforce_frozen
        else None
    )
    if before_snapshot != after_snapshot:
        raise P7RunnerError("P7 source/input snapshot changed during evaluation")
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "catalog_sha256": _sha256_file(catalog),
            "selection_recorded": False,
            "selection_sample_count": len(samples),
            "model_spec_sha256": hashlib.sha256(
                json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "semantic_index_lock": lock_summary,
            "frozen_inputs": frozen_inputs,
            "source_snapshot": before_snapshot,
            "worker_factory": worker_factory,
        },
        "initial": {
            "passed": initial_passed,
            "alignment": {key: value for key, value in alignment.items() if not key.endswith("_routes")},
            "recall": recall,
            "integrity_gates": integrity_gate,
            "recall_gates": recall_gate,
            "resource_gates": resource_gate,
            "worker_processes": initial_workers,
            "dense_route_sha256": canonical_sha256(dense_records),
            "control": {key: value for key, value in control.items() if key not in {"response_records", "route_records"}},
            "shadow": {key: value for key, value in shadow.items() if key not in {"response_records", "route_records"}},
        },
        "repeatability": repeat_gate,
        "decision": "permit_fresh_p8_preregistration" if initial_passed and repeat_gate["passed"] else "reject_p7_bge",
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"P7 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False,
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.rename(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen target-blind P7 study")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--index-lock", type=Path, default=DEFAULT_INDEX_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"P7 output already exists: {args.output}")
    result = run_evaluation(
        catalog=args.catalog,
        selection=args.selection,
        spec_path=args.spec,
        model_dir=args.model_dir,
        index_dir=args.index_dir,
        index_lock=args.index_lock,
    )
    _atomic_write_json(args.output, result)
    print(f"[p7] decision={result['decision']} wrote={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
