from __future__ import annotations

"""Run the frozen, target-blind P9 selection and confirmation protocol."""

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from scripts.verify_official_assets import (  # noqa: E402
    EXPECTED_CATALOG_SHA256,
    EXPECTED_EVALUATOR_BLOB,
    EXPECTED_PUBLIC_BLOB,
    git_blob_sha1,
)
from starter.response_contract import ContractRecorder  # noqa: E402


SCHEMA_VERSION = "p9.compact-negative-evaluation.v1"
SPEC_SCHEMA_VERSION = "p9.compact-negative-matrix.v1"
LOCK_SCHEMA_VERSION = "p9.prereg-lock.v1"
WORKER_LOCK_SCHEMA_VERSION = "p9.worker-lock.v1"
WORKER_SPEC_SCHEMA_VERSION = "p9.worker-spec.v1"
METADATA_SCHEMA_VERSION = "p9.explicit-negative-corpora.v1"
EVIDENCE_SCHEMA_VERSION = "p9.compact-negative-evidence.v1"
EVIDENCE_MAX_BYTES = 16_777_216
EXPECTED_ORIGIN_URL = "https://github.com/lamperriat/techjam-err402.git"
EXPECTED_ORIGIN_URL_SHA256 = "bc84c0d712a5e9f381a0738891e461982e3cb0c52bc9d5e601c6abcbf0f860e5"
DEFAULT_SPEC = PROJECT_ROOT / "configs" / "p9_compact_negative_matrix.json"
DEFAULT_LOCK = PROJECT_ROOT / "configs" / "p9_prereg_lock.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "p9_compact_negative_evaluation.json"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "catalog.jsonl"
DEFAULT_PUBLIC = PROJECT_ROOT / "data" / "public_set.jsonl"
DEFAULT_EVIDENCE = PROJECT_ROOT / "experiments" / "p9_negative_evidence.sqlite"
DEFAULT_EVIDENCE_METADATA = PROJECT_ROOT / "experiments" / "p9_negative_evidence.metadata.json"
DEFAULT_METADATA = PROJECT_ROOT / "experiments" / "p9_explicit_negative_corpora.metadata.json"
DEFAULT_CORPORA = {
    "selection": PROJECT_ROOT / "experiments" / "p9_selection_product_disjoint.jsonl",
    "confirmation": PROJECT_ROOT / "experiments" / "p9_confirmation_product_disjoint.jsonl",
}
DEFAULT_PRIORS = {
    "p1": PROJECT_ROOT / "experiments" / "p1_derived_product_disjoint.jsonl",
    "p5": PROJECT_ROOT / "experiments" / "p5_selection_product_disjoint.jsonl",
    "p6": PROJECT_ROOT / "experiments" / "p6_selection_product_disjoint.jsonl",
    "p7": PROJECT_ROOT / "experiments" / "p7_selection_product_disjoint.jsonl",
    "p8_selection": PROJECT_ROOT / "experiments" / "p8_selection_product_disjoint.jsonl",
    "p8_confirmation": PROJECT_ROOT / "experiments" / "p8_confirmation_product_disjoint.jsonl",
}
SNAPSHOT_INPUT_FILENAMES = {
    "catalog": "catalog.jsonl",
    "evidence": "negative-evidence.sqlite",
    "released_public": "released-public.jsonl",
    "prior_p1": "prior-p1.jsonl",
    "prior_p5": "prior-p5.jsonl",
    "prior_p6": "prior-p6.jsonl",
    "prior_p7": "prior-p7.jsonl",
    "prior_p8_selection": "prior-p8-selection.jsonl",
    "prior_p8_confirmation": "prior-p8-confirmation.jsonl",
    "corpus_metadata": "corpus-metadata.json",
    "corpus_selection": "p9-selection.jsonl",
    "corpus_confirmation": "p9-confirmation.jsonl",
}
ROLES = {
    "control": "P9.C00.r08_coverage",
    "shadow": "P9.S00.compact_negative_shadow",
    "active": "P9.R01.compact_negative_partition",
}
BASELINE_ROLE = "P9.B00.served_agent"
ROLE_ORDER = tuple(ROLES.values())
WORKER_ROLES = (BASELINE_ROLE, *ROLE_ORDER)
SCENARIOS = {"boundary", "browsing", "buying", "intent_override"}
RR_SCALE = 2520
CONTRIBUTION_SCALE = 25_200
REQUIRED_SOURCE_PATHS = {
    "builder": "scripts/build_p9_selection_corpus.py",
    "p8_builder": "scripts/build_p8_selection_corpus.py",
    "evidence_builder": "scripts/build_p9_evidence.py",
    "lock_builder": "scripts/build_p9_prereg_lock.py",
    "starter_init": "starter/__init__.py",
    "p8_negative": "starter/p8_negative.py",
    "p9_evidence": "starter/p9_evidence.py",
    "p9_lab": "starter/p9_lab.py",
    "p9_worker": "scripts/p9_worker.py",
    "evaluate_p9": "scripts/evaluate_p9.py",
    "verify_official_assets": "scripts/verify_official_assets.py",
    "agent": "starter/agent.py",
    "coverage": "starter/coverage.py",
    "attributes": "starter/attributes.py",
    "slot_ledger": "starter/slot_ledger.py",
    "clarification": "starter/clarification.py",
    "reranker": "starter/reranker.py",
    "response_contract": "starter/response_contract.py",
    "evaluator": "evaluator/local_evaluator.py",
}
REQUIRED_SOURCE_NAMES = frozenset(REQUIRED_SOURCE_PATHS)
WORKER_RUNTIME_SOURCE_NAMES = frozenset(
    {
        "starter_init",
        "p8_negative",
        "p9_evidence",
        "p9_lab",
        "p9_worker",
        "agent",
        "coverage",
        "attributes",
        "slot_ledger",
        "clarification",
        "reranker",
    }
)
CANDIDATE_RUNTIME_SOURCE_NAMES = WORKER_RUNTIME_SOURCE_NAMES - {"p9_worker"}
FORBIDDEN_CANDIDATE_IMPORTS = frozenset(
    {
        "builtins",
        "ctypes",
        "importlib",
        "marshal",
        "mmap",
        "multiprocessing",
        "pickle",
        "socket",
        "subprocess",
        "winreg",
    }
)
FORBIDDEN_DYNAMIC_NAMES = frozenset(
    {"__import__", "compile", "eval", "exec", "globals", "locals"}
)
MAX_WORKER_MESSAGE_BYTES = 1_048_576
MAX_ARTIFACT_BYTES = 4_194_304
MAX_ARTIFACT_DEPTH = 12
MAX_ARTIFACT_ITEMS = 20_000
MAX_ARTIFACT_STRING_BYTES = 4_096
ASIN_SHAPE = re.compile(r"[A-Z0-9]{10}")
WORKER_FORBIDDEN_WORDS = (
    "ground_truth",
    "target",
    "sample_id",
    "scenario",
    "results",
    "label",
    "evaluator",
    "selection",
    "confirmation",
    "public_set",
)
ARTIFACT_FORBIDDEN_KEYS = {
    "ground_truth",
    "target",
    "target_id",
    "target_asin",
    "sample_id",
    "session_id",
    "sessions",
    "route_records",
    "response_records",
    "audit_records",
    "raw_routes",
    "_sessions",
    "parent_asin",
    "recommendations",
    "message",
    "ask_attribute",
    "user_profile",
    "intent_card",
    "behavior",
    "user_message",
}


class P9RunnerError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if any(not isinstance(row, dict) for row in rows):
        raise P9RunnerError(f"JSONL rows must be objects: {path}")
    return rows


def _canonical_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_bytes(row) + b"\n")
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P9RunnerError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise P9RunnerError(f"JSON root must be an object: {path}")
    return value


def _safe_project_path(project_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise P9RunnerError(f"{label} path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise P9RunnerError(f"{label} path is not safely project-relative")
    root = project_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise P9RunnerError(f"{label} path escapes project root") from exc
    return path


def _validate_hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(rf"[a-f0-9]{{{length}}}", value) is None:
        raise P9RunnerError(f"{label} must be {length} lowercase hexadecimal characters")
    return value


def _validate_file_entry(
    project_root: Path,
    entry: Any,
    label: str,
    *,
    extras: set[str] = frozenset(),
) -> tuple[Path, dict[str, Any]]:
    expected_keys = {"path", "bytes", "sha256", *extras}
    if not isinstance(entry, dict) or set(entry) != expected_keys:
        raise P9RunnerError(f"{label} has an invalid strict file schema")
    path = _safe_project_path(project_root, entry.get("path"), label)
    size = entry.get("bytes")
    sha = _validate_hex(entry.get("sha256"), 64, f"{label} SHA-256")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise P9RunnerError(f"{label} bytes must be a positive integer")
    if not path.is_file():
        raise P9RunnerError(f"{label} is missing: {path}")
    if path.stat().st_size != size or _sha256_file(path) != sha:
        raise P9RunnerError(f"{label} does not match its frozen bytes/SHA-256")
    return path, dict(entry)


def validate_matrix_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    roots = {
        "schema_version", "worker_factory", "roles", "served_control", "mechanism",
        "resource_limits", "promotion_gates",
    }
    if set(spec) != roots or spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise P9RunnerError("P9 matrix spec has an invalid strict root schema")
    if spec.get("worker_factory") != "starter.p9_lab:create_p9_agent":
        raise P9RunnerError("P9 matrix spec has an unexpected worker factory")
    if spec.get("roles") != ROLES:
        raise P9RunnerError("P9 matrix spec role registry is not frozen")
    served = {"retrieval_mode": "coverage", "rerank_mode": "off", "question_policy": "fast"}
    if spec.get("served_control") != served:
        raise P9RunnerError("P9 C00 is not the exact served coverage/off/fast control")
    mechanism = spec.get("mechanism")
    mechanism_keys = {
        "candidate_pool", "executable_constraint", "evidence_asset",
        "product_evidence_min_confidence", "product_description_is_evidence",
        "candidate_states", "ordering", "tie_break", "no_executable_constraint",
        "tail_policy",
    }
    if not isinstance(mechanism, dict) or set(mechanism) != mechanism_keys:
        raise P9RunnerError("P9 compact-negative mechanism has an invalid strict schema")
    constraint = mechanism.get("executable_constraint")
    expected_constraint = {
        "status": "active",
        "polarity": -1,
        "hardness": "hard",
        "confidence": 1.0,
        "source": "excluded_terms",
        "version": "current_goal_only",
        "value": "single_normalized_ascii_token",
        "allowed_slots": ["audience", "closure", "color", "material", "style", "use_case"],
    }
    evidence = mechanism.get("evidence_asset")
    if (
        mechanism.get("candidate_pool") != 50
        or mechanism.get("product_evidence_min_confidence") != 0.9
        or mechanism.get("product_description_is_evidence") is not False
        or mechanism.get("candidate_states") != ["compatible", "unknown", "explicit_violation"]
        or constraint != expected_constraint
        or not isinstance(evidence, dict)
        or set(evidence) != {
            "schema_version", "registry_sha256", "semantics_sha256", "catalog_only",
            "label_free", "maximum_bytes",
        }
        or evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("catalog_only") is not True
        or evidence.get("label_free") is not True
        or evidence.get("maximum_bytes") != EVIDENCE_MAX_BYTES
    ):
        raise P9RunnerError("P9 compact evidence mechanism is not frozen")
    _validate_hex(evidence.get("registry_sha256"), 64, "evidence registry SHA-256")
    _validate_hex(evidence.get("semantics_sha256"), 64, "evidence semantics SHA-256")
    limits = spec.get("resource_limits")
    expected_limits = {
        "bootstrap_ratio": 1.2,
        "wall_ratio": 1.3,
        "response_p95_ratio": 1.3,
        "peak_rss_ratio": 1.2,
        "evidence_asset_max_bytes": EVIDENCE_MAX_BYTES,
        "rss_sample_ms": 10.0,
        "bootstrap_timeout_seconds": 120.0,
        "request_timeout_seconds": 30.0,
        "finalize_timeout_seconds": 30.0,
        "exit_timeout_seconds": 10.0,
        "cumulative_worker_io_timeout_seconds": 180.0,
    }
    if limits != expected_limits:
        raise P9RunnerError("P9 resource limits are not frozen")
    expected_gates = {
        "hit_rate_non_decrease", "mrr_strict_increase", "mttc_non_increase",
        "technical_score_strict_increase", "scenario_hit_rate_non_decrease",
        "zero_hit_to_miss", "repeat_exact",
    }
    gates = spec.get("promotion_gates")
    if not isinstance(gates, dict) or set(gates) != expected_gates or not all(
        value is True for value in gates.values()
    ):
        raise P9RunnerError("P9 promotion gates are not strict and frozen")
    return json.loads(_canonical_bytes(spec))


def _git(project_root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.resolve().as_posix()}", *arguments],
            cwd=project_root,
            capture_output=True,
            text=not binary,
            check=False,
            timeout=30,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        raise P9RunnerError(f"Git command timed out: {' '.join(arguments)}") from exc
    if completed.returncode:
        raise P9RunnerError(f"Git command failed: {' '.join(arguments)}")
    return completed.stdout if binary else str(completed.stdout).strip()


def _assert_tracked(project_root: Path, path: Path, label: str) -> None:
    relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    _git(project_root, "ls-files", "--error-unmatch", relative)


def validate_worker_source_boundary(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    for word in WORKER_FORBIDDEN_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", source, re.IGNORECASE):
            raise P9RunnerError(f"P9 worker contains parent-only vocabulary: {word}")
    tree = ast.parse(source)
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    if "evaluator" in imported:
        raise P9RunnerError("P9 worker imports a parent-only package")
    return {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "ast_sha256": _stable_sha256(ast.dump(tree, include_attributes=False)),
        "forbidden_vocabulary_absent": True,
        "parent_only_import_absent": True,
    }


def validate_candidate_source_safety(
    project_root: Path, source_files: Mapping[str, Any]
) -> dict[str, Any]:
    if not CANDIDATE_RUNTIME_SOURCE_NAMES.issubset(source_files):
        raise P9RunnerError("P9 staged candidate source registry is incomplete")
    proof: list[dict[str, Any]] = []
    for name in sorted(CANDIDATE_RUNTIME_SOURCE_NAMES):
        entry = source_files[name]
        path = _safe_project_path(project_root, entry["path"], f"candidate source {name}")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imports.update(
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        if imports & FORBIDDEN_CANDIDATE_IMPORTS:
            raise P9RunnerError(f"P9 candidate source imports a forbidden module: {name}")
        if any(
            isinstance(node, ast.Name) and node.id in FORBIDDEN_DYNAMIC_NAMES
            for node in ast.walk(tree)
        ):
            raise P9RunnerError(f"P9 candidate source uses dynamic execution: {name}")
        if any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            and node.attr == "modules"
            for node in ast.walk(tree)
        ):
            raise P9RunnerError(f"P9 candidate source accesses the module registry: {name}")
        if any(
            isinstance(node, ast.Constant) and node.value == "__main__"
            for node in ast.walk(tree)
        ):
            raise P9RunnerError(f"P9 candidate source references a dynamic entrypoint: {name}")
        proof.append(
            {
                "name": name,
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "ast_sha256": _stable_sha256(ast.dump(tree, include_attributes=False)),
            }
        )
    return {
        "schema_version": "p9.candidate-direct-ast-scan.v1",
        "source_file_count": len(proof),
        "source_registry_sha256": _stable_sha256(proof),
        "direct_dangerous_imports_absent": True,
        "direct_dynamic_names_absent": True,
        "direct_sys_modules_access_absent": True,
        "direct_main_entrypoint_literal_absent": True,
        "hostile_native_code_sandboxed": False,
    }


def _validate_source_lock(
    project_root: Path,
    source: Any,
    *,
    enforce_git: bool,
) -> dict[str, Any]:
    if not isinstance(source, dict) or set(source) != {
        "git_commit",
        "git_branch",
        "remote_proof",
        "files",
    }:
        raise P9RunnerError("P9 source lock has an invalid strict schema")
    commit = _validate_hex(source.get("git_commit"), 40, "source commit")
    branch = source.get("git_branch")
    remote_proof = source.get("remote_proof")
    files = source.get("files")
    if (
        not isinstance(branch, str)
        or not branch
        or not isinstance(files, dict)
        or not isinstance(remote_proof, dict)
        or set(remote_proof)
        != {"remote", "head_ref", "advertised_head", "url_sha256", "verified"}
        or remote_proof.get("remote") not in {"origin", "fixture"}
        or remote_proof.get("head_ref") != f"refs/heads/{branch}"
        or _validate_hex(
            remote_proof.get("advertised_head"), 40, "source remote advertised head"
        )
        != commit
        or _validate_hex(remote_proof.get("url_sha256"), 64, "source origin URL SHA-256")
        not in {EXPECTED_ORIGIN_URL_SHA256, "0" * 64}
        or not isinstance(remote_proof.get("verified"), bool)
    ):
        raise P9RunnerError("P9 source revision is incomplete")
    if set(files) != set(REQUIRED_SOURCE_NAMES):
        missing = sorted(REQUIRED_SOURCE_NAMES - set(files))
        extra = sorted(set(files) - REQUIRED_SOURCE_NAMES)
        raise P9RunnerError(f"P9 source lock registry differs; missing={missing}, extra={extra}")
    identities: dict[str, Any] = {}
    for name, entry in files.items():
        path, frozen = _validate_file_entry(
            project_root, entry, f"source {name}", extras={"git_blob_sha1"}
        )
        declared_blob = _validate_hex(frozen["git_blob_sha1"], 40, f"source {name} Git blob")
        if path.resolve() != (project_root / REQUIRED_SOURCE_PATHS[name]).resolve():
            raise P9RunnerError(f"source {name} does not use its canonical project path")
        identities[name] = frozen
        if enforce_git:
            _assert_tracked(project_root, path, f"source {name}")
            relative = path.resolve().relative_to(project_root.resolve()).as_posix()
            working_blob = str(_git(project_root, "hash-object", f"--path={relative}", "--", relative))
            commit_blob = str(_git(project_root, "rev-parse", f"{commit}:{relative}"))
            if working_blob != declared_blob or commit_blob != declared_blob:
                raise P9RunnerError(f"source {name} differs from the preregistered Git blob")
    if enforce_git:
        origin_url = str(_git(project_root, "remote", "get-url", "origin"))
        if (
            remote_proof.get("remote") != "origin"
            or remote_proof.get("verified") is not True
            or remote_proof.get("url_sha256") != EXPECTED_ORIGIN_URL_SHA256
            or origin_url != EXPECTED_ORIGIN_URL
            or hashlib.sha256(origin_url.encode("utf-8")).hexdigest()
            != EXPECTED_ORIGIN_URL_SHA256
        ):
            raise P9RunnerError("P9 source lock lacks a formal direct remote proof")
        if _git(project_root, "branch", "--show-current") != branch:
            raise P9RunnerError("current branch differs from the P9 source lock")
        completed = subprocess.run(
            [
                "git", "-c", f"safe.directory={project_root.resolve().as_posix()}",
                "merge-base", "--is-ancestor", commit, "HEAD",
            ],
            cwd=project_root,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise P9RunnerError("P9 preregistration commit is not an ancestor of HEAD")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "remote_proof": dict(remote_proof),
        "files": identities,
    }


def _validate_official_evaluator(path: Path) -> str:
    frozen = "7c808347b31ef3121a9cbc4810ac3eb325f950ba"
    if EXPECTED_EVALUATOR_BLOB != frozen or git_blob_sha1(path) != frozen:
        raise P9RunnerError("P9 source lock does not bind the official evaluator blob")
    return frozen


def _validate_source_target_scan(
    value: Any, source: Mapping[str, Any], identifier_count: int
) -> dict[str, Any]:
    keys = {
        "source_file_count",
        "identifier_count",
        "match_count",
        "passed",
        "proof_sha256",
    }
    files = source.get("files")
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or not isinstance(files, Mapping)
        or value.get("source_file_count") != len(files)
        or value.get("identifier_count") != identifier_count
        or value.get("match_count") != 0
        or value.get("passed") is not True
    ):
        raise P9RunnerError("P9 source identifier scan proof is invalid")
    proof = {
        "source_files": [
            {
                "name": name,
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
            }
            for name, entry in sorted(files.items())
        ],
        "source_file_count": len(files),
        "identifier_count": identifier_count,
        "match_count": 0,
    }
    if value.get("proof_sha256") != _stable_sha256(proof):
        raise P9RunnerError("P9 source identifier scan digest differs from the lock")
    return dict(value)


def _overlap_values(value: Any, key: str = "") -> list[Any]:
    if isinstance(value, Mapping):
        found: list[Any] = []
        for nested_key, nested in value.items():
            found.extend(_overlap_values(nested, str(nested_key)))
        return found
    return [value] if "overlap" in key.lower() else []


def _validate_metadata(metadata: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise P9RunnerError("P9 corpus metadata schema is not frozen")
    catalog = metadata.get("catalog_source")
    inputs = metadata.get("input_sources")
    corpora = metadata.get("corpora")
    exclusions = metadata.get("exclusions")
    outputs = metadata.get("outputs")
    if not all(isinstance(value, dict) for value in (catalog, inputs, corpora, exclusions, outputs)):
        raise P9RunnerError("P9 corpus metadata sections are incomplete")
    assert isinstance(catalog, dict) and isinstance(inputs, dict)
    assert isinstance(corpora, dict) and isinstance(outputs, dict)
    if (
        catalog.get("sha256") != lock["catalog"]["sha256"]
        or catalog.get("loaded_product_count") != lock["catalog"]["rows"]
        or catalog.get("frozen_sha256_verified") is not True
        or catalog.get("expected_count_verified") is not True
    ):
        raise P9RunnerError("P9 metadata catalog identity differs from the lock")
    input_names = {
        "released_public": "released_public",
        "prior_p1_derived": "p1",
        "prior_p5_derived": "p5",
        "prior_p6_derived": "p6",
        "prior_p7_derived": "p7",
        "prior_p8_selection": "p8_selection",
        "prior_p8_confirmation": "p8_confirmation",
    }
    if set(inputs) != set(input_names):
        raise P9RunnerError("P9 metadata input registry is incomplete")
    for metadata_name, lock_name in input_names.items():
        entry = inputs[metadata_name]
        locked = lock["released_public"] if lock_name == "released_public" else lock["priors"][lock_name]
        if (
            not isinstance(entry, dict)
            or entry.get("sample_count") != locked["rows"]
            or entry.get("unique_target_count") != locked["rows"]
            or entry.get("frozen_samples_sha256_verified") is not True
        ):
            raise P9RunnerError(f"P9 metadata identity failed for {metadata_name}")
        if metadata_name == "released_public" and (
            entry.get("git_blob_sha1_lf") != lock["released_public"]["git_blob_sha1_lf"]
            or entry.get("frozen_git_blob_verified") is not True
        ):
            raise P9RunnerError("P9 metadata public Git blob identity failed")
    if set(corpora) != {"selection", "confirmation"}:
        raise P9RunnerError("P9 metadata split registry is incomplete")
    for split in ("selection", "confirmation"):
        entry = corpora[split]
        locked = lock["corpora"][split]
        if (
            not isinstance(entry, dict)
            or entry.get("sample_count") != locked["rows"]
            or entry.get("unique_target_count") != locked["rows"]
            or entry.get("samples_sha256") != locked["canonical_samples_sha256"]
            or entry.get("scenario_counts") != locked["scenario_counts"]
        ):
            raise P9RunnerError(f"P9 metadata {split} identity differs from the lock")
        output = outputs.get(split)
        if (
            not isinstance(output, dict)
            or output.get("expected_frozen_samples_sha256") != locked["canonical_samples_sha256"]
            or output.get("samples_file_sha256") != locked["canonical_samples_sha256"]
            or output.get("frozen_samples_sha256_verified") is not True
        ):
            raise P9RunnerError(f"P9 metadata {split} output is not self-frozen")
    overlaps = _overlap_values(exclusions)
    if not overlaps or any(value != 0 for value in overlaps):
        raise P9RunnerError("P9 metadata does not prove zero exclusion overlap")
    return {
        "schema_version": metadata["schema_version"],
        "catalog_verified": True,
        "input_registry_verified": True,
        "p8_exclusion_verified": True,
        "split_registry_verified": True,
        "zero_overlap_aggregates_verified": True,
    }


def _validate_protocol_lock(
    project_root: Path,
    lock: Mapping[str, Any],
    *,
    spec_path: Path,
    enforce_git: bool,
) -> dict[str, Any]:
    roots = {
        "schema_version", "source", "spec", "catalog", "released_public", "priors",
        "evidence", "evidence_metadata", "corpus_metadata", "corpora",
        "source_target_scan",
    }
    if set(lock) != roots or lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise P9RunnerError("P9 preregistration lock has an invalid strict root schema")
    source = _validate_source_lock(project_root, lock.get("source"), enforce_git=enforce_git)
    evaluator_path = _safe_project_path(
        project_root, source["files"]["evaluator"]["path"], "official evaluator"
    )
    _validate_official_evaluator(evaluator_path)
    locked_spec_path, spec_entry = _validate_file_entry(project_root, lock.get("spec"), "matrix spec")
    if locked_spec_path.resolve() != spec_path.resolve():
        raise P9RunnerError("requested P9 matrix spec differs from the lock")
    catalog_path, catalog_entry = _validate_file_entry(
        project_root, lock.get("catalog"), "catalog", extras={"rows"}
    )
    public_path, public_entry = _validate_file_entry(
        project_root,
        lock.get("released_public"),
        "released public",
        extras={"rows", "git_blob_sha1_lf"},
    )
    if catalog_entry["sha256"] != EXPECTED_CATALOG_SHA256 or catalog_entry["rows"] != 50_000:
        raise P9RunnerError("P9 lock does not bind the official 50,000-row catalog")
    if (
        _validate_hex(public_entry["git_blob_sha1_lf"], 40, "public Git blob") != EXPECTED_PUBLIC_BLOB
        or git_blob_sha1(public_path) != EXPECTED_PUBLIC_BLOB
    ):
        raise P9RunnerError("P9 lock does not bind the official public corpus")
    priors = lock.get("priors")
    if not isinstance(priors, dict) or set(priors) != set(DEFAULT_PRIORS):
        raise P9RunnerError("P9 prior-corpus lock registry is incomplete")
    prior_paths: dict[str, Path] = {}
    prior_entries: dict[str, Any] = {}
    for name, entry in priors.items():
        path, frozen = _validate_file_entry(project_root, entry, f"prior {name}", extras={"rows"})
        prior_paths[name] = path
        prior_entries[name] = frozen
    evidence_path, evidence_entry = _validate_file_entry(
        project_root,
        lock.get("evidence"),
        "compact evidence",
        extras={"schema_version", "registry_sha256", "semantics_sha256", "catalog_rows"},
    )
    spec_evidence = _load_json_object(spec_path)["mechanism"]["evidence_asset"]
    if (
        evidence_entry["bytes"] > EVIDENCE_MAX_BYTES
        or evidence_entry["schema_version"] != EVIDENCE_SCHEMA_VERSION
        or evidence_entry["registry_sha256"] != spec_evidence["registry_sha256"]
        or evidence_entry["semantics_sha256"] != spec_evidence["semantics_sha256"]
        or evidence_entry["catalog_rows"] != catalog_entry["rows"]
    ):
        raise P9RunnerError("P9 compact evidence identity or size gate failed")
    evidence_metadata_path, evidence_metadata_entry = _validate_file_entry(
        project_root, lock.get("evidence_metadata"), "compact evidence metadata"
    )
    evidence_metadata = _load_json_object(evidence_metadata_path)
    if (
        evidence_metadata.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence_metadata.get("target_blind") is not True
        or evidence_metadata.get("label_free") is not True
        or not isinstance(evidence_metadata.get("catalog"), dict)
        or evidence_metadata["catalog"].get("sha256") != catalog_entry["sha256"]
        or evidence_metadata["catalog"].get("rows") != catalog_entry["rows"]
        or not isinstance(evidence_metadata.get("evidence"), dict)
        or evidence_metadata["evidence"].get("bytes") != evidence_entry["bytes"]
        or evidence_metadata["evidence"].get("sha256") != evidence_entry["sha256"]
        or evidence_metadata["evidence"].get("registry_sha256") != evidence_entry["registry_sha256"]
        or evidence_metadata["evidence"].get("semantics_sha256") != evidence_entry["semantics_sha256"]
    ):
        raise P9RunnerError("P9 compact evidence metadata differs from the frozen asset")
    metadata_path, metadata_entry = _validate_file_entry(
        project_root, lock.get("corpus_metadata"), "corpus metadata"
    )
    corpora = lock.get("corpora")
    if not isinstance(corpora, dict) or set(corpora) != set(DEFAULT_CORPORA):
        raise P9RunnerError("P9 split lock registry is incomplete")
    corpus_paths: dict[str, Path] = {}
    corpus_entries: dict[str, Any] = {}
    for split, entry in corpora.items():
        path, frozen = _validate_file_entry(
            project_root,
            entry,
            f"{split} split",
            extras={"rows", "canonical_samples_sha256", "scenario_counts"},
        )
        _validate_hex(frozen["canonical_samples_sha256"], 64, f"{split} canonical SHA-256")
        counts = frozen["scenario_counts"]
        if (
            not isinstance(frozen["rows"], int)
            or frozen["rows"] <= 0
            or not isinstance(counts, dict)
            or set(counts) != SCENARIOS
            or any(not isinstance(value, int) or value <= 0 for value in counts.values())
            or sum(counts.values()) != frozen["rows"]
        ):
            raise P9RunnerError(f"P9 {split} count/scenario lock is invalid")
        corpus_paths[split] = path
        corpus_entries[split] = frozen
    if len({path.resolve() for path in corpus_paths.values()}) != 2:
        raise P9RunnerError("P9 selection and confirmation paths must differ")
    metadata_summary = _validate_metadata(_load_json_object(metadata_path), lock)
    identifier_count = public_entry["rows"] + sum(
        entry["rows"] for entry in prior_entries.values()
    ) + sum(entry["rows"] for entry in corpus_entries.values())
    source_target_scan = _validate_source_target_scan(
        lock.get("source_target_scan"), source, identifier_count
    )
    if enforce_git:
        _assert_tracked(project_root, spec_path, "matrix spec")
    return {
        "source": source,
        "source_target_scan": source_target_scan,
        "spec": spec_entry,
        "catalog": catalog_entry,
        "released_public": public_entry,
        "priors": prior_entries,
        "evidence": evidence_entry,
        "evidence_metadata": evidence_metadata_entry,
        "corpus_metadata": metadata_entry,
        "corpora": corpus_entries,
        "metadata_summary": metadata_summary,
        "paths": {
            "catalog": catalog_path,
            "released_public": public_path,
            "priors": prior_paths,
            "evidence": evidence_path,
            "evidence_metadata": evidence_metadata_path,
            "corpus_metadata": metadata_path,
            "corpora": corpus_paths,
        },
    }


def _git_snapshot(project_root: Path) -> dict[str, Any]:
    branch = str(_git(project_root, "branch", "--show-current"))
    head = str(_git(project_root, "rev-parse", "HEAD"))
    status = str(_git(project_root, "status", "--porcelain=v1", "--untracked-files=all"))
    if not branch or not head or status:
        raise P9RunnerError("P9 requires a named branch and completely clean worktree")
    origin_url = str(_git(project_root, "remote", "get-url", "origin"))
    origin_url_sha256 = hashlib.sha256(origin_url.encode("utf-8")).hexdigest()
    if origin_url != EXPECTED_ORIGIN_URL or origin_url_sha256 != EXPECTED_ORIGIN_URL_SHA256:
        raise P9RunnerError("P9 requires the credential-free official HTTPS origin")
    head_ref = f"refs/heads/{branch}"
    advertised = str(_git(project_root, "ls-remote", "--heads", "origin", head_ref))
    lines = [line for line in advertised.splitlines() if line.strip()]
    match = (
        re.fullmatch(rf"([a-f0-9]{{40}})\t{re.escape(head_ref)}", lines[0])
        if len(lines) == 1
        else None
    )
    if match is None or match.group(1) != head:
        raise P9RunnerError("P9 requires direct origin branch proof to equal HEAD")
    return {
        "branch": branch,
        "head": head,
        "remote": "origin",
        "head_ref": head_ref,
        "advertised_head": match.group(1),
        "origin_url_sha256": origin_url_sha256,
        "online_verified": True,
        "clean": True,
    }


def _identity_snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _flatten_protocol_paths(
    protocol: Mapping[str, Any], spec: Path, lock: Path, project_root: Path
) -> dict[str, Path]:
    paths = protocol["paths"]
    return {
        "spec": spec,
        "lock": lock,
        "catalog": paths["catalog"],
        "released_public": paths["released_public"],
        "evidence": paths["evidence"],
        "evidence_metadata": paths["evidence_metadata"],
        "corpus_metadata": paths["corpus_metadata"],
        **{f"prior_{name}": path for name, path in paths["priors"].items()},
        **{f"corpus_{name}": path for name, path in paths["corpora"].items()},
        **{
            f"source_{name}": _safe_project_path(project_root, entry["path"], f"source {name}")
            for name, entry in protocol["source"]["files"].items()
        },
    }


def preflight(
    *,
    spec_path: Path = DEFAULT_SPEC,
    lock_path: Path = DEFAULT_LOCK,
    project_root: Path = PROJECT_ROOT,
    enforce_git: bool = True,
    require_defaults: bool = True,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    spec_path = spec_path.resolve()
    lock_path = lock_path.resolve()
    if require_defaults and (
        spec_path != (project_root / "configs" / "p9_compact_negative_matrix.json").resolve()
        or lock_path != (project_root / "configs" / "p9_prereg_lock.json").resolve()
    ):
        raise P9RunnerError("formal P9 requires the default frozen spec and lock")
    if not spec_path.is_file() or not lock_path.is_file():
        raise P9RunnerError("P9 spec and preregistration lock must exist")
    spec = validate_matrix_spec(_load_json_object(spec_path))
    protocol = _validate_protocol_lock(
        project_root, _load_json_object(lock_path), spec_path=spec_path, enforce_git=enforce_git
    )
    worker_boundary = {
        "worker_source": validate_worker_source_boundary(
            project_root / "scripts" / "p9_worker.py"
        ),
        "candidate_sources": validate_candidate_source_safety(
            project_root, protocol["source"]["files"]
        ),
        "boundary_kind": "staged trusted Python with audit enforcement",
        "hostile_native_code_sandboxed": False,
    }
    if enforce_git:
        _assert_tracked(project_root, lock_path, "P9 preregistration lock")
        git = _git_snapshot(project_root)
    else:
        git = {
            "branch": None,
            "head": None,
            "remote": None,
            "head_ref": None,
            "advertised_head": None,
            "origin_url_sha256": None,
            "online_verified": None,
            "clean": None,
        }
    snapshot = _identity_snapshot(
        _flatten_protocol_paths(protocol, spec_path, lock_path, project_root)
    )
    return {
        "spec": spec,
        "protocol": protocol,
        "git": git,
        "worker_boundary": worker_boundary,
        "identity_snapshot": snapshot,
        "summary": {
            "schema_version": LOCK_SCHEMA_VERSION,
            "selection_rows": protocol["corpora"]["selection"]["rows"],
            "confirmation_rows": protocol["corpora"]["confirmation"]["rows"],
            "selection_sha256": protocol["corpora"]["selection"]["sha256"],
            "confirmation_sha256": protocol["corpora"]["confirmation"]["sha256"],
            "catalog_sha256": protocol["catalog"]["sha256"],
            "released_public_git_blob_sha1_lf": protocol["released_public"]["git_blob_sha1_lf"],
            "prior_sha256": {name: entry["sha256"] for name, entry in protocol["priors"].items()},
            "evidence": {
                "bytes": protocol["evidence"]["bytes"],
                "sha256": protocol["evidence"]["sha256"],
                "maximum_bytes": EVIDENCE_MAX_BYTES,
            },
            "confirmation_rows_parsed": False,
        },
    }


def _target_values(rows: list[dict[str, Any]], label: str) -> list[str]:
    values = [
        str(row.get("ground_truth", {}).get("parent_asin") or "").strip()
        for row in rows
    ]
    if not values or not all(values):
        raise P9RunnerError(f"{label} contains an empty ground-truth identifier")
    return values


def _snapshot_runtime_path(preflight_state: Mapping[str, Any], name: str) -> Path:
    snapshot = preflight_state.get("asset_snapshot")
    paths = snapshot.get("paths") if isinstance(snapshot, Mapping) else None
    path = paths.get(name) if isinstance(paths, Mapping) else None
    if name not in SNAPSHOT_INPUT_FILENAMES or not isinstance(path, Path):
        raise P9RunnerError("P9 conversation input is not backed by the frozen snapshot")
    return path


def _load_split(
    split: str,
    preflight_state: Mapping[str, Any],
    *,
    catalog_ids: set[str],
    excluded_targets: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    if split not in {"selection", "confirmation"}:
        raise P9RunnerError("unknown P9 split")
    protocol = preflight_state["protocol"]
    path = _snapshot_runtime_path(preflight_state, f"corpus_{split}")
    entry = protocol["corpora"][split]
    rows = _jsonl_rows(path)
    if len(rows) != entry["rows"] or _canonical_rows_sha256(rows) != entry["canonical_samples_sha256"]:
        raise P9RunnerError(f"P9 {split} parsed identity differs from its lock")
    identifiers = [str(row.get("sample_id") or "") for row in rows]
    targets = _target_values(rows, f"P9 {split}")
    counts = dict(sorted(Counter(str(row.get("scenario_type") or "") for row in rows).items()))
    if (
        len(set(identifiers)) != len(rows)
        or not all(identifiers)
        or len(set(targets)) != len(rows)
        or set(targets) - catalog_ids
        or set(targets) & excluded_targets
        or counts != entry["scenario_counts"]
    ):
        raise P9RunnerError(f"P9 {split} uniqueness, exclusion, or scenario gate failed")
    return rows, {
        "sample_count": len(rows),
        "unique_sample_count": len(set(identifiers)),
        "unique_target_count": len(set(targets)),
        "canonical_samples_sha256": entry["canonical_samples_sha256"],
        "scenario_counts": counts,
        "all_targets_in_catalog": True,
        "excluded_target_overlap": 0,
    }, set(targets)


def _prior_target_set(preflight_state: Mapping[str, Any]) -> set[str]:
    protocol = preflight_state["protocol"]
    sources = {
        "released_public": _snapshot_runtime_path(preflight_state, "released_public"),
        **{
            name: _snapshot_runtime_path(preflight_state, f"prior_{name}")
            for name in DEFAULT_PRIORS
        },
    }
    combined: set[str] = set()
    for name, path in sources.items():
        rows = _jsonl_rows(path)
        locked = protocol["released_public"] if name == "released_public" else protocol["priors"][name]
        targets = _target_values(rows, name)
        if len(rows) != locked["rows"] or len(set(targets)) != len(rows):
            raise P9RunnerError(f"P9 exclusion source {name} failed count/uniqueness")
        overlap = combined & set(targets)
        if overlap:
            raise P9RunnerError(f"P9 exclusion sources overlap at {name}")
        combined.update(targets)
    return combined


_SANITIZED_FORBIDDEN = (
    "corpus", "public", "prior", "ground_truth", "sample_id", "scenario", "target"
)


def _worker_spec_payload(preflight_state: Mapping[str, Any]) -> dict[str, Any]:
    spec = preflight_state["spec"]
    payload = {
        "schema_version": WORKER_SPEC_SCHEMA_VERSION,
        "protocol_spec_sha256": preflight_state["protocol"]["spec"]["sha256"],
        "roles": spec["roles"],
        "served_control": spec["served_control"],
        "mechanism": spec["mechanism"],
    }
    serialized = _canonical_bytes(payload).decode("utf-8").lower()
    if any(word in serialized for word in _SANITIZED_FORBIDDEN):
        raise P9RunnerError("sanitized worker spec contains parent-only material")
    return payload


def _worker_lock_payload(
    preflight_state: Mapping[str, Any], *, worker_spec_sha256: str
) -> dict[str, Any]:
    protocol = preflight_state["protocol"]
    payload = {
        "schema_version": WORKER_LOCK_SCHEMA_VERSION,
        "spec_sha256": _validate_hex(worker_spec_sha256, 64, "worker spec SHA-256"),
        "protocol_spec_sha256": protocol["spec"]["sha256"],
        "catalog_sha256": protocol["catalog"]["sha256"],
        "protocol_lock_sha256": preflight_state["identity_snapshot"]["lock"]["sha256"],
        "evidence": {
            "bytes": protocol["evidence"]["bytes"],
            "sha256": protocol["evidence"]["sha256"],
        },
        "roles": preflight_state["spec"]["roles"],
    }
    serialized = _canonical_bytes(payload).decode("utf-8").lower()
    if any(word in serialized for word in _SANITIZED_FORBIDDEN):
        raise P9RunnerError("sanitized worker lock contains parent-only material")
    return payload


def _is_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & 0x400)


def _stage_worker_runtime(
    preflight_state: Mapping[str, Any], stage_root: Path
) -> tuple[Path, str]:
    source = preflight_state["protocol"]["source"]["files"]
    if not WORKER_RUNTIME_SOURCE_NAMES.issubset(source):
        raise P9RunnerError("P9 worker runtime source lock is incomplete")
    manifest: dict[str, Any] = {}
    for name in sorted(WORKER_RUNTIME_SOURCE_NAMES):
        entry = source[name]
        expected_path = REQUIRED_SOURCE_PATHS[name]
        if not isinstance(entry, dict) or entry.get("path") != expected_path:
            raise P9RunnerError(f"P9 staged source path differs for {name}")
        live = _safe_project_path(PROJECT_ROOT, expected_path, f"worker source {name}")
        if _is_reparse_or_symlink(live):
            raise P9RunnerError(f"P9 worker source is not a regular frozen file: {name}")
        destination = _safe_project_path(stage_root, expected_path, f"staged source {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(live, destination)
        if (
            destination.stat().st_size != entry.get("bytes")
            or _sha256_file(destination) != entry.get("sha256")
        ):
            raise P9RunnerError(f"P9 staged source identity differs for {name}")
        destination.chmod(0o444)
        manifest[name] = {
            "path": expected_path,
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
    worker = stage_root / REQUIRED_SOURCE_PATHS["p9_worker"]
    return worker, _stable_sha256(manifest)


def _verify_worker_runtime(
    preflight_state: Mapping[str, Any],
    stage_root: Path,
    generated: Mapping[str, tuple[int, str]],
    scratch: Path,
) -> None:
    expected: dict[str, tuple[int, str]] = {
        preflight_state["protocol"]["source"]["files"][name]["path"]: (
            preflight_state["protocol"]["source"]["files"][name]["bytes"],
            preflight_state["protocol"]["source"]["files"][name]["sha256"],
        )
        for name in WORKER_RUNTIME_SOURCE_NAMES
    }
    expected.update(generated)
    seen: set[str] = set()
    for path in stage_root.rglob("*"):
        if _is_reparse_or_symlink(path):
            raise P9RunnerError("P9 staged runtime contains a link/reparse point")
        if path.is_dir():
            continue
        relative = path.relative_to(stage_root).as_posix()
        identity = expected.get(relative)
        if identity is None:
            raise P9RunnerError("P9 staged runtime contains an unexpected file")
        if path.stat().st_size != identity[0] or _sha256_file(path) != identity[1]:
            raise P9RunnerError("P9 staged runtime changed during execution")
        seen.add(relative)
    if seen != set(expected):
        raise P9RunnerError("P9 staged runtime is incomplete after execution")
    scratch_files = [path for path in scratch.rglob("*") if path.is_file()]
    if any(_is_reparse_or_symlink(path) for path in scratch.rglob("*")):
        raise P9RunnerError("P9 worker scratch contains a link/reparse point")
    if len(scratch_files) != 1 or scratch_files[0].name != "stderr.bin":
        raise P9RunnerError("P9 worker scratch contains an unexpected file")
    if scratch_files[0].stat().st_size != 0:
        raise P9RunnerError("P9 worker wrote unexpected diagnostics")


def _read_only(path: Path) -> bool:
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    if os.name == "nt":
        return bool(attributes & 0x1)
    return path.stat(follow_symlinks=False).st_mode & 0o222 == 0


def _create_asset_snapshot(
    preflight_state: Mapping[str, Any], snapshot_root: Path
) -> dict[str, Any]:
    snapshot_root.mkdir(parents=True, exist_ok=False)
    protocol = preflight_state["protocol"]
    protocol_paths = protocol["paths"]
    sources = {
        "catalog": protocol_paths["catalog"],
        "evidence": protocol_paths["evidence"],
        "released_public": protocol_paths["released_public"],
        "corpus_metadata": protocol_paths["corpus_metadata"],
        **{
            f"prior_{name}": path
            for name, path in protocol_paths["priors"].items()
        },
        **{
            f"corpus_{split}": path
            for split, path in protocol_paths["corpora"].items()
        },
    }
    locked_entries = {
        "catalog": protocol["catalog"],
        "evidence": protocol["evidence"],
        "released_public": protocol["released_public"],
        "corpus_metadata": protocol["corpus_metadata"],
        **{
            f"prior_{name}": entry
            for name, entry in protocol["priors"].items()
        },
        **{
            f"corpus_{split}": entry
            for split, entry in protocol["corpora"].items()
        },
    }
    if set(sources) != set(SNAPSHOT_INPUT_FILENAMES) or set(locked_entries) != set(
        SNAPSHOT_INPUT_FILENAMES
    ):
        raise P9RunnerError("P9 frozen input snapshot registry is incomplete")
    files: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for name, filename in SNAPSHOT_INPUT_FILENAMES.items():
        source = sources[name]
        locked = locked_entries[name]
        if _is_reparse_or_symlink(source):
            raise P9RunnerError(f"P9 {name} source is a link/reparse point")
        destination = snapshot_root / filename
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if (
            destination.stat().st_size != locked["bytes"]
            or _sha256_file(destination) != locked["sha256"]
        ):
            raise P9RunnerError(f"P9 {name} changed while creating the frozen snapshot")
        destination.chmod(0o444)
        paths[name] = destination
        files[name] = {
            "filename": filename,
            "bytes": locked["bytes"],
            "sha256": locked["sha256"],
        }
    snapshot = {"root": snapshot_root, "paths": paths, "files": files}
    _verify_asset_snapshot(snapshot)
    return snapshot


def _verify_asset_snapshot(snapshot: Mapping[str, Any]) -> None:
    root = snapshot.get("root")
    paths = snapshot.get("paths")
    files = snapshot.get("files")
    if (
        set(snapshot) != {"root", "paths", "files"}
        or not isinstance(root, Path)
        or not root.is_dir()
        or _is_reparse_or_symlink(root)
        or not isinstance(paths, Mapping)
        or not isinstance(files, Mapping)
        or set(paths) != set(SNAPSHOT_INPUT_FILENAMES)
        or set(files) != set(SNAPSHOT_INPUT_FILENAMES)
    ):
        raise P9RunnerError("P9 asset snapshot has an invalid strict schema")
    for name, entry in files.items():
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"filename", "bytes", "sha256"}
            or entry.get("filename") != SNAPSHOT_INPUT_FILENAMES[name]
            or not isinstance(entry.get("bytes"), int)
            or isinstance(entry.get("bytes"), bool)
            or entry["bytes"] < 0
            or re.fullmatch(r"[a-f0-9]{64}", str(entry.get("sha256") or "")) is None
        ):
            raise P9RunnerError("P9 asset snapshot manifest is invalid")
    expected_relative = {entry["filename"] for entry in files.values()}
    observed_relative: set[str] = set()
    for path in root.rglob("*"):
        if _is_reparse_or_symlink(path):
            raise P9RunnerError("P9 asset snapshot contains a link/reparse point")
        if path.is_dir():
            raise P9RunnerError("P9 asset snapshot contains an unexpected directory")
        observed_relative.add(path.relative_to(root).as_posix())
    if observed_relative != expected_relative:
        raise P9RunnerError("P9 asset snapshot contains unexpected or missing files")
    for name, path in paths.items():
        entry = files[name]
        if (
            path != root / entry["filename"]
            or not path.is_file()
            or not _read_only(path)
            or path.stat().st_size != entry["bytes"]
            or _sha256_file(path) != entry["sha256"]
        ):
            raise P9RunnerError(f"P9 frozen {name} snapshot identity changed")


def _runtime_state_with_asset_snapshot(
    preflight_state: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    protocol = dict(preflight_state["protocol"])
    paths = dict(protocol["paths"])
    frozen = snapshot["paths"]
    paths.update(
        {
            "catalog": frozen["catalog"],
            "evidence": frozen["evidence"],
            "released_public": frozen["released_public"],
            "corpus_metadata": frozen["corpus_metadata"],
        }
    )
    paths["priors"] = {
        name: frozen[f"prior_{name}"] for name in DEFAULT_PRIORS
    }
    paths["corpora"] = {
        split: frozen[f"corpus_{split}"] for split in DEFAULT_CORPORA
    }
    protocol["paths"] = paths
    return {**preflight_state, "protocol": protocol, "asset_snapshot": snapshot}


def _asset_snapshot_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "p9.asset-snapshot.v1",
        "copy_count": len(snapshot["files"]),
        "created_before_corpus_parse": True,
        "all_later_read_inputs_snapshotted": True,
        "confirmation_snapshot_byte_identity_only": True,
        "confirmation_semantic_parse_gated": True,
        "files": json.loads(_canonical_bytes(snapshot["files"])),
        "read_only": True,
        "per_role_verification": True,
        "final_verification": True,
    }


def _minimal_worker_environment(scratch: Path) -> dict[str, str]:
    environment = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
    }
    if os.name == "nt":
        for name in ("SystemRoot", "WINDIR"):
            value = os.environ.get(name)
            if value:
                environment[name] = value
        environment.update({"TEMP": str(scratch), "TMP": str(scratch)})
    else:
        environment.update(
            {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TMPDIR": str(scratch)}
        )
    return environment


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise P9RunnerError(f"{label} must be a non-negative integer")
    return value


def _validate_worker_capture(value: Any, role: str) -> dict[str, Any]:
    roots = {
        "schema_version",
        "role",
        "configuration",
        "stats",
        "integrity_errors",
        "hashes",
        "function_hashes",
    }
    if not isinstance(value, dict) or set(value) != roots or value.get("role") != role:
        raise P9RunnerError("P9 worker capture has an invalid strict root schema")
    if len(_canonical_bytes(value)) > 131_072:
        raise P9RunnerError("P9 worker capture exceeds its byte limit")
    configuration = value.get("configuration")
    stats = value.get("stats")
    hashes = value.get("hashes")
    function_hashes = value.get("function_hashes")
    errors = value.get("integrity_errors")
    if not all(isinstance(item, dict) for item in (configuration, stats, hashes, function_hashes)):
        raise P9RunnerError("P9 worker capture sections must be objects")
    if not isinstance(errors, list) or len(errors) > 32:
        raise P9RunnerError("P9 worker capture integrity hashes are invalid")
    for index, error_hash in enumerate(errors):
        _validate_hex(error_hash, 64, f"capture integrity hash {index}")
    assert isinstance(configuration, dict) and isinstance(stats, dict)
    assert isinstance(hashes, dict) and isinstance(function_hashes, dict)
    if role == BASELINE_ROLE:
        if (
            value.get("schema_version") != "p9.served-agent-reference.v1"
            or configuration
            != {
                "retrieval_mode": "coverage",
                "rerank_mode": "off",
                "question_policy": "fast",
                "offline": True,
                "evidence_opened": False,
            }
            or set(stats) != {"turns", "exception_count"}
            or hashes
            or set(function_hashes) != {"served_agent_class_sha256"}
        ):
            raise P9RunnerError("P9 served capture is not frozen")
        _nonnegative_int(stats["turns"], "served turns")
        _nonnegative_int(stats["exception_count"], "served exception count")
        _validate_hex(
            function_hashes["served_agent_class_sha256"],
            64,
            "served Agent class SHA-256",
        )
        return json.loads(_canonical_bytes(value))

    common_configuration = {
        "retrieval_mode",
        "rerank_mode",
        "question_policy",
        "target_blind",
        "label_free",
        "spec_sha256",
        "lock_sha256",
        "protocol_spec_sha256",
        "spec_schema_version",
        "lock_schema_version",
        "evidence_opened",
    }
    expected_open = role in {ROLES["shadow"], ROLES["active"]}
    expected_configuration = set(common_configuration)
    if expected_open:
        expected_configuration.update(
            {"evidence_identity_verified", "evidence_bytes", "evidence_sha256"}
        )
    if (
        value.get("schema_version") != "p9.compact-negative-lab.v1"
        or set(configuration) != expected_configuration
        or configuration.get("retrieval_mode") != "coverage"
        or configuration.get("rerank_mode") != "off"
        or configuration.get("question_policy") != "fast"
        or configuration.get("target_blind") is not True
        or configuration.get("label_free") is not True
        or configuration.get("spec_schema_version") != WORKER_SPEC_SCHEMA_VERSION
        or configuration.get("lock_schema_version") != WORKER_LOCK_SCHEMA_VERSION
        or configuration.get("evidence_opened") is not expected_open
    ):
        raise P9RunnerError("P9 experiment capture configuration is not frozen")
    for key in ("spec_sha256", "lock_sha256", "protocol_spec_sha256"):
        _validate_hex(configuration[key], 64, f"capture {key}")
    if expected_open:
        if configuration.get("evidence_identity_verified") is not True:
            raise P9RunnerError("P9 experiment capture did not verify compact evidence")
        _nonnegative_int(configuration.get("evidence_bytes"), "capture evidence bytes")
        _validate_hex(configuration.get("evidence_sha256"), 64, "capture evidence SHA-256")
    counter_keys = {
        "turns",
        "activations",
        "output_changes",
        "shadow_changes",
        "fallbacks",
        "exact_exception_fallbacks",
        "exception_count",
        "executable_constraint_total",
        "ignored_record_total",
        "sidecar_rows_read",
        "violation_fallback_candidate_total",
    }
    expected_stats = {
        "schema_version",
        "evidence_schema_version",
        "spec_sha256",
        "frozen_parameters_sha256",
        *counter_keys,
        "partition_totals",
        "reason_counts",
        "rejection_counts",
    }
    if (
        set(stats) != expected_stats
        or stats.get("schema_version") != "p9.compact-negative-lab.v1"
        or stats.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION
    ):
        raise P9RunnerError("P9 experiment capture stats are not frozen")
    for key in counter_keys:
        _nonnegative_int(stats[key], f"capture stats {key}")
    for key in ("spec_sha256", "frozen_parameters_sha256"):
        _validate_hex(stats[key], 64, f"capture stats {key}")
    partition = stats.get("partition_totals")
    if not isinstance(partition, dict) or set(partition) != {
        "compatible",
        "unknown",
        "explicit_violation",
    }:
        raise P9RunnerError("P9 experiment partition totals are invalid")
    for key, count in partition.items():
        _nonnegative_int(count, f"partition total {key}")
    allowed_counter_maps = {
        "reason_counts": {
            "control",
            "no_executable_negatives",
            "empty_candidate_pool",
            "partitioned",
            "exception_fallback",
            "instrumentation_exception_fallback",
        },
        "rejection_counts": {
            "not_active",
            "stale_goal_version",
            "not_negative",
            "not_hard",
            "untrusted_source",
            "not_full_confidence",
            "slot_not_allowed",
            "value_not_single_token",
            "duplicate",
            "compile_exception",
        },
    }
    for name, allowed in allowed_counter_maps.items():
        counts = stats.get(name)
        if not isinstance(counts, dict) or not set(counts).issubset(allowed):
            raise P9RunnerError(f"P9 capture {name} has an invalid strict schema")
        for key, count in counts.items():
            _nonnegative_int(count, f"capture {name}.{key}")
    if set(hashes) != {"audit_sha256", "responses_sha256"}:
        raise P9RunnerError("P9 capture hash registry is invalid")
    for key, digest in hashes.items():
        _validate_hex(digest, 64, f"capture {key}")
    expected_function_hashes = {
        "compile_negative_constraints",
        "classify_masks",
        "stable_compact_partition",
        "CompactEvidenceStore.fetch",
        "P9Agent._rank_candidates",
    }
    if set(function_hashes) != expected_function_hashes:
        raise P9RunnerError("P9 capture function registry is invalid")
    for key, digest in function_hashes.items():
        _validate_hex(digest, 64, f"capture function {key}")
    return json.loads(_canonical_bytes(value))


def _validate_worker_bundle(value: Any, role: str, factory: str) -> dict[str, Any]:
    roots = {
        "schema_version",
        "role",
        "factory",
        "capture",
        "generic_exception_count",
        "generic_exception_classes",
        "network_attempt_count",
        "read_denied_attempt_count",
        "process_denied_attempt_count",
        "audit_network_denied_attempt_count",
        "evidence_open_count",
        "response_count",
        "response_sha256",
        "timing",
        "memory",
    }
    if (
        not isinstance(value, dict)
        or set(value) != roots
        or value.get("schema_version") != "p9.worker-bundle.v1"
        or value.get("role") != role
        or value.get("factory") != factory
    ):
        raise P9RunnerError("P9 worker bundle has an invalid strict root schema")
    value = json.loads(_canonical_bytes(value))
    value["capture"] = _validate_worker_capture(value["capture"], role)
    for key in (
        "generic_exception_count",
        "network_attempt_count",
        "read_denied_attempt_count",
        "process_denied_attempt_count",
        "audit_network_denied_attempt_count",
        "evidence_open_count",
        "response_count",
    ):
        _nonnegative_int(value.get(key), f"worker bundle {key}")
    classes = value.get("generic_exception_classes")
    if (
        not isinstance(classes, list)
        or len(classes) > 32
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", item) is None
            for item in classes
        )
    ):
        raise P9RunnerError("P9 worker exception classes are invalid")
    if value["generic_exception_count"] != len(classes):
        raise P9RunnerError("P9 worker exception aggregate is inconsistent")
    _validate_hex(value.get("response_sha256"), 64, "worker response SHA-256")
    timing = value.get("timing")
    memory = value.get("memory")
    if not isinstance(timing, dict) or set(timing) != {"respond_latency"}:
        raise P9RunnerError("P9 worker timing schema is invalid")
    latency = timing.get("respond_latency")
    if not isinstance(latency, dict) or set(latency) != {"count", "mean_ms", "p95_ms", "max_ms"}:
        raise P9RunnerError("P9 worker latency schema is invalid")
    _nonnegative_int(latency.get("count"), "worker latency count")
    if latency["count"] != value["response_count"]:
        raise P9RunnerError("P9 worker latency count differs from response count")
    if value["capture"]["stats"].get("turns") != value["response_count"]:
        raise P9RunnerError("P9 worker capture turn count differs from response count")
    for key in ("mean_ms", "p95_ms", "max_ms"):
        number = latency.get(key)
        if number is not None and (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(float(number))
            or number < 0
        ):
            raise P9RunnerError(f"P9 worker latency {key} is invalid")
    memory_keys = {
        "backend",
        "sampling_interval_ms",
        "peak_rss_bytes",
        "available",
        "windows_metric",
        "covers_process_lifetime_peak",
    }
    if not isinstance(memory, dict) or set(memory) != memory_keys:
        raise P9RunnerError("P9 worker memory schema is invalid")
    expected_backend = (
        "Windows GetProcessMemoryInfo PeakWorkingSetSize"
        if os.name == "nt"
        else "resource.getrusage ru_maxrss"
    )
    expected_windows_metric = "PeakWorkingSetSize" if os.name == "nt" else None
    if (
        memory.get("backend") != expected_backend
        or memory.get("windows_metric") != expected_windows_metric
        or memory.get("available") is not True
        or memory.get("covers_process_lifetime_peak") is not True
        or not isinstance(memory.get("sampling_interval_ms"), (int, float))
        or isinstance(memory.get("sampling_interval_ms"), bool)
        or not 0 < float(memory["sampling_interval_ms"]) <= 10.0
    ):
        raise P9RunnerError("P9 worker memory is not a lifetime-peak measurement")
    _nonnegative_int(memory.get("peak_rss_bytes"), "worker peak RSS")
    return value


@dataclass
class WorkerClient:
    role: str
    process: subprocess.Popen[bytes]
    nonce: str
    stderr_path: Path
    bootstrap_wall_seconds: float
    request_timeout_seconds: float
    finalize_timeout_seconds: float
    exit_timeout_seconds: float
    stage_manifest_sha256: str
    worker_io_epoch_monotonic: float
    cumulative_worker_io_timeout_seconds: float
    _messages: queue.Queue[tuple[str, Any]] = field(
        default_factory=lambda: queue.Queue(maxsize=2)
    )
    _reader_thread: threading.Thread | None = None
    _next_request_id: int = 1
    _next_ordinal: int = 1
    _ordinal_by_opaque_id: dict[str, int] = field(default_factory=dict)
    _response_digest: Any = field(default_factory=hashlib.sha256)
    response_count: int = 0

    @classmethod
    def start(
        cls,
        role: str,
        *,
        catalog: Path,
        evidence: Path,
        spec: Path,
        worker_lock: Path,
        worker_factory: str,
        rss_sample_ms: float,
        stderr_path: Path,
        worker_script: Path,
        working_directory: Path,
        environment: Mapping[str, str],
        bootstrap_timeout_seconds: float,
        request_timeout_seconds: float,
        finalize_timeout_seconds: float,
        exit_timeout_seconds: float,
        cumulative_worker_io_timeout_seconds: float,
        stage_manifest_sha256: str,
    ) -> "WorkerClient":
        if role not in WORKER_ROLES:
            raise P9RunnerError(f"unknown P9 role: {role}")
        nonce = uuid.uuid4().hex
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(worker_script),
            "--role", role,
            "--nonce", nonce,
            "--factory", worker_factory,
            "--catalog", str(catalog),
            "--evidence", str(evidence),
            "--spec", str(spec),
            "--lock", str(worker_lock),
            "--rss-ms", str(rss_sample_ms),
        ]
        forbidden = (
            "ground_truth", "sample_id", "scenario", "selection", "confirmation",
            "public_set", "prior_p",
        )
        joined = " ".join(command).lower()
        if any(word in joined for word in forbidden):
            raise P9RunnerError("P9 worker command leaks parent-only material")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise P9RunnerError("P9 worker environment is invalid")
        error_handle = stderr_path.open("wb")
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=error_handle,
            env=dict(environment),
            text=False,
            bufsize=0,
            close_fds=True,
        )
        error_handle.close()
        client = cls(
            role=role,
            process=process,
            nonce=nonce,
            stderr_path=stderr_path,
            bootstrap_wall_seconds=0.0,
            request_timeout_seconds=request_timeout_seconds,
            finalize_timeout_seconds=finalize_timeout_seconds,
            exit_timeout_seconds=exit_timeout_seconds,
            stage_manifest_sha256=_validate_hex(
                stage_manifest_sha256, 64, "staged runtime manifest SHA-256"
            ),
            worker_io_epoch_monotonic=started,
            cumulative_worker_io_timeout_seconds=cumulative_worker_io_timeout_seconds,
        )
        client._start_reader()
        ready = client._read_message(
            client._remaining_timeout(bootstrap_timeout_seconds), "bootstrap"
        )
        client.bootstrap_wall_seconds = time.monotonic() - started
        if ready != {"kind": "ready", "nonce": nonce, "role": role}:
            client.abort()
            raise P9RunnerError("invalid P9 worker ready message")
        if client.bootstrap_wall_seconds <= 0:
            client.abort()
            raise P9RunnerError("P9 worker bootstrap timing is unavailable")
        return client

    def _remaining_timeout(self, phase_limit_seconds: float) -> float:
        remaining = (
            self.worker_io_epoch_monotonic
            + self.cumulative_worker_io_timeout_seconds
            - time.monotonic()
        )
        if remaining <= 0:
            self.abort()
            raise P9RunnerError("P9 cumulative worker-I/O deadline expired")
        return min(float(phase_limit_seconds), remaining)

    def _start_reader(self) -> None:
        if self.process.stdout is None:
            raise P9RunnerError("worker stdout is unavailable")
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"p9-reader-{self.role}",
            daemon=True,
        )
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        assert self.process.stdout is not None
        try:
            while True:
                line = self.process.stdout.readline(MAX_WORKER_MESSAGE_BYTES + 1)
                if not line:
                    self._messages.put(("eof", None))
                    return
                if len(line) > MAX_WORKER_MESSAGE_BYTES or not line.endswith(b"\n"):
                    self._messages.put(("error", "worker protocol line is invalid"))
                    return
                try:
                    value = json.loads(line.decode("utf-8", errors="strict"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._messages.put(("error", "worker protocol JSON is invalid"))
                    return
                if not isinstance(value, dict):
                    self._messages.put(("error", "worker protocol message must be an object"))
                    return
                self._messages.put(("message", value))
        except BaseException as exc:
            self._messages.put(("error", f"worker reader failed: {type(exc).__name__}"))

    def _stderr_identity(self) -> dict[str, Any]:
        try:
            payload = self.stderr_path.read_bytes()
        except OSError:
            payload = b""
        return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    def _read_message(self, timeout_seconds: float, phase: str) -> dict[str, Any]:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            self.abort()
            raise P9RunnerError(f"P9 worker {phase} deadline is invalid")
        try:
            kind, value = self._messages.get(timeout=float(timeout_seconds))
        except queue.Empty as exc:
            self.abort()
            raise P9RunnerError(f"P9 worker {phase} deadline expired") from exc
        if kind != "message":
            self.abort()
            raise P9RunnerError(
                f"P9 worker {phase} failed ({kind}); stderr={self._stderr_identity()}"
            )
        return value

    def _write_request(self, encoded: bytes, timeout_seconds: float, phase: str) -> None:
        if self.process.stdin is None:
            self.abort()
            raise P9RunnerError("worker stdin is unavailable")
        completed: queue.Queue[str | None] = queue.Queue(maxsize=1)

        def write() -> None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(encoded)
                self.process.stdin.flush()
                completed.put(None)
            except BaseException as exc:
                completed.put(type(exc).__name__)

        thread = threading.Thread(
            target=write,
            name=f"p9-writer-{self.role}",
            daemon=True,
        )
        thread.start()
        try:
            error_class = completed.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            self.abort()
            raise P9RunnerError(f"P9 worker {phase} write deadline expired") from exc
        if error_class is not None:
            self.abort()
            raise P9RunnerError(f"P9 worker {phase} write failed: {error_class}")

    def _request(self, operation: str, **payload: Any) -> Any:
        request_id = self._next_request_id
        self._next_request_id += 1
        request = {"request_id": request_id, "operation": operation, **payload}
        encoded = _canonical_bytes(request) + b"\n"
        if len(encoded) > 65_536:
            self.abort()
            raise P9RunnerError("P9 worker request exceeds its byte limit")
        phase_timeout = (
            self.finalize_timeout_seconds
            if operation == "finalize"
            else self.request_timeout_seconds
        )
        timeout = self._remaining_timeout(phase_timeout)
        deadline = time.monotonic() + timeout
        self._write_request(encoded, max(0.0, deadline - time.monotonic()), operation)
        reply = self._read_message(max(0.0, deadline - time.monotonic()), operation)
        if reply.get("request_id") != request_id:
            raise P9RunnerError("worker reply request ID mismatch")
        if reply.get("kind") == "error":
            raise P9RunnerError(f"worker {operation} failed: {reply.get('error_class')}")
        if operation == "finalize":
            if reply.get("kind") != "result" or not isinstance(reply.get("bundle"), dict):
                raise P9RunnerError("worker finalize did not return a bundle")
            return reply["bundle"]
        if reply.get("kind") != "reply":
            raise P9RunnerError("worker returned an invalid reply")
        return reply.get("value")

    def reset(self, opaque_id: str, user_profile: dict[str, Any]) -> None:
        if opaque_id in self._ordinal_by_opaque_id:
            raise P9RunnerError("official driver reused an opaque conversation ID")
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        self._ordinal_by_opaque_id[opaque_id] = ordinal
        self._request("reset", ordinal=ordinal, user_profile=dict(user_profile))

    def respond(self, opaque_id: str, user_message: str, turn: int, top_k: int) -> dict[str, Any]:
        ordinal = self._ordinal_by_opaque_id.get(opaque_id)
        if ordinal is None:
            raise P9RunnerError("respond received an unknown conversation ID")
        value = self._request(
            "respond", ordinal=ordinal, user_message=user_message, turn=turn, top_k=top_k
        )
        response = value.get("response") if isinstance(value, dict) else None
        if not isinstance(response, dict):
            raise P9RunnerError("worker response is not an object")
        self.response_count += 1
        self._response_digest.update(
            _canonical_bytes({"ordinal": ordinal, "turn": turn, "response": response}) + b"\n"
        )
        return response

    def finalize(self) -> dict[str, Any]:
        bundle = _validate_worker_bundle(
            self._request("finalize"),
            self.role,
            "starter.agent:Agent"
            if self.role == BASELINE_ROLE
            else "starter.p9_lab:create_p9_agent",
        )
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            return_code = self.process.wait(
                timeout=self._remaining_timeout(self.exit_timeout_seconds)
            )
        except subprocess.TimeoutExpired as exc:
            self.abort()
            raise P9RunnerError("P9 worker exit deadline expired") from exc
        if self.process.stdout is not None:
            self.process.stdout.close()
        if return_code != 0:
            raise P9RunnerError(
                f"P9 worker failed after finalize; stderr={self._stderr_identity()}"
            )
        if (
            bundle.get("response_count") != self.response_count
            or bundle.get("response_sha256") != self._response_digest.hexdigest()
        ):
            raise P9RunnerError("parent and worker response captures disagree")
        bundle["worker_process"] = {
            "separate_process": True,
            "pid": self.process.pid,
            "nonce": self.nonce,
            "role": self.role,
            "staged_runtime": True,
            "python_audit_boundary": True,
            "minimal_environment": True,
            "hostile_native_code_sandboxed": False,
            "isolated_python_flags": ["-I", "-S", "-B"],
            "stage_manifest_sha256": self.stage_manifest_sha256,
        }
        bundle.setdefault("timing", {})["bootstrap_wall_seconds"] = self.bootstrap_wall_seconds
        bundle["timing"]["elapsed_at_final_worker_io_seconds"] = (
            time.monotonic() - self.worker_io_epoch_monotonic
        )
        bundle["timing"]["cumulative_worker_io_timeout_seconds"] = (
            self.cumulative_worker_io_timeout_seconds
        )
        return bundle

    def abort(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=self.exit_timeout_seconds)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.process.kill()
            except OSError:
                pass
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)


def _session_exact_values(session: Mapping[str, Any]) -> dict[str, int]:
    hit = bool(session.get("hit"))
    rank = session.get("best_rank")
    turn = session.get("first_hit_turn")
    if hit:
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= 10:
            raise P9RunnerError("hit conversation has an invalid Top-10 rank")
        if not isinstance(turn, int) or isinstance(turn, bool) or not 1 <= turn <= 10:
            raise P9RunnerError("hit conversation has an invalid first-hit turn")
        rr_units = RR_SCALE // rank
        mttc_turn = turn
        contribution = CONTRIBUTION_SCALE // 2 + 3 * rr_units + CONTRIBUTION_SCALE // 50 * (11 - turn)
    else:
        if rank is not None or turn is not None:
            raise P9RunnerError("miss conversation reports a rank or first-hit turn")
        rr_units = 0
        mttc_turn = 11
        contribution = 0
    return {
        "hit": int(hit),
        "rr_x2520": rr_units,
        "mttc_turn": mttc_turn,
        "official_contribution_x25200": contribution,
    }


def build_exact_totals(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_samples: Counter[str] = Counter()
    scenario_hits: Counter[str] = Counter()
    hit_count = rr_sum = mttc_sum = contribution_sum = 0
    for session in sessions:
        exact = _session_exact_values(session)
        scenario = str(session.get("scenario_type") or "")
        scenario_samples[scenario] += 1
        scenario_hits[scenario] += exact["hit"]
        hit_count += exact["hit"]
        rr_sum += exact["rr_x2520"]
        mttc_sum += exact["mttc_turn"]
        contribution_sum += exact["official_contribution_x25200"]
    if set(scenario_samples) != SCENARIOS or any(
        scenario_samples[name] <= 0 for name in SCENARIOS
    ):
        raise P9RunnerError("official conversation ledger has an invalid scenario registry")
    return {
        "sample_count": len(sessions),
        "hit_count": hit_count,
        "rr_sum_x2520": rr_sum,
        "mttc_turn_sum": mttc_sum,
        "official_contribution_sum_x25200": contribution_sum,
        "scenario_sample_counts": dict(sorted(scenario_samples.items())),
        "scenario_hit_counts": dict(sorted(scenario_hits.items())),
    }


def _metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "sample_count",
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
        "reported_token_usage",
        "scenario_metrics",
    }
    projected = {key: result.get(key) for key in keys}
    _nonnegative_int(projected["sample_count"], "official sample count")
    for key in (
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
    ):
        value = projected[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise P9RunnerError(f"official aggregate {key} is invalid")
    usage = projected["reported_token_usage"]
    if not isinstance(usage, dict) or set(usage) != {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }:
        raise P9RunnerError("official token aggregate has an invalid strict schema")
    for key, value in usage.items():
        _nonnegative_int(value, f"official token aggregate {key}")
    scenarios = projected["scenario_metrics"]
    if not isinstance(scenarios, dict) or set(scenarios) != SCENARIOS:
        raise P9RunnerError("official scenario aggregate has an invalid strict schema")
    for name, values in scenarios.items():
        if not isinstance(values, dict) or set(values) != {
            "sample_count",
            "hit_rate_at_10",
            "mrr",
            "mttc",
        }:
            raise P9RunnerError(f"official scenario aggregate is invalid for {name}")
        _nonnegative_int(values["sample_count"], f"official {name} sample count")
        for key in ("hit_rate_at_10", "mrr", "mttc"):
            number = values[key]
            if (
                not isinstance(number, (int, float))
                or isinstance(number, bool)
                or not math.isfinite(float(number))
            ):
                raise P9RunnerError(f"official scenario {name}.{key} is invalid")
    return json.loads(_canonical_bytes(projected))


def _exact_totals_match(run: Mapping[str, Any]) -> bool:
    totals = run["exact_totals"]
    metrics = run["metrics"]
    count = int(totals["sample_count"])
    if count <= 0:
        return False
    expected = {
        "hit_rate_at_10": round(int(totals["hit_count"]) / count, 6),
        "mrr": round(int(totals["rr_sum_x2520"]) / (RR_SCALE * count), 6),
        "mttc": round(int(totals["mttc_turn_sum"]) / count, 6),
        "recommended_technical_score": round(
            int(totals["official_contribution_sum_x25200"]) / (CONTRIBUTION_SCALE * count), 6
        ),
    }
    return all(float(metrics.get(key, -1.0)) == value for key, value in expected.items())


def _hash_strings(values: Iterable[str]) -> str:
    return _stable_sha256(sorted(str(value) for value in values))


def _all_exception_counters_zero(stats: Mapping[str, Any]) -> bool:
    counters = [
        value
        for key, value in stats.items()
        if "exception" in str(key).lower() and str(key).lower().endswith(("count", "errors"))
    ]
    return all(isinstance(value, int) and not isinstance(value, bool) and value == 0 for value in counters)


def _function_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def run_role(
    role: str,
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
    *,
    preflight_state: Mapping[str, Any],
    spec_path: Path,
    worker_factory: str | None = None,
) -> dict[str, Any]:
    protocol = preflight_state["protocol"]
    spec = preflight_state["spec"]
    with tempfile.TemporaryDirectory(prefix="p9-isolated-") as directory:
        root = Path(directory)
        stage = root / "stage"
        scratch = root / "scratch"
        stage.mkdir()
        scratch.mkdir()
        worker_script, stage_manifest_sha256 = _stage_worker_runtime(
            preflight_state, stage
        )
        runtime_data = stage / "runtime"
        runtime_data.mkdir()
        worker_spec = runtime_data / "worker-spec.json"
        worker_spec.write_bytes(_canonical_bytes(_worker_spec_payload(preflight_state)) + b"\n")
        worker_lock = runtime_data / "worker-lock.json"
        worker_lock.write_bytes(
            _canonical_bytes(
                _worker_lock_payload(preflight_state, worker_spec_sha256=_sha256_file(worker_spec))
            )
            + b"\n"
        )
        generated = {
            "runtime/worker-spec.json": (
                worker_spec.stat().st_size,
                _sha256_file(worker_spec),
            ),
            "runtime/worker-lock.json": (
                worker_lock.stat().st_size,
                _sha256_file(worker_lock),
            ),
        }
        worker_spec.chmod(0o444)
        worker_lock.chmod(0o444)
        stderr_path = scratch / "stderr.bin"
        limits = spec["resource_limits"]
        worker = WorkerClient.start(
            role,
            catalog=protocol["paths"]["catalog"],
            evidence=protocol["paths"]["evidence"],
            spec=worker_spec,
            worker_lock=worker_lock,
            worker_factory=worker_factory or spec["worker_factory"],
            rss_sample_ms=float(limits["rss_sample_ms"]),
            stderr_path=stderr_path,
            worker_script=worker_script,
            working_directory=stage,
            environment=_minimal_worker_environment(scratch),
            bootstrap_timeout_seconds=float(limits["bootstrap_timeout_seconds"]),
            request_timeout_seconds=float(limits["request_timeout_seconds"]),
            finalize_timeout_seconds=float(limits["finalize_timeout_seconds"]),
            exit_timeout_seconds=float(limits["exit_timeout_seconds"]),
            cumulative_worker_io_timeout_seconds=float(
                limits["cumulative_worker_io_timeout_seconds"]
            ),
            stage_manifest_sha256=stage_manifest_sha256,
        )
        recorder = ContractRecorder(worker, catalog_ids)
        try:
            started = time.perf_counter()
            official = evaluate(recorder, samples, catalog_ids, categories, products)
            wall = time.perf_counter() - started
            bundle = worker.finalize()
        except BaseException:
            worker.abort()
            raise
        finally:
            _verify_asset_snapshot(preflight_state["asset_snapshot"])
        _verify_worker_runtime(preflight_state, stage, generated, scratch)
    sessions = official.get("sessions")
    if not isinstance(sessions, list) or any(not isinstance(item, dict) for item in sessions):
        raise P9RunnerError("official driver returned an invalid conversation ledger")
    capture = bundle.pop("capture")
    capture_errors = capture["integrity_errors"]
    capture_hashes = capture["hashes"]
    function_hashes = capture["function_hashes"]
    if not isinstance(capture_hashes, dict) or not isinstance(function_hashes, dict):
        raise P9RunnerError("worker capture hashes are invalid")
    return {
        "role": role,
        "configuration": capture["configuration"],
        "stats": capture["stats"],
        "metrics": _metrics(official),
        "exact_totals": build_exact_totals(sessions),
        "functional_result_sha256": _stable_sha256(official),
        "response_trace_sha256": bundle["response_sha256"],
        "capture_hashes": capture_hashes,
        "function_hashes": function_hashes,
        "contract": {
            "error_count": len(recorder.errors),
            "errors_sha256": _hash_strings(recorder.errors),
        },
        "integrity": {
            "error_count": len(capture_errors),
            "errors_sha256": _hash_strings(capture_errors),
        },
        "runtime": {
            "network_attempt_count": bundle.get("network_attempt_count"),
            "audit_network_denied_attempt_count": bundle.get(
                "audit_network_denied_attempt_count"
            ),
            "read_denied_attempt_count": bundle.get("read_denied_attempt_count"),
            "process_denied_attempt_count": bundle.get("process_denied_attempt_count"),
            "evidence_open_count": bundle.get("evidence_open_count"),
            "generic_exception_count": bundle.get("generic_exception_count"),
            "generic_exception_classes_sha256": _hash_strings(
                bundle.get("generic_exception_classes", [])
            ),
        },
        "timing": {
            "bootstrap_wall_seconds": bundle.get("timing", {}).get("bootstrap_wall_seconds"),
            "elapsed_at_final_worker_io_seconds": bundle.get("timing", {}).get(
                "elapsed_at_final_worker_io_seconds"
            ),
            "cumulative_worker_io_timeout_seconds": bundle.get("timing", {}).get(
                "cumulative_worker_io_timeout_seconds"
            ),
            "evaluation_wall_seconds": wall,
            "respond_latency": bundle.get("timing", {}).get("respond_latency", {}),
        },
        "memory": bundle.get("memory", {}),
        "worker_process": bundle.get("worker_process", {}),
        "_sessions": sessions,
    }


def _session_changes(run: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, int]:
    current = {str(item["sample_id"]): item for item in run["_sessions"]}
    baseline = {str(item["sample_id"]): item for item in control["_sessions"]}
    if set(current) != set(baseline):
        raise P9RunnerError("P9 runs contain different conversation identifiers")
    hit_to_miss = miss_to_hit = rank_improvements = earlier_hits = 0
    for key in current:
        now = _session_exact_values(current[key])
        before = _session_exact_values(baseline[key])
        hit_to_miss += int(before["hit"] == 1 and now["hit"] == 0)
        miss_to_hit += int(before["hit"] == 0 and now["hit"] == 1)
        rank_improvements += int(now["rr_x2520"] > before["rr_x2520"])
        earlier_hits += int(now["mttc_turn"] < before["mttc_turn"])
    return {
        "hit_to_miss_count": hit_to_miss,
        "miss_to_hit_count": miss_to_hit,
        "rank_improvement_count": rank_improvements,
        "earlier_hit_count": earlier_hits,
    }


def _common_gates(run: Mapping[str, Any], expected_count: int) -> dict[str, bool]:
    stats = run.get("stats", {})
    worker = run.get("worker_process", {})
    bootstrap = run.get("timing", {}).get("bootstrap_wall_seconds")
    worker_io_elapsed = run.get("timing", {}).get("elapsed_at_final_worker_io_seconds")
    worker_io_limit = run.get("timing", {}).get("cumulative_worker_io_timeout_seconds")
    return {
        "contract_clean": run.get("contract", {}).get("error_count") == 0,
        "integrity_clean": run.get("integrity", {}).get("error_count") == 0,
        "network_attempts_zero": run.get("runtime", {}).get("network_attempt_count") == 0,
        "audit_network_attempts_zero": (
            run.get("runtime", {}).get("audit_network_denied_attempt_count") == 0
        ),
        "read_boundary_denials_zero": (
            run.get("runtime", {}).get("read_denied_attempt_count") == 0
        ),
        "process_creation_attempts_zero": (
            run.get("runtime", {}).get("process_denied_attempt_count") == 0
        ),
        "generic_exceptions_zero": run.get("runtime", {}).get("generic_exception_count") == 0,
        "lab_exception_counters_zero": isinstance(stats, dict) and _all_exception_counters_zero(stats),
        "complete_official_aggregate": run.get("exact_totals", {}).get("sample_count") == expected_count,
        "exact_totals_match_official_metrics": _exact_totals_match(run),
        "bootstrap_timed_through_ready": isinstance(bootstrap, (int, float)) and bootstrap > 0,
        "cumulative_worker_io_deadline_met": (
            isinstance(worker_io_elapsed, (int, float))
            and isinstance(worker_io_limit, (int, float))
            and worker_io_limit == 180.0
            and 0 < worker_io_elapsed <= worker_io_limit
        ),
        "fresh_external_process": (
            worker.get("separate_process") is True
            and isinstance(worker.get("pid"), int)
            and worker.get("pid") != os.getpid()
            and worker.get("staged_runtime") is True
            and worker.get("python_audit_boundary") is True
            and worker.get("minimal_environment") is True
            and worker.get("hostile_native_code_sandboxed") is False
            and worker.get("isolated_python_flags") == ["-I", "-S", "-B"]
            and re.fullmatch(
                r"[a-f0-9]{64}", str(worker.get("stage_manifest_sha256") or "")
            )
            is not None
            and re.fullmatch(r"[a-f0-9]{32}", str(worker.get("nonce") or "")) is not None
        ),
    }


def _evidence_open_gates(
    run: Mapping[str, Any], evidence: Mapping[str, Any], *, expected_open: bool
) -> dict[str, bool]:
    configuration = run.get("configuration", {})
    if not isinstance(configuration, Mapping):
        configuration = {}
    if not expected_open:
        return {
            "evidence_not_opened": configuration.get("evidence_opened") is False,
            "evidence_identity_not_loaded": all(
                key not in configuration
                for key in ("evidence_identity_verified", "evidence_sha256", "evidence_bytes")
            ),
            "evidence_file_open_count_zero": (
                run.get("runtime", {}).get("evidence_open_count") == 0
            ),
        }
    return {
        "evidence_opened_before_ready": configuration.get("evidence_opened") is True,
        "evidence_file_open_count_positive": (
            isinstance(run.get("runtime", {}).get("evidence_open_count"), int)
            and run["runtime"]["evidence_open_count"] > 0
        ),
        "evidence_identity_verified": configuration.get("evidence_identity_verified") is True,
        "evidence_sha256_exact": configuration.get("evidence_sha256") == evidence.get("sha256"),
        "evidence_bytes_exact": configuration.get("evidence_bytes") == evidence.get("bytes"),
        "evidence_within_16_mib": (
            isinstance(evidence.get("bytes"), int) and 0 < evidence["bytes"] <= EVIDENCE_MAX_BYTES
        ),
    }


def gate_baseline(
    run: Mapping[str, Any], expected_count: int, spec: Mapping[str, Any]
) -> dict[str, Any]:
    configuration = run.get("configuration", {})
    gates = {
        **_common_gates(run, expected_count),
        "role_exact": run.get("role") == BASELINE_ROLE,
        "served_coverage_off_fast_exact": all(
            configuration.get(key) == value for key, value in spec["served_control"].items()
        ),
        "evidence_not_opened": configuration.get("evidence_opened") is False,
        "evidence_file_open_count_zero": (
            run.get("runtime", {}).get("evidence_open_count") == 0
        ),
    }
    return {"decision": "served_reference" if all(gates.values()) else "invalid_served_reference", "gates": gates}


def gate_control(
    run: Mapping[str, Any],
    reference: Mapping[str, Any],
    expected_count: int,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    configuration = run.get("configuration", {})
    gates = {
        **_common_gates(run, expected_count),
        **_evidence_open_gates(run, {}, expected_open=False),
        "role_exact": run.get("role") == ROLES["control"],
        "served_coverage_off_fast_exact": all(
            configuration.get(key) == value for key, value in spec["served_control"].items()
        ),
        "functional_output_equals_served_agent": run.get("functional_result_sha256") == reference.get("functional_result_sha256"),
        "response_trace_equals_served_agent": run.get("response_trace_sha256") == reference.get("response_trace_sha256"),
        "exact_totals_equal_served_agent": run.get("exact_totals") == reference.get("exact_totals"),
    }
    return {"decision": "control" if all(gates.values()) else "invalid_control", "gates": gates}


def gate_shadow(
    run: Mapping[str, Any], control: Mapping[str, Any], expected_count: int, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    stats = run.get("stats", {})
    gates = {
        **_common_gates(run, expected_count),
        **_evidence_open_gates(run, evidence, expected_open=True),
        "role_exact": run.get("role") == ROLES["shadow"],
        "diagnostic_activation_positive": int(stats.get("activations", 0)) > 0,
        "functional_output_equals_control": run.get("functional_result_sha256") == control.get("functional_result_sha256"),
        "response_trace_equals_control": run.get("response_trace_sha256") == control.get("response_trace_sha256"),
        "exact_totals_equal_control": run.get("exact_totals") == control.get("exact_totals"),
    }
    return {"decision": "shadow_only" if all(gates.values()) else "invalid_shadow", "gates": gates}


def _resource_gates(
    run: Mapping[str, Any], control: Mapping[str, Any], limits: Mapping[str, Any]
) -> dict[str, bool]:
    bootstrap = float(run.get("timing", {}).get("bootstrap_wall_seconds") or 0.0)
    base_bootstrap = float(control.get("timing", {}).get("bootstrap_wall_seconds") or 0.0)
    wall = float(run.get("timing", {}).get("evaluation_wall_seconds") or 0.0)
    base_wall = float(control.get("timing", {}).get("evaluation_wall_seconds") or 0.0)
    p95 = float(run.get("timing", {}).get("respond_latency", {}).get("p95_ms") or 0.0)
    base_p95 = float(control.get("timing", {}).get("respond_latency", {}).get("p95_ms") or 0.0)
    peak = run.get("memory", {}).get("peak_rss_bytes")
    base_peak = control.get("memory", {}).get("peak_rss_bytes")
    return {
        "bootstrap_within_1_20x": (
            base_bootstrap > 0
            and bootstrap <= float(limits["bootstrap_ratio"]) * base_bootstrap
        ),
        "wall_within_1_30x": base_wall > 0 and wall <= float(limits["wall_ratio"]) * base_wall,
        "response_p95_within_1_30x": (
            base_p95 > 0 and p95 <= float(limits["response_p95_ratio"]) * base_p95
        ),
        "peak_rss_within_1_20x": (
            run.get("memory", {}).get("available") is True
            and control.get("memory", {}).get("available") is True
            and run.get("memory", {}).get("covers_process_lifetime_peak") is True
            and control.get("memory", {}).get("covers_process_lifetime_peak") is True
            and isinstance(peak, int)
            and isinstance(base_peak, int)
            and base_peak > 0
            and peak <= float(limits["peak_rss_ratio"]) * base_peak
        ),
    }


def _positive_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return None
    return numerator_value / denominator_value if denominator_value > 0 else None


def gate_active(
    run: Mapping[str, Any],
    control: Mapping[str, Any],
    expected_count: int,
    spec: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    totals = run["exact_totals"]
    baseline = control["exact_totals"]
    stats = run.get("stats", {})
    changes = _session_changes(run, control)
    scenario_regressions = [
        name
        for name in sorted(SCENARIOS)
        if int(totals["scenario_hit_counts"].get(name, -1))
        < int(baseline["scenario_hit_counts"].get(name, -1))
    ]
    gates = {
        **_common_gates(run, expected_count),
        **_evidence_open_gates(run, evidence, expected_open=True),
        "role_exact": run.get("role") == ROLES["active"],
        "activation_positive": int(stats.get("activations", 0)) > 0,
        "output_changes_positive": int(stats.get("output_changes", 0)) > 0,
        "hit_rate_non_decrease": int(totals["hit_count"]) >= int(baseline["hit_count"]),
        "mrr_strict_increase": int(totals["rr_sum_x2520"]) > int(baseline["rr_sum_x2520"]),
        "mttc_non_increase": int(totals["mttc_turn_sum"]) <= int(baseline["mttc_turn_sum"]),
        "technical_score_strict_increase": (
            int(totals["official_contribution_sum_x25200"])
            > int(baseline["official_contribution_sum_x25200"])
        ),
        "four_scenario_hit_rates_non_decrease": not scenario_regressions,
        "zero_hit_to_miss": changes["hit_to_miss_count"] == 0,
        **_resource_gates(run, control, spec["resource_limits"]),
    }
    return {
        "decision": "eligible" if all(gates.values()) else "reject",
        "gates": gates,
        "exact_delta_vs_control": {
            key: int(totals[key]) - int(baseline[key])
            for key in (
                "hit_count", "rr_sum_x2520", "mttc_turn_sum",
                "official_contribution_sum_x25200",
            )
        },
        "session_changes_vs_control": changes,
        "scenario_hit_count_regressions": scenario_regressions,
        "resource_ratios": {
            "bootstrap": _positive_ratio(
                run.get("timing", {}).get("bootstrap_wall_seconds"),
                control.get("timing", {}).get("bootstrap_wall_seconds"),
            ),
            "wall": _positive_ratio(
                run.get("timing", {}).get("evaluation_wall_seconds"),
                control.get("timing", {}).get("evaluation_wall_seconds"),
            ),
            "response_p95": _positive_ratio(
                run.get("timing", {}).get("respond_latency", {}).get("p95_ms"),
                control.get("timing", {}).get("respond_latency", {}).get("p95_ms"),
            ),
            "peak_rss": _positive_ratio(
                run.get("memory", {}).get("peak_rss_bytes"),
                control.get("memory", {}).get("peak_rss_bytes"),
            ),
        },
    }


def repeat_exact(initial: Mapping[str, Any], repeated: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "different_worker_nonce": initial.get("worker_process", {}).get("nonce") != repeated.get("worker_process", {}).get("nonce"),
        "functional_result_exact": initial.get("functional_result_sha256") == repeated.get("functional_result_sha256"),
        "response_trace_exact": initial.get("response_trace_sha256") == repeated.get("response_trace_sha256"),
        "exact_totals_exact": initial.get("exact_totals") == repeated.get("exact_totals"),
        "capture_hashes_exact": initial.get("capture_hashes") == repeated.get("capture_hashes"),
        "function_hashes_exact": initial.get("function_hashes") == repeated.get("function_hashes"),
        "contract_exact": initial.get("contract") == repeated.get("contract"),
        "integrity_exact": initial.get("integrity") == repeated.get("integrity"),
        "configuration_exact": initial.get("configuration") == repeated.get("configuration"),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _run_initial_split(
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
    *,
    preflight_state: Mapping[str, Any],
    spec_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    runs = {
        BASELINE_ROLE: run_role(
            BASELINE_ROLE,
            samples,
            catalog_ids,
            categories,
            products,
            preflight_state=preflight_state,
            spec_path=spec_path,
            worker_factory="starter.agent:Agent",
        ),
        **{
            role: run_role(
                role,
                samples,
                catalog_ids,
                categories,
                products,
                preflight_state=preflight_state,
                spec_path=spec_path,
            )
            for role in ROLE_ORDER
        },
    }
    count = len(samples)
    evidence = preflight_state["protocol"]["evidence"]
    gates = {
        BASELINE_ROLE: gate_baseline(runs[BASELINE_ROLE], count, preflight_state["spec"]),
        ROLES["control"]: gate_control(
            runs[ROLES["control"]], runs[BASELINE_ROLE], count, preflight_state["spec"]
        ),
        ROLES["shadow"]: gate_shadow(
            runs[ROLES["shadow"]], runs[ROLES["control"]], count, evidence
        ),
        ROLES["active"]: gate_active(
            runs[ROLES["active"]], runs[ROLES["control"]], count,
            preflight_state["spec"], evidence,
        ),
    }
    return runs, gates


def _repeat_control_active(
    initial: Mapping[str, Mapping[str, Any]],
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
    *,
    preflight_state: Mapping[str, Any],
    spec_path: Path,
) -> dict[str, Any]:
    repeated = {
        BASELINE_ROLE: run_role(
            BASELINE_ROLE,
            samples,
            catalog_ids,
            categories,
            products,
            preflight_state=preflight_state,
            spec_path=spec_path,
            worker_factory="starter.agent:Agent",
        ),
        **{
            role: run_role(
                role,
                samples,
                catalog_ids,
                categories,
                products,
                preflight_state=preflight_state,
                spec_path=spec_path,
            )
            for role in (ROLES["control"], ROLES["active"])
        },
    }
    count = len(samples)
    evidence = preflight_state["protocol"]["evidence"]
    baseline_gate = gate_baseline(repeated[BASELINE_ROLE], count, preflight_state["spec"])
    control_gate = gate_control(
        repeated[ROLES["control"]], repeated[BASELINE_ROLE], count, preflight_state["spec"]
    )
    active_gate = gate_active(
        repeated[ROLES["active"]], repeated[ROLES["control"]], count,
        preflight_state["spec"], evidence,
    )
    exact = {role: repeat_exact(initial[role], repeated[role]) for role in repeated}
    return {
        "attempted": True,
        "passed": (
            baseline_gate["decision"] == "served_reference"
            and control_gate["decision"] == "control"
            and active_gate["decision"] == "eligible"
            and all(value["passed"] for value in exact.values())
        ),
        "runs": repeated,
        "gates": {
            BASELINE_ROLE: baseline_gate,
            ROLES["control"]: control_gate,
            ROLES["active"]: active_gate,
        },
        "exact": exact,
    }


def _run_summary(run: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "role",
        "configuration",
        "stats",
        "metrics",
        "exact_totals",
        "functional_result_sha256",
        "response_trace_sha256",
        "capture_hashes",
        "function_hashes",
        "contract",
        "integrity",
        "runtime",
        "timing",
        "memory",
        "worker_process",
    }
    if set(run) != {*keys, "_sessions"}:
        raise P9RunnerError("P9 role run has an invalid strict summary schema")
    return {key: run[key] for key in sorted(keys)}


def _split_artifact(
    corpus: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, Any]],
    repeat: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "corpus": dict(corpus),
        "initial": {
            "runs": {role: _run_summary(run) for role, run in runs.items()},
            "gates": dict(gates),
        },
        "repeat": {
            key: (
                {role: _run_summary(run) for role, run in value.items()}
                if key == "runs" and isinstance(value, dict)
                else value
            )
            for key, value in repeat.items()
        },
    }


def _assert_artifact_safe(value: Any) -> None:
    try:
        encoded = _canonical_bytes(value)
    except (TypeError, ValueError) as exc:
        raise P9RunnerError("P9 artifact is not strict canonical JSON") from exc
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise P9RunnerError("P9 artifact exceeds its byte limit")
    state = {"items": 0}

    def visit(item: Any, depth: int) -> None:
        state["items"] += 1
        if state["items"] > MAX_ARTIFACT_ITEMS or depth > MAX_ARTIFACT_DEPTH:
            raise P9RunnerError("P9 artifact exceeds its structural limit")
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise P9RunnerError("P9 artifact keys must be strings")
                lowered = key.lower()
                if lowered in ARTIFACT_FORBIDDEN_KEYS:
                    raise P9RunnerError(f"P9 artifact contains prohibited key: {key}")
                visit(key, depth + 1)
                visit(nested, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested, depth + 1)
            return
        if item is None or isinstance(item, bool) or isinstance(item, int):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise P9RunnerError("P9 artifact contains a non-finite number")
            return
        if isinstance(item, str):
            if len(item.encode("utf-8")) > MAX_ARTIFACT_STRING_BYTES or any(
                character in item for character in ("\x00", "\r", "\n")
            ):
                raise P9RunnerError("P9 artifact contains an invalid string")
            if ASIN_SHAPE.fullmatch(item):
                raise P9RunnerError("P9 artifact contains an ASIN-shaped value")
            return
        raise P9RunnerError("P9 artifact contains an unsupported JSON type")

    visit(value, 0)


def run_evaluation(
    *,
    spec_path: Path = DEFAULT_SPEC,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    before = preflight(spec_path=spec_path, lock_path=lock_path)
    with tempfile.TemporaryDirectory(prefix="p9-assets-") as directory:
        snapshot = _create_asset_snapshot(before, Path(directory) / "snapshot")
        runtime_state = _runtime_state_with_asset_snapshot(before, snapshot)
        try:
            return _run_evaluation_with_snapshot(
                before=before,
                runtime_state=runtime_state,
                snapshot=snapshot,
                spec_path=spec_path,
                lock_path=lock_path,
            )
        finally:
            _verify_asset_snapshot(snapshot)


def _run_evaluation_with_snapshot(
    *,
    before: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    spec_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    protocol = runtime_state["protocol"]
    catalog_ids, categories, products = catalog_index(protocol["paths"]["catalog"])
    excluded = _prior_target_set(runtime_state)
    selection_rows, selection_corpus, selection_targets = _load_split(
        "selection", runtime_state, catalog_ids=catalog_ids, excluded_targets=excluded
    )
    selection_runs, selection_gates = _run_initial_split(
        selection_rows,
        catalog_ids,
        categories,
        products,
        preflight_state=runtime_state,
        spec_path=spec_path,
    )
    selection_ready = (
        selection_gates[BASELINE_ROLE]["decision"] == "served_reference"
        and selection_gates[ROLES["control"]]["decision"] == "control"
        and selection_gates[ROLES["shadow"]]["decision"] == "shadow_only"
        and selection_gates[ROLES["active"]]["decision"] == "eligible"
    )
    selection_repeat: dict[str, Any] = {
        "attempted": False,
        "passed": False,
        "reason": "selection active role was not eligible",
    }
    if selection_ready:
        selection_repeat = _repeat_control_active(
            selection_runs,
            selection_rows,
            catalog_ids,
            categories,
            products,
            preflight_state=runtime_state,
            spec_path=spec_path,
        )
    selection_passed = bool(selection_ready and selection_repeat.get("passed"))

    confirmation_artifact: dict[str, Any] = {
        "identity_bytes_hashed_preflight": True,
        "semantic_parse_executed": False,
        "official_aggregate_executed": False,
        "reason": "selection did not pass eligibility and exact-repeat gates",
    }
    promotion = False
    if selection_passed:
        confirmation_rows, confirmation_corpus, _ = _load_split(
            "confirmation",
            runtime_state,
            catalog_ids=catalog_ids,
            excluded_targets=excluded | selection_targets,
        )
        confirmation_runs, confirmation_gates = _run_initial_split(
            confirmation_rows,
            catalog_ids,
            categories,
            products,
            preflight_state=runtime_state,
            spec_path=spec_path,
        )
        confirmation_ready = (
            confirmation_gates[BASELINE_ROLE]["decision"] == "served_reference"
            and confirmation_gates[ROLES["control"]]["decision"] == "control"
            and confirmation_gates[ROLES["shadow"]]["decision"] == "shadow_only"
            and confirmation_gates[ROLES["active"]]["decision"] == "eligible"
        )
        confirmation_repeat: dict[str, Any] = {
            "attempted": False,
            "passed": False,
            "reason": "confirmation active role was not eligible",
        }
        if confirmation_ready:
            confirmation_repeat = _repeat_control_active(
                confirmation_runs,
                confirmation_rows,
                catalog_ids,
                categories,
                products,
                preflight_state=runtime_state,
                spec_path=spec_path,
            )
        promotion = bool(confirmation_ready and confirmation_repeat.get("passed"))
        confirmation_artifact = {
            "identity_bytes_hashed_preflight": True,
            "semantic_parse_executed": True,
            "official_aggregate_executed": True,
            **_split_artifact(
                confirmation_corpus,
                confirmation_runs,
                confirmation_gates,
                confirmation_repeat,
            ),
        }

    _verify_asset_snapshot(snapshot)
    after = preflight(spec_path=spec_path, lock_path=lock_path)
    if before["git"] != after["git"] or before["identity_snapshot"] != after["identity_snapshot"]:
        raise P9RunnerError("P9 source or input identity changed during evaluation")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "decision": "promote_p9_r01" if promotion else "retain_p9_c00",
        "winner_id": ROLES["active"] if promotion else ROLES["control"],
        "public_evaluation_run": False,
        "inputs": {
            **before["summary"],
            "confirmation_rows_parsed": selection_passed,
            "confirmation_identity_bytes_hashed_preflight": True,
            "confirmation_semantic_parse_executed": selection_passed,
            "worker_factory": before["spec"]["worker_factory"],
            "roles": before["spec"]["roles"],
            "served_control": before["spec"]["served_control"],
            "resource_limits": before["spec"]["resource_limits"],
        },
        "selection": _split_artifact(
            selection_corpus, selection_runs, selection_gates, selection_repeat
        ),
        "confirmation": confirmation_artifact,
        "promotion": {
            "eligible": promotion,
            "public_confirmation_allowed": promotion,
            "shadow_can_win": False,
            "tie_policy": "Every strict gate must pass on both untouched splits; otherwise retain C00.",
        },
        "provenance": {
            "git": before["git"],
            "locked_source": before["protocol"]["source"],
            "source_target_scan": before["protocol"]["source_target_scan"],
            "asset_snapshot": _asset_snapshot_summary(snapshot),
            "worker_boundary": before["worker_boundary"],
            "input_identity_sha256": _stable_sha256(before["identity_snapshot"]),
            "snapshot_stable": True,
            "parent_function_hashes": {
                "official_evaluate": _function_sha256(evaluate),
                "gate_active": _function_sha256(gate_active),
                "repeat_exact": _function_sha256(repeat_exact),
            },
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
            },
        },
        "boundary": (
            "Only the parent process opens frozen conversation rows and computes official "
            "aggregates. Each role runs in a fresh offline process with sanitized spec/lock "
            "files. The compact sidecar is catalog-only; its cold open and identity check "
            "are included in bootstrap timing. Every locked input read after preflight, "
            "including catalog, sidecar, released public, priors, corpus metadata, and both "
            "P9 split files, is copied once as bytes into a read-only, hash-verified snapshot "
            "before any conversation row is parsed; all roles use only those copies. Copying "
            "confirmation bytes does not parse them: confirmation rows are parsed and executed "
            "only after selection plus exact repeat pass. Locked starter candidates pass a "
            "direct dangerous-import/dynamic-name AST scan. The Python audit boundary is not "
            "an OS sandbox against hostile native code. The cumulative worker-I/O deadline is "
            "enforced only at bootstrap, request, finalize, and exit I/O boundaries. "
            "The artifact stores aggregates and hashes only."
        ),
    }
    _assert_artifact_safe(artifact)
    return artifact


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"P9 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-preflight",
        action="store_true",
        help="validate identities and Git gates without parsing or running either split",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_preflight:
        state = preflight(spec_path=args.spec, lock_path=args.lock)
        print(json.dumps(state["summary"], indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if args.output.exists():
        raise FileExistsError(f"P9 output already exists: {args.output}")
    artifact = run_evaluation(spec_path=args.spec, lock_path=args.lock)
    _atomic_write_json(args.output, artifact)
    print(
        f"[p9] decision={artifact['decision']} winner={artifact['winner_id']} wrote={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
