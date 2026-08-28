from __future__ import annotations

"""Run the frozen, target-blind P8 selection and confirmation protocol."""

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
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


SCHEMA_VERSION = "p8.explicit-negative-evaluation.v1"
SPEC_SCHEMA_VERSION = "p8.explicit-negative-matrix.v1"
LOCK_SCHEMA_VERSION = "p8.prereg-lock.v1"
WORKER_LOCK_SCHEMA_VERSION = "p8.worker-lock.v1"
WORKER_SPEC_SCHEMA_VERSION = "p8.worker-spec.v1"
METADATA_SCHEMA_VERSION = "p8.explicit-negative-corpora.v1"
DEFAULT_SPEC = PROJECT_ROOT / "configs" / "p8_explicit_negative_matrix.json"
DEFAULT_LOCK = PROJECT_ROOT / "configs" / "p8_prereg_lock.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "p8_explicit_negative_evaluation.json"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "catalog.jsonl"
DEFAULT_PUBLIC = PROJECT_ROOT / "data" / "public_set.jsonl"
DEFAULT_METADATA = PROJECT_ROOT / "experiments" / "p8_explicit_negative_corpora.metadata.json"
DEFAULT_CORPORA = {
    "selection": PROJECT_ROOT / "experiments" / "p8_selection_product_disjoint.jsonl",
    "confirmation": PROJECT_ROOT / "experiments" / "p8_confirmation_product_disjoint.jsonl",
}
DEFAULT_PRIORS = {
    "p1": PROJECT_ROOT / "experiments" / "p1_derived_product_disjoint.jsonl",
    "p5": PROJECT_ROOT / "experiments" / "p5_selection_product_disjoint.jsonl",
    "p6": PROJECT_ROOT / "experiments" / "p6_selection_product_disjoint.jsonl",
    "p7": PROJECT_ROOT / "experiments" / "p7_selection_product_disjoint.jsonl",
}
ROLES = {
    "control": "P8.C00.r08_coverage",
    "shadow": "P8.S00.explicit_negative_shadow",
    "active": "P8.R01.explicit_negative_partition",
}
BASELINE_ROLE = "P8.B00.served_agent"
ROLE_ORDER = tuple(ROLES.values())
WORKER_ROLES = (BASELINE_ROLE, *ROLE_ORDER)
SCENARIOS = {"boundary", "browsing", "buying", "intent_override"}
RR_SCALE = 2520
CONTRIBUTION_SCALE = 25_200
REQUIRED_SOURCE_PATHS = {
    "builder": "scripts/build_p8_selection_corpus.py",
    "lock_builder": "scripts/build_p8_prereg_lock.py",
    "p8_negative": "starter/p8_negative.py",
    "p8_lab": "starter/p8_lab.py",
    "p8_worker": "scripts/p8_worker.py",
    "evaluate_p8": "scripts/evaluate_p8.py",
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
}


class P8RunnerError(RuntimeError):
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
        raise P8RunnerError(f"JSONL rows must be objects: {path}")
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
        raise P8RunnerError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise P8RunnerError(f"JSON root must be an object: {path}")
    return value


def _safe_project_path(project_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise P8RunnerError(f"{label} path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise P8RunnerError(f"{label} path is not safely project-relative")
    root = project_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise P8RunnerError(f"{label} path escapes project root") from exc
    return path


def _validate_hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(rf"[a-f0-9]{{{length}}}", value) is None:
        raise P8RunnerError(f"{label} must be {length} lowercase hexadecimal characters")
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
        raise P8RunnerError(f"{label} has an invalid strict file schema")
    path = _safe_project_path(project_root, entry.get("path"), label)
    size = entry.get("bytes")
    sha = _validate_hex(entry.get("sha256"), 64, f"{label} SHA-256")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise P8RunnerError(f"{label} bytes must be a positive integer")
    if not path.is_file():
        raise P8RunnerError(f"{label} is missing: {path}")
    if path.stat().st_size != size or _sha256_file(path) != sha:
        raise P8RunnerError(f"{label} does not match its frozen bytes/SHA-256")
    return path, dict(entry)


def validate_matrix_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    expected_roots = {
        "schema_version",
        "worker_factory",
        "roles",
        "served_control",
        "mechanism",
        "resource_limits",
        "promotion_gates",
    }
    if set(spec) != expected_roots or spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise P8RunnerError("P8 matrix spec has an invalid strict root schema")
    if spec.get("worker_factory") != "starter.p8_lab:create_p8_agent":
        raise P8RunnerError("P8 matrix spec has an unexpected worker factory")
    if spec.get("roles") != ROLES:
        raise P8RunnerError("P8 matrix spec role registry is not frozen")
    if spec.get("served_control") != {
        "retrieval_mode": "coverage",
        "rerank_mode": "off",
        "question_policy": "fast",
    }:
        raise P8RunnerError("P8 C00 is not the exact served coverage/off/fast control")
    mechanism = spec.get("mechanism")
    expected_mechanism_keys = {
        "candidate_pool",
        "executable_constraint",
        "product_evidence_min_confidence",
        "product_description_is_evidence",
        "candidate_states",
        "aggregate",
        "ordering",
        "tie_break",
        "no_executable_constraint",
        "tail_policy",
    }
    if not isinstance(mechanism, dict) or set(mechanism) != expected_mechanism_keys:
        raise P8RunnerError("P8 mechanism has an invalid strict schema")
    constraint = mechanism.get("executable_constraint")
    if (
        mechanism.get("candidate_pool") != 50
        or mechanism.get("product_evidence_min_confidence") != 0.9
        or mechanism.get("product_description_is_evidence") is not False
        or mechanism.get("candidate_states")
        != ["compatible", "unknown", "explicit_violation"]
        or not isinstance(constraint, dict)
        or constraint.get("status") != "active"
        or constraint.get("polarity") != -1
        or constraint.get("hardness") != "hard"
        or constraint.get("confidence") != 1.0
        or constraint.get("source") != "excluded_terms"
        or constraint.get("version") != "current_goal_only"
        or constraint.get("value") != "single_ascii_token"
        or constraint.get("allowed_slots")
        != ["audience", "material", "color", "closure", "style", "use_case"]
    ):
        raise P8RunnerError("P8 executable-negative mechanism is not frozen")
    limits = spec.get("resource_limits")
    if limits != {
        "wall_ratio": 1.3,
        "response_p95_ratio": 1.3,
        "peak_rss_ratio": 1.2,
        "rss_sample_ms": 10.0,
    }:
        raise P8RunnerError("P8 resource limits are not frozen")
    expected_gates = {
        "hit_rate_non_decrease",
        "mrr_strict_increase",
        "mttc_non_increase",
        "technical_score_strict_increase",
        "scenario_hit_rate_non_decrease",
        "zero_hit_to_miss",
        "repeat_exact",
    }
    gates = spec.get("promotion_gates")
    if not isinstance(gates, dict) or set(gates) != expected_gates or not all(
        value is True for value in gates.values()
    ):
        raise P8RunnerError("P8 promotion gates are not strict and frozen")
    return json.loads(_canonical_bytes(spec))


def _git(project_root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={project_root.resolve().as_posix()}", *arguments],
        cwd=project_root,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if completed.returncode:
        raise P8RunnerError(f"Git command failed: {' '.join(arguments)}")
    return completed.stdout if binary else str(completed.stdout).strip()


def _assert_tracked(project_root: Path, path: Path, label: str) -> None:
    relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    _git(project_root, "ls-files", "--error-unmatch", relative)


def validate_worker_source_boundary(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    for word in WORKER_FORBIDDEN_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", source, re.IGNORECASE):
            raise P8RunnerError(f"P8 worker contains parent-only vocabulary: {word}")
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
        raise P8RunnerError("P8 worker imports a parent-only package")
    return {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "ast_sha256": _stable_sha256(ast.dump(tree, include_attributes=False)),
        "forbidden_vocabulary_absent": True,
        "parent_only_import_absent": True,
    }


def _validate_source_lock(
    project_root: Path,
    source: Any,
    *,
    enforce_git: bool,
) -> dict[str, Any]:
    if not isinstance(source, dict) or set(source) != {"git_commit", "git_branch", "files"}:
        raise P8RunnerError("P8 source lock has an invalid strict schema")
    commit = _validate_hex(source.get("git_commit"), 40, "source commit")
    branch = source.get("git_branch")
    files = source.get("files")
    if not isinstance(branch, str) or not branch or not isinstance(files, dict):
        raise P8RunnerError("P8 source revision is incomplete")
    if set(files) != set(REQUIRED_SOURCE_NAMES):
        missing = sorted(REQUIRED_SOURCE_NAMES - set(files))
        extra = sorted(set(files) - REQUIRED_SOURCE_NAMES)
        raise P8RunnerError(
            "P8 source lock registry differs from the frozen set; missing="
            f"{missing}, extra={extra}"
        )
    identities: dict[str, Any] = {}
    for name, entry in files.items():
        path, frozen = _validate_file_entry(
            project_root, entry, f"source {name}", extras={"git_blob_sha1"}
        )
        declared_blob = _validate_hex(
            frozen["git_blob_sha1"], 40, f"source {name} Git blob"
        )
        expected_path = (project_root / REQUIRED_SOURCE_PATHS[name]).resolve()
        if path.resolve() != expected_path:
            raise P8RunnerError(f"source {name} does not use its canonical project path")
        identities[name] = frozen
        if enforce_git:
            _assert_tracked(project_root, path, f"source {name}")
            relative = path.resolve().relative_to(project_root.resolve()).as_posix()
            working_blob = str(
                _git(project_root, "hash-object", f"--path={relative}", "--", relative)
            )
            commit_blob = str(_git(project_root, "rev-parse", f"{commit}:{relative}"))
            if working_blob != declared_blob or commit_blob != declared_blob:
                raise P8RunnerError(f"source {name} differs from the preregistered Git blob")
    if enforce_git:
        if _git(project_root, "branch", "--show-current") != branch:
            raise P8RunnerError("current branch differs from the P8 source lock")
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={project_root.resolve().as_posix()}",
                "merge-base",
                "--is-ancestor",
                commit,
                "HEAD",
            ],
            cwd=project_root,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise P8RunnerError("P8 preregistration commit is not an ancestor of HEAD")
    return {"git_commit": commit, "git_branch": branch, "files": identities}


def _validate_official_evaluator(path: Path) -> str:
    frozen = "7c808347b31ef3121a9cbc4810ac3eb325f950ba"
    if EXPECTED_EVALUATOR_BLOB != frozen or git_blob_sha1(path) != frozen:
        raise P8RunnerError("P8 source lock does not bind the official evaluator blob")
    return frozen


def _validate_metadata(
    metadata: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise P8RunnerError("P8 corpus metadata schema is not frozen")
    catalog = metadata.get("catalog_source")
    inputs = metadata.get("input_sources")
    corpora = metadata.get("corpora")
    exclusions = metadata.get("exclusions")
    outputs = metadata.get("outputs")
    if not all(
        isinstance(value, dict) for value in (catalog, inputs, corpora, exclusions, outputs)
    ):
        raise P8RunnerError("P8 corpus metadata sections are incomplete")
    assert (
        isinstance(catalog, dict)
        and isinstance(inputs, dict)
        and isinstance(corpora, dict)
        and isinstance(outputs, dict)
    )
    if (
        catalog.get("sha256") != lock["catalog"]["sha256"]
        or catalog.get("loaded_product_count") != lock["catalog"]["rows"]
        or catalog.get("frozen_sha256_verified") is not True
        or catalog.get("expected_count_verified") is not True
    ):
        raise P8RunnerError("P8 metadata catalog identity differs from the lock")
    input_names = {
        "released_public": "released_public",
        "prior_p1_derived": "p1",
        "prior_p5_derived": "p5",
        "prior_p6_derived": "p6",
        "prior_p7_derived": "p7",
    }
    if set(inputs) != set(input_names):
        raise P8RunnerError("P8 metadata input registry is incomplete")
    for metadata_name, lock_name in input_names.items():
        entry = inputs[metadata_name]
        locked = lock["released_public"] if lock_name == "released_public" else lock["priors"][lock_name]
        if (
            not isinstance(entry, dict)
            or entry.get("sample_count") != locked["rows"]
            or entry.get("unique_target_count") != locked["rows"]
            or entry.get("frozen_samples_sha256_verified") is not True
        ):
            raise P8RunnerError(f"P8 metadata identity failed for {metadata_name}")
        if metadata_name == "released_public" and (
            entry.get("git_blob_sha1_lf") != lock["released_public"]["git_blob_sha1_lf"]
            or entry.get("frozen_git_blob_verified") is not True
        ):
            raise P8RunnerError("P8 metadata public Git blob identity failed")
    if set(corpora) != {"selection", "confirmation"}:
        raise P8RunnerError("P8 metadata split registry is incomplete")
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
            raise P8RunnerError(f"P8 metadata {split} identity differs from the lock")
        output = outputs.get(split)
        if (
            not isinstance(output, dict)
            or output.get("expected_frozen_samples_sha256")
            != locked["canonical_samples_sha256"]
            or output.get("samples_file_sha256") != locked["canonical_samples_sha256"]
            or output.get("frozen_samples_sha256_verified") is not True
        ):
            raise P8RunnerError(f"P8 metadata {split} output is not self-frozen")
    def overlap_values(value: Any, key: str = "") -> list[Any]:
        if isinstance(value, Mapping):
            found: list[Any] = []
            for nested_key, nested in value.items():
                found.extend(overlap_values(nested, str(nested_key)))
            return found
        return [value] if "overlap" in key.lower() else []

    numeric_exclusions = overlap_values(exclusions)
    if not numeric_exclusions or any(value != 0 for value in numeric_exclusions):
        raise P8RunnerError("P8 metadata does not prove zero exclusion overlap")
    return {
        "schema_version": metadata["schema_version"],
        "catalog_verified": True,
        "input_registry_verified": True,
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
        "schema_version",
        "source",
        "spec",
        "catalog",
        "released_public",
        "priors",
        "corpus_metadata",
        "corpora",
    }
    if set(lock) != roots or lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise P8RunnerError("P8 preregistration lock has an invalid strict root schema")
    source = _validate_source_lock(project_root, lock.get("source"), enforce_git=enforce_git)
    evaluator_path = _safe_project_path(
        project_root, source["files"]["evaluator"]["path"], "official evaluator"
    )
    _validate_official_evaluator(evaluator_path)
    locked_spec_path, spec_entry = _validate_file_entry(project_root, lock.get("spec"), "matrix spec")
    if locked_spec_path.resolve() != spec_path.resolve():
        raise P8RunnerError("requested P8 matrix spec differs from the lock")
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
        raise P8RunnerError("P8 lock does not bind the official 50,000-row catalog")
    if (
        _validate_hex(public_entry["git_blob_sha1_lf"], 40, "public Git blob")
        != EXPECTED_PUBLIC_BLOB
        or git_blob_sha1(public_path) != EXPECTED_PUBLIC_BLOB
    ):
        raise P8RunnerError("P8 lock does not bind the official public corpus")
    priors = lock.get("priors")
    if not isinstance(priors, dict) or set(priors) != set(DEFAULT_PRIORS):
        raise P8RunnerError("P8 prior-corpus lock registry is incomplete")
    prior_paths: dict[str, Path] = {}
    prior_entries: dict[str, Any] = {}
    for name, entry in priors.items():
        path, frozen = _validate_file_entry(
            project_root, entry, f"prior {name}", extras={"rows"}
        )
        prior_paths[name] = path
        prior_entries[name] = frozen
    metadata_path, metadata_entry = _validate_file_entry(
        project_root, lock.get("corpus_metadata"), "corpus metadata"
    )
    corpora = lock.get("corpora")
    if not isinstance(corpora, dict) or set(corpora) != set(DEFAULT_CORPORA):
        raise P8RunnerError("P8 split lock registry is incomplete")
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
            raise P8RunnerError(f"P8 {split} count/scenario lock is invalid")
        corpus_paths[split] = path
        corpus_entries[split] = frozen
    if len({path.resolve() for path in corpus_paths.values()}) != 2:
        raise P8RunnerError("P8 selection and confirmation paths must differ")
    metadata = _load_json_object(metadata_path)
    metadata_summary = _validate_metadata(metadata, lock)
    if enforce_git:
        _assert_tracked(project_root, spec_path, "matrix spec")
    return {
        "source": source,
        "spec": spec_entry,
        "catalog": catalog_entry,
        "released_public": public_entry,
        "priors": prior_entries,
        "corpus_metadata": metadata_entry,
        "corpora": corpus_entries,
        "metadata_summary": metadata_summary,
        "paths": {
            "catalog": catalog_path,
            "released_public": public_path,
            "priors": prior_paths,
            "corpus_metadata": metadata_path,
            "corpora": corpus_paths,
        },
    }


def _git_snapshot(project_root: Path) -> dict[str, Any]:
    branch = str(_git(project_root, "branch", "--show-current"))
    head = str(_git(project_root, "rev-parse", "HEAD"))
    status = str(_git(project_root, "status", "--porcelain=v1", "--untracked-files=all"))
    if not branch or not head or status:
        raise P8RunnerError("P8 requires a named branch and completely clean worktree")
    origin_head = str(_git(project_root, "rev-parse", f"origin/{branch}"))
    if origin_head != head:
        raise P8RunnerError("P8 requires HEAD to equal pushed origin/<branch>")
    return {
        "branch": branch,
        "head": head,
        "origin_head": origin_head,
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
        spec_path != (project_root / "configs" / "p8_explicit_negative_matrix.json").resolve()
        or lock_path != (project_root / "configs" / "p8_prereg_lock.json").resolve()
    ):
        raise P8RunnerError("formal P8 requires the default frozen spec and lock")
    if not spec_path.is_file() or not lock_path.is_file():
        raise P8RunnerError("P8 spec and preregistration lock must exist")
    spec = validate_matrix_spec(_load_json_object(spec_path))
    lock = _load_json_object(lock_path)
    protocol = _validate_protocol_lock(
        project_root, lock, spec_path=spec_path, enforce_git=enforce_git
    )
    worker_path = project_root / "scripts" / "p8_worker.py"
    worker_boundary = validate_worker_source_boundary(worker_path)
    if enforce_git:
        _assert_tracked(project_root, lock_path, "P8 preregistration lock")
        git = _git_snapshot(project_root)
    else:
        git = {"branch": None, "head": None, "origin_head": None, "clean": None}
    all_paths = _flatten_protocol_paths(protocol, spec_path, lock_path, project_root)
    snapshot = _identity_snapshot(all_paths)
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
            "prior_sha256": {
                name: entry["sha256"] for name, entry in protocol["priors"].items()
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
        raise P8RunnerError(f"{label} contains an empty ground-truth identifier")
    return values


def _load_split(
    split: str,
    preflight_state: Mapping[str, Any],
    *,
    catalog_ids: set[str],
    excluded_targets: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    if split not in {"selection", "confirmation"}:
        raise P8RunnerError("unknown P8 split")
    protocol = preflight_state["protocol"]
    path = protocol["paths"]["corpora"][split]
    entry = protocol["corpora"][split]
    rows = _jsonl_rows(path)
    if len(rows) != entry["rows"] or _canonical_rows_sha256(rows) != entry["canonical_samples_sha256"]:
        raise P8RunnerError(f"P8 {split} parsed identity differs from its lock")
    identifiers = [str(row.get("sample_id") or "") for row in rows]
    targets = _target_values(rows, f"P8 {split}")
    counts = dict(sorted(Counter(str(row.get("scenario_type") or "") for row in rows).items()))
    if (
        len(set(identifiers)) != len(rows)
        or not all(identifiers)
        or len(set(targets)) != len(rows)
        or set(targets) - catalog_ids
        or set(targets) & excluded_targets
        or counts != entry["scenario_counts"]
    ):
        raise P8RunnerError(f"P8 {split} uniqueness, exclusion, or scenario gate failed")
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
        "released_public": protocol["paths"]["released_public"],
        **protocol["paths"]["priors"],
    }
    combined: set[str] = set()
    for name, path in sources.items():
        rows = _jsonl_rows(path)
        locked = protocol["released_public"] if name == "released_public" else protocol["priors"][name]
        targets = _target_values(rows, name)
        if len(rows) != locked["rows"] or len(set(targets)) != len(rows):
            raise P8RunnerError(f"P8 exclusion source {name} failed count/uniqueness")
        overlap = combined & set(targets)
        if overlap:
            raise P8RunnerError(f"P8 exclusion sources overlap at {name}")
        combined.update(targets)
    return combined


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
    forbidden = (
        "corpus", "public", "prior", "ground_truth", "sample_id", "scenario", "target"
    )
    if any(word in serialized for word in forbidden):
        raise P8RunnerError("sanitized worker spec contains parent-only material")
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
        "roles": preflight_state["spec"]["roles"],
    }
    serialized = _canonical_bytes(payload).decode("utf-8").lower()
    forbidden = ("corpus", "public", "prior", "ground_truth", "sample_id", "scenario", "target")
    if any(word in serialized for word in forbidden):
        raise P8RunnerError("sanitized worker lock contains parent-only material")
    return payload


@dataclass
class WorkerClient:
    role: str
    process: subprocess.Popen[str]
    nonce: str
    stderr_path: Path
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
        spec: Path,
        worker_lock: Path,
        worker_factory: str,
        rss_sample_ms: float,
        stderr_path: Path,
    ) -> "WorkerClient":
        if role not in WORKER_ROLES:
            raise P8RunnerError(f"unknown P8 role: {role}")
        nonce = uuid.uuid4().hex
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "p8_worker.py"),
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
            "--lock",
            str(worker_lock),
            "--rss-ms",
            str(rss_sample_ms),
        ]
        forbidden = ("ground_truth", "sample_id", "scenario", "selection", "confirmation", "public_set")
        joined = " ".join(command).lower()
        if any(word in joined for word in forbidden):
            raise P8RunnerError("P8 worker command leaks parent-only material")
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
        client = cls(role, process, nonce, stderr_path)
        ready = client._read_message()
        if ready != {"kind": "ready", "nonce": nonce, "role": role}:
            client.abort()
            raise P8RunnerError("invalid P8 worker ready message")
        return client

    def _read_message(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise P8RunnerError("worker stdout is unavailable")
        line = self.process.stdout.readline()
        if not line:
            self.process.wait(timeout=5)
            detail = self.stderr_path.read_text(encoding="utf-8", errors="replace")
            raise P8RunnerError(f"P8 worker exited unexpectedly: {detail[-1000:]}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise P8RunnerError("worker protocol message must be an object")
        return value

    def _request(self, operation: str, **payload: Any) -> Any:
        if self.process.stdin is None:
            raise P8RunnerError("worker stdin is unavailable")
        request_id = self._next_request_id
        self._next_request_id += 1
        request = {"request_id": request_id, "operation": operation, **payload}
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        reply = self._read_message()
        if reply.get("request_id") != request_id:
            raise P8RunnerError("worker reply request ID mismatch")
        if reply.get("kind") == "error":
            raise P8RunnerError(f"worker {operation} failed: {reply.get('error_class')}")
        if operation == "finalize":
            if reply.get("kind") != "result" or not isinstance(reply.get("bundle"), dict):
                raise P8RunnerError("worker finalize did not return a bundle")
            return reply["bundle"]
        if reply.get("kind") != "reply":
            raise P8RunnerError("worker returned an invalid reply")
        return reply.get("value")

    def reset(self, opaque_id: str, user_profile: dict[str, Any]) -> None:
        if opaque_id in self._ordinal_by_opaque_id:
            raise P8RunnerError("official driver reused an opaque conversation ID")
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        self._ordinal_by_opaque_id[opaque_id] = ordinal
        self._request("reset", ordinal=ordinal, user_profile=dict(user_profile))

    def respond(self, opaque_id: str, user_message: str, turn: int, top_k: int) -> dict[str, Any]:
        ordinal = self._ordinal_by_opaque_id.get(opaque_id)
        if ordinal is None:
            raise P8RunnerError("respond received an unknown conversation ID")
        value = self._request(
            "respond",
            ordinal=ordinal,
            user_message=user_message,
            turn=turn,
            top_k=top_k,
        )
        response = value.get("response") if isinstance(value, dict) else None
        if not isinstance(response, dict):
            raise P8RunnerError("worker response is not an object")
        self.response_count += 1
        self._response_digest.update(
            _canonical_bytes({"ordinal": ordinal, "turn": turn, "response": response}) + b"\n"
        )
        return response

    def finalize(self) -> dict[str, Any]:
        bundle = self._request("finalize")
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=30)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if return_code != 0:
            detail = self.stderr_path.read_text(encoding="utf-8", errors="replace")
            raise P8RunnerError(f"P8 worker failed after finalize: {detail[-1000:]}")
        expected_hash = self._response_digest.hexdigest()
        if bundle.get("response_count") != self.response_count or bundle.get("response_sha256") != expected_hash:
            raise P8RunnerError("parent and worker response captures disagree")
        bundle["worker_process"] = {
            "isolated": True,
            "pid": self.process.pid,
            "nonce": self.nonce,
            "role": self.role,
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


def _session_exact_values(session: Mapping[str, Any]) -> dict[str, int]:
    hit = bool(session.get("hit"))
    rank = session.get("best_rank")
    turn = session.get("first_hit_turn")
    if hit:
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= 10:
            raise P8RunnerError("hit conversation has an invalid Top-10 rank")
        if not isinstance(turn, int) or isinstance(turn, bool) or not 1 <= turn <= 10:
            raise P8RunnerError("hit conversation has an invalid first-hit turn")
        rr_units = RR_SCALE // rank
        mttc_turn = turn
        contribution = CONTRIBUTION_SCALE // 2 + 3 * rr_units + CONTRIBUTION_SCALE // 50 * (11 - turn)
    else:
        if rank is not None or turn is not None:
            raise P8RunnerError("miss conversation reports a rank or first-hit turn")
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
    keys = (
        "sample_count",
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
        "reported_token_usage",
        "scenario_metrics",
    )
    return {key: result.get(key) for key in keys}


def _exact_totals_match(run: Mapping[str, Any]) -> bool:
    totals = run["exact_totals"]
    metrics = run["metrics"]
    count = int(totals["sample_count"])
    expected = {
        "hit_rate_at_10": round(int(totals["hit_count"]) / count, 6),
        "mrr": round(int(totals["rr_sum_x2520"]) / (RR_SCALE * count), 6),
        "mttc": round(int(totals["mttc_turn_sum"]) / count, 6),
        "recommended_technical_score": round(
            int(totals["official_contribution_sum_x25200"]) / (CONTRIBUTION_SCALE * count), 6
        ),
    }
    return count > 0 and all(float(metrics.get(key, -1.0)) == value for key, value in expected.items())


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
    with tempfile.TemporaryDirectory(prefix="p8-isolated-") as directory:
        root = Path(directory)
        worker_spec = root / "worker-spec.json"
        worker_spec.write_bytes(_canonical_bytes(_worker_spec_payload(preflight_state)) + b"\n")
        worker_lock = root / "worker-lock.json"
        worker_lock.write_bytes(
            _canonical_bytes(
                _worker_lock_payload(
                    preflight_state, worker_spec_sha256=_sha256_file(worker_spec)
                )
            )
            + b"\n"
        )
        stderr_path = root / "stderr.log"
        worker = WorkerClient.start(
            role,
            catalog=protocol["paths"]["catalog"],
            spec=worker_spec,
            worker_lock=worker_lock,
            worker_factory=worker_factory or spec["worker_factory"],
            rss_sample_ms=float(spec["resource_limits"]["rss_sample_ms"]),
            stderr_path=stderr_path,
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
    sessions = official.get("sessions")
    if not isinstance(sessions, list) or any(not isinstance(item, dict) for item in sessions):
        raise P8RunnerError("official driver returned an invalid conversation ledger")
    capture_errors = bundle.pop("integrity_errors", [])
    capture_hashes = bundle.get("hashes", {})
    function_hashes = bundle.get("function_hashes", {})
    if not isinstance(capture_hashes, dict) or not isinstance(function_hashes, dict):
        raise P8RunnerError("worker capture hashes are invalid")
    return {
        "role": role,
        "configuration": bundle.get("configuration", {}),
        "stats": bundle.get("stats", {}),
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
            "generic_exception_count": bundle.get("generic_exception_count"),
            "generic_exception_classes_sha256": _hash_strings(
                bundle.get("generic_exception_classes", [])
            ),
        },
        "timing": {
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
        raise P8RunnerError("P8 runs contain different conversation identifiers")
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
    return {
        "contract_clean": run.get("contract", {}).get("error_count") == 0,
        "integrity_clean": run.get("integrity", {}).get("error_count") == 0,
        "network_attempts_zero": run.get("runtime", {}).get("network_attempt_count") == 0,
        "generic_exceptions_zero": run.get("runtime", {}).get("generic_exception_count") == 0,
        "lab_exception_counters_zero": isinstance(stats, dict) and _all_exception_counters_zero(stats),
        "complete_official_aggregate": run.get("exact_totals", {}).get("sample_count") == expected_count,
        "exact_totals_match_official_metrics": _exact_totals_match(run),
        "fresh_external_process": (
            worker.get("isolated") is True
            and isinstance(worker.get("pid"), int)
            and worker.get("pid") != os.getpid()
            and re.fullmatch(r"[a-f0-9]{32}", str(worker.get("nonce") or "")) is not None
        ),
    }


def gate_baseline(
    run: Mapping[str, Any], expected_count: int, spec: Mapping[str, Any]
) -> dict[str, Any]:
    configuration = run.get("configuration", {})
    served = spec["served_control"]
    gates = {
        **_common_gates(run, expected_count),
        "role_exact": run.get("role") == BASELINE_ROLE,
        "served_coverage_off_fast_exact": all(
            configuration.get(key) == value for key, value in served.items()
        ),
    }
    return {
        "decision": "served_reference" if all(gates.values()) else "invalid_served_reference",
        "gates": gates,
    }


def gate_control(
    run: Mapping[str, Any],
    reference: Mapping[str, Any],
    expected_count: int,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    configuration = run.get("configuration", {})
    served = spec["served_control"]
    gates = {
        **_common_gates(run, expected_count),
        "role_exact": run.get("role") == ROLES["control"],
        "served_coverage_off_fast_exact": all(configuration.get(key) == value for key, value in served.items()),
        "functional_output_equals_served_agent": (
            run.get("functional_result_sha256") == reference.get("functional_result_sha256")
        ),
        "response_trace_equals_served_agent": (
            run.get("response_trace_sha256") == reference.get("response_trace_sha256")
        ),
        "exact_totals_equal_served_agent": (
            run.get("exact_totals") == reference.get("exact_totals")
        ),
    }
    return {"decision": "control" if all(gates.values()) else "invalid_control", "gates": gates}


def gate_shadow(
    run: Mapping[str, Any], control: Mapping[str, Any], expected_count: int
) -> dict[str, Any]:
    stats = run.get("stats", {})
    gates = {
        **_common_gates(run, expected_count),
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
    wall = float(run.get("timing", {}).get("evaluation_wall_seconds") or 0.0)
    base_wall = float(control.get("timing", {}).get("evaluation_wall_seconds") or 0.0)
    p95 = float(run.get("timing", {}).get("respond_latency", {}).get("p95_ms") or 0.0)
    base_p95 = float(control.get("timing", {}).get("respond_latency", {}).get("p95_ms") or 0.0)
    peak = run.get("memory", {}).get("peak_rss_bytes")
    base_peak = control.get("memory", {}).get("peak_rss_bytes")
    return {
        "wall_within_1_30x": base_wall > 0 and wall <= float(limits["wall_ratio"]) * base_wall,
        "response_p95_within_1_30x": base_p95 > 0 and p95 <= float(limits["response_p95_ratio"]) * base_p95,
        "peak_rss_within_1_20x": (
            run.get("memory", {}).get("available") is True
            and control.get("memory", {}).get("available") is True
            and isinstance(peak, int)
            and isinstance(base_peak, int)
            and base_peak > 0
            and peak <= float(limits["peak_rss_ratio"]) * base_peak
        ),
    }


def gate_active(
    run: Mapping[str, Any],
    control: Mapping[str, Any],
    expected_count: int,
    spec: Mapping[str, Any],
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
                "hit_count",
                "rr_sum_x2520",
                "mttc_turn_sum",
                "official_contribution_sum_x25200",
            )
        },
        "session_changes_vs_control": changes,
        "scenario_hit_count_regressions": scenario_regressions,
    }


def repeat_exact(initial: Mapping[str, Any], repeated: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "different_worker_nonce": (
            initial.get("worker_process", {}).get("nonce")
            != repeated.get("worker_process", {}).get("nonce")
        ),
        "functional_result_exact": initial.get("functional_result_sha256") == repeated.get("functional_result_sha256"),
        "response_trace_exact": initial.get("response_trace_sha256") == repeated.get("response_trace_sha256"),
        "exact_totals_exact": initial.get("exact_totals") == repeated.get("exact_totals"),
        "capture_hashes_exact": initial.get("capture_hashes") == repeated.get("capture_hashes"),
        "function_hashes_exact": initial.get("function_hashes") == repeated.get("function_hashes"),
        "contract_exact": initial.get("contract") == repeated.get("contract"),
        "integrity_exact": initial.get("integrity") == repeated.get("integrity"),
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
    gates = {
        BASELINE_ROLE: gate_baseline(runs[BASELINE_ROLE], count, preflight_state["spec"]),
        ROLES["control"]: gate_control(
            runs[ROLES["control"]], runs[BASELINE_ROLE], count, preflight_state["spec"]
        ),
        ROLES["shadow"]: gate_shadow(runs[ROLES["shadow"]], runs[ROLES["control"]], count),
        ROLES["active"]: gate_active(
            runs[ROLES["active"]], runs[ROLES["control"]], count, preflight_state["spec"]
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
    baseline_gate = gate_baseline(
        repeated[BASELINE_ROLE], len(samples), preflight_state["spec"]
    )
    control_gate = gate_control(
        repeated[ROLES["control"]],
        repeated[BASELINE_ROLE],
        len(samples),
        preflight_state["spec"],
    )
    active_gate = gate_active(
        repeated[ROLES["active"]],
        repeated[ROLES["control"]],
        len(samples),
        preflight_state["spec"],
    )
    exact = {
        role: repeat_exact(initial[role], repeated[role])
        for role in repeated
    }
    return {
        "attempted": True,
        "passed": (
            control_gate["decision"] == "control"
            and baseline_gate["decision"] == "served_reference"
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
    return {
        key: value
        for key, value in run.items()
        if key != "_sessions"
    }


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
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in ARTIFACT_FORBIDDEN_KEYS:
                raise P8RunnerError(f"P8 artifact contains prohibited key: {key}")
            _assert_artifact_safe(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_artifact_safe(nested)


def run_evaluation(
    *,
    spec_path: Path = DEFAULT_SPEC,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    before = preflight(spec_path=spec_path, lock_path=lock_path)
    protocol = before["protocol"]
    catalog_ids, categories, products = catalog_index(protocol["paths"]["catalog"])
    excluded = _prior_target_set(before)
    selection_rows, selection_corpus, selection_targets = _load_split(
        "selection", before, catalog_ids=catalog_ids, excluded_targets=excluded
    )
    selection_runs, selection_gates = _run_initial_split(
        selection_rows,
        catalog_ids,
        categories,
        products,
        preflight_state=before,
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
            preflight_state=before,
            spec_path=spec_path,
        )
    selection_passed = bool(selection_ready and selection_repeat.get("passed"))

    confirmation_artifact: dict[str, Any] = {
        "opened": False,
        "reason": "selection did not pass eligibility and exact-repeat gates",
    }
    promotion = False
    if selection_passed:
        confirmation_rows, confirmation_corpus, _ = _load_split(
            "confirmation",
            before,
            catalog_ids=catalog_ids,
            excluded_targets=excluded | selection_targets,
        )
        confirmation_runs, confirmation_gates = _run_initial_split(
            confirmation_rows,
            catalog_ids,
            categories,
            products,
            preflight_state=before,
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
                preflight_state=before,
                spec_path=spec_path,
            )
        promotion = bool(confirmation_ready and confirmation_repeat.get("passed"))
        confirmation_artifact = {
            "opened": True,
            **_split_artifact(
                confirmation_corpus,
                confirmation_runs,
                confirmation_gates,
                confirmation_repeat,
            ),
        }

    after = preflight(spec_path=spec_path, lock_path=lock_path)
    if before["git"] != after["git"] or before["identity_snapshot"] != after["identity_snapshot"]:
        raise P8RunnerError("P8 source or input identity changed during evaluation")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "decision": "promote_p8_r01" if promotion else "retain_p8_c00",
        "winner_id": ROLES["active"] if promotion else ROLES["control"],
        "public_evaluation_run": False,
        "inputs": {
            **before["summary"],
            "confirmation_rows_parsed": selection_passed,
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
            "aggregates. Each role runs in a fresh offline process with a sanitized lock. "
            "The artifact contains aggregate metrics, exact totals, gates, resources, and hashes only."
        ),
    }
    _assert_artifact_safe(artifact)
    return artifact


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"P8 output already exists: {path}")
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
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-preflight",
        action="store_true",
        help=(
            "validate raw file identities and Git gates without parsing conversation "
            "rows or running either split"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_preflight:
        state = preflight(spec_path=args.spec, lock_path=args.lock)
        print(json.dumps(state["summary"], indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    artifact = run_evaluation(spec_path=args.spec, lock_path=args.lock)
    _atomic_write_json(args.output, artifact)
    print(
        f"[p8] decision={artifact['decision']} winner={artifact['winner_id']} wrote={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
