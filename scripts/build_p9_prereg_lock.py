from __future__ import annotations

"""Create the immutable P9 preregistration lock without running conversations."""

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "p9.prereg-lock.v1"
SPEC_SCHEMA_VERSION = "p9.compact-negative-matrix.v1"
METADATA_SCHEMA_VERSION = "p9.explicit-negative-corpora.v1"
EVIDENCE_SCHEMA_VERSION = "p9.compact-negative-evidence.v1"
EXPECTED_CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
EXPECTED_PUBLIC_GIT_BLOB_SHA1 = "121dbec9c1368c81cd887d6959e62507512139c0"
EXPECTED_EVIDENCE_BYTES = 1_486_848
EXPECTED_EVIDENCE_SHA256 = "2bc5846b7f6efb2e8395ea99b6bca5b585fb1507d23d6289dbc00d7600d22128"
EXPECTED_EVIDENCE_METADATA_SHA256 = "24c585d0b6b16a532bee68d2122f8bd468dfcc6889f8b2282377886e770601a2"
EXPECTED_REGISTRY_SHA256 = "6e007f76e29aa97d06de7aa8c65f4cfe4fe505a8ec9c04e131d971bef9892fe6"
EXPECTED_SEMANTICS_SHA256 = "a527cb016e64e87fe3edfc571a9793700ffabcfe75fc31893e531a584dd54a31"
EXPECTED_ORIGIN_URL = "https://github.com/lamperriat/techjam-err402.git"
EXPECTED_ORIGIN_URL_SHA256 = "bc84c0d712a5e9f381a0738891e461982e3cb0c52bc9d5e601c6abcbf0f860e5"
EVIDENCE_MAX_BYTES = 16_777_216
EXPECTED_CANONICAL_SHA256 = {
    "released_public": "6c726257fec25575716ee65b095f94c48402b6e14e83341518610f45fbfbec6d",
    "p1": "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae",
    "p5": "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c",
    "p6": "27544cdb6ed9495808c35bbab09b4dbadcb88a1d75d162f17bb4fba6ee8841c7",
    "p7": "bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546",
    "p8_selection": "1c11d73d7c8ced617ce874e15a563f240731ca9654ed42bcc4f773b7b4da81ee",
    "p8_confirmation": "3ae6f8ff7ab0362399b348c3443daa5b7138aab9cf72e944b7e11dd71d7d3dde",
    "selection": "6298cbd6d7507f4b163ab4979a86ff109e0dffa90557e3b28e5d20d129e5be9f",
    "confirmation": "4bbd9d53f32e3773de18bab881ba6e5ef0887ca86701897798ee086430ed08d9",
}
EXPECTED_SCENARIO_COUNTS = {
    "boundary": 10,
    "browsing": 80,
    "buying": 80,
    "intent_override": 30,
}
SOURCE_PATHS = {
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


class PreregLockError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenExpectations:
    catalog_sha256: str = EXPECTED_CATALOG_SHA256
    catalog_rows: int = 50_000
    public_git_blob_sha1: str = EXPECTED_PUBLIC_GIT_BLOB_SHA1
    public_rows: int = 200
    prior_rows: int = 200
    split_rows: int = 200
    evidence_bytes: int = EXPECTED_EVIDENCE_BYTES
    evidence_sha256: str = EXPECTED_EVIDENCE_SHA256
    evidence_metadata_sha256: str = EXPECTED_EVIDENCE_METADATA_SHA256
    registry_sha256: str = EXPECTED_REGISTRY_SHA256
    semantics_sha256: str = EXPECTED_SEMANTICS_SHA256
    canonical_sha256: Mapping[str, str] | None = None
    scenario_counts: Mapping[str, int] | None = None

    def canonical(self) -> dict[str, str]:
        return dict(self.canonical_sha256 or EXPECTED_CANONICAL_SHA256)

    def scenarios(self) -> dict[str, int]:
        return dict(self.scenario_counts or EXPECTED_SCENARIO_COUNTS)


@dataclass(frozen=True)
class LockPaths:
    spec: Path
    catalog: Path
    released_public: Path
    priors: Mapping[str, Path]
    evidence: Path
    evidence_metadata: Path
    corpus_metadata: Path
    corpora: Mapping[str, Path]
    output: Path


def default_paths(project_root: Path = PROJECT_ROOT) -> LockPaths:
    root = project_root.resolve()
    return LockPaths(
        spec=root / "configs" / "p9_compact_negative_matrix.json",
        catalog=root / "data" / "catalog.jsonl",
        released_public=root / "data" / "public_set.jsonl",
        priors={
            "p1": root / "experiments" / "p1_derived_product_disjoint.jsonl",
            "p5": root / "experiments" / "p5_selection_product_disjoint.jsonl",
            "p6": root / "experiments" / "p6_selection_product_disjoint.jsonl",
            "p7": root / "experiments" / "p7_selection_product_disjoint.jsonl",
            "p8_selection": root / "experiments" / "p8_selection_product_disjoint.jsonl",
            "p8_confirmation": root / "experiments" / "p8_confirmation_product_disjoint.jsonl",
        },
        evidence=root / "experiments" / "p9_negative_evidence.sqlite",
        evidence_metadata=root / "experiments" / "p9_negative_evidence.metadata.json",
        corpus_metadata=root / "experiments" / "p9_explicit_negative_corpora.metadata.json",
        corpora={
            "selection": root / "experiments" / "p9_selection_product_disjoint.jsonl",
            "confirmation": root / "experiments" / "p9_confirmation_product_disjoint.jsonl",
        },
        output=root / "configs" / "p9_prereg_lock.json",
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise PreregLockError(f"locked path escapes project root: {path}") from exc
    if not resolved.is_file():
        raise PreregLockError(f"locked file is missing: {resolved}")
    return {"path": relative, "bytes": resolved.stat().st_size, "sha256": _sha256_file(resolved)}


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreregLockError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise PreregLockError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise PreregLockError(f"JSONL row is not an object: {path}")
                    rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreregLockError(f"invalid JSONL: {path}") from exc
    return rows


def _canonical_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_bytes(row) + b"\n")
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _raw_git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


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
        raise PreregLockError(f"Git command timed out: {' '.join(arguments)}") from exc
    if completed.returncode:
        raise PreregLockError(f"Git command failed: {' '.join(arguments)}")
    return completed.stdout if binary else str(completed.stdout).strip()


def capture_pushed_clean_revision(project_root: Path) -> dict[str, Any]:
    branch = str(_git(project_root, "branch", "--show-current"))
    head = str(_git(project_root, "rev-parse", "HEAD"))
    status = str(_git(project_root, "status", "--porcelain=v1", "--untracked-files=all"))
    if not branch:
        raise PreregLockError("P9 lock requires a named Git branch")
    if re.fullmatch(r"[a-f0-9]{40}", head) is None:
        raise PreregLockError("P9 lock requires a valid HEAD commit")
    if status:
        raise PreregLockError("P9 lock requires a completely clean worktree")
    origin_url = str(_git(project_root, "remote", "get-url", "origin"))
    if origin_url != EXPECTED_ORIGIN_URL:
        raise PreregLockError("P9 lock requires the credential-free official HTTPS origin")
    origin_url_sha256 = hashlib.sha256(origin_url.encode("utf-8")).hexdigest()
    if origin_url_sha256 != EXPECTED_ORIGIN_URL_SHA256:
        raise PreregLockError("P9 official origin URL digest is not frozen")
    head_ref = f"refs/heads/{branch}"
    advertised = str(_git(project_root, "ls-remote", "--heads", "origin", head_ref))
    lines = [line for line in advertised.splitlines() if line.strip()]
    match = re.fullmatch(rf"([a-f0-9]{{40}})\t{re.escape(head_ref)}", lines[0]) if len(lines) == 1 else None
    if match is None or match.group(1) != head:
        raise PreregLockError("P9 lock requires direct origin branch proof to equal HEAD")
    return {
        "git_commit": head,
        "git_branch": branch,
        "remote_proof": {
            "remote": "origin",
            "head_ref": head_ref,
            "advertised_head": match.group(1),
            "url_sha256": origin_url_sha256,
            "verified": True,
        },
    }


def _required_paths_from_runner(runner_path: Path) -> dict[str, str]:
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SOURCE_PATHS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if (
                isinstance(value, dict)
                and value
                and all(isinstance(key, str) and isinstance(path, str) for key, path in value.items())
            ):
                return dict(value)
    raise PreregLockError("evaluate_p9.REQUIRED_SOURCE_PATHS must be a literal string map")


def _tracked_head_identity(
    path: Path, project_root: Path, *, include_git_blob: bool = False
) -> dict[str, Any]:
    entry = _file_entry(path, project_root)
    relative = entry["path"]
    _git(project_root, "ls-files", "--error-unmatch", relative)
    working_blob = str(_git(project_root, "hash-object", f"--path={relative}", "--", relative))
    head_blob = str(_git(project_root, "rev-parse", f"HEAD:{relative}"))
    if re.fullmatch(r"[a-f0-9]{40}", working_blob) is None or working_blob != head_blob:
        raise PreregLockError(f"working content differs from HEAD Git blob: {relative}")
    return {**entry, **({"git_blob_sha1": head_blob} if include_git_blob else {})}


def capture_source_lock(project_root: Path, revision: Mapping[str, Any]) -> dict[str, Any]:
    required = _required_paths_from_runner(project_root / SOURCE_PATHS["evaluate_p9"])
    if required != SOURCE_PATHS:
        raise PreregLockError("lock builder source registry differs from evaluate_p9")
    return {
        "git_commit": revision["git_commit"],
        "git_branch": revision["git_branch"],
        "remote_proof": revision["remote_proof"],
        "files": {
            name: _tracked_head_identity(project_root / relative, project_root, include_git_blob=True)
            for name, relative in sorted(required.items())
        },
    }


def _inspect_conversation_rows(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    targets = [str(row.get("ground_truth", {}).get("parent_asin") or "").strip() for row in rows]
    if not rows or not all(targets) or len(set(targets)) != len(targets):
        raise PreregLockError(f"conversation rows have empty or duplicate targets: {path}")
    scenarios = dict(sorted(Counter(str(row.get("scenario_type") or "") for row in rows).items()))
    return {
        "rows": rows,
        "row_count": len(rows),
        "canonical_samples_sha256": _canonical_rows_sha256(rows),
        "scenario_counts": scenarios,
        "target_ids": set(targets),
    }


def _overlap_values(value: Any, key: str = "") -> list[Any]:
    if isinstance(value, Mapping):
        found: list[Any] = []
        for nested_key, nested in value.items():
            found.extend(_overlap_values(nested, str(nested_key)))
        return found
    return [value] if "overlap" in key.lower() else []


def _validate_corpus_metadata(
    metadata: Mapping[str, Any],
    *,
    catalog_sha256: str,
    catalog_rows: int,
    inputs: Mapping[str, Mapping[str, Any]],
    corpora: Mapping[str, Mapping[str, Any]],
    public_blob: str,
) -> None:
    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise PreregLockError("P9 corpus metadata schema is invalid")
    catalog = metadata.get("catalog_source")
    metadata_inputs = metadata.get("input_sources")
    metadata_corpora = metadata.get("corpora")
    exclusions = metadata.get("exclusions")
    outputs = metadata.get("outputs")
    if not all(
        isinstance(value, dict)
        for value in (catalog, metadata_inputs, metadata_corpora, exclusions, outputs)
    ):
        raise PreregLockError("P9 corpus metadata sections are incomplete")
    assert isinstance(catalog, dict) and isinstance(metadata_inputs, dict)
    assert isinstance(metadata_corpora, dict) and isinstance(outputs, dict)
    if (
        catalog.get("sha256") != catalog_sha256
        or catalog.get("loaded_product_count") != catalog_rows
        or catalog.get("frozen_sha256_verified") is not True
        or catalog.get("expected_count_verified") is not True
    ):
        raise PreregLockError("P9 metadata catalog identity is invalid")
    names = {
        "released_public": "released_public",
        "prior_p1_derived": "p1",
        "prior_p5_derived": "p5",
        "prior_p6_derived": "p6",
        "prior_p7_derived": "p7",
        "prior_p8_selection": "p8_selection",
        "prior_p8_confirmation": "p8_confirmation",
    }
    if set(metadata_inputs) != set(names):
        raise PreregLockError("P9 metadata input registry is incomplete")
    for metadata_name, source_name in names.items():
        entry = metadata_inputs[metadata_name]
        observed = inputs[source_name]
        if (
            not isinstance(entry, dict)
            or entry.get("sample_count") != observed["row_count"]
            or entry.get("unique_target_count") != observed["row_count"]
            or entry.get("canonical_samples_sha256") != observed["canonical_samples_sha256"]
            or entry.get("frozen_samples_sha256_verified") is not True
        ):
            raise PreregLockError(f"P9 metadata identity is invalid for {metadata_name}")
    public_entry = metadata_inputs["released_public"]
    if (
        public_entry.get("git_blob_sha1_lf") != public_blob
        or public_entry.get("frozen_git_blob_verified") is not True
    ):
        raise PreregLockError("P9 metadata public Git blob is invalid")
    if set(metadata_corpora) != {"selection", "confirmation"}:
        raise PreregLockError("P9 metadata corpus registry is incomplete")
    for split, observed in corpora.items():
        entry = metadata_corpora[split]
        if (
            not isinstance(entry, dict)
            or entry.get("sample_count") != observed["row_count"]
            or entry.get("unique_target_count") != observed["row_count"]
            or entry.get("samples_sha256") != observed["canonical_samples_sha256"]
            or entry.get("scenario_counts") != observed["scenario_counts"]
        ):
            raise PreregLockError(f"P9 metadata identity is invalid for {split}")
        output = outputs.get(split)
        if (
            not isinstance(output, dict)
            or output.get("expected_frozen_samples_sha256") != observed["canonical_samples_sha256"]
            or output.get("samples_file_sha256") != observed["canonical_samples_sha256"]
            or output.get("frozen_samples_sha256_verified") is not True
        ):
            raise PreregLockError(f"P9 metadata frozen output proof is invalid for {split}")
    overlaps = _overlap_values(exclusions)
    if not overlaps or any(value != 0 for value in overlaps):
        raise PreregLockError("P9 metadata does not prove zero target overlap")


def _validate_evidence_metadata(
    metadata: Mapping[str, Any],
    *,
    catalog_sha256: str,
    catalog_rows: int,
    evidence_bytes: int,
    evidence_sha256: str,
    registry_sha256: str,
    semantics_sha256: str,
) -> None:
    catalog = metadata.get("catalog")
    evidence = metadata.get("evidence")
    if (
        metadata.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or metadata.get("target_blind") is not True
        or metadata.get("label_free") is not True
        or not isinstance(catalog, dict)
        or catalog.get("sha256") != catalog_sha256
        or catalog.get("rows") != catalog_rows
        or not isinstance(evidence, dict)
        or evidence.get("bytes") != evidence_bytes
        or evidence.get("sha256") != evidence_sha256
        or evidence.get("registry_sha256") != registry_sha256
        or evidence.get("semantics_sha256") != semantics_sha256
    ):
        raise PreregLockError("P9 evidence metadata differs from the frozen asset")


def _assert_target_free(value: Any, target_ids: set[str]) -> None:
    payload = _canonical_bytes(value).decode("utf-8")
    leaked = next((identifier for identifier in target_ids if identifier in payload), None)
    if leaked is not None:
        raise PreregLockError("P9 preregistration lock contains a target identifier")
    prohibited = {"ground_truth", "target", "target_id", "target_asin", "parent_asin", "sample_id"}

    def visit(nested: Any) -> None:
        if isinstance(nested, Mapping):
            for key, child in nested.items():
                if str(key).lower() in prohibited:
                    raise PreregLockError(f"P9 preregistration lock contains prohibited key: {key}")
                visit(child)
        elif isinstance(nested, (list, tuple)):
            for child in nested:
                visit(child)

    visit(value)


def _source_identifier_scan(
    source: Mapping[str, Any], project_root: Path, identifiers: set[str]
) -> dict[str, Any]:
    files = source.get("files")
    if not isinstance(files, Mapping) or not identifiers:
        raise PreregLockError("P9 source identifier scan inputs are incomplete")
    needles = tuple(identifier.encode("utf-8").lower() for identifier in sorted(identifiers))
    match_count = 0
    proof_files: list[dict[str, Any]] = []
    for name, entry in sorted(files.items()):
        if not isinstance(entry, Mapping):
            raise PreregLockError("P9 source identifier scan encountered an invalid entry")
        path = project_root / str(entry.get("path") or "")
        payload = path.read_bytes().lower()
        matched = any(needle in payload for needle in needles)
        match_count += int(matched)
        proof_files.append(
            {"name": name, "bytes": entry.get("bytes"), "sha256": entry.get("sha256")}
        )
    if match_count:
        raise PreregLockError("P9 frozen source hardcodes a locked product identifier")
    proof = {
        "source_files": proof_files,
        "source_file_count": len(proof_files),
        "identifier_count": len(identifiers),
        "match_count": 0,
    }
    return {
        "source_file_count": len(proof_files),
        "identifier_count": len(identifiers),
        "match_count": 0,
        "passed": True,
        "proof_sha256": hashlib.sha256(_canonical_bytes(proof)).hexdigest(),
    }


def build_prereg_lock(
    *,
    project_root: Path = PROJECT_ROOT,
    paths: LockPaths | None = None,
    expectations: FrozenExpectations | None = None,
    enforce_git: bool = True,
    require_defaults: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    paths = paths or default_paths(root)
    expectations = expectations or FrozenExpectations()
    canonical_expected = expectations.canonical()
    scenario_expected = expectations.scenarios()
    if paths.output.exists():
        raise FileExistsError(f"P9 preregistration lock already exists: {paths.output}")
    if require_defaults and paths != default_paths(root):
        raise PreregLockError("formal P9 lock generation requires default paths")
    if set(paths.priors) != {
        "p1", "p5", "p6", "p7", "p8_selection", "p8_confirmation"
    } or set(paths.corpora) != {"selection", "confirmation"}:
        raise PreregLockError("P9 lock input registry is incomplete")

    if enforce_git:
        revision = capture_pushed_clean_revision(root)
        source = capture_source_lock(root, revision)
        spec_entry = _tracked_head_identity(paths.spec, root)
    else:
        revision = {
            "git_commit": "0" * 40,
            "git_branch": "fixture",
            "remote_proof": {
                "remote": "fixture",
                "head_ref": "refs/heads/fixture",
                "advertised_head": "0" * 40,
                "url_sha256": "0" * 64,
                "verified": False,
            },
        }
        required = _required_paths_from_runner(root / SOURCE_PATHS["evaluate_p9"])
        source = {
            "git_commit": revision["git_commit"],
            "git_branch": revision["git_branch"],
            "remote_proof": revision["remote_proof"],
            "files": {
                name: {
                    **_file_entry(root / relative, root),
                    "git_blob_sha1": _raw_git_blob_sha1(root / relative),
                }
                for name, relative in sorted(required.items())
            },
        }
        spec_entry = _file_entry(paths.spec, root)

    spec = _load_json_object(paths.spec)
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise PreregLockError("P9 matrix spec schema is invalid")
    mechanism = spec.get("mechanism")
    asset_spec = mechanism.get("evidence_asset") if isinstance(mechanism, dict) else None
    resource_limits = spec.get("resource_limits")
    if (
        not isinstance(asset_spec, dict)
        or asset_spec.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or asset_spec.get("registry_sha256") != expectations.registry_sha256
        or asset_spec.get("semantics_sha256") != expectations.semantics_sha256
        or asset_spec.get("catalog_only") is not True
        or asset_spec.get("label_free") is not True
        or asset_spec.get("maximum_bytes") != EVIDENCE_MAX_BYTES
        or resource_limits
        != {
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
    ):
        raise PreregLockError("P9 matrix does not bind the frozen evidence/bootstrap contract")
    catalog_entry = _file_entry(paths.catalog, root)
    with paths.catalog.open("rb") as handle:
        catalog_rows = sum(1 for line in handle if line.strip())
    if catalog_entry["sha256"] != expectations.catalog_sha256 or catalog_rows != expectations.catalog_rows:
        raise PreregLockError("official catalog identity differs from frozen expectations")

    public = _inspect_conversation_rows(paths.released_public)
    public_blob = _git_blob_sha1(paths.released_public)
    if (
        public["row_count"] != expectations.public_rows
        or public_blob != expectations.public_git_blob_sha1
        or public["canonical_samples_sha256"] != canonical_expected["released_public"]
    ):
        raise PreregLockError("official public corpus identity differs from frozen expectations")
    prior_observations = {name: _inspect_conversation_rows(path) for name, path in paths.priors.items()}
    for name, observed in prior_observations.items():
        if (
            observed["row_count"] != expectations.prior_rows
            or observed["canonical_samples_sha256"] != canonical_expected[name]
        ):
            raise PreregLockError(f"prior {name} identity differs from frozen expectations")
    corpus_observations = {split: _inspect_conversation_rows(path) for split, path in paths.corpora.items()}
    for split, observed in corpus_observations.items():
        if (
            observed["row_count"] != expectations.split_rows
            or observed["canonical_samples_sha256"] != canonical_expected[split]
            or observed["scenario_counts"] != scenario_expected
            or _sha256_file(paths.corpora[split]) != observed["canonical_samples_sha256"]
        ):
            raise PreregLockError(f"P9 {split} identity differs from frozen expectations")

    evidence_entry = _file_entry(paths.evidence, root)
    if (
        evidence_entry["bytes"] != expectations.evidence_bytes
        or evidence_entry["sha256"] != expectations.evidence_sha256
        or evidence_entry["bytes"] > EVIDENCE_MAX_BYTES
    ):
        raise PreregLockError("P9 evidence identity or 16 MiB size gate failed")
    evidence_metadata_entry = _file_entry(paths.evidence_metadata, root)
    if evidence_metadata_entry["sha256"] != expectations.evidence_metadata_sha256:
        raise PreregLockError("P9 evidence metadata identity differs from frozen expectations")
    _validate_evidence_metadata(
        _load_json_object(paths.evidence_metadata),
        catalog_sha256=catalog_entry["sha256"],
        catalog_rows=catalog_rows,
        evidence_bytes=evidence_entry["bytes"],
        evidence_sha256=evidence_entry["sha256"],
        registry_sha256=expectations.registry_sha256,
        semantics_sha256=expectations.semantics_sha256,
    )

    inputs = {"released_public": public, **prior_observations}
    _validate_corpus_metadata(
        _load_json_object(paths.corpus_metadata),
        catalog_sha256=catalog_entry["sha256"],
        catalog_rows=catalog_rows,
        inputs=inputs,
        corpora=corpus_observations,
        public_blob=public_blob,
    )
    all_target_ids: set[str] = set()
    for observed in (*inputs.values(), *corpus_observations.values()):
        overlap = all_target_ids & observed["target_ids"]
        if overlap:
            raise PreregLockError("P9 locked corpora are not product-disjoint, including P8")
        all_target_ids.update(observed["target_ids"])
    source_target_scan = _source_identifier_scan(source, root, all_target_ids)

    lock = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "source_target_scan": source_target_scan,
        "spec": spec_entry,
        "catalog": {**catalog_entry, "rows": catalog_rows},
        "released_public": {
            **_file_entry(paths.released_public, root),
            "rows": public["row_count"],
            "git_blob_sha1_lf": public_blob,
        },
        "priors": {
            name: {**_file_entry(paths.priors[name], root), "rows": observed["row_count"]}
            for name, observed in sorted(prior_observations.items())
        },
        "evidence": {
            **evidence_entry,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "registry_sha256": expectations.registry_sha256,
            "semantics_sha256": expectations.semantics_sha256,
            "catalog_rows": catalog_rows,
        },
        "evidence_metadata": evidence_metadata_entry,
        "corpus_metadata": _file_entry(paths.corpus_metadata, root),
        "corpora": {
            split: {
                **_file_entry(paths.corpora[split], root),
                "rows": observed["row_count"],
                "canonical_samples_sha256": observed["canonical_samples_sha256"],
                "scenario_counts": observed["scenario_counts"],
            }
            for split, observed in sorted(corpus_observations.items())
        },
    }
    _assert_target_free(lock, all_target_ids)
    return lock


def atomic_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"P9 preregistration lock already exists: {path}")
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
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=defaults.spec)
    parser.add_argument("--catalog", type=Path, default=defaults.catalog)
    parser.add_argument("--public-set", type=Path, default=defaults.released_public)
    for name in ("p1", "p5", "p6", "p7", "p8_selection", "p8_confirmation"):
        parser.add_argument(f"--prior-{name.replace('_', '-')}", type=Path, default=defaults.priors[name])
    parser.add_argument("--evidence", type=Path, default=defaults.evidence)
    parser.add_argument("--evidence-metadata", type=Path, default=defaults.evidence_metadata)
    parser.add_argument("--metadata", type=Path, default=defaults.corpus_metadata)
    parser.add_argument("--selection", type=Path, default=defaults.corpora["selection"])
    parser.add_argument("--confirmation", type=Path, default=defaults.corpora["confirmation"])
    parser.add_argument("--output", type=Path, default=defaults.output)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = LockPaths(
        spec=args.spec.resolve(),
        catalog=args.catalog.resolve(),
        released_public=args.public_set.resolve(),
        priors={
            name: getattr(args, f"prior_{name}").resolve()
            for name in ("p1", "p5", "p6", "p7", "p8_selection", "p8_confirmation")
        },
        evidence=args.evidence.resolve(),
        evidence_metadata=args.evidence_metadata.resolve(),
        corpus_metadata=args.metadata.resolve(),
        corpora={"selection": args.selection.resolve(), "confirmation": args.confirmation.resolve()},
        output=args.output.resolve(),
    )
    lock = build_prereg_lock(paths=paths)
    atomic_create(paths.output, lock)
    print(f"[p9-lock] commit={lock['source']['git_commit']} wrote={paths.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
