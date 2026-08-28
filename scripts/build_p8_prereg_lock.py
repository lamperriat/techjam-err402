from __future__ import annotations

"""Create the immutable P8 preregistration lock without running any conversations."""

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_p8_selection_corpus import (  # noqa: E402
    P8_CONFIRMATION_FROZEN_SAMPLES_SHA256,
    P8_SELECTION_FROZEN_SAMPLES_SHA256,
)

SCHEMA_VERSION = "p8.prereg-lock.v1"
SPEC_SCHEMA_VERSION = "p8.explicit-negative-matrix.v1"
METADATA_SCHEMA_VERSION = "p8.explicit-negative-corpora.v1"
EXPECTED_CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
EXPECTED_PUBLIC_GIT_BLOB_SHA1 = "121dbec9c1368c81cd887d6959e62507512139c0"
EXPECTED_CANONICAL_SHA256 = {
    "released_public": "6c726257fec25575716ee65b095f94c48402b6e14e83341518610f45fbfbec6d",
    "p1": "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae",
    "p5": "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c",
    "p6": "27544cdb6ed9495808c35bbab09b4dbadcb88a1d75d162f17bb4fba6ee8841c7",
    "p7": "bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546",
    "selection": P8_SELECTION_FROZEN_SAMPLES_SHA256,
    "confirmation": P8_CONFIRMATION_FROZEN_SAMPLES_SHA256,
}
EXPECTED_SCENARIO_COUNTS = {
    "boundary": 10,
    "browsing": 80,
    "buying": 80,
    "intent_override": 30,
}
SOURCE_PATHS = {
    "builder": "scripts/build_p8_selection_corpus.py",
    "p8_negative": "starter/p8_negative.py",
    "p8_lab": "starter/p8_lab.py",
    "p8_worker": "scripts/p8_worker.py",
    "evaluate_p8": "scripts/evaluate_p8.py",
    "agent": "starter/agent.py",
    "coverage": "starter/coverage.py",
    "attributes": "starter/attributes.py",
    "slot_ledger": "starter/slot_ledger.py",
    "response_contract": "starter/response_contract.py",
    "evaluator": "evaluator/local_evaluator.py",
    "verify_official_assets": "scripts/verify_official_assets.py",
    "clarification": "starter/clarification.py",
    "reranker": "starter/reranker.py",
    "lock_builder": "scripts/build_p8_prereg_lock.py",
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
    corpus_metadata: Path
    corpora: Mapping[str, Path]
    output: Path


def default_paths(project_root: Path = PROJECT_ROOT) -> LockPaths:
    root = project_root.resolve()
    return LockPaths(
        spec=root / "configs" / "p8_explicit_negative_matrix.json",
        catalog=root / "data" / "catalog.jsonl",
        released_public=root / "data" / "public_set.jsonl",
        priors={
            "p1": root / "experiments" / "p1_derived_product_disjoint.jsonl",
            "p5": root / "experiments" / "p5_selection_product_disjoint.jsonl",
            "p6": root / "experiments" / "p6_selection_product_disjoint.jsonl",
            "p7": root / "experiments" / "p7_selection_product_disjoint.jsonl",
        },
        corpus_metadata=root / "experiments" / "p8_explicit_negative_corpora.metadata.json",
        corpora={
            "selection": root / "experiments" / "p8_selection_product_disjoint.jsonl",
            "confirmation": root / "experiments" / "p8_confirmation_product_disjoint.jsonl",
        },
        output=root / "configs" / "p8_prereg_lock.json",
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
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


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
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _raw_git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _git(project_root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={project_root.resolve().as_posix()}", *arguments],
        cwd=project_root,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if completed.returncode:
        raise PreregLockError(f"Git command failed: {' '.join(arguments)}")
    return completed.stdout if binary else str(completed.stdout).strip()


def capture_pushed_clean_revision(project_root: Path) -> dict[str, str]:
    branch = str(_git(project_root, "branch", "--show-current"))
    head = str(_git(project_root, "rev-parse", "HEAD"))
    status = str(_git(project_root, "status", "--porcelain=v1", "--untracked-files=all"))
    if not branch:
        raise PreregLockError("P8 lock requires a named Git branch")
    if not re.fullmatch(r"[a-f0-9]{40}", head):
        raise PreregLockError("P8 lock requires a valid HEAD commit")
    if status:
        raise PreregLockError("P8 lock requires a completely clean worktree")
    origin_head = str(_git(project_root, "rev-parse", f"origin/{branch}"))
    if origin_head != head:
        raise PreregLockError("P8 lock requires HEAD to equal origin/<branch>")
    return {"git_commit": head, "git_branch": branch}


def _required_names_from_runner(runner_path: Path) -> set[str]:
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "REQUIRED_SOURCE_NAMES" for target in node.targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, set) and all(isinstance(item, str) for item in value):
                    return value
    raise PreregLockError("evaluate_p8.REQUIRED_SOURCE_NAMES is not a literal string set")


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
            raise PreregLockError("evaluate_p8.REQUIRED_SOURCE_PATHS must be a literal string map")
    names = _required_names_from_runner(runner_path)
    missing = names - set(SOURCE_PATHS)
    if missing:
        raise PreregLockError(
            "source path catalog is missing evaluate_p8 requirements: "
            + ", ".join(sorted(missing))
        )
    return {name: SOURCE_PATHS[name] for name in sorted(names)}


def _tracked_head_identity(
    path: Path,
    project_root: Path,
    *,
    include_git_blob: bool = False,
) -> dict[str, Any]:
    entry = _file_entry(path, project_root)
    relative = entry["path"]
    _git(project_root, "ls-files", "--error-unmatch", relative)
    working_blob = str(
        _git(project_root, "hash-object", f"--path={relative}", "--", relative)
    )
    head_blob = str(_git(project_root, "rev-parse", f"HEAD:{relative}"))
    if (
        re.fullmatch(r"[a-f0-9]{40}", working_blob) is None
        or working_blob != head_blob
    ):
        raise PreregLockError(f"working content differs from HEAD Git blob: {relative}")
    return {**entry, **({"git_blob_sha1": head_blob} if include_git_blob else {})}


def capture_source_lock(project_root: Path, revision: Mapping[str, str]) -> dict[str, Any]:
    required = _required_paths_from_runner(project_root / SOURCE_PATHS["evaluate_p8"])
    selected = {**required, "lock_builder": SOURCE_PATHS["lock_builder"]}
    files = {
        name: _tracked_head_identity(
            project_root / relative, project_root, include_git_blob=True
        )
        for name, relative in selected.items()
    }
    return {
        "git_commit": revision["git_commit"],
        "git_branch": revision["git_branch"],
        "files": files,
    }


def _inspect_conversation_rows(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    targets = [
        str(row.get("ground_truth", {}).get("parent_asin") or "").strip()
        for row in rows
    ]
    if not rows or not all(targets) or len(set(targets)) != len(targets):
        raise PreregLockError(f"conversation rows have empty or duplicate target identifiers: {path}")
    scenarios = dict(sorted(Counter(str(row.get("scenario_type") or "") for row in rows).items()))
    return {
        "rows": rows,
        "row_count": len(rows),
        "canonical_samples_sha256": _canonical_rows_sha256(rows),
        "scenario_counts": scenarios,
        "target_ids": set(targets),
    }


def _zero_overlap_values(value: Any, key: str = "") -> list[Any]:
    if isinstance(value, Mapping):
        found: list[Any] = []
        for nested_key, nested in value.items():
            found.extend(_zero_overlap_values(nested, str(nested_key)))
        return found
    return [value] if "overlap" in key.lower() else []


def _validate_metadata(
    metadata: Mapping[str, Any],
    *,
    catalog_sha256: str,
    catalog_rows: int,
    inputs: Mapping[str, Mapping[str, Any]],
    corpora: Mapping[str, Mapping[str, Any]],
    public_blob: str,
) -> None:
    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise PreregLockError("P8 corpus metadata schema is invalid")
    catalog = metadata.get("catalog_source")
    metadata_inputs = metadata.get("input_sources")
    metadata_corpora = metadata.get("corpora")
    exclusions = metadata.get("exclusions")
    outputs = metadata.get("outputs")
    if not all(
        isinstance(value, dict)
        for value in (catalog, metadata_inputs, metadata_corpora, exclusions, outputs)
    ):
        raise PreregLockError("P8 corpus metadata sections are incomplete")
    assert isinstance(catalog, dict)
    assert isinstance(metadata_inputs, dict)
    assert isinstance(metadata_corpora, dict)
    assert isinstance(outputs, dict)
    if (
        catalog.get("sha256") != catalog_sha256
        or catalog.get("loaded_product_count") != catalog_rows
        or catalog.get("frozen_sha256_verified") is not True
        or catalog.get("expected_count_verified") is not True
    ):
        raise PreregLockError("P8 metadata catalog identity is invalid")
    names = {
        "released_public": "released_public",
        "prior_p1_derived": "p1",
        "prior_p5_derived": "p5",
        "prior_p6_derived": "p6",
        "prior_p7_derived": "p7",
    }
    if set(metadata_inputs) != set(names):
        raise PreregLockError("P8 metadata input registry is incomplete")
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
            raise PreregLockError(f"P8 metadata identity is invalid for {metadata_name}")
    public_entry = metadata_inputs["released_public"]
    if (
        public_entry.get("git_blob_sha1_lf") != public_blob
        or public_entry.get("frozen_git_blob_verified") is not True
    ):
        raise PreregLockError("P8 metadata public Git blob is invalid")
    if set(metadata_corpora) != {"selection", "confirmation"}:
        raise PreregLockError("P8 metadata corpus registry is incomplete")
    for split, observed in corpora.items():
        entry = metadata_corpora[split]
        if (
            not isinstance(entry, dict)
            or entry.get("sample_count") != observed["row_count"]
            or entry.get("unique_target_count") != observed["row_count"]
            or entry.get("samples_sha256") != observed["canonical_samples_sha256"]
            or entry.get("scenario_counts") != observed["scenario_counts"]
        ):
            raise PreregLockError(f"P8 metadata identity is invalid for {split}")
        output = outputs.get(split)
        if (
            not isinstance(output, dict)
            or output.get("expected_frozen_samples_sha256")
            != observed["canonical_samples_sha256"]
            or output.get("samples_file_sha256")
            != observed["canonical_samples_sha256"]
            or output.get("frozen_samples_sha256_verified") is not True
        ):
            raise PreregLockError(f"P8 metadata frozen output proof is invalid for {split}")
    overlaps = _zero_overlap_values(exclusions)
    if not overlaps or any(value != 0 for value in overlaps):
        raise PreregLockError("P8 metadata does not prove zero target overlap")


def _assert_target_free(value: Any, target_ids: set[str]) -> None:
    payload = _canonical_bytes(value).decode("utf-8")
    leaked = next((identifier for identifier in target_ids if identifier in payload), None)
    if leaked is not None:
        raise PreregLockError("P8 preregistration lock contains a target identifier")
    prohibited_keys = {"ground_truth", "target", "target_id", "target_asin", "parent_asin", "sample_id"}

    def visit(nested: Any) -> None:
        if isinstance(nested, Mapping):
            for key, child in nested.items():
                if str(key).lower() in prohibited_keys:
                    raise PreregLockError(f"P8 preregistration lock contains prohibited key: {key}")
                visit(child)
        elif isinstance(nested, (list, tuple)):
            for child in nested:
                visit(child)

    visit(value)


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
        raise FileExistsError(f"P8 preregistration lock already exists: {paths.output}")
    if require_defaults and paths != default_paths(root):
        raise PreregLockError("formal P8 lock generation requires default paths")
    if set(paths.priors) != {"p1", "p5", "p6", "p7"} or set(paths.corpora) != {
        "selection",
        "confirmation",
    }:
        raise PreregLockError("P8 lock input registry is incomplete")

    if enforce_git:
        revision = capture_pushed_clean_revision(root)
        source = capture_source_lock(root, revision)
        spec_entry = _tracked_head_identity(paths.spec, root)
    else:
        revision = {"git_commit": "0" * 40, "git_branch": "fixture"}
        selected_sources = _required_paths_from_runner(root / SOURCE_PATHS["evaluate_p8"])
        selected_sources["lock_builder"] = SOURCE_PATHS["lock_builder"]
        source = {
            "git_commit": revision["git_commit"],
            "git_branch": revision["git_branch"],
            "files": {
                name: {
                    **_file_entry(root / relative, root),
                    "git_blob_sha1": _raw_git_blob_sha1(root / relative),
                }
                for name, relative in selected_sources.items()
            },
        }
        spec_entry = _file_entry(paths.spec, root)

    spec = _load_json_object(paths.spec)
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise PreregLockError("P8 matrix spec schema is invalid")
    catalog_entry = _file_entry(paths.catalog, root)
    with paths.catalog.open("rb") as handle:
        catalog_rows = sum(1 for line in handle if line.strip())
    if (
        catalog_entry["sha256"] != expectations.catalog_sha256
        or catalog_rows != expectations.catalog_rows
    ):
        raise PreregLockError("official catalog identity differs from frozen expectations")

    public = _inspect_conversation_rows(paths.released_public)
    public_blob = _git_blob_sha1(paths.released_public)
    if (
        public["row_count"] != expectations.public_rows
        or public_blob != expectations.public_git_blob_sha1
        or public["canonical_samples_sha256"] != canonical_expected["released_public"]
    ):
        raise PreregLockError("official public corpus identity differs from frozen expectations")
    prior_observations = {
        name: _inspect_conversation_rows(path) for name, path in paths.priors.items()
    }
    for name, observed in prior_observations.items():
        if (
            observed["row_count"] != expectations.prior_rows
            or observed["canonical_samples_sha256"] != canonical_expected[name]
        ):
            raise PreregLockError(f"prior {name} identity differs from frozen expectations")
    corpus_observations = {
        split: _inspect_conversation_rows(path) for split, path in paths.corpora.items()
    }
    for split, observed in corpus_observations.items():
        if (
            observed["row_count"] != expectations.split_rows
            or observed["canonical_samples_sha256"] != canonical_expected[split]
            or observed["scenario_counts"] != scenario_expected
            or _sha256_file(paths.corpora[split]) != observed["canonical_samples_sha256"]
        ):
            raise PreregLockError(f"P8 {split} identity differs from frozen expectations")

    inputs = {"released_public": public, **prior_observations}
    metadata = _load_json_object(paths.corpus_metadata)
    _validate_metadata(
        metadata,
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
            raise PreregLockError("P8 locked conversation corpora are not product-disjoint")
        all_target_ids.update(observed["target_ids"])

    lock = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
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
        raise FileExistsError(f"P8 preregistration lock already exists: {path}")
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
    parser.add_argument("--prior-p1", type=Path, default=defaults.priors["p1"])
    parser.add_argument("--prior-p5", type=Path, default=defaults.priors["p5"])
    parser.add_argument("--prior-p6", type=Path, default=defaults.priors["p6"])
    parser.add_argument("--prior-p7", type=Path, default=defaults.priors["p7"])
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
            "p1": args.prior_p1.resolve(),
            "p5": args.prior_p5.resolve(),
            "p6": args.prior_p6.resolve(),
            "p7": args.prior_p7.resolve(),
        },
        corpus_metadata=args.metadata.resolve(),
        corpora={
            "selection": args.selection.resolve(),
            "confirmation": args.confirmation.resolve(),
        },
        output=args.output.resolve(),
    )
    lock = build_prereg_lock(paths=paths)
    atomic_create(paths.output, lock)
    print(
        f"[p8-lock] commit={lock['source']['git_commit']} wrote={paths.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
