"""Shared pre/post provenance snapshots for long-running promotion gates."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "p4.gate-provenance.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(project_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={project_root.resolve().as_posix()}",
            *arguments,
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )


def git_snapshot(project_root: Path) -> dict[str, Any]:
    branch = _git(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = _git(project_root, "rev-parse", "HEAD")
    status = _git(project_root, "status", "--porcelain")
    if branch.returncode or commit.returncode or status.returncode:
        raise RuntimeError("unable to capture Git provenance")
    return {
        "branch": branch.stdout.strip(),
        "commit": commit.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
    }


def hash_snapshot(paths: dict[str, Path]) -> dict[str, str]:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "provenance inputs are missing: " + ", ".join(sorted(missing))
        )
    return {name: sha256(path) for name, path in paths.items()}


def capture_gate_snapshot(
    project_root: Path,
    *,
    source_paths: dict[str, Path],
    input_paths: dict[str, Path],
    selection_commit: str,
    frozen_architecture_path: Path,
    local_selection_artifact: Path | None = None,
    expected_selection_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    relative_architecture = frozen_architecture_path.resolve().relative_to(
        project_root.resolve()
    ).as_posix()
    ancestor = _git(
        project_root,
        "merge-base",
        "--is-ancestor",
        selection_commit,
        "HEAD",
    )
    unchanged = _git(
        project_root,
        "diff",
        "--quiet",
        selection_commit,
        "--",
        relative_architecture,
    )
    local_exists = bool(
        local_selection_artifact is not None and local_selection_artifact.is_file()
    )
    local_hash = sha256(local_selection_artifact) if local_exists else None
    return {
        "schema_version": SCHEMA_VERSION,
        "git": git_snapshot(project_root),
        "source_sha256": hash_snapshot(source_paths),
        "input_sha256": hash_snapshot(input_paths),
        "selection_evidence": {
            "selection_commit": selection_commit,
            "selection_commit_is_ancestor": ancestor.returncode == 0,
            "frozen_architecture_path": relative_architecture,
            "frozen_architecture_unchanged_since_selection": unchanged.returncode == 0,
            "local_full_selection_artifact_present": local_exists,
            "local_full_selection_artifact_sha256": local_hash,
            "expected_full_selection_artifact_sha256": (
                expected_selection_artifact_sha256
            ),
            "local_full_selection_artifact_hash_matches": (
                local_hash == expected_selection_artifact_sha256
                if local_hash is not None and expected_selection_artifact_sha256
                else None
            ),
        },
    }


def validate_clean_frozen_snapshot(snapshot: dict[str, Any]) -> None:
    evidence = snapshot["selection_evidence"]
    failures: list[str] = []
    if snapshot["git"]["dirty"]:
        failures.append("Git worktree is dirty")
    if not evidence["selection_commit_is_ancestor"]:
        failures.append("selection commit is not an ancestor of HEAD")
    if not evidence["frozen_architecture_unchanged_since_selection"]:
        failures.append("frozen architecture source changed after selection")
    if evidence["local_full_selection_artifact_present"] and not evidence[
        "local_full_selection_artifact_hash_matches"
    ]:
        failures.append("local full selection artifact hash does not match")
    if failures:
        raise RuntimeError("invalid frozen-winner provenance: " + "; ".join(failures))


def assert_gate_snapshot_stable(
    preflight: dict[str, Any],
    postflight: dict[str, Any],
) -> None:
    keys = ("git", "source_sha256", "input_sha256", "selection_evidence")
    changed = [key for key in keys if preflight[key] != postflight[key]]
    if changed:
        raise RuntimeError(
            "gate source/input/Git state changed during evaluation: "
            + ", ".join(changed)
        )
