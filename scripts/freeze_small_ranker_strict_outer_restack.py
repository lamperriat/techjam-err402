"""Freeze two complete v2.8 Stage-1 passes without opening outcome labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_small_ranker_strict_outer_restack as build  # noqa: E402
from scripts import probe_small_ranker_strict_outer_restack as probe  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-strict-outer-restack-stage1-freeze.v1"
HELD_FIELDS = (
    "current_chosen",
    "current_activation",
    "final_chosen",
    "final_activation",
)


class StrictRestackFreezeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return probe._sha256(path)


def _array_sha256(value: np.ndarray) -> str:
    return probe._array_sha256(np.asarray(value))


def _canonical_sha256(value: object) -> str:
    return probe._canonical_sha256(value)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StrictRestackFreezeError("expected a JSON object")
    return value


def _source_snapshot() -> dict[str, str]:
    return {
        "freezer_sha256": _sha256(Path(__file__).resolve()),
        "builder_sha256": _sha256(
            ROOT / "scripts/build_small_ranker_strict_outer_restack.py"
        ),
        "selector_subset_sha256": _sha256(
            ROOT / "scripts/small_ranker_portfolio_selector_py39.py"
        ),
        "preregistration_sha256": _sha256(build.PREREGISTRATION),
        "stage1_amendment_sha256": _sha256(build.STAGE1_AMENDMENT),
        "stage0_manifest_sha256": _sha256(build.STAGE0_MANIFEST),
    }


def _outer_result_path(stage_root: Path, pass_name: str, outer_fold: int) -> Path:
    return stage_root / pass_name / ("outer_%d" % outer_fold) / "outer_complete.json"


def _expected_model_topology(outer_fold: int) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    domains = [
        "A_%d%d" % (outer_fold, inner_fold)
        for inner_fold in range(base.OUTER_FOLDS)
    ] + ["T_%d" % outer_fold]
    generic = {
        (model_id, domain)
        for domain in domains
        for model_id in build.GENERIC_MODEL_IDS
    }
    focused = {("focused_ndcg_d3", domain) for domain in domains}
    return generic, focused


def _rebuild_outer_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    sources = result["sources"]
    source_snapshot = {
        key: sources[key]
        for key in (
            "builder_sha256",
            "preregistration_sha256",
            "stage0_amendment_sha256",
            "stage1_amendment_sha256",
            "stage0_manifest_sha256",
            "stage0_result_sha256",
            "helper_sha256",
            "config_sha256",
        )
    }
    model_identity = sorted(
        [
            build._model_identity(row)
            for row in (
                *result["models"]["generic"],
                *result["models"]["focused"],
            )
        ],
        key=lambda row: (str(row["domain"]), str(row["model_id"])),
    )
    current = result["current"]
    focused_cache = result["focused_cache"]
    portfolio_scores = result["portfolio_score_files"]
    runtime = result["runtime"]
    portfolio_labels = result["portfolio_training_labels"]
    selector = result["selector"]
    coverage = result["coverage"]
    return {
        "outer_fold": result["outer_fold"],
        "source_snapshot": source_snapshot,
        "dependencies": result["dependencies"],
        "domains": result["domains"],
        "selected_training_rows": {
            key: value
            for key, value in result["selected_training_rows"].items()
            if key != "build_seconds"
        },
        "models": model_identity,
        "oof_score_arrays": build._file_array_hashes(
            portfolio_scores["oof_generic"]
        ),
        "focused_oof_array": portfolio_scores["oof_focused"]["array_sha256"],
        "oof_coverage_sha256": coverage["oof_sha256"],
        "focused_coverage_sha256": coverage["focused_sha256"],
        "current": {
            "oof_files": build._file_array_hashes(current["oof_files"]),
            "oof_aux_files": build._file_array_hashes(current["oof_aux_files"]),
            "inner_gate": current["inner_gate"],
            "full_files": build._file_array_hashes(current["full_files"]),
            "full_gate": current["full_gate"],
        },
        "focused_cache": {
            "counts": {
                key: value
                for key, value in focused_cache.items()
                if key != "files"
            },
            "files": build._file_array_hashes(focused_cache["files"]),
            "partitions": result["focused_partitions"],
        },
        "rrf": {
            "oof": result["rrf"]["oof"]["array_sha256"],
            "full": result["rrf"]["full"]["array_sha256"],
        },
        "runtime": {
            key: {
                **build._record_without_files(value),
                "files": build._file_array_hashes(value["files"]),
            }
            for key, value in runtime.items()
        },
        "portfolio_training_labels": {
            **build._record_without_files(portfolio_labels),
            "files": build._file_array_hashes(portfolio_labels["files"]),
        },
        "selector": {
            **build._record_without_files(selector),
            "files": build._file_array_hashes(selector["files"]),
        },
        "held": build._file_array_hashes(result["held"]),
        "stage0_prefix_parity": {
            key: value
            for key, value in result["stage0_prefix_parity"].items()
            if key != "stage0_pass"
        },
    }


def _load_outer(stage_root: Path, pass_name: str, outer_fold: int) -> dict[str, Any]:
    path = _outer_result_path(stage_root, pass_name, outer_fold)
    if not path.is_file() or path.is_symlink():
        raise StrictRestackFreezeError(
            "outer cache unavailable: %s/%d" % (pass_name, outer_fold)
        )
    result = _load_json(path)
    identity = result.get("identity")
    models = result.get("models")
    held = result.get("held")
    domains = result.get("domains")
    if not all(
        isinstance(value, dict)
        for value in (identity, models, held, domains)
    ):
        raise StrictRestackFreezeError(
            "outer cache structure failed: %s/%d" % (pass_name, outer_fold)
        )
    generic_records = models.get("generic")
    focused_records = models.get("focused")
    if not isinstance(generic_records, list) or not isinstance(focused_records, list):
        raise StrictRestackFreezeError(
            "outer model records failed: %s/%d" % (pass_name, outer_fold)
        )
    try:
        model_identity = sorted(
            [
                build._model_identity(row)
                for row in (*generic_records, *focused_records)
            ],
            key=lambda row: (str(row["domain"]), str(row["model_id"])),
        )
        generic_topology = {
            (str(row["model_id"]), str(row["domain"]))
            for row in generic_records
        }
        focused_topology = {
            (str(row["model_id"]), str(row["domain"]))
            for row in focused_records
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise StrictRestackFreezeError(
            "outer model identity failed: %s/%d" % (pass_name, outer_fold)
        ) from exc
    expected_generic, expected_focused = _expected_model_topology(outer_fold)
    required_held = {"session_ordinal", *HELD_FIELDS}
    identity_held = identity.get("held")
    held_bound = (
        isinstance(identity_held, dict)
        and set(held) == required_held
        and set(identity_held) == required_held
        and all(
            isinstance(held[name], dict)
            and held[name].get("array_sha256") == identity_held[name]
            for name in required_held
        )
        and len({str(held[name].get("path", "")) for name in required_held})
        == len(required_held)
    )
    held_outcome_rows = result.get("privacy", {}).get(
        "held_outcome_rows_retained_or_supplied_to_fit_selection_or_metric"
    )
    try:
        rebuilt_identity = _rebuild_outer_identity(result)
    except (KeyError, TypeError, ValueError) as exc:
        raise StrictRestackFreezeError(
            "outer identity bridge failed: %s/%d" % (pass_name, outer_fold)
        ) from exc
    if (
        result.get("schema_version") != build.SCHEMA_VERSION
        or result.get("status") != "OUTER_CACHE_COMPLETE"
        or result.get("pass_name") != pass_name
        or type(result.get("outer_fold")) is not int
        or result.get("outer_fold") != outer_fold
        or type(held_outcome_rows) is not int
        or held_outcome_rows != 0
        or result.get("privacy", {}).get("held_state_or_outcome_metric_computed")
        is not False
        or type(models.get("generic_count")) is not int
        or models.get("generic_count") != len(generic_records)
        or type(models.get("focused_count")) is not int
        or models.get("focused_count") != len(focused_records)
        or type(models.get("total_count")) is not int
        or models.get("total_count") != len(generic_records) + len(focused_records)
        or len(generic_records) != 24
        or len(focused_records) != 6
        or generic_topology != expected_generic
        or focused_topology != expected_focused
        or _canonical_sha256(model_identity)
        != _canonical_sha256(identity.get("models"))
        or _canonical_sha256(domains)
        != _canonical_sha256(identity.get("domains"))
        or not held_bound
        or _canonical_sha256(rebuilt_identity)
        != _canonical_sha256(identity)
        or result.get("identity_sha256") != _canonical_sha256(identity)
    ):
        raise StrictRestackFreezeError(
            "outer cache contract failed: %s/%d" % (pass_name, outer_fold)
        )
    stage0_files = result.get("stage0_parity_files")
    if outer_fold == 0:
        if not isinstance(stage0_files, dict) or set(stage0_files) != {
            "rrf3_score",
            "rrf3_choice",
        }:
            raise StrictRestackFreezeError("Stage-0 parity files are incomplete")
        try:
            rebuilt_stage0 = build._stage0_prefix_parity(
                pass_name,
                generic_records,
                focused_records,
                result["current"]["oof_files"],
                result["current"]["inner_gate"],
                result["focused_partitions"],
                stage0_files["rrf3_score"],
                stage0_files["rrf3_choice"],
            )
        except (KeyError, TypeError, ValueError, build.StrictRestackBuildError) as exc:
            raise StrictRestackFreezeError("Stage-0 parity bridge failed") from exc
        if _canonical_sha256(rebuilt_stage0) != _canonical_sha256(
            result["stage0_prefix_parity"]
        ):
            raise StrictRestackFreezeError("Stage-0 parity record drifted")
    elif stage0_files != {}:
        raise StrictRestackFreezeError("unexpected Stage-0 parity files")
    recorded_source = identity.get("source_snapshot")
    if not isinstance(recorded_source, dict):
        raise StrictRestackFreezeError("outer cache lacks a source snapshot")
    return result


def _validate_outer_pair(
    first: Mapping[str, Any], repeat: Mapping[str, Any]
) -> dict[str, Any]:
    first_hash = first.get("identity_sha256")
    repeat_hash = repeat.get("identity_sha256")
    if (
        not isinstance(first_hash, str)
        or not isinstance(repeat_hash, str)
        or first_hash != repeat_hash
        or first.get("identity") != repeat.get("identity")
    ):
        raise StrictRestackFreezeError(
            "outer exact repeat differs: %s"
            % build._first_difference(first.get("identity"), repeat.get("identity"))
        )
    identity = first["identity"]
    return {
        "outer_fold": int(first["outer_fold"]),
        "equal": True,
        "identity_sha256": _canonical_sha256(identity),
        "models_compared": len(identity["models"]),
        "stage0_prefix_parity": identity["stage0_prefix_parity"],
    }


def _record_path(
    stage_root: Path,
    shard_root: Path,
    record: Mapping[str, Any],
) -> Path:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute():
        raise StrictRestackFreezeError("cache path must be repo-relative")
    unresolved = ROOT / relative
    path = unresolved.resolve()
    if (
        ROOT not in path.parents
        or stage_root not in path.parents
        or shard_root not in path.parents
        or not path.is_file()
        or unresolved.is_symlink()
        or _sha256(path) != record.get("sha256")
        or path.stat().st_size != int(record.get("bytes", -1))
    ):
        raise StrictRestackFreezeError("cache file contract failed")
    return path


def _iter_file_records(value: object) -> Sequence[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            if {"path", "sha256", "bytes"}.issubset(node):
                records.append(node)
                return
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return records


def _npy_header(path: Path) -> tuple[tuple[int, ...], str]:
    try:
        with path.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            shape, _fortran_order, dtype = np.lib.format._read_array_header(
                handle, version
            )
    except (OSError, ValueError) as exc:
        raise StrictRestackFreezeError("cache NPY header is invalid") from exc
    return tuple(shape), str(dtype)


def _validate_all_shard_files(
    stage_root: Path,
    pass_name: str,
    outer_fold: int,
    result: Mapping[str, Any],
) -> int:
    shard_root = _outer_result_path(stage_root, pass_name, outer_fold).parent.resolve()
    recorded_paths: set[Path] = set()
    for record in _iter_file_records(result):
        path = _record_path(stage_root, shard_root, record)
        if path in recorded_paths:
            raise StrictRestackFreezeError("cache file record is reused")
        recorded_paths.add(path)
        has_array_schema = all(
            key in record for key in ("shape", "dtype", "array_sha256")
        )
        if has_array_schema:
            if (
                path.suffix.lower() != ".npy"
                or _npy_header(path)
                != (tuple(record["shape"]), str(record["dtype"]))
                or base._identity_shape_scan(path) != 0
            ):
                raise StrictRestackFreezeError("cache NPY physical contract failed")
        elif path.suffix.lower() == ".npy":
            raise StrictRestackFreezeError("cache NPY record lacks array schema")
    actual_paths = {
        path.resolve()
        for path in shard_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    complete_path = _outer_result_path(stage_root, pass_name, outer_fold).resolve()
    if actual_paths != recorded_paths | {complete_path}:
        raise StrictRestackFreezeError("outer shard contains unbound or missing files")
    if any(path.is_symlink() for path in shard_root.rglob("*")):
        raise StrictRestackFreezeError("outer shard contains a symlink")
    return len(recorded_paths)


def _load_array_record(
    stage_root: Path,
    shard_root: Path,
    record: Mapping[str, Any],
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> np.ndarray:
    path = _record_path(stage_root, shard_root, record)
    if base._identity_shape_scan(path) != 0:
        raise StrictRestackFreezeError("cache array identity scan failed")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise StrictRestackFreezeError("cache array could not be loaded") from exc
    if (
        tuple(value.shape) != expected_shape
        or str(value.dtype) != expected_dtype
        or list(value.shape) != list(record.get("shape", []))
        or str(value.dtype) != str(record.get("dtype"))
        or _array_sha256(value) != record.get("array_sha256")
    ):
        raise StrictRestackFreezeError("cache array logical contract failed")
    return np.asarray(value)


def _validate_model_artifacts(
    stage_root: Path,
    pass_name: str,
    outer_fold: int,
    result: Mapping[str, Any],
) -> None:
    shard_root = _outer_result_path(stage_root, pass_name, outer_fold).parent.resolve()
    if not shard_root.is_dir() or shard_root.is_symlink():
        raise StrictRestackFreezeError("outer shard directory is invalid")
    domains = result["domains"]
    seen_paths: set[Path] = set()
    for family in ("generic", "focused"):
        for record in result["models"][family]:
            model_id = str(record["model_id"])
            domain = str(record["domain"])
            expected_stem = model_id + "__" + domain
            domain_record = domains.get(domain)
            if not isinstance(domain_record, dict):
                raise StrictRestackFreezeError("model domain record is unavailable")
            if family == "generic":
                train_bound = (
                    record.get("train_sessions") == domain_record.get("sessions")
                    and record.get("train_session_mask_sha256")
                    == domain_record.get("mask_sha256")
                )
            else:
                train_bound = (
                    record.get("parent_domain_mask_sha256")
                    == domain_record.get("mask_sha256")
                )
            if domain.startswith("A_%d" % outer_fold):
                score_domain = domains.get("V" + domain[1:])
                expected_score_sessions = (
                    score_domain.get("sessions") if isinstance(score_domain, dict) else None
                )
                expected_score_sha256 = (
                    score_domain.get("session_sha256")
                    if isinstance(score_domain, dict)
                    else None
                )
            elif domain == "T_%d" % outer_fold:
                expected_score_sessions = base.SESSION_COUNT
                expected_score_sha256 = _array_sha256(build.ALL_SESSIONS)
            else:
                raise StrictRestackFreezeError("model domain is outside the outer topology")
            if (
                not train_bound
                or record.get("score_sessions") != expected_score_sessions
                or record.get("score_session_sha256") != expected_score_sha256
            ):
                raise StrictRestackFreezeError("model domain lineage is invalid")
            model_path = _record_path(stage_root, shard_root, record["model"])
            if (
                model_path.name != expected_stem + ".json"
                or record["model"].get("asin_shape_matches") != 0
                or base._identity_shape_scan(model_path) != 0
            ):
                raise StrictRestackFreezeError("model JSON contract failed")
            score = _load_array_record(
                stage_root,
                shard_root,
                record["score"],
                (int(expected_score_sessions), base.TURN_COUNT, base.CANDIDATE_COUNT),
                "float32",
            )
            choice = _load_array_record(
                stage_root,
                shard_root,
                record["choice"],
                (int(expected_score_sessions), base.TURN_COUNT),
                "uint8",
            )
            if (
                Path(str(record["score"]["path"])).name != expected_stem + ".npy"
                or Path(str(record["choice"]["path"])).name != expected_stem + ".npy"
                or not np.isfinite(score).all()
                or np.any(choice >= base.CANDIDATE_COUNT)
            ):
                raise StrictRestackFreezeError("model score/choice contract failed")
            artifact_paths = {
                model_path,
                (ROOT / str(record["score"]["path"])).resolve(),
                (ROOT / str(record["choice"]["path"])).resolve(),
            }
            if len(artifact_paths) != 3 or seen_paths.intersection(artifact_paths):
                raise StrictRestackFreezeError("model artifact path is reused")
            seen_paths.update(artifact_paths)


def _validate_held_array(name: str, value: np.ndarray) -> None:
    expected_shape, expected_dtype = _held_schema(name)
    if tuple(value.shape) != expected_shape or str(value.dtype) != expected_dtype:
        raise StrictRestackFreezeError("held array schema failed: %s" % name)
    if name.endswith("chosen") and np.any(value >= base.CANDIDATE_COUNT):
        raise StrictRestackFreezeError("held choice is out of range: %s" % name)


def _held_schema(name: str) -> tuple[tuple[int, ...], str]:
    shape = (400,) if name == "session_ordinal" else (400, base.TURN_COUNT)
    dtype = {
        "session_ordinal": "int16",
        "current_chosen": "uint8",
        "current_activation": "bool",
        "final_chosen": "uint8",
        "final_activation": "bool",
    }[name]
    return shape, dtype


def _write_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    return probe._write_npy_exclusive(path, np.asarray(value))


def run(stage_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    stage_root = stage_root.resolve()
    experiments_root = (ROOT / "experiments").resolve()
    if (
        not stage_root.is_dir()
        or stage_root.is_symlink()
        or experiments_root not in stage_root.parents
    ):
        raise StrictRestackFreezeError(
            "stage root must be an existing directory below experiments"
        )
    freeze_dir = stage_root / "frozen"
    result_path = stage_root / "stage1_cache_result.json"
    if freeze_dir.exists() or freeze_dir.is_symlink() or result_path.exists() or result_path.is_symlink():
        raise StrictRestackFreezeError("freeze outputs must be new")
    _preregistration, _amendment, stage0_manifest = build._validate_protocol()
    source_snapshot = _source_snapshot()
    pairs = []
    outer_results: dict[str, list[dict[str, Any]]] = {"first": [], "repeat": []}
    worker_result_files: dict[str, list[dict[str, Any]]] = {
        "first": [],
        "repeat": [],
    }
    physical_file_counts: dict[str, list[int]] = {"first": [], "repeat": []}
    recorded_snapshots = []
    for outer_fold in range(base.OUTER_FOLDS):
        first = _load_outer(stage_root, "first", outer_fold)
        repeat = _load_outer(stage_root, "repeat", outer_fold)
        pair = _validate_outer_pair(first, repeat)
        if outer_fold == 0:
            parity = pair["stage0_prefix_parity"]
            stage0_identity_sha256 = stage0_manifest.get("exact_repeat", {}).get(
                "identity_sha256"
            )
            if (
                parity.get("equal") is not True
                or parity.get("models_compared") != 9
                or parity.get("expected_identity_sha256")
                != stage0_identity_sha256
                or parity.get("actual_identity_sha256")
                != stage0_identity_sha256
            ):
                raise StrictRestackFreezeError("outer-0 Stage-0 prefix parity is absent")
        elif pair["stage0_prefix_parity"].get("applicable") is not False:
            raise StrictRestackFreezeError("unexpected Stage-0 parity scope")
        pairs.append(pair)
        for pass_name, result in (("first", first), ("repeat", repeat)):
            outer_results[pass_name].append(result)
            recorded_snapshots.append(result["identity"]["source_snapshot"])
            _validate_model_artifacts(stage_root, pass_name, outer_fold, result)
            physical_file_counts[pass_name].append(
                _validate_all_shard_files(
                    stage_root, pass_name, outer_fold, result
                )
            )
            result_file = _outer_result_path(stage_root, pass_name, outer_fold)
            worker_result_files[pass_name].append(
                {
                    "path": result_file.relative_to(ROOT).as_posix(),
                    "sha256": _sha256(result_file),
                    "bytes": result_file.stat().st_size,
                }
            )
    if any(snapshot != recorded_snapshots[0] for snapshot in recorded_snapshots[1:]):
        raise StrictRestackFreezeError("outer workers used different source snapshots")
    if recorded_snapshots[0].get("builder_sha256") != source_snapshot["builder_sha256"]:
        raise StrictRestackFreezeError("current builder differs from outer worker source")
    for pass_name in ("first", "repeat"):
        generic_count = sum(
            int(row["models"]["generic_count"])
            for row in outer_results[pass_name]
        )
        focused_count = sum(
            int(row["models"]["focused_count"])
            for row in outer_results[pass_name]
        )
        if generic_count != 120 or focused_count != 30:
            raise StrictRestackFreezeError("pass model topology is incomplete")
    stitched = {
        "current_chosen": np.empty(
            (base.SESSION_COUNT, base.TURN_COUNT), dtype=np.uint8
        ),
        "current_activation": np.empty(
            (base.SESSION_COUNT, base.TURN_COUNT), dtype=bool
        ),
        "final_chosen": np.empty(
            (base.SESSION_COUNT, base.TURN_COUNT), dtype=np.uint8
        ),
        "final_activation": np.empty(
            (base.SESSION_COUNT, base.TURN_COUNT), dtype=bool
        ),
    }
    coverage = np.zeros(base.SESSION_COUNT, dtype=np.uint8)
    for outer_fold, first in enumerate(outer_results["first"]):
        repeat = outer_results["repeat"][outer_fold]
        first_shard = _outer_result_path(stage_root, "first", outer_fold).parent.resolve()
        repeat_shard = _outer_result_path(stage_root, "repeat", outer_fold).parent.resolve()
        first_sessions = _load_array_record(
            stage_root,
            first_shard,
            first["held"]["session_ordinal"],
            *_held_schema("session_ordinal"),
        )
        repeat_sessions = _load_array_record(
            stage_root,
            repeat_shard,
            repeat["held"]["session_ordinal"],
            *_held_schema("session_ordinal"),
        )
        _validate_held_array("session_ordinal", first_sessions)
        _validate_held_array("session_ordinal", repeat_sessions)
        if (
            not np.array_equal(first_sessions, repeat_sessions)
            or len(first_sessions) != 400
            or np.any(first_sessions < 0)
            or np.any(first_sessions >= base.SESSION_COUNT)
            or len(np.unique(first_sessions)) != len(first_sessions)
        ):
            raise StrictRestackFreezeError("held session lineage is invalid")
        held_mask = np.zeros(base.SESSION_COUNT, dtype=np.uint8)
        held_mask[first_sessions] = 1
        domain = first["domains"]["H_%d" % outer_fold]
        if (
            _array_sha256(held_mask) != domain["mask_sha256"]
            or _array_sha256(first_sessions.astype(np.int16))
            != domain["session_sha256"]
        ):
            raise StrictRestackFreezeError("held session/domain hash mismatch")
        coverage[first_sessions] += 1
        for name in HELD_FIELDS:
            first_value = _load_array_record(
                stage_root,
                first_shard,
                first["held"][name],
                *_held_schema(name),
            )
            repeat_value = _load_array_record(
                stage_root,
                repeat_shard,
                repeat["held"][name],
                *_held_schema(name),
            )
            _validate_held_array(name, first_value)
            _validate_held_array(name, repeat_value)
            if not np.array_equal(first_value, repeat_value):
                raise StrictRestackFreezeError("held array repeat differs: %s" % name)
            stitched[name][first_sessions] = first_value
    if not np.all(coverage == 1):
        raise StrictRestackFreezeError("held shards do not cover all sessions once")
    freeze_dir.mkdir()
    stitched_files = {
        name: _write_array(freeze_dir / (name + ".npy"), value)
        for name, value in stitched.items()
    }
    coverage_file = _write_array(freeze_dir / "held_coverage.npy", coverage)
    identity = {
        "source_snapshot": source_snapshot,
        "worker_source_snapshot": recorded_snapshots[0],
        "outer_pairs": pairs,
        "models_per_pass": {
            "generic": 120,
            "focused": 30,
            "total": 150,
        },
        "passes": 2,
        "total_xgboost_fits": 300,
        "physical_cache_files": physical_file_counts,
        "held_coverage_sha256": coverage_file["array_sha256"],
        "stitched_arrays": {
            name: record["array_sha256"] for name, record in stitched_files.items()
        },
    }
    if _source_snapshot() != source_snapshot:
        raise StrictRestackFreezeError("freeze source changed during execution")
    for records in worker_result_files.values():
        for record in records:
            path = (ROOT / record["path"]).resolve()
            if (
                _sha256(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]
            ):
                raise StrictRestackFreezeError(
                    "worker result changed during freeze"
                )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.8-STRICT-OUTER-RESTACK-STAGE1",
        "status": "CACHE_REPEAT_FROZEN",
        "evidence_boundary": "target-free cache only; outcome labels were not opened",
        "stage_root": stage_root.relative_to(ROOT).as_posix(),
        "outer_pairs": pairs,
        "exact_repeat": {
            "equal": True,
            "outer_shards_compared": 5,
            "models_compared_per_pass": 150,
            "identity_sha256": _canonical_sha256(identity),
        },
        "stitched": {
            "files": stitched_files,
            "held_coverage": coverage_file,
        },
        "sources": source_snapshot,
        "worker_sources": recorded_snapshots[0],
        "audited_worker_result_files": worker_result_files,
        "audited_physical_cache_files": physical_file_counts,
        "privacy": {
            "outcome_label_archive_opened": False,
            "held_state_or_metric_computed": False,
            "frozen_current_comparator_opened": False,
            "agent_or_official_evaluator_started": False,
            "calibration_selection_confirmation_public_or_external_opened": False,
        },
        "resource": {
            "freeze_wall_seconds": round(time.perf_counter() - started, 6),
            "worker_wall_seconds": {
                pass_name: round(
                    sum(
                        float(row["resource"]["total_wall_seconds"])
                        for row in outer_results[pass_name]
                    ),
                    6,
                )
                for pass_name in ("first", "repeat")
            },
            "worker_peak_working_set_bytes": max(
                int(row["resource"]["peak_working_set_bytes"])
                for rows in outer_results.values()
                for row in rows
            ),
        },
        "identity": identity,
    }
    probe._assert_no_identity_matches(result)
    probe._write_json_exclusive(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "exact_repeat": True,
                "outer_shards": 5,
                "models_per_pass": 150,
                "result": result_path.relative_to(ROOT).as_posix(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    run(args.stage_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
