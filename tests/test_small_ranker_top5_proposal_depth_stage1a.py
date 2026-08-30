from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/build_small_ranker_top5_proposal_depth_stage1a.py"


def _subject():
    from scripts import build_small_ranker_top5_proposal_depth_stage1a

    return build_small_ranker_top5_proposal_depth_stage1a


def _file_record(seed: str) -> dict:
    return {
        "path": "pass/" + seed + ".npy",
        "sha256": "file-" + seed,
        "array_sha256": "array-" + seed,
        "bytes": 128,
        "shape": [1],
        "dtype": "uint8",
        "asin_shape_matches": 0,
    }


def _outer_record(outer_fold: int, identity_sha256: str, pass_name: str) -> dict:
    subject = _subject()
    phases = {}
    for phase in subject.mechanics.PHASES:
        sessions = 400 if phase == "held_H" else 1600
        phases[phase] = {
            "sessions": sessions,
            "turns": sessions * 10,
            "available_action_rows": sessions * 10,
            "action_turns": sessions * 10,
            "width_mean": 1.0,
            "width_p50_higher": 1,
            "width_p95_higher": 1,
            "width_max": 1,
            "session_order_sha256": "order-" + phase,
            "feature_order_sha256": "features",
            "old_top1_action_rows": sessions * 5,
            "k1_full_surface_parity": True,
            "old_top1_subset": True,
            "causal_latch_at_most_one": True,
            "keep_composition_exact": True,
            "files": {
                field: _file_record("%s-%s" % (phase, field))
                for field in subject.mechanics.SURFACE_FIELDS
            },
        }
    identity = {
        "outer_fold": outer_fold,
        "source_outer_identity_sha256": "outer-source-%d" % outer_fold,
        "phases": {
            phase: {
                "array_sha256": {
                    field: phases[phase]["files"][field]["array_sha256"]
                    for field in subject.mechanics.SURFACE_FIELDS
                }
            }
            for phase in subject.mechanics.PHASES
        },
    }
    return {
        "pass_name": pass_name,
        "outer_fold": outer_fold,
        "status": "TARGET_FREE_SURFACE_COMPLETE",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "phases": phases,
        "sources": {
            "probe": "same",
            "projected_features": "same",
            "outer_result": pass_name + "-outer",
        },
        "privacy": {
            "label_archive_opened": False,
            "outcome_member_accesses": 0,
            "held_state_or_metric_computed": False,
            "agent_or_evaluator_started": False,
        },
    }


def test_stage1a_ast_allows_only_safe_hash_pinned_probe() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, alias.asname) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.extend(
                ("%s.%s" % (node.module, alias.name), alias.asname)
                for alias in node.names
            )
    allowed = {
        "__future__.annotations",
        "argparse",
        "hashlib",
        "json",
        "pathlib.Path",
        "sys",
        "time",
        "typing.Any",
        "typing.Dict",
        "typing.Mapping",
        "typing.Optional",
        "typing.Sequence",
        "typing.Tuple",
        "numpy",
        "scripts.probe_small_ranker_top5_proposal_depth",
    }
    assert {name for name, _alias in imports} == allowed
    assert not any(
        name.startswith(("evaluator", "starter")) for name, _alias in imports
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "load"
        for node in ast.walk(tree)
    )
    source_text = SOURCE.read_text(encoding="utf-8")
    for forbidden in (
        "_load_outer_inputs",
        "positive_index",
        "eligible_from",
        "policy_session_state",
        "NpzFile",
        "import_module",
    ):
        assert forbidden not in source_text


def test_stage0_prerequisite_is_hash_bound(monkeypatch) -> None:
    subject = _subject()
    prereg, result = subject._validate_protocol()
    assert prereg["single_algorithmic_variable"]["new"] == 5
    assert result["status"] == "CACHE_REPEAT_FROZEN"
    monkeypatch.setitem(subject.EXPECTED_HASHES, "stage0_result", "0" * 64)
    with pytest.raises(subject.Stage1AError, match="stage0_result"):
        subject._validate_protocol()


def test_held_partition_requires_exact_repeat_and_exact_global_cover() -> None:
    subject = _subject()
    held = [
        np.arange(fold * 400, (fold + 1) * 400, dtype=np.int16)
        for fold in range(5)
    ]
    training = []
    for values in held:
        mask = np.ones(2000, dtype=bool)
        mask[values] = False
        training.append(np.flatnonzero(mask).astype(np.int16))
    record = subject._held_partition(training, held, training, held)
    assert record["unique_sessions"] == 2000
    assert record["missing_sessions"] == 0
    assert record["overlap_sessions"] == 0
    assert record["coverage_count_array_sha256"] == (
        subject.V28_HELD_COVERAGE_SHA256
    )

    overlapping = [value.copy() for value in held]
    overlapping[1][0] = overlapping[0][0]
    with pytest.raises(subject.Stage1AError):
        subject._held_partition(training, overlapping, training, overlapping)
    changed_repeat = [value.copy() for value in held]
    changed_repeat[0] = changed_repeat[0][::-1]
    with pytest.raises(subject.Stage1AError, match="repeat or complement"):
        subject._held_partition(training, held, training, changed_repeat)


def test_outer_pair_requires_physical_repeat_and_outer_identity() -> None:
    subject = _subject()
    first = _outer_record(2, "identity-2", "first")
    repeat = _outer_record(2, "identity-2", "repeat")
    summary = subject._outer_pair_summary(
        first, repeat, 2, "outer-source-2"
    )
    assert summary["outer_fold"] == 2
    assert summary["physical_files_across_passes"] == 54

    drifted = deepcopy(repeat)
    drifted["phases"]["held_H"]["files"]["features"]["sha256"] = "changed"
    with pytest.raises(subject.Stage1AError, match="physical repeat"):
        subject._outer_pair_summary(first, drifted, 2, "outer-source-2")
    with pytest.raises(subject.Stage1AError, match="identity"):
        subject._outer_pair_summary(first, repeat, 2, "wrong-outer-source")


def test_aggregate_identity_is_pass_neutral_and_sensitive() -> None:
    subject = _subject()
    records = [
        _outer_record(fold, "identity-%d" % fold, "first") for fold in range(5)
    ]
    held = {
        "sessions": 2000,
        "outer_fold_by_session_sha256": "owner",
        "coverage_count_array_sha256": subject.V28_HELD_COVERAGE_SHA256,
    }
    first = subject._aggregate_identity(records, held)
    repeat_records = deepcopy(records)
    for record in repeat_records:
        record["pass_name"] = "repeat"
        record["resource"] = {"wall_seconds": 999.0}
    repeat = subject._aggregate_identity(repeat_records, held)
    assert first == repeat
    assert subject.mechanics._canonical_sha256(first) == (
        subject.mechanics._canonical_sha256(repeat)
    )

    changed = deepcopy(repeat_records)
    changed[3]["identity"]["phases"]["oof_T"]["array_sha256"][
        "features"
    ] = "changed"
    assert subject.mechanics._canonical_sha256(
        subject._aggregate_identity(changed, held)
    ) != subject.mechanics._canonical_sha256(first)


def test_output_root_is_direct_new_v2_9_stage1a_child() -> None:
    subject = _subject()
    valid = ROOT / "experiments/fast_track/small_ranker_v2_9/stage1a_unit_never_create"
    assert subject._validate_output_root(valid) == valid.resolve()
    with pytest.raises(subject.Stage1AError):
        subject._validate_output_root(
            ROOT / "experiments/fast_track/small_ranker_v2_9/not_stage1a"
        )
    with pytest.raises(subject.Stage1AError):
        subject._validate_output_root(
            ROOT / "experiments/fast_track/small_ranker_v2_9/stage1a_nested/child"
        )


def test_physical_output_audit_rejects_post_record_file_drift(
    tmp_path: Path, monkeypatch
) -> None:
    subject = _subject()
    output_root = tmp_path / "out"
    output_root.mkdir()
    monkeypatch.setattr(subject, "ROOT", tmp_path)

    pass_records = {pass_name: [] for pass_name in subject.PASSES}
    sequence = 0
    for pass_name in subject.PASSES:
        for outer_fold in subject.OUTER_FOLDS:
            phases = {}
            for phase in subject.mechanics.PHASES:
                files = {}
                for field in subject.mechanics.SURFACE_FIELDS:
                    path = (
                        output_root
                        / pass_name
                        / ("outer_%d" % outer_fold)
                        / phase
                        / (field + ".npy")
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value = np.asarray([sequence % 251], dtype=np.uint8)
                    np.save(path, value, allow_pickle=False)
                    files[field] = {
                        "path": path.relative_to(tmp_path).as_posix(),
                        "sha256": subject._sha256(path),
                        "array_sha256": subject.mechanics._array_sha256(value),
                        "bytes": path.stat().st_size,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "asin_shape_matches": 0,
                    }
                    sequence += 1
                phases[phase] = {"files": files}
            pass_records[pass_name].append(
                {"outer_fold": outer_fold, "phases": phases}
            )

    audit = subject._verify_output_records(output_root, pass_records)
    assert audit["registered_files"] == subject.EXPECTED_ARRAY_FILES
    assert audit["all_file_and_array_hashes_verified"] is True

    drift_path = (
        output_root
        / "repeat"
        / "outer_4"
        / subject.mechanics.PHASES[-1]
        / (subject.mechanics.SURFACE_FIELDS[-1] + ".npy")
    )
    np.save(drift_path, np.asarray([255], dtype=np.uint8), allow_pickle=False)
    with pytest.raises(subject.Stage1AError, match="physical file"):
        subject._verify_output_records(output_root, pass_records)
