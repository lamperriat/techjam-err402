from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import probe_small_ranker_top5_proposal_depth as subject


def _file_record(path: Path, value: np.ndarray) -> dict:
    return {
        "path": str(path),
        "sha256": subject._sha256(path),
        "array_sha256": subject._array_sha256(value),
        "bytes": path.stat().st_size,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "asin_shape_matches": 0,
    }


def _expanded_surface(
    candidates: np.ndarray, source_mask: np.ndarray, available: np.ndarray
) -> subject.ExpandedRuntimeSurface:
    leading = candidates.shape[:2]
    return subject.ExpandedRuntimeSurface(
        current_chosen=np.zeros(leading, dtype=np.uint8),
        current_activation=np.zeros(leading, dtype=bool),
        current_choice=np.zeros(leading, dtype=np.uint8),
        incumbent=np.zeros(leading, dtype=np.uint8),
        family_choices=np.zeros((*leading, 3, 5), dtype=np.uint8),
        candidates=candidates,
        source_mask=source_mask,
        available=available,
        features=np.zeros((*candidates.shape, 19), dtype=np.float32),
    )


class _ArrayLoader:
    def __init__(self, values: dict) -> None:
        self.values = values

    def load(self, record):
        return self.values[record]


class _SyntheticFeatures:
    shape = (2_000, 10, 100, 133)

    def __getitem__(self, item):
        _sessions, _turns, candidates, feature_index = item
        return (
            np.asarray(candidates, dtype=np.float32) / 100.0
            + np.float32(feature_index) / 1_000.0
        )


def test_stage0_ast_has_closed_import_and_npy_load_boundary() -> None:
    path = subject.Path(__file__).resolve().parents[1] / "scripts" / (
        "probe_small_ranker_top5_proposal_depth.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed_roots = {
        "__future__",
        "argparse",
        "ctypes",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "numpy",
        "os",
        "pathlib",
        "re",
        "resource",
        "sys",
        "time",
        "typing",
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed_roots
    assert not imported & {"scripts", "evaluator", "starter"}

    forbidden_names = {
        "_load_outer_inputs",
        "_attach_isolated_labels",
        "_isolated_action_labels",
        "policy_session_state",
        "positive_index",
        "eligible_from",
        "NpzFile",
        "__import__",
        "import_module",
        "eval",
        "exec",
        "compile",
    }
    used_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    used_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not forbidden_names & used_names
    assert not (forbidden_names - {"compile"}) & used_attributes

    load_functions = []
    for function in (
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ):
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            target = call.func
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "np"
                and target.attr == "load"
            ):
                load_functions.append(function.name)
                keywords = {item.arg: item.value for item in call.keywords}
                assert ast.literal_eval(keywords["allow_pickle"]) is False
                assert ast.literal_eval(keywords["mmap_mode"]) == "r"
    assert load_functions == ["_load_npy_mmap"]
    assert subject.TOP_K == 5
    assert subject.FAMILY_COUNT == 3
    assert subject.MAX_ACTIONS == 15
    assert subject._canonical_sha256(list(subject.FEATURE_NAMES)) == (
        "f5a8fa2a1d6c2d191a43c01f741b65d0850e928291a64ee4e8c996fafa7b0bec"
    )


def test_protocol_hashes_and_fixed_semantics() -> None:
    prereg, amendment, result = subject._validate_protocol()
    assert prereg["proposal_contract"]["raw_depth"] == 5
    assert prereg["proposal_contract"]["refill_after_exclusion"] is False
    assert prereg["proposal_contract"]["maximum_unique_actions_per_turn"] == 15
    assert amendment["compatibility_oracle"]["required_equal_fields"] == list(
        subject.SURFACE_FIELDS
    )
    assert amendment["stage0_scope"]["passes"] == ["first", "repeat"]
    assert amendment["stage0_scope"]["phases"] == list(subject.PHASES)
    assert result["privacy"]["outcome_label_archive_opened"] is False


def test_array_allowlist_rejects_unknown_npy_and_npz_before_open(
    tmp_path: Path, monkeypatch
) -> None:
    records = []
    for index in range(subject.EXPECTED_STAGE0_ARRAY_FILES):
        value = np.asarray([index], dtype=np.int16)
        path = tmp_path / ("allowed_%02d.npy" % index)
        np.save(path, value, allow_pickle=False)
        records.append(_file_record(path, value))
    arrays = subject._FrozenArrays(tmp_path, records)
    np.testing.assert_array_equal(arrays.load(records[0]), np.asarray([0]))

    unknown_value = np.asarray([99], dtype=np.int16)
    unknown_path = tmp_path / "unknown.npy"
    np.save(unknown_path, unknown_value, allow_pickle=False)
    unknown_record = _file_record(unknown_path, unknown_value)
    archive_path = tmp_path / "forbidden.npz"
    archive_path.write_bytes(b"not an archive")
    archive_record = {
        "path": str(archive_path),
        "sha256": "unused",
        "array_sha256": "unused",
        "bytes": archive_path.stat().st_size,
        "shape": [1],
        "dtype": "int16",
        "asin_shape_matches": 0,
    }
    calls = {"hash": 0, "load": 0}

    def unexpected_hash(_path):
        calls["hash"] += 1
        raise AssertionError("a rejected path was hashed")

    def unexpected_load(_path):
        calls["load"] += 1
        raise AssertionError("a rejected path was loaded")

    monkeypatch.setattr(subject, "_sha256", unexpected_hash)
    monkeypatch.setattr(subject, "_load_npy_mmap", unexpected_load)
    with pytest.raises(subject.Top5ProbeError, match="allow-list"):
        arrays.load(unknown_record)
    with pytest.raises(subject.Top5ProbeError, match="frozen source root"):
        arrays.load(archive_record)
    assert calls == {"hash": 0, "load": 0}


def test_raw_top5_precedes_exclusion_and_never_refills() -> None:
    scores = np.full((1, 10, 100), -1_000.0, dtype=np.float32)
    scores[..., 1] = 1_000.0
    for candidate, score in (
        (3, 100.0),
        (12, 99.0),
        (20, 98.0),
        (21, 98.0),
        (22, 97.0),
        (23, 96.0),
        (24, 95.0),
    ):
        scores[..., candidate] = score
    incumbent = np.full((1, 10), 3, dtype=np.uint8)
    current = np.full((1, 10), 12, dtype=np.uint8)
    raw = subject._stable_raw_topk(scores, incumbent, 5)
    np.testing.assert_array_equal(raw[0, 0], [3, 12, 20, 21, 22])

    candidates, source_mask, available = subject._deduplicate_topk(
        np.stack((raw, raw, raw), axis=2), current, incumbent
    )
    np.testing.assert_array_equal(candidates[0, 0, :3], [20, 21, 22])
    np.testing.assert_array_equal(source_mask[0, 0, :3], [0b111] * 3)
    assert available[0, 0].sum() == 3
    assert 23 not in candidates[0, 0]
    assert 24 not in candidates[0, 0]


def test_dedup_support_padding_winner_ties_and_causal_latch() -> None:
    raw = np.asarray(
        [[[[40, 30, 20, 50, 60], [30, 25, 40, 70, 80], [25, 20, 90, 40, 30]]]],
        dtype=np.uint8,
    )
    current = np.asarray([[70]], dtype=np.uint8)
    incumbent = np.asarray([[3]], dtype=np.uint8)
    candidates, source_mask, available = subject._deduplicate_topk(
        raw, current, incumbent
    )
    np.testing.assert_array_equal(
        candidates[0, 0, :8], [20, 25, 30, 40, 50, 60, 80, 90]
    )
    np.testing.assert_array_equal(
        source_mask[0, 0, :8], [0b101, 0b110, 0b111, 0b111, 1, 1, 2, 4]
    )
    np.testing.assert_array_equal(candidates[0, 0, 8:], -1)
    np.testing.assert_array_equal(source_mask[0, 0, 8:], 0)
    assert not available[0, 0, 8:].any()

    utility = np.zeros(candidates.shape, dtype=np.float32)
    _slot, winner, value, winner_available = subject._within_turn_winner(
        candidates, source_mask, available, utility
    )
    assert winner.item() == 30
    utility[0, 0, 4] = np.nextafter(np.float32(0), np.float32(1))
    _slot, winner, value, winner_available = subject._within_turn_winner(
        candidates, source_mask, available, utility
    )
    assert winner.item() == 50

    winner_candidates = np.asarray([[20, 21, 22], [30, 31, 32]], dtype=np.int16)
    winner_utilities = np.asarray([[0.4, 0.5, 0.8], [0.9, 0.9, 0.9]], dtype=np.float32)
    winner_availability = np.ones((2, 3), dtype=bool)
    supplement, supplemental_choice = subject._causal_latch(
        winner_candidates, winner_utilities, winner_availability, 0.5
    )
    np.testing.assert_array_equal(supplement, [[False, True, False], [True, False, False]])
    np.testing.assert_array_equal(
        supplemental_choice, [[-1, 21, -1], [30, -1, -1]]
    )


def test_build_surface_k5_feature_contract() -> None:
    current_scores = -np.broadcast_to(
        np.arange(100, dtype=np.float32), (1, 10, 100)
    ).copy()
    family_scores = []
    for candidates in (range(20, 25), range(22, 27), range(24, 29)):
        values = current_scores.copy()
        for rank, candidate in enumerate(candidates):
            values[..., candidate] = 100.0 - rank
        family_scores.append(values)
    chosen = np.full((1, 10), 3, dtype=np.uint8)
    activation = np.zeros((1, 10), dtype=bool)
    incumbent = np.full((1, 10), 3, dtype=np.uint8)
    surface = subject._build_surface(
        _SyntheticFeatures(),
        np.asarray([0], dtype=np.int16),
        current_scores,
        family_scores,
        chosen,
        activation,
        incumbent,
        5,
    )
    assert surface.family_choices.shape == (1, 10, 3, 5)
    assert surface.candidates.shape == (1, 10, 15)
    assert surface.features.shape == (1, 10, 15, 19)
    assert np.isfinite(surface.features).all()
    np.testing.assert_array_equal(
        surface.features[..., 3],
        np.where(
            surface.available,
            surface.available.sum(axis=2, dtype=np.float32)[..., None] / 15.0,
            0.0,
        ),
    )
    for family in range(3):
        np.testing.assert_array_equal(
            surface.features[..., 4 + family],
            ((surface.source_mask & (1 << family)) != 0).astype(np.float32),
        )
    assert np.all(surface.candidates[surface.available] >= 10)
    assert np.all(surface.features[~surface.available] == 0.0)


def test_k1_full_surface_parity_is_all_nine_fields_bit_exact() -> None:
    leading = (1, 1)
    rebuilt = subject.ExpandedRuntimeSurface(
        current_chosen=np.zeros(leading, dtype=np.uint8),
        current_activation=np.zeros(leading, dtype=bool),
        current_choice=np.ones(leading, dtype=np.uint8),
        incumbent=np.full(leading, 2, dtype=np.uint8),
        family_choices=np.asarray([[[[10], [11], [12]]]], dtype=np.uint8),
        candidates=np.asarray([[[10, 11, 12]]], dtype=np.int16),
        source_mask=np.asarray([[[1, 2, 4]]], dtype=np.uint8),
        available=np.ones((1, 1, 3), dtype=bool),
        features=np.zeros((1, 1, 3, 19), dtype=np.float32),
    )
    old_files = {name: name for name in subject.SURFACE_FIELDS}
    expected = {
        name: np.asarray(getattr(rebuilt, name)).copy()
        for name in subject.SURFACE_FIELDS
    }
    expected["family_choices"] = expected["family_choices"][..., 0]
    hashes = subject._assert_k1_parity(
        rebuilt, old_files, _ArrayLoader(expected)
    )
    assert set(hashes) == set(subject.SURFACE_FIELDS)

    for field in subject.SURFACE_FIELDS:
        altered = {name: value.copy() for name, value in expected.items()}
        flat = altered[field].reshape(-1)
        if flat.dtype == np.bool_:
            flat[0] = ~flat[0]
        elif np.issubdtype(flat.dtype, np.floating):
            flat[0] = np.nextafter(flat[0], np.float32(1))
        else:
            flat[0] = flat[0] + 1
        with pytest.raises(subject.Top5ProbeError, match=field):
            subject._assert_k1_parity(
                rebuilt, old_files, _ArrayLoader(altered)
            )


def test_top5_preserves_top1_candidate_and_support_superset() -> None:
    old_candidates = np.asarray([[[20, 30, -1]]], dtype=np.int16)
    old_available = np.asarray([[[True, True, False]]])
    old_mask = np.asarray([[[0b001, 0b110, 0]]], dtype=np.uint8)
    old_files = {
        "candidates": "candidates",
        "available": "available",
        "source_mask": "source_mask",
    }
    loader = _ArrayLoader(
        {
            "candidates": old_candidates,
            "available": old_available,
            "source_mask": old_mask,
        }
    )
    candidates = np.full((1, 1, 15), -1, dtype=np.int16)
    source_mask = np.zeros((1, 1, 15), dtype=np.uint8)
    available = np.zeros((1, 1, 15), dtype=bool)
    candidates[0, 0, :2] = [20, 30]
    source_mask[0, 0, :2] = [0b011, 0b111]
    available[0, 0, :2] = True
    expanded = _expanded_surface(candidates, source_mask, available)
    assert subject._assert_top1_subset(old_files, expanded, loader) == 2

    lost_support = source_mask.copy()
    lost_support[0, 0, 0] = 0b010
    with pytest.raises(subject.Top5ProbeError, match="support bit"):
        subject._assert_top1_subset(
            old_files,
            _expanded_surface(candidates, lost_support, available),
            loader,
        )
    missing = available.copy()
    missing[0, 0, 1] = False
    with pytest.raises(subject.Top5ProbeError, match="support bit"):
        subject._assert_top1_subset(
            old_files,
            _expanded_surface(candidates, source_mask, missing),
            loader,
        )


def test_stage0_identity_is_pass_neutral_and_semantically_sensitive() -> None:
    record = {
        "files": {
            name: {
                "path": "first/%s.npy" % name,
                "sha256": "file-" + name,
                "array_sha256": "array-" + name,
                "bytes": 1,
            }
            for name in subject.SURFACE_FIELDS
        },
        "available_action_rows": 7,
        "action_turns": 5,
        "width_histogram": {"0": 2, "1": 5},
        "feature_order_sha256": "features",
        "wall_seconds": 1.0,
    }
    repeat = deepcopy(record)
    for value in repeat["files"].values():
        value["path"] = value["path"].replace("first", "repeat")
        value["sha256"] = "different-file-sha"
        value["bytes"] = 999
    repeat["wall_seconds"] = 999.0
    first_identity = subject._phase_identity(record)
    repeat_identity = subject._phase_identity(repeat)
    assert first_identity == repeat_identity
    assert subject._canonical_sha256(first_identity) == subject._canonical_sha256(
        repeat_identity
    )
    assert set(first_identity) == {
        "array_sha256",
        "available_action_rows",
        "action_turns",
        "width_histogram",
        "feature_order_sha256",
    }
    changed = deepcopy(first_identity)
    changed["array_sha256"][subject.SURFACE_FIELDS[0]] = "changed"
    assert subject._canonical_sha256(changed) != subject._canonical_sha256(
        first_identity
    )
    with pytest.raises(ValueError):
        subject._canonical_sha256({"invalid": float("nan")})
