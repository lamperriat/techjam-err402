from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pytest

from scripts import probe_small_ranker_top5_proposal_depth as mechanics
from scripts import small_ranker_portfolio_selector_py39 as frozen_selector


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/build_small_ranker_top5_proposal_depth_stage1b.py"
EXPECTED_LABEL_MEMBERS = (
    "baseline_rank",
    "positive_index",
    "eligible_from",
    "inner_fold",
    "family_index",
)
FORBIDDEN_LABEL_MEMBERS = {
    "baseline_session_hit",
    "outer_fold",
    "popularity_code",
    "taxonomy_code",
    "training_indices",
    "training_length",
}
RUNTIME_FIELDS = mechanics.SURFACE_FIELDS
DERIVED_LABEL_FIELDS = (
    "rescue",
    "rescue_weights",
    "regret",
    "regret_weights",
    "rr_loss",
    "mttc_loss",
)


def _subject():
    assert SOURCE.is_file(), "Stage1b builder has not been added"
    return importlib.import_module(
        "scripts.build_small_ranker_top5_proposal_depth_stage1b"
    )


class _FakeArchive:
    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.arrays = dict(arrays)
        self.accesses: list[str] = []
        self.closed = False

    @property
    def files(self):  # pragma: no cover - touching this is itself a failure
        raise AssertionError("the sealed archive member list must not be inspected")

    def __iter__(self):  # pragma: no cover - touching this is itself a failure
        raise AssertionError("the sealed archive must not be iterated")

    def __enter__(self):
        assert not self.closed
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.closed = True

    def __getitem__(self, name: str) -> np.ndarray:
        assert not self.closed
        self.accesses.append(name)
        if name not in EXPECTED_LABEL_MEMBERS:
            raise AssertionError("unexpected archive member access: %s" % name)
        return self.arrays[name]


class _FakeLoader:
    def __init__(self, archive: _FakeArchive) -> None:
        self.archive = archive
        self.calls: list[tuple[Path, dict[str, Any]]] = []

    def __call__(self, path: Path, **kwargs: Any) -> _FakeArchive:
        self.calls.append((Path(path), dict(kwargs)))
        return self.archive


def _sealed_test_path(tmp_path: Path) -> tuple[Path, str, int]:
    path = tmp_path / "labels_v2.npz"
    payload = b"synthetic sealed archive; FakeLoader supplies members"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest(), len(payload)


def _fake_label_arrays(
    *, bad_shape: Optional[str] = None, bad_dtype: Optional[str] = None
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    training = np.arange(1600, dtype=np.int16)
    held = np.arange(1600, 2000, dtype=np.int16)
    sentinels = {
        "baseline_rank": 251,
        "positive_index": 30001,
        "eligible_from": 251,
        "inner_fold": 251,
        "family_index": 2_000_000_001,
    }
    arrays = {
        "baseline_rank": np.zeros((2000, 10), dtype=np.uint8),
        "positive_index": np.full((2000, 10), -1, dtype=np.int16),
        "eligible_from": np.ones(2000, dtype=np.uint8),
        "inner_fold": (np.arange(2000) % 5).astype(np.uint8),
        "family_index": np.arange(2000, dtype=np.int32),
    }
    for name, value in sentinels.items():
        arrays[name][held] = value
    if bad_shape is not None:
        arrays[bad_shape] = arrays[bad_shape][:-1].copy()
    if bad_dtype is not None:
        arrays[bad_dtype] = arrays[bad_dtype].astype(np.float64)
    return arrays, training, sentinels


def _runtime(sessions: int, width: int) -> frozen_selector.RuntimePortfolioSurface:
    turns = 10
    current_chosen = np.full((sessions, turns), 13, dtype=np.uint8)
    current_activation = np.zeros((sessions, turns), dtype=bool)
    incumbent = np.full((sessions, turns), 9, dtype=np.uint8)
    current_choice = incumbent.copy()
    family_choices = np.full((sessions, turns, 3), 10, dtype=np.uint8)
    candidates = np.full((sessions, turns, width), -1, dtype=np.int16)
    source_mask = np.zeros((sessions, turns, width), dtype=np.uint8)
    available = np.zeros((sessions, turns, width), dtype=bool)
    for slot in range(width):
        candidates[..., slot] = 10 + slot
        source_mask[..., slot] = 1 << (slot % 3)
        available[..., slot] = True
    features = np.zeros(
        (sessions, turns, width, len(frozen_selector.FEATURE_NAMES)),
        dtype=np.float32,
    )
    return frozen_selector.RuntimePortfolioSurface(
        current_chosen=current_chosen,
        current_activation=current_activation,
        current_choice=current_choice,
        incumbent=incumbent,
        family_choices=family_choices,
        candidates=candidates,
        source_mask=source_mask,
        available=available,
        features=features,
    )


def _small_labels(sessions: int) -> dict[str, np.ndarray]:
    return {
        "baseline_rank": np.zeros((sessions, 10), dtype=np.uint8),
        "positive_index": np.full((sessions, 10), -1, dtype=np.int16),
        "eligible_from": np.ones(sessions, dtype=np.uint8),
        "inner_fold": (np.arange(sessions) % 5).astype(np.uint8),
        "family_index": np.arange(sessions, dtype=np.int32),
    }


def _names_in_call(call: ast.Call) -> set[str]:
    direct = [*call.args, *(keyword.value for keyword in call.keywords)]
    return {node.id for node in direct if isinstance(node, ast.Name)}


def test_stage1b_ast_closes_import_and_archive_member_boundaries() -> None:
    subject = _subject()
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_roots.add((node.module or "").split(".")[0])
    assert not imported_roots & {"evaluator", "starter"}

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not names & {"__import__", "eval", "exec"}
    assert "import_module" not in attributes
    assert not attributes & {"files", "namelist", "infolist"}
    assert tuple(subject.LABEL_MEMBER_SPECS) == EXPECTED_LABEL_MEMBERS

    loader_source = inspect.getsource(subject._load_t_only_labels)
    loader_tree = ast.parse(loader_source)
    string_literals = {
        node.value
        for node in ast.walk(loader_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not string_literals & FORBIDDEN_LABEL_MEMBERS
    archive_attributes = {
        node.attr
        for node in ast.walk(loader_tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"archive", "labels_archive"}
    }
    assert not archive_attributes & {
        "files",
        "keys",
        "items",
        "values",
        "namelist",
        "infolist",
    }


def test_protocol_validation_never_touches_label_archive() -> None:
    subject = _subject()
    source = inspect.getsource(subject._validate_protocol_without_labels)
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_load_t_only_labels" not in called_names
    assert "load" not in called_attributes
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        function = call.func.id if isinstance(call.func, ast.Name) else ""
        attribute = call.func.attr if isinstance(call.func, ast.Attribute) else ""
        if function in {"_sha256", "_load_t_only_labels"} or attribute in {
            "open",
            "read_bytes",
            "stat",
            "is_file",
            "is_symlink",
            "load",
        }:
            argument_names = {
                node.id
                for argument in (*call.args, *(row.value for row in call.keywords))
                for node in ast.walk(argument)
                if isinstance(node, ast.Name)
            }
            assert not any("LABEL_ARCHIVE" in name for name in argument_names)


def test_run_freezes_complete_split_lineage_before_first_label_hash() -> None:
    subject = _subject()
    source = inspect.getsource(subject.run)
    first_label_hash = source.index(
        "label_archive_start = _audit_label_archive(Path(label_archive))"
    )
    for required_before_hash in (
        "source_start = _source_snapshot()",
        "stage1a._validate_protocol()",
        'stage1a._load_partition_orders(\n        frozen_v28_result, "first"',
        'stage1a._load_partition_orders(\n        frozen_v28_result, "repeat"',
        "held_coverage = _validate_held_coverage(first_held)",
        "_validate_training_order(first_training[fold])",
        'raise Stage1BError("T_o is not the exact ordered H_o complement")',
        'expected_held.get("per_outer_training_order_sha256")',
    ):
        assert source.index(required_before_hash) < first_label_hash


def test_t_only_loader_reads_exact_five_members_and_closes_before_use(
    tmp_path: Path,
) -> None:
    subject = _subject()
    path, digest, byte_count = _sealed_test_path(tmp_path)
    arrays, training, sentinels = _fake_label_arrays()
    archive = _FakeArchive(arrays)
    loader = _FakeLoader(archive)

    labels_t = subject._load_t_only_labels(
        path,
        training,
        expected_sha256=digest,
        expected_bytes=byte_count,
        np_load=loader,
    )

    assert archive.closed is True
    assert archive.accesses == list(EXPECTED_LABEL_MEMBERS)
    assert len(loader.calls) == 1
    assert loader.calls[0][1].get("allow_pickle") is False
    assert set(labels_t) == set(EXPECTED_LABEL_MEMBERS)
    for name, value in labels_t.items():
        assert len(value) == 1600
        assert sentinels[name] not in np.asarray(value)
        assert np.asarray(value).flags.writeable is False

    # Supervision begins only after the archive context has closed.
    runtime = _runtime(1600, 3)
    subject._attach_t_only_labels(runtime, labels_t)
    assert archive.closed is True


@pytest.mark.parametrize(
    "fault,member",
    [
        ("missing", None),
        ("hash", None),
        ("bytes", None),
        ("shape", "baseline_rank"),
        ("dtype", "positive_index"),
    ],
)
def test_t_only_loader_fails_closed_on_path_hash_shape_or_dtype(
    tmp_path: Path, fault: str, member: Optional[str]
) -> None:
    subject = _subject()
    path, digest, byte_count = _sealed_test_path(tmp_path)
    arrays, training, _sentinels = _fake_label_arrays(
        bad_shape=member if fault == "shape" else None,
        bad_dtype=member if fault == "dtype" else None,
    )
    archive = _FakeArchive(arrays)
    loader = _FakeLoader(archive)
    if fault == "missing":
        path = tmp_path / "missing.npz"
    elif fault == "hash":
        digest = "0" * 64
    elif fault == "bytes":
        byte_count += 1

    with pytest.raises(subject.Stage1BError):
        subject._load_t_only_labels(
            path,
            training,
            expected_sha256=digest,
            expected_bytes=byte_count,
            np_load=loader,
        )
    if fault in {"missing", "hash", "bytes"}:
        assert loader.calls == []
    else:
        assert archive.closed is True


def test_dynamic_width_three_matches_frozen_label_helper() -> None:
    subject = _subject()
    runtime = _runtime(3, 3)
    labels = _small_labels(3)
    labels["positive_index"][:, 0] = np.asarray([10, 11, 12])

    expected = frozen_selector._attach_isolated_labels(runtime, labels)
    actual = subject._attach_t_only_labels(runtime, labels)
    for name in (*RUNTIME_FIELDS, *DERIVED_LABEL_FIELDS):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_dynamic_width_three_full_keep_pipeline_matches_v28(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import build_small_ranker_strict_outer_restack as reference

    subject = _subject()
    runtime = _runtime(10, 3)
    labels = _small_labels(10)
    expected_training = frozen_selector._attach_isolated_labels(runtime, labels)
    actual_training = subject._attach_t_only_labels(runtime, labels)
    current_state = subject.metric.policy_session_state(
        labels, runtime.current_chosen, runtime.current_activation
    )
    captured: dict[str, dict[str, np.ndarray]] = {"expected": {}, "actual": {}}

    def writer(side: str):
        def write(path: Path, value: np.ndarray) -> dict[str, Any]:
            array = np.asarray(value).copy()
            captured[side][path.name] = array
            return {
                "path": "%s/%s" % (side, path.name),
                "sha256": mechanics._array_sha256(array),
                "array_sha256": mechanics._array_sha256(array),
                "bytes": int(array.nbytes),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "asin_shape_matches": 0,
            }

        return write

    monkeypatch.setattr(reference, "_write_array", writer("expected"))
    monkeypatch.setattr(subject, "_write_array", writer("actual"))
    expected_chosen, expected_activation, expected_record = reference._strict_selector(
        expected_training,
        labels,
        current_state,
        runtime,
        runtime,
        Path("expected"),
    )
    actual_chosen, actual_activation, actual_record = subject._run_selector(
        actual_training,
        labels,
        current_state,
        runtime,
        runtime,
        Path("actual"),
    )

    np.testing.assert_array_equal(actual_chosen, expected_chosen)
    np.testing.assert_array_equal(actual_activation, expected_activation)
    assert actual_record["selected_quantile"] == expected_record["selected_quantile"]
    assert (
        actual_record["mapped_reference_threshold"]
        == expected_record["mapped_reference_threshold"]
    )
    assert actual_record["inner_selection"] == expected_record["inner_selection"]
    assert actual_record["inner_fit_readiness"] == expected_record["inner_fit_readiness"]
    assert set(captured["actual"]) == set(captured["expected"])
    for name in captured["expected"]:
        np.testing.assert_array_equal(
            captured["actual"][name], captured["expected"][name]
        )


def test_dynamic_width_fifteen_labels_slot_fourteen_and_preserves_runtime() -> None:
    subject = _subject()
    runtime = _runtime(1, 15)
    labels = _small_labels(1)
    labels["positive_index"][0, 0] = 24
    before = {
        name: np.asarray(getattr(runtime, name)).copy() for name in RUNTIME_FIELDS
    }

    attached = subject._attach_t_only_labels(runtime, labels)

    assert attached.rescue.shape == (1, 10, 15)
    assert attached.rescue[0, 0, 14] == 1
    assert int(attached.rescue.sum()) == 1
    for name in RUNTIME_FIELDS:
        np.testing.assert_array_equal(getattr(runtime, name), before[name])
        np.testing.assert_array_equal(getattr(attached, name), before[name])


def test_held_application_api_and_selector_calls_never_accept_held_labels() -> None:
    subject = _subject()
    apply_parameters = inspect.signature(subject._apply_held_policy).parameters
    forbidden = ("label", "target", "outcome", "positive", "oracle")
    assert not any(
        token in name.lower() for name in apply_parameters for token in forbidden
    )
    run_parameters = inspect.signature(subject._run_selector).parameters
    assert not any(
        name.lower().startswith("labels_h")
        or "held_label" in name.lower()
        or "held_outcome" in name.lower()
        for name in run_parameters
    )

    run_tree = ast.parse(inspect.getsource(subject._run_selector))
    for call in (node for node in ast.walk(run_tree) if isinstance(node, ast.Call)):
        names = _names_in_call(call)
        assert not ({"held", "labels_t"} <= names), ast.unparse(call)


def test_repeat_validation_accepts_identity_and_rejects_drift() -> None:
    subject = _subject()
    identity = {
        "outer_fold": 2,
        "selected_quantile": 0.5,
        "final_sha256": "2" * 64,
    }
    first = {
        "outer_fold": 2,
        "identity": identity,
        "identity_sha256": subject._canonical_sha256(identity),
        "files": {},
    }
    repeat = {
        "outer_fold": 2,
        "identity": dict(identity),
        "identity_sha256": first["identity_sha256"],
        "files": {},
    }
    summary = subject._validate_repeat(first, repeat)
    assert summary["equal"] is True
    assert summary["outer_fold"] == 2

    repeat["identity"]["final_sha256"] = "b" * 64
    repeat["identity_sha256"] = subject._canonical_sha256(repeat["identity"])
    with pytest.raises(subject.Stage1BError, match="repeat|identity|drift"):
        subject._validate_repeat(first, repeat)


def test_held_coverage_requires_exact_disjoint_cover() -> None:
    subject = _subject()
    held_orders = tuple(
        np.arange(fold * 400, (fold + 1) * 400, dtype=np.int16)
        for fold in range(5)
    )
    record = subject._validate_held_coverage(held_orders)
    assert record["unique_sessions"] == 2000
    assert record["missing_sessions"] == 0
    assert record["overlap_sessions"] == 0

    overlap = list(held_orders)
    overlap[4] = overlap[4].copy()
    overlap[4][0] = overlap[0][0]
    with pytest.raises(subject.Stage1BError, match="held|cover|overlap"):
        subject._validate_held_coverage(tuple(overlap))


def test_identity_short_circuit_and_nonidentity_stage2_eligibility() -> None:
    subject = _subject()
    current_chosen = np.full((2000, 10), 10, dtype=np.uint8)
    current_activation = np.zeros((2000, 10), dtype=bool)
    identical = subject._stage2_decision(
        current_chosen,
        current_activation,
        current_chosen.copy(),
        current_activation.copy(),
    )
    assert identical["identity_short_circuit"] is True
    assert identical["stage2_eligible_after_tracked_manifest"] is False
    assert identical["stage2_outcome_protocol_authorized"] is False
    assert identical["held_outcome_attach_runs"] == 0

    changed_chosen = current_chosen.copy()
    changed_activation = current_activation.copy()
    changed_chosen[17, 3] = 42
    changed_activation[17, 3] = True
    nonidentity = subject._stage2_decision(
        current_chosen,
        current_activation,
        changed_chosen,
        changed_activation,
    )
    assert nonidentity["identity_short_circuit"] is False
    assert nonidentity["stage2_eligible_after_tracked_manifest"] is True
    assert nonidentity["stage2_preparation_authorized_now"] is False
    assert nonidentity["stage2_outcome_protocol_authorized"] is False
    assert nonidentity["held_outcome_attach_runs"] == 0
