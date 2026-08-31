from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import e0_embedding_candidate_worker as worker
from scripts import probe_e0_embedding_candidate_recall as probe


def _identifier(index: int) -> str:
    return f"A{index:09d}"


def _catalog(count: int = 620) -> frozenset[str]:
    return frozenset(_identifier(index) for index in range(count))


def _pool(count: int = 400) -> tuple[str, ...]:
    return tuple(_identifier(index) for index in range(count))


def _trace_fixture() -> tuple[
    list[dict[str, object]], list[tuple[str, ...]], frozenset[str]
]:
    c200_lengths = (100, 137, 200, 120)
    union_lengths = (150, 240, 400, 120)
    references = [_pool(length) for length in c200_lengths]
    records = [
        {"candidates": _pool(length), "ordinal": 1, "turn": turn}
        for turn, length in zip(range(1, 5), union_lengths, strict=True)
    ]
    return records, references, _catalog()


def _write_trace(path: Path, records: list[dict[str, object]]) -> bytes:
    payload = b"".join(
        worker.canonical_trace_line(
            int(row["ordinal"]), int(row["turn"]), row["candidates"]
        )
        for row in records
    )
    path.write_bytes(payload)
    return payload


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


@pytest.mark.parametrize("old_length", [100, 137, 200])
def test_dense_tail_preserves_complete_variable_c200_prefix(old_length: int) -> None:
    c200 = _pool(old_length)
    dense = (
        *reversed(c200[:10]),
        *(_identifier(index) for index in range(old_length, 520)),
    )[:400]

    result = worker.append_dense_unseen_tail(c200, dense, _catalog(), limit=400)

    assert result[:old_length] == c200
    assert len(result) <= 400
    assert len(result) == len(set(result))
    assert c200 == _pool(old_length)


def test_dense_tail_stably_skips_seen_and_duplicates_and_caps_at_400() -> None:
    c200 = _pool(100)
    dense = (
        c200[7],
        _identifier(450),
        _identifier(201),
        _identifier(450),
        c200[2],
        _identifier(200),
        *(_identifier(index) for index in range(202, 620)),
    )[:400]

    result = worker.append_dense_unseen_tail(c200, dense, _catalog(), limit=400)

    assert len(result) == 400
    assert result[:100] == c200
    assert result[100:103] == (
        _identifier(450),
        _identifier(201),
        _identifier(200),
    )
    expected_tail = tuple(dict.fromkeys(value for value in dense if value not in c200))
    assert result[100:] == expected_tail[:300]


def test_dense_tail_does_not_pad_when_query_route_is_empty() -> None:
    c200 = _pool(137)
    assert worker.append_dense_unseen_tail(c200, (), _catalog()) == c200


@pytest.mark.parametrize("limit", [399, 401, True])
def test_dense_tail_rejects_any_nonfrozen_limit(limit: object) -> None:
    with pytest.raises(worker.E0WorkerError):
        worker.append_dense_unseen_tail(_pool(100), (), _catalog(), limit=limit)


def test_dense_tail_rejects_bad_prefix_or_dense_members() -> None:
    duplicated = list(_pool(100))
    duplicated[-1] = duplicated[0]
    unknown = (*_pool(100), "Z999999999")
    non_string: list[object] = [*_pool(100), 7]
    overdepth = tuple(_identifier(index) for index in range(401))

    for c200, dense in (
        (_pool(99), ()),
        (_pool(201), ()),
        (duplicated, ()),
        (_pool(100), unknown),
        (_pool(100), non_string),
        (_pool(100), overdepth),
    ):
        with pytest.raises(worker.E0WorkerError):
            worker.append_dense_unseen_tail(c200, dense, _catalog())


def test_validate_e0_rejects_reference_loss_reorder_duplicate_and_invention() -> None:
    c200 = _pool(137)
    valid = _pool(240)
    assert worker.validate_e0_candidates(valid, c200, _catalog()) == valid

    reordered = list(valid)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    duplicated = list(valid)
    duplicated[-1] = duplicated[0]
    invented = list(valid)
    invented[-1] = "Z999999999"
    for invalid in (reordered, duplicated, invented, _pool(136), _pool(401)):
        with pytest.raises(worker.E0WorkerError):
            worker.validate_e0_candidates(invalid, c200, _catalog())


class _ControlBase:
    def _apply_p11(
        self,
        state: object,
        rankings: dict[str, list[str]],
        candidate_rowids: dict[str, int],
        query_terms: list[str],
    ) -> tuple[list[str], dict[str, object]]:
        del state, candidate_rowids, query_terms
        production = tuple(rankings["final"])
        self._c200_last_capture = {"c200": production}
        return list(production), {"configured_mode": "control", "output_changed": False}


class _SyntheticE0Agent(worker.E0CaptureAgent, _ControlBase):
    pass


class _FakeIndex:
    def __init__(self, identifiers: tuple[str, ...], *, should_run: bool = True) -> None:
        self.identifiers = identifiers
        self.should_run = should_run
        self.calls: list[tuple[str, object, int]] = []

    def search_query(self, query: str, encoder: object, *, top_k: int) -> list[object]:
        if not self.should_run:
            raise AssertionError("empty query invoked semantic search")
        self.calls.append((query, encoder, top_k))
        return [SimpleNamespace(parent_asin=value) for value in self.identifiers]


def _synthetic_agent(index: _FakeIndex) -> _SyntheticE0Agent:
    agent = object.__new__(_SyntheticE0Agent)
    agent._e0_encoder = object()
    agent._e0_index = index
    agent._e0_catalog_ids = _catalog()
    agent._c200_last_capture = None
    return agent


def test_agent_shadow_uses_dense_order_but_never_changes_served_top10() -> None:
    c200 = _pool(100)
    dense = tuple(_identifier(index) for index in range(50, 450))
    index = _FakeIndex(dense)
    agent = _synthetic_agent(index)
    rankings = {"broad": list(_pool(120)), "final": list(c200)}
    rowids = {value: index for index, value in enumerate(_pool(500))}
    ranking_before = {key: tuple(value) for key, value in rankings.items()}
    rowids_before = dict(rowids)

    served, diagnostics = agent._apply_p11(
        object(), rankings, rowids, ["blue", "walking", "jacket"]
    )

    assert tuple(served) == c200
    assert tuple(served[: worker.TOP_K]) == c200[: worker.TOP_K]
    assert diagnostics == {"configured_mode": "control", "output_changed": False}
    assert {key: tuple(value) for key, value in rankings.items()} == ranking_before
    assert rowids == rowids_before
    assert index.calls == [
        ("blue walking jacket", agent._e0_encoder, worker.DENSE_DEPTH)
    ]
    assert agent._c200_last_capture["e0_candidates"][:100] == c200
    assert agent._c200_last_capture["e0_candidates"][100] == _identifier(100)
    assert agent._c200_last_capture["dense_count"] == 400
    assert agent._c200_last_capture["query_empty"] is False


def test_agent_empty_query_skips_encoder_and_keeps_only_c200() -> None:
    c200 = _pool(137)
    agent = _synthetic_agent(_FakeIndex((), should_run=False))
    served, _ = agent._apply_p11(
        object(), {"final": list(c200)}, {value: 1 for value in c200}, ["", ""]
    )
    assert tuple(served) == c200
    assert agent._c200_last_capture["e0_candidates"] == c200
    assert agent._c200_last_capture["dense_count"] == 0
    assert agent._c200_last_capture["query_empty"] is True


def test_agent_rejects_duplicate_or_short_nonempty_dense_route() -> None:
    for dense in (
        tuple(_identifier(index) for index in range(399)),
        (*tuple(_identifier(index) for index in range(399)), _identifier(0)),
    ):
        agent = _synthetic_agent(_FakeIndex(dense))
        with pytest.raises(worker.E0WorkerError):
            agent._apply_p11(
                object(),
                {"final": list(_pool(100))},
                {value: 1 for value in _pool(500)},
                ["query"],
            )


def test_canonical_trace_is_minimal_utf8_sorted_and_single_lf() -> None:
    candidates = _pool(137)
    actual = worker.canonical_trace_line(7, 3, candidates)
    assert actual == _canonical(
        {"candidates": list(candidates), "ordinal": 7, "turn": 3}
    )
    assert actual.endswith(b"\n") and not actual.endswith(b"\n\n")
    assert set(json.loads(actual)) == {"candidates", "ordinal", "turn"}


def test_canonical_trace_rejects_bool_coordinates_and_bad_candidate_shape() -> None:
    for ordinal, turn, candidates in (
        (True, 1, _pool(100)),
        (1, True, _pool(100)),
        (0, 1, _pool(100)),
        (2001, 1, _pool(100)),
        (1, 0, _pool(100)),
        (1, 11, _pool(100)),
        (1, 1, _pool(99)),
    ):
        with pytest.raises(worker.E0WorkerError):
            worker.canonical_trace_line(ordinal, turn, candidates)


def test_trace_publish_is_exclusive_and_never_truncates(tmp_path: Path) -> None:
    output = tmp_path / "trace.jsonl"
    payload = worker.canonical_trace_line(1, 1, _pool(100))
    worker._publish_trace_exclusive(output, [payload])
    assert output.read_bytes() == payload
    with pytest.raises(FileExistsError):
        worker._publish_trace_exclusive(output, [b"replacement\n"])
    assert output.read_bytes() == payload


def test_context_boundary_and_worker_receipt_are_identifier_private() -> None:
    assert worker.validate_context_turn({"message": "blue walking jacket"}) == (
        "blue walking jacket"
    )
    for invalid in (
        {"message": "blue", "target": _identifier(1)},
        {"message": "blue", "outer_fold": 1},
        {"message": "show B0ABCDEFGH"},
        {"message": ""},
    ):
        with pytest.raises(worker.E0WorkerError):
            worker.validate_context_turn(invalid)

    catalog = {_identifier(1)}
    worker._receipt_privacy_scan(
        {"candidate_recall": {"count": 3}, "trace_sha256": "a" * 64},
        catalog_ids=catalog,
    )
    for invalid in (
        {"candidates": ["redacted"]},
        {"query_terms": ["blue"]},
        {"safe": _identifier(1).lower()},
    ):
        with pytest.raises(worker.E0WorkerError):
            worker._receipt_privacy_scan(invalid, catalog_ids=catalog)


def test_worker_and_runner_schema_versions_are_identical() -> None:
    assert worker.SCHEMA_VERSION == probe.WORKER_SCHEMA_VERSION


def test_validate_trace_binds_full_c200_prefix_order_and_hash() -> None:
    records, references, catalog = _trace_fixture()
    result = probe.validate_trace_records(
        records, references, catalog, expected_records=4
    )
    expected = b"".join(
        worker.canonical_trace_line(
            int(row["ordinal"]), int(row["turn"]), row["candidates"]
        )
        for row in records
    )
    assert result.lengths == (150, 240, 400, 120)
    assert result.c200_lengths == (100, 137, 200, 120)
    assert result.canonical_trace_sha256 == hashlib.sha256(expected).hexdigest()
    assert result.canonical_trace_bytes == len(expected)


def test_validate_trace_rejects_reference_mismatch_schema_and_order_drift() -> None:
    records, references, catalog = _trace_fixture()
    prefix_loss = [dict(row) for row in records]
    changed = list(prefix_loss[1]["candidates"])
    changed[0], changed[1] = changed[1], changed[0]
    prefix_loss[1]["candidates"] = tuple(changed)
    extra = [dict(row) for row in records]
    extra[0]["dense_ms"] = 1.0
    bool_coordinate = [dict(row) for row in records]
    bool_coordinate[0]["turn"] = True
    out_of_order = [records[1], records[0], *records[2:]]

    for invalid in (prefix_loss, extra, bool_coordinate, out_of_order):
        with pytest.raises(probe.E0ProbeError):
            probe.validate_trace_records(
                invalid, references, catalog, expected_records=4
            )


def test_trace_loader_requires_canonical_bytes_and_can_drop_candidate_records(
    tmp_path: Path,
) -> None:
    records, references, catalog = _trace_fixture()
    output = tmp_path / "trace.jsonl"
    payload = _write_trace(output, records)
    plain = lambda path, **_kwargs: Path(path).resolve(strict=True)
    with patch.object(probe, "_require_plain", side_effect=plain):
        retained = probe.load_and_validate_e0_trace(
            output, references, catalog, expected_records=4
        )
        compact = probe.load_and_validate_e0_trace(
            output,
            references,
            catalog,
            expected_records=4,
            retain_records=False,
        )
    assert retained.canonical_trace_sha256 == hashlib.sha256(payload).hexdigest()
    assert compact.records == ()
    assert compact.lengths == retained.lengths

    output.write_bytes(payload + b"\n")
    with patch.object(probe, "_require_plain", side_effect=plain):
        with pytest.raises(probe.E0ProbeError):
            probe.load_and_validate_e0_trace(
                output, references, catalog, expected_records=4
            )


@pytest.mark.parametrize(
    ("rank", "expected_c200", "expected_c400"),
    [(9, True, True), (100, True, True), (199, True, True), (200, False, True), (399, False, True)],
)
def test_candidate_recall_cutoff_boundaries(
    rank: int, expected_c200: bool, expected_c400: bool
) -> None:
    target = "Z999999999"
    candidates = list(_pool())
    candidates[rank] = target
    turns = [
        {"candidates": candidates, "ordinal": 1, "turn": 1},
        *(
            {"candidates": _pool(), "ordinal": 1, "turn": turn}
            for turn in range(2, 11)
        ),
    ]
    flags = probe.candidate_recall_flags(target, 1, turns)
    assert flags[200] is expected_c200
    assert flags[400] is expected_c400


def test_variable_c200_len100_never_claims_dense_rank101_as_baseline_recall() -> None:
    target = "Z999999999"
    first_turn = [*_pool(100), target, *_pool(400)[101:]]
    turns = [
        {"candidates": first_turn, "ordinal": 1, "turn": 1},
        *(
            {"candidates": _pool(), "ordinal": 1, "turn": turn}
            for turn in range(2, 11)
        ),
    ]
    flags = probe.candidate_recall_flags(
        target,
        1,
        turns,
        baseline_lengths=[100] * 10,
    )
    assert flags[100] is False
    assert flags[200] is False
    assert flags[400] is True

    with pytest.raises(probe.E0ProbeError):
        probe.candidate_recall_flags(
            target,
            1,
            turns,
            baseline_lengths=[100] * 9,
        )


def test_aggregate_reports_only_anonymous_disjoint_recall_views() -> None:
    cutoffs = probe.CUTOFFS
    flags = [
        {cutoff: True for cutoff in cutoffs},
        {cutoff: cutoff == 400 for cutoff in cutoffs},
        {cutoff: False for cutoff in cutoffs},
        {cutoff: cutoff == 400 for cutoff in cutoffs},
    ]
    result = probe.aggregate_candidate_recall(
        flags,
        outer_fold=[0, 1, 0, 2],
        family_index=[10, 11, 10, 12],
        taxonomy=["clothing", "shoes", "clothing", "jewelry"],
    )
    assert result["all_sessions"]["c200"]["count"] == 1
    assert result["all_sessions"]["c400"]["count"] == 3
    assert result["increment"]["count"] == 2
    assert result["increment"]["outer_fold_span"] == 2
    assert result["increment"]["non_clothing_count"] == 2
    assert result["family_disjoint_audit"]["valid"] is True
    assert not ({"target", "candidates", "per_session"} & set(result))


def test_aggregate_rejects_family_cross_fold_or_integer_membership() -> None:
    flags = [{cutoff: False for cutoff in probe.CUTOFFS} for _ in range(2)]
    with pytest.raises(probe.E0ProbeError):
        probe.aggregate_candidate_recall(
            flags,
            outer_fold=[0, 1],
            family_index=[7, 7],
            taxonomy=["clothing", "clothing"],
        )
    flags[0][400] = 1
    with pytest.raises(probe.E0ProbeError):
        probe.aggregate_candidate_recall(
            flags,
            outer_fold=[0, 1],
            family_index=[7, 8],
            taxonomy=["clothing", "shoes"],
        )


def test_inflation_uses_variable_c200_denominator() -> None:
    result = probe.inflation_summary(
        [100, 200, 300, 400],
        [100, 120, 150, 200],
        trace_bytes=4_000,
    )
    assert result["candidate_cells"] == 1_000
    assert result["sealed_c200_candidate_cells"] == 570
    assert result["candidate_cell_ratio_over_c100"] == 2.5
    assert result["candidate_cell_ratio_over_c200"] == pytest.approx(1.754386)


def _entrypoint_environment(*, module: bool) -> dict[str, str]:
    environment = probe._offline_environment()
    if module:
        environment["PYTHONPATH"] = str(probe.ROOT)
    else:
        environment.pop("PYTHONPATH", None)
    return environment


def test_direct_and_module_entrypoints_import_evaluator_from_arbitrary_cwd(
    tmp_path: Path,
) -> None:
    direct, module = probe._entrypoint_check_commands()
    assert direct[:3] == [
        str(probe.EXPECTED_EXECUTABLE),
        "-B",
        str(probe.RUNNER_PATH),
    ]
    assert module[:4] == [
        str(probe.EXPECTED_EXECUTABLE),
        "-B",
        "-m",
        "scripts.probe_e0_embedding_candidate_recall",
    ]
    for command, is_module in ((direct, False), (module, True)):
        completed = subprocess.run(
            command,
            cwd=tmp_path,
            env=_entrypoint_environment(module=is_module),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr.decode(
            "utf-8", errors="replace"
        )
        value = json.loads(completed.stdout)
        assert value["status"] == "ENTRYPOINT_SELF_CHECK_PASS"
        assert value["required_module"] == "evaluator.local_evaluator"
        assert value["project_root_bootstrapped"] is True


def test_missing_evaluator_fails_closed_without_creating_receipt(tmp_path: Path) -> None:
    isolated = tmp_path / "isolated"
    scripts = isolated / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "__init__.py").write_text("", encoding="utf-8")
    copied_runner = scripts / probe.RUNNER_PATH.name
    shutil.copy2(probe.RUNNER_PATH, copied_runner)
    working = tmp_path / "arbitrary-cwd"
    working.mkdir()
    environment = probe._offline_environment()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            str(probe.EXPECTED_EXECUTABLE),
            "-B",
            str(copied_runner),
            "--entrypoint-self-check",
            "--require-module",
            "evaluator.local_evaluator",
        ],
        cwd=working,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert completed.returncode != 0
    assert not (
        isolated
        / "experiments/fast_track/small_ranker_v2_18_frozen_embedding_e0_20260831.json"
    ).exists()


def test_runner_worker_direct_and_module_commands_have_identical_arguments(
    tmp_path: Path,
) -> None:
    arguments = {
        "nonce": "a" * 32,
        "reference": tmp_path / "reference.jsonl",
        "trace": tmp_path / "trace.jsonl",
        "session_limit": 2,
    }
    direct = probe._worker_command(mode="direct", **arguments)
    module = probe._worker_command(mode="module", **arguments)
    assert direct[:3] == [
        str(probe.EXPECTED_EXECUTABLE),
        "-B",
        str(probe.WORKER_PATH),
    ]
    assert module[:4] == [
        str(probe.EXPECTED_EXECUTABLE),
        "-B",
        "-m",
        "scripts.e0_embedding_candidate_worker",
    ]
    assert direct[3:] == module[4:]
    joined = " ".join(direct)
    assert "proxy" not in joined.casefold()
    assert "label" not in joined.casefold()


def _plain_test_path(path: Path, *, directory: bool = False) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as error:
        raise probe.E0ProbeError("synthetic path is absent") from error
    if directory and not resolved.is_dir():
        raise probe.E0ProbeError("synthetic parent is not a directory")
    if not directory and not resolved.is_file():
        raise probe.E0ProbeError("synthetic input is not a file")
    return resolved


def test_receipt_is_prepared_parent_o_excl_and_never_overwrites(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "result.json"
    with patch.object(probe, "OUTPUT_PATH", missing), patch.object(
        probe, "_require_plain", side_effect=_plain_test_path
    ):
        with pytest.raises(probe.E0ProbeError):
            probe._open_receipt("a" * 40)
    assert not missing.parent.exists()

    output = tmp_path / "result.json"
    with patch.object(probe, "OUTPUT_PATH", output), patch.object(
        probe, "_require_plain", side_effect=_plain_test_path
    ):
        descriptor = probe._open_receipt("b" * 40)
        os.close(descriptor)
        before = output.read_bytes()
        assert json.loads(before)["status"] == "PENDING_ONE_SHOT_CONSUMED"
        with pytest.raises(probe.E0ProbeError):
            probe._open_receipt("b" * 40)
        assert output.read_bytes() == before


@pytest.mark.parametrize(
    "existing",
    [b"", b"partial", b'{"status":"PENDING_ONE_SHOT_CONSUMED"}\n'],
)
def test_receipt_refuses_every_existing_state(tmp_path: Path, existing: bytes) -> None:
    output = tmp_path / "result.json"
    output.write_bytes(existing)
    with patch.object(probe, "OUTPUT_PATH", output), patch.object(
        probe, "_require_plain", side_effect=_plain_test_path
    ):
        with pytest.raises(probe.E0ProbeError):
            probe._open_receipt("c" * 40)
    assert output.read_bytes() == existing


def test_receipt_flags_include_exclusive_without_truncate(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    observed: list[int] = []

    def race(_path: str, flags: int, _mode: int) -> int:
        observed.append(flags)
        raise FileExistsError("synthetic race")

    with patch.object(probe, "OUTPUT_PATH", output), patch.object(
        probe, "_require_plain", side_effect=_plain_test_path
    ), patch.object(probe.os, "open", side_effect=race):
        with pytest.raises(probe.E0ProbeError):
            probe._open_receipt("d" * 40)
    assert observed
    required = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    assert observed[0] & required == required
    assert observed[0] & os.O_TRUNC == 0


def test_pending_receipt_baseexception_is_durably_invalid(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    original = probe._write_descriptor
    calls = 0

    def interrupt_once(descriptor: int, value: object) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return original(descriptor, value)

    with patch.object(probe, "OUTPUT_PATH", output), patch.object(
        probe, "_require_plain", side_effect=_plain_test_path
    ), patch.object(probe, "_write_descriptor", side_effect=interrupt_once):
        with pytest.raises(KeyboardInterrupt):
            probe._open_receipt("e" * 40)
    invalid = json.loads(output.read_text(encoding="utf-8"))
    assert invalid["status"] == "INVALID_ONE_SHOT_CONSUMED"
    assert invalid["phase"] == "pending_receipt_write"
    assert invalid["rerun_forbidden"] is True


def test_post_receipt_baseexception_retry_seals_invalid_and_forbids_rerun(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    commit = "f" * 40
    with patch.object(probe, "OUTPUT_PATH", output), patch.object(
        probe, "_require_plain", side_effect=_plain_test_path
    ):
        descriptor = probe._open_receipt(commit)
        original = probe._write_descriptor
        calls = 0

        def interrupt_once(fd: int, value: object) -> tuple[int, str]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt
            return original(fd, value)

        with patch.object(probe, "_write_descriptor", side_effect=interrupt_once):
            probe._write_invalid_receipt(
                descriptor,
                commit,
                RuntimeError("synthetic failure"),
                phase="synthetic_post_receipt",
            )
        invalid = json.loads(output.read_text(encoding="utf-8"))
        assert invalid["status"] == "INVALID_ONE_SHOT_CONSUMED"
        assert invalid["phase"] == "synthetic_post_receipt"
        assert invalid["error_type"] == "RuntimeError"
        with pytest.raises(probe.E0ProbeError):
            probe._open_receipt(commit)


def test_failed_final_entrypoint_gate_cannot_open_receipt() -> None:
    open_receipt = patch.object(probe, "_open_receipt")
    with patch.object(probe, "preflight_only", return_value=object()), patch.object(
        probe,
        "_verify_entrypoints_before_receipt",
        side_effect=probe.E0ProbeError("synthetic missing evaluator"),
    ), open_receipt as observed:
        with pytest.raises(probe.E0ProbeError, match="missing evaluator"):
            probe.run("a" * 40)
    observed.assert_not_called()


def test_synthetic_preflight_runs_entrypoints_and_smoke_without_receipt(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    output = tmp_path / "formal-result.json"
    snapshot = (1, 2, 3)
    catalog_identity = probe.FileIdentity(
        probe.CATALOG_BYTES, probe.CATALOG_ROWS, probe.CATALOG_SHA256, snapshot
    )
    context_identity = probe.FileIdentity(
        probe.CONTEXT_BYTES, probe.CONTEXT_ROWS, probe.CONTEXT_SHA256, snapshot
    )
    references = tuple(_pool(100) for _ in range(20))
    sequence: list[str] = []

    def entrypoints() -> dict[str, object]:
        sequence.append("entrypoints")
        return {"receipt_created": False}

    def smoke(*_args: object) -> dict[str, object]:
        sequence.append("smoke")
        return {"direct_module_exact_repeat": True, "receipt_created": False}

    with patch.object(probe, "OUTPUT_PATH", output), patch.object(
        probe, "_assert_fresh_formal_outputs"
    ), patch.object(probe, "_validate_environment", return_value={"formal": True}), patch.object(
        probe,
        "_load_preregistration",
        return_value={"schema_version": "synthetic-prereg"},
    ), patch.object(probe, "_validate_git_checkpoint", return_value={"commit": commit}), patch.object(
        probe, "_load_catalog_ids", return_value=(_catalog(), catalog_identity)
    ), patch.object(probe, "_file_identity", return_value=context_identity), patch.object(
        probe,
        "_load_c200_reference",
        return_value=(references, {"replicas": [], "candidate_cells": 2_000}),
    ), patch.object(probe, "_validate_assets", return_value={"frozen": True}), patch.object(
        probe, "_verify_entrypoints_before_receipt", side_effect=entrypoints
    ), patch.object(probe, "_smoke_workers", side_effect=smoke), patch.object(
        probe, "_process_memory", return_value=(1, 1)
    ), patch.object(probe, "_open_receipt") as open_receipt:
        checked = probe.preflight_only(commit)

    assert sequence == ["entrypoints", "smoke"]
    assert checked.entrypoint_checks["receipt_created"] is False
    assert checked.smoke["receipt_created"] is False
    assert not output.exists()
    open_receipt.assert_not_called()


def test_result_privacy_allows_cutoff_aggregates_but_rejects_identity_surfaces() -> None:
    catalog_identifier = "123456789X"
    safe = {
        "candidate_recall": {
            "all_sessions": {
                "c200": {"count": 1986, "fraction": 0.993},
                "c400": {"count": 1987, "fraction": 0.9935},
            },
            "increment": {"count": 1, "outer_fold_span": 1},
        },
        "trace_sha256": "a" * 64,
    }
    assert probe._result_privacy_scan(
        safe, catalog_ids={catalog_identifier}
    ) is None

    for invalid in (
        {"candidates": ["redacted"]},
        {"target": "hidden"},
        {"per_session": [0]},
        {"safe": "B0ABCDEFGH"},
        {"safe": catalog_identifier.lower()},
        {"safe": [0] * probe.SESSION_COUNT},
    ):
        with pytest.raises(probe.E0ProbeError):
            probe._result_privacy_scan(invalid, catalog_ids={catalog_identifier})


def test_preflight_and_formal_source_order_keep_outcomes_after_closed_traces() -> None:
    preflight_source = inspect.getsource(probe.preflight_only)
    for forbidden in (
        "PROXY_PATH",
        "LABEL_PATH",
        "_open_receipt",
        "_open_proxy_after_receipt",
        "_load_fold_labels_after_traces",
    ):
        assert forbidden not in preflight_source
    assert preflight_source.index("_verify_entrypoints_before_receipt") < (
        preflight_source.index("_smoke_workers")
    )
    assert '"receipt_created": False' not in preflight_source

    run_source = inspect.getsource(probe.run)
    preflight_at = run_source.index("preflight_only")
    entrypoint_at = run_source.index("_verify_entrypoints_before_receipt")
    receipt_at = run_source.index("descriptor = _open_receipt")
    worker_at = run_source.index("_worker_pair")
    trace_at = run_source.index("load_and_validate_e0_trace")
    resource_at = run_source.index("inflation_summary")
    source_gate_at = run_source.index("_rehash_target_free")
    evaluator_at = run_source.index(
        'importlib.import_module("scripts.probe_c200_candidate_recall")'
    )
    proxy_at = run_source.index("PROXY_PATH")
    label_at = run_source.index("LABEL_PATH")
    assert (
        preflight_at
        < entrypoint_at
        < receipt_at
        < worker_at
        < trace_at
        < resource_at
        < source_gate_at
        < evaluator_at
        < proxy_at
        < label_at
    )
    assert "except BaseException" in run_source
    assert "run_evaluation" not in run_source
    assert "LocalEvaluator" not in run_source


def test_worker_source_is_target_free_and_closes_before_trace_publish() -> None:
    run_source = inspect.getsource(worker.run)
    for forbidden in (
        "PROXY_PATH",
        "LABEL_PATH",
        "ground_truth",
        "eligible_from",
        "local_evaluator",
        "run_evaluation",
        "LocalEvaluator",
    ):
        assert forbidden not in run_source
    close_gate = run_source.index("if not (agent_closed and sqlite_closed")
    publish = run_source.index("_publish_trace_exclusive")
    assert close_gate < publish


def test_cli_has_fixed_output_and_no_output_override() -> None:
    parser_source = inspect.getsource(probe._parser)
    assert '"--output"' not in parser_source and "'--output'" not in parser_source
    assert '"--entrypoint-self-check"' in parser_source
    assert '"--preflight-only"' in parser_source
    assert '"--run"' in parser_source
