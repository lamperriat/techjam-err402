from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from scripts import c200_candidate_worker as worker
from scripts import probe_c200_candidate_recall as probe


CUTOFFS = (10, 20, 50, 100, 200)


def _catalog(count: int = 240) -> frozenset[str]:
    return frozenset(f"item-{index:03d}" for index in range(count))


def _pool(count: int = 200) -> tuple[str, ...]:
    return tuple(f"item-{index:03d}" for index in range(count))


def _pool_with_target(rank: int, target: str) -> tuple[str, ...]:
    values = list(_pool())
    values[rank] = target
    return tuple(values)


def _context_turn() -> dict[str, object]:
    return {
        "message": "I need a blue walking jacket.",
        "goal_messages": ["I need a blue walking jacket."],
        "category_text": "jacket",
        "active_terms": ["blue", "walking"],
        "excluded_terms": [],
        "query_terms": ["jacket", "blue", "walking"],
        "version": 1,
        "version_anchor_turn": 1,
        "override_count": 0,
        "current_turn_override": False,
        "active_records": [],
        "retired_records": [],
        "hard_clause_terms": [],
        "budget_upper": None,
    }


def _trace_fixture() -> tuple[list[dict[str, object]], list[tuple[str, ...]], frozenset[str]]:
    c100 = _pool(100)
    c200 = _pool(120)
    records = [
        {"ordinal": 1, "turn": turn, "c200": c200}
        for turn in (1, 2, 3, 4)
    ]
    frozen = [c100 for _ in records]
    return records, frozen, _catalog()


def _write_canonical_trace(path: Path, records: list[dict[str, object]]) -> bytes:
    payload = b"".join(
        worker.canonical_trace_line(
            int(record["ordinal"]),
            int(record["turn"]),
            record["c200"],
        )
        for record in records
    )
    path.write_bytes(payload)
    return payload


def _worker_receipt_payload() -> dict[str, object]:
    return {
        "kind": "receipt",
        "nonce": "a" * 32,
        "trace_sha256": "b" * 64,
        "trace_bytes": 1,
        "record_count": 20_000,
        "summary": {
            "schema_version": worker.SCHEMA_VERSION,
            "environment": {
                "executable": str(probe.EXPECTED_EXECUTABLE.resolve()),
                "python": probe.EXPECTED_PYTHON,
                "sqlite": probe.EXPECTED_SQLITE,
                "pythonhashseed": "0",
                "network_attempt_count": 0,
                "gpu_used": False,
                "gpu_peak_bytes": 0,
            },
            "configuration": {
                "p11_mode": "control",
                "small_ranker_mode": "off",
                "question_policy": "fast",
                "rerank_mode": "off",
                "retrieval_mode": "coverage",
            },
            "pool_lengths": {
                "min": 100,
                "p50": 150,
                "p95": 200,
                "max": 200,
                "mean": 150.0,
                "records": 20_000,
                "candidate_cells": 3_000_000,
            },
            "latency": {
                "respond": {
                    "count": 20_000,
                    "p50_milliseconds": 1.0,
                    "p95_milliseconds": 2.0,
                    "maximum_milliseconds": 3.0,
                },
                "capture": {
                    "count": 20_000,
                    "p50_microseconds": 1.0,
                    "p95_microseconds": 2.0,
                    "maximum_microseconds": 3.0,
                },
            },
            "resources": {"peak_working_set_bytes": 1},
            "lifecycle": {
                "agent_closed_before_trace_publish": True,
                "sqlite_closed_before_trace_publish": True,
            },
        },
    }


def _canonical_payload(value: object) -> bytes:
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


@pytest.mark.parametrize("count", [100, 137, 200])
def test_validate_c200_preserves_order_without_padding(count: int) -> None:
    candidates = _pool(count)
    assert worker.validate_c200(candidates, _catalog()) == candidates


@pytest.mark.parametrize("count", [0, 99, 201])
def test_validate_c200_rejects_out_of_contract_lengths(count: int) -> None:
    with pytest.raises(worker.C200WorkerError):
        worker.validate_c200(_pool(count), _catalog())


def test_validate_c200_rejects_duplicate_empty_nonstring_and_unknown_members() -> None:
    duplicate = list(_pool(100))
    duplicate[-1] = duplicate[0]
    unknown = list(_pool(100))
    unknown[-1] = "not-in-catalog"
    empty = list(_pool(100))
    empty[-1] = ""
    mixed: list[object] = list(_pool(100))
    mixed[-1] = 7

    for candidates, catalog in (
        (duplicate, _catalog()),
        (unknown, _catalog()),
        (empty, frozenset((*_catalog(), ""))),
        (mixed, frozenset((*_catalog(), 7))),
    ):
        with pytest.raises(worker.C200WorkerError):
            worker.validate_c200(candidates, catalog)


def test_canonical_trace_line_has_exact_schema_utf8_and_one_lf() -> None:
    candidates = _pool(100)
    actual = worker.canonical_trace_line(7, 3, candidates)
    expected = (
        json.dumps(
            {"ordinal": 7, "turn": 3, "c200": list(candidates)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert actual == expected
    assert actual.endswith(b"\n") and not actual.endswith(b"\n\n")
    assert set(json.loads(actual)) == {"ordinal", "turn", "c200"}


@pytest.mark.parametrize(
    ("ordinal", "turn"),
    [
        (True, 1),
        (1, True),
        (1.0, 1),
        (1, 1.0),
        (0, 1),
        (2001, 1),
        (1, 0),
        (1, 11),
    ],
)
def test_canonical_trace_line_rejects_bool_and_invalid_coordinates(
    ordinal: object, turn: object
) -> None:
    with pytest.raises(worker.C200WorkerError):
        worker.canonical_trace_line(ordinal, turn, _pool(100))


def test_actual_agent_hook_captures_pre_p11_r08_without_changing_served_order() -> None:
    candidates = list(_pool(120))
    candidate_rowids = {identifier: index for index, identifier in enumerate(candidates)}
    state = object()
    agent = object.__new__(worker.C200CaptureAgent)
    agent._c200_last_capture = None
    diagnostics = {"configured_mode": "control", "output_changed": False}

    with patch.object(
        worker.Agent,
        "_apply_p11",
        return_value=(list(candidates), diagnostics),
    ):
        served, observed = worker.C200CaptureAgent._apply_p11(
            agent,
            state,
            {"final": candidates},
            candidate_rowids,
            ["blue", "jacket"],
        )

    assert served == candidates
    assert observed == diagnostics
    assert agent._c200_last_capture["c200"] == tuple(candidates)
    assert agent._c200_last_capture["state_identity"] == id(state)


def test_worker_trace_publish_is_exclusive_and_never_truncates(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    payload = worker.canonical_trace_line(1, 1, _pool(100))
    worker._publish_trace_exclusive(path, [payload])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        worker._publish_trace_exclusive(path, [b"replacement\n"])
    assert before == payload
    assert path.read_bytes() == before


def test_context_turn_full_schema_returns_only_current_message() -> None:
    context = _context_turn()
    assert worker.validate_context_turn(context) == context["message"]


def test_context_turn_rejects_missing_extra_forbidden_identifier_and_control_data() -> None:
    missing = _context_turn()
    missing.pop("query_terms")
    extra = {**_context_turn(), "extra": 1}
    forbidden = {**_context_turn(), "target": "secret"}
    identifier = {**_context_turn(), "message": "show B0ABCDEFGH please"}
    control = {**_context_turn(), "message": "bad\x00message"}

    for value in (missing, extra, forbidden, identifier, control):
        with pytest.raises(worker.C200WorkerError):
            worker.validate_context_turn(value)


def test_context_turn_rejects_exact_non_b0_catalog_token_without_false_positive() -> None:
    catalog_identifier = "123456789X"
    leaked = {**_context_turn(), "message": f"show {catalog_identifier} please"}
    with pytest.raises(worker.C200WorkerError):
        worker._validate_context_turn_payload(
            leaked,
            frozenset({catalog_identifier}),
        )

    benign = {**_context_turn(), "message": "I need waterproof boots."}
    assert (
        worker._validate_context_turn_payload(
            benign,
            frozenset({catalog_identifier}),
        )["message"]
        == benign["message"]
    )


@pytest.mark.parametrize("name", ["version", "version_anchor_turn", "override_count"])
def test_context_turn_rejects_bool_as_integer(name: str) -> None:
    context = _context_turn()
    context[name] = True
    with pytest.raises(worker.C200WorkerError):
        worker.validate_context_turn(context)


def test_validate_trace_records_binds_order_schema_prefix_catalog_and_identity() -> None:
    records, frozen, catalog = _trace_fixture()
    validation = probe.validate_trace_records(
        records,
        frozen,
        catalog,
        expected_records=4,
    )
    expected_trace = b"".join(
        worker.canonical_trace_line(
            int(record["ordinal"]), int(record["turn"]), record["c200"]
        )
        for record in records
    )
    expected_c100 = b"".join(
        json.dumps(
            {
                "c100": list(prefix),
                "ordinal": int(record["ordinal"]),
                "turn": int(record["turn"]),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for record, prefix in zip(records, frozen)
    )

    assert isinstance(validation, probe.TraceValidation)
    assert len(validation.records) == 4
    assert tuple(validation.lengths) == (120, 120, 120, 120)
    assert validation.canonical_trace_sha256 == hashlib.sha256(expected_trace).hexdigest()
    assert validation.canonical_trace_bytes == len(expected_trace)
    assert validation.normalized_c100_sha256 == hashlib.sha256(expected_c100).hexdigest()
    assert validation.normalized_c100_bytes == len(expected_c100)


def test_validate_trace_records_rejects_prefix_loss_reorder_and_duplicate() -> None:
    records, frozen, catalog = _trace_fixture()
    changed = [dict(record) for record in records]
    pool = list(changed[0]["c200"])
    pool[0], pool[1] = pool[1], pool[0]
    changed[0]["c200"] = tuple(pool)
    with pytest.raises(probe.C200ProbeError):
        probe.validate_trace_records(changed, frozen, catalog, expected_records=4)

    duplicate = [dict(record) for record in records]
    pool = list(duplicate[0]["c200"])
    pool[100] = pool[0]
    duplicate[0]["c200"] = tuple(pool)
    with pytest.raises(probe.C200ProbeError):
        probe.validate_trace_records(duplicate, frozen, catalog, expected_records=4)


def test_validate_trace_records_rejects_schema_bool_and_row_order_drift() -> None:
    records, frozen, catalog = _trace_fixture()

    extra = [dict(record) for record in records]
    extra[0]["timing"] = 1
    with pytest.raises(probe.C200ProbeError):
        probe.validate_trace_records(extra, frozen, catalog, expected_records=4)

    bool_coordinate = [dict(record) for record in records]
    bool_coordinate[0]["turn"] = True
    with pytest.raises(probe.C200ProbeError):
        probe.validate_trace_records(
            bool_coordinate, frozen, catalog, expected_records=4
        )

    out_of_order = [records[1], records[0], *records[2:]]
    with pytest.raises(probe.C200ProbeError):
        probe.validate_trace_records(out_of_order, frozen, catalog, expected_records=4)

    float_coordinate = [dict(record) for record in records]
    float_coordinate[0]["ordinal"] = 1.0
    with pytest.raises(probe.C200ProbeError):
        probe.validate_trace_records(
            float_coordinate, frozen, catalog, expected_records=4
        )


def test_worker_receipt_rejects_equal_but_wrong_json_scalar_types() -> None:
    valid = _worker_receipt_payload()
    assert probe._validate_worker_receipt(
        _canonical_payload(valid), nonce="a" * 32
    )["record_count"] == 20_000

    wrong_record_count = _worker_receipt_payload()
    wrong_record_count["record_count"] = 20_000.0
    wrong_lifecycle = _worker_receipt_payload()
    wrong_lifecycle["summary"]["lifecycle"][
        "agent_closed_before_trace_publish"
    ] = 1
    wrong_network = _worker_receipt_payload()
    wrong_network["summary"]["environment"]["network_attempt_count"] = False
    for value in (wrong_record_count, wrong_lifecycle, wrong_network):
        with pytest.raises(probe.C200ProbeError):
            probe._validate_worker_receipt(
                _canonical_payload(value), nonce="a" * 32
            )


def test_trace_loader_delegates_to_same_validator_and_rejects_noncanonical_rows(
    tmp_path: Path,
) -> None:
    records, frozen, catalog = _trace_fixture()
    path = tmp_path / "trace.jsonl"
    payload = _write_canonical_trace(path, records)
    loaded = probe.load_and_validate_c200_trace(
        path,
        frozen,
        catalog,
        expected_records=4,
    )
    direct = probe.validate_trace_records(
        records,
        frozen,
        catalog,
        expected_records=4,
    )
    assert loaded.canonical_trace_sha256 == direct.canonical_trace_sha256
    assert loaded.canonical_trace_sha256 == hashlib.sha256(payload).hexdigest()
    assert loaded.records == direct.records

    path.write_bytes(payload + b"\n")
    with pytest.raises(probe.C200ProbeError):
        probe.load_and_validate_c200_trace(
            path,
            frozen,
            catalog,
            expected_records=4,
        )


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (9, {10: True, 20: True, 50: True, 100: True, 200: True}),
        (10, {10: False, 20: True, 50: True, 100: True, 200: True}),
        (49, {10: False, 20: False, 50: True, 100: True, 200: True}),
        (50, {10: False, 20: False, 50: False, 100: True, 200: True}),
        (99, {10: False, 20: False, 50: False, 100: True, 200: True}),
        (100, {10: False, 20: False, 50: False, 100: False, 200: True}),
        (199, {10: False, 20: False, 50: False, 100: False, 200: True}),
    ],
)
def test_candidate_recall_cutoff_boundaries(rank: int, expected: dict[int, bool]) -> None:
    target = "target-item"
    turns = [
        {"ordinal": 1, "turn": 1, "c200": _pool_with_target(rank, target)},
        *[
            {"ordinal": 1, "turn": turn, "c200": _pool()}
            for turn in range(2, 11)
        ],
    ]
    assert probe.candidate_recall_flags(target, 1, turns) == expected


def test_candidate_recall_honors_eligible_turn_and_any_later_turn() -> None:
    target = "target-item"
    preeligible = _pool_with_target(2, target)
    absent = _pool()
    c200_only = _pool_with_target(150, target)
    turns = [
        {"ordinal": 1, "turn": 1, "c200": preeligible},
        {"ordinal": 1, "turn": 2, "c200": absent},
        {"ordinal": 1, "turn": 3, "c200": c200_only},
        *[
            {"ordinal": 1, "turn": turn, "c200": absent}
            for turn in range(4, 11)
        ],
    ]
    assert probe.candidate_recall_flags(target, 2, turns) == {
        10: False,
        20: False,
        50: False,
        100: False,
        200: True,
    }
    assert probe.candidate_recall_flags(target, 4, turns) == {
        cutoff: False for cutoff in CUTOFFS
    }


def test_aggregate_candidate_recall_reports_frontier_partitions_and_spans() -> None:
    flags = [
        {10: True, 20: True, 50: True, 100: True, 200: True},
        {10: False, 20: False, 50: False, 100: False, 200: True},
        {10: False, 20: False, 50: False, 100: False, 200: False},
        {10: False, 20: False, 50: False, 100: False, 200: True},
    ]
    result = probe.aggregate_candidate_recall(
        flags,
        outer_fold=[0, 1, 0, 2],
        family_index=[10, 11, 10, 12],
        taxonomy=["clothing", "shoes", "clothing", "jewelry"],
    )

    assert {
        "all_sessions",
        "c100_absent_frontier",
        "increment",
        "by_outer_fold",
        "target_uniform",
        "by_taxonomy",
        "family_disjoint_audit",
    } <= set(result)
    assert result["all_sessions"]["c100"]["count"] == 1
    assert result["all_sessions"]["c200"]["count"] == 3
    assert result["c100_absent_frontier"]["c200"]["count"] == 2
    assert result["increment"]["count"] == 2
    assert result["increment"]["target_cluster_count"] == 2
    assert result["increment"]["outer_fold_span"] == 2
    assert result["increment"]["taxonomy_span"] == 2
    assert result["family_disjoint_audit"]["valid"] is True


def test_aggregate_candidate_recall_rejects_family_cross_fold_leakage() -> None:
    flags = [{cutoff: False for cutoff in CUTOFFS} for _ in range(2)]
    with pytest.raises(probe.C200ProbeError):
        probe.aggregate_candidate_recall(
            flags,
            outer_fold=[0, 1],
            family_index=[7, 7],
            taxonomy=["clothing", "clothing"],
        )


def test_inflation_summary_uses_candidate_cells_and_frozen_reference_bytes() -> None:
    result = probe.inflation_summary(
        [100, 130, 150, 200],
        trace_bytes=1_800,
        reference_bytes=1_000,
    )
    assert result["c200_length"]["minimum"] == 100
    assert result["c200_length"]["maximum"] == 200
    assert result["c200_length"]["mean"] == pytest.approx(145.0)
    assert result["added_candidates"]["minimum"] == 0
    assert result["added_candidates"]["maximum"] == 100
    assert result["added_candidates"]["mean"] == pytest.approx(45.0)
    assert result["candidate_cell_ratio"] == pytest.approx(1.45)
    assert result["trace_byte_ratio"] == pytest.approx(1.8)


@pytest.mark.parametrize("lengths", [[], [99], [201]])
def test_inflation_summary_rejects_invalid_lengths(lengths: list[int]) -> None:
    with pytest.raises(probe.C200ProbeError):
        probe.inflation_summary(lengths, trace_bytes=10, reference_bytes=10)


def test_receipt_requires_prepared_parent_and_is_durable_exclusive() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        missing = root / "missing" / "result.json"
        with patch.object(probe, "ROOT", root), patch.object(
            probe, "OUTPUT_PATH", missing
        ):
            with pytest.raises(probe.C200ProbeError):
                probe._open_receipt("a" * 40)
        assert not missing.parent.exists()

        parent = root / "ready"
        parent.mkdir()
        output = parent / "result.json"
        with patch.object(probe, "ROOT", root), patch.object(
            probe, "OUTPUT_PATH", output
        ):
            descriptor = probe._open_receipt("b" * 40)
            os.close(descriptor)
            before = output.read_bytes()
            assert json.loads(before)["status"] == "CONSUMED_PENDING_RERUN_FORBIDDEN"
            with pytest.raises(probe.C200ProbeError):
                probe._open_receipt("b" * 40)
            assert output.read_bytes() == before


@pytest.mark.parametrize(
    "existing",
    [b"", b"partial", b'{"status":"CONSUMED_PENDING_RERUN_FORBIDDEN"}\n', b'{"status":"INVALID_ONE_SHOT_CONSUMED"}\n'],
)
def test_receipt_existing_states_are_never_overwritten(existing: bytes) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        parent = root / "ready"
        parent.mkdir()
        output = parent / "result.json"
        output.write_bytes(existing)
        with patch.object(probe, "ROOT", root), patch.object(
            probe, "OUTPUT_PATH", output
        ):
            with pytest.raises(probe.C200ProbeError):
                probe._open_receipt("c" * 40)
        assert output.read_bytes() == existing


def test_receipt_uses_o_excl_without_truncate_and_rejects_escape() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        parent = root / "ready"
        parent.mkdir()
        output = parent / "result.json"
        observed: list[int] = []

        def race(_path: str, flags: int, _mode: int) -> int:
            observed.append(flags)
            raise FileExistsError("race")

        with patch.object(probe, "ROOT", root), patch.object(
            probe, "OUTPUT_PATH", output
        ), patch.object(probe.os, "open", side_effect=race):
            with pytest.raises(probe.C200ProbeError):
                probe._open_receipt("d" * 40)
        assert observed
        assert observed[0] & (os.O_WRONLY | os.O_CREAT | os.O_EXCL) == (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )
        assert observed[0] & os.O_TRUNC == 0

        escape = root.parent / "escape-result.json"
        with patch.object(probe, "ROOT", root), patch.object(
            probe, "OUTPUT_PATH", escape
        ):
            with pytest.raises(probe.C200ProbeError):
                probe._open_receipt("d" * 40)


def test_receipt_baseexception_is_durably_invalid_and_cannot_rerun() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        parent = root / "ready"
        parent.mkdir()
        output = parent / "result.json"
        original = probe._write_descriptor
        calls = 0

        def interrupt_once(descriptor: int, value: object) -> tuple[int, str]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt
            return original(descriptor, value)

        with patch.object(probe, "ROOT", root), patch.object(
            probe, "OUTPUT_PATH", output
        ), patch.object(probe, "_write_descriptor", side_effect=interrupt_once):
            with pytest.raises(probe.C200ProbeError):
                probe._open_receipt("e" * 40)
        invalid = json.loads(output.read_text(encoding="utf-8"))
        assert invalid["status"] == "INVALID_ONE_SHOT_CONSUMED"
        assert invalid["error_class"] == "KeyboardInterrupt"
        assert invalid["rerun_forbidden"] is True


def test_invalid_receipt_retries_one_interrupted_seal_on_same_descriptor() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "invalid.json"
        descriptor = os.open(
            str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
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
                "f" * 40,
                RuntimeError("synthetic"),
                phase="synthetic_failure",
            )
        invalid = json.loads(output.read_text(encoding="utf-8"))
        assert calls == 2
        assert invalid["status"] == "INVALID_ONE_SHOT_CONSUMED"
        assert invalid["error_class"] == "RuntimeError"


def test_result_privacy_allows_required_aggregates_and_rejects_identity_data() -> None:
    safe = {
        "target_uniform": {"c200": {"count": 3, "fraction": 0.75}},
        "increment": {"target_cluster_count": 2, "outer_fold_span": 2},
        "by_taxonomy": {"clothing": {"count": 1}},
        "family_disjoint_audit": {"valid": True},
        "source_hashes": {"trace": "a" * 64},
    }
    assert probe._result_privacy_scan(safe) is None

    forbidden = (
        {"session_id": 1},
        {"nested": {"sample_id": 1}},
        {"product_id": "x"},
        {"target": "x"},
        {"eligible_from": 1},
        {"message": "visible"},
        {"safe": "B0ABCDEFGH"},
        {"by_taxonomy": {"B0ABCDEFGH": {"count": 1}}},
        {"safe": np.zeros(1, dtype=np.uint8)},
        {"safe": [0] * 2000},
    )
    for payload in forbidden:
        with pytest.raises(probe.C200ProbeError):
            probe._result_privacy_scan(payload)

    for payload in (
        {"safe": "123456789X"},
        {"by_taxonomy": {"123456789X": {"count": 1}}},
    ):
        with pytest.raises(probe.C200ProbeError):
            probe._result_privacy_scan(
                payload,
                catalog_ids={"123456789X"},
            )


def test_formal_source_orders_receipt_workers_and_target_attach() -> None:
    preflight_source = inspect.getsource(probe.preflight_only)
    for forbidden in ("PROXY_PATH", "LABEL_PATH", "ground_truth", "eligible_from"):
        assert forbidden not in preflight_source

    run_source = inspect.getsource(probe.run)
    receipt_at = run_source.index("_open_receipt")
    proxy_at = run_source.index("PROXY_PATH")
    worker_trace_at = run_source.index("load_and_validate_c200_trace")
    label_at = run_source.index("LABEL_PATH")
    assert receipt_at < proxy_at
    assert worker_trace_at < label_at
    assert "except BaseException" in run_source

    worker_source = inspect.getsource(worker.run)
    close_at = worker_source.index("agent.close()")
    publish_at = worker_source.index("_publish_trace_exclusive")
    assert close_at < publish_at
