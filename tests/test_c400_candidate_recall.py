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

from scripts import c400_candidate_worker as worker
from scripts import probe_c400_candidate_recall as probe


CUTOFFS = (10, 20, 50, 100, 200, 400)


def _identifier(index: int) -> str:
    return f"A{index:09d}"


def _catalog(count: int = 520) -> frozenset[str]:
    return frozenset(_identifier(index) for index in range(count))


def _pool(count: int = 400) -> tuple[str, ...]:
    return tuple(_identifier(index) for index in range(count))


def _pool_with_target(rank: int, target: str) -> tuple[str, ...]:
    values = list(_pool())
    values[rank] = target
    return tuple(values)


def _matched_candidate_rows(
    *, turn: int, c400: tuple[str, ...], c200_length: int
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one C400 row and its sealed, variable-length C200 surface."""
    return (
        {"ordinal": 1, "turn": turn, "c400": c400},
        {"ordinal": 1, "turn": turn, "c200": c400[:c200_length]},
    )


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


def _trace_fixture() -> tuple[
    list[dict[str, object]], list[tuple[str, ...]], frozenset[str]
]:
    c200_lengths = (100, 137, 200, 120)
    c400_lengths = (150, 240, 400, 120)
    frozen_c200 = [_pool(length) for length in c200_lengths]
    records = [
        {"ordinal": 1, "turn": turn, "c400": _pool(length)}
        for turn, length in zip(range(1, 5), c400_lengths, strict=True)
    ]
    return records, frozen_c200, _catalog()


def _write_canonical_trace(path: Path, records: list[dict[str, object]]) -> bytes:
    payload = b"".join(
        worker.canonical_trace_line(
            int(record["ordinal"]),
            int(record["turn"]),
            record["c400"],
        )
        for record in records
    )
    path.write_bytes(payload)
    return payload


@pytest.mark.parametrize("old_length", [100, 137, 200])
def test_append_unseen_tail_preserves_full_variable_c200_prefix(old_length: int) -> None:
    c200 = _pool(old_length)
    expanded = (
        *reversed(c200[:10]),
        *_pool(400)[old_length:],
        *_pool(400)[old_length : old_length + 5],
    )
    c200_before = tuple(c200)
    expanded_before = tuple(expanded)

    result = worker.append_unseen_tail(c200, expanded, _catalog(), limit=400)

    assert result[:old_length] == c200
    assert len(result) <= 400
    assert len(result) == len(set(result))
    assert result[old_length:] == tuple(
        value for value in _pool(400)[old_length:] if value not in c200
    )
    assert c200 == c200_before
    assert expanded == expanded_before


def test_append_unseen_tail_does_not_pad_and_stably_deduplicates_expansion() -> None:
    c200 = _pool(120)
    expanded = (
        c200[0],
        _identifier(200),
        _identifier(201),
        _identifier(200),
        c200[5],
        _identifier(202),
    )
    result = worker.append_unseen_tail(c200, expanded, _catalog(), limit=400)
    assert result == (*c200, _identifier(200), _identifier(201), _identifier(202))


def test_append_unseen_tail_caps_at_400_without_reordering_tail() -> None:
    c200 = _pool(100)
    expanded = tuple(_identifier(index) for index in range(100, 500))
    result = worker.append_unseen_tail(c200, expanded, _catalog(), limit=400)
    assert len(result) == 400
    assert result[:100] == c200
    assert result[100:] == expanded[:300]


def test_append_unseen_tail_rejects_invalid_old_or_expanded_members() -> None:
    duplicated_old = [*_pool(100)]
    duplicated_old[-1] = duplicated_old[0]
    unknown = (*_pool(100), "Z999999999")
    non_string: list[object] = [*_pool(100), 7]

    for c200, expanded in (
        (duplicated_old, _pool(200)),
        (_pool(100), unknown),
        (_pool(100), non_string),
        (_pool(99), _pool(200)),
        (_pool(201), _pool(400)),
    ):
        with pytest.raises(worker.C400WorkerError):
            worker.append_unseen_tail(c200, expanded, _catalog(), limit=400)


@pytest.mark.parametrize("length", [100, 137, 200, 399, 400])
def test_validate_c400_accepts_variable_prefix_and_catalog_membership(length: int) -> None:
    c200 = _pool(min(137, length))
    c400 = _pool(length)
    assert worker.validate_c400(c400, c200, _catalog()) == c400


def test_validate_c400_rejects_prefix_loss_reorder_duplicate_and_oversize() -> None:
    c200 = _pool(137)
    reordered = list(_pool(240))
    reordered[0], reordered[1] = reordered[1], reordered[0]
    duplicate = list(_pool(240))
    duplicate[-1] = duplicate[0]
    unknown = list(_pool(240))
    unknown[-1] = "Z999999999"

    for c400 in (reordered, duplicate, unknown, _pool(136), _pool(401)):
        with pytest.raises(worker.C400WorkerError):
            worker.validate_c400(c400, c200, _catalog())


def test_canonical_trace_line_has_only_c400_coordinates_and_one_lf() -> None:
    candidates = _pool(240)
    actual = worker.canonical_trace_line(7, 3, candidates)
    expected = (
        json.dumps(
            {"c400": list(candidates), "ordinal": 7, "turn": 3},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert actual == expected
    assert actual.endswith(b"\n") and not actual.endswith(b"\n\n")
    assert set(json.loads(actual)) == {"c400", "ordinal", "turn"}


@pytest.mark.parametrize(
    ("ordinal", "turn"),
    [(True, 1), (1, True), (1.0, 1), (1, 1.0), (0, 1), (2001, 1), (1, 0), (1, 11)],
)
def test_canonical_trace_line_rejects_noninteger_and_invalid_coordinates(
    ordinal: object, turn: object
) -> None:
    with pytest.raises(worker.C400WorkerError):
        worker.canonical_trace_line(ordinal, turn, _pool(100))


def test_context_boundary_returns_only_current_message_and_rejects_hidden_fields() -> None:
    context = _context_turn()
    assert worker.validate_context_turn(context) == context["message"]

    for invalid in (
        {key: value for key, value in context.items() if key != "query_terms"},
        {**context, "future_turns": ["secret"]},
        {**context, "target": _identifier(1)},
        {**context, "message": "bad\x00message"},
    ):
        with pytest.raises(worker.C400WorkerError):
            worker.validate_context_turn(invalid)


def test_validate_trace_records_binds_variable_c200_prefix_and_identities() -> None:
    records, frozen_c200, catalog = _trace_fixture()
    validation = probe.validate_trace_records(
        records,
        frozen_c200,
        catalog,
        expected_records=4,
    )
    expected_trace = b"".join(
        worker.canonical_trace_line(
            int(record["ordinal"]), int(record["turn"]), record["c400"]
        )
        for record in records
    )
    expected_c200 = b"".join(
        json.dumps(
            {
                "c200": list(prefix),
                "ordinal": int(record["ordinal"]),
                "turn": int(record["turn"]),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for record, prefix in zip(records, frozen_c200, strict=True)
    )

    assert isinstance(validation, probe.TraceValidation)
    assert validation.records == tuple(
        {
            "c400": tuple(record["c400"]),
            "ordinal": record["ordinal"],
            "turn": record["turn"],
        }
        for record in records
    )
    assert validation.lengths == (150, 240, 400, 120)
    assert validation.c200_lengths == (100, 137, 200, 120)
    assert validation.canonical_trace_sha256 == hashlib.sha256(expected_trace).hexdigest()
    assert validation.canonical_trace_bytes == len(expected_trace)
    assert validation.normalized_c200_sha256 == hashlib.sha256(expected_c200).hexdigest()
    assert validation.normalized_c200_bytes == len(expected_c200)


def test_validate_trace_records_rejects_prefix_schema_and_order_drift() -> None:
    records, frozen_c200, catalog = _trace_fixture()

    prefix_loss = [dict(record) for record in records]
    changed = list(prefix_loss[1]["c400"])
    changed[0], changed[1] = changed[1], changed[0]
    prefix_loss[1]["c400"] = tuple(changed)

    duplicate = [dict(record) for record in records]
    changed = list(duplicate[2]["c400"])
    changed[-1] = changed[0]
    duplicate[2]["c400"] = tuple(changed)

    extra = [dict(record) for record in records]
    extra[0]["timing"] = 1

    float_coordinate = [dict(record) for record in records]
    float_coordinate[0]["turn"] = 1.0

    out_of_order = [records[1], records[0], *records[2:]]

    for invalid in (prefix_loss, duplicate, extra, float_coordinate, out_of_order):
        with pytest.raises(probe.C400ProbeError):
            probe.validate_trace_records(
                invalid,
                frozen_c200,
                catalog,
                expected_records=4,
            )


def test_trace_loader_uses_same_validator_and_rejects_noncanonical_rows(
    tmp_path: Path,
) -> None:
    records, frozen_c200, catalog = _trace_fixture()
    path = tmp_path / "trace.jsonl"
    payload = _write_canonical_trace(path, records)
    loaded = probe.load_and_validate_c400_trace(
        path,
        frozen_c200,
        catalog,
        expected_records=4,
    )
    direct = probe.validate_trace_records(
        records,
        frozen_c200,
        catalog,
        expected_records=4,
    )
    assert loaded == direct
    assert loaded.canonical_trace_sha256 == hashlib.sha256(payload).hexdigest()
    assert path.read_bytes() == payload

    path.write_bytes(payload + b"\n")
    with pytest.raises(probe.C400ProbeError):
        probe.load_and_validate_c400_trace(
            path,
            frozen_c200,
            catalog,
            expected_records=4,
        )


def test_trace_identity_detects_one_tail_change_while_c200_identity_stays_fixed() -> None:
    records, frozen_c200, catalog = _trace_fixture()
    baseline = probe.validate_trace_records(
        records,
        frozen_c200,
        catalog,
        expected_records=4,
    )
    changed = [dict(record) for record in records]
    tail = list(changed[0]["c400"])
    tail[-1] = _identifier(500)
    changed[0]["c400"] = tuple(tail)
    candidate = probe.validate_trace_records(
        changed,
        frozen_c200,
        catalog,
        expected_records=4,
    )
    assert candidate.canonical_trace_sha256 != baseline.canonical_trace_sha256
    assert candidate.normalized_c200_sha256 == baseline.normalized_c200_sha256
    assert candidate.normalized_c200_bytes == baseline.normalized_c200_bytes


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (9, {10: True, 20: True, 50: True, 100: True, 200: True, 400: True}),
        (10, {10: False, 20: True, 50: True, 100: True, 200: True, 400: True}),
        (49, {10: False, 20: False, 50: True, 100: True, 200: True, 400: True}),
        (50, {10: False, 20: False, 50: False, 100: True, 200: True, 400: True}),
        (99, {10: False, 20: False, 50: False, 100: True, 200: True, 400: True}),
        (100, {10: False, 20: False, 50: False, 100: False, 200: True, 400: True}),
        (136, {10: False, 20: False, 50: False, 100: False, 200: True, 400: True}),
        (137, {10: False, 20: False, 50: False, 100: False, 200: False, 400: True}),
        (199, {10: False, 20: False, 50: False, 100: False, 200: False, 400: True}),
        (200, {10: False, 20: False, 50: False, 100: False, 200: False, 400: True}),
        (399, {10: False, 20: False, 50: False, 100: False, 200: False, 400: True}),
    ],
)
def test_candidate_recall_cutoff_boundaries(rank: int, expected: dict[int, bool]) -> None:
    target = "Z999999999"
    pairs = [
        _matched_candidate_rows(
            turn=1,
            c400=_pool_with_target(rank, target),
            c200_length=137,
        ),
        *[
            _matched_candidate_rows(turn=turn, c400=_pool(), c200_length=137)
            for turn in range(2, 11)
        ],
    ]
    c400_turns = [pair[0] for pair in pairs]
    sealed_c200_turns = [pair[1] for pair in pairs]
    assert probe.candidate_recall_flags(
        target,
        1,
        c400_turns,
        sealed_c200_turns=sealed_c200_turns,
    ) == expected


def test_candidate_recall_honors_eligibility_and_any_later_turn() -> None:
    target = "Z999999999"
    pairs = [
        _matched_candidate_rows(
            turn=1,
            c400=_pool_with_target(2, target),
            c200_length=120,
        ),
        _matched_candidate_rows(turn=2, c400=_pool(), c200_length=120),
        _matched_candidate_rows(
            turn=3,
            c400=_pool_with_target(150, target),
            c200_length=120,
        ),
        *[
            _matched_candidate_rows(turn=turn, c400=_pool(), c200_length=120)
            for turn in range(4, 11)
        ],
    ]
    c400_turns = [pair[0] for pair in pairs]
    sealed_c200_turns = [pair[1] for pair in pairs]
    assert probe.candidate_recall_flags(
        target,
        2,
        c400_turns,
        sealed_c200_turns=sealed_c200_turns,
    ) == {
        10: False,
        20: False,
        50: False,
        100: False,
        200: False,
        400: True,
    }
    assert probe.candidate_recall_flags(
        target,
        4,
        c400_turns,
        sealed_c200_turns=sealed_c200_turns,
    ) == {
        cutoff: False for cutoff in CUTOFFS
    }


def test_candidate_recall_rejects_missing_or_misaligned_sealed_c200_surface() -> None:
    target = "Z999999999"
    c400_row, sealed_row = _matched_candidate_rows(
        turn=1,
        c400=_pool(),
        c200_length=137,
    )
    with pytest.raises(probe.C400ProbeError):
        probe.candidate_recall_flags(
            target,
            1,
            [c400_row],
            sealed_c200_turns=[],
        )

    wrong_prefix = dict(sealed_row)
    wrong_prefix["c200"] = tuple(reversed(sealed_row["c200"]))
    with pytest.raises(probe.C400ProbeError):
        probe.candidate_recall_flags(
            target,
            1,
            [c400_row],
            sealed_c200_turns=[wrong_prefix],
        )


def test_aggregate_reports_c200_frontier_increment_and_disjoint_spans() -> None:
    flags = [
        {10: True, 20: True, 50: True, 100: True, 200: True, 400: True},
        {10: False, 20: False, 50: False, 100: False, 200: False, 400: True},
        {10: False, 20: False, 50: False, 100: False, 200: False, 400: False},
        {10: False, 20: False, 50: False, 100: False, 200: False, 400: True},
    ]
    result = probe.aggregate_candidate_recall(
        flags,
        sealed_c200_lengths=[100, 101, 137, 138, 200, 199, 120, 121],
        outer_fold=[0, 1, 0, 2],
        family_index=[10, 11, 10, 12],
        taxonomy=["clothing", "shoes", "clothing", "jewelry"],
    )

    assert {
        "all_sessions",
        "c200_absent_frontier",
        "increment",
        "by_outer_fold",
        "target_uniform",
        "by_taxonomy",
        "family_disjoint_audit",
        "sealed_c200_surface",
    } <= set(result)
    assert result["all_sessions"]["c200"]["count"] == 1
    assert result["all_sessions"]["c400"]["count"] == 3
    assert result["c200_absent_frontier"]["c400"]["count"] == 2
    assert result["increment"]["count"] == 2
    assert result["increment"]["target_cluster_count"] == 2
    assert result["increment"]["outer_fold_span"] == 2
    assert result["increment"]["taxonomy_span"] == 2
    assert result["family_disjoint_audit"]["valid"] is True
    assert result["sealed_c200_surface"]["records"] == 8
    assert result["sealed_c200_surface"]["turns_per_session"] == 2
    assert result["sealed_c200_surface"]["candidate_cells"] == 1116
    assert result["sealed_c200_surface"]["length"]["minimum"] == 100
    assert result["sealed_c200_surface"]["length"]["maximum"] == 200


def test_aggregate_rejects_family_cross_fold_and_nonboolean_flags() -> None:
    flags = [{cutoff: False for cutoff in CUTOFFS} for _ in range(2)]
    with pytest.raises(probe.C400ProbeError):
        probe.aggregate_candidate_recall(
            flags,
            sealed_c200_lengths=[100, 137],
            outer_fold=[0, 1],
            family_index=[7, 7],
            taxonomy=["clothing", "clothing"],
        )

    invalid = [dict(flags[0]), dict(flags[1])]
    invalid[0][400] = 1
    with pytest.raises(probe.C400ProbeError):
        probe.aggregate_candidate_recall(
            invalid,
            sealed_c200_lengths=[100, 137],
            outer_fold=[0, 1],
            family_index=[7, 8],
            taxonomy=["clothing", "shoes"],
        )

    for invalid_lengths in (
        [100],
        [100, 137, 200],
        [99, 137],
        [100, 201],
        [100, True],
    ):
        with pytest.raises(probe.C400ProbeError):
            probe.aggregate_candidate_recall(
                flags,
                sealed_c200_lengths=invalid_lengths,
                outer_fold=[0, 1],
                family_index=[7, 8],
                taxonomy=["clothing", "shoes"],
            )


def test_inflation_summary_uses_both_frozen_c100_and_c200_denominators() -> None:
    result = probe.inflation_summary(
        [100, 200, 300, 400],
        [100, 120, 150, 200],
        trace_bytes=4_000,
        c100_bytes=1_000,
        c200_bytes=2_000,
    )
    assert result["c400_length"]["minimum"] == 100
    assert result["c400_length"]["maximum"] == 400
    assert result["c400_length"]["mean"] == pytest.approx(250.0)
    assert result["added_over_c200"]["minimum"] == 0
    assert result["added_over_c200"]["maximum"] == 200
    assert result["added_over_c200"]["mean"] == pytest.approx(107.5)
    assert result["candidate_cell_ratio_over_c100"] == pytest.approx(2.5)
    assert result["candidate_cell_ratio_over_c200"] == pytest.approx(1000 / 570)
    assert result["trace_byte_ratio_over_c100"] == pytest.approx(4.0)
    assert result["trace_byte_ratio_over_c200"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("c400", "c200"),
    [([], []), ([99], [100]), ([401], [200]), ([100], [101]), ([100, 200], [100])],
)
def test_inflation_summary_rejects_invalid_or_misaligned_lengths(
    c400: list[int], c200: list[int]
) -> None:
    with pytest.raises(probe.C400ProbeError):
        probe.inflation_summary(
            c400,
            c200,
            trace_bytes=10,
            c100_bytes=10,
            c200_bytes=10,
        )


def test_receipt_requires_prepared_parent_and_is_durable_exclusive() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        missing = root / "missing" / "result.json"
        with patch.object(probe, "ROOT", root), patch.object(probe, "OUTPUT_PATH", missing):
            with pytest.raises(probe.C400ProbeError):
                probe._open_receipt("a" * 40)
        assert not missing.parent.exists()

        parent = root / "ready"
        parent.mkdir()
        output = parent / "result.json"
        with patch.object(probe, "ROOT", root), patch.object(probe, "OUTPUT_PATH", output):
            descriptor = probe._open_receipt("b" * 40)
            os.close(descriptor)
            before = output.read_bytes()
            assert json.loads(before)["status"] == "CONSUMED_PENDING_RERUN_FORBIDDEN"
            with pytest.raises(probe.C400ProbeError):
                probe._open_receipt("b" * 40)
            assert output.read_bytes() == before


@pytest.mark.parametrize(
    "existing",
    [
        b"",
        b"partial",
        b'{"status":"CONSUMED_PENDING_RERUN_FORBIDDEN"}\n',
        b'{"status":"INVALID_ONE_SHOT_CONSUMED"}\n',
    ],
)
def test_receipt_never_overwrites_any_existing_state(existing: bytes) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        parent = root / "ready"
        parent.mkdir()
        output = parent / "result.json"
        output.write_bytes(existing)
        with patch.object(probe, "ROOT", root), patch.object(probe, "OUTPUT_PATH", output):
            with pytest.raises(probe.C400ProbeError):
                probe._open_receipt("c" * 40)
        assert output.read_bytes() == existing


def test_receipt_uses_o_excl_without_truncate_and_rejects_escape_or_reparse() -> None:
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
            with pytest.raises(probe.C400ProbeError):
                probe._open_receipt("d" * 40)
        assert observed
        assert observed[0] & (os.O_WRONLY | os.O_CREAT | os.O_EXCL) == (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )
        assert observed[0] & os.O_TRUNC == 0

        escape = root.parent / "escape-result.json"
        with patch.object(probe, "ROOT", root), patch.object(probe, "OUTPUT_PATH", escape):
            with pytest.raises(probe.C400ProbeError):
                probe._open_receipt("d" * 40)

        with patch.object(probe, "ROOT", root), patch.object(
            probe, "OUTPUT_PATH", output
        ), patch.object(
            probe,
            "_is_link_or_reparse",
            side_effect=lambda path: Path(path) == parent,
        ):
            with pytest.raises(probe.C400ProbeError):
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
            with pytest.raises(probe.C400ProbeError):
                probe._open_receipt("e" * 40)
        invalid = json.loads(output.read_text(encoding="utf-8"))
        assert invalid["status"] == "INVALID_ONE_SHOT_CONSUMED"
        assert invalid["error_class"] == "KeyboardInterrupt"
        assert invalid["rerun_forbidden"] is True


def test_post_receipt_invalid_seal_retries_transient_baseexception() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        parent = root / "ready"
        parent.mkdir()
        output = parent / "result.json"
        commit = "f" * 40
        with patch.object(probe, "ROOT", root), patch.object(
            probe, "OUTPUT_PATH", output
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
            assert invalid["error_class"] == "RuntimeError"
            assert invalid["phase"] == "synthetic_post_receipt"
            with pytest.raises(probe.C400ProbeError):
                probe._open_receipt(commit)


def test_worker_receipt_privacy_covers_non_b0_catalog_exceptions() -> None:
    exceptions = frozenset(
        {
            "123456789X",
            "223456789X",
            "323456789X",
            "423456789X",
            "523456789X",
            "623456789X",
        }
    )
    safe = {"trace_sha256": "a" * 64, "record_count": 20_000}
    assert worker._receipt_privacy_scan(
        safe,
        non_b0_catalog_ids=exceptions,
    ) is None

    for payload in (
        {"safe": "123456789X"},
        {"safe": "123456789x"},
        {"nested": {"123456789X": {"count": 1}}},
        {"safe": "B0ABCDEFGH"},
    ):
        with pytest.raises(worker.C400WorkerError):
            worker._receipt_privacy_scan(
                payload,
                non_b0_catalog_ids=exceptions,
            )


def test_result_privacy_accepts_aggregates_and_rejects_identifiers_or_vectors() -> None:
    non_b0_identifier = "123456789X"
    catalog = frozenset({non_b0_identifier})
    safe = {
        "target_uniform": {"c400": {"count": 3, "fraction": 0.75}},
        "increment": {"target_cluster_count": 2, "outer_fold_span": 2},
        "by_taxonomy": {"clothing": {"count": 1}},
        "family_disjoint_audit": {"valid": True},
        "source_hashes": {"trace": "a" * 64},
    }
    assert probe._result_privacy_scan(safe, catalog_ids=catalog) is None

    forbidden = (
        {"session_id": 1},
        {"nested": {"sample_id": 1}},
        {"target": "hidden"},
        {"eligible_from": 1},
        {"message": "visible"},
        {"safe": "B0ABCDEFGH"},
        {"safe": non_b0_identifier.lower()},
        {"by_taxonomy": {non_b0_identifier: {"count": 1}}},
        {"safe": np.zeros(1, dtype=np.uint8)},
        {"safe": [0] * 2000},
    )
    for payload in forbidden:
        with pytest.raises(probe.C400ProbeError):
            probe._result_privacy_scan(payload, catalog_ids=catalog)


def test_formal_source_order_has_no_c200_rerun_context_rebuild_or_full_evaluator() -> None:
    preflight_source = inspect.getsource(probe.preflight_only)
    for forbidden in ("PROXY_PATH", "LABEL_PATH", "ground_truth", "eligible_from"):
        assert forbidden not in preflight_source

    run_source = inspect.getsource(probe.run)
    receipt_at = run_source.index("_open_receipt")
    worker_at = run_source.index("_run_one_worker")
    trace_at = run_source.index("load_and_validate_c400_trace")
    proxy_at = run_source.index("PROXY_PATH")
    label_at = run_source.index("LABEL_PATH")
    assert receipt_at < worker_at < trace_at < proxy_at <= label_at
    assert "except BaseException" in run_source
    for forbidden in (
        "probe_c200_candidate_recall.run(",
        "c200_candidate_worker.py",
        "materialize_visible_context",
        "run_evaluation",
        "LocalEvaluator",
    ):
        assert forbidden not in run_source

    parser_source = inspect.getsource(probe._parser)
    assert '"--output"' not in parser_source and "'--output'" not in parser_source

    worker_source = inspect.getsource(worker.run)
    assert "PROXY_PATH" not in worker_source
    assert "LABEL_PATH" not in worker_source
    assert "ground_truth" not in worker_source
    assert "eligible_from" not in worker_source
    for forbidden in ("local_evaluator", "run_evaluation", "LocalEvaluator"):
        assert forbidden not in worker_source
    assert worker_source.index("agent.close()") < worker_source.index(
        "_publish_trace_exclusive"
    )


def test_worker_adds_only_limit320_broad_and_preserves_production_surfaces() -> None:
    assert worker.PRODUCTION_BROAD_LIMIT == 120
    assert worker.DIAGNOSTIC_BROAD_LIMIT == 320
    assert worker.STRICT_LIMIT == 80
    assert worker.MAX_CANDIDATES == 400

    expanded_sql = worker.EXPANDED_BROAD_SQL.upper()
    assert expanded_sql.count("LIMIT 320") == 1
    assert "LIMIT 120" not in expanded_sql
    assert "LIMIT 80" not in expanded_sql

    expanded_source = inspect.getsource(worker.C400CaptureAgent._expanded_r08)
    assert expanded_source.count(".execute(") == 1
    assert 'rankings.get("broad"' in expanded_source
    assert 'rankings.get("strict"' in expanded_source
    for forbidden in ("strict_expression", "STRICT_SQL", "LIMIT 80"):
        assert forbidden not in expanded_source

    apply_source = inspect.getsource(worker.C400CaptureAgent._apply_p11)
    snapshot_at = apply_source.index("ranking_snapshot")
    expansion_at = apply_source.index("self._expanded_r08")
    production_at = apply_source.index("super()._apply_p11")
    assert snapshot_at < expansion_at < production_at
    assert "append_unseen_tail(" in apply_source
    assert "production_final[:200]" not in apply_source
    assert "!= ranking_snapshot" in apply_source
    assert "candidate_rowids != rowid_snapshot" in apply_source
    assert "tuple(served) != production_final" in apply_source

    run_source = inspect.getsource(worker.run)
    assert "_served_identifiers(response) != sealed_c200[:TOP_K]" in run_source
