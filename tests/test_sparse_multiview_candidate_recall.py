from __future__ import annotations

import inspect
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import probe_sparse_multiview_candidate_recall as probe
from scripts import sparse_multiview_candidate_worker as worker
from starter import sparse_multiview as sparse
from starter.attributes import AttributeValue, ProductAttributeView
from starter.slot_ledger import ACTIVE, SUPERSEDED


def _identifier(index: int) -> str:
    return f"A{index:09d}"


def _pool(count: int, *, start: int = 0) -> tuple[str, ...]:
    return tuple(_identifier(index) for index in range(start, start + count))


def _record(
    slot: str,
    value: str,
    *,
    polarity: int = 1,
    hardness: str = "hard",
    source_turn: int = 1,
    version: int = 2,
    status: str = ACTIVE,
) -> dict[str, object]:
    return {
        "slot": slot,
        "value": value,
        "polarity": polarity,
        "hardness": hardness,
        "source_turn": source_turn,
        "version": version,
        "status": status,
    }


def _attribute(
    value: str,
    *,
    source: str = "features",
    confidence: float = 1.0,
) -> AttributeValue:
    return AttributeValue(value=value, source=source, confidence=confidence, raw=value)


def _flag_row(*, c200: bool, c320: bool) -> dict[int, bool]:
    return {
        cutoff: (c320 if cutoff == 320 else c200)
        for cutoff in probe.CUTOFFS
    }


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


def test_registry_rewrite_is_deterministic_field_isolated_and_current_version_only() -> None:
    records = [
        _record("material", "linen"),
        _record("style", "casual"),
        _record("use_case", "wedding", version=1),
        _record("material", "silk", status=SUPERSEDED),
        _record("color", "red"),
        _record("material", "wool", polarity=-1),
    ]

    first = sparse.build_rewrite_query(
        category_text="Dresses",
        active_terms=["red", "Acme brand", "$50"],
        excluded_terms=["wool"],
        current_version=2,
        records=records,
    )
    second = sparse.build_rewrite_query(
        category_text="Dresses",
        active_terms=reversed(["red", "Acme brand", "$50"]),
        excluded_terms=["wool"],
        current_version=2,
        records=reversed(records),
    )

    assert first == second
    assert first.activated is True
    assert first.category_values == ("dress",)
    assert first.attribute_values == (("material", "linen"), ("style", "casual"))
    assert first.category_terms == ("dress", "dresses")
    assert first.attribute_terms == ("casual", "linen")
    assert first.expression == (
        '({title categories} : ("dress" OR "dresses")) AND '
        '({title features details description} : ("casual" OR "linen"))'
    )
    assert "Acme" not in first.expression
    assert "$50" not in first.expression
    assert "wool" not in first.expression


@pytest.mark.parametrize(
    ("category", "records"),
    [
        ("unregistered widget", [_record("material", "linen")]),
        ("dress", []),
        ("dress", [_record("color", "red")]),
        (
            "dress",
            [
                _record("brand", "Acme"),
                _record("price", "50"),
                _record("feature", "waterproof"),
            ],
        ),
        ("dress", [_record("material", "linen", version=1)]),
        ("dress", [_record("material", "linen", status=SUPERSEDED)]),
    ],
)
def test_registry_rewrite_fails_closed_without_both_frozen_clauses(
    category: str, records: list[dict[str, object]]
) -> None:
    query = sparse.build_rewrite_query(
        category_text=category,
        active_terms=["linen", "waterproof"],
        excluded_terms=[],
        current_version=2,
        records=records,
    )

    assert query.activated is False
    assert query.expression == ""
    if category == "unregistered widget":
        assert not query.category_terms
        assert query.attribute_terms == ("linen",)
    else:
        assert query.category_terms
        assert not query.attribute_terms


def test_hard_rules_use_only_visible_current_active_hard_constraints() -> None:
    records = [
        _record("material", "linen"),
        _record("audience", "women"),
        _record("color", "red", polarity=-1),
        _record("material", "silk", polarity=-1, hardness="soft"),
        _record("color", "blue", polarity=-1, version=1),
        _record("style", "formal", status=SUPERSEDED),
    ]

    rules = sparse.compile_hard_conflict_rules(
        category_text="dresses",
        active_terms=["linen", "women"],
        excluded_terms=["red", "silk", "blue"],
        current_version=2,
        records=records,
    )

    assert rules.negative == (("color", "red"),)
    assert dict(rules.positive) == {
        "audience": ("women",),
        "category": ("dress",),
        "material": ("linen",),
    }


def test_default_off_never_opens_agent_and_returns_exact_prefix(tmp_path: Path) -> None:
    prefix = _pool(137)
    with patch.object(sparse, "Agent", side_effect=AssertionError("Agent must stay closed")):
        runtime = sparse.SparseMultiviewExpander(tmp_path / "absent.jsonl")
        runtime.validate()
        result = runtime.expand(
            prefix,
            category_text="dress",
            active_terms=["linen"],
            excluded_terms=[],
            current_version=2,
            records=[_record("material", "linen")],
        )

    assert result.enabled is False
    assert result.activated is False
    assert result.candidates == prefix
    assert result.prefix == prefix
    assert not result.route and not result.novel_route and not result.tail
    runtime.close()
    runtime.close()
    assert runtime.closed is True
    with pytest.raises(sparse.SparseMultiviewClosedError):
        runtime.expand(
            prefix,
            category_text="dress",
            active_terms=[],
            excluded_terms=[],
            current_version=2,
            records=[],
        )


def test_enabled_shadow_stably_appends_only_novel_route_members() -> None:
    prefix = _pool(100)
    novel = _pool(5, start=200)
    route = (prefix[7], novel[0], prefix[2], novel[1], *novel[2:])
    runtime = sparse.SparseMultiviewExpander(Path("."), enabled=False)
    runtime.enabled = True
    runtime._agent = SimpleNamespace(close=lambda: None)
    hits = tuple((index, identifier) for index, identifier in enumerate(route, start=1))

    try:
        with patch.object(runtime, "_query_route", return_value=hits), patch.object(
            runtime,
            "_views",
            return_value={value: ProductAttributeView(value) for value in novel},
        ):
            result = runtime.expand(
                prefix,
                category_text="dress",
                active_terms=["linen"],
                excluded_terms=[],
                current_version=2,
                records=[_record("material", "linen")],
            )
    finally:
        runtime.close()

    assert result.activated is True
    assert result.route == route
    assert result.novel_route == novel
    assert result.tail == novel
    assert result.candidates == (*prefix, *novel)
    assert result.candidates[:10] == prefix[:10]


@pytest.mark.parametrize("length", [99, 201])
def test_prefix_validation_rejects_nonfrozen_lengths(length: int, tmp_path: Path) -> None:
    runtime = sparse.SparseMultiviewExpander(tmp_path / "unused.jsonl")
    with pytest.raises(sparse.SparseMultiviewValidationError):
        runtime.expand(
            _pool(length),
            category_text="dress",
            active_terms=[],
            excluded_terms=[],
            current_version=2,
            records=[],
        )
    runtime.close()


def test_prefix_validation_rejects_duplicates(tmp_path: Path) -> None:
    prefix = [*_pool(100)]
    prefix[-1] = prefix[0]
    runtime = sparse.SparseMultiviewExpander(tmp_path / "unused.jsonl")
    with pytest.raises(sparse.SparseMultiviewValidationError):
        runtime.expand(
            prefix,
            category_text="dress",
            active_terms=[],
            excluded_terms=[],
            current_version=2,
            records=[],
        )
    runtime.close()


def test_conflict_mask_drops_only_reliable_explicit_conflicts() -> None:
    identifiers = ("UNKNOWN", "NEGATIVE", "MATERIAL", "CATEGORY", "DESCRIPTION", "MATCH")
    rules = sparse.HardConflictRules(
        negative=(("color", "red"),),
        positive=(("category", ("dress",)), ("material", ("linen",))),
    )
    views = {
        "UNKNOWN": ProductAttributeView("UNKNOWN"),
        "NEGATIVE": ProductAttributeView(
            "NEGATIVE", color=(_attribute("red"),)
        ),
        "MATERIAL": ProductAttributeView(
            "MATERIAL", material=(_attribute("polyester", source="details.Material"),)
        ),
        "CATEGORY": ProductAttributeView(
            "CATEGORY", category=(_attribute("shoe", source="categories"),)
        ),
        "DESCRIPTION": ProductAttributeView(
            "DESCRIPTION",
            color=(_attribute("red", source="description"),),
            material=(_attribute("polyester", source="description"),),
        ),
        "MATCH": ProductAttributeView(
            "MATCH",
            category=(_attribute("dress", source="categories"),),
            material=(_attribute("linen", source="details.Material"),),
        ),
    }

    result = sparse.apply_hard_conflict_mask(identifiers, views, rules)

    assert result.identifiers == ("UNKNOWN", "DESCRIPTION", "MATCH")
    assert result.dropped == ("NEGATIVE", "MATERIAL", "CATEGORY")
    assert result.negative_violation_count == 1
    assert result.positive_conflict_count == 2
    assert result.conflict_count == 3


def test_worker_trace_is_canonical_minified_lf_json() -> None:
    candidates = _pool(100)
    payload = worker.canonical_trace_line(1, 2, candidates)

    assert payload == _canonical(
        {"candidates": list(candidates), "ordinal": 1, "turn": 2}
    )
    assert payload.endswith(b"\n") and not payload.endswith(b"\r\n")


def test_worker_expansion_contract_preserves_prefix_and_mask_subsequence() -> None:
    prefix = _pool(100)
    novel = _pool(3, start=200)
    catalog = frozenset((*prefix, *novel))
    result = SimpleNamespace(
        activated=True,
        candidates=(*prefix, novel[0], novel[2]),
        conflict_count=1,
        novel_route=novel,
        prefix=prefix,
        route=(prefix[0], *novel),
        tail=(novel[0], novel[2]),
        tail_conflict_count=0,
    )

    assert worker.validate_expansion_result(result, prefix, catalog) == result.candidates

    for mutation in (
        {"prefix": (*prefix[1:], prefix[0])},
        {"candidates": (*prefix[1:], prefix[0], novel[0], novel[2])},
        {"novel_route": tuple(reversed(novel))},
        {"tail": (novel[2], novel[0])},
        {"conflict_count": 0},
        {"tail_conflict_count": -1},
    ):
        invalid = SimpleNamespace(**{**vars(result), **mutation})
        with pytest.raises(worker.SparseMultiviewWorkerError):
            worker.validate_expansion_result(invalid, prefix, catalog)


def test_worker_network_audit_fails_closed() -> None:
    audit = worker.OfflineNetworkAudit()
    audit.hook("open", ())
    assert audit.attempt_count == 0

    with pytest.raises(PermissionError):
        audit.hook("socket.connect", ())
    assert audit.attempt_count == 1
    assert audit.event_counts == {"socket.connect": 1}


def test_worker_receipt_privacy_rejects_keys_and_identifiers_case_insensitively() -> None:
    catalog_id = _identifier(1)
    worker._receipt_privacy_scan(
        {"status": "SAFE", "sha256": "a" * 64}, catalog_ids={catalog_id}
    )

    for invalid in (
        {"target": "redacted"},
        {"nested": {"candidates": []}},
        {"safe": "B0ABCDEFGH"},
        {"safe": catalog_id},
        {"safe": catalog_id.lower()},
    ):
        with pytest.raises(worker.SparseMultiviewWorkerError):
            worker._receipt_privacy_scan(invalid, catalog_ids={catalog_id})


def test_worker_error_receipt_is_sanitized_and_uses_top_level_progress() -> None:
    progress = worker.WorkerProgress(
        phase="SYNTHETIC_PHASE", nonce="a" * 32, last_completed_session=7
    )
    try:
        raise RuntimeError("SECRET TARGET MESSAGE")
    except RuntimeError as error:
        receipt = worker._error_receipt(error, progress)

    encoded = _canonical(receipt)
    assert receipt["status"] == "ERROR"
    assert receipt["phase"] == "SYNTHETIC_PHASE"
    assert receipt["last_completed_session"] == 7
    assert receipt["partial_trace"]["rows"] == 0
    assert b"SECRET TARGET MESSAGE" not in encoded
    assert "message" not in worker._walk_keys(receipt)


def test_worker_cli_accepts_only_frozen_smoke_and_formal_limits() -> None:
    common = [
        "--nonce",
        "a" * 32,
        "--catalog",
        "catalog",
        "--context",
        "context",
        "--c200-reference",
        "reference",
        "--trace-output",
        "trace",
    ]
    for limit in worker.ALLOWED_SESSION_LIMITS:
        assert worker._parser().parse_args([*common, "--session-limit", str(limit)]).session_limit == limit
    with pytest.raises(worker.SparseMultiviewWorkerError):
        worker._parser().parse_args([*common, "--session-limit", "10"])


def _success_worker_receipt(
    *, nonce: str = "a" * 32, session_limit: int = 20
) -> dict[str, object]:
    return {
        "error_code": "NONE",
        "kind": "receipt",
        "last_completed_session": session_limit,
        "nonce": nonce,
        "phase": "COMPLETE",
        "record_count": session_limit * worker.TURN_COUNT,
        "schema_version": worker.SCHEMA_VERSION,
        "status": "SUCCESS",
        "summary": {
            "activation": {"activated_records": 1},
            "input_identities": {},
            "latency": {
                "extra_route_and_mask": {"p95_milliseconds": 1.0},
                "per_turn": {"p95_milliseconds": 2.0},
            },
            "lifecycle": {"atomic_exclusive_trace_publish": True},
            "mask": {
                "tail_duplicate_count": 0,
                "tail_explicit_conflict_count": 0,
            },
            "processed_sessions": session_limit,
            "processed_turns": session_limit * worker.TURN_COUNT,
            "resources": {
                "gpu_peak_bytes": 0,
                "network_attempt_count": 0,
                "peak_working_set_bytes": 1,
                "wall_seconds": 1.0,
            },
            "session_limit": session_limit,
        },
        "trace_bytes": 1,
        "trace_sha256": "b" * 64,
    }


def test_success_receipt_rejects_any_nonzero_tail_conflict_count() -> None:
    nonce = "a" * 32
    valid = _success_worker_receipt(nonce=nonce)
    assert probe._validate_worker_receipt(
        _canonical(valid), nonce=nonce, session_limit=20
    )["status"] == "SUCCESS"

    invalid = json.loads(json.dumps(valid))
    invalid["summary"]["mask"]["tail_explicit_conflict_count"] = 1
    with pytest.raises(probe.SparseProbeError) as failure:
        probe._validate_worker_receipt(
            _canonical(invalid), nonce=nonce, session_limit=20
        )
    assert failure.value.code == "RESOURCE_GATE"


def test_worker_lexical_input_and_output_allowlists_run_before_any_stat() -> None:
    nonce = "c" * 32
    valid = SimpleNamespace(
        nonce=nonce,
        session_limit=20,
        catalog=Path(str(worker.EXPECTED_CATALOG_PATH)),
        context=Path(str(worker.EXPECTED_CONTEXT_PATH)),
        c200_reference=Path(str(next(iter(worker.EXPECTED_C200_REFERENCE_PATHS)))),
        trace_output=Path(str(worker.EXPECTED_TRACE_ROOT / "synthetic.jsonl")),
    )
    progress = worker.WorkerProgress()
    with patch.object(worker, "_require_regular_file") as require_file, patch.object(
        worker, "_snapshot"
    ) as snapshot:
        worker._validate_arguments(valid, progress)
    require_file.assert_not_called()
    snapshot.assert_not_called()
    assert progress.nonce == nonce

    wrong_input = SimpleNamespace(**vars(valid))
    wrong_input.catalog = Path(r"D:\tiktok\not-allowlisted\catalog.jsonl")
    with pytest.raises(worker.SparseMultiviewWorkerError) as input_failure:
        worker._validate_arguments(wrong_input, worker.WorkerProgress())
    assert input_failure.value.error_code == "INPUT_PATH_NOT_ALLOWLISTED"

    wrong_output = SimpleNamespace(**vars(valid))
    wrong_output.trace_output = Path(r"D:\tiktok\outside-formal-root.jsonl")
    with pytest.raises(worker.SparseMultiviewWorkerError) as output_failure:
        worker._validate_arguments(wrong_output, worker.WorkerProgress())
    assert output_failure.value.error_code == "OUTPUT_PATH_NOT_ALLOWLISTED"


def test_pinned_blob_normalization_and_imported_module_origin_guard(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.py"
    source.write_bytes(b"alpha\r\nbeta\r\n")
    normalized = b"alpha\nbeta\n"
    expected = hashlib.sha1(
        b"blob " + str(len(normalized)).encode("ascii") + b"\0" + normalized
    ).hexdigest()
    assert worker._raw_git_blob_sha1(source) == expected

    c200_contract, sparse_module, registry_hash = worker._load_runtime_after_audit()
    worker._verify_imported_module_origins(c200_contract, sparse_module, registry_hash)

    def forged_registry() -> str:
        return "forged"

    with pytest.raises(worker.SparseMultiviewWorkerError) as forged:
        worker._verify_imported_module_origins(
            c200_contract, sparse_module, forged_registry
        )
    assert forged.value.error_code == "RUNTIME_MODULE_IDENTITY"


def test_runner_worktree_blob_uses_git_lf_identity_on_windows(tmp_path: Path) -> None:
    source = tmp_path / "tracked.py"
    source.write_bytes(b"first\r\nsecond\r\n")
    clean = b"first\nsecond\n"
    expected = hashlib.sha1(
        b"blob " + str(len(clean)).encode("ascii") + b"\0" + clean
    ).hexdigest()

    with patch.object(probe, "_guard_experiment_data"):
        assert probe._worktree_blob(source) == expected


def test_atomic_trace_publish_is_hard_link_and_never_copies_over_existing(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "trace.partial"
    output = tmp_path / "trace.jsonl"
    partial.write_bytes(b"closed-and-fsynced-partial\n")

    worker._publish_partial_exclusive(partial, output)
    assert output.read_bytes() == partial.read_bytes()
    assert os.path.samefile(partial, output)

    second_partial = tmp_path / "second.partial"
    existing = tmp_path / "existing.jsonl"
    second_partial.write_bytes(b"new-partial\n")
    existing.write_bytes(b"existing-destination\n")
    with patch.object(worker.os, "link") as link:
        with pytest.raises(worker.SparseMultiviewWorkerError) as failure:
            worker._publish_partial_exclusive(second_partial, existing)
    assert failure.value.error_code == "TRACE_ALREADY_EXISTS"
    link.assert_not_called()
    assert existing.read_bytes() == b"existing-destination\n"
    assert second_partial.read_bytes() == b"new-partial\n"


@pytest.mark.parametrize(
    ("script", "module"),
    [
        (probe.RUNNER_PATH, "scripts.probe_sparse_multiview_candidate_recall"),
        (probe.WORKER_PATH, "scripts.sparse_multiview_candidate_worker"),
    ],
)
def test_direct_and_module_entrypoint_self_checks(
    script: Path, module: str, tmp_path: Path
) -> None:
    direct = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--entrypoint-self-check",
            "--require-module",
            "json",
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    as_module = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            module,
            "--entrypoint-self-check",
            "--require-module",
            "json",
        ],
        cwd=probe.ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )

    for completed in (direct, as_module):
        assert completed.returncode == 0
        assert not completed.stderr
        assert json.loads(completed.stdout)["status"] == "ENTRYPOINT_SELF_CHECK_PASS"

    missing = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--entrypoint-self-check",
            "--require-module",
            "v219_intentionally_absent_required_module",
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert missing.returncode != 0


def _trace_fixture() -> tuple[
    list[dict[str, object]], list[tuple[str, ...]], frozenset[str]
]:
    references = [_pool(100), _pool(137)]
    records = [
        {"candidates": [*_pool(100), _identifier(300)], "ordinal": 1, "turn": 1},
        {"candidates": list(_pool(137)), "ordinal": 1, "turn": 2},
    ]
    return records, references, frozenset(_pool(500))


def test_runner_trace_validation_preserves_complete_variable_prefix() -> None:
    records, references, catalog = _trace_fixture()
    result = probe.validate_trace_records(
        records, references, catalog, expected_records=2
    )

    assert result.lengths == (101, 137)
    assert result.c200_lengths == (100, 137)
    assert result.activation_turns == 1
    assert result.activation_sessions == 1
    assert result.record_count == 2


@pytest.mark.parametrize("failure", ["reorder", "duplicate", "invention", "schema"])
def test_runner_trace_validation_rejects_prefix_or_schema_drift(failure: str) -> None:
    records, references, catalog = _trace_fixture()
    first = dict(records[0])
    candidates = list(first["candidates"])
    if failure == "reorder":
        candidates[0], candidates[1] = candidates[1], candidates[0]
        first["candidates"] = candidates
    elif failure == "duplicate":
        candidates[-1] = candidates[0]
        first["candidates"] = candidates
    elif failure == "invention":
        candidates[-1] = "Z999999999"
        first["candidates"] = candidates
    else:
        first["extra"] = True
    records[0] = first

    with pytest.raises(probe.SparseProbeError):
        probe.validate_trace_records(records, references, catalog, expected_records=2)


def test_load_trace_requires_canonical_lf_json(tmp_path: Path) -> None:
    references = [_pool(100) for _ in range(probe.TURN_COUNT)]
    records = [
        {
            "candidates": list(reference),
            "ordinal": 1,
            "turn": turn,
        }
        for turn, reference in enumerate(references, start=1)
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_bytes(
        b"".join(
            worker.canonical_trace_line(1, turn, row["candidates"])
            for turn, row in enumerate(records, start=1)
        )
    )

    checked = probe.load_and_validate_trace(
        trace, references, frozenset(_pool(500)), session_limit=1
    )
    assert checked.record_count == probe.TURN_COUNT

    trace.write_bytes(
        b"".join(
            json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
            for row in records
        )
    )
    with pytest.raises(probe.SparseProbeError):
        probe.load_and_validate_trace(
            trace, references, frozenset(_pool(500)), session_limit=1
        )


def test_post_attach_trace_identity_mismatch_fails_the_frozen_gate() -> None:
    before = probe.TraceValidation(
        records=(),
        lengths=(101,),
        c200_lengths=(100,),
        canonical_trace_sha256="a" * 64,
        canonical_trace_bytes=1_000,
        record_count=1,
        activation_turns=1,
        activation_sessions=1,
    )
    exact = replace(before)
    changed = probe.TraceValidation(
        records=(),
        lengths=(102,),
        c200_lengths=(100,),
        canonical_trace_sha256="b" * 64,
        canonical_trace_bytes=1_001,
        record_count=1,
        activation_turns=1,
        activation_sessions=1,
    )

    assert probe._same_trace_validation(before, exact) is True
    assert probe._same_trace_validation(before, changed) is False
    guarded_source = inspect.getsource(probe._run_guarded)
    assert "POST_ATTACH_TRACE_MUTATION" in guarded_source
    assert guarded_source.index("flags, outcome_trace_a = _flags_from_trace") < (
        guarded_source.index("POST_ATTACH_TRACE_MUTATION")
    )


def test_candidate_recall_uses_variable_c200_for_baseline_and_c320_for_union() -> None:
    target = _identifier(500)
    candidates = [*_pool(100), target, *_pool(219, start=101)]
    turns = [{"candidates": candidates, "ordinal": 1, "turn": 1}]

    flags = probe.candidate_recall_flags(
        target, 1, turns, baseline_lengths=[100]
    )

    assert flags[100] is False
    assert flags[200] is False
    assert flags[320] is True

    delayed = probe.candidate_recall_flags(
        target, 2, turns, baseline_lengths=[100]
    )
    assert not any(delayed.values())


def test_candidate_recall_c200_uses_each_turns_full_sealed_prefix() -> None:
    target = _identifier(500)
    candidates = [*_pool(199), target]
    flags = probe.candidate_recall_flags(
        target,
        1,
        [{"candidates": candidates, "ordinal": 1, "turn": 1}],
        baseline_lengths=[200],
    )
    assert flags[100] is False
    assert flags[200] is True
    assert flags[320] is True


def test_aggregate_reports_uniform_target_clusters_and_anonymous_spans() -> None:
    flags = [
        _flag_row(c200=True, c320=True),
        _flag_row(c200=False, c320=True),
        _flag_row(c200=False, c320=False),
        _flag_row(c200=False, c320=True),
    ]
    result = probe.aggregate_candidate_recall(
        flags,
        outer_fold=[0, 1, 0, 2],
        family_index=[10, 11, 12, 13],
        taxonomy=["clothing", "shoes", "clothing", "jewelry"],
        targets=["CLUSTER_A", "CLUSTER_A", "CLUSTER_B", "CLUSTER_C"],
    )

    assert result["all_sessions"]["c200"]["count"] == 1
    assert result["all_sessions"]["c320"]["count"] == 3
    assert result["c200_absent_frontier"]["sessions"] == 3
    assert result["increment"] == {
        "count": 2,
        "outer_fold_span": 2,
        "taxonomy_span": 2,
        "non_clothing_count": 2,
        "target_cluster_count": 2,
    }
    uniform = result["exact_target_cluster_uniform"]
    assert uniform["cluster_count"] == 3
    assert uniform["c200_fraction"] == pytest.approx(1 / 6, abs=1e-9)
    assert uniform["c320_complete_union_fraction"] == pytest.approx(2 / 3, abs=1e-9)
    assert uniform["delta"] == pytest.approx(0.5, abs=1e-9)
    assert not ({"target", "per_session", "membership_vector"} & set(result))


def test_raw_target_uniform_helper_drives_the_strict_positive_gate() -> None:
    flags = [
        _flag_row(c200=True, c320=True),
        _flag_row(c200=False, c320=True),
        _flag_row(c200=False, c320=False),
        _flag_row(c200=False, c320=True),
    ]
    baseline, candidate, delta = probe._exact_target_uniform_raw(
        flags, ["CLUSTER_A", "CLUSTER_A", "CLUSTER_B", "CLUSTER_C"]
    )
    assert baseline == pytest.approx(1 / 6)
    assert candidate == pytest.approx(2 / 3)
    assert delta == pytest.approx(0.5)
    assert delta > 0

    guarded_source = inspect.getsource(probe._run_guarded)
    assert "uniform_raw_delta = _exact_target_uniform_raw(flags, targets)[2]" in guarded_source
    assert "and uniform_raw_delta > 0" in guarded_source

    with pytest.raises(probe.SparseProbeError):
        probe._exact_target_uniform_raw(flags[:-1], ["A", "B", "C", "D"])


def test_aggregate_rejects_family_cross_fold_and_non_boolean_flags() -> None:
    flags = [_flag_row(c200=False, c320=False) for _ in range(2)]
    with pytest.raises(probe.SparseProbeError):
        probe.aggregate_candidate_recall(
            flags,
            outer_fold=[0, 1],
            family_index=[7, 7],
            taxonomy=["clothing", "shoes"],
            targets=["A", "B"],
        )

    flags[0][320] = 1  # type: ignore[assignment]
    with pytest.raises(probe.SparseProbeError):
        probe.aggregate_candidate_recall(
            flags,
            outer_fold=[0, 1],
            family_index=[7, 8],
            taxonomy=["clothing", "shoes"],
            targets=["A", "B"],
        )


def test_runner_privacy_allows_aggregates_and_rejects_identity_surfaces() -> None:
    catalog_id = _identifier(1)
    safe = {
        "candidate_recall": {
            "c200": {"count": 1986, "fraction": 0.993},
            "c320": {"count": 1988, "fraction": 0.994},
        },
        "record_count": 20_000,
        "trace_sha256": "a" * 64,
    }
    probe._result_privacy_scan(safe, {catalog_id})

    for invalid in (
        {"candidates": []},
        {"target": "redacted"},
        {"per_session": [0]},
        {"records": []},
        {"ordinal": 1},
        {"turn": 1},
        {"query_terms": ["redacted"]},
        {"outer_fold": 0},
        {"family_index": 7},
        {"safe": "B0ABCDEFGH"},
        {"safe": catalog_id},
        {"safe": catalog_id.lower()},
        {"safe": [0] * probe.SESSION_COUNT},
    ):
        with pytest.raises(probe.SparseProbeError):
            probe._result_privacy_scan(invalid, {catalog_id})


def test_pre_receipt_path_guard_is_lexical_fail_closed_without_stat(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed.jsonl"
    proxy = tmp_path / "proxy.jsonl"
    label = tmp_path / "labels.npz"
    with patch.object(
        probe, "_pre_receipt_paths", return_value=frozenset({probe._lexical(allowed)})
    ), patch.object(probe, "PROXY_PATH", proxy), patch.object(
        probe, "LABEL_PATH", label
    ):
        probe._guard_experiment_data(allowed)
        with pytest.raises(probe.SparseProbeError) as denied:
            probe._guard_experiment_data(proxy)
        assert denied.value.code == "DATA_PATH_DENIED"
        probe._guard_experiment_data(proxy, post_receipt=True)
        probe._guard_experiment_data(label, post_receipt=True)

    assert not allowed.exists() and not proxy.exists() and not label.exists()


def test_file_identity_runs_data_guard_before_any_path_access(tmp_path: Path) -> None:
    path = tmp_path / "must-not-be-touched"
    with patch.object(
        probe,
        "_guard_experiment_data",
        side_effect=probe.SparseProbeError("denied", "DATA_PATH_DENIED"),
    ) as guard, patch.object(probe, "_require_plain") as require_plain:
        with pytest.raises(probe.SparseProbeError) as denied:
            probe._file_identity(path, "synthetic")

    assert denied.value.code == "DATA_PATH_DENIED"
    guard.assert_called_once_with(path, post_receipt=False)
    require_plain.assert_not_called()
    assert not path.exists()


def test_receipt_uses_exclusive_create_and_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    commit = "a" * 40
    with patch.object(probe, "OUTPUT_PATH", output):
        descriptor = probe._open_receipt(commit)
        os.close(descriptor)
        before = output.read_bytes()
        assert json.loads(before)["status"] == "PENDING_ONE_SHOT_CONSUMED"
        with pytest.raises(probe.SparseProbeError) as duplicate:
            probe._open_receipt(commit)

    assert duplicate.value.code == "RECEIPT_PREEXISTS"
    assert output.read_bytes() == before


@pytest.mark.parametrize("existing", [b"", b"partial", b'{"status":"complete"}\n'])
def test_receipt_refuses_every_existing_state(tmp_path: Path, existing: bytes) -> None:
    output = tmp_path / "result.json"
    output.write_bytes(existing)
    with patch.object(probe, "OUTPUT_PATH", output):
        with pytest.raises(probe.SparseProbeError):
            probe._open_receipt("b" * 40)
    assert output.read_bytes() == existing


def test_post_receipt_failure_is_compact_sanitized_and_permanent(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    commit = "c" * 40
    with patch.object(probe, "OUTPUT_PATH", output):
        descriptor = probe._open_receipt(commit)
        try:
            raise RuntimeError("SECRET TARGET MESSAGE")
        except RuntimeError as error:
            probe._write_invalid_receipt(
                descriptor, commit, error, phase="SYNTHETIC_POST_RECEIPT"
            )

        invalid = json.loads(output.read_text(encoding="utf-8"))
        assert invalid["status"] == "INVALID_ONE_SHOT_CONSUMED"
        assert invalid["phase"] == "SYNTHETIC_POST_RECEIPT"
        assert invalid["rerun_forbidden"] is True
        assert "SECRET TARGET MESSAGE" not in output.read_text(encoding="utf-8")
        with pytest.raises(probe.SparseProbeError):
            probe._open_receipt(commit)


def test_invalid_receipt_discloses_compliance_incident_and_stays_under_24kb() -> None:
    try:
        raise probe.SparseProbeError("SECRET TARGET MESSAGE", "SYNTHETIC_FAILURE")
    except probe.SparseProbeError as error:
        invalid = probe._invalid_value(
            "d" * 40, error, phase="SYNTHETIC_POST_RECEIPT"
        )

    compliance = invalid["compliance"]
    assert compliance == {
        "overall_orchestration_clean": False,
        "pre_preregistration_boundary_event":
            "PRE_PREREG_PROTECTED_PATH_ACCESS_INCIDENT_RECORDED",
        "qualification_does_not_erase_prior_incident": True,
    }
    assert invalid["algorithm_interpretation"] == (
        "implementation_or_integrity_failure_not_algorithm_no_go"
    )
    assert invalid["receipt_size"]["canonical_bytes_estimate"] == len(
        probe._canonical_bytes(invalid)
    ) + 1
    assert invalid["receipt_size"]["canonical_bytes_estimate"] <= probe.RECEIPT_BYTES_MAXIMUM
    assert b"SECRET TARGET MESSAGE" not in probe._canonical_bytes(invalid)
    probe._result_privacy_scan(invalid, {_identifier(1)})


def test_receipt_size_gate_is_exact_and_cannot_truncate_before_rejection(
    tmp_path: Path,
) -> None:
    compact = {"status": "SYNTHETIC"}
    estimate = probe._seal_receipt_size_estimate(compact)
    assert estimate == len(probe._canonical_bytes(compact)) + 1
    assert estimate <= probe.RECEIPT_BYTES_MAXIMUM

    with pytest.raises(probe.SparseProbeError) as oversized:
        probe._seal_receipt_size_estimate({"payload": "x" * probe.RECEIPT_BYTES_MAXIMUM})
    assert oversized.value.code == "RECEIPT_WRITE"

    destination = tmp_path / "receipt.json"
    destination.write_bytes(b"sentinel")
    descriptor = os.open(destination, os.O_RDWR)
    try:
        with pytest.raises(probe.SparseProbeError):
            probe._write_descriptor(
                descriptor, {"payload": "x" * probe.RECEIPT_BYTES_MAXIMUM}
            )
    finally:
        os.close(descriptor)
    assert destination.read_bytes() == b"sentinel"


def test_runner_main_never_emits_raw_traceback_or_stderr(capsysbinary) -> None:
    exit_code = probe.main(
        [
            "--entrypoint-self-check",
            "--require-module",
            "v219_intentionally_absent_required_module",
        ]
    )
    captured = capsysbinary.readouterr()
    value = json.loads(captured.out)

    assert exit_code == 2
    assert not captured.err
    assert value["status"] == "ERROR"
    assert value["error_code"] == "UNEXPECTED_EXCEPTION"
    assert value["raw_traceback_or_stderr_emitted"] is False
    assert "traceback" not in value and "stderr" not in value
    assert captured.out == _canonical(value)


def test_preflight_or_final_entrypoint_failure_cannot_open_receipt() -> None:
    with patch.object(
        probe,
        "preflight_only",
        side_effect=probe.SparseProbeError("preflight", "ENTRYPOINT_FAILURE"),
    ), patch.object(probe, "_open_receipt") as open_receipt:
        with pytest.raises(probe.SparseProbeError):
            probe.run("a" * 40)
        open_receipt.assert_not_called()

    with patch.object(probe, "preflight_only", return_value=object()), patch.object(
        probe,
        "_verify_entrypoints_before_receipt",
        side_effect=probe.SparseProbeError("missing evaluator", "ENTRYPOINT_FAILURE"),
    ), patch.object(probe, "_open_receipt") as open_receipt:
        with pytest.raises(probe.SparseProbeError):
            probe.run("a" * 40)
        open_receipt.assert_not_called()


def _strict_worker_error_receipt(*, nonce: str = "d" * 32) -> dict[str, object]:
    return {
        "error_code": "SYNTHETIC_FAILURE",
        "kind": "receipt",
        "last_completed_session": 7,
        "nonce": nonce,
        "partial_trace": {"bytes": 11, "rows": 3, "sha256": "a" * 64},
        "phase": "TRAJECTORY",
        "resources": {
            "gpu_peak_bytes": 0,
            "network_attempt_count": 0,
            "peak_working_set_backend": "synthetic",
            "peak_working_set_bytes": 1,
            "wall_seconds": 1.0,
        },
        "schema_version": worker.SCHEMA_VERSION,
        "source_identities": {},
        "status": "ERROR",
        "traceback": {
            "exception_type": "SparseMultiviewWorkerError",
            "sha256": "b" * 64,
            "top_frame": {
                "file": "sparse_multiview_candidate_worker.py",
                "function": "run",
                "line": 100,
            },
        },
    }


def test_runner_parses_exact_worker_error_progress_and_binds_parent_stderr() -> None:
    nonce = "d" * 32
    error = _strict_worker_error_receipt(nonce=nonce)
    assert "stderr" not in error
    stderr = b"synthetic stderr"
    completed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout=_canonical(error), stderr=stderr
    )
    with patch.object(probe, "_run_subprocess", return_value=completed):
        with pytest.raises(probe.SparseProbeError) as failure:
            probe._run_worker(
                mode="direct",
                nonce=nonce,
                reference=Path("reference"),
                trace=Path("trace"),
                session_limit=20,
            )

    diagnostic = failure.value.diagnostic  # type: ignore[attr-defined]
    assert diagnostic["phase"] == "TRAJECTORY"
    assert diagnostic["error_code"] == "SYNTHETIC_FAILURE"
    assert diagnostic["last_completed_session"] == 7
    assert diagnostic["partial_trace"]["rows"] == 3
    assert diagnostic["stderr"] == {
        "bytes": len(stderr),
        "sha256": hashlib.sha256(stderr).hexdigest(),
    }
    assert "synthetic stderr" not in json.dumps(diagnostic)


def test_runner_error_receipt_schema_is_exact_and_invalid_schema_fails_closed() -> None:
    nonce = "e" * 32
    valid = _strict_worker_error_receipt(nonce=nonce)
    parsed = probe._validate_worker_error_receipt(
        _canonical(valid), nonce=nonce, session_limit=20, captured_stderr=b""
    )
    assert parsed["phase"] == "TRAJECTORY"
    assert parsed["stderr"] == {"bytes": 0, "sha256": worker.EMPTY_SHA256}

    pinned = json.loads(json.dumps(valid))
    pinned["source_identities"] = {
        "starter/agent.py": {
            "bytes": 1,
            "rows": 1,
            "sha256": "c" * 64,
            "raw_git_blob_sha1": probe.PINNED_BLOBS["starter/agent.py"],
        }
    }
    assert probe._validate_worker_error_receipt(
        _canonical(pinned), nonce=nonce, session_limit=20, captured_stderr=b""
    )["phase"] == "TRAJECTORY"
    pinned["source_identities"]["starter/agent.py"]["raw_git_blob_sha1"] = "0" * 40
    with pytest.raises(probe.SparseProbeError):
        probe._validate_worker_error_receipt(
            _canonical(pinned), nonce=nonce, session_limit=20, captured_stderr=b""
        )

    invalid = dict(valid)
    invalid["stderr"] = {"bytes": 0, "sha256": worker.EMPTY_SHA256}
    with pytest.raises(probe.SparseProbeError) as schema_failure:
        probe._validate_worker_error_receipt(
            _canonical(invalid), nonce=nonce, session_limit=20, captured_stderr=b""
        )
    assert schema_failure.value.code == "WORKER_ERROR_CONTRACT"

    completed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout=_canonical(invalid), stderr=b""
    )
    with patch.object(probe, "_run_subprocess", return_value=completed):
        with pytest.raises(probe.SparseProbeError) as worker_failure:
            probe._run_worker(
                mode="direct",
                nonce=nonce,
                reference=Path("reference"),
                trace=Path("trace"),
                session_limit=20,
            )
    assert worker_failure.value.diagnostic["error_code"] == "INVALID_ERROR_RECEIPT"  # type: ignore[attr-defined]


def test_checkpoint_contract_and_target_attach_order_are_frozen() -> None:
    assert probe.IMPLEMENTATION_PATHS == {
        "starter/sparse_multiview.py",
        "scripts/sparse_multiview_candidate_worker.py",
        "scripts/probe_sparse_multiview_candidate_recall.py",
        "tests/test_sparse_multiview_candidate_recall.py",
    }
    checkpoint_source = inspect.getsource(probe._validate_git_checkpoint)
    assert '"status", "--porcelain=v1", "--untracked-files=all"' in checkpoint_source
    assert "not status" in checkpoint_source
    assert "_changed_paths(head) == IMPLEMENTATION_PATHS" in checkpoint_source
    assert 'rev-parse", "HEAD^"' in checkpoint_source

    preflight_source = inspect.getsource(probe.preflight_only)
    for forbidden in (
        "PROXY_PATH",
        "LABEL_PATH",
        "_open_receipt",
        "_open_proxy_after_receipt",
        "_load_fold_labels_after_traces",
    ):
        assert forbidden not in preflight_source

    run_source = inspect.getsource(probe._run_guarded)
    assert (
        run_source.index("preflight_only")
        < run_source.index("_verify_entrypoints_before_receipt")
        < run_source.index("_validate_git_checkpoint")
        < run_source.index("_source_checkpoint")
        < run_source.index("descriptor = _open_receipt")
        < run_source.index("_run_pair")
        < run_source.index("_trace_pair_gate")
        < run_source.index('importlib.import_module("scripts.probe_c200_candidate_recall")')
        < run_source.index("PROXY_PATH")
        < run_source.index("LABEL_PATH")
    )
    wrapper_source = inspect.getsource(probe.run)
    assert wrapper_source.index("_install_process_audit_guard") < wrapper_source.index(
        "_run_guarded"
    )
