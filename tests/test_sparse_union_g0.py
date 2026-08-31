from __future__ import annotations

import ast
import copy
from fractions import Fraction
import hashlib
import inspect
import json
from pathlib import Path, PureWindowsPath
import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import probe_sparse_union_g0 as probe
from scripts import sparse_union_g0_worker as worker
from scripts import v221_safe_bootstrap as bootstrap
from starter import sparse_union_g0 as sparse
from starter.attributes import AttributeValue, ProductAttributeView
from starter.slot_ledger import ACTIVE, DELETED, SUPERSEDED


LAYER_NAMES = ("fts_route", "product_view", "mask_decision")
LAYER_KEYS = {
    "lookups",
    "hits",
    "misses",
    "evictions",
    "size",
    "capacity",
    "avoided_operations",
}


def _identifier(index: int, prefix: str = "A") -> str:
    return f"{prefix}{index:09d}"


def _pool(
    count: int, *, start: int = 0, prefix: str = "A"
) -> tuple[str, ...]:
    return tuple(
        _identifier(index, prefix) for index in range(start, start + count)
    )


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
    return AttributeValue(
        value=value,
        source=source,
        confidence=confidence,
        raw=value,
    )


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


def _synthetic_source_blobs() -> dict[str, str]:
    return {
        relative: f"{index:040x}"
        for index, relative in enumerate(sorted(probe.IMPLEMENTATION_PATHS), start=1)
    }


def _write_synthetic_preflight_chain(
    tmp_path: Path,
    *,
    commit: str,
    blobs: dict[str, str],
) -> tuple[
    dict[str, Path],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, int | str]],
]:
    paths = {
        "claim": tmp_path / "preflight-claim.json",
        "outer": tmp_path / "preflight-outer.json",
        "terminal": tmp_path / "preflight-terminal.json",
    }
    claim: dict[str, object] = {
        "attempt_consumed": True,
        "branch": probe.BRANCH,
        "experiment_id": probe.EXPERIMENT_ID,
        "implementation_commit": commit,
        "mode": "preflight",
        "one_shot": True,
        "preregistration": {
            "blob": probe.PREREG_BLOB,
            "commit": probe.PREREG_COMMIT,
        },
        "preregistration_commit": probe.PREREG_COMMIT,
        "recorded_on": "2026-08-31",
        "schema_version": "small-ranker-v2.21-durable-one-shot-claim.v1",
        "target_source_blobs": {
            "bootstrap": blobs[probe.BOOTSTRAP_RELATIVE],
            "runner": blobs[probe.RUNNER_RELATIVE],
            "worker": blobs[probe.WORKER_RELATIVE],
        },
    }
    claim_raw = _canonical(claim)
    paths["claim"].write_bytes(claim_raw)
    claim_identity = {
        "bytes": len(claim_raw),
        "sha256": hashlib.sha256(claim_raw).hexdigest(),
    }
    bootstrap_attestation = {
        "bootstrap_blob": blobs[probe.BOOTSTRAP_RELATIVE],
        "guarded_path": True,
        "mode": "direct",
        "pycache_prefix": (
            r"D:\tiktok\.v221_runtime\v221-0123456789abcdef0123456789abcdef\pycache"
        ),
        "source_only": True,
        "target_blob": blobs[probe.RUNNER_RELATIVE],
    }
    receipt = {
        "bootstrap": bootstrap_attestation,
        "claim": claim_identity,
        "device": {
            "gpu_peak_bytes": 0,
            "reason": "frozen sparse FTS/mask/Fraction-RRF backend",
            "selected": "CPU",
        },
        "entrypoint_regression": {
            "legacy_module_denied_direct": True,
            "legacy_module_denied_module": True,
            "runner_direct": True,
            "runner_module": True,
            "worker_direct": True,
            "worker_module": True,
        },
        "experiment_id": probe.EXPERIMENT_ID,
        "git": {
            "branch": probe.BRANCH,
            "commit": commit,
            "implementation_blobs": blobs,
            "object_only_git": True,
            "preregistration_commit": probe.PREREG_COMMIT,
            "pushed": True,
            "remote": probe.REMOTE_URL,
        },
        "implementation": {
            "branch": probe.BRANCH,
            "commit": commit,
            "default_off": True,
            "preregistration_commit": probe.PREREG_COMMIT,
            "served_top10_unchanged": True,
            "target_blind": True,
        },
        "integrity": {
            "exact_triplet_each_stage": True,
            "legacy_route_executions": 0,
            "network_attempt_count": 0,
            "ordered_variable_c200_prefix": True,
            "target_sources_opened": False,
        },
        "mode": "preflight",
        "next": "candidate-recall",
        "preregistration": {
            "bytes": probe.PREREG_BYTES,
            "rows": 1,
            "sha256": probe.PREREG_SHA256,
        },
        "recorded_on": "2026-08-31",
        "rerun_forbidden": True,
        "runtime": {"synthetic": True},
        "schema_version": probe.SCHEMA_VERSION,
        "sources": {
            "catalog": {"sha256": "1" * 64},
            "sealed_c200": {"sha256": "2" * 64},
            "visible_context": {"sha256": "3" * 64},
        },
        "stages": [
            {"exact_triplet": True, "session_limit": 20},
            {"exact_triplet": True, "session_limit": 100},
        ],
        "status": "TARGET_FREE_PREFLIGHT_COMPLETE",
    }
    outer: dict[str, object] = {
        "bootstrap": bootstrap_attestation,
        "target_exit_code": 0,
        "target_receipt": receipt,
    }
    outer_raw = _canonical(outer)
    paths["outer"].write_bytes(outer_raw)
    outer_identity = {
        "bytes": len(outer_raw),
        "sha256": hashlib.sha256(outer_raw).hexdigest(),
    }
    terminal: dict[str, object] = {
        "implementation_commit": commit,
        "mode": "preflight",
        "outer": outer_identity,
        "preregistration": {
            "blob": probe.PREREG_BLOB,
            "commit": probe.PREREG_COMMIT,
        },
        "process_exit_code": 0,
        "raw_stderr_retained": False,
        "recorded_on": "2026-08-31",
        "schema_version": "small-ranker-v2.21-durable-terminal.v1",
        "status": "COMPLETE",
        "target_exit_code": 0,
        "target_receipt": receipt,
    }
    terminal_raw = _canonical(terminal)
    paths["terminal"].write_bytes(terminal_raw)
    identities = {
        "claim": claim_identity,
        "outer": outer_identity,
        "terminal": {
            "bytes": len(terminal_raw),
            "sha256": hashlib.sha256(terminal_raw).hexdigest(),
        },
    }
    return paths, claim, outer, terminal, identities


def _product(
    index: int,
    material: str,
    *,
    category: str = "dress",
    color: str = "blue",
) -> dict[str, object]:
    return {
        "parent_asin": _identifier(index, "Z"),
        "title": f"{material} {category}",
        "categories": [category],
        "features": [material, color],
        "details": {"Material": material, "Color": color},
        "store": "synthetic",
        "description": f"synthetic {material} {category}",
    }


def _write_catalog(path: Path, products: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(
            json.dumps(
                product,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
            for product in products
        )
    )


def _runtime(
    tmp_path: Path,
    *,
    cache_enabled: bool,
    name: str,
) -> sparse.SparseUnionG0Expander:
    catalog = tmp_path / f"{name}.jsonl"
    _write_catalog(
        catalog,
        [
            _product(1, "linen"),
            _product(2, "cotton"),
            _product(3, "silk", category="shoe"),
        ],
    )
    return sparse.SparseUnionG0Expander(
        catalog,
        enabled=True,
        cache_enabled=cache_enabled,
    )


def _expand(
    runtime: sparse.SparseUnionG0Expander,
    *,
    prefix: tuple[str, ...] | None = None,
) -> sparse.ExpansionResult:
    return runtime.expand(
        prefix or _pool(100),
        category_text="dress",
        active_terms=["linen"],
        excluded_terms=[],
        current_version=2,
        records=[_record("material", "linen")],
    )


def _assert_cache_accounting(diagnostics: dict[str, object]) -> None:
    assert set(diagnostics) == {"enabled", "closed", "clears", *LAYER_NAMES}
    assert type(diagnostics["enabled"]) is bool
    assert type(diagnostics["closed"]) is bool
    assert type(diagnostics["clears"]) is int
    for name in LAYER_NAMES:
        layer = diagnostics[name]
        assert isinstance(layer, dict)
        assert set(layer) == LAYER_KEYS
        assert all(type(value) is int and value >= 0 for value in layer.values())
        assert layer["lookups"] == layer["hits"] + layer["misses"]
        assert layer["avoided_operations"] == layer["hits"]
        assert layer["size"] <= layer["capacity"]


def _view_row(*enabled: str) -> dict[str, bool]:
    selected = set(enabled)
    return {view: view in selected for view in probe.VIEW_ORDER}


def _trace_with_records(
    records: tuple[tuple[tuple[str, ...], int], ...]
) -> probe.TraceAudit:
    return probe.TraceAudit(
        bytes=1,
        rows=len(records),
        sha256="a" * 64,
        c200_cells=1,
        candidate_cells=1,
        reference_prefix_bytes=1,
        expansion_turns=1,
        expansion_sessions=1,
        min_candidates=100,
        max_candidates=400,
        records=records,
    )


def _manual_npy(
    values: tuple[int, ...],
    *,
    descriptor: str = "<i4",
    version: int = 1,
    fortran_order: bool = False,
    shape: tuple[int, ...] | None = None,
) -> bytes:
    widths = {"|u1": (1, False), "<u2": (2, False), "<i4": (4, True)}
    width, signed = widths.get(descriptor, (4, True))
    header = (
        "{'descr': "
        + repr(descriptor)
        + ", 'fortran_order': "
        + repr(fortran_order)
        + ", 'shape': "
        + repr(shape if shape is not None else (len(values),))
        + ", }\n"
    ).encode("latin-1")
    if version == 1:
        prefix = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header))
    elif version == 2:
        prefix = b"\x93NUMPY\x02\x00" + struct.pack("<I", len(header))
    else:
        prefix = b"\x93NUMPY" + bytes((version, 0)) + struct.pack("<I", len(header))
    payload = b"".join(
        int(value).to_bytes(width, byteorder="little", signed=signed)
        for value in values
    )
    return prefix + header + payload


def test_dual_routes_are_field_isolated_registry_only_and_current_goal_only() -> None:
    records = [
        _record("material", "linen"),
        _record("style", "casual"),
        _record("use_case", "wedding", version=1),
        _record("material", "silk", status=SUPERSEDED),
        _record("color", "red"),
        _record("material", "wool", polarity=-1),
    ]

    first = sparse.build_route_queries(
        category_text="Dresses",
        active_terms=["red", "Acme brand", "$50"],
        excluded_terms=["wool"],
        current_version=2,
        records=records,
    )
    second = sparse.build_route_queries(
        category_text="Dresses",
        active_terms=reversed(["red", "Acme brand", "$50"]),
        excluded_terms=["wool"],
        current_version=2,
        records=reversed(records),
    )

    assert first == second
    assert first.category.canonical_values == (("category", "dress"),)
    assert first.category.terms == ("dress", "dresses")
    assert first.category.expression == (
        '{title categories} : ("dress" OR "dresses")'
    )
    assert first.positive_core.canonical_values == (
        ("material", "linen"),
        ("style", "casual"),
    )
    assert first.positive_core.terms == ("casual", "linen")
    assert first.positive_core.expression == (
        '{title features details description} : ("casual" OR "linen")'
    )
    assert " AND " not in first.category.expression.upper()
    assert " AND " not in first.positive_core.expression.upper()
    combined = first.category.expression + first.positive_core.expression
    assert "Acme" not in combined and "$50" not in combined and "wool" not in combined


def test_fake_sql_executes_exactly_two_isolated_routes_and_no_legacy_and() -> None:
    prefix = _pool(100)
    category_id = _identifier(1, "C")
    core_id = _identifier(1, "P")
    calls: list[tuple[str, str]] = []

    class FakeConnection:
        def execute(self, sql: str, parameters: tuple[str, ...]):
            expression = parameters[0]
            calls.append((sql, expression))
            if expression.startswith("{title categories}"):
                return ((1, category_id),)
            if expression.startswith("{title features details description}"):
                return ((2, core_id),)
            raise AssertionError("unexpected FTS route")

    runtime = sparse.SparseUnionG0Expander(Path("."), enabled=False)
    runtime.enabled = True
    runtime._agent = SimpleNamespace(connection=FakeConnection(), close=lambda: None)
    try:
        with patch.object(runtime, "_validate_fast_locked", return_value=None), patch.object(
            runtime,
            "_views",
            return_value={
                category_id: ProductAttributeView(category_id),
                core_id: ProductAttributeView(core_id),
            },
        ):
            result = runtime.expand(
                prefix,
                category_text="dress",
                active_terms=["linen"],
                excluded_terms=[],
                current_version=2,
                records=[_record("material", "linen", hardness="soft")],
            )
    finally:
        runtime.close()

    assert len(calls) == 2
    assert all(sql == sparse.FTS_QUERY_SQL for sql, _expression in calls)
    assert calls[0][1].startswith("{title categories} :")
    assert calls[1][1].startswith("{title features details description} :")
    assert all(" AND " not in expression.upper() for _sql, expression in calls)
    assert result.tail == (category_id, core_id)
    assert result.legacy_route_executions == 0
    assert runtime.route_diagnostics()["legacy_route_executions"] == 0


def test_exact_fraction_rrf_k60_and_frozen_tie_break() -> None:
    prefix = _pool(100)
    result = sparse.fuse_route_candidates(
        prefix,
        ("D", "A", "B"),
        ("D", "C", "B"),
    )

    assert result.tail == ("D", "B", "A", "C")
    by_id = {item.identifier: item for item in result.items}
    assert by_id["D"].score == Fraction(2, 61)
    assert by_id["B"].score == Fraction(2, 63)
    assert by_id["A"].score == by_id["C"].score == Fraction(1, 62)
    assert by_id["A"].category_rank == 2
    assert by_id["A"].positive_core_rank == 121
    assert by_id["C"].category_rank == 121
    assert by_id["C"].positive_core_rank == 2
    assert all(type(item.score) is Fraction for item in result.items)


def test_complete_variable_c200_is_ordered_prefix_and_union_is_capped_at_400() -> None:
    prefix = _pool(200, start=300, prefix="S")
    category = _pool(120, prefix="C")
    core = _pool(120, prefix="P")

    result = sparse.fuse_route_candidates(prefix, category, core)

    assert result.prefix == prefix
    assert result.candidates[: len(prefix)] == prefix
    assert result.candidates[:10] == prefix[:10]
    assert len(result.candidates) == 400
    assert len(result.tail) == 200
    assert len(result.candidates) == len(set(result.candidates))
    assert set(result.tail) <= set(category) | set(core)


@pytest.mark.parametrize("length", [99, 201])
def test_variable_c200_rejects_out_of_contract_lengths(
    length: int, tmp_path: Path
) -> None:
    runtime = sparse.SparseUnionG0Expander(tmp_path / "absent.jsonl")
    try:
        with pytest.raises(sparse.SparseUnionG0ValidationError):
            runtime.expand(
                _pool(length),
                category_text="dress",
                active_terms=[],
                excluded_terms=[],
                current_version=2,
                records=[],
            )
    finally:
        runtime.close()


def test_explicit_hard_mask_unknown_metadata_and_confidence_boundary() -> None:
    identifiers = (
        "UNKNOWN",
        "NEGATIVE_AT_BOUNDARY",
        "NEGATIVE_BELOW",
        "POSITIVE_AT_BOUNDARY",
        "POSITIVE_BELOW",
        "UNRELIABLE_SOURCE",
        "MATCH",
    )
    rules = sparse.HardConflictRules(
        negative=(("color", "red"),),
        positive=(("category", ("dress",)), ("material", ("linen",))),
    )
    views = {
        "UNKNOWN": ProductAttributeView("UNKNOWN"),
        "NEGATIVE_AT_BOUNDARY": ProductAttributeView(
            "NEGATIVE_AT_BOUNDARY",
            color=(_attribute("red", confidence=0.90),),
        ),
        "NEGATIVE_BELOW": ProductAttributeView(
            "NEGATIVE_BELOW",
            color=(_attribute("red", confidence=0.899999),),
        ),
        "POSITIVE_AT_BOUNDARY": ProductAttributeView(
            "POSITIVE_AT_BOUNDARY",
            material=(
                _attribute("polyester", source="details.Material", confidence=0.90),
            ),
        ),
        "POSITIVE_BELOW": ProductAttributeView(
            "POSITIVE_BELOW",
            material=(
                _attribute(
                    "polyester", source="details.Material", confidence=0.899999
                ),
            ),
        ),
        "UNRELIABLE_SOURCE": ProductAttributeView(
            "UNRELIABLE_SOURCE",
            material=(_attribute("polyester", source="description"),),
        ),
        "MATCH": ProductAttributeView(
            "MATCH",
            category=(_attribute("dress", source="categories"),),
            material=(_attribute("linen", source="details.Material"),),
        ),
    }

    result = sparse.apply_hard_conflict_mask(identifiers, views, rules)

    assert result.dropped == ("NEGATIVE_AT_BOUNDARY", "POSITIVE_AT_BOUNDARY")
    assert result.identifiers == (
        "UNKNOWN",
        "NEGATIVE_BELOW",
        "POSITIVE_BELOW",
        "UNRELIABLE_SOURCE",
        "MATCH",
    )
    assert result.negative_violation_count == 1
    assert result.positive_conflict_count == 1


def test_hard_rules_ignore_stale_retired_soft_and_nonvisible_constraints() -> None:
    records = [
        _record("material", "linen"),
        _record("audience", "women"),
        _record("color", "red", polarity=-1),
        _record("style", "formal", version=1),
        _record("use_case", "wedding", status=SUPERSEDED),
        _record("closure", "zip", status=DELETED),
        _record("material", "silk", polarity=-1, hardness="soft"),
        _record("color", "blue", polarity=-1, version=1),
        _record("style", "casual", polarity=-1, status=SUPERSEDED),
    ]

    rules = sparse.compile_hard_conflict_rules(
        category_text="dresses",
        active_terms=["linen", "women"],
        excluded_terms=["red", "silk", "blue", "casual"],
        current_version=2,
        records=records,
    )

    assert rules.negative == (("color", "red"),)
    assert dict(rules.positive) == {
        "audience": ("women",),
        "category": ("dress",),
        "material": ("linen",),
    }


def test_default_off_opens_no_agent_preserves_prefix_and_close_is_idempotent(
    tmp_path: Path,
) -> None:
    prefix = _pool(137)
    with patch.object(sparse, "Agent", side_effect=AssertionError("must stay off")):
        runtime = sparse.SparseUnionG0Expander(tmp_path / "absent.jsonl")
        runtime.validate()
        result = runtime.expand(
            prefix,
            category_text="dress",
            active_terms=["linen"],
            excluded_terms=[],
            current_version=2,
            records=[_record("material", "linen")],
        )

    assert result.enabled is False and result.activated is False
    assert result.candidates == result.prefix == prefix
    assert result.candidates[:10] == prefix[:10]
    assert not result.category_route and not result.positive_core_route
    assert not result.tail and result.legacy_route_executions == 0
    runtime.close()
    runtime.close()
    assert runtime.closed is True
    with pytest.raises(sparse.SparseUnionG0ClosedError):
        _expand(runtime, prefix=prefix)


def test_cached_and_uncached_three_layer_semantics_hits_and_close(
    tmp_path: Path,
) -> None:
    uncached = _runtime(tmp_path, cache_enabled=False, name="uncached")
    cached = _runtime(tmp_path, cache_enabled=True, name="cached")
    try:
        uncached_results = [_expand(uncached), _expand(uncached)]
        cached_results = [_expand(cached), _expand(cached)]
        assert [result.as_dict() for result in cached_results] == [
            result.as_dict() for result in uncached_results
        ]
        uncached_diagnostics = uncached.cache_diagnostics()
        cached_diagnostics = cached.cache_diagnostics()
        _assert_cache_accounting(uncached_diagnostics)
        _assert_cache_accounting(cached_diagnostics)
        assert all(
            uncached_diagnostics[name]["lookups"] == 0 for name in LAYER_NAMES
        )
        assert all(cached_diagnostics[name]["hits"] > 0 for name in LAYER_NAMES)
        assert all(cached_diagnostics[name]["misses"] > 0 for name in LAYER_NAMES)
    finally:
        uncached.close()
        cached.close()

    closed = cached.cache_diagnostics()
    _assert_cache_accounting(closed)
    assert closed["closed"] is True and closed["clears"] == 1
    assert all(closed[name]["size"] == 0 for name in LAYER_NAMES)


def test_worker_validates_dual_route_prefix_mask_and_exact_rrf_contract() -> None:
    prefix = _pool(100)
    category = (prefix[1], "D", "A", "B")
    core = (prefix[2], "D", "C", "B", "X")
    catalog = frozenset((*prefix, *category, *core))
    queries = SimpleNamespace(
        category=SimpleNamespace(activated=True),
        positive_core=SimpleNamespace(activated=True),
    )
    result = SimpleNamespace(
        activated=True,
        candidates=(*prefix, "D", "B", "A", "C"),
        category_filtered=("D", "A", "B"),
        category_novel=("D", "A", "B"),
        category_route=category,
        conflict_count=1,
        enabled=True,
        legacy_route_executions=0,
        positive_core_filtered=("D", "C", "B"),
        positive_core_novel=("D", "C", "B", "X"),
        positive_core_route=core,
        prefix=prefix,
        queries=queries,
        tail=("D", "B", "A", "C"),
        tail_conflict_count=0,
    )

    assert worker.validate_expansion_result(result, prefix, catalog) == result.candidates
    for mutation in (
        {"prefix": (*prefix[1:], prefix[0])},
        {"candidates": (*prefix[1:], prefix[0], *result.tail)},
        {"category_filtered": tuple(reversed(result.category_filtered))},
        {"tail": tuple(reversed(result.tail))},
        {"conflict_count": 0},
        {"legacy_route_executions": 1},
        {"tail_conflict_count": 1},
    ):
        invalid = SimpleNamespace(**{**vars(result), **mutation})
        with pytest.raises(worker.SparseUnionG0WorkerError):
            worker.validate_expansion_result(invalid, prefix, catalog)


def test_worker_trace_semantic_receipt_and_cli_contracts() -> None:
    candidates = _pool(100)
    trace = worker.canonical_trace_line(1, 2, candidates)
    assert trace == _canonical(
        {"candidates": list(candidates), "ordinal": 1, "turn": 2}
    )
    assert trace.endswith(b"\n") and not trace.endswith(b"\r\n")

    result = SimpleNamespace(as_dict=lambda: {"enabled": True, "tail": []})
    assert worker.canonical_semantic_line(1, 2, result) == _canonical(
        {"enabled": True, "tail": []}
    )
    worker._receipt_privacy_scan({"status": "SAFE", "sha256": "a" * 64})
    for invalid in ({"target": "hidden"}, {"nested": {"candidates": []}}):
        with pytest.raises(worker.SparseUnionG0WorkerError):
            worker._receipt_privacy_scan(invalid)

    parser = worker._parser()
    common = [
        "--nonce",
        "a" * 32,
        "--catalog",
        str(worker.EXPECTED_CATALOG_PATH),
        "--context",
        str(worker.EXPECTED_CONTEXT_PATH),
        "--c200-reference",
        str(next(iter(worker.EXPECTED_C200_REFERENCE_PATHS))),
        "--trace-output",
        r"D:\tiktok\.v221_runtime\synthetic1\trace.jsonl",
    ]
    for limit in worker.ALLOWED_SESSION_LIMITS:
        parsed = parser.parse_args([*common, "--session-limit", str(limit)])
        assert parsed.session_limit == limit
    with pytest.raises(worker.SparseUnionG0WorkerError):
        parser.parse_args([*common, "--session-limit", "21"])


def test_worker_cli_failure_emits_one_sanitized_receipt(capsysbinary) -> None:
    exit_code = worker.main(["--unsupported"])
    captured = capsysbinary.readouterr()
    assert exit_code == 1
    assert captured.err == b""
    assert captured.out.endswith(b"\n") and captured.out.count(b"\n") == 1
    receipt = json.loads(captured.out)
    assert receipt["status"] == "ERROR"
    assert receipt["error_code"] == "ARGUMENT_INVALID"
    assert receipt["phase"] == "ARGUMENT_VALIDATION"
    assert "message" not in json.dumps(receipt).casefold()
    assert set(receipt["traceback"]) == {
        "exception_type",
        "sha256",
        "top_frame",
    }


def test_worker_runtime_paths_and_cache_route_receipts_fail_closed_lexically() -> None:
    with patch.object(Path, "stat", side_effect=AssertionError("must be lexical")), patch.object(
        Path, "resolve", side_effect=AssertionError("must be lexical")
    ):
        for denied in (
            r"D:\tiktok\.v221_runtime\small_ranker_v2_19_old\trace.jsonl",
            r"D:\tiktok\.v221_runtime\small_ranker_v2_20_old\trace.jsonl",
            r"D:\tiktok\.v221_runtime\small_ranker_v2_20b_old\trace.jsonl",
        ):
            with pytest.raises(worker.SparseUnionG0WorkerError):
                worker._guard_legacy_namespaces(PureWindowsPath(denied))
        with pytest.raises(worker.SparseUnionG0WorkerError):
            worker._lexical_windows_path(
                Path(r"D:\tiktok\.v221_runtime\run\..\trace.jsonl")
            )

    cache = {
        "enabled": True,
        "closed": False,
        "clears": 0,
        **{
            name: {
                "lookups": 2,
                "hits": 1,
                "misses": 1,
                "evictions": 0,
                "size": 1,
                "capacity": worker.CACHE_CAPACITIES[name],
                "avoided_operations": 1,
            }
            for name in LAYER_NAMES
        },
    }
    assert worker._cache_contract(cache, after_close=False)["closed"] is False
    route = {
        "category_route_executions": 1,
        "positive_core_route_executions": 1,
        "legacy_route_executions": 0,
        "registry_sha256": worker.EXPECTED_ATTRIBUTE_REGISTRY_SHA256,
        "closed": False,
    }
    assert worker._route_contract(route)["legacy_route_executions"] == 0
    with pytest.raises(worker.SparseUnionG0WorkerError):
        worker._route_contract({**route, "legacy_route_executions": 1})


def test_worker_source_declares_no_legacy_runtime_dependency() -> None:
    source = Path(worker.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert "starter.sparse_multiview" not in imported
    assert "scripts.sparse_multiview_candidate_worker" not in imported
    assert worker.PROHIBITED_RUNTIME_MODULES == {
        "starter.sparse_multiview",
        "scripts.sparse_multiview_candidate_worker",
    }
    assert "legacy_route_executions" in source


def test_bootstrap_is_v221_source_only_and_guards_old_namespaces() -> None:
    assert bootstrap.PROJECT_ROOT == "D:/tiktok/techjam-v2-21-gloss-g0"
    assert bootstrap.RUNTIME_BASE == "D:/tiktok/.v221_runtime"
    assert bootstrap.RUNNER_PATH.endswith("/scripts/probe_sparse_union_g0.py")
    assert bootstrap.WORKER_PATH.endswith("/scripts/sparse_union_g0_worker.py")
    assert bootstrap.RUNNER_MODULE == "scripts.probe_sparse_union_g0"
    assert bootstrap.WORKER_MODULE == "scripts.sparse_union_g0_worker"
    assert bootstrap._LOCAL_SOURCE_ROWS["starter.sparse_union_g0"][0] == (
        "starter/sparse_union_g0.py"
    )
    for name in ("sparse_multiview", "sparse_multiview_candidate_worker"):
        assert all(name not in key for key in bootstrap._LOCAL_SOURCE_ROWS)

    base = bootstrap.PROJECT_ROOT + "/experiments/fast_track/"
    for marker in bootstrap._DENIED_EXPERIMENT_PREFIXES:
        assert bootstrap._is_old_experiment_denied(base + marker + "receipt.json")
        assert bootstrap._is_old_experiment_denied(base + marker + "receipt.json:ads")
    assert not bootstrap._is_old_experiment_denied(
        base + "small_ranker_v2_21_dual_view_rrf_g0_preflight_20260831.json"
    )


def test_bootstrap_guarded_path_hides_project_root_and_rejects_mutation() -> None:
    guarded = bootstrap.GuardedPath(("D:/450/conda/envs/tiktok/Lib",))
    assert bootstrap.PROJECT_ROOT in guarded
    assert bootstrap.PROJECT_ROOT not in tuple(guarded)
    assert guarded[:] == ("D:/450/conda/envs/tiktok/Lib",)
    for operation in (
        lambda: guarded.append(bootstrap.PROJECT_ROOT),
        lambda: guarded.insert(0, bootstrap.PROJECT_ROOT),
        lambda: guarded.extend((bootstrap.PROJECT_ROOT,)),
        lambda: guarded.__iadd__((bootstrap.PROJECT_ROOT,)),
        lambda: guarded.__setitem__(slice(None), (bootstrap.PROJECT_ROOT,)),
    ):
        with pytest.raises(bootstrap.BootstrapError):
            operation()


def test_runner_and_powershell_freeze_durable_claim_outer_terminal_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    runner_path = root / "scripts" / "probe_sparse_union_g0.py"
    powershell_path = root / "scripts" / "run_v221_preflight.ps1"
    assert runner_path.is_file(), "v2.21 runner must exist before implementation freeze"
    assert powershell_path.is_file(), "v2.21 PowerShell orchestrator must exist"

    runner = runner_path.read_text(encoding="utf-8")
    powershell = powershell_path.read_text(encoding="utf-8")
    combined = runner + "\n" + powershell
    for basename in (
        "small_ranker_v2_21_dual_view_rrf_g0_preflight_claim_20260831.json",
        "small_ranker_v2_21_dual_view_rrf_g0_preflight_outer_20260831.json",
        "small_ranker_v2_21_dual_view_rrf_g0_preflight_20260831.json",
        "small_ranker_v2_21_dual_view_rrf_g0_candidate_recall_claim_20260831.json",
        "small_ranker_v2_21_dual_view_rrf_g0_candidate_recall_outer_20260831.json",
        "small_ranker_v2_21_dual_view_rrf_g0_candidate_recall_20260831.json",
    ):
        assert basename in combined
    assert "O_EXCL" in combined or "CreateNew" in combined
    assert "fsync" in combined or "Flush(true)" in combined
    assert "INVALID_ONE_SHOT_CONSUMED" in combined
    assert "COMPLETE" in combined
    assert "[AllowEmptyCollection()][byte[]]$Bytes" in powershell
    assert "$captured.Stderr.Length -ne 0" in powershell
    assert '"--min-parents=2", "$PreregCommit..$Commit"' in powershell
    git_checkpoint = inspect.getsource(probe._validate_git_checkpoint)
    assert '"merge-base", "--is-ancestor"' in git_checkpoint
    assert '"--min-parents=2"' in git_checkpoint
    assert "PREREG_COMMIT," in git_checkpoint
    assert "implementation_commit," in git_checkpoint
    assert "D:/tiktok/.v221_runtime" in combined or r"D:\tiktok\.v221_runtime" in combined
    assert "Remove-Item" not in powershell

    transaction = powershell[powershell.index("$claimed = $false") :]
    mode_normalization = transaction.index("$Mode = $Mode.ToLowerInvariant()")
    attempt_path_selection = transaction.index("$attemptPaths = $ModePaths[$Mode]")
    claim_write = transaction.index(
        'Write-ExclusiveBytes -Path ([string]$attemptPaths["claim"])'
    )
    process_close = transaction.index("$processCapture = Invoke-CapturedProcess")
    outer_write = transaction.index(
        'Write-ExclusiveBytes -Path ([string]$attemptPaths["outer"])'
    )
    outer_parse = transaction.index("$parsedOuter = Parse-And-ValidateOuter")
    terminal_write = transaction.index(
        'Write-ExclusiveBytes -Path ([string]$attemptPaths["result"])'
    )
    assert mode_normalization < attempt_path_selection < claim_write
    assert claim_write < process_close < outer_write < outer_parse < terminal_write
    exclusive_writer = powershell[
        powershell.index("function Write-ExclusiveBytes") : powershell.index(
            "function Assert-ReceiptPrivacy"
        )
    ]
    assert "[System.IO.FileMode]::CreateNew" in exclusive_writer
    assert "$stream.Flush($true)" in exclusive_writer
    preclaim_checkpoint = powershell[
        powershell.index("function Assert-PushedCheckpoint") : powershell.index(
            "$claimed = $false"
        )
    ]
    assert "BOOTSTRAP_MANIFEST_BLOB_DRIFT" in preclaim_checkpoint
    assert "$RunnerRelative, $WorkerRelative, $UnionRelative" in preclaim_checkpoint
    assert "$bootstrapText.Contains($expected)" in preclaim_checkpoint

    preclaim_checks = powershell[
        powershell.index("function Invoke-PreclaimEntrypointCheck") : powershell.index(
            "$claimed = $false"
        )
    ]
    assert 'foreach ($invocationMode in @("direct", "module"))' in preclaim_checks
    assert 'Module = "scripts.probe_sparse_union_g0"' in preclaim_checks
    assert 'Module = "scripts.sparse_union_g0_worker"' in preclaim_checks
    assert '"evaluator_imported"' in preclaim_checks
    assert "$outer.target_receipt.evaluator_imported -ne $true" in preclaim_checks
    assert '"--preclaim-chain-self-check"' in preclaim_checks
    assert "PRECLAIM_PREREQUISITE_DIVERGENCE" in preclaim_checks

    preclaim_entrypoints = transaction.index(
        "Invoke-PreclaimEntrypointChecks -Blobs $sourceBlobs"
    )
    preclaim_chain = transaction.index("Invoke-PreclaimPrerequisiteCheck -Commit")
    assert preclaim_entrypoints < preclaim_chain < claim_write


@pytest.mark.parametrize("target_name", ("runner", "worker"))
@pytest.mark.parametrize("invocation_mode", ("direct", "module"))
def test_preclaim_entrypoint_target_dynamically_imports_evaluator(
    target_name: str,
    invocation_mode: str,
    capsysbinary,
) -> None:
    # The source-only bootstrap selects direct versus literal-module loading;
    # both modes invoke this same target self-check after installing its guarded
    # importer.  Exercise the target contract once per frozen matrix cell and
    # spy on the evaluator import rather than touching any formal input.
    target = probe if target_name == "runner" else worker
    entrypoint = target._entrypoint_self_check
    original_import = target.importlib.import_module
    imported: list[str] = []

    def recording_import(name: str, *args, **kwargs):
        imported.append(name)
        return original_import(name, *args, **kwargs)

    with patch.object(target.importlib, "import_module", side_effect=recording_import):
        assert (
            entrypoint(
                [
                    "--entrypoint-self-check",
                    "--require-module",
                    "starter.sparse_union_g0",
                ]
            )
            == 0
        ), invocation_mode
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    receipt = json.loads(captured.out)
    probe._entrypoint_receipt(receipt)
    assert receipt["evaluator_imported"] is True
    assert "evaluator.local_evaluator" in imported


@pytest.mark.parametrize("target_name", ("runner", "worker"))
def test_preclaim_entrypoint_missing_evaluator_fails_closed_without_receipt(
    target_name: str,
    capsysbinary,
) -> None:
    target = probe if target_name == "runner" else worker
    original_import = target.importlib.import_module

    def evaluator_missing(name: str, *args, **kwargs):
        if name == "evaluator.local_evaluator":
            raise ModuleNotFoundError("synthetic evaluator regression")
        return original_import(name, *args, **kwargs)

    with patch.object(target.importlib, "import_module", side_effect=evaluator_missing):
        assert (
            target.main(
                [
                    "--entrypoint-self-check",
                    "--require-module",
                    "starter.sparse_union_g0",
                ]
            )
            == 2
        )
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert captured.err == b""


def test_prereg_exact_six_path_allowlist_and_cpu_only_device_are_reflected() -> None:
    root = Path(__file__).resolve().parents[1]
    prereg = json.loads(
        (root / "configs" / "small_ranker_v2_21.dual_view_rrf_g0_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    assert prereg["checkpoint_choreography"]["implementation_exact_changed_paths"] == [
        "starter/sparse_union_g0.py",
        "scripts/sparse_union_g0_worker.py",
        "scripts/probe_sparse_union_g0.py",
        "scripts/v221_safe_bootstrap.py",
        "scripts/run_v221_preflight.ps1",
        "tests/test_sparse_union_g0.py",
    ]
    assert prereg["device_selection"]["selected"] == "CPU"
    assert prereg["device_selection"]["gpu_used"] is False
    assert prereg["device_selection"]["gpu_peak_bytes"] == 0
    assert sparse.RRF_K == 60
    assert sparse.ROUTE_LIMIT == 120
    assert sparse.CANDIDATE_CAP == 400
    assert sparse.FTS_ROUTE_CACHE_CAPACITY == 512
    assert sparse.PRODUCT_VIEW_CACHE_CAPACITY == 4096
    assert sparse.MASK_DECISION_CACHE_CAPACITY == 16384


def test_formal_sources_contain_target_free_before_candidate_attach_order() -> None:
    root = Path(__file__).resolve().parents[1]
    runner_path = root / "scripts" / "probe_sparse_union_g0.py"
    if not runner_path.is_file():
        pytest.fail("v2.21 runner must exist before implementation freeze")
    runner = runner_path.read_text(encoding="utf-8")
    assert "preflight" in runner.casefold()
    assert "candidate" in runner.casefold()
    assert "proxy" in runner.casefold()
    assert "labels" in runner.casefold()
    assert runner.casefold().find("preflight") < runner.casefold().find("proxy")
    assert "starter.sparse_multiview" not in inspect.getsource(worker._load_runtime_after_audit)

    preflight_source = inspect.getsource(probe._run_preflight)
    assert "_load_proxy_targets" not in preflight_source
    assert "_load_numeric_labels" not in preflight_source
    assert "audit_guard.allow_targets" not in preflight_source

    candidate_source = inspect.getsource(probe._run_candidate)
    exact_pair = candidate_source.index("_pair_semantic_gate")
    source_recheck = candidate_source.index("sources_pre_attach = _source_checkpoint()")
    target_gate = candidate_source.index("audit_guard.allow_targets()")
    proxy_attach = candidate_source.index("_load_proxy_targets")
    label_attach = candidate_source.index("_load_numeric_labels")
    assert exact_pair < source_recheck < target_gate < proxy_attach < label_attach
    assert candidate_source.count("_invoke_worker(") == 2
    assert 'mode="direct"' in candidate_source
    assert 'mode="module"' in candidate_source


def test_runner_contract_is_exactly_v221_cpu_only_and_default_off() -> None:
    assert probe.EXPERIMENT_ID == "SR-V2.21-CLAUSE-ISOLATED-DUAL-VIEW-RRF-G0"
    assert probe.BRANCH == "small-ranker-v2.21-gloss-g0"
    assert probe.PREREG_COMMIT == "b81351a0657411ab04810bb4740b35b407d175cc"
    assert probe.PREREG_BLOB == "5ee89ebda59c3dbf973fc3cd3f127ec34f47d1fa"
    assert probe.IMPLEMENTATION_PATHS == {
        "starter/sparse_union_g0.py",
        "scripts/sparse_union_g0_worker.py",
        "scripts/probe_sparse_union_g0.py",
        "scripts/v221_safe_bootstrap.py",
        "scripts/run_v221_preflight.ps1",
        "tests/test_sparse_union_g0.py",
    }
    assert probe.ALLOWED_PREFLIGHT_LIMITS == (20, 100)
    assert probe.SESSION_COUNT == 2_000
    assert probe.ROUTE_LIMIT == 120
    assert probe.MAX_CANDIDATES == 400
    assert probe.PAIR_WALL_MAXIMUM == 60.0
    assert probe.FORMAL_WALL_MAXIMUM == 1_800.0

    preflight_source = inspect.getsource(probe._run_preflight)
    assert '"default_off": True' in preflight_source
    assert '"selected": "CPU"' in preflight_source
    assert '"gpu_peak_bytes": 0' in preflight_source
    assert '"served_top10_unchanged": True' in preflight_source


def test_membership_uses_all_seven_views_variable_prefix_and_eligibility() -> None:
    generic = tuple(f"P{index:03d}" for index in range(300))
    default_record = (generic[:120], 120)
    records = [default_record] * (probe.SESSION_COUNT * probe.TURN_COUNT)
    targets = tuple(f"TARGET-{index}" for index in range(probe.SESSION_COUNT))
    eligibility = [1] * probe.SESSION_COUNT

    def placed(target: str, position: int, *, prefix_length: int) -> tuple[tuple[str, ...], int]:
        candidates = list(generic[:300])
        candidates[position - 1] = target
        return tuple(candidates), prefix_length

    # Session zero sees the target only before the current intent version is
    # eligible; the evidence must be ignored completely.
    eligibility[0] = 3
    records[0] = placed(targets[0], 5, prefix_length=120)
    records[1] = placed(targets[0], 5, prefix_length=120)
    cases = (
        (1, 5, 120),
        (2, 15, 120),
        (3, 30, 120),
        (4, 70, 120),
        (5, 150, 160),
        (6, 150, 120),
        (7, 250, 200),
    )
    for session, position, prefix_length in cases:
        records[session * probe.TURN_COUNT] = placed(
            targets[session], position, prefix_length=prefix_length
        )

    flags = probe._membership_flags(
        _trace_with_records(tuple(records)), targets, tuple(eligibility)
    )

    assert tuple(flags[0]) == probe.VIEW_ORDER
    assert flags[0] == _view_row()
    assert flags[1] == _view_row(*probe.VIEW_ORDER)
    assert flags[2] == _view_row(*probe.VIEW_ORDER[1:])
    assert flags[3] == _view_row(*probe.VIEW_ORDER[2:])
    assert flags[4] == _view_row(*probe.VIEW_ORDER[3:])
    assert flags[5] == _view_row(*probe.VIEW_ORDER[4:])
    assert flags[6] == _view_row("EXPANDED_FIXED_K200", "C400_COMPLETE_UNION")
    assert flags[7] == _view_row("C400_COMPLETE_UNION")
    assert all(flags[index] == _view_row() for index in range(8, probe.SESSION_COUNT))


def test_candidate_aggregate_exposes_promotion_distribution_and_uniform_gain() -> None:
    sealed_views = probe.VIEW_ORDER[:5]
    rows = [_view_row(*probe.VIEW_ORDER) for _ in range(1_986)]
    rows.extend(_view_row() for _ in range(probe.SESSION_COUNT - len(rows)))
    gain_indices = (1_986, 1_987)
    for index in gain_indices:
        rows[index] = _view_row("C400_COMPLETE_UNION")
    targets = tuple(f"TARGET-{index}" for index in range(probe.SESSION_COUNT))
    outer_fold = tuple(index % 5 for index in range(probe.SESSION_COUNT))
    family_index = tuple(range(probe.SESSION_COUNT))
    taxonomy = [1] * probe.SESSION_COUNT
    taxonomy[gain_indices[1]] = 3

    aggregate, uniform_delta = probe._aggregate_candidate_recall(
        tuple(rows),
        targets=targets,
        outer_fold=outer_fold,
        family_index=family_index,
        taxonomy_code=tuple(taxonomy),
    )

    assert list(aggregate["all_2000_sessions"]) == list(probe.VIEW_ORDER)
    assert all(
        aggregate["all_2000_sessions"][view]["count"] == 1_986
        for view in sealed_views
    )
    assert aggregate["all_2000_sessions"]["C400_COMPLETE_UNION"]["count"] == 1_988
    assert aggregate["c200_absent_frontier"]["sessions"] == 14
    assert aggregate["increment"] == {
        "count": 2,
        "outer_fold_span": 2,
        "taxonomy_span": 2,
        "non_clothing_count": 1,
        "target_cluster_count": 2,
    }
    assert uniform_delta > 0.0
    assert aggregate["exact_target_cluster_uniform"]["delta"] > 0.0
    assert aggregate["family_disjoint_audit"] == {
        "valid": True,
        "family_count": probe.SESSION_COUNT,
        "families_crossing_outer_folds": 0,
    }
    assert sum(item["increment"] for item in aggregate["by_outer_fold"]) == 2
    assert aggregate["by_taxonomy"]["shoes"]["increment"] == 1


def test_candidate_aggregate_rejects_family_crossing_outer_folds() -> None:
    rows = tuple(_view_row() for _ in range(probe.SESSION_COUNT))
    targets = tuple(f"TARGET-{index}" for index in range(probe.SESSION_COUNT))
    folds = tuple(index % 5 for index in range(probe.SESSION_COUNT))
    families = list(range(probe.SESSION_COUNT))
    families[1] = families[0]
    with pytest.raises(probe.SparseUnionProbeError) as failure:
        probe._aggregate_candidate_recall(
            rows,
            targets=targets,
            outer_fold=folds,
            family_index=tuple(families),
            taxonomy_code=(1,) * probe.SESSION_COUNT,
        )
    assert failure.value.code == "FAMILY_CROSSES_FOLD"


def test_manual_npy_integer_parser_supports_frozen_little_endian_types() -> None:
    signed = _manual_npy((-7, 0, 42), descriptor="<i4", version=1)
    unsigned = _manual_npy((0, 255, 65_535), descriptor="<u2", version=2)
    assert probe._parse_npy_integer(signed, expected_count=3) == (-7, 0, 42)
    assert probe._parse_npy_integer(unsigned, expected_count=3) == (0, 255, 65_535)

    invalid = (
        (signed[:-1], 3, "NPY_PAYLOAD"),
        (_manual_npy((1,), descriptor=">i4"), 1, "NPY_DTYPE"),
        (_manual_npy((1,), shape=(2,)), 1, "NPY_SCHEMA"),
        (_manual_npy((1,), fortran_order=True), 1, "NPY_SCHEMA"),
        (_manual_npy((1,), version=9), 1, "NPY_VERSION"),
    )
    for raw, expected_count, code in invalid:
        with pytest.raises(probe.SparseUnionProbeError) as failure:
            probe._parse_npy_integer(raw, expected_count=expected_count)
        assert failure.value.code == code


def test_durable_preflight_chain_and_candidate_claim_are_byte_bound(
    tmp_path: Path,
) -> None:
    commit = "c" * 40
    blobs = _synthetic_source_blobs()
    paths, claim, _outer, _terminal, identities = _write_synthetic_preflight_chain(
        tmp_path, commit=commit, blobs=blobs
    )
    with patch.multiple(
        probe,
        PREFLIGHT_CLAIM_PATH=paths["claim"],
        PREFLIGHT_OUTER_PATH=paths["outer"],
        PREFLIGHT_RESULT_PATH=paths["terminal"],
    ):
        assert probe._validate_preflight_terminal(commit, blobs) == identities

    candidate_claim = {
        **claim,
        "mode": "candidate",
        "preflight_prerequisite": identities,
    }
    candidate_path = tmp_path / "candidate-claim.json"
    candidate_path.write_bytes(_canonical(candidate_claim))
    parsed, candidate_identity = probe._parse_durable_claim(
        candidate_path, "candidate", commit, blobs
    )
    assert parsed["preflight_prerequisite"] == identities
    assert candidate_identity == {
        "bytes": len(_canonical(candidate_claim)),
        "sha256": hashlib.sha256(_canonical(candidate_claim)).hexdigest(),
    }

    # The candidate runner compares the immutable claim's prerequisite with a
    # fresh validation of all three prior files before it opens any source.
    with (
        patch.object(
            probe,
            "_parse_durable_claim",
            return_value=(parsed, candidate_identity),
        ),
        patch.object(probe, "_validate_preflight_chain", return_value=identities),
        patch.object(
            probe,
            "_source_checkpoint",
            side_effect=RuntimeError("binding passed before source access"),
        ),
        pytest.raises(RuntimeError, match="binding passed before source access"),
    ):
        probe._run_candidate(
            implementation_commit=commit,
            git_report={"implementation_blobs": blobs},
            attestation={},
            audit_guard=object(),
        )

    tampered_prerequisite = copy.deepcopy(identities)
    tampered_prerequisite["outer"]["sha256"] = "f" * 64
    tampered_claim = {**parsed, "preflight_prerequisite": tampered_prerequisite}
    with (
        patch.object(
            probe,
            "_parse_durable_claim",
            return_value=(tampered_claim, candidate_identity),
        ),
        patch.object(probe, "_validate_preflight_chain", return_value=identities),
        patch.object(
            probe,
            "_source_checkpoint",
            side_effect=AssertionError("must fail before source access"),
        ),
        pytest.raises(probe.SparseUnionProbeError) as failure,
    ):
        probe._run_candidate(
            implementation_commit=commit,
            git_report={"implementation_blobs": blobs},
            attestation={},
            audit_guard=object(),
        )
    assert failure.value.code == "CANDIDATE_CLAIM_PREFLIGHT_BINDING"


@pytest.mark.parametrize(
    ("location", "mutation", "expected_code"),
    (
        ("claim", ("recorded_on", "2026-09-01"), "CLAIM_SEMANTICS"),
        ("outer", ("target_exit_code", 1), "PREFLIGHT_OUTER_NOT_COMPLETE"),
        (
            "terminal",
            ("outer", {"bytes": 1, "sha256": "f" * 64}),
            "PREFLIGHT_TERMINAL_NOT_COMPLETE",
        ),
    ),
)
def test_durable_preflight_chain_single_file_tamper_fails_closed(
    tmp_path: Path,
    location: str,
    mutation: tuple[str, object],
    expected_code: str,
) -> None:
    commit = "c" * 40
    blobs = _synthetic_source_blobs()
    paths, claim, outer, terminal, _identities = _write_synthetic_preflight_chain(
        tmp_path, commit=commit, blobs=blobs
    )
    documents = {"claim": claim, "outer": outer, "terminal": terminal}
    changed = copy.deepcopy(documents[location])
    changed[mutation[0]] = mutation[1]
    paths[location].write_bytes(_canonical(changed))
    with (
        patch.multiple(
            probe,
            PREFLIGHT_CLAIM_PATH=paths["claim"],
            PREFLIGHT_OUTER_PATH=paths["outer"],
            PREFLIGHT_RESULT_PATH=paths["terminal"],
        ),
        pytest.raises(probe.SparseUnionProbeError) as failure,
    ):
        probe._validate_preflight_chain(commit, blobs)
    assert failure.value.code == expected_code


def test_durable_preflight_receipt_links_claim_and_terminal_exactly(
    tmp_path: Path,
) -> None:
    commit = "c" * 40
    blobs = _synthetic_source_blobs()
    paths, _claim, outer, terminal, _identities = _write_synthetic_preflight_chain(
        tmp_path, commit=commit, blobs=blobs
    )

    changed_outer = copy.deepcopy(outer)
    changed_outer["target_receipt"]["claim"]["sha256"] = "e" * 64
    paths["outer"].write_bytes(_canonical(changed_outer))
    with (
        patch.multiple(
            probe,
            PREFLIGHT_CLAIM_PATH=paths["claim"],
            PREFLIGHT_OUTER_PATH=paths["outer"],
            PREFLIGHT_RESULT_PATH=paths["terminal"],
        ),
        pytest.raises(probe.SparseUnionProbeError) as failure,
    ):
        probe._validate_preflight_chain(commit, blobs)
    assert failure.value.code == "PREFLIGHT_RECEIPT_NOT_ELIGIBLE"

    # Restore a valid outer, then mutate only the terminal's embedded receipt.
    paths, _claim, _outer, terminal, _identities = _write_synthetic_preflight_chain(
        tmp_path, commit=commit, blobs=blobs
    )
    changed_terminal = copy.deepcopy(terminal)
    changed_terminal["target_receipt"]["status"] = "WRONG"
    paths["terminal"].write_bytes(_canonical(changed_terminal))
    with (
        patch.multiple(
            probe,
            PREFLIGHT_CLAIM_PATH=paths["claim"],
            PREFLIGHT_OUTER_PATH=paths["outer"],
            PREFLIGHT_RESULT_PATH=paths["terminal"],
        ),
        pytest.raises(probe.SparseUnionProbeError) as failure,
    ):
        probe._validate_preflight_chain(commit, blobs)
    assert failure.value.code == "PREFLIGHT_TERMINAL_NOT_COMPLETE"


def test_runner_cli_failure_is_sanitized_and_never_enters_formal_run(
    capsysbinary,
) -> None:
    parsed = probe._parser().parse_args(
        [
            "--run",
            "--mode",
            "preflight",
            "--implementation-commit",
            "a" * 40,
        ]
    )
    assert parsed.run is True and parsed.mode == "preflight"
    with pytest.raises(probe.SparseUnionProbeError):
        probe._parser().parse_args(
            ["--run", "--mode", "unknown", "--implementation-commit", "a" * 40]
        )

    with patch.object(probe, "run", side_effect=AssertionError("formal run forbidden")):
        exit_code = probe.main(
            [
                "--run",
                "--mode",
                "candidate",
                "--implementation-commit",
                "not-a-commit",
            ]
        )
    captured = capsysbinary.readouterr()
    assert exit_code == 1 and captured.err == b""
    receipt = json.loads(captured.out)
    assert receipt == {
        "error_code": "CLI_CONTRACT",
        "experiment_id": probe.EXPERIMENT_ID,
        "identifiers_disclosed": False,
        "mode": "candidate",
        "rerun_forbidden": True,
        "schema_version": probe.SCHEMA_VERSION,
        "status": "INVALID_ONE_SHOT_CONSUMED",
        "traceback_disclosed": False,
    }


def test_bootstrap_manifest_binds_runner_worker_and_worker_blob_cli() -> None:
    assert bootstrap._is_hex_blob(bootstrap.RUNNER_BLOB)
    assert bootstrap._is_hex_blob(bootstrap.WORKER_BLOB)
    manifest = (
        (
            bootstrap.RUNNER_PATH,
            bootstrap.RUNNER_MODULE,
            bootstrap.RUNNER_BLOB,
        ),
        (
            bootstrap.WORKER_PATH,
            bootstrap.WORKER_MODULE,
            bootstrap.WORKER_BLOB,
        ),
    )
    for mode in ("direct", "module"):
        for path, module, blob in manifest:
            parsed = {
                "--mode": mode,
                "--target-path": path,
                "--target-module": module,
                "--target-blob": blob,
                "--bootstrap-blob": "b" * 40,
            }
            assert bootstrap._match_target(parsed) == {
                "mode": mode,
                "path": path,
                "module": module,
                "blob": blob,
            }
            with pytest.raises(bootstrap.BootstrapError):
                bootstrap._match_target({**parsed, "--target-path": path + "."})

    parser = worker._parser()
    options = {action.dest for action in parser._actions}
    assert {"expected_worker_blob", "expected_union_blob"} <= options
    args = SimpleNamespace(
        semantic_audit=True,
        expected_worker_blob="a" * 40,
        expected_union_blob="b" * 40,
    )
    with patch.object(
        worker,
        "_raw_git_blob_sha1",
        side_effect=lambda path: "a" * 40 if Path(path).name == Path(worker.__file__).name else "b" * 40,
    ):
        worker._validate_semantic_source_blobs(args)
    with pytest.raises(worker.SparseUnionG0WorkerError) as failure:
        worker._validate_semantic_source_blobs(
            SimpleNamespace(
                semantic_audit=True,
                expected_worker_blob="not-a-blob",
                expected_union_blob="b" * 40,
            )
        )
    assert failure.value.error_code == "EXPECTED_SOURCE_BLOB_MISMATCH"
