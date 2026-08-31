from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import probe_sparse_multiview_cache_preflight as preflight
from scripts import sparse_multiview_candidate_worker as worker
from scripts import v220b_safe_bootstrap as bootstrap
from starter import sparse_multiview as sparse
from starter.slot_ledger import ACTIVE


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


def _pool(count: int, *, start: int = 0, prefix: str = "A") -> tuple[str, ...]:
    return tuple(_identifier(index, prefix) for index in range(start, start + count))


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


def _product(index: int, material: str, *, color: str = "blue") -> dict[str, object]:
    return {
        "parent_asin": _identifier(index, "Z"),
        "title": f"{material} dress",
        "categories": ["dress"],
        "features": [material, color],
        "details": {"Material": material, "Color": color},
        "store": "synthetic",
        "description": f"synthetic {material} garment",
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
    products: list[dict[str, object]] | None = None,
    name: str = "catalog",
) -> sparse.SparseMultiviewExpander:
    catalog = tmp_path / f"{name}.jsonl"
    _write_catalog(
        catalog,
        products
        or [
            _product(1, "linen"),
            _product(2, "cotton"),
            _product(3, "wool"),
        ],
    )
    return sparse.SparseMultiviewExpander(
        catalog,
        enabled=True,
        cache_enabled=cache_enabled,
    )


def _expand(
    runtime: sparse.SparseMultiviewExpander,
    material: str = "linen",
    *,
    prefix: tuple[str, ...] | None = None,
    excluded_color: str | None = None,
) -> sparse.ExpansionResult:
    records = [_record("material", material)]
    excluded: list[str] = []
    if excluded_color is not None:
        records.append(_record("color", excluded_color, polarity=-1))
        excluded.append(excluded_color)
    return runtime.expand(
        prefix or _pool(100),
        category_text="dress",
        active_terms=[material],
        excluded_terms=excluded,
        current_version=2,
        records=records,
    )


def _assert_layer_accounting(diagnostics: dict[str, object]) -> None:
    assert set(diagnostics) == {
        "enabled",
        "closed",
        "clears",
        *LAYER_NAMES,
    }
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


def _canonical_line(value: object) -> bytes:
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


def test_cache_constructor_is_default_off_and_strictly_boolean(tmp_path: Path) -> None:
    with patch.object(sparse, "Agent", side_effect=AssertionError("Agent must not open")):
        runtime = sparse.SparseMultiviewExpander(tmp_path / "absent.jsonl")
        diagnostics = runtime.cache_diagnostics()
        runtime.close()

    _assert_layer_accounting(diagnostics)
    assert diagnostics["enabled"] is False
    assert diagnostics["closed"] is False
    assert all(diagnostics[name]["lookups"] == 0 for name in LAYER_NAMES)

    for invalid in (None, 0, 1, "yes", object()):
        with pytest.raises(sparse.SparseMultiviewValidationError):
            sparse.SparseMultiviewExpander(
                tmp_path / "unused.jsonl",
                cache_enabled=invalid,  # type: ignore[arg-type]
            )


def test_cached_and_uncached_expansion_have_exact_full_field_parity(tmp_path: Path) -> None:
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
        _assert_layer_accounting(uncached_diagnostics)
        _assert_layer_accounting(cached_diagnostics)
        assert all(uncached_diagnostics[name]["lookups"] == 0 for name in LAYER_NAMES)
        assert all(cached_diagnostics[name]["hits"] > 0 for name in LAYER_NAMES)
        assert all(cached_diagnostics[name]["misses"] > 0 for name in LAYER_NAMES)
    finally:
        uncached.close()
        cached.close()


def test_cache_never_reuses_prefix_novel_route_or_final_result(tmp_path: Path) -> None:
    cached = _runtime(tmp_path, cache_enabled=True)
    uncached = _runtime(tmp_path, cache_enabled=False, name="uncached")
    route_identifier = _identifier(1, "Z")
    first_prefix = _pool(100)
    second_prefix = (route_identifier, *_pool(99, start=1))
    try:
        for runtime in (cached, uncached):
            first = _expand(runtime, prefix=first_prefix)
            second = _expand(runtime, prefix=second_prefix)
            assert first.prefix == first_prefix
            assert first.tail == (route_identifier,)
            assert second.prefix == second_prefix
            assert second.tail == ()
            assert second.candidates == second_prefix

        assert cached.cache_diagnostics()["fts_route"]["hits"] >= 1
    finally:
        cached.close()
        uncached.close()


def test_three_layer_lru_promotion_and_eviction_are_deterministic(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, cache_enabled=True)
    # Shrink only this already-validated synthetic instance so the public expand
    # path exercises eviction without weakening the frozen module constants.
    runtime._cache_capacities = {name: 2 for name in LAYER_NAMES}
    try:
        outputs = [_expand(runtime, material) for material in (
            "linen",
            "cotton",
            "linen",
            "wool",
            "cotton",
        )]
        assert [result.tail for result in outputs] == [
            (_identifier(1, "Z"),),
            (_identifier(2, "Z"),),
            (_identifier(1, "Z"),),
            (_identifier(3, "Z"),),
            (_identifier(2, "Z"),),
        ]
        diagnostics = runtime.cache_diagnostics()
        _assert_layer_accounting(diagnostics)
        for name in LAYER_NAMES:
            layer = diagnostics[name]
            assert layer["capacity"] == 2
            assert layer["size"] == 2
            assert layer["evictions"] == 2
            assert layer["hits"] > 0
    finally:
        runtime.close()


class _TruncatedCursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._rows = cursor.fetchall()

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._rows[:-1]


class _ConnectionProxy:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        fail_sql: str | None = None,
        truncate_sql: str | None = None,
    ) -> None:
        self._connection = connection
        self.fail_sql = fail_sql
        self.truncate_sql = truncate_sql

    def execute(
        self,
        sql: str,
        parameters: tuple[object, ...] | list[object] = (),
    ) -> sqlite3.Cursor | _TruncatedCursor:
        if self.fail_sql is not None and self.fail_sql in sql:
            raise sqlite3.OperationalError("synthetic SQL failure")
        cursor = self._connection.execute(sql, parameters)
        if self.truncate_sql is not None and self.truncate_sql in sql:
            return _TruncatedCursor(cursor)
        return cursor

    def __getattr__(self, name: str) -> object:
        return getattr(self._connection, name)


def test_failed_fts_fill_is_transactional_and_preserves_exception_class(
    tmp_path: Path,
) -> None:
    runtimes = [
        _runtime(tmp_path, cache_enabled=False, name="uncached"),
        _runtime(tmp_path, cache_enabled=True, name="cached"),
    ]
    observed: list[type[BaseException]] = []
    try:
        for runtime in runtimes:
            agent = runtime._agent
            assert agent is not None
            original = agent.connection
            agent.connection = _ConnectionProxy(  # type: ignore[assignment]
                original,
                fail_sql="WHERE products MATCH",
            )
            with pytest.raises(sqlite3.OperationalError) as failure:
                _expand(runtime)
            observed.append(type(failure.value))
            agent.connection = original
        assert observed == [sqlite3.OperationalError, sqlite3.OperationalError]
        diagnostics = runtimes[1].cache_diagnostics()
        assert diagnostics["fts_route"]["size"] == 0
        assert diagnostics["product_view"]["size"] == 0
        assert diagnostics["mask_decision"]["size"] == 0
        assert _expand(runtimes[1]).tail == (_identifier(1, "Z"),)
    finally:
        for runtime in runtimes:
            runtime.close()


def test_failed_view_batch_inserts_no_views_from_that_batch(tmp_path: Path) -> None:
    products = [_product(1, "linen"), _product(2, "linen")]
    runtime = _runtime(
        tmp_path,
        cache_enabled=True,
        products=products,
    )
    agent = runtime._agent
    assert agent is not None
    original = agent.connection
    agent.connection = _ConnectionProxy(  # type: ignore[assignment]
        original,
        truncate_sql="FROM products WHERE rowid IN",
    )
    try:
        with pytest.raises(sparse.SparseMultiviewValidationError):
            _expand(runtime)
        diagnostics = runtime.cache_diagnostics()
        assert diagnostics["fts_route"]["size"] == 1
        assert diagnostics["product_view"]["size"] == 0
        assert diagnostics["mask_decision"]["size"] == 0
    finally:
        agent.connection = original
        runtime.close()


def test_failed_mask_invocation_inserts_no_staged_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    products = [
        _product(1, "linen", color="blue"),
        _product(2, "linen", color="blue"),
    ]
    runtime = _runtime(tmp_path, cache_enabled=True, products=products)
    original = sparse.classify_candidate
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic classifier failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(sparse, "classify_candidate", fail_second)
    try:
        with pytest.raises(RuntimeError, match="synthetic classifier failure"):
            _expand(runtime, excluded_color="red")
        diagnostics = runtime.cache_diagnostics()
        assert diagnostics["fts_route"]["size"] == 1
        assert diagnostics["product_view"]["size"] == 2
        assert diagnostics["mask_decision"]["size"] == 0
    finally:
        runtime.close()


def test_close_clears_once_preserves_counters_and_rejects_post_close(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, cache_enabled=True)
    _expand(runtime)
    before = runtime.cache_diagnostics()
    runtime.close()
    after = runtime.cache_diagnostics()
    runtime.close()
    idempotent = runtime.cache_diagnostics()

    _assert_layer_accounting(before)
    _assert_layer_accounting(after)
    assert before["closed"] is False and after["closed"] is True
    assert after["clears"] == before["clears"] + 1
    assert idempotent == after
    for name in LAYER_NAMES:
        assert before[name]["size"] > 0
        assert after[name]["size"] == 0
        assert {
            key: after[name][key]
            for key in LAYER_KEYS - {"size"}
        } == {
            key: before[name][key]
            for key in LAYER_KEYS - {"size"}
        }
    with pytest.raises(sparse.SparseMultiviewClosedError):
        _expand(runtime)


def test_expand_and_close_are_serialized_by_the_instance_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path, cache_enabled=True)
    entered = threading.Event()
    release = threading.Event()
    close_finished = threading.Event()
    failures: list[BaseException] = []
    original = runtime._query_route

    def blocking_query(query: sparse.RewriteQuery) -> tuple[tuple[int, str], ...]:
        entered.set()
        if not release.wait(5):
            raise AssertionError("synthetic query was not released")
        return original(query)

    monkeypatch.setattr(runtime, "_query_route", blocking_query)

    def expand_target() -> None:
        try:
            _expand(runtime)
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)

    def close_target() -> None:
        try:
            runtime.close()
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)
        finally:
            close_finished.set()

    expand_thread = threading.Thread(target=expand_target)
    close_thread = threading.Thread(target=close_target)
    expand_thread.start()
    assert entered.wait(5)
    close_thread.start()
    assert not close_finished.wait(0.05)
    release.set()
    expand_thread.join(5)
    close_thread.join(5)

    assert not expand_thread.is_alive() and not close_thread.is_alive()
    assert failures == []
    assert runtime.closed is True
    assert runtime.cache_diagnostics()["closed"] is True


def test_explicit_catalog_drift_clears_cache_and_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, cache_enabled=True)
    _expand(runtime)
    before = runtime.cache_diagnostics()
    agent = runtime._agent
    assert agent is not None
    agent.connection.execute("PRAGMA query_only=OFF")
    agent.connection.execute(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
        (_identifier(99, "Z"), "linen dress", "dress", "linen", "", "", ""),
    )
    try:
        with pytest.raises(sparse.SparseMultiviewValidationError):
            runtime.validate()
        after = runtime.cache_diagnostics()
        assert after["clears"] == before["clears"] + 1
        assert all(after[name]["size"] == 0 for name in LAYER_NAMES)
        assert runtime.closed is True and after["closed"] is True
        with pytest.raises(sparse.SparseMultiviewClosedError):
            _expand(runtime)
    finally:
        runtime.close()


def test_cache_diagnostics_are_aggregate_private_copies(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, cache_enabled=True)
    try:
        result = _expand(runtime)
        diagnostics = runtime.cache_diagnostics()
        _assert_layer_accounting(diagnostics)
        encoded = _canonical_line(diagnostics)
        assert result.route[0].encode("ascii") not in encoded
        assert b"linen" not in encoded and b"dress" not in encoded
        assert b"query" not in encoded and b"rule" not in encoded
        diagnostics["fts_route"]["hits"] = 999  # type: ignore[index]
        assert runtime.cache_diagnostics()["fts_route"]["hits"] != 999
    finally:
        runtime.close()


def test_worker_semantic_line_is_canonical_full_result_and_order_stable() -> None:
    result = SimpleNamespace(
        as_dict=lambda: {
            "activated": True,
            "candidates": list(_pool(100)),
            "conflict_count": 0,
            "enabled": True,
            "mechanism": "synthetic",
            "novel_route": [],
            "prefix": list(_pool(100)),
            "query": {"activated": True},
            "route": [],
            "rules": {"negative": [], "positive": {}},
            "schema_version": "synthetic.v1",
            "tail": [],
            "tail_conflict_count": 0,
        }
    )
    line = worker.canonical_semantic_line(3, 7, result)
    value = json.loads(line)
    assert value == result.as_dict()
    assert line == _canonical_line(value)
    assert line.endswith(b"\n") and not line.endswith(b"\r\n")

    digest = hashlib.sha256()
    for turn in (1, 2):
        digest.update(worker.canonical_semantic_line(1, turn, result))
    expected = hashlib.sha256(
        worker.canonical_semantic_line(1, 1, result)
        + worker.canonical_semantic_line(1, 2, result)
    ).hexdigest()
    assert digest.hexdigest() == expected


@pytest.mark.parametrize(
    ("ordinal", "turn", "result"),
    [
        (0, 1, SimpleNamespace(as_dict=lambda: {})),
        (1, 0, SimpleNamespace(as_dict=lambda: {})),
        (True, 1, SimpleNamespace(as_dict=lambda: {})),
        (1, True, SimpleNamespace(as_dict=lambda: {})),
        (1, 1, object()),
    ],
)
def test_worker_semantic_line_rejects_invalid_coordinates_or_result(
    ordinal: object,
    turn: object,
    result: object,
) -> None:
    with pytest.raises(worker.SparseMultiviewWorkerError):
        worker.canonical_semantic_line(ordinal, turn, result)  # type: ignore[arg-type]


def _worker_common_arguments() -> list[str]:
    return [
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
        "--session-limit",
        "20",
    ]


def test_worker_flags_are_explicit_and_legacy_invocation_remains_cache_off() -> None:
    parser = worker._parser()
    legacy = parser.parse_args(_worker_common_arguments())
    control = parser.parse_args([*_worker_common_arguments(), "--semantic-audit"])
    cached = parser.parse_args(
        [*_worker_common_arguments(), "--semantic-audit", "--semantic-cache"]
    )

    assert legacy.semantic_audit is False and legacy.semantic_cache is False
    assert control.semantic_audit is True and control.semantic_cache is False
    assert cached.semantic_audit is True and cached.semantic_cache is True
    assert legacy.expected_worker_blob is None and legacy.expected_sparse_blob is None
    assert worker.SCHEMA_VERSION == "small-ranker-registry-ca-g0-worker-summary.v1"


def test_worker_source_blob_handshake_and_v219_trace_deny_are_presealed() -> None:
    blobs = {
        "scripts/sparse_multiview_candidate_worker.py": "a" * 40,
        "starter/sparse_multiview.py": "b" * 40,
    }
    command = preflight._worker_command(
        mode="direct",
        nonce="c" * 32,
        reference=preflight.C200_REFERENCE_PATHS[0],
        trace=preflight.RESULT_PATH.with_name("synthetic.jsonl"),
        session_limit=20,
        cached=True,
        implementation_blobs=blobs,
    )
    assert command[command.index("--expected-worker-blob") + 1] == (
        "b44a0c2cbb4c9b4d34aedd6795dbed1ff24a5020"
    )
    assert command[command.index("--expected-sparse-blob") + 1] == (
        "4adf065b0384ab5d45f7bd4582bf7aaf217348a5"
    )

    arguments = SimpleNamespace(
        semantic_audit=True,
        expected_worker_blob="a" * 40,
        expected_sparse_blob="b" * 40,
    )
    with patch.object(
        worker,
        "_raw_git_blob_sha1",
        side_effect=lambda path: "a" * 40
        if Path(path).name == "sparse_multiview_candidate_worker.py"
        else "b" * 40,
    ):
        worker._validate_semantic_source_blobs(arguments)
    arguments.expected_sparse_blob = "c" * 40
    with patch.object(worker, "_raw_git_blob_sha1", return_value="a" * 40):
        with pytest.raises(worker.SparseMultiviewWorkerError):
            worker._validate_semantic_source_blobs(arguments)

    denied = Path(str(worker.V219_CACHE_DENIED_ROOT / "replica.jsonl"))
    with patch.object(
        worker,
        "_require_real_ancestry",
        side_effect=AssertionError("path probe occurred before lexical deny"),
    ):
        with pytest.raises(worker.SparseMultiviewWorkerError) as failure:
            worker._validate_trace_paths(denied, "d" * 32)
    assert failure.value.error_code == "V219_NAMESPACE_DENIED"


def _layer(
    capacity: int,
    *,
    size: int = 1,
    hits: int = 2,
    misses: int = 1,
    evictions: int = 0,
) -> dict[str, int]:
    return {
        "lookups": hits + misses,
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "size": size,
        "capacity": capacity,
        "avoided_operations": hits,
    }


def _core_diagnostics(*, closed: bool, clears: int) -> dict[str, object]:
    size = 0 if closed else 1
    return {
        "enabled": True,
        "closed": closed,
        "clears": clears,
        "fts_route": _layer(preflight.CACHE_CAPACITIES["fts_route"], size=size),
        "product_view": _layer(
            preflight.CACHE_CAPACITIES["product_view"],
            size=size,
        ),
        "mask_decision": _layer(
            preflight.CACHE_CAPACITIES["mask_decision"],
            size=size,
        ),
    }


def _cache_envelope() -> dict[str, object]:
    return {
        "before_close": _core_diagnostics(closed=False, clears=0),
        "after_close": _core_diagnostics(closed=True, clears=1),
    }


def _semantic_trace(*, rows: int = 200, digest: str = "d" * 64) -> dict[str, object]:
    return {"rows": rows, "sha256": digest}


def _worker_receipt(
    *,
    cached: bool,
    nonce: str = "a" * 32,
    session_limit: int = 20,
    semantic_digest: str = "d" * 64,
) -> dict[str, object]:
    records = session_limit * worker.TURN_COUNT
    latency = {
        "count": records,
        "maximum_milliseconds": 4.0,
        "p50_milliseconds": 1.0,
        "p95_milliseconds": 2.0,
    }
    summary: dict[str, object] = {
        "activation": {"activated_records": 1, "inactive_records": records - 1},
        "configuration": {
            "diagnostic_only": True,
            "route_limit": 120,
            "served_top10_unchanged": True,
            "stable_append_after_complete_variable_c200": True,
        },
        "environment": {
            "gpu_peak_bytes": 0,
            "gpu_used": False,
            "network_attempt_count": 0,
        },
        "input_identities": {
            "catalog": {
                "bytes": preflight.CATALOG_BYTES,
                "rows": preflight.CATALOG_ROWS,
                "sha256": preflight.CATALOG_SHA256,
            },
            "sealed_c200_reference": {
                "bytes": preflight.C200_TRACE_BYTES,
                "rows": preflight.C200_TRACE_ROWS,
                "sha256": preflight.C200_TRACE_SHA256,
            },
            "visible_context": {
                "bytes": preflight.CONTEXT_BYTES,
                "rows": preflight.CONTEXT_ROWS,
                "sha256": preflight.CONTEXT_SHA256,
            },
        },
        "latency": {
            "context_container_parse": dict(latency),
            "extra_route_and_mask": dict(latency),
            "per_turn": dict(latency),
        },
        "lifecycle": {"sqlite_closed_before_trace_publish": True},
        "mask": {
            "evaluated_novel_candidates": 1,
            "removed_explicit_conflicts": 0,
            "tail_duplicate_count": 0,
            "tail_explicit_conflict_count": 0,
        },
        "pool_lengths": {},
        "prefix_integrity": {
            "c200_duplicate_count": 0,
            "c200_loss_count": 0,
            "c200_reorder_count": 0,
            "top10_change_count": 0,
        },
        "processed_sessions": session_limit,
        "processed_turns": records,
        "resources": {
            "candidate_cell_ratio_over_c200": 1.1,
            "gpu_peak_bytes": 0,
            "network_attempt_count": 0,
            "peak_working_set_backend": "synthetic",
            "peak_working_set_bytes": 1,
            "trace_byte_ratio_over_c200": 1.1,
            "wall_seconds": 1.0,
        },
        "semantic_trace": _semantic_trace(rows=records, digest=semantic_digest),
        "session_limit": session_limit,
        "source_identities": {},
    }
    if cached:
        summary["cache"] = _cache_envelope()
    return {
        "error_code": "NONE",
        "kind": "receipt",
        "last_completed_session": session_limit,
        "nonce": nonce,
        "phase": "COMPLETE",
        "record_count": records,
        "schema_version": worker.SCHEMA_VERSION,
        "status": "SUCCESS",
        "summary": summary,
        "trace_bytes": 1,
        "trace_sha256": "b" * 64,
    }


def _result(
    name: str,
    *,
    cached: bool,
    wall: float,
    semantic_digest: str = "d" * 64,
) -> dict[str, object]:
    return {
        "receipt": _worker_receipt(
            cached=cached,
            nonce={
                "control": "a" * 32,
                "cached_direct": "b" * 32,
                "cached_module": "c" * 32,
            }[name],
            semantic_digest=semantic_digest,
        ),
        "mode": "module" if name == "cached_module" else "direct",
        "cached": cached,
        "parent_wall_seconds": wall,
        "stderr": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
    }


def _results() -> dict[str, dict[str, object]]:
    return {
        "control": _result("control", cached=False, wall=999.0),
        "cached_direct": _result("cached_direct", cached=True, wall=29.75),
        "cached_module": _result("cached_module", cached=True, wall=30.25),
    }


def test_worker_cache_and_semantic_summaries_pass_privacy_scan() -> None:
    summary = _worker_receipt(cached=True)["summary"]
    worker._receipt_privacy_scan(summary, catalog_ids={_identifier(1, "Z")})
    encoded = _canonical_line(summary)
    assert _identifier(1, "Z").encode("ascii") not in encoded
    assert b"query" not in encoded and b"rule" not in encoded


def test_runner_validates_exact_cache_before_after_close_contract() -> None:
    value = _cache_envelope()
    preflight._validate_cache_diagnostics(value)

    mutations: list[tuple[str, object]] = [
        ("extra", True),
        ("after_open", False),
        ("clear_count", 4),
        ("nonzero_after_size", 1),
        ("bad_capacity", 255),
        ("bad_accounting", 99),
        ("bad_avoided", 99),
    ]
    for mutation, replacement in mutations:
        invalid = json.loads(json.dumps(value))
        if mutation == "extra":
            invalid["extra"] = replacement
        elif mutation == "after_open":
            invalid["after_close"]["closed"] = replacement
        elif mutation == "clear_count":
            invalid["after_close"]["clears"] = replacement
        elif mutation == "nonzero_after_size":
            invalid["after_close"]["fts_route"]["size"] = replacement
        elif mutation == "bad_capacity":
            invalid["before_close"]["fts_route"]["capacity"] = replacement
        elif mutation == "bad_accounting":
            invalid["before_close"]["fts_route"]["lookups"] = replacement
        elif mutation == "bad_avoided":
            invalid["before_close"]["fts_route"]["avoided_operations"] = replacement
        with pytest.raises(preflight.SparseCachePreflightError) as failure:
            preflight._validate_cache_diagnostics(invalid)
        assert failure.value.code == "CACHE_CONTRACT"


def test_runner_worker_receipt_requires_semantic_audit_and_cache_by_mode() -> None:
    for cached, nonce in ((False, "a" * 32), (True, "b" * 32)):
        receipt = _worker_receipt(cached=cached, nonce=nonce)
        validated = preflight._validate_worker_receipt(
            _canonical_line(receipt),
            nonce=nonce,
            session_limit=20,
            cached=cached,
            catalog_ids=frozenset(),
        )
        assert validated["summary"]["semantic_trace"]["rows"] == 200

        missing_semantic = json.loads(json.dumps(receipt))
        del missing_semantic["summary"]["semantic_trace"]
        with pytest.raises(preflight.SparseCachePreflightError) as semantic_failure:
            preflight._validate_worker_receipt(
                _canonical_line(missing_semantic),
                nonce=nonce,
                session_limit=20,
                cached=cached,
                catalog_ids=frozenset(),
            )
        assert semantic_failure.value.code == "WORKER_CONTRACT"

    cached_receipt = _worker_receipt(cached=True, nonce="b" * 32)
    del cached_receipt["summary"]["cache"]
    with pytest.raises(preflight.SparseCachePreflightError):
        preflight._validate_worker_receipt(
            _canonical_line(cached_receipt),
            nonce="b" * 32,
            session_limit=20,
            cached=True,
            catalog_ids=frozenset(),
        )

    control_receipt = _worker_receipt(cached=False, nonce="a" * 32)
    control_receipt["summary"]["cache"] = _cache_envelope()
    with pytest.raises(preflight.SparseCachePreflightError):
        preflight._validate_worker_receipt(
            _canonical_line(control_receipt),
            nonce="a" * 32,
            session_limit=20,
            cached=False,
            catalog_ids=frozenset(),
        )


def test_runner_lexically_denies_v219_namespace_before_any_path_probe() -> None:
    denied = (
        preflight.V219_RESULT_DENIED_PATH,
        preflight.V219_CACHE_DENIED_ROOT,
        preflight.V219_CACHE_DENIED_ROOT / "replica_a.jsonl",
    )
    for path in denied:
        assert preflight._is_v219_denied_lexically(path) is True

    similar = preflight.V219_CACHE_DENIED_ROOT.with_name(
        preflight.V219_CACHE_DENIED_ROOT.name + "_not_the_namespace"
    )
    assert preflight._is_v219_denied_lexically(similar) is False

    with patch.object(Path, "resolve", side_effect=AssertionError("resolve called")), patch.object(
        Path,
        "stat",
        side_effect=AssertionError("stat called"),
    ), patch.object(Path, "exists", side_effect=AssertionError("exists called")), patch.object(
        Path,
        "is_file",
        side_effect=AssertionError("is_file called"),
    ):
        with pytest.raises(preflight.SparseCachePreflightError) as failure:
            preflight._guard_v219_namespace(preflight.V219_CACHE_DENIED_ROOT / "child")
    assert failure.value.code == "V219_NAMESPACE_DENIED"


def test_runner_cache_stage_requires_hits_and_exact_accounting() -> None:
    results = _results()
    preflight._validate_stage_cache_gates(results)

    for name in ("cached_direct", "cached_module"):
        invalid = json.loads(json.dumps(results))
        layer = invalid[name]["receipt"]["summary"]["cache"]["before_close"][
            "fts_route"
        ]
        layer["lookups"] -= layer["hits"]
        layer["hits"] = 0
        layer["avoided_operations"] = 0
        with pytest.raises(preflight.SparseCachePreflightError) as failure:
            preflight._validate_stage_cache_gates(invalid)
        assert failure.value.code == "CACHE_CONTRACT"

    nondeterministic = json.loads(json.dumps(results))
    direct = nondeterministic["cached_direct"]["receipt"]["summary"]["cache"]
    direct["before_close"]["fts_route"]["hits"] += 1
    direct["before_close"]["fts_route"]["lookups"] += 1
    direct["before_close"]["fts_route"]["avoided_operations"] += 1
    direct["after_close"]["fts_route"]["hits"] += 1
    direct["after_close"]["fts_route"]["lookups"] += 1
    direct["after_close"]["fts_route"]["avoided_operations"] += 1
    with pytest.raises(preflight.SparseCachePreflightError) as failure:
        preflight._validate_stage_cache_gates(nondeterministic)
    assert failure.value.code == "CACHE_CONTRACT"


def _frozen_trace(**changes: object) -> preflight.TraceValidation:
    values: dict[str, object] = {
        "trace_sha256": preflight.FROZEN_20_TRACE_SHA256,
        "trace_bytes": preflight.FROZEN_20_TRACE_BYTES,
        "record_count": preflight.FROZEN_20_TRACE_ROWS,
        "activation_turns": preflight.FROZEN_20_ACTIVATED_TURNS,
        "activation_sessions": 20,
        "candidate_cells": 27_500,
        "c200_cells": 20_000,
        "reference_prefix_bytes": 321_000,
        "min_candidates": 100,
        "max_candidates": 320,
    }
    values.update(changes)
    return preflight.TraceValidation(**values)


def test_runner_stage20_is_hard_bound_to_the_frozen_v219_identity() -> None:
    results = _results()
    for result in results.values():
        result["receipt"]["summary"]["activation"] = {
            "activated_records": preflight.FROZEN_20_ACTIVATED_TURNS,
            "inactive_records": (
                preflight.FROZEN_20_TRACE_ROWS
                - preflight.FROZEN_20_ACTIVATED_TURNS
            ),
        }
    preflight._validate_frozen_stage20(_frozen_trace(), results)
    for change in (
        {"trace_sha256": "0" * 64},
        {"trace_bytes": preflight.FROZEN_20_TRACE_BYTES + 1},
        {"record_count": preflight.FROZEN_20_TRACE_ROWS - 1},
    ):
        with pytest.raises(preflight.SparseCachePreflightError) as failure:
            preflight._validate_frozen_stage20(_frozen_trace(**change), results)
        assert failure.value.code == "FROZEN_20_IDENTITY"

    activation_drift = json.loads(json.dumps(results))
    activation_drift["cached_module"]["receipt"]["summary"]["activation"][
        "activated_records"
    ] -= 1
    with pytest.raises(preflight.SparseCachePreflightError) as failure:
        preflight._validate_frozen_stage20(_frozen_trace(), activation_drift)
    assert failure.value.code == "FROZEN_20_IDENTITY"


def _write_trace_fixture(
    path: Path,
    *,
    key: str,
    values: tuple[str, ...],
    tail: tuple[str, ...] = (),
) -> bytes:
    payload = b"".join(
        _canonical_line(
            {
                key: [*values, *tail] if key == "candidates" else list(values),
                "ordinal": 1,
                "turn": turn,
            }
        )
        for turn in range(1, worker.TURN_COUNT + 1)
    )
    path.write_bytes(payload)
    return payload


def _bind_synthetic_trace_receipts(
    results: dict[str, dict[str, object]],
    traces: dict[str, Path],
) -> None:
    for name, path in traces.items():
        payload = path.read_bytes()
        receipt = results[name]["receipt"]
        receipt["record_count"] = worker.TURN_COUNT
        receipt["trace_bytes"] = len(payload)
        receipt["trace_sha256"] = hashlib.sha256(payload).hexdigest()
        receipt["summary"]["semantic_trace"]["rows"] = worker.TURN_COUNT


def test_runner_triplet_gate_requires_exact_trace_and_semantic_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = _pool(100)
    tail = (_identifier(500),)
    catalog_ids = frozenset((*prefix, *tail, _identifier(501)))
    reference_a = tmp_path / "reference_a.jsonl"
    reference_b = tmp_path / "reference_b.jsonl"
    _write_trace_fixture(reference_a, key="c200", values=prefix)
    _write_trace_fixture(reference_b, key="c200", values=prefix)
    traces = {
        "control": tmp_path / "control.jsonl",
        "cached_direct": tmp_path / "cached_direct.jsonl",
        "cached_module": tmp_path / "cached_module.jsonl",
    }
    for path in traces.values():
        _write_trace_fixture(path, key="candidates", values=prefix, tail=tail)
    results = _results()
    _bind_synthetic_trace_receipts(results, traces)

    monkeypatch.setattr(
        preflight,
        "C200_REFERENCE_PATHS",
        (reference_a, reference_b),
    )
    temp_key = preflight._lexical_key(tmp_path)
    preflight._REGISTERED_TEMP_ROOTS.add(temp_key)
    try:
        validation = preflight._trace_triplet_gate(
            results,
            traces,
            catalog_ids,
            session_limit=1,
        )
        assert validation.record_count == worker.TURN_COUNT
        assert validation.activation_turns == worker.TURN_COUNT

        semantic_mismatch = json.loads(json.dumps(results))
        semantic_mismatch["cached_module"]["receipt"]["summary"][
            "semantic_trace"
        ]["sha256"] = "e" * 64
        with pytest.raises(preflight.SparseCachePreflightError) as semantic_failure:
            preflight._trace_triplet_gate(
                semantic_mismatch,
                traces,
                catalog_ids,
                session_limit=1,
            )
        assert semantic_failure.value.code == "SEMANTIC_PARITY"

        _write_trace_fixture(
            traces["cached_module"],
            key="candidates",
            values=prefix,
            tail=(_identifier(501),),
        )
        trace_mismatch = json.loads(json.dumps(results))
        _bind_synthetic_trace_receipts(trace_mismatch, traces)
        with pytest.raises(preflight.SparseCachePreflightError) as trace_failure:
            preflight._trace_triplet_gate(
                trace_mismatch,
                traces,
                catalog_ids,
                session_limit=1,
            )
        assert trace_failure.value.code == "SEMANTIC_PARITY"
    finally:
        preflight._REGISTERED_TEMP_ROOTS.discard(temp_key)


def test_runner_pair_wall_uses_only_unrounded_cached_child_times() -> None:
    results = _results()
    assert preflight._cached_pair_wall(results) == pytest.approx(60.0)

    results["control"]["parent_wall_seconds"] = 100_000.0
    assert preflight._cached_pair_wall(results) == pytest.approx(60.0)

    results["cached_module"]["parent_wall_seconds"] = 30.2500001
    with pytest.raises(preflight.SparseCachePreflightError) as failure:
        preflight._cached_pair_wall(results)
    assert failure.value.code == "RESOURCE_GATE"


def test_runner_frozen_constants_and_implementation_allowlist_match_prereg() -> None:
    assert preflight.FROZEN_20_TRACE_ROWS == 200
    assert preflight.FROZEN_20_TRACE_BYTES == 441_241
    assert preflight.FROZEN_20_TRACE_SHA256 == (
        "e5177f1a69fe1e79d5d9d4729952c9dfcfac0325689aa13e31dc860fbf38e45a"
    )
    assert preflight.FROZEN_20_ACTIVATED_TURNS == 116
    assert preflight.PAIR_WALL_SECONDS_MAXIMUM == 60.0
    assert preflight.CACHE_CAPACITIES == {
        "fts_route": 256,
        "product_view": 4096,
        "mask_decision": 16384,
    }
    assert preflight.IMPLEMENTATION_PATHS == {
        "scripts/v220b_safe_bootstrap.py",
        "scripts/probe_sparse_multiview_cache_preflight.py",
        "tests/test_sparse_multiview_cache.py",
    }
    assert preflight.PREREG_COMMIT == "d073ca9b766b61e529a74e2656c1fcb7bbce9d86"
    git_gate_source = inspect.getsource(preflight._validate_git_checkpoint)
    assert '"status"' not in git_gate_source
    assert '"diff", "--cached"' not in git_gate_source
    assert '"diff", "--name-only"' not in git_gate_source
    assert '"diff-tree"' in inspect.getsource(preflight._diff_paths)
    assert "worktree_paths" in git_gate_source


def test_v220b_formal_run_requires_bootstrap_before_any_probe() -> None:
    with patch.object(
        preflight,
        "_assert_fresh_result",
        side_effect=AssertionError("result path must not be probed"),
    ), patch.object(
        preflight,
        "_install_process_audit_guard",
        side_effect=AssertionError("formal audit must not start"),
    ):
        with patch.object(
            preflight.sys,
            preflight.BOOTSTRAP_ATTESTATION_ATTRIBUTE,
            None,
            create=True,
        ):
            with pytest.raises(preflight.SparseCachePreflightError) as failure:
                preflight.run("0" * 40)
    assert failure.value.code == "BOOTSTRAP_ATTESTATION"


def test_v219_cross_worktree_deny_is_purely_lexical() -> None:
    result = preflight.V219_RESULT_BASENAME
    cache = preflight.V219_CACHE_BASENAME
    denied = (
        rf"D:\tiktok\techjam-v2-20b-sparse-cache\experiments\fast_track\{result}",
        rf"d:/TIKTOK/old-worktree/EXPERIMENTS/FAST_TRACK/{cache}",
        rf"D:/tiktok/arbitrary/experiments/fast_track/{cache}/nested/row.jsonl",
        rf"D:/tiktok/arbitrary/experiments/fast_track/{result}:stream",
        rf"D:/tiktok/arbitrary/experiments/fast_track/{result}. ",
        rf"D:/tiktok/arbitrary/experiments/fast_track/{cache} :stream/nested",
    )
    with patch.object(Path, "stat", side_effect=AssertionError("filesystem probe")), patch.object(
        Path, "resolve", side_effect=AssertionError("filesystem probe")
    ):
        assert all(preflight._is_v219_denied_lexically(path) for path in denied)
        assert not preflight._is_v219_denied_lexically(
            rf"D:/tiktok/arbitrary/experiments/fast_track/{result}.near"
        )
        assert not preflight._is_v219_denied_lexically(
            rf"D:/tiktok/arbitrary/experiments/other/{cache}"
        )


def test_git_command_policy_is_object_only() -> None:
    commit = "a" * 40
    assert preflight._git_command_allowed(("rev-parse", "HEAD"), False)
    assert preflight._git_command_allowed(
        ("diff-tree", "--no-commit-id", "--name-only", "-r", commit), False
    )
    assert preflight._git_command_allowed(("cat-file", "blob", commit), True)
    assert preflight._git_command_allowed(
        ("config", "--local", "--no-includes", "--get-all", "remote.origin.url"),
        False,
    )
    for denied in (
        ("status", "--short"),
        ("diff", "--name-only"),
        ("ls-files", "--others"),
        ("fetch", "origin"),
        ("rev-parse", "HEAD:experiments"),
    ):
        assert not preflight._git_command_allowed(denied, False)


def test_child_environment_drops_inherited_python_and_git_controls() -> None:
    poisoned = {
        "PYTHONPATH": "poison",
        "PYTHONHOME": "poison",
        "GIT_DIR": "poison",
        "GIT_WORK_TREE": "poison",
        "GIT_OBJECT_DIRECTORY": "poison",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "poison",
        "GIT_CONFIG_GLOBAL": "poison",
        "PATH": "poison",
    }
    with patch.dict(os.environ, poisoned, clear=True):
        child = preflight._offline_environment()
    assert not any(key.startswith("GIT_") for key in child)
    assert child["PYTHONHASHSEED"] == "0"
    assert child["PYTHONDONTWRITEBYTECODE"] == "1"
    assert child["PYTHONNOUSERSITE"] == "1"
    assert "poison" not in child.values()


def test_outer_bootstrap_envelope_is_canonical_and_bound() -> None:
    pycache = Path(r"D:\tiktok\.v220b_runtime\synthetic\pycache")
    bootstrap_blob = "a" * 40
    target_blob = "b" * 40
    receipt = {"status": "PASS"}
    envelope = {
        "bootstrap": {
            "mode": "direct",
            "bootstrap_blob": bootstrap_blob,
            "target_blob": target_blob,
            "source_only": True,
            "guarded_path": True,
            "pycache_prefix": pycache.as_posix(),
        },
        "target_exit_code": 0,
        "target_receipt": receipt,
    }
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=preflight._canonical_bytes(envelope) + b"\n", stderr=b""
    )
    assert preflight._validate_bootstrap_envelope(
        completed,
        mode="direct",
        target_blob=target_blob,
        bootstrap_blob=bootstrap_blob,
        pycache_prefix=pycache,
    ) == (0, receipt)

    raw_receipt = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=preflight._canonical_bytes(receipt) + b"\n", stderr=b""
    )
    with pytest.raises(preflight.SparseCachePreflightError):
        preflight._validate_bootstrap_envelope(
            raw_receipt,
            mode="direct",
            target_blob=target_blob,
            bootstrap_blob=bootstrap_blob,
            pycache_prefix=pycache,
        )


def test_runtime_cleanup_identity_is_frozen_after_materialization(tmp_path: Path) -> None:
    runtime_base = tmp_path / "runtime"
    runtime_base.mkdir()
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_bytes(b"pass\n")
    blob = preflight._git_blob_bytes(bootstrap.read_bytes())
    with patch.object(preflight, "RUNTIME_TEMP_ROOT", runtime_base), patch.object(
        preflight, "BOOTSTRAP_PATH", bootstrap
    ):
        launch = preflight._prepare_bootstrap_launch(
            mode="module",
            target_path=tmp_path / "target.py",
            target_module="scripts.synthetic",
            target_blob="b" * 40,
            bootstrap_blob=blob,
            target_argv=("--synthetic",),
        )
        allocated = launch.runtime_root
        assert allocated.exists()
        preflight._cleanup_bootstrap_launch(launch)
        assert not allocated.exists()


def test_prepare_failure_cleans_only_its_fresh_runtime(tmp_path: Path) -> None:
    runtime_base = tmp_path / "runtime"
    runtime_base.mkdir()
    sentinel = runtime_base / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with patch.object(preflight, "RUNTIME_TEMP_ROOT", runtime_base), patch.object(
        preflight,
        "_materialize_bootstrap_launch",
        side_effect=preflight.SparseCachePreflightError("synthetic", "SYNTHETIC"),
    ):
        with pytest.raises(preflight.SparseCachePreflightError):
            preflight._prepare_bootstrap_launch(
                mode="direct",
                target_path=tmp_path / "target.py",
                target_module="scripts.synthetic",
                target_blob="b" * 40,
                bootstrap_blob="a" * 40,
                target_argv=(),
            )
    assert tuple(path.name for path in runtime_base.iterdir()) == ("sentinel.txt",)


def test_nested_receipt_validation_precedes_source_recheck_cleanup_and_timer_stop(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    pycache = tmp_path / "pycache"
    pycache.mkdir()
    bootstrap_blob = "a" * 40
    target_blob = "b" * 40
    receipt = {"status": "PASS"}
    envelope = {
        "bootstrap": {
            "mode": "direct",
            "bootstrap_blob": bootstrap_blob,
            "target_blob": target_blob,
            "source_only": True,
            "guarded_path": True,
            "pycache_prefix": pycache.as_posix(),
        },
        "target_exit_code": 0,
        "target_receipt": receipt,
    }
    launch = preflight.BootstrapLaunch(
        command=("synthetic",),
        environment={},
        runtime_root=tmp_path,
        runtime_snapshot=(0, 0, 0),
        pycache_prefix=pycache,
    )

    def validate_nested(value: dict[str, object]) -> dict[str, object]:
        events.append("nested")
        return value

    with patch.object(
        preflight,
        "_prepare_bootstrap_launch",
        side_effect=lambda **_kwargs: (events.append("prepare") or launch),
    ), patch.object(
        preflight,
        "_run_subprocess",
        side_effect=lambda *_args, **_kwargs: (
            events.append("launch")
            or subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=preflight._canonical_bytes(envelope) + b"\n",
                stderr=b"",
            )
        ),
    ), patch.object(
        preflight,
        "_worktree_blob",
        side_effect=lambda path: (
            events.append("source")
            or (bootstrap_blob if path == preflight.BOOTSTRAP_PATH else target_blob)
        ),
    ), patch.object(
        preflight,
        "_cleanup_bootstrap_launch",
        side_effect=lambda _launch: events.append("cleanup"),
    ):
        observed = preflight._invoke_bootstrap(
            mode="direct",
            target_path=preflight.WORKER_PATH,
            target_module="scripts.sparse_multiview_candidate_worker",
            target_blob=target_blob,
            bootstrap_blob=bootstrap_blob,
            target_argv=(),
            timeout=1.0,
            receipt_validator=validate_nested,
        )
    assert observed["validated_receipt"] == receipt
    assert events == ["prepare", "launch", "nested", "source", "source", "cleanup"]
    assert observed["parent_wall_seconds"] >= 0.0


def test_formal_parent_has_unrounded_aggregate_wall_gate() -> None:
    source = inspect.getsource(preflight.run)
    assert "total_internal_wall > FORMAL_WALL_SECONDS_MAXIMUM" in source
    assert "time.perf_counter() - started > FORMAL_WALL_SECONDS_MAXIMUM" in source


def test_bootstrap_manifest_binds_final_runner_blob_and_rejects_aliases() -> None:
    runner_raw = preflight.RUNNER_PATH.read_bytes()
    assert bootstrap.RUNNER_BLOB == preflight._git_blob_bytes(runner_raw)
    assert bootstrap.RUNNER_BLOB != "0" * 40
    parsed = {
        "--mode": "direct",
        "--target-path": bootstrap.RUNNER_PATH,
        "--target-module": bootstrap.RUNNER_MODULE,
        "--target-blob": bootstrap.RUNNER_BLOB,
        "--bootstrap-blob": "a" * 40,
    }
    assert bootstrap._match_target(parsed)["blob"] == bootstrap.RUNNER_BLOB
    for bad_path in (
        bootstrap.RUNNER_PATH + ".",
        bootstrap.RUNNER_PATH + ":stream",
        bootstrap.PROJECT_ROOT + "/scripts/../scripts/probe_sparse_multiview_cache_preflight.py",
    ):
        with pytest.raises(bootstrap.BootstrapError):
            bootstrap._match_target({**parsed, "--target-path": bad_path})


def test_bootstrap_guarded_path_virtual_root_and_mutators() -> None:
    guarded = bootstrap.GuardedPath(("D:/450/conda/envs/tiktok/Lib",))
    assert bootstrap.PROJECT_ROOT in guarded
    assert bootstrap.PROJECT_ROOT not in tuple(guarded)
    assert guarded[:] == ("D:/450/conda/envs/tiktok/Lib",)
    for operation in (
        lambda: guarded.append(bootstrap.PROJECT_ROOT),
        lambda: guarded.insert(0, bootstrap.PROJECT_ROOT),
        lambda: guarded.extend((bootstrap.PROJECT_ROOT,)),
        lambda: guarded.__iadd__((bootstrap.PROJECT_ROOT,)),
        lambda: guarded.__imul__(2),
        lambda: guarded.__setitem__(slice(None), (bootstrap.PROJECT_ROOT,)),
    ):
        with pytest.raises(bootstrap.BootstrapError):
            operation()
    assert tuple(guarded) == ("D:/450/conda/envs/tiktok/Lib",)


def test_bootstrap_v219_deny_covers_ads_tail_and_descendants() -> None:
    base = bootstrap.PROJECT_ROOT + "/experiments/fast_track/"
    assert bootstrap._is_v219_denied(base + bootstrap._V219_RESULT + ":stream")
    assert bootstrap._is_v219_denied(base + bootstrap._V219_RESULT + ". ")
    assert bootstrap._is_v219_denied(base + bootstrap._V219_CACHE + "/nested.jsonl")
    assert not bootstrap._is_v219_denied(base + bootstrap._V219_RESULT + ".near")
