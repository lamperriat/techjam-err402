"""Isolated target-blind worker for the frozen broad-LIMIT C400 probe.

The worker consumes only the official catalog, the already sealed
identifier-free visible-context cache, and one sealed C200 reference trace.
Its production Agent remains the frozen R08 control configuration.  At the
``_apply_p11`` entry boundary a read-only diagnostic query changes exactly one
variable: broad BM25 LIMIT 120 becomes LIMIT 320 while the existing strict
LIMIT-80 route is reused without another strict query.  Expanded fusion and
coverage use the frozen R08 score and tie order.

The diagnostic C400 is the complete sealed C200 followed by unseen expanded
R08 candidates in their expanded order.  It is never served.  Candidate rows
stay private until the Agent and SQLite are closed, all three inputs are
rehashed, and an exclusive trace is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import c200_candidate_worker as c200_contract  # noqa: E402
from starter.agent import SessionState, _terms  # noqa: E402
from starter.coverage import order_by_query_coverage  # noqa: E402


SCHEMA_VERSION = "small-ranker-c400-candidate-worker-summary.v1"

SESSION_COUNT = 2_000
TURN_COUNT = 10
RECORD_COUNT = SESSION_COUNT * TURN_COUNT
TOP_K = 10

PRODUCTION_BROAD_LIMIT = 120
DIAGNOSTIC_BROAD_LIMIT = 320
STRICT_LIMIT = 80
MAX_CANDIDATES = DIAGNOSTIC_BROAD_LIMIT + STRICT_LIMIT

EXPECTED_CATALOG_BYTES = 60_546_327
EXPECTED_CATALOG_ROWS = 50_000
EXPECTED_CATALOG_SHA256 = (
    "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
)
EXPECTED_CONTEXT_BYTES = 47_168_882
EXPECTED_CONTEXT_ROWS = 2_000
EXPECTED_CONTEXT_TURNS = 20_000
EXPECTED_CONTEXT_SHA256 = (
    "f30a98700da5d480731fe7e82c87c40a22f06de290e069e20dc68f9fefecd20f"
)
EXPECTED_REDACTED_MESSAGE_COUNT = 8

EXPECTED_C200_REFERENCE_BYTES = 32_226_135
EXPECTED_C200_REFERENCE_ROWS = 20_000
EXPECTED_C200_REFERENCE_SHA256 = (
    "a8589749376f48f019997a618481578dde36be4ca1fc723e8ed00056c23e40dc"
)
EXPECTED_C200_CANDIDATE_CELLS = 2_425_785

EXPECTED_EXECUTABLE = Path(r"D:\450\conda\envs\tiktok\python.exe")
EXPECTED_PYTHON = "3.11.16"
EXPECTED_SQLITE = "3.53.4"
EXPECTED_NUMPY = "2.4.6"

REFERENCE_C100_CANONICAL_BYTES = 26_690_930
MAX_TRACE_BYTES_OVER_C100 = int(REFERENCE_C100_CANONICAL_BYTES * 4.1)
MAX_TRACE_BYTES_OVER_C200 = int(EXPECTED_C200_REFERENCE_BYTES * 3.4)
MAX_TRACE_BYTES = min(MAX_TRACE_BYTES_OVER_C100, MAX_TRACE_BYTES_OVER_C200)
MAX_CANDIDATE_CELLS = 8_000_000
MAX_WORKING_SET_BYTES = 2_147_483_648

NONCE_RE = re.compile(r"[0-9a-f]{32}")
CATALOG_IDENTIFIER_TOKEN_RE = re.compile(
    r"(?<![A-Z0-9])[A-Z0-9]{10}(?![A-Z0-9])",
    re.IGNORECASE,
)
EXPANDED_BROAD_SQL = (
    "SELECT rowid, parent_asin FROM products WHERE products MATCH ? "
    "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) "
    f"LIMIT {DIAGNOSTIC_BROAD_LIMIT}"
)


class C400WorkerError(RuntimeError):
    """Raised when a frozen C400 worker invariant is violated."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise C400WorkerError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _c400_shape(candidates: object) -> tuple[str, ...]:
    if not isinstance(candidates, (list, tuple)):
        raise C400WorkerError("C400 is not an ordered sequence")
    result = tuple(candidates)
    if (
        not c200_contract.MIN_CANDIDATES <= len(result) <= MAX_CANDIDATES
        or any(
            not isinstance(identifier, str) or not identifier
            for identifier in result
        )
    ):
        raise C400WorkerError("C400 candidate shape is invalid")
    if len(result) != len(set(result)):
        raise C400WorkerError("C400 candidate shape is invalid")
    return result


def validate_context_turn(turn: object) -> str:
    """Return only the current visible message under the C400 error contract."""

    try:
        return c200_contract.validate_context_turn(turn)
    except c200_contract.C200WorkerError as error:
        raise C400WorkerError("visible context turn is invalid") from error


def append_unseen_tail(
    sealed_c200: object,
    expanded: object,
    catalog_ids: Iterable[str],
    *,
    limit: int = MAX_CANDIDATES,
) -> tuple[str, ...]:
    """Stable-append unseen expanded candidates after the complete sealed C200."""

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit != MAX_CANDIDATES
    ):
        raise C400WorkerError("C400 append limit must remain frozen at 400")
    if not isinstance(sealed_c200, (list, tuple)) or any(
        not isinstance(identifier, str) or not identifier
        for identifier in sealed_c200
    ):
        raise C400WorkerError("sealed C200 is not a valid ordered sequence")
    if not isinstance(expanded, (list, tuple)) or any(
        not isinstance(identifier, str) or not identifier
        for identifier in expanded
    ):
        raise C400WorkerError("expanded candidates are not a valid ordered sequence")
    catalog = (
        catalog_ids
        if isinstance(catalog_ids, (set, frozenset, dict))
        else frozenset(catalog_ids)
    )
    try:
        prefix = c200_contract.validate_c200(sealed_c200, catalog)
    except c200_contract.C200WorkerError as error:
        raise C400WorkerError("sealed C200 candidate surface is invalid") from error
    if any(identifier not in catalog for identifier in expanded):
        raise C400WorkerError("expanded candidates contain a non-catalog identifier")

    result = list(prefix)
    seen = set(prefix)
    for identifier in expanded:
        if identifier in seen:
            continue
        result.append(identifier)
        seen.add(identifier)
        if len(result) == limit:
            break
    return tuple(result)


def validate_c400(
    candidates: object,
    sealed_c200: object,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    """Validate catalog membership and the complete variable-length C200 prefix."""

    catalog = (
        catalog_ids
        if isinstance(catalog_ids, (set, frozenset, dict))
        else frozenset(catalog_ids)
    )
    try:
        reference = c200_contract.validate_c200(sealed_c200, catalog)
    except c200_contract.C200WorkerError as error:
        raise C400WorkerError("sealed C200 candidate surface is invalid") from error
    result = _c400_shape(candidates)
    if (
        len(result) < len(reference)
        or result[: len(reference)] != reference
        or any(identifier not in catalog for identifier in result)
    ):
        raise C400WorkerError("C400 lost, reordered, or invented a candidate")
    return result


def canonical_trace_line(ordinal: int, turn: int, c400: object) -> bytes:
    """Encode the only allowed candidate-bearing C400 trace record."""

    if (
        not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or not 1 <= ordinal <= SESSION_COUNT
        or not isinstance(turn, int)
        or isinstance(turn, bool)
        or not 1 <= turn <= TURN_COUNT
    ):
        raise C400WorkerError("C400 trace coordinate is invalid")
    values = _c400_shape(c400)
    return _canonical_bytes(
        {"c400": list(values), "ordinal": ordinal, "turn": turn}
    ) + b"\n"


def parse_c200_reference_line(
    line: bytes,
    *,
    ordinal: int,
    turn: int,
    catalog_ids: Iterable[str],
) -> tuple[str, ...]:
    """Parse one canonical sealed C200 row at its exact global coordinate."""

    try:
        value = json.loads(
            line.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                C400WorkerError(f"C200 reference contains non-finite {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise C400WorkerError("C200 reference JSONL is invalid") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"c200", "ordinal", "turn"}
        or value.get("ordinal") != ordinal
        or isinstance(value.get("ordinal"), bool)
        or value.get("turn") != turn
        or isinstance(value.get("turn"), bool)
    ):
        raise C400WorkerError("C200 reference schema or order drifted")
    try:
        candidates = c200_contract.validate_c200(value.get("c200"), catalog_ids)
        expected_line = c200_contract.canonical_trace_line(
            ordinal, turn, candidates
        )
    except c200_contract.C200WorkerError as error:
        raise C400WorkerError("C200 reference candidate surface drifted") from error
    if line != expected_line:
        raise C400WorkerError("C200 reference row is not canonical LF JSON")
    return candidates


def _validate_c200_reference_identity(
    path: Path,
    catalog_ids: frozenset[str],
) -> tuple[c200_contract.SourceIdentity, int]:
    c200_contract._require_regular_input(path, "sealed C200 reference")
    before = c200_contract._snapshot(path)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    candidate_cells = 0
    with path.open("rb") as handle:
        for line in handle:
            row_count += 1
            ordinal = (row_count - 1) // TURN_COUNT + 1
            turn = (row_count - 1) % TURN_COUNT + 1
            if row_count > RECORD_COUNT or not line.strip():
                raise C400WorkerError("C200 reference row count or framing drifted")
            candidates = parse_c200_reference_line(
                line,
                ordinal=ordinal,
                turn=turn,
                catalog_ids=catalog_ids,
            )
            digest.update(line)
            byte_count += len(line)
            candidate_cells += len(candidates)
    after = c200_contract._snapshot(path)
    identity = c200_contract.SourceIdentity(
        byte_count, row_count, digest.hexdigest(), after
    )
    if (
        before != after
        or identity.report()
        != {
            "bytes": EXPECTED_C200_REFERENCE_BYTES,
            "rows": EXPECTED_C200_REFERENCE_ROWS,
            "sha256": EXPECTED_C200_REFERENCE_SHA256,
        }
        or candidate_cells != EXPECTED_C200_CANDIDATE_CELLS
    ):
        raise C400WorkerError("sealed C200 reference identity drifted")
    return identity, candidate_cells


def _validate_declared_inputs(args: argparse.Namespace) -> None:
    expected = {
        "catalog_bytes": EXPECTED_CATALOG_BYTES,
        "catalog_rows": EXPECTED_CATALOG_ROWS,
        "catalog_sha256": EXPECTED_CATALOG_SHA256,
        "context_bytes": EXPECTED_CONTEXT_BYTES,
        "context_rows": EXPECTED_CONTEXT_ROWS,
        "context_turns": EXPECTED_CONTEXT_TURNS,
        "context_sha256": EXPECTED_CONTEXT_SHA256,
        "c200_reference_bytes": EXPECTED_C200_REFERENCE_BYTES,
        "c200_reference_rows": EXPECTED_C200_REFERENCE_ROWS,
        "c200_reference_sha256": EXPECTED_C200_REFERENCE_SHA256,
    }
    if any(getattr(args, name) != value for name, value in expected.items()):
        raise C400WorkerError("declared frozen input identity drifted")
    if not NONCE_RE.fullmatch(str(args.nonce)):
        raise C400WorkerError("nonce must be 32 lowercase hexadecimal characters")


def _validate_environment() -> dict[str, Any]:
    base = c200_contract._validate_environment()
    if np.__version__ != EXPECTED_NUMPY:
        raise C400WorkerError("frozen NumPy identity drifted")
    return {
        "executable": base["executable"],
        "python": base["python"],
        "sqlite": base["sqlite"],
        "numpy": np.__version__,
        "pythonhashseed": base["pythonhashseed"],
        "network_attempt_count": 0,
        "gpu_used": False,
        "gpu_peak_bytes": 0,
    }


class C400CaptureAgent(c200_contract.C200CaptureAgent):
    """Frozen control Agent with one read-only diagnostic broad expansion."""

    def _expanded_r08(
        self,
        rankings: Mapping[str, Sequence[str]],
        candidate_rowids: Mapping[str, int],
        query_terms: Sequence[str],
    ) -> tuple[tuple[str, ...], int, frozenset[str]]:
        broad_expression = self._fts_expression(list(query_terms))
        if not broad_expression:
            raise C400WorkerError("diagnostic broad expression is empty")
        rows = self.connection.execute(
            EXPANDED_BROAD_SQL,
            (broad_expression,),
        ).fetchall()
        expanded_broad = tuple(str(row[1]) for row in rows)
        production_broad = tuple(str(value) for value in rankings.get("broad", ()))
        strict = tuple(str(value) for value in rankings.get("strict", ()))
        if (
            len(production_broad) > PRODUCTION_BROAD_LIMIT
            or len(expanded_broad) > DIAGNOSTIC_BROAD_LIMIT
            or len(strict) > STRICT_LIMIT
            or len(expanded_broad) != len(set(expanded_broad))
            or len(strict) != len(set(strict))
            or expanded_broad[: len(production_broad)] != production_broad
        ):
            raise C400WorkerError("diagnostic broad or strict route identity drifted")

        expanded_rowids = dict(candidate_rowids)
        expanded_rowids.update((str(row[1]), int(row[0])) for row in rows)
        if any(identifier not in expanded_rowids for identifier in strict):
            raise C400WorkerError("reused strict route lost its production rowid")

        broad_rank = {
            identifier: rank
            for rank, identifier in enumerate(expanded_broad, start=1)
        }
        strict_rank = {
            identifier: rank for rank, identifier in enumerate(strict, start=1)
        }
        union = dict.fromkeys([*expanded_broad, *strict])
        fused = sorted(
            union,
            key=lambda identifier: (
                -self._fusion_score(identifier, broad_rank, strict_rank),
                broad_rank.get(identifier, 10**9),
                identifier,
            ),
        )
        if len(fused) > MAX_CANDIDATES:
            raise C400WorkerError("diagnostic expanded union exceeded 400")
        searchable_fields = self._load_coverage_fields(fused, expanded_rowids)
        expanded_final, _diagnostics = order_by_query_coverage(
            query_terms,
            fused,
            searchable_fields,
            _terms,
        )
        if (
            len(expanded_final) != len(fused)
            or len(expanded_final) != len(set(expanded_final))
            or set(expanded_final) != set(fused)
        ):
            raise C400WorkerError("diagnostic coverage ordering changed membership")
        return (
            tuple(expanded_final),
            len(expanded_broad),
            frozenset(expanded_rowids),
        )

    def _apply_p11(
        self,
        state: SessionState,
        rankings: dict[str, list[str]],
        candidate_rowids: dict[str, int],
        query_terms: list[str],
    ) -> tuple[list[str], dict[str, Any]]:
        ranking_snapshot = {
            route: tuple(values) for route, values in rankings.items()
        }
        rowid_snapshot = dict(candidate_rowids)
        production_final = tuple(str(value) for value in rankings.get("final", ()))

        expansion_started = time.perf_counter_ns()
        expanded_final, expanded_broad_count, expanded_catalog_ids = self._expanded_r08(
            rankings,
            candidate_rowids,
            query_terms,
        )
        c400 = append_unseen_tail(
            production_final,
            expanded_final,
            expanded_catalog_ids,
            limit=MAX_CANDIDATES,
        )
        expansion_milliseconds = (
            time.perf_counter_ns() - expansion_started
        ) / 1_000_000.0
        served, diagnostics = super()._apply_p11(
            state,
            rankings,
            candidate_rowids,
            query_terms,
        )
        if (
            {route: tuple(values) for route, values in rankings.items()}
            != ranking_snapshot
            or candidate_rowids != rowid_snapshot
            or tuple(served) != production_final
        ):
            raise C400WorkerError("diagnostic expansion mutated production ranking")
        capture = self._c200_last_capture
        if (
            not isinstance(capture, dict)
            or tuple(capture.get("c200", ())) != production_final
        ):
            raise C400WorkerError("production C200 capture drifted")
        capture.update(
            {
                "c400": c400,
                "expansion_milliseconds": expansion_milliseconds,
                "expanded_broad_count": expanded_broad_count,
            }
        )
        return served, diagnostics


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise C400WorkerError("cannot summarize an empty measurement")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _latency_summary(
    values: Sequence[float],
    *,
    unit: str,
) -> dict[str, int | float]:
    if not values or unit != "milliseconds":
        raise C400WorkerError("latency surface is invalid")
    numbers = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in numbers):
        raise C400WorkerError("latency contains a non-finite measurement")
    return {
        "count": len(numbers),
        "p50_milliseconds": round(_nearest_rank(numbers, 0.50), 6),
        "p95_milliseconds": round(_nearest_rank(numbers, 0.95), 6),
        "maximum_milliseconds": round(max(numbers), 6),
    }


def _pool_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise C400WorkerError("candidate pool surface is empty")
    cells = sum(values)
    if cells > MAX_CANDIDATE_CELLS:
        raise C400WorkerError("C400 candidate cells exceed the frozen budget")
    return {
        "min": min(values),
        "p50": int(_nearest_rank(values, 0.50)),
        "p95": int(_nearest_rank(values, 0.95)),
        "max": max(values),
        "mean": round(statistics.fmean(values), 6),
        "records": len(values),
        "candidate_cells": cells,
    }


def _validate_end_identity(
    path: Path,
    label: str,
    expected: c200_contract.SourceIdentity,
) -> None:
    observed = c200_contract._raw_jsonl_identity(path, label)
    if observed.report() != expected.report() or observed.snapshot != expected.snapshot:
        raise C400WorkerError(f"{label} changed during the worker run")


def _receipt_privacy_scan(
    value: object,
    *,
    non_b0_catalog_ids: Iterable[str] = (),
) -> None:
    forbidden = c200_contract.FORBIDDEN_CONTEXT_KEYS | frozenset(
        {
            "c200",
            "c400",
            "message",
            "messages",
            "per_session",
            "membership_vector",
        }
    )
    if c200_contract._walk_keys(value) & forbidden:
        raise C400WorkerError("aggregate receipt contains a forbidden key")
    payload = _canonical_bytes(value).decode("utf-8")
    if c200_contract.ASIN_SHAPE_RE.search(payload):
        raise C400WorkerError("aggregate receipt contains an identifier-shaped token")
    exceptions = frozenset(str(identifier).casefold() for identifier in non_b0_catalog_ids)
    if len(exceptions) not in {0, 6}:
        raise C400WorkerError("non-B0 catalog exception surface drifted")
    tokens = {
        match.group(0).casefold()
        for match in CATALOG_IDENTIFIER_TOKEN_RE.finditer(payload)
    }
    if tokens & exceptions:
        raise C400WorkerError("aggregate receipt contains a non-B0 catalog identifier")


def _agent_factory(catalog_path: Path) -> C400CaptureAgent:
    return C400CaptureAgent(catalog_path)


def run(
    args: argparse.Namespace,
    *,
    agent_factory: Callable[[Path], Any] = _agent_factory,
    network_audit: c200_contract.OfflineNetworkAudit | None = None,
) -> dict[str, Any]:
    _validate_declared_inputs(args)
    environment = _validate_environment()
    c200_contract._check_trace_output(args.trace_output)

    audit = network_audit or c200_contract.OfflineNetworkAudit()
    sys.addaudithook(audit.hook)

    catalog_identity, catalog_ids = c200_contract._catalog_identity(args.catalog)
    context_identity, context_turns, redacted_messages = (
        c200_contract._validate_context_identity(args.context, catalog_ids)
    )
    reference_identity, reference_cells = _validate_c200_reference_identity(
        args.c200_reference,
        catalog_ids,
    )
    if catalog_identity.report() != {
        "bytes": EXPECTED_CATALOG_BYTES,
        "rows": EXPECTED_CATALOG_ROWS,
        "sha256": EXPECTED_CATALOG_SHA256,
    }:
        raise C400WorkerError("official catalog identity drifted")
    if (
        context_identity.report()
        != {
            "bytes": EXPECTED_CONTEXT_BYTES,
            "rows": EXPECTED_CONTEXT_ROWS,
            "sha256": EXPECTED_CONTEXT_SHA256,
        }
        or context_turns != EXPECTED_CONTEXT_TURNS
        or redacted_messages != EXPECTED_REDACTED_MESSAGE_COUNT
        or reference_cells != EXPECTED_C200_CANDIDATE_CELLS
    ):
        raise C400WorkerError("sealed target-free source identity drifted")

    trace_lines: list[bytes] = []
    trace_digest = hashlib.sha256()
    trace_bytes = 0
    pool_lengths: list[int] = []
    respond_milliseconds: list[float] = []
    expansion_milliseconds: list[float] = []
    expanded_broad_lengths: list[int] = []
    agent: Any | None = None
    agent_closed = False
    sqlite_closed = False
    processed_sessions = 0
    reference_rows = 0
    try:
        agent = agent_factory(args.catalog)
        with args.context.open("rb") as context_handle, args.c200_reference.open(
            "rb"
        ) as reference_handle:
            for ordinal, context_line in enumerate(context_handle, start=1):
                if ordinal > SESSION_COUNT:
                    raise C400WorkerError("visible context has excess sessions")
                contexts = c200_contract._parse_context_container(
                    context_line, catalog_ids
                )
                session_id = f"conversation_{ordinal}"
                agent.reset(session_id, {})
                try:
                    for turn, context in enumerate(contexts, start=1):
                        reference_line = reference_handle.readline()
                        if not reference_line:
                            raise C400WorkerError("sealed C200 reference ended early")
                        reference_rows += 1
                        sealed_c200 = parse_c200_reference_line(
                            reference_line,
                            ordinal=ordinal,
                            turn=turn,
                            catalog_ids=catalog_ids,
                        )

                        respond_started = time.perf_counter_ns()
                        response = agent.respond(
                            session_id,
                            validate_context_turn(context),
                            turn,
                            TOP_K,
                        )
                        respond_milliseconds.append(
                            (time.perf_counter_ns() - respond_started) / 1_000_000.0
                        )
                        capture = agent.take_last_capture(session_id)
                        production_c200 = tuple(capture.get("c200", ()))
                        c400 = validate_c400(
                            capture.get("c400", ()),
                            sealed_c200,
                            catalog_ids,
                        )
                        latency = capture.get("expansion_milliseconds")
                        expanded_broad_count = capture.get("expanded_broad_count")
                        if (
                            production_c200 != sealed_c200
                            or not isinstance(latency, (int, float))
                            or isinstance(latency, bool)
                            or not math.isfinite(float(latency))
                            or float(latency) < 0.0
                            or not isinstance(expanded_broad_count, int)
                            or isinstance(expanded_broad_count, bool)
                            or not 0 <= expanded_broad_count <= DIAGNOSTIC_BROAD_LIMIT
                        ):
                            raise C400WorkerError(
                                "production C200 or diagnostic capture drifted"
                            )
                        if c200_contract._served_identifiers(response) != sealed_c200[:TOP_K]:
                            raise C400WorkerError("served Top10 differs from sealed C200")

                        trace_line = canonical_trace_line(ordinal, turn, c400)
                        trace_bytes += len(trace_line)
                        if trace_bytes > MAX_TRACE_BYTES:
                            raise C400WorkerError("C400 trace exceeds frozen byte budget")
                        trace_lines.append(trace_line)
                        trace_digest.update(trace_line)
                        pool_lengths.append(len(c400))
                        expansion_milliseconds.append(float(latency))
                        expanded_broad_lengths.append(expanded_broad_count)
                finally:
                    agent.drop_session(session_id)
                processed_sessions += 1
            if reference_handle.read(1) != b"":
                raise C400WorkerError("sealed C200 reference has excess rows")
        if (
            processed_sessions != SESSION_COUNT
            or reference_rows != RECORD_COUNT
            or len(trace_lines) != RECORD_COUNT
        ):
            raise C400WorkerError("C400 trajectory is incomplete")
    finally:
        if agent is not None:
            agent.close()
            agent_closed = bool(getattr(agent, "_closed", True))
            sqlite_closed = bool(getattr(agent, "_c200_sqlite_closed", True))

    if not agent_closed or not sqlite_closed:
        raise C400WorkerError("Agent or SQLite was not closed before publication")
    if audit.attempt_count != 0:
        raise C400WorkerError("network activity was attempted")

    _validate_end_identity(args.catalog, "catalog", catalog_identity)
    _validate_end_identity(args.context, "visible context", context_identity)
    _validate_end_identity(
        args.c200_reference,
        "sealed C200 reference",
        reference_identity,
    )

    peak_rss, _rss_backend = c200_contract._peak_rss_bytes()
    if peak_rss is None or not 0 < peak_rss <= MAX_WORKING_SET_BYTES:
        raise C400WorkerError("worker lifetime peak working set is outside budget")

    c200_contract._publish_trace_exclusive(args.trace_output, trace_lines)
    published = c200_contract._raw_jsonl_identity(
        args.trace_output, "published C400 trace"
    )
    if (
        published.bytes != trace_bytes
        or published.rows != RECORD_COUNT
        or published.sha256 != trace_digest.hexdigest()
    ):
        raise C400WorkerError("published C400 trace identity drifted")

    environment["network_attempt_count"] = audit.attempt_count
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "configuration": {
            "p11_mode": "control",
            "small_ranker_mode": "off",
            "question_policy": "fast",
            "rerank_mode": "off",
            "retrieval_mode": "coverage",
            "production_broad_limit": PRODUCTION_BROAD_LIMIT,
            "diagnostic_broad_limit": DIAGNOSTIC_BROAD_LIMIT,
            "strict_limit": STRICT_LIMIT,
            "stable_append": True,
        },
        "pool_lengths": _pool_summary(pool_lengths),
        "expanded_broad_lengths": _pool_summary(expanded_broad_lengths),
        "latency": {
            "respond": _latency_summary(
                respond_milliseconds,
                unit="milliseconds",
            ),
            "expansion": _latency_summary(
                expansion_milliseconds,
                unit="milliseconds",
            ),
        },
        "resources": {"peak_working_set_bytes": peak_rss},
        "lifecycle": {
            "agent_closed_before_trace_publish": agent_closed,
            "sqlite_closed_before_trace_publish": sqlite_closed,
            "catalog_unchanged_before_trace_publish": True,
            "context_unchanged_before_trace_publish": True,
            "c200_reference_unchanged_before_trace_publish": True,
        },
    }
    receipt: dict[str, Any] = {
        "kind": "receipt",
        "nonce": args.nonce,
        "trace_sha256": trace_digest.hexdigest(),
        "trace_bytes": trace_bytes,
        "record_count": len(trace_lines),
        "summary": summary,
    }
    non_b0_catalog_ids = frozenset(
        identifier
        for identifier in catalog_ids
        if c200_contract.ASIN_SHAPE_RE.fullmatch(identifier) is None
    )
    if len(non_b0_catalog_ids) != 6:
        raise C400WorkerError("non-B0 catalog identifier count drifted")
    _receipt_privacy_scan(
        receipt,
        non_b0_catalog_ids=non_b0_catalog_ids,
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-bytes", type=int, required=True)
    parser.add_argument("--catalog-rows", type=int, required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--context-bytes", type=int, required=True)
    parser.add_argument("--context-rows", type=int, required=True)
    parser.add_argument("--context-turns", type=int, required=True)
    parser.add_argument("--context-sha256", required=True)
    parser.add_argument("--c200-reference", type=Path, required=True)
    parser.add_argument("--c200-reference-bytes", type=int, required=True)
    parser.add_argument("--c200-reference-rows", type=int, required=True)
    parser.add_argument("--c200-reference-sha256", required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    receipt = run(_parser().parse_args(argv))
    sys.stdout.buffer.write(_canonical_bytes(receipt) + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
