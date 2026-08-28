"""Experiment-only R08 Top-10 scorer roles for P11.

The served Agent remains unchanged.  C00 is an exact ``coverage/off/fast``
control, S00 computes the frozen scorer without changing output, and R01 may
only permute the already-served R08 Top 10.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from starter.agent import Agent, SessionState, _terms
from starter.attributes import normalize_value
from starter.p11_features import (
    P11FeatureStore,
    P11RerankResult,
    PositiveConstraint,
    REGISTRY_SHA256 as FEATURE_REGISTRY_SHA256,
    SCHEMA_VERSION as FEATURE_SCHEMA_VERSION,
    SCORER_VERSION,
    SEMANTICS_SHA256 as FEATURE_SEMANTICS_SHA256,
    rerank_top10_preserving_membership,
)
from starter.p8_negative import compile_negative_constraints
from starter.p9_evidence import OFFICIAL_CATALOG_ROWS, OFFICIAL_CATALOG_SHA256


SCHEMA_VERSION = "p11.top10-lab.v1"
CONTROL_ID = "P11.C00.r08_coverage"
SHADOW_ID = "P11.S00.top10_linear_shadow"
ACTIVE_ID = "P11.R01.top10_linear"
ROLE_IDS = (CONTROL_ID, SHADOW_ID, ACTIVE_ID)

_HARD_MARKER_RE = re.compile(
    r"\b(?:key\s+requirement|what\s+matters|what\s+i\s+need|"
    r"must\s+have|required|requirement|need)\b",
    re.IGNORECASE,
)
_HARD_PREFIX_RE = re.compile(
    r"^.*\b(?:key\s+requirement(?:\s+is)?|what\s+matters(?:\s+is)?|"
    r"what\s+i\s+need(?:\s+is)?|must\s+have|required|requirement(?:\s+is)?|"
    r"need)\b\s*:?[ ]*",
    re.IGNORECASE | re.DOTALL,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_record(digest: Any, record: Mapping[str, object]) -> None:
    digest.update(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _latest_hard_clause_terms(state: SessionState) -> tuple[str, ...]:
    """Extract one current-goal hard clause once per turn, outside scoring loops."""

    source_turns = [
        int(record.source_turn)
        for record in state.slot_ledger.active_records()
        if record.polarity > 0
        and record.hardness == "hard"
        and record.version == state.version
        and record.source_turn >= state.version_anchor_turn
    ]
    if not source_turns:
        return ()
    source_turn = max(source_turns)
    if not 1 <= source_turn <= len(state.messages):
        return ()
    message = state.messages[source_turn - 1]
    if _HARD_MARKER_RE.search(message) is None:
        return ()
    fragment = _HARD_PREFIX_RE.sub("", message).strip(" .,:;-")
    excluded = {normalize_value(value) for value in state.excluded_terms}
    return tuple(
        term for term in _terms(fragment) if normalize_value(term) not in excluded
    )[:12]


@dataclass(slots=True)
class P11Stats:
    turns: int = 0
    activations: int = 0
    proposed_changes: int = 0
    output_changes: int = 0
    fallbacks: int = 0
    exception_count: int = 0
    sidecar_rows_read: int = 0
    maximum_rows_per_fetch: int = 0
    top10_membership_violation_count: int = 0
    tail_change_count: int = 0
    hard_clause_turns: int = 0
    reason_counts: Counter[str] = field(default_factory=Counter)
    conflict_state_counts: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["reason_counts"] = dict(sorted(self.reason_counts.items()))
        value["conflict_state_counts"] = dict(
            sorted(self.conflict_state_counts.items())
        )
        return value


class P11Agent(Agent):
    """Exact R08 control with optional shadow/active within-set reranking."""

    def __init__(
        self,
        catalog_path: str | Path,
        role: str,
        *,
        sidecar_path: str | Path | None = None,
        expected_sidecar: tuple[int, str] | None = None,
    ) -> None:
        if role not in ROLE_IDS:
            raise ValueError(f"unknown P11 role: {role}")
        self.p11_role = role
        self.p11_stats = P11Stats()
        self._feature_store: P11FeatureStore | None = None
        self._sidecar_configuration: dict[str, object] = {"sidecar_opened": False}
        self._candidate_rowids: Mapping[str, int] | None = None
        self._state_session_indexes: dict[int, int] = {}
        self._next_session_index = 1
        self._audit_digest = hashlib.sha256()
        self._response_digest = hashlib.sha256()
        self._integrity_errors: list[str] = []
        super().__init__(
            catalog_path,
            llm_client=None,
            question_policy="fast",
            rerank_mode="off",
            retrieval_mode="coverage",
        )
        if role != CONTROL_ID:
            self._open_sidecar(sidecar_path, expected_sidecar)

    def _open_sidecar(
        self,
        sidecar_path: str | Path | None,
        expected_sidecar: tuple[int, str] | None,
    ) -> None:
        try:
            if sidecar_path is None or expected_sidecar is None:
                raise FileNotFoundError("P11 sidecar and identity are required")
            path = Path(sidecar_path).resolve()
            identity = (path.stat().st_size, _sha256_file(path))
            if identity != expected_sidecar:
                raise ValueError("P11 sidecar identity mismatch")
            self._feature_store = P11FeatureStore(
                path,
                expected_catalog_sha256=OFFICIAL_CATALOG_SHA256,
                expected_catalog_rows=OFFICIAL_CATALOG_ROWS,
            )
            self._sidecar_configuration = {
                "sidecar_opened": True,
                "sidecar_identity_verified": True,
                "sidecar_bytes": identity[0],
                "sidecar_sha256": identity[1],
            }
        except Exception as error:
            self._feature_store = None
            self._sidecar_configuration = {
                "sidecar_opened": False,
                "sidecar_identity_verified": False,
            }
            self._integrity_errors.append(f"sidecar_init:{type(error).__name__}")

    def reset(self, session_id: str, user_profile: dict) -> None:
        with self._lock:
            previous = self._sessions.get(session_id)
            if previous is not None:
                self._state_session_indexes.pop(id(previous), None)
            super().reset(session_id, user_profile)
            state = self._sessions[session_id]
            self._state_session_indexes[id(state)] = self._next_session_index
            self._next_session_index += 1

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            super().drop_session(session_id)
            if state is not None:
                self._state_session_indexes.pop(id(state), None)

    def _load_coverage_fields(
        self,
        parent_asins: list[str],
        candidate_rowids: dict[str, int],
    ) -> dict[str, tuple[str, str, str, str, str, str]]:
        if self.p11_role != CONTROL_ID:
            self._candidate_rowids = candidate_rowids
        return super()._load_coverage_fields(parent_asins, candidate_rowids)

    @staticmethod
    def _positive_constraints(state: SessionState) -> tuple[PositiveConstraint, ...]:
        return tuple(
            PositiveConstraint(
                slot=record.slot,
                value=record.value,
                hardness=record.hardness,
                source_turn=record.source_turn,
                version=record.version,
            )
            for record in state.slot_ledger.active_records()
            if record.polarity > 0
        )

    def _score(self, state: SessionState, rankings: Mapping[str, list[str]]) -> P11RerankResult:
        baseline = rankings["final"]
        if not baseline:
            return P11RerankResult((), False, "empty", False, {})
        if self._feature_store is None:
            raise RuntimeError("P11 feature store is unavailable")
        rowids = self._candidate_rowids
        if rowids is None:
            raise RuntimeError("P11 catalog rowids were not captured")
        head = baseline[:10]
        requested = []
        for identifier in head:
            rowid = rowids.get(identifier)
            if rowid is None:
                raise ValueError("P11 candidate rowid is missing")
            requested.append((int(rowid), identifier))
        query_terms = self._query_terms(state)
        batch = self._feature_store.fetch_top10(requested, query_terms)
        self.p11_stats.sidecar_rows_read += len(requested)
        self.p11_stats.maximum_rows_per_fetch = max(
            self.p11_stats.maximum_rows_per_fetch, len(requested)
        )
        negatives = compile_negative_constraints(
            state.slot_ledger.records,
            current_version=state.version,
        ).constraints
        hard_terms = _latest_hard_clause_terms(state)
        self.p11_stats.hard_clause_turns += int(bool(hard_terms))
        return rerank_top10_preserving_membership(
            baseline,
            batch,
            query_terms=query_terms,
            broad_ranks={
                identifier: rank
                for rank, identifier in enumerate(rankings["broad"], start=1)
            },
            strict_ranks={
                identifier: rank
                for rank, identifier in enumerate(rankings["strict"], start=1)
            },
            fused_ranks={
                identifier: rank
                for rank, identifier in enumerate(rankings["fused"], start=1)
            },
            positive_constraints=self._positive_constraints(state),
            negative_constraints=negatives,
            query_subtypes=self._feature_store.resolve_query_subtypes(
                state.category_text
            ),
            hard_clause_terms=hard_terms,
            current_turn=max(1, len(state.messages)),
            current_version=state.version,
        )

    def _record_result(
        self,
        state: SessionState,
        baseline: list[str],
        result: P11RerankResult,
        served: list[str],
    ) -> None:
        proposed = list(result.identifiers)
        original_head = baseline[:10]
        proposed_head = proposed[:10]
        membership_violation = set(original_head) != set(proposed_head)
        tail_change = baseline[10:] != proposed[10:]
        self.p11_stats.activations += int(result.reason == "scored")
        self.p11_stats.proposed_changes += int(proposed_head != original_head)
        self.p11_stats.output_changes += int(served[:10] != original_head)
        self.p11_stats.fallbacks += int(result.fallback)
        self.p11_stats.exception_count += int(result.fallback)
        self.p11_stats.top10_membership_violation_count += int(membership_violation)
        self.p11_stats.tail_change_count += int(tail_change)
        self.p11_stats.reason_counts[result.reason] += 1
        conflict_counts = Counter(
            breakdown.conflict_state for breakdown in result.breakdowns.values()
        )
        self.p11_stats.conflict_state_counts.update(conflict_counts)
        _stream_record(
            self._audit_digest,
            {
                "session_index": self._state_session_indexes.get(id(state), -1),
                "turn": len(state.messages),
                "reason": result.reason,
                "fallback": result.fallback,
                "proposed_change": proposed_head != original_head,
                "output_change": served[:10] != original_head,
                "membership_preserved": not membership_violation,
                "tail_preserved": not tail_change,
                "conflict_counts": dict(sorted(conflict_counts.items())),
            },
        )

    def _rank_candidates(self, state: SessionState) -> dict[str, list[str]]:
        self._candidate_rowids = None
        try:
            rankings = super()._rank_candidates(state)
            baseline = list(rankings["final"])
            self.p11_stats.turns += 1
            if self.p11_role == CONTROL_ID:
                result = P11RerankResult(
                    tuple(baseline), False, "control", False, {}
                )
            else:
                try:
                    result = self._score(state, rankings)
                except Exception as error:
                    result = P11RerankResult(
                        tuple(baseline),
                        True,
                        f"fallback:{type(error).__name__}",
                        False,
                        {},
                    )
            proposed = list(result.identifiers)
            valid = (
                len(proposed) == len(baseline)
                and set(proposed[:10]) == set(baseline[:10])
                and proposed[10:] == baseline[10:]
            )
            if not valid:
                result = P11RerankResult(
                    tuple(baseline), True, "fallback:BoundaryViolation", False, {}
                )
                proposed = baseline
            served = proposed if self.p11_role == ACTIVE_ID else baseline
            try:
                self._record_result(state, baseline, result, served)
            except Exception as error:
                self._integrity_errors.append(
                    f"result_record:{type(error).__name__}"
                )
                served = baseline
                self.p11_stats.exception_count += 1
                self.p11_stats.fallbacks += 1
                self.p11_stats.reason_counts["fallback:InstrumentationError"] += 1
            return {**rankings, "final": list(served)}
        finally:
            self._candidate_rowids = None

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        with self._lock:
            state = self._sessions.get(session_id)
            session_index = (
                self._state_session_indexes.get(id(state)) if state is not None else None
            )
            response = super().respond(session_id, user_message, turn, top_k)
            if session_index is None:
                self._integrity_errors.append("response_without_session_index")
                return response
            try:
                _stream_record(
                    self._response_digest,
                    {
                        "session_index": session_index,
                        "turn": turn,
                        "response": response,
                    },
                )
            except Exception as error:
                self._integrity_errors.append(
                    f"response_stream:{type(error).__name__}"
                )
            return response

    def export_p11_blind_capture(self) -> dict[str, Any]:
        with self._lock:
            functions = {
                "latest_hard_clause_terms": _latest_hard_clause_terms,
                "P11FeatureStore.fetch_top10": P11FeatureStore.fetch_top10,
                "rerank_top10_preserving_membership": (
                    rerank_top10_preserving_membership
                ),
                "P11Agent._rank_candidates": P11Agent._rank_candidates,
            }
            return {
                "schema_version": SCHEMA_VERSION,
                "role": self.p11_role,
                "configuration": {
                    "retrieval_mode": "coverage",
                    "rerank_mode": "off",
                    "question_policy": "fast",
                    "top10_membership_preserved": True,
                    "tail_preserved": True,
                    "target_blind": True,
                    "label_free": True,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "scorer_version": SCORER_VERSION,
                    "feature_registry_sha256": FEATURE_REGISTRY_SHA256,
                    "feature_semantics_sha256": FEATURE_SEMANTICS_SHA256,
                    **self._sidecar_configuration,
                },
                "stats": self.p11_stats.as_dict(),
                "integrity_errors": list(self._integrity_errors),
                "hashes": {
                    "audit_sha256": self._audit_digest.copy().hexdigest(),
                    "responses_sha256": self._response_digest.copy().hexdigest(),
                },
                "function_hashes": {
                    name: hashlib.sha256(
                        inspect.getsource(function).encode("utf-8")
                    ).hexdigest()
                    for name, function in sorted(functions.items())
                },
            }

    def close(self) -> None:
        with self._lock:
            try:
                if self._feature_store is not None:
                    self._feature_store.close()
                    self._feature_store = None
            finally:
                self.connection.close()


def create_p11_agent(
    catalog_path: str | Path,
    role: str,
    *,
    sidecar_path: str | Path | None = None,
    expected_sidecar: tuple[int, str] | None = None,
) -> P11Agent:
    return P11Agent(
        catalog_path,
        role,
        sidecar_path=sidecar_path,
        expected_sidecar=expected_sidecar,
    )


__all__ = [
    "ACTIVE_ID",
    "CONTROL_ID",
    "P11Agent",
    "P11Stats",
    "ROLE_IDS",
    "SCHEMA_VERSION",
    "SHADOW_ID",
    "create_p11_agent",
]
