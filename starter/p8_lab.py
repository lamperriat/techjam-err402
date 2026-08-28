"""Isolated P8 lab for high-confidence explicit-negative execution.

The submitted Agent is unchanged. C00 is the exact served coverage/off/fast
control, S00 computes the same proposal without serving it, and R01 alone may
serve the stable compatibility partition.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from starter.agent import Agent, SessionState
from starter.attributes import ProductAttributeView, build_product_attribute_view
from starter.p8_negative import (
    ALLOWED_NEGATIVE_SLOTS,
    CANDIDATE_POOL,
    COMPATIBLE,
    EXPLICIT_VIOLATION,
    MIN_EVIDENCE_CONFIDENCE,
    SCHEMA_VERSION as NEGATIVE_SCHEMA_VERSION,
    UNKNOWN,
    NegativeCompilation,
    NegativePartition,
    classify_candidate,
    compile_negative_constraints,
    stable_negative_partition,
)


SCHEMA_VERSION = "p8.explicit-negative-lab.v1"
C00 = "P8.C00.r08_coverage"
S00 = "P8.S00.explicit_negative_shadow"
R01 = "P8.R01.explicit_negative_partition"
CONTROL_ID = C00
SHADOW_ID = S00
ACTIVE_ID = R01
_AUDIT_RECORD_LIMIT = 2_000


@dataclass(frozen=True, slots=True)
class P8Spec:
    variant_id: str
    family: str
    mechanism: str
    stage_graph: tuple[str, ...]
    description: str
    parameters: tuple[tuple[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_NEGATIVE_PARAMETERS = (
    ("required_status", "active"),
    ("required_hardness", "hard"),
    ("required_polarity", -1),
    ("required_confidence", 1.0),
    ("required_source", "excluded_terms"),
    ("required_goal_version", "current"),
    ("allowed_slots", ",".join(sorted(ALLOWED_NEGATIVE_SLOTS))),
    ("value_shape", "single_normalized_ascii_token"),
    ("candidate_pool", CANDIDATE_POOL),
    ("minimum_catalog_evidence_confidence", MIN_EVIDENCE_CONFIDENCE),
    ("catalog_description_evidence", False),
    ("partition_order", "compatible,unknown,explicit_violation"),
    ("top_k", 10),
)

SPECS = (
    P8Spec(
        C00,
        "control",
        "r08_coverage",
        ("visible_state", "broad120+strict80_rrf", "coverage_cascade", "top10"),
        "Exact served coverage/off/fast control.",
    ),
    P8Spec(
        S00,
        "diagnostic",
        "explicit_negative_shadow",
        (
            "r08_coverage",
            "current_hard_negative_compile",
            "catalog_compatibility_partition",
            "shadow_only",
        ),
        "Computes the frozen negative partition without changing output.",
        _NEGATIVE_PARAMETERS,
    ),
    P8Spec(
        R01,
        "constraint_execution",
        "explicit_negative_partition",
        (
            "r08_coverage",
            "current_hard_negative_compile",
            "compatible_then_unknown_then_violation",
            "deterministic_violation_fallback",
            "top10",
        ),
        "Serves the stable negative partition with deterministic short-list fallback.",
        _NEGATIVE_PARAMETERS,
    ),
)
SPEC_BY_ID = {spec.variant_id: spec for spec in SPECS}


def validate_registry(specs: Iterable[P8Spec] = SPECS) -> None:
    materialized = tuple(specs)
    identifiers = [spec.variant_id for spec in materialized]
    mechanisms = [spec.mechanism for spec in materialized]
    stage_graphs = [spec.stage_graph for spec in materialized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("P8 variant IDs must be unique")
    if len(mechanisms) != len(set(mechanisms)):
        raise ValueError("P8 mechanisms must be unique")
    if len(stage_graphs) != len(set(stage_graphs)):
        raise ValueError("P8 stage graphs must be unique")
    if set(identifiers) != {C00, S00, R01}:
        raise ValueError("P8 registry must contain the frozen control, shadow, and active roles")


validate_registry()


@dataclass(slots=True)
class P8Stats:
    turns: int = 0
    activations: int = 0
    output_changes: int = 0
    shadow_changes: int = 0
    fallbacks: int = 0
    exception_count: int = 0
    executable_constraint_total: int = 0
    ignored_record_total: int = 0
    violation_fallback_candidate_total: int = 0
    compatible_total: int = 0
    unknown_total: int = 0
    explicit_violation_total: int = 0
    reason_counts: Counter[str] = field(default_factory=Counter)
    rejection_counts: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "turns": self.turns,
            "activations": self.activations,
            "output_changes": self.output_changes,
            "shadow_changes": self.shadow_changes,
            "fallbacks": self.fallbacks,
            "exception_count": self.exception_count,
            "executable_constraint_total": self.executable_constraint_total,
            "ignored_record_total": self.ignored_record_total,
            "violation_fallback_candidate_total": self.violation_fallback_candidate_total,
            "partition_totals": {
                COMPATIBLE: self.compatible_total,
                UNKNOWN: self.unknown_total,
                EXPLICIT_VIOLATION: self.explicit_violation_total,
            },
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class NegativeComputation:
    identifiers: tuple[str, ...]
    compilation: NegativeCompilation
    partition: NegativePartition
    reason: str
    exception_class: str | None = None


def canonical_jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for record in records
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registration_metadata(
    spec_path: str | Path | None,
    lock_path: str | Path | None,
) -> dict[str, object]:
    if (spec_path is None) != (lock_path is None):
        raise ValueError("spec_path and lock_path must be provided together")
    if spec_path is None or lock_path is None:
        return {}
    paths = {"spec": Path(spec_path), "lock": Path(lock_path)}
    payloads: dict[str, Mapping[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"P8 {name} file does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"P8 {name} file must be a readable JSON object") from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"P8 {name} file must contain a JSON object")
        payloads[name] = payload
    spec_sha256 = _sha256_file(paths["spec"])
    declared_spec_sha256 = payloads["lock"].get("spec_sha256")
    if declared_spec_sha256 is not None and declared_spec_sha256 != spec_sha256:
        raise ValueError("P8 lock spec_sha256 does not match the supplied spec")
    declared_protocol_sha256 = payloads["lock"].get("protocol_spec_sha256")
    worker_protocol_sha256 = payloads["spec"].get("protocol_spec_sha256")
    if (
        declared_protocol_sha256 is not None
        and declared_protocol_sha256 != worker_protocol_sha256
    ):
        raise ValueError("P8 worker spec and lock disagree on the protocol spec")
    return {
        "spec_sha256": spec_sha256,
        "lock_sha256": _sha256_file(paths["lock"]),
        "protocol_spec_sha256": worker_protocol_sha256,
        "spec_schema_version": payloads["spec"].get("schema_version"),
        "lock_schema_version": payloads["lock"].get("schema_version"),
    }


class P8Agent(Agent):
    """Experiment-only served-control wrapper for the single P8 mechanism."""

    def __init__(
        self,
        catalog_path: str | Path,
        variant_id: str = C00,
        *,
        registration: Mapping[str, object] | None = None,
    ) -> None:
        if variant_id not in SPEC_BY_ID:
            raise ValueError(f"unknown P8 variant: {variant_id}")
        self.p8_spec = SPEC_BY_ID[variant_id]
        self.p8_stats = P8Stats()
        self._p8_rowids: dict[str, int] = {}
        self._state_session_indexes: dict[int, int] = {}
        self._next_session_index = 0
        self._p8_audit: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()
        self._p8_responses: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()
        self._integrity_errors: list[str] = []
        self._registration = dict(registration or {})
        super().__init__(
            catalog_path,
            llm_client=None,
            question_policy="fast",
            trace_sink=None,
            rerank_mode="off",
            retrieval_mode="coverage",
        )

    def _build_index(self) -> None:
        super()._build_index()
        if self.p8_spec.variant_id != C00:
            self._p8_rowids = {
                str(parent_asin): int(rowid)
                for rowid, parent_asin in self.connection.execute(
                    "SELECT rowid, parent_asin FROM products"
                )
            }

    def reset(self, session_id: str, user_profile: dict) -> None:
        with self._lock:
            previous = self._sessions.get(session_id)
            if previous is not None:
                self._state_session_indexes.pop(id(previous), None)
            session_index = self._next_session_index
            self._next_session_index += 1
            super().reset(session_id, user_profile)
            self._state_session_indexes[id(self._sessions[session_id])] = session_index

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            super().drop_session(session_id)
            if state is not None:
                self._state_session_indexes.pop(id(state), None)

    def experiment_stats(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "negative_schema_version": NEGATIVE_SCHEMA_VERSION,
            "spec": self.p8_spec.as_dict(),
            "frozen_parameters": dict(_NEGATIVE_PARAMETERS),
            **self.p8_stats.as_dict(),
        }

    def experiment_audit(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(list(self._p8_audit.values()))

    @staticmethod
    def _empty_partition(identifiers: Iterable[str]) -> NegativePartition:
        values = tuple(identifiers)
        return NegativePartition(
            identifiers=values,
            states=tuple((identifier, UNKNOWN) for identifier in values),
            counts=((COMPATIBLE, 0), (UNKNOWN, len(values)), (EXPLICIT_VIOLATION, 0)),
            violation_fallback_count=0,
            top_k=10,
            candidate_pool=CANDIDATE_POOL,
        )

    def _load_p8_views(
        self, identifiers: Iterable[str]
    ) -> dict[str, ProductAttributeView]:
        ordered = list(dict.fromkeys(identifiers))
        views: dict[str, ProductAttributeView] = {}
        missing: list[str] = []
        for identifier in ordered:
            cached = self._attribute_view_cache.get(identifier)
            if cached is None:
                missing.append(identifier)
            else:
                self._attribute_view_cache.move_to_end(identifier)
                views[identifier] = cached
        rowids = [self._p8_rowids[value] for value in missing if value in self._p8_rowids]
        if rowids:
            placeholders = ",".join("?" for _ in rowids)
            rows = self.connection.execute(
                "SELECT parent_asin, title, categories, features, details, store "
                f"FROM products WHERE rowid IN ({placeholders})",
                rowids,
            ).fetchall()
            for row in rows:
                identifier = str(row[0])
                view = build_product_attribute_view({
                    "parent_asin": identifier,
                    "title": row[1],
                    "categories": row[2],
                    "features": row[3],
                    "details": row[4],
                    "store": row[5],
                })
                views[identifier] = view
                self._attribute_view_cache[identifier] = view
                self._attribute_view_cache.move_to_end(identifier)
                while len(self._attribute_view_cache) > 10_000:
                    self._attribute_view_cache.popitem(last=False)
        return views

    def _compute_negative(
        self,
        state: SessionState,
        baseline: list[str],
    ) -> NegativeComputation:
        compilation = compile_negative_constraints(
            state.slot_ledger.records,
            current_version=state.version,
        )
        if not compilation.constraints:
            return NegativeComputation(
                tuple(baseline),
                compilation,
                self._empty_partition(baseline),
                "no_executable_negatives",
            )
        if not baseline:
            return NegativeComputation(
                (), compilation, self._empty_partition(()), "empty_candidate_pool"
            )
        views = self._load_p8_views(baseline)
        partition = stable_negative_partition(
            baseline,
            views,
            compilation.constraints,
            top_k=10,
            candidate_pool=CANDIDATE_POOL,
        )
        return NegativeComputation(
            partition.identifiers,
            compilation,
            partition,
            "partitioned",
        )

    def _diagnostics(
        self,
        state: SessionState,
        baseline: list[str],
        proposed: list[str],
        served: list[str],
        computation: NegativeComputation,
    ) -> dict[str, Any]:
        partition = computation.partition.as_dict()
        return {
            "schema_version": NEGATIVE_SCHEMA_VERSION,
            "lab_schema_version": SCHEMA_VERSION,
            "variant_id": self.p8_spec.variant_id,
            "mode": self.p8_spec.mechanism,
            "reason": computation.reason,
            "active": computation.reason == "partitioned",
            "affects_output": self.p8_spec.variant_id == R01,
            "would_change_top_10": proposed[:10] != baseline[:10],
            "changed_top_10": served[:10] != baseline[:10],
            "fallback": computation.partition.violation_fallback_count > 0,
            "current_goal_version": state.version,
            "compilation": computation.compilation.as_dict(),
            "partition": partition,
            "baseline_top10": baseline[:10],
            "proposal_top10": proposed[:10],
            "output_top10": served[:10],
            "exception_class": computation.exception_class,
            "target_blind": True,
            "label_free": True,
        }

    def _record_audit(
        self,
        state: SessionState,
        diagnostics: Mapping[str, Any],
    ) -> None:
        session_index = self._state_session_indexes.get(id(state), -1)
        coordinate = (session_index, len(state.messages))
        record = {
            "session_index": coordinate[0],
            "turn": coordinate[1],
            **copy.deepcopy(dict(diagnostics)),
        }
        self._p8_audit[coordinate] = record
        self._p8_audit.move_to_end(coordinate)
        while len(self._p8_audit) > _AUDIT_RECORD_LIMIT:
            self._p8_audit.popitem(last=False)

    def _attach_negative_diagnostics(
        self, state: SessionState, diagnostics: dict[str, Any]
    ) -> None:
        existing = self._ranking_diagnostics.get(
            id(state), self._empty_rerank_diagnostics()
        )
        self._store_ranking_diagnostics(
            state, {**existing, "explicit_negative": diagnostics}
        )

    def _rank_candidates(self, state: SessionState) -> dict[str, list[str]]:
        rankings = super()._rank_candidates(state)
        baseline = list(rankings["final"])
        self.p8_stats.turns += 1

        if self.p8_spec.variant_id == C00:
            empty = NegativeCompilation((), 0, ())
            computation = NegativeComputation(
                tuple(baseline), empty, self._empty_partition(baseline), "control"
            )
        else:
            try:
                computation = self._compute_negative(state, baseline)
            except Exception as error:  # the isolated experiment must preserve R08
                compilation = compile_negative_constraints(
                    state.slot_ledger.records,
                    current_version=state.version,
                )
                computation = NegativeComputation(
                    tuple(baseline),
                    compilation,
                    self._empty_partition(baseline),
                    "exception_fallback",
                    type(error).__name__,
                )

        proposed = list(computation.identifiers)
        served = proposed if self.p8_spec.variant_id == R01 else baseline
        diagnostics = self._diagnostics(
            state, baseline, proposed, served, computation
        )
        self._attach_negative_diagnostics(state, diagnostics)
        self._record_audit(state, diagnostics)

        counts = dict(computation.partition.counts)
        rejection_counts = dict(computation.compilation.rejection_counts)
        proposed_changed = proposed[:10] != baseline[:10]
        output_changed = served[:10] != baseline[:10]
        active = computation.reason == "partitioned"
        ignored = sum(rejection_counts.values())
        self.p8_stats.activations += int(active)
        self.p8_stats.output_changes += int(output_changed)
        self.p8_stats.shadow_changes += int(proposed_changed)
        self.p8_stats.fallbacks += int(computation.partition.violation_fallback_count > 0)
        self.p8_stats.exception_count += int(computation.reason == "exception_fallback")
        self.p8_stats.executable_constraint_total += len(computation.compilation.constraints)
        self.p8_stats.ignored_record_total += ignored
        self.p8_stats.violation_fallback_candidate_total += (
            computation.partition.violation_fallback_count
        )
        self.p8_stats.compatible_total += counts.get(COMPATIBLE, 0)
        self.p8_stats.unknown_total += counts.get(UNKNOWN, 0)
        self.p8_stats.explicit_violation_total += counts.get(EXPLICIT_VIOLATION, 0)
        self.p8_stats.reason_counts[computation.reason] += 1
        self.p8_stats.rejection_counts.update(rejection_counts)
        return {**rankings, "final": served}

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
            coordinate = (session_index, turn)
            if coordinate in self._p8_responses:
                self._integrity_errors.append("duplicate_response_coordinate")
            else:
                self._p8_responses[coordinate] = {
                    "session_index": session_index,
                    "turn": turn,
                    "response": copy.deepcopy(response),
                }
            return response

    def debug_negative_diagnostics(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id: {session_id}")
            state = self._sessions[session_id]
            diagnostics = self._ranking_diagnostics.get(id(state))
            if diagnostics is None or "explicit_negative" not in diagnostics:
                self._rank_candidates(state)
                diagnostics = self._ranking_diagnostics[id(state)]
            return copy.deepcopy(diagnostics["explicit_negative"])

    def export_p8_blind_capture(self) -> dict[str, Any]:
        with self._lock:
            audit = list(self._p8_audit.values())
            responses = list(self._p8_responses.values())
            function_sources = {
                "compile_negative_constraints": inspect.getsource(
                    compile_negative_constraints
                ),
                "classify_candidate": inspect.getsource(classify_candidate),
                "stable_negative_partition": inspect.getsource(
                    stable_negative_partition
                ),
                "P8Agent._rank_candidates": inspect.getsource(
                    P8Agent._rank_candidates
                ),
            }
            return {
                "schema_version": SCHEMA_VERSION,
                "role": self.p8_spec.variant_id,
                "configuration": {
                    "retrieval_mode": "coverage",
                    "rerank_mode": "off",
                    "question_policy": "fast",
                    "target_blind": True,
                    "label_free": True,
                    **self._registration,
                },
                "stats": self.experiment_stats(),
                "integrity_errors": copy.deepcopy(self._integrity_errors),
                "hashes": {
                    "audit_sha256": hashlib.sha256(
                        canonical_jsonl_bytes(audit)
                    ).hexdigest(),
                    "responses_sha256": hashlib.sha256(
                        canonical_jsonl_bytes(responses)
                    ).hexdigest(),
                },
                "function_hashes": {
                    name: hashlib.sha256(source.encode("utf-8")).hexdigest()
                    for name, source in sorted(function_sources.items())
                },
            }


def _normalize_role(role: str) -> str:
    aliases = {
        "C00": C00,
        "S00": S00,
        "R01": R01,
        C00: C00,
        S00: S00,
        R01: R01,
    }
    try:
        return aliases[role]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown P8 role: {role}") from error


def create_p8_agent(
    *,
    role: str,
    catalog_path: str | Path,
    spec_path: str | Path | None = None,
    lock_path: str | Path | None = None,
) -> P8Agent:
    registration = _registration_metadata(spec_path, lock_path)
    return P8Agent(
        catalog_path,
        _normalize_role(role),
        registration=registration,
    )


__all__ = [
    "ACTIVE_ID",
    "C00",
    "CONTROL_ID",
    "NegativeComputation",
    "P8Agent",
    "P8Spec",
    "R01",
    "S00",
    "SCHEMA_VERSION",
    "SHADOW_ID",
    "SPECS",
    "SPEC_BY_ID",
    "canonical_jsonl_bytes",
    "create_p8_agent",
    "validate_registry",
]
