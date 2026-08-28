"""Isolated P9 lab using a bounded compact-evidence sidecar."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from starter.agent import Agent, SessionState
from starter.p8_negative import (
    ALLOWED_NEGATIVE_SLOTS,
    CANDIDATE_POOL,
    COMPATIBLE,
    EXPLICIT_VIOLATION,
    MIN_EVIDENCE_CONFIDENCE,
    UNKNOWN,
    NegativeCompilation,
    compile_negative_constraints,
)
from starter.p9_evidence import (
    REGISTRY_SHA256,
    SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION,
    SEMANTICS_SHA256,
    CompactEvidenceStore,
    CompactPartition,
    classify_masks,
    stable_compact_partition,
)


SCHEMA_VERSION = "p9.compact-negative-lab.v1"
C00 = "P9.C00.r08_coverage"
S00 = "P9.S00.compact_negative_shadow"
R01 = "P9.R01.compact_negative_partition"
CONTROL_ID = C00
SHADOW_ID = S00
ACTIVE_ID = R01


@dataclass(frozen=True, slots=True)
class P9Spec:
    variant_id: str
    family: str
    mechanism: str
    stage_graph: tuple[str, ...]
    description: str
    parameters: tuple[tuple[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_COMPACT_PARAMETERS = (
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
    ("evidence_schema_version", EVIDENCE_SCHEMA_VERSION),
    ("registry_sha256", REGISTRY_SHA256),
    ("semantics_sha256", SEMANTICS_SHA256),
)

SPECS = (
    P9Spec(
        C00,
        "control",
        "r08_coverage",
        ("visible_state", "broad120+strict80_rrf", "coverage_cascade", "top10"),
        "Exact served coverage/off/fast control; the evidence sidecar is not opened.",
    ),
    P9Spec(
        S00,
        "diagnostic",
        "compact_negative_shadow",
        (
            "r08_coverage",
            "current_hard_negative_compile",
            "rowid_sidecar_lookup",
            "compact_compatible_then_unknown_then_violation",
            "shadow_only",
        ),
        "Computes the compact partition while serving the exact control output.",
        _COMPACT_PARAMETERS,
    ),
    P9Spec(
        R01,
        "constraint_execution",
        "compact_negative_partition",
        (
            "r08_coverage",
            "current_hard_negative_compile",
            "rowid_sidecar_lookup",
            "compact_compatible_then_unknown_then_violation",
            "deterministic_violation_fallback",
            "top10",
        ),
        "Serves the compact stable partition with exact R08 exception fallback.",
        _COMPACT_PARAMETERS,
    ),
)
SPEC_BY_ID = {spec.variant_id: spec for spec in SPECS}


def validate_registry(specs: Iterable[P9Spec] = SPECS) -> None:
    materialized = tuple(specs)
    identifiers = [spec.variant_id for spec in materialized]
    mechanisms = [spec.mechanism for spec in materialized]
    stage_graphs = [spec.stage_graph for spec in materialized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("P9 variant IDs must be unique")
    if len(mechanisms) != len(set(mechanisms)):
        raise ValueError("P9 mechanisms must be unique")
    if len(stage_graphs) != len(set(stage_graphs)):
        raise ValueError("P9 stage graphs must be unique")
    if set(identifiers) != {C00, S00, R01}:
        raise ValueError("P9 registry must contain the frozen control, shadow, and active roles")


validate_registry()


@dataclass(slots=True)
class P9Stats:
    turns: int = 0
    activations: int = 0
    output_changes: int = 0
    shadow_changes: int = 0
    fallbacks: int = 0
    exact_exception_fallbacks: int = 0
    exception_count: int = 0
    executable_constraint_total: int = 0
    ignored_record_total: int = 0
    sidecar_rows_read: int = 0
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
            "exact_exception_fallbacks": self.exact_exception_fallbacks,
            "exception_count": self.exception_count,
            "executable_constraint_total": self.executable_constraint_total,
            "ignored_record_total": self.ignored_record_total,
            "sidecar_rows_read": self.sidecar_rows_read,
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
class P9Computation:
    identifiers: tuple[str, ...]
    compilation: NegativeCompilation
    partition: CompactPartition
    reason: str
    exception_class: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"P9 {name} file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"P9 {name} file must be readable JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"P9 {name} file must contain a JSON object")
    return payload


def _registration_metadata(
    spec_path: str | Path | None,
    lock_path: str | Path | None,
    *,
    include_evidence: bool,
) -> tuple[dict[str, object], tuple[int, str] | None]:
    if (spec_path is None) != (lock_path is None):
        raise ValueError("spec_path and lock_path must be provided together")
    if spec_path is None or lock_path is None:
        return {}, None
    spec = Path(spec_path)
    lock = Path(lock_path)
    spec_payload = _read_json_object(spec, "spec")
    lock_payload = _read_json_object(lock, "lock")
    spec_sha256 = _sha256_file(spec)
    declared_spec_sha256 = lock_payload.get("spec_sha256")
    if declared_spec_sha256 is not None and declared_spec_sha256 != spec_sha256:
        raise ValueError("P9 lock spec_sha256 does not match the supplied spec")
    declared_protocol_sha256 = lock_payload.get("protocol_spec_sha256")
    worker_protocol_sha256 = spec_payload.get("protocol_spec_sha256")
    if (
        declared_protocol_sha256 is not None
        and declared_protocol_sha256 != worker_protocol_sha256
    ):
        raise ValueError("P9 worker spec and lock disagree on the protocol spec")
    registration = {
        "spec_sha256": spec_sha256,
        "lock_sha256": _sha256_file(lock),
        "protocol_spec_sha256": worker_protocol_sha256,
        "spec_schema_version": spec_payload.get("schema_version"),
        "lock_schema_version": lock_payload.get("schema_version"),
    }
    if not include_evidence:
        return registration, None
    evidence = lock_payload.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {"bytes", "sha256"}:
        raise ValueError("P9 lock evidence must contain exactly bytes and sha256")
    expected_bytes = evidence.get("bytes")
    expected_sha256 = evidence.get("sha256")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
    ):
        raise ValueError("P9 lock evidence bytes must be a positive integer")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("P9 lock evidence sha256 must be lowercase hexadecimal")
    return registration, (expected_bytes, expected_sha256)


def _stream_record(digest: Any, record: Mapping[str, Any]) -> None:
    line = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest.update(line)
    digest.update(b"\n")


class P9Agent(Agent):
    """Coverage control plus an optional bounded compact-negative layer."""

    def __init__(
        self,
        catalog_path: str | Path,
        variant_id: str,
        *,
        evidence_path: str | Path | None = None,
        registration: Mapping[str, object] | None = None,
        expected_evidence: tuple[int, str] | None = None,
    ) -> None:
        if variant_id not in SPEC_BY_ID:
            raise ValueError(f"unknown P9 variant: {variant_id}")
        self.p9_spec = SPEC_BY_ID[variant_id]
        self.p9_stats = P9Stats()
        self._registration = dict(registration or {})
        self._evidence: CompactEvidenceStore | None = None
        self._evidence_configuration: dict[str, object] = {"evidence_opened": False}
        self._p9_candidate_rowids: Mapping[str, int] | None = None
        self._state_session_indexes: dict[int, int] = {}
        self._next_session_index = 0
        self._audit_digest = hashlib.sha256()
        self._response_digest = hashlib.sha256()
        self._integrity_errors: list[str] = []
        super().__init__(
            catalog_path,
            llm_client=None,
            question_policy="fast",
            trace_sink=None,
            rerank_mode="off",
            retrieval_mode="coverage",
        )
        if variant_id != C00:
            self._open_evidence(evidence_path, expected_evidence)

    def _open_evidence(
        self,
        evidence_path: str | Path | None,
        expected_evidence: tuple[int, str] | None,
    ) -> None:
        try:
            if evidence_path is None:
                raise FileNotFoundError("P9 evidence path is required")
            path = Path(evidence_path).resolve()
            evidence_bytes = path.stat().st_size
            evidence_sha256 = _sha256_file(path)
            if expected_evidence is not None and (
                (evidence_bytes, evidence_sha256) != expected_evidence
            ):
                raise ValueError("P9 evidence identity does not match the lock")
            evidence = CompactEvidenceStore(path)
            self._evidence = evidence
            self._evidence_configuration = {
                "evidence_opened": True,
                "evidence_identity_verified": True,
                "evidence_bytes": evidence_bytes,
                "evidence_sha256": evidence_sha256,
            }
        except Exception as error:
            self._evidence = None
            self._evidence_configuration = {
                "evidence_opened": False,
                "evidence_identity_verified": False,
            }
            self._integrity_errors.append(f"evidence_init:{type(error).__name__}")

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

    @staticmethod
    def _empty_partition(identifiers: Iterable[str]) -> CompactPartition:
        values = tuple(identifiers)
        return CompactPartition(values, 0, len(values), 0, 0)

    def _load_coverage_fields(
        self,
        parent_asins: list[str],
        candidate_rowids: dict[str, int],
    ) -> dict[str, tuple[str, str, str, str, str, str]]:
        if self.p9_spec.variant_id != C00:
            self._p9_candidate_rowids = candidate_rowids
        return super()._load_coverage_fields(parent_asins, candidate_rowids)

    def _compute_compact(
        self,
        state: SessionState,
        baseline: list[str],
    ) -> P9Computation:
        compilation = compile_negative_constraints(
            state.slot_ledger.records,
            current_version=state.version,
        )
        if not compilation.constraints:
            return P9Computation(
                tuple(baseline),
                compilation,
                self._empty_partition(baseline),
                "no_executable_negatives",
            )
        if not baseline:
            return P9Computation(
                (), compilation, self._empty_partition(()), "empty_candidate_pool"
            )
        if self._evidence is None:
            raise RuntimeError("P9 compact evidence is unavailable")
        candidate_rowids = self._p9_candidate_rowids
        if candidate_rowids is None:
            raise RuntimeError("P9 candidate rowids were not captured")
        pool = baseline[:CANDIDATE_POOL]
        requested: list[tuple[int, str]] = []
        for identifier in pool:
            rowid = candidate_rowids.get(identifier)
            if rowid is None:
                raise ValueError("P9 candidate has no captured catalog rowid")
            requested.append((int(rowid), identifier))
        evidence = self._evidence.fetch(requested)
        partition = stable_compact_partition(
            baseline,
            evidence,
            compilation.constraints,
            top_k=10,
            candidate_pool=CANDIDATE_POOL,
        )
        self.p9_stats.sidecar_rows_read += len(evidence)
        return P9Computation(
            partition.identifiers,
            compilation,
            partition,
            "partitioned",
        )

    def _record_computation(
        self,
        state: SessionState,
        baseline: list[str],
        proposed: list[str],
        served: list[str],
        computation: P9Computation,
    ) -> None:
        counts = dict(computation.partition.counts)
        rejections = dict(computation.compilation.rejection_counts)
        proposed_changed = proposed[:10] != baseline[:10]
        output_changed = served[:10] != baseline[:10]
        active = computation.reason == "partitioned"
        exception_fallback = computation.reason == "exception_fallback"
        violation_fallback = computation.partition.violation_fallback_count > 0
        self.p9_stats.activations += int(active)
        self.p9_stats.output_changes += int(output_changed)
        self.p9_stats.shadow_changes += int(proposed_changed)
        self.p9_stats.fallbacks += int(violation_fallback)
        self.p9_stats.exact_exception_fallbacks += int(exception_fallback)
        self.p9_stats.exception_count += int(exception_fallback)
        self.p9_stats.executable_constraint_total += len(computation.compilation.constraints)
        self.p9_stats.ignored_record_total += sum(rejections.values())
        self.p9_stats.violation_fallback_candidate_total += (
            computation.partition.violation_fallback_count
        )
        self.p9_stats.compatible_total += counts.get(COMPATIBLE, 0)
        self.p9_stats.unknown_total += counts.get(UNKNOWN, 0)
        self.p9_stats.explicit_violation_total += counts.get(EXPLICIT_VIOLATION, 0)
        self.p9_stats.reason_counts[computation.reason] += 1
        self.p9_stats.rejection_counts.update(rejections)
        session_index = self._state_session_indexes.get(id(state), -1)
        try:
            _stream_record(self._audit_digest, {
                "session_index": session_index,
                "turn": len(state.messages),
                "reason": computation.reason,
                "active": active,
                "would_change_top_10": proposed_changed,
                "changed_top_10": output_changed,
                "executable_constraints": len(computation.compilation.constraints),
                "ignored_records": sum(rejections.values()),
                "partition_counts": counts,
                "violation_fallback_count": computation.partition.violation_fallback_count,
                "exception_class": computation.exception_class,
            })
        except Exception as error:
            self._integrity_errors.append(f"audit_stream:{type(error).__name__}")

    def _rank_candidates(self, state: SessionState) -> dict[str, list[str]]:
        self._p9_candidate_rowids = None
        try:
            rankings = super()._rank_candidates(state)
            baseline = list(rankings["final"])
            self.p9_stats.turns += 1
            if self.p9_spec.variant_id == C00:
                empty = NegativeCompilation((), 0, ())
                computation = P9Computation(
                    tuple(baseline), empty, self._empty_partition(baseline), "control"
                )
            else:
                try:
                    computation = self._compute_compact(state, baseline)
                except Exception as error:
                    try:
                        compilation = compile_negative_constraints(
                            state.slot_ledger.records,
                            current_version=state.version,
                        )
                    except Exception:
                        compilation = NegativeCompilation((), 0, (("compile_exception", 1),))
                    computation = P9Computation(
                        tuple(baseline),
                        compilation,
                        self._empty_partition(baseline),
                        "exception_fallback",
                        type(error).__name__,
                    )
            proposed = list(computation.identifiers)
            served = proposed if self.p9_spec.variant_id == R01 else baseline
            try:
                self._record_computation(state, baseline, proposed, served, computation)
            except Exception as error:
                served = baseline
                self.p9_stats.exception_count += 1
                self.p9_stats.exact_exception_fallbacks += 1
                self.p9_stats.reason_counts["instrumentation_exception_fallback"] += 1
                self._integrity_errors.append(
                    f"computation_record:{type(error).__name__}"
                )
            return {**rankings, "final": served}
        finally:
            self._p9_candidate_rowids = None

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
                _stream_record(self._response_digest, {
                    "session_index": session_index,
                    "turn": turn,
                    "response": response,
                })
            except Exception as error:
                self._integrity_errors.append(f"response_stream:{type(error).__name__}")
            return response

    def experiment_stats(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "spec": self.p9_spec.as_dict(),
            "frozen_parameters": dict(_COMPACT_PARAMETERS),
            **self.p9_stats.as_dict(),
        }

    def export_p9_blind_capture(self) -> dict[str, Any]:
        with self._lock:
            function_sources = {
                "compile_negative_constraints": inspect.getsource(
                    compile_negative_constraints
                ),
                "classify_masks": inspect.getsource(classify_masks),
                "stable_compact_partition": inspect.getsource(
                    stable_compact_partition
                ),
                "CompactEvidenceStore.fetch": inspect.getsource(
                    CompactEvidenceStore.fetch
                ),
                "P9Agent._rank_candidates": inspect.getsource(
                    P9Agent._rank_candidates
                ),
            }
            return {
                "schema_version": SCHEMA_VERSION,
                "role": self.p9_spec.variant_id,
                "configuration": {
                    "retrieval_mode": "coverage",
                    "rerank_mode": "off",
                    "question_policy": "fast",
                    "target_blind": True,
                    "label_free": True,
                    **self._registration,
                    **self._evidence_configuration,
                },
                "stats": self.experiment_stats(),
                "integrity_errors": list(self._integrity_errors),
                "hashes": {
                    "audit_sha256": self._audit_digest.copy().hexdigest(),
                    "responses_sha256": self._response_digest.copy().hexdigest(),
                },
                "function_hashes": {
                    name: hashlib.sha256(source.encode("utf-8")).hexdigest()
                    for name, source in sorted(function_sources.items())
                },
            }

    def close(self) -> None:
        with self._lock:
            if self._evidence is not None:
                self._evidence.close()
                self._evidence = None
            self.connection.close()


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
        raise ValueError(f"unknown P9 role: {role}") from error


def create_p9_agent(
    *,
    role: str,
    catalog_path: str | Path,
    evidence_path: str | Path,
    spec_path: str | Path | None = None,
    lock_path: str | Path | None = None,
) -> P9Agent:
    variant_id = _normalize_role(role)
    registration, expected_evidence = _registration_metadata(
        spec_path,
        lock_path,
        include_evidence=variant_id != C00,
    )
    return P9Agent(
        catalog_path,
        variant_id,
        evidence_path=evidence_path,
        registration=registration,
        expected_evidence=expected_evidence,
    )


__all__ = [
    "ACTIVE_ID",
    "C00",
    "CONTROL_ID",
    "P9Agent",
    "P9Computation",
    "P9Spec",
    "R01",
    "S00",
    "SCHEMA_VERSION",
    "SHADOW_ID",
    "SPECS",
    "SPEC_BY_ID",
    "create_p9_agent",
    "validate_registry",
]
