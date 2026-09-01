"""Reversible production adapter for the frozen P11 Top-10 scorer.

The adapter owns no conversation state and never changes candidate membership.  It
maps the current served R08 rankings and visible :class:`SessionState` into the
already-frozen P11 feature/scoring contract.  Any initialization, feature, scoring,
instrumentation, or boundary failure returns the complete R08 ranking unchanged.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent import _terms
from .attributes import normalize_value
from .p11_features import (
    P11FeatureStore,
    PositiveConstraint,
    REGISTRY_SHA256,
    SCHEMA_VERSION as FEATURE_SCHEMA_VERSION,
    SCORER_VERSION,
    SEMANTICS_SHA256,
    rerank_top10_preserving_membership,
)
from .p8_negative import compile_negative_constraints
from .p9_evidence import OFFICIAL_CATALOG_ROWS, OFFICIAL_CATALOG_SHA256


SCHEMA_VERSION = "p11.production-bridge.v1"
MODES = ("off", "control", "shadow", "active")
EXPECTED_SIDECAR_BYTES = 32_501_760
EXPECTED_SIDECAR_SHA256 = (
    "83b6d8c04be6666173806b6e9cb03301eecb8ca58a60272bfa719e6533380473"
)
DEFAULT_SIDECAR = Path(__file__).resolve().parent / "assets" / "p11_features.sqlite"
_FETCH_BINDING_MESSAGES = (
    "missing requested candidate rows",
    "rowid-to-ASIN binding mismatch",
)
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


def _catalog_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            rows += int(bool(line.strip()))
    return digest.hexdigest(), rows


def _score_dicts(result: Any) -> dict[str, dict[str, object]]:
    return {
        str(identifier): breakdown.as_dict()
        for identifier, breakdown in result.breakdowns.items()
    }


def _positive_constraints(state: Any) -> tuple[PositiveConstraint, ...]:
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


def _latest_hard_clause_terms(state: Any) -> tuple[str, ...]:
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


@dataclass(frozen=True, slots=True)
class BridgeOutcome:
    identifiers: tuple[str, ...]
    diagnostics: Mapping[str, object]


class P11ProductionBridge:
    """One-sidecar, target-blind adapter with exact R08 fallback semantics."""

    def __init__(
        self,
        mode: str,
        sidecar_path: str | Path | None = None,
        *,
        expected_sidecar_bytes: int = EXPECTED_SIDECAR_BYTES,
        expected_sidecar_sha256: str = EXPECTED_SIDECAR_SHA256,
        expected_catalog_rows: int = OFFICIAL_CATALOG_ROWS,
        expected_catalog_sha256: str = OFFICIAL_CATALOG_SHA256,
        catalog_path: str | Path | None = None,
    ) -> None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"shadow", "active"}:
            raise ValueError("P11ProductionBridge mode must be shadow or active")
        self.mode = normalized_mode
        self._store: P11FeatureStore | None = None
        self._sidecar_path: str | None = None
        self._identity_verified = False
        self._initialization_code: str | None = None
        self._closed = False
        self.turns = 0
        self.proposed_changes = 0
        self.output_changes = 0
        self.fallbacks = 0
        self.sidecar_rows_read = 0
        self.maximum_rows_per_fetch = 0
        self.reason_counts: Counter[str] = Counter()

        try:
            if catalog_path is not None:
                catalog = Path(catalog_path).resolve()
                if not catalog.is_file():
                    self._initialization_code = "catalog_missing"
                    return
                if _catalog_identity(catalog) != (
                    str(expected_catalog_sha256).lower(),
                    int(expected_catalog_rows),
                ):
                    self._initialization_code = "catalog_identity_mismatch"
                    return
            path = Path(sidecar_path) if sidecar_path is not None else DEFAULT_SIDECAR
            path = path.resolve()
            self._sidecar_path = str(path)
            if not path.is_file():
                self._initialization_code = "sidecar_missing"
                return
            expected_identity = (
                int(expected_sidecar_bytes),
                str(expected_sidecar_sha256).lower(),
            )
            identity_before_open = (path.stat().st_size, _sha256_file(path))
            if identity_before_open != expected_identity:
                self._initialization_code = "sidecar_identity_mismatch"
                return
            self._store = P11FeatureStore(
                path,
                expected_catalog_sha256=expected_catalog_sha256,
                expected_catalog_rows=expected_catalog_rows,
            )
            identity_after_open = (path.stat().st_size, _sha256_file(path))
            if identity_after_open != identity_before_open:
                self._discard_store()
                self._initialization_code = "sidecar_identity_changed"
                return
            self._identity_verified = True
        except FileNotFoundError:
            self._discard_store()
            self._initialization_code = "sidecar_missing"
        except (OSError, ValueError):
            self._discard_store()
            self._initialization_code = "sidecar_invalid"
        except Exception:
            self._discard_store()
            self._initialization_code = "sidecar_init_failure"

    def _discard_store(self) -> None:
        store, self._store = self._store, None
        if store is not None:
            try:
                store.close()
            except Exception:
                pass

    def record_adapter_failure(self) -> None:
        """Account for an exception that escaped the bridge call boundary."""

        self.turns += 1
        self.fallbacks += 1
        self.reason_counts["bridge_adapter_failure"] += 1

    def _diagnostics(
        self,
        *,
        configured_mode: str,
        effective_mode: str,
        identity_verified: bool,
        reason_code: str,
        baseline: Sequence[str],
        proposed: Sequence[str],
        served: Sequence[str],
        breakdowns: Mapping[str, Mapping[str, object]] | None = None,
    ) -> dict[str, object]:
        baseline_head = list(baseline[:10])
        proposed_head = list(proposed[:10])
        served_head = list(served[:10])
        return {
            "schema_version": SCHEMA_VERSION,
            "configured_mode": configured_mode,
            "effective_mode": effective_mode,
            "identity_verified": identity_verified,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "scorer_version": SCORER_VERSION,
            "feature_registry_sha256": REGISTRY_SHA256,
            "feature_semantics_sha256": SEMANTICS_SHA256,
            "sidecar_path": self._sidecar_path,
            "sidecar_bytes": EXPECTED_SIDECAR_BYTES,
            "sidecar_sha256": EXPECTED_SIDECAR_SHA256,
            "reason_code": reason_code,
            "fallback": effective_mode == "fallback",
            "baseline_top10": baseline_head,
            "proposed_top10": proposed_head,
            "served_top10": served_head,
            "changed_top10_order": proposed_head != baseline_head,
            "output_changed": served_head != baseline_head,
            "top10_membership_preserved": set(proposed_head) == set(baseline_head),
            "tail_preserved": list(proposed[10:]) == list(baseline[10:]),
            "breakdowns": dict(breakdowns or {}),
        }

    def status(self) -> dict[str, object]:
        if self._closed:
            effective = "fallback"
            reason = "bridge_closed"
        else:
            effective = self.mode if self._identity_verified else "fallback"
            reason = self._initialization_code or "ready"
        return {
            "schema_version": SCHEMA_VERSION,
            "configured_mode": self.mode,
            "effective_mode": effective,
            "identity_verified": self._identity_verified,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "scorer_version": SCORER_VERSION,
            "feature_registry_sha256": REGISTRY_SHA256,
            "feature_semantics_sha256": SEMANTICS_SHA256,
            "sidecar_path": self._sidecar_path,
            "sidecar_bytes": EXPECTED_SIDECAR_BYTES,
            "sidecar_sha256": EXPECTED_SIDECAR_SHA256,
            "reason_code": reason,
            "fallback": effective == "fallback",
            "stats": {
                "turns": self.turns,
                "proposed_changes": self.proposed_changes,
                "output_changes": self.output_changes,
                "fallbacks": self.fallbacks,
                "sidecar_rows_read": self.sidecar_rows_read,
                "maximum_rows_per_fetch": self.maximum_rows_per_fetch,
                "reason_counts": dict(sorted(self.reason_counts.items())),
            },
        }

    def _fallback(
        self,
        baseline: Sequence[str],
        reason_code: str,
    ) -> BridgeOutcome:
        original = tuple(str(identifier) for identifier in baseline)
        self.fallbacks += 1
        self.reason_counts[reason_code] += 1
        return BridgeOutcome(
            original,
            self._diagnostics(
                configured_mode=self.mode,
                effective_mode="fallback",
                identity_verified=self._identity_verified,
                reason_code=reason_code,
                baseline=original,
                proposed=original,
                served=original,
            ),
        )

    def apply(
        self,
        state: Any,
        rankings: Mapping[str, Sequence[str]],
        candidate_rowids: Mapping[str, int],
        query_terms: Sequence[str],
    ) -> BridgeOutcome:
        """Apply the frozen scorer once, or return the exact full baseline."""

        baseline = tuple(str(identifier) for identifier in rankings.get("final", ()))
        self.turns += 1
        if self._closed:
            return self._fallback(baseline, "bridge_closed")
        if not self._identity_verified or self._store is None:
            return self._fallback(
                baseline,
                self._initialization_code or "sidecar_unavailable",
            )
        try:
            requested: list[tuple[int, str]] = []
            for identifier in baseline[:10]:
                rowid = candidate_rowids.get(identifier)
                if not isinstance(rowid, int) or isinstance(rowid, bool) or rowid <= 0:
                    return self._fallback(baseline, "candidate_row_missing")
                requested.append((rowid, identifier))
            try:
                batch = self._store.fetch_top10(requested, query_terms)
                self.sidecar_rows_read += len(requested)
                self.maximum_rows_per_fetch = max(
                    self.maximum_rows_per_fetch,
                    len(requested),
                )
                query_subtypes = self._store.resolve_query_subtypes(state.category_text)
            except ValueError as error:
                code = (
                    "candidate_binding_failure"
                    if any(message in str(error) for message in _FETCH_BINDING_MESSAGES)
                    else "feature_failure"
                )
                return self._fallback(baseline, code)
            except Exception:
                return self._fallback(baseline, "feature_failure")

            result = rerank_top10_preserving_membership(
                baseline,
                batch,
                query_terms=query_terms,
                broad_ranks={
                    identifier: rank
                    for rank, identifier in enumerate(rankings.get("broad", ()), start=1)
                },
                strict_ranks={
                    identifier: rank
                    for rank, identifier in enumerate(rankings.get("strict", ()), start=1)
                },
                fused_ranks={
                    identifier: rank
                    for rank, identifier in enumerate(rankings.get("fused", ()), start=1)
                },
                positive_constraints=_positive_constraints(state),
                negative_constraints=compile_negative_constraints(
                    state.slot_ledger.records,
                    current_version=state.version,
                ).constraints,
                query_subtypes=query_subtypes,
                hard_clause_terms=_latest_hard_clause_terms(state),
                current_turn=max(1, len(state.messages)),
                current_version=state.version,
            )
            if result.fallback:
                return self._fallback(baseline, "score_failure")
            proposed = tuple(str(identifier) for identifier in result.identifiers)
            valid = (
                len(proposed) == len(baseline)
                and len(set(proposed)) == len(proposed)
                and set(proposed[:10]) == set(baseline[:10])
                and proposed[10:] == baseline[10:]
            )
            if not valid:
                return self._fallback(baseline, "boundary_violation")
            served = proposed if self.mode == "active" else baseline
            breakdowns = _score_dicts(result)
            changed = proposed[:10] != baseline[:10]
            self.proposed_changes += int(changed)
            self.output_changes += int(served[:10] != baseline[:10])
            reason_code = "scored" if result.reason == "scored" else "empty"
            self.reason_counts[reason_code] += 1
            return BridgeOutcome(
                served,
                self._diagnostics(
                    configured_mode=self.mode,
                    effective_mode=self.mode,
                    identity_verified=True,
                    reason_code=reason_code,
                    baseline=baseline,
                    proposed=proposed,
                    served=served,
                    breakdowns=breakdowns,
                ),
            )
        except Exception:
            return self._fallback(baseline, "bridge_failure")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        store, self._store = self._store, None
        self._identity_verified = False
        self._initialization_code = "bridge_closed"
        if store is not None:
            store.close()


__all__ = [
    "BridgeOutcome",
    "DEFAULT_SIDECAR",
    "EXPECTED_SIDECAR_BYTES",
    "EXPECTED_SIDECAR_SHA256",
    "MODES",
    "P11ProductionBridge",
    "SCHEMA_VERSION",
]
