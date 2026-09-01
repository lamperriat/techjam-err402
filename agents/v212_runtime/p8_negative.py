"""Pure, target-blind primitives for the isolated P8 negative experiment.

Only current-version, explicit, high-confidence hard negatives are executable.
Catalog descriptions are deliberately outside the evidence model.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from .attributes import (
    ProductAttributeView,
    normalize_value,
    product_slot,
)
from .slot_ledger import ACTIVE


SCHEMA_VERSION = "p8.explicit-negative.v1"
COMPATIBLE = "compatible"
UNKNOWN = "unknown"
EXPLICIT_VIOLATION = "explicit_violation"
PARTITION_ORDER = (COMPATIBLE, UNKNOWN, EXPLICIT_VIOLATION)
CANDIDATE_POOL = 50
MIN_EVIDENCE_CONFIDENCE = 0.90
ALLOWED_NEGATIVE_SLOTS = frozenset({
    "audience",
    "material",
    "color",
    "closure",
    "style",
    "use_case",
})
_RELIABLE_SOURCES = frozenset({"categories", "title", "features", "details", "store"})


@dataclass(frozen=True, slots=True)
class ExecutableNegative:
    slot: str
    value: str
    record_id: int
    source_turn: int
    version: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NegativeCompilation:
    constraints: tuple[ExecutableNegative, ...]
    examined_count: int
    rejection_counts: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "constraints": [constraint.as_dict() for constraint in self.constraints],
            "examined_count": self.examined_count,
            "executable_count": len(self.constraints),
            "rejection_counts": dict(self.rejection_counts),
        }


@dataclass(frozen=True, slots=True)
class CandidateCompatibility:
    state: str
    violations: tuple[str, ...] = ()
    known_compatible: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NegativePartition:
    identifiers: tuple[str, ...]
    states: tuple[tuple[str, str], ...]
    counts: tuple[tuple[str, int], ...]
    violation_fallback_count: int
    top_k: int
    candidate_pool: int

    def as_dict(self) -> dict[str, object]:
        return {
            "identifiers": list(self.identifiers),
            "states": dict(self.states),
            "counts": dict(self.counts),
            "violation_fallback_count": self.violation_fallback_count,
            "fallback_rule": (
                "append original-order explicit violations only when compatible plus "
                "unknown candidates cannot fill top_k"
            ),
            "top_k": self.top_k,
            "candidate_pool": self.candidate_pool,
            "base_tail_count": max(0, len(self.identifiers) - sum(dict(self.counts).values())),
        }


def _record_rejection(record: object, current_version: int) -> str | None:
    if getattr(record, "status", None) != ACTIVE:
        return "not_active"
    if getattr(record, "version", None) != current_version:
        return "stale_goal_version"
    if getattr(record, "polarity", None) != -1:
        return "not_negative"
    if getattr(record, "hardness", None) != "hard":
        return "not_hard"
    if getattr(record, "source", None) != "excluded_terms":
        return "untrusted_source"
    confidence = getattr(record, "confidence", None)
    if isinstance(confidence, bool) or confidence != 1.0:
        return "not_full_confidence"
    slot = normalize_value(getattr(record, "slot", "")).replace(" ", "_")
    if slot not in ALLOWED_NEGATIVE_SLOTS:
        return "slot_not_allowed"
    value = normalize_value(getattr(record, "value", ""))
    if not re.fullmatch(r"[a-z0-9]+", value):
        return "value_not_single_token"
    return None


def compile_negative_constraints(
    records: Iterable[object],
    *,
    current_version: int,
) -> NegativeCompilation:
    """Compile the sole P8 primitive from visible active-ledger records."""

    selected: dict[tuple[str, str], ExecutableNegative] = {}
    rejections: Counter[str] = Counter()
    examined = 0
    for record in records:
        examined += 1
        rejection = _record_rejection(record, current_version)
        if rejection is not None:
            rejections[rejection] += 1
            continue
        slot = normalize_value(getattr(record, "slot")).replace(" ", "_")
        value = normalize_value(getattr(record, "value"))
        constraint = ExecutableNegative(
            slot=slot,
            value=value,
            record_id=int(getattr(record, "record_id")),
            source_turn=int(getattr(record, "source_turn")),
            version=int(getattr(record, "version")),
        )
        key = (slot, value)
        previous = selected.get(key)
        if previous is None or constraint.record_id < previous.record_id:
            selected[key] = constraint
        else:
            rejections["duplicate"] += 1
    constraints = tuple(selected[key] for key in sorted(selected))
    return NegativeCompilation(
        constraints=constraints,
        examined_count=examined,
        rejection_counts=tuple(sorted(rejections.items())),
    )


def _is_reliable_source(source: str) -> bool:
    return source in _RELIABLE_SOURCES or source.startswith("details.")


def classify_candidate(
    view: ProductAttributeView,
    constraints: Iterable[ExecutableNegative],
) -> CandidateCompatibility:
    """Return compatible/unknown/explicit_violation from reliable fields only."""

    violations: list[str] = []
    compatible: list[str] = []
    unknown: list[str] = []
    for constraint in constraints:
        key = f"{constraint.slot}={constraint.value}"
        values = tuple(
            item
            for item in product_slot(view, constraint.slot)
            if _is_reliable_source(item.source)
            and item.confidence >= MIN_EVIDENCE_CONFIDENCE
        )
        if any(item.value == constraint.value for item in values):
            violations.append(key)
        elif values:
            compatible.append(key)
        else:
            unknown.append(key)
    if violations:
        state = EXPLICIT_VIOLATION
    elif unknown:
        state = UNKNOWN
    else:
        state = COMPATIBLE
    return CandidateCompatibility(
        state=state,
        violations=tuple(violations),
        known_compatible=tuple(compatible),
        unknown=tuple(unknown),
    )


def stable_negative_partition(
    identifiers: Iterable[str],
    views: Mapping[str, ProductAttributeView],
    constraints: Iterable[ExecutableNegative],
    *,
    top_k: int = 10,
    candidate_pool: int = CANDIDATE_POOL,
) -> NegativePartition:
    """Stable compatible -> unknown -> violation partition with bounded fallback."""

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if (
        not isinstance(candidate_pool, int)
        or isinstance(candidate_pool, bool)
        or candidate_pool < top_k
    ):
        raise ValueError("candidate_pool must be an integer at least as large as top_k")
    ordered = tuple(dict.fromkeys(str(identifier) for identifier in identifiers))
    pool = ordered[:candidate_pool]
    base_tail = ordered[candidate_pool:]
    materialized = tuple(constraints)
    if not materialized:
        states = tuple((identifier, UNKNOWN) for identifier in ordered)
        return NegativePartition(
            identifiers=ordered,
            states=states,
            counts=((COMPATIBLE, 0), (UNKNOWN, len(ordered)), (EXPLICIT_VIOLATION, 0)),
            violation_fallback_count=0,
            top_k=top_k,
            candidate_pool=candidate_pool,
        )

    partitions = {state: [] for state in PARTITION_ORDER}
    states: list[tuple[str, str]] = []
    for identifier in pool:
        view = views.get(identifier, ProductAttributeView(parent_asin=identifier))
        state = classify_candidate(view, materialized).state
        partitions[state].append(identifier)
        states.append((identifier, state))
    partitioned = tuple(
        identifier
        for state in PARTITION_ORDER
        for identifier in partitions[state]
    )
    final = (*partitioned, *base_tail)
    safe_count = len(partitions[COMPATIBLE]) + len(partitions[UNKNOWN])
    violation_fallback_count = max(0, min(top_k, len(final)) - min(top_k, safe_count))
    return NegativePartition(
        identifiers=final,
        states=tuple(states),
        counts=tuple((state, len(partitions[state])) for state in PARTITION_ORDER),
        violation_fallback_count=violation_fallback_count,
        top_k=top_k,
        candidate_pool=candidate_pool,
    )


__all__ = [
    "ALLOWED_NEGATIVE_SLOTS",
    "CANDIDATE_POOL",
    "COMPATIBLE",
    "CandidateCompatibility",
    "EXPLICIT_VIOLATION",
    "ExecutableNegative",
    "NegativeCompilation",
    "NegativePartition",
    "MIN_EVIDENCE_CONFIDENCE",
    "PARTITION_ORDER",
    "SCHEMA_VERSION",
    "UNKNOWN",
    "classify_candidate",
    "compile_negative_constraints",
    "stable_negative_partition",
]
