"""Target-blind progressive-other ordering primitives.

This module deliberately knows nothing about evaluator outcomes or target products.  It
accepts only causal, explicitly parsed attribute/value-family evidence and the already
frozen candidate order.  Runtime wiring is kept separate so malformed evidence can fall
back to the v2.12 order without partially mutating session state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import re

from .versioned_pagination import GRACE_PAGES, PAGE_SIZE, fixed_two_page_grace_order


SCHEMA_VERSION = "small-ranker-progressive-other.v1"
MAX_FAMILY_RECORDS = 16
MAX_FAMILY_COUNT = 3
_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9 _./+-]{0,63}$")


class ProgressiveOtherContractError(ValueError):
    """Raised when target-blind family evidence violates the frozen contract."""


@dataclass(frozen=True, order=True)
class FamilyKey:
    """Normalized attribute/value family used for bounded negative memory."""

    attribute: str
    value: str


def _normalized_token(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ProgressiveOtherContractError(f"{label} must be a string")
    normalized = " ".join(value.strip().lower().split())
    if not normalized or not _TOKEN_RE.fullmatch(normalized):
        raise ProgressiveOtherContractError(f"{label} is invalid")
    return normalized


def family_key(attribute: object, value: object) -> FamilyKey:
    """Build one validated, normalized family key."""

    return FamilyKey(
        _normalized_token(attribute, "family attribute"),
        _normalized_token(value, "family value"),
    )


def _coerce_key(value: object) -> FamilyKey:
    if isinstance(value, FamilyKey):
        return family_key(value.attribute, value.value)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
    ):
        return family_key(value[0], value[1])
    raise ProgressiveOtherContractError("family evidence must contain two-part keys")


def _validated_memory(memory: Mapping[object, object]) -> dict[FamilyKey, int]:
    if not isinstance(memory, Mapping):
        raise ProgressiveOtherContractError("family memory must be a mapping")
    if len(memory) > MAX_FAMILY_RECORDS:
        raise ProgressiveOtherContractError("family memory exceeds its fixed bound")
    validated: dict[FamilyKey, int] = {}
    for raw_key, raw_count in memory.items():
        key = _coerce_key(raw_key)
        if (
            not isinstance(raw_count, int)
            or isinstance(raw_count, bool)
            or not 1 <= raw_count <= MAX_FAMILY_COUNT
        ):
            raise ProgressiveOtherContractError("family count is invalid")
        if key in validated:
            raise ProgressiveOtherContractError("family memory contains duplicates")
        validated[key] = raw_count
    return validated


def update_family_memory(
    memory: Mapping[object, object],
    *,
    rejected: Iterable[object] = (),
    affirmed: Iterable[object] = (),
    reset: bool = False,
) -> dict[FamilyKey, int]:
    """Return a bounded memory update without mutating the caller's mapping.

    Affirmations are applied before same-turn rejections.  Runtime parsing must never
    emit both for one family; rejecting that ambiguity here would make an otherwise safe
    positive restatement recreate negative memory, so matching rejections are ignored.
    """

    updated = {} if reset else _validated_memory(memory)
    affirmed_keys = {_coerce_key(value) for value in affirmed}
    rejected_keys = {_coerce_key(value) for value in rejected} - affirmed_keys
    for key in affirmed_keys:
        updated.pop(key, None)
    for key in sorted(rejected_keys):
        if key not in updated and len(updated) >= MAX_FAMILY_RECORDS:
            continue
        updated[key] = min(MAX_FAMILY_COUNT, updated.get(key, 0) + 1)
    return updated


def _validated_identifiers(values: Sequence[object], label: str) -> list[str]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ProgressiveOtherContractError(f"{label} must be a sequence")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ProgressiveOtherContractError(f"{label} contains an invalid identifier")
        result.append(value)
    if len(result) != len(set(result)):
        raise ProgressiveOtherContractError(f"{label} contains duplicate identifiers")
    return result


def _candidate_keys(
    candidate_families: Mapping[object, Iterable[object]],
) -> dict[str, frozenset[FamilyKey]]:
    if not isinstance(candidate_families, Mapping):
        raise ProgressiveOtherContractError("candidate families must be a mapping")
    result: dict[str, frozenset[FamilyKey]] = {}
    for raw_identifier, raw_keys in candidate_families.items():
        if not isinstance(raw_identifier, str) or not raw_identifier:
            raise ProgressiveOtherContractError("candidate family identifier is invalid")
        if isinstance(raw_keys, (str, bytes, bytearray)):
            raise ProgressiveOtherContractError("candidate family values must be iterable keys")
        result[raw_identifier] = frozenset(_coerce_key(value) for value in raw_keys)
    return result


def progressive_other_order(
    order: Sequence[object],
    served: Sequence[object],
    *,
    intent_age: int,
    family_memory: Mapping[object, object],
    candidate_families: Mapping[object, Iterable[object]],
    top_k: int = PAGE_SIZE,
) -> list[str]:
    """Apply stable progressive-other partitioning after the frozen grace period.

    The returned list is always a full permutation of ``order``.  Unknown candidate
    attributes are neutral, never inferred as conflicts, and seen candidates remain the
    final fallback exactly as in v2.12.
    """

    if (
        not isinstance(intent_age, int)
        or isinstance(intent_age, bool)
        or intent_age < 1
    ):
        raise ProgressiveOtherContractError("intent_age must be a positive integer")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ProgressiveOtherContractError("top_k must be a positive integer")
    if not math.isfinite(float(top_k)):
        raise ProgressiveOtherContractError("top_k must be finite")

    frozen_order = fixed_two_page_grace_order(
        _validated_identifiers(order, "order"),
        _validated_identifiers(served, "served"),
        intent_age=intent_age,
        top_k=top_k,
    )
    memory = _validated_memory(family_memory)
    if intent_age <= GRACE_PAGES or not memory:
        return list(frozen_order)

    families = _candidate_keys(candidate_families)
    rejected = frozenset(memory)
    served_set = set(str(value) for value in served)
    unseen = [identifier for identifier in frozen_order if identifier not in served_set]
    seen = [identifier for identifier in frozen_order if identifier in served_set]
    clean_unseen = [
        identifier
        for identifier in unseen
        if not (families.get(identifier, frozenset()) & rejected)
    ]
    rejected_unseen = [
        identifier
        for identifier in unseen
        if families.get(identifier, frozenset()) & rejected
    ]
    ranked = clean_unseen + rejected_unseen + seen
    if len(ranked) != len(frozen_order) or set(ranked) != set(frozen_order):
        raise ProgressiveOtherContractError("progressive ordering changed membership")
    return ranked


__all__ = [
    "FamilyKey",
    "MAX_FAMILY_COUNT",
    "MAX_FAMILY_RECORDS",
    "ProgressiveOtherContractError",
    "SCHEMA_VERSION",
    "family_key",
    "progressive_other_order",
    "update_family_memory",
]
