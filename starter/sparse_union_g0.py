"""Deterministic clause-isolated sparse candidate union for v2.21 G0.

This module is deliberately target-blind, diagnostic-only, and default-off.  An
enabled runtime executes exactly two independent registry-derived FTS routes:
one over category fields and one over positive core-attribute fields.  It then
applies a conservative explicit-conflict mask to novel candidates and appends
an exact-Fraction RRF tail after an untouched variable-length C200 prefix.

No query in this module combines the two clauses.  The public execution audit
therefore has a permanently-zero ``legacy_route_executions`` counter.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from starter.agent import Agent
from starter.attributes import (
    CATEGORIES,
    SLOT_VOCABULARIES,
    ConversationConstraintView,
    ProductAttributeView,
    attribute_registry_sha256,
    build_conversation_constraint_view,
    build_product_attribute_view,
    normalize_value,
    product_slot,
)
from starter.p8_negative import (
    ALLOWED_NEGATIVE_SLOTS,
    EXPLICIT_VIOLATION,
    MIN_EVIDENCE_CONFIDENCE,
    ExecutableNegative,
    classify_candidate,
)
from starter.slot_ledger import ACTIVE


SCHEMA_VERSION = "small-ranker-sparse-union-g0.v1"
MECHANISM = "CLAUSE_ISOLATED_DUAL_VIEW_RRF_G0"
EXPECTED_ATTRIBUTE_REGISTRY_SHA256 = (
    "1d85fc42f49fd9374238d98b8feaeab8d76269b0987740256fe60e666757d2ca"
)

CATEGORY_ROUTE = "category"
POSITIVE_CORE_ROUTE = "positive_core"
ROUTE_NAMES = (CATEGORY_ROUTE, POSITIVE_CORE_ROUTE)
ROUTE_LIMIT = 120
TERM_LIMIT = 24
RRF_K = 60
PREFIX_MINIMUM = 100
PREFIX_LIMIT = 200
CANDIDATE_CAP = 400
CORE_ATTRIBUTE_SLOTS = ("material", "style", "use_case")
POSITIVE_CONFLICT_SLOTS = (
    "category",
    "audience",
    "material",
    "color",
    "closure",
    "style",
    "use_case",
    "size",
    "width",
)

FTS_COLUMNS = (
    "parent_asin",
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)
FTS_TOKENIZER = "unicode61 remove_diacritics 2"
FTS_QUERY_SQL = (
    "SELECT rowid,parent_asin FROM products WHERE products MATCH ? "
    "ORDER BY bm25(products,0.0,6.0,4.0,2.5,2.5,1.5,1.0),"
    "parent_asin ASC LIMIT 120"
)
CATEGORY_FIELDS = ("title", "categories")
POSITIVE_CORE_FIELDS = ("title", "features", "details", "description")

FTS_ROUTE_CACHE_CAPACITY = 512
PRODUCT_VIEW_CACHE_CAPACITY = 4096
MASK_DECISION_CACHE_CAPACITY = 16384

_SINGLE_TOKEN_RE = re.compile(r"[a-z0-9]+\Z")
_RELIABLE_SOURCES = frozenset(
    {"categories", "title", "features", "details", "store"}
)
_EXPECTED_FTS_COLUMNS = (
    "parent_asin",
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)
_EXPECTED_ALLOWED_NEGATIVE_SLOTS = frozenset(
    {"audience", "material", "color", "closure", "style", "use_case"}
)
_EXPECTED_RELIABLE_SOURCES = frozenset(
    {"categories", "title", "features", "details", "store"}
)
_CACHE_MISS = object()
_CANONICAL_VALUES = {
    "category": frozenset(normalize_value(value) for value in CATEGORIES.values()),
    **{
        slot: frozenset(normalize_value(value) for value in vocabulary.values())
        for slot, vocabulary in SLOT_VOCABULARIES.items()
    },
}


class SparseUnionG0Error(RuntimeError):
    """Base error for the isolated G0 runtime."""


class SparseUnionG0ValidationError(ValueError, SparseUnionG0Error):
    """Raised when a frozen input or runtime invariant is violated."""


class SparseUnionG0ClosedError(SparseUnionG0Error):
    """Raised when an operation is attempted after permanent close."""


@dataclass(frozen=True, slots=True)
class RouteQuery:
    """One independently executable registry-only FTS route."""

    route: str
    activated: bool
    canonical_values: tuple[tuple[str, str], ...] = ()
    terms: tuple[str, ...] = ()
    expression: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "activated": self.activated,
            "canonical_values": [list(value) for value in self.canonical_values],
            "terms": list(self.terms),
            "expression": self.expression,
        }


@dataclass(frozen=True, slots=True)
class DualRouteQueries:
    """The fixed category-only and positive-core-only query pair."""

    category: RouteQuery
    positive_core: RouteQuery

    @property
    def activated(self) -> bool:
        return self.category.activated or self.positive_core.activated

    def as_dict(self) -> dict[str, object]:
        return {
            "activated": self.activated,
            "category": self.category.as_dict(),
            "positive_core": self.positive_core.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class HardConflictRules:
    """Executable current-goal constraints for the conservative novel-tail mask."""

    negative: tuple[tuple[str, str], ...] = ()
    positive: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "negative": [list(value) for value in self.negative],
            "positive": {slot: list(values) for slot, values in self.positive},
        }


@dataclass(frozen=True, slots=True)
class HardConflictMaskResult:
    """Stable mask result; missing catalog evidence is always retained."""

    identifiers: tuple[str, ...]
    dropped: tuple[str, ...]
    negative_violation_count: int
    positive_conflict_count: int

    @property
    def conflict_count(self) -> int:
        return len(self.dropped)

    def as_dict(self) -> dict[str, object]:
        return {
            "identifiers": list(self.identifiers),
            "dropped": list(self.dropped),
            "conflict_count": self.conflict_count,
            "negative_violation_count": self.negative_violation_count,
            "positive_conflict_count": self.positive_conflict_count,
        }


@dataclass(frozen=True, slots=True)
class FusionItem:
    """One exact-RRF novel candidate and its deterministic audit fields."""

    identifier: str
    score: Fraction
    supporting_route_count: int
    minimum_route_rank: int
    category_rank: int
    positive_core_rank: int

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "score_numerator": self.score.numerator,
            "score_denominator": self.score.denominator,
            "supporting_route_count": self.supporting_route_count,
            "minimum_route_rank": self.minimum_route_rank,
            "category_rank": self.category_rank,
            "positive_core_rank": self.positive_core_rank,
        }


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Stable prefix plus a bounded exact-RRF novel tail."""

    candidates: tuple[str, ...]
    prefix: tuple[str, ...]
    tail: tuple[str, ...]
    items: tuple[FusionItem, ...]

    @property
    def dual_support_count(self) -> int:
        return sum(item.supporting_route_count == 2 for item in self.items)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidates": list(self.candidates),
            "prefix": list(self.prefix),
            "tail": list(self.tail),
            "items": [item.as_dict() for item in self.items],
            "dual_support_count": self.dual_support_count,
        }


@dataclass(frozen=True, slots=True)
class ExpansionResult:
    """Complete G0 expansion with an untouched C200 ordered prefix."""

    candidates: tuple[str, ...]
    prefix: tuple[str, ...]
    category_route: tuple[str, ...]
    positive_core_route: tuple[str, ...]
    category_novel: tuple[str, ...]
    positive_core_novel: tuple[str, ...]
    category_filtered: tuple[str, ...]
    positive_core_filtered: tuple[str, ...]
    tail: tuple[str, ...]
    enabled: bool
    activated: bool
    queries: DualRouteQueries
    rules: HardConflictRules
    conflict_count: int
    tail_conflict_count: int
    legacy_route_executions: int = 0

    @property
    def category_activated(self) -> bool:
        return self.queries.category.activated

    @property
    def positive_core_activated(self) -> bool:
        return self.queries.positive_core.activated

    @property
    def dual_support_count(self) -> int:
        return len(set(self.category_filtered) & set(self.positive_core_filtered))

    @property
    def union_novel_count(self) -> int:
        return len(set(self.category_filtered) | set(self.positive_core_filtered))

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mechanism": MECHANISM,
            "enabled": self.enabled,
            "activated": self.activated,
            "category_activated": self.category_activated,
            "positive_core_activated": self.positive_core_activated,
            "candidates": list(self.candidates),
            "prefix": list(self.prefix),
            "category_route": list(self.category_route),
            "positive_core_route": list(self.positive_core_route),
            "category_novel": list(self.category_novel),
            "positive_core_novel": list(self.positive_core_novel),
            "category_filtered": list(self.category_filtered),
            "positive_core_filtered": list(self.positive_core_filtered),
            "tail": list(self.tail),
            "conflict_count": self.conflict_count,
            "tail_conflict_count": self.tail_conflict_count,
            "dual_support_count": self.dual_support_count,
            "union_novel_count": self.union_novel_count,
            "legacy_route_executions": self.legacy_route_executions,
            "queries": self.queries.as_dict(),
            "rules": self.rules.as_dict(),
        }


@dataclass(slots=True)
class _CacheCounters:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    avoided_operations: int = 0

    def snapshot(self, *, size: int, capacity: int) -> dict[str, int]:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": size,
            "capacity": capacity,
            "avoided_operations": self.avoided_operations,
        }


@dataclass(frozen=True, slots=True)
class _DatabaseGeneration:
    total_changes: int
    schema_version: int
    columns: tuple[str, ...]
    schema_sql: str
    row_count: int
    distinct_identifier_count: int
    empty_identifier_count: int


@dataclass(frozen=True, slots=True)
class _Record:
    slot: str
    value: str
    polarity: int
    hardness: str
    source_turn: int
    version: int
    status: str
    order: int


def _registry(slot: str) -> Mapping[str, str]:
    if slot == "category":
        return CATEGORIES
    vocabulary = SLOT_VOCABULARIES.get(slot)
    if vocabulary is None:
        raise SparseUnionG0ValidationError(f"slot has no frozen registry: {slot}")
    return vocabulary


def _canonical_value(slot: str, value: object) -> str | None:
    normalized = normalize_value(value)
    if not normalized:
        return None
    registry = _registry(slot)
    direct = registry.get(normalized)
    if direct is not None:
        return normalize_value(direct)
    if normalized in _CANONICAL_VALUES[slot]:
        return normalized
    return None


def _registry_terms(slot: str, canonical_values: Iterable[str]) -> tuple[str, ...]:
    wanted = frozenset(canonical_values)
    if not wanted:
        return ()
    terms = set(wanted)
    terms.update(
        normalize_value(surface)
        for surface, canonical in _registry(slot).items()
        if normalize_value(canonical) in wanted
    )
    return tuple(sorted(term for term in terms if term))


def _field(record: object, name: str) -> object:
    if isinstance(record, Mapping):
        if name not in record:
            raise SparseUnionG0ValidationError(
                f"ledger record is missing required field: {name}"
            )
        return record[name]
    if not hasattr(record, name):
        raise SparseUnionG0ValidationError(
            f"ledger record is missing required field: {name}"
        )
    return getattr(record, name)


def _records(records: Iterable[object]) -> tuple[_Record, ...]:
    if isinstance(records, (str, bytes, Mapping)):
        raise SparseUnionG0ValidationError("records must be an iterable of records")
    try:
        materialized = tuple(records)
    except TypeError as error:
        raise SparseUnionG0ValidationError("records must be iterable") from error
    result: list[_Record] = []
    for order, record in enumerate(materialized, start=1):
        status = _field(record, "status")
        version = _field(record, "version")
        polarity = _field(record, "polarity")
        source_turn = _field(record, "source_turn")
        if not isinstance(status, str) or not status:
            raise SparseUnionG0ValidationError(
                "ledger status must be a non-empty string"
            )
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise SparseUnionG0ValidationError("ledger version must be positive")
        if polarity not in (-1, 1) or isinstance(polarity, bool):
            raise SparseUnionG0ValidationError("ledger polarity must be -1 or +1")
        if (
            not isinstance(source_turn, int)
            or isinstance(source_turn, bool)
            or source_turn < 0
        ):
            raise SparseUnionG0ValidationError(
                "ledger source_turn must be non-negative"
            )
        slot = normalize_value(_field(record, "slot")).replace(" ", "_")
        value = normalize_value(_field(record, "value"))
        hardness = normalize_value(_field(record, "hardness"))
        if not slot or not value or hardness not in {"hard", "soft"}:
            raise SparseUnionG0ValidationError("ledger slot/value/hardness is invalid")
        result.append(
            _Record(
                slot=slot,
                value=value,
                polarity=int(polarity),
                hardness=hardness,
                source_turn=int(source_turn),
                version=int(version),
                status=status,
                order=order,
            )
        )
    return tuple(result)


def _current_records(
    records: tuple[_Record, ...], current_version: int
) -> tuple[_Record, ...]:
    return tuple(
        record
        for record in records
        if record.status == ACTIVE and record.version == current_version
    )


def _terms_input(values: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise SparseUnionG0ValidationError(
            f"{name} must be an iterable of strings"
        )
    try:
        result = tuple(values)
    except TypeError as error:
        raise SparseUnionG0ValidationError(f"{name} must be iterable") from error
    if any(not isinstance(value, str) for value in result):
        raise SparseUnionG0ValidationError(f"{name} must contain only strings")
    return result


def _validated_context(
    *,
    category_text: str,
    active_terms: Iterable[str],
    excluded_terms: Iterable[str],
    current_version: int,
    records: Iterable[object],
) -> tuple[ConversationConstraintView, tuple[_Record, ...]]:
    if not isinstance(category_text, str):
        raise SparseUnionG0ValidationError("category_text must be a string")
    if (
        not isinstance(current_version, int)
        or isinstance(current_version, bool)
        or current_version < 1
    ):
        raise SparseUnionG0ValidationError(
            "current_version must be a positive integer"
        )
    active = _terms_input(active_terms, "active_terms")
    excluded = _terms_input(excluded_terms, "excluded_terms")
    materialized_records = _records(records)
    intent = build_conversation_constraint_view(category_text, active, excluded)
    return intent, materialized_records


def _quoted(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _route_expression(fields: tuple[str, ...], terms: tuple[str, ...]) -> str:
    if not terms:
        return ""
    return "{" + " ".join(fields) + "} : (" + " OR ".join(
        _quoted(term) for term in terms
    ) + ")"


def _queries_from_validated(
    intent: ConversationConstraintView,
    records: tuple[_Record, ...],
    current_version: int,
) -> DualRouteQueries:
    category_values = tuple(
        sorted(
            {
                canonical
                for value in intent.category_terms
                if (canonical := _canonical_value("category", value)) is not None
            }
        )
    )
    core_values = tuple(
        sorted(
            {
                (record.slot, canonical)
                for record in _current_records(records, current_version)
                if record.polarity == 1 and record.slot in CORE_ATTRIBUTE_SLOTS
                if (canonical := _canonical_value(record.slot, record.value)) is not None
            }
        )
    )
    category_terms = _registry_terms("category", category_values)[:TERM_LIMIT]
    core_surfaces: set[str] = set()
    for slot, canonical in core_values:
        core_surfaces.update(_registry_terms(slot, (canonical,)))
    core_terms = tuple(sorted(core_surfaces))[:TERM_LIMIT]

    category_query = RouteQuery(
        route=CATEGORY_ROUTE,
        activated=bool(category_values),
        canonical_values=tuple(("category", value) for value in category_values),
        terms=category_terms,
        expression=_route_expression(CATEGORY_FIELDS, category_terms),
    )
    positive_core_query = RouteQuery(
        route=POSITIVE_CORE_ROUTE,
        activated=bool(core_values),
        canonical_values=core_values,
        terms=core_terms,
        expression=_route_expression(POSITIVE_CORE_FIELDS, core_terms),
    )
    return DualRouteQueries(
        category=category_query,
        positive_core=positive_core_query,
    )


def build_route_queries(
    *,
    category_text: str,
    active_terms: Iterable[str],
    excluded_terms: Iterable[str],
    current_version: int,
    records: Iterable[object],
) -> DualRouteQueries:
    """Build both independent routes without touching SQLite."""

    intent, materialized_records = _validated_context(
        category_text=category_text,
        active_terms=active_terms,
        excluded_terms=excluded_terms,
        current_version=current_version,
        records=records,
    )
    return _queries_from_validated(intent, materialized_records, current_version)


def _rules_from_validated(
    intent: ConversationConstraintView,
    queries: DualRouteQueries,
    records: tuple[_Record, ...],
    current_version: int,
) -> HardConflictRules:
    current = _current_records(records, current_version)
    visible_negatives = {
        (constraint.slot, constraint.value)
        for constraint in intent.negative
        if constraint.polarity == -1
    }
    negative: set[tuple[str, str]] = set()
    positive: dict[str, set[str]] = {
        "category": {
            value for slot, value in queries.category.canonical_values if slot == "category"
        }
    }
    for record in current:
        if record.hardness != "hard":
            continue
        if record.polarity == -1 and record.slot in ALLOWED_NEGATIVE_SLOTS:
            canonical = _canonical_value(record.slot, record.value)
            if (
                canonical is not None
                and _SINGLE_TOKEN_RE.fullmatch(canonical)
                and (record.slot, canonical) in visible_negatives
            ):
                negative.add((record.slot, canonical))
        elif record.polarity == 1 and record.slot in POSITIVE_CONFLICT_SLOTS:
            canonical = _canonical_value(record.slot, record.value)
            if canonical is not None:
                positive.setdefault(record.slot, set()).add(canonical)
    return HardConflictRules(
        negative=tuple(sorted(negative)),
        positive=tuple(
            (slot, tuple(sorted(values)))
            for slot, values in sorted(positive.items())
            if values
        ),
    )


def compile_hard_conflict_rules(
    *,
    category_text: str,
    active_terms: Iterable[str],
    excluded_terms: Iterable[str],
    current_version: int,
    records: Iterable[object],
) -> HardConflictRules:
    """Compile only current-version explicit constraints from visible state."""

    intent, materialized_records = _validated_context(
        category_text=category_text,
        active_terms=active_terms,
        excluded_terms=excluded_terms,
        current_version=current_version,
        records=records,
    )
    queries = _queries_from_validated(intent, materialized_records, current_version)
    return _rules_from_validated(intent, queries, materialized_records, current_version)


def _reliable_values(view: ProductAttributeView, slot: str) -> frozenset[str]:
    values: set[str] = set()
    for item in product_slot(view, slot):
        if not (
            item.confidence >= MIN_EVIDENCE_CONFIDENCE
            and (
                item.source in _RELIABLE_SOURCES
                or item.source.startswith("details.")
            )
        ):
            continue
        canonical = _canonical_value(slot, item.value)
        if canonical is not None:
            values.add(canonical)
    return frozenset(values)


def _mask_inputs(
    rules: HardConflictRules,
) -> tuple[tuple[ExecutableNegative, ...], dict[str, tuple[str, ...]]]:
    executable_negatives = tuple(
        ExecutableNegative(
            slot=slot,
            value=value,
            record_id=index,
            source_turn=0,
            version=0,
        )
        for index, (slot, value) in enumerate(rules.negative, start=1)
    )
    return executable_negatives, dict(rules.positive)


def _mask_decision(
    view: ProductAttributeView,
    executable_negatives: tuple[ExecutableNegative, ...],
    positive: Mapping[str, tuple[str, ...]],
) -> tuple[bool, bool]:
    negative_violation = bool(
        executable_negatives
        and classify_candidate(view, executable_negatives).state
        == EXPLICIT_VIOLATION
    )
    positive_conflict = any(
        bool(evidence := _reliable_values(view, slot))
        and evidence.isdisjoint(requested)
        for slot, requested in positive.items()
    )
    return negative_violation, positive_conflict


def _mask_result(
    ordered: tuple[str, ...],
    decisions: Mapping[str, tuple[bool, bool]],
) -> HardConflictMaskResult:
    kept: list[str] = []
    dropped: list[str] = []
    negative_count = 0
    positive_count = 0
    for identifier in ordered:
        negative_violation, positive_conflict = decisions[identifier]
        if negative_violation or positive_conflict:
            dropped.append(identifier)
            negative_count += int(negative_violation)
            positive_count += int(positive_conflict)
        else:
            kept.append(identifier)
    return HardConflictMaskResult(
        identifiers=tuple(kept),
        dropped=tuple(dropped),
        negative_violation_count=negative_count,
        positive_conflict_count=positive_count,
    )


def apply_hard_conflict_mask(
    identifiers: Iterable[str],
    views: Mapping[str, ProductAttributeView],
    rules: HardConflictRules,
) -> HardConflictMaskResult:
    """Drop only reliable explicit conflicts while preserving route order."""

    ordered = _ordered_identifiers(identifiers, "identifiers")
    if not isinstance(views, Mapping):
        raise SparseUnionG0ValidationError("views must be a mapping")
    if not isinstance(rules, HardConflictRules):
        raise SparseUnionG0ValidationError("rules must be HardConflictRules")
    executable_negatives, positive = _mask_inputs(rules)
    decisions: dict[str, tuple[bool, bool]] = {}
    for identifier in ordered:
        view = views.get(identifier)
        if view is None:
            view = ProductAttributeView(parent_asin=identifier)
        if not isinstance(view, ProductAttributeView):
            raise SparseUnionG0ValidationError(
                "views must contain ProductAttributeView values"
            )
        decisions[identifier] = _mask_decision(
            view, executable_negatives, positive
        )
    return _mask_result(ordered, decisions)


def _ordered_identifiers(values: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise SparseUnionG0ValidationError(f"{name} must be an ordered iterable")
    try:
        ordered = tuple(values)
    except TypeError as error:
        raise SparseUnionG0ValidationError(f"{name} must be iterable") from error
    if any(not isinstance(value, str) or not value for value in ordered):
        raise SparseUnionG0ValidationError(
            f"{name} identifiers must be non-empty strings"
        )
    if len(ordered) != len(set(ordered)):
        raise SparseUnionG0ValidationError(f"{name} identifiers must be unique")
    return ordered


def _prefix(c200: Iterable[str]) -> tuple[str, ...]:
    prefix = _ordered_identifiers(c200, "c200")
    if not PREFIX_MINIMUM <= len(prefix) <= PREFIX_LIMIT:
        raise SparseUnionG0ValidationError(
            "c200 length must be between 100 and 200"
        )
    return prefix


def fuse_route_candidates(
    prefix: Iterable[str],
    category_filtered: Iterable[str],
    positive_core_filtered: Iterable[str],
) -> FusionResult:
    """Fuse two already-filtered novel routes using exact ``Fraction`` RRF."""

    stable_prefix = _prefix(prefix)
    category = _ordered_identifiers(category_filtered, "category_filtered")
    positive_core = _ordered_identifiers(
        positive_core_filtered, "positive_core_filtered"
    )
    if len(category) > ROUTE_LIMIT or len(positive_core) > ROUTE_LIMIT:
        raise SparseUnionG0ValidationError("a filtered route exceeded LIMIT 120")
    prefix_members = frozenset(stable_prefix)
    if prefix_members.intersection(category) or prefix_members.intersection(positive_core):
        raise SparseUnionG0ValidationError(
            "filtered routes must not contain a C200 identifier"
        )

    category_ranks = {
        identifier: rank for rank, identifier in enumerate(category, start=1)
    }
    positive_core_ranks = {
        identifier: rank for rank, identifier in enumerate(positive_core, start=1)
    }
    identifiers = set(category_ranks) | set(positive_core_ranks)
    items: list[FusionItem] = []
    absent_rank = ROUTE_LIMIT + 1
    for identifier in identifiers:
        category_rank = category_ranks.get(identifier, absent_rank)
        positive_core_rank = positive_core_ranks.get(identifier, absent_rank)
        present_ranks = tuple(
            rank
            for rank in (category_rank, positive_core_rank)
            if rank != absent_rank
        )
        score = sum(
            (Fraction(1, RRF_K + rank) for rank in present_ranks),
            start=Fraction(0, 1),
        )
        items.append(
            FusionItem(
                identifier=identifier,
                score=score,
                supporting_route_count=len(present_ranks),
                minimum_route_rank=min(present_ranks),
                category_rank=category_rank,
                positive_core_rank=positive_core_rank,
            )
        )
    ordered_items = tuple(
        sorted(
            items,
            key=lambda item: (
                -item.score,
                -item.supporting_route_count,
                item.minimum_route_rank,
                item.category_rank,
                item.positive_core_rank,
                item.identifier,
            ),
        )
    )
    tail_capacity = max(0, CANDIDATE_CAP - len(stable_prefix))
    tail = tuple(item.identifier for item in ordered_items[:tail_capacity])
    candidates = (*stable_prefix, *tail)
    if (
        tuple(candidates[: len(stable_prefix)]) != stable_prefix
        or len(candidates) > CANDIDATE_CAP
        or len(candidates) != len(set(candidates))
    ):
        raise SparseUnionG0ValidationError("stable RRF append invariant failed")
    return FusionResult(
        candidates=tuple(candidates),
        prefix=stable_prefix,
        tail=tail,
        items=ordered_items,
    )


def _implementation_generation() -> tuple[object, ...]:
    return (
        attribute_registry_sha256,
        build_conversation_constraint_view,
        build_product_attribute_view,
        classify_candidate,
        normalize_value,
        product_slot,
        _canonical_value,
        _queries_from_validated,
        _rules_from_validated,
        _reliable_values,
        _mask_inputs,
        _mask_decision,
        _mask_result,
        fuse_route_candidates,
    )


_EXPECTED_IMPLEMENTATION_GENERATION = _implementation_generation()


class SparseUnionG0Expander:
    """Default-off Agent-backed runtime for the frozen dual-view G0 union."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        enabled: bool = False,
        cache_enabled: bool = False,
    ) -> None:
        if not isinstance(enabled, bool):
            raise SparseUnionG0ValidationError("enabled must be a bool")
        if not isinstance(cache_enabled, bool):
            raise SparseUnionG0ValidationError("cache_enabled must be a bool")
        if not isinstance(catalog_path, (str, Path)):
            raise SparseUnionG0ValidationError("catalog_path must be a path")
        self.catalog_path = Path(catalog_path)
        self.enabled = enabled
        self.cache_enabled = cache_enabled
        self._lock = threading.RLock()
        self._closed = False
        self._agent: Agent | None = None
        self._database_generation: _DatabaseGeneration | None = None
        self._implementation_generation = _EXPECTED_IMPLEMENTATION_GENERATION
        self._cache_capacities = {
            "fts_route": FTS_ROUTE_CACHE_CAPACITY,
            "product_view": PRODUCT_VIEW_CACHE_CAPACITY,
            "mask_decision": MASK_DECISION_CACHE_CAPACITY,
        }
        if any(
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity < 1
            for capacity in self._cache_capacities.values()
        ):
            raise SparseUnionG0ValidationError(
                "cache capacities must be positive integers"
            )
        self._fts_route_cache: OrderedDict[
            tuple[str, str], tuple[tuple[int, str], ...]
        ] = OrderedDict()
        self._product_view_cache: OrderedDict[
            tuple[int, str], ProductAttributeView
        ] = OrderedDict()
        self._mask_decision_cache: OrderedDict[
            tuple[HardConflictRules, int, str], tuple[bool, bool]
        ] = OrderedDict()
        self._cache_counters = {
            "fts_route": _CacheCounters(),
            "product_view": _CacheCounters(),
            "mask_decision": _CacheCounters(),
        }
        self._cache_clears = 0
        self._route_executions = {
            CATEGORY_ROUTE: 0,
            POSITIVE_CORE_ROUTE: 0,
        }
        if enabled:
            try:
                self._agent = Agent(
                    self.catalog_path,
                    llm_client=None,
                    question_policy="fast",
                    trace_sink=None,
                    rerank_mode="off",
                    retrieval_mode="control",
                    p11_mode="off",
                    small_ranker_mode="off",
                )
                self._agent.connection.execute("PRAGMA query_only=ON")
                self._database_generation = self._read_database_generation(
                    self._agent
                )
                self.validate()
            except BaseException:
                with self._lock:
                    if self._cache_clears == 0:
                        self._clear_caches_locked()
                    agent, self._agent = self._agent, None
                    self._database_generation = None
                    self._closed = True
                if agent is not None:
                    try:
                        agent.close()
                    except BaseException:
                        pass
                raise

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise SparseUnionG0ClosedError("sparse union G0 expander is closed")

    @staticmethod
    def _read_database_generation(agent: Agent) -> _DatabaseGeneration:
        columns = tuple(
            str(row[1])
            for row in agent.connection.execute(
                "PRAGMA table_info(products)"
            ).fetchall()
        )
        schema_row = agent.connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='products'"
        ).fetchone()
        schema_sql = re.sub(r"\s+", " ", str(schema_row[0] if schema_row else ""))
        total, distinct_total, empty_total = agent.connection.execute(
            "SELECT COUNT(*),COUNT(DISTINCT parent_asin),"
            "SUM(CASE WHEN parent_asin='' THEN 1 ELSE 0 END) FROM products"
        ).fetchone()
        schema_version_row = agent.connection.execute(
            "PRAGMA schema_version"
        ).fetchone()
        return _DatabaseGeneration(
            total_changes=int(agent.connection.total_changes),
            schema_version=int(schema_version_row[0] if schema_version_row else -1),
            columns=columns,
            schema_sql=schema_sql,
            row_count=int(total),
            distinct_identifier_count=int(distinct_total),
            empty_identifier_count=int(empty_total or 0),
        )

    def _clear_caches_locked(self) -> None:
        self._fts_route_cache.clear()
        self._product_view_cache.clear()
        self._mask_decision_cache.clear()
        self._cache_clears += 1

    def _permanent_close_locked(self) -> Agent | None:
        self._clear_caches_locked()
        self._closed = True
        self._database_generation = None
        agent, self._agent = self._agent, None
        return agent

    def cache_diagnostics(self) -> dict[str, object]:
        """Return aggregate cache counters without exposing semantic keys."""

        with self._lock:
            return {
                "enabled": self.cache_enabled,
                "closed": self._closed,
                "clears": self._cache_clears,
                "fts_route": self._cache_counters["fts_route"].snapshot(
                    size=len(self._fts_route_cache),
                    capacity=self._cache_capacities["fts_route"],
                ),
                "product_view": self._cache_counters["product_view"].snapshot(
                    size=len(self._product_view_cache),
                    capacity=self._cache_capacities["product_view"],
                ),
                "mask_decision": self._cache_counters["mask_decision"].snapshot(
                    size=len(self._mask_decision_cache),
                    capacity=self._cache_capacities["mask_decision"],
                ),
            }

    def route_diagnostics(self) -> dict[str, object]:
        """Return aggregate route execution counts and the frozen registry hash."""

        with self._lock:
            return {
                "category_route_executions": self._route_executions[CATEGORY_ROUTE],
                "positive_core_route_executions": self._route_executions[
                    POSITIVE_CORE_ROUTE
                ],
                "legacy_route_executions": 0,
                "registry_sha256": attribute_registry_sha256(),
                "closed": self._closed,
            }

    def _cache_lookup(
        self,
        layer: str,
        cache: OrderedDict[Any, Any],
        key: object,
    ) -> tuple[bool, Any]:
        counters = self._cache_counters[layer]
        counters.lookups += 1
        value = cache.get(key, _CACHE_MISS)
        if value is _CACHE_MISS:
            counters.misses += 1
            return False, None
        cache.move_to_end(key)
        counters.hits += 1
        counters.avoided_operations += 1
        return True, value

    def _cache_insert(
        self,
        layer: str,
        cache: OrderedDict[Any, Any],
        key: object,
        value: object,
    ) -> None:
        cache[key] = value
        cache.move_to_end(key)
        capacity = self._cache_capacities[layer]
        while len(cache) > capacity:
            cache.popitem(last=False)
            self._cache_counters[layer].evictions += 1

    def _validate_constants_locked(self) -> None:
        if (
            ROUTE_LIMIT != 120
            or TERM_LIMIT != 24
            or RRF_K != 60
            or PREFIX_MINIMUM != 100
            or PREFIX_LIMIT != 200
            or CANDIDATE_CAP != 400
            or tuple(CORE_ATTRIBUTE_SLOTS) != ("material", "style", "use_case")
            or tuple(ROUTE_NAMES) != ("category", "positive_core")
            or tuple(CATEGORY_FIELDS) != ("title", "categories")
            or tuple(POSITIVE_CORE_FIELDS)
            != ("title", "features", "details", "description")
            or tuple(FTS_COLUMNS) != _EXPECTED_FTS_COLUMNS
            or FTS_TOKENIZER != "unicode61 remove_diacritics 2"
            or FTS_QUERY_SQL
            != (
                "SELECT rowid,parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products,0.0,6.0,4.0,2.5,2.5,1.5,1.0),"
                "parent_asin ASC LIMIT 120"
            )
            or FTS_ROUTE_CACHE_CAPACITY != 512
            or PRODUCT_VIEW_CACHE_CAPACITY != 4096
            or MASK_DECISION_CACHE_CAPACITY != 16384
            or _RELIABLE_SOURCES != _EXPECTED_RELIABLE_SOURCES
            or ALLOWED_NEGATIVE_SLOTS != _EXPECTED_ALLOWED_NEGATIVE_SLOTS
            or MIN_EVIDENCE_CONFIDENCE != 0.90
            or _implementation_generation() != self._implementation_generation
            or attribute_registry_sha256()
            != EXPECTED_ATTRIBUTE_REGISTRY_SHA256
        ):
            raise SparseUnionG0ValidationError("frozen G0 constants drifted")
        capacities = {
            "fts_route": FTS_ROUTE_CACHE_CAPACITY,
            "product_view": PRODUCT_VIEW_CACHE_CAPACITY,
            "mask_decision": MASK_DECISION_CACHE_CAPACITY,
        }
        if capacities != self._cache_capacities:
            raise SparseUnionG0ValidationError("cache capacities drifted")

    def _validate_fast_locked(self) -> None:
        self._validate_constants_locked()
        if not self.enabled:
            if self._agent is not None or self._database_generation is not None:
                raise SparseUnionG0ValidationError(
                    "disabled runtime unexpectedly opened an Agent"
                )
            return
        agent = self._agent
        generation = self._database_generation
        if agent is None or generation is None:
            raise SparseUnionG0ValidationError("enabled runtime has no Agent")
        if (
            int(agent.connection.total_changes) != generation.total_changes
            or int(agent.connection.execute("PRAGMA schema_version").fetchone()[0])
            != generation.schema_version
        ):
            raise SparseUnionG0ValidationError("Agent catalog generation drifted")
        query_only_row = agent.connection.execute("PRAGMA query_only").fetchone()
        if not query_only_row or int(query_only_row[0]) != 1:
            raise SparseUnionG0ValidationError("Agent catalog is not query-only")

    def _validate_full_locked(self) -> None:
        self._validate_fast_locked()
        if not self.enabled:
            return
        agent = self._agent
        generation = self._database_generation
        if agent is None or generation is None:
            raise SparseUnionG0ValidationError("enabled runtime has no Agent")
        if (
            agent.p11_mode != "off"
            or agent.small_ranker_mode != "off"
            or agent.rerank_mode != "off"
            or agent.retrieval_mode != "control"
            or agent._p11_bridge is not None
            or agent._small_ranker is not None
        ):
            raise SparseUnionG0ValidationError("Agent mode isolation drifted")
        database_rows = agent.connection.execute("PRAGMA database_list").fetchall()
        if not database_rows or any(str(row[2]) for row in database_rows):
            raise SparseUnionG0ValidationError("Agent catalog is not in-memory")
        current = self._read_database_generation(agent)
        if current.columns != _EXPECTED_FTS_COLUMNS:
            raise SparseUnionG0ValidationError("Agent FTS columns drifted")
        if "tokenize='unicode61 remove_diacritics 2'" not in current.schema_sql:
            raise SparseUnionG0ValidationError("Agent FTS tokenizer drifted")
        if (
            current.row_count != current.distinct_identifier_count
            or current.empty_identifier_count
        ):
            raise SparseUnionG0ValidationError(
                "Agent catalog identifiers are empty or duplicated"
            )
        if current != generation:
            raise SparseUnionG0ValidationError("Agent catalog generation drifted")

    def validate(self) -> None:
        """Fail closed permanently if constants or database generation drifted."""

        with self._lock:
            self._require_open()
            try:
                self._validate_full_locked()
            except BaseException:
                agent_to_close = self._permanent_close_locked()
                if agent_to_close is not None:
                    try:
                        agent_to_close.close()
                    except BaseException:
                        pass
                raise

    @staticmethod
    def _validate_route_query(query: RouteQuery) -> None:
        if query.route == CATEGORY_ROUTE:
            expected_fields = CATEGORY_FIELDS
        elif query.route == POSITIVE_CORE_ROUTE:
            expected_fields = POSITIVE_CORE_FIELDS
        else:
            raise SparseUnionG0ValidationError("unknown G0 route")
        expected_expression = _route_expression(expected_fields, query.terms)
        if (
            not query.activated
            or not query.terms
            or len(query.terms) > TERM_LIMIT
            or query.expression != expected_expression
            or " AND " in query.expression.upper()
        ):
            raise SparseUnionG0ValidationError("G0 route expression drifted")

    def _execute_query_route(
        self, query: RouteQuery
    ) -> tuple[tuple[int, str], ...]:
        self._validate_route_query(query)
        agent = self._agent
        if agent is None:
            raise SparseUnionG0ValidationError("enabled runtime has no Agent")
        rows = tuple(agent.connection.execute(FTS_QUERY_SQL, (query.expression,)))
        self._route_executions[query.route] += 1
        if len(rows) > ROUTE_LIMIT:
            raise SparseUnionG0ValidationError("G0 route exceeded LIMIT 120")
        hits = tuple((int(row[0]), str(row[1])) for row in rows)
        identifiers = tuple(identifier for _rowid, identifier in hits)
        if (
            any(rowid <= 0 or not identifier for rowid, identifier in hits)
            or len(identifiers) != len(set(identifiers))
        ):
            raise SparseUnionG0ValidationError("G0 route shape is invalid")
        return hits

    def _query_route(self, query: RouteQuery) -> tuple[tuple[int, str], ...]:
        if not query.activated:
            return ()
        if not self.cache_enabled:
            return self._execute_query_route(query)
        key = (query.route, query.expression)
        found, cached = self._cache_lookup(
            "fts_route", self._fts_route_cache, key
        )
        if found:
            return cached
        hits = self._execute_query_route(query)
        self._cache_insert("fts_route", self._fts_route_cache, key, hits)
        return hits

    @staticmethod
    def _combined_hits(
        *routes: tuple[tuple[int, str], ...]
    ) -> tuple[tuple[int, str], ...]:
        by_identifier: dict[str, int] = {}
        ordered: list[tuple[int, str]] = []
        for route in routes:
            for rowid, identifier in route:
                previous = by_identifier.get(identifier)
                if previous is not None:
                    if previous != rowid:
                        raise SparseUnionG0ValidationError(
                            "catalog identifier maps to multiple rowids"
                        )
                    continue
                by_identifier[identifier] = rowid
                ordered.append((rowid, identifier))
        return tuple(ordered)

    def _load_product_views(
        self,
        hits: tuple[tuple[int, str], ...],
        identifiers: tuple[str, ...],
    ) -> dict[str, ProductAttributeView]:
        if not identifiers:
            return {}
        agent = self._agent
        if agent is None:
            raise SparseUnionG0ValidationError("enabled runtime has no Agent")
        by_identifier = {identifier: rowid for rowid, identifier in hits}
        if any(identifier not in by_identifier for identifier in identifiers):
            raise SparseUnionG0ValidationError("missing route row identity")
        rowids = [by_identifier[identifier] for identifier in identifiers]
        placeholders = ",".join("?" for _ in rowids)
        rows = agent.connection.execute(
            "SELECT rowid,parent_asin,title,categories,features,details,store,description "
            f"FROM products WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        views: dict[str, ProductAttributeView] = {}
        for row in rows:
            identifier = str(row[1])
            expected_rowid = by_identifier.get(identifier)
            if expected_rowid is None or int(row[0]) != expected_rowid:
                raise SparseUnionG0ValidationError("G0 catalog row identity drifted")
            views[identifier] = build_product_attribute_view(
                {
                    "parent_asin": identifier,
                    "title": str(row[2] or ""),
                    "categories": str(row[3] or ""),
                    "features": str(row[4] or ""),
                    "details": str(row[5] or ""),
                    "store": str(row[6] or ""),
                    "description": str(row[7] or ""),
                }
            )
        if set(views) != set(identifiers):
            raise SparseUnionG0ValidationError("G0 catalog view is incomplete")
        return views

    def _views(
        self,
        hits: tuple[tuple[int, str], ...],
        identifiers: tuple[str, ...],
    ) -> dict[str, ProductAttributeView]:
        if not identifiers:
            return {}
        if not self.cache_enabled:
            return self._load_product_views(hits, identifiers)
        by_identifier = {identifier: rowid for rowid, identifier in hits}
        views: dict[str, ProductAttributeView] = {}
        missing: list[str] = []
        for identifier in identifiers:
            key = (by_identifier[identifier], identifier)
            found, cached = self._cache_lookup(
                "product_view", self._product_view_cache, key
            )
            if found:
                views[identifier] = cached
            else:
                missing.append(identifier)
        if missing:
            missing_order = tuple(missing)
            staged = self._load_product_views(hits, missing_order)
            for identifier in missing_order:
                view = staged[identifier]
                key = (by_identifier[identifier], identifier)
                self._cache_insert(
                    "product_view", self._product_view_cache, key, view
                )
                views[identifier] = view
        return {identifier: views[identifier] for identifier in identifiers}

    def _apply_mask(
        self,
        identifiers: tuple[str, ...],
        views: Mapping[str, ProductAttributeView],
        hits: tuple[tuple[int, str], ...],
        rules: HardConflictRules,
    ) -> HardConflictMaskResult:
        if not self.cache_enabled:
            return apply_hard_conflict_mask(identifiers, views, rules)
        ordered = _ordered_identifiers(identifiers, "identifiers")
        by_identifier = {identifier: rowid for rowid, identifier in hits}
        decisions: dict[str, tuple[bool, bool]] = {}
        staged: list[
            tuple[tuple[HardConflictRules, int, str], tuple[bool, bool]]
        ] = []
        executable_negatives, positive = _mask_inputs(rules)
        for identifier in ordered:
            if identifier not in by_identifier:
                raise SparseUnionG0ValidationError("missing mask row identity")
            key = (rules, by_identifier[identifier], identifier)
            found, cached = self._cache_lookup(
                "mask_decision", self._mask_decision_cache, key
            )
            if found:
                decisions[identifier] = cached
                continue
            view = views.get(identifier)
            if view is None:
                view = ProductAttributeView(parent_asin=identifier)
            if not isinstance(view, ProductAttributeView):
                raise SparseUnionG0ValidationError(
                    "views must contain ProductAttributeView values"
                )
            decision = _mask_decision(view, executable_negatives, positive)
            decisions[identifier] = decision
            staged.append((key, decision))
        result = _mask_result(ordered, decisions)
        for key, decision in staged:
            self._cache_insert(
                "mask_decision", self._mask_decision_cache, key, decision
            )
        return result

    def expand(
        self,
        c200: Iterable[str],
        *,
        category_text: str,
        active_terms: Iterable[str],
        excluded_terms: Iterable[str],
        current_version: int,
        records: Iterable[object],
    ) -> ExpansionResult:
        """Append the masked exact-RRF novel union after untouched C200."""

        with self._lock:
            self._require_open()
            prefix = _prefix(c200)
            if not self.enabled:
                empty_queries = DualRouteQueries(
                    category=RouteQuery(route=CATEGORY_ROUTE, activated=False),
                    positive_core=RouteQuery(
                        route=POSITIVE_CORE_ROUTE, activated=False
                    ),
                )
                return ExpansionResult(
                    candidates=prefix,
                    prefix=prefix,
                    category_route=(),
                    positive_core_route=(),
                    category_novel=(),
                    positive_core_novel=(),
                    category_filtered=(),
                    positive_core_filtered=(),
                    tail=(),
                    enabled=False,
                    activated=False,
                    queries=empty_queries,
                    rules=HardConflictRules(),
                    conflict_count=0,
                    tail_conflict_count=0,
                )

            intent, materialized_records = _validated_context(
                category_text=category_text,
                active_terms=active_terms,
                excluded_terms=excluded_terms,
                current_version=current_version,
                records=records,
            )
            queries = _queries_from_validated(
                intent, materialized_records, current_version
            )
            rules = _rules_from_validated(
                intent, queries, materialized_records, current_version
            )
            try:
                self._validate_fast_locked()
                category_hits = self._query_route(queries.category)
                positive_core_hits = self._query_route(queries.positive_core)
                category_route = tuple(
                    identifier for _rowid, identifier in category_hits
                )
                positive_core_route = tuple(
                    identifier for _rowid, identifier in positive_core_hits
                )
                prefix_members = frozenset(prefix)
                category_novel = tuple(
                    identifier
                    for identifier in category_route
                    if identifier not in prefix_members
                )
                positive_core_novel = tuple(
                    identifier
                    for identifier in positive_core_route
                    if identifier not in prefix_members
                )
                combined_hits = self._combined_hits(
                    category_hits, positive_core_hits
                )
                all_novel = tuple(
                    dict.fromkeys((*category_novel, *positive_core_novel))
                )
                views = self._views(combined_hits, all_novel)
                category_mask = self._apply_mask(
                    category_novel, views, combined_hits, rules
                )
                positive_core_mask = self._apply_mask(
                    positive_core_novel, views, combined_hits, rules
                )
                fusion = fuse_route_candidates(
                    prefix,
                    category_mask.identifiers,
                    positive_core_mask.identifiers,
                )
                tail_views = {
                    identifier: views[identifier]
                    for identifier in fusion.tail
                    if identifier in views
                }
                tail_audit = self._apply_mask(
                    fusion.tail, tail_views, combined_hits, rules
                )
                self._validate_fast_locked()
            except BaseException:
                agent_to_close = self._permanent_close_locked()
                if agent_to_close is not None:
                    try:
                        agent_to_close.close()
                    except BaseException:
                        pass
                raise

            if (
                tuple(fusion.candidates[: len(prefix)]) != prefix
                or fusion.prefix != prefix
                or len(fusion.candidates) > CANDIDATE_CAP
                or len(fusion.candidates) != len(set(fusion.candidates))
                or tail_audit.conflict_count
                or self.route_diagnostics()["legacy_route_executions"] != 0
            ):
                agent_to_close = self._permanent_close_locked()
                if agent_to_close is not None:
                    try:
                        agent_to_close.close()
                    except BaseException:
                        pass
                raise SparseUnionG0ValidationError("G0 stable union invariant failed")
            return ExpansionResult(
                candidates=fusion.candidates,
                prefix=prefix,
                category_route=category_route,
                positive_core_route=positive_core_route,
                category_novel=category_novel,
                positive_core_novel=positive_core_novel,
                category_filtered=category_mask.identifiers,
                positive_core_filtered=positive_core_mask.identifiers,
                tail=fusion.tail,
                enabled=True,
                activated=queries.activated,
                queries=queries,
                rules=rules,
                conflict_count=(
                    category_mask.conflict_count
                    + positive_core_mask.conflict_count
                ),
                tail_conflict_count=tail_audit.conflict_count,
                legacy_route_executions=0,
            )

    def close(self) -> None:
        """Idempotently close the owned Agent and clear all cache state."""

        with self._lock:
            if self._closed:
                return
            agent = self._permanent_close_locked()
        if agent is not None:
            agent.close()

    def __enter__(self) -> "SparseUnionG0Expander":
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def validate() -> None:
    """Validate frozen module-level constants without opening a catalog."""

    runtime = SparseUnionG0Expander(Path("."), enabled=False)
    try:
        runtime.validate()
    finally:
        runtime.close()


__all__ = [
    "CANDIDATE_CAP",
    "CATEGORY_FIELDS",
    "CATEGORY_ROUTE",
    "CORE_ATTRIBUTE_SLOTS",
    "DualRouteQueries",
    "EXPECTED_ATTRIBUTE_REGISTRY_SHA256",
    "ExpansionResult",
    "FTS_COLUMNS",
    "FTS_QUERY_SQL",
    "FTS_ROUTE_CACHE_CAPACITY",
    "FTS_TOKENIZER",
    "FusionItem",
    "FusionResult",
    "HardConflictMaskResult",
    "HardConflictRules",
    "MASK_DECISION_CACHE_CAPACITY",
    "MECHANISM",
    "POSITIVE_CONFLICT_SLOTS",
    "POSITIVE_CORE_FIELDS",
    "POSITIVE_CORE_ROUTE",
    "PREFIX_LIMIT",
    "PREFIX_MINIMUM",
    "PRODUCT_VIEW_CACHE_CAPACITY",
    "ROUTE_LIMIT",
    "ROUTE_NAMES",
    "RRF_K",
    "RouteQuery",
    "SCHEMA_VERSION",
    "SparseUnionG0ClosedError",
    "SparseUnionG0Error",
    "SparseUnionG0Expander",
    "SparseUnionG0ValidationError",
    "TERM_LIMIT",
    "apply_hard_conflict_mask",
    "attribute_registry_sha256",
    "build_route_queries",
    "compile_hard_conflict_rules",
    "fuse_route_candidates",
    "validate",
]
