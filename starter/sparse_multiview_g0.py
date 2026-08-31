"""Deterministic four-view sparse candidate union for v2.22B G0.

This module is deliberately target-blind, diagnostic-only, and default-off. An
enabled runtime executes exactly four independently fielded registry-derived FTS
routes over the Agent's local in-memory SQLite catalog, applies the same
current-version explicit hard-conflict mask to each novel route tail, and then
appends an exact-Fraction RRF tail after an untouched variable-length C200
prefix.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
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


SCHEMA_VERSION = "small-ranker-sparse-multiview-g0.v1"
MECHANISM = "MULTIVIEW_SPARSE_RRF_G0"
EXPECTED_ATTRIBUTE_REGISTRY_SHA256 = (
    "1d85fc42f49fd9374238d98b8feaeab8d76269b0987740256fe60e666757d2ca"
)

FULL_POSITIVE_ROUTE = "full_positive"
EXACT_ACTIVE_ROUTE = "exact_active"
CATEGORY_ONLY_ROUTE = "category_only"
TITLE_STORE_EXACT_ROUTE = "title_store_exact"
ROUTE_NAMES = (
    FULL_POSITIVE_ROUTE,
    EXACT_ACTIVE_ROUTE,
    CATEGORY_ONLY_ROUTE,
    TITLE_STORE_EXACT_ROUTE,
)
ROUTE_LIMIT = 120
TERM_LIMIT = 24
RRF_K = 60
PREFIX_MINIMUM = 100
PREFIX_LIMIT = 200
CANDIDATE_CAP = 400
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
FTS_SCHEMA_SQL = (
    "CREATE VIRTUAL TABLE products USING fts5("
    "parent_asin UNINDEXED, title, categories, features, details, store, "
    "description, tokenize='unicode61 remove_diacritics 2')"
)
FULL_POSITIVE_FIELDS = (
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)
EXACT_ACTIVE_FIELDS = ("title", "features", "details", "store", "description")
CATEGORY_ONLY_FIELDS = ("title", "categories")
TITLE_STORE_EXACT_FIELDS = ("title", "store")
ROUTE_FIELDS = {
    FULL_POSITIVE_ROUTE: FULL_POSITIVE_FIELDS,
    EXACT_ACTIVE_ROUTE: EXACT_ACTIVE_FIELDS,
    CATEGORY_ONLY_ROUTE: CATEGORY_ONLY_FIELDS,
    TITLE_STORE_EXACT_ROUTE: TITLE_STORE_EXACT_FIELDS,
}

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


class SparseMultiviewG0Error(RuntimeError):
    """Base error for the isolated multiview G0 runtime."""


class SparseMultiviewG0ValidationError(ValueError, SparseMultiviewG0Error):
    """Raised when a frozen input or runtime invariant is violated."""


class SparseMultiviewG0ClosedError(SparseMultiviewG0Error):
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
class MultiRouteQueries:
    """The four fixed sparse route queries."""

    full_positive: RouteQuery
    exact_active: RouteQuery
    category_only: RouteQuery
    title_store_exact: RouteQuery

    @property
    def activated(self) -> bool:
        return any(query.activated for _route, query in self.items())

    def items(self) -> tuple[tuple[str, RouteQuery], ...]:
        return (
            (FULL_POSITIVE_ROUTE, self.full_positive),
            (EXACT_ACTIVE_ROUTE, self.exact_active),
            (CATEGORY_ONLY_ROUTE, self.category_only),
            (TITLE_STORE_EXACT_ROUTE, self.title_store_exact),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "activated": self.activated,
            FULL_POSITIVE_ROUTE: self.full_positive.as_dict(),
            EXACT_ACTIVE_ROUTE: self.exact_active.as_dict(),
            CATEGORY_ONLY_ROUTE: self.category_only.as_dict(),
            TITLE_STORE_EXACT_ROUTE: self.title_store_exact.as_dict(),
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
    full_positive_rank: int
    exact_active_rank: int
    category_only_rank: int
    title_store_exact_rank: int

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "score_numerator": self.score.numerator,
            "score_denominator": self.score.denominator,
            "supporting_route_count": self.supporting_route_count,
            "minimum_route_rank": self.minimum_route_rank,
            "full_positive_rank": self.full_positive_rank,
            "exact_active_rank": self.exact_active_rank,
            "category_only_rank": self.category_only_rank,
            "title_store_exact_rank": self.title_store_exact_rank,
        }


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Stable prefix plus a bounded exact-RRF novel tail."""

    candidates: tuple[str, ...]
    prefix: tuple[str, ...]
    tail: tuple[str, ...]
    items: tuple[FusionItem, ...]

    @property
    def multiroute_support_count(self) -> int:
        return sum(item.supporting_route_count >= 2 for item in self.items)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidates": list(self.candidates),
            "prefix": list(self.prefix),
            "tail": list(self.tail),
            "items": [item.as_dict() for item in self.items],
            "multiroute_support_count": self.multiroute_support_count,
        }


@dataclass(frozen=True, slots=True)
class ExpansionResult:
    """Complete four-view expansion with an untouched C200 ordered prefix."""

    candidates: tuple[str, ...]
    prefix: tuple[str, ...]
    full_positive_route: tuple[str, ...]
    exact_active_route: tuple[str, ...]
    category_only_route: tuple[str, ...]
    title_store_exact_route: tuple[str, ...]
    full_positive_novel: tuple[str, ...]
    exact_active_novel: tuple[str, ...]
    category_only_novel: tuple[str, ...]
    title_store_exact_novel: tuple[str, ...]
    full_positive_filtered: tuple[str, ...]
    exact_active_filtered: tuple[str, ...]
    category_only_filtered: tuple[str, ...]
    title_store_exact_filtered: tuple[str, ...]
    full_positive_filtered_ranked: tuple[tuple[str, int], ...]
    exact_active_filtered_ranked: tuple[tuple[str, int], ...]
    category_only_filtered_ranked: tuple[tuple[str, int], ...]
    title_store_exact_filtered_ranked: tuple[tuple[str, int], ...]
    tail: tuple[str, ...]
    enabled: bool
    activated: bool
    queries: MultiRouteQueries
    rules: HardConflictRules
    conflict_count: int
    tail_conflict_count: int
    fusion_items: tuple[FusionItem, ...] = ()
    route_latency_ns: tuple[tuple[str, int], ...] = ()
    hard_mask_latency_ns: int = 0
    legacy_route_executions: int = 0

    @property
    def full_positive_activated(self) -> bool:
        return self.queries.full_positive.activated

    @property
    def exact_active_activated(self) -> bool:
        return self.queries.exact_active.activated

    @property
    def category_only_activated(self) -> bool:
        return self.queries.category_only.activated

    @property
    def title_store_exact_activated(self) -> bool:
        return self.queries.title_store_exact.activated

    @property
    def multiroute_support_count(self) -> int:
        filtered = (
            set(self.full_positive_filtered),
            set(self.exact_active_filtered),
            set(self.category_only_filtered),
            set(self.title_store_exact_filtered),
        )
        count = 0
        all_identifiers = set().union(*filtered)
        for identifier in all_identifiers:
            if sum(identifier in values for values in filtered) >= 2:
                count += 1
        return count

    @property
    def union_novel_count(self) -> int:
        return len(
            set(self.full_positive_filtered)
            | set(self.exact_active_filtered)
            | set(self.category_only_filtered)
            | set(self.title_store_exact_filtered)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mechanism": MECHANISM,
            "enabled": self.enabled,
            "activated": self.activated,
            "full_positive_activated": self.full_positive_activated,
            "exact_active_activated": self.exact_active_activated,
            "category_only_activated": self.category_only_activated,
            "title_store_exact_activated": self.title_store_exact_activated,
            "candidates": list(self.candidates),
            "prefix": list(self.prefix),
            "full_positive_route": list(self.full_positive_route),
            "exact_active_route": list(self.exact_active_route),
            "category_only_route": list(self.category_only_route),
            "title_store_exact_route": list(self.title_store_exact_route),
            "full_positive_novel": list(self.full_positive_novel),
            "exact_active_novel": list(self.exact_active_novel),
            "category_only_novel": list(self.category_only_novel),
            "title_store_exact_novel": list(self.title_store_exact_novel),
            "full_positive_filtered": list(self.full_positive_filtered),
            "exact_active_filtered": list(self.exact_active_filtered),
            "category_only_filtered": list(self.category_only_filtered),
            "title_store_exact_filtered": list(self.title_store_exact_filtered),
            "full_positive_filtered_ranked": [
                [identifier, rank]
                for identifier, rank in self.full_positive_filtered_ranked
            ],
            "exact_active_filtered_ranked": [
                [identifier, rank]
                for identifier, rank in self.exact_active_filtered_ranked
            ],
            "category_only_filtered_ranked": [
                [identifier, rank]
                for identifier, rank in self.category_only_filtered_ranked
            ],
            "title_store_exact_filtered_ranked": [
                [identifier, rank]
                for identifier, rank in self.title_store_exact_filtered_ranked
            ],
            "tail": list(self.tail),
            "conflict_count": self.conflict_count,
            "tail_conflict_count": self.tail_conflict_count,
            "multiroute_support_count": self.multiroute_support_count,
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
class _CatalogEvidence:
    raw: bytes
    offsets: Mapping[str, tuple[int, int]]

    def __len__(self) -> int:
        return len(self.offsets)

    def view(self, identifier: str) -> ProductAttributeView:
        bounds = self.offsets.get(identifier)
        if bounds is None:
            raise SparseMultiviewG0ValidationError(
                "G0 catalog evidence identifier is missing"
            )
        start, end = bounds
        try:
            product = json.loads(self.raw[start:end])
        except (UnicodeError, json.JSONDecodeError) as error:
            raise SparseMultiviewG0ValidationError(
                "G0 catalog evidence row cannot be decoded"
            ) from error
        if not isinstance(product, Mapping) or product.get("parent_asin") != identifier:
            raise SparseMultiviewG0ValidationError(
                "G0 catalog evidence identity drifted"
            )
        view = build_product_attribute_view(product)
        if view.parent_asin != identifier:
            raise SparseMultiviewG0ValidationError(
                "G0 catalog attribute-view identity drifted"
            )
        return view


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
        raise SparseMultiviewG0ValidationError(f"slot has no frozen registry: {slot}")
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
            raise SparseMultiviewG0ValidationError(
                f"ledger record is missing required field: {name}"
            )
        return record[name]
    if not hasattr(record, name):
        raise SparseMultiviewG0ValidationError(
            f"ledger record is missing required field: {name}"
        )
    return getattr(record, name)


def _records(records: Iterable[object]) -> tuple[_Record, ...]:
    if isinstance(records, (str, bytes, Mapping)):
        raise SparseMultiviewG0ValidationError("records must be an iterable of records")
    try:
        materialized = tuple(records)
    except TypeError as error:
        raise SparseMultiviewG0ValidationError("records must be iterable") from error
    result: list[_Record] = []
    for order, record in enumerate(materialized, start=1):
        status = _field(record, "status")
        version = _field(record, "version")
        polarity = _field(record, "polarity")
        source_turn = _field(record, "source_turn")
        if not isinstance(status, str) or not status:
            raise SparseMultiviewG0ValidationError(
                "ledger status must be a non-empty string"
            )
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise SparseMultiviewG0ValidationError("ledger version must be positive")
        if polarity not in (-1, 1) or isinstance(polarity, bool):
            raise SparseMultiviewG0ValidationError("ledger polarity must be -1 or +1")
        if (
            not isinstance(source_turn, int)
            or isinstance(source_turn, bool)
            or source_turn < 0
        ):
            raise SparseMultiviewG0ValidationError(
                "ledger source_turn must be non-negative"
            )
        slot = normalize_value(_field(record, "slot")).replace(" ", "_")
        value = normalize_value(_field(record, "value"))
        hardness = normalize_value(_field(record, "hardness"))
        if not slot or not value or hardness not in {"hard", "soft"}:
            raise SparseMultiviewG0ValidationError("ledger slot/value/hardness is invalid")
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
        raise SparseMultiviewG0ValidationError(
            f"{name} must be an iterable of strings"
        )
    try:
        result = tuple(values)
    except TypeError as error:
        raise SparseMultiviewG0ValidationError(f"{name} must be iterable") from error
    if any(not isinstance(value, str) for value in result):
        raise SparseMultiviewG0ValidationError(f"{name} must contain only strings")
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
        raise SparseMultiviewG0ValidationError("category_text must be a string")
    if (
        not isinstance(current_version, int)
        or isinstance(current_version, bool)
        or current_version < 1
    ):
        raise SparseMultiviewG0ValidationError(
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


def _route_terms(values: Iterable[str]) -> tuple[str, ...]:
    """Apply the frozen lexical unique/cap operation once, after route union."""

    return tuple(
        sorted(
            {
                normalized
                for value in values
                if (normalized := normalize_value(value))
            }
        )
    )[:TERM_LIMIT]


def _exact_active_terms(intent: ConversationConstraintView) -> tuple[str, ...]:
    """Return the complete exact-term source; each route applies its own cap."""

    return tuple(
        sorted(
            {
                normalize_value(term)
                for term in intent.exact_terms
                if normalize_value(term)
            }
        )
    )


def _positive_surface_terms(
    current: tuple[_Record, ...],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    values = tuple(
        sorted(
            {
                (record.slot, canonical)
                for record in current
                if record.polarity == 1 and record.slot in POSITIVE_CONFLICT_SLOTS
                if (canonical := _canonical_value(record.slot, record.value)) is not None
            }
        )
    )
    surfaces: set[str] = set()
    for slot, canonical in values:
        surfaces.update(_registry_terms(slot, (canonical,)))
    return values, tuple(sorted(surfaces))


def _queries_from_validated(
    intent: ConversationConstraintView,
    records: tuple[_Record, ...],
    current_version: int,
) -> MultiRouteQueries:
    current = _current_records(records, current_version)
    category_values = tuple(
        sorted(
            {
                canonical
                for value in intent.category_terms
                if (canonical := _canonical_value("category", value)) is not None
            }
        )
    )
    category_source_terms = _registry_terms("category", category_values)
    exact_source_terms = _exact_active_terms(intent)
    positive_values, positive_terms = _positive_surface_terms(current)
    full_positive_terms = _route_terms(
        (*category_source_terms, *positive_terms, *exact_source_terms)
    )
    exact_terms = _route_terms(exact_source_terms)
    category_terms = _route_terms(category_source_terms)
    full_positive_values = tuple(
        sorted(
            set(("category", value) for value in category_values)
            | set(positive_values)
        )
    )

    full_positive = RouteQuery(
        route=FULL_POSITIVE_ROUTE,
        activated=bool(full_positive_terms),
        canonical_values=full_positive_values,
        terms=full_positive_terms,
        expression=_route_expression(FULL_POSITIVE_FIELDS, full_positive_terms),
    )
    exact_active = RouteQuery(
        route=EXACT_ACTIVE_ROUTE,
        activated=bool(exact_terms),
        canonical_values=(),
        terms=exact_terms,
        expression=_route_expression(EXACT_ACTIVE_FIELDS, exact_terms),
    )
    category_only = RouteQuery(
        route=CATEGORY_ONLY_ROUTE,
        activated=bool(category_terms),
        canonical_values=tuple(("category", value) for value in category_values),
        terms=category_terms,
        expression=_route_expression(CATEGORY_ONLY_FIELDS, category_terms),
    )
    title_store_exact = RouteQuery(
        route=TITLE_STORE_EXACT_ROUTE,
        activated=bool(exact_terms),
        canonical_values=(),
        terms=exact_terms,
        expression=_route_expression(TITLE_STORE_EXACT_FIELDS, exact_terms),
    )
    return MultiRouteQueries(
        full_positive=full_positive,
        exact_active=exact_active,
        category_only=category_only,
        title_store_exact=title_store_exact,
    )


def build_route_queries(
    *,
    category_text: str,
    active_terms: Iterable[str],
    excluded_terms: Iterable[str],
    current_version: int,
    records: Iterable[object],
) -> MultiRouteQueries:
    """Build all four independent routes without touching SQLite."""

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
    queries: MultiRouteQueries,
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
            value
            for slot, value in queries.category_only.canonical_values
            if slot == "category"
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
    ordered: tuple[str, ...], decisions: Mapping[str, tuple[bool, bool]]
) -> HardConflictMaskResult:
    kept: list[str] = []
    dropped: list[str] = []
    negative_violation_count = 0
    positive_conflict_count = 0
    for identifier in ordered:
        decision = decisions.get(identifier)
        if decision is None:
            raise SparseMultiviewG0ValidationError("missing mask decision")
        negative_violation, positive_conflict = decision
        if negative_violation or positive_conflict:
            dropped.append(identifier)
            negative_violation_count += int(negative_violation)
            positive_conflict_count += int(positive_conflict)
        else:
            kept.append(identifier)
    return HardConflictMaskResult(
        identifiers=tuple(kept),
        dropped=tuple(dropped),
        negative_violation_count=negative_violation_count,
        positive_conflict_count=positive_conflict_count,
    )


def apply_hard_conflict_mask(
    identifiers: Iterable[str],
    views: Mapping[str, ProductAttributeView],
    rules: HardConflictRules,
) -> HardConflictMaskResult:
    if isinstance(identifiers, (str, bytes, Mapping)):
        raise SparseMultiviewG0ValidationError("identifiers must be an ordered iterable")
    ordered = tuple(identifiers)
    if any(not isinstance(identifier, str) or not identifier for identifier in ordered):
        raise SparseMultiviewG0ValidationError("identifiers must be non-empty strings")
    if len(ordered) != len(set(ordered)):
        raise SparseMultiviewG0ValidationError("identifiers must be unique")
    if not isinstance(views, Mapping):
        raise SparseMultiviewG0ValidationError("views must be a mapping")
    if not isinstance(rules, HardConflictRules):
        raise SparseMultiviewG0ValidationError("rules must be HardConflictRules")
    executable_negatives, positive = _mask_inputs(rules)
    decisions: dict[str, tuple[bool, bool]] = {}
    for identifier in ordered:
        view = views.get(identifier)
        if view is None:
            view = ProductAttributeView(parent_asin=identifier)
        if not isinstance(view, ProductAttributeView):
            raise SparseMultiviewG0ValidationError(
                "views must contain ProductAttributeView values"
            )
        decisions[identifier] = _mask_decision(view, executable_negatives, positive)
    return _mask_result(ordered, decisions)


def _ordered_identifiers(values: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise SparseMultiviewG0ValidationError(f"{name} must be an ordered iterable")
    try:
        ordered = tuple(values)
    except TypeError as error:
        raise SparseMultiviewG0ValidationError(f"{name} must be iterable") from error
    if any(not isinstance(value, str) or not value for value in ordered):
        raise SparseMultiviewG0ValidationError(
            f"{name} identifiers must be non-empty strings"
        )
    if len(ordered) != len(set(ordered)):
        raise SparseMultiviewG0ValidationError(f"{name} identifiers must be unique")
    return ordered


def _prefix(c200: Iterable[str]) -> tuple[str, ...]:
    prefix = _ordered_identifiers(c200, "c200")
    if not PREFIX_MINIMUM <= len(prefix) <= PREFIX_LIMIT:
        raise SparseMultiviewG0ValidationError("c200 prefix cardinality is invalid")
    return prefix


def _ranked_route_candidates(
    values: Iterable[str | tuple[str, int]], name: str
) -> tuple[tuple[str, int], ...]:
    """Validate route survivors while retaining original one-based FTS ranks.

    Plain identifiers remain supported for the public pure helper and receive
    contiguous ranks.  The Agent-backed path always supplies explicit ranks so
    prefix exclusion and the hard mask leave their original holes intact.
    """

    if isinstance(values, (str, bytes, Mapping)):
        raise SparseMultiviewG0ValidationError(f"{name} must be an ordered iterable")
    try:
        materialized = tuple(values)
    except TypeError as error:
        raise SparseMultiviewG0ValidationError(f"{name} must be iterable") from error
    ranked: list[tuple[str, int]] = []
    for fallback_rank, value in enumerate(materialized, start=1):
        if isinstance(value, str):
            identifier, rank = value, fallback_rank
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            identifier, rank = value
        else:
            raise SparseMultiviewG0ValidationError(
                f"{name} values must be identifiers or (identifier, rank) pairs"
            )
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 1 <= rank <= ROUTE_LIMIT
        ):
            raise SparseMultiviewG0ValidationError(
                f"{name} contains an invalid identifier or rank"
            )
        ranked.append((identifier, rank))
    identifiers = tuple(identifier for identifier, _rank in ranked)
    ranks = tuple(rank for _identifier, rank in ranked)
    if (
        len(ranked) > ROUTE_LIMIT
        or len(identifiers) != len(set(identifiers))
        or len(ranks) != len(set(ranks))
        or any(right <= left for left, right in zip(ranks, ranks[1:]))
    ):
        raise SparseMultiviewG0ValidationError(
            f"{name} must have unique identifiers and strictly increasing ranks"
        )
    return tuple(ranked)


def fuse_route_candidates(
    stable_prefix: Iterable[str],
    full_positive_filtered: Iterable[str | tuple[str, int]],
    exact_active_filtered: Iterable[str | tuple[str, int]],
    category_only_filtered: Iterable[str | tuple[str, int]],
    title_store_exact_filtered: Iterable[str | tuple[str, int]],
) -> FusionResult:
    prefix = _prefix(stable_prefix)
    prefix_members = frozenset(prefix)
    ranked_by_route = {
        FULL_POSITIVE_ROUTE: _ranked_route_candidates(
            full_positive_filtered, "full_positive_filtered"
        ),
        EXACT_ACTIVE_ROUTE: _ranked_route_candidates(
            exact_active_filtered, "exact_active_filtered"
        ),
        CATEGORY_ONLY_ROUTE: _ranked_route_candidates(
            category_only_filtered, "category_only_filtered"
        ),
        TITLE_STORE_EXACT_ROUTE: _ranked_route_candidates(
            title_store_exact_filtered, "title_store_exact_filtered"
        ),
    }
    if any(
        prefix_members.intersection(identifier for identifier, _rank in values)
        for values in ranked_by_route.values()
    ):
        raise SparseMultiviewG0ValidationError("route contains a prefix identifier")
    rank_maps = {
        route: dict(values) for route, values in ranked_by_route.items()
    }
    identifiers = set().union(*rank_maps.values())
    absent_rank = ROUTE_LIMIT + 1
    items: list[FusionItem] = []
    for identifier in identifiers:
        route_ranks = {
            route: ranks.get(identifier, absent_rank)
            for route, ranks in rank_maps.items()
        }
        present_ranks = [
            rank for rank in route_ranks.values() if rank != absent_rank
        ]
        score = sum(
            (Fraction(1, RRF_K + rank) for rank in present_ranks),
            Fraction(0, 1),
        )
        items.append(
            FusionItem(
                identifier=identifier,
                score=score,
                supporting_route_count=len(present_ranks),
                minimum_route_rank=min(present_ranks),
                full_positive_rank=route_ranks[FULL_POSITIVE_ROUTE],
                exact_active_rank=route_ranks[EXACT_ACTIVE_ROUTE],
                category_only_rank=route_ranks[CATEGORY_ONLY_ROUTE],
                title_store_exact_rank=route_ranks[TITLE_STORE_EXACT_ROUTE],
            )
        )
    ordered_items = tuple(
        sorted(
            items,
            key=lambda item: (
                -item.score,
                -item.supporting_route_count,
                item.minimum_route_rank,
                item.full_positive_rank,
                item.exact_active_rank,
                item.category_only_rank,
                item.title_store_exact_rank,
                item.identifier,
            ),
        )
    )
    tail_capacity = max(0, CANDIDATE_CAP - len(prefix))
    tail = tuple(item.identifier for item in ordered_items[:tail_capacity])
    candidates = (*prefix, *tail)
    if (
        tuple(candidates[: len(prefix)]) != prefix
        or len(candidates) > CANDIDATE_CAP
        or len(candidates) != len(set(candidates))
    ):
        raise SparseMultiviewG0ValidationError("stable RRF append invariant failed")
    return FusionResult(
        candidates=tuple(candidates),
        prefix=prefix,
        tail=tail,
        items=ordered_items,
    )


def _build_catalog_evidence(catalog_path: Path) -> _CatalogEvidence:
    """Index immutable raw structured rows for bounded lazy attribute views.

    FTS text is intentionally not used for masking: Agent flattens mappings and
    lists while indexing, which would lose the preregistered details-key grammar.
    """

    offsets: dict[str, tuple[int, int]] = {}
    try:
        raw = catalog_path.read_bytes()
        line_number = 0
        start = 0
        while start < len(raw):
            line_number += 1
            newline = raw.find(b"\n", start)
            end = len(raw) if newline < 0 else newline
            line = raw[start:end]
            if line.strip():
                product = json.loads(line)
                if not isinstance(product, Mapping):
                    raise SparseMultiviewG0ValidationError(
                        f"catalog row {line_number} is not a mapping"
                    )
                identifier_value = product.get("parent_asin")
                if not isinstance(identifier_value, str) or not identifier_value:
                    raise SparseMultiviewG0ValidationError(
                        f"catalog row {line_number} has no identifier"
                    )
                if identifier_value in offsets:
                    raise SparseMultiviewG0ValidationError(
                        "catalog identifiers must be unique"
                    )
                offsets[identifier_value] = (start, end)
            if newline < 0:
                break
            start = newline + 1
    except SparseMultiviewG0ValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SparseMultiviewG0ValidationError(
            "catalog attribute-view construction failed"
        ) from error
    if not offsets:
        raise SparseMultiviewG0ValidationError("catalog contains no products")
    return _CatalogEvidence(raw=raw, offsets=MappingProxyType(offsets))


def _implementation_generation() -> tuple[object, ...]:
    return (
        attribute_registry_sha256,
        build_conversation_constraint_view,
        build_product_attribute_view,
        classify_candidate,
        normalize_value,
        product_slot,
        _build_catalog_evidence,
        _canonical_value,
        _route_expression,
        _route_terms,
        _queries_from_validated,
        _rules_from_validated,
        _reliable_values,
        _mask_inputs,
        _mask_decision,
        _mask_result,
        _ranked_route_candidates,
        fuse_route_candidates,
    )


_EXPECTED_IMPLEMENTATION_GENERATION = _implementation_generation()


class SparseMultiviewG0Expander:
    """Default-off Agent-backed runtime for the frozen four-view G0 union."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        enabled: bool = False,
        cache_enabled: bool = False,
    ) -> None:
        if not isinstance(enabled, bool):
            raise SparseMultiviewG0ValidationError("enabled must be a bool")
        if not isinstance(cache_enabled, bool):
            raise SparseMultiviewG0ValidationError("cache_enabled must be a bool")
        if not isinstance(catalog_path, (str, Path)):
            raise SparseMultiviewG0ValidationError("catalog_path must be a path")
        self.catalog_path = Path(catalog_path)
        self.enabled = enabled
        self.cache_enabled = cache_enabled
        self._lock = threading.RLock()
        self._closed = False
        self._agent: Agent | None = None
        self._database_generation: _DatabaseGeneration | None = None
        self._catalog_evidence: _CatalogEvidence | None = None
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
            raise SparseMultiviewG0ValidationError(
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
        self._route_executions = {route: 0 for route in ROUTE_NAMES}
        if self.enabled:
            try:
                self._agent = Agent(
                    self.catalog_path,
                    rerank_mode="off",
                    retrieval_mode="control",
                    small_ranker_mode="off",
                    p11_mode="off",
                )
                self._agent.connection.execute("PRAGMA query_only=ON")
                self._database_generation = self._read_database_generation(self._agent)
                self._catalog_evidence = _build_catalog_evidence(self.catalog_path)
                self._bind_catalog_evidence_locked()
                self._validate_full_locked()
            except BaseException:
                agent_to_close = self._permanent_close_locked()
                if agent_to_close is not None:
                    try:
                        agent_to_close.close()
                    except BaseException:
                        pass
                raise

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise SparseMultiviewG0ClosedError("runtime is permanently closed")

    def _permanent_close_locked(self) -> Agent | None:
        self._closed = True
        self._database_generation = None
        self._catalog_evidence = None
        self._clear_caches_locked()
        agent, self._agent = self._agent, None
        return agent

    @staticmethod
    def _read_database_generation(agent: Agent) -> _DatabaseGeneration:
        columns = tuple(
            str(row[1])
            for row in agent.connection.execute("PRAGMA table_info(products)")
        )
        schema_row = agent.connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='products'"
        ).fetchone()
        schema_sql = re.sub(
            r"\s+", " ", str(schema_row[0] if schema_row else "")
        ).strip()
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

    def _bind_catalog_evidence_locked(self) -> None:
        agent = self._agent
        generation = self._database_generation
        evidence = self._catalog_evidence
        if agent is None or generation is None or evidence is None:
            raise SparseMultiviewG0ValidationError(
                "catalog evidence binding is incomplete"
            )
        if len(evidence) != generation.row_count:
            raise SparseMultiviewG0ValidationError(
                "catalog evidence row count drifted"
            )
        observed = 0
        for row in agent.connection.execute("SELECT parent_asin FROM products"):
            identifier = str(row[0])
            if identifier not in evidence.offsets:
                raise SparseMultiviewG0ValidationError(
                    "catalog evidence identifier drifted"
                )
            observed += 1
        if observed != len(evidence):
            raise SparseMultiviewG0ValidationError(
                "catalog evidence identifier cardinality drifted"
            )

    def _clear_caches_locked(self) -> None:
        self._fts_route_cache.clear()
        self._product_view_cache.clear()
        self._mask_decision_cache.clear()
        self._cache_clears += 1

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
                "full_positive_route_executions": self._route_executions[
                    FULL_POSITIVE_ROUTE
                ],
                "exact_active_route_executions": self._route_executions[
                    EXACT_ACTIVE_ROUTE
                ],
                "category_only_route_executions": self._route_executions[
                    CATEGORY_ONLY_ROUTE
                ],
                "title_store_exact_route_executions": self._route_executions[
                    TITLE_STORE_EXACT_ROUTE
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
            or tuple(POSITIVE_CONFLICT_SLOTS)
            != (
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
            or tuple(ROUTE_NAMES)
            != (
                "full_positive",
                "exact_active",
                "category_only",
                "title_store_exact",
            )
            or tuple(FULL_POSITIVE_FIELDS)
            != ("title", "categories", "features", "details", "store", "description")
            or tuple(EXACT_ACTIVE_FIELDS)
            != ("title", "features", "details", "store", "description")
            or tuple(CATEGORY_ONLY_FIELDS) != ("title", "categories")
            or tuple(TITLE_STORE_EXACT_FIELDS) != ("title", "store")
            or tuple(FTS_COLUMNS) != _EXPECTED_FTS_COLUMNS
            or FTS_TOKENIZER != "unicode61 remove_diacritics 2"
            or FTS_QUERY_SQL
            != (
                "SELECT rowid,parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products,0.0,6.0,4.0,2.5,2.5,1.5,1.0),"
                "parent_asin ASC LIMIT 120"
            )
            or FTS_SCHEMA_SQL
            != (
                "CREATE VIRTUAL TABLE products USING fts5("
                "parent_asin UNINDEXED, title, categories, features, details, "
                "store, description, tokenize='unicode61 remove_diacritics 2')"
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
            raise SparseMultiviewG0ValidationError("frozen G0 constants drifted")
        capacities = {
            "fts_route": FTS_ROUTE_CACHE_CAPACITY,
            "product_view": PRODUCT_VIEW_CACHE_CAPACITY,
            "mask_decision": MASK_DECISION_CACHE_CAPACITY,
        }
        if capacities != self._cache_capacities:
            raise SparseMultiviewG0ValidationError("cache capacities drifted")

    def _validate_fast_locked(self) -> None:
        self._validate_constants_locked()
        if not self.enabled:
            if (
                self._agent is not None
                or self._database_generation is not None
                or self._catalog_evidence is not None
            ):
                raise SparseMultiviewG0ValidationError(
                    "disabled runtime unexpectedly opened an Agent"
                )
            return
        agent = self._agent
        generation = self._database_generation
        evidence = self._catalog_evidence
        if agent is None or generation is None or evidence is None:
            raise SparseMultiviewG0ValidationError("enabled runtime has no Agent")
        if len(evidence) != generation.row_count:
            raise SparseMultiviewG0ValidationError(
                "catalog evidence generation drifted"
            )
        if (
            int(agent.connection.total_changes) != generation.total_changes
            or int(agent.connection.execute("PRAGMA schema_version").fetchone()[0])
            != generation.schema_version
        ):
            raise SparseMultiviewG0ValidationError("Agent catalog generation drifted")
        query_only_row = agent.connection.execute("PRAGMA query_only").fetchone()
        if not query_only_row or int(query_only_row[0]) != 1:
            raise SparseMultiviewG0ValidationError("Agent catalog is not query-only")

    def _validate_full_locked(self) -> None:
        self._validate_fast_locked()
        if not self.enabled:
            return
        agent = self._agent
        generation = self._database_generation
        evidence = self._catalog_evidence
        if agent is None or generation is None or evidence is None:
            raise SparseMultiviewG0ValidationError("enabled runtime has no Agent")
        if (
            agent.p11_mode != "off"
            or agent.small_ranker_mode != "off"
            or agent.rerank_mode != "off"
            or agent.retrieval_mode != "control"
            or agent._p11_bridge is not None
            or agent._small_ranker is not None
        ):
            raise SparseMultiviewG0ValidationError("Agent mode isolation drifted")
        database_rows = agent.connection.execute("PRAGMA database_list").fetchall()
        if not database_rows or any(str(row[2]) for row in database_rows):
            raise SparseMultiviewG0ValidationError("Agent catalog is not in-memory")
        current = self._read_database_generation(agent)
        if current.columns != _EXPECTED_FTS_COLUMNS:
            raise SparseMultiviewG0ValidationError("Agent FTS columns drifted")
        if current.schema_sql != re.sub(r"\s+", " ", FTS_SCHEMA_SQL).strip():
            raise SparseMultiviewG0ValidationError("Agent FTS schema drifted")
        if (
            current.row_count != current.distinct_identifier_count
            or current.empty_identifier_count
        ):
            raise SparseMultiviewG0ValidationError(
                "Agent catalog identifiers are empty or duplicated"
            )
        if current != generation:
            raise SparseMultiviewG0ValidationError("Agent catalog generation drifted")
        if len(evidence) != current.row_count:
            raise SparseMultiviewG0ValidationError(
                "catalog evidence row count drifted"
            )

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
        fields = ROUTE_FIELDS.get(query.route)
        if fields is None:
            raise SparseMultiviewG0ValidationError("unknown G0 route")
        expected_expression = _route_expression(fields, query.terms)
        if (
            not query.activated
            or not query.terms
            or len(query.terms) > TERM_LIMIT
            or query.terms != _route_terms(query.terms)
            or query.expression != expected_expression
        ):
            raise SparseMultiviewG0ValidationError("G0 route expression drifted")

    def _execute_query_route(
        self, query: RouteQuery
    ) -> tuple[tuple[tuple[int, str], ...], int]:
        self._validate_route_query(query)
        agent = self._agent
        if agent is None:
            raise SparseMultiviewG0ValidationError("enabled runtime has no Agent")
        started = time.perf_counter_ns()
        rows = tuple(agent.connection.execute(FTS_QUERY_SQL, (query.expression,)))
        elapsed_ns = max(1, time.perf_counter_ns() - started)
        self._route_executions[query.route] += 1
        if len(rows) > ROUTE_LIMIT:
            raise SparseMultiviewG0ValidationError("G0 route exceeded LIMIT 120")
        hits = tuple((int(row[0]), str(row[1])) for row in rows)
        identifiers = tuple(identifier for _rowid, identifier in hits)
        if (
            any(rowid <= 0 or not identifier for rowid, identifier in hits)
            or len(identifiers) != len(set(identifiers))
        ):
            raise SparseMultiviewG0ValidationError("G0 route shape is invalid")
        return hits, elapsed_ns

    def _query_route(
        self, query: RouteQuery
    ) -> tuple[tuple[tuple[int, str], ...], int]:
        if not query.activated:
            return (), 0
        if not self.cache_enabled:
            return self._execute_query_route(query)
        key = (query.route, query.expression)
        found, cached = self._cache_lookup(
            "fts_route", self._fts_route_cache, key
        )
        if found:
            return cached, 0
        hits, elapsed_ns = self._execute_query_route(query)
        self._cache_insert("fts_route", self._fts_route_cache, key, hits)
        return hits, elapsed_ns

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
                        raise SparseMultiviewG0ValidationError(
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
        catalog_evidence = self._catalog_evidence
        if agent is None or catalog_evidence is None:
            raise SparseMultiviewG0ValidationError("enabled runtime has no Agent")
        by_identifier = {identifier: rowid for rowid, identifier in hits}
        if any(identifier not in by_identifier for identifier in identifiers):
            raise SparseMultiviewG0ValidationError("missing route row identity")
        views: dict[str, ProductAttributeView] = {}
        for identifier in identifiers:
            view = catalog_evidence.view(identifier)
            views[identifier] = view
        if set(views) != set(identifiers):
            raise SparseMultiviewG0ValidationError("G0 catalog view is incomplete")
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
        staged: list[tuple[tuple[HardConflictRules, int, str], tuple[bool, bool]]] = []
        executable_negatives, positive = _mask_inputs(rules)
        for identifier in ordered:
            if identifier not in by_identifier:
                raise SparseMultiviewG0ValidationError("missing mask row identity")
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
                raise SparseMultiviewG0ValidationError(
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
                empty_queries = MultiRouteQueries(
                    full_positive=RouteQuery(
                        route=FULL_POSITIVE_ROUTE, activated=False
                    ),
                    exact_active=RouteQuery(
                        route=EXACT_ACTIVE_ROUTE, activated=False
                    ),
                    category_only=RouteQuery(
                        route=CATEGORY_ONLY_ROUTE, activated=False
                    ),
                    title_store_exact=RouteQuery(
                        route=TITLE_STORE_EXACT_ROUTE, activated=False
                    ),
                )
                return ExpansionResult(
                    candidates=prefix,
                    prefix=prefix,
                    full_positive_route=(),
                    exact_active_route=(),
                    category_only_route=(),
                    title_store_exact_route=(),
                    full_positive_novel=(),
                    exact_active_novel=(),
                    category_only_novel=(),
                    title_store_exact_novel=(),
                    full_positive_filtered=(),
                    exact_active_filtered=(),
                    category_only_filtered=(),
                    title_store_exact_filtered=(),
                    full_positive_filtered_ranked=(),
                    exact_active_filtered_ranked=(),
                    category_only_filtered_ranked=(),
                    title_store_exact_filtered_ranked=(),
                    tail=(),
                    enabled=False,
                    activated=False,
                    queries=empty_queries,
                    rules=HardConflictRules(),
                    conflict_count=0,
                    tail_conflict_count=0,
                    fusion_items=(),
                    route_latency_ns=tuple((route, 0) for route in ROUTE_NAMES),
                    hard_mask_latency_ns=0,
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
                hits_by_route: dict[str, tuple[tuple[int, str], ...]] = {}
                route_latency_ns: dict[str, int] = {}
                for route, query in queries.items():
                    hits, elapsed_ns = self._query_route(query)
                    hits_by_route[route] = hits
                    route_latency_ns[route] = elapsed_ns
                routes_by_name = {
                    route: tuple(identifier for _rowid, identifier in hits)
                    for route, hits in hits_by_route.items()
                }
                prefix_members = frozenset(prefix)
                ranked_novel_by_route = {
                    route: tuple(
                        (identifier, rank)
                        for rank, identifier in enumerate(
                            routes_by_name[route], start=1
                        )
                        if identifier not in prefix_members
                    )
                    for route in ROUTE_NAMES
                }
                novel_by_route = {
                    route: tuple(
                        identifier
                        for identifier, _rank in ranked_novel_by_route[route]
                    )
                    for route in ROUTE_NAMES
                }
                combined_hits = self._combined_hits(*hits_by_route.values())
                all_novel = tuple(
                    dict.fromkeys(
                        identifier
                        for route in ROUTE_NAMES
                        for identifier in novel_by_route[route]
                    )
                )
                mask_started = time.perf_counter_ns()
                views = self._views(combined_hits, all_novel)
                masks_by_route = {
                    route: self._apply_mask(
                        novel_by_route[route], views, combined_hits, rules
                    )
                    for route in ROUTE_NAMES
                }
                hard_mask_latency_ns = max(1, time.perf_counter_ns() - mask_started)
                rank_maps_by_route = {
                    route: dict(ranked_novel_by_route[route])
                    for route in ROUTE_NAMES
                }
                filtered_ranked_by_route = {
                    route: tuple(
                        (identifier, rank_maps_by_route[route][identifier])
                        for identifier in masks_by_route[route].identifiers
                    )
                    for route in ROUTE_NAMES
                }
                fusion = fuse_route_candidates(
                    prefix,
                    filtered_ranked_by_route[FULL_POSITIVE_ROUTE],
                    filtered_ranked_by_route[EXACT_ACTIVE_ROUTE],
                    filtered_ranked_by_route[CATEGORY_ONLY_ROUTE],
                    filtered_ranked_by_route[TITLE_STORE_EXACT_ROUTE],
                )
                tail_views = {
                    identifier: views[identifier]
                    for identifier in fusion.tail
                    if identifier in views
                }
                tail_mask_started = time.perf_counter_ns()
                tail_audit = self._apply_mask(
                    fusion.tail, tail_views, combined_hits, rules
                )
                hard_mask_latency_ns += max(
                    1, time.perf_counter_ns() - tail_mask_started
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
                or any(
                    tuple(identifier for identifier, _rank in filtered_ranked_by_route[route])
                    != masks_by_route[route].identifiers
                    for route in ROUTE_NAMES
                )
                or tail_audit.conflict_count
                or self.route_diagnostics()["legacy_route_executions"] != 0
            ):
                agent_to_close = self._permanent_close_locked()
                if agent_to_close is not None:
                    try:
                        agent_to_close.close()
                    except BaseException:
                        pass
                raise SparseMultiviewG0ValidationError("G0 stable union invariant failed")
            return ExpansionResult(
                candidates=fusion.candidates,
                prefix=prefix,
                full_positive_route=routes_by_name[FULL_POSITIVE_ROUTE],
                exact_active_route=routes_by_name[EXACT_ACTIVE_ROUTE],
                category_only_route=routes_by_name[CATEGORY_ONLY_ROUTE],
                title_store_exact_route=routes_by_name[TITLE_STORE_EXACT_ROUTE],
                full_positive_novel=novel_by_route[FULL_POSITIVE_ROUTE],
                exact_active_novel=novel_by_route[EXACT_ACTIVE_ROUTE],
                category_only_novel=novel_by_route[CATEGORY_ONLY_ROUTE],
                title_store_exact_novel=novel_by_route[TITLE_STORE_EXACT_ROUTE],
                full_positive_filtered=masks_by_route[FULL_POSITIVE_ROUTE].identifiers,
                exact_active_filtered=masks_by_route[EXACT_ACTIVE_ROUTE].identifiers,
                category_only_filtered=masks_by_route[
                    CATEGORY_ONLY_ROUTE
                ].identifiers,
                title_store_exact_filtered=masks_by_route[
                    TITLE_STORE_EXACT_ROUTE
                ].identifiers,
                full_positive_filtered_ranked=filtered_ranked_by_route[
                    FULL_POSITIVE_ROUTE
                ],
                exact_active_filtered_ranked=filtered_ranked_by_route[
                    EXACT_ACTIVE_ROUTE
                ],
                category_only_filtered_ranked=filtered_ranked_by_route[
                    CATEGORY_ONLY_ROUTE
                ],
                title_store_exact_filtered_ranked=filtered_ranked_by_route[
                    TITLE_STORE_EXACT_ROUTE
                ],
                tail=fusion.tail,
                enabled=True,
                activated=queries.activated,
                queries=queries,
                rules=rules,
                conflict_count=sum(
                    masks_by_route[route].conflict_count for route in ROUTE_NAMES
                ),
                tail_conflict_count=tail_audit.conflict_count,
                fusion_items=fusion.items,
                route_latency_ns=tuple(
                    (route, route_latency_ns[route]) for route in ROUTE_NAMES
                ),
                hard_mask_latency_ns=hard_mask_latency_ns,
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

    def __enter__(self) -> "SparseMultiviewG0Expander":
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def validate() -> None:
    """Validate frozen module-level constants without opening a catalog."""

    runtime = SparseMultiviewG0Expander(Path("."), enabled=False)
    try:
        runtime.validate()
    finally:
        runtime.close()


__all__ = [
    "CANDIDATE_CAP",
    "CATEGORY_ONLY_FIELDS",
    "CATEGORY_ONLY_ROUTE",
    "EXACT_ACTIVE_FIELDS",
    "EXACT_ACTIVE_ROUTE",
    "EXPECTED_ATTRIBUTE_REGISTRY_SHA256",
    "ExpansionResult",
    "FTS_COLUMNS",
    "FTS_QUERY_SQL",
    "FTS_ROUTE_CACHE_CAPACITY",
    "FTS_SCHEMA_SQL",
    "FTS_TOKENIZER",
    "FULL_POSITIVE_FIELDS",
    "FULL_POSITIVE_ROUTE",
    "FusionItem",
    "FusionResult",
    "HardConflictMaskResult",
    "HardConflictRules",
    "MASK_DECISION_CACHE_CAPACITY",
    "MECHANISM",
    "MultiRouteQueries",
    "POSITIVE_CONFLICT_SLOTS",
    "PREFIX_LIMIT",
    "PREFIX_MINIMUM",
    "PRODUCT_VIEW_CACHE_CAPACITY",
    "ROUTE_LIMIT",
    "ROUTE_NAMES",
    "RRF_K",
    "RouteQuery",
    "SCHEMA_VERSION",
    "SparseMultiviewG0ClosedError",
    "SparseMultiviewG0Error",
    "SparseMultiviewG0Expander",
    "SparseMultiviewG0ValidationError",
    "TERM_LIMIT",
    "TITLE_STORE_EXACT_FIELDS",
    "TITLE_STORE_EXACT_ROUTE",
    "apply_hard_conflict_mask",
    "attribute_registry_sha256",
    "build_route_queries",
    "compile_hard_conflict_rules",
    "fuse_route_candidates",
    "validate",
]
