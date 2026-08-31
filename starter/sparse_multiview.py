"""Frozen target-free sparse rewrite tail for the v2.19 recall probe.

The module is deliberately default-off and does not call ``Agent.respond``.  When
explicitly enabled it reuses the Agent's local in-memory SQLite FTS5 catalog,
executes one preregistered field-isolated route, masks only explicit catalog
conflicts in the novel route tail, and appends that tail after an untouched C200
prefix.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
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


SCHEMA_VERSION = "small-ranker-sparse-multiview.v1"
MECHANISM = "REGISTRY_CATEGORY_AND_CORE_ATTRIBUTE_REWRITE_TAIL"
EXPECTED_ATTRIBUTE_REGISTRY_SHA256 = (
    "1d85fc42f49fd9374238d98b8feaeab8d76269b0987740256fe60e666757d2ca"
)
ROUTE_LIMIT = 120
TERM_LIMIT = 24
PREFIX_MINIMUM = 100
PREFIX_LIMIT = 200
CANDIDATE_CAP = 400
STRUCTURAL_CANDIDATE_MAXIMUM = PREFIX_LIMIT + ROUTE_LIMIT
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
FTS_ROUTE_CACHE_CAPACITY = 256
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
_EXPECTED_FTS_QUERY_SQL = (
    "SELECT rowid,parent_asin FROM products WHERE products MATCH ? "
    "ORDER BY bm25(products,0.0,6.0,4.0,2.5,2.5,1.5,1.0),"
    "parent_asin ASC LIMIT 120"
)
_EXPECTED_FTS_TOKENIZER = "unicode61 remove_diacritics 2"
_EXPECTED_RELIABLE_SOURCES = frozenset(
    {"categories", "title", "features", "details", "store"}
)
_EXPECTED_ALLOWED_NEGATIVE_SLOTS = frozenset(
    {"audience", "material", "color", "closure", "style", "use_case"}
)
_CACHE_MISS = object()
_CANONICAL_VALUES = {
    "category": frozenset(normalize_value(value) for value in CATEGORIES.values()),
    **{
        slot: frozenset(normalize_value(value) for value in vocabulary.values())
        for slot, vocabulary in SLOT_VOCABULARIES.items()
    },
}


class SparseMultiviewError(RuntimeError):
    """Base error for the isolated rewrite-tail runtime."""


class SparseMultiviewValidationError(ValueError, SparseMultiviewError):
    """Raised when a frozen input or runtime invariant is violated."""


class SparseMultiviewClosedError(SparseMultiviewError):
    """Raised when an operation is attempted after close."""


@dataclass(frozen=True, slots=True)
class RewriteQuery:
    """Canonical values, reverse-registry surfaces, and the exact FTS expression."""

    activated: bool
    category_values: tuple[str, ...] = ()
    attribute_values: tuple[tuple[str, str], ...] = ()
    category_terms: tuple[str, ...] = ()
    attribute_terms: tuple[str, ...] = ()
    expression: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "activated": self.activated,
            "category_values": list(self.category_values),
            "attribute_values": [list(value) for value in self.attribute_values],
            "category_terms": list(self.category_terms),
            "attribute_terms": list(self.attribute_terms),
            "expression": self.expression,
        }


@dataclass(frozen=True, slots=True)
class HardConflictRules:
    """Executable current-version constraints for the conservative tail mask."""

    negative: tuple[tuple[str, str], ...] = ()
    positive: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "negative": [list(value) for value in self.negative],
            "positive": {
                slot: list(values) for slot, values in self.positive
            },
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
class HardConflictMaskResult:
    """Stable mask result; ``dropped`` contains explicit conflicts only."""

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
class ExpansionResult:
    """Complete stable-prefix expansion returned to an isolated worker."""

    candidates: tuple[str, ...]
    prefix: tuple[str, ...]
    route: tuple[str, ...]
    novel_route: tuple[str, ...]
    tail: tuple[str, ...]
    activated: bool
    query: RewriteQuery
    conflict_count: int
    tail_conflict_count: int = 0
    enabled: bool = False
    rules: HardConflictRules = HardConflictRules()

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mechanism": MECHANISM,
            "enabled": self.enabled,
            "activated": self.activated,
            "candidates": list(self.candidates),
            "prefix": list(self.prefix),
            "route": list(self.route),
            "novel_route": list(self.novel_route),
            "tail": list(self.tail),
            "conflict_count": self.conflict_count,
            "tail_conflict_count": self.tail_conflict_count,
            "query": self.query.as_dict(),
            "rules": self.rules.as_dict(),
        }


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
        raise SparseMultiviewValidationError(f"slot has no frozen registry: {slot}")
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
            raise SparseMultiviewValidationError(
                f"ledger record is missing required field: {name}"
            )
        return record[name]
    if not hasattr(record, name):
        raise SparseMultiviewValidationError(
            f"ledger record is missing required field: {name}"
        )
    return getattr(record, name)


def _records(records: Iterable[object], current_version: int) -> tuple[_Record, ...]:
    if isinstance(records, (str, bytes, Mapping)):
        raise SparseMultiviewValidationError("records must be an iterable of records")
    result: list[_Record] = []
    try:
        materialized = tuple(records)
    except TypeError as error:
        raise SparseMultiviewValidationError("records must be iterable") from error
    for order, record in enumerate(materialized, start=1):
        status = _field(record, "status")
        version = _field(record, "version")
        polarity = _field(record, "polarity")
        source_turn = _field(record, "source_turn")
        if not isinstance(status, str) or not status:
            raise SparseMultiviewValidationError("ledger status must be a non-empty string")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version < 1
        ):
            raise SparseMultiviewValidationError("ledger version must be positive")
        if polarity not in (-1, 1) or isinstance(polarity, bool):
            raise SparseMultiviewValidationError("ledger polarity must be -1 or +1")
        if (
            not isinstance(source_turn, int)
            or isinstance(source_turn, bool)
            or source_turn < 0
        ):
            raise SparseMultiviewValidationError("ledger source_turn must be non-negative")
        slot = normalize_value(_field(record, "slot")).replace(" ", "_")
        value = normalize_value(_field(record, "value"))
        hardness = normalize_value(_field(record, "hardness"))
        if not slot or not value or hardness not in {"hard", "soft"}:
            raise SparseMultiviewValidationError("ledger slot/value/hardness is invalid")
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


def _current_records(records: tuple[_Record, ...], version: int) -> tuple[_Record, ...]:
    return tuple(
        record
        for record in records
        if record.status == ACTIVE and record.version == version
    )


def _terms_input(values: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise SparseMultiviewValidationError(f"{name} must be an iterable of strings")
    try:
        result = tuple(values)
    except TypeError as error:
        raise SparseMultiviewValidationError(f"{name} must be iterable") from error
    if any(not isinstance(value, str) for value in result):
        raise SparseMultiviewValidationError(f"{name} must contain only strings")
    return result


def _validated_context(
    *,
    category_text: str,
    active_terms: Iterable[str],
    excluded_terms: Iterable[str],
    current_version: int,
    records: Iterable[object],
) -> tuple[
    ConversationConstraintView,
    tuple[_Record, ...],
]:
    if not isinstance(category_text, str):
        raise SparseMultiviewValidationError("category_text must be a string")
    if (
        not isinstance(current_version, int)
        or isinstance(current_version, bool)
        or current_version < 1
    ):
        raise SparseMultiviewValidationError("current_version must be a positive integer")
    active = _terms_input(active_terms, "active_terms")
    excluded = _terms_input(excluded_terms, "excluded_terms")
    materialized_records = _records(records, current_version)
    intent = build_conversation_constraint_view(
        category_text,
        active,
        excluded,
    )
    return intent, materialized_records


def _quoted(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _query_from_validated(
    intent: ConversationConstraintView,
    records: tuple[_Record, ...],
    current_version: int,
) -> RewriteQuery:
    category_values = tuple(
        sorted(
            {
                canonical
                for value in intent.category_terms
                if (canonical := _canonical_value("category", value)) is not None
            }
        )
    )
    attribute_values = tuple(
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
    attribute_surfaces: set[str] = set()
    for slot, canonical in attribute_values:
        attribute_surfaces.update(_registry_terms(slot, (canonical,)))
    attribute_terms = tuple(sorted(attribute_surfaces))[:TERM_LIMIT]

    activated = bool(category_values and attribute_values)
    expression = ""
    if activated:
        category_clause = (
            "{title categories} : ("
            + " OR ".join(_quoted(term) for term in category_terms)
            + ")"
        )
        attribute_clause = (
            "{title features details description} : ("
            + " OR ".join(_quoted(term) for term in attribute_terms)
            + ")"
        )
        expression = f"({category_clause}) AND ({attribute_clause})"
    return RewriteQuery(
        activated=activated,
        category_values=category_values,
        attribute_values=attribute_values,
        category_terms=category_terms,
        attribute_terms=attribute_terms,
        expression=expression,
    )


def build_rewrite_query(
    *,
    category_text: str,
    active_terms: Iterable[str],
    excluded_terms: Iterable[str],
    current_version: int,
    records: Iterable[object],
) -> RewriteQuery:
    """Build the sole frozen registry-only query without touching SQLite."""

    intent, materialized_records = _validated_context(
        category_text=category_text,
        active_terms=active_terms,
        excluded_terms=excluded_terms,
        current_version=current_version,
        records=records,
    )
    return _query_from_validated(intent, materialized_records, current_version)


def _rules_from_validated(
    intent: ConversationConstraintView,
    query: RewriteQuery,
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
        "category": set(query.category_values),
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
    query = _query_from_validated(intent, materialized_records, current_version)
    return _rules_from_validated(
        intent,
        query,
        materialized_records,
        current_version,
    )


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
    """Drop only reliable explicit violations while preserving route order."""

    if isinstance(identifiers, (str, bytes, Mapping)):
        raise SparseMultiviewValidationError("identifiers must be an ordered iterable")
    ordered = tuple(identifiers)
    if any(not isinstance(identifier, str) or not identifier for identifier in ordered):
        raise SparseMultiviewValidationError("identifiers must be non-empty strings")
    if len(ordered) != len(set(ordered)):
        raise SparseMultiviewValidationError("identifiers must be unique")
    if not isinstance(views, Mapping):
        raise SparseMultiviewValidationError("views must be a mapping")
    if not isinstance(rules, HardConflictRules):
        raise SparseMultiviewValidationError("rules must be HardConflictRules")

    executable_negatives, positive = _mask_inputs(rules)
    decisions: dict[str, tuple[bool, bool]] = {}
    for identifier in ordered:
        view = views.get(identifier)
        if view is None:
            view = ProductAttributeView(parent_asin=identifier)
        if not isinstance(view, ProductAttributeView):
            raise SparseMultiviewValidationError(
                "views must contain ProductAttributeView values"
            )
        decisions[identifier] = _mask_decision(
            view,
            executable_negatives,
            positive,
        )
    return _mask_result(ordered, decisions)


def _prefix(c200: Iterable[str]) -> tuple[str, ...]:
    if isinstance(c200, (str, bytes, Mapping)):
        raise SparseMultiviewValidationError("c200 must be an ordered iterable")
    try:
        prefix = tuple(c200)
    except TypeError as error:
        raise SparseMultiviewValidationError("c200 must be iterable") from error
    if not PREFIX_MINIMUM <= len(prefix) <= PREFIX_LIMIT:
        raise SparseMultiviewValidationError(
            "c200 length must be between 100 and 200"
        )
    if any(not isinstance(identifier, str) or not identifier for identifier in prefix):
        raise SparseMultiviewValidationError("c200 identifiers must be non-empty strings")
    if len(prefix) != len(set(prefix)):
        raise SparseMultiviewValidationError("c200 identifiers must be unique")
    return prefix


def _implementation_generation() -> tuple[object, ...]:
    return (
        attribute_registry_sha256,
        build_conversation_constraint_view,
        build_product_attribute_view,
        classify_candidate,
        normalize_value,
        product_slot,
        _canonical_value,
        _query_from_validated,
        _rules_from_validated,
        _reliable_values,
        _mask_inputs,
        _mask_decision,
        _mask_result,
    )


_EXPECTED_IMPLEMENTATION_GENERATION = _implementation_generation()


class SparseMultiviewExpander:
    """Default-off Agent-backed runtime for one frozen diagnostic route."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        enabled: bool = False,
        cache_enabled: bool = False,
    ) -> None:
        if not isinstance(enabled, bool):
            raise SparseMultiviewValidationError("enabled must be a bool")
        if not isinstance(cache_enabled, bool):
            raise SparseMultiviewValidationError("cache_enabled must be a bool")
        if not isinstance(catalog_path, (str, Path)):
            raise SparseMultiviewValidationError("catalog_path must be a path")
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
            raise SparseMultiviewValidationError("cache capacities must be positive integers")
        self._fts_route_cache: OrderedDict[
            str, tuple[tuple[int, str], ...]
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
                    self._closed = True
                if agent is not None:
                    agent.close()
                raise

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise SparseMultiviewClosedError("sparse multiview expander is closed")

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

    def _validate_locked(self) -> None:
        if (
            ROUTE_LIMIT != 120
            or TERM_LIMIT != 24
            or PREFIX_MINIMUM != 100
            or PREFIX_LIMIT != 200
            or CANDIDATE_CAP != 400
            or STRUCTURAL_CANDIDATE_MAXIMUM != 320
            or FTS_ROUTE_CACHE_CAPACITY != 256
            or PRODUCT_VIEW_CACHE_CAPACITY != 4096
            or MASK_DECISION_CACHE_CAPACITY != 16384
            or tuple(CORE_ATTRIBUTE_SLOTS) != ("material", "style", "use_case")
            or tuple(FTS_COLUMNS) != _EXPECTED_FTS_COLUMNS
            or FTS_TOKENIZER != _EXPECTED_FTS_TOKENIZER
            or FTS_QUERY_SQL != _EXPECTED_FTS_QUERY_SQL
            or _RELIABLE_SOURCES != _EXPECTED_RELIABLE_SOURCES
            or ALLOWED_NEGATIVE_SLOTS != _EXPECTED_ALLOWED_NEGATIVE_SLOTS
            or MIN_EVIDENCE_CONFIDENCE != 0.90
            or _implementation_generation() != self._implementation_generation
            or attribute_registry_sha256()
            != EXPECTED_ATTRIBUTE_REGISTRY_SHA256
        ):
            raise SparseMultiviewValidationError("frozen route constants drifted")
        current_capacities = {
            "fts_route": FTS_ROUTE_CACHE_CAPACITY,
            "product_view": PRODUCT_VIEW_CACHE_CAPACITY,
            "mask_decision": MASK_DECISION_CACHE_CAPACITY,
        }
        if current_capacities != self._cache_capacities:
            raise SparseMultiviewValidationError("cache capacities drifted")
        if not self.enabled:
            if self._agent is not None or self._database_generation is not None:
                raise SparseMultiviewValidationError(
                    "disabled runtime unexpectedly opened an Agent"
                )
            return
        agent = self._agent
        if agent is None or self._database_generation is None:
            raise SparseMultiviewValidationError("enabled runtime has no Agent")
        if (
            agent.p11_mode != "off"
            or agent.small_ranker_mode != "off"
            or agent.rerank_mode != "off"
            or agent.retrieval_mode != "control"
            or agent._p11_bridge is not None
            or agent._small_ranker is not None
        ):
            raise SparseMultiviewValidationError("Agent mode isolation drifted")
        database_rows = agent.connection.execute("PRAGMA database_list").fetchall()
        if not database_rows or any(str(row[2]) for row in database_rows):
            raise SparseMultiviewValidationError("Agent catalog is not in-memory")
        query_only_row = agent.connection.execute("PRAGMA query_only").fetchone()
        if not query_only_row or int(query_only_row[0]) != 1:
            raise SparseMultiviewValidationError("Agent catalog is not query-only")
        generation = self._read_database_generation(agent)
        if generation.columns != _EXPECTED_FTS_COLUMNS:
            raise SparseMultiviewValidationError("Agent FTS columns drifted")
        if "tokenize='unicode61 remove_diacritics 2'" not in generation.schema_sql:
            raise SparseMultiviewValidationError("Agent FTS tokenizer drifted")
        if (
            generation.row_count != generation.distinct_identifier_count
            or generation.empty_identifier_count
        ):
            raise SparseMultiviewValidationError(
                "Agent catalog identifiers are empty or duplicated"
            )
        if generation != self._database_generation:
            raise SparseMultiviewValidationError("Agent catalog generation drifted")

    def validate(self) -> None:
        """Fail closed if constants or the Agent FTS runtime drifted."""

        with self._lock:
            self._require_open()
            try:
                self._validate_locked()
            except BaseException:
                self._clear_caches_locked()
                self._closed = True
                self._database_generation = None
                agent, self._agent = self._agent, None
                if agent is not None:
                    try:
                        agent.close()
                    except BaseException:
                        # Preserve the validation failure as the authoritative
                        # fail-closed cause; the instance is already detached.
                        pass
                raise

    def _execute_query_route(
        self,
        query: RewriteQuery,
    ) -> tuple[tuple[int, str], ...]:
        agent = self._agent
        if agent is None:
            raise SparseMultiviewValidationError("enabled runtime has no Agent")
        rows = tuple(agent.connection.execute(FTS_QUERY_SQL, (query.expression,)))
        if len(rows) > ROUTE_LIMIT:
            raise SparseMultiviewValidationError("rewrite route exceeded LIMIT 120")
        hits = tuple((int(row[0]), str(row[1])) for row in rows)
        identifiers = tuple(identifier for _rowid, identifier in hits)
        if (
            any(rowid <= 0 or not identifier for rowid, identifier in hits)
            or len(identifiers) != len(set(identifiers))
        ):
            raise SparseMultiviewValidationError("rewrite route shape is invalid")
        return hits

    def _query_route(self, query: RewriteQuery) -> tuple[tuple[int, str], ...]:
        if not self.cache_enabled:
            return self._execute_query_route(query)
        found, cached = self._cache_lookup(
            "fts_route",
            self._fts_route_cache,
            query.expression,
        )
        if found:
            return cached
        hits = self._execute_query_route(query)
        self._cache_insert(
            "fts_route",
            self._fts_route_cache,
            query.expression,
            hits,
        )
        return hits

    def _load_product_views(
        self,
        hits: tuple[tuple[int, str], ...],
        identifiers: tuple[str, ...],
    ) -> dict[str, ProductAttributeView]:
        if not identifiers:
            return {}
        agent = self._agent
        if agent is None:
            raise SparseMultiviewValidationError("enabled runtime has no Agent")
        by_identifier = {identifier: rowid for rowid, identifier in hits}
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
                raise SparseMultiviewValidationError("rewrite catalog row identity drifted")
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
            raise SparseMultiviewValidationError("rewrite catalog view is incomplete")
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
                "product_view",
                self._product_view_cache,
                key,
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
                    "product_view",
                    self._product_view_cache,
                    key,
                    view,
                )
                views[identifier] = view
        return {identifier: views[identifier] for identifier in identifiers}

    def _apply_mask(
        self,
        identifiers: Iterable[str],
        views: Mapping[str, ProductAttributeView],
        hits: tuple[tuple[int, str], ...],
        rules: HardConflictRules,
    ) -> HardConflictMaskResult:
        if not self.cache_enabled:
            return apply_hard_conflict_mask(identifiers, views, rules)
        if isinstance(identifiers, (str, bytes, Mapping)):
            raise SparseMultiviewValidationError("identifiers must be an ordered iterable")
        ordered = tuple(identifiers)
        if any(not isinstance(identifier, str) or not identifier for identifier in ordered):
            raise SparseMultiviewValidationError("identifiers must be non-empty strings")
        if len(ordered) != len(set(ordered)):
            raise SparseMultiviewValidationError("identifiers must be unique")
        if not isinstance(views, Mapping):
            raise SparseMultiviewValidationError("views must be a mapping")
        if not isinstance(rules, HardConflictRules):
            raise SparseMultiviewValidationError("rules must be HardConflictRules")

        by_identifier = {identifier: rowid for rowid, identifier in hits}
        decisions: dict[str, tuple[bool, bool]] = {}
        staged: list[
            tuple[tuple[HardConflictRules, int, str], str, tuple[bool, bool]]
        ] = []
        executable_negatives, positive = _mask_inputs(rules)
        for identifier in ordered:
            rowid = by_identifier[identifier]
            key = (rules, rowid, identifier)
            found, cached = self._cache_lookup(
                "mask_decision",
                self._mask_decision_cache,
                key,
            )
            if found:
                decisions[identifier] = cached
                continue
            view = views.get(identifier)
            if view is None:
                view = ProductAttributeView(parent_asin=identifier)
            if not isinstance(view, ProductAttributeView):
                raise SparseMultiviewValidationError(
                    "views must contain ProductAttributeView values"
                )
            decision = _mask_decision(view, executable_negatives, positive)
            decisions[identifier] = decision
            staged.append((key, identifier, decision))
        result = _mask_result(ordered, decisions)
        for key, _identifier, decision in staged:
            self._cache_insert(
                "mask_decision",
                self._mask_decision_cache,
                key,
                decision,
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
        """Append the masked LIMIT-120 route after an untouched C200 prefix."""

        with self._lock:
            self._require_open()
            prefix = _prefix(c200)
            if not self.enabled:
                return ExpansionResult(
                    candidates=prefix,
                    prefix=prefix,
                    route=(),
                    novel_route=(),
                    tail=(),
                    activated=False,
                    query=RewriteQuery(activated=False),
                    conflict_count=0,
                    enabled=False,
                )

            intent, materialized_records = _validated_context(
                category_text=category_text,
                active_terms=active_terms,
                excluded_terms=excluded_terms,
                current_version=current_version,
                records=records,
            )
            query = _query_from_validated(
                intent,
                materialized_records,
                current_version,
            )
            rules = _rules_from_validated(
                intent,
                query,
                materialized_records,
                current_version,
            )
            if not query.activated:
                return ExpansionResult(
                    candidates=prefix,
                    prefix=prefix,
                    route=(),
                    novel_route=(),
                    tail=(),
                    activated=False,
                    query=query,
                    conflict_count=0,
                    enabled=True,
                    rules=rules,
                )

            hits = self._query_route(query)
            route = tuple(identifier for _rowid, identifier in hits)
            prefix_members = frozenset(prefix)
            novel_route = tuple(
                identifier for identifier in route if identifier not in prefix_members
            )
            views = self._views(hits, novel_route)
            mask = self._apply_mask(novel_route, views, hits, rules)
            tail_capacity = max(0, CANDIDATE_CAP - len(prefix))
            tail = mask.identifiers[:tail_capacity]
            tail_audit = self._apply_mask(tail, views, hits, rules)
            candidates = (*prefix, *tail)
            if (
                tuple(candidates[: len(prefix)]) != prefix
                or len(candidates) > CANDIDATE_CAP
                or len(candidates) > len(prefix) + ROUTE_LIMIT
                or len(candidates) != len(set(candidates))
            ):
                raise SparseMultiviewValidationError("stable append invariant failed")
            return ExpansionResult(
                candidates=tuple(candidates),
                prefix=prefix,
                route=route,
                novel_route=novel_route,
                tail=tuple(tail),
                activated=True,
                query=query,
                conflict_count=mask.conflict_count,
                tail_conflict_count=tail_audit.conflict_count,
                enabled=True,
                rules=rules,
            )

    def close(self) -> None:
        """Idempotently close the owned Agent and its in-memory SQLite database."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            agent, self._agent = self._agent, None
            self._clear_caches_locked()
            if agent is not None:
                agent.close()

    def __enter__(self) -> "SparseMultiviewExpander":
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def validate() -> None:
    """Validate frozen module-level constants without opening a catalog."""

    runtime = SparseMultiviewExpander(Path("."), enabled=False)
    try:
        runtime.validate()
    finally:
        runtime.close()


__all__ = [
    "CANDIDATE_CAP",
    "CORE_ATTRIBUTE_SLOTS",
    "ExpansionResult",
    "EXPECTED_ATTRIBUTE_REGISTRY_SHA256",
    "FTS_ROUTE_CACHE_CAPACITY",
    "FTS_QUERY_SQL",
    "FTS_TOKENIZER",
    "HardConflictMaskResult",
    "HardConflictRules",
    "MECHANISM",
    "MASK_DECISION_CACHE_CAPACITY",
    "PREFIX_MINIMUM",
    "PREFIX_LIMIT",
    "POSITIVE_CONFLICT_SLOTS",
    "PRODUCT_VIEW_CACHE_CAPACITY",
    "ROUTE_LIMIT",
    "RewriteQuery",
    "SCHEMA_VERSION",
    "STRUCTURAL_CANDIDATE_MAXIMUM",
    "SparseMultiviewClosedError",
    "SparseMultiviewError",
    "SparseMultiviewExpander",
    "SparseMultiviewValidationError",
    "TERM_LIMIT",
    "apply_hard_conflict_mask",
    "build_rewrite_query",
    "compile_hard_conflict_rules",
    "validate",
]
