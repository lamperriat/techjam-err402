"""Target-blind OOV character-gram bridge over the existing catalog FTS lexicon.

The mechanism is diagnostic and default-off.  It preserves the complete ordered
variable-C200 prefix and appends at most 192 hard-conflict-safe candidates from
one field-isolated sparse bridge route.  No evaluator target, dense score,
training artifact, synonym table, or earlier v2.22 route is a runtime input.
"""

from __future__ import annotations

from array import array
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from types import MappingProxyType
from typing import Any, TypeVar

from starter.agent import Agent
from starter.attributes import (
    CATEGORIES,
    NOISE_VALUES,
    SLOT_VOCABULARIES,
    ProductAttributeView,
    build_conversation_constraint_view,
    build_product_attribute_view,
    normalize_value,
    product_slot,
)
from starter.p8_negative import (
    ALLOWED_NEGATIVE_SLOTS,
    EXPLICIT_VIOLATION,
    ExecutableNegative,
    classify_candidate,
)
from starter.slot_ledger import ACTIVE


SCHEMA_VERSION = "small-ranker-v2.23-oov-chargram-lexicon-bridge-g0.v1"
UNKNOWN_CATEGORY_TOKEN = "unknown_category_token"
EXACT_ACTIVE_TOKEN = "exact_active_token"
SOURCE_KIND_ORDER = (UNKNOWN_CATEGORY_TOKEN, EXACT_ACTIVE_TOKEN)
SOURCE_FIELDS = {
    UNKNOWN_CATEGORY_TOKEN: ("title", "categories"),
    EXACT_ACTIVE_TOKEN: ("title", "features", "details", "store", "description"),
}
FTS_COLUMNS = (
    "parent_asin",
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)
FTS_SCHEMA_SQL = (
    "CREATE VIRTUAL TABLE products USING fts5("
    "parent_asin UNINDEXED, title, categories, features, details, store, "
    "description, tokenize='unicode61 remove_diacritics 2')"
)
FTS_QUERY_SQL = (
    "SELECT rowid,parent_asin FROM products WHERE products MATCH ? "
    "ORDER BY bm25(products,0.0,6.0,4.0,2.5,2.5,1.5,1.0),"
    "parent_asin ASC LIMIT 32"
)
PREFIX_MINIMUM = 100
PREFIX_LIMIT = 200
TAIL_LIMIT = 192
CANDIDATE_CAP = 400
SOURCE_LIMIT = 6
BRIDGE_LIMIT = 4
ROUTE_LIMIT = 32
POSTING_UNION_LIMIT = 8192
MIN_DICE = Fraction(2, 3)
BRIDGE_CACHE_CAPACITY = 4096
FTS_ROUTE_CACHE_CAPACITY = 512
PRODUCT_VIEW_CACHE_CAPACITY = 4096
MASK_DECISION_CACHE_CAPACITY = 16384
MIN_EVIDENCE_CONFIDENCE = 0.90
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
RELIABLE_SOURCES = frozenset(
    {"categories", "title", "features", "details", "store"}
)
FROZEN_STOPWORDS = frozenset(
    {
        "all",
        "and",
        "available",
        "budget",
        "color",
        "colors",
        "dollar",
        "dollars",
        "fabric",
        "heather",
        "heathers",
        "imported",
        "looking",
        "made",
        "need",
        "or",
        "other",
        "over",
        "please",
        "price",
        "something",
        "under",
        "want",
        "with",
    }
)

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+", re.ASCII)
_ELIGIBLE_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{3,23}\Z", re.ASCII)
_SINGLE_TOKEN_RE = re.compile(r"[a-z0-9]+\Z", re.ASCII)
_EXPECTED_ALLOWED_NEGATIVE_SLOTS = frozenset(
    {"audience", "material", "color", "closure", "style", "use_case"}
)
_CANONICAL_VALUES = {
    "category": frozenset(normalize_value(value) for value in CATEGORIES.values()),
    **{
        slot: frozenset(normalize_value(value) for value in vocabulary.values())
        for slot, vocabulary in SLOT_VOCABULARIES.items()
    },
}
_REGISTRY_BY_NORMALIZED_SURFACE = {
    "category": {
        normalize_value(surface): normalize_value(canonical)
        for surface, canonical in CATEGORIES.items()
    },
    **{
        slot: {
            normalize_value(surface): normalize_value(canonical)
            for surface, canonical in vocabulary.items()
        }
        for slot, vocabulary in SLOT_VOCABULARIES.items()
    },
}


class OovChargramBridgeG0Error(RuntimeError):
    """Base error for the isolated bridge runtime."""


class OovChargramBridgeG0ValidationError(
    ValueError, OovChargramBridgeG0Error
):
    """A deterministic contract failure."""


class OovChargramBridgeG0ResourceError(OovChargramBridgeG0Error):
    """A frozen resource boundary was exceeded."""


class OovChargramBridgeG0ClosedError(OovChargramBridgeG0Error):
    """The permanently closed runtime was used."""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    kind: str
    token: str
    fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BridgeMatch:
    token: str
    edit_distance: int
    dice: Fraction
    global_doc_frequency: int

    def as_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "edit_distance": self.edit_distance,
            "dice_numerator": self.dice.numerator,
            "dice_denominator": self.dice.denominator,
            "global_doc_frequency": self.global_doc_frequency,
        }


@dataclass(frozen=True, slots=True)
class BridgeSource:
    source: SourceRecord
    matches: tuple[BridgeMatch, ...]

    @property
    def best(self) -> BridgeMatch:
        if not self.matches:
            raise OovChargramBridgeG0ValidationError("bridge source has no match")
        return self.matches[0]


@dataclass(frozen=True, slots=True)
class SourceRoute:
    source: BridgeSource
    expression: str
    identifiers: tuple[str, ...]
    novel_identifiers: tuple[str, ...]
    filtered_identifiers: tuple[str, ...]
    latency_ns: int


@dataclass(frozen=True, slots=True)
class HardConflictRules:
    negative: tuple[tuple[str, str], ...] = ()
    positive: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class HardConflictMaskResult:
    identifiers: tuple[str, ...]
    dropped: tuple[str, ...]
    negative_violation_count: int
    positive_conflict_count: int

    @property
    def conflict_count(self) -> int:
        return len(self.dropped)


@dataclass(frozen=True, slots=True)
class ExpansionResult:
    candidates: tuple[str, ...]
    prefix: tuple[str, ...]
    tail: tuple[str, ...]
    enabled: bool
    activated: bool
    source_records: tuple[SourceRecord, ...]
    bridge_sources: tuple[BridgeSource, ...]
    routes: tuple[SourceRoute, ...]
    novel_identifiers: tuple[str, ...]
    filtered_identifiers: tuple[str, ...]
    rules: HardConflictRules
    conflict_count: int
    negative_violation_count: int
    positive_conflict_count: int
    tail_conflict_count: int
    bridge_lookup_latency_ns: int
    fts_route_latency_ns: int
    hard_mask_latency_ns: int
    query_only_readback_one: bool
    controlled_write_rejected: bool
    write_guard_unchanged: bool
    legacy_route_executions: int = 0
    fallback: bool = False
    fallback_code: str = "NONE"


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


@dataclass(slots=True)
class _CacheCounters:
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0


@dataclass(frozen=True, slots=True)
class _DatabaseGeneration:
    schema_version: int
    total_changes: int
    row_count: int
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CatalogEvidence:
    raw: bytes
    offsets: Mapping[str, tuple[int, int]]


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _ASCII_TOKEN_RE.finditer(normalize_value(value)))


def _registry_consumed_tokens() -> frozenset[str]:
    values: set[str] = set()
    for vocabulary in (CATEGORIES, *SLOT_VOCABULARIES.values()):
        for surface, canonical in vocabulary.items():
            values.update(_tokens(surface))
            values.update(_tokens(canonical))
    return frozenset(values)


_REGISTRY_CONSUMED_TOKENS = _registry_consumed_tokens()


def _terms_input(values: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise OovChargramBridgeG0ValidationError(f"{name} must be an iterable")
    try:
        result = tuple(values)
    except TypeError as error:
        raise OovChargramBridgeG0ValidationError(f"{name} must be iterable") from error
    if any(not isinstance(value, str) for value in result):
        raise OovChargramBridgeG0ValidationError(f"{name} must contain strings")
    return result


def canonical_source_records(
    *,
    category_text: str,
    active_terms: Iterable[str],
    excluded_terms: Iterable[str],
) -> tuple[SourceRecord, ...]:
    """Extract only eligible positive source-kind/token pairs."""

    if not isinstance(category_text, str):
        raise OovChargramBridgeG0ValidationError("category_text must be a string")
    active = _terms_input(active_terms, "active_terms")
    excluded = _terms_input(excluded_terms, "excluded_terms")
    intent = build_conversation_constraint_view(category_text, active, excluded)
    pairs: set[tuple[str, str]] = set()
    for phrase in intent.category_terms:
        for token in _tokens(phrase):
            if (
                _ELIGIBLE_TOKEN_RE.fullmatch(token)
                and token not in _REGISTRY_CONSUMED_TOKENS
                and token not in FROZEN_STOPWORDS
                and token not in NOISE_VALUES
            ):
                pairs.add((UNKNOWN_CATEGORY_TOKEN, token))
    for phrase in intent.exact_terms:
        for token in _tokens(phrase):
            if (
                _ELIGIBLE_TOKEN_RE.fullmatch(token)
                and token not in _REGISTRY_CONSUMED_TOKENS
                and token not in FROZEN_STOPWORDS
                and token not in NOISE_VALUES
            ):
                pairs.add((EXACT_ACTIVE_TOKEN, token))
    kind_rank = {kind: index for index, kind in enumerate(SOURCE_KIND_ORDER)}
    ordered = sorted(pairs, key=lambda item: (item[1], kind_rank[item[0]]))
    return tuple(SourceRecord(kind, token, SOURCE_FIELDS[kind]) for kind, token in ordered)


def boundary_trigrams(token: str) -> tuple[str, ...]:
    if not isinstance(token, str) or _ELIGIBLE_TOKEN_RE.fullmatch(token) is None:
        raise OovChargramBridgeG0ValidationError("token is not bridge-eligible")
    bounded = "^" + token + "$"
    return tuple(sorted({bounded[index : index + 3] for index in range(len(bounded) - 2)}))


def levenshtein_distance(left: str, right: str, maximum: int | None = None) -> int:
    if not isinstance(left, str) or not isinstance(right, str):
        raise OovChargramBridgeG0ValidationError("distance inputs must be strings")
    if maximum is not None and (
        not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0
    ):
        raise OovChargramBridgeG0ValidationError("maximum must be non-negative")
    if left == right:
        return 0
    if maximum is not None and abs(len(left) - len(right)) > maximum:
        return maximum + 1
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row_index, right_char in enumerate(right, start=1):
        current = [row_index]
        row_minimum = row_index
        for column_index, left_char in enumerate(left, start=1):
            value = min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + int(left_char != right_char),
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        previous = current
        if maximum is not None and row_minimum > maximum:
            return maximum + 1
    distance = previous[-1]
    return maximum + 1 if maximum is not None and distance > maximum else distance


def _field(record: object, name: str) -> object:
    if isinstance(record, Mapping):
        if name not in record:
            raise OovChargramBridgeG0ValidationError(f"record missing {name}")
        return record[name]
    if not hasattr(record, name):
        raise OovChargramBridgeG0ValidationError(f"record missing {name}")
    return getattr(record, name)


def _records(records: Iterable[object]) -> tuple[_Record, ...]:
    if isinstance(records, (str, bytes, Mapping)):
        raise OovChargramBridgeG0ValidationError("records must be an iterable")
    try:
        source = tuple(records)
    except TypeError as error:
        raise OovChargramBridgeG0ValidationError("records must be iterable") from error
    result: list[_Record] = []
    for order, record in enumerate(source, start=1):
        status = _field(record, "status")
        version = _field(record, "version")
        polarity = _field(record, "polarity")
        source_turn = _field(record, "source_turn")
        slot = normalize_value(_field(record, "slot")).replace(" ", "_")
        value = normalize_value(_field(record, "value"))
        hardness = normalize_value(_field(record, "hardness"))
        if not isinstance(status, str) or not status:
            raise OovChargramBridgeG0ValidationError("record status is invalid")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise OovChargramBridgeG0ValidationError("record version is invalid")
        if polarity not in (-1, 1) or isinstance(polarity, bool):
            raise OovChargramBridgeG0ValidationError("record polarity is invalid")
        if (
            not isinstance(source_turn, int)
            or isinstance(source_turn, bool)
            or source_turn < 0
        ):
            raise OovChargramBridgeG0ValidationError("record source_turn is invalid")
        if not slot or not value or hardness not in {"hard", "soft"}:
            raise OovChargramBridgeG0ValidationError("record fields are invalid")
        result.append(
            _Record(
                slot,
                value,
                int(polarity),
                hardness,
                int(source_turn),
                int(version),
                status,
                order,
            )
        )
    return tuple(result)


def _canonical_value(slot: str, value: object) -> str | None:
    normalized = normalize_value(value)
    registry = _REGISTRY_BY_NORMALIZED_SURFACE.get(slot)
    if registry is None:
        return None
    canonical = registry.get(normalized, normalized)
    return canonical if canonical in _CANONICAL_VALUES[slot] else None


def compile_hard_conflict_rules(
    *,
    category_text: str,
    active_terms: Iterable[str],
    excluded_terms: Iterable[str],
    current_version: int,
    records: Iterable[object],
) -> HardConflictRules:
    if not isinstance(category_text, str):
        raise OovChargramBridgeG0ValidationError("category_text must be a string")
    if (
        not isinstance(current_version, int)
        or isinstance(current_version, bool)
        or current_version < 1
    ):
        raise OovChargramBridgeG0ValidationError("current_version is invalid")
    active = _terms_input(active_terms, "active_terms")
    excluded = _terms_input(excluded_terms, "excluded_terms")
    materialized = _records(records)
    intent = build_conversation_constraint_view(category_text, active, excluded)
    visible_negatives = {
        (constraint.slot, constraint.value)
        for constraint in intent.negative
        if constraint.polarity == -1
    }
    negative: set[tuple[str, str]] = set()
    positive: dict[str, set[str]] = {"category": set()}
    for value in intent.category_terms:
        canonical = _canonical_value("category", value)
        if canonical is not None:
            positive["category"].add(canonical)
    for record in materialized:
        if (
            record.status != ACTIVE
            or record.version != current_version
            or record.hardness != "hard"
        ):
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


def _reliable_values(view: ProductAttributeView, slot: str) -> frozenset[str]:
    result: set[str] = set()
    for item in product_slot(view, slot):
        if not (
            item.confidence >= MIN_EVIDENCE_CONFIDENCE
            and (
                item.source in RELIABLE_SOURCES
                or item.source.startswith("details.")
            )
        ):
            continue
        canonical = _canonical_value(slot, item.value)
        if canonical is not None:
            result.add(canonical)
    return frozenset(result)


def _mask_decision(
    view: ProductAttributeView,
    rules: HardConflictRules,
) -> tuple[bool, bool]:
    executable = tuple(
        ExecutableNegative(
            slot=slot,
            value=value,
            record_id=index,
            source_turn=0,
            version=0,
        )
        for index, (slot, value) in enumerate(rules.negative, start=1)
    )
    negative_violation = bool(
        executable and classify_candidate(view, executable).state == EXPLICIT_VIOLATION
    )
    positive_conflict = any(
        bool(evidence := _reliable_values(view, slot))
        and evidence.isdisjoint(requested)
        for slot, requested in rules.positive
    )
    return negative_violation, positive_conflict


def apply_hard_conflict_mask(
    identifiers: Iterable[str],
    views: Mapping[str, ProductAttributeView],
    rules: HardConflictRules,
) -> HardConflictMaskResult:
    if isinstance(identifiers, (str, bytes, Mapping)):
        raise OovChargramBridgeG0ValidationError("identifiers must be ordered")
    ordered = tuple(identifiers)
    if (
        any(not isinstance(identifier, str) or not identifier for identifier in ordered)
        or len(ordered) != len(set(ordered))
    ):
        raise OovChargramBridgeG0ValidationError("identifiers are invalid")
    if not isinstance(views, Mapping) or not isinstance(rules, HardConflictRules):
        raise OovChargramBridgeG0ValidationError("mask inputs are invalid")
    kept: list[str] = []
    dropped: list[str] = []
    negative_count = 0
    positive_count = 0
    for identifier in ordered:
        view = views.get(identifier)
        if view is None:
            view = ProductAttributeView(parent_asin=identifier)
        if not isinstance(view, ProductAttributeView):
            raise OovChargramBridgeG0ValidationError("view is invalid")
        negative, positive = _mask_decision(view, rules)
        if negative or positive:
            dropped.append(identifier)
            negative_count += int(negative)
            positive_count += int(positive)
        else:
            kept.append(identifier)
    return HardConflictMaskResult(
        tuple(kept), tuple(dropped), negative_count, positive_count
    )


def _prefix(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise OovChargramBridgeG0ValidationError("C200 must be ordered")
    prefix = tuple(values)
    if (
        not PREFIX_MINIMUM <= len(prefix) <= PREFIX_LIMIT
        or any(not isinstance(value, str) or not value for value in prefix)
        or len(prefix) != len(set(prefix))
    ):
        raise OovChargramBridgeG0ValidationError("C200 prefix is invalid")
    return prefix


def stable_append_candidates(
    prefix: Iterable[str], ranked_identifiers: Iterable[str]
) -> tuple[str, ...]:
    frozen = _prefix(prefix)
    if isinstance(ranked_identifiers, (str, bytes, Mapping)):
        raise OovChargramBridgeG0ValidationError("ranked identifiers are invalid")
    tail_cap = min(TAIL_LIMIT, CANDIDATE_CAP - len(frozen))
    seen = set(frozen)
    tail: list[str] = []
    for identifier in ranked_identifiers:
        if not isinstance(identifier, str) or not identifier:
            raise OovChargramBridgeG0ValidationError("ranked identifier is invalid")
        if identifier in seen:
            continue
        seen.add(identifier)
        tail.append(identifier)
        if len(tail) == tail_cap:
            break
    return (*frozen, *tail)


_K = TypeVar("_K")
_V = TypeVar("_V")


def _cache_get(
    cache: OrderedDict[_K, _V], counters: _CacheCounters, key: _K
) -> tuple[bool, _V | None]:
    try:
        value = cache.pop(key)
    except KeyError:
        counters.misses += 1
        return False, None
    cache[key] = value
    counters.hits += 1
    return True, value


def _cache_put(
    cache: OrderedDict[_K, _V],
    counters: _CacheCounters,
    key: _K,
    value: _V,
    capacity: int,
) -> None:
    cache.pop(key, None)
    cache[key] = value
    counters.inserts += 1
    while len(cache) > capacity:
        cache.popitem(last=False)
        counters.evictions += 1


def _build_catalog_evidence(path: Path) -> _CatalogEvidence:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise OovChargramBridgeG0ValidationError("catalog read failed") from error
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    while start < len(raw):
        newline = raw.find(b"\n", start)
        end = len(raw) if newline < 0 else newline
        line = raw[start:end]
        if line.strip():
            try:
                product = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise OovChargramBridgeG0ValidationError("catalog row invalid") from error
            identifier = product.get("parent_asin") if isinstance(product, Mapping) else None
            if not isinstance(identifier, str) or not identifier or identifier in offsets:
                raise OovChargramBridgeG0ValidationError("catalog identifier invalid")
            offsets[identifier] = (start, end)
        if newline < 0:
            break
        start = newline + 1
    if not offsets:
        raise OovChargramBridgeG0ValidationError("catalog is empty")
    return _CatalogEvidence(raw, MappingProxyType(offsets))


class OovChargramBridgeG0Expander:
    """Owned, default-off sparse bridge runtime."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        enabled: bool = False,
        cache_enabled: bool = False,
    ) -> None:
        if not isinstance(enabled, bool) or not isinstance(cache_enabled, bool):
            raise OovChargramBridgeG0ValidationError("flags must be booleans")
        if not isinstance(catalog_path, (str, Path)):
            raise OovChargramBridgeG0ValidationError("catalog path is invalid")
        self.catalog_path = Path(catalog_path)
        self.enabled = enabled
        self.cache_enabled = cache_enabled
        self._lock = threading.RLock()
        self._closed = False
        self._agent: Agent | None = None
        self._generation: _DatabaseGeneration | None = None
        self._evidence: _CatalogEvidence | None = None
        self._tokens: tuple[str, ...] = ()
        self._token_gram_counts: tuple[int, ...] = ()
        self._global_df: tuple[int, ...] = ()
        self._field_tokens: Mapping[str, frozenset[str]] = MappingProxyType({})
        self._field_df: Mapping[str, Mapping[str, int]] = MappingProxyType({})
        self._gram_postings: Mapping[str, memoryview] = MappingProxyType({})
        self._query_only_readback_one = False
        self._controlled_write_rejected = False
        self._write_guard_unchanged = False
        self._bridge_cache: OrderedDict[
            tuple[object, ...], tuple[BridgeMatch, ...]
        ] = OrderedDict()
        self._fts_cache: OrderedDict[
            tuple[object, ...], tuple[str, ...]
        ] = OrderedDict()
        self._product_cache: OrderedDict[
            tuple[object, ...], ProductAttributeView
        ] = OrderedDict()
        self._mask_cache: OrderedDict[
            tuple[object, ...], tuple[bool, bool]
        ] = OrderedDict()
        self._counters = {
            "oov_bridge": _CacheCounters(),
            "fts_route": _CacheCounters(),
            "product_view": _CacheCounters(),
            "mask_decision": _CacheCounters(),
        }
        self._route_executions = 0
        self._bridge_lookups = 0
        self._cache_clears = 0
        if self.enabled:
            try:
                self._initialize()
                self._validate_full_locked()
            except BaseException:
                agent = self._permanent_close_locked()
                if agent is not None:
                    try:
                        agent.close()
                    except BaseException:
                        pass
                raise

    @property
    def closed(self) -> bool:
        return self._closed

    @staticmethod
    def _database_generation(agent: Agent) -> _DatabaseGeneration:
        columns = tuple(
            str(row[1]) for row in agent.connection.execute("PRAGMA table_info(products)")
        )
        schema_row = agent.connection.execute("PRAGMA schema_version").fetchone()
        count_row = agent.connection.execute("SELECT COUNT(*) FROM products").fetchone()
        return _DatabaseGeneration(
            schema_version=int(schema_row[0] if schema_row else -1),
            total_changes=int(agent.connection.total_changes),
            row_count=int(count_row[0] if count_row else -1),
            columns=columns,
        )

    @staticmethod
    def _validate_vocab_columns(
        connection: sqlite3.Connection, table: str, expected: tuple[str, ...]
    ) -> None:
        columns = tuple(
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if columns != expected:
            raise OovChargramBridgeG0ValidationError("FTS vocabulary schema drifted")

    def _initialize(self) -> None:
        agent = Agent(
            self.catalog_path,
            question_policy="fast",
            rerank_mode="off",
            retrieval_mode="control",
            p11_mode="off",
            small_ranker_mode="off",
        )
        self._agent = agent
        connection = agent.connection
        connection.execute(
            "CREATE VIRTUAL TABLE products_vocab_col "
            "USING fts5vocab(products,'col')"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE products_vocab_row "
            "USING fts5vocab(products,'row')"
        )
        self._validate_vocab_columns(
            connection, "products_vocab_col", ("term", "col", "doc", "cnt")
        )
        self._validate_vocab_columns(
            connection, "products_vocab_row", ("term", "doc", "cnt")
        )

        field_terms: dict[str, set[str]] = {
            field: set() for fields in SOURCE_FIELDS.values() for field in fields
        }
        field_df: dict[str, dict[str, int]] = {
            field: {} for field in field_terms
        }
        previous_pair: tuple[str, str] | None = None
        for raw_term, raw_column, raw_doc, raw_count in connection.execute(
            "SELECT term,col,doc,cnt FROM products_vocab_col ORDER BY term,col"
        ):
            term = str(raw_term)
            column = str(raw_column)
            pair = (term, column)
            if pair == previous_pair:
                raise OovChargramBridgeG0ValidationError("duplicate column vocabulary")
            previous_pair = pair
            if (
                not isinstance(raw_doc, int)
                or isinstance(raw_doc, bool)
                or raw_doc < 0
                or not isinstance(raw_count, int)
                or isinstance(raw_count, bool)
                or raw_count < 0
            ):
                raise OovChargramBridgeG0ValidationError("invalid column vocabulary")
            if column in field_terms:
                field_terms[column].add(term)
                field_df[column][term] = int(raw_doc)

        row_df: dict[str, int] = {}
        previous_term: str | None = None
        for raw_term, raw_doc, raw_count in connection.execute(
            "SELECT term,doc,cnt FROM products_vocab_row ORDER BY term"
        ):
            term = str(raw_term)
            if term == previous_term or term in row_df:
                raise OovChargramBridgeG0ValidationError("duplicate row vocabulary")
            previous_term = term
            if (
                not isinstance(raw_doc, int)
                or isinstance(raw_doc, bool)
                or raw_doc < 0
                or not isinstance(raw_count, int)
                or isinstance(raw_count, bool)
                or raw_count < 0
            ):
                raise OovChargramBridgeG0ValidationError("invalid row vocabulary")
            row_df[term] = int(raw_doc)

        column_terms = set().union(*field_terms.values())
        if any(token not in row_df for token in column_terms):
            raise OovChargramBridgeG0ValidationError("row vocabulary join failed")
        tokens = tuple(
            sorted(
                token
                for token in column_terms
                if _ELIGIBLE_TOKEN_RE.fullmatch(token) is not None
            )
        )
        postings: dict[str, array] = defaultdict(lambda: array("I"))
        gram_counts: list[int] = []
        for token_id, token in enumerate(tokens):
            grams = boundary_trigrams(token)
            gram_counts.append(len(grams))
            for gram in grams:
                postings[gram].append(token_id)
        self._tokens = tokens
        self._token_gram_counts = tuple(gram_counts)
        self._global_df = tuple(row_df[token] for token in tokens)
        self._field_tokens = MappingProxyType(
            {field: frozenset(values) for field, values in field_terms.items()}
        )
        self._field_df = MappingProxyType(
            {
                field: MappingProxyType(dict(values))
                for field, values in field_df.items()
            }
        )
        self._gram_postings = MappingProxyType(
            {
                gram: memoryview(values.tobytes()).cast("I")
                for gram, values in postings.items()
            }
        )
        self._evidence = _build_catalog_evidence(self.catalog_path)

        schema_before = int(connection.execute("PRAGMA schema_version").fetchone()[0])
        changes_before = int(connection.total_changes)
        connection.execute("PRAGMA query_only=ON")
        readback = connection.execute("PRAGMA query_only").fetchone()
        self._query_only_readback_one = bool(readback and int(readback[0]) == 1)
        rejected = False
        try:
            connection.execute("CREATE TABLE v223_query_only_negative(x INTEGER)")
        except sqlite3.OperationalError as error:
            rejected = getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_READONLY
        self._controlled_write_rejected = rejected
        readback_after = connection.execute("PRAGMA query_only").fetchone()
        self._write_guard_unchanged = bool(
            readback_after
            and int(readback_after[0]) == 1
            and int(connection.execute("PRAGMA schema_version").fetchone()[0])
            == schema_before
            and int(connection.total_changes) == changes_before
        )
        if not (
            self._query_only_readback_one
            and self._controlled_write_rejected
            and self._write_guard_unchanged
        ):
            raise OovChargramBridgeG0ValidationError("query-only contract failed")
        self._generation = self._database_generation(agent)
        if self._generation.row_count != len(self._evidence.offsets):
            raise OovChargramBridgeG0ValidationError("catalog generation mismatch")

    def _require_open(self) -> None:
        if self._closed:
            raise OovChargramBridgeG0ClosedError("runtime is closed")

    def _clear_caches_locked(self) -> None:
        for cache in (
            self._bridge_cache,
            self._fts_cache,
            self._product_cache,
            self._mask_cache,
        ):
            cache.clear()
        self._cache_clears += 1

    def _permanent_close_locked(self) -> Agent | None:
        if self._closed:
            return None
        self._closed = True
        self._clear_caches_locked()
        self._tokens = ()
        self._token_gram_counts = ()
        self._global_df = ()
        self._field_tokens = MappingProxyType({})
        self._field_df = MappingProxyType({})
        self._gram_postings = MappingProxyType({})
        self._generation = None
        self._evidence = None
        agent, self._agent = self._agent, None
        return agent

    def _validate_fast_locked(self) -> None:
        if self._closed:
            raise OovChargramBridgeG0ClosedError("runtime is closed")
        if not self.enabled:
            if self._agent is not None or self._generation is not None:
                raise OovChargramBridgeG0ValidationError("disabled runtime opened state")
            return
        if (
            self._agent is None
            or self._generation is None
            or self._evidence is None
            or not self._query_only_readback_one
            or not self._controlled_write_rejected
            or not self._write_guard_unchanged
        ):
            raise OovChargramBridgeG0ValidationError("runtime initialization drifted")
        readback = self._agent.connection.execute("PRAGMA query_only").fetchone()
        if not readback or int(readback[0]) != 1:
            raise OovChargramBridgeG0ValidationError("query-only drifted")

    def _validate_full_locked(self) -> None:
        self._validate_fast_locked()
        if not self.enabled:
            return
        agent = self._agent
        generation = self._generation
        evidence = self._evidence
        if agent is None or generation is None or evidence is None:
            raise OovChargramBridgeG0ValidationError("runtime is incomplete")
        if (
            agent.p11_mode != "off"
            or agent.small_ranker_mode != "off"
            or agent.rerank_mode != "off"
            or agent.retrieval_mode != "control"
            or agent._p11_bridge is not None
            or agent._small_ranker is not None
        ):
            raise OovChargramBridgeG0ValidationError("Agent isolation drifted")
        databases = agent.connection.execute("PRAGMA database_list").fetchall()
        if not databases or any(str(row[2]) for row in databases):
            raise OovChargramBridgeG0ValidationError("catalog database is not in-memory")
        current = self._database_generation(agent)
        if current != generation or current.columns != FTS_COLUMNS:
            raise OovChargramBridgeG0ValidationError("database generation drifted")
        schema = agent.connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='products'"
        ).fetchone()
        normalized_schema = re.sub(r"\s+", " ", str(schema[0] if schema else "")).strip()
        if normalized_schema != re.sub(r"\s+", " ", FTS_SCHEMA_SQL).strip():
            raise OovChargramBridgeG0ValidationError("FTS schema drifted")
        if current.row_count != len(evidence.offsets):
            raise OovChargramBridgeG0ValidationError("catalog evidence drifted")
        if (
            tuple(sorted(self._field_df)) != tuple(sorted(self._field_tokens))
            or any(
                frozenset(self._field_df[field]) != self._field_tokens[field]
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for value in self._field_df[field].values()
                )
                for field in self._field_tokens
            )
        ):
            raise OovChargramBridgeG0ValidationError("column vocabulary state drifted")

    def validate(self) -> None:
        with self._lock:
            self._require_open()
            try:
                self._validate_full_locked()
            except BaseException:
                agent = self._permanent_close_locked()
                if agent is not None:
                    try:
                        agent.close()
                    except BaseException:
                        pass
                raise

    def _cache_key_prefix(self) -> tuple[int, int, int]:
        generation = self._generation
        if generation is None:
            raise OovChargramBridgeG0ValidationError("database generation missing")
        return (
            generation.schema_version,
            generation.total_changes,
            generation.row_count,
        )

    def _map_source(self, source: SourceRecord) -> tuple[BridgeMatch, ...]:
        if any(source.token in self._field_tokens[field] for field in source.fields):
            return ()
        key = (*self._cache_key_prefix(), source.kind, source.token, source.fields)
        if self.cache_enabled:
            found, cached = _cache_get(
                self._bridge_cache, self._counters["oov_bridge"], key
            )
            if found:
                return tuple(cached or ())
        self._bridge_lookups += 1
        source_grams = boundary_trigrams(source.token)
        intersections: dict[int, int] = {}
        for gram in source_grams:
            for token_id in self._gram_postings.get(gram, ()):
                intersections[int(token_id)] = intersections.get(int(token_id), 0) + 1
        if len(intersections) > POSTING_UNION_LIMIT:
            raise OovChargramBridgeG0ResourceError("LEXICON_RESOURCE")
        maximum = 1 if len(source.token) <= 7 else 2
        matches: list[BridgeMatch] = []
        for token_id in sorted(intersections):
            token = self._tokens[token_id]
            if not any(token in self._field_tokens[field] for field in source.fields):
                continue
            if abs(len(source.token) - len(token)) > maximum:
                continue
            dice = Fraction(
                2 * intersections[token_id],
                len(source_grams) + self._token_gram_counts[token_id],
            )
            if dice < MIN_DICE:
                continue
            edit = levenshtein_distance(source.token, token, maximum)
            if edit > maximum:
                continue
            matches.append(
                BridgeMatch(token, edit, dice, self._global_df[token_id])
            )
        matches.sort(
            key=lambda item: (
                item.edit_distance,
                -item.dice,
                -item.global_doc_frequency,
                item.token,
            )
        )
        result = tuple(matches[:BRIDGE_LIMIT])
        if self.cache_enabled:
            _cache_put(
                self._bridge_cache,
                self._counters["oov_bridge"],
                key,
                result,
                BRIDGE_CACHE_CAPACITY,
            )
        return result

    @staticmethod
    def _quote(token: str) -> str:
        return '"' + token.replace('"', '""') + '"'

    @classmethod
    def _expression(cls, source: BridgeSource) -> str:
        return "{" + " ".join(source.source.fields) + "} : (" + " OR ".join(
            cls._quote(match.token) for match in source.matches
        ) + ")"

    def _query(self, source: BridgeSource) -> tuple[tuple[str, ...], int, str]:
        expression = self._expression(source)
        key = (
            *self._cache_key_prefix(),
            source.source.kind,
            source.source.token,
            source.source.fields,
        )
        if self.cache_enabled:
            found, cached = _cache_get(
                self._fts_cache, self._counters["fts_route"], key
            )
            if found:
                return tuple(cached or ()), 0, expression
        agent = self._agent
        if agent is None:
            raise OovChargramBridgeG0ValidationError("Agent missing")
        started = time.perf_counter_ns()
        rows = tuple(agent.connection.execute(FTS_QUERY_SQL, (expression,)))
        elapsed = max(1, time.perf_counter_ns() - started)
        self._route_executions += 1
        identifiers = tuple(str(row[1]) for row in rows)
        if (
            len(identifiers) > ROUTE_LIMIT
            or any(not identifier for identifier in identifiers)
            or len(identifiers) != len(set(identifiers))
        ):
            raise OovChargramBridgeG0ValidationError("FTS route output invalid")
        if self.cache_enabled:
            _cache_put(
                self._fts_cache,
                self._counters["fts_route"],
                key,
                identifiers,
                FTS_ROUTE_CACHE_CAPACITY,
            )
        return identifiers, elapsed, expression

    def _view(self, identifier: str) -> ProductAttributeView:
        key = (*self._cache_key_prefix(), identifier)
        if self.cache_enabled:
            found, cached = _cache_get(
                self._product_cache, self._counters["product_view"], key
            )
            if found and cached is not None:
                return cached
        evidence = self._evidence
        if evidence is None or identifier not in evidence.offsets:
            raise OovChargramBridgeG0ValidationError("catalog evidence missing")
        start, end = evidence.offsets[identifier]
        try:
            product = json.loads(evidence.raw[start:end])
        except (UnicodeError, json.JSONDecodeError) as error:
            raise OovChargramBridgeG0ValidationError("catalog evidence invalid") from error
        if not isinstance(product, Mapping):
            raise OovChargramBridgeG0ValidationError("catalog product invalid")
        view = build_product_attribute_view(product)
        if view.parent_asin != identifier:
            raise OovChargramBridgeG0ValidationError("product view identity drifted")
        if self.cache_enabled:
            _cache_put(
                self._product_cache,
                self._counters["product_view"],
                key,
                view,
                PRODUCT_VIEW_CACHE_CAPACITY,
            )
        return view

    def _apply_mask(
        self, identifiers: tuple[str, ...], rules: HardConflictRules
    ) -> HardConflictMaskResult:
        decisions: dict[str, tuple[bool, bool]] = {}
        for identifier in identifiers:
            key = (*self._cache_key_prefix(), rules, identifier)
            if self.cache_enabled:
                found, cached = _cache_get(
                    self._mask_cache, self._counters["mask_decision"], key
                )
                if found and cached is not None:
                    decisions[identifier] = cached
                    continue
            decision = _mask_decision(self._view(identifier), rules)
            decisions[identifier] = decision
            if self.cache_enabled:
                _cache_put(
                    self._mask_cache,
                    self._counters["mask_decision"],
                    key,
                    decision,
                    MASK_DECISION_CACHE_CAPACITY,
                )
        kept: list[str] = []
        dropped: list[str] = []
        negative_count = 0
        positive_count = 0
        for identifier in identifiers:
            negative, positive = decisions[identifier]
            if negative or positive:
                dropped.append(identifier)
                negative_count += int(negative)
                positive_count += int(positive)
            else:
                kept.append(identifier)
        return HardConflictMaskResult(
            tuple(kept), tuple(dropped), negative_count, positive_count
        )

    def _empty_result(self, prefix: tuple[str, ...]) -> ExpansionResult:
        return ExpansionResult(
            candidates=prefix,
            prefix=prefix,
            tail=(),
            enabled=False,
            activated=False,
            source_records=(),
            bridge_sources=(),
            routes=(),
            novel_identifiers=(),
            filtered_identifiers=(),
            rules=HardConflictRules(),
            conflict_count=0,
            negative_violation_count=0,
            positive_conflict_count=0,
            tail_conflict_count=0,
            bridge_lookup_latency_ns=0,
            fts_route_latency_ns=0,
            hard_mask_latency_ns=0,
            query_only_readback_one=False,
            controlled_write_rejected=False,
            write_guard_unchanged=False,
        )

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
        with self._lock:
            self._require_open()
            prefix = _prefix(c200)
            if not self.enabled:
                return self._empty_result(prefix)
            try:
                self._validate_fast_locked()
                active = _terms_input(active_terms, "active_terms")
                excluded = _terms_input(excluded_terms, "excluded_terms")
                materialized_records = tuple(records)
                extracted_sources = canonical_source_records(
                    category_text=category_text,
                    active_terms=active,
                    excluded_terms=excluded,
                )
                # The preregistered source is field-OOV, not merely unknown to
                # the structured attribute registry.  Apply that exclusion
                # before mapping and before exposing source_records.
                sources = tuple(
                    source
                    for source in extracted_sources
                    if not any(
                        source.token in self._field_tokens[field]
                        for field in source.fields
                    )
                )
                lookup_started = time.perf_counter_ns()
                mapped = tuple(
                    BridgeSource(source, matches)
                    for source in sources
                    if (matches := self._map_source(source))
                )
                kind_rank = {kind: index for index, kind in enumerate(SOURCE_KIND_ORDER)}
                mapped = tuple(
                    sorted(
                        mapped,
                        key=lambda item: (
                            item.best.edit_distance,
                            -item.best.dice,
                            -item.best.global_doc_frequency,
                            item.best.token,
                            item.source.token,
                            kind_rank[item.source.kind],
                        ),
                    )[:SOURCE_LIMIT]
                )
                lookup_elapsed = max(1, time.perf_counter_ns() - lookup_started)
                prefix_set = frozenset(prefix)
                preliminary_routes: list[SourceRoute] = []
                route_elapsed_total = 0
                raw_novel: list[str] = []
                raw_seen: set[str] = set(prefix)
                for source in mapped:
                    identifiers, elapsed, expression = self._query(source)
                    route_elapsed_total += elapsed
                    novel = tuple(
                        identifier for identifier in identifiers if identifier not in prefix_set
                    )
                    preliminary_routes.append(
                        SourceRoute(source, expression, identifiers, novel, (), elapsed)
                    )
                    for identifier in novel:
                        if identifier not in raw_seen:
                            raw_seen.add(identifier)
                            raw_novel.append(identifier)
                rules = compile_hard_conflict_rules(
                    category_text=category_text,
                    active_terms=active,
                    excluded_terms=excluded,
                    current_version=current_version,
                    records=materialized_records,
                )
                mask_started = time.perf_counter_ns()
                mask = self._apply_mask(tuple(raw_novel), rules)
                mask_elapsed = max(1, time.perf_counter_ns() - mask_started)
                kept_set = frozenset(mask.identifiers)
                routes = tuple(
                    SourceRoute(
                        route.source,
                        route.expression,
                        route.identifiers,
                        route.novel_identifiers,
                        tuple(
                            identifier
                            for identifier in route.novel_identifiers
                            if identifier in kept_set
                        ),
                        route.latency_ns,
                    )
                    for route in preliminary_routes
                )
                candidates = stable_append_candidates(prefix, mask.identifiers)
                tail = candidates[len(prefix) :]
                tail_audit = self._apply_mask(tuple(tail), rules)
                mask_elapsed += max(1, time.perf_counter_ns() - mask_started) - mask_elapsed
                self._validate_fast_locked()
                if (
                    candidates[: len(prefix)] != prefix
                    or len(candidates) > CANDIDATE_CAP
                    or len(candidates) != len(set(candidates))
                    or tail_audit.conflict_count
                    or tuple(tail_audit.identifiers) != tuple(tail)
                ):
                    raise OovChargramBridgeG0ValidationError("stable append invariant failed")
                return ExpansionResult(
                    candidates=tuple(candidates),
                    prefix=prefix,
                    tail=tuple(tail),
                    enabled=True,
                    activated=bool(mapped),
                    source_records=sources,
                    bridge_sources=mapped,
                    routes=routes,
                    novel_identifiers=tuple(raw_novel),
                    filtered_identifiers=mask.identifiers,
                    rules=rules,
                    conflict_count=mask.conflict_count,
                    negative_violation_count=mask.negative_violation_count,
                    positive_conflict_count=mask.positive_conflict_count,
                    tail_conflict_count=tail_audit.conflict_count,
                    bridge_lookup_latency_ns=lookup_elapsed,
                    fts_route_latency_ns=route_elapsed_total,
                    hard_mask_latency_ns=mask_elapsed,
                    query_only_readback_one=self._query_only_readback_one,
                    controlled_write_rejected=self._controlled_write_rejected,
                    write_guard_unchanged=self._write_guard_unchanged,
                )
            except BaseException:
                agent = self._permanent_close_locked()
                if agent is not None:
                    try:
                        agent.close()
                    except BaseException:
                        pass
                raise

    def safe_expand(self, *args: object, **kwargs: object) -> ExpansionResult:
        """Deployment wrapper: expose fallback explicitly; formal code forbids it."""

        try:
            return self.expand(*args, **kwargs)  # type: ignore[arg-type]
        except BaseException:
            try:
                prefix = _prefix(args[0] if args else kwargs.get("c200", ()))
            except BaseException:
                raise
            result = self._empty_result(prefix)
            return ExpansionResult(
                **{
                    field: getattr(result, field)
                    for field in result.__dataclass_fields__
                    if field not in {"fallback", "fallback_code"}
                },
                fallback=True,
                fallback_code="EXPANSION_FAILURE",
            )

    def cache_diagnostics(self) -> dict[str, dict[str, int | bool]]:
        with self._lock:
            capacities = {
                "oov_bridge": BRIDGE_CACHE_CAPACITY,
                "fts_route": FTS_ROUTE_CACHE_CAPACITY,
                "product_view": PRODUCT_VIEW_CACHE_CAPACITY,
                "mask_decision": MASK_DECISION_CACHE_CAPACITY,
            }
            caches = {
                "oov_bridge": self._bridge_cache,
                "fts_route": self._fts_cache,
                "product_view": self._product_cache,
                "mask_decision": self._mask_cache,
            }
            return {
                name: {
                    "capacity": capacities[name],
                    "size": len(caches[name]),
                    "hits": self._counters[name].hits,
                    "misses": self._counters[name].misses,
                    "inserts": self._counters[name].inserts,
                    "evictions": self._counters[name].evictions,
                    "closed": self._closed,
                }
                for name in capacities
            }

    def route_diagnostics(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "bridge_lookups": self._bridge_lookups,
                "fts_route_executions": self._route_executions,
                "legacy_route_executions": 0,
                "query_only_readback_one": self._query_only_readback_one,
                "controlled_write_rejected": self._controlled_write_rejected,
                "write_guard_unchanged": self._write_guard_unchanged,
                "cache_clears": self._cache_clears,
                "closed": self._closed,
            }

    def close(self) -> None:
        with self._lock:
            agent = self._permanent_close_locked()
        if agent is not None:
            agent.close()

    def __enter__(self) -> "OovChargramBridgeG0Expander":
        self._require_open()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def validate() -> None:
    if (
        SOURCE_KIND_ORDER != (UNKNOWN_CATEGORY_TOKEN, EXACT_ACTIVE_TOKEN)
        or SOURCE_FIELDS[UNKNOWN_CATEGORY_TOKEN] != ("title", "categories")
        or SOURCE_FIELDS[EXACT_ACTIVE_TOKEN]
        != ("title", "features", "details", "store", "description")
        or FTS_COLUMNS
        != (
            "parent_asin",
            "title",
            "categories",
            "features",
            "details",
            "store",
            "description",
        )
        or FTS_QUERY_SQL
        != (
            "SELECT rowid,parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products,0.0,6.0,4.0,2.5,2.5,1.5,1.0),"
            "parent_asin ASC LIMIT 32"
        )
        or ALLOWED_NEGATIVE_SLOTS != _EXPECTED_ALLOWED_NEGATIVE_SLOTS
        or (PREFIX_MINIMUM, PREFIX_LIMIT, TAIL_LIMIT, CANDIDATE_CAP)
        != (100, 200, 192, 400)
        or (SOURCE_LIMIT, BRIDGE_LIMIT, ROUTE_LIMIT, POSTING_UNION_LIMIT)
        != (6, 4, 32, 8192)
        or MIN_DICE != Fraction(2, 3)
    ):
        raise OovChargramBridgeG0ValidationError("frozen constants drifted")


validate()


__all__ = [
    "BRIDGE_LIMIT",
    "BridgeMatch",
    "BridgeSource",
    "CANDIDATE_CAP",
    "EXACT_ACTIVE_TOKEN",
    "ExpansionResult",
    "HardConflictMaskResult",
    "HardConflictRules",
    "OovChargramBridgeG0ClosedError",
    "OovChargramBridgeG0Error",
    "OovChargramBridgeG0Expander",
    "OovChargramBridgeG0ResourceError",
    "OovChargramBridgeG0ValidationError",
    "POSTING_UNION_LIMIT",
    "PREFIX_LIMIT",
    "PREFIX_MINIMUM",
    "ROUTE_LIMIT",
    "SCHEMA_VERSION",
    "SOURCE_LIMIT",
    "SourceRecord",
    "SourceRoute",
    "TAIL_LIMIT",
    "UNKNOWN_CATEGORY_TOKEN",
    "apply_hard_conflict_mask",
    "boundary_trigrams",
    "canonical_source_records",
    "compile_hard_conflict_rules",
    "levenshtein_distance",
    "stable_append_candidates",
    "validate",
]
