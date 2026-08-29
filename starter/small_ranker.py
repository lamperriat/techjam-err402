"""Pure-Python, fail-closed runtime for the frozen small-ranker research artifact.

This module deliberately has no dependency on NumPy, XGBoost, scikit-learn,
PyTorch, a network service, or a second catalog index.  It reconstructs the
target-blind C100 feature contract from visible state and the existing P11
SQLite sidecar, scores the exported trees, and may replace only P11 slot 10.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from starter.attributes import (
    ConversationConstraintView,
    ProductAttributeView,
    normalize_value,
)
from starter.p11_features import (
    NEGATIVE_SLOT_ORDER,
    P11FeatureStore,
    REGISTRY_SHA256 as P11_REGISTRY_SHA256,
    SCHEMA_VERSION as P11_SCHEMA_VERSION,
    SEMANTICS_SHA256 as P11_SEMANTICS_SHA256,
)
from starter.reranker import score_candidate


SCHEMA_VERSION = "small-ranker-runtime.v1"
ARTIFACT_SCHEMA_VERSION = "small-ranker-runtime-artifact.v1"
MODES = ("off", "shadow", "active")
TURN_COUNT = 10
CANDIDATE_COUNT = 100
EVIDENCE_CACHE_LIMIT = 16_384
HISTORY_LIMIT = 128
MAX_ARTIFACT_BYTES = 2_000_000
ASIN_SHAPE_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE)
PRICE_RE = re.compile(
    r"(?:\b(?:under|below|less than|up to|max(?:imum)?|budget(?: is| of| around)?|price(?: is)?)[^\d$]{0,12}|\$)"
    r"(\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
CONSTRAINT_SLOTS = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "price",
    "feature",
    "use_case",
)
RANK_ROUTES = (
    "coverage",
    "p11",
    "broad",
    "strict",
    "fused",
    "structured",
    "semantic",
)
ROUTE_CUTOFFS = {
    "coverage": 100,
    "p11": 10,
    "broad": 120,
    "strict": 80,
    "fused": 200,
    "structured": 10,
    "semantic": 10,
}
QUERY_VIEWS = ("turn", "goal", "active")
FIELD_NAMES = ("title_category", "features_details", "description_store")
GATE_FEATURE_NAMES = (
    "ranker_margin",
    "ranker_top_gap",
    "challenger_score",
    "incumbent_score",
    "coverage_rank_fraction",
    "broad_presence",
    "strict_presence",
    "structured_presence",
    "semantic_presence",
    "top10_route_agreement_fraction",
    "route_rank_dispersion",
    "active_title_category_idf_coverage",
    "active_features_details_idf_coverage",
    "active_description_store_idf_coverage",
    "active_token_recall",
    "active_rare_term_coverage",
    "active_bigram_coverage",
    "hard_clause_coverage",
    "explicit_negative_violation",
    "missing_positive_evidence_fraction",
    "query_specificity_fraction",
    "goal_age_fraction",
    "challenger_minus_incumbent_active_recall",
    "challenger_minus_incumbent_hard_coverage",
    "challenger_minus_incumbent_conflict_sum",
)


def _feature_names() -> tuple[str, ...]:
    names: list[str] = ["turn_fraction"]
    for route in RANK_ROUTES:
        names.extend(
            (f"{route}_presence", f"{route}_rank_fraction", f"{route}_reciprocal_rank")
        )
    names.extend(
        (
            "top10_route_agreement_fraction",
            "route_rank_mean",
            "route_rank_min",
            "route_rank_max",
            "route_rank_dispersion",
            *(f"{route}_incumbent_rr_margin" for route in RANK_ROUTES),
            "previous_presence",
            "previous_rank_fraction",
            "previous_reciprocal_rank",
            "best_historical_rank_fraction",
            "rank_velocity",
            "turn_persistence_fraction",
            "p11_structured_top10_jaccard",
            "p11_semantic_top10_jaccard",
            "broad_strict_top10_jaccard",
            "coverage_fused_top10_jaccard",
            "mean_top10_route_jaccard",
            "top10_vote_entropy",
        )
    )
    for view in QUERY_VIEWS:
        names.extend(f"{view}_{field}_idf_coverage" for field in FIELD_NAMES)
        names.extend(
            (
                f"{view}_token_recall",
                f"{view}_token_precision",
                f"{view}_token_jaccard",
                f"{view}_rare_term_coverage",
                f"{view}_char3_overlap",
                f"{view}_char4_overlap",
                f"{view}_bigram_coverage",
                f"{view}_trigram_coverage",
            )
        )
    for slot in CONSTRAINT_SLOTS:
        names.extend(
            (
                f"{slot}_observed",
                f"{slot}_compatible",
                f"{slot}_unknown",
                f"{slot}_conflict",
            )
        )
    names.extend(
        (
            "hard_clause_coverage",
            "explicit_negative_violation",
            "missing_positive_evidence_fraction",
            "current_turn_override",
            "retired_goal_evidence_conflict",
            "price_missing",
            "active_constraint_count_fraction",
            "hard_constraint_count_fraction",
            "negative_constraint_count_fraction",
            "query_specificity_fraction",
            "goal_age_fraction",
            "goal_version_fraction",
            "override_count_fraction",
            "candidate_bayesian_rating_percentile",
            "candidate_popularity_percentile",
            "title_category_length_log",
            "features_details_length_log",
            "description_store_length_log",
        )
    )
    return tuple(names)


FEATURE_NAMES = _feature_names()
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


class SmallRankerRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeOutcome:
    identifiers: tuple[str, ...]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class QueryView:
    tokens: tuple[str, ...]
    token_set: frozenset[str]
    idf_total: float
    rare_tokens: frozenset[str]
    char3_bits: int
    char4_bits: int
    bigram_bits: int
    trigram_bits: int


@dataclass(frozen=True, slots=True)
class StaticEvidence:
    catalog_rowid: int
    field_tokens: tuple[frozenset[str], frozenset[str], frozenset[str]]
    combined_tokens: frozenset[str]
    observed_values: frozenset[str]
    inferred_values: frozenset[str]
    observed_by_slot: Mapping[str, frozenset[str]]
    inferred_by_slot: Mapping[str, frozenset[str]]
    char3_bits: int
    char4_bits: int
    bigram_bits: int
    trigram_bits: int
    bayesian: float
    popularity: float


@dataclass(frozen=True, slots=True)
class BudgetConstraint:
    kind: str
    amount: float


_AMOUNT = r"(?P<amount>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
_CURRENCY_AMOUNT = rf"(?:usd\s*)?\$?\s*{_AMOUNT}"
_REQUIRED_CURRENCY_AMOUNT = rf"(?:usd\s*\$?|\$)\s*{_AMOUNT}"
_BUDGET_PATTERNS: Mapping[str, tuple[re.Pattern[str], ...]] = {
    "around": (
        re.compile(
            rf"\bbudget\b.{{0,24}}?\b(?:around|about|approximately|approx\.?)\s*{_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:around|about|approximately|approx\.?)\s*{_REQUIRED_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
    ),
    "max": (
        re.compile(
            rf"\bbudget\b.{{0,24}}?\b(?:under|below|up\s+to|at\s+most|maximum(?:\s+of)?|max(?:\s+of)?|no\s+more\s+than)\s*{_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:under|below|up\s+to|at\s+most|no\s+more\s+than)\s*{_REQUIRED_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\bbudget\b.{{0,24}}?{_CURRENCY_AMOUNT}\s*(?:or\s+less|max(?:imum)?)\b",
            re.IGNORECASE,
        ),
    ),
    "min": (
        re.compile(
            rf"\bbudget\b.{{0,24}}?\b(?:over|above|at\s+least|minimum(?:\s+of)?|min(?:\s+of)?|no\s+less\s+than)\s*{_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:over|above|at\s+least|no\s+less\s+than)\s*{_REQUIRED_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\bbudget\b.{{0,24}}?{_CURRENCY_AMOUNT}\s*(?:or\s+more|min(?:imum)?)\b",
            re.IGNORECASE,
        ),
    ),
}
_NO_BUDGET_PATTERNS = (
    re.compile(
        r"\b(?:do\s+not|don't)\s+have\s+(?:an?\s+)?(?:additional\s+)?preference\s+for\s+(?:the\s+)?budget\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:additional\s+)?(?:preference|limit)\s+(?:for|on)\s+(?:the\s+)?budget\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bno\s+budget\s+(?:preference|limit)\b", re.IGNORECASE),
    re.compile(r"\bbudget\s+(?:does\s+not|doesn't)\s+matter\b", re.IGNORECASE),
    re.compile(r"\b(?:flexible|open)\s+(?:about|on|with)\s+(?:the\s+)?budget\b", re.IGNORECASE),
    re.compile(r"\bany\s+budget\b", re.IGNORECASE),
)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _sum_f32(values: Iterable[float]) -> float:
    # NumPy's contiguous float32 reduction uses eight accumulators below its
    # larger pairwise block.  The frozen projection was produced that way, so
    # reproduce the ordering without importing NumPy in the served path.
    items = [_f32(value) for value in values]
    if len(items) < 8:
        total = 0.0
        for value in items:
            total = _f32(total + value)
        return total
    lanes = list(items[:8])
    cursor = 8
    while cursor <= len(items) - 8:
        for lane in range(8):
            lanes[lane] = _f32(lanes[lane] + items[cursor + lane])
        cursor += 8
    left = _f32(_f32(lanes[0] + lanes[1]) + _f32(lanes[2] + lanes[3]))
    right = _f32(_f32(lanes[4] + lanes[5]) + _f32(lanes[6] + lanes[7]))
    total = _f32(left + right)
    for value in items[cursor:]:
        total = _f32(total + value)
    return total


def _mean_f32(values: Sequence[float]) -> float:
    return _f32(_sum_f32(values) / len(values)) if values else 0.0


def _hash_ngrams(values: Sequence[str], size: int, bits: int = 2048) -> int:
    if len(values) < size:
        return 0
    result = 0
    for index in range(len(values) - size + 1):
        gram = "\x1f".join(values[index : index + size]).encode("utf-8")
        position = int.from_bytes(hashlib.blake2s(gram, digest_size=4).digest(), "little") % bits
        result |= 1 << position
    return result


def _hash_char_ngrams(text: str, size: int, bits: int = 2048) -> int:
    compact = f" {normalize_value(text)} "
    if len(compact) < size:
        return 0
    result = 0
    for index in range(len(compact) - size + 1):
        gram = compact[index : index + size].encode("utf-8")
        position = int.from_bytes(hashlib.blake2s(gram, digest_size=4).digest(), "little") % bits
        result |= 1 << position
    return result


def _bit_recall(query_bits: int, candidate_bits: int) -> float:
    denominator = query_bits.bit_count()
    return (query_bits & candidate_bits).bit_count() / denominator if denominator else 0.0


def _by_slot(values: Iterable[str]) -> dict[str, frozenset[str]]:
    collected: dict[str, set[str]] = {}
    for item in values:
        slot, separator, value = str(item).partition("=")
        if separator and slot and value:
            collected.setdefault(slot, set()).add(value)
    return {slot: frozenset(items) for slot, items in collected.items()}


def _query_view(tokens: Sequence[str], idf: Mapping[str, float]) -> QueryView:
    unique = tuple(dict.fromkeys(str(item) for item in tokens if str(item)))[:50]
    default_idf = math.log(50_001.0) + 1.0
    weights = [(token, float(idf.get(token, default_idf))) for token in unique]
    rare_count = max(1, math.ceil(len(weights) / 3)) if weights else 0
    rare = frozenset(
        token
        for token, _weight in sorted(weights, key=lambda item: (-item[1], item[0]))[:rare_count]
    )
    normalized_text = " ".join(unique)
    return QueryView(
        tokens=unique,
        token_set=frozenset(unique),
        idf_total=sum(weight for _token, weight in weights),
        rare_tokens=rare,
        char3_bits=_hash_char_ngrams(normalized_text, 3),
        char4_bits=_hash_char_ngrams(normalized_text, 4),
        bigram_bits=_hash_ngrams(unique, 2),
        trigram_bits=_hash_ngrams(unique, 3),
    )


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    first, second = set(left), set(right)
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    probabilities = [_f32(count / total) for count in counts]
    nonzero_count = sum(int(value > 0) for value in counts)
    if nonzero_count <= 1:
        return 0.0
    terms = [
        _f32(value * _f32(math.log(value))) if value > 0.0 else 0.0
        for value in probabilities
    ]
    numerator = _f32(-_sum_f32(terms))
    return numerator / math.log(max(nonzero_count, 2))


def _lexical_values(
    query: QueryView, evidence: StaticEvidence, idf: Mapping[str, float]
) -> list[float]:
    values: list[float] = []
    default_idf = math.log(50_001.0) + 1.0
    for field_tokens in evidence.field_tokens:
        numerator = sum(
            float(idf.get(token, default_idf)) for token in query.token_set & field_tokens
        )
        values.append(numerator / query.idf_total if query.idf_total else 0.0)
    intersection = query.token_set & evidence.combined_tokens
    union = query.token_set | evidence.combined_tokens
    values.extend(
        (
            len(intersection) / len(query.token_set) if query.token_set else 0.0,
            len(intersection) / len(evidence.combined_tokens) if evidence.combined_tokens else 0.0,
            len(intersection) / len(union) if union else 0.0,
            len(query.rare_tokens & evidence.combined_tokens) / len(query.rare_tokens)
            if query.rare_tokens
            else 0.0,
            _bit_recall(query.char3_bits, evidence.char3_bits),
            _bit_recall(query.char4_bits, evidence.char4_bits),
            _bit_recall(query.bigram_bits, evidence.bigram_bits),
            _bit_recall(query.trigram_bits, evidence.trigram_bits),
        )
    )
    return values


def _record_dict(record: Any) -> dict[str, Any]:
    return {
        "slot": str(record.slot),
        "value": str(record.value),
        "polarity": int(record.polarity),
        "hardness": str(record.hardness),
        "source_turn": int(record.source_turn),
        "version": int(record.version),
        "status": str(record.status),
    }


def _budget_upper(messages: Sequence[str]) -> float | None:
    values = [float(match.group(1)) for message in messages for match in PRICE_RE.finditer(message)]
    values = [value for value in values if math.isfinite(value) and 0.0 <= value <= 1_000_000]
    return values[-1] if values else None


def _constraint_values(
    context: Mapping[str, Any], evidence: StaticEvidence, price: float | None
) -> list[float]:
    active = [item for item in context.get("active_records", ()) if isinstance(item, Mapping)]
    retired = [item for item in context.get("retired_records", ()) if isinstance(item, Mapping)]
    positives: dict[str, list[Mapping[str, Any]]] = {}
    negatives: dict[str, list[Mapping[str, Any]]] = {}
    for record in active:
        slot = str(record.get("slot", ""))
        (positives if int(record.get("polarity", 0)) > 0 else negatives).setdefault(
            slot, []
        ).append(record)

    values: list[float] = []
    positive_total = 0
    positive_missing = 0
    explicit_negative_violation = False
    for slot in CONSTRAINT_SLOTS:
        if slot == "price":
            budget = context.get("budget_upper")
            query_present = isinstance(budget, (int, float)) and not isinstance(budget, bool)
            observed = float(query_present and price is not None)
            compatible = float(query_present and price is not None and float(price) <= float(budget))
            unknown = float(query_present and price is None)
            conflict = float(query_present and price is not None and float(price) > float(budget))
            if query_present:
                positive_total += 1
                positive_missing += int(bool(unknown))
            values.extend((observed, compatible, unknown, conflict))
            continue
        requested = positives.get(slot, ())
        rejected = negatives.get(slot, ())
        observed_values = evidence.observed_by_slot.get(slot, frozenset())
        inferred_values = evidence.inferred_by_slot.get(slot, frozenset())
        requested_values = {
            normalize_value(item.get("value"))
            for item in requested
            if normalize_value(item.get("value"))
        }
        rejected_values = {
            normalize_value(item.get("value"))
            for item in rejected
            if normalize_value(item.get("value"))
        }
        observed_match = bool(requested_values & observed_values)
        inferred_match = bool(requested_values & inferred_values)
        if slot == "feature" and requested_values:
            observed_match = observed_match or any(
                set(value.split()) <= evidence.combined_tokens for value in requested_values
            )
        negative_hit = bool(rejected_values & (observed_values | inferred_values))
        if slot == "feature" and rejected_values:
            negative_hit = negative_hit or any(
                set(value.split()) <= evidence.combined_tokens for value in rejected_values
            )
        explicit_negative_violation = explicit_negative_violation or negative_hit
        query_present = bool(requested_values or rejected_values)
        candidate_has_slot = bool(observed_values or inferred_values)
        positive_mismatch = bool(
            requested_values and candidate_has_slot and not (observed_match or inferred_match)
        )
        unknown = bool(
            requested_values and not candidate_has_slot and not observed_match and not inferred_match
        )
        values.extend(
            (
                float(observed_match),
                float(observed_match or inferred_match),
                float(query_present and unknown),
                float(negative_hit or positive_mismatch),
            )
        )
        positive_total += len(requested_values)
        positive_missing += sum(
            int(
                value not in observed_values
                and value not in inferred_values
                and not (slot == "feature" and set(value.split()) <= evidence.combined_tokens)
            )
            for value in requested_values
        )

    hard_records = [
        item
        for item in active
        if item.get("hardness") == "hard" and int(item.get("polarity", 0)) > 0
    ]
    hard_hits = 0
    for record in hard_records:
        slot = str(record.get("slot", ""))
        value = normalize_value(record.get("value"))
        hard_hits += int(
            f"{slot}={value}" in evidence.observed_values
            or f"{slot}={value}" in evidence.inferred_values
            or (slot == "feature" and set(value.split()) <= evidence.combined_tokens)
        )
    hard_terms = frozenset(str(item) for item in context.get("hard_clause_terms", ()) if str(item))
    lexical_hard_coverage = (
        len(hard_terms & evidence.combined_tokens) / len(hard_terms) if hard_terms else 0.0
    )
    hard_constraint_coverage = hard_hits / len(hard_records) if hard_records else lexical_hard_coverage
    retired_positive = [item for item in retired if int(item.get("polarity", 0)) > 0]
    retired_match = any(
        f"{record.get('slot')}={normalize_value(record.get('value'))}"
        in (evidence.observed_values | evidence.inferred_values)
        for record in retired_positive
    )
    active_match = any(
        f"{record.get('slot')}={normalize_value(record.get('value'))}"
        in (evidence.observed_values | evidence.inferred_values)
        for record in active
        if int(record.get("polarity", 0)) > 0
    )
    active_count = len(active)
    hard_count = sum(int(item.get("hardness") == "hard") for item in active)
    negative_count = sum(int(int(item.get("polarity", 0)) < 0) for item in active)
    query_terms = tuple(str(item) for item in context.get("query_terms", ()))
    turn = int(context.get("turn", 1))
    version_anchor = int(context.get("version_anchor_turn", 1))
    values.extend(
        (
            hard_constraint_coverage,
            float(explicit_negative_violation),
            positive_missing / positive_total if positive_total else 0.0,
            float(bool(context.get("current_turn_override"))),
            float(bool(retired_match and not active_match)),
            float(price is None),
            min(active_count, 20) / 20.0,
            min(hard_count, 10) / 10.0,
            min(negative_count, 10) / 10.0,
            min(len(query_terms), 50) / 50.0,
            min(max(1, turn - version_anchor + 1), 10) / 10.0,
            min(int(context.get("version", 1)), 10) / 10.0,
            min(int(context.get("override_count", 0)), 5) / 5.0,
            evidence.bayesian,
            evidence.popularity,
            math.log1p(len(evidence.field_tokens[0])) / math.log(501.0),
            math.log1p(len(evidence.field_tokens[1])) / math.log(501.0),
            math.log1p(len(evidence.field_tokens[2])) / math.log(501.0),
        )
    )
    return values


def _latest_budget(messages: Sequence[str]) -> BudgetConstraint | None:
    selected: BudgetConstraint | None = None
    for raw_message in messages:
        message = str(raw_message).replace("\u2019", "'")
        events: list[tuple[int, int, BudgetConstraint | None]] = []
        for pattern in _NO_BUDGET_PATTERNS:
            events.extend((match.start(), match.end(), None) for match in pattern.finditer(message))
        for kind, patterns in _BUDGET_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(message):
                    try:
                        amount = float(match.group("amount").replace(",", ""))
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if math.isfinite(amount) and amount >= 0:
                        events.append((match.start(), match.end(), BudgetConstraint(kind, amount)))
        for _start, _end, event in sorted(events, key=lambda item: (item[0], item[1])):
            selected = event
    return selected


def _budget_match(price: float, budget: BudgetConstraint) -> float:
    if budget.kind == "max":
        return 1.0 if price <= budget.amount else 0.0
    if budget.kind == "min":
        return 1.0 if price >= budget.amount else 0.0
    return max(0.0, 1.0 - abs(price - budget.amount) / max(budget.amount, 1.0))


def rank_structured_c50(
    candidate_ids: Sequence[str],
    intent: ConversationConstraintView,
    product_views: Mapping[str, ProductAttributeView],
    normalized_priors: Mapping[str, float],
    messages: Sequence[str],
) -> tuple[str, ...]:
    original = tuple(candidate_ids)
    if len(original) > 50 or len(original) != len(set(original)):
        return original
    try:
        budget = _latest_budget(messages)
        scores: list[tuple[float, int, str]] = []
        for index, identifier in enumerate(original):
            prior = float(normalized_priors[identifier])
            product = product_views[identifier]
            if not 0.0 <= prior <= 1.0 or product.parent_asin != identifier:
                return original
            final = float(score_candidate(intent, product, prior).total)
            if budget is not None:
                price = product.price
                if price is None or not math.isfinite(float(price)) or float(price) < 0:
                    return original
                final = 0.85 * final + 0.15 * _budget_match(float(price), budget)
            if not math.isfinite(final):
                return original
            scores.append((final, index, identifier))
        scores.sort(key=lambda item: (-item[0], item[1]))
        ranked = tuple(item[2] for item in scores)
        return ranked if set(ranked) == set(original) else original
    except Exception:
        return original


def score_tree_model(model: Mapping[str, Any], row: Sequence[float]) -> float:
    """Score one row with float32 accumulation matching exported XGBoost trees."""

    score = _f32(float(model["base_score"]))
    for tree in model["trees"]:
        node = 0
        left = tree["l"]
        right = tree["r"]
        features = tree["f"]
        values = tree["v"]
        defaults = tree["d"]
        while int(left[node]) >= 0:
            feature_value = float(row[int(features[node])])
            if math.isnan(feature_value):
                node = int(left[node]) if int(defaults[node]) else int(right[node])
            elif feature_value < float(values[node]):
                node = int(left[node])
            else:
                node = int(right[node])
        score = _f32(score + float(values[node]))
    return score


def gate_probability(gate: Mapping[str, Any], values: Sequence[float]) -> float:
    mean = gate["mean"]
    scale = gate["scale"]
    coefficients = gate["coef"]
    linear = float(gate["intercept"])
    for value, center, width, coefficient in zip(values, mean, scale, coefficients, strict=True):
        linear += float(coefficient) * (float(value) - float(center)) / float(width)
    if linear >= 0:
        return 1.0 / (1.0 + math.exp(-linear))
    exponential = math.exp(linear)
    return exponential / (1.0 + exponential)


def swap_slot10(
    baseline: Sequence[str], challenger: str, incumbent: str
) -> tuple[str, ...]:
    """Swap one challenger into slot 10 while preserving ranks 1-9 and membership."""

    original = tuple(str(identifier) for identifier in baseline)
    if (
        len(original) < 10
        or len(set(original)) != len(original)
        or original[9] != incumbent
        or challenger not in original
    ):
        raise SmallRankerRuntimeError("slot10 swap boundary is invalid")
    if challenger == incumbent:
        return original
    proposed = list(original)
    challenger_position = proposed.index(challenger)
    if challenger_position < 10:
        raise SmallRankerRuntimeError("slot10 challenger is protected")
    proposed[9], proposed[challenger_position] = proposed[challenger_position], proposed[9]
    result = tuple(proposed)
    if result[:9] != original[:9] or set(result) != set(original):
        raise SmallRankerRuntimeError("slot10 swap violated the serving boundary")
    return result


def _validate_tree_model(model: Mapping[str, Any], feature_count: int, rounds: int) -> None:
    trees = model.get("trees")
    if not isinstance(trees, list) or len(trees) != rounds:
        raise SmallRankerRuntimeError("ranker tree count mismatch")
    if not math.isfinite(float(model.get("base_score"))):
        raise SmallRankerRuntimeError("ranker base score is invalid")
    for tree in trees:
        if not isinstance(tree, Mapping):
            raise SmallRankerRuntimeError("ranker tree is invalid")
        arrays = [tree.get(name) for name in ("l", "r", "f", "v", "d")]
        if not all(isinstance(value, list) for value in arrays):
            raise SmallRankerRuntimeError("ranker tree arrays are invalid")
        lengths = {len(value) for value in arrays}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
            raise SmallRankerRuntimeError("ranker tree array lengths mismatch")
        node_count = next(iter(lengths))
        for node, (left, right, feature, value, default) in enumerate(zip(*arrays, strict=True)):
            if not math.isfinite(float(value)) or int(default) not in (0, 1):
                raise SmallRankerRuntimeError("ranker tree value is invalid")
            if int(left) < 0:
                if int(right) >= 0:
                    raise SmallRankerRuntimeError("ranker leaf children mismatch")
                continue
            if not (0 <= int(left) < node_count and 0 <= int(right) < node_count):
                raise SmallRankerRuntimeError("ranker child index is invalid")
            if not 0 <= int(feature) < feature_count or int(left) == node or int(right) == node:
                raise SmallRankerRuntimeError("ranker split is invalid")


class SmallRankerRuntime:
    """Bounded stateful runtime; all failures return the exact P11 ranking."""

    def __init__(self, mode: str, artifact_path: str | Path, sidecar_path: str | Path) -> None:
        self.mode = str(mode).strip().lower()
        if self.mode not in {"shadow", "active"}:
            raise ValueError("SmallRankerRuntime mode must be shadow or active")
        path = Path(artifact_path).resolve()
        if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise SmallRankerRuntimeError("small-ranker artifact is missing or oversized")
        raw = path.read_text(encoding="utf-8")
        if ASIN_SHAPE_RE.search(raw):
            raise SmallRankerRuntimeError("small-ranker artifact contains an identity-shaped token")
        artifact = json.loads(raw)
        if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise SmallRankerRuntimeError("small-ranker artifact schema mismatch")
        if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
            raise SmallRankerRuntimeError("small-ranker feature registry mismatch")
        ranker = artifact.get("ranker")
        gate = artifact.get("gate")
        if not isinstance(ranker, Mapping) or not isinstance(gate, Mapping):
            raise SmallRankerRuntimeError("small-ranker artifact sections are missing")
        rounds = int(ranker.get("rounds", 0))
        model = ranker.get("model")
        if not isinstance(model, Mapping):
            raise SmallRankerRuntimeError("small-ranker model is missing")
        _validate_tree_model(model, len(FEATURE_NAMES), rounds)
        if tuple(gate.get("feature_names", ())) != GATE_FEATURE_NAMES:
            raise SmallRankerRuntimeError("small-ranker gate registry mismatch")
        for name in ("mean", "scale", "coef"):
            values = gate.get(name)
            if not isinstance(values, list) or len(values) != len(GATE_FEATURE_NAMES):
                raise SmallRankerRuntimeError("small-ranker gate vector mismatch")
            if not all(math.isfinite(float(value)) for value in values):
                raise SmallRankerRuntimeError("small-ranker gate vector is non-finite")
        if any(float(value) <= 0 for value in gate["scale"]):
            raise SmallRankerRuntimeError("small-ranker gate scale is invalid")
        threshold = float(gate.get("threshold"))
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise SmallRankerRuntimeError("small-ranker threshold is invalid")

        resolved_sidecar = Path(sidecar_path).resolve()
        if not resolved_sidecar.is_file():
            raise SmallRankerRuntimeError("P11 sidecar is unavailable")
        uri = f"{resolved_sidecar.as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        connection.execute("PRAGMA query_only=ON")
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if (
                metadata.get("schema_version") != P11_SCHEMA_VERSION
                or metadata.get("registry_sha256") != P11_REGISTRY_SHA256
                or metadata.get("semantics_sha256") != P11_SEMANTICS_SHA256
                or int(metadata.get("catalog_rows", 0)) != 50_000
            ):
                raise SmallRankerRuntimeError("P11 sidecar metadata mismatch")
        except Exception:
            connection.close()
            raise
        self.artifact_path = path
        self.sidecar_path = resolved_sidecar
        self.artifact = artifact
        self.model = model
        self.gate = gate
        self.connection = connection
        self._evidence_cache: OrderedDict[str, StaticEvidence] = OrderedDict()
        self._history: OrderedDict[int, dict[int, dict[str, int]]] = OrderedDict()
        self._closed = False
        self._stats: Counter[str] = Counter(
            turns=0,
            proposals=0,
            activations=0,
            output_changes=0,
            fallbacks=0,
            evidence_rows_read=0,
        )
        self._reason_counts: Counter[str] = Counter()

    def status(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "configured_mode": self.mode,
            "effective_mode": "fallback" if self._closed else self.mode,
            "reason_code": "runtime_closed" if self._closed else "ready",
            "fallback": self._closed,
            "artifact_path": str(self.artifact_path),
            "artifact_bytes": self.artifact_path.stat().st_size if self.artifact_path.is_file() else None,
            "runtime_dependencies": "python-stdlib + existing starter modules + P11 SQLite",
            "semantic_route": "missing",
            "stats": dict(self._stats),
            "reason_counts": dict(sorted(self._reason_counts.items())),
        }

    def reset_state(self, state: Any) -> None:
        self._history.pop(id(state), None)

    def observe_coverage(self, state: Any, coverage: Sequence[str]) -> None:
        """Record one visible turn even when an upstream invariant fails closed."""

        self._record_history(state, len(state.messages), tuple(coverage[:CANDIDATE_COUNT]))

    def _previous_ranks(self, state: Any, turn: int) -> list[dict[str, int]]:
        history = self._history.get(id(state), {})
        return [history[index] for index in sorted(history) if index < turn]

    def _record_history(self, state: Any, turn: int, coverage: Sequence[str]) -> None:
        if len(coverage) != CANDIDATE_COUNT:
            return
        key = id(state)
        history = self._history.setdefault(key, {})
        history[turn] = {identifier: rank for rank, identifier in enumerate(coverage, 1)}
        self._history.move_to_end(key)
        while len(self._history) > HISTORY_LIMIT:
            self._history.popitem(last=False)

    def _idf(self, tokens: Iterable[str]) -> dict[str, float]:
        unique = tuple(dict.fromkeys(str(token) for token in tokens if str(token)))
        if not unique:
            return {}
        result: dict[str, float] = {}
        for offset in range(0, len(unique), 100):
            block = unique[offset : offset + 100]
            placeholders = ",".join("?" for _ in block)
            for term, raw_df in self.connection.execute(
                "SELECT term, document_frequency FROM term_stats WHERE term IN ("
                + placeholders
                + ")",
                block,
            ):
                result[str(term)] = math.log((50_000 + 1.0) / (int(raw_df) + 1.0)) + 1.0
        return result

    def _fetch_evidence(
        self, candidates: Sequence[str], candidate_rowids: Mapping[str, int]
    ) -> dict[str, StaticEvidence]:
        candidate_ids = tuple(candidates)
        if len(candidate_ids) > EVIDENCE_CACHE_LIMIT:
            raise SmallRankerRuntimeError("C100 candidate set exceeds evidence cache")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise SmallRankerRuntimeError("C100 candidate identifiers are not unique")

        # Protect every cached member of the current C100 before inserting new
        # evidence.  Without this touch, an old-but-current row can be the LRU
        # victim while a missing row is inserted, and the final lookup fails.
        # The cache is much larger than C100, so all current rows remain
        # resident while unrelated historical rows are evicted first.
        for identifier in candidate_ids:
            if identifier in self._evidence_cache:
                self._evidence_cache.move_to_end(identifier)

        missing = [
            identifier
            for identifier in candidate_ids
            if identifier not in self._evidence_cache
        ]
        if missing:
            rowids = [candidate_rowids.get(identifier) for identifier in missing]
            if any(not isinstance(rowid, int) or isinstance(rowid, bool) or rowid <= 0 for rowid in rowids):
                raise SmallRankerRuntimeError("C100 candidate rowid is missing")
            placeholders = ",".join("?" for _ in rowids)
            rows = self.connection.execute(
                "SELECT * FROM evidence WHERE catalog_rowid IN (" + placeholders + ")",
                rowids,
            ).fetchall()
            by_rowid = {int(row[0]): row for row in rows}
            if len(by_rowid) != len(missing):
                raise SmallRankerRuntimeError("P11 sidecar C100 evidence is incomplete")
            mask_count = len(NEGATIVE_SLOT_ORDER)
            for identifier, rowid in zip(missing, rowids, strict=True):
                row = by_rowid[int(rowid)]
                if str(row[1]) != identifier:
                    raise SmallRankerRuntimeError("P11 sidecar C100 binding mismatch")
                field_sequences, observed, inferred, _observed_subtypes, _inferred_subtypes = (
                    P11FeatureStore._decode_feature_blob(row[2])
                )
                field_tokens = tuple(
                    frozenset(token for sequence in sequences for token in sequence.split())
                    for sequences in field_sequences
                )
                ordered_tokens = tuple(
                    token
                    for sequences in field_sequences
                    for sequence in sequences
                    for token in sequence.split()
                )
                combined_text = " ".join(ordered_tokens)
                evidence = StaticEvidence(
                    catalog_rowid=int(rowid),
                    field_tokens=(field_tokens[0], field_tokens[1], field_tokens[2]),
                    combined_tokens=frozenset(ordered_tokens),
                    observed_values=observed,
                    inferred_values=inferred,
                    observed_by_slot=_by_slot(observed),
                    inferred_by_slot=_by_slot(inferred),
                    char3_bits=_hash_char_ngrams(combined_text, 3),
                    char4_bits=_hash_char_ngrams(combined_text, 4),
                    bigram_bits=_hash_ngrams(ordered_tokens, 2),
                    trigram_bits=_hash_ngrams(ordered_tokens, 3),
                    bayesian=float(row[3 + mask_count]),
                    popularity=float(row[4 + mask_count]),
                )
                self._evidence_cache[identifier] = evidence
                self._evidence_cache.move_to_end(identifier)
                while len(self._evidence_cache) > EVIDENCE_CACHE_LIMIT:
                    self._evidence_cache.popitem(last=False)
            self._stats["evidence_rows_read"] += len(missing)
        result: dict[str, StaticEvidence] = {}
        for identifier in candidate_ids:
            evidence = self._evidence_cache[identifier]
            if evidence.catalog_rowid != candidate_rowids.get(identifier):
                raise SmallRankerRuntimeError("cached C100 evidence binding mismatch")
            self._evidence_cache.move_to_end(identifier)
            result[identifier] = evidence
        return result

    @staticmethod
    def _rank_priors(
        candidates: Sequence[str], rankings: Mapping[str, Sequence[str]]
    ) -> dict[str, float]:
        broad = tuple(rankings.get("broad", ()))
        strict = tuple(rankings.get("strict", ()))
        broad_rank = {identifier: rank for rank, identifier in enumerate(broad, 1)}
        strict_rank = {identifier: rank for rank, identifier in enumerate(strict, 1)}
        raw = {
            identifier: (
                (1.0 / (60.0 + broad_rank[identifier]) if identifier in broad_rank else 0.0)
                + (1.8 / (20.0 + strict_rank[identifier]) if identifier in strict_rank else 0.0)
            )
            for identifier in candidates
        }
        maximum = max(raw.values(), default=0.0)
        if maximum <= 0.0 or any(value <= 0.0 for value in raw.values()):
            raise SmallRankerRuntimeError("C50 weighted-RRF priors are incomplete")
        return {identifier: value / maximum for identifier, value in raw.items()}

    def _features(
        self,
        *,
        state: Any,
        coverage: Sequence[str],
        p11: Sequence[str],
        rankings: Mapping[str, Sequence[str]],
        structured: Sequence[str],
        candidate_rowids: Mapping[str, int],
        prices: Mapping[str, float | None],
        turn_terms: Sequence[str],
        goal_terms: Sequence[str],
        query_terms: Sequence[str],
        current_turn_override: bool,
        hard_clause_terms: Sequence[str],
    ) -> list[list[float]]:
        turn = len(state.messages)
        route_lists: dict[str, tuple[str, ...]] = {
            "coverage": tuple(coverage),
            "p11": tuple(p11[:10]),
            "broad": tuple(rankings.get("broad", ())),
            "strict": tuple(rankings.get("strict", ())),
            "fused": tuple(rankings.get("fused", ())),
            "structured": tuple(structured[:10]),
            "semantic": (),
        }
        route_maps = {
            name: {identifier: rank for rank, identifier in enumerate(values, 1)}
            for name, values in route_lists.items()
        }
        route_top10 = {name: values[:10] for name, values in route_lists.items()}
        fixed_pairs = (
            ("p11", "structured"),
            ("p11", "semantic"),
            ("broad", "strict"),
            ("coverage", "fused"),
        )
        group_jaccards = [
            _jaccard(route_top10[left], route_top10[right]) for left, right in fixed_pairs
        ]
        all_pairs = [
            _jaccard(route_top10[left], route_top10[right])
            for left_index, left in enumerate(RANK_ROUTES)
            for right in RANK_ROUTES[left_index + 1 :]
        ]
        group_jaccards.append(
            _mean_f32([_f32(value) for value in all_pairs]) if all_pairs else 0.0
        )
        vote_counts = [
            sum(int(identifier in route_top10[route]) for route in RANK_ROUTES)
            for identifier in coverage
        ]
        vote_entropy = _entropy(vote_counts)
        idf = self._idf((*turn_terms, *goal_terms, *query_terms))
        query_views = (
            _query_view(turn_terms, idf),
            _query_view(goal_terms, idf),
            _query_view(query_terms, idf),
        )
        evidence = self._fetch_evidence(coverage, candidate_rowids)
        active_records = [_record_dict(record) for record in state.slot_ledger.active_records()]
        retired_records = [
            _record_dict(record) for record in state.slot_ledger.records if str(record.status) != "active"
        ]
        goal_messages = state.messages[max(0, int(state.version_anchor_turn) - 1) :]
        context = {
            "query_terms": tuple(query_terms),
            "version": int(state.version),
            "version_anchor_turn": int(state.version_anchor_turn),
            "override_count": int(state.override_count),
            "current_turn_override": bool(current_turn_override),
            "active_records": active_records,
            "retired_records": retired_records,
            "hard_clause_terms": tuple(hard_clause_terms),
            "budget_upper": _budget_upper(goal_messages),
            "turn": turn,
        }
        previous_ranks = self._previous_ranks(state, turn)
        incumbent = p11[9]
        rows: list[list[float]] = []
        for identifier in coverage:
            values: list[float] = [turn / TURN_COUNT]
            normalized_ranks: list[float] = []
            top10_votes = 0
            for route in RANK_ROUTES:
                rank = route_maps[route].get(identifier)
                cutoff = ROUTE_CUTOFFS[route]
                values.extend(
                    (
                        float(rank is not None),
                        rank / cutoff if rank is not None else 1.25,
                        1.0 / rank if rank is not None else 0.0,
                    )
                )
                if rank is not None:
                    normalized_ranks.append(_f32(rank / cutoff))
                top10_votes += int(identifier in route_top10[route])
            mean_rank = (
                _sum_f32(normalized_ranks) / len(normalized_ranks)
                if normalized_ranks
                else 1.25
            )
            dispersion = (
                math.sqrt(sum((value - mean_rank) ** 2 for value in normalized_ranks) / len(normalized_ranks))
                if normalized_ranks
                else 0.0
            )
            values.extend(
                (
                    top10_votes / len(RANK_ROUTES),
                    mean_rank,
                    min(normalized_ranks) if normalized_ranks else 1.25,
                    max(normalized_ranks) if normalized_ranks else 1.25,
                    dispersion,
                )
            )
            for route in RANK_ROUTES:
                candidate_rank = route_maps[route].get(identifier)
                incumbent_rank = route_maps[route].get(incumbent)
                values.append(
                    (1.0 / candidate_rank if candidate_rank else 0.0)
                    - (1.0 / incumbent_rank if incumbent_rank else 0.0)
                )
            historical = [mapping[identifier] for mapping in previous_ranks if identifier in mapping]
            previous = historical[-1] if historical else None
            current = route_maps["coverage"].get(identifier)
            values.extend(
                (
                    float(previous is not None),
                    previous / CANDIDATE_COUNT if previous is not None else 1.25,
                    1.0 / previous if previous is not None else 0.0,
                    min(historical) / CANDIDATE_COUNT if historical else 1.25,
                    (previous - current) / CANDIDATE_COUNT
                    if previous is not None and current is not None
                    else 0.0,
                    len(historical) / max(1, turn - 1),
                    *group_jaccards,
                    vote_entropy,
                )
            )
            for view in query_views:
                values.extend(_lexical_values(view, evidence[identifier], idf))
            values.extend(_constraint_values(context, evidence[identifier], prices.get(identifier)))
            if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
                raise SmallRankerRuntimeError("runtime feature row is invalid")
            rows.append([_f32(value) for value in values])
        return rows

    @staticmethod
    def _gate_features(
        rows: Sequence[Sequence[float]],
        scores: Sequence[float],
        chosen: int,
        incumbent: int,
        margin: float,
        top_gap: float,
    ) -> list[float]:
        challenger_row = rows[chosen]
        incumbent_row = rows[incumbent]
        static_names = GATE_FEATURE_NAMES[4:22]
        values = [margin, top_gap, scores[chosen], scores[incumbent]]
        values.extend(challenger_row[FEATURE_INDEX[name]] for name in static_names)
        values.extend(
            (
                challenger_row[FEATURE_INDEX["active_token_recall"]]
                - incumbent_row[FEATURE_INDEX["active_token_recall"]],
                challenger_row[FEATURE_INDEX["hard_clause_coverage"]]
                - incumbent_row[FEATURE_INDEX["hard_clause_coverage"]],
                sum(
                    challenger_row[FEATURE_INDEX[f"{slot}_conflict"]]
                    for slot in CONSTRAINT_SLOTS
                )
                - sum(
                    incumbent_row[FEATURE_INDEX[f"{slot}_conflict"]]
                    for slot in CONSTRAINT_SLOTS
                ),
            )
        )
        if len(values) != len(GATE_FEATURE_NAMES):
            raise SmallRankerRuntimeError("runtime gate row is invalid")
        return [_f32(value) for value in values]

    def _fallback(self, baseline: tuple[str, ...], reason: str) -> RuntimeOutcome:
        self._stats["fallbacks"] += 1
        self._reason_counts[reason] += 1
        head = list(baseline[:10])
        return RuntimeOutcome(
            baseline,
            {
                **self.status(),
                "effective_mode": "fallback",
                "reason_code": reason,
                "fallback": True,
                "baseline_top10": head,
                "proposed_top10": head,
                "served_top10": head,
                "activated": False,
                "output_changed": False,
                "p11_ranks_1_9_preserved": True,
            },
        )

    def apply(
        self,
        *,
        state: Any,
        coverage_ids: Sequence[str],
        p11_ids: Sequence[str],
        rankings: Mapping[str, Sequence[str]],
        candidate_rowids: Mapping[str, int],
        product_views: Mapping[str, ProductAttributeView],
        prices: Mapping[str, float | None],
        intent: ConversationConstraintView,
        turn_terms: Sequence[str],
        goal_terms: Sequence[str],
        query_terms: Sequence[str],
        current_turn_override: bool,
        hard_clause_terms: Sequence[str],
    ) -> RuntimeOutcome:
        baseline = tuple(str(identifier) for identifier in p11_ids)
        coverage = tuple(str(identifier) for identifier in coverage_ids[:CANDIDATE_COUNT])
        turn = len(state.messages)
        self._stats["turns"] += 1
        try:
            if self._closed:
                return self._fallback(baseline, "runtime_closed")
            if (
                len(coverage) != CANDIDATE_COUNT
                or len(set(coverage)) != CANDIDATE_COUNT
                or len(baseline) < CANDIDATE_COUNT
                or set(baseline[:CANDIDATE_COUNT]) != set(coverage)
                or len(set(baseline)) != len(baseline)
                or set(baseline[:10]) != set(coverage[:10])
                or turn < 1
                or turn > TURN_COUNT
            ):
                return self._fallback(baseline, "candidate_boundary_failure")
            incumbent_identifier = baseline[9]
            incumbent = coverage.index(incumbent_identifier)
            c50 = coverage[:50]
            if set(product_views) != set(c50) or set(prices) != set(coverage):
                return self._fallback(baseline, "catalog_view_failure")
            structured = rank_structured_c50(
                c50,
                intent,
                product_views,
                self._rank_priors(c50, rankings),
                tuple(state.messages),
            )
            rows = self._features(
                state=state,
                coverage=coverage,
                p11=baseline,
                rankings=rankings,
                structured=structured,
                candidate_rowids=candidate_rowids,
                prices=prices,
                turn_terms=turn_terms,
                goal_terms=goal_terms,
                query_terms=query_terms,
                current_turn_override=current_turn_override,
                hard_clause_terms=hard_clause_terms,
            )
            scores = [score_tree_model(self.model, row) for row in rows]
            allowed = [index for index in range(CANDIDATE_COUNT) if index >= 10 or index == incumbent]
            chosen = allowed[0]
            for index in allowed[1:]:
                if scores[index] > scores[chosen]:
                    chosen = index
            ordered_scores = sorted((scores[index] for index in allowed), reverse=True)
            margin = _f32(scores[chosen] - scores[incumbent])
            top_gap = _f32(scores[chosen] - ordered_scores[1])
            gate_values = self._gate_features(rows, scores, chosen, incumbent, margin, top_gap)
            probability = gate_probability(self.gate, gate_values)
            threshold = float(self.gate["threshold"])
            action_available = chosen != incumbent
            activated = bool(action_available and probability >= threshold)
            proposed = list(baseline)
            if activated:
                challenger_identifier = coverage[chosen]
                proposed = list(
                    swap_slot10(
                        baseline,
                        challenger_identifier,
                        incumbent_identifier,
                    )
                )
            if proposed[:9] != list(baseline[:9]) or len(proposed) != len(baseline) or set(proposed) != set(baseline):
                return self._fallback(baseline, "slot10_boundary_failure")
            proposed_tuple = tuple(proposed)
            served = proposed_tuple if self.mode == "active" else baseline
            reason = "activated" if activated else ("incumbent_best" if not action_available else "gate_rejected")
            self._reason_counts[reason] += 1
            self._stats["proposals"] += int(action_available)
            self._stats["activations"] += int(activated)
            self._stats["output_changes"] += int(served[:10] != baseline[:10])
            return RuntimeOutcome(
                served,
                {
                    **self.status(),
                    "reason_code": reason,
                    "fallback": False,
                    "baseline_top10": list(baseline[:10]),
                    "proposed_top10": list(proposed_tuple[:10]),
                    "served_top10": list(served[:10]),
                    "activated": activated,
                    "output_changed": served[:10] != baseline[:10],
                    "p11_ranks_1_9_preserved": proposed_tuple[:9] == baseline[:9],
                    "full_membership_preserved": set(proposed_tuple) == set(baseline),
                    "challenger_coverage_rank": chosen + 1,
                    "incumbent_coverage_rank": incumbent + 1,
                    "ranker_margin": margin,
                    "ranker_top_gap": top_gap,
                    "gate_probability": probability,
                    "gate_threshold": threshold,
                    "feature_count": len(FEATURE_NAMES),
                    "candidate_count": CANDIDATE_COUNT,
                },
            )
        except Exception as error:
            return self._fallback(baseline, f"runtime_error:{type(error).__name__}")
        finally:
            self._record_history(state, turn, coverage)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.connection.close()
        self._evidence_cache.clear()
        self._history.clear()


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "FEATURE_NAMES",
    "GATE_FEATURE_NAMES",
    "MODES",
    "RuntimeOutcome",
    "SCHEMA_VERSION",
    "SmallRankerRuntime",
    "SmallRankerRuntimeError",
    "gate_probability",
    "rank_structured_c50",
    "score_tree_model",
    "swap_slot10",
]
