"""Target-blind ranking primitives for the P12 action oracle.

The functions in this module receive only visible conversation state, catalog
views, and candidate-local scores.  They never receive a sample identifier,
scenario, target product, evaluator result, or label-derived feature.

These formulas are newly frozen for the P12 v1 oracle.  They are experimental
actions, not previously validated or production-promoted ranking policies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from starter.attributes import ConversationConstraintView, ProductAttributeView
from starter.p8_negative import EXPLICIT_VIOLATION, ExecutableNegative
from starter.p11_features import CONFLICT_BUCKETS, CandidateScore
from starter.p9_evidence import (
    CompactPartition,
    SLOT_ORDER,
    VALUE_BITS,
    stable_compact_partition,
)
from starter.reranker import score_candidate


SCHEMA_VERSION = "p12.target-blind-actions.v1"
MAX_CANDIDATES = 50

KEEP_R08 = "KEEP_R08"
KEEP_P11 = "KEEP_P11"
CANDIDATE_RERANK = "CANDIDATE_RERANK"
FROZEN_SEMANTIC_RERANK = "FROZEN_SEMANTIC_RERANK"
RESULT_AWARE_REWRITE_RETRIEVE = "RESULT_AWARE_REWRITE_RETRIEVE"
COMPACT_NEGATIVE_C50 = "COMPACT_NEGATIVE_C50"
GUARDED_COMPACT_SLOT10 = "GUARDED_COMPACT_SLOT10"
GUARDED_COMPACT_SLOT10_STRICT = "GUARDED_COMPACT_SLOT10_STRICT"
P11_EVIDENCE_NOVEL_SLOT10 = "P11_EVIDENCE_NOVEL_SLOT10"
HARD_CLAUSE_NOVEL_SLOT10 = "HARD_CLAUSE_NOVEL_SLOT10"
TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10 = "TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10"
ASK = "ASK"
ACTION_IDS = (
    KEEP_R08,
    KEEP_P11,
    CANDIDATE_RERANK,
    FROZEN_SEMANTIC_RERANK,
    RESULT_AWARE_REWRITE_RETRIEVE,
    COMPACT_NEGATIVE_C50,
    GUARDED_COMPACT_SLOT10,
    GUARDED_COMPACT_SLOT10_STRICT,
    P11_EVIDENCE_NOVEL_SLOT10,
    HARD_CLAUSE_NOVEL_SLOT10,
    TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10,
    ASK,
)

P11_EVIDENCE_WEIGHTS = {
    "idf_any_field_coverage": 0.24,
    "title_category_coverage": 0.11,
    "features_details_coverage": 0.08,
    "description_store_coverage": 0.03,
    "latest_hard_clause_coverage": 0.10,
    "subtype_consistency": 0.10,
    "positive_constraint_evidence": 0.09,
}
P11_EVIDENCE_MARGIN_MIN = 0.12
P11_RELEVANCE_MARGIN_MIN = 0.04
P11_RUNNER_UP_MARGIN_MIN = 0.03
P11_SUPPORT_COUNT_MIN = 2
P11_LEXICAL_IDF_DELTA_MIN = 0.15
P11_LEXICAL_FIELD_DELTA_MIN = 0.20
P11_HARD_DELTA_MIN = 0.50
P11_SUBTYPE_DELTA_MIN = 0.50
P11_POSITIVE_DELTA_MIN = 0.45

HARD_CLAUSE_TERM_COUNT_MIN = 4
HARD_CHALLENGER_COVERAGE_MIN = 0.85
HARD_INCUMBENT_COVERAGE_MAX = 0.40
HARD_RUNNER_UP_MARGIN_MIN = 0.25
HARD_RELEVANCE_REGRESSION_MAX = 0.02

TWO_SIGNAL_LEXICAL_WEIGHTS = {
    "idf_any_field_coverage": 0.60,
    "title_category_coverage": 0.25,
    "features_details_coverage": 0.15,
}
TWO_SIGNAL_CONSTRAINT_WEIGHTS = {
    "latest_hard_clause_coverage": 0.50,
    "positive_constraint_evidence": 0.30,
    "subtype_consistency": 0.20,
}
TWO_SIGNAL_LEXICAL_INCUMBENT_MARGIN_MIN = 0.15
TWO_SIGNAL_LEXICAL_RUNNER_MARGIN_MIN = 0.05
TWO_SIGNAL_CONSTRAINT_INCUMBENT_MARGIN_MIN = 0.20
TWO_SIGNAL_CONSTRAINT_RUNNER_MARGIN_MIN = 0.10
TWO_SIGNAL_RELEVANCE_MARGIN_MIN = 0.02

_AMOUNT = r"(?P<amount>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
_CURRENCY_AMOUNT = rf"(?:usd\s*)?\$?\s*{_AMOUNT}"
_REQUIRED_CURRENCY_AMOUNT = rf"(?:usd\s*\$?|\$)\s*{_AMOUNT}"

BudgetKind = Literal["around", "max", "min"]

_BUDGET_PATTERNS: dict[BudgetKind, tuple[re.Pattern[str], ...]] = {
    "around": (
        re.compile(
            rf"\bbudget\b.{{0,24}}?\b(?:around|about|approximately|approx\.?)\s*"
            rf"{_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:around|about|approximately|approx\.?)\s*"
            rf"{_REQUIRED_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
    ),
    "max": (
        re.compile(
            rf"\bbudget\b.{{0,24}}?\b(?:under|below|up\s+to|at\s+most|"
            rf"maximum(?:\s+of)?|max(?:\s+of)?|no\s+more\s+than)\s*"
            rf"{_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:under|below|up\s+to|at\s+most|no\s+more\s+than)\s*"
            rf"{_REQUIRED_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\bbudget\b.{{0,24}}?{_CURRENCY_AMOUNT}\s*"
            rf"(?:or\s+less|max(?:imum)?)\b",
            re.IGNORECASE,
        ),
    ),
    "min": (
        re.compile(
            rf"\bbudget\b.{{0,24}}?\b(?:over|above|at\s+least|"
            rf"minimum(?:\s+of)?|min(?:\s+of)?|no\s+less\s+than)\s*"
            rf"{_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:over|above|at\s+least|no\s+less\s+than)\s*"
            rf"{_REQUIRED_CURRENCY_AMOUNT}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\bbudget\b.{{0,24}}?{_CURRENCY_AMOUNT}\s*"
            rf"(?:or\s+more|min(?:imum)?)\b",
            re.IGNORECASE,
        ),
    ),
}

_NO_BUDGET_PREFERENCE_PATTERNS = (
    re.compile(
        r"\b(?:do\s+not|don't)\s+have\s+(?:an?\s+)?(?:additional\s+)?"
        r"preference\s+for\s+(?:the\s+)?budget\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:additional\s+)?(?:preference|limit)\s+(?:for|on)\s+"
        r"(?:the\s+)?budget\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bno\s+budget\s+(?:preference|limit)\b", re.IGNORECASE),
    re.compile(
        r"\bbudget\s+(?:does\s+not|doesn't)\s+matter\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:flexible|open)\s+(?:about|on|with)\s+(?:the\s+)?budget\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bany\s+budget\b", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class BudgetConstraint:
    kind: BudgetKind
    amount: float

    def __post_init__(self) -> None:
        if self.kind not in {"around", "max", "min"}:
            raise ValueError("budget kind must be around, max, or min")
        if (
            not isinstance(self.amount, (int, float))
            or isinstance(self.amount, bool)
            or not math.isfinite(float(self.amount))
            or float(self.amount) < 0
        ):
            raise ValueError("budget amount must be a finite non-negative number")


def _amount(match: re.Match[str]) -> float | None:
    try:
        value = float(match.group("amount").replace(",", ""))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def latest_budget_constraint(
    visible_message_history: Sequence[str],
) -> BudgetConstraint | None:
    """Return the latest visible budget event, with later no-preference clearing it."""

    if isinstance(visible_message_history, (str, bytes, bytearray)):
        raise TypeError("visible_message_history must be a sequence of strings")
    selected: BudgetConstraint | None = None
    for raw_message in visible_message_history:
        if not isinstance(raw_message, str):
            raise TypeError("visible_message_history must contain only strings")
        message = raw_message.replace("\u2019", "'")
        events: list[tuple[int, int, BudgetConstraint | None]] = []
        for pattern in _NO_BUDGET_PREFERENCE_PATTERNS:
            events.extend((match.start(), match.end(), None) for match in pattern.finditer(message))
        for kind, patterns in _BUDGET_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(message):
                    amount = _amount(match)
                    if amount is not None:
                        events.append(
                            (
                                match.start(),
                                match.end(),
                                BudgetConstraint(kind, amount),
                            )
                        )
        for _, _, event in sorted(events, key=lambda item: (item[0], item[1])):
            selected = event
    return selected


def _original(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    try:
        return tuple(candidate_ids)
    except TypeError:
        return ()


def _valid_pool(pool: tuple[str, ...]) -> bool:
    return (
        len(pool) <= MAX_CANDIDATES
        and all(isinstance(identifier, str) and bool(identifier) for identifier in pool)
        and len(pool) == len(set(pool))
    )


def _finite_unit(value: object) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    number = float(value)
    return number if 0.0 <= number <= 1.0 else None


def numeric_budget_match(price: float, budget: BudgetConstraint) -> float:
    """Return the frozen P12 v1 budget score in ``[0, 1]``.

    ``max`` and ``min`` are binary constraint matches.  ``around`` is linear
    relative distance, ``max(0, 1 - abs(price - amount) / max(amount, 1))``.
    """

    if (
        not isinstance(price, (int, float))
        or isinstance(price, bool)
        or not math.isfinite(float(price))
        or float(price) < 0
        or not isinstance(budget, BudgetConstraint)
    ):
        raise ValueError("price and budget must be finite non-negative values")
    price = float(price)
    amount = float(budget.amount)
    if budget.kind == "max":
        return 1.0 if price <= amount else 0.0
    if budget.kind == "min":
        return 1.0 if price >= amount else 0.0
    denominator = max(amount, 1.0)
    return max(0.0, 1.0 - abs(price - amount) / denominator)


def rank_structured_c50(
    candidate_ids: Sequence[str],
    intent: ConversationConstraintView,
    product_views: Mapping[str, ProductAttributeView],
    normalized_priors: Mapping[str, float],
    visible_message_history: Sequence[str] = (),
) -> tuple[str, ...]:
    """Rerank one exact C50 pool with categorical and visible numeric evidence.

    Without a visible budget, the existing ``score_candidate`` total is used.
    With one, the frozen v1 final score is ``0.85 * existing_total + 0.15 *
    numeric_budget_match``.

    Any invalid, incomplete, non-finite, duplicate, or oversized input returns
    the original sequence.  Successful output is a permutation of exactly the
    supplied pool; equal final scores retain original order.
    """

    original = _original(candidate_ids)
    if not _valid_pool(original):
        return original
    if (
        not isinstance(intent, ConversationConstraintView)
        or not isinstance(product_views, Mapping)
        or not isinstance(normalized_priors, Mapping)
    ):
        return original
    try:
        budget = latest_budget_constraint(visible_message_history)
    except (TypeError, ValueError):
        return original

    scores: list[tuple[float, int, str]] = []
    for index, identifier in enumerate(original):
        prior = _finite_unit(normalized_priors.get(identifier))
        product = product_views.get(identifier)
        if (
            prior is None
            or not isinstance(product, ProductAttributeView)
            or product.parent_asin != identifier
        ):
            return original
        try:
            existing_total = float(score_candidate(intent, product, prior).total)
        except Exception:
            return original
        if not math.isfinite(existing_total):
            return original
        final = existing_total
        if budget is not None:
            price = product.price
            if (
                not isinstance(price, (int, float))
                or isinstance(price, bool)
                or not math.isfinite(float(price))
                or float(price) < 0
            ):
                return original
            final = 0.85 * existing_total + 0.15 * numeric_budget_match(
                float(price), budget
            )
        if not math.isfinite(final):
            return original
        scores.append((final, index, identifier))

    scores.sort(key=lambda item: (-item[0], item[1]))
    ranked = tuple(item[2] for item in scores)
    return ranked if set(ranked) == set(original) and len(ranked) == len(original) else original


def rank_frozen_semantic_c50(
    candidate_ids: Sequence[str],
    cosine_by_id: Mapping[str, float],
) -> tuple[str, ...]:
    """Apply the frozen 0.65 rank-prior / 0.35 min-max-cosine blend."""

    original = _original(candidate_ids)
    if not _valid_pool(original) or len(original) <= 1:
        return original
    if not isinstance(cosine_by_id, Mapping):
        return original
    cosine_values: list[float] = []
    for identifier in original:
        value = cosine_by_id.get(identifier)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            return original
        cosine_values.append(float(value))
    minimum = min(cosine_values)
    maximum = max(cosine_values)
    if maximum == minimum:
        return original

    denominator = maximum - minimum
    rank_denominator = len(original) - 1
    scored = []
    for index, (identifier, cosine) in enumerate(zip(original, cosine_values, strict=True)):
        rank_prior = 1.0 - index / rank_denominator
        normalized_cosine = (cosine - minimum) / denominator
        final = 0.65 * rank_prior + 0.35 * normalized_cosine
        if not math.isfinite(final):
            return original
        scored.append((final, index, identifier))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = tuple(item[2] for item in scored)
    return ranked if set(ranked) == set(original) and len(ranked) == len(original) else original


def _compact_partition(
    candidate_ids: Sequence[str],
    compact_evidence: Mapping[str, Sequence[int]],
    negative_constraints: Sequence[ExecutableNegative],
) -> tuple[tuple[str, ...], CompactPartition | None]:
    original = _original(candidate_ids)
    if not _valid_pool(original) or not isinstance(compact_evidence, Mapping):
        return original, None
    try:
        constraints = tuple(negative_constraints)
    except TypeError:
        return original, None
    if not constraints:
        return original, None
    if any(
        not isinstance(constraint, ExecutableNegative)
        or not VALUE_BITS.get(constraint.slot, {}).get(constraint.value, 0)
        for constraint in constraints
    ):
        return original, None
    try:
        if set(compact_evidence) != set(original):
            return original, None
        for identifier in original:
            masks = compact_evidence[identifier]
            if (
                isinstance(masks, (str, bytes, bytearray))
                or len(masks) != len(SLOT_ORDER)
            ):
                return original, None
            for slot, mask in zip(SLOT_ORDER, masks, strict=True):
                valid_bits = sum(VALUE_BITS[slot].values())
                if (
                    not isinstance(mask, int)
                    or isinstance(mask, bool)
                    or mask < 0
                    or mask & ~valid_bits
                ):
                    return original, None
        partition = stable_compact_partition(
            original,
            compact_evidence,
            constraints,
            top_k=10,
            candidate_pool=MAX_CANDIDATES,
        )
    except (KeyError, TypeError, ValueError):
        return original, None
    if (
        len(partition.identifiers) != len(original)
        or set(partition.identifiers) != set(original)
    ):
        return original, None
    return original, partition


def rank_compact_negative_c50(
    candidate_ids: Sequence[str],
    compact_evidence: Mapping[str, Sequence[int]],
    negative_constraints: Sequence[ExecutableNegative],
) -> tuple[str, ...]:
    """Stable compatible/unknown/violation partition over one P11-based C50."""

    original, partition = _compact_partition(
        candidate_ids, compact_evidence, negative_constraints
    )
    return original if partition is None else partition.identifiers


def rank_guarded_compact_slot10(
    candidate_ids: Sequence[str],
    compact_evidence: Mapping[str, Sequence[int]],
    negative_constraints: Sequence[ExecutableNegative],
) -> tuple[str, ...]:
    """Swap rank 10 for the first C50 outsider in a strictly better class.

    Compact classes are ordered compatible, unknown, explicit violation.  The
    P11 ranks 1-9 must be free of explicit violations and remain byte-for-byte
    unchanged.  Missing or invalid evidence is an exact no-op.
    """

    original, partition = _compact_partition(
        candidate_ids, compact_evidence, negative_constraints
    )
    if partition is None or len(original) <= 10:
        return original
    compatible_end = partition.compatible_count
    violation_start = compatible_end + partition.unknown_count
    compatible = frozenset(partition.identifiers[:compatible_end])
    violations = frozenset(partition.identifiers[violation_start:])
    if any(identifier in violations for identifier in original[:9]):
        return original
    unknown_end = violation_start
    unknown = frozenset(partition.identifiers[compatible_end:unknown_end])
    class_order = {
        **{identifier: 0 for identifier in compatible},
        **{identifier: 1 for identifier in unknown},
        **{identifier: 2 for identifier in violations},
    }
    incumbent_class = class_order.get(original[9])
    if incumbent_class is None or incumbent_class == 0:
        return original
    challenger_index = next(
        (
            index
            for index, identifier in enumerate(original[10:], start=10)
            if class_order.get(identifier, incumbent_class) < incumbent_class
        ),
        None,
    )
    if challenger_index is None:
        return original
    ranked = list(original)
    ranked[9], ranked[challenger_index] = ranked[challenger_index], ranked[9]
    result = tuple(ranked)
    if (
        result[:9] != original[:9]
        or len(result) != len(original)
        or set(result) != set(original)
        or len(set(result[:10]) ^ set(original[:10])) != 2
    ):
        return original
    return result


def rank_guarded_compact_slot10_strict(
    candidate_ids: Sequence[str],
    compact_evidence: Mapping[str, Sequence[int]],
    negative_constraints: Sequence[ExecutableNegative],
) -> tuple[str, ...]:
    """Apply one adjacent, high-confidence compact repair at the Top-10 edge.

    The incumbent P11 rank 10 must be an explicit violation and the adjacent
    P11 rank 11 must be compatible.  Ranks 1-9 must contain no explicit
    violation.  The only permitted change is swapping ranks 10 and 11.
    """

    original, partition = _compact_partition(
        candidate_ids, compact_evidence, negative_constraints
    )
    if partition is None or len(original) <= 10:
        return original
    compatible_end = partition.compatible_count
    violation_start = compatible_end + partition.unknown_count
    compatible = frozenset(partition.identifiers[:compatible_end])
    violations = frozenset(partition.identifiers[violation_start:])
    if (
        any(identifier in violations for identifier in original[:9])
        or original[9] not in violations
        or original[10] not in compatible
    ):
        return original
    ranked = list(original)
    ranked[9], ranked[10] = ranked[10], ranked[9]
    result = tuple(ranked)
    if (
        result[:9] != original[:9]
        or result[9] != original[10]
        or result[10] != original[9]
        or result[11:] != original[11:]
        or len(result) != len(original)
        or set(result) != set(original)
        or len(set(result[:10]) ^ set(original[:10])) != 2
    ):
        return original
    return result


@dataclass(frozen=True, slots=True)
class _NovelSlotContext:
    original: tuple[str, ...]
    incumbent: str
    safe_novel: tuple[str, ...]
    scores: Mapping[str, CandidateScore]
    evidence_scores: Mapping[str, float]


def _finite_non_negative(value: object) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        return None
    return float(value)


def _candidate_evidence_score(score: object) -> float | None:
    """Validate one frozen P11 breakdown and return its catalog evidence score."""

    if not isinstance(score, CandidateScore):
        return None
    if score.conflict_state not in CONFLICT_BUCKETS:
        return None
    if (
        _finite_non_negative(score.total) is None
        or _finite_non_negative(score.tie_bonus) is None
    ):
        return None
    unit_fields = (
        "relevance",
        "broad_rank_prior",
        "strict_rank_prior",
        "rrf_rank_prior",
        *P11_EVIDENCE_WEIGHTS,
    )
    components: dict[str, float] = {}
    for field in unit_fields:
        value = _finite_unit(getattr(score, field, None))
        if value is None:
            return None
        components[field] = value
    if round(float(score.total), 12) != round(
        components["relevance"] + float(score.tie_bonus), 12
    ):
        return None
    evidence = sum(
        weight * components[field]
        for field, weight in P11_EVIDENCE_WEIGHTS.items()
    )
    return round(evidence, 12) if math.isfinite(evidence) else None


def _novel_slot_context(
    candidate_ids: Sequence[str],
    structured_ids: Sequence[str],
    semantic_ids: Sequence[str],
    candidate_scores: Mapping[str, CandidateScore],
) -> _NovelSlotContext | None:
    """Build the shared target-blind context for all Family 2 actions."""

    original = _original(candidate_ids)
    structured = _original(structured_ids)
    semantic = _original(semantic_ids)
    if (
        not _valid_pool(original)
        or len(original) <= 10
        or not _valid_pool(structured)
        or not _valid_pool(semantic)
        or len(structured) != len(original)
        or len(semantic) != len(original)
        or set(structured) != set(original)
        or set(semantic) != set(original)
        or not isinstance(candidate_scores, Mapping)
    ):
        return None
    try:
        if set(candidate_scores) != set(original):
            return None
        scores = {identifier: candidate_scores[identifier] for identifier in original}
    except (KeyError, TypeError, ValueError):
        return None

    evidence_scores: dict[str, float] = {}
    for identifier, score in scores.items():
        evidence = _candidate_evidence_score(score)
        if evidence is None:
            return None
        evidence_scores[identifier] = evidence

    incumbent = original[9]
    incumbent_bucket = CONFLICT_BUCKETS[scores[incumbent].conflict_state]
    excluded = frozenset((*structured[:10], *semantic[:10]))
    novel_tail = tuple(
        identifier for identifier in original[10:] if identifier not in excluded
    )
    safe_novel = tuple(
        identifier
        for identifier in novel_tail
        if scores[identifier].conflict_state != EXPLICIT_VIOLATION
        and CONFLICT_BUCKETS[scores[identifier].conflict_state] <= incumbent_bucket
    )
    if len(safe_novel) < 2:
        return None
    return _NovelSlotContext(
        original=original,
        incumbent=incumbent,
        safe_novel=safe_novel,
        scores=scores,
        evidence_scores=evidence_scores,
    )


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 12)


def _ordered_safe_novel(
    context: _NovelSlotContext,
    primary_scores: Mapping[str, float],
) -> tuple[str, ...]:
    base_rank = {identifier: index for index, identifier in enumerate(context.original)}
    return tuple(
        sorted(
            context.safe_novel,
            key=lambda identifier: (
                -round(float(primary_scores[identifier]), 12),
                base_rank[identifier],
            ),
        )
    )


def _swap_rank10(
    original: tuple[str, ...], challenger: str
) -> tuple[str, ...]:
    try:
        challenger_index = original.index(challenger)
    except ValueError:
        return original
    if challenger_index < 10:
        return original
    ranked = list(original)
    ranked[9], ranked[challenger_index] = ranked[challenger_index], ranked[9]
    result = tuple(ranked)
    if (
        result[:9] != original[:9]
        or result[9] != challenger
        or result[challenger_index] != original[9]
        or len(result) != len(original)
        or set(result) != set(original)
        or len(set(result[:10]) ^ set(original[:10])) != 2
    ):
        return original
    return result


def rank_p11_evidence_novel_slot10(
    candidate_ids: Sequence[str],
    structured_ids: Sequence[str],
    semantic_ids: Sequence[str],
    candidate_scores: Mapping[str, CandidateScore],
    *,
    has_non_category_signal: bool,
) -> tuple[str, ...]:
    """Admit one novel-tail item only on corroborated P11 catalog evidence."""

    original = _original(candidate_ids)
    if has_non_category_signal is not True:
        return original
    context = _novel_slot_context(
        candidate_ids, structured_ids, semantic_ids, candidate_scores
    )
    if context is None:
        return original
    ordered = _ordered_safe_novel(context, context.evidence_scores)
    challenger, runner_up = ordered[:2]
    incumbent = context.incumbent
    challenger_score = context.scores[challenger]
    incumbent_score = context.scores[incumbent]

    if (
        _delta(
            context.evidence_scores[challenger],
            context.evidence_scores[incumbent],
        )
        < P11_EVIDENCE_MARGIN_MIN
        or _delta(challenger_score.relevance, incumbent_score.relevance)
        < P11_RELEVANCE_MARGIN_MIN
        or _delta(
            context.evidence_scores[challenger],
            context.evidence_scores[runner_up],
        )
        < P11_RUNNER_UP_MARGIN_MIN
        or _delta(
            challenger_score.latest_hard_clause_coverage,
            incumbent_score.latest_hard_clause_coverage,
        )
        < 0.0
        or _delta(
            challenger_score.subtype_consistency,
            incumbent_score.subtype_consistency,
        )
        < 0.0
        or _delta(
            challenger_score.positive_constraint_evidence,
            incumbent_score.positive_constraint_evidence,
        )
        < 0.0
    ):
        return original

    lexical_support = (
        _delta(
            challenger_score.idf_any_field_coverage,
            incumbent_score.idf_any_field_coverage,
        )
        >= P11_LEXICAL_IDF_DELTA_MIN
        and max(
            _delta(
                challenger_score.title_category_coverage,
                incumbent_score.title_category_coverage,
            ),
            _delta(
                challenger_score.features_details_coverage,
                incumbent_score.features_details_coverage,
            ),
        )
        >= P11_LEXICAL_FIELD_DELTA_MIN
    )
    support_count = sum(
        (
            lexical_support,
            _delta(
                challenger_score.latest_hard_clause_coverage,
                incumbent_score.latest_hard_clause_coverage,
            )
            >= P11_HARD_DELTA_MIN,
            _delta(
                challenger_score.subtype_consistency,
                incumbent_score.subtype_consistency,
            )
            >= P11_SUBTYPE_DELTA_MIN,
            _delta(
                challenger_score.positive_constraint_evidence,
                incumbent_score.positive_constraint_evidence,
            )
            >= P11_POSITIVE_DELTA_MIN,
        )
    )
    if support_count < P11_SUPPORT_COUNT_MIN:
        return original
    return _swap_rank10(context.original, challenger)


def rank_hard_clause_novel_slot10(
    candidate_ids: Sequence[str],
    structured_ids: Sequence[str],
    semantic_ids: Sequence[str],
    candidate_scores: Mapping[str, CandidateScore],
    *,
    hard_clause_term_count: int,
) -> tuple[str, ...]:
    """Admit one novel-tail item on a unique, field-local hard-clause match."""

    original = _original(candidate_ids)
    if (
        not isinstance(hard_clause_term_count, int)
        or isinstance(hard_clause_term_count, bool)
        or hard_clause_term_count < HARD_CLAUSE_TERM_COUNT_MIN
    ):
        return original
    context = _novel_slot_context(
        candidate_ids, structured_ids, semantic_ids, candidate_scores
    )
    if context is None:
        return original
    hard_scores = {
        identifier: context.scores[identifier].latest_hard_clause_coverage
        for identifier in context.safe_novel
    }
    ordered = _ordered_safe_novel(context, hard_scores)
    challenger, runner_up = ordered[:2]
    incumbent = context.incumbent
    challenger_score = context.scores[challenger]
    incumbent_score = context.scores[incumbent]
    if (
        challenger_score.latest_hard_clause_coverage
        < HARD_CHALLENGER_COVERAGE_MIN
        or incumbent_score.latest_hard_clause_coverage
        > HARD_INCUMBENT_COVERAGE_MAX
        or _delta(
            challenger_score.latest_hard_clause_coverage,
            context.scores[runner_up].latest_hard_clause_coverage,
        )
        < HARD_RUNNER_UP_MARGIN_MIN
        or _delta(
            challenger_score.idf_any_field_coverage,
            incumbent_score.idf_any_field_coverage,
        )
        < 0.0
        or _delta(
            challenger_score.positive_constraint_evidence,
            incumbent_score.positive_constraint_evidence,
        )
        < 0.0
        or _delta(
            challenger_score.subtype_consistency,
            incumbent_score.subtype_consistency,
        )
        < 0.0
        or _delta(challenger_score.relevance, incumbent_score.relevance)
        < -HARD_RELEVANCE_REGRESSION_MAX
    ):
        return original
    return _swap_rank10(context.original, challenger)


def rank_two_signal_consensus_novel_slot10(
    candidate_ids: Sequence[str],
    structured_ids: Sequence[str],
    semantic_ids: Sequence[str],
    candidate_scores: Mapping[str, CandidateScore],
    *,
    has_non_category_signal: bool,
) -> tuple[str, ...]:
    """Admit one novel-tail item only when lexical and constraint signals agree."""

    original = _original(candidate_ids)
    if has_non_category_signal is not True:
        return original
    context = _novel_slot_context(
        candidate_ids, structured_ids, semantic_ids, candidate_scores
    )
    if context is None:
        return original

    lexical_scores = {
        identifier: round(
            sum(
                weight * getattr(score, field)
                for field, weight in TWO_SIGNAL_LEXICAL_WEIGHTS.items()
            ),
            12,
        )
        for identifier, score in context.scores.items()
    }
    constraint_scores = {
        identifier: round(
            sum(
                weight * getattr(score, field)
                for field, weight in TWO_SIGNAL_CONSTRAINT_WEIGHTS.items()
            ),
            12,
        )
        for identifier, score in context.scores.items()
    }
    lexical_ordered = _ordered_safe_novel(context, lexical_scores)
    constraint_ordered = _ordered_safe_novel(context, constraint_scores)
    lexical_challenger, lexical_runner_up = lexical_ordered[:2]
    constraint_challenger, constraint_runner_up = constraint_ordered[:2]
    if lexical_challenger != constraint_challenger:
        return original

    challenger = lexical_challenger
    incumbent = context.incumbent
    challenger_score = context.scores[challenger]
    incumbent_score = context.scores[incumbent]
    if (
        _delta(lexical_scores[challenger], lexical_scores[incumbent])
        < TWO_SIGNAL_LEXICAL_INCUMBENT_MARGIN_MIN
        or _delta(
            lexical_scores[challenger], lexical_scores[lexical_runner_up]
        )
        < TWO_SIGNAL_LEXICAL_RUNNER_MARGIN_MIN
        or _delta(constraint_scores[challenger], constraint_scores[incumbent])
        < TWO_SIGNAL_CONSTRAINT_INCUMBENT_MARGIN_MIN
        or _delta(
            constraint_scores[challenger],
            constraint_scores[constraint_runner_up],
        )
        < TWO_SIGNAL_CONSTRAINT_RUNNER_MARGIN_MIN
        or _delta(challenger_score.relevance, incumbent_score.relevance)
        < TWO_SIGNAL_RELEVANCE_MARGIN_MIN
    ):
        return original
    return _swap_rank10(context.original, challenger)


__all__ = [
    "ACTION_IDS",
    "ASK",
    "BudgetConstraint",
    "CANDIDATE_RERANK",
    "COMPACT_NEGATIVE_C50",
    "FROZEN_SEMANTIC_RERANK",
    "GUARDED_COMPACT_SLOT10",
    "GUARDED_COMPACT_SLOT10_STRICT",
    "HARD_CLAUSE_NOVEL_SLOT10",
    "KEEP_P11",
    "KEEP_R08",
    "MAX_CANDIDATES",
    "P11_EVIDENCE_NOVEL_SLOT10",
    "RESULT_AWARE_REWRITE_RETRIEVE",
    "SCHEMA_VERSION",
    "TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10",
    "latest_budget_constraint",
    "numeric_budget_match",
    "rank_compact_negative_c50",
    "rank_frozen_semantic_c50",
    "rank_guarded_compact_slot10",
    "rank_guarded_compact_slot10_strict",
    "rank_hard_clause_novel_slot10",
    "rank_p11_evidence_novel_slot10",
    "rank_structured_c50",
    "rank_two_signal_consensus_novel_slot10",
]
