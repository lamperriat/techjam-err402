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
from starter.reranker import score_candidate


SCHEMA_VERSION = "p12.target-blind-actions.v1"
MAX_CANDIDATES = 50

KEEP_R08 = "KEEP_R08"
KEEP_P11 = "KEEP_P11"
CANDIDATE_RERANK = "CANDIDATE_RERANK"
FROZEN_SEMANTIC_RERANK = "FROZEN_SEMANTIC_RERANK"
RESULT_AWARE_REWRITE_RETRIEVE = "RESULT_AWARE_REWRITE_RETRIEVE"
ASK = "ASK"
ACTION_IDS = (
    KEEP_R08,
    KEEP_P11,
    CANDIDATE_RERANK,
    FROZEN_SEMANTIC_RERANK,
    RESULT_AWARE_REWRITE_RETRIEVE,
    ASK,
)

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


__all__ = [
    "ACTION_IDS",
    "ASK",
    "BudgetConstraint",
    "CANDIDATE_RERANK",
    "FROZEN_SEMANTIC_RERANK",
    "KEEP_P11",
    "KEEP_R08",
    "MAX_CANDIDATES",
    "RESULT_AWARE_REWRITE_RETRIEVE",
    "SCHEMA_VERSION",
    "latest_budget_constraint",
    "numeric_budget_match",
    "rank_frozen_semantic_c50",
    "rank_structured_c50",
]
