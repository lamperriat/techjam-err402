"""Candidate-aware clarification diagnostics that never change Agent output."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from .attributes import ProductAttributeView, product_slot
from .slot_ledger import normalize_slot


SCHEMA_VERSION = "p3.question-value.v1"
QUESTION_SLOTS = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "use_case",
    "budget",
    "brand",
    "feature",
)
ANSWERABILITY = {
    "category": 0.95,
    "material": 1.0,
    "color": 1.0,
    "size": 0.95,
    "style": 0.85,
    "use_case": 0.85,
    "budget": 0.95,
    "brand": 0.55,
    "feature": 0.65,
}
MIN_COVERAGE = 0.20
LONG_TAIL_CARDINALITY = 8


@dataclass(frozen=True, slots=True)
class QuestionValue:
    attribute: str
    score: float
    information_gain: float
    coverage: float
    answerability: float
    turn_cost: float
    covered_candidates: int
    distinct_values: int
    value_counts: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _budget_bucket(price: float | None) -> str:
    if price is None:
        return ""
    if price < 25:
        return "under 25"
    if price < 50:
        return "25 to 49"
    if price < 100:
        return "50 to 99"
    return "100 or more"


def _candidate_value(product: ProductAttributeView, attribute: str) -> str:
    if attribute == "budget":
        return _budget_bucket(product.price)
    values = product_slot(product, attribute)
    if not values:
        return ""
    # One product contributes one candidate answer. Joining every extracted value
    # creates artificial high-entropy combination buckets for multi-valued fields.
    return min(values, key=lambda item: (-item.confidence, item.value)).value


def _question_value(
    products: tuple[ProductAttributeView, ...],
    attribute: str,
    turn: int,
) -> QuestionValue | None:
    values = [value for product in products if (value := _candidate_value(product, attribute))]
    if len(values) < 2:
        return None
    counts = Counter(values)
    if len(counts) < 2:
        return None
    coverage = len(values) / len(products)
    if coverage < MIN_COVERAGE:
        return None
    entropy = -sum(
        (count / len(values)) * math.log2(count / len(values))
        for count in counts.values()
    )
    information_gain = entropy / math.log2(len(counts))
    answerability = ANSWERABILITY[attribute]
    if attribute in {"brand", "feature"} and len(counts) > LONG_TAIL_CARDINALITY:
        answerability *= LONG_TAIL_CARDINALITY / len(counts)
    turn_cost = 0.04 + 0.01 * max(1, min(turn, 10))
    score = information_gain * coverage * answerability - turn_cost
    return QuestionValue(
        attribute=attribute,
        score=round(score, 9),
        information_gain=round(information_gain, 9),
        coverage=round(coverage, 9),
        answerability=round(answerability, 9),
        turn_cost=round(turn_cost, 9),
        covered_candidates=len(values),
        distinct_values=len(counts),
        value_counts=tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]),
    )


def rank_question_values(
    product_views: Mapping[str, ProductAttributeView],
    candidate_ids: Iterable[str],
    *,
    blocked_attributes: Iterable[str],
    turn: int,
) -> dict[str, object]:
    blocked = {normalize_slot(attribute) for attribute in blocked_attributes}
    products = tuple(
        product_views[parent_asin]
        for parent_asin in candidate_ids
        if parent_asin in product_views
    )
    values = [
        value
        for attribute in QUESTION_SLOTS
        if normalize_slot(attribute) not in blocked
        if (value := _question_value(products, attribute, turn)) is not None
    ]
    order = {attribute: index for index, attribute in enumerate(QUESTION_SLOTS)}
    values.sort(key=lambda value: (-value.score, order[value.attribute]))
    selected = (
        values[0].attribute
        if turn < 10 and values and values[0].score > 0
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "shadow",
        "candidate_count": len(products),
        "blocked_attributes": sorted(blocked),
        "selected_attribute": selected,
        "selection_reason": (
            "turn_limit"
            if turn >= 10 and values
            else "positive_question_value" if selected else "no_positive_question_value"
        ),
        "candidates": [value.as_dict() for value in values],
        "formula": (
            "normalized_information_gain * coverage * answerability - turn_cost; "
            "brand/feature answerability is cardinality-penalized"
        ),
    }


def empty_question_shadow(reason: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "shadow",
        "candidate_count": 0,
        "blocked_attributes": [],
        "selected_attribute": None,
        "candidates": [],
        "reason": reason,
    }
