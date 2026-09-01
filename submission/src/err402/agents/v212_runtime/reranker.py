from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .attributes import (
    ConversationConstraintView,
    ProductAttributeView,
    normalize_value,
    product_slot,
)


SCORER_VERSION = "p2.constraint-rerank.v2"
RERANK_TOP_N = 50
PRESERVED_TOP_K = 10
WEIGHTS = {
    "rrf_prior": 0.45,
    "category_consistency": 0.15,
    "positive_slot_match": 0.25,
    "exact_feature_match": 0.15,
    "negative_violation": -0.10,
}


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    total: float
    rrf_prior: float
    category_consistency: float
    positive_slot_match: float
    exact_feature_match: float
    negative_violation: float
    matched_evidence: tuple[str, ...] = ()
    coverage_signature: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _tokens(values: tuple[str, ...]) -> set[str]:
    return {
        token
        for value in values
        for token in normalize_value(value).split()
        if len(token) > 1
    }


def _product_text_values(view: ProductAttributeView) -> tuple[str, ...]:
    fields = (
        view.category,
        view.audience,
        view.material,
        view.color,
        view.closure,
        view.style,
        view.use_case,
        view.size,
        view.width,
        view.brand,
        view.feature_phrases,
    )
    return tuple(item.value for values in fields for item in values)


def _slot_match(view: ProductAttributeView, slot: str, value: str) -> tuple[float, str | None]:
    candidates = product_slot(view, slot)
    for candidate in candidates:
        if candidate.value == value:
            return candidate.confidence, f"{slot}={value}"
    return 0.0, None


def _requested_slots(intent: ConversationConstraintView) -> tuple[str, ...]:
    slots = set(intent.classified_slots)
    slots.update(constraint.slot for constraint in (*intent.positive, *intent.negative))
    if intent.category_terms:
        slots.add("category")
    return tuple(sorted(slots))


def _coverage_signature(
    intent: ConversationConstraintView,
    product: ProductAttributeView,
) -> tuple[str, ...]:
    return tuple(
        slot
        for slot in _requested_slots(intent)
        if (
            product.price is not None
            if slot == "price"
            else bool(product_slot(product, slot))
        )
    )


def score_candidate(
    intent: ConversationConstraintView,
    product: ProductAttributeView,
    normalized_rrf: float,
) -> ScoreBreakdown:
    category_tokens = _tokens(tuple(item.value for item in product.category))
    requested_category = set(intent.category_terms)
    category_consistency = (
        len(category_tokens & requested_category) / len(requested_category)
        if requested_category
        else 0.0
    )

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    matched: list[str] = []
    for constraint in intent.positive:
        value, evidence = _slot_match(product, constraint.slot, constraint.value)
        positive_scores.append(value * constraint.confidence)
        if evidence:
            matched.append(evidence)
    for constraint in intent.negative:
        value, evidence = _slot_match(product, constraint.slot, constraint.value)
        negative_scores.append(value * constraint.confidence)
        if evidence:
            matched.append(f"excluded:{evidence}")
    positive_slot_match = (
        sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
    )
    negative_violation = (
        sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
    )

    product_values = _product_text_values(product)
    product_tokens = _tokens(product_values)
    exact_hits = [
        term
        for term in intent.exact_terms
        if term in product_values or term in product_tokens
    ]
    exact_feature_match = (
        len(exact_hits) / len(intent.exact_terms) if intent.exact_terms else 0.0
    )
    matched.extend(f"exact={term}" for term in exact_hits)
    for term in intent.excluded_exact_terms:
        if term in product_values or term in product_tokens:
            negative_violation = max(negative_violation, 1.0)
            matched.append(f"excluded:exact={term}")

    components = {
        "rrf_prior": max(0.0, min(1.0, normalized_rrf)),
        "category_consistency": max(0.0, min(1.0, category_consistency)),
        "positive_slot_match": max(0.0, min(1.0, positive_slot_match)),
        "exact_feature_match": max(0.0, min(1.0, exact_feature_match)),
        "negative_violation": max(0.0, min(1.0, negative_violation)),
    }
    total = sum(WEIGHTS[name] * value for name, value in components.items())
    return ScoreBreakdown(
        total=round(total, 9),
        matched_evidence=tuple(dict.fromkeys(matched)),
        coverage_signature=_coverage_signature(intent, product),
        **{name: round(value, 9) for name, value in components.items()},
    )


def has_usable_evidence(intent: ConversationConstraintView) -> bool:
    return bool(
        intent.category_terms
        or intent.positive
        or intent.negative
        or intent.exact_terms
        or intent.excluded_exact_terms
    )


def rerank_top_n(
    fused: list[str],
    fusion_scores: Mapping[str, float],
    product_views: Mapping[str, ProductAttributeView],
    intent: ConversationConstraintView,
    top_n: int = RERANK_TOP_N,
) -> tuple[list[str], dict[str, ScoreBreakdown]]:
    original = list(fused)
    pool = original[: max(0, top_n)]
    if not pool or not has_usable_evidence(intent):
        return original, {}
    maximum = max((fusion_scores.get(asin, 0.0) for asin in pool), default=0.0)
    if maximum <= 0:
        return original, {}
    breakdowns = {
        asin: score_candidate(
            intent,
            product_views.get(asin, ProductAttributeView(parent_asin=asin)),
            fusion_scores.get(asin, 0.0) / maximum,
        )
        for asin in pool
    }
    safe_pool = list(pool[:PRESERVED_TOP_K])
    # Adjacent candidates may swap only when the catalog exposes the same requested
    # slots for both. This prevents richer metadata from demoting unknown evidence.
    for index in range(1, len(safe_pool)):
        candidate = safe_pool[index]
        position = index
        while position > 0:
            incumbent = safe_pool[position - 1]
            candidate_score = breakdowns[candidate]
            incumbent_score = breakdowns[incumbent]
            if candidate_score.coverage_signature != incumbent_score.coverage_signature:
                break
            if candidate_score.total <= incumbent_score.total:
                break
            safe_pool[position - 1], safe_pool[position] = candidate, incumbent
            position -= 1
    return [*safe_pool, *original[len(safe_pool):]], breakdowns
