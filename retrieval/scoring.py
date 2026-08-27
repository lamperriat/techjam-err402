from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from retrieval.catalog import CandidatePool, CatalogIndex, ProductRecord, normalized_text, terms


Intent = Literal["buying", "browsing"]


@dataclass(frozen=True)
class BudgetConstraint:
    maximum: float
    hard: bool


@dataclass(frozen=True)
class QueryContext:
    intent: Intent
    category: str
    constraints: tuple[str, ...]
    department: str | None
    budget: BudgetConstraint | None


@dataclass(frozen=True)
class ScoredProduct:
    product: ProductRecord
    score: float
    components: dict[str, float]


# Every component is normalized to [0, 1] before these weights are applied.
# Inactive semantic features are omitted and the remaining weights are
# renormalized, so missing customer constraints do not become negative signals.
INTENT_WEIGHTS: dict[Intent, dict[str, float]] = {
    "buying": {
        "lexical": 0.25,
        "category": 0.35,
        "constraint": 0.20,
        "department": 0.05,
        "bayesian_rating": 0.05,
        "popularity": 0.10,
    },
    "browsing": {
        "lexical": 0.35,
        "category": 0.20,
        "constraint": 0.15,
        "department": 0.05,
        "bayesian_rating": 0.10,
        "popularity": 0.15,
    },
}
PRICE_WEIGHT = 0.15


class ProductScorer:
    """Intent-weighted reranker with documented, independently adjustable features.

    Feature definitions:
    - lexical: FTS rank percentile; category-only candidates receive zero.
    - category: query-category token recall, with exact coarse matches set to one.
    - constraint: mean token recall for accumulated customer constraints.
    - department: exact explicit-audience match, contradiction, or neutral unknown.
    - bayesian_rating: rating shrunk toward the catalog mean using the catalog
      median rating count as the confidence parameter.
    - popularity: globally normalized log1p(rating_number).
    - price: budget compliance, activated only when the customer gives a budget.
    """

    def __init__(self, catalog: CatalogIndex) -> None:
        self.catalog = catalog

    def score(self, pool: CandidatePool, context: QueryContext) -> list[ScoredProduct]:
        lexical_count = len(pool.lexical_ranks)
        active_weights = dict(INTENT_WEIGHTS[context.intent])
        if not context.constraints:
            active_weights.pop("constraint")
        if not context.department:
            active_weights.pop("department")
        weight_total = sum(active_weights.values())

        scored: list[ScoredProduct] = []
        for parent_asin in pool.parent_asins:
            product = self.catalog.products[parent_asin]
            components = {
                "lexical": self._lexical_score(
                    pool.lexical_ranks.get(parent_asin), lexical_count
                ),
                "category": self._category_score(product, context.category),
                "constraint": self._constraint_score(product, context.constraints),
                "department": self._department_score(product, context.department),
                "bayesian_rating": self._bayesian_rating_score(product),
                "popularity": self._popularity_score(product),
            }
            ordinary_score = sum(
                active_weights[name] * components[name] for name in active_weights
            ) / weight_total
            if context.budget:
                components["price"] = self._price_score(product, context.budget)
                final_score = (1.0 - PRICE_WEIGHT) * ordinary_score + PRICE_WEIGHT * components["price"]
            else:
                final_score = ordinary_score
            scored.append(ScoredProduct(product, final_score, components))

        scored.sort(
            key=lambda item: (
                -item.score,
                pool.lexical_ranks.get(item.product.parent_asin, math.inf),
                item.product.parent_asin,
            )
        )
        return scored

    @staticmethod
    def _lexical_score(rank: int | None, result_count: int) -> float:
        if rank is None:
            return 0.0
        if result_count <= 1:
            return 1.0
        return 1.0 - (rank - 1) / (result_count - 1)

    @staticmethod
    def _category_score(product: ProductRecord, category: str) -> float:
        if normalized_text(product.coarse_category) == normalized_text(category):
            return 1.0
        query_terms = set(terms(category))
        if not query_terms:
            return 0.0
        return len(query_terms & product.category_terms) / len(query_terms)

    @staticmethod
    def _constraint_score(product: ProductRecord, constraints: tuple[str, ...]) -> float:
        if not constraints:
            return 0.0
        recalls: list[float] = []
        for constraint in constraints:
            constraint_terms = set(terms(constraint))
            if not constraint_terms:
                continue
            matched = sum(
                f" {term} " in product.searchable_tokens for term in constraint_terms
            )
            recalls.append(matched / len(constraint_terms))
        return sum(recalls) / len(recalls) if recalls else 0.0

    @staticmethod
    def _department_score(product: ProductRecord, department: str | None) -> float:
        if department is None:
            return 0.0
        if product.department is None:
            return 0.5
        return float(product.department == department)

    def _bayesian_rating_score(self, product: ProductRecord) -> float:
        confidence = self.catalog.median_rating_number
        count = product.rating_number
        adjusted = (
            count / (count + confidence) * product.average_rating
            + confidence / (count + confidence) * self.catalog.mean_rating
        )
        return max(0.0, min(1.0, (adjusted - 1.0) / 4.0))

    def _popularity_score(self, product: ProductRecord) -> float:
        return math.log1p(product.rating_number) / math.log1p(self.catalog.max_rating_number)

    @staticmethod
    def _price_score(product: ProductRecord, budget: BudgetConstraint) -> float:
        if product.price is None:
            return 0.0 if budget.hard else 0.5
        if product.price <= budget.maximum:
            return 0.75 if product.price_is_lower_bound else 1.0
        excess_ratio = (product.price - budget.maximum) / max(budget.maximum, 1.0)
        return max(0.0, 1.0 - excess_ratio)
