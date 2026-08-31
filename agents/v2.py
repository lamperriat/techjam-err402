from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agents.v1 import OVERRIDE_RE, AgentV1, SessionState
from retrieval.attributes import ExtractedAttributeIndex
from retrieval.catalog import (
    CandidatePool,
    extract_department,
    fine_category_group,
    normalized_text,
    terms,
)
from retrieval.scoring import QueryContext, ScoredProduct


QuestionMode = Literal["benchmark", "native"]
QuestionSource = Literal["core", "catalog", "specific"]
MIN_ATTRIBUTE_PRODUCTS = 5
QUESTION_CANDIDATE_LIMIT = 100

# A useful deterministic question needs at least two choices that occur more
# than once. Above eight recurring choices, the square-root penalty reduces the
# high-cardinality bias without imposing a hard cutoff.
MIN_RECURRING_VALUE_COUNT = 2
MAX_ANSWER_CHOICES = 8


@dataclass(frozen=True)
class QuestionFacet:
    facet: str
    family: str
    source: QuestionSource
    benchmark_attribute: str


@dataclass(frozen=True)
class QuestionUtility:
    discrimination: float
    expected_answerability: float
    recurring_value_count: int
    cardinality_penalty: float
    recurring_values: frozenset[str]


@dataclass(frozen=True)
class FacetEvaluation:
    facet: QuestionFacet
    utility: QuestionUtility
    answerable_products: frozenset[str]
    category_novelty: float


@dataclass
class V2SessionState(SessionState):
    asked_public_attributes: set[str] = field(default_factory=set)
    last_asked_public_attribute: str | None = None


FAMILY_PRIORS = {
    "clothing": {
        "size_fit": 1.00,
        "comfort": 0.93,
        "style": 0.80,
        "material": 0.72,
        "durability": 0.63,
        "brand": -0.25,
        "color": 0.40,
        "occasion": 0.30,
        "weather_requirement": 0.25,
        "budget": 0.10,
    },
    "shoes": {
        "comfort": 1.00,
        "size_fit": 0.96,
        "durability": 0.72,
        "functional_purpose": 0.65,
        "brand": -0.25,
        "material": 0.55,
        "color": 0.55,
        "style": 0.50,
        "terrain_weather": 0.45,
        "budget": 0.10,
    },
    "jewelry": {
        "gemstone_details": 1.00,
        "occasion": 0.74,
        "material_metal": 0.74,
        "personalization": 0.69,
        "gemstone_color_birthstone": 0.60,
        "design_style": 0.54,
        "gemstone_size_carat": 0.53,
        "ethical_sourcing": 0.37,
        "construction": 0.30,
        "ring_size": 0.30,
        "brand": -0.25,
        "budget": 0.10,
    },
    "other": {
        "size_fit": 0.50,
        "feature": 0.50,
        "style": 0.50,
        "material": 0.50,
        "color": 0.50,
        "use_case": 0.50,
        "brand": -0.25,
        "budget": 0.10,
    },
}


COMMON_FACETS = {
    "clothing": (
        QuestionFacet("size_fit", "size_fit", "core", "size"),
        QuestionFacet("style", "style", "core", "style"),
        QuestionFacet("material", "material", "core", "material"),
        QuestionFacet("color", "color", "core", "color"),
        QuestionFacet("use_case", "occasion", "core", "use_case"),
        QuestionFacet("brand", "brand", "catalog", "brand"),
        QuestionFacet("budget", "budget", "catalog", "budget"),
    ),
    "shoes": (
        QuestionFacet("size_fit", "size_fit", "core", "size"),
        QuestionFacet("style", "style", "core", "style"),
        QuestionFacet("material", "material", "core", "material"),
        QuestionFacet("color", "color", "core", "color"),
        QuestionFacet("use_case", "functional_purpose", "core", "use_case"),
        QuestionFacet("brand", "brand", "catalog", "brand"),
        QuestionFacet("budget", "budget", "catalog", "budget"),
    ),
    "jewelry": (
        QuestionFacet("material", "material_metal", "core", "material"),
        QuestionFacet("color", "gemstone_color_birthstone", "core", "color"),
        QuestionFacet("style", "design_style", "core", "style"),
        QuestionFacet("size_fit", "ring_size", "core", "size"),
        QuestionFacet("use_case", "occasion", "core", "use_case"),
        QuestionFacet("brand", "brand", "catalog", "brand"),
        QuestionFacet("budget", "budget", "catalog", "budget"),
    ),
    "other": (
        QuestionFacet("size_fit", "size_fit", "core", "size"),
        QuestionFacet("style", "style", "core", "style"),
        QuestionFacet("material", "material", "core", "material"),
        QuestionFacet("color", "color", "core", "color"),
        QuestionFacet("use_case", "use_case", "core", "use_case"),
        QuestionFacet("brand", "brand", "catalog", "brand"),
        QuestionFacet("budget", "budget", "catalog", "budget"),
    ),
}


SPECIFIC_FACETS = {
    "clothing": {
        "comfort": ("cushioning", "breathability", "stretch"),
        "durability": ("construction", "stitching"),
        "style": ("closure",),
        "weather_requirement": (
            "water_resistance",
            "moisture_wicking",
            "sun_protection",
            "insulation",
        ),
    },
    "shoes": {
        "comfort": ("cushioning", "arch_support", "footbed", "breathability"),
        "durability": ("construction", "toe_protection"),
        "functional_purpose": ("shoe_type", "stability", "shock_absorption"),
        "style": ("closure", "heel_type", "toe_style"),
        "terrain_weather": (
            "water_resistance",
            "slip_resistance",
            "traction",
            "insulation",
        ),
    },
    "jewelry": {
        "gemstone_details": (
            "gemstone",
            "stone_type",
            "gemstone_type",
            "stone_cut",
            "gemstone_cut",
            "clarity",
            "diamond_clarity",
            "gemstone_treatment",
        ),
        "personalization": ("engraving", "personalization"),
        "gemstone_color_birthstone": ("birthstone",),
        "design_style": ("design", "finish", "chain_type"),
        "gemstone_size_carat": (
            "carat_weight",
            "diamond_carat_weight",
            "total_carat_weight",
            "stone_size",
        ),
        "ethical_sourcing": ("conflict_free", "certification"),
        "construction": ("handmade", "handcrafted", "setting", "stone_setting"),
        "ring_size": ("ring_size",),
    },
    "other": {
        "feature": (
            "closure",
            "water_resistance",
            "construction",
            "movement",
            "capacity",
            "uv_protection",
            "polarization",
        ),
    },
}


PROFILE_FAMILIES = {
    "material": {"material", "material_metal"},
    "style": {"style", "design_style"},
    "fit": {"size_fit", "ring_size"},
    "weather": {"weather_requirement", "terrain_weather"},
    "warmth": {"weather_requirement", "terrain_weather"},
    "performance": {"functional_purpose"},
    "comfort": {"comfort"},
    "durability": {"durability"},
    "feature": {"feature"},
}


class AgentV2(AgentV1):
    """V1 retrieval with offline-LLM fine-grained clarification facets."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        attributes_path: str | Path = "results/catalog_attributes_processed.jsonl",
        question_mode: QuestionMode = "benchmark",
    ) -> None:
        if question_mode not in ("benchmark", "native"):
            raise ValueError("question_mode must be 'benchmark' or 'native'")
        super().__init__(catalog_path)
        self.attributes = ExtractedAttributeIndex(attributes_path)
        if self.attributes.product_ids != set(self.catalog.products):
            self.close()
            raise ValueError("Processed attributes and catalog product IDs do not match")
        self.question_mode = question_mode

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = V2SessionState(user_profile=dict(user_profile))

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        if not isinstance(state, V2SessionState):
            raise RuntimeError("V2 session state was not initialized correctly")

        self._update_state(state, user_message, turn)
        ranked = self._rank_products(state)
        unseen_ranked = [
            item
            for item in ranked
            if item.product.parent_asin not in state.shown_product_ids
        ]
        question = self._select_fine_question(
            state,
            ranked[:QUESTION_CANDIDATE_LIMIT],
        )
        public_attribute: str | None = None
        if question:
            public_attribute = (
                question.benchmark_attribute
                if self.question_mode == "benchmark"
                else question.facet
            )
            state.last_asked_attribute = question.facet
            state.last_asked_public_attribute = public_attribute
            state.asked_attributes.add(question.facet)
            state.asked_public_attributes.add(public_attribute)
            state.follow_up_attributes.discard(question.facet)
            state.question_counts[question.facet] += 1
        else:
            state.last_asked_attribute = None
            state.last_asked_public_attribute = None

        recommendations = unseen_ranked[:top_k]
        state.shown_product_ids.update(
            item.product.parent_asin for item in recommendations
        )

        return {
            "message": (
                self._question_message(question.facet)
                if question
                else "Here are the closest matches based on what you have told me."
            ),
            "ask_attribute": public_attribute,
            "recommendations": [
                {
                    "parent_asin": item.product.parent_asin,
                    "score": round(item.score, 6),
                }
                for item in recommendations
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _update_state(self, state: SessionState, message: str, turn: int) -> None:
        pending_public_attribute = (
            state.last_asked_public_attribute
            if isinstance(state, V2SessionState)
            else None
        )
        super()._update_state(state, message, turn)
        if (
            isinstance(state, V2SessionState)
            and pending_public_attribute
            and OVERRIDE_RE.match(message)
        ):
            state.asked_public_attributes.discard(pending_public_attribute)

    @staticmethod
    def _query_text(state: V2SessionState) -> str:
        constraint_texts = tuple(constraint.text for constraint in state.constraints)
        return " ".join((state.category, *constraint_texts))

    def _score_pool(
        self,
        state: V2SessionState,
        pool: CandidatePool,
    ) -> list[ScoredProduct]:
        constraint_texts = tuple(constraint.text for constraint in state.constraints)
        query_text = self._query_text(state)
        context = QueryContext(
            intent=state.intent,
            category=state.category,
            constraints=constraint_texts,
            department=extract_department(query_text),
            budget=self._budget_constraint(state.constraints),
        )
        return self.scorer.score(pool, context)

    def _rank_products(self, state: V2SessionState) -> list[ScoredProduct]:
        query_text = self._query_text(state)
        pool = self.catalog.candidates(state.category, query_text)
        return self._prioritize_exact_category(
            self._score_pool(state, pool),
            state.category,
        )

    def _select_fine_question(
        self,
        state: V2SessionState,
        candidates: list[ScoredProduct],
    ) -> QuestionFacet | None:
        if not candidates:
            return None
        group = self._question_group(candidates)
        facets = self._question_facets(group)
        profile_families = self._profile_families(state.user_profile)
        fresh_evaluations: list[FacetEvaluation] = []
        follow_up_evaluations: list[FacetEvaluation] = []
        for facet in facets:
            if facet.facet in state.no_preference_attributes:
                continue
            is_follow_up = facet.facet in state.follow_up_attributes
            is_fresh = facet.facet not in state.asked_attributes and not (
                self.question_mode == "benchmark"
                and facet.benchmark_attribute in state.asked_public_attributes
            )
            if not is_fresh and not is_follow_up:
                continue
            product_values = [
                (candidate.product.parent_asin, value)
                for candidate in candidates
                if (value := self._facet_value(candidate, facet)) is not None
            ]
            if len(product_values) < MIN_ATTRIBUTE_PRODUCTS:
                continue
            values = [value for _, value in product_values]
            if not is_follow_up and self._facet_is_disclosed(state, facet, values):
                continue
            utility = self._question_utility(values, len(candidates))
            if utility is None:
                continue
            novelty = self._category_novelty(
                values,
                utility.recurring_values,
                state.category,
            )
            if novelty <= 0:
                continue
            answerable_products = frozenset(
                parent_asin
                for parent_asin, value in product_values
                if normalized_text(value) in utility.recurring_values
            )
            evaluation = FacetEvaluation(facet, utility, answerable_products, novelty)
            if is_fresh:
                fresh_evaluations.append(evaluation)
            else:
                follow_up_evaluations.append(evaluation)
        evaluations = fresh_evaluations or follow_up_evaluations
        if not evaluations:
            return None

        by_family: dict[str, list[FacetEvaluation]] = {}
        for evaluation in evaluations:
            by_family.setdefault(evaluation.facet.family, []).append(evaluation)

        family_scores: dict[str, tuple[float, FacetEvaluation]] = {}
        for family, members in by_family.items():
            best_leaf = max(
                members,
                key=lambda item: (
                    self._leaf_data_utility(item),
                    -facets.index(item.facet),
                ),
            )
            family_answerable = len(
                set().union(*(member.answerable_products for member in members))
            ) / len(candidates)
            score = best_leaf.category_novelty * (
                0.50 * best_leaf.utility.discrimination
                + 0.15 * family_answerable
                + 0.30 * FAMILY_PRIORS[group][family]
                + 0.05 * float(family in profile_families)
            )
            family_scores[family] = (score, best_leaf)

        return max(
            family_scores.values(),
            key=lambda result: (
                result[0],
                -facets.index(result[1].facet),
            ),
        )[1].facet

    @staticmethod
    def _leaf_data_utility(evaluation: FacetEvaluation) -> float:
        return evaluation.category_novelty * (
            0.50 * evaluation.utility.discrimination
            + 0.15 * evaluation.utility.expected_answerability
        )

    @staticmethod
    def _question_utility(
        values: list[str],
        candidate_count: int,
    ) -> QuestionUtility | None:
        counts = Counter(normalized_text(value) for value in values)
        recurring = {
            value: count
            for value, count in counts.items()
            if count >= MIN_RECURRING_VALUE_COUNT
        }
        if len(recurring) < 2 or candidate_count <= 0:
            return None

        recurring_products = sum(recurring.values())
        expected_answerability = recurring_products / candidate_count
        entropy = -sum(
            (count / recurring_products) * math.log(count / recurring_products)
            for count in recurring.values()
        ) / math.log(len(recurring))
        cardinality_penalty = min(
            1.0,
            math.sqrt(MAX_ANSWER_CHOICES / len(recurring)),
        )
        return QuestionUtility(
            discrimination=expected_answerability * entropy * cardinality_penalty,
            expected_answerability=expected_answerability,
            recurring_value_count=len(recurring),
            cardinality_penalty=cardinality_penalty,
            recurring_values=frozenset(recurring),
        )

    @staticmethod
    def _question_group(candidates: list[ScoredProduct]) -> str:
        counts = Counter(
            fine_category_group(list(candidate.product.category_path))
            for candidate in candidates
        )
        return counts.most_common(1)[0][0]

    @classmethod
    def _facet_is_disclosed(
        cls,
        state: V2SessionState,
        facet: QuestionFacet,
        values: list[str],
    ) -> bool:
        if facet.facet == "budget" and cls._budget_constraint(state.constraints):
            return True
        for constraint in state.constraints:
            constraint_terms = set(terms(constraint.text))
            if not constraint_terms:
                continue
            for value in values:
                value_terms = set(terms(value))
                if value_terms and (
                    value_terms <= constraint_terms
                    or constraint_terms <= value_terms
                ):
                    return True
        return False

    @staticmethod
    def _category_novelty(
        values: list[str],
        recurring_values: frozenset[str],
        category: str,
    ) -> float:
        category_terms = set(terms(category))
        recurring = [
            value
            for value in values
            if normalized_text(value) in recurring_values
        ]
        if not category_terms or not recurring:
            return 1.0
        implied = sum(
            bool(value_terms := set(terms(value)))
            and value_terms <= category_terms
            for value in recurring
        )
        return 1.0 - implied / len(recurring)

    @staticmethod
    def _question_facets(group: str) -> tuple[QuestionFacet, ...]:
        specific = tuple(
            QuestionFacet(name, family, "specific", "feature")
            for family, names in SPECIFIC_FACETS[group].items()
            for name in names
        )
        return (*COMMON_FACETS[group], *specific)

    def _facet_value(
        self,
        candidate: ScoredProduct,
        facet: QuestionFacet,
    ) -> str | None:
        parent_asin = candidate.product.parent_asin
        if facet.source == "core":
            return self.attributes.core_value(parent_asin, facet.facet)
        if facet.source == "specific":
            return self.attributes.specific_value(parent_asin, facet.facet)
        return candidate.product.attribute_value(facet.facet)

    @staticmethod
    def _question_message(facet: str) -> str:
        if facet == "budget":
            return "What budget range should I use?"
        if facet == "use_case":
            return "What activity or occasion should this product support?"
        return f"Do you have a {facet.replace('_', ' ')} preference?"

    @staticmethod
    def _profile_families(profile: dict) -> set[str]:
        families: set[str] = set()
        for tag in profile.get("preference_tags") or []:
            families.update(PROFILE_FAMILIES.get(str(tag).strip().lower(), set()))
        return families
