from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agents.v1 import AgentV1, SessionState
from retrieval.attributes import ExtractedAttributeIndex
from retrieval.catalog import extract_department, normalized_text
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


@dataclass
class V2SessionState(SessionState):
    asked_public_attributes: set[str] = field(default_factory=set)


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
        constraint_texts = tuple(constraint.text for constraint in state.constraints)
        query_text = " ".join((state.category, *constraint_texts))
        pool = self.catalog.candidates(state.category, query_text)
        context = QueryContext(
            intent=state.intent,
            category=state.category,
            constraints=constraint_texts,
            department=extract_department(query_text),
            budget=self._budget_constraint(state.constraints),
        )
        ranked = self.scorer.score(pool, context)
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
            state.asked_attributes.add(question.facet)
            state.asked_public_attributes.add(public_attribute)
        else:
            state.last_asked_attribute = None

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
                for item in ranked[:top_k]
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _select_fine_question(
        self,
        state: V2SessionState,
        candidates: list[ScoredProduct],
    ) -> QuestionFacet | None:
        group = self.catalog.question_category(state.category)
        facets = self._question_facets(group)
        profile_families = self._profile_families(state.user_profile)
        scores: dict[QuestionFacet, float] = {}
        for facet in facets:
            if facet.facet in state.asked_attributes | state.no_preference_attributes:
                continue
            if (
                self.question_mode == "benchmark"
                and facet.benchmark_attribute in state.asked_public_attributes
            ):
                continue
            values = [
                value
                for candidate in candidates
                if (value := self._facet_value(candidate, facet)) is not None
            ]
            if len(values) < MIN_ATTRIBUTE_PRODUCTS:
                continue
            utility = self._question_utility(values, len(candidates))
            if utility is None:
                continue
            scores[facet] = (
                0.50 * utility.discrimination
                + 0.15 * utility.expected_answerability
                + 0.30 * FAMILY_PRIORS[group][facet.family]
                + 0.05 * float(facet.family in profile_families)
            )
        if not scores:
            return None
        return max(scores, key=lambda facet: (scores[facet], -facets.index(facet)))

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
        )

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
