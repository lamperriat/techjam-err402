from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from retrieval.catalog import CatalogIndex, extract_department, normalized_text
from retrieval.scoring import (
    BudgetConstraint,
    Intent,
    ProductScorer,
    QueryContext,
    ScoredProduct,
    ScoringConfig,
)


BUYING_INITIAL_RE = re.compile(
    r"^I'm looking for (.+?)\. A key requirement is: (.+)\.$",
    re.DOTALL,
)
BROWSING_INITIAL_RE = re.compile(
    r"^I'm looking for (.+), but I'm still exploring\.$",
    re.DOTALL,
)
PREFERENCE_INITIAL_RE = re.compile(r"^I'm looking for (.+?)\. (.+)$", re.DOTALL)
OVERRIDE_RE = re.compile(
    r"^Actually, ignore my earlier preference\. What I need is: (.+)\.$",
    re.DOTALL,
)
CLARIFICATION_RE = re.compile(r"^For that, what matters is: (.+)\.$", re.DOTALL)
BUDGET_PATTERNS = (
    re.compile(r"(?:under|below|less than|up to|maximum|max)\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
    re.compile(r"budget(?:\s+(?:around|of))?\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
    re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(?:or less|maximum|max)", re.I),
)

QUESTION_ATTRIBUTES = (
    "material", "color", "style", "size", "budget", "feature", "use_case",
)
MIN_ATTRIBUTE_PRODUCTS = 5
QUESTION_CANDIDATE_LIMIT = 100
MAX_QUESTIONS_PER_ATTRIBUTE = 2
MALFORMED_CATEGORY_PREFIX = "shoes & jewelry "

# These question weights account for both customer importance and catalog
# coverage. Brand is intentionally excluded from V1's question policy: the
# local evaluator cannot provide a brand-specific response.
CATEGORY_ATTRIBUTE_WEIGHTS = {
    "clothing": {
        "material": 0.72,
        "color": 0.40,
        "style": 0.80,
        "size": 1.00,
        "budget": 0.10,
        "feature": 0.93,
        "use_case": 0.25,
    },
    "shoes": {
        "material": 0.55,
        "color": 0.55,
        "style": 0.50,
        "size": 0.96,
        "budget": 0.10,
        "feature": 1.00,
        "use_case": 0.65,
    },
    "jewelry": {
        "material": 0.74,
        "color": 0.60,
        "style": 0.85,
        "size": 0.30,
        "budget": 0.10,
        "feature": 1.00,
        "use_case": 0.75,
    },
}
QUESTION_TEMPLATES = {
    "material": "Do you have a material preference?",
    "color": "Do you have a color preference?",
    "style": "Do you have a department, style, or fit preference?",
    "size": "What size or fit do you need?",
    "budget": "What budget range should I use?",
    "feature": "Which product feature matters most to you?",
    "use_case": "What activity or use case should this product support?",
}
PROFILE_ATTRIBUTE_MAP = {
    "material": {"material"},
    "style": {"style"},
    "fit": {"style"},
    "weather": {"use_case", "feature"},
    "warmth": {"use_case", "feature"},
    "performance": {"use_case"},
    "comfort": {"feature"},
    "durability": {"feature"},
}


@dataclass(frozen=True)
class Constraint:
    text: str
    hard: bool
    source: str
    constraint_id: str | None = None


@dataclass
class SessionState:
    user_profile: dict
    intent: Intent = "buying"
    category: str = ""
    constraints: list[Constraint] = field(default_factory=list)
    overridable_preference: str | None = None
    asked_attributes: set[str] = field(default_factory=set)
    no_preference_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None
    question_counts: Counter[str] = field(default_factory=Counter)
    follow_up_attributes: set[str] = field(default_factory=set)
    shown_product_ids: set[str] = field(default_factory=set)


class AgentV1:
    """First non-LLM agent with stateful weighted retrieval and clarification."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        scoring_config: ScoringConfig | None = None,
    ) -> None:
        self.catalog = CatalogIndex(catalog_path)
        self.scorer = ProductScorer(self.catalog, scoring_config)
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(user_profile=dict(user_profile))

    def close(self) -> None:
        self.catalog.close()

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
        ranked = self._prioritize_exact_category(
            self.scorer.score(pool, context),
            state.category,
        )
        unseen_ranked = [
            item
            for item in ranked
            if item.product.parent_asin not in state.shown_product_ids
        ]
        ask_attribute = self._select_question(state, ranked[:QUESTION_CANDIDATE_LIMIT])
        state.last_asked_attribute = ask_attribute
        if ask_attribute:
            state.asked_attributes.add(ask_attribute)
            state.follow_up_attributes.discard(ask_attribute)
            state.question_counts[ask_attribute] += 1

        recommendations = unseen_ranked[:top_k]
        state.shown_product_ids.update(
            item.product.parent_asin for item in recommendations
        )

        return {
            "message": (
                QUESTION_TEMPLATES[ask_attribute]
                if ask_attribute
                else "Here are the closest matches based on what you have told me."
            ),
            "ask_attribute": ask_attribute,
            "recommendations": [
                {
                    "parent_asin": item.product.parent_asin,
                    "score": round(item.score, 6),
                }
                for item in recommendations
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    @staticmethod
    def _prioritize_exact_category(
        ranked: list[ScoredProduct],
        category: str,
    ) -> list[ScoredProduct]:
        if not normalized_text(category).startswith(MALFORMED_CATEGORY_PREFIX):
            return ranked
        exact = [item for item in ranked if item.components["category"] == 1.0]
        partial = [item for item in ranked if item.components["category"] != 1.0]
        return [*exact, *partial]

    def _update_state(self, state: SessionState, message: str, turn: int) -> None:
        if turn == 1:
            self._parse_initial_message(state, message)
            return

        override = OVERRIDE_RE.match(message)
        if override:
            if state.last_asked_attribute:
                state.asked_attributes.discard(state.last_asked_attribute)
                state.follow_up_attributes.discard(state.last_asked_attribute)
            if state.overridable_preference:
                old_value = normalized_text(state.overridable_preference)
                state.constraints = [
                    constraint
                    for constraint in state.constraints
                    if normalized_text(constraint.text) != old_value
                ]
            state.overridable_preference = None
            state.intent = "buying"
            state.shown_product_ids.clear()
            self._add_constraint(state, override.group(1), hard=True, source="override")
            return

        clarification = CLARIFICATION_RE.match(message)
        if clarification:
            for value in clarification.group(1).split(";"):
                self._add_constraint(
                    state,
                    value,
                    hard=state.intent == "buying",
                    source="clarification",
                )
            if (
                state.last_asked_attribute
                and state.question_counts[state.last_asked_attribute]
                < MAX_QUESTIONS_PER_ATTRIBUTE
            ):
                state.follow_up_attributes.add(state.last_asked_attribute)
            return

        if message.startswith("I don't have") and state.last_asked_attribute:
            state.no_preference_attributes.add(state.last_asked_attribute)
            state.follow_up_attributes.discard(state.last_asked_attribute)

    def _parse_initial_message(self, state: SessionState, message: str) -> None:
        buying = BUYING_INITIAL_RE.match(message)
        if buying:
            state.intent = "buying"
            state.category = buying.group(1).strip()
            self._add_constraint(state, buying.group(2), hard=True, source="initial")
            return

        browsing = BROWSING_INITIAL_RE.match(message)
        if browsing:
            state.intent = "browsing"
            state.category = browsing.group(1).strip()
            return

        preference = PREFERENCE_INITIAL_RE.match(message)
        if preference:
            state.intent = "buying"
            state.category = preference.group(1).strip()
            state.overridable_preference = preference.group(2).strip()
            self._add_constraint(
                state,
                state.overridable_preference,
                hard=False,
                source="initial_preference",
            )
            return

        state.intent = "buying"
        state.category = message.strip()

    @staticmethod
    def _add_constraint(
        state: SessionState,
        text: str,
        hard: bool,
        source: str,
    ) -> None:
        cleaned = re.sub(r"\s+", " ", text).strip(" .")
        if not cleaned:
            return
        key = normalized_text(cleaned)
        for index, existing in enumerate(state.constraints):
            if normalized_text(existing.text) == key:
                if hard and not existing.hard:
                    state.constraints[index] = Constraint(existing.text, True, existing.source)
                return
        state.constraints.append(Constraint(cleaned, hard, source))

    @staticmethod
    def _budget_constraint(constraints: list[Constraint]) -> BudgetConstraint | None:
        budget: BudgetConstraint | None = None
        for constraint in constraints:
            for pattern in BUDGET_PATTERNS:
                match = pattern.search(constraint.text)
                if match:
                    budget = BudgetConstraint(float(match.group(1)), constraint.hard)
                    break
        return budget

    def _select_question(
        self,
        state: SessionState,
        candidates: list[ScoredProduct],
    ) -> str | None:
        profile_attributes = self._profile_attributes(state.user_profile)
        attribute_weights = CATEGORY_ATTRIBUTE_WEIGHTS[
            self.catalog.question_category(state.category)
        ]
        fresh_attributes = [
            attribute
            for attribute in QUESTION_ATTRIBUTES
            if attribute not in state.asked_attributes
            and attribute not in state.no_preference_attributes
        ]
        follow_up_attributes = [
            attribute
            for attribute in QUESTION_ATTRIBUTES
            if attribute in state.follow_up_attributes
            and attribute not in state.no_preference_attributes
        ]

        for attributes in (fresh_attributes, follow_up_attributes):
            scores: dict[str, float] = {}
            for attribute in attributes:
                values = [
                    value
                    for candidate in candidates
                    if (value := candidate.product.attribute_value(attribute)) is not None
                ]
                if len(values) < MIN_ATTRIBUTE_PRODUCTS:
                    continue
                information_gain = self._information_gain(values, len(candidates))
                profile_relevance = float(attribute in profile_attributes)
                scores[attribute] = (
                    0.65 * information_gain
                    + 0.30 * attribute_weights[attribute]
                    + 0.05 * profile_relevance
                )
            if scores:
                return max(
                    scores,
                    key=lambda attribute: (
                        scores[attribute],
                        -QUESTION_ATTRIBUTES.index(attribute),
                    ),
                )
        return None

    @staticmethod
    def _normalized_entropy(values: list[str]) -> float:
        counts = Counter(normalized_text(value) for value in values)
        if len(counts) <= 1:
            return 0.0
        total = sum(counts.values())
        entropy = -sum(
            (count / total) * math.log(count / total) for count in counts.values()
        )
        return entropy / math.log(len(counts))

    @classmethod
    def _information_gain(cls, values: list[str], candidate_count: int) -> float:
        coverage = len(values) / candidate_count
        return coverage * cls._normalized_entropy(values)

    @staticmethod
    def _profile_attributes(profile: dict) -> set[str]:
        attributes: set[str] = set()
        for tag in profile.get("preference_tags") or []:
            attributes.update(PROFILE_ATTRIBUTE_MAP.get(normalized_text(tag), set()))
        return attributes


# TODO: Add offline LLM attribute extraction so sparse free-text features can
# participate in categorical information-gain calculations.
