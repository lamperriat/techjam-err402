from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .v1 import Constraint
from .v2 import AgentV2, QUESTION_CANDIDATE_LIMIT, QuestionFacet, V2SessionState
from ..retrieval.catalog import CandidatePool, extract_department, normalized_text
from ..retrieval.scoring import QueryContext, ScoredProduct
from ..utils.llm_client import LLMClient, TokenUsage


V3Mode = Literal["benchmark", "interactive"]
CATALOG_GROUPS = frozenset({"clothing", "shoes", "jewelry", "other", "unknown"})
PROFILE_TAGS = frozenset(
    {
        "material",
        "style",
        "fit",
        "weather",
        "warmth",
        "performance",
        "comfort",
        "durability",
        "feature",
    }
)
PARSER_KEYS = frozenset(
    {
        "browsing_probability",
        "end_conversation",
        "category_action",
        "category_query",
        "catalog_group",
        "constraints_to_add",
        "constraint_ids_to_remove",
        "no_preference_facets",
        "profile",
    }
)
MAX_CONSTRAINT_LENGTH = 160
MAX_MESSAGE_LENGTH = 500
DISABLE_THINKING = {"thinking": {"type": "disabled"}}


@dataclass(frozen=True)
class ParsedConstraint:
    text: str
    hard: bool


@dataclass(frozen=True)
class ParsedTurn:
    browsing_probability: float
    end_conversation: bool
    category_action: Literal["keep", "replace", "clear"]
    category_query: str | None
    catalog_group: str
    constraints_to_add: tuple[ParsedConstraint, ...]
    constraint_ids_to_remove: tuple[str, ...]
    no_preference_facets: tuple[str, ...]
    profile: dict[str, object]


@dataclass
class V3SessionState(V2SessionState):
    browsing_probability: float = 0.5
    catalog_group: str = "unknown"
    next_constraint_id: int = 1
    last_asked_facets: tuple[str, ...] = field(default_factory=tuple)


class AgentV3(AgentV2):
    """V2 retrieval with LLM-based state parsing and question wording."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        attributes_path: str | Path = "results/catalog_attributes_processed.jsonl",
        mode: V3Mode = "benchmark",
        llm_client: LLMClient | None = None,
    ) -> None:
        if mode not in ("benchmark", "interactive"):
            raise ValueError("mode must be 'benchmark' or 'interactive'")
        super().__init__(
            catalog_path,
            attributes_path,
            question_mode="benchmark" if mode == "benchmark" else "native",
        )
        self.mode = mode
        self.llm_client = llm_client or LLMClient()

    @property
    def token_usage(self) -> TokenUsage:
        """Cumulative provider-reported usage for this agent instance."""
        return self.llm_client.total_usage

    def reset(self, session_id: str, user_profile: dict) -> None:
        profile = dict(user_profile)
        profile.setdefault("summary", "")
        profile.setdefault("preference_tags", [])
        self._sessions[session_id] = V3SessionState(user_profile=profile)

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
        if not isinstance(state, V3SessionState):
            raise RuntimeError("V3 session state was not initialized correctly")

        parsed_turn = self._parse_turn(state, user_message)
        if parsed_turn.end_conversation:
            return {
                "message": "Thanks for shopping with me. Goodbye!",
                "ask_attribute": None,
                "ask_attributes": [],
                "recommendations": [],
                "end_conversation": True,
                "usage": self.llm_client.consume_usage().as_dict(),
            }

        next_state = copy.deepcopy(state)
        self._apply_turn(next_state, parsed_turn)
        if next_state.catalog_group == "unknown" or not next_state.category:
            facets = ("product_type",)
            message = self._generate_question(next_state, user_message, facets)
            self._record_question(next_state, (), facets)
            self._sessions[session_id] = next_state
            return {
                "message": message,
                "ask_attribute": None,
                "ask_attributes": list(facets),
                "recommendations": [],
                "end_conversation": False,
                "usage": self.llm_client.consume_usage().as_dict(),
            }

        ranked = self._rank_products(next_state)
        unseen_ranked = [
            item
            for item in ranked
            if item.product.parent_asin not in next_state.shown_product_ids
        ]
        facets = self._select_question_facets(
            next_state,
            ranked[:QUESTION_CANDIDATE_LIMIT],
        )
        public_attributes = tuple(
            facet.benchmark_attribute for facet in facets
        )
        if facets:
            message = self._generate_question(
                next_state,
                user_message,
                tuple(facet.facet for facet in facets),
            )
        else:
            message = "Here are the closest matches based on what you have told me."
        self._record_question(
            next_state,
            facets,
            tuple(facet.facet for facet in facets),
        )

        recommendations = unseen_ranked[:top_k]
        next_state.shown_product_ids.update(
            item.product.parent_asin for item in recommendations
        )
        self._sessions[session_id] = next_state
        return {
            "message": message,
            "ask_attribute": public_attributes[0] if self.mode == "benchmark" and facets else None,
            "ask_attributes": list(
                public_attributes if self.mode == "benchmark" else tuple(facet.facet for facet in facets)
            ),
            "recommendations": [self._recommendation_payload(item) for item in recommendations],
            "end_conversation": False,
            "usage": self.llm_client.consume_usage().as_dict(),
        }

    def _parse_turn(self, state: V3SessionState, user_message: str) -> ParsedTurn:
        payload = self.llm_client.generate_json(
            self._parser_messages(state, user_message),
            temperature=0,
            max_tokens=1000,
            extra_body=DISABLE_THINKING,
        )
        return self._validate_parsed_turn(payload, state)

    @staticmethod
    def _parser_messages(
        state: V3SessionState,
        user_message: str,
    ) -> list[dict[str, str]]:
        current_state = {
            "category_query": state.category or None,
            "catalog_group": state.catalog_group,
            "constraints": [
                {
                    "id": constraint.constraint_id,
                    "text": constraint.text,
                    "hard": constraint.hard,
                }
                for constraint in state.constraints
            ],
            "profile": state.user_profile,
            "last_asked_facets": list(state.last_asked_facets),
        }
        system = """You parse one shopping message into a state update. Return only a JSON object.
You do not retrieve products, rank products, choose retrieval weights, or choose question facets.
Use browsing_probability from 0.0 (definite purchase intent) to 1.0 (open-ended browsing).
Set end_conversation true only for an explicit farewell or request to end.
category_action is keep, replace, or clear. Use replace only when the customer states
a product type; category_query must then be a concise product-category phrase grounded
in the customer's words. catalog_group must be one of clothing, shoes, jewelry, other,
or unknown. Use unknown and clear when no product type is known.
When category_action is keep, set category_query to null and repeat the current
catalog_group.
Add only concise, customer-stated shopping constraints. Mark a constraint hard only for
an explicit requirement, prohibition, or firm budget. Remove an existing constraint only
when the customer explicitly changes or rejects it, using an ID from current_state.
no_preference_facets may contain only facets listed in last_asked_facets.
Replace profile with a concise current summary and zero or more tags from: material,
style, fit, weather, warmth, performance, comfort, durability, feature.
constraints_to_add must be an array of objects with exactly these keys: text and hard.
For example: {"text": "24K gold", "hard": false}. Do not use type, attribute,
or value keys for a constraint.
Return exactly these keys: browsing_probability, end_conversation, category_action,
category_query, catalog_group, constraints_to_add, constraint_ids_to_remove,
no_preference_facets, profile."""
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {"current_state": current_state, "user_message": user_message},
                    ensure_ascii=False,
                ),
            },
        ]

    @classmethod
    def _validate_parsed_turn(
        cls,
        payload: dict[str, Any],
        state: V3SessionState,
    ) -> ParsedTurn:
        if set(payload) != PARSER_KEYS:
            raise ValueError("LLM parser response has an unexpected schema")
        probability = payload["browsing_probability"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("browsing_probability must be a number")
        browsing_probability = float(probability)
        if not math.isfinite(browsing_probability) or not 0.0 <= browsing_probability <= 1.0:
            raise ValueError("browsing_probability must be in [0, 1]")
        end_conversation = payload["end_conversation"]
        if not isinstance(end_conversation, bool):
            raise ValueError("end_conversation must be a boolean")

        category_action = payload["category_action"]
        if category_action not in {"keep", "replace", "clear"}:
            raise ValueError("category_action must be keep, replace, or clear")
        category_query = payload["category_query"]
        if category_action == "replace":
            category_query = cls._validated_text(
                category_query,
                "category_query",
                MAX_CONSTRAINT_LENGTH,
            )
        elif category_action == "keep" and category_query is not None:
            repeated_category = cls._validated_text(
                category_query,
                "category_query",
                MAX_CONSTRAINT_LENGTH,
            )
            if normalized_text(repeated_category) != normalized_text(state.category):
                raise ValueError("a kept category must match the current category")
            category_query = None
        elif category_query is not None:
            raise ValueError("category_query is only allowed when replacing a category")
        catalog_group = payload["catalog_group"]
        if catalog_group not in CATALOG_GROUPS:
            raise ValueError("catalog_group is invalid")
        if category_action == "replace" and catalog_group == "unknown":
            raise ValueError("a replacement category cannot have an unknown group")
        if category_action == "clear" and catalog_group != "unknown":
            raise ValueError("a cleared category must have an unknown group")
        if category_action == "keep" and catalog_group != state.catalog_group:
            raise ValueError("a kept category must preserve its catalog group")

        active_ids = {
            constraint.constraint_id
            for constraint in state.constraints
            if constraint.constraint_id is not None
        }
        removal_ids = cls._validated_string_list(
            payload["constraint_ids_to_remove"],
            "constraint_ids_to_remove",
        )
        if len(set(removal_ids)) != len(removal_ids) or not set(removal_ids) <= active_ids:
            raise ValueError("constraint_ids_to_remove contains an unknown ID")
        no_preference_facets = cls._validated_string_list(
            payload["no_preference_facets"],
            "no_preference_facets",
        )
        if (
            len(set(no_preference_facets)) != len(no_preference_facets)
            or not set(no_preference_facets) <= set(state.last_asked_facets)
        ):
            raise ValueError("no_preference_facets must be facets from the last question")

        additions = cls._validated_constraints(payload["constraints_to_add"])
        profile = cls._validated_profile(payload["profile"])
        return ParsedTurn(
            browsing_probability=browsing_probability,
            end_conversation=end_conversation,
            category_action=category_action,
            category_query=category_query,
            catalog_group=catalog_group,
            constraints_to_add=additions,
            constraint_ids_to_remove=tuple(removal_ids),
            no_preference_facets=tuple(no_preference_facets),
            profile=profile,
        )

    @classmethod
    def _validated_constraints(cls, value: object) -> tuple[ParsedConstraint, ...]:
        if not isinstance(value, list):
            raise ValueError("constraints_to_add must be a list")
        constraints: list[ParsedConstraint] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("each constraint must contain text and hard")
            if set(item) == {"text", "hard"}:
                text_value = item["text"]
            elif set(item) == {"type", "value", "hard"}:
                text_value = item["value"]
            else:
                raise ValueError("each constraint must contain text and hard")
            text = cls._validated_text(text_value, "constraint text", MAX_CONSTRAINT_LENGTH)
            hard = item["hard"]
            if not isinstance(hard, bool):
                raise ValueError("constraint hard must be a boolean")
            normalized = normalized_text(text)
            if normalized in seen:
                raise ValueError("constraints_to_add contains duplicate constraints")
            seen.add(normalized)
            constraints.append(ParsedConstraint(text, hard))
        return tuple(constraints)

    @classmethod
    def _validated_profile(cls, value: object) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != {"summary", "preference_tags"}:
            raise ValueError("profile must contain summary and preference_tags")
        summary = cls._validated_text(value["summary"], "profile summary", MAX_MESSAGE_LENGTH)
        tags = cls._validated_string_list(value["preference_tags"], "preference_tags")
        if len(set(tags)) != len(tags) or not set(tags) <= PROFILE_TAGS:
            raise ValueError("profile contains an invalid preference tag")
        return {"summary": summary, "preference_tags": tags}

    @staticmethod
    def _validated_string_list(value: object, name: str) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{name} must be a list of strings")
        return value

    @staticmethod
    def _validated_text(value: object, name: str, limit: int) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        cleaned = re.sub(r"\s+", " ", value).strip(" .")
        if not cleaned or len(cleaned) > limit:
            raise ValueError(f"{name} must be non-empty and at most {limit} characters")
        return cleaned

    @staticmethod
    def _apply_turn(state: V3SessionState, parsed_turn: ParsedTurn) -> None:
        state_changed = False
        state.browsing_probability = parsed_turn.browsing_probability
        state.intent = "browsing" if parsed_turn.browsing_probability >= 0.5 else "buying"
        state.user_profile = parsed_turn.profile
        if parsed_turn.category_action == "replace":
            state.category = parsed_turn.category_query or ""
            state.catalog_group = parsed_turn.catalog_group
            state.asked_attributes.clear()
            state.asked_public_attributes.clear()
            state.no_preference_attributes.clear()
            state.question_counts.clear()
            state.last_asked_facets = ()
            state_changed = True
        elif parsed_turn.category_action == "clear":
            state.category = ""
            state.catalog_group = "unknown"
            state.asked_attributes.clear()
            state.asked_public_attributes.clear()
            state.no_preference_attributes.clear()
            state.question_counts.clear()
            state.last_asked_facets = ()
            state_changed = True

        if parsed_turn.constraint_ids_to_remove:
            removal_ids = set(parsed_turn.constraint_ids_to_remove)
            state.constraints = [
                constraint
                for constraint in state.constraints
                if constraint.constraint_id not in removal_ids
            ]
            state_changed = True
        existing = {normalized_text(constraint.text) for constraint in state.constraints}
        for constraint in parsed_turn.constraints_to_add:
            if normalized_text(constraint.text) in existing:
                continue
            state.constraints.append(
                Constraint(
                    constraint.text,
                    constraint.hard,
                    "llm",
                    f"c{state.next_constraint_id}",
                )
            )
            state.next_constraint_id += 1
            existing.add(normalized_text(constraint.text))
            state_changed = True
        state.no_preference_attributes.update(parsed_turn.no_preference_facets)
        state.follow_up_attributes.clear()
        if state_changed:
            state.shown_product_ids.clear()

    def _score_pool(
        self,
        state: V3SessionState,
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
            browsing_probability=state.browsing_probability,
        )
        return self.scorer.score(pool, context)

    def _select_question_facets(
        self,
        state: V3SessionState,
        candidates: list[ScoredProduct],
    ) -> tuple[QuestionFacet, ...]:
        first = self._select_fine_question(state, candidates)
        if first is None or self.mode == "benchmark":
            return () if first is None else (first,)
        state.asked_attributes.add(first.facet)
        try:
            second = self._select_fine_question(state, candidates)
        finally:
            state.asked_attributes.discard(first.facet)
        return (first,) if second is None else (first, second)

    def _generate_question(
        self,
        state: V3SessionState,
        user_message: str,
        facets: tuple[str, ...],
    ) -> str:
        payload = self.llm_client.generate_json(
            [
                {
                    "role": "system",
                    "content": """Write one concise, friendly shopping follow-up question. Return only
a JSON object with exactly one key, message. Ask only about every supplied selected_facet
and do not introduce another attribute, product, recommendation, or answer. When the
facet is product_type, ask what type of product the customer wants.""",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "selected_facets": list(facets),
                            "profile": state.user_profile,
                            "latest_user_message": user_message,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0,
            max_tokens=180,
            extra_body=DISABLE_THINKING,
        )
        if set(payload) != {"message"}:
            raise ValueError("LLM question response has an unexpected schema")
        return self._validated_text(payload["message"], "question message", MAX_MESSAGE_LENGTH)

    def _record_question(
        self,
        state: V3SessionState,
        facets: tuple[QuestionFacet, ...],
        facet_names: tuple[str, ...],
    ) -> None:
        state.last_asked_facets = facet_names
        state.last_asked_attribute = facet_names[0] if facet_names else None
        state.last_asked_public_attribute = (
            facets[0].benchmark_attribute if self.mode == "benchmark" and facets else None
        )
        for facet in facets:
            state.asked_attributes.add(facet.facet)
            state.question_counts[facet.facet] += 1
            if self.mode == "benchmark":
                state.asked_public_attributes.add(facet.benchmark_attribute)

    @staticmethod
    def _recommendation_payload(item: ScoredProduct) -> dict[str, object]:
        return {
            "parent_asin": item.product.parent_asin,
            "title": item.product.title,
            "price": item.product.price,
            "price_is_lower_bound": item.product.price_is_lower_bound,
            "rating_number": item.product.rating_number,
            "average_rating": item.product.average_rating,
            "description": item.product.description,
            "score": round(item.score, 6),
        }
