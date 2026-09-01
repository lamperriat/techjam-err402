from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.v1 import Constraint
from agents.v2 import QuestionFacet
from agents.v3 import AgentV3, ParsedConstraint, V3SessionState
from retrieval.scoring import ProductScorer, QueryContext, ScoringConfig
from utils.llm_client import TokenUsage


def product(index: int) -> dict:
    return {
        "parent_asin": f"P{index}",
        "title": f"Running shoe {index}",
        "features": [f"Foam technology {index % 2}"],
        "description": [f"Comfortable running shoe number {index}"],
        "price": 20.0 + index,
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes"],
        "details": {"Department": "womens"},
        "average_rating": 4.2,
        "rating_number": 100,
        "store": "Example",
    }


def attributes(index: int) -> dict:
    return {
        "parent_asin": f"P{index}",
        "attributes": {
            "material": [],
            "color": [],
            "size_fit": [],
            "style": [],
            "use_case": [],
            "specific_attributes": [
                {
                    "name": "cushioning",
                    "value": f"Foam technology {index % 2}",
                    "evidence": f"Foam technology {index % 2}",
                }
            ],
        },
    }


def parser_response(**overrides: object) -> dict:
    response = {
        "browsing_probability": 0.2,
        "end_conversation": False,
        "category_action": "replace",
        "category_query": "women's running shoes",
        "catalog_group": "shoes",
        "constraints_to_add": [{"text": "wide fit", "hard": True}],
        "constraint_ids_to_remove": [],
        "no_preference_facets": [],
        "profile": {
            "summary": "Needs comfortable wide running shoes.",
            "preference_tags": ["fit", "comfort"],
        },
    }
    response.update(overrides)
    return response


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []
        self.controls: list[dict[str, object]] = []
        self.total_usage = TokenUsage()
        self._unreported_usage = TokenUsage()

    def generate_json(self, messages: list[dict[str, str]], **controls: object) -> dict:
        self.calls.append(messages)
        self.controls.append(controls)
        usage = TokenUsage(10, 2)
        self.total_usage += usage
        self._unreported_usage += usage
        return self.responses.pop(0)

    def consume_usage(self) -> TokenUsage:
        usage = self._unreported_usage
        self._unreported_usage = TokenUsage()
        return usage


class AgentV3Test(unittest.TestCase):
    def _paths(self, directory: str, count: int = 6) -> tuple[Path, Path]:
        root = Path(directory)
        catalog_path = root / "catalog.jsonl"
        attributes_path = root / "attributes.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product(index)) + "\n" for index in range(count)),
            encoding="utf-8",
        )
        attributes_path.write_text(
            json.dumps({"record_type": "metadata"})
            + "\n"
            + "".join(json.dumps(attributes(index)) + "\n" for index in range(count)),
            encoding="utf-8",
        )
        return catalog_path, attributes_path

    def test_interactive_turn_parses_state_and_asks_two_facets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory)
            llm = FakeLLM([parser_response(), {"message": "Would you prefer more cushioning or arch support?"}])
            agent = AgentV3(catalog_path, attributes_path, mode="interactive", llm_client=llm)
            self.addCleanup(agent.close)
            agent.reset("session", {})
            facets = [
                QuestionFacet("cushioning", "comfort", "specific", "feature"),
                QuestionFacet("arch_support", "comfort", "specific", "feature"),
            ]

            with patch.object(agent, "_select_fine_question", side_effect=facets):
                response = agent.respond("session", "I need wide running shoes.", turn=1, top_k=2)

            state = agent._sessions["session"]

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.controls[0]["max_tokens"], 1000)
        self.assertEqual(llm.controls[1]["max_tokens"], 180)
        self.assertEqual(llm.controls[0]["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(llm.controls[1]["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(response["ask_attribute"], None)
        self.assertEqual(response["ask_attributes"], ["cushioning", "arch_support"])
        self.assertEqual(response["usage"], {"prompt_tokens": 20, "completion_tokens": 4})
        self.assertEqual(state.category, "women's running shoes")
        self.assertEqual(state.catalog_group, "shoes")
        self.assertEqual(state.constraints[0], Constraint("wide fit", True, "llm", "c1"))
        self.assertEqual(state.user_profile["preference_tags"], ["fit", "comfort"])
        self.assertIn(
            "Needs comfortable wide running shoes",
            AgentV3._parser_messages(state, "Do you have another option?")[1]["content"],
        )
        self.assertIn("title", response["recommendations"][0])
        self.assertIn("description", response["recommendations"][0])

    def test_benchmark_mode_exposes_one_compatible_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory)
            llm = FakeLLM([parser_response(), {"message": "How much cushioning do you need?"}])
            agent = AgentV3(catalog_path, attributes_path, mode="benchmark", llm_client=llm)
            self.addCleanup(agent.close)
            agent.reset("session", {})
            facet = QuestionFacet("cushioning", "comfort", "specific", "feature")

            with patch.object(agent, "_select_fine_question", return_value=facet):
                response = agent.respond("session", "I need wide running shoes.", turn=1, top_k=2)

        self.assertEqual(response["ask_attribute"], "feature")
        self.assertEqual(response["ask_attributes"], ["feature"])
        self.assertEqual(len(llm.calls), 2)

    def test_end_turn_skips_question_and_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory)
            llm = FakeLLM([parser_response(end_conversation=True)])
            agent = AgentV3(catalog_path, attributes_path, llm_client=llm)
            self.addCleanup(agent.close)
            agent.reset("session", {})

            response = agent.respond("session", "Thanks, goodbye.", turn=1, top_k=10)

        self.assertTrue(response["end_conversation"])
        self.assertEqual(response["recommendations"], [])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(response["usage"], {"prompt_tokens": 10, "completion_tokens": 2})

    def test_unknown_category_asks_for_product_type_without_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory)
            llm = FakeLLM(
                [
                    parser_response(
                        category_action="clear",
                        category_query=None,
                        catalog_group="unknown",
                        constraints_to_add=[],
                    ),
                    {"message": "What type of product are you shopping for?"},
                ]
            )
            agent = AgentV3(catalog_path, attributes_path, mode="interactive", llm_client=llm)
            self.addCleanup(agent.close)
            agent.reset("session", {})

            response = agent.respond("session", "I need a gift.", turn=1, top_k=10)

        self.assertEqual(response["ask_attributes"], ["product_type"])
        self.assertEqual(response["recommendations"], [])
        self.assertEqual(len(llm.calls), 2)

    def test_invalid_question_response_does_not_commit_the_parsed_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory)
            llm = FakeLLM([parser_response(), {"unexpected": "schema"}])
            agent = AgentV3(catalog_path, attributes_path, mode="interactive", llm_client=llm)
            self.addCleanup(agent.close)
            agent.reset("session", {})
            facets = [
                QuestionFacet("cushioning", "comfort", "specific", "feature"),
                None,
            ]

            with patch.object(agent, "_select_fine_question", side_effect=facets):
                with self.assertRaisesRegex(ValueError, "question response"):
                    agent.respond("session", "I need wide running shoes.", turn=1, top_k=2)

            state = agent._sessions["session"]

        self.assertEqual(state.category, "")
        self.assertEqual(state.constraints, [])

    def test_removal_uses_only_known_constraint_ids(self) -> None:
        state = V3SessionState(
            user_profile={"summary": "", "preference_tags": []},
            category="women's shoes",
            catalog_group="shoes",
            constraints=[Constraint("red", False, "llm", "c1")],
            next_constraint_id=2,
        )
        parsed = AgentV3._validate_parsed_turn(
            parser_response(
                category_action="keep",
                category_query=None,
                catalog_group="shoes",
                constraints_to_add=[],
                constraint_ids_to_remove=["c1"],
            ),
            state,
        )

        AgentV3._apply_turn(state, parsed)

        self.assertEqual(state.constraints, [])
        with self.assertRaisesRegex(ValueError, "unknown ID"):
            AgentV3._validate_parsed_turn(
                parser_response(
                    category_action="keep",
                    category_query=None,
                    catalog_group="shoes",
                    constraints_to_add=[],
                    constraint_ids_to_remove=["not-a-constraint"],
                ),
                state,
            )

    def test_keep_action_accepts_a_repeated_current_category(self) -> None:
        state = V3SessionState(
            user_profile={"summary": "", "preference_tags": []},
            category="gold necklace",
            catalog_group="jewelry",
        )

        parsed = AgentV3._validate_parsed_turn(
            parser_response(
                category_action="keep",
                category_query="gold necklace",
                catalog_group="jewelry",
                constraints_to_add=[],
            ),
            state,
        )

        self.assertIsNone(parsed.category_query)

    def test_normalizes_deepseek_type_value_constraints(self) -> None:
        state = V3SessionState(user_profile={"summary": "", "preference_tags": []})

        parsed = AgentV3._validate_parsed_turn(
            parser_response(
                constraints_to_add=[
                    {"type": "material", "value": "24K gold", "hard": False}
                ]
            ),
            state,
        )

        self.assertEqual(parsed.constraints_to_add, (ParsedConstraint("24K gold", False),))

    def test_blends_buying_and_browsing_component_weights(self) -> None:
        scorer = object.__new__(ProductScorer)
        scorer.config = ScoringConfig.default()

        weights = scorer._intent_weights(
            QueryContext("buying", "shoes", (), None, None, browsing_probability=0.25)
        )

        self.assertAlmostEqual(weights["lexical"], 0.275)
        self.assertAlmostEqual(weights["category"], 0.3125)


if __name__ == "__main__":
    unittest.main()
