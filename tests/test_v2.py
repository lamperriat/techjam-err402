from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.v1 import Constraint
from agents.v2 import AgentV2, QuestionFacet, V2SessionState
from retrieval.attributes import ExtractedAttributeIndex


def product(index: int) -> dict:
    return {
        "parent_asin": f"P{index}",
        "title": f"Running shoe {index}",
        "features": [f"Foam technology {index % 2}"],
        "description": [],
        "price": 10.0 + index * 10,
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


class AgentV2Test(unittest.TestCase):
    def _paths(self, directory: str, count: int = 6) -> tuple[Path, Path]:
        root = Path(directory)
        catalog_path = root / "catalog.jsonl"
        attributes_path = root / "attributes.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product(index)) + "\n" for index in range(count)),
            encoding="utf-8",
        )
        attributes_path.write_text(
            json.dumps({"record_type": "metadata", "postprocessing": {}})
            + "\n"
            + "".join(json.dumps(attributes(index)) + "\n" for index in range(count)),
            encoding="utf-8",
        )
        return catalog_path, attributes_path

    def test_attribute_index_exposes_primary_core_and_specific_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, attributes_path = self._paths(directory, count=1)
            record = attributes(0)
            record["attributes"]["material"] = [
                {"value": "Leather", "evidence": "Leather"},
                {"value": "Rubber", "evidence": "Rubber"},
            ]
            attributes_path.write_text(
                json.dumps({"record_type": "metadata"})
                + "\n"
                + json.dumps(record)
                + "\n",
                encoding="utf-8",
            )

            index = ExtractedAttributeIndex(attributes_path)

            self.assertEqual(index.core_value("P0", "material"), "Leather")
            self.assertEqual(
                index.specific_value("P0", "cushioning"),
                "Foam technology 0",
            )

    def test_question_utility_rejects_singleton_only_values(self) -> None:
        utility = AgentV2._question_utility(
            ["one", "two", "three", "four", "five"],
            candidate_count=5,
        )

        self.assertIsNone(utility)

    def test_question_utility_uses_recurring_value_answerability(self) -> None:
        utility = AgentV2._question_utility(
            ["red", "red", "blue", "blue", "singleton"],
            candidate_count=10,
        )

        self.assertIsNotNone(utility)
        assert utility is not None
        self.assertAlmostEqual(utility.expected_answerability, 0.4)
        self.assertAlmostEqual(utility.discrimination, 0.4)
        self.assertEqual(utility.recurring_value_count, 2)

    def test_question_utility_penalizes_too_many_answer_choices(self) -> None:
        values = [value for index in range(18) for value in (f"v{index}", f"v{index}")]

        utility = AgentV2._question_utility(values, candidate_count=len(values))

        self.assertIsNotNone(utility)
        assert utility is not None
        self.assertLess(utility.cardinality_penalty, 1.0)
        self.assertAlmostEqual(utility.cardinality_penalty, (8 / 18) ** 0.5)

    def test_disclosed_constraint_excludes_matching_facet(self) -> None:
        state = V2SessionState(
            user_profile={},
            constraints=[Constraint("genuine leather", True, "initial")],
        )

        disclosed = AgentV2._facet_is_disclosed(
            state,
            QuestionFacet("material", "material", "core", "material"),
            ["Leather", "Cotton"],
        )
        unrelated = AgentV2._facet_is_disclosed(
            state,
            QuestionFacet("closure", "style", "specific", "feature"),
            ["Zipper", "Pull On"],
        )

        self.assertTrue(disclosed)
        self.assertFalse(unrelated)

    def test_category_novelty_penalizes_implied_recurring_values(self) -> None:
        novelty = AgentV2._category_novelty(
            ["Water Shoes", "Water Shoes", "Walking", "Walking"],
            frozenset({"water shoes", "walking"}),
            "Athletic Water Shoes",
        )

        self.assertEqual(novelty, 0.5)

    def test_malformed_category_prioritizes_exact_matches(self) -> None:
        partial = type("Scored", (), {"components": {"category": 0.5}})()
        exact = type("Scored", (), {"components": {"category": 1.0}})()

        ranked = AgentV2._prioritize_exact_category(
            [partial, exact],
            "Shoes & Jewelry Women",
        )

        self.assertEqual(ranked, [exact, partial])

    def test_v2_ranking_applies_malformed_category_tier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory)
            agent = AgentV2(catalog_path, attributes_path)
            self.addCleanup(agent.close)
            with patch.object(
                agent,
                "_prioritize_exact_category",
                wraps=agent._prioritize_exact_category,
            ) as prioritize:
                agent._rank_products(
                    V2SessionState(
                        user_profile={},
                        category="Shoes & Jewelry Women",
                    )
                )

        prioritize.assert_called_once()
        self.assertEqual(prioritize.call_args.args[1], "Shoes & Jewelry Women")

    def test_benchmark_mode_asks_fine_question_but_maps_attribute_to_feature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory)
            agent = AgentV2(catalog_path, attributes_path, question_mode="benchmark")
            self.addCleanup(agent.close)
            agent.reset("session", {})

            first = agent.respond(
                "session",
                "I'm looking for Women Shoes, but I'm still exploring.",
                turn=1,
                top_k=10,
            )
            second = agent.respond(
                "session",
                "I don't have an additional preference for feature.",
                turn=2,
                top_k=10,
            )

            self.assertEqual(first["ask_attribute"], "feature")
            self.assertIn("cushioning", first["message"])
            self.assertNotEqual(second["ask_attribute"], "feature")
            self.assertIn("cushioning", agent._sessions["session"].asked_attributes)

    def test_positive_answer_allows_one_fine_facet_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory)
            agent = AgentV2(catalog_path, attributes_path, question_mode="benchmark")
            self.addCleanup(agent.close)
            agent.reset("session", {})

            first = agent.respond(
                "session",
                "I'm looking for Women Shoes, but I'm still exploring.",
                turn=1,
                top_k=2,
            )
            fresh_question = agent.respond(
                "session",
                "For that, what matters is: Foam technology 0; Foam technology 1.",
                turn=2,
                top_k=2,
            )
            follow_up = agent.respond(
                "session",
                "I don't have an additional preference for budget.",
                turn=3,
                top_k=2,
            )
            exhausted = agent.respond(
                "session",
                "For that, what matters is: Foam technology 0.",
                turn=4,
                top_k=2,
            )
            state = agent._sessions["session"]

        self.assertEqual(first["ask_attribute"], "feature")
        self.assertEqual(fresh_question["ask_attribute"], "budget")
        self.assertEqual(follow_up["ask_attribute"], "feature")
        self.assertIsNone(exhausted["ask_attribute"])
        self.assertEqual(state.question_counts["cushioning"], 2)

    def test_v2_recommendations_are_not_repeated_across_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory)
            agent = AgentV2(catalog_path, attributes_path, question_mode="benchmark")
            self.addCleanup(agent.close)
            agent.reset("session", {})

            first = agent.respond(
                "session",
                "I'm looking for Women Shoes, but I'm still exploring.",
                turn=1,
                top_k=2,
            )
            second = agent.respond(
                "session",
                "I don't have an additional preference for feature.",
                turn=2,
                top_k=2,
            )

        first_ids = {item["parent_asin"] for item in first["recommendations"]}
        second_ids = {item["parent_asin"] for item in second["recommendations"]}
        self.assertFalse(first_ids & second_ids)

    def test_override_reopens_interrupted_public_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory)
            agent = AgentV2(catalog_path, attributes_path, question_mode="benchmark")
            self.addCleanup(agent.close)
            agent.reset("session", {})
            state = agent._sessions["session"]
            state.asked_attributes.add("cushioning")
            state.asked_public_attributes.add("feature")
            state.last_asked_attribute = "cushioning"
            state.last_asked_public_attribute = "feature"

            agent._update_state(
                state,
                "Actually, ignore my earlier preference. What I need is: nylon.",
                turn=3,
            )

        self.assertNotIn("cushioning", state.asked_attributes)
        self.assertNotIn("feature", state.asked_public_attributes)

    def test_native_mode_exposes_the_fine_grained_facet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory)
            agent = AgentV2(catalog_path, attributes_path, question_mode="native")
            self.addCleanup(agent.close)
            agent.reset("session", {})

            response = agent.respond(
                "session",
                "I'm looking for Women Shoes, but I'm still exploring.",
                turn=1,
                top_k=10,
            )

            self.assertEqual(response["ask_attribute"], "cushioning")
            self.assertIn("cushioning", response["message"])

    def test_requires_processed_attributes_for_every_catalog_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, attributes_path = self._paths(directory, count=2)
            lines = attributes_path.read_text(encoding="utf-8").splitlines()
            attributes_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "product IDs do not match"):
                AgentV2(catalog_path, attributes_path)


if __name__ == "__main__":
    unittest.main()
