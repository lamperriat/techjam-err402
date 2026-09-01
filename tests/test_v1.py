from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agents.registry import agent_names, get_agent_spec
from agents.v1 import AgentV1, QUESTION_ATTRIBUTES
from retrieval.catalog import (
    CandidatePool,
    CatalogIndex,
    category_group,
    fine_category_group,
    normalize_price,
)
from retrieval.scoring import ProductScorer, QueryContext


def product_row(index: int, category: str = "Shoes") -> dict:
    colors = ("black", "white", "blue", "red", "green", "yellow")
    materials = ("leather", "cotton", "nylon", "wool", "silk", "polyester")
    return {
        "parent_asin": f"P{index}",
        "title": f"{colors[index]} {materials[index]} running {category}",
        "features": [materials[index], "comfortable everyday feature"],
        "description": [f"A {colors[index]} option"],
        "price": (10.0, 20.0, 30.0, 40.0, 60.0, 120.0)[index],
        "categories": ["Clothing, Shoes & Jewelry", "Women", category],
        "details": {"Department": "womens", "Color": colors[index]},
        "average_rating": 4.2,
        "rating_number": 100,
        "store": "Example",
    }


class AgentV1Test(unittest.TestCase):
    def _catalog_path(self, directory: str, count: int = 6) -> Path:
        path = Path(directory) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(product_row(index)) + "\n" for index in range(count)),
            encoding="utf-8",
        )
        return path

    def test_price_normalization_distinguishes_lower_bound_and_missing(self) -> None:
        self.assertEqual(normalize_price(19.99), (19.99, False))
        self.assertEqual(normalize_price("from 12.99"), (12.99, True))
        self.assertEqual(normalize_price("—"), (None, False))

    def test_full_catalog_path_selects_question_prior_category(self) -> None:
        self.assertEqual(
            category_group(["Clothing, Shoes & Jewelry", "Women", "Shoes", "Athletic"]),
            "shoes",
        )
        self.assertEqual(
            category_group(["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Rings"]),
            "jewelry",
        )
        self.assertEqual(
            category_group(["Clothing, Shoes & Jewelry", "Women", "Accessories"]),
            "clothing",
        )

    def test_v2_category_router_separates_mixed_accessories(self) -> None:
        self.assertEqual(
            fine_category_group(
                [
                    "Clothing, Shoes & Jewelry",
                    "Jewelry & Watch Accessories",
                    "Shoe Care & Accessories",
                    "Shoelaces",
                ]
            ),
            "shoes",
        )
        self.assertEqual(
            fine_category_group(
                ["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Earrings"]
            ),
            "jewelry",
        )
        self.assertEqual(
            fine_category_group(
                ["Clothing, Shoes & Jewelry", "Women", "Watches", "Wrist Watches"]
            ),
            "other",
        )

    def test_question_category_uses_catalog_mapping_then_default_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            rows = [product_row(0, "Athletic Walking"), product_row(1, "Athletic Walking")]
            rows[0]["categories"] = [
                "Clothing, Shoes & Jewelry", "Shoes", "Athletic", "Walking",
            ]
            rows[1]["categories"] = [
                "Clothing, Shoes & Jewelry", "Shoes", "Athletic", "Walking",
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            catalog = CatalogIndex(path)
            self.addCleanup(catalog.close)

            self.assertEqual(catalog.question_category("Athletic Walking"), "shoes")
            self.assertEqual(catalog.question_category("running shoes"), "shoes")
            self.assertEqual(catalog.question_category("general wearable"), "clothing")

    def test_buying_category_weight_can_overcome_one_lexical_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            rows = [product_row(0, "Shoes"), product_row(1, "Hats")]
            rows[1]["categories"] = ["Clothing, Shoes & Jewelry", "Men", "Hats"]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            catalog = CatalogIndex(path)
            self.addCleanup(catalog.connection.close)
            pool = CandidatePool(("P0", "P1"), {"P1": 1, "P0": 2})
            context = QueryContext("buying", "Women Shoes", (), None, None)

            ranked = ProductScorer(catalog).score(pool, context)

        self.assertEqual(ranked[0].product.parent_asin, "P0")

    def test_malformed_category_prioritizes_exact_matches(self) -> None:
        partial = SimpleNamespace(components={"category": 0.5})
        exact = SimpleNamespace(components={"category": 1.0})

        ranked = AgentV1._prioritize_exact_category(
            [partial, exact],
            "Shoes & Jewelry Women",
        )

        self.assertEqual(ranked, [exact, partial])

    def test_normal_category_preserves_score_order(self) -> None:
        partial = SimpleNamespace(components={"category": 0.5})
        exact = SimpleNamespace(components={"category": 1.0})

        ranked = AgentV1._prioritize_exact_category(
            [partial, exact],
            "Women Dresses",
        )

        self.assertEqual(ranked, [partial, exact])

    def test_override_removes_only_initial_preference_and_keeps_question_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = AgentV1(self._catalog_path(directory))
            self.addCleanup(agent.catalog.connection.close)
            agent.reset("session", {"preference_tags": ["material"]})
            first = agent.respond(
                "session",
                "I'm looking for Women Shoes. Buckle closure",
                turn=1,
                top_k=10,
            )
            asked_before_override = set(agent._sessions["session"].asked_attributes)
            agent.respond(
                "session",
                "For that, what matters is: leather.",
                turn=2,
                top_k=2,
            )
            shown_before_override = set(agent._sessions["session"].shown_product_ids)
            override_response = agent.respond(
                "session",
                "Actually, ignore my earlier preference. What I need is: nylon.",
                turn=3,
                top_k=2,
            )
            state = agent._sessions["session"]

        self.assertIsNotNone(first["ask_attribute"])
        self.assertEqual(state.asked_attributes & asked_before_override, asked_before_override)
        self.assertEqual({item.text for item in state.constraints}, {"leather", "nylon"})
        self.assertEqual(state.intent, "buying")
        self.assertGreater(len(shown_before_override), len(state.shown_product_ids))
        self.assertEqual(
            state.shown_product_ids,
            {item["parent_asin"] for item in override_response["recommendations"]},
        )

    def test_recommendations_are_not_repeated_across_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = AgentV1(self._catalog_path(directory))
            self.addCleanup(agent.catalog.connection.close)
            agent.reset("session", {})
            first = agent.respond(
                "session",
                "I'm looking for Women Shoes, but I'm still exploring.",
                turn=1,
                top_k=2,
            )
            second = agent.respond(
                "session",
                f"I don't have an additional preference for {first['ask_attribute']}.",
                turn=2,
                top_k=2,
            )
            state = agent._sessions["session"]

        first_ids = {item["parent_asin"] for item in first["recommendations"]}
        second_ids = {item["parent_asin"] for item in second["recommendations"]}
        self.assertFalse(first_ids & second_ids)
        self.assertEqual(state.shown_product_ids, first_ids | second_ids)

    def test_question_requires_five_products_with_the_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = AgentV1(self._catalog_path(directory, count=4))
            self.addCleanup(agent.catalog.connection.close)
            agent.reset("session", {})

            response = agent.respond(
                "session",
                "I'm looking for Women Shoes, but I'm still exploring.",
                turn=1,
                top_k=10,
            )

        self.assertIsNone(response["ask_attribute"])

    def test_question_information_gain_accounts_for_attribute_coverage(self) -> None:
        values = ["red", "blue", "green", "black", "white"]

        information_gain = AgentV1._information_gain(values, candidate_count=10)

        self.assertAlmostEqual(information_gain, 0.5)

    def test_question_policy_excludes_brand(self) -> None:
        self.assertNotIn("brand", QUESTION_ATTRIBUTES)

    def test_question_attributes_are_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = AgentV1(self._catalog_path(directory))
            self.addCleanup(agent.catalog.connection.close)
            agent.reset("session", {})
            first = agent.respond(
                "session",
                "I'm looking for Women Shoes, but I'm still exploring.",
                turn=1,
                top_k=10,
            )
            second = agent.respond(
                "session",
                f"I don't have an additional preference for {first['ask_attribute']}.",
                turn=2,
                top_k=10,
            )

        self.assertIsNotNone(first["ask_attribute"])
        self.assertNotEqual(first["ask_attribute"], second["ask_attribute"])

    def test_positive_answer_allows_one_later_attribute_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = AgentV1(self._catalog_path(directory))
            self.addCleanup(agent.catalog.connection.close)
            agent.reset("session", {})
            first = agent.respond(
                "session",
                "I'm looking for Women Shoes, but I'm still exploring.",
                turn=1,
                top_k=2,
            )
            state = agent._sessions["session"]
            state.asked_attributes.update(QUESTION_ATTRIBUTES)

            follow_up = agent.respond(
                "session",
                "For that, what matters is: leather; padded collar.",
                turn=2,
                top_k=2,
            )
            after_follow_up = agent.respond(
                "session",
                "For that, what matters is: rubber sole.",
                turn=3,
                top_k=2,
            )

        self.assertEqual(follow_up["ask_attribute"], first["ask_attribute"])
        self.assertIsNone(after_follow_up["ask_attribute"])
        self.assertEqual(state.question_counts[first["ask_attribute"]], 2)

    def test_override_reopens_the_unanswered_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = AgentV1(self._catalog_path(directory))
            self.addCleanup(agent.catalog.connection.close)
            agent.reset("session", {})
            state = agent._sessions["session"]
            state.asked_attributes.update({"material", "feature"})
            state.last_asked_attribute = "feature"
            state.question_counts.update({"material": 1, "feature": 1})

            agent._update_state(
                state,
                "Actually, ignore my earlier preference. What I need is: nylon.",
                turn=3,
            )

        self.assertIn("material", state.asked_attributes)
        self.assertNotIn("feature", state.asked_attributes)

    def test_registry_exposes_described_agents(self) -> None:
        self.assertEqual(
            agent_names(),
            ("baseline", "v1", "v1-tuned", "v2", "v2-embedding", "v3"),
        )
        self.assertIn("Original non-LLM baseline", get_agent_spec("baseline").description)
        self.assertIn("First non-LLM agent", get_agent_spec("v1").description)
        self.assertIn("offline-LLM product attributes", get_agent_spec("v2").description)
        self.assertIn("dense retrieval", get_agent_spec("v2-embedding").description)
        self.assertIn("LLM-based conversational state parsing", get_agent_spec("v3").description)


if __name__ == "__main__":
    unittest.main()
