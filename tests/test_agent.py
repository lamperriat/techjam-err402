from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from starter.agent import Agent
from utils.llm_client import TokenUsage


PRODUCTS = [
    {
        "parent_asin": "A-BLUE-COTTON",
        "title": "Women's blue cotton casual summer dress",
        "categories": ["Clothing", "Women", "Dresses"],
        "features": ["cotton", "blue", "casual", "comfortable"],
        "details": {"material": "cotton", "color": "blue"},
    },
    {
        "parent_asin": "A-RED-POLY",
        "title": "Women's red polyester formal dress",
        "categories": ["Clothing", "Women", "Dresses"],
        "features": ["polyester", "red", "formal"],
        "details": {"material": "polyester", "color": "red"},
    },
    {
        "parent_asin": "A-BLACK-SHOE",
        "title": "Men's black mesh running shoe",
        "categories": ["Clothing", "Men", "Shoes"],
        "features": ["black", "mesh", "running", "wide"],
    },
]


class AgentTest(unittest.TestCase):
    def _catalog(self, directory: str) -> Path:
        catalog_path = Path(directory) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in PRODUCTS),
            encoding="utf-8",
        )
        return catalog_path

    def test_runs_without_llm_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {}, clear=True):
                agent = Agent(self._catalog(directory))
            self.addCleanup(agent.connection.close)
            agent.reset("test-session", {})
            response = agent.respond(
                "test-session", "I'm looking for women's dresses.", turn=1, top_k=10
            )

        self.assertIsNone(agent.llm_client)
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertTrue(response["recommendations"])
        self.assertEqual(agent.question_policy, "fast")

    def test_reports_injected_llm_usage(self) -> None:
        llm_client = Mock()
        llm_client.consume_usage.return_value = TokenUsage(12, 3)
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory), llm_client=llm_client)
            self.addCleanup(agent.connection.close)
            agent.reset("test-session", {})
            response = agent.respond("test-session", "blue cotton dress", 1, 10)

        self.assertEqual(response["usage"], {"prompt_tokens": 12, "completion_tokens": 3})

    def test_accumulates_constraints_across_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory))
            self.addCleanup(agent.connection.close)
            agent.reset("multi", {})
            agent.respond(
                "multi", "I'm looking for women's dresses, but I'm still exploring.", 1, 10
            )
            response = agent.respond(
                "multi", "For that, what matters is: cotton; color: blue.", 2, 10
            )
            snapshot = agent.debug_snapshot("multi")

        self.assertEqual(response["recommendations"][0]["parent_asin"], "A-BLUE-COTTON")
        self.assertTrue({"dresses", "cotton", "blue"} <= set(snapshot["query_terms"]))

    def test_override_removes_stale_preference_but_preserves_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory))
            self.addCleanup(agent.connection.close)
            agent.reset("override", {})
            agent.respond(
                "override", "I'm looking for women's dresses. red polyester formal", 1, 10
            )
            response = agent.respond(
                "override",
                "Actually, ignore my earlier preference. What I need is: blue cotton casual.",
                3,
                10,
            )
            snapshot = agent.debug_snapshot("override")

        self.assertEqual(response["recommendations"][0]["parent_asin"], "A-BLUE-COTTON")
        self.assertFalse({"red", "polyester"} & set(snapshot["query_terms"]))
        self.assertIn("dresses", snapshot["query_terms"])
        self.assertEqual(snapshot["override_count"], 1)

    def test_override_preserves_later_clarification_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory))
            self.addCleanup(agent.connection.close)
            agent.reset("selective", {})
            agent.respond("selective", "I'm looking for women's dresses. red formal", 1, 10)
            agent.respond(
                "selective", "For that, what matters is: cotton, not polyester.", 2, 10
            )
            agent.respond(
                "selective",
                "Actually, ignore my earlier preference. What I need is: blue casual.",
                3,
                10,
            )
            snapshot = agent.debug_snapshot("selective")

        self.assertFalse({"red", "formal"} & set(snapshot["query_terms"]))
        self.assertTrue({"cotton", "blue"} <= set(snapshot["query_terms"]))
        self.assertIn("polyester", snapshot["excluded_terms"])
        self.assertNotIn("polyester", snapshot["query_terms"])

    def test_repeated_override_moves_version_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory))
            self.addCleanup(agent.connection.close)
            agent.reset("repeat", {})
            agent.respond("repeat", "I'm looking for women's dresses. red formal", 1, 10)
            agent.respond(
                "repeat", "Actually, ignore my earlier preference. What I need is: blue casual.", 3, 10
            )
            agent.respond("repeat", "For that, what matters is: cotton.", 4, 10)
            agent.respond(
                "repeat", "Actually, change my mind. What I need is: black running.", 5, 10
            )
            snapshot = agent.debug_snapshot("repeat")

        self.assertFalse({"blue", "casual"} & set(snapshot["query_terms"]))
        self.assertTrue({"cotton", "black"} <= set(snapshot["query_terms"]))
        self.assertEqual(snapshot["override_count"], 2)
        self.assertEqual(snapshot["version_anchor_turn"], 5)

    def test_plain_actually_does_not_trigger_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory))
            self.addCleanup(agent.connection.close)
            agent.reset("plain-actually", {})
            agent.respond(
                "plain-actually", "I'm looking for women's dresses. red formal", 1, 10
            )
            agent.respond("plain-actually", "Actually, cotton sounds fine.", 2, 10)
            agent.respond("plain-actually", "What I need is: blue.", 3, 10)
            snapshot = agent.debug_snapshot("plain-actually")

        self.assertEqual(snapshot["override_count"], 0)
        self.assertTrue({"red", "formal", "cotton", "blue"} <= set(snapshot["query_terms"]))

    def test_category_change_clears_old_goal_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory))
            self.addCleanup(agent.connection.close)
            agent.reset("goal-change", {})
            first = agent.respond(
                "goal-change", "I'm looking for women's dresses. cotton", 1, 10
            )
            agent.respond(
                "goal-change",
                f"I don't have an additional preference for {first['ask_attribute']}.",
                2,
                10,
            )
            agent.respond(
                "goal-change", "I'm looking for men's shoes. black running", 3, 10
            )
            snapshot = agent.debug_snapshot("goal-change")

        self.assertEqual(snapshot["category_text"], "men's shoes")
        self.assertFalse({"dresses", "cotton"} & set(snapshot["query_terms"]))
        self.assertEqual(snapshot["exhausted_attributes"], [])
        self.assertNotIn("other", snapshot["asked_attributes"])

    def test_negative_phrase_excludes_meaningful_term(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory))
            self.addCleanup(agent.connection.close)
            agent.reset("negative", {})
            agent.respond(
                "negative", "I'm looking for women's dresses. not too formal", 1, 10
            )
            snapshot = agent.debug_snapshot("negative")

        self.assertIn("formal", snapshot["excluded_terms"])
        self.assertNotIn("formal", snapshot["query_terms"])
        self.assertNotIn("too", snapshot["query_terms"])

    def test_no_preference_is_exhausted_and_not_reasked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory))
            self.addCleanup(agent.connection.close)
            agent.reset("boundary", {})
            first = agent.respond(
                "boundary", "I'm looking for women's dresses, but I'm still exploring.", 1, 10
            )
            asked = first["ask_attribute"]
            second = agent.respond(
                "boundary",
                f"I don't have a preference for {asked}; please use your judgment.",
                2,
                10,
            )
            snapshot = agent.debug_snapshot("boundary")

        self.assertIn(asked, snapshot["exhausted_attributes"])
        self.assertEqual(second["ask_attribute"], "other")
        self.assertNotIn("preference", snapshot["query_terms"])

    def test_fast_and_conservative_question_policies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            fast = Agent(catalog, question_policy="fast")
            conservative = Agent(catalog, question_policy="conservative")
            self.addCleanup(fast.connection.close)
            self.addCleanup(conservative.connection.close)
            for agent, session in ((fast, "fast"), (conservative, "conservative")):
                agent.reset(session, {})
                first = agent.respond(
                    session, "I'm looking for women's dresses. cotton", 1, 10
                )
                second = agent.respond(
                    session,
                    f"I don't have an additional preference for {first['ask_attribute']}.",
                    2,
                    10,
                )
                if agent is fast:
                    self.assertEqual(second["ask_attribute"], "other")
                else:
                    self.assertNotEqual(second["ask_attribute"], "other")

    def test_debug_rankings_expose_broad_strict_and_fused_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory))
            self.addCleanup(agent.connection.close)
            agent.reset("routes", {})
            agent.respond("routes", "blue cotton dress", 1, 10)
            rankings = agent.debug_rankings("routes")

        self.assertEqual(set(rankings), {"broad", "strict", "fused"})
        self.assertIn("A-BLUE-COTTON", rankings["fused"])

    def test_response_contract_and_final_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory))
            self.addCleanup(agent.connection.close)
            agent.reset("contract", {})
            response = agent.respond("contract", "clothing", 10, 100)
            with self.assertRaises(ValueError):
                agent.respond("contract", "clothing", True, 10)
            with self.assertRaises(ValueError):
                agent.respond("contract", "clothing", 1, True)

        self.assertIsNone(response["ask_attribute"])
        self.assertLessEqual(len(response["recommendations"]), 10)
        self.assertTrue(
            all(item["parent_asin"] in {product["parent_asin"] for product in PRODUCTS}
                for item in response["recommendations"])
        )

    def test_trace_events_are_target_blind_and_report_stateful_routes(self) -> None:
        events: list[dict] = []
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(directory), trace_sink=events.append)
            self.addCleanup(agent.connection.close)
            agent.reset("opaque-session", {"summary": "local test"})
            agent.respond(
                "opaque-session", "I'm looking for women's dresses. blue cotton", 1, 10
            )

        layers = {event["layer"] for event in events}
        self.assertEqual(
            layers, {"session", "parse", "retrieval", "state", "policy", "output"}
        )
        retrieval = next(event["data"] for event in events if event["layer"] == "retrieval")
        self.assertEqual(set(retrieval["route_counts"]), {"broad", "strict", "fused"})
        serialized = json.dumps(events).lower()
        self.assertNotIn("ground_truth", serialized)
        self.assertNotIn("target_asin", serialized)


if __name__ == "__main__":
    unittest.main()
