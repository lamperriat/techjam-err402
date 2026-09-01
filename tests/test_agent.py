from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.v1 import AgentV1
from starter.agent import Agent, BaselineAgent


class AgentTest(unittest.TestCase):
    def test_default_export_implements_official_interface_with_v1(self) -> None:
        self.assertIs(Agent, AgentV1)
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                json.dumps({
                    "parent_asin": "A",
                    "title": "Blue cotton running shoe",
                    "features": ["lightweight"],
                    "description": ["walking shoe"],
                    "price": 20.0,
                    "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes"],
                    "details": {"Department": "womens"},
                    "average_rating": 4.2,
                    "rating_number": 10,
                    "store": "Example",
                }) + "\n",
                encoding="utf-8",
            )

            agent = Agent(catalog_path)
            self.addCleanup(agent.close)
            agent.reset("test-session", {})
            response = agent.respond(
                "test-session",
                "I'm looking for Women Shoes, but I'm still exploring.",
                turn=1,
                top_k=10,
            )

        self.assertIsInstance(response["message"], str)
        self.assertIn(
            response["ask_attribute"],
            {"material", "color", "style", "size", "budget", "feature", "use_case", None},
        )
        self.assertEqual(response["recommendations"][0]["parent_asin"], "A")
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_baseline_has_no_llm_dependency_and_reports_zero_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                json.dumps({"parent_asin": "A", "title": "Test product"}) + "\n",
                encoding="utf-8",
            )

            agent = BaselineAgent(catalog_path)
            self.addCleanup(agent.connection.close)
            agent.reset("test-session", {})
            response = agent.respond("test-session", "Test product", turn=1, top_k=10)

        self.assertFalse(hasattr(agent, "llm_client"))
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})


if __name__ == "__main__":
    unittest.main()
