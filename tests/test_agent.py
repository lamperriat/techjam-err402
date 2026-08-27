from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import BaselineAgent


class AgentTest(unittest.TestCase):
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
