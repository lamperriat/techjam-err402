from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from starter.agent import Agent
from utils.llm_client import TokenUsage


class AgentTest(unittest.TestCase):
    def test_runs_without_llm_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                json.dumps({"parent_asin": "A", "title": "Test product"}) + "\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=True):
                agent = Agent(catalog_path)
                self.addCleanup(agent.connection.close)
                agent.reset("test-session", {})
                response = agent.respond("test-session", "Test product", turn=1, top_k=10)

        self.assertIsNone(agent.llm_client)
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_reports_injected_llm_usage(self) -> None:
        llm_client = Mock()
        llm_client.consume_usage.return_value = TokenUsage(12, 3)
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                json.dumps({"parent_asin": "A", "title": "Test product"}) + "\n",
                encoding="utf-8",
            )

            agent = Agent(catalog_path, llm_client=llm_client)
            self.addCleanup(agent.connection.close)
            agent.reset("test-session", {})
            response = agent.respond("test-session", "Test product", turn=1, top_k=10)

        self.assertIs(agent.llm_client, llm_client)
        self.assertEqual(response["usage"], {"prompt_tokens": 12, "completion_tokens": 3})


if __name__ == "__main__":
    unittest.main()
