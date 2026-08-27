from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from starter.agent import BaselineAgent
from utils.llm_client import TokenUsage


class AgentTest(unittest.TestCase):
    @patch("starter.agent.LLMClient")
    def test_initializes_llm_client_and_reports_usage(self, llm_client: Mock) -> None:
        llm_client.return_value.consume_usage.return_value = TokenUsage(12, 3)
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

        llm_client.assert_called_once_with()
        self.assertIs(agent.llm_client, llm_client.return_value)
        self.assertEqual(response["usage"], {"prompt_tokens": 12, "completion_tokens": 3})


if __name__ == "__main__":
    unittest.main()
