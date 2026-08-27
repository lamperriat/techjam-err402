from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from evaluator.local_evaluator import catalog_index
from observer.server import make_handler
from observer.trace import TraceRunner
from starter.agent import Agent


class ObserverTraceTest(unittest.TestCase):
    def test_trace_exposes_layers_without_changing_agent_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            products = [
                {
                    "parent_asin": "A",
                    "title": "Blue cotton running shoe",
                    "features": ["cotton", "running"],
                    "categories": ["Clothing", "Shoes"],
                    "price": 49.0,
                },
                {
                    "parent_asin": "B",
                    "title": "Black leather winter boot",
                    "features": ["leather", "winter"],
                    "categories": ["Clothing", "Boots"],
                    "price": 89.0,
                },
            ]
            catalog_path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            catalog_ids, categories, product_index = catalog_index(catalog_path)
            sample = {
                "sample_id": "public_test_1",
                "scenario_type": "buying",
                "difficulty_bucket": "easy",
                "user_profile": {"summary": "Likes practical shoes"},
                "ground_truth": {"parent_asin": "A"},
            }
            runner = TraceRunner(
                Agent(catalog_path),
                [sample],
                catalog_ids,
                categories,
                product_index,
            )

            trace = runner.trace("public_test_1")

        self.assertTrue(trace["result"]["hit"])
        self.assertEqual(trace["result"]["first_hit_turn"], 1)
        self.assertEqual(trace["turns"][0]["target_top10_rank"], 1)
        self.assertEqual(trace["turns"][0]["retrieval"]["target_retrieval_rank"], 1)
        self.assertGreaterEqual(trace["turns"][0]["retrieval"]["candidate_count"], 1)
        self.assertIn("cotton", trace["turns"][0]["retrieval"]["terms"])
        self.assertEqual(runner.list_sessions()["sessions"][0]["sample_id"], "public_test_1")

    def test_unknown_sample_is_rejected(self) -> None:
        runner = TraceRunner(None, [], set(), {}, {})
        with self.assertRaises(KeyError):
            runner.trace("missing")

    def test_http_api_exposes_sessions_and_rejects_unknown_routes(self) -> None:
        class FakeRuntime:
            def list_sessions(self) -> dict:
                return {"metrics": {"hit_rate_at_10": 0.5}, "sessions": []}

            def trace(self, sample_id: str, refresh: bool = False) -> dict:
                return {"sample_id": sample_id, "refresh": refresh}

        server = HTTPServer(("127.0.0.1", 0), make_handler(FakeRuntime()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base_url = f"http://127.0.0.1:{server.server_port}"

        with urllib.request.urlopen(f"{base_url}/api/sessions") as response:
            payload = json.load(response)
        self.assertEqual(payload["metrics"]["hit_rate_at_10"], 0.5)

        with urllib.request.urlopen(f"{base_url}/api/trace?sample_id=public_1&refresh=1") as response:
            trace = json.load(response)
        self.assertEqual(trace, {"sample_id": "public_1", "refresh": True})

        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(f"{base_url}/missing")
        self.assertEqual(context.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
