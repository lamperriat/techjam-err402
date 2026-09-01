from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agents.v212 import ASSET_DIR, AgentV212


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class AgentV212Test(unittest.TestCase):
    def test_frozen_assets_are_present_and_match_submission(self) -> None:
        expected = {
            "p11_features.sqlite": "83b6d8c04be6666173806b6e9cb03301eecb8ca58a60272bfa719e6533380473",
            "small_ranker_fold_safe_v1.json": "f8d0b6c0e402edeb34b1e35119c5295449888bc1be713607e88337fa874d16dc",
        }
        submission_assets = (
            PROJECT_ROOT / "submission/src/err402/agents/v212_runtime/assets"
        )
        for filename, expected_hash in expected.items():
            self.assertEqual(_sha256(ASSET_DIR / filename), expected_hash)
            self.assertEqual(_sha256(submission_assets / filename), expected_hash)

    def test_tiny_catalog_falls_back_safely(self) -> None:
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
            agent = AgentV212(catalog_path)
            self.addCleanup(agent.close)
            agent.reset("session", {})
            response = agent.respond(
                "session",
                "I'm looking for Women Shoes, but I'm still exploring.",
                turn=1,
                top_k=10,
            )

        self.assertEqual(response["recommendations"][0]["parent_asin"], "A")
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertTrue(agent._p11_status()["fallback"])


if __name__ == "__main__":
    unittest.main()
