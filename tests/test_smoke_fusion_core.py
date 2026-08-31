from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.smoke_fusion_core import run_smoke


def _catalog(path: Path, count: int = 240) -> Path:
    categories = ("Shoes", "Dresses", "Jewelry", "Jackets")
    colors = ("blue", "black", "red", "green")
    rows = []
    for index in range(count):
        category = categories[index % len(categories)]
        color = colors[index % len(colors)]
        rows.append(
            {
                "parent_asin": f"P{index:04d}",
                "title": f"{color} lightweight {category}",
                "categories": ["Clothing, Shoes & Jewelry", category],
                "features": ["lightweight", "comfortable"],
                "details": {
                    "Color": color,
                    "Material": "cotton",
                    "Size": "medium",
                },
                "store": "Example",
                "description": ["casual walking option"],
                "price": 20.0 + index % 30,
                "average_rating": 4.0,
                "rating_number": index + 1,
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


class DeterministicAgent:
    def __init__(self, catalog_path: Path) -> None:
        self.identifiers = [
            json.loads(line)["parent_asin"]
            for line in catalog_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        start = (turn - 1) * top_k
        page = self.identifiers[start : start + top_k]
        return {
            "message": "ok",
            "ask_attribute": None if turn == 10 else "material",
            "recommendations": [
                {"parent_asin": identifier, "score": 1.0}
                for identifier in page
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def close(self) -> None:
        pass


class InvalidAgent(DeterministicAgent):
    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        return {
            "message": "bad",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": self.identifiers[0]},
                {"parent_asin": self.identifiers[0]},
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


class FusionCoreSmokeTest(unittest.TestCase):
    def test_target_free_repeat_and_exposure_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _catalog(Path(directory) / "catalog.jsonl")
            result = run_smoke(DeterministicAgent, catalog, sessions=20)

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["exact_repeat"]["passed"])
        self.assertEqual(
            result["validation_totals_two_replicas"][
                "same_version_repeat_errors"
            ],
            0,
        )
        self.assertEqual(result["privacy"]["public_opened"], False)

    def test_duplicate_repeat_and_turn10_question_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _catalog(Path(directory) / "catalog.jsonl")
            result = run_smoke(InvalidAgent, catalog, sessions=20)

        totals = result["validation_totals_two_replicas"]
        self.assertEqual(result["status"], "FAIL")
        self.assertGreater(totals["page_duplicate_errors"], 0)
        self.assertGreater(totals["same_version_repeat_errors"], 0)
        self.assertGreater(totals["turn10_question_errors"], 0)


if __name__ == "__main__":
    unittest.main()
