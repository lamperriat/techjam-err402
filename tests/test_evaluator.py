from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    main,
    metric_summary,
    normalize_recommendations,
)


class EchoTargetAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        asin = "A"
        if "B" in user_message:
            asin = "B"
        return {"message": "ok", "ask_attribute": None, "recommendations": [{"parent_asin": asin}]}


class EvaluatorTest(unittest.TestCase):
    def test_normalization_preserves_first_valid_unique_order(self) -> None:
        payload = [
            {"parent_asin": "A"}, {"parent_asin": "bad"}, {"parent_asin": "A"},
            "B", {"parent_asin": "C"},
        ]
        self.assertEqual(normalize_recommendations(payload, {"A", "B", "C"}), ["A", "B", "C"])

    def test_metric_summary_assigns_turn_11_to_miss(self) -> None:
        sessions = [
            {"hit": True, "reciprocal_rank": .5, "first_hit_turn": 2},
            {"hit": False, "reciprocal_rank": 0.0, "first_hit_turn": None},
        ]
        self.assertEqual(metric_summary(sessions), {
            "sample_count": 2,
            "hit_rate_at_10": .5,
            "mrr": .25,
            "mttc": 6.5,
        })

    def test_evaluate_derives_hidden_fields_when_public_set_omits_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            catalog_rows = [
                {
                    "parent_asin": "A",
                    "title": "Blue running shoe",
                    "features": ["cotton"],
                    "details": {"department": "womens"},
                    "description": ["walking shoe"],
                    "categories": ["Clothing", "Shoes"],
                    "store": "Example",
                    "average_rating": 4.2,
                    "rating_number": 10,
                    "price": 49.0,
                },
                {
                    "parent_asin": "B",
                    "title": "Black winter boot",
                    "features": ["leather"],
                    "details": {"department": "womens"},
                    "description": ["winter boot"],
                    "categories": ["Clothing", "Boots"],
                    "store": "Example",
                    "average_rating": 4.4,
                    "rating_number": 12,
                    "price": 89.0,
                },
            ]
            catalog_path.write_text("".join(json.dumps(row) + "\n" for row in catalog_rows), encoding="utf-8")
            catalog_ids, categories, products = catalog_index(catalog_path)
            samples = [{
                "sample_id": "public_v2_0001",
                "scenario_type": "buying",
                "user_profile": {"summary": "x"},
                "ground_truth": {"parent_asin": "A"},
            }]
            result = evaluate(EchoTargetAgent(), samples, catalog_ids, categories, products)
            self.assertEqual(result["hit_rate_at_10"], 1.0)

    def test_quiet_cli_prints_only_json_and_writes_nested_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            dataset_path = root / "public_set.jsonl"
            output_path = root / "results" / "baseline.json"
            catalog_path.write_text(
                json.dumps({
                    "parent_asin": "A",
                    "title": "Cotton shoe",
                    "features": ["cotton"],
                    "description": [],
                    "price": 20.0,
                    "categories": ["Clothing", "Shoes"],
                    "details": {},
                    "average_rating": 4.2,
                    "rating_number": 10,
                    "store": "Example",
                }) + "\n",
                encoding="utf-8",
            )
            dataset_path.write_text(
                json.dumps({
                    "sample_id": "sample",
                    "scenario_type": "buying",
                    "user_profile": {},
                    "ground_truth": {"parent_asin": "A"},
                }) + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            arguments = [
                "local_evaluator",
                "--catalog", str(catalog_path),
                "--dataset", str(dataset_path),
                "--output", str(output_path),
                "--agent", "baseline",
                "--quiet",
            ]

            with patch.object(sys, "argv", arguments):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    main()

            self.assertTrue(output_path.exists())

        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
