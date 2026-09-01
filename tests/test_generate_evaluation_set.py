from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from generate_evaluation_set import (
    category_bucket,
    generate_samples,
    weighted_unique_sample,
)


class GenerateEvaluationSetTest(unittest.TestCase):
    def test_category_bucket_ignores_umbrella_category(self) -> None:
        self.assertEqual(
            category_bucket(["Clothing, Shoes & Jewelry", "Women", "Shoes"]),
            "shoes",
        )
        self.assertEqual(
            category_bucket(["Clothing, Shoes & Jewelry", "Women", "Jewelry"]),
            "jewelry",
        )
        self.assertEqual(
            category_bucket(["Clothing, Shoes & Jewelry", "Women", "Clothing"]),
            "clothing",
        )

    def test_weighted_sample_rejects_more_products_than_are_available(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 1 are eligible"):
            weighted_unique_sample({"A": [{}]}, 2, random.Random(1))

    def test_generation_is_deterministic_unique_and_excludes_public_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            records_path = root / "records.csv"
            public_path = root / "public.jsonl"
            categories = ("Clothing", "Shoes", "Jewelry")
            catalog_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "parent_asin": f"P{index:02d}",
                            "categories": [
                                "Clothing, Shoes & Jewelry",
                                "Women",
                                categories[index % len(categories)],
                            ],
                        }
                    )
                    + "\n"
                    for index in range(25)
                ),
                encoding="utf-8",
            )
            with records_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["user_id", "parent_asin", "rating", "timestamp", "history"],
                )
                writer.writeheader()
                for index in range(25):
                    for repeat in range(index % 3 + 1):
                        writer.writerow(
                            {
                                "user_id": f"U{index:02d}_{repeat}",
                                "parent_asin": f"P{index:02d}",
                                "rating": str(index % 5 + 1),
                                "timestamp": "0",
                                "history": "H1 H2 H3 H4",
                            }
                        )
            public_path.write_text(
                json.dumps(
                    {
                        "ground_truth": {"parent_asin": "P00"},
                        "user_profile": {"preference_tags": ["comfort", "fit"]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            first = generate_samples(records_path, catalog_path, public_path, 20, seed=7)
            second = generate_samples(records_path, catalog_path, public_path, 20, seed=7)

        self.assertEqual(first, second)
        asins = [sample["ground_truth"]["parent_asin"] for sample in first]
        self.assertEqual(len(asins), len(set(asins)))
        self.assertNotIn("P00", asins)
        self.assertEqual(
            Counter(sample["scenario_type"] for sample in first),
            {"buying": 8, "browsing": 8, "intent_override": 3, "boundary": 1},
        )
        expected_difficulty = {
            "buying": "easy",
            "browsing": "medium",
            "intent_override": "hard",
            "boundary": "medium",
        }
        self.assertTrue(
            all(
                sample["difficulty_bucket"]
                == expected_difficulty[sample["scenario_type"]]
                for sample in first
            )
        )
        self.assertTrue(
            all(
                sample["user_profile"]["purchase_frequency"]
                == "3-4 prior purchases"
                for sample in first
            )
        )
        self.assertTrue(
            all(
                sample["user_profile"]["preference_tags"] == ["comfort", "fit"]
                for sample in first
            )
        )


if __name__ == "__main__":
    unittest.main()
