from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from extract_attribute_pilot import (
    PilotConfig,
    run_pilot,
    sample_products,
    validate_extraction,
)
from retrieval.catalog import category_group
from utils.llm_client import LLMConfig, TokenUsage


def product(group: str, index: int, leaf: str) -> dict:
    if group == "shoes":
        categories = ["Clothing, Shoes & Jewelry", "Women", "Shoes", leaf]
    elif group == "jewelry":
        categories = ["Clothing, Shoes & Jewelry", "Women", "Jewelry", leaf]
    else:
        categories = ["Clothing, Shoes & Jewelry", "Women", leaf]
    return {
        "parent_asin": f"{group}-{index}",
        "title": f"Blue {leaf}",
        "features": ["Memory foam footbed"],
        "description": [],
        "categories": categories,
        "details": {"Department": "Women", "Brand": "Example"},
        "price": 25.0,
        "average_rating": 4.5,
        "rating_number": 20,
        "store": "Example Store",
    }


def valid_response() -> dict:
    return {
        "material": [],
        "color": [{"value": "Blue", "evidence": "Blue Leaf 0"}],
        "size_fit": [],
        "style": [],
        "use_case": [],
        "specific_attributes": [
            {
                "name": "cushioning",
                "value": "Memory foam footbed",
                "evidence": "Memory foam footbed",
            }
        ],
    }


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.config = LLMConfig("test-key", "fake-model")
        self.responses = responses
        self.calls = 0
        self._usage = TokenUsage()

    def generate_json(
        self,
        messages,
        *,
        temperature=None,
        max_tokens=None,
        extra_body=None,
    ):
        self.asserted_temperature = temperature
        self.asserted_max_tokens = max_tokens
        self.asserted_extra_body = extra_body
        response = self.responses[self.calls]
        self.calls += 1
        self._usage += TokenUsage(10, 2)
        return response

    def consume_usage(self) -> TokenUsage:
        usage = self._usage
        self._usage = TokenUsage()
        return usage


class SampleProductsTest(unittest.TestCase):
    def test_samples_equal_groups_deterministically_with_leaf_cap(self) -> None:
        products = [
            product(group, leaf_index * 10 + item_index, f"Leaf {leaf_index}")
            for group in ("clothing", "shoes", "jewelry")
            for leaf_index in range(3)
            for item_index in range(3)
        ]
        config = PilotConfig(samples_per_group=2, leaf_cap=1, seed=17)

        first = sample_products(products, config)
        second = sample_products(products, config)

        self.assertEqual(
            [item["parent_asin"] for item in first],
            [item["parent_asin"] for item in second],
        )
        counts = Counter(category_group(item["categories"]) for item in first)
        self.assertEqual(counts, Counter({"clothing": 2, "shoes": 2, "jewelry": 2}))
        paths = Counter(tuple(item["categories"]) for item in first)
        self.assertTrue(all(count <= 1 for count in paths.values()))

    def test_rejects_impossible_leaf_cap(self) -> None:
        products = [
            product(group, index, "Only Leaf")
            for group in ("clothing", "shoes", "jewelry")
            for index in range(2)
        ]

        with self.assertRaisesRegex(ValueError, "Cannot sample"):
            sample_products(products, PilotConfig(samples_per_group=2, leaf_cap=1))


class ValidateExtractionTest(unittest.TestCase):
    def test_keeps_only_grounded_non_subjective_values(self) -> None:
        source = {
            "title": "Blue trail shoe",
            "features": ["Memory foam footbed", "Reinforced rubber toe"],
        }
        raw = {
            "material": [{"value": "rubber", "evidence": "Reinforced rubber toe"}],
            "color": [
                {"value": "Blue", "evidence": "Blue trail shoe"},
                {"value": "Red", "evidence": "Red trail shoe"},
            ],
            "size_fit": [],
            "style": [],
            "use_case": [],
            "specific_attributes": [
                {
                    "name": "cushioning",
                    "value": "Memory foam footbed",
                    "evidence": "Memory foam footbed",
                },
                {"name": "comfort", "value": "comfortable", "evidence": "Memory foam footbed"},
            ],
        }

        result = validate_extraction(raw, source)

        self.assertEqual(len(result.attributes["material"]), 1)
        self.assertEqual(len(result.attributes["color"]), 1)
        self.assertEqual(len(result.attributes["specific_attributes"]), 1)
        self.assertEqual(len(result.rejected_attributes), 2)
        self.assertEqual(result.schema_errors, [])

    def test_reports_missing_and_unexpected_fields(self) -> None:
        result = validate_extraction(
            {"material": "cotton", "unknown": []},
            {"title": "Cotton shirt"},
        )

        self.assertIn("material must be an array", result.schema_errors)
        self.assertIn("color must be an array", result.schema_errors)
        self.assertIn("unexpected root fields: unknown", result.schema_errors)


class RunPilotTest(unittest.TestCase):
    def test_writes_incremental_records_with_usage_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            output_path = root / "pilot.jsonl"
            products = [
                product("clothing", 0, "Leaf 0"),
                product("shoes", 0, "Leaf 0"),
                product("jewelry", 0, "Leaf 0"),
            ]
            catalog_path.write_text(
                "".join(json.dumps(item) + "\n" for item in products),
                encoding="utf-8",
            )
            config = PilotConfig(samples_per_group=1, leaf_cap=1, seed=3)
            first_client = FakeLLM([valid_response(), valid_response(), valid_response()])

            summary = run_pilot(
                catalog_path,
                output_path,
                config,
                first_client,
                show_progress=False,
            )

            records = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertTrue(all(record["status"] == "success" for record in records))
            self.assertTrue(all(record["usage"]["total_tokens"] == 12 for record in records))
            self.assertTrue(all("raw_extraction" not in record for record in records))
            self.assertTrue(all("catalog" not in record for record in records))
            self.assertEqual(summary["usage"]["total_tokens"], 36)
            self.assertEqual(first_client.asserted_temperature, 0)
            self.assertEqual(first_client.asserted_max_tokens, 800)
            self.assertEqual(
                first_client.asserted_extra_body,
                {"thinking": {"type": "disabled"}},
            )

            resumed_client = FakeLLM([])
            resumed = run_pilot(
                catalog_path,
                output_path,
                config,
                resumed_client,
                resume=True,
                show_progress=False,
            )

            self.assertEqual(resumed_client.calls, 0)
            self.assertEqual(resumed["successful_products"], 3)
            self.assertEqual(resumed["usage"]["total_tokens"], 36)

    def test_does_not_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            output_path = root / "pilot.jsonl"
            products = [
                product("clothing", 0, "Leaf 0"),
                product("shoes", 0, "Leaf 0"),
                product("jewelry", 0, "Leaf 0"),
            ]
            catalog_path.write_text(
                "".join(json.dumps(item) + "\n" for item in products),
                encoding="utf-8",
            )
            output_path.write_text("existing\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                run_pilot(
                    catalog_path,
                    output_path,
                    PilotConfig(samples_per_group=1, leaf_cap=1),
                    FakeLLM([]),
                    show_progress=False,
                )


if __name__ == "__main__":
    unittest.main()
