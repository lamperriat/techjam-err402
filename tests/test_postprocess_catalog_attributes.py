from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from postprocess_catalog_attributes import (
    exact_catalog_substring,
    normalize_attribute_names,
    postprocess_catalog,
    repair_evidence,
)


def empty_attributes() -> dict[str, list[dict[str, str]]]:
    return {
        "material": [],
        "color": [],
        "size_fit": [],
        "style": [],
        "use_case": [],
        "specific_attributes": [],
    }


class ExactCatalogSubstringTest(unittest.TestCase):
    def test_returns_original_case_and_whitespace(self) -> None:
        source = {"features": ["Includes a Memory   Foam footbed"]}

        match = exact_catalog_substring("memory foam", source)

        self.assertEqual(match, "Memory   Foam")

    def test_returns_none_for_unsupported_value(self) -> None:
        self.assertIsNone(exact_catalog_substring("waterproof", {"title": "Rain boot"}))

    def test_does_not_match_inside_a_larger_word(self) -> None:
        self.assertIsNone(exact_catalog_substring("cos", {"title": "Cosplay wig"}))

    def test_matches_a_single_character_size_at_token_boundaries(self) -> None:
        self.assertEqual(exact_catalog_substring("M", {"title": "Size M shirt"}), "M")


class RepairEvidenceTest(unittest.TestCase):
    def test_repairs_grounded_value_with_exact_short_source_slice(self) -> None:
        rejected = [
            {
                "field": "specific_attributes",
                "reason": "evidence is not an exact substring of the supplied product data",
                "entry": {
                    "name": "closure_type",
                    "value": "ZIPPER",
                    "evidence": "This product has a ZIPPER closure.",
                },
            }
        ]

        repaired, count = repair_evidence(
            empty_attributes(),
            rejected,
            {"details": {"Closure": "Zipper"}},
        )

        self.assertEqual(count, 1)
        self.assertEqual(
            repaired["specific_attributes"],
            [{"name": "closure_type", "value": "ZIPPER", "evidence": "Zipper"}],
        )

    def test_does_not_repair_unsupported_or_subjective_values(self) -> None:
        rejected = [
            {
                "field": "specific_attributes",
                "reason": "evidence exceeds 80 characters",
                "entry": {
                    "name": "quality",
                    "value": "premium",
                    "evidence": "x" * 81,
                },
            }
        ]

        repaired, count = repair_evidence(
            empty_attributes(),
            rejected,
            {"title": "Premium shoe"},
        )

        self.assertEqual(count, 0)
        self.assertEqual(repaired["specific_attributes"], [])


class NormalizeAttributeNamesTest(unittest.TestCase):
    def test_applies_explicit_aliases_and_removes_created_duplicates(self) -> None:
        attributes = empty_attributes()
        attributes["specific_attributes"] = [
            {"name": "closure", "value": "Zipper", "evidence": "Zipper"},
            {"name": "closure_type", "value": "zipper", "evidence": "zipper"},
            {"name": "anti_slip", "value": "Rubber grip", "evidence": "Rubber grip"},
        ]

        normalized, aliases, duplicates = normalize_attribute_names(attributes)

        self.assertEqual(aliases, 2)
        self.assertEqual(duplicates, 1)
        self.assertEqual(
            [entry["name"] for entry in normalized["specific_attributes"]],
            ["closure", "slip_resistance"],
        )


class PostprocessCatalogTest(unittest.TestCase):
    def test_writes_complete_processed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            raw_path = root / "raw.jsonl"
            output_path = root / "processed.jsonl"
            item = {
                "parent_asin": "product-1",
                "title": "Blue shoe with Zipper closure",
                "features": [],
                "description": [],
                "categories": ["Shoes"],
                "details": {},
            }
            catalog_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            raw_record = {
                "parent_asin": "product-1",
                "status": "success",
                "attributes": empty_attributes(),
                "rejected_attributes": [
                    {
                        "field": "specific_attributes",
                        "reason": "evidence is not an exact substring of the supplied product data",
                        "entry": {
                            "name": "closure_type",
                            "value": "Zipper",
                            "evidence": "a Zipper closure",
                        },
                    }
                ],
            }
            raw_path.write_text(
                json.dumps({"record_type": "metadata", "experiment": {}})
                + "\n"
                + json.dumps(raw_record)
                + "\n",
                encoding="utf-8",
            )

            summary = postprocess_catalog(
                catalog_path,
                raw_path,
                output_path,
                show_progress=False,
            )

            lines = output_path.read_text(encoding="utf-8").splitlines()
            processed = json.loads(lines[1])
            self.assertEqual(summary["products"], 1)
            self.assertEqual(summary["evidence_repairs"], 1)
            self.assertEqual(summary["aliased_names"], 1)
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                processed["attributes"]["specific_attributes"],
                [{"name": "closure", "value": "Zipper", "evidence": "Zipper"}],
            )


if __name__ == "__main__":
    unittest.main()
