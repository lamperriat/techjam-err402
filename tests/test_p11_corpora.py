from __future__ import annotations

import copy
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

from scripts import build_p11_corpora as builder


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(builder._canonical_json_bytes(row) for row in rows)


class P11CorpusBuilderTests(unittest.TestCase):
    def _products(self, count: int = 140) -> dict[str, dict]:
        products: dict[str, dict] = {}
        for index in range(count):
            parent_asin = f"P{index:04d}"
            material = "cotton" if index % 2 == 0 else "polyester"
            products[parent_asin] = {
                "parent_asin": parent_asin,
                "title": f"Women's {material} dress {index}",
                "features": [f"Made from {material} fabric", "Comfortable fit"],
                "description": ["A casual design for office wear and travel."],
                "price": float(10 + (index % 30) * 5),
                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Women",
                    "Clothing",
                    "Dresses",
                ],
                "details": {"Department": "Women"},
                "average_rating": 4.5,
                "rating_number": index + 1,
                "store": "Example Store",
            }
        return products

    def _opened(self) -> dict[str, list[dict]]:
        names = (
            ("released_public", "public_"),
            ("p1", "derived_p1_"),
            ("p5", "derived_p5_"),
            ("p6", "derived_p6_"),
            ("p7", "derived_p7_"),
            ("p8_selection", "derived_p8_selection_"),
            ("p8_confirmation", "derived_p8_confirmation_"),
            ("p9_selection", "derived_p9_selection_"),
            ("p9_confirmation", "derived_p9_confirmation_"),
        )
        return {
            name: [
                {
                    "category_bucket": "test",
                    "difficulty_bucket": "test",
                    "ground_truth": {"parent_asin": f"P{index:04d}"},
                    "sample_id": f"{prefix}0001",
                    "scenario_type": "buying",
                    "user_profile": {},
                }
            ]
            for index, (name, prefix) in enumerate(names)
        }

    def _protocol(
        self,
        products: dict[str, dict],
        opened: dict[str, list[dict]],
    ) -> dict:
        prefixes = {
            name: rows[0]["sample_id"].rsplit("0001", 1)[0]
            for name, rows in opened.items()
        }
        common_main = {
            "count": 4,
            "scenario_counts": {
                "boundary": 1,
                "browsing": 1,
                "buying": 1,
                "intent_override": 1,
            },
            "expected_samples_sha256": None,
        }
        return {
            "schema_version": builder.PROTOCOL_SCHEMA_VERSION,
            "catalog": {"count": len(products), "path": "catalog.jsonl", "sha256": ""},
            "opened_target_union_count": len(opened),
            "opened_corpora": {
                name: {
                    "path": f"opened/{name}.jsonl",
                    "rows": len(rows),
                    "sample_id_prefix": prefixes[name],
                    "canonical_samples_sha256": builder._samples_sha256(rows),
                }
                for name, rows in opened.items()
            },
            "representative_popularity_bins": [
                {"low": 0.0, "high": 0.5, "count": 2},
                {"low": 0.5, "high": 1.0, "count": 2},
            ],
            "failure_plan_seed": "test-failure-plan",
            "budget_thresholds": [30, 60, 90, 120],
            "budget_minimum_peer_count_per_side": 2,
            "metadata_filename": "metadata.json",
            "splits": {
                "primary": {
                    **common_main,
                    "filename": "primary.jsonl",
                    "sample_id_prefix": "test_primary_",
                    "seed": "test-primary",
                },
                "uniform_tail": {
                    **common_main,
                    "filename": "tail.jsonl",
                    "sample_id_prefix": "test_tail_",
                    "seed": "test-tail",
                    "maximum_popularity_percentile_exclusive": 0.9,
                },
                "confirmation": {
                    **common_main,
                    "filename": "confirmation.jsonl",
                    "sample_id_prefix": "test_confirmation_",
                    "seed": "test-confirmation",
                },
                "failure_negative": {
                    **common_main,
                    "filename": "negative.jsonl",
                    "sample_id_prefix": "test_negative_",
                    "seed": "test-negative",
                },
                "failure_budget": {
                    "count": 4,
                    "filename": "budget.jsonl",
                    "sample_id_prefix": "test_budget_",
                    "seed": "test-budget",
                    "scenario_counts": {"buying": 2, "browsing": 2},
                    "expected_samples_sha256": None,
                },
                "failure_override": {
                    "count": 4,
                    "filename": "override.jsonl",
                    "sample_id_prefix": "test_override_",
                    "seed": "test-override",
                    "scenario_counts": {"intent_override": 4},
                    "expected_samples_sha256": None,
                },
                "failure_missing_evidence": {
                    "count": 4,
                    "filename": "missing.jsonl",
                    "sample_id_prefix": "test_missing_",
                    "seed": "test-missing",
                    "scenario_counts": {"buying": 2, "browsing": 2},
                    "expected_samples_sha256": None,
                },
            },
        }

    def test_deterministic_disjoint_stratified_and_failure_semantics(self) -> None:
        products = self._products()
        opened = self._opened()
        protocol = self._protocol(products, opened)

        first, first_metadata = builder.build_p11_corpora(opened, products, protocol)
        second, second_metadata = builder.build_p11_corpora(opened, products, protocol)

        self.assertEqual(first, second)
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_metadata["opened_registry"]["target_union_count"], 9)
        self.assertEqual(
            first_metadata["opened_vs_new_target_overlaps"],
            {name: 0 for name in first},
        )
        self.assertEqual(
            first_metadata["outputs"]["primary"]["popularity_bin_counts"], [2, 2]
        )
        self.assertEqual(
            first_metadata["outputs"]["confirmation"]["popularity_bin_counts"], [2, 2]
        )
        self.assertLess(
            first_metadata["outputs"]["uniform_tail"]["popularity_percentile"]["maximum"],
            0.9,
        )

        opened_targets = set().union(*(builder._target_ids(rows) for rows in opened.values()))
        target_sets = {name: builder._target_ids(rows) for name, rows in first.items()}
        for targets in target_sets.values():
            self.assertFalse(targets & opened_targets)
        for left, right in combinations(target_sets, 2):
            self.assertFalse(target_sets[left] & target_sets[right])

        for row in first["failure_negative"]:
            audit = row["behavior"]["explicit_negative"]
            self.assertFalse(audit["description_used_as_evidence"])
            self.assertIn(audit["slot"], builder.NEGATIVE_VOCABULARIES)

        for row in first["failure_budget"]:
            audit = row["behavior"]["p11_failure_audit"]
            target_product = products[row["ground_truth"]["parent_asin"]]
            self.assertLessEqual(float(target_product["price"]), float(audit["threshold"]))
            self.assertGreaterEqual(audit["lower_peer_count"], 2)
            self.assertGreaterEqual(audit["upper_peer_count"], 2)

        for row in first["failure_override"]:
            behavior = row["behavior"]
            audit = behavior["p11_failure_audit"]
            self.assertEqual(row["scenario_type"], "intent_override")
            self.assertNotEqual(audit["old_value"], audit["new_value"])
            self.assertIn(audit["new_value"], behavior["override"]["new_value"])
            self.assertIn(behavior["override"]["turn"], (3, 4))

        for row in first["failure_missing_evidence"]:
            audit = row["behavior"]["p11_failure_audit"]
            product = products[row["ground_truth"]["parent_asin"]]
            description = " ".join(product["description"]).casefold()
            structured = " ".join(
                builder._flatten_text(
                    {
                        field: product.get(field)
                        for field in ("title", "categories", "features", "details", "store")
                    }
                )
            ).casefold()
            self.assertEqual(audit["evidence_source"], "description_only")
            self.assertIn(audit["value"], description)
            self.assertNotIn(audit["value"], structured)

    def test_rejects_hash_drift_and_opened_target_overlap(self) -> None:
        products = self._products()
        opened = self._opened()
        protocol = self._protocol(products, opened)

        drifted = copy.deepcopy(opened)
        drifted["p1"][0]["sample_id"] = "derived_p1_changed"
        with self.assertRaisesRegex(builder.CorpusBuildError, "p1 canonical sample"):
            builder.build_p11_corpora(drifted, products, protocol)

        overlapping = copy.deepcopy(opened)
        overlapping["p1"][0]["ground_truth"]["parent_asin"] = overlapping[
            "released_public"
        ][0]["ground_truth"]["parent_asin"]
        overlap_protocol = self._protocol(products, overlapping)
        with self.assertRaisesRegex(builder.CorpusBuildError, "registry overlaps"):
            builder.build_p11_corpora(overlapping, products, overlap_protocol)

    def test_path_build_refuses_overwrite_and_preserves_first_outputs(self) -> None:
        products = self._products()
        opened = self._opened()
        protocol = self._protocol(products, opened)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "catalog.jsonl"
            catalog_path.write_bytes(_jsonl_bytes(list(products.values())))
            protocol["catalog"]["sha256"] = builder._file_sha256(catalog_path)
            opened_dir = root / "opened"
            opened_dir.mkdir()
            for name, rows in opened.items():
                (opened_dir / f"{name}.jsonl").write_bytes(_jsonl_bytes(rows))
            protocol_path = root / "protocol.json"
            protocol_path.write_text(
                json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "outputs"

            metadata = builder.build_and_write_p11_corpora(
                root, protocol_path, output_dir
            )
            self.assertEqual(
                metadata["protocol_file_sha256"], builder._file_sha256(protocol_path)
            )
            self.assertEqual(
                metadata["builder_source"]["sha256"],
                builder._file_sha256(Path(builder.__file__).resolve()),
            )
            expected_files = {
                output_dir / spec["filename"] for spec in protocol["splits"].values()
            } | {output_dir / protocol["metadata_filename"]}
            self.assertTrue(all(path.is_file() for path in expected_files))
            before = {path: path.read_bytes() for path in expected_files}
            for name, spec in protocol["splits"].items():
                self.assertEqual(
                    builder._file_sha256(output_dir / spec["filename"]),
                    metadata["outputs"][name]["samples_sha256"],
                )

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                builder.build_and_write_p11_corpora(root, protocol_path, output_dir)
            self.assertEqual(before, {path: path.read_bytes() for path in expected_files})

    def test_tracked_protocol_freezes_the_full_opened_registry(self) -> None:
        protocol_path = Path(__file__).resolve().parents[1] / "configs" / "p11_corpus_protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["schema_version"], builder.PROTOCOL_SCHEMA_VERSION)
        self.assertEqual(len(protocol["opened_corpora"]), 9)
        self.assertEqual(protocol["opened_target_union_count"], 1800)
        self.assertEqual(
            sum(item["count"] for item in protocol["representative_popularity_bins"]),
            200,
        )
        self.assertEqual(
            {name: spec["count"] for name, spec in protocol["splits"].items()},
            {
                "primary": 200,
                "uniform_tail": 200,
                "confirmation": 200,
                "failure_negative": 80,
                "failure_budget": 80,
                "failure_override": 80,
                "failure_missing_evidence": 80,
            },
        )
        filenames = [spec["filename"] for spec in protocol["splits"].values()]
        self.assertEqual(len(filenames), len(set(filenames)))


if __name__ == "__main__":
    unittest.main()
