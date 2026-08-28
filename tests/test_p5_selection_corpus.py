from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_p5_selection_corpus import (
    DEFAULT_COUNT,
    DEFAULT_METADATA_OUTPUT,
    DEFAULT_OUTPUT,
    DEFAULT_SEED,
    SAMPLE_ID_PREFIX,
    _parser,
    build_and_write_p5_selection_corpus,
    build_p5_selection_corpus,
    main,
)
from scripts.evaluate_generalization import build_product_disjoint_samples


def _sample(sample_id: str, parent_asin: str) -> dict:
    return {
        "sample_id": sample_id,
        "ground_truth": {"parent_asin": parent_asin},
    }


def _products(count: int) -> dict[str, dict]:
    return {
        f"P{index:04d}": {
            "parent_asin": f"P{index:04d}",
            "title": f"Product {index}",
            "categories": ["Clothing", "Test"],
        }
        for index in range(count)
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class P5SelectionCorpusTest(unittest.TestCase):
    def test_builder_is_deterministic_stratified_and_double_disjoint(self) -> None:
        products = _products(450)
        public = [_sample(f"public_{index:04d}", f"P{index:04d}") for index in range(10)]
        prior = [
            _sample(f"derived_p1_{index:04d}", f"P{index + 10:04d}")
            for index in range(10)
        ]

        first, first_metadata = build_p5_selection_corpus(
            public, prior, products, 200, "p5-test-seed"
        )
        second, second_metadata = build_p5_selection_corpus(
            public, prior, products, 200, "p5-test-seed"
        )

        self.assertEqual(first, second)
        self.assertEqual(first_metadata, second_metadata)
        self.assertTrue(
            all(sample["sample_id"].startswith(SAMPLE_ID_PREFIX) for sample in first)
        )
        selected = {
            sample["ground_truth"]["parent_asin"] for sample in first
        }
        self.assertFalse(selected & {f"P{index:04d}" for index in range(20)})
        self.assertEqual(first_metadata["public_target_overlap"], 0)
        self.assertEqual(first_metadata["prior_derived_target_overlap"], 0)
        self.assertEqual(first_metadata["unique_target_count"], 200)
        self.assertEqual(
            first_metadata["scenario_counts"],
            {"boundary": 10, "browsing": 80, "buying": 80, "intent_override": 30},
        )
        self.assertIn("not a private-distribution proxy", first_metadata["boundary"])
        self.assertEqual(len(first_metadata["samples_sha256"]), 64)

    def test_reuse_does_not_change_p1_generator_behavior(self) -> None:
        products = _products(30)
        public = [_sample("public_0001", "P0000")]
        prior = [_sample("derived_p1_0001", "P0001")]
        before, before_metadata = build_product_disjoint_samples(
            public, products, 5, "legacy-seed"
        )

        build_p5_selection_corpus(public, prior, products, 5, "p5-seed")
        after, after_metadata = build_product_disjoint_samples(
            public, products, 5, "legacy-seed"
        )

        self.assertEqual(before, after)
        self.assertEqual(before_metadata, after_metadata)
        self.assertTrue(before[0]["sample_id"].startswith("derived_p1_"))

    def test_path_builder_validates_sources_and_writes_exact_hash(self) -> None:
        products = _products(15)
        public = [_sample("public_0001", "P0000"), _sample("public_0002", "P0001")]
        prior = [
            _sample("derived_p1_0001", "P0002"),
            _sample("derived_p1_0002", "P0003"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            public_path = root / "public.jsonl"
            prior_path = root / "prior.jsonl"
            output_path = root / "selection.jsonl"
            metadata_path = root / "selection.metadata.json"
            _write_jsonl(catalog_path, list(products.values()))
            _write_jsonl(public_path, public)
            _write_jsonl(prior_path, prior)

            metadata = build_and_write_p5_selection_corpus(
                catalog_path,
                public_path,
                prior_path,
                output_path,
                metadata_path,
                count=5,
                seed="path-seed",
                expected_catalog_count=15,
                expected_public_count=2,
                expected_prior_count=2,
            )
            written = output_path.read_bytes()
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(hashlib.sha256(written).hexdigest(), metadata["samples_sha256"])
        self.assertEqual(stored, metadata)
        self.assertTrue(metadata["catalog_source"]["expected_count_verified"])
        self.assertEqual(metadata["public_target_overlap"], 0)
        self.assertEqual(metadata["prior_derived_target_overlap"], 0)

    def test_path_builder_rejects_public_prior_target_overlap(self) -> None:
        products = _products(8)
        public = [_sample("public_0001", "P0000")]
        prior = [_sample("derived_p1_0001", "P0000")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            public_path = root / "public.jsonl"
            prior_path = root / "prior.jsonl"
            _write_jsonl(catalog_path, list(products.values()))
            _write_jsonl(public_path, public)
            _write_jsonl(prior_path, prior)
            with self.assertRaisesRegex(ValueError, "overlap"):
                build_and_write_p5_selection_corpus(
                    catalog_path,
                    public_path,
                    prior_path,
                    root / "selection.jsonl",
                    root / "selection.metadata.json",
                    count=2,
                    expected_catalog_count=8,
                    expected_public_count=1,
                    expected_prior_count=1,
                )

    def test_cli_defaults_and_argument_forwarding(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.count, DEFAULT_COUNT)
        self.assertEqual(args.seed, DEFAULT_SEED)
        self.assertEqual(args.output, DEFAULT_OUTPUT)
        self.assertEqual(args.metadata_output, DEFAULT_METADATA_OUTPUT)

        metadata = {
            "sample_count": 7,
            "public_target_overlap": 0,
            "prior_derived_target_overlap": 0,
            "samples_sha256": "a" * 64,
        }
        with patch(
            "scripts.build_p5_selection_corpus.build_and_write_p5_selection_corpus",
            return_value=metadata,
        ) as build:
            exit_code = main(
                [
                    "--count",
                    "7",
                    "--seed",
                    "cli-seed",
                    "--output",
                    "out.jsonl",
                    "--metadata-output",
                    "meta.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(build.call_args.kwargs, {"count": 7, "seed": "cli-seed"})
        self.assertEqual(build.call_args.args[3:], (Path("out.jsonl"), Path("meta.json")))


if __name__ == "__main__":
    unittest.main()
