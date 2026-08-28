from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_p6_selection_corpus import (
    DEFAULT_COUNT,
    DEFAULT_METADATA_OUTPUT,
    DEFAULT_OUTPUT,
    DEFAULT_P1_PATH,
    DEFAULT_P5_PATH,
    DEFAULT_SEED,
    EXPECTED_SCENARIO_COUNTS,
    P1_FROZEN_SAMPLES_SHA256,
    P5_FROZEN_SHA256,
    SAMPLE_ID_PREFIX,
    _parser,
    build_and_write_p6_selection_corpus,
    build_p6_selection_corpus,
    main,
)
from scripts.evaluate_generalization import _samples_sha256


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


def _batch(prefix: str, start: int, count: int) -> list[dict]:
    return [
        _sample(f"{prefix}{index + 1:04d}", f"P{start + index:04d}")
        for index in range(count)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class P6SelectionCorpusTest(unittest.TestCase):
    def test_builder_is_deterministic_stratified_and_triple_disjoint(self) -> None:
        products = _products(900)
        public = _batch("public_", 0, 10)
        p1 = _batch("derived_p1_", 10, 10)
        p5 = _batch("derived_p5_", 20, 10)

        first, first_metadata = build_p6_selection_corpus(
            public, p1, p5, products, 200, "p6-test-seed"
        )
        second, second_metadata = build_p6_selection_corpus(
            public, p1, p5, products, 200, "p6-test-seed"
        )

        self.assertEqual(first, second)
        self.assertEqual(first_metadata, second_metadata)
        self.assertTrue(
            all(sample["sample_id"].startswith(SAMPLE_ID_PREFIX) for sample in first)
        )
        selected = {sample["ground_truth"]["parent_asin"] for sample in first}
        self.assertEqual(len(selected), 200)
        self.assertFalse(selected & {f"P{index:04d}" for index in range(30)})
        self.assertEqual(first_metadata["public_target_overlap"], 0)
        self.assertEqual(first_metadata["prior_p1_target_overlap"], 0)
        self.assertEqual(first_metadata["prior_p5_target_overlap"], 0)
        self.assertEqual(first_metadata["scenario_counts"], EXPECTED_SCENARIO_COUNTS)
        self.assertIn("not a private-distribution proxy", first_metadata["boundary"])
        self.assertEqual(
            first_metadata["samples_sha256"],
            hashlib.sha256(
                "".join(
                    json.dumps(
                        sample,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                    for sample in first
                ).encode("utf-8")
            ).hexdigest(),
        )

    def test_builder_rejects_non_disjoint_inputs_and_wrong_prior_ids(self) -> None:
        products = _products(30)
        public = _batch("public_", 0, 2)
        p1 = _batch("derived_p1_", 2, 2)
        p5 = _batch("derived_p5_", 4, 2)

        with self.subTest("pairwise target overlap"):
            overlapping_p5 = [p5[0], _sample("derived_p5_0002", "P0002")]
            with self.assertRaisesRegex(ValueError, "P1/P5 input targets overlap"):
                build_p6_selection_corpus(
                    public, p1, overlapping_p5, products, 3, "overlap-seed"
                )

        with self.subTest("P5 sample ID"):
            invalid_p5 = [dict(p5[0]), dict(p5[1])]
            invalid_p5[0]["sample_id"] = "derived_p1_9999"
            with self.assertRaisesRegex(ValueError, "prior P5 corpus"):
                build_p6_selection_corpus(
                    public, p1, invalid_p5, products, 3, "id-seed"
                )

    def test_path_builder_writes_exact_hash_and_three_way_audit(self) -> None:
        products = _products(20)
        public = _batch("public_", 0, 2)
        p1 = _batch("derived_p1_", 2, 2)
        p5 = _batch("derived_p5_", 4, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            public_path = root / "public.jsonl"
            p1_path = root / "p1.jsonl"
            p5_path = root / "p5.jsonl"
            output_path = root / "selection.jsonl"
            metadata_path = root / "selection.metadata.json"
            _write_jsonl(catalog_path, list(products.values()))
            _write_jsonl(public_path, public)
            _write_jsonl(p1_path, p1)
            _write_jsonl(p5_path, p5)
            p5_hash = hashlib.sha256(p5_path.read_bytes()).hexdigest()

            metadata = build_and_write_p6_selection_corpus(
                catalog_path,
                public_path,
                p1_path,
                p5_path,
                output_path,
                metadata_path,
                count=5,
                seed="path-seed",
                expected_catalog_count=20,
                expected_public_count=2,
                expected_p1_count=2,
                expected_p5_count=2,
                expected_p1_samples_sha256=_samples_sha256(p1),
                expected_p5_sha256=p5_hash,
            )
            written = output_path.read_bytes()
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(hashlib.sha256(written).hexdigest(), metadata["samples_sha256"])
        self.assertEqual(stored, metadata)
        self.assertTrue(metadata["catalog_source"]["expected_count_verified"])
        self.assertTrue(
            metadata["input_sources"]["prior_p5_derived"][
                "frozen_sha256_verified"
            ]
        )
        self.assertEqual(metadata["exclusions"]["combined_unique_target_count"], 6)
        self.assertEqual(metadata["public_target_overlap"], 0)
        self.assertEqual(metadata["prior_p1_target_overlap"], 0)
        self.assertEqual(metadata["prior_p5_target_overlap"], 0)

    def test_path_builder_rejects_p5_hash_drift_before_writing(self) -> None:
        products = _products(12)
        public = _batch("public_", 0, 1)
        p1 = _batch("derived_p1_", 1, 1)
        p5 = _batch("derived_p5_", 2, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            public_path = root / "public.jsonl"
            p1_path = root / "p1.jsonl"
            p5_path = root / "p5.jsonl"
            output_path = root / "selection.jsonl"
            _write_jsonl(catalog_path, list(products.values()))
            _write_jsonl(public_path, public)
            _write_jsonl(p1_path, p1)
            _write_jsonl(p5_path, p5)

            with self.assertRaisesRegex(ValueError, "P5 frozen SHA-256 mismatch"):
                build_and_write_p6_selection_corpus(
                    catalog_path,
                    public_path,
                    p1_path,
                    p5_path,
                    output_path,
                    root / "selection.metadata.json",
                    count=2,
                    expected_catalog_count=12,
                    expected_public_count=1,
                    expected_p1_count=1,
                    expected_p5_count=1,
                    expected_p1_samples_sha256=_samples_sha256(p1),
                    expected_p5_sha256="0" * 64,
                )
            self.assertFalse(output_path.exists())

    def test_path_builder_rejects_p1_canonical_hash_drift_before_writing(self) -> None:
        products = _products(12)
        public = _batch("public_", 0, 1)
        p1 = _batch("derived_p1_", 1, 1)
        p5 = _batch("derived_p5_", 2, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            public_path = root / "public.jsonl"
            p1_path = root / "p1.jsonl"
            p5_path = root / "p5.jsonl"
            output_path = root / "selection.jsonl"
            _write_jsonl(catalog_path, list(products.values()))
            _write_jsonl(public_path, public)
            _write_jsonl(p1_path, p1)
            _write_jsonl(p5_path, p5)

            with self.assertRaisesRegex(
                ValueError, "P1 frozen sample SHA-256 mismatch"
            ):
                build_and_write_p6_selection_corpus(
                    catalog_path,
                    public_path,
                    p1_path,
                    p5_path,
                    output_path,
                    root / "selection.metadata.json",
                    count=2,
                    expected_catalog_count=12,
                    expected_public_count=1,
                    expected_p1_count=1,
                    expected_p5_count=1,
                    expected_p1_samples_sha256="0" * 64,
                    expected_p5_sha256=hashlib.sha256(
                        p5_path.read_bytes()
                    ).hexdigest(),
                )
            self.assertFalse(output_path.exists())

    def test_path_builder_rejects_output_collision_with_frozen_input(self) -> None:
        products = _products(12)
        public = _batch("public_", 0, 1)
        p1 = _batch("derived_p1_", 1, 1)
        p5 = _batch("derived_p5_", 2, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            public_path = root / "public.jsonl"
            p1_path = root / "p1.jsonl"
            p5_path = root / "p5.jsonl"
            _write_jsonl(catalog_path, list(products.values()))
            _write_jsonl(public_path, public)
            _write_jsonl(p1_path, p1)
            _write_jsonl(p5_path, p5)

            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                build_and_write_p6_selection_corpus(
                    catalog_path,
                    public_path,
                    p1_path,
                    p5_path,
                    p5_path,
                    root / "selection.metadata.json",
                    count=2,
                    expected_catalog_count=12,
                    expected_public_count=1,
                    expected_p1_count=1,
                    expected_p5_count=1,
                    expected_p1_samples_sha256=_samples_sha256(p1),
                    expected_p5_sha256=hashlib.sha256(
                        p5_path.read_bytes()
                    ).hexdigest(),
                )

    def test_real_frozen_p5_hash_constant_matches_required_digest(self) -> None:
        self.assertEqual(
            P5_FROZEN_SHA256,
            "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c",
        )
        self.assertEqual(
            P1_FROZEN_SAMPLES_SHA256,
            "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae",
        )

    def test_cli_defaults_and_argument_forwarding(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.prior_p1, DEFAULT_P1_PATH)
        self.assertEqual(args.prior_p5, DEFAULT_P5_PATH)
        self.assertEqual(args.count, DEFAULT_COUNT)
        self.assertEqual(args.seed, DEFAULT_SEED)
        self.assertEqual(args.output, DEFAULT_OUTPUT)
        self.assertEqual(args.metadata_output, DEFAULT_METADATA_OUTPUT)

        metadata = {
            "sample_count": 7,
            "public_target_overlap": 0,
            "prior_p1_target_overlap": 0,
            "prior_p5_target_overlap": 0,
            "samples_sha256": "a" * 64,
        }
        with patch(
            "scripts.build_p6_selection_corpus.build_and_write_p6_selection_corpus",
            return_value=metadata,
        ) as build:
            exit_code = main(
                [
                    "--count",
                    "7",
                    "--seed",
                    "cli-seed",
                    "--prior-p1",
                    "p1.jsonl",
                    "--prior-p5",
                    "p5.jsonl",
                    "--output",
                    "out.jsonl",
                    "--metadata-output",
                    "meta.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(build.call_args.kwargs, {"count": 7, "seed": "cli-seed"})
        self.assertEqual(
            build.call_args.args[2:],
            (
                Path("p1.jsonl"),
                Path("p5.jsonl"),
                Path("out.jsonl"),
                Path("meta.json"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
