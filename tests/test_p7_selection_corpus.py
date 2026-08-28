from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

from scripts.build_p7_selection_corpus import (
    CATALOG_FROZEN_SHA256,
    DEFAULT_COUNT,
    DEFAULT_METADATA_OUTPUT,
    DEFAULT_OUTPUT,
    DEFAULT_P1_PATH,
    DEFAULT_P5_PATH,
    DEFAULT_P6_PATH,
    DEFAULT_SEED,
    EXPECTED_SCENARIO_COUNTS,
    P1_FROZEN_SHA256,
    P1_FROZEN_SAMPLES_SHA256,
    P5_FROZEN_SHA256,
    P5_FROZEN_SAMPLES_SHA256,
    P6_FROZEN_SHA256,
    P6_FROZEN_SAMPLES_SHA256,
    P7_FROZEN_SHA256,
    PUBLIC_FROZEN_GIT_BLOB_SHA1,
    SAMPLE_ID_PREFIX,
    _parser,
    build_and_write_p7_selection_corpus,
    build_p7_selection_corpus,
    main,
)
from scripts.evaluate_generalization import _samples_sha256
from scripts.verify_official_assets import git_blob_sha1


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


def _fixture(root: Path) -> tuple[dict[str, Path], dict[str, list[dict]], dict]:
    rows = {
        "public": _batch("public_", 0, 2),
        "p1": _batch("derived_p1_", 2, 2),
        "p5": _batch("derived_p5_", 4, 2),
        "p6": _batch("derived_p6_", 6, 2),
    }
    products = _products(30)
    paths = {
        "catalog": root / "catalog.jsonl",
        "public": root / "public.jsonl",
        "p1": root / "p1.jsonl",
        "p5": root / "p5.jsonl",
        "p6": root / "p6.jsonl",
        "output": root / "selection.jsonl",
        "metadata": root / "selection.metadata.json",
    }
    _write_jsonl(paths["catalog"], list(products.values()))
    for name in ("public", "p1", "p5", "p6"):
        _write_jsonl(paths[name], rows[name])
    return paths, rows, products


def _hash_kwargs(paths: dict[str, Path]) -> dict[str, str]:
    p1_rows = [
        json.loads(line)
        for line in paths["p1"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "expected_catalog_sha256": hashlib.sha256(
            paths["catalog"].read_bytes()
        ).hexdigest(),
        "expected_public_git_blob_sha1": git_blob_sha1(paths["public"]),
        "expected_p1_samples_sha256": _samples_sha256(p1_rows),
        "expected_output_sha256": None,
        "expected_p5_samples_sha256": _samples_sha256(
            [json.loads(line) for line in paths["p5"].read_text(encoding="utf-8").splitlines()]
        ),
        "expected_p6_samples_sha256": _samples_sha256(
            [json.loads(line) for line in paths["p6"].read_text(encoding="utf-8").splitlines()]
        ),
    }


def _path_build_kwargs(paths: dict[str, Path]) -> dict:
    return {
        "count": 5,
        "seed": "p7-path-seed",
        "expected_catalog_count": 30,
        "expected_public_count": 2,
        "expected_p1_count": 2,
        "expected_p5_count": 2,
        "expected_p6_count": 2,
        **_hash_kwargs(paths),
    }


class P7SelectionCorpusTest(unittest.TestCase):
    def test_builder_is_deterministic_stratified_and_quadruple_disjoint(self) -> None:
        products = _products(1_000)
        public = _batch("public_", 0, 10)
        p1 = _batch("derived_p1_", 10, 10)
        p5 = _batch("derived_p5_", 20, 10)
        p6 = _batch("derived_p6_", 30, 10)

        first, first_metadata = build_p7_selection_corpus(
            public, p1, p5, p6, products, 200, "p7-test-seed"
        )
        second, second_metadata = build_p7_selection_corpus(
            public, p1, p5, p6, products, 200, "p7-test-seed"
        )

        self.assertEqual(first, second)
        self.assertEqual(first_metadata, second_metadata)
        self.assertTrue(
            all(sample["sample_id"].startswith(SAMPLE_ID_PREFIX) for sample in first)
        )
        selected = {sample["ground_truth"]["parent_asin"] for sample in first}
        self.assertEqual(len(selected), 200)
        self.assertFalse(selected & {f"P{index:04d}" for index in range(40)})
        self.assertEqual(first_metadata["scenario_counts"], EXPECTED_SCENARIO_COUNTS)
        self.assertTrue(
            all(value == 0 for value in first_metadata["target_overlaps"].values())
        )
        self.assertEqual(
            set(first_metadata["exclusions"]["pairwise_input_target_overlaps"]),
            {
                "released_public__prior_p1_derived",
                "released_public__prior_p5_derived",
                "released_public__prior_p6_derived",
                "prior_p1_derived__prior_p5_derived",
                "prior_p1_derived__prior_p6_derived",
                "prior_p5_derived__prior_p6_derived",
            },
        )
        self.assertTrue(
            all(
                value == 0
                for value in first_metadata["exclusions"][
                    "pairwise_input_target_overlaps"
                ].values()
            )
        )
        self.assertIn("not a private-distribution proxy", first_metadata["boundary"])
        canonical = "".join(
            json.dumps(
                sample,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for sample in first
        ).encode("utf-8")
        self.assertEqual(
            first_metadata["samples_sha256"], hashlib.sha256(canonical).hexdigest()
        )

    def test_builder_rejects_each_pairwise_input_overlap(self) -> None:
        products = _products(50)
        batches = [
            _batch("public_", 0, 2),
            _batch("derived_p1_", 2, 2),
            _batch("derived_p5_", 4, 2),
            _batch("derived_p6_", 6, 2),
        ]
        for left, right in combinations(range(4), 2):
            with self.subTest(left=left, right=right):
                changed = [[dict(row) for row in batch] for batch in batches]
                changed[right][0] = {
                    **changed[right][0],
                    "ground_truth": dict(changed[left][0]["ground_truth"]),
                }
                with self.assertRaisesRegex(ValueError, "input targets overlap"):
                    build_p7_selection_corpus(
                        changed[0],
                        changed[1],
                        changed[2],
                        changed[3],
                        products,
                        3,
                        "overlap-seed",
                    )

    def test_builder_rejects_wrong_or_duplicate_prior_sample_ids(self) -> None:
        products = _products(50)
        batches = [
            _batch("public_", 0, 2),
            _batch("derived_p1_", 2, 2),
            _batch("derived_p5_", 4, 2),
            _batch("derived_p6_", 6, 2),
        ]
        for index, label in enumerate(("released public", "P1", "P5", "P6")):
            with self.subTest(label=label):
                changed = [[dict(row) for row in batch] for batch in batches]
                changed[index][0]["sample_id"] = "wrong_0001"
                with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
                    build_p7_selection_corpus(
                        changed[0],
                        changed[1],
                        changed[2],
                        changed[3],
                        products,
                        3,
                        "id-seed",
                    )

    def test_path_builder_writes_exact_hash_and_four_way_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _, _ = _fixture(Path(directory))
            metadata = build_and_write_p7_selection_corpus(
                paths["catalog"],
                paths["public"],
                paths["p1"],
                paths["p5"],
                paths["p6"],
                paths["output"],
                paths["metadata"],
                **_path_build_kwargs(paths),
            )
            written = paths["output"].read_bytes()
            stored = json.loads(paths["metadata"].read_text(encoding="utf-8"))

        self.assertEqual(hashlib.sha256(written).hexdigest(), metadata["samples_sha256"])
        self.assertEqual(stored, metadata)
        self.assertTrue(metadata["catalog_source"]["expected_count_verified"])
        self.assertEqual(metadata["exclusions"]["combined_unique_target_count"], 8)
        self.assertTrue(
            metadata["input_sources"]["released_public"][
                "frozen_git_blob_verified"
            ]
        )
        self.assertTrue(
            all(
                metadata["input_sources"][name]["frozen_samples_sha256_verified"]
                for name in (
                    "prior_p1_derived",
                    "prior_p5_derived",
                    "prior_p6_derived",
                )
            )
        )
        self.assertTrue(all(value == 0 for value in metadata["target_overlaps"].values()))

    def test_path_builder_rejects_hash_drift_for_each_frozen_input(self) -> None:
        for name in ("catalog", "public", "p1", "p5", "p6"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                paths, _, _ = _fixture(Path(directory))
                kwargs = _path_build_kwargs(paths)
                key = (
                    "expected_public_git_blob_sha1" if name == "public" else
                    "expected_catalog_sha256" if name == "catalog" else
                    f"expected_{name}_samples_sha256"
                )
                kwargs[key] = "0" * (40 if name == "public" else 64)
                pattern = (
                    "released_public.*Git blob mismatch" if name == "public" else
                    "catalog.*SHA-256 mismatch" if name == "catalog" else
                    f"prior_{name}_derived.*canonical sample SHA-256 mismatch"
                )
                with self.assertRaisesRegex(ValueError, pattern):
                    build_and_write_p7_selection_corpus(
                        paths["catalog"],
                        paths["public"],
                        paths["p1"],
                        paths["p5"],
                        paths["p6"],
                        paths["output"],
                        paths["metadata"],
                        **kwargs,
                    )
                self.assertFalse(paths["output"].exists())
                self.assertFalse(paths["metadata"].exists())

    def test_path_builder_rejects_p1_canonical_or_output_drift_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _, _ = _fixture(Path(directory))
            kwargs = _path_build_kwargs(paths)
            kwargs["expected_p1_samples_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                ValueError, "prior_p1_derived frozen canonical sample"
            ):
                build_and_write_p7_selection_corpus(
                    paths["catalog"], paths["public"], paths["p1"], paths["p5"],
                    paths["p6"], paths["output"], paths["metadata"], **kwargs
                )
            self.assertFalse(paths["output"].exists())

        with tempfile.TemporaryDirectory() as directory:
            paths, _, _ = _fixture(Path(directory))
            kwargs = _path_build_kwargs(paths)
            kwargs["expected_output_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "P7 frozen output"):
                build_and_write_p7_selection_corpus(
                    paths["catalog"], paths["public"], paths["p1"], paths["p5"],
                    paths["p6"], paths["output"], paths["metadata"], **kwargs
                )
            self.assertFalse(paths["output"].exists())
            self.assertFalse(paths["metadata"].exists())

    def test_path_builder_rejects_all_output_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _, _ = _fixture(Path(directory))
            frozen_inputs = [paths[name] for name in ("catalog", "public", "p1", "p5", "p6")]
            for collision in frozen_inputs:
                with self.subTest(collision=collision.name):
                    with self.assertRaisesRegex(ValueError, "must not overwrite"):
                        build_and_write_p7_selection_corpus(
                            paths["catalog"],
                            paths["public"],
                            paths["p1"],
                            paths["p5"],
                            paths["p6"],
                            collision,
                            paths["metadata"],
                            **_path_build_kwargs(paths),
                        )
            with self.assertRaisesRegex(ValueError, "must be different"):
                build_and_write_p7_selection_corpus(
                    paths["catalog"],
                    paths["public"],
                    paths["p1"],
                    paths["p5"],
                    paths["p6"],
                    paths["output"],
                    paths["output"],
                    **_path_build_kwargs(paths),
                )

    def test_real_frozen_hash_constants_match_required_digests(self) -> None:
        self.assertEqual(
            CATALOG_FROZEN_SHA256,
            "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67",
        )
        self.assertEqual(
            PUBLIC_FROZEN_GIT_BLOB_SHA1,
            "121dbec9c1368c81cd887d6959e62507512139c0",
        )
        self.assertEqual(
            P1_FROZEN_SHA256,
            "265a6dae0d9029d54333fbce980b23981b5332d967fc2b450924b05443cadc46",
        )
        self.assertEqual(
            P1_FROZEN_SAMPLES_SHA256,
            "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae",
        )
        self.assertEqual(
            P5_FROZEN_SHA256,
            "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c",
        )
        self.assertEqual(P5_FROZEN_SAMPLES_SHA256, P5_FROZEN_SHA256)
        self.assertEqual(
            P6_FROZEN_SHA256,
            "27544cdb6ed9495808c35bbab09b4dbadcb88a1d75d162f17bb4fba6ee8841c7",
        )
        self.assertEqual(P6_FROZEN_SAMPLES_SHA256, P6_FROZEN_SHA256)
        self.assertEqual(
            P7_FROZEN_SHA256,
            "bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546",
        )

    def test_cli_defaults_and_argument_forwarding(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.prior_p1, DEFAULT_P1_PATH)
        self.assertEqual(args.prior_p5, DEFAULT_P5_PATH)
        self.assertEqual(args.prior_p6, DEFAULT_P6_PATH)
        self.assertEqual(args.count, DEFAULT_COUNT)
        self.assertEqual(args.seed, DEFAULT_SEED)
        self.assertEqual(args.output, DEFAULT_OUTPUT)
        self.assertEqual(args.metadata_output, DEFAULT_METADATA_OUTPUT)

        metadata = {
            "sample_count": 7,
            "target_overlaps": {
                "released_public": 0,
                "prior_p1_derived": 0,
                "prior_p5_derived": 0,
                "prior_p6_derived": 0,
            },
            "samples_sha256": "a" * 64,
        }
        with patch(
            "scripts.build_p7_selection_corpus.build_and_write_p7_selection_corpus",
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
                    "--prior-p6",
                    "p6.jsonl",
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
                Path("p6.jsonl"),
                Path("out.jsonl"),
                Path("meta.json"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
