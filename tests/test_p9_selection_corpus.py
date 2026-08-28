from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

from scripts.build_p9_selection_corpus import (
    CONFIRMATION_SAMPLE_ID_PREFIX,
    DEFAULT_CONFIRMATION_OUTPUT,
    DEFAULT_CONFIRMATION_SEED,
    DEFAULT_COUNT,
    DEFAULT_METADATA_OUTPUT,
    DEFAULT_P1_PATH,
    DEFAULT_P5_PATH,
    DEFAULT_P6_PATH,
    DEFAULT_P7_PATH,
    DEFAULT_P8_CONFIRMATION_PATH,
    DEFAULT_P8_SELECTION_PATH,
    DEFAULT_SELECTION_OUTPUT,
    DEFAULT_SELECTION_SEED,
    EXPECTED_SCENARIO_COUNTS,
    MIN_NEGATIVE_SUPPORT,
    P9_CONFIRMATION_FROZEN_SAMPLES_SHA256,
    P9_SELECTION_FROZEN_SAMPLES_SHA256,
    SELECTION_SAMPLE_ID_PREFIX,
    _parser,
    _samples_sha256,
    build_and_write_p9_selection_corpora,
    build_p9_selection_corpora,
    main,
)
from scripts.verify_official_assets import git_blob_sha1


def _sample(sample_id: str, parent_asin: str) -> dict:
    return {
        "sample_id": sample_id,
        "ground_truth": {"parent_asin": parent_asin},
    }


def _product(index: int) -> dict:
    identifier = f"P{index:05d}"
    colors = ("blue", "black", "white", "green")
    materials = ("cotton", "linen", "nylon", "leather")
    styles = ("casual", "classic", "sporty", "elegant")
    audiences = ("women", "men", "girls", "boys")
    closures = ("button", "zipper", "buckle", "snap")
    uses = ("running", "office", "hiking", "travel")
    bucket = index % 4
    return {
        "parent_asin": identifier,
        "title": f"{colors[bucket]} {materials[bucket]} shirt {index}",
        "features": [f"{styles[bucket]} {uses[bucket]} design"],
        "description": ["Description is not constraint evidence."],
        "price": 20.0 + index,
        "categories": ["Clothing", "Shirts"],
        "details": {
            "Material": materials[bucket],
            "Color": colors[bucket],
            "Style": styles[bucket],
            "Department": audiences[bucket],
            "Closure": closures[bucket],
        },
        "average_rating": 4.0,
        "rating_number": 10,
        "store": "Fixture",
    }


def _products(count: int) -> dict[str, dict]:
    products = [_product(index) for index in range(count)]
    return {product["parent_asin"]: product for product in products}


def _batch(prefix: str, start: int, count: int) -> list[dict]:
    return [
        _sample(f"{prefix}{index + 1:04d}", f"P{start + index:05d}")
        for index in range(count)
    ]


def _batches(count: int = 2) -> list[list[dict]]:
    prefixes = (
        "public_",
        "derived_p1_",
        "derived_p5_",
        "derived_p6_",
        "derived_p7_",
        "derived_p8_selection_",
        "derived_p8_confirmation_",
    )
    return [
        _batch(prefix, index * count, count)
        for index, prefix in enumerate(prefixes)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(root: Path) -> tuple[dict[str, Path], dict[str, list[dict]]]:
    names = ("public", "p1", "p5", "p6", "p7", "p8s", "p8c")
    rows = dict(zip(names, _batches(), strict=True))
    paths = {
        "catalog": root / "catalog.jsonl",
        **{name: root / f"{name}.jsonl" for name in names},
        "selection": root / "selection.jsonl",
        "confirmation": root / "confirmation.jsonl",
        "metadata": root / "corpora.metadata.json",
    }
    _write_jsonl(paths["catalog"], list(_products(60).values()))
    for name in names:
        _write_jsonl(paths[name], rows[name])
    return paths, rows


def _path_kwargs(paths: dict[str, Path]) -> dict:
    names = ("public", "p1", "p5", "p6", "p7", "p8s", "p8c")
    rows = {
        name: [
            json.loads(line)
            for line in paths[name].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for name in names
    }
    return {
        "count": 5,
        "selection_seed": "p9-selection-test",
        "confirmation_seed": "p9-confirmation-test",
        "expected_catalog_count": 60,
        "expected_public_count": 2,
        "expected_p1_count": 2,
        "expected_p5_count": 2,
        "expected_p6_count": 2,
        "expected_p7_count": 2,
        "expected_p8_selection_count": 2,
        "expected_p8_confirmation_count": 2,
        "expected_catalog_sha256": hashlib.sha256(
            paths["catalog"].read_bytes()
        ).hexdigest(),
        "expected_public_git_blob_sha1": git_blob_sha1(paths["public"]),
        "expected_public_samples_sha256": _samples_sha256(rows["public"]),
        "expected_p1_samples_sha256": _samples_sha256(rows["p1"]),
        "expected_p5_samples_sha256": _samples_sha256(rows["p5"]),
        "expected_p6_samples_sha256": _samples_sha256(rows["p6"]),
        "expected_p7_samples_sha256": _samples_sha256(rows["p7"]),
        "expected_p8_selection_samples_sha256": _samples_sha256(rows["p8s"]),
        "expected_p8_confirmation_samples_sha256": _samples_sha256(rows["p8c"]),
        "expected_selection_output_sha256": None,
        "expected_confirmation_output_sha256": None,
    }


def _run_path_builder(paths: dict[str, Path], **kwargs: object) -> dict:
    return build_and_write_p9_selection_corpora(
        paths["catalog"],
        paths["public"],
        paths["p1"],
        paths["p5"],
        paths["p6"],
        paths["p7"],
        paths["p8s"],
        paths["p8c"],
        paths["selection"],
        paths["confirmation"],
        paths["metadata"],
        **kwargs,
    )


class P9SelectionCorpusTest(unittest.TestCase):
    def test_builder_is_deterministic_stratified_and_disjoint_from_all_inputs(self) -> None:
        products = _products(1_000)
        batches = _batches(10)
        first = build_p9_selection_corpora(
            *batches, products, 200, "selection-seed", "confirmation-seed"
        )
        second = build_p9_selection_corpora(
            *batches, products, 200, "selection-seed", "confirmation-seed"
        )

        self.assertEqual(first, second)
        selection, confirmation, metadata = first
        self.assertEqual(len(selection), 200)
        self.assertEqual(len(confirmation), 200)
        self.assertEqual(
            metadata["corpora"]["selection"]["scenario_counts"],
            EXPECTED_SCENARIO_COUNTS,
        )
        self.assertEqual(
            metadata["corpora"]["confirmation"]["scenario_counts"],
            EXPECTED_SCENARIO_COUNTS,
        )
        selection_targets = {
            sample["ground_truth"]["parent_asin"] for sample in selection
        }
        confirmation_targets = {
            sample["ground_truth"]["parent_asin"] for sample in confirmation
        }
        excluded = {
            sample["ground_truth"]["parent_asin"]
            for batch in batches
            for sample in batch
        }
        self.assertEqual(len(selection_targets), 200)
        self.assertEqual(len(confirmation_targets), 200)
        self.assertFalse(selection_targets & confirmation_targets)
        self.assertFalse((selection_targets | confirmation_targets) & excluded)
        self.assertTrue(
            all(
                sample["sample_id"].startswith(SELECTION_SAMPLE_ID_PREFIX)
                for sample in selection
            )
        )
        self.assertTrue(
            all(
                sample["sample_id"].startswith(CONFIRMATION_SAMPLE_ID_PREFIX)
                for sample in confirmation
            )
        )
        pairwise = metadata["exclusions"]["pairwise_input_target_overlaps"]
        self.assertEqual(len(pairwise), 21)
        self.assertTrue(all(value == 0 for value in pairwise.values()))
        self.assertTrue(
            all(
                value == 0
                for value in metadata["exclusions"][
                    "selected_target_overlaps"
                ].values()
            )
        )

    def test_every_materialized_negative_keeps_p8_catalog_only_gates(self) -> None:
        products = _products(700)
        selection, confirmation, metadata = build_p9_selection_corpora(
            *_batches(5), products, 200, "negative-a", "negative-b"
        )

        for sample in [*selection, *confirmation]:
            audit = sample["behavior"]["explicit_negative"]
            self.assertGreaterEqual(
                audit["catalog_document_support"], MIN_NEGATIVE_SUPPORT
            )
            self.assertIn(audit["category_bucket_level"], {"leaf", "coarse"})
            self.assertFalse(audit["description_used_as_evidence"])
        self.assertEqual(
            metadata["generator"]["category_bucket_fallback_order"],
            ["leaf", "coarse"],
        )
        self.assertFalse(metadata["generator"]["global_category_fallback_used"])
        self.assertFalse(metadata["generator"]["agent_used"])
        self.assertFalse(metadata["generator"]["fts_used"])
        self.assertFalse(metadata["generator"]["prior_results_used"])
        self.assertFalse(metadata["generator"]["prior_metrics_used"])

    def test_builder_rejects_all_twenty_one_input_overlaps(self) -> None:
        products = _products(100)
        batches = _batches()
        for left, right in combinations(range(7), 2):
            with self.subTest(left=left, right=right):
                changed = [[dict(row) for row in batch] for batch in batches]
                changed[right][0] = {
                    **changed[right][0],
                    "ground_truth": dict(changed[left][0]["ground_truth"]),
                }
                with self.assertRaisesRegex(ValueError, "input targets overlap"):
                    build_p9_selection_corpora(
                        *changed, products, 2, "overlap-a", "overlap-b"
                    )

    def test_path_builder_writes_canonical_aggregate_only_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = _fixture(Path(directory))
            metadata = _run_path_builder(paths, **_path_kwargs(paths))
            selection_payload = paths["selection"].read_bytes()
            confirmation_payload = paths["confirmation"].read_bytes()
            stored = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for path in (paths["selection"], paths["confirmation"])
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            temporary_files = list(Path(directory).glob("*.tmp"))

        self.assertEqual(stored, metadata)
        self.assertEqual(
            hashlib.sha256(selection_payload).hexdigest(),
            metadata["corpora"]["selection"]["samples_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(confirmation_payload).hexdigest(),
            metadata["corpora"]["confirmation"]["samples_sha256"],
        )
        self.assertEqual(
            metadata["exclusions"]["combined_unique_input_target_count"], 14
        )
        self.assertTrue(
            all(
                source["frozen_samples_sha256_verified"]
                for source in metadata["input_sources"].values()
            )
        )
        serialized = json.dumps(metadata, sort_keys=True)
        self.assertTrue(
            all(row["ground_truth"]["parent_asin"] not in serialized for row in rows)
        )
        self.assertFalse(temporary_files)

    def test_path_builder_rejects_both_p8_hash_drifts_before_write(self) -> None:
        cases = (
            "expected_p8_selection_samples_sha256",
            "expected_p8_confirmation_samples_sha256",
        )
        for key in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                paths, _ = _fixture(Path(directory))
                kwargs = _path_kwargs(paths)
                kwargs[key] = "0" * 64
                with self.assertRaisesRegex(
                    ValueError, "prior_p8_.*canonical sample SHA-256 mismatch"
                ):
                    _run_path_builder(paths, **kwargs)
                self.assertFalse(paths["selection"].exists())
                self.assertFalse(paths["confirmation"].exists())
                self.assertFalse(paths["metadata"].exists())

    def test_path_builder_rejects_input_and_output_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = _fixture(Path(directory))
            frozen_inputs = [
                paths[name]
                for name in (
                    "catalog", "public", "p1", "p5", "p6", "p7", "p8s", "p8c"
                )
            ]
            original_selection = paths["selection"]
            for collision in frozen_inputs:
                with self.subTest(collision=collision.name):
                    paths["selection"] = collision
                    with self.assertRaisesRegex(ValueError, "must not overwrite"):
                        _run_path_builder(paths, **_path_kwargs(paths))
            paths["selection"] = original_selection

            original_confirmation = paths["confirmation"]
            paths["confirmation"] = paths["selection"]
            with self.assertRaisesRegex(ValueError, "outputs must differ"):
                _run_path_builder(paths, **_path_kwargs(paths))
            paths["confirmation"] = original_confirmation

    def test_cli_defaults_and_forwarding(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.prior_p1, DEFAULT_P1_PATH)
        self.assertEqual(args.prior_p5, DEFAULT_P5_PATH)
        self.assertEqual(args.prior_p6, DEFAULT_P6_PATH)
        self.assertEqual(args.prior_p7, DEFAULT_P7_PATH)
        self.assertEqual(args.prior_p8_selection, DEFAULT_P8_SELECTION_PATH)
        self.assertEqual(args.prior_p8_confirmation, DEFAULT_P8_CONFIRMATION_PATH)
        self.assertEqual(args.count, DEFAULT_COUNT)
        self.assertEqual(args.selection_seed, DEFAULT_SELECTION_SEED)
        self.assertEqual(args.confirmation_seed, DEFAULT_CONFIRMATION_SEED)
        self.assertEqual(args.selection_output, DEFAULT_SELECTION_OUTPUT)
        self.assertEqual(args.confirmation_output, DEFAULT_CONFIRMATION_OUTPUT)
        self.assertEqual(args.metadata_output, DEFAULT_METADATA_OUTPUT)

        metadata = {
            "corpora": {
                "selection": {"sample_count": 7, "samples_sha256": "a" * 64},
                "confirmation": {"sample_count": 7, "samples_sha256": "b" * 64},
            }
        }
        with patch(
            "scripts.build_p9_selection_corpus.build_and_write_p9_selection_corpora",
            return_value=metadata,
        ) as build:
            exit_code = main(
                [
                    "--count", "7",
                    "--selection-seed", "selection",
                    "--confirmation-seed", "confirmation",
                    "--prior-p8-selection", "p8s.jsonl",
                    "--prior-p8-confirmation", "p8c.jsonl",
                    "--selection-output", "selection.jsonl",
                    "--confirmation-output", "confirmation.jsonl",
                    "--metadata-output", "metadata.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            build.call_args.kwargs,
            {
                "count": 7,
                "selection_seed": "selection",
                "confirmation_seed": "confirmation",
            },
        )
        self.assertEqual(
            build.call_args.args[6:],
            (
                Path("p8s.jsonl"),
                Path("p8c.jsonl"),
                Path("selection.jsonl"),
                Path("confirmation.jsonl"),
                Path("metadata.json"),
            ),
        )

    def test_real_output_hash_constants_are_frozen(self) -> None:
        self.assertEqual(
            P9_SELECTION_FROZEN_SAMPLES_SHA256,
            "6298cbd6d7507f4b163ab4979a86ff109e0dffa90557e3b28e5d20d129e5be9f",
        )
        self.assertEqual(
            P9_CONFIRMATION_FROZEN_SAMPLES_SHA256,
            "4bbd9d53f32e3773de18bab881ba6e5ef0887ca86701897798ee086430ed08d9",
        )


if __name__ == "__main__":
    unittest.main()
