from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

from scripts.build_p8_selection_corpus import (
    CATALOG_FROZEN_SHA256,
    CONFIRMATION_SAMPLE_ID_PREFIX,
    DEFAULT_CONFIRMATION_OUTPUT,
    DEFAULT_CONFIRMATION_SEED,
    DEFAULT_COUNT,
    DEFAULT_METADATA_OUTPUT,
    DEFAULT_P1_PATH,
    DEFAULT_P5_PATH,
    DEFAULT_P6_PATH,
    DEFAULT_P7_PATH,
    DEFAULT_SELECTION_OUTPUT,
    DEFAULT_SELECTION_SEED,
    EXPECTED_SCENARIO_COUNTS,
    MIN_NEGATIVE_SUPPORT,
    NEGATIVE_TEMPLATES,
    NEGATIVE_VOCABULARIES,
    P1_FROZEN_SAMPLES_SHA256,
    P5_FROZEN_SAMPLES_SHA256,
    P6_FROZEN_SAMPLES_SHA256,
    P7_FROZEN_SAMPLES_SHA256,
    P8_CONFIRMATION_FROZEN_SAMPLES_SHA256,
    P8_SELECTION_FROZEN_SAMPLES_SHA256,
    PUBLIC_FROZEN_GIT_BLOB_SHA1,
    PUBLIC_FROZEN_SAMPLES_SHA256,
    SELECTION_SAMPLE_ID_PREFIX,
    _bucket_document_frequencies,
    _constraint_plan,
    _product_evidence,
    _parser,
    _samples_sha256,
    build_and_write_p8_selection_corpora,
    build_p8_selection_corpora,
    main,
)
from scripts.verify_official_assets import git_blob_sha1
from starter.agent import _parse_turn
from starter.attributes import build_product_attribute_view
from starter.p8_negative import ALLOWED_NEGATIVE_SLOTS


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
        "description": ["Description is deliberately not constraint evidence."],
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
    return {
        product["parent_asin"]: product
        for index in range(count)
        if (product := _product(index))
    }


def _batch(prefix: str, start: int, count: int) -> list[dict]:
    return [
        _sample(f"{prefix}{index + 1:04d}", f"P{start + index:05d}")
        for index in range(count)
    ]


def _batches(count: int = 2) -> list[list[dict]]:
    return [
        _batch("public_", 0, count),
        _batch("derived_p1_", count, count),
        _batch("derived_p5_", count * 2, count),
        _batch("derived_p6_", count * 3, count),
        _batch("derived_p7_", count * 4, count),
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(root: Path) -> tuple[dict[str, Path], dict[str, list[dict]], dict]:
    batch_list = _batches()
    rows = dict(zip(("public", "p1", "p5", "p6", "p7"), batch_list, strict=True))
    products = _products(40)
    paths = {
        "catalog": root / "catalog.jsonl",
        "public": root / "public.jsonl",
        "p1": root / "p1.jsonl",
        "p5": root / "p5.jsonl",
        "p6": root / "p6.jsonl",
        "p7": root / "p7.jsonl",
        "selection": root / "selection.jsonl",
        "confirmation": root / "confirmation.jsonl",
        "metadata": root / "corpora.metadata.json",
    }
    _write_jsonl(paths["catalog"], list(products.values()))
    for name in ("public", "p1", "p5", "p6", "p7"):
        _write_jsonl(paths[name], rows[name])
    return paths, rows, products


def _path_kwargs(paths: dict[str, Path]) -> dict:
    rows = {
        name: [
            json.loads(line)
            for line in paths[name].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for name in ("public", "p1", "p5", "p6", "p7")
    }
    return {
        "count": 5,
        "selection_seed": "p8-selection-test",
        "confirmation_seed": "p8-confirmation-test",
        "expected_catalog_count": 40,
        "expected_public_count": 2,
        "expected_p1_count": 2,
        "expected_p5_count": 2,
        "expected_p6_count": 2,
        "expected_p7_count": 2,
        "expected_catalog_sha256": hashlib.sha256(
            paths["catalog"].read_bytes()
        ).hexdigest(),
        "expected_public_git_blob_sha1": git_blob_sha1(paths["public"]),
        "expected_public_samples_sha256": _samples_sha256(rows["public"]),
        "expected_p1_samples_sha256": _samples_sha256(rows["p1"]),
        "expected_p5_samples_sha256": _samples_sha256(rows["p5"]),
        "expected_p6_samples_sha256": _samples_sha256(rows["p6"]),
        "expected_p7_samples_sha256": _samples_sha256(rows["p7"]),
        "expected_selection_output_sha256": None,
        "expected_confirmation_output_sha256": None,
    }


def _run_path_builder(paths: dict[str, Path], **kwargs: object) -> dict:
    return build_and_write_p8_selection_corpora(
        paths["catalog"],
        paths["public"],
        paths["p1"],
        paths["p5"],
        paths["p6"],
        paths["p7"],
        paths["selection"],
        paths["confirmation"],
        paths["metadata"],
        **kwargs,
    )


class P8SelectionCorpusTest(unittest.TestCase):
    def test_builder_is_deterministic_stratified_and_fully_disjoint(self) -> None:
        products = _products(1_500)
        batches = _batches(20)

        first = build_p8_selection_corpora(
            *batches,
            products,
            200,
            "selection-seed",
            "confirmation-seed",
        )
        second = build_p8_selection_corpora(
            *batches,
            products,
            200,
            "selection-seed",
            "confirmation-seed",
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
        self.assertTrue(
            all(
                value == 0
                for value in metadata["exclusions"][
                    "pairwise_input_target_overlaps"
                ].values()
            )
        )
        self.assertTrue(
            all(
                value == 0
                for value in metadata["exclusions"][
                    "selected_target_overlaps"
                ].values()
            )
        )

    def test_materialized_constraints_are_supported_and_catalog_consistent(self) -> None:
        products = _products(700)
        batches = _batches(5)
        selection, confirmation, metadata = build_p8_selection_corpora(
            *batches, products, 200, "negative-a", "negative-b"
        )

        seen_templates: set[str] = set()
        for sample in [*selection, *confirmation]:
            audit = sample["behavior"]["explicit_negative"]
            seen_templates.add(audit["template"])
            self.assertIn(audit["template"], NEGATIVE_TEMPLATES)
            self.assertEqual(
                audit["phrase"],
                audit["template"].format(value=audit["excluded_value"]),
            )
            self.assertNotIn(" ", audit["excluded_value"])
            self.assertFalse(audit["description_used_as_evidence"])
            self.assertGreaterEqual(
                audit["catalog_document_support"], MIN_NEGATIVE_SUPPORT
            )
            self.assertIn(audit["category_bucket_level"], {"leaf", "coarse"})
            self.assertNotEqual(audit["positive_value"], audit["excluded_value"])
            self.assertNotEqual(audit["positive_evidence_source"], "description")

            target = sample["ground_truth"]["parent_asin"]
            view = build_product_attribute_view(products[target])
            target_values = {item.value for item in getattr(view, audit["slot"])}
            self.assertIn(audit["positive_value"], target_values)
            self.assertNotIn(audit["excluded_value"], target_values)
            self.assertEqual(
                sample["intent_card"]["hard_constraints"], [audit["phrase"]]
            )
            self.assertEqual(
                sample["intent_card"]["soft_preferences"],
                [audit["positive_anchor"]],
            )
            parsed = _parse_turn(
                f"A key requirement is: {audit['phrase']}.",
                current_category="shirts",
            )
            self.assertIn(audit["excluded_value"], parsed.negative_terms)
            if sample["scenario_type"] == "intent_override":
                override = sample["behavior"]["override"]
                self.assertIn(audit["phrase"], override["new_value"])
                self.assertIn(audit["positive_anchor"], override["new_value"])

        self.assertEqual(seen_templates, set(NEGATIVE_TEMPLATES))
        self.assertTrue(metadata["generator"]["intent_fields_pre_materialized"])
        self.assertFalse(metadata["generator"]["agent_used"])
        self.assertFalse(metadata["generator"]["fts_used"])
        self.assertFalse(metadata["generator"]["prior_results_used"])
        self.assertFalse(metadata["generator"]["description_used_as_evidence"])
        self.assertFalse(metadata["generator"]["global_category_fallback_used"])
        for corpus in metadata["corpora"].values():
            support = corpus["selected_negative_support"]
            self.assertGreaterEqual(support["min"], MIN_NEGATIVE_SUPPORT)
            self.assertLessEqual(support["min"], support["median"])
            self.assertLessEqual(support["median"], support["max"])
            self.assertEqual(support["min_support"], MIN_NEGATIVE_SUPPORT)

    def test_description_only_attribute_is_not_eligible_evidence(self) -> None:
        product = _product(1)
        product["title"] = "Plain shirt"
        product["features"] = []
        product["details"] = {}
        product["categories"] = ["Clothing", "Shirts"]
        product["description"] = [
            "blue cotton casual running women button"
        ]

        self.assertIsNone(_product_evidence(product))

    def test_title_only_low_confidence_attribute_is_not_eligible_evidence(self) -> None:
        product = _product(1)
        product["title"] = "Blue cotton women's running button shirt"
        product["features"] = []
        product["details"] = {}
        product["categories"] = ["Clothing", "Shirts"]
        product["description"] = []

        self.assertIsNone(_product_evidence(product))
        self.assertEqual(set(NEGATIVE_VOCABULARIES), set(ALLOWED_NEGATIVE_SLOTS))

    def test_negative_uses_highest_frequency_value_in_shared_category(self) -> None:
        def color_product(index: int, color: str, description: str = "") -> dict:
            product = _product(index)
            product["title"] = f"Plain shirt {index}"
            product["features"] = []
            product["details"] = {"Color": color}
            product["categories"] = ["Clothing", "Shirts"]
            product["description"] = [description] if description else []
            return product

        products = [color_product(0, "blue", "purple purple purple")]
        products.extend(color_product(index, "red") for index in range(1, 7))
        products.extend(color_product(index, "green") for index in range(7, 10))
        evidence = {
            product["parent_asin"]: _product_evidence(product) for product in products
        }
        self.assertTrue(all(value is not None for value in evidence.values()))
        typed_evidence = {
            key: value for key, value in evidence.items() if value is not None
        }
        frequencies = _bucket_document_frequencies(typed_evidence)

        plan = _constraint_plan(
            typed_evidence["P00000"],
            frequencies,
            "frequency-seed",
            "P00000",
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["negative_value"], "red")
        self.assertEqual(plan["negative_support"], 6)
        self.assertEqual(plan["negative_bucket_level"], "leaf")
        self.assertNotEqual(plan["negative_value"], "purple")

    def test_bucket_fallback_stays_within_shared_reliable_category(self) -> None:
        def categorized_color(index: int, color: str, leaf: str) -> dict:
            product = _product(index)
            product["title"] = f"Plain item {index}"
            product["features"] = []
            product["details"] = {"Color": color}
            product["categories"] = ["Clothing", "Shirts", leaf]
            product["description"] = []
            return product

        target = categorized_color(0, "blue", "T-Shirts")
        peers = [categorized_color(index, "red", "Tops") for index in range(1, 5)]
        unrelated = [
            categorized_color(index, "green", "Earrings")
            for index in range(5, 12)
        ]
        for product in unrelated:
            product["categories"] = ["Clothing", "Jewelry", "Earrings"]
        products = [target, *peers, *unrelated]
        evidence = {
            product["parent_asin"]: value
            for product in products
            if (value := _product_evidence(product)) is not None
        }
        frequencies = _bucket_document_frequencies(evidence)

        plan = _constraint_plan(
            evidence["P00000"], frequencies, "fallback-seed", "P00000"
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["negative_value"], "red")
        self.assertEqual(plan["negative_support"], 4)
        self.assertEqual(plan["negative_bucket_level"], "coarse")

    def test_builder_rejects_every_pairwise_input_overlap(self) -> None:
        products = _products(100)
        batches = _batches()
        for left, right in combinations(range(5), 2):
            with self.subTest(left=left, right=right):
                changed = [[dict(row) for row in batch] for batch in batches]
                changed[right][0] = {
                    **changed[right][0],
                    "ground_truth": dict(changed[left][0]["ground_truth"]),
                }
                with self.assertRaisesRegex(ValueError, "input targets overlap"):
                    build_p8_selection_corpora(
                        *changed,
                        products,
                        2,
                        "overlap-a",
                        "overlap-b",
                    )

    def test_builder_rejects_invalid_ids_missing_targets_and_shortfall(self) -> None:
        products = _products(30)
        batches = _batches()
        for index in range(5):
            with self.subTest(index=index):
                changed = [[dict(row) for row in batch] for batch in batches]
                changed[index][0]["sample_id"] = "wrong_0001"
                with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
                    build_p8_selection_corpora(
                        *changed, products, 2, "invalid-a", "invalid-b"
                    )

        missing = [[dict(row) for row in batch] for batch in batches]
        missing[4][0] = {
            **missing[4][0],
            "ground_truth": {"parent_asin": "NOT_IN_CATALOG"},
        }
        with self.assertRaisesRegex(ValueError, "missing from catalog"):
            build_p8_selection_corpora(
                *missing, products, 2, "missing-a", "missing-b"
            )

        with self.assertRaisesRegex(ValueError, "eligible disjoint targets"):
            build_p8_selection_corpora(
                *batches, products, 11, "short-a", "short-b"
            )

    def test_path_builder_validates_hashes_writes_canonical_and_aggregate_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _, _ = _fixture(Path(directory))
            metadata = _run_path_builder(paths, **_path_kwargs(paths))
            selection_payload = paths["selection"].read_bytes()
            confirmation_payload = paths["confirmation"].read_bytes()
            stored = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            selection_targets = {
                row["ground_truth"]["parent_asin"]
                for row in (
                    json.loads(line)
                    for line in selection_payload.decode("utf-8").splitlines()
                )
            }
            confirmation_targets = {
                row["ground_truth"]["parent_asin"]
                for row in (
                    json.loads(line)
                    for line in confirmation_payload.decode("utf-8").splitlines()
                )
            }
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
        self.assertFalse(selection_targets & confirmation_targets)
        self.assertTrue(metadata["catalog_source"]["frozen_sha256_verified"])
        self.assertTrue(metadata["catalog_source"]["expected_count_verified"])
        self.assertTrue(
            metadata["input_sources"]["released_public"]["frozen_git_blob_verified"]
        )
        self.assertTrue(
            all(
                source["frozen_samples_sha256_verified"]
                for source in metadata["input_sources"].values()
            )
        )
        self.assertFalse(temporary_files)
        serialized_metadata = json.dumps(metadata, sort_keys=True)
        self.assertTrue(
            all(target not in serialized_metadata for target in selection_targets)
        )
        self.assertTrue(
            all(target not in serialized_metadata for target in confirmation_targets)
        )

    def test_path_builder_rejects_each_frozen_hash_before_write(self) -> None:
        cases = (
            ("catalog", "expected_catalog_sha256", 64, "catalog.*SHA-256 mismatch"),
            ("public-blob", "expected_public_git_blob_sha1", 40, "Git blob mismatch"),
            (
                "public-content",
                "expected_public_samples_sha256",
                64,
                "released_public.*canonical sample SHA-256 mismatch",
            ),
            (
                "p1",
                "expected_p1_samples_sha256",
                64,
                "prior_p1_derived.*canonical sample SHA-256 mismatch",
            ),
            (
                "p5",
                "expected_p5_samples_sha256",
                64,
                "prior_p5_derived.*canonical sample SHA-256 mismatch",
            ),
            (
                "p6",
                "expected_p6_samples_sha256",
                64,
                "prior_p6_derived.*canonical sample SHA-256 mismatch",
            ),
            (
                "p7",
                "expected_p7_samples_sha256",
                64,
                "prior_p7_derived.*canonical sample SHA-256 mismatch",
            ),
        )
        for label, key, length, pattern in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                paths, _, _ = _fixture(Path(directory))
                kwargs = _path_kwargs(paths)
                kwargs[key] = "0" * length
                with self.assertRaisesRegex(ValueError, pattern):
                    _run_path_builder(paths, **kwargs)
                self.assertFalse(paths["selection"].exists())
                self.assertFalse(paths["confirmation"].exists())
                self.assertFalse(paths["metadata"].exists())

    def test_path_builder_rejects_frozen_output_drift_before_write(self) -> None:
        for name in ("selection", "confirmation"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                paths, _, _ = _fixture(Path(directory))
                kwargs = _path_kwargs(paths)
                kwargs[f"expected_{name}_output_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, f"P8 {name} frozen output"):
                    _run_path_builder(paths, **kwargs)
                self.assertFalse(paths["selection"].exists())
                self.assertFalse(paths["confirmation"].exists())
                self.assertFalse(paths["metadata"].exists())

    def test_path_builder_rejects_input_and_output_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _, _ = _fixture(Path(directory))
            inputs = [
                paths[name]
                for name in ("catalog", "public", "p1", "p5", "p6", "p7")
            ]
            for collision in inputs:
                with self.subTest(collision=collision.name):
                    original = paths["selection"]
                    paths["selection"] = collision
                    with self.assertRaisesRegex(ValueError, "must not overwrite"):
                        _run_path_builder(paths, **_path_kwargs(paths))
                    paths["selection"] = original

            original_confirmation = paths["confirmation"]
            paths["confirmation"] = paths["selection"]
            with self.assertRaisesRegex(ValueError, "outputs must differ"):
                _run_path_builder(paths, **_path_kwargs(paths))
            paths["confirmation"] = original_confirmation

    def test_atomic_replace_failure_leaves_no_partial_outputs_or_temps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _, _ = _fixture(root)
            with patch(
                "scripts.build_p8_selection_corpus.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    _run_path_builder(paths, **_path_kwargs(paths))

            self.assertFalse(paths["selection"].exists())
            self.assertFalse(paths["confirmation"].exists())
            self.assertFalse(paths["metadata"].exists())
            self.assertFalse(list(root.glob("*.tmp")))

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
            PUBLIC_FROZEN_SAMPLES_SHA256,
            "6c726257fec25575716ee65b095f94c48402b6e14e83341518610f45fbfbec6d",
        )
        self.assertEqual(
            P1_FROZEN_SAMPLES_SHA256,
            "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae",
        )
        self.assertEqual(
            P5_FROZEN_SAMPLES_SHA256,
            "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c",
        )
        self.assertEqual(
            P6_FROZEN_SAMPLES_SHA256,
            "27544cdb6ed9495808c35bbab09b4dbadcb88a1d75d162f17bb4fba6ee8841c7",
        )
        self.assertEqual(
            P7_FROZEN_SAMPLES_SHA256,
            "bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546",
        )
        self.assertEqual(
            P8_SELECTION_FROZEN_SAMPLES_SHA256,
            "1c11d73d7c8ced617ce874e15a563f240731ca9654ed42bcc4f773b7b4da81ee",
        )
        self.assertEqual(
            P8_CONFIRMATION_FROZEN_SAMPLES_SHA256,
            "3ae6f8ff7ab0362399b348c3443daa5b7138aab9cf72e944b7e11dd71d7d3dde",
        )

    def test_cli_defaults_and_argument_forwarding(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.prior_p1, DEFAULT_P1_PATH)
        self.assertEqual(args.prior_p5, DEFAULT_P5_PATH)
        self.assertEqual(args.prior_p6, DEFAULT_P6_PATH)
        self.assertEqual(args.prior_p7, DEFAULT_P7_PATH)
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
            "scripts.build_p8_selection_corpus.build_and_write_p8_selection_corpora",
            return_value=metadata,
        ) as build:
            exit_code = main(
                [
                    "--count", "7",
                    "--selection-seed", "selection",
                    "--confirmation-seed", "confirmation",
                    "--prior-p1", "p1.jsonl",
                    "--prior-p5", "p5.jsonl",
                    "--prior-p6", "p6.jsonl",
                    "--prior-p7", "p7.jsonl",
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
            build.call_args.args[2:],
            (
                Path("p1.jsonl"),
                Path("p5.jsonl"),
                Path("p6.jsonl"),
                Path("p7.jsonl"),
                Path("selection.jsonl"),
                Path("confirmation.jsonl"),
                Path("metadata.json"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
