from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import socket
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from scripts.build_amazon_validation_proxy import (
    ProxyBuildConfig,
    _exclusive_publish,
    build_proxy,
    load_config,
    main,
)


SPLIT_FILES = {
    "train_explore": "proxy_train_explore.jsonl",
    "calibration": "proxy_calibration.jsonl",
    "selection": "proxy_selection.jsonl",
    "confirmation": "proxy_confirmation.sealed.jsonl",
}
SCENARIO_COUNTS = {
    "buying": 8,
    "browsing": 8,
    "intent_override": 3,
    "boundary": 1,
}
RAW_SOURCE_KEYS = {"user_id", "rating", "timestamp", "history"}


def _asin(index: int) -> str:
    return f"B{index:09d}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _product(index: int) -> dict:
    taxonomies = (
        ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Dresses"],
        ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Sneakers"],
        ["Clothing, Shoes & Jewelry", "Jewelry", "Necklaces"],
        ["Clothing, Shoes & Jewelry", "Accessories", "Belts"],
    )
    colors = ("blue", "black", "silver", "brown")
    materials = ("cotton", "mesh", "sterling silver", "leather")
    color = colors[index % len(colors)]
    material = materials[index % len(materials)]
    return {
        "parent_asin": _asin(index),
        "title": f"{color} {material} catalog product {index}",
        "features": [f"{color} finish", f"{material} material"],
        "description": ["Frozen catalog-only description."],
        "price": 20.0 + index,
        "categories": taxonomies[index % len(taxonomies)],
        "details": {"Color": color, "Material": material},
        "average_rating": 4.0,
        "rating_number": index + 1,
        "store": "Fixture Store",
    }


def _validation_rows(count: int = 180, *, mutated: bool = False) -> list[dict[str, str]]:
    history = " ".join([_asin(216), _asin(217), _asin(218)])
    rows = [
        {
            "user_id": f"{'changed' if mutated else 'raw'}-user-{index:04d}",
            "parent_asin": _asin(index),
            "rating": "1.23456789" if mutated else "4.87654321",
            "timestamp": str((1800000000000 if mutated else 1700000000000) + index),
            "history": history,
        }
        for index in range(count)
    ]
    duplicate = dict(rows[10])
    duplicate["user_id"] = "changed-duplicate-user" if mutated else "raw-duplicate-user"
    rows.append(duplicate)
    return list(reversed(rows)) if mutated else rows


def _write_validation(path: Path, rows: list[dict[str, str]], *, extra_column: bool = False) -> None:
    fieldnames = ["user_id", "parent_asin", "rating", "timestamp", "history"]
    if extra_column:
        fieldnames.append("review_text")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            if extra_column:
                payload["review_text"] = "RAW-REVIEW-MUST-NOT-BE-USED"
            writer.writerow(payload)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        nested: set[str] = set()
        for item in value.values():
            nested.update(_all_keys(item))
        return set(value) | nested
    if isinstance(value, list):
        nested = set()
        for item in value:
            nested.update(_all_keys(item))
        return nested
    return set()


def _targets(rows: list[dict]) -> set[str]:
    return {str(row["ground_truth"]["parent_asin"]) for row in rows}


class ProxyFixture:
    def __init__(
        self,
        root: Path,
        *,
        source_count: int = 180,
        mutated: bool = False,
        extra_column: bool = False,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.catalog = root / "catalog.jsonl"
        self.validation = root / "Clothing_Shoes_and_Jewelry.valid.csv"
        self.public = root / "public_set.jsonl"
        self.manual = root / "manual_exclusions.json"
        self.consumed = root / "consumed.jsonl"
        self.output = root / "proxy"

        _canonical_jsonl(self.catalog, [_product(index) for index in range(220)])
        _write_validation(
            self.validation,
            _validation_rows(source_count, mutated=mutated),
            extra_column=extra_column,
        )
        _canonical_jsonl(
            self.public,
            [
                {"sample_id": "public_1", "ground_truth": {"parent_asin": _asin(0)}},
                {"sample_id": "public_2", "ground_truth": {"parent_asin": _asin(1)}},
            ],
        )
        self.manual.write_text(
            json.dumps(
                {
                    "schema_version": "track4.p12-manual-exclusions.v1",
                    "targets": [_asin(2)],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _canonical_jsonl(
            self.consumed,
            [
                {"sample_id": "consumed_1", "ground_truth": {"parent_asin": _asin(3)}},
                {"sample_id": "consumed_2", "ground_truth": {"parent_asin": _asin(4)}},
            ],
        )

    def config(self, *, output: Path | None = None, seed: str = "proxy-fixture-v1") -> ProxyBuildConfig:
        return ProxyBuildConfig(
            validation_csv=self.validation,
            catalog_path=self.catalog,
            public_path=self.public,
            manual_exclusions_path=self.manual,
            consumed_corpora=(self.consumed,),
            output_dir=output or self.output,
            seed=seed,
            split_counts={name: 20 for name in SPLIT_FILES},
            expected_validation_sha256=_sha256(self.validation),
            expected_catalog_sha256=_sha256(self.catalog),
            expected_public_sha256=_sha256(self.public),
            expected_manual_exclusions_sha256=_sha256(self.manual),
            expected_consumed_sha256=(_sha256(self.consumed),),
        )


class AmazonValidationProxyTests(unittest.TestCase):
    def _assert_no_outputs(self, output: Path) -> None:
        for filename in (*SPLIT_FILES.values(), "manifest.json", "audit.json"):
            self.assertFalse((output / filename).exists(), filename)

    def test_builds_group_disjoint_excluded_private_safe_splits_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProxyFixture(Path(directory))
            with patch.object(
                socket, "create_connection", side_effect=AssertionError("network attempted")
            ):
                manifest = build_proxy(fixture.config())

            stored_manifest = json.loads(
                (fixture.output / "manifest.json").read_text(encoding="utf-8")
            )
            audit_path = fixture.output / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(stored_manifest, manifest)
            self.assertEqual(manifest["schema_version"], "track4.amazon-validation-proxy.v1")
            self.assertEqual(manifest["source"]["role"], "validation")
            self.assertEqual(manifest["source"]["test_rows_read"], 0)
            self.assertEqual(manifest["source"]["sha256"], _sha256(fixture.validation))
            self.assertEqual(manifest["catalog"]["sha256"], _sha256(fixture.catalog))
            self.assertEqual(manifest["exclusions"]["public_target_count"], 2)
            self.assertEqual(manifest["exclusions"]["manual_target_count"], 1)
            self.assertEqual(manifest["exclusions"]["consumed_target_count"], 2)
            self.assertEqual(manifest["exclusions"]["union_target_count"], 5)
            self.assertEqual(manifest["exclusions"]["output_overlap_count"], 0)
            self.assertEqual(manifest["audit"]["filename"], "audit.json")
            self.assertEqual(manifest["audit"]["sha256"], _sha256(audit_path))
            self.assertEqual(
                audit["schema_version"], "track4.amazon-validation-proxy-audit.v1"
            )
            self.assertTrue(all(audit["checks"].values()))
            self.assertEqual(audit["coverage"]["proxy_rows"], 80)

            split_targets: dict[str, set[str]] = {}
            forbidden_values = {
                "raw-duplicate-user",
                _asin(216),
                _asin(217),
                _asin(218),
                "4.87654321",
            }
            for split, filename in SPLIT_FILES.items():
                path = fixture.output / filename
                rows = _load_jsonl(path)
                split_targets[split] = _targets(rows)
                self.assertEqual(len(rows), 20)
                self.assertGreater(len(split_targets[split]), 0)
                self.assertLessEqual(len(split_targets[split]), 20)
                self.assertEqual(Counter(row["scenario_type"] for row in rows), SCENARIO_COUNTS)
                self.assertEqual(manifest["splits"][split]["filename"], filename)
                self.assertEqual(manifest["splits"][split]["rows"], 20)
                self.assertEqual(
                    manifest["splits"][split]["unique_targets"],
                    len(split_targets[split]),
                )
                self.assertEqual(manifest["splits"][split]["sha256"], _sha256(path))
                self.assertEqual(
                    manifest["splits"][split]["sealed"], split == "confirmation"
                )
                self.assertEqual(manifest["splits"][split]["scenario_counts"], SCENARIO_COUNTS)
                serialized = path.read_text(encoding="utf-8")
                for forbidden in forbidden_values:
                    self.assertNotIn(forbidden, serialized)
                for row in rows:
                    self.assertTrue(RAW_SOURCE_KEYS.isdisjoint(_all_keys(row)))
                    self.assertEqual(
                        set(row["user_profile"]),
                        {
                            "purchase_frequency",
                            "average_prior_rating",
                            "rating_style",
                            "preference_tags",
                            "summary",
                        },
                    )
                    self.assertIsNone(row["user_profile"]["average_prior_rating"])
                    self.assertEqual(row["user_profile"]["rating_style"], "unknown")
                    self.assertIn(
                        row["taxonomy"]["group"],
                        {"clothing", "shoes", "jewelry", "accessories-other"},
                    )
                    self.assertIsInstance(row["taxonomy"]["leaf_path"], list)
                    self.assertTrue(row["taxonomy"]["leaf_path"])

            names = list(split_targets)
            for index, left in enumerate(names):
                for right in names[index + 1 :]:
                    self.assertFalse(split_targets[left] & split_targets[right])
            all_targets = set().union(*split_targets.values())
            self.assertEqual(len(all_targets), sum(map(len, split_targets.values())))
            self.assertFalse(all_targets & {_asin(index) for index in range(5)})

    def test_repeated_targets_are_allowed_within_but_never_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProxyFixture(Path(directory))
            history = " ".join([_asin(216), _asin(217), _asin(218)])
            repeated_rows = [
                {
                    "user_id": f"repeat-user-{target}-{occurrence}",
                    "parent_asin": _asin(target),
                    "rating": "5.0",
                    "timestamp": str(1700000000000 + occurrence),
                    "history": history,
                }
                for target in (80, 84, 88, 92)
                for occurrence in range(25)
            ]
            _write_validation(fixture.validation, repeated_rows)
            manifest = build_proxy(fixture.config())
            audit = json.loads(
                (fixture.output / "audit.json").read_text(encoding="utf-8")
            )

            targets_by_split: dict[str, set[str]] = {}
            for split, filename in SPLIT_FILES.items():
                rows = _load_jsonl(fixture.output / filename)
                targets = [str(row["ground_truth"]["parent_asin"]) for row in rows]
                targets_by_split[split] = set(targets)
                self.assertEqual(len(rows), 20)
                self.assertEqual(len(targets_by_split[split]), 1)
                self.assertEqual(manifest["splits"][split]["unique_targets"], 1)
                self.assertEqual(
                    audit["split_distributions"][split]["repeated_target_rows"], 19
                )

            names = list(targets_by_split)
            for index, left in enumerate(names):
                for right in names[index + 1 :]:
                    self.assertFalse(targets_by_split[left] & targets_by_split[right])
            self.assertTrue(audit["checks"]["pairwise_target_overlap_zero"])

    def test_tracked_nested_config_maps_without_opening_large_assets(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "amazon_validation_proxy_v1.json"
        )
        with patch(
            "scripts.build_amazon_validation_proxy._verify_file",
            side_effect=AssertionError("load_config attempted to verify a large asset"),
        ):
            config = load_config(config_path)

        self.assertTrue(config.production_pinned)
        self.assertEqual(
            config.loaded_config_canonical_sha256,
            _canonical_json_sha256(config_path),
        )
        self.assertEqual(
            Path(config.validation_csv).as_posix(),
            "data/external/amazon_reviews_2023/Clothing_Shoes_and_Jewelry.valid.csv",
        )
        self.assertEqual(Path(config.catalog_path).as_posix(), "data/catalog.jsonl")
        self.assertEqual(Path(config.public_path).as_posix(), "data/public_set.jsonl")
        self.assertEqual(
            Path(config.manual_exclusions_path).as_posix(),
            "configs/p12_manual_target_exclusions.json",
        )
        self.assertEqual(Path(config.output_dir).as_posix(), "experiments/fast_track/proxy_v1")
        self.assertEqual(config.split_counts, {name: 2000 for name in SPLIT_FILES})
        self.assertEqual(
            config.scenario_counts,
            {name: {key: value * 100 for key, value in SCENARIO_COUNTS.items()} for name in SPLIT_FILES},
        )
        self.assertEqual(len(config.consumed_corpora), 15)
        self.assertFalse(
            any(str(value.get("name")) == "released_public" for value in config.consumed_corpora)
        )
        self.assertEqual(config.expected_consumed_union_count, 2720)
        self.assertEqual(config.expected_manual_count, 0)
        self.assertEqual(config.expected_validation_bytes, 345027412)
        self.assertTrue(config.source_url.endswith("Clothing_Shoes_and_Jewelry.valid.csv"))
        self.assertNotIn(".test.", config.source_url.casefold())
        self.assertEqual(Path(config.config_path), config_path)

    def test_tampered_nested_production_declarations_fail_during_load(self) -> None:
        tracked_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "amazon_validation_proxy_v1.json"
        )
        tracked_value = json.loads(tracked_path.read_text(encoding="utf-8"))

        def remove_registry_entry(value: dict) -> None:
            value["exclusions"]["consumed_corpora"].pop()

        cases = {
            "source_header": lambda value: value["source"].__setitem__(
                "header", "parent_asin,user_id,rating,timestamp,history"
            ),
            "source_official_sha256_pin": lambda value: value["source"].__setitem__(
                "sha256", "0" * 64
            ),
            "output_path": lambda value: value["outputs"]["splits"][
                "selection"
            ].__setitem__("path", "experiments/fast_track/proxy_v1/escaped.jsonl"),
            "confirmation_sealed": lambda value: value["outputs"]["splits"][
                "confirmation"
            ].__setitem__("sealed", False),
            "total_rows": lambda value: value["outputs"].__setitem__(
                "total_rows", 7_999
            ),
            "file_hash_mode": lambda value: value["exclusions"].__setitem__(
                "file_hash_mode", "SHA-256 over raw bytes"
            ),
            "consumed_registry": remove_registry_entry,
            "manual_schema": lambda value: value["exclusions"][
                "manual_target_ledger"
            ].__setitem__("schema_version", "track4.p12-manual-exclusions.v0"),
        }

        for name, mutate in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                value = json.loads(json.dumps(tracked_value))
                mutate(value)
                config_path = Path(directory) / "amazon_validation_proxy_v1.json"
                config_path.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                with patch(
                    "scripts.build_amazon_validation_proxy._verify_file",
                    side_effect=AssertionError("tampered config reached asset verification"),
                ):
                    with self.assertRaisesRegex(
                        ValueError, "(?i)invalid pinned production config|pinned production"
                    ):
                        load_config(config_path)

    def test_cli_rejects_flat_config_before_build_or_asset_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "flat-proxy-config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "validation_csv": str(root / "must-not-open.valid.csv"),
                        "catalog_path": str(root / "must-not-open-catalog.jsonl"),
                        "public_path": str(root / "must-not-open-public.jsonl"),
                        "manual_exclusions_path": None,
                        "output_dir": str(root / "must-not-create-output"),
                        "seed": "flat-config-must-not-run",
                        "split_counts": {name: 20 for name in SPLIT_FILES},
                        "expected_validation_sha256": "0" * 64,
                        "expected_catalog_sha256": "0" * 64,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )

            with patch(
                "scripts.build_amazon_validation_proxy.build_proxy",
                side_effect=AssertionError("flat CLI config reached build_proxy"),
            ) as build:
                with self.assertRaisesRegex(
                    ValueError, "(?i)CLI accepts only the pinned production proxy config"
                ):
                    main(["--config", str(config_path)])
            build.assert_not_called()
            self.assertFalse((root / "must-not-create-output").exists())

    def test_programmatic_production_forgery_fails_before_asset_verification(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "amazon_validation_proxy_v1.json"
        )
        production = load_config(config_path)
        cases = {
            "missing_config_path": (
                replace(production, config_path=None),
                "(?i)tracked pinned proxy config path",
            ),
            "replaced_seed": (
                replace(production, seed="programmatic-production-forgery"),
                "(?i)fields differ from the tracked pinned proxy config",
            ),
            "replaced_output_dir": (
                replace(production, output_dir="experiments/fast_track/forged-proxy"),
                "(?i)fields differ from the tracked pinned proxy config",
            ),
        }

        for name, (forged, message) in cases.items():
            with self.subTest(case=name), patch(
                "scripts.build_amazon_validation_proxy._verify_file",
                side_effect=AssertionError("forged production config reached asset verification"),
            ) as verify:
                with self.assertRaisesRegex(ValueError, message):
                    build_proxy(forged)
                verify.assert_not_called()

    def test_exclusive_publish_rolls_back_when_a_later_hard_link_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destinations = [root / f"output-{index}.json" for index in range(4)]
            payloads = {
                destination: f'{{"index":{index}}}\n'.encode("utf-8")
                for index, destination in enumerate(destinations)
            }
            real_link = os.link
            link_calls = 0

            def fail_third_link(source: Path, destination: Path) -> None:
                nonlocal link_calls
                link_calls += 1
                if link_calls == 3:
                    raise OSError("injected third-link publication failure")
                real_link(source, destination)

            with patch(
                "scripts.build_amazon_validation_proxy.os.link",
                side_effect=fail_third_link,
            ):
                with self.assertRaisesRegex(
                    OSError, "injected third-link publication failure"
                ):
                    _exclusive_publish(payloads)

            self.assertEqual(link_calls, 3)
            self.assertEqual(list(root.iterdir()), [])

    def test_test_split_path_is_rejected_before_any_content_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProxyFixture(Path(directory))
            test_path = fixture.validation.with_name("Clothing_Shoes_and_Jewelry.test.csv")
            test_path.write_bytes(fixture.validation.read_bytes())
            config = replace(
                fixture.config(),
                validation_csv=test_path,
                expected_validation_sha256=_sha256(test_path),
            )
            with patch("builtins.open", side_effect=AssertionError("test file opened")), patch(
                "pathlib.Path.open", side_effect=AssertionError("test file opened")
            ):
                with self.assertRaisesRegex(ValueError, "(?i)test|validation"):
                    build_proxy(config)
            self._assert_no_outputs(fixture.output)

    def test_split_bytes_are_deterministic_and_ignore_raw_source_only_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = ProxyFixture(root / "first")
            second = ProxyFixture(root / "second", mutated=True)
            first_config = first.config(output=first.output)
            repeat_output = root / "repeat"
            build_proxy(first_config)
            build_proxy(replace(first_config, output_dir=repeat_output))
            build_proxy(second.config())

            for filename in (*SPLIT_FILES.values(), "audit.json", "manifest.json"):
                self.assertEqual(
                    (first.output / filename).read_bytes(),
                    (repeat_output / filename).read_bytes(),
                    filename,
                )
            for filename in (*SPLIT_FILES.values(), "audit.json"):
                self.assertEqual(
                    (first.output / filename).read_bytes(),
                    (second.output / filename).read_bytes(),
                    filename,
                )

            changed_seed_output = root / "changed-seed"
            build_proxy(
                replace(first_config, output_dir=changed_seed_output, seed="proxy-fixture-v2")
            )
            self.assertTrue(
                any(
                    (first.output / filename).read_bytes()
                    != (changed_seed_output / filename).read_bytes()
                    for filename in SPLIT_FILES.values()
                )
            )

    def test_tracked_identical_evidence_allows_missing_splits_but_nothing_else(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ProxyFixture(root / "source")
            config = fixture.config()
            build_proxy(config)
            evidence = {
                filename: (fixture.output / filename).read_bytes()
                for filename in ("manifest.json", "audit.json")
            }

            fresh_output = root / "fresh-checkout"
            fresh_output.mkdir()
            for filename, payload in evidence.items():
                (fresh_output / filename).write_bytes(payload)
            build_proxy(replace(config, output_dir=fresh_output))
            for filename, payload in evidence.items():
                self.assertEqual((fresh_output / filename).read_bytes(), payload)
            for filename in SPLIT_FILES.values():
                self.assertTrue((fresh_output / filename).is_file())

            differing_output = root / "differing-evidence"
            differing_output.mkdir()
            (differing_output / "manifest.json").write_bytes(
                evidence["manifest.json"] + b"\n"
            )
            differing_bytes = (differing_output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                build_proxy(replace(config, output_dir=differing_output))
            self.assertEqual(
                (differing_output / "manifest.json").read_bytes(), differing_bytes
            )
            self.assertFalse((differing_output / "audit.json").exists())
            for filename in SPLIT_FILES.values():
                self.assertFalse((differing_output / filename).exists(), filename)

            blocked_output = root / "existing-derived-row"
            blocked_output.mkdir()
            existing_split = blocked_output / SPLIT_FILES["selection"]
            shutil.copyfile(fixture.output / SPLIT_FILES["selection"], existing_split)
            original = existing_split.read_bytes()
            with self.assertRaises(FileExistsError):
                build_proxy(replace(config, output_dir=blocked_output))
            self.assertEqual(existing_split.read_bytes(), original)
            for filename in (
                *(
                    value
                    for value in SPLIT_FILES.values()
                    if value != SPLIT_FILES["selection"]
                ),
                "manifest.json",
                "audit.json",
            ):
                self.assertFalse((blocked_output / filename).exists(), filename)

    def test_source_hash_schema_and_consumed_hash_drift_fail_without_outputs(self) -> None:
        cases = ("validation_hash", "catalog_hash", "consumed_hash", "schema")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = ProxyFixture(Path(directory), extra_column=case == "schema")
                config = fixture.config()
                if case == "validation_hash":
                    config = replace(config, expected_validation_sha256="0" * 64)
                elif case == "catalog_hash":
                    config = replace(config, expected_catalog_sha256="0" * 64)
                elif case == "consumed_hash":
                    config = replace(config, expected_consumed_sha256=("0" * 64,))
                with self.assertRaisesRegex(ValueError, "(?i)sha|schema|column"):
                    build_proxy(config)
                self._assert_no_outputs(fixture.output)

    def test_source_row_capacity_shortfall_fails_without_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProxyFixture(Path(directory), source_count=70)
            with self.assertRaisesRegex(
                ValueError, "(?i)source sessions|source capacity|eligible|shortfall|80"
            ):
                build_proxy(fixture.config())
            self._assert_no_outputs(fixture.output)

    def test_empty_manual_ledger_is_allowed_but_missing_ledger_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProxyFixture(Path(directory))
            fixture.manual.write_text(
                json.dumps(
                    {
                        "schema_version": "track4.p12-manual-exclusions.v1",
                        "targets": [],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            config = replace(
                fixture.config(),
                expected_manual_exclusions_sha256=_sha256(fixture.manual),
            )
            manifest = build_proxy(config)
            self.assertEqual(manifest["exclusions"]["manual_target_count"], 0)

            missing_output = Path(directory) / "missing-output"
            missing = fixture.manual.with_name("missing-manual-exclusions.txt")
            missing_config = replace(
                config,
                output_dir=missing_output,
                manual_exclusions_path=missing,
            )
            with self.assertRaises((FileNotFoundError, ValueError)):
                build_proxy(missing_config)
            self._assert_no_outputs(missing_output)


if __name__ == "__main__":
    unittest.main()
