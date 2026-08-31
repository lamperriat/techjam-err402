from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from starter.attributes import AttributeValue, ProductAttributeView
from starter import oov_chargram_bridge_g0 as bridge
from scripts import oov_chargram_bridge_g0_worker as worker
from scripts import probe_oov_chargram_bridge_g0 as probe


def _record(
    slot: str,
    value: str,
    *,
    polarity: int = 1,
    hardness: str = "hard",
    version: int = 2,
    status: str = "active",
) -> SimpleNamespace:
    return SimpleNamespace(
        slot=slot,
        value=value,
        polarity=polarity,
        hardness=hardness,
        source_turn=1,
        version=version,
        status=status,
    )


def _attribute(
    value: str,
    *,
    source: str = "features",
    confidence: float = 1.0,
) -> AttributeValue:
    return AttributeValue(
        value=value,
        source=source,
        confidence=confidence,
        raw=value,
    )


def _prefix(count: int = 100) -> tuple[str, ...]:
    return tuple(f"P{index:09d}" for index in range(count))


def _product(identifier: str, title: str, category: str = "dress") -> dict[str, object]:
    return {
        "parent_asin": identifier,
        "title": title,
        "categories": [category],
        "features": [title, "casual"],
        "details": {"Style": "casual"},
        "store": "synthetic",
        "description": title,
    }


def _write_catalog(root: Path) -> Path:
    path = root / "catalog.jsonl"
    products = [
        _product(identifier, f"Basic cotton dress {index}")
        for index, identifier in enumerate(_prefix())
    ]
    products.extend(
        (
            _product("Z000000001", "Leather Sneakers", "shoes"),
            _product("Z000000002", "Leather Sandals", "shoes"),
            _product("Z000000003", "Canvas Sneaker Cleaner", "shoes"),
        )
    )
    path.write_text(
        "".join(json.dumps(product, sort_keys=True) + "\n" for product in products),
        encoding="utf-8",
        newline="\n",
    )
    return path


class CanonicalSourceTests(unittest.TestCase):
    def test_category_typo_and_exact_typo_are_distinct_sources(self) -> None:
        sources = bridge.canonical_source_records(
            category_text="sneekers",
            active_terms=("sneekers", "lether"),
            excluded_terms=(),
        )

        self.assertIn(
            (bridge.UNKNOWN_CATEGORY_TOKEN, "sneekers"),
            tuple((item.kind, item.token) for item in sources),
        )
        self.assertIn(
            (bridge.EXACT_ACTIVE_TOKEN, "sneekers"),
            tuple((item.kind, item.token) for item in sources),
        )
        self.assertIn(
            (bridge.EXACT_ACTIVE_TOKEN, "lether"),
            tuple((item.kind, item.token) for item in sources),
        )

    def test_registry_components_negatives_short_and_noise_are_excluded(self) -> None:
        sources = bridge.canonical_source_records(
            category_text="sneakers",
            active_terms=("blue lether", "new balnce", "want"),
            excluded_terms=("lether", "sequns"),
        )
        values = tuple((item.kind, item.token) for item in sources)

        self.assertNotIn((bridge.EXACT_ACTIVE_TOKEN, "blue"), values)
        self.assertNotIn((bridge.EXACT_ACTIVE_TOKEN, "want"), values)
        self.assertNotIn((bridge.EXACT_ACTIVE_TOKEN, "new"), values)
        self.assertIn((bridge.EXACT_ACTIVE_TOKEN, "lether"), values)
        self.assertIn((bridge.EXACT_ACTIVE_TOKEN, "balnce"), values)
        self.assertFalse(any(token == "sequns" for _kind, token in values))

    def test_boundary_trigrams_are_unique_sorted_and_bounded(self) -> None:
        self.assertEqual(
            bridge.boundary_trigrams("aaaa"),
            ("^aa", "aa$", "aaa"),
        )
        with self.assertRaises(ValueError):
            bridge.boundary_trigrams("ab")

    def test_bounded_levenshtein(self) -> None:
        self.assertEqual(bridge.levenshtein_distance("sneekers", "sneakers", 2), 1)
        self.assertEqual(bridge.levenshtein_distance("lether", "leather", 1), 1)
        self.assertEqual(bridge.levenshtein_distance("abcd", "wxyz", 1), 2)


class HardMaskOracleTests(unittest.TestCase):
    def test_every_supported_slot_matches_parent_on_disjoint_and_unknown_evidence(self) -> None:
        cases = {
            "category": ("dress", "shoe"),
            "audience": ("women", "men"),
            "material": ("linen", "leather"),
            "color": ("blue", "red"),
            "closure": ("zipper", "button"),
            "style": ("casual", "formal"),
            "use_case": ("hiking", "wedding"),
            "size": ("small", "large"),
            "width": ("wide", "narrow"),
        }
        for slot, (requested, observed) in cases.items():
            with self.subTest(slot=slot):
                records = (_record(slot, requested, version=4),)
                category = requested if slot == "category" else "other"
                actual_rules = bridge.compile_hard_conflict_rules(
                    category_text=category,
                    active_terms=(requested,),
                    excluded_terms=(),
                    current_version=4,
                    records=records,
                )
                source = "categories" if slot == "category" else "features"
                views = {
                    "disjoint": ProductAttributeView(
                        parent_asin="disjoint",
                        **{slot: (_attribute(observed, source=source),)},
                    ),
                    "unknown": ProductAttributeView(parent_asin="unknown"),
                }
                actual = bridge.apply_hard_conflict_mask(
                    tuple(views), views, actual_rules
                )

                self.assertIn(slot, dict(actual_rules.positive))
                self.assertEqual(actual_rules.negative, ())
                self.assertEqual(actual.identifiers, ("unknown",))
                self.assertEqual(actual.dropped, ("disjoint",))
                self.assertEqual(actual.positive_conflict_count, 1)

    def test_every_allowed_negative_slot_matches_parent(self) -> None:
        cases = {
            "audience": "women",
            "material": "leather",
            "color": "red",
            "closure": "zipper",
            "style": "formal",
            "use_case": "wedding",
        }
        for slot, forbidden in cases.items():
            with self.subTest(slot=slot):
                records = (_record(slot, forbidden, polarity=-1, version=5),)
                actual_rules = bridge.compile_hard_conflict_rules(
                    category_text="other",
                    active_terms=(),
                    excluded_terms=(forbidden,),
                    current_version=5,
                    records=records,
                )
                views = {
                    "violating": ProductAttributeView(
                        parent_asin="violating",
                        **{slot: (_attribute(forbidden),)},
                    ),
                    "unknown": ProductAttributeView(parent_asin="unknown"),
                }
                actual = bridge.apply_hard_conflict_mask(
                    tuple(views), views, actual_rules
                )

                self.assertIn((slot, forbidden), actual_rules.negative)
                self.assertEqual(actual_rules.positive, ())
                self.assertEqual(actual.identifiers, ("unknown",))
                self.assertEqual(actual.dropped, ("violating",))
                self.assertEqual(actual.negative_violation_count, 1)

    def test_rules_and_mask_are_field_for_field_parent_oracle_equivalent(self) -> None:
        fixtures = (
            (
                "women sandals",
                ("linen",),
                ("red",),
                2,
                (
                    _record("color", "red", polarity=-1),
                    _record("material", "linen"),
                    _record("style", "casual", hardness="soft"),
                    _record("color", "blue", version=1),
                ),
            ),
            (
                "other",
                ("blue",),
                (),
                3,
                (
                    _record("color", "blue", version=3),
                    _record("material", "leather", status="superseded", version=3),
                ),
            ),
        )
        views = {
            "known": ProductAttributeView(
                parent_asin="known",
                category=(_attribute("sandal", source="categories"),),
                material=(_attribute("linen"),),
                color=(_attribute("blue"),),
            ),
            "violating": ProductAttributeView(
                parent_asin="violating",
                category=(_attribute("dress", source="categories"),),
                material=(_attribute("leather"),),
                color=(_attribute("red"),),
            ),
            "unknown": ProductAttributeView(parent_asin="unknown"),
        }
        identifiers = tuple(views)

        for category, active, excluded, version, records in fixtures:
            with self.subTest(category=category, version=version):
                actual_rules = bridge.compile_hard_conflict_rules(
                    category_text=category,
                    active_terms=active,
                    excluded_terms=excluded,
                    current_version=version,
                    records=records,
                )
                actual = bridge.apply_hard_conflict_mask(
                    identifiers, views, actual_rules
                )
                repeated_rules = bridge.compile_hard_conflict_rules(
                    category_text=category,
                    active_terms=active,
                    excluded_terms=excluded,
                    current_version=version,
                    records=records,
                )
                repeated = bridge.apply_hard_conflict_mask(
                    identifiers, views, repeated_rules
                )
                self.assertEqual(actual_rules, repeated_rules)
                self.assertEqual(actual, repeated)

    def test_current_hard_positive_and_negative_rules_match_parent_contract(self) -> None:
        rules = bridge.compile_hard_conflict_rules(
            category_text="other",
            active_terms=("red",),
            excluded_terms=("leather",),
            current_version=2,
            records=(
                _record("color", "red", version=2),
                _record("material", "leather", polarity=-1, version=2),
                _record("color", "blue", version=1),
                _record("style", "casual", hardness="soft", version=2),
            ),
        )
        self.assertEqual(dict(rules.positive)["color"], ("red",))
        self.assertIn(("material", "leather"), rules.negative)
        self.assertNotIn("style", dict(rules.positive))

        views = {
            "blue": ProductAttributeView(
                parent_asin="blue",
                color=(_attribute("blue"),),
            ),
            "leather": ProductAttributeView(
                parent_asin="leather",
                color=(_attribute("red"),),
                material=(_attribute("leather"),),
            ),
            "unknown": ProductAttributeView(parent_asin="unknown"),
        }
        result = bridge.apply_hard_conflict_mask(
            ("blue", "leather", "unknown"), views, rules
        )
        self.assertEqual(result.identifiers, ("unknown",))
        self.assertEqual(result.dropped, ("blue", "leather"))
        self.assertEqual(result.negative_violation_count, 1)
        self.assertEqual(result.positive_conflict_count, 1)

    def test_pinned_isolated_exhaustive_oracle_and_synthetic_gate(self) -> None:
        self.assertNotIn("starter.sparse_multiview_g0", sys.modules)
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
        ):
            first = probe._oracle_differential_summary()
            second = probe._oracle_differential_summary()
        self.assertEqual(first, second)
        self.assertEqual(first["fixture_count"], probe.ORACLE_FIXTURE_COUNT)
        self.assertEqual(first["comparison_count"], probe.ORACLE_COMPARISON_COUNT)
        self.assertEqual(first["matrix_sha256"], probe.ORACLE_MATRIX_SHA256)
        self.assertIs(first["algorithm_fts_cache_privacy_pass"], True)
        self.assertIs(first["isolated_namespace"], True)
        self.assertNotIn("starter.sparse_multiview_g0", sys.modules)


class StableAdmissionTests(unittest.TestCase):
    def test_prefix_is_untouched_and_tail_is_deduplicated_and_capped(self) -> None:
        prefix = _prefix()
        ranked = tuple(prefix[:3]) + tuple(f"Z{index:09d}" for index in range(250))
        candidates = bridge.stable_append_candidates(prefix, ranked)

        self.assertEqual(candidates[: len(prefix)], prefix)
        self.assertEqual(len(candidates), 292)
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertFalse(set(prefix) & set(candidates[len(prefix) :]))

    def test_prefix_validation_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            bridge.stable_append_candidates(("duplicate", "duplicate"), ("new",))


class ExpanderLifecycleTests(unittest.TestCase):
    def test_default_off_never_opens_catalog_and_returns_prefix(self) -> None:
        prefix = _prefix()
        expander = bridge.OovChargramBridgeG0Expander(
            "does-not-exist.jsonl", enabled=False
        )
        try:
            result = expander.expand(
                prefix,
                category_text="sneekers",
                active_terms=("lether",),
                excluded_terms=(),
                current_version=1,
                records=(),
            )
            self.assertEqual(result.candidates, prefix)
            self.assertEqual(result.prefix, prefix)
            self.assertEqual(result.tail, ())
            self.assertFalse(result.enabled)
        finally:
            expander.close()
        self.assertTrue(expander.closed)

    def test_small_catalog_maps_oov_executes_single_route_and_preserves_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _write_catalog(Path(directory))
            prefix = _prefix()
            with bridge.OovChargramBridgeG0Expander(
                catalog, enabled=True, cache_enabled=True
            ) as expander:
                expander.validate()
                first = expander.expand(
                    prefix,
                    category_text="sneakerss",
                    active_terms=("leatherr",),
                    excluded_terms=(),
                    current_version=1,
                    records=(),
                )
                second = expander.expand(
                    prefix,
                    category_text="sneakerss",
                    active_terms=("leatherr",),
                    excluded_terms=(),
                    current_version=1,
                    records=(),
                )
                cache = expander.cache_diagnostics()

                self.assertTrue(first.activated)
                self.assertEqual(first.candidates[: len(prefix)], prefix)
                self.assertIn("Z000000001", first.tail)
                self.assertTrue(first.routes)
                self.assertTrue(all(len(route.identifiers) <= 32 for route in first.routes))
                self.assertTrue(
                    all(route.expression.startswith("{") for route in first.routes)
                )
                self.assertTrue(
                    all(
                        route.source.source.token not in route.expression
                        for route in first.routes
                    )
                )
                self.assertEqual(first.candidates, second.candidates)
                self.assertTrue(first.query_only_readback_one)
                self.assertTrue(first.controlled_write_rejected)
                self.assertGreater(cache["oov_bridge"]["hits"], 0)
                self.assertGreater(cache["fts_route"]["hits"], 0)
                self.assertEqual(first.tail_conflict_count, 0)

    def test_exact_dice_gate_rejects_low_overlap_edit_one_typos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _write_catalog(Path(directory))
            prefix = _prefix()
            with bridge.OovChargramBridgeG0Expander(
                catalog, enabled=True, cache_enabled=False
            ) as expander:
                result = expander.expand(
                    prefix,
                    category_text="sneekers",
                    active_terms=("lether",),
                    excluded_terms=(),
                    current_version=1,
                    records=(),
                )

                self.assertTrue(result.source_records)
                self.assertFalse(result.activated)
                self.assertEqual(result.bridge_sources, ())
                self.assertEqual(result.candidates, prefix)

    def test_field_present_tokens_are_excluded_before_source_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _write_catalog(Path(directory))
            with bridge.OovChargramBridgeG0Expander(
                catalog, enabled=True, cache_enabled=False
            ) as expander:
                result = expander.expand(
                    _prefix(),
                    category_text="other",
                    active_terms=("leather", "leatherr"),
                    excluded_terms=(),
                    current_version=1,
                    records=(),
                )

                self.assertNotIn("leather", {item.token for item in result.source_records})
                self.assertIn("leatherr", {item.token for item in result.source_records})

    def test_module_validate(self) -> None:
        bridge.validate()


class WorkerContractTests(unittest.TestCase):
    def test_four_layer_cache_pair_and_old_shape_rejection(self) -> None:
        before = {
            name: {
                "capacity": capacity,
                "size": 1,
                "hits": 2,
                "misses": 1,
                "inserts": 1,
                "evictions": 0,
                "closed": False,
            }
            for name, capacity in worker.CACHE_CAPACITIES.items()
        }
        after = {
            name: {**values, "size": 0, "closed": True}
            for name, values in before.items()
        }
        normalized_before, normalized_after = worker._cache_pair_contract(
            before, after
        )
        self.assertEqual(set(normalized_before), set(worker.CACHE_CAPACITIES))
        self.assertTrue(all(item["closed"] for item in normalized_after.values()))

        incomplete = dict(before)
        incomplete.pop("oov_bridge")
        with self.assertRaises(worker.OovChargramBridgeG0WorkerError):
            worker._cache_contract(incomplete, after_close=False)

    def test_failure_receipt_is_exact_finite_and_identifier_free(self) -> None:
        progress = worker.WorkerProgress(
            phase="TRAJECTORY",
            stage_id="preclaim_synthetic",
            session_limit=100,
            last_completed_session=7,
        )
        try:
            worker.validate_expansion_result(None, _prefix(), frozenset(_prefix()))
        except BaseException as error:
            receipt = worker._error_receipt(error, progress)
        else:
            self.fail("synthetic contract failure did not fail")

        self.assertEqual(
            set(receipt),
            {
                "child_exit_code",
                "failure_origin",
                "failure_site_id",
                "kind",
                "progress_bucket",
                "rss_bucket",
                "schema_version",
                "stack_hash",
                "stage_id",
                "status",
                "stderr_nonempty",
                "wall_time_bucket",
                "worker_error_code",
                "worker_phase",
            },
        )
        self.assertEqual(receipt["schema_version"], worker.FAILURE_SCHEMA_VERSION)
        self.assertEqual(receipt["progress_bucket"], "PARTIAL")
        self.assertEqual(receipt["worker_error_code"], "EXPANSION_CONTRACT")
        self.assertNotIn("nonce", receipt)
        self.assertNotIn("traceback", receipt)
        self.assertEqual(len(str(receipt["stack_hash"])), 64)
        self.assertEqual(
            probe._validate_worker_failure_receipt(
                receipt, expected_stage_id="preclaim_synthetic"
            ),
            receipt,
        )
        with self.assertRaises(probe.SparseUnionProbeError):
            probe._validate_worker_failure_receipt(
                {**receipt, "child_exit_code": True},
                expected_stage_id="preclaim_synthetic",
            )

    def test_worker_result_validator_accepts_single_route_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _write_catalog(Path(directory))
            identifiers = frozenset((*_prefix(), "Z000000001", "Z000000002", "Z000000003"))
            with bridge.OovChargramBridgeG0Expander(
                catalog, enabled=True, cache_enabled=False
            ) as expander:
                result = expander.expand(
                    _prefix(),
                    category_text="sneakerss",
                    active_terms=("leatherr",),
                    excluded_terms=(),
                    current_version=1,
                    records=(),
                )
                candidates = worker.validate_expansion_result(
                    result, _prefix(), identifiers
                )
                lookup, route, mask = worker.expansion_timing_contract(result)

        self.assertEqual(candidates, result.candidates)
        self.assertGreater(lookup, 0)
        self.assertGreater(route, 0)
        self.assertGreater(mask, 0)


if __name__ == "__main__":
    unittest.main()
