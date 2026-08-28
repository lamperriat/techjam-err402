from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.agent import Agent
from starter.p5_lab import (
    C00,
    R01,
    S00,
    SCHEMA_VERSION,
    SPECS,
    SPEC_BY_ID,
    P5Agent,
    PrfComputation,
    validate_registry,
)


PRODUCTS = [
    {
        "parent_asin": f"DRESS-{index:02d}",
        "title": (
            f"Women's cotton dress style {index} breathable lightweight"
        ),
        "categories": ["Clothing", "Women", "Dresses"],
        "features": [
            "cotton breathable fabric",
            "lightweight comfortable pockets",
            f"collection {index}",
        ],
        "details": {
            "material": "cotton breathable",
            "fit": "lightweight casual",
        },
        "store": f"Dress Store {index % 3}",
        "description": "A casual summer option for everyday wear.",
    }
    for index in range(16)
] + [
    {
        "parent_asin": "BLACK-RUNNER",
        "title": "Men's black mesh running shoes",
        "categories": ["Clothing", "Men", "Shoes"],
        "features": ["black mesh", "running support"],
        "details": {"material": "mesh", "color": "black"},
        "store": "Runner Store",
    },
    {
        "parent_asin": "SILVER-NECKLACE",
        "title": "Women's silver pendant necklace",
        "categories": ["Clothing", "Women", "Jewelry", "Necklaces"],
        "features": ["silver pendant", "gift box"],
        "details": {"material": "silver"},
        "store": "Jewelry Store",
    },
]


class P5LabTests(unittest.TestCase):
    def _catalog(self, directory: str) -> Path:
        path = Path(directory) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(product) + "\n" for product in PRODUCTS),
            encoding="utf-8",
        )
        return path

    def _feedback_catalog(self, directory: str) -> Path:
        feedback_products = [
            {
                "parent_asin": f"FEEDBACK-{index:02d}",
                "title": "Women's cotton dress breathable lightweight",
                "categories": ["Clothing", "Women", "Dresses"],
                "features": ["breathable fabric", "lightweight construction"],
                "details": {
                    "comfort": "breathable",
                    "weight": "lightweight",
                },
                "store": "Dress Store",
            }
            for index in range(8)
        ]
        filler_products = [
            {
                "parent_asin": f"FILLER-{index:03d}",
                "title": f"Miscellaneous jewelry accessory model {index}",
                "categories": ["Jewelry", "Accessories"],
                "features": [f"decorative item {index}"],
                "details": {"kind": f"ornament {index}"},
                "store": "Accessory Store",
            }
            for index in range(492)
        ]
        path = Path(directory) / "feedback-catalog.jsonl"
        path.write_text(
            "".join(
                json.dumps(product) + "\n"
                for product in [*feedback_products, *filler_products]
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _safe_proposal(
        agent: P5Agent,
        rankings: dict[str, list[str]],
        *,
        active: bool = True,
    ) -> PrfComputation:
        baseline = list(rankings["final"])
        candidate = baseline[10]
        proposed = [
            *baseline[:9],
            candidate,
            *(value for value in baseline[9:] if value != candidate),
        ]
        diagnostics = agent._empty_prf_diagnostics("safe_promotion")
        diagnostics.update({
            "active": active,
            "feedback_terms": ["breathable", "lightweight"],
            "route_candidate_count": 1,
            "new_candidate_count": 0,
            "would_change_top_10": True,
            "top10_added": [candidate],
            "top10_removed": [baseline[9]],
            "fusion": {"guard": "tail_only"},
        })
        return PrfComputation(tuple(proposed), (candidate,), diagnostics)

    def test_registry_is_exact_and_unique(self) -> None:
        validate_registry()
        self.assertEqual(set(SPEC_BY_ID), {C00, S00, R01})
        self.assertEqual(len({spec.mechanism for spec in SPECS}), 3)
        self.assertEqual(len({spec.stage_graph for spec in SPECS}), 3)
        self.assertEqual(SCHEMA_VERSION, "p5.prf-lab.v1")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown P5 variant"):
                P5Agent(self._catalog(directory), "P5.invalid")

    def test_control_is_response_equal_to_served_r08(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            baseline = Agent(catalog, retrieval_mode="coverage", rerank_mode="off")
            control = P5Agent(catalog, C00)
            self.addCleanup(baseline.connection.close)
            self.addCleanup(control.connection.close)
            baseline.reset("same", {})
            control.reset("same", {})
            messages = [
                "I'm looking for women's dresses, but I'm still exploring.",
                "For that, what matters is: cotton; color: blue.",
                "I don't have an additional preference for feature.",
            ]
            for turn, message in enumerate(messages, start=1):
                self.assertEqual(
                    control.respond("same", message, turn, 10),
                    baseline.respond("same", message, turn, 10),
                )

        self.assertEqual(control.experiment_stats()["turns"], 3)
        self.assertEqual(control.experiment_stats()["output_changes"], 0)
        self.assertEqual(control._p5_rowids, {})

    def test_shadow_computes_proposal_without_affecting_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            baseline = Agent(catalog, retrieval_mode="coverage", rerank_mode="off")
            shadow = P5Agent(catalog, S00)
            self.addCleanup(baseline.connection.close)
            self.addCleanup(shadow.connection.close)
            baseline.reset("same", {})
            shadow.reset("same", {})

            def proposal(
                state: object,
                query_terms: list[str],
                rankings: dict[str, list[str]],
            ) -> PrfComputation:
                return self._safe_proposal(shadow, rankings)

            with patch.object(shadow, "_compute_prf", side_effect=proposal):
                observed = shadow.respond(
                    "same", "I'm looking for women's cotton dresses.", 1, 10
                )
            expected = baseline.respond(
                "same", "I'm looking for women's cotton dresses.", 1, 10
            )

        self.assertEqual(observed, expected)
        diagnostics = shadow.debug_prf_diagnostics("same")
        self.assertFalse(diagnostics["affects_output"])
        self.assertTrue(diagnostics["would_change_top_10"])
        self.assertFalse(diagnostics["changed_top_10"])
        stats = shadow.experiment_stats()
        self.assertEqual(stats["activations"], 1)
        self.assertEqual(stats["shadow_changes"], 1)
        self.assertEqual(stats["output_changes"], 0)

    def test_active_variant_serves_only_the_guarded_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            baseline = Agent(catalog, retrieval_mode="coverage", rerank_mode="off")
            active = P5Agent(catalog, R01)
            self.addCleanup(baseline.connection.close)
            self.addCleanup(active.connection.close)
            baseline.reset("base", {})
            active.reset("active", {})
            message = "I'm looking for women's cotton dresses."
            expected = baseline.respond("base", message, 1, 10)

            def proposal(
                state: object,
                query_terms: list[str],
                rankings: dict[str, list[str]],
            ) -> PrfComputation:
                return self._safe_proposal(active, rankings)

            with patch.object(active, "_compute_prf", side_effect=proposal):
                observed = active.respond("active", message, 1, 10)

        expected_ids = [item["parent_asin"] for item in expected["recommendations"]]
        observed_ids = [item["parent_asin"] for item in observed["recommendations"]]
        self.assertEqual(observed_ids[:9], expected_ids[:9])
        self.assertNotEqual(observed_ids[9], expected_ids[9])
        self.assertEqual(active.experiment_stats()["output_changes"], 1)
        self.assertTrue(active.debug_prf_diagnostics("active")["changed_top_10"])

    def test_active_pipeline_delegates_final_order_to_guarded_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active = P5Agent(self._catalog(directory), R01)
            self.addCleanup(active.connection.close)
            active.reset("guard", {})

            def seeds(
                state: object,
                query_terms: list[str],
                rankings: dict[str, list[str]],
            ) -> tuple[list[str], dict[str, object]]:
                selected = rankings["final"][:5]
                return selected, {
                    "reason": "seed_consensus",
                    "query_term_count": len(query_terms),
                    "seed_depth": 5,
                    "seed_ids": selected,
                    "seed_coverages": {value: 3 for value in selected},
                    "maximum_seed_coverage": 3,
                    "seed_coverage_floor": 2,
                    "dual_route_seed_count": 3,
                }

            def fuse(
                query_terms: list[str],
                excluded_terms: set[str],
                feedback_terms: list[str],
                rankings: dict[str, list[str]],
                route: list[str],
                searchable_fields: dict[str, tuple[str, ...]],
                config: object,
                tokenize: object,
            ) -> tuple[list[str], dict[str, object]]:
                computation = self._safe_proposal(active, rankings)
                return list(computation.identifiers), {"guard": "tail_only"}

            with (
                patch.object(active, "_select_seeds", side_effect=seeds),
                patch.object(active, "_prf_route", return_value=["DRESS-10"]),
                patch(
                    "starter.p5_lab.extract_feedback_terms",
                    return_value=(
                        ["breathable", "lightweight"],
                        {"term_diagnostics": []},
                    ),
                ),
                patch(
                    "starter.p5_lab.build_prf_expression",
                    return_value='("women") AND ("breathable" OR "lightweight")',
                ),
                patch("starter.p5_lab.guarded_prf_fusion", side_effect=fuse) as guarded,
            ):
                active.respond(
                    "guard", "I'm looking for women's cotton dresses.", 1, 10
                )

        self.assertEqual(guarded.call_count, 1)
        self.assertEqual(
            guarded.call_args.args[2], ["breathable", "lightweight"]
        )
        self.assertEqual(active.experiment_stats()["output_changes"], 1)

    def test_real_shadow_pipeline_extracts_catalog_idf_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shadow = P5Agent(self._feedback_catalog(directory), S00)
            self.addCleanup(shadow.connection.close)
            shadow.reset("real", {})
            response = shadow.respond(
                "real", "I'm looking for women's cotton dresses.", 1, 10
            )
            diagnostics = shadow.debug_prf_diagnostics("real")

        self.assertEqual(len(response["recommendations"]), 8)
        self.assertTrue(diagnostics["active"])
        self.assertEqual(
            diagnostics["feedback_terms"], ["breathable", "lightweight"]
        )
        self.assertEqual(diagnostics["reason"], "no_safe_promotion")
        self.assertEqual(diagnostics["route_candidate_count"], 8)
        self.assertFalse(diagnostics["changed_top_10"])
        self.assertEqual(shadow.experiment_stats()["activations"], 1)

    def test_shadow_builds_catalog_rowid_and_vocab_without_label_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shadow = P5Agent(self._catalog(directory), S00)
            self.addCleanup(shadow.connection.close)
            tables = {
                str(row[0])
                for row in shadow.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            vocabulary_rows = shadow.connection.execute(
                "SELECT count(*) FROM products_prf_vocab"
            ).fetchone()[0]

        self.assertEqual(len(shadow._p5_rowids), len(PRODUCTS))
        self.assertEqual(shadow._p5_document_count, len(PRODUCTS))
        self.assertIn("products_prf_vocab", tables)
        self.assertGreater(vocabulary_rows, 0)
        for name, expected in dict(SPEC_BY_ID[S00].parameters).items():
            self.assertEqual(getattr(shadow.prf_config, name), expected)

        source = inspect.getsource(P5Agent).casefold()
        blocked = (
            "ground_" + "truth",
            "intent_" + "card",
            "scenario_" + "type",
            "sample_" + "id",
            "beha" + "vior",
        )
        for value in blocked:
            self.assertNotIn(value, source)
        constructor_names = set(inspect.signature(P5Agent.__init__).parameters)
        self.assertTrue(set(blocked).isdisjoint(constructor_names))

    def test_feedback_is_recomputed_after_override_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active = P5Agent(self._catalog(directory), R01)
            self.addCleanup(active.connection.close)
            active.reset("session", {})
            observed_queries: list[tuple[str, ...]] = []

            def current_only(
                state: object,
                query_terms: list[str],
                rankings: dict[str, list[str]],
            ) -> PrfComputation:
                observed_queries.append(tuple(query_terms))
                diagnostics = active._empty_prf_diagnostics("no_safe_promotion")
                diagnostics["feedback_terms"] = [
                    f"feedback_{query_terms[0]}" if query_terms else ""
                ]
                return PrfComputation(tuple(rankings["final"]), (), diagnostics)

            with patch.object(active, "_compute_prf", side_effect=current_only):
                active.respond(
                    "session", "I'm looking for women's red cotton dresses.", 1, 10
                )
                active.respond(
                    "session",
                    "Actually, switch from women's red cotton dresses to men's black running shoes.",
                    2,
                    10,
                )
                active.reset("session", {})
                active.respond(
                    "session", "I'm looking for silver necklaces.", 1, 10
                )

        self.assertTrue({"women", "red", "cotton", "dresses"} <= set(observed_queries[0]))
        self.assertTrue({"men", "black", "running", "shoes"} <= set(observed_queries[1]))
        self.assertFalse({"women", "red", "cotton", "dresses"} & set(observed_queries[1]))
        self.assertTrue({"silver", "necklaces"} <= set(observed_queries[2]))
        self.assertFalse({"black", "running", "shoes"} & set(observed_queries[2]))
        diagnostics = active.debug_prf_diagnostics("session")
        self.assertEqual(diagnostics["feedback_terms"], ["feedback_silver"])
        self.assertEqual(len(active._ranking_diagnostics), 1)


if __name__ == "__main__":
    unittest.main()
