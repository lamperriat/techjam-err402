from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.agent import Agent
from starter.p6_lab import (
    ACTIVE_ID,
    C00,
    CONTROL_ID,
    DepthComputation,
    P6Agent,
    R01,
    S00,
    SCHEMA_VERSION,
    SHADOW_ID,
    SPECS,
    SPEC_BY_ID,
    validate_registry,
)


def _product(index: int) -> dict[str, object]:
    return {
        "parent_asin": f"DRESS-{index:03d}",
        "title": f"Women's cotton dress collection {index}",
        "categories": ["Clothing", "Women", "Dresses"],
        "features": ["cotton dress", "comfortable everyday style"],
        "details": {"material": "cotton", "fit": "casual"},
        "store": f"Store {index % 7}",
        "description": "A cotton dress for daily use.",
    }


class P6LabTests(unittest.TestCase):
    def _catalog(self, directory: str, count: int = 260) -> Path:
        path = Path(directory) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(_product(index)) + "\n" for index in range(count)),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _proposal(
        agent: P6Agent,
        rankings: dict[str, list[str]],
        deep_ids: list[str],
    ) -> tuple[list[str], dict[str, object]]:
        baseline = list(rankings["final"])
        newcomer = next(value for value in deep_ids if value not in baseline)
        proposed = [
            *baseline[:9],
            newcomer,
            *(value for value in baseline[9:] if value != newcomer),
        ]
        diagnostics = agent._empty_depth_diagnostics("safe_tail_admission")
        diagnostics.update({
            "active": True,
            "reason": "safe_tail_admission",
            "query_terms": ["women", "cotton", "dresses"],
            "trigger": {
                "enabled": True,
                "rejection_reasons": [],
                "broad_count": 120,
                "final_count": len(baseline),
            },
            "prefix": {"matches": True, "validated_count": 120},
            "tail": {
                "candidate_count": len(deep_ids) - 120,
                "new_candidate_count": len(deep_ids) - 120,
                "proposals": [{"identifier": newcomer, "coverage": 4}],
            },
            "guard": {
                "applied": True,
                "reason": "safe_tail_admission",
                "protected_top9": baseline[:9],
                "incumbent": baseline[9],
                "replacement": newcomer,
                "top9_unchanged": True,
                "newcomers": [newcomer],
            },
            "final_top10": proposed[:10],
        })
        return proposed, diagnostics

    def test_registry_and_exports_are_exact(self) -> None:
        validate_registry()
        self.assertEqual(set(SPEC_BY_ID), {
            "P6.C00.r08_coverage",
            "P6.S00.adaptive_depth_shadow",
            "P6.R01.guarded_broad_depth_doubling",
        })
        self.assertEqual((CONTROL_ID, SHADOW_ID, ACTIVE_ID), (C00, S00, R01))
        self.assertEqual(len({spec.mechanism for spec in SPECS}), 3)
        self.assertEqual(len({spec.stage_graph for spec in SPECS}), 3)
        self.assertEqual(SCHEMA_VERSION, "p6.adaptive-depth-lab.v1")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown P6 variant"):
                P6Agent(self._catalog(directory, 12), "P6.invalid")

    def test_control_is_response_equal_to_served_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory, 32)
            baseline = Agent(catalog, retrieval_mode="coverage", rerank_mode="off")
            control = P6Agent(catalog, C00)
            self.addCleanup(baseline.connection.close)
            self.addCleanup(control.connection.close)
            baseline.reset("same", {})
            control.reset("same", {})
            messages = [
                "I'm looking for women's dresses, but I'm still exploring.",
                "For that, what matters is: cotton and casual style.",
                "I don't have an additional preference for color.",
            ]
            for turn, message in enumerate(messages, start=1):
                self.assertEqual(
                    control.respond("same", message, turn, 10),
                    baseline.respond("same", message, turn, 10),
                )

        self.assertEqual(control.experiment_stats()["turns"], 3)
        self.assertEqual(control.experiment_stats()["deep_candidate_total"], 0)
        self.assertEqual(len(control.experiment_audit()), 3)

    def test_shadow_computes_proposal_but_is_response_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            baseline = Agent(catalog, retrieval_mode="coverage", rerank_mode="off")
            shadow = P6Agent(catalog, S00)
            self.addCleanup(baseline.connection.close)
            self.addCleanup(shadow.connection.close)
            baseline.reset("same", {})
            shadow.reset("same", {})

            def propose(*args: object) -> tuple[list[str], dict[str, object]]:
                rankings = args[2]
                deep_ids = args[3]
                assert isinstance(rankings, dict)
                assert isinstance(deep_ids, list)
                return self._proposal(shadow, rankings, deep_ids)

            message = "I'm looking for women's breathable cotton dresses."
            with patch("starter.p6_lab.guarded_depth_admission", side_effect=propose):
                observed = shadow.respond("same", message, 1, 10)
            expected = baseline.respond("same", message, 1, 10)

        self.assertEqual(observed, expected)
        diagnostics = shadow.debug_adaptive_depth_diagnostics("same")
        self.assertFalse(diagnostics["affects_output"])
        self.assertTrue(diagnostics["would_change_top_10"])
        self.assertFalse(diagnostics["changed_top_10"])
        self.assertEqual(shadow.experiment_stats()["shadow_changes"], 1)
        self.assertEqual(shadow.experiment_stats()["output_changes"], 0)

    def test_active_serves_only_guarded_rank_ten_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            baseline = Agent(catalog, retrieval_mode="coverage", rerank_mode="off")
            active = P6Agent(catalog, R01)
            self.addCleanup(baseline.connection.close)
            self.addCleanup(active.connection.close)
            baseline.reset("base", {})
            active.reset("active", {})
            message = "I'm looking for women's breathable cotton dresses."

            def propose(*args: object) -> tuple[list[str], dict[str, object]]:
                rankings = args[2]
                deep_ids = args[3]
                assert isinstance(rankings, dict)
                assert isinstance(deep_ids, list)
                return self._proposal(active, rankings, deep_ids)

            expected = baseline.respond("base", message, 1, 10)
            with patch("starter.p6_lab.guarded_depth_admission", side_effect=propose) as guard:
                observed = active.respond("active", message, 1, 10)

        expected_ids = [item["parent_asin"] for item in expected["recommendations"]]
        observed_ids = [item["parent_asin"] for item in observed["recommendations"]]
        self.assertEqual(observed_ids[:9], expected_ids[:9])
        self.assertNotEqual(observed_ids[9], expected_ids[9])
        self.assertEqual(guard.call_count, 1)
        self.assertEqual(active.experiment_stats()["output_changes"], 1)

    def test_real_deep_route_is_exact_240_with_base_120_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shadow = P6Agent(self._catalog(directory), S00)
            self.addCleanup(shadow.connection.close)
            shadow.reset("depth", {})
            response = shadow.respond(
                "depth", "I'm looking for women's breathable cotton dresses.", 1, 10
            )
            diagnostics = shadow.debug_adaptive_depth_diagnostics("depth")

        route = diagnostics["route_audit"]
        self.assertEqual(len(response["recommendations"]), 10)
        self.assertEqual(len(route["base_broad_ids"]), 120)
        self.assertEqual(len(route["deep_broad_ids"]), 240)
        self.assertEqual(route["deep_broad_ids"][:120], route["base_broad_ids"])
        self.assertTrue(diagnostics["prefix"]["matches"])
        self.assertTrue(diagnostics["target_blind"])
        audit = shadow.experiment_audit()[0]
        self.assertEqual(audit["session_index"], 0)
        self.assertEqual(
            audit["base_union_pool"],
            list(dict.fromkeys([*audit["base_pool"], *audit["strict_pool"]])),
        )
        self.assertEqual(
            audit["deep_union_pool"],
            list(dict.fromkeys([*audit["deep_pool"], *audit["strict_pool"]])),
        )

    def test_precheck_skip_does_not_issue_deep_sql(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shadow = P6Agent(self._catalog(directory), S00)
            self.addCleanup(shadow.connection.close)
            shadow.reset("skip", {})
            with patch.object(shadow, "_deep_broad_route") as deep_query:
                shadow.respond(
                    "skip", "I'm looking for women's cotton dresses.", 1, 10
                )
            diagnostics = shadow.debug_adaptive_depth_diagnostics("skip")

        deep_query.assert_not_called()
        self.assertFalse(diagnostics["deep_query_executed"])
        self.assertFalse(diagnostics["triggered"])
        self.assertIsNone(diagnostics["prefix"]["matches"])
        self.assertEqual(
            diagnostics["guard"]["reason"], "incumbent_coverage_already_full"
        )
        self.assertEqual(shadow.experiment_stats()["deep_queries"], 0)
        audit = shadow.experiment_audit()[0]
        self.assertEqual(audit["deep_pool"], [])
        self.assertEqual(audit["deep_union_pool"], audit["base_union_pool"])

    def test_diagnostics_preserve_base_layers_and_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shadow = P6Agent(self._catalog(directory), S00)
            self.addCleanup(shadow.connection.close)
            shadow.reset("bounded", {})
            shadow.respond(
                "bounded", "I'm looking for women's breathable cotton dresses.", 1, 10
            )
            complete = shadow.debug_rerank_diagnostics("bounded")
            depth = shadow.debug_adaptive_depth_diagnostics("bounded")

        self.assertIn("coverage", complete)
        self.assertIn("breakdowns", complete)
        self.assertIn("question_shadow", complete)
        self.assertIn("adaptive_depth", complete)
        self.assertLessEqual(len(depth["route_audit"]["base_broad_ids"]), 120)
        self.assertLessEqual(len(depth["route_audit"]["deep_broad_ids"]), 240)
        self.assertLessEqual(len(depth["route_audit"]["output_top10"]), 10)

    def test_empty_exception_and_prefix_mismatch_fall_back_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            baseline = Agent(catalog, retrieval_mode="coverage", rerank_mode="off")
            active = P6Agent(catalog, R01)
            self.addCleanup(baseline.connection.close)
            self.addCleanup(active.connection.close)
            baseline.reset("empty-base", {})
            active.reset("empty-active", {})
            message = "I'm looking for unobtainium ceremonial helmets."
            self.assertEqual(
                active.respond("empty-active", message, 1, 10),
                baseline.respond("empty-base", message, 1, 10),
            )
            diagnostics = active.debug_adaptive_depth_diagnostics("empty-active")

        self.assertFalse(diagnostics["deep_query_executed"])
        self.assertEqual(diagnostics["route_audit"]["deep_broad_ids"], [])

        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            baseline = Agent(catalog, retrieval_mode="coverage", rerank_mode="off")
            active = P6Agent(catalog, R01)
            self.addCleanup(baseline.connection.close)
            self.addCleanup(active.connection.close)
            baseline.reset("base", {})
            active.reset("active", {})
            message = "I'm looking for women's breathable cotton dresses."
            expected = baseline.respond("base", message, 1, 10)
            with patch.object(active, "_deep_broad_route", side_effect=RuntimeError("boom")):
                observed = active.respond("active", message, 1, 10)
            diagnostics = active.debug_adaptive_depth_diagnostics("active")

        self.assertEqual(observed, expected)
        self.assertEqual(diagnostics["reason"], "exception_fallback")
        self.assertEqual(diagnostics["exception_class"], "RuntimeError")
        self.assertTrue(diagnostics["deep_query_executed"])
        self.assertTrue(diagnostics["target_blind"])

        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            baseline = Agent(catalog, retrieval_mode="coverage", rerank_mode="off")
            active = P6Agent(catalog, R01)
            self.addCleanup(baseline.connection.close)
            self.addCleanup(active.connection.close)
            baseline.reset("base2", {})
            active.reset("active2", {})
            expected = baseline.respond("base2", message, 1, 10)
            original = active._deep_broad_route(active._query_terms(active._sessions["active2"]))
            reversed_route = (list(reversed(original[0])), original[1])
            with patch.object(active, "_deep_broad_route", return_value=reversed_route):
                observed = active.respond("active2", message, 1, 10)
            diagnostics = active.debug_adaptive_depth_diagnostics("active2")

        self.assertEqual(observed, expected)
        self.assertFalse(diagnostics["prefix"]["matches"])
        self.assertEqual(active.experiment_stats()["prefix_mismatches"], 1)
        audit = active.experiment_audit()[0]
        self.assertTrue(audit["deep_query_executed"])
        self.assertEqual(audit["deep_union_pool"], audit["base_union_pool"])

    def test_override_and_reset_recompute_visible_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active = P6Agent(self._catalog(directory), R01)
            self.addCleanup(active.connection.close)
            active.reset("state", {})
            observed: list[tuple[str, ...]] = []

            def current(
                state: object,
                query_terms: list[str],
                rankings: dict[str, list[str]],
            ) -> DepthComputation:
                observed.append(tuple(query_terms))
                return DepthComputation(
                    tuple(rankings["final"]),
                    (),
                    active._empty_depth_diagnostics("no_admission"),
                )

            with patch.object(active, "_compute_depth", side_effect=current):
                active.respond(
                    "state", "I'm looking for women's red cotton dresses.", 1, 10
                )
                active.respond(
                    "state",
                    "Actually, switch from women's red cotton dresses to men's black running shoes.",
                    2,
                    10,
                )
                active.reset("state", {})
                active.respond("state", "I'm looking for silver necklaces.", 1, 10)

        self.assertTrue({"women", "red", "cotton", "dresses"} <= set(observed[0]))
        self.assertTrue({"men", "black", "running", "shoes"} <= set(observed[1]))
        self.assertFalse({"women", "red", "cotton", "dresses"} & set(observed[1]))
        self.assertTrue({"silver", "necklaces"} <= set(observed[2]))
        self.assertFalse({"black", "running", "shoes"} & set(observed[2]))
        self.assertEqual(len(active.experiment_audit()), 3)
        self.assertEqual(len(active._state_session_indexes), 1)
        active.drop_session("state")
        self.assertEqual(active._state_session_indexes, {})

    def test_contract_and_constructor_are_label_blind(self) -> None:
        catalog_ids = {f"DRESS-{index:03d}" for index in range(24)}
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory, 24)
            for variant in (C00, S00, R01):
                with self.subTest(variant=variant):
                    agent = P6Agent(catalog, variant)
                    try:
                        agent.reset("contract", {})
                        response = agent.respond(
                            "contract", "I'm looking for women's cotton dresses.", 1, 10
                        )
                    finally:
                        agent.connection.close()
                    self.assertEqual(
                        set(response),
                        {"message", "ask_attribute", "recommendations", "usage"},
                    )
                    identifiers = [
                        item["parent_asin"] for item in response["recommendations"]
                    ]
                    self.assertLessEqual(len(identifiers), 10)
                    self.assertEqual(len(identifiers), len(set(identifiers)))
                    self.assertTrue(set(identifiers) <= catalog_ids)

        constructor_names = set(inspect.signature(P6Agent.__init__).parameters)
        self.assertEqual(
            constructor_names,
            {"self", "catalog_path", "variant_id", "question_policy"},
        )
        source = inspect.getsource(P6Agent).casefold()
        blocked = (
            "ground_" + "truth",
            "intent_" + "card",
            "scenario_" + "type",
            "sample_" + "id",
            "beha" + "vior",
        )
        for value in blocked:
            self.assertNotIn(value, source)


if __name__ == "__main__":
    unittest.main()
