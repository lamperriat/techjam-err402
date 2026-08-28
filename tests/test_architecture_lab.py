from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.architecture_lab import (
    CONTROL_ID,
    SPECS,
    ArchitectureAgent,
    ArchitectureSpec,
    validate_registry,
)
from starter.slot_ledger import SlotRecord


PRODUCTS = [
    {
        "parent_asin": "BLUE-COTTON-30",
        "title": "Women's blue cotton casual summer dress",
        "categories": ["Clothing", "Women", "Dresses"],
        "features": ["cotton", "blue", "casual", "comfortable pockets"],
        "details": {"material": "cotton", "color": "blue"},
        "price": 30.0,
        "store": "Alpha",
    },
    {
        "parent_asin": "RED-POLY-100",
        "title": "Women's red polyester formal dress",
        "categories": ["Clothing", "Women", "Dresses"],
        "features": ["polyester", "red", "formal evening style"],
        "details": {"material": "polyester", "color": "red"},
        "price": 100.0,
        "store": "Beta",
    },
    {
        "parent_asin": "BLACK-RUNNER",
        "title": "Men's black mesh running shoe",
        "categories": ["Clothing", "Men", "Shoes"],
        "features": ["black", "mesh", "running", "wide"],
        "details": {"material": "mesh", "color": "black"},
        "price": 55.0,
        "store": "Gamma",
    },
    {
        "parent_asin": "UNKNOWN-PRICE",
        "title": "Women's blue linen everyday dress",
        "categories": ["Clothing", "Women", "Dresses"],
        "features": ["linen", "blue", "everyday"],
        "details": {"material": "linen", "color": "blue"},
        "store": "Delta",
    },
    *[
        {
            "parent_asin": f"EXTRA-{index:02d}",
            "title": f"Women's cotton dress style {index}",
            "categories": ["Clothing", "Women", "Dresses"],
            "features": ["cotton", "dress", f"feature {index}"],
            "details": {"material": "cotton"},
            "price": 20.0 + index,
            "store": f"Store {index % 3}",
        }
        for index in range(1, 13)
    ],
]


class ArchitectureLabTests(unittest.TestCase):
    def _catalog_products(self, directory: str, products: list[dict]) -> Path:
        path = Path(directory) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        return path

    def _catalog(self, directory: str) -> Path:
        return self._catalog_products(directory, PRODUCTS)

    def test_registry_has_unique_non_parameter_architectures(self) -> None:
        validate_registry()
        self.assertGreaterEqual(len(SPECS) - 1, 10)
        self.assertEqual(len({spec.mechanism for spec in SPECS}), len(SPECS))
        self.assertEqual(len({spec.stage_graph for spec in SPECS}), len(SPECS))

        duplicate = ArchitectureSpec(
            "R99.parameter_copy",
            "retrieval",
            "field_rrf_copy",
            next(spec.stage_graph for spec in SPECS if spec.variant_id == "R01.field_rrf"),
            "Only a parameter copy.",
            (("weight", 2.0),),
        )
        with self.assertRaisesRegex(ValueError, "stage graphs"):
            validate_registry((*SPECS, duplicate))

    def test_control_is_response_equal_to_explicit_control_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            default = Agent(catalog, retrieval_mode="control")
            control = ArchitectureAgent(catalog, CONTROL_ID)
            self.addCleanup(default.connection.close)
            self.addCleanup(control.connection.close)
            default.reset("same", {})
            control.reset("same", {})
            messages = [
                "I'm looking for women's dresses, but I'm still exploring.",
                "For that, what matters is: cotton; color: blue.",
                "I don't have an additional preference for feature.",
            ]
            for turn, message in enumerate(messages, start=1):
                self.assertEqual(
                    control.respond("same", message, turn, 10),
                    default.respond("same", message, turn, 10),
                )

    def test_promoted_agent_is_response_equal_to_frozen_r08(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            promoted = Agent(catalog, retrieval_mode="coverage")
            frozen = ArchitectureAgent(catalog, "R08.coverage_cascade")
            self.addCleanup(promoted.connection.close)
            self.addCleanup(frozen.connection.close)
            promoted.reset("same", {})
            frozen.reset("same", {})
            messages = [
                "I'm looking for women's dresses, but I'm still exploring.",
                "For that, what matters is: cotton; color: blue.",
                "I don't have an additional preference for feature.",
            ]
            for turn, message in enumerate(messages, start=1):
                self.assertEqual(
                    frozen.respond("same", message, turn, 10),
                    promoted.respond("same", message, turn, 10),
                )

        self.assertEqual(promoted.retrieval_mode, "coverage")
        self.assertEqual(frozen.retrieval_mode, "control")

    def test_all_variants_preserve_contract_and_catalog_boundary(self) -> None:
        catalog_ids = {product["parent_asin"] for product in PRODUCTS}
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            for spec in SPECS:
                with self.subTest(spec=spec.variant_id):
                    agent = ArchitectureAgent(catalog, spec.variant_id)
                    try:
                        agent.reset("contract", {})
                        response = agent.respond(
                            "contract",
                            "I'm looking for women's cotton dresses around $30, but I'm still exploring.",
                            1,
                            10,
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

    def test_candidate_carryover_is_scoped_to_goal_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = ArchitectureAgent(
                self._catalog(directory), "R10.candidate_carryover"
            )
            self.addCleanup(agent.connection.close)
            agent.reset("carry", {})
            agent.respond("carry", "I'm looking for women's dresses.", 1, 10)
            agent.respond("carry", "For that, what matters is cotton.", 2, 10)
            activations_before_override = agent.variant_stats.activations
            agent.respond(
                "carry",
                "Actually, ignore my earlier preference. What I need is: black running shoes.",
                3,
                10,
            )

        self.assertEqual(activations_before_override, 1)
        self.assertEqual(agent.variant_stats.activations, 1)

    def test_numeric_budget_ranks_known_near_price_without_rejecting_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = ArchitectureAgent(self._catalog(directory), "R12.numeric_budget")
            self.addCleanup(agent.connection.close)
            agent.reset("budget", {})
            response = agent.respond(
                "budget", "I'm looking for women's dresses around $30.", 1, 10
            )
            identifiers = [item["parent_asin"] for item in response["recommendations"]]
            all_identifiers = agent.debug_rankings("budget")["final"]

        self.assertIn("BLUE-COTTON-30", identifiers)
        self.assertIn("UNKNOWN-PRICE", all_identifiers)
        self.assertLess(
            all_identifiers.index("BLUE-COTTON-30"),
            all_identifiers.index("RED-POLY-100"),
        )
        self.assertGreater(agent.variant_stats.activations, 0)

    def test_numeric_budget_requires_price_context_and_rejects_measurements(self) -> None:
        self.assertIsNone(
            ArchitectureAgent._budget_constraint(
                ["I need a necklace about 21.25inch long."]
            )
        )
        self.assertIsNone(
            ArchitectureAgent._budget_constraint(
                ["I want a bag under 30 cm wide."]
            )
        )
        self.assertIsNone(
            ArchitectureAgent._budget_constraint(
                ["I need around 30 items."]
            )
        )
        self.assertEqual(
            ArchitectureAgent._budget_constraint(
                ["My budget is around 30 dollars."]
            ),
            ("around", 30.0),
        )
        self.assertEqual(
            ArchitectureAgent._budget_constraint(["Please keep the price under 45."]),
            ("under", 45.0),
        )
        self.assertEqual(
            ArchitectureAgent._budget_constraint(["I can spend up to $55."]),
            ("under", 55.0),
        )

    def test_numeric_budget_is_cleared_by_no_preference_and_goal_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = ArchitectureAgent(self._catalog(directory), "R12.numeric_budget")
            self.addCleanup(agent.connection.close)
            agent.reset("budget-clear", {})
            agent.respond(
                "budget-clear", "I'm looking for women's dresses around $30.", 1, 10
            )
            after_budget = agent.variant_stats.activations
            agent.respond(
                "budget-clear",
                "I don't have an additional preference for budget.",
                2,
                10,
            )
            after_exhaustion = agent.variant_stats.activations
            agent.respond(
                "budget-clear",
                "Actually, ignore my earlier preference. What I need is: black running shoes.",
                3,
                10,
            )

        self.assertEqual(after_budget, 1)
        self.assertEqual(after_exhaustion, 1)
        self.assertEqual(agent.variant_stats.activations, 1)

    def test_slot_filter_never_backfills_known_negative_conflicts(self) -> None:
        products = [
            {
                "parent_asin": "SAFE-COTTON",
                "title": "Women's cotton dress",
                "categories": ["Clothing", "Women", "Dresses"],
                "features": ["cotton"],
                "details": {"material": "cotton"},
                "store": "Safe",
            },
            *[
                {
                    "parent_asin": f"CONFLICT-{index:02d}",
                    "title": f"Women's polyester dress {index}",
                    "categories": ["Clothing", "Women", "Dresses"],
                    "features": ["polyester"],
                    "details": {"material": "polyester"},
                    "store": "Conflict",
                }
                for index in range(12)
            ],
        ]
        with tempfile.TemporaryDirectory() as directory:
            agent = ArchitectureAgent(
                self._catalog_products(directory, products),
                "R09.slot_filter_relax",
            )
            self.addCleanup(agent.connection.close)
            agent.reset("negative", {})
            agent.respond(
                "negative",
                "I'm looking for women's dresses.",
                1,
                10,
            )
            state = agent._sessions["negative"]
            state.slot_ledger.records.append(SlotRecord(
                record_id=999,
                slot="material",
                value="polyester",
                polarity=-1,
                hardness="hard",
                source="test_visible_negative",
                confidence=1.0,
                source_turn=1,
                version=state.version,
            ))
            rankings = agent._rank_candidates(state)

        identifiers = rankings["final"][:10]
        self.assertEqual(identifiers, ["SAFE-COTTON"])
        self.assertIn(
            "negative_guard_shortfall",
            agent.variant_stats.relaxation_counts,
        )
        self.assertEqual(agent.variant_stats.fallbacks, 1)

    def test_browse_routing_is_scoped_to_current_goal_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            for variant_id in ("R11.browse_mmr", "R13.intent_router"):
                with self.subTest(variant_id=variant_id):
                    agent = ArchitectureAgent(catalog, variant_id)
                    try:
                        session_id = variant_id
                        agent.reset(session_id, {})
                        agent.respond(
                            session_id,
                            "I'm looking for women's dresses, but I'm still exploring.",
                            1,
                            10,
                        )
                        agent.respond(
                            session_id,
                            "Actually, change my mind. I'm looking for black running shoes.",
                            2,
                            10,
                        )
                        stats = agent.experiment_stats()
                    finally:
                        agent.connection.close()

                    if variant_id == "R11.browse_mmr":
                        self.assertEqual(stats["route_counts"].get("browse_mmr"), 2)
                        self.assertEqual(stats["activations"], 1)
                    else:
                        self.assertEqual(stats["route_counts"].get("router_browse"), 1)
                        self.assertEqual(stats["route_counts"].get("router_turn"), 1)

    def test_constructor_has_no_private_evaluator_inputs(self) -> None:
        names = set(ArchitectureAgent.__init__.__code__.co_varnames)
        self.assertTrue(
            names.isdisjoint({
                "ground_truth",
                "intent_card",
                "behavior",
                "scenario_type",
                "sample_id",
            })
        )


if __name__ == "__main__":
    unittest.main()
