from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import MethodType

from starter.agent import Agent, SessionState
from starter.attributes import build_conversation_constraint_view
from starter.p11_features import P11RerankResult
from starter.p11_lab import (
    ACTIVE_ID,
    CONTROL_ID,
    SHADOW_ID,
    P11Agent,
    _latest_hard_clause_terms,
)


def _catalog(path: Path, count: int = 16) -> None:
    rows = [
        {
            "parent_asin": f"P11ITEM{index:03d}",
            "title": f"blue cotton shirt item {index}",
            "categories": ["Clothing", "Men", "Shirts"],
            "features": ["cotton", "button closure"],
            "details": {"Color": "Blue", "Material": "Cotton"},
            "store": "Fixture",
            "description": "A blue cotton shirt for work.",
            "average_rating": 4.0,
            "rating_number": index,
            "price": 20.0,
        }
        for index in range(1, count + 1)
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _reversed_score(
    self: P11Agent,
    _state: SessionState,
    rankings: dict[str, list[str]],
) -> P11RerankResult:
    baseline = list(rankings["final"])
    proposed = [*reversed(baseline[:10]), *baseline[10:]]
    return P11RerankResult(tuple(proposed), False, "scored", True, {})


def _invalid_score(
    self: P11Agent,
    _state: SessionState,
    rankings: dict[str, list[str]],
) -> P11RerankResult:
    baseline = list(rankings["final"])
    return P11RerankResult(tuple(baseline[1:]), False, "scored", True, {})


class P11LabTests(unittest.TestCase):
    def test_close_releases_the_feature_store_and_base_connection(self) -> None:
        class FeatureStoreProbe:
            closed = False

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            _catalog(catalog)
            agent = P11Agent(catalog, CONTROL_ID)
            connection = agent.connection
            probe = FeatureStoreProbe()
            agent._feature_store = probe  # type: ignore[assignment]

            agent.close()

            self.assertTrue(probe.closed)
            self.assertIsNone(agent._feature_store)
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_latest_hard_clause_uses_current_goal_and_exclusions(self) -> None:
        state = SessionState(profile={})
        state.messages = [
            "A key requirement is blue cotton.",
            "Actually, ignore that. What I need is red waterproof hiking boots.",
        ]
        state.version = 2
        state.version_anchor_turn = 2
        state.excluded_terms = {"red"}
        state.slot_ledger.reconcile(
            build_conversation_constraint_view(
                "hiking boots", ["waterproof"], state.excluded_terms
            ),
            turn=2,
            version=2,
            message=state.messages[1],
        )

        terms = _latest_hard_clause_terms(state)

        self.assertNotIn("blue", terms)
        self.assertNotIn("red", terms)
        self.assertIn("waterproof", terms)
        self.assertIn("hiking", terms)
        self.assertIn("boots", terms)

    def test_control_is_exact_served_agent_and_does_not_open_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            _catalog(catalog)
            served = Agent(catalog)
            control = P11Agent(catalog, CONTROL_ID)
            served.reset("served", {})
            control.reset("control", {})

            expected = served.respond(
                "served", "I'm looking for shirts. A key requirement is cotton.", 1, 10
            )
            actual = control.respond(
                "control", "I'm looking for shirts. A key requirement is cotton.", 1, 10
            )

            self.assertEqual(actual, expected)
            capture = control.export_p11_blind_capture()
            self.assertFalse(capture["configuration"]["sidecar_opened"])
            self.assertEqual(capture["stats"]["output_changes"], 0)

    def test_active_changes_order_only_and_shadow_keeps_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            _catalog(catalog)
            baseline_agent = P11Agent(catalog, CONTROL_ID)
            active = P11Agent(catalog, CONTROL_ID)
            shadow = P11Agent(catalog, CONTROL_ID)
            active.p11_role = ACTIVE_ID
            shadow.p11_role = SHADOW_ID
            active._score = MethodType(_reversed_score, active)
            shadow._score = MethodType(_reversed_score, shadow)
            for identifier, agent in (
                ("base", baseline_agent), ("active", active), ("shadow", shadow)
            ):
                agent.reset(identifier, {})
            message = "I'm looking for shirts. A key requirement is cotton."

            baseline = baseline_agent.respond("base", message, 1, 10)
            active_response = active.respond("active", message, 1, 10)
            shadow_response = shadow.respond("shadow", message, 1, 10)
            baseline_ids = [row["parent_asin"] for row in baseline["recommendations"]]
            active_ids = [row["parent_asin"] for row in active_response["recommendations"]]

            self.assertEqual(set(active_ids), set(baseline_ids))
            self.assertEqual(active_ids, list(reversed(baseline_ids)))
            self.assertEqual(shadow_response, baseline)
            self.assertEqual(active.p11_stats.output_changes, 1)
            self.assertEqual(shadow.p11_stats.proposed_changes, 1)
            self.assertEqual(shadow.p11_stats.output_changes, 0)
            self.assertEqual(active.p11_stats.top10_membership_violation_count, 0)

    def test_invalid_proposal_falls_back_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            _catalog(catalog)
            baseline_agent = P11Agent(catalog, CONTROL_ID)
            active = P11Agent(catalog, CONTROL_ID)
            active.p11_role = ACTIVE_ID
            active._score = MethodType(_invalid_score, active)
            baseline_agent.reset("base", {})
            active.reset("active", {})
            message = "I'm looking for shirts. A key requirement is cotton."

            expected = baseline_agent.respond("base", message, 1, 10)
            actual = active.respond("active", message, 1, 10)

            self.assertEqual(actual, expected)
            self.assertEqual(active.p11_stats.fallbacks, 1)
            self.assertEqual(active.p11_stats.output_changes, 0)


if __name__ == "__main__":
    unittest.main()
