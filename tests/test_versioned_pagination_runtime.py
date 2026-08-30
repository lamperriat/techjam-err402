from __future__ import annotations

import unittest

from scripts.evaluate_versioned_unseen_pagination import fixed_two_page_grace
from starter.agent import Agent, SessionState
from starter.versioned_pagination import (
    VersionedPaginationError,
    fixed_two_page_grace_order,
)


P11_READY = {
    "effective_mode": "active",
    "identity_verified": True,
    "fallback": False,
}
SMALL_RANKER_READY = {"effective_mode": "active", "fallback": False}


class VersionedPaginationRuntimeTests(unittest.TestCase):
    @staticmethod
    def _agent(mode: str) -> Agent:
        agent = Agent.__new__(Agent)
        agent.pagination_mode = mode
        return agent

    def test_runtime_matches_frozen_replay_top10(self) -> None:
        order = tuple(f"item-{index}" for index in range(30))
        served = {"item-0", "item-2", "item-4"}
        for intent_age in (1, 2, 3, 8):
            runtime = fixed_two_page_grace_order(order, served, intent_age)
            replay = fixed_two_page_grace(order, served, intent_age)
            self.assertEqual(runtime[:10], replay)
            self.assertEqual(set(runtime), set(order))

    def test_two_grace_pages_then_stable_unseen_page(self) -> None:
        agent = self._agent("active")
        state = SessionState(profile={})
        order = [f"item-{index}" for index in range(30)]

        first, first_diagnostics = agent._apply_pagination(
            state, order, P11_READY, SMALL_RANKER_READY
        )
        self.assertEqual(first, order)
        self.assertEqual(first_diagnostics["intent_age"], 1)
        agent._record_pagination_page(state, first[:10], first_diagnostics)

        second, second_diagnostics = agent._apply_pagination(
            state, order, P11_READY, SMALL_RANKER_READY
        )
        self.assertEqual(second, order)
        self.assertEqual(second_diagnostics["intent_age"], 2)
        agent._record_pagination_page(state, second[:10], second_diagnostics)

        third, third_diagnostics = agent._apply_pagination(
            state, order, P11_READY, SMALL_RANKER_READY
        )
        self.assertEqual(third[:10], order[10:20])
        self.assertEqual(len(third[:10]), len(set(third[:10])))
        self.assertTrue(third_diagnostics["activated"])
        self.assertTrue(third_diagnostics["full_membership_preserved"])

    def test_intent_version_change_resets_grace_and_served_state(self) -> None:
        agent = self._agent("active")
        state = SessionState(
            profile={},
            pagination_version=1,
            pagination_page_count=4,
            pagination_served_ids={"item-0", "item-1"},
        )
        state.version = 2
        order = [f"item-{index}" for index in range(20)]

        proposed, diagnostics = agent._apply_pagination(
            state, order, P11_READY, SMALL_RANKER_READY
        )
        self.assertEqual(proposed, order)
        self.assertEqual(diagnostics["intent_age"], 1)
        self.assertEqual(diagnostics["served_before_count"], 0)
        agent._record_pagination_page(state, proposed[:10], diagnostics)
        self.assertEqual(state.pagination_version, 2)
        self.assertEqual(state.pagination_page_count, 1)
        self.assertEqual(state.pagination_served_ids, set(order[:10]))

    def test_off_and_upstream_failure_are_exact_identity(self) -> None:
        order = [f"item-{index}" for index in range(20)]
        state = SessionState(profile={})

        off, off_diagnostics = self._agent("off")._apply_pagination(
            state, order, {}, {}
        )
        self.assertEqual(off, order)
        self.assertEqual(off_diagnostics["effective_mode"], "off")

        fallback, fallback_diagnostics = self._agent("active")._apply_pagination(
            state, order, {**P11_READY, "fallback": True}, SMALL_RANKER_READY
        )
        self.assertEqual(fallback, order)
        self.assertTrue(fallback_diagnostics["fallback"])
        self.assertFalse(fallback_diagnostics["output_changed"])

    def test_invalid_order_fails_closed_without_recording(self) -> None:
        agent = self._agent("active")
        state = SessionState(
            profile={}, pagination_version=1, pagination_page_count=2
        )
        order = ["same"] * 10
        proposed, diagnostics = agent._apply_pagination(
            state, order, P11_READY, SMALL_RANKER_READY
        )
        self.assertEqual(proposed, order)
        self.assertTrue(diagnostics["fallback"])
        agent._record_pagination_page(state, proposed, diagnostics)
        self.assertEqual(state.pagination_page_count, 2)

    def test_pure_runtime_rejects_invalid_age_and_duplicate_order(self) -> None:
        order = tuple(f"item-{index}" for index in range(10))
        with self.assertRaises(VersionedPaginationError):
            fixed_two_page_grace_order(order, set(), 0)
        with self.assertRaises(VersionedPaginationError):
            fixed_two_page_grace_order(("same",) * 10, set(), 3)


if __name__ == "__main__":
    unittest.main()
