from __future__ import annotations

import unittest

from starter.fusion_other import (
    BOUNDARY_REPLY,
    EXHAUSTED_REPLY,
    FusionOtherAgent,
    MODE_ACTIVE,
    MODE_OFF,
    OTHER_MESSAGE,
)


class ParentState:
    def __init__(self) -> None:
        self.version = 1

    def debug_snapshot(self, _session_id: str) -> dict:
        return {"version": self.version}


class FakeA:
    def __init__(self) -> None:
        self.parent = ParentState()
        self.response = {
            "message": "A question",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": f"P{i}"} for i in range(10)],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        self.closed = False

    def reset(self, _session_id: str, _profile: dict) -> None:
        self.parent.version = 1

    def respond(self, _session_id: str, message: str, _turn: int, _top_k: int) -> dict:
        if message.startswith("Actually, ignore my earlier preference."):
            self.parent.version += 1
        return self.response

    def close(self) -> None:
        self.closed = True

    def evaluation_diagnostics(self) -> dict[str, object]:
        return {
            "schema_version": "fake-a-evaluation-diagnostics.v1",
            "turns": 7,
        }


class FusionOtherTests(unittest.TestCase):
    def _agent(self, mode: str = MODE_ACTIVE) -> tuple[FusionOtherAgent, FakeA]:
        base = FakeA()
        agent = FusionOtherAgent(other_mode=mode, base_agent=base)
        agent.reset("s", {})
        return agent, base

    def test_off_returns_exact_a_object(self) -> None:
        agent, base = self._agent(MODE_OFF)
        actual = agent.respond("s", "hello", 1, 10)
        self.assertIs(actual, base.response)

    def test_two_informative_replies_then_falls_back_to_a(self) -> None:
        agent, base = self._agent()
        first = agent.respond("s", "initial", 1, 10)
        second = agent.respond(
            "s", "For that, what matters is: blue; cotton.", 2, 10
        )
        third = agent.respond(
            "s", "For that, what matters is: waterproof.", 3, 10
        )
        self.assertEqual(first["ask_attribute"], "other")
        self.assertEqual(second["ask_attribute"], "other")
        self.assertEqual(first["message"], OTHER_MESSAGE)
        self.assertIs(third, base.response)
        state = agent.debug_other("s")
        self.assertEqual(state["other_asks"], 2)
        self.assertEqual(state["other_informative_replies"], 2)
        self.assertEqual(state["disclosed_constraints"], 3)

    def test_exhausted_reply_immediately_falls_back(self) -> None:
        agent, base = self._agent()
        agent.respond("s", "initial", 1, 10)
        response = agent.respond("s", EXHAUSTED_REPLY, 2, 10)
        self.assertIs(response, base.response)
        self.assertTrue(agent.debug_other("s")["other_exhausted"])

    def test_invalid_a_snapshot_falls_back_without_lifecycle_commit(self) -> None:
        agent, base = self._agent()
        before = agent.debug_other("s")
        base.parent = object()
        response = agent.respond("s", "initial", 1, 10)
        self.assertIs(response, base.response)
        self.assertEqual(agent.debug_other("s"), before)

    def test_boundary_allows_three_total_asks_but_not_four(self) -> None:
        agent, base = self._agent()
        self.assertEqual(agent.respond("s", "initial", 1, 10)["ask_attribute"], "other")
        self.assertEqual(agent.respond("s", BOUNDARY_REPLY, 2, 10)["ask_attribute"], "other")
        self.assertEqual(
            agent.respond("s", "For that, what matters is: blue.", 3, 10)["ask_attribute"],
            "other",
        )
        fourth = agent.respond("s", "For that, what matters is: cotton.", 4, 10)
        self.assertIs(fourth, base.response)
        state = agent.debug_other("s")
        self.assertTrue(state["other_boundary_sentinel_seen"])
        self.assertEqual(state["other_asks"], 3)
        self.assertEqual(state["other_informative_replies"], 2)

    def test_override_resets_only_other_lifecycle(self) -> None:
        agent, _base = self._agent()
        agent.respond("s", "initial", 1, 10)
        agent.respond("s", "For that, what matters is: blue.", 2, 10)
        response = agent.respond(
            "s",
            "Actually, ignore my earlier preference. What I need is: cotton.",
            3,
            10,
        )
        state = agent.debug_other("s")
        self.assertEqual(response["ask_attribute"], "other")
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["other_asks"], 1)
        self.assertEqual(state["other_informative_replies"], 0)
        diagnostics = agent.evaluation_diagnostics()
        self.assertEqual(diagnostics["other_activation_sessions"], 1)
        self.assertEqual(diagnostics["other_activation_turns"], 3)
        self.assertEqual(diagnostics["other_informative_replies"], 1)
        self.assertEqual(diagnostics["other_disclosed_constraints"], 1)

    def test_turn_ten_never_asks_other_and_rankings_are_unchanged(self) -> None:
        agent, base = self._agent()
        response = agent.respond("s", "initial", 10, 10)
        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(response["recommendations"], base.response["recommendations"])

        agent, base = self._agent()
        response = agent.respond("s", "initial", 1, 10)
        self.assertEqual(response["recommendations"], base.response["recommendations"])
        self.assertEqual(response["usage"], base.response["usage"])

    def test_evaluation_diagnostics_aggregate_without_changing_response(self) -> None:
        agent, base = self._agent()
        first = agent.respond("s", "initial", 1, 10)
        agent.respond("s", "For that, what matters is: blue; cotton.", 2, 10)
        agent.respond("s", "For that, what matters is: waterproof.", 3, 10)

        agent.reset("exhausted", {})
        agent.respond("exhausted", "initial", 1, 10)
        exhausted_response = agent.respond("exhausted", EXHAUSTED_REPLY, 2, 10)

        agent.reset("boundary", {})
        agent.respond("boundary", "initial", 1, 10)
        boundary_response = agent.respond("boundary", BOUNDARY_REPLY, 2, 10)

        diagnostics = agent.evaluation_diagnostics()
        self.assertEqual(
            diagnostics["fusion_core"],
            {
                "schema_version": "fake-a-evaluation-diagnostics.v1",
                "turns": 7,
            },
        )
        self.assertEqual(diagnostics["other_activation_sessions"], 3)
        self.assertEqual(diagnostics["other_activation_turns"], 5)
        self.assertEqual(diagnostics["other_informative_replies"], 2)
        self.assertEqual(diagnostics["other_disclosed_constraints"], 3)
        self.assertEqual(diagnostics["other_boundary_sentinel_replies"], 1)
        self.assertEqual(diagnostics["other_exhausted_replies"], 1)
        self.assertEqual(first["recommendations"], base.response["recommendations"])
        self.assertIs(exhausted_response, base.response)
        self.assertEqual(
            boundary_response["recommendations"], base.response["recommendations"]
        )


if __name__ == "__main__":
    unittest.main()
