from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
import unittest

from starter.teammate_v212_fusion import TeammateV212FusionA
from starter.teammate_bounded_other import TeammateBoundedOtherAgent


def response(*identifiers: str) -> dict:
    return {
        "message": "specific",
        "ask_attribute": "color",
        "recommendations": [
            {"parent_asin": identifier, "score": 1.0}
            for identifier in identifiers
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


class FakeTeammate:
    def __init__(self) -> None:
        self._sessions = {}
        self.calls = 0

    def reset(self, session_id, _profile):
        self._sessions[session_id] = SimpleNamespace(
            asked_attributes=set(), follow_up_attributes=set(),
            question_counts=Counter(), last_asked_attribute=None,
        )

    def respond(self, session_id, *_args):
        self.calls += 1
        state = self._sessions[session_id]
        state.asked_attributes.add("color")
        state.question_counts["color"] += 1
        state.last_asked_attribute = "color"
        start = (self.calls - 1) * 2
        return response(f"T{start + 1}", f"T{start + 2}")

    def close(self): pass


class FakeExpert:
    def __init__(self) -> None:
        self._sessions = {}

    def reset(self, session_id, _profile):
        self._sessions[session_id] = SimpleNamespace(
            pending_attribute=None, pending_turn=None
        )

    def respond(self, session_id, *_args):
        state = self._sessions[session_id]
        state.pending_attribute = "material"
        state.pending_turn = 1
        return response("hidden")

    @staticmethod
    def debug_rankings(_session_id):
        return {"final": ["T1", "T2", "T3", "T4", "V1", "V2", "V3"]}

    def close(self): pass


class TeammateV212FusionTests(unittest.TestCase):
    def make_agent(self):
        agent = TeammateV212FusionA(
            teammate=FakeTeammate(), rank_expert=FakeExpert()
        )
        agent.reset("s", {})
        return agent

    def test_two_t0_pages_then_unseen_v212_tail(self):
        agent = self.make_agent()
        first = agent.respond("s", "initial", 1, 2)
        second = agent.respond("s", "reply", 2, 2)
        third = agent.respond("s", "reply", 3, 2)
        self.assertEqual([row["parent_asin"] for row in first["recommendations"]], ["T1", "T2"])
        self.assertEqual([row["parent_asin"] for row in second["recommendations"]], ["T3", "T4"])
        self.assertEqual([row["parent_asin"] for row in third["recommendations"]], ["V1", "V2"])
        self.assertIsNone(agent.rank_expert._sessions["s"].pending_attribute)

    def test_override_restarts_two_page_grace(self):
        agent = self.make_agent()
        agent.respond("s", "initial", 1, 2)
        agent.respond("s", "reply", 2, 2)
        override = "Actually, ignore my earlier preference. What I need is: blue."
        actual = agent.respond("s", override, 3, 2)
        self.assertEqual(actual["ask_attribute"], "color")
        self.assertEqual(agent.evaluation_diagnostics()["intent_resets"], 1)
        self.assertEqual(agent.evaluation_diagnostics()["grace_turns"], 3)

    def test_b_can_cancel_only_visible_t0_question(self):
        agent = self.make_agent()
        agent.respond("s", "initial", 1, 2)
        self.assertTrue(agent.cancel_last_question("s", "color"))
        state = agent.teammate._sessions["s"]
        self.assertIsNone(state.last_asked_attribute)
        self.assertEqual(state.question_counts, Counter())

    def test_version_b_changes_only_question_contract_on_first_page(self):
        base = self.make_agent()
        agent = TeammateBoundedOtherAgent(
            base_agent=base, replace_specific=True
        )
        agent.reset("b", {})
        actual = agent.respond("b", "initial", 1, 2)
        self.assertEqual(actual["ask_attribute"], "other")
        self.assertEqual(
            [row["parent_asin"] for row in actual["recommendations"]],
            ["T1", "T2"],
        )
        self.assertEqual(base.evaluation_diagnostics()["grace_turns"], 1)


if __name__ == "__main__":
    unittest.main()
