from __future__ import annotations

import unittest
from collections import Counter
from types import SimpleNamespace

from starter.teammate_bounded_other import (
    OTHER_MESSAGE, TeammateBoundedOtherAgent, is_intent_override,
)


class FakeBase:
    def __init__(self) -> None:
        self.ask = None
        self.response = {
            "message": "base", "ask_attribute": None,
            "recommendations": [{"parent_asin": "P1"}], "usage": {"prompt_tokens": 0},
        }

    def reset(self, _session_id, _profile): pass
    def close(self): pass

    def respond(self, *_args):
        if self.ask is None:
            return self.response
        return {**self.response, "message": "specific", "ask_attribute": self.ask}


class StatefulFakeBase(FakeBase):
    def __init__(self) -> None:
        super().__init__()
        self.ask = "color"
        self._sessions = {}

    def reset(self, session_id, _profile):
        self._sessions[session_id] = SimpleNamespace(
            asked_attributes=set(), follow_up_attributes=set(),
            last_asked_attribute=None, question_counts=Counter(),
        )

    def respond(self, session_id, *_args):
        state = self._sessions[session_id]
        state.asked_attributes.add("color")
        state.last_asked_attribute = "color"
        state.question_counts["color"] += 1
        return super().respond()


class BoundedOtherTests(unittest.TestCase):
    def setUp(self):
        self.base = FakeBase()
        self.agent = TeammateBoundedOtherAgent(base_agent=self.base)
        self.agent.reset("s", {})

    def test_two_informative_rounds_preserve_ranking_and_usage(self):
        first = self.agent.respond("s", "initial", 1, 10)
        second = self.agent.respond("s", "For that, what matters is: blue; cotton.", 2, 10)
        third = self.agent.respond("s", "For that, what matters is: waterproof.", 3, 10)
        self.assertEqual((first["ask_attribute"], second["ask_attribute"]), ("other", "other"))
        self.assertEqual(first["message"], OTHER_MESSAGE)
        self.assertIs(first["recommendations"], self.base.response["recommendations"])
        self.assertIs(first["usage"], self.base.response["usage"])
        self.assertIs(third, self.base.response)
        self.assertEqual(self.agent.debug_other("s")["disclosed_constraints"], 3)

    def test_specific_base_question_and_turn_ten_are_exact_base(self):
        self.base.ask = "color"
        self.assertEqual(self.agent.respond("s", "initial", 1, 10)["ask_attribute"], "color")
        self.base.ask = None
        self.assertIs(self.agent.respond("s", "later", 10, 10), self.base.response)

    def test_exhausted_stops_and_boundary_consumes_one_of_two_asks(self):
        self.agent.respond("s", "initial", 1, 10)
        exhausted = self.agent.respond(
            "s", "I don't have an additional preference for other.", 2, 10
        )
        self.assertIs(exhausted, self.base.response)
        self.agent.reset("b", {})
        self.agent.respond("b", "initial", 1, 10)
        second = self.agent.respond(
            "b", "I don't have a preference for other; please use your judgment.", 2, 10
        )
        self.assertEqual(second["ask_attribute"], "other")
        self.assertIs(self.agent.respond("b", "unknown", 3, 10), self.base.response)

    def test_canonical_and_noncanonical_override_reset_limits(self):
        self.agent.respond("s", "initial", 1, 10)
        self.agent.respond("s", "For that, what matters is: blue.", 2, 10)
        canonical = "Actually, ignore my earlier preference. What I need is: cotton."
        self.assertTrue(is_intent_override(canonical))
        self.assertEqual(self.agent.respond("s", canonical, 3, 10)["ask_attribute"], "other")
        self.agent.respond("s", "For that, what matters is: linen.", 4, 10)
        self.assertTrue(is_intent_override("Changed my mind: wool instead."))
        self.assertEqual(
            self.agent.respond("s", "Changed my mind: wool instead.", 5, 10)["ask_attribute"],
            "other",
        )
        self.assertEqual(self.agent.debug_other("s")["version"], 3)
        self.assertEqual(self.agent.evaluation_diagnostics()["other_override_resets"], 2)

    def test_forced_other_cancels_unserved_specific_question(self):
        base = StatefulFakeBase()
        agent = TeammateBoundedOtherAgent(
            base_agent=base, replace_specific=True
        )
        agent.reset("forced", {})
        actual = agent.respond("forced", "initial", 1, 10)
        state = base._sessions["forced"]
        self.assertEqual(actual["ask_attribute"], "other")
        self.assertIsNone(state.last_asked_attribute)
        self.assertNotIn("color", state.asked_attributes)
        self.assertEqual(state.question_counts, Counter())


if __name__ == "__main__":
    unittest.main()
