from __future__ import annotations

from types import SimpleNamespace
import unittest

from starter.teammate_first_page_fusion import (
    IntentRoutedFusionAgent, TeammateFirstPageFusionAgent,
)


def response(identifier: str, *, ask: str | None = None) -> dict:
    return {
        "message": "question",
        "ask_attribute": ask,
        "recommendations": [{"parent_asin": identifier, "score": 1.0}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


class FakeTeammate:
    def reset(self, *_args): pass
    def close(self): pass
    def respond(self, *_args): return response("T0")


class FakeFusion:
    def __init__(self):
        self.base = SimpleNamespace(
            fusion=SimpleNamespace(_sessions={}),
            _served_versions={},
            _all_served={},
        )

    def reset(self, session_id, _profile):
        self.base.fusion._sessions[session_id] = SimpleNamespace(served={"B"})
        self.base._served_versions[session_id] = (1, {"B"})
        self.base._all_served[session_id] = {"B"}

    def close(self): pass
    def respond(self, *_args): return response("B", ask="other")
    def evaluation_diagnostics(self): return {"schema_version": "fake.v1", "turns": 1}


class FirstPageFusionTests(unittest.TestCase):
    def test_turn_one_uses_teammate_rows_but_fusion_question(self):
        fusion = FakeFusion()
        agent = TeammateFirstPageFusionAgent(
            teammate=FakeTeammate(), fusion_b=fusion
        )
        agent.reset("s", {})
        actual = agent.respond("s", "query", 1, 1)
        self.assertEqual(actual["recommendations"][0]["parent_asin"], "T0")
        self.assertEqual(actual["ask_attribute"], "other")
        self.assertEqual(fusion.base.fusion._sessions["s"].served, {"T0"})
        self.assertEqual(fusion.base._served_versions["s"], (1, {"T0"}))

    def test_later_turn_is_exact_fusion_response(self):
        fusion = FakeFusion()
        agent = TeammateFirstPageFusionAgent(
            teammate=FakeTeammate(), fusion_b=fusion
        )
        agent.reset("s", {})
        actual = agent.respond("s", "reply", 2, 1)
        self.assertEqual(actual, response("B", ask="other"))

    def test_visible_intent_router_keeps_one_backend_for_whole_session(self):
        teammate = FakeTeammate()
        fusion = FakeFusion()
        agent = IntentRoutedFusionAgent(teammate=teammate, fusion_b=fusion)
        agent.reset("browse", {})
        first = agent.respond(
            "browse", "I'm looking for shirts, but I'm still exploring.", 1, 1
        )
        second = agent.respond("browse", "later", 2, 1)
        self.assertEqual(first["recommendations"][0]["parent_asin"], "T0")
        self.assertEqual(second["recommendations"][0]["parent_asin"], "T0")

        agent.reset("buy", {})
        buying = agent.respond(
            "buy", "I'm looking for shirts. A key requirement is: cotton.", 1, 1
        )
        self.assertEqual(buying["recommendations"][0]["parent_asin"], "B")
        diagnostics = agent.evaluation_diagnostics()
        self.assertEqual(diagnostics["teammate_routes"], 1)
        self.assertEqual(diagnostics["fusion_b_routes"], 1)


if __name__ == "__main__":
    unittest.main()
