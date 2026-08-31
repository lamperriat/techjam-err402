from types import SimpleNamespace
import unittest

from starter.teammate_safe_deltas import P11TailUnionAgent
from vendor.teammate_v1.err402.retrieval.catalog import CandidatePool

class FakeCatalog:
    def __init__(self):
        self.current = ("A", "U")
        self.products = {
            key: SimpleNamespace(parent_asin=key)
            for key in ("A", "U", "S", "X", "Y", "N1", "N2")
        }

    def candidates(self, _category, _query):
        return CandidatePool(self.current, {key: index + 1 for index, key in enumerate(self.current)})


class FakeScorer:
    def score(self, pool, _context):
        scores = {"N1": 0.2, "N2": 0.9}
        return [
            SimpleNamespace(product=SimpleNamespace(parent_asin=key), score=scores[key])
            for key in sorted(pool.parent_asins, key=lambda value: -scores[value])
        ]


class FakeBase:
    def __init__(self, responses):
        self.responses = list(responses)
        self.catalog = FakeCatalog()
        self.scorer = FakeScorer()
        self._sessions = {}

    def reset(self, session_id, _profile):
        self._sessions[session_id] = SimpleNamespace(
            intent="buying", category="hats", constraints=[], shown_product_ids=set()
        )

    def respond(self, session_id, _message, _turn, _top_k):
        response = self.responses.pop(0)
        self._sessions[session_id].shown_product_ids.update(
            row["parent_asin"] for row in response["recommendations"]
        )
        return response

    def close(self):
        pass


def response(*identifiers):
    return {"message": "v1", "ask_attribute": "other", "recommendations": [
        {"parent_asin": key, "score": 1.0} for key in identifiers
    ], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


class TeammateSafeDeltaTests(unittest.TestCase):
    def test_first_and_full_pages_are_same_objects_without_tail_lookup(self):
        first, full, switched = response("S"), response("X", "Y"), response("S", "Y")
        calls = []
        base = FakeBase([first, full, switched])
        agent = P11TailUnionAgent(base_agent=base, tail_provider=lambda *args: calls.append(args), state_guard=True)
        agent.reset("s", {})
        self.assertIs(agent.respond("s", "one", 1, 2), first)
        self.assertIs(agent.respond("s", "two", 2, 2), full)
        self.assertIs(agent.respond("s", "Switch from hats to shoes.", 3, 2), switched)
        self.assertEqual(calls, [])
        self.assertEqual(base._sessions["s"].shown_product_ids, {"S", "Y"})
        self.assertEqual(agent.evaluation_diagnostics()["s1_activations"], 1)

    def test_short_later_page_appends_only_scored_unseen_newcomers(self):
        first, short = response("S", "X", "Y"), response("U")
        histories = []
        provider = lambda _sid, _profile, history: histories.append(history) or ("A", "S", "N1", "N1", "N2", "BAD")
        base = FakeBase([first, short])
        agent = P11TailUnionAgent(base_agent=base, tail_provider=provider)
        agent.reset("s", {"preference_tags": []})
        agent.respond("s", "one", 1, 3)
        actual = agent.respond("s", "two", 2, 3)
        self.assertEqual([row["parent_asin"] for row in actual["recommendations"]], ["U", "N2", "N1"])
        self.assertEqual([row["score"] for row in actual["recommendations"]], [1.0, 0.9, 0.2])
        self.assertEqual(histories, [( (1, "one"), (2, "two") )])
        self.assertEqual(base._sessions["s"].shown_product_ids & {"N1", "N2"}, {"N1", "N2"})
        self.assertEqual(agent.evaluation_diagnostics()["activations"], 1)

    def test_tail_failure_returns_exact_t0_object_without_extra_commit(self):
        first, short = response("S", "X"), response("U")
        def fail(*_args):
            raise RuntimeError("no tail")
        base = FakeBase([first, short])
        agent = P11TailUnionAgent(base_agent=base, tail_provider=fail)
        agent.reset("s", {})
        agent.respond("s", "one", 1, 2)
        self.assertIs(agent.respond("s", "two", 2, 2), short)
        self.assertEqual(base._sessions["s"].shown_product_ids, {"S", "X", "U"})
        self.assertEqual(agent.evaluation_diagnostics()["fallbacks"], 1)


if __name__ == "__main__":
    unittest.main()
