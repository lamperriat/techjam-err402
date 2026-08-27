from __future__ import annotations

import unittest

from scripts.evaluate_generalization import (
    AUDIT_RULES,
    CHALLENGE_RULES,
    DEV_RULES,
    PerturbedAgent,
    _session_changes,
    _suite_names,
    _suite_registry_sha256,
    build_product_disjoint_samples,
    perturb_message,
)


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, int]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.profile = (session_id, user_profile)

    def respond(self, session_id: str, message: str, turn: int, top_k: int) -> dict:
        self.calls.append((session_id, message, turn, top_k))
        return {"message": "ok", "ask_attribute": None, "recommendations": []}


class GeneralizationTest(unittest.TestCase):
    def test_dev_and_challenge_paraphrases_preserve_visible_constraint(self) -> None:
        original = (
            "Actually, ignore my earlier preference. "
            "What I need is: blue cotton casual."
        )
        dev, dev_applied = perturb_message(original, DEV_RULES)
        challenge, challenge_applied = perturb_message(original, CHALLENGE_RULES)
        audit, audit_applied = perturb_message(original, AUDIT_RULES)

        self.assertIn("blue cotton casual", dev)
        self.assertIn("blue cotton casual", challenge)
        self.assertIn("blue cotton casual", audit)
        self.assertIn("override_dev", dev_applied)
        self.assertIn("override_challenge", challenge_applied)
        self.assertIn("override_audit", audit_applied)
        self.assertNotEqual(dev, challenge)

    def test_adapter_changes_only_visible_message(self) -> None:
        delegate = FakeAgent()
        adapter = PerturbedAgent(delegate, DEV_RULES)
        profile = {"summary": "neutral"}
        adapter.reset("opaque", profile)
        response = adapter.respond(
            "opaque", "I'm looking for shoes, but I'm still exploring.", 1, 10
        )

        self.assertEqual(delegate.profile, ("opaque", profile))
        self.assertEqual(delegate.calls[0][0::2], ("opaque", 1))
        self.assertIn("I need shoes", delegate.calls[0][1])
        self.assertEqual(response["message"], "ok")
        self.assertEqual(adapter.stats()["transformed_messages"], 1)

    def test_product_disjoint_samples_are_deterministic_and_stratified(self) -> None:
        public = [
            {
                "sample_id": "public_1",
                "ground_truth": {"parent_asin": "PUBLIC"},
            }
        ]
        products = {
            "PUBLIC": {"title": "Public", "categories": ["Clothing"]},
            **{
                f"P{index:03d}": {
                    "title": f"Product {index}",
                    "categories": ["Clothing", "Test"],
                }
                for index in range(250)
            },
        }
        first, first_metadata = build_product_disjoint_samples(
            public, products, 200, "fixed-seed"
        )
        second, second_metadata = build_product_disjoint_samples(
            public, products, 200, "fixed-seed"
        )

        self.assertEqual(first, second)
        self.assertEqual(first_metadata["samples_sha256"], second_metadata["samples_sha256"])
        self.assertEqual(first_metadata["public_target_overlap"], 0)
        self.assertEqual(first_metadata["unique_target_count"], 200)
        self.assertEqual(
            first_metadata["scenario_counts"],
            {"boundary": 10, "browsing": 80, "buying": 80, "intent_override": 30},
        )
        self.assertNotIn(
            "PUBLIC", {sample["ground_truth"]["parent_asin"] for sample in first}
        )

    def test_suite_selection_composes_default_and_audit(self) -> None:
        self.assertEqual(
            _suite_names(["default", "combined_audit"]),
            ["combined_dev", "combined_challenge", "combined_audit"],
        )
        self.assertIn("override_dev", _suite_names(["all"]))
        self.assertEqual(len(_suite_registry_sha256()), 64)

    def test_session_change_uses_official_weighted_contribution(self) -> None:
        baseline = {
            "sessions": [{
                "sample_id": "paired",
                "hit": True,
                "first_hit_turn": 1,
                "best_rank": 10,
                "reciprocal_rank": 0.1,
            }]
        }
        current = {
            "sessions": [{
                "sample_id": "paired",
                "hit": True,
                "first_hit_turn": 2,
                "best_rank": 1,
                "reciprocal_rank": 1.0,
            }]
        }

        changes = _session_changes(current, baseline)

        self.assertEqual(changes["official_score_improvement_count"], 1)
        self.assertEqual(changes["later_hit_count"], 1)
        self.assertEqual(changes["rank_improvement_count"], 1)


if __name__ == "__main__":
    unittest.main()
