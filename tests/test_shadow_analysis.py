from __future__ import annotations

import json
import unittest

from observer.shadow_analysis import ShadowPolicyRecorder


class ShadowPolicyRecorderTests(unittest.TestCase):
    def test_aggregates_disagreements_components_and_scenarios(self) -> None:
        recorder = ShadowPolicyRecorder()
        recorder.record(
            sample_id="public_1",
            scenario_type="buying",
            turn=1,
            actual_attribute="other",
            question_shadow={
                "candidate_count": 50,
                "blocked_attributes": ["category"],
                "selected_attribute": "color",
                "candidates": [{
                    "score": 0.7,
                    "information_gain": 0.9,
                    "coverage": 1.0,
                    "answerability": 1.0,
                    "turn_cost": 0.05,
                }],
            },
        )
        recorder.record(
            sample_id="public_2",
            scenario_type="boundary",
            turn=1,
            actual_attribute=None,
            question_shadow={
                "candidate_count": 0,
                "blocked_attributes": [],
                "selected_attribute": None,
                "candidates": [],
                "reason": "disabled",
            },
        )

        artifact = recorder.artifact()

        self.assertTrue(artifact["target_blind"])
        self.assertNotIn("ground_truth", json.dumps(artifact["events"]))
        self.assertEqual(artifact["summary"]["turn_count"], 2)
        self.assertEqual(artifact["summary"]["disagreement_count"], 1)
        self.assertEqual(artifact["summary"]["shadow_attribute_counts"]["color"], 1)
        self.assertEqual(
            artifact["scenario_summaries"]["buying"]["selected_component_means"][
                "information_gain"
            ],
            0.9,
        )

    def test_budget_alias_is_checked_against_blocked_price_slot(self) -> None:
        recorder = ShadowPolicyRecorder()
        recorder.record(
            sample_id="derived_1",
            scenario_type="buying",
            turn=2,
            actual_attribute="budget",
            question_shadow={
                "candidate_count": 10,
                "blocked_attributes": ["price"],
                "selected_attribute": "budget",
                "candidates": [{"score": 0.5}],
            },
        )

        self.assertEqual(
            recorder.artifact()["summary"]["blocked_selection_violations"], 1
        )


if __name__ == "__main__":
    unittest.main()
