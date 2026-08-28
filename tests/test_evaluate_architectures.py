from __future__ import annotations

import unittest

from scripts.evaluate_architectures import (
    CONTROL_ID,
    ContractRecorder,
    assert_control_integrity,
    count_effective_non_control,
    gate_variant,
    select_candidates,
    validate_selection_samples,
    validate_response,
)


def metrics(score: float, *, hit: float = 0.9, mrr: float = 0.6, mttc: float = 3.0) -> dict:
    scenarios = {
        name: {"sample_count": 1, "hit_rate_at_10": hit, "mrr": mrr, "mttc": mttc}
        for name in ("buying", "browsing", "intent_override", "boundary")
    }
    return {
        "sample_count": 1,
        "hit_rate_at_10": hit,
        "mrr": mrr,
        "mttc": mttc,
        "efficiency": 0.8,
        "recommended_technical_score": score,
        "reported_token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "scenario_metrics": scenarios,
    }


def run(
    variant_id: str,
    score: float,
    *,
    activations: int = 1,
    output_changes: int = 1,
) -> dict:
    return {
        "variant_id": variant_id,
        "metrics": metrics(score),
        "stats": {"activations": activations, "output_changes": output_changes},
        "contract_errors": [],
        "sessions": [{
            "sample_id": "derived_1",
            "scenario_type": "buying",
            "hit": True,
            "first_hit_turn": 1,
            "best_rank": 1,
            "reciprocal_rank": 1.0,
        }],
    }


class ArchitectureRunnerTests(unittest.TestCase):
    def test_contract_validator_is_stricter_than_public_evaluator(self) -> None:
        valid = {
            "message": "Here are matches.",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": "A"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        self.assertEqual(validate_response(valid, {"A"}), [])

        invalid = {
            **valid,
            "recommendations": ["A"],
            "debug": {"target_rank": 1},
        }
        errors = validate_response(invalid, {"A"})
        self.assertIn("response contains undeclared keys", errors)
        self.assertIn("recommendation is not a contract object", errors)
        self.assertIn(
            "ask_attribute is outside the official enum",
            validate_response({**valid, "ask_attribute": ["color"]}, {"A"}),
        )
        self.assertIn(
            "recommendation score is not a finite number",
            validate_response(
                {**valid, "recommendations": [{"parent_asin": "A", "score": float("nan")}]},
                {"A"},
            ),
        )

    def test_contract_recorder_records_delegate_exceptions_before_evaluator_swallow(self) -> None:
        class RaisingAgent:
            def respond(self, *args: object) -> dict:
                raise RuntimeError("boom")

        recorder = ContractRecorder(RaisingAgent(), {"A"})  # type: ignore[arg-type]
        with self.assertRaisesRegex(RuntimeError, "boom"):
            recorder.respond("session", "message", 2, 10)
        self.assertTrue(any("turn 2: RuntimeError: boom" in item for item in recorder.errors))

    def test_inactive_architecture_is_not_counted(self) -> None:
        control = run(CONTROL_ID, 0.8, activations=0, output_changes=0)
        candidate = run("R00.inactive", 0.8, activations=0, output_changes=0)
        candidate.update(gate_variant(candidate, control))
        self.assertEqual(candidate["decision"], "not_counted")
        self.assertFalse(candidate["gates"]["effective_architecture"])

    def test_invalid_control_is_rejected_before_comparison(self) -> None:
        control = run(CONTROL_ID, 0.8, activations=0, output_changes=0)
        control["contract_errors"] = ["turn 1: invalid response"]
        gated = gate_variant(control, control)
        self.assertEqual(gated["decision"], "invalid_control")
        with self.assertRaisesRegex(RuntimeError, "control integrity failure"):
            assert_control_integrity(control, 1)

    def test_contract_invalid_architecture_does_not_count_as_effective(self) -> None:
        control = run(CONTROL_ID, 0.8, activations=0, output_changes=0)
        control.update(gate_variant(control, control))
        valid_reject = run("R01.valid_reject", 0.7)
        valid_reject.update(gate_variant(valid_reject, control))
        invalid = run("R02.invalid", 0.9)
        invalid["contract_errors"] = ["turn 1: invalid response"]
        invalid.update(gate_variant(invalid, control))

        self.assertEqual(count_effective_non_control([control, valid_reject, invalid]), 1)
        selection = select_candidates([control, valid_reject, invalid], confirm_top=3)
        self.assertNotIn("R02.invalid", selection["confirmation_candidates"])

    def test_gate_rejects_hit_to_miss_even_when_aggregate_score_ties(self) -> None:
        control = run(CONTROL_ID, 0.8)
        candidate = run("R00.regression", 0.8)
        candidate["sessions"][0].update({
            "hit": False,
            "first_hit_turn": None,
            "best_rank": None,
            "reciprocal_rank": 0.0,
        })
        candidate.update(gate_variant(candidate, control))
        self.assertEqual(candidate["decision"], "reject")
        self.assertFalse(candidate["gates"]["zero_hit_to_miss"])

    def test_selection_reports_raw_and_eligible_winners_separately(self) -> None:
        control = run(CONTROL_ID, 0.80, activations=0, output_changes=0)
        control["decision"] = "control"
        eligible = run("R01.eligible", 0.81)
        eligible["decision"] = "eligible"
        rejected = run("R02.raw", 0.82)
        rejected["decision"] = "reject"
        selection = select_candidates(
            [control, eligible, rejected],
            confirm_top=2,
        )

        self.assertEqual(selection["raw_score_winner"], "R02.raw")
        self.assertEqual(
            selection["eligible_winner_before_confirmation"], "R01.eligible"
        )
        self.assertIn(CONTROL_ID, selection["confirmation_candidates"])

    def test_confirmation_always_includes_eligible_winner_behind_rejected_raw_scores(self) -> None:
        control = run(CONTROL_ID, 0.80, activations=0, output_changes=0)
        control["decision"] = "control"
        eligible = run("R01.eligible", 0.81)
        eligible["decision"] = "eligible"
        rejected = run("R02.rejected", 0.99)
        rejected["decision"] = "reject"

        selection = select_candidates(
            [control, eligible, rejected],
            confirm_top=1,
        )

        self.assertIn("R01.eligible", selection["confirmation_candidates"])
        self.assertIn(CONTROL_ID, selection["confirmation_candidates"])

    def test_selection_corpus_validation_rejects_duplicates_and_wrong_mix(self) -> None:
        public = [{"ground_truth": {"parent_asin": "PUBLIC"}}]
        samples = [
            {
                "sample_id": "duplicate",
                "ground_truth": {"parent_asin": "A"},
                "scenario_type": "buying",
            },
            {
                "sample_id": "duplicate",
                "ground_truth": {"parent_asin": "B"},
                "scenario_type": "buying",
            },
        ]
        with self.assertRaisesRegex(ValueError, "sample IDs"):
            validate_selection_samples(
                samples,
                public,
                {"A", "B", "PUBLIC"},
                expected_count=2,
            )


if __name__ == "__main__":
    unittest.main()
