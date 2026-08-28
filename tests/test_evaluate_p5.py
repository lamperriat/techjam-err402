from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.evaluate_p5 import (
    ACTIVE_ID,
    CONTROL_ID,
    SHADOW_ID,
    build_confirmation,
    gate_variant,
    load_frozen_inputs,
    main,
    select_winner,
    served_reference_bridge,
    validate_selection_samples,
)


SCENARIOS = ("boundary", "browsing", "buying", "intent_override")


def metric_payload(
    score: float,
    *,
    hit: float = 0.9,
    mrr: float = 0.6,
    mttc: float = 3.0,
    scenario_hit: dict[str, float] | None = None,
) -> dict:
    scenario_hit = scenario_hit or {}
    return {
        "sample_count": 4,
        "hit_rate_at_10": hit,
        "mrr": mrr,
        "mttc": mttc,
        "efficiency": 0.8,
        "recommended_technical_score": score,
        "reported_token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "scenario_metrics": {
            scenario: {
                "sample_count": 1,
                "hit_rate_at_10": scenario_hit.get(scenario, hit),
                "mrr": mrr,
                "mttc": mttc,
            }
            for scenario in SCENARIOS
        },
    }


def session_payload(sample_id: str, *, hit: bool = True) -> dict:
    return {
        "sample_id": sample_id,
        "scenario_type": "buying",
        "hit": hit,
        "first_hit_turn": 1 if hit else None,
        "best_rank": 1 if hit else None,
        "reciprocal_rank": 1.0 if hit else 0.0,
    }


def run_payload(
    variant_id: str,
    score: float,
    *,
    functional_hash: str | None = None,
    response_trace_hash: str | None = None,
    hit: float = 0.9,
    mrr: float = 0.6,
    mttc: float = 3.0,
    evaluation_seconds: float = 10.0,
    scenario_hit: dict[str, float] | None = None,
) -> dict:
    resolved_functional_hash = functional_hash or variant_id
    return {
        "variant_id": variant_id,
        "stats": {"activations": 2, "output_changes": 1},
        "timing": {"evaluation_seconds": evaluation_seconds},
        "contract_errors": [],
        "metrics": metric_payload(
            score,
            hit=hit,
            mrr=mrr,
            mttc=mttc,
            scenario_hit=scenario_hit,
        ),
        "functional_result_sha256": resolved_functional_hash,
        "response_trace_sha256": response_trace_hash or resolved_functional_hash,
        "sessions": [session_payload(f"derived_p5_{index:04d}") for index in range(1, 5)],
    }


class P5RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_ids = {f"derived_p5_{index:04d}" for index in range(1, 5)}
        self.control = run_payload(
            CONTROL_ID,
            0.80,
            functional_hash="control-functional",
        )

    def test_selection_validation_checks_both_exclusion_sets(self) -> None:
        samples = [
            {
                "sample_id": f"derived_p5_{index:04d}",
                "scenario_type": "buying",
                "ground_truth": {"parent_asin": f"P5-{index}"},
            }
            for index in range(1, 5)
        ]
        public = [
            {"ground_truth": {"parent_asin": f"PUBLIC-{index}"}}
            for index in range(1, 3)
        ]
        prior = [
            {"ground_truth": {"parent_asin": f"PRIOR-{index}"}}
            for index in range(1, 3)
        ]
        validation = validate_selection_samples(
            samples,
            public,
            prior,
            {f"P5-{index}" for index in range(1, 5)},
            expected_count=4,
            expected_exclusion_count=2,
        )
        self.assertEqual(validation["released_public_target_overlap"], 0)
        self.assertEqual(validation["prior_p1_derived_target_overlap"], 0)
        self.assertFalse(validation["released_public_evaluated"])

        prior[0]["ground_truth"]["parent_asin"] = "P5-1"
        with self.assertRaisesRegex(ValueError, "prior P1-derived targets"):
            validate_selection_samples(
                samples,
                public,
                prior,
                {f"P5-{index}" for index in range(1, 5)},
                expected_count=4,
                expected_exclusion_count=2,
            )

    def test_frozen_selection_file_hash_is_hard_gated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.jsonl"
            selection.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_frozen_inputs(
                    root / "catalog.jsonl",
                    selection,
                    root / "public.jsonl",
                    root / "prior.jsonl",
                )

    def test_selection_validation_rejects_non_p5_id_and_catalog_miss(self) -> None:
        samples = [
            {
                "sample_id": f"derived_p5_{index:04d}",
                "scenario_type": "buying",
                "ground_truth": {"parent_asin": f"P5-{index}"},
            }
            for index in range(1, 5)
        ]
        public = [{"ground_truth": {"parent_asin": "PUBLIC"}}]
        prior = [{"ground_truth": {"parent_asin": "PRIOR"}}]
        samples[0]["sample_id"] = "derived_p1_0001"
        with self.assertRaisesRegex(ValueError, "derived_p5_"):
            validate_selection_samples(
                samples,
                public,
                prior,
                {f"P5-{index}" for index in range(1, 5)},
                expected_count=4,
                expected_exclusion_count=1,
            )
        samples[0]["sample_id"] = "derived_p5_0001"
        with self.assertRaisesRegex(ValueError, "frozen catalog"):
            validate_selection_samples(
                samples,
                public,
                prior,
                {"P5-1", "P5-2", "P5-3"},
                expected_count=4,
                expected_exclusion_count=1,
            )

    def test_active_gate_requires_strict_score_gain_and_runtime_cap(self) -> None:
        candidate = run_payload(
            ACTIVE_ID,
            0.81,
            hit=0.91,
            mrr=0.61,
            mttc=2.9,
            evaluation_seconds=13.0,
        )
        gated = gate_variant(candidate, self.control, self.control, self.sample_ids)
        self.assertEqual(gated["decision"], "eligible")
        self.assertTrue(gated["gates"]["evaluation_time_within_1_30x"])

        candidate["metrics"]["recommended_technical_score"] = 0.80
        gated = gate_variant(candidate, self.control, self.control, self.sample_ids)
        self.assertEqual(gated["decision"], "reject")
        self.assertFalse(gated["gates"]["technical_score_strict_improvement"])

        candidate["metrics"]["recommended_technical_score"] = 0.81
        candidate["timing"]["evaluation_seconds"] = 13.000001
        gated = gate_variant(candidate, self.control, self.control, self.sample_ids)
        self.assertEqual(gated["decision"], "reject")
        self.assertFalse(gated["gates"]["evaluation_time_within_1_30x"])

    def test_control_gate_requires_exact_served_coverage_reference(self) -> None:
        reference = {
            **self.control,
            "variant_id": "served.Agent.coverage_off",
            "sessions": [dict(item) for item in self.control["sessions"]],
        }
        bridge = served_reference_bridge(self.control, reference, self.sample_ids)
        self.assertTrue(bridge["passed"])
        gated = gate_variant(
            self.control, self.control, reference, self.sample_ids
        )
        self.assertEqual(gated["decision"], "control")

        reference["functional_result_sha256"] = "different"
        bridge = served_reference_bridge(self.control, reference, self.sample_ids)
        self.assertFalse(bridge["passed"])
        gated = gate_variant(
            self.control, self.control, reference, self.sample_ids
        )
        self.assertEqual(gated["decision"], "invalid_control")

        reference["functional_result_sha256"] = "control-functional"
        reference["response_trace_sha256"] = "different-response"
        bridge = served_reference_bridge(self.control, reference, self.sample_ids)
        self.assertFalse(bridge["passed"])
        self.assertFalse(
            bridge["checks"]["control_response_trace_equals_served_reference"]
        )

    def test_active_gate_rejects_scenario_regression_and_hit_to_miss(self) -> None:
        candidate = run_payload(
            ACTIVE_ID,
            0.81,
            hit=0.91,
            mrr=0.61,
            mttc=2.9,
            scenario_hit={"boundary": 0.8},
        )
        candidate["sessions"][0] = session_payload("derived_p5_0001", hit=False)
        gated = gate_variant(candidate, self.control, self.control, self.sample_ids)
        self.assertEqual(gated["decision"], "reject")
        self.assertFalse(gated["gates"]["scenario_hit_rate_non_decrease"])
        self.assertFalse(gated["gates"]["zero_hit_to_miss"])
        self.assertEqual(gated["scenario_hit_rate_regressions"], ["boundary"])

    def test_shadow_must_be_functionally_equal_and_is_never_selectable(self) -> None:
        shadow = run_payload(
            SHADOW_ID,
            0.80,
            functional_hash="control-functional",
        )
        shadow_gate = gate_variant(shadow, self.control, self.control, self.sample_ids)
        self.assertEqual(shadow_gate["decision"], "shadow_only")

        active = run_payload(ACTIVE_ID, 0.79)
        active_gate = gate_variant(active, self.control, self.control, self.sample_ids)
        control_gate = gate_variant(
            self.control, self.control, self.control, self.sample_ids
        )
        selection = select_winner(
            {
                CONTROL_ID: control_gate,
                SHADOW_ID: shadow_gate,
                ACTIVE_ID: active_gate,
            },
            {"attempted": False, "passed": False},
        )
        self.assertEqual(selection["winner_id"], CONTROL_ID)
        self.assertFalse(selection["shadow_can_win"])
        self.assertNotIn(SHADOW_ID, selection["selectable_variant_ids"])

        shadow["functional_result_sha256"] = "changed"
        invalid_gate = gate_variant(
            shadow, self.control, self.control, self.sample_ids
        )
        self.assertEqual(invalid_gate["decision"], "invalid_shadow")

        shadow["functional_result_sha256"] = "control-functional"
        shadow["response_trace_sha256"] = "changed-response"
        invalid_gate = gate_variant(
            shadow, self.control, self.control, self.sample_ids
        )
        self.assertEqual(invalid_gate["decision"], "invalid_shadow")
        self.assertFalse(invalid_gate["gates"]["response_trace_equals_control"])

    def test_active_wins_only_after_exact_repeat_confirmation(self) -> None:
        shadow = run_payload(
            SHADOW_ID,
            0.80,
            functional_hash="control-functional",
        )
        active = run_payload(
            ACTIVE_ID,
            0.81,
            hit=0.91,
            mrr=0.61,
            mttc=2.9,
            functional_hash="active-functional",
        )
        gates = {
            CONTROL_ID: gate_variant(
                self.control, self.control, self.control, self.sample_ids
            ),
            SHADOW_ID: gate_variant(
                shadow, self.control, self.control, self.sample_ids
            ),
            ACTIVE_ID: gate_variant(
                active, self.control, self.control, self.sample_ids
            ),
        }
        repeat = {**active, "sessions": [dict(item) for item in active["sessions"]]}
        confirmation = build_confirmation(active, repeat, self.sample_ids)
        self.assertTrue(confirmation["passed"])
        selection = select_winner(gates, confirmation)
        self.assertEqual(selection["decision"], "promote_active")
        self.assertEqual(selection["winner_id"], ACTIVE_ID)

        repeat["functional_result_sha256"] = "different"
        failed = build_confirmation(active, repeat, self.sample_ids)
        retained = select_winner(gates, failed)
        self.assertEqual(retained["winner_id"], CONTROL_ID)
        self.assertEqual(retained["decision"], "retain_control_confirmation_failed")

        repeat["functional_result_sha256"] = "active-functional"
        repeat["response_trace_sha256"] = "different-response"
        failed = build_confirmation(active, repeat, self.sample_ids)
        self.assertFalse(failed["checks"]["strict_response_trace_hash_equal"])
        self.assertFalse(failed["passed"])

    def test_cli_writes_artifact_and_returns_success_for_valid_experiment(self) -> None:
        artifact = {
            "selection": {
                "decision": "retain_control_active_rejected",
                "winner_id": CONTROL_ID,
                "experiment_valid": True,
            },
            "corpus": {"sha256": "frozen"},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with mock.patch(
                "scripts.evaluate_p5.run_selection", return_value=artifact
            ) as runner:
                exit_code = main(["--output", str(output)])
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), artifact)
            runner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
