from __future__ import annotations

import copy
import unittest
from collections import defaultdict
from unittest.mock import patch

from scripts.official_metric_bridge import (
    rebuild_official_metrics,
    validate_official_metrics,
)
from scripts.p11_gates import (
    MIN_BOOTSTRAP_RESAMPLES,
    _session_contribution,
    evaluate_p11_gates,
)


FLAGS = {
    "exact_repeat": True,
    "contract_clean": True,
    "target_blind": True,
    "network_attempts_zero": True,
    "token_usage_zero": True,
    "exceptions_zero": True,
}


def _session(sample_id: str, rank: int | None, *, scenario: str = "buying") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "hit": rank is not None,
        "first_hit_turn": 1 if rank is not None else None,
        "best_rank": rank,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
    }


def _scenario_metrics(sessions: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in sessions:
        grouped[str(row["scenario_type"])].append(row)
    summaries = {}
    for scenario, rows in sorted(grouped.items()):
        overall = rebuild_official_metrics(rows)
        summaries[scenario] = {
            key: overall[key]
            for key in ("sample_count", "hit_rate_at_10", "mrr", "mttc")
        }
    return summaries


def _run(
    sessions: list[dict[str, object]],
    *,
    wall: float = 100.0,
    p95: float = 100.0,
    rss: int = 100,
) -> dict[str, object]:
    return {
        "sessions": sessions,
        "metrics": rebuild_official_metrics(sessions),
        "scenario_metrics": _scenario_metrics(sessions),
        "resources": {
            "wall_seconds": wall,
            "p95_latency_ms": p95,
            "peak_rss_bytes": rss,
        },
    }


def _improvement_rows(prefix: str, low_mix: int = 3) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    control = []
    candidate = []
    for index in range(10):
        if index < low_mix:
            before, after = 9, 8
        else:
            before, after = 8, 7
        scenario = "buying" if index % 2 == 0 else "open_browsing"
        sample_id = f"{prefix}-{index:02d}"
        control.append(_session(sample_id, before, scenario=scenario))
        candidate.append(_session(sample_id, after, scenario=scenario))
    return control, candidate


def _valid_inputs() -> dict[str, object]:
    primary_control, primary_candidate = _improvement_rows("p")
    confirmation_control, confirmation_candidate = _improvement_rows("c")
    tail = [
        _session(f"u-{index:02d}", 7, scenario="tail_buying")
        for index in range(10)
    ]
    return {
        "primary": {
            "served": _run(copy.deepcopy(primary_control)),
            "control": _run(primary_control),
            "candidate": _run(primary_candidate, wall=115.0, p95=120.0, rss=110),
        },
        "confirmation": {
            "served": _run(copy.deepcopy(confirmation_control)),
            "control": _run(confirmation_control),
            "candidate": _run(
                confirmation_candidate, wall=115.0, p95=120.0, rss=110
            ),
        },
        "uniform_tail": {
            "served": _run(copy.deepcopy(tail)),
            "control": _run(copy.deepcopy(tail)),
            "candidate": _run(copy.deepcopy(tail), wall=115.0, p95=120.0, rss=110),
        },
    }


class P11GateTests(unittest.TestCase):
    def test_bootstrap_session_contribution_equals_unrounded_linear_score(self) -> None:
        sessions = [
            {**_session("one", 1), "first_hit_turn": 2},
            {**_session("two", 5), "first_hit_turn": 7},
            _session("miss", None),
        ]
        count = len(sessions)
        hit_rate = sum(int(row["hit"] is True) for row in sessions) / count
        mrr = sum(float(row["reciprocal_rank"]) for row in sessions) / count
        mttc = sum(
            int(row["first_hit_turn"])
            if row["first_hit_turn"] is not None
            else 11
            for row in sessions
        ) / count
        unrounded_score = 0.5 * hit_rate + 0.3 * mrr + 0.2 * (11 - mttc) / 10

        self.assertAlmostEqual(
            sum(_session_contribution(row) for row in sessions) / count,
            unrounded_score,
            places=15,
        )

    def test_complete_evidence_passes_at_inclusive_boundaries_and_is_stable(self) -> None:
        inputs = _valid_inputs()

        first = evaluate_p11_gates(inputs, FLAGS)
        second = evaluate_p11_gates(inputs, FLAGS)

        self.assertEqual(first, second)
        self.assertEqual(
            set(first), {"passed", "checks", "reasons", "deltas", "ci"}
        )
        self.assertTrue(first["passed"])
        self.assertEqual(first["reasons"], [])
        self.assertEqual(
            first["deltas"]["primary"]["recommended_technical_score"], 0.005
        )
        self.assertEqual(
            first["deltas"]["primary"]["resource_ratios"],
            {
                "wall_seconds": 1.15,
                "p95_latency_ms": 1.2,
                "peak_rss_bytes": 1.1,
            },
        )
        self.assertEqual(
            first["deltas"]["primary"]["resource_ratios_vs_served"],
            first["deltas"]["primary"]["resource_ratios"],
        )
        self.assertGreater(first["ci"]["primary"]["lower"], 0.0)
        self.assertGreater(first["ci"]["confirmation"]["lower"], 0.0)
        self.assertEqual(
            first["ci"]["primary"]["resamples"], MIN_BOOTSTRAP_RESAMPLES
        )

    def test_calls_exact_official_metric_bridge_for_every_run(self) -> None:
        with patch(
            "scripts.p11_gates.validate_official_metrics",
            wraps=validate_official_metrics,
        ) as bridge:
            result = evaluate_p11_gates(_valid_inputs(), FLAGS)

        self.assertTrue(result["passed"])
        self.assertEqual(bridge.call_count, 9)

    def test_primary_absolute_delta_below_boundary_rejects(self) -> None:
        inputs = _valid_inputs()
        control, candidate = _improvement_rows("p", low_mix=4)
        inputs["primary"] = {
            "served": _run(copy.deepcopy(control)),
            "control": _run(control),
            "candidate": _run(candidate),
        }

        result = evaluate_p11_gates(inputs, FLAGS)

        self.assertGreater(
            result["deltas"]["primary"]["recommended_technical_score"], 0.0
        )
        self.assertLess(
            result["deltas"]["primary"]["recommended_technical_score"], 0.005
        )
        self.assertFalse(
            result["checks"]["primary.technical_score_delta_at_least_0_005"]
        )
        self.assertIn(
            "primary.technical_score_delta_at_least_0_005", result["reasons"]
        )

    def test_confirmation_requires_strict_improvement_and_positive_ci(self) -> None:
        inputs = _valid_inputs()
        control = inputs["confirmation"]["control"]["sessions"]
        inputs["confirmation"]["candidate"] = _run(copy.deepcopy(control))

        result = evaluate_p11_gates(inputs, FLAGS)

        self.assertFalse(
            result["checks"]["confirmation.technical_score_strict_increase"]
        )
        self.assertFalse(
            result["checks"]["confirmation.bootstrap_ci_lower_above_zero"]
        )
        self.assertEqual(result["ci"]["confirmation"]["lower"], 0.0)

    def test_bootstrap_rejects_when_improvement_is_not_reliably_paired(self) -> None:
        inputs = _valid_inputs()
        control = [_session(f"p-{index}", 10) for index in range(10)]
        candidate = copy.deepcopy(control)
        candidate[0] = _session("p-0", 1)
        inputs["primary"] = {
            "served": _run(copy.deepcopy(control)),
            "control": _run(control),
            "candidate": _run(candidate),
        }

        result = evaluate_p11_gates(inputs, FLAGS)

        self.assertGreaterEqual(
            result["deltas"]["primary"]["recommended_technical_score"], 0.005
        )
        self.assertEqual(result["ci"]["primary"]["lower"], 0.0)
        self.assertFalse(
            result["checks"]["primary.bootstrap_ci_lower_above_zero"]
        )

    def test_uniform_tail_quality_regression_rejects(self) -> None:
        inputs = _valid_inputs()
        control = [_session(f"u-{index}", 1) for index in range(10)]
        candidate = [_session(f"u-{index}", 2) for index in range(10)]
        inputs["uniform_tail"] = {
            "served": _run(copy.deepcopy(control)),
            "control": _run(control),
            "candidate": _run(candidate),
        }

        result = evaluate_p11_gates(inputs, FLAGS)

        self.assertFalse(result["checks"]["uniform_tail.mrr_non_decrease"])
        self.assertFalse(
            result["checks"]["uniform_tail.technical_score_non_decrease"]
        )

    def test_any_resource_overage_or_non_true_audit_flag_rejects(self) -> None:
        inputs = _valid_inputs()
        inputs["primary"]["candidate"]["resources"]["wall_seconds"] = 115.000001
        flags = dict(FLAGS)
        flags["target_blind"] = 1

        result = evaluate_p11_gates(inputs, flags)

        self.assertFalse(
            result["checks"]["primary.resource.wall_seconds_within_limit"]
        )
        self.assertFalse(result["checks"]["audit.target_blind"])

    def test_resources_must_pass_against_served_and_control(self) -> None:
        inputs = _valid_inputs()
        inputs["primary"]["served"]["resources"]["wall_seconds"] = 99.0

        result = evaluate_p11_gates(inputs, FLAGS)

        self.assertTrue(
            result["checks"]["primary.resource.wall_seconds_within_limit"]
        )
        self.assertFalse(
            result["checks"][
                "primary.resource.wall_seconds_vs_served_within_limit"
            ]
        )

    def test_missing_served_run_fails_closed(self) -> None:
        inputs = _valid_inputs()
        del inputs["primary"]["served"]

        result = evaluate_p11_gates(inputs, FLAGS)

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["primary.served.run_valid"])

    def test_metric_mismatch_of_one_micro_unit_fails_closed(self) -> None:
        inputs = _valid_inputs()
        inputs["primary"]["candidate"]["metrics"]["mrr"] += 0.000001

        result = evaluate_p11_gates(inputs, FLAGS)

        self.assertFalse(
            result["checks"]["primary.candidate.official_metrics_valid"]
        )
        self.assertIn(
            "primary.candidate.official_rounding_mismatch:mrr", result["reasons"]
        )

    def test_too_few_bootstrap_resamples_fails_without_computing_ci(self) -> None:
        result = evaluate_p11_gates(
            _valid_inputs(), FLAGS, bootstrap_resamples=MIN_BOOTSTRAP_RESAMPLES - 1
        )

        self.assertFalse(result["checks"]["bootstrap.resamples_at_least_10000"])
        self.assertNotIn("primary", result["ci"])
        self.assertFalse(
            result["checks"]["primary.bootstrap_ci_lower_above_zero"]
        )

    def test_hit_to_miss_rejects_even_when_total_hit_rate_is_unchanged(self) -> None:
        inputs = _valid_inputs()
        control, candidate = _improvement_rows("p")
        control[0] = _session("p-00", 9, scenario="buying")
        control[1] = _session("p-01", None, scenario="open_browsing")
        candidate[0] = _session("p-00", None, scenario="buying")
        candidate[1] = _session("p-01", 1, scenario="open_browsing")
        inputs["primary"] = {
            "served": _run(copy.deepcopy(control)),
            "control": _run(control),
            "candidate": _run(candidate),
        }

        result = evaluate_p11_gates(inputs, FLAGS)

        self.assertEqual(result["deltas"]["primary"]["hit_to_miss_count"], 1)
        self.assertFalse(result["checks"]["primary.zero_hit_to_miss"])
        self.assertFalse(
            result["checks"]["primary.scenario.buying.hit_rate_non_decrease"]
        )

    def test_malformed_session_ledgers_fail_closed_without_raising(self) -> None:
        mutations = []

        sessions_string = _valid_inputs()
        sessions_string["primary"]["control"]["sessions"] = "not-a-ledger"
        mutations.append(sessions_string)

        non_mapping_row = _valid_inputs()
        non_mapping_row["primary"]["control"]["sessions"][0] = "not-a-row"
        mutations.append(non_mapping_row)

        bad_first_hit_turn = _valid_inputs()
        bad_first_hit_turn["primary"]["control"]["sessions"][0][
            "first_hit_turn"
        ] = "bad"
        mutations.append(bad_first_hit_turn)

        for inputs in mutations:
            with self.subTest(value=inputs["primary"]["control"]["sessions"]):
                result = evaluate_p11_gates(inputs, FLAGS)
                self.assertEqual(
                    set(result), {"passed", "checks", "reasons", "deltas", "ci"}
                )
                self.assertFalse(result["passed"])
                self.assertTrue(result["reasons"])

    def test_official_bridge_exception_fails_closed(self) -> None:
        with patch(
            "scripts.p11_gates.validate_official_metrics",
            side_effect=RuntimeError("fixture bridge failure"),
        ):
            result = evaluate_p11_gates(_valid_inputs(), FLAGS)

        self.assertFalse(result["passed"])
        self.assertTrue(
            all(
                result["checks"][f"{split}.control.official_metrics_valid"]
                is False
                for split in ("primary", "confirmation", "uniform_tail")
            )
        )


if __name__ == "__main__":
    unittest.main()
