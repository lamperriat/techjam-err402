from __future__ import annotations

import unittest

from scripts.compare_frozen_winner import _metric_delta, _scenario_hit_regressions


class FrozenWinnerComparisonTest(unittest.TestCase):
    def test_metric_delta_uses_candidate_minus_control(self) -> None:
        control = {
            "hit_rate_at_10": 0.9,
            "mrr": 0.4,
            "mttc": 4.0,
            "efficiency": 0.7,
            "recommended_technical_score": 0.7,
        }
        candidate = {
            "hit_rate_at_10": 0.95,
            "mrr": 0.5,
            "mttc": 3.5,
            "efficiency": 0.75,
            "recommended_technical_score": 0.8,
        }
        delta = _metric_delta(candidate, control)
        self.assertEqual(delta["hit_rate_at_10"], 0.05)
        self.assertEqual(delta["mttc"], -0.5)

    def test_scenario_gate_is_hit_rate_specific(self) -> None:
        control = {
            "scenario_metrics": {
                "buying": {"hit_rate_at_10": 0.9, "mrr": 0.7}
            }
        }
        candidate = {
            "scenario_metrics": {
                "buying": {"hit_rate_at_10": 0.9, "mrr": 0.6}
            }
        }
        self.assertEqual(_scenario_hit_regressions(candidate, control), [])


if __name__ == "__main__":
    unittest.main()
