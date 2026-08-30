import unittest

import numpy as np

from scripts.analyze_small_ranker_rr_regret_gate import single_action_rr_regret


class ReciprocalRankRegretLabelsTest(unittest.TestCase):
    def _labels(self, baseline_rank: np.ndarray, positive: np.ndarray) -> dict:
        return {
            "baseline_rank": baseline_rank,
            "positive_index": positive,
            "eligible_from": np.ones(len(baseline_rank), dtype=np.uint8),
        }

    def test_early_rank10_preempts_better_later_hit(self) -> None:
        baseline = np.zeros((1, 10), dtype=np.uint8)
        baseline[0, 1] = 1
        positive = np.full((1, 10), -1, dtype=np.int16)
        positive[0, 0] = 7
        chosen = np.zeros((1, 10), dtype=np.uint8)
        chosen[0, 0] = 7
        incumbent = np.zeros((1, 10), dtype=np.uint8)
        regret = single_action_rr_regret(
            self._labels(baseline, positive), chosen, incumbent
        )
        self.assertAlmostEqual(float(regret[0, 0]), 0.9, places=6)
        self.assertEqual(int(np.count_nonzero(regret)), 1)

    def test_removing_only_rank10_hit_has_point_one_regret(self) -> None:
        baseline = np.zeros((1, 10), dtype=np.uint8)
        baseline[0, 1] = 10
        positive = np.full((1, 10), -1, dtype=np.int16)
        positive[0, 1] = 3
        chosen = np.zeros((1, 10), dtype=np.uint8)
        chosen[0, 1] = 4
        incumbent = np.zeros((1, 10), dtype=np.uint8)
        regret = single_action_rr_regret(
            self._labels(baseline, positive), chosen, incumbent
        )
        self.assertAlmostEqual(float(regret[0, 1]), 0.1, places=6)

    def test_true_rescue_does_not_create_regret(self) -> None:
        baseline = np.zeros((1, 10), dtype=np.uint8)
        positive = np.full((1, 10), -1, dtype=np.int16)
        positive[0, 0] = 7
        chosen = np.zeros((1, 10), dtype=np.uint8)
        chosen[0, 0] = 7
        incumbent = np.zeros((1, 10), dtype=np.uint8)
        regret = single_action_rr_regret(
            self._labels(baseline, positive), chosen, incumbent
        )
        self.assertEqual(float(regret.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
