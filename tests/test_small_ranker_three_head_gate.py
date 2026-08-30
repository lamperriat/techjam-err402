import unittest

import numpy as np

from scripts.analyze_small_ranker_three_head_gate import single_action_hit_loss


class DirectHitLossLabelsTest(unittest.TestCase):
    def _labels(self, baseline_rank: np.ndarray, positive: np.ndarray) -> dict:
        return {
            "baseline_rank": baseline_rank,
            "positive_index": positive,
            "eligible_from": np.ones(len(baseline_rank), dtype=np.uint8),
        }

    def test_removing_only_hit_is_loss(self) -> None:
        baseline = np.zeros((1, 10), dtype=np.uint8)
        baseline[0, 1] = 10
        positive = np.full((1, 10), -1, dtype=np.int16)
        positive[0, 1] = 3
        chosen = np.zeros((1, 10), dtype=np.uint8)
        chosen[0, 1] = 4
        incumbent = np.zeros((1, 10), dtype=np.uint8)
        loss = single_action_hit_loss(
            self._labels(baseline, positive), chosen, incumbent
        )
        self.assertEqual(int(loss[0, 1]), 1)
        self.assertEqual(int(loss.sum()), 1)

    def test_later_protected_hit_prevents_loss(self) -> None:
        baseline = np.zeros((1, 10), dtype=np.uint8)
        baseline[0, 1] = 10
        baseline[0, 3] = 2
        positive = np.full((1, 10), -1, dtype=np.int16)
        positive[0, 1] = 3
        chosen = np.zeros((1, 10), dtype=np.uint8)
        chosen[0, 1] = 4
        incumbent = np.zeros((1, 10), dtype=np.uint8)
        loss = single_action_hit_loss(
            self._labels(baseline, positive), chosen, incumbent
        )
        self.assertEqual(int(loss.sum()), 0)

    def test_rescue_is_not_loss(self) -> None:
        baseline = np.zeros((1, 10), dtype=np.uint8)
        positive = np.full((1, 10), -1, dtype=np.int16)
        positive[0, 0] = 7
        chosen = np.zeros((1, 10), dtype=np.uint8)
        chosen[0, 0] = 7
        incumbent = np.zeros((1, 10), dtype=np.uint8)
        loss = single_action_hit_loss(
            self._labels(baseline, positive), chosen, incumbent
        )
        self.assertEqual(int(loss.sum()), 0)


if __name__ == "__main__":
    unittest.main()
