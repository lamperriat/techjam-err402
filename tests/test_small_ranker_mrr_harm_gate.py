from __future__ import annotations

import numpy as np

from scripts import analyze_small_ranker_mrr_harm_gate as diagnostic


def test_first_hit_harm_labels_only_earliest_rank10_removal() -> None:
    labels = {
        "baseline_rank": np.asarray([[10, 10, 0], [5, 10, 0]], dtype=np.uint8),
        "positive_index": np.asarray([[7, 8, -1], [4, 8, -1]], dtype=np.int16),
        "eligible_from": np.asarray([1, 1], dtype=np.uint8),
    }
    chosen = np.asarray([[3, 3, 3], [3, 3, 3]], dtype=np.uint8)
    incumbent = np.asarray([[2, 2, 2], [2, 2, 2]], dtype=np.uint8)
    result = diagnostic.first_hit_harm_labels(labels, chosen, incumbent)
    np.testing.assert_array_equal(result, [[1, 0, 0], [0, 0, 0]])


def test_first_hit_harm_is_cleared_when_challenger_is_target() -> None:
    labels = {
        "baseline_rank": np.asarray([[10]], dtype=np.uint8),
        "positive_index": np.asarray([[7]], dtype=np.int16),
        "eligible_from": np.asarray([1], dtype=np.uint8),
    }
    chosen = np.asarray([[7]], dtype=np.uint8)
    incumbent = np.asarray([[2]], dtype=np.uint8)
    result = diagnostic.first_hit_harm_labels(labels, chosen, incumbent)
    assert result.item() == 0
