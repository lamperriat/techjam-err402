from __future__ import annotations

import numpy as np

from scripts import analyze_small_ranker_metric_gate as diagnostic


def _labels() -> dict[str, np.ndarray]:
    return {
        "baseline_rank": np.asarray([[10, 0], [0, 0]], dtype=np.uint8),
        "positive_index": np.asarray([[4, -1], [-1, 3]], dtype=np.int16),
        "eligible_from": np.asarray([1, 1], dtype=np.uint8),
    }


def test_policy_state_protects_top9_and_replaces_slot10() -> None:
    labels = _labels()
    chosen = np.asarray([[3, 2], [2, 3]], dtype=np.uint8)
    active = np.asarray([[True, False], [False, True]])
    state = diagnostic.policy_session_state(labels, chosen, active)
    np.testing.assert_array_equal(state["hit"], [False, True])
    np.testing.assert_array_equal(state["first_rank"], [0, 10])
    np.testing.assert_array_equal(state["first_turn"], [11, 2])


def test_official_metrics_and_transitions_capture_rank_tradeoff() -> None:
    labels = _labels()
    chosen = np.asarray([[3, 2], [2, 3]], dtype=np.uint8)
    inactive = np.zeros((2, 2), dtype=bool)
    active = np.asarray([[False, False], [False, True]])
    baseline = diagnostic.policy_session_state(labels, chosen, inactive)
    policy = diagnostic.policy_session_state(labels, chosen, active)
    result = diagnostic.transition_metrics(
        baseline, policy, active, np.asarray([True, True])
    )
    assert result["miss_to_hit"] == 1
    assert result["hit_to_miss"] == 0
    assert result["net_hits"] == 1
    assert result["mrr_delta"] > 0
    assert result["mttc_delta"] < 0
