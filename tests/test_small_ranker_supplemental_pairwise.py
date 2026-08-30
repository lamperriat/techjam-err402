from __future__ import annotations

import numpy as np

from scripts import evaluate_small_ranker_supplemental_pairwise as subject


def test_compose_policy_keeps_current_and_overrides_only_supplement() -> None:
    current_chosen = np.asarray([[4, 5], [6, 7]], dtype=np.uint8)
    current_activation = np.asarray([[True, False], [False, True]])
    pairwise_chosen = np.asarray([[8, 9], [10, 11]], dtype=np.uint8)
    keep = np.zeros_like(current_activation)
    chosen, activation = subject._compose_policy(
        current_chosen, current_activation, pairwise_chosen, keep
    )
    assert np.array_equal(chosen, current_chosen)
    assert np.array_equal(activation, current_activation)

    supplement = np.asarray([[False, True], [True, False]])
    chosen, activation = subject._compose_policy(
        current_chosen, current_activation, pairwise_chosen, supplement
    )
    assert chosen.tolist() == [[4, 9], [10, 7]]
    assert activation.tolist() == [[True, True], [True, True]]


def test_isolated_labels_are_relative_to_complete_current_policy() -> None:
    labels = {
        "baseline_rank": np.asarray([[0, 0], [10, 0], [10, 0]], dtype=np.int16),
        "positive_index": np.asarray([[3, -1], [4, -1], [5, -1]], dtype=np.int16),
        "eligible_from": np.asarray([1, 1, 1], dtype=np.uint8),
    }
    current_chosen = np.asarray([[1, 1], [4, 1], [5, 1]], dtype=np.uint8)
    current_activation = np.asarray(
        [[False, False], [True, False], [True, False]]
    )
    pairwise_chosen = np.asarray([[3, 2], [8, 2], [8, 9]], dtype=np.uint8)
    action = np.asarray([[True, False], [True, False], [False, True]])

    rescue, regret, rr_loss, mttc_loss = subject._isolated_labels(
        labels,
        current_chosen,
        current_activation,
        pairwise_chosen,
        action,
    )

    assert rescue.tolist() == [[1, 0], [0, 0], [0, 0]]
    assert regret.tolist() == [[0, 0], [1, 0], [0, 0]]
    assert np.isclose(rr_loss[1, 0], 0.1)
    assert mttc_loss[1, 0] == 10.0
    assert float(rr_loss.sum()) == float(rr_loss[1, 0])
    assert float(mttc_loss.sum()) == 10.0


def test_allowed_rank_fraction_uses_stable_allowed_slot10_order() -> None:
    scores = np.arange(12, dtype=np.float32).reshape(1, 1, 12)
    scores[0, 0, 9] = 11.0
    scores[0, 0, 10] = 11.0
    scores[0, 0, 11] = 12.0
    choice = np.asarray([[9]], dtype=np.uint8)
    incumbent = np.asarray([[9]], dtype=np.uint8)
    fraction = subject._allowed_rank_fraction(scores, choice, incumbent)
    # Allowed candidates are incumbent 9 plus 10 and 11.  Stable index order
    # breaks the 9/10 tie, while candidate 11 is strictly better.
    assert fraction.shape == (1, 1)
    assert np.isclose(fraction[0, 0], 2.0 / 3.0)


def test_session_weight_normalization_prevents_turn_count_inflation() -> None:
    raw = np.asarray([[1.0, 3.0, 5.0], [7.0, 0.0, 0.0]])
    action = np.asarray([[True, True, False], [True, False, False]])
    normalized = subject._session_normalize_weights(raw, action)
    assert np.allclose(normalized.sum(axis=1), [1.0, 1.0])
    assert np.allclose(normalized[0], [0.25, 0.75, 0.0])
    assert np.allclose(normalized[1], [1.0, 0.0, 0.0])


def test_keep_threshold_serializes_without_nonstandard_infinity() -> None:
    assert subject._serialized_threshold(float("inf")) == "KEEP"
    assert subject._serialized_threshold(0.25) == 0.25
    payload = {
        "inner_threshold": subject._serialized_threshold(float("inf")),
        "mapped_threshold": subject._serialized_threshold(float("inf")),
    }
    assert len(subject._canonical_sha256(payload)) == 64
