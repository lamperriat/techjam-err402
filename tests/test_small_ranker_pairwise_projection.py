from __future__ import annotations

import numpy as np

from scripts import evaluate_small_ranker_pairwise_projection as evaluation


def _state(hit: list[bool], rank: list[int], turn: list[int]) -> dict[str, np.ndarray]:
    return {
        "hit": np.asarray(hit, dtype=bool),
        "first_rank": np.asarray(rank, dtype=np.int16),
        "first_turn": np.asarray(turn, dtype=np.int16),
    }


def test_pairwise_promotion_gate_requires_every_fold_nonregression(
    monkeypatch,
) -> None:
    labels = {"outer_fold": np.asarray([0, 1, 2, 3, 4], dtype=np.uint8)}
    current = _state([True] * 5, [10] * 5, [2] * 5)
    challenger = _state([True] * 5, [9] * 5, [1] * 5)
    activation = np.ones((5, 1), dtype=bool)
    monkeypatch.setattr(evaluation, "CURRENT_HR", 0.9)
    passed, global_delta, folds = evaluation._promotion_gate(
        current, challenger, activation, labels
    )
    assert passed
    assert global_delta["mrr_delta"] > 0
    assert all(row["net_hits"] == 0 for row in folds)

    regressed = _state([True] * 5, [9, 9, 10, 9, 9], [1, 1, 3, 1, 1])
    passed, _global_delta, folds = evaluation._promotion_gate(
        current, regressed, activation, labels
    )
    assert not passed
    assert folds[2]["mttc_delta"] > 0
