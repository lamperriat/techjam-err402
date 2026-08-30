from __future__ import annotations

import numpy as np

from scripts import evaluate_small_ranker_hardcase_residual as residual


def test_allowed_indices_protect_top9_and_keep_incumbent() -> None:
    allowed = residual._allowed_indices(7)
    assert allowed[0] == 7
    assert set(allowed[1:]) == set(range(10, 100))
    assert not set(range(9)) - {7} & set(allowed)


def test_pairwise_linear_weights_rank_positive_difference() -> None:
    positive = np.asarray([[2.0, 1.0], [1.5, 0.5]], dtype=np.float32)
    x = np.vstack((positive, -positive))
    cache = residual.PairCache(
        x=x,
        y=np.asarray([1, 1, 0, 0], dtype=np.uint8),
        session=np.asarray([0, 0, 1, 1], dtype=np.int32),
        weight=np.ones(4, dtype=np.float64),
        hard_session=np.asarray([True, False]),
    )
    weights, audit = residual._fit_rank_weights(
        cache, np.asarray([True, True]), seed=1
    )
    assert np.all(positive @ weights > 0)
    assert audit["pair_rows"] == 4
    assert audit["hard_sessions"] == 1
