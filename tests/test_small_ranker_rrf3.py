from __future__ import annotations

import numpy as np

from scripts import evaluate_small_ranker_rrf3 as subject


def test_stable_ranks_break_equal_scores_by_candidate_order(monkeypatch) -> None:
    monkeypatch.setattr(subject.base, "CANDIDATE_COUNT", 4)
    scores = np.asarray([[[0.5, 0.8, 0.8, 0.1]]], dtype=np.float32)
    assert subject.stable_ranks(scores).tolist() == [[[3, 1, 2, 4]]]


def test_rrf3_uses_equal_reciprocal_ranks(monkeypatch) -> None:
    monkeypatch.setattr(subject.base, "CANDIDATE_COUNT", 3)
    first = np.asarray([[[3.0, 2.0, 1.0]]], dtype=np.float32)
    second = np.asarray([[[1.0, 3.0, 2.0]]], dtype=np.float32)
    third = np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    scores = subject.rrf_scores([first, second, third])
    expected = np.asarray(
        [[[
            1 / 61 + 1 / 63 + 1 / 63,
            1 / 62 + 1 / 61 + 1 / 62,
            1 / 63 + 1 / 62 + 1 / 61,
        ]]],
        dtype=np.float32,
    )
    assert np.allclose(scores, expected)
    assert int(np.argmax(scores, axis=2)[0, 0]) == 1
