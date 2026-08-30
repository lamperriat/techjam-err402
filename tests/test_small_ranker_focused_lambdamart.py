from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import build_small_ranker_focused_cache as cache
from scripts import evaluate_small_ranker_focused_stage_a as stage_a
from scripts import train_small_ranker_focused_outer_oof as trainer


def test_allowed_91_has_incumbent_then_expansion_without_protected_intrusion() -> None:
    allowed = cache._allowed_indices(7)
    assert len(allowed) == 91
    assert allowed[0] == 7
    assert allowed[1:].tolist() == list(range(10, 100))
    assert len(np.unique(allowed)) == 91
    with pytest.raises(cache.FocusedCacheError):
        cache._allowed_indices(10)


def test_focused_cohort_respects_eligibility_and_inclusive_first_hit(monkeypatch) -> None:
    monkeypatch.setattr(cache.base, "SESSION_COUNT", 4)
    monkeypatch.setattr(cache.base, "TURN_COUNT", 4)
    monkeypatch.setattr(cache, "EXPECTED_HARD_SESSIONS", 1)
    labels = {
        "positive_index": np.asarray(
            [
                [-1, 10, 11, -1],
                [10, -1, 12, 13],
                [10, 10, 10, 10],
                [5, 5, 5, 5],
            ],
            dtype=np.int16,
        ),
        "eligible_from": np.asarray([2, 1, 1, 1], dtype=np.uint8),
        "outer_fold": np.asarray([0, 1, 2, 3], dtype=np.uint8),
        "inner_fold": np.asarray([1, 2, 3, 4], dtype=np.uint8),
    }
    state = {
        "hit": np.asarray([False, True, True, False]),
        "first_rank": np.asarray([0, 10, 1, 0], dtype=np.int16),
        "first_turn": np.asarray([11, 3, 1, 11], dtype=np.int16),
    }
    incumbent = np.zeros((4, 4), dtype=np.uint8)
    selected = cache._select_groups(labels, state, incumbent)
    assert selected.hard_session.tolist() == [True, False, False, False]
    assert selected.control_session.tolist() == [False, True, False, False]
    assert list(zip(selected.session.tolist(), selected.turn.tolist())) == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
    ]
    assert selected.hard.tolist() == [1, 1, 0, 0]


def test_candidate_row_mapping_has_one_positive() -> None:
    incumbent = 4
    allowed = cache._allowed_indices(incumbent)
    projected = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    target = 27
    rows = projected[allowed]
    relevance = (allowed == target).astype(np.uint8)
    assert np.array_equal(rows, projected[[incumbent, *range(10, 100)]])
    assert relevance.shape == (91,)
    assert int(relevance.sum()) == 1
    assert int(np.argmax(relevance)) == 18


def test_exclusive_npy_memmap_is_loadable_and_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "cache.npy"
    mapped = cache._exclusive_npy_memmap(
        path, dtype=np.float32, shape=(2, 3)
    )
    mapped[:] = np.arange(6, dtype=np.float32).reshape(2, 3)
    mapped.flush()
    del mapped
    loaded = np.load(path, allow_pickle=False)
    assert loaded.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    with pytest.raises(FileExistsError):
        cache._exclusive_npy_memmap(path, dtype=np.float32, shape=(2, 3))


def test_partition_weights_are_session_normalized_then_balanced() -> None:
    session = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int16)
    hard = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    selected = np.ones(6, dtype=bool)
    weight = trainer.partition_group_weights(session, hard, selected)
    assert np.isclose(weight.mean(), 1.0)
    assert np.isclose(weight[hard.astype(bool)].sum(), 3.0)
    assert np.isclose(weight[~hard.astype(bool)].sum(), 3.0)
    assert np.isclose(weight[session == 0].sum(), weight[session == 1].sum())
    with pytest.raises(trainer.FocusedTrainingError):
        trainer.partition_group_weights(session, hard, hard.astype(bool))


def test_local_training_artifact_rejects_absolute_path() -> None:
    with pytest.raises(trainer.FocusedTrainingError):
        trainer._local_regular_file(r"C:\outside\scores.npy")


@pytest.mark.parametrize(
    ("sessions", "folds", "expected"),
    [(13, 3, False), (14, 2, False), (14, 3, True)],
)
def test_stage_a_gate_boundaries(sessions: int, folds: int, expected: bool) -> None:
    assert stage_a.stage_a_gate(sessions, folds) is expected


def test_stage_a_oracle_counts_sessions_once_and_only_after_eligibility(monkeypatch) -> None:
    monkeypatch.setattr(stage_a.base, "SESSION_COUNT", 4)
    monkeypatch.setattr(stage_a.base, "TURN_COUNT", 3)
    monkeypatch.setattr(stage_a.base, "OUTER_FOLDS", 3)
    action = np.asarray(
        [[True, True, True], [True, True, False], [True, False, False], [True, True, True]]
    )
    chosen = np.asarray(
        [[5, 7, 7], [8, 8, 1], [9, 1, 1], [6, 6, 6]], dtype=np.uint8
    )
    surface = SimpleNamespace(action=action, pairwise_chosen=chosen)
    labels = {
        "positive_index": np.asarray(
            [[5, 7, 7], [8, 8, -1], [9, -1, -1], [6, 6, 6]], dtype=np.int16
        ),
        "eligible_from": np.asarray([2, 1, 2, 1], dtype=np.uint8),
        "outer_fold": np.asarray([0, 1, 2, 0], dtype=np.uint8),
    }
    state = {"hit": np.asarray([False, False, False, True])}
    oracle = stage_a._proposal_oracle(surface, labels, state)
    # Session 0 has two eligible correct turns but is one rescue; session 2's
    # only correct action is before eligibility and must not count.
    assert oracle["reachable_current_miss_sessions"] == 2
    assert oracle["correct_action_turns"] == 4
    assert oracle["reachable_by_outer_fold"] == [1, 1, 0]
    assert oracle["maximum_zero_harm_hits"] == 3
