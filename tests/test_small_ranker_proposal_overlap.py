from __future__ import annotations

import numpy as np

from scripts import analyze_small_ranker_proposal_overlap as subject


def test_membership_summary_reports_unique_overlap_and_direction(monkeypatch) -> None:
    monkeypatch.setattr(subject.base, "SESSION_COUNT", 6)
    monkeypatch.setattr(subject.base, "OUTER_FOLDS", 3)
    reachable = {
        "pairwise": np.asarray([1, 1, 1, 0, 0, 0], dtype=bool),
        "rrf3": np.asarray([1, 0, 0, 1, 0, 0], dtype=bool),
        "focused_lambdamart": np.asarray([0, 1, 0, 1, 1, 0], dtype=bool),
    }
    outer = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint8)
    result = subject._membership_summary(reachable, outer, current_hits=0)
    assert result["surfaces"]["pairwise"]["sessions"] == 3
    assert result["surfaces"]["pairwise"]["unique_over_other_two"] == 1
    assert result["intersections"]["pairwise_and_rrf3"] == 1
    assert result["intersections"]["pairwise_and_focused"] == 1
    assert result["intersections"]["all_three"] == 0
    assert result["union"]["sessions"] == 5
    assert result["union"]["by_outer_fold"] == [2, 2, 1]
    assert result["direction_gate"]["portfolio_worth_testing"] is False
