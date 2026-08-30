from __future__ import annotations

import math

import numpy as np
import pytest

from scripts import export_small_ranker_fold_safe_artifact as exporter


def test_threshold_quantile_uses_higher_and_keep_is_infinite() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    assert exporter._threshold_at_quantile(values, 0.5) == pytest.approx(0.3)
    assert math.isinf(exporter._threshold_at_quantile(values, 1.0))
    with pytest.raises(exporter.ArtifactFreezeError):
        exporter._threshold_at_quantile(np.asarray([]), 0.5)


def test_quantile_choice_obeys_metric_then_safety_tiebreaks() -> None:
    weaker = {
        "technical_score_delta": 0.1,
        "net_hits": 2,
        "inner_fold_net_hits": [1, 0, 1],
        "activation_turns": 10,
        "quantile": 0.5,
    }
    stronger = {**weaker, "inner_fold_net_hits": [1, 1, 1]}
    assert exporter._quantile_choice_key(stronger) > exporter._quantile_choice_key(
        weaker
    )
    fewer_actions = {**stronger, "activation_turns": 9}
    assert exporter._quantile_choice_key(
        fewer_actions
    ) > exporter._quantile_choice_key(stronger)


def test_serialized_head_probability_matches_logistic_formula() -> None:
    head = {
        "mean": [1.0, 2.0],
        "scale": [2.0, 4.0],
        "coef": [0.5, -1.0],
        "intercept": 0.25,
    }
    row = np.asarray([3.0, 6.0], dtype=np.float64)
    expected = 1.0 / (1.0 + math.exp(0.25))
    assert exporter._head_probability(head, row) == pytest.approx(expected)
    actual = exporter._serialized_head_probabilities(head, row[None, :])
    assert actual.tolist() == pytest.approx([expected])


def test_artifact_scan_rejects_only_exact_forbidden_keys() -> None:
    safe = {
        "privacy": {"target_features": False},
        "sources": [{"target_blind_feature_cache_sha256": "abc"}],
    }
    unsafe = {"admission": {"target": "forbidden"}, "asin": "forbidden"}
    assert exporter._artifact_key_scan(safe) == []
    assert sorted(exporter._artifact_key_scan(unsafe)) == ["asin", "target"]


def test_oof_gate_requires_every_fold_to_be_safe() -> None:
    global_metrics = {
        "policy": {"hit_rate_at_10": 0.9715},
        "hit_to_miss": 0,
        "mrr_delta": 0.001,
        "mttc_delta": -0.1,
        "technical_score_delta": 0.01,
    }
    safe_fold = {
        "net_hits": 1,
        "hit_to_miss": 0,
        "mrr_delta": 0.001,
        "mttc_delta": -0.1,
    }
    result = {"global": global_metrics, "folds": [safe_fold] * 5}
    assert exporter._oof_gate_passed(result)
    result["folds"] = [{**safe_fold, "mrr_delta": -0.001}, *([safe_fold] * 4)]
    assert not exporter._oof_gate_passed(result)


def test_output_must_be_below_current_worktree(tmp_path) -> None:
    with pytest.raises(exporter.ArtifactFreezeError):
        exporter._require_output_below_root(tmp_path.resolve())
    exporter._require_output_below_root(
        (exporter.ROOT / "experiments" / "fixture").resolve()
    )
