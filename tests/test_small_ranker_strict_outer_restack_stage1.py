from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts import build_small_ranker_strict_outer_restack as build
from scripts import export_small_ranker_fold_safe_artifact as frozen
from scripts import freeze_small_ranker_strict_outer_restack as freezer
from scripts import small_ranker_portfolio_selector_py39 as selector
from scripts import train_small_ranker as base


ROOT = Path(__file__).resolve().parents[1]


class SparseFeatureCarrier:
    shape = (
        base.SESSION_COUNT,
        base.TURN_COUNT,
        base.CANDIDATE_COUNT,
        base.FEATURE_COUNT,
    )

    def __getitem__(self, key):
        session, turn, candidate, feature = key
        session, turn, candidate = np.broadcast_arrays(session, turn, candidate)
        return (
            np.asarray(session, dtype=np.float32) * 0.01
            + np.asarray(turn, dtype=np.float32) * 0.001
            + np.asarray(candidate, dtype=np.float32) * 0.0001
            + np.float32(feature) * 0.00001
        )


class LocalFeatureCarrier:
    shape = SparseFeatureCarrier.shape

    def __init__(self, local: np.ndarray, sessions: np.ndarray):
        self.local = local
        self.mapping = np.full(base.SESSION_COUNT, -1, dtype=np.int32)
        self.mapping[np.asarray(sessions, dtype=np.int64)] = np.arange(len(sessions))

    def __getitem__(self, key):
        session, turn, candidate, feature = key
        local_session = self.mapping[np.asarray(session, dtype=np.int64)]
        if np.any(local_session < 0):
            raise AssertionError("unexpected global session")
        return self.local[local_session, turn, candidate, feature]


def _score_surface(choices: np.ndarray) -> np.ndarray:
    scores = np.zeros((*choices.shape, base.CANDIDATE_COUNT), dtype=np.float32)
    rows = np.arange(len(choices))[:, None]
    turns = np.arange(base.TURN_COUNT)[None, :]
    scores[rows, turns, choices] = 10.0
    return scores


def _empty_runtime(session_count: int) -> selector.RuntimePortfolioSurface:
    current = np.zeros((session_count, base.TURN_COUNT), dtype=np.uint8)
    candidates = np.full(
        (session_count, base.TURN_COUNT, selector.MAX_ACTIONS),
        -1,
        dtype=np.int16,
    )
    available = np.zeros(candidates.shape, dtype=bool)
    return selector.RuntimePortfolioSurface(
        current_chosen=current,
        current_activation=np.zeros_like(current, dtype=bool),
        current_choice=current.copy(),
        incumbent=current.copy(),
        family_choices=np.zeros(
            (session_count, base.TURN_COUNT, len(selector.FAMILY_NAMES)),
            dtype=np.uint8,
        ),
        candidates=candidates,
        source_mask=np.zeros(candidates.shape, dtype=np.uint8),
        available=available,
        features=np.zeros(
            (*candidates.shape, len(selector.FEATURE_NAMES)), dtype=np.float32
        ),
    )


def test_stage1_protocol_binds_full_retrain_and_keep_semantics():
    amendment = json.loads(build.STAGE1_AMENDMENT.read_text(encoding="utf-8"))
    assert amendment["hypothesis_or_promotion_gate_changed"] is False
    assert amendment["full_retrain_rule"]["fits_per_pass"] == 150
    assert amendment["full_retrain_rule"]["required_total_xgboost_fits"] == 300
    assert "do not count" in amendment["full_retrain_rule"]["stage0_reuse"]
    assert "positive infinity" in amendment["semantic_clarifications"]["current_gate_KEEP"]
    assert "preserves" in amendment["semantic_clarifications"]["selector_KEEP"]
    prereg, validated, manifest = build._validate_protocol()
    assert prereg["experiment_id"] == amendment["experiment_id"]
    assert validated == amendment
    assert manifest["decision"]["stage_1_authorized"] is True


def test_python39_subset_constants_match_frozen_source():
    source = ast.parse(
        (ROOT / "scripts/evaluate_small_ranker_portfolio_selector.py").read_text(
            encoding="utf-8"
        )
    )
    assignments = {}
    for node in source.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "MODEL_SEED",
                "MAX_ACTIONS",
                "MIN_RESCUE_SESSIONS",
                "MIN_RESCUE_FAMILIES",
                "REGRET_MULTIPLIER",
                "FAMILY_NAMES",
                "FEATURE_NAMES",
            }:
                assignments[target.id] = ast.literal_eval(node.value)
    for name, expected in assignments.items():
        assert getattr(selector, name) == expected
    assert selector.BIT_COUNT.tolist() == [
        bin(value).count("1") for value in range(256)
    ]


def test_runtime_surface_is_shape_generic_and_uses_global_feature_sessions():
    feature_sessions = np.asarray([3, 17], dtype=np.int16)
    incumbent = np.zeros((2, base.TURN_COUNT), dtype=np.uint8)
    current_choice = np.full_like(incumbent, 10)
    pairwise_choice = np.full_like(incumbent, 11)
    rrf_choice = np.full_like(incumbent, 12)
    focused_choice = np.full_like(incumbent, 11)
    current_scores = _score_surface(current_choice)
    family_scores = [
        _score_surface(pairwise_choice),
        _score_surface(rrf_choice),
        _score_surface(focused_choice),
    ]
    runtime = build._build_runtime_surface_any(
        SparseFeatureCarrier(),
        feature_sessions,
        current_scores,
        family_scores,
        current_choice,
        np.ones_like(current_choice, dtype=bool),
        incumbent,
    )
    assert runtime.features.shape == (
        2,
        base.TURN_COUNT,
        selector.MAX_ACTIONS,
        len(selector.FEATURE_NAMES),
    )
    assert np.all(runtime.available.sum(axis=2) == 2)
    assert np.all(runtime.candidates[..., 0] == 11)
    assert np.all(runtime.candidates[..., 1] == 12)
    assert not np.any(runtime.candidates[runtime.available] == 10)
    coverage_feature = runtime.features[..., 14]
    assert np.all(coverage_feature[1][runtime.available[1]] > coverage_feature[0][runtime.available[0]])
    assert np.isfinite(runtime.features).all()


@pytest.mark.skipif(sys.version_info < (3, 10), reason="frozen report chain is 3.10+")
def test_runtime_surface_all_fields_match_frozen_v27(monkeypatch):
    from scripts import evaluate_small_ranker_portfolio_selector as old
    from scripts import probe_small_ranker_strict_outer_restack as probe

    rng = np.random.default_rng(402)
    sessions = np.asarray([17, 3], dtype=np.int16)
    local_features = rng.normal(
        size=(2, base.TURN_COUNT, base.CANDIDATE_COUNT, base.FEATURE_COUNT)
    ).astype(np.float32)
    incumbent = np.zeros((2, base.TURN_COUNT), dtype=np.uint8)
    current_choice = np.full_like(incumbent, 10)
    current_scores = rng.normal(
        size=(2, base.TURN_COUNT, base.CANDIDATE_COUNT)
    ).astype(np.float32)
    family_scores = [
        rng.normal(size=current_scores.shape).astype(np.float32)
        for _ in range(len(selector.FAMILY_NAMES))
    ]
    current_activation = rng.random(current_choice.shape) > 0.5
    monkeypatch.setattr(old.base, "choose_slot10", probe.choose_slot10_any)
    expected = old._build_runtime_surface(
        local_features,
        current_scores,
        family_scores,
        current_choice,
        current_activation,
        incumbent,
    )
    actual = build._build_runtime_surface_any(
        LocalFeatureCarrier(local_features, sessions),
        sessions,
        current_scores,
        family_scores,
        current_choice,
        current_activation,
        incumbent,
    )
    for name in build.RUNTIME_FIELDS:
        assert np.array_equal(getattr(actual, name), getattr(expected, name)), name


def test_inference_interfaces_do_not_accept_labels():
    forbidden = ("label", "target", "outcome", "positive")
    for function in (
        build._inference_surface,
        build._build_runtime_surface_any,
        build._slice_runtime,
        build._fit_full_current_gate,
    ):
        parameters = inspect.signature(function).parameters
        assert not any(
            token in name.lower() for name in parameters for token in forbidden
        )
    assert "labels_t" in build.OuterInputs.__dataclass_fields__
    assert not any(
        name.startswith("labels_h") or "held_outcome" in name
        for name in build.OuterInputs.__dataclass_fields__
    )


def test_domain_masks_cover_train_once_and_fail_on_crossing_family():
    outer = np.repeat(np.arange(base.OUTER_FOLDS, dtype=np.uint8), 400)
    inner = (np.arange(base.SESSION_COUNT) % base.OUTER_FOLDS).astype(np.uint8)
    family = np.arange(base.SESSION_COUNT, dtype=np.int32)
    t_sessions = np.flatnonzero(outer != 2).astype(np.int16)
    h_sessions = np.flatnonzero(outer == 2).astype(np.int16)
    inputs = build.OuterInputs(
        raw_features=np.empty(0),
        projected_features=np.empty(0),
        outer_fold=outer,
        inner_fold=inner,
        family_index=family,
        t_sessions=t_sessions,
        h_sessions=h_sessions,
        labels_t={},
        paths={},
    )
    records = build._domain_records(inputs, 2)
    assert records["H_2"]["sessions"] == 400
    assert records["T_2"]["sessions"] == 1600
    assert sum(records[f"V_2{fold}"]["sessions"] for fold in range(5)) == 1600
    crossed = family.copy()
    crossed[400] = crossed[0]
    crossed_inputs = build.OuterInputs(
        **{**inputs.__dict__, "family_index": crossed}
    )
    with pytest.raises(build.StrictRestackBuildError, match="family crosses"):
        build._domain_records(crossed_inputs, 2)
    invalid_inner = inner.copy()
    invalid_inner[t_sessions[0]] = 5
    invalid_inputs = build.OuterInputs(
        **{**inputs.__dict__, "inner_fold": invalid_inner}
    )
    with pytest.raises(build.StrictRestackBuildError, match="coverage"):
        build._domain_records(invalid_inputs, 2)


def test_current_reference_quantile_excludes_held_utility(monkeypatch):
    feature_count = len(base.GATE_FEATURE_NAMES)
    train_shape = (5, base.TURN_COUNT)
    train_action = np.zeros(train_shape, dtype=bool)
    train_action[:, 0] = True
    train_features = np.zeros((*train_shape, feature_count), dtype=np.float32)
    training = frozen.ActionSurface(
        incumbent=np.zeros(train_shape, dtype=np.uint8),
        chosen=np.ones(train_shape, dtype=np.uint8),
        action=train_action,
        gate_features=train_features,
        rescue=np.ones(train_shape, dtype=np.uint8),
        rescue_weights=np.ones(train_shape, dtype=np.float64),
        regret=np.zeros(train_shape, dtype=np.uint8),
        regret_weights=np.ones(train_shape, dtype=np.float64),
    )
    infer_shape = (base.SESSION_COUNT, base.TURN_COUNT)
    infer_action = np.zeros(infer_shape, dtype=bool)
    infer_action[:, 0] = True
    infer_features = np.zeros((*infer_shape, feature_count), dtype=np.float32)
    infer_features[:, 0, 0] = np.arange(base.SESSION_COUNT, dtype=np.float32)
    infer_features[1600:, 0, 0] = 1_000_000.0
    inference = build.InferenceSurface(
        incumbent=np.zeros(infer_shape, dtype=np.uint8),
        chosen=np.ones(infer_shape, dtype=np.uint8),
        action=infer_action,
        gate_features=infer_features,
    )

    class Dummy:
        def __init__(self, probability):
            self.probability = probability

    def fake_fit(_x, y, _weights, _seed):
        return Dummy(float(np.mean(y))), np.zeros(feature_count), np.ones(feature_count)

    def fake_predict(model, _mean, _scale, x):
        return x[:, 0] if model.probability > 0.5 else np.zeros(len(x))

    monkeypatch.setattr(base, "_fit_gate_model", fake_fit)
    monkeypatch.setattr(base, "_predict_gate", fake_predict)
    t_mask = np.arange(base.SESSION_COUNT) < 1600
    activation, record = build._fit_full_current_gate(
        training, inference, 0, 0.5, t_mask
    )
    expected = float(
        np.quantile(np.arange(1600, dtype=np.float32), 0.5, method="higher")
    )
    assert record["mapped_reference_threshold"] == expected
    assert np.all(activation[1600:, 0])
    assert not np.any(activation[:, 1:])


def test_current_keep_maps_to_no_replacement(monkeypatch):
    feature_count = len(base.GATE_FEATURE_NAMES)
    training_shape = (2, base.TURN_COUNT)
    action = np.zeros(training_shape, dtype=bool)
    action[:, 0] = True
    training = frozen.ActionSurface(
        incumbent=np.zeros(training_shape, dtype=np.uint8),
        chosen=np.ones(training_shape, dtype=np.uint8),
        action=action,
        gate_features=np.ones((*training_shape, feature_count), dtype=np.float32),
        rescue=np.ones(training_shape, dtype=np.uint8),
        rescue_weights=np.ones(training_shape),
        regret=np.zeros(training_shape, dtype=np.uint8),
        regret_weights=np.ones(training_shape),
    )
    shape = (base.SESSION_COUNT, base.TURN_COUNT)
    inference = build.InferenceSurface(
        incumbent=np.zeros(shape, dtype=np.uint8),
        chosen=np.ones(shape, dtype=np.uint8),
        action=np.ones(shape, dtype=bool),
        gate_features=np.ones((*shape, feature_count), dtype=np.float32),
    )

    class Dummy:
        probability = 0.5

    monkeypatch.setattr(
        base,
        "_fit_gate_model",
        lambda *_args: (Dummy(), np.zeros(feature_count), np.ones(feature_count)),
    )
    monkeypatch.setattr(
        base,
        "_predict_gate",
        lambda _model, _mean, _scale, x: np.full(len(x), 0.5),
    )
    activation, record = build._fit_full_current_gate(
        training,
        inference,
        0,
        frozen.KEEP_QUANTILE,
        np.arange(base.SESSION_COUNT) < 1600,
    )
    assert record["mapped_reference_threshold"] == "inf"
    assert not np.any(activation)


def test_selector_keep_preserves_current_without_held_labels(monkeypatch):
    training_runtime = _empty_runtime(2)
    training = selector.PortfolioSurface(
        **training_runtime.__dict__,
        rescue=np.zeros(training_runtime.available.shape, dtype=np.uint8),
        rescue_weights=np.zeros(training_runtime.available.shape),
        regret=np.zeros(training_runtime.available.shape, dtype=np.uint8),
        regret_weights=np.zeros(training_runtime.available.shape),
        rr_loss=np.zeros(training_runtime.available.shape, dtype=np.float32),
        mttc_loss=np.zeros(training_runtime.available.shape, dtype=np.float32),
    )
    labels = {
        "baseline_rank": np.zeros((2, base.TURN_COUNT), dtype=np.int16),
        "positive_index": np.full((2, base.TURN_COUNT), -1, dtype=np.int16),
        "eligible_from": np.ones(2, dtype=np.uint8),
        "inner_fold": np.asarray([0, 1], dtype=np.uint8),
        "family_index": np.asarray([10, 11], dtype=np.int32),
    }
    current_state = {
        "hit": np.zeros(2, dtype=bool),
        "first_rank": np.zeros(2, dtype=np.int16),
        "first_turn": np.full(2, 11, dtype=np.int16),
    }
    monkeypatch.setattr(
        build,
        "_write_array",
        lambda _path, value: {"array_sha256": build._array_sha256(np.asarray(value))},
    )
    empty_held = _empty_runtime(1)
    nontrivial_chosen = np.full(
        (1, base.TURN_COUNT), 23, dtype=np.uint8
    )
    nontrivial_activation = np.zeros_like(nontrivial_chosen, dtype=bool)
    nontrivial_activation[:, 4] = True
    held = selector.RuntimePortfolioSurface(
        **{
            **empty_held.__dict__,
            "current_chosen": nontrivial_chosen,
            "current_activation": nontrivial_activation,
            "current_choice": np.where(
                nontrivial_activation,
                nontrivial_chosen,
                empty_held.incumbent,
            ).astype(np.uint8),
        }
    )
    final_chosen, final_activation, record = build._strict_selector(
        training,
        labels,
        current_state,
        _empty_runtime(2),
        held,
        ROOT / "experiments" / "unit-test-not-written",
    )
    assert record["selected_quantile"] == frozen.KEEP_QUANTILE
    assert record["inner_selection"]["status"] == "KEEP_INSUFFICIENT_INNER_FIT"
    assert np.array_equal(final_chosen, held.current_chosen)
    assert np.array_equal(final_activation, held.current_activation)


def test_empty_held_actions_do_not_control_outer_selector_fit(monkeypatch):
    training_runtime = _empty_runtime(5)
    available = training_runtime.available.copy()
    available[:, :, 0] = True
    candidates = training_runtime.candidates.copy()
    candidates[:, :, 0] = 10
    source_mask = training_runtime.source_mask.copy()
    source_mask[:, :, 0] = 1
    features = training_runtime.features.copy()
    training_runtime = selector.RuntimePortfolioSurface(
        **{
            **training_runtime.__dict__,
            "available": available,
            "candidates": candidates,
            "source_mask": source_mask,
            "features": features,
        }
    )
    rescue = np.zeros(available.shape, dtype=np.uint8)
    regret = np.zeros(available.shape, dtype=np.uint8)
    rescue[0, 0, 0] = 1
    regret[1, 0, 0] = 1
    training = selector.PortfolioSurface(
        **training_runtime.__dict__,
        rescue=rescue,
        rescue_weights=np.ones(available.shape),
        regret=regret,
        regret_weights=np.ones(available.shape),
        rr_loss=np.zeros(available.shape, dtype=np.float32),
        mttc_loss=np.zeros(available.shape, dtype=np.float32),
    )
    labels = {
        "baseline_rank": np.zeros((5, base.TURN_COUNT), dtype=np.int16),
        "positive_index": np.full((5, base.TURN_COUNT), -1, dtype=np.int16),
        "eligible_from": np.ones(5, dtype=np.uint8),
        "inner_fold": np.arange(5, dtype=np.uint8),
        "family_index": np.arange(5, dtype=np.int32),
    }
    current_state = {
        "hit": np.zeros(5, dtype=bool),
        "first_rank": np.zeros(5, dtype=np.int16),
        "first_turn": np.full(5, 11, dtype=np.int16),
    }

    class Dummy:
        probability = 0.5

    monkeypatch.setattr(selector, "_fit_readiness", lambda *_args: {"ready": True})
    monkeypatch.setattr(
        selector,
        "_select_inner_quantile",
        lambda *_args: {"quantile": 0.5, "inner_threshold": 0.0},
    )
    monkeypatch.setattr(
        selector,
        "_fit_gate_model",
        lambda x, *_args: (
            Dummy(),
            np.zeros(x.shape[1]),
            np.ones(x.shape[1]),
        ),
    )
    monkeypatch.setattr(selector, "_validate_fitted_model", lambda _model: None)
    monkeypatch.setattr(
        base,
        "_predict_gate",
        lambda _model, _mean, _scale, x: np.full(len(x), 0.5),
    )
    monkeypatch.setattr(
        build,
        "_write_array",
        lambda _path, value: {"array_sha256": build._array_sha256(np.asarray(value))},
    )
    final_chosen, final_activation, record = build._strict_selector(
        training,
        labels,
        current_state,
        training_runtime,
        _empty_runtime(1),
        ROOT / "experiments" / "unit-test-not-written",
    )
    assert len(record["outer_head_models"]) == 2
    assert record["inner_selection"]["status"] == "FINITE_SELECTED"
    assert not np.any(final_activation)
    assert np.array_equal(final_chosen, np.zeros_like(final_chosen))


def test_selector_maps_quantile_on_reference_then_applies_to_held(monkeypatch):
    def action_runtime(count, feature_values):
        runtime = _empty_runtime(count)
        available = runtime.available.copy()
        available[:, 0, 0] = True
        candidates = runtime.candidates.copy()
        candidates[:, 0, 0] = 10
        source_mask = runtime.source_mask.copy()
        source_mask[:, 0, 0] = 1
        features = runtime.features.copy()
        features[:, 0, 0, 0] = np.asarray(feature_values, dtype=np.float32)
        return selector.RuntimePortfolioSurface(
            **{
                **runtime.__dict__,
                "available": available,
                "candidates": candidates,
                "source_mask": source_mask,
                "features": features,
            }
        )

    training_runtime = action_runtime(5, np.zeros(5))
    rescue = np.zeros(training_runtime.available.shape, dtype=np.uint8)
    regret = np.zeros(training_runtime.available.shape, dtype=np.uint8)
    rescue[0, 0, 0] = 1
    regret[1, 0, 0] = 1
    training = selector.PortfolioSurface(
        **training_runtime.__dict__,
        rescue=rescue,
        rescue_weights=np.ones(training_runtime.available.shape),
        regret=regret,
        regret_weights=np.ones(training_runtime.available.shape),
        rr_loss=np.zeros(training_runtime.available.shape, dtype=np.float32),
        mttc_loss=np.zeros(training_runtime.available.shape, dtype=np.float32),
    )
    labels = {
        "baseline_rank": np.zeros((5, base.TURN_COUNT), dtype=np.int16),
        "positive_index": np.full((5, base.TURN_COUNT), -1, dtype=np.int16),
        "eligible_from": np.ones(5, dtype=np.uint8),
        "inner_fold": np.arange(5, dtype=np.uint8),
        "family_index": np.arange(5, dtype=np.int32),
    }
    current_state = {
        "hit": np.zeros(5, dtype=bool),
        "first_rank": np.zeros(5, dtype=np.int16),
        "first_turn": np.full(5, 11, dtype=np.int16),
    }

    class Dummy:
        def __init__(self, head):
            self.head = head
            self.probability = float(head)

    calls = {"fit": 0}

    def fake_fit(x, *_args):
        head = calls["fit"] % 2
        calls["fit"] += 1
        return Dummy(head), np.zeros(x.shape[1]), np.ones(x.shape[1])

    def fake_predict(model, _mean, _scale, x):
        if model.head == 0:
            return 0.5 + 0.1 * x[:, 0]
        return np.full(len(x), 0.2)

    monkeypatch.setattr(selector, "_fit_readiness", lambda *_args: {"ready": True})
    monkeypatch.setattr(
        selector,
        "_select_inner_quantile",
        lambda *_args: {"quantile": 0.5, "inner_threshold": 0.0},
    )
    monkeypatch.setattr(selector, "_fit_gate_model", fake_fit)
    monkeypatch.setattr(selector, "_validate_fitted_model", lambda _model: None)
    monkeypatch.setattr(base, "_predict_gate", fake_predict)
    captured = {}
    original_causal = selector._causal_policy

    def capture_map(winner, available, sessions, quantile):
        captured["reference_winner"] = winner.copy()
        captured["reference_available"] = available.copy()
        captured["quantile"] = quantile
        assert np.all(sessions)
        return 0.45

    def capture_causal(candidates, support, available, utility, threshold, sessions):
        captured["held_utility"] = utility.copy()
        captured["held_threshold"] = threshold
        return original_causal(
            candidates, support, available, utility, threshold, sessions
        )

    monkeypatch.setattr(selector, "_map_outer_quantile", capture_map)
    monkeypatch.setattr(selector, "_causal_policy", capture_causal)
    monkeypatch.setattr(
        build,
        "_write_array",
        lambda _path, value: {"array_sha256": build._array_sha256(np.asarray(value))},
    )
    reference = action_runtime(2, [1.0, 2.0])
    held = action_runtime(1, [9.0])
    final_chosen, final_activation, record = build._strict_selector(
        training,
        labels,
        current_state,
        reference,
        held,
        ROOT / "experiments" / "unit-test-not-written",
    )
    assert np.allclose(
        captured["reference_winner"][:, 0], np.asarray([0.4, 0.5])
    )
    assert np.isclose(captured["held_utility"][0, 0, 0], 1.2)
    assert captured["held_threshold"] == 0.45
    assert record["mapped_reference_threshold"] == 0.45
    assert final_activation[0, 0]
    assert final_chosen[0, 0] == 10


@pytest.mark.parametrize("pass_name", ["first", "repeat"])
def test_stage0_virtual_prefix_schema_matches_recorded_oracle(pass_name):
    manifest = json.loads(build.STAGE0_MANIFEST.read_text(encoding="utf-8"))
    result_path = ROOT / manifest["result"]["path"]
    if not result_path.is_file():
        pytest.skip("gitignored Stage-0 oracle is unavailable")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    pass_record = next(row for row in result["passes"] if row["name"] == pass_name)
    generic = pass_record["models"][:-1]
    focused = [pass_record["models"][-1]]
    focused_cache = pass_record["focused_cache"]
    subset = {
        "A_00": {
            "query_groups": focused_cache["query_groups"],
            "hard_sessions": focused_cache["hard_sessions"],
            "control_sessions": focused_cache["control_sessions"],
            "hard_groups": focused_cache["hard_groups"],
            "control_groups": focused_cache["control_groups"],
            "group_weight_sha256": focused_cache["group_weight_sha256"],
            "arrays": {
                key: value["array_sha256"]
                for key, value in focused_cache["files"].items()
            },
        }
    }
    parity = build._stage0_prefix_parity(
        pass_name,
        generic,
        focused,
        pass_record["current"]["files"],
        pass_record["current"]["gate"],
        subset,
        pass_record["rrf3"]["score"],
        pass_record["rrf3"]["choice"],
    )
    assert parity["equal"] is True
    assert parity["models_compared"] == 9
    assert parity["expected_identity_sha256"] == manifest["exact_repeat"][
        "identity_sha256"
    ]
    assert parity["actual_identity_sha256"] == manifest["exact_repeat"][
        "identity_sha256"
    ]


def test_causal_latch_uses_earliest_passing_turn():
    candidates = np.asarray([[10, 11, 12]], dtype=np.int16)
    utility = np.asarray([[0.1, 0.9, 1.0]], dtype=np.float32)
    available = np.ones_like(candidates, dtype=bool)
    supplement, choice = selector._causal_latch(
        candidates, utility, available, 0.5
    )
    assert supplement.tolist() == [[False, True, False]]
    assert choice.tolist() == [[-1, 11, -1]]


def test_freezer_accepts_only_pass_neutral_outer_identity():
    identity = {
        "models": [{"model_sha256": "a"}, {"model_sha256": "b"}],
        "stage0_prefix_parity": {
            "applicable": True,
            "equal": True,
            "models_compared": 9,
        },
    }
    identity_sha256 = freezer._canonical_sha256(identity)
    first = {
        "outer_fold": 0,
        "identity": identity,
        "identity_sha256": identity_sha256,
    }
    repeat = json.loads(json.dumps(first))
    comparison = freezer._validate_outer_pair(first, repeat)
    assert comparison == {
        "outer_fold": 0,
        "equal": True,
        "identity_sha256": freezer._canonical_sha256(identity),
        "models_compared": 2,
        "stage0_prefix_parity": identity["stage0_prefix_parity"],
    }

    repeat["identity"]["models"][1]["model_sha256"] = "changed"
    with pytest.raises(freezer.StrictRestackFreezeError, match="exact repeat differs"):
        freezer._validate_outer_pair(first, repeat)

    numeric_first = {
        "outer_fold": 1,
        "identity": {"models": [], "stage0_prefix_parity": {"x": 0}},
    }
    numeric_repeat = {
        "outer_fold": 1,
        "identity": {"models": [], "stage0_prefix_parity": {"x": 0.0}},
    }
    numeric_first["identity_sha256"] = freezer._canonical_sha256(
        numeric_first["identity"]
    )
    numeric_repeat["identity_sha256"] = freezer._canonical_sha256(
        numeric_repeat["identity"]
    )
    assert numeric_first["identity"] == numeric_repeat["identity"]
    with pytest.raises(freezer.StrictRestackFreezeError, match="exact repeat differs"):
        freezer._validate_outer_pair(numeric_first, numeric_repeat)


def test_freezer_source_snapshot_binds_target_free_stage1_sources():
    snapshot = freezer._source_snapshot()
    assert snapshot == {
        "freezer_sha256": freezer._sha256(Path(inspect.getfile(freezer))),
        "builder_sha256": freezer._sha256(Path(inspect.getfile(build))),
        "selector_subset_sha256": freezer._sha256(
            ROOT / "scripts/small_ranker_portfolio_selector_py39.py"
        ),
        "preregistration_sha256": freezer._sha256(build.PREREGISTRATION),
        "stage1_amendment_sha256": freezer._sha256(build.STAGE1_AMENDMENT),
        "stage0_manifest_sha256": freezer._sha256(build.STAGE0_MANIFEST),
    }
    assert not any("outcome" in name or "target" in name for name in snapshot)


def test_freezer_requires_exact_held_schema_and_model_topology():
    freezer._validate_held_array(
        "session_ordinal", np.arange(400, dtype=np.int16)
    )
    freezer._validate_held_array(
        "current_chosen", np.zeros((400, base.TURN_COUNT), dtype=np.uint8)
    )
    freezer._validate_held_array(
        "final_activation", np.zeros((400, base.TURN_COUNT), dtype=bool)
    )
    with pytest.raises(freezer.StrictRestackFreezeError, match="schema"):
        freezer._validate_held_array(
            "session_ordinal", np.arange(400, dtype=np.float64) + 0.75
        )
    with pytest.raises(freezer.StrictRestackFreezeError, match="schema"):
        freezer._validate_held_array(
            "current_chosen", np.zeros(base.TURN_COUNT, dtype=np.uint8)
        )
    invalid_choice = np.zeros((400, base.TURN_COUNT), dtype=np.uint8)
    invalid_choice[0, 0] = base.CANDIDATE_COUNT
    with pytest.raises(freezer.StrictRestackFreezeError, match="out of range"):
        freezer._validate_held_array("final_chosen", invalid_choice)

    generic, focused = freezer._expected_model_topology(3)
    assert len(generic) == 24
    assert len(focused) == 6
    assert {domain for _model, domain in focused} == {
        "A_30",
        "A_31",
        "A_32",
        "A_33",
        "A_34",
        "T_3",
    }


def test_freezer_physically_audits_every_shard_file(tmp_path, monkeypatch):
    monkeypatch.setattr(freezer, "ROOT", tmp_path)
    stage_root = tmp_path / "experiments" / "stage1"
    shard_root = stage_root / "first" / "outer_0"
    shard_root.mkdir(parents=True)
    complete = shard_root / "outer_complete.json"
    complete.write_text("{}", encoding="utf-8")
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    array_path = shard_root / "surface.npy"
    np.save(array_path, value, allow_pickle=False)
    record = {
        "path": array_path.relative_to(tmp_path).as_posix(),
        "sha256": freezer._sha256(array_path),
        "bytes": array_path.stat().st_size,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "array_sha256": freezer._array_sha256(value),
    }
    result = {"surface": record}
    assert freezer._validate_all_shard_files(
        stage_root, "first", 0, result
    ) == 1

    unbound = shard_root / "unbound.npy"
    np.save(unbound, np.zeros(1, dtype=np.uint8), allow_pickle=False)
    with pytest.raises(freezer.StrictRestackFreezeError, match="unbound or missing"):
        freezer._validate_all_shard_files(stage_root, "first", 0, result)
