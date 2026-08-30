from __future__ import annotations

from dataclasses import replace
import io
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import evaluate_small_ranker_top5_proposal_depth_stage2 as subject


class _Raises:
    def __init__(self, expected: type[BaseException]) -> None:
        self.expected = expected

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        if exc_type is None:
            raise AssertionError("expected %s" % self.expected.__name__)
        if not issubclass(exc_type, self.expected):
            return False
        return True


class _TestSupport:
    @staticmethod
    def raises(expected: type[BaseException]) -> _Raises:
        return _Raises(expected)


pytest = _TestSupport()


def _state(hit: np.ndarray, rank: np.ndarray, turn: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "hit": np.asarray(hit, dtype=bool),
        "first_rank": np.asarray(rank, dtype=np.int16),
        "first_turn": np.asarray(turn, dtype=np.int16),
    }


def test_raw_and_logical_array_hash_contracts_are_distinct_and_c_order_stable() -> None:
    c_order = np.arange(12, dtype=np.int16).reshape(3, 4)
    f_order = np.asfortranarray(c_order)

    assert subject._raw_array_sha256(c_order) == subject._raw_array_sha256(f_order)
    assert subject._logical_array_sha256(c_order) == subject._logical_array_sha256(f_order)
    assert subject._raw_array_sha256(c_order) != subject._logical_array_sha256(c_order)
    assert subject._raw_array_sha256(c_order) != subject._raw_array_sha256(
        c_order.astype(np.int32)
    )


def test_baseline_session_hit_uses_only_eligible_turns_and_rejects_bad_schema() -> None:
    ranks = np.zeros((6, 10), dtype=np.uint8)
    ranks[1, 0] = 10
    ranks[2, 1] = 1
    ranks[3, 2] = 10
    ranks[4, 8] = 1
    ranks[5, 9] = 10
    eligible = np.asarray([1, 1, 3, 3, 10, 10], dtype=np.uint8)

    assert subject.derive_baseline_session_hit(ranks, eligible).tolist() == [
        0,
        1,
        0,
        1,
        0,
        1,
    ]

    for bad_ranks, bad_eligible in (
        (np.zeros((1, 9), dtype=np.uint8), np.ones(1, dtype=np.uint8)),
        (np.full((1, 10), 11, dtype=np.uint8), np.ones(1, dtype=np.uint8)),
        (np.zeros((1, 10), dtype=np.uint8), np.zeros(1, dtype=np.uint8)),
        (np.zeros((1, 10), dtype=np.uint8), np.full(1, 11, dtype=np.uint8)),
    ):
        with pytest.raises(subject.Stage2Error):
            subject.derive_baseline_session_hit(bad_ranks, bad_eligible)


def test_policy_state_protects_top9_replaces_rank10_and_honors_eligibility() -> None:
    baseline = np.zeros((4, 10), dtype=np.uint8)
    positive = np.full((4, 10), -1, dtype=np.int16)
    chosen = np.zeros((4, 10), dtype=np.uint8)
    active = np.zeros((4, 10), dtype=bool)
    eligible = np.asarray([1, 1, 1, 3], dtype=np.uint8)

    baseline[0, 0] = 4
    active[0, 0] = True  # wrong action cannot remove protected rank 1..9
    baseline[1, 0] = 10
    active[1, 0] = True  # wrong action does remove unprotected rank 10
    positive[2, 0] = 7
    chosen[2, 0] = 7
    active[2, 0] = True  # correct action creates rank 10
    baseline[3, 1] = 1  # pre-eligible and ignored
    baseline[3, 3] = 2

    state = subject.policy_session_state(
        {
            "baseline_rank": baseline,
            "positive_index": positive,
            "eligible_from": eligible,
        },
        chosen,
        active,
    )

    assert state["hit"].tolist() == [True, False, True, True]
    assert state["first_rank"].tolist() == [4, 0, 10, 2]
    assert state["first_turn"].tolist() == [1, 11, 1, 4]


def test_official_metric_rounding_matches_known_boundary_vectors() -> None:
    ranks = [1] * 8 + [5] + [7] * 6 + [8] + [9] * 5 + [10] * 16
    hit = np.zeros(200, dtype=bool)
    hit[: len(ranks)] = True
    first_rank = np.zeros(200, dtype=np.int16)
    first_rank[: len(ranks)] = ranks
    first_turn = np.full(200, 11, dtype=np.int16)
    first_turn[0] = 1
    first_turn[1 : len(ranks)] = 2

    metrics = subject.exact_metrics(
        _state(hit, first_rank, first_turn), np.ones(200, dtype=bool)
    )

    assert metrics["rounded"] == {
        "hit_rate_at_10": 0.185,
        "mrr": 0.056688,
        "mttc": 9.33,
        "efficiency": 0.167,
        "technical_score": 0.142906,
    }

    second = subject.exact_metrics(
        _state(
            np.asarray([1, 1, 0]),
            np.asarray([7, 10, 0]),
            np.asarray([1, 1, 11]),
        ),
        np.ones(3, dtype=bool),
    )
    assert second["rounded"]["efficiency"] == 0.666667
    assert second["rounded"]["technical_score"] == 0.490952


class _FakeArchive:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self.values = values
        self.accessed: list[str] = []
        self.closed = False

    @property
    def files(self) -> list[str]:
        raise AssertionError("archive.files must never be enumerated")

    def __getitem__(self, name: str) -> np.ndarray:
        self.accessed.append(name)
        if name not in self.values:
            raise AssertionError("unexpected label member access: %s" % name)
        return self.values[name]

    def close(self) -> None:
        self.closed = True


def test_label_loader_accesses_exactly_five_members_in_order_on_same_handle() -> None:
    outer = np.repeat(np.arange(5, dtype=np.uint8), 400)
    inner = np.tile(np.arange(5, dtype=np.uint8), 400)
    archive = _FakeArchive(
        {
            "baseline_rank": np.zeros((2000, 10), dtype=np.uint8),
            "positive_index": np.full((2000, 10), -1, dtype=np.int16),
            "eligible_from": np.ones(2000, dtype=np.uint8),
            "outer_fold": outer,
            "inner_fold": inner,
        }
    )
    handle = io.BytesIO(b"synthetic-not-an-npz")
    calls: list[tuple[object, bool]] = []

    def fake_load(received: object, *, allow_pickle: bool) -> _FakeArchive:
        calls.append((received, allow_pickle))
        return archive

    outcomes = subject._load_outcomes_from_open_handle(handle, np_load=fake_load)

    assert calls == [(handle, False)]
    assert archive.accessed == [name for name, _shape, _dtype in subject.LABEL_MEMBER_SPECS]
    assert archive.closed is True
    assert outcomes.baseline_rank.flags.writeable is False
    assert handle.closed is False


def test_real_numpy_loader_keeps_caller_owned_synthetic_handle_open() -> None:
    handle = io.BytesIO()
    np.savez(
        handle,
        baseline_rank=np.zeros((2000, 10), dtype=np.uint8),
        positive_index=np.full((2000, 10), -1, dtype=np.int16),
        eligible_from=np.ones(2000, dtype=np.uint8),
        outer_fold=np.repeat(np.arange(5, dtype=np.uint8), 400),
        inner_fold=np.tile(np.arange(5, dtype=np.uint8), 400),
    )
    handle.seek(0)

    subject._load_outcomes_from_open_handle(handle)

    assert handle.closed is False
    handle.seek(0)
    assert handle.read(2) == b"PK"


def test_training_label_derivation_is_row_isolated_from_held_outcomes() -> None:
    baseline = np.zeros((4, 10), dtype=np.uint8)
    positive = np.full((4, 10), -1, dtype=np.int16)
    eligible = np.ones(4, dtype=np.uint8)
    outer = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    inner = np.asarray([0, 1, 2, 3], dtype=np.uint8)
    positive[0, 0] = 4
    outcomes = subject.OutcomeBundle(baseline, positive, eligible, outer, inner)
    poisoned = replace(
        outcomes,
        positive_index=positive.copy(),
        baseline_rank=baseline.copy(),
    )
    poisoned.positive_index[2:] = 99
    poisoned.baseline_rank[2:] = 10
    train_rows = np.asarray([0, 1], dtype=np.int64)
    first = subject._training_labels(outcomes, train_rows)
    second = subject._training_labels(poisoned, train_rows)

    assert set(first) == {
        "baseline_rank",
        "positive_index",
        "eligible_from",
        "inner_fold",
        "baseline_session_hit",
    }
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_fit_counter_separates_helper_invocations_from_constant_returns() -> None:
    class Constant:
        pass

    class FakeBase:
        _ConstantGate = Constant

        @staticmethod
        def _fit_gate_model(
            x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int
        ) -> tuple[object, np.ndarray, np.ndarray]:
            model = Constant() if int(seed) == 1 else object()
            return model, np.zeros(x.shape[1]), np.ones(x.shape[1])

    counters = subject.FitCounters()
    x = np.zeros((2, 3), dtype=np.float32)
    y = np.zeros(2, dtype=np.uint8)
    weights = np.ones(2)
    subject._fit_head(x, y, weights, 1, counters, FakeBase)
    subject._fit_head(x, y, weights, 2, counters, FakeBase)

    assert counters.helper_invocations == 2
    assert counters.constant_gate_returns == 1
    assert counters.liblinear_fit_calls == 1
    assert counters.helper_invocations == (
        counters.constant_gate_returns + counters.liblinear_fit_calls
    )


def test_fit_seed_schedule_covers_exactly_sixty_helper_invocations() -> None:
    seeds = []
    for outer in range(5):
        for inner in range(5):
            for head in (0, 1):
                seeds.append(subject._fit_seed(head, outer, inner))
        for head in (0, 1):
            seeds.append(subject._fit_seed(head, outer))

    assert len(seeds) == 60
    # Historical inner/full-T formulas intentionally share a few numeric
    # seeds; topology is defined by 60 helper calls, not seed uniqueness.
    assert len(set(seeds)) < 60
    assert subject._fit_seed(0, 0, 0) == subject.BASE_SEED
    assert subject._fit_seed(1, 4) == subject.BASE_SEED + 10_000 + 404
    with pytest.raises(subject.Stage2Error):
        subject._fit_seed(2, 0)


def test_higher_quantile_and_keep_mapping_are_frozen() -> None:
    values = np.asarray([0.1, 0.1, 0.9, 1.0], dtype=np.float32)

    assert math.isclose(
        subject._threshold_at_quantile(values, 0.5), 0.9, abs_tol=1e-6
    )
    assert math.isinf(subject._threshold_at_quantile(values, 1.0))
    with pytest.raises(subject.Stage2Error):
        subject._threshold_at_quantile(np.asarray([], dtype=np.float32), 0.5)


def test_secondary_gate_requires_exact_safety_and_strictly_more_than_1943_hits() -> None:
    owner = np.repeat(np.arange(5, dtype=np.uint8), 400)
    baseline_hit = np.zeros(2000, dtype=bool)
    baseline_hit[:1943] = True
    baseline_rank = np.zeros(2000, dtype=np.int16)
    baseline_rank[:1943] = 1
    baseline_turn = np.full(2000, 11, dtype=np.int16)
    baseline_turn[:1943] = 1
    policy_hit = baseline_hit.copy()
    policy_hit[1943] = True
    policy_rank = baseline_rank.copy()
    policy_rank[1943] = 10
    policy_turn = baseline_turn.copy()
    policy_turn[1943] = 10
    activation = np.zeros((2000, 10), dtype=bool)
    activation[1943, 9] = True

    payload = subject.comparison_payload(
        _state(baseline_hit, baseline_rank, baseline_turn),
        _state(policy_hit, policy_rank, policy_turn),
        activation,
        owner,
    )
    gate = subject._comparison_gate(payload, secondary=True)

    assert gate["pass"] is True
    assert gate["checks"]["policy_hits_strictly_above_1943"] is True

    hidden_harm = json.loads(json.dumps(payload))
    hidden_harm["aggregate"]["exact_delta"]["mrr"] = {
        "numerator": -1,
        "denominator": 10**12,
        "decimal": -1e-12,
    }
    hidden_harm["aggregate"]["rounded_delta"]["mrr"] = 0.0
    assert subject._comparison_gate(hidden_harm, secondary=True)["pass"] is False

    hidden_turn_harm = json.loads(json.dumps(payload))
    hidden_turn_harm["aggregate"]["exact_delta"]["mttc"] = {
        "numerator": 1,
        "denominator": 10**12,
        "decimal": 1e-12,
    }
    hidden_turn_harm["aggregate"]["rounded_delta"]["mttc"] = 0.0
    assert subject._comparison_gate(hidden_turn_harm, secondary=True)["pass"] is False


def test_one_shot_receipt_is_exclusive_and_invalid_payload_overwrites_zero_byte_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stage2" / "result.json"
    handle = subject._open_one_shot_receipt(output, tmp_path)
    assert output.exists() and output.stat().st_size == 0
    subject._write_receipt_payload(handle, {"status": "INVALID_STAGE2_ONE_SHOT_CONSUMED"})
    handle.close()

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "INVALID_STAGE2_ONE_SHOT_CONSUMED"
    )
    with pytest.raises(subject.Stage2Error):
        subject._open_one_shot_receipt(output, tmp_path)


def test_best_effort_cleanup_never_overrides_consumed_or_committed_state() -> None:
    class FailingClose:
        def close(self) -> None:
            raise OSError("synthetic close failure")

    assert subject._best_effort_close(FailingClose()) is False
    assert subject._best_effort_close(io.BytesIO()) is True


def test_resource_measurement_unavailable_fails_closed() -> None:
    assert subject._require_process_memory("synthetic", lambda: (10, 20)) == (10, 20)
    for unavailable in ((0, 0), (1, 0), (0, 1)):
        with pytest.raises(subject.Stage2Error):
            subject._require_process_memory(
                "synthetic", lambda value=unavailable: value
            )


def test_outer_registry_mapping_is_order_independent_and_fail_closed() -> None:
    rows = [{"outer_fold": fold, "value": fold * 2} for fold in (3, 0, 4, 1, 2)]
    mapped = subject._outer_map(rows, "synthetic")
    assert [mapped[fold]["value"] for fold in range(5)] == [0, 2, 4, 6, 8]

    with pytest.raises(subject.Stage2Error):
        subject._outer_map(rows[:-1], "synthetic")
    duplicate = list(rows)
    duplicate[-1] = {"outer_fold": 3}
    with pytest.raises(subject.Stage2Error):
        subject._outer_map(duplicate, "synthetic")


def test_registered_array_allowlist_is_exactly_thirty_four_named_paths() -> None:
    paths = subject._expected_registered_array_paths()

    assert len(paths) == 34
    assert len({str(path).lower() for path in paths.values()}) == 34
    assert paths["array:v28:outer:0:session_ordinal"].name == "session_ordinal.npy"
    assert paths["array:stage1a:outer:4:candidates"].name == "candidates.npy"
    assert paths["array:stage1b:outer:3:supplement"].name == "supplement.npy"
    assert paths["array:policy:final_activation"].name == "final_activation.npy"


def test_final_policy_must_be_exact_current_plus_registered_supplement() -> None:
    current_chosen = np.zeros((2, 10), dtype=np.uint8)
    current_activation = np.zeros((2, 10), dtype=bool)
    supplement = np.zeros((2, 10), dtype=bool)
    supplement[1, 4] = True
    candidates = np.full((2, 10, 2), -1, dtype=np.int16)
    available = np.zeros((2, 10, 2), dtype=bool)
    candidates[1, 4, 0] = 7
    available[1, 4, 0] = True
    final_chosen = current_chosen.copy()
    final_chosen[1, 4] = 7
    final_activation = current_activation | supplement

    subject._validate_final_composition(
        current_chosen,
        current_activation,
        final_chosen,
        final_activation,
        supplement,
        candidates,
        available,
    )
    broken = final_chosen.copy()
    broken[1, 4] = 9
    with pytest.raises(subject.Stage2Error):
        subject._validate_final_composition(
            current_chosen,
            current_activation,
            broken,
            final_activation,
            supplement,
            candidates,
            available,
        )


def test_report_privacy_allows_negative_audits_but_rejects_raw_outcome_keys() -> None:
    subject._result_privacy_scan(
        {
            "target_runtime_features": 0,
            "identifiers_or_target_strings": 0,
            "membership_arrays": 0,
        }
    )
    with pytest.raises(subject.Stage2Error):
        subject._result_privacy_scan({"positive_index": [1, 2, 3]})
    with pytest.raises(subject.Stage2Error):
        subject._result_privacy_scan({"value": "B012345678"})
    with pytest.raises(subject.Stage2Error):
        subject._result_privacy_scan({"session_ordinals": [0] * 2000})


def test_physical_audit_identity_is_bound_into_strict_decision_hash() -> None:
    first = subject._physical_decision_binding(
        {"identity_sha256": "a" * 64, "input_count": 55, "total_bytes": 1234},
        "b" * 64,
        99,
        (1, 2, 3, 4),
    )
    second = subject._physical_decision_binding(
        {"identity_sha256": "c" * 64, "input_count": 55, "total_bytes": 1234},
        "b" * 64,
        99,
        (1, 2, 3, 4),
    )

    assert subject._canonical_sha256({"required": first}) != subject._canonical_sha256(
        {"required": second}
    )


def test_legacy_quantile_metric_and_official_final_metric_are_intentionally_separate() -> None:
    ranks = [1] * 8 + [5] + [7] * 6 + [8] + [9] * 5 + [10] * 16
    hit = np.zeros(200, dtype=bool)
    hit[: len(ranks)] = True
    first_rank = np.zeros(200, dtype=np.int16)
    first_rank[: len(ranks)] = ranks
    first_turn = np.full(200, 11, dtype=np.int16)
    first_turn[0] = 1
    first_turn[1 : len(ranks)] = 2
    state = _state(hit, first_rank, first_turn)
    mask = np.ones(200, dtype=bool)

    assert subject._legacy_rounded_metrics(state, mask)["technical_score"] == 0.142907
    assert subject.exact_metrics(state, mask)["rounded"]["technical_score"] == 0.142906


def test_proposal_union_counts_each_reachable_session_once() -> None:
    hit = np.ones(2000, dtype=bool)
    hit[-1] = False
    state = _state(
        hit,
        np.where(hit, 1, 0),
        np.where(hit, 1, 11),
    )
    candidates = np.full((2000, 10, 15), -1, dtype=np.int16)
    available = np.zeros_like(candidates, dtype=bool)
    candidates[-1, 0, :2] = 7
    available[-1, 0, :2] = True
    positive = np.full((2000, 10), -1, dtype=np.int16)
    positive[-1, 0] = 7
    outcomes = subject.OutcomeBundle(
        np.zeros((2000, 10), dtype=np.uint8),
        positive,
        np.ones(2000, dtype=np.uint8),
        np.repeat(np.arange(5, dtype=np.uint8), 400),
        np.tile(np.arange(5, dtype=np.uint8), 400),
    )
    owner = outcomes.outer_fold

    report = subject._proposal_union_one(
        "domain_local_current", state, candidates, available, outcomes, owner
    )

    assert report["correct_action_rows_on_current_misses"] == 2
    assert report["reachable_sessions"] == 1
    assert report["maximum_zero_harm_ceiling_hits"] == 2000


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        parameters = list(inspect.signature(function).parameters)
        if not parameters:
            suite.addTest(unittest.FunctionTestCase(function, description=name))
        elif parameters == ["tmp_path"]:
            def with_temporary_path(fn=function) -> None:
                with tempfile.TemporaryDirectory() as directory:
                    fn(Path(directory))

            suite.addTest(unittest.FunctionTestCase(with_temporary_path, description=name))
        else:
            raise AssertionError("unsupported synthetic test fixture: %s" % name)
    return suite


if __name__ == "__main__":
    unittest.main()
