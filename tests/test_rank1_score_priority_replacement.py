from __future__ import annotations

from fractions import Fraction
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_rank1_score_priority_replacement as replay


def _features(version_rows: list[list[int]]) -> np.ndarray:
    versions = np.asarray(version_rows, dtype=np.int16)
    result = np.zeros(
        (
            versions.shape[0],
            versions.shape[1],
            replay.CANDIDATE_COUNT,
            replay.FEATURE_COUNT,
        ),
        dtype=np.float32,
    )
    result[..., replay.VERSION_FEATURE_INDEX] = (
        versions.astype(np.float32) / np.float32(10.0)
    )[:, :, None]
    reset = np.zeros(versions.shape, dtype=bool)
    reset[:, 0] = True
    if versions.shape[1] > 1:
        reset[:, 1:] = np.diff(versions, axis=1) == 1
    ages = np.full(versions.shape, np.float32(0.2), dtype=np.float32)
    ages[reset] = np.float32(0.1)
    result[..., replay.GOAL_AGE_FEATURE_INDEX] = ages[:, :, None]
    override = np.zeros(versions.shape, dtype=np.float32)
    override[reset & (np.arange(versions.shape[1])[None, :] > 0)] = 1.0
    result[..., replay.CURRENT_OVERRIDE_FEATURE_INDEX] = override[:, :, None]
    counts = np.cumsum(override, axis=1).astype(np.float32) / np.float32(5.0)
    result[..., replay.OVERRIDE_COUNT_FEATURE_INDEX] = counts[:, :, None]
    return result


def _turn(prefix: str = "item") -> dict[str, object]:
    c100 = tuple(f"{prefix}-{index}" for index in range(100))
    return {"c100": c100, "actions": {"KEEP_P11": c100[:10]}}


def _metric(
    sample_count: int,
    hit_rate: Fraction,
    mrr: Fraction,
    mttc: Fraction,
) -> replay.MetricValues:
    efficiency = min(
        Fraction(1), max(Fraction(0), (Fraction(11) - mttc) / 10)
    )
    technical = hit_rate / 2 + 3 * mrr / 10 + efficiency / 5
    return replay.MetricValues(
        sample_count,
        hit_rate,
        mrr,
        mttc,
        efficiency,
        technical,
    )


class ScorePriorityPolicyTests(unittest.TestCase):
    def test_raw_float32_order_and_raw_ordinal_tie_break(self) -> None:
        scores = np.full(100, np.float32(-1.0), dtype=np.float32)
        scores[3] = np.float32(2.0)
        scores[7] = np.float32(2.0)
        scores[80] = np.float32(1.0)
        scores[12] = np.nextafter(
            np.float32(1.0), np.float32(np.inf), dtype=np.float32
        )
        order = replay.score_priority_ordinals(scores)
        self.assertEqual(order.dtype, np.uint8)
        self.assertEqual(order.shape, (100,))
        self.assertEqual(order[:4].tolist(), [3, 7, 12, 80])
        self.assertEqual(set(order.tolist()), set(range(100)))

    def test_score_schema_and_nonfinite_values_fail_closed(self) -> None:
        bad_values = (
            np.zeros(99, dtype=np.float32),
            np.zeros(100, dtype=np.float64),
            np.concatenate(
                (np.zeros(99, dtype=np.float32), np.asarray([np.nan], np.float32))
            ),
            np.concatenate(
                (np.zeros(99, dtype=np.float32), np.asarray([np.inf], np.float32))
            ),
        )
        for values in bad_values:
            with self.subTest(shape=values.shape, dtype=str(values.dtype)):
                with self.assertRaises(replay.Rank1ReplayError):
                    replay.score_priority_ordinals(values)

    def test_unseen_rank1_is_identity_and_inputs_are_not_mutated(self) -> None:
        raw = tuple(f"item-{index}" for index in range(100))
        order = tuple(reversed(raw[:10])) + raw[10:]
        scores = np.arange(100, dtype=np.float32)
        scores_before = scores.copy()
        served = {"item-50"}
        served_before = set(served)
        page = replay.rank1_score_priority_replacement(
            order, raw, scores, served
        )
        self.assertEqual(page, order[:10])
        np.testing.assert_array_equal(scores, scores_before)
        self.assertEqual(served, served_before)
        self.assertEqual(order, tuple(reversed(raw[:10])) + raw[10:])

    def test_seen_rank1_selects_best_unseen_legal_raw_ordinal(self) -> None:
        raw = tuple(f"item-{index}" for index in range(100))
        order = raw[:10] + (raw[40], raw[20]) + tuple(
            value for value in raw[10:] if value not in {raw[20], raw[40]}
        )
        scores = np.zeros(100, dtype=np.float32)
        scores[20] = scores[40] = np.float32(5.0)
        page = replay.rank1_score_priority_replacement(
            order, raw, scores, {raw[0], raw[10]}
        )
        self.assertEqual(page, (raw[20], *order[1:10]))
        self.assertEqual(page[1:], order[1:10])

    def test_activation_swap_preserves_raw_score_alignment(self) -> None:
        turn = _turn()
        raw = tuple(turn["c100"])
        order = replay.reconstruct_v19_order(turn, 20, True)
        self.assertEqual(order[9], raw[20])
        self.assertIn(raw[9], order[10:])
        scores = np.zeros(100, dtype=np.float32)
        scores[9] = np.float32(10.0)
        scores[10] = np.float32(9.0)
        page = replay.rank1_score_priority_replacement(
            order, raw, scores, {order[0]}
        )
        self.assertEqual(page[0], raw[9])
        self.assertEqual(page[1:], order[1:10])

    def test_exhausted_legal_pool_is_identity(self) -> None:
        raw = tuple(f"item-{index}" for index in range(100))
        scores = np.arange(100, dtype=np.float32)
        served = {raw[0], *raw[10:]}
        self.assertEqual(
            replay.rank1_score_priority_replacement(raw, raw, scores, served),
            raw[:10],
        )

    def test_duplicate_set_mismatch_and_invalid_scores_fail_closed(self) -> None:
        raw = tuple(f"item-{index}" for index in range(100))
        scores = np.zeros(100, dtype=np.float32)
        invalid = (
            ((raw[0],) * 100, raw, scores),
            (raw, (raw[0],) * 100, scores),
            (raw[:-1] + ("other",), raw, scores),
            (raw, raw, scores.astype(np.float64)),
        )
        for order, raw_c100, values in invalid:
            with self.subTest(first=order[0]):
                with self.assertRaises(replay.Rank1ReplayError):
                    replay.rank1_score_priority_replacement(
                        order, raw_c100, values, {order[0]}
                    )


class ReplayAndResetTests(unittest.TestCase):
    def test_actual_candidate_page_drives_served_state_and_reset(self) -> None:
        traces = (tuple(_turn() for _ in range(4)),)
        scores = np.broadcast_to(
            np.arange(100, dtype=np.float32), (1, 4, 100)
        ).copy()
        chosen = np.zeros((1, 4), dtype=np.int16)
        activation = np.zeros((1, 4), dtype=bool)
        versions = np.asarray([[1, 1, 1, 2]], dtype=np.int16)
        before = scores.copy()
        result = replay.replay_score_priority_pages(
            traces,
            scores,
            chosen,
            activation,
            versions,
            measure_timing=True,
        )
        np.testing.assert_array_equal(
            result.candidate_pages[0, :, 0], [0, 99, 98, 0]
        )
        np.testing.assert_array_equal(
            result.changed, [[False, True, True, False]]
        )
        np.testing.assert_array_equal(result.last_reset_turn, [4])
        self.assertTrue(result.structural["ranks_2_to_10_byte_identical"])
        self.assertEqual(result.timing["sample_count"], 4)
        np.testing.assert_array_equal(scores, before)

    def test_score_priority_replay_is_exactly_repeatable(self) -> None:
        traces = (tuple(_turn() for _ in range(3)),)
        scores = np.broadcast_to(
            np.arange(100, dtype=np.float32), (1, 3, 100)
        ).copy()
        chosen = np.zeros((1, 3), dtype=np.int16)
        activation = np.zeros((1, 3), dtype=bool)
        versions = np.ones((1, 3), dtype=np.int16)
        first = replay.replay_score_priority_pages(
            traces, scores, chosen, activation, versions
        )
        second = replay.replay_score_priority_pages(
            traces, scores, chosen, activation, versions
        )
        np.testing.assert_array_equal(first.baseline_pages, second.baseline_pages)
        np.testing.assert_array_equal(first.candidate_pages, second.candidate_pages)
        np.testing.assert_array_equal(first.changed, second.changed)
        self.assertEqual(first.identity, second.identity)

    def test_fixed_two_page_grace_and_version_reset(self) -> None:
        traces = (tuple(_turn() for _ in range(7)),)
        chosen = np.zeros((1, 7), dtype=np.int16)
        activation = np.zeros((1, 7), dtype=bool)
        versions = np.asarray([[1, 1, 1, 1, 2, 2, 2]], dtype=np.int16)
        result = replay.replay_grace_pages(
            traces, chosen, activation, versions
        )
        np.testing.assert_array_equal(
            result.candidate_pages[0, :, 0], [0, 0, 10, 20, 0, 0, 10]
        )
        np.testing.assert_array_equal(
            result.changed,
            [[False, False, True, True, False, False, True]],
        )
        np.testing.assert_array_equal(result.last_reset_turn, [5])

    def test_ascii_page_digest_matches_frozen_line_format(self) -> None:
        first = tuple(f"A{index}" for index in range(10))
        second = tuple(f"B{index}" for index in range(10))
        third = tuple(f"C{index}" for index in range(10))
        pages = ((first, second), (third,))
        payload = "".join("|".join(page) + "\n" for page in (first, second, third))
        expected = hashlib.sha256(payload.encode("ascii")).hexdigest()
        self.assertEqual(replay.ascii_page_digest(pages), expected)
        changed = ((first, tuple(reversed(second))), (third,))
        self.assertNotEqual(
            replay.ascii_page_digest(pages), replay.ascii_page_digest(changed)
        )

    def test_version_decode_is_causal_and_candidate_invariant(self) -> None:
        versions, reset, audit = replay.decode_intent_versions(
            _features([[1, 1, 2], [1, 1, 1]])
        )
        np.testing.assert_array_equal(versions, [[1, 1, 2], [1, 1, 1]])
        np.testing.assert_array_equal(
            reset, [[True, False, True], [True, False, False]]
        )
        self.assertEqual(audit["version_increment_boundaries"], 1)
        self.assertFalse(audit["proxy_or_eligible_input_used"])
        features = _features([[1, 1, 1]])
        features[0, 1, 1, replay.VERSION_FEATURE_INDEX] = np.float32(0.2)
        with self.assertRaises(replay.Rank1ReplayError):
            replay.decode_intent_versions(features)


class PartitionAndLoaderTests(unittest.TestCase):
    @staticmethod
    def _unbalanced_partition() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outer = np.repeat(np.arange(5, dtype=np.uint8), 400)
        inner_block = np.concatenate(
            (
                np.full(120, 0, np.uint8),
                np.full(100, 1, np.uint8),
                np.full(80, 2, np.uint8),
                np.full(60, 3, np.uint8),
                np.full(40, 4, np.uint8),
            )
        )
        inner = np.tile(inner_block, 5)
        action = np.ones((2000, 10), dtype=bool)
        return outer, inner, action

    def test_valid_unbalanced_source_aligned_partition_passes(self) -> None:
        outer, inner, action = self._unbalanced_partition()
        audit = replay.audit_nested_partition(outer, inner, action)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["outer_counts"], [400] * 5)
        self.assertEqual(audit["inner_counts"], [600, 500, 400, 300, 200])
        self.assertEqual(np.asarray(audit["outer_train_inner_counts"]).shape, (5, 5))
        self.assertTrue(np.all(np.asarray(audit["model_train_action_counts"]) > 0))
        self.assertTrue(np.all(np.asarray(audit["model_valid_action_counts"]) > 0))

    def test_partition_rejects_missing_inner_range_imbalance_and_empty_action(self) -> None:
        outer, inner, action = self._unbalanced_partition()
        cases: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        missing = inner.copy()
        missing[(outer != 0) & (missing == 4)] = 3
        cases.append((outer, missing, action))
        invalid_range = inner.copy()
        invalid_range[0] = 5
        cases.append((outer, invalid_range, action))
        imbalanced_outer = outer.copy()
        imbalanced_outer[0] = 1
        cases.append((imbalanced_outer, inner, action))
        empty_action = action.copy()
        empty_action[(outer != 0) & (inner == 4)] = False
        cases.append((outer, inner, empty_action))
        for case_outer, case_inner, case_action in cases:
            with self.subTest(
                outer_counts=np.bincount(case_outer, minlength=6).tolist()
            ):
                with self.assertRaises(replay.Rank1ReplayError):
                    replay.audit_nested_partition(
                        case_outer, case_inner, case_action
                    )

    def test_label_loader_accesses_exactly_five_members_in_order(self) -> None:
        outer, inner, _action = self._unbalanced_partition()
        source = {
            "baseline_rank": np.zeros((2000, 10), dtype=np.uint8),
            "positive_index": np.full((2000, 10), -1, dtype=np.int16),
            "eligible_from": np.ones(2000, dtype=np.uint8),
            "outer_fold": outer,
            "inner_fold": inner,
            "forbidden_extra": np.ones(1, dtype=np.uint8),
        }

        class TrackingArchive:
            def __init__(self) -> None:
                self.accesses: list[str] = []
                self.close_count = 0

            def __getitem__(self, name: str) -> np.ndarray:
                self.accesses.append(name)
                return source[name]

            def close(self) -> None:
                self.close_count += 1

        archive = TrackingArchive()
        load_calls: list[tuple[object, bool]] = []

        def fake_np_load(handle: object, *, allow_pickle: bool) -> TrackingArchive:
            load_calls.append((handle, allow_pickle))
            return archive

        sentinel = object()
        outcomes = replay.load_outcomes_from_open_handle(sentinel, fake_np_load)
        expected = tuple(name for name, _shape, _dtype in replay.LABEL_MEMBER_SPECS)
        self.assertEqual(
            expected,
            (
                "baseline_rank",
                "positive_index",
                "eligible_from",
                "outer_fold",
                "inner_fold",
            ),
        )
        self.assertEqual(load_calls, [(sentinel, False)])
        self.assertEqual(tuple(archive.accesses), expected)
        self.assertNotIn("forbidden_extra", archive.accesses)
        self.assertEqual(archive.close_count, 1)
        for name in expected:
            copied = getattr(outcomes, name)
            self.assertFalse(copied.flags.writeable)
            self.assertFalse(np.shares_memory(copied, source[name]))


class MetricAndComparatorGateTests(unittest.TestCase):
    def test_official_three_session_rounding_uses_rounded_components(self) -> None:
        state = {
            "hit": np.asarray([True, False, False]),
            "first_rank": np.asarray([3, 0, 0]),
            "first_turn": np.asarray([1, 11, 11]),
        }
        metric = replay.metric_values(state, np.ones(3, dtype=bool))
        self.assertEqual(metric.hit_rate_at_10, Fraction(1, 3))
        self.assertEqual(metric.mrr, Fraction(1, 9))
        self.assertEqual(metric.mttc, Fraction(23, 3))
        self.assertEqual(metric.technical_score, Fraction(4, 15))
        self.assertEqual(
            metric.official(),
            {
                "sample_count": 3,
                "hit_rate_at_10": 0.333333,
                "mrr": 0.111111,
                "mttc": 7.666667,
                "efficiency": 0.333333,
                "technical_score": 0.266666,
            },
        )

    def test_exact_gain_cannot_pass_when_official_delta_is_zero(self) -> None:
        incumbent = _metric(
            10_000_000,
            Fraction(1, 2),
            Fraction(7, 10),
            Fraction(3),
        )
        exact_up = _metric(
            10_000_000,
            Fraction(1, 2) + Fraction(1, 10_000_000),
            Fraction(7, 10),
            Fraction(3),
        )
        exact_down = _metric(
            10_000_000,
            Fraction(1, 2) - Fraction(1, 10_000_000),
            Fraction(7, 10),
            Fraction(3),
        )
        self.assertFalse(
            replay._dual_gt(exact_up, incumbent, "hit_rate_at_10")
        )
        self.assertFalse(
            replay._dual_ge(exact_down, incumbent, "hit_rate_at_10")
        )
        self.assertFalse(
            replay._dual_le(exact_up, incumbent, "hit_rate_at_10")
        )

    def test_local_comparator_gate_passes_only_no_regression_pairs(self) -> None:
        incumbent = _metric(2000, Fraction(1953, 2000), Fraction(7, 10), Fraction(3))
        candidate = _metric(2000, Fraction(49, 50), Fraction(4, 5), Fraction(2))
        incumbent_fold = _metric(400, Fraction(39, 40), Fraction(7, 10), Fraction(3))
        candidate_fold = _metric(400, Fraction(49, 50), Fraction(4, 5), Fraction(2))
        pairs = [(candidate_fold, incumbent_fold)] * 5
        self.assertTrue(replay.passes_local_gates(candidate, incumbent, pairs))

        worse_fold = _metric(400, Fraction(39, 40), Fraction(3, 5), Fraction(4))
        self.assertFalse(
            replay.passes_local_gates(
                candidate, incumbent, [(worse_fold, incumbent_fold), *pairs[1:]]
            )
        )

    def test_local_hr_exact_gain_with_official_tie_fails(self) -> None:
        incumbent = _metric(2000, Fraction(1953, 2000), Fraction(7, 10), Fraction(3))
        candidate = _metric(
            2_000_000,
            Fraction(1_953_001, 2_000_000),
            Fraction(4, 5),
            Fraction(2),
        )
        good_fold = _metric(400, Fraction(49, 50), Fraction(4, 5), Fraction(2))
        incumbent_fold = _metric(400, Fraction(39, 40), Fraction(7, 10), Fraction(3))
        self.assertGreater(candidate.hit_rate_at_10, incumbent.hit_rate_at_10)
        self.assertEqual(
            candidate.official()["hit_rate_at_10"],
            incumbent.official()["hit_rate_at_10"],
        )
        self.assertFalse(
            replay.passes_local_gates(
                candidate, incumbent, [(good_fold, incumbent_fold)] * 5
            )
        )

    def test_global_comparator_gate_passes_strictly_better_candidate(self) -> None:
        grace = _metric(2000, Fraction(991, 1000), Fraction(69, 100), Fraction(3))
        candidate = _metric(2000, Fraction(397, 400), Fraction(4, 5), Fraction(2))
        grace_fold = _metric(400, Fraction(99, 100), Fraction(69, 100), Fraction(3))
        candidate_fold = _metric(400, Fraction(397, 400), Fraction(4, 5), Fraction(2))
        dominance = {
            "hit_to_miss": 0,
            "later_first_hit": 0,
            "same_turn_worse_rank": 0,
            "earlier_hit_not_rank1": 0,
            "new_hit_not_rank1": 0,
            "reset_eligibility_mismatch": 0,
        }
        self.assertTrue(
            replay.passes_global_gates(
                candidate,
                grace,
                [(candidate_fold, grace_fold)] * 5,
                dominance,
                True,
            )
        )
        harmed = dict(dominance, hit_to_miss=1)
        self.assertFalse(
            replay.passes_global_gates(
                candidate,
                grace,
                [(candidate_fold, grace_fold)] * 5,
                harmed,
                True,
            )
        )
        self.assertFalse(
            replay.passes_global_gates(
                candidate,
                grace,
                [(candidate_fold, grace_fold)] * 5,
                dominance,
                False,
            )
        )

    def test_global_hr_exactly_point_991_fails(self) -> None:
        grace = _metric(2000, Fraction(991, 1000), Fraction(69, 100), Fraction(3))
        candidate = _metric(2000, Fraction(991, 1000), Fraction(4, 5), Fraction(2))
        candidate_fold = _metric(400, Fraction(99, 100), Fraction(4, 5), Fraction(2))
        grace_fold = _metric(400, Fraction(99, 100), Fraction(69, 100), Fraction(3))
        dominance = {
            "hit_to_miss": 0,
            "later_first_hit": 0,
            "same_turn_worse_rank": 0,
            "earlier_hit_not_rank1": 0,
            "new_hit_not_rank1": 0,
            "reset_eligibility_mismatch": 0,
        }
        self.assertEqual(candidate.official()["hit_rate_at_10"], 0.991)
        self.assertFalse(
            replay.passes_global_gates(
                candidate,
                grace,
                [(candidate_fold, grace_fold)] * 5,
                dominance,
                True,
            )
        )

    def test_formal_membership_gates_reject_harm(self) -> None:
        local = _metric(
            2000, Fraction(1953, 2000), Fraction(7, 10), Fraction(3)
        )
        candidate = _metric(
            2000, Fraction(49, 50), Fraction(4, 5), Fraction(2)
        )
        local_fold = _metric(
            400, Fraction(39, 40), Fraction(7, 10), Fraction(3)
        )
        candidate_fold = _metric(
            400, Fraction(49, 50), Fraction(4, 5), Fraction(2)
        )
        transition = replay.Transition(local, candidate, 7, 0, 1, 1)
        safe_folds = [
            replay.Transition(local_fold, candidate_fold, 1, 0, 1, 1)
            for _ in range(5)
        ]
        pairs = [(candidate_fold, local_fold)] * 5
        self.assertTrue(
            replay.passes_local_gates(
                candidate,
                local,
                pairs,
                transition=transition,
                fold_transitions=safe_folds,
                require_membership=True,
            )
        )
        harmed = replay.Transition(local, candidate, 8, 1, 1, 1)
        self.assertFalse(
            replay.passes_local_gates(
                candidate,
                local,
                pairs,
                transition=harmed,
                fold_transitions=safe_folds,
                require_membership=True,
            )
        )
        self.assertFalse(
            replay.passes_local_gates(
                candidate,
                local,
                pairs,
                transition=transition,
                fold_transitions=None,
                require_membership=True,
            )
        )


class ReceiptAndBoundaryTests(unittest.TestCase):
    def test_receipt_is_durable_exclusive_and_existing_states_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "nested" / "result.json"
            handle = replay.open_one_shot_receipt(output, root, "a" * 40)
            handle.close()
            marker = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(marker["status"], "CONSUMED_PENDING_RERUN_FORBIDDEN")
            before = output.read_bytes()
            inode = output.stat().st_ino
            with self.assertRaises(replay.Rank1ReplayError):
                replay.open_one_shot_receipt(output, root, "a" * 40)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(output.stat().st_ino, inode)

            for index, payload in enumerate((b"", b"{partial")):
                existing = root / f"existing-{index}.json"
                existing.write_bytes(payload)
                with self.assertRaises(replay.Rank1ReplayError):
                    replay.open_one_shot_receipt(existing, root, "b" * 40)
                self.assertEqual(existing.read_bytes(), payload)

    def test_post_o_excl_fdopen_write_and_fsync_faults_are_consumed(self) -> None:
        implementation = "c" * 40

        def assert_invalid(output: Path, root: Path) -> None:
            before = output.read_bytes()
            self.assertEqual(
                json.loads(before.decode("utf-8"))["status"],
                "INVALID_ONE_SHOT_CONSUMED",
            )
            with self.assertRaises(replay.Rank1ReplayError):
                replay.open_one_shot_receipt(output, root, implementation)
            self.assertEqual(output.read_bytes(), before)

        with self.subTest(failure="fdopen"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                output = root / "fdopen.json"
                real_open = replay.os.open
                real_close = replay.os.close
                descriptors: list[int] = []
                closed: list[int] = []

                def tracking_open(*args: object, **kwargs: object) -> int:
                    descriptor = real_open(*args, **kwargs)
                    descriptors.append(descriptor)
                    return descriptor

                def tracking_close(descriptor: int) -> None:
                    closed.append(descriptor)
                    real_close(descriptor)

                with patch.object(replay.os, "open", side_effect=tracking_open), patch.object(
                    replay.os, "close", side_effect=tracking_close
                ), patch.object(replay.os, "fdopen", side_effect=OSError("fdopen")):
                    with self.assertRaises(replay.Rank1ReplayConsumedError):
                        replay.open_one_shot_receipt(output, root, implementation)
                self.assertEqual(closed, descriptors)
                assert_invalid(output, root)

        with self.subTest(failure="marker_write"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                output = root / "write.json"
                receipt_module = inspect.getmodule(replay.open_one_shot_receipt)
                self.assertIsNotNone(receipt_module)
                original = receipt_module._write_receipt_payload
                calls = 0

                def fail_once(handle: object, value: object) -> tuple[int, str]:
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise OSError("write")
                    return original(handle, value)

                with patch.object(
                    receipt_module, "_write_receipt_payload", side_effect=fail_once
                ):
                    with self.assertRaises(replay.Rank1ReplayConsumedError):
                        replay.open_one_shot_receipt(output, root, implementation)
                self.assertEqual(calls, 2)
                assert_invalid(output, root)

        with self.subTest(failure="marker_fsync"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                output = root / "fsync.json"
                real_fsync = replay.os.fsync
                calls = 0

                def fail_once(descriptor: int) -> None:
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise OSError("fsync")
                    real_fsync(descriptor)

                with patch.object(replay.os, "fsync", side_effect=fail_once):
                    with self.assertRaises(replay.Rank1ReplayConsumedError):
                        replay.open_one_shot_receipt(output, root, implementation)
                self.assertEqual(calls, 2)
                assert_invalid(output, root)

    def test_o_excl_race_has_no_truncate_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "race.json"
            observed: list[int] = []

            def race(_path: str, flags: int, _mode: int) -> int:
                observed.append(flags)
                raise FileExistsError("race")

            with patch.object(replay.os, "open", side_effect=race):
                with self.assertRaises(replay.Rank1ReplayError):
                    replay.open_one_shot_receipt(output, root, "d" * 40)
            self.assertEqual(len(observed), 1)
            self.assertEqual(
                observed[0] & (os.O_CREAT | os.O_EXCL | os.O_RDWR),
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
            )
            self.assertEqual(observed[0] & os.O_TRUNC, 0)

    def test_output_path_and_cli_are_fixed_and_symlink_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "root"
            root.mkdir()
            outside = parent / "outside" / "result.json"
            traversal = root / ".." / "escape" / "result.json"
            with self.assertRaises(replay.Rank1ReplayError):
                replay._check_output_components(outside, root)
            with self.assertRaises(replay.Rank1ReplayError):
                replay._check_output_components(traversal, root)
            link = root / "link"
            candidate = link / "result.json"
            real_exists = Path.exists
            with patch.object(
                Path,
                "exists",
                new=lambda self: self == link or real_exists(self),
            ), patch.object(Path, "is_symlink", new=lambda self: self == link):
                with self.assertRaises(replay.Rank1ReplayError):
                    replay._check_output_components(candidate, root)

        self.assertEqual(
            tuple(inspect.signature(replay.run).parameters),
            ("implementation_commit",),
        )
        self.assertIn(
            "open_one_shot_receipt(OUTPUT_PATH, ROOT", inspect.getsource(replay.run)
        )
        with patch.object(
            sys,
            "argv",
            ["score-priority", "--implementation-commit", "e" * 40, "--output", "x"],
        ), patch.object(replay, "run") as mocked_run, patch(
            "sys.stderr", new=io.StringIO()
        ):
            with self.assertRaises(SystemExit):
                replay.main()
        mocked_run.assert_not_called()

    def test_result_privacy_rejects_identifiers_and_session_vectors(self) -> None:
        replay._result_privacy_scan({"safe": [1, 2, 3]})
        with self.assertRaises(replay.Rank1ReplayError):
            replay._result_privacy_scan({"target_asin": "B012345678"})
        with self.assertRaises(replay.Rank1ReplayError):
            replay._result_privacy_scan({"values": [0] * replay.SESSION_COUNT})
        with self.assertRaises(replay.Rank1ReplayError):
            replay._result_privacy_scan({"array": np.zeros(1, dtype=np.uint8)})


class BindingAndPreflightTests(unittest.TestCase):
    def test_preregistration_binds_score_comparators_and_fixed_counts(self) -> None:
        bound = replay._validate_preregistration()
        self.assertEqual(bound["preregistration_commit"], replay.PREREG_COMMIT)
        self.assertEqual(replay.EXPECTED_INNER_FITS, 50)
        self.assertEqual(replay.EXPECTED_OUTER_FITS, 10)
        self.assertEqual(replay.EXPECTED_FIT_INVOCATIONS, 60)
        self.assertEqual(replay.EXPECTED_SELECTION_INVOCATIONS, 5)
        self.assertEqual(
            replay.EXPECTED_FOLD_QUANTILES,
            (25 / 64, 44 / 64, 26 / 64, 55 / 64, 32 / 64),
        )
        prereg = json.loads(replay.PREREG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            prereg["score_priority_contract"]["priority_order"],
            "numpy.lexsort((original_raw_c100_ordinal, -float32_score)) without rounding, tolerance, normalization, softmax, blend, threshold, or refill",
        )
        self.assertEqual(
            prereg["frozen_comparator_reproduction"]["fit_invocations_exact"],
            60,
        )
        self.assertEqual(
            prereg["frozen_comparator_reproduction"][
                "quantile_selection_invocations_exact"
            ],
            5,
        )

    def test_git_gate_rejects_dirty_tree_before_other_git_reads(self) -> None:
        implementation = "a" * 40
        calls: list[tuple[str, ...]] = []

        def dirty_git(args: tuple[str, ...]) -> str:
            calls.append(args)
            if args[:2] == ("status", "--porcelain=v1"):
                return "?? unexpected.py"
            raise AssertionError("dirty gate must fail before further Git reads")

        with patch.object(replay, "_git", side_effect=dirty_git):
            with self.assertRaises(replay.Rank1ReplayError):
                replay._validate_git_checkpoint(implementation)
        self.assertEqual(len(calls), 1)

    def test_git_gate_source_binds_exact_three_commit_choreography(self) -> None:
        source = inspect.getsource(replay._validate_git_checkpoint)
        for name in (
            "COMPARABILITY_COMMIT",
            "PREREG_COMMIT",
            "IMPLEMENTATION_PATHS",
            "PREREG_PATH_SET",
            "COMPARABILITY_PATH_SET",
            "REMOTE_REF",
        ):
            self.assertIn(name, source)
        self.assertIn("_commit_parent", source)
        self.assertIn("_changed_paths", source)
        self.assertIn("untracked-files=all", source)

    def test_preflight_is_target_free_and_run_consumes_before_label_access(self) -> None:
        preflight_source = inspect.getsource(replay.preflight_only)
        for forbidden in (
            "LABEL_PATH",
            "load_outcomes_from_open_handle",
            "positive_index",
            "eligible_from",
        ):
            self.assertNotIn(forbidden, preflight_source)
        run_source = inspect.getsource(replay.run)
        self.assertLess(
            run_source.index("open_one_shot_receipt"),
            run_source.index("LABEL_PATH.open"),
        )
        self.assertNotIn("frozen._load_inputs", run_source)
        target_free_source = inspect.getsource(replay._load_target_free_inputs)
        self.assertNotIn("_load_proxy_rows", target_free_source)
        self.assertNotIn("_eligible_turn", target_free_source)
        self.assertLess(
            run_source.index("_write_receipt_payload(receipt, result)"),
            run_source.index("final_written = True"),
        )
        self.assertLess(
            run_source.index("final_written = True"),
            run_source.index("_safe_close(receipt)"),
        )

    def test_identity_and_structural_ordering_are_protocol_safe(self) -> None:
        source = inspect.getsource(replay.run)
        self.assertLess(
            source.index("score_equals_local ="),
            source.index("baseline_state = state_from_positive_index"),
        )
        self.assertIn("if score_equals_local:", source)
        self.assertIn("target_metrics_computed\": False", source)
        self.assertLess(
            source.index("if not structural_ok:"),
            source.index("local_pass = bool("),
        )
        structural_block = source[
            source.index("if not structural_ok:") : source.index(
                "local_pass = bool("
            )
        ]
        self.assertIn("raise Rank1ReplayError", structural_block)
        global_block = source[
            source.index("global_pass = bool(") : source.index(
                "if global_pass:"
            )
        ]
        self.assertIn("local_pass", global_block)


if __name__ == "__main__":
    unittest.main()
