from __future__ import annotations

from fractions import Fraction
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_grace_score_tail_smoke as replay


def _raw(prefix: str = "item") -> tuple[str, ...]:
    return tuple(f"{prefix}-{index}" for index in range(100))


def _turn(prefix: str = "item") -> dict[str, object]:
    c100 = _raw(prefix)
    return {"c100": c100, "actions": {"KEEP_P11": c100[:10]}}


def _metric(
    sample_count: int,
    hit_rate: Fraction,
    mrr: Fraction,
    mttc: Fraction,
) -> replay.MetricValues:
    efficiency = min(Fraction(1), max(Fraction(0), (Fraction(11) - mttc) / 10))
    technical = hit_rate / 2 + 3 * mrr / 10 + efficiency / 5
    return replay.MetricValues(
        sample_count, hit_rate, mrr, mttc, efficiency, technical
    )


class GraceScoreTailPolicyTests(unittest.TestCase):
    def test_age_three_page_is_h_then_score_priority_tail_then_seen_fallback(self) -> None:
        raw = _raw()
        scores = np.zeros(100, dtype=np.float32)
        scores[20] = scores[40] = np.float32(5.0)
        scores[30] = np.float32(4.0)
        priority = replay.score_priority_ordinals(scores)
        served = set(raw[2:10])
        before = set(served)

        page = replay.compose_grace_score_tail_page(
            raw, raw, priority, served, age=3
        )

        self.assertEqual(
            page,
            (
                raw[0],
                raw[1],
                raw[20],
                raw[40],
                raw[30],
                raw[10],
                raw[11],
                raw[12],
                raw[13],
                raw[14],
            ),
        )
        self.assertEqual(served, before)

    def test_age_one_and_two_are_exact_identity_even_when_already_served(self) -> None:
        raw = _raw()
        priority = np.arange(99, -1, -1, dtype=np.uint8)
        served = set(raw)
        for age in (1, 2):
            with self.subTest(age=age):
                self.assertEqual(
                    replay.compose_grace_score_tail_page(
                        raw, raw, priority, served, age
                    ),
                    raw[:10],
                )

    def test_seen_members_are_only_final_fallback(self) -> None:
        raw = _raw()
        scores = np.zeros(100, dtype=np.float32)
        scores[99] = np.float32(10.0)
        served = set(raw) - {raw[0], raw[99]}
        page = replay.compose_grace_score_tail_page(
            raw,
            raw,
            replay.score_priority_ordinals(scores),
            served,
            age=3,
        )
        self.assertEqual(page, (raw[0], raw[99], *raw[1:9]))

    def test_actual_returned_pages_update_served_and_version_reset_regrants_grace(self) -> None:
        traces = (tuple(_turn() for _ in range(6)),)
        scores = np.broadcast_to(
            np.arange(100, dtype=np.float32), (1, 6, 100)
        ).copy()
        chosen = np.zeros((1, 6), dtype=np.int16)
        activation = np.zeros((1, 6), dtype=bool)
        versions = np.asarray([[1, 1, 1, 2, 2, 2]], dtype=np.int16)

        result = replay.replay_grace_score_tail_pages(
            traces, scores, chosen, activation, versions, measure_timing=True
        )

        np.testing.assert_array_equal(
            result.candidate_pages[0, :, 0], [0, 0, 99, 0, 0, 99]
        )
        np.testing.assert_array_equal(
            result.candidate_pages[0, 2], np.arange(99, 89, -1, dtype=np.int16)
        )
        np.testing.assert_array_equal(
            result.changed, [[False, False, True, False, False, True]]
        )
        np.testing.assert_array_equal(result.last_reset_turn, [4])
        self.assertEqual(result.timing["sample_count"], 6)

        repeat = replay.replay_grace_score_tail_pages(
            traces, scores, chosen, activation, versions, measure_timing=True
        )
        np.testing.assert_array_equal(result.candidate_pages, repeat.candidate_pages)
        np.testing.assert_array_equal(result.changed, repeat.changed)
        self.assertEqual(result.identity, repeat.identity)

        four_turns = (tuple(_turn() for _ in range(4)),)
        continued = replay.replay_grace_score_tail_pages(
            four_turns,
            scores[:, :4],
            chosen[:, :4],
            activation[:, :4],
            np.ones((1, 4), dtype=np.int16),
        )
        np.testing.assert_array_equal(
            continued.candidate_pages[0, 3],
            np.arange(89, 79, -1, dtype=np.int16),
        )

    def test_invalid_identity_priority_score_and_replay_schemas_fail_closed(self) -> None:
        raw = _raw()
        priority = np.arange(100, dtype=np.uint8)
        invalid_pages = (
            ((raw[0],) * 100, raw, priority, 3),
            (raw, (raw[0],) * 100, priority, 3),
            (raw[:-1] + ("other",), raw, priority, 3),
            (raw, raw, np.zeros(100, dtype=np.uint8), 3),
            (raw, raw, np.arange(99, dtype=np.uint8), 3),
            (raw, raw, priority, 0),
        )
        for order, raw_c100, candidate_priority, age in invalid_pages:
            with self.subTest(age=age, first=order[0]):
                with self.assertRaises(replay.Rank1ReplayError):
                    replay.compose_grace_score_tail_page(
                        order, raw_c100, candidate_priority, set(), age
                    )

        for scores in (
            np.zeros(100, dtype=np.float64),
            np.full(100, np.nan, dtype=np.float32),
            np.zeros(99, dtype=np.float32),
        ):
            with self.subTest(score_shape=scores.shape, dtype=str(scores.dtype)):
                with self.assertRaises(replay.Rank1ReplayError):
                    replay.score_priority_ordinals(scores)

        traces = ((_turn(),),)
        chosen = np.zeros((1, 1), dtype=np.int16)
        activation = np.zeros((1, 1), dtype=bool)
        versions = np.ones((1, 1), dtype=np.int16)
        with self.assertRaises(replay.Rank1ReplayError):
            replay.replay_grace_score_tail_pages(
                traces,
                np.zeros((1, 1, 100), dtype=np.float64),
                chosen,
                activation,
                versions,
            )


class PrefixAndGraceMaskTests(unittest.TestCase):
    def test_prefix_canonicalization_normalizes_ordinal_and_appends_one_lf(self) -> None:
        records = [
            {"z": 2, "ordinal": 2, "nested": {"b": 1, "a": "café"}},
            {"ordinal": 1, "values": [2, 1]},
        ]
        before = json.loads(json.dumps(records, ensure_ascii=False))
        normalized = [dict(records[0], ordinal=1002), dict(records[1], ordinal=1001)]
        expected = b"".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
            for row in normalized
        )

        actual = replay.canonical_trace_prefix(records, shard_number=3)

        self.assertEqual(actual, expected)
        self.assertEqual(hashlib.sha256(actual).hexdigest(), hashlib.sha256(expected).hexdigest())
        self.assertEqual(records, before)
        with self.assertRaises((replay.Rank1ReplayError, ValueError)):
            replay.canonical_trace_prefix(
                [{"ordinal": 1, "bad": float("nan")}], shard_number=1
            )

    def test_grace_mask_is_exactly_age_at_most_two_and_pages_are_identical(self) -> None:
        versions = np.asarray(
            [[1, 1, 1, 2, 2, 2], [1, 2, 2, 2, 3, 3]], dtype=np.int16
        )
        age, grace_mask = replay.intent_age_and_grace_mask(versions)
        np.testing.assert_array_equal(
            age,
            [[1, 2, 3, 1, 2, 3], [1, 1, 2, 3, 1, 2]],
        )
        np.testing.assert_array_equal(grace_mask, age <= 2)

        traces = tuple(tuple(_turn(f"s{session}") for _ in range(6)) for session in range(2))
        scores = np.broadcast_to(
            np.arange(100, dtype=np.float32), (2, 6, 100)
        ).copy()
        bundle = replay.replay_grace_score_tail_pages(
            traces,
            scores,
            np.zeros((2, 6), dtype=np.int16),
            np.zeros((2, 6), dtype=bool),
            versions,
        )
        np.testing.assert_array_equal(
            bundle.candidate_pages[grace_mask], bundle.baseline_pages[grace_mask]
        )
        self.assertTrue(bundle.structural["grace_mask_pages_array_equal"])
        self.assertTrue(bundle.structural["grace_mask_pages_raw_bytes_equal"])


class SmokeGateTests(unittest.TestCase):
    def test_gate_requires_exact_and_official_safe_gain_and_all_mechanical_flags(self) -> None:
        baseline = _metric(100, Fraction(99, 100), Fraction(7, 10), Fraction(3))
        candidate = _metric(100, Fraction(1), Fraction(71, 100), Fraction(29, 10))
        transition = replay.Transition(baseline, candidate, 1, 0, 2, 1)
        self.assertTrue(replay.passes_smoke_gates(transition, True, True))
        self.assertFalse(
            replay.passes_smoke_gates(
                replay.Transition(baseline, candidate, 2, 1, 2, 1), True, True
            )
        )
        self.assertFalse(replay.passes_smoke_gates(transition, False, True))
        self.assertFalse(replay.passes_smoke_gates(transition, True, False))

        exact_only_baseline = _metric(
            100, Fraction(9_910_001, 10_000_000), Fraction(7, 10), Fraction(3)
        )
        exact_only_candidate = _metric(
            100, Fraction(9_910_002, 10_000_000), Fraction(7, 10), Fraction(3)
        )
        self.assertGreater(
            exact_only_candidate.hit_rate_at_10,
            exact_only_baseline.hit_rate_at_10,
        )
        self.assertEqual(
            exact_only_candidate.official()["hit_rate_at_10"],
            exact_only_baseline.official()["hit_rate_at_10"],
        )
        self.assertFalse(
            replay.passes_smoke_gates(
                replay.Transition(
                    exact_only_baseline, exact_only_candidate, 1, 0, 1, 1
                ),
                True,
                True,
            )
        )


class ReceiptAndBoundaryTests(unittest.TestCase):
    def test_safe_close_cannot_preempt_invalid_receipt_cleanup(self) -> None:
        class InterruptingHandle:
            def close(self) -> None:
                raise KeyboardInterrupt

        replay._safe_close_handle(InterruptingHandle())
        with patch.object(replay.os, "close", side_effect=SystemExit):
            replay._safe_close_descriptor(7)

    def test_receipt_requires_prepared_real_parent_and_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            missing = root / "missing" / "result.json"
            with self.assertRaises(replay.Rank1ReplayError):
                replay._open_receipt(missing, root, "a" * 40)
            self.assertFalse(missing.parent.exists())

            parent = root / "ready"
            parent.mkdir()
            output = parent / "result.json"
            descriptor = replay._open_receipt(output, root, "b" * 40)
            os.close(descriptor)
            before = output.read_bytes()
            self.assertEqual(
                json.loads(before.decode("utf-8"))["status"],
                "CONSUMED_PENDING_RERUN_FORBIDDEN",
            )
            with self.assertRaises(replay.Rank1ReplayError):
                replay._open_receipt(output, root, "b" * 40)
            self.assertEqual(output.read_bytes(), before)

    def test_receipt_uses_o_excl_without_truncate_and_rejects_reparse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent = root / "ready"
            parent.mkdir()
            output = parent / "result.json"
            observed: list[int] = []

            def race(_path: str, flags: int, _mode: int) -> int:
                observed.append(flags)
                raise FileExistsError("race")

            with patch.object(replay.os, "open", side_effect=race):
                with self.assertRaises(replay.Rank1ReplayError):
                    replay._open_receipt(output, root, "c" * 40)
            self.assertEqual(
                observed[0] & (os.O_WRONLY | os.O_CREAT | os.O_EXCL),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            )
            self.assertEqual(observed[0] & os.O_RDWR, 0)
            self.assertEqual(observed[0] & os.O_TRUNC, 0)

            with patch.object(
                replay,
                "_is_link_or_reparse",
                side_effect=lambda path: path == parent,
            ):
                with self.assertRaises(replay.Rank1ReplayError):
                    replay._open_receipt(output, root, "d" * 40)
            self.assertFalse(output.exists())

    def test_baseexception_after_consumption_writes_invalid_and_forbids_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent = root / "ready"
            parent.mkdir()
            output = parent / "result.json"
            original = replay._write_descriptor
            calls = 0

            def interrupt_once(descriptor: int, value: object) -> tuple[int, str]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise KeyboardInterrupt
                return original(descriptor, value)

            with patch.object(replay, "_write_descriptor", side_effect=interrupt_once):
                with self.assertRaises(replay.Rank1ReplayConsumedError):
                    replay._open_receipt(output, root, "e" * 40)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "INVALID_ONE_SHOT_CONSUMED")
            self.assertEqual(payload["error_class"], "KeyboardInterrupt")
            self.assertTrue(payload["rerun_forbidden"])
            with self.assertRaises(replay.Rank1ReplayError):
                replay._open_receipt(output, root, "e" * 40)

    def test_privacy_fixed_cli_and_target_access_order_fail_closed(self) -> None:
        replay._result_privacy_scan({"safe": {"count": 2, "folds": [1, 1]}})
        for payload in (
            {"target_asin": "B012345678"},
            {"per_session": [1]},
            {"values": [0] * 100},
            {"array": np.zeros(1, dtype=np.uint8)},
        ):
            with self.subTest(payload_type=type(next(iter(payload.values()))).__name__):
                with self.assertRaises(replay.Rank1ReplayError):
                    replay._result_privacy_scan(payload)

        self.assertEqual(tuple(inspect.signature(replay.run).parameters), ("implementation_commit",))
        run_source = inspect.getsource(replay.run)
        self.assertLess(run_source.index("_open_receipt"), run_source.index("LABEL_PATH.open"))
        self.assertIn("except BaseException", run_source)
        preflight_source = inspect.getsource(replay.preflight_only)
        for forbidden in ("LABEL_PATH", "positive_index", "eligible_from"):
            self.assertNotIn(forbidden, preflight_source)

    def test_git_gate_binds_exact_path_set_and_rejects_dirty_tree_first(self) -> None:
        self.assertEqual(
            replay.IMPLEMENTATION_PATHS,
            {
                "scripts/evaluate_grace_score_tail_smoke.py",
                "tests/test_grace_score_tail_smoke.py",
            },
        )
        source = inspect.getsource(replay._validate_git_checkpoint)
        for name in ("PREREG_COMMIT", "IMPLEMENTATION_PATHS", "REMOTE_REF"):
            self.assertIn(name, source)
        self.assertIn("untracked-files=all", source)

        calls: list[tuple[str, ...]] = []

        def dirty_git(args: tuple[str, ...]) -> str:
            calls.append(args)
            if args[:2] == ("status", "--porcelain=v1"):
                return "?? unexpected.py"
            raise AssertionError("dirty gate must fail before other Git reads")

        with patch.object(replay, "_git", side_effect=dirty_git):
            with self.assertRaises(replay.Rank1ReplayError):
                replay._validate_git_checkpoint("f" * 40)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
