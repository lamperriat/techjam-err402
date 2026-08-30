from __future__ import annotations

from fractions import Fraction
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

from scripts import evaluate_rank1_seen_replacement as replay


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
    encoded = versions.astype(np.float32) / np.float32(10.0)
    result[..., replay.VERSION_FEATURE_INDEX] = encoded[:, :, None]
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


class Rank1PolicyTests(unittest.TestCase):
    def test_unseen_rank1_is_identity(self) -> None:
        order = tuple(f"item-{index}" for index in range(100))
        page = replay.rank1_seen_replacement(order, {"item-4"})
        self.assertEqual(page, order[:10])

    def test_seen_rank1_uses_highest_unseen_tail_only(self) -> None:
        order = tuple(f"item-{index}" for index in range(100))
        page = replay.rank1_seen_replacement(
            order, {"item-0", "item-10", "item-40"}
        )
        self.assertEqual(page, ("item-11", *order[1:10]))
        self.assertEqual(page[1:], order[1:10])

    def test_tail_exhaustion_is_identity(self) -> None:
        order = tuple(f"item-{index}" for index in range(100))
        served = {order[0], *order[10:]}
        self.assertEqual(
            replay.rank1_seen_replacement(order, served), order[:10]
        )

    def test_duplicate_or_short_order_fails_closed(self) -> None:
        with self.assertRaises(replay.Rank1ReplayError):
            replay.rank1_seen_replacement(("same",) * 100, set())
        with self.assertRaises(replay.Rank1ReplayError):
            replay.rank1_seen_replacement(tuple(map(str, range(99))), set())

    def test_v19_order_reconstructs_slot10_swap(self) -> None:
        turn = _turn()
        c100 = tuple(turn["c100"])
        turn["actions"] = {"KEEP_P11": tuple(reversed(c100[:10]))}
        unchanged = replay.reconstruct_v19_order(turn, 0, False)
        self.assertEqual(unchanged[:10], tuple(reversed(c100[:10])))
        changed = replay.reconstruct_v19_order(turn, 20, True)
        self.assertEqual(changed[9], "item-20")
        self.assertEqual(changed[20], "item-0")
        self.assertEqual(set(changed), set(c100))

    def test_activated_top10_challenger_fails_closed(self) -> None:
        with self.assertRaises(replay.Rank1ReplayError):
            replay.reconstruct_v19_order(_turn(), 5, True)


class VersionResetTests(unittest.TestCase):
    def test_version_delta_is_candidate_invariant_causal_reset(self) -> None:
        features = _features([[1, 1, 2], [1, 1, 1]])
        versions, reset, audit = replay.decode_intent_versions(features)
        np.testing.assert_array_equal(versions, [[1, 1, 2], [1, 1, 1]])
        np.testing.assert_array_equal(
            reset, [[True, False, True], [True, False, False]]
        )
        self.assertEqual(audit["version_increment_boundaries"], 1)
        self.assertFalse(audit["proxy_or_eligible_input_used"])

    def test_candidate_dependent_version_fails_closed(self) -> None:
        features = _features([[1, 1, 1]])
        features[0, 1, 1, replay.VERSION_FEATURE_INDEX] = np.float32(0.2)
        with self.assertRaises(replay.Rank1ReplayError):
            replay.decode_intent_versions(features)

    def test_version_jump_or_decrease_fails_closed(self) -> None:
        with self.assertRaises(replay.Rank1ReplayError):
            replay.decode_intent_versions(_features([[1, 3, 3]]))
        with self.assertRaises(replay.Rank1ReplayError):
            replay.decode_intent_versions(_features([[1, 2, 1]]))

    def test_early_saturation_and_bad_reset_age_fail_closed(self) -> None:
        with self.assertRaises(replay.Rank1ReplayError):
            replay.decode_intent_versions(_features([[1, 10, 10]]))
        features = _features([[1, 1, 2]])
        features[0, 2, :, replay.GOAL_AGE_FEATURE_INDEX] = np.float32(0.2)
        with self.assertRaises(replay.Rank1ReplayError):
            replay.decode_intent_versions(features)

    def test_bad_float_encoding_fails_closed(self) -> None:
        features = _features([[1, 1, 1]])
        features[0, 1, :, replay.VERSION_FEATURE_INDEX] = np.float32(0.1001)
        with self.assertRaises(replay.Rank1ReplayError):
            replay.decode_intent_versions(features)


class ReplayStateTests(unittest.TestCase):
    def test_actual_candidate_page_updates_seen_and_version_resets(self) -> None:
        traces = (
            (_turn("a"), _turn("a"), _turn("a")),
            (_turn("b"), _turn("b"), _turn("b")),
        )
        chosen = np.zeros((2, 3), dtype=np.int16)
        activation = np.zeros((2, 3), dtype=bool)
        versions = np.asarray([[1, 1, 2], [1, 1, 1]], dtype=np.int16)
        result = replay.replay_pages(
            traces, chosen, activation, versions, measure_timing=True
        )
        np.testing.assert_array_equal(
            result.changed, [[False, True, False], [False, True, True]]
        )
        self.assertEqual(int(result.candidate_pages[0, 1, 0]), 10)
        self.assertEqual(int(result.candidate_pages[0, 2, 0]), 0)
        self.assertEqual(int(result.candidate_pages[1, 2, 0]), 11)
        np.testing.assert_array_equal(result.last_reset_turn, [3, 1])
        self.assertTrue(result.structural["ranks_2_to_10_byte_identical"])
        self.assertEqual(result.timing["sample_count"], 6)

    def test_policy_state_obeys_eligibility_without_proxy(self) -> None:
        pages = np.asarray(
            [
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 3,
                [[10, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 3,
            ],
            dtype=np.int16,
        )
        positive = np.asarray([[0, 0, 0], [10, 10, 10]], dtype=np.int16)
        state = replay.state_from_positive_index(
            pages, positive, np.asarray([2, 1], dtype=np.uint8)
        )
        np.testing.assert_array_equal(state["hit"], [True, True])
        np.testing.assert_array_equal(state["first_turn"], [2, 1])
        np.testing.assert_array_equal(state["first_rank"], [1, 1])

    def test_dominance_audit_catches_each_harm_class(self) -> None:
        baseline = {
            "hit": np.asarray([True, True, True, False]),
            "first_turn": np.asarray([2, 2, 2, 11]),
            "first_rank": np.asarray([1, 2, 2, 0]),
        }
        candidate = {
            "hit": np.asarray([False, True, True, True]),
            "first_turn": np.asarray([11, 3, 2, 1]),
            "first_rank": np.asarray([0, 1, 3, 2]),
        }
        audit = replay.dominance_audit(baseline, candidate)
        self.assertEqual(audit["hit_to_miss"], 1)
        self.assertEqual(audit["later_first_hit"], 1)
        self.assertEqual(audit["same_turn_worse_rank"], 1)
        self.assertEqual(audit["new_hit_not_rank1"], 1)


class MetricGateTests(unittest.TestCase):
    def test_exact_and_official_safe_gain_passes(self) -> None:
        baseline = {
            "hit": np.asarray([True] * 9 + [False]),
            "first_turn": np.asarray([2] * 9 + [11]),
            "first_rank": np.asarray([1] * 9 + [0]),
        }
        candidate = {
            "hit": np.asarray([True] * 10),
            "first_turn": np.asarray([1] * 10),
            "first_rank": np.asarray([1] * 10),
        }
        changed = np.ones((10, 1), dtype=bool)
        aggregate = replay.transition_metrics(
            baseline, candidate, changed, np.ones(10, dtype=bool)
        )
        folds = [
            replay.transition_metrics(
                baseline,
                candidate,
                changed,
                np.arange(10) // 2 == fold,
            )
            for fold in range(5)
        ]
        self.assertTrue(
            replay.passes_promotion_gates(aggregate, folds, True, True)
        )

    def test_exact_negative_cannot_hide_behind_official_zero(self) -> None:
        self.assertFalse(
            replay._dual_nonnegative(Fraction(-1, 10_000_000), 0.0)
        )
        self.assertFalse(
            replay._dual_nonpositive(Fraction(1, 10_000_000), 0.0)
        )
        self.assertFalse(
            replay._dual_strict_positive(Fraction(1, 10_000_000), 0.0)
        )

    def test_official_rounding_uses_rounded_component_metrics(self) -> None:
        state = {
            "hit": np.asarray([True, False, False]),
            "first_rank": np.asarray([3, 0, 0]),
            "first_turn": np.asarray([1, 11, 11]),
        }
        metric = replay.metric_values(state, np.ones(3, dtype=bool))
        self.assertEqual(metric.hit_rate_at_10, Fraction(1, 3))
        self.assertEqual(metric.mrr, Fraction(1, 9))
        self.assertEqual(metric.mttc, Fraction(23, 3))
        self.assertEqual(metric.efficiency, Fraction(1, 3))
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
        self.assertNotEqual(
            metric.official()["technical_score"],
            round(float(metric.technical_score), 6),
        )


class ReceiptAndBoundaryTests(unittest.TestCase):
    def test_receipt_is_durable_nonempty_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "nested" / "result.json"
            handle = replay.open_one_shot_receipt(output, root, "a" * 40)
            handle.close()
            marker = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                marker["status"], "CONSUMED_PENDING_RERUN_FORBIDDEN"
            )
            self.assertGreater(output.stat().st_size, 0)
            with self.assertRaises(replay.Rank1ReplayError):
                replay.open_one_shot_receipt(output, root, "a" * 40)

    def test_invalid_payload_overwrites_same_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "result.json"
            handle = replay.open_one_shot_receipt(output, root, "b" * 40)
            replay._write_receipt_payload(
                handle, {"status": "INVALID_ONE_SHOT_CONSUMED"}
            )
            handle.close()
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["status"],
                "INVALID_ONE_SHOT_CONSUMED",
            )

    def test_post_o_excl_failures_are_permanently_consumed(self) -> None:
        implementation = "c" * 40

        def assert_invalid_and_permanent(output: Path, root: Path) -> None:
            before = output.read_bytes()
            self.assertEqual(
                json.loads(before.decode("utf-8"))["status"],
                "INVALID_ONE_SHOT_CONSUMED",
            )
            inode = output.stat().st_ino
            with self.assertRaises(replay.Rank1ReplayError):
                replay.open_one_shot_receipt(output, root, implementation)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(output.stat().st_ino, inode)

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
                assert_invalid_and_permanent(output, root)

        with self.subTest(failure="marker_write"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                output = root / "write.json"
                original = replay._write_receipt_payload
                calls = 0

                def fail_once(handle: object, value: object) -> tuple[int, str]:
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise OSError("marker write")
                    return original(handle, value)

                with patch.object(replay, "_write_receipt_payload", side_effect=fail_once):
                    with self.assertRaises(replay.Rank1ReplayConsumedError):
                        replay.open_one_shot_receipt(output, root, implementation)
                self.assertEqual(calls, 2)
                assert_invalid_and_permanent(output, root)

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
                        raise OSError("marker fsync")
                    real_fsync(descriptor)

                with patch.object(replay.os, "fsync", side_effect=fail_once):
                    with self.assertRaises(replay.Rank1ReplayConsumedError):
                        replay.open_one_shot_receipt(output, root, implementation)
                self.assertEqual(calls, 2)
                assert_invalid_and_permanent(output, root)

    def test_existing_receipt_states_are_rejected_without_mutation(self) -> None:
        payloads = (
            b"",
            b"{partial",
            b'{"status":"CONSUMED_PENDING_RERUN_FORBIDDEN"}\n',
            b'{"status":"INVALID_ONE_SHOT_CONSUMED"}\n',
            b'{"status":"NO_GO_CLOSE_RANK1_SEEN_REPLACEMENT"}\n',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for index, payload in enumerate(payloads):
                with self.subTest(index=index):
                    output = root / f"existing-{index}.json"
                    output.write_bytes(payload)
                    inode = output.stat().st_ino
                    with patch.object(
                        replay.os, "open", side_effect=AssertionError("must not open")
                    ) as mocked_open:
                        with self.assertRaises(replay.Rank1ReplayError):
                            replay.open_one_shot_receipt(output, root, "d" * 40)
                    mocked_open.assert_not_called()
                    self.assertEqual(output.read_bytes(), payload)
                    self.assertEqual(output.stat().st_ino, inode)

    def test_output_path_and_cli_are_fixed(self) -> None:
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
            self.assertFalse(outside.exists())
            self.assertFalse(traversal.exists())

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

        self.assertEqual(tuple(inspect.signature(replay.run).parameters), ("implementation_commit",))
        source = inspect.getsource(replay.run)
        self.assertIn("open_one_shot_receipt(OUTPUT_PATH, ROOT", source)
        with patch.object(
            sys, "argv", ["rank1", "--implementation-commit", "e" * 40, "--output", "x"]
        ), patch.object(replay, "run") as mocked_run, patch(
            "sys.stderr", new=io.StringIO()
        ):
            with self.assertRaises(SystemExit):
                replay.main()
        mocked_run.assert_not_called()

    def test_o_excl_race_is_rejected_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "race.json"
            observed: list[int] = []

            def race(_path: str, flags: int, _mode: int) -> int:
                observed.append(flags)
                raise FileExistsError("race")

            with patch.object(replay.os, "open", side_effect=race):
                with self.assertRaises(replay.Rank1ReplayError):
                    replay.open_one_shot_receipt(output, root, "f" * 40)
            self.assertEqual(len(observed), 1)
            self.assertEqual(
                observed[0] & (os.O_CREAT | os.O_EXCL | os.O_RDWR),
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
            )
            self.assertEqual(observed[0] & os.O_TRUNC, 0)

    def test_label_loader_reads_only_five_frozen_members(self) -> None:
        fold = np.repeat(np.arange(5, dtype=np.uint8), 400)
        source = {
            "baseline_rank": np.zeros((2000, 10), dtype=np.uint8),
            "positive_index": np.full((2000, 10), -1, dtype=np.int16),
            "eligible_from": np.ones(2000, dtype=np.uint8),
            "outer_fold": fold,
            "inner_fold": fold,
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

        sentinel = object()
        archive = TrackingArchive()
        load_calls: list[tuple[object, bool]] = []

        def fake_np_load(handle: object, *, allow_pickle: bool) -> TrackingArchive:
            load_calls.append((handle, allow_pickle))
            return archive

        outcomes = replay.load_outcomes_from_open_handle(sentinel, fake_np_load)
        self.assertEqual(outcomes.positive_index.shape, (2000, 10))
        expected = tuple(name for name, _shape, _dtype in replay.LABEL_MEMBER_SPECS)
        self.assertEqual(
            expected,
            ("baseline_rank", "positive_index", "eligible_from", "outer_fold", "inner_fold"),
        )
        self.assertEqual(load_calls, [(sentinel, False)])
        self.assertEqual(tuple(archive.accesses), expected)
        self.assertNotIn("forbidden_extra", archive.accesses)
        self.assertEqual(archive.close_count, 1)
        for name in expected:
            copied = getattr(outcomes, name)
            self.assertFalse(copied.flags.writeable)
            self.assertFalse(np.shares_memory(copied, source[name]))

    def test_run_source_orders_receipt_before_label_open(self) -> None:
        source = inspect.getsource(replay.run)
        self.assertLess(
            source.index("open_one_shot_receipt"),
            source.index("LABEL_PATH.open"),
        )
        self.assertNotIn("frozen._load_inputs", source)
        target_free_source = inspect.getsource(replay._load_target_free_inputs)
        self.assertNotIn("_load_proxy_rows", target_free_source)
        self.assertNotIn("_eligible_turn", target_free_source)
        self.assertLess(
            source.index("_write_receipt_payload(receipt, result)"),
            source.index("final_written = True"),
        )
        self.assertLess(
            source.index("final_written = True"),
            source.index("_safe_close(receipt)"),
        )

    def test_amended_preregistration_and_comparator_contract_are_bound(self) -> None:
        bound = replay._validate_preregistration()
        self.assertEqual(bound["original_commit"], replay.ORIGINAL_PREREG_COMMIT)
        self.assertEqual(
            bound["protocol_amendment_commit"], replay.PREREG_COMMIT
        )
        self.assertEqual(
            replay.EXPECTED_INNER_FITS + replay.EXPECTED_OUTER_FITS,
            replay.EXPECTED_FIT_INVOCATIONS,
        )
        self.assertEqual(replay.EXPECTED_SELECTION_INVOCATIONS, 5)
        self.assertEqual(
            replay.EXPECTED_FOLD_QUANTILES,
            (25 / 64, 44 / 64, 26 / 64, 55 / 64, 32 / 64),
        )

    def test_git_gate_rejects_dirty_tree_and_accepts_exact_chain(self) -> None:
        implementation = "a" * 40
        prereg_relative = replay.PREREG_PATH.relative_to(replay.ROOT).as_posix()
        selector_relative = replay.SELECTOR_SOURCE_PATH.relative_to(replay.ROOT).as_posix()

        def clean_git(args: tuple[str, ...]) -> str:
            if args[:2] == ("status", "--porcelain=v1"):
                return ""
            if args == ("rev-parse", "HEAD"):
                return implementation
            if args == ("symbolic-ref", "--short", "HEAD"):
                return replay.BRANCH
            if args == ("remote", "get-url", replay.REMOTE):
                return replay.REMOTE_URL
            if args == ("rev-parse", replay.REMOTE_REF):
                return implementation
            if args == ("rev-parse", replay.PREREG_COMMIT + ":" + prereg_relative):
                return replay.PREREG_BLOB_OID
            if args == (
                "rev-parse",
                replay.ORIGINAL_PREREG_COMMIT + ":" + prereg_relative,
            ):
                return replay.ORIGINAL_PREREG_BLOB_OID
            if args == ("rev-parse", replay.BASE_COMMIT + ":" + selector_relative):
                return replay.SELECTOR_SOURCE_BLOB_OID
            if args == ("rev-list", "--parents", "-n", "1", implementation):
                return implementation + " " + replay.PREREG_COMMIT
            if args == ("rev-list", "--parents", "-n", "1", replay.PREREG_COMMIT):
                return replay.PREREG_COMMIT + " " + replay.ORIGINAL_PREREG_COMMIT
            if args == (
                "rev-list",
                "--parents",
                "-n",
                "1",
                replay.ORIGINAL_PREREG_COMMIT,
            ):
                return replay.ORIGINAL_PREREG_COMMIT + " " + replay.BASE_COMMIT
            if args[-1] == implementation and args[0] == "diff-tree":
                return "\n".join(sorted(replay.IMPLEMENTATION_PATHS))
            if args[-1] == replay.PREREG_COMMIT and args[0] == "diff-tree":
                return "\n".join(sorted(replay.PREREG_PATH_SET))
            if args[-1] == replay.ORIGINAL_PREREG_COMMIT and args[0] == "diff-tree":
                return "\n".join(sorted(replay.PREREG_PATH_SET))
            if args[0] == "rev-parse" and args[1].startswith(implementation + ":"):
                return "9" * 40
            raise AssertionError(args)

        with patch.object(replay, "_git", side_effect=clean_git):
            result = replay._validate_git_checkpoint(implementation)
        self.assertTrue(result["remote_equal"])

        def dirty_git(args: tuple[str, ...]) -> str:
            if args[:2] == ("status", "--porcelain=v1"):
                return "?? unexpected.py"
            return clean_git(args)

        with patch.object(replay, "_git", side_effect=dirty_git):
            with self.assertRaises(replay.Rank1ReplayError):
                replay._validate_git_checkpoint(implementation)

        def skipped_original_prereg(args: tuple[str, ...]) -> str:
            if args == (
                "rev-list",
                "--parents",
                "-n",
                "1",
                replay.PREREG_COMMIT,
            ):
                return replay.PREREG_COMMIT + " " + replay.BASE_COMMIT
            return clean_git(args)

        with patch.object(replay, "_git", side_effect=skipped_original_prereg):
            with self.assertRaises(replay.Rank1ReplayError):
                replay._validate_git_checkpoint(implementation)

        def extra_amendment_path(args: tuple[str, ...]) -> str:
            if args[-1] == replay.PREREG_COMMIT and args[0] == "diff-tree":
                return "\n".join((*sorted(replay.PREREG_PATH_SET), "unexpected.txt"))
            return clean_git(args)

        with patch.object(replay, "_git", side_effect=extra_amendment_path):
            with self.assertRaises(replay.Rank1ReplayError):
                replay._validate_git_checkpoint(implementation)

    def test_result_privacy_scan_rejects_identity_and_vectors(self) -> None:
        replay._result_privacy_scan({"safe": [1, 2, 3]})
        with self.assertRaises(replay.Rank1ReplayError):
            replay._result_privacy_scan({"target_asin": "B012345678"})
        with self.assertRaises(replay.Rank1ReplayError):
            replay._result_privacy_scan({"values": [0] * replay.SESSION_COUNT})


if __name__ == "__main__":
    unittest.main()
