from __future__ import annotations

import copy
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from scripts import evaluate_p12_action_oracle as runner
from scripts.p12_actions import (
    ACTION_IDS,
    ASK,
    CANDIDATE_RERANK,
    FROZEN_SEMANTIC_RERANK,
    KEEP_P11,
    KEEP_R08,
    RESULT_AWARE_REWRITE_RETRIEVE,
)


def _catalog_ids(count: int = 121) -> list[str]:
    return [f"B{index:09d}" for index in range(count)]


def _trace_row(
    turn: int,
    *,
    ordinal: int = 1,
    order: list[str] | None = None,
) -> dict:
    ranking = list(order or _catalog_ids()[:100])
    if len(ranking) != 100:
        raise AssertionError("synthetic R08 ranking must contain exactly 100 items")
    p11 = list(ranking)
    p11[0], p11[1] = p11[1], p11[0]
    c50_reverse = list(reversed(ranking[:50]))
    c50_rotate = ranking[1:50] + ranking[:1]
    return {
        "ordinal": ordinal,
        "turn": turn,
        "actions": {
            KEEP_R08: ranking[:10],
            KEEP_P11: p11[:10],
            CANDIDATE_RERANK: c50_reverse[:10],
            FROZEN_SEMANTIC_RERANK: c50_rotate[:10],
            RESULT_AWARE_REWRITE_RETRIEVE: ranking[:10],
            ASK: list(p11[:10]),
        },
        "candidate_pools": {
            "c20": ranking[:20],
            "c50": ranking[:50],
            "c100": ranking,
        },
    }


def _valid_trace() -> list[dict]:
    return [_trace_row(turn) for turn in range(1, 11)]


class SplitSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "proxy": {"allowed_splits": copy.deepcopy(runner.ALLOWED_SPLITS)}
        }

    def test_rejects_confirmation_and_unknown_before_any_io(self) -> None:
        forbidden_names = (
            "confirmation",
            "proxy_confirmation",
            "proxy_confirmation.sealed.jsonl",
            "../selection",
            "public",
            "",
        )
        with (
            mock.patch.object(
                runner, "_resolve_regular_file", side_effect=AssertionError("path touched")
            ) as resolve,
            mock.patch.object(
                runner, "_sha256", side_effect=AssertionError("hash touched")
            ) as sha256,
            mock.patch.object(
                Path, "open", side_effect=AssertionError("file opened")
            ) as opened,
            mock.patch.object(
                Path, "stat", side_effect=AssertionError("file stated")
            ) as stated,
        ):
            for split in forbidden_names:
                with self.subTest(split=split):
                    with self.assertRaises(runner.OracleRunError):
                        runner.select_split(self.config, split)
            resolve.assert_not_called()
            sha256.assert_not_called()
            opened.assert_not_called()
            stated.assert_not_called()

    def test_run_rejects_confirmation_before_config_or_process_operations(self) -> None:
        with (
            mock.patch.object(
                runner,
                "load_frozen_config",
                side_effect=AssertionError("config opened"),
            ) as config,
            mock.patch.object(
                runner.subprocess,
                "Popen",
                side_effect=AssertionError("process started"),
            ) as popen,
        ):
            with self.assertRaisesRegex(runner.OracleRunError, "not available"):
                runner.run("confirmation")
            config.assert_not_called()
            popen.assert_not_called()

    def test_allowed_split_requires_the_exact_frozen_declaration(self) -> None:
        self.config["proxy"]["allowed_splits"]["selection"]["path"] = (
            runner.SEALED_CONFIRMATION_PATH
        )
        with self.assertRaisesRegex(runner.OracleRunError, "declaration drifted"):
            runner.select_split(self.config, "selection")


class FrozenConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.valid = json.loads(
            (runner.REPO_ROOT / runner.DEFAULT_CONFIG).read_text(encoding="utf-8")
        )

    def _assert_forgery_rejected(self, mutate) -> None:
        forged = copy.deepcopy(self.valid)
        mutate(forged)
        with (
            mock.patch.object(runner, "_resolve_regular_file", return_value=Path("unused")),
            mock.patch.object(runner, "_load_json", return_value=forged),
        ):
            with self.assertRaisesRegex(runner.OracleRunError, "frozen .*config"):
                runner.load_frozen_config()

    def test_security_critical_paths_are_frozen(self) -> None:
        mutations = {
            "manifest": lambda value: value["proxy"].__setitem__(
                "manifest_path", "attacker/manifest.json"
            ),
            "catalog": lambda value: value["catalog"].__setitem__(
                "path", "attacker/catalog.jsonl"
            ),
            "worker": lambda value: value["runtime"].__setitem__(
                "worker_path", "attacker/worker.py"
            ),
            "output": lambda value: value["runtime"].__setitem__(
                "output_root", "../outside"
            ),
            "oracle eligible": lambda value: value["actions"].__setitem__(
                "oracle_eligible", [KEEP_P11]
            ),
            "gate": lambda value: value["go_no_go"].__setitem__(
                "oracle_hr_delta_min", -1.0
            ),
            "bootstrap": lambda value: value["evaluation"]["bootstrap"].__setitem__(
                "resamples", 1
            ),
            "sidecar": lambda value: value["p11"].__setitem__(
                "sidecar_sha256", "0" * 64
            ),
            "semantic": lambda value: value["semantic"].__setitem__(
                "full_catalog_search_allowed", True
            ),
            "parallelism": lambda value: value["runtime"].__setitem__(
                "parallel_workers", 1
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                self._assert_forgery_rejected(mutate)

    def test_output_directory_rejects_a_reparse_ancestor(self) -> None:
        real = runner._is_reparse_point

        def forged(path: Path) -> bool:
            return path.name == "experiments" or real(path)

        with mock.patch.object(runner, "_is_reparse_point", side_effect=forged):
            with self.assertRaisesRegex(runner.OracleRunError, "directory is unsafe"):
                runner._resolve_repository_directory(
                    runner.EXPECTED_OUTPUT_ROOT, create=True
                )


class BlindProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = {
            "purchase_frequency": "occasional",
            "average_prior_rating": None,
            "rating_style": "unknown",
            "preference_tags": ["simple", "durable"],
            "summary": "A concise target-blind summary.",
        }
        self.target = "B900000001"
        self.other_catalog_id = "B900000002"
        self.catalog = {self.target, self.other_catalog_id}

    def test_accepts_only_the_safe_profile_projection(self) -> None:
        projected = runner.project_profile(self.profile)
        self.assertEqual(set(projected), runner.SAFE_PROFILE_KEYS)
        self.assertIsNone(projected["average_prior_rating"])
        self.assertEqual(projected["rating_style"], "unknown")

        unsafe_profiles = []
        extra = {**self.profile, "scenario": "buying"}
        unsafe_profiles.append(extra)
        rating = {**self.profile, "average_prior_rating": 4.9}
        unsafe_profiles.append(rating)
        style = {**self.profile, "rating_style": "lenient"}
        unsafe_profiles.append(style)
        too_many_tags = {**self.profile, "preference_tags": ["a", "b", "c", "d"]}
        unsafe_profiles.append(too_many_tags)
        for unsafe in unsafe_profiles:
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(runner.OracleRunError):
                    runner.project_profile(unsafe)

    def test_rejects_label_shaped_keys_at_any_depth(self) -> None:
        payload = {
            "operation": "reset",
            "ordinal": 1,
            "user_profile": {**self.profile, "nested": {"ground_truth": "hidden"}},
        }
        with self.assertRaisesRegex(runner.OracleRunError, "label-shaped key"):
            runner.assert_blind_rpc(
                payload,
                current_target=self.target,
                sample_id="proxy-sample-1",
                catalog_ids=self.catalog,
            )

    def test_rejects_current_target_sample_and_any_catalog_asin_in_values(self) -> None:
        messages = {
            "target": f"please find {self.target}",
            "sample": "context proxy-sample-1 must stay hidden",
            "catalog": f"a different product {self.other_catalog_id}",
        }
        for label, message in messages.items():
            with self.subTest(label=label):
                with self.assertRaises(runner.OracleRunError):
                    runner.assert_blind_rpc(
                        {"operation": "respond", "user_message": message},
                        current_target=self.target,
                        sample_id="proxy-sample-1",
                        catalog_ids=self.catalog,
                    )

    def test_safe_rpc_payload_is_accepted(self) -> None:
        runner.assert_blind_rpc(
            {
                "operation": "respond",
                "ordinal": 1,
                "turn": 2,
                "top_k": 10,
                "user_message": "I prefer a blue casual jacket under $100.",
            },
            current_target=self.target,
            sample_id="proxy-sample-1",
            catalog_ids=self.catalog,
        )


class TraceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = set(_catalog_ids())

    def assert_invalid(self, rows: list[dict], message: str) -> None:
        with self.assertRaisesRegex(runner.OracleRunError, message):
            runner.validate_trace(rows, 1, self.ids)

    def test_accepts_exact_ten_turn_trace(self) -> None:
        grouped = runner.validate_trace(_valid_trace(), 1, self.ids)
        self.assertEqual(list(grouped), [1])
        self.assertEqual([row["turn"] for row in grouped[1]], list(range(1, 11)))

    def test_accepts_valid_short_candidate_pools(self) -> None:
        ranking = _catalog_ids()[:7]
        p11 = [ranking[1], ranking[0], *ranking[2:]]
        rows = [
            {
                "ordinal": 1,
                "turn": turn,
                "actions": {
                    KEEP_R08: list(ranking),
                    KEEP_P11: list(p11),
                    CANDIDATE_RERANK: list(reversed(ranking)),
                    FROZEN_SEMANTIC_RERANK: list(ranking),
                    RESULT_AWARE_REWRITE_RETRIEVE: list(ranking),
                    ASK: list(p11),
                },
                "candidate_pools": {
                    "c20": list(ranking),
                    "c50": list(ranking),
                    "c100": list(ranking),
                },
            }
            for turn in range(1, 11)
        ]

        grouped = runner.validate_trace(rows, 1, self.ids)

        self.assertEqual(grouped[1][0]["actions"][KEEP_R08], ranking)

    def test_requires_every_turn_exactly_once(self) -> None:
        self.assert_invalid(_valid_trace()[:-1], "turns are incomplete")
        duplicated = _valid_trace()
        duplicated[-1]["turn"] = 9
        self.assert_invalid(duplicated, "turns are incomplete")

    def test_ask_must_exactly_equal_keep_p11(self) -> None:
        rows = _valid_trace()
        rows[0]["actions"][ASK] = list(rows[0]["actions"][KEEP_R08])
        self.assert_invalid(rows, "ASK trace must reuse")

    def test_p11_must_preserve_top10_set(self) -> None:
        top10 = _valid_trace()
        top10[0]["actions"][KEEP_P11][0] = top10[0]["candidate_pools"]["c100"][11]
        top10[0]["actions"][ASK] = list(top10[0]["actions"][KEEP_P11])
        self.assert_invalid(top10, "Top10 membership")

    def test_candidate_pools_are_exact_nested_prefixes(self) -> None:
        rows = _valid_trace()
        rows[0]["candidate_pools"]["c20"][0], rows[0]["candidate_pools"]["c20"][1] = (
            rows[0]["candidate_pools"]["c20"][1],
            rows[0]["candidate_pools"]["c20"][0],
        )
        self.assert_invalid(rows, "C10 is not")

        rows = _valid_trace()
        rows[0]["candidate_pools"]["c50"][20], rows[0]["candidate_pools"]["c50"][21] = (
            rows[0]["candidate_pools"]["c50"][21],
            rows[0]["candidate_pools"]["c50"][20],
        )
        self.assert_invalid(rows, "C50 is not a prefix")

    def test_rejects_malformed_action_set_and_rankings(self) -> None:
        missing = _valid_trace()
        del missing[0]["actions"][RESULT_AWARE_REWRITE_RETRIEVE]
        self.assert_invalid(missing, "action set mismatch")

        duplicate = _valid_trace()
        duplicate[0]["actions"][KEEP_R08][1] = duplicate[0]["actions"][KEEP_R08][0]
        self.assert_invalid(duplicate, "invalid worker ranking")

        outsider = _valid_trace()
        outsider[0]["actions"][RESULT_AWARE_REWRITE_RETRIEVE][0] = "Z999999999"
        self.assert_invalid(outsider, "invalid worker ranking")


class PostCloseLabelJoinTests(unittest.TestCase):
    def test_override_eligibility_excludes_pre_switch_hits_and_controls_recall(self) -> None:
        ids = _catalog_ids()
        target = ids[110]
        base = ids[:100]
        pre_switch = [target] + base[:99]
        post_switch = base[:14] + [target] + base[14:99]

        rows = []
        for turn in range(1, 11):
            order = pre_switch if turn == 1 else (base if turn == 2 else post_switch)
            row = _trace_row(turn, order=order)
            if turn >= 3:
                c50 = row["candidate_pools"]["c50"]
                row["actions"][CANDIDATE_RERANK] = [target] + [
                    value for value in c50 if value != target
                ][:9]
            rows.append(row)

        trace = runner.validate_trace(rows, 1, set(ids))
        samples = [{"ground_truth": {"parent_asin": target}}]
        ledger = [
            {
                "target_id": target,
                "eligible_from_turn": 3,
                "scenario": "intent_override",
                "taxonomy": "apparel",
                "difficulty": "medium",
                "popularity": "tail",
                "source_weight": 1.0,
            }
        ]
        joined, recalls = runner.join_labels_after_close(samples, ledger, trace)

        self.assertFalse(joined[0]["actions"][KEEP_R08]["hit"])
        self.assertFalse(joined[0]["actions"][KEEP_P11]["hit"])
        self.assertEqual(
            joined[0]["actions"][CANDIDATE_RERANK],
            {
                "hit": True,
                "first_hit_turn": 3,
                "first_rank": 1,
                "best_rank": 1,
                "reciprocal_rank": 1.0,
            },
        )
        self.assertEqual(
            recalls,
            {
                "recall_at_10": 0.0,
                "recall_at_20": 1.0,
                "recall_at_50": 1.0,
                "recall_at_100": 1.0,
            },
        )
        self.assertEqual(
            runner._eligible_from_turn(
                {"scenario_type": "intent_override", "behavior": {"override": {"turn": 3}}}
            ),
            3,
        )
        aggregate = runner.aggregate_action_oracle(
            joined,
            action_ids=list(ACTION_IDS),
            oracle_eligible_actions=[KEEP_P11, CANDIDATE_RERANK],
            bootstrap_resamples=10,
            bootstrap_seed=1,
        )
        self.assertEqual(
            aggregate["actions"][CANDIDATE_RERANK]["metrics"]
            ["row_uniform_official"]["hit_rate_at_10"],
            1.0,
        )

    def test_aggregate_discards_join_only_identifiers(self) -> None:
        target = "B999999999"
        miss = {
            "hit": False,
            "first_hit_turn": None,
            "first_rank": None,
            "best_rank": None,
            "reciprocal_rank": 0.0,
        }
        record = {
            "ordinal": 1,
            "target_id": target,
            "scenario": "buying",
            "taxonomy": "apparel",
            "difficulty": "easy",
            "popularity": "head",
            "source_weight": 1.0,
            "actions": {action: dict(miss) for action in ACTION_IDS},
        }
        aggregate = runner.aggregate_action_oracle(
            [record],
            action_ids=list(ACTION_IDS),
            oracle_eligible_actions=[action for action in ACTION_IDS if action != ASK],
            baseline_action=KEEP_P11,
            bootstrap_resamples=10,
            bootstrap_seed=7,
        )
        serialized = json.dumps(aggregate, sort_keys=True)
        self.assertNotIn(target, serialized)
        self.assertNotIn("target_id", serialized)
        self.assertNotIn("sample_id", serialized)
        self.assertNotIn("parent_asin", serialized)

        runner.assert_identifier_free_artifact(aggregate, {target})


class GoNoGoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (runner.REPO_ROOT / runner.DEFAULT_CONFIG).read_text(encoding="utf-8")
        )
        baseline_metrics = {"recommended_technical_score": 0.7}
        candidate_relative = {
            "net_rescues": 2,
            "positive_net_scenario_span": 2,
            "positive_net_taxonomy_span": 2,
        }
        self.aggregate = {
            "actions": {
                KEEP_P11: {
                    "metrics": {"row_uniform_official": baseline_metrics},
                    "relative_to_baseline": {},
                },
                CANDIDATE_RERANK: {"relative_to_baseline": dict(candidate_relative)},
                FROZEN_SEMANTIC_RERANK: {"relative_to_baseline": dict(candidate_relative)},
                RESULT_AWARE_REWRITE_RETRIEVE: {
                    "relative_to_baseline": dict(candidate_relative)
                },
            },
            "oracle": {
                "metrics": {
                    "row_uniform_official": {"recommended_technical_score": 0.71}
                },
                "relative_to_baseline": {"hit_rate_delta": 0.02},
                "paired_utility_bootstrap_ci": {"lower": 0.001},
            },
        }
        self.worker = {
            "network_attempt_count": 0,
            "semantic_failure_count": 0,
            "rewrite_failure_count": 0,
            "full_catalog_search_calls": 0,
            "p11_invariant_failure_count": 0,
        }

    def test_selection_go_requires_signal_ci_stable_action_and_worker_integrity(self) -> None:
        decision = runner.build_go_no_go(
            self.aggregate, self.worker, self.config, decision_eligible=True
        )

        self.assertEqual(decision["status"], "GO")
        self.assertTrue(decision["cage_r10_implementation_authorized"])
        self.assertEqual(
            decision["stable_deployable_actions"],
            [
                CANDIDATE_RERANK,
                FROZEN_SEMANTIC_RERANK,
                RESULT_AWARE_REWRITE_RETRIEVE,
            ],
        )

    def test_integrity_failure_is_no_go_and_nonselection_never_authorizes(self) -> None:
        failed_worker = {**self.worker, "network_attempt_count": 1}
        failed = runner.build_go_no_go(
            self.aggregate, failed_worker, self.config, decision_eligible=True
        )
        exploratory = runner.build_go_no_go(
            self.aggregate, self.worker, self.config, decision_eligible=False
        )

        self.assertEqual(failed["status"], "NO_GO")
        self.assertFalse(failed["worker_integrity_passed"])
        self.assertEqual(exploratory["status"], "NON_DECISION_SIGNAL_ONLY")
        self.assertFalse(exploratory["cage_r10_implementation_authorized"])

    def test_artifact_guard_rejects_nested_identifier_keys_and_asin_values(self) -> None:
        target = "B999999999"
        with self.assertRaisesRegex(runner.OracleRunError, "identifier keys"):
            runner.assert_identifier_free_artifact(
                {"worker": {"debug": {"sample_id": "opaque"}}}, {target}
            )
        with self.assertRaisesRegex(runner.OracleRunError, "catalog identifier"):
            runner.assert_identifier_free_artifact(
                {"worker": {"last_value": f"observed {target}"}}, {target}
            )


class WorkerProtocolHelperTests(unittest.TestCase):
    def _bare_client(self, item: object) -> runner.WorkerClient:
        client = runner.WorkerClient.__new__(runner.WorkerClient)
        client._queue = queue.Queue()
        client._queue.put(item)
        client._timeout = 0.01
        client._stderr = []
        return client

    def test_receive_requires_exact_json_object_transport(self) -> None:
        self.assertEqual(
            self._bare_client('{"kind":"ready","nonce":"n"}\n')._receive(),
            {"kind": "ready", "nonce": "n"},
        )
        with self.assertRaisesRegex(runner.OracleRunError, "non-JSON"):
            self._bare_client("diagnostic noise\n")._receive()
        with self.assertRaisesRegex(runner.OracleRunError, "not an object"):
            self._bare_client("[]\n")._receive()


def _worker_summary(scale: int = 1) -> dict:
    return {
        "trajectory": {
            "fixed_turns": 10,
            "top_k": 10,
            "completed_sessions": scale,
            "respond_count": scale * 10,
        },
        "actions": {
            "ids": list(ACTION_IDS),
            "result_aware_computation_count": scale * 10,
        },
        "semantic": {
            "mode": "candidate_only_c50",
            "query_count": scale * 10,
            "candidate_matrix_rows_read": scale * 500,
            "maximum_candidate_rows_read": 40 + scale,
            "full_catalog_search_calls": 0,
            "failure_count": 0,
        },
        "memory": {"peak_rss_bytes": scale * 1000},
        "network_attempt_count": 0,
        "semantic_failure_count": 0,
        "rewrite_failure_count": 0,
        "full_catalog_search_calls": 0,
        "p11_invariant_failure_count": 0,
    }


class ParallelShardTests(unittest.TestCase):
    def test_contiguous_balanced_chunks_use_at_most_four_workers(self) -> None:
        samples = [{"row": index} for index in range(10)]
        chunks = runner.balanced_contiguous_chunks(samples, 4)
        self.assertEqual([start for _, start, _ in chunks], [1, 4, 7, 9])
        self.assertEqual([len(chunk) for _, _, chunk in chunks], [3, 3, 2, 2])
        self.assertEqual(
            [row["row"] for _, _, chunk in chunks for row in chunk], list(range(10))
        )
        self.assertEqual(len(runner.balanced_contiguous_chunks(samples[:2], 4)), 2)

    def test_fake_parallel_workers_receive_local_ordinals_and_unique_nonces(self) -> None:
        instances = []

        class FakeClient:
            def __init__(self, config, trace_path, nonce):
                self.trace_path = trace_path
                self.nonce = nonce
                self.local_ordinals = []
                self.aborted = False
                instances.append(self)

            def finalize(self):
                count = len(self.local_ordinals)
                return runner.WorkerReceipt(
                    "0" * 64, count * 10, _worker_summary(count)
                )

            def abort(self):
                self.aborted = True

        def fake_replay(worker, samples, *args):
            worker.local_ordinals = list(range(1, len(samples) + 1))
            return [{"local": ordinal} for ordinal in worker.local_ordinals]

        samples = [{"row": index} for index in range(9)]
        paths = [Path(f"fake-{index}.jsonl") for index in range(4)]
        config = {"runtime": {"parallel_workers": 4}}
        with (
            mock.patch.object(runner, "WorkerClient", FakeClient),
            mock.patch.object(runner, "replay_blind_trajectories", fake_replay),
        ):
            shards = runner._run_blind_shards(
                config, samples, {}, {}, set(), paths, "base-nonce"
            )

        self.assertEqual([shard.global_start for shard in shards], [1, 4, 6, 8])
        self.assertEqual([len(shard.samples) for shard in shards], [3, 2, 2, 2])
        self.assertEqual(
            sorted(client.local_ordinals for client in instances),
            [[1, 2], [1, 2], [1, 2], [1, 2, 3]],
        )
        self.assertEqual(len({client.nonce for client in instances}), 4)
        self.assertTrue(all(client.aborted for client in instances))

    def test_peer_failure_aborts_every_fake_worker(self) -> None:
        instances = []
        barrier = threading.Barrier(4)

        class FakeClient:
            def __init__(self, config, trace_path, nonce):
                self.aborted = False
                instances.append(self)

            def abort(self):
                self.aborted = True

        def failing_replay(worker, samples, *args):
            abort_event = args[-1]
            barrier.wait(timeout=2)
            if samples[0]["fail"]:
                raise runner.OracleRunError("synthetic shard failure")
            abort_event.wait(timeout=2)
            raise runner.OracleRunError("peer stopped")

        samples = [{"fail": index == 0} for index in range(4)]
        with (
            mock.patch.object(runner, "WorkerClient", FakeClient),
            mock.patch.object(runner, "replay_blind_trajectories", failing_replay),
        ):
            with self.assertRaises(runner.OracleRunError):
                runner._run_blind_shards(
                    {"runtime": {"parallel_workers": 4}},
                    samples,
                    {},
                    {},
                    set(),
                    [Path(f"fail-{index}.jsonl") for index in range(4)],
                    "nonce",
                )
        self.assertEqual(len(instances), 4)
        self.assertTrue(all(client.aborted for client in instances))

    def test_sharded_join_matches_single_worker_metrics_and_combined_digest(self) -> None:
        ids = set(_catalog_ids())
        samples = [{"ground_truth": {"parent_asin": _catalog_ids()[0]}} for _ in range(5)]
        ledger = [
            {
                "target_id": _catalog_ids()[0],
                "eligible_from_turn": 1,
                "scenario": "buying",
                "taxonomy": "apparel",
                "difficulty": "easy",
                "popularity": "head",
                "source_weight": 1.0,
            }
            for _ in samples
        ]

        def materialize(path: Path, count: int) -> runner.WorkerReceipt:
            rows = [
                _trace_row(turn, ordinal=ordinal)
                for ordinal in range(1, count + 1)
                for turn in range(1, 11)
            ]
            path.write_bytes(b"".join(runner._canonical_bytes(row) for row in rows))
            return runner.WorkerReceipt(runner._sha256(path), len(rows), _worker_summary(count))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            single_path = root / "single.jsonl"
            single = [
                runner.ShardResult(
                    0, 1, samples, single_path, materialize(single_path, 5), ledger
                )
            ]
            first_path, second_path = root / "first.jsonl", root / "second.jsonl"
            parallel = [
                runner.ShardResult(
                    0, 1, samples[:3], first_path, materialize(first_path, 3), ledger[:3]
                ),
                runner.ShardResult(
                    1, 4, samples[3:], second_path, materialize(second_path, 2), ledger[3:]
                ),
            ]
            single_joined, single_recall, single_digest = runner._join_closed_shards(single, ids)
            parallel_joined, parallel_recall, parallel_digest = (
                runner._join_closed_shards(parallel, ids)
            )

        self.assertEqual(parallel_joined, single_joined)
        self.assertEqual(parallel_recall, single_recall)
        self.assertEqual(parallel_digest, single_digest)
        self.assertEqual(parallel_recall["cutoffs"]["10"], {"hit_count": 5, "rate": 1.0})
        kwargs = {
            "action_ids": list(ACTION_IDS),
            "oracle_eligible_actions": [action for action in ACTION_IDS if action != ASK],
            "baseline_action": KEEP_P11,
            "bootstrap_resamples": 10,
            "bootstrap_seed": 5,
        }
        self.assertEqual(
            runner.aggregate_action_oracle(parallel_joined, **kwargs),
            runner.aggregate_action_oracle(single_joined, **kwargs),
        )

    def test_worker_summary_sums_gate_counts_and_totals_but_maxes_peaks(self) -> None:
        shards = [
            runner.ShardResult(
                index,
                index + 1,
                [{}] * scale,
                Path("unused"),
                runner.WorkerReceipt("0" * 64, scale * 10, _worker_summary(scale)),
                [],
            )
            for index, scale in enumerate((1, 2, 3))
        ]
        merged = runner._merge_worker_summaries(shards)
        self.assertEqual(merged["parallel_workers"], 3)
        self.assertEqual(merged["trajectory"]["completed_sessions"], 6)
        self.assertEqual(merged["actions"]["result_aware_computation_count"], 60)
        self.assertEqual(merged["semantic"]["candidate_matrix_rows_read"], 3000)
        self.assertEqual(merged["semantic"]["maximum_candidate_rows_read"], 43)
        self.assertTrue(merged["memory"]["all_worker_peak_rss_available"])
        self.assertEqual(merged["memory"]["peak_rss_bytes_max"], 3000)
        self.assertEqual(
            merged["memory"]["sum_of_worker_peak_rss_bytes_upper_bound"], 6000
        )
        self.assertFalse(merged["memory"]["parent_process_rss_included"])
        for key in (
            "network_attempt_count",
            "semantic_failure_count",
            "rewrite_failure_count",
            "full_catalog_search_calls",
            "p11_invariant_failure_count",
        ):
            self.assertEqual(merged[key], 0)
        self.assertEqual(len(merged["per_shard"]), 3)

    def test_worker_summary_reports_unavailable_rss_without_zero_substitution(self) -> None:
        summary = _worker_summary()
        summary["memory"] = {"peak_rss_bytes": None, "backend": "unavailable"}
        shard = runner.ShardResult(
            0,
            1,
            [{}],
            Path("unused"),
            runner.WorkerReceipt("0" * 64, 10, summary),
            [],
        )
        merged = runner._merge_worker_summaries([shard])
        self.assertFalse(merged["memory"]["all_worker_peak_rss_available"])
        self.assertIsNone(merged["memory"]["peak_rss_bytes_max"])
        self.assertIsNone(
            merged["memory"]["sum_of_worker_peak_rss_bytes_upper_bound"]
        )


if __name__ == "__main__":
    unittest.main()
