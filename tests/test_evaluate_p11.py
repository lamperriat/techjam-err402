from __future__ import annotations

import copy
import io
import importlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_p11
from scripts.official_metric_bridge import rebuild_official_metrics


def _scenario_metrics(sessions: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in sessions:
        grouped[str(row["scenario_type"])].append(row)
    output = {}
    for scenario, rows in sorted(grouped.items()):
        metrics = rebuild_official_metrics(rows)
        output[scenario] = {
            key: metrics[key]
            for key in ("sample_count", "hit_rate_at_10", "mrr", "mttc")
        }
    return output


def _audit_categories(stdout: str) -> list[str]:
    return [
        str(message["category"])
        for message in (json.loads(line) for line in stdout.splitlines())
        if message.get("kind") == "audit-denied"
    ]


class MockRoleRunner:
    def __init__(self, *, improve_primary: bool = True, duplicate_process: bool = False) -> None:
        self.improve_primary = improve_primary
        self.duplicate_process = duplicate_process
        self.calls: list[tuple[str, str]] = []
        self.next_process = 1000

    def __call__(
        self,
        role: str,
        samples: list[dict[str, object]],
        _catalog_ids: set[str],
        _categories: dict[str, list[str]],
        _products: dict[str, dict[str, object]],
        **_runtime: object,
    ) -> dict[str, object]:
        sample_prefix = str(samples[0]["sample_id"])[0]
        split = {"p": "primary", "u": "uniform_tail", "c": "confirmation"}[
            sample_prefix
        ]
        self.calls.append((split, role))
        sessions = []
        for index, sample in enumerate(samples):
            before = 9 if index < 3 else 8
            after = 8 if index < 3 else 7
            rank = before
            if role == evaluate_p11.ACTIVE_ID and split != "uniform_tail":
                if split != "primary" or self.improve_primary:
                    rank = after
            if split == "uniform_tail":
                rank = 7
            sessions.append(
                {
                    "sample_id": sample["sample_id"],
                    "scenario_type": sample["scenario_type"],
                    "hit": True,
                    "first_hit_turn": 1,
                    "best_rank": rank,
                    "reciprocal_rank": 1.0 / rank,
                }
            )
        metrics = rebuild_official_metrics(sessions)
        functional_hash = evaluate_p11._stable_sha256(
            {"split": split, "sessions": sessions}
        )
        response_hash = functional_hash
        if role in {
            evaluate_p11.BASELINE_ID,
            evaluate_p11.CONTROL_ID,
            evaluate_p11.SHADOW_ID,
        }:
            baseline_sessions = copy.deepcopy(sessions)
            functional_hash = evaluate_p11._stable_sha256(
                {"split": split, "sessions": baseline_sessions}
            )
            response_hash = functional_hash
        if role == evaluate_p11.BASELINE_ID:
            configuration = {
                "retrieval_mode": "coverage",
                "rerank_mode": "off",
                "question_policy": "fast",
                "sidecar_opened": False,
            }
            schema = "p11.served-reference.v1"
        else:
            configuration = {
                "retrieval_mode": "coverage",
                "rerank_mode": "off",
                "question_policy": "fast",
                "top10_membership_preserved": True,
                "tail_preserved": True,
                "target_blind": True,
                "label_free": True,
                "feature_schema_version": evaluate_p11.FEATURE_SCHEMA_VERSION,
                "scorer_version": evaluate_p11.SCORER_VERSION,
                "feature_registry_sha256": evaluate_p11.FEATURE_REGISTRY_SHA256,
                "feature_semantics_sha256": evaluate_p11.FEATURE_SEMANTICS_SHA256,
                "sidecar_opened": role
                in {evaluate_p11.SHADOW_ID, evaluate_p11.ACTIVE_ID},
            }
            schema = "p11.top10-lab.v1"
        process = 1000 if self.duplicate_process else self.next_process
        self.next_process += 1
        return {
            "role": role,
            "sessions": sessions,
            "metrics": metrics,
            "scenario_metrics": _scenario_metrics(sessions),
            "resources": {
                "wall_seconds": 110.0 if role == evaluate_p11.ACTIVE_ID else 100.0,
                "p95_latency_ms": 110.0 if role == evaluate_p11.ACTIVE_ID else 100.0,
                "peak_rss_bytes": 105 if role == evaluate_p11.ACTIVE_ID else 100,
            },
            "functional_result_sha256": functional_hash,
            "response_trace_sha256": response_hash,
            "capture": {
                "schema_version": schema,
                "role": role,
                "configuration": configuration,
                "stats": {
                    "turns": len(sessions),
                    "exception_count": 0,
                    "top10_membership_violation_count": 0,
                    "tail_change_count": 0,
                },
                "integrity_errors": [],
                "hashes": {"audit_sha256": "a" * 64},
                "function_hashes": {"scorer": "b" * 64},
            },
            "audit": {
                "contract_error_count": 0,
                "contract_errors_sha256": "c" * 64,
                "integrity_error_count": 0,
                "integrity_errors_sha256": "d" * 64,
                "network_attempt_count": 0,
                "preimport_denied_attempt_count": 0,
                "preimport_denied_attempt_counts": {
                    "lifecycle": 0,
                    "network": 0,
                    "process": 0,
                    "read": 0,
                    "sqlite": 0,
                },
                "preimport_audit_state_missing_count": 0,
                "generic_exception_count": 0,
                "generic_exception_classes_sha256": "e" * 64,
                "reported_token_total": 0,
                "capture_exception_count": 0,
                "top10_membership_violation_count": 0,
                "tail_change_count": 0,
            },
            "_process": {
                "pid": process,
                "nonce": f"{process:032x}",
                "separate_process": True,
                "staged_runtime": True,
                "pre_import_read_process_boundary": True,
                "python_audit_pre_import_network_fail_closed": True,
                "preimport_denied_attempt_accounting": True,
                "preimport_supervisor_event_stream": True,
                "preimport_audit_record_loaded_after_clean_exit": True,
                "agent_close_and_candidate_atexit_audited": True,
                "peak_rss_measured_after_candidate_atexit": True,
                "sqlite_memory_allowed_all_roles": True,
                "sqlite_attach_detach_extension_authorizer": True,
                "immutable_sidecar_sqlite_role_scoped": True,
                "sidecar_read_allowed": role
                in {evaluate_p11.SHADOW_ID, evaluate_p11.ACTIVE_ID},
            },
        }


class P11RunnerFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.spec = root / "experiment.json"
        self.prereg = root / "unused-fixture-prereg.json"
        self.output = root / "unused-fixture-output.json"
        self.attempt_marker = root / "unused-fixture-attempt.json"
        self.confirmation_marker = root / "unused-fixture-confirmation.json"
        self.protocol = root / "corpus-protocol.json"
        self.catalog = root / "catalog.jsonl"
        self.sidecar = root / "features.sqlite"
        self.corpora = {
            "primary": root / "primary.jsonl",
            "uniform_tail": root / "uniform.jsonl",
            "confirmation": root / "confirmation.jsonl",
        }
        self.catalog.write_text("{}\n", encoding="utf-8")
        self.sidecar.write_bytes(b"catalog-only-sidecar")
        self.targets: list[str] = []
        split_specs = {}
        for split, prefix, offset in (
            ("primary", "p", 1),
            ("uniform_tail", "u", 11),
            ("confirmation", "c", 21),
        ):
            rows = []
            for index in range(10):
                target = f"A{offset + index:09d}"
                self.targets.append(target)
                rows.append(
                    {
                        "ground_truth": {"parent_asin": target},
                        "sample_id": f"{prefix}-{index:02d}",
                        "scenario_type": "buying"
                        if index % 2 == 0
                        else "browsing",
                        "user_profile": {"summary": "fixture"},
                    }
                )
            payload = "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            )
            self.corpora[split].write_text(payload, encoding="utf-8", newline="\n")
            split_specs[split] = {
                "count": 10,
                "expected_samples_sha256": evaluate_p11._sha256_file(
                    self.corpora[split]
                ),
                "filename": self.corpora[split].name,
                "sample_id_prefix": f"{prefix}-",
                "scenario_counts": {"browsing": 5, "buying": 5},
            }
        protocol = {
            "schema_version": evaluate_p11.CORPUS_PROTOCOL_SCHEMA_VERSION,
            "catalog": {
                "count": len(self.targets),
                "path": self.catalog.name,
                "sha256": evaluate_p11._sha256_file(self.catalog),
            },
            "splits": split_specs,
        }
        self.protocol.write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.spec.write_bytes(evaluate_p11.DEFAULT_SPEC.read_bytes())

    def catalog_index(self, _path: Path) -> tuple[set[str], dict, dict]:
        return set(self.targets), {}, {target: {} for target in self.targets}

    def run(self, runner: MockRoleRunner) -> dict[str, object]:
        with patch("scripts.evaluate_p11.catalog_index", side_effect=self.catalog_index):
            return evaluate_p11.run_evaluation(
                prereg_lock_path=self.prereg,
                spec_path=self.spec,
                corpus_protocol_path=self.protocol,
                catalog_path=self.catalog,
                sidecar_path=self.sidecar,
                corpus_paths=self.corpora,
                output_path=self.output,
                attempt_marker_path=self.attempt_marker,
                confirmation_marker_path=self.confirmation_marker,
                role_runner=runner,
                formal=False,
            )


class EvaluateP11Tests(unittest.TestCase):
    def test_disjointness_proof_requires_exact_opened_new_and_cross_maps(self) -> None:
        opened_specs = {
            name: {
                "rows": 200,
                "canonical_samples_sha256": digest,
            }
            for name, digest in evaluate_p11.EXPECTED_OPENED_CORPUS_SHA256.items()
        }
        protocol = {
            "opened_corpora": opened_specs,
            "opened_target_union_count": 1800,
            "splits": {
                name: {"count": 1} for name in evaluate_p11.LOCK_ALL_SPLITS
            },
        }
        metadata = {
            "new_pairwise_target_overlaps": {
                f"{left}__{right}": 0
                for left, right in combinations(
                    sorted(evaluate_p11.LOCK_ALL_SPLITS), 2
                )
            },
            "new_target_union_count": len(evaluate_p11.LOCK_ALL_SPLITS),
            "opened_registry": {
                "corpora": {
                    name: {
                        "rows": 200,
                        "unique_targets": 200,
                        "canonical_samples_sha256": digest,
                    }
                    for name, digest in (
                        evaluate_p11.EXPECTED_OPENED_CORPUS_SHA256.items()
                    )
                },
                "pairwise_target_overlaps": {
                    f"{left}__{right}": 0
                    for left, right in combinations(sorted(opened_specs), 2)
                },
                "target_union_count": 1800,
            },
            "opened_vs_new_target_overlaps": {
                name: 0 for name in evaluate_p11.LOCK_ALL_SPLITS
            },
        }
        evaluate_p11._validate_disjointness_proof(protocol, metadata)

        mutations = {
            "missing_opened_protocol_entry": lambda p, _m: p[
                "opened_corpora"
            ].pop(next(iter(p["opened_corpora"]))),
            "empty_new_pairwise": lambda _p, m: m.update(
                {"new_pairwise_target_overlaps": {}}
            ),
            "missing_opened_observation": lambda _p, m: m["opened_registry"][
                "corpora"
            ].pop(next(iter(m["opened_registry"]["corpora"]))),
            "empty_opened_pairwise": lambda _p, m: m["opened_registry"].update(
                {"pairwise_target_overlaps": {}}
            ),
            "missing_cross_map": lambda _p, m: m.pop(
                "opened_vs_new_target_overlaps"
            ),
            "nonzero_cross_overlap": lambda _p, m: m[
                "opened_vs_new_target_overlaps"
            ].update({"primary": 1}),
        }
        for name, mutator in mutations.items():
            with self.subTest(name=name):
                changed_protocol = copy.deepcopy(protocol)
                changed_metadata = copy.deepcopy(metadata)
                mutator(changed_protocol, changed_metadata)
                with self.assertRaises(evaluate_p11.P11RunnerError):
                    evaluate_p11._validate_disjointness_proof(
                        changed_protocol, changed_metadata
                    )

    def test_full_mock_protocol_passes_nonformal_but_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = P11RunnerFixture(Path(directory))
            runner = MockRoleRunner()

            artifact = fixture.run(runner)

        self.assertEqual(artifact["decision"], "nonformal_candidate_pass")
        self.assertTrue(artifact["promotion_gate"]["passed"])
        self.assertFalse(artifact["formal_evaluation"])
        self.assertFalse(artifact["promotion_eligible"])
        self.assertEqual(artifact["winner_id"], evaluate_p11.CONTROL_ID)
        self.assertFalse(artifact["public_evaluation_run"])
        self.assertEqual(
            artifact["execution"]["global_deadline"],
            {
                "applied": False,
                "reason": "formal-only policy; nonformal fixtures are excluded",
            },
        )
        self.assertEqual(
            artifact["execution"]["observed_order"],
            evaluate_p11.EXPECTED_EXECUTION_ORDER,
        )
        self.assertTrue(
            artifact["execution"]["confirmation"]["semantic_parse_executed"]
        )
        for split in ("primary", "uniform_tail", "confirmation"):
            scan = artifact["execution"][split]["source_identifier_scan"]
            self.assertEqual(scan["match_count"], 0)
            self.assertTrue(scan["passed"])
        self.assertEqual(artifact["provenance"]["fresh_subprocess_count"], 18)
        for split in ("primary", "confirmation"):
            repeat = artifact["execution"][split]["repeat"]
            self.assertEqual(set(repeat["runs"]), set(evaluate_p11.REPEAT_ROLES))
            self.assertTrue(repeat["split_gate"]["passed"])
        payload = json.dumps(artifact, sort_keys=True)
        for target in fixture.targets:
            self.assertNotIn(target, payload)
        self.assertNotIn('"sessions"', payload)
        self.assertNotIn('"sample_id"', payload)

    def test_confirmation_is_hashed_but_not_parsed_when_primary_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = P11RunnerFixture(Path(directory))
            runner = MockRoleRunner(improve_primary=False)
            confirmation_sha256 = evaluate_p11._sha256_file(
                fixture.corpora["confirmation"]
            )
            parsed: list[Path] = []
            real_load = importlib.import_module(
                "evaluator.local_evaluator"
            ).load_jsonl

            def recording_load(path: Path) -> list[dict]:
                parsed.append(Path(path))
                return real_load(path)

            with patch(
                "scripts.evaluate_p11.catalog_index", side_effect=fixture.catalog_index
            ), patch("scripts.evaluate_p11.load_jsonl", side_effect=recording_load):
                artifact = evaluate_p11.run_evaluation(
                    prereg_lock_path=fixture.prereg,
                    spec_path=fixture.spec,
                    corpus_protocol_path=fixture.protocol,
                    catalog_path=fixture.catalog,
                    sidecar_path=fixture.sidecar,
                    corpus_paths=fixture.corpora,
                    output_path=fixture.output,
                    attempt_marker_path=fixture.attempt_marker,
                    confirmation_marker_path=fixture.confirmation_marker,
                    role_runner=runner,
                    formal=False,
                )

        self.assertEqual(artifact["decision"], "nonformal_retain")
        self.assertEqual(parsed, [fixture.corpora["primary"]])
        self.assertFalse(
            artifact["execution"]["confirmation"]["semantic_parse_executed"]
        )
        self.assertEqual(
            artifact["execution"]["confirmation"]["source_identifier_scan"],
            {"executed": False, "reason": "confirmation was not opened"},
        )
        self.assertEqual(
            artifact["inputs"]["identities"]["data"]["confirmation"]["sha256"],
            confirmation_sha256,
        )

    def test_duplicate_process_identity_stops_before_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = P11RunnerFixture(Path(directory))
            artifact = fixture.run(MockRoleRunner(duplicate_process=True))

        self.assertEqual(artifact["decision"], "nonformal_retain")
        self.assertFalse(
            artifact["execution"]["primary"]["initial"]["boundary_checks"][
                "all_initial_processes_fresh"
            ]
        )
        self.assertFalse(
            artifact["execution"]["confirmation"]["semantic_parse_executed"]
        )

    def test_source_config_or_data_mutation_is_detected_before_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = P11RunnerFixture(Path(directory))
            runner = MockRoleRunner()
            original = runner.__call__
            calls = 0

            def mutating_runner(*args: object, **kwargs: object) -> dict[str, object]:
                nonlocal calls
                calls += 1
                result = original(*args, **kwargs)
                if calls == 18:
                    fixture.sidecar.write_bytes(b"changed-sidecar")
                return result

            with patch(
                "scripts.evaluate_p11.catalog_index", side_effect=fixture.catalog_index
            ):
                with self.assertRaisesRegex(
                    evaluate_p11.P11RunnerError, "changed during evaluation"
                ):
                    evaluate_p11.run_evaluation(
                        prereg_lock_path=fixture.prereg,
                        spec_path=fixture.spec,
                        corpus_protocol_path=fixture.protocol,
                        catalog_path=fixture.catalog,
                        sidecar_path=fixture.sidecar,
                        corpus_paths=fixture.corpora,
                        output_path=fixture.output,
                        attempt_marker_path=fixture.attempt_marker,
                        confirmation_marker_path=fixture.confirmation_marker,
                        role_runner=mutating_runner,
                        formal=False,
                    )

    def test_worker_request_schema_has_no_label_channel(self) -> None:
        encoded = evaluate_p11._worker_request(
            "respond",
            7,
            ordinal=3,
            user_message="show me a blue dress",
            turn=2,
            top_k=10,
        )
        request = json.loads(encoded)
        self.assertEqual(
            set(request),
            {
                "request_id",
                "operation",
                "ordinal",
                "user_message",
                "turn",
                "top_k",
            },
        )
        with self.assertRaisesRegex(
            evaluate_p11.P11RunnerError, "parent-only boundary"
        ):
            evaluate_p11._worker_request(
                "respond",
                8,
                ordinal=3,
                user_message="blue",
                turn=2,
                top_k=10,
                sample_id="forbidden",
            )

    def test_parent_rejects_current_label_and_asin_shaped_outbound_content(self) -> None:
        class Delegate:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def reset(self, *_args: object) -> None:
                self.calls.append("reset")

            def respond(self, *_args: object) -> dict[str, object]:
                self.calls.append("respond")
                return {}

        delegate = Delegate()
        boundary = evaluate_p11.ParentLabelBoundary(delegate, ("CAT0000042",))
        with self.assertRaisesRegex(evaluate_p11.P11RunnerError, "current label"):
            boundary.reset("opaque-1", {"nested": {"key": "CAT0000042"}})
        self.assertEqual(delegate.calls, [])

        delegate = Delegate()
        boundary = evaluate_p11.ParentLabelBoundary(delegate, ("CAT0000042",))
        boundary.reset("opaque-2", {"summary": "fixture shopper"})
        shaped = "B0" + "12345678"
        with self.assertRaisesRegex(evaluate_p11.P11RunnerError, "ASIN-shaped"):
            boundary.respond("opaque-2", f"show {shaped}", 1, 10)
        self.assertEqual(delegate.calls, ["reset"])

    def test_repeat_resource_gate_rejects_repeat_only_overage(self) -> None:
        class RepeatOverageRunner(MockRoleRunner):
            def __init__(self) -> None:
                super().__init__()
                self.active_primary_calls = 0

            def __call__(self, *args: object, **kwargs: object) -> dict[str, object]:
                result = super().__call__(*args, **kwargs)
                role = str(args[0])
                samples = args[1]
                if role == evaluate_p11.ACTIVE_ID and str(samples[0]["sample_id"]).startswith("p"):
                    self.active_primary_calls += 1
                    if self.active_primary_calls == 2:
                        result["resources"]["wall_seconds"] = 200.0
                return result

        with tempfile.TemporaryDirectory() as directory:
            fixture = P11RunnerFixture(Path(directory))
            artifact = fixture.run(RepeatOverageRunner())

        repeat = artifact["execution"]["primary"]["repeat"]
        self.assertFalse(repeat["split_gate"]["passed"])
        self.assertFalse(repeat["passed"])
        self.assertFalse(
            artifact["execution"]["confirmation"]["semantic_parse_executed"]
        )

    def test_corpus_must_be_hash_frozen_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = P11RunnerFixture(Path(directory))
            protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
            protocol["splits"]["confirmation"]["expected_samples_sha256"] = None
            fixture.protocol.write_text(json.dumps(protocol), encoding="utf-8")

            with self.assertRaisesRegex(
                evaluate_p11.P11RunnerError, "not hash-frozen"
            ):
                evaluate_p11.preflight(
                    spec_path=fixture.spec,
                    corpus_protocol_path=fixture.protocol,
                    catalog_path=fixture.catalog,
                    sidecar_path=fixture.sidecar,
                    corpus_paths=fixture.corpora,
                )

    def test_global_deadline_policy_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = P11RunnerFixture(Path(directory))
            spec = json.loads(fixture.spec.read_text(encoding="utf-8"))
            spec["deadline_policy"]["formal_evaluation_seconds"] = 5_401
            fixture.spec.write_text(json.dumps(spec), encoding="utf-8")

            with self.assertRaisesRegex(
                evaluate_p11.P11RunnerError, "global deadline policy"
            ):
                evaluate_p11.preflight(
                    spec_path=fixture.spec,
                    corpus_protocol_path=fixture.protocol,
                    catalog_path=fixture.catalog,
                    sidecar_path=fixture.sidecar,
                    corpus_paths=fixture.corpora,
                )

    def test_expired_global_deadline_fails_before_phase_entry(self) -> None:
        with patch("scripts.evaluate_p11.time.monotonic", return_value=100.0):
            with self.assertRaisesRegex(
                evaluate_p11.P11RunnerError,
                "global deadline exceeded before primary initial roles",
            ):
                evaluate_p11._check_global_deadline(
                    100.0, "primary initial roles"
                )

    def test_worker_global_deadline_timeout_terminates_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            now = time.monotonic()
            client = evaluate_p11.WorkerClient(
                role=evaluate_p11.ACTIVE_ID,
                process=process,
                nonce="a" * 32,
                stderr_path=root / "stderr.bin",
                bootstrap_seconds=0.0,
                stage_manifest_sha256="b" * 64,
                deadline_monotonic=now + 60.0,
                global_deadline_monotonic=now + 0.05,
            )
            client._start_reader()
            try:
                with self.assertRaisesRegex(
                    evaluate_p11.P11RunnerError, "formal global deadline exceeded"
                ):
                    client._read(30.0, "fixture response")
                self.assertIsNotNone(process.poll())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()

    def test_role_wall_time_includes_staging_bootstrap_and_finalize(self) -> None:
        class FakeProcess:
            @staticmethod
            def poll() -> int:
                return 0

        class FakeWorker:
            process = FakeProcess()

            @staticmethod
            def finalize() -> dict[str, object]:
                return {}

            @staticmethod
            def abort() -> None:
                return None

        class FakeRecorder:
            errors: list[str] = []

        official = {
            "sample_count": 0,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": 0.0,
            "efficiency": 0.0,
            "recommended_technical_score": 0.0,
            "sessions": [],
            "scenario_metrics": {},
            "reported_token_usage": {"total_tokens": 0},
        }
        bundle = {
            "capture": {
                "stats": {"exception_count": 0},
                "integrity_errors": [],
            },
            "timing": {"respond_latency": {"p95_ms": 0.0}},
            "memory": {"peak_rss_bytes": 1},
            "response_sha256": "a" * 64,
            "asset_validation": {},
            "network_attempt_count": 0,
            "preimport_audit": {
                "schema_version": "p11.parent-verified-preimport-audit.v3",
                "record_bound_to_process": True,
                "record_loaded_after_clean_exit": True,
                "agent_close_and_candidate_atexit_covered": True,
                "supervisor_event_stream_verified": True,
                "denied_attempt_counts": {
                    "lifecycle": 0,
                    "network": 0,
                    "process": 0,
                    "read": 0,
                    "sqlite": 0,
                },
                "denied_attempt_total": 0,
                "post_atexit_memory": {
                    "schema_version": "p11.post-atexit-memory.v1",
                    "backend": "fixture",
                    "peak_rss_bytes": 1,
                    "available": True,
                    "covers_candidate_execution_through_atexit": True,
                },
                "record_sha256": "f" * 64,
            },
            "generic_exception_count": 0,
            "generic_exception_classes": [],
            "worker_process": {},
        }

        def stage(destination: Path, **_kwargs: object) -> tuple[Path, Path, str]:
            destination.mkdir()
            return destination / "bootstrap.py", destination / "worker.py", "b" * 64

        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text("{}\n", encoding="utf-8")
            with patch(
                "scripts.evaluate_p11.time.perf_counter", side_effect=(10.0, 15.0)
            ), patch(
                "scripts.evaluate_p11._stage_worker_runtime", side_effect=stage
            ), patch.object(
                evaluate_p11.WorkerClient, "start", return_value=FakeWorker()
            ), patch(
                "scripts.evaluate_p11.ContractRecorder", return_value=FakeRecorder()
            ), patch(
                "scripts.evaluate_p11.evaluate", return_value=official
            ), patch(
                "scripts.evaluate_p11._validate_worker_bundle", return_value=bundle
            ), patch("scripts.evaluate_p11._verify_stage"):
                result = evaluate_p11._run_role(
                    evaluate_p11.ACTIVE_ID,
                    [],
                    set(),
                    {},
                    {},
                    catalog_path=catalog,
                    sidecar_path=Path(directory) / "sidecar.sqlite",
                    sidecar_identity={"bytes": 1, "sha256": "c" * 64},
                    worker_path=Path(directory) / "worker.py",
                )

        self.assertEqual(result["resources"]["wall_seconds"], 5.0)

    def test_formal_attempt_marker_persists_after_global_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prereg = root / "lock.json"
            prereg.write_text("{}\n", encoding="utf-8")
            output = root / "result.json"
            attempt = root / "attempt.json"
            confirmation = root / "confirmation.json"
            before = {
                "spec": json.loads(evaluate_p11.DEFAULT_SPEC.read_text(encoding="utf-8")),
                "protocol": {"catalog": {"count": 0}},
                "identity_snapshot": {"data": {"sidecar": {"bytes": 1, "sha256": "a" * 64}}},
                "source_scan": {"passed": True},
            }

            def deadline_check(_deadline: float | None, phase: str) -> None:
                if phase == "formal-attempt marker completion":
                    raise evaluate_p11.P11RunnerError(
                        "P11 formal global deadline exceeded during fixture"
                    )

            with patch.object(evaluate_p11, "DEFAULT_OUTPUT", output), patch.object(
                evaluate_p11, "DEFAULT_ATTEMPT_MARKER", attempt
            ), patch.object(
                evaluate_p11, "DEFAULT_CONFIRMATION_MARKER", confirmation
            ), patch(
                "scripts.evaluate_p11._formal_paths_are_defaults", return_value=True
            ), patch(
                "scripts.evaluate_p11.preflight", return_value=before
            ), patch(
                "scripts.evaluate_p11.validate_prereg_lock",
                return_value={
                    "sha256": "c" * 64,
                    "source_commit": "d" * 40,
                    "git": {"head": "d" * 40},
                },
            ), patch(
                "scripts.evaluate_p11._formal_extra_identity_snapshot",
                return_value={},
            ), patch(
                "scripts.evaluate_p11._load_runtime_dependencies"
            ), patch(
                "scripts.evaluate_p11.catalog_index", return_value=(set(), {}, {})
            ), patch(
                "scripts.evaluate_p11._check_global_deadline",
                side_effect=deadline_check,
            ):
                with self.assertRaisesRegex(
                    evaluate_p11.P11RunnerError, "global deadline exceeded"
                ):
                    evaluate_p11.run_evaluation(
                        prereg_lock_path=prereg,
                        output_path=output,
                        attempt_marker_path=attempt,
                        confirmation_marker_path=confirmation,
                        formal=True,
                    )

            self.assertTrue(attempt.is_file())
            self.assertFalse(confirmation.exists())
            self.assertFalse(output.exists())

    def test_dry_preflight_does_not_start_global_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lock.json"
            lock.write_text("{}\n", encoding="utf-8")
            state = {
                "spec": {},
                "protocol": {},
                "identity_snapshot": {},
                "source_scan": {},
            }
            with patch(
                "scripts.evaluate_p11.preflight", return_value=state
            ), patch(
                "scripts.evaluate_p11.validate_prereg_lock", return_value={}
            ), patch(
                "scripts.evaluate_p11._load_runtime_dependencies"
            ), patch(
                "scripts.evaluate_p11.worker_smoke_preflight",
                return_value={"passed": True},
            ), patch(
                "scripts.evaluate_p11.time.monotonic",
                side_effect=AssertionError("dry preflight used global clock"),
            ), patch("builtins.print"):
                result = evaluate_p11.main(
                    ["--dry-preflight", "--prereg-lock", str(lock)]
                )

            self.assertEqual(result, 0)
            self.assertNotIn(
                "--formal-deadline-seconds",
                evaluate_p11._parser()._option_string_actions,
            )

    def test_atomic_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            evaluate_p11._atomic_write_json(output, {"passed": True})
            original = output.read_bytes()

            with self.assertRaises(FileExistsError):
                evaluate_p11._atomic_write_json(output, {"passed": False})

            self.assertEqual(output.read_bytes(), original)

    def test_atomic_writer_cleans_temp_on_fsync_and_link_failures(self) -> None:
        for failing_call in ("fsync", "link"):
            with self.subTest(failing_call=failing_call), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "result.json"
                target = f"scripts.evaluate_p11.os.{failing_call}"
                with patch(target, side_effect=OSError("fixture failure")):
                    with self.assertRaises(OSError):
                        evaluate_p11._atomic_write_json(output, {"passed": True})

                self.assertFalse(output.exists())
                self.assertEqual(list(root.glob(".result.json.*.tmp")), [])

    def test_atomic_writer_rejects_and_removes_a_post_link_deadline_overrun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            clock = [0.0]
            real_link = os.link

            def delayed_link(source: object, destination: object) -> None:
                real_link(source, destination)
                clock[0] = 2.0

            with patch(
                "scripts.evaluate_p11.time.monotonic", side_effect=lambda: clock[0]
            ), patch("scripts.evaluate_p11.os.link", side_effect=delayed_link):
                with self.assertRaisesRegex(
                    evaluate_p11.P11RunnerError, "global deadline exceeded"
                ):
                    evaluate_p11._atomic_write_json(
                        output,
                        {"passed": True},
                        deadline_monotonic=1.0,
                        phase="formal evaluation artifact",
                    )

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".result.json.*.tmp")), [])

    def test_formal_git_recomputes_official_source_ancestry(self) -> None:
        locked = "a" * 40
        head = "b" * 40
        unrelated = "c" * 40
        branch = "p4-architecture-search"

        def fake_git(*arguments: str, deadline_monotonic: float | None = None) -> str:
            del deadline_monotonic
            responses = {
                ("branch", "--show-current"): branch,
                ("rev-parse", "HEAD"): head,
                ("status", "--porcelain=v1", "--untracked-files=all"): "",
                ("rev-list", "--count", f"{locked}..{head}"): "1",
                ("rev-parse", f"{head}^"): locked,
                (
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    head,
                ): "configs/p11_prereg_lock.json",
                ("remote", "get-url", "origin"): evaluate_p11.EXPECTED_ORIGIN_URL,
                ("remote", "get-url", "upstream"): evaluate_p11.EXPECTED_UPSTREAM_URL,
                (
                    "ls-remote",
                    "--heads",
                    "origin",
                    f"refs/heads/{branch}",
                ): f"{head}\trefs/heads/{branch}",
                (
                    "ls-remote",
                    "--heads",
                    "upstream",
                    "refs/heads/main",
                ): f"{evaluate_p11.EXPECTED_UPSTREAM_HEAD}\trefs/heads/main",
                (
                    "rev-parse",
                    "refs/remotes/upstream/main",
                ): evaluate_p11.EXPECTED_UPSTREAM_HEAD,
                (
                    "merge-base",
                    locked,
                    evaluate_p11.EXPECTED_UPSTREAM_HEAD,
                ): unrelated,
            }
            return responses[arguments]

        source = {
            "git_commit": locked,
            "git_branch": branch,
            "remote_proof": {
                "verified": True,
                "advertised_head": locked,
            },
            "official_upstream": {
                "verified": True,
                "advertised_head": evaluate_p11.EXPECTED_UPSTREAM_HEAD,
            },
        }
        with patch("scripts.evaluate_p11._git", side_effect=fake_git), self.assertRaisesRegex(
            evaluate_p11.P11RunnerError, "not based on the official upstream"
        ):
            evaluate_p11._verify_formal_git(source)

    def test_artifact_safety_rejects_embedded_ids_in_keys_and_values(self) -> None:
        malicious = "prefix-B012345678-suffix"
        for artifact in ({malicious: True}, {"note": malicious}):
            with self.subTest(artifact=artifact), self.assertRaisesRegex(
                evaluate_p11.P11RunnerError, "raw product identifier"
            ):
                evaluate_p11._assert_artifact_safe(artifact)

        with self.assertRaisesRegex(
            evaluate_p11.P11RunnerError, "raw product identifier"
        ):
            evaluate_p11._assert_artifact_safe(
                {"note": "contains-FIXTURE-TARGET-01-here"},
                forbidden_identifiers={"FIXTURE-TARGET-01"},
            )

    def test_formal_mode_cannot_disable_boundaries_or_inject_runner(self) -> None:
        attempts = (
            {"role_runner": MockRoleRunner()},
            {"enforce_formal_git": False},
            {"require_default_formal_paths": False},
        )
        for kwargs in attempts:
            with self.subTest(kwargs=tuple(kwargs)), self.assertRaisesRegex(
                evaluate_p11.P11RunnerError, "forbids injected runners"
            ):
                evaluate_p11.run_evaluation(**kwargs)

    def test_formal_mode_requires_lock_before_any_worker_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-lock.json"
            with patch(
                "scripts.evaluate_p11._formal_paths_are_defaults", return_value=True
            ), patch("scripts.evaluate_p11._run_role") as worker:
                with self.assertRaisesRegex(
                    evaluate_p11.P11RunnerError, "requires configs/p11_prereg_lock.json"
                ):
                    evaluate_p11.run_evaluation(prereg_lock_path=missing)

            worker.assert_not_called()

    def test_nonformal_mode_requires_fixture_runner_and_nondefault_assets(self) -> None:
        with self.assertRaisesRegex(
            evaluate_p11.P11RunnerError, "injected fixture runner"
        ):
            evaluate_p11.run_evaluation(formal=False)
        with self.assertRaisesRegex(
            evaluate_p11.P11RunnerError, "cannot use preregistered formal assets"
        ):
            evaluate_p11.run_evaluation(
                formal=False, role_runner=MockRoleRunner()
            )
        self.assertNotIn(
            "--nonformal", evaluate_p11._parser()._option_string_actions
        )

    def test_formal_runtime_rejects_preloaded_or_injected_dependencies(self) -> None:
        with patch.object(evaluate_p11, "catalog_index", lambda _: None):
            with self.assertRaisesRegex(
                evaluate_p11.P11RunnerError, "loaded before lock validation"
            ):
                evaluate_p11._load_runtime_dependencies(formal=True)

    def test_fresh_loader_passes_but_transitive_preload_fails_in_isolated_process(self) -> None:
        fresh = subprocess.run(
            [
                sys.executable,
                "-c",
                "from scripts import evaluate_p11 as r; "
                "r._load_runtime_dependencies(formal=True); print('PASS')",
            ],
            cwd=evaluate_p11.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(fresh.returncode, 0, fresh.stderr)
        self.assertEqual(fresh.stdout.strip(), "PASS")

        preloaded = subprocess.run(
            [
                sys.executable,
                "-c",
                "import starter.p11_features; "
                "from scripts import evaluate_p11 as r; "
                "r._load_runtime_dependencies(formal=True)",
            ],
            cwd=evaluate_p11.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertNotEqual(preloaded.returncode, 0)
        self.assertIn("source closure was loaded before lock validation", preloaded.stderr)

    def test_runtime_gate_contract_drift_fails_closed(self) -> None:
        import scripts.p11_gates as gates

        cases = (
            ("PRIMARY_MIN_SCORE_DELTA", Decimal("0.004")),
            ("SPLIT_NAMES", ("primary", "uniform_tail", "confirmation")),
            ("REQUIRED_FLAGS", tuple(gates.REQUIRED_FLAGS[:-1])),
            (
                "RESOURCE_LIMITS",
                {**gates.RESOURCE_LIMITS, "wall_seconds": Decimal("1.16")},
            ),
        )
        for name, value in cases:
            with self.subTest(name=name), patch.object(gates, name, value):
                with self.assertRaisesRegex(
                    evaluate_p11.P11RunnerError, "gate constants"
                ):
                    evaluate_p11._load_runtime_dependencies(formal=False)

    def test_one_shot_marker_is_exclusive_and_persists_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "attempt.json"
            payload = {
                "schema_version": "p11.formal-attempt.v1",
                "preregistration_sha256": "a" * 64,
                "source_commit": "b" * 40,
                "attempt_nonce": "c" * 32,
            }

            def create() -> str:
                try:
                    evaluate_p11._exclusive_marker(marker, payload, "fixture")
                except evaluate_p11.P11RunnerError:
                    return "rejected"
                return "created"

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = sorted(pool.map(lambda _: create(), range(2)))

            self.assertEqual(outcomes, ["created", "rejected"])
            self.assertEqual(json.loads(marker.read_text(encoding="utf-8")), payload)
            self.assertEqual(list(root.glob(".attempt.json.*.tmp")), [])

            try:
                raise RuntimeError("simulated post-marker crash")
            except RuntimeError:
                pass
            self.assertTrue(marker.is_file())

    def test_parent_source_does_not_import_candidate_modules_pre_boundary(self) -> None:
        runner_source = Path(evaluate_p11.__file__).read_text(encoding="utf-8")
        lock_source = (
            evaluate_p11.PROJECT_ROOT / "scripts" / "build_p11_prereg_lock.py"
        ).read_text(encoding="utf-8")
        for source in (runner_source, lock_source):
            self.assertNotIn("from starter.p11_features import", source)
            self.assertNotIn("from starter.p11_lab import", source)

    def test_target_blind_source_scan_rejects_sqlite_authorizer_bypass(self) -> None:
        variants = {
            "direct_reset": "connection.set_authorizer(None)\n",
            "joined_reset": "getattr(connection, 'set_' + 'authorizer')(None)\n",
            "joined_attach": "QUERY = 'AT' + 'TACH DATA' + 'BASE ? AS leaked'\n",
            "direct_constructor": "from sqlite3 import Connection\n",
        }
        for name, source in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = {
                    item: f"{item}.py"
                    for item in evaluate_p11.TARGET_BLIND_SOURCE_NAMES
                }
                for relative in paths.values():
                    (root / relative).write_text("VALUE = 'safe'\n", encoding="utf-8")
                (root / paths["p11_lab"]).write_text(source, encoding="utf-8")
                with patch.object(evaluate_p11, "PROJECT_ROOT", root), patch.object(
                    evaluate_p11, "SOURCE_PATHS", paths
                ):
                    proof = evaluate_p11._target_blind_source_scan()
            self.assertFalse(proof["passed"])
            self.assertGreater(proof["finding_count"], 0)

    def test_preimport_record_binds_valid_post_atexit_memory(self) -> None:
        counts = {
            "lifecycle": 0,
            "network": 0,
            "process": 0,
            "read": 0,
            "sqlite": 0,
        }
        nonce = "4" * 32
        record = {
            "schema_version": "p11.preimport-audit.v3",
            "role": evaluate_p11.CONTROL_ID,
            "nonce": nonce,
            "denied_attempt_counts": counts,
            "denied_attempt_total": 0,
            "post_atexit_memory": {
                "schema_version": "p11.post-atexit-memory.v1",
                "backend": "fixture",
                "peak_rss_bytes": 4096,
                "available": True,
                "covers_candidate_execution_through_atexit": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preimport-audit.json"
            path.write_bytes(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
                + b"\n"
            )
            verified = evaluate_p11._load_preimport_audit_record(
                path,
                role=evaluate_p11.CONTROL_ID,
                nonce=nonce,
                process_returncode=0,
                supervisor_denied_attempt_counts=counts,
            )
            self.assertEqual(
                verified["schema_version"],
                "p11.parent-verified-preimport-audit.v3",
            )
            self.assertEqual(
                verified["post_atexit_memory"], record["post_atexit_memory"]
            )

            record["post_atexit_memory"]["peak_rss_bytes"] = 0
            path.write_bytes(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
                + b"\n"
            )
            with self.assertRaisesRegex(
                evaluate_p11.P11RunnerError, "post-atexit memory value is invalid"
            ):
                evaluate_p11._load_preimport_audit_record(
                    path,
                    role=evaluate_p11.CONTROL_ID,
                    nonce=nonce,
                    process_returncode=0,
                    supervisor_denied_attempt_counts=counts,
                )

    def test_runner_asin_scan_rejects_static_encodings(self) -> None:
        sources = {
            "bytes": "FROZEN = b'B012345678'\n",
            "concatenated": "FROZEN = 'B0' + '12345678'\n",
            "escaped_bytes": r"FROZEN = b'\x42\x30\x31\x32\x33\x34\x35\x36\x37\x38'"
            + "\n",
            "f_string": "FROZEN = f\"{'B0'}{'12345678'}\"\n",
            "joined": "FROZEN = ''.join(('B0', '12345678'))\n",
        }
        for name, source_text in sources.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "fixture.py"
                source.write_text(source_text, encoding="utf-8")
                source_files = {
                    key: {"path": str(source), "bytes": 1, "sha256": "a" * 64}
                    for key in evaluate_p11.RUNTIME_ASIN_SCAN_NAMES
                }
                with self.assertRaisesRegex(
                    evaluate_p11.P11RunnerError, "ASIN-shaped string literal"
                ):
                    evaluate_p11._independent_asin_literal_scan(source_files)

    def test_postload_exact_source_scan_is_aggregate_bound_and_fail_closed(self) -> None:
        target = "ZXCV-PRIVATE-42"
        source_variants = {
            "raw": f"FROZEN = {target!r}\n",
            "bytes": f"FROZEN = {target.encode('ascii')!r}\n",
            "concatenated": "FROZEN = 'ZXCV-' + 'PRIVATE-42'\n",
            "joined": "FROZEN = ''.join(('ZXCV-', 'PRIVATE-42'))\n",
        }
        for name, source_text in source_variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = {
                    key: f"{index:02d}_{key}.py"
                    for index, key in enumerate(evaluate_p11.RUNTIME_ASIN_SCAN_NAMES)
                }
                for relative in paths.values():
                    (root / relative).write_text("VALUE = 'safe'\n", encoding="utf-8")
                (root / paths["p11_lab"]).write_text(source_text, encoding="utf-8")
                source_files = {
                    key: {"bytes": 1, "sha256": "a" * 64} for key in paths
                }
                with patch.object(evaluate_p11, "PROJECT_ROOT", root), patch.object(
                    evaluate_p11, "SOURCE_PATHS", paths
                ):
                    with self.assertRaises(evaluate_p11.P11RunnerError) as raised:
                        evaluate_p11._postload_source_identifier_scan(
                            source_files,
                            {target},
                            target_registry_sha256=evaluate_p11._stable_sha256([target]),
                        )
                self.assertNotIn(target, str(raised.exception))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                key: f"{index:02d}_{key}.py"
                for index, key in enumerate(evaluate_p11.RUNTIME_ASIN_SCAN_NAMES)
            }
            for relative in paths.values():
                (root / relative).write_text("VALUE = 'safe'\n", encoding="utf-8")
            source_files = {
                key: {"bytes": 1, "sha256": "a" * 64} for key in paths
            }
            with patch.object(evaluate_p11, "PROJECT_ROOT", root), patch.object(
                evaluate_p11, "SOURCE_PATHS", paths
            ):
                proof = evaluate_p11._postload_source_identifier_scan(
                    source_files,
                    {target},
                    target_registry_sha256=evaluate_p11._stable_sha256([target]),
                )
            self.assertEqual(proof["source_file_count"], len(paths))
            self.assertEqual(proof["identifier_count"], 1)
            self.assertEqual(proof["match_count"], 0)
            self.assertTrue(proof["passed"])
            self.assertNotIn(target, json.dumps(proof, sort_keys=True))
            with patch.object(evaluate_p11, "PROJECT_ROOT", root), patch.object(
                evaluate_p11, "SOURCE_PATHS", paths
            ):
                with self.assertRaisesRegex(
                    evaluate_p11.P11RunnerError, "scan inputs are invalid"
                ):
                    evaluate_p11._postload_source_identifier_scan(
                        source_files,
                        {target},
                        target_registry_sha256="b" * 64,
                    )

    def test_primary_source_scan_failure_stops_before_all_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = P11RunnerFixture(Path(directory))
            runner = MockRoleRunner()
            with patch(
                "scripts.evaluate_p11._postload_source_identifier_scan",
                side_effect=evaluate_p11.P11RunnerError(
                    "P11 frozen source contains a split product identifier"
                ),
            ), self.assertRaises(evaluate_p11.P11RunnerError):
                fixture.run(runner)
        self.assertEqual(runner.calls, [])

    def test_uniform_source_scan_failure_stops_before_uniform_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = P11RunnerFixture(Path(directory))
            runner = MockRoleRunner()
            real_scan = evaluate_p11._postload_source_identifier_scan

            def fail_uniform(
                source_files: dict[str, object],
                identifiers: set[str],
                *,
                target_registry_sha256: str,
            ) -> dict[str, object]:
                if "A000000011" in identifiers:
                    raise evaluate_p11.P11RunnerError(
                        "P11 frozen source contains a split product identifier"
                    )
                return real_scan(
                    source_files,
                    identifiers,
                    target_registry_sha256=target_registry_sha256,
                )

            with patch(
                "scripts.evaluate_p11._postload_source_identifier_scan",
                side_effect=fail_uniform,
            ), self.assertRaises(evaluate_p11.P11RunnerError):
                fixture.run(runner)
        self.assertEqual({split for split, _role in runner.calls}, {"primary"})

    def test_confirmation_source_scan_failure_stops_before_confirmation_roles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = P11RunnerFixture(Path(directory))
            runner = MockRoleRunner()
            real_scan = evaluate_p11._postload_source_identifier_scan

            def fail_confirmation(
                source_files: dict[str, object],
                identifiers: set[str],
                *,
                target_registry_sha256: str,
            ) -> dict[str, object]:
                if "A000000021" in identifiers:
                    raise evaluate_p11.P11RunnerError(
                        "P11 frozen source contains a split product identifier"
                    )
                return real_scan(
                    source_files,
                    identifiers,
                    target_registry_sha256=target_registry_sha256,
                )

            with patch(
                "scripts.evaluate_p11._postload_source_identifier_scan",
                side_effect=fail_confirmation,
            ), self.assertRaises(evaluate_p11.P11RunnerError):
                fixture.run(runner)
        self.assertNotIn("confirmation", {split for split, _role in runner.calls})

    def test_preimport_bootstrap_blocks_network_and_control_sidecar_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            sidecar = root / "features.sqlite"
            catalog.write_text("{}\n", encoding="utf-8")
            connection = sqlite3.connect(sidecar)
            connection.execute("CREATE TABLE fixture (value INTEGER)")
            connection.commit()
            connection.close()
            scratch = root / "scratch"
            scratch.mkdir()
            bootstrap, _, _ = evaluate_p11._stage_worker_runtime(
                root / "control-stage",
                role=evaluate_p11.CONTROL_ID,
                worker_path=evaluate_p11.DEFAULT_WORKER,
                catalog_path=catalog,
                sidecar_path=sidecar,
                scratch=scratch,
            )
            network_probe = bootstrap.parent / "network_probe.py"
            network_probe.write_text(
                "import socket\nsocket.socket()\n", encoding="utf-8"
            )
            denied_network = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(bootstrap),
                    str(network_probe),
                ],
                cwd=bootstrap.parent,
                env=evaluate_p11._minimal_worker_environment(scratch),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(denied_network.returncode, 96)
            self.assertIn("network", _audit_categories(denied_network.stdout))

            sidecar_probe = bootstrap.parent / "sidecar_probe.py"
            sidecar_probe.write_text(
                f"from pathlib import Path\nPath({str(sidecar)!r}).read_bytes()\n",
                encoding="utf-8",
            )
            denied_sidecar = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(bootstrap),
                    str(sidecar_probe),
                ],
                cwd=bootstrap.parent,
                env=evaluate_p11._minimal_worker_environment(scratch),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(denied_sidecar.returncode, 96)
            self.assertIn("read", _audit_categories(denied_sidecar.stdout))

            sqlite_probe = bootstrap.parent / "sqlite_probe.py"
            sqlite_probe.write_text(
                f"import sqlite3\nsqlite3.connect({str(sidecar)!r})\n",
                encoding="utf-8",
            )
            denied_sqlite = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(bootstrap),
                    str(sqlite_probe),
                ],
                cwd=bootstrap.parent,
                env=evaluate_p11._minimal_worker_environment(scratch),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(denied_sqlite.returncode, 96)
            self.assertIn("sqlite", _audit_categories(denied_sqlite.stdout))

            attach_probe = bootstrap.parent / "sqlite_attach_probe.py"
            attach_probe.write_text(
                "import sqlite3\n"
                "connection = sqlite3.connect(':memory:')\n"
                f"connection.execute('ATTACH DATABASE ? AS leaked', ({str(sidecar)!r},))\n"
                "connection.execute('SELECT value FROM leaked.fixture').fetchone()\n",
                encoding="utf-8",
            )
            denied_attach = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(bootstrap),
                    str(attach_probe),
                ],
                cwd=bootstrap.parent,
                env=evaluate_p11._minimal_worker_environment(scratch),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(denied_attach.returncode, 96)
            self.assertIn("sqlite", _audit_categories(denied_attach.stdout))
            self.assertFalse((scratch / "preimport-audit.json").exists())

            direct_constructor_probe = bootstrap.parent / "sqlite_connection_probe.py"
            direct_constructor_probe.write_text(
                "from sqlite3 import Connection\n"
                "connection = Connection(':memory:')\n"
                f"connection.execute('AT' + 'TACH DATA' + 'BASE ? AS leaked', ({str(sidecar)!r},))\n"
                "connection.execute('SELECT value FROM leaked.fixture').fetchone()\n",
                encoding="utf-8",
            )
            denied_direct_constructor = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(bootstrap),
                    str(direct_constructor_probe),
                    "--role",
                    evaluate_p11.CONTROL_ID,
                    "--nonce",
                    "6" * 32,
                ],
                cwd=bootstrap.parent,
                env=evaluate_p11._minimal_worker_environment(scratch),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(denied_direct_constructor.returncode, 96)
            self.assertIn(
                "sqlite", _audit_categories(denied_direct_constructor.stdout)
            )

            direct_path_probe = bootstrap.parent / "sqlite_connection_path_probe.py"
            direct_path_probe.write_text(
                "from sqlite3 import Connection\n"
                f"Connection({str(sidecar)!r}).execute('SELECT value FROM fixture').fetchone()\n",
                encoding="utf-8",
            )
            denied_direct_path = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(bootstrap),
                    str(direct_path_probe),
                    "--role",
                    evaluate_p11.CONTROL_ID,
                    "--nonce",
                    "3" * 32,
                ],
                cwd=bootstrap.parent,
                env=evaluate_p11._minimal_worker_environment(scratch),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(denied_direct_path.returncode, 96)
            self.assertIn("sqlite", _audit_categories(denied_direct_path.stdout))

            authorizer_reset_probe = bootstrap.parent / "sqlite_reset_probe.py"
            authorizer_reset_probe.write_text(
                "import sqlite3\n"
                "connection = sqlite3.connect(':memory:')\n"
                "getattr(connection, 'set_' + 'authorizer')(None)\n",
                encoding="utf-8",
            )
            denied_authorizer_reset = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(bootstrap),
                    str(authorizer_reset_probe),
                    "--role",
                    evaluate_p11.CONTROL_ID,
                    "--nonce",
                    "5" * 32,
                ],
                cwd=bootstrap.parent,
                env=evaluate_p11._minimal_worker_environment(scratch),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(denied_authorizer_reset.returncode, 96)
            self.assertIn("sqlite", _audit_categories(denied_authorizer_reset.stdout))

            active_scratch = root / "active-scratch"
            active_scratch.mkdir()
            active_bootstrap, _, _ = evaluate_p11._stage_worker_runtime(
                root / "active-stage",
                role=evaluate_p11.ACTIVE_ID,
                worker_path=evaluate_p11.DEFAULT_WORKER,
                catalog_path=catalog,
                sidecar_path=sidecar,
                scratch=active_scratch,
            )
            active_probe = active_bootstrap.parent / "sidecar_probe.py"
            active_probe.write_text(
                "import sqlite3\n"
                f"sqlite3.connect({(sidecar.resolve().as_uri() + '?mode=ro&immutable=1')!r}, uri=True).close()\n",
                encoding="utf-8",
            )
            allowed_sidecar = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(active_bootstrap),
                    str(active_probe),
                ],
                cwd=active_bootstrap.parent,
                env=evaluate_p11._minimal_worker_environment(active_scratch),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(
                allowed_sidecar.returncode,
                0,
                f"stderr={allowed_sidecar.stderr}\nstdout={allowed_sidecar.stdout}",
            )

    def test_oob_audit_fail_stops_lifecycle_read_and_candidate_atexit(self) -> None:
        probes = {
            "lifecycle": (
                "import atexit\n"
                "def callback():\n"
                "    pass\n"
                "atexit.unregister(callback)\n"
            ),
            "read": "open(SECRET, encoding='utf-8').read()\n",
            "network": (
                "import atexit\n"
                "import socket\n"
                "def callback():\n"
                "    socket.socket()\n"
                "atexit.register(callback)\n"
            ),
        }
        for expected_category, source in probes.items():
            with self.subTest(category=expected_category), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                catalog = root / "catalog.jsonl"
                sidecar = root / "features.sqlite"
                secret = root / "secret.txt"
                catalog.write_text("{}\n", encoding="utf-8")
                sidecar.write_bytes(b"fixture")
                secret.write_text("forbidden", encoding="utf-8")
                scratch = root / "scratch"
                scratch.mkdir()
                bootstrap, _, _ = evaluate_p11._stage_worker_runtime(
                    root / "stage",
                    role=evaluate_p11.CONTROL_ID,
                    worker_path=evaluate_p11.DEFAULT_WORKER,
                    catalog_path=catalog,
                    sidecar_path=sidecar,
                    scratch=scratch,
                )
                probe = bootstrap.parent / "audit_probe.py"
                probe.write_text(
                    f"SECRET = {str(secret)!r}\n" + source,
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-B",
                        str(bootstrap),
                        str(probe),
                        "--role",
                        evaluate_p11.CONTROL_ID,
                        "--nonce",
                        "7" * 32,
                    ],
                    cwd=bootstrap.parent,
                    env=evaluate_p11._minimal_worker_environment(scratch),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 96)
                self.assertIn(expected_category, _audit_categories(completed.stdout))
                self.assertFalse((scratch / "preimport-audit.json").exists())

    def test_oob_audit_forgery_or_path_sabotage_forces_nonzero_exit(self) -> None:
        for sabotage in ("file", "directory"):
            with self.subTest(sabotage=sabotage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                catalog = root / "catalog.jsonl"
                sidecar = root / "features.sqlite"
                catalog.write_text("{}\n", encoding="utf-8")
                sidecar.write_bytes(b"fixture")
                scratch = root / "scratch"
                scratch.mkdir()
                bootstrap, _, _ = evaluate_p11._stage_worker_runtime(
                    root / "stage",
                    role=evaluate_p11.CONTROL_ID,
                    worker_path=evaluate_p11.DEFAULT_WORKER,
                    catalog_path=catalog,
                    sidecar_path=sidecar,
                    scratch=scratch,
                )
                nonce = "8" * 32
                record_path = scratch / "preimport-audit.json"
                forged = {
                    "schema_version": "p11.preimport-audit.v2",
                    "role": evaluate_p11.CONTROL_ID,
                    "nonce": nonce,
                    "denied_attempt_counts": {
                        "lifecycle": 0,
                        "network": 0,
                        "process": 0,
                        "read": 0,
                        "sqlite": 0,
                    },
                    "denied_attempt_total": 0,
                }
                probe = bootstrap.parent / "audit_sabotage_probe.py"
                action = (
                    f"open({str(record_path)!r}, 'w', encoding='utf-8').write({(json.dumps(forged) + chr(10))!r})"
                    if sabotage == "file"
                    else f"os.mkdir({str(record_path)!r})"
                )
                probe.write_text(
                    "import atexit\n"
                    "import os\n"
                    "def sabotage_record():\n"
                    f"    {action}\n"
                    "atexit.register(sabotage_record)\n",
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-B",
                        str(bootstrap),
                        str(probe),
                        "--role",
                        evaluate_p11.CONTROL_ID,
                        "--nonce",
                        nonce,
                    ],
                    cwd=bootstrap.parent,
                    env=evaluate_p11._minimal_worker_environment(scratch),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                self.assertNotEqual(completed.returncode, 0)
                with self.assertRaisesRegex(
                    evaluate_p11.P11RunnerError, "clean worker exit"
                ):
                    evaluate_p11._load_preimport_audit_record(
                        record_path,
                        role=evaluate_p11.CONTROL_ID,
                        nonce=nonce,
                        process_returncode=completed.returncode,
                        supervisor_denied_attempt_counts={
                            name: 0 for name in evaluate_p11.AUDIT_DENIAL_CATEGORIES
                        },
                    )

    def test_denial_fail_stops_before_candidate_can_reset_closure_counter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            sidecar = root / "features.sqlite"
            catalog.write_text("{}\n", encoding="utf-8")
            sidecar.write_bytes(b"fixture")
            scratch = root / "scratch"
            scratch.mkdir()
            bootstrap, _, _ = evaluate_p11._stage_worker_runtime(
                root / "stage",
                role=evaluate_p11.CONTROL_ID,
                worker_path=evaluate_p11.DEFAULT_WORKER,
                catalog_path=catalog,
                sidecar_path=sidecar,
                scratch=scratch,
            )
            nonce = "9" * 32
            record_path = scratch / "preimport-audit.json"
            probe = bootstrap.parent / "audit_counter_reset_probe.py"
            probe.write_text(
                "import atexit\n"
                "import gc\n"
                "import socket\n"
                "def reset_private_counter():\n"
                "    try:\n"
                "        socket.socket()\n"
                "    except PermissionError:\n"
                "        pass\n"
                "    expected = {'lifecycle', 'network', 'process', 'read', 'sqlite'}\n"
                "    for value in gc.get_objects():\n"
                "        if getattr(value, '__name__', '') != '_emit_final_record':\n"
                "            continue\n"
                "        for cell in getattr(value, '__closure__', ()) or ():\n"
                "            item = cell.cell_contents\n"
                "            if isinstance(item, dict) and set(item) == expected:\n"
                "                for key in item:\n"
                "                    item[key] = 0\n"
                "atexit.register(reset_private_counter)\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(bootstrap),
                    str(probe),
                    "--role",
                    evaluate_p11.CONTROL_ID,
                    "--nonce",
                    nonce,
                ],
                cwd=bootstrap.parent,
                env=evaluate_p11._minimal_worker_environment(scratch),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 96)
            self.assertIn("network", _audit_categories(completed.stdout))
            self.assertFalse(record_path.exists())

    def test_worker_client_drains_post_finalize_audit_event_before_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            sidecar = root / "features.sqlite"
            catalog.write_text("{}\n", encoding="utf-8")
            sidecar.write_bytes(b"fixture")
            scratch = root / "scratch"
            scratch.mkdir()
            worker_source = root / "late_audit_worker.py"
            worker_source.write_text(
                "import atexit\n"
                "import gc\n"
                "import json\n"
                "import os\n"
                "import socket\n"
                "import sys\n"
                "def reset_private_counter():\n"
                "    try:\n"
                "        os.dup2(2, 1)\n"
                "    except PermissionError:\n"
                "        pass\n"
                "    try:\n"
                "        socket.socket()\n"
                "    except PermissionError:\n"
                "        pass\n"
                "    expected = {'lifecycle', 'network', 'process', 'read', 'sqlite'}\n"
                "    for value in gc.get_objects():\n"
                "        if getattr(value, '__name__', '') != '_emit_final_record':\n"
                "            continue\n"
                "        for cell in getattr(value, '__closure__', ()) or ():\n"
                "            item = cell.cell_contents\n"
                "            if isinstance(item, dict) and set(item) == expected:\n"
                "                for key in item:\n"
                "                    item[key] = 0\n"
                "atexit.register(reset_private_counter)\n"
                "arguments = sys.argv[1:]\n"
                "role = arguments[arguments.index('--role') + 1]\n"
                "nonce = arguments[arguments.index('--nonce') + 1]\n"
                "def emit(value):\n"
                "    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\\n')\n"
                "    sys.stdout.flush()\n"
                "emit({'kind': 'ready', 'nonce': nonce, 'role': role})\n"
                "for line in sys.stdin:\n"
                "    request = json.loads(line)\n"
                "    if request['operation'] == 'finalize':\n"
                "        emit({'bundle': {}, 'kind': 'result', 'request_id': request['request_id']})\n"
                "        break\n",
                encoding="utf-8",
            )
            bootstrap, staged_worker, manifest_sha256 = (
                evaluate_p11._stage_worker_runtime(
                    root / "stage",
                    role=evaluate_p11.CONTROL_ID,
                    worker_path=worker_source,
                    catalog_path=catalog,
                    sidecar_path=sidecar,
                    scratch=scratch,
                )
            )
            client = evaluate_p11.WorkerClient.start(
                evaluate_p11.CONTROL_ID,
                catalog_path=catalog,
                sidecar_path=sidecar,
                sidecar_identity=evaluate_p11._file_identity(sidecar),
                worker_path=staged_worker,
                bootstrap_path=bootstrap,
                stage_manifest_sha256=manifest_sha256,
                scratch=scratch,
            )
            with self.assertRaisesRegex(
                evaluate_p11.P11RunnerError, "worker exited unsuccessfully"
            ):
                client.finalize()
            self.assertEqual(client.process.returncode, 96)
            self.assertGreaterEqual(client._supervisor_audit_counts["lifecycle"], 1)

    def test_worker_reader_validates_bound_audit_event_stream(self) -> None:
        role = evaluate_p11.CONTROL_ID
        nonce = "a" * 32

        class FakeProcess:
            def __init__(self, payload: bytes) -> None:
                self.stdout = io.BytesIO(payload)

        def client_for(messages: list[dict[str, object]]) -> evaluate_p11.WorkerClient:
            payload = b"".join(
                json.dumps(message, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
                + b"\n"
                for message in messages
            )
            return evaluate_p11.WorkerClient(
                role,
                FakeProcess(payload),  # type: ignore[arg-type]
                nonce,
                Path("unused-stderr"),
                0.0,
                "b" * 64,
                time.monotonic() + 30,
                None,
            )

        event = {
            "kind": "audit-denied",
            "category": "network",
            "role": role,
            "nonce": nonce,
            "sequence": 1,
        }
        valid = client_for([event, {"kind": "ready"}])
        valid._reader_loop()
        self.assertEqual(valid._supervisor_audit_counts["network"], 1)
        self.assertEqual(valid._messages.get_nowait(), ("message", {"kind": "ready"}))
        self.assertEqual(valid._messages.get_nowait(), ("eof", None))

        invalid_streams = {
            "wrong_nonce": [{**event, "nonce": "c" * 32}],
            "bool_sequence": [{**event, "sequence": True}],
            "duplicate_sequence": [event, event],
            "extra_key": [{**event, "extra": 1}],
        }
        for name, messages in invalid_streams.items():
            with self.subTest(name=name):
                invalid = client_for(messages)
                invalid._reader_loop()
                self.assertEqual(
                    invalid._messages.get_nowait(), ("error", "invalid audit event")
                )


if __name__ == "__main__":
    unittest.main()
