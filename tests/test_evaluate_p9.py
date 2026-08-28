from __future__ import annotations

import json
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import evaluate_p9


EVIDENCE = {
    "bytes": 1_486_848,
    "sha256": "2bc5846b7f6efb2e8395ea99b6bca5b585fb1507d23d6289dbc00d7600d22128",
}


def _file_entry(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": evaluate_p9._sha256_file(path),
        "git_blob_sha1": "f" * 40,
    }


def _sessions(rank: int = 2, turn: int = 2) -> list[dict]:
    return [
        {
            "sample_id": f"row_{index}",
            "scenario_type": scenario,
            "hit": True,
            "first_hit_turn": turn,
            "best_rank": rank,
            "reciprocal_rank": 1.0 / rank,
        }
        for index, scenario in enumerate(sorted(evaluate_p9.SCENARIOS), start=1)
    ]


def _run(
    role: str,
    *,
    rank: int = 2,
    bootstrap: float = 1.0,
    wall: float = 1.0,
    p95: float = 1.0,
    rss: int = 100,
) -> dict:
    sessions = _sessions(rank)
    totals = evaluate_p9.build_exact_totals(sessions)
    count = totals["sample_count"]
    configuration = {
        "retrieval_mode": "coverage",
        "rerank_mode": "off",
        "question_policy": "fast",
        "evidence_opened": role in {evaluate_p9.ROLES["shadow"], evaluate_p9.ROLES["active"]},
    }
    if configuration["evidence_opened"]:
        configuration.update({
            "evidence_identity_verified": True,
            "evidence_bytes": EVIDENCE["bytes"],
            "evidence_sha256": EVIDENCE["sha256"],
        })
    return {
        "role": role,
        "configuration": configuration,
        "stats": {"activations": 4, "output_changes": 4, "exception_count": 0},
        "metrics": {
            "sample_count": count,
            "hit_rate_at_10": round(totals["hit_count"] / count, 6),
            "mrr": round(totals["rr_sum_x2520"] / (evaluate_p9.RR_SCALE * count), 6),
            "mttc": round(totals["mttc_turn_sum"] / count, 6),
            "recommended_technical_score": round(
                totals["official_contribution_sum_x25200"]
                / (evaluate_p9.CONTRIBUTION_SCALE * count),
                6,
            ),
        },
        "exact_totals": totals,
        "functional_result_sha256": f"functional-{role}-{rank}",
        "response_trace_sha256": f"response-{role}-{rank}",
        "capture_hashes": {"audit_sha256": "a" * 64},
        "function_hashes": {"function": "b" * 64},
        "contract": {"error_count": 0, "errors_sha256": "c" * 64},
        "integrity": {"error_count": 0, "errors_sha256": "d" * 64},
        "runtime": {
            "network_attempt_count": 0,
            "audit_network_denied_attempt_count": 0,
            "read_denied_attempt_count": 0,
            "process_denied_attempt_count": 0,
            "evidence_open_count": (
                1 if role in {evaluate_p9.ROLES["shadow"], evaluate_p9.ROLES["active"]} else 0
            ),
            "generic_exception_count": 0,
            "generic_exception_classes_sha256": "e" * 64,
        },
        "timing": {
            "bootstrap_wall_seconds": bootstrap,
            "elapsed_at_final_worker_io_seconds": wall + bootstrap,
            "cumulative_worker_io_timeout_seconds": 180.0,
            "evaluation_wall_seconds": wall,
            "respond_latency": {"p95_ms": p95},
        },
        "memory": {
            "available": True,
            "peak_rss_bytes": rss,
            "covers_process_lifetime_peak": True,
        },
        "worker_process": {
            "separate_process": True,
            "pid": 999_999,
            "nonce": "1" * 32,
            "role": role,
            "staged_runtime": True,
            "python_audit_boundary": True,
            "minimal_environment": True,
            "hostile_native_code_sandboxed": False,
            "isolated_python_flags": ["-I", "-S", "-B"],
            "stage_manifest_sha256": "f" * 64,
        },
        "_sessions": sessions,
    }


class P9SpecAndBoundaryTests(unittest.TestCase):
    def test_live_matrix_spec_is_strict_and_sidecar_is_bounded(self) -> None:
        validated = evaluate_p9.validate_matrix_spec(
            evaluate_p9._load_json_object(evaluate_p9.DEFAULT_SPEC)
        )
        self.assertEqual(validated["roles"], evaluate_p9.ROLES)
        self.assertEqual(validated["mechanism"]["candidate_pool"], 50)
        self.assertEqual(
            validated["resource_limits"]["evidence_asset_max_bytes"], 16_777_216
        )
        self.assertEqual(validated["resource_limits"]["bootstrap_ratio"], 1.2)
        self.assertEqual(validated["resource_limits"]["bootstrap_timeout_seconds"], 120.0)
        self.assertEqual(validated["resource_limits"]["request_timeout_seconds"], 30.0)
        self.assertEqual(validated["resource_limits"]["finalize_timeout_seconds"], 30.0)
        self.assertEqual(
            validated["resource_limits"]["cumulative_worker_io_timeout_seconds"], 180.0
        )

    def test_spec_rejects_a_relaxed_gate_or_asset_limit(self) -> None:
        spec = evaluate_p9._load_json_object(evaluate_p9.DEFAULT_SPEC)
        spec["promotion_gates"]["mrr_strict_increase"] = False
        with self.assertRaisesRegex(evaluate_p9.P9RunnerError, "promotion"):
            evaluate_p9.validate_matrix_spec(spec)
        spec = evaluate_p9._load_json_object(evaluate_p9.DEFAULT_SPEC)
        spec["resource_limits"]["evidence_asset_max_bytes"] += 1
        with self.assertRaisesRegex(evaluate_p9.P9RunnerError, "resource"):
            evaluate_p9.validate_matrix_spec(spec)

    def test_worker_source_boundary_is_enforced(self) -> None:
        summary = evaluate_p9.validate_worker_source_boundary(
            evaluate_p9.PROJECT_ROOT / "scripts" / "p9_worker.py"
        )
        self.assertTrue(summary["forbidden_vocabulary_absent"])
        self.assertTrue(summary["parent_only_import_absent"])
        locked = {
            name: {"path": evaluate_p9.REQUIRED_SOURCE_PATHS[name]}
            for name in evaluate_p9.CANDIDATE_RUNTIME_SOURCE_NAMES
        }
        safety = evaluate_p9.validate_candidate_source_safety(
            evaluate_p9.PROJECT_ROOT, locked
        )
        self.assertTrue(safety["direct_dangerous_imports_absent"])
        self.assertFalse(safety["hostile_native_code_sandboxed"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsafe = {}
            for name in evaluate_p9.CANDIDATE_RUNTIME_SOURCE_NAMES:
                path = root / f"{name}.py"
                path.write_text("import ctypes\n" if name == "agent" else "VALUE = 1\n")
                unsafe[name] = {"path": path.relative_to(root).as_posix()}
            with self.assertRaisesRegex(evaluate_p9.P9RunnerError, "forbidden module"):
                evaluate_p9.validate_candidate_source_safety(root, unsafe)

    def test_worker_payloads_are_sanitized_and_lock_binds_asset_without_path(self) -> None:
        state = {
            "protocol": {
                "spec": {"sha256": "a" * 64},
                "catalog": {"sha256": "b" * 64},
                "evidence": EVIDENCE,
            },
            "identity_snapshot": {"lock": {"sha256": "c" * 64}},
            "spec": {
                "roles": evaluate_p9.ROLES,
                "served_control": {
                    "retrieval_mode": "coverage", "rerank_mode": "off", "question_policy": "fast"
                },
                "mechanism": {"candidate_pool": 50},
            },
        }
        worker_spec = evaluate_p9._worker_spec_payload(state)
        worker_lock = evaluate_p9._worker_lock_payload(state, worker_spec_sha256="d" * 64)
        self.assertEqual(worker_lock["evidence"], EVIDENCE)
        self.assertNotIn("path", worker_lock["evidence"])
        for payload in (worker_spec, worker_lock):
            encoded = json.dumps(payload).lower()
            for word in evaluate_p9._SANITIZED_FORBIDDEN:
                self.assertNotIn(word, encoded)

    def test_worker_argv_contains_only_safe_bootstrap_assets(self) -> None:
        fake_process = Mock()
        fake_process.pid = 12345
        fake_process.poll.return_value = 0
        fake_process.stdin = None
        fake_process.stdout = None
        popen = Mock(return_value=fake_process)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            evaluate_p9.uuid, "uuid4", return_value=SimpleNamespace(hex="a" * 32)
        ), patch.object(
            evaluate_p9.subprocess, "Popen", popen
        ), patch.object(
            evaluate_p9.time, "monotonic", side_effect=[1.0, 1.1, 1.2]
        ), patch.object(
            evaluate_p9.WorkerClient,
            "_start_reader",
        ), patch.object(
            evaluate_p9.WorkerClient,
            "_read_message",
            return_value={"kind": "ready", "nonce": "a" * 32, "role": evaluate_p9.ROLES["control"]},
        ) as read_message:
            root = Path(directory)
            evaluate_p9.WorkerClient.start(
                evaluate_p9.ROLES["control"],
                catalog=root / "catalog.jsonl",
                evidence=root / "evidence.sqlite",
                spec=root / "matrix.json",
                worker_lock=root / "worker-lock.json",
                worker_factory="starter.p9_lab:create_p9_agent",
                rss_sample_ms=10.0,
                stderr_path=root / "stderr.log",
                worker_script=root / "stage" / "scripts" / "p9_worker.py",
                working_directory=root / "stage",
                environment={"OMP_NUM_THREADS": "1"},
                bootstrap_timeout_seconds=120.0,
                request_timeout_seconds=30.0,
                finalize_timeout_seconds=30.0,
                exit_timeout_seconds=10.0,
                cumulative_worker_io_timeout_seconds=180.0,
                stage_manifest_sha256="f" * 64,
            )
        command = popen.call_args.args[0]
        self.assertEqual(
            {value for value in command if value.startswith("--")},
            {"--role", "--nonce", "--factory", "--catalog", "--evidence", "--spec", "--lock", "--rss-ms"},
        )
        self.assertEqual(command[:4], [evaluate_p9.sys.executable, "-I", "-S", "-B"])
        self.assertEqual(popen.call_args.kwargs["cwd"], root / "stage")
        self.assertEqual(popen.call_args.kwargs["env"], {"OMP_NUM_THREADS": "1"})
        read_message.assert_called_once_with(120.0, "bootstrap")
        joined = " ".join(command).lower()
        for word in ("ground_truth", "sample_id", "scenario", "selection", "confirmation", "public_set", "prior_p"):
            self.assertNotIn(word, joined)

    def test_staged_runtime_is_hash_bound_and_environment_drops_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in evaluate_p9.WORKER_RUNTIME_SOURCE_NAMES:
                path = root / evaluate_p9.REQUIRED_SOURCE_PATHS[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{name}\n".encode())
                files[name] = _file_entry(path, root)
            state = {"protocol": {"source": {"files": files}}}
            stage = root / "stage"
            stage.mkdir()
            with patch.object(evaluate_p9, "PROJECT_ROOT", root):
                worker, manifest = evaluate_p9._stage_worker_runtime(state, stage)
            self.assertEqual(worker, stage / "scripts" / "p9_worker.py")
            self.assertRegex(manifest, r"^[a-f0-9]{64}$")
            with patch.dict(
                evaluate_p9.os.environ,
                {"OPENAI_API_KEY": "blocked", "SECRET_VALUE": "blocked"},
                clear=False,
            ):
                environment = evaluate_p9._minimal_worker_environment(root / "scratch")
            self.assertNotIn("OPENAI_API_KEY", environment)
            self.assertNotIn("SECRET_VALUE", environment)
            self.assertNotIn("PYTHONPATH", environment)

    def test_asset_snapshot_detaches_live_paths_and_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = root / "live"
            live.mkdir()
            live_paths = {
                name: live / filename
                for name, filename in evaluate_p9.SNAPSHOT_INPUT_FILENAMES.items()
            }
            frozen_payloads = {
                name: f"{name}-frozen\n".encode("utf-8") for name in live_paths
            }
            for name, path in live_paths.items():
                path.write_bytes(frozen_payloads[name])
            entries = {
                name: {
                    "bytes": path.stat().st_size,
                    "sha256": evaluate_p9._sha256_file(path),
                }
                for name, path in live_paths.items()
            }
            state = {
                "protocol": {
                    "catalog": entries["catalog"],
                    "evidence": entries["evidence"],
                    "released_public": entries["released_public"],
                    "corpus_metadata": entries["corpus_metadata"],
                    "priors": {
                        name: entries[f"prior_{name}"]
                        for name in evaluate_p9.DEFAULT_PRIORS
                    },
                    "corpora": {
                        split: entries[f"corpus_{split}"]
                        for split in evaluate_p9.DEFAULT_CORPORA
                    },
                    "paths": {
                        "catalog": live_paths["catalog"],
                        "evidence": live_paths["evidence"],
                        "released_public": live_paths["released_public"],
                        "corpus_metadata": live_paths["corpus_metadata"],
                        "priors": {
                            name: live_paths[f"prior_{name}"]
                            for name in evaluate_p9.DEFAULT_PRIORS
                        },
                        "corpora": {
                            split: live_paths[f"corpus_{split}"]
                            for split in evaluate_p9.DEFAULT_CORPORA
                        },
                    },
                }
            }
            snapshot = evaluate_p9._create_asset_snapshot(state, root / "snapshot")
            runtime = evaluate_p9._runtime_state_with_asset_snapshot(state, snapshot)
            for path in live_paths.values():
                path.write_bytes(b"live-mutated\n")
            self.assertEqual(len(snapshot["files"]), 12)
            self.assertNotEqual(
                runtime["protocol"]["paths"]["catalog"], live_paths["catalog"]
            )
            self.assertEqual(
                runtime["protocol"]["paths"]["catalog"].read_bytes(),
                frozen_payloads["catalog"],
            )
            self.assertEqual(
                runtime["protocol"]["paths"]["released_public"].read_bytes(),
                frozen_payloads["released_public"],
            )
            self.assertEqual(
                runtime["protocol"]["paths"]["priors"]["p8_confirmation"].read_bytes(),
                frozen_payloads["prior_p8_confirmation"],
            )
            self.assertEqual(
                evaluate_p9._snapshot_runtime_path(
                    runtime, "corpus_confirmation"
                ).read_bytes(),
                frozen_payloads["corpus_confirmation"],
            )
            evaluate_p9._verify_asset_snapshot(snapshot)
            frozen_confirmation = evaluate_p9._snapshot_runtime_path(
                runtime, "corpus_confirmation"
            )
            frozen_confirmation.chmod(0o666)
            frozen_confirmation.write_bytes(b"tampered")
            with self.assertRaisesRegex(evaluate_p9.P9RunnerError, "snapshot identity"):
                evaluate_p9._verify_asset_snapshot(snapshot)

    def test_reader_deadline_kills_worker_fail_closed(self) -> None:
        process = Mock()
        process.poll.return_value = None
        process.stdin = None
        process.stdout = None
        client = evaluate_p9.WorkerClient(
            role=evaluate_p9.ROLES["control"],
            process=process,
            nonce="a" * 32,
            stderr_path=Path("missing-stderr"),
            bootstrap_wall_seconds=0.0,
            request_timeout_seconds=0.01,
            finalize_timeout_seconds=0.01,
            exit_timeout_seconds=0.01,
            stage_manifest_sha256="f" * 64,
            worker_io_epoch_monotonic=evaluate_p9.time.monotonic(),
            cumulative_worker_io_timeout_seconds=180.0,
            _messages=queue.Queue(),
        )
        with self.assertRaisesRegex(evaluate_p9.P9RunnerError, "deadline expired"):
            client._read_message(0.001, "request")
        process.kill.assert_called()

    def test_writer_deadline_kills_worker_fail_closed(self) -> None:
        class BlockingInput:
            def __init__(self) -> None:
                self.release = evaluate_p9.threading.Event()

            def write(self, _payload: bytes) -> None:
                self.release.wait(1.0)

            def flush(self) -> None:
                return None

            def close(self) -> None:
                self.release.set()

        stream = BlockingInput()
        process = Mock()
        process.poll.return_value = None
        process.kill.side_effect = stream.release.set
        process.stdin = stream
        process.stdout = None
        client = evaluate_p9.WorkerClient(
            role=evaluate_p9.ROLES["control"],
            process=process,
            nonce="a" * 32,
            stderr_path=Path("missing-stderr"),
            bootstrap_wall_seconds=0.0,
            request_timeout_seconds=0.01,
            finalize_timeout_seconds=0.01,
            exit_timeout_seconds=0.01,
            stage_manifest_sha256="f" * 64,
            worker_io_epoch_monotonic=evaluate_p9.time.monotonic(),
            cumulative_worker_io_timeout_seconds=180.0,
        )
        with self.assertRaisesRegex(evaluate_p9.P9RunnerError, "write deadline"):
            client._write_request(b"{}\n", 0.001, "request")
        process.kill.assert_called()


class P9LockAndGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = evaluate_p9.validate_matrix_spec(
            evaluate_p9._load_json_object(evaluate_p9.DEFAULT_SPEC)
        )

    def test_live_evaluator_is_the_frozen_official_blob(self) -> None:
        self.assertEqual(
            evaluate_p9._validate_official_evaluator(
                evaluate_p9.PROJECT_ROOT / "evaluator" / "local_evaluator.py"
            ),
            "7c808347b31ef3121a9cbc4810ac3eb325f950ba",
        )

    def test_git_snapshot_uses_direct_remote_advertisement(self) -> None:
        head = "a" * 40

        def fake_git(_root: Path, *arguments: str, binary: bool = False):
            values = {
                ("branch", "--show-current"): "pre",
                ("rev-parse", "HEAD"): head,
                ("status", "--porcelain=v1", "--untracked-files=all"): "",
                ("remote", "get-url", "origin"): evaluate_p9.EXPECTED_ORIGIN_URL,
                ("ls-remote", "--heads", "origin", "refs/heads/pre"): (
                    f"{head}\trefs/heads/pre"
                ),
            }
            return values[arguments]

        with patch.object(evaluate_p9, "_git", side_effect=fake_git) as git:
            snapshot = evaluate_p9._git_snapshot(Path("."))
        self.assertTrue(snapshot["online_verified"])
        self.assertEqual(snapshot["advertised_head"], head)
        self.assertIn(
            ((Path("."), "ls-remote", "--heads", "origin", "refs/heads/pre"), {}),
            [(call.args, call.kwargs) for call in git.call_args_list],
        )
        def local_remote(_root: Path, *arguments: str, binary: bool = False):
            if arguments == ("remote", "get-url", "origin"):
                return "file:///tmp/fake.git"
            return fake_git(_root, *arguments, binary=binary)

        with patch.object(evaluate_p9, "_git", side_effect=local_remote):
            with self.assertRaisesRegex(evaluate_p9.P9RunnerError, "official HTTPS"):
                evaluate_p9._git_snapshot(Path("."))

    def test_source_lock_verifies_current_files_and_git_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in sorted(evaluate_p9.REQUIRED_SOURCE_NAMES):
                path = root / evaluate_p9.REQUIRED_SOURCE_PATHS[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frozen\n")
                files[name] = _file_entry(path, root)
            source = {
                "git_commit": "a" * 40,
                "git_branch": "branch",
                "remote_proof": {
                    "remote": "origin",
                    "head_ref": "refs/heads/branch",
                    "advertised_head": "a" * 40,
                    "url_sha256": evaluate_p9.EXPECTED_ORIGIN_URL_SHA256,
                    "verified": True,
                },
                "files": files,
            }

            def fake_git(_root: Path, *arguments: str, binary: bool = False):
                if arguments[0] in {"hash-object", "rev-parse"}:
                    return "f" * 40
                if arguments[0] == "ls-remote":
                    return f"{'a' * 40}\trefs/heads/branch"
                if arguments[:3] == ("remote", "get-url", "origin"):
                    return evaluate_p9.EXPECTED_ORIGIN_URL
                if arguments[:2] == ("branch", "--show-current"):
                    return "branch"
                return b"" if binary else ""

            with patch.object(evaluate_p9, "_git", side_effect=fake_git), patch.object(
                evaluate_p9.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ):
                validated = evaluate_p9._validate_source_lock(root, source, enforce_git=True)
        self.assertEqual(set(validated["files"]), evaluate_p9.REQUIRED_SOURCE_NAMES)

    def test_control_requires_direct_bridge_and_never_opens_sidecar(self) -> None:
        reference = _run(evaluate_p9.BASELINE_ROLE)
        control = _run(evaluate_p9.ROLES["control"])
        control["functional_result_sha256"] = reference["functional_result_sha256"]
        control["response_trace_sha256"] = reference["response_trace_sha256"]
        gate = evaluate_p9.gate_control(control, reference, 4, self.spec)
        self.assertEqual(gate["decision"], "control")
        self.assertTrue(gate["gates"]["evidence_not_opened"])
        control["configuration"]["evidence_sha256"] = EVIDENCE["sha256"]
        gate = evaluate_p9.gate_control(control, reference, 4, self.spec)
        self.assertEqual(gate["decision"], "invalid_control")
        self.assertFalse(gate["gates"]["evidence_identity_not_loaded"])

    def test_shadow_must_open_asset_but_return_exact_control_output(self) -> None:
        control = _run(evaluate_p9.ROLES["control"])
        shadow = _run(evaluate_p9.ROLES["shadow"])
        shadow["functional_result_sha256"] = control["functional_result_sha256"]
        shadow["response_trace_sha256"] = control["response_trace_sha256"]
        gate = evaluate_p9.gate_shadow(shadow, control, 4, EVIDENCE)
        self.assertEqual(gate["decision"], "shadow_only")
        shadow["configuration"]["evidence_sha256"] = "0" * 64
        self.assertEqual(
            evaluate_p9.gate_shadow(shadow, control, 4, EVIDENCE)["decision"],
            "invalid_shadow",
        )

    def test_active_requires_quality_sidecar_and_all_resource_gates(self) -> None:
        control = _run(evaluate_p9.ROLES["control"], rank=2)
        active = _run(evaluate_p9.ROLES["active"], rank=1, bootstrap=1.2, wall=1.3, p95=1.3, rss=120)
        gate = evaluate_p9.gate_active(active, control, 4, self.spec, EVIDENCE)
        self.assertEqual(gate["decision"], "eligible", gate)
        self.assertTrue(all(gate["gates"].values()), gate)

        active = _run(evaluate_p9.ROLES["active"], rank=1, bootstrap=1.21, wall=1.31, p95=1.31, rss=121)
        gate = evaluate_p9.gate_active(active, control, 4, self.spec, EVIDENCE)
        self.assertEqual(gate["decision"], "reject")
        for name in (
            "bootstrap_within_1_20x", "wall_within_1_30x",
            "response_p95_within_1_30x", "peak_rss_within_1_20x",
        ):
            self.assertFalse(gate["gates"][name], name)

    def test_repeat_and_artifact_redaction_are_strict(self) -> None:
        initial = _run(evaluate_p9.ROLES["active"], rank=1)
        repeated = json.loads(json.dumps(initial))
        repeated["worker_process"]["nonce"] = "2" * 32
        self.assertTrue(evaluate_p9.repeat_exact(initial, repeated)["passed"])
        repeated["configuration"]["evidence_bytes"] += 1
        self.assertFalse(evaluate_p9.repeat_exact(initial, repeated)["passed"])
        for key in ("sessions", "sample_id", "ground_truth", "route_records"):
            with self.assertRaisesRegex(evaluate_p9.P9RunnerError, "prohibited"):
                evaluate_p9._assert_artifact_safe({"nested": {key: []}})
        with self.assertRaisesRegex(evaluate_p9.P9RunnerError, "ASIN-shaped"):
            evaluate_p9._assert_artifact_safe({"safe": "B012345678"})

    def test_atomic_output_never_overwrites_and_main_rejects_before_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            evaluate_p9._atomic_write_json(output, {"schema_version": "one"})
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                evaluate_p9._atomic_write_json(output, {"schema_version": "two"})
            self.assertEqual(output.read_bytes(), original)
            with patch.object(evaluate_p9, "run_evaluation") as run:
                with self.assertRaises(FileExistsError):
                    evaluate_p9.main(["--output", str(output)])
            run.assert_not_called()


class P9ProtocolFlowTests(unittest.TestCase):
    def test_confirmation_is_never_opened_when_selection_is_rejected(self) -> None:
        state = {
            "protocol": {
                "paths": {"catalog": Path("catalog")},
                "source": {},
                "source_target_scan": {
                    "source_file_count": 0,
                    "identifier_count": 0,
                    "match_count": 0,
                    "passed": True,
                    "proof_sha256": "f" * 64,
                },
            },
            "spec": {
                "worker_factory": "starter.p9_lab:create_p9_agent",
                "roles": evaluate_p9.ROLES,
                "served_control": {},
                "resource_limits": {},
            },
            "summary": {},
            "git": {},
            "worker_boundary": {},
            "identity_snapshot": {},
        }
        gates = {
            evaluate_p9.BASELINE_ROLE: {"decision": "served_reference"},
            evaluate_p9.ROLES["control"]: {"decision": "control"},
            evaluate_p9.ROLES["shadow"]: {"decision": "shadow_only"},
            evaluate_p9.ROLES["active"]: {"decision": "reject"},
        }
        split_loader = Mock(return_value=([{}], {"sample_count": 1}, {"asin"}))
        snapshot_paths = {
            name: Path("snapshot") / filename
            for name, filename in evaluate_p9.SNAPSHOT_INPUT_FILENAMES.items()
        }
        snapshot = {
            "root": Path("snapshot"),
            "paths": snapshot_paths,
            "files": {
                name: {"filename": filename, "bytes": 1, "sha256": "a" * 64}
                for name, filename in evaluate_p9.SNAPSHOT_INPUT_FILENAMES.items()
            },
        }
        with patch.object(evaluate_p9, "preflight", return_value=state), patch.object(
            evaluate_p9, "_create_asset_snapshot", return_value=snapshot
        ), patch.object(
            evaluate_p9, "_verify_asset_snapshot"
        ), patch.object(
            evaluate_p9, "catalog_index", return_value=(set(), {}, {})
        ), patch.object(
            evaluate_p9, "_prior_target_set", return_value=set()
        ), patch.object(
            evaluate_p9, "_load_split", split_loader
        ), patch.object(
            evaluate_p9, "_run_initial_split", return_value=({}, gates)
        ):
            artifact = evaluate_p9.run_evaluation()
        self.assertEqual(split_loader.call_count, 1)
        self.assertEqual(split_loader.call_args.args[0], "selection")
        self.assertTrue(artifact["confirmation"]["identity_bytes_hashed_preflight"])
        self.assertFalse(artifact["confirmation"]["semantic_parse_executed"])
        self.assertFalse(artifact["confirmation"]["official_aggregate_executed"])
        self.assertEqual(artifact["decision"], "retain_p9_c00")

    def test_cli_exposes_non_metric_dry_preflight(self) -> None:
        self.assertTrue(evaluate_p9._parser().parse_args(["--dry-preflight"]).dry_preflight)


if __name__ == "__main__":
    unittest.main()
