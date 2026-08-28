from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import evaluate_p8


def _file_entry(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": evaluate_p8._sha256_file(path),
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
        for index, scenario in enumerate(sorted(evaluate_p8.SCENARIOS), start=1)
    ]


def _run(role: str, *, rank: int = 2, wall: float = 1.0, p95: float = 1.0, rss: int = 100) -> dict:
    sessions = _sessions(rank)
    totals = evaluate_p8.build_exact_totals(sessions)
    count = totals["sample_count"]
    metrics = {
        "sample_count": count,
        "hit_rate_at_10": round(totals["hit_count"] / count, 6),
        "mrr": round(totals["rr_sum_x2520"] / (evaluate_p8.RR_SCALE * count), 6),
        "mttc": round(totals["mttc_turn_sum"] / count, 6),
        "recommended_technical_score": round(
            totals["official_contribution_sum_x25200"]
            / (evaluate_p8.CONTRIBUTION_SCALE * count),
            6,
        ),
    }
    return {
        "role": role,
        "configuration": {
            "retrieval_mode": "coverage",
            "rerank_mode": "off",
            "question_policy": "fast",
        },
        "stats": {"activations": 4, "output_changes": 4, "exception_count": 0},
        "metrics": metrics,
        "exact_totals": totals,
        "functional_result_sha256": f"functional-{role}-{rank}",
        "response_trace_sha256": f"response-{role}-{rank}",
        "capture_hashes": {"audit_sha256": "a" * 64},
        "function_hashes": {"function": "b" * 64},
        "contract": {"error_count": 0, "errors_sha256": "c" * 64},
        "integrity": {"error_count": 0, "errors_sha256": "d" * 64},
        "runtime": {
            "network_attempt_count": 0,
            "generic_exception_count": 0,
            "generic_exception_classes_sha256": "e" * 64,
        },
        "timing": {
            "evaluation_wall_seconds": wall,
            "respond_latency": {"p95_ms": p95},
        },
        "memory": {"available": True, "peak_rss_bytes": rss},
        "worker_process": {
            "isolated": True,
            "pid": 999_999,
            "nonce": "1" * 32,
            "role": role,
        },
        "_sessions": sessions,
    }


class P8SpecAndBoundaryTests(unittest.TestCase):
    def test_live_matrix_spec_is_strict_and_served_control_is_exact(self) -> None:
        spec = evaluate_p8._load_json_object(evaluate_p8.DEFAULT_SPEC)
        validated = evaluate_p8.validate_matrix_spec(spec)
        self.assertEqual(validated["roles"], evaluate_p8.ROLES)
        self.assertEqual(
            validated["served_control"],
            {"retrieval_mode": "coverage", "rerank_mode": "off", "question_policy": "fast"},
        )
        self.assertEqual(validated["mechanism"]["candidate_pool"], 50)

    def test_spec_rejects_a_relaxed_promotion_gate(self) -> None:
        spec = evaluate_p8._load_json_object(evaluate_p8.DEFAULT_SPEC)
        spec["promotion_gates"]["mrr_strict_increase"] = False
        with self.assertRaisesRegex(evaluate_p8.P8RunnerError, "promotion"):
            evaluate_p8.validate_matrix_spec(spec)

    def test_worker_source_boundary_is_enforced(self) -> None:
        summary = evaluate_p8.validate_worker_source_boundary(
            evaluate_p8.PROJECT_ROOT / "scripts" / "p8_worker.py"
        )
        self.assertTrue(summary["forbidden_vocabulary_absent"])
        self.assertTrue(summary["parent_only_import_absent"])

    def test_worker_lock_is_aggregate_only(self) -> None:
        state = {
            "protocol": {
                "spec": {"sha256": "a" * 64},
                "catalog": {"sha256": "b" * 64},
            },
            "identity_snapshot": {"lock": {"sha256": "c" * 64}},
            "spec": {
                "roles": evaluate_p8.ROLES,
                "served_control": {
                    "retrieval_mode": "coverage",
                    "rerank_mode": "off",
                    "question_policy": "fast",
                },
                "mechanism": {"candidate_pool": 50},
            },
        }
        worker_spec = evaluate_p8._worker_spec_payload(state)
        payload = evaluate_p8._worker_lock_payload(
            state, worker_spec_sha256="d" * 64
        )
        encoded = json.dumps(payload).lower()
        for word in ("corpus", "public", "prior", "ground_truth", "sample_id", "scenario", "target"):
            self.assertNotIn(word, encoded)
        self.assertEqual(payload["spec_sha256"], "d" * 64)
        self.assertEqual(payload["protocol_spec_sha256"], "a" * 64)

        encoded_spec = json.dumps(worker_spec).lower()
        self.assertEqual(worker_spec["schema_version"], evaluate_p8.WORKER_SPEC_SCHEMA_VERSION)
        for word in (
            "corpus", "public", "prior", "ground_truth", "sample_id", "scenario", "target"
        ):
            self.assertNotIn(word, encoded_spec)

    def test_worker_argv_contains_only_safe_bootstrap_paths(self) -> None:
        fake_process = SimpleNamespace(pid=12345)
        popen = Mock(return_value=fake_process)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            evaluate_p8.uuid, "uuid4", return_value=SimpleNamespace(hex="a" * 32)
        ), patch.object(
            evaluate_p8.subprocess, "Popen", popen
        ), patch.object(
            evaluate_p8.WorkerClient,
            "_read_message",
            return_value={
                "kind": "ready",
                "nonce": "a" * 32,
                "role": evaluate_p8.ROLES["control"],
            },
        ):
            root = Path(directory)
            evaluate_p8.WorkerClient.start(
                evaluate_p8.ROLES["control"],
                catalog=root / "catalog.jsonl",
                spec=root / "matrix.json",
                worker_lock=root / "worker-lock.json",
                worker_factory="starter.p8_lab:create_p8_agent",
                rss_sample_ms=10.0,
                stderr_path=root / "stderr.log",
            )
        command = popen.call_args.args[0]
        joined = " ".join(command).lower()
        self.assertEqual(
            {value for value in command if value.startswith("--")},
            {"--role", "--nonce", "--factory", "--catalog", "--spec", "--lock", "--rss-ms"},
        )
        for word in ("ground_truth", "sample_id", "scenario", "selection", "confirmation", "public_set"):
            self.assertNotIn(word, joined)


class P8LockTests(unittest.TestCase):
    def test_live_evaluator_is_the_frozen_official_blob(self) -> None:
        self.assertEqual(
            evaluate_p8._validate_official_evaluator(
                evaluate_p8.PROJECT_ROOT / "evaluator" / "local_evaluator.py"
            ),
            "7c808347b31ef3121a9cbc4810ac3eb325f950ba",
        )

    def test_source_lock_verifies_current_files_and_preregistered_git_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in sorted(evaluate_p8.REQUIRED_SOURCE_NAMES):
                path = root / evaluate_p8.REQUIRED_SOURCE_PATHS[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frozen\n")
                files[name] = _file_entry(path, root)
            source = {"git_commit": "a" * 40, "git_branch": "branch", "files": files}

            def fake_git(_root: Path, *arguments: str, binary: bool = False):
                if arguments[0] == "show":
                    return b"frozen\n"
                if arguments[:2] == ("branch", "--show-current"):
                    return "branch"
                return b"" if binary else ""

            with patch.object(evaluate_p8, "_git", side_effect=fake_git), patch.object(
                evaluate_p8.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ):
                validated = evaluate_p8._validate_source_lock(root, source, enforce_git=True)
            self.assertEqual(validated["git_commit"], "a" * 40)
            self.assertEqual(set(validated["files"]), evaluate_p8.REQUIRED_SOURCE_NAMES)

    def test_source_lock_rejects_a_git_blob_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in sorted(evaluate_p8.REQUIRED_SOURCE_NAMES):
                path = root / evaluate_p8.REQUIRED_SOURCE_PATHS[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frozen\n")
                files[name] = _file_entry(path, root)
            source = {"git_commit": "a" * 40, "git_branch": "branch", "files": files}

            def fake_git(_root: Path, *arguments: str, binary: bool = False):
                if arguments[0] == "show":
                    return b"changed\n"
                if arguments[:2] == ("branch", "--show-current"):
                    return "branch"
                return b"" if binary else ""

            with patch.object(evaluate_p8, "_git", side_effect=fake_git):
                with self.assertRaisesRegex(evaluate_p8.P8RunnerError, "Git blob"):
                    evaluate_p8._validate_source_lock(root, source, enforce_git=True)


class P8MetricGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = evaluate_p8.validate_matrix_spec(
            evaluate_p8._load_json_object(evaluate_p8.DEFAULT_SPEC)
        )

    def test_exact_totals_reconstruct_every_official_component(self) -> None:
        run = _run(evaluate_p8.ROLES["control"])
        self.assertTrue(evaluate_p8._exact_totals_match(run))
        self.assertEqual(run["exact_totals"]["scenario_sample_counts"], {
            name: 1 for name in sorted(evaluate_p8.SCENARIOS)
        })

    def test_active_requires_strict_mrr_and_technical_score_improvement(self) -> None:
        control = _run(evaluate_p8.ROLES["control"], rank=2)
        active = _run(evaluate_p8.ROLES["active"], rank=1)
        gate = evaluate_p8.gate_active(active, control, 4, self.spec)
        self.assertEqual(gate["decision"], "eligible")
        self.assertTrue(all(gate["gates"].values()), gate)

        tied = _run(evaluate_p8.ROLES["active"], rank=2)
        gate = evaluate_p8.gate_active(tied, control, 4, self.spec)
        self.assertEqual(gate["decision"], "reject")
        self.assertFalse(gate["gates"]["mrr_strict_increase"])
        self.assertFalse(gate["gates"]["technical_score_strict_increase"])

    def test_control_requires_an_independent_served_agent_bridge(self) -> None:
        reference = _run(evaluate_p8.BASELINE_ROLE, rank=2)
        control = _run(evaluate_p8.ROLES["control"], rank=2)
        control["functional_result_sha256"] = reference["functional_result_sha256"]
        control["response_trace_sha256"] = reference["response_trace_sha256"]
        gate = evaluate_p8.gate_control(control, reference, 4, self.spec)
        self.assertEqual(gate["decision"], "control")

        control["response_trace_sha256"] = "different"
        gate = evaluate_p8.gate_control(control, reference, 4, self.spec)
        self.assertEqual(gate["decision"], "invalid_control")
        self.assertFalse(gate["gates"]["response_trace_equals_served_agent"])

    def test_active_rejects_network_activation_and_resource_failures(self) -> None:
        control = _run(evaluate_p8.ROLES["control"], rank=2)
        active = _run(evaluate_p8.ROLES["active"], rank=1, wall=1.31, p95=1.31, rss=121)
        active["runtime"]["network_attempt_count"] = 1
        active["stats"]["activations"] = 0
        gate = evaluate_p8.gate_active(active, control, 4, self.spec)
        self.assertEqual(gate["decision"], "reject")
        for name in (
            "network_attempts_zero", "activation_positive", "wall_within_1_30x",
            "response_p95_within_1_30x", "peak_rss_within_1_20x",
        ):
            self.assertFalse(gate["gates"][name], name)

    def test_repeat_requires_exact_functional_response_totals_and_capture_hashes(self) -> None:
        initial = _run(evaluate_p8.ROLES["active"], rank=1)
        repeated = json.loads(json.dumps(initial))
        repeated["worker_process"]["nonce"] = "2" * 32
        self.assertTrue(evaluate_p8.repeat_exact(initial, repeated)["passed"])
        repeated["capture_hashes"]["audit_sha256"] = "f" * 64
        self.assertFalse(evaluate_p8.repeat_exact(initial, repeated)["passed"])

    def test_artifact_rejects_per_conversation_or_raw_route_material(self) -> None:
        for key in ("sessions", "sample_id", "ground_truth", "route_records"):
            with self.assertRaisesRegex(evaluate_p8.P8RunnerError, "prohibited"):
                evaluate_p8._assert_artifact_safe({"nested": {key: []}})
        evaluate_p8._assert_artifact_safe({
            "scenario_hit_counts": {"buying": 1},
            "exact_totals": {"hit_count": 1},
        })


class P8ProtocolFlowTests(unittest.TestCase):
    def test_confirmation_rows_are_not_opened_when_selection_is_rejected(self) -> None:
        state = {
            "protocol": {
                "paths": {"catalog": Path("catalog")},
                "source": {},
            },
            "spec": {
                "worker_factory": "starter.p8_lab:create_p8_agent",
                "roles": evaluate_p8.ROLES,
                "served_control": {},
                "resource_limits": {},
            },
            "summary": {},
            "git": {},
            "worker_boundary": {},
            "identity_snapshot": {},
        }
        gates = {
            evaluate_p8.BASELINE_ROLE: {"decision": "served_reference"},
            evaluate_p8.ROLES["control"]: {"decision": "control"},
            evaluate_p8.ROLES["shadow"]: {"decision": "shadow_only"},
            evaluate_p8.ROLES["active"]: {"decision": "reject"},
        }
        split_loader = Mock(return_value=([{}], {"sample_count": 1}, {"asin"}))
        with patch.object(evaluate_p8, "preflight", return_value=state), patch.object(
            evaluate_p8, "catalog_index", return_value=(set(), {}, {})
        ), patch.object(
            evaluate_p8, "_prior_target_set", return_value=set()
        ), patch.object(
            evaluate_p8, "_load_split", split_loader
        ), patch.object(
            evaluate_p8, "_run_initial_split", return_value=({}, gates)
        ):
            artifact = evaluate_p8.run_evaluation()
        self.assertEqual(split_loader.call_count, 1)
        self.assertEqual(split_loader.call_args.args[0], "selection")
        self.assertFalse(artifact["confirmation"]["opened"])
        self.assertEqual(artifact["decision"], "retain_p8_c00")

    def test_cli_exposes_non_metric_dry_preflight(self) -> None:
        args = evaluate_p8._parser().parse_args(["--dry-preflight"])
        self.assertTrue(args.dry_preflight)


if __name__ == "__main__":
    unittest.main()
