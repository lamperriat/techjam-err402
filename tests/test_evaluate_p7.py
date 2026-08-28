from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_p7 import (
    C00,
    S00,
    P7RunnerError,
    WorkerClient,
    _canonical_json_sha256,
    _eligible_coordinates,
    canonical_json_line,
    canonical_sha256,
    dense_canonical_records,
    initial_gates_pass,
    integrity_gates,
    normalize_response_records,
    normalize_route_records,
    posthoc_recall,
    repeatability_gates,
    resource_gates,
    target_blind_alignment,
    validate_index_lock,
    validate_worker_processes,
)
from scripts.p7_worker import NetworkGuard


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dense(prefix: str = "D") -> list[dict[str, str]]:
    return [
        {"parent_asin": f"{prefix}{index:03d}", "score": float(120 - index).hex()}
        for index in range(120)
    ]


class FakeP7Agent:
    def __init__(self, role: str) -> None:
        self.role = role
        self.routes: list[dict] = []
        self.responses: list[dict] = []

    def reset(self, ordinal: int, user_profile: dict) -> None:
        if not isinstance(ordinal, int) or not isinstance(user_profile, dict):
            raise AssertionError("worker leaked a non-ordinal session coordinate")

    def respond(self, ordinal: int, user_message: str, turn: int, top_k: int) -> dict:
        query = "" if user_message == "EMPTY" else user_message.lower()
        dense = _dense() if self.role == S00 and query else []
        response = {
            "message": "ok",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": "A"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        self.routes.append(
            {
                "ordinal": ordinal,
                "turn": turn,
                "query": query,
                "empty_query": not bool(query),
                "broad": ["A", "B"],
                "strict": ["A"],
                "dense": dense,
                "query_search_ns": 1_000_000,
            }
        )
        self.responses.append(
            {"ordinal": ordinal, "turn": turn, "response": response}
        )
        return response

    def semantic_stats(self) -> dict:
        empty = sum(record["empty_query"] for record in self.routes)
        return {
            "schema_version": "fake",
            "role": self.role,
            "route_record_count": len(self.routes),
            "response_record_count": len(self.responses),
            "dense_call_count": len(self.routes) - empty if self.role == S00 else 0,
            "empty_query_count": empty,
            "semantic_exception_count": 0,
            "capture_exception_count": 0,
            "integrity_error_count": 0,
            "cold_initialization_seconds": 0.01 if self.role == S00 else None,
            "required_asset_bytes": 100 if self.role == S00 else 0,
        }

    def export_target_blind_capture(self) -> dict:
        stable = [
            {key: value for key, value in record.items() if key != "query_search_ns"}
            for record in self.routes
        ]
        dense = [
            {
                "ordinal": record["ordinal"],
                "turn": record["turn"],
                "query": record["query"],
                "dense": record["dense"],
            }
            for record in self.routes
        ]
        return {
            "schema_version": "fake",
            "role": self.role,
            "target_blind": True,
            "label_free": True,
            "route_records": self.routes,
            "response_records": self.responses,
            "integrity_errors": [],
            "stats": self.semantic_stats(),
            "hashes": {
                "routes_sha256": canonical_sha256(stable),
                "dense_routes_sha256": canonical_sha256(dense),
                "responses_sha256": canonical_sha256(self.responses),
            },
        }

    def close(self) -> None:
        return None


def create_fake_p7_agent(**kwargs: object) -> FakeP7Agent:
    expected = {"role", "catalog_path", "spec_path", "model_dir", "index_dir", "lock_path"}
    if set(kwargs) != expected:
        raise AssertionError(f"unexpected worker bootstrap keys: {set(kwargs)}")
    return FakeP7Agent(str(kwargs["role"]))


def _bundle(role: str, messages: tuple[str, ...] = ("RED",)) -> dict:
    agent = FakeP7Agent(role)
    agent.reset(1, {})
    for turn, message in enumerate(messages, start=1):
        agent.respond(1, message, turn, 10)
    exported = agent.export_target_blind_capture()
    return {
        "role": role,
        "response_records": exported["response_records"],
        "route_records": exported["route_records"],
        "semantic_stats": exported["stats"],
        "lab_integrity_errors": [],
        "lab_hashes": exported["hashes"],
        "generic_exception_count": 0,
        "network_attempt_count": 0,
        "memory": {
            "available": True,
            "absolute_peak_rss_bytes": 100 if role == C00 else 120,
        },
        "evaluation_wall_seconds": 1.0 if role == C00 else 1.2,
    }


class P7RunnerTest(unittest.TestCase):
    def test_direct_script_bootstrap_exposes_project_imports(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "evaluate_p7.py"
        probe = (
            "import runpy,sys;"
            f"ns=runpy.run_path({str(script)!r},run_name='p7_import_probe');"
            "assert str(ns['PROJECT_ROOT']) in sys.path;"
            "import evaluator.local_evaluator"
        )
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", probe],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )

    def test_canonical_jsonl_is_utf8_sorted_compact_lf_and_rejects_identity_keys(self) -> None:
        self.assertEqual(canonical_json_line({"z": "\u00e9", "a": 1}), b'{"a":1,"z":"\xc3\xa9"}\n')
        with self.assertRaisesRegex(P7RunnerError, "prohibited"):
            canonical_json_line({"ordinal": 1, "nested": {"session_id": "secret"}})

    def test_target_blind_alignment_integrity_and_repeat_hashes(self) -> None:
        control, shadow = _bundle(C00), _bundle(S00)
        alignment = target_blind_alignment(control, shadow)
        self.assertTrue(alignment["passed"])
        integrity = integrity_gates(
            control, shadow, {"A", "B", *(f"D{index:03d}" for index in range(120))}
        )
        self.assertTrue(integrity["passed"], integrity)
        repeat = _bundle(S00)
        repeated = repeatability_gates(control, shadow, repeat, {(1, 1)})
        self.assertTrue(repeated["passed"])

        repeat["route_records"][0]["dense"][0]["score"] = float(999).hex()
        self.assertFalse(
            repeatability_gates(control, shadow, repeat, {(1, 1)})["passed"]
        )

    def test_integrity_rejects_dense_depth_contract_and_bad_response_membership(self) -> None:
        control, shadow = _bundle(C00), _bundle(S00)
        shadow["route_records"][0]["dense"].pop()
        shadow["response_records"][0]["response"]["recommendations"] = [
            {"parent_asin": "OUTSIDE"}
        ]
        gate = integrity_gates(
            control, shadow, {"A", "B", *(f"D{index:03d}" for index in range(120))}
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["shadow_nonempty_routes_dense120"])
        self.assertFalse(gate["checks"]["shadow_strict_response_contract"])

    def test_posthoc_override_filter_and_dense_recall_cutoffs(self) -> None:
        samples = [
            {
                "sample_id": "one",
                "scenario_type": "buying",
                "ground_truth": {"parent_asin": "T1"},
                "intent_card": {},
                "behavior": {},
            },
            {
                "sample_id": "two",
                "scenario_type": "intent_override",
                "ground_truth": {"parent_asin": "T2"},
                "intent_card": {},
                "behavior": {"override": {"turn": 2}},
            },
        ]
        routes = [
            {
                "ordinal": 1,
                "turn": 1,
                "query": "q1",
                "empty_query": False,
                "broad": ["A"],
                "strict": [],
                "dense": [
                    {"parent_asin": value, "score": float(120 - index).hex()}
                    for index, value in enumerate([*(["X"] * 9), "T1"])
                ],
                "query_search_ns": 1,
            },
            {
                "ordinal": 2,
                "turn": 1,
                "query": "old",
                "empty_query": False,
                "broad": [],
                "strict": [],
                "dense": [{"parent_asin": "T2", "score": "0x1.0p+0"}],
                "query_search_ns": 1,
            },
            {
                "ordinal": 2,
                "turn": 2,
                "query": "new",
                "empty_query": False,
                "broad": [],
                "strict": [],
                "dense": [{"parent_asin": "T2", "score": "0x1.0p+0"}],
                "query_search_ns": 1,
            },
        ]
        eligible = _eligible_coordinates(samples, {}, routes)
        self.assertEqual(eligible, {(1, 1), (2, 2)})
        recall = posthoc_recall(samples, routes, eligible)
        self.assertEqual(recall["rescued_session_count"], 2)
        self.assertEqual(recall["rescued_scenario_type_count"], 2)
        self.assertEqual(recall["dense_recalled_session_count_at_k"]["10"], 2)
        self.assertFalse(recall["per_target_identifiers_recorded"])

    def test_resource_gate_uses_only_eligible_nonempty_route_latencies(self) -> None:
        control, shadow = _bundle(C00, ("RED", "EMPTY")), _bundle(S00, ("RED", "EMPTY"))
        spec = {
            "evaluation": {
                "resource_gates": {
                    "required_asset_bytes_max": 200,
                    "cold_initialization_seconds_max": 1.0,
                    "query_search_p95_milliseconds_max": 40.0,
                    "shadow_to_control_evaluation_wall_ratio_max": 1.5,
                    "shadow_to_control_absolute_peak_rss_ratio_max": 1.5,
                    "semantic_exception_count_max": 0,
                    "network_attempt_count_max": 0,
                }
            }
        }
        gate = resource_gates(spec, control, shadow, {(1, 1)})
        self.assertTrue(gate["passed"], gate)
        self.assertEqual(gate["observed"]["query_search_p95_milliseconds"], 1.0)

    def test_initial_gate_authorizes_repeat_only_when_every_gate_passes(self) -> None:
        passed = {"passed": True}
        self.assertTrue(initial_gates_pass(passed, passed, passed, passed))
        self.assertFalse(initial_gates_pass(passed, {"passed": False}, passed, passed))
        workers = validate_worker_processes(
            {"role": C00, "worker_process": {"isolated": True, "role": C00, "nonce": "a" * 32}},
            {"role": S00, "worker_process": {"isolated": True, "role": S00, "nonce": "b" * 32}},
        )
        self.assertTrue(workers["passed"])

    def test_network_guard_blocks_and_counts_socket_connect(self) -> None:
        guard = NetworkGuard()
        guard.install()
        try:
            with self.assertRaisesRegex(OSError, "disabled"):
                socket_value = __import__("socket").socket()
                try:
                    socket_value.connect(("127.0.0.1", 9))
                finally:
                    socket_value.close()
            self.assertEqual(guard.attempt_count, 1)
        finally:
            guard.restore()

    def test_fresh_worker_receives_only_ordinal_visible_turn_and_bootstrap_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stderr = root / "stderr.log"
            client = WorkerClient.start(
                S00,
                catalog=root / "catalog",
                spec=root / "spec",
                model_dir=root / "model",
                index_dir=root / "index",
                index_lock=root / "lock",
                worker_factory="tests.test_evaluate_p7:create_fake_p7_agent",
                stderr_path=stderr,
            )
            pid = client.process.pid
            client.reset("opaque_uuid_that_must_not_be_serialized", {})
            response = client.respond(
                "opaque_uuid_that_must_not_be_serialized", "RED", 1, 10
            )
            self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
            bundle = client.finalize()
            self.assertGreater(pid, 0)
            serialized = json.dumps(bundle, sort_keys=True)
            self.assertNotIn("opaque_uuid", serialized)
            self.assertNotIn('"pid"', serialized)
            self.assertEqual(bundle["network_attempt_count"], 0)
            self.assertEqual(bundle["response_records"][0]["ordinal"], 1)

    def test_strict_index_lock_bridges_every_artifact_and_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "starter").mkdir()
            (root / "configs").mkdir()
            (root / "data").mkdir()
            index_dir = root / "index"
            index_dir.mkdir()
            builder = root / "scripts" / "builder.py"
            semantic = root / "starter" / "semantic.py"
            builder.write_text("builder\n", encoding="utf-8")
            semantic.write_text("semantic\n", encoding="utf-8")
            spec_path = root / "configs" / "spec.json"
            spec = {
                "index": {"shape": [1, 2]},
                "evaluation": {"resource_gates": {"required_asset_bytes_max": 1000}},
            }
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            catalog = root / "data" / "catalog.jsonl"
            catalog.write_text('{"parent_asin":"A"}\n', encoding="utf-8")
            matrix = index_dir / "matrix.npy"
            asins = index_dir / "asins.txt"
            matrix.write_bytes(b"matrix")
            asins.write_bytes(b"A\n")
            observation = {
                "wall_seconds": 1.0,
                "rss_backend": "test",
                "baseline_rss_bytes": 10,
                "peak_rss_bytes": 20,
                "peak_delta_from_baseline_bytes": 10,
            }
            manifest = {
                "schema_version": "p7.semantic-index.v1",
                "model_spec_sha256": _canonical_json_sha256(spec),
                "catalog_sha256": _sha(catalog),
                "matrix": {
                    "path": "matrix.npy",
                    "bytes": matrix.stat().st_size,
                    "sha256": _sha(matrix),
                    "dtype": "float32",
                    "shape": [1, 2],
                },
                "ordered_asins": {
                    "path": "asins.txt",
                    "bytes": asins.stat().st_size,
                    "sha256": _sha(asins),
                    "count": 1,
                },
                "preprocessing": {"canonical_documents_sha256": "d" * 64},
                "asset_byte_scope": {"required_asset_bytes": 100},
                "build_resources": observation,
            }
            manifest_path = index_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            lock = {
                "schema_version": "p7.semantic-index-lock.v1",
                "source": {
                    "git_commit": "a" * 40,
                    "git_branch": "pre",
                    "builder": {"path": "scripts/builder.py", "bytes": builder.stat().st_size, "sha256": _sha(builder)},
                    "semantic": {"path": "starter/semantic.py", "bytes": semantic.stat().st_size, "sha256": _sha(semantic)},
                },
                "model_spec": {
                    "path": "configs/spec.json",
                    "raw_bytes": spec_path.stat().st_size,
                    "raw_sha256": _sha(spec_path),
                    "canonical_sha256": _canonical_json_sha256(spec),
                },
                "catalog": {"path": "data/catalog.jsonl", "bytes": catalog.stat().st_size, "sha256": _sha(catalog), "rows": 1},
                "index": {
                    "directory": "index",
                    "manifest": {"path": "manifest.json", "bytes": manifest_path.stat().st_size, "sha256": _sha(manifest_path), "schema_version": "p7.semantic-index.v1"},
                    "matrix": manifest["matrix"],
                    "ordered_asins": {**manifest["ordered_asins"], "encoding": "utf-8-lf", "line_ending": "LF"},
                    "canonical_documents_sha256": "d" * 64,
                },
                "asset_scope": {"required_asset_bytes": 100, "required_asset_bytes_max": 1000},
                "build_observation": observation,
            }
            lock_path = root / "configs" / "lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            summary = validate_index_lock(
                lock_path,
                project_root=root,
                spec_path=spec_path,
                catalog_path=catalog,
                index_dir=index_dir,
                enforce_git=False,
            )
            self.assertEqual(summary["matrix_sha256"], _sha(matrix))
            matrix.write_bytes(b"drift")
            with self.assertRaisesRegex(P7RunnerError, "identity"):
                validate_index_lock(
                    lock_path,
                    project_root=root,
                    spec_path=spec_path,
                    catalog_path=catalog,
                    index_dir=index_dir,
                    enforce_git=False,
                )


if __name__ == "__main__":
    unittest.main()
