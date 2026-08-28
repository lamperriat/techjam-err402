from __future__ import annotations

import argparse
import ast
import os
import re
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import p9_worker


class _CaptureAgent:
    def export_p9_blind_capture(self) -> dict:
        counters = {key: 0 for key in p9_worker.COUNTER_KEYS}
        return {
            "schema_version": "p9.compact-negative-lab.v1",
            "role": "P9.C00.r08_coverage",
            "configuration": {
                "retrieval_mode": "coverage",
                "rerank_mode": "off",
                "question_policy": "fast",
                "target_blind": True,
                "label_free": True,
                "spec_sha256": "a" * 64,
                "lock_sha256": "b" * 64,
                "protocol_spec_sha256": "c" * 64,
                "spec_schema_version": "p9.worker-spec.v1",
                "lock_schema_version": "p9.worker-lock.v1",
                "evidence_opened": False,
            },
            "stats": {
                "schema_version": "p9.compact-negative-lab.v1",
                "evidence_schema_version": "p9.compact-negative-evidence.v1",
                "spec": p9_worker._expected_experiment_spec("P9.C00.r08_coverage"),
                "frozen_parameters": dict(p9_worker.COMPACT_PARAMETERS),
                **counters,
                "partition_totals": {
                    "compatible": 0,
                    "unknown": 0,
                    "explicit_violation": 0,
                },
                "reason_counts": {},
                "rejection_counts": {},
            },
            "integrity_errors": [],
            "hashes": {"audit_sha256": "d" * 64, "responses_sha256": "e" * 64},
            "function_hashes": {key: "f" * 64 for key in p9_worker.FUNCTION_HASH_KEYS},
        }


class P9WorkerIsolationTests(unittest.TestCase):
    def test_source_and_ast_exclude_parent_only_vocabulary_and_imports(self) -> None:
        source = Path(p9_worker.__file__).read_text(encoding="utf-8")
        forbidden = (
            "ground_truth", "target", "sample_id", "scenario", "results",
            "label", "evaluator", "selection", "confirmation", "public_set",
        )
        for word in forbidden:
            self.assertIsNone(re.search(rf"\b{re.escape(word)}\b", source, re.IGNORECASE), word)
        tree = ast.parse(source)
        imported = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported.update(
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        self.assertNotIn("evaluator", imported)

    def test_argv_yields_only_the_eight_bootstrap_inputs(self) -> None:
        args = p9_worker.parse_args([
            "--role", "P9.C00.r08_coverage",
            "--nonce", "a" * 32,
            "--factory", "m:f",
            "--catalog", "c",
            "--evidence", "e",
            "--spec", "s",
            "--lock", "l",
            "--rss-ms", "10",
        ])
        self.assertEqual(
            set(vars(args)),
            {"role", "nonce", "factory", "catalog", "evidence", "spec", "lock", "rss_ms"},
        )

    def test_extra_child_namespace_is_rejected_before_bootstrap(self) -> None:
        args = argparse.Namespace(
            role="P9.C00.r08_coverage",
            nonce="a" * 32,
            factory="m:f",
            catalog=Path("c"),
            evidence=Path("e"),
            spec=Path("s"),
            lock=Path("l"),
            rss_ms=10.0,
            extra=True,
        )
        with self.assertRaisesRegex(p9_worker.WorkerError, "namespace"):
            p9_worker.run(args)

    def test_nonce_and_unknown_role_are_rejected(self) -> None:
        base = dict(
            role="P9.C00.r08_coverage",
            nonce="bad",
            factory="m:f",
            catalog=Path("c"),
            evidence=Path("e"),
            spec=Path("s"),
            lock=Path("l"),
            rss_ms=10.0,
        )
        with self.assertRaisesRegex(p9_worker.WorkerError, "nonce"):
            p9_worker.run(argparse.Namespace(**base))
        base.update(role="unknown", nonce="a" * 32)
        with self.assertRaisesRegex(p9_worker.WorkerError, "namespace"):
            p9_worker.run(argparse.Namespace(**base))

    def test_capture_accepts_only_aggregate_strict_schema(self) -> None:
        captured = p9_worker._capture(_CaptureAgent(), "P9.C00.r08_coverage", 0)
        self.assertEqual(set(captured), p9_worker.CAPTURE_KEYS)

        class Extra(_CaptureAgent):
            def export_p9_blind_capture(self) -> dict:
                value = super().export_p9_blind_capture()
                value["extra"] = []
                return value

        with self.assertRaisesRegex(p9_worker.WorkerError, "root schema"):
            p9_worker._capture(Extra(), "P9.C00.r08_coverage", 0)

        class WrongRole(_CaptureAgent):
            def export_p9_blind_capture(self) -> dict:
                value = super().export_p9_blind_capture()
                value["role"] = "P9.S00.compact_negative_shadow"
                return value

        with self.assertRaisesRegex(p9_worker.WorkerError, "differs"):
            p9_worker._capture(WrongRole(), "P9.C00.r08_coverage", 0)

        class NestedExtra(_CaptureAgent):
            def export_p9_blind_capture(self) -> dict:
                value = super().export_p9_blind_capture()
                value["configuration"]["extra"] = {"arbitrary": []}
                return value

        with self.assertRaisesRegex(p9_worker.WorkerError, "configuration"):
            p9_worker._capture(NestedExtra(), "P9.C00.r08_coverage", 0)

    def test_protocol_requests_reject_every_extra_input(self) -> None:
        request = {
            "request_id": 1,
            "operation": "respond",
            "ordinal": 1,
            "turn": 1,
            "user_message": "hello",
            "top_k": 10,
        }
        parsed, operation = p9_worker._validate_request(request)
        self.assertEqual(parsed, request)
        self.assertEqual(operation, "respond")
        with self.assertRaisesRegex(p9_worker.WorkerError, "strict schema"):
            p9_worker._validate_request({**request, "extra": "blocked"})

    def test_network_guard_counts_and_denies_attempts(self) -> None:
        guard = p9_worker.NetworkGuard()
        with self.assertRaisesRegex(OSError, "disabled"):
            guard._deny()
        self.assertEqual(guard.attempt_count, 1)

    def test_network_guard_covers_datagram_and_name_resolution(self) -> None:
        guard = p9_worker.NetworkGuard()
        guard.install()
        try:
            with self.assertRaisesRegex(OSError, "disabled"):
                socket.getaddrinfo("localhost", 80)
            sock = socket.socket()
            try:
                with self.assertRaisesRegex(OSError, "disabled"):
                    sock.sendto(b"x", ("127.0.0.1", 9))
            finally:
                sock.close()
        finally:
            guard.restore()
        self.assertEqual(guard.attempt_count, 2)

    def test_runtime_boundary_rejects_os_open_write_flags_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog"
            evidence = root / "evidence"
            spec = root / "spec"
            lock = root / "lock"
            for path in (catalog, evidence, spec, lock):
                path.write_bytes(b"frozen")
            boundary = p9_worker.RuntimeBoundary(
                root,
                (catalog, evidence, spec, lock),
                evidence_path=evidence,
            )
            with self.assertRaisesRegex(PermissionError, "denied"):
                boundary._audit("open", (catalog, None, os.O_WRONLY | os.O_TRUNC))
            with self.assertRaisesRegex(PermissionError, "disabled"):
                boundary._audit("subprocess.Popen", ())
            self.assertEqual(catalog.read_bytes(), b"frozen")
            self.assertEqual(boundary.read_denied_attempt_count, 1)
            self.assertEqual(boundary.process_denied_attempt_count, 1)

    def test_latency_summary_uses_nearest_rank_p95(self) -> None:
        values = [value * 1_000_000 for value in range(1, 21)]
        summary = p9_worker._latency_summary(values)
        self.assertEqual(summary["count"], 20)
        self.assertEqual(summary["p95_ms"], 19.0)
        self.assertEqual(summary["max_ms"], 20.0)

    def test_windows_rss_selects_os_lifetime_peak_field(self) -> None:
        source = Path(p9_worker.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_windows_peak_rss_bytes"
        )
        returned = {
            node.attr for node in ast.walk(function)
            if isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and node.attr.endswith("WorkingSetSize")
        }
        self.assertEqual(returned, {"PeakWorkingSetSize"})
        with patch.object(p9_worker, "_windows_peak_rss_bytes", return_value=123):
            self.assertEqual(
                p9_worker._rss_bytes(),
                (123, "Windows GetProcessMemoryInfo PeakWorkingSetSize"),
            )

    @unittest.skipUnless(os.name == "nt", "Windows process counters integration")
    def test_windows_process_peak_is_positive(self) -> None:
        peak = p9_worker._windows_peak_rss_bytes()
        self.assertIsInstance(peak, int, p9_worker._WINDOWS_RSS_ERROR)
        self.assertGreater(peak, 0)


if __name__ == "__main__":
    unittest.main()
