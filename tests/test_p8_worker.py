from __future__ import annotations

import argparse
import ast
import ctypes
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import p8_worker


class _CaptureAgent:
    def export_p8_blind_capture(self) -> dict:
        return {
            "schema_version": "p8.test.v1",
            "role": "P8.C00.r08_coverage",
            "configuration": {},
            "stats": {},
            "integrity_errors": [],
            "hashes": {},
            "function_hashes": {},
        }


class P8WorkerIsolationTests(unittest.TestCase):
    def test_source_and_ast_exclude_parent_only_vocabulary_and_imports(self) -> None:
        source = Path(p8_worker.__file__).read_text(encoding="utf-8")
        forbidden = (
            "ground_truth", "target", "sample_id", "scenario", "results",
            "label", "evaluator", "selection", "confirmation",
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

    def test_argv_yields_only_the_seven_bootstrap_inputs(self) -> None:
        args = p8_worker.parse_args([
            "--role", "P8.C00.r08_coverage",
            "--nonce", "a" * 32,
            "--factory", "m:f",
            "--catalog", "c",
            "--spec", "s",
            "--lock", "l",
            "--rss-ms", "10",
        ])
        self.assertEqual(
            set(vars(args)),
            {"role", "nonce", "factory", "catalog", "spec", "lock", "rss_ms"},
        )

    def test_extra_child_namespace_is_rejected_before_bootstrap(self) -> None:
        args = argparse.Namespace(
            role="P8.C00.r08_coverage",
            nonce="a" * 32,
            factory="m:f",
            catalog=Path("c"),
            spec=Path("s"),
            lock=Path("l"),
            rss_ms=10.0,
            extra=True,
        )
        with self.assertRaisesRegex(p8_worker.WorkerError, "namespace"):
            p8_worker.run(args)

    def test_nonce_and_unknown_role_are_rejected(self) -> None:
        base = dict(
            role="P8.C00.r08_coverage",
            nonce="bad",
            factory="m:f",
            catalog=Path("c"),
            spec=Path("s"),
            lock=Path("l"),
            rss_ms=10.0,
        )
        with self.assertRaisesRegex(p8_worker.WorkerError, "nonce"):
            p8_worker.run(argparse.Namespace(**base))
        base.update(role="unknown", nonce="a" * 32)
        with self.assertRaisesRegex(p8_worker.WorkerError, "namespace"):
            p8_worker.run(argparse.Namespace(**base))

    def test_capture_accepts_only_aggregate_strict_schema(self) -> None:
        captured = p8_worker._capture(
            _CaptureAgent(), "P8.C00.r08_coverage", 0
        )
        self.assertEqual(set(captured), p8_worker.CAPTURE_KEYS)

        class Extra(_CaptureAgent):
            def export_p8_blind_capture(self) -> dict:
                value = super().export_p8_blind_capture()
                value["extra"] = []
                return value

        with self.assertRaisesRegex(p8_worker.WorkerError, "root schema"):
            p8_worker._capture(Extra(), "P8.C00.r08_coverage", 0)

    def test_network_guard_counts_and_denies_attempts(self) -> None:
        guard = p8_worker.NetworkGuard()
        with self.assertRaisesRegex(OSError, "disabled"):
            guard._deny()
        self.assertEqual(guard.attempt_count, 1)

    def test_latency_summary_uses_nearest_rank_p95(self) -> None:
        values = [value * 1_000_000 for value in range(1, 21)]
        summary = p8_worker._latency_summary(values)
        self.assertEqual(summary["count"], 20)
        self.assertEqual(summary["p95_ms"], 19.0)
        self.assertEqual(summary["max_ms"], 20.0)

    def test_windows_rss_selects_os_lifetime_peak_field(self) -> None:
        source = Path(p8_worker.__file__).read_text(encoding="utf-8")
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
        with patch.object(p8_worker, "_windows_peak_rss_bytes", return_value=123):
            self.assertEqual(
                p8_worker._rss_bytes(),
                (123, "Windows GetProcessMemoryInfo PeakWorkingSetSize"),
            )

    @unittest.skipUnless(os.name == "nt", "Windows process counters integration")
    def test_windows_process_peak_is_positive(self) -> None:
        peak = p8_worker._windows_peak_rss_bytes()
        self.assertIsInstance(peak, int, p8_worker._WINDOWS_RSS_ERROR)
        self.assertGreater(peak, 0)


if __name__ == "__main__":
    unittest.main()
