from __future__ import annotations

import argparse
import ast
import re
import unittest
import os
import ctypes
from pathlib import Path
from unittest.mock import patch

from scripts import p7_worker


class P7WorkerIsolationTests(unittest.TestCase):
    def test_source_and_ast_exclude_parent_only_vocabulary_and_imports(self) -> None:
        source = Path(p7_worker.__file__).read_text(encoding="utf-8")
        forbidden = (
            "selection", "output", "ground_truth", "scenario", "sample_id",
            "target", "materialize", "evaluator", "posthoc",
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

    def test_argv_yields_exact_required_namespace(self) -> None:
        args = p7_worker.parse_args([
            "--role", "P7.C00.r08_coverage", "--nonce", "a" * 32,
            "--factory", "m:f", "--catalog", "c", "--spec", "s",
            "--model", "m", "--index", "i", "--lock", "l", "--rss-ms", "10",
        ])
        self.assertEqual(
            set(vars(args)),
            {"role", "nonce", "factory", "catalog", "spec", "model", "index", "lock", "rss_ms"},
        )

    def test_extra_child_namespace_is_rejected_before_bootstrap(self) -> None:
        args = argparse.Namespace(
            role="P7.C00.r08_coverage", nonce="a" * 32, factory="m:f",
            catalog=Path("c"), spec=Path("s"), model=Path("m"), index=Path("i"),
            lock=Path("l"), rss_ms=10.0, extra=True,
        )
        with self.assertRaisesRegex(p7_worker.WorkerError, "namespace"):
            p7_worker.run(args)

    def test_windows_rss_selects_os_lifetime_peak_field(self) -> None:
        source = Path(p7_worker.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_windows_peak_rss_bytes"
        )
        returned_attributes = {
            node.attr for node in ast.walk(function)
            if isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and node.attr.endswith("WorkingSetSize")
        }
        self.assertEqual(returned_attributes, {"PeakWorkingSetSize"})
        with patch.object(p7_worker, "_windows_peak_rss_bytes", return_value=123):
            self.assertEqual(
                p7_worker._rss_bytes(),
                (123, "Windows GetProcessMemoryInfo PeakWorkingSetSize"),
            )

    @unittest.skipUnless(os.name == "nt", "Windows process counters integration")
    def test_windows_process_peak_is_real_and_not_below_current_working_set(self) -> None:
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        self.assertTrue(psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb))
        self.assertGreaterEqual(
            int(counters.PeakWorkingSetSize), int(counters.WorkingSetSize)
        )
        # Read the implementation peak after the comparison baseline. Constructing the
        # ctypes wrappers itself may grow the current working set by one page.
        peak = p7_worker._windows_peak_rss_bytes()
        self.assertIsInstance(peak, int, p7_worker._WINDOWS_RSS_ERROR)
        self.assertGreater(peak, 0)
        self.assertGreaterEqual(peak, int(counters.WorkingSetSize))
        observed, backend = p7_worker._rss_bytes()
        self.assertIsInstance(observed, int)
        self.assertIn("PeakWorkingSetSize", backend)


if __name__ == "__main__":
    unittest.main()
