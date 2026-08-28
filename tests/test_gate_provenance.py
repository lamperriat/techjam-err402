from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.gate_provenance import (
    assert_gate_snapshot_stable,
    hash_snapshot,
    sha256,
)


class GateProvenanceTest(unittest.TestCase):
    def test_hash_snapshot_is_named_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.txt"
            path.write_text("stable", encoding="utf-8")
            result = hash_snapshot({"source": path})
            expected = sha256(path)

        self.assertEqual(result, {"source": expected})

    def test_stability_check_detects_source_change(self) -> None:
        snapshot = {
            "git": {"commit": "A", "branch": "test", "dirty": False},
            "source_sha256": {"agent": "A"},
            "input_sha256": {"catalog": "B"},
            "selection_evidence": {"valid": True},
        }
        changed = {
            **snapshot,
            "source_sha256": {"agent": "CHANGED"},
        }
        with self.assertRaisesRegex(RuntimeError, "source_sha256"):
            assert_gate_snapshot_stable(snapshot, changed)


if __name__ == "__main__":
    unittest.main()
