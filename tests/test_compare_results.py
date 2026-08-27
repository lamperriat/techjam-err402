from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from scripts.compare_results import differences


class CompareResultsTest(unittest.TestCase):
    def _run(self, *paths: Path) -> subprocess.CompletedProcess[str]:
        script = Path(__file__).resolve().parents[1] / "scripts" / "compare_results.py"
        return subprocess.run(
            [sys.executable, str(script), "--assert-equal", *(str(path) for path in paths)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_line_endings_and_formatting_are_semantically_equal(self) -> None:
        payload = {"sample_count": 1, "sessions": [{"sample_id": "one", "best_rank": 1}]}
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "expected.json"
            actual = Path(directory) / "actual.json"
            expected.write_bytes((json.dumps(payload, indent=2) + "\n").encode())
            actual.write_bytes((json.dumps(payload, separators=(",", ":")) + "\r\n").encode())
            result = self._run(expected, actual)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("STRICT MATCH", result.stdout)

    def test_changed_session_fails_even_when_aggregate_is_same(self) -> None:
        expected_payload = {
            "hit_rate_at_10": 1.0,
            "sessions": [{"sample_id": "one", "best_rank": 1}],
        }
        actual_payload = {
            "hit_rate_at_10": 1.0,
            "sessions": [{"sample_id": "one", "best_rank": 2}],
        }
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "expected.json"
            actual = Path(directory) / "actual.json"
            expected.write_text(json.dumps(expected_payload), encoding="utf-8")
            actual.write_text(json.dumps(actual_payload), encoding="utf-8")
            result = self._run(expected, actual)

        self.assertEqual(result.returncode, 1)
        self.assertIn("$.sessions[0].best_rank", result.stderr)

    def test_missing_key_and_reordered_list_are_reported(self) -> None:
        found = differences(
            {
                "usage.total": {"total": 0},
                "sessions": ["one", "two"],
                "precise": Decimal("0.100000000000000005"),
            },
            {
                "usage.total": {},
                "sessions": ["two", "one"],
                "precise": Decimal("0.1"),
            },
        )
        self.assertTrue(any("missing from actual" in item for item in found))
        self.assertTrue(any("$.sessions[0]" in item for item in found))
        self.assertTrue(any('$["usage.total"].total' in item for item in found))
        self.assertTrue(any("$.precise" in item for item in found))

    def test_invalid_json_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "expected.json"
            actual = Path(directory) / "actual.json"
            expected.write_text("{}", encoding="utf-8")
            for invalid in ("not-json", '{"value": NaN}', '{"value": 1, "value": 2}'):
                with self.subTest(invalid=invalid):
                    actual.write_text(invalid, encoding="utf-8")
                    result = self._run(expected, actual)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Cannot read", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
