from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_official_assets import (
    EXPECTED_FIELDS,
    catalog_summary,
    git_blob_sha1,
    public_summary,
)


class OfficialAssetVerifierTests(unittest.TestCase):
    def test_git_blob_hash_normalizes_windows_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.txt"
            path.write_bytes(b"one\r\ntwo\r\n")
            payload = b"one\ntwo\n"
            expected = hashlib.sha1(
                f"blob {len(payload)}\0".encode("ascii") + payload
            ).hexdigest()
            self.assertEqual(git_blob_sha1(path), expected)

    def test_catalog_and_public_summaries_detect_integrity_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            product = {field: None for field in EXPECTED_FIELDS}
            product.update({"parent_asin": "A", "title": "Example"})
            catalog_path.write_text(
                json.dumps(product) + "\n" + json.dumps(product) + "\n",
                encoding="utf-8",
            )
            catalog, identifiers = catalog_summary(catalog_path)
            self.assertEqual(catalog["row_count"], 2)
            self.assertEqual(catalog["duplicate_id_count"], 1)
            self.assertEqual(catalog["schema_mismatch_count"], 0)

            public_path = root / "public.jsonl"
            sample = {
                "sample_id": "sample",
                "scenario_type": "buying",
                "ground_truth": {"parent_asin": "MISSING"},
            }
            public_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            public = public_summary(public_path, identifiers)
            self.assertEqual(public["missing_target_count"], 1)
            self.assertEqual(public["scenario_counts"], {"buying": 1})


if __name__ == "__main__":
    unittest.main()
