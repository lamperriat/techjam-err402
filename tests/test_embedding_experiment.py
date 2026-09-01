from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from embedding.build_catalog_embeddings import (
    build_product_text,
    load_documents,
    main,
)


class EmbeddingExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.catalog_path = self.directory / "catalog.jsonl"
        self.attributes_path = self.directory / "attributes.jsonl"
        self.catalog_path.write_text(
            json.dumps(
                {
                    "parent_asin": "A1",
                    "title": "Trail Runner",
                    "categories": ["Clothing, Shoes & Jewelry", "Shoes", "Running"],
                    "store": "Example",
                    "details": {"Department": "Womens"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.attributes_path.write_text(
            json.dumps({"record_type": "metadata"})
            + "\n"
            + json.dumps(
                {
                    "parent_asin": "A1",
                    "attributes": {
                        "material": [{"value": "mesh", "evidence": "mesh upper"}],
                        "color": [],
                        "size_fit": [],
                        "style": [{"value": "low top", "evidence": "low top"}],
                        "use_case": [{"value": "trail running", "evidence": "trail"}],
                        "specific_attributes": [
                            {
                                "name": "traction",
                                "value": "lugged outsole",
                                "evidence": "lugged outsole",
                            }
                        ],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_build_product_text_uses_values_but_not_evidence(self) -> None:
        product = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        record = json.loads(self.attributes_path.read_text(encoding="utf-8").splitlines()[1])

        text = build_product_text(product, record)

        self.assertIn("title: Trail Runner", text)
        self.assertIn("material: mesh", text)
        self.assertIn("specific attributes: traction: lugged outsole", text)
        self.assertNotIn("mesh upper", text)

    def test_load_documents_validates_matching_product_ids(self) -> None:
        documents = load_documents(self.catalog_path, self.attributes_path)
        self.assertEqual([document.parent_asin for document in documents], ["A1"])

        self.attributes_path.write_text(
            json.dumps({"record_type": "metadata"}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Processed attributes missing for A1"):
            load_documents(self.catalog_path, self.attributes_path)

    def test_dry_run_needs_no_embedding_dependencies_or_output(self) -> None:
        output = StringIO()
        output_directory = self.directory / "output"
        with redirect_stdout(output):
            return_code = main(
                [
                    "--catalog",
                    str(self.catalog_path),
                    "--attributes",
                    str(self.attributes_path),
                    "--output-dir",
                    str(output_directory),
                    "--dry-run",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(return_code, 0)
        self.assertTrue(payload["dry_run"])
        self.assertFalse(payload["model_was_loaded"])
        self.assertEqual(payload["documents"]["products"], 1)
        self.assertFalse(output_directory.exists())

    def test_limit_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "--limit must be positive"):
            main(
                [
                    "--catalog",
                    str(self.catalog_path),
                    "--attributes",
                    str(self.attributes_path),
                    "--limit",
                    "0",
                    "--dry-run",
                ]
            )


if __name__ == "__main__":
    unittest.main()
