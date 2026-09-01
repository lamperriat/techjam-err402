from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from profile_catalog_categories import profile_catalog_categories


class ProfileCatalogCategoriesTest(unittest.TestCase):
    def test_counts_labels_nodes_and_department_coverage(self) -> None:
        category_paths = [
            ["Root", "Women", "Watches", "Wrist Watches"],
            ["Root", "Men", "Watches", "Wrist Watches"],
            ["Root", "Promotion", "Wrist Watches"],
            ["Root", "Women", "Shoes"],
            [],
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                "".join(
                    json.dumps({"categories": categories}) + "\n"
                    for categories in category_paths
                ),
                encoding="utf-8",
            )
            result = profile_catalog_categories(catalog_path, top=5)

        self.assertEqual(result["products"], 5)
        self.assertEqual(result["products_without_categories"], 1)
        self.assertEqual(result["path_depth_distribution"], {3: 2, 4: 2})
        self.assertEqual(result["finest_level"]["unique_labels"], 2)
        self.assertEqual(result["finest_level"]["unique_nodes"], 4)
        self.assertEqual(result["second_last_level"]["unique_labels"], 3)
        self.assertEqual(result["second_last_level"]["unique_nodes"], 4)

        audit = result["second_level_department_audit"]
        self.assertEqual(audit["products_with_second_level"], 4)
        self.assertEqual(audit["department_products"], 3)
        self.assertEqual(audit["department_fraction"], 0.75)
        self.assertFalse(audit["is_always_department"])
        self.assertEqual(audit["non_department_unique_labels"], 1)

    def test_rejects_non_positive_top_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "top must be positive"):
            profile_catalog_categories(Path("unused.jsonl"), top=0)


if __name__ == "__main__":
    unittest.main()
