from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

from scripts.build_p11_sidecar import build_sidecar
from starter.p11_features import (
    MAX_DECOMPRESSED_FEATURE_BYTES,
    SQL_NEGATIVE_MASK_COLUMNS,
    P11FeatureStore,
)


def product(
    identifier: str,
    *,
    title: str,
    categories: list[str],
    color: str,
    description: str,
    rating: float,
    rating_count: int,
) -> dict[str, object]:
    return {
        "parent_asin": identifier,
        "title": title,
        "features": [f"{color} cotton", "waterproof lightweight"],
        "description": [description],
        "price": 25.0,
        "categories": categories,
        "details": {"Color": color, "Material": "cotton"},
        "average_rating": rating,
        "rating_number": rating_count,
        "store": "Example Store",
    }


class BuildP11SidecarTests(unittest.TestCase):
    def _catalog(self, directory: Path) -> Path:
        products = (
            product(
                "A",
                title="Rare red hoop earrings",
                categories=["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Earrings", "Hoop"],
                color="red",
                description="rare statement earrings for a beach wedding",
                rating=4.8,
                rating_count=100,
            ),
            product(
                "B",
                title="Common blue stud earrings",
                categories=["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Earrings", "Stud"],
                color="blue",
                description="common everyday earrings",
                rating=4.0,
                rating_count=20,
            ),
            product(
                "C",
                title="Common black running shoe",
                categories=["Clothing, Shoes & Jewelry", "Women", "Shoes", "Running Shoe"],
                color="black",
                description="common cushioned running shoe",
                rating=3.8,
                rating_count=5,
            ),
        )
        path = directory / "catalog.jsonl"
        path.write_text(
            "".join(
                json.dumps(item, sort_keys=True) + "\n"
                for item in products
            ),
            encoding="utf-8",
        )
        return path

    def test_catalog_only_builder_store_and_bounded_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            catalog = self._catalog(directory)
            sidecar = directory / "features.sqlite"
            metadata_path = directory / "features.metadata.json"
            metadata = build_sidecar(
                catalog,
                sidecar,
                metadata_path,
                expected_catalog_sha256=None,
                expected_catalog_rows=3,
            )

            self.assertTrue(sidecar.is_file())
            self.assertEqual(metadata["catalog"]["rows"], 3)
            self.assertEqual(
                metadata["sidecar"]["sha256"],
                hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            connection = sqlite3.connect(sidecar)
            try:
                columns = tuple(
                    row[1]
                    for row in connection.execute("PRAGMA table_info(evidence)")
                )
                self.assertEqual(
                    columns,
                    (
                        "catalog_rowid",
                        "parent_asin",
                        "feature_blob",
                        *SQL_NEGATIVE_MASK_COLUMNS,
                        "bayesian_rating_percentile",
                        "popularity_percentile",
                    ),
                )
                payload = connection.execute(
                    "SELECT feature_blob FROM evidence WHERE catalog_rowid=1"
                ).fetchone()[0]
                self.assertIsInstance(payload, bytes)
                term_columns = tuple(
                    row[1]
                    for row in connection.execute("PRAGMA table_info(term_stats)")
                )
                self.assertEqual(
                    term_columns,
                    ("term", "document_frequency"),
                )
            finally:
                connection.close()
            store = P11FeatureStore(
                sidecar,
                expected_catalog_sha256=metadata["catalog"]["sha256"],
                expected_catalog_rows=3,
            )
            try:
                batch = store.fetch_top10(
                    ((1, "A"), (2, "B")),
                    ("rare", "common"),
                )
                self.assertEqual(set(batch.evidence), {"A", "B"})
                self.assertGreater(batch.idf_by_term["rare"], batch.idf_by_term["common"])
                self.assertAlmostEqual(
                    batch.idf_by_term["rare"],
                    math.log(1.0 + (3 - 1 + 0.5) / (1 + 0.5)),
                )
                self.assertIn("rare", batch.evidence["A"].field_tokens[0])
                self.assertIn("color=red", batch.evidence["A"].observed_values)
                self.assertIn("hoop", batch.evidence["A"].observed_subtypes)
                self.assertIn("hoop", store.resolve_query_subtypes("hoop earrings"))
                with self.assertRaisesRegex(ValueError, "Top 10"):
                    store.fetch_top10(
                        tuple((1, f"A-{index}") for index in range(11)),
                        (),
                    )
                with self.assertRaisesRegex(ValueError, "binding"):
                    store.fetch_top10(((1, "WRONG"),), ())
            finally:
                store.close()

    def test_corrupt_and_oversize_feature_blobs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "byte limit"):
            P11FeatureStore._decode_feature_blob(
                zlib.compress(b"x" * (MAX_DECOMPRESSED_FEATURE_BYTES + 1))
            )

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            catalog = self._catalog(directory)
            sidecar = directory / "features.sqlite"
            metadata_path = directory / "features.metadata.json"
            metadata = build_sidecar(
                catalog,
                sidecar,
                metadata_path,
                expected_catalog_sha256=None,
                expected_catalog_rows=3,
            )
            connection = sqlite3.connect(sidecar)
            try:
                connection.execute(
                    "UPDATE evidence SET feature_blob=? WHERE catalog_rowid=1",
                    (sqlite3.Binary(b"not-a-zlib-stream"),),
                )
                connection.commit()
            finally:
                connection.close()
            store = P11FeatureStore(
                sidecar,
                expected_catalog_sha256=metadata["catalog"]["sha256"],
                expected_catalog_rows=3,
            )
            try:
                with self.assertRaisesRegex(ValueError, "compressed UTF-8"):
                    store.fetch_top10(((1, "A"),), ())
            finally:
                store.close()

    def test_builder_is_deterministic_for_the_same_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            catalog = self._catalog(directory)
            hashes: list[str] = []
            for suffix in ("one", "two"):
                metadata = build_sidecar(
                    catalog,
                    directory / f"{suffix}.sqlite",
                    directory / f"{suffix}.json",
                    expected_catalog_sha256=None,
                    expected_catalog_rows=3,
                )
                hashes.append(str(metadata["sidecar"]["sha256"]))
            self.assertEqual(hashes[0], hashes[1])

    def test_builder_never_overwrites_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            catalog = self._catalog(directory)
            sidecar = directory / "features.sqlite"
            metadata_path = directory / "features.metadata.json"
            sidecar.write_bytes(b"owned-by-user")

            with self.assertRaises(FileExistsError):
                build_sidecar(
                    catalog,
                    sidecar,
                    metadata_path,
                    expected_catalog_sha256=None,
                    expected_catalog_rows=3,
                )

            self.assertEqual(sidecar.read_bytes(), b"owned-by-user")
            self.assertFalse(metadata_path.exists())


if __name__ == "__main__":
    unittest.main()
