from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from retrieval.embeddings import ProductEmbeddingIndex


class FakeEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def encode_query(self, text: str) -> list[float]:
        return self.vector


class ProductEmbeddingIndexTest(unittest.TestCase):
    def _directory(self, root: Path, ids: list[str]) -> Path:
        directory = root / "embeddings"
        directory.mkdir()
        vectors = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]][: len(ids)],
            dtype=np.float32,
        )
        np.save(directory / "catalog_embeddings.npy", vectors)
        (directory / "product_ids.json").write_text(
            json.dumps(ids),
            encoding="utf-8",
        )
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "configuration": {
                        "model": "fake",
                        "dimensions": 2,
                        "max_length": 16,
                    },
                    "runtime": {"normalized_embeddings": True},
                }
            ),
            encoding="utf-8",
        )
        return directory

    def test_search_normalizes_query_and_returns_descending_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._directory(Path(temporary), ["A", "B", "C"])
            index = ProductEmbeddingIndex(
                directory,
                ["A", "B", "C"],
                FakeEmbedder([2.0, 0.0]),
            )

            matches = index.search("query", limit=2)

        self.assertEqual([match.parent_asin for match in matches], ["A", "B"])
        self.assertAlmostEqual(matches[0].similarity, 1.0)
        self.assertAlmostEqual(matches[1].similarity, 0.0)

    def test_rejects_catalog_order_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._directory(Path(temporary), ["A", "B"])
            with self.assertRaisesRegex(ValueError, "do not match catalog order"):
                ProductEmbeddingIndex(
                    directory,
                    ["B", "A"],
                    FakeEmbedder([1.0, 0.0]),
                )

    def test_rejects_invalid_query_vector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._directory(Path(temporary), ["A", "B"])
            index = ProductEmbeddingIndex(
                directory,
                ["A", "B"],
                FakeEmbedder([0.0, 0.0]),
            )
            with self.assertRaisesRegex(ValueError, "finite, nonzero"):
                index.search("query", limit=1)


if __name__ == "__main__":
    unittest.main()
