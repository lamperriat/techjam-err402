from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agents.registry import agent_names
from agents.v2 import V2SessionState
from agents.v2_embedding import AgentV2Embedding
from retrieval.embeddings import DenseMatch
from tests.test_v2 import attributes, product


class FakeEmbedder:
    def encode_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class AgentV2EmbeddingTest(unittest.TestCase):
    def _paths(self, directory: Path) -> tuple[Path, Path, Path]:
        catalog_path = directory / "catalog.jsonl"
        attributes_path = directory / "attributes.jsonl"
        embeddings_path = directory / "embeddings"
        embeddings_path.mkdir()
        products = [product(index) for index in range(6)]
        catalog_path.write_text(
            "".join(json.dumps(item) + "\n" for item in products),
            encoding="utf-8",
        )
        attributes_path.write_text(
            json.dumps({"record_type": "metadata"})
            + "\n"
            + "".join(json.dumps(attributes(index)) + "\n" for index in range(6)),
            encoding="utf-8",
        )
        vectors = np.asarray(
            [[-1.0, 0.0] for _ in range(5)] + [[1.0, 0.0]],
            dtype=np.float32,
        )
        np.save(embeddings_path / "catalog_embeddings.npy", vectors)
        (embeddings_path / "product_ids.json").write_text(
            json.dumps([item["parent_asin"] for item in products]),
            encoding="utf-8",
        )
        (embeddings_path / "manifest.json").write_text(
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
        return catalog_path, attributes_path, embeddings_path

    def test_dense_candidate_is_reranked_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            agent = AgentV2Embedding(
                *paths,
                query_embedder=FakeEmbedder(),
                embedding_weights={"buying": 0.15, "browsing": 0.25},
                embedding_score_mode="minmax",
                embedding_requires_constraints=False,
            )
            self.addCleanup(agent.close)
            ranked = agent._rank_products(
                V2SessionState(user_profile={}, intent="browsing")
            )

        self.assertEqual(ranked[0].product.parent_asin, "P5")
        self.assertEqual(ranked[0].components["embedding"], 1.0)

    def test_hybrid_ranking_applies_malformed_category_tier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            agent = AgentV2Embedding(*paths, query_embedder=FakeEmbedder())
            self.addCleanup(agent.close)
            with patch.object(
                agent,
                "_prioritize_exact_category",
                wraps=agent._prioritize_exact_category,
            ) as prioritize:
                agent._rank_products(
                    V2SessionState(
                        user_profile={},
                        category="Shoes & Jewelry Women",
                    )
                )

        prioritize.assert_called_once()
        self.assertEqual(prioritize.call_args.args[1], "Shoes & Jewelry Women")

    def test_agent_is_registered_separately(self) -> None:
        self.assertIn("v2", agent_names())
        self.assertIn("v2-embedding", agent_names())

    def test_dense_score_modes_have_different_calibration(self) -> None:
        matches = [
            DenseMatch("A", 0.8),
            DenseMatch("B", 0.7),
            DenseMatch("C", 0.6),
        ]

        minmax = AgentV2Embedding._dense_scores(matches, "minmax")
        margin = AgentV2Embedding._dense_scores(matches, "margin")

        self.assertAlmostEqual(minmax["A"], 1.0)
        self.assertAlmostEqual(minmax["B"], 0.5)
        self.assertAlmostEqual(margin["A"], 0.2)
        self.assertAlmostEqual(margin["B"], 0.1)
        self.assertEqual(margin["C"], 0.0)

    def test_embedding_can_require_a_customer_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            agent = AgentV2Embedding(
                *paths,
                query_embedder=FakeEmbedder(),
                embedding_requires_constraints=True,
            )
            union_only = AgentV2Embedding(
                *paths,
                query_embedder=FakeEmbedder(),
                embedding_weights={"buying": 0.0, "browsing": 0.0},
            )
            self.addCleanup(agent.close)
            self.addCleanup(union_only.close)
            ranked = agent._rank_products(
                V2SessionState(user_profile={}, intent="browsing")
            )
            expected = union_only._rank_products(
                V2SessionState(user_profile={}, intent="browsing")
            )

        self.assertEqual(
            [(item.product.parent_asin, item.score) for item in ranked],
            [(item.product.parent_asin, item.score) for item in expected],
        )


if __name__ == "__main__":
    unittest.main()
