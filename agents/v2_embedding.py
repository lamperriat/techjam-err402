from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from agents.v2 import AgentV2, V2SessionState
from retrieval.catalog import CandidatePool
from retrieval.embeddings import DenseMatch, ProductEmbeddingIndex, QueryEmbedder
from retrieval.scoring import ScoredProduct


DENSE_CANDIDATE_LIMIT = 1000
EmbeddingScoreMode = Literal["minmax", "margin"]
EMBEDDING_WEIGHTS = {
    "buying": 0.05,
    "browsing": 0.08,
}


class AgentV2Embedding(AgentV2):
    """V2 with hybrid lexical/category and dense candidate retrieval."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        attributes_path: str | Path = "results/catalog_attributes_processed.jsonl",
        embeddings_path: str | Path = "results/embeddings/qwen3_embedding_0_6b",
        query_embedder: QueryEmbedder | None = None,
        embedding_weights: dict[str, float] | None = None,
        embedding_score_mode: EmbeddingScoreMode = "margin",
        embedding_requires_constraints: bool = True,
    ) -> None:
        if embedding_score_mode not in ("minmax", "margin"):
            raise ValueError("embedding_score_mode must be 'minmax' or 'margin'")
        self.embedding_weights = dict(embedding_weights or EMBEDDING_WEIGHTS)
        if set(self.embedding_weights) != {"buying", "browsing"} or any(
            not 0.0 <= weight < 1.0 for weight in self.embedding_weights.values()
        ):
            raise ValueError("embedding weights must define buying and browsing in [0, 1)")
        self.embedding_score_mode = embedding_score_mode
        self.embedding_requires_constraints = embedding_requires_constraints
        super().__init__(catalog_path, attributes_path)
        try:
            self.embedding_index = ProductEmbeddingIndex(
                embeddings_path,
                tuple(self.catalog.products),
                query_embedder,
            )
        except Exception:
            self.close()
            raise

    def _rank_products(self, state: V2SessionState) -> list[ScoredProduct]:
        query_text = self._query_text(state)
        lexical_pool = self.catalog.candidates(state.category, query_text)
        dense_matches = self.embedding_index.search(query_text, DENSE_CANDIDATE_LIMIT)
        dense_ids = [match.parent_asin for match in dense_matches]
        hybrid_pool = CandidatePool(
            tuple(dict.fromkeys([*lexical_pool.parent_asins, *dense_ids])),
            lexical_pool.lexical_ranks,
        )
        base_ranked = self._score_pool(state, hybrid_pool)

        dense_similarities = {
            match.parent_asin: match.similarity for match in dense_matches
        }
        dense_scores = self._dense_scores(dense_matches, self.embedding_score_mode)
        embedding_weight = (
            self.embedding_weights[state.intent]
            if state.constraints or not self.embedding_requires_constraints
            else 0.0
        )
        reranked = [
            ScoredProduct(
                item.product,
                (1.0 - embedding_weight) * item.score
                + embedding_weight * dense_scores.get(item.product.parent_asin, 0.0),
                {
                    **item.components,
                    "embedding_similarity": dense_similarities.get(
                        item.product.parent_asin,
                        0.0,
                    ),
                    "embedding": dense_scores.get(item.product.parent_asin, 0.0),
                },
            )
            for item in base_ranked
        ]
        reranked.sort(
            key=lambda item: (
                -item.score,
                -item.components["embedding"],
                lexical_pool.lexical_ranks.get(item.product.parent_asin, math.inf),
                item.product.parent_asin,
            )
        )
        return self._prioritize_exact_category(reranked, state.category)

    @staticmethod
    def _dense_scores(
        matches: list[DenseMatch],
        mode: EmbeddingScoreMode,
    ) -> dict[str, float]:
        if not matches:
            return {}
        if len(matches) == 1:
            return {matches[0].parent_asin: 1.0}
        minimum = matches[-1].similarity
        if mode == "margin":
            return {
                match.parent_asin: max(0.0, match.similarity - minimum)
                for match in matches
            }
        spread = matches[0].similarity - minimum
        if spread <= 1e-12:
            return {}
        return {
            match.parent_asin: (match.similarity - minimum) / spread
            for match in matches
        }
