from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


class QueryEmbedder(Protocol):
    def encode_query(self, text: str) -> Sequence[float]:
        ...


@dataclass(frozen=True)
class DenseMatch:
    parent_asin: str
    similarity: float


class QwenQueryEmbedder:
    """Lazy local query encoder matching the offline Qwen catalog vectors."""

    def __init__(
        self,
        model_name: str,
        dimensions: int,
        max_length: int,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Embedding retrieval requires the packages in requirements.txt"
            ) from error

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        torch_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"torch_dtype": torch_dtype},
            truncate_dim=dimensions,
        )
        self.model.max_seq_length = max_length

    def encode_query(self, text: str) -> Sequence[float]:
        return self.model.encode(
            [text],
            prompt_name="query",
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]


class ProductEmbeddingIndex:
    """Validated exact-search index over normalized catalog embeddings."""

    def __init__(
        self,
        directory: str | Path,
        expected_product_ids: Sequence[str],
        query_embedder: QueryEmbedder | None = None,
    ) -> None:
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("Embedding retrieval requires numpy") from error

        self.directory = Path(directory)
        manifest = json.loads(
            (self.directory / "manifest.json").read_text(encoding="utf-8")
        )
        configuration = manifest.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError("Embedding manifest has no configuration")
        runtime = manifest.get("runtime")
        if not isinstance(runtime, dict) or runtime.get("normalized_embeddings") is not True:
            raise ValueError("Catalog embeddings must be normalized")

        self.product_ids = json.loads(
            (self.directory / "product_ids.json").read_text(encoding="utf-8")
        )
        if self.product_ids != list(expected_product_ids):
            raise ValueError("Embedding product IDs do not match catalog order")
        if len(self.product_ids) != len(set(self.product_ids)):
            raise ValueError("Embedding product IDs are not unique")

        self.vectors = np.load(
            self.directory / "catalog_embeddings.npy",
            mmap_mode="r",
        )
        dimensions = int(configuration.get("dimensions", 0))
        if self.vectors.shape != (len(self.product_ids), dimensions):
            raise ValueError(
                f"Embedding shape {self.vectors.shape} does not match manifest"
            )
        self.dimensions = dimensions
        self.query_embedder = query_embedder or QwenQueryEmbedder(
            model_name=str(configuration["model"]),
            dimensions=dimensions,
            max_length=int(configuration["max_length"]),
        )

    def search(self, query: str, limit: int) -> list[DenseMatch]:
        import numpy as np

        if limit <= 0:
            return []
        vector = np.asarray(self.query_embedder.encode_query(query), dtype=np.float32)
        if vector.shape != (self.dimensions,):
            raise ValueError(
                f"Query embedding shape {vector.shape} does not match ({self.dimensions},)"
            )
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm == 0.0 or not np.isfinite(vector).all():
            raise ValueError("Query embedding must contain finite, nonzero values")
        vector /= norm

        similarities = self.vectors @ vector
        count = min(limit, len(self.product_ids))
        if count == len(self.product_ids):
            indices = np.arange(count)
        else:
            indices = np.argpartition(similarities, len(similarities) - count)[-count:]
        ordered = sorted(
            indices.tolist(),
            key=lambda index: (
                -float(similarities[index]),
                self.product_ids[index],
            ),
        )
        return [
            DenseMatch(self.product_ids[index], float(similarities[index]))
            for index in ordered
        ]
