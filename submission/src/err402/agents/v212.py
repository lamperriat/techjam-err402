from __future__ import annotations

from pathlib import Path

from .v212_runtime.agent import Agent as V212RuntimeAgent


ASSET_DIR = Path(__file__).with_name("v212_runtime") / "assets"


class AgentV212(V212RuntimeAgent):
    """Frozen v2.12 ranking stack with P11 and two-page pagination active."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        super().__init__(
            catalog_path,
            llm_client=None,
            p11_mode="active",
            p11_sidecar_path=ASSET_DIR / "p11_features.sqlite",
            small_ranker_mode="active",
            small_ranker_artifact_path=ASSET_DIR / "small_ranker_fold_safe_v1.json",
            pagination_mode="active",
        )


__all__ = ["AgentV212"]
