from __future__ import annotations

import json
from pathlib import Path

from agents.v1 import AgentV1
from retrieval.scoring import ScoringConfig


DEFAULT_CONFIG_PATH = Path("results/v1_tuning/best_config.json")


class AgentV1Tuned(AgentV1):
    """V1 using parameters learned by the benchmark tuning pipeline."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Tuned V1 config not found at {path}; run "
                "benchmark_training/tune_v1.py first"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(
            payload.get("scoring"), dict
        ):
            raise ValueError(f"Invalid tuned V1 config: {path}")
        super().__init__(catalog_path, ScoringConfig.from_dict(payload["scoring"]))
