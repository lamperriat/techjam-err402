from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_agent import _parser, main, run_evaluation


RESULT = {
    "sample_count": 1,
    "hit_rate_at_10": 1.0,
    "mrr": 0.5,
    "mttc": 2.0,
    "efficiency": 0.9,
    "recommended_technical_score": 0.83,
    "reported_token_usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    },
    "scenario_metrics": {},
    "sessions": [{"sample_id": "sample", "hit": True}],
}


class EvaluateAgentTest(unittest.TestCase):
    def test_cli_defaults_to_off_reranking(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.rerank_mode, "off")
        self.assertEqual(args.question_policy, "fast")

    def test_run_propagates_mode_and_closes_agent(self) -> None:
        constructions: list[dict[str, object]] = []

        class Connection:
            closed = False

            def close(self) -> None:
                self.closed = True

        class FakeAgent:
            def __init__(self, catalog_path: Path, **kwargs: object) -> None:
                self.connection = Connection()
                constructions.append({
                    "catalog_path": catalog_path,
                    "kwargs": kwargs,
                    "agent": self,
                })

        with (
            patch("scripts.evaluate_agent.load_jsonl", return_value=[{"sample_id": "s"}]),
            patch(
                "scripts.evaluate_agent.catalog_index",
                return_value=({"P"}, {"P": ["Clothing"]}, {"P": {}}),
            ),
            patch("scripts.evaluate_agent.Agent", FakeAgent),
            patch("scripts.evaluate_agent.evaluate", return_value=RESULT) as evaluate_mock,
        ):
            result, elapsed = run_evaluation(
                Path("catalog.jsonl"),
                Path("dataset.jsonl"),
                question_policy="boundary",
                rerank_mode="shadow",
            )

        self.assertIs(result, RESULT)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(
            constructions[0]["kwargs"],
            {
                "llm_client": None,
                "question_policy": "boundary",
                "rerank_mode": "shadow",
            },
        )
        self.assertTrue(constructions[0]["agent"].connection.closed)
        evaluate_mock.assert_called_once()

    def test_main_writes_raw_evaluator_result_and_provenance_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            dataset = root / "dataset.jsonl"
            output = root / "active.json"
            catalog.write_text('{}\n', encoding="utf-8")
            dataset.write_text('{}\n', encoding="utf-8")

            with patch(
                "scripts.evaluate_agent.run_evaluation",
                return_value=(RESULT, 1.25),
            ) as run_mock:
                exit_code = main([
                    "--catalog",
                    str(catalog),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--rerank-mode",
                    "active",
                ])

            manifest_path = root / "active.manifest.json"
            written_result = json.loads(output.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(written_result, RESULT)
        self.assertNotIn("schema_version", written_result)
        self.assertEqual(manifest["schema_version"], "p2.evaluate-agent.v1")
        self.assertEqual(manifest["configuration"]["rerank_mode"], "active")
        self.assertEqual(manifest["run"]["metrics"]["hit_rate_at_10"], 1.0)
        self.assertEqual(len(manifest["implementation"]["agent_source_sha256"]), 64)
        self.assertEqual(len(manifest["implementation"]["attribute_source_sha256"]), 64)
        self.assertEqual(len(manifest["implementation"]["reranker_source_sha256"]), 64)
        run_mock.assert_called_once_with(
            catalog,
            dataset,
            question_policy="fast",
            rerank_mode="active",
        )


if __name__ == "__main__":
    unittest.main()
