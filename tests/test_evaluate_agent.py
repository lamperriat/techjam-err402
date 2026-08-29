from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_agent import (
    _close_agent,
    _parser,
    _resolve_cli_p11_mode,
    main,
    run_evaluation,
)


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
        self.assertIsNone(args.retrieval_mode)
        self.assertEqual(args.p11_mode, "active")
        self.assertIsNone(args.p11_sidecar)

    def test_explicit_legacy_cli_flags_default_p11_to_off(self) -> None:
        parser = _parser()
        for raw in (
            ["--question-policy", "fast"],
            ["--rerank-mode", "off"],
            ["--retrieval-mode", "coverage"],
            ["--question-policy=fast"],
            ["--rerank-mode=off"],
            ["--retrieval-mode=coverage"],
        ):
            with self.subTest(raw=raw):
                self.assertEqual(
                    _resolve_cli_p11_mode(parser.parse_args(raw), list(raw)),
                    "off",
                )

        raw = ["--retrieval-mode", "coverage", "--p11-mode", "active"]
        self.assertEqual(
            _resolve_cli_p11_mode(parser.parse_args(raw), raw),
            "active",
        )

    def test_programmatic_legacy_arguments_default_p11_to_off(self) -> None:
        observed_modes: list[str] = []

        class FakeAgent:
            def __init__(self, _catalog: Path, **kwargs: object) -> None:
                observed_modes.append(str(kwargs["p11_mode"]))

            def close(self) -> None:
                return None

            def _p11_status(self) -> dict[str, object]:
                mode = observed_modes[-1]
                return {
                    "configured_mode": mode,
                    "effective_mode": mode,
                    "fallback": False,
                    "reason_code": "ready",
                    "identity_verified": mode == "active",
                }

        common = (
            patch("scripts.evaluate_agent.load_jsonl", return_value=[]),
            patch(
                "scripts.evaluate_agent.catalog_index",
                return_value=(set(), {}, {}),
            ),
            patch("scripts.evaluate_agent.Agent", FakeAgent),
            patch("scripts.evaluate_agent.evaluate", return_value=RESULT),
        )
        with common[0], common[1], common[2], common[3]:
            default_result = run_evaluation(
                Path("catalog.jsonl"), Path("dataset.jsonl")
            )
            run_evaluation(
                Path("catalog.jsonl"),
                Path("dataset.jsonl"),
                question_policy="fast",
            )

        self.assertEqual(observed_modes, ["active", "off"])
        self.assertEqual(len(default_result), 2)

    def test_run_propagates_mode_and_closes_agent(self) -> None:
        constructions: list[dict[str, object]] = []

        class Connection:
            closed = False

            def close(self) -> None:
                self.closed = True

        class FakeAgent:
            def __init__(self, catalog_path: Path, **kwargs: object) -> None:
                self.connection = Connection()
                self.close_calls = 0
                constructions.append({
                    "catalog_path": catalog_path,
                    "kwargs": kwargs,
                    "agent": self,
                })

            def close(self) -> None:
                self.close_calls += 1

            def _p11_status(self) -> dict[str, object]:
                mode = str(constructions[-1]["kwargs"]["p11_mode"])
                return {
                    "configured_mode": mode,
                    "effective_mode": mode,
                    "fallback": False,
                    "reason_code": "disabled" if mode == "off" else "ready",
                    "identity_verified": mode in {"shadow", "active"},
                    "schema_version": "p11.production-bridge.v1",
                    "feature_schema_version": None,
                    "scorer_version": None,
                    "feature_registry_sha256": None,
                    "feature_semantics_sha256": None,
                }

        with (
            patch("scripts.evaluate_agent.load_jsonl", return_value=[{"sample_id": "s"}]),
            patch(
                "scripts.evaluate_agent.catalog_index",
                return_value=({"P"}, {"P": ["Clothing"]}, {"P": {}}),
            ),
            patch("scripts.evaluate_agent.Agent", FakeAgent),
            patch("scripts.evaluate_agent.evaluate", return_value=RESULT) as evaluate_mock,
        ):
            result, elapsed, p11_status = run_evaluation(
                Path("catalog.jsonl"),
                Path("dataset.jsonl"),
                question_policy="boundary",
                rerank_mode="shadow",
                retrieval_mode="control",
                p11_mode="off",
                _include_p11_status=True,
            )

        self.assertIs(result, RESULT)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(p11_status["effective_mode"], "off")
        self.assertEqual(
            constructions[0]["kwargs"],
            {
                "llm_client": None,
                "question_policy": "boundary",
                "rerank_mode": "shadow",
                "retrieval_mode": "control",
                "p11_mode": "off",
                "p11_sidecar_path": None,
            },
        )
        self.assertEqual(constructions[0]["agent"].close_calls, 1)
        self.assertFalse(constructions[0]["agent"].connection.closed)
        evaluate_mock.assert_called_once()

    def test_close_agent_falls_back_to_legacy_connection(self) -> None:
        class Connection:
            closed = False

            def close(self) -> None:
                self.closed = True

        class LegacyAgent:
            connection = Connection()

        agent = LegacyAgent()
        _close_agent(agent)
        self.assertTrue(agent.connection.closed)

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
                return_value=(
                    RESULT,
                    1.25,
                    {
                        "configured_mode": "off",
                        "effective_mode": "off",
                        "fallback": False,
                        "reason_code": "disabled",
                        "identity_verified": False,
                    },
                ),
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
        self.assertEqual(manifest["schema_version"], "p11.evaluate-agent.v3")
        self.assertEqual(manifest["configuration"]["rerank_mode"], "active")
        self.assertEqual(manifest["configuration"]["retrieval_mode"], "control")
        self.assertEqual(manifest["configuration"]["p11_mode"], "off")
        self.assertEqual(manifest["run"]["p11"]["effective_mode"], "off")
        self.assertEqual(manifest["run"]["metrics"]["hit_rate_at_10"], 1.0)
        self.assertEqual(len(manifest["implementation"]["agent_source_sha256"]), 64)
        self.assertEqual(len(manifest["implementation"]["attribute_source_sha256"]), 64)
        self.assertEqual(len(manifest["implementation"]["reranker_source_sha256"]), 64)
        self.assertEqual(len(manifest["implementation"]["slot_ledger_source_sha256"]), 64)
        self.assertEqual(len(manifest["implementation"]["clarification_source_sha256"]), 64)
        run_mock.assert_called_once_with(
            catalog,
            dataset,
            question_policy="fast",
            rerank_mode="active",
            retrieval_mode="control",
            p11_mode="off",
            p11_sidecar_path=None,
            _include_p11_status=True,
        )


if __name__ == "__main__":
    unittest.main()
