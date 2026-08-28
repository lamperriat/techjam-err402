from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

from evaluator.local_evaluator import catalog_index
from observer import launcher
from observer.events import TRACE_SCHEMA_VERSION, TraceRecorder
from observer.runtime import StaleRuntimeError, WorkbenchRuntime
from observer.server import ExclusiveHTTPServer, make_handler
from observer.trace import TraceRunner, _agent_diagnostics
from starter.agent import Agent
from starter.coverage import SCHEMA_VERSION as COVERAGE_SCHEMA_VERSION


class ObserverTraceTest(unittest.TestCase):
    def test_static_workbench_exposes_p3_shadow_components(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        app = (project_root / "observer" / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        page = (project_root / "observer" / "static" / "index.html").read_text(
            encoding="utf-8"
        )

        for marker in (
            "Active slots:",
            "Retired:",
            "information_gain",
            "answerability",
            "turn_cost",
            "Candidate split:",
            "blocked_attributes",
        ):
            self.assertIn(marker, app)
        self.assertIn("slot ledger / 候选感知澄清 shadow", page)

    def test_one_click_launcher_defaults_workbench_to_coverage_off(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "TECHJAM_RETRIEVAL_MODE": "control",
                    "TECHJAM_RERANK_MODE": "active",
                },
                clear=True,
            ),
            patch.object(launcher, "_running_project", return_value=None),
            patch("observer.server.main") as serve,
        ):
            launcher.main()
            self.assertEqual(os.environ["TECHJAM_RETRIEVAL_MODE"], "coverage")
            self.assertEqual(os.environ["TECHJAM_RERANK_MODE"], "off")
            serve.assert_called_once_with()

    def test_coverage_retrieval_rejects_shadow_and_active_rerank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                json.dumps({"parent_asin": "A", "title": "Running shoe"}) + "\n",
                encoding="utf-8",
            )
            for rerank_mode in ("shadow", "active"):
                with self.subTest(rerank_mode=rerank_mode):
                    with self.assertRaisesRegex(
                        ValueError, "coverage retrieval requires rerank_mode=off"
                    ):
                        Agent(
                            catalog_path,
                            retrieval_mode="coverage",
                            rerank_mode=rerank_mode,
                        )

    def test_trace_exposes_layers_without_changing_agent_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            products = [
                {
                    "parent_asin": "A",
                    "title": "Blue cotton running shoe",
                    "features": ["cotton", "running"],
                    "categories": ["Clothing", "Shoes"],
                    "price": 49.0,
                },
                {
                    "parent_asin": "B",
                    "title": "Black leather winter boot",
                    "features": ["leather", "winter"],
                    "categories": ["Clothing", "Boots"],
                    "price": 89.0,
                },
            ]
            catalog_path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            catalog_ids, categories, product_index = catalog_index(catalog_path)
            sample = {
                "sample_id": "public_test_1",
                "scenario_type": "buying",
                "difficulty_bucket": "easy",
                "user_profile": {"summary": "Likes practical shoes"},
                "ground_truth": {"parent_asin": "A"},
            }
            recorder = TraceRecorder()
            recorder.emit({"session_id": "schema_probe", "schema_version": "untrusted"})
            self.assertEqual(
                recorder.events("schema_probe")[0]["schema_version"], TRACE_SCHEMA_VERSION
            )
            recorder.clear("schema_probe")
            runner = TraceRunner(
                Agent(
                    catalog_path,
                    trace_sink=recorder.emit,
                    rerank_mode="off",
                    retrieval_mode="coverage",
                ),
                [sample],
                catalog_ids,
                categories,
                product_index,
                recorder=recorder,
            )

            trace = runner.trace("public_test_1")
            plain_agent = Agent(
                catalog_path,
                rerank_mode="off",
                retrieval_mode="coverage",
            )
            plain_agent.reset("plain_session", sample["user_profile"])
            plain_response = plain_agent.respond(
                "plain_session", trace["turns"][0]["user_message"], 1, 10
            )
            plain_agent.connection.close()

        self.assertTrue(trace["result"]["hit"])
        self.assertEqual(trace["result"]["first_hit_turn"], 1)
        self.assertEqual(trace["result"]["diagnosis"], "SUCCESS")
        self.assertEqual(trace["turns"][0]["target_top10_rank"], 1)
        self.assertEqual(trace["turns"][0]["retrieval"]["posthoc_target_rank"], 1)
        self.assertGreaterEqual(trace["turns"][0]["retrieval"]["candidate_count"], 1)
        self.assertIn("cotton", trace["turns"][0]["retrieval"]["terms"])
        events = trace["turns"][0]["agent_events"]
        self.assertEqual(
            {event["layer"] for event in events},
            {"session", "parse", "retrieval", "state", "policy", "output"},
        )
        self.assertTrue(all(event["schema_version"] == TRACE_SCHEMA_VERSION for event in events))
        self.assertTrue(all("public_test_1" not in event["session_id"] for event in events))
        self.assertEqual(
            set(trace["turns"][0]["retrieval"]["route_counts"]),
            {"broad", "strict", "fused", "reranked", "final"},
        )
        self.assertEqual(trace["rerank_mode"], "off")
        self.assertEqual(trace["retrieval_mode"], "coverage")
        retrieval = trace["turns"][0]["retrieval"]
        self.assertEqual(retrieval["retrieval_mode"], "coverage")
        self.assertTrue(retrieval["coverage_diagnostics"]["active"])
        self.assertEqual(
            retrieval["coverage_diagnostics"]["schema_version"],
            COVERAGE_SCHEMA_VERSION,
        )
        retrieval_event = next(
            event["data"]
            for event in events
            if event["layer"] == "retrieval"
        )
        self.assertEqual(
            [item["parent_asin"] for item in retrieval_event["reranked_top_results"]],
            [item["parent_asin"] for item in retrieval_event["raw_fused_top_results"]],
        )
        self.assertTrue(retrieval_event["coverage"]["active"])
        self.assertTrue(
            all(
                item["matched_query_term_count"] is not None
                for item in retrieval_event["final_top_results"]
            )
        )
        self.assertEqual(
            trace["turns"][0]["retrieval"]["posthoc_target_rank"],
            trace["turns"][0]["retrieval"]["target_final_rank"],
        )
        self.assertEqual(runner.agent._sessions, {})
        self.assertEqual(trace["turns"][0]["validation"]["invalid_catalog_count"], 0)
        self.assertEqual(trace["turns"][0]["failure_code"], "HIT")
        self.assertEqual(
            plain_response["recommendations"],
            [
                {"parent_asin": item["parent_asin"]}
                for item in trace["turns"][0]["recommendations"]
            ],
        )
        self.assertEqual(plain_response["ask_attribute"], trace["turns"][0]["ask_attribute"])
        self.assertEqual(runner.list_sessions()["sessions"][0]["sample_id"], "public_test_1")

    def test_trace_uses_final_route_and_preserves_all_route_ranks(self) -> None:
        products = {
            "A": {
                "parent_asin": "A",
                "title": "Target running shoe",
                "categories": ["Shoes"],
            },
            "B": {
                "parent_asin": "B",
                "title": "Distractor running shoe",
                "categories": ["Shoes"],
            },
        }
        sample = {
            "sample_id": "public_route_test",
            "scenario_type": "buying",
            "difficulty_bucket": "easy",
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
        }

        class RouteAgent:
            rerank_mode = "active"
            retrieval_mode = "control"

            def reset(self, session_id: str, user_profile: dict) -> None:
                del session_id, user_profile

            def drop_session(self, session_id: str) -> None:
                del session_id

            def respond(
                self, session_id: str, user_message: str, turn: int, top_k: int
            ) -> dict:
                del session_id, user_message, turn, top_k
                return {
                    "message": "Final route response",
                    "ask_attribute": None,
                    "recommendations": [{"parent_asin": "A"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }

            def debug_snapshot(self, session_id: str) -> dict:
                del session_id
                return {"query_terms": ["shoe"]}

            def debug_rankings(self, session_id: str) -> dict[str, list[str]]:
                del session_id
                return {
                    "broad": ["A", "B"],
                    "strict": ["B", "A"],
                    "fused": ["B", "A"],
                    "reranked": ["A", "B"],
                    "final": ["A", "B"],
                }

        runner = TraceRunner(
            RouteAgent(),
            [sample],
            {"A", "B"},
            {"A": ["Shoes"], "B": ["Shoes"]},
            products,
        )
        trace = runner.trace("public_route_test")
        retrieval = trace["turns"][0]["retrieval"]
        self.assertEqual(retrieval["target_fused_rank"], 2)
        self.assertEqual(retrieval["target_reranked_rank"], 1)
        self.assertEqual(retrieval["target_final_rank"], 1)
        self.assertEqual(retrieval["posthoc_target_rank"], 1)
        self.assertEqual(retrieval["actual_route"], "final")
        self.assertEqual(retrieval["rerank_mode"], "active")
        self.assertEqual(retrieval["retrieval_mode"], "control")

        class LegacyAgent:
            def debug_rankings(self, session_id: str) -> dict[str, list[str]]:
                del session_id
                return {"broad": ["A"], "strict": [], "fused": ["B", "A"]}

        legacy = _agent_diagnostics(LegacyAgent(), "legacy", "A")
        self.assertEqual(legacy["target_final_rank"], 2)
        self.assertEqual(legacy["target_reranked_rank"], 2)
        self.assertEqual(legacy["actual_route"], "fused")
        self.assertEqual(legacy["rerank_mode"], "off")
        self.assertEqual(legacy["retrieval_mode"], "control")
        self.assertEqual(legacy["coverage_diagnostics"], {"active": False})

    def test_trace_distinguishes_control_from_coverage_final_route(self) -> None:
        products = {
            "A": {"parent_asin": "A", "title": "Target shoe", "categories": ["Shoes"]},
            "B": {"parent_asin": "B", "title": "Control shoe", "categories": ["Shoes"]},
        }
        sample = {
            "sample_id": "coverage_route_test",
            "scenario_type": "buying",
            "difficulty_bucket": "easy",
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
        }

        class CoverageRouteAgent:
            rerank_mode = "off"
            retrieval_mode = "coverage"

            def reset(self, session_id: str, user_profile: dict) -> None:
                del session_id, user_profile

            def drop_session(self, session_id: str) -> None:
                del session_id

            def respond(
                self, session_id: str, user_message: str, turn: int, top_k: int
            ) -> dict:
                del session_id, user_message, turn, top_k
                return {
                    "message": "Coverage final response",
                    "ask_attribute": None,
                    "recommendations": [{"parent_asin": "A"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }

            def debug_rankings(self, session_id: str) -> dict[str, list[str]]:
                del session_id
                return {
                    "broad": ["A", "B"],
                    "strict": ["B", "A"],
                    "fused": ["B", "A"],
                    "reranked": ["B", "A"],
                    "final": ["A", "B"],
                }

            def debug_rerank_diagnostics(self, session_id: str) -> dict:
                del session_id
                return {
                    "coverage": {
                        "schema_version": COVERAGE_SCHEMA_VERSION,
                        "active": True,
                        "query_term_count": 2,
                        "candidate_count": 2,
                        "covered_candidate_count": 2,
                        "maximum_coverage": 2,
                        "coverage_histogram": {"1": 1, "2": 1},
                        "changed_top_10": True,
                    }
                }

        runner = TraceRunner(
            CoverageRouteAgent(),
            [sample],
            {"A", "B"},
            {"A": ["Shoes"], "B": ["Shoes"]},
            products,
        )
        trace = runner.trace("coverage_route_test")
        retrieval = trace["turns"][0]["retrieval"]

        self.assertEqual(trace["retrieval_mode"], "coverage")
        self.assertEqual(retrieval["target_fused_rank"], 2)
        self.assertEqual(retrieval["target_reranked_rank"], 2)
        self.assertEqual(retrieval["target_final_rank"], 1)
        self.assertEqual(retrieval["posthoc_target_rank"], 1)
        self.assertTrue(retrieval["coverage_diagnostics"]["changed_top_10"])

    def test_workbench_exposes_data_lab_and_background_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            dataset_path = root / "public.jsonl"
            results_path = root / "results.json"
            product = {
                "parent_asin": "A",
                "title": "Blue cotton running shoe",
                "features": ["cotton", "running"],
                "categories": ["Clothing", "Shoes"],
                "price": 49.0,
            }
            sample = {
                "sample_id": "public_test_1",
                "scenario_type": "buying",
                "difficulty_bucket": "easy",
                "user_profile": {"summary": "Likes practical shoes"},
                "ground_truth": {"parent_asin": "A"},
            }
            catalog_text = json.dumps(product) + "\n"
            catalog_path.write_text(catalog_text, encoding="utf-8")
            dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            for relative_path in (
                "starter/slot_ledger.py",
                "starter/clarification.py",
                "starter/coverage.py",
                "observer/shadow_analysis.py",
            ):
                source = root / relative_path
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"# fixed {relative_path}\n", encoding="utf-8")
            runtime = WorkbenchRuntime.from_paths(
                catalog_path,
                dataset_path,
                results_path,
                project_root=root,
                rerank_mode="shadow",
                retrieval_mode="control",
            )
            self.addCleanup(runtime.close)

            overview = runtime.overview()
            self.assertEqual(overview["index"]["rows"], 1)
            self.assertEqual(overview["runtime"]["rerank_mode"], "shadow")
            self.assertEqual(overview["runtime"]["retrieval_mode"], "control")
            health = runtime.health()
            self.assertEqual(health["rerank_mode"], "shadow")
            self.assertEqual(health["retrieval_mode"], "control")
            self.assertIn("attributes", overview["source_state"]["files"])
            self.assertIn("reranker", overview["source_state"]["files"])
            self.assertIn("slot_ledger", overview["source_state"]["files"])
            self.assertIn("clarification", overview["source_state"]["files"])
            self.assertIn("coverage", overview["source_state"]["files"])
            self.assertIn("shadow_analysis", overview["source_state"]["files"])
            self.assertFalse(overview["source_state"]["files"]["coverage"]["changed"])
            self.assertEqual(
                len(overview["source_state"]["files"]["coverage"]["loaded_sha256"]),
                64,
            )
            self.assertEqual(overview["pipeline"][1]["status"], "implemented")
            self.assertEqual(overview["pipeline"][6]["status"], "implemented")
            coverage_layer = next(
                item
                for item in overview["pipeline"]
                if item["layer"] == "Query-term coverage cascade"
            )
            self.assertEqual(coverage_layer["status"], "implemented")
            self.assertEqual(coverage_layer["mode"], "control")
            rerank_layer = next(
                item
                for item in overview["pipeline"]
                if item["layer"] == "Constraint reranking"
            )
            self.assertEqual(rerank_layer["mode"], "shadow")
            clarification_layer = next(
                item
                for item in overview["pipeline"]
                if item["layer"] == "Clarification policy"
            )
            self.assertEqual(clarification_layer["status"], "implemented")
            self.assertEqual(clarification_layer["mode"], "shadow diagnostic")
            self.assertEqual(runtime.catalog("cotton")["items"][0]["parent_asin"], "A")
            self.assertEqual(runtime.product("A")["title"], product["title"])

            lab = runtime.lab_reset()
            self.assertEqual(lab["rerank_mode"], "shadow")
            self.assertEqual(lab["retrieval_mode"], "control")
            reply = runtime.lab_respond(lab["session_id"], "cotton running shoe")
            self.assertEqual(reply["rerank_mode"], "shadow")
            self.assertEqual(reply["retrieval_mode"], "control")
            self.assertEqual(reply["recommendations"][0]["parent_asin"], "A")
            self.assertIn("retrieval", {event["layer"] for event in reply["events"]})
            lab_retrieval = next(
                event["data"] for event in reply["events"] if event["layer"] == "retrieval"
            )
            self.assertEqual(lab_retrieval["retrieval_mode"], "control")
            self.assertFalse(lab_retrieval["coverage"]["active"])
            self.assertEqual(
                [item["parent_asin"] for item in lab_retrieval["final_top_results"]],
                [item["parent_asin"] for item in lab_retrieval["raw_fused_top_results"]],
            )

            job = runtime.start_evaluation()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                current = next(
                    item for item in runtime.jobs()["jobs"] if item["job_id"] == job["job_id"]
                )
                if current["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.02)
            self.assertEqual(current["status"], "completed")
            self.assertTrue(results_path.exists())
            self.assertEqual(current["summary"]["sample_count"], 1)
            self.assertEqual(current["summary"]["rerank_mode"], "shadow")
            self.assertEqual(current["summary"]["retrieval_mode"], "control")
            manifest_path = next((root / "experiments").glob("*/manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run"]["top_k"], 10)
            self.assertIn("agent_source_sha256", manifest["implementation"])
            self.assertEqual(manifest["implementation"]["question_policy"], "fast")
            self.assertEqual(manifest["implementation"]["rerank_mode"], "shadow")
            self.assertEqual(manifest["implementation"]["retrieval_mode"], "control")
            self.assertIn("attributes_source_sha256", manifest["implementation"])
            self.assertIn("reranker_source_sha256", manifest["implementation"])
            self.assertIn("slot_ledger_source_sha256", manifest["implementation"])
            self.assertIn("clarification_source_sha256", manifest["implementation"])
            self.assertEqual(
                len(manifest["implementation"]["coverage_source_sha256"]), 64
            )
            self.assertEqual(
                len(manifest["implementation"]["shadow_analysis_source_sha256"]), 64
            )
            self.assertEqual(
                len(manifest["implementation"]["slot_ledger_source_sha256"]), 64
            )
            self.assertEqual(
                len(manifest["implementation"]["clarification_source_sha256"]), 64
            )
            self.assertEqual(
                manifest["implementation"]["slot_ledger_schema_version"],
                "p3.slot-ledger.v1",
            )
            self.assertEqual(
                manifest["implementation"]["question_value_schema_version"],
                "p3.question-value.v1",
            )
            self.assertEqual(
                manifest["implementation"]["coverage_schema_version"],
                COVERAGE_SCHEMA_VERSION,
            )
            self.assertEqual(manifest["implementation"]["clarification_mode"], "shadow")
            self.assertTrue(manifest["shadow_policy_analysis"]["target_blind"])
            self.assertEqual(manifest["shadow_policy_analysis"]["turn_count"], 1)
            self.assertTrue(manifest_path.with_name("shadow_policy.json").exists())
            self.assertEqual(manifest["catalog_sha256"], overview["data"][0]["sha256"])
            self.assertEqual(manifest["dataset_sha256"], overview["data"][1]["sha256"])

            for _ in range(21):
                runtime.lab_reset()
            self.assertLessEqual(len(runtime.trace_runner.agent._sessions), 20)

            catalog_path.write_text(catalog_text + "\n", encoding="utf-8")
            self.assertTrue(runtime.overview()["source_state"]["files"]["catalog"]["changed"])
            with self.assertRaises(StaleRuntimeError):
                runtime.trace("public_test_1")
            catalog_path.write_text(catalog_text, encoding="utf-8")

            source_path = root / "starter" / "coverage.py"
            source_path.write_text("# changed after server startup\n", encoding="utf-8")
            self.assertTrue(runtime.overview()["source_state"]["restart_required"])
            self.assertTrue(
                runtime.overview()["source_state"]["files"]["coverage"]["changed"]
            )
            with self.assertRaises(StaleRuntimeError):
                runtime.start_evaluation()

    def test_unknown_sample_is_rejected(self) -> None:
        runner = TraceRunner(None, [], set(), {}, {})
        with self.assertRaises(KeyError):
            runner.trace("missing")

    def test_generalization_job_writes_manifest_and_shares_evaluation_mutex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            dataset_path = root / "public.jsonl"
            results_path = root / "results.json"
            generalization_source = root / "scripts" / "evaluate_generalization.py"
            generalization_source.parent.mkdir(parents=True)
            generalization_source.write_text("# fixed test runner\n", encoding="utf-8")
            for relative_path in (
                "starter/slot_ledger.py",
                "starter/clarification.py",
                "starter/coverage.py",
                "observer/shadow_analysis.py",
            ):
                source = root / relative_path
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"# fixed {relative_path}\n", encoding="utf-8")
            product = {
                "parent_asin": "A",
                "title": "Blue cotton running shoe",
                "features": ["cotton", "running"],
                "categories": ["Clothing", "Shoes"],
            }
            sample = {
                "sample_id": "public_test_1",
                "scenario_type": "buying",
                "difficulty_bucket": "easy",
                "user_profile": {},
                "ground_truth": {"parent_asin": "A"},
            }
            catalog_path.write_text(json.dumps(product) + "\n", encoding="utf-8")
            dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            runtime = WorkbenchRuntime.from_paths(
                catalog_path,
                dataset_path,
                results_path,
                project_root=root,
                rerank_mode="shadow",
                retrieval_mode="control",
            )
            self.addCleanup(runtime.close)

            evaluation_job, created = runtime._new_job("evaluation")
            self.assertTrue(created)
            evaluation_job.status = "running"
            blocked = runtime.start_generalization()
            self.assertEqual(blocked["job_id"], evaluation_job.job_id)
            self.assertEqual(blocked["kind"], "evaluation")
            evaluation_job.status = "completed"

            process_options: dict[str, object] = {}
            process_command: list[str] = []

            class FakeProcess:
                def __init__(self, command: list[str], **options: object) -> None:
                    process_options.update(options)
                    process_command.extend(command)
                    output = Path(command[command.index("--output") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    metrics = {
                        "sample_count": 1,
                        "hit_rate_at_10": 1.0,
                        "mrr": 1.0,
                        "mttc": 1.0,
                        "efficiency": 1.0,
                        "recommended_technical_score": 1.0,
                        "scenario_metrics": {},
                    }
                    robustness = {
                        "all_suites_robust_hit_count": 1,
                        "all_suites_robust_hit_rate": 1.0,
                    }
                    artifact = {
                        "corpora": {
                            "released_public": {
                                "robustness": robustness,
                                "suites": {
                                    "canonical": {"metrics": metrics},
                                    "combined_dev": {"metrics": metrics},
                                    "combined_challenge": {"metrics": metrics},
                                },
                            },
                            "derived_product_disjoint": {
                                "seed": "fixed",
                                "sample_count": 1,
                                "samples_sha256": "abc",
                                "unique_target_count": 1,
                                "public_target_overlap": 0,
                                "robustness": robustness,
                            },
                        }
                    }
                    output.write_text(json.dumps(artifact), encoding="utf-8")
                    self.stdout = iter(
                        f"[generalization] run/{index}: score=1.000000\n"
                        for index in range(6)
                    )
                    self.returncode: int | None = None

                def poll(self) -> int | None:
                    return self.returncode

                def terminate(self) -> None:
                    self.returncode = -15

                def wait(self, timeout: float | None = None) -> int:
                    del timeout
                    if self.returncode is None:
                        self.returncode = 0
                    return self.returncode

            provenance = runtime._capture_provenance()
            with patch.object(
                runtime, "_capture_provenance", return_value=provenance
            ), patch("observer.runtime.subprocess.Popen", FakeProcess):
                job = runtime.start_generalization()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    current = next(
                        item
                        for item in runtime.jobs()["jobs"]
                        if item["job_id"] == job["job_id"]
                    )
                    if current["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(0.02)

            self.assertEqual(current["status"], "completed")
            self.assertEqual((current["current"], current["total"]), (6, 6))
            self.assertEqual(
                process_options["env"]["TECHJAM_RERANK_MODE"],  # type: ignore[index]
                "shadow",
            )
            self.assertEqual(
                process_options["env"]["TECHJAM_RETRIEVAL_MODE"],  # type: ignore[index]
                "control",
            )
            self.assertEqual(
                process_command[process_command.index("--rerank-mode") + 1],
                "shadow",
            )
            self.assertEqual(
                process_command[process_command.index("--retrieval-mode") + 1],
                "control",
            )
            self.assertEqual(current["summary"]["rerank_mode"], "shadow")
            self.assertEqual(current["summary"]["retrieval_mode"], "control")
            self.assertEqual(
                current["summary"]["released_public"]["robustness"][
                    "all_suites_robust_hit_rate"
                ],
                1.0,
            )
            manifest_path = next(
                (root / "experiments").glob("*_generalization/manifest.json")
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "generalization")
            self.assertEqual(manifest["metrics"]["hit_rate_at_10"], 1.0)
            self.assertIn(
                "generalization_source_sha256", manifest["implementation"]
            )
            self.assertEqual(manifest["implementation"]["rerank_mode"], "shadow")
            self.assertEqual(manifest["implementation"]["retrieval_mode"], "control")
            self.assertIn("attributes_source_sha256", manifest["implementation"])
            self.assertIn("reranker_source_sha256", manifest["implementation"])
            self.assertIn("slot_ledger_source_sha256", manifest["implementation"])
            self.assertIn("clarification_source_sha256", manifest["implementation"])
            self.assertEqual(
                len(manifest["implementation"]["coverage_source_sha256"]), 64
            )
            self.assertEqual(
                len(manifest["implementation"]["shadow_analysis_source_sha256"]), 64
            )
            self.assertEqual(
                len(manifest["implementation"]["slot_ledger_source_sha256"]), 64
            )
            self.assertEqual(
                len(manifest["implementation"]["clarification_source_sha256"]), 64
            )
            self.assertEqual(
                manifest["implementation"]["slot_ledger_schema_version"],
                "p3.slot-ledger.v1",
            )
            self.assertEqual(
                manifest["implementation"]["question_value_schema_version"],
                "p3.question-value.v1",
            )
            self.assertEqual(
                manifest["implementation"]["coverage_schema_version"],
                COVERAGE_SCHEMA_VERSION,
            )
            self.assertEqual(manifest["implementation"]["clarification_mode"], "shadow")

            generalization_source.write_text("# changed\n", encoding="utf-8")
            with self.assertRaises(StaleRuntimeError):
                runtime.start_generalization()

    def test_http_api_exposes_sessions_and_rejects_unknown_routes(self) -> None:
        class FakeRuntime:
            def health(self) -> dict:
                return {"status": "ok", "branch": "pre"}

            def list_sessions(self) -> dict:
                return {"metrics": {"hit_rate_at_10": 0.5}, "sessions": []}

            def trace(self, sample_id: str, refresh: bool = False) -> dict:
                return {"sample_id": sample_id, "refresh": refresh}

            def start_evaluation(self) -> dict:
                return {"job_id": "evaluation_test", "status": "queued"}

            def start_generalization(self) -> dict:
                return {"job_id": "generalization_test", "status": "queued"}

        server = HTTPServer(("127.0.0.1", 0), make_handler(FakeRuntime()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base_url = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(f"{base_url}/api/token") as response:
            api_token = json.load(response)["token"]
        auth_headers = {"X-Observer-Token": api_token}

        sessions_request = urllib.request.Request(
            f"{base_url}/api/sessions", headers=auth_headers
        )
        with urllib.request.urlopen(sessions_request) as response:
            payload = json.load(response)
        self.assertEqual(payload["metrics"]["hit_rate_at_10"], 0.5)

        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(f"{base_url}/api/sessions")
        self.assertEqual(context.exception.code, 403)

        cross_site = urllib.request.Request(
            f"{base_url}/api/sessions",
            headers={**auth_headers, "Sec-Fetch-Site": "cross-site"},
        )
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(cross_site)
        self.assertEqual(context.exception.code, 403)

        trace_request = urllib.request.Request(
            f"{base_url}/api/trace?sample_id=public_1&refresh=1", headers=auth_headers
        )
        with urllib.request.urlopen(trace_request) as response:
            trace = json.load(response)
        self.assertEqual(trace, {"sample_id": "public_1", "refresh": True})

        request = urllib.request.Request(
            f"{base_url}/api/jobs/evaluation",
            data=b"{}",
            headers={"Content-Type": "application/json", **auth_headers},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            job = json.load(response)
        self.assertEqual(job["job_id"], "evaluation_test")

        robustness_request = urllib.request.Request(
            f"{base_url}/api/jobs/generalization",
            data=b"{}",
            headers={"Content-Type": "application/json", **auth_headers},
            method="POST",
        )
        with urllib.request.urlopen(robustness_request) as response:
            robustness_job = json.load(response)
        self.assertEqual(robustness_job["job_id"], "generalization_test")

        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(f"{base_url}/missing")
        self.assertEqual(context.exception.code, 404)

    def test_exclusive_server_can_bind_an_ephemeral_loopback_port(self) -> None:
        class FakeRuntime:
            def health(self) -> dict:
                return {"status": "ok"}

        server = ExclusiveHTTPServer(("127.0.0.1", 0), make_handler(FakeRuntime()))
        self.addCleanup(server.server_close)
        self.assertGreater(server.server_port, 0)


if __name__ == "__main__":
    unittest.main()
