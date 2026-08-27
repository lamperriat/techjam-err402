from __future__ import annotations

import json
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
from observer.events import TRACE_SCHEMA_VERSION, TraceRecorder
from observer.runtime import StaleRuntimeError, WorkbenchRuntime
from observer.server import ExclusiveHTTPServer, make_handler
from observer.trace import TraceRunner
from starter.agent import Agent


class ObserverTraceTest(unittest.TestCase):
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
                Agent(catalog_path, trace_sink=recorder.emit),
                [sample],
                catalog_ids,
                categories,
                product_index,
                recorder=recorder,
            )

            trace = runner.trace("public_test_1")
            plain_agent = Agent(catalog_path)
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
            {"broad", "strict", "fused"},
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
            runtime = WorkbenchRuntime.from_paths(
                catalog_path, dataset_path, results_path, project_root=root
            )
            self.addCleanup(runtime.close)

            overview = runtime.overview()
            self.assertEqual(overview["index"]["rows"], 1)
            self.assertEqual(overview["pipeline"][1]["status"], "implemented")
            self.assertEqual(overview["pipeline"][6]["status"], "implemented")
            self.assertEqual(runtime.catalog("cotton")["items"][0]["parent_asin"], "A")
            self.assertEqual(runtime.product("A")["title"], product["title"])

            lab = runtime.lab_reset()
            reply = runtime.lab_respond(lab["session_id"], "cotton running shoe")
            self.assertEqual(reply["recommendations"][0]["parent_asin"], "A")
            self.assertIn("retrieval", {event["layer"] for event in reply["events"]})

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
            manifest_path = next((root / "experiments").glob("*/manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run"]["top_k"], 10)
            self.assertIn("agent_source_sha256", manifest["implementation"])
            self.assertEqual(manifest["implementation"]["question_policy"], "fast")
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

            source_path = root / "starter" / "agent.py"
            source_path.parent.mkdir(parents=True)
            source_path.write_text("# changed after server startup\n", encoding="utf-8")
            self.assertTrue(runtime.overview()["source_state"]["restart_required"])
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
                catalog_path, dataset_path, results_path, project_root=root
            )
            self.addCleanup(runtime.close)

            evaluation_job, created = runtime._new_job("evaluation")
            self.assertTrue(created)
            evaluation_job.status = "running"
            blocked = runtime.start_generalization()
            self.assertEqual(blocked["job_id"], evaluation_job.job_id)
            self.assertEqual(blocked["kind"], "evaluation")
            evaluation_job.status = "completed"

            class FakeProcess:
                def __init__(self, command: list[str], **_: object) -> None:
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
