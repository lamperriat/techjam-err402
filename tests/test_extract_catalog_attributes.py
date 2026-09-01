from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from extract_catalog_attributes import (
    CatalogRunConfig,
    experiment_metadata,
    initialize_output,
    load_checkpoint,
    recover_captured_responses,
    run_catalog_extraction,
)
from utils.llm_client import LLMConfig, TokenUsage


def product(index: int) -> dict:
    return {
        "parent_asin": f"product-{index}",
        "title": f"Blue Product {index}",
        "features": [],
        "description": [],
        "categories": ["Clothing, Shoes & Jewelry"],
        "details": {},
    }


def valid_response() -> dict:
    return {
        "material": [],
        "color": [{"value": "Blue", "evidence": "Blue"}],
        "size_fit": [],
        "style": [],
        "use_case": [],
        "specific_attributes": [],
    }


class CallTracker:
    def __init__(self, failed_titles: set[str] | None = None) -> None:
        self.failed_titles = failed_titles or set()
        self.lock = threading.Lock()
        self.calls = 0
        self.active = 0
        self.max_active = 0


class FakeLLM:
    def __init__(self, tracker: CallTracker) -> None:
        self.config = LLMConfig("test-key", "fake-model")
        self.tracker = tracker
        self._usage = TokenUsage()

    def generate_json(
        self,
        messages,
        *,
        temperature=None,
        max_tokens=None,
        extra_body=None,
    ):
        content = messages[-1]["content"]
        with self.tracker.lock:
            self.tracker.calls += 1
            self.tracker.active += 1
            self.tracker.max_active = max(
                self.tracker.max_active,
                self.tracker.active,
            )
        try:
            time.sleep(0.02)
            if any(title in content for title in self.tracker.failed_titles):
                raise RuntimeError("simulated failure")
            self._usage += TokenUsage(prompt_tokens=10, completion_tokens=2)
            return valid_response()
        finally:
            with self.tracker.lock:
                self.tracker.active -= 1

    def consume_usage(self) -> TokenUsage:
        usage = self._usage
        self._usage = TokenUsage()
        return usage


class CatalogExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.llm_config = LLMConfig("test-key", "fake-model")

    def test_bounds_concurrency_and_resumes_only_failed_products(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            output_path = root / "attributes.jsonl"
            catalog_path.write_text(
                "".join(json.dumps(product(index)) + "\n" for index in range(12)),
                encoding="utf-8",
            )

            first_tracker = CallTracker({"Blue Product 3"})
            first_summary = run_catalog_extraction(
                catalog_path,
                output_path,
                CatalogRunConfig(workers=4),
                self.llm_config,
                client_factory=lambda: FakeLLM(first_tracker),
                show_progress=False,
            )

            self.assertEqual(first_tracker.calls, 12)
            self.assertEqual(first_tracker.max_active, 4)
            self.assertEqual(first_summary["successful_this_run"], 11)
            self.assertEqual(first_summary["errors_this_run"], 1)
            self.assertEqual(first_summary["remaining_products"], 1)

            second_tracker = CallTracker()
            second_summary = run_catalog_extraction(
                catalog_path,
                output_path,
                CatalogRunConfig(workers=4),
                self.llm_config,
                client_factory=lambda: FakeLLM(second_tracker),
                show_progress=False,
            )

            self.assertEqual(second_tracker.calls, 1)
            self.assertEqual(second_summary["attempted_this_run"], 1)
            self.assertEqual(second_summary["successful_products_total"], 12)
            self.assertEqual(second_summary["remaining_products"], 0)

            checkpoint = load_checkpoint(
                output_path,
                experiment_metadata(catalog_path, self.llm_config.model),
            )
            self.assertEqual(checkpoint.error_records, 0)

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["record_type"], "metadata")
            self.assertEqual(len(records), 14)

    def test_repairs_only_an_incomplete_final_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            output_path = root / "attributes.jsonl"
            catalog_path.write_text(json.dumps(product(0)) + "\n", encoding="utf-8")
            experiment = experiment_metadata(catalog_path, self.llm_config.model)
            metadata = {"record_type": "metadata", "experiment": experiment}
            output_path.write_text(
                json.dumps(metadata) + "\n" + '{"parent_asin":',
                encoding="utf-8",
            )

            checkpoint = load_checkpoint(output_path, experiment)

            self.assertEqual(checkpoint.completed, frozenset())
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                json.dumps(metadata) + "\n",
            )

    def test_rejects_incompatible_resume_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            output_path = root / "attributes.jsonl"
            catalog_path.write_text(json.dumps(product(0)) + "\n", encoding="utf-8")
            metadata = {
                "record_type": "metadata",
                "experiment": experiment_metadata(catalog_path, "different-model"),
            }
            output_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "incompatible extraction settings"):
                load_checkpoint(
                    output_path,
                    experiment_metadata(catalog_path, self.llm_config.model),
                )

    def test_recovers_a_captured_response_without_another_llm_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            output_path = root / "attributes.jsonl"
            item = product(0)
            catalog_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            experiment = experiment_metadata(catalog_path, self.llm_config.model)
            initialize_output(output_path, experiment)
            malformed = json.dumps(valid_response()).replace(
                '"evidence"',
                '" "evidence"',
                1,
            )
            with output_path.open("a", encoding="utf-8") as output:
                output.write(
                    json.dumps(
                        {
                            "parent_asin": item["parent_asin"],
                            "status": "error",
                            "raw_response": malformed,
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                            },
                        }
                    )
                    + "\n"
                )

            recovered = recover_captured_responses(
                output_path,
                [item],
                self.llm_config.model,
            )
            checkpoint = load_checkpoint(output_path, experiment)

            self.assertEqual(recovered, 1)
            self.assertEqual(checkpoint.completed, {item["parent_asin"]})
            self.assertEqual(checkpoint.usage, TokenUsage(10, 2))


if __name__ == "__main__":
    unittest.main()
