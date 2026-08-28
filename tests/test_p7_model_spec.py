from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "configs" / "p7_bge_small_en_v1_5.json"


class P7ModelSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_frozen_model_and_runtime_are_cpu_local_only(self) -> None:
        self.assertEqual(self.spec["schema_version"], "p7.semantic-model-spec.v1")
        self.assertEqual(self.spec["model"]["repository"], "BAAI/bge-small-en-v1.5")
        self.assertRegex(self.spec["model"]["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(self.spec["model"]["license"], "MIT")
        self.assertEqual(
            self.spec["runtime"]["execution_provider"], "CPUExecutionProvider"
        )
        self.assertEqual(
            self.spec["runtime"]["graph_optimization_level"], "ORT_ENABLE_ALL"
        )
        self.assertEqual(
            self.spec["runtime"]["environment_before_numpy_or_onnxruntime_import"],
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_OFFLINE": "1",
            },
        )
        self.assertFalse(self.spec["runtime"]["network_required"])

    def test_preprocessing_is_fully_specified(self) -> None:
        encoder = self.spec["encoder"]
        self.assertEqual(encoder["model_supported_max_length"], 512)
        self.assertEqual(encoder["max_length"], 256)
        self.assertTrue(encoder["max_length_includes_special_tokens"])
        self.assertIn("resource cutoff", encoder["max_length_reason"])
        self.assertTrue(encoder["add_special_tokens"])
        self.assertEqual(encoder["truncation_side"], "right")
        self.assertEqual(encoder["padding_side"], "right")
        serialization = self.spec["document"]["value_serialization"]
        self.assertIn("recursively", serialization["mapping"])
        self.assertIn("recursively", serialization["sequence"])
        self.assertIn("empty dense route", self.spec["query"]["empty_query"])

    def test_selection_and_gate_contract_is_machine_readable(self) -> None:
        evaluation = self.spec["evaluation"]
        self.assertEqual(
            evaluation["selection_corpus"]["samples_sha256"],
            "bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546",
        )
        self.assertFalse(
            evaluation["selection_corpus"]["public_evaluation_authorized"]
        )
        self.assertEqual(evaluation["route_depths"]["sparse_broad"], 120)
        self.assertEqual(evaluation["route_depths"]["sparse_strict"], 80)
        self.assertEqual(evaluation["route_depths"]["dense_rescue"], 120)
        self.assertEqual(
            evaluation["recall_gates"]["minimum_rescued_sessions"], 5
        )
        self.assertEqual(
            evaluation["recall_gates"]["minimum_rescued_scenario_types"], 2
        )
        self.assertEqual(
            evaluation["resource_gates"]["required_asset_bytes_max"], 225_000_000
        )
        self.assertEqual(
            evaluation["resource_gates"]["query_search_p95_milliseconds_max"],
            40.0,
        )
        self.assertFalse(
            evaluation["decision"]["active_recommendation_candidate_in_p7"]
        )
        self.assertIn(
            "dense_union_session_recalled", evaluation["recall_definitions"]
        )
        self.assertIn(
            "canonical_records", evaluation["repeatability"]
        )

    def test_semantic_requirements_lock_direct_and_transitive_versions(self) -> None:
        lines = [
            line.strip()
            for line in (PROJECT_ROOT / "requirements-semantic.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertGreaterEqual(len(lines), 20)
        self.assertTrue(all("==" in line and " @ " not in line for line in lines))
        self.assertIn("numpy==2.4.6", lines)
        self.assertIn("onnxruntime==1.29.0", lines)
        self.assertIn("tokenizers==0.23.1", lines)
        self.assertIn("huggingface-hub==1.29.0", lines)

    def test_required_model_asset_manifest_has_unique_exact_entries(self) -> None:
        required = self.spec["required_files"]
        paths = [entry["path"] for entry in required]
        self.assertEqual(len(required), 11)
        self.assertEqual(len(paths), len(set(paths)))
        for entry in required:
            self.assertGreater(entry["bytes"], 0)
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
        onnx = next(entry for entry in required if entry["path"] == "onnx/model.onnx")
        self.assertEqual(onnx["bytes"], 133_093_490)
        self.assertEqual(
            onnx["sha256"],
            "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
        )

    def test_bundled_license_notice_is_present_and_frozen(self) -> None:
        notice = PROJECT_ROOT / self.spec["model"]["license_notice"]
        payload = notice.read_bytes().replace(b"\r\n", b"\n")
        self.assertIn(b"MIT License", payload)
        self.assertIn(b"Copyright (c) 2022 staoxiao", payload)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "587a673933425dbc36ec61268d3b954051b2d3ef3c9b322ede357976055ffdd5",
        )


if __name__ == "__main__":
    unittest.main()
