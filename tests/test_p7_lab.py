from __future__ import annotations

import ast
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from starter.agent import Agent
from starter.p7_lab import (
    C00,
    CONTROL_ID,
    DENSE_DEPTH,
    INDEX_LOCK_SCHEMA_VERSION,
    P7CaptureAgent,
    S00,
    SCHEMA_VERSION,
    SHADOW_ID,
    canonical_jsonl_bytes,
    create_p7_agent,
    validate_p7_index_lock,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _product(index: int) -> dict[str, object]:
    return {
        "parent_asin": f"ITEM-{index:03d}",
        "title": f"Women's cotton summer dress {index}",
        "categories": ["Clothing", "Women", "Dresses"],
        "features": ["breathable cotton", "casual summer style"],
        "details": {"material": "cotton", "color": "red"},
        "store": f"Store {index % 5}",
        "description": "Comfortable dress for daily use.",
    }


def _catalog(root: Path, count: int = 150) -> Path:
    path = root / "catalog.jsonl"
    path.write_text(
        "".join(json.dumps(_product(index)) + "\n" for index in range(count)),
        encoding="utf-8",
    )
    return path


def _file_entry(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _lock_fixture(root: Path) -> dict[str, object]:
    project = root / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "starter").mkdir()
    (project / "configs").mkdir()
    (project / "data").mkdir()
    (project / "third_party" / "fake-model").mkdir(parents=True)
    index_dir = project / "experiments" / "p7_index"
    index_dir.mkdir(parents=True)
    builder = project / "scripts" / "build_p7_semantic_index.py"
    semantic = project / "starter" / "semantic.py"
    builder.write_text("# frozen builder\n", encoding="utf-8")
    semantic.write_text("# frozen semantic core\n", encoding="utf-8")
    license_path = project / "third_party" / "fake-model" / "LICENSE"
    license_path.write_text("MIT License\nfrozen fixture\n", encoding="utf-8")
    spec_path = project / "configs" / "p7.json"
    spec = {
        "schema_version": "p7.semantic-model-spec.v1",
        "model": {"license_notice": "third_party/fake-model/LICENSE"},
        "required_files": [
            {"path": "model-a", "bytes": 11, "sha256": "a" * 64},
            {"path": "model-b", "bytes": 19, "sha256": "b" * 64},
        ],
        "evaluation": {
            "resource_gates": {"required_asset_bytes_max": 225_000_000}
        },
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    canonical_spec_sha = hashlib.sha256(
        json.dumps(
            spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    catalog = project / "data" / "catalog.jsonl"
    catalog.write_bytes(b'{"parent_asin":"A"}\n{"parent_asin":"B"}\n{"parent_asin":"C"}\n')
    matrix = index_dir / "embeddings.npy"
    matrix.write_bytes(b"fake-npy-index")
    asins = index_dir / "parent_asins.txt"
    asins.write_bytes(b"A\nB\nC\n")
    document_sha = "d" * 64
    resources = {
        "wall_seconds": 12.5,
        "rss_backend": "test-rss",
        "baseline_rss_bytes": 10,
        "peak_rss_bytes": 20,
        "peak_delta_from_baseline_bytes": 10,
    }
    model_required_file_bytes = sum(
        entry["bytes"] for entry in spec["required_files"]
    )
    manifest = {
        "schema_version": "p7.semantic-index.v1",
        "model_spec_serialization": (
            "UTF-8 canonical JSON; object keys sorted; compact separators; "
            "ensure_ascii=false"
        ),
        "model_spec_sha256": canonical_spec_sha,
        "catalog_sha256": _sha(catalog),
        "rows": 3,
        "dimensions": 2,
        "matrix": {
            "path": matrix.name,
            "bytes": matrix.stat().st_size,
            "sha256": _sha(matrix),
            "dtype": "float32",
            "shape": [3, 2],
            "format": "NumPy .npy",
        },
        "ordered_asins": {
            "path": asins.name,
            "bytes": asins.stat().st_size,
            "sha256": _sha(asins),
            "count": 3,
            "encoding": "utf-8-lf",
            "line_ending": "LF",
        },
        "preprocessing": {"canonical_documents_sha256": document_sha},
        "model": {"license_notice": _file_entry(license_path, project)},
        "asset_byte_scope": {
            "model_required_files_bytes": model_required_file_bytes,
            "matrix_bytes": matrix.stat().st_size,
            "ordered_asins_bytes": asins.stat().st_size,
            "license_notice_bytes": license_path.stat().st_size,
            "required_asset_bytes_excluding_manifest": 0,
            "manifest_path": "semantic-index.manifest.json",
            "manifest_bytes": 0,
            "required_asset_bytes": 0,
        },
        "build_resources": resources,
    }
    manifest_path = index_dir / "semantic-index.manifest.json"
    excluding_manifest = (
        model_required_file_bytes
        + matrix.stat().st_size
        + asins.stat().st_size
        + license_path.stat().st_size
    )
    manifest["asset_byte_scope"][
        "required_asset_bytes_excluding_manifest"
    ] = excluding_manifest
    for _ in range(20):
        payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
        manifest_bytes = len(payload)
        asset_bytes = excluding_manifest + manifest_bytes
        if (
            manifest["asset_byte_scope"]["manifest_bytes"] == manifest_bytes
            and manifest["asset_byte_scope"]["required_asset_bytes"] == asset_bytes
        ):
            break
        manifest["asset_byte_scope"]["manifest_bytes"] = manifest_bytes
        manifest["asset_byte_scope"]["required_asset_bytes"] = asset_bytes
    else:
        raise AssertionError("fixture manifest byte-size fixed point failed")
    manifest_path.write_bytes(payload)
    lock = {
        "schema_version": INDEX_LOCK_SCHEMA_VERSION,
        "source": {
            "git_commit": "a" * 40,
            "git_branch": "p4-architecture-search",
            "builder": _file_entry(builder, project),
            "semantic": _file_entry(semantic, project),
        },
        "model_spec": {
            "path": spec_path.relative_to(project).as_posix(),
            "raw_bytes": spec_path.stat().st_size,
            "raw_sha256": _sha(spec_path),
            "canonical_sha256": canonical_spec_sha,
        },
        "catalog": {
            **_file_entry(catalog, project),
            "rows": 3,
        },
        "index": {
            "directory": index_dir.relative_to(project).as_posix(),
            "manifest": {
                "path": manifest_path.name,
                "bytes": manifest_path.stat().st_size,
                "sha256": _sha(manifest_path),
                "schema_version": "p7.semantic-index.v1",
            },
            "matrix": {
                "path": matrix.name,
                "bytes": matrix.stat().st_size,
                "sha256": _sha(matrix),
                "dtype": "float32",
                "shape": [3, 2],
            },
            "ordered_asins": {
                "path": asins.name,
                "bytes": asins.stat().st_size,
                "sha256": _sha(asins),
                "count": 3,
                "encoding": "utf-8-lf",
                "line_ending": "LF",
            },
            "canonical_documents_sha256": document_sha,
        },
        "asset_scope": {
            "required_asset_bytes": asset_bytes,
            "required_asset_bytes_max": 225_000_000,
        },
        "build_observation": resources,
    }
    lock_path = project / "configs" / "p7_semantic_index_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return {
        "project": project,
        "spec": spec_path,
        "catalog": catalog,
        "index": index_dir,
        "matrix": matrix,
        "manifest": manifest_path,
        "license": license_path,
        "lock": lock_path,
        "lock_object": lock,
    }


class P7CaptureLabTest(unittest.TestCase):
    def test_roles_and_constructor_freeze_exact_sparse_configuration(self) -> None:
        self.assertEqual((CONTROL_ID, SHADOW_ID), (C00, S00))
        self.assertEqual(SCHEMA_VERSION, "p7.target-blind-capture.v1")
        with tempfile.TemporaryDirectory() as directory:
            catalog = _catalog(Path(directory), 20)
            control = P7CaptureAgent(catalog, C00)
            shadow = P7CaptureAgent(
                catalog, S00, dense_search=lambda _query, _top_k: []
            )
            self.addCleanup(control.close)
            self.addCleanup(shadow.close)
            self.assertIs(type(control), type(shadow))
            self.assertEqual(control.retrieval_mode, "coverage")
            self.assertEqual(control.rerank_mode, "off")
            self.assertEqual(control.question_policy, "fast")
            self.assertIsNone(control.llm_client)
            self.assertIsNone(control.trace_sink)
            stats = control.semantic_stats()
            self.assertEqual(
                set(stats),
                {
                    "schema_version",
                    "role",
                    "route_record_count",
                    "response_record_count",
                    "dense_call_count",
                    "empty_query_count",
                    "semantic_exception_count",
                    "capture_exception_count",
                    "integrity_error_count",
                    "cold_initialization_seconds",
                    "required_asset_bytes",
                },
            )
            self.assertIsNone(stats["cold_initialization_seconds"])
            self.assertEqual(stats["required_asset_bytes"], 0)
            with self.assertRaisesRegex(ValueError, "must not receive"):
                P7CaptureAgent(catalog, C00, dense_search=lambda _q, _k: [])
            with self.assertRaisesRegex(ValueError, "requires"):
                P7CaptureAgent(catalog, S00)

    def test_control_and_shadow_responses_are_exact_and_capture_ordinal_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _catalog(Path(directory))
            baseline = Agent(
                catalog,
                question_policy="fast",
                rerank_mode="off",
                retrieval_mode="coverage",
            )
            control = P7CaptureAgent(catalog, C00)
            shadow = P7CaptureAgent(
                catalog, S00, dense_search=lambda _query, _top_k: [("ITEM-149", 0.25)]
            )
            self.addCleanup(baseline.connection.close)
            self.addCleanup(control.close)
            self.addCleanup(shadow.close)
            baseline.reset("baseline", {})
            control.reset(7, {})
            shadow.reset(7, {})
            messages = [
                "I'm looking for women's red cotton dresses.",
                "Actually, switch from women's red cotton dresses to men's black running shoes.",
            ]
            for turn, message in enumerate(messages, start=1):
                expected = baseline.respond("baseline", message, turn, 10)
                self.assertEqual(control.respond(7, message, turn, 10), expected)
                self.assertEqual(shadow.respond(7, message, turn, 10), expected)

            capture = shadow.export_target_blind_capture()
            encoded = json.dumps(capture, allow_nan=False)
            self.assertEqual(
                set(capture),
                {
                    "schema_version",
                    "role",
                    "configuration",
                    "target_blind",
                    "label_free",
                    "route_records",
                    "response_records",
                    "integrity_errors",
                    "stats",
                    "hashes",
                },
            )
            self.assertNotIn("baseline", encoded)
            self.assertEqual([row["ordinal"] for row in capture["route_records"]], [7, 7])
            self.assertEqual([row["turn"] for row in capture["route_records"]], [1, 2])
            self.assertEqual(capture["stats"]["route_record_count"], 2)
            self.assertEqual(capture["stats"]["response_record_count"], 2)
            self.assertTrue(capture["target_blind"])
            self.assertTrue(capture["label_free"])
            self.assertFalse({"women", "red", "cotton", "dresses"} & set(
                capture["route_records"][1]["query"].split()
            ))
            stable_routes = [
                {key: value for key, value in record.items() if key != "query_search_ns"}
                for record in capture["route_records"]
            ]
            dense_routes = [
                {
                    "ordinal": record["ordinal"],
                    "turn": record["turn"],
                    "query": record["query"],
                    "dense": record["dense"],
                }
                for record in capture["route_records"]
            ]
            self.assertEqual(
                capture["hashes"]["routes_sha256"],
                hashlib.sha256(canonical_jsonl_bytes(stable_routes)).hexdigest(),
            )
            self.assertEqual(
                capture["hashes"]["dense_routes_sha256"],
                hashlib.sha256(canonical_jsonl_bytes(dense_routes)).hexdigest(),
            )
            self.assertEqual(
                capture["hashes"]["responses_sha256"],
                hashlib.sha256(
                    canonical_jsonl_bytes(capture["response_records"])
                ).hexdigest(),
            )

    def test_rank_hook_calls_parent_once_and_copies_exact_depths_and_score_hex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _catalog(Path(directory), 20)
            calls: list[tuple[str, int]] = []

            def dense(query: str, top_k: int) -> list[tuple[str, float]]:
                calls.append((query, top_k))
                return [("Z", 0.5), ("A", -0.0)]

            shadow = P7CaptureAgent(catalog, S00, dense_search=dense)
            self.addCleanup(shadow.close)
            shadow.reset(1, {})
            rankings = {
                "broad": [f"B{index:03d}" for index in range(130)],
                "strict": [f"S{index:03d}" for index in range(90)],
                "fused": ["B000"],
                "reranked": ["B000"],
                "final": ["B000"],
            }
            with patch.object(
                Agent, "_rank_candidates", autospec=True, return_value=rankings
            ) as parent_rank, patch(
                "starter.p7_lab.time.perf_counter_ns", side_effect=[100, 175]
            ):
                response = shadow.respond(
                    1, "I'm looking for women's cotton dresses.", 1, 10
                )
            parent_rank.assert_called_once()
            self.assertEqual(response["recommendations"], [{"parent_asin": "B000"}])
            record = shadow.route_records()[0]
            self.assertEqual(len(record["broad"]), 120)
            self.assertEqual(record["broad"], rankings["broad"][:120])
            self.assertEqual(len(record["strict"]), 80)
            self.assertEqual(record["strict"], rankings["strict"][:80])
            self.assertEqual(record["query_search_ns"], 75)
            self.assertEqual(calls[0][1], DENSE_DEPTH)
            self.assertEqual(
                record["dense"],
                [
                    {"parent_asin": "Z", "score": float(0.5).hex()},
                    {"parent_asin": "A", "score": float(-0.0).hex()},
                ],
            )

    def test_debug_rankings_never_creates_capture_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shadow = P7CaptureAgent(
                _catalog(Path(directory), 30),
                S00,
                dense_search=lambda _query, _top_k: [],
            )
            self.addCleanup(shadow.close)
            shadow.reset("opaque-session", {})
            shadow.debug_rankings("opaque-session")
            self.assertEqual(shadow.route_records(), [])
            shadow.respond(
                "opaque-session", "I'm looking for women's cotton dresses.", 1, 10
            )
            self.assertEqual(len(shadow.route_records()), 1)
            shadow.debug_rankings("opaque-session")
            self.assertEqual(len(shadow.route_records()), 1)

    def test_dense_exception_and_invalid_route_preserve_sparse_response(self) -> None:
        for result, exception_name in (
            (RuntimeError("boom"), "RuntimeError"),
            ([("A", 1.0), ("A", 0.5)], "ValueError"),
        ):
            with self.subTest(exception_name=exception_name), tempfile.TemporaryDirectory() as directory:
                catalog = _catalog(Path(directory), 40)
                baseline = Agent(catalog, retrieval_mode="coverage", rerank_mode="off")

                def dense(_query: str, _top_k: int) -> object:
                    if isinstance(result, Exception):
                        raise result
                    return result

                shadow = P7CaptureAgent(catalog, S00, dense_search=dense)
                self.addCleanup(baseline.connection.close)
                self.addCleanup(shadow.close)
                baseline.reset("base", {})
                shadow.reset(1, {})
                message = "I'm looking for women's cotton dresses."
                expected = baseline.respond("base", message, 1, 10)
                observed = shadow.respond(1, message, 1, 10)
                self.assertEqual(observed, expected)
                record = shadow.route_records()[0]
                self.assertEqual(record["dense"], [])
                self.assertEqual(record["semantic_exception_class"], exception_name)
                self.assertEqual(shadow.semantic_stats()["semantic_exception_count"], 1)

    def test_empty_query_records_reached_turn_without_dense_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dense = Mock(return_value=[])
            shadow = P7CaptureAgent(
                _catalog(Path(directory), 20), S00, dense_search=dense
            )
            self.addCleanup(shadow.close)
            shadow.reset(1, {})
            with patch.object(shadow, "_query_terms", return_value=[]):
                shadow.respond(1, "x", 1, 10)
            dense.assert_not_called()
            record = shadow.route_records()[0]
            self.assertEqual(record["query"], "")
            self.assertTrue(record["empty_query"])
            self.assertEqual(record["dense"], [])
            self.assertEqual(shadow.semantic_stats()["empty_query_count"], 1)

    def test_response_object_is_returned_unchanged_and_recorded_by_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            control = P7CaptureAgent(_catalog(Path(directory), 10), C00)
            self.addCleanup(control.close)
            control.reset(1, {})
            response = {
                "message": "exact",
                "ask_attribute": None,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
            with patch.object(Agent, "respond", return_value=response):
                observed = control.respond(1, "message", 1, 10)
            self.assertIs(observed, response)
            response["message"] = "mutated later"
            self.assertEqual(
                control.response_records()[0]["response"]["message"], "exact"
            )

    def test_hashes_exclude_durations_but_include_ordered_routes(self) -> None:
        base_record = {
            "ordinal": 1,
            "turn": 1,
            "query": "red shoes",
            "empty_query": False,
            "broad": ["A"],
            "strict": [],
            "dense": [{"parent_asin": "B", "score": float(0.5).hex()}],
            "query_search_ns": 10,
        }
        self.assertTrue(canonical_jsonl_bytes([base_record]).endswith(b"\n"))
        with tempfile.TemporaryDirectory() as directory:
            first = P7CaptureAgent(
                _catalog(Path(directory), 5), S00, dense_search=lambda _q, _k: []
            )
            second = P7CaptureAgent(
                Path(directory) / "catalog.jsonl",
                S00,
                dense_search=lambda _q, _k: [],
            )
            self.addCleanup(first.close)
            self.addCleanup(second.close)
            for agent, times in ((first, [10, 20]), (second, [100, 900])):
                agent.reset(1, {})
                with patch("starter.p7_lab.time.perf_counter_ns", side_effect=times):
                    agent.respond(1, "I'm looking for cotton dresses.", 1, 10)
            self.assertNotEqual(
                first.route_records()[0]["query_search_ns"],
                second.route_records()[0]["query_search_ns"],
            )
            self.assertEqual(
                first.export_target_blind_capture()["hashes"],
                second.export_target_blind_capture()["hashes"],
            )

    def test_reset_accepts_explicit_ordinal_and_never_serializes_session_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            control = P7CaptureAgent(_catalog(Path(directory), 10), C00)
            self.addCleanup(control.close)
            control.reset(9, {})
            control.respond(9, "I'm looking for cotton dresses.", 1, 10)
            control.drop_session(9)
            with self.assertRaisesRegex(ValueError, "duplicate corpus ordinal"):
                control.reset(9, {})
            control.reset("uuid-like-private-key", {})
            control.respond(
                "uuid-like-private-key", "I'm looking for silver jewelry.", 1, 10
            )
            payload = json.dumps(control.export_target_blind_capture())
            self.assertNotIn("uuid-like-private-key", payload)
            self.assertEqual(
                [record["ordinal"] for record in control.route_records()], [9, 10]
            )

    def test_lock_validator_hard_gates_sources_inputs_index_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _lock_fixture(Path(directory))
            verified = validate_p7_index_lock(
                fixture["catalog"],
                fixture["spec"],
                fixture["index"],
                lock_path=fixture["lock"],
                project_root=fixture["project"],
                expected_catalog_sha256=_sha(fixture["catalog"]),
                expected_catalog_rows=3,
            )
            self.assertEqual(verified, fixture["lock_object"])

            matrix = fixture["matrix"]
            assert isinstance(matrix, Path)
            matrix.write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "matrix bytes mismatch"):
                validate_p7_index_lock(
                    fixture["catalog"],
                    fixture["spec"],
                    fixture["index"],
                    lock_path=fixture["lock"],
                    project_root=fixture["project"],
                    expected_catalog_sha256=_sha(fixture["catalog"]),
                    expected_catalog_rows=3,
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = _lock_fixture(Path(directory))
            lock = fixture["lock_object"]
            lock["unexpected"] = True
            fixture["lock"].write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "extra=.*unexpected"):
                validate_p7_index_lock(
                    fixture["catalog"],
                    fixture["spec"],
                    fixture["index"],
                    lock_path=fixture["lock"],
                    project_root=fixture["project"],
                    expected_catalog_sha256=_sha(fixture["catalog"]),
                    expected_catalog_rows=3,
                )

    def test_lock_validator_rejects_tampered_or_missing_bundled_license(self) -> None:
        for failure in ("tampered", "missing"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                fixture = _lock_fixture(Path(directory))
                license_path = fixture["license"]
                assert isinstance(license_path, Path)
                if failure == "tampered":
                    license_path.write_bytes(b"X" * license_path.stat().st_size)
                    expected_error = ValueError
                    pattern = "license.*SHA-256 mismatch"
                else:
                    license_path.unlink()
                    expected_error = FileNotFoundError
                    pattern = "license.*missing"
                with self.assertRaisesRegex(expected_error, pattern):
                    validate_p7_index_lock(
                        fixture["catalog"],
                        fixture["spec"],
                        fixture["index"],
                        lock_path=fixture["lock"],
                        project_root=fixture["project"],
                        expected_catalog_sha256=_sha(fixture["catalog"]),
                        expected_catalog_rows=3,
                    )

    def test_lock_validator_recomputes_total_against_colluding_fake_totals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _lock_fixture(Path(directory))
            manifest_path = fixture["manifest"]
            lock_path = fixture["lock"]
            assert isinstance(manifest_path, Path) and isinstance(lock_path, Path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fake_total = manifest["asset_byte_scope"]["required_asset_bytes"] + 100
            manifest["asset_byte_scope"]["required_asset_bytes"] = fake_total
            for _ in range(20):
                payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
                if manifest["asset_byte_scope"]["manifest_bytes"] == len(payload):
                    break
                manifest["asset_byte_scope"]["manifest_bytes"] = len(payload)
            else:
                raise AssertionError("fake manifest byte-size fixed point failed")
            manifest_path.write_bytes(payload)

            lock = fixture["lock_object"]
            lock["index"]["manifest"]["bytes"] = manifest_path.stat().st_size
            lock["index"]["manifest"]["sha256"] = _sha(manifest_path)
            lock["asset_scope"]["required_asset_bytes"] = fake_total
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "required_asset_bytes.*recomputed"
            ):
                validate_p7_index_lock(
                    fixture["catalog"],
                    fixture["spec"],
                    fixture["index"],
                    lock_path=lock_path,
                    project_root=fixture["project"],
                    expected_catalog_sha256=_sha(fixture["catalog"]),
                    expected_catalog_rows=3,
                )

    def test_factory_control_avoids_semantic_import_and_shadow_uses_lock_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _catalog(Path(directory), 20)
            with patch("starter.p7_lab.importlib.import_module") as dynamic_import:
                control = create_p7_agent(
                    "C00", catalog, "missing-spec", "missing-model", "missing-index"
                )
            self.addCleanup(control.close)
            dynamic_import.assert_not_called()

            with patch(
                "starter.p7_lab.validate_p7_index_lock",
                side_effect=ValueError("lock failed"),
            ) as lock_gate, patch(
                "starter.p7_lab.importlib.import_module"
            ) as dynamic_import:
                with self.assertRaisesRegex(ValueError, "lock failed"):
                    create_p7_agent(
                        "S00",
                        catalog,
                        "spec",
                        "model",
                        "index",
                        lock_path="frozen-lock.json",
                    )
            lock_gate.assert_called_once()
            self.assertEqual(
                lock_gate.call_args.kwargs["lock_path"], "frozen-lock.json"
            )
            dynamic_import.assert_not_called()

    def test_factory_shadow_loads_local_runtime_and_closes_both_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _catalog(Path(directory), 20)
            encoder = SimpleNamespace(_np=object(), close=Mock())
            index = SimpleNamespace(
                search_query=Mock(return_value=[("ITEM-001", 0.75)]), close=Mock()
            )
            encoder_type = SimpleNamespace(
                from_frozen_assets=Mock(return_value=encoder)
            )
            index_type = SimpleNamespace(load=Mock(return_value=index))
            semantic = SimpleNamespace(
                load_semantic_spec=Mock(return_value={"frozen": True}),
                OfflineSemanticEncoder=encoder_type,
                SemanticIndex=index_type,
            )
            lock = {
                "catalog": {"sha256": "c" * 64},
                "asset_scope": {"required_asset_bytes": 123},
            }
            with patch(
                "starter.p7_lab.validate_p7_index_lock", return_value=lock
            ) as lock_gate, patch(
                "starter.p7_lab.time.perf_counter", side_effect=[10.0, 10.25]
            ), patch(
                "starter.p7_lab.importlib.import_module", return_value=semantic
            ) as dynamic_import:
                shadow = create_p7_agent(
                    "S00",
                    catalog,
                    "spec.json",
                    "model",
                    "index",
                    lock_path="frozen-lock.json",
                )
            lock_gate.assert_called_once()
            self.assertEqual(
                lock_gate.call_args.kwargs["lock_path"], "frozen-lock.json"
            )
            dynamic_import.assert_called_once_with("starter.semantic")
            self.assertEqual(
                index_type.load.call_args.kwargs["expected_catalog_sha256"],
                "c" * 64,
            )
            stats = shadow.semantic_stats()
            self.assertEqual(stats["cold_initialization_seconds"], 0.25)
            self.assertEqual(stats["required_asset_bytes"], 123)
            shadow.reset(1, {})
            shadow.respond(1, "I'm looking for cotton dresses.", 1, 10)
            index.search_query.assert_called_once()
            shadow.close()
            index.close.assert_called_once()
            encoder.close.assert_called_once()

    def test_source_has_no_label_inputs_or_direct_semantic_runtime_imports(self) -> None:
        source_path = PROJECT_ROOT / "starter" / "p7_lab.py"
        source = source_path.read_text(encoding="utf-8")
        folded = source.casefold()
        blocked = (
            "ground_" + "truth",
            "scenario_" + "type",
            "sample_" + "id",
            "intent_" + "card",
            "beha" + "vior",
            "public_" + "set",
        )
        for value in blocked:
            self.assertNotIn(value, folded)
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue({"numpy", "tokenizers", "onnxruntime"}.isdisjoint(imported))
        constructor = set(inspect.signature(P7CaptureAgent.__init__).parameters)
        self.assertFalse({"profile", "label", "target", "sample"} & constructor)


if __name__ == "__main__":
    unittest.main()
