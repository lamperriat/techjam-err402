from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_p7_semantic_index import (
    DEFAULT_CATALOG,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPEC,
    FROZEN_INDEX_SHAPE,
    FROZEN_MODEL_SPEC_SHA256,
    MANIFEST_FILENAME,
    MATRIX_FILENAME,
    ORDERED_ASINS_FILENAME,
    SCHEMA_VERSION,
    _parser,
    build_semantic_index,
    canonical_model_spec_sha256,
    main,
    prepare_runtime_environment,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(root: Path, *, rows: int = 5, dimensions: int = 3) -> dict[str, object]:
    project_root = root / "project"
    project_root.mkdir()
    model_dir = root / "model"
    (model_dir / "onnx").mkdir(parents=True)
    model_payloads = {
        "config.json": b'{"hidden_size":3}\n',
        "onnx/model.onnx": b"fake-onnx-model-for-unit-tests\n",
    }
    required_files = []
    for relative, payload in model_payloads.items():
        path = model_dir / relative
        path.write_bytes(payload)
        required_files.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    license_path = project_root / "third_party" / "fake" / "LICENSE"
    license_path.parent.mkdir(parents=True)
    license_path.write_text("MIT License\nunit fixture\n", encoding="utf-8")

    letters = [chr(ord("A") + index) for index in range(rows)]
    products = [
        {
            "parent_asin": f"P{letter}",
            "title": f"Title {letter}",
            "categories": ["Clothing", letter],
        }
        for letter in reversed(letters)
    ]
    catalog_path = root / "catalog.jsonl"
    _write_jsonl(catalog_path, products)

    spec = {
        "schema_version": "p7.semantic-model-spec.v1",
        "model": {
            "repository": "fixture/fake-encoder",
            "revision": "1" * 40,
            "license": "MIT",
            "license_notice": "third_party/fake/LICENSE",
        },
        "runtime": {
            "python": platform_python_version(),
            "numpy": "test",
            "onnxruntime": "test",
            "tokenizers": "test",
            "execution_provider": "CPUExecutionProvider",
            "execution_mode": "ORT_SEQUENTIAL",
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "network_required": False,
            "environment_before_numpy_or_onnxruntime_import": {
                "P7_TEST_THREAD_LIMIT": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_OFFLINE": "1",
            },
        },
        "encoder": {
            "embedding_dimensions": dimensions,
            "batch_size": 2,
            "pooling": "first_token_cls",
            "normalization": "l2_float32",
            "max_length": 16,
        },
        "document": {
            "schema_version": "p7.catalog-document.test-v1",
            "field_order": ["title", "categories"],
            "field_separator": " | ",
        },
        "index": {
            "catalog_order": "parent_asin_ascending",
            "matrix_dtype": "float32",
            "shape": [rows, dimensions],
        },
        "required_files": required_files,
    }
    spec_path = root / "spec.json"
    spec_path.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "project_root": project_root,
        "model_dir": model_dir,
        "catalog": catalog_path,
        "spec": spec_path,
        "spec_object": spec,
        "output": root / "index",
        "ordered_asins": [f"P{letter}" for letter in letters],
    }


def platform_python_version() -> str:
    return ".".join(str(value) for value in sys.version_info[:3])


class FakeEncoder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.closed = False

    def encode_documents(self, texts: list[str]) -> object:
        import numpy

        self.batches.append(list(texts))
        rows = []
        for text in texts:
            letter = text.rsplit(" ", 1)[-1]
            ordinal = float(ord(letter) - ord("A") + 1)
            raw = numpy.asarray([ordinal, 1.0, ordinal + 1.0], dtype=numpy.float32)
            rows.append(raw / numpy.linalg.norm(raw))
        return numpy.stack(rows)

    def close(self) -> None:
        self.closed = True


def _document(product: dict, document_spec: dict) -> str:
    if document_spec["schema_version"] != "p7.catalog-document.test-v1":
        raise AssertionError("builder supplied the wrong frozen document spec")
    return f"title: {product['title']}"


def _build_kwargs(fixture: dict[str, object]) -> dict[str, object]:
    catalog = fixture["catalog"]
    spec_object = fixture["spec_object"]
    assert isinstance(catalog, Path)
    assert isinstance(spec_object, dict)
    return {
        "expected_catalog_sha256": _sha256(catalog),
        "expected_model_spec_sha256": canonical_model_spec_sha256(spec_object),
        "expected_shape": tuple(spec_object["index"]["shape"]),
        "project_root": fixture["project_root"],
        "document_builder": _document,
        "enforce_fresh_runtime_import": False,
        "enforce_runtime_versions": False,
        "rss_sample_ms": 1.0,
    }


class P7SemanticIndexBuilderTest(unittest.TestCase):
    def test_frozen_spec_hash_and_shape_constants_match_tracked_config(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        spec = json.loads((project_root / DEFAULT_SPEC).read_text(encoding="utf-8"))
        self.assertEqual(canonical_model_spec_sha256(spec), FROZEN_MODEL_SPEC_SHA256)
        self.assertEqual(tuple(spec["index"]["shape"]), FROZEN_INDEX_SHAPE)

    def test_builds_sorted_batched_float32_index_and_complete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            encoder = FakeEncoder()
            manifest = build_semantic_index(
                fixture["spec"],
                fixture["catalog"],
                fixture["model_dir"],
                fixture["output"],
                encoder_factory=lambda _spec, _model_dir: encoder,
                **_build_kwargs(fixture),
            )
            output = fixture["output"]
            assert isinstance(output, Path)
            stored = json.loads(
                (output / MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            asins = (output / ORDERED_ASINS_FILENAME).read_bytes()
            import numpy

            matrix = numpy.load(output / MATRIX_FILENAME, allow_pickle=False)

            self.assertEqual(stored, manifest)
            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            self.assertEqual(
                manifest["model_spec_serialization"],
                "UTF-8 canonical JSON; object keys sorted; compact separators; "
                "ensure_ascii=false",
            )
            self.assertEqual(manifest["rows"], 5)
            self.assertEqual(manifest["dimensions"], 3)
            self.assertEqual(manifest["matrix"]["format"], "NumPy .npy")
            self.assertEqual(manifest["ordered_asins"]["count"], 5)
            self.assertEqual(manifest["ordered_asins"]["encoding"], "utf-8-lf")
            self.assertEqual(matrix.shape, (5, 3))
            self.assertEqual(matrix.dtype, numpy.dtype("float32"))
            self.assertTrue(numpy.all(numpy.isfinite(matrix)))
            self.assertTrue(numpy.allclose(numpy.linalg.norm(matrix, axis=1), 1.0))
            self.assertEqual(asins.decode("utf-8").splitlines(), fixture["ordered_asins"])
            self.assertTrue(asins.endswith(b"\n"))
            self.assertEqual([len(batch) for batch in encoder.batches], [2, 2, 1])
            self.assertEqual(
                [text for batch in encoder.batches for text in batch],
                [f"title: Title {letter}" for letter in "ABCDE"],
            )
            self.assertTrue(encoder.closed)
            self.assertEqual(manifest["matrix"]["sha256"], _sha256(output / MATRIX_FILENAME))
            self.assertEqual(
                manifest["ordered_asins"]["sha256"],
                _sha256(output / ORDERED_ASINS_FILENAME),
            )
            self.assertEqual(
                manifest["model_spec_sha256"],
                canonical_model_spec_sha256(fixture["spec_object"]),
            )
            self.assertEqual(manifest["model"]["required_file_count"], 2)
            self.assertTrue(manifest["model"]["all_required_files_verified"])
            self.assertTrue(
                all(entry["verified"] for entry in manifest["model"]["required_files"])
            )
            self.assertFalse(manifest["source"]["evaluation_labels_read"])
            self.assertEqual(manifest["integrity"]["labels_or_session_files_opened"], [])
            self.assertGreaterEqual(manifest["build_resources"]["wall_seconds"], 0)
            self.assertIn("peak_rss_bytes", manifest["build_resources"])
            self.assertEqual(
                manifest["asset_byte_scope"]["manifest_bytes"],
                (output / MANIFEST_FILENAME).stat().st_size,
            )
            expected_total = (
                manifest["asset_byte_scope"]["required_asset_bytes_excluding_manifest"]
                + manifest["asset_byte_scope"]["manifest_bytes"]
            )
            self.assertEqual(
                manifest["asset_byte_scope"]["required_asset_bytes"], expected_total
            )
            self.assertIsNone(manifest["asset_byte_scope"]["manifest_sha256"])

    def test_model_spec_hash_is_canonical_not_raw_file_hash(self) -> None:
        first = {"z": [3, 2, 1], "a": {"é": True}}
        second = {"a": {"é": True}, "z": [3, 2, 1]}
        self.assertEqual(
            canonical_model_spec_sha256(first), canonical_model_spec_sha256(second)
        )
        raw = json.dumps(first, indent=4, ensure_ascii=True).encode("utf-8")
        self.assertNotEqual(
            canonical_model_spec_sha256(first), hashlib.sha256(raw).hexdigest()
        )

    def test_existing_output_fails_before_model_factory_and_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            output = fixture["output"]
            assert isinstance(output, Path)
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("do not replace", encoding="utf-8")
            factory_called = False

            def factory(_spec: dict, _model_dir: Path) -> FakeEncoder:
                nonlocal factory_called
                factory_called = True
                return FakeEncoder()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                build_semantic_index(
                    fixture["spec"],
                    fixture["catalog"],
                    fixture["model_dir"],
                    output,
                    encoder_factory=factory,
                    **_build_kwargs(fixture),
                )
            self.assertFalse(factory_called)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not replace")

    def test_catalog_and_model_hash_drift_fail_before_encoding_or_output(self) -> None:
        for drift in ("catalog", "model"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as directory:
                fixture = _fixture(Path(directory))
                kwargs = _build_kwargs(fixture)
                if drift == "catalog":
                    kwargs["expected_catalog_sha256"] = "0" * 64
                    pattern = "catalog SHA-256 mismatch"
                else:
                    model_dir = fixture["model_dir"]
                    assert isinstance(model_dir, Path)
                    (model_dir / "config.json").write_bytes(b"drift")
                    pattern = "model asset size mismatch"
                called = False

                def factory(_spec: dict, _model_dir: Path) -> FakeEncoder:
                    nonlocal called
                    called = True
                    return FakeEncoder()

                with self.assertRaisesRegex(ValueError, pattern):
                    build_semantic_index(
                        fixture["spec"],
                        fixture["catalog"],
                        fixture["model_dir"],
                        fixture["output"],
                        encoder_factory=factory,
                        **kwargs,
                    )
                self.assertFalse(called)
                output = fixture["output"]
                assert isinstance(output, Path)
                self.assertFalse(output.exists())

    def test_rejects_duplicate_asins_and_asset_path_traversal(self) -> None:
        for failure in ("duplicate_asin", "asset_escape"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                fixture = _fixture(Path(directory))
                if failure == "duplicate_asin":
                    catalog = fixture["catalog"]
                    assert isinstance(catalog, Path)
                    rows = [
                        json.loads(line)
                        for line in catalog.read_text(encoding="utf-8").splitlines()
                    ]
                    rows[-1]["parent_asin"] = rows[0]["parent_asin"]
                    _write_jsonl(catalog, rows)
                    pattern = "must be unique"
                else:
                    spec_path = fixture["spec"]
                    assert isinstance(spec_path, Path)
                    spec = json.loads(spec_path.read_text(encoding="utf-8"))
                    spec["required_files"][0]["path"] = "../outside.bin"
                    spec_path.write_text(json.dumps(spec), encoding="utf-8")
                    pattern = "must stay below model_dir"
                with self.assertRaisesRegex(ValueError, pattern):
                    build_semantic_index(
                        fixture["spec"],
                        fixture["catalog"],
                        fixture["model_dir"],
                        fixture["output"],
                        encoder_factory=lambda _spec, _model_dir: FakeEncoder(),
                        **_build_kwargs(fixture),
                    )
                output = fixture["output"]
                assert isinstance(output, Path)
                self.assertFalse(output.exists())

    def test_invalid_encoder_outputs_never_publish_partial_directory(self) -> None:
        class InvalidEncoder(FakeEncoder):
            def __init__(self, mode: str) -> None:
                super().__init__()
                self.mode = mode

            def encode_documents(self, texts: list[str]) -> object:
                import numpy

                if self.mode == "shape":
                    return numpy.ones((len(texts), 2), dtype=numpy.float32)
                if self.mode == "nan":
                    values = numpy.zeros((len(texts), 3), dtype=numpy.float32)
                    values[:, 0] = numpy.nan
                    return values
                return numpy.ones((len(texts), 3), dtype=numpy.float32)

        for mode, pattern in (
            ("shape", "encoder batch shape"),
            ("nan", "non-finite"),
            ("norm", "not frozen L2-normalized"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                fixture = _fixture(Path(directory))
                encoder = InvalidEncoder(mode)
                with self.assertRaisesRegex(ValueError, pattern):
                    build_semantic_index(
                        fixture["spec"],
                        fixture["catalog"],
                        fixture["model_dir"],
                        fixture["output"],
                        encoder_factory=lambda _spec, _model_dir: encoder,
                        **_build_kwargs(fixture),
                    )
                output = fixture["output"]
                assert isinstance(output, Path)
                self.assertFalse(output.exists())
                self.assertEqual(list(output.parent.glob(f".{output.name}.tmp-*")), [])
                self.assertTrue(encoder.closed)

    def test_failed_atomic_publish_cleans_owned_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            encoder = FakeEncoder()
            with patch(
                "scripts.build_p7_semantic_index.os.rename",
                side_effect=OSError("simulated publish failure"),
            ), self.assertRaisesRegex(OSError, "publish failure"):
                build_semantic_index(
                    fixture["spec"],
                    fixture["catalog"],
                    fixture["model_dir"],
                    fixture["output"],
                    encoder_factory=lambda _spec, _model_dir: encoder,
                    **_build_kwargs(fixture),
                )
            output = fixture["output"]
            assert isinstance(output, Path)
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(f".{output.name}.tmp-*")), [])
            self.assertTrue(encoder.closed)

    def test_frozen_environment_is_applied_and_fresh_import_can_be_enforced(self) -> None:
        spec = {
            "runtime": {
                "environment_before_numpy_or_onnxruntime_import": {
                    "P7_BUILDER_ENV_TEST": "frozen-value"
                }
            }
        }
        with patch.dict(sys.modules, {"numpy": object()}):
            with self.assertRaisesRegex(RuntimeError, "imported before frozen environment"):
                prepare_runtime_environment(spec, enforce_fresh_import=True)
        self.assertEqual(os.environ["P7_BUILDER_ENV_TEST"], "frozen-value")

    def test_parser_defaults_and_main_keep_fresh_import_boundary_mandatory(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.spec, DEFAULT_SPEC)
        self.assertEqual(args.catalog, DEFAULT_CATALOG)
        self.assertEqual(args.model_dir, DEFAULT_MODEL_DIR)
        self.assertEqual(args.output_dir, DEFAULT_OUTPUT_DIR)
        fake_manifest = {
            "rows": 5,
            "dimensions": 3,
            "matrix": {"sha256": "a" * 64},
        }
        with patch(
            "scripts.build_p7_semantic_index.build_semantic_index",
            return_value=fake_manifest,
        ) as build:
            result = main(
                [
                    "--spec",
                    "frozen.json",
                    "--catalog",
                    "catalog.jsonl",
                    "--model-dir",
                    "model",
                    "--output-dir",
                    "index",
                    "--rss-sample-ms",
                    "2.5",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            build.call_args.args,
            (
                Path("frozen.json"),
                Path("catalog.jsonl"),
                Path("model"),
                Path("index"),
            ),
        )
        self.assertEqual(build.call_args.kwargs["rss_sample_ms"], 2.5)
        self.assertIs(build.call_args.kwargs["enforce_fresh_runtime_import"], True)

    def test_builder_source_has_no_evaluator_or_label_file_dependency(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "build_p7_semantic_index.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("from evaluator", source)
        self.assertNotIn("import evaluator", source)
        self.assertNotIn("public_set.jsonl", source)
        self.assertNotIn("ground_truth", source)


if __name__ == "__main__":
    unittest.main()
