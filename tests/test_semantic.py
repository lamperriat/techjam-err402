from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from starter.semantic import (
    DEFAULT_INDEX_MANIFEST,
    INDEX_SCHEMA_VERSION,
    DenseHit,
    OfflineSemanticEncoder,
    RuntimeModules,
    SemanticAssetError,
    SemanticContractError,
    SemanticIndex,
    SemanticRuntimeError,
    canonical_document,
    canonical_json_sha256,
    load_runtime_modules,
    load_semantic_spec,
    normalize_text,
    prepare_runtime_environment,
    serialize_catalog_value,
    validate_model_assets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "configs" / "p7_bge_small_en_v1_5.json"
# unittest discovery may execute another module that imports NumPy before this class's
# setUpClass. Establish the test process environment at module discovery time; production
# workers still use the stricter sys.modules-aware path exercised below.
_BOOTSTRAP_SPEC = load_semantic_spec(SPEC_PATH)
prepare_runtime_environment(_BOOTSTRAP_SPEC, loaded_modules=set())


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _small_spec(base: dict, rows: int = 3, dimensions: int = 3) -> dict:
    spec = copy.deepcopy(base)
    spec["encoder"]["embedding_dimensions"] = dimensions
    spec["encoder"]["batch_size"] = 2
    spec["encoder"]["max_length"] = 8
    spec["index"]["shape"] = [rows, dimensions]
    tokenizer = b"fake-tokenizer"
    model = b"fake-onnx-model"
    spec["required_files"] = [
        {
            "path": "tokenizer.json",
            "bytes": len(tokenizer),
            "sha256": _sha(tokenizer),
        },
        {
            "path": "onnx/model.onnx",
            "bytes": len(model),
            "sha256": _sha(model),
        },
    ]
    return spec


def _write_model_assets(root: Path) -> None:
    (root / "onnx").mkdir(parents=True)
    (root / "tokenizer.json").write_bytes(b"fake-tokenizer")
    (root / "onnx" / "model.onnx").write_bytes(b"fake-onnx-model")


class _FakeEncoding:
    def __init__(self, width: int) -> None:
        self.ids = [101] + [7] * width + [102]
        self.attention_mask = [1] * len(self.ids)
        self.type_ids = [0] * len(self.ids)


class _FakeTokenizer:
    last: "_FakeTokenizer | None" = None

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.truncation: dict | None = None
        self.no_padding_called = False
        _FakeTokenizer.last = self

    @classmethod
    def from_file(cls, _: str) -> "_FakeTokenizer":
        return cls()

    def no_padding(self) -> None:
        self.no_padding_called = True

    def enable_truncation(self, **kwargs: object) -> None:
        self.truncation = dict(kwargs)

    def token_to_id(self, token: str) -> int | None:
        return 0 if token == "[PAD]" else None

    def encode_batch(
        self, texts: list[str], *, add_special_tokens: bool
    ) -> list[_FakeEncoding]:
        if not add_special_tokens:
            raise AssertionError("special tokens must stay enabled")
        self.batches.append(list(texts))
        return [_FakeEncoding(1 + (len(text) % 3)) for text in texts]


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    last: "_FakeSession | None" = None

    def __init__(self, _: str, *, sess_options: object, providers: list[str]) -> None:
        self.options = sess_options
        self.providers = providers
        self.last_feeds: dict | None = None
        _FakeSession.last = self

    def get_providers(self) -> list[str]:
        return list(self.providers)

    def get_inputs(self) -> list[_FakeInput]:
        return [
            _FakeInput("input_ids"),
            _FakeInput("attention_mask"),
            _FakeInput("token_type_ids"),
        ]

    def run(self, _: object, feeds: dict) -> list[object]:
        self.last_feeds = feeds
        np = importlib.import_module("numpy")
        batch, width = feeds["input_ids"].shape
        hidden = np.zeros((batch, width, 3), dtype=np.float32)
        hidden[:, 0, :] = np.asarray([3.0, 4.0, 0.0], dtype=np.float32)
        return [hidden]


class _FakeSessionOptions:
    pass


class _FakeOrt:
    __version__ = "1.29.0"
    SessionOptions = _FakeSessionOptions
    InferenceSession = _FakeSession
    ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
    GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")


class _FakeTokenizers:
    __version__ = "0.23.1"
    Tokenizer = _FakeTokenizer


def _write_index(
    root: Path,
    spec: dict,
    np: object,
    *,
    asins: tuple[str, ...] = ("A", "B", "C"),
    matrix_values: object | None = None,
    catalog_sha256: str = "c" * 64,
) -> dict:
    matrix = (
        np.asarray(matrix_values, dtype=np.float32)
        if matrix_values is not None
        else np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=np.float32,
        )
    )
    matrix_path = root / "catalog.npy"
    asins_path = root / "ordered_asins.txt"
    with matrix_path.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    asins_path.write_bytes(("\n".join(asins) + "\n").encode("utf-8"))
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "model_spec_serialization": (
            "UTF-8 canonical JSON; object keys sorted; compact separators; "
            "ensure_ascii=false"
        ),
        "model_spec_sha256": canonical_json_sha256(spec),
        "catalog_sha256": catalog_sha256,
        "rows": int(matrix.shape[0]),
        "dimensions": int(matrix.shape[1]),
        "matrix": {
            "path": matrix_path.name,
            "bytes": matrix_path.stat().st_size,
            "sha256": _sha(matrix_path.read_bytes()),
            "dtype": "float32",
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "format": "NumPy .npy",
        },
        "ordered_asins": {
            "path": asins_path.name,
            "bytes": asins_path.stat().st_size,
            "sha256": _sha(asins_path.read_bytes()),
            "count": len(asins),
            "encoding": "utf-8-lf",
        },
    }
    (root / DEFAULT_INDEX_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return manifest


class SemanticCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.full_spec = _BOOTSTRAP_SPEC
        cls.np = importlib.import_module("numpy")

    def test_module_has_no_direct_optional_runtime_import(self) -> None:
        tree = ast.parse(
            (PROJECT_ROOT / "starter" / "semantic.py").read_text(encoding="utf-8")
        )
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue({"numpy", "tokenizers", "onnxruntime"}.isdisjoint(imported))

    def test_recursive_catalog_document_is_exact_and_deterministic(self) -> None:
        product = {
            "title": "  Ｔee\tshirt  ",
            "categories": ["Women", None, "", ["Summer", "Casual"]],
            "details": {
                "z": [1, None, True],
                "a": False,
                " A ": {"b": 2.5, "A": " x\n y "},
                "omitted": None,
            },
            "features": [],
            "store": "  Example\u00a0 Shop ",
            "description": "",
            "ignored": "must not appear",
        }
        expected = (
            "title: Tee shirt | categories: Women ; Summer ; Casual | "
            "details: A = A = x y ; b = 2.5 ; a = false ; "
            "z = 1 ; true | store: Example Shop"
        )
        self.assertEqual(
            canonical_document(product, self.full_spec["document"]), expected
        )
        self.assertEqual(normalize_text(" A\u2003 B\nＣ "), "A B C")
        self.assertEqual(serialize_catalog_value({"b": 1, "A": 2}), "A = 2 ; b = 1")
        with self.assertRaisesRegex(SemanticContractError, "finite"):
            serialize_catalog_value(float("nan"))
        with self.assertRaisesRegex(SemanticContractError, "unsupported"):
            serialize_catalog_value({1, 2})

    def test_model_assets_require_safe_paths_exact_sizes_and_hashes(self) -> None:
        spec = _small_spec(self.full_spec)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_model_assets(root)
            validated = validate_model_assets(root, spec)
            self.assertEqual([row["path"] for row in validated], ["tokenizer.json", "onnx/model.onnx"])

            drifted = copy.deepcopy(spec)
            drifted["required_files"][0]["bytes"] += 1
            with self.assertRaisesRegex(SemanticAssetError, "size mismatch"):
                validate_model_assets(root, drifted)

            drifted = copy.deepcopy(spec)
            drifted["required_files"][1]["sha256"] = "0" * 64
            with self.assertRaisesRegex(SemanticAssetError, "SHA-256 mismatch"):
                validate_model_assets(root, drifted)

            unsafe = copy.deepcopy(spec)
            unsafe["required_files"][0]["path"] = "../escape"
            with self.assertRaisesRegex(SemanticContractError, "unsafe"):
                validate_model_assets(root, unsafe)

    def test_environment_is_set_before_dynamic_import_and_conflicts_fail_late(self) -> None:
        required = dict(
            self.full_spec["runtime"][
                "environment_before_numpy_or_onnxruntime_import"
            ]
        )
        custom = {"OMP_NUM_THREADS": "99"}
        observed = prepare_runtime_environment(
            self.full_spec, environ=custom, loaded_modules=set()
        )
        self.assertEqual(observed, required)
        self.assertEqual({key: custom[key] for key in required}, required)

        custom["OMP_NUM_THREADS"] = "99"
        with self.assertRaisesRegex(SemanticRuntimeError, "before runtime import"):
            prepare_runtime_environment(
                self.full_spec, environ=custom, loaded_modules={"numpy"}
            )

        fake_modules = {
            "numpy": SimpleNamespace(__version__="2.4.6"),
            "tokenizers": SimpleNamespace(__version__="0.23.1"),
            "onnxruntime": SimpleNamespace(__version__="1.29.0"),
        }

        def dynamic_import(name: str) -> object:
            self.assertEqual(
                {key: os.environ.get(key) for key in required}, required
            )
            return fake_modules[name]

        with patch("starter.semantic.importlib.import_module", side_effect=dynamic_import):
            modules = load_runtime_modules(self.full_spec)
        self.assertIs(modules.numpy, fake_modules["numpy"])

    def test_encoder_batches_cls_l2_float32_and_query_instruction(self) -> None:
        spec = _small_spec(self.full_spec)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_model_assets(root)
            modules = RuntimeModules(self.np, _FakeTokenizers, _FakeOrt)
            with patch("starter.semantic.load_runtime_modules", return_value=modules):
                encoder = OfflineSemanticEncoder.from_frozen_assets(spec, root)

            documents = encoder.encode_documents(["a", "bb", "ccc"])
            self.assertEqual(documents.dtype, self.np.float32)
            self.assertEqual(documents.shape, (3, 3))
            self.np.testing.assert_array_equal(
                documents,
                self.np.asarray(
                    [[0.6, 0.8, 0.0], [0.6, 0.8, 0.0], [0.6, 0.8, 0.0]],
                    dtype=self.np.float32,
                ),
            )
            tokenizer = _FakeTokenizer.last
            session = _FakeSession.last
            self.assertIsNotNone(tokenizer)
            self.assertIsNotNone(session)
            assert tokenizer is not None and session is not None
            self.assertEqual(len(tokenizer.batches), 2)
            self.assertEqual(tokenizer.truncation, {"max_length": 8, "direction": "right"})
            self.assertTrue(tokenizer.no_padding_called)
            self.assertEqual(session.last_feeds["input_ids"].dtype, self.np.int64)
            self.assertIn("token_type_ids", session.last_feeds)
            self.assertEqual(session.options.execution_mode, "sequential")
            self.assertEqual(session.options.graph_optimization_level, "all")
            self.assertEqual(session.options.intra_op_num_threads, 1)
            self.assertEqual(session.options.inter_op_num_threads, 1)

            query = encoder.encode_query("red shoes")
            self.np.testing.assert_array_equal(
                query, self.np.asarray([0.6, 0.8, 0.0], dtype=self.np.float32)
            )
            self.assertEqual(
                tokenizer.batches[-1][0],
                spec["encoder"]["query_instruction"] + "red shoes",
            )
            before = len(tokenizer.batches)
            with self.assertRaisesRegex(SemanticContractError, "empty queries"):
                encoder.encode_query("  ")
            with self.assertRaisesRegex(SemanticContractError, "must be strings"):
                encoder.encode_queries([123])  # type: ignore[list-item]
            self.assertEqual(len(tokenizer.batches), before)
            encoder.close()
            with self.assertRaisesRegex(SemanticRuntimeError, "closed"):
                encoder.encode_documents(["x"])

    def test_encoder_rejects_zero_norm_or_wrong_hidden_shape(self) -> None:
        spec = _small_spec(self.full_spec)
        tokenizer = _FakeTokenizer()

        class BadSession(_FakeSession):
            def run(self, _: object, feeds: dict) -> list[object]:
                return [self_np.zeros((len(feeds["input_ids"]), 1, 2), dtype=self_np.float32)]

        self_np = self.np
        session = BadSession("unused", sess_options=object(), providers=["CPUExecutionProvider"])
        encoder = OfflineSemanticEncoder(spec, tokenizer, session, self.np, 0)
        with self.assertRaisesRegex(SemanticRuntimeError, "hidden-state shape"):
            encoder.encode_documents(["x"])

        class ZeroSession(_FakeSession):
            def run(self, _: object, feeds: dict) -> list[object]:
                return [self_np.zeros((len(feeds["input_ids"]), 2, 3), dtype=self_np.float32)]

        encoder = OfflineSemanticEncoder(
            spec,
            _FakeTokenizer(),
            ZeroSession("unused", sess_options=object(), providers=["CPUExecutionProvider"]),
            self.np,
            0,
        )
        with self.assertRaisesRegex(SemanticRuntimeError, "zero-norm"):
            encoder.encode_documents(["x"])

    def test_index_load_validates_manifest_and_exact_float32_tie_order(self) -> None:
        spec = _small_spec(self.full_spec)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_index(root, spec, self.np)
            index = SemanticIndex.load(
                spec,
                root,
                expected_catalog_sha256="c" * 64,
                numpy_module=self.np,
            )
            self.assertEqual(index.manifest, manifest)
            hits = index.search_vector(
                self.np.asarray([1.0, 0.0, 0.0], dtype=self.np.float32), top_k=3
            )
            self.assertEqual(
                hits,
                [DenseHit("A", 1.0), DenseHit("B", 1.0), DenseHit("C", 0.5)],
            )
            self.assertTrue(all(isinstance(hit.score, float) for hit in hits))

            class BombEncoder:
                def encode_query(self, _: str) -> object:
                    raise AssertionError("empty query must not invoke encoder")

            self.assertEqual(index.search_query(" \t\n", BombEncoder()), [])
            index.close()
            with self.assertRaisesRegex(SemanticRuntimeError, "closed"):
                index.search_vector([1.0, 0.0, 0.0])

    def test_index_rejects_identity_shape_hash_and_asin_order_drift(self) -> None:
        spec = _small_spec(self.full_spec)

        def attempt(mutator: object, pattern: str) -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = _write_index(root, spec, self.np)
                mutator(root, manifest)
                (root / DEFAULT_INDEX_MANIFEST).write_text(
                    json.dumps(manifest, sort_keys=True), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    (SemanticContractError, SemanticAssetError), pattern
                ):
                    SemanticIndex.load(
                        spec,
                        root,
                        expected_catalog_sha256="c" * 64,
                        numpy_module=self.np,
                    )

        attempt(
            lambda _root, manifest: manifest.__setitem__(
                "model_spec_sha256", "0" * 64
            ),
            "model-spec SHA-256 mismatch",
        )
        attempt(
            lambda _root, manifest: manifest.__setitem__("catalog_sha256", "d" * 64),
            "catalog SHA-256 mismatch",
        )
        attempt(
            lambda _root, manifest: manifest["matrix"].__setitem__("shape", [2, 3]),
            "matrix shape mismatch",
        )

        def corrupt_matrix(root: Path, _manifest: dict) -> None:
            payload = bytearray((root / "catalog.npy").read_bytes())
            payload[0] ^= 1
            (root / "catalog.npy").write_bytes(payload)

        attempt(corrupt_matrix, "matrix SHA-256 mismatch")

        def wrong_matrix_dtype(root: Path, manifest: dict) -> None:
            path = root / "catalog.npy"
            with path.open("wb") as handle:
                self.np.save(
                    handle,
                    self.np.ones((3, 3), dtype=self.np.float64),
                    allow_pickle=False,
                )
            manifest["matrix"]["bytes"] = path.stat().st_size
            manifest["matrix"]["sha256"] = _sha(path.read_bytes())

        attempt(wrong_matrix_dtype, "matrix dtype.*float32")

        def fortran_matrix(root: Path, manifest: dict) -> None:
            path = root / "catalog.npy"
            values = self.np.asfortranarray(
                self.np.ones((3, 3), dtype=self.np.float32)
            )
            with path.open("wb") as handle:
                self.np.save(handle, values, allow_pickle=False)
            manifest["matrix"]["bytes"] = path.stat().st_size
            manifest["matrix"]["sha256"] = _sha(path.read_bytes())

        attempt(fortran_matrix, "C-contiguous")

        def reorder_asins(root: Path, manifest: dict) -> None:
            path = root / "ordered_asins.txt"
            path.write_bytes(b"B\nA\nC\n")
            manifest["ordered_asins"]["sha256"] = _sha(path.read_bytes())

        attempt(reorder_asins, "unique ascending row order")

    def test_index_rejects_nonfinite_vectors_and_invalid_top_k(self) -> None:
        spec = _small_spec(self.full_spec)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_index(root, spec, self.np)
            index = SemanticIndex.load(
                spec,
                root,
                expected_catalog_sha256="c" * 64,
                numpy_module=self.np,
            )
            with self.assertRaisesRegex(SemanticContractError, "shape"):
                index.search_vector([1.0, 0.0])
            with self.assertRaisesRegex(SemanticContractError, "non-finite"):
                index.search_vector([float("nan"), 0.0, 0.0])
            with self.assertRaisesRegex(SemanticContractError, "top_k"):
                index.search_vector([1.0, 0.0, 0.0], top_k=-1)
            self.assertEqual(index.search_vector([1.0, 0.0, 0.0], top_k=0), [])
            index.close()


if __name__ == "__main__":
    unittest.main()
