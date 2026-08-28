from __future__ import annotations

"""Offline, deterministic semantic retrieval primitives for the P7 shadow study.

This module intentionally imports only the Python standard library at module load.
NumPy, tokenizers, and ONNX Runtime are imported dynamically only after the frozen
runtime environment has been installed and verified.
"""

import hashlib
import importlib
import json
import math
import os
import platform
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple


SPEC_SCHEMA_VERSION = "p7.semantic-model-spec.v1"
DOCUMENT_SCHEMA_VERSION = "p7.catalog-document.v1"
INDEX_SCHEMA_VERSION = "p7.semantic-index.v1"
DEFAULT_SPEC_PATH = Path("configs/p7_bge_small_en_v1_5.json")
DEFAULT_INDEX_MANIFEST = "semantic-index.manifest.json"
_OPTIONAL_RUNTIME_MODULES = ("numpy", "tokenizers", "onnxruntime")


class SemanticContractError(ValueError):
    """A frozen semantic contract or its input is invalid."""


class SemanticAssetError(SemanticContractError):
    """A required model or index asset failed identity validation."""


class SemanticRuntimeError(RuntimeError):
    """The optional local semantic runtime cannot be initialized safely."""


class DenseHit(NamedTuple):
    parent_asin: str
    score: float


@dataclass(frozen=True)
class RuntimeModules:
    numpy: Any
    tokenizers: Any
    onnxruntime: Any


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticContractError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticContractError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SemanticContractError(f"JSON root must be an object: {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the frozen platform-independent compact JSON representation."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SemanticContractError(f"value is not canonical JSON: {exc}") from exc


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise SemanticAssetError(f"cannot hash asset {path}: {exc}") from exc
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticContractError(message)


def _validate_document_contract(document: Mapping[str, Any]) -> None:
    _require(
        document.get("schema_version") == DOCUMENT_SCHEMA_VERSION,
        "wrong document schema",
    )
    _require(
        document.get("unicode_normalization") == "NFKC",
        "unsupported Unicode normalization",
    )
    _require(
        document.get("whitespace") == "collapse_to_single_ascii_space",
        "unsupported whitespace normalization",
    )
    _require(isinstance(document.get("field_order"), list), "document field order missing")
    _require(
        len(document["field_order"]) == len(set(document["field_order"])),
        "document field order contains duplicates",
    )
    _require(
        all(isinstance(field, str) and field for field in document["field_order"]),
        "document field names must be non-empty strings",
    )
    _require(
        document.get("mapping_order") == "casefolded_key_then_original_key",
        "unsupported mapping order",
    )
    _require(
        document.get("sequence_order") == "preserve_catalog_order",
        "unsupported sequence order",
    )
    _require(
        document.get("field_format") == "field_name: normalized_value",
        "unsupported field format",
    )
    _require(document.get("field_separator") == " | ", "unsupported field separator")
    _require(document.get("missing_fields") == "omit", "unsupported missing-field policy")
    serialization = document.get("value_serialization")
    _require(isinstance(serialization, Mapping), "value-serialization contract missing")
    _require(serialization.get("null_or_empty_string") == "omit", "unsupported null policy")
    _require(
        serialization.get("string")
        == "NFKC then collapse Unicode whitespace to one ASCII space and strip",
        "unsupported string serialization",
    )
    _require(
        serialization.get("boolean") == "lowercase JSON literal true or false",
        "unsupported boolean serialization",
    )
    _require(
        serialization.get("number") == "locale-independent JSON number",
        "unsupported number serialization",
    )
    _require(
        serialization.get("sequence")
        == (
            "recursively serialize non-omitted elements in source order and join "
            "with space-semicolon-space"
        ),
        "unsupported sequence serialization",
    )
    _require(
        serialization.get("mapping")
        == (
            "normalize keys as strings; sort by (normalized_key.casefold(), "
            "normalized_key); recursively serialize values; render non-omitted "
            "entries as key space-equals-space value and join with "
            "space-semicolon-space"
        ),
        "unsupported mapping serialization",
    )
    _require(
        serialization.get("nested_values") == "apply these rules recursively",
        "unsupported nested-value serialization",
    )
    _require(
        serialization.get("separator_escaping")
        == "none; literal separators in source content remain literal after normalization",
        "unsupported separator escaping",
    )


def _safe_asset_path(root: Path, relative: Any, label: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), f"{label} path is invalid")
    candidate = Path(relative)
    _require(not candidate.is_absolute(), f"{label} path must be relative")
    _require(
        all(part not in {"", ".", ".."} for part in candidate.parts),
        f"{label} path contains an unsafe component",
    )
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise SemanticContractError(f"{label} path escapes its asset root") from exc
    return resolved


def validate_semantic_spec(spec: Mapping[str, Any]) -> None:
    """Validate all implementation-relevant fields in the frozen P7 model spec."""

    _require(spec.get("schema_version") == SPEC_SCHEMA_VERSION, "wrong model spec schema")
    runtime = spec.get("runtime")
    encoder = spec.get("encoder")
    document = spec.get("document")
    index = spec.get("index")
    required_files = spec.get("required_files")
    for value, label in (
        (runtime, "runtime"),
        (encoder, "encoder"),
        (document, "document"),
        (index, "index"),
    ):
        _require(isinstance(value, Mapping), f"model spec {label} must be an object")
    _require(isinstance(required_files, list), "required_files must be a list")

    environment = runtime.get("environment_before_numpy_or_onnxruntime_import")
    _require(isinstance(environment, Mapping) and environment, "runtime environment missing")
    _require(
        all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()),
        "runtime environment keys and values must be strings",
    )
    _require(
        dict(environment)
        == {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
        },
        "unsupported frozen runtime environment",
    )
    _require(runtime.get("execution_provider") == "CPUExecutionProvider", "CPU provider required")
    _require(runtime.get("execution_mode") == "ORT_SEQUENTIAL", "sequential ORT required")
    _require(
        runtime.get("graph_optimization_level") == "ORT_ENABLE_ALL",
        "frozen ORT optimization level required",
    )
    _require(runtime.get("intra_op_threads") == 1, "ORT intra-op threads must be one")
    _require(runtime.get("inter_op_threads") == 1, "ORT inter-op threads must be one")
    _require(runtime.get("network_required") is False, "semantic runtime must be offline")

    dimensions = encoder.get("embedding_dimensions")
    max_length = encoder.get("max_length")
    batch_size = encoder.get("batch_size")
    _require(isinstance(dimensions, int) and dimensions > 0, "invalid embedding dimensions")
    _require(isinstance(max_length, int) and max_length > 0, "invalid encoder max length")
    _require(isinstance(batch_size, int) and batch_size > 0, "invalid encoder batch size")
    _require(
        isinstance(encoder.get("model_supported_max_length"), int)
        and max_length <= encoder["model_supported_max_length"],
        "encoder cutoff exceeds the model-supported length",
    )
    _require(
        encoder.get("max_length_includes_special_tokens") is True,
        "max length must include special tokens",
    )
    _require(encoder.get("pooling") == "first_token_cls", "unsupported pooling")
    _require(encoder.get("normalization") == "l2_float32", "unsupported normalization")
    _require(encoder.get("padding") == "longest_in_batch", "unsupported padding")
    _require(encoder.get("padding_side") == "right", "unsupported padding side")
    _require(encoder.get("truncation") is True, "encoder truncation must be enabled")
    _require(encoder.get("truncation_side") == "right", "unsupported truncation side")
    _require(encoder.get("add_special_tokens") is True, "special tokens must be enabled")
    _require(encoder.get("input_ids_dtype") == "int64", "input IDs must be int64")
    _require(encoder.get("attention_mask_dtype") == "int64", "attention mask must be int64")
    _require(
        encoder.get("token_type_ids")
        == (
            "use tokenizer output when present; otherwise an int64 zero array matching "
            "input_ids"
        ),
        "unsupported token-type ID policy",
    )
    _require(isinstance(encoder.get("query_instruction"), str), "query instruction missing")
    _require(isinstance(encoder.get("document_instruction"), str), "document instruction missing")

    _validate_document_contract(document)

    query = spec.get("query")
    _require(isinstance(query, Mapping), "model spec query must be an object")
    _require(
        query.get("schema_version") == "p7.visible-state-query.v1",
        "wrong semantic-query schema",
    )
    _require(query.get("term_separator") == " ", "unsupported query-term separator")
    _require(query.get("excluded_terms_removed") is True, "excluded query terms must be removed")
    _require(query.get("target_blind") is True, "semantic query must remain target-blind")
    _require(
        query.get("empty_query")
        == "return an empty dense route without invoking the tokenizer or model",
        "unsupported empty-query policy",
    )

    shape = index.get("shape")
    _require(
        isinstance(shape, list)
        and len(shape) == 2
        and all(isinstance(value, int) and value > 0 for value in shape),
        "invalid index shape",
    )
    _require(shape[1] == dimensions, "index and encoder dimensions disagree")
    _require(index.get("catalog_order") == "parent_asin_ascending", "unsupported catalog order")
    _require(index.get("matrix_dtype") == "float32", "unsupported matrix dtype")
    _require(index.get("search") == "exact_full_matrix_dot_product", "unsupported search")
    _require(
        index.get("ranking") == "score_desc_then_parent_asin_asc",
        "unsupported ranking",
    )
    _require(index.get("ann_or_vector_database") is False, "ANN must remain disabled")

    seen_paths: set[str] = set()
    for position, entry in enumerate(required_files):
        _require(isinstance(entry, Mapping), f"required_files[{position}] must be an object")
        path = entry.get("path")
        _require(isinstance(path, str) and path not in seen_paths, "duplicate or invalid required path")
        seen_paths.add(path)
        _require(
            isinstance(entry.get("bytes"), int) and entry["bytes"] > 0,
            f"invalid required size for {path}",
        )
        _require(
            isinstance(entry.get("sha256"), str)
            and len(entry["sha256"]) == 64
            and all(char in "0123456789abcdef" for char in entry["sha256"]),
            f"invalid required SHA-256 for {path}",
        )


def load_semantic_spec(path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    spec = _read_json_object(path)
    validate_semantic_spec(spec)
    return spec


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def serialize_catalog_value(value: Any) -> str | None:
    """Recursively serialize one JSON catalog value under the frozen P7 rules."""

    if value is None:
        return None
    if isinstance(value, str):
        normalized = normalize_text(value)
        return normalized or None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SemanticContractError("catalog number must be finite")
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, Mapping):
        entries: list[tuple[str, str]] = []
        for raw_key, raw_value in value.items():
            key = normalize_text(str(raw_key))
            serialized = serialize_catalog_value(raw_value)
            if serialized is not None:
                entries.append((key, serialized))
        entries.sort(key=lambda item: (item[0].casefold(), item[0]))
        if not entries:
            return None
        return " ; ".join(f"{key} = {serialized}" for key, serialized in entries)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        serialized_items = [
            serialized
            for item in value
            if (serialized := serialize_catalog_value(item)) is not None
        ]
        return " ; ".join(serialized_items) if serialized_items else None
    raise SemanticContractError(
        f"catalog value has unsupported non-JSON type {type(value).__name__}"
    )


def canonical_document(
    product: Mapping[str, Any], document_spec: Mapping[str, Any]
) -> str:
    """Serialize one catalog product in the exact field and recursive value order."""

    _validate_document_contract(document_spec)
    fields = document_spec.get("field_order")
    _require(isinstance(fields, list), "document field order missing")
    separator = document_spec.get("field_separator")
    _require(isinstance(separator, str), "document field separator missing")
    parts: list[str] = []
    for raw_field in fields:
        _require(isinstance(raw_field, str), "document field names must be strings")
        serialized = serialize_catalog_value(product.get(raw_field))
        if serialized is not None:
            parts.append(f"{raw_field}: {serialized}")
    return separator.join(parts)


def validate_model_assets(
    model_dir: Path, spec: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Validate every required model file by safe path, exact size, and SHA-256."""

    validate_semantic_spec(spec)
    root = model_dir.resolve()
    if not root.is_dir():
        raise SemanticAssetError(f"model asset directory does not exist: {root}")
    validated: list[dict[str, Any]] = []
    for entry in spec["required_files"]:
        path = _safe_asset_path(root, entry["path"], "model asset")
        if not path.is_file():
            raise SemanticAssetError(f"required model asset is missing: {entry['path']}")
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise SemanticAssetError(
                f"model asset size mismatch for {entry['path']}: {size} != {entry['bytes']}"
            )
        digest = _file_sha256(path)
        if digest != entry["sha256"]:
            raise SemanticAssetError(
                f"model asset SHA-256 mismatch for {entry['path']}: "
                f"{digest} != {entry['sha256']}"
            )
        validated.append({"path": entry["path"], "bytes": size, "sha256": digest})
    return tuple(validated)


def prepare_runtime_environment(
    spec: Mapping[str, Any],
    *,
    environ: Any | None = None,
    loaded_modules: set[str] | None = None,
) -> dict[str, str]:
    """Install frozen environment values before optional numerical imports.

    In a fresh worker inherited values are overwritten deterministically. If any optional
    runtime is already loaded, every value must already match because changing it is too
    late to guarantee the frozen execution contract.
    """

    validate_semantic_spec(spec)
    target = os.environ if environ is None else environ
    loaded = set(sys.modules) if loaded_modules is None else loaded_modules
    already_loaded = sorted(set(_OPTIONAL_RUNTIME_MODULES) & loaded)
    required = dict(spec["runtime"]["environment_before_numpy_or_onnxruntime_import"])
    if already_loaded:
        mismatches = {
            key: (target.get(key), value)
            for key, value in required.items()
            if target.get(key) != value
        }
        if mismatches:
            raise SemanticRuntimeError(
                "frozen environment was not established before runtime import "
                f"({', '.join(already_loaded)}): {mismatches}"
            )
    else:
        for key, value in required.items():
            target[key] = value
    observed = {key: target.get(key) for key in required}
    if observed != required:
        raise SemanticRuntimeError(
            f"failed to establish frozen semantic environment: {observed}"
        )
    return required


def _module_version(module: Any, name: str) -> str:
    version = getattr(module, "__version__", None)
    if not isinstance(version, str):
        raise SemanticRuntimeError(f"{name} does not expose a string __version__")
    return version


def load_runtime_modules(spec: Mapping[str, Any]) -> RuntimeModules:
    """Set/verify the environment, then dynamically import the pinned runtime."""

    prepare_runtime_environment(spec)
    expected_python = str(spec["runtime"]["python"])
    if platform.python_version() != expected_python:
        raise SemanticRuntimeError(
            f"Python version {platform.python_version()} != frozen {expected_python}"
        )
    try:
        numpy_module = importlib.import_module("numpy")
        tokenizers_module = importlib.import_module("tokenizers")
        onnxruntime_module = importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise SemanticRuntimeError(f"optional semantic dependency is unavailable: {exc}") from exc
    expected_versions = {
        "numpy": str(spec["runtime"]["numpy"]),
        "tokenizers": str(spec["runtime"]["tokenizers"]),
        "onnxruntime": str(spec["runtime"]["onnxruntime"]),
    }
    modules = {
        "numpy": numpy_module,
        "tokenizers": tokenizers_module,
        "onnxruntime": onnxruntime_module,
    }
    for name, module in modules.items():
        actual = _module_version(module, name)
        if actual != expected_versions[name]:
            raise SemanticRuntimeError(
                f"{name} version {actual} != frozen {expected_versions[name]}"
            )
    return RuntimeModules(numpy_module, tokenizers_module, onnxruntime_module)


class OfflineSemanticEncoder:
    """Pinned BGE-small CPU ONNX encoder using CLS pooling and float32 L2."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        tokenizer: Any,
        session: Any,
        numpy_module: Any,
        pad_id: int,
    ) -> None:
        self.spec = spec
        self._tokenizer = tokenizer
        self._session = session
        self._np = numpy_module
        self._pad_id = pad_id
        self._closed = False
        self._input_names = {item.name for item in session.get_inputs()}
        if not {"input_ids", "attention_mask"}.issubset(self._input_names):
            raise SemanticRuntimeError(
                "ONNX model must expose input_ids and attention_mask inputs"
            )

    @classmethod
    def from_frozen_assets(
        cls, spec: Mapping[str, Any], model_dir: Path
    ) -> "OfflineSemanticEncoder":
        validate_model_assets(model_dir, spec)
        modules = load_runtime_modules(spec)
        tokenizer_path = model_dir.resolve() / "tokenizer.json"
        model_path = model_dir.resolve() / "onnx" / "model.onnx"
        try:
            tokenizer = modules.tokenizers.Tokenizer.from_file(str(tokenizer_path))
            if hasattr(tokenizer, "no_padding"):
                tokenizer.no_padding()
            tokenizer.enable_truncation(
                max_length=int(spec["encoder"]["max_length"]), direction="right"
            )
            pad_id = tokenizer.token_to_id("[PAD]")
            if not isinstance(pad_id, int):
                raise SemanticRuntimeError("tokenizer does not define an integer [PAD] ID")

            options = modules.onnxruntime.SessionOptions()
            options.execution_mode = getattr(
                modules.onnxruntime.ExecutionMode,
                str(spec["runtime"]["execution_mode"]),
            )
            options.graph_optimization_level = getattr(
                modules.onnxruntime.GraphOptimizationLevel,
                str(spec["runtime"]["graph_optimization_level"]),
            )
            options.intra_op_num_threads = int(spec["runtime"]["intra_op_threads"])
            options.inter_op_num_threads = int(spec["runtime"]["inter_op_threads"])
            provider = str(spec["runtime"]["execution_provider"])
            session = modules.onnxruntime.InferenceSession(
                str(model_path), sess_options=options, providers=[provider]
            )
            providers = list(session.get_providers())
            if providers != [provider]:
                raise SemanticRuntimeError(
                    f"ONNX provider mismatch: {providers!r} != {[provider]!r}"
                )
        except SemanticRuntimeError:
            raise
        except Exception as exc:
            raise SemanticRuntimeError(f"cannot initialize frozen BGE runtime: {exc}") from exc
        return cls(spec, tokenizer, session, modules.numpy, pad_id)

    def _ensure_open(self) -> None:
        if self._closed or self._session is None:
            raise SemanticRuntimeError("semantic encoder is closed")

    def _encode_batch(self, texts: list[str]) -> Any:
        self._ensure_open()
        if not texts:
            return self._np.empty(
                (0, int(self.spec["encoder"]["embedding_dimensions"])),
                dtype=self._np.float32,
            )
        try:
            encodings = self._tokenizer.encode_batch(
                texts,
                add_special_tokens=bool(self.spec["encoder"]["add_special_tokens"]),
            )
        except Exception as exc:
            raise SemanticRuntimeError(f"tokenization failed: {exc}") from exc
        if len(encodings) != len(texts):
            raise SemanticRuntimeError("tokenizer returned the wrong batch size")
        max_length = int(self.spec["encoder"]["max_length"])
        rows: list[tuple[list[int], list[int], list[int]]] = []
        longest = 0
        for encoding in encodings:
            ids = list(encoding.ids)[:max_length]
            if not ids:
                raise SemanticRuntimeError("tokenizer produced an empty token sequence")
            attention = list(getattr(encoding, "attention_mask", []))[: len(ids)]
            if len(attention) != len(ids):
                attention = [1] * len(ids)
            token_types = list(getattr(encoding, "type_ids", []))[: len(ids)]
            if len(token_types) != len(ids):
                token_types = [0] * len(ids)
            longest = max(longest, len(ids))
            rows.append((ids, attention, token_types))
        padded_ids: list[list[int]] = []
        padded_attention: list[list[int]] = []
        padded_types: list[list[int]] = []
        for ids, attention, token_types in rows:
            padding = longest - len(ids)
            padded_ids.append(ids + [self._pad_id] * padding)
            padded_attention.append(attention + [0] * padding)
            padded_types.append(token_types + [0] * padding)
        feeds = {
            "input_ids": self._np.asarray(padded_ids, dtype=self._np.int64),
            "attention_mask": self._np.asarray(padded_attention, dtype=self._np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = self._np.asarray(
                padded_types, dtype=self._np.int64
            )
        try:
            outputs = self._session.run(None, feeds)
        except Exception as exc:
            raise SemanticRuntimeError(f"ONNX inference failed: {exc}") from exc
        if not outputs:
            raise SemanticRuntimeError("ONNX model returned no outputs")
        hidden = self._np.asarray(outputs[0], dtype=self._np.float32)
        dimensions = int(self.spec["encoder"]["embedding_dimensions"])
        if hidden.ndim != 3 or hidden.shape[0] != len(texts) or hidden.shape[2] != dimensions:
            raise SemanticRuntimeError(
                f"unexpected ONNX hidden-state shape {tuple(hidden.shape)}"
            )
        cls_vectors = self._np.array(hidden[:, 0, :], dtype=self._np.float32, copy=True)
        norms = self._np.sqrt(
            self._np.sum(
                cls_vectors * cls_vectors, axis=1, dtype=self._np.float32
            )
        ).astype(self._np.float32, copy=False)
        if bool(self._np.any(~self._np.isfinite(cls_vectors))) or bool(
            self._np.any(~self._np.isfinite(norms))
        ):
            raise SemanticRuntimeError("encoder output contains non-finite values")
        if bool(self._np.any(norms <= self._np.float32(0.0))):
            raise SemanticRuntimeError("encoder output contains a zero-norm CLS vector")
        normalized = (cls_vectors / norms[:, None]).astype(
            self._np.float32, copy=False
        )
        return normalized

    def _encode_texts(self, texts: Sequence[str], instruction: str) -> Any:
        values = list(texts)
        if any(not isinstance(value, str) for value in values):
            raise SemanticContractError("encoder inputs must be strings")
        batch_size = int(self.spec["encoder"]["batch_size"])
        batches = [
            self._encode_batch([instruction + value for value in values[start : start + batch_size]])
            for start in range(0, len(values), batch_size)
        ]
        if not batches:
            return self._np.empty(
                (0, int(self.spec["encoder"]["embedding_dimensions"])),
                dtype=self._np.float32,
            )
        return self._np.concatenate(batches, axis=0).astype(
            self._np.float32, copy=False
        )

    def encode_documents(self, documents: Sequence[str]) -> Any:
        return self._encode_texts(
            documents, str(self.spec["encoder"]["document_instruction"])
        )

    def encode_queries(self, queries: Sequence[str]) -> Any:
        values = list(queries)
        if any(not isinstance(value, str) for value in values):
            raise SemanticContractError("encoder inputs must be strings")
        if any(not value.strip() for value in values):
            raise SemanticContractError("empty queries must use the empty dense route")
        return self._encode_texts(
            values, str(self.spec["encoder"]["query_instruction"])
        )

    def encode_query(self, query: str) -> Any:
        if not isinstance(query, str) or not query.strip():
            raise SemanticContractError("empty queries must use the empty dense route")
        return self.encode_queries([query])[0]

    def close(self) -> None:
        self._session = None
        self._tokenizer = None
        self._closed = True

    def __enter__(self) -> "OfflineSemanticEncoder":
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _validate_index_file(
    root: Path, entry: Mapping[str, Any], label: str
) -> tuple[Path, int, str]:
    _require(isinstance(entry, Mapping), f"{label} manifest entry must be an object")
    path = _safe_asset_path(root, entry.get("path"), label)
    expected_bytes = entry.get("bytes")
    expected_hash = entry.get("sha256")
    _require(isinstance(expected_bytes, int) and expected_bytes > 0, f"invalid {label} bytes")
    _require(
        isinstance(expected_hash, str)
        and len(expected_hash) == 64
        and all(char in "0123456789abcdef" for char in expected_hash),
        f"invalid {label} SHA-256",
    )
    if not path.is_file():
        raise SemanticAssetError(f"{label} file is missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SemanticAssetError(
            f"{label} size mismatch: {actual_bytes} != {expected_bytes}"
        )
    actual_hash = _file_sha256(path)
    if actual_hash != expected_hash:
        raise SemanticAssetError(
            f"{label} SHA-256 mismatch: {actual_hash} != {expected_hash}"
        )
    return path, actual_bytes, actual_hash


class SemanticIndex:
    """Validated memory-mapped float32 matrix with exact deterministic search."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        manifest: Mapping[str, Any],
        asins: tuple[str, ...],
        matrix: Any,
        numpy_module: Any,
    ) -> None:
        self.spec = spec
        self.manifest = manifest
        self.asins = asins
        self.matrix = matrix
        self._np = numpy_module
        self._closed = False

    @classmethod
    def load(
        cls,
        spec: Mapping[str, Any],
        index_dir: Path,
        *,
        expected_catalog_sha256: str,
        manifest_name: str = DEFAULT_INDEX_MANIFEST,
        numpy_module: Any | None = None,
    ) -> "SemanticIndex":
        validate_semantic_spec(spec)
        _require(
            isinstance(expected_catalog_sha256, str)
            and len(expected_catalog_sha256) == 64
            and all(char in "0123456789abcdef" for char in expected_catalog_sha256),
            "expected catalog SHA-256 is invalid",
        )
        root = index_dir.resolve()
        if not root.is_dir():
            raise SemanticAssetError(f"semantic index directory does not exist: {root}")
        manifest_path = _safe_asset_path(root, manifest_name, "semantic-index manifest")
        manifest = _read_json_object(manifest_path)
        _require(manifest.get("schema_version") == INDEX_SCHEMA_VERSION, "wrong index schema")
        _require(
            manifest.get("model_spec_serialization")
            == "UTF-8 canonical JSON; object keys sorted; compact separators; ensure_ascii=false",
            "wrong model-spec serialization declaration",
        )
        expected_spec_hash = canonical_json_sha256(spec)
        _require(
            manifest.get("model_spec_sha256") == expected_spec_hash,
            "semantic index model-spec SHA-256 mismatch",
        )
        _require(
            manifest.get("catalog_sha256") == expected_catalog_sha256,
            "semantic index catalog SHA-256 mismatch",
        )
        shape = list(spec["index"]["shape"])
        rows, dimensions = shape
        _require(manifest.get("rows") == rows, "semantic index row count mismatch")
        _require(
            manifest.get("dimensions") == dimensions,
            "semantic index dimension mismatch",
        )
        matrix_entry = manifest.get("matrix")
        asins_entry = manifest.get("ordered_asins")
        _require(isinstance(matrix_entry, Mapping), "matrix manifest entry missing")
        _require(isinstance(asins_entry, Mapping), "ordered-ASIN manifest entry missing")
        _require(matrix_entry.get("dtype") == "float32", "matrix dtype mismatch")
        _require(matrix_entry.get("shape") == shape, "matrix shape mismatch")
        _require(
            matrix_entry.get("format") == "NumPy .npy",
            "matrix format must be NumPy .npy",
        )
        _require(asins_entry.get("count") == rows, "ordered-ASIN count mismatch")
        _require(asins_entry.get("encoding") == "utf-8-lf", "ordered-ASIN encoding mismatch")
        matrix_path, _, _ = _validate_index_file(root, matrix_entry, "semantic matrix")
        asins_path, _, _ = _validate_index_file(root, asins_entry, "ordered-ASIN")

        try:
            asin_payload = asins_path.read_bytes()
            asin_text = asin_payload.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise SemanticAssetError(f"cannot decode ordered-ASIN file: {exc}") from exc
        if b"\r" in asin_payload or not asin_payload.endswith(b"\n"):
            raise SemanticAssetError("ordered-ASIN file must use LF and end with LF")
        asins = tuple(asin_text[:-1].split("\n"))
        if (
            len(asins) != rows
            or any(not asin or asin.strip() != asin for asin in asins)
            or tuple(sorted(asins)) != asins
            or len(set(asins)) != len(asins)
        ):
            raise SemanticAssetError(
                "ordered-ASIN file must contain the exact unique ascending row order"
            )

        if numpy_module is None:
            modules = load_runtime_modules(spec)
            numpy_module = modules.numpy
        else:
            prepare_runtime_environment(spec)
        if str(getattr(numpy_module, "__version__", "")) != str(spec["runtime"]["numpy"]):
            raise SemanticRuntimeError("NumPy version does not match frozen model spec")
        try:
            matrix = numpy_module.load(
                matrix_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        except Exception as exc:
            raise SemanticAssetError(f"cannot memory-map semantic matrix: {exc}") from exc
        if matrix.dtype != numpy_module.dtype(numpy_module.float32):
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None:
                mmap.close()
            raise SemanticAssetError(
                f"semantic matrix dtype {matrix.dtype} != frozen float32"
            )
        if tuple(matrix.shape) != (rows, dimensions):
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None:
                mmap.close()
            raise SemanticAssetError(
                f"semantic matrix shape {tuple(matrix.shape)} != {(rows, dimensions)}"
            )
        if not bool(matrix.flags.c_contiguous):
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None:
                mmap.close()
            raise SemanticAssetError("semantic matrix must be C-contiguous")
        return cls(spec, manifest, asins, matrix, numpy_module)

    def _ensure_open(self) -> None:
        if self._closed or self.matrix is None:
            raise SemanticRuntimeError("semantic index is closed")

    def search_vector(self, query_vector: Any, top_k: int = 120) -> list[DenseHit]:
        self._ensure_open()
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise SemanticContractError("top_k must be a non-negative integer")
        if top_k == 0:
            return []
        vector = self._np.asarray(query_vector, dtype=self._np.float32)
        dimensions = int(self.spec["encoder"]["embedding_dimensions"])
        if vector.shape != (dimensions,):
            raise SemanticContractError(
                f"query vector shape {tuple(vector.shape)} != {(dimensions,)}"
            )
        if bool(self._np.any(~self._np.isfinite(vector))):
            raise SemanticContractError("query vector contains non-finite values")
        scores = self._np.asarray(self.matrix @ vector, dtype=self._np.float32)
        if scores.shape != (len(self.asins),) or bool(
            self._np.any(~self._np.isfinite(scores))
        ):
            raise SemanticRuntimeError("semantic search produced invalid scores")
        # ASIN rows are already strictly ascending. Stable score sorting therefore gives
        # the frozen secondary parent_asin-ascending tie break without object sorting.
        order = self._np.argsort(-scores, kind="stable")[: min(top_k, len(self.asins))]
        return [
            DenseHit(self.asins[int(index)], float(self._np.float32(scores[int(index)])))
            for index in order
        ]

    def search_query(
        self, query: str, encoder: OfflineSemanticEncoder, top_k: int = 120
    ) -> list[DenseHit]:
        if not isinstance(query, str):
            raise SemanticContractError("semantic query must be a string")
        if not query.strip():
            return []
        return self.search_vector(encoder.encode_query(query), top_k=top_k)

    def close(self) -> None:
        mmap = getattr(self.matrix, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self.matrix = None
        self._closed = True

    def __enter__(self) -> "SemanticIndex":
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
