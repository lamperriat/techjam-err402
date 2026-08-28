from __future__ import annotations

"""Build the frozen P7 catalog embedding index without reading evaluation labels.

This module deliberately imports only the Python standard library at module load time.
The frozen thread/runtime environment is installed before NumPy, ONNX Runtime,
tokenizers, or :mod:`starter.semantic` can be imported.  The production CLI has no
option to relax that boundary; tests use injected fake encoders and a small catalog.
"""

import argparse
import ctypes
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_SPEC = Path("configs/p7_bge_small_en_v1_5.json")
DEFAULT_CATALOG = Path("data/catalog.jsonl")
DEFAULT_MODEL_DIR = Path("experiments/p7_assets/bge-small-en-v1.5")
DEFAULT_OUTPUT_DIR = Path("experiments/p7_index")
MATRIX_FILENAME = "embeddings.npy"
ORDERED_ASINS_FILENAME = "parent_asins.txt"
MANIFEST_FILENAME = "semantic-index.manifest.json"
SCHEMA_VERSION = "p7.semantic-index.v1"
FROZEN_CATALOG_SHA256 = (
    "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
)
FROZEN_MODEL_SPEC_SHA256 = (
    "e71d0cad480c89eac25ad2b276de9a4e7153e1ec2f3bdcc793682f183a592200"
)
FROZEN_INDEX_SHAPE = (50_000, 384)
_RUNTIME_MODULES = ("numpy", "onnxruntime", "tokenizers")


class DocumentEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> object:
        """Return one normalized embedding row for each supplied document."""

    def close(self) -> None:
        """Release model/runtime resources."""


EncoderFactory = Callable[[dict[str, Any], Path], DocumentEncoder]
DocumentBuilder = Callable[[Mapping[str, Any], Mapping[str, Any]], str]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_model_spec_sha256(spec: Mapping[str, Any]) -> str:
    """Hash parsed JSON, independent of indentation and host line endings."""

    return hashlib.sha256(_canonical_json_bytes(spec)).hexdigest()


def _display_path(path: Path) -> str:
    return path.as_posix()


def _validate_relative_asset_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model required_files paths must be non-empty strings")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"model asset path must stay below model_dir: {value!r}")
    return path


def _validate_spec(spec: dict[str, Any]) -> tuple[int, int, int]:
    if spec.get("schema_version") != "p7.semantic-model-spec.v1":
        raise ValueError("unexpected P7 semantic model spec schema_version")

    try:
        shape = spec["index"]["shape"]
        rows = int(shape[0])
        dimensions = int(shape[1])
        configured_dimensions = int(spec["encoder"]["embedding_dimensions"])
        batch_size = int(spec["encoder"]["batch_size"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("semantic model spec has an invalid index/encoder shape") from exc
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("semantic model spec index.shape must be a two-item list")
    if rows <= 0 or dimensions <= 0 or batch_size <= 0:
        raise ValueError("semantic rows, dimensions, and batch size must be positive")
    if dimensions != configured_dimensions:
        raise ValueError("index dimensions disagree with encoder embedding_dimensions")
    if spec["index"].get("matrix_dtype") != "float32":
        raise ValueError("P7 semantic index matrix_dtype must be float32")
    if spec["index"].get("catalog_order") != "parent_asin_ascending":
        raise ValueError("P7 semantic index must use parent_asin_ascending order")

    environment = spec.get("runtime", {}).get(
        "environment_before_numpy_or_onnxruntime_import"
    )
    if not isinstance(environment, dict) or not environment:
        raise ValueError("semantic model spec must freeze the pre-import environment")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("frozen runtime environment must contain string keys/values")

    required = spec.get("required_files")
    if not isinstance(required, list) or not required:
        raise ValueError("semantic model spec must list required model files")
    paths = [_validate_relative_asset_path(entry.get("path")) for entry in required]
    if len(paths) != len(set(paths)):
        raise ValueError("semantic model spec contains duplicate model asset paths")
    return rows, dimensions, batch_size


def prepare_runtime_environment(
    spec: Mapping[str, Any],
    *,
    enforce_fresh_import: bool,
) -> dict[str, str]:
    """Install frozen environment values before importing threaded runtimes."""

    configured = spec["runtime"]["environment_before_numpy_or_onnxruntime_import"]
    applied = {str(key): str(value) for key, value in configured.items()}
    for key, value in applied.items():
        os.environ[key] = value

    already_imported = sorted(
        module
        for module in _RUNTIME_MODULES
        if module in sys.modules
    )
    if enforce_fresh_import and already_imported:
        raise RuntimeError(
            "threaded semantic runtime was imported before frozen environment: "
            + ", ".join(already_imported)
        )
    return applied


def _safe_child(root: Path, relative: Path) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"asset escapes model directory: {relative.as_posix()}") from exc
    return candidate


def _validate_model_assets(
    spec: Mapping[str, Any],
    model_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    verified: list[dict[str, Any]] = []
    total_bytes = 0
    for entry in spec["required_files"]:
        relative = _validate_relative_asset_path(entry.get("path"))
        path = _safe_child(model_dir, relative)
        if not path.is_file():
            raise FileNotFoundError(f"required model asset is missing: {path}")
        actual_bytes = path.stat().st_size
        actual_sha256 = _file_sha256(path)
        try:
            expected_bytes = int(entry["bytes"])
            expected_sha256 = str(entry["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid frozen asset entry: {relative.as_posix()}") from exc
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"model asset size mismatch for {relative.as_posix()}: "
                f"{actual_bytes} != {expected_bytes}"
            )
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"model asset SHA-256 mismatch for {relative.as_posix()}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        verified.append(
            {
                "path": relative.as_posix(),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "verified": True,
            }
        )
        total_bytes += actual_bytes
    return verified, total_bytes


def _validate_license_asset(
    spec: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    relative = _validate_relative_asset_path(spec["model"].get("license_notice"))
    path = _safe_child(project_root, relative)
    if not path.is_file():
        raise FileNotFoundError(f"bundled model license notice is missing: {path}")
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _load_catalog(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"catalog contains a blank row at line {line_number}")
            try:
                product = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid catalog JSON at line {line_number}") from exc
            if not isinstance(product, dict):
                raise ValueError(f"catalog row {line_number} is not an object")
            parent_asin = product.get("parent_asin")
            if not isinstance(parent_asin, str) or not parent_asin.strip():
                raise ValueError(f"catalog row {line_number} has invalid parent_asin")
            if parent_asin != parent_asin.strip():
                raise ValueError(f"catalog row {line_number} parent_asin has outer whitespace")
            products.append(product)
    if len(products) != expected_rows:
        raise ValueError(
            f"catalog row count {len(products)} != frozen index rows {expected_rows}"
        )
    products.sort(key=lambda product: product["parent_asin"])
    ordered = [product["parent_asin"] for product in products]
    if any(left >= right for left, right in zip(ordered, ordered[1:])):
        raise ValueError("catalog parent_asin values must be unique after strict sorting")
    return products


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _documents_digest_update(
    digest: Any,
    parent_asin: str,
    document: str,
) -> None:
    asin_payload = parent_asin.encode("utf-8")
    document_payload = document.encode("utf-8")
    digest.update(len(asin_payload).to_bytes(8, "big"))
    digest.update(asin_payload)
    digest.update(len(document_payload).to_bytes(8, "big"))
    digest.update(document_payload)


def _windows_rss_bytes() -> int | None:
    if os.name != "nt":
        return None
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        succeeded = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
    except (AttributeError, OSError):
        return None
    return int(counters.WorkingSetSize) if succeeded else None


def _procfs_rss_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    try:
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, TypeError, ValueError):
        return None


def _resource_peak_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def _current_rss_bytes() -> tuple[int | None, str]:
    value = _windows_rss_bytes()
    if value is not None:
        return value, "Windows GetProcessMemoryInfo WorkingSetSize"
    value = _procfs_rss_bytes()
    if value is not None:
        return value, "/proc/self/statm resident pages"
    value = _resource_peak_bytes()
    if value is not None:
        return value, "resource.getrusage ru_maxrss fallback"
    return None, "unavailable"


class _PeakRssSampler:
    def __init__(self, interval_ms: float) -> None:
        if interval_ms <= 0:
            raise ValueError("RSS sampling interval must be positive")
        self.interval_ms = float(interval_ms)
        self.interval_seconds = self.interval_ms / 1000.0
        self.backend = "uninitialized"
        self.peak: int | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def sample(self) -> int | None:
        value, backend = _current_rss_bytes()
        with self._lock:
            self.backend = backend
            if value is not None:
                self.peak = value if self.peak is None else max(self.peak, value)
        return value

    def start(self) -> int | None:
        baseline = self.sample()
        self._thread = threading.Thread(
            target=self._run,
            name="p7-index-rss-sampler",
            daemon=True,
        )
        self._thread.start()
        return baseline

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def stop(self) -> int | None:
        self.sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
        return self.peak


def _runtime_versions(numpy_module: Any) -> dict[str, str | None]:
    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "numpy": str(numpy_module.__version__),
        "onnxruntime": package_version("onnxruntime"),
        "tokenizers": package_version("tokenizers"),
        "platform": platform.platform(),
    }


def _validate_runtime_versions(
    spec: Mapping[str, Any],
    observed: Mapping[str, str | None],
) -> None:
    failures = []
    for name in ("python", "numpy", "onnxruntime", "tokenizers"):
        expected = str(spec["runtime"].get(name, ""))
        if observed.get(name) != expected:
            failures.append(f"{name} {observed.get(name)!r} != frozen {expected!r}")
    if failures:
        raise RuntimeError("semantic runtime version mismatch: " + "; ".join(failures))


def _manifest_payload_with_self_size(manifest: dict[str, Any]) -> bytes:
    """Resolve manifest byte count without pretending a self-hash is possible."""

    resource = manifest["asset_byte_scope"]
    for _ in range(16):
        payload = (
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        size = len(payload)
        total = resource["required_asset_bytes_excluding_manifest"] + size
        if (
            resource.get("manifest_bytes") == size
            and resource.get("required_asset_bytes") == total
        ):
            return payload
        resource["manifest_bytes"] = size
        resource["required_asset_bytes"] = total
    raise RuntimeError("manifest byte-size fixed point did not converge")


def _production_semantic_api() -> tuple[EncoderFactory, DocumentBuilder]:
    semantic = importlib.import_module("starter.semantic")
    encoder_type = semantic.OfflineSemanticEncoder

    def factory(spec: dict[str, Any], model_dir: Path) -> DocumentEncoder:
        return encoder_type.from_frozen_assets(spec, model_dir)

    return factory, semantic.canonical_document


def build_semantic_index(
    spec_path: Path,
    catalog_path: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    expected_catalog_sha256: str = FROZEN_CATALOG_SHA256,
    expected_model_spec_sha256: str = FROZEN_MODEL_SPEC_SHA256,
    expected_shape: tuple[int, int] = FROZEN_INDEX_SHAPE,
    project_root: Path = PROJECT_ROOT,
    encoder_factory: EncoderFactory | None = None,
    document_builder: DocumentBuilder | None = None,
    enforce_fresh_runtime_import: bool = True,
    enforce_runtime_versions: bool = True,
    rss_sample_ms: float = 10.0,
) -> dict[str, Any]:
    """Validate frozen inputs, encode the catalog, and atomically publish an index."""

    spec_path = Path(spec_path)
    catalog_path = Path(catalog_path)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    project_root = Path(project_root)
    if output_dir.exists():
        raise FileExistsError(f"semantic index output already exists: {output_dir}")
    if not spec_path.is_file():
        raise FileNotFoundError(f"semantic model spec is missing: {spec_path}")
    if not catalog_path.is_file():
        raise FileNotFoundError(f"frozen catalog is missing: {catalog_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"local semantic model directory is missing: {model_dir}")

    started = time.perf_counter()
    sampler = _PeakRssSampler(rss_sample_ms)
    baseline_rss = sampler.start()
    temp_dir: Path | None = None
    encoder: DocumentEncoder | None = None
    peak_rss: int | None = None
    try:
        spec_raw_sha256 = _file_sha256(spec_path)
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("semantic model spec is not valid JSON") from exc
        if not isinstance(spec, dict):
            raise ValueError("semantic model spec root must be an object")
        rows, dimensions, batch_size = _validate_spec(spec)
        model_spec_sha256 = canonical_model_spec_sha256(spec)
        if model_spec_sha256 != expected_model_spec_sha256.lower():
            raise ValueError(
                "frozen semantic model spec SHA-256 mismatch: "
                f"{model_spec_sha256} != {expected_model_spec_sha256.lower()}"
            )
        if (rows, dimensions) != expected_shape:
            raise ValueError(
                f"semantic index shape {(rows, dimensions)} != frozen {expected_shape}"
            )
        applied_environment = prepare_runtime_environment(
            spec,
            enforce_fresh_import=enforce_fresh_runtime_import,
        )

        catalog_sha256 = _file_sha256(catalog_path)
        if catalog_sha256 != expected_catalog_sha256.lower():
            raise ValueError(
                "frozen catalog SHA-256 mismatch: "
                f"{catalog_sha256} != {expected_catalog_sha256.lower()}"
            )
        model_files, model_file_bytes = _validate_model_assets(spec, model_dir)
        license_asset = _validate_license_asset(spec, project_root)
        products = _load_catalog(catalog_path, rows)
        ordered_asins = [str(product["parent_asin"]) for product in products]

        if encoder_factory is None or document_builder is None:
            production_factory, production_document_builder = _production_semantic_api()
            encoder_factory = encoder_factory or production_factory
            document_builder = document_builder or production_document_builder

        numpy = importlib.import_module("numpy")
        runtime_observed = _runtime_versions(numpy)
        if enforce_runtime_versions:
            _validate_runtime_versions(spec, runtime_observed)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.tmp-",
                dir=output_dir.parent,
            )
        )
        matrix_path = temp_dir / MATRIX_FILENAME
        asins_path = temp_dir / ORDERED_ASINS_FILENAME
        manifest_path = temp_dir / MANIFEST_FILENAME
        asin_payload = "".join(f"{parent_asin}\n" for parent_asin in ordered_asins).encode(
            "utf-8"
        )
        _write_bytes_exclusive(asins_path, asin_payload)

        encoder = encoder_factory(spec, model_dir)
        matrix = numpy.lib.format.open_memmap(
            matrix_path,
            mode="w+",
            dtype=numpy.float32,
            shape=(rows, dimensions),
        )
        document_digest = hashlib.sha256()
        encoded_rows = 0
        try:
            for start in range(0, rows, batch_size):
                batch_products = products[start : start + batch_size]
                documents: list[str] = []
                for product in batch_products:
                    document = document_builder(product, spec["document"])
                    if not isinstance(document, str) or not document:
                        raise ValueError(
                            "canonical_document returned an empty/non-string product document"
                        )
                    documents.append(document)
                    _documents_digest_update(
                        document_digest,
                        str(product["parent_asin"]),
                        document,
                    )
                embeddings = numpy.asarray(
                    encoder.encode_documents(documents),
                    dtype=numpy.float32,
                )
                expected_shape = (len(documents), dimensions)
                if embeddings.shape != expected_shape:
                    raise ValueError(
                        f"encoder batch shape {embeddings.shape} != {expected_shape}"
                    )
                if not bool(numpy.isfinite(embeddings).all()):
                    raise ValueError("encoder produced a non-finite embedding")
                norms = numpy.linalg.norm(embeddings, axis=1)
                if not bool(numpy.all(numpy.abs(norms - 1.0) <= 1e-4)):
                    raise ValueError("encoder documents are not frozen L2-normalized vectors")
                matrix[start : start + len(documents)] = embeddings
                encoded_rows += len(documents)
            if encoded_rows != rows:
                raise RuntimeError("semantic encoder did not produce every frozen catalog row")
            matrix.flush()
        finally:
            memory_map = getattr(matrix, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
            del matrix
            gc.collect()

        matrix_bytes = matrix_path.stat().st_size
        matrix_sha256 = _file_sha256(matrix_path)
        asin_bytes = asins_path.stat().st_size
        asin_sha256 = _file_sha256(asins_path)
        peak_rss = sampler.stop()
        build_seconds = time.perf_counter() - started

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "model_spec_sha256": model_spec_sha256,
            "model_spec_serialization": (
                "UTF-8 canonical JSON; object keys sorted; compact separators; "
                "ensure_ascii=false"
            ),
            "catalog_sha256": catalog_sha256,
            "rows": rows,
            "dimensions": dimensions,
            "matrix": {
                "path": MATRIX_FILENAME,
                "bytes": matrix_bytes,
                "sha256": matrix_sha256,
                "dtype": "float32",
                "shape": [rows, dimensions],
                "format": "NumPy .npy",
            },
            "ordered_asins": {
                "path": ORDERED_ASINS_FILENAME,
                "bytes": asin_bytes,
                "sha256": asin_sha256,
                "encoding": "utf-8-lf",
                "line_ending": "LF",
                "count": len(ordered_asins),
                "ordering": "strict parent_asin ascending; one unique ID per line",
            },
            "source": {
                "model_spec": {
                    "path": _display_path(spec_path),
                    "raw_bytes": spec_path.stat().st_size,
                    "raw_sha256": spec_raw_sha256,
                    "canonical_sha256": model_spec_sha256,
                    "canonical_serialization": (
                        "parsed JSON; UTF-8; object keys sorted; compact separators; "
                        "ensure_ascii=false"
                    ),
                },
                "catalog": {
                    "path": _display_path(catalog_path),
                    "bytes": catalog_path.stat().st_size,
                    "sha256": catalog_sha256,
                    "expected_frozen_sha256": expected_catalog_sha256.lower(),
                    "frozen_sha256_verified": True,
                    "rows": rows,
                },
                "evaluation_labels_read": False,
                "inputs": ["semantic model spec", "participant catalog", "model assets"],
            },
            "model": {
                "repository": spec["model"]["repository"],
                "revision": spec["model"]["revision"],
                "license": spec["model"]["license"],
                "directory": _display_path(model_dir),
                "required_file_count": len(model_files),
                "required_file_bytes": model_file_bytes,
                "required_files": model_files,
                "all_required_files_verified": True,
                "license_notice": license_asset,
            },
            "preprocessing": {
                "document_schema_version": spec["document"]["schema_version"],
                "document": spec["document"],
                "encoder": spec["encoder"],
                "canonical_documents_sha256": document_digest.hexdigest(),
                "canonical_documents_digest_format": (
                    "for each parent_asin-ascending row: uint64-be ASIN byte length, "
                    "ASIN UTF-8, uint64-be document byte length, document UTF-8"
                ),
            },
            "runtime": {
                "frozen": spec["runtime"],
                "environment_applied_before_threaded_runtime_import": applied_environment,
                "fresh_import_boundary_enforced": enforce_fresh_runtime_import,
                "observed": runtime_observed,
            },
            "build_resources": {
                "wall_seconds": round(build_seconds, 6),
                "wall_definition": (
                    "function entry through completed matrix/ASIN hashing and final RSS "
                    "sample, before manifest serialization and atomic directory publish"
                ),
                "rss_backend": sampler.backend,
                "rss_sampling_interval_ms": sampler.interval_ms,
                "baseline_rss_bytes": baseline_rss,
                "peak_rss_bytes": peak_rss,
                "peak_delta_from_baseline_bytes": (
                    peak_rss - baseline_rss
                    if peak_rss is not None and baseline_rss is not None
                    else None
                ),
            },
            "asset_byte_scope": {
                "definition": (
                    "all frozen model required_files + generated matrix + ordered-ASIN "
                    "file + this manifest + bundled third-party license; excludes catalog, "
                    "Python packages/wheels, and caches"
                ),
                "model_required_files_bytes": model_file_bytes,
                "matrix_bytes": matrix_bytes,
                "ordered_asins_bytes": asin_bytes,
                "license_notice_bytes": license_asset["bytes"],
                "required_asset_bytes_excluding_manifest": (
                    model_file_bytes + matrix_bytes + asin_bytes + license_asset["bytes"]
                ),
                "manifest_path": MANIFEST_FILENAME,
                "manifest_bytes": 0,
                "manifest_sha256": None,
                "manifest_sha256_note": (
                    "intentionally null because a file cannot contain its own SHA-256; "
                    "hash the completed manifest externally"
                ),
                "required_asset_bytes": 0,
            },
            "integrity": {
                "catalog_only_target_blind_build": True,
                "labels_or_session_files_opened": [],
                "network_required": False,
                "output_directory_preexisted": False,
                "publication": "same-filesystem temporary directory then single rename",
            },
        }
        manifest_payload = _manifest_payload_with_self_size(manifest)
        _write_bytes_exclusive(manifest_path, manifest_payload)
        if len(manifest_payload) != manifest["asset_byte_scope"]["manifest_bytes"]:
            raise RuntimeError("written manifest byte count disagrees with manifest")

        if output_dir.exists():
            raise FileExistsError(
                f"semantic index output appeared during build: {output_dir}"
            )
        os.rename(temp_dir, output_dir)
        temp_dir = None
        return manifest
    finally:
        if encoder is not None:
            try:
                encoder.close()
            except Exception:
                # A completed or failed build must still preserve its original outcome.
                pass
        if peak_rss is None:
            sampler.stop()
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen, catalog-only P7 BGE index from local model assets."
        )
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rss-sample-ms",
        type=float,
        default=10.0,
        help="Current-RSS sampling interval during the offline build (default: 10).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_semantic_index(
        args.spec,
        args.catalog,
        args.model_dir,
        args.output_dir,
        rss_sample_ms=args.rss_sample_ms,
        enforce_fresh_runtime_import=True,
    )
    print(
        "[p7-index] "
        f"rows={manifest['rows']} dimensions={manifest['dimensions']} "
        f"matrix_sha256={manifest['matrix']['sha256']}",
        flush=True,
    )
    print(f"[p7-index] wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
