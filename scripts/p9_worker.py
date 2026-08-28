from __future__ import annotations

"""Minimal offline process host for one frozen P9 Agent role."""

import argparse
import ctypes
import hashlib
import inspect
import json
import math
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLES = {
    "P9.B00.served_agent",
    "P9.C00.r08_coverage",
    "P9.S00.compact_negative_shadow",
    "P9.R01.compact_negative_partition",
}
FROZEN_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
}
CAPTURE_KEYS = {
    "schema_version",
    "role",
    "configuration",
    "stats",
    "integrity_errors",
    "hashes",
    "function_hashes",
}
REQUEST_KEYS = {
    "reset": {"request_id", "operation", "ordinal", "user_profile"},
    "respond": {"request_id", "operation", "ordinal", "turn", "user_message", "top_k"},
    "finalize": {"request_id", "operation"},
}
_WINDOWS_RSS_ERROR: str | None = None
MAX_CAPTURE_BYTES = 131_072
HEX64 = re.compile(r"[a-f0-9]{64}")
COUNTER_KEYS = {
    "turns",
    "activations",
    "output_changes",
    "shadow_changes",
    "fallbacks",
    "exact_exception_fallbacks",
    "exception_count",
    "executable_constraint_total",
    "ignored_record_total",
    "sidecar_rows_read",
    "violation_fallback_candidate_total",
}
REASON_KEYS = {
    "control",
    "no_executable_negatives",
    "empty_candidate_pool",
    "partitioned",
    "exception_fallback",
    "instrumentation_exception_fallback",
}
REJECTION_KEYS = {
    "not_active",
    "stale_goal_version",
    "not_negative",
    "not_hard",
    "untrusted_source",
    "not_full_confidence",
    "slot_not_allowed",
    "value_not_single_token",
    "duplicate",
    "compile_exception",
}
FUNCTION_HASH_KEYS = {
    "compile_negative_constraints",
    "classify_masks",
    "stable_compact_partition",
    "CompactEvidenceStore.fetch",
    "P9Agent._rank_candidates",
}
COMPACT_PARAMETERS = {
    "required_status": "active",
    "required_hardness": "hard",
    "required_polarity": -1,
    "required_confidence": 1.0,
    "required_source": "excluded_terms",
    "required_goal_version": "current",
    "allowed_slots": "audience,closure,color,material,style,use_case",
    "value_shape": "single_normalized_ascii_token",
    "candidate_pool": 50,
    "minimum_catalog_evidence_confidence": 0.9,
    "catalog_description_evidence": False,
    "partition_order": "compatible,unknown,explicit_violation",
    "top_k": 10,
    "evidence_schema_version": "p9.compact-negative-evidence.v1",
    "registry_sha256": "6e007f76e29aa97d06de7aa8c65f4cfe4fe505a8ec9c04e131d971bef9892fe6",
    "semantics_sha256": "a527cb016e64e87fe3edfc571a9793700ffabcfe75fc31893e531a584dd54a31",
}
_NETWORK_METHODS = (
    "connect",
    "connect_ex",
    "send",
    "sendall",
    "sendto",
    "sendmsg",
    "bind",
    "listen",
    "accept",
)
_NETWORK_FUNCTIONS = (
    "create_connection",
    "getaddrinfo",
    "gethostbyname",
    "gethostbyname_ex",
    "gethostbyaddr",
)


class WorkerError(RuntimeError):
    pass


class NetworkGuard:
    def __init__(self) -> None:
        self.attempt_count = 0
        self._lock = threading.Lock()
        self._originals: dict[tuple[Any, str], Any] = {}

    def _deny(self, *_: Any, **__: Any) -> Any:
        with self._lock:
            self.attempt_count += 1
        raise OSError("P9 worker network access is disabled")

    def install(self) -> None:
        for owner, names in (
            (socket.socket, _NETWORK_METHODS),
            (socket, _NETWORK_FUNCTIONS),
        ):
            for name in names:
                original = getattr(owner, name, None)
                if original is not None:
                    self._originals[(owner, name)] = original
                    setattr(owner, name, self._deny)

    def restore(self) -> None:
        for (owner, name), original in self._originals.items():
            setattr(owner, name, original)
        self._originals.clear()


class RuntimeBoundary:
    """Fail-closed audit boundary for reads and process creation."""

    _WRITE_EVENTS = {
        "os.remove",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.mkdir",
        "os.link",
        "os.symlink",
        "os.truncate",
    }
    _PROCESS_EVENTS = {
        "os.system",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.spawn",
        "os.exec",
        "os.popen",
        "os.startfile",
        "pty.spawn",
    }

    def __init__(
        self,
        staged_root: Path,
        readable_files: tuple[Path, ...],
        *,
        evidence_path: Path,
    ) -> None:
        self.staged_root = staged_root.resolve()
        self.readable_files = frozenset(item.resolve() for item in readable_files)
        self.evidence_file = evidence_path.resolve()
        roots = {Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()}
        roots.add(Path(sys.executable).resolve().parent)
        self.readable_roots = tuple(sorted(roots, key=lambda item: str(item).lower()))
        self.read_denied_attempt_count = 0
        self.process_denied_attempt_count = 0
        self.network_denied_attempt_count = 0
        self.evidence_open_count = 0
        self._lock = threading.Lock()

    @staticmethod
    def _resolved(value: Any) -> Path | None:
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, bytes):
            value = os.fsdecode(value)
        if not isinstance(value, str) or not value:
            return None
        if value.startswith("file:"):
            parsed = urlparse(value)
            value = unquote(parsed.path)
            if os.name == "nt" and re.match(r"^/[A-Za-z]:", value):
                value = value[1:]
        try:
            return Path(value).resolve()
        except (OSError, RuntimeError, ValueError):
            return None

    def _is_allowed(self, value: Any) -> bool:
        resolved = self._resolved(value)
        if resolved is None:
            return False
        if resolved in self.readable_files:
            return True
        for root in (self.staged_root, *self.readable_roots):
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _deny_read(self) -> None:
        with self._lock:
            self.read_denied_attempt_count += 1
        raise PermissionError("P9 worker read boundary denied access")

    def _deny_process(self) -> None:
        with self._lock:
            self.process_denied_attempt_count += 1
        raise PermissionError("P9 worker process creation is disabled")

    def _deny_network(self) -> None:
        with self._lock:
            self.network_denied_attempt_count += 1
        raise PermissionError("P9 worker network access is disabled")

    @staticmethod
    def _open_is_write(mode: Any, flags: Any = 0) -> bool:
        if isinstance(mode, str):
            if any(character in mode for character in "wax+"):
                return True
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
        return isinstance(flags, int) and bool(flags & write_flags)

    def _audit(self, event: str, args: tuple[Any, ...]) -> None:
        if event.startswith("subprocess.") or any(
            event == prefix or event.startswith(f"{prefix}.")
            for prefix in self._PROCESS_EVENTS
        ):
            self._deny_process()
        if event in self._WRITE_EVENTS:
            self._deny_read()
        if event == "open":
            value = args[0] if args else None
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            if isinstance(value, int):
                return
            if self._open_is_write(mode, flags) or not self._is_allowed(value):
                self._deny_read()
            if self._resolved(value) == self.evidence_file:
                with self._lock:
                    self.evidence_open_count += 1
        elif event in {"os.listdir", "os.scandir"}:
            value = args[0] if args else None
            if isinstance(value, int) or not self._is_allowed(value):
                self._deny_read()
        elif event == "os.chdir":
            self._deny_read()
        elif event == "sqlite3.connect":
            value = args[0] if args else None
            if value != ":memory:" and not self._is_allowed(value):
                self._deny_read()
            if self._resolved(value) == self.evidence_file:
                with self._lock:
                    self.evidence_open_count += 1
        elif event.startswith("socket."):
            self._deny_network()

    def install(self) -> None:
        sys.addaudithook(self._audit)


def _windows_peak_rss_bytes() -> int | None:
    global _WINDOWS_RSS_ERROR
    _WINDOWS_RSS_ERROR = None
    if os.name != "nt":
        return None
    from ctypes import wintypes

    class Counters(ctypes.Structure):
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

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        succeeded = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
    except (AttributeError, OSError) as exc:
        _WINDOWS_RSS_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    if not succeeded:
        _WINDOWS_RSS_ERROR = (
            f"GetProcessMemoryInfo failed with WinError {ctypes.get_last_error()}"
        )
        return None
    value = int(counters.PeakWorkingSetSize)
    if value <= 0:
        _WINDOWS_RSS_ERROR = "GetProcessMemoryInfo returned a non-positive peak"
        return None
    return value


def _rss_bytes() -> tuple[int | None, str]:
    value = _windows_peak_rss_bytes()
    if value is not None:
        return value, "Windows GetProcessMemoryInfo PeakWorkingSetSize"
    if os.name == "nt":
        detail = _WINDOWS_RSS_ERROR or "unknown error"
        return None, f"Windows PeakWorkingSetSize unavailable: {detail}"
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        value = peak if sys.platform == "darwin" else peak * 1024
        return value, "resource.getrusage ru_maxrss"
    except (ImportError, OSError, ValueError):
        return None, "lifetime peak unavailable"


class PeakRssSampler:
    def __init__(self, interval_ms: float) -> None:
        if not 0 < interval_ms <= 10.0:
            raise WorkerError("RSS interval must be in (0, 10] ms")
        self.interval_ms = float(interval_ms)
        self.backend = "uninitialized"
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read(self) -> None:
        value, backend = _rss_bytes()
        self.backend = backend
        if value is not None:
            self.peak = value if self.peak is None else max(self.peak, value)

    def start(self) -> None:
        self._read()
        self._thread = threading.Thread(None, self._run, "p9-rss", (), {}, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_ms / 1000.0):
            self._read()

    def stop(self) -> int | None:
        self._read()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak


def _factory(reference: str) -> Callable[..., Any]:
    if reference.count(":") != 1:
        raise WorkerError("factory reference must be module:function")
    module_name, function_name = reference.split(":", 1)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    module = __import__(module_name, fromlist=[function_name])
    value = getattr(module, function_name, None)
    if not callable(value):
        raise WorkerError("factory reference is not callable")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reply(value: Mapping[str, Any]) -> None:
    sys.stdout.buffer.write(_canonical_bytes(value) + b"\n")
    sys.stdout.buffer.flush()


def _hex64(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise WorkerError(f"Agent capture {name} must be a lowercase SHA-256")
    return value


def _counter(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WorkerError(f"Agent capture {name} must be a non-negative integer")
    return value


def _counter_map(value: Any, allowed: set[str], name: str) -> dict[str, int]:
    if not isinstance(value, dict) or not set(value).issubset(allowed):
        raise WorkerError(f"Agent capture {name} has an invalid strict schema")
    return {key: _counter(value[key], f"{name}.{key}") for key in sorted(value)}


def _expected_experiment_spec(role: str) -> dict[str, Any]:
    if role == "P9.C00.r08_coverage":
        return {
            "variant_id": role,
            "family": "control",
            "mechanism": "r08_coverage",
            "stage_graph": [
                "visible_state",
                "broad120+strict80_rrf",
                "coverage_cascade",
                "top10",
            ],
            "description": (
                "Exact served coverage/off/fast control; the evidence sidecar is not opened."
            ),
            "parameters": [],
        }
    common = [
        "r08_coverage",
        "current_hard_negative_compile",
        "rowid_sidecar_lookup",
        "compact_compatible_then_unknown_then_violation",
    ]
    if role == "P9.S00.compact_negative_shadow":
        family = "diagnostic"
        mechanism = "compact_negative_shadow"
        stage_graph = [*common, "shadow_only"]
        description = "Computes the compact partition while serving the exact control output."
    else:
        family = "constraint_execution"
        mechanism = "compact_negative_partition"
        stage_graph = [*common, "deterministic_violation_fallback", "top10"]
        description = "Serves the compact stable partition with exact R08 exception fallback."
    return {
        "variant_id": role,
        "family": family,
        "mechanism": mechanism,
        "stage_graph": stage_graph,
        "description": description,
        "parameters": [[key, value] for key, value in COMPACT_PARAMETERS.items()],
    }


def _sanitize_experiment_capture(exported: Mapping[str, Any], role: str) -> dict[str, Any]:
    configuration = exported["configuration"]
    if not isinstance(configuration, dict):
        raise WorkerError("Agent capture configuration must be an object")
    required_configuration = {
        "retrieval_mode": "coverage",
        "rerank_mode": "off",
        "question_policy": "fast",
        "target_blind": True,
        "label_free": True,
        "spec_schema_version": "p9.worker-spec.v1",
        "lock_schema_version": "p9.worker-lock.v1",
    }
    evidence_opened = role != "P9.C00.r08_coverage"
    expected_keys = {
        *required_configuration,
        "spec_sha256",
        "lock_sha256",
        "protocol_spec_sha256",
        "evidence_opened",
    }
    if evidence_opened:
        expected_keys.update(
            {"evidence_identity_verified", "evidence_bytes", "evidence_sha256"}
        )
    if set(configuration) != expected_keys or any(
        configuration.get(key) != value for key, value in required_configuration.items()
    ):
        raise WorkerError("Agent capture configuration is not frozen")
    if configuration.get("evidence_opened") is not evidence_opened:
        raise WorkerError("Agent capture evidence mode differs from its role")
    sanitized_configuration = dict(required_configuration)
    sanitized_configuration.update(
        {
            "spec_sha256": _hex64(configuration.get("spec_sha256"), "spec_sha256"),
            "lock_sha256": _hex64(configuration.get("lock_sha256"), "lock_sha256"),
            "protocol_spec_sha256": _hex64(
                configuration.get("protocol_spec_sha256"), "protocol_spec_sha256"
            ),
            "evidence_opened": evidence_opened,
        }
    )
    if evidence_opened:
        evidence_bytes = configuration.get("evidence_bytes")
        if (
            configuration.get("evidence_identity_verified") is not True
            or not isinstance(evidence_bytes, int)
            or isinstance(evidence_bytes, bool)
            or evidence_bytes <= 0
        ):
            raise WorkerError("Agent capture evidence identity is invalid")
        sanitized_configuration.update(
            {
                "evidence_identity_verified": True,
                "evidence_bytes": evidence_bytes,
                "evidence_sha256": _hex64(
                    configuration.get("evidence_sha256"), "evidence_sha256"
                ),
            }
        )

    stats = exported["stats"]
    expected_stats = {
        "schema_version",
        "evidence_schema_version",
        "spec",
        "frozen_parameters",
        *COUNTER_KEYS,
        "partition_totals",
        "reason_counts",
        "rejection_counts",
    }
    if not isinstance(stats, dict) or set(stats) != expected_stats:
        raise WorkerError("Agent capture stats has an invalid strict schema")
    expected_spec = _expected_experiment_spec(role)
    if (
        stats.get("schema_version") != "p9.compact-negative-lab.v1"
        or stats.get("evidence_schema_version") != "p9.compact-negative-evidence.v1"
        or stats.get("spec") != expected_spec
        or stats.get("frozen_parameters") != COMPACT_PARAMETERS
    ):
        raise WorkerError("Agent capture experiment identity is not frozen")
    partition = _counter_map(
        stats.get("partition_totals"),
        {"compatible", "unknown", "explicit_violation"},
        "partition_totals",
    )
    if set(partition) != {"compatible", "unknown", "explicit_violation"}:
        raise WorkerError("Agent capture partition totals are incomplete")
    sanitized_stats = {
        "schema_version": "p9.compact-negative-lab.v1",
        "evidence_schema_version": "p9.compact-negative-evidence.v1",
        "spec_sha256": hashlib.sha256(_canonical_bytes(expected_spec)).hexdigest(),
        "frozen_parameters_sha256": hashlib.sha256(
            _canonical_bytes(COMPACT_PARAMETERS)
        ).hexdigest(),
        **{key: _counter(stats.get(key), key) for key in sorted(COUNTER_KEYS)},
        "partition_totals": partition,
        "reason_counts": _counter_map(stats.get("reason_counts"), REASON_KEYS, "reason_counts"),
        "rejection_counts": _counter_map(
            stats.get("rejection_counts"), REJECTION_KEYS, "rejection_counts"
        ),
    }
    errors = exported["integrity_errors"]
    if (
        not isinstance(errors, list)
        or len(errors) > 32
        or any(not isinstance(item, str) or len(item) > 256 for item in errors)
    ):
        raise WorkerError("Agent capture integrity error list is invalid")
    hashes = exported["hashes"]
    if not isinstance(hashes, dict) or set(hashes) != {"audit_sha256", "responses_sha256"}:
        raise WorkerError("Agent capture hashes have an invalid strict schema")
    function_hashes = exported["function_hashes"]
    if not isinstance(function_hashes, dict) or set(function_hashes) != FUNCTION_HASH_KEYS:
        raise WorkerError("Agent capture function hashes have an invalid strict schema")
    return {
        "schema_version": "p9.compact-negative-lab.v1",
        "role": role,
        "configuration": sanitized_configuration,
        "stats": sanitized_stats,
        "integrity_errors": [
            hashlib.sha256(item.encode("utf-8")).hexdigest() for item in errors
        ],
        "hashes": {key: _hex64(hashes[key], key) for key in sorted(hashes)},
        "function_hashes": {
            key: _hex64(function_hashes[key], key) for key in sorted(function_hashes)
        },
    }


def _capture(agent: Any, role: str, response_count: int) -> dict[str, Any]:
    reader = getattr(agent, "export_p9_blind_capture", None)
    if callable(reader):
        exported = reader()
    elif role == "P9.B00.served_agent":
        source = inspect.getsource(type(agent))
        exported = {
            "schema_version": "p9.served-agent-reference.v1",
            "role": role,
            "configuration": {
                "retrieval_mode": "coverage",
                "rerank_mode": "off",
                "question_policy": "fast",
                "offline": True,
                "evidence_opened": False,
            },
            "stats": {"turns": response_count, "exception_count": 0},
            "integrity_errors": [],
            "hashes": {},
            "function_hashes": {
                "served_agent_class_sha256": hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest()
            },
        }
    else:
        raise WorkerError("Agent capture interface is incomplete")
    try:
        encoded = _canonical_bytes(exported)
    except (TypeError, ValueError) as exc:
        raise WorkerError("Agent capture is not canonical JSON") from exc
    if len(encoded) > MAX_CAPTURE_BYTES:
        raise WorkerError("Agent capture exceeds its byte limit")
    exported = json.loads(encoded)
    if not isinstance(exported, dict) or set(exported) != CAPTURE_KEYS:
        raise WorkerError("Agent capture root schema is invalid")
    if exported.get("role") != role:
        raise WorkerError("Agent capture role differs from the worker role")
    if role != "P9.B00.served_agent":
        return _sanitize_experiment_capture(exported, role)
    if (
        exported.get("schema_version") != "p9.served-agent-reference.v1"
        or exported.get("configuration")
        != {
            "retrieval_mode": "coverage",
            "rerank_mode": "off",
            "question_policy": "fast",
            "offline": True,
            "evidence_opened": False,
        }
        or exported.get("stats") != {"turns": response_count, "exception_count": 0}
        or exported.get("integrity_errors") != []
        or exported.get("hashes") != {}
        or not isinstance(exported.get("function_hashes"), dict)
        or set(exported["function_hashes"]) != {"served_agent_class_sha256"}
    ):
        raise WorkerError("served Agent capture is not frozen")
    return {
        **exported,
        "function_hashes": {
            "served_agent_class_sha256": _hex64(
                exported["function_hashes"]["served_agent_class_sha256"],
                "served_agent_class_sha256",
            )
        },
    }


def _latency_summary(values: list[int]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean_ms": None, "p95_ms": None, "max_ms": None}
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "count": len(ordered),
        "mean_ms": round(sum(ordered) / len(ordered) / 1_000_000.0, 6),
        "p95_ms": round(p95 / 1_000_000.0, 6),
        "max_ms": round(ordered[-1] / 1_000_000.0, 6),
    }


def _validate_request(value: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(value, dict):
        raise WorkerError("worker request must be an object")
    operation = value.get("operation")
    expected = REQUEST_KEYS.get(operation) if isinstance(operation, str) else None
    if expected is None or set(value) != expected:
        raise WorkerError("worker request has an invalid strict schema")
    return value, operation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--role", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--rss-ms", type=float, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    expected = {
        "role", "nonce", "factory", "catalog", "evidence", "spec", "lock", "rss_ms"
    }
    if set(vars(args)) != expected or args.role not in ROLES:
        raise WorkerError("invalid worker bootstrap namespace")
    if re.fullmatch(r"[a-f0-9]{32}", args.nonce) is None:
        raise WorkerError("invalid worker nonce")
    for key, value in FROZEN_ENVIRONMENT.items():
        os.environ[key] = value
    if any(name in sys.modules for name in ("numpy", "tokenizers", "onnxruntime")):
        raise WorkerError("optional runtime imported before bootstrap")
    guard = NetworkGuard()
    guard.install()
    rss = PeakRssSampler(args.rss_ms)
    rss.start()
    boundary = RuntimeBoundary(
        PROJECT_ROOT,
        (args.catalog, args.evidence, args.spec, args.lock),
        evidence_path=args.evidence,
    )
    boundary.install()
    agent: Any = None
    exceptions: list[str] = []
    latencies: list[int] = []
    response_count = 0
    response_digest = hashlib.sha256()
    try:
        factory = _factory(args.factory)
        if args.role == "P9.B00.served_agent":
            if args.factory != "starter.agent:Agent":
                raise WorkerError("served Agent role requires the frozen direct factory")
            agent = factory(
                args.catalog,
                llm_client=None,
                question_policy="fast",
                trace_sink=None,
                rerank_mode="off",
                retrieval_mode="coverage",
            )
        else:
            if args.factory != "starter.p9_lab:create_p9_agent":
                raise WorkerError("P9 experiment roles require the frozen lab factory")
            agent = factory(
                role=args.role,
                catalog_path=args.catalog,
                evidence_path=args.evidence,
                spec_path=args.spec,
                lock_path=args.lock,
            )
        _reply({"kind": "ready", "nonce": args.nonce, "role": args.role})
        for line in sys.stdin.buffer:
            request, operation = _validate_request(json.loads(line.decode("utf-8")))
            request_id = request.get("request_id")
            try:
                if operation == "reset":
                    ordinal = int(request["ordinal"])
                    profile = request["user_profile"]
                    if ordinal <= 0 or not isinstance(profile, dict):
                        raise WorkerError("invalid reset request")
                    agent.reset(ordinal, profile)
                    reply_value: Any = {"ok": True}
                elif operation == "respond":
                    ordinal = int(request["ordinal"])
                    turn = int(request["turn"])
                    message = request["user_message"]
                    top_k = int(request["top_k"])
                    if (
                        ordinal <= 0
                        or not isinstance(message, str)
                        or not 1 <= turn <= 10
                        or top_k != 10
                    ):
                        raise WorkerError("invalid respond request")
                    started = time.perf_counter_ns()
                    response = agent.respond(ordinal, message, turn, top_k)
                    latencies.append(time.perf_counter_ns() - started)
                    response_count += 1
                    response_digest.update(
                        _canonical_bytes(
                            {"ordinal": ordinal, "turn": turn, "response": response}
                        )
                        + b"\n"
                    )
                    reply_value = {"response": response}
                elif operation == "finalize":
                    captured = _capture(agent, args.role, response_count)
                    closer = getattr(agent, "close", None)
                    if callable(closer):
                        closer()
                    agent = None
                    peak = rss.stop()
                    bundle = {
                        "schema_version": "p9.worker-bundle.v1",
                        "role": args.role,
                        "factory": args.factory,
                        "capture": captured,
                        "generic_exception_count": len(exceptions),
                        "generic_exception_classes": exceptions,
                        "network_attempt_count": guard.attempt_count,
                        "read_denied_attempt_count": boundary.read_denied_attempt_count,
                        "process_denied_attempt_count": boundary.process_denied_attempt_count,
                        "audit_network_denied_attempt_count": (
                            boundary.network_denied_attempt_count
                        ),
                        "evidence_open_count": boundary.evidence_open_count,
                        "response_count": response_count,
                        "response_sha256": response_digest.hexdigest(),
                        "timing": {"respond_latency": _latency_summary(latencies)},
                        "memory": {
                            "backend": rss.backend,
                            "sampling_interval_ms": rss.interval_ms,
                            "peak_rss_bytes": peak,
                            "available": peak is not None and rss.backend != "unavailable",
                            "windows_metric": (
                                "PeakWorkingSetSize" if os.name == "nt" else None
                            ),
                            "covers_process_lifetime_peak": (
                                os.name == "nt" or rss.backend == "resource.getrusage ru_maxrss"
                            ),
                        },
                    }
                    _reply({"kind": "result", "request_id": request_id, "bundle": bundle})
                    return 0
                else:
                    raise WorkerError("unknown operation")
                _reply({"kind": "reply", "request_id": request_id, "value": reply_value})
            except Exception as exc:
                exceptions.append(type(exc).__name__)
                _reply(
                    {
                        "kind": "error",
                        "request_id": request_id,
                        "error_class": type(exc).__name__,
                    }
                )
        raise WorkerError("input closed before finalize")
    finally:
        rss.stop()
        if agent is not None:
            closer = getattr(agent, "close", None)
            if callable(closer):
                closer()
            else:
                connection = getattr(agent, "connection", None)
                if connection is not None:
                    connection.close()
        guard.restore()


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
