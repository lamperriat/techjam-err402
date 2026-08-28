from __future__ import annotations

"""Minimal offline process host for one frozen P8 Agent role."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLES = {
    "P8.B00.served_agent",
    "P8.C00.r08_coverage",
    "P8.S00.explicit_negative_shadow",
    "P8.R01.explicit_negative_partition",
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
_WINDOWS_RSS_ERROR: str | None = None


class WorkerError(RuntimeError):
    pass


class NetworkGuard:
    def __init__(self) -> None:
        self.attempt_count = 0
        self._lock = threading.Lock()
        self._original_connect: Any = None
        self._original_connect_ex: Any = None
        self._original_create_connection: Any = None

    def _deny(self, *_: Any, **__: Any) -> Any:
        with self._lock:
            self.attempt_count += 1
        raise OSError("P8 worker network access is disabled")

    def install(self) -> None:
        self._original_connect = socket.socket.connect
        self._original_connect_ex = socket.socket.connect_ex
        self._original_create_connection = socket.create_connection
        socket.socket.connect = self._deny  # type: ignore[method-assign]
        socket.socket.connect_ex = self._deny  # type: ignore[method-assign]
        socket.create_connection = self._deny  # type: ignore[assignment]

    def restore(self) -> None:
        if self._original_connect is not None:
            socket.socket.connect = self._original_connect
            socket.socket.connect_ex = self._original_connect_ex
            socket.create_connection = self._original_create_connection


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
    statm = Path("/proc/self/statm")
    if statm.exists():
        try:
            pages = int(statm.read_text(encoding="ascii").split()[1])
            return pages * int(os.sysconf("SC_PAGE_SIZE")), "/proc/self/statm resident pages"
        except (IndexError, OSError, TypeError, ValueError):
            pass
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return (peak if sys.platform == "darwin" else peak * 1024), "resource.getrusage"
    except (ImportError, OSError, ValueError):
        return None, "unavailable"


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
        self._thread = threading.Thread(None, self._run, "p8-rss", (), {}, daemon=True)
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


def _capture(agent: Any, role: str, response_count: int) -> dict[str, Any]:
    reader = getattr(agent, "export_p8_blind_capture", None)
    if callable(reader):
        exported = reader()
    elif role == "P8.B00.served_agent":
        source = inspect.getsource(type(agent))
        exported = {
            "schema_version": "p8.served-agent-reference.v1",
            "role": role,
            "configuration": {
                "retrieval_mode": "coverage",
                "rerank_mode": "off",
                "question_policy": "fast",
                "offline": True,
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
    if not isinstance(exported, dict) or set(exported) != CAPTURE_KEYS:
        raise WorkerError("Agent capture root schema is invalid")
    if exported.get("role") not in ROLES:
        raise WorkerError("Agent capture role is invalid")
    for key in ("configuration", "stats", "hashes", "function_hashes"):
        if not isinstance(exported.get(key), dict):
            raise WorkerError(f"Agent capture {key} must be an object")
    errors = exported.get("integrity_errors")
    if not isinstance(errors, list) or any(not isinstance(item, str) for item in errors):
        raise WorkerError("Agent capture integrity_errors must be strings")
    return json.loads(_canonical_bytes(exported))


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--role", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--rss-ms", type=float, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    expected = {"role", "nonce", "factory", "catalog", "spec", "lock", "rss_ms"}
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
    agent: Any = None
    exceptions: list[str] = []
    latencies: list[int] = []
    response_count = 0
    response_digest = hashlib.sha256()
    try:
        factory = _factory(args.factory)
        if args.role == "P8.B00.served_agent":
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
            if args.factory != "starter.p8_lab:create_p8_agent":
                raise WorkerError("P8 experiment roles require the frozen lab factory")
            agent = factory(
                role=args.role,
                catalog_path=args.catalog,
                spec_path=args.spec,
                lock_path=args.lock,
            )
        _reply({"kind": "ready", "nonce": args.nonce, "role": args.role})
        for line in sys.stdin.buffer:
            request = json.loads(line.decode("utf-8"))
            request_id = request.get("request_id")
            operation = request.get("operation")
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
                    peak = rss.stop()
                    captured = _capture(agent, args.role, response_count)
                    bundle = {
                        "schema_version": "p8.worker-bundle.v1",
                        "role": args.role,
                        "factory": args.factory,
                        **captured,
                        "generic_exception_count": len(exceptions),
                        "generic_exception_classes": exceptions,
                        "network_attempt_count": guard.attempt_count,
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
                            "covers_process_lifetime_peak": os.name == "nt",
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
