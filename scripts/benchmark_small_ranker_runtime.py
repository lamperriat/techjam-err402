"""Benchmark the frozen ranker in Agent without evaluator data or targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.agent import Agent  # noqa: E402


SCHEMA_VERSION = "small-ranker-runtime-benchmark.v1"
ARTIFACT_SHA256 = (
    "f8d0b6c0e402edeb34b1e35119c5295449888bc1be713607e88337fa874d16dc"
)
DEFAULT_ARTIFACT = ROOT / "starter/assets/small_ranker_fold_safe_v1.json"
DEFAULT_CATALOG = ROOT / "data/catalog.jsonl"
FORBIDDEN_RUNTIME_ROOTS = ("numpy", "xgboost", "sklearn", "pandas", "torch")
MESSAGES = (
    "I need a comfortable black dress for an outdoor summer wedding under $100.",
    "It is for a woman.",
    "Medium size, please.",
    "No polyester.",
    "A midi style would be nice.",
    "Keep it dark and understated.",
    "It should work outdoors in warm weather.",
    "Please favor affordable choices.",
    "Show me the strongest matches now.",
    "Give me the final recommendations.",
)


class RuntimeBenchmarkError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 < fraction <= 1.0:
        raise ValueError("invalid percentile input")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _peak_process_rss_bytes() -> int:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
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
            )

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        get_process_memory_info.restype = wintypes.BOOL
        if not get_process_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _run_session(
    agent: Agent, session_id: str, turn_latencies_ms: list[float]
) -> dict[str, Any]:
    agent.reset(session_id, {})
    fallback_count = 0
    activation_count = 0
    output_change_count = 0
    for turn, message in enumerate(MESSAGES, 1):
        started = time.perf_counter()
        response = agent.respond(session_id, message, turn, len(MESSAGES))
        turn_latencies_ms.append((time.perf_counter() - started) * 1000.0)
        if not isinstance(response, dict):
            raise RuntimeBenchmarkError("Agent response is not a mapping")
        diagnostics = agent.debug_rerank_diagnostics(session_id).get(
            "small_ranker", {}
        )
        fallback_count += int(bool(diagnostics.get("fallback")))
        activation_count += int(bool(diagnostics.get("activated")))
        output_change_count += int(bool(diagnostics.get("output_changed")))
    return {
        "turns": len(MESSAGES),
        "fallbacks": fallback_count,
        "activations": activation_count,
        "output_changes": output_change_count,
    }


def run(
    catalog_path: Path,
    artifact_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    catalog_path = catalog_path.resolve()
    artifact_path = artifact_path.resolve()
    output_path = output_path.resolve()
    if artifact_path != DEFAULT_ARTIFACT.resolve():
        raise RuntimeBenchmarkError("benchmark artifact path is not frozen")
    if not catalog_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError("catalog or artifact is unavailable")
    if _sha256(artifact_path) != ARTIFACT_SHA256:
        raise RuntimeBenchmarkError("benchmark artifact hash mismatch")
    if output_path.exists() or ROOT not in output_path.parents:
        raise RuntimeBenchmarkError("benchmark output must be new and local")

    initialization_started = time.perf_counter()
    agent = Agent(
        catalog_path,
        llm_client=None,
        p11_mode="active",
        small_ranker_mode="active",
        small_ranker_artifact_path=artifact_path,
    )
    initialization_seconds = time.perf_counter() - initialization_started
    turn_latencies_ms: list[float] = []
    try:
        initial_status = agent._small_ranker_status()
        if (
            initial_status.get("effective_mode") != "active"
            or initial_status.get("fallback")
        ):
            raise RuntimeBenchmarkError("fold-safe runtime did not initialize")
        single_started = time.perf_counter()
        single = _run_session(agent, "benchmark-single", turn_latencies_ms)
        single_seconds = time.perf_counter() - single_started

        batch_started = time.perf_counter()
        batch_rows = [
            _run_session(agent, f"benchmark-batch-{index}", turn_latencies_ms)
            for index in range(10)
        ]
        batch_seconds = time.perf_counter() - batch_started
        final_status = json.loads(json.dumps(agent._small_ranker_status()))
    finally:
        agent.close()

    forbidden = sorted(
        root
        for root in FORBIDDEN_RUNTIME_ROOTS
        if root in sys.modules
        or any(name.startswith(root + ".") for name in sys.modules)
    )
    fallback_count = int(single["fallbacks"]) + sum(
        int(row["fallbacks"]) for row in batch_rows
    )
    completed_turns = int(single["turns"]) + sum(
        int(row["turns"]) for row in batch_rows
    )
    passed = bool(
        completed_turns == 110
        and fallback_count == 0
        and not forbidden
        and final_status.get("fallback") is False
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "evaluator_imported": False,
            "dataset_opened": False,
            "target_or_label_opened": False,
            "handwritten_sessions": 11,
            "completed_turns": completed_turns,
        },
        "inputs": {
            "catalog_sha256": _sha256(catalog_path),
            "artifact_sha256": _sha256(artifact_path),
            "runtime_sha256": _sha256(ROOT / "starter/small_ranker.py"),
            "agent_sha256": _sha256(ROOT / "starter/agent.py"),
            "runner_sha256": _sha256(Path(__file__).resolve()),
        },
        "runtime": {
            "initial_status": initial_status,
            "final_status": final_status,
            "initialization_seconds": initialization_seconds,
            "single_session_seconds": single_seconds,
            "batch_sessions": len(batch_rows),
            "batch_seconds": batch_seconds,
            "batch_sessions_per_second": len(batch_rows) / batch_seconds,
            "turn_latency_ms": {
                "count": len(turn_latencies_ms),
                "p50": _percentile(turn_latencies_ms, 0.50),
                "p95": _percentile(turn_latencies_ms, 0.95),
                "maximum": max(turn_latencies_ms),
            },
            "peak_rss_bytes": _peak_process_rss_bytes(),
            "fallbacks": fallback_count,
            "activations": int(single["activations"])
            + sum(int(row["activations"]) for row in batch_rows),
            "output_changes": int(single["output_changes"])
            + sum(int(row["output_changes"]) for row in batch_rows),
            "loaded_forbidden_training_libraries": forbidden,
        },
        "decision": {
            "runtime_benchmark_passed": passed,
            "promotion_evidence": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.catalog, args.artifact, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["decision"]["runtime_benchmark_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
