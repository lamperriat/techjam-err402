"""Run the single authorized train_explore limit=100 runtime smoke.

This is a functional deployment check, not a promotion evaluation.  It runs the
released evaluator interaction loop once, emits aggregate metrics/invariants
only, and cannot open calibration, selection, confirmation, or public data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402


SCHEMA_VERSION = "small-ranker-runtime-smoke.v1.1"
AUTHORIZED_LIMIT = 100
DEFAULT_DATASET = ROOT / "experiments/fast_track/proxy_v1/proxy_train_explore.jsonl"
DEFAULT_ARTIFACT = (
    ROOT
    / "experiments/fast_track/small_ranker_v1/oof_batch_v1/research_runtime_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/fast_track/small_ranker_v1/oof_batch_v1/runtime_smoke_limit100.json"
)
FORBIDDEN_RUNTIME_ROOTS = ("numpy", "xgboost", "sklearn", "pandas", "torch")


class RuntimeSmokeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_rank_percentile(
    values: list[float], percentile: float
) -> float | None:
    """Return a deterministic nearest-rank percentile without NumPy."""

    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _peak_process_rss_bytes() -> int:
    """Read the OS process high-water RSS using only the Python stdlib."""

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
        peak = int(counters.PeakWorkingSetSize)
    else:
        import resource

        raw_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = raw_peak if sys.platform == "darwin" else raw_peak * 1024
    if peak <= 0:
        raise RuntimeSmokeError("OS returned a non-positive peak RSS")
    return peak


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=AUTHORIZED_LIMIT)
    return parser


def _small_invariants(value: Mapping[str, Any]) -> dict[str, bool]:
    baseline = value.get("baseline_top10")
    proposed = value.get("proposed_top10")
    served = value.get("served_top10")
    fallback = bool(value.get("fallback"))
    activated = bool(value.get("activated"))
    if not all(isinstance(item, list) for item in (baseline, proposed, served)):
        return {
            "diagnostic_shape": False,
            "ranks_1_9": False,
            "slot10_only": False,
            "served_mode": False,
        }
    assert isinstance(baseline, list) and isinstance(proposed, list) and isinstance(served, list)
    diagnostic_shape = len(baseline) == len(proposed) == len(served) == 10
    ranks_1_9 = diagnostic_shape and proposed[:9] == baseline[:9]
    changed_positions = [
        index for index, (before, after) in enumerate(zip(baseline, proposed, strict=True)) if before != after
    ]
    slot10_only = (
        not activated and not changed_positions
    ) or (
        activated and changed_positions == [9] and proposed[9] not in baseline[:9]
    )
    served_mode = served == (baseline if fallback else proposed)
    return {
        "diagnostic_shape": diagnostic_shape,
        "ranks_1_9": ranks_1_9,
        "slot10_only": slot10_only,
        "served_mode": served_mode,
    }


def run(
    catalog_path: Path,
    dataset_path: Path,
    artifact_path: Path,
    output_path: Path,
    *,
    limit: int,
) -> dict[str, Any]:
    if limit != AUTHORIZED_LIMIT:
        raise RuntimeSmokeError("this runner is locked to train_explore limit=100")
    for path in (catalog_path, dataset_path, artifact_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if dataset_path.resolve() != DEFAULT_DATASET.resolve():
        raise RuntimeSmokeError("runtime smoke may open only the train_explore proxy")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    samples = load_jsonl(dataset_path)
    if len(samples) != 2_000:
        raise RuntimeSmokeError("train_explore source must contain exactly 2,000 sessions")
    selected = samples[:limit]
    catalog_ids, categories, products = catalog_index(catalog_path)
    events: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    invariant_failures: Counter[str] = Counter()
    elapsed_ms: list[float] = []

    def trace_sink(event: dict[str, Any]) -> None:
        layer = str(event.get("layer", ""))
        events[layer] += 1
        data = event.get("data")
        if layer == "output" and isinstance(data, Mapping):
            value = data.get("elapsed_ms")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                elapsed_ms.append(float(value))
        if layer != "retrieval" or not isinstance(data, Mapping):
            return
        small = data.get("small_ranker")
        if not isinstance(small, Mapping):
            invariant_failures["missing_small_ranker_diagnostics"] += 1
            return
        reasons[str(small.get("reason_code", "missing"))] += 1
        events["runtime_fallback"] += int(bool(small.get("fallback")))
        events["runtime_activated"] += int(bool(small.get("activated")))
        events["runtime_output_changed"] += int(bool(small.get("output_changed")))
        for name, passed in _small_invariants(small).items():
            invariant_failures[name] += int(not passed)

    started = time.perf_counter()
    agent = Agent(
        catalog_path,
        llm_client=None,
        p11_mode="active",
        small_ranker_mode="active",
        small_ranker_artifact_path=artifact_path,
        trace_sink=trace_sink,
    )
    try:
        initial_status = agent._small_ranker_status()
        if initial_status.get("effective_mode") != "active" or initial_status.get("fallback"):
            raise RuntimeSmokeError("small-ranker runtime did not initialize active")
        result = evaluate(agent, selected, catalog_ids, categories, products)
        final_status = json.loads(json.dumps(agent._small_ranker_status()))
    finally:
        agent.close()
    wall_seconds = time.perf_counter() - started
    resource_error: str | None = None
    try:
        peak_rss_bytes = _peak_process_rss_bytes()
    except Exception as error:
        peak_rss_bytes = None
        resource_error = type(error).__name__
    p95_latency_ms = _nearest_rank_percentile(elapsed_ms, 0.95)
    resource_measurement_complete = bool(
        p95_latency_ms is not None
        and peak_rss_bytes is not None
        and peak_rss_bytes > 0
    )
    loaded_forbidden = sorted(
        root
        for root in FORBIDDEN_RUNTIME_ROOTS
        if root in sys.modules or any(name.startswith(root + ".") for name in sys.modules)
    )
    evaluated_turns = int(events["output"])
    functional_passed = bool(
        result.get("sample_count") == limit
        and events["session"] == limit
        and evaluated_turns > 0
        and events["retrieval"] == evaluated_turns
        and events["runtime_fallback"] == 0
        and events["runtime_activated"] > 0
        and events["runtime_output_changed"] == events["runtime_activated"]
        and not any(invariant_failures.values())
        and not loaded_forbidden
        and resource_measurement_complete
    )
    aggregate_metrics = {
        key: value
        for key, value in result.items()
        if key != "sessions"
    }
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    oof_metrics = artifact.get("oof_freeze", {}).get("global_threshold_metrics", {})
    oof_gate_preserved = bool(
        oof_metrics.get("net_hits", 0) >= 10
        and oof_metrics.get("hit_to_miss") == 0
        and artifact.get("parity", {}).get("full_c100_rank_order_exact") is True
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "split": "train_explore",
            "source_rows": len(samples),
            "limit": limit,
            "complete_split": False,
            "purpose": "functional runtime smoke only",
        },
        "inputs": {
            "catalog_sha256": _sha256(catalog_path),
            "dataset_sha256": _sha256(dataset_path),
            "artifact_sha256": _sha256(artifact_path),
            "runtime_source_sha256": _sha256(ROOT / "starter/small_ranker.py"),
            "agent_source_sha256": _sha256(ROOT / "starter/agent.py"),
            "runner_source_sha256": _sha256(Path(__file__)),
        },
        "metrics": aggregate_metrics,
        "runtime": {
            "initial_status": initial_status,
            "final_status": final_status,
            "events": dict(sorted(events.items())),
            "reason_counts": dict(sorted(reasons.items())),
            "invariant_failures": dict(sorted(invariant_failures.items())),
            "loaded_forbidden_training_libraries": loaded_forbidden,
            "turn_latency_ms": {
                "count": len(elapsed_ms),
                "mean": round(sum(elapsed_ms) / len(elapsed_ms), 6) if elapsed_ms else None,
                "p95_nearest_rank": round(p95_latency_ms, 6) if p95_latency_ms is not None else None,
                "maximum": round(max(elapsed_ms), 6) if elapsed_ms else None,
            },
            "process_memory": {
                "peak_rss_bytes": peak_rss_bytes,
                "scope": "OS process-lifetime high-water mark",
                "error_code": resource_error,
            },
            "resource_measurement_complete": resource_measurement_complete,
        },
        "privacy": {
            "runtime_received_target": False,
            "target_used_only_by_released_evaluator": True,
            "session_rows_serialized": False,
        },
        "oof_gate_preserved": oof_gate_preserved,
        "decision": {
            "functional_smoke_passed": functional_passed,
            "resource_measurement_complete": resource_measurement_complete,
            "calibration_authorized": bool(functional_passed and oof_gate_preserved),
        },
        "timing_seconds": {"wall": round(wall_seconds, 6)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.catalog, args.dataset, args.artifact, args.output, limit=args.limit)
    print(
        json.dumps(
            {
                "metrics": result["metrics"],
                "runtime": result["runtime"],
                "decision": result["decision"],
                "timing_seconds": result["timing_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["decision"]["functional_smoke_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
