from __future__ import annotations

"""Offline, target-blind resource and route-recall benchmark.

The live Agent boundary in this module is deliberately the same as the competition
boundary: reset receives only an opaque session ID and aggregate profile, while
respond receives only the visible message, turn, and requested top-k.  Route lists
are captured while the Agent computes them.  Ground-truth IDs are joined only after
evaluation has finished and the Agent connection has been closed.
"""

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    evaluate,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent, SessionState  # noqa: E402


SCHEMA_VERSION = "track4-resource-recall-v1"
ROUTES = ("broad", "strict", "fused")
RECALL_CUTOFFS = (10, 20, 50, 80, 120)
OFFICIAL_METRIC_KEYS = (
    "sample_count",
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
    "reported_token_usage",
    "scenario_metrics",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _round_seconds(value: float) -> float:
    return round(value, 6)


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    """Return a deterministic nearest-rank percentile.

    Nearest-rank uses ceil(p * N), clamped to [1, N].  Unlike interpolation it
    always reports an actually observed call latency and behaves sensibly for
    tiny smoke-test samples.
    """

    if not values:
        return None
    ordered = sorted(values)
    index = max(1, min(len(ordered), math.ceil(percentile * len(ordered)))) - 1
    return ordered[index]


def latency_summary(latencies_ns: Iterable[int]) -> dict[str, int | float | None | str]:
    values_ms = [value / 1_000_000.0 for value in latencies_ns]

    def rounded(value: float | None) -> float | None:
        return None if value is None else round(value, 6)

    return {
        "count": len(values_ms),
        "method": "nearest-rank over individual Agent.respond wall times",
        "mean_ms": rounded(statistics.fmean(values_ms) if values_ms else None),
        "p50_ms": rounded(_nearest_rank(values_ms, 0.50)),
        "p95_ms": rounded(_nearest_rank(values_ms, 0.95)),
        "p99_ms": rounded(_nearest_rank(values_ms, 0.99)),
        "max_ms": rounded(max(values_ms) if values_ms else None),
    }


def _windows_rss_bytes() -> int | None:
    """Read this process's current working set without third-party packages."""

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
        process = kernel32.GetCurrentProcess()
        succeeded = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
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
    """Last-resort Unix high-water mark; not a current-RSS measurement."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def current_rss_bytes() -> tuple[int | None, str]:
    value = _windows_rss_bytes()
    if value is not None:
        return value, "Windows GetProcessMemoryInfo WorkingSetSize"
    value = _procfs_rss_bytes()
    if value is not None:
        return value, "/proc/self/statm resident pages"
    value = _resource_peak_bytes()
    if value is not None:
        return value, "resource.getrusage ru_maxrss fallback (process high-water mark)"
    return None, "unavailable"


class PeakRssSampler:
    """Small stdlib-only current-RSS sampler with per-stage high-water marks."""

    def __init__(self, interval_ms: float) -> None:
        if interval_ms <= 0:
            raise ValueError("RSS sampling interval must be positive")
        self.interval_seconds = interval_ms / 1000.0
        self.interval_ms = interval_ms
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.backend = "uninitialized"
        self.latest: int | None = None
        self.global_peak: int | None = None
        self.stage_peak: int | None = None

    def sample(self) -> int | None:
        value, backend = current_rss_bytes()
        with self._lock:
            self.backend = backend
            self.latest = value
            if value is not None:
                self.global_peak = (
                    value if self.global_peak is None else max(self.global_peak, value)
                )
                self.stage_peak = (
                    value if self.stage_peak is None else max(self.stage_peak, value)
                )
        return value

    def start(self) -> int | None:
        baseline = self.sample()
        self._thread = threading.Thread(
            target=self._run,
            name="track4-rss-sampler",
            daemon=True,
        )
        self._thread.start()
        return baseline

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def begin_stage(self) -> int | None:
        current = self.sample()
        with self._lock:
            self.stage_peak = current
        return current

    def end_stage(self) -> tuple[int | None, int | None]:
        current = self.sample()
        with self._lock:
            return self.stage_peak, current

    def stop(self) -> int | None:
        self.sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
        with self._lock:
            return self.global_peak


@dataclass
class TurnCapture:
    turn: int
    rankings: dict[str, tuple[str, ...]]


@dataclass
class SessionCapture:
    turns: list[TurnCapture] = field(default_factory=list)


class RankCaptureAgent(Agent):
    """Capture the route lists already computed inside ``Agent.respond``.

    Calling ``debug_rankings`` after every response would execute all retrieval
    queries a second time and inflate evaluator wall time.  This subclass stores
    a reference to the result of the same ``_rank_candidates`` call that powers
    the response.  It never receives or stores a sample ID or target product.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._capture_local = threading.local()
        super().__init__(*args, **kwargs)

    def _rank_candidates(self, state: SessionState) -> dict[str, list[str]]:
        rankings = super()._rank_candidates(state)
        if getattr(self._capture_local, "active", False):
            self._capture_local.rankings = rankings
        return rankings

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        self._capture_local.active = True
        self._capture_local.rankings = None
        try:
            return super().respond(session_id, user_message, turn, top_k)
        finally:
            self._capture_local.active = False

    def take_last_rankings(self) -> dict[str, list[str]] | None:
        rankings = getattr(self._capture_local, "rankings", None)
        self._capture_local.rankings = None
        return rankings


class TargetBlindBenchmarkProbe:
    """Timing/capture wrapper that exposes only the official Agent arguments."""

    def __init__(self, delegate: RankCaptureAgent) -> None:
        self.delegate = delegate
        self.latencies_ns: list[int] = []
        self.sessions: list[SessionCapture] = []
        self._session_by_opaque_id: dict[str, SessionCapture] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.delegate.reset(session_id, user_profile)
        capture = SessionCapture()
        self.sessions.append(capture)
        self._session_by_opaque_id[session_id] = capture

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        started = time.perf_counter_ns()
        try:
            return self.delegate.respond(session_id, user_message, turn, top_k)
        finally:
            self.latencies_ns.append(time.perf_counter_ns() - started)
            rankings = self.delegate.take_last_rankings()
            if rankings is not None:
                normalized = {
                    route: tuple(str(value) for value in rankings.get(route, []))
                    for route in ROUTES
                }
                self._session_by_opaque_id[session_id].turns.append(
                    TurnCapture(turn=turn, rankings=normalized)
                )


def _best_route_rank(
    turns: list[TurnCapture],
    target: str,
    route: str,
    eligible_first_turn: int,
) -> tuple[int | None, int | None]:
    best_rank: int | None = None
    best_turn: int | None = None
    for capture in turns:
        if capture.turn < eligible_first_turn:
            continue
        try:
            rank = capture.rankings[route].index(target) + 1
        except ValueError:
            continue
        if best_rank is None or rank < best_rank or (
            rank == best_rank and (best_turn is None or capture.turn < best_turn)
        ):
            best_rank = rank
            best_turn = capture.turn
    return best_rank, best_turn


def build_route_audit(
    samples: list[dict[str, Any]],
    evaluator_result: dict[str, Any],
    captures: list[SessionCapture],
    products: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Join targets to target-blind traces after interaction has completed."""

    if len(samples) != len(captures):
        raise RuntimeError(
            f"capture/session mismatch: {len(captures)} captures for {len(samples)} samples"
        )
    result_sessions = {
        str(item["sample_id"]): item for item in evaluator_result.get("sessions", [])
    }
    if len(result_sessions) != len(samples):
        raise RuntimeError(
            f"result/session mismatch: {len(result_sessions)} results for {len(samples)} samples"
        )

    route_best_ranks: dict[str, list[int | None]] = {route: [] for route in ROUTES}
    observed_route_limits: dict[str, int] = {route: 0 for route in ROUTES}
    misses: list[dict[str, Any]] = []

    for sample, capture in zip(samples, captures):
        sample_id = str(sample["sample_id"])
        target = str(sample["ground_truth"]["parent_asin"])
        official = result_sessions[sample_id]
        _, behavior = materialize_hidden_fields(sample, products)
        override = behavior.get("override") or {}
        eligible_first_turn = (
            int(override.get("turn", 3))
            if sample.get("scenario_type") == "intent_override"
            else 1
        )
        eligible_turns = [
            item for item in capture.turns if item.turn >= eligible_first_turn
        ]
        best: dict[str, tuple[int | None, int | None]] = {}
        for route in ROUTES:
            observed_route_limits[route] = max(
                observed_route_limits[route],
                *(len(item.rankings[route]) for item in eligible_turns),
                0,
            )
            best[route] = _best_route_rank(
                capture.turns,
                target,
                route,
                eligible_first_turn,
            )
            route_best_ranks[route].append(best[route][0])

        if not official.get("hit"):
            misses.append({
                "sample_id": sample_id,
                "scenario_type": str(sample.get("scenario_type", "")),
                "posthoc_target_parent_asin": target,
                "eligible_first_turn": eligible_first_turn,
                "observed_eligible_turns": [item.turn for item in eligible_turns],
                "best_route_ranks": {
                    route: best[route][0] for route in ROUTES
                },
                "best_route_turns": {
                    route: best[route][1] for route in ROUTES
                },
                "best_fused_rank": best["fused"][0],
                "best_fused_turn": best["fused"][1],
            })

    denominator = len(samples)
    route_metrics: dict[str, Any] = {}
    for route in ROUTES:
        counts = {
            str(cutoff): sum(
                rank is not None and rank <= cutoff
                for rank in route_best_ranks[route]
            )
            for cutoff in RECALL_CUTOFFS
        }
        route_metrics[route] = {
            "maximum_observed_candidate_count": observed_route_limits[route],
            "hit_count_at_k": counts,
            "recall_at_k": {
                key: round(value / denominator, 6) if denominator else 0.0
                for key, value in counts.items()
            },
        }

    return {
        "denominator": denominator,
        "cutoffs": list(RECALL_CUTOFFS),
        "definition": (
            "For each route and session, use the target's best rank across evaluator-observed "
            "eligible turns. Intent-override turns before the generated override are excluded, "
            "matching official hit eligibility. Rankings are captured target-blind; this join "
            "runs only after evaluation."
        ),
        "routes": route_metrics,
        "official_miss_count": len(misses),
        "public_misses": misses,
    }


def _select_samples(
    samples: list[dict[str, Any]],
    scenarios: tuple[str, ...],
    offset: int,
    limit: int,
) -> list[dict[str, Any]]:
    selected = [
        sample
        for sample in samples
        if not scenarios or str(sample.get("scenario_type")) in scenarios
    ]
    selected = selected[offset:]
    return selected if limit == 0 else selected[:limit]


def _official_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in OFFICIAL_METRIC_KEYS}


def run_once(
    catalog_path: Path,
    dataset_path: Path,
    *,
    run_number: int,
    question_policy: str,
    scenarios: tuple[str, ...] = (),
    sample_offset: int = 0,
    sample_limit: int = 0,
    rss_sample_ms: float = 10.0,
) -> dict[str, Any]:
    """Run one isolated load/build/evaluate/post-hoc measurement in this process."""

    gc.collect()
    sampler = PeakRssSampler(rss_sample_ms)
    baseline_rss = sampler.start()
    run_started = time.perf_counter()
    agent: RankCaptureAgent | None = None

    try:
        sampler.begin_stage()
        input_started = time.perf_counter()
        all_samples = load_jsonl(dataset_path)
        samples = _select_samples(all_samples, scenarios, sample_offset, sample_limit)
        if not samples:
            raise ValueError("sample selection is empty")
        catalog_ids, categories, products = catalog_index(catalog_path)
        input_seconds = time.perf_counter() - input_started
        input_peak, post_input_rss = sampler.end_stage()

        sampler.begin_stage()
        index_started = time.perf_counter()
        # Explicit None is the no-key/no-network default.  Environment credentials
        # are neither read nor copied into the artifact.
        agent = RankCaptureAgent(
            catalog_path,
            llm_client=None,
            question_policy=question_policy,
        )
        index_build_seconds = time.perf_counter() - index_started
        index_peak, post_index_rss = sampler.end_stage()
        llm_disabled = agent.llm_client is None

        probe = TargetBlindBenchmarkProbe(agent)
        sampler.begin_stage()
        evaluator_started = time.perf_counter()
        evaluator_result = evaluate(
            probe,
            samples,
            catalog_ids,
            categories,
            products,
        )
        evaluator_wall_seconds = time.perf_counter() - evaluator_started
        evaluator_peak, post_evaluator_rss = sampler.end_stage()

        # Close all retrieval state before labels are joined to captured routes.
        agent.connection.close()
        agent = None

        sampler.begin_stage()
        posthoc_started = time.perf_counter()
        route_audit = build_route_audit(
            samples,
            evaluator_result,
            probe.sessions,
            products,
        )
        posthoc_seconds = time.perf_counter() - posthoc_started
        posthoc_peak, post_posthoc_rss = sampler.end_stage()
        total_seconds = time.perf_counter() - run_started
        total_peak = sampler.stop()

        usage = evaluator_result.get("reported_token_usage") or {}
        no_key_verified = bool(
            llm_disabled
            and int(usage.get("prompt_tokens", 0)) == 0
            and int(usage.get("completion_tokens", 0)) == 0
        )
        return {
            "run_number": run_number,
            "sample_count": len(samples),
            "scenario_counts": dict(sorted(Counter(
                str(sample.get("scenario_type", "")) for sample in samples
            ).items())),
            "timing_seconds": {
                "input_load": _round_seconds(input_seconds),
                "index_build": _round_seconds(index_build_seconds),
                "evaluator_wall": _round_seconds(evaluator_wall_seconds),
                "posthoc_route_join": _round_seconds(posthoc_seconds),
                "total": _round_seconds(total_seconds),
            },
            "respond_call_count": len(probe.latencies_ns),
            "respond_latency": latency_summary(probe.latencies_ns),
            "memory": {
                "backend": sampler.backend,
                "sampling_interval_ms": rss_sample_ms,
                "baseline_rss_bytes": baseline_rss,
                "input_peak_rss_bytes": input_peak,
                "post_input_rss_bytes": post_input_rss,
                "index_peak_rss_bytes": index_peak,
                "post_index_rss_bytes": post_index_rss,
                "evaluator_peak_rss_bytes": evaluator_peak,
                "post_evaluator_rss_bytes": post_evaluator_rss,
                "posthoc_peak_rss_bytes": posthoc_peak,
                "post_posthoc_rss_bytes": post_posthoc_rss,
                "run_peak_rss_bytes": total_peak,
                "run_peak_delta_from_baseline_bytes": (
                    total_peak - baseline_rss
                    if total_peak is not None and baseline_rss is not None
                    else None
                ),
            },
            "no_key_default": {
                "llm_client_disabled": llm_disabled,
                "network_required": False,
                "credential_values_recorded": False,
                "agent_closed_before_posthoc_label_join": True,
                "verified": no_key_verified,
            },
            "official_metrics": _official_metrics(evaluator_result),
            "official_result_sha256": _stable_sha256(evaluator_result),
            "route_audit_sha256": _stable_sha256(route_audit),
            "route_audit": route_audit,
        }
    finally:
        if agent is not None:
            agent.connection.close()
        sampler.stop()


def build_benchmark(
    catalog_path: Path,
    dataset_path: Path,
    *,
    runs: int = 2,
    question_policy: str = "fast",
    scenarios: tuple[str, ...] = (),
    sample_offset: int = 0,
    sample_limit: int = 0,
    rss_sample_ms: float = 10.0,
    verbose: bool = False,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if sample_offset < 0:
        raise ValueError("sample offset must be non-negative")
    if sample_limit < 0:
        raise ValueError("sample limit must be non-negative (0 means all)")
    if rss_sample_ms <= 0:
        raise ValueError("RSS sampling interval must be positive")

    run_results: list[dict[str, Any]] = []
    for run_number in range(1, runs + 1):
        if verbose:
            print(
                f"[benchmark] run {run_number}/{runs}: load, index, evaluate, audit",
                flush=True,
            )
        result = run_once(
            catalog_path,
            dataset_path,
            run_number=run_number,
            question_policy=question_policy,
            scenarios=scenarios,
            sample_offset=sample_offset,
            sample_limit=sample_limit,
            rss_sample_ms=rss_sample_ms,
        )
        run_results.append(result)
        if verbose:
            metrics = result["official_metrics"]
            print(
                f"[benchmark] run {run_number}: HR={metrics['hit_rate_at_10']:.6f} "
                f"MRR={metrics['mrr']:.6f} MTTC={metrics['mttc']:.6f} "
                f"calls={result['respond_call_count']}",
                flush=True,
            )

    signatures = [
        (
            item["official_result_sha256"],
            item["route_audit_sha256"],
            item["respond_call_count"],
        )
        for item in run_results
    ]
    determinism_checked = len(run_results) >= 2
    deterministic = len(set(signatures)) == 1 if determinism_checked else None

    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "catalog_path": str(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
            "dataset_path": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "agent_source_sha256": _sha256(PROJECT_ROOT / "starter" / "agent.py"),
            "evaluator_source_sha256": _sha256(
                PROJECT_ROOT / "evaluator" / "local_evaluator.py"
            ),
            "benchmark_source_sha256": _sha256(Path(__file__)),
            "runs": runs,
            "question_policy": question_policy,
            "scenario_filter": list(scenarios),
            "sample_offset": sample_offset,
            "sample_limit": sample_limit,
            "rss_sample_ms": rss_sample_ms,
            "network_required": False,
            "llm_client": None,
        },
        "methodology": {
            "latency": (
                "perf_counter_ns around each Agent.respond call; nearest-rank percentiles"
            ),
            "evaluator_wall": (
                "single wall interval around the official evaluator; route lists are captured "
                "from its existing retrieval call, not recomputed"
            ),
            "peak_rss": (
                "current process RSS sampled by a stdlib-only background thread plus stage "
                "boundary samples; process-wide and platform-dependent"
            ),
            "target_blindness": (
                "The timed wrapper sees only reset/respond API arguments. Target IDs are joined "
                "to immutable route captures after evaluation and after SQLite is closed."
            ),
            "determinism": (
                "Exact SHA-256 equality of complete official results and route audits, plus "
                "respond-call-count equality; timing and RSS are intentionally excluded."
            ),
        },
        "determinism": {
            "checked": determinism_checked,
            "run_count": len(run_results),
            "status": (
                "passed" if deterministic else "failed"
            ) if determinism_checked else "not_checked",
            "all_functional_outputs_equal": deterministic,
            "signatures": [
                {
                    "run_number": item["run_number"],
                    "official_result_sha256": item["official_result_sha256"],
                    "route_audit_sha256": item["route_audit_sha256"],
                    "respond_call_count": item["respond_call_count"],
                }
                for item in run_results
            ],
        },
        "all_runs_no_key_default_verified": all(
            item["no_key_default"]["verified"] for item in run_results
        ),
        "runs": run_results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure offline Agent resources, determinism, and target-blind route recall."
        )
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/resource_recall_baseline.json"),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Independent repetitions; use 1 for a quick smoke run (default: 2).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Evaluate at most N selected sessions; 0 means all (default: 0).",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Skip this many sessions after scenario filtering (default: 0).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        choices=("buying", "browsing", "intent_override", "boundary"),
        help="Restrict to a scenario; repeat to select several.",
    )
    parser.add_argument(
        "--question-policy",
        choices=("fast", "boundary", "conservative"),
        default="fast",
    )
    parser.add_argument(
        "--rss-sample-ms",
        type=float,
        default=10.0,
        help="Current-RSS polling interval in milliseconds (default: 10).",
    )
    parser.add_argument(
        "--allow-nondeterminism",
        action="store_true",
        help="Write the artifact but return success if repeated functional outputs differ.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = build_benchmark(
        args.catalog,
        args.dataset,
        runs=args.runs,
        question_policy=args.question_policy,
        scenarios=tuple(dict.fromkeys(args.scenario)),
        sample_offset=args.sample_offset,
        sample_limit=args.sample_limit,
        rss_sample_ms=args.rss_sample_ms,
        verbose=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[benchmark] wrote {args.output}", flush=True)
    determinism = artifact["determinism"]
    if (
        determinism["checked"]
        and determinism["status"] != "passed"
        and not args.allow_nondeterminism
    ):
        print("[benchmark] functional determinism check failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
