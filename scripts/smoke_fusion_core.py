"""Target-free ten-turn smoke test for :class:`FusionCoreAgent`.

The runner reads only the supplied catalog and generates a fixed, target-independent
conversation schedule.  It never imports an evaluator or accepts a dataset path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = "fusion-core-target-free-smoke.v1"
TURN_COUNT = 10
TOP_K = 10
ALLOWED_ASK_ATTRIBUTES = {
    None,
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
}
INITIAL_MESSAGES = (
    "I'm looking for shoes. A key requirement is: lightweight.",
    "I'm looking for dresses, but I'm still exploring.",
    "I'm looking for jewelry. Something casual would be nice.",
    "I'm looking for jackets. A key requirement is: waterproof.",
)
CLARIFICATION_VALUES = {
    "category": "shoes",
    "material": "cotton",
    "color": "blue",
    "size": "medium",
    "style": "casual",
    "brand": "Example",
    "budget": "under $50",
    "feature": "lightweight",
    "use_case": "walking",
    "other": "comfort",
}
OVERRIDE_MESSAGE = (
    "Actually, ignore my earlier preference. What I need is: blue."
)


class SmokeContractError(RuntimeError):
    """Raised for a malformed catalog, response, or runner input."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _rss_bytes() -> int:
    """Return the best dependency-free process working-set observation available."""

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            )
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                return int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError, ValueError):
            pass
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except (ImportError, OSError, ValueError):
        return 0


def load_catalog_ids(path: str | Path) -> tuple[str, ...]:
    catalog_path = Path(path)
    identifiers: list[str] = []
    with catalog_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise SmokeContractError(f"catalog row {line_number} is not an object")
            identifier = row.get("parent_asin")
            if not isinstance(identifier, str) or not identifier:
                raise SmokeContractError(
                    f"catalog row {line_number} has no valid parent_asin"
                )
            identifiers.append(identifier)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise SmokeContractError("catalog must contain unique non-empty identifiers")
    return tuple(identifiers)


def _profile(session_index: int) -> dict[str, object]:
    tags = ("comfort", "durability", "style")
    return {
        "purchase_frequency": "unknown",
        "average_prior_rating": None,
        "rating_style": "unknown",
        "preference_tags": [tags[session_index % len(tags)]],
        "summary": "Synthetic target-free smoke profile.",
    }


def next_visible_message(
    *, session_index: int, next_turn: int, ask_attribute: str | None
) -> tuple[str, bool]:
    """Return one deterministic reply and whether it starts a new intent version."""

    if next_turn == 6 and session_index % 5 == 0:
        return OVERRIDE_MESSAGE, True
    if ask_attribute is None:
        return "I don't have any additional preferences.", False
    selector = hashlib.sha256(
        f"{session_index}:{next_turn}:{ask_attribute}".encode("utf-8")
    ).digest()[0]
    if selector % 2 == 0:
        return (
            f"I don't have an additional preference for {ask_attribute}.",
            False,
        )
    value = CLARIFICATION_VALUES.get(ask_attribute, "comfort")
    return f"For that, what matters is: {value}.", False


def _canonical_response(response: object) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    if not isinstance(response, Mapping):
        return {
            "message": "",
            "ask_attribute": None,
            "recommendations": [],
            "usage": None,
        }, ["response_not_mapping"]

    message = response.get("message")
    if not isinstance(message, str):
        errors.append("message_not_string")
        message = ""
    ask_attribute = response.get("ask_attribute")
    if ask_attribute not in ALLOWED_ASK_ATTRIBUTES:
        errors.append("ask_attribute_invalid")
        ask_attribute = None

    raw_recommendations = response.get("recommendations")
    if not isinstance(raw_recommendations, list):
        errors.append("recommendations_not_list")
        raw_recommendations = []
    elif len(raw_recommendations) > TOP_K:
        errors.append("recommendations_exceed_top_k")
    recommendations: list[dict[str, object]] = []
    for row in raw_recommendations:
        if not isinstance(row, Mapping):
            errors.append("recommendation_not_mapping")
            continue
        identifier = row.get("parent_asin")
        if not isinstance(identifier, str) or not identifier:
            errors.append("recommendation_id_invalid")
            continue
        canonical: dict[str, object] = {"parent_asin": identifier}
        if "score" in row:
            score = row.get("score")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
            ):
                errors.append("recommendation_score_invalid")
            else:
                canonical["score_hex"] = float(score).hex()
        recommendations.append(canonical)

    usage = response.get("usage")
    canonical_usage: dict[str, int] | None = None
    if usage is not None:
        if isinstance(usage, Mapping) and all(
            isinstance(usage.get(key), int)
            and not isinstance(usage.get(key), bool)
            and int(usage[key]) >= 0
            for key in ("prompt_tokens", "completion_tokens")
        ):
            canonical_usage = {
                "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
            }
        else:
            errors.append("usage_invalid")
    return {
        "message": message,
        "ask_attribute": ask_attribute,
        "recommendations": recommendations,
        "usage": canonical_usage,
    }, errors


def _close_agent(agent: object) -> None:
    close = getattr(agent, "close", None)
    if callable(close):
        close()
        return
    catalog = getattr(agent, "catalog", None)
    connection = getattr(catalog, "connection", None)
    if connection is not None:
        connection.close()


def run_replica(
    agent_factory: Callable[[Path], object],
    catalog_path: Path,
    catalog_ids: Sequence[str],
    sessions: int,
) -> dict[str, object]:
    catalog_set = frozenset(catalog_ids)
    trace: list[dict[str, object]] = []
    counters = {
        "response_contract_errors": 0,
        "catalog_membership_errors": 0,
        "page_duplicate_errors": 0,
        "same_version_repeat_errors": 0,
        "insufficient_unseen_fallback_turns": 0,
        "short_page_turns": 0,
        "turn10_question_errors": 0,
    }
    latencies_ms: list[float] = []
    started = time.perf_counter()
    peak_rss = _rss_bytes()
    agent = agent_factory(catalog_path)
    try:
        for session_index in range(sessions):
            session_id = f"fusion-smoke-{session_index:04d}"
            reset = getattr(agent, "reset", None)
            respond = getattr(agent, "respond", None)
            if not callable(reset) or not callable(respond):
                raise SmokeContractError("agent must expose reset and respond")
            reset(session_id, _profile(session_index))
            version = 1
            version_age = 1
            seen_by_version: dict[int, set[str]] = {version: set()}
            user_message = INITIAL_MESSAGES[session_index % len(INITIAL_MESSAGES)]
            for turn in range(1, TURN_COUNT + 1):
                call_started = time.perf_counter_ns()
                response = respond(session_id, user_message, turn, TOP_K)
                latencies_ms.append(
                    (time.perf_counter_ns() - call_started) / 1_000_000.0
                )
                peak_rss = max(peak_rss, _rss_bytes())
                canonical, response_errors = _canonical_response(response)
                counters["response_contract_errors"] += len(response_errors)
                rows = canonical["recommendations"]
                assert isinstance(rows, list)
                identifiers = [str(row["parent_asin"]) for row in rows]
                counters["catalog_membership_errors"] += sum(
                    identifier not in catalog_set for identifier in identifiers
                )
                if len(identifiers) != len(set(identifiers)):
                    counters["page_duplicate_errors"] += 1
                if len(identifiers) < TOP_K:
                    counters["short_page_turns"] += 1

                seen = seen_by_version.setdefault(version, set())
                globally_available_unseen = len(catalog_set - seen)
                fallback = globally_available_unseen < TOP_K
                if fallback:
                    counters["insufficient_unseen_fallback_turns"] += 1
                repeated = set(identifiers) & seen
                if version_age >= 2 and repeated and not fallback:
                    counters["same_version_repeat_errors"] += 1
                seen.update(identifier for identifier in identifiers if identifier in catalog_set)

                ask_attribute = canonical["ask_attribute"]
                if turn == TURN_COUNT and ask_attribute is not None:
                    counters["turn10_question_errors"] += 1
                trace.append(
                    {
                        "session": session_index,
                        "turn": turn,
                        "intent_version": version,
                        "intent_age": version_age,
                        "input": user_message,
                        "response": canonical,
                        "fallback": fallback,
                    }
                )
                if turn < TURN_COUNT:
                    user_message, is_override = next_visible_message(
                        session_index=session_index,
                        next_turn=turn + 1,
                        ask_attribute=(
                            str(ask_attribute) if ask_attribute is not None else None
                        ),
                    )
                    if is_override:
                        version += 1
                        version_age = 1
                        seen_by_version.setdefault(version, set())
                    else:
                        version_age += 1
    finally:
        _close_agent(agent)
    wall_seconds = time.perf_counter() - started
    peak_rss = max(peak_rss, _rss_bytes())
    return {
        "trace": trace,
        "trace_sha256": _sha256(trace),
        "counters": counters,
        "resources": {
            "wall_seconds": round(wall_seconds, 6),
            "respond_latency_ms": {
                "p50": round(_percentile(latencies_ms, 0.50), 6),
                "p95": round(_percentile(latencies_ms, 0.95), 6),
                "maximum": round(max(latencies_ms, default=0.0), 6),
            },
            "rss_peak_bytes": peak_rss,
        },
    }


def run_smoke(
    agent_factory: Callable[[Path], object],
    catalog_path: str | Path,
    sessions: int,
    *,
    replicas: int = 2,
) -> dict[str, object]:
    if sessions not in {20, 100}:
        raise SmokeContractError("sessions must be exactly 20 or 100")
    if replicas not in {1, 2}:
        raise SmokeContractError("replicas must be one or two")
    path = Path(catalog_path)
    catalog_ids = load_catalog_ids(path)
    total_started = time.perf_counter()
    first = run_replica(agent_factory, path, catalog_ids, sessions)
    second = (
        run_replica(agent_factory, path, catalog_ids, sessions)
        if replicas == 2
        else None
    )
    exact_repeat = second is not None and first["trace"] == second["trace"]
    first_counters = dict(first["counters"])
    second_counters = dict(second["counters"]) if second is not None else {}
    aggregate = {
        key: int(first_counters[key]) + int(second_counters.get(key, 0))
        for key in sorted(first_counters)
    }
    hard_error_keys = (
        "response_contract_errors",
        "catalog_membership_errors",
        "page_duplicate_errors",
        "same_version_repeat_errors",
        "turn10_question_errors",
    )
    passed = (
        (replicas == 1 or exact_repeat)
        and all(aggregate[key] == 0 for key in hard_error_keys)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if passed else "FAIL",
        "scope": "target-free synthetic-visible-message smoke; no evaluator or outcomes",
        "sessions_per_replica": sessions,
        "replicas": replicas,
        "turns_per_session": TURN_COUNT,
        "catalog_rows": len(catalog_ids),
        "catalog_id_set_sha256": _sha256(sorted(catalog_ids)),
        "message_schedule_sha256": _sha256(
            {
                "initial": INITIAL_MESSAGES,
                "clarifications": CLARIFICATION_VALUES,
                "override": OVERRIDE_MESSAGE,
            }
        ),
        "exact_repeat": {
            "checked": replicas == 2,
            "passed": exact_repeat if replicas == 2 else None,
            "trace_records": sessions * TURN_COUNT,
            "trace_sha256": first["trace_sha256"],
            "replica_sha256_equal": (
                first["trace_sha256"] == second["trace_sha256"]
                if second is not None
                else None
            ),
        },
        "validation_totals_two_replicas": aggregate,
        "resources": {
            "replica_a": first["resources"],
            "replica_b": second["resources"] if second is not None else None,
            "total_wall_seconds": round(time.perf_counter() - total_started, 6),
        },
        "privacy": {
            "catalog_only": True,
            "public_opened": False,
            "proxy_opened": False,
            "labels_opened": False,
            "outcome_opened": False,
        },
    }


def _agent_factory(catalog_path: Path) -> object:
    from starter.fusion_core import FusionCoreAgent

    return FusionCoreAgent(
        catalog_path,
        mode="active",
        force_v212_parent=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--sessions", type=int, choices=(20, 100), default=20)
    parser.add_argument("--replicas", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        result = run_smoke(
            _agent_factory, args.catalog, args.sessions, replicas=args.replicas
        )
    except (ImportError, OSError, SmokeContractError, ValueError, json.JSONDecodeError) as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "ERROR",
            "error_code": type(error).__name__,
            "privacy": {
                "public_opened": False,
                "proxy_opened": False,
                "labels_opened": False,
                "outcome_opened": False,
            },
        }
    payload = _canonical_bytes(result)
    if args.output:
        Path(args.output).write_bytes(payload)
    sys.stdout.buffer.write(payload)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
