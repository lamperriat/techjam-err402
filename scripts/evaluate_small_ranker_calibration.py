"""Execute the one frozen, untouched small-ranker calibration evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    evaluate,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
)
from scripts.evaluate_small_ranker_smoke import (  # noqa: E402
    FORBIDDEN_RUNTIME_ROOTS,
    _small_invariants,
)
from starter.agent import Agent  # noqa: E402


SCHEMA_VERSION = "small-ranker-untouched-calibration.v1"
DEFAULT_FREEZE = ROOT / "configs/small_ranker_v1.runtime_freeze.json"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/fast_track/small_ranker_v1/oof_batch_v1/untouched_calibration.json"
)


class CalibrationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(record: Mapping[str, Any]) -> Path:
    path = ROOT / str(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or _sha256(path) != str(record["sha256"])
    ):
        raise CalibrationError(f"frozen file identity mismatch: {record.get('path')}")
    return path


def _load_freeze(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "small-ranker-runtime-freeze.v1":
        raise CalibrationError("runtime freeze schema mismatch")
    if value.get("untouched_calibration", {}).get("run_count_allowed") != 1:
        raise CalibrationError("calibration run count is not frozen to one")
    resolved: dict[str, Path] = {}
    for name, record in value["frozen_inputs"].items():
        if isinstance(record, Mapping) and {"path", "bytes", "sha256"} <= set(record):
            resolved[name] = _verify_file(record)
    calibration = value["untouched_calibration"]
    resolved["calibration_dataset"] = _verify_file(calibration["dataset"])
    resolved["baseline_aggregate"] = _verify_file(calibration["baseline_aggregate"])
    shards: list[Path] = []
    for raw_path, raw_bytes, raw_hash in calibration["baseline_blind_shards"]:
        shards.append(
            _verify_file(
                {"path": raw_path, "bytes": raw_bytes, "sha256": raw_hash}
            )
        )
    resolved["freeze"] = path
    for index, shard in enumerate(shards):
        resolved[f"baseline_shard_{index}"] = shard
    artifact = json.loads(resolved["research_artifact"].read_text(encoding="utf-8"))
    if float(artifact["gate"]["threshold"]) != float(value["served_policy"]["gate_threshold"]):
        raise CalibrationError("frozen runtime threshold mismatch")
    return value, resolved


def _load_baseline_turns(shards: Sequence[Path], sample_count: int) -> list[list[list[str]]]:
    turns: list[list[list[str] | None]] = [
        [None for _turn in range(10)] for _sample in range(sample_count)
    ]
    rows = 0
    if not shards or sample_count % len(shards):
        raise CalibrationError("frozen P11 baseline shard count is invalid")
    sessions_per_shard = sample_count // len(shards)
    for shard_index, shard in enumerate(shards):
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                local_ordinal = int(value["ordinal"])
                ordinal = shard_index * sessions_per_shard + local_ordinal
                turn = int(value["turn"])
                ranking = value.get("actions", {}).get("KEEP_P11")
                if (
                    not 1 <= local_ordinal <= sessions_per_shard
                    or not 1 <= ordinal <= sample_count
                    or not 1 <= turn <= 10
                    or not isinstance(ranking, list)
                    or len(ranking) != 10
                    or len(set(str(item) for item in ranking)) != 10
                    or turns[ordinal - 1][turn - 1] is not None
                ):
                    raise CalibrationError("frozen P11 baseline trace is invalid")
                turns[ordinal - 1][turn - 1] = [str(item) for item in ranking]
                rows += 1
    if rows != sample_count * 10 or any(
        ranking is None for session in turns for ranking in session
    ):
        raise CalibrationError("frozen P11 baseline trace is incomplete")
    return [[list(ranking or ()) for ranking in session] for session in turns]


def _evaluation_metrics(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical = (
        0.50 * overall["hit_rate_at_10"]
        + 0.30 * overall["mrr"]
        + 0.20 * efficiency
    )
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical, 6),
    }


def _baseline_sessions(
    samples: Sequence[dict[str, Any]],
    products: Mapping[str, dict[str, Any]],
    turns: Sequence[Sequence[Sequence[str]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for sample, session_turns in zip(samples, turns, strict=True):
        target = str(sample["ground_truth"]["parent_asin"])
        _card, behavior = materialize_hidden_fields(dict(sample), products)  # type: ignore[arg-type]
        override_applied = str(sample["scenario_type"]) != "intent_override"
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn, ranking in enumerate(session_turns, 1):
            if override_applied and target in ranking:
                hit_turn = turn
                best_rank = list(ranking).index(target) + 1
                break
            if turn < 10:
                override = behavior.get("override") or {}
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
        result.append(
            {
                "sample_id": str(sample["sample_id"]),
                "scenario_type": str(sample["scenario_type"]),
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            }
        )
    return result


def _transition_metrics(
    baseline: Sequence[Mapping[str, Any]],
    active: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(baseline) != len(active):
        raise CalibrationError("baseline and active session counts differ")
    miss_to_hit = 0
    hit_to_miss = 0
    scenario: dict[str, Counter[str]] = defaultdict(Counter)
    for before, after in zip(baseline, active, strict=True):
        if before.get("sample_id") != after.get("sample_id"):
            raise CalibrationError("active calibration order differs from frozen baseline")
        before_hit = bool(before.get("hit"))
        after_hit = bool(after.get("hit"))
        rescue = int(not before_hit and after_hit)
        harm = int(before_hit and not after_hit)
        miss_to_hit += rescue
        hit_to_miss += harm
        key = str(before.get("scenario_type"))
        scenario[key]["miss_to_hit"] += rescue
        scenario[key]["hit_to_miss"] += harm
    scenario_records = {
        key: {
            "miss_to_hit": value["miss_to_hit"],
            "hit_to_miss": value["hit_to_miss"],
            "net_hits": value["miss_to_hit"] - value["hit_to_miss"],
        }
        for key, value in sorted(scenario.items())
    }
    return {
        "miss_to_hit": miss_to_hit,
        "hit_to_miss": hit_to_miss,
        "net_hits": miss_to_hit - hit_to_miss,
        "hr_delta": round((miss_to_hit - hit_to_miss) / len(baseline), 6),
        "positive_rescue_scenario_span": sum(
            int(value["net_hits"] > 0) for value in scenario_records.values()
        ),
        "by_scenario": scenario_records,
    }


def run(freeze_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    freeze, paths = _load_freeze(freeze_path)
    dataset = load_jsonl(paths["calibration_dataset"])
    expected_rows = int(freeze["untouched_calibration"]["dataset"]["rows"])
    if len(dataset) != expected_rows or expected_rows != 2_000:
        raise CalibrationError("untouched calibration row count mismatch")
    catalog_path = ROOT / "data/catalog.jsonl"
    catalog_ids, categories, products = catalog_index(catalog_path)
    baseline_turns = _load_baseline_turns(
        [paths[f"baseline_shard_{index}"] for index in range(4)],
        expected_rows,
    )
    baseline_sessions = _baseline_sessions(dataset, products, baseline_turns)
    baseline_metrics = _evaluation_metrics(baseline_sessions)
    frozen_baseline = freeze["untouched_calibration"]["baseline_aggregate"]
    if (
        baseline_metrics["hit_rate_at_10"] != frozen_baseline["keep_p11_hr_at_10"]
        or baseline_metrics["mrr"] != frozen_baseline["keep_p11_mrr"]
        or baseline_metrics["mttc"] != frozen_baseline["keep_p11_mttc"]
    ):
        raise CalibrationError("reconstructed P11 baseline differs from frozen aggregate")

    events: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    invariant_failures: Counter[str] = Counter()
    elapsed_ms: list[float] = []
    started = time.perf_counter()

    def trace_sink(event: dict[str, Any]) -> None:
        layer = str(event.get("layer", ""))
        events[layer] += 1
        if layer == "session" and events[layer] % 100 == 0:
            print(
                json.dumps(
                    {
                        "calibration_sessions_started": events[layer],
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
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

    agent = Agent(
        catalog_path,
        llm_client=None,
        p11_mode="active",
        small_ranker_mode="active",
        small_ranker_artifact_path=paths["research_artifact"],
        trace_sink=trace_sink,
    )
    try:
        initial_status = agent._small_ranker_status()
        if initial_status.get("effective_mode") != "active" or initial_status.get("fallback"):
            raise CalibrationError("frozen runtime did not initialize active")
        active_result = evaluate(agent, dataset, catalog_ids, categories, products)
        final_status = json.loads(json.dumps(agent._small_ranker_status()))
    finally:
        agent.close()
    wall_seconds = time.perf_counter() - started
    active_sessions = active_result.get("sessions")
    if not isinstance(active_sessions, list) or len(active_sessions) != expected_rows:
        raise CalibrationError("active calibration session result is incomplete")
    transitions = _transition_metrics(baseline_sessions, active_sessions)
    loaded_forbidden = sorted(
        root
        for root in FORBIDDEN_RUNTIME_ROOTS
        if root in sys.modules or any(name.startswith(root + ".") for name in sys.modules)
    )
    functional_passed = bool(
        events["session"] == expected_rows
        and events["output"] > 0
        and events["retrieval"] == events["output"]
        and events["runtime_fallback"] == 0
        and events["runtime_output_changed"] == events["runtime_activated"]
        and not any(invariant_failures.values())
        and not loaded_forbidden
    )
    gate = freeze["untouched_calibration"]["promotion_gate"]
    metric_passed = bool(
        transitions["hr_delta"] >= float(gate["minimum_hr_delta"])
        and transitions["net_hits"] >= int(gate["minimum_net_hits"])
        and transitions["hit_to_miss"] <= int(gate["maximum_hit_to_miss"])
        and transitions["positive_rescue_scenario_span"]
        >= int(gate["minimum_positive_rescue_scenarios"])
    )
    active_metrics = {
        key: value for key, value in active_result.items() if key != "sessions"
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "freeze": {
            "path": freeze_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(freeze_path),
            "identities_verified_before_run": True,
            "tuning_after_observation": False,
        },
        "scope": {
            "split": "calibration",
            "sample_count": expected_rows,
            "run_ordinal": 1,
            "complete_split": True,
        },
        "baseline_keep_p11": baseline_metrics,
        "active_small_ranker": active_metrics,
        "transitions": transitions,
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
                "maximum": round(max(elapsed_ms), 6) if elapsed_ms else None,
            },
        },
        "privacy": {
            "runtime_received_target": False,
            "target_join_location": "released evaluator and aggregate transition audit only",
            "session_rows_serialized": False,
        },
        "promotion_gate": {
            "frozen_requirements": gate,
            "functional_passed": functional_passed,
            "metric_passed": metric_passed,
            "passed": bool(functional_passed and metric_passed),
        },
        "decision": {
            "promote_runtime_artifact": bool(functional_passed and metric_passed),
            "allow_post_calibration_tuning": False,
        },
        "timing_seconds": {"wall": round(wall_seconds, 6)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.freeze.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "baseline_keep_p11": result["baseline_keep_p11"],
                "active_small_ranker": result["active_small_ranker"],
                "transitions": result["transitions"],
                "runtime": result["runtime"],
                "promotion_gate": result["promotion_gate"],
                "decision": result["decision"],
                "timing_seconds": result["timing_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["promotion_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
