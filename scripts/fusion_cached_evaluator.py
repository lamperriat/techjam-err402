"""Thin one-shot bridge from one frozen Agent to the cached 2k OOF benchmark.

The command creates an exclusive claim before opening any outcome-bearing input,
runs the official simulator twice over one in-memory cohort, and compares the
identifier-free ledger with either frozen v2.12 or Version A.  It never selects
a public corpus and accepts every data path explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.official_metric_bridge import rebuild_official_metrics
from scripts.smoke_fusion_core import _rss_bytes
from evaluator.local_evaluator import catalog_index, evaluate


class FusionEvaluationError(ValueError):
    pass


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _factory(spec: str) -> Callable[..., object]:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise FusionEvaluationError("agent factory must be module:attribute")
    result = getattr(importlib.import_module(module_name), attribute)
    if not callable(result):
        raise FusionEvaluationError("agent factory is not callable")
    return result


def _summary(a: list[dict[str, object]], b: list[dict[str, object]], folds: np.ndarray) -> dict[str, object]:
    def metrics(rows: list[dict[str, object]], indices: Sequence[int]) -> dict[str, int | float]:
        return rebuild_official_metrics([rows[i] for i in indices])
    all_indices = list(range(len(a)))
    transitions = {"miss_to_hit": sum(not bool(x["hit"]) and bool(y["hit"]) for x, y in zip(a, b, strict=True)),
                   "hit_to_miss": sum(bool(x["hit"]) and not bool(y["hit"]) for x, y in zip(a, b, strict=True))}
    transitions["net"] = transitions["miss_to_hit"] - transitions["hit_to_miss"]
    per_fold = []
    for fold in sorted(int(x) for x in np.unique(folds)):
        indices = np.flatnonzero(folds == fold).tolist()
        per_fold.append({"outer_fold": fold, "a": metrics(a, indices), "b": metrics(b, indices)})
    return {"a": metrics(a, all_indices), "b": metrics(b, all_indices), "transitions": transitions, "per_fold": per_fold}


def _candidate_breakdowns(
    ledger: list[dict[str, object]],
    samples: Sequence[Mapping[str, object]],
    categories: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    def grouped_metrics(groups: Mapping[str, list[int]]) -> dict[str, object]:
        return {
            name: rebuild_official_metrics([ledger[index] for index in indices])
            for name, indices in sorted(groups.items())
            if indices
        }

    scenario: dict[str, list[int]] = {}
    taxonomy: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        scenario.setdefault(str(sample.get("scenario_type", "unknown")), []).append(index)
        ground_truth = sample.get("ground_truth")
        target = ground_truth.get("parent_asin") if isinstance(ground_truth, Mapping) else None
        nodes = " ".join(categories.get(str(target), ())).lower()
        if "jewelry" in nodes:
            group = "jewelry"
        elif "shoe" in nodes or "boot" in nodes:
            group = "shoes"
        elif "clothing" in nodes:
            group = "clothing"
        else:
            group = "accessories-other"
        taxonomy.setdefault(group, []).append(index)
    cumulative = {
        str(turn): round(
            sum(
                row["first_hit_turn"] is not None
                and int(row["first_hit_turn"]) <= turn
                for row in ledger
            )
            / len(ledger),
            6,
        )
        for turn in range(1, 11)
    }
    return {
        "first_turn_hr": cumulative["1"],
        "cumulative_hit_rate_by_turn": cumulative,
        "scenario": grouped_metrics(scenario),
        "taxonomy": grouped_metrics(taxonomy),
    }


def _normalized_sessions(result: Mapping[str, object]) -> list[dict[str, object]]:
    rows = result.get("sessions")
    if not isinstance(rows, list):
        raise FusionEvaluationError("official evaluator omitted sessions")
    return [{key: row.get(key) for key in ("hit", "first_hit_turn", "best_rank", "reciprocal_rank")}
            for row in rows if isinstance(row, Mapping)]


def _evaluate_and_close(
    factory: Callable[..., object],
    flags: Mapping[str, object],
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    agent = factory(**dict(flags))
    try:
        ledger = _normalized_sessions(
            evaluate(agent, samples, catalog_ids, categories, products)
        )
        diagnostics_fn = getattr(agent, "evaluation_diagnostics", None)
        diagnostics = diagnostics_fn() if callable(diagnostics_fn) else {}
        if not isinstance(diagnostics, Mapping):
            raise FusionEvaluationError("agent diagnostics must be a mapping")
        return ledger, dict(diagnostics)
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()


def _official_repeat(factory: Callable[..., object], flags: Mapping[str, object], samples: list[dict],
                     catalog_ids: set[str], categories: dict[str, list[str]], products: dict[str, dict]) -> tuple[list[dict[str, object]], str, dict[str, object]]:
    first, first_diagnostics = _evaluate_and_close(
        factory, flags, samples, catalog_ids, categories, products
    )
    repeat, repeat_diagnostics = _evaluate_and_close(
        factory, flags, samples, catalog_ids, categories, products
    )
    first_hash = _sha({"ledger": first, "diagnostics": first_diagnostics})
    repeat_hash = _sha({"ledger": repeat, "diagnostics": repeat_diagnostics})
    if first_hash != repeat_hash:
        raise FusionEvaluationError("official evaluator exact repeat mismatch")
    return first, _sha(first), first_diagnostics


def _cached_v212_ledger(
    samples: Sequence[Mapping[str, object]],
    source_root: Path,
    projection_root: Path,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Reconstruct the frozen grace2 OOF session ledger without a new cache."""

    from scripts import analyze_small_ranker_remaining_misses as attribution
    from scripts import evaluate_versioned_unseen_pagination as pagination
    from scripts import export_small_ranker_fold_safe_artifact as frozen
    from scripts import train_p12_counterfactual_router as trace_source

    inputs = frozen._load_inputs(source_root.resolve(), projection_root.resolve())
    surface = frozen._action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    activation, _ = attribution._reproduce_nested_activation(
        surface, inputs.labels, seed=40220260830
    )
    if hashlib.sha256(activation.tobytes()).hexdigest() != pagination.EXPECTED_ACTIVATION_SHA256:
        raise FusionEvaluationError("frozen v2.12 activation identity mismatch")
    if hashlib.sha256(surface.chosen.tobytes()).hexdigest() != pagination.EXPECTED_CHOSEN_SHA256:
        raise FusionEvaluationError("frozen v2.12 choice identity mismatch")
    traces = attribution._load_traces(source_root.resolve())
    eligible = np.asarray(inputs.labels["eligible_from"], dtype=np.int16)
    folds = np.asarray(inputs.labels["outer_fold"], dtype=np.int16)
    if len(samples) != len(traces) or eligible.shape != (len(samples),):
        raise FusionEvaluationError("v2.12 cohort dimensions changed")

    ledger: list[dict[str, object]] = []
    for session_index, (sample, turns) in enumerate(zip(samples, traces, strict=True)):
        ground_truth = sample.get("ground_truth")
        if not isinstance(ground_truth, Mapping):
            raise FusionEvaluationError("proxy ground_truth is malformed")
        target = ground_truth.get("parent_asin")
        if not isinstance(target, str) or not target:
            raise FusionEvaluationError("proxy target is malformed")
        reset_turn = trace_source._eligible_turn(sample)
        if int(eligible[session_index]) != reset_turn:
            raise FusionEvaluationError("v2.12 eligible turn mismatch")
        served: set[str] = set()
        version_start_turn = 1
        first_turn: int | None = None
        best_rank: int | None = None
        for turn_index, turn_row in enumerate(turns):
            turn_number = turn_index + 1
            if reset_turn > 1 and turn_number == reset_turn:
                served.clear()
                version_start_turn = turn_number
            order = pagination.reconstruct_current_order(
                turn_row,
                int(surface.chosen[session_index, turn_index]),
                bool(activation[session_index, turn_index]),
            )
            page = pagination.fixed_two_page_grace(
                order,
                served,
                turn_number - version_start_turn + 1,
            )
            served.update(page)
            if turn_number < int(eligible[session_index]):
                continue
            if target in page:
                first_turn = turn_number
                best_rank = page.index(target) + 1
                break
        ledger.append({
            "hit": first_turn is not None,
            "first_hit_turn": first_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    expected = {
        "hit_rate_at_10": 0.991,
        "mrr": 0.695795,
        "mttc": 2.869,
        "recommended_technical_score": 0.866858,
    }
    observed = rebuild_official_metrics(ledger)
    if any(observed[key] != value for key, value in expected.items()):
        raise FusionEvaluationError("frozen v2.12 metric identity mismatch")
    return ledger, folds


def _load_identifier_free_ledger(path: Path) -> list[dict[str, object]]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != "fusion_session_ledger_v1":
        raise FusionEvaluationError("invalid comparator ledger schema")
    rows = payload.get("sessions")
    if not isinstance(rows, list):
        raise FusionEvaluationError("invalid comparator ledger rows")
    normalized = _normalized_sessions({"sessions": rows})
    rebuild_official_metrics(normalized)
    if payload.get("sha256") != _sha(normalized):
        raise FusionEvaluationError("comparator ledger hash mismatch")
    return normalized


def attach_candidate_once(
    *,
    factory: Callable[..., object],
    flags: Mapping[str, object],
    proxy_path: Path,
    labels_path: Path,
    catalog_path: Path,
    claim_path: Path,
    ledger_output: Path,
    comparator_ledger: Path | None = None,
    source_root: Path | None = None,
    projection_root: Path | None = None,
) -> dict[str, object]:
    """Evaluate one frozen candidate twice and compare to one frozen ledger."""

    started = time.perf_counter()
    try:
        descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"state": "TARGET_ATTACH_CLAIMED", "mode": "official_candidate"}, handle, sort_keys=True)
        with proxy_path.open("r", encoding="utf-8") as handle:
            samples = [json.loads(line) for line in handle if line.strip()]
        catalog_ids, categories, products = catalog_index(str(catalog_path))
        candidate, candidate_hash, runtime_diagnostics = _official_repeat(
            factory, flags, samples, catalog_ids, categories, products
        )
        with np.load(labels_path, allow_pickle=False) as archive:
            eligible = np.asarray(archive["eligible_from"], dtype=np.int16).copy()
            folds = np.asarray(archive["outer_fold"], dtype=np.int16).copy()
        if eligible.shape != (len(samples),) or folds.shape != (len(samples),):
            raise FusionEvaluationError("candidate cohort dimensions changed")
        if comparator_ledger is not None:
            comparator = _load_identifier_free_ledger(comparator_ledger)
            comparator_name = "version_a"
        elif source_root is not None and projection_root is not None:
            comparator, cached_folds = _cached_v212_ledger(
                samples, source_root, projection_root
            )
            if not np.array_equal(folds, cached_folds):
                raise FusionEvaluationError("candidate/incumbent folds differ")
            comparator_name = "v2.12"
        else:
            raise FusionEvaluationError("one comparator source is required")
        if len(comparator) != len(candidate):
            raise FusionEvaluationError("candidate/comparator session counts differ")
        ledger_payload = {
            "schema": "fusion_session_ledger_v1",
            "sessions": candidate,
            "sha256": candidate_hash,
        }
        if ledger_output.exists():
            raise FusionEvaluationError("candidate ledger output already exists")
        ledger_output.write_bytes(_canonical(ledger_payload))
        return {
            "status": "VALID",
            "schema": "fusion_official_candidate_v1",
            "comparator_name": comparator_name,
            "candidate_ledger_sha256": candidate_hash,
            "candidate_ledger_path": str(ledger_output),
            "runtime_diagnostics": runtime_diagnostics,
            "candidate_breakdowns": _candidate_breakdowns(
                candidate, samples, categories
            ),
            "resource": {
                "wall_seconds": round(time.perf_counter() - started, 6),
                "rss_peak_bytes": _rss_bytes(),
            },
            **_summary(comparator, candidate, folds),
        }
    except FileExistsError:
        raise FusionEvaluationError("one-shot claim already exists") from None
    except Exception as exc:
        return {
            "status": "INVALID_ONE_SHOT_CONSUMED",
            "schema": "fusion_official_candidate_v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_dataset(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, Mapping):
        payload = payload.get("sessions", payload.get("samples"))
    if not isinstance(payload, list) or not all(isinstance(row, Mapping) for row in payload):
        raise FusionEvaluationError("dataset must contain a JSON array or JSONL objects")
    return [dict(row) for row in payload]


def evaluate_public_single(
    *,
    factory: Callable[..., object],
    flags: Mapping[str, object],
    dataset_path: Path,
    catalog_path: Path,
) -> dict[str, object]:
    """Run one public cohort twice and return no per-session material."""

    started = time.perf_counter()
    peak_rss = _rss_bytes()
    samples = _load_dataset(dataset_path)
    catalog_ids, categories, products = catalog_index(str(catalog_path))
    ledger, ledger_hash, diagnostics = _official_repeat(
        factory, flags, samples, catalog_ids, categories, products
    )
    if len(ledger) != len(samples):
        raise FusionEvaluationError("official evaluator returned incomplete cohort")
    peak_rss = max(peak_rss, _rss_bytes())
    return {
        "status": "VALID",
        "schema": "fusion_public_single_v1",
        "session_count": len(ledger),
        "exact_repeat": True,
        "ledger_sha256": ledger_hash,
        "metrics": rebuild_official_metrics(ledger),
        "breakdowns": _candidate_breakdowns(ledger, samples, categories),
        "runtime_diagnostics": diagnostics,
        "resource": {
            "wall_seconds": round(time.perf_counter() - started, 6),
            "rss_bytes": peak_rss,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    single = sub.add_parser("official-single")
    single.add_argument("--agent-factory", required=True)
    single.add_argument("--flags", default="{}")
    single.add_argument("--comparator-ledger", type=Path)
    single.add_argument("--source-root", type=Path)
    single.add_argument("--projection-root", type=Path)
    for name in ("proxy", "labels", "catalog", "claim", "ledger-output", "output"):
        single.add_argument("--" + name, type=Path, required=True)
    public = sub.add_parser("public-single")
    public.add_argument("--agent-factory", required=True)
    public.add_argument("--flags", default="{}")
    for name in ("dataset", "catalog", "output"):
        public.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "public-single":
        result = evaluate_public_single(
            factory=_factory(args.agent_factory),
            flags=json.loads(args.flags),
            dataset_path=args.dataset,
            catalog_path=args.catalog,
        )
    else:
        result = attach_candidate_once(
            factory=_factory(args.agent_factory),
            flags=json.loads(args.flags),
            proxy_path=args.proxy,
            labels_path=args.labels,
            catalog_path=args.catalog,
            claim_path=args.claim,
            ledger_output=args.ledger_output,
            comparator_ledger=args.comparator_ledger,
            source_root=args.source_root,
            projection_root=args.projection_root,
        )
    args.output.write_bytes(_canonical(result))
    return 0 if result.get("status") != "INVALID_ONE_SHOT_CONSUMED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
