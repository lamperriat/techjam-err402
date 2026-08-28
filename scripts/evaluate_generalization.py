from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.gate_provenance import (
    assert_gate_snapshot_stable,
    capture_gate_snapshot,
    validate_clean_frozen_snapshot,
)
from starter.agent import RETRIEVAL_MODES, Agent, resolve_retrieval_mode
from starter.architecture_lab import SPEC_BY_ID, ArchitectureAgent
from starter.attributes import SCHEMA_VERSION as ATTRIBUTE_SCHEMA_VERSION
from starter.clarification import SCHEMA_VERSION as QUESTION_VALUE_SCHEMA_VERSION
from starter.frozen_winner import (
    FROZEN_WINNER_ID,
    SELECTION_COMMIT,
    SELECTION_CORPUS_SHA256,
    SELECTION_RESULT_SHA256,
    validate_frozen_winner_configuration,
)
from starter.reranker import SCORER_VERSION
from starter.response_contract import (
    SCHEMA_VERSION as RESPONSE_CONTRACT_SCHEMA_VERSION,
    ContractRecorder,
)
from starter.slot_ledger import SCHEMA_VERSION as SLOT_LEDGER_SCHEMA_VERSION


SCHEMA_VERSION = "p4.generalization.v3"
RERANK_MODES = ("off", "shadow", "active")
METRIC_KEYS = (
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
)
SCENARIO_WEIGHTS = (
    ("buying", 0.40),
    ("browsing", 0.40),
    ("intent_override", 0.15),
    ("boundary", 0.05),
)


def _replacement(pattern: str, replacement: str) -> Callable[[str], tuple[str, bool]]:
    compiled = re.compile(pattern, re.IGNORECASE)

    def apply(message: str) -> tuple[str, bool]:
        transformed, count = compiled.subn(replacement, message)
        return transformed, count > 0

    setattr(apply, "pattern_source", pattern)
    setattr(apply, "replacement_source", replacement)
    return apply


RULES: dict[str, Callable[[str], tuple[str, bool]]] = {
    "opener_dev": _replacement(
        r"\bI(?:'m| am) looking for\s+",
        "I need ",
    ),
    "requirement_dev": _replacement(
        r"\bA key requirement is:\s*",
        "My main requirement is ",
    ),
    "clarification_dev": _replacement(
        r"\bFor that, what matters is:\s*",
        "The main thing that matters is ",
    ),
    "boundary_dev": _replacement(
        r"\bI don't have a preference for ([a-z_]+); please use your judgment\.",
        r"No strong preference on \1; choose what fits best.",
    ),
    "additional_dev": _replacement(
        r"\bI don't have an additional preference for ([a-z_]+)\.",
        r"\1 is flexible for me.",
    ),
    "override_dev": _replacement(
        r"\bActually,\s*ignore my earlier preference\.\s*What I need is:\s*(.+)",
        r"Please disregard what I said before. I would rather have: \1",
    ),
    "feedback_dev": _replacement(
        r"\bThose options are not quite right yet\. Ask me about one specific attribute\.",
        "Those do not fit yet. Please ask one focused question.",
    ),
    "opener_challenge": _replacement(
        r"\bI(?:'m| am) looking for\s+",
        "I am shopping for ",
    ),
    "requirement_challenge": _replacement(
        r"\bA key requirement is:\s*",
        "The most important detail is ",
    ),
    "clarification_challenge": _replacement(
        r"\bFor that, what matters is:\s*",
        "The deciding factor for me is ",
    ),
    "boundary_challenge": _replacement(
        r"\bI don't have a preference for ([a-z_]+); please use your judgment\.",
        r"Any \1 works for me; you can decide.",
    ),
    "additional_challenge": _replacement(
        r"\bI don't have an additional preference for ([a-z_]+)\.",
        r"\1 is not important to me.",
    ),
    "override_challenge": _replacement(
        r"\bActually,\s*ignore my earlier preference\.\s*What I need is:\s*(.+)",
        r"Forget my previous choice. Please prioritize: \1",
    ),
    "feedback_challenge": _replacement(
        r"\bThose options are not quite right yet\. Ask me about one specific attribute\.",
        "These are not a match yet. Could you narrow it down with one question?",
    ),
    "opener_audit": _replacement(
        r"\bI(?:'m| am) looking for\s+",
        "I'd like ",
    ),
    "requirement_audit": _replacement(
        r"\bA key requirement is:\s*",
        "I absolutely must have ",
    ),
    "clarification_audit": _replacement(
        r"\bFor that, what matters is:\s*",
        "I prefer ",
    ),
    "boundary_audit": _replacement(
        r"\bI don't have a preference for ([a-z_]+); please use your judgment\.",
        r"I don't care about \1; you decide.",
    ),
    "additional_audit": _replacement(
        r"\bI don't have an additional preference for ([a-z_]+)\.",
        r"Either \1 is fine.",
    ),
    "override_audit": _replacement(
        r"\bActually,\s*ignore my earlier preference\.\s*What I need is:\s*(.+)",
        r"Changed my mind. I prefer: \1",
    ),
    "feedback_audit": _replacement(
        r"\bThose options are not quite right yet\. Ask me about one specific attribute\.",
        "None of those fit. Ask another focused question.",
    ),
}

DEV_RULES = (
    "opener_dev",
    "requirement_dev",
    "clarification_dev",
    "boundary_dev",
    "additional_dev",
    "override_dev",
    "feedback_dev",
)
CHALLENGE_RULES = (
    "opener_challenge",
    "requirement_challenge",
    "clarification_challenge",
    "boundary_challenge",
    "additional_challenge",
    "override_challenge",
    "feedback_challenge",
)
AUDIT_RULES = (
    "opener_audit",
    "requirement_audit",
    "clarification_audit",
    "boundary_audit",
    "additional_audit",
    "override_audit",
    "feedback_audit",
)
SUITES: dict[str, tuple[str, ...]] = {
    "canonical": (),
    "opener_dev": ("opener_dev",),
    "requirements_dev": ("requirement_dev", "clarification_dev"),
    "no_preference_dev": ("boundary_dev", "additional_dev"),
    "override_dev": ("override_dev",),
    "feedback_dev": ("feedback_dev",),
    "combined_dev": DEV_RULES,
    "combined_challenge": CHALLENGE_RULES,
    "combined_audit": AUDIT_RULES,
}


def _suite_registry_sha256() -> str:
    payload = {
        "rules": {
            name: {
                "pattern": getattr(rule, "pattern_source"),
                "replacement": getattr(rule, "replacement_source"),
            }
            for name, rule in sorted(RULES.items())
        },
        "suites": {name: list(rules) for name, rules in sorted(SUITES.items())},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def perturb_message(message: str, rule_names: tuple[str, ...]) -> tuple[str, list[str]]:
    transformed = message
    applied: list[str] = []
    for name in rule_names:
        transformed, changed = RULES[name](transformed)
        if changed:
            applied.append(name)
    return transformed, applied


class PerturbedAgent:
    """Target-blind adapter that changes only the visible user message."""

    def __init__(self, delegate: Any, rule_names: tuple[str, ...]) -> None:
        self.delegate = delegate
        self.rule_names = rule_names
        self.total_messages = 0
        self.transformed_messages = 0
        self.rule_counts: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.delegate.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        self.total_messages += 1
        transformed, applied = perturb_message(user_message, self.rule_names)
        if applied:
            self.transformed_messages += 1
            self.rule_counts.update(applied)
            if len(self.examples) < 8:
                self.examples.append({
                    "turn": turn,
                    "rules": applied,
                    "before": user_message,
                    "after": transformed,
                })
        return self.delegate.respond(session_id, transformed, turn, top_k)

    def stats(self) -> dict[str, Any]:
        return {
            "total_messages": self.total_messages,
            "transformed_messages": self.transformed_messages,
            "rule_counts": dict(sorted(self.rule_counts.items())),
            "examples": self.examples,
            "target_blind_boundary": (
                "reset receives only the allowed aggregate user_profile and an opaque session_id; "
                "respond receives only that opaque ID, visible user_message, turn, and top_k. The "
                "adapter never receives sample_id, scenario, target ID, intent card, or prior result."
            ),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _samples_sha256(samples: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _scenario_counts(total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError("derived sample count must be positive")
    raw = [(name, total * weight) for name, weight in SCENARIO_WEIGHTS]
    counts = {name: int(value) for name, value in raw}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda item: (-(item[1] - int(item[1])), item[0]))
    for name, _ in order[:remainder]:
        counts[name] += 1
    return counts


def build_product_disjoint_samples(
    public_samples: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    count: int,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    public_targets = {
        str(sample.get("ground_truth", {}).get("parent_asin", ""))
        for sample in public_samples
    }
    eligible = [
        parent_asin
        for parent_asin, product in products.items()
        if parent_asin not in public_targets
        and str(product.get("title") or "").strip()
        and product.get("categories")
    ]
    eligible.sort(
        key=lambda parent_asin: hashlib.sha256(
            f"{seed}\0{parent_asin}".encode("utf-8")
        ).hexdigest()
    )
    if count > len(eligible):
        raise ValueError(f"requested {count} derived samples but only {len(eligible)} are eligible")

    scenario_counts = _scenario_counts(count)
    scenarios = [
        scenario
        for scenario, _ in SCENARIO_WEIGHTS
        for _ in range(scenario_counts[scenario])
    ]
    random.Random(f"{seed}\0scenarios").shuffle(scenarios)
    profile = {
        "purchase_frequency": "not provided",
        "average_prior_rating": None,
        "rating_style": "not provided",
        "preference_tags": [],
        "summary": "Neutral profile for a derived product-disjoint stress session.",
    }
    samples = [
        {
            "category_bucket": "derived",
            "difficulty_bucket": "unlabeled",
            "ground_truth": {"parent_asin": parent_asin},
            "sample_id": f"derived_p1_{index:04d}",
            "scenario_type": scenarios[index - 1],
            "user_profile": dict(profile),
        }
        for index, parent_asin in enumerate(eligible[:count], start=1)
    ]
    derived_targets = {
        str(sample["ground_truth"]["parent_asin"]) for sample in samples
    }
    metadata = {
        "seed": seed,
        "sample_count": len(samples),
        "scenario_counts": dict(sorted(Counter(scenarios).items())),
        "samples_sha256": _samples_sha256(samples),
        "unique_target_count": len(derived_targets),
        "public_target_overlap": len(public_targets & derived_targets),
        "selection": (
            "SHA-256(seed + NUL + parent_asin), excluding all released public targets; "
            "requires non-empty title and categories"
        ),
        "boundary": (
            "This is a catalog-derived local stress set, not organizer private data and not "
            "an estimate of the hidden leaderboard distribution."
        ),
    }
    return samples, metadata


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (*METRIC_KEYS, "sample_count", "scenario_metrics", "reported_token_usage")
    }


def _delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        left = current.get(key)
        right = baseline.get(key)
        result[key] = (
            round(float(left) - float(right), 6)
            if isinstance(left, (int, float)) and isinstance(right, (int, float))
            else None
        )
    return result


def _session_changes(
    current: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    base_by_id = {
        str(item["sample_id"]): item for item in baseline.get("sessions", [])
    }
    score_regressions: list[str] = []
    score_improvements: list[str] = []
    hit_to_miss: list[str] = []
    miss_to_hit: list[str] = []
    earlier_hit: list[str] = []
    later_hit: list[str] = []
    rank_improvements: list[str] = []
    rank_regressions: list[str] = []
    unchanged_score = 0
    score_delta_sum = 0.0

    def components(item: dict[str, Any]) -> tuple[float, int, int, int]:
        hit = int(bool(item.get("hit")))
        turn = item.get("first_hit_turn")
        scored_turn = int(turn) if isinstance(turn, int) else 11
        rank = item.get("best_rank")
        scored_rank = int(rank) if isinstance(rank, int) else 11
        reciprocal_rank = float(item.get("reciprocal_rank") or 0.0)
        efficiency = max(0.0, min(1.0, (11 - scored_turn) / 10.0))
        contribution = 0.50 * hit + 0.30 * reciprocal_rank + 0.20 * efficiency
        return contribution, hit, scored_turn, scored_rank

    for item in current.get("sessions", []):
        sample_id = str(item["sample_id"])
        base = base_by_id.get(sample_id)
        if base is None:
            continue
        current_score, current_hit, current_turn, current_rank = components(item)
        base_score, base_hit, base_turn, base_rank = components(base)
        delta = current_score - base_score
        score_delta_sum += delta
        if delta < -1e-12:
            score_regressions.append(sample_id)
        elif delta > 1e-12:
            score_improvements.append(sample_id)
        else:
            unchanged_score += 1
        if base_hit and not current_hit:
            hit_to_miss.append(sample_id)
        elif current_hit and not base_hit:
            miss_to_hit.append(sample_id)
        if base_hit and current_hit:
            if current_turn < base_turn:
                earlier_hit.append(sample_id)
            elif current_turn > base_turn:
                later_hit.append(sample_id)
            if current_rank < base_rank:
                rank_improvements.append(sample_id)
            elif current_rank > base_rank:
                rank_regressions.append(sample_id)
    return {
        "comparison_basis": (
            "Per-session official contribution: 0.50*hit + 0.30*reciprocal_rank "
            "+ 0.20*efficiency; component changes are reported separately."
        ),
        "official_score_regression_count": len(score_regressions),
        "official_score_improvement_count": len(score_improvements),
        "official_score_unchanged_count": unchanged_score,
        "official_score_delta_sum": round(score_delta_sum, 9),
        "official_score_regression_sample_ids": score_regressions,
        "official_score_improvement_sample_ids": score_improvements,
        "hit_to_miss_count": len(hit_to_miss),
        "miss_to_hit_count": len(miss_to_hit),
        "hit_to_miss_sample_ids": hit_to_miss,
        "miss_to_hit_sample_ids": miss_to_hit,
        "earlier_hit_count": len(earlier_hit),
        "later_hit_count": len(later_hit),
        "rank_improvement_count": len(rank_improvements),
        "rank_regression_count": len(rank_regressions),
    }


def _robust_summary(runs: dict[str, dict[str, Any]], sample_count: int) -> dict[str, Any]:
    hit_sets = []
    for run in runs.values():
        hit_sets.append({
            str(item["sample_id"])
            for item in run["result"].get("sessions", [])
            if item.get("hit")
        })
    robust_hits = set.intersection(*hit_sets) if hit_sets else set()
    return {
        "all_suites_robust_hit_count": len(robust_hits),
        "all_suites_robust_hit_rate": (
            round(len(robust_hits) / sample_count, 6) if sample_count else 0.0
        ),
        "worst_suite_hit_rate_at_10": min(
            run["metrics"]["hit_rate_at_10"] for run in runs.values()
        ),
        "worst_suite_mrr": min(run["metrics"]["mrr"] for run in runs.values()),
        "worst_suite_mttc": max(run["metrics"]["mttc"] for run in runs.values()),
        "worst_suite_technical_score": min(
            run["metrics"]["recommended_technical_score"] for run in runs.values()
        ),
    }


def evaluate_suites(
    catalog_path: Path,
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
    suite_names: list[str],
    question_policy: str,
    corpus_name: str,
    rerank_mode: str = "off",
    architecture_variant: str | None = None,
    retrieval_mode: str | None = None,
) -> dict[str, Any]:
    validate_frozen_winner_configuration(
        architecture_variant,
        question_policy=question_policy,
        rerank_mode=rerank_mode,
    )
    if architecture_variant and retrieval_mode is not None:
        raise ValueError("frozen architecture gates do not accept retrieval_mode")
    resolved_retrieval_mode = (
        None
        if architecture_variant
        else resolve_retrieval_mode(retrieval_mode, rerank_mode)
    )
    runs: dict[str, dict[str, Any]] = {}
    canonical_result: dict[str, Any] | None = None
    ordered_names = list(dict.fromkeys(["canonical", *suite_names]))
    for suite_name in ordered_names:
        print(f"[generalization] {corpus_name}/{suite_name}: building index", flush=True)
        build_started = time.perf_counter()
        delegate = (
            ArchitectureAgent(
                catalog_path,
                architecture_variant,
                question_policy=question_policy,
            )
            if architecture_variant
            else Agent(
                catalog_path,
                question_policy=question_policy,
                rerank_mode=rerank_mode,
                retrieval_mode=resolved_retrieval_mode,
            )
        )
        build_seconds = round(time.perf_counter() - build_started, 3)
        contract = ContractRecorder(delegate, catalog_ids)
        adapter = PerturbedAgent(contract, SUITES[suite_name])
        try:
            evaluation_started = time.perf_counter()
            result = evaluate(adapter, samples, catalog_ids, categories, products)
            evaluation_seconds = round(time.perf_counter() - evaluation_started, 3)
        finally:
            delegate.connection.close()
        if contract.errors:
            raise RuntimeError(
                f"suite {suite_name} violated the strict response contract: "
                + "; ".join(contract.errors[:5])
            )
        if suite_name == "canonical":
            canonical_result = result
        assert canonical_result is not None
        transform_stats = adapter.stats()
        missing_rules = [
            rule_name
            for rule_name in SUITES[suite_name]
            if not transform_stats["rule_counts"].get(rule_name)
        ]
        transform_stats["selected_rule_count"] = len(SUITES[suite_name])
        transform_stats["covered_rule_count"] = (
            len(SUITES[suite_name]) - len(missing_rules)
        )
        transform_stats["missing_rules"] = missing_rules
        transform_stats["coverage_valid"] = not missing_rules
        if corpus_name == "released_public" and missing_rules:
            raise RuntimeError(
                f"suite {suite_name} did not transform released-public messages for: "
                f"{', '.join(missing_rules)}"
            )
        runs[suite_name] = {
            "rules": list(SUITES[suite_name]),
            "transform": transform_stats,
            "timing": {
                "index_build_seconds": build_seconds,
                "evaluation_seconds": evaluation_seconds,
                "respond_latency": {
                    "count": len(contract.latencies_ms),
                    "mean_ms": round(
                        sum(contract.latencies_ms) / len(contract.latencies_ms), 6
                    ) if contract.latencies_ms else None,
                    "max_ms": round(max(contract.latencies_ms), 6)
                    if contract.latencies_ms else None,
                },
            },
            "contract_errors": list(contract.errors),
            "architecture_stats": (
                delegate.experiment_stats() if architecture_variant else None
            ),
            "metrics": _metrics(result),
            "delta_vs_canonical": _delta(result, canonical_result),
            "session_changes_vs_canonical": _session_changes(result, canonical_result),
            "result": result,
        }
        print(
            f"[generalization] {corpus_name}/{suite_name}: "
            f"score={result['recommended_technical_score']:.6f} "
            f"HR={result['hit_rate_at_10']:.6f} MRR={result['mrr']:.6f} "
            f"MTTC={result['mttc']:.6f}",
            flush=True,
        )
    return {
        "sample_count": len(samples),
        "robustness": _robust_summary(runs, len(samples)),
        "suites": runs,
    }


def _suite_names(values: list[str]) -> list[str]:
    if not values:
        return ["combined_dev", "combined_challenge"]
    allowed = set(SUITES) | {"default", "all"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown suites: {', '.join(unknown)}")
    expanded: list[str] = []
    for value in values:
        if value == "default":
            expanded.extend(("combined_dev", "combined_challenge"))
        elif value == "all":
            expanded.extend(name for name in SUITES if name != "canonical")
        elif value != "canonical":
            expanded.append(value)
    return list(dict.fromkeys(expanded))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run target-blind phrase and product-disjoint generalization stress tests."
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/p1_generalization.json")
    )
    parser.add_argument(
        "--corpus", choices=("public", "derived", "both"), default="public"
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Suite name, 'default' (combined dev/challenge), or 'all'. Repeat as needed.",
    )
    parser.add_argument("--derived-count", type=int, default=200)
    parser.add_argument("--seed", default="track4-p1-product-disjoint-v1")
    parser.add_argument(
        "--question-policy",
        choices=("fast", "boundary", "conservative"),
        default="fast",
    )
    parser.add_argument(
        "--rerank-mode",
        choices=RERANK_MODES,
        default="off",
        help=(
            "Reranker execution mode: off preserves P1, shadow measures without changing "
            "recommendations, active serves reranked output (default: off)."
        ),
    )
    parser.add_argument(
        "--architecture-variant",
        choices=(FROZEN_WINNER_ID,),
        help=(
            "Run one frozen target-blind Architecture Lab variant. This is a gate for a "
            "preselected winner, not a public-set architecture search."
        ),
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=RETRIEVAL_MODES,
        help=(
            "control serves weighted RRF; coverage serves promoted R08. Default: coverage "
            "when rerank is off, otherwise control."
        ),
    )
    parser.add_argument("--write-derived-dataset", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    validate_frozen_winner_configuration(
        args.architecture_variant,
        question_policy=args.question_policy,
        rerank_mode=args.rerank_mode,
    )
    if args.architecture_variant and args.retrieval_mode is not None:
        raise ValueError("frozen architecture gates do not accept --retrieval-mode")
    retrieval_mode = (
        None
        if args.architecture_variant
        else resolve_retrieval_mode(args.retrieval_mode, args.rerank_mode)
    )

    suite_names = _suite_names(args.suite)
    expected_winner_suites = [name for name in SUITES if name != "canonical"]
    if args.architecture_variant:
        if args.corpus != "public":
            raise ValueError("the frozen-winner generalization gate requires corpus=public")
        if suite_names != expected_winner_suites:
            raise ValueError(
                "the frozen-winner generalization gate requires --suite all"
            )
    public_samples = load_jsonl(args.dataset)
    if args.architecture_variant and len(public_samples) != 200:
        raise ValueError(
            "the frozen-winner released-public gate requires exactly 200 sessions"
        )
    catalog_ids, categories, products = catalog_index(args.catalog)
    provenance_sources = {
        "runner": Path(__file__).resolve(),
        "gate_provenance": PROJECT_ROOT / "scripts" / "gate_provenance.py",
        "agent": PROJECT_ROOT / "starter" / "agent.py",
        "architecture_lab": PROJECT_ROOT / "starter" / "architecture_lab.py",
        "attributes": PROJECT_ROOT / "starter" / "attributes.py",
        "clarification": PROJECT_ROOT / "starter" / "clarification.py",
        "coverage": PROJECT_ROOT / "starter" / "coverage.py",
        "frozen_winner": PROJECT_ROOT / "starter" / "frozen_winner.py",
        "reranker": PROJECT_ROOT / "starter" / "reranker.py",
        "response_contract": PROJECT_ROOT / "starter" / "response_contract.py",
        "slot_ledger": PROJECT_ROOT / "starter" / "slot_ledger.py",
        "evaluator": PROJECT_ROOT / "evaluator" / "local_evaluator.py",
        "tracked_selection_summary": PROJECT_ROOT / "docs" / "p4_architecture_results.json",
    }
    provenance_inputs = {
        "catalog": args.catalog.resolve(),
        "released_public": args.dataset.resolve(),
    }
    preflight = capture_gate_snapshot(
        PROJECT_ROOT,
        source_paths=provenance_sources,
        input_paths=provenance_inputs,
        selection_commit=SELECTION_COMMIT,
        frozen_architecture_path=PROJECT_ROOT / "starter" / "architecture_lab.py",
        local_selection_artifact=(
            PROJECT_ROOT / "experiments" / "p4_architecture_search.json"
        ),
        expected_selection_artifact_sha256=SELECTION_RESULT_SHA256,
    )
    if args.architecture_variant:
        validate_clean_frozen_snapshot(preflight)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "catalog_path": str(args.catalog),
            "catalog_sha256": _sha256(args.catalog),
            "dataset_path": str(args.dataset),
            "dataset_sha256": _sha256(args.dataset),
            "question_policy": args.question_policy,
            "rerank_mode": args.rerank_mode,
            "retrieval_mode": (
                f"architecture:{args.architecture_variant}"
                if args.architecture_variant
                else retrieval_mode
            ),
            "architecture_variant": args.architecture_variant,
            "architecture_spec": (
                SPEC_BY_ID[args.architecture_variant].as_dict()
                if args.architecture_variant
                else None
            ),
            "suite_names": ["canonical", *suite_names],
            "suite_registry_sha256": _suite_registry_sha256(),
            "runner_source_sha256": _sha256(Path(__file__)),
            "architecture_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "architecture_lab.py"
            ),
            "frozen_winner_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "frozen_winner.py"
            ),
            "evaluation_role": (
                "post_selection_frozen_winner_gate"
                if args.architecture_variant
                else "generalization_evaluation"
            ),
            "selection_performed": False,
            "public_used_for_selection": False,
            "candidate_count": 1 if args.architecture_variant else 0,
            "selection_commit": SELECTION_COMMIT,
            "selection_corpus_sha256": SELECTION_CORPUS_SHA256,
            "selection_result_sha256": SELECTION_RESULT_SHA256,
            "agent_source_sha256": _sha256(PROJECT_ROOT / "starter" / "agent.py"),
            "attribute_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "attributes.py"
            ),
            "reranker_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "reranker.py"
            ),
            "slot_ledger_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "slot_ledger.py"
            ),
            "clarification_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "clarification.py"
            ),
            "coverage_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "coverage.py"
            ),
            "response_contract_source_sha256": _sha256(
                PROJECT_ROOT / "starter" / "response_contract.py"
            ),
            "evaluator_source_sha256": _sha256(
                PROJECT_ROOT / "evaluator" / "local_evaluator.py"
            ),
            "attribute_schema_version": ATTRIBUTE_SCHEMA_VERSION,
            "reranker_scorer_version": SCORER_VERSION,
            "slot_ledger_schema_version": SLOT_LEDGER_SCHEMA_VERSION,
            "question_value_schema_version": QUESTION_VALUE_SCHEMA_VERSION,
            "response_contract_schema_version": RESPONSE_CONTRACT_SCHEMA_VERSION,
            "network_required": False,
            "target_blind_transform": True,
        },
        "corpora": {},
        "provenance": {"preflight": preflight},
    }
    if args.corpus in {"public", "both"}:
        artifact["corpora"]["released_public"] = evaluate_suites(
            args.catalog,
            public_samples,
            catalog_ids,
            categories,
            products,
            suite_names,
            args.question_policy,
            "released_public",
            args.rerank_mode,
            args.architecture_variant,
            retrieval_mode,
        )
    if args.corpus in {"derived", "both"}:
        derived_samples, metadata = build_product_disjoint_samples(
            public_samples, products, args.derived_count, args.seed
        )
        if args.write_derived_dataset is not None:
            args.write_derived_dataset.parent.mkdir(parents=True, exist_ok=True)
            args.write_derived_dataset.write_text(
                "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in derived_samples),
                encoding="utf-8",
            )
        artifact["corpora"]["derived_product_disjoint"] = {
            **metadata,
            **evaluate_suites(
                args.catalog,
                derived_samples,
                catalog_ids,
                categories,
                products,
                suite_names,
                args.question_policy,
                "derived_product_disjoint",
                args.rerank_mode,
                args.architecture_variant,
                retrieval_mode,
            ),
        }

    if args.architecture_variant:
        public = artifact["corpora"]["released_public"]
        runs = public["suites"]
        noncanonical = [runs[name] for name in expected_winner_suites]
        checks = {
            "all_registered_phrase_suites_present": (
                list(runs) == ["canonical", *expected_winner_suites]
            ),
            "all_runs_complete": all(
                run["metrics"]["sample_count"] == 200 for run in runs.values()
            ),
            "strict_response_contract_clean": all(
                not run["contract_errors"] for run in runs.values()
            ),
            "all_selected_rules_covered": all(
                run["transform"]["coverage_valid"] for run in noncanonical
            ),
            "zero_canonical_hit_to_phrase_miss": all(
                run["session_changes_vs_canonical"]["hit_to_miss_count"] == 0
                for run in noncanonical
            ),
        }
        artifact["frozen_winner_gate"] = {
            "winner": FROZEN_WINNER_ID,
            "checks": checks,
            "passed": all(checks.values()),
        }

    postflight = capture_gate_snapshot(
        PROJECT_ROOT,
        source_paths=provenance_sources,
        input_paths=provenance_inputs,
        selection_commit=SELECTION_COMMIT,
        frozen_architecture_path=PROJECT_ROOT / "starter" / "architecture_lab.py",
        local_selection_artifact=(
            PROJECT_ROOT / "experiments" / "p4_architecture_search.json"
        ),
        expected_selection_artifact_sha256=SELECTION_RESULT_SHA256,
    )
    assert_gate_snapshot_stable(preflight, postflight)
    artifact["provenance"]["postflight"] = postflight
    artifact["provenance"]["snapshot_stable"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[generalization] wrote {args.output}", flush=True)
    if args.architecture_variant and not artifact["frozen_winner_gate"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
