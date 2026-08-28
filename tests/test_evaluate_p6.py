from __future__ import annotations

import json
import hashlib
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluator.local_evaluator import load_jsonl
from scripts.evaluate_p6 import (
    ACTIVE_ID,
    ConfirmationWorkerFailure,
    CONTROL_ID,
    DEFAULT_P1,
    DEFAULT_P5,
    DEFAULT_SELECTION,
    EXPECTED_P1_SAMPLES_SHA256,
    EXPECTED_P5_SHA256,
    EXPECTED_SELECTION_SHA256,
    SERVED_REFERENCE_ID,
    SHADOW_ID,
    _sha256,
    _source_paths,
    _samples_sha256,
    assert_clean_preregistered_snapshot,
    attempt_confirmation,
    build_confirmation,
    build_exact_totals,
    build_posthoc_pool_audit,
    exact_totals_match_metrics,
    gate_variant,
    load_frozen_inputs,
    main,
    policy_common_turn_bridge,
    run_clean_confirmation,
    run_isolated_worker,
    run_selection_workers,
    select_winner,
    served_reference_bridge,
    validate_official_asset_hashes,
    validate_output_path,
    validate_active_invariants,
    validate_selection_samples,
)


def _metrics(sessions: list[dict]) -> tuple[dict, dict]:
    totals = build_exact_totals(sessions)
    count = len(sessions)
    hit_rate = round(totals["hit_count"] / count, 6)
    mrr = round(totals["rr_sum_x2520"] / (2520 * count), 6)
    mttc = round(totals["mttc_turn_sum"] / count, 6)
    efficiency = round(max(0.0, min(1.0, (11.0 - mttc) / 10.0)), 6)
    score = round(0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency, 6)
    return (
        {
            "sample_count": count,
            "hit_rate_at_10": hit_rate,
            "mrr": mrr,
            "mttc": mttc,
            "efficiency": efficiency,
            "recommended_technical_score": score,
            "reported_token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "scenario_metrics": {},
        },
        totals,
    )


def _session(
    sample_id: str,
    scenario: str,
    rank: int | None,
    turn: int | None,
) -> dict:
    return {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "hit": rank is not None,
        "first_hit_turn": turn,
        "best_rank": rank,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
    }


def _audit_record(
    session_index: int,
    *,
    changed: bool = False,
    deep_query: bool = False,
    turn: int = 1,
) -> dict:
    base = [f"p{index:03d}" for index in range(120)] if deep_query else ["a", "b"]
    deep = [*base, "newcomer"] if deep_query else []
    baseline = [f"p{index:03d}" for index in range(10)]
    proposal = [*baseline[:9], "newcomer"] if changed else list(baseline)
    return {
        "session_index": session_index,
        "turn": turn,
        "query_terms": ["dress", "linen"],
        "excluded_terms": ["wool"],
        "base_pool": base,
        "deep_pool": deep,
        "strict_pool": [],
        "base_union_pool": base,
        "deep_union_pool": deep,
        "baseline_top10": baseline,
        "proposal_top10": proposal,
        "served_top10": proposal if changed else baseline,
        "active": changed,
        "deep_query_executed": deep_query,
        "prefix": {"matches": True if deep_query else None},
        "trigger": {"enabled": deep_query},
        "guard": {"applied": changed},
        "coverage_by_parent_asin": {baseline[-1]: 1, "newcomer": 2},
        "matched_excluded_terms_by_parent_asin": {},
        "reason": "accepted" if changed else "not_triggered",
    }


def _run(
    variant_id: str,
    sessions: list[dict],
    *,
    functional_hash: str,
    response_hash: str,
    records: list[dict] | None = None,
    output_changes: int = 0,
    evaluation_seconds: float = 10.0,
    p95_ms: float = 20.0,
    rss_increment_bytes: int | None = 100,
    worker_pid: int = 101,
    worker_phase: str = "selection",
    worker_nonce: str = "1" * 32,
) -> dict:
    metrics, totals = _metrics(sessions)
    response_sessions = [
        [{"turn": record["turn"], "response": {}}]
        for record in (records or [])
    ]
    if not records:
        response_sessions = [[{"turn": 1, "response": {}}] for _ in sessions]
    turn_audit = records or []
    turn_audit_hash = hashlib.sha256(
        json.dumps(turn_audit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "variant_id": variant_id,
        "stats": {
            "activations": output_changes,
            "output_changes": output_changes,
        },
        "turn_audit": turn_audit,
        "turn_audit_sha256": turn_audit_hash,
        "timing": {
            "evaluation_seconds": evaluation_seconds,
            "respond_latency": {"p95_ms": p95_ms},
        },
        "memory": {
            "backend": "test-current-rss",
            "baseline_rss_bytes": 1_000,
            "peak_rss_bytes": (
                None if rss_increment_bytes is None else 1_000 + rss_increment_bytes
            ),
            "peak_rss_increment_bytes": rss_increment_bytes,
            "available": rss_increment_bytes is not None,
        },
        "contract_errors": [],
        "metrics": metrics,
        "exact_totals": totals,
        "functional_result_sha256": functional_hash,
        "response_trace_sha256": response_hash,
        "response_sessions": response_sessions,
        "sessions": sessions,
        "worker_process": {
            "isolated": True,
            "pid": worker_pid,
            "phase": worker_phase,
            "worker_nonce": worker_nonce,
        },
    }


class EvaluateP6Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.control_sessions = [
            _session("derived_p6_0001", "buying", 2, 3),
            _session("derived_p6_0002", "browsing", None, None),
        ]
        self.active_sessions = [
            _session("derived_p6_0001", "buying", 1, 3),
            _session("derived_p6_0002", "browsing", None, None),
        ]
        self.sample_ids = {"derived_p6_0001", "derived_p6_0002"}
        self.control = _run(
            CONTROL_ID,
            self.control_sessions,
            functional_hash="control-functional",
            response_hash="control-response",
            worker_pid=101,
            worker_nonce="1" * 32,
        )
        self.served = _run(
            "served.Agent.coverage_off",
            self.control_sessions,
            functional_hash="control-functional",
            response_hash="control-response",
            worker_pid=100,
            worker_nonce="0" * 32,
        )
        records = [
            _audit_record(0, changed=True, deep_query=True),
            _audit_record(1),
        ]
        self.active = _run(
            ACTIVE_ID,
            self.active_sessions,
            functional_hash="active-functional",
            response_hash="active-response",
            records=records,
            output_changes=1,
            evaluation_seconds=12.0,
            p95_ms=24.0,
            rss_increment_bytes=110,
            worker_pid=103,
            worker_nonce="3" * 32,
        )
        self.pool_audit = {
            "alignment": {"passed": True},
            "base_union_recalled_session_count": 1,
            "deep_union_recalled_session_count": 2,
            "rescued_session_count": 1,
        }

    def test_source_snapshot_includes_every_direct_agent_dependency(self) -> None:
        paths = _source_paths()
        self.assertTrue(
            {
                "runner",
                "p6_lab",
                "adaptive_depth",
                "agent",
                "coverage",
                "reranker",
                "attributes",
                "clarification",
                "slot_ledger",
                "response_contract",
                "evaluator",
                "selection_builder",
                "generalization_helpers",
                "resource_measurement",
                "official_asset_verifier",
            }
            <= set(paths)
        )
        self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_frozen_corpus_and_prior_hashes_are_exact(self) -> None:
        self.assertEqual(_sha256(DEFAULT_SELECTION), EXPECTED_SELECTION_SHA256)
        self.assertEqual(_sha256(DEFAULT_P5), EXPECTED_P5_SHA256)
        self.assertEqual(
            _samples_sha256(load_jsonl(DEFAULT_P1)), EXPECTED_P1_SAMPLES_SHA256
        )

    def test_official_catalog_and_public_content_hashes_are_hard_gated(self) -> None:
        verified = validate_official_asset_hashes(
            Path("data/catalog.jsonl"), Path("data/public_set.jsonl")
        )
        self.assertTrue(verified["catalog_hash_verified"])
        self.assertTrue(verified["released_public_blob_verified"])
        with mock.patch("scripts.evaluate_p6._sha256", return_value="drifted"):
            with self.assertRaisesRegex(ValueError, "official catalog"):
                validate_official_asset_hashes(
                    Path("data/catalog.jsonl"), Path("data/public_set.jsonl")
                )

    def test_dirty_or_unnamed_preregistration_snapshot_is_rejected(self) -> None:
        clean = {
            "git": {
                "branch": "p4-architecture-search",
                "commit": "abc123",
                "dirty": False,
                "status_porcelain": [],
            }
        }
        assert_clean_preregistered_snapshot(clean)
        dirty = json.loads(json.dumps(clean))
        dirty["git"]["dirty"] = True
        dirty["git"]["status_porcelain"] = [" M starter/p6_lab.py"]
        with self.assertRaisesRegex(RuntimeError, "dirty preregistration tree"):
            assert_clean_preregistered_snapshot(dirty)

    def test_selection_validation_requires_three_disjoint_exclusions(self) -> None:
        def row(sample_id: str, target: str, scenario: str = "buying") -> dict:
            return {
                "sample_id": sample_id,
                "scenario_type": scenario,
                "ground_truth": {"parent_asin": target},
            }

        selected = [row("derived_p6_0001", "p6")]
        public = [row("public_1", "public")]
        p1 = [row("derived_p1_0001", "p1")]
        p5 = [row("derived_p5_0001", "p5")]
        result = validate_selection_samples(
            selected,
            public,
            p1,
            p5,
            {"p6", "public", "p1", "p5"},
            expected_count=1,
            expected_exclusion_count=1,
        )
        self.assertEqual(result["prior_p5_derived_target_overlap"], 0)
        with self.assertRaisesRegex(ValueError, "prior_p5_derived"):
            validate_selection_samples(
                selected,
                public,
                p1,
                [row("derived_p5_0001", "p6")],
                {"p6", "public", "p1"},
                expected_count=1,
                expected_exclusion_count=1,
            )

    def test_selection_file_hash_is_checked_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.jsonl"
            with mock.patch(
                "scripts.evaluate_p6.validate_official_asset_hashes", return_value={}
            ), mock.patch("scripts.evaluate_p6._sha256", return_value="wrong"):
                with self.assertRaisesRegex(ValueError, "P6 selection corpus"):
                    load_frozen_inputs(missing, missing, missing, missing, missing)

    def test_exact_totals_use_integer_rank_and_turn_contributions(self) -> None:
        totals = build_exact_totals(self.control_sessions)
        self.assertEqual(totals["hit_count"], 1)
        self.assertEqual(totals["rr_sum_x2520"], 1260)
        self.assertEqual(totals["mttc_turn_sum"], 14)
        self.assertEqual(
            totals["official_contribution_sum_x25200"],
            12_600 + 3 * 1260 + 504 * 8,
        )
        self.assertTrue(exact_totals_match_metrics(self.control))
        self.control["metrics"]["mrr"] += 0.000001
        self.assertFalse(exact_totals_match_metrics(self.control))

    def test_served_bridge_requires_response_function_and_exact_totals(self) -> None:
        bridge = served_reference_bridge(self.control, self.served, self.sample_ids)
        self.assertTrue(bridge["passed"])
        self.served["response_trace_sha256"] = "different"
        self.assertFalse(
            served_reference_bridge(self.control, self.served, self.sample_ids)["passed"]
        )

    def test_shadow_is_exact_and_never_selectable(self) -> None:
        records = [_audit_record(0), _audit_record(1)]
        shadow = _run(
            SHADOW_ID,
            self.control_sessions,
            functional_hash="control-functional",
            response_hash="control-response",
            records=records,
        )
        gate = gate_variant(shadow, self.control, self.served, self.sample_ids)
        self.assertEqual(gate["decision"], "shadow_only")
        shadow["response_trace_sha256"] = "changed"
        self.assertEqual(
            gate_variant(shadow, self.control, self.served, self.sample_ids)["decision"],
            "invalid_shadow",
        )

    def test_active_gate_uses_exact_scenario_and_both_resource_limits(self) -> None:
        gate = gate_variant(
            self.active,
            self.control,
            self.served,
            self.sample_ids,
            self.pool_audit,
        )
        self.assertEqual(gate["decision"], "eligible")
        self.assertTrue(gate["gates"]["technical_score_strict_exact_improvement"])
        self.assertTrue(gate["gates"]["scenario_hit_counts_non_decrease"])
        self.assertTrue(gate["gates"]["evaluation_time_within_1_30x"])
        self.assertTrue(gate["gates"]["response_p95_within_1_30x"])
        self.assertTrue(gate["gates"]["peak_rss_increment_within_1_20x"])
        self.assertTrue(gate["gates"]["absolute_peak_rss_within_1_20x"])
        self.assertTrue(gate["gates"]["evaluation_time_vs_served_within_1_30x"])
        self.assertTrue(gate["gates"]["response_p95_vs_served_within_1_30x"])
        self.assertTrue(gate["gates"]["absolute_peak_rss_vs_served_within_1_20x"])
        self.assertTrue(gate["gates"]["deep_union_session_recall_strict_improvement"])

        self.active["timing"]["respond_latency"]["p95_ms"] = 26.1
        rejected = gate_variant(
            self.active,
            self.control,
            self.served,
            self.sample_ids,
            self.pool_audit,
        )
        self.assertEqual(rejected["decision"], "reject")
        self.assertFalse(rejected["gates"]["response_p95_within_1_30x"])

    def test_active_gate_rejects_unavailable_rss_or_no_pool_recall_gain(self) -> None:
        self.active["memory"].update(
            {"available": False, "peak_rss_increment_bytes": None}
        )
        gate = gate_variant(
            self.active,
            self.control,
            self.served,
            self.sample_ids,
            self.pool_audit,
        )
        self.assertFalse(gate["gates"]["peak_rss_increment_within_1_20x"])
        no_gain = {
            **self.pool_audit,
            "deep_union_recalled_session_count": 1,
            "rescued_session_count": 0,
        }
        gate = gate_variant(
            self.active,
            self.control,
            self.served,
            self.sample_ids,
            no_gain,
        )
        self.assertFalse(
            gate["gates"]["deep_union_session_recall_strict_improvement"]
        )
        self.assertFalse(gate["gates"]["deep_union_rescued_session_count_positive"])

    def test_active_gate_rejects_absolute_peak_rss_above_limit(self) -> None:
        self.active["memory"]["peak_rss_bytes"] = 1_400
        gate = gate_variant(
            self.active,
            self.control,
            self.served,
            self.sample_ids,
            self.pool_audit,
        )
        self.assertFalse(gate["gates"]["absolute_peak_rss_within_1_20x"])
        self.assertEqual(gate["decision"], "reject")

    def test_active_gate_also_uses_true_served_agent_resource_baseline(self) -> None:
        self.served["timing"]["evaluation_seconds"] = 8.0
        self.served["timing"]["respond_latency"]["p95_ms"] = 15.0
        self.served["memory"]["peak_rss_bytes"] = 800
        gate = gate_variant(
            self.active,
            self.control,
            self.served,
            self.sample_ids,
            self.pool_audit,
        )
        self.assertFalse(gate["gates"]["evaluation_time_vs_served_within_1_30x"])
        self.assertFalse(gate["gates"]["response_p95_vs_served_within_1_30x"])
        self.assertFalse(gate["gates"]["absolute_peak_rss_vs_served_within_1_20x"])
        self.assertEqual(gate["decision"], "reject")

    def test_active_gate_rejects_exact_official_contribution_regression(self) -> None:
        sessions = [
            _session("derived_p6_0001", "buying", 3, 3),
            _session("derived_p6_0002", "browsing", 1, 2),
        ]
        regressing = _run(
            ACTIVE_ID,
            sessions,
            functional_hash="regression",
            response_hash="regression",
            records=self.active["turn_audit"],
            output_changes=1,
        )
        gate = gate_variant(
            regressing,
            self.control,
            self.served,
            self.sample_ids,
            self.pool_audit,
        )
        self.assertFalse(gate["gates"]["zero_official_contribution_regression"])
        self.assertEqual(gate["decision"], "reject")

    def test_active_invariants_are_independently_checked(self) -> None:
        invariants = validate_active_invariants(self.active)
        self.assertTrue(invariants["passed"])
        self.active["turn_audit"][0]["proposal_top10"][0] = "bad-prefix"
        self.active["turn_audit"][0]["served_top10"][0] = "bad-prefix"
        failed = validate_active_invariants(self.active)
        self.assertFalse(failed["top9_preserved"])
        self.assertFalse(failed["passed"])

    def test_active_invariants_reject_excluded_newcomer_and_route_mismatch(self) -> None:
        record = self.active["turn_audit"][0]
        record["matched_excluded_terms_by_parent_asin"] = {"newcomer": ["wool"]}
        record["deep_pool"][0] = "wrong"
        failed = validate_active_invariants(self.active)
        self.assertFalse(failed["excluded_terms_respected"])
        self.assertFalse(failed["base_is_deep_prefix"])

    def test_active_invariants_require_exact_rank10_and_no_base_union_overlap(self) -> None:
        record = self.active["turn_audit"][0]
        record["base_union_pool"].append("newcomer")
        record["proposal_top10"] = ["newcomer", *record["proposal_top10"][:9]]
        record["served_top10"] = list(record["proposal_top10"])
        failed = validate_active_invariants(self.active)
        self.assertIn(
            "record_0:newcomer_overlaps_base_union", failed["violations"]
        )
        self.assertIn("record_0:top9_not_preserved", failed["violations"])

    def test_policy_bridge_compares_only_common_turn_prefix_without_recommendations(self) -> None:
        control = json.loads(json.dumps(self.control))
        active = json.loads(json.dumps(self.control))
        shared = {
            "message": "same question",
            "ask_attribute": "color",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        control["response_sessions"] = [[
            {"turn": 1, "response": {**shared, "recommendations": ["a"]}},
            {"turn": 2, "response": {**shared, "recommendations": ["b"]}},
        ]]
        active["response_sessions"] = [[
            {"turn": 1, "response": {**shared, "recommendations": ["different"]}}
        ]]
        bridge = policy_common_turn_bridge(active, control)
        self.assertTrue(bridge["passed"])
        self.assertEqual(bridge["common_turn_count"], 1)
        active["response_sessions"][0][0]["response"]["ask_attribute"] = "style"
        self.assertFalse(policy_common_turn_bridge(active, control)["passed"])

    def test_posthoc_pool_audit_excludes_pre_override_turns(self) -> None:
        records = [
            {
                **_audit_record(0, deep_query=True),
                "base_pool": ["other"],
                "deep_pool": ["other", "target0"],
                "base_union_pool": ["other"],
                "deep_union_pool": ["other", "target0"],
            },
            {
                **_audit_record(1, deep_query=True, turn=1),
                "base_pool": ["other"],
                "deep_pool": ["other", "target1"],
                "base_union_pool": ["other"],
                "deep_union_pool": ["other", "target1"],
            },
            {
                **_audit_record(1, deep_query=True, turn=2),
                "base_pool": ["other"],
                "deep_pool": ["other", "target1"],
                "base_union_pool": ["other"],
                "deep_union_pool": ["other", "target1"],
            },
            {
                **_audit_record(1, deep_query=True, turn=3),
                "base_pool": ["other"],
                "deep_pool": ["other", "target1"],
                "base_union_pool": ["other"],
                "deep_union_pool": ["other", "target1"],
            },
            {
                **_audit_record(1, deep_query=True, turn=4),
                "base_pool": ["other"],
                "deep_pool": ["other", "target1"],
                "base_union_pool": ["other"],
                "deep_union_pool": ["other", "target1"],
            },
        ]
        shadow = {
            "variant_id": SHADOW_ID,
            "turn_audit": records,
            "response_sessions": [
                [{"turn": 1, "response": {}}],
                [
                    {"turn": 1, "response": {}},
                    {"turn": 2, "response": {}},
                    {"turn": 3, "response": {}},
                    {"turn": 4, "response": {}},
                ],
            ],
            "sessions": [{}, {}],
        }
        samples = [
            {
                "scenario_type": "buying",
                "intent_card": {},
                "behavior": {"scenario_type": "buying"},
                "ground_truth": {"parent_asin": "target0"},
            },
            {
                "scenario_type": "intent_override",
                "intent_card": {},
                "behavior": {"override": {"turn": 4}},
                "ground_truth": {"parent_asin": "target1"},
            },
        ]
        audit = build_posthoc_pool_audit(shadow, samples, {})
        self.assertEqual(audit["intent_override_pre_switch_turns_excluded"], 3)
        self.assertEqual(audit["eligible_posthoc_turn_count"], 2)
        self.assertEqual(audit["deep_only_target_recovery_turn_count"], 2)
        self.assertEqual(audit["rescued_session_count"], 2)
        self.assertFalse(audit["labels_exposed_to_agent"])

    def test_repeat_confirmation_requires_exact_hashes_and_resources(self) -> None:
        repeat_served = json.loads(json.dumps(self.served))
        repeat_control = json.loads(json.dumps(self.control))
        repeat_active = json.loads(json.dumps(self.active))
        for run, pid, nonce in (
            (repeat_served, 200, "a" * 32),
            (repeat_control, 201, "b" * 32),
            (repeat_active, 203, "c" * 32),
        ):
            run["worker_process"] = {
                "isolated": True,
                "pid": pid,
                "phase": "confirmation",
                "worker_nonce": nonce,
            }
        pair = {
            "isolated_processes": True,
            "confirmation_worker_process_count": 3,
            "distinct_worker_pid_count": 3,
            "worker_pids": [200, 201, 203],
            "distinct_worker_nonce_count": 3,
            "worker_nonces": ["a" * 32, "b" * 32, "c" * 32],
            "runs": {
                SERVED_REFERENCE_ID: repeat_served,
                CONTROL_ID: repeat_control,
                ACTIVE_ID: repeat_active,
            },
        }
        confirmation = build_confirmation(
            self.active, self.control, self.served, pair, self.sample_ids
        )
        self.assertTrue(confirmation["passed"])
        self.assertEqual(confirmation["confirmation_worker_process_count"], 3)
        self.assertEqual(confirmation["served_control_active_total_worker_process_count"], 6)
        self.assertEqual(confirmation["runs_per_variant"][ACTIVE_ID], 2)
        changed_pair = json.loads(json.dumps(pair))
        changed_pair["runs"][ACTIVE_ID]["functional_result_sha256"] = "different"
        self.assertFalse(
            build_confirmation(
                self.active,
                self.control,
                self.served,
                changed_pair,
                self.sample_ids,
            )["passed"]
        )
        served_resource_bad = json.loads(json.dumps(pair))
        served_resource_bad["runs"][SERVED_REFERENCE_ID]["timing"][
            "evaluation_seconds"
        ] = 5.0
        self.assertFalse(
            build_confirmation(
                self.active,
                self.control,
                self.served,
                served_resource_bad,
                self.sample_ids,
            )["checks"]["evaluation_time_vs_served_within_1_30x"]
        )
        audit_changed = json.loads(json.dumps(pair))
        audit_changed["runs"][ACTIVE_ID]["turn_audit_sha256"] = "different"
        self.assertFalse(
            build_confirmation(
                self.active,
                self.control,
                self.served,
                audit_changed,
                self.sample_ids,
            )["passed"]
        )

    def test_selection_promotes_only_eligible_confirmed_active(self) -> None:
        gates = {
            CONTROL_ID: {"decision": "control"},
            SHADOW_ID: {"decision": "shadow_only"},
            ACTIVE_ID: {"decision": "eligible"},
        }
        promoted = select_winner(gates, {"attempted": True, "passed": True})
        self.assertEqual(promoted["decision"], "promote_active")
        self.assertTrue(promoted["public_confirmation_allowed"])
        retained = select_winner(gates, {"attempted": True, "passed": False})
        self.assertEqual(retained["winner_id"], CONTROL_ID)
        self.assertEqual(retained["decision"], "retain_control_confirmation_failed")
        self.assertFalse(retained["public_confirmation_allowed"])

    def test_confirmation_worker_failure_is_recorded_and_retains_control(self) -> None:
        completed_served = json.loads(json.dumps(self.served))
        completed_served["worker_process"] = {
            "isolated": True,
            "pid": 200,
            "phase": "confirmation",
            "worker_nonce": "a" * 32,
        }
        failure = ConfirmationWorkerFailure(
            CONTROL_ID,
            2,
            {SERVED_REFERENCE_ID: completed_served},
            RuntimeError("child failed"),
        )
        with mock.patch(
            "scripts.evaluate_p6.run_clean_confirmation", side_effect=failure
        ):
            confirmation = attempt_confirmation(
                self.active,
                self.control,
                self.served,
                self.sample_ids,
                Path("catalog"),
                Path("selection"),
                Path("public"),
                Path("p1"),
                Path("p5"),
            )
        self.assertTrue(confirmation["attempted"])
        self.assertFalse(confirmation["passed"])
        self.assertEqual(confirmation["reason"], "confirmation_worker_failure")
        self.assertEqual(confirmation["confirmation_worker_attempt_count"], 2)
        self.assertEqual(confirmation["confirmation_worker_process_count"], 1)
        gates = {
            CONTROL_ID: {"decision": "control"},
            SHADOW_ID: {"decision": "shadow_only"},
            ACTIVE_ID: {"decision": "eligible"},
        }
        selection = select_winner(gates, confirmation)
        self.assertEqual(selection["decision"], "retain_control_confirmation_failed")
        self.assertEqual(selection["winner_id"], CONTROL_ID)

    def test_clean_confirmation_uses_three_distinct_variant_processes(self) -> None:
        pids = {
            SERVED_REFERENCE_ID: 200,
            CONTROL_ID: 201,
            ACTIVE_ID: 203,
        }

        def fake_worker(worker_id: str, phase: str, *_: object) -> dict:
            return {
                "variant_id": worker_id,
                "worker_process": {
                    "isolated": True,
                    "pid": pids[worker_id],
                    "phase": phase,
                    "worker_nonce": f"{pids[worker_id]:032x}",
                },
            }

        with mock.patch(
            "scripts.evaluate_p6.run_isolated_worker", side_effect=fake_worker
        ) as worker:
            result = run_clean_confirmation(
                Path("catalog"), Path("selection"), Path("public"), Path("p1"), Path("p5")
            )
        self.assertTrue(result["isolated_processes"])
        self.assertEqual(result["confirmation_worker_process_count"], 3)
        self.assertEqual(result["distinct_worker_pid_count"], 3)
        self.assertEqual(worker.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in worker.call_args_list],
            [SERVED_REFERENCE_ID, CONTROL_ID, ACTIVE_ID],
        )

    def test_isolated_worker_command_validates_child_pid_and_cleans_temp_output(self) -> None:
        outputs: list[Path] = []

        class FakeProcess:
            pid = 4321
            returncode = 0

            def __init__(self, command: list[str]) -> None:
                self.command = command

            def communicate(self, timeout: int | None = None) -> tuple[str, str]:
                self.assert_timeout(timeout)
                output = Path(
                    self.command[self.command.index("--isolated-worker-output") + 1]
                )
                worker_id = self.command[
                    self.command.index("--isolated-worker-id") + 1
                ]
                phase = self.command[
                    self.command.index("--isolated-worker-phase") + 1
                ]
                nonce = self.command[
                    self.command.index("--isolated-worker-nonce") + 1
                ]
                outputs.append(output)
                output.write_text(
                    json.dumps(
                        {
                            "isolated_process": True,
                            "worker_id": worker_id,
                            "phase": phase,
                            "worker_nonce": nonce,
                            "pid": self.pid,
                            "run": {"variant_id": worker_id},
                        }
                    ),
                    encoding="utf-8",
                )
                return "", ""

            @staticmethod
            def assert_timeout(timeout: int | None) -> None:
                if timeout != 900:
                    raise AssertionError("unexpected worker timeout")

        def fake_popen(command: list[str], **_: object) -> FakeProcess:
            return FakeProcess(command)

        with mock.patch("scripts.evaluate_p6.os.getpid", return_value=999), mock.patch(
            "scripts.evaluate_p6.subprocess.Popen", side_effect=fake_popen
        ) as launcher:
            run = run_isolated_worker(
                ACTIVE_ID,
                "selection",
                Path("catalog"),
                Path("selection"),
                Path("public"),
                Path("p1"),
                Path("p5"),
            )
        self.assertEqual(run["worker_process"]["pid"], 4321)
        self.assertEqual(launcher.call_count, 1)
        command = launcher.call_args.args[0]
        self.assertEqual(command[0], __import__("sys").executable)
        self.assertIn("--isolated-worker-output", command)
        self.assertTrue(outputs)
        self.assertFalse(outputs[0].exists())
        self.assertFalse(outputs[0].parent.exists())

    def test_isolated_worker_failure_and_parent_pid_are_rejected(self) -> None:
        failed = mock.Mock(pid=4321, returncode=9)
        failed.communicate.return_value = ("", "boom")
        with mock.patch("scripts.evaluate_p6.subprocess.Popen", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "isolated worker"):
                run_isolated_worker(
                    ACTIVE_ID,
                    "selection",
                    Path("catalog"),
                    Path("selection"),
                    Path("public"),
                    Path("p1"),
                    Path("p5"),
                )

        timed_out = mock.Mock(pid=4321, returncode=None)
        timed_out.communicate.side_effect = [
            subprocess.TimeoutExpired(["worker"], 900),
            ("", ""),
        ]
        with mock.patch("scripts.evaluate_p6.subprocess.Popen", return_value=timed_out):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                run_isolated_worker(
                    ACTIVE_ID,
                    "selection",
                    Path("catalog"),
                    Path("selection"),
                    Path("public"),
                    Path("p1"),
                    Path("p5"),
                )
        timed_out.kill.assert_called_once()

        def same_pid(command: list[str], **_: object) -> mock.Mock:
            process = mock.Mock(pid=999, returncode=0)

            def communicate(timeout: int | None = None) -> tuple[str, str]:
                output = Path(command[command.index("--isolated-worker-output") + 1])
                output.write_text(
                    json.dumps(
                        {
                            "isolated_process": True,
                            "worker_id": ACTIVE_ID,
                            "phase": "selection",
                            "worker_nonce": command[
                                command.index("--isolated-worker-nonce") + 1
                            ],
                            "pid": 999,
                            "run": {"variant_id": ACTIVE_ID},
                        }
                    ),
                    encoding="utf-8",
                )
                return "", ""

            process.communicate.side_effect = communicate
            return process

        with mock.patch("scripts.evaluate_p6.os.getpid", return_value=999), mock.patch(
            "scripts.evaluate_p6.subprocess.Popen", side_effect=same_pid
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid metadata"):
                run_isolated_worker(
                    ACTIVE_ID,
                    "selection",
                    Path("catalog"),
                    Path("selection"),
                    Path("public"),
                    Path("p1"),
                    Path("p5"),
                )

        def wrong_nonce(command: list[str], **_: object) -> mock.Mock:
            process = mock.Mock(pid=4321, returncode=0)

            def communicate(timeout: int | None = None) -> tuple[str, str]:
                output = Path(command[command.index("--isolated-worker-output") + 1])
                output.write_text(
                    json.dumps(
                        {
                            "isolated_process": True,
                            "worker_id": ACTIVE_ID,
                            "phase": "selection",
                            "worker_nonce": "0" * 32,
                            "pid": 4321,
                            "run": {"variant_id": ACTIVE_ID},
                        }
                    ),
                    encoding="utf-8",
                )
                return "", ""

            process.communicate.side_effect = communicate
            return process

        with mock.patch(
            "scripts.evaluate_p6.uuid.uuid4",
            return_value=mock.Mock(hex="f" * 32),
        ), mock.patch("scripts.evaluate_p6.subprocess.Popen", side_effect=wrong_nonce):
            with self.assertRaisesRegex(RuntimeError, "invalid metadata"):
                run_isolated_worker(
                    ACTIVE_ID,
                    "selection",
                    Path("catalog"),
                    Path("selection"),
                    Path("public"),
                    Path("p1"),
                    Path("p5"),
                )

    def test_selection_workers_launch_four_distinct_processes(self) -> None:
        worker_ids = [SERVED_REFERENCE_ID, CONTROL_ID, SHADOW_ID, ACTIVE_ID]

        def fake_worker(worker_id: str, phase: str, *_: object) -> dict:
            index = worker_ids.index(worker_id)
            return {
                "variant_id": worker_id,
                "worker_process": {
                    "isolated": True,
                    "pid": 1000 + index,
                    "phase": phase,
                    "worker_nonce": f"{index:032x}",
                },
            }

        with mock.patch(
            "scripts.evaluate_p6.run_isolated_worker", side_effect=fake_worker
        ) as worker:
            served, runs, pids = run_selection_workers(
                Path("catalog"), Path("selection"), Path("public"), Path("p1"), Path("p5")
            )
        self.assertEqual(served["variant_id"], SERVED_REFERENCE_ID)
        self.assertEqual(set(runs), {CONTROL_ID, SHADOW_ID, ACTIVE_ID})
        self.assertEqual(len(set(pids)), 4)
        self.assertEqual(worker.call_count, 4)

        def reused_pid(worker_id: str, phase: str, *_: object) -> dict:
            index = worker_ids.index(worker_id)
            return {
                "variant_id": worker_id,
                "worker_process": {
                    "isolated": True,
                    "pid": 777,
                    "phase": phase,
                    "worker_nonce": f"{index:032x}",
                },
            }

        with mock.patch(
            "scripts.evaluate_p6.run_isolated_worker", side_effect=reused_pid
        ):
            _, _, reused_pids = run_selection_workers(
                Path("catalog"),
                Path("selection"),
                Path("public"),
                Path("p1"),
                Path("p5"),
            )
        self.assertEqual(reused_pids, [777, 777, 777, 777])

        def duplicate_nonce(worker_id: str, phase: str, *_: object) -> dict:
            return {
                "variant_id": worker_id,
                "worker_process": {
                    "isolated": True,
                    "pid": 1000 + worker_ids.index(worker_id),
                    "phase": phase,
                    "worker_nonce": "d" * 32,
                },
            }

        with mock.patch(
            "scripts.evaluate_p6.run_isolated_worker", side_effect=duplicate_nonce
        ):
            with self.assertRaisesRegex(RuntimeError, "unique parent-issued nonces"):
                run_selection_workers(
                    Path("catalog"),
                    Path("selection"),
                    Path("public"),
                    Path("p1"),
                    Path("p5"),
                )

    def test_runtime_algorithm_sources_contain_no_target_access(self) -> None:
        for relative in ("starter/adaptive_depth.py", "starter/p6_lab.py"):
            source = Path(relative).read_text(encoding="utf-8")
            self.assertIsNone(
                re.search(r"\b(?:ground_truth|target|target_id|target_asin)\b", source)
            )

    def test_output_path_cannot_overwrite_inputs_sources_or_a_directory(self) -> None:
        catalog = Path("data/catalog.jsonl")
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            validate_output_path(catalog, [catalog])
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            validate_output_path(Path("scripts/evaluate_p6.py"), [catalog])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not a directory"):
                validate_output_path(Path(directory), [catalog])
            existing = Path(directory) / "worker.json"
            existing.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new temporary file"):
                validate_output_path(existing, [catalog], must_not_exist=True)

    def test_cli_writes_valid_artifact(self) -> None:
        artifact = {
            "selection": {
                "decision": "retain_control_active_rejected",
                "winner_id": CONTROL_ID,
                "experiment_valid": True,
            },
            "corpus": {"sha256": "frozen"},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with mock.patch(
                "scripts.evaluate_p6.run_selection", return_value=artifact
            ) as runner:
                exit_code = main(["--output", str(output)])
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), artifact)
            runner.assert_called_once()

    def test_cli_isolated_worker_writes_one_run_only(self) -> None:
        payload = {
            "isolated_process": True,
            "worker_id": ACTIVE_ID,
            "phase": "selection",
            "worker_nonce": "f" * 32,
            "pid": 4321,
            "run": {"variant_id": ACTIVE_ID},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "worker.json"
            with mock.patch(
                "scripts.evaluate_p6.run_single_worker", return_value=payload
            ) as worker:
                exit_code = main(
                    [
                        "--isolated-worker-output",
                        str(output),
                        "--isolated-worker-id",
                        ACTIVE_ID,
                        "--isolated-worker-phase",
                        "selection",
                        "--isolated-worker-nonce",
                        "f" * 32,
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            worker.assert_called_once()


if __name__ == "__main__":
    unittest.main()
