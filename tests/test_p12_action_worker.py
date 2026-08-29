from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import p12_action_worker as worker
from scripts import p12_actions
from starter.p5_lab import R01 as P5_R01


SAFE_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": None,
    "rating_style": "unknown",
    "preference_tags": ["clothing", "shoes"],
    "summary": "Usually shops for clothing and shoes.",
}


def _request(value: dict[str, object]) -> dict[str, object]:
    line = worker._canonical_bytes(value) + b"\n"
    return worker._parse_request(line)


def _ids(count: int = 120) -> tuple[str, ...]:
    return tuple(f"item-{index:03d}" for index in range(count))


def _valid_diagnostics() -> dict[str, object]:
    return {
        "configured_mode": "active",
        "effective_mode": "active",
        "identity_verified": True,
        "fallback": False,
        "reason_code": "scored",
        "top10_membership_preserved": True,
        "tail_preserved": True,
    }


class _FakeCandidateRows:
    def __init__(self, rows: tuple[int, ...]) -> None:
        self.rows = rows

    def __matmul__(self, vector: object) -> list[float]:
        if vector != (1.0,):
            raise AssertionError("unexpected query vector")
        return [float(row) / 100.0 for row in self.rows]


class _FakeMatrix:
    def __init__(self) -> None:
        self.reads: list[tuple[int, ...]] = []

    def __getitem__(self, rows: object) -> _FakeCandidateRows:
        if not isinstance(rows, list) or any(not isinstance(row, int) for row in rows):
            raise AssertionError("semantic matrix access was not advanced candidate indexing")
        selected = tuple(rows)
        self.reads.append(selected)
        return _FakeCandidateRows(selected)


class _FakeEncoder:
    def __init__(self, close_log: list[str]) -> None:
        self.queries: list[str] = []
        self.close_log = close_log

    def encode_query(self, query: str) -> tuple[float]:
        self.queries.append(query)
        return (1.0,)

    def close(self) -> None:
        self.close_log.append("encoder")


class _FakeIndex:
    def __init__(self, close_log: list[str]) -> None:
        self.asins = list(_ids(80))
        self.matrix = _FakeMatrix()
        self.close_log = close_log

    def search_query(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("full-catalog search_query must never be called")

    def search_vector(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("full-catalog search_vector must never be called")

    def close(self) -> None:
        self.close_log.append("index")


class _FakeP11Agent:
    def __init__(self, close_log: list[str]) -> None:
        self.close_log = close_log
        self.closed = False
        self.sessions: set[str] = set()
        self.capture: dict[str, object] | None = None
        self.messages: list[tuple[str, str, int, int]] = []

    def _p11_status(self) -> dict[str, object]:
        if self.closed:
            return {
                **_valid_diagnostics(),
                "effective_mode": "off",
                "fallback": True,
                "reason_code": "agent_closed",
            }
        return _valid_diagnostics()

    def reset(self, session_id: str, _profile: dict[str, object]) -> None:
        self.sessions.add(session_id)

    def respond(
        self, session_id: str, message: str, turn: int, top_k: int
    ) -> dict[str, object]:
        if session_id not in self.sessions:
            raise AssertionError("unknown P11 session")
        self.messages.append((session_id, message, turn, top_k))
        r08 = _ids()
        p11 = (r08[1], r08[0], *r08[2:])
        p11_c50 = p11[:50]
        compact_negative = (*p11_c50[10:20], *p11_c50[:10], *p11_c50[20:])
        guarded_compact = list(p11_c50)
        guarded_compact[9], guarded_compact[11] = (
            guarded_compact[11],
            guarded_compact[9],
        )
        guarded_compact_strict = list(p11_c50)
        guarded_compact_strict[9], guarded_compact_strict[10] = (
            guarded_compact_strict[10],
            guarded_compact_strict[9],
        )
        family2_rankings: dict[str, tuple[str, ...]] = {}
        for name, challenger_index in (
            ("p11_evidence_novel_slot10_full", 11),
            ("hard_clause_novel_slot10_full", 12),
            ("two_signal_consensus_novel_slot10_full", 13),
        ):
            ranked = list(p11_c50)
            ranked[9], ranked[challenger_index] = (
                ranked[challenger_index],
                ranked[9],
            )
            family2_rankings[name] = tuple(ranked)
        family3_rankings: dict[str, tuple[str, ...]] = {}
        for name, challenger_index in (
            ("visible_constraint_rank_fusion_slot10_full", 14),
            ("dual_boundary_consensus_slot10_full", 15),
            ("recent_override_rank_fusion_slot10_full", 16),
        ):
            ranked = list(p11_c50)
            ranked[9], ranked[challenger_index] = (
                ranked[challenger_index],
                ranked[9],
            )
            family3_rankings[name] = tuple(ranked)
        self.capture = {
            "r08_full": r08,
            "p11_full": p11,
            "candidate_pools": {
                "c20": r08[:20],
                "c50": r08[:50],
                "c100": r08[:100],
            },
            "structured_full": tuple(reversed(r08[:50])),
            "semantic_full": (*r08[1:50], r08[0]),
            "compact_negative_full": compact_negative,
            "guarded_compact_slot10_full": tuple(guarded_compact),
            "guarded_compact_slot10_strict_full": tuple(
                guarded_compact_strict
            ),
            **family2_rankings,
            **family3_rankings,
            "p11_invariants": worker._validate_p11_invariants(
                r08, p11, _valid_diagnostics()
            ),
        }
        return {
            "ask_attribute": "color",
            "recommendations": [
                {"parent_asin": identifier} for identifier in p11[:top_k]
            ],
        }

    def take_last_capture(self, _session_id: str) -> dict[str, object]:
        if self.capture is None:
            raise AssertionError("missing P11 capture")
        capture, self.capture = self.capture, None
        return capture

    def drop_session(self, session_id: str) -> None:
        self.sessions.remove(session_id)

    def p12_timing(self) -> dict[str, list[float]]:
        return {"structured": [0.001] * 10, "semantic": [0.002] * 10}

    def p12_compact_summary(self) -> dict[str, object]:
        return {
            "counts": {
                "total_turns": 10,
                "turns_with_executable_constraints": 10,
            },
            "compiler_rejection_counts": {},
            "privacy": "aggregate counts only; no text, values, identifiers, ordinals, or labels",
        }

    def p12_family2_summary(self) -> dict[str, object]:
        return {
            "counts": {
                "total_turns": 10,
                "turns_with_complete_c50_scores": 10,
            },
            "privacy": "aggregate counts only; no text, values, identifiers, ordinals, or labels",
        }

    def p12_family3_summary(self) -> dict[str, object]:
        return {
            "counts": {"total_turns": 10, "compute_failure_count": 0},
            "reason_counts": {
                action: {
                    reason: 10 if reason == "activated" else 0
                    for reason in worker.FAMILY3_DECISION_REASONS
                }
                for action in (
                    p12_actions.VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10,
                    p12_actions.DUAL_BOUNDARY_CONSENSUS_SLOT10,
                    p12_actions.RECENT_OVERRIDE_RANK_FUSION_SLOT10,
                )
            },
            "compute_failure_counts": {
                p12_actions.VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10: 0,
                p12_actions.DUAL_BOUNDARY_CONSENSUS_SLOT10: 0,
                p12_actions.RECENT_OVERRIDE_RANK_FUSION_SLOT10: 0,
            },
            "privacy": "aggregate counts only; no text, values, identifiers, ordinals, or labels",
        }

    def close(self) -> None:
        self.closed = True
        self.close_log.append("p11")


class _FakeP5Agent:
    def __init__(self, close_log: list[str]) -> None:
        self.close_log = close_log
        self.closed = False
        self.sessions: set[str] = set()
        self.capture: dict[str, object] | None = None
        self.messages: list[tuple[str, str, int, int]] = []

    def reset(self, session_id: str, _profile: dict[str, object]) -> None:
        self.sessions.add(session_id)

    def respond(
        self, session_id: str, message: str, turn: int, top_k: int
    ) -> dict[str, object]:
        if session_id not in self.sessions:
            raise AssertionError("unknown P5 session")
        self.messages.append((session_id, message, turn, top_k))
        base = _ids()
        result_aware = (*base[10:20], *base[:10], *base[20:])
        self.capture = {
            "full": result_aware,
            "variant_id": P5_R01,
            "base": "R08",
        }
        return {
            "ask_attribute": None,
            "recommendations": [
                {"parent_asin": identifier} for identifier in result_aware[:top_k]
            ],
        }

    def take_last_capture(self, _session_id: str) -> dict[str, object]:
        if self.capture is None:
            raise AssertionError("missing P5 capture")
        capture, self.capture = self.capture, None
        return capture

    def drop_session(self, session_id: str) -> None:
        self.sessions.remove(session_id)

    def close(self) -> None:
        self.closed = True
        self.close_log.append("p5")


class _FakeSemantic:
    def __init__(self, close_log: list[str]) -> None:
        self.close_log = close_log
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.close_log.append("semantic")

    def summary(self) -> dict[str, object]:
        return {
            "mode": "candidate_only_c50",
            "query_count": 10,
            "candidate_matrix_rows_read": 500,
            "maximum_candidate_rows_read": 50,
            "full_catalog_search_calls": 0,
            "failure_count": 0,
        }


class RequestBoundaryTests(unittest.TestCase):
    def test_exact_request_schemas_and_safe_profile(self) -> None:
        reset = _request(
            {
                "operation": "reset",
                "request_id": 1,
                "ordinal": 1,
                "user_profile": dict(SAFE_PROFILE),
            }
        )
        self.assertEqual(set(reset), worker.REQUEST_SCHEMAS["reset"])
        self.assertEqual(reset["user_profile"], SAFE_PROFILE)

        respond = _request(
            {
                "operation": "respond",
                "request_id": 2,
                "ordinal": 1,
                "user_message": "I would like a blue jacket.",
                "turn": 1,
                "top_k": 10,
            }
        )
        self.assertEqual(set(respond), worker.REQUEST_SCHEMAS["respond"])

        for malformed in (
            {**reset, "target_id": "secret"},
            {**respond, "extra": None},
            {key: value for key, value in respond.items() if key != "turn"},
            {**respond, "top_k": 9},
            {**respond, "user_message": "show me B012345678"},
            {**respond, "request_id": True},
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(worker.P12WorkerError):
                    _request(malformed)

        unsafe_profile = dict(SAFE_PROFILE)
        unsafe_profile["history"] = []
        with self.assertRaises(worker.P12WorkerError):
            _request(
                {
                    "operation": "reset",
                    "request_id": 1,
                    "ordinal": 1,
                    "user_profile": unsafe_profile,
                }
            )

    def test_duplicate_json_keys_are_rejected_at_any_depth(self) -> None:
        with self.assertRaises(worker.P12WorkerError):
            worker._parse_request(
                b'{"operation":"finalize","request_id":1,"request_id":1}\n'
            )
        with self.assertRaises(worker.P12WorkerError):
            worker._parse_request(
                b'{"operation":"reset","request_id":1,"ordinal":1,'
                b'"user_profile":{"purchase_frequency":"not provided",'
                b'"average_prior_rating":null,"rating_style":"unknown",'
                b'"preference_tags":[],"summary":"safe","summary":"safe"}}\n'
            )

    def test_tracked_text_identity_normalizes_lf_crlf_and_lone_cr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = []
            for name, payload in (
                ("lf.json", b'{\n  "key": "value"\n}\n'),
                ("crlf.json", b'{\r\n  "key": "value"\r\n}\r\n'),
                ("cr.json", b'{\r  "key": "value"\r}\r'),
            ):
                path = root / name
                path.write_bytes(payload)
                identities.append(worker._lf_normalized_identity(path))
            self.assertEqual(identities[0], identities[1])
            self.assertEqual(identities[0], identities[2])

    def test_exact_parent_visible_response_shapes(self) -> None:
        self.assertEqual(
            worker._ready_response("0" * 32),
            {"kind": "ready", "nonce": "0" * 32},
        )
        self.assertEqual(
            worker._success_response("reset", 1, None),
            {"kind": "reply", "request_id": 1, "value": None},
        )
        self.assertEqual(
            worker._success_response("respond", 2, {"ask_attribute": "color"}),
            {
                "kind": "reply",
                "request_id": 2,
                "value": {"ask_attribute": "color"},
            },
        )
        self.assertEqual(
            worker._success_response("drop", 3, None),
            {"kind": "reply", "request_id": 3, "value": None},
        )
        receipt = {
            "trace_sha256": "a" * 64,
            "record_count": 10,
            "worker_summary": {"ok": True},
        }
        self.assertEqual(
            worker._success_response("finalize", 4, receipt),
            {"kind": "receipt", "request_id": 4, **receipt},
        )
        with self.assertRaises(worker.P12WorkerError):
            worker._success_response(
                "respond",
                2,
                {
                    "ask_attribute": None,
                    "recommendations": [{"parent_asin": "secret"}],
                },
            )

    def test_network_audit_is_fail_closed_and_counted(self) -> None:
        guard = worker.NetworkAuditGuard()
        guard.hook("open", ())
        guard.hook("socket.gethostname", ())
        with self.assertRaises(PermissionError):
            guard.hook("socket.connect", ())
        self.assertEqual(guard.attempt_count, 1)
        self.assertEqual(guard.local_metadata_count, 1)
        self.assertEqual(
            guard.event_counts,
            {"socket.connect": 1, "socket.gethostname": 1},
        )


class P11InvariantTests(unittest.TestCase):
    def test_active_membership_and_tail_invariants(self) -> None:
        r08 = _ids(20)
        p11 = (r08[1], r08[0], *r08[2:])
        snapshot = worker._validate_p11_invariants(
            r08, p11, _valid_diagnostics()
        )
        self.assertTrue(snapshot["observed_membership_preserved"])
        self.assertTrue(snapshot["observed_tail_preserved"])
        self.assertEqual(worker._validate_p11_invariant_snapshot(snapshot), snapshot)

        bad_status = _valid_diagnostics()
        bad_status["fallback"] = True
        for served, diagnostics in (
            (p11, bad_status),
            ((r08[10], *r08[1:10], r08[0], *r08[11:]), _valid_diagnostics()),
            ((*p11[:10], p11[11], p11[10], *p11[12:]), _valid_diagnostics()),
        ):
            with self.subTest(served=served, diagnostics=diagnostics):
                with self.assertRaises(worker.P12WorkerError):
                    worker._validate_p11_invariants(r08, served, diagnostics)

    def test_served_recommendations_are_bound_to_the_same_capture(self) -> None:
        captured = _ids(20)
        response = {
            "ask_attribute": "color",
            "recommendations": [
                {"parent_asin": identifier} for identifier in captured[:10]
            ],
        }
        self.assertEqual(
            worker._validate_served_response(response, captured, "fixture"), "color"
        )
        with self.assertRaises(worker.P12WorkerError):
            worker._validate_served_response(None, captured, "fixture")
        with self.assertRaises(worker.P12WorkerError):
            worker._validate_served_response(
                {**response, "recommendations": list(captured[:10])},
                captured,
                "fixture",
            )
        with self.assertRaises(worker.P12WorkerError):
            worker._validate_served_response(
                {
                    **response,
                    "recommendations": [
                        {"parent_asin": identifier}
                        for identifier in reversed(captured[:10])
                    ],
                },
                captured,
                "fixture",
            )
        with self.assertRaises(worker.P12WorkerError):
            worker._validate_served_response(
                {**response, "ask_attribute": "ground_truth"}, captured, "fixture"
            )

    def test_structured_priors_reuse_normalized_weighted_rrf(self) -> None:
        rankings = {
            "broad": ["item-a", "item-b", "item-c"],
            "strict": ["item-b", "item-c"],
        }
        candidates = ("item-b", "item-a", "item-c")
        priors = worker.P12CaptureAgent._rank_priors(candidates, rankings)
        broad_rank = {"item-a": 1, "item-b": 2, "item-c": 3}
        strict_rank = {"item-b": 1, "item-c": 2}
        raw = {
            identifier: worker.Agent._fusion_score(
                identifier, broad_rank, strict_rank
            )
            for identifier in candidates
        }
        maximum = max(raw.values())
        self.assertEqual(
            priors,
            {identifier: value / maximum for identifier, value in raw.items()},
        )
        self.assertEqual(max(priors.values()), 1.0)
        with self.assertRaises(worker.P12WorkerError):
            worker.P12CaptureAgent._rank_priors(
                ("outside-real-routes",), rankings
            )


class CandidateSemanticTests(unittest.TestCase):
    def test_encode_query_then_only_exact_candidate_matrix_rows(self) -> None:
        close_log: list[str] = []
        encoder = _FakeEncoder(close_log)
        index = _FakeIndex(close_log)
        runtime = worker.CandidateSemanticRuntime(encoder, index)
        candidates = ("item-017", "item-003", "item-041", "item-009")

        ranked = runtime.rank(["blue", "jacket"], candidates)

        self.assertEqual(encoder.queries, ["blue jacket"])
        self.assertEqual(index.matrix.reads, [(17, 3, 41, 9)])
        self.assertEqual(set(ranked), set(candidates))
        self.assertEqual(runtime.summary()["full_catalog_search_calls"], 0)
        self.assertEqual(runtime.summary()["maximum_candidate_rows_read"], 4)
        runtime.close()
        self.assertEqual(close_log, ["encoder", "index"])

    def test_missing_registry_entry_and_over_c50_fail_closed(self) -> None:
        close_log: list[str] = []
        runtime = worker.CandidateSemanticRuntime(
            _FakeEncoder(close_log), _FakeIndex(close_log)
        )
        with self.assertRaises(KeyError):
            runtime.rank(["query"], ["not-in-registry"])
        with self.assertRaises(worker.P12WorkerError):
            runtime.rank(["query"], list(_ids(51)))
        self.assertEqual(runtime.summary()["failure_count"], 1)


class _FakeFamily2Store:
    def __init__(self) -> None:
        self.fetches: list[tuple[tuple[int, str], ...]] = []
        self.subtype_calls: list[str] = []

    def resolve_query_subtypes(self, category_text: str) -> tuple[str, ...]:
        self.subtype_calls.append(category_text)
        return ("dress",)

    def fetch_top10(
        self, requested: list[tuple[int, str]], query_terms: list[str]
    ) -> object:
        self.assert_query = tuple(query_terms)
        self.fetches.append(tuple(requested))
        return SimpleNamespace(idf_by_term={"blue": 1.0})


def _family2_score() -> worker.CandidateScore:
    return worker.CandidateScore(
        total=0.5,
        relevance=0.5,
        tie_bonus=0.0,
        conflict_state="not_applicable",
        broad_rank_prior=0.5,
        strict_rank_prior=0.5,
        rrf_rank_prior=0.5,
        idf_any_field_coverage=0.5,
        title_category_coverage=0.5,
        features_details_coverage=0.5,
        description_store_coverage=0.5,
        latest_hard_clause_coverage=0.5,
        subtype_consistency=0.5,
        positive_constraint_evidence=0.5,
    )


class Family2C50ScoringTests(unittest.TestCase):
    def test_scores_c50_in_bounded_chunks_with_one_frozen_context(self) -> None:
        identifiers = _ids(23)
        store = _FakeFamily2Store()
        agent = object.__new__(worker.P12CaptureAgent)
        agent._p12_family2_store = store
        agent._p12_family2_counts = Counter()
        state = SimpleNamespace(
            category_text="dress",
            messages=["A key requirement is blue."],
            version=1,
        )

        def scored(chunk: tuple[str, ...], _batch: object, **kwargs: object) -> object:
            self.assertEqual(len(kwargs["broad_ranks"]), len(identifiers))
            self.assertEqual(kwargs["hard_clause_terms"], ("blue", "formal", "event", "dress"))
            return SimpleNamespace(
                fallback=False,
                breakdowns={identifier: _family2_score() for identifier in chunk},
            )

        with (
            patch.object(
                worker,
                "_positive_constraints",
                return_value=(SimpleNamespace(slot="color"),),
            ),
            patch.object(
                worker,
                "_latest_hard_clause_terms",
                return_value=("blue", "formal", "event", "dress"),
            ),
            patch.object(
                worker,
                "rerank_top10_preserving_membership",
                side_effect=scored,
            ),
        ):
            scores, hard_terms, preference_count = agent._p12_c50_scores(
                state,
                identifiers,
                {identifier: index for index, identifier in enumerate(identifiers, 1)},
                {name: list(identifiers) for name in ("broad", "strict", "fused")},
                ["blue"],
                (),
            )

        self.assertEqual(set(scores), set(identifiers))
        self.assertEqual(hard_terms, ("blue", "formal", "event", "dress"))
        self.assertEqual(preference_count, 1)
        self.assertEqual([len(chunk) for chunk in store.fetches], [10, 10, 3])
        self.assertEqual(agent._p12_family2_counts["feature_fetch_calls"], 3)
        self.assertEqual(agent._p12_family2_counts["sidecar_rows_read"], 23)
        self.assertEqual(agent._p12_family2_counts["maximum_rows_per_fetch"], 10)
        self.assertEqual(
            agent._p12_family2_counts["turns_with_complete_c50_scores"], 1
        )

    def test_one_chunk_failure_discards_all_partial_scores(self) -> None:
        identifiers = _ids(11)
        store = _FakeFamily2Store()
        agent = object.__new__(worker.P12CaptureAgent)
        agent._p12_family2_store = store
        agent._p12_family2_counts = Counter()
        state = SimpleNamespace(category_text="shoes", messages=["visible"], version=1)
        calls = 0

        def scored(chunk: tuple[str, ...], _batch: object, **_kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                fallback=calls == 2,
                breakdowns={identifier: _family2_score() for identifier in chunk},
            )

        with (
            patch.object(worker, "_positive_constraints", return_value=()),
            patch.object(worker, "_latest_hard_clause_terms", return_value=()),
            patch.object(
                worker,
                "rerank_top10_preserving_membership",
                side_effect=scored,
            ),
        ):
            scores, _, _ = agent._p12_c50_scores(
                state,
                identifiers,
                {identifier: index for index, identifier in enumerate(identifiers, 1)},
                {name: list(identifiers) for name in ("broad", "strict", "fused")},
                ["shoe"],
                (),
            )

        self.assertEqual(scores, {})
        self.assertEqual(agent._p12_family2_counts["score_fail_closed_turns"], 1)
        self.assertEqual(agent._p12_family2_counts["feature_fetch_calls"], 2)
        self.assertEqual(agent._p12_family2_counts["sidecar_rows_read"], 11)
        self.assertEqual(agent._p12_family2_counts["maximum_rows_per_fetch"], 10)
        self.assertEqual(
            agent._p12_family2_counts["turns_with_complete_c50_scores"], 0
        )

    def test_visible_state_helper_failure_is_an_exact_family2_fallback(self) -> None:
        identifiers = _ids(11)
        agent = object.__new__(worker.P12CaptureAgent)
        agent._p12_family2_store = _FakeFamily2Store()
        agent._p12_family2_counts = Counter()
        state = SimpleNamespace(category_text="shoes", messages=["visible"], version=1)

        with patch.object(
            worker, "_positive_constraints", side_effect=ValueError("bad state")
        ):
            scores, hard_terms, preference_count = agent._p12_c50_scores(
                state,
                identifiers,
                {identifier: index for index, identifier in enumerate(identifiers, 1)},
                {name: list(identifiers) for name in ("broad", "strict", "fused")},
                ["shoe"],
                (),
            )

        self.assertEqual(scores, {})
        self.assertEqual(hard_terms, ())
        self.assertEqual(preference_count, 0)
        self.assertEqual(agent._p12_family2_counts["score_fail_closed_turns"], 1)
        self.assertEqual(agent._p12_family2_counts["feature_fetch_calls"], 0)


class Family3WorkerHookTests(unittest.TestCase):
    @staticmethod
    def _capture_agent() -> worker.P12CaptureAgent:
        agent = object.__new__(worker.P12CaptureAgent)
        agent._p12_family3_counts = Counter(
            {"total_turns": 0, "compute_failure_count": 0}
        )
        agent._p12_family3_reason_counts = {
            action: Counter(
                {reason: 0 for reason in worker.FAMILY3_DECISION_REASONS}
            )
            for action in (
                p12_actions.VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10,
                p12_actions.DUAL_BOUNDARY_CONSENSUS_SLOT10,
                p12_actions.RECENT_OVERRIDE_RANK_FUSION_SLOT10,
            )
        }
        agent._p12_family3_failure_counts = Counter(
            {action: 0 for action in agent._p12_family3_reason_counts}
        )
        return agent

    def test_reason_funnel_activation_and_per_action_exception_fallback(self) -> None:
        agent = self._capture_agent()
        baseline = _ids(50)
        activated = list(baseline)
        activated[9], activated[11] = activated[11], activated[9]
        noop = SimpleNamespace(identifiers=baseline, reason="rank_guard")
        active = SimpleNamespace(identifiers=tuple(activated), reason="activated")
        state = SimpleNamespace(messages=["first", "second"], version=2, version_anchor_turn=1)

        with (
            patch.object(
                p12_actions,
                "decide_visible_constraint_rank_fusion_slot10",
                return_value=noop,
            ),
            patch.object(
                p12_actions,
                "decide_dual_boundary_consensus_slot10",
                return_value=active,
            ),
            patch.object(
                p12_actions,
                "decide_recent_override_rank_fusion_slot10",
                side_effect=RuntimeError("unexpected"),
            ),
        ):
            ranked = agent._p12_family3_rankings(
                state,
                baseline,
                baseline,
                baseline,
                {},
                1,
                (),
            )

        self.assertEqual(
            ranked[p12_actions.VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10], baseline
        )
        self.assertEqual(
            ranked[p12_actions.DUAL_BOUNDARY_CONSENSUS_SLOT10], tuple(activated)
        )
        self.assertEqual(
            ranked[p12_actions.RECENT_OVERRIDE_RANK_FUSION_SLOT10], baseline
        )
        self.assertEqual(agent._p12_family3_counts["total_turns"], 1)
        self.assertEqual(agent._p12_family3_counts["compute_failure_count"], 1)
        self.assertEqual(
            agent._p12_family3_reason_counts[
                p12_actions.VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10
            ]["rank_guard"],
            1,
        )
        self.assertEqual(
            agent._p12_family3_reason_counts[
                p12_actions.DUAL_BOUNDARY_CONSENSUS_SLOT10
            ]["activated"],
            1,
        )
        self.assertEqual(
            agent._p12_family3_failure_counts[
                p12_actions.RECENT_OVERRIDE_RANK_FUSION_SLOT10
            ],
            1,
        )
        funnel = agent.p12_family3_summary()
        self.assertEqual(
            funnel["eligibility_counts"][
                p12_actions.DUAL_BOUNDARY_CONSENSUS_SLOT10
            ]["relevance_guard_passed"],
            1,
        )
        self.assertNotIn("item-", json.dumps(funnel, sort_keys=True))
        self.assertIn("privacy", funnel)

    def test_compose_rejects_malformed_family3_single_slot_capture(self) -> None:
        p11_agent = _FakeP11Agent([])
        p11_agent.reset("conversation_1", dict(SAFE_PROFILE))
        p11_agent.respond("conversation_1", "visible", 1, 10)
        p11_capture = p11_agent.take_last_capture("conversation_1")
        p5_agent = _FakeP5Agent([])
        p5_agent.reset("conversation_1", dict(SAFE_PROFILE))
        p5_agent.respond("conversation_1", "visible", 1, 10)
        p5_capture = p5_agent.take_last_capture("conversation_1")
        malformed = list(
            p11_capture["visible_constraint_rank_fusion_slot10_full"]
        )
        malformed[8], malformed[14] = malformed[14], malformed[8]
        p11_capture["visible_constraint_rank_fusion_slot10_full"] = tuple(malformed)

        with self.assertRaisesRegex(worker.P12WorkerError, "single-slot guard"):
            worker._compose_trace_record(1, 1, p11_capture, p5_capture)


class ResourceMeasurementTests(unittest.TestCase):
    def test_peak_rss_backend_returns_a_positive_measurement(self) -> None:
        peak_rss, backend = worker._peak_rss_bytes()
        self.assertIsInstance(peak_rss, int)
        self.assertGreater(peak_rss, 0)
        self.assertNotEqual(backend, "unavailable")


class RuntimeTraceTests(unittest.TestCase):
    @staticmethod
    def _runtime(path: Path, close_log: list[str]) -> worker.P12ActionRuntime:
        return worker.P12ActionRuntime(
            _FakeP11Agent(close_log),
            _FakeP5Agent(close_log),
            _FakeSemantic(close_log),
            path,
            asset_validation={"fixture": True},
        )

    def test_ten_turn_trace_stays_private_until_ordered_close_and_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / "blind-actions.jsonl"
            close_log: list[str] = []
            runtime = self._runtime(trace_path, close_log)
            self.assertIsNone(
                runtime.handle(
                    _request(
                        {
                            "operation": "reset",
                            "request_id": 1,
                            "ordinal": 1,
                            "user_profile": dict(SAFE_PROFILE),
                        }
                    )
                )
            )
            for turn in range(1, 11):
                value = runtime.handle(
                    _request(
                        {
                            "operation": "respond",
                            "request_id": turn + 1,
                            "ordinal": 1,
                            "user_message": f"visible message {turn}",
                            "turn": turn,
                            "top_k": 10,
                        }
                    )
                )
                self.assertEqual(value, {"ask_attribute": "color"})
                self.assertFalse(trace_path.exists())
            self.assertEqual(runtime.p11_agent.messages, runtime.p5_agent.messages)
            self.assertIsNone(
                runtime.handle(
                    _request(
                        {
                            "operation": "drop",
                            "request_id": 12,
                            "ordinal": 1,
                        }
                    )
                )
            )
            self.assertFalse(trace_path.exists())

            original_publish = worker._write_trace_exclusive

            def observed_publish(path: Path, lines: list[bytes]) -> None:
                self.assertTrue(runtime.p11_agent.closed)
                self.assertTrue(runtime.p5_agent.closed)
                self.assertTrue(runtime.semantic.closed)
                close_log.append("trace")
                original_publish(path, lines)

            with patch.object(
                worker, "_write_trace_exclusive", side_effect=observed_publish
            ):
                receipt = runtime.handle(
                    _request({"operation": "finalize", "request_id": 13})
                )

            self.assertEqual(close_log, ["p11", "p5", "semantic", "trace"])
            trace_bytes = trace_path.read_bytes()
            self.assertEqual(
                receipt["trace_sha256"], hashlib.sha256(trace_bytes).hexdigest()
            )
            self.assertEqual(receipt["record_count"], 10)
            records = [json.loads(line) for line in trace_bytes.splitlines()]
            self.assertEqual(len(records), 10)
            for turn, record in enumerate(records, start=1):
                self.assertEqual(set(record), {"ordinal", "turn", "actions", "candidate_pools"})
                self.assertEqual(record["ordinal"], 1)
                self.assertEqual(record["turn"], turn)
                self.assertEqual(set(record["actions"]), set(p12_actions.ACTION_IDS))
                self.assertEqual(set(record["candidate_pools"]), {"c20", "c50", "c100"})
                self.assertEqual(len(record["candidate_pools"]["c20"]), 20)
                self.assertEqual(len(record["candidate_pools"]["c50"]), 50)
                self.assertEqual(len(record["candidate_pools"]["c100"]), 100)
                self.assertEqual(
                    record["actions"][p12_actions.KEEP_R08],
                    record["candidate_pools"]["c20"][:10],
                )
                self.assertEqual(
                    record["actions"][p12_actions.ASK],
                    record["actions"][p12_actions.KEEP_P11],
                )
                self.assertNotEqual(
                    record["actions"][p12_actions.RESULT_AWARE_REWRITE_RETRIEVE],
                    record["actions"][p12_actions.KEEP_R08],
                )
                self.assertNotEqual(
                    set(record["actions"][p12_actions.COMPACT_NEGATIVE_C50]),
                    set(record["actions"][p12_actions.KEEP_P11]),
                )
                self.assertEqual(
                    len(
                        set(record["actions"][p12_actions.GUARDED_COMPACT_SLOT10])
                        ^ set(record["actions"][p12_actions.KEEP_P11])
                    ),
                    2,
                )
                self.assertEqual(
                    record["actions"][
                        p12_actions.GUARDED_COMPACT_SLOT10_STRICT
                    ][9],
                    record["candidate_pools"]["c50"][10],
                )
                for action in (
                    p12_actions.P11_EVIDENCE_NOVEL_SLOT10,
                    p12_actions.HARD_CLAUSE_NOVEL_SLOT10,
                    p12_actions.TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10,
                    p12_actions.VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10,
                    p12_actions.DUAL_BOUNDARY_CONSENSUS_SLOT10,
                    p12_actions.RECENT_OVERRIDE_RANK_FUSION_SLOT10,
                ):
                    self.assertEqual(
                        len(
                            set(record["actions"][action])
                            ^ set(record["actions"][p12_actions.KEEP_P11])
                        ),
                        2,
                    )
                    added = set(record["actions"][action]) - set(
                        record["actions"][p12_actions.KEEP_P11]
                    )
                    self.assertTrue(
                        added.isdisjoint(
                            set(record["actions"][p12_actions.CANDIDATE_RERANK])
                            | set(
                                record["actions"][
                                    p12_actions.FROZEN_SEMANTIC_RERANK
                                ]
                            )
                        )
                    )

            summary = receipt["worker_summary"]
            self.assertEqual(summary["trajectory"]["fixed_turns"], 10)
            self.assertEqual(summary["trajectory"]["respond_count"], 10)
            self.assertEqual(summary["actions"]["result_aware_base"], "R08+P5.R01")
            self.assertEqual(summary["actions"]["result_aware_computation_count"], 10)
            self.assertEqual(
                summary["actions"]["activation_definition"],
                "per-turn Top10 member set differs from KEEP_P11",
            )
            self.assertEqual(
                summary["actions"]["membership_activation_turn_counts"],
                {
                    p12_actions.COMPACT_NEGATIVE_C50: 10,
                    p12_actions.GUARDED_COMPACT_SLOT10: 10,
                    p12_actions.GUARDED_COMPACT_SLOT10_STRICT: 10,
                    p12_actions.P11_EVIDENCE_NOVEL_SLOT10: 10,
                    p12_actions.HARD_CLAUSE_NOVEL_SLOT10: 10,
                    p12_actions.TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10: 10,
                    p12_actions.VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10: 10,
                    p12_actions.DUAL_BOUNDARY_CONSENSUS_SLOT10: 10,
                    p12_actions.RECENT_OVERRIDE_RANK_FUSION_SLOT10: 10,
                },
            )
            self.assertEqual(
                summary["actions"]["membership_activation_session_counts"],
                {
                    p12_actions.COMPACT_NEGATIVE_C50: 1,
                    p12_actions.GUARDED_COMPACT_SLOT10: 1,
                    p12_actions.GUARDED_COMPACT_SLOT10_STRICT: 1,
                    p12_actions.P11_EVIDENCE_NOVEL_SLOT10: 1,
                    p12_actions.HARD_CLAUSE_NOVEL_SLOT10: 1,
                    p12_actions.TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10: 1,
                    p12_actions.VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10: 1,
                    p12_actions.DUAL_BOUNDARY_CONSENSUS_SLOT10: 1,
                    p12_actions.RECENT_OVERRIDE_RANK_FUSION_SLOT10: 1,
                },
            )
            self.assertEqual(
                summary["actions"]["compact_funnel"]["counts"],
                {
                    "total_turns": 10,
                    "turns_with_executable_constraints": 10,
                },
            )
            self.assertEqual(
                summary["actions"]["family2_funnel"]["counts"],
                {
                    "total_turns": 10,
                    "turns_with_complete_c50_scores": 10,
                },
            )
            self.assertEqual(
                summary["actions"]["family3_funnel"]["counts"],
                {"total_turns": 10, "compute_failure_count": 0},
            )
            self.assertEqual(
                summary["actions"]["family3_funnel"]["reason_counts"][
                    p12_actions.VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10
                ]["activated"],
                10,
            )
            self.assertEqual(summary["p11"]["per_turn_invariants_verified"], 10)
            self.assertEqual(summary["full_catalog_search_calls"], 0)
            self.assertEqual(summary["semantic_failure_count"], 0)
            self.assertEqual(summary["rewrite_failure_count"], 0)
            self.assertEqual(summary["p11_invariant_failure_count"], 0)
            self.assertEqual(summary["family2_score_failure_count"], 0)
            self.assertEqual(summary["family3_compute_failure_count"], 0)
            self.assertTrue(summary["trace_written_after_components_closed"])

    def test_drop_and_finalize_before_ten_turns_publish_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / "must-not-exist.jsonl"
            close_log: list[str] = []
            runtime = self._runtime(trace_path, close_log)
            runtime.reset(1, dict(SAFE_PROFILE))
            with self.assertRaises(worker.P12WorkerError):
                runtime.drop(1)
            with self.assertRaises(worker.P12WorkerError):
                runtime.finalize()
            self.assertFalse(trace_path.exists())
            runtime.abort()
            self.assertEqual(close_log, ["p11", "p5", "semantic"])

    def test_exclusive_trace_publish_refuses_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / "existing.jsonl"
            trace_path.write_bytes(b"existing\n")
            with self.assertRaises(FileExistsError):
                worker._write_trace_exclusive(trace_path, [b"replacement\n"])
            self.assertEqual(trace_path.read_bytes(), b"existing\n")


if __name__ == "__main__":
    unittest.main()
