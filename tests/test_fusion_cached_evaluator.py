from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import fusion_cached_evaluator as ev


MISS = [{
    "hit": False,
    "first_hit_turn": None,
    "best_rank": None,
    "reciprocal_rank": 0.0,
}]
HIT = [{
    "hit": True,
    "first_hit_turn": 2,
    "best_rank": 3,
    "reciprocal_rank": 1.0 / 3,
}]


class FusionCachedEvaluatorTests(unittest.TestCase):
    def test_public_single_cli_writes_compact_repeat_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "public.jsonl"
            dataset.write_text(
                '\n'.join([
                    json.dumps({"scenario_type": "buying", "ground_truth": {"parent_asin": "A"}}),
                    json.dumps({"scenario_type": "browse", "ground_truth": {"parent_asin": "B"}}),
                ]) + "\n",
                encoding="utf-8",
            )
            output = root / "summary.json"
            ledger = [HIT[0], MISS[0]]
            diagnostics = {"schema_version": "synthetic.v1", "turns": 2}
            factory = object()
            with (
                mock.patch.object(ev, "_factory", return_value=factory),
                mock.patch.object(
                    ev,
                    "catalog_index",
                    return_value=({"A", "B"}, {"A": ["Shoes"], "B": ["Jewelry"]}, {}),
                ),
                mock.patch.object(
                    ev,
                    "_parallel_official_repeat",
                    return_value=(ledger, ev._sha(ledger), diagnostics),
                ) as repeat,
            ):
                code = ev.main([
                    "public-single", "--agent-factory", "fake:Agent",
                    "--flags", '{"mode":"active"}', "--dataset", str(dataset),
                    "--catalog", str(root / "catalog"), "--output", str(output),
                ])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(payload["session_count"], 2)
            self.assertEqual(payload["ledger_sha256"], ev._sha(ledger))
            self.assertEqual(payload["runtime_diagnostics"], diagnostics)
            self.assertIn("first_turn_hr", payload["breakdowns"])
            self.assertIn("scenario", payload["breakdowns"])
            self.assertIn("taxonomy", payload["breakdowns"])
            self.assertNotIn("sessions", payload)
            self.assertEqual(repeat.call_args.args[0], "fake:Agent")
            self.assertEqual(repeat.call_args.args[1], {"mode": "active"})
            self.assertEqual(repeat.call_args.args[4], 4)

    def test_parallel_helpers_preserve_order_and_merge_core_diagnostics(self) -> None:
        samples = [{"sample_id": str(index)} for index in range(5)]
        self.assertEqual(
            [(start, len(rows)) for start, rows in ev._sample_chunks(samples, 2)],
            [(0, 3), (3, 2)],
        )
        diagnostics = [{
            "schema_version": "fusion-core-evaluation-diagnostics.v1",
            "turns": 2,
            "fusion_active_turns": 2,
            "fallback_turns": 0,
            "same_version_repeat_slots": 0,
            "hard_conflicts": 1,
            "parent_route_added": 2,
            "candidate_count": 20,
            "sessions": 1,
            "mean_distinct_served": 10.0,
            "mean_candidate_count": 10.0,
        }, {
            "schema_version": "fusion-core-evaluation-diagnostics.v1",
            "turns": 4,
            "fusion_active_turns": 3,
            "fallback_turns": 1,
            "same_version_repeat_slots": 0,
            "hard_conflicts": 2,
            "parent_route_added": 3,
            "candidate_count": 32,
            "sessions": 2,
            "mean_distinct_served": 8.0,
            "mean_candidate_count": 8.0,
        }]
        merged = ev._merge_diagnostics(diagnostics)
        self.assertEqual(merged["turns"], 6)
        self.assertEqual(merged["sessions"], 3)
        self.assertEqual(merged["hard_conflicts"], 3)
        self.assertEqual(merged["mean_distinct_served"], 8.666667)
        self.assertEqual(merged["mean_candidate_count"], 8.666667)

    def test_parallel_repeat_runs_two_replicas_with_fixed_process_budget(self) -> None:
        class ImmediateFuture:
            def __init__(self, value: object) -> None:
                self.value = value

            def result(self) -> object:
                return self.value

        budgets: list[int] = []

        class ImmediatePool:
            def __init__(self, max_workers: int) -> None:
                budgets.append(max_workers)

            def __enter__(self) -> "ImmediatePool":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def submit(self, fn: object, *args: object) -> ImmediateFuture:
                return ImmediateFuture(fn(*args))

        def worker(
            _factory_spec: str,
            _flags: object,
            _catalog_path: str,
            start: int,
            rows: list[dict],
        ) -> tuple[int, list[dict[str, object]], dict[str, object]]:
            ledger = [dict(HIT[0]) for _ in rows]
            diagnostics = {
                "schema_version": "fusion-core-evaluation-diagnostics.v1",
                "turns": len(rows),
                "fusion_active_turns": len(rows),
                "fallback_turns": 0,
                "same_version_repeat_slots": 0,
                "hard_conflicts": 0,
                "parent_route_added": 0,
                "candidate_count": 10 * len(rows),
                "sessions": len(rows),
                "mean_distinct_served": 10.0,
                "mean_candidate_count": 10.0,
            }
            return start, ledger, diagnostics

        samples = [{"sample_id": str(index)} for index in range(4)]
        with (
            mock.patch.object(ev, "ProcessPoolExecutor", ImmediatePool),
            mock.patch.object(ev, "_parallel_chunk_worker", side_effect=worker),
        ):
            ledger, digest, diagnostics = ev._parallel_official_repeat(
                "fake:Agent", {}, samples, Path("catalog"), 4
            )
        self.assertEqual(budgets, [4])
        self.assertEqual(len(ledger), 4)
        self.assertEqual(digest, ev._sha(ledger))
        self.assertEqual(diagnostics["sessions"], 4)

    def test_official_repeat_reuses_samples_and_closes_agents(self) -> None:
        seen: list[int] = []
        agents: list[object] = []

        class Agent:
            closed = False

            def evaluation_diagnostics(self) -> dict[str, object]:
                return {"schema_version": "synthetic.v1", "turns": 1}

            def close(self) -> None:
                self.closed = True

        def factory() -> Agent:
            agent = Agent()
            agents.append(agent)
            return agent

        def fake_evaluate(agent: object, samples: list[dict], *_: object) -> dict:
            seen.append(id(samples))
            return {"sessions": HIT}

        samples = [{"sample_id": "s"}]
        with mock.patch.object(ev, "evaluate", side_effect=fake_evaluate):
            ledger, digest, diagnostics = ev._official_repeat(
                factory, {}, samples, {"x"}, {}, {}
            )
        self.assertEqual(ledger, HIT)
        self.assertEqual(digest, ev._sha(HIT))
        self.assertEqual(diagnostics["schema_version"], "synthetic.v1")
        self.assertEqual(diagnostics["turns"], 1)
        self.assertEqual(len(set(seen)), 1)
        self.assertTrue(all(getattr(agent, "closed") for agent in agents))

    def test_single_candidate_writes_identifier_free_hashed_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proxy = root / "proxy"
            proxy.write_text('{"sample_id":"s"}\n', encoding="utf-8")
            labels = root / "labels.npz"
            np.savez(labels, eligible_from=[1], outer_fold=[0])
            with (
                mock.patch.object(ev, "catalog_index", return_value=({"x"}, {}, {})),
                mock.patch.object(
                    ev,
                    "_official_repeat",
                    return_value=(HIT, ev._sha(HIT), {"schema_version": "synthetic.v1"}),
                ),
                mock.patch.object(
                    ev,
                    "_cached_v212_ledger",
                    return_value=(MISS, np.asarray([0], dtype=np.int16)),
                ),
            ):
                ledger = root / "candidate_ledger.json"
                result = ev.attach_candidate_once(
                    factory=lambda: object(),
                    flags={},
                    proxy_path=proxy,
                    labels_path=labels,
                    catalog_path=root / "catalog",
                    claim_path=root / "claim",
                    ledger_output=ledger,
                    source_root=root / "source",
                    projection_root=root / "projection",
                )
            self.assertEqual(result["status"], "VALID")
            self.assertEqual(result["transitions"]["miss_to_hit"], 1)
            self.assertEqual(
                result["runtime_diagnostics"]["schema_version"], "synthetic.v1"
            )
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "fusion_session_ledger_v1")
            self.assertNotIn("sample_id", ledger.read_text(encoding="utf-8"))
            self.assertEqual(payload["sha256"], ev._sha(HIT))

    def test_post_claim_failure_is_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_proxy = root / "proxy"
            bad_proxy.write_text("not-json\n", encoding="utf-8")
            result = ev.attach_candidate_once(
                factory=lambda: object(),
                flags={},
                proxy_path=bad_proxy,
                labels_path=root / "labels",
                catalog_path=root / "catalog",
                claim_path=root / "claim",
                ledger_output=root / "ledger",
                source_root=root / "source",
                projection_root=root / "projection",
            )
            self.assertEqual(result["status"], "INVALID_ONE_SHOT_CONSUMED")


if __name__ == "__main__":
    unittest.main()
