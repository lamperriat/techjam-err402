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
