from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from starter.fusion_core import FusionCore, FusionCoreAgent, MODE_ACTIVE, MODE_OFF
from vendor.teammate_v1.err402.agents.v1 import AgentV1
from vendor.teammate_v1.err402.retrieval.catalog import (
    CandidatePool,
    CatalogIndex,
    ProductRecord,
)
from vendor.teammate_v1.err402.retrieval.scoring import ScoredProduct


ROOT = Path(__file__).resolve().parents[1]


def product(
    identifier: str,
    *,
    category: str = "dress",
    color: str | None = None,
    material: str | None = None,
    price: float | None = 20.0,
) -> ProductRecord:
    return ProductRecord(
        parent_asin=identifier,
        title=identifier,
        description="",
        category_path=("Clothing", category),
        coarse_category=category,
        category_terms=frozenset({category}),
        searchable_tokens=f" {category} ",
        average_rating=4.0,
        rating_number=10,
        price=price,
        price_is_lower_bound=False,
        department=None,
        material=material,
        color=color,
        style=None,
        size=None,
        brand=None,
        use_case=None,
        has_features=False,
    )


def record(
    slot: str,
    value: str,
    *,
    polarity: int = 1,
    hardness: str = "soft",
    source_turn: int = 1,
    version: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        slot=slot,
        value=value,
        polarity=polarity,
        hardness=hardness,
        source_turn=source_turn,
        version=version,
        status="active",
    )


class Ledger:
    def __init__(self, records: list[object]) -> None:
        self._records = tuple(records)

    def active_records(self) -> tuple[object, ...]:
        return self._records


def state(
    records: list[object] | None = None,
    *,
    category: str = "dress",
    version: int = 1,
    asked: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        category_text=category,
        version=version,
        slot_ledger=Ledger(records or []),
        asked_attributes=list(asked),
        exhausted_attributes=set(),
        profile={},
    )


class FakeCatalog:
    def __init__(self, products: list[ProductRecord]) -> None:
        self.products = {item.parent_asin: item for item in products}
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def candidates(self, category: str, query: str) -> CandidatePool:
        self.calls.append((category, query))
        identifiers = tuple(self.products)
        return CandidatePool(
            identifiers,
            {identifier: index for index, identifier in enumerate(identifiers, 1)},
        )

    @staticmethod
    def question_category(_category: str) -> str:
        return "clothing"

    def close(self) -> None:
        self.closed = True


class FakeScorer:
    def __init__(self, catalog: FakeCatalog, order: list[str] | None = None) -> None:
        self.catalog = catalog
        self.order = order or list(catalog.products)
        self.calls = 0

    def score(self, _pool: CandidatePool, _context: object) -> list[ScoredProduct]:
        self.calls += 1
        return [
            ScoredProduct(
                self.catalog.products[identifier],
                1.0 - index / 100.0,
                {"category": 1.0},
            )
            for index, identifier in enumerate(self.order)
        ]


class BrokenScorer(FakeScorer):
    def score(self, _pool: CandidatePool, _context: object) -> list[ScoredProduct]:
        raise RuntimeError("synthetic")


class FakeParent:
    def __init__(self, response: dict, snapshot: object) -> None:
        self.response = response
        self.snapshot = snapshot
        self.calls: list[tuple] = []

    def reset(self, session_id: str, profile: dict) -> None:
        self.calls.append(("reset", session_id, profile))

    def respond(self, session_id: str, message: str, turn: int, top_k: int) -> dict:
        self.calls.append(("respond", session_id, message, turn, top_k))
        return self.response

    def debug_snapshot(self, session_id: str) -> object:
        self.calls.append(("snapshot", session_id))
        return self.snapshot

    def close(self) -> None:
        self.calls.append(("close",))


def core(
    products: list[ProductRecord],
    *,
    scorer_type: type[FakeScorer] = FakeScorer,
) -> tuple[FusionCore, FakeCatalog]:
    catalog = FakeCatalog(products)
    scorer = scorer_type(catalog)
    return (
        FusionCore(mode=MODE_ACTIVE, catalog=catalog, scorer=scorer),
        catalog,
    )


class FusionCoreTests(unittest.TestCase):
    def test_default_off_is_exact_noop_without_catalog_initialization(self) -> None:
        adapter = FusionCore(mode=MODE_OFF)
        result = adapter.apply(
            "unused",
            object(),
            turn=1,
            top_k=2,
            fallback_order=("B", "A"),
        )
        self.assertEqual(result.identifiers, ("B", "A"))
        self.assertEqual(result.diagnostics["effective_mode"], "off")
        self.assertFalse(result.diagnostics["state_committed"])

    def test_vendored_catalog_keeps_fixed_fts_candidate_depth(self) -> None:
        self.assertEqual(CatalogIndex.LEXICAL_CANDIDATE_LIMIT, 1000)

    def test_uses_v1_candidate_and_product_scorer_order(self) -> None:
        adapter, catalog = core([product("P0"), product("P1"), product("P2")])
        adapter.reset("s")
        result = adapter.apply(
            "s",
            state([record("material", "cotton")]),
            turn=1,
            top_k=2,
            fallback_order=("F0", "F1"),
        )
        self.assertEqual(result.identifiers, ("P0", "P1", "P2"))
        self.assertEqual(catalog.calls, [("dress", "dress cotton")])
        self.assertFalse(result.diagnostics["fallback"])

    def test_unknown_is_neutral_and_explicit_negative_violation_is_last(self) -> None:
        adapter, _catalog = core([
            product("RED", color="red"),
            product("UNKNOWN", color=None),
            product("BLUE", color="blue"),
        ])
        adapter.reset("s")
        result = adapter.apply(
            "s",
            state([record("color", "red", polarity=-1, hardness="hard")]),
            turn=1,
            top_k=2,
            fallback_order=("F0", "F1"),
        )
        self.assertEqual(result.identifiers, ("UNKNOWN", "BLUE", "RED"))
        self.assertEqual(result.diagnostics["hard_conflict_count"], 1)
        self.assertTrue(result.diagnostics["unknown_is_neutral"])

    def test_unknown_positive_evidence_is_not_a_conflict(self) -> None:
        adapter, _catalog = core([
            product("RED", color="red"),
            product("UNKNOWN", color=None),
            product("BLUE", color="blue"),
        ])
        adapter.reset("s")
        result = adapter.apply(
            "s",
            state([record("color", "blue", hardness="hard")]),
            turn=1,
            top_k=2,
            fallback_order=("F0", "F1"),
        )
        self.assertEqual(result.identifiers, ("UNKNOWN", "BLUE", "RED"))
        self.assertEqual(result.diagnostics["hard_conflict_count"], 1)

    def test_immediate_no_repeat_with_stable_seen_fallback(self) -> None:
        adapter, _catalog = core([product(f"P{index}") for index in range(4)])
        adapter.reset("s")
        first = adapter.apply(
            "s", state(), turn=1, top_k=2, fallback_order=("F0", "F1")
        )
        second = adapter.apply(
            "s", state(), turn=2, top_k=2, fallback_order=("F0", "F1")
        )
        third = adapter.apply(
            "s", state(), turn=3, top_k=2, fallback_order=("F0", "F1")
        )
        self.assertEqual(first.identifiers[:2], ("P0", "P1"))
        self.assertEqual(second.identifiers[:2], ("P2", "P3"))
        self.assertEqual(third.identifiers[:2], ("P0", "P1"))
        self.assertTrue(second.diagnostics["immediate_no_repeat"])

    def test_selective_override_resets_served_but_keeps_independent_slot(self) -> None:
        adapter, _catalog = core([product("P0"), product("P1"), product("P2")])
        adapter.reset("s")
        first_state = state([
            record("material", "cotton"),
            record("color", "blue"),
        ])
        adapter.apply(
            "s", first_state, turn=1, top_k=1, fallback_order=("F0",)
        )
        second_state = state(
            [
                record("material", "linen", version=2, source_turn=2),
                record("color", "blue"),
            ],
            version=2,
        )
        result = adapter.apply(
            "s", second_state, turn=2, top_k=1, fallback_order=("F0",)
        )
        self.assertEqual(result.identifiers[0], "P0")
        self.assertFalse(result.diagnostics["category_reset"])
        self.assertTrue(result.diagnostics["intent_version_reset"])
        self.assertEqual(result.diagnostics["selective_removed_constraints"], 1)
        self.assertEqual(result.diagnostics["selective_added_constraints"], 1)
        keys = adapter.debug_memory("s")["constraint_keys"]
        self.assertIn(("color", "blue", 1), keys)

    def test_agent_off_returns_exact_parent_object_without_snapshot(self) -> None:
        response = {"recommendations": [], "message": "raw", "ask_attribute": "other"}
        parent = FakeParent(response, state())
        agent = FusionCoreAgent(mode=MODE_OFF, parent_agent=parent)
        agent.reset("s", {"locale": "en"})
        actual = agent.respond("s", "hello", 1, 10)
        self.assertIs(actual, response)
        self.assertFalse(any(call[0] == "snapshot" for call in parent.calls))

    def test_agent_active_advances_parent_and_changes_only_allowed_fields(self) -> None:
        response = {
            "recommendations": [{"parent_asin": "F0", "score": 0.2}],
            "message": "parent",
            "ask_attribute": "other",
            "usage": {"prompt_tokens": 7},
            "opaque": "preserve",
        }
        parent = FakeParent(response, state())
        adapter, _catalog = core([product("P0"), product("P1")])
        agent = FusionCoreAgent(
            mode=MODE_ACTIVE, parent_agent=parent, fusion_core=adapter
        )
        agent.reset("s", {})
        actual = agent.respond("s", "hello", 10, 1)
        self.assertEqual([call[0] for call in parent.calls[-2:]], ["respond", "snapshot"])
        self.assertEqual(actual["recommendations"][0]["parent_asin"], "P0")
        self.assertIsNone(actual["ask_attribute"])
        self.assertEqual(actual["usage"], response["usage"])
        self.assertEqual(actual["opaque"], "preserve")

    def test_category_change_resets_no_repeat_memory(self) -> None:
        adapter, _catalog = core([product("P0"), product("P1")])
        adapter.reset("s")
        adapter.apply("s", state(), turn=1, top_k=1, fallback_order=("F0",))
        result = adapter.apply(
            "s",
            state(category="shoe", version=2),
            turn=2,
            top_k=1,
            fallback_order=("F0",),
        )
        self.assertEqual(result.identifiers[0], "P0")
        self.assertTrue(result.diagnostics["category_reset"])

    def test_ask_other_is_forbidden_and_failure_does_not_commit(self) -> None:
        adapter, _catalog = core([product("P0"), product("P1")])
        adapter.reset("s")
        with patch.object(AgentV1, "_select_question", return_value="other"):
            rejected = adapter.apply(
                "s", state(), turn=1, top_k=1, fallback_order=("F0", "F1")
            )
        self.assertEqual(rejected.identifiers, ("F0", "F1"))
        self.assertFalse(rejected.diagnostics["state_committed"])
        self.assertEqual(adapter.debug_memory("s")["last_turn"], 0)

        accepted = adapter.apply(
            "s", state(), turn=1, top_k=1, fallback_order=("F0", "F1")
        )
        self.assertEqual(accepted.identifiers[0], "P0")
        self.assertTrue(accepted.diagnostics["ask_other_forbidden"])

    def test_backend_exception_is_exact_fallback_and_atomic(self) -> None:
        adapter, _catalog = core(
            [product("P0"), product("P1")], scorer_type=BrokenScorer
        )
        adapter.reset("s")
        result = adapter.apply(
            "s", state(), turn=1, top_k=1, fallback_order=("B", "A")
        )
        self.assertEqual(result.identifiers, ("B", "A"))
        self.assertEqual(result.diagnostics["reason_code"], "adapter_failure")
        self.assertEqual(adapter.debug_memory("s")["last_turn"], 0)

    def test_non_monotonic_turn_fails_closed_without_mutation(self) -> None:
        adapter, _catalog = core([product("P0"), product("P1")])
        adapter.reset("s")
        adapter.apply("s", state(), turn=1, top_k=1, fallback_order=("F0",))
        before = adapter.debug_memory("s")
        result = adapter.apply(
            "s", state(), turn=1, top_k=1, fallback_order=("F0",)
        )
        self.assertEqual(result.identifiers, ("F0",))
        self.assertEqual(adapter.debug_memory("s"), before)

    def test_vendor_manifest_matches_exact_bytes(self) -> None:
        manifest_path = ROOT / "vendor" / "teammate_v1" / "VENDOR_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["source_commit"],
            "5df5d51e7578e80616f45fcaa89ec977347845fa",
        )
        for relative, identity in manifest["files"].items():
            raw = (manifest_path.parent / relative).read_bytes()
            self.assertEqual(len(raw), identity["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), identity["sha256"])


if __name__ == "__main__":
    unittest.main()
