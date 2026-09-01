from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from observer.fusion_demo import FusionDemo


@dataclass(frozen=True)
class FakeConstraint:
    text: str
    hard: bool = True
    source: str = "visible"


@dataclass
class FakeState:
    intent: str = "buying"
    category: str = "running shoes"
    constraints: list[FakeConstraint] = field(
        default_factory=lambda: [FakeConstraint("black")]
    )
    asked_attributes: set[str] = field(default_factory=set)
    shown_product_ids: set[str] = field(default_factory=set)


class FakeCatalog:
    def candidates(self, category: str, query_text: str) -> Any:
        del category, query_text
        return SimpleNamespace(
            parent_asins=("P1", "P2", "P3"),
            lexical_ranks={"P1": 1, "P2": 2},
        )


class FakeT0:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.catalog = FakeCatalog()
        self._sessions: dict[str, FakeState] = {}
        self.rows = rows or [
            {"parent_asin": "P1", "score": 2.0},
            {"parent_asin": "P2", "score": 1.0},
        ]
        self.closed = False

    def reset(self, session_id: str, profile: dict[str, Any]) -> None:
        del profile
        self._sessions[session_id] = FakeState()

    def respond(
        self, session_id: str, message: str, turn: int, top_k: int
    ) -> dict[str, Any]:
        del message, turn
        state = self._sessions[session_id]
        identifiers = [
            str(row.get("parent_asin"))
            for row in self.rows[:top_k]
            if isinstance(row, dict) and row.get("parent_asin")
        ]
        state.shown_product_ids.update(identifiers)
        state.asked_attributes.add("material")
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": list(self.rows),
        }

    def close(self) -> None:
        self.closed = True


class FakeFusionA:
    def __init__(self, *, fallback_page: int | None = None) -> None:
        self.teammate = FakeT0()
        self._pages: dict[str, int] = {}
        self._served: dict[str, set[str]] = {}
        self._rank_fallbacks = 0
        self._fallback_page = fallback_page
        self._last_fallback = False
        self.closed = False

    def reset(self, session_id: str, profile: dict[str, Any]) -> None:
        self.teammate.reset(session_id, profile)
        self._pages[session_id] = 0
        self._served[session_id] = set()

    def respond(
        self, session_id: str, message: str, turn: int, top_k: int
    ) -> dict[str, Any]:
        response = self.teammate.respond(session_id, message, turn, top_k)
        self._pages[session_id] += 1
        page = self._pages[session_id]
        self._last_fallback = page == self._fallback_page
        if self._last_fallback:
            self._rank_fallbacks += 1
        elif page > 2:
            response = {
                **response,
                "recommendations": [{"parent_asin": "P3"}],
            }
        self._served[session_id].update(
            row["parent_asin"] for row in response["recommendations"]
        )
        return response

    def _rank_order(self, session_id: str) -> tuple[str, ...]:
        del session_id
        if self._last_fallback:
            raise RuntimeError("synthetic expert failure")
        return ("P3", "P2", "P1")

    def evaluation_diagnostics(self) -> dict[str, Any]:
        return {"rank_fallbacks": self._rank_fallbacks}

    def close(self) -> None:
        self.closed = True
        self.teammate.close()


class FakeFusionB:
    def __init__(self) -> None:
        self.base = FakeFusionA()
        self._other: dict[str, dict[str, Any]] = {}
        self.closed = False

    def reset(self, session_id: str, profile: dict[str, Any]) -> None:
        self.base.reset(session_id, profile)
        self._other[session_id] = {
            "schema_version": "fake-other.v1",
            "asks": 0,
            "pending": False,
        }

    def respond(
        self, session_id: str, message: str, turn: int, top_k: int
    ) -> dict[str, Any]:
        response = self.base.respond(session_id, message, turn, top_k)
        self._other[session_id].update(asks=1, pending=True)
        return {
            **response,
            "message": "What other requirement matters most?",
            "ask_attribute": "other",
        }

    def debug_other(self, session_id: str) -> dict[str, Any]:
        return dict(self._other[session_id])

    def close(self) -> None:
        self.closed = True
        self.base.close()


class FusionDemoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.catalog_path = Path(self.temporary.name) / "catalog.jsonl"
        self.catalog_path.write_text("{}\n", encoding="utf-8")
        self.products = {
            identifier: {
                "parent_asin": identifier,
                "title": f"Product {identifier}",
                "price": rank * 10.0,
                "categories": ["Clothing", "Shoes"],
            }
            for rank, identifier in enumerate(("P1", "P2", "P3"), start=1)
        }
        self.created: dict[str, list[Any]] = {"t0": [], "a": [], "b": []}

        def factory(variant: str, constructor: Any) -> Any:
            def create(path: Path) -> Any:
                self.assertEqual(path, self.catalog_path)
                agent = constructor()
                self.created[variant].append(agent)
                return agent

            return create

        self.factories = {
            "t0": factory("t0", FakeT0),
            "a": factory("a", FakeFusionA),
            "b": factory("b", FakeFusionB),
        }

    def demo(self, factories: dict[str, Any] | None = None) -> FusionDemo:
        value = FusionDemo(
            self.catalog_path,
            self.products,
            factories=factories or self.factories,
        )
        self.addCleanup(value.close)
        return value

    @staticmethod
    def route(payload: dict[str, Any]) -> str:
        return next(
            event["value"]
            for event in payload["events"]
            if event["layer"] == "Page router"
        )

    def test_t0_a_and_b_reset_and_respond_use_selected_factory(self) -> None:
        demo = self.demo()
        expectations = {
            "t0": ("material", None, None),
            "a": ("material", 1, None),
            "b": ("other", 1, 1),
        }
        for variant, (attribute, page, other_asks) in expectations.items():
            with self.subTest(variant=variant):
                reset = demo.reset(variant, {"summary": variant})
                result = demo.respond(reset["session_id"], "running shoes")
                self.assertEqual(result["variant"], variant)
                self.assertEqual(result["turn"], 1)
                self.assertEqual(result["response"]["ask_attribute"], attribute)
                self.assertEqual(result["state"]["fusion_page"], page)
                if other_asks is None:
                    self.assertIsNone(result["state"]["other"])
                else:
                    self.assertEqual(result["state"]["other"]["asks"], other_asks)

    def test_switch_closes_old_agent_and_expires_old_session(self) -> None:
        demo = self.demo()
        old_reset = demo.reset("t0")
        old_agent = self.created["t0"][0]
        new_reset = demo.reset("a")

        self.assertTrue(old_agent.closed)
        self.assertFalse(new_reset["reused_index"])
        with self.assertRaises(KeyError):
            demo.respond(old_reset["session_id"], "stale request")
        self.assertEqual(
            demo.respond(new_reset["session_id"], "fresh request")["variant"],
            "a",
        )

    def test_same_variant_reuses_index_but_expires_previous_session(self) -> None:
        demo = self.demo()
        first = demo.reset("a")
        agent = self.created["a"][0]
        second = demo.reset("a")

        self.assertTrue(second["reused_index"])
        self.assertIs(agent, self.created["a"][0])
        self.assertFalse(agent.closed)
        with self.assertRaises(KeyError):
            demo.respond(first["session_id"], "expired")

    def test_invalid_session_and_ten_turn_limit(self) -> None:
        demo = self.demo()
        reset = demo.reset("t0")
        with self.assertRaises(KeyError):
            demo.respond("not-the-active-session", "request")
        for turn in range(1, 11):
            self.assertEqual(
                demo.respond(reset["session_id"], f"request {turn}")["turn"],
                turn,
            )
        with self.assertRaisesRegex(ValueError, "limited to 10 turns"):
            demo.respond(reset["session_id"], "turn eleven")

    def test_route_distinguishes_grace_fallback_and_recovered_expert_tail(self) -> None:
        created: list[FakeFusionA] = []

        def create_a(path: Path) -> FakeFusionA:
            self.assertEqual(path, self.catalog_path)
            agent = FakeFusionA(fallback_page=3)
            created.append(agent)
            return agent

        demo = self.demo({"t0": FakeT0, "a": create_a, "b": FakeFusionB})
        reset = demo.reset("a")
        session_id = reset["session_id"]

        self.assertIn("grace", self.route(demo.respond(session_id, "page one")).lower())
        demo.respond(session_id, "page two")
        self.assertIn("fallback", self.route(demo.respond(session_id, "page three")).lower())
        recovered = self.route(demo.respond(session_id, "page four")).lower()
        self.assertIn("expert", recovered)
        self.assertNotIn("fallback", recovered)
        self.assertEqual(created[0].evaluation_diagnostics()["rank_fallbacks"], 1)

    def test_recommendations_are_catalog_valid_unique_and_capped_at_ten(self) -> None:
        for index in range(4, 13):
            identifier = f"P{index}"
            self.products[identifier] = {
                "parent_asin": identifier,
                "title": f"Product {identifier}",
                "categories": ["Clothing"],
            }
        valid_rows = [{"parent_asin": f"P{index}"} for index in range(1, 13)]

        def valid_factory(path: Path) -> FakeT0:
            self.assertEqual(path, self.catalog_path)
            return FakeT0(valid_rows)

        demo = self.demo({"t0": valid_factory, "a": FakeFusionA, "b": FakeFusionB})
        reset = demo.reset("t0")
        result = demo.respond(reset["session_id"], "request")
        identifiers = [item["parent_asin"] for item in result["recommendations"]]
        self.assertEqual(identifiers, [f"P{index}" for index in range(1, 11)])
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(set(identifiers).issubset(self.products))
        self.assertEqual(result["contract"], {
            "raw_recommendations": 12,
            "valid_unique_top10": 10,
            "dropped_or_beyond_top10": 2,
        })

        dirty_rows = [
            {"parent_asin": "P1"},
            {"parent_asin": "P1"},
            {"parent_asin": "UNKNOWN"},
            {},
            {"parent_asin": "P2"},
        ]
        dirty_demo = self.demo({
            "t0": lambda _path: FakeT0(dirty_rows),
            "a": FakeFusionA,
            "b": FakeFusionB,
        })
        dirty_reset = dirty_demo.reset("t0")
        normalized = dirty_demo.respond(dirty_reset["session_id"], "request")
        self.assertEqual(
            [item["parent_asin"] for item in normalized["recommendations"]],
            ["P1", "P2"],
        )
        self.assertEqual(normalized["contract"], {
            "raw_recommendations": 5,
            "valid_unique_top10": 2,
            "dropped_or_beyond_top10": 3,
        })


if __name__ == "__main__":
    unittest.main()
