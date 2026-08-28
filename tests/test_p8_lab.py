from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.agent import Agent
from starter.p8_lab import (
    ACTIVE_ID,
    C00,
    CONTROL_ID,
    P8Agent,
    R01,
    S00,
    SCHEMA_VERSION,
    SHADOW_ID,
    SPECS,
    SPEC_BY_ID,
    create_p8_agent,
    validate_registry,
)


def product(
    identifier: str,
    *,
    color: str | None = None,
    category: str = "dress",
    material: str = "cotton",
    description: str = "Everyday item.",
) -> dict[str, object]:
    audience = "Men" if category == "shoe" else "Women"
    category_label = "Shoes" if category == "shoe" else "Dresses"
    details: dict[str, str] = {"Material": material, "Department": audience}
    if color is not None:
        details["Color"] = color
    return {
        "parent_asin": identifier,
        "title": f"{audience}'s {category} everyday option {identifier}",
        "categories": ["Clothing", audience, category_label],
        "features": [f"comfortable {category}", "easy care"],
        "details": details,
        "store": "Example Store",
        "description": description,
    }


class P8LabTests(unittest.TestCase):
    def _catalog(self, directory: str, products: list[dict[str, object]] | None = None) -> Path:
        if products is None:
            products = [
                *[product(f"R-{index:02d}", color="red") for index in range(6)],
                *[product(f"B-{index:02d}", color="blue") for index in range(10)],
                *[product(f"U-{index:02d}") for index in range(6)],
                *[product(f"S-{index:02d}", color="black", category="shoe") for index in range(12)],
            ]
        path = Path(directory) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in products),
            encoding="utf-8",
        )
        return path

    def test_registry_and_frozen_constructor_configuration_are_exact(self) -> None:
        validate_registry()
        self.assertEqual(set(SPEC_BY_ID), {
            "P8.C00.r08_coverage",
            "P8.S00.explicit_negative_shadow",
            "P8.R01.explicit_negative_partition",
        })
        self.assertEqual((CONTROL_ID, SHADOW_ID, ACTIVE_ID), (C00, S00, R01))
        self.assertEqual(len({spec.mechanism for spec in SPECS}), 3)
        self.assertEqual(len({spec.stage_graph for spec in SPECS}), 3)
        active_parameters = dict(SPEC_BY_ID[R01].parameters)
        self.assertEqual(active_parameters["candidate_pool"], 50)
        self.assertEqual(active_parameters["minimum_catalog_evidence_confidence"], 0.90)
        self.assertFalse(active_parameters["catalog_description_evidence"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown P8 variant"):
                P8Agent(self._catalog(directory), "P8.invalid")

    def test_control_is_response_and_route_equal_to_explicit_served_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            served = Agent(
                catalog,
                retrieval_mode="coverage",
                rerank_mode="off",
                question_policy="fast",
            )
            control = P8Agent(catalog, C00)
            self.addCleanup(served.connection.close)
            self.addCleanup(control.connection.close)
            served.reset("same", {})
            control.reset("same", {})
            messages = [
                "I'm looking for women's dresses. not red",
                "No preference for material.",
                "Actually, replace cotton with linen.",
            ]
            for turn, message in enumerate(messages, start=1):
                self.assertEqual(
                    control.respond("same", message, turn, 10),
                    served.respond("same", message, turn, 10),
                )
                self.assertEqual(
                    control.debug_rankings("same"), served.debug_rankings("same")
                )
        self.assertEqual(control.experiment_stats()["turns"], 6)
        self.assertEqual(control.experiment_stats()["output_changes"], 0)

    def test_shadow_proposes_partition_but_is_exact_output_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            control = P8Agent(catalog, C00)
            shadow = P8Agent(catalog, S00)
            self.addCleanup(control.connection.close)
            self.addCleanup(shadow.connection.close)
            control.reset("same", {})
            shadow.reset("same", {})
            message = "I'm looking for women's dresses. not red"
            expected = control.respond("same", message, 1, 10)
            observed = shadow.respond("same", message, 1, 10)
            diagnostics = shadow.debug_negative_diagnostics("same")

        self.assertEqual(observed, expected)
        self.assertTrue(diagnostics["active"])
        self.assertTrue(diagnostics["would_change_top_10"])
        self.assertFalse(diagnostics["changed_top_10"])
        self.assertFalse(diagnostics["affects_output"])
        self.assertEqual(shadow.experiment_stats()["shadow_changes"], 1)
        self.assertEqual(shadow.experiment_stats()["output_changes"], 0)

    def test_active_serves_compatible_then_unknown_without_known_violations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active = P8Agent(self._catalog(directory), R01)
            self.addCleanup(active.connection.close)
            active.reset("negative", {})
            response = active.respond(
                "negative", "I'm looking for women's dresses. not red", 1, 10
            )
            diagnostics = active.debug_negative_diagnostics("negative")

        identifiers = [item["parent_asin"] for item in response["recommendations"]]
        self.assertTrue(identifiers)
        self.assertFalse(any(identifier.startswith("R-") for identifier in identifiers))
        self.assertTrue(diagnostics["changed_top_10"])
        self.assertEqual(diagnostics["partition"]["counts"]["explicit_violation"], 6)
        self.assertEqual(diagnostics["partition"]["violation_fallback_count"], 0)
        self.assertEqual(active.experiment_stats()["output_changes"], 1)

    def test_short_list_fallback_is_stable_and_explicit(self) -> None:
        products = [
            product("R-0", color="red"),
            product("R-1", color="red"),
            product("R-2", color="red"),
            product("B-0", color="blue"),
            product("B-1", color="blue"),
            product("U-0"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            active = P8Agent(self._catalog(directory, products), R01)
            self.addCleanup(active.connection.close)
            active.reset("short", {})
            response = active.respond(
                "short", "I'm looking for women's dresses. not red", 1, 10
            )
            diagnostics = active.debug_negative_diagnostics("short")

        identifiers = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(identifiers[:3], ["B-0", "B-1", "U-0"])
        self.assertEqual(identifiers[3:], ["R-0", "R-1", "R-2"])
        self.assertTrue(diagnostics["fallback"])
        self.assertEqual(diagnostics["partition"]["violation_fallback_count"], 3)
        self.assertEqual(active.experiment_stats()["fallbacks"], 1)

    def test_no_preference_and_category_reset_retire_negative_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            active = P8Agent(catalog, R01)
            control = P8Agent(catalog, C00)
            self.addCleanup(active.connection.close)
            self.addCleanup(control.connection.close)
            for agent in (active, control):
                agent.reset("state", {})
                agent.respond(
                    "state", "I'm looking for women's dresses. not red", 1, 10
                )
            observed = active.respond("state", "No preference for color.", 2, 10)
            expected = control.respond("state", "No preference for color.", 2, 10)
            no_preference = active.debug_negative_diagnostics("state")
            self.assertEqual(observed, expected)
            self.assertEqual(no_preference["reason"], "no_executable_negatives")

            observed = active.respond(
                "state", "I'm looking for men's shoes. blue", 3, 10
            )
            expected = control.respond(
                "state", "I'm looking for men's shoes. blue", 3, 10
            )
            category_reset = active.debug_negative_diagnostics("state")

        self.assertEqual(observed, expected)
        self.assertEqual(category_reset["reason"], "no_executable_negatives")
        self.assertEqual(category_reset["compilation"]["executable_count"], 0)

    def test_selective_override_exposes_stale_negative_and_stops_safely(self) -> None:
        cases = (
            "Actually, replace red with blue.",
            "Actually, replace cotton with linen.",
        )
        for case_index, override in enumerate(cases):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                catalog = self._catalog(directory)
                active = P8Agent(catalog, R01)
                control = P8Agent(catalog, C00)
                self.addCleanup(active.connection.close)
                self.addCleanup(control.connection.close)
                session = f"override-{case_index}"
                for agent in (active, control):
                    agent.reset(session, {})
                    agent.respond(
                        session,
                        "I'm looking for women's cotton dresses. not red",
                        1,
                        10,
                    )
                observed = active.respond(session, override, 2, 10)
                expected = control.respond(session, override, 2, 10)
                diagnostics = active.debug_negative_diagnostics(session)
                snapshot = active.debug_snapshot(session)

                self.assertEqual(observed, expected)
                self.assertEqual(diagnostics["reason"], "no_executable_negatives")
                self.assertGreaterEqual(
                    diagnostics["compilation"]["rejection_counts"].get(
                        "stale_goal_version", 0
                    ),
                    1,
                )
                active_red = [
                    record
                    for record in snapshot["slot_ledger"]["active"]
                    if record["value"] == "red" and record["polarity"] == -1
                ]
                self.assertTrue(active_red)
                self.assertLess(active_red[0]["version"], snapshot["version"])

    def test_exception_falls_back_to_exact_control_and_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory)
            active = P8Agent(catalog, R01)
            control = P8Agent(catalog, C00)
            self.addCleanup(active.connection.close)
            self.addCleanup(control.connection.close)
            active.reset("active", {})
            control.reset("control", {})
            message = "I'm looking for women's dresses. not red"
            expected = control.respond("control", message, 1, 10)
            with patch.object(active, "_load_p8_views", side_effect=RuntimeError("boom")):
                observed = active.respond("active", message, 1, 10)
            diagnostics = active.debug_negative_diagnostics("active")

        self.assertEqual(observed, expected)
        self.assertEqual(diagnostics["reason"], "exception_fallback")
        self.assertEqual(diagnostics["exception_class"], "RuntimeError")
        self.assertEqual(active.experiment_stats()["exception_count"], 1)

    def test_factory_and_blind_capture_have_a_frozen_hash_only_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._catalog(directory)
            spec = root / "spec.json"
            spec.write_text(json.dumps({"schema_version": "p8.spec.test"}), encoding="utf-8")
            spec_sha = hashlib.sha256(spec.read_bytes()).hexdigest()
            lock = root / "lock.json"
            lock.write_text(json.dumps({
                "schema_version": "p8.lock.test",
                "spec_sha256": spec_sha,
            }), encoding="utf-8")
            control = create_p8_agent(
                role="C00",
                catalog_path=catalog,
                spec_path=spec,
                lock_path=lock,
            )
            shadow = create_p8_agent(
                role="S00",
                catalog_path=catalog,
                spec_path=spec,
                lock_path=lock,
            )
            self.addCleanup(control.connection.close)
            self.addCleanup(shadow.connection.close)
            for agent in (control, shadow):
                agent.reset("capture", {})
                agent.respond(
                    "capture", "I'm looking for women's dresses. not red", 1, 10
                )
            control_capture = control.export_p8_blind_capture()
            shadow_capture = shadow.export_p8_blind_capture()

        expected_keys = {
            "schema_version",
            "role",
            "configuration",
            "stats",
            "integrity_errors",
            "hashes",
            "function_hashes",
        }
        self.assertEqual(set(control_capture), expected_keys)
        self.assertEqual(control_capture["schema_version"], SCHEMA_VERSION)
        self.assertEqual(control_capture["integrity_errors"], [])
        self.assertEqual(
            control_capture["hashes"]["responses_sha256"],
            shadow_capture["hashes"]["responses_sha256"],
        )
        self.assertNotEqual(
            control_capture["hashes"]["audit_sha256"],
            shadow_capture["hashes"]["audit_sha256"],
        )
        self.assertEqual(control_capture["configuration"]["spec_sha256"], spec_sha)
        self.assertEqual(control_capture["stats"]["output_changes"], 0)
        self.assertIn("exception_count", control_capture["stats"])

    def test_contract_is_catalog_only_and_has_no_external_decision_inputs(self) -> None:
        constructor = set(inspect.signature(P8Agent.__init__).parameters)
        factory = set(inspect.signature(create_p8_agent).parameters)
        self.assertEqual(constructor, {"self", "catalog_path", "variant_id", "registration"})
        self.assertEqual(
            factory, {"role", "catalog_path", "spec_path", "lock_path"}
        )
        source = inspect.getsource(P8Agent).casefold()
        self.assertNotIn("from evaluator", source)
        self.assertNotIn("import evaluator", source)
        for forbidden in ("sample_id", "scenario_type", "results_path"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
