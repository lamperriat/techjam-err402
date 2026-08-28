from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from scripts.build_p9_evidence import build_sidecar
from starter import p9_evidence
from starter.agent import Agent
from starter.p9_evidence import CompactEvidenceStore
from starter.p9_lab import (
    ACTIVE_ID,
    C00,
    CONTROL_ID,
    P9Agent,
    R01,
    S00,
    SCHEMA_VERSION,
    SHADOW_ID,
    SPECS,
    SPEC_BY_ID,
    create_p9_agent,
    validate_registry,
)


def product(
    identifier: str,
    *,
    color: str | None = None,
    category: str = "dress",
    material: str = "cotton",
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
        "features": [f"comfortable casual {category}", "easy care"],
        "details": details,
        "store": "Example Store",
        "description": "red" if color is None else "Everyday item",
    }


class P9LabTests(unittest.TestCase):
    def _assets(
        self,
        root: Path,
        products: list[dict[str, object]] | None = None,
    ) -> tuple[Path, Path]:
        if products is None:
            products = [
                *[product(f"R-{index:02d}", color="red") for index in range(6)],
                *[product(f"B-{index:02d}", color="blue") for index in range(10)],
                *[product(f"U-{index:02d}") for index in range(6)],
                *[
                    product(f"S-{index:02d}", color="black", category="shoe")
                    for index in range(12)
                ],
            ]
        catalog = root / "catalog.jsonl"
        catalog.write_text(
            "".join(json.dumps(item) + "\n" for item in products),
            encoding="utf-8",
        )
        evidence = root / "evidence.sqlite"
        build_sidecar(
            catalog,
            evidence,
            root / "evidence.metadata.json",
            expected_catalog_sha256=None,
            expected_catalog_rows=None,
        )
        return catalog, evidence

    @staticmethod
    def _catalog_identity(catalog: Path) -> tuple[str, int]:
        digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
        rows = sum(
            1
            for line in catalog.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return digest, rows

    def _agent(
        self,
        catalog: Path,
        variant: str,
        *,
        evidence_path: Path | None = None,
    ) -> P9Agent:
        catalog_sha256, catalog_rows = self._catalog_identity(catalog)
        with patch.multiple(
            p9_evidence,
            OFFICIAL_CATALOG_SHA256=catalog_sha256,
            OFFICIAL_CATALOG_ROWS=catalog_rows,
        ):
            return P9Agent(
                catalog,
                variant,
                evidence_path=evidence_path,
            )

    def test_registry_and_factory_contract_are_frozen(self) -> None:
        validate_registry()
        self.assertEqual(set(SPEC_BY_ID), {C00, S00, R01})
        self.assertEqual((CONTROL_ID, SHADOW_ID, ACTIVE_ID), (C00, S00, R01))
        self.assertEqual(len({spec.mechanism for spec in SPECS}), 3)
        parameters = dict(SPEC_BY_ID[R01].parameters)
        self.assertEqual(parameters["candidate_pool"], 50)
        self.assertEqual(parameters["minimum_catalog_evidence_confidence"], 0.90)
        self.assertFalse(parameters["catalog_description_evidence"])
        self.assertEqual(
            set(inspect.signature(create_p9_agent).parameters),
            {"role", "catalog_path", "evidence_path", "spec_path", "lock_path"},
        )

    def test_control_never_opens_sidecar_and_is_exact_served_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            root = Path(directory)
            catalog, _evidence = self._assets(root)
            served = Agent(
                catalog,
                retrieval_mode="coverage",
                rerank_mode="off",
                question_policy="fast",
            )
            control = create_p9_agent(
                role=C00,
                catalog_path=catalog,
                evidence_path=root / "does-not-exist.sqlite",
            )
            cleanup.callback(served.connection.close)
            cleanup.callback(control.close)
            served.reset("same", {})
            control.reset("same", {})
            for turn, message in enumerate((
                "I'm looking for women's dresses. not red",
                "No preference for material.",
                "Actually, replace cotton with linen.",
            ), start=1):
                self.assertEqual(
                    control.respond("same", message, turn, 10),
                    served.respond("same", message, turn, 10),
                )
            capture = control.export_p9_blind_capture()
            self.assertFalse(capture["configuration"]["evidence_opened"])
            self.assertNotIn("evidence_sha256", capture["configuration"])
            self.assertNotIn("evidence_bytes", capture["configuration"])
            self.assertEqual(capture["integrity_errors"], [])

    def test_shadow_is_exact_output_safe_and_active_serves_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            catalog, evidence = self._assets(Path(directory))
            control = self._agent(catalog, C00)
            shadow = self._agent(catalog, S00, evidence_path=evidence)
            active = self._agent(catalog, R01, evidence_path=evidence)
            cleanup.callback(control.close)
            cleanup.callback(shadow.close)
            cleanup.callback(active.close)
            for agent in (control, shadow, active):
                agent.reset("same", {})
            message = "I'm looking for women's dresses. not red"
            expected = control.respond("same", message, 1, 10)
            shadow_response = shadow.respond("same", message, 1, 10)
            active_response = active.respond("same", message, 1, 10)

            self.assertEqual(shadow_response, expected)
            active_ids = [item["parent_asin"] for item in active_response["recommendations"]]
            self.assertTrue(active_ids)
            self.assertFalse(any(identifier.startswith("R-") for identifier in active_ids))
            self.assertEqual(shadow.experiment_stats()["shadow_changes"], 1)
            self.assertEqual(shadow.experiment_stats()["output_changes"], 0)
            self.assertEqual(active.experiment_stats()["output_changes"], 1)
            self.assertEqual(active.experiment_stats()["partition_totals"]["explicit_violation"], 6)

    def test_only_first_fifty_are_looked_up_and_no_full_cache_is_created(self) -> None:
        products = [
            product(f"R-{index:03d}", color="red" if index % 2 else "blue")
            for index in range(130)
        ]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            catalog, evidence = self._assets(Path(directory), products)
            active = self._agent(catalog, R01, evidence_path=evidence)
            cleanup.callback(active.close)
            active.reset("bounded", {})
            store = active._evidence
            self.assertIsNotNone(store)
            assert store is not None
            with patch.object(store, "fetch", wraps=store.fetch) as fetch:
                active.respond(
                    "bounded", "I'm looking for women's dresses. not red", 1, 10
                )
            requested = fetch.call_args.args[0]
            self.assertLessEqual(len(requested), 50)
            self.assertFalse(hasattr(active, "_p9_rowids"))
            self.assertEqual(len(active._attribute_view_cache), 0)
            self.assertIsNone(active._p9_candidate_rowids)

    def test_sidecar_exception_falls_back_to_exact_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            catalog, evidence = self._assets(Path(directory))
            control = self._agent(catalog, C00)
            active = self._agent(catalog, R01, evidence_path=evidence)
            cleanup.callback(control.close)
            cleanup.callback(active.close)
            control.reset("same", {})
            active.reset("same", {})
            message = "I'm looking for women's dresses. not red"
            expected = control.respond("same", message, 1, 10)
            with patch.object(
                CompactEvidenceStore,
                "fetch",
                side_effect=RuntimeError("injected"),
            ):
                observed = active.respond("same", message, 1, 10)
            self.assertEqual(observed, expected)
            stats = active.experiment_stats()
            self.assertEqual(stats["exception_count"], 1)
            self.assertEqual(stats["exact_exception_fallbacks"], 1)

    def test_instrumentation_exception_also_falls_back_to_exact_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            catalog, evidence = self._assets(Path(directory))
            control = self._agent(catalog, C00)
            active = self._agent(catalog, R01, evidence_path=evidence)
            cleanup.callback(control.close)
            cleanup.callback(active.close)
            control.reset("same", {})
            active.reset("same", {})
            message = "I'm looking for women's dresses. not red"
            expected = control.respond("same", message, 1, 10)
            with patch.object(
                active,
                "_record_computation",
                side_effect=RuntimeError("injected"),
            ):
                observed = active.respond("same", message, 1, 10)
            self.assertEqual(observed, expected)
            self.assertEqual(active.experiment_stats()["exception_count"], 1)
            self.assertIn(
                "computation_record:RuntimeError",
                active.export_p9_blind_capture()["integrity_errors"],
            )

    def test_stale_override_and_no_preference_stop_execution_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            catalog, evidence = self._assets(Path(directory))
            control = self._agent(catalog, C00)
            active = self._agent(catalog, R01, evidence_path=evidence)
            cleanup.callback(control.close)
            cleanup.callback(active.close)
            for agent in (control, active):
                agent.reset("state", {})
                agent.respond(
                    "state", "I'm looking for women's cotton dresses. not red", 1, 10
                )
            expected = control.respond(
                "state", "Actually, replace red with blue.", 2, 10
            )
            observed = active.respond(
                "state", "Actually, replace red with blue.", 2, 10
            )
            self.assertEqual(observed, expected)
            self.assertEqual(
                active.experiment_stats()["reason_counts"]["no_executable_negatives"],
                1,
            )

            expected = control.respond("state", "No preference for color.", 3, 10)
            observed = active.respond("state", "No preference for color.", 3, 10)
            self.assertEqual(observed, expected)

    def test_capture_is_streaming_hash_only_and_lock_identity_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            root = Path(directory)
            catalog, evidence = self._assets(root)
            spec = root / "spec.json"
            spec.write_text(json.dumps({"schema_version": "p9.spec.test"}), encoding="utf-8")
            evidence_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
            lock = root / "lock.json"
            lock.write_text(json.dumps({
                "schema_version": "p9.lock.test",
                "evidence": {"bytes": evidence.stat().st_size, "sha256": evidence_sha},
            }), encoding="utf-8")
            catalog_sha256, catalog_rows = self._catalog_identity(catalog)
            with patch.multiple(
                p9_evidence,
                OFFICIAL_CATALOG_SHA256=catalog_sha256,
                OFFICIAL_CATALOG_ROWS=catalog_rows,
            ):
                shadow = create_p9_agent(
                    role=S00,
                    catalog_path=catalog,
                    evidence_path=evidence,
                    spec_path=spec,
                    lock_path=lock,
                )
            cleanup.callback(shadow.close)
            shadow.reset("capture", {})
            shadow.respond(
                "capture", "I'm looking for women's dresses. not red", 1, 10
            )
            capture = shadow.export_p9_blind_capture()

            self.assertEqual(set(capture), {
                "schema_version",
                "role",
                "configuration",
                "stats",
                "integrity_errors",
                "hashes",
                "function_hashes",
            })
            self.assertEqual(capture["schema_version"], SCHEMA_VERSION)
            self.assertTrue(capture["configuration"]["evidence_opened"])
            self.assertTrue(capture["configuration"]["evidence_identity_verified"])
            self.assertEqual(capture["configuration"]["evidence_sha256"], evidence_sha)
            self.assertEqual(capture["configuration"]["evidence_bytes"], evidence.stat().st_size)
            self.assertEqual(capture["integrity_errors"], [])
            self.assertFalse(hasattr(shadow, "_p9_audit"))
            self.assertFalse(hasattr(shadow, "_p9_responses"))
            self.assertEqual(len(capture["hashes"]["responses_sha256"]), 64)

    def test_runtime_source_has_no_external_decision_inputs(self) -> None:
        source = inspect.getsource(P9Agent).casefold()
        self.assertNotIn("from evaluator", source)
        self.assertNotIn("import evaluator", source)
        for forbidden in ("sample_id", "scenario_type", "results_path"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
