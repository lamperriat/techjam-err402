from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter import p11_bridge, p11_features
from starter.agent import DEFAULT_P11_MODE, P11_MODES, Agent
from starter.p11_bridge import P11ProductionBridge
from starter.p11_features import P11RerankResult
from starter.p11_lab import ACTIVE_ID, P11Agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
P11_CONFIG = PROJECT_ROOT / "configs" / "p11_production_bridge.json"
P11_LOCK = PROJECT_ROOT / "configs" / "p11_prereg_lock.json"
P11_ASSET = PROJECT_ROOT / "starter" / "assets" / "p11_features.sqlite"
P11_ASSET_MANIFEST = (
    PROJECT_ROOT / "starter" / "assets" / "p11_features.manifest.json"
)
EXPECTED_SIDECAR_BYTES = 32_501_760
EXPECTED_SIDECAR_SHA256 = (
    "83b6d8c04be6666173806b6e9cb03301eecb8ca58a60272bfa719e6533380473"
)
EXPECTED_FEATURES_SHA256 = (
    "636168a599f8fd872a0bc8dc369691af6cd03eaa446aaaf136f809eb5627588c"
)
EXPECTED_LAB_SHA256 = (
    "8ccc1007f0c20965c10b37727d459fc80c9ce06e8bf5245a75aa4a44ba8205b9"
)
MESSAGE = (
    "I'm looking for women's dresses. "
    "A key requirement is blue cotton casual summer wear."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_source_sha256(path: Path) -> str:
    """Match the LF bytes frozen by P11 across Git's Windows CRLF checkout."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _official_catalog() -> Path | None:
    candidates = (
        PROJECT_ROOT / "data" / "catalog.jsonl",
        PROJECT_ROOT.parent / "techjam-err402" / "data" / "catalog.jsonl",
    )
    return next((path for path in candidates if path.is_file()), None)


def _write_fixture_catalog(root: Path, count: int = 18) -> Path:
    path = root / "catalog.jsonl"
    rows = [
        {
            "parent_asin": f"FIXTURE-{index:03d}",
            "title": f"Women's blue cotton casual summer dress {index}",
            "categories": ["Clothing", "Women", "Dresses"],
            "features": ["blue", "cotton", "casual", "summer"],
            "details": {"Color": "Blue", "Material": "Cotton"},
            "store": "Fixture",
            "description": "A casual blue cotton summer dress.",
            "price": 20.0 + index,
            "average_rating": 4.0,
            "rating_number": index,
        }
        for index in range(1, count + 1)
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _recommendation_ids(response: dict) -> list[str]:
    return [str(row["parent_asin"]) for row in response["recommendations"]]


def _production_dependency_paths() -> tuple[Path, ...]:
    """Resolve the local Python dependency closure of the served P11 path."""

    pending = ["starter.agent", "starter.p11_bridge"]
    modules: set[str] = set()
    paths: set[Path] = set()
    while pending:
        module = pending.pop()
        if module in modules or not module.startswith("starter."):
            continue
        modules.add(module)
        path = PROJECT_ROOT.joinpath(*module.split(".")).with_suffix(".py")
        if not path.is_file():
            continue
        paths.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and str(node.module or "").startswith(
                "starter."
            ):
                pending.append(str(node.module))
            elif isinstance(node, ast.Import):
                pending.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("starter.")
                )
    return tuple(sorted(paths))


def _snapshot_without_p11(agent: Agent, session_id: str) -> dict:
    snapshot = agent.debug_snapshot(session_id)
    snapshot.pop("p11", None)
    return snapshot


def _assert_response_contract(
    case: unittest.TestCase,
    response: dict,
    *,
    catalog_ids: set[str],
    top_k: int,
) -> None:
    case.assertIs(type(response), dict)
    case.assertEqual(
        set(response), {"message", "ask_attribute", "recommendations", "usage"}
    )
    case.assertIs(type(response["message"]), str)
    case.assertTrue(response["message"])
    case.assertTrue(
        response["ask_attribute"] is None
        or type(response["ask_attribute"]) is str
    )

    recommendations = response["recommendations"]
    case.assertIs(type(recommendations), list)
    case.assertLessEqual(len(recommendations), min(top_k, 10))
    identifiers: list[str] = []
    for recommendation in recommendations:
        case.assertIs(type(recommendation), dict)
        case.assertEqual(set(recommendation), {"parent_asin"})
        identifier = recommendation["parent_asin"]
        case.assertIs(type(identifier), str)
        case.assertIn(identifier, catalog_ids)
        identifiers.append(identifier)
    case.assertEqual(len(identifiers), len(set(identifiers)))

    usage = response["usage"]
    case.assertIs(type(usage), dict)
    case.assertEqual(set(usage), {"prompt_tokens", "completion_tokens"})
    for value in usage.values():
        case.assertIs(type(value), int)
        case.assertGreaterEqual(value, 0)


def _assert_single_fallback(
    case: unittest.TestCase,
    agent: Agent,
    reason_code: str,
    *,
    expected_rows_read: int = 0,
) -> None:
    bridge = agent._p11_bridge
    case.assertIsNotNone(bridge)
    stats_before_debug = dict(bridge.status()["stats"])
    session_id = next(iter(agent._sessions))
    agent.debug_rankings(session_id)
    agent.debug_rerank_diagnostics(session_id)
    agent.debug_p11_diagnostics(session_id)
    stats_after_debug = dict(bridge.status()["stats"])
    case.assertEqual(stats_after_debug, stats_before_debug)
    case.assertEqual(stats_after_debug["turns"], 1)
    case.assertEqual(stats_after_debug["fallbacks"], 1)
    case.assertEqual(stats_after_debug["reason_counts"], {reason_code: 1})
    case.assertEqual(stats_after_debug["sidecar_rows_read"], expected_rows_read)
    case.assertLessEqual(stats_after_debug["maximum_rows_per_fetch"], 10)


class P11ProductionIdentityTests(unittest.TestCase):
    def test_config_module_lock_and_tracked_asset_identities_are_exact(self) -> None:
        config = json.loads(P11_CONFIG.read_text(encoding="utf-8"))
        lock = json.loads(P11_LOCK.read_text(encoding="utf-8"))
        asset_manifest = json.loads(P11_ASSET_MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(tuple(config["modes"]), P11_MODES)
        self.assertEqual(config["default_mode_during_phase_a"], "off")
        self.assertEqual(DEFAULT_P11_MODE, "active")
        self.assertEqual(config["served_default_after_phase_a"], DEFAULT_P11_MODE)
        self.assertEqual(
            config["sidecar"]["path"], "starter/assets/p11_features.sqlite"
        )
        self.assertEqual(config["sidecar"]["bytes"], EXPECTED_SIDECAR_BYTES)
        self.assertEqual(config["sidecar"]["sha256"], EXPECTED_SIDECAR_SHA256)
        self.assertEqual(lock["sidecar"]["bytes"], EXPECTED_SIDECAR_BYTES)
        self.assertEqual(lock["sidecar"]["sha256"], EXPECTED_SIDECAR_SHA256)
        self.assertEqual(
            config["sidecar"]["catalog_rows"], lock["sidecar"]["catalog_rows"]
        )
        self.assertEqual(
            config["sidecar"]["catalog_sha256"],
            lock["sidecar"]["catalog_sha256"],
        )
        self.assertEqual(config["frozen_scorer"], lock["feature_contract"])
        for key in ("bytes", "sha256", "catalog_rows", "catalog_sha256"):
            self.assertEqual(asset_manifest[key], config["sidecar"][key])
        for key in (
            "feature_schema_version",
            "scorer_version",
            "feature_registry_sha256",
            "feature_semantics_sha256",
        ):
            self.assertEqual(asset_manifest[key], config["frozen_scorer"][key])

        self.assertEqual(p11_bridge.EXPECTED_SIDECAR_BYTES, EXPECTED_SIDECAR_BYTES)
        self.assertEqual(
            p11_bridge.EXPECTED_SIDECAR_SHA256, EXPECTED_SIDECAR_SHA256
        )
        self.assertEqual(p11_bridge.DEFAULT_SIDECAR, P11_ASSET)
        self.assertEqual(p11_features.SCHEMA_VERSION, "p11.top10-features.v2")
        self.assertEqual(p11_features.SCORER_VERSION, "p11.top10-linear.v3")
        self.assertEqual(
            p11_features.REGISTRY_SHA256,
            config["frozen_scorer"]["feature_registry_sha256"],
        )
        self.assertEqual(
            p11_features.SEMANTICS_SHA256,
            config["frozen_scorer"]["feature_semantics_sha256"],
        )

        self.assertTrue(P11_ASSET.is_file())
        self.assertEqual(P11_ASSET.stat().st_size, EXPECTED_SIDECAR_BYTES)
        self.assertEqual(_sha256(P11_ASSET), EXPECTED_SIDECAR_SHA256)

    def test_frozen_p11_sources_match_the_formal_lock(self) -> None:
        lock = json.loads(P11_LOCK.read_text(encoding="utf-8"))
        sources = lock["source"]["files"]
        expectations = {
            "p11_features": EXPECTED_FEATURES_SHA256,
            "p11_lab": EXPECTED_LAB_SHA256,
        }
        for name, expected in expectations.items():
            with self.subTest(source=name):
                path = PROJECT_ROOT / sources[name]["path"]
                self.assertEqual(sources[name]["sha256"], expected)
                self.assertEqual(_frozen_source_sha256(path), expected)

    def test_production_dependency_closure_has_no_label_evaluator_or_asin_leakage(
        self,
    ) -> None:
        blocked_names = {
            "ground_" + "truth",
            "target_" + "asin",
            "sample_" + "id",
            "scenario_" + "type",
            "intent_" + "card",
            "public_" + "membership",
            "evaluation_" + "result",
            "first_" + "hit_turn",
            "best_" + "rank",
        }
        paths = _production_dependency_paths()
        relative_paths = {
            path.relative_to(PROJECT_ROOT).as_posix() for path in paths
        }
        self.assertTrue(
            {
                "starter/agent.py",
                "starter/p11_bridge.py",
                "starter/p11_features.py",
                "starter/p8_negative.py",
                "starter/p9_evidence.py",
            }.issubset(relative_paths)
        )
        self.assertNotIn("starter/p11_lab.py", relative_paths)

        asin_literal = re.compile(r"(?<![A-Z0-9])B[A-Z0-9]{9}(?![A-Z0-9])")
        for path in paths:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            with self.subTest(source=path.relative_to(PROJECT_ROOT).as_posix()):
                imported_roots = {
                    alias.name.split(".", 1)[0]
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import)
                    for alias in node.names
                } | {
                    str(node.module or "").split(".", 1)[0]
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                }
                self.assertFalse(imported_roots & {"evaluator", "observer", "scripts"})

                identifiers = {
                    node.id.casefold()
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                } | {
                    node.attr.casefold()
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                }
                self.assertFalse(identifiers & blocked_names)

                string_literals = (
                    node.value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)
                )
                leaked_asins = [
                    match.group(0)
                    for value in string_literals
                    for match in asin_literal.finditer(value)
                ]
                self.assertEqual(leaked_asins, [])


class P11ProductionFallbackTests(unittest.TestCase):
    def test_mode_validation_and_frozen_configuration_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _write_fixture_catalog(Path(directory))
            with self.assertRaisesRegex(ValueError, "p11_mode"):
                Agent(catalog, p11_mode="fallback")
            for name, kwargs in (
                ("question", {"question_policy": "boundary"}),
                ("rerank", {"rerank_mode": "shadow"}),
                ("retrieval", {"retrieval_mode": "control"}),
            ):
                with self.subTest(conflict=name):
                    with self.assertRaisesRegex(ValueError, "requires"):
                        Agent(catalog, p11_mode="control", **kwargs)

    def test_off_and_control_are_multiturn_response_route_and_session_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _write_fixture_catalog(Path(directory))
            with patch(
                "starter.p11_bridge.P11ProductionBridge",
                side_effect=AssertionError("off/control must not construct P11 bridge"),
            ) as bridge_factory:
                off = Agent(catalog, p11_mode="off")
                control = Agent(catalog, p11_mode="control")
            try:
                for agent in (off, control):
                    agent.reset("same-session", {})
                messages = (
                    MESSAGE,
                    "No preference for brand.",
                    "I do not want polyester; comfort is important.",
                    "Actually, replace blue with red.",
                    "Actually, replace red with green.",
                    "No preference for material.",
                )
                for turn, message in enumerate(messages, start=1):
                    with self.subTest(turn=turn):
                        off_response = off.respond(
                            "same-session", message, turn, 10
                        )
                        control_response = control.respond(
                            "same-session", message, turn, 10
                        )
                        self.assertEqual(control_response, off_response)
                        self.assertEqual(
                            control_response["ask_attribute"],
                            off_response["ask_attribute"],
                        )
                        self.assertEqual(
                            control.debug_rankings("same-session"),
                            off.debug_rankings("same-session"),
                        )
                        self.assertEqual(
                            control._sessions["same-session"],
                            off._sessions["same-session"],
                        )
                        self.assertEqual(
                            _snapshot_without_p11(control, "same-session"),
                            _snapshot_without_p11(off, "same-session"),
                        )

                self.assertEqual(control.p11_mode, "control")
                self.assertEqual(
                    control.debug_p11_diagnostics("same-session")["reason_code"],
                    "control_exact",
                )
                self.assertFalse(
                    control.debug_p11_diagnostics("same-session")["identity_verified"]
                )
                bridge_factory.assert_not_called()
            finally:
                off.close()
                control.close()

    def test_missing_and_wrong_identity_fail_closed_to_exact_r08(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = _write_fixture_catalog(root)
            missing = root / "missing.sqlite"
            wrong = root / "wrong.sqlite"
            wrong.write_bytes(b"not-the-frozen-sidecar")

            control = Agent(catalog, p11_mode="control")
            try:
                control.reset("control", {})
                expected_response = control.respond("control", MESSAGE, 1, 10)
                expected_rankings = control.debug_rankings("control")
            finally:
                control.close()

            catalog_mismatch = Agent(catalog, p11_mode="active")
            try:
                catalog_mismatch.reset("catalog-mismatch", {})
                self.assertEqual(
                    catalog_mismatch.respond("catalog-mismatch", MESSAGE, 1, 10),
                    expected_response,
                )
                mismatch_diagnostics = catalog_mismatch.debug_p11_diagnostics(
                    "catalog-mismatch"
                )
                self.assertEqual(
                    mismatch_diagnostics["reason_code"],
                    "catalog_identity_mismatch",
                )
                self.assertEqual(
                    mismatch_diagnostics["effective_mode"], "fallback"
                )
                _assert_single_fallback(
                    self, catalog_mismatch, "catalog_identity_mismatch"
                )
            finally:
                catalog_mismatch.close()

            cases = (
                ("active", missing, "sidecar_missing"),
                ("shadow", wrong, "sidecar_identity_mismatch"),
            )
            catalog_sha256 = _sha256(catalog)

            def fixture_bridge(
                mode: str,
                sidecar_path: str | Path | None,
                *,
                catalog_path: str | Path | None = None,
            ) -> P11ProductionBridge:
                return P11ProductionBridge(
                    mode,
                    sidecar_path,
                    expected_catalog_rows=18,
                    expected_catalog_sha256=catalog_sha256,
                    catalog_path=catalog_path,
                )

            for mode, sidecar, reason in cases:
                with self.subTest(mode=mode, reason=reason):
                    with patch(
                        "starter.p11_bridge.P11ProductionBridge",
                        side_effect=fixture_bridge,
                    ):
                        agent = Agent(
                            catalog,
                            p11_mode=mode,
                            p11_sidecar_path=sidecar,
                        )
                    try:
                        agent.reset(mode, {})
                        actual_response = agent.respond(mode, MESSAGE, 1, 10)
                        actual_rankings = agent.debug_rankings(mode)
                        diagnostics = agent.debug_p11_diagnostics(mode)
                        turns_before = agent._p11_bridge.status()["stats"]["turns"]
                        agent.debug_rankings(mode)
                        agent.debug_rerank_diagnostics(mode)
                        agent.debug_p11_diagnostics(mode)
                        turns_after = agent._p11_bridge.status()["stats"]["turns"]

                        self.assertEqual(actual_response, expected_response)
                        self.assertEqual(actual_rankings, expected_rankings)
                        self.assertEqual(diagnostics["effective_mode"], "fallback")
                        self.assertEqual(diagnostics["reason_code"], reason)
                        self.assertTrue(diagnostics["fallback"])
                        self.assertFalse(diagnostics["identity_verified"])
                        self.assertEqual(turns_before, 1)
                        self.assertEqual(turns_after, turns_before)
                        _assert_single_fallback(self, agent, reason)
                    finally:
                        agent.close()

    def test_bridge_close_status_and_apply_after_close_are_exact_fallback(self) -> None:
        bridge = P11ProductionBridge("active", P11_ASSET)
        self.assertTrue(bridge.status()["identity_verified"])
        self.assertEqual(bridge._store.connection.execute("PRAGMA query_only").fetchone(), (1,))
        bridge.close()
        bridge.close()

        baseline = tuple(f"ITEM-{index:02d}" for index in range(15))
        outcome = bridge.apply(
            object(),
            {
                "broad": baseline,
                "strict": baseline,
                "fused": baseline,
                "final": baseline,
            },
            {},
            (),
        )
        status = bridge.status()

        self.assertEqual(outcome.identifiers, baseline)
        self.assertEqual(outcome.diagnostics["reason_code"], "bridge_closed")
        self.assertEqual(outcome.diagnostics["effective_mode"], "fallback")
        self.assertTrue(outcome.diagnostics["fallback"])
        self.assertTrue(outcome.diagnostics["top10_membership_preserved"])
        self.assertTrue(outcome.diagnostics["tail_preserved"])
        self.assertEqual(status["effective_mode"], "fallback")
        self.assertEqual(status["reason_code"], "bridge_closed")
        self.assertFalse(status["identity_verified"])
        self.assertEqual(status["stats"]["turns"], 1)
        self.assertEqual(status["stats"]["fallbacks"], 1)
        self.assertEqual(
            status["stats"]["reason_counts"], {"bridge_closed": 1}
        )


class P11ProductionOfficialAssetTests(unittest.TestCase):
    def test_served_default_is_active_and_explicit_off_restores_r08(self) -> None:
        catalog = _official_catalog()
        if catalog is None:
            self.skipTest("official catalog is not available in this checkout")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TECHJAM_P11_MODE", None)
            served = Agent(catalog)
        try:
            self.assertEqual(served.p11_mode, "active")
            self.assertEqual(served._p11_status()["effective_mode"], "active")
            self.assertTrue(served._p11_status()["identity_verified"])
        finally:
            served.close()

        with patch.dict(os.environ, {"TECHJAM_P11_MODE": "off"}):
            fallback = Agent(catalog)
        try:
            self.assertEqual(fallback.p11_mode, "off")
            self.assertEqual(fallback._p11_status()["effective_mode"], "off")
            self.assertIsNone(fallback._p11_bridge)
        finally:
            fallback.close()

    def test_explicit_legacy_r08_configuration_stays_off_after_promotion(self) -> None:
        catalog = _official_catalog()
        if catalog is None:
            self.skipTest("official catalog is not available in this checkout")

        historical = Agent(
            catalog,
            question_policy="fast",
            rerank_mode="off",
            retrieval_mode="coverage",
        )
        promoted = Agent(catalog)
        try:
            self.assertEqual(historical.p11_mode, "off")
            self.assertIsNone(historical._p11_bridge)
            self.assertEqual(promoted.p11_mode, "active")
            self.assertEqual(promoted._p11_status()["effective_mode"], "active")
        finally:
            historical.close()
            promoted.close()

    def test_runtime_feature_score_adapter_and_boundary_failures_are_exact_r08(
        self,
    ) -> None:
        catalog = _official_catalog()
        if catalog is None:
            self.skipTest("official catalog is not available in this checkout")

        control = Agent(catalog, p11_mode="control")
        try:
            control.reset("control", {})
            expected_response = control.respond("control", MESSAGE, 1, 10)
            expected_rankings = control.debug_rankings("control")
        finally:
            control.close()

        def score_fallback(
            identifiers: tuple[str, ...], *_args: object, **_kwargs: object
        ) -> P11RerankResult:
            return P11RerankResult(
                tuple(identifiers), True, "forced-score-fallback", False, {}
            )

        cases = (
            (
                "feature",
                lambda active: patch.object(
                    active._p11_bridge._store,
                    "fetch_top10",
                    side_effect=RuntimeError("injected generic feature failure"),
                ),
                "feature_failure",
                0,
            ),
            (
                "score",
                lambda _active: patch(
                    "starter.p11_bridge.rerank_top10_preserving_membership",
                    side_effect=score_fallback,
                ),
                "score_failure",
                10,
            ),
            (
                "adapter",
                lambda _active: patch(
                    "starter.p11_bridge._positive_constraints",
                    side_effect=RuntimeError("injected adapter failure"),
                ),
                "bridge_failure",
                10,
            ),
            (
                "outer_adapter",
                lambda active: patch.object(
                    active._p11_bridge,
                    "apply",
                    side_effect=RuntimeError("injected bridge call failure"),
                ),
                "bridge_adapter_failure",
                0,
            ),
            (
                "boundary",
                lambda _active: patch(
                    "starter.p11_bridge.rerank_top10_preserving_membership",
                    side_effect=lambda identifiers, *_args, **_kwargs: P11RerankResult(
                        ("OUTSIDE-CANDIDATE", *tuple(identifiers)[1:]),
                        False,
                        "scored",
                        True,
                        {},
                    ),
                ),
                "boundary_violation",
                10,
            ),
        )
        for name, patch_factory, expected_reason, expected_rows_read in cases:
            with self.subTest(failure=name):
                active = Agent(catalog, p11_mode="active")
                try:
                    active.reset(name, {})
                    with patch_factory(active):
                        response = active.respond(name, MESSAGE, 1, 10)
                    self.assertEqual(response, expected_response)
                    self.assertEqual(active.debug_rankings(name), expected_rankings)
                    diagnostics = active.debug_p11_diagnostics(name)
                    self.assertEqual(diagnostics["effective_mode"], "fallback")
                    self.assertEqual(diagnostics["reason_code"], expected_reason)
                    self.assertTrue(diagnostics["fallback"])
                    self.assertTrue(diagnostics["top10_membership_preserved"])
                    self.assertTrue(diagnostics["tail_preserved"])
                    _assert_single_fallback(
                        self,
                        active,
                        expected_reason,
                        expected_rows_read=expected_rows_read,
                    )
                finally:
                    active.close()

    def test_actual_rowid_to_asin_binding_mismatch_falls_back_exactly(self) -> None:
        catalog = _official_catalog()
        if catalog is None:
            self.skipTest("official catalog is not available in this checkout")

        control = Agent(catalog, p11_mode="control")
        active = Agent(catalog, p11_mode="active")
        try:
            control.reset("control", {})
            expected_response = control.respond("control", MESSAGE, 1, 10)
            expected_rankings = control.debug_rankings("control")

            original_apply = active._p11_bridge.apply

            def apply_with_wrong_binding(
                state: object,
                rankings: dict[str, list[str]],
                candidate_rowids: dict[str, int],
                query_terms: list[str],
            ) -> object:
                head = rankings["final"][:2]
                self.assertEqual(len(head), 2)
                wrong = dict(candidate_rowids)
                wrong[head[0]], wrong[head[1]] = wrong[head[1]], wrong[head[0]]
                return original_apply(state, rankings, wrong, query_terms)

            active.reset("binding", {})
            with patch.object(
                active._p11_bridge,
                "apply",
                side_effect=apply_with_wrong_binding,
            ):
                response = active.respond("binding", MESSAGE, 1, 10)
            self.assertEqual(response, expected_response)
            self.assertEqual(active.debug_rankings("binding"), expected_rankings)
            diagnostics = active.debug_p11_diagnostics("binding")
            self.assertEqual(diagnostics["reason_code"], "candidate_binding_failure")
            self.assertTrue(diagnostics["fallback"])
            _assert_single_fallback(self, active, "candidate_binding_failure")
        finally:
            control.close()
            active.close()

    def test_reset_drop_and_concurrent_sessions_score_once_per_response(self) -> None:
        catalog = _official_catalog()
        if catalog is None:
            self.skipTest("official catalog is not available in this checkout")

        agent = Agent(catalog, p11_mode="shadow")
        try:
            agent.reset("reused", {})
            agent.respond("reused", MESSAGE, 1, 10)
            agent.reset("reused", {})
            with self.assertRaises(KeyError):
                agent.debug_rankings("missing")
            agent.respond("reused", MESSAGE, 1, 10)
            agent.drop_session("reused")
            with self.assertRaises(KeyError):
                agent.debug_rankings("reused")

            def run(index: int) -> list[str]:
                session_id = f"parallel-{index}"
                agent.reset(session_id, {})
                response = agent.respond(session_id, MESSAGE, 1, 10)
                return _recommendation_ids(response)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                outputs = list(pool.map(run, range(8)))
            self.assertTrue(all(output == outputs[0] for output in outputs))
            stats = agent._p11_bridge.status()["stats"]
            self.assertEqual(stats["turns"], 10)
            self.assertEqual(stats["sidecar_rows_read"], 100)
            self.assertEqual(stats["maximum_rows_per_fetch"], 10)
            self.assertEqual(stats["fallbacks"], 0)
        finally:
            agent.close()

    def test_shadow_and_active_are_target_blind_single_pass_top10_reranks(self) -> None:
        catalog = _official_catalog()
        if catalog is None:
            self.skipTest("official catalog is not available in this checkout")

        control = Agent(catalog, p11_mode="control")
        try:
            control.reset("control", {})
            control_response = control.respond("control", MESSAGE, 1, 10)
            control_rankings = control.debug_rankings("control")
        finally:
            control.close()

        observed: dict[str, tuple[dict, dict[str, list[str]], dict, list[dict]]] = {}
        for mode in ("shadow", "active"):
            events: list[dict] = []
            agent = Agent(catalog, p11_mode=mode, trace_sink=events.append)
            try:
                self.assertTrue(agent._p11_bridge.status()["identity_verified"])
                agent.reset(mode, {})
                response = agent.respond(mode, MESSAGE, 1, 10)
                rankings = agent.debug_rankings(mode)
                diagnostics = agent.debug_p11_diagnostics(mode)
                stats_before = dict(agent._p11_bridge.status()["stats"])

                for _ in range(3):
                    self.assertEqual(agent.debug_rankings(mode), rankings)
                    agent.debug_rerank_diagnostics(mode)
                    self.assertEqual(agent.debug_p11_diagnostics(mode), diagnostics)
                stats_after = dict(agent._p11_bridge.status()["stats"])

                self.assertEqual(stats_before, stats_after)
                self.assertEqual(stats_after["turns"], 1)
                self.assertEqual(diagnostics["configured_mode"], mode)
                self.assertEqual(diagnostics["effective_mode"], mode)
                self.assertEqual(diagnostics["reason_code"], "scored")
                self.assertTrue(diagnostics["identity_verified"])
                self.assertFalse(diagnostics["fallback"])
                self.assertTrue(diagnostics["top10_membership_preserved"])
                self.assertTrue(diagnostics["tail_preserved"])
                self.assertEqual(
                    diagnostics["baseline_top10"], control_rankings["final"][:10]
                )
                self.assertEqual(
                    set(diagnostics["proposed_top10"]),
                    set(control_rankings["final"][:10]),
                )
                self.assertEqual(
                    _recommendation_ids(response), rankings["final"][:10]
                )

                serialized = json.dumps(events, sort_keys=True).casefold()
                for blocked in (
                    "ground_" + "truth",
                    "target_" + "asin",
                    "sample_" + "id",
                    "scenario_" + "type",
                    "intent_" + "card",
                ):
                    self.assertNotIn(blocked, serialized)
                observed[mode] = (response, rankings, diagnostics, events)
            finally:
                agent.close()

        shadow_response, shadow_rankings, shadow_diagnostics, _ = observed["shadow"]
        active_response, active_rankings, active_diagnostics, _ = observed["active"]

        self.assertEqual(shadow_response, control_response)
        self.assertEqual(shadow_rankings, control_rankings)
        self.assertFalse(shadow_diagnostics["output_changed"])
        self.assertEqual(
            shadow_diagnostics["served_top10"], shadow_diagnostics["baseline_top10"]
        )

        for route in ("broad", "strict", "fused", "reranked"):
            self.assertEqual(active_rankings[route], control_rankings[route])
        self.assertEqual(
            active_rankings["final"][10:], control_rankings["final"][10:]
        )
        self.assertEqual(
            active_rankings["final"][:10], active_diagnostics["proposed_top10"]
        )
        self.assertEqual(
            _recommendation_ids(active_response), active_diagnostics["served_top10"]
        )
        self.assertEqual(
            {key: value for key, value in active_response.items() if key != "recommendations"},
            {key: value for key, value in control_response.items() if key != "recommendations"},
        )

    def test_repeated_override_negative_supersede_and_unknown_match_frozen_p11(
        self,
    ) -> None:
        catalog = _official_catalog()
        if catalog is None:
            self.skipTest("official catalog is not available in this checkout")

        production = Agent(catalog, p11_mode="active")
        frozen = P11Agent(
            catalog,
            ACTIVE_ID,
            sidecar_path=P11_ASSET,
            expected_sidecar=(EXPECTED_SIDECAR_BYTES, EXPECTED_SIDECAR_SHA256),
        )
        turns = (
            "I'm looking for women's red cotton dresses. A key requirement is casual style.",
            "I do not want red, and I still need a cotton dress.",
            "Actually, replace cotton with linen.",
            "Actually, replace linen with silk.",
            "I do not want wool; what matters is a blue silk dress.",
            "Actually, replace blue with green.",
            "No preference for material.",
        )
        conflict_states: set[str] = set()
        try:
            production.reset("production", {})
            frozen.reset("frozen", {})
            for turn, message in enumerate(turns, start=1):
                with self.subTest(turn=turn):
                    production_response = production.respond(
                        "production", message, turn, 10
                    )
                    frozen_response = frozen.respond("frozen", message, turn, 10)
                    self.assertEqual(production_response, frozen_response)
                    self.assertEqual(
                        production.debug_rankings("production"),
                        frozen.debug_rankings("frozen"),
                    )
                    self.assertEqual(
                        _snapshot_without_p11(production, "production"),
                        _snapshot_without_p11(frozen, "frozen"),
                    )
                    conflict_states.update(
                        str(value.get("conflict_state"))
                        for value in production.debug_p11_diagnostics(
                            "production"
                        ).get("breakdowns", {}).values()
                    )

            snapshot = production.debug_snapshot("production")
            records = snapshot["slot_ledger"]["records"]
            self.assertGreaterEqual(snapshot["override_count"], 3)
            self.assertTrue(
                any(
                    record["value"] == "red"
                    and record["polarity"] == 1
                    and record["status"] == "superseded"
                    for record in records
                )
            )
            self.assertTrue(any(record["polarity"] == -1 for record in records))
            self.assertIn("unknown", conflict_states)
        finally:
            production.close()
            frozen.close()

    def test_active_concurrent_sessions_match_sequential_reference(self) -> None:
        catalog = _official_catalog()
        if catalog is None:
            self.skipTest("official catalog is not available in this checkout")

        transcripts = {
            "dress": (
                "I'm looking for women's cotton dresses. I do not want red.",
                "What matters is a casual blue style.",
                "Actually, replace cotton with linen.",
            ),
            "shoe": (
                "I'm looking for men's running shoes. A key requirement is lightweight mesh.",
                "I do not want leather.",
                "Actually, replace black with blue.",
            ),
            "jewelry": (
                "I'm looking for silver necklaces. What matters is a modern style.",
                "No preference for brand.",
                "I do not want gold.",
            ),
            "coat": (
                "I'm looking for women's winter coats. I need wool and a classic style.",
                "I do not want polyester.",
                "Actually, replace wool with fleece.",
            ),
        }

        def execute(
            agent: Agent, session_id: str, messages: tuple[str, ...]
        ) -> tuple[list[dict], list[dict[str, list[str]]], dict]:
            agent.reset(session_id, {})
            responses: list[dict] = []
            routes: list[dict[str, list[str]]] = []
            for turn, message in enumerate(messages, start=1):
                responses.append(agent.respond(session_id, message, turn, 10))
                routes.append(agent.debug_rankings(session_id))
            return responses, routes, _snapshot_without_p11(agent, session_id)

        reference = Agent(catalog, p11_mode="active")
        parallel_agent = Agent(catalog, p11_mode="active")
        try:
            expected = {
                name: execute(reference, f"reference-{name}", messages)
                for name, messages in transcripts.items()
            }

            def run(item: tuple[str, tuple[str, ...]]) -> tuple[str, tuple]:
                name, messages = item
                return name, execute(
                    parallel_agent, f"concurrent-{name}", messages
                )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(transcripts)
            ) as pool:
                observed = dict(pool.map(run, transcripts.items()))

            self.assertEqual(observed, expected)
            stats = parallel_agent._p11_bridge.status()["stats"]
            self.assertEqual(
                stats["turns"], sum(len(messages) for messages in transcripts.values())
            )
            self.assertEqual(stats["fallbacks"], 0)
            self.assertLessEqual(stats["maximum_rows_per_fetch"], 10)
        finally:
            reference.close()
            parallel_agent.close()

    def test_every_turn_obeys_response_contract_and_catalog_identity(self) -> None:
        catalog = _official_catalog()
        if catalog is None:
            self.skipTest("official catalog is not available in this checkout")

        with catalog.open(encoding="utf-8") as handle:
            catalog_ids = {
                str(json.loads(line)["parent_asin"])
                for line in handle
                if line.strip()
            }
        self.assertEqual(len(catalog_ids), 50_000)
        agent = Agent(catalog, p11_mode="active")
        turns = (
            ("I'm looking for women's blue cotton dresses.", 1),
            ("I do not want polyester.", 3),
            ("What matters is a casual summer style.", 10),
            ("Actually, replace blue with red.", 20),
            ("No preference for brand.", 7),
            ("Actually, replace cotton with linen.", 10),
            ("I do not want wool.", 10),
            ("The key requirement is a red linen beach dress.", 10),
            ("No preference for size.", 10),
            ("Please show the strongest matches now.", 10),
        )
        try:
            agent.reset("contract", {})
            for turn, (message, top_k) in enumerate(turns, start=1):
                with self.subTest(turn=turn, top_k=top_k):
                    response = agent.respond("contract", message, turn, top_k)
                    _assert_response_contract(
                        self,
                        response,
                        catalog_ids=catalog_ids,
                        top_k=top_k,
                    )
                    rankings = agent.debug_rankings("contract")
                    self.assertEqual(
                        _recommendation_ids(response),
                        rankings["final"][: min(top_k, 10)],
                    )
                    diagnostics = agent.debug_p11_diagnostics("contract")
                    self.assertTrue(diagnostics["top10_membership_preserved"])
                    self.assertTrue(diagnostics["tail_preserved"])

            stats = agent._p11_bridge.status()["stats"]
            self.assertEqual(stats["turns"], len(turns))
            self.assertEqual(stats["fallbacks"], 0)
            self.assertEqual(sum(stats["reason_counts"].values()), len(turns))
        finally:
            agent.close()


if __name__ == "__main__":
    unittest.main()
