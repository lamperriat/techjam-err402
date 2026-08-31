from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
from types import SimpleNamespace
import unittest

from scripts import probe_sparse_multiview_g0 as probe
from scripts import sparse_multiview_g0_worker as worker
from starter import sparse_multiview_g0 as sparse
from starter.attributes import AttributeValue, ProductAttributeView
from starter.slot_ledger import ACTIVE, DELETED, SUPERSEDED


def _record(
    slot: str,
    value: str,
    *,
    polarity: int = 1,
    hardness: str = "hard",
    version: int = 2,
    status: str = "active",
) -> SimpleNamespace:
    return SimpleNamespace(
        slot=slot,
        value=value,
        polarity=polarity,
        hardness=hardness,
        source_turn=1,
        version=version,
        status=status,
    )


def _attribute(
    value: str,
    *,
    source: str = "features",
    confidence: float = 1.0,
) -> AttributeValue:
    return AttributeValue(
        value=value,
        source=source,
        confidence=confidence,
        raw=value,
    )


def _pool(count: int, *, prefix: str = "P", start: int = 0) -> tuple[str, ...]:
    return tuple(f"{prefix}{index:09d}" for index in range(start, start + count))


def _product(index: int, material: str) -> dict[str, object]:
    return {
        "parent_asin": f"Z{index:09d}",
        "title": f"{material} dress",
        "categories": ["dress"],
        "features": [material, "casual"],
        "details": {"Material": material, "Style": "casual"},
        "store": "brandx",
        "description": f"synthetic {material} dress",
    }


class SparseMultiviewG0Tests(unittest.TestCase):
    def test_build_route_queries_freezes_four_field_isolated_routes(self) -> None:
        queries = sparse.build_route_queries(
            category_text="women sandals",
            active_terms=("linen", "casual", "brandx"),
            excluded_terms=(),
            current_version=2,
            records=(
                _record("material", "linen", hardness="soft"),
                _record("style", "casual", hardness="soft"),
                _record("audience", "women", hardness="soft"),
                _record("material", "cotton", version=1),
            ),
        )

        self.assertTrue(queries.activated)
        self.assertEqual(queries.category_only.route, "category_only")
        self.assertTrue(queries.category_only.terms)
        self.assertEqual(queries.exact_active.route, "exact_active")
        self.assertEqual(queries.title_store_exact.route, "title_store_exact")
        self.assertEqual(queries.full_positive.route, "full_positive")
        self.assertNotIn(" AND ", queries.full_positive.expression.upper())
        self.assertNotIn(" AND ", queries.category_only.expression.upper())
        self.assertIn("{title categories}", queries.category_only.expression)
        self.assertIn("linen", queries.full_positive.terms)
        self.assertIn("casual", queries.full_positive.terms)
        self.assertTrue(queries.exact_active.activated)
        self.assertNotIn(" AND ", queries.exact_active.expression.upper())
        self.assertNotIn(" AND ", queries.title_store_exact.expression.upper())
        self.assertIn("{title store}", queries.title_store_exact.expression)
        self.assertIn("brandx", queries.exact_active.terms)

    def test_compile_hard_conflict_rules_uses_current_version_and_hard_records_only(self) -> None:
        rules = sparse.compile_hard_conflict_rules(
            category_text="women sandals",
            active_terms=("linen",),
            excluded_terms=("red",),
            current_version=2,
            records=(
                _record("color", "red", polarity=-1, hardness="hard", version=2),
                _record("material", "linen", polarity=1, hardness="hard", version=2),
                _record("style", "casual", polarity=1, hardness="soft", version=2),
                _record("color", "blue", polarity=-1, hardness="hard", version=1),
            ),
        )

        self.assertIn(("color", "red"), rules.negative)
        positive = dict(rules.positive)
        self.assertEqual(positive["material"], ("linen",))
        self.assertNotIn("style", positive)

    def test_fuse_route_candidates_uses_exact_rrf_and_stable_route_tiebreak(self) -> None:
        prefix = tuple(f"P{i:03d}" for i in range(100))
        fused = sparse.fuse_route_candidates(
            prefix,
            ("A", "B", "C"),
            ("C", "A"),
            ("B", "D"),
            ("D", "A"),
        )

        self.assertEqual(fused.prefix, prefix)
        self.assertEqual(fused.tail[:4], ("A", "B", "D", "C"))
        by_id = {item.identifier: item for item in fused.items}
        self.assertEqual(by_id["A"].supporting_route_count, 3)
        self.assertEqual(by_id["A"].full_positive_rank, 1)
        self.assertEqual(by_id["A"].exact_active_rank, 2)
        self.assertEqual(by_id["A"].title_store_exact_rank, 2)
        self.assertEqual(by_id["B"].category_only_rank, 1)

    def test_masked_route_rank_holes_are_preserved_in_exact_rrf(self) -> None:
        prefix = _pool(100)
        fused = sparse.fuse_route_candidates(
            prefix,
            (("A", 1), ("C", 3)),
            (("A", 5),),
            (),
            (),
        )

        by_id = {item.identifier: item for item in fused.items}
        self.assertEqual(by_id["A"].score, Fraction(1, 61) + Fraction(1, 65))
        self.assertEqual(by_id["A"].exact_active_rank, 5)
        self.assertEqual(by_id["C"].score, Fraction(1, 63))
        self.assertEqual(by_id["C"].full_positive_rank, 3)
        self.assertEqual(fused.tail[:2], ("A", "C"))

    def test_fusion_caps_complete_union_at_400_without_touching_prefix(self) -> None:
        prefix = _pool(100)
        routes = tuple(
            tuple((f"{route}{rank:03d}", rank) for rank in range(1, 121))
            for route in ("A", "B", "C", "D")
        )
        fused = sparse.fuse_route_candidates(prefix, *routes)

        self.assertEqual(len(fused.candidates), 400)
        self.assertEqual(fused.candidates[:100], prefix)
        self.assertEqual(len(fused.candidates), len(set(fused.candidates)))

    def test_route_term_cap_is_24_after_global_lexical_union(self) -> None:
        terms = tuple(f"term{index:02d}value" for index in range(30, 0, -1))
        queries = sparse.build_route_queries(
            category_text="other",
            active_terms=terms,
            excluded_terms=(),
            current_version=1,
            records=(),
        )

        expected = tuple(sorted(terms))[:24]
        self.assertEqual(sparse.TERM_LIMIT, 24)
        self.assertEqual(queries.exact_active.terms, expected)
        self.assertEqual(queries.title_store_exact.terms, expected)
        self.assertEqual(queries.full_positive.terms, expected)

    def test_retrieval_uses_active_current_soft_records_but_mask_does_not(self) -> None:
        records = (
            _record("material", "linen", hardness="soft", status=ACTIVE),
            _record("style", "formal", status=DELETED),
            _record("color", "blue", status=SUPERSEDED),
            _record("closure", "zipper", version=1),
        )
        queries = sparse.build_route_queries(
            category_text="other",
            active_terms=(),
            excluded_terms=(),
            current_version=2,
            records=records,
        )
        rules = sparse.compile_hard_conflict_rules(
            category_text="other",
            active_terms=(),
            excluded_terms=(),
            current_version=2,
            records=records,
        )

        self.assertIn("linen", queries.full_positive.terms)
        self.assertNotIn("formal", queries.full_positive.terms)
        self.assertNotIn("blue", queries.full_positive.terms)
        self.assertNotIn("zipper", queries.full_positive.terms)
        self.assertNotIn("material", dict(rules.positive))

    def test_hard_mask_keeps_unknown_and_drops_only_reliable_conflicts(self) -> None:
        identifiers = ("unknown", "weak", "description", "reliable", "details")
        views = {
            "unknown": ProductAttributeView(parent_asin="unknown"),
            "weak": ProductAttributeView(
                parent_asin="weak",
                material=(_attribute("cotton", source="title", confidence=0.82),),
            ),
            "description": ProductAttributeView(
                parent_asin="description",
                material=(_attribute("cotton", source="description", confidence=1.0),),
            ),
            "reliable": ProductAttributeView(
                parent_asin="reliable",
                material=(_attribute("cotton", source="features", confidence=0.90),),
            ),
            "details": ProductAttributeView(
                parent_asin="details",
                material=(_attribute("cotton", source="details.material", confidence=0.90),),
            ),
        }
        rules = sparse.HardConflictRules(positive=(("material", ("linen",)),))

        masked = sparse.apply_hard_conflict_mask(identifiers, views, rules)

        self.assertEqual(masked.identifiers, ("unknown", "weak", "description"))
        self.assertEqual(masked.dropped, ("reliable", "details"))
        self.assertEqual(masked.positive_conflict_count, 2)

    def test_enabled_synthetic_runtime_preserves_prefix_mask_and_cache_semantics(self) -> None:
        catalog = Path(self._tmpdir.name) / "catalog.jsonl"
        catalog.write_text(
            "".join(
                json.dumps(product, sort_keys=True, separators=(",", ":")) + "\n"
                for product in (_product(1, "linen"), _product(2, "cotton"))
            ),
            encoding="utf-8",
        )
        prefix = _pool(100)
        runtime = sparse.SparseMultiviewG0Expander(
            catalog,
            enabled=True,
            cache_enabled=True,
        )
        try:
            first = runtime.expand(
                prefix,
                category_text="dress",
                active_terms=("linen", "brandx"),
                excluded_terms=(),
                current_version=2,
                records=(_record("material", "linen"),),
            )
            second = runtime.expand(
                prefix,
                category_text="dress",
                active_terms=("linen", "brandx"),
                excluded_terms=(),
                current_version=2,
                records=(_record("material", "linen"),),
            )
            diagnostics = runtime.cache_diagnostics()
        finally:
            runtime.close()

        self.assertEqual(first.candidates[:100], prefix)
        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual(first.fusion_items, second.fusion_items)
        self.assertGreater(first.conflict_count, 0)
        self.assertEqual(first.tail_conflict_count, 0)
        for route_name, ranked in (
            ("full_positive", first.full_positive_filtered_ranked),
            ("exact_active", first.exact_active_filtered_ranked),
            ("category_only", first.category_only_filtered_ranked),
            ("title_store_exact", first.title_store_exact_filtered_ranked),
        ):
            route = getattr(first, f"{route_name}_route")
            self.assertTrue(all(route[rank - 1] == identifier for identifier, rank in ranked))
        self.assertEqual(tuple(name for name, _value in first.route_latency_ns), sparse.ROUTE_NAMES)
        self.assertTrue(all(value > 0 for _name, value in first.route_latency_ns))
        self.assertGreater(first.hard_mask_latency_ns, 0)
        self.assertGreater(diagnostics["fts_route"]["hits"], 0)
        self.assertGreater(diagnostics["product_view"]["hits"], 0)
        self.assertGreater(diagnostics["mask_decision"]["hits"], 0)

    def test_invalid_prefix_and_rank_shapes_fail_closed(self) -> None:
        with self.assertRaises(sparse.SparseMultiviewG0ValidationError):
            sparse.fuse_route_candidates(_pool(99), (), (), (), ())
        with self.assertRaises(sparse.SparseMultiviewG0ValidationError):
            sparse.fuse_route_candidates(_pool(100), (("A", 2), ("B", 1)), (), (), ())
        with self.assertRaises(sparse.SparseMultiviewG0ValidationError):
            sparse.fuse_route_candidates(_pool(100), (("A", 121),), (), (), ())

    def test_disabled_expander_preserves_prefix_and_emits_empty_routes(self) -> None:
        prefix = tuple(f"P{i:03d}" for i in range(100))
        runtime = sparse.SparseMultiviewG0Expander(Path("."), enabled=False)
        try:
            result = runtime.expand(
                prefix,
                category_text="women sandals",
                active_terms=("linen",),
                excluded_terms=(),
                current_version=1,
                records=(),
            )
        finally:
            runtime.close()

        self.assertFalse(result.enabled)
        self.assertEqual(result.candidates, prefix)
        self.assertEqual(result.tail, ())
        self.assertEqual(result.full_positive_route, ())
        self.assertEqual(result.exact_active_route, ())
        self.assertEqual(result.category_only_route, ())
        self.assertEqual(result.title_store_exact_route, ())

    def test_worker_validates_four_route_contract_and_prefix_integrity(self) -> None:
        prefix = tuple(f"P{i:03d}" for i in range(100))
        full_positive = ("A", "B", "C")
        exact_active = ("C", "A")
        category_only = ("B", "D")
        title_store_exact = ("D", "A")
        fusion = sparse.fuse_route_candidates(
            prefix,
            tuple((identifier, rank) for rank, identifier in enumerate(full_positive, 1)),
            tuple((identifier, rank) for rank, identifier in enumerate(exact_active, 1)),
            tuple((identifier, rank) for rank, identifier in enumerate(category_only, 1)),
            tuple((identifier, rank) for rank, identifier in enumerate(title_store_exact, 1)),
        )
        result = SimpleNamespace(
            enabled=True,
            activated=True,
            candidates=fusion.candidates,
            prefix=prefix,
            full_positive_route=full_positive,
            exact_active_route=exact_active,
            category_only_route=category_only,
            title_store_exact_route=title_store_exact,
            full_positive_novel=full_positive,
            exact_active_novel=exact_active,
            category_only_novel=category_only,
            title_store_exact_novel=title_store_exact,
            full_positive_filtered=full_positive,
            exact_active_filtered=exact_active,
            category_only_filtered=category_only,
            title_store_exact_filtered=title_store_exact,
            full_positive_filtered_ranked=(("A", 1), ("B", 2), ("C", 3)),
            exact_active_filtered_ranked=(("C", 1), ("A", 2)),
            category_only_filtered_ranked=(("B", 1), ("D", 2)),
            title_store_exact_filtered_ranked=(("D", 1), ("A", 2)),
            tail=fusion.tail,
            fusion_items=fusion.items,
            conflict_count=0,
            tail_conflict_count=0,
            multiroute_support_count=fusion.multiroute_support_count,
            union_novel_count=len(fusion.items),
            route_latency_ns=tuple((route, 1) for route in sparse.ROUTE_NAMES),
            hard_mask_latency_ns=1,
            legacy_route_executions=0,
            queries=SimpleNamespace(
                full_positive=SimpleNamespace(activated=True),
                exact_active=SimpleNamespace(activated=True),
                category_only=SimpleNamespace(activated=True),
                title_store_exact=SimpleNamespace(activated=True),
            ),
        )

        candidates = worker.validate_expansion_result(
            result,
            prefix,
            set(prefix) | {"A", "B", "C", "D"},
        )
        self.assertEqual(candidates, result.candidates)

        tampered = SimpleNamespace(**vars(result))
        tampered.full_positive_filtered_ranked = (("A", 2), ("B", 3), ("C", 4))
        with self.assertRaises(worker.SparseMultiviewG0WorkerError):
            worker.validate_expansion_result(
                tampered,
                prefix,
                set(prefix) | {"A", "B", "C", "D"},
            )

    def test_worker_route_contract_accepts_four_route_diagnostics(self) -> None:
        route = {
            "full_positive_route_executions": 1,
            "exact_active_route_executions": 1,
            "category_only_route_executions": 1,
            "title_store_exact_route_executions": 1,
            "legacy_route_executions": 0,
            "registry_sha256": worker.EXPECTED_ATTRIBUTE_REGISTRY_SHA256,
            "closed": False,
        }

        self.assertEqual(worker._route_contract(route), route)
        with self.assertRaises(worker.SparseMultiviewG0WorkerError):
            worker._route_contract({**route, "legacy_route_executions": 1})

    def test_probe_runtime_cleanup_uses_stable_directory_identity(self) -> None:
        tmp_path = Path(self._tmpdir.name)
        runtime_base = tmp_path / "runtime"
        runtime_base.mkdir()
        original_base = probe.RUNTIME_BASE
        probe.RUNTIME_BASE = runtime_base
        try:
            root = runtime_base / "v222b-test"
            root.mkdir()
            identity = probe._directory_identity(root.stat())
            (root / "pycache").mkdir()
            (root / "temp").mkdir()
            probe._cleanup_runtime_path(root, identity, None)
            self.assertFalse(root.exists())
        finally:
            probe.RUNTIME_BASE = original_base

    def test_probe_runtime_cleanup_accepts_only_verified_trace_hardlinks(self) -> None:
        tmp_path = Path(self._tmpdir.name)
        runtime_base = tmp_path / "runtime-hardlink"
        runtime_base.mkdir()
        original_base = probe.RUNTIME_BASE
        probe.RUNTIME_BASE = runtime_base
        try:
            nonce = "a" * 32
            root = runtime_base / f"v222b-{nonce}"
            root.mkdir()
            identity = probe._directory_identity(root.stat())
            (root / "pycache").mkdir()
            (root / "temp").mkdir()
            partial = root / f".trace-{nonce}.jsonl.{nonce}.partial"
            final = root / f"trace-{nonce}.jsonl"
            partial.write_bytes(b"{}\n")
            os.link(partial, final)
            self.assertTrue(os.path.samefile(partial, final))
            probe._cleanup_runtime_path(root, identity, final)
            self.assertFalse(root.exists())
        finally:
            probe.RUNTIME_BASE = original_base

    def test_probe_prereg_key_matches_v222b_choreography(self) -> None:
        source = Path(probe.__file__).read_text(encoding="utf-8")
        self.assertIn(
            "implementation_exact_cumulative_changed_paths_no_renames", source
        )
        self.assertNotIn("implementation_exact_changed_paths", source)

    def test_prereg_freezes_cpu_multiview_and_six_new_paths(self) -> None:
        prereg = json.loads(
            (
                Path(__file__).parents[1]
                / "configs"
                / "small_ranker_v2_22.multiview_sparse_rrf_g0_preregistration.json"
            ).read_text(encoding="utf-8")
        )
        paths = prereg["checkpoint_choreography"][
            "implementation_exact_cumulative_changed_paths_no_renames"
        ]

        self.assertEqual(prereg["normalization"]["term_cap_per_route"], 24)
        self.assertEqual(prereg["device_selection"]["selected"], "CPU")
        self.assertFalse(prereg["device_selection"]["gpu_used"])
        self.assertEqual(len(paths), 6)
        self.assertTrue(all("sparse_multiview_g0" in path or "v222" in path for path in paths))
        self.assertNotIn(
            "EXPANDED_FIXED_K200",
            prereg["candidate_recall_receipt"]["baseline_counts"],
        )
        self.assertEqual(
            prereg["protected_data_and_git"]["formal_git_executable"]["path"],
            "C:/Program Files/Git/mingw64/bin/git.exe",
        )

    def test_formal_sources_have_v222b_prereg_identity(self) -> None:
        root = Path(__file__).parents[1]
        sources = "\n".join(
            (root / relative).read_text(encoding="utf-8")
            for relative in (
                "scripts/probe_sparse_multiview_g0.py",
                "scripts/v222_safe_bootstrap.py",
                "scripts/run_v222_preflight.ps1",
            )
        )

        self.assertNotIn("techjam-v2-21-gloss-g0", sources)
        self.assertNotIn("b81351a0657411ab04810bb4740b35b407d175cc", sources)
        self.assertIn("a2e324e21730cf1c243dcc2647a894b30ad515d2", sources)
        self.assertIn("small-ranker-v2.22b-sparse-multiview", sources)
        self.assertIn("mingw64", sources)
        self.assertIn("include.path=/dev/null", sources)
        self.assertNotIn("include.path=NUL", sources)
        self.assertEqual(
            probe.PREREG_COMMIT,
            "a2e324e21730cf1c243dcc2647a894b30ad515d2",
        )
        self.assertEqual(
            probe.PREREG_CORRECTION_CHAIN,
            (
                "14ac9f0b90b5dd6dbb9cc799ba99f6a1c8b0c0e5",
                "68a84ab49c670716f65df24dd260724e00ba0661",
                "eaae35b32d5ee143b317872c60b230863b5c8e29",
                "a2e324e21730cf1c243dcc2647a894b30ad515d2",
            ),
        )
        self.assertEqual(probe.PREREG_CORRECTION_CHAIN[-1], probe.PREREG_COMMIT)
        self.assertEqual(
            probe.PREREG_BLOB,
            "e534bc7a9a304a03869e951f290fa2b96d51dee7",
        )

    def test_v222b_identity_and_git_controls_are_frozen(self) -> None:
        self.assertEqual(
            probe.SCHEMA_VERSION,
            "small-ranker-v2.22b-multiview-sparse-rrf-g0-probe.v1",
        )
        self.assertEqual(
            probe.WORKER_SCHEMA_VERSION,
            "small-ranker-v2.22b-multiview-sparse-rrf-g0-worker-summary.v1",
        )
        self.assertEqual(probe.WORKER_SCHEMA_VERSION, worker.SCHEMA_VERSION)
        self.assertEqual(
            worker.EXPECTED_PREREGISTRATION_BLOB_SHA1,
            "e534bc7a9a304a03869e951f290fa2b96d51dee7",
        )
        self.assertEqual(
            probe.EXPERIMENT_ID,
            "SR-V2.22B-TARGET-BLIND-MULTIVIEW-SPARSE-RRF-G0",
        )
        self.assertEqual(str(probe.RUNTIME_BASE), r"D:\tiktok\.v222b_runtime")
        self.assertEqual(
            str(worker.EXPECTED_RUNTIME_ROOT), r"D:\tiktok\.v222b_runtime"
        )
        self.assertEqual(probe.GIT_PREFIX[-1], "include.path=/dev/null")
        gitdir = probe.EXPECTED_GITDIR
        self.assertEqual(
            probe.EXPECTED_GIT_CONTROL_FILES[gitdir / "gitdir"],
            {
                "bytes": 47,
                "sha256": "094e1cea6a66a0e4a994dbd565bc102377d8ef072a50c23fa259345800de595c",
            },
        )
        self.assertEqual(
            probe.EXPECTED_GIT_CONTROL_FILES[gitdir / "commondir"],
            {
                "bytes": 6,
                "sha256": "340ddcb67a6204f742cd1e28e5b462622dde7daaa8ee36001897196aacdc6d47",
            },
        )
        self.assertEqual(
            probe.EXPECTED_GIT_CONTROL_FILES[gitdir / "HEAD"],
            {
                "bytes": 53,
                "sha256": "a632024088a4dc2054b446066d167e9641cfec37f1db8c09b937eaf2d3dbe62a",
            },
        )
        self.assertIn("v2_22b", probe.PREFLIGHT_CLAIM_PATH.name)
        self.assertIn("v2_22b", probe.PREFLIGHT_OUTER_PATH.name)
        self.assertIn("v2_22b", probe.PREFLIGHT_RESULT_PATH.name)
        self.assertIn("v2_22b", probe.CANDIDATE_CLAIM_PATH.name)

    def test_git_lf_prereg_identity_normalizes_crlf(self) -> None:
        path = Path(self._tmpdir.name) / "prereg.json"
        git_lf = b'{\n  "status": "frozen"\n}\n'
        path.write_bytes(git_lf.replace(b"\n", b"\r\n"))
        normalized, identity = probe._git_lf_identity(
            path,
            {
                "bytes": len(git_lf),
                "sha256": hashlib.sha256(git_lf).hexdigest(),
            },
        )
        self.assertEqual(normalized, git_lf)
        self.assertEqual(identity.rows, 3)
        self.assertEqual(probe._git_blob_from_raw(normalized), probe._git_blob_from_raw(git_lf))

    def test_v222b_guards_old_v222_namespaces(self) -> None:
        old_artifact = (
            r"D:\tiktok\techjam-v2-22b-sparse-multiview\experiments\fast_track"
            r"\small_ranker_v2_22_preflight_20260831.json"
        )
        current_artifact = old_artifact.replace("v2_22_", "v2_22b_")
        guard = probe._FormalAuditGuard()
        with self.assertRaises(PermissionError):
            guard.hook("open", (old_artifact,))
        guard.hook("open", (current_artifact,))
        with self.assertRaises(worker.SparseMultiviewG0WorkerError):
            worker._guard_legacy_namespaces(PureWindowsPath(old_artifact))
        with self.assertRaises(worker.SparseMultiviewG0WorkerError):
            worker._guard_legacy_namespaces(
                PureWindowsPath(r"D:\tiktok\small-ranker-v2.22-sparse-multiview")
            )
        worker._guard_legacy_namespaces(PureWindowsPath(current_artifact))

    def test_bootstrap_points_at_v222b_worktree_and_modules(self) -> None:
        self.assertEqual(
            probe.BRANCH, "small-ranker-v2.22b-sparse-multiview"
        )
        self.assertEqual(
            bootstrap_project_root(), "D:/tiktok/techjam-v2-22b-sparse-multiview"
        )
        self.assertEqual(
            probe.RUNNER_RELATIVE, "scripts/probe_sparse_multiview_g0.py"
        )
        self.assertEqual(
            probe.WORKER_RELATIVE, "scripts/sparse_multiview_g0_worker.py"
        )

    def test_runner_and_worker_direct_module_entrypoints_match(self) -> None:
        root = Path(__file__).parents[1]
        expected = {
            "c200_contract_imported": True,
            "evaluator_imported": True,
            "legacy_runtime_absent": True,
            "project_root_bootstrapped": True,
            "required_module": "starter.sparse_multiview_g0",
            "status": "ENTRYPOINT_SELF_CHECK_PASS",
        }
        for script, module in (
            (
                "scripts/sparse_multiview_g0_worker.py",
                "scripts.sparse_multiview_g0_worker",
            ),
            (
                "scripts/probe_sparse_multiview_g0.py",
                "scripts.probe_sparse_multiview_g0",
            ),
        ):
            receipts = []
            for command in (
                (sys.executable, script, "--entrypoint-self-check"),
                (sys.executable, "-m", module, "--entrypoint-self-check"),
            ):
                completed = subprocess.run(
                    command,
                    cwd=root,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"{command!r}: {completed.stderr!r}",
                )
                self.assertEqual(completed.stderr, "")
                receipts.append(json.loads(completed.stdout))
            self.assertEqual(receipts, [expected, expected])

    def setUp(self) -> None:
        self._tmpdir = __import__("tempfile").TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()


def bootstrap_project_root() -> str:
    from scripts import v222_safe_bootstrap as bootstrap

    return bootstrap.PROJECT_ROOT
