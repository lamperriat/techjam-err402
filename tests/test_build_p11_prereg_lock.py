from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

from scripts import build_p11_prereg_lock as lock_builder


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(lock_builder._canonical_bytes(row) + b"\n" for row in rows))


def _session(sample_id: str, target: str, scenario: str) -> dict:
    return {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "user_profile": {
            "purchase_frequency": "not provided",
            "average_prior_rating": None,
            "rating_style": "not provided",
            "preference_tags": [],
            "summary": "fixture",
        },
        "ground_truth": {"parent_asin": target},
    }


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        for relative in lock_builder.SOURCE_PATHS.values():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture source\n", encoding="utf-8")

        self.catalog = self.root / "data" / "catalog.jsonl"
        catalog_rows = [
            {
                "parent_asin": f"CAT{index:07d}",
                "title": f"Product {index}",
                "features": [],
                "description": [],
                "price": None,
                "categories": ["Clothing"],
                "details": {},
                "average_rating": 4.0,
                "rating_number": index,
                "store": "Fixture",
            }
            for index in range(1, 21)
        ]
        _write_jsonl(self.catalog, catalog_rows)

        self.public = self.root / "data" / "public_set.jsonl"
        public_rows = [_session("public_0001", "CAT0000001", "buying")]
        _write_jsonl(self.public, public_rows)

        split_rows = {
            "primary": [
                _session("p_1", "CAT0000002", "buying"),
                _session("p_2", "CAT0000003", "browsing"),
            ],
            "uniform_tail": [
                _session("u_1", "CAT0000004", "boundary"),
                _session("u_2", "CAT0000005", "buying"),
            ],
            "confirmation": [
                _session("c_1", "CAT0000006", "browsing"),
                _session("c_2", "CAT0000007", "intent_override"),
            ],
            "failure_negative": [
                _session("n_1", "CAT0000008", "buying")
            ],
            "failure_budget": [
                _session("b_1", "CAT0000009", "browsing")
            ],
            "failure_override": [
                _session("o_1", "CAT0000010", "intent_override")
            ],
            "failure_missing_evidence": [
                _session("m_1", "CAT0000011", "buying")
            ],
        }
        self.corpora = {
            name: self.root / "experiments" / f"{name}.jsonl"
            for name in lock_builder.ALL_SPLITS
        }
        for name, rows in split_rows.items():
            _write_jsonl(self.corpora[name], rows)

        self.split_rows = {name: len(rows) for name, rows in split_rows.items()}
        self.split_scenarios = {
            name: dict(
                sorted(
                    {
                        scenario: sum(row["scenario_type"] == scenario for row in rows)
                        for scenario in {row["scenario_type"] for row in rows}
                    }.items()
                )
            )
            for name, rows in split_rows.items()
        }
        self.corpus_hashes = {
            name: lock_builder._sha256_file(path) for name, path in self.corpora.items()
        }

        self.protocol = self.root / "configs" / "p11_corpus_protocol.json"
        protocol = {
            "schema_version": lock_builder.CORPUS_PROTOCOL_SCHEMA_VERSION,
            "catalog": {
                "count": len(catalog_rows),
                "path": "data/catalog.jsonl",
                "sha256": lock_builder._sha256_file(self.catalog),
            },
            "opened_target_union_count": 1800,
            "opened_corpora": {
                **{
                    f"prior_{index}": {
                        "rows": 200,
                        "canonical_samples_sha256": f"{index:064x}",
                    }
                    for index in range(1, 9)
                },
                "released_public": {
                    "rows": 1,
                    "canonical_samples_sha256": lock_builder._canonical_rows_sha256(public_rows),
                },
            },
            "splits": {
                name: {
                    "count": self.split_rows[name],
                    "scenario_counts": self.split_scenarios[name],
                    "expected_samples_sha256": self.corpus_hashes[name],
                    "filename": self.corpora[name].name,
                    "sample_id_prefix": f"{name}_",
                    "seed": f"fixture-{name}",
                }
                for name in lock_builder.ALL_SPLITS
            },
        }
        _write_json(self.protocol, protocol)

        self.spec = self.root / "configs" / "p11_top10_experiment.json"
        spec = {
            "schema_version": lock_builder.SPEC_SCHEMA_VERSION,
            "artifact_policy": dict(lock_builder.EXPECTED_ARTIFACT_POLICY),
            "bootstrap": dict(lock_builder.EXPECTED_BOOTSTRAP),
            "corpus_protocol": {
                "path": "configs/p11_corpus_protocol.json",
                "schema_version": lock_builder.CORPUS_PROTOCOL_SCHEMA_VERSION,
            },
            "deadline_policy": dict(lock_builder.EXPECTED_DEADLINE_POLICY),
            "execution_order": list(lock_builder.EXPECTED_EXECUTION_ORDER),
            "feature_contract": lock_builder._feature_contract(),
            "promotion_gates": dict(lock_builder.EXPECTED_PROMOTION_GATES),
            "public_evaluation_run": False,
            "resource_limits": dict(lock_builder.EXPECTED_RESOURCE_LIMITS),
            "roles": dict(lock_builder.EXPECTED_ROLES),
            "served_control": {
                "question_policy": "fast",
                "rerank_mode": "off",
                "retrieval_mode": "coverage",
            },
            "sidecar_policy": dict(lock_builder.EXPECTED_SIDECAR_POLICY),
        }
        _write_json(self.spec, spec)

        self.evaluator = self.root / "evaluator" / "local_evaluator.py"
        self.evaluation_config = self.root / "docs" / "evaluation_config.json"
        _write_json(self.evaluation_config, lock_builder.EXPECTED_EVALUATION_CONFIG)

        self.sidecar = self.root / "experiments" / "p11_features.sqlite"
        self.sidecar.write_bytes(b"fixture-sidecar-v1")
        self.sidecar_metadata = self.root / "experiments" / "p11_features.metadata.json"
        feature = lock_builder._feature_contract()
        _write_json(
            self.sidecar_metadata,
            {
                "schema_version": feature["feature_schema_version"],
                "catalog": {
                    "rows": len(catalog_rows),
                    "sha256": lock_builder._sha256_file(self.catalog),
                },
                "sidecar": {
                    "bytes": self.sidecar.stat().st_size,
                    "sha256": lock_builder._sha256_file(self.sidecar),
                    "registry_sha256": feature["feature_registry_sha256"],
                    "semantics_sha256": feature["feature_semantics_sha256"],
                },
                "target_blind": True,
                "label_free": True,
            },
        )

        self.corpus_metadata = self.root / "experiments" / "p11_corpora.metadata.json"
        pairs = {
            f"{left}__{right}": 0
            for left, right in combinations(sorted(lock_builder.ALL_SPLITS), 2)
        }
        opened_pairs = {
            f"{left}__{right}": 0
            for left, right in combinations(sorted(protocol["opened_corpora"]), 2)
        }
        metadata = {
            "schema_version": lock_builder.CORPUS_METADATA_SCHEMA_VERSION,
            "protocol_file_sha256": lock_builder._sha256_file(self.protocol),
            "protocol_sha256": lock_builder._stable_sha256(protocol),
            "builder_source": {
                "sha256": lock_builder._sha256_file(
                    self.root / lock_builder.SOURCE_PATHS["corpus_builder"]
                )
            },
            "catalog": {
                "product_count": len(catalog_rows),
                "sha256": lock_builder._sha256_file(self.catalog),
            },
            "outputs": {
                name: {
                    "sample_count": self.split_rows[name],
                    "unique_target_count": self.split_rows[name],
                    "scenario_counts": self.split_scenarios[name],
                    "samples_sha256": self.corpus_hashes[name],
                }
                for name in lock_builder.ALL_SPLITS
            },
            "output_files": {
                name: {"sha256": self.corpus_hashes[name]}
                for name in lock_builder.ALL_SPLITS
            },
            "new_pairwise_target_overlaps": pairs,
            "new_target_union_count": sum(self.split_rows.values()),
            "opened_registry": {
                "target_union_count": 1800,
                "pairwise_target_overlaps": opened_pairs,
                "corpora": {
                    name: {
                        "rows": value["rows"],
                        "unique_targets": value["rows"],
                        "canonical_samples_sha256": value[
                            "canonical_samples_sha256"
                        ],
                    }
                    for name, value in protocol["opened_corpora"].items()
                },
            },
            "opened_vs_new_target_overlaps": {
                name: 0 for name in lock_builder.ALL_SPLITS
            },
            "selection_boundaries": {
                "confirmation_role": "unopened until candidate and weights are frozen",
                "released_public_used_for_weight_search": False,
                "evaluation_result_json_read": False,
                "agent_used": False,
            },
        }
        _write_json(self.corpus_metadata, metadata)

        self.output = self.root / "configs" / "p11_prereg_lock.json"
        self.paths = lock_builder.LockPaths(
            spec=self.spec,
            corpus_protocol=self.protocol,
            catalog=self.catalog,
            released_public=self.public,
            evaluator=self.evaluator,
            evaluation_config=self.evaluation_config,
            corpus_metadata=self.corpus_metadata,
            corpora=self.corpora,
            sidecar=self.sidecar,
            sidecar_metadata=self.sidecar_metadata,
            output=self.output,
        )
        self.expectations = lock_builder.FrozenExpectations(
            catalog_sha256=lock_builder._sha256_file(self.catalog),
            catalog_rows=len(catalog_rows),
            public_sha256=lock_builder._sha256_file(self.public),
            public_canonical_sha256=lock_builder._canonical_rows_sha256(public_rows),
            public_git_blob_sha1=lock_builder._git_blob_sha1_lf(self.public),
            public_rows=1,
            public_scenarios={"buying": 1},
            evaluator_sha256=lock_builder._sha256_file(self.evaluator),
            evaluator_git_blob_sha1=lock_builder._raw_git_blob_sha1(self.evaluator),
            evaluation_config_sha256=lock_builder._sha256_file(self.evaluation_config),
            evaluation_config_git_blob_sha1=lock_builder._raw_git_blob_sha1(
                self.evaluation_config
            ),
            split_rows=self.split_rows,
            split_scenarios=self.split_scenarios,
            corpus_sha256=self.corpus_hashes,
            opened_corpora=protocol["opened_corpora"],
            opened_target_union_count=1800,
        )

    def build(self) -> dict:
        return lock_builder.build_prereg_lock(
            project_root=self.root,
            paths=self.paths,
            expectations=self.expectations,
            enforce_git=False,
            require_defaults=False,
        )


class BuildP11PreregLockTests(unittest.TestCase):
    def test_lock_builder_does_not_import_candidate_runtime(self) -> None:
        source = Path(lock_builder.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from starter.p11_features import", source)
        self.assertEqual(
            lock_builder._feature_contract(),
            lock_builder.EXPECTED_FEATURE_CONTRACT,
        )

    def test_live_source_registry_matches_runner_and_includes_lock_builder(self) -> None:
        required = lock_builder._required_paths_from_runner(
            lock_builder.PROJECT_ROOT / "scripts" / "evaluate_p11.py"
        )
        self.assertEqual(required, lock_builder.SOURCE_PATHS)
        self.assertEqual(
            required["lock_builder"], "scripts/build_p11_prereg_lock.py"
        )
        self.assertEqual(
            set(lock_builder.RUNTIME_ASIN_SCAN_NAMES), set(lock_builder.SOURCE_PATHS)
        )

    def test_lock_has_every_required_identity_and_confirmation_is_hash_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            original = lock_builder._inspect_conversation_rows
            inspected: list[Path] = []

            def recording(path: Path):
                inspected.append(path.resolve())
                return original(path)

            with patch.object(lock_builder, "_inspect_conversation_rows", side_effect=recording):
                lock = fixture.build()

            self.assertEqual(
                set(lock),
                {
                    "schema_version",
                    "source",
                    "source_target_scan",
                    "source_asin_literal_scan",
                    "official",
                    "experiment",
                    "corpus_metadata",
                    "corpora",
                    "sidecar",
                    "sidecar_metadata",
                    "roles",
                    "feature_contract",
                    "protocol",
                },
            )
            self.assertEqual(set(lock["corpora"]), set(lock_builder.ALL_SPLITS))
            self.assertFalse(lock["corpora"]["confirmation"]["semantic_parse_executed"])
            self.assertNotIn(fixture.corpora["confirmation"].resolve(), inspected)
            self.assertEqual(lock["official"]["evaluation_config"]["parsed"]["top_k"], 10)
            self.assertEqual(lock["official"]["evaluation_config"]["parsed"]["max_turns"], 10)
            self.assertEqual(lock["roles"], lock_builder.EXPECTED_ROLES)
            self.assertEqual(
                lock["protocol"]["deadline_policy"],
                lock_builder.EXPECTED_DEADLINE_POLICY,
            )
            self.assertFalse(lock["protocol"]["public_evaluation_run"])
            self.assertTrue(lock["source_asin_literal_scan"]["passed"])
            self.assertEqual(
                lock["sidecar"]["registry_sha256"],
                lock["feature_contract"]["feature_registry_sha256"],
            )
            encoded = json.dumps(lock)
            for identifier in ("CAT0000001", "CAT0000002", "CAT0000011"):
                self.assertNotIn(identifier, encoded)

    def test_runtime_source_rejects_asin_literal_without_confirmation_parse(self) -> None:
        for source_name in ("p11_lab", "scripts_init", "lock_builder"):
            with self.subTest(source_name=source_name), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                source = fixture.root / lock_builder.SOURCE_PATHS[source_name]
                source.write_text("FROZEN = 'B012345678'\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    lock_builder.PreregLockError, "ASIN-shaped string literal"
                ):
                    fixture.build()

    def test_runtime_source_rejects_statically_constructed_asins(self) -> None:
        sources = {
            "bytes": "FROZEN = b'B012345678'\n",
            "concatenated": "FROZEN = 'B0' + '12345678'\n",
            "escaped_bytes": r"FROZEN = b'\x42\x30\x31\x32\x33\x34\x35\x36\x37\x38'"
            + "\n",
            "f_string": "FROZEN = f\"{'B0'}{'12345678'}\"\n",
            "joined": "FROZEN = ''.join(('B0', '12345678'))\n",
        }
        for name, source_text in sources.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                source = fixture.root / lock_builder.SOURCE_PATHS["p11_lab"]
                source.write_text(source_text, encoding="utf-8")
                with self.assertRaisesRegex(
                    lock_builder.PreregLockError, "ASIN-shaped string literal"
                ):
                    fixture.build()

    def test_deadline_policy_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            spec = json.loads(fixture.spec.read_text(encoding="utf-8"))
            spec["deadline_policy"]["formal_evaluation_seconds"] = 5_399
            _write_json(fixture.spec, spec)

            with self.assertRaisesRegex(
                lock_builder.PreregLockError, "global deadline policy"
            ):
                fixture.build()

    def test_corpus_or_sidecar_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.corpora["failure_budget"].write_bytes(
                fixture.corpora["failure_budget"].read_bytes() + b" \n"
            )
            with self.assertRaisesRegex(lock_builder.PreregLockError, "metadata identity"):
                fixture.build()

        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.sidecar.write_bytes(b"changed")
            with self.assertRaisesRegex(lock_builder.PreregLockError, "sidecar metadata"):
                fixture.build()

    def test_disjointness_proof_rejects_weakened_registries(self) -> None:
        def metadata_case(mutator) -> None:
            with tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                metadata = json.loads(
                    fixture.corpus_metadata.read_text(encoding="utf-8")
                )
                mutator(metadata)
                _write_json(fixture.corpus_metadata, metadata)
                with self.assertRaisesRegex(
                    lock_builder.PreregLockError, "target disjointness"
                ):
                    fixture.build()

        cases = {
            "missing_opened_observation": lambda value: value["opened_registry"][
                "corpora"
            ].pop(next(iter(value["opened_registry"]["corpora"]))),
            "empty_opened_pairwise": lambda value: value["opened_registry"].update(
                {"pairwise_target_overlaps": {}}
            ),
            "missing_cross_map": lambda value: value.pop(
                "opened_vs_new_target_overlaps"
            ),
            "nonzero_cross_overlap": lambda value: value[
                "opened_vs_new_target_overlaps"
            ].update({"primary": 1}),
        }
        for name, mutator in cases.items():
            with self.subTest(name=name):
                metadata_case(mutator)

        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
            protocol["opened_corpora"].pop(next(iter(protocol["opened_corpora"])))
            _write_json(fixture.protocol, protocol)
            with self.assertRaisesRegex(
                lock_builder.PreregLockError, "opened-corpus registry"
            ):
                fixture.build()

    def test_existing_output_blocks_build_before_git_and_atomic_create_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.output.write_text("keep", encoding="utf-8")
            with patch.object(lock_builder, "capture_pushed_clean_revision") as git:
                with self.assertRaises(FileExistsError):
                    lock_builder.build_prereg_lock(
                        project_root=fixture.root,
                        paths=fixture.paths,
                        expectations=fixture.expectations,
                        require_defaults=False,
                    )
            git.assert_not_called()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lock.json"
            lock_builder.atomic_create(output, {"schema_version": "first"})
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                lock_builder.atomic_create(output, {"schema_version": "second"})
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(output.parent.glob(".lock.json.*.tmp")), [])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lock.json"
            with patch.object(lock_builder.os, "fsync", side_effect=OSError("disk")):
                with self.assertRaisesRegex(OSError, "disk"):
                    lock_builder.atomic_create(output, {"schema_version": "failed"})
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(".lock.json.*.tmp")), [])

    def test_temporary_git_repo_rejects_dirty_and_unpushed_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            origin = root / "origin.git"
            upstream = root / "upstream.git"
            self._git(root, "init", "-b", "pre", str(repo))
            self._git(repo, "config", "user.email", "fixture@example.invalid")
            self._git(repo, "config", "user.name", "Fixture")
            (repo / "tracked.txt").write_text("one\n", encoding="utf-8")
            self._git(repo, "add", "tracked.txt")
            self._git(repo, "commit", "-m", "initial")
            self._git(root, "init", "--bare", str(origin))
            self._git(root, "init", "--bare", str(upstream))
            self._git(repo, "remote", "add", "origin", str(origin))
            self._git(repo, "remote", "add", "upstream", str(upstream))
            self._git(repo, "push", "origin", "HEAD:refs/heads/pre")
            self._git(repo, "push", "upstream", "HEAD:refs/heads/main")
            self._git(repo, "fetch", "upstream", "main:refs/remotes/upstream/main")
            head = self._git(repo, "rev-parse", "HEAD").strip()

            with (
                patch.object(lock_builder, "EXPECTED_ORIGIN_URL", str(origin)),
                patch.object(lock_builder, "EXPECTED_UPSTREAM_URL", str(upstream)),
                patch.object(lock_builder, "EXPECTED_UPSTREAM_HEAD", head),
            ):
                proof = lock_builder.capture_pushed_clean_revision(repo)
                self.assertEqual(proof["git_commit"], head)
                self.assertTrue(proof["remote_proof"]["verified"])
                self.assertTrue(proof["official_upstream"]["verified"])

                dirty = repo / "dirty.txt"
                dirty.write_text("dirty\n", encoding="utf-8")
                with self.assertRaisesRegex(lock_builder.PreregLockError, "clean worktree"):
                    lock_builder.capture_pushed_clean_revision(repo)
                dirty.unlink()

                (repo / "tracked.txt").write_text("two\n", encoding="utf-8")
                self._git(repo, "add", "tracked.txt")
                self._git(repo, "commit", "-m", "unpushed")
                with self.assertRaisesRegex(lock_builder.PreregLockError, "origin branch proof"):
                    lock_builder.capture_pushed_clean_revision(repo)

    def test_git_identity_uses_clean_filter_for_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self._git(repo, "init", "-b", "pre")
            self._git(repo, "config", "user.email", "fixture@example.invalid")
            self._git(repo, "config", "user.name", "Fixture")
            (repo / ".gitattributes").write_text("*.py text eol=lf\n", encoding="utf-8")
            source = repo / "source.py"
            source.write_bytes(b"print('ok')\r\n")
            self._git(repo, "add", ".gitattributes", "source.py")
            self._git(repo, "commit", "-m", "line endings")
            identity = lock_builder._tracked_head_identity(source, repo)
            self.assertEqual(
                identity["git_blob_sha1"], self._git(repo, "rev-parse", "HEAD:source.py").strip()
            )
            self.assertEqual(identity["sha256"], lock_builder._sha256_file(source))

    @staticmethod
    def _git(cwd: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout


if __name__ == "__main__":
    unittest.main()
