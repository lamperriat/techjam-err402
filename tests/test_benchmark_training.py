from __future__ import annotations

import csv
import io
import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import optuna

from agents.v1_tuned import AgentV1Tuned
from benchmark_training.generate_dataset import balanced_session_targets
from benchmark_training.generate_dataset import main as generate_main
from benchmark_training.generate_dataset import stratified_product_partition
from benchmark_training.tune_v1 import main as tune_main
from benchmark_training.tune_v1 import (
    build_parser as build_tuning_parser,
    select_product_unique_sessions,
    session_objective,
    study_signature,
    suggest_scoring_config,
    validate_dataset_boundaries,
)
from retrieval.scoring import INTENT_WEIGHTS, ScoringConfig
from tests.test_v1 import product_row


class BenchmarkTrainingTest(unittest.TestCase):
    def test_fast_tuning_defaults_fit_the_runtime_budget(self) -> None:
        args = build_tuning_parser().parse_args([])

        self.assertEqual(args.trials, 20)
        self.assertEqual(args.validation_candidates, 3)
        self.assertEqual(
            (args.train_sessions, args.validation_sessions, args.test_sessions),
            (500, 150, 150),
        )

    def test_product_split_is_deterministic_disjoint_and_complete(self) -> None:
        products = [f"P{index:03d}" for index in range(120)]
        categories = {
            product: ("clothing", "shoes", "jewelry")[index % 3]
            for index, product in enumerate(products)
        }

        first = stratified_product_partition(products, categories, seed=17)
        second = stratified_product_partition(products, categories, seed=17)

        self.assertEqual(first, second)
        target_sets = {name: set(split) for name, split in first.items()}
        self.assertEqual(sum(map(len, target_sets.values())), len(products))
        self.assertFalse(target_sets["train"] & target_sets["validation"])
        self.assertFalse(target_sets["train"] & target_sets["test"])
        self.assertFalse(target_sets["validation"] & target_sets["test"])

    def test_repeated_session_targets_are_balanced(self) -> None:
        targets = balanced_session_targets(
            [f"P{index}" for index in range(10)],
            session_count=27,
            rng=random.Random(4),
        )
        counts = Counter(targets)

        self.assertEqual(len(targets), 27)
        self.assertEqual(set(counts.values()), {2, 3})

    def test_session_objective_matches_benchmark_formula(self) -> None:
        result = {
            "sessions": [
                {
                    "hit": True,
                    "reciprocal_rank": 0.5,
                    "first_hit_turn": 3,
                },
                {
                    "hit": False,
                    "reciprocal_rank": 0.0,
                    "first_hit_turn": None,
                },
            ]
        }

        self.assertAlmostEqual(session_objective(result), (0.81 + 0.0) / 2)

    def test_dataset_validation_rejects_target_leakage(self) -> None:
        def sample(sample_id: str, parent_asin: str) -> dict:
            return {
                "sample_id": sample_id,
                "ground_truth": {"parent_asin": parent_asin},
            }

        with self.assertRaisesRegex(ValueError, "train and validation"):
            validate_dataset_boundaries(
                {
                    "train": [sample("train", "A")],
                    "validation": [sample("validation", "A")],
                    "test": [sample("test", "B")],
                },
                [sample("public", "C")],
            )

    def test_fast_subset_prefers_unique_products_and_is_deterministic(self) -> None:
        samples = [
            {
                "sample_id": f"S{target}_{repeat}",
                "ground_truth": {"parent_asin": f"P{target}"},
            }
            for target in range(10)
            for repeat in range(3)
        ]

        first = select_product_unique_sessions(samples, limit=8, seed=5)
        second = select_product_unique_sessions(samples, limit=8, seed=5)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(
            len({sample["ground_truth"]["parent_asin"] for sample in first}),
            8,
        )

    def test_study_signature_changes_with_selected_sessions(self) -> None:
        first = study_signature(
            {
                "train": [{"sample_id": "A"}],
                "validation": [{"sample_id": "B"}],
                "test": [{"sample_id": "C"}],
            }
        )
        second = study_signature(
            {
                "train": [{"sample_id": "D"}],
                "validation": [{"sample_id": "B"}],
                "test": [{"sample_id": "C"}],
            }
        )

        self.assertNotEqual(first, second)

    def test_baseline_trial_parameters_reproduce_default_scoring(self) -> None:
        parameters = {
            f"{intent}_{component}_multiplier": 1.0
            for intent, weights in INTENT_WEIGHTS.items()
            for component in weights
        }
        parameters.update(
            {"price_weight": 0.15, "bayesian_confidence_scale": 1.0}
        )

        config = suggest_scoring_config(optuna.trial.FixedTrial(parameters))

        self.assertEqual(config, ScoringConfig.default())

    def test_tuned_agent_loads_validated_scoring_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            config_path = root / "config.json"
            catalog_path.write_text(
                "".join(json.dumps(product_row(index, "Shoes")) + "\n" for index in range(3)),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "scoring": ScoringConfig.default().to_dict(),
                    }
                ),
                encoding="utf-8",
            )

            agent = AgentV1Tuned(catalog_path, config_path)
            self.addCleanup(agent.close)

        self.assertEqual(agent.scorer.config, ScoringConfig.default())

    def test_scoring_config_rejects_incomplete_weights(self) -> None:
        value = ScoringConfig.default().to_dict()
        del value["intent_weights"]["buying"]["category"]

        with self.assertRaisesRegex(ValueError, "must define"):
            ScoringConfig.from_dict(value)

    def test_generator_and_tuner_dry_run_integrate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            records_path = root / "records.csv"
            public_path = root / "public.jsonl"
            data_dir = root / "splits"
            results_dir = root / "results"
            products = []
            for index in range(50):
                item = product_row(index % 6, "Shoes")
                item["parent_asin"] = f"P{index}"
                products.append(item)
            catalog_path.write_text(
                "".join(json.dumps(item) + "\n" for item in products),
                encoding="utf-8",
            )
            with records_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["parent_asin", "rating", "history"],
                )
                writer.writeheader()
                for index in range(50):
                    writer.writerow(
                        {
                            "parent_asin": f"P{index}",
                            "rating": "5",
                            "history": "H1 H2",
                        }
                    )
            public_path.write_text(
                json.dumps(
                    {
                        "sample_id": "public_0001",
                        "category_bucket": "shoes",
                        "difficulty_bucket": "easy",
                        "ground_truth": {"parent_asin": "P0"},
                        "scenario_type": "buying",
                        "user_profile": {"preference_tags": ["comfort"]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "generate_dataset.py",
                    "--records",
                    str(records_path),
                    "--catalog",
                    str(catalog_path),
                    "--public-set",
                    str(public_path),
                    "--output-dir",
                    str(data_dir),
                    "--samples",
                    "120",
                    "--quiet",
                ],
            ), redirect_stdout(io.StringIO()) as output:
                generate_main()

            manifest = json.loads(output.getvalue())
            self.assertEqual(manifest["eligible_product_count"], 49)
            self.assertEqual(manifest["generated_sample_count"], 120)
            self.assertTrue(manifest["balanced_repeated_targets"])

            with patch.object(
                sys,
                "argv",
                [
                    "tune_v1.py",
                    "--catalog",
                    str(catalog_path),
                    "--data-dir",
                    str(data_dir),
                    "--public-set",
                    str(public_path),
                    "--output-dir",
                    str(results_dir),
                    "--dry-run",
                    "--dry-run-sessions",
                    "1",
                    "--quiet",
                ],
            ), redirect_stdout(io.StringIO()) as output:
                tune_main()

            dry_run = json.loads(output.getvalue())
            self.assertTrue(dry_run["dry_run"])
            self.assertEqual(set(dry_run["results"]), {"train", "validation", "test"})
            self.assertFalse(results_dir.exists())

            with patch.object(
                sys,
                "argv",
                [
                    "tune_v1.py",
                    "--catalog",
                    str(catalog_path),
                    "--data-dir",
                    str(data_dir),
                    "--public-set",
                    str(public_path),
                    "--output-dir",
                    str(results_dir),
                    "--trials",
                    "1",
                    "--validation-candidates",
                    "1",
                    "--quiet",
                ],
            ), redirect_stdout(io.StringIO()) as output:
                tune_main()

            completed = json.loads(output.getvalue())
            self.assertEqual(completed["completed_trials"], 1)
            self.assertTrue((results_dir / "study.db").is_file())
            tuned_agent = AgentV1Tuned(catalog_path, results_dir / "best_config.json")
            tuned_agent.close()


if __name__ == "__main__":
    unittest.main()
