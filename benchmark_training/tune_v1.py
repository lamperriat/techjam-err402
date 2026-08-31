from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
from optuna.trial import FrozenTrial, TrialState
from tqdm.auto import tqdm

from agents.v1 import AgentV1
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from retrieval.scoring import INTENT_WEIGHTS, ProductScorer, ScoringConfig


TUNED_WEIGHT_COMPONENTS = {"lexical", "category", "constraint", "popularity"}
SEARCH_SPACE_VERSION = 3


def suggest_scoring_config(trial: optuna.Trial) -> ScoringConfig:
    """Sample normalized component weights around the documented V1 baseline."""
    intent_weights: dict = {}
    for intent, baseline_weights in INTENT_WEIGHTS.items():
        raw = {
            component: baseline
            * (
                trial.suggest_float(
                    f"{intent}_{component}_multiplier",
                    0.25,
                    4.0,
                    log=True,
                )
                if component in TUNED_WEIGHT_COMPONENTS
                else 1.0
            )
            for component, baseline in baseline_weights.items()
        }
        total = sum(raw.values())
        intent_weights[intent] = {
            component: value / total for component, value in raw.items()
        }
    return ScoringConfig(
        intent_weights=intent_weights,
        price_weight=trial.suggest_float("price_weight", 0.0, 0.35),
        bayesian_confidence_scale=trial.suggest_float(
            "bayesian_confidence_scale",
            0.25,
            4.0,
            log=True,
        ),
    )


def session_objective(result: dict) -> float:
    sessions = result["sessions"]
    if not sessions:
        return 0.0
    total = 0.0
    for session in sessions:
        first_hit_turn = session["first_hit_turn"] or 11
        total += (
            0.50 * float(session["hit"])
            + 0.30 * float(session["reciprocal_rank"])
            + 0.02 * (11 - first_hit_turn)
        )
    return total / len(sessions)


def _evaluate_config(
    agent: AgentV1,
    config: ScoringConfig,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    description: str,
    show_progress: bool,
    position: int = 0,
    leave: bool = True,
) -> dict:
    agent.scorer = ProductScorer(agent.catalog, config)
    agent._sessions.clear()
    return evaluate(
        agent,
        samples,
        catalog_ids,
        categories,
        products,
        show_progress=show_progress,
        progress_description=description,
        progress_position=position,
        progress_leave=leave,
    )


def _summary(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "sessions"}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _complete_trials(study: optuna.Study) -> list[FrozenTrial]:
    return [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and isinstance(trial.user_attrs.get("scoring"), dict)
    ]


def validate_dataset_boundaries(
    datasets: dict[str, list[dict]],
    public_samples: list[dict],
) -> None:
    target_sets: dict[str, set[str]] = {}
    sample_ids: set[str] = set()
    for name in ("train", "validation", "test"):
        samples = datasets.get(name)
        if not samples:
            raise ValueError(f"{name} split must not be empty")
        split_sample_ids = {str(sample["sample_id"]) for sample in samples}
        if len(split_sample_ids) != len(samples) or sample_ids & split_sample_ids:
            raise ValueError("generated sample IDs must be unique across splits")
        sample_ids.update(split_sample_ids)
        target_sets[name] = {
            str(sample["ground_truth"]["parent_asin"]) for sample in samples
        }
    names = tuple(target_sets)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if target_sets[names[left]] & target_sets[names[right]]:
                raise ValueError(
                    f"target products overlap between {names[left]} and {names[right]}"
                )
    public_targets = {
        str(sample["ground_truth"]["parent_asin"]) for sample in public_samples
    }
    if set().union(*target_sets.values()) & public_targets:
        raise ValueError("generated splits overlap with public target products")


def select_product_unique_sessions(
    samples: list[dict],
    limit: int,
    seed: int,
) -> list[dict]:
    """Select deterministic sessions, preferring one session per target product."""
    if limit <= 0:
        raise ValueError("session limit must be positive")
    if limit >= len(samples):
        return list(samples)

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    selected: list[dict] = []
    selected_ids: set[str] = set()
    target_ids: set[str] = set()
    for sample in shuffled:
        target = str(sample["ground_truth"]["parent_asin"])
        if target in target_ids:
            continue
        selected.append(sample)
        selected_ids.add(str(sample["sample_id"]))
        target_ids.add(target)
        if len(selected) == limit:
            return selected

    for sample in shuffled:
        sample_id = str(sample["sample_id"])
        if sample_id in selected_ids:
            continue
        selected.append(sample)
        if len(selected) == limit:
            break
    return selected


def study_signature(datasets: dict[str, list[dict]]) -> dict:
    signature = {"search_space_version": SEARCH_SPACE_VERSION, "splits": {}}
    for name, samples in datasets.items():
        identifiers = "\0".join(str(sample["sample_id"]) for sample in samples)
        signature["splits"][name] = {
            "sample_count": len(samples),
            "sample_ids_sha256": hashlib.sha256(identifiers.encode()).hexdigest(),
        }
    return signature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune V1 ranking parameters on product-disjoint generated sessions."
    )
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "local-data/v1_tuning",
    )
    parser.add_argument(
        "--public-set",
        type=Path,
        default=PROJECT_ROOT / "data/public_set.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/v1_tuning",
    )
    parser.add_argument("--study-name", default="v1-ranking-fast-v1")
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Target total completed trials; reruns resume up to this number.",
    )
    parser.add_argument("--validation-candidates", type=int, default=3)
    parser.add_argument("--train-sessions", type=int, default=500)
    parser.add_argument("--validation-sessions", type=int, default=150)
    parser.add_argument("--test-sessions", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate baseline V1 on small split prefixes without creating a study.",
    )
    parser.add_argument("--dry-run-sessions", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.trials <= 0:
        raise ValueError("trials must be positive")
    if args.validation_candidates <= 0:
        raise ValueError("validation-candidates must be positive")
    if args.dry_run_sessions <= 0:
        raise ValueError("dry-run-sessions must be positive")
    if min(args.train_sessions, args.validation_sessions, args.test_sessions) <= 0:
        raise ValueError("split session limits must be positive")

    split_paths = {
        name: args.data_dir / f"{name}.jsonl"
        for name in ("train", "validation", "test")
    }
    full_datasets = {name: load_jsonl(path) for name, path in split_paths.items()}
    public_samples = load_jsonl(args.public_set)
    validate_dataset_boundaries(full_datasets, public_samples)
    limits = {
        "train": args.train_sessions,
        "validation": args.validation_sessions,
        "test": args.test_sessions,
    }
    datasets = {
        name: select_product_unique_sessions(samples, limits[name], args.seed + index)
        for index, (name, samples) in enumerate(full_datasets.items())
    }
    validate_dataset_boundaries(datasets, public_samples)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = AgentV1(args.catalog)
    show_progress = not args.quiet
    study_storage: optuna.storages.RDBStorage | None = None

    if not args.quiet:
        print(
            "Tuning subset: "
            + ", ".join(f"{name}={len(samples)}" for name, samples in datasets.items())
            + f", trials={args.trials}, validation_candidates={args.validation_candidates}",
            file=sys.stderr,
        )

    try:
        if args.dry_run:
            config = ScoringConfig.default()
            results = {}
            for name, samples in datasets.items():
                subset = samples[: args.dry_run_sessions]
                result = _evaluate_config(
                    agent,
                    config,
                    subset,
                    catalog_ids,
                    categories,
                    products,
                    f"Dry run {name}",
                    show_progress,
                )
                results[name] = {
                    **_summary(result),
                    "objective": round(session_objective(result), 9),
                }
            print(json.dumps({"dry_run": True, "results": results}, indent=2))
            return

        args.output_dir.mkdir(parents=True, exist_ok=True)
        storage_path = (args.output_dir / "study.db").resolve()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{storage_path}",
        )
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=args.seed,
                n_startup_trials=5,
            ),
            storage=study_storage,
            load_if_exists=True,
        )
        signature = study_signature(datasets)
        stored_signature = study.user_attrs.get("signature")
        if stored_signature is None:
            if study.trials:
                raise ValueError(
                    "existing study has no dataset signature; choose a new --study-name"
                )
            study.set_user_attr("signature", signature)
        elif stored_signature != signature:
            raise ValueError(
                "existing study uses a different dataset or search space; "
                "choose a new --study-name"
            )
        completed_before = len(_complete_trials(study))
        remaining = max(0, args.trials - completed_before)
        trial_bar = tqdm(
            total=args.trials,
            initial=min(completed_before, args.trials),
            desc="Tuning V1",
            unit="trial",
            disable=not show_progress,
            position=0,
        )

        def objective(trial: optuna.Trial) -> float:
            config = suggest_scoring_config(trial)
            result = _evaluate_config(
                agent,
                config,
                datasets["train"],
                catalog_ids,
                categories,
                products,
                f"Trial {trial.number} train",
                show_progress,
                position=1,
                leave=False,
            )
            score = session_objective(result)
            trial.set_user_attr("scoring", config.to_dict())
            trial.set_user_attr("metrics", _summary(result))
            return score

        def update_trial_bar(study: optuna.Study, trial: FrozenTrial) -> None:
            trial_bar.update()
            trial_bar.set_postfix(
                best=f"{study.best_value:.6f}",
                current=f"{trial.value:.6f}" if trial.value is not None else "failed",
                refresh=False,
            )

        try:
            study.optimize(
                objective,
                n_trials=remaining,
                callbacks=[update_trial_bar],
                show_progress_bar=False,
            )
        finally:
            trial_bar.close()

        completed = sorted(
            _complete_trials(study),
            key=lambda trial: float(trial.value),
            reverse=True,
        )
        if not completed:
            raise RuntimeError("study contains no completed trials")
        candidates = completed[: min(args.validation_candidates, len(completed))]
        validation_bar = tqdm(
            candidates,
            desc="Selecting on validation",
            unit="config",
            disable=not show_progress,
            position=0,
        )
        validation_results: list[tuple[float, FrozenTrial, ScoringConfig, dict]] = []
        for candidate in validation_bar:
            config = ScoringConfig.from_dict(candidate.user_attrs["scoring"])
            result = _evaluate_config(
                agent,
                config,
                datasets["validation"],
                catalog_ids,
                categories,
                products,
                f"Trial {candidate.number} validation",
                show_progress,
                position=1,
                leave=False,
            )
            score = session_objective(result)
            validation_results.append((score, candidate, config, result))
            validation_bar.set_postfix(best=f"{max(item[0] for item in validation_results):.6f}")

        validation_score, selected_trial, selected_config, validation_result = max(
            validation_results,
            key=lambda item: (item[0], float(item[1].value)),
        )
        test_result = _evaluate_config(
            agent,
            selected_config,
            datasets["test"],
            catalog_ids,
            categories,
            products,
            "Final generated test",
            show_progress,
        )
        public_result = _evaluate_config(
            agent,
            selected_config,
            public_samples,
            catalog_ids,
            categories,
            products,
            "Public evaluation",
            show_progress,
        )
        best_config = {
            "schema_version": 1,
            "agent": "v1-tuned",
            "scoring": selected_config.to_dict(),
            "selection": {
                "study_name": args.study_name,
                "completed_trials": len(completed),
                "selected_trial": selected_trial.number,
                "training_objective": selected_trial.value,
                "validation_objective": validation_score,
                "test_objective": session_objective(test_result),
                "public_objective": session_objective(public_result),
                "dataset_signature": signature,
            },
        }
        _write_json(args.output_dir / "best_config.json", best_config)
        _write_json(args.output_dir / "validation_result.json", validation_result)
        _write_json(args.output_dir / "test_result.json", test_result)
        _write_json(args.output_dir / "public_result.json", public_result)
        print(
            json.dumps(
                {
                    **best_config["selection"],
                    "best_config": str(args.output_dir / "best_config.json"),
                    "validation": _summary(validation_result),
                    "test": _summary(test_result),
                    "public": _summary(public_result),
                },
                indent=2,
            )
        )
    finally:
        if study_storage is not None:
            study_storage.remove_session()
            study_storage.engine.dispose()
        agent.close()


if __name__ == "__main__":
    main()
