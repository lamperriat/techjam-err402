from __future__ import annotations

"""Run the target-blind P11 Top-10 reranker promotion protocol.

Only this parent process opens conversation corpora and holds labels.  Every
role is evaluated through a fresh JSONL worker process that receives only an
opaque ordinal, user profile, user message, turn, and Top-K value.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from types import CodeType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = "p11.top10-evaluation.v1"
SPEC_SCHEMA_VERSION = "p11.top10-experiment.v1"
CORPUS_PROTOCOL_SCHEMA_VERSION = "p11.corpus-protocol.v1"
PREREG_SCHEMA_VERSION = "p11.prereg-lock.v1"
DEFAULT_BOOTSTRAP_SEED = 20260829
MIN_BOOTSTRAP_RESAMPLES = 10_000
BASELINE_ID = "P11.B00.served_agent"
CONTROL_ID = "P11.C00.r08_coverage"
SHADOW_ID = "P11.S00.top10_linear_shadow"
ACTIVE_ID = "P11.R01.top10_linear"
ROLE_ORDER = (BASELINE_ID, CONTROL_ID, SHADOW_ID, ACTIVE_ID)
REPEAT_ROLES = (BASELINE_ID, CONTROL_ID, ACTIVE_ID)
SPLIT_ORDER = ("primary", "uniform_tail", "confirmation")
FORMAL_SPLITS = SPLIT_ORDER
DIAGNOSTIC_SPLITS = (
    "failure_negative",
    "failure_budget",
    "failure_override",
    "failure_missing_evidence",
)
LOCK_ALL_SPLITS = (*FORMAL_SPLITS, *DIAGNOSTIC_SPLITS)

DEFAULT_SPEC = PROJECT_ROOT / "configs" / "p11_top10_experiment.json"
DEFAULT_PREREG_LOCK = PROJECT_ROOT / "configs" / "p11_prereg_lock.json"
DEFAULT_CORPUS_PROTOCOL = PROJECT_ROOT / "configs" / "p11_corpus_protocol.json"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "catalog.jsonl"
DEFAULT_RELEASED_PUBLIC = PROJECT_ROOT / "data" / "public_set.jsonl"
DEFAULT_EVALUATION_CONFIG = PROJECT_ROOT / "docs" / "evaluation_config.json"
DEFAULT_CORPUS_METADATA = PROJECT_ROOT / "experiments" / "p11_corpora.metadata.json"
DEFAULT_SIDECAR = PROJECT_ROOT / "experiments" / "p11_features.sqlite"
DEFAULT_SIDECAR_METADATA = PROJECT_ROOT / "experiments" / "p11_features.metadata.json"
DEFAULT_WORKER = PROJECT_ROOT / "scripts" / "p11_worker.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "p11_top10_evaluation.json"
DEFAULT_ATTEMPT_MARKER = PROJECT_ROOT / "experiments" / "p11_formal_attempt.json"
DEFAULT_CONFIRMATION_MARKER = (
    PROJECT_ROOT / "experiments" / "p11_confirmation_consumed.json"
)
DEFAULT_CORPORA = {
    "primary": PROJECT_ROOT / "experiments" / "p11_primary_representative.jsonl",
    "uniform_tail": PROJECT_ROOT / "experiments" / "p11_uniform_tail.jsonl",
    "confirmation": PROJECT_ROOT / "experiments" / "p11_confirmation.jsonl",
}
DEFAULT_DIAGNOSTIC_CORPORA = {
    "failure_negative": PROJECT_ROOT / "experiments" / "p11_failure_negative.jsonl",
    "failure_budget": PROJECT_ROOT / "experiments" / "p11_failure_budget.jsonl",
    "failure_override": PROJECT_ROOT / "experiments" / "p11_failure_override.jsonl",
    "failure_missing_evidence": PROJECT_ROOT
    / "experiments"
    / "p11_failure_missing_evidence.jsonl",
}

EXPECTED_CATALOG_SHA256 = (
    "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
)
EXPECTED_PUBLIC_SHA256 = (
    "571359a8a69014c43fc30d39c996c4a28e875dccc249dffc707358757beb16c0"
)
EXPECTED_PUBLIC_GIT_BLOB_SHA1 = "121dbec9c1368c81cd887d6959e62507512139c0"
EXPECTED_EVALUATOR_SHA256 = (
    "7121a6324d77eb7ebb9794555713fd0b89c658e7883d818c3537c4bae1cf0fa1"
)
EXPECTED_EVALUATOR_GIT_BLOB_SHA1 = "7c808347b31ef3121a9cbc4810ac3eb325f950ba"
EXPECTED_EVALUATION_CONFIG_SHA256 = (
    "bbf22ef47c4837c268031d0b0a7dacb0bcab2c157f3ce01eee37612a7511097d"
)
EXPECTED_EVALUATION_CONFIG_GIT_BLOB_SHA1 = (
    "22671ca7ee3b665e61b98b7b00e51543b3936a84"
)
EXPECTED_ORIGIN_URL = "https://github.com/lamperriat/techjam-err402.git"
EXPECTED_UPSTREAM_URL = (
    "https://github.com/TechJam2026/techjam-conversational-search.git"
)
EXPECTED_UPSTREAM_HEAD = "34078351e1c3615e5505a2e829600b56a542e462"
EXPECTED_OPENED_CORPUS_SHA256 = {
    "p1": "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae",
    "p5": "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c",
    "p6": "27544cdb6ed9495808c35bbab09b4dbadcb88a1d75d162f17bb4fba6ee8841c7",
    "p7": "bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546",
    "p8_confirmation": "3ae6f8ff7ab0362399b348c3443daa5b7138aab9cf72e944b7e11dd71d7d3dde",
    "p8_selection": "1c11d73d7c8ced617ce874e15a563f240731ca9654ed42bcc4f773b7b4da81ee",
    "p9_confirmation": "4bbd9d53f32e3773de18bab881ba6e5ef0887ca86701897798ee086430ed08d9",
    "p9_selection": "6298cbd6d7507f4b163ab4979a86ff109e0dffa90557e3b28e5d20d129e5be9f",
    "released_public": "6c726257fec25575716ee65b095f94c48402b6e14e83341518610f45fbfbec6d",
}
EXPECTED_OPENED_TARGET_UNION_COUNT = 1_800
EXPECTED_SCENARIO_COUNTS = {
    "boundary": 10,
    "browsing": 80,
    "buying": 80,
    "intent_override": 30,
}
EXPECTED_EVALUATION_CONFIG = {
    "catalog_id_field": "parent_asin",
    "top_k": 10,
    "max_turns": 10,
    "miss_turn_value": 11,
    "exact_match": True,
    "metrics": ["hit_rate_at_10", "mrr", "mttc", "reported_token_usage"],
    "scenario_metrics": ["buying", "browsing", "intent_override", "boundary"],
    "efficiency_formula": "clip((11 - mttc) / 10, 0, 1)",
    "recommended_composite": {
        "hit_rate_at_10": 0.5,
        "mrr": 0.3,
        "efficiency": 0.2,
    },
}
EXPECTED_PROMOTION_GATES = {
    "all_scenario_hit_rates_non_decrease": True,
    "confirmation_technical_score_strict_increase": True,
    "contract_clean": True,
    "exceptions_zero": True,
    "fresh_exact_repeat": True,
    "hit_rate_non_decrease": True,
    "mrr_non_decrease": True,
    "mttc_no_worse": True,
    "network_attempts_zero": True,
    "primary_minimum_absolute_technical_score_delta": 0.005,
    "primary_technical_score_strict_increase": True,
    "target_blind": True,
    "token_usage_zero": True,
    "uniform_tail_quality_non_regression": True,
    "zero_hit_to_miss": True,
}
EXPECTED_RESOURCE_LIMITS = {
    "peak_rss_ratio": 1.1,
    "response_p95_ratio": 1.2,
    "wall_ratio": 1.15,
}
EXPECTED_DEADLINE_POLICY = {
    "clock": "time.monotonic",
    "dry_preflight_applies": False,
    "formal_evaluation_seconds": 5_400,
    "scope": "entire_formal_run_including_all_role_process_phases",
}
EXPECTED_ARTIFACT_POLICY = {
    "aggregate_and_hash_only": True,
    "atomic_fail_if_exists": True,
    "raw_asins_allowed": False,
    "raw_sample_ids_allowed": False,
    "raw_sessions_allowed": False,
}

REQUIRED_SOURCE_PATHS = {
    "scripts_init": "scripts/__init__.py",
    "lock_builder": "scripts/build_p11_prereg_lock.py",
    "corpus_builder": "scripts/build_p11_corpora.py",
    "sidecar_builder": "scripts/build_p11_sidecar.py",
    "p8_corpus_builder": "scripts/build_p8_selection_corpus.py",
    "metric_bridge": "scripts/official_metric_bridge.py",
    "p11_gates": "scripts/p11_gates.py",
    "p11_worker": "scripts/p11_worker.py",
    "evaluate_p11": "scripts/evaluate_p11.py",
    "verify_official_assets": "scripts/verify_official_assets.py",
    "starter_init": "starter/__init__.py",
    "agent": "starter/agent.py",
    "coverage": "starter/coverage.py",
    "attributes": "starter/attributes.py",
    "slot_ledger": "starter/slot_ledger.py",
    "clarification": "starter/clarification.py",
    "reranker": "starter/reranker.py",
    "response_contract": "starter/response_contract.py",
    "p8_negative": "starter/p8_negative.py",
    "p9_evidence": "starter/p9_evidence.py",
    "p11_features": "starter/p11_features.py",
    "p11_lab": "starter/p11_lab.py",
    "evaluator_init": "evaluator/__init__.py",
    "official_evaluator": "evaluator/local_evaluator.py",
}
SOURCE_PATHS = REQUIRED_SOURCE_PATHS
RUNTIME_ASIN_SCAN_NAMES = tuple(REQUIRED_SOURCE_PATHS)
TARGET_BLIND_SOURCE_NAMES = ("p11_worker", "p11_features", "p11_lab")
WORKER_RUNTIME_SOURCE_NAMES = (
    "p11_worker",
    "p11_features",
    "p11_lab",
    "starter_init",
    "agent",
    "coverage",
    "attributes",
    "slot_ledger",
    "clarification",
    "reranker",
    "p8_negative",
    "p9_evidence",
)
TARGET_BLIND_FORBIDDEN = (
    "__closure__",
    "__defaults__",
    "_p11_raw_sqlite",
    "attach database",
    "ctypes",
    "enable_load_extension",
    "getprocessmemoryinfo",
    "getrusage",
    "ground_truth",
    "_sqlite3",
    "load_extension",
    "resource",
    "sample_id",
    "set_authorizer",
    "sqlite3.connection",
    "from sqlite3 import connection",
    "public_set.jsonl",
    "p11_primary",
    "p11_confirmation",
)

MAX_WORKER_MESSAGE_BYTES = 1_048_576
MAX_WORKER_REQUEST_BYTES = 65_536
MAX_ARTIFACT_BYTES = 4_194_304
AUDIT_DENIAL_CATEGORIES = ("lifecycle", "network", "process", "read", "sqlite")
ASIN_SHAPE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE)
HEX64 = re.compile(r"[a-f0-9]{64}")
MAX_STATIC_SCAN_VALUE_BYTES = 1_048_576
MAX_STATIC_SCAN_VALUES = 100_000
BOOTSTRAP_TIMEOUT_SECONDS = 120.0
REQUEST_TIMEOUT_SECONDS = 30.0
FINALIZE_TIMEOUT_SECONDS = 30.0
EXIT_TIMEOUT_SECONDS = 30.0
ROLE_DEADLINE_SECONDS = 1800.0
WORKER_CLEANUP_RESERVE_SECONDS = 0.1
FORMAL_EVALUATION_SECONDS = float(
    EXPECTED_DEADLINE_POLICY["formal_evaluation_seconds"]
)

EXPECTED_EXECUTION_ORDER = [
    "primary_initial",
    "primary_exact_repeat_if_eligible",
    "uniform_tail_non_regression_if_primary_repeat_passes",
    "confirmation_semantic_parse_if_all_previous_gates_pass",
    "confirmation_initial",
    "confirmation_exact_repeat_if_eligible",
    "final_promotion_gate",
]
EXPECTED_ROLES = {
    "served": BASELINE_ID,
    "control": CONTROL_ID,
    "shadow": SHADOW_ID,
    "active": ACTIVE_ID,
}
EXPECTED_FEATURE_CONTRACT = {
    "feature_registry_sha256": "c2c6b4309e5bbf8e092f625957ae5f0cdeb193adcc48d552e5291837803749b1",
    "feature_schema_version": "p11.top10-features.v2",
    "feature_semantics_sha256": "abae7be9ab9073593ca40309177408adf20e460e3153fe95ec942fb53b47a488",
    "scorer_version": "p11.top10-linear.v3",
    "tail_preserved": True,
    "top10_membership_preserved": True,
}
FEATURE_REGISTRY_SHA256 = str(EXPECTED_FEATURE_CONTRACT["feature_registry_sha256"])
FEATURE_SCHEMA_VERSION = str(EXPECTED_FEATURE_CONTRACT["feature_schema_version"])
FEATURE_SEMANTICS_SHA256 = str(EXPECTED_FEATURE_CONTRACT["feature_semantics_sha256"])
SCORER_VERSION = str(EXPECTED_FEATURE_CONTRACT["scorer_version"])


class P11RunnerError(RuntimeError):
    pass


def _deadline_timeout(
    deadline_monotonic: float | None,
    maximum_seconds: float,
    phase: str,
) -> float:
    if deadline_monotonic is None:
        return maximum_seconds
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise P11RunnerError(
            f"P11 formal global deadline exceeded before {phase}"
        )
    return min(maximum_seconds, remaining)


def _check_global_deadline(
    deadline_monotonic: float | None, phase: str
) -> None:
    _deadline_timeout(deadline_monotonic, FORMAL_EVALUATION_SECONDS, phase)


catalog_index: Any = None
evaluate: Any = None
load_jsonl: Any = None
evaluate_p11_gates: Any = None
ContractRecorder: Any = None


def _load_runtime_dependencies(*, formal: bool) -> None:
    """Import locked project code only after preflight/prereg validation."""

    global ContractRecorder, catalog_index, evaluate, evaluate_p11_gates, load_jsonl
    import importlib

    module_paths = {
        "scripts.build_p11_prereg_lock": PROJECT_ROOT
        / "scripts"
        / "build_p11_prereg_lock.py",
        "evaluator.local_evaluator": PROJECT_ROOT / "evaluator" / "local_evaluator.py",
        "scripts.p11_gates": PROJECT_ROOT / "scripts" / "p11_gates.py",
        "starter.response_contract": PROJECT_ROOT / "starter" / "response_contract.py",
    }
    closure_paths: dict[str, Path] = {}
    for relative in REQUIRED_SOURCE_PATHS.values():
        path = Path(relative)
        parts = list(path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        closure_paths[".".join(parts)] = (PROJECT_ROOT / path).resolve()
    if formal:
        allowed_preloaded = {"scripts", "scripts.evaluate_p11"}
        loaded_closure = {
            name: sys.modules[name]
            for name in closure_paths
            if name in sys.modules
        }
        if any(
            value is not None
            for value in (
                catalog_index,
                evaluate,
                load_jsonl,
                evaluate_p11_gates,
                ContractRecorder,
            )
        ) or any(name not in allowed_preloaded for name in loaded_closure):
            raise P11RunnerError(
                "P11 formal source closure was loaded before lock validation"
            )
        for name, module in loaded_closure.items():
            if (
                Path(str(getattr(module, "__file__", ""))).resolve()
                != closure_paths[name]
            ):
                raise P11RunnerError(
                    "P11 formal preloaded source module came from an unexpected path"
                )

    builder = importlib.import_module("scripts.build_p11_prereg_lock")
    frozen_checks = (
        builder.SCHEMA_VERSION == PREREG_SCHEMA_VERSION,
        builder.SOURCE_PATHS == REQUIRED_SOURCE_PATHS,
        tuple(builder.ALL_SPLITS) == tuple(LOCK_ALL_SPLITS),
        tuple(builder.EXPECTED_EXECUTION_ORDER) == tuple(EXPECTED_EXECUTION_ORDER),
        builder.EXPECTED_ROLES == EXPECTED_ROLES,
        builder.EXPECTED_FEATURE_CONTRACT == EXPECTED_FEATURE_CONTRACT,
        builder.EXPECTED_PROMOTION_GATES == EXPECTED_PROMOTION_GATES,
        builder.EXPECTED_RESOURCE_LIMITS == EXPECTED_RESOURCE_LIMITS,
        builder.EXPECTED_DEADLINE_POLICY == EXPECTED_DEADLINE_POLICY,
        builder.EXPECTED_ARTIFACT_POLICY == EXPECTED_ARTIFACT_POLICY,
        builder.EXPECTED_EVALUATION_CONFIG == EXPECTED_EVALUATION_CONFIG,
    )
    if not all(frozen_checks):
        raise P11RunnerError("P11 runner and locked builder constants differ")

    evaluator = importlib.import_module("evaluator.local_evaluator")
    gates = importlib.import_module("scripts.p11_gates")
    response_contract = importlib.import_module("starter.response_contract")
    loaded_modules = {
        "scripts.build_p11_prereg_lock": builder,
        "evaluator.local_evaluator": evaluator,
        "scripts.p11_gates": gates,
        "starter.response_contract": response_contract,
    }
    if any(
        Path(str(getattr(module, "__file__", ""))).resolve()
        != module_paths[name].resolve()
        for name, module in loaded_modules.items()
    ):
        raise P11RunnerError("P11 runtime dependency loaded from an unexpected path")
    for name, expected_path in closure_paths.items():
        module = sys.modules.get(name)
        if module is not None and (
            Path(str(getattr(module, "__file__", ""))).resolve() != expected_path
        ):
            raise P11RunnerError("P11 transitive source module path is invalid")
    if formal or catalog_index is None:
        catalog_index = evaluator.catalog_index
    if formal or evaluate is None:
        evaluate = evaluator.evaluate
    if formal or load_jsonl is None:
        load_jsonl = evaluator.load_jsonl
    if formal or evaluate_p11_gates is None:
        evaluate_p11_gates = gates.evaluate_p11_gates
    if formal or ContractRecorder is None:
        ContractRecorder = response_contract.ContractRecorder
    if (
        gates.DEFAULT_BOOTSTRAP_SEED != DEFAULT_BOOTSTRAP_SEED
        or gates.MIN_BOOTSTRAP_RESAMPLES != MIN_BOOTSTRAP_RESAMPLES
        or tuple(gates.SPLIT_NAMES)
        != ("primary", "confirmation", "uniform_tail")
        or tuple(gates.FRESH_SPLITS) != ("primary", "confirmation")
        or tuple(gates.REQUIRED_FLAGS)
        != (
            "exact_repeat",
            "contract_clean",
            "target_blind",
            "network_attempts_zero",
            "token_usage_zero",
            "exceptions_zero",
        )
        or {name: str(value) for name, value in gates.RESOURCE_LIMITS.items()}
        != {
            "wall_seconds": "1.15",
            "p95_latency_ms": "1.20",
            "peak_rss_bytes": "1.10",
        }
        or str(gates.PRIMARY_MIN_SCORE_DELTA) != "0.005"
    ):
        raise P11RunnerError("P11 gate constants differ from the frozen runner")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P11RunnerError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise P11RunnerError(f"{label} must be a JSON object")
    return value


def _validate_spec(spec: Mapping[str, Any]) -> None:
    expected_roots = {
        "artifact_policy",
        "bootstrap",
        "corpus_protocol",
        "deadline_policy",
        "execution_order",
        "feature_contract",
        "promotion_gates",
        "public_evaluation_run",
        "resource_limits",
        "roles",
        "schema_version",
        "served_control",
        "sidecar_policy",
    }
    if set(spec) != expected_roots or spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise P11RunnerError("P11 experiment spec root is not frozen")
    if spec.get("roles") != EXPECTED_ROLES:
        raise P11RunnerError("P11 role registry is not frozen")
    if spec.get("served_control") != {
        "retrieval_mode": "coverage",
        "rerank_mode": "off",
        "question_policy": "fast",
    }:
        raise P11RunnerError("P11 served control is not coverage/off/fast")
    if spec.get("feature_contract") != EXPECTED_FEATURE_CONTRACT:
        raise P11RunnerError("P11 feature contract is not frozen")
    if spec.get("execution_order") != EXPECTED_EXECUTION_ORDER:
        raise P11RunnerError("P11 execution order is not frozen")
    if spec.get("corpus_protocol") != {
        "path": "configs/p11_corpus_protocol.json",
        "schema_version": CORPUS_PROTOCOL_SCHEMA_VERSION,
    }:
        raise P11RunnerError("P11 corpus protocol reference is not frozen")
    if spec.get("public_evaluation_run") is not False:
        raise P11RunnerError("P11 released public evaluation must remain disabled")
    bootstrap = spec.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or bootstrap != {
        "confidence": 0.95,
        "lower_bound_strictly_above_zero": True,
        "method": "paired_nearest_rank_percentile",
        "resamples": MIN_BOOTSTRAP_RESAMPLES,
        "seed": DEFAULT_BOOTSTRAP_SEED,
        "unit": "paired_unrounded_session_technical_score_delta",
    }:
        raise P11RunnerError("P11 bootstrap protocol is not frozen")
    if spec.get("resource_limits") != {
        "wall_ratio": 1.15,
        "response_p95_ratio": 1.2,
        "peak_rss_ratio": 1.1,
    }:
        raise P11RunnerError("P11 resource limits are not frozen")
    if spec.get("deadline_policy") != EXPECTED_DEADLINE_POLICY:
        raise P11RunnerError("P11 global deadline policy is not frozen")
    gates = spec.get("promotion_gates")
    required_true = {
        "all_scenario_hit_rates_non_decrease",
        "confirmation_technical_score_strict_increase",
        "contract_clean",
        "exceptions_zero",
        "fresh_exact_repeat",
        "hit_rate_non_decrease",
        "mrr_non_decrease",
        "mttc_no_worse",
        "network_attempts_zero",
        "primary_technical_score_strict_increase",
        "target_blind",
        "token_usage_zero",
        "uniform_tail_quality_non_regression",
        "zero_hit_to_miss",
    }
    if (
        not isinstance(gates, Mapping)
        or set(gates) != {*required_true, "primary_minimum_absolute_technical_score_delta"}
        or any(gates.get(name) is not True for name in required_true)
        or gates.get("primary_minimum_absolute_technical_score_delta") != 0.005
    ):
        raise P11RunnerError("P11 promotion gates are not frozen")
    artifact_policy = spec.get("artifact_policy")
    if not isinstance(artifact_policy, Mapping) or any(
        artifact_policy.get(name) is not expected
        for name, expected in {
            "aggregate_and_hash_only": True,
            "atomic_fail_if_exists": True,
            "raw_asins_allowed": False,
            "raw_sample_ids_allowed": False,
            "raw_sessions_allowed": False,
        }.items()
    ):
        raise P11RunnerError("P11 artifact policy is not frozen")
    sidecar = spec.get("sidecar_policy")
    if not isinstance(sidecar, Mapping) or sidecar != {
        "catalog_only": True,
        "identity_hashed_before_and_after": True,
        "maximum_bytes": 33_554_432,
        "opened_by_active_and_shadow_only": True,
    }:
        raise P11RunnerError("P11 sidecar policy is not frozen")


def _validate_corpus_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != CORPUS_PROTOCOL_SCHEMA_VERSION:
        raise P11RunnerError("P11 corpus protocol schema is invalid")
    catalog = protocol.get("catalog")
    splits = protocol.get("splits")
    if not isinstance(catalog, Mapping) or not isinstance(splits, Mapping):
        raise P11RunnerError("P11 corpus protocol sections are invalid")
    if (
        not isinstance(catalog.get("count"), int)
        or isinstance(catalog.get("count"), bool)
        or int(catalog["count"]) <= 0
        or HEX64.fullmatch(str(catalog.get("sha256") or "")) is None
    ):
        raise P11RunnerError("P11 catalog lock is invalid")
    for split in SPLIT_ORDER:
        value = splits.get(split)
        if not isinstance(value, Mapping):
            raise P11RunnerError(f"P11 {split} corpus lock is missing")
        digest = value.get("expected_samples_sha256")
        if (
            not isinstance(value.get("count"), int)
            or isinstance(value.get("count"), bool)
            or int(value["count"]) <= 0
            or HEX64.fullmatch(str(digest or "")) is None
        ):
            raise P11RunnerError(
                f"P11 {split} corpus is not hash-frozen; build and lock it first"
            )


def _validate_disjointness_proof(
    protocol: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    """Independently require the complete opened, new, and cross-overlap proof."""

    opened_specs = protocol.get("opened_corpora")
    if (
        not isinstance(opened_specs, Mapping)
        or set(opened_specs) != set(EXPECTED_OPENED_CORPUS_SHA256)
        or protocol.get("opened_target_union_count")
        != EXPECTED_OPENED_TARGET_UNION_COUNT
        or any(
            not isinstance(opened_specs.get(name), Mapping)
            or opened_specs[name].get("rows") != 200
            or opened_specs[name].get("canonical_samples_sha256") != digest
            for name, digest in EXPECTED_OPENED_CORPUS_SHA256.items()
        )
    ):
        raise P11RunnerError("P11 protocol opened-corpus proof is not frozen")

    new_overlaps = metadata.get("new_pairwise_target_overlaps")
    expected_new_pairs = {
        f"{left}__{right}"
        for left, right in combinations(sorted(LOCK_ALL_SPLITS), 2)
    }
    if (
        not isinstance(new_overlaps, Mapping)
        or set(new_overlaps) != expected_new_pairs
        or any(value != 0 for value in new_overlaps.values())
        or metadata.get("new_target_union_count")
        != sum(int(protocol["splits"][name]["count"]) for name in LOCK_ALL_SPLITS)
    ):
        raise P11RunnerError("P11 metadata new-target proof is incomplete")

    opened = metadata.get("opened_registry")
    observations = opened.get("corpora") if isinstance(opened, Mapping) else None
    opened_overlaps = (
        opened.get("pairwise_target_overlaps")
        if isinstance(opened, Mapping)
        else None
    )
    expected_opened_pairs = {
        f"{left}__{right}"
        for left, right in combinations(sorted(EXPECTED_OPENED_CORPUS_SHA256), 2)
    }
    if (
        not isinstance(opened, Mapping)
        or set(opened)
        != {"corpora", "pairwise_target_overlaps", "target_union_count"}
        or opened.get("target_union_count") != EXPECTED_OPENED_TARGET_UNION_COUNT
        or not isinstance(observations, Mapping)
        or set(observations) != set(EXPECTED_OPENED_CORPUS_SHA256)
        or any(
            not isinstance(observations.get(name), Mapping)
            or dict(observations[name])
            != {
                "rows": 200,
                "unique_targets": 200,
                "canonical_samples_sha256": digest,
            }
            for name, digest in EXPECTED_OPENED_CORPUS_SHA256.items()
        )
        or not isinstance(opened_overlaps, Mapping)
        or set(opened_overlaps) != expected_opened_pairs
        or any(value != 0 for value in opened_overlaps.values())
    ):
        raise P11RunnerError("P11 metadata opened-target proof is incomplete")

    opened_vs_new = metadata.get("opened_vs_new_target_overlaps")
    if (
        not isinstance(opened_vs_new, Mapping)
        or set(opened_vs_new) != set(LOCK_ALL_SPLITS)
        or any(value != 0 for value in opened_vs_new.values())
    ):
        raise P11RunnerError("P11 metadata opened-to-new proof is incomplete")


def _identity_snapshot(
    *,
    spec_path: Path,
    corpus_protocol_path: Path,
    catalog_path: Path,
    sidecar_path: Path,
    corpus_paths: Mapping[str, Path],
) -> dict[str, Any]:
    sources = {
        name: _file_identity(PROJECT_ROOT / relative)
        for name, relative in sorted(SOURCE_PATHS.items())
    }
    data = {
        "catalog": _file_identity(catalog_path),
        "sidecar": _file_identity(sidecar_path),
        **{
            split: _file_identity(corpus_paths[split])
            for split in SPLIT_ORDER
        },
    }
    return {
        "source": sources,
        "config": {
            "experiment": _file_identity(spec_path),
            "corpus_protocol": _file_identity(corpus_protocol_path),
        },
        "data": data,
    }


def _target_blind_source_scan() -> dict[str, Any]:
    findings: list[str] = []
    identities = []
    for name in TARGET_BLIND_SOURCE_NAMES:
        path = PROJECT_ROOT / SOURCE_PATHS[name]
        try:
            static_values = _static_source_text_values(path)
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            raise P11RunnerError(
                "P11 target-blind source cannot be statically scanned"
            ) from exc
        source_sha256 = _sha256_file(path)
        identities.append((name, source_sha256))
        for token in TARGET_BLIND_FORBIDDEN:
            if any(token in value.casefold() for value in static_values):
                findings.append(f"{name}:{token}")
    return {
        "passed": not findings,
        "finding_count": len(findings),
        "finding_registry_sha256": _stable_sha256(findings),
        "scanned_source_sha256": _stable_sha256(identities),
    }


def preflight(
    *,
    spec_path: Path,
    corpus_protocol_path: Path,
    catalog_path: Path,
    sidecar_path: Path,
    corpus_paths: Mapping[str, Path],
) -> dict[str, Any]:
    if set(corpus_paths) != set(SPLIT_ORDER):
        raise P11RunnerError("P11 corpus path registry is invalid")
    spec = _load_object(spec_path, "P11 experiment spec")
    protocol = _load_object(corpus_protocol_path, "P11 corpus protocol")
    _validate_spec(spec)
    _validate_corpus_protocol(protocol)
    snapshot = _identity_snapshot(
        spec_path=spec_path,
        corpus_protocol_path=corpus_protocol_path,
        catalog_path=catalog_path,
        sidecar_path=sidecar_path,
        corpus_paths=corpus_paths,
    )
    if snapshot["data"]["catalog"]["sha256"] != protocol["catalog"]["sha256"]:
        raise P11RunnerError("P11 catalog SHA-256 differs from corpus protocol")
    if snapshot["data"]["sidecar"]["bytes"] > spec["sidecar_policy"]["maximum_bytes"]:
        raise P11RunnerError("P11 sidecar exceeds its frozen byte limit")
    for split in SPLIT_ORDER:
        expected = protocol["splits"][split]["expected_samples_sha256"]
        if snapshot["data"][split]["sha256"] != expected:
            raise P11RunnerError(f"P11 {split} SHA-256 differs from corpus protocol")
    source_scan = _target_blind_source_scan()
    if not source_scan["passed"]:
        raise P11RunnerError("P11 candidate source contains parent-only label vocabulary")
    return {
        "spec": spec,
        "protocol": protocol,
        "identity_snapshot": snapshot,
        "source_scan": source_scan,
    }


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise P11RunnerError(f"P11 formal path escapes the project root: {path}") from exc


def _git(
    *arguments: str, deadline_monotonic: float | None = None
) -> str:
    timeout = _deadline_timeout(
        deadline_monotonic, 30.0, f"Git {' '.join(arguments)}"
    )
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT_ROOT.resolve().as_posix()}",
                *arguments,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        if (
            deadline_monotonic is not None
            and time.monotonic() >= deadline_monotonic
        ):
            raise P11RunnerError(
                f"P11 formal global deadline exceeded during Git {' '.join(arguments)}"
            ) from exc
        raise P11RunnerError(f"P11 Git command timed out: {' '.join(arguments)}") from exc
    if completed.returncode:
        raise P11RunnerError(f"P11 Git command failed: {' '.join(arguments)}")
    return completed.stdout.strip()


def _git_blob(path: Path, *, deadline_monotonic: float | None = None) -> str:
    relative = _project_relative(path)
    return _git(
        "hash-object",
        f"--path={relative}",
        "--",
        relative,
        deadline_monotonic=deadline_monotonic,
    )


def _locked_file_entry(
    path: Path,
    *,
    git_blob: bool = False,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    entry = {"path": _project_relative(path), **_file_identity(path)}
    if git_blob:
        entry["git_blob_sha1"] = _git_blob(
            path, deadline_monotonic=deadline_monotonic
        )
    return entry


def _require_file_entry(
    observed: object,
    path: Path,
    label: str,
    *,
    git_blob: bool = False,
    deadline_monotonic: float | None = None,
) -> None:
    if not isinstance(observed, Mapping):
        raise P11RunnerError(f"P11 preregistration {label} entry is invalid")
    actual = _locked_file_entry(
        path,
        git_blob=git_blob,
        deadline_monotonic=deadline_monotonic,
    )
    if any(observed.get(key) != value for key, value in actual.items()):
        raise P11RunnerError(f"P11 preregistration {label} identity differs")


def _formal_paths_are_defaults(
    *,
    prereg_lock_path: Path,
    spec_path: Path,
    corpus_protocol_path: Path,
    catalog_path: Path,
    released_public_path: Path,
    evaluation_config_path: Path,
    corpus_metadata_path: Path,
    sidecar_path: Path,
    sidecar_metadata_path: Path,
    corpus_paths: Mapping[str, Path],
    diagnostic_paths: Mapping[str, Path],
    worker_path: Path,
) -> bool:
    expected = {
        "prereg_lock": DEFAULT_PREREG_LOCK,
        "spec": DEFAULT_SPEC,
        "corpus_protocol": DEFAULT_CORPUS_PROTOCOL,
        "catalog": DEFAULT_CATALOG,
        "released_public": DEFAULT_RELEASED_PUBLIC,
        "evaluation_config": DEFAULT_EVALUATION_CONFIG,
        "corpus_metadata": DEFAULT_CORPUS_METADATA,
        "sidecar": DEFAULT_SIDECAR,
        "sidecar_metadata": DEFAULT_SIDECAR_METADATA,
        "worker": DEFAULT_WORKER,
        **{f"formal:{name}": path for name, path in DEFAULT_CORPORA.items()},
        **{
            f"diagnostic:{name}": path
            for name, path in DEFAULT_DIAGNOSTIC_CORPORA.items()
        },
    }
    observed = {
        "prereg_lock": prereg_lock_path,
        "spec": spec_path,
        "corpus_protocol": corpus_protocol_path,
        "catalog": catalog_path,
        "released_public": released_public_path,
        "evaluation_config": evaluation_config_path,
        "corpus_metadata": corpus_metadata_path,
        "sidecar": sidecar_path,
        "sidecar_metadata": sidecar_metadata_path,
        "worker": worker_path,
        **{f"formal:{name}": path for name, path in corpus_paths.items()},
        **{f"diagnostic:{name}": path for name, path in diagnostic_paths.items()},
    }
    return set(observed) == set(expected) and all(
        observed[name].resolve() == path.resolve() for name, path in expected.items()
    )


def _verify_formal_git(
    source: Mapping[str, Any], *, deadline_monotonic: float | None = None
) -> dict[str, Any]:
    branch = _git(
        "branch", "--show-current", deadline_monotonic=deadline_monotonic
    )
    head = _git("rev-parse", "HEAD", deadline_monotonic=deadline_monotonic)
    status = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        deadline_monotonic=deadline_monotonic,
    )
    locked_commit = str(source.get("git_commit") or "")
    remote_proof = source.get("remote_proof")
    upstream_proof = source.get("official_upstream")
    if not branch or branch != source.get("git_branch") or status:
        raise P11RunnerError("P11 formal run requires the locked branch and a clean worktree")
    if re.fullmatch(r"[a-f0-9]{40}", locked_commit) is None:
        raise P11RunnerError("P11 locked source commit is invalid")
    if (
        not isinstance(remote_proof, Mapping)
        or remote_proof.get("verified") is not True
        or remote_proof.get("advertised_head") != locked_commit
        or not isinstance(upstream_proof, Mapping)
        or upstream_proof.get("verified") is not True
        or upstream_proof.get("advertised_head") != EXPECTED_UPSTREAM_HEAD
    ):
        raise P11RunnerError("P11 locked source remote proof is invalid")
    commit_count = _git(
        "rev-list",
        "--count",
        f"{locked_commit}..{head}",
        deadline_monotonic=deadline_monotonic,
    )
    parent = _git(
        "rev-parse", f"{head}^", deadline_monotonic=deadline_monotonic
    )
    changed_paths = _git(
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        head,
        deadline_monotonic=deadline_monotonic,
    ).splitlines()
    expected_lock_path = _project_relative(DEFAULT_PREREG_LOCK)
    if (
        commit_count != "1"
        or parent != locked_commit
        or changed_paths != [expected_lock_path]
    ):
        raise P11RunnerError(
            "P11 HEAD must be the single lock-only commit after the locked source commit"
        )
    origin_url = _git(
        "remote", "get-url", "origin", deadline_monotonic=deadline_monotonic
    )
    upstream_url = _git(
        "remote", "get-url", "upstream", deadline_monotonic=deadline_monotonic
    )
    if origin_url != EXPECTED_ORIGIN_URL or upstream_url != EXPECTED_UPSTREAM_URL:
        raise P11RunnerError("P11 formal remotes differ from the official registry")
    origin_ref = f"refs/heads/{branch}"
    origin_lines = _git(
        "ls-remote",
        "--heads",
        "origin",
        origin_ref,
        deadline_monotonic=deadline_monotonic,
    ).splitlines()
    if origin_lines != [f"{head}\t{origin_ref}"]:
        raise P11RunnerError("P11 formal HEAD is not pushed to the origin branch")
    upstream_ref = "refs/heads/main"
    upstream_lines = _git(
        "ls-remote",
        "--heads",
        "upstream",
        upstream_ref,
        deadline_monotonic=deadline_monotonic,
    ).splitlines()
    if upstream_lines != [f"{EXPECTED_UPSTREAM_HEAD}\t{upstream_ref}"]:
        raise P11RunnerError("P11 official upstream proof changed")
    if _git(
        "rev-parse",
        "refs/remotes/upstream/main",
        deadline_monotonic=deadline_monotonic,
    ) != EXPECTED_UPSTREAM_HEAD:
        raise P11RunnerError("P11 local official-upstream tracking head changed")
    merge_base = _git(
        "merge-base",
        locked_commit,
        EXPECTED_UPSTREAM_HEAD,
        deadline_monotonic=deadline_monotonic,
    )
    if merge_base != EXPECTED_UPSTREAM_HEAD:
        raise P11RunnerError(
            "P11 locked source commit is not based on the official upstream HEAD"
        )
    return {
        "branch": branch,
        "head": head,
        "locked_source_commit": locked_commit,
        "locked_source_is_ancestor": True,
        "lock_commit_count_from_source": 1,
        "lock_commit_changed_paths": [expected_lock_path],
        "head_parent_equals_locked_source": True,
        "clean": True,
        "origin_head_equals_head": True,
        "official_upstream_head": EXPECTED_UPSTREAM_HEAD,
        "locked_source_based_on_official_upstream": True,
    }


_NO_STATIC_VALUE = object()


def _static_value_size(value: str | bytes) -> int:
    return len(value.encode("utf-8")) if isinstance(value, str) else len(value)


def _bounded_static_expression(node: ast.AST) -> object:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, bytes, int)) and not isinstance(node.value, bool):
            return node.value
        return _NO_STATIC_VALUE
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _bounded_static_expression(node.left)
        right = _bounded_static_expression(node.right)
        if type(left) is not type(right) or not isinstance(left, (str, bytes)):
            return _NO_STATIC_VALUE
        if _static_value_size(left) + _static_value_size(right) > MAX_STATIC_SCAN_VALUE_BYTES:
            raise ValueError("static value exceeds scan limit")
        return left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _bounded_static_expression(node.left)
        right = _bounded_static_expression(node.right)
        if isinstance(left, (str, bytes)) and isinstance(right, int):
            text, count = left, right
        elif isinstance(right, (str, bytes)) and isinstance(left, int):
            text, count = right, left
        else:
            return _NO_STATIC_VALUE
        if count < 0 or _static_value_size(text) * count > MAX_STATIC_SCAN_VALUE_BYTES:
            raise ValueError("static value exceeds scan limit")
        return text * count
    if isinstance(node, ast.JoinedStr):
        fragments: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                fragment = value.value
            elif (
                isinstance(value, ast.FormattedValue)
                and value.conversion in {-1, ord("s")}
                and value.format_spec is None
            ):
                nested = _bounded_static_expression(value.value)
                if nested is _NO_STATIC_VALUE:
                    return _NO_STATIC_VALUE
                fragment = str(nested)
            else:
                return _NO_STATIC_VALUE
            fragments.append(fragment)
            if sum(len(item.encode("utf-8")) for item in fragments) > MAX_STATIC_SCAN_VALUE_BYTES:
                raise ValueError("static value exceeds scan limit")
        return "".join(fragments)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], (ast.List, ast.Tuple))
    ):
        separator = _bounded_static_expression(node.func.value)
        if not isinstance(separator, (str, bytes)):
            return _NO_STATIC_VALUE
        items = [_bounded_static_expression(item) for item in node.args[0].elts]
        if any(type(item) is not type(separator) for item in items):
            return _NO_STATIC_VALUE
        size = sum(_static_value_size(item) for item in items)
        size += max(0, len(items) - 1) * _static_value_size(separator)
        if size > MAX_STATIC_SCAN_VALUE_BYTES:
            raise ValueError("static value exceeds scan limit")
        return separator.join(items)
    return _NO_STATIC_VALUE


def _static_source_text_values(path: Path) -> set[str]:
    source_text = path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(path))
    compiled = compile(source_text, str(path), "exec", dont_inherit=True, optimize=0)
    values: set[str] = set()

    def add(value: object) -> None:
        if isinstance(value, str):
            if _static_value_size(value) > MAX_STATIC_SCAN_VALUE_BYTES:
                raise ValueError("static value exceeds scan limit")
            values.add(value)
        elif isinstance(value, bytes):
            if len(value) > MAX_STATIC_SCAN_VALUE_BYTES:
                raise ValueError("static value exceeds scan limit")
            values.add(value.decode("latin-1"))
        if len(values) > MAX_STATIC_SCAN_VALUES:
            raise ValueError("too many static values to scan")

    add(source_text)
    pending = [compiled]
    while pending:
        code = pending.pop()
        for value in code.co_consts:
            if isinstance(value, CodeType):
                pending.append(value)
            else:
                add(value)
    for node in ast.walk(tree):
        add(_bounded_static_expression(node))
    return values


def _independent_asin_literal_scan(
    source_files: Mapping[str, Any],
) -> dict[str, Any]:
    pattern = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE)
    proof = []
    count = 0
    for name in RUNTIME_ASIN_SCAN_NAMES:
        entry = source_files.get(name)
        if not isinstance(entry, Mapping):
            raise P11RunnerError("P11 runtime source registry is incomplete")
        path = PROJECT_ROOT / str(entry.get("path") or "")
        try:
            static_values = _static_source_text_values(path)
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            raise P11RunnerError("P11 runtime source cannot be statically scanned") from exc
        count += sum(len(pattern.findall(value)) for value in static_values)
        proof.append(
            {"name": name, "bytes": entry.get("bytes"), "sha256": entry.get("sha256")}
        )
    if count:
        raise P11RunnerError(
            "P11 runtime source contains a complete ASIN-shaped string literal"
        )
    return {
        "source_file_count": len(proof),
        "asin_literal_count": count,
        "passed": count == 0,
        "proof_sha256": _stable_sha256(proof),
    }


def _postload_source_identifier_scan(
    source_files: Mapping[str, Any],
    identifiers: set[str],
    *,
    target_registry_sha256: str,
) -> dict[str, Any]:
    if (
        set(source_files) != set(RUNTIME_ASIN_SCAN_NAMES)
        or not identifiers
        or not isinstance(target_registry_sha256, str)
        or HEX64.fullmatch(target_registry_sha256) is None
        or target_registry_sha256 != _stable_sha256(sorted(identifiers))
    ):
        raise P11RunnerError("P11 post-load source scan inputs are invalid")
    needles = tuple(sorted(identifier.casefold() for identifier in identifiers))
    proof: list[dict[str, Any]] = []
    for name in RUNTIME_ASIN_SCAN_NAMES:
        entry = source_files.get(name)
        if not isinstance(entry, Mapping):
            raise P11RunnerError("P11 post-load source registry is incomplete")
        path = PROJECT_ROOT / SOURCE_PATHS[name]
        try:
            static_values = {
                value.casefold() for value in _static_source_text_values(path)
            }
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            raise P11RunnerError(
                "P11 post-load source cannot be statically scanned"
            ) from exc
        if any(needle in value for needle in needles for value in static_values):
            raise P11RunnerError(
                "P11 frozen source contains a split product identifier"
            )
        proof.append(
            {"name": name, "bytes": entry.get("bytes"), "sha256": entry.get("sha256")}
        )
    return {
        "executed": True,
        "source_file_count": len(proof),
        "identifier_count": len(identifiers),
        "match_count": 0,
        "passed": True,
        "proof_sha256": _stable_sha256(
            {
                "sources": proof,
                "target_registry_sha256": target_registry_sha256,
            }
        ),
    }


def validate_prereg_lock(
    prereg_lock_path: Path,
    *,
    spec_path: Path,
    corpus_protocol_path: Path,
    catalog_path: Path,
    released_public_path: Path,
    evaluation_config_path: Path,
    corpus_metadata_path: Path,
    sidecar_path: Path,
    sidecar_metadata_path: Path,
    corpus_paths: Mapping[str, Path],
    diagnostic_paths: Mapping[str, Path],
    worker_path: Path,
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    enforce_git: bool = True,
    require_defaults: bool = True,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Validate every preregistered identity before a formal evaluation."""

    if require_defaults and not _formal_paths_are_defaults(
        prereg_lock_path=prereg_lock_path,
        spec_path=spec_path,
        corpus_protocol_path=corpus_protocol_path,
        catalog_path=catalog_path,
        released_public_path=released_public_path,
        evaluation_config_path=evaluation_config_path,
        corpus_metadata_path=corpus_metadata_path,
        sidecar_path=sidecar_path,
        sidecar_metadata_path=sidecar_metadata_path,
        corpus_paths=corpus_paths,
        diagnostic_paths=diagnostic_paths,
        worker_path=worker_path,
    ):
        raise P11RunnerError("P11 formal evaluation requires every default frozen path")
    lock = _load_object(prereg_lock_path, "P11 preregistration lock")
    roots = {
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
    }
    if set(lock) != roots or lock.get("schema_version") != PREREG_SCHEMA_VERSION:
        raise P11RunnerError("P11 preregistration lock root is invalid")
    source = lock.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "git_commit",
        "git_branch",
        "remote_proof",
        "official_upstream",
        "files",
    }:
        raise P11RunnerError("P11 preregistration source proof is invalid")
    files = source.get("files")
    if not isinstance(files, Mapping) or set(files) != set(REQUIRED_SOURCE_PATHS):
        raise P11RunnerError("P11 preregistration source registry is incomplete")
    for name, relative in REQUIRED_SOURCE_PATHS.items():
        _require_file_entry(
            files[name],
            PROJECT_ROOT / relative,
            f"source.{name}",
            git_blob=True,
            deadline_monotonic=deadline_monotonic,
        )
    target_scan = lock.get("source_target_scan")
    asin_scan = lock.get("source_asin_literal_scan")
    if (
        not isinstance(target_scan, Mapping)
        or target_scan.get("passed") is not True
        or target_scan.get("match_count") != 0
        or target_scan.get("source_file_count") != len(REQUIRED_SOURCE_PATHS)
    ):
        raise P11RunnerError("P11 preregistration target-identifier scan failed")
    independent_scan = _independent_asin_literal_scan(files)
    if (
        independent_scan.get("passed") is not True
        or independent_scan.get("asin_literal_count") != 0
        or not isinstance(asin_scan, Mapping)
        or dict(asin_scan) != independent_scan
    ):
        raise P11RunnerError("P11 preregistration ASIN-literal scan differs")

    official = lock.get("official")
    if not isinstance(official, Mapping) or set(official) != {
        "catalog",
        "released_public",
        "evaluator",
        "evaluation_config",
    }:
        raise P11RunnerError("P11 preregistration official asset registry is invalid")
    _require_file_entry(
        official["catalog"],
        catalog_path,
        "official.catalog",
        deadline_monotonic=deadline_monotonic,
    )
    catalog_lock = official["catalog"]
    if (
        catalog_lock.get("sha256") != EXPECTED_CATALOG_SHA256
        or catalog_lock.get("rows") != 50_000
        or catalog_lock.get("unique_ids") != 50_000
    ):
        raise P11RunnerError("P11 preregistration catalog is not official")
    _require_file_entry(
        official["released_public"],
        released_public_path,
        "official.released_public",
        git_blob=True,
        deadline_monotonic=deadline_monotonic,
    )
    public_lock = official["released_public"]
    if (
        public_lock.get("sha256") != EXPECTED_PUBLIC_SHA256
        or public_lock.get("git_blob_sha1_lf") != EXPECTED_PUBLIC_GIT_BLOB_SHA1
        or public_lock.get("rows") != 200
        or public_lock.get("scenario_counts") != EXPECTED_SCENARIO_COUNTS
    ):
        raise P11RunnerError("P11 preregistration public-set identity is not official")
    _require_file_entry(
        official["evaluator"], PROJECT_ROOT / REQUIRED_SOURCE_PATHS["official_evaluator"],
        "official.evaluator", git_blob=True, deadline_monotonic=deadline_monotonic
    )
    if (
        official["evaluator"].get("sha256") != EXPECTED_EVALUATOR_SHA256
        or official["evaluator"].get("git_blob_sha1") != EXPECTED_EVALUATOR_GIT_BLOB_SHA1
    ):
        raise P11RunnerError("P11 official evaluator identity differs")
    _require_file_entry(
        official["evaluation_config"], evaluation_config_path,
        "official.evaluation_config",
        git_blob=True,
        deadline_monotonic=deadline_monotonic,
    )
    parsed_config = _load_object(evaluation_config_path, "official evaluation config")
    if (
        official["evaluation_config"].get("sha256")
        != EXPECTED_EVALUATION_CONFIG_SHA256
        or official["evaluation_config"].get("git_blob_sha1")
        != EXPECTED_EVALUATION_CONFIG_GIT_BLOB_SHA1
        or official["evaluation_config"].get("parsed") != EXPECTED_EVALUATION_CONFIG
        or parsed_config != EXPECTED_EVALUATION_CONFIG
    ):
        raise P11RunnerError("P11 official evaluation config differs")

    experiment = lock.get("experiment")
    if not isinstance(experiment, Mapping) or set(experiment) != {"spec", "corpus_protocol"}:
        raise P11RunnerError("P11 preregistration experiment registry is invalid")
    _require_file_entry(
        experiment["spec"],
        spec_path,
        "experiment.spec",
        git_blob=True,
        deadline_monotonic=deadline_monotonic,
    )
    _require_file_entry(
        experiment["corpus_protocol"], corpus_protocol_path,
        "experiment.corpus_protocol",
        git_blob=True,
        deadline_monotonic=deadline_monotonic,
    )
    _require_file_entry(
        lock["corpus_metadata"],
        corpus_metadata_path,
        "corpus_metadata",
        deadline_monotonic=deadline_monotonic,
    )
    metadata = _load_object(corpus_metadata_path, "P11 corpus metadata")
    _validate_disjointness_proof(protocol, metadata)

    corpora = lock.get("corpora")
    all_paths = {**dict(corpus_paths), **dict(diagnostic_paths)}
    if not isinstance(corpora, Mapping) or set(corpora) != set(LOCK_ALL_SPLITS):
        raise P11RunnerError("P11 preregistration corpus registry is incomplete")
    for name in LOCK_ALL_SPLITS:
        _require_file_entry(
            corpora[name],
            all_paths[name],
            f"corpora.{name}",
            deadline_monotonic=deadline_monotonic,
        )
        expected = protocol["splits"][name]
        expected_parsed = name != "confirmation"
        if (
            corpora[name].get("rows") != expected["count"]
            or corpora[name].get("scenario_counts") != expected["scenario_counts"]
            or corpora[name].get("sha256") != expected["expected_samples_sha256"]
            or corpora[name].get("semantic_parse_executed") is not expected_parsed
        ):
            raise P11RunnerError(f"P11 preregistration {name} corpus differs")

    _require_file_entry(
        lock["sidecar"],
        sidecar_path,
        "sidecar",
        deadline_monotonic=deadline_monotonic,
    )
    sidecar_lock = lock["sidecar"]
    if (
        sidecar_lock.get("bytes", 0) <= 0
        or sidecar_lock.get("bytes") > spec["sidecar_policy"]["maximum_bytes"]
        or sidecar_lock.get("catalog_rows") != 50_000
        or sidecar_lock.get("catalog_sha256") != EXPECTED_CATALOG_SHA256
        or sidecar_lock.get("registry_sha256") != FEATURE_REGISTRY_SHA256
        or sidecar_lock.get("semantics_sha256") != FEATURE_SEMANTICS_SHA256
    ):
        raise P11RunnerError("P11 preregistration sidecar binding differs")
    _require_file_entry(
        lock["sidecar_metadata"],
        sidecar_metadata_path,
        "sidecar_metadata",
        deadline_monotonic=deadline_monotonic,
    )
    sidecar_metadata = _load_object(sidecar_metadata_path, "P11 sidecar metadata")
    if (
        sidecar_metadata.get("schema_version") != FEATURE_SCHEMA_VERSION
        or sidecar_metadata.get("target_blind") is not True
        or sidecar_metadata.get("label_free") is not True
        or sidecar_metadata.get("catalog", {}).get("rows") != 50_000
        or sidecar_metadata.get("catalog", {}).get("sha256")
        != EXPECTED_CATALOG_SHA256
        or sidecar_metadata.get("sidecar", {}).get("sha256")
        != sidecar_lock.get("sha256")
        or sidecar_metadata.get("sidecar", {}).get("bytes") != sidecar_lock.get("bytes")
        or sidecar_metadata.get("sidecar", {}).get("registry_sha256")
        != FEATURE_REGISTRY_SHA256
        or sidecar_metadata.get("sidecar", {}).get("semantics_sha256")
        != FEATURE_SEMANTICS_SHA256
    ):
        raise P11RunnerError("P11 sidecar metadata differs from preregistration")

    locked_protocol = lock.get("protocol")
    if (
        lock.get("roles") != spec["roles"]
        or lock.get("feature_contract") != spec["feature_contract"]
        or not isinstance(locked_protocol, Mapping)
        or locked_protocol.get("execution_order") != spec["execution_order"]
        or locked_protocol.get("deadline_policy") != EXPECTED_DEADLINE_POLICY
        or locked_protocol.get("promotion_gates") != EXPECTED_PROMOTION_GATES
        or locked_protocol.get("resource_limits") != EXPECTED_RESOURCE_LIMITS
        or locked_protocol.get("artifact_policy") != EXPECTED_ARTIFACT_POLICY
        or locked_protocol.get("public_evaluation_run") is not False
        or locked_protocol.get("formal_splits") != list(FORMAL_SPLITS)
        or locked_protocol.get("diagnostic_splits") != list(DIAGNOSTIC_SPLITS)
        or locked_protocol.get("confirmation")
        != {
            "locked_by_bytes_and_sha256_only": True,
            "semantic_parse_executed_by_lock_builder": False,
        }
    ):
        raise P11RunnerError("P11 preregistration protocol differs from current spec")
    git_proof = _verify_formal_git(
        source, deadline_monotonic=deadline_monotonic
    ) if enforce_git else {
        "enforced": False,
        "reason": "explicit non-production fixture validation",
    }
    return {
        "schema_version": PREREG_SCHEMA_VERSION,
        "sha256": _sha256_file(prereg_lock_path),
        "bytes": prereg_lock_path.stat().st_size,
        "source_commit": source["git_commit"],
        "source_branch": source["git_branch"],
        "git": git_proof,
        "source_target_scan": dict(target_scan),
        "source_asin_literal_scan": independent_scan,
        "confirmation_semantic_parse_executed_by_lock_builder": False,
        "all_identities_verified": True,
    }


def _formal_extra_identity_snapshot(
    *,
    prereg_lock_path: Path,
    released_public_path: Path,
    evaluation_config_path: Path,
    corpus_metadata_path: Path,
    sidecar_metadata_path: Path,
    diagnostic_paths: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "prereg_lock": _file_identity(prereg_lock_path),
        "released_public": _file_identity(released_public_path),
        "evaluation_config": _file_identity(evaluation_config_path),
        "corpus_metadata": _file_identity(corpus_metadata_path),
        "sidecar_metadata": _file_identity(sidecar_metadata_path),
        "diagnostic_corpora": {
            name: _file_identity(diagnostic_paths[name])
            for name in DIAGNOSTIC_SPLITS
        },
    }


def _minimal_worker_environment(scratch: Path) -> dict[str, str]:
    environment = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
    }
    if os.name == "nt":
        for name in ("SystemRoot", "WINDIR"):
            if value := os.environ.get(name):
                environment[name] = value
        environment.update({"TEMP": str(scratch), "TMP": str(scratch)})
    else:
        environment.update(
            {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TMPDIR": str(scratch)}
        )
    return environment


def _stage_manifest(stage: Path) -> dict[str, dict[str, int | str]]:
    return {
        path.relative_to(stage).as_posix(): _file_identity(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    }


def _stage_worker_runtime(
    destination: Path,
    *,
    role: str,
    worker_path: Path,
    catalog_path: Path,
    sidecar_path: Path,
    scratch: Path,
) -> tuple[Path, Path, str]:
    """Copy the worker runtime and install a pre-import read/process boundary."""

    stage = destination.resolve()
    stage.mkdir(parents=True, exist_ok=False)
    for name in WORKER_RUNTIME_SOURCE_NAMES:
        relative = Path(SOURCE_PATHS[name])
        source = worker_path if name == "p11_worker" else PROJECT_ROOT / relative
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    allowed_read_roots = [stage, scratch.resolve(), Path(sys.base_prefix).resolve()]
    if os.name == "nt":
        for name in ("SystemRoot", "WINDIR"):
            if value := os.environ.get(name):
                allowed_read_roots.append(Path(value).resolve())
    bootstrap = stage / "p11_preimport_boundary.py"
    allowed_read_files = [catalog_path.resolve()]
    sidecar_allowed = role in {SHADOW_ID, ACTIVE_ID}
    if sidecar_allowed:
        allowed_read_files.append(sidecar_path.resolve())
    bootstrap_source = f'''from __future__ import annotations
import atexit
import json
import os
import runpy
import sqlite3
import sys
import traceback
from pathlib import Path

if os.name == "nt":
    import ctypes
    import ctypes.wintypes
else:
    import resource

READ_ROOTS = {[str(path) for path in allowed_read_roots]!r}
READ_FILES = {[str(path) for path in allowed_read_files]!r}
WRITE_ROOT = {str(scratch.resolve())!r}
AUDIT_RECORD_PATH = {str((scratch / "preimport-audit.json").resolve())!r}
SIDECAR_SQLITE_URI = {
        sidecar_path.resolve().as_uri() + "?mode=ro&immutable=1"
        if sidecar_allowed
        else None
    !r}
PROCESS_EVENTS = {{
    "os.system", "os.posix_spawn", "os.posix_spawnp", "os.spawn",
    "os.startfile", "os.fork", "os.forkpty", "pty.spawn", "subprocess.Popen"
}}

if os.name == "nt":
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _GET_CURRENT_PROCESS = ctypes.windll.kernel32.GetCurrentProcess
    _GET_CURRENT_PROCESS.argtypes = []
    _GET_CURRENT_PROCESS.restype = ctypes.wintypes.HANDLE
    _GET_PROCESS_MEMORY_INFO = ctypes.windll.psapi.GetProcessMemoryInfo
    _GET_PROCESS_MEMORY_INFO.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        ctypes.wintypes.DWORD,
    ]
    _GET_PROCESS_MEMORY_INFO.restype = ctypes.wintypes.BOOL

    def _post_atexit_peak_rss(
        _counter_type=PROCESS_MEMORY_COUNTERS,
        _sizeof=ctypes.sizeof,
        _byref=ctypes.byref,
        _get_process_memory_info=_GET_PROCESS_MEMORY_INFO,
        _get_current_process=_GET_CURRENT_PROCESS,
        _int=int,
    ):
        counters = _counter_type()
        counters.cb = _sizeof(counters)
        success = _get_process_memory_info(
            _get_current_process(),
            _byref(counters),
            counters.cb,
        )
        return (
            _int(counters.PeakWorkingSetSize) if success else None,
            "Windows GetProcessMemoryInfo PeakWorkingSetSize",
        )
else:
    def _post_atexit_peak_rss(
        _getrusage=resource.getrusage,
        _rusage_self=resource.RUSAGE_SELF,
        _darwin=sys.platform == "darwin",
        _int=int,
    ):
        try:
            value = _int(_getrusage(_rusage_self).ru_maxrss)
        except (AttributeError, OSError, TypeError, ValueError):
            return None, "unavailable"
        if _darwin:
            return value, "resource.getrusage ru_maxrss bytes"
        return value * 1024, "resource.getrusage ru_maxrss KiB"

def _resolved(value):
    try:
        return Path(os.fsdecode(value)).resolve()
    except (OSError, TypeError, ValueError):
        return None

def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

def _install_private_audit():
    counts = {{
        "lifecycle": 0,
        "network": 0,
        "process": 0,
        "read": 0,
        "sqlite": 0,
    }}
    arguments = list(sys.argv[1:])

    def _option(name):
        try:
            return arguments[arguments.index(name) + 1]
        except (ValueError, IndexError):
            return ""

    bound_role = _option("--role")
    bound_nonce = _option("--nonce")
    encode = json.dumps
    raw_open = os.open
    raw_write = os.write
    raw_fsync = os.fsync
    raw_close = os.close
    raw_fstat = os.fstat
    raw_sqlite_connect = sqlite3.connect
    raw_sqlite_connection = sqlite3.Connection
    measure_peak_rss = _post_atexit_peak_rss
    hard_exit = os._exit
    audit_output_fd = sys.stdout.fileno()
    audit_output_identity = raw_fstat(audit_output_fd)
    audit_sequence = 0
    run_exitfuncs = getattr(atexit, "_run_exitfuncs", None)
    if not callable(run_exitfuncs):
        raise RuntimeError("CPython atexit execution hook is unavailable")
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    platform_name = os.name
    sqlite_ok = sqlite3.SQLITE_OK
    sqlite_deny = sqlite3.SQLITE_DENY
    sqlite_attach = sqlite3.SQLITE_ATTACH
    sqlite_detach = sqlite3.SQLITE_DETACH
    sqlite_function = sqlite3.SQLITE_FUNCTION

    def _sqlite_authorizer(action_code, argument_1, argument_2, _database, _trigger):
        if action_code in (sqlite_attach, sqlite_detach):
            _deny("sqlite", "sqlite ATTACH and DETACH are disabled")
            return sqlite_deny
        function_name = argument_2 if argument_2 is not None else argument_1
        if (
            action_code == sqlite_function
            and isinstance(function_name, str)
            and function_name.casefold() == "load_extension"
        ):
            _deny("sqlite", "sqlite extension loading is disabled")
            return sqlite_deny
        return sqlite_ok

    def _deny(kind, message):
        nonlocal audit_sequence
        counts[kind] += 1
        audit_sequence += 1
        event = {{
            "kind": "audit-denied",
            "category": kind,
            "role": bound_role,
            "nonce": bound_nonce,
            "sequence": audit_sequence,
        }}
        payload = (encode(event, sort_keys=True, separators=(",", ":")) + "\\n").encode("utf-8")
        try:
            if raw_fstat(audit_output_fd) != audit_output_identity:
                hard_exit(98)
            view = memoryview(payload)
            while view:
                written = raw_write(audit_output_fd, view)
                if written <= 0:
                    raise OSError("audit event write made no progress")
                view = view[written:]
        except BaseException:
            hard_exit(98)
        hard_exit(96)

    def _audit(event, event_arguments):
        if event == "p11.lifecycle":
            _deny("lifecycle", "worker lifecycle audit mutation is disabled")
        if event.startswith("socket."):
            _deny("network", "network access is disabled before worker import")
        if (
            event in PROCESS_EVENTS
            or event.startswith("os.exec")
            or event.startswith("os.spawn")
        ):
            _deny("process", "process creation is disabled")
        if event == "sqlite3.connect":
            database = event_arguments[0] if event_arguments else None
            allowed = database == ":memory:" or (
                SIDECAR_SQLITE_URI is not None and database == SIDECAR_SQLITE_URI
            )
            if not allowed:
                _deny("sqlite", "sqlite access is outside the role-specific sidecar boundary")
        if event in ("sqlite3.enable_load_extension", "sqlite3.load_extension"):
            _deny("sqlite", "sqlite extension loading is disabled")
        if event == "open" and event_arguments:
            if isinstance(event_arguments[0], int):
                return
            path = _resolved(event_arguments[0])
            if path is None:
                _deny("read", "unresolved file access is disabled")
            roots = [Path(value) for value in READ_ROOTS]
            files = [Path(value) for value in READ_FILES]
            if path not in files and not any(_within(path, root) for root in roots):
                _deny("read", "file read is outside the staged runtime")
            mode = event_arguments[1] if len(event_arguments) > 1 else None
            flags = event_arguments[2] if len(event_arguments) > 2 else 0
            writes = (
                isinstance(mode, str) and any(marker in mode for marker in "wax+")
            ) or (
                isinstance(flags, int)
                and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            )
            if writes and not _within(path, Path(WRITE_ROOT)):
                _deny("read", "file writes are restricted to worker scratch")
        if event in {{"os.listdir", "os.scandir"}} and event_arguments:
            path = _resolved(event_arguments[0])
            if path is None or not any(_within(path, Path(root)) for root in READ_ROOTS):
                _deny("read", "directory enumeration is outside the staged runtime")

    def _emit_final_record():
        peak_rss_bytes, peak_rss_backend = measure_peak_rss()
        memory = {{
            "schema_version": "p11.post-atexit-memory.v1",
            "backend": peak_rss_backend,
            "peak_rss_bytes": peak_rss_bytes,
            "available": peak_rss_bytes is not None,
            "covers_candidate_execution_through_atexit": peak_rss_bytes is not None,
        }}
        record = {{
            "schema_version": "p11.preimport-audit.v3",
            "role": bound_role,
            "nonce": bound_nonce,
            "denied_attempt_counts": dict(sorted(counts.items())),
            "denied_attempt_total": sum(counts.values()),
            "post_atexit_memory": memory,
        }}
        payload = (encode(record, sort_keys=True, separators=(",", ":")) + "\\n").encode("utf-8")
        descriptor = raw_open(
            AUDIT_RECORD_PATH,
            write_flags,
            0o600,
        )
        try:
            view = memoryview(payload)
            while view:
                written = raw_write(descriptor, view)
                if written <= 0:
                    raise OSError("audit record write made no progress")
                view = view[written:]
            raw_fsync(descriptor)
        finally:
            raw_close(descriptor)
        if platform_name != "nt" and directory_flag:
            descriptor = raw_open(WRITE_ROOT, os.O_RDONLY | directory_flag)
            try:
                raw_fsync(descriptor)
            finally:
                raw_close(descriptor)

    sys.addaudithook(_audit)

    class _GuardedCursor:
        __slots__ = ("_p11_raw_sqlite_cursor", "_p11_guarded_owner")

        def __init__(self, cursor, owner):
            object.__setattr__(self, "_p11_raw_sqlite_cursor", cursor)
            object.__setattr__(self, "_p11_guarded_owner", owner)

        def execute(self, *args, **kwargs):
            raw = object.__getattribute__(self, "_p11_raw_sqlite_cursor")
            raw.execute(*args, **kwargs)
            return self

        def executemany(self, *args, **kwargs):
            raw = object.__getattribute__(self, "_p11_raw_sqlite_cursor")
            raw.executemany(*args, **kwargs)
            return self

        def fetchone(self):
            return object.__getattribute__(self, "_p11_raw_sqlite_cursor").fetchone()

        def fetchmany(self, *args, **kwargs):
            return object.__getattribute__(self, "_p11_raw_sqlite_cursor").fetchmany(
                *args, **kwargs
            )

        def fetchall(self):
            return object.__getattribute__(self, "_p11_raw_sqlite_cursor").fetchall()

        def close(self):
            return object.__getattribute__(self, "_p11_raw_sqlite_cursor").close()

        @property
        def connection(self):
            return object.__getattribute__(self, "_p11_guarded_owner")

        @property
        def description(self):
            return object.__getattribute__(self, "_p11_raw_sqlite_cursor").description

        @property
        def rowcount(self):
            return object.__getattribute__(self, "_p11_raw_sqlite_cursor").rowcount

        def __iter__(self):
            return self

        def __next__(self):
            return next(object.__getattribute__(self, "_p11_raw_sqlite_cursor"))

    class _GuardedConnection:
        __slots__ = ("_p11_raw_sqlite_connection",)

        def __init__(self, connection):
            object.__setattr__(self, "_p11_raw_sqlite_connection", connection)

        def execute(self, *args, **kwargs):
            raw = object.__getattribute__(self, "_p11_raw_sqlite_connection")
            return _GuardedCursor(raw.execute(*args, **kwargs), self)

        def executemany(self, *args, **kwargs):
            raw = object.__getattribute__(self, "_p11_raw_sqlite_connection")
            return _GuardedCursor(raw.executemany(*args, **kwargs), self)

        def cursor(self, *args, **kwargs):
            raw = object.__getattribute__(self, "_p11_raw_sqlite_connection")
            return _GuardedCursor(raw.cursor(*args, **kwargs), self)

        def commit(self):
            return object.__getattribute__(self, "_p11_raw_sqlite_connection").commit()

        def rollback(self):
            return object.__getattribute__(self, "_p11_raw_sqlite_connection").rollback()

        def close(self):
            return object.__getattribute__(self, "_p11_raw_sqlite_connection").close()

        def set_authorizer(self, *_args, **_kwargs):
            _deny("sqlite", "sqlite authorizer mutation is disabled")

        def enable_load_extension(self, *_args, **_kwargs):
            _deny("sqlite", "sqlite extension loading is disabled")

        def load_extension(self, *_args, **_kwargs):
            _deny("sqlite", "sqlite extension loading is disabled")

        def __enter__(self):
            raw = object.__getattribute__(self, "_p11_raw_sqlite_connection")
            raw.__enter__()
            return self

        def __exit__(self, *args):
            raw = object.__getattribute__(self, "_p11_raw_sqlite_connection")
            return raw.__exit__(*args)

    def _protect_sqlite_connection(connection):
        try:
            connection.set_authorizer(_sqlite_authorizer)
        except BaseException:
            try:
                connection.close()
            finally:
                _deny("sqlite", "sqlite authorizer installation failed")
        return _GuardedConnection(connection)

    def _validate_sqlite_database_argument(args, kwargs):
        if args:
            database = args[0]
        else:
            database = kwargs.get("database")
        allowed = database == ":memory:" or (
            SIDECAR_SQLITE_URI is not None and database == SIDECAR_SQLITE_URI
        )
        if not allowed:
            _deny("sqlite", "sqlite access is outside the role-specific boundary")

    def _guarded_sqlite_connect(*args, **kwargs):
        _validate_sqlite_database_argument(args, kwargs)
        return _protect_sqlite_connection(raw_sqlite_connect(*args, **kwargs))

    def _guarded_sqlite_connection(*args, **kwargs):
        _validate_sqlite_database_argument(args, kwargs)
        return _protect_sqlite_connection(raw_sqlite_connection(*args, **kwargs))

    sqlite3.connect = _guarded_sqlite_connect
    for module_name in ("_sqlite3", "sqlite3.dbapi2"):
        module = sys.modules.get(module_name)
        if module is not None:
            module.connect = _guarded_sqlite_connect
            module.Connection = _guarded_sqlite_connection
    sqlite3.Connection = _guarded_sqlite_connection

    def _atexit_mutation_disabled(*_args, _emit=sys.audit, **_kwargs):
        _emit("p11.lifecycle")
        raise PermissionError("the bootstrap audit finalizer is immutable")

    for name in ("unregister", "_clear", "_run_exitfuncs"):
        if hasattr(atexit, name):
            setattr(atexit, name, _atexit_mutation_disabled)

    if hasattr(os, "_exit"):
        os._exit = _atexit_mutation_disabled
    low_level_os = sys.modules.get(os.name)
    for module in (os, low_level_os):
        if module is None:
            continue
        for name in ("close", "closerange", "dup", "dup2", "pipe", "pipe2"):
            if hasattr(module, name):
                setattr(module, name, _atexit_mutation_disabled)

    def _seal_exit_registration():
        atexit.register = _atexit_mutation_disabled

    return run_exitfuncs, _emit_final_record, _seal_exit_registration

_run_candidate_exitfuncs, _emit_final_record, _seal_exit_registration = (
    _install_private_audit()
)
worker = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(__file__).resolve().parent))
worker_exit_code = 0
try:
    runpy.run_path(worker, run_name="__main__")
except SystemExit as exc:
    if exc.code is None:
        worker_exit_code = 0
    elif isinstance(exc.code, int) and not isinstance(exc.code, bool):
        worker_exit_code = int(exc.code)
    else:
        worker_exit_code = 1
except BaseException:
    traceback.print_exc()
    worker_exit_code = 1

_seal_exit_registration()
try:
    _run_candidate_exitfuncs()
except BaseException:
    traceback.print_exc()
    worker_exit_code = 1
try:
    _emit_final_record()
except BaseException:
    traceback.print_exc()
    worker_exit_code = 97
raise SystemExit(worker_exit_code)
'''
    bootstrap.write_text(bootstrap_source, encoding="utf-8", newline="\n")
    manifest = _stage_manifest(stage)
    return bootstrap, stage / SOURCE_PATHS["p11_worker"], _stable_sha256(manifest)


def _verify_stage(stage: Path, expected_sha256: str) -> None:
    if _stable_sha256(_stage_manifest(stage)) != expected_sha256:
        raise P11RunnerError("P11 staged worker runtime changed during evaluation")


def _worker_request(operation: str, request_id: int, **payload: Any) -> bytes:
    allowed = {
        "reset": {"request_id", "operation", "ordinal", "user_profile"},
        "respond": {
            "request_id",
            "operation",
            "ordinal",
            "user_message",
            "turn",
            "top_k",
        },
        "finalize": {"request_id", "operation"},
    }
    request = {"request_id": request_id, "operation": operation, **payload}
    if operation not in allowed or set(request) != allowed[operation]:
        raise P11RunnerError("P11 worker request would cross the parent-only boundary")
    encoded = _canonical_bytes(request) + b"\n"
    if len(encoded) > MAX_WORKER_REQUEST_BYTES:
        raise P11RunnerError("P11 worker request exceeds its byte limit")
    return encoded


def _load_preimport_audit_record(
    path: Path,
    *,
    role: str,
    nonce: str,
    process_returncode: int | None,
    supervisor_denied_attempt_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Read the bootstrap-owned record only after a clean worker exit."""

    if process_returncode != 0:
        raise P11RunnerError("P11 pre-import audit requires a clean worker exit")
    if path.is_symlink() or not path.is_file():
        raise P11RunnerError("P11 pre-import audit record is missing or unsafe")
    size = path.stat().st_size
    if not 0 < size <= 4_096:
        raise P11RunnerError("P11 pre-import audit record size is invalid")
    try:
        payload = path.read_bytes()
        if len(payload) != size or not payload.endswith(b"\n"):
            raise ValueError("non-canonical record framing")
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise P11RunnerError("P11 pre-import audit record is invalid") from exc
    counts = value.get("denied_attempt_counts") if isinstance(value, dict) else None
    memory_value = value.get("post_atexit_memory") if isinstance(value, dict) else None
    expected_count_keys = set(AUDIT_DENIAL_CATEGORIES)
    supervisor_counts = dict(supervisor_denied_attempt_counts)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "role",
            "nonce",
            "denied_attempt_counts",
            "denied_attempt_total",
            "post_atexit_memory",
        }
        or value.get("schema_version") != "p11.preimport-audit.v3"
        or value.get("role") != role
        or value.get("nonce") != nonce
        or not isinstance(counts, dict)
        or set(counts) != expected_count_keys
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in counts.values()
        )
        or value.get("denied_attempt_total") != sum(counts.values())
        or set(supervisor_counts) != expected_count_keys
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in supervisor_counts.values()
        )
    ):
        raise P11RunnerError("P11 pre-import audit record binding is invalid")
    if counts != supervisor_counts:
        raise P11RunnerError(
            "P11 pre-import audit record differs from the supervisor event stream"
        )
    memory = _validate_post_atexit_memory(
        memory_value, label="P11 pre-import post-atexit memory"
    )
    return {
        "schema_version": "p11.parent-verified-preimport-audit.v3",
        "record_bound_to_process": True,
        "record_loaded_after_clean_exit": True,
        "agent_close_and_candidate_atexit_covered": True,
        "supervisor_event_stream_verified": True,
        "denied_attempt_counts": dict(sorted(counts.items())),
        "denied_attempt_total": sum(counts.values()),
        "post_atexit_memory": memory,
        "record_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_post_atexit_memory(value: object, *, label: str) -> dict[str, Any]:
    """Validate bootstrap-owned peak RSS measured after candidate exit hooks."""

    expected = {
        "schema_version",
        "backend",
        "peak_rss_bytes",
        "available",
        "covers_candidate_execution_through_atexit",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("schema_version") != "p11.post-atexit-memory.v1"
        or not isinstance(value.get("backend"), str)
        or not value.get("backend")
        or not isinstance(value.get("available"), bool)
        or not isinstance(
            value.get("covers_candidate_execution_through_atexit"), bool
        )
    ):
        raise P11RunnerError(f"{label} schema is invalid")
    peak = value.get("peak_rss_bytes")
    available = value["available"]
    covered = value["covers_candidate_execution_through_atexit"]
    if available:
        if (
            not isinstance(peak, int)
            or isinstance(peak, bool)
            or peak <= 0
            or covered is not True
        ):
            raise P11RunnerError(f"{label} value is invalid")
    elif peak is not None or covered is not False:
        raise P11RunnerError(f"{label} unavailable state is invalid")
    return json.loads(_canonical_bytes(value))


def _assert_worker_bound_content(
    value: object, *, current_target: str, label: str
) -> None:
    """Reject any label-shaped material before serializing a worker request."""

    try:
        text = _canonical_bytes(value).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise P11RunnerError(f"P11 worker-bound {label} is not canonical JSON") from exc
    if current_target and current_target.casefold() in text.casefold():
        raise P11RunnerError(f"P11 worker-bound {label} contains the current label")
    if ASIN_SHAPE.search(text) is not None:
        raise P11RunnerError(f"P11 worker-bound {label} contains an ASIN-shaped identifier")


@dataclass
class ParentLabelBoundary:
    """Keep the evaluator's current label in the parent process only."""

    delegate: Any
    expected_targets: tuple[str, ...]
    _next_index: int = 0
    _session_targets: dict[str, str] = field(default_factory=dict)

    def reset(self, opaque_id: str, user_profile: dict[str, Any]) -> None:
        if self._next_index >= len(self.expected_targets):
            raise P11RunnerError("P11 evaluator reset exceeded the parent label ledger")
        target = self.expected_targets[self._next_index]
        self._next_index += 1
        if opaque_id in self._session_targets:
            raise P11RunnerError("P11 evaluator reused a parent label session")
        _assert_worker_bound_content(
            user_profile, current_target=target, label="user profile"
        )
        self._session_targets[opaque_id] = target
        self.delegate.reset(opaque_id, user_profile)

    def respond(
        self, opaque_id: str, user_message: str, turn: int, top_k: int
    ) -> dict[str, Any]:
        target = self._session_targets.get(opaque_id)
        if target is None:
            raise P11RunnerError("P11 evaluator used an unknown parent label session")
        _assert_worker_bound_content(
            user_message, current_target=target, label="user message"
        )
        return self.delegate.respond(opaque_id, user_message, turn, top_k)

    def assert_complete(self) -> None:
        if self._next_index != len(self.expected_targets):
            raise P11RunnerError("P11 evaluator did not consume the parent label ledger")


@dataclass
class WorkerClient:
    role: str
    process: subprocess.Popen[bytes]
    nonce: str
    stderr_path: Path
    bootstrap_seconds: float
    stage_manifest_sha256: str
    deadline_monotonic: float
    global_deadline_monotonic: float | None
    _messages: queue.Queue[tuple[str, Any]] = field(
        default_factory=queue.Queue
    )
    _writes: queue.Queue[
        tuple[bytes | None, queue.Queue[tuple[str, Any]]]
    ] = field(default_factory=queue.Queue)
    _reader: threading.Thread | None = None
    _writer: threading.Thread | None = None
    _next_request_id: int = 1
    _next_ordinal: int = 1
    _ordinals: dict[str, int] = field(default_factory=dict)
    _response_digest: Any = field(default_factory=hashlib.sha256)
    _supervisor_audit_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in AUDIT_DENIAL_CATEGORIES}
    )
    _next_audit_sequence: int = 1
    response_count: int = 0

    @classmethod
    def start(
        cls,
        role: str,
        *,
        catalog_path: Path,
        sidecar_path: Path,
        sidecar_identity: Mapping[str, Any],
        worker_path: Path,
        bootstrap_path: Path,
        stage_manifest_sha256: str,
        scratch: Path,
        global_deadline_monotonic: float | None = None,
    ) -> "WorkerClient":
        if role not in ROLE_ORDER:
            raise P11RunnerError("unknown P11 worker role")
        _check_global_deadline(
            global_deadline_monotonic, f"{role} worker process start"
        )
        nonce = uuid.uuid4().hex
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(bootstrap_path),
            str(worker_path),
            "--role",
            role,
            "--nonce",
            nonce,
            "--catalog",
            str(catalog_path),
            "--sidecar",
            str(sidecar_path),
            "--sidecar-bytes",
            str(sidecar_identity["bytes"]),
            "--sidecar-sha256",
            str(sidecar_identity["sha256"]),
        ]
        lowered = " ".join(command).casefold()
        if any(token in lowered for token in TARGET_BLIND_FORBIDDEN):
            raise P11RunnerError("P11 worker command contains parent-only material")
        stderr_path = scratch / "stderr.bin"
        audit_record_path = scratch / "preimport-audit.json"
        if audit_record_path.exists() or audit_record_path.is_symlink():
            raise P11RunnerError("P11 pre-import audit path was not fresh")
        stderr_handle = stderr_path.open("wb")
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                command,
                cwd=bootstrap_path.parent,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                env=_minimal_worker_environment(scratch),
                text=False,
                bufsize=0,
                close_fds=True,
            )
        finally:
            stderr_handle.close()
        client = cls(
            role,
            process,
            nonce,
            stderr_path,
            0.0,
            stage_manifest_sha256,
            started + ROLE_DEADLINE_SECONDS,
            global_deadline_monotonic,
        )
        client._start_reader()
        client._start_writer()
        ready = client._read(BOOTSTRAP_TIMEOUT_SECONDS, "bootstrap")
        client.bootstrap_seconds = time.monotonic() - started
        if ready != {"kind": "ready", "nonce": nonce, "role": role}:
            client.abort()
            raise P11RunnerError("P11 worker ready message is invalid")
        return client

    def _start_reader(self) -> None:
        if self.process.stdout is None:
            raise P11RunnerError("P11 worker stdout is unavailable")
        self._reader = threading.Thread(
            target=self._reader_loop, name=f"p11-reader-{self.role}", daemon=True
        )
        self._reader.start()

    def _start_writer(self) -> None:
        if self.process.stdin is None:
            raise P11RunnerError("P11 worker stdin is unavailable")
        self._writer = threading.Thread(
            target=self._writer_loop, name=f"p11-writer-{self.role}", daemon=True
        )
        self._writer.start()

    def _writer_loop(self) -> None:
        assert self.process.stdin is not None
        while True:
            payload, acknowledgement = self._writes.get()
            try:
                if payload is None:
                    self.process.stdin.close()
                    acknowledgement.put(("closed", None))
                    return
                self.process.stdin.write(payload)
                self.process.stdin.flush()
                acknowledgement.put(("written", None))
            except BaseException as exc:
                acknowledgement.put(("error", type(exc).__name__))
                return

    def _reader_loop(self) -> None:
        assert self.process.stdout is not None
        try:
            while True:
                line = self.process.stdout.readline(MAX_WORKER_MESSAGE_BYTES + 1)
                if not line:
                    self._messages.put(("eof", None))
                    return
                if len(line) > MAX_WORKER_MESSAGE_BYTES or not line.endswith(b"\n"):
                    self._messages.put(("error", "invalid line"))
                    return
                try:
                    message = json.loads(line.decode("utf-8", errors="strict"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._messages.put(("error", "invalid JSON"))
                    return
                if not isinstance(message, dict):
                    self._messages.put(("error", "non-object"))
                    return
                if message.get("kind") == "audit-denied":
                    if (
                        set(message)
                        != {"kind", "category", "role", "nonce", "sequence"}
                        or message.get("category") not in AUDIT_DENIAL_CATEGORIES
                        or message.get("role") != self.role
                        or message.get("nonce") != self.nonce
                        or message.get("sequence") != self._next_audit_sequence
                        or not isinstance(message.get("sequence"), int)
                        or isinstance(message.get("sequence"), bool)
                    ):
                        self._messages.put(("error", "invalid audit event"))
                        return
                    category = str(message["category"])
                    self._supervisor_audit_counts[category] += 1
                    self._next_audit_sequence += 1
                    continue
                self._messages.put(("message", message))
        except BaseException as exc:
            self._messages.put(("error", type(exc).__name__))

    def _io_timeout(self, timeout: float, phase: str) -> float:
        now = time.monotonic()
        global_remaining = math.inf
        if self.global_deadline_monotonic is not None:
            global_remaining = self.global_deadline_monotonic - now
            if global_remaining <= WORKER_CLEANUP_RESERVE_SECONDS:
                try:
                    self.abort()
                except P11RunnerError:
                    pass
                raise P11RunnerError(
                    f"P11 formal global deadline exceeded (cleanup reserve) during {phase}"
                )
            global_remaining -= WORKER_CLEANUP_RESERVE_SECONDS
        role_remaining = self.deadline_monotonic - now
        if role_remaining <= 0:
            self.abort()
            raise P11RunnerError("P11 worker absolute role deadline exceeded")
        return min(timeout, role_remaining, global_remaining)

    def _read(self, timeout: float, phase: str) -> dict[str, Any]:
        bounded_timeout = self._io_timeout(timeout, phase)
        try:
            kind, value = self._messages.get(timeout=bounded_timeout)
        except queue.Empty as exc:
            global_expired = (
                self.global_deadline_monotonic is not None
                and time.monotonic()
                >= self.global_deadline_monotonic - WORKER_CLEANUP_RESERVE_SECONDS
            )
            role_expired = time.monotonic() >= self.deadline_monotonic
            try:
                self.abort()
            except P11RunnerError:
                pass
            if global_expired:
                raise P11RunnerError(
                    f"P11 formal global deadline exceeded during {phase}"
                ) from exc
            if role_expired:
                raise P11RunnerError(
                    "P11 worker absolute role deadline exceeded"
                ) from exc
            raise P11RunnerError(f"P11 worker {phase} timeout") from exc
        if kind != "message":
            self.abort()
            raise P11RunnerError(f"P11 worker {phase} failed: {kind}")
        return value

    def _write(self, payload: bytes, timeout: float, phase: str) -> None:
        if self._writer is None or not self._writer.is_alive():
            self.abort()
            raise P11RunnerError("P11 worker writer thread is unavailable")
        acknowledgement: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        bounded_timeout = self._io_timeout(timeout, phase)
        self._writes.put_nowait((payload, acknowledgement))
        try:
            kind, detail = acknowledgement.get(timeout=bounded_timeout)
        except queue.Empty as exc:
            global_expired = (
                self.global_deadline_monotonic is not None
                and time.monotonic()
                >= self.global_deadline_monotonic - WORKER_CLEANUP_RESERVE_SECONDS
            )
            self.abort()
            if global_expired:
                raise P11RunnerError(
                    f"P11 formal global deadline exceeded during {phase}"
                ) from exc
            raise P11RunnerError(f"P11 worker {phase} timeout") from exc
        if kind != "written":
            self.abort()
            raise P11RunnerError(f"P11 worker {phase} failed: {detail}")

    def _close_writer(self) -> None:
        if self._writer is None or not self._writer.is_alive():
            return
        acknowledgement: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        timeout = self._io_timeout(EXIT_TIMEOUT_SECONDS, "worker stdin close")
        self._writes.put_nowait((None, acknowledgement))
        try:
            kind, detail = acknowledgement.get(timeout=timeout)
        except queue.Empty as exc:
            self.abort()
            raise P11RunnerError("P11 worker stdin close timeout") from exc
        if kind != "closed":
            self.abort()
            raise P11RunnerError(f"P11 worker stdin close failed: {detail}")
        self._writer.join(
            timeout=self._io_timeout(5.0, "writer shutdown")
        )
        if self._writer.is_alive():
            self.abort()
            raise P11RunnerError("P11 worker writer thread did not terminate")

    def _request(self, operation: str, **payload: Any) -> Any:
        request_id = self._next_request_id
        self._next_request_id += 1
        encoded = _worker_request(operation, request_id, **payload)
        timeout = FINALIZE_TIMEOUT_SECONDS if operation == "finalize" else REQUEST_TIMEOUT_SECONDS
        self._write(encoded, timeout, f"{operation} request write")
        reply = self._read(timeout, operation)
        if reply.get("request_id") != request_id:
            raise P11RunnerError("P11 worker reply request ID mismatch")
        if reply.get("kind") == "error":
            raise P11RunnerError(f"P11 worker {operation} failed")
        if operation == "finalize":
            if reply.get("kind") != "result" or not isinstance(reply.get("bundle"), dict):
                raise P11RunnerError("P11 worker finalize reply is invalid")
            return reply["bundle"]
        if reply.get("kind") != "reply":
            raise P11RunnerError("P11 worker reply is invalid")
        return reply.get("value")

    def reset(self, opaque_id: str, user_profile: dict[str, Any]) -> None:
        if opaque_id in self._ordinals:
            raise P11RunnerError("official evaluator reused an opaque session")
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        self._ordinals[opaque_id] = ordinal
        self._request("reset", ordinal=ordinal, user_profile=dict(user_profile))

    def respond(
        self, opaque_id: str, user_message: str, turn: int, top_k: int
    ) -> dict[str, Any]:
        ordinal = self._ordinals.get(opaque_id)
        if ordinal is None:
            raise P11RunnerError("worker received an unknown opaque session")
        value = self._request(
            "respond",
            ordinal=ordinal,
            user_message=user_message,
            turn=turn,
            top_k=top_k,
        )
        response = value.get("response") if isinstance(value, dict) else None
        if not isinstance(response, dict):
            raise P11RunnerError("P11 worker response is not an object")
        self.response_count += 1
        self._response_digest.update(
            _canonical_bytes({"ordinal": ordinal, "turn": turn, "response": response})
            + b"\n"
        )
        return response

    def finalize(self) -> dict[str, Any]:
        bundle = self._request("finalize")
        self._close_writer()
        try:
            return_code = self.process.wait(
                timeout=self._io_timeout(EXIT_TIMEOUT_SECONDS, "worker exit")
            )
        except subprocess.TimeoutExpired as exc:
            global_expired = (
                self.global_deadline_monotonic is not None
                and time.monotonic() >= self.global_deadline_monotonic
            )
            self.abort()
            if global_expired:
                raise P11RunnerError(
                    "P11 formal global deadline exceeded during worker exit"
                ) from exc
            raise P11RunnerError("P11 worker exit timeout") from exc
        if self._reader is not None:
            self._reader.join(timeout=self._io_timeout(5.0, "reader shutdown"))
            if self._reader.is_alive():
                self.abort()
                raise P11RunnerError("P11 worker reader thread did not terminate")
        if self.process.stdout is not None:
            self.process.stdout.close()
        if return_code != 0:
            raise P11RunnerError("P11 worker exited unsuccessfully")
        trailing_messages: list[tuple[str, Any]] = []
        while True:
            try:
                trailing_messages.append(self._messages.get_nowait())
            except queue.Empty:
                break
        if trailing_messages != [("eof", None)]:
            raise P11RunnerError("P11 worker emitted invalid trailing output")
        self._io_timeout(5.0, "pre-import audit record read")
        preimport_audit = _load_preimport_audit_record(
            self.stderr_path.parent / "preimport-audit.json",
            role=self.role,
            nonce=self.nonce,
            process_returncode=self.process.returncode,
            supervisor_denied_attempt_counts=self._supervisor_audit_counts,
        )
        self._io_timeout(5.0, "pre-import audit record validation completion")
        raw_network_attempts = bundle.get("network_attempt_count")
        if (
            not isinstance(raw_network_attempts, int)
            or isinstance(raw_network_attempts, bool)
            or raw_network_attempts < 0
        ):
            raise P11RunnerError("P11 worker network audit count is invalid")
        bundle["preimport_audit"] = preimport_audit
        bundle["memory"] = dict(preimport_audit["post_atexit_memory"])
        bundle["network_attempt_count"] = (
            raw_network_attempts
            + preimport_audit["denied_attempt_counts"]["network"]
        )
        if (
            bundle.get("response_count") != self.response_count
            or bundle.get("response_sha256") != self._response_digest.hexdigest()
        ):
            raise P11RunnerError("parent and P11 worker response traces differ")
        bundle["timing"]["bootstrap_wall_seconds"] = self.bootstrap_seconds
        bundle["worker_process"] = {
            "pid": self.process.pid,
            "nonce": self.nonce,
            "separate_process": self.process.pid != os.getpid(),
            "staged_runtime": True,
            "pre_import_read_process_boundary": True,
            "python_audit_pre_import_network_fail_closed": True,
            "preimport_denied_attempt_accounting": True,
            "preimport_supervisor_event_stream": True,
            "preimport_audit_record_loaded_after_clean_exit": True,
            "agent_close_and_candidate_atexit_audited": True,
            "peak_rss_measured_after_candidate_atexit": True,
            "sqlite_memory_allowed_all_roles": True,
            "sqlite_attach_detach_extension_authorizer": True,
            "immutable_sidecar_sqlite_role_scoped": True,
            "sidecar_read_allowed": self.role in {SHADOW_ID, ACTIVE_ID},
            "stage_manifest_sha256": self.stage_manifest_sha256,
        }
        return bundle

    def abort(self) -> None:
        timed_out = False
        try:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=self._cleanup_timeout(EXIT_TIMEOUT_SECONDS))
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError:
            timed_out = self.process.poll() is None
        if self._writer is not None and self._writer.is_alive():
            acknowledgement: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
            self._writes.put_nowait((None, acknowledgement))
            self._writer.join(timeout=self._cleanup_timeout(5.0))
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if self._reader is not None:
            self._reader.join(timeout=self._cleanup_timeout(5.0))
        if self.process.poll() is None:
            raise P11RunnerError("P11 worker could not be terminated")

    def _cleanup_timeout(self, maximum: float) -> float:
        now = time.monotonic()
        remaining = [maximum, max(0.0, self.deadline_monotonic - now)]
        if self.global_deadline_monotonic is not None:
            remaining.append(max(0.0, self.global_deadline_monotonic - now))
        return min(remaining)


def _validate_capture(capture: object, role: str) -> dict[str, Any]:
    roots = {
        "schema_version",
        "role",
        "configuration",
        "stats",
        "integrity_errors",
        "hashes",
        "function_hashes",
    }
    if not isinstance(capture, dict) or set(capture) != roots or capture.get("role") != role:
        raise P11RunnerError("P11 worker capture schema is invalid")
    configuration = capture.get("configuration")
    stats = capture.get("stats")
    errors = capture.get("integrity_errors")
    hashes = capture.get("hashes")
    function_hashes = capture.get("function_hashes")
    if not all(isinstance(value, dict) for value in (configuration, stats, hashes, function_hashes)):
        raise P11RunnerError("P11 worker capture sections are invalid")
    if not isinstance(errors, list):
        raise P11RunnerError("P11 worker integrity errors are invalid")
    if role == BASELINE_ID:
        if capture.get("schema_version") != "p11.served-reference.v1" or configuration != {
            "retrieval_mode": "coverage",
            "rerank_mode": "off",
            "question_policy": "fast",
            "sidecar_opened": False,
        }:
            raise P11RunnerError("P11 served capture is not the R08 fallback")
    else:
        if (
            capture.get("schema_version") != "p11.top10-lab.v1"
            or configuration.get("retrieval_mode") != "coverage"
            or configuration.get("rerank_mode") != "off"
            or configuration.get("question_policy") != "fast"
            or configuration.get("top10_membership_preserved") is not True
            or configuration.get("tail_preserved") is not True
            or configuration.get("target_blind") is not True
            or configuration.get("label_free") is not True
            or configuration.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
            or configuration.get("scorer_version") != SCORER_VERSION
            or configuration.get("feature_registry_sha256") != FEATURE_REGISTRY_SHA256
            or configuration.get("feature_semantics_sha256") != FEATURE_SEMANTICS_SHA256
        ):
            raise P11RunnerError("P11 candidate capture contract is invalid")
        expected_open = role in {SHADOW_ID, ACTIVE_ID}
        if configuration.get("sidecar_opened") is not expected_open:
            raise P11RunnerError("P11 role opened the sidecar outside its boundary")
        if expected_open and configuration.get("sidecar_identity_verified") is not True:
            raise P11RunnerError("P11 sidecar identity was not verified")
    for registry in (hashes, function_hashes):
        if any(HEX64.fullmatch(str(value)) is None for value in registry.values()):
            raise P11RunnerError("P11 capture hash registry is invalid")
    return json.loads(_canonical_bytes(capture))


def _validate_worker_bundle(
    bundle: object,
    role: str,
    *,
    catalog_identity: Mapping[str, Any],
    sidecar_identity: Mapping[str, Any],
) -> dict[str, Any]:
    roots = {
        "schema_version",
        "role",
        "asset_validation",
        "capture",
        "response_count",
        "response_sha256",
        "generic_exception_count",
        "generic_exception_classes",
        "network_attempt_count",
        "preimport_audit",
        "timing",
        "memory",
        "worker_process",
    }
    if (
        not isinstance(bundle, dict)
        or set(bundle) != roots
        or bundle.get("schema_version") != "p11.worker-bundle.v1"
        or bundle.get("role") != role
    ):
        raise P11RunnerError("P11 worker bundle schema is invalid")
    bundle = json.loads(_canonical_bytes(bundle))
    bundle["capture"] = _validate_capture(bundle["capture"], role)
    assets = bundle.get("asset_validation")
    expected_sidecar_required = role in {SHADOW_ID, ACTIVE_ID}
    if (
        not isinstance(assets, dict)
        or assets.get("schema_version") != "p11.worker-assets.v1"
        or assets.get("catalog")
        != {
            **dict(catalog_identity),
            "verified_official": True,
        }
    ):
        raise P11RunnerError("P11 worker did not bind the official catalog identity")
    sidecar_assets = assets.get("sidecar")
    expected_sidecar_assets = (
        {
            "required": True,
            "opened_for_identity": True,
            "verified": True,
            "bytes": sidecar_identity["bytes"],
            "sha256": sidecar_identity["sha256"],
        }
        if expected_sidecar_required
        else {
            "required": False,
            "opened_for_identity": False,
            "verified": True,
            "bytes": None,
            "sha256": None,
        }
    )
    if sidecar_assets != expected_sidecar_assets:
        raise P11RunnerError("P11 worker sidecar asset boundary is invalid")
    for key in ("response_count", "generic_exception_count", "network_attempt_count"):
        value = bundle.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise P11RunnerError(f"P11 worker {key} is invalid")
    if HEX64.fullmatch(str(bundle.get("response_sha256") or "")) is None:
        raise P11RunnerError("P11 worker response hash is invalid")
    preimport = bundle.get("preimport_audit")
    denied_counts = (
        preimport.get("denied_attempt_counts")
        if isinstance(preimport, dict)
        else None
    )
    if (
        not isinstance(preimport, dict)
        or set(preimport)
        != {
            "schema_version",
            "record_bound_to_process",
            "record_loaded_after_clean_exit",
            "agent_close_and_candidate_atexit_covered",
            "supervisor_event_stream_verified",
            "denied_attempt_counts",
            "denied_attempt_total",
            "post_atexit_memory",
            "record_sha256",
        }
        or preimport.get("schema_version")
        != "p11.parent-verified-preimport-audit.v3"
        or preimport.get("record_bound_to_process") is not True
        or preimport.get("record_loaded_after_clean_exit") is not True
        or preimport.get("agent_close_and_candidate_atexit_covered") is not True
        or preimport.get("supervisor_event_stream_verified") is not True
        or not isinstance(denied_counts, dict)
        or set(denied_counts)
        != {"lifecycle", "network", "process", "read", "sqlite"}
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in denied_counts.values()
        )
        or preimport.get("denied_attempt_total") != sum(denied_counts.values())
        or HEX64.fullmatch(str(preimport.get("record_sha256") or "")) is None
        or bundle.get("network_attempt_count") < denied_counts["network"]
    ):
        raise P11RunnerError("P11 pre-import audit accounting is invalid")
    timing = bundle.get("timing")
    memory = bundle.get("memory")
    process = bundle.get("worker_process")
    if not isinstance(timing, dict) or not isinstance(memory, dict) or not isinstance(process, dict):
        raise P11RunnerError("P11 worker observations are invalid")
    verified_memory = _validate_post_atexit_memory(
        preimport.get("post_atexit_memory"),
        label="P11 parent-verified post-atexit memory",
    )
    if memory != verified_memory:
        raise P11RunnerError("P11 worker memory differs from the bootstrap-owned record")
    latency = timing.get("respond_latency")
    if not isinstance(latency, dict) or set(latency) != {"count", "mean_ms", "p95_ms", "max_ms"}:
        raise P11RunnerError("P11 worker latency summary is invalid")
    if latency.get("count") != bundle["response_count"]:
        raise P11RunnerError("P11 worker latency count differs from response count")
    if (
        process.get("separate_process") is not True
        or process.get("staged_runtime") is not True
        or process.get("pre_import_read_process_boundary") is not True
        or process.get("python_audit_pre_import_network_fail_closed") is not True
        or process.get("preimport_denied_attempt_accounting") is not True
        or process.get("preimport_supervisor_event_stream") is not True
        or process.get("preimport_audit_record_loaded_after_clean_exit") is not True
        or process.get("agent_close_and_candidate_atexit_audited") is not True
        or process.get("peak_rss_measured_after_candidate_atexit") is not True
        or process.get("sqlite_memory_allowed_all_roles") is not True
        or process.get("sqlite_attach_detach_extension_authorizer") is not True
        or process.get("immutable_sidecar_sqlite_role_scoped") is not True
        or process.get("sidecar_read_allowed")
        is not (role in {SHADOW_ID, ACTIVE_ID})
    ):
        raise P11RunnerError("P11 role process boundary is invalid")
    return bundle


def _official_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "sample_count",
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
    )
    return {field: result[field] for field in fields}


def _run_role(
    role: str,
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
    *,
    catalog_path: Path,
    sidecar_path: Path,
    sidecar_identity: Mapping[str, Any],
    worker_path: Path,
    global_deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    wall_started = time.perf_counter()
    _check_global_deadline(
        global_deadline_monotonic, f"{role} staging"
    )
    with tempfile.TemporaryDirectory(prefix="p11-worker-") as directory:
        root = Path(directory)
        scratch = root / "scratch"
        scratch.mkdir()
        bootstrap_path, staged_worker_path, stage_manifest_sha256 = (
            _stage_worker_runtime(
                root / "stage",
                role=role,
                worker_path=worker_path,
                catalog_path=catalog_path,
                sidecar_path=sidecar_path,
                scratch=scratch,
            )
        )
        worker: WorkerClient | None = None
        try:
            worker = WorkerClient.start(
                role,
                catalog_path=catalog_path,
                sidecar_path=sidecar_path,
                sidecar_identity=sidecar_identity,
                worker_path=staged_worker_path,
                bootstrap_path=bootstrap_path,
                stage_manifest_sha256=stage_manifest_sha256,
                scratch=scratch,
                global_deadline_monotonic=global_deadline_monotonic,
            )
            parent_boundary = ParentLabelBoundary(
                worker,
                tuple(
                    str(sample["ground_truth"]["parent_asin"])
                    for sample in samples
                ),
            )
            recorder = ContractRecorder(parent_boundary, catalog_ids)
            _check_global_deadline(
                global_deadline_monotonic, f"{role} official evaluation"
            )
            official = evaluate(recorder, samples, catalog_ids, categories, products)
            parent_boundary.assert_complete()
            _check_global_deadline(
                global_deadline_monotonic, f"{role} worker finalize"
            )
            bundle = _validate_worker_bundle(
                worker.finalize(),
                role,
                catalog_identity={
                    "bytes": catalog_path.stat().st_size,
                    "rows": len(catalog_ids),
                    "sha256": _sha256_file(catalog_path),
                },
                sidecar_identity=sidecar_identity,
            )
            _check_global_deadline(
                global_deadline_monotonic, f"{role} bundle validation completion"
            )
        finally:
            if worker is not None:
                worker.abort()
            _check_global_deadline(
                global_deadline_monotonic, f"{role} staged runtime verification"
            )
            _verify_stage(root / "stage", stage_manifest_sha256)
            _check_global_deadline(
                global_deadline_monotonic, f"{role} temporary cleanup entry"
            )
    _check_global_deadline(
        global_deadline_monotonic, f"{role} temporary cleanup completion"
    )
    wall_seconds = time.perf_counter() - wall_started
    sessions = official.get("sessions")
    scenarios = official.get("scenario_metrics")
    usage = official.get("reported_token_usage")
    if not isinstance(sessions, list) or not isinstance(scenarios, dict) or not isinstance(usage, dict):
        raise P11RunnerError("official evaluator returned an invalid P11 ledger")
    capture = bundle["capture"]
    stats = capture["stats"]
    integrity_errors = capture["integrity_errors"]
    p95 = bundle["timing"]["respond_latency"].get("p95_ms")
    peak = bundle["memory"].get("peak_rss_bytes")
    return {
        "role": role,
        "sessions": sessions,
        "metrics": _official_metrics(official),
        "scenario_metrics": scenarios,
        "resources": {
            "wall_seconds": wall_seconds,
            "p95_latency_ms": p95,
            "peak_rss_bytes": peak,
        },
        "functional_result_sha256": _stable_sha256(official),
        "response_trace_sha256": bundle["response_sha256"],
        "asset_validation": bundle["asset_validation"],
        "capture": capture,
        "audit": {
            "contract_error_count": len(recorder.errors),
            "contract_errors_sha256": _stable_sha256(sorted(recorder.errors)),
            "integrity_error_count": len(integrity_errors),
            "integrity_errors_sha256": _stable_sha256(sorted(integrity_errors)),
            "network_attempt_count": bundle["network_attempt_count"],
            "preimport_denied_attempt_count": bundle["preimport_audit"][
                "denied_attempt_total"
            ],
            "preimport_denied_attempt_counts": bundle["preimport_audit"][
                "denied_attempt_counts"
            ],
            "preimport_audit_state_missing_count": int(
                bundle["preimport_audit"]["record_loaded_after_clean_exit"] is not True
            ),
            "generic_exception_count": bundle["generic_exception_count"],
            "generic_exception_classes_sha256": _stable_sha256(
                bundle["generic_exception_classes"]
            ),
            "reported_token_total": usage.get("total_tokens"),
            "capture_exception_count": stats.get("exception_count", 0),
            "top10_membership_violation_count": stats.get(
                "top10_membership_violation_count", 0
            ),
            "tail_change_count": stats.get("tail_change_count", 0),
        },
        "_process": bundle["worker_process"],
    }


def worker_smoke_preflight(
    *, catalog_path: Path, sidecar_path: Path, worker_path: Path
) -> dict[str, Any]:
    """Start/finalize every staged role with zero sessions and no labels."""

    catalog_identity = {
        "bytes": catalog_path.stat().st_size,
        "rows": 50_000,
        "sha256": _sha256_file(catalog_path),
    }
    sidecar_identity = _file_identity(sidecar_path)
    roles: dict[str, Any] = {}
    for role in ROLE_ORDER:
        with tempfile.TemporaryDirectory(prefix="p11-smoke-") as directory:
            root = Path(directory)
            scratch = root / "scratch"
            scratch.mkdir()
            bootstrap, staged_worker, manifest_sha256 = _stage_worker_runtime(
                root / "stage",
                role=role,
                worker_path=worker_path,
                catalog_path=catalog_path,
                sidecar_path=sidecar_path,
                scratch=scratch,
            )
            worker: WorkerClient | None = None
            try:
                worker = WorkerClient.start(
                    role,
                    catalog_path=catalog_path,
                    sidecar_path=sidecar_path,
                    sidecar_identity=sidecar_identity,
                    worker_path=staged_worker,
                    bootstrap_path=bootstrap,
                    stage_manifest_sha256=manifest_sha256,
                    scratch=scratch,
                )
                bundle = _validate_worker_bundle(
                    worker.finalize(),
                    role,
                    catalog_identity=catalog_identity,
                    sidecar_identity=sidecar_identity,
                )
            except BaseException:
                if worker is not None:
                    worker.abort()
                raise
            finally:
                _verify_stage(root / "stage", manifest_sha256)
        if bundle["response_count"] != 0:
            raise P11RunnerError("P11 zero-session worker smoke returned responses")
        if (
            bundle["network_attempt_count"] != 0
            or bundle["preimport_audit"]["denied_attempt_total"] != 0
            or bundle["generic_exception_count"] != 0
        ):
            raise P11RunnerError("P11 zero-session worker smoke observed an audit violation")
        if (
            bundle["memory"].get("available") is not True
            or bundle["memory"].get(
                "covers_candidate_execution_through_atexit"
            )
            is not True
            or not isinstance(bundle["memory"].get("peak_rss_bytes"), int)
            or isinstance(bundle["memory"].get("peak_rss_bytes"), bool)
            or bundle["memory"]["peak_rss_bytes"] <= 0
        ):
            raise P11RunnerError(
                "P11 zero-session worker smoke lacks post-atexit peak RSS evidence"
            )
        roles[role] = {
            "asset_validation": bundle["asset_validation"],
            "capture_sha256": _stable_sha256(bundle["capture"]),
            "process_identity_sha256": _stable_sha256(bundle["worker_process"]),
            "network_attempt_count": bundle["network_attempt_count"],
            "preimport_denied_attempt_count": bundle["preimport_audit"][
                "denied_attempt_total"
            ],
            "generic_exception_count": bundle["generic_exception_count"],
            "post_atexit_memory": bundle["memory"],
        }
    return {
        "schema_version": "p11.worker-smoke.v1",
        "zero_session": True,
        "roles": roles,
        "passed": True,
    }


RoleRunner = Callable[..., dict[str, Any]]


def _run_roles(
    roles: Sequence[str],
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
    *,
    role_runner: RoleRunner,
    runtime: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    deadline = runtime.get("global_deadline_monotonic")
    for role in roles:
        _check_global_deadline(deadline, f"{role} role entry")
        runs[role] = role_runner(
            role,
            samples,
            catalog_ids,
            categories,
            products,
            catalog_path=runtime["catalog_path"],
            sidecar_path=runtime["sidecar_path"],
            sidecar_identity=runtime["sidecar_identity"],
            worker_path=runtime["worker_path"],
            global_deadline_monotonic=deadline,
        )
        _check_global_deadline(deadline, f"{role} role completion")
    return runs


def _load_split(
    split: str,
    path: Path,
    protocol: Mapping[str, Any],
    catalog_ids: set[str],
    excluded_targets: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    rows = load_jsonl(path)
    expected = protocol["splits"][split]
    if len(rows) != int(expected["count"]):
        raise P11RunnerError(f"P11 {split} row count differs from protocol")
    sample_ids: list[str] = []
    targets: list[str] = []
    scenarios: Counter[str] = Counter()
    prefix = str(expected.get("sample_id_prefix") or "")
    for row in rows:
        sample_id = row.get("sample_id")
        scenario = row.get("scenario_type")
        target = row.get("ground_truth", {}).get("parent_asin")
        if (
            not isinstance(sample_id, str)
            or not sample_id.startswith(prefix)
            or not isinstance(scenario, str)
            or not scenario
            or not isinstance(target, str)
            or target not in catalog_ids
        ):
            raise P11RunnerError(f"P11 {split} row schema is invalid")
        sample_ids.append(sample_id)
        targets.append(target)
        scenarios[scenario] += 1
    if len(set(sample_ids)) != len(rows) or len(set(targets)) != len(rows):
        raise P11RunnerError(f"P11 {split} identifiers are not unique")
    target_set = set(targets)
    if target_set & excluded_targets:
        raise P11RunnerError(f"P11 {split} targets overlap an earlier P11 split")
    if dict(sorted(scenarios.items())) != dict(sorted(expected["scenario_counts"].items())):
        raise P11RunnerError(f"P11 {split} scenario counts differ from protocol")
    return rows, {
        "rows": len(rows),
        "file_sha256": _sha256_file(path),
        "canonical_rows_sha256": _stable_sha256(rows),
        "sample_registry_sha256": _stable_sha256(sorted(sample_ids)),
        "target_registry_sha256": _stable_sha256(sorted(targets)),
        "scenario_counts": dict(sorted(scenarios.items())),
    }, target_set


def _process_token(run: Mapping[str, Any]) -> str:
    process = run.get("_process")
    if not isinstance(process, Mapping):
        return ""
    return _stable_sha256({"pid": process.get("pid"), "nonce": process.get("nonce")})


def _audit_flags(runs: Sequence[Mapping[str, Any]], *, exact_repeat: bool) -> dict[str, bool]:
    return {
        "exact_repeat": exact_repeat,
        "contract_clean": all(run.get("audit", {}).get("contract_error_count") == 0 for run in runs),
        "target_blind": all(
            run.get("role") == BASELINE_ID
            or (
                run.get("capture", {}).get("configuration", {}).get("target_blind") is True
                and run.get("capture", {}).get("configuration", {}).get("label_free") is True
            )
            for run in runs
        ),
        "network_attempts_zero": all(
            run.get("audit", {}).get("network_attempt_count") == 0 for run in runs
        ),
        "token_usage_zero": all(
            run.get("audit", {}).get("reported_token_total") == 0 for run in runs
        ),
        "exceptions_zero": all(
            run.get("audit", {}).get(key) == 0
            for run in runs
            for key in (
                "generic_exception_count",
                "preimport_denied_attempt_count",
                "preimport_audit_state_missing_count",
                "capture_exception_count",
                "integrity_error_count",
                "top10_membership_violation_count",
                "tail_change_count",
            )
        ),
    }


def _role_boundary_checks(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    def process_boundary(role: str, run: Mapping[str, Any]) -> bool:
        process = run.get("_process")
        return bool(
            isinstance(process, Mapping)
            and process.get("separate_process") is True
            and process.get("staged_runtime") is True
            and process.get("pre_import_read_process_boundary") is True
            and process.get("python_audit_pre_import_network_fail_closed") is True
            and process.get("preimport_denied_attempt_accounting") is True
            and process.get("preimport_supervisor_event_stream") is True
            and process.get("preimport_audit_record_loaded_after_clean_exit") is True
            and process.get("agent_close_and_candidate_atexit_audited") is True
            and process.get("peak_rss_measured_after_candidate_atexit") is True
            and process.get("sqlite_memory_allowed_all_roles") is True
            and process.get("sqlite_attach_detach_extension_authorizer") is True
            and process.get("immutable_sidecar_sqlite_role_scoped") is True
            and process.get("sidecar_read_allowed")
            is (role in {SHADOW_ID, ACTIVE_ID})
        )

    checks = {
        "all_initial_roles_present": set(runs) == set(ROLE_ORDER),
        "all_roles_match_registry": all(run.get("role") == role for role, run in runs.items()),
        "all_initial_processes_fresh": len({_process_token(run) for run in runs.values()}) == len(runs),
        "all_pre_import_boundaries_enforced": all(
            process_boundary(role, run) for role, run in runs.items()
        ),
    }
    if set(runs) != set(ROLE_ORDER):
        return checks
    served = runs[BASELINE_ID]
    control = runs[CONTROL_ID]
    shadow = runs[SHADOW_ID]
    active = runs[ACTIVE_ID]
    checks.update(
        {
            "served_is_coverage_off_fast": served.get("capture", {}).get("configuration")
            == {
                "retrieval_mode": "coverage",
                "rerank_mode": "off",
                "question_policy": "fast",
                "sidecar_opened": False,
            },
            "control_equals_served": control.get("functional_result_sha256")
            == served.get("functional_result_sha256")
            and control.get("response_trace_sha256") == served.get("response_trace_sha256"),
            "shadow_equals_control": shadow.get("functional_result_sha256")
            == control.get("functional_result_sha256")
            and shadow.get("response_trace_sha256") == control.get("response_trace_sha256"),
            "active_top10_and_tail_boundary_clean": active.get("audit", {}).get(
                "top10_membership_violation_count"
            )
            == 0
            and active.get("audit", {}).get("tail_change_count") == 0,
        }
    )
    checks.update(
        {
            f"audit_{name}": passed
            for name, passed in _audit_flags(list(runs.values()), exact_repeat=True).items()
            if name != "exact_repeat"
        }
    )
    return checks


def _gate_probe(
    split: str,
    runs: Mapping[str, Mapping[str, Any]],
    flags: Mapping[str, bool],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    raw = evaluate_p11_gates(
        {
            split: {
                "served": runs[BASELINE_ID],
                "control": runs[CONTROL_ID],
                "candidate": runs[ACTIVE_ID],
            }
        },
        {**flags, "exact_repeat": True},
        bootstrap_seed=int(spec["bootstrap"]["seed"]),
        bootstrap_resamples=int(spec["bootstrap"]["resamples"]),
    )
    prefixes = (f"{split}.", "bootstrap.", "audit.")
    checks = {
        name: passed
        for name, passed in raw["checks"].items()
        if name.startswith(prefixes)
    }
    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
        "deltas": raw["deltas"].get(split, {}),
        "ci": raw["ci"].get(split),
    }


def _deterministic_run_hash(run: Mapping[str, Any]) -> str:
    return _stable_sha256(
        {
            key: value
            for key, value in run.items()
            if key not in {"resources", "_process"}
        }
    )


def _repeat_exact(initial: Mapping[str, Any], repeated: Mapping[str, Any]) -> dict[str, Any]:
    initial_hash = _deterministic_run_hash(initial)
    repeated_hash = _deterministic_run_hash(repeated)
    checks = {
        "fresh_process": _process_token(initial) != _process_token(repeated),
        "deterministic_result_exact": initial_hash == repeated_hash,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "initial_deterministic_sha256": initial_hash,
        "repeated_deterministic_sha256": repeated_hash,
    }


def _run_summary(run: Mapping[str, Any]) -> dict[str, Any]:
    capture = run["capture"]
    return {
        "role": run["role"],
        "metrics": run["metrics"],
        "scenario_metrics": run["scenario_metrics"],
        "resources": run["resources"],
        "functional_result_sha256": run["functional_result_sha256"],
        "response_trace_sha256": run["response_trace_sha256"],
        "asset_validation": run.get("asset_validation", {}),
        "configuration": capture["configuration"],
        "stats": capture["stats"],
        "capture_hashes": capture["hashes"],
        "function_hashes": capture["function_hashes"],
        "audit": run["audit"],
        "process_identity_sha256": _process_token(run),
    }


def _repeat_roles(
    split: str,
    initial: Mapping[str, Mapping[str, Any]],
    samples: list[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
    *,
    role_runner: RoleRunner,
    runtime: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    repeated = _run_roles(
        REPEAT_ROLES,
        samples,
        catalog_ids,
        categories,
        products,
        role_runner=role_runner,
        runtime=runtime,
    )
    exact = {role: _repeat_exact(initial[role], repeated[role]) for role in REPEAT_ROLES}
    flags = _audit_flags(list(repeated.values()), exact_repeat=True)
    repeat_gate = _gate_probe(split, repeated, flags, spec)
    passed = (
        all(value["passed"] for value in exact.values())
        and all(flags.values())
        and repeat_gate["passed"]
    )
    return {
        "attempted": True,
        "passed": passed,
        "runs": {role: _run_summary(run) for role, run in repeated.items()},
        "exact": exact,
        "audit_flags": flags,
        "split_gate": repeat_gate,
    }, list(repeated.values())


def _split_artifact(
    corpus: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    boundary: Mapping[str, bool],
    gate: Mapping[str, Any],
    repeat: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "corpus": dict(corpus),
        "initial": {
            "runs": {role: _run_summary(run) for role, run in runs.items()},
            "boundary_checks": dict(boundary),
            "gate": dict(gate),
        },
    }
    if repeat is not None:
        value["repeat"] = dict(repeat)
    return value


def _assert_artifact_safe(
    value: Mapping[str, Any], *, forbidden_identifiers: set[str] | None = None
) -> None:
    payload = _canonical_bytes(value)
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise P11RunnerError("P11 artifact exceeds its byte limit")
    forbidden_keys = {
        "sessions",
        "session_id",
        "sample_id",
        "ground_truth",
        "target",
        "target_id",
        "target_asin",
        "parent_asin",
    }

    identifiers = forbidden_identifiers or set()

    def check_string(text: str) -> None:
        if ASIN_SHAPE.search(text) is not None or any(
            identifier and identifier in text for identifier in identifiers
        ):
            raise P11RunnerError("P11 artifact contains a raw product identifier")

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str) or key.casefold() in forbidden_keys:
                    raise P11RunnerError("P11 artifact contains a raw label key")
                check_string(key)
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            check_string(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise P11RunnerError("P11 artifact contains a non-finite number")

    visit(value)


def _exclusive_marker(
    path: Path,
    value: Mapping[str, Any],
    label: str,
    *,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    try:
        _atomic_write_json(
            path,
            value,
            deadline_monotonic=deadline_monotonic,
            phase=f"{label} marker",
            preserve_published_on_deadline=True,
        )
    except FileExistsError as exc:
        raise P11RunnerError(f"P11 {label} marker already exists") from exc
    return _file_identity(path)


def run_evaluation(
    *,
    prereg_lock_path: Path = DEFAULT_PREREG_LOCK,
    spec_path: Path = DEFAULT_SPEC,
    corpus_protocol_path: Path = DEFAULT_CORPUS_PROTOCOL,
    catalog_path: Path = DEFAULT_CATALOG,
    released_public_path: Path = DEFAULT_RELEASED_PUBLIC,
    evaluation_config_path: Path = DEFAULT_EVALUATION_CONFIG,
    corpus_metadata_path: Path = DEFAULT_CORPUS_METADATA,
    sidecar_path: Path = DEFAULT_SIDECAR,
    sidecar_metadata_path: Path = DEFAULT_SIDECAR_METADATA,
    corpus_paths: Mapping[str, Path] = DEFAULT_CORPORA,
    diagnostic_paths: Mapping[str, Path] = DEFAULT_DIAGNOSTIC_CORPORA,
    worker_path: Path = DEFAULT_WORKER,
    output_path: Path = DEFAULT_OUTPUT,
    attempt_marker_path: Path = DEFAULT_ATTEMPT_MARKER,
    confirmation_marker_path: Path = DEFAULT_CONFIRMATION_MARKER,
    role_runner: RoleRunner | None = None,
    formal: bool = True,
    enforce_formal_git: bool = True,
    require_default_formal_paths: bool = True,
) -> dict[str, Any]:
    formal_started_monotonic = time.monotonic() if formal else None
    formal_deadline_monotonic = (
        formal_started_monotonic + FORMAL_EVALUATION_SECONDS
        if formal_started_monotonic is not None
        else None
    )
    _check_global_deadline(
        formal_deadline_monotonic, "formal argument validation"
    )
    if not formal:
        if role_runner is None:
            raise P11RunnerError("P11 nonformal mode requires an injected fixture runner")
        protected = {
            "prereg_lock": (prereg_lock_path, DEFAULT_PREREG_LOCK),
            "spec": (spec_path, DEFAULT_SPEC),
            "corpus_protocol": (corpus_protocol_path, DEFAULT_CORPUS_PROTOCOL),
            "catalog": (catalog_path, DEFAULT_CATALOG),
            "sidecar": (sidecar_path, DEFAULT_SIDECAR),
            "output": (output_path, DEFAULT_OUTPUT),
            "attempt_marker": (attempt_marker_path, DEFAULT_ATTEMPT_MARKER),
            "confirmation_marker": (
                confirmation_marker_path,
                DEFAULT_CONFIRMATION_MARKER,
            ),
            **{
                f"corpus:{name}": (corpus_paths.get(name), default)
                for name, default in DEFAULT_CORPORA.items()
            },
        }
        if any(
            observed is None or Path(observed).resolve() == default.resolve()
            for observed, default in protected.values()
        ):
            raise P11RunnerError(
                "P11 nonformal fixtures cannot use preregistered formal assets"
            )
    if formal and (
        role_runner is not None
        or enforce_formal_git is not True
        or require_default_formal_paths is not True
    ):
        raise P11RunnerError(
            "P11 formal evaluation forbids injected runners and disabled formal checks"
        )
    if formal and not prereg_lock_path.is_file():
        raise P11RunnerError(
            "P11 formal evaluation requires configs/p11_prereg_lock.json"
        )
    if formal and not _formal_paths_are_defaults(
        prereg_lock_path=prereg_lock_path,
        spec_path=spec_path,
        corpus_protocol_path=corpus_protocol_path,
        catalog_path=catalog_path,
        released_public_path=released_public_path,
        evaluation_config_path=evaluation_config_path,
        corpus_metadata_path=corpus_metadata_path,
        sidecar_path=sidecar_path,
        sidecar_metadata_path=sidecar_metadata_path,
        corpus_paths=corpus_paths,
        diagnostic_paths=diagnostic_paths,
        worker_path=worker_path,
    ):
        raise P11RunnerError("P11 formal evaluation requires every default frozen path")
    if formal and any(
        observed.resolve() != expected.resolve()
        for observed, expected in (
            (output_path, DEFAULT_OUTPUT),
            (attempt_marker_path, DEFAULT_ATTEMPT_MARKER),
            (confirmation_marker_path, DEFAULT_CONFIRMATION_MARKER),
        )
    ):
        raise P11RunnerError("P11 formal output and one-shot marker paths are frozen")
    if formal and output_path.exists():
        raise P11RunnerError("P11 formal output already exists")
    _check_global_deadline(formal_deadline_monotonic, "preflight")
    before = preflight(
        spec_path=spec_path,
        corpus_protocol_path=corpus_protocol_path,
        catalog_path=catalog_path,
        sidecar_path=sidecar_path,
        corpus_paths=corpus_paths,
    )
    formal_proof: dict[str, Any] = {
        "validated": False,
        "reason": "explicit nonformal mode",
    }
    formal_identity_before: dict[str, Any] | None = None
    if formal:
        _check_global_deadline(
            formal_deadline_monotonic, "preregistration validation"
        )
        if set(diagnostic_paths) != set(DIAGNOSTIC_SPLITS):
            raise P11RunnerError("P11 formal diagnostic corpus registry is incomplete")
        formal_proof = validate_prereg_lock(
            prereg_lock_path,
            spec_path=spec_path,
            corpus_protocol_path=corpus_protocol_path,
            catalog_path=catalog_path,
            released_public_path=released_public_path,
            evaluation_config_path=evaluation_config_path,
            corpus_metadata_path=corpus_metadata_path,
            sidecar_path=sidecar_path,
            sidecar_metadata_path=sidecar_metadata_path,
            corpus_paths=corpus_paths,
            diagnostic_paths=diagnostic_paths,
            worker_path=worker_path,
            spec=before["spec"],
            protocol=before["protocol"],
            enforce_git=enforce_formal_git,
            require_defaults=require_default_formal_paths,
            deadline_monotonic=formal_deadline_monotonic,
        )
        _check_global_deadline(
            formal_deadline_monotonic, "preregistration validation completion"
        )
        formal_proof["validated"] = True
        formal_identity_before = _formal_extra_identity_snapshot(
            prereg_lock_path=prereg_lock_path,
            released_public_path=released_public_path,
            evaluation_config_path=evaluation_config_path,
            corpus_metadata_path=corpus_metadata_path,
            sidecar_metadata_path=sidecar_metadata_path,
            diagnostic_paths=diagnostic_paths,
        )
    _check_global_deadline(formal_deadline_monotonic, "runtime dependency loading")
    _load_runtime_dependencies(formal=formal)
    runner = role_runner or _run_role
    _check_global_deadline(formal_deadline_monotonic, "catalog indexing")
    catalog_ids, categories, products = catalog_index(catalog_path)
    _check_global_deadline(
        formal_deadline_monotonic, "catalog indexing completion"
    )
    if len(catalog_ids) != int(before["protocol"]["catalog"]["count"]):
        raise P11RunnerError("P11 parsed catalog count differs from protocol")
    runtime = {
        "catalog_path": catalog_path,
        "sidecar_path": sidecar_path,
        "sidecar_identity": before["identity_snapshot"]["data"]["sidecar"],
        "worker_path": worker_path,
        "global_deadline_monotonic": formal_deadline_monotonic,
    }
    attempt_nonce: str | None = None
    attempt_marker_identity: dict[str, Any] | None = None
    confirmation_marker_identity: dict[str, Any] | None = None
    if formal:
        _check_global_deadline(
            formal_deadline_monotonic, "formal-attempt marker creation"
        )
        if confirmation_marker_path.exists():
            raise P11RunnerError("P11 confirmation-consumption marker already exists")
        attempt_nonce = uuid.uuid4().hex
        attempt_marker_identity = _exclusive_marker(
            attempt_marker_path,
            {
                "schema_version": "p11.formal-attempt.v1",
                "preregistration_sha256": formal_proof["sha256"],
                "source_commit": formal_proof["source_commit"],
                "attempt_nonce": attempt_nonce,
            },
            "formal-attempt",
            deadline_monotonic=formal_deadline_monotonic,
        )
        _check_global_deadline(
            formal_deadline_monotonic, "formal-attempt marker completion"
        )
    observed_order: list[str] = []
    all_runs: list[Mapping[str, Any]] = []
    process_tokens: set[str] = set()

    diagnostic_targets: set[str] = set()
    forbidden_artifact_identifiers: set[str] = set()
    diagnostic_artifact: dict[str, Any] = {
        "validation_executed": False,
        "execution_status": "diagnostics_not_executed",
        "reason": "explicit nonformal mode does not claim failure-slice validation",
        "runs_per_slice": 0,
    }
    if formal:
        diagnostic_summaries: dict[str, Any] = {}
        for name in DIAGNOSTIC_SPLITS:
            _check_global_deadline(
                formal_deadline_monotonic, f"{name} diagnostic validation"
            )
            diagnostic_rows, summary, targets = _load_split(
                name,
                diagnostic_paths[name],
                before["protocol"],
                catalog_ids,
                diagnostic_targets,
            )
            diagnostic_targets.update(targets)
            forbidden_artifact_identifiers.update(targets)
            forbidden_artifact_identifiers.update(
                str(row["sample_id"]) for row in diagnostic_rows
            )
            diagnostic_summaries[name] = summary
        diagnostic_artifact = {
            "validation_executed": True,
            "schema_hash_and_pairwise_disjointness_passed": True,
            "confirmation_disjointness_proof": "frozen_corpus_metadata_and_prereg_lock",
            "execution_status": "diagnostics_not_executed",
            "runs_per_slice": 0,
            "slices": diagnostic_summaries,
        }

    def register(runs: Mapping[str, Mapping[str, Any]]) -> bool:
        tokens = [_process_token(run) for run in runs.values()]
        fresh = all(tokens) and not (set(tokens) & process_tokens) and len(set(tokens)) == len(tokens)
        process_tokens.update(tokens)
        all_runs.extend(runs.values())
        return bool(fresh)

    _check_global_deadline(formal_deadline_monotonic, "primary split loading")
    primary_rows, primary_corpus, primary_targets = _load_split(
        "primary",
        corpus_paths["primary"],
        before["protocol"],
        catalog_ids,
        diagnostic_targets,
    )
    forbidden_artifact_identifiers.update(primary_targets)
    forbidden_artifact_identifiers.update(
        str(row["sample_id"]) for row in primary_rows
    )
    _check_global_deadline(
        formal_deadline_monotonic, "primary source identifier scan"
    )
    primary_source_scan = _postload_source_identifier_scan(
        before["identity_snapshot"]["source"],
        primary_targets,
        target_registry_sha256=primary_corpus["target_registry_sha256"],
    )
    _check_global_deadline(
        formal_deadline_monotonic, "primary source identifier scan completion"
    )
    observed_order.append("primary_initial")
    _check_global_deadline(formal_deadline_monotonic, "primary initial roles")
    primary_runs = _run_roles(
        ROLE_ORDER,
        primary_rows,
        catalog_ids,
        categories,
        products,
        role_runner=runner,
        runtime=runtime,
    )
    primary_globally_fresh = register(primary_runs)
    primary_boundary = _role_boundary_checks(primary_runs)
    primary_boundary["globally_fresh_processes"] = primary_globally_fresh
    primary_flags = _audit_flags(list(primary_runs.values()), exact_repeat=True)
    _check_global_deadline(formal_deadline_monotonic, "primary initial gate")
    primary_gate = _gate_probe("primary", primary_runs, primary_flags, before["spec"])
    primary_eligible = all(primary_boundary.values()) and primary_gate["passed"]
    primary_repeat: dict[str, Any] = {
        "attempted": False,
        "passed": False,
        "reason": "primary initial role or quality gate failed",
    }
    primary_repeat_runs: list[Mapping[str, Any]] = []
    if primary_eligible:
        observed_order.append("primary_exact_repeat_if_eligible")
        _check_global_deadline(formal_deadline_monotonic, "primary exact repeat")
        primary_repeat, primary_repeat_runs = _repeat_roles(
            "primary",
            primary_runs,
            primary_rows,
            catalog_ids,
            categories,
            products,
            role_runner=runner,
            runtime=runtime,
            spec=before["spec"],
        )
        repeated_tokens = {
            summary["process_identity_sha256"]
            for summary in primary_repeat.get("runs", {}).values()
        }
        globally_fresh = not (repeated_tokens & process_tokens) and len(repeated_tokens) == len(REPEAT_ROLES)
        process_tokens.update(repeated_tokens)
        all_runs.extend(primary_repeat_runs)
        primary_repeat["globally_fresh_processes"] = globally_fresh
        primary_repeat["passed"] = bool(primary_repeat["passed"] and globally_fresh)
    primary_passed = primary_eligible and primary_repeat["passed"]

    uniform_artifact: dict[str, Any] = {
        "semantic_parse_executed": False,
        "official_aggregate_executed": False,
        "source_identifier_scan": {
            "executed": False,
            "reason": "uniform-tail was not opened",
        },
        "reason": "primary did not pass initial and exact-repeat gates",
    }
    uniform_passed = False
    uniform_targets: set[str] = set()
    uniform_runs: dict[str, dict[str, Any]] = {}
    if primary_passed:
        observed_order.append("uniform_tail_non_regression_if_primary_repeat_passes")
        _check_global_deadline(formal_deadline_monotonic, "uniform-tail split loading")
        uniform_rows, uniform_corpus, uniform_targets = _load_split(
            "uniform_tail",
            corpus_paths["uniform_tail"],
            before["protocol"],
            catalog_ids,
            diagnostic_targets | primary_targets,
        )
        forbidden_artifact_identifiers.update(uniform_targets)
        forbidden_artifact_identifiers.update(
            str(row["sample_id"]) for row in uniform_rows
        )
        _check_global_deadline(
            formal_deadline_monotonic, "uniform-tail source identifier scan"
        )
        uniform_source_scan = _postload_source_identifier_scan(
            before["identity_snapshot"]["source"],
            uniform_targets,
            target_registry_sha256=uniform_corpus["target_registry_sha256"],
        )
        _check_global_deadline(
            formal_deadline_monotonic,
            "uniform-tail source identifier scan completion",
        )
        uniform_runs = _run_roles(
            ROLE_ORDER,
            uniform_rows,
            catalog_ids,
            categories,
            products,
            role_runner=runner,
            runtime=runtime,
        )
        uniform_fresh = register(uniform_runs)
        uniform_boundary = _role_boundary_checks(uniform_runs)
        uniform_boundary["globally_fresh_processes"] = uniform_fresh
        uniform_flags = _audit_flags(list(uniform_runs.values()), exact_repeat=True)
        _check_global_deadline(formal_deadline_monotonic, "uniform-tail gate")
        uniform_gate = _gate_probe(
            "uniform_tail", uniform_runs, uniform_flags, before["spec"]
        )
        uniform_passed = all(uniform_boundary.values()) and uniform_gate["passed"]
        uniform_artifact = {
            "semantic_parse_executed": True,
            "official_aggregate_executed": True,
            "source_identifier_scan": uniform_source_scan,
            **_split_artifact(
                uniform_corpus, uniform_runs, uniform_boundary, uniform_gate
            ),
        }

    confirmation_artifact: dict[str, Any] = {
        "identity_bytes_hashed_preflight": True,
        "semantic_parse_executed": False,
        "official_aggregate_executed": False,
        "source_identifier_scan": {
            "executed": False,
            "reason": "confirmation was not opened",
        },
        "reason": "primary repeat or uniform-tail non-regression gate failed",
    }
    confirmation_passed = False
    confirmation_runs: dict[str, dict[str, Any]] = {}
    confirmation_targets: set[str] = set()
    confirmation_repeat: dict[str, Any] = {
        "attempted": False,
        "passed": False,
        "reason": "confirmation was not opened",
    }
    if primary_passed and uniform_passed:
        observed_order.append("confirmation_semantic_parse_if_all_previous_gates_pass")
        _check_global_deadline(
            formal_deadline_monotonic, "confirmation marker and split loading"
        )
        if formal:
            confirmation_marker_identity = _exclusive_marker(
                confirmation_marker_path,
                {
                    "schema_version": "p11.confirmation-consumed.v1",
                    "preregistration_sha256": formal_proof["sha256"],
                    "source_commit": formal_proof["source_commit"],
                    "attempt_nonce": attempt_nonce,
                },
                "confirmation-consumption",
                deadline_monotonic=formal_deadline_monotonic,
            )
            _check_global_deadline(
                formal_deadline_monotonic, "confirmation marker completion"
            )
        confirmation_rows, confirmation_corpus, confirmation_targets = _load_split(
            "confirmation",
            corpus_paths["confirmation"],
            before["protocol"],
            catalog_ids,
            diagnostic_targets | primary_targets | uniform_targets,
        )
        forbidden_artifact_identifiers.update(confirmation_targets)
        forbidden_artifact_identifiers.update(
            str(row["sample_id"]) for row in confirmation_rows
        )
        _check_global_deadline(
            formal_deadline_monotonic, "confirmation source identifier scan"
        )
        confirmation_source_scan = _postload_source_identifier_scan(
            before["identity_snapshot"]["source"],
            confirmation_targets,
            target_registry_sha256=confirmation_corpus["target_registry_sha256"],
        )
        _check_global_deadline(
            formal_deadline_monotonic,
            "confirmation source identifier scan completion",
        )
        observed_order.append("confirmation_initial")
        _check_global_deadline(formal_deadline_monotonic, "confirmation initial roles")
        confirmation_runs = _run_roles(
            ROLE_ORDER,
            confirmation_rows,
            catalog_ids,
            categories,
            products,
            role_runner=runner,
            runtime=runtime,
        )
        confirmation_fresh = register(confirmation_runs)
        confirmation_boundary = _role_boundary_checks(confirmation_runs)
        confirmation_boundary["globally_fresh_processes"] = confirmation_fresh
        confirmation_flags = _audit_flags(
            list(confirmation_runs.values()), exact_repeat=True
        )
        _check_global_deadline(formal_deadline_monotonic, "confirmation initial gate")
        confirmation_gate = _gate_probe(
            "confirmation", confirmation_runs, confirmation_flags, before["spec"]
        )
        confirmation_eligible = (
            all(confirmation_boundary.values()) and confirmation_gate["passed"]
        )
        confirmation_repeat_runs: list[Mapping[str, Any]] = []
        if confirmation_eligible:
            observed_order.append("confirmation_exact_repeat_if_eligible")
            _check_global_deadline(
                formal_deadline_monotonic, "confirmation exact repeat"
            )
            confirmation_repeat, confirmation_repeat_runs = _repeat_roles(
                "confirmation",
                confirmation_runs,
                confirmation_rows,
                catalog_ids,
                categories,
                products,
                role_runner=runner,
                runtime=runtime,
                spec=before["spec"],
            )
            repeated_tokens = {
                summary["process_identity_sha256"]
                for summary in confirmation_repeat.get("runs", {}).values()
            }
            globally_fresh = not (repeated_tokens & process_tokens) and len(repeated_tokens) == len(REPEAT_ROLES)
            process_tokens.update(repeated_tokens)
            all_runs.extend(confirmation_repeat_runs)
            confirmation_repeat["globally_fresh_processes"] = globally_fresh
            confirmation_repeat["passed"] = bool(
                confirmation_repeat["passed"] and globally_fresh
            )
        confirmation_passed = confirmation_eligible and confirmation_repeat["passed"]
        confirmation_artifact = {
            "identity_bytes_hashed_preflight": True,
            "semantic_parse_executed": True,
            "official_aggregate_executed": True,
            "source_identifier_scan": confirmation_source_scan,
            **_split_artifact(
                confirmation_corpus,
                confirmation_runs,
                confirmation_boundary,
                confirmation_gate,
                confirmation_repeat,
            ),
        }

    final_gate: dict[str, Any] = {
        "passed": False,
        "checks": {},
        "reasons": ["confirmation did not pass initial and exact-repeat gates"],
        "deltas": {},
        "ci": {},
    }
    if primary_passed and uniform_passed and confirmation_passed:
        observed_order.append("final_promotion_gate")
        _check_global_deadline(formal_deadline_monotonic, "final promotion gate")
        exact_repeat = primary_repeat["passed"] and confirmation_repeat["passed"]
        flags = _audit_flags(all_runs, exact_repeat=exact_repeat)
        final_gate = evaluate_p11_gates(
            {
                "primary": {
                    "served": primary_runs[BASELINE_ID],
                    "control": primary_runs[CONTROL_ID],
                    "candidate": primary_runs[ACTIVE_ID],
                },
                "uniform_tail": {
                    "served": uniform_runs[BASELINE_ID],
                    "control": uniform_runs[CONTROL_ID],
                    "candidate": uniform_runs[ACTIVE_ID],
                },
                "confirmation": {
                    "served": confirmation_runs[BASELINE_ID],
                    "control": confirmation_runs[CONTROL_ID],
                    "candidate": confirmation_runs[ACTIVE_ID],
                },
            },
            flags,
            bootstrap_seed=int(before["spec"]["bootstrap"]["seed"]),
            bootstrap_resamples=int(before["spec"]["bootstrap"]["resamples"]),
        )

    _check_global_deadline(formal_deadline_monotonic, "final identity validation")
    after = _identity_snapshot(
        spec_path=spec_path,
        corpus_protocol_path=corpus_protocol_path,
        catalog_path=catalog_path,
        sidecar_path=sidecar_path,
        corpus_paths=corpus_paths,
    )
    if after != before["identity_snapshot"]:
        raise P11RunnerError("P11 source, config, sidecar, or data changed during evaluation")
    if formal:
        formal_identity_after = _formal_extra_identity_snapshot(
            prereg_lock_path=prereg_lock_path,
            released_public_path=released_public_path,
            evaluation_config_path=evaluation_config_path,
            corpus_metadata_path=corpus_metadata_path,
            sidecar_metadata_path=sidecar_metadata_path,
            diagnostic_paths=diagnostic_paths,
        )
        if formal_identity_after != formal_identity_before:
            raise P11RunnerError(
                "P11 preregistration or formal-only assets changed during evaluation"
            )
        source_after = _load_object(
            prereg_lock_path, "P11 preregistration lock"
        ).get("source")
        if not isinstance(source_after, Mapping):
            raise P11RunnerError("P11 formal source proof disappeared")
        final_git_proof = _verify_formal_git(
            source_after, deadline_monotonic=formal_deadline_monotonic
        )
        if final_git_proof != formal_proof.get("git"):
            raise P11RunnerError("P11 formal Git state changed during evaluation")
        if (
            attempt_marker_identity is None
            or _file_identity(attempt_marker_path) != attempt_marker_identity
            or (
                confirmation_marker_identity is not None
                and _file_identity(confirmation_marker_path)
                != confirmation_marker_identity
            )
        ):
            raise P11RunnerError("P11 one-shot marker changed during evaluation")
    _check_global_deadline(
        formal_deadline_monotonic, "promotion decision and artifact construction"
    )
    candidate_gate_passed = bool(final_gate["passed"])
    promotion = bool(
        formal and formal_proof.get("validated") is True and candidate_gate_passed
    )
    if formal:
        decision = "promote_p11_r01" if promotion else "retain_r08_control"
    else:
        decision = (
            "nonformal_candidate_pass" if candidate_gate_passed else "nonformal_retain"
        )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "winner_id": ACTIVE_ID if promotion else CONTROL_ID,
        "formal_evaluation": formal,
        "candidate_gate_passed": candidate_gate_passed,
        "promotion_eligible": promotion,
        "public_evaluation_run": False,
        "inputs": {
            "identities": before["identity_snapshot"],
            "identity_snapshot_sha256": _stable_sha256(before["identity_snapshot"]),
            "confirmation_identity_bytes_hashed_preflight": True,
            "confirmation_semantic_parse_executed": confirmation_artifact[
                "semantic_parse_executed"
            ],
            "preregistration": formal_proof,
            "formal_only_identity_snapshot_sha256": (
                _stable_sha256(formal_identity_before)
                if formal_identity_before is not None
                else None
            ),
            "one_shot_markers": {
                "formal_attempt": attempt_marker_identity,
                "confirmation_consumed": confirmation_marker_identity,
            },
        },
        "execution": {
            "global_deadline": (
                {
                    "applied": True,
                    "clock": EXPECTED_DEADLINE_POLICY["clock"],
                    "elapsed_seconds": round(
                        time.monotonic() - formal_started_monotonic, 6
                    ),
                    "limit_seconds": int(FORMAL_EVALUATION_SECONDS),
                    "passed": True,
                    "scope": EXPECTED_DEADLINE_POLICY["scope"],
                }
                if formal_started_monotonic is not None
                else {
                    "applied": False,
                    "reason": "formal-only policy; nonformal fixtures are excluded",
                }
            ),
            "observed_order": observed_order,
            "primary": {
                "source_identifier_scan": primary_source_scan,
                **_split_artifact(
                    primary_corpus,
                    primary_runs,
                    primary_boundary,
                    primary_gate,
                    primary_repeat,
                ),
            },
            "uniform_tail": uniform_artifact,
            "confirmation": confirmation_artifact,
            "diagnostic_slices": diagnostic_artifact,
        },
        "promotion_gate": final_gate,
        "provenance": {
            "source_target_blind_scan": before["source_scan"],
            "source_config_data_sidecar_snapshot_stable": True,
            "formal_only_identity_snapshot_stable": formal,
            "fresh_subprocess_count": len(process_tokens),
            "python_version": platform.python_version(),
        },
        "boundary": (
            "The parent alone parses corpus labels and calls the official evaluator. "
            "Fresh workers receive only opaque ordinal/profile/message/turn/top_k requests. "
            "Each split receives an exact post-load source-identifier scan before any "
            "worker for that split; only aggregate proof is retained. Pre-import audit "
            "denials are supervisor-counted and fail-stop the worker. "
            "Confirmation bytes are hashed at preflight but parsed only after primary "
            "eligibility, primary exact repeat, and uniform-tail non-regression pass. "
            "The artifact contains aggregate observations and hashes only."
        ),
    }
    _assert_artifact_safe(
        artifact, forbidden_identifiers=forbidden_artifact_identifiers
    )
    _check_global_deadline(formal_deadline_monotonic, "artifact validation completion")
    if formal:
        artifact["execution"]["global_deadline"]["elapsed_seconds"] = round(
            time.monotonic() - formal_started_monotonic, 6
        )
        _assert_artifact_safe(
            artifact, forbidden_identifiers=forbidden_artifact_identifiers
        )
        _atomic_write_json(
            output_path,
            artifact,
            deadline_monotonic=formal_deadline_monotonic,
            phase="formal evaluation artifact",
        )
    return artifact


def _atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    deadline_monotonic: float | None = None,
    phase: str = "atomic JSON output",
    preserve_published_on_deadline: bool = False,
) -> None:
    _check_global_deadline(deadline_monotonic, f"{phase} existence check")
    if path.exists():
        raise FileExistsError(f"P11 output already exists: {path}")
    _check_global_deadline(deadline_monotonic, f"{phase} directory preparation")
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_global_deadline(deadline_monotonic, f"{phase} serialization")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        _check_global_deadline(deadline_monotonic, f"{phase} temporary creation")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temporary = Path(handle.name)
            _check_global_deadline(deadline_monotonic, f"{phase} temporary write")
            handle.write(payload)
            _check_global_deadline(deadline_monotonic, f"{phase} temporary flush")
            handle.flush()
            _check_global_deadline(deadline_monotonic, f"{phase} temporary fsync")
            os.fsync(handle.fileno())
        _check_global_deadline(deadline_monotonic, f"{phase} atomic publication")
        os.link(temporary, path)
        try:
            _check_global_deadline(
                deadline_monotonic, f"{phase} atomic publication completion"
            )
        except P11RunnerError:
            if not preserve_published_on_deadline:
                try:
                    path.unlink()
                except OSError as cleanup_error:
                    raise P11RunnerError(
                        f"P11 {phase} exceeded its deadline and cleanup failed"
                    ) from cleanup_error
            raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg-lock", type=Path, default=DEFAULT_PREREG_LOCK)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--corpus-protocol", type=Path, default=DEFAULT_CORPUS_PROTOCOL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--public-set", type=Path, default=DEFAULT_RELEASED_PUBLIC)
    parser.add_argument(
        "--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG
    )
    parser.add_argument("--corpus-metadata", type=Path, default=DEFAULT_CORPUS_METADATA)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument(
        "--sidecar-metadata", type=Path, default=DEFAULT_SIDECAR_METADATA
    )
    parser.add_argument("--primary", type=Path, default=DEFAULT_CORPORA["primary"])
    parser.add_argument("--uniform-tail", type=Path, default=DEFAULT_CORPORA["uniform_tail"])
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CORPORA["confirmation"])
    for name, default in DEFAULT_DIAGNOSTIC_CORPORA.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus_paths = {
        "primary": args.primary,
        "uniform_tail": args.uniform_tail,
        "confirmation": args.confirmation,
    }
    diagnostic_paths = {
        name: getattr(args, name) for name in DEFAULT_DIAGNOSTIC_CORPORA
    }
    formal = True
    if args.dry_preflight:
        if not args.prereg_lock.is_file():
            raise P11RunnerError(
                "P11 formal evaluation requires configs/p11_prereg_lock.json"
            )
        state = preflight(
            spec_path=args.spec,
            corpus_protocol_path=args.corpus_protocol,
            catalog_path=args.catalog,
            sidecar_path=args.sidecar,
            corpus_paths=corpus_paths,
        )
        formal_proof = validate_prereg_lock(
            args.prereg_lock,
            spec_path=args.spec,
            corpus_protocol_path=args.corpus_protocol,
            catalog_path=args.catalog,
            released_public_path=args.public_set,
            evaluation_config_path=args.evaluation_config,
            corpus_metadata_path=args.corpus_metadata,
            sidecar_path=args.sidecar,
            sidecar_metadata_path=args.sidecar_metadata,
            corpus_paths=corpus_paths,
            diagnostic_paths=diagnostic_paths,
            worker_path=args.worker,
            spec=state["spec"],
            protocol=state["protocol"],
            enforce_git=True,
            require_defaults=True,
        )
        formal_proof["validated"] = True
        _load_runtime_dependencies(formal=True)
        smoke = worker_smoke_preflight(
            catalog_path=args.catalog,
            sidecar_path=args.sidecar,
            worker_path=args.worker,
        )
        print(
            json.dumps(
                {
                    "identity_snapshot_sha256": _stable_sha256(
                        state["identity_snapshot"]
                    ),
                    "source_target_blind_scan": state["source_scan"],
                    "formal_evaluation": formal,
                    "preregistration": formal_proof,
                    "worker_smoke_preflight": smoke,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.output.exists():
        raise FileExistsError(f"P11 output already exists: {args.output}")
    artifact = run_evaluation(
        prereg_lock_path=args.prereg_lock,
        spec_path=args.spec,
        corpus_protocol_path=args.corpus_protocol,
        catalog_path=args.catalog,
        released_public_path=args.public_set,
        evaluation_config_path=args.evaluation_config,
        corpus_metadata_path=args.corpus_metadata,
        sidecar_path=args.sidecar,
        sidecar_metadata_path=args.sidecar_metadata,
        corpus_paths=corpus_paths,
        diagnostic_paths=diagnostic_paths,
        worker_path=args.worker,
        output_path=args.output,
        formal=formal,
    )
    print(
        f"[p11] decision={artifact['decision']} winner={artifact['winner_id']} wrote={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
