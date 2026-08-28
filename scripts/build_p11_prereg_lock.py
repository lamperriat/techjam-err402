from __future__ import annotations

"""Create the immutable P11 preregistration lock without running sessions."""

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import CodeType
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "p11.prereg-lock.v1"
SPEC_SCHEMA_VERSION = "p11.top10-experiment.v1"
CORPUS_PROTOCOL_SCHEMA_VERSION = "p11.corpus-protocol.v1"
CORPUS_METADATA_SCHEMA_VERSION = "p11.corpora.v1"

EXPECTED_CATALOG_SHA256 = (
    "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
)
EXPECTED_PUBLIC_SHA256 = (
    "571359a8a69014c43fc30d39c996c4a28e875dccc249dffc707358757beb16c0"
)
EXPECTED_PUBLIC_CANONICAL_SHA256 = (
    "6c726257fec25575716ee65b095f94c48402b6e14e83341518610f45fbfbec6d"
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

EXPECTED_EXECUTION_ORDER = (
    "primary_initial",
    "primary_exact_repeat_if_eligible",
    "uniform_tail_non_regression_if_primary_repeat_passes",
    "confirmation_semantic_parse_if_all_previous_gates_pass",
    "confirmation_initial",
    "confirmation_exact_repeat_if_eligible",
    "final_promotion_gate",
)
EXPECTED_ROLES = {
    "active": "P11.R01.top10_linear",
    "control": "P11.C00.r08_coverage",
    "served": "P11.B00.served_agent",
    "shadow": "P11.S00.top10_linear_shadow",
}
EXPECTED_BOOTSTRAP = {
    "confidence": 0.95,
    "lower_bound_strictly_above_zero": True,
    "method": "paired_nearest_rank_percentile",
    "resamples": 10_000,
    "seed": 20260829,
    "unit": "paired_unrounded_session_technical_score_delta",
}
EXPECTED_DEADLINE_POLICY = {
    "clock": "time.monotonic",
    "dry_preflight_applies": False,
    "formal_evaluation_seconds": 5_400,
    "scope": "entire_formal_run_including_all_role_process_phases",
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
EXPECTED_ARTIFACT_POLICY = {
    "aggregate_and_hash_only": True,
    "atomic_fail_if_exists": True,
    "raw_asins_allowed": False,
    "raw_sample_ids_allowed": False,
    "raw_sessions_allowed": False,
}
EXPECTED_SIDECAR_POLICY = {
    "catalog_only": True,
    "identity_hashed_before_and_after": True,
    "maximum_bytes": 33_554_432,
    "opened_by_active_and_shadow_only": True,
}
EXPECTED_FEATURE_CONTRACT = {
    "feature_registry_sha256": "c2c6b4309e5bbf8e092f625957ae5f0cdeb193adcc48d552e5291837803749b1",
    "feature_schema_version": "p11.top10-features.v2",
    "feature_semantics_sha256": "abae7be9ab9073593ca40309177408adf20e460e3153fe95ec942fb53b47a488",
    "scorer_version": "p11.top10-linear.v3",
    "tail_preserved": True,
    "top10_membership_preserved": True,
}
EXPECTED_SCENARIO_COUNTS = {
    "boundary": 10,
    "browsing": 80,
    "buying": 80,
    "intent_override": 30,
}
EXPECTED_SPLIT_ROWS = {
    "primary": 200,
    "uniform_tail": 200,
    "confirmation": 200,
    "failure_negative": 80,
    "failure_budget": 80,
    "failure_override": 80,
    "failure_missing_evidence": 80,
}
EXPECTED_SPLIT_SCENARIOS = {
    "primary": EXPECTED_SCENARIO_COUNTS,
    "uniform_tail": EXPECTED_SCENARIO_COUNTS,
    "confirmation": EXPECTED_SCENARIO_COUNTS,
    "failure_negative": {
        "boundary": 4,
        "browsing": 32,
        "buying": 32,
        "intent_override": 12,
    },
    "failure_budget": {"browsing": 40, "buying": 40},
    "failure_override": {"intent_override": 80},
    "failure_missing_evidence": {"browsing": 40, "buying": 40},
}
EXPECTED_CORPUS_SHA256 = {
    "primary": "1d578694c3226d1b008d2c9f2f252ed63d114a544c82c218c06116b13c00cf84",
    "uniform_tail": "87d2334dd28dded92df2d8c8897f7f9552efb655bc74488d49dafe2f6efc1dfd",
    "confirmation": "6dfdcdaf8cd6a091a9b82c192b076ad4e48a89b4023d5ef65394a6d6daf737ba",
    "failure_negative": "c0c593dc90af45ec9f3dcdfaaace286f9b0a53c52d0a833c8d292a4488126290",
    "failure_budget": "a522134897f7ab8348c327a9a53d30075033dc379bcec610777201abfbb6ee91",
    "failure_override": "1eeb7e552f2ef0ce8aae413adb7a6393891f1e265c774728ed6ba3b35685df95",
    "failure_missing_evidence": "2aca6b723b592b84caf173fb55c231e8572d0844da9f57cd0c89b9e0489f4ef9",
}
EXPECTED_OPENED_CORPORA = {
    "p1": {
        "canonical_samples_sha256": "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae",
        "path": "experiments/p1_derived_product_disjoint.jsonl",
        "rows": 200,
        "sample_id_prefix": "derived_p1_",
    },
    "p5": {
        "canonical_samples_sha256": "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c",
        "path": "experiments/p5_selection_product_disjoint.jsonl",
        "rows": 200,
        "sample_id_prefix": "derived_p5_",
    },
    "p6": {
        "canonical_samples_sha256": "27544cdb6ed9495808c35bbab09b4dbadcb88a1d75d162f17bb4fba6ee8841c7",
        "path": "experiments/p6_selection_product_disjoint.jsonl",
        "rows": 200,
        "sample_id_prefix": "derived_p6_",
    },
    "p7": {
        "canonical_samples_sha256": "bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546",
        "path": "experiments/p7_selection_product_disjoint.jsonl",
        "rows": 200,
        "sample_id_prefix": "derived_p7_",
    },
    "p8_confirmation": {
        "canonical_samples_sha256": "3ae6f8ff7ab0362399b348c3443daa5b7138aab9cf72e944b7e11dd71d7d3dde",
        "path": "experiments/p8_confirmation_product_disjoint.jsonl",
        "rows": 200,
        "sample_id_prefix": "derived_p8_confirmation_",
    },
    "p8_selection": {
        "canonical_samples_sha256": "1c11d73d7c8ced617ce874e15a563f240731ca9654ed42bcc4f773b7b4da81ee",
        "path": "experiments/p8_selection_product_disjoint.jsonl",
        "rows": 200,
        "sample_id_prefix": "derived_p8_selection_",
    },
    "p9_confirmation": {
        "canonical_samples_sha256": "4bbd9d53f32e3773de18bab881ba6e5ef0887ca86701897798ee086430ed08d9",
        "path": "experiments/p9_confirmation_product_disjoint.jsonl",
        "rows": 200,
        "sample_id_prefix": "derived_p9_confirmation_",
    },
    "p9_selection": {
        "canonical_samples_sha256": "6298cbd6d7507f4b163ab4979a86ff109e0dffa90557e3b28e5d20d129e5be9f",
        "path": "experiments/p9_selection_product_disjoint.jsonl",
        "rows": 200,
        "sample_id_prefix": "derived_p9_selection_",
    },
    "released_public": {
        "canonical_samples_sha256": EXPECTED_PUBLIC_CANONICAL_SHA256,
        "path": "data/public_set.jsonl",
        "rows": 200,
        "sample_id_prefix": "public_",
    },
}
EXPECTED_OPENED_TARGET_UNION_COUNT = 1_800

FORMAL_SPLITS = ("primary", "uniform_tail", "confirmation")
DIAGNOSTIC_SPLITS = (
    "failure_negative",
    "failure_budget",
    "failure_override",
    "failure_missing_evidence",
)
ALL_SPLITS = (*FORMAL_SPLITS, *DIAGNOSTIC_SPLITS)

SOURCE_PATHS = {
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
RUNTIME_ASIN_SCAN_NAMES = tuple(SOURCE_PATHS)
_ASIN_LITERAL_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE)
MAX_STATIC_SCAN_VALUE_BYTES = 1_048_576
MAX_STATIC_SCAN_VALUES = 100_000

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


class PreregLockError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenExpectations:
    catalog_sha256: str = EXPECTED_CATALOG_SHA256
    catalog_rows: int = 50_000
    public_sha256: str = EXPECTED_PUBLIC_SHA256
    public_canonical_sha256: str = EXPECTED_PUBLIC_CANONICAL_SHA256
    public_git_blob_sha1: str = EXPECTED_PUBLIC_GIT_BLOB_SHA1
    public_rows: int = 200
    public_scenarios: Mapping[str, int] | None = None
    evaluator_sha256: str = EXPECTED_EVALUATOR_SHA256
    evaluator_git_blob_sha1: str = EXPECTED_EVALUATOR_GIT_BLOB_SHA1
    evaluation_config_sha256: str = EXPECTED_EVALUATION_CONFIG_SHA256
    evaluation_config_git_blob_sha1: str = EXPECTED_EVALUATION_CONFIG_GIT_BLOB_SHA1
    split_rows: Mapping[str, int] | None = None
    split_scenarios: Mapping[str, Mapping[str, int]] | None = None
    corpus_sha256: Mapping[str, str] | None = None
    opened_corpora: Mapping[str, Mapping[str, Any]] | None = None
    opened_target_union_count: int = EXPECTED_OPENED_TARGET_UNION_COUNT

    def rows(self) -> dict[str, int]:
        return dict(self.split_rows or EXPECTED_SPLIT_ROWS)

    def public_scenario_counts(self) -> dict[str, int]:
        return dict(self.public_scenarios or EXPECTED_SCENARIO_COUNTS)

    def scenarios(self) -> dict[str, dict[str, int]]:
        source = self.split_scenarios or EXPECTED_SPLIT_SCENARIOS
        return {name: dict(value) for name, value in source.items()}

    def corpus_hashes(self) -> dict[str, str]:
        return dict(self.corpus_sha256 or EXPECTED_CORPUS_SHA256)

    def opened_registry(self) -> dict[str, dict[str, Any]]:
        source = self.opened_corpora or EXPECTED_OPENED_CORPORA
        return {name: dict(value) for name, value in source.items()}


@dataclass(frozen=True)
class LockPaths:
    spec: Path
    corpus_protocol: Path
    catalog: Path
    released_public: Path
    evaluator: Path
    evaluation_config: Path
    corpus_metadata: Path
    corpora: Mapping[str, Path]
    sidecar: Path
    sidecar_metadata: Path
    output: Path


def default_paths(project_root: Path = PROJECT_ROOT) -> LockPaths:
    root = project_root.resolve()
    return LockPaths(
        spec=root / "configs" / "p11_top10_experiment.json",
        corpus_protocol=root / "configs" / "p11_corpus_protocol.json",
        catalog=root / "data" / "catalog.jsonl",
        released_public=root / "data" / "public_set.jsonl",
        evaluator=root / "evaluator" / "local_evaluator.py",
        evaluation_config=root / "docs" / "evaluation_config.json",
        corpus_metadata=root / "experiments" / "p11_corpora.metadata.json",
        corpora={
            "primary": root / "experiments" / "p11_primary_representative.jsonl",
            "uniform_tail": root / "experiments" / "p11_uniform_tail.jsonl",
            "confirmation": root / "experiments" / "p11_confirmation.jsonl",
            "failure_negative": root / "experiments" / "p11_failure_negative.jsonl",
            "failure_budget": root / "experiments" / "p11_failure_budget.jsonl",
            "failure_override": root / "experiments" / "p11_failure_override.jsonl",
            "failure_missing_evidence": root
            / "experiments"
            / "p11_failure_missing_evidence.jsonl",
        },
        sidecar=root / "experiments" / "p11_features.sqlite",
        sidecar_metadata=root / "experiments" / "p11_features.metadata.json",
        output=root / "configs" / "p11_prereg_lock.json",
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise PreregLockError(f"locked path escapes project root: {path}") from exc
    if not resolved.is_file():
        raise PreregLockError(f"locked file is missing: {relative}")
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreregLockError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise PreregLockError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PreregLockError(f"JSONL row is not an object: {path}")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreregLockError(f"invalid JSONL: {path}") from exc
    return rows


def _canonical_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_bytes(row) + b"\n")
    return digest.hexdigest()


def _git_blob_sha1_lf(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _raw_git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _git(project_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={project_root.resolve().as_posix()}",
                *arguments,
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        raise PreregLockError(f"Git command timed out: {' '.join(arguments)}") from exc
    if completed.returncode:
        raise PreregLockError(f"Git command failed: {' '.join(arguments)}")
    return completed.stdout.strip()


def _advertised_head(project_root: Path, remote: str, ref: str) -> str:
    advertised = _git(project_root, "ls-remote", "--heads", remote, ref)
    lines = [line for line in advertised.splitlines() if line.strip()]
    match = (
        re.fullmatch(rf"([a-f0-9]{{40}})\t{re.escape(ref)}", lines[0])
        if len(lines) == 1
        else None
    )
    if match is None:
        raise PreregLockError(f"remote {remote} did not advertise exactly {ref}")
    return match.group(1)


def capture_pushed_clean_revision(project_root: Path) -> dict[str, Any]:
    branch = _git(project_root, "branch", "--show-current")
    head = _git(project_root, "rev-parse", "HEAD")
    status = _git(project_root, "status", "--porcelain=v1", "--untracked-files=all")
    if not branch:
        raise PreregLockError("P11 lock requires a named Git branch")
    if re.fullmatch(r"[a-f0-9]{40}", head) is None:
        raise PreregLockError("P11 lock requires a valid HEAD commit")
    if status:
        raise PreregLockError("P11 lock requires a completely clean worktree")

    origin_url = _git(project_root, "remote", "get-url", "origin")
    upstream_url = _git(project_root, "remote", "get-url", "upstream")
    if origin_url != EXPECTED_ORIGIN_URL:
        raise PreregLockError("P11 lock requires the frozen credential-free origin URL")
    if upstream_url != EXPECTED_UPSTREAM_URL:
        raise PreregLockError("P11 lock requires the official upstream URL")

    branch_ref = f"refs/heads/{branch}"
    origin_head = _advertised_head(project_root, "origin", branch_ref)
    if origin_head != head:
        raise PreregLockError("P11 lock requires origin branch proof to equal HEAD")

    upstream_ref = "refs/heads/main"
    upstream_head = _advertised_head(project_root, "upstream", upstream_ref)
    local_upstream = _git(project_root, "rev-parse", "refs/remotes/upstream/main")
    if upstream_head != local_upstream or upstream_head != EXPECTED_UPSTREAM_HEAD:
        raise PreregLockError("P11 lock requires the frozen latest official upstream HEAD")
    merge_base = _git(project_root, "merge-base", head, upstream_head)
    if merge_base != upstream_head:
        raise PreregLockError("P11 HEAD is not based on the frozen official upstream HEAD")

    return {
        "git_commit": head,
        "git_branch": branch,
        "remote_proof": {
            "remote": "origin",
            "head_ref": branch_ref,
            "advertised_head": origin_head,
            "url_sha256": hashlib.sha256(origin_url.encode()).hexdigest(),
            "verified": True,
        },
        "official_upstream": {
            "remote": "upstream",
            "head_ref": upstream_ref,
            "advertised_head": upstream_head,
            "local_tracking_head": local_upstream,
            "is_head_ancestor": True,
            "url_sha256": hashlib.sha256(upstream_url.encode()).hexdigest(),
            "verified": True,
        },
    }


def _required_paths_from_runner(runner_path: Path) -> dict[str, str]:
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SOURCE_PATHS"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if (
            isinstance(value, dict)
            and value
            and all(isinstance(key, str) and isinstance(path, str) for key, path in value.items())
        ):
            return dict(value)
    raise PreregLockError("evaluate_p11.REQUIRED_SOURCE_PATHS must be a literal string map")


def _tracked_head_identity(path: Path, project_root: Path) -> dict[str, Any]:
    entry = _file_entry(path, project_root)
    relative = str(entry["path"])
    _git(project_root, "ls-files", "--error-unmatch", relative)
    # --path applies the repository's clean filter and line-ending policy.
    working_blob = _git(project_root, "hash-object", f"--path={relative}", "--", relative)
    head_blob = _git(project_root, "rev-parse", f"HEAD:{relative}")
    if re.fullmatch(r"[a-f0-9]{40}", working_blob) is None or working_blob != head_blob:
        raise PreregLockError(f"working content differs from HEAD Git blob: {relative}")
    return {**entry, "git_blob_sha1": head_blob}


def capture_source_lock(project_root: Path, revision: Mapping[str, Any]) -> dict[str, Any]:
    required = _required_paths_from_runner(project_root / SOURCE_PATHS["evaluate_p11"])
    if required != SOURCE_PATHS:
        raise PreregLockError("lock builder source registry differs from evaluate_p11")
    return {
        **revision,
        "files": {
            name: _tracked_head_identity(project_root / relative, project_root)
            for name, relative in sorted(required.items())
        },
    }


def _catalog_identity(path: Path, expected_rows: int) -> tuple[dict[str, Any], set[str]]:
    identifiers: set[str] = set()
    rows = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                identifier = str(value.get("parent_asin") or "") if isinstance(value, dict) else ""
                if not identifier or identifier in identifiers:
                    raise PreregLockError("official catalog has an empty or duplicate identifier")
                identifiers.add(identifier)
                rows += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreregLockError("official catalog is not valid JSONL") from exc
    if rows != expected_rows:
        raise PreregLockError("official catalog row count differs from frozen expectations")
    return {"rows": rows, "unique_ids": len(identifiers)}, identifiers


def _inspect_conversation_rows(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    sample_ids: list[str] = []
    targets: list[str] = []
    scenarios: Counter[str] = Counter()
    for row in rows:
        sample_id = row.get("sample_id")
        target = row.get("ground_truth", {}).get("parent_asin")
        scenario = row.get("scenario_type")
        if not all(isinstance(value, str) and value for value in (sample_id, target, scenario)):
            raise PreregLockError(f"conversation row schema is invalid: {path}")
        sample_ids.append(sample_id)
        targets.append(target)
        scenarios[scenario] += 1
    if len(set(sample_ids)) != len(rows) or len(set(targets)) != len(rows):
        raise PreregLockError(f"conversation identifiers are not unique: {path}")
    return {
        "rows": len(rows),
        "canonical_samples_sha256": _canonical_rows_sha256(rows),
        "scenario_counts": dict(sorted(scenarios.items())),
        "target_ids": set(targets),
    }


def _feature_contract() -> dict[str, Any]:
    # The preregistration process must never import candidate runtime code.
    # These literals are frozen alongside this tracked builder and checked
    # against both the experiment spec and sidecar metadata.
    return dict(EXPECTED_FEATURE_CONTRACT)


def _validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
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
    if set(spec) != expected_roots:
        raise PreregLockError("P11 experiment spec root is not frozen")
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise PreregLockError("P11 experiment spec schema is invalid")
    feature_contract = _feature_contract()
    if spec.get("roles") != EXPECTED_ROLES:
        raise PreregLockError("P11 role IDs differ from the frozen registry")
    if spec.get("feature_contract") != feature_contract:
        raise PreregLockError("P11 feature registry or semantics are not frozen")
    if tuple(spec.get("execution_order") or ()) != EXPECTED_EXECUTION_ORDER:
        raise PreregLockError("P11 execution order differs from the frozen protocol")
    if spec.get("public_evaluation_run") is not False:
        raise PreregLockError("P11 preregistration forbids a public evaluation")
    if spec.get("served_control") != {
        "question_policy": "fast",
        "rerank_mode": "off",
        "retrieval_mode": "coverage",
    }:
        raise PreregLockError("P11 served control is not coverage/off/fast")
    if spec.get("bootstrap") != EXPECTED_BOOTSTRAP:
        raise PreregLockError("P11 bootstrap protocol is not frozen")
    if spec.get("deadline_policy") != EXPECTED_DEADLINE_POLICY:
        raise PreregLockError("P11 global deadline policy is not frozen")
    if spec.get("promotion_gates") != EXPECTED_PROMOTION_GATES:
        raise PreregLockError("P11 promotion gates are not frozen")
    if spec.get("resource_limits") != EXPECTED_RESOURCE_LIMITS:
        raise PreregLockError("P11 resource limits are not frozen")
    if spec.get("artifact_policy") != EXPECTED_ARTIFACT_POLICY:
        raise PreregLockError("P11 artifact policy is not frozen")
    if spec.get("sidecar_policy") != EXPECTED_SIDECAR_POLICY:
        raise PreregLockError("P11 sidecar policy is not frozen")
    if spec.get("corpus_protocol") != {
        "path": "configs/p11_corpus_protocol.json",
        "schema_version": CORPUS_PROTOCOL_SCHEMA_VERSION,
    }:
        raise PreregLockError("P11 corpus protocol reference is not frozen")
    return feature_contract


def _validate_protocol(
    protocol: Mapping[str, Any], expectations: FrozenExpectations
) -> None:
    if protocol.get("schema_version") != CORPUS_PROTOCOL_SCHEMA_VERSION:
        raise PreregLockError("P11 corpus protocol schema is invalid")
    catalog = protocol.get("catalog")
    splits = protocol.get("splits")
    opened = protocol.get("opened_corpora")
    if not all(isinstance(value, Mapping) for value in (catalog, splits, opened)):
        raise PreregLockError("P11 corpus protocol sections are incomplete")
    assert isinstance(catalog, Mapping) and isinstance(splits, Mapping)
    assert isinstance(opened, Mapping)
    if catalog.get("count") != expectations.catalog_rows or catalog.get("sha256") != expectations.catalog_sha256:
        raise PreregLockError("P11 protocol catalog lock differs from official data")
    if set(splits) != set(ALL_SPLITS):
        raise PreregLockError("P11 corpus registry is incomplete")
    rows = expectations.rows()
    scenarios = expectations.scenarios()
    hashes = expectations.corpus_hashes()
    for name in ALL_SPLITS:
        value = splits[name]
        if (
            not isinstance(value, Mapping)
            or value.get("count") != rows[name]
            or dict(value.get("scenario_counts") or {}) != scenarios[name]
            or value.get("expected_samples_sha256") != hashes[name]
        ):
            raise PreregLockError(f"P11 {name} protocol lock differs from expectations")
    expected_opened = expectations.opened_registry()
    if (
        set(opened) != set(expected_opened)
        or any(
            not isinstance(opened.get(name), Mapping)
            or dict(opened[name]) != expected
            for name, expected in expected_opened.items()
        )
        or protocol.get("opened_target_union_count")
        != expectations.opened_target_union_count
    ):
        raise PreregLockError("P11 opened-corpus registry is incomplete")


def _validate_corpus_metadata(
    metadata: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    corpus_entries: Mapping[str, Mapping[str, Any]],
    builder_path: Path,
) -> None:
    if metadata.get("schema_version") != CORPUS_METADATA_SCHEMA_VERSION:
        raise PreregLockError("P11 corpus metadata schema is invalid")
    if metadata.get("protocol_file_sha256") != _sha256_file(protocol_path):
        raise PreregLockError("P11 corpus metadata protocol identity differs")
    if metadata.get("protocol_sha256") != _stable_sha256(protocol):
        raise PreregLockError("P11 corpus metadata canonical protocol differs")
    builder = metadata.get("builder_source")
    catalog = metadata.get("catalog")
    outputs = metadata.get("outputs")
    output_files = metadata.get("output_files")
    if not all(isinstance(value, Mapping) for value in (builder, catalog, outputs, output_files)):
        raise PreregLockError("P11 corpus metadata sections are incomplete")
    if builder.get("sha256") != _sha256_file(builder_path):
        raise PreregLockError("P11 corpus builder identity differs from metadata")
    if (
        catalog.get("product_count") != protocol["catalog"]["count"]
        or catalog.get("sha256") != protocol["catalog"]["sha256"]
    ):
        raise PreregLockError("P11 corpus metadata catalog identity differs")
    if set(outputs) != set(ALL_SPLITS) or set(output_files) != set(ALL_SPLITS):
        raise PreregLockError("P11 corpus metadata output registry is incomplete")
    for name in ALL_SPLITS:
        expected = protocol["splits"][name]
        observed = outputs[name]
        output_file = output_files[name]
        entry = corpus_entries[name]
        if (
            not isinstance(observed, Mapping)
            or not isinstance(output_file, Mapping)
            or observed.get("sample_count") != expected["count"]
            or observed.get("unique_target_count") != expected["count"]
            or observed.get("scenario_counts") != expected["scenario_counts"]
            or observed.get("samples_sha256") != expected["expected_samples_sha256"]
            or output_file.get("sha256") != entry["sha256"]
            or entry["sha256"] != expected["expected_samples_sha256"]
        ):
            raise PreregLockError(f"P11 {name} metadata identity differs")
    overlaps = metadata.get("new_pairwise_target_overlaps")
    expected_pairs = {
        f"{left}__{right}"
        for left, right in combinations(sorted(ALL_SPLITS), 2)
    }
    if (
        not isinstance(overlaps, Mapping)
        or set(overlaps) != expected_pairs
        or any(value != 0 for value in overlaps.values())
        or metadata.get("new_target_union_count") != sum(
            int(protocol["splits"][name]["count"]) for name in ALL_SPLITS
        )
    ):
        raise PreregLockError("P11 corpus metadata does not prove new-target disjointness")
    opened = metadata.get("opened_registry")
    opened_specs = protocol["opened_corpora"]
    expected_opened_pairs = {
        f"{left}__{right}"
        for left, right in combinations(sorted(opened_specs), 2)
    }
    opened_observations = (
        opened.get("corpora") if isinstance(opened, Mapping) else None
    )
    opened_overlaps = (
        opened.get("pairwise_target_overlaps")
        if isinstance(opened, Mapping)
        else None
    )
    if (
        not isinstance(opened, Mapping)
        or set(opened) != {
            "corpora",
            "pairwise_target_overlaps",
            "target_union_count",
        }
        or opened.get("target_union_count") != protocol["opened_target_union_count"]
        or not isinstance(opened_observations, Mapping)
        or set(opened_observations) != set(opened_specs)
        or any(
            not isinstance(opened_observations.get(name), Mapping)
            or dict(opened_observations[name])
            != {
                "rows": spec["rows"],
                "unique_targets": spec["rows"],
                "canonical_samples_sha256": spec["canonical_samples_sha256"],
            }
            for name, spec in opened_specs.items()
        )
        or not isinstance(opened_overlaps, Mapping)
        or set(opened_overlaps) != expected_opened_pairs
        or any(value != 0 for value in opened_overlaps.values())
    ):
        raise PreregLockError("P11 corpus metadata does not prove opened-target disjointness")
    opened_vs_new = metadata.get("opened_vs_new_target_overlaps")
    if (
        not isinstance(opened_vs_new, Mapping)
        or set(opened_vs_new) != set(ALL_SPLITS)
        or any(value != 0 for value in opened_vs_new.values())
    ):
        raise PreregLockError(
            "P11 corpus metadata does not prove opened-to-new target disjointness"
        )
    boundaries = metadata.get("selection_boundaries")
    if (
        not isinstance(boundaries, Mapping)
        or boundaries.get("confirmation_role") != "unopened until candidate and weights are frozen"
        or boundaries.get("released_public_used_for_weight_search") is not False
        or boundaries.get("evaluation_result_json_read") is not False
        or boundaries.get("agent_used") is not False
    ):
        raise PreregLockError("P11 selection boundaries are not frozen")


def _validate_sidecar_metadata(
    metadata: Mapping[str, Any],
    *,
    sidecar_entry: Mapping[str, Any],
    catalog_entry: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
) -> None:
    catalog = metadata.get("catalog")
    sidecar = metadata.get("sidecar")
    if (
        metadata.get("schema_version") != feature_contract["feature_schema_version"]
        or metadata.get("target_blind") is not True
        or metadata.get("label_free") is not True
        or not isinstance(catalog, Mapping)
        or catalog.get("rows") != catalog_entry["rows"]
        or catalog.get("sha256") != catalog_entry["sha256"]
        or not isinstance(sidecar, Mapping)
        or sidecar.get("bytes") != sidecar_entry["bytes"]
        or sidecar.get("sha256") != sidecar_entry["sha256"]
        or sidecar.get("registry_sha256") != feature_contract["feature_registry_sha256"]
        or sidecar.get("semantics_sha256") != feature_contract["feature_semantics_sha256"]
    ):
        raise PreregLockError("P11 sidecar metadata differs from the frozen asset")


def _source_identifier_scan(
    source: Mapping[str, Any], project_root: Path, identifiers: set[str]
) -> dict[str, Any]:
    files = source.get("files")
    if not isinstance(files, Mapping) or not identifiers:
        raise PreregLockError("P11 source identifier scan inputs are incomplete")
    needles = tuple(value.encode("utf-8").lower() for value in sorted(identifiers))
    proof: list[dict[str, Any]] = []
    for name, entry in sorted(files.items()):
        path = project_root / str(entry["path"])
        if any(needle in path.read_bytes().lower() for needle in needles):
            raise PreregLockError("P11 frozen source hardcodes a locked product identifier")
        proof.append({"name": name, "bytes": entry["bytes"], "sha256": entry["sha256"]})
    return {
        "source_file_count": len(proof),
        "identifier_count": len(identifiers),
        "match_count": 0,
        "passed": True,
        "proof_sha256": _stable_sha256(proof),
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


def _source_asin_literal_scan(
    source: Mapping[str, Any], project_root: Path
) -> dict[str, Any]:
    """Reject complete ASIN literals without opening confirmation semantics."""

    files = source.get("files")
    if not isinstance(files, Mapping) or not set(RUNTIME_ASIN_SCAN_NAMES) <= set(files):
        raise PreregLockError("P11 runtime ASIN-literal scan registry is incomplete")
    proof: list[dict[str, Any]] = []
    for name in RUNTIME_ASIN_SCAN_NAMES:
        entry = files[name]
        if not isinstance(entry, Mapping):
            raise PreregLockError("P11 runtime ASIN-literal scan entry is invalid")
        path = project_root / str(entry["path"])
        try:
            static_values = _static_source_text_values(path)
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            raise PreregLockError("P11 runtime source cannot be statically scanned") from exc
        for value in static_values:
            if _ASIN_LITERAL_RE.search(value):
                raise PreregLockError(
                    "P11 runtime source contains a complete ASIN-shaped string literal"
                )
        proof.append({"name": name, "bytes": entry["bytes"], "sha256": entry["sha256"]})
    return {
        "source_file_count": len(proof),
        "asin_literal_count": 0,
        "passed": True,
        "proof_sha256": _stable_sha256(proof),
    }


def _assert_target_free(value: Any, identifiers: set[str]) -> None:
    payload = _canonical_bytes(value).decode("utf-8")
    if any(identifier in payload for identifier in identifiers):
        raise PreregLockError("P11 preregistration lock contains a product identifier")
    prohibited = {"ground_truth", "target", "target_id", "target_asin", "parent_asin", "sample_id"}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if str(key).casefold() in prohibited:
                    raise PreregLockError("P11 preregistration lock contains a raw-label key")
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)


def build_prereg_lock(
    *,
    project_root: Path = PROJECT_ROOT,
    paths: LockPaths | None = None,
    expectations: FrozenExpectations | None = None,
    enforce_git: bool = True,
    require_defaults: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    paths = paths or default_paths(root)
    expectations = expectations or FrozenExpectations()
    if paths.output.exists():
        raise FileExistsError(f"P11 preregistration lock already exists: {paths.output}")
    if require_defaults and paths != default_paths(root):
        raise PreregLockError("formal P11 lock generation requires default paths")
    if set(paths.corpora) != set(ALL_SPLITS):
        raise PreregLockError("P11 lock corpus registry is incomplete")

    if enforce_git:
        revision = capture_pushed_clean_revision(root)
        source = capture_source_lock(root, revision)
        spec_entry = _tracked_head_identity(paths.spec, root)
        protocol_entry = _tracked_head_identity(paths.corpus_protocol, root)
        evaluator_entry = _tracked_head_identity(paths.evaluator, root)
        public_entry = _tracked_head_identity(paths.released_public, root)
        evaluation_config_entry = _tracked_head_identity(paths.evaluation_config, root)
    else:
        revision = {
            "git_commit": "0" * 40,
            "git_branch": "fixture",
            "remote_proof": {"verified": False},
            "official_upstream": {"verified": False},
        }
        source = {
            **revision,
            "files": {
                name: {
                    **_file_entry(root / relative, root),
                    "git_blob_sha1": _raw_git_blob_sha1(root / relative),
                }
                for name, relative in sorted(SOURCE_PATHS.items())
            },
        }
        spec_entry = _file_entry(paths.spec, root)
        protocol_entry = _file_entry(paths.corpus_protocol, root)
        evaluator_entry = {
            **_file_entry(paths.evaluator, root),
            "git_blob_sha1": _raw_git_blob_sha1(paths.evaluator),
        }
        public_entry = {
            **_file_entry(paths.released_public, root),
            "git_blob_sha1": _raw_git_blob_sha1(paths.released_public),
        }
        evaluation_config_entry = {
            **_file_entry(paths.evaluation_config, root),
            "git_blob_sha1": _raw_git_blob_sha1(paths.evaluation_config),
        }

    spec = _load_json_object(paths.spec)
    feature_contract = _validate_spec(spec)
    protocol = _load_json_object(paths.corpus_protocol)
    _validate_protocol(protocol, expectations)

    catalog_entry = _file_entry(paths.catalog, root)
    catalog_observed, catalog_ids = _catalog_identity(paths.catalog, expectations.catalog_rows)
    catalog_entry = {**catalog_entry, **catalog_observed}
    if catalog_entry["sha256"] != expectations.catalog_sha256:
        raise PreregLockError("official catalog SHA-256 differs from frozen expectations")

    public = _inspect_conversation_rows(paths.released_public)
    public_blob = _git_blob_sha1_lf(paths.released_public)
    if (
        public_entry["sha256"] != expectations.public_sha256
        or public_blob != expectations.public_git_blob_sha1
        or public["rows"] != expectations.public_rows
        or public["canonical_samples_sha256"] != expectations.public_canonical_sha256
        or public["scenario_counts"] != expectations.public_scenario_counts()
        or not public["target_ids"] <= catalog_ids
    ):
        raise PreregLockError("official public-set identity differs from frozen expectations")
    if (
        evaluator_entry["sha256"] != expectations.evaluator_sha256
        or evaluator_entry["git_blob_sha1"] != expectations.evaluator_git_blob_sha1
    ):
        raise PreregLockError("official evaluator identity differs from frozen expectations")
    evaluation_config = _load_json_object(paths.evaluation_config)
    if (
        evaluation_config_entry["sha256"] != expectations.evaluation_config_sha256
        or evaluation_config_entry["git_blob_sha1"]
        != expectations.evaluation_config_git_blob_sha1
        or evaluation_config != EXPECTED_EVALUATION_CONFIG
    ):
        raise PreregLockError("official evaluation config differs from frozen expectations")

    corpus_entries = {
        name: _file_entry(paths.corpora[name], root) for name in ALL_SPLITS
    }
    corpus_metadata_entry = _file_entry(paths.corpus_metadata, root)
    corpus_metadata = _load_json_object(paths.corpus_metadata)
    _validate_corpus_metadata(
        corpus_metadata,
        protocol=protocol,
        protocol_path=paths.corpus_protocol,
        corpus_entries=corpus_entries,
        builder_path=root / SOURCE_PATHS["corpus_builder"],
    )

    # Confirmation is deliberately byte-hashed above but never JSON-decoded here.
    parsed_splits = (*FORMAL_SPLITS[:2], *DIAGNOSTIC_SPLITS)
    observations: dict[str, dict[str, Any]] = {}
    seen_targets = set(public["target_ids"])
    for name in parsed_splits:
        observed = _inspect_conversation_rows(paths.corpora[name])
        expected = protocol["splits"][name]
        if (
            observed["rows"] != expected["count"]
            or observed["canonical_samples_sha256"] != expected["expected_samples_sha256"]
            or observed["scenario_counts"] != expected["scenario_counts"]
            or not observed["target_ids"] <= catalog_ids
            or observed["target_ids"] & seen_targets
        ):
            raise PreregLockError(f"P11 {name} schema, identity, or disjointness differs")
        seen_targets.update(observed["target_ids"])
        observations[name] = observed

    sidecar_entry = _file_entry(paths.sidecar, root)
    if sidecar_entry["bytes"] > int(spec["sidecar_policy"]["maximum_bytes"]):
        raise PreregLockError("P11 sidecar exceeds its frozen size limit")
    sidecar_metadata_entry = _file_entry(paths.sidecar_metadata, root)
    sidecar_metadata = _load_json_object(paths.sidecar_metadata)
    _validate_sidecar_metadata(
        sidecar_metadata,
        sidecar_entry=sidecar_entry,
        catalog_entry=catalog_entry,
        feature_contract=feature_contract,
    )

    source_target_scan = _source_identifier_scan(source, root, seen_targets)
    source_asin_literal_scan = _source_asin_literal_scan(source, root)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "source_target_scan": source_target_scan,
        "source_asin_literal_scan": source_asin_literal_scan,
        "official": {
            "catalog": catalog_entry,
            "released_public": {
                **public_entry,
                "rows": public["rows"],
                "scenario_counts": public["scenario_counts"],
                "git_blob_sha1_lf": public_blob,
            },
            "evaluator": evaluator_entry,
            "evaluation_config": {
                **evaluation_config_entry,
                "parsed": evaluation_config,
            },
        },
        "experiment": {
            "spec": spec_entry,
            "corpus_protocol": protocol_entry,
        },
        "corpus_metadata": corpus_metadata_entry,
        "corpora": {
            name: {
                **corpus_entries[name],
                "rows": int(protocol["splits"][name]["count"]),
                "scenario_counts": protocol["splits"][name]["scenario_counts"],
                "semantic_parse_executed": name != "confirmation",
            }
            for name in ALL_SPLITS
        },
        "sidecar": {
            **sidecar_entry,
            "catalog_rows": catalog_entry["rows"],
            "catalog_sha256": catalog_entry["sha256"],
            "registry_sha256": feature_contract["feature_registry_sha256"],
            "semantics_sha256": feature_contract["feature_semantics_sha256"],
        },
        "sidecar_metadata": sidecar_metadata_entry,
        "roles": dict(spec["roles"]),
        "feature_contract": feature_contract,
        "protocol": {
            "deadline_policy": dict(spec["deadline_policy"]),
            "execution_order": list(spec["execution_order"]),
            "promotion_gates": dict(spec["promotion_gates"]),
            "resource_limits": dict(spec["resource_limits"]),
            "artifact_policy": dict(spec["artifact_policy"]),
            "public_evaluation_run": False,
            "confirmation": {
                "locked_by_bytes_and_sha256_only": True,
                "semantic_parse_executed_by_lock_builder": False,
            },
            "formal_splits": list(FORMAL_SPLITS),
            "diagnostic_splits": list(DIAGNOSTIC_SPLITS),
        },
    }
    _assert_target_free(lock, seen_targets)
    return lock


def atomic_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"P11 preregistration lock already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
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
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=defaults.spec)
    parser.add_argument("--corpus-protocol", type=Path, default=defaults.corpus_protocol)
    parser.add_argument("--catalog", type=Path, default=defaults.catalog)
    parser.add_argument("--public-set", type=Path, default=defaults.released_public)
    parser.add_argument("--evaluator", type=Path, default=defaults.evaluator)
    parser.add_argument("--evaluation-config", type=Path, default=defaults.evaluation_config)
    parser.add_argument("--corpus-metadata", type=Path, default=defaults.corpus_metadata)
    for name in ALL_SPLITS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=defaults.corpora[name])
    parser.add_argument("--sidecar", type=Path, default=defaults.sidecar)
    parser.add_argument("--sidecar-metadata", type=Path, default=defaults.sidecar_metadata)
    parser.add_argument("--output", type=Path, default=defaults.output)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = LockPaths(
        spec=args.spec.resolve(),
        corpus_protocol=args.corpus_protocol.resolve(),
        catalog=args.catalog.resolve(),
        released_public=args.public_set.resolve(),
        evaluator=args.evaluator.resolve(),
        evaluation_config=args.evaluation_config.resolve(),
        corpus_metadata=args.corpus_metadata.resolve(),
        corpora={name: getattr(args, name).resolve() for name in ALL_SPLITS},
        sidecar=args.sidecar.resolve(),
        sidecar_metadata=args.sidecar_metadata.resolve(),
        output=args.output.resolve(),
    )
    lock = build_prereg_lock(paths=paths)
    atomic_create(paths.output, lock)
    print(
        f"[p11-lock] commit={lock['source']['git_commit']} wrote={paths.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
