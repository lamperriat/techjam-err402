from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import MAX_TURNS, TOP_K, evaluate, normalize_recommendations
from observer.events import TRACE_SCHEMA_VERSION
from observer.shadow_analysis import (
    SCHEMA_VERSION as SHADOW_ANALYSIS_SCHEMA_VERSION,
    ShadowPolicyRecorder,
)
from observer.trace import (
    TraceRunner,
    _agent_p11_mode,
    _agent_p11_status,
    _agent_retrieval_mode,
    _agent_rerank_mode,
    _create_agent,
    _product_view,
)
from starter.agent import Agent, _terms
from starter.attributes import SCHEMA_VERSION as ATTRIBUTE_SCHEMA_VERSION
from starter.clarification import SCHEMA_VERSION as QUESTION_VALUE_SCHEMA_VERSION
from starter.coverage import SCHEMA_VERSION as COVERAGE_SCHEMA_VERSION
from starter.reranker import SCORER_VERSION
from starter.slot_ledger import SCHEMA_VERSION as SLOT_LEDGER_SCHEMA_VERSION


DOCUMENTS = {
    "readme": ("Project README", "README.md", "Project"),
    "workbench": ("Agent Workbench guide", "docs/agent_workbench.md", "Development"),
    "workflow": ("Development workflow", "docs/development_workflow.md", "Development"),
    "implementation": ("Implementation status", "docs/implementation_log.md", "Development"),
    "competition": ("Competition specification", "docs/competition_specification.md", "Official kit"),
    "submission": ("Submission rules", "docs/submission_rules.md", "Official kit"),
    "contract": ("Agent API contract", "docs/agent_api_contract.json", "Official kit"),
    "evaluation": ("Evaluation configuration", "docs/evaluation_config.json", "Official kit"),
    "baseline": ("Baseline results", "docs/baseline_results.json", "Official kit"),
    "data_inventory": ("Official data inventory", "docs/data_inventory.md", "Audit"),
    "brief": ("Local challenge brief", "problem-statement.md", "Project copy"),
    "plan": ("Internal implementation plan", "docs/internal_plan.md", "Local only"),
    "architecture": ("Current architecture", "docs/current_architecture.md", "Local only"),
    "fusion_handoff": (
        "Teammate + Fusion A/B website handoff",
        "docs/teammate_ab_website_handoff.md",
        "Fusion A/B",
    ),
    "fusion_evidence": (
        "Fusion A/B showcase evidence",
        "docs/teammate_ab_website.json",
        "Fusion A/B",
    ),
    "source_agent": ("Current Agent source", "starter/agent.py", "Source"),
    "source_teammate_t0": (
        "Teammate T0 AgentV1 source",
        "vendor/teammate_v1/err402/agents/v1.py",
        "Fusion A/B source",
    ),
    "source_strict_fusion_ab": (
        "Strict Fusion A/B adapter",
        "starter/teammate_v212_fusion.py",
        "Fusion A/B source",
    ),
    "source_bounded_other": (
        "Bounded other lifecycle",
        "starter/teammate_bounded_other.py",
        "Fusion A/B source",
    ),
    "source_attributes": (
        "Normalized product attributes",
        "starter/attributes.py",
        "Source",
    ),
    "source_reranker": ("Constraint reranker", "starter/reranker.py", "Source"),
    "source_slot_ledger": ("Normalized slot ledger", "starter/slot_ledger.py", "Source"),
    "source_clarification": (
        "Candidate-aware clarification shadow",
        "starter/clarification.py",
        "Source",
    ),
    "source_coverage": (
        "Promoted query-term coverage cascade",
        "starter/coverage.py",
        "Source",
    ),
    "source_shadow_analysis": (
        "Cross-session clarification shadow analysis",
        "observer/shadow_analysis.py",
        "Source",
    ),
    "source_evaluator": ("Local evaluator (official scoring behavior)", "evaluator/local_evaluator.py", "Source"),
    "source_generalization": (
        "P1 generalization evaluator",
        "scripts/evaluate_generalization.py",
        "Source",
    ),
    "source_runtime": ("Workbench runtime source", "observer/runtime.py", "Source"),
    "source_trace": ("Session trace source", "observer/trace.py", "Source"),
    "source_server": ("Workbench HTTP source", "observer/server.py", "Source"),
}


class StaleRuntimeError(RuntimeError):
    """Raised when disk code or evaluation inputs no longer match the loaded runtime."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    return {
        "sample_count": result.get("sample_count"),
        "hit_rate_at_10": result.get("hit_rate_at_10"),
        "mrr": result.get("mrr"),
        "mttc": result.get("mttc"),
        "efficiency": result.get("efficiency"),
        "recommended_technical_score": result.get(
            "recommended_technical_score", result.get("technical_score")
        ),
        "reported_token_usage": result.get("reported_token_usage"),
        "scenario_metrics": result.get("scenario_metrics"),
    }


def _close_agent(agent: Any) -> None:
    close = getattr(agent, "close", None)
    if callable(close):
        close()
        return
    agent.connection.close()


@dataclass
class JobRecord:
    job_id: str
    kind: str
    status: str = "queued"
    current: int = 0
    total: int = 0
    message: str = "Queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    summary: dict[str, Any] | None = None
    logs: list[str] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        with self._lock:
            self.logs.append(line)
            self.logs[:] = self.logs[-300:]
            self.message = message

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": self.job_id,
                "kind": self.kind,
                "status": self.status,
                "current": self.current,
                "total": self.total,
                "progress": 0.0 if not self.total else round(self.current / self.total, 4),
                "message": self.message,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_seconds": self.elapsed_seconds,
                "summary": self.summary,
                "logs": list(self.logs),
            }


class _ProgressAgent:
    def __init__(self, agent: Agent, job: JobRecord, samples: list[dict[str, Any]]) -> None:
        self.agent = agent
        self.job = job
        self.samples = samples
        self.total = len(samples)
        self.started = 0
        self.current_sample: dict[str, Any] = {}
        self.shadow_policy = ShadowPolicyRecorder()

    def reset(self, session_id: str, user_profile: dict) -> None:
        if self.job.cancel_event.is_set():
            raise InterruptedError("Evaluation cancelled")
        self.current_sample = self.samples[self.started]
        self.started += 1
        with self.job._lock:
            self.job.current = max(0, self.started - 1)
            self.job.total = self.total
            self.job.message = f"Evaluating session {self.started}/{self.total}"
        if self.started == 1 or self.started % 10 == 0:
            self.job.log(f"Started session {self.started}/{self.total}")
        self.agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if self.job.cancel_event.is_set():
            raise InterruptedError("Evaluation cancellation requested")
        response = self.agent.respond(session_id, user_message, turn, top_k)
        try:
            shadow = self.agent.debug_rerank_diagnostics(session_id).get(
                "question_shadow"
            )
        except (KeyError, RuntimeError):
            shadow = None
        self.shadow_policy.record(
            sample_id=self.current_sample.get("sample_id", "unknown"),
            scenario_type=self.current_sample.get("scenario_type", "unknown"),
            turn=turn,
            actual_attribute=response.get("ask_attribute"),
            question_shadow=shadow,
        )
        return response


class WorkbenchRuntime:
    def __init__(
        self,
        trace_runner: TraceRunner,
        project_root: Path,
        catalog_path: Path,
        dataset_path: Path,
        results_path: Path,
        baseline_result: dict[str, Any] | None,
        initialized_seconds: float,
    ) -> None:
        self.trace_runner = trace_runner
        self.project_root = project_root
        self.catalog_path = catalog_path
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.experiments_path = project_root / "experiments"
        self.baseline_result = baseline_result or {}
        self.initialized_seconds = initialized_seconds
        self.started_at = _utc_now()
        self.catalog_sha256 = _sha256(catalog_path)
        self.dataset_sha256 = _sha256(dataset_path)
        self._jobs: dict[str, JobRecord] = {}
        self._labs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._agent_lock = threading.RLock()
        self._git = self._git_state()
        self.rerank_mode = _agent_rerank_mode(self.trace_runner.agent)
        self.retrieval_mode = _agent_retrieval_mode(self.trace_runner.agent)
        self.p11_mode = _agent_p11_mode(self.trace_runner.agent)
        self.project_id = hashlib.sha256(
            str(self.project_root).casefold().encode("utf-8")
        ).hexdigest()[:16]
        self._source_paths = {
            "agent": self.project_root / "starter" / "agent.py",
            "attributes": self.project_root / "starter" / "attributes.py",
            "reranker": self.project_root / "starter" / "reranker.py",
            "slot_ledger": self.project_root / "starter" / "slot_ledger.py",
            "clarification": self.project_root / "starter" / "clarification.py",
            "coverage": self.project_root / "starter" / "coverage.py",
            "p11_bridge": self.project_root / "starter" / "p11_bridge.py",
            "p11_features": self.project_root / "starter" / "p11_features.py",
            "shadow_analysis": self.project_root / "observer" / "shadow_analysis.py",
            "evaluator": self.project_root / "evaluator" / "local_evaluator.py",
            "generalization": self.project_root / "scripts" / "evaluate_generalization.py",
        }
        self._input_paths = {
            "catalog": self.catalog_path,
            "dataset": self.dataset_path,
        }
        p11_sidecar_path = _agent_p11_status(self.trace_runner.agent).get(
            "sidecar_path"
        )
        if isinstance(p11_sidecar_path, str) and p11_sidecar_path:
            self._input_paths["p11_sidecar"] = Path(p11_sidecar_path)
        self._loaded_source_hashes = {
            name: _sha256(path) for name, path in self._source_paths.items()
        }
        self._loaded_input_hashes = {
            name: _sha256(path) for name, path in self._input_paths.items()
        }

    @classmethod
    def from_paths(
        cls,
        catalog_path: str | Path = "data/catalog.jsonl",
        dataset_path: str | Path = "data/public_set.jsonl",
        results_path: str | Path = "results.json",
        project_root: str | Path | None = None,
        rerank_mode: str | None = None,
        retrieval_mode: str | None = None,
        p11_mode: str | None = None,
    ) -> WorkbenchRuntime:
        started = time.perf_counter()
        root = Path(project_root or Path.cwd()).resolve()

        def resolve(value: str | Path) -> Path:
            path = Path(value)
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        catalog = resolve(catalog_path)
        dataset = resolve(dataset_path)
        results = resolve(results_path)
        runner = TraceRunner.from_paths(
            catalog,
            dataset,
            results,
            rerank_mode=rerank_mode,
            retrieval_mode=retrieval_mode,
            p11_mode=p11_mode,
        )
        baseline_path = root / "docs" / "baseline_results.json"
        baseline = (
            json.loads(baseline_path.read_text(encoding="utf-8"))
            if baseline_path.exists()
            else None
        )
        return cls(
            runner,
            root,
            catalog,
            dataset,
            results,
            baseline,
            time.perf_counter() - started,
        )

    def _git_state(self) -> dict[str, Any]:
        def run(*args: str) -> str | None:
            try:
                result = subprocess.run(
                    ["git", "-c", f"safe.directory={self.project_root.as_posix()}", *args],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            return result.stdout.strip()

        return {
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": run("rev-parse", "--short", "HEAD"),
            "dirty": bool(run("status", "--short")),
        }

    def health(self) -> dict[str, Any]:
        source_state = self._source_state()
        return {
            "status": "ok",
            "started_at": self.started_at,
            "branch": self._git.get("branch"),
            "commit": self._git.get("commit"),
            "trace_schema": TRACE_SCHEMA_VERSION,
            "project_id": self.project_id,
            "rerank_mode": self.rerank_mode,
            "retrieval_mode": self.retrieval_mode,
            "p11_mode": self.p11_mode,
            "p11": _agent_p11_status(self.trace_runner.agent),
            "restart_required": source_state["restart_required"],
        }

    def _source_state(self) -> dict[str, Any]:
        items = {}
        restart_required = False
        monitored = [
            (name, path, "source", self._loaded_source_hashes.get(name))
            for name, path in self._source_paths.items()
        ] + [
            (name, path, "input", self._loaded_input_hashes.get(name))
            for name, path in self._input_paths.items()
        ]
        for name, path, kind, loaded in monitored:
            disk = _sha256(path)
            changed = loaded != disk
            restart_required = restart_required or changed
            try:
                display_path = str(path.relative_to(self.project_root))
            except ValueError:
                display_path = str(path)
            items[name] = {
                "path": display_path,
                "kind": kind,
                "loaded_sha256": loaded,
                "disk_sha256": disk,
                "changed": changed,
            }
        return {"restart_required": restart_required, "files": items}

    def _require_current_source(self) -> None:
        if self._source_state()["restart_required"]:
            raise StaleRuntimeError(
                "Agent/attributes/reranker/coverage/slot-ledger/clarification/shadow-analysis/"
                "p11-bridge/p11-features/evaluator/generalization "
                "source or loaded catalog/dataset changed after Workbench startup; restart "
                "Workbench before running it"
            )

    def close(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            with job._lock:
                if job.status in {"queued", "running", "cancelling"}:
                    job.cancel_event.set()
                process = job.process
            if process is not None and process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 6.0
        for job in jobs:
            process = job.process
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill()
            thread = job.thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._agent_lock:
            _close_agent(self.trace_runner.agent)

    def overview(self) -> dict[str, Any]:
        agent = self.trace_runner.agent
        with self._agent_lock:
            indexed_rows = int(agent.connection.execute("SELECT count(*) FROM products").fetchone()[0])
        latest = {**self.trace_runner.metrics}
        self._git = self._git_state()
        return {
            "runtime": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
                "started_at": self.started_at,
                "initialization_seconds": round(self.initialized_seconds, 3),
                "trace_schema": TRACE_SCHEMA_VERSION,
                "network_required": False,
                "project_id": self.project_id,
                "rerank_mode": self.rerank_mode,
                "retrieval_mode": self.retrieval_mode,
                "p11_mode": self.p11_mode,
                "p11": _agent_p11_status(agent),
            },
            "repository": self._git,
            "source_state": self._source_state(),
            "data": [
                self._file_status("catalog", self.catalog_path, len(self.trace_runner.products), self.catalog_sha256),
                self._file_status("public sessions", self.dataset_path, len(self.trace_runner.samples), self.dataset_sha256),
                self._file_status("latest results", self.results_path, latest.get("sample_count"), None),
            ],
            "index": {
                "engine": "SQLite FTS5 in-memory",
                "rows": indexed_rows,
                "tokenizer": "unicode61 remove_diacritics 2",
                "fields": ["title", "categories", "features", "details", "store", "description"],
                "bm25_weights": [0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0],
            },
            "pipeline": self.pipeline(),
            "baseline_metrics": _metrics(self.baseline_result),
            "latest_metrics": _metrics(latest),
            "ground_truth_boundary": (
                "The official simulator uses hidden state only to generate permitted user messages. "
                "Raw labels and prior results never enter Agent decision features; target-rank and "
                "scoring annotations are joined after Agent.respond."
            ),
        }

    def _file_status(
        self, label: str, path: Path, records: int | None, sha256: str | None
    ) -> dict[str, Any]:
        return {
            "label": label,
            "path": str(path.relative_to(self.project_root)) if path.is_relative_to(self.project_root) else str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "records": records,
            "sha256": sha256,
        }

    def pipeline(self) -> list[dict[str, Any]]:
        return [
            {
                "layer": "Agent API surface",
                "status": "implemented",
                "detail": (
                    "reset/respond shapes; official harness supplies turns 1-10 and top_k=10; "
                    "Agent validates turn/top_k and emits capped catalog-backed recommendations"
                ),
                "source": "starter/agent.py · evaluator/local_evaluator.py",
            },
            {"layer": "Session state", "status": "implemented", "detail": "Versioned multi-turn terms plus an auditable shadow slot history with active/superseded/deleted lifecycle", "source": "starter/agent.py + starter/slot_ledger.py"},
            {"layer": "Parsing", "status": "implemented", "detail": "Deterministic category, constraint, negation, override, and attribute-class parsing", "source": "starter/agent.py"},
            {"layer": "Sparse retrieval", "status": "implemented", "detail": "Field-weighted SQLite FTS5 broad OR Top 120 and strict AND Top 80 routes", "source": "starter/agent.py"},
            {
                "layer": "Normalized product attributes",
                "status": "implemented",
                "detail": (
                    "Target-blind normalized category, audience, material, color, closure, "
                    "style, use-case, size, width, brand, price, and feature evidence"
                ),
                "source": "starter/attributes.py",
            },
            {"layer": "Dense retrieval", "status": "not implemented", "detail": "Roadmap experiment, not an official requirement", "source": None},
            {"layer": "Fusion", "status": "implemented", "detail": "Deterministic weighted RRF over broad and strict sparse routes", "source": "starter/agent.py"},
            {
                "layer": "Query-term coverage cascade",
                "status": "implemented",
                "mode": self.retrieval_mode,
                "detail": (
                    "Promoted R08 deterministically orders the fused pool by distinct visible "
                    "query-term matches and preserves fused rank on ties; coverage mode serves "
                    "this order, while control mode bypasses it"
                ),
                "source": "starter/coverage.py + starter/agent.py",
            },
            {
                "layer": "Constraint reranking",
                "status": "implemented",
                "mode": self.rerank_mode,
                "detail": (
                    "Deterministic Top-50 constraint scores run only in shadow/active modes; "
                    "off bypasses this layer, shadow leaves the selected retrieval route "
                    "unchanged, and active is permitted only with control retrieval"
                ),
                "source": "starter/attributes.py + starter/reranker.py + starter/agent.py",
            },
            {
                "layer": "P11 frozen Top-10 reranking",
                "status": "implemented",
                "mode": self.p11_mode,
                "detail": (
                    "The frozen sparse/field/constraint scorer may only permute R08's "
                    "existing Top 10; identity or scoring failures preserve full R08 order"
                ),
                "source": "starter/p11_bridge.py + starter/p11_features.py",
            },
            {"layer": "Clarification policy", "status": "implemented", "mode": "shadow diagnostic", "detail": "The served policy remains fixed-order; shadow QuestionValue ranks attributes by information gain, coverage, answerability, and turn cost", "source": "starter/agent.py + starter/clarification.py"},
            {"layer": "Scoring", "status": "implemented", "detail": "Deterministic local evaluator; scoring behavior cross-checked with official 3407835", "source": "evaluator/local_evaluator.py"},
        ]

    def list_sessions(self) -> dict[str, Any]:
        with self._lock:
            return self.trace_runner.list_sessions()

    def trace(self, sample_id: str, refresh: bool = False) -> dict[str, Any]:
        self._require_current_source()
        with self._agent_lock:
            return self.trace_runner.trace(sample_id, refresh)

    def catalog(self, query: str = "", offset: int = 0, limit: int = 30) -> dict[str, Any]:
        offset = max(0, offset)
        limit = max(1, min(50, limit))
        terms = list(dict.fromkeys(_terms(query)))[:20]
        if not terms:
            identifiers = list(self.trace_runner.products)[offset:offset + limit]
            total = len(self.trace_runner.products)
            scores: dict[str, float] = {}
        else:
            expression = " OR ".join(f'"{term}"' for term in terms)
            with self._agent_lock:
                connection = self.trace_runner.agent.connection
                total = int(connection.execute(
                    "SELECT count(*) FROM products WHERE products MATCH ?", (expression,)
                ).fetchone()[0])
                rows = connection.execute(
                    "SELECT parent_asin, "
                    "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS rank_score "
                    "FROM products WHERE products MATCH ? ORDER BY rank_score LIMIT ? OFFSET ?",
                    (expression, limit, offset),
                ).fetchall()
            identifiers = [str(row[0]) for row in rows]
            scores = {str(row[0]): round(float(row[1]), 8) for row in rows}
        return {
            "query": query,
            "terms": terms,
            "offset": offset,
            "limit": limit,
            "total": total,
            "items": [
                {**_product_view(self.trace_runner.products[parent_asin], ""), "bm25_score": scores.get(parent_asin)}
                for parent_asin in identifiers
            ],
        }

    def product(self, parent_asin: str) -> dict[str, Any]:
        if parent_asin not in self.trace_runner.products:
            raise KeyError(f"Unknown parent_asin: {parent_asin}")
        return self.trace_runner.products[parent_asin]

    def documents(self) -> dict[str, Any]:
        items = []
        for document_id, (title, relative, group) in DOCUMENTS.items():
            path = self.project_root / relative
            if path.exists():
                items.append({
                    "document_id": document_id,
                    "title": title,
                    "path": relative,
                    "group": group,
                    "bytes": path.stat().st_size,
                })
        return {"documents": items}

    def document(self, document_id: str) -> dict[str, Any]:
        if document_id not in DOCUMENTS:
            raise KeyError(f"Unknown document: {document_id}")
        title, relative, group = DOCUMENTS[document_id]
        path = self.project_root / relative
        if not path.exists():
            raise KeyError(f"Document is unavailable: {document_id}")
        return {
            "document_id": document_id,
            "title": title,
            "path": relative,
            "group": group,
            "content": path.read_text(encoding="utf-8"),
        }

    def fusion_showcase(self) -> dict[str, Any]:
        """Return tracked, frozen A/B evidence without running an evaluator."""

        path = self.project_root / "docs" / "teammate_ab_website.json"
        if not path.exists():
            raise KeyError("Fusion A/B showcase evidence is unavailable")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("benchmarks"), dict):
            raise ValueError("Fusion A/B showcase evidence is malformed")
        return payload

    def experiments(self) -> dict[str, Any]:
        items = [{
            "experiment_id": "official_weak_bm25",
            "label": "Official weak BM25 reference",
            "kind": "reference",
            "created_at": None,
            "metrics": _metrics(self.baseline_result),
        }]
        if self.trace_runner.metrics:
            items.append({
                "experiment_id": "latest_results",
                "label": "Latest local results.json",
                "kind": "local",
                "created_at": datetime.fromtimestamp(
                    self.results_path.stat().st_mtime, timezone.utc
                ).isoformat(timespec="seconds") if self.results_path.exists() else None,
                "metrics": _metrics(self.trace_runner.metrics),
            })
        if self.experiments_path.exists():
            for manifest_path in sorted(self.experiments_path.glob("*/manifest.json"), reverse=True):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                items.append(manifest)
        return {"experiments": items}

    def jobs(self) -> dict[str, Any]:
        with self._lock:
            records = sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)
        return {"jobs": [job.snapshot() for job in records]}

    def _new_job(self, kind: str) -> tuple[JobRecord, bool]:
        with self._lock:
            for existing in self._jobs.values():
                same_resource = existing.kind == kind or {
                    existing.kind, kind
                } <= {"evaluation", "generalization"}
                if same_resource and existing.status in {
                    "queued", "running", "cancelling", "finalizing"
                }:
                    return existing, False
            job = JobRecord(job_id=f"{kind}_{uuid.uuid4().hex[:10]}", kind=kind)
            self._jobs[job.job_id] = job
            return job, True

    def _capture_provenance(self) -> dict[str, Any]:
        repository = self._git_state()
        self._git = repository
        return {
            "repository": repository,
            "source_hashes": dict(self._loaded_source_hashes),
            "input_hashes": dict(self._loaded_input_hashes),
        }

    def start_evaluation(self) -> dict[str, Any]:
        self._require_current_source()
        job, created = self._new_job("evaluation")
        if not created:
            return job.snapshot()
        provenance = self._capture_provenance()
        job.thread = threading.Thread(
            target=self._run_evaluation, args=(job, provenance), daemon=True
        )
        job.thread.start()
        return job.snapshot()

    def _run_evaluation(self, job: JobRecord, provenance: dict[str, Any]) -> None:
        started = time.perf_counter()
        agent: Agent | None = None
        try:
            self._require_current_source()
            if job.cancel_event.is_set():
                raise InterruptedError("Evaluation cancelled before startup")
            with job._lock:
                job.status = "running"
                job.started_at = _utc_now()
                job.total = len(self.trace_runner.samples)
            job.log(
                "Building a fresh in-memory Agent index "
                f"(retrieval mode: {self.retrieval_mode}; rerank mode: {self.rerank_mode}; "
                f"P11 mode: {self.p11_mode}; "
                "slot ledger: shadow; "
                f"candidate clarification: {'shadow' if self.rerank_mode != 'off' else 'disabled'})"
            )
            agent = _create_agent(
                self.catalog_path,
                rerank_mode=self.rerank_mode,
                retrieval_mode=self.retrieval_mode,
                p11_mode=self.p11_mode,
            )
            samples = list(self.trace_runner.samples.values())
            progress_agent = _ProgressAgent(agent, job, samples)
            result = evaluate(
                progress_agent,
                samples,
                self.trace_runner.catalog_ids,
                self.trace_runner.categories,
                self.trace_runner.products,
            )
            self._require_current_source()
            if job.cancel_event.is_set():
                raise InterruptedError("Evaluation cancelled")
            with job._lock:
                job.status = "finalizing"
                job.message = "Finalizing versioned result artifact"
            job.log("Writing results and versioned experiment artifact")
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_public_eval"
            experiment_dir = self.experiments_path / experiment_id
            experiment_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(experiment_dir / "results.json", result)
            shadow_policy = progress_agent.shadow_policy.artifact()
            self._write_json(experiment_dir / "shadow_policy.json", shadow_policy)
            evaluation_seconds = round(time.perf_counter() - started, 3)
            p11_status = _agent_p11_status(agent)
            repository = provenance["repository"]
            source_hashes = provenance["source_hashes"]
            input_hashes = provenance["input_hashes"]
            manifest = {
                "experiment_id": experiment_id,
                "label": f"Public evaluator · {repository.get('commit') or 'unknown commit'}",
                "kind": "local",
                "created_at": _utc_now(),
                "repository": repository,
                "catalog_sha256": input_hashes["catalog"],
                "dataset_sha256": input_hashes["dataset"],
                "implementation": {
                    "agent": "starter.agent.Agent",
                    "question_policy": agent.question_policy,
                    "rerank_mode": _agent_rerank_mode(agent),
                    "retrieval_mode": _agent_retrieval_mode(agent),
                    "p11_mode": _agent_p11_mode(agent),
                    "p11": p11_status,
                    "agent_source_sha256": source_hashes["agent"],
                    "attributes_source_sha256": source_hashes["attributes"],
                    "reranker_source_sha256": source_hashes["reranker"],
                    "slot_ledger_source_sha256": source_hashes["slot_ledger"],
                    "clarification_source_sha256": source_hashes["clarification"],
                    "coverage_source_sha256": source_hashes["coverage"],
                    "p11_bridge_source_sha256": source_hashes["p11_bridge"],
                    "p11_features_source_sha256": source_hashes["p11_features"],
                    "shadow_analysis_source_sha256": source_hashes["shadow_analysis"],
                    "evaluator_source_sha256": source_hashes["evaluator"],
                    "attribute_schema_version": ATTRIBUTE_SCHEMA_VERSION,
                    "reranker_scorer_version": SCORER_VERSION,
                    "slot_ledger_schema_version": SLOT_LEDGER_SCHEMA_VERSION,
                    "question_value_schema_version": QUESTION_VALUE_SCHEMA_VERSION,
                    "coverage_schema_version": COVERAGE_SCHEMA_VERSION,
                    "clarification_mode": (
                        "shadow" if _agent_rerank_mode(agent) != "off" else "disabled"
                    ),
                    "trace_schema": TRACE_SCHEMA_VERSION,
                },
                "shadow_policy_analysis": {
                    "schema_version": SHADOW_ANALYSIS_SCHEMA_VERSION,
                    "artifact": "shadow_policy.json",
                    "target_blind": True,
                    **shadow_policy["summary"],
                },
                "run": {
                    "python": platform.python_version(),
                    "sqlite": sqlite3.sqlite_version,
                    "sample_count": len(samples),
                    "max_turns": MAX_TURNS,
                    "top_k": TOP_K,
                    "network_required": False,
                    "functional_elapsed_seconds": evaluation_seconds,
                },
                "metrics": _metrics(result),
            }
            self._write_json(experiment_dir / "manifest.json", manifest)
            self._write_json(self.results_path, result)
            with self._lock:
                self.trace_runner.replace_evaluation_result(result)
            with job._lock:
                job.current = job.total
                job.status = "completed"
                job.summary = {
                    **_metrics(result),
                    "rerank_mode": _agent_rerank_mode(agent),
                    "retrieval_mode": _agent_retrieval_mode(agent),
                    "p11_mode": _agent_p11_mode(agent),
                    "p11_effective_mode": p11_status.get("effective_mode"),
                    "shadow_policy": shadow_policy["summary"],
                }
            job.log("Evaluation completed")
        except InterruptedError as exc:
            with job._lock:
                job.status = "cancelled"
            job.log(str(exc))
        except Exception as exc:
            with job._lock:
                job.status = "failed"
            job.log(f"{type(exc).__name__}: {exc}")
        finally:
            if agent is not None:
                try:
                    _close_agent(agent)
                except Exception as exc:
                    with job._lock:
                        job.status = "failed"
                    job.log(f"Agent shutdown failed: {type(exc).__name__}: {exc}")
            with job._lock:
                job.finished_at = _utc_now()
                job.elapsed_seconds = round(time.perf_counter() - started, 3)

    def start_generalization(self) -> dict[str, Any]:
        self._require_current_source()
        job, created = self._new_job("generalization")
        if not created:
            return job.snapshot()
        provenance = self._capture_provenance()
        job.thread = threading.Thread(
            target=self._run_generalization,
            args=(job, provenance),
            daemon=True,
        )
        job.thread.start()
        return job.snapshot()

    def _run_generalization(
        self, job: JobRecord, provenance: dict[str, Any]
    ) -> None:
        started = time.perf_counter()
        try:
            self._require_current_source()
            if job.cancel_event.is_set():
                raise InterruptedError("Generalization evaluation cancelled before startup")
            with job._lock:
                job.status = "running"
                job.started_at = _utc_now()
                job.total = 6
            executable = Path(sys.executable)
            if executable.name.lower() == "pythonw.exe" and executable.with_name("python.exe").exists():
                executable = executable.with_name("python.exe")
            experiment_id = (
                datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_generalization"
            )
            experiment_dir = self.experiments_path / experiment_id
            artifact_path = experiment_dir / "results.json"
            command = [
                str(executable),
                str(self.project_root / "scripts" / "evaluate_generalization.py"),
                "--catalog",
                str(self.catalog_path),
                "--dataset",
                str(self.dataset_path),
                "--corpus",
                "both",
                "--suite",
                "default",
                "--rerank-mode",
                self.rerank_mode,
                "--retrieval-mode",
                self.retrieval_mode,
                "--output",
                str(artifact_path),
            ]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            environment = os.environ.copy()
            environment["TECHJAM_RERANK_MODE"] = self.rerank_mode
            environment["TECHJAM_RETRIEVAL_MODE"] = self.retrieval_mode
            environment["TECHJAM_P11_MODE"] = "off"
            job.log(
                "Running canonical, phrase-perturbed, and product-disjoint stress suites "
                f"(retrieval mode: {self.retrieval_mode}; rerank mode: {self.rerank_mode}; "
                "P11 mode: off for legacy-runner provenance)"
            )
            job.process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=flags,
                env=environment,
            )
            assert job.process.stdout is not None
            for line in job.process.stdout:
                clean_line = line.rstrip()
                job.log(clean_line)
                if "[generalization]" in clean_line and "score=" in clean_line:
                    with job._lock:
                        job.current = min(job.total, job.current + 1)
                if job.cancel_event.is_set() and job.process.poll() is None:
                    job.process.terminate()
            return_code = job.process.wait()
            if job.cancel_event.is_set():
                raise InterruptedError("Generalization evaluation cancelled")
            if return_code != 0:
                raise RuntimeError(
                    f"Generalization evaluator exited with code {return_code}"
                )
            self._require_current_source()
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            corpora = artifact.get("corpora") or {}
            public = corpora.get("released_public") or {}
            derived = corpora.get("derived_product_disjoint") or {}
            public_suites = public.get("suites") or {}
            canonical_metrics = (
                public_suites.get("canonical", {}).get("metrics") or {}
            )
            summary = {
                "artifact": str(artifact_path.relative_to(self.project_root)),
                "rerank_mode": self.rerank_mode,
                "retrieval_mode": self.retrieval_mode,
                "p11_mode": "off",
                "released_public": {
                    "robustness": public.get("robustness"),
                    "canonical": canonical_metrics,
                    "combined_dev": public_suites.get("combined_dev", {}).get("metrics"),
                    "combined_challenge": public_suites.get(
                        "combined_challenge", {}
                    ).get("metrics"),
                },
                "derived_product_disjoint": {
                    "metadata": {
                        key: derived.get(key)
                        for key in (
                            "seed",
                            "sample_count",
                            "samples_sha256",
                            "unique_target_count",
                            "public_target_overlap",
                        )
                    },
                    "robustness": derived.get("robustness"),
                },
            }
            repository = provenance["repository"]
            manifest = {
                "experiment_id": experiment_id,
                "label": (
                    "P1 phrase + product-disjoint robustness · "
                    f"{repository.get('commit') or 'unknown commit'}"
                ),
                "kind": "generalization",
                "created_at": _utc_now(),
                "repository": repository,
                "catalog_sha256": provenance["input_hashes"]["catalog"],
                "dataset_sha256": provenance["input_hashes"]["dataset"],
                "implementation": {
                    "rerank_mode": self.rerank_mode,
                    "retrieval_mode": self.retrieval_mode,
                    "p11_mode": "off",
                    "agent_source_sha256": provenance["source_hashes"]["agent"],
                    "attributes_source_sha256": provenance["source_hashes"][
                        "attributes"
                    ],
                    "reranker_source_sha256": provenance["source_hashes"]["reranker"],
                    "slot_ledger_source_sha256": provenance["source_hashes"][
                        "slot_ledger"
                    ],
                    "clarification_source_sha256": provenance["source_hashes"][
                        "clarification"
                    ],
                    "coverage_source_sha256": provenance["source_hashes"]["coverage"],
                    "shadow_analysis_source_sha256": provenance["source_hashes"][
                        "shadow_analysis"
                    ],
                    "evaluator_source_sha256": provenance["source_hashes"]["evaluator"],
                    "generalization_source_sha256": provenance["source_hashes"][
                        "generalization"
                    ],
                    "attribute_schema_version": ATTRIBUTE_SCHEMA_VERSION,
                    "reranker_scorer_version": SCORER_VERSION,
                    "slot_ledger_schema_version": SLOT_LEDGER_SCHEMA_VERSION,
                    "question_value_schema_version": QUESTION_VALUE_SCHEMA_VERSION,
                    "coverage_schema_version": COVERAGE_SCHEMA_VERSION,
                    "clarification_mode": (
                        "shadow" if self.rerank_mode != "off" else "disabled"
                    ),
                },
                "run": {
                    "python": platform.python_version(),
                    "network_required": False,
                    "functional_elapsed_seconds": round(
                        time.perf_counter() - started, 3
                    ),
                },
                "metrics": canonical_metrics,
                "robustness": {
                    "released_public": public.get("robustness"),
                    "derived_product_disjoint": derived.get("robustness"),
                },
            }
            self._write_json(experiment_dir / "manifest.json", manifest)
            with job._lock:
                job.current = job.total
                job.status = "completed"
                job.summary = summary
            job.log("Generalization evaluation completed")
        except InterruptedError as exc:
            with job._lock:
                job.status = "cancelled"
            job.log(str(exc))
        except Exception as exc:
            with job._lock:
                job.status = "failed"
            job.log(f"{type(exc).__name__}: {exc}")
        finally:
            with job._lock:
                job.finished_at = _utc_now()
                job.elapsed_seconds = round(time.perf_counter() - started, 3)

    def start_tests(self) -> dict[str, Any]:
        job, created = self._new_job("tests")
        if not created:
            return job.snapshot()
        job.thread = threading.Thread(target=self._run_tests, args=(job,), daemon=True)
        job.thread.start()
        return job.snapshot()

    def _run_tests(self, job: JobRecord) -> None:
        started = time.perf_counter()
        try:
            if job.cancel_event.is_set():
                raise InterruptedError("Tests cancelled before startup")
            with job._lock:
                job.status = "running"
                job.started_at = _utc_now()
                job.total = 1
            executable = Path(sys.executable)
            if executable.name.lower() == "pythonw.exe" and executable.with_name("python.exe").exists():
                executable = executable.with_name("python.exe")
            command = [str(executable), "-m", "unittest", "discover", "-s", "tests", "-v"]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            environment = os.environ.copy()
            for name in (
                "TECHJAM_QUESTION_POLICY",
                "TECHJAM_RERANK_MODE",
                "TECHJAM_RETRIEVAL_MODE",
                "TECHJAM_P11_MODE",
                "TECHJAM_P11_SIDECAR_PATH",
            ):
                environment.pop(name, None)
            job.log("Running repository unit tests")
            job.process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=flags,
                env=environment,
            )
            assert job.process.stdout is not None
            for line in job.process.stdout:
                job.log(line.rstrip())
                if job.cancel_event.is_set() and job.process.poll() is None:
                    job.process.terminate()
            return_code = job.process.wait()
            with job._lock:
                job.current = 1
                job.status = "cancelled" if job.cancel_event.is_set() else (
                    "completed" if return_code == 0 else "failed"
                )
                job.summary = {"return_code": return_code}
            job.log(f"Tests finished with exit code {return_code}")
        except InterruptedError as exc:
            with job._lock:
                job.status = "cancelled"
            job.log(str(exc))
        except Exception as exc:
            with job._lock:
                job.status = "failed"
            job.log(f"{type(exc).__name__}: {exc}")
        finally:
            with job._lock:
                job.finished_at = _utc_now()
                job.elapsed_seconds = round(time.perf_counter() - started, 3)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job: {job_id}")
        with job._lock:
            if job.status in {"queued", "running", "cancelling"}:
                job.cancel_event.set()
                job.status = "cancelling"
                job.message = "Cancellation requested"
            elif job.status == "finalizing":
                job.message = "Result finalization has already started and will finish safely"
            process = job.process
        if process is not None and process.poll() is None:
            process.terminate()
        return job.snapshot()

    def lab_reset(self, profile: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_current_source()
        session_id = f"lab_{uuid.uuid4().hex}"
        safe_profile = profile or {
            "purchase_frequency": "not provided",
            "average_prior_rating": None,
            "rating_style": "not provided",
            "preference_tags": [],
            "summary": "Manual local lab session",
        }
        recorder = self.trace_runner.recorder
        if recorder is not None:
            recorder.clear(session_id)
        with self._agent_lock:
            self.trace_runner.agent.reset(session_id, safe_profile)
        with self._lock:
            self._labs[session_id] = {"turn": 0, "profile": safe_profile, "history": []}
            if len(self._labs) > 20:
                evicted_session = next(iter(self._labs))
                self._labs.pop(evicted_session)
                drop_session = getattr(self.trace_runner.agent, "drop_session", None)
                if callable(drop_session):
                    drop_session(evicted_session)
                if recorder is not None:
                    recorder.clear(evicted_session)
        return {
            "session_id": session_id,
            "turn": 0,
            "profile": safe_profile,
            "history": [],
            "rerank_mode": self.rerank_mode,
            "retrieval_mode": self.retrieval_mode,
            "p11_mode": self.p11_mode,
            "p11": _agent_p11_status(self.trace_runner.agent),
        }

    def lab_respond(self, session_id: str, message: str) -> dict[str, Any]:
        self._require_current_source()
        with self._lock:
            lab = self._labs.get(session_id)
        if lab is None:
            raise KeyError("Unknown or expired lab session; reset the lab first")
        message = str(message).strip()
        if not message:
            raise ValueError("message is required")
        turn = int(lab["turn"]) + 1
        if turn > 10:
            raise ValueError("The Agent contract allows at most 10 turns")
        with self._agent_lock:
            response = self.trace_runner.agent.respond(session_id, message, turn, 10)
        ranked = normalize_recommendations(
            response.get("recommendations"), self.trace_runner.catalog_ids
        )
        events = (
            self.trace_runner.recorder.events(session_id, turn)
            if self.trace_runner.recorder is not None
            else []
        )
        entry = {
            "turn": turn,
            "user_message": message,
            "response": response,
            "events": events,
            "recommendations": [
                _product_view(self.trace_runner.products[parent_asin], "")
                for parent_asin in ranked
            ],
        }
        with self._lock:
            lab["turn"] = turn
            lab["history"].append(entry)
        return {
            "session_id": session_id,
            "rerank_mode": self.rerank_mode,
            "retrieval_mode": self.retrieval_mode,
            "p11_mode": self.p11_mode,
            "p11": _agent_p11_status(self.trace_runner.agent),
            **entry,
        }

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
