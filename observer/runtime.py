from __future__ import annotations

import hashlib
import json
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
from observer.trace import TraceRunner, _product_view
from starter.agent import Agent, _terms


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
    "brief": ("Local challenge brief", "problem-statement.md", "Project copy"),
    "plan": ("Internal implementation plan", "docs/internal_plan.md", "Local only"),
    "architecture": ("Current architecture", "docs/current_architecture.md", "Local only"),
    "source_agent": ("Current Agent source", "starter/agent.py", "Source"),
    "source_evaluator": ("Official evaluator source", "evaluator/local_evaluator.py", "Source"),
    "source_runtime": ("Workbench runtime source", "observer/runtime.py", "Source"),
    "source_trace": ("Session trace source", "observer/trace.py", "Source"),
    "source_server": ("Workbench HTTP source", "observer/server.py", "Source"),
}


class StaleRuntimeError(RuntimeError):
    """Raised when disk source no longer matches the code loaded by the server."""


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
    def __init__(self, agent: Agent, job: JobRecord, total: int) -> None:
        self.agent = agent
        self.job = job
        self.total = total
        self.started = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        if self.job.cancel_event.is_set():
            raise InterruptedError("Evaluation cancelled")
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
        return self.agent.respond(session_id, user_message, turn, top_k)


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
        self.project_id = hashlib.sha256(
            str(self.project_root).casefold().encode("utf-8")
        ).hexdigest()[:16]
        self._source_paths = {
            "agent": self.project_root / "starter" / "agent.py",
            "evaluator": self.project_root / "evaluator" / "local_evaluator.py",
        }
        self._loaded_source_hashes = {
            name: _sha256(path) for name, path in self._source_paths.items()
        }

    @classmethod
    def from_paths(
        cls,
        catalog_path: str | Path = "data/catalog.jsonl",
        dataset_path: str | Path = "data/public_set.jsonl",
        results_path: str | Path = "results.json",
        project_root: str | Path | None = None,
    ) -> WorkbenchRuntime:
        started = time.perf_counter()
        root = Path(project_root or Path.cwd()).resolve()

        def resolve(value: str | Path) -> Path:
            path = Path(value)
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        catalog = resolve(catalog_path)
        dataset = resolve(dataset_path)
        results = resolve(results_path)
        runner = TraceRunner.from_paths(catalog, dataset, results)
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
            "restart_required": source_state["restart_required"],
        }

    def _source_state(self) -> dict[str, Any]:
        items = {}
        restart_required = False
        for name, path in self._source_paths.items():
            loaded = self._loaded_source_hashes.get(name)
            disk = _sha256(path)
            changed = loaded != disk
            restart_required = restart_required or changed
            items[name] = {
                "path": str(path.relative_to(self.project_root)),
                "loaded_sha256": loaded,
                "disk_sha256": disk,
                "changed": changed,
            }
        return {"restart_required": restart_required, "files": items}

    def _require_current_source(self) -> None:
        if self._source_state()["restart_required"]:
            raise StaleRuntimeError(
                "Agent/evaluator source changed after Workbench startup; restart Workbench before running it"
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
            self.trace_runner.agent.connection.close()

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
                "status": "baseline-only",
                "detail": (
                    "reset/respond shapes; official harness supplies turns 1-10 and top_k=10; "
                    "no strict response validator"
                ),
                "source": "starter/agent.py · evaluator/local_evaluator.py",
            },
            {"layer": "Session state", "status": "baseline-only", "detail": "Reset guard only; no accumulated slots or override ledger", "source": "starter/agent.py"},
            {"layer": "Parsing", "status": "implemented", "detail": "Regex tokenization, stopword removal, max 40 unique terms", "source": "starter/agent.py"},
            {"layer": "Sparse retrieval", "status": "implemented", "detail": "Field-weighted SQLite FTS5 BM25", "source": "starter/agent.py"},
            {"layer": "Attribute gate", "status": "not implemented", "detail": "No structured hard filtering or relaxation", "source": None},
            {"layer": "Dense retrieval", "status": "not implemented", "detail": "Roadmap experiment, not an official requirement", "source": None},
            {"layer": "Fusion", "status": "not implemented", "detail": "No RRF or learned fusion", "source": None},
            {"layer": "Reranking", "status": "baseline-only", "detail": "Final order is the BM25 order", "source": "starter/agent.py"},
            {"layer": "Clarification policy", "status": "not implemented", "detail": "ask_attribute is always null", "source": "starter/agent.py"},
            {"layer": "Scoring", "status": "implemented", "detail": "Official deterministic local evaluator", "source": "evaluator/local_evaluator.py"},
        ]

    def list_sessions(self) -> dict[str, Any]:
        with self._lock:
            return self.trace_runner.list_sessions()

    def trace(self, sample_id: str, refresh: bool = False) -> dict[str, Any]:
        if refresh:
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
                if existing.kind == kind and existing.status in {
                    "queued", "running", "cancelling", "finalizing"
                }:
                    return existing, False
            job = JobRecord(job_id=f"{kind}_{uuid.uuid4().hex[:10]}", kind=kind)
            self._jobs[job.job_id] = job
            return job, True

    def start_evaluation(self) -> dict[str, Any]:
        self._require_current_source()
        self._git = self._git_state()
        job, created = self._new_job("evaluation")
        if not created:
            return job.snapshot()
        job.thread = threading.Thread(
            target=self._run_evaluation, args=(job,), daemon=True
        )
        job.thread.start()
        return job.snapshot()

    def _run_evaluation(self, job: JobRecord) -> None:
        started = time.perf_counter()
        agent: Agent | None = None
        try:
            if job.cancel_event.is_set():
                raise InterruptedError("Evaluation cancelled before startup")
            with job._lock:
                job.status = "running"
                job.started_at = _utc_now()
                job.total = len(self.trace_runner.samples)
            job.log("Building a fresh in-memory Agent index")
            agent = Agent(self.catalog_path)
            samples = list(self.trace_runner.samples.values())
            progress_agent = _ProgressAgent(agent, job, len(samples))
            result = evaluate(
                progress_agent,
                samples,
                self.trace_runner.catalog_ids,
                self.trace_runner.categories,
                self.trace_runner.products,
            )
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
            evaluation_seconds = round(time.perf_counter() - started, 3)
            manifest = {
                "experiment_id": experiment_id,
                "label": f"Public evaluator · {self._git.get('commit') or 'unknown commit'}",
                "kind": "local",
                "created_at": _utc_now(),
                "repository": self._git,
                "catalog_sha256": self.catalog_sha256,
                "dataset_sha256": self.dataset_sha256,
                "implementation": {
                    "agent": "starter.agent.Agent",
                    "agent_source_sha256": _sha256(self.project_root / "starter" / "agent.py"),
                    "evaluator_source_sha256": _sha256(
                        self.project_root / "evaluator" / "local_evaluator.py"
                    ),
                    "trace_schema": TRACE_SCHEMA_VERSION,
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
                job.summary = _metrics(result)
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
                agent.connection.close()
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
            job.log("Running repository unit tests")
            job.process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=flags,
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
                if recorder is not None:
                    recorder.clear(evicted_session)
        return {"session_id": session_id, "turn": 0, "profile": safe_profile, "history": []}

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
        return {"session_id": session_id, **entry}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
