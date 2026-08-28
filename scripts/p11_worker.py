"""Isolated JSONL worker for the P11 served/control/shadow/active roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.agent import Agent  # noqa: E402
from starter.p9_evidence import (  # noqa: E402
    OFFICIAL_CATALOG_ROWS,
    OFFICIAL_CATALOG_SHA256,
)
from starter.p11_lab import (  # noqa: E402
    ACTIVE_ID,
    CONTROL_ID,
    SHADOW_ID,
    create_p11_agent,
)


BASELINE_ID = "P11.B00.served_agent"
ROLES = {BASELINE_ID, CONTROL_ID, SHADOW_ID, ACTIVE_ID}
MAX_REQUEST_BYTES = 65_536
MAX_RESPONSE_BYTES = 1_048_576
MAX_SIDECAR_BYTES = 33_554_432


class WorkerProtocolError(RuntimeError):
    pass


class NetworkAuditGuard:
    """Deny and count Python-audited network activity in this process."""

    def __init__(self) -> None:
        self.attempt_count = 0

    def hook(self, event: str, _arguments: tuple[object, ...]) -> None:
        if event.startswith("socket."):
            self.attempt_count += 1
            raise PermissionError("network activity is disabled")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reply(value: Mapping[str, object]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    if len(payload) > MAX_RESPONSE_BYTES:
        raise WorkerProtocolError("worker response exceeds byte limit")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_identity(path: Path) -> tuple[int, int, str]:
    """Stream the exact catalog bytes while independently counting JSONL rows."""

    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            byte_count += len(line)
            if line.strip():
                row_count += 1
    return byte_count, row_count, digest.hexdigest()


def _validate_assets(args: argparse.Namespace) -> dict[str, object]:
    """Bind every role to the official catalog before constructing its Agent."""

    if not args.catalog.is_file():
        raise WorkerProtocolError("catalog identity is invalid")
    catalog_bytes, catalog_rows, catalog_sha256 = _catalog_identity(args.catalog)
    if (
        catalog_rows != OFFICIAL_CATALOG_ROWS
        or catalog_sha256 != OFFICIAL_CATALOG_SHA256
    ):
        raise WorkerProtocolError("catalog identity is invalid")

    sidecar_required = args.role in {SHADOW_ID, ACTIVE_ID}
    sidecar_validation: dict[str, object] = {
        "required": sidecar_required,
        "opened_for_identity": False,
        "verified": not sidecar_required,
        "bytes": None,
        "sha256": None,
    }
    if sidecar_required:
        expected_hash = str(args.sidecar_sha256)
        if (
            not isinstance(args.sidecar_bytes, int)
            or isinstance(args.sidecar_bytes, bool)
            or not 0 < args.sidecar_bytes <= MAX_SIDECAR_BYTES
            or len(expected_hash) != 64
            or expected_hash != expected_hash.lower()
            or any(character not in "0123456789abcdef" for character in expected_hash)
            or not args.sidecar.is_file()
        ):
            raise WorkerProtocolError("sidecar identity is invalid")
        actual_bytes = args.sidecar.stat().st_size
        actual_hash = _sha256_file(args.sidecar)
        if actual_bytes != args.sidecar_bytes or actual_hash != expected_hash:
            raise WorkerProtocolError("sidecar identity is invalid")
        sidecar_validation = {
            "required": True,
            "opened_for_identity": True,
            "verified": True,
            "bytes": actual_bytes,
            "sha256": actual_hash,
        }

    return {
        "schema_version": "p11.worker-assets.v1",
        "catalog": {
            "bytes": catalog_bytes,
            "rows": catalog_rows,
            "sha256": catalog_sha256,
            "verified_official": True,
        },
        "sidecar": sidecar_validation,
    }


def _latency_summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values) / 1_000_000.0,
        "p95_ms": ordered[index] / 1_000_000.0,
        "max_ms": ordered[-1] / 1_000_000.0,
    }


def _request(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
        raise WorkerProtocolError("request line is invalid")
    try:
        value = json.loads(line.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError("request JSON is invalid") from exc
    if not isinstance(value, dict):
        raise WorkerProtocolError("request must be an object")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=sorted(ROLES))
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--sidecar-bytes", type=int, required=True)
    parser.add_argument("--sidecar-sha256", required=True)
    return parser.parse_args(argv)


def _build_agent(args: argparse.Namespace) -> Any:
    if args.role == BASELINE_ID:
        return Agent(
            args.catalog,
            llm_client=None,
            question_policy="fast",
            rerank_mode="off",
            retrieval_mode="coverage",
        )
    if args.role == CONTROL_ID:
        return create_p11_agent(args.catalog, args.role)
    return create_p11_agent(
        args.catalog,
        args.role,
        sidecar_path=args.sidecar,
        expected_sidecar=(args.sidecar_bytes, args.sidecar_sha256),
    )


def run(args: argparse.Namespace) -> int:
    if args.role not in ROLES or len(args.nonce) != 32:
        raise WorkerProtocolError("role or nonce is invalid")
    asset_validation = _validate_assets(args)

    network_guard = NetworkAuditGuard()
    sys.addaudithook(network_guard.hook)
    bootstrap_started = time.perf_counter()
    agent = _build_agent(args)
    bootstrap_seconds = time.perf_counter() - bootstrap_started
    response_digest = hashlib.sha256()
    response_count = 0
    generic_exception_count = 0
    generic_exception_classes: list[str] = []
    latencies_ns: list[int] = []
    known_ordinals: set[int] = set()
    _reply({"kind": "ready", "nonce": args.nonce, "role": args.role})

    while True:
        line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            break
        request_id: object = None
        try:
            request = _request(line)
            request_id = request.get("request_id")
            if not isinstance(request_id, int) or isinstance(request_id, bool):
                raise WorkerProtocolError("request_id must be an integer")
            operation = request.get("operation")
            if operation == "reset":
                if set(request) != {"request_id", "operation", "ordinal", "user_profile"}:
                    raise WorkerProtocolError("reset request schema is invalid")
                ordinal = request.get("ordinal")
                profile = request.get("user_profile")
                if (
                    not isinstance(ordinal, int)
                    or isinstance(ordinal, bool)
                    or ordinal <= 0
                    or ordinal in known_ordinals
                    or not isinstance(profile, dict)
                ):
                    raise WorkerProtocolError("reset payload is invalid")
                known_ordinals.add(ordinal)
                agent.reset(f"conversation_{ordinal}", profile)
                _reply({"request_id": request_id, "kind": "reply", "value": None})
                continue
            if operation == "respond":
                if set(request) != {
                    "request_id", "operation", "ordinal", "user_message", "turn", "top_k"
                }:
                    raise WorkerProtocolError("respond request schema is invalid")
                ordinal = request.get("ordinal")
                message = request.get("user_message")
                turn = request.get("turn")
                top_k = request.get("top_k")
                if (
                    ordinal not in known_ordinals
                    or not isinstance(message, str)
                    or not isinstance(turn, int)
                    or isinstance(turn, bool)
                    or not isinstance(top_k, int)
                    or isinstance(top_k, bool)
                ):
                    raise WorkerProtocolError("respond payload is invalid")
                started = time.perf_counter_ns()
                response = agent.respond(
                    f"conversation_{ordinal}", message, turn, top_k
                )
                latencies_ns.append(time.perf_counter_ns() - started)
                response_count += 1
                response_digest.update(
                    _canonical_bytes(
                        {"ordinal": ordinal, "turn": turn, "response": response}
                    )
                    + b"\n"
                )
                _reply(
                    {
                        "request_id": request_id,
                        "kind": "reply",
                        "value": {"response": response},
                    }
                )
                continue
            if operation == "finalize":
                if set(request) != {"request_id", "operation"}:
                    raise WorkerProtocolError("finalize request schema is invalid")
                capture = (
                    {
                        "schema_version": "p11.served-reference.v1",
                        "role": BASELINE_ID,
                        "configuration": {
                            "retrieval_mode": "coverage",
                            "rerank_mode": "off",
                            "question_policy": "fast",
                            "sidecar_opened": False,
                        },
                        "stats": {"turns": response_count, "exception_count": 0},
                        "integrity_errors": [],
                        "hashes": {},
                        "function_hashes": {},
                    }
                    if args.role == BASELINE_ID
                    else agent.export_p11_blind_capture()
                )
                close = getattr(agent, "close", None)
                if callable(close):
                    close()
                bundle = {
                    "schema_version": "p11.worker-bundle.v1",
                    "role": args.role,
                    "asset_validation": asset_validation,
                    "capture": capture,
                    "response_count": response_count,
                    "response_sha256": response_digest.hexdigest(),
                    "generic_exception_count": generic_exception_count,
                    "generic_exception_classes": generic_exception_classes,
                    "network_attempt_count": network_guard.attempt_count,
                    "timing": {
                        "bootstrap_wall_seconds": bootstrap_seconds,
                        "respond_latency": _latency_summary(latencies_ns),
                    },
                    "memory": {
                        "schema_version": "p11.worker-memory-untrusted.v1",
                        "available": False,
                    },
                }
                _reply(
                    {"request_id": request_id, "kind": "result", "bundle": bundle}
                )
                return 0
            raise WorkerProtocolError("operation is invalid")
        except Exception as exc:
            generic_exception_count += 1
            generic_exception_classes.append(type(exc).__name__)
            _reply(
                {
                    "request_id": request_id,
                    "kind": "error",
                    "error_class": type(exc).__name__,
                }
            )
    return 2


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
