"""Isolated, target-blind worker for the P12 action oracle.

The worker never receives a sample identifier, scenario, target, label, or
evaluator stratum.  It receives only an opaque ordinal, the exact safe profile,
visible user messages, turn, and Top-K.  Candidate-bearing action traces remain
private until both Agents and the semantic runtime are closed; the trace is then
published with exclusive creation and the process exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import p12_actions  # noqa: E402
from starter.agent import ALLOWED_ATTRIBUTES, Agent, SessionState  # noqa: E402
from starter.attributes import (  # noqa: E402
    ProductAttributeView,
    build_conversation_constraint_view,
    build_product_attribute_view,
)
from starter.p5_lab import P5Agent, R01 as P5_R01  # noqa: E402
from starter.p11_bridge import (  # noqa: E402
    EXPECTED_SIDECAR_BYTES,
    EXPECTED_SIDECAR_SHA256,
)
from starter.p9_evidence import (  # noqa: E402
    OFFICIAL_CATALOG_ROWS,
    OFFICIAL_CATALOG_SHA256,
)
from starter.semantic import (  # noqa: E402
    OfflineSemanticEncoder,
    SemanticIndex,
    canonical_json_sha256,
    load_semantic_spec,
)


SCHEMA_VERSION = "p12.action-worker.v1"
TRACE_SCHEMA_VERSION = "p12.blind-action-trace.v1"
MAX_REQUEST_BYTES = 65_536
MAX_RESPONSE_BYTES = 1_048_576
MAX_TRACE_BYTES = 512 * 1024 * 1024
MAX_TURNS = 10
TOP_K = 10
MAX_MESSAGE_BYTES = 8_192
MAX_SUMMARY_BYTES = 1_024
MAX_PROFILE_TAGS = 3
ASIN_SHAPE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE)
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

PROFILE_KEYS = {
    "purchase_frequency",
    "average_prior_rating",
    "rating_style",
    "preference_tags",
    "summary",
}
PURCHASE_FREQUENCIES = {
    "not provided",
    "1-2 prior purchases",
    "3-4 prior purchases",
    "5-9 prior purchases",
    "10+ prior purchases",
}
PROFILE_TAGS = {
    "clothing",
    "shoes",
    "jewelry",
    "accessories",
    "sportswear",
    "costumes",
    "bags",
}
REQUEST_SCHEMAS = {
    "reset": {"request_id", "operation", "ordinal", "user_profile"},
    "respond": {
        "request_id",
        "operation",
        "ordinal",
        "user_message",
        "turn",
        "top_k",
    },
    "drop": {"request_id", "operation", "ordinal"},
    "finalize": {"request_id", "operation"},
}
LABEL_SHAPED_KEYS = {
    "ground_truth",
    "label",
    "parent_asin",
    "sample_id",
    "scenario_type",
    "target",
    "target_asin",
    "target_id",
    "evaluation_strata",
    "source_weight",
    "taxonomy",
    "difficulty_bucket",
    "category_bucket",
    "user_id",
    "rating",
    "timestamp",
    "history",
}

EXPECTED_CATALOG_BYTES = 60_546_327
EXPECTED_SPEC_BYTES = 11_976
EXPECTED_SPEC_SHA256 = "c27107edcab9f40f0aa8ba7b003434672e1c6c7fe7714b86f762768d7d3a4614"
EXPECTED_SPEC_CANONICAL_SHA256 = (
    "e71d0cad480c89eac25ad2b276de9a4e7153e1ec2f3bdcc793682f183a592200"
)
EXPECTED_INDEX_MANIFEST_BYTES = 9_474
EXPECTED_INDEX_MANIFEST_SHA256 = (
    "cca932a8b4d0a160e0a409ec6ce9cf3b68c99e3b95bddb911b9c7d83b67365ba"
)
EXPECTED_MATRIX_BYTES = 76_800_128
EXPECTED_MATRIX_SHA256 = (
    "84897381c106b909b9e3d44229187d12f23796f108cfec97904db1cbeeb2d407"
)
EXPECTED_ASINS_BYTES = 550_000
EXPECTED_ASINS_SHA256 = (
    "3af465b23ff2d33614501472edf02d2953ccfc170d2fe3348d55cd51c8ef0d54"
)


class P12WorkerError(RuntimeError):
    """Raised when a target-blind worker invariant is violated."""


class NetworkAuditGuard:
    """Deny network-capable socket events while auditing local metadata calls."""

    def __init__(self) -> None:
        self.attempt_count = 0
        self.local_metadata_count = 0
        self.event_counts: Counter[str] = Counter()

    def hook(self, event: str, _arguments: tuple[object, ...]) -> None:
        if not event.startswith("socket."):
            return
        self.event_counts[event] += 1
        # ONNX Runtime reads the local machine name during offline session
        # construction.  `gethostname` neither resolves a name nor opens or
        # addresses a socket, so report it separately instead of misclassifying
        # it as a network attempt.  Every other socket audit event fails closed.
        if event == "socket.gethostname":
            self.local_metadata_count += 1
            return
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    if path.is_symlink() or not path.is_file():
        raise P12WorkerError(f"required pinned asset is missing or unsafe: {path}")
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _lf_normalized_identity(path: Path) -> dict[str, int | str]:
    """Hash a tracked text asset using its Git-LF byte identity.

    Git may materialize a pinned text blob with CRLF on Windows.  The semantic
    spec lock describes the LF blob, so normalize CRLF and lone CR before
    checking its frozen byte count and digest.  Binary assets remain raw-hashed.
    """

    if path.is_symlink() or not path.is_file():
        raise P12WorkerError(f"required pinned text asset is missing or unsafe: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise P12WorkerError(f"cannot read pinned text asset: {path}") from exc
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return {
        "bytes": len(normalized),
        "sha256": hashlib.sha256(normalized).hexdigest(),
    }


def _catalog_identity(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    row_count = 0
    byte_count = 0
    if path.is_symlink() or not path.is_file():
        raise P12WorkerError("official catalog is missing or unsafe")
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            byte_count += len(line)
            row_count += int(bool(line.strip()))
    return {
        "bytes": byte_count,
        "rows": row_count,
        "sha256": digest.hexdigest(),
    }


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise P12WorkerError("JSON contains a duplicate object key")
        value[key] = nested
    return value


def _strict_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P12WorkerError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise P12WorkerError(f"{label} must be a JSON object")
    return value


def _validate_no_label_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or key.casefold() in LABEL_SHAPED_KEYS:
                raise P12WorkerError("request contains a label-shaped key")
            _validate_no_label_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_no_label_keys(nested)


def _validate_visible_text(value: object, *, label: str, maximum_bytes: int) -> str:
    if not isinstance(value, str):
        raise P12WorkerError(f"{label} must be a string")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise P12WorkerError(f"{label} exceeds its byte limit")
    if CONTROL_CHARACTERS.search(value):
        raise P12WorkerError(f"{label} contains a control character")
    if ASIN_SHAPE.search(value):
        raise P12WorkerError(f"{label} contains an ASIN-shaped identifier")
    return value


def _validate_profile(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != PROFILE_KEYS:
        raise P12WorkerError("user_profile schema is invalid")
    _validate_no_label_keys(value)
    frequency = value.get("purchase_frequency")
    tags = value.get("preference_tags")
    summary = _validate_visible_text(
        value.get("summary"), label="profile summary", maximum_bytes=MAX_SUMMARY_BYTES
    )
    if frequency not in PURCHASE_FREQUENCIES:
        raise P12WorkerError("purchase_frequency is outside the safe enum")
    if value.get("average_prior_rating") is not None:
        raise P12WorkerError("average_prior_rating must remain null")
    if value.get("rating_style") != "unknown":
        raise P12WorkerError("rating_style must remain unknown")
    if (
        not isinstance(tags, list)
        or len(tags) > MAX_PROFILE_TAGS
        or len(tags) != len(set(tags))
        or any(not isinstance(tag, str) or tag not in PROFILE_TAGS for tag in tags)
    ):
        raise P12WorkerError("preference_tags are outside the safe enum")
    return {
        "purchase_frequency": frequency,
        "average_prior_rating": None,
        "rating_style": "unknown",
        "preference_tags": list(tags),
        "summary": summary,
    }


def _parse_request(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
        raise P12WorkerError("request line framing is invalid")
    try:
        value = json.loads(
            line.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P12WorkerError("request JSON is invalid") from exc
    if not isinstance(value, dict):
        raise P12WorkerError("request must be an object")
    operation = value.get("operation")
    if operation not in REQUEST_SCHEMAS or set(value) != REQUEST_SCHEMAS[operation]:
        raise P12WorkerError("request schema is invalid")
    _validate_no_label_keys(value)
    request_id = value.get("request_id")
    if not isinstance(request_id, int) or isinstance(request_id, bool) or request_id <= 0:
        raise P12WorkerError("request_id must be a positive integer")
    if operation in {"reset", "respond", "drop"}:
        ordinal = value.get("ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal <= 0:
            raise P12WorkerError("ordinal must be a positive integer")
    if operation == "reset":
        value["user_profile"] = _validate_profile(value["user_profile"])
    elif operation == "respond":
        value["user_message"] = _validate_visible_text(
            value["user_message"],
            label="user_message",
            maximum_bytes=MAX_MESSAGE_BYTES,
        )
        turn = value.get("turn")
        if not isinstance(turn, int) or isinstance(turn, bool) or not 1 <= turn <= MAX_TURNS:
            raise P12WorkerError("turn must be an integer from 1 to 10")
        if value.get("top_k") != TOP_K or isinstance(value.get("top_k"), bool):
            raise P12WorkerError("top_k must equal 10")
    return value


def _reply(stream: BinaryIO, value: Mapping[str, object]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    if len(payload) > MAX_RESPONSE_BYTES:
        raise P12WorkerError("worker response exceeds its byte limit")
    stream.write(payload)
    stream.flush()


def _ready_response(nonce: str) -> dict[str, object]:
    """Return the exact parent-visible startup response."""

    return {"kind": "ready", "nonce": nonce}


def _success_response(
    operation: str, request_id: int, value: object
) -> dict[str, object]:
    """Return the exact success envelope for one accepted operation."""

    if operation == "finalize":
        if not isinstance(value, Mapping) or set(value) != {
            "trace_sha256",
            "record_count",
            "worker_summary",
        }:
            raise P12WorkerError("finalize receipt payload is invalid")
        return {
            "kind": "receipt",
            "request_id": request_id,
            "trace_sha256": value["trace_sha256"],
            "record_count": value["record_count"],
            "worker_summary": value["worker_summary"],
        }
    if operation in {"reset", "drop"} and value is not None:
        raise P12WorkerError(f"{operation} must return null")
    if operation == "respond":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"ask_attribute"}
            or (
                value.get("ask_attribute") is not None
                and not isinstance(value.get("ask_attribute"), str)
            )
        ):
            raise P12WorkerError("respond value schema is invalid")
        value = {"ask_attribute": value.get("ask_attribute")}
    if operation not in {"reset", "respond", "drop"}:
        raise P12WorkerError("success response operation is invalid")
    return {"kind": "reply", "request_id": request_id, "value": value}


def _validate_identifier_sequence(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise P12WorkerError(f"{label} is not a sequence")
    result = tuple(value)
    if (
        any(not isinstance(item, str) or not item for item in result)
        or len(result) != len(set(result))
    ):
        raise P12WorkerError(f"{label} contains invalid or duplicate identifiers")
    return result


def _validate_served_response(
    response: object, captured_final: object, label: str
) -> str | None:
    """Bind a real Agent response to the final ranking captured in that call."""

    if not isinstance(response, Mapping):
        raise P12WorkerError(f"{label} response is not an object")
    raw_served = response.get("recommendations")
    if not isinstance(raw_served, list):
        raise P12WorkerError(f"{label} served recommendations are not a list")
    served_values: list[str] = []
    for item in raw_served:
        if not isinstance(item, Mapping) or set(item) != {"parent_asin"}:
            raise P12WorkerError(
                f"{label} served recommendation schema is invalid"
            )
        served_values.append(item["parent_asin"])
    served = _validate_identifier_sequence(
        served_values, f"{label} served recommendations"
    )
    captured = _validate_identifier_sequence(captured_final, f"{label} captured final")
    if served != captured[:TOP_K]:
        raise P12WorkerError(f"{label} served recommendations differ from capture")
    ask_attribute = response.get("ask_attribute")
    if ask_attribute is not None and ask_attribute not in ALLOWED_ATTRIBUTES:
        raise P12WorkerError(f"{label} ask_attribute is outside the official enum")
    return ask_attribute


def _validate_p11_status(status: object) -> dict[str, Any]:
    if not isinstance(status, Mapping):
        raise P12WorkerError("P11 status is missing")
    checks = {
        "configured_active": status.get("configured_mode") == "active",
        "effective_active": status.get("effective_mode") == "active",
        "identity_verified": status.get("identity_verified") is True,
        "no_fallback": status.get("fallback") is False,
    }
    if not all(checks.values()):
        raise P12WorkerError(f"P11 active status invariant failed: {checks}")
    return {**checks, "reason_code": str(status.get("reason_code") or "")}


def _validate_p11_invariants(
    r08_full: Sequence[str], p11_full: Sequence[str], diagnostics: object
) -> dict[str, Any]:
    baseline = _validate_identifier_sequence(r08_full, "R08 final")
    served = _validate_identifier_sequence(p11_full, "P11 final")
    if not isinstance(diagnostics, Mapping):
        raise P12WorkerError("P11 per-turn diagnostics are missing")
    status = _validate_p11_status(diagnostics)
    checks = {
        **status,
        "diagnostic_membership_preserved": diagnostics.get(
            "top10_membership_preserved"
        )
        is True,
        "diagnostic_tail_preserved": diagnostics.get("tail_preserved") is True,
        "observed_length_preserved": len(served) == len(baseline),
        "observed_membership_preserved": set(served[:TOP_K])
        == set(baseline[:TOP_K]),
        "observed_tail_preserved": served[TOP_K:] == baseline[TOP_K:],
    }
    if not all(value is True or key == "reason_code" for key, value in checks.items()):
        raise P12WorkerError(f"P11 per-turn invariant failed: {checks}")
    return checks


def _validate_p11_invariant_snapshot(value: object) -> dict[str, Any]:
    """Revalidate the target-free invariant snapshot stored by the capture hook."""

    expected = {
        "configured_active",
        "effective_active",
        "identity_verified",
        "no_fallback",
        "reason_code",
        "diagnostic_membership_preserved",
        "diagnostic_tail_preserved",
        "observed_length_preserved",
        "observed_membership_preserved",
        "observed_tail_preserved",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise P12WorkerError("P11 invariant snapshot schema is invalid")
    snapshot = dict(value)
    if not isinstance(snapshot["reason_code"], str) or not all(
        snapshot[key] is True for key in expected - {"reason_code"}
    ):
        raise P12WorkerError("P11 invariant snapshot contains a failed check")
    return snapshot


def _rank_semantic_c50(
    candidate_ids: Sequence[str], cosine_by_id: Mapping[str, float]
) -> tuple[str, ...]:
    ranker = getattr(p12_actions, "rank_semantic_c50", None)
    if ranker is None:
        ranker = p12_actions.rank_frozen_semantic_c50
    return tuple(ranker(candidate_ids, cosine_by_id))


class CandidateSemanticRuntime:
    """Pinned BGE runtime restricted to matrix rows belonging to one exact C50."""

    def __init__(self, encoder: Any, index: Any) -> None:
        self.encoder = encoder
        self.index = index
        self._row_by_asin = {
            identifier: row for row, identifier in enumerate(index.asins)
        }
        if len(self._row_by_asin) != len(index.asins):
            raise P12WorkerError("semantic ordered-ASIN registry is not unique")
        self.query_count = 0
        self.candidate_matrix_rows_read = 0
        self.maximum_candidate_rows_read = 0
        self.full_catalog_search_calls = 0
        self.failure_count = 0
        self.closed = False

    def rank(self, query_terms: Sequence[str], candidate_ids: Sequence[str]) -> tuple[str, ...]:
        candidates = _validate_identifier_sequence(candidate_ids, "semantic C50")
        if len(candidates) > p12_actions.MAX_CANDIDATES:
            raise P12WorkerError("semantic candidate pool exceeds C50")
        query = " ".join(str(term) for term in query_terms if str(term).strip()).strip()
        if not candidates or not query:
            return candidates
        try:
            row_indexes = [self._row_by_asin[identifier] for identifier in candidates]
            vector = self.encoder.encode_query(query)
            # This is the security-critical operation: advanced-index exactly the
            # candidate rows first, then dot only that <=50-row matrix with the query.
            candidate_matrix = self.index.matrix[row_indexes]
            scores = candidate_matrix @ vector
            values = [float(value) for value in scores]
            if len(values) != len(candidates) or any(
                not math.isfinite(value) for value in values
            ):
                raise P12WorkerError("candidate-only semantic scores are invalid")
            cosine_by_id = dict(zip(candidates, values, strict=True))
            ranked = _rank_semantic_c50(candidates, cosine_by_id)
            if len(ranked) != len(candidates) or set(ranked) != set(candidates):
                raise P12WorkerError("semantic action changed C50 membership")
            self.query_count += 1
            self.candidate_matrix_rows_read += len(row_indexes)
            self.maximum_candidate_rows_read = max(
                self.maximum_candidate_rows_read, len(row_indexes)
            )
            return ranked
        except Exception:
            self.failure_count += 1
            raise

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "candidate_only_c50",
            "query_count": self.query_count,
            "candidate_matrix_rows_read": self.candidate_matrix_rows_read,
            "maximum_candidate_rows_read": self.maximum_candidate_rows_read,
            "full_catalog_search_calls": self.full_catalog_search_calls,
            "failure_count": self.failure_count,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        errors: list[BaseException] = []
        for component in (self.encoder, self.index):
            try:
                component.close()
            except BaseException as exc:  # pragma: no cover - defensive cleanup
                errors.append(exc)
        if errors:
            raise P12WorkerError("semantic runtime close failed") from errors[0]


def _load_catalog_prices(path: Path) -> dict[str, float | None]:
    prices: dict[str, float | None] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            identifier = str(product["parent_asin"])
            if identifier in prices:
                raise P12WorkerError("catalog contains a duplicate parent_asin")
            raw_price = product.get("price")
            try:
                price = float(raw_price) if raw_price not in (None, "") else None
            except (TypeError, ValueError):
                price = None
            prices[identifier] = (
                price if price is not None and math.isfinite(price) and price >= 0 else None
            )
    if len(prices) != OFFICIAL_CATALOG_ROWS:
        raise P12WorkerError("catalog price view row count is invalid")
    return prices


class P12CaptureAgent(Agent):
    """P11-active Agent capturing all action candidates in the same `_apply_p11`."""

    def __init__(
        self,
        catalog_path: Path,
        sidecar_path: Path,
        semantic: CandidateSemanticRuntime,
    ) -> None:
        self._p12_semantic = semantic
        self._p12_prices = _load_catalog_prices(catalog_path)
        self._p12_last_capture: dict[str, Any] | None = None
        self._p12_structured_seconds: list[float] = []
        self._p12_semantic_seconds: list[float] = []
        super().__init__(
            catalog_path,
            llm_client=None,
            question_policy="fast",
            rerank_mode="off",
            retrieval_mode="coverage",
            p11_mode="active",
            p11_sidecar_path=sidecar_path,
        )
        _validate_p11_status(self._p11_status())

    def _p12_product_views(
        self, candidates: Sequence[str], candidate_rowids: Mapping[str, int]
    ) -> dict[str, ProductAttributeView]:
        rowids = [candidate_rowids[value] for value in candidates if value in candidate_rowids]
        if len(rowids) != len(candidates):
            raise P12WorkerError("C50 candidate rowid is missing")
        if not rowids:
            return {}
        placeholders = ",".join("?" for _ in rowids)
        rows = self.connection.execute(
            "SELECT rowid, parent_asin, title, categories, features, details, store, description "
            f"FROM products WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        views = {
            str(row[1]): build_product_attribute_view(
                {
                    "parent_asin": row[1],
                    "title": row[2],
                    "categories": row[3],
                    "features": row[4],
                    "details": row[5],
                    "store": row[6],
                    "description": row[7],
                    "price": self._p12_prices.get(str(row[1])),
                }
            )
            for row in rows
        }
        if set(views) != set(candidates):
            raise P12WorkerError("C50 product view is incomplete")
        return views

    @staticmethod
    def _rank_priors(
        candidates: Sequence[str], rankings: Mapping[str, Sequence[str]]
    ) -> dict[str, float]:
        """Reproduce P2's normalized weighted-RRF prior on the exact C50."""

        if not candidates:
            return {}
        broad = _validate_identifier_sequence(rankings.get("broad"), "broad route")
        strict = _validate_identifier_sequence(rankings.get("strict"), "strict route")
        broad_rank = {
            identifier: rank for rank, identifier in enumerate(broad, start=1)
        }
        strict_rank = {
            identifier: rank for rank, identifier in enumerate(strict, start=1)
        }
        raw = {
            identifier: Agent._fusion_score(identifier, broad_rank, strict_rank)
            for identifier in candidates
        }
        maximum = max(raw.values(), default=0.0)
        if maximum <= 0.0 or any(value <= 0.0 for value in raw.values()):
            raise P12WorkerError("C50 cannot be bound to real weighted-RRF priors")
        return {identifier: value / maximum for identifier, value in raw.items()}

    def _apply_p11(
        self,
        state: SessionState,
        rankings: dict[str, list[str]],
        candidate_rowids: dict[str, int],
        query_terms: list[str],
    ) -> tuple[list[str], dict[str, Any]]:
        r08_full = tuple(rankings["final"])
        p11_final, diagnostics = super()._apply_p11(
            state, rankings, candidate_rowids, query_terms
        )
        p11_full = tuple(p11_final)
        invariant = _validate_p11_invariants(r08_full, p11_full, diagnostics)
        c20 = r08_full[:20]
        c50 = r08_full[:50]
        c100 = r08_full[:100]

        structured_started = time.perf_counter()
        intent = build_conversation_constraint_view(
            state.category_text, state.active_terms, state.excluded_terms
        )
        product_views = self._p12_product_views(c50, candidate_rowids)
        structured = tuple(
            p12_actions.rank_structured_c50(
                c50,
                intent,
                product_views,
                self._rank_priors(c50, rankings),
                tuple(state.messages),
            )
        )
        self._p12_structured_seconds.append(time.perf_counter() - structured_started)
        if len(structured) != len(c50) or set(structured) != set(c50):
            raise P12WorkerError("structured action changed C50 membership")

        semantic_started = time.perf_counter()
        semantic = self._p12_semantic.rank(query_terms, c50)
        self._p12_semantic_seconds.append(time.perf_counter() - semantic_started)
        self._p12_last_capture = {
            "state_identity": id(state),
            "r08_full": r08_full,
            "p11_full": p11_full,
            "candidate_pools": {"c20": c20, "c50": c50, "c100": c100},
            "structured_full": structured,
            "semantic_full": semantic,
            "p11_invariants": invariant,
        }
        return list(p11_full), diagnostics

    def take_last_capture(self, session_id: str) -> dict[str, Any]:
        state = self._sessions.get(session_id)
        capture, self._p12_last_capture = self._p12_last_capture, None
        if (
            capture is None
            or state is None
            or capture.get("state_identity") != id(state)
        ):
            raise P12WorkerError("P11 action capture is missing or misbound")
        return capture

    def p12_timing(self) -> dict[str, list[float]]:
        return {
            "structured": list(self._p12_structured_seconds),
            "semantic": list(self._p12_semantic_seconds),
        }


class P12P5CaptureAgent(P5Agent):
    """Real P5 R01 route capture driven by the same visible messages."""

    def __init__(self, catalog_path: Path) -> None:
        self._p12_last_capture: dict[str, Any] | None = None
        super().__init__(catalog_path, P5_R01, question_policy="fast")

    def _rank_candidates(self, state: SessionState) -> dict[str, list[str]]:
        rankings = super()._rank_candidates(state)
        self._p12_last_capture = {
            "state_identity": id(state),
            "full": tuple(rankings["final"]),
            "variant_id": P5_R01,
            "base": "R08",
        }
        return rankings

    def take_last_capture(self, session_id: str) -> dict[str, Any]:
        state = self._sessions.get(session_id)
        capture, self._p12_last_capture = self._p12_last_capture, None
        if (
            capture is None
            or state is None
            or capture.get("state_identity") != id(state)
        ):
            raise P12WorkerError("P5 result-aware capture is missing or misbound")
        return capture

    def close(self) -> None:
        super().close()


def _latency_summary(seconds: Sequence[float]) -> dict[str, int | float | None]:
    milliseconds = sorted(float(value) * 1000.0 for value in seconds)
    if not milliseconds:
        return {"count": 0, "mean_ms": None, "p95_ms": None, "max_ms": None}
    p95_index = max(0, math.ceil(0.95 * len(milliseconds)) - 1)
    return {
        "count": len(milliseconds),
        "mean_ms": round(statistics.fmean(milliseconds), 6),
        "p95_ms": round(milliseconds[p95_index], 6),
        "max_ms": round(milliseconds[-1], 6),
    }


def _peak_rss_bytes() -> tuple[int | None, str]:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            process = kernel32.GetCurrentProcess()
            success = psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            if success:
                return int(counters.PeakWorkingSetSize), "Windows PeakWorkingSetSize"
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    else:
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return (
                value if sys.platform == "darwin" else value * 1024,
                "resource.getrusage ru_maxrss",
            )
        except (ImportError, OSError, TypeError, ValueError):
            pass
    return None, "unavailable"


def _compose_trace_record(
    ordinal: int,
    turn: int,
    p11_capture: Mapping[str, Any],
    p5_capture: Mapping[str, Any],
) -> dict[str, Any]:
    r08 = _validate_identifier_sequence(p11_capture.get("r08_full"), "R08 capture")
    p11 = _validate_identifier_sequence(p11_capture.get("p11_full"), "P11 capture")
    pools = p11_capture.get("candidate_pools")
    if not isinstance(pools, Mapping) or set(pools) != {"c20", "c50", "c100"}:
        raise P12WorkerError("candidate pool capture schema is invalid")
    c20 = _validate_identifier_sequence(pools["c20"], "C20")
    c50 = _validate_identifier_sequence(pools["c50"], "C50")
    c100 = _validate_identifier_sequence(pools["c100"], "C100")
    if c20 != r08[:20] or c50 != r08[:50] or c100 != r08[:100]:
        raise P12WorkerError("candidate pools are not exact R08 prefixes")
    _validate_p11_invariant_snapshot(p11_capture.get("p11_invariants"))
    if (
        len(p11) != len(r08)
        or set(p11[:TOP_K]) != set(r08[:TOP_K])
        or p11[TOP_K:] != r08[TOP_K:]
    ):
        raise P12WorkerError("P11 capture differs from its verified invariant snapshot")
    structured = _validate_identifier_sequence(
        p11_capture.get("structured_full"), "structured C50"
    )
    semantic = _validate_identifier_sequence(
        p11_capture.get("semantic_full"), "semantic C50"
    )
    if len(structured) != len(c50) or set(structured) != set(c50):
        raise P12WorkerError("structured capture is not a C50 permutation")
    if len(semantic) != len(c50) or set(semantic) != set(c50):
        raise P12WorkerError("semantic capture is not a C50 permutation")
    if (
        not isinstance(p5_capture, Mapping)
        or p5_capture.get("variant_id") != P5_R01
        or p5_capture.get("base") != "R08"
    ):
        raise P12WorkerError("result-aware capture is not real P5 R01 over R08")
    result_aware = _validate_identifier_sequence(
        p5_capture.get("full"), "P5 R01 result-aware final"
    )
    actions = {
        p12_actions.KEEP_R08: list(r08[:TOP_K]),
        p12_actions.KEEP_P11: list(p11[:TOP_K]),
        p12_actions.CANDIDATE_RERANK: list(structured[:TOP_K]),
        p12_actions.FROZEN_SEMANTIC_RERANK: list(semantic[:TOP_K]),
        p12_actions.RESULT_AWARE_REWRITE_RETRIEVE: list(result_aware[:TOP_K]),
        p12_actions.ASK: list(p11[:TOP_K]),
    }
    if set(actions) != set(p12_actions.ACTION_IDS):
        raise P12WorkerError("blind action registry is incomplete")
    return {
        "ordinal": ordinal,
        "turn": turn,
        "actions": actions,
        "candidate_pools": {
            "c20": list(c20),
            "c50": list(c50),
            "c100": list(c100),
        },
    }


def _write_trace_exclusive(path: Path, lines: Sequence[bytes]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"P12 trace already exists: {path}")
    if not path.parent.is_dir():
        raise P12WorkerError("P12 trace parent directory must already exist")
    with path.open("xb") as handle:
        for line in lines:
            handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


class P12ActionRuntime:
    """State machine shared by the production JSONL loop and tiny fake tests."""

    def __init__(
        self,
        p11_agent: Any,
        p5_agent: Any,
        semantic: Any,
        trace_path: Path,
        *,
        network_guard: NetworkAuditGuard | None = None,
        asset_validation: Mapping[str, Any] | None = None,
        initialization_seconds: float = 0.0,
    ) -> None:
        self.p11_agent = p11_agent
        self.p5_agent = p5_agent
        self.semantic = semantic
        self.trace_path = Path(trace_path)
        self.network_guard = network_guard or NetworkAuditGuard()
        self.asset_validation = dict(asset_validation or {})
        self.initialization_seconds = float(initialization_seconds)
        self._next_request_id = 1
        self._next_ordinal = 1
        self._sessions: dict[int, int] = {}
        self._trace_lines: list[bytes] = []
        self._trace_digest = hashlib.sha256()
        self._trace_bytes = 0
        self._completed_sessions = 0
        self._respond_seconds: list[float] = []
        self._p5_seconds: list[float] = []
        self._closed_components: list[str] = []
        self._finalized = False
        self.failure_counts: Counter[str] = Counter()
        self.result_aware_computation_count = 0
        _validate_p11_status(self.p11_agent._p11_status())

    def _accept_request_id(self, request: Mapping[str, Any]) -> None:
        if request["request_id"] != self._next_request_id:
            raise P12WorkerError("request_id is not the next exact integer")
        self._next_request_id += 1

    @staticmethod
    def _session_id(ordinal: int) -> str:
        return f"conversation_{ordinal}"

    def reset(self, ordinal: int, profile: dict[str, Any]) -> None:
        if ordinal != self._next_ordinal or ordinal in self._sessions:
            raise P12WorkerError("ordinal is not the next fresh session")
        session_id = self._session_id(ordinal)
        self.p11_agent.reset(session_id, dict(profile))
        try:
            self.p5_agent.reset(session_id, dict(profile))
        except Exception:
            self.p11_agent.drop_session(session_id)
            raise
        self._sessions[ordinal] = 1
        self._next_ordinal += 1

    def respond(
        self, ordinal: int, user_message: str, turn: int, top_k: int
    ) -> dict[str, str | None]:
        if self._sessions.get(ordinal) != turn:
            raise P12WorkerError("respond turn is not the next exact session turn")
        if top_k != TOP_K:
            raise P12WorkerError("top_k must equal 10")
        session_id = self._session_id(ordinal)
        started = time.perf_counter()
        p11_response = self.p11_agent.respond(session_id, user_message, turn, top_k)
        self._respond_seconds.append(time.perf_counter() - started)
        p11_capture = self.p11_agent.take_last_capture(session_id)
        started = time.perf_counter()
        p5_response = self.p5_agent.respond(session_id, user_message, turn, top_k)
        self._p5_seconds.append(time.perf_counter() - started)
        p5_capture = self.p5_agent.take_last_capture(session_id)
        ask_attribute = _validate_served_response(
            p11_response, p11_capture.get("p11_full"), "P11"
        )
        _validate_served_response(p5_response, p5_capture.get("full"), "P5 R01")
        self.result_aware_computation_count += 1
        record = _compose_trace_record(ordinal, turn, p11_capture, p5_capture)
        line = _canonical_bytes(record) + b"\n"
        if self._trace_bytes + len(line) > MAX_TRACE_BYTES:
            raise P12WorkerError("blind action trace exceeds its byte limit")
        self._trace_lines.append(line)
        self._trace_digest.update(line)
        self._trace_bytes += len(line)
        self._sessions[ordinal] = turn + 1
        return {"ask_attribute": ask_attribute}

    def drop(self, ordinal: int) -> None:
        if self._sessions.get(ordinal) != MAX_TURNS + 1:
            raise P12WorkerError("drop requires exactly ten completed turns")
        session_id = self._session_id(ordinal)
        self.p11_agent.drop_session(session_id)
        self.p5_agent.drop_session(session_id)
        del self._sessions[ordinal]
        self._completed_sessions += 1

    def _close_components(self) -> None:
        errors: list[BaseException] = []
        for name, component in (
            ("p11_agent", self.p11_agent),
            ("p5_agent", self.p5_agent),
            ("semantic", self.semantic),
        ):
            if name in self._closed_components:
                continue
            try:
                component.close()
                self._closed_components.append(name)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise P12WorkerError("P12 component close failed") from errors[0]

    def abort(self) -> None:
        if not self._finalized:
            self._close_components()

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise P12WorkerError("P12 runtime was already finalized")
        if self._sessions:
            raise P12WorkerError("finalize requires every session to be dropped")
        final_started = time.perf_counter()
        # Agent.close() intentionally clears the P11 bridge, so capture the
        # verified active status before closing every candidate-bearing runtime.
        p11_status = _validate_p11_status(self.p11_agent._p11_status())
        p12_timing = (
            self.p11_agent.p12_timing()
            if callable(getattr(self.p11_agent, "p12_timing", None))
            else {"structured": [], "semantic": []}
        )
        self._close_components()
        if self._closed_components != ["p11_agent", "p5_agent", "semantic"]:
            raise P12WorkerError("P12 component close order is invalid")
        semantic_summary = self.semantic.summary()
        if (
            semantic_summary.get("full_catalog_search_calls") != 0
            or int(semantic_summary.get("maximum_candidate_rows_read", 0)) > 50
            or int(semantic_summary.get("failure_count", 0)) != 0
        ):
            raise P12WorkerError("candidate-only semantic invariant failed")
        _write_trace_exclusive(self.trace_path, self._trace_lines)
        self._finalized = True
        peak_rss, rss_backend = _peak_rss_bytes()
        worker_summary = {
            "schema_version": SCHEMA_VERSION,
            "trajectory": {
                "fixed_turns": MAX_TURNS,
                "top_k": TOP_K,
                "completed_sessions": self._completed_sessions,
                "respond_count": len(self._trace_lines),
            },
            "actions": {
                "ids": list(p12_actions.ACTION_IDS),
                "candidate_pool_base": "R08",
                "structured_base": "R08.C50",
                "semantic_base": "R08.C50",
                "result_aware_base": "R08+P5.R01",
                "result_aware_computation_count": self.result_aware_computation_count,
            },
            "p11": {
                "mode": "active",
                "per_turn_invariants_verified": len(self._trace_lines),
                "status": p11_status,
            },
            "semantic": semantic_summary,
            "timing": {
                "initialization_seconds": round(self.initialization_seconds, 6),
                "p11_respond": _latency_summary(self._respond_seconds),
                "p5_respond": _latency_summary(self._p5_seconds),
                "structured": _latency_summary(p12_timing.get("structured", [])),
                "semantic": _latency_summary(p12_timing.get("semantic", [])),
                "finalize_seconds": round(time.perf_counter() - final_started, 6),
            },
            "memory": {"peak_rss_bytes": peak_rss, "backend": rss_backend},
            "failure_counts": dict(sorted(self.failure_counts.items())),
            "semantic_failure_count": int(semantic_summary["failure_count"]),
            "rewrite_failure_count": int(self.failure_counts["rewrite"]),
            "p11_invariant_failure_count": int(
                self.failure_counts["p11_invariant"]
            ),
            "network_attempt_count": self.network_guard.attempt_count,
            "local_socket_metadata_count": self.network_guard.local_metadata_count,
            "socket_audit_event_counts": dict(
                sorted(self.network_guard.event_counts.items())
            ),
            "full_catalog_search_calls": semantic_summary["full_catalog_search_calls"],
            "trace_written_after_components_closed": True,
            "asset_validation": self.asset_validation,
        }
        return {
            "trace_sha256": self._trace_digest.hexdigest(),
            "record_count": len(self._trace_lines),
            "worker_summary": worker_summary,
        }

    def handle(self, request: Mapping[str, Any]) -> Any:
        self._accept_request_id(request)
        operation = request["operation"]
        if operation == "reset":
            self.reset(request["ordinal"], request["user_profile"])
            return None
        if operation == "respond":
            return self.respond(
                request["ordinal"],
                request["user_message"],
                request["turn"],
                request["top_k"],
            )
        if operation == "drop":
            self.drop(request["ordinal"])
            return None
        if operation == "finalize":
            return self.finalize()
        raise P12WorkerError("unknown operation")


def _validate_semantic_lock(
    lock_path: Path,
    spec_path: Path,
    catalog_identity: Mapping[str, Any],
    index_dir: Path,
) -> dict[str, Any]:
    lock = _strict_json_object(lock_path, "semantic lock")
    spec_identity = _lf_normalized_identity(spec_path)
    expected_model = {
        "path": "configs/p7_bge_small_en_v1_5.json",
        "raw_bytes": EXPECTED_SPEC_BYTES,
        "raw_sha256": EXPECTED_SPEC_SHA256,
        "canonical_sha256": EXPECTED_SPEC_CANONICAL_SHA256,
    }
    expected_catalog = {
        "path": "data/catalog.jsonl",
        "bytes": EXPECTED_CATALOG_BYTES,
        "sha256": OFFICIAL_CATALOG_SHA256,
        "rows": OFFICIAL_CATALOG_ROWS,
    }
    index = lock.get("index") if isinstance(lock, Mapping) else None
    if (
        lock.get("schema_version") != "p7.semantic-index-lock.v1"
        or lock.get("model_spec") != expected_model
        or lock.get("catalog") != expected_catalog
        or spec_identity
        != {"bytes": EXPECTED_SPEC_BYTES, "sha256": EXPECTED_SPEC_SHA256}
        or dict(catalog_identity) != {
            "bytes": EXPECTED_CATALOG_BYTES,
            "rows": OFFICIAL_CATALOG_ROWS,
            "sha256": OFFICIAL_CATALOG_SHA256,
        }
        or not isinstance(index, Mapping)
    ):
        raise P12WorkerError("semantic lock does not bind the frozen assets")
    manifest = index.get("manifest")
    matrix = index.get("matrix")
    asins = index.get("ordered_asins")
    if (
        index.get("directory") != "experiments/p7_index"
        or not isinstance(manifest, Mapping)
        or (manifest.get("bytes"), manifest.get("sha256"))
        != (EXPECTED_INDEX_MANIFEST_BYTES, EXPECTED_INDEX_MANIFEST_SHA256)
        or not isinstance(matrix, Mapping)
        or (matrix.get("bytes"), matrix.get("sha256"))
        != (EXPECTED_MATRIX_BYTES, EXPECTED_MATRIX_SHA256)
        or not isinstance(asins, Mapping)
        or (asins.get("bytes"), asins.get("sha256"))
        != (EXPECTED_ASINS_BYTES, EXPECTED_ASINS_SHA256)
    ):
        raise P12WorkerError("semantic index lock entries are not frozen")
    manifest_path = index_dir / str(manifest.get("path"))
    if _file_identity(manifest_path) != {
        "bytes": EXPECTED_INDEX_MANIFEST_BYTES,
        "sha256": EXPECTED_INDEX_MANIFEST_SHA256,
    }:
        raise P12WorkerError("semantic manifest identity differs from the lock")
    return {
        "lock_schema_version": lock["schema_version"],
        "spec_sha256": spec_identity["sha256"],
        "index_manifest_sha256": EXPECTED_INDEX_MANIFEST_SHA256,
        "candidate_only": True,
    }


def _build_runtime(
    args: argparse.Namespace, network_guard: NetworkAuditGuard
) -> P12ActionRuntime:
    started = time.perf_counter()
    catalog_identity = _catalog_identity(args.catalog)
    if (
        args.catalog_rows != OFFICIAL_CATALOG_ROWS
        or args.catalog_sha256 != OFFICIAL_CATALOG_SHA256
        or catalog_identity
        != {
            "bytes": EXPECTED_CATALOG_BYTES,
            "rows": OFFICIAL_CATALOG_ROWS,
            "sha256": OFFICIAL_CATALOG_SHA256,
        }
    ):
        raise P12WorkerError("official catalog identity is invalid")
    sidecar_identity = _file_identity(args.sidecar)
    if (
        args.sidecar_bytes != EXPECTED_SIDECAR_BYTES
        or args.sidecar_sha256 != EXPECTED_SIDECAR_SHA256
        or sidecar_identity
        != {
            "bytes": EXPECTED_SIDECAR_BYTES,
            "sha256": EXPECTED_SIDECAR_SHA256,
        }
    ):
        raise P12WorkerError("P11 sidecar identity is invalid")
    semantic_lock = _validate_semantic_lock(
        args.semantic_lock, args.semantic_spec, catalog_identity, args.semantic_index_dir
    )
    spec = load_semantic_spec(args.semantic_spec)
    if canonical_json_sha256(spec) != EXPECTED_SPEC_CANONICAL_SHA256:
        raise P12WorkerError("semantic spec canonical identity is invalid")
    encoder = OfflineSemanticEncoder.from_frozen_assets(spec, args.semantic_model_dir)
    index: SemanticIndex | None = None
    semantic: CandidateSemanticRuntime | None = None
    p11_agent: P12CaptureAgent | None = None
    p5_agent: P12P5CaptureAgent | None = None
    try:
        index = SemanticIndex.load(
            spec,
            args.semantic_index_dir,
            expected_catalog_sha256=OFFICIAL_CATALOG_SHA256,
        )
        semantic = CandidateSemanticRuntime(encoder, index)
        p11_agent = P12CaptureAgent(args.catalog, args.sidecar, semantic)
        p5_agent = P12P5CaptureAgent(args.catalog)
        # Detect input replacement during the expensive construction phase.
        if _catalog_identity(args.catalog) != catalog_identity:
            raise P12WorkerError("official catalog changed during initialization")
        if _file_identity(args.sidecar) != sidecar_identity:
            raise P12WorkerError("P11 sidecar changed during initialization")
        _validate_p11_status(p11_agent._p11_status())
        return P12ActionRuntime(
            p11_agent,
            p5_agent,
            semantic,
            args.trace_output,
            network_guard=network_guard,
            asset_validation={
                "catalog": catalog_identity,
                "sidecar": sidecar_identity,
                "semantic": semantic_lock,
                "p11_active": True,
            },
            initialization_seconds=time.perf_counter() - started,
        )
    except BaseException:
        for component in (p11_agent, p5_agent, semantic):
            if component is not None:
                try:
                    component.close()
                except BaseException:
                    pass
        if semantic is None:
            if index is not None:
                index.close()
            encoder.close()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-rows", type=int, required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--sidecar-bytes", type=int, required=True)
    parser.add_argument("--sidecar-sha256", required=True)
    parser.add_argument("--semantic-spec", type=Path, required=True)
    parser.add_argument("--semantic-lock", type=Path, required=True)
    parser.add_argument("--semantic-model-dir", type=Path, required=True)
    parser.add_argument("--semantic-index-dir", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    return parser


def run(
    args: argparse.Namespace,
    *,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
    runtime_factory: Any = _build_runtime,
) -> int:
    if not re.fullmatch(r"[0-9a-f]{32}", str(args.nonce)):
        raise P12WorkerError("nonce must be 32 lowercase hexadecimal characters")
    if args.trace_output.exists() or args.trace_output.is_symlink():
        raise FileExistsError(f"P12 trace already exists: {args.trace_output}")
    input_stream = input_stream or sys.stdin.buffer
    output_stream = output_stream or sys.stdout.buffer
    network_guard = NetworkAuditGuard()
    sys.addaudithook(network_guard.hook)
    runtime = runtime_factory(args, network_guard)
    _reply(output_stream, _ready_response(args.nonce))
    finalized = False
    try:
        while True:
            line = input_stream.readline(MAX_REQUEST_BYTES + 1)
            if not line:
                return 2
            request_id: object = None
            try:
                request = _parse_request(line)
                request_id = request["request_id"]
                value = runtime.handle(request)
                if request["operation"] == "finalize":
                    finalized = True
                    _reply(output_stream, _success_response("finalize", request_id, value))
                    return 0
                _reply(
                    output_stream,
                    _success_response(request["operation"], request_id, value),
                )
            except Exception as exc:
                _reply(
                    output_stream,
                    {
                        "kind": "error",
                        "request_id": request_id,
                        "error_class": type(exc).__name__,
                    },
                )
                return 2
    finally:
        if not finalized:
            runtime.abort()


def main(argv: list[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
