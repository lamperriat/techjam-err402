"""Build the one-shot rich, target-blind C100 cache for small-ranker v1.

The executable has three deliberately separate phases:

1. a trusted simulator adapter materializes only participant-visible messages;
2. an isolated subprocess builds and seals numeric features without a proxy path;
3. labels and product-family folds are joined only after the feature SHA-256 exists.

Large outputs live under ``experiments/`` and are ignored by Git.  The tracked
manifest contains aggregate counts and hashes only; no ASIN, sample id, raw
profile, or reversible family mapping is serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
)
from scripts import evaluate_p12_action_oracle as oracle  # noqa: E402
from scripts import train_p12_counterfactual_router as old_cache  # noqa: E402
from starter.agent import Agent, SessionState, _terms  # noqa: E402
from starter.attributes import build_conversation_constraint_view, normalize_value  # noqa: E402
from starter.p11_bridge import _latest_hard_clause_terms  # noqa: E402
from starter.p11_features import (  # noqa: E402
    NEGATIVE_SLOT_ORDER,
    P11FeatureStore,
)
from starter.slot_ledger import DELETED, SUPERSEDED  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/small_ranker_v1.json"
SESSION_COUNT = 2_000
TURN_COUNT = 10
CANDIDATE_COUNT = 100
TRAINING_ROWS = 40
SCHEMA_VERSION = "small-ranker-cache-manifest.v1"
CONTEXT_SCHEMA_VERSION = "small-ranker-visible-context.v1"
ASIN_SHAPE_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE)
PRICE_RE = re.compile(
    r"(?:\b(?:under|below|less than|up to|max(?:imum)?|budget(?: is| of| around)?|price(?: is)?)[^\d$]{0,12}|\$)"
    r"(\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
VARIANT_TOKENS = frozenset(
    {
        "black", "white", "blue", "red", "green", "pink", "purple", "brown",
        "gray", "grey", "yellow", "orange", "beige", "silver", "gold", "navy",
        "small", "medium", "large", "xl", "xxl", "xs", "xxs", "size", "sizes",
        "pack", "pair", "set", "piece", "pieces", "count", "inch", "inches",
        "women", "womens", "woman", "men", "mens", "man", "unisex", "adult",
        "new", "classic", "fashion", "with", "and", "for", "the", "a", "an",
    }
)
CONSTRAINT_SLOTS = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "price",
    "feature",
    "use_case",
)
RANK_ROUTES = (
    "coverage",
    "p11",
    "broad",
    "strict",
    "fused",
    "structured",
    "semantic",
)
ROUTE_CUTOFFS = {
    "coverage": 100,
    "p11": 10,
    "broad": 120,
    "strict": 80,
    "fused": 200,
    "structured": 10,
    "semantic": 10,
}
QUERY_VIEWS = ("turn", "goal", "active")
FIELD_NAMES = ("title_category", "features_details", "description_store")


def _feature_names() -> tuple[str, ...]:
    names: list[str] = ["turn_fraction"]
    for route in RANK_ROUTES:
        names.extend((f"{route}_presence", f"{route}_rank_fraction", f"{route}_reciprocal_rank"))
    names.extend(
        (
            "top10_route_agreement_fraction",
            "route_rank_mean",
            "route_rank_min",
            "route_rank_max",
            "route_rank_dispersion",
            *(f"{route}_incumbent_rr_margin" for route in RANK_ROUTES),
            "previous_presence",
            "previous_rank_fraction",
            "previous_reciprocal_rank",
            "best_historical_rank_fraction",
            "rank_velocity",
            "turn_persistence_fraction",
            "p11_structured_top10_jaccard",
            "p11_semantic_top10_jaccard",
            "broad_strict_top10_jaccard",
            "coverage_fused_top10_jaccard",
            "mean_top10_route_jaccard",
            "top10_vote_entropy",
        )
    )
    for view in QUERY_VIEWS:
        names.extend(f"{view}_{field}_idf_coverage" for field in FIELD_NAMES)
        names.extend(
            (
                f"{view}_token_recall",
                f"{view}_token_precision",
                f"{view}_token_jaccard",
                f"{view}_rare_term_coverage",
                f"{view}_char3_overlap",
                f"{view}_char4_overlap",
                f"{view}_bigram_coverage",
                f"{view}_trigram_coverage",
            )
        )
    for slot in CONSTRAINT_SLOTS:
        names.extend(
            (
                f"{slot}_observed",
                f"{slot}_compatible",
                f"{slot}_unknown",
                f"{slot}_conflict",
            )
        )
    names.extend(
        (
            "hard_clause_coverage",
            "explicit_negative_violation",
            "missing_positive_evidence_fraction",
            "current_turn_override",
            "retired_goal_evidence_conflict",
            "price_missing",
            "active_constraint_count_fraction",
            "hard_constraint_count_fraction",
            "negative_constraint_count_fraction",
            "query_specificity_fraction",
            "goal_age_fraction",
            "goal_version_fraction",
            "override_count_fraction",
            "candidate_bayesian_rating_percentile",
            "candidate_popularity_percentile",
            "title_category_length_log",
            "features_details_length_log",
            "description_store_length_log",
        )
    )
    return tuple(names)


FEATURE_NAMES = _feature_names()
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {"asin", "parent_asin", "target", "target_id", "ground_truth", "sample_id", "user_id", "ordinal"}
)


class SmallRankerCacheError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _walk_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.add(str(key).casefold())
            result.update(_walk_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            result.update(_walk_keys(item))
    return result


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "small-ranker.v1" or value.get("split") != "train_explore":
        raise SmallRankerCacheError("small-ranker config schema/split is invalid")
    if len(value.get("training", {}).get("configs", ())) > 6:
        raise SmallRankerCacheError("more than six preregistered model configs")
    return value


def _resolve_input(config: Mapping[str, Any], name: str) -> Path:
    spec = config["inputs"][name]
    path = (ROOT / str(spec["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(spec.get("bytes", path.stat().st_size)) != path.stat().st_size:
        raise SmallRankerCacheError(f"{name} byte count mismatch")
    if _sha256(path) != str(spec["sha256"]):
        raise SmallRankerCacheError(f"{name} SHA-256 mismatch")
    return path


def _json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SmallRankerCacheError(f"non-object JSONL row in {path}")
                rows.append(value)
    return rows


def _load_catalog(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    products: dict[str, dict[str, Any]] = {}
    categories: dict[str, list[str]] = {}
    for product in _json_lines(path):
        identifier = str(product.get("parent_asin", ""))
        if not identifier or identifier in products:
            raise SmallRankerCacheError("catalog identity is blank or duplicated")
        products[identifier] = product
        categories[identifier] = [str(item) for item in product.get("categories") or ()]
    if len(products) != 50_000:
        raise SmallRankerCacheError("official catalog must contain 50,000 products")
    return products, categories


def _budget_upper(messages: Sequence[str]) -> float | None:
    values = [float(match.group(1)) for message in messages for match in PRICE_RE.finditer(message)]
    values = [value for value in values if math.isfinite(value) and 0.0 <= value <= 1_000_000]
    return values[-1] if values else None


def _safe_record(record: Any) -> dict[str, Any]:
    return {
        "slot": str(record.slot),
        "value": str(record.value),
        "polarity": int(record.polarity),
        "hardness": str(record.hardness),
        "source_turn": int(record.source_turn),
        "version": int(record.version),
        "status": str(record.status),
    }


def materialize_visible_context(
    proxy_path: Path,
    catalog_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Trusted simulator boundary; its output contains visible context only."""

    samples = _json_lines(proxy_path)
    products, categories = _load_catalog(catalog_path)
    if len(samples) != SESSION_COUNT:
        raise SmallRankerCacheError("train_explore proxy row count mismatch")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    parser = object.__new__(Agent)
    parser.question_policy = "fast"
    redacted_messages = 0
    context_digest = hashlib.sha256()
    with output_path.open("xb") as output:
        for sample in samples:
            sample_id = str(sample.get("sample_id", ""))
            label = str(sample.get("ground_truth", {}).get("parent_asin", ""))
            if label not in products or not sample_id:
                raise SmallRankerCacheError("trusted simulator source row is invalid")
            card, behavior = materialize_hidden_fields(dict(sample), products)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = effective["scenario_type"] != "intent_override"
            message = initial_message(effective, coarse_category(categories[label]), disclosed)
            state = SessionState(profile=oracle.project_profile(sample.get("user_profile")))
            turns: list[dict[str, Any]] = []
            for turn in range(1, TURN_COUNT + 1):
                visible, redacted = oracle.sanitize_worker_visible_message(message)
                redacted_messages += int(redacted > 0)
                previous_version = state.version
                parsed = Agent._update_state(parser, state, visible, turn)
                state.slot_ledger.reconcile(
                    build_conversation_constraint_view(
                        state.category_text, state.active_terms, state.excluded_terms
                    ),
                    turn=turn,
                    version=state.version,
                    message=visible,
                    suppressed_slots=state.exhausted_attributes,
                    retired_status=(
                        SUPERSEDED if parsed.is_override or state.version != previous_version else DELETED
                    ),
                )
                active_records = [_safe_record(record) for record in state.slot_ledger.active_records()]
                retired_records = [
                    _safe_record(record)
                    for record in state.slot_ledger.records
                    if record.status != "active"
                ]
                query_terms = Agent._query_terms(parser, state)
                goal_messages = state.messages[max(0, state.version_anchor_turn - 1) :]
                row = {
                    "message": visible,
                    "goal_messages": list(goal_messages),
                    "category_text": state.category_text,
                    "active_terms": list(state.active_terms),
                    "excluded_terms": sorted(state.excluded_terms),
                    "query_terms": list(query_terms),
                    "version": int(state.version),
                    "version_anchor_turn": int(state.version_anchor_turn),
                    "override_count": int(state.override_count),
                    "current_turn_override": bool(parsed.is_override),
                    "active_records": active_records,
                    "retired_records": retired_records,
                    "hard_clause_terms": list(_latest_hard_clause_terms(state)),
                    "budget_upper": _budget_upper(goal_messages),
                }
                if _walk_keys(row) & FORBIDDEN_ARTIFACT_KEYS:
                    raise SmallRankerCacheError("visible context contains a forbidden key")
                serialized = json.dumps(row, sort_keys=True, ensure_ascii=False)
                if label.casefold() in serialized.casefold() or sample_id in serialized or ASIN_SHAPE_RE.search(serialized):
                    raise SmallRankerCacheError("identity token leaked into visible context")
                turns.append(row)
                ask_attribute = Agent._select_question(state, turn)
                if turn < TURN_COUNT:
                    override = effective.get("behavior", {}).get("override") or {}
                    if not override_applied and turn + 1 == int(override.get("turn", 3)):
                        override_applied = True
                        new_value = str(override.get("new_value", ""))
                        if new_value:
                            disclosed.add(new_value)
                        message = str(override.get("message", "Actually, please ignore my earlier preference."))
                    else:
                        message, boundary_used = customer_reply(
                            effective, ask_attribute, disclosed, boundary_used
                        )
            container = {"schema_version": CONTEXT_SCHEMA_VERSION, "turns": turns}
            payload = json.dumps(
                container, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8") + b"\n"
            output.write(payload)
            context_digest.update(payload)
        output.flush()
        os.fsync(output.fileno())
    return {
        "rows": len(samples),
        "turns": len(samples) * TURN_COUNT,
        "bytes": output_path.stat().st_size,
        "sha256": context_digest.hexdigest(),
        "redacted_message_count": redacted_messages,
    }


def _hash_ngrams(values: Sequence[str], size: int, bits: int = 2048) -> int:
    if len(values) < size:
        return 0
    result = 0
    for index in range(len(values) - size + 1):
        gram = "\x1f".join(values[index : index + size]).encode("utf-8")
        position = int.from_bytes(hashlib.blake2s(gram, digest_size=4).digest(), "little") % bits
        result |= 1 << position
    return result


def _hash_char_ngrams(text: str, size: int, bits: int = 2048) -> int:
    compact = f" {normalize_value(text)} "
    if len(compact) < size:
        return 0
    result = 0
    for index in range(len(compact) - size + 1):
        gram = compact[index : index + size].encode("utf-8")
        position = int.from_bytes(hashlib.blake2s(gram, digest_size=4).digest(), "little") % bits
        result |= 1 << position
    return result


def _bit_recall(query_bits: int, candidate_bits: int) -> float:
    denominator = query_bits.bit_count()
    return (query_bits & candidate_bits).bit_count() / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True)
class StaticEvidence:
    field_tokens: tuple[frozenset[str], frozenset[str], frozenset[str]]
    combined_tokens: frozenset[str]
    observed_values: frozenset[str]
    inferred_values: frozenset[str]
    observed_by_slot: Mapping[str, frozenset[str]]
    inferred_by_slot: Mapping[str, frozenset[str]]
    char3_bits: int
    char4_bits: int
    bigram_bits: int
    trigram_bits: int
    bayesian: float
    popularity: float
    price: float | None


class EvidenceRepository:
    """Bounded decoded view over the frozen P11 sidecar plus catalog prices."""

    def __init__(self, sidecar_path: Path, catalog_path: Path) -> None:
        uri = f"{sidecar_path.resolve().as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        columns = connection.execute("PRAGMA table_info(evidence)").fetchall()
        self._column_names = tuple(str(row[1]) for row in columns)
        self._rows = {
            str(row[1]): tuple(row[2:])
            for row in connection.execute("SELECT * FROM evidence")
        }
        catalog_rows = int(connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0])
        self.idf = {
            str(term): math.log((catalog_rows + 1.0) / (int(df) + 1.0)) + 1.0
            for term, df in connection.execute("SELECT term, document_frequency FROM term_stats")
        }
        connection.close()
        self.prices: dict[str, float | None] = {}
        for row in _json_lines(catalog_path):
            identifier = str(row["parent_asin"])
            match = re.search(r"\d+(?:\.\d+)?", str(row.get("price") or ""))
            price = float(match.group()) if match else None
            self.prices[identifier] = price if price is not None and math.isfinite(price) else None
        if len(self._rows) != 50_000 or set(self._rows) != set(self.prices):
            raise SmallRankerCacheError("P11 sidecar/catalog identity binding mismatch")

    @staticmethod
    def _by_slot(values: Iterable[str]) -> dict[str, frozenset[str]]:
        collected: dict[str, set[str]] = {}
        for item in values:
            slot, separator, value = str(item).partition("=")
            if separator and slot and value:
                collected.setdefault(slot, set()).add(value)
        return {slot: frozenset(items) for slot, items in collected.items()}

    @lru_cache(maxsize=16_384)
    def get(self, identifier: str) -> StaticEvidence:
        try:
            row = self._rows[identifier]
        except KeyError as error:
            raise SmallRankerCacheError("candidate is missing from P11 sidecar") from error
        # row begins with feature_blob because catalog_rowid and parent_asin were removed.
        field_sequences, observed, inferred, _observed_subtypes, _inferred_subtypes = (
            P11FeatureStore._decode_feature_blob(row[0])
        )
        field_tokens = tuple(
            frozenset(token for sequence in sequences for token in sequence.split())
            for sequences in field_sequences
        )
        ordered_tokens = tuple(
            token
            for sequences in field_sequences
            for sequence in sequences
            for token in sequence.split()
        )
        combined_tokens = frozenset(ordered_tokens)
        combined_text = " ".join(ordered_tokens)
        mask_count = len(NEGATIVE_SLOT_ORDER)
        bayesian = float(row[1 + mask_count])
        popularity = float(row[2 + mask_count])
        return StaticEvidence(
            field_tokens=(field_tokens[0], field_tokens[1], field_tokens[2]),
            combined_tokens=combined_tokens,
            observed_values=observed,
            inferred_values=inferred,
            observed_by_slot=self._by_slot(observed),
            inferred_by_slot=self._by_slot(inferred),
            char3_bits=_hash_char_ngrams(combined_text, 3),
            char4_bits=_hash_char_ngrams(combined_text, 4),
            bigram_bits=_hash_ngrams(ordered_tokens, 2),
            trigram_bits=_hash_ngrams(ordered_tokens, 3),
            bayesian=bayesian,
            popularity=popularity,
            price=self.prices[identifier],
        )


class SparseIndex:
    """Exact offline reconstruction of Agent broad/strict/fused routes."""

    def __init__(self, catalog_path: Path) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        for row in _json_lines(catalog_path):
            def text(value: object) -> str:
                if value is None:
                    return ""
                if isinstance(value, dict):
                    return " ".join(f"{key} {item}" for key, item in sorted(value.items()))
                if isinstance(value, (list, tuple)):
                    return " ".join(str(item) for item in value)
                return str(value)

            batch.append(
                (
                    str(row["parent_asin"]),
                    text(row.get("title")),
                    text(row.get("categories")),
                    text(row.get("features")),
                    text(row.get("details")),
                    text(row.get("store")),
                    text(row.get("description")),
                )
            )
            if len(batch) == 1_000:
                self.connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            self.connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def rankings(self, query_terms: Sequence[str]) -> dict[str, tuple[str, ...]]:
        broad_expression = Agent._fts_expression(list(query_terms))
        if not broad_expression:
            return {"broad": (), "strict": (), "fused": ()}
        broad = tuple(
            str(row[0])
            for row in self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT 120",
                (broad_expression,),
            )
        )
        strict: tuple[str, ...] = ()
        if len(query_terms) >= 2:
            expression = Agent._strict_fts_expression(list(query_terms))
            if expression:
                strict = tuple(
                    str(row[0])
                    for row in self.connection.execute(
                        "SELECT parent_asin FROM products WHERE products MATCH ? "
                        "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT 80",
                        (expression,),
                    )
                )
        broad_rank = {identifier: rank for rank, identifier in enumerate(broad, 1)}
        strict_rank = {identifier: rank for rank, identifier in enumerate(strict, 1)}
        fused = tuple(
            sorted(
                dict.fromkeys((*broad, *strict)),
                key=lambda identifier: (
                    -Agent._fusion_score(identifier, broad_rank, strict_rank),
                    broad_rank.get(identifier, 10**9),
                    identifier,
                ),
            )
        )
        return {"broad": broad, "strict": strict, "fused": fused}


@dataclass(frozen=True, slots=True)
class QueryView:
    tokens: tuple[str, ...]
    token_set: frozenset[str]
    idf_total: float
    rare_tokens: frozenset[str]
    char3_bits: int
    char4_bits: int
    bigram_bits: int
    trigram_bits: int


def _query_view(text_or_terms: object, idf: Mapping[str, float]) -> QueryView:
    if isinstance(text_or_terms, str):
        tokens = tuple(dict.fromkeys(_terms(text_or_terms)))[:50]
    else:
        tokens = tuple(dict.fromkeys(str(item) for item in text_or_terms if str(item)))[:50]  # type: ignore[union-attr]
    weights = [(token, float(idf.get(token, math.log(50_001.0) + 1.0))) for token in tokens]
    rare_count = max(1, math.ceil(len(weights) / 3)) if weights else 0
    rare = frozenset(token for token, _weight in sorted(weights, key=lambda item: (-item[1], item[0]))[:rare_count])
    normalized_text = " ".join(tokens)
    return QueryView(
        tokens=tokens,
        token_set=frozenset(tokens),
        idf_total=sum(weight for _token, weight in weights),
        rare_tokens=rare,
        char3_bits=_hash_char_ngrams(normalized_text, 3),
        char4_bits=_hash_char_ngrams(normalized_text, 4),
        bigram_bits=_hash_ngrams(tokens, 2),
        trigram_bits=_hash_ngrams(tokens, 3),
    )


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    nonzero = [count / total for count in counts if count > 0]
    if total <= 0 or len(nonzero) <= 1:
        return 0.0
    return -sum(value * math.log(value) for value in nonzero) / math.log(len(nonzero))


def _rank_map(values: Sequence[str]) -> dict[str, int]:
    return {identifier: rank for rank, identifier in enumerate(values, 1)}


def _lexical_values(query: QueryView, evidence: StaticEvidence, idf: Mapping[str, float]) -> list[float]:
    values: list[float] = []
    for field_tokens in evidence.field_tokens:
        numerator = sum(float(idf.get(token, math.log(50_001.0) + 1.0)) for token in query.token_set & field_tokens)
        values.append(numerator / query.idf_total if query.idf_total else 0.0)
    intersection = query.token_set & evidence.combined_tokens
    union = query.token_set | evidence.combined_tokens
    values.extend(
        (
            len(intersection) / len(query.token_set) if query.token_set else 0.0,
            len(intersection) / len(evidence.combined_tokens) if evidence.combined_tokens else 0.0,
            len(intersection) / len(union) if union else 0.0,
            len(query.rare_tokens & evidence.combined_tokens) / len(query.rare_tokens) if query.rare_tokens else 0.0,
            _bit_recall(query.char3_bits, evidence.char3_bits),
            _bit_recall(query.char4_bits, evidence.char4_bits),
            _bit_recall(query.bigram_bits, evidence.bigram_bits),
            _bit_recall(query.trigram_bits, evidence.trigram_bits),
        )
    )
    return values


def _constraint_values(context: Mapping[str, Any], evidence: StaticEvidence) -> list[float]:
    active = [item for item in context.get("active_records", ()) if isinstance(item, Mapping)]
    retired = [item for item in context.get("retired_records", ()) if isinstance(item, Mapping)]
    positives: dict[str, list[Mapping[str, Any]]] = {}
    negatives: dict[str, list[Mapping[str, Any]]] = {}
    for record in active:
        slot = str(record.get("slot", ""))
        (positives if int(record.get("polarity", 0)) > 0 else negatives).setdefault(slot, []).append(record)

    values: list[float] = []
    positive_total = 0
    positive_missing = 0
    explicit_negative_violation = False
    for slot in CONSTRAINT_SLOTS:
        if slot == "price":
            budget = context.get("budget_upper")
            query_present = isinstance(budget, (int, float)) and not isinstance(budget, bool)
            observed = float(query_present and evidence.price is not None)
            compatible = float(query_present and evidence.price is not None and float(evidence.price) <= float(budget))
            unknown = float(query_present and evidence.price is None)
            conflict = float(query_present and evidence.price is not None and float(evidence.price) > float(budget))
            if query_present:
                positive_total += 1
                positive_missing += int(bool(unknown))
            values.extend((observed, compatible, unknown, conflict))
            continue

        requested = positives.get(slot, ())
        rejected = negatives.get(slot, ())
        observed_values = evidence.observed_by_slot.get(slot, frozenset())
        inferred_values = evidence.inferred_by_slot.get(slot, frozenset())
        requested_values = {normalize_value(item.get("value")) for item in requested if normalize_value(item.get("value"))}
        rejected_values = {normalize_value(item.get("value")) for item in rejected if normalize_value(item.get("value"))}
        # Exact free-form constraints are also matched against field tokens/text evidence.
        observed_match = bool(requested_values & observed_values)
        inferred_match = bool(requested_values & inferred_values)
        if slot == "feature" and requested_values:
            observed_match = observed_match or any(
                set(value.split()) <= evidence.combined_tokens for value in requested_values
            )
        negative_hit = bool(rejected_values & (observed_values | inferred_values))
        if slot == "feature" and rejected_values:
            negative_hit = negative_hit or any(
                set(value.split()) <= evidence.combined_tokens for value in rejected_values
            )
        explicit_negative_violation = explicit_negative_violation or negative_hit
        query_present = bool(requested_values or rejected_values)
        candidate_has_slot = bool(observed_values or inferred_values)
        positive_mismatch = bool(requested_values and candidate_has_slot and not (observed_match or inferred_match))
        unknown = bool(requested_values and not candidate_has_slot and not observed_match and not inferred_match)
        values.extend(
            (
                float(observed_match),
                float(observed_match or inferred_match),
                float(query_present and unknown),
                float(negative_hit or positive_mismatch),
            )
        )
        positive_total += len(requested_values)
        positive_missing += sum(
            int(
                value not in observed_values
                and value not in inferred_values
                and not (slot == "feature" and set(value.split()) <= evidence.combined_tokens)
            )
            for value in requested_values
        )

    hard_records = [item for item in active if item.get("hardness") == "hard" and int(item.get("polarity", 0)) > 0]
    hard_hits = 0
    for record in hard_records:
        slot = str(record.get("slot", ""))
        value = normalize_value(record.get("value"))
        hard_hits += int(
            f"{slot}={value}" in evidence.observed_values
            or f"{slot}={value}" in evidence.inferred_values
            or (slot == "feature" and set(value.split()) <= evidence.combined_tokens)
        )
    hard_terms = frozenset(str(item) for item in context.get("hard_clause_terms", ()) if str(item))
    lexical_hard_coverage = (
        len(hard_terms & evidence.combined_tokens) / len(hard_terms) if hard_terms else 0.0
    )
    hard_constraint_coverage = (
        hard_hits / len(hard_records) if hard_records else lexical_hard_coverage
    )
    retired_positive = [item for item in retired if int(item.get("polarity", 0)) > 0]
    retired_match = any(
        f"{record.get('slot')}={normalize_value(record.get('value'))}" in (evidence.observed_values | evidence.inferred_values)
        for record in retired_positive
    )
    active_match = any(
        f"{record.get('slot')}={normalize_value(record.get('value'))}" in (evidence.observed_values | evidence.inferred_values)
        for record in active
        if int(record.get("polarity", 0)) > 0
    )
    active_count = len(active)
    hard_count = sum(int(item.get("hardness") == "hard") for item in active)
    negative_count = sum(int(int(item.get("polarity", 0)) < 0) for item in active)
    query_terms = tuple(str(item) for item in context.get("query_terms", ()))
    turn = int(context.get("turn", 1))
    version_anchor = int(context.get("version_anchor_turn", 1))
    values.extend(
        (
            hard_constraint_coverage,
            float(explicit_negative_violation),
            positive_missing / positive_total if positive_total else 0.0,
            float(bool(context.get("current_turn_override"))),
            float(bool(retired_match and not active_match)),
            float(evidence.price is None),
            min(active_count, 20) / 20.0,
            min(hard_count, 10) / 10.0,
            min(negative_count, 10) / 10.0,
            min(len(query_terms), 50) / 50.0,
            min(max(1, turn - version_anchor + 1), 10) / 10.0,
            min(int(context.get("version", 1)), 10) / 10.0,
            min(int(context.get("override_count", 0)), 5) / 5.0,
            evidence.bayesian,
            evidence.popularity,
            math.log1p(len(evidence.field_tokens[0])) / math.log(501.0),
            math.log1p(len(evidence.field_tokens[1])) / math.log(501.0),
            math.log1p(len(evidence.field_tokens[2])) / math.log(501.0),
        )
    )
    return values


def build_feature_row(
    *,
    identifier: str,
    turn: int,
    route_maps: Mapping[str, Mapping[str, int]],
    route_top10: Mapping[str, Sequence[str]],
    incumbent: str,
    previous_ranks: Sequence[Mapping[str, int]],
    query_views: Sequence[QueryView],
    context: Mapping[str, Any],
    evidence: StaticEvidence,
    idf: Mapping[str, float],
    group_top10_jaccards: Sequence[float],
    vote_entropy: float,
) -> np.ndarray:
    values: list[float] = [turn / TURN_COUNT]
    normalized_ranks: list[float] = []
    top10_votes = 0
    for route in RANK_ROUTES:
        rank = route_maps[route].get(identifier)
        cutoff = ROUTE_CUTOFFS[route]
        values.extend(
            (
                float(rank is not None),
                rank / cutoff if rank is not None else 1.25,
                1.0 / rank if rank is not None else 0.0,
            )
        )
        if rank is not None:
            normalized_ranks.append(rank / cutoff)
        top10_votes += int(identifier in route_top10[route])
    values.extend(
        (
            top10_votes / len(RANK_ROUTES),
            sum(normalized_ranks) / len(normalized_ranks) if normalized_ranks else 1.25,
            min(normalized_ranks) if normalized_ranks else 1.25,
            max(normalized_ranks) if normalized_ranks else 1.25,
            float(np.std(np.asarray(normalized_ranks, dtype=np.float64))) if normalized_ranks else 0.0,
        )
    )
    for route in RANK_ROUTES:
        candidate_rank = route_maps[route].get(identifier)
        incumbent_rank = route_maps[route].get(incumbent)
        values.append(
            (1.0 / candidate_rank if candidate_rank else 0.0)
            - (1.0 / incumbent_rank if incumbent_rank else 0.0)
        )
    historical = [mapping[identifier] for mapping in previous_ranks if identifier in mapping]
    previous = historical[-1] if historical else None
    current = route_maps["coverage"].get(identifier)
    values.extend(
        (
            float(previous is not None),
            previous / CANDIDATE_COUNT if previous is not None else 1.25,
            1.0 / previous if previous is not None else 0.0,
            min(historical) / CANDIDATE_COUNT if historical else 1.25,
            (previous - current) / CANDIDATE_COUNT if previous is not None and current is not None else 0.0,
            len(historical) / max(1, turn - 1),
            *group_top10_jaccards,
            vote_entropy,
        )
    )
    for view in query_views:
        values.extend(_lexical_values(view, evidence, idf))
    mutable_context = dict(context)
    mutable_context["turn"] = turn
    values.extend(_constraint_values(mutable_context, evidence))
    if len(values) != len(FEATURE_NAMES):
        raise SmallRankerCacheError(
            f"feature row width mismatch: {len(values)} != {len(FEATURE_NAMES)}"
        )
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise SmallRankerCacheError("non-finite rich feature row")
    return array


def _load_context(path: Path) -> list[list[dict[str, Any]]]:
    sessions: list[list[dict[str, Any]]] = []
    for container in _json_lines(path):
        if container.get("schema_version") != CONTEXT_SCHEMA_VERSION:
            raise SmallRankerCacheError("visible-context schema mismatch")
        turns = container.get("turns")
        if not isinstance(turns, list) or len(turns) != TURN_COUNT:
            raise SmallRankerCacheError("visible-context turn count mismatch")
        for turn in turns:
            if not isinstance(turn, dict) or _walk_keys(turn) & FORBIDDEN_ARTIFACT_KEYS:
                raise SmallRankerCacheError("visible-context row failed privacy validation")
            serialized = json.dumps(turn, sort_keys=True, ensure_ascii=False)
            if ASIN_SHAPE_RE.search(serialized):
                raise SmallRankerCacheError("visible-context contains an identifier-shaped token")
        sessions.append(turns)
    if len(sessions) != SESSION_COUNT:
        raise SmallRankerCacheError("visible-context session count mismatch")
    return sessions


def _top10_diagnostics(route_top10: Mapping[str, Sequence[str]], candidates: Sequence[str]) -> tuple[list[float], float]:
    pairs = (
        ("p11", "structured"),
        ("p11", "semantic"),
        ("broad", "strict"),
        ("coverage", "fused"),
    )
    fixed = [_jaccard(route_top10[left], route_top10[right]) for left, right in pairs]
    all_pairs = [
        _jaccard(route_top10[left], route_top10[right])
        for left_index, left in enumerate(RANK_ROUTES)
        for right in RANK_ROUTES[left_index + 1 :]
    ]
    fixed.append(sum(all_pairs) / len(all_pairs) if all_pairs else 0.0)
    counts = [sum(int(identifier in route_top10[route]) for route in RANK_ROUTES) for identifier in candidates]
    return fixed, _entropy(counts)


def build_target_blind_features(
    context_path: Path,
    catalog_path: Path,
    sidecar_path: Path,
    output_path: Path,
    phase_manifest_path: Path,
) -> dict[str, Any]:
    """Feature-only phase.  Its signature intentionally has no proxy/label input."""

    started = time.perf_counter()
    if output_path.exists() or output_path.is_symlink() or phase_manifest_path.exists() or phase_manifest_path.is_symlink():
        raise FileExistsError("feature phase output already exists")
    contexts = _load_context(context_path)
    old_cache._validate_aggregate()
    trace, trace_identifiers = old_cache._load_traces()
    if len(trace) != SESSION_COUNT:
        raise SmallRankerCacheError("blind trace session count mismatch")
    evidence_repository = EvidenceRepository(sidecar_path, catalog_path)
    sparse = SparseIndex(catalog_path)
    index_seconds = time.perf_counter() - started
    features = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT, len(FEATURE_NAMES)),
    )
    route_query_seconds = 0.0
    row_feature_seconds = 0.0
    for session_index, (turn_contexts, turns) in enumerate(zip(contexts, trace, strict=True)):
        previous_ranks: list[dict[str, int]] = []
        for turn_index, (context, trace_turn) in enumerate(zip(turn_contexts, turns, strict=True)):
            turn = turn_index + 1
            candidates = tuple(str(item) for item in trace_turn["c100"])
            if len(candidates) != CANDIDATE_COUNT or len(set(candidates)) != CANDIDATE_COUNT:
                raise SmallRankerCacheError("blind C100 pool is incomplete or duplicated")
            p11 = tuple(str(item) for item in trace_turn["actions"]["KEEP_P11"])
            structured = tuple(str(item) for item in trace_turn["actions"]["CANDIDATE_RERANK"])
            semantic = tuple(str(item) for item in trace_turn["actions"]["FROZEN_SEMANTIC_RERANK"])
            if len(p11) != 10 or set(p11) != set(candidates[:10]):
                raise SmallRankerCacheError("P11 trace violates its Top10 membership invariant")
            tick = time.perf_counter()
            sparse_routes = sparse.rankings(tuple(str(item) for item in context["query_terms"]))
            route_query_seconds += time.perf_counter() - tick
            if not set(candidates) <= set(sparse_routes["fused"]):
                raise SmallRankerCacheError("offline broad/strict reconstruction does not cover C100")
            route_lists: dict[str, tuple[str, ...]] = {
                "coverage": candidates,
                "p11": p11,
                "broad": sparse_routes["broad"],
                "strict": sparse_routes["strict"],
                "fused": sparse_routes["fused"],
                "structured": structured,
                "semantic": semantic,
            }
            route_maps = {name: _rank_map(values) for name, values in route_lists.items()}
            route_top10 = {name: values[:10] for name, values in route_lists.items()}
            group_jaccards, vote_entropy = _top10_diagnostics(route_top10, candidates)
            query_views = (
                _query_view(str(context["message"]), evidence_repository.idf),
                _query_view(" ".join(str(item) for item in context["goal_messages"]), evidence_repository.idf),
                _query_view(context["query_terms"], evidence_repository.idf),
            )
            incumbent = p11[9]
            tick = time.perf_counter()
            for candidate_index, identifier in enumerate(candidates):
                features[session_index, turn_index, candidate_index] = build_feature_row(
                    identifier=identifier,
                    turn=turn,
                    route_maps=route_maps,
                    route_top10=route_top10,
                    incumbent=incumbent,
                    previous_ranks=previous_ranks,
                    query_views=query_views,
                    context=context,
                    evidence=evidence_repository.get(identifier),
                    idf=evidence_repository.idf,
                    group_top10_jaccards=group_jaccards,
                    vote_entropy=vote_entropy,
                )
            row_feature_seconds += time.perf_counter() - tick
            previous_ranks.append(route_maps["coverage"])
        if (session_index + 1) % 100 == 0:
            features.flush()
            print(
                json.dumps({"feature_phase_sessions": session_index + 1, "elapsed_seconds": round(time.perf_counter() - started, 3)}),
                flush=True,
            )
    features.flush()
    del features
    feature_sha256 = _sha256(output_path)
    metadata: dict[str, Any] = {
        "schema_version": "small-ranker-feature-phase.v1",
        "label_source_opened": False,
        "feature_shape": [SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT, len(FEATURE_NAMES)],
        "feature_dtype": "float32",
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_sha256": _canonical_sha256(list(FEATURE_NAMES)),
        "feature_file": {
            "bytes": output_path.stat().st_size,
            "sha256": feature_sha256,
        },
        "blind_trace_identifier_count": len(trace_identifiers),
        "timing_seconds": {
            "index_and_inputs": round(index_seconds, 6),
            "route_queries": round(route_query_seconds, 6),
            "feature_rows": round(row_feature_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
    }
    if _walk_keys(metadata) & FORBIDDEN_ARTIFACT_KEYS:
        raise SmallRankerCacheError("feature phase metadata contains a forbidden key")
    phase_manifest_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _family_signature(product: Mapping[str, Any], title_token_limit: int = 14) -> str:
    categories = [normalize_value(item) for item in product.get("categories") or () if normalize_value(item)]
    leaf = categories[-1] if categories else "unknown"
    store = normalize_value(product.get("store")) or "unknown"
    title_tokens = [
        token
        for token in TOKEN_RE.findall(normalize_value(product.get("title")))
        if token not in VARIANT_TOKENS and not token.isdigit() and not re.fullmatch(r"\d+(?:oz|mm|cm|in|xl)?", token)
    ]
    signature = " ".join(title_tokens[:title_token_limit]) or leaf
    return f"{leaf}\x1f{store}\x1f{signature}"


def _family_folds(
    signatures: Sequence[str],
    baseline_hit: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mapping: dict[str, int] = {}
    family_index = np.empty(len(signatures), dtype=np.int32)
    for index, signature in enumerate(signatures):
        if signature not in mapping:
            mapping[signature] = len(mapping)
        family_index[index] = mapping[signature]
    by_family: dict[int, list[int]] = {}
    for session, family in enumerate(family_index.tolist()):
        by_family.setdefault(int(family), []).append(session)
    signature_by_family = {mapping[signature]: signature for signature in mapping}
    ordered = sorted(
        by_family,
        key=lambda family: (
            -len(by_family[family]),
            -sum(int(not baseline_hit[session]) for session in by_family[family]),
            hashlib.sha256(f"{seed}:outer:{signature_by_family[family]}".encode()).hexdigest(),
        ),
    )
    fold_sessions = [0] * folds
    fold_misses = [0] * folds
    family_outer: dict[int, int] = {}
    for family in ordered:
        size = len(by_family[family])
        misses = sum(int(not baseline_hit[session]) for session in by_family[family])
        # Session balance is primary.  Miss balance is only a tie-breaker;
        # making it primary can strand every zero-miss family in one fold.
        fold = min(range(folds), key=lambda value: (fold_sessions[value], fold_misses[value], value))
        family_outer[family] = fold
        fold_sessions[fold] += size
        fold_misses[fold] += misses
    outer = np.asarray([family_outer[int(family)] for family in family_index], dtype=np.uint8)
    inner = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}:inner:{signature}".encode()).digest()[:4], "little"
            )
            % folds
            for signature in signatures
        ],
        dtype=np.uint8,
    )
    for family in np.unique(family_index):
        mask = family_index == family
        if len(np.unique(outer[mask])) != 1 or len(np.unique(inner[mask])) != 1:
            raise SmallRankerCacheError("one product family crosses a fold boundary")
    return family_index, outer, inner, len(mapping)


def _deterministic_training_indices(
    group_index: int,
    positive_index: int,
    turn: Mapping[str, Any],
    feature_group: np.ndarray,
) -> tuple[np.ndarray, int]:
    if not 0 <= positive_index < CANDIDATE_COUNT:
        return np.full(TRAINING_ROWS, -1, dtype=np.int16), 0
    candidates = tuple(str(item) for item in turn["c100"])
    positions = {identifier: index for index, identifier in enumerate(candidates)}
    selected: list[int] = list(range(20))
    selected.append(positive_index)
    for action in ("KEEP_P11", "CANDIDATE_RERANK", "FROZEN_SEMANTIC_RERANK"):
        selected.extend(positions[item] for item in turn["actions"][action] if item in positions)
    for route in ("broad", "strict"):
        presence = feature_group[:, FEATURE_INDEX[f"{route}_presence"]] > 0.5
        ranks = feature_group[:, FEATURE_INDEX[f"{route}_rank_fraction"]]
        route_indices = np.flatnonzero(presence)
        selected.extend(route_indices[np.argsort(ranks[route_indices], kind="stable")[:10]].tolist())
    selected = list(dict.fromkeys(selected))
    cursor = 0
    while len(selected) < TRAINING_ROWS and cursor < 320:
        candidate = 20 + ((group_index * 37 + cursor * 29 + 11) % 80)
        if candidate not in selected:
            selected.append(candidate)
        cursor += 1
    selected = selected[:TRAINING_ROWS]
    output = np.full(TRAINING_ROWS, -1, dtype=np.int16)
    output[: len(selected)] = np.asarray(selected, dtype=np.int16)
    if positive_index not in output or not set(range(10)) <= set(output.tolist()):
        raise SmallRankerCacheError("hard-negative selection omitted target or baseline Top10")
    return output, len(selected)


def _numeric_audit(arrays: Mapping[str, np.ndarray]) -> None:
    if not arrays:
        raise SmallRankerCacheError("label array registry is empty")
    for name, array in arrays.items():
        if not name or not isinstance(array, np.ndarray) or array.dtype.kind not in "biuf":
            raise SmallRankerCacheError(f"non-numeric label array: {name}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise SmallRankerCacheError(f"non-finite label array: {name}")


def _identity_shape_scan(path: Path) -> int:
    pattern = re.compile(rb"B0[A-Z0-9]{8}", re.IGNORECASE)
    matches = 0
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            payload = overlap + chunk
            matches += len(pattern.findall(payload))
            overlap = payload[-9:]
    return matches


def _array_registry(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {"dtype": array.dtype.str, "shape": list(array.shape)}
        for name, array in sorted(arrays.items())
    }


def join_labels_after_feature_seal(
    *,
    proxy_path: Path,
    catalog_path: Path,
    feature_path: Path,
    label_path: Path,
    seed: int,
    family_title_token_limit: int,
) -> tuple[dict[str, Any], set[str]]:
    """Join labels only after the immutable feature digest is available."""

    if not feature_path.is_file():
        raise SmallRankerCacheError("feature cache must be sealed before label join")
    sealed_feature_sha256 = _sha256(feature_path)
    samples = _json_lines(proxy_path)
    old_cache._validate_aggregate()
    trace, _trace_identifiers = old_cache._load_traces()
    labels, eligible_values = old_cache._load_proxy()
    if len(samples) != SESSION_COUNT or list(labels) != [str(row["ground_truth"]["parent_asin"]) for row in samples]:
        raise SmallRankerCacheError("proxy label order is inconsistent")
    products, _categories = _load_catalog(catalog_path)
    feature_cache = np.load(feature_path, mmap_mode="r")
    expected_shape = (SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT, len(FEATURE_NAMES))
    if feature_cache.shape != expected_shape or feature_cache.dtype != np.dtype("float32"):
        raise SmallRankerCacheError("sealed feature cache shape/dtype mismatch")

    positive_index = np.full((SESSION_COUNT, TURN_COUNT), -1, dtype=np.int16)
    baseline_rank = np.zeros((SESSION_COUNT, TURN_COUNT), dtype=np.uint8)
    training_indices = np.full((SESSION_COUNT, TURN_COUNT, TRAINING_ROWS), -1, dtype=np.int16)
    training_length = np.zeros((SESSION_COUNT, TURN_COUNT), dtype=np.uint8)
    for session_index, turns in enumerate(trace):
        label = labels[session_index]
        for turn_index, turn in enumerate(turns):
            candidates = tuple(str(item) for item in turn["c100"])
            try:
                positive = candidates.index(label)
                positive_index[session_index, turn_index] = positive
            except ValueError:
                positive = -1
            p11 = tuple(str(item) for item in turn["actions"]["KEEP_P11"])
            try:
                baseline_rank[session_index, turn_index] = p11.index(label) + 1
            except ValueError:
                pass
            group_index = session_index * TURN_COUNT + turn_index
            indices, length = _deterministic_training_indices(
                group_index,
                positive,
                turn,
                feature_cache[session_index, turn_index],
            )
            training_indices[session_index, turn_index] = indices
            training_length[session_index, turn_index] = length

    eligible_from = np.asarray(eligible_values, dtype=np.uint8)
    baseline_hit = np.asarray(
        [
            int(np.any(baseline_rank[session, eligible_from[session] - 1 :] > 0))
            for session in range(SESSION_COUNT)
        ],
        dtype=np.uint8,
    )
    signatures = [_family_signature(products[label], family_title_token_limit) for label in labels]
    family_index, outer_fold, inner_fold, family_count = _family_folds(
        signatures, baseline_hit, 5, seed
    )
    taxonomy_labels = [str(row.get("taxonomy", {}).get("group", "unknown")) for row in samples]
    popularity_labels = [str(row.get("evaluation_strata", {}).get("popularity", "unknown")) for row in samples]
    taxonomy_values = {value: index for index, value in enumerate(sorted(set(taxonomy_labels)))}
    popularity_values = {value: index for index, value in enumerate(sorted(set(popularity_labels)))}
    arrays: dict[str, np.ndarray] = {
        "positive_index": positive_index,
        "baseline_rank": baseline_rank,
        "baseline_session_hit": baseline_hit,
        "eligible_from": eligible_from,
        "training_indices": training_indices,
        "training_length": training_length,
        "family_index": family_index,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "taxonomy_code": np.asarray([taxonomy_values[value] for value in taxonomy_labels], dtype=np.uint8),
        "popularity_code": np.asarray([popularity_values[value] for value in popularity_labels], dtype=np.uint8),
    }
    _numeric_audit(arrays)
    if label_path.exists() or label_path.is_symlink():
        raise FileExistsError(label_path)
    with label_path.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    if _sha256(feature_path) != sealed_feature_sha256:
        raise SmallRankerCacheError("feature cache changed during label join")
    if any(_identity_shape_scan(path) for path in (feature_path, label_path)):
        raise SmallRankerCacheError("identity-shaped token found in numeric cache")
    positives_per_fold = [
        int(np.sum((outer_fold[:, None] == fold) & (positive_index >= 0)))
        for fold in range(5)
    ]
    family_counts_per_fold = [int(len(set(family_index[outer_fold == fold].tolist()))) for fold in range(5)]
    session_counts_per_fold = [int(np.sum(outer_fold == fold)) for fold in range(5)]
    return (
        {
            "sealed_feature_sha256_before_join": sealed_feature_sha256,
            "arrays": _array_registry(arrays),
            "label_file": {"bytes": label_path.stat().st_size, "sha256": _sha256(label_path)},
            "baseline_hit_sessions": int(baseline_hit.sum()),
            "baseline_miss_sessions": int(SESSION_COUNT - baseline_hit.sum()),
            "c100_positive_query_groups": int(np.sum(positive_index >= 0)),
            "c100_absent_query_groups": int(np.sum(positive_index < 0)),
            "trainable_rows": int(training_length.sum()),
            "trainable_query_groups": int(np.sum(training_length > 0)),
            "unique_label_count": len(set(labels)),
            "product_family_count": family_count,
            "family_cross_parent_merge_count": len(set(labels)) - family_count,
            "family_grouping_evidence_boundary": (
                "catalog-metadata signatures are used first; if their count equals the "
                "unique-label count, this split is exact-parent grouped and may be optimistic"
            ),
            "session_counts_per_fold": session_counts_per_fold,
            "positive_query_groups_per_fold": positives_per_fold,
            "families_per_fold": family_counts_per_fold,
            "taxonomy_codebook": taxonomy_values,
            "popularity_codebook": popularity_values,
            "family_rule_sha256": _canonical_sha256(
                {
                    "rule": "leaf-category + normalized-store + variant-stripped-title-signature",
                    "variant_tokens": sorted(VARIANT_TOKENS),
                    "title_token_limit": family_title_token_limit,
                }
            ),
        },
        set(labels),
    )


def build(
    config_path: Path,
    *,
    resume_feature_sha256: str | None = None,
    completed_feature_seconds: float | None = None,
    feature_generator_sha256: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = config_path.resolve()
    config = _load_config(config_path)
    config_sha256 = _sha256(config_path)
    catalog_path = _resolve_input(config, "catalog")
    proxy_path = _resolve_input(config, "proxy")
    sidecar_path = _resolve_input(config, "p11_sidecar")
    for filename, expected_hash in config["inputs"]["trace_shards"]:
        trace_path = ROOT / config["inputs"]["trace_dir"] / filename
        if not trace_path.is_file() or _sha256(trace_path) != expected_hash:
            raise SmallRankerCacheError(f"blind trace identity mismatch: {filename}")

    cache_dir = (ROOT / config["cache"]["directory"]).resolve()
    experiment_root = (ROOT / "experiments/fast_track").resolve()
    if not _inside(cache_dir, experiment_root):
        raise SmallRankerCacheError("cache output escapes experiments/fast_track")
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / str(config["cache"]["features"])
    label_path = cache_dir / str(config["cache"]["labels"])
    manifest_path = (ROOT / str(config["cache"]["manifest"])).resolve()
    if not _inside(manifest_path, (ROOT / "configs").resolve()):
        raise SmallRankerCacheError("tracked manifest escapes configs")
    resuming = resume_feature_sha256 is not None
    if resuming:
        if not feature_path.is_file() or _sha256(feature_path) != resume_feature_sha256:
            raise SmallRankerCacheError("resume requires the exact completed feature SHA-256")
        for output in (label_path, manifest_path):
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
        if completed_feature_seconds is None or completed_feature_seconds <= 0:
            raise SmallRankerCacheError("resume requires the completed feature wall time")
        if not feature_generator_sha256 or not re.fullmatch(r"[0-9a-f]{64}", feature_generator_sha256):
            raise SmallRankerCacheError("resume requires the pre-fix feature-generator SHA-256")
    else:
        for output in (feature_path, label_path, manifest_path):
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)

    with tempfile.TemporaryDirectory(prefix="small-ranker-context-", dir=cache_dir) as temporary:
        temporary_path = Path(temporary)
        context_path = temporary_path / "visible_context.jsonl"
        phase_manifest_path = temporary_path / "feature_phase.json"
        tick = time.perf_counter()
        context_stats = materialize_visible_context(proxy_path, catalog_path, context_path)
        context_seconds = time.perf_counter() - tick
        # The feature worker receives neither proxy path nor labels.  It can only
        # see the sanitized context, frozen blind trace, catalog, and sidecar.
        tick = time.perf_counter()
        if resuming:
            feature_cache = np.load(feature_path, mmap_mode="r")
            expected_shape = (SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT, len(FEATURE_NAMES))
            if feature_cache.shape != expected_shape or feature_cache.dtype != np.dtype("float32"):
                raise SmallRankerCacheError("resumed feature cache shape/dtype mismatch")
            del feature_cache
            feature_seconds = float(completed_feature_seconds)
            phase_metadata = {
                "schema_version": "small-ranker-feature-phase.v1",
                "label_source_opened": False,
                "feature_shape": list(expected_shape),
                "feature_dtype": "float32",
                "feature_names": list(FEATURE_NAMES),
                "feature_schema_sha256": _canonical_sha256(list(FEATURE_NAMES)),
                "feature_file": {
                    "bytes": feature_path.stat().st_size,
                    "sha256": resume_feature_sha256,
                },
                "resume": {
                    "reason": "post-seal tuple/list label-order assertion fix",
                    "feature_generator_sha256": feature_generator_sha256,
                    "feature_rebuilt": False,
                    "validation_seconds": round(time.perf_counter() - tick, 6),
                },
            }
        else:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--internal-feature-phase",
                "--context",
                str(context_path),
                "--catalog",
                str(catalog_path),
                "--sidecar",
                str(sidecar_path),
                "--features",
                str(feature_path),
                "--phase-manifest",
                str(phase_manifest_path),
            ]
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise SmallRankerCacheError("isolated target-blind feature subprocess failed")
            feature_seconds = time.perf_counter() - tick
            phase_metadata = json.loads(phase_manifest_path.read_text(encoding="utf-8"))
        if phase_metadata.get("label_source_opened") is not False:
            raise SmallRankerCacheError("feature phase did not prove label isolation")
        if phase_metadata.get("feature_file", {}).get("sha256") != _sha256(feature_path):
            raise SmallRankerCacheError("feature phase seal mismatch")
        tick = time.perf_counter()
        label_metadata, labels = join_labels_after_feature_seal(
            proxy_path=proxy_path,
            catalog_path=catalog_path,
            feature_path=feature_path,
            label_path=label_path,
            seed=int(config["seed"]),
            family_title_token_limit=int(config["family_grouping"]["title_token_limit"]),
        )
        label_seconds = time.perf_counter() - tick

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split": "train_explore",
        "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": config_sha256},
        "sources": {
            "catalog_sha256": config["inputs"]["catalog"]["sha256"],
            "proxy_sha256": config["inputs"]["proxy"]["sha256"],
            "p11_sidecar_sha256": config["inputs"]["p11_sidecar"]["sha256"],
            "blind_trace_sha256": [value[1] for value in config["inputs"]["trace_shards"]],
        },
        "feature_cache": {
            "path": feature_path.relative_to(ROOT).as_posix(),
            **phase_metadata["feature_file"],
            "shape": phase_metadata["feature_shape"],
            "dtype": phase_metadata["feature_dtype"],
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "feature_schema_sha256": phase_metadata["feature_schema_sha256"],
            "rows": SESSION_COUNT * TURN_COUNT * CANDIDATE_COUNT,
            "query_groups": SESSION_COUNT * TURN_COUNT,
        },
        "label_cache": {
            "path": label_path.relative_to(ROOT).as_posix(),
            **label_metadata["label_file"],
            "arrays": label_metadata["arrays"],
        },
        "context_adapter": {
            **context_stats,
            "ephemeral_deleted_after_build": True,
            "official_simulator_only": True,
        },
        "label_join": {
            key: value
            for key, value in label_metadata.items()
            if key not in {"arrays", "label_file"}
        },
        "phase_boundary": {
            "feature_worker_received_proxy_path": False,
            "feature_worker_opened_label_source": False,
            "feature_sha256_computed_before_label_join": True,
            "label_only_numeric_arrays": True,
            "resumed_post_seal_join": resuming,
            "feature_rebuilt_during_resume": False if resuming else None,
        },
        "privacy": {
            "identity_features": False,
            "string_or_object_cache_arrays": 0,
            "reversible_family_mapping": False,
            "numeric_cache_asin_shape_matches": 0,
            "visible_context_asin_shape_matches": 0,
            "forbidden_artifact_keys": sorted(FORBIDDEN_ARTIFACT_KEYS),
        },
        "timing_seconds": {
            "visible_context_adapter": round(context_seconds, 6),
            "isolated_feature_phase": round(feature_seconds, 6),
            "post_seal_label_join": round(label_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
        "builder": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
            "feature_generator_sha256": feature_generator_sha256 or _sha256(Path(__file__).resolve()),
        },
    }
    if _walk_keys(manifest) & {"asin", "parent_asin", "target_id", "ground_truth", "sample_id", "user_id", "ordinal"}:
        raise SmallRankerCacheError("tracked manifest contains a forbidden identity key")
    serialized_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if ASIN_SHAPE_RE.search(serialized_manifest) or any(label in serialized_manifest for label in labels):
        raise SmallRankerCacheError("tracked manifest contains an identity value")
    manifest["canonical_sha256"] = _canonical_sha256(manifest)
    with manifest_path.open("xb") as handle:
        handle.write(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--internal-feature-phase", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--context", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--catalog", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--sidecar", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--features", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--phase-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume-feature-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--completed-feature-seconds", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--feature-generator-sha256", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.internal_feature_phase:
        required = (args.context, args.catalog, args.sidecar, args.features, args.phase_manifest)
        if any(value is None for value in required):
            raise SmallRankerCacheError("internal feature phase arguments are incomplete")
        result = build_target_blind_features(
            args.context.resolve(),
            args.catalog.resolve(),
            args.sidecar.resolve(),
            args.features.resolve(),
            args.phase_manifest.resolve(),
        )
        print(json.dumps({"feature_sha256": result["feature_file"]["sha256"]}, sort_keys=True))
        return 0
    result = build(
        args.config,
        resume_feature_sha256=args.resume_feature_sha256,
        completed_feature_seconds=args.completed_feature_seconds,
        feature_generator_sha256=args.feature_generator_sha256,
    )
    print(
        json.dumps(
            {
                "feature_bytes": result["feature_cache"]["bytes"],
                "feature_count": result["feature_cache"]["feature_count"],
                "feature_sha256": result["feature_cache"]["sha256"],
                "label_sha256": result["label_cache"]["sha256"],
                "manifest": result["config"]["path"].replace("small_ranker_v1.json", "small_ranker_v1.cache.manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SmallRankerCacheError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[small-ranker-cache] {error}", file=sys.stderr)
        raise SystemExit(1)
