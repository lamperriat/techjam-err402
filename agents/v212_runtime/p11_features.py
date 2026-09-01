"""Catalog-only evidence and a fixed Top-10-preserving P11 scorer.

This module is experiment infrastructure.  It neither imports evaluator data nor
changes the served :class:`starter.agent.Agent`.  Static catalog evidence is read
from a frozen SQLite sidecar in one bounded batch; the online scorer only
permutes the exact R08 Top 10 and preserves the remaining tail byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .agent import _terms
from .attributes import normalize_value
from .p8_negative import (
    COMPATIBLE,
    EXPLICIT_VIOLATION,
    UNKNOWN,
    ExecutableNegative,
)
from .p9_evidence import (
    REGISTRY_SHA256 as P9_REGISTRY_SHA256,
    SEMANTICS_SHA256 as P9_SEMANTICS_SHA256,
    SLOT_ORDER as NEGATIVE_SLOT_ORDER,
    classify_masks,
    compile_mask_constraints,
)


SCHEMA_VERSION = "p11.top10-features.v2"
SCORER_VERSION = "p11.top10-linear.v3"
MAX_TOP_K = 10
MAX_QUERY_TERMS = 50
MAX_HARD_CLAUSE_TERMS = 12
HARD_CLAUSE_NGRAM_WEIGHTS = ((3, 1.0), (2, 0.5))
FULL_CLAUSE_MIN_TERMS = 4
FIELD_GROUPS = (
    "title_category",
    "features_details",
    "description_store",
)
FIELD_WEIGHTS = (1.0, 0.9, 0.5)
VALUE_SEPARATOR = "\x1f"
SEQUENCE_SEPARATOR = "\x1e"
COMPONENT_SEPARATOR = "\x1d"
FEATURE_COMPONENT_COUNT = len(FIELD_GROUPS) + 4
FEATURE_COMPRESSION_LEVEL = 9
FEATURE_ENCODING = "zlib-components-v1"
MAX_DECOMPRESSED_FEATURE_BYTES = 1_048_576

WEIGHTS = {
    "broad_rank_prior": 0.03,
    "strict_rank_prior": 0.06,
    "rrf_rank_prior": 0.16,
    "idf_any_field_coverage": 0.24,
    "title_category_coverage": 0.11,
    "features_details_coverage": 0.08,
    "description_store_coverage": 0.03,
    "latest_hard_clause_coverage": 0.10,
    "subtype_consistency": 0.10,
    "positive_constraint_evidence": 0.09,
}
TIE_WEIGHTS = {
    "subtype_bayesian_rating_percentile": 0.0015,
    "subtype_log_rating_count_percentile": 0.0005,
}
NEAR_TIE_MAX_DELTA = round(sum(TIE_WEIGHTS.values()), 12)
SCORE_DECIMAL_PLACES = 12
SCORE_SCALE = 10 ** SCORE_DECIMAL_PLACES
POSITIVE_EVIDENCE_VALUES = {
    "observed": 1.0,
    "inferred": 0.45,
    "unknown": 0.0,
}
CONFLICT_BUCKETS = {
    COMPATIBLE: 0,
    UNKNOWN: 1,
    EXPLICIT_VIOLATION: 2,
    "not_applicable": 0,
}
SQL_NEGATIVE_MASK_COLUMNS = tuple(
    f"negative_{slot}_mask" for slot in NEGATIVE_SLOT_ORDER
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


REGISTRY_SHA256 = _canonical_sha256({
    "schema_version": SCHEMA_VERSION,
    "scorer_version": SCORER_VERSION,
    "field_groups": FIELD_GROUPS,
    "field_weights": FIELD_WEIGHTS,
    "weights": WEIGHTS,
    "tie_weights": TIE_WEIGHTS,
    "near_tie": {
        "maximum_relevance_delta": NEAR_TIE_MAX_DELTA,
        "score_decimal_places": SCORE_DECIMAL_PLACES,
        "group_anchor": "highest_remaining_relevance_within_conflict_bucket",
        "quality_bonus_scope": "within_near_tie_group_only",
    },
    "hard_clause": {
        "maximum_terms": MAX_HARD_CLAUSE_TERMS,
        "ngram_width_weights": HARD_CLAUSE_NGRAM_WEIGHTS,
        "full_clause_minimum_terms": FULL_CLAUSE_MIN_TERMS,
        "full_clause_weight": "sum_of_local_ngram_weights",
    },
    "positive_evidence_values": POSITIVE_EVIDENCE_VALUES,
    "negative_slot_order": NEGATIVE_SLOT_ORDER,
    "p9_registry_sha256": P9_REGISTRY_SHA256,
    "feature_blob": {
        "encoding": FEATURE_ENCODING,
        "level": FEATURE_COMPRESSION_LEVEL,
        "component_count": FEATURE_COMPONENT_COUNT,
        "component_separator": ord(COMPONENT_SEPARATOR),
        "sequence_separator": ord(SEQUENCE_SEPARATOR),
        "value_separator": ord(VALUE_SEPARATOR),
    },
    "term_stats": {
        "columns": ("term", "document_frequency"),
        "idf_reconstructed_from_frozen_catalog_rows": True,
    },
})
SEMANTICS_SHA256 = _canonical_sha256({
    "candidate_boundary": "exact_r08_final_top10",
    "tail": "preserved_exactly",
    "idf": "ln(1+(N-df+0.5)/(df+0.5))",
    "conflict_order": (COMPATIBLE, UNKNOWN, EXPLICIT_VIOLATION),
    "near_tie_order": (
        "partition_by_conflict_then_descending_relevance; greedily group from the "
        "highest remaining relevance when anchor-current <= 0.002 after exact "
        "12-decimal integer scaling; apply catalog quality bonus only inside each group"
    ),
    "hard_clause_coverage": (
        "exact field-local bigram/trigram evidence plus an equal-mass exact full-clause "
        "match for clauses of at least four terms"
    ),
    "unknown_is_violation": False,
    "p9_semantics_sha256": P9_SEMANTICS_SHA256,
    "catalog_text_storage": "one compressed sequence payload without token copies",
    "field_tokens": "derived by whitespace split after bounded Top10 decompression",
    "runtime_json_parsing": False,
    "runtime_candidate_regex": False,
})


def encode_values(values: Iterable[str]) -> str:
    """Encode normalized, unique scalar values without JSON."""

    cleaned = sorted({str(value) for value in values if str(value)})
    if any(
        separator in value
        for value in cleaned
        for separator in (COMPONENT_SEPARATOR, SEQUENCE_SEPARATOR, VALUE_SEPARATOR)
    ):
        raise ValueError("P11 values contain a reserved separator")
    return VALUE_SEPARATOR.join(cleaned)


def encode_sequences(values: Iterable[str]) -> str:
    """Encode normalized token sequences while preserving first occurrence."""

    cleaned = tuple(dict.fromkeys(str(value) for value in values if str(value)))
    if any(
        separator in value
        for value in cleaned
        for separator in (COMPONENT_SEPARATOR, SEQUENCE_SEPARATOR, VALUE_SEPARATOR)
    ):
        raise ValueError("P11 sequences contain a reserved separator")
    return SEQUENCE_SEPARATOR.join(cleaned)


def encode_feature_blob(
    field_sequences: Sequence[Iterable[str]],
    observed_values: Iterable[str],
    inferred_values: Iterable[str],
    observed_subtypes: Iterable[str],
    inferred_subtypes: Iterable[str],
) -> bytes:
    """Encode one product's static evidence once, without JSON or token copies."""

    if len(field_sequences) != len(FIELD_GROUPS):
        raise ValueError("P11 feature payload requires exactly three field groups")
    components = (
        *(encode_sequences(values) for values in field_sequences),
        encode_values(observed_values),
        encode_values(inferred_values),
        encode_values(observed_subtypes),
        encode_values(inferred_subtypes),
    )
    raw = COMPONENT_SEPARATOR.join(components).encode("utf-8")
    if len(raw) > MAX_DECOMPRESSED_FEATURE_BYTES:
        raise ValueError("P11 feature payload exceeds the decompressed byte limit")
    return zlib.compress(raw, FEATURE_COMPRESSION_LEVEL)


def normalize_query_terms(values: Iterable[str]) -> tuple[str, ...]:
    """Apply the served sparse tokenizer once at query level, never per candidate."""

    return tuple(dict.fromkeys(_terms(" ".join(str(value) for value in values))))[
        :MAX_QUERY_TERMS
    ]


@dataclass(frozen=True, slots=True)
class PositiveConstraint:
    slot: str
    value: str
    hardness: str = "soft"
    source_turn: int = 1
    version: int = 1


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    parent_asin: str
    field_tokens: tuple[frozenset[str], frozenset[str], frozenset[str]]
    field_sequences: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    observed_values: frozenset[str]
    inferred_values: frozenset[str]
    observed_subtypes: frozenset[str]
    inferred_subtypes: frozenset[str]
    negative_masks: tuple[int, ...]
    bayesian_rating_percentile: float
    popularity_percentile: float


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    evidence: Mapping[str, CandidateEvidence]
    idf_by_term: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class CandidateScore:
    total: float
    relevance: float
    tie_bonus: float
    conflict_state: str
    broad_rank_prior: float
    strict_rank_prior: float
    rrf_rank_prior: float
    idf_any_field_coverage: float
    title_category_coverage: float
    features_details_coverage: float
    description_store_coverage: float
    latest_hard_clause_coverage: float
    subtype_consistency: float
    positive_constraint_evidence: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class P11RerankResult:
    identifiers: tuple[str, ...]
    fallback: bool
    reason: str
    changed_top10_order: bool
    breakdowns: Mapping[str, CandidateScore]


class P11FeatureStore:
    """Read-only catalog sidecar with a strict, at-most-ten-row fetch API."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_catalog_sha256: str | None = None,
        expected_catalog_rows: int | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"P11 sidecar does not exist: {self.path}")
        uri = f"{self.path.as_uri()}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.connection.execute("PRAGMA query_only=ON")
        try:
            self.metadata = self._validate(
                expected_catalog_sha256=expected_catalog_sha256,
                expected_catalog_rows=expected_catalog_rows,
            )
        except Exception:
            self.connection.close()
            raise

    def _validate(
        self,
        *,
        expected_catalog_sha256: str | None,
        expected_catalog_rows: int | None,
    ) -> dict[str, str]:
        rows = self.connection.execute("SELECT key, value FROM metadata").fetchall()
        metadata = {str(key): str(value) for key, value in rows}
        required = {
            "schema_version": SCHEMA_VERSION,
            "registry_sha256": REGISTRY_SHA256,
            "semantics_sha256": SEMANTICS_SHA256,
            "negative_slot_order": ",".join(NEGATIVE_SLOT_ORDER),
            "feature_encoding": FEATURE_ENCODING,
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"P11 sidecar {key} mismatch")
        if expected_catalog_sha256 is not None:
            if metadata.get("catalog_sha256") != expected_catalog_sha256.lower():
                raise ValueError("P11 sidecar catalog SHA-256 mismatch")
        try:
            catalog_rows = int(metadata["catalog_rows"])
        except (KeyError, ValueError) as error:
            raise ValueError("P11 sidecar catalog_rows is invalid") from error
        if expected_catalog_rows is not None and catalog_rows != expected_catalog_rows:
            raise ValueError("P11 sidecar catalog row count mismatch")

        columns = self.connection.execute("PRAGMA table_info(evidence)").fetchall()
        expected_columns = (
            "catalog_rowid",
            "parent_asin",
            "feature_blob",
            *SQL_NEGATIVE_MASK_COLUMNS,
            "bayesian_rating_percentile",
            "popularity_percentile",
        )
        if tuple(str(column[1]) for column in columns) != expected_columns:
            raise ValueError("P11 evidence table schema mismatch")
        term_columns = self.connection.execute(
            "PRAGMA table_info(term_stats)"
        ).fetchall()
        if tuple(str(column[1]) for column in term_columns) != (
            "term",
            "document_frequency",
        ):
            raise ValueError("P11 term_stats table schema mismatch")
        summary = self.connection.execute(
            "SELECT COUNT(*), MIN(catalog_rowid), MAX(catalog_rowid), "
            "COUNT(DISTINCT parent_asin) FROM evidence"
        ).fetchone()
        if summary != (catalog_rows, 1, catalog_rows, catalog_rows):
            raise ValueError("P11 sidecar rows are not continuous and unique")
        return metadata

    @staticmethod
    def _decode_values(value: object) -> frozenset[str]:
        text = str(value or "")
        return frozenset(item for item in text.split(VALUE_SEPARATOR) if item)

    @staticmethod
    def _decode_sequences(value: object) -> tuple[str, ...]:
        text = str(value or "")
        return tuple(item for item in text.split(SEQUENCE_SEPARATOR) if item)

    @classmethod
    def _decode_feature_blob(
        cls,
        value: object,
    ) -> tuple[
        tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
        frozenset[str],
        frozenset[str],
        frozenset[str],
        frozenset[str],
    ]:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise ValueError("P11 feature payload must be a BLOB")
        try:
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(
                bytes(value),
                MAX_DECOMPRESSED_FEATURE_BYTES + 1,
            )
            if (
                len(raw) > MAX_DECOMPRESSED_FEATURE_BYTES
                or decompressor.unconsumed_tail
            ):
                raise ValueError(
                    "P11 feature payload exceeds the decompressed byte limit"
                )
            if not decompressor.eof or decompressor.unused_data:
                raise ValueError("P11 feature payload is not one complete zlib stream")
            text = raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, zlib.error) as error:
            raise ValueError("P11 feature payload is not valid compressed UTF-8") from error
        components = text.split(COMPONENT_SEPARATOR)
        if len(components) != FEATURE_COMPONENT_COUNT:
            raise ValueError("P11 feature payload component count mismatch")
        sequences = tuple(
            cls._decode_sequences(value) for value in components[: len(FIELD_GROUPS)]
        )
        return (
            (sequences[0], sequences[1], sequences[2]),
            cls._decode_values(components[3]),
            cls._decode_values(components[4]),
            cls._decode_values(components[5]),
            cls._decode_values(components[6]),
        )

    def fetch_top10(
        self,
        requested: Sequence[tuple[int, str]],
        query_terms: Iterable[str],
    ) -> FeatureBatch:
        """Fetch exactly the requested bounded rows and the query terms' frozen IDF."""

        if len(requested) > MAX_TOP_K:
            raise ValueError("P11 fetch is limited to the current Top 10")
        if not requested:
            return FeatureBatch({}, {})
        normalized_requested: list[tuple[int, str]] = []
        for rowid, parent_asin in requested:
            if not isinstance(rowid, int) or isinstance(rowid, bool) or rowid <= 0:
                raise ValueError("P11 catalog rowids must be positive integers")
            identifier = str(parent_asin)
            if not identifier:
                raise ValueError("P11 parent_asin must be non-empty")
            normalized_requested.append((rowid, identifier))
        rowids = [rowid for rowid, _ in normalized_requested]
        identifiers = [identifier for _, identifier in normalized_requested]
        if len(rowids) != len(set(rowids)) or len(identifiers) != len(set(identifiers)):
            raise ValueError("P11 requested rows and ASINs must be unique")

        placeholders = ",".join("?" for _ in rowids)
        rows = self.connection.execute(
            "SELECT * FROM evidence WHERE catalog_rowid IN (" + placeholders + ")",
            rowids,
        ).fetchall()
        by_rowid = {int(row[0]): row for row in rows}
        if len(by_rowid) != len(normalized_requested):
            raise ValueError("P11 sidecar is missing requested candidate rows")

        evidence: dict[str, CandidateEvidence] = {}
        for rowid, expected_asin in normalized_requested:
            row = by_rowid[rowid]
            if str(row[1]) != expected_asin:
                raise ValueError("P11 sidecar rowid-to-ASIN binding mismatch")
            field_sequences, observed, inferred, observed_subtypes, inferred_subtypes = (
                self._decode_feature_blob(row[2])
            )
            masks_start = 3
            masks_end = masks_start + len(NEGATIVE_SLOT_ORDER)
            negative_masks = tuple(int(value) for value in row[masks_start:masks_end])
            if len(negative_masks) != len(NEGATIVE_SLOT_ORDER) or any(
                value < 0 for value in negative_masks
            ):
                raise ValueError("P11 sidecar negative masks are invalid")
            bayesian = float(row[masks_end])
            popularity = float(row[masks_end + 1])
            if not (
                math.isfinite(bayesian)
                and math.isfinite(popularity)
                and 0.0 <= bayesian <= 1.0
                and 0.0 <= popularity <= 1.0
            ):
                raise ValueError("P11 sidecar priors are outside [0, 1]")
            evidence[expected_asin] = CandidateEvidence(
                parent_asin=expected_asin,
                field_tokens=tuple(
                    frozenset(
                        token
                        for sequence in sequences
                        for token in sequence.split()
                    )
                    for sequences in field_sequences
                ),
                field_sequences=field_sequences,
                observed_values=observed,
                inferred_values=inferred,
                observed_subtypes=observed_subtypes,
                inferred_subtypes=inferred_subtypes,
                negative_masks=negative_masks,
                bayesian_rating_percentile=bayesian,
                popularity_percentile=popularity,
            )

        terms = normalize_query_terms(query_terms)
        idf_by_term: dict[str, float] = {}
        if terms:
            term_placeholders = ",".join("?" for _ in terms)
            term_rows = self.connection.execute(
                "SELECT term, document_frequency FROM term_stats WHERE term IN ("
                + term_placeholders
                + ")",
                terms,
            ).fetchall()
            catalog_rows = int(self.metadata["catalog_rows"])
            for term, raw_df in term_rows:
                document_frequency = int(raw_df)
                if not 1 <= document_frequency <= catalog_rows:
                    raise ValueError(
                        "P11 sidecar document frequency is outside catalog bounds"
                    )
                idf = math.log(
                    1.0
                    + (catalog_rows - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                if not math.isfinite(idf) or idf <= 0.0:
                    raise ValueError("P11 sidecar IDF must be finite and positive")
                idf_by_term[str(term)] = idf
        return FeatureBatch(evidence, idf_by_term)

    def resolve_query_subtypes(self, category_text: str) -> tuple[str, ...]:
        """Resolve longest catalog-derived subtype n-grams from visible text."""

        tokens = normalize_value(category_text).split()[:12]
        candidates = {
            " ".join(tokens[start:start + width])
            for width in range(1, min(4, len(tokens)) + 1)
            for start in range(0, len(tokens) - width + 1)
        }
        if not candidates:
            return ()
        placeholders = ",".join("?" for _ in candidates)
        rows = self.connection.execute(
            "SELECT subtype, document_frequency FROM subtype_stats WHERE subtype IN ("
            + placeholders
            + ")",
            sorted(candidates),
        ).fetchall()
        ranked = sorted(
            ((str(subtype), int(df)) for subtype, df in rows),
            key=lambda item: (-len(item[0].split()), item[1], item[0]),
        )
        return tuple(subtype for subtype, _ in ranked[:4])

    def close(self) -> None:
        self.connection.close()


def _rank_prior(rank: object, offset: int) -> float:
    if rank is None:
        return 0.0
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("route ranks must be positive integers or None")
    return (offset + 1.0) / (offset + rank)


def _idf_coverage(
    terms: Sequence[str],
    idf_by_term: Mapping[str, float],
    token_sets: Sequence[frozenset[str]],
) -> tuple[float, tuple[float, float, float]]:
    weighted = [(term, float(idf_by_term[term])) for term in terms if term in idf_by_term]
    if not weighted:
        return 0.0, (0.0, 0.0, 0.0)
    denominator = sum(value for _, value in weighted)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("IDF denominator must be finite and positive")
    field = tuple(
        sum(value for term, value in weighted if term in tokens) / denominator
        for tokens in token_sets
    )
    any_coverage = (
        sum(
            value
            for term, value in weighted
            if any(term in tokens for tokens in token_sets)
        )
        / denominator
    )
    return any_coverage, (field[0], field[1], field[2])


def _prepare_hard_clause_grams(
    hard_clause_terms: Iterable[str],
) -> tuple[tuple[str, float], ...]:
    """Tokenize the visible hard clause once before scoring any candidate."""

    ordered_terms = _terms(" ".join(str(value) for value in hard_clause_terms))[
        :MAX_HARD_CLAUSE_TERMS
    ]
    grams: list[tuple[str, float]] = []
    for width, weight in HARD_CLAUSE_NGRAM_WEIGHTS:
        grams.extend(
            (" ".join(ordered_terms[index:index + width]), weight)
            for index in range(0, len(ordered_terms) - width + 1)
        )
    if len(ordered_terms) >= FULL_CLAUSE_MIN_TERMS:
        local_weight = sum(weight for _, weight in grams)
        grams.append((" ".join(ordered_terms), local_weight))
    return tuple(grams)


def _hard_clause_coverage(
    grams: Sequence[tuple[str, float]],
    field_sequences: Sequence[Sequence[str]],
) -> float:
    if not grams:
        return 0.0
    numerator = 0.0
    denominator = sum(weight for _, weight in grams)
    for gram, weight in grams:
        padded = f" {gram} "
        best = 0.0
        for field_weight, sequences in zip(FIELD_WEIGHTS, field_sequences):
            if any(padded in f" {sequence} " for sequence in sequences):
                best = max(best, field_weight)
        numerator += weight * best
    return numerator / denominator


def _prepare_positive_constraints(
    constraints: Iterable[PositiveConstraint],
    *,
    current_turn: int,
    current_version: int,
) -> tuple[tuple[str, float], ...]:
    if current_turn <= 0 or current_version <= 0:
        raise ValueError("current turn and version must be positive")
    prepared: list[tuple[str, float]] = []
    for constraint in constraints:
        slot = normalize_value(constraint.slot).replace(" ", "_")
        value = normalize_value(constraint.value)
        if not slot or not value:
            continue
        if (
            not isinstance(constraint.source_turn, int)
            or isinstance(constraint.source_turn, bool)
            or not 1 <= constraint.source_turn <= current_turn
        ):
            raise ValueError("positive constraint source_turn is invalid")
        if (
            not isinstance(constraint.version, int)
            or isinstance(constraint.version, bool)
            or not 1 <= constraint.version <= current_version
        ):
            raise ValueError("positive constraint version is invalid")
        version_age = current_version - constraint.version
        version_weight = max(0.5, 0.75 ** version_age)
        turn_weight = 0.8 + 0.2 * constraint.source_turn / current_turn
        hardness_weight = 1.25 if constraint.hardness == "hard" else 1.0
        prepared.append((
            f"{slot}={value}",
            hardness_weight * version_weight * turn_weight,
        ))
    return tuple(prepared)


def _positive_evidence(
    prepared: Sequence[tuple[str, float]],
    candidate: CandidateEvidence,
) -> float:
    if not prepared:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for key, weight in prepared:
        if key in candidate.observed_values:
            state = "observed"
        elif key in candidate.inferred_values:
            state = "inferred"
        else:
            state = "unknown"
        numerator += weight * POSITIVE_EVIDENCE_VALUES[state]
        denominator += weight
    return numerator / denominator if denominator else 0.0


def _subtype_consistency(
    query_subtypes: frozenset[str],
    candidate: CandidateEvidence,
) -> float:
    if not query_subtypes:
        return 0.0
    if query_subtypes & candidate.observed_subtypes:
        return 1.0
    if query_subtypes & candidate.inferred_subtypes:
        return 0.5
    return 0.0


def _fallback(identifiers: Sequence[str], reason: str) -> P11RerankResult:
    return P11RerankResult(
        tuple(identifiers),
        True,
        reason,
        False,
        {},
    )


def _score_units(value: float) -> int:
    """Convert a frozen 12-decimal score to exact deterministic integer units."""

    if not math.isfinite(value):
        raise ValueError("P11 ordering score must be finite")
    return int(round(value * SCORE_SCALE))


def _near_tie_groups(
    identifiers: Sequence[str],
    breakdowns: Mapping[str, CandidateScore],
    base_rank: Mapping[str, int],
) -> tuple[tuple[str, ...], ...]:
    """Partition one conflict bucket into bounded, non-chaining near-tie groups."""

    relevance_order = sorted(
        identifiers,
        key=lambda identifier: (
            -_score_units(breakdowns[identifier].relevance),
            base_rank[identifier],
        ),
    )
    threshold_units = _score_units(NEAR_TIE_MAX_DELTA)
    groups: list[tuple[str, ...]] = []
    cursor = 0
    while cursor < len(relevance_order):
        anchor_units = _score_units(breakdowns[relevance_order[cursor]].relevance)
        end = cursor + 1
        while end < len(relevance_order):
            current_units = _score_units(breakdowns[relevance_order[end]].relevance)
            if anchor_units - current_units > threshold_units:
                break
            end += 1
        groups.append(tuple(relevance_order[cursor:end]))
        cursor = end
    return tuple(groups)


def _order_with_near_tie_quality(
    identifiers: Sequence[str],
    breakdowns: Mapping[str, CandidateScore],
    base_rank: Mapping[str, int],
) -> tuple[str, ...]:
    """Apply quality priors only inside relevance near-ties and conflict buckets."""

    ordered: list[str] = []
    for bucket in sorted(set(CONFLICT_BUCKETS.values())):
        members = tuple(
            identifier
            for identifier in identifiers
            if CONFLICT_BUCKETS[breakdowns[identifier].conflict_state] == bucket
        )
        for group in _near_tie_groups(members, breakdowns, base_rank):
            ordered.extend(sorted(
                group,
                key=lambda identifier: (
                    -_score_units(breakdowns[identifier].total),
                    base_rank[identifier],
                ),
            ))
    return tuple(ordered)


def rerank_top10_preserving_membership(
    identifiers: Sequence[str],
    batch: FeatureBatch,
    *,
    query_terms: Iterable[str],
    broad_ranks: Mapping[str, int],
    strict_ranks: Mapping[str, int],
    fused_ranks: Mapping[str, int],
    positive_constraints: Iterable[PositiveConstraint] = (),
    negative_constraints: Iterable[ExecutableNegative] = (),
    query_subtypes: Iterable[str] = (),
    hard_clause_terms: Iterable[str] = (),
    current_turn: int = 1,
    current_version: int = 1,
) -> P11RerankResult:
    """Safely rerank only the current R08 Top 10, or return it exactly."""

    original = tuple(str(identifier) for identifier in identifiers)
    try:
        if any(not identifier for identifier in original):
            raise ValueError("candidate identifiers must be non-empty")
        if len(original) != len(set(original)):
            raise ValueError("candidate identifiers must be unique")
        head = original[:MAX_TOP_K]
        tail = original[MAX_TOP_K:]
        if not head:
            return P11RerankResult(original, False, "empty", False, {})
        if set(batch.evidence) != set(head):
            raise ValueError("feature batch must contain exactly the R08 Top 10")

        terms = normalize_query_terms(query_terms)
        idf_by_term = {
            str(term): float(value) for term, value in batch.idf_by_term.items()
        }
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in idf_by_term.values()
        ):
            raise ValueError("IDF values must be finite and positive")
        prepared_positive = _prepare_positive_constraints(
            positive_constraints,
            current_turn=current_turn,
            current_version=current_version,
        )
        normalized_subtypes = frozenset(
            value
            for raw in query_subtypes
            if (value := normalize_value(raw))
        )
        negatives = tuple(negative_constraints)
        compiled_negatives = compile_mask_constraints(negatives) if negatives else ()
        hard_grams = _prepare_hard_clause_grams(hard_clause_terms)

        base_rank = {identifier: rank for rank, identifier in enumerate(head, start=1)}
        breakdowns: dict[str, CandidateScore] = {}
        for identifier in head:
            candidate = batch.evidence[identifier]
            if candidate.parent_asin != identifier:
                raise ValueError("feature evidence is bound to the wrong ASIN")
            any_coverage, field_coverage = _idf_coverage(
                terms,
                idf_by_term,
                candidate.field_tokens,
            )
            conflict_state = (
                classify_masks(candidate.negative_masks, compiled_negatives)
                if compiled_negatives
                else "not_applicable"
            )
            components = {
                "broad_rank_prior": _rank_prior(broad_ranks.get(identifier), 60),
                "strict_rank_prior": _rank_prior(strict_ranks.get(identifier), 20),
                "rrf_rank_prior": _rank_prior(fused_ranks.get(identifier), 60),
                "idf_any_field_coverage": any_coverage,
                "title_category_coverage": field_coverage[0],
                "features_details_coverage": field_coverage[1],
                "description_store_coverage": field_coverage[2],
                "latest_hard_clause_coverage": _hard_clause_coverage(
                    hard_grams,
                    candidate.field_sequences,
                ),
                "subtype_consistency": _subtype_consistency(
                    normalized_subtypes,
                    candidate,
                ),
                "positive_constraint_evidence": _positive_evidence(
                    prepared_positive,
                    candidate,
                ),
            }
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in components.values()
            ):
                raise ValueError("P11 feature component is outside [0, 1]")
            relevance = sum(WEIGHTS[name] * value for name, value in components.items())
            tie_bonus = (
                TIE_WEIGHTS["subtype_bayesian_rating_percentile"]
                * candidate.bayesian_rating_percentile
                + TIE_WEIGHTS["subtype_log_rating_count_percentile"]
                * candidate.popularity_percentile
            )
            total = relevance + tie_bonus
            if not math.isfinite(total):
                raise ValueError("P11 candidate total is not finite")
            breakdowns[identifier] = CandidateScore(
                total=round(total, 12),
                relevance=round(relevance, 12),
                tie_bonus=round(tie_bonus, 12),
                conflict_state=conflict_state,
                **{name: round(value, 12) for name, value in components.items()},
            )

        proposed_head = _order_with_near_tie_quality(
            head,
            breakdowns,
            base_rank,
        )
        if (
            len(proposed_head) != len(head)
            or len(set(proposed_head)) != len(proposed_head)
            or set(proposed_head) != set(head)
        ):
            raise ValueError("P11 proposal changed Top-10 membership")
        final = (*proposed_head, *tail)
        if final[MAX_TOP_K:] != tail:
            raise ValueError("P11 proposal changed the R08 tail")
        return P11RerankResult(
            final,
            False,
            "scored",
            proposed_head != head,
            breakdowns,
        )
    except Exception as error:
        return _fallback(original, f"fallback:{type(error).__name__}")


__all__ = [
    "CONFLICT_BUCKETS",
    "FULL_CLAUSE_MIN_TERMS",
    "HARD_CLAUSE_NGRAM_WEIGHTS",
    "NEAR_TIE_MAX_DELTA",
    "CandidateEvidence",
    "CandidateScore",
    "FIELD_GROUPS",
    "FEATURE_ENCODING",
    "FeatureBatch",
    "MAX_TOP_K",
    "P11FeatureStore",
    "P11RerankResult",
    "POSITIVE_EVIDENCE_VALUES",
    "PositiveConstraint",
    "REGISTRY_SHA256",
    "SCHEMA_VERSION",
    "SCORER_VERSION",
    "SEMANTICS_SHA256",
    "SQL_NEGATIVE_MASK_COLUMNS",
    "TIE_WEIGHTS",
    "WEIGHTS",
    "encode_feature_blob",
    "encode_sequences",
    "encode_values",
    "normalize_query_terms",
    "rerank_top10_preserving_membership",
]
