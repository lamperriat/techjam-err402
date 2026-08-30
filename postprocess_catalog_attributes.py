"""Normalize and conservatively repair raw catalog attribute extractions."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from extract_attribute_pilot import (
    ALL_EXTRACTION_FIELDS,
    CORE_ENTRY_LIMIT,
    MAX_EVIDENCE_CHARS,
    SPECIFIC_ENTRY_LIMIT,
    load_catalog,
    product_input,
    validate_extraction,
)


POSTPROCESSING_VERSION = 1
DEFAULT_RAW = Path("results/catalog_attributes_raw.jsonl")
DEFAULT_OUTPUT = Path("results/catalog_attributes_processed.jsonl")
EVIDENCE_REPAIR_REASONS = {
    "evidence is not an exact substring of the supplied product data",
    "evidence exceeds 80 characters",
    "value is not an exact substring of evidence",
    "evidence is not a non-empty string",
}

# Explicit aliases only: this is not intended to become a product ontology.
ATTRIBUTE_NAME_ALIASES = {
    "closure_type": "closure",
    "closure_system": "closure",
    "insole_cushioning": "cushioning",
    "footbed_cushioning": "cushioning",
    "midsole_cushioning": "cushioning",
    "cushioning_system": "cushioning",
    "cushioning_technology": "cushioning",
    "non_slip": "slip_resistance",
    "anti_slip": "slip_resistance",
    "slip_resistant": "slip_resistance",
    "nonslip": "slip_resistance",
    "water_protection": "water_resistance",
    "waterproof": "water_resistance",
    "water_repellent": "water_resistance",
    "water_repellency": "water_resistance",
    "water_proof": "water_resistance",
    "water_resistant": "water_resistance",
    "waterproofing": "water_resistance",
}


def _source_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _source_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _source_strings(item)
    elif value not in (None, ""):
        yield str(value)


def _normalized_with_offsets(text: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    offsets: list[int] = []
    for offset, character in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            if normalized.isspace():
                if characters and characters[-1] != " ":
                    characters.append(" ")
                    offsets.append(offset)
            else:
                characters.append(normalized)
                offsets.append(offset)
    if characters and characters[-1] == " ":
        characters.pop()
        offsets.pop()
    return "".join(characters), offsets


def exact_catalog_substring(value: str, source: Mapping[str, Any]) -> str | None:
    """Return the exact source slice matching a normalized value, if present."""
    needle, _ = _normalized_with_offsets(value.strip())
    if not needle:
        return None
    for text in _source_strings(source):
        normalized, offsets = _normalized_with_offsets(text)
        search_from = 0
        while True:
            start = normalized.find(needle, search_from)
            if start < 0:
                break
            end = start + len(needle)
            left_boundary = (
                not needle[0].isalnum()
                or start == 0
                or not normalized[start - 1].isalnum()
            )
            right_boundary = (
                not needle[-1].isalnum()
                or end == len(normalized)
                or not normalized[end].isalnum()
            )
            if left_boundary and right_boundary:
                return text[offsets[start] : offsets[end - 1] + 1]
            search_from = start + 1
    return None


def repair_evidence(
    attributes: Mapping[str, list[dict[str, str]]],
    rejected_attributes: Iterable[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, str]]], int]:
    """Restore rejected entries only when their values are catalog-grounded."""
    repaired = {
        field: [dict(entry) for entry in attributes.get(field, [])]
        for field in ALL_EXTRACTION_FIELDS
    }
    repair_count = 0
    for rejection in rejected_attributes:
        field = rejection.get("field")
        entry = rejection.get("entry")
        if (
            rejection.get("reason") not in EVIDENCE_REPAIR_REASONS
            or field not in ALL_EXTRACTION_FIELDS
            or not isinstance(entry, dict)
            or not isinstance(entry.get("value"), str)
        ):
            continue
        limit = (
            SPECIFIC_ENTRY_LIMIT
            if field == "specific_attributes"
            else CORE_ENTRY_LIMIT
        )
        if len(repaired[field]) >= limit:
            continue
        evidence = exact_catalog_substring(entry["value"], source)
        if evidence is None or len(evidence) > MAX_EVIDENCE_CHARS:
            continue

        candidate = dict(entry)
        candidate["evidence"] = evidence
        candidate_document = {
            candidate_field: [] for candidate_field in ALL_EXTRACTION_FIELDS
        }
        candidate_document[field] = [candidate]
        validation = validate_extraction(candidate_document, source)
        if not validation.attributes[field]:
            continue
        candidate = validation.attributes[field][0]
        duplicate_key = (
            candidate.get("name"),
            unicodedata.normalize("NFKC", candidate["value"]).casefold(),
        )
        existing_keys = {
            (
                existing.get("name"),
                unicodedata.normalize("NFKC", existing["value"]).casefold(),
            )
            for existing in repaired[field]
        }
        if duplicate_key in existing_keys:
            continue
        repaired[field].append(candidate)
        repair_count += 1
    return repaired, repair_count


def normalize_attribute_names(
    attributes: Mapping[str, list[dict[str, str]]],
) -> tuple[dict[str, list[dict[str, str]]], int, int]:
    """Apply the explicit alias table and remove alias-created duplicates."""
    normalized = {
        field: [dict(entry) for entry in attributes.get(field, [])]
        for field in ALL_EXTRACTION_FIELDS
    }
    alias_count = 0
    duplicate_count = 0
    specific: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in normalized["specific_attributes"]:
        original_name = entry["name"]
        entry["name"] = ATTRIBUTE_NAME_ALIASES.get(original_name, original_name)
        alias_count += entry["name"] != original_name
        key = (
            entry["name"],
            unicodedata.normalize("NFKC", entry["value"]).casefold(),
        )
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        specific.append(entry)
    normalized["specific_attributes"] = specific
    return normalized, alias_count, duplicate_count


def load_latest_successes(raw_path: Path) -> tuple[dict[str, Any], dict[str, dict]]:
    """Load the latest record per product and require a complete extraction."""
    latest: dict[str, dict] = {}
    with raw_path.open(encoding="utf-8") as handle:
        metadata = json.loads(next(handle))
        if metadata.get("record_type") != "metadata":
            raise ValueError(f"Missing metadata header in {raw_path}")
        for line in handle:
            if line.strip():
                record = json.loads(line)
                latest[str(record["parent_asin"])] = record
    failures = [
        parent_asin
        for parent_asin, record in latest.items()
        if record.get("status") != "success"
    ]
    if failures:
        raise ValueError(
            f"Raw extraction still has {len(failures)} failed products; retry it first"
        )
    return metadata, latest


def postprocess_catalog(
    catalog_path: str | Path,
    raw_path: str | Path,
    output_path: str | Path,
    *,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Create a complete processed attribute file without modifying raw data."""
    catalog_path = Path(catalog_path)
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(f"Remove stale temporary output first: {temporary_path}")

    raw_metadata, latest = load_latest_successes(raw_path)
    products = load_catalog(catalog_path)
    catalog_ids = {str(product["parent_asin"]) for product in products}
    if catalog_ids != set(latest):
        raise ValueError("Catalog and latest raw extraction product IDs do not match")

    totals = {"evidence_repairs": 0, "aliased_names": 0, "duplicates_removed": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("x", encoding="utf-8") as output:
            metadata = {
                "record_type": "metadata",
                "postprocessing": {
                    "version": POSTPROCESSING_VERSION,
                    "catalog": str(catalog_path.resolve()),
                    "raw_input": str(raw_path.resolve()),
                    "raw_experiment": raw_metadata.get("experiment"),
                    "attribute_name_aliases": ATTRIBUTE_NAME_ALIASES,
                },
            }
            output.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            progress = tqdm(
                products,
                desc="Post-processing attributes",
                unit="product",
                disable=not show_progress,
            )
            for product in progress:
                parent_asin = str(product["parent_asin"])
                raw = latest[parent_asin]
                attributes, repaired = repair_evidence(
                    raw.get("attributes", {}),
                    raw.get("rejected_attributes", []),
                    product_input(product),
                )
                attributes, aliased, duplicates = normalize_attribute_names(attributes)
                totals["evidence_repairs"] += repaired
                totals["aliased_names"] += aliased
                totals["duplicates_removed"] += duplicates
                record: dict[str, Any] = {
                    "parent_asin": parent_asin,
                    "attributes": attributes,
                }
                if repaired or aliased or duplicates:
                    record["postprocessing"] = {
                        "evidence_repairs": repaired,
                        "aliased_names": aliased,
                        "duplicates_removed": duplicates,
                    }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.set_postfix(**totals, refresh=False)
        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return {
        "products": len(products),
        **totals,
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            postprocess_catalog(args.catalog, args.raw, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
