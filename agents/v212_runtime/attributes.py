"""Deterministic, target-blind attribute views for shortlist reranking.

This module consumes catalog fields and visible conversation state only.  It has
no evaluator, public-set, network, profile, or ASIN-specific dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "p2.attributes.v1"

ALIASES = {
    "grey": "gray",
    "mens": "men",
    "womens": "women",
    "ladies": "women",
    "pullon": "pull on",
    "pull ons": "pull on",
    "slipon": "slip on",
    "slip ons": "slip on",
    "t shirt": "tshirt",
    "tee shirt": "tshirt",
}
_POSSESSIVE_RE = re.compile(r"\b(men|women|boy|girl)'s\b")
_SEPARATOR_RE = re.compile(r"[-_/\u2010-\u2015\u2212]+")
_NON_VALUE_RE = re.compile(r"[^a-z0-9.$% ]+")
_SPACE_RE = re.compile(r"\s+")
_ALIAS_PATTERNS = tuple(
    (re.compile(rf"\b{re.escape(old)}\b"), new)
    for old, new in sorted(ALIASES.items(), key=lambda item: (-len(item[0]), item[0]))
)


def _identity(*values: str) -> dict[str, str]:
    return {value: value for value in values}


CATEGORIES = {
    **_identity(
        "bag", "belt", "blazer", "boot", "bra", "coat", "dress",
        "earring", "glove", "handbag", "hat", "hoodie", "jacket",
        "jeans", "jewelry", "jumpsuit", "leggings", "necklace", "pants",
        "ring", "sandal", "shirt", "shoe", "shorts", "skirt", "sneaker",
        "sock", "suit", "sweater", "swimwear", "tshirt", "top",
        "underwear", "wallet", "watch", "costume", "robe",
    ),
    "accessories": "accessory",
    "bags": "bag",
    "belts": "belt",
    "blazers": "blazer",
    "boots": "boot",
    "bras": "bra",
    "coats": "coat",
    "dresses": "dress",
    "earrings": "earring",
    "gloves": "glove",
    "handbags": "handbag",
    "hats": "hat",
    "hoodies": "hoodie",
    "jackets": "jacket",
    "jumpsuits": "jumpsuit",
    "necklaces": "necklace",
    "rings": "ring",
    "sandals": "sandal",
    "shirts": "shirt",
    "shoes": "shoe",
    "skirts": "skirt",
    "sneakers": "sneaker",
    "socks": "sock",
    "suits": "suit",
    "sweaters": "sweater",
    "tops": "top",
    "wallets": "wallet",
    "watches": "watch",
    "costumes": "costume",
    "robes": "robe",
}
AUDIENCE = {
    "men": "men",
    "male": "men",
    "women": "women",
    "female": "women",
    "boy": "boys",
    "boys": "boys",
    "girl": "girls",
    "girls": "girls",
    "kid": "kids",
    "kids": "kids",
    "child": "kids",
    "children": "kids",
    "unisex": "unisex",
    "baby": "baby",
}
MATERIALS = _identity(
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "linen", "denim", "fleece", "mesh", "suede", "canvas",
    "rubber", "satin", "velvet", "acrylic", "viscose", "cashmere",
    "lycra", "elastane", "stainless steel",
)
COLORS = _identity(
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "purple", "yellow", "orange", "beige", "navy", "gold", "silver",
    "tan", "khaki", "burgundy", "maroon", "teal", "ivory", "cream",
    "multicolor",
)
CLOSURES = {
    "zip": "zipper",
    "zipper": "zipper",
    "button": "button",
    "buttons": "button",
    "lace up": "lace up",
    "laces": "lace up",
    "pull on": "pull on",
    "slip on": "slip on",
    "hook and loop": "hook and loop",
    "velcro": "hook and loop",
    "drawstring": "drawstring",
    "buckle": "buckle",
    "clasp": "clasp",
    "snap": "snap",
}
STYLES = _identity(
    "casual", "formal", "vintage", "classic", "modern", "sporty", "athletic",
    "slim", "relaxed", "oversized", "elegant", "novelty", "compression",
    "thermal", "regular fit", "slim fit", "loose fit", "floral", "striped",
    "plaid", "solid",
)
USE_CASES = _identity(
    "hiking", "running", "gym", "winter", "outdoor", "work", "walking",
    "workout", "office", "wedding", "travel", "school", "beach", "rain",
    "snow", "cycling", "swimming", "yoga", "dance", "costume",
)
SIZES = _identity(
    "xxs", "xs", "small", "medium", "large", "xl", "xxl", "xxxl",
    "petite", "tall", "plus size", "one size",
)
WIDTHS = {
    "extra wide": "extra wide",
    "wide": "wide",
    "narrow": "narrow",
    "regular width": "regular",
}
FEATURES = {
    **_identity(
        "waterproof", "water resistant", "breathable", "lightweight",
        "machine wash", "moisture wicking", "quick dry", "hypoallergenic",
        "uv protection", "fleece lined", "stretch", "non slip", "cushioned",
        "arch support", "short sleeve", "long sleeve", "sleeveless",
        "crew neck", "v neck", "shawl collar", "hooded", "adjustable",
        "reversible", "insulated", "wrinkle resistant",
    ),
    "pocket": "pocket",
    "pockets": "pocket",
}

SLOT_VOCABULARIES: dict[str, dict[str, str]] = {
    "audience": AUDIENCE,
    "material": MATERIALS,
    "color": COLORS,
    "closure": CLOSURES,
    "style": STYLES,
    "use_case": USE_CASES,
    "size": SIZES,
    "width": WIDTHS,
    "feature": FEATURES,
}
_ORDERED_VOCABULARIES = {
    id(vocabulary): tuple(
        sorted(vocabulary.items(), key=lambda item: (-len(item[0]), item[0]))
    )
    for vocabulary in (*SLOT_VOCABULARIES.values(), CATEGORIES)
}
SOURCE_CONFIDENCE = {
    "categories": 1.0,
    "details": 0.98,
    "store": 1.0,
    "features": 0.90,
    "title": 0.82,
}
NOISE_VALUES = {
    "all", "other", "heather", "heathers", "imported", "colors", "color",
    "made", "usa", "available", "fabric", "and", "or",
}
GENERIC_CATEGORIES = {
    "clothing", "clothing shoes and jewelry", "clothing shoes jewelry",
    "shoes and jewelry", "shoes jewelry", "department", "men", "women",
    "boys", "girls", "kids", "baby", "unisex",
}
CLASSIFICATION_SLOT_ALIASES = {
    "budget": "price",
    "feature_phrases": "feature",
}


@dataclass(frozen=True, slots=True)
class AttributeValue:
    value: str
    source: str
    confidence: float
    raw: str


@dataclass(frozen=True, slots=True)
class ProductAttributeView:
    parent_asin: str
    category: tuple[AttributeValue, ...] = ()
    audience: tuple[AttributeValue, ...] = ()
    material: tuple[AttributeValue, ...] = ()
    color: tuple[AttributeValue, ...] = ()
    closure: tuple[AttributeValue, ...] = ()
    style: tuple[AttributeValue, ...] = ()
    use_case: tuple[AttributeValue, ...] = ()
    size: tuple[AttributeValue, ...] = ()
    width: tuple[AttributeValue, ...] = ()
    brand: tuple[AttributeValue, ...] = ()
    price: float | None = None
    feature_phrases: tuple[AttributeValue, ...] = ()


@dataclass(frozen=True, slots=True)
class ConstraintValue:
    slot: str
    value: str
    polarity: int
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class ConversationConstraintView:
    category_terms: tuple[str, ...] = ()
    positive: tuple[ConstraintValue, ...] = ()
    negative: tuple[ConstraintValue, ...] = ()
    exact_terms: tuple[str, ...] = ()
    excluded_exact_terms: tuple[str, ...] = ()
    classified_slots: tuple[str, ...] = ()


def normalize_value(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("&", " and ")
    text = _POSSESSIVE_RE.sub(r"\1", text)
    text = _SEPARATOR_RE.sub(" ", text)
    text = _NON_VALUE_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    for pattern, replacement in _ALIAS_PATTERNS:
        text = pattern.sub(replacement, text)
    return _SPACE_RE.sub(" ", text).strip()


def _flatten(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, Mapping):
        flattened: list[str] = []
        for key in sorted(value, key=lambda item: normalize_value(item)):
            for item in _flatten(value[key]):
                flattened.append(f"{key}: {item}")
        return flattened
    if isinstance(value, (list, tuple)):
        return [item for value_item in value for item in _flatten(value_item)]
    if isinstance(value, (set, frozenset)):
        return sorted((str(item) for item in value), key=normalize_value)
    return [str(value)]


def _ordered_items(vocabulary: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    cached = _ORDERED_VOCABULARIES.get(id(vocabulary))
    if cached is not None:
        return cached
    return tuple(sorted(vocabulary.items(), key=lambda item: (-len(item[0]), item[0])))


def _extract_vocab_normalized(
    normalized: str,
    vocabulary: Mapping[str, str],
) -> list[str]:
    padded = f" {normalized} "
    return sorted({
        canonical
        for phrase, canonical in _ordered_items(vocabulary)
        if f" {phrase} " in padded
    })


def _extract_vocab(text: object, vocabulary: Mapping[str, str]) -> list[str]:
    return _extract_vocab_normalized(normalize_value(text), vocabulary)


def _attribute(value: str, source: str, confidence: float, raw: object) -> AttributeValue:
    return AttributeValue(
        value=normalize_value(value),
        source=source,
        confidence=round(float(confidence), 3),
        raw=str(raw)[:240],
    )


def _dedupe(values: Iterable[AttributeValue]) -> tuple[AttributeValue, ...]:
    selected: dict[str, AttributeValue] = {}
    for item in values:
        if not item.value or item.value in NOISE_VALUES or item.value.isdigit():
            continue
        previous = selected.get(item.value)
        item_rank = (item.confidence, "." in item.source, item.source, item.raw)
        previous_rank = (
            (previous.confidence, "." in previous.source, previous.source, previous.raw)
            if previous is not None
            else None
        )
        if previous_rank is None or item_rank > previous_rank:
            selected[item.value] = item
    return tuple(selected[value] for value in sorted(selected))


def _category_values(raw: object) -> list[str]:
    normalized = normalize_value(raw)
    if not normalized or normalized in GENERIC_CATEGORIES:
        return []
    extracted = _extract_vocab_normalized(normalized, CATEGORIES)
    if extracted:
        return extracted
    tokens = normalized.split()
    if len(tokens) <= 4 and not _extract_vocab_normalized(normalized, AUDIENCE):
        return [normalized]
    return []


def _detail_slot(key: object) -> str | None:
    normalized = normalize_value(key)
    if any(marker in normalized for marker in ("department", "gender", "audience")):
        return "audience"
    if "material" in normalized or "fabric" in normalized:
        return "material"
    if "color" in normalized:
        return "color"
    if "closure" in normalized:
        return "closure"
    if "width" in normalized:
        return "width"
    if "size" in normalized:
        return "size"
    if any(marker in normalized for marker in ("style", "pattern", "fit")):
        return "style"
    if any(marker in normalized for marker in ("brand", "manufacturer")):
        return "brand"
    return None


def _structured_details(details: object) -> dict[str, list[AttributeValue]]:
    result = {slot: [] for slot in (*SLOT_VOCABULARIES, "brand")}
    if not isinstance(details, Mapping):
        return result
    for key in sorted(details, key=lambda item: normalize_value(item)):
        slot = _detail_slot(key)
        if slot is None:
            continue
        source = f"details.{normalize_value(key) or 'value'}"
        for raw_value in _flatten(details[key]):
            raw = f"{key}: {raw_value}"
            if slot == "brand":
                brand = normalize_value(raw_value)
                if brand:
                    result[slot].append(
                        _attribute(brand, source, SOURCE_CONFIDENCE["details"], raw)
                    )
                continue
            for value in _extract_vocab(raw_value, SLOT_VOCABULARIES[slot]):
                result[slot].append(
                    _attribute(value, source, SOURCE_CONFIDENCE["details"], raw)
                )
    return result


def _price(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
        if match is None:
            return None
        number = float(match.group())
    return round(number, 2) if math.isfinite(number) and number >= 0 else None


def build_product_attribute_view(product: Mapping[str, Any]) -> ProductAttributeView:
    slots: dict[str, list[AttributeValue]] = {
        slot: [] for slot in (*SLOT_VOCABULARIES, "category", "brand")
    }
    categories = _flatten(product.get("categories"))
    for raw in categories:
        for value in _category_values(raw):
            slots["category"].append(_attribute(value, "categories", 1.0, raw))

    structured = _structured_details(product.get("details"))
    for slot, values in structured.items():
        slots[slot].extend(values)

    for raw in _flatten(product.get("store")):
        brand = normalize_value(raw)
        if brand:
            slots["brand"].append(_attribute(brand, "store", 1.0, raw))

    sources = (
        ("categories", categories),
        ("title", _flatten(product.get("title"))),
        ("features", _flatten(product.get("features"))),
    )
    details = product.get("details")
    if details and not isinstance(details, Mapping):
        sources = (*sources, ("details", _flatten(details)))
    for source, raw_values in sources:
        confidence = SOURCE_CONFIDENCE[source]
        for raw in raw_values:
            normalized = normalize_value(raw)
            if (
                source in {"categories", "title"}
                and normalized not in GENERIC_CATEGORIES
            ):
                for value in _extract_vocab_normalized(normalized, CATEGORIES):
                    slots["category"].append(_attribute(value, source, confidence, raw))
            for slot, vocabulary in SLOT_VOCABULARIES.items():
                for value in _extract_vocab_normalized(normalized, vocabulary):
                    slots[slot].append(_attribute(value, source, confidence, raw))

    return ProductAttributeView(
        parent_asin=str(product.get("parent_asin") or ""),
        category=_dedupe(slots["category"]),
        audience=_dedupe(slots["audience"]),
        material=_dedupe(slots["material"]),
        color=_dedupe(slots["color"]),
        closure=_dedupe(slots["closure"]),
        style=_dedupe(slots["style"]),
        use_case=_dedupe(slots["use_case"]),
        size=_dedupe(slots["size"]),
        width=_dedupe(slots["width"]),
        brand=_dedupe(slots["brand"]),
        price=_price(product.get("price")),
        feature_phrases=_dedupe(slots["feature"]),
    )


def _constraints_from_text(text: object, polarity: int, source: str) -> list[ConstraintValue]:
    constraints: list[ConstraintValue] = []
    normalized = normalize_value(text)
    for slot, vocabulary in SLOT_VOCABULARIES.items():
        for value in _extract_vocab_normalized(normalized, vocabulary):
            constraints.append(ConstraintValue(slot, value, polarity, 1.0, source))
    return constraints


def _matched_vocabulary_tokens(text: str) -> set[str]:
    matched: set[str] = set()
    padded = f" {text} "
    for vocabulary in (*SLOT_VOCABULARIES.values(), CATEGORIES):
        for phrase, _canonical in _ordered_items(vocabulary):
            if f" {phrase} " in padded:
                matched.update(phrase.split())
    return matched


def _is_meaningful_exact(value: str) -> bool:
    return bool(value and value not in NOISE_VALUES and not value.isdigit())


def _classification_items(
    classifications: Mapping[str, object] | Iterable[str] | None,
) -> tuple[list[ConstraintValue], set[str]]:
    def canonical_slot(value: object) -> str:
        normalized = normalize_value(value).replace(" ", "_")
        return CLASSIFICATION_SLOT_ALIASES.get(normalized, normalized)

    if classifications is None:
        return [], set()
    if not isinstance(classifications, Mapping):
        raw_slots = [classifications] if isinstance(classifications, str) else classifications
        return [], {
            canonical_slot(slot)
            for slot in raw_slots
            if canonical_slot(slot)
        }
    constraints: list[ConstraintValue] = []
    slots: set[str] = set()
    for raw_slot in sorted(classifications, key=lambda item: normalize_value(item)):
        slot = canonical_slot(raw_slot)
        if not slot:
            continue
        slots.add(slot)
        for raw in _flatten(classifications[raw_slot]):
            normalized = normalize_value(raw)
            if not _is_meaningful_exact(normalized):
                continue
            vocabulary = CATEGORIES if slot == "category" else SLOT_VOCABULARIES.get(slot)
            values = _extract_vocab(normalized, vocabulary) if vocabulary else [normalized]
            for value in values:
                constraints.append(ConstraintValue(slot, value, 1, 1.0, "classification"))
    return constraints, slots


def build_conversation_constraint_view(
    category_text: str,
    active_terms: Iterable[str],
    excluded_terms: Iterable[str],
    classifications: Mapping[str, object] | Iterable[str] | None = None,
) -> ConversationConstraintView:
    category_normalized = normalize_value(category_text)
    category_terms = set(_extract_vocab(category_normalized, CATEGORIES))
    if not category_terms and category_normalized not in GENERIC_CATEGORIES:
        consumed = _matched_vocabulary_tokens(category_normalized)
        category_terms.update(
            token
            for token in category_normalized.split()
            if len(token) > 1 and token not in consumed and token not in NOISE_VALUES
        )

    positive = _constraints_from_text(category_normalized, 1, "category_text")
    active_raw = [str(term) for term in active_terms]
    active_normalized = normalize_value(" ".join(active_raw))
    positive.extend(_constraints_from_text(active_normalized, 1, "active_terms"))

    negative_raw = [str(term) for term in excluded_terms]
    negative_normalized = normalize_value(" ".join(negative_raw))
    negative = _constraints_from_text(negative_normalized, -1, "excluded_terms")

    classified_constraints, classified_slots = _classification_items(classifications)
    positive.extend(classified_constraints)

    active_consumed = _matched_vocabulary_tokens(active_normalized)
    exact = {
        normalized
        for raw in active_raw
        if (normalized := normalize_value(raw))
        and _is_meaningful_exact(normalized)
        and not set(normalized.split()) <= active_consumed
    }
    negative_consumed = _matched_vocabulary_tokens(negative_normalized)
    excluded_exact = {
        normalized
        for raw in negative_raw
        if (normalized := normalize_value(raw))
        and _is_meaningful_exact(normalized)
        and not set(normalized.split()) <= negative_consumed
    }

    def unique_constraints(values: Iterable[ConstraintValue]) -> tuple[ConstraintValue, ...]:
        selected = {
            (item.slot, item.value, item.polarity): item
            for item in values
            if _is_meaningful_exact(item.value)
        }
        return tuple(selected[key] for key in sorted(selected))

    positive_view = unique_constraints(positive)
    negative_view = unique_constraints(negative)
    classified_slots.update(item.slot for item in (*positive_view, *negative_view))
    return ConversationConstraintView(
        category_terms=tuple(sorted(category_terms)),
        positive=positive_view,
        negative=negative_view,
        exact_terms=tuple(sorted(exact)),
        excluded_exact_terms=tuple(sorted(excluded_exact)),
        classified_slots=tuple(sorted(classified_slots)),
    )


def product_slot(view: ProductAttributeView, slot: str) -> tuple[AttributeValue, ...]:
    if slot in {"feature", "feature_phrases"}:
        return view.feature_phrases
    value = getattr(view, slot, ())
    return value if isinstance(value, tuple) else ()


def attribute_registry_sha256() -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "aliases": ALIASES,
        "categories": CATEGORIES,
        "slot_vocabularies": SLOT_VOCABULARIES,
        "source_confidence": SOURCE_CONFIDENCE,
        "noise_values": sorted(NOISE_VALUES),
        "generic_categories": sorted(GENERIC_CATEGORIES),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def product_view_as_dict(view: ProductAttributeView) -> dict[str, Any]:
    return asdict(view)
