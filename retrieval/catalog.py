from __future__ import annotations

import json
import math
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
STYLE_KEYS = ("Style", "Fit Type", "Pattern", "Neck Style", "Sleeve Type", "Closure Type")
BRAND_KEYS = ("Brand", "Brand Name", "Manufacturer")
USE_CASE_KEYS = ("Sport", "Sport Type", "Occasion", "Theme")
GENERIC_CATEGORIES = {
    "clothing",
    "clothing shoes jewelry",
}
SHOE_CATEGORY_TERMS = {
    "boot", "boots", "cleat", "cleats", "footwear", "sandal", "sandals",
    "shoe", "shoes", "slipper", "slippers", "sneaker", "sneakers",
}
JEWELRY_CATEGORY_TERMS = {
    "anklet", "anklets", "bracelet", "bracelets", "brooch", "brooches",
    "charm", "charms", "earring", "earrings", "jewelry", "necklace",
    "necklaces", "pendant", "pendants", "ring", "rings",
}


def text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def coarse_category(values: list[str]) -> str:
    """Mirror the evaluator's category text exposed in initial messages."""
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def specific_categories(values: list[str]) -> list[str]:
    return [
        value
        for value in values
        if " ".join(terms(value)) not in GENERIC_CATEGORIES
    ]


def category_group(values: list[str]) -> str:
    """Map a full catalog category path to a question-prior group."""
    nodes = [normalized_text(value) for value in values[1:]]
    if "shoes" in nodes or "boot shop" in nodes:
        return "shoes"
    if "jewelry" in nodes or any("jewelry" in node for node in nodes):
        return "jewelry"
    return "clothing"


def normalize_price(value: object) -> tuple[float | None, bool]:
    """Return a usable price and whether it represents a lower bound."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number >= 0:
            return number, False
    if isinstance(value, str):
        match = re.fullmatch(r"from\s+([0-9]+(?:\.[0-9]+)?)", value.strip(), re.IGNORECASE)
        if match:
            return float(match.group(1)), True
    return None, False


def price_band(price: float | None) -> str | None:
    if price is None:
        return None
    if price < 15:
        return "under $15"
    if price < 25:
        return "$15-$25"
    if price < 50:
        return "$25-$50"
    if price < 100:
        return "$50-$100"
    return "over $100"


def normalize_department(value: object) -> str | None:
    if value in (None, ""):
        return None
    tokens = set(terms(str(value).replace("-", " ")))
    if "unisex" in tokens:
        if tokens & {"baby", "infant"}:
            return "unisex baby"
        if tokens & {"boy", "boys", "girl", "girls", "child", "children", "kid", "kids"}:
            return "unisex child"
        return "unisex adult"
    if tokens & {"women", "womens", "woman", "female", "ladies", "lady"}:
        return "womens"
    if tokens & {"men", "mens", "man", "male"}:
        return "mens"
    if tokens & {"girls", "girl"}:
        return "girls"
    if tokens & {"boys", "boy"}:
        return "boys"
    if tokens & {"baby", "infant"}:
        return "baby"
    normalized = normalized_text(value)
    return normalized or None


def extract_department(text: str) -> str | None:
    """Extract only explicit audience terms from customer-visible text."""
    return normalize_department(text) if set(terms(text)) & {
        "unisex", "baby", "infant", "women", "womens", "woman", "female",
        "ladies", "lady", "men", "mens", "man", "male", "girls", "girl",
        "boys", "boy", "child", "children", "kid", "kids",
    } else None


def first_detail(details: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = details.get(key)
        if value not in (None, "", [], {}):
            return normalized_text(value)
    return None


@dataclass(frozen=True)
class ProductRecord:
    parent_asin: str
    category_path: tuple[str, ...]
    coarse_category: str
    category_terms: frozenset[str]
    searchable_tokens: str
    average_rating: float
    rating_number: int
    price: float | None
    price_is_lower_bound: bool
    department: str | None
    material: str | None
    color: str | None
    style: str | None
    size: str | None
    brand: str | None
    use_case: str | None
    has_features: bool

    def attribute_value(self, attribute: str) -> str | None:
        if attribute == "material":
            return self.material
        if attribute == "color":
            return self.color
        if attribute == "style":
            return self.department or self.style
        if attribute == "size":
            return self.size
        if attribute == "brand":
            return self.brand
        if attribute == "budget":
            return price_band(self.price)
        if attribute == "use_case":
            return self.use_case
        if attribute == "feature" and self.has_features:
            return "present"
        return None


@dataclass(frozen=True)
class CandidatePool:
    parent_asins: tuple[str, ...]
    lexical_ranks: dict[str, int]


class CatalogIndex:
    """In-memory catalog records plus SQLite FTS candidate retrieval."""

    # TODO: Scale the lexical candidate limit with category size and compare it
    # with this fixed limit.
    LEXICAL_CANDIDATE_LIMIT = 1000

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.products: dict[str, ProductRecord] = {}
        self._coarse_category_index: dict[str, list[str]] = defaultdict(list)
        self._coarse_category_groups: dict[str, Counter[str]] = defaultdict(Counter)
        ratings: list[float] = []
        rating_numbers: list[int] = []
        self._build_index(ratings, rating_numbers)
        self.mean_rating = statistics.fmean(ratings)
        self.median_rating_number = statistics.median(rating_numbers)
        self.max_rating_number = max(rating_numbers)

    def _build_index(self, ratings: list[float], rating_numbers: list[int]) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                title = text_value(product.get("title"))
                categories = [str(value) for value in product.get("categories") or []]
                searchable_categories = specific_categories(categories)
                features = text_value(product.get("features"))
                details = product.get("details") if isinstance(product.get("details"), dict) else {}
                details_text = text_value(details)
                store = text_value(product.get("store"))
                description = text_value(product.get("description"))
                searchable = " ".join(
                    (
                        title,
                        features,
                        details_text,
                        description,
                        " ".join(searchable_categories),
                        store,
                    )
                )
                searchable_terms = terms(searchable)
                searchable_token_text = f" {' '.join(searchable_terms)} "
                material_match = MATERIAL_RE.search(searchable)
                color_match = COLOR_RE.search(searchable)
                category = coarse_category(categories)
                department = normalize_department(details.get("Department"))
                price, price_is_lower_bound = normalize_price(product.get("price"))
                average_rating = float(product["average_rating"])
                rating_number = int(product["rating_number"])
                record = ProductRecord(
                    parent_asin=parent_asin,
                    category_path=tuple(categories),
                    coarse_category=category,
                    category_terms=frozenset(terms(" ".join(searchable_categories))),
                    searchable_tokens=searchable_token_text,
                    average_rating=average_rating,
                    rating_number=rating_number,
                    price=price,
                    price_is_lower_bound=price_is_lower_bound,
                    department=department,
                    material=material_match.group(1).lower() if material_match else None,
                    color=color_match.group(1).lower() if color_match else None,
                    style=first_detail(details, STYLE_KEYS),
                    size=first_detail(details, ("Size",)),
                    brand=first_detail(details, BRAND_KEYS) or normalized_text(store) or None,
                    use_case=first_detail(details, USE_CASE_KEYS),
                    has_features=bool(product.get("features")),
                )
                self.products[parent_asin] = record
                normalized_category = normalized_text(category)
                self._coarse_category_index[normalized_category].append(parent_asin)
                self._coarse_category_groups[normalized_category][
                    category_group(categories)
                ] += 1
                ratings.append(average_rating)
                rating_numbers.append(rating_number)
                batch.append(
                    (
                        parent_asin,
                        title,
                        " ".join(searchable_categories),
                        features,
                        details_text,
                        store,
                        description,
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def candidates(self, category: str, query_text: str) -> CandidatePool:
        query_terms = list(dict.fromkeys(terms(query_text)))[:40]
        lexical_ranks: dict[str, int] = {}
        if query_terms:
            expression = " OR ".join(f'"{term}"' for term in query_terms)
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, self.LEXICAL_CANDIDATE_LIMIT),
            ).fetchall()
            lexical_ranks = {str(row[0]): rank for rank, row in enumerate(rows, start=1)}

        category_matches = self._coarse_category_index.get(normalized_text(category), [])
        combined = list(dict.fromkeys([*category_matches, *lexical_ranks]))
        return CandidatePool(tuple(combined), lexical_ranks)

    def question_category(self, category: str) -> str:
        catalog_groups = self._coarse_category_groups.get(normalized_text(category))
        if catalog_groups:
            return catalog_groups.most_common(1)[0][0]
        category_terms = set(terms(category))
        if category_terms & SHOE_CATEGORY_TERMS:
            return "shoes"
        if category_terms & JEWELRY_CATEGORY_TERMS:
            return "jewelry"
        return "clothing"

    def close(self) -> None:
        self.connection.close()
