"""Rank reproducible direct competitors for a seller product."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file


DEFAULT_EXCLUDED_PHRASES = (
    "baby",
    "child",
    "children",
    "kid",
    "kids",
    "toddler",
    "beard",
    "conditioner bar",
    "hair mask",
    "hair grease",
    "shampoo",
    "bundle",
    "pack of",
    "two pack",
    "three pack",
    "2 pack",
    "3 pack",
    "4 pack",
    "5 pack",
    "6 pack",
    "2 count",
    "3 count",
    "4 count",
    "5 count",
    "6 count",
)

CONCEPT_PHRASES = {
    "shampoo": ("shampoo", "cleanser", "cleansing"),
    "conditioner": ("conditioner", "conditioning"),
    "two_in_one": ("2 in 1", "2in1", "two in one"),
    "three_in_one": ("3 in 1", "3in1", "three in one"),
    "leave_in": ("leave in",),
    "spray": ("spray", "mist"),
    "detangling": ("detangle", "detangler", "detangling"),
    "frizz_control": (
        "anti frizz",
        "frizz control",
        "frizzy",
        "flyaway",
        "flyaways",
        "smooth",
        "smooths",
    ),
    "damage_breakage": (
        "breakage",
        "anti breakage",
        "split end",
        "split ends",
        "damaged hair",
        "repair",
        "repairs",
        "repairing",
    ),
    "growth_strength": (
        "hair growth",
        "grow long",
        "strengthen",
        "strengthens",
        "strengthening",
    ),
    "dry_hair": (
        "dry hair",
        "moisture",
        "moisturizing",
        "hydrate",
        "hydrates",
        "hydrating",
    ),
    "curls": ("curl", "curls", "curly", "wavy", "waves"),
    "heat_protection": (
        "heat protectant",
        "heat protection",
        "thermal protection",
    ),
}


def _product_format(text: str) -> str:
    """Identify title-level formats that should not compete with each other."""
    if any(_contains_phrase(text, phrase) for phrase in ("shampoo bar", "solid shampoo")):
        return "bar"
    if any(
        _contains_phrase(text, phrase)
        for phrase in (
            "hair color",
            "color depositing",
            "colour depositing",
            "hair dye",
            "semi permanent",
        )
    ):
        return "hair_color"
    if any(
        _contains_phrase(text, phrase)
        for phrase in ("set", "bundle", "kit", "pack of", "shampoo and conditioner pack")
    ):
        return "set"
    if _is_leave_in(text):
        return "leave_in"
    return "standard"


def _audience(text: str) -> str:
    return "child" if any(
        _contains_phrase(text, phrase)
        for phrase in ("baby", "child", "children", "kid", "kids", "toddler")
    ) else "general"


def _same_brand(left: str, right: str) -> bool:
    """Match exact brands and branded sub-lines such as Dove Men+Care."""
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter + " ")


@dataclass(frozen=True)
class CompetitorSelectionReport:
    selection_version: str
    seller_parent_asin: str
    seller_title: str
    category_path_ids: list[str]
    catalog_candidate_count: int
    eligible_candidate_count: int
    selected_competitor_count: int
    selected_parent_asins: list[str]
    minimum_review_count: int
    selection_profile: str
    excluded_phrases: list[str]
    ranking_path: str
    ranking_sha256: str
    warning: str


def _artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = _normalize_text(phrase)
    if not normalized_phrase:
        return False
    return bool(re.search(rf"\b{re.escape(normalized_phrase)}\b", text))


def _concepts(text: str) -> set[str]:
    return {
        concept
        for concept, phrases in CONCEPT_PHRASES.items()
        if any(_contains_phrase(text, phrase) for phrase in phrases)
    }


def _is_leave_in(text: str) -> bool:
    return any(
        _contains_phrase(text, phrase)
        for phrase in ("leave in", "leave on", "no rinse")
    )


def _price_similarity(seller_price: Any, candidate_price: Any) -> float:
    if pd.isna(seller_price) or pd.isna(candidate_price):
        return 0.5
    left = max(float(seller_price), 0.01)
    right = max(float(candidate_price), 0.01)
    return max(0.0, 1.0 - abs(math.log(right / left)) / math.log(4.0))


def rank_direct_competitors(
    catalog: pd.DataFrame,
    *,
    seller_parent_asin: str,
    category_path_ids: list[str],
    top_k: int = 4,
    minimum_review_count: int = 100,
    excluded_phrases: tuple[str, ...] = DEFAULT_EXCLUDED_PHRASES,
    selection_profile: Literal["conditioner", "catalog_path"] = "conditioner",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return eligible candidates ranked by transparent product similarity."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if minimum_review_count < 1:
        raise ValueError("minimum_review_count must be positive")
    required = {
        "parent_asin",
        "category_path_id",
        "product_title",
        "product_search_text",
        "store",
        "price_at_collection",
        "average_rating",
        "rating_number",
        "review_count",
    }
    missing_columns = required - set(catalog.columns)
    if missing_columns:
        raise ValueError(f"catalog is missing columns: {sorted(missing_columns)}")
    if not category_path_ids:
        raise ValueError("category_path_ids must not be empty")

    seller_rows = catalog[catalog["parent_asin"].astype(str) == seller_parent_asin]
    if len(seller_rows) != 1:
        raise ValueError("seller_parent_asin must resolve to exactly one catalog row")
    seller = seller_rows.iloc[0]
    seller_title = _normalize_text(seller["product_title"])
    seller_brand = _normalize_text(seller["store"])
    if selection_profile not in {"conditioner", "catalog_path"}:
        raise ValueError("unknown competitor selection profile")
    seller_excluded_phrase = ""
    if selection_profile == "conditioner":
        seller_excluded_phrase = next(
            (
                phrase
                for phrase in excluded_phrases
                if _contains_phrase(seller_title, phrase)
            ),
            "",
        )
        if seller_excluded_phrase:
            raise ValueError(
                "analysis supports adult single-product conditioners only; "
                f"the product matched excluded phrase: {seller_excluded_phrase}"
            )
    seller_is_leave_in = _is_leave_in(seller_title)
    seller_format = _product_format(seller_title)
    seller_audience = _audience(seller_title)
    if (
        selection_profile == "conditioner"
        and not seller_is_leave_in
        and not _contains_phrase(seller_title, "conditioner")
    ):
        raise ValueError(
            "analysis supports leave-in and rinse-out conditioner products only"
        )

    scope = catalog[catalog["category_path_id"].isin(category_path_ids)].copy()
    scope["normalized_title"] = scope["product_title"].map(_normalize_text)
    scope["normalized_brand"] = scope["store"].map(_normalize_text)
    scope["is_seller"] = scope["parent_asin"].astype(str) == seller_parent_asin
    scope["is_leave_in"] = scope["normalized_title"].map(_is_leave_in)
    scope["is_conditioner"] = scope["normalized_title"].map(
        lambda value: _contains_phrase(value, "conditioner")
    )
    scope["same_conditioner_format"] = (
        scope["is_leave_in"]
        if seller_is_leave_in
        else (~scope["is_leave_in"] & scope["is_conditioner"])
    )
    scope["product_format"] = scope["normalized_title"].map(_product_format)
    scope["audience"] = scope["normalized_title"].map(_audience)
    scope["same_brand"] = scope["normalized_brand"].map(
        lambda value: _same_brand(seller_brand, value)
    )
    scope["excluded_phrase"] = (
        scope["normalized_title"].map(
            lambda value: next(
                (
                    phrase
                    for phrase in excluded_phrases
                    if _contains_phrase(value, phrase)
                ),
                "",
            )
        )
        if selection_profile == "conditioner"
        else ""
    )
    scope["enough_reviews"] = scope["review_count"].fillna(0).astype(int) >= int(
        minimum_review_count
    )
    same_format = (
        scope["same_conditioner_format"]
        if selection_profile == "conditioner"
        else scope["product_format"].eq(seller_format)
    )
    eligible_mask = (
        ~scope["is_seller"]
        & same_format
        & scope["audience"].eq(seller_audience)
        & ~scope["same_brand"]
        & (scope["excluded_phrase"] == "")
        & scope["enough_reviews"]
    )
    eligible = scope[eligible_mask].copy()
    counts = {
        "catalog_candidate_count": int(len(scope)),
        "eligible_candidate_count": int(len(eligible)),
    }
    if len(eligible) < top_k:
        raise ValueError(
            f"only {len(eligible)} eligible competitors for requested top_k={top_k}"
        )

    # Titles carry the comparable offer and use case. Long descriptions often
    # mention many generic hair benefits and make unrelated products look close.
    seller_search = seller_title
    candidate_search = eligible["normalized_title"].tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=25_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([seller_search, *candidate_search])
    eligible["text_similarity"] = cosine_similarity(matrix[0:1], matrix[1:]).ravel()

    seller_concepts = _concepts(seller_search)
    eligible["matched_concepts"] = eligible["normalized_title"].map(
        lambda value: sorted(seller_concepts & _concepts(_normalize_text(value)))
    )
    eligible["concept_similarity"] = eligible["matched_concepts"].map(
        lambda values: len(values) / max(1, len(seller_concepts))
    )
    eligible["price_similarity"] = eligible["price_at_collection"].map(
        lambda value: _price_similarity(seller["price_at_collection"], value)
    )
    maximum_log_reviews = math.log1p(float(eligible["review_count"].max()))
    eligible["review_support_score"] = eligible["review_count"].map(
        lambda value: math.log1p(float(value)) / maximum_log_reviews
    )
    eligible["direct_competitor_score"] = (
        0.45 * eligible["text_similarity"]
        + 0.30 * eligible["concept_similarity"]
        + 0.10 * eligible["price_similarity"]
        + 0.15 * eligible["review_support_score"]
    )
    eligible = eligible.sort_values(
        ["direct_competitor_score", "review_count", "parent_asin"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    eligible["rank"] = np.arange(1, len(eligible) + 1)
    # Prefer a benchmark with different brands so one manufacturer cannot
    # dominate the comparison. If the niche has too few distinct brands, fill
    # the remaining slots by score.
    selected_indexes: list[int] = []
    selected_brands: set[str] = set()
    for index, row in eligible.iterrows():
        brand_key = str(row["normalized_brand"] or "").strip()
        if brand_key and brand_key in selected_brands:
            continue
        selected_indexes.append(int(index))
        if brand_key:
            selected_brands.add(brand_key)
        if len(selected_indexes) == top_k:
            break
    if len(selected_indexes) < top_k:
        selected_indexes.extend(
            int(index)
            for index in eligible.index
            if int(index) not in selected_indexes
        )
        selected_indexes = selected_indexes[:top_k]
    eligible["selected"] = eligible.index.isin(selected_indexes)
    eligible["matched_concepts_json"] = eligible["matched_concepts"].map(
        lambda values: json.dumps(values, ensure_ascii=False)
    )
    return eligible[
        [
            "rank",
            "selected",
            "parent_asin",
            "store",
            "product_title",
            "price_at_collection",
            "average_rating",
            "rating_number",
            "review_count",
            "direct_competitor_score",
            "text_similarity",
            "concept_similarity",
            "price_similarity",
            "review_support_score",
            "matched_concepts_json",
        ]
    ], counts


def select_direct_competitors(
    catalog_path: str | Path,
    ranking_path: str | Path,
    *,
    seller_parent_asin: str,
    category_path_ids: list[str],
    selection_version: str,
    top_k: int = 4,
    minimum_review_count: int = 100,
    selection_profile: Literal["conditioner", "catalog_path"] = "conditioner",
    report_path: str | Path | None = None,
) -> CompetitorSelectionReport:
    """Rank, persist, and identify the direct competitor set."""
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit='256MB'")
        connection.execute("SET threads=2")
        catalog = connection.execute(
            """
            SELECT parent_asin, category_path_id, product_title,
                   product_search_text, store, price_at_collection,
                   average_rating, rating_number, review_count
            FROM read_parquet(?)
            WHERE category_path_id IN (SELECT UNNEST(?))
            """,
            [str(catalog_path), category_path_ids],
        ).fetchdf()
    finally:
        connection.close()
    ranking, counts = rank_direct_competitors(
        catalog,
        seller_parent_asin=seller_parent_asin,
        category_path_ids=category_path_ids,
        top_k=top_k,
        minimum_review_count=minimum_review_count,
        selection_profile=selection_profile,
    )
    destination = Path(ranking_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        ranking.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    seller_row = catalog[catalog["parent_asin"].astype(str) == seller_parent_asin].iloc[0]
    selected = ranking[ranking["selected"]]
    report = CompetitorSelectionReport(
        selection_version=selection_version,
        seller_parent_asin=seller_parent_asin,
        seller_title=str(seller_row["product_title"]),
        category_path_ids=category_path_ids,
        catalog_candidate_count=counts["catalog_candidate_count"],
        eligible_candidate_count=counts["eligible_candidate_count"],
        selected_competitor_count=int(len(selected)),
        selected_parent_asins=selected["parent_asin"].astype(str).tolist(),
        minimum_review_count=minimum_review_count,
        selection_profile=selection_profile,
        excluded_phrases=(
            list(DEFAULT_EXCLUDED_PHRASES)
            if selection_profile == "conditioner"
            else []
        ),
        ranking_path=_artifact_reference(destination),
        ranking_sha256=sha256_file(destination),
        warning=(
            "Historical catalog ranking for a demo workspace; direct competitors "
            "must still be displayed with product titles and selection provenance."
        ),
    )
    if report_path is not None:
        Path(report_path).write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog_path", type=Path)
    parser.add_argument("ranking_path", type=Path)
    parser.add_argument("--seller-parent-asin", required=True)
    parser.add_argument("--category-path-id", action="append", required=True)
    parser.add_argument("--selection-version", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--minimum-review-count", type=int, default=100)
    parser.add_argument(
        "--selection-profile",
        choices=("conditioner", "catalog_path"),
        default="conditioner",
    )
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    report = select_direct_competitors(
        args.catalog_path,
        args.ranking_path,
        seller_parent_asin=args.seller_parent_asin,
        category_path_ids=args.category_path_id,
        selection_version=args.selection_version,
        top_k=args.top_k,
        minimum_review_count=args.minimum_review_count,
        selection_profile=args.selection_profile,
        report_path=args.report_path,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
