"""Deterministic product rankings for seller-versus-competitor questions."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


COMPARISON_ASPECT_IDS = (
    "moisture_and_hydration",
    "softness",
    "frizz_control",
    "detangling",
    "greasy_oily_finish",
    "hair_weight",
    "price_affordability",
)

_REPURCHASE_PATTERN = re.compile(
    r"\b(?:would|will|definitely|certainly|probably)?\s*"
    r"(?:buy|purchase|order|get)(?:\s+(?:it|this|these|the product))?\s+again\b"
    r"|\b(?:repurchase|re-purchase|reorder|re-order)(?:d|ing)?\b"
    r"|\bkeep(?:s|ing)?\s+(?:buying|purchasing|ordering)\b",
    flags=re.IGNORECASE,
)
_NEGATED_REPURCHASE_PATTERN = re.compile(
    r"\b(?:not|never|won't|wouldn't|will not|would not|don't|do not)\b"
    r".{0,35}\b(?:buy|purchase|order|get|repurchase|reorder)\b",
    flags=re.IGNORECASE,
)


def _rank_row(row: Any, sentiment: str) -> dict[str, Any]:
    count = int(getattr(row, f"{sentiment}_count"))
    share = float(getattr(row, f"{sentiment}_share"))
    mentions = int(row.aspect_review_count)
    return {
        "parent_asin": str(row.parent_asin),
        "store": None if pd.isna(row.store) else str(row.store),
        "product_title": str(row.product_title),
        "product_review_count": int(row.product_review_count),
        "average_rating": float(row.product_average_rating),
        "mention_count": mentions,
        "sentiment_count": count,
        "sentiment_share": share,
        "support_sufficient": mentions >= 5 and count >= 2,
    }


def _aspect_rankings(
    product_mart: pd.DataFrame,
    overview: pd.DataFrame,
    *,
    aspect_names: dict[str, str],
    limit: int,
) -> list[dict[str, Any]]:
    selected = product_mart[
        product_mart["aspect_id"].isin(COMPARISON_ASPECT_IDS)
    ].copy()
    if "product_average_rating" not in selected.columns:
        selected = selected.merge(
            overview[["parent_asin", "average_rating"]].rename(
                columns={"average_rating": "product_average_rating"}
            ),
            on="parent_asin",
            how="inner",
            validate="many_to_one",
        )
    output: list[dict[str, Any]] = []
    for aspect_id in COMPARISON_ASPECT_IDS:
        rows = selected[selected["aspect_id"] == aspect_id]
        rankings: dict[str, list[dict[str, Any]]] = {}
        for sentiment in ("positive", "negative"):
            ranked = rows[rows[f"{sentiment}_count"] > 0].sort_values(
                [f"{sentiment}_count", f"{sentiment}_share", "aspect_review_count"],
                ascending=False,
            )
            rankings[sentiment] = [
                _rank_row(row, sentiment)
                for row in ranked.head(limit).itertuples(index=False)
            ]
        output.append(
            {
                "aspect_id": aspect_id,
                "aspect_name": aspect_names.get(aspect_id, aspect_id),
                "positive_products": rankings["positive"],
                "negative_products": rankings["negative"],
            }
        )
    return output


def _repurchase_rankings(
    reviews: pd.DataFrame, overview: pd.DataFrame, *, limit: int
) -> list[dict[str, Any]]:
    text = reviews["review_text"].fillna("").astype(str)
    positive = text.str.contains(_REPURCHASE_PATTERN, regex=True)
    negated = text.str.contains(_NEGATED_REPURCHASE_PATTERN, regex=True)
    signals = reviews[positive & ~negated]
    counts = signals.groupby("parent_asin").agg(
        signal_count=("review_id", "nunique")
    )
    ranked = overview.merge(
        counts, on="parent_asin", how="left", validate="one_to_one"
    )
    ranked["signal_count"] = ranked["signal_count"].fillna(0).astype(int)
    ranked["signal_share"] = ranked["signal_count"] / ranked["review_count"]
    ranked = ranked.sort_values(
        ["signal_count", "signal_share", "review_count"], ascending=False
    )
    return [
        {
            "parent_asin": str(row.parent_asin),
            "store": None if pd.isna(row.store) else str(row.store),
            "product_title": str(row.product_title),
            "review_count": int(row.review_count),
            "average_rating": float(row.average_rating),
            "signal_count": int(row.signal_count),
            "signal_share": float(row.signal_share),
            "support_sufficient": int(row.signal_count) >= 3,
        }
        for row in ranked.head(limit).itertuples(index=False)
    ]


def _recurring_complaints(
    product_mart: pd.DataFrame,
    overview: pd.DataFrame,
    *,
    aspect_names: dict[str, str],
    limit: int,
) -> list[dict[str, Any]]:
    if "product_average_rating" in product_mart.columns:
        rows = product_mart[product_mart["product_average_rating"] >= 4.0].copy()
    else:
        high_rated = overview[overview["average_rating"] >= 4.0]
        rows = product_mart.merge(
            high_rated[["parent_asin", "average_rating"]].rename(
                columns={"average_rating": "product_average_rating"}
            ),
            on="parent_asin",
            how="inner",
            validate="many_to_one",
        )
    rows = rows[rows["negative_count"] >= 3].sort_values(
        ["negative_count", "negative_share", "aspect_review_count"],
        ascending=False,
    ).drop_duplicates("parent_asin")
    return [
        {
            "parent_asin": str(row.parent_asin),
            "store": None if pd.isna(row.store) else str(row.store),
            "product_title": str(row.product_title),
            "average_rating": float(row.product_average_rating),
            "aspect_id": str(row.aspect_id),
            "aspect_name": aspect_names.get(str(row.aspect_id), str(row.canonical_name)),
            "mention_count": int(row.aspect_review_count),
            "negative_count": int(row.negative_count),
            "negative_share": float(row.negative_share),
        }
        for row in rows.head(limit).itertuples(index=False)
    ]


def _seller_reasons(
    product_mart: pd.DataFrame,
    seller_parent_asins: list[str],
    *,
    aspect_names: dict[str, str],
    sentiment: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = product_mart[
        product_mart["parent_asin"].isin(seller_parent_asins)
        & (product_mart[f"{sentiment}_count"] > 0)
    ].sort_values(
        [f"{sentiment}_count", f"{sentiment}_share", "aspect_review_count"],
        ascending=False,
    )
    return [
        {
            "aspect_id": str(row.aspect_id),
            "aspect_name": aspect_names.get(str(row.aspect_id), str(row.canonical_name)),
            "mention_count": int(row.aspect_review_count),
            "sentiment_count": int(getattr(row, f"{sentiment}_count")),
            "sentiment_share": float(getattr(row, f"{sentiment}_share")),
        }
        for row in rows.head(limit).itertuples(index=False)
    ]


def build_product_comparison_insights(
    product_mart: pd.DataFrame,
    reviews: pd.DataFrame,
    *,
    seller_parent_asins: list[str],
    competitor_parent_asins: list[str],
    aspect_names: dict[str, str],
    limit: int = 5,
) -> dict[str, Any]:
    """Build exact product comparisons inside one fixed workspace scope."""
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    selected_ids = seller_parent_asins + competitor_parent_asins
    scoped_reviews = reviews[reviews["parent_asin"].isin(selected_ids)].copy()
    scoped_mart = product_mart[
        product_mart["parent_asin"].isin(selected_ids)
    ].copy()
    if "comparative_analytics_allowed" in scoped_mart.columns:
        scoped_mart = scoped_mart[
            scoped_mart["comparative_analytics_allowed"].astype(bool)
        ].copy()
    if scoped_reviews.empty:
        raise ValueError("workspace review scope is empty")
    overview = scoped_reviews.groupby("parent_asin", sort=False).agg(
        store=("store", "first"),
        product_title=("product_title", "first"),
        review_count=("review_id", "nunique"),
        average_rating=("rating", "mean"),
    ).reset_index()
    return {
        "aspect_rankings": _aspect_rankings(
            scoped_mart, overview, aspect_names=aspect_names, limit=limit
        ),
        "repurchase_products": _repurchase_rankings(
            scoped_reviews, overview, limit=limit
        ),
        "high_rating_recurring_complaints": _recurring_complaints(
            scoped_mart, overview, aspect_names=aspect_names, limit=limit
        ),
        "seller_top_praise_reasons": _seller_reasons(
            scoped_mart,
            seller_parent_asins,
            aspect_names=aspect_names,
            sentiment="positive",
            limit=limit,
        ),
        "seller_top_criticism_reasons": _seller_reasons(
            scoped_mart,
            seller_parent_asins,
            aspect_names=aspect_names,
            sentiment="negative",
            limit=limit,
        ),
    }


def build_global_niche_comparison_insights(
    rankings: pd.DataFrame,
    *,
    competitor_niche_id: str,
    aspect_names: dict[str, str],
    limit: int = 5,
) -> dict[str, Any]:
    """Read precomputed rankings for every eligible product in one exact niche."""
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    scoped = rankings[
        rankings["competitor_niche_id"].eq(competitor_niche_id)
        & rankings["comparative_analytics_allowed"].astype(bool)
    ].copy()
    if scoped.empty:
        raise ValueError(f"no global rankings for niche: {competitor_niche_id}")
    aspect_rankings: list[dict[str, Any]] = []
    for aspect_id in COMPARISON_ASPECT_IDS:
        rows = scoped[scoped["aspect_id"].eq(aspect_id)]
        result: dict[str, list[dict[str, Any]]] = {}
        for sentiment in ("positive", "negative"):
            rank_column = f"{sentiment}_frequency_rank"
            ranked = rows[rows[rank_column].notna()].sort_values(
                [rank_column, f"{sentiment}_quality_rank", "parent_asin"],
                na_position="last",
            )
            result[sentiment] = [
                _rank_row(row, sentiment)
                for row in ranked.head(limit).itertuples(index=False)
            ]
        aspect_rankings.append(
            {
                "aspect_id": aspect_id,
                "aspect_name": aspect_names.get(aspect_id, aspect_id),
                "positive_products": result["positive"],
                "negative_products": result["negative"],
            }
        )

    complaint_rows = scoped[
        scoped["is_high_rating_recurring_complaint"].astype(bool)
    ].sort_values(
        ["negative_count", "negative_smoothed_share", "aspect_review_count"],
        ascending=False,
    ).drop_duplicates("parent_asin")
    recurring = [
        {
            "parent_asin": str(row.parent_asin),
            "store": None if pd.isna(row.store) else str(row.store),
            "product_title": str(row.product_title),
            "average_rating": float(row.product_average_rating),
            "aspect_id": str(row.aspect_id),
            "aspect_name": aspect_names.get(
                str(row.aspect_id), str(row.canonical_name)
            ),
            "mention_count": int(row.aspect_review_count),
            "negative_count": int(row.negative_count),
            "negative_share": float(row.negative_share),
        }
        for row in complaint_rows.head(limit).itertuples(index=False)
    ]
    return {
        "aspect_rankings": aspect_rankings,
        "high_rating_recurring_complaints": recurring,
        "comparison_product_count": int(scoped["parent_asin"].nunique()),
        "comparison_niche_id": competitor_niche_id,
    }
