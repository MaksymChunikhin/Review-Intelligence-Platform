"""Retrieve diverse review evidence with lexical relevance and metadata filters."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize_text(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _evidence_phrases(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [str(item["text"]) for item in json.loads(str(value))]


def retrieve_review_evidence(
    reviews: pd.DataFrame,
    aspect_observations: pd.DataFrame,
    *,
    query: str,
    parent_asins: list[str],
    aspect_id: str | None = None,
    sentiment: str | None = None,
    top_k: int = 8,
    max_per_product: int = 2,
    dense_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rank grounded review evidence while keeping product coverage diverse."""
    normalized_query = _normalize_text(query)
    if len(normalized_query) < 2:
        raise ValueError("query must contain at least two searchable characters")
    if not parent_asins:
        raise ValueError("parent_asins must not be empty")
    if not 1 <= top_k <= 25:
        raise ValueError("top_k must be between 1 and 25")
    if max_per_product < 1:
        raise ValueError("max_per_product must be positive")
    if sentiment not in {None, "negative", "neutral", "positive"}:
        raise ValueError("sentiment must be negative, neutral, or positive")

    required_reviews = {
        "review_id",
        "parent_asin",
        "rating",
        "review_timestamp",
        "review_text",
        "product_title",
        "store",
        "helpful_vote",
    }
    required_observations = {
        "review_id",
        "aspect_id",
        "aspect_probability",
        "aspect_sentiment",
        "evidence_json",
    }
    if missing := required_reviews - set(reviews.columns):
        raise ValueError(f"reviews are missing columns: {sorted(missing)}")
    if missing := required_observations - set(aspect_observations.columns):
        raise ValueError(f"aspect observations are missing columns: {sorted(missing)}")

    scoped_reviews = reviews[reviews["parent_asin"].isin(parent_asins)].copy()
    observations = aspect_observations[
        aspect_observations["review_id"].isin(scoped_reviews["review_id"])
    ].copy()
    if aspect_id is not None:
        observations = observations[observations["aspect_id"] == aspect_id]
    if sentiment is not None:
        observations = observations[
            observations["aspect_sentiment"] == sentiment
        ]
    if observations.empty:
        return []

    observations = observations.sort_values(
        ["aspect_probability", "aspect_id"], ascending=[False, True]
    ).drop_duplicates("review_id")
    candidates = observations.merge(
        scoped_reviews[list(required_reviews)],
        on="review_id",
        how="inner",
        validate="one_to_one",
    )
    candidates["normalized_review_text"] = candidates["review_text"].map(
        _normalize_text
    )
    candidates = candidates[
        candidates["normalized_review_text"].str.len() >= 2
    ].drop_duplicates("normalized_review_text")
    if candidates.empty:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(
        [normalized_query, *candidates["normalized_review_text"].tolist()]
    )
    candidates["lexical_score"] = cosine_similarity(
        matrix[0:1], matrix[1:]
    ).ravel()
    candidates["dense_score"] = candidates["review_id"].astype(str).map(
        dense_scores or {}
    ).fillna(0.0).clip(lower=0.0, upper=1.0)
    candidates = candidates[
        (candidates["lexical_score"] > 0) | (candidates["dense_score"] > 0)
    ].copy()
    if candidates.empty:
        return []
    maximum_helpful = max(1.0, float(candidates["helpful_vote"].fillna(0).max()))
    candidates["helpful_score"] = np.log1p(
        candidates["helpful_vote"].fillna(0).clip(lower=0)
    ) / np.log1p(maximum_helpful)
    if dense_scores is None:
        candidates["retrieval_score"] = (
            0.80 * candidates["lexical_score"]
            + 0.15 * candidates["aspect_probability"].fillna(0).clip(0, 1)
            + 0.05 * candidates["helpful_score"]
        )
        retrieval_method = "lexical_tfidf_v1"
    else:
        candidates["retrieval_score"] = (
            0.55 * candidates["lexical_score"]
            + 0.25 * candidates["dense_score"]
            + 0.15 * candidates["aspect_probability"].fillna(0).clip(0, 1)
            + 0.05 * candidates["helpful_score"]
        )
        retrieval_method = "hybrid_tfidf_lsa_v1"
    candidates = candidates.sort_values(
        ["retrieval_score", "helpful_vote", "review_id"],
        ascending=[False, False, True],
    )

    selected: list[dict[str, Any]] = []
    product_counts: dict[str, int] = {}
    for row in candidates.itertuples(index=False):
        product_id = str(row.parent_asin)
        if product_counts.get(product_id, 0) >= max_per_product:
            continue
        selected.append(
            {
                "review_id": str(row.review_id),
                "parent_asin": product_id,
                "product_title": str(row.product_title),
                "store": None if pd.isna(row.store) else str(row.store),
                "rating": float(row.rating),
                "review_timestamp": pd.Timestamp(row.review_timestamp).isoformat(),
                "review_text": str(row.review_text),
                "aspect_id": str(row.aspect_id),
                "aspect_sentiment": str(row.aspect_sentiment),
                "aspect_probability": float(row.aspect_probability),
                "evidence_phrases": _evidence_phrases(row.evidence_json),
                "helpful_vote": (
                    0 if pd.isna(row.helpful_vote) else int(row.helpful_vote)
                ),
                "lexical_score": float(row.lexical_score),
                "dense_score": float(row.dense_score),
                "retrieval_score": float(row.retrieval_score),
                "retrieval_method": retrieval_method,
            }
        )
        product_counts[product_id] = product_counts.get(product_id, 0) + 1
        if len(selected) >= top_k:
            break
    return selected
