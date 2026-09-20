"""Create deterministic seller-attention signals from Amazon star ratings."""

from __future__ import annotations

from numbers import Real

import pandas as pd


SELLER_ATTENTION_POLICY_VERSION = "amazon_rating_attention_v1"
SELLER_ATTENTION_LABELS = ("needs_attention", "satisfied")


def rating_to_seller_attention(rating: float) -> str:
    """Map 1–3 stars to attention and 4–5 stars to satisfaction.

    This is a seller-oriented business rule, not a prediction of textual
    sentiment. A three-star review may use neutral language while still
    identifying an opportunity to improve the product.
    """
    if not isinstance(rating, Real) or isinstance(rating, bool):
        raise ValueError(f"Rating must be a number from 1 to 5: {rating!r}")
    numeric_rating = float(rating)
    if numeric_rating not in (1.0, 2.0, 3.0, 4.0, 5.0):
        raise ValueError(
            f"Rating must be an integer-like value from 1 to 5: {rating}"
        )
    return "needs_attention" if numeric_rating <= 3 else "satisfied"


def add_seller_attention(
    reviews: pd.DataFrame,
    *,
    rating_column: str = "rating",
) -> pd.DataFrame:
    """Return review rows with a versioned seller-attention business signal."""
    if rating_column not in reviews.columns:
        raise ValueError(f"Review data is missing the rating column: {rating_column}")
    result = reviews.copy()
    result["seller_attention"] = result[rating_column].map(
        rating_to_seller_attention
    )
    result["seller_attention_policy_version"] = SELLER_ATTENTION_POLICY_VERSION
    return result
