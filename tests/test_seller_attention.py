"""Test the deterministic seller-attention business signal."""

import pandas as pd
import pytest

from src.analytics.seller_attention import (
    SELLER_ATTENTION_POLICY_VERSION,
    add_seller_attention,
    rating_to_seller_attention,
)


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (1, "needs_attention"),
        (2.0, "needs_attention"),
        (3, "needs_attention"),
        (4.0, "satisfied"),
        (5, "satisfied"),
    ],
)
def test_rating_to_seller_attention(rating: float, expected: str) -> None:
    assert rating_to_seller_attention(rating) == expected


@pytest.mark.parametrize("rating", [0, 3.5, 6, None, "3", True])
def test_rating_to_seller_attention_rejects_invalid_values(rating: object) -> None:
    with pytest.raises(ValueError, match="Rating must"):
        rating_to_seller_attention(rating)  # type: ignore[arg-type]


def test_add_seller_attention_preserves_input_and_records_policy() -> None:
    reviews = pd.DataFrame({"review_id": ["r1", "r2"], "rating": [3.0, 5.0]})

    result = add_seller_attention(reviews)

    assert "seller_attention" not in reviews.columns
    assert result["seller_attention"].tolist() == ["needs_attention", "satisfied"]
    assert result["seller_attention_policy_version"].eq(
        SELLER_ATTENTION_POLICY_VERSION
    ).all()
