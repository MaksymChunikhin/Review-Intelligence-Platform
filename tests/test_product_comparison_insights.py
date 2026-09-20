"""Tests for deterministic product-comparison question facts."""

import pandas as pd

from src.analytics.product_comparison_insights import (
    build_global_niche_comparison_insights,
    build_product_comparison_insights,
)


def test_build_product_comparison_insights_ranks_full_scope() -> None:
    reviews = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "parent_asin": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "rating": [5, 5, 4, 5, 4, 4],
            "review_text": [
                "I will buy this again",
                "soft hair",
                "great",
                "I would not buy this again",
                "soft",
                "dry",
            ],
            "product_title": ["Seller"] * 3 + ["Competitor"] * 3,
            "store": ["Seller Brand"] * 3 + ["Competitor Brand"] * 3,
        }
    )
    mart = pd.DataFrame(
        {
            "parent_asin": ["p1", "p2", "p1", "p2"],
            "aspect_id": ["softness", "softness", "value_for_money", "value_for_money"],
            "aspect_review_count": [8, 6, 3, 2],
            "positive_count": [7, 3, 2, 1],
            "negative_count": [0, 3, 0, 1],
            "positive_share": [0.875, 0.5, 0.667, 0.5],
            "negative_share": [0.0, 0.5, 0.0, 0.5],
            "product_review_count": [3, 3, 3, 3],
            "product_title": ["Seller", "Competitor", "Seller", "Competitor"],
            "store": ["Seller Brand", "Competitor Brand", "Seller Brand", "Competitor Brand"],
            "canonical_name": ["Softness", "Softness", "Value", "Value"],
        }
    )

    result = build_product_comparison_insights(
        mart,
        reviews,
        seller_parent_asins=["p1"],
        competitor_parent_asins=["p2"],
        aspect_names={"softness": "Softness", "value_for_money": "Value for Money"},
    )

    softness = next(
        item for item in result["aspect_rankings"] if item["aspect_id"] == "softness"
    )
    assert softness["positive_products"][0]["parent_asin"] == "p1"
    assert softness["negative_products"][0]["parent_asin"] == "p2"
    assert result["repurchase_products"][0]["parent_asin"] == "p1"
    assert result["repurchase_products"][0]["signal_count"] == 1
    assert result["repurchase_products"][1]["signal_count"] == 0
    assert result["high_rating_recurring_complaints"][0]["parent_asin"] == "p2"
    assert result["seller_top_praise_reasons"][0]["aspect_id"] == "softness"


def test_global_rankings_are_limited_to_one_exact_niche() -> None:
    rankings = pd.DataFrame(
        {
            "parent_asin": ["p1", "p2", "p3"],
            "competitor_niche_id": ["shampoos", "shampoos", "conditioners"],
            "aspect_id": ["softness", "softness", "softness"],
            "comparative_analytics_allowed": [True, True, True],
            "positive_frequency_rank": [2, 1, 1],
            "positive_quality_rank": [1, 2, 1],
            "negative_frequency_rank": [1, 2, 1],
            "negative_quality_rank": [2, 1, 1],
            "positive_count": [9, 12, 100],
            "negative_count": [4, 3, 100],
            "positive_share": [0.75, 0.60, 0.90],
            "negative_share": [0.25, 0.15, 0.90],
            "negative_smoothed_share": [0.2, 0.1, 0.8],
            "aspect_review_count": [12, 20, 110],
            "product_review_count": [30, 40, 200],
            "product_average_rating": [4.4, 4.5, 4.8],
            "product_title": ["Shampoo A", "Shampoo B", "Conditioner"],
            "store": ["A", "B", "C"],
            "canonical_name": ["Softness"] * 3,
            "is_high_rating_recurring_complaint": [True, True, True],
        }
    )
    result = build_global_niche_comparison_insights(
        rankings,
        competitor_niche_id="shampoos",
        aspect_names={"softness": "Softness"},
    )
    softness = next(
        item for item in result["aspect_rankings"] if item["aspect_id"] == "softness"
    )
    assert result["comparison_product_count"] == 2
    assert softness["positive_products"][0]["parent_asin"] == "p2"
    assert all(
        item["parent_asin"] != "p3"
        for item in result["high_rating_recurring_complaints"]
    )
