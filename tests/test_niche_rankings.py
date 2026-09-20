"""Tests for exact-niche product rankings."""

from pathlib import Path

import pandas as pd

from src.analytics.niche_rankings import build_niche_rankings


def test_niche_rankings_filter_scope_and_unreliable_aspects(tmp_path: Path) -> None:
    source = tmp_path / "product_mart.parquet"
    rows = []
    for asin, positive, negative, rating in (
        ("a", 8, 2, 4.4),
        ("b", 3, 7, 4.2),
    ):
        rows.append(
            {
                "marts_version": "marts_v1",
                "parent_asin": asin,
                "aspect_id": "softness",
                "canonical_name": "Softness",
                "competitor_niche_id": "conditioners",
                "competitor_niche_name": "Conditioners",
                "comparison_status": "eligible",
                "product_review_count": 20,
                "product_average_rating": rating,
                "aspect_review_count": 10,
                "positive_count": positive,
                "negative_count": negative,
                "comparative_analytics_allowed": True,
                "reliability_tier": "strong",
            }
        )
    rows.append(
        {
            **rows[0],
            "aspect_id": "greasy_oily_finish",
            "canonical_name": "Greasy Finish",
            "comparative_analytics_allowed": False,
            "reliability_tier": "experimental",
        }
    )
    pd.DataFrame(rows).to_parquet(source, index=False)
    output = tmp_path / "rankings.parquet"

    report = build_niche_rankings(
        source,
        output,
        rankings_version="rankings_v1",
    )
    ranked = pd.read_parquet(output).set_index("parent_asin")

    assert report.excluded_aspect_count == 1
    assert report.ranked_aspect_count == 1
    assert ranked.loc["a", "positive_frequency_rank"] == 1
    assert ranked.loc["b", "negative_frequency_rank"] == 1
    assert ranked.loc["b", "is_high_rating_recurring_complaint"]
