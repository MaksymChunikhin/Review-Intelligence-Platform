"""Tests for reproducible direct-competitor selection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics.competitor_selection import (
    rank_direct_competitors,
    select_direct_competitors,
)


def _catalog() -> pd.DataFrame:
    rows = [
        ("seller", "Seller", "Adult Leave-In Conditioner Spray Detangler Anti-Frizz", 8.0, 500),
        ("direct", "Other", "Adult Leave-In Conditioner Spray Detangler Anti-Frizz", 9.0, 400),
        ("kids", "KidsCo", "Kids Leave-In Conditioner Spray Detangler", 9.0, 900),
        ("bundle", "BundleCo", "Leave-In Conditioner Spray Pack of 3", 24.0, 800),
        ("rinse", "RinseCo", "Moisturizing Rinse Out Conditioner", 8.0, 700),
        ("same-brand", "Seller", "Leave-In Conditioner Cream", 8.0, 600),
    ]
    return pd.DataFrame(
        {
            "parent_asin": [row[0] for row in rows],
            "category_path_id": ["conditioners"] * len(rows),
            "product_title": [row[2] for row in rows],
            "product_search_text": [row[2] for row in rows],
            "store": [row[1] for row in rows],
            "price_at_collection": [row[3] for row in rows],
            "average_rating": [4.5] * len(rows),
            "rating_number": [1_000] * len(rows),
            "review_count": [row[4] for row in rows],
        }
    )


def test_rank_direct_competitors_excludes_wrong_audience_and_format() -> None:
    ranking, counts = rank_direct_competitors(
        _catalog(),
        seller_parent_asin="seller",
        category_path_ids=["conditioners"],
        top_k=1,
        minimum_review_count=100,
    )
    assert counts == {
        "catalog_candidate_count": 6,
        "eligible_candidate_count": 1,
    }
    assert ranking.loc[0, "parent_asin"] == "direct"
    assert bool(ranking.loc[0, "selected"])


def test_select_direct_competitors_persists_ranking_and_report(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.parquet"
    ranking_path = tmp_path / "ranking.parquet"
    report_path = tmp_path / "report.json"
    _catalog().to_parquet(catalog_path, index=False)

    report = select_direct_competitors(
        catalog_path,
        ranking_path,
        seller_parent_asin="seller",
        category_path_ids=["conditioners"],
        selection_version="selection_v1",
        top_k=1,
        report_path=report_path,
    )

    assert report.selected_parent_asins == ["direct"]
    assert ranking_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8"))[
        "ranking_sha256"
    ] == report.ranking_sha256


def test_rank_direct_competitors_supports_rinse_out_conditioner() -> None:
    catalog = pd.concat(
        [
            _catalog(),
            pd.DataFrame(
                {
                    "parent_asin": ["rinse-direct"],
                    "category_path_id": ["conditioners"],
                    "product_title": ["Daily Moisturizing Smooth Conditioner"],
                    "product_search_text": [
                        "Daily Moisturizing Smooth Conditioner"
                    ],
                    "store": ["Another Brand"],
                    "price_at_collection": [9.0],
                    "average_rating": [4.5],
                    "rating_number": [1_000],
                    "review_count": [650],
                }
            ),
        ],
        ignore_index=True,
    )

    ranking, _ = rank_direct_competitors(
        catalog,
        seller_parent_asin="rinse",
        category_path_ids=["conditioners"],
        top_k=1,
        minimum_review_count=100,
    )

    assert ranking.loc[0, "parent_asin"] == "rinse-direct"
    assert "leave-in" not in ranking.loc[0, "product_title"].lower()


def test_catalog_path_profile_keeps_product_format_consistent() -> None:
    catalog = _catalog().copy()
    catalog.loc[catalog["parent_asin"] == "seller", "product_title"] = (
        "Daily 2 in 1 Shampoo and Conditioner"
    )
    catalog.loc[catalog["parent_asin"] == "direct", "product_title"] = (
        "Moisturizing 2 in 1 Shampoo and Conditioner"
    )
    catalog.loc[catalog["parent_asin"] == "kids", "product_title"] = (
        "Kids 2 in 1 Shampoo and Conditioner"
    )
    catalog.loc[catalog["parent_asin"] == "bundle", "product_title"] = (
        "2 in 1 Shampoo and Conditioner Set"
    )
    catalog.loc[catalog["parent_asin"] == "rinse", "product_title"] = (
        "Color Depositing 2 in 1 Shampoo and Conditioner"
    )
    ranking, counts = rank_direct_competitors(
        catalog,
        seller_parent_asin="seller",
        category_path_ids=["conditioners"],
        top_k=1,
        minimum_review_count=100,
        selection_profile="catalog_path",
    )
    assert ranking.loc[0, "parent_asin"] == "direct"
    assert counts["eligible_candidate_count"] == 1


def test_brand_subline_is_not_selected_as_a_competitor() -> None:
    catalog = _catalog().copy()
    catalog.loc[catalog["parent_asin"] == "seller", "store"] = "DOVE MEN + CARE"
    catalog.loc[catalog["parent_asin"] == "direct", "store"] = "Dove"
    catalog.loc[catalog["parent_asin"] == "rinse", "store"] = "Other Brand"
    alternative = catalog.loc[catalog["parent_asin"] == "direct"].copy()
    alternative["parent_asin"] = "other-direct"
    alternative["store"] = "Independent"
    catalog = pd.concat([catalog, alternative], ignore_index=True)
    ranking, _ = rank_direct_competitors(
        catalog,
        seller_parent_asin="seller",
        category_path_ids=["conditioners"],
        top_k=1,
        minimum_review_count=100,
        selection_profile="catalog_path",
    )
    assert ranking.loc[0, "parent_asin"] == "other-direct"
