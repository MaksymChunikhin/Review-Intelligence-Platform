"""Tests for compact niche-review materialization."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.niche_reviews import materialize_niche_reviews


def test_materialize_niche_reviews_filters_scope_and_drops_user_id(tmp_path: Path) -> None:
    config_path = tmp_path / "niche.json"
    config_path.write_text(
        json.dumps(
            {
                "niche_id": "conditioners",
                "niche_version": "conditioners_v1",
                "dataset_version": "dataset_v1",
                "category_registry_schema_version": "registry_v1",
                "display_name": "Conditioners",
                "status": "approved",
                "category_path_ids": ["path-1"],
            }
        ),
        encoding="utf-8",
    )
    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "dataset_version": ["dataset_v1", "dataset_v1"],
            "parent_asin": ["p1", "p2"],
            "asin": ["a1", "a2"],
            "rating": [5.0, 1.0],
            "review_timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "verified_purchase": [True, False],
            "helpful_vote": [0, 1],
            "review_text": ["soft hair", "wrong product"],
            "user_id": ["u1", "u2"],
        }
    ).to_parquet(reviews_path, index=False)
    catalog_path = tmp_path / "catalog.parquet"
    pd.DataFrame(
        {
            "parent_asin": ["p1", "p2"],
            "product_title": ["One", "Two"],
            "store": ["Store", "Store"],
            "category_path_id": ["path-1", "path-2"],
            "category_path_text": ["Conditioners", "Other"],
        }
    ).to_parquet(catalog_path, index=False)
    output_path = tmp_path / "niche_reviews.parquet"
    report = materialize_niche_reviews(
        config_path, reviews_path, catalog_path, output_path
    )
    output = pd.read_parquet(output_path)
    assert report.review_count == 1
    assert output["review_id"].tolist() == ["r1"]
    assert "user_id" not in output.columns
    assert output["analytical_family_id"].tolist() == ["conditioners"]
    assert output["competitor_niche_id"].tolist() == ["conditioners"]
    assert output["comparison_status"].tolist() == ["eligible"]
    assert report.analytical_family_count == 1
    assert report.competitor_niche_count == 1
    assert report.quarantined_review_count == 0
