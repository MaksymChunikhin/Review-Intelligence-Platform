"""Tests for approved niche population validation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.niche_scope import profile_niche_population
from src.schemas.workspace import ProductNicheDefinition


def make_niche() -> ProductNicheDefinition:
    """Return an approved niche for population fixtures."""
    return ProductNicheDefinition(
        niche_id="hair_conditioners",
        niche_version="hair_conditioners_v1",
        dataset_version="beauty_test_v1",
        category_registry_schema_version="amazon_category_registry_v1",
        display_name="Hair Conditioners",
        status="approved",
        category_path_ids=["amazon-category:conditioners"],
    )


def write_scope_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write catalog, review, and registry Parquet fixtures."""
    catalog_path = tmp_path / "catalog.parquet"
    pd.DataFrame(
        {
            "dataset_version": ["beauty_test_v1"] * 3,
            "parent_asin": ["P1", "P2", "P3"],
            "product_title": ["One", "Two", "Other"],
            "store": ["A", "B", "C"],
            "category_path_id": [
                "amazon-category:conditioners",
                "amazon-category:conditioners",
                "amazon-category:other",
            ],
            "category_path_text": ["Beauty > Conditioners"] * 2
            + ["Beauty > Other"],
            "review_count": [3, 2, 1],
            "reviewed_asin_count": [2, 1, 1],
        }
    ).to_parquet(catalog_path, index=False)

    reviews_path = tmp_path / "reviews.parquet"
    long_text = " ".join(["word"] * 121)
    medium_text = " ".join(["word"] * 41)
    pd.DataFrame(
        {
            "review_id": ["R1", "R2", "R3", "R4", "R5", "R6"],
            "dataset_version": ["beauty_test_v1"] * 6,
            "parent_asin": ["P1", "P1", "P1", "P2", "P2", "P3"],
            "asin": ["A1", "A2", "A1", "A3", "A3", "A4"],
            "rating": [1.0, 5.0, 3.0, 4.0, 2.0, 5.0],
            "review_timestamp": pd.to_datetime(
                [
                    "2021-01-01",
                    "2022-01-01",
                    "2023-01-01",
                    "2021-06-01",
                    "2022-06-01",
                    "2023-06-01",
                ],
                utc=True,
            ),
            "verified_purchase": [True, True, False, True, False, True],
            "helpful_vote": [0, 2, 1, 0, 0, 0],
            "review_text": [
                "one two three four five six",
                medium_text,
                long_text,
                "six useful words about this conditioner",
                "too short",
                "unrelated product review has six words",
            ],
        }
    ).to_parquet(reviews_path, index=False)

    registry_path = tmp_path / "registry.parquet"
    pd.DataFrame(
        {
            "category_path_id": ["amazon-category:conditioners"],
            "category_path_text": ["Beauty > Conditioners"],
            "product_record_count": [2],
            "distinct_parent_asin_count": [2],
            "dataset_version": ["beauty_test_v1"],
            "registry_schema_version": ["amazon_category_registry_v1"],
        }
    ).to_parquet(registry_path, index=False)
    return catalog_path, reviews_path, registry_path


def test_profile_niche_population_reconciles_products_and_reviews(
    tmp_path: Path,
) -> None:
    catalog, reviews, registry = write_scope_fixture(tmp_path)
    report_path = tmp_path / "population.json"

    report = profile_niche_population(
        make_niche(),
        catalog,
        reviews,
        registry,
        report_path=report_path,
        memory_limit="512MB",
    )

    assert report.catalog_product_count == 2
    assert report.reviewed_product_count == 2
    assert report.reviewed_variation_asin_count == 3
    assert report.review_count == 5
    assert report.distinct_review_asin_count == 3
    assert report.verified_purchase_count == 3
    assert report.helpful_review_count == 2
    assert report.review_date_min == "2021-01-01"
    assert report.review_date_max == "2023-01-01"
    assert json.loads(report_path.read_text())["niche_status"] == "approved"


def test_profile_niche_population_rejects_unknown_path(tmp_path: Path) -> None:
    catalog, reviews, registry = write_scope_fixture(tmp_path)
    niche = make_niche().model_copy(
        update={"category_path_ids": ["amazon-category:missing"]}
    )

    with pytest.raises(ValueError, match="absent from registry"):
        profile_niche_population(niche, catalog, reviews, registry)
