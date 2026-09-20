"""Tests for the complete Amazon product catalog builder."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.analytics.category_registry import build_category_registry
from src.analytics.product_catalog import (
    PRODUCT_CATALOG_SCHEMA,
    build_product_catalog,
)


def test_build_product_catalog_tracks_quality_and_review_coverage(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_records = [
        {
            "parent_asin": "P1",
            "title": "Conditioner One",
            "main_category": "All Beauty",
            "categories": ["Beauty & Personal Care", "Hair Care", "Conditioners"],
            "store": "Brand One",
            "average_rating": 4.5,
            "rating_number": 20,
            "price": 12.5,
            "features": ["Soft hair"],
            "description": [],
            "details": {"Item Form": "Cream"},
        },
        {
            "parent_asin": "P2",
            "title": "Conditioner Two",
            "main_category": None,
            "categories": ["Beauty & Personal Care", "Hair Care", "Conditioners"],
            "store": None,
            "average_rating": 4.0,
            "rating_number": 3,
            "price": "from 10.00",
            "features": [],
            "description": [],
            "details": {},
        },
        {
            "parent_asin": "P3",
            "title": None,
            "main_category": "All Beauty",
            "categories": ["Beauty & Personal Care", "Skin Care"],
            "store": "Brand Three",
            "average_rating": 5.0,
            "rating_number": 1,
            "price": None,
            "features": [],
            "description": [],
            "details": {},
        },
    ]
    metadata_path.write_text(
        "".join(json.dumps(record) + "\n" for record in metadata_records),
        encoding="utf-8",
    )
    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {
            "parent_asin": ["P1", "P1", "P3", "P9"],
            "asin": ["A1", "A2", "A3", "A9"],
            "review_timestamp": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
                utc=True,
            ),
        }
    ).to_parquet(reviews_path, index=False)
    registry_path = tmp_path / "registry.parquet"
    build_category_registry(
        metadata_path,
        registry_path,
        dataset_version="beauty_test_v1",
        dataset_category="Beauty_and_Personal_Care",
        registry_schema_version="amazon_category_registry_v1",
        memory_limit="512MB",
    )
    # The catalog and registry can be built from independently normalized
    # representations of the same Amazon path.
    metadata_records[0]["categories"] = [
        " Ｂｅａｕｔｙ ＆ Ｐｅｒｓｏｎａｌ Ｃａｒｅ ",
        "hair   care",
        "CONDITIONERS",
    ]
    metadata_path.write_text(
        "".join(json.dumps(record) + "\n" for record in metadata_records),
        encoding="utf-8",
    )
    output_path = tmp_path / "catalog.parquet"

    report = build_product_catalog(
        metadata_path,
        reviews_path,
        registry_path,
        output_path,
        dataset_version="beauty_test_v1",
        dataset_category="Beauty_and_Personal_Care",
        catalog_schema_version="amazon_product_catalog_v1",
        memory_limit="512MB",
    )
    catalog = pd.read_parquet(output_path)

    assert report.output_rows == 3
    assert report.invalid_price_rows == 1
    assert report.missing_price_rows == 1
    assert report.missing_product_title_rows == 1
    assert report.unmatched_category_path_rows == 0
    assert report.matched_review_rows == 3
    assert report.unmatched_review_rows == 1
    assert report.review_join_coverage == 0.75
    assert report.catalog_products_with_reviews == 2
    assert catalog.loc[catalog["parent_asin"] == "P1", "review_count"].item() == 2
    assert list(
        catalog.loc[catalog["parent_asin"] == "P1", "category_path"].item()
    ) == ["Beauty & Personal Care", "Hair Care", "Conditioners"]
    assert catalog["category_path_id"].notna().all()
    assert pq.ParquetFile(output_path).schema_arrow.equals(
        PRODUCT_CATALOG_SCHEMA,
        check_metadata=False,
    )
    assert not (tmp_path / "catalog.tmp.parquet").exists()


def test_build_product_catalog_rejects_duplicate_parent_asins(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "duplicate_metadata.jsonl"
    record = {
        "parent_asin": "P1",
        "title": "Conditioner One",
        "main_category": "All Beauty",
        "categories": ["Beauty", "Hair Care"],
        "store": "Brand",
        "average_rating": 4.5,
        "rating_number": 2,
        "price": 10.0,
        "features": [],
        "description": [],
        "details": {},
    }
    metadata_path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )
    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {
            "parent_asin": ["P1"],
            "asin": ["A1"],
            "review_timestamp": pd.to_datetime(["2023-01-01"], utc=True),
        }
    ).to_parquet(reviews_path, index=False)
    registry_path = tmp_path / "registry.parquet"
    build_category_registry(
        metadata_path,
        registry_path,
        dataset_version="beauty_test_v1",
        dataset_category="Beauty_and_Personal_Care",
        registry_schema_version="amazon_category_registry_v1",
        memory_limit="512MB",
    )

    with pytest.raises(ValueError, match="one row per parent_asin"):
        build_product_catalog(
            metadata_path,
            reviews_path,
            registry_path,
            tmp_path / "catalog.parquet",
            dataset_version="beauty_test_v1",
            dataset_category="Beauty_and_Personal_Care",
            catalog_schema_version="amazon_product_catalog_v1",
            memory_limit="512MB",
        )

    assert not (tmp_path / "catalog.parquet").exists()
