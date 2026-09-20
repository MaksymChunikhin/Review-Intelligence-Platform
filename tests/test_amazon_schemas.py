"""Tests for canonical Amazon data contracts."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.ingestion.amazon_records import transform_amazon_product_record
from src.schemas.amazon import (
    AmazonEnrichedReviewRecord,
    AmazonProductRecord,
    AmazonReviewRecord,
)


def test_review_contract_normalizes_timestamp_to_utc() -> None:
    review = AmazonReviewRecord(
        review_id="dataset:review:000000001",
        dataset_version="dataset",
        dataset_category="Beauty_and_Personal_Care",
        source_record_index=1,
        asin="B000000001",
        parent_asin="B000000000",
        rating=5,
        review_title="Works",
        review_body="Works very well",
        review_text="Works Works very well",
        review_timestamp=datetime.fromisoformat("2023-01-01T02:00:00+02:00"),
        source_timestamp_ms=1672531200000,
        verified_purchase=True,
        helpful_vote=3,
    )

    assert review.review_timestamp.tzinfo == timezone.utc
    assert review.review_timestamp.isoformat() == "2023-01-01T00:00:00+00:00"


def test_review_contract_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="must include a timezone"):
        AmazonReviewRecord(
            review_id="dataset:review:000000001",
            dataset_version="dataset",
            dataset_category="Beauty_and_Personal_Care",
            source_record_index=1,
            asin="B000000001",
            parent_asin="B000000000",
            rating=5,
            review_title="Works",
            review_body="Works very well",
            review_text="Works Works very well",
            review_timestamp=datetime(2023, 1, 1),
            source_timestamp_ms=1672531200000,
        )


def test_product_contract_preserves_amazon_category_path() -> None:
    product = AmazonProductRecord(
        dataset_version="dataset",
        dataset_category="Beauty_and_Personal_Care",
        source_record_index=0,
        parent_asin="B000000000",
        product_title="Conditioner",
        main_category="All Beauty",
        category_path=["Beauty & Personal Care", "Hair Care", "Conditioners"],
    )

    assert product.category_path[-1] == "Conditioners"


def test_product_adapter_preserves_structure_and_reports_missing_price() -> None:
    product = transform_amazon_product_record(
        {
            "parent_asin": "B000000000",
            "title": "Conditioner",
            "main_category": "All Beauty",
            "categories": ["Beauty & Personal Care", "Hair Care"],
            "features": [" Softens hair ", ""],
            "description": [],
            "price": None,
            "images": [{"variant": "MAIN"}],
            "videos": [],
            "store": "Example Brand",
            "average_rating": 4.5,
            "rating_number": 10,
            "details": {"Item Form": "Cream"},
            "bought_together": None,
        },
        dataset_version="dataset",
        dataset_category="Beauty_and_Personal_Care",
        source_record_index=12,
    )

    assert product.source_record_index == 12
    assert product.category_path == ["Beauty & Personal Care", "Hair Care"]
    assert product.features == ["Softens hair"]
    assert product.details == {"Item Form": "Cream"}
    assert product.metadata_quality_flags == ["missing_price"]


def test_enriched_contract_rejects_wrong_parent_product() -> None:
    review = AmazonReviewRecord(
        review_id="dataset:review:000000001",
        dataset_version="dataset",
        dataset_category="Beauty_and_Personal_Care",
        source_record_index=1,
        asin="B000000001",
        parent_asin="B000000000",
        rating=5,
        review_title="Works",
        review_body="Works very well",
        review_text="Works Works very well",
        review_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
        source_timestamp_ms=1672531200000,
    )
    product = AmazonProductRecord(
        dataset_version="dataset",
        dataset_category="Beauty_and_Personal_Care",
        source_record_index=2,
        parent_asin="B999999999",
        product_title="Another product",
    )

    with pytest.raises(ValidationError, match="parent_asin values must match"):
        AmazonEnrichedReviewRecord(review=review, product=product)
