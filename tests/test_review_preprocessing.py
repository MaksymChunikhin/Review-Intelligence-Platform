"""Tests for canonical Amazon review preprocessing."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.preprocessing.reviews import (
    CANONICAL_REVIEW_SCHEMA,
    ReviewPreprocessingConfig,
    preprocess_reviews,
    transform_review_chunk,
    validate_canonical_review_parquet,
)


def make_source_review(**overrides: object) -> dict[str, object]:
    """Return one valid source record with optional test overrides."""
    record: dict[str, object] = {
        "rating": 5.0,
        "title": "Great conditioner",
        "text": "Leaves my hair soft.",
        "images": [],
        "asin": "B000000001",
        "parent_asin": "B000000000",
        "user_id": "USER-1",
        "timestamp": 1672531200000,
        "helpful_vote": 1,
        "verified_purchase": True,
    }
    record.update(overrides)
    return record


def make_config(*, deduplicate: bool = True) -> ReviewPreprocessingConfig:
    """Return the deterministic test pipeline configuration."""
    return ReviewPreprocessingConfig(
        dataset_version="beauty_test_v1",
        dataset_category="Beauty_and_Personal_Care",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        chunk_size=2,
        deduplicate=deduplicate,
    )


def test_transform_review_chunk_reports_primary_drop_reasons() -> None:
    frame = pd.DataFrame(
        [
            make_source_review(),
            make_source_review(rating=7),
            make_source_review(parent_asin=""),
            make_source_review(title="", text=""),
        ]
    )

    transformed, quality = transform_review_chunk(
        frame, config=make_config(), source_record_offset=10
    )

    assert len(transformed) == 1
    assert transformed.loc[0, "review_id"] == "beauty_test_v1:review:000000010"
    assert transformed.loc[0, "review_timestamp"].isoformat() == (
        "2023-01-01T00:00:00+00:00"
    )
    assert quality["dropped_by_primary_reason"] == {
        "invalid_rating": 1,
        "missing_parent_asin": 1,
        "empty_review_text": 1,
    }


def test_preprocess_reviews_is_atomic_and_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "reviews.jsonl"
    # Anonymous reviews use NULL as a real deduplication key; review_id must not
    # make otherwise exact source duplicates artificially unique.
    first = make_source_review(user_id=None)
    records = [
        first,
        first.copy(),
        make_source_review(
            asin="B000000002",
            parent_asin="B000000002",
            user_id="USER-2",
            title="Useful",
            text="",
        ),
        make_source_review(rating=0),
    ]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output = tmp_path / "canonical.parquet"
    report_path = tmp_path / "report.json"

    report = preprocess_reviews(
        source,
        output,
        config=make_config(),
        report_path=report_path,
    )
    result = pd.read_parquet(output)

    assert report.input_rows == 4
    assert report.valid_rows_before_deduplication == 3
    assert report.duplicate_rows_removed == 1
    assert report.output_rows == 2
    assert report.dropped_by_primary_reason == {"invalid_rating": 1}
    assert result["source_record_index"].tolist() == [0, 2]
    assert result["review_id"].is_unique
    assert report_path.is_file()
    assert not list(tmp_path.glob(".*.stage.parquet"))
    assert not list(tmp_path.glob(".*.deduplicated.parquet"))

    validation = validate_canonical_review_parquet(output)
    assert validation.is_valid
    assert validation.column_nullability_match
    assert validation.row_count == 2
    assert set(validation.required_null_counts.values()) == {0}
    assert validation.column_names_match
    assert validation.column_types_match
    assert pq.ParquetFile(output).schema_arrow.equals(
        CANONICAL_REVIEW_SCHEMA,
        check_metadata=False,
    )


def test_preprocess_reviews_quarantines_malformed_integer_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "malformed_reviews.jsonl"
    base_timestamp = 1_672_531_200_000
    records = [
        make_source_review(user_id="USER-1", helpful_vote="not-a-number"),
        make_source_review(
            user_id="USER-2",
            timestamp=base_timestamp + 1,
            helpful_vote=1.5,
        ),
        make_source_review(
            user_id="USER-3",
            timestamp=base_timestamp + 2,
            helpful_vote=-1,
        ),
        make_source_review(
            user_id="USER-4",
            timestamp=base_timestamp + 3,
            helpful_vote=2_147_483_648,
            verified_purchase="yes",
        ),
        make_source_review(
            user_id="USER-5",
            timestamp=base_timestamp + 4,
            helpful_vote=None,
        ),
        make_source_review(user_id="USER-6", timestamp="not-a-number"),
        make_source_review(user_id="USER-7", timestamp=base_timestamp + 0.5),
        make_source_review(user_id="USER-8", timestamp=10**40),
    ]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output = tmp_path / "canonical.parquet"

    report = preprocess_reviews(source, output, config=make_config())
    result = pd.read_parquet(output)

    assert report.input_rows == 8
    assert report.valid_rows_before_deduplication == 5
    assert report.output_rows == 5
    assert report.dropped_by_primary_reason == {"invalid_timestamp": 3}
    assert report.invalid_optional_values == {
        "helpful_vote": 4,
        "verified_purchase": 1,
    }
    assert result["helpful_vote"].isna().all()
    assert validate_canonical_review_parquet(output).is_valid
