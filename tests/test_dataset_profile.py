"""Tests for deterministic canonical dataset profiles."""

import json
from pathlib import Path

from src.analytics.dataset_profile import profile_canonical_reviews
from src.preprocessing.reviews import preprocess_reviews
from tests.test_review_preprocessing import make_config, make_source_review


def test_profile_canonical_reviews_reports_exact_counts(tmp_path: Path) -> None:
    source = tmp_path / "reviews.jsonl"
    records = [
        make_source_review(),
        make_source_review(
            asin="B000000002",
            parent_asin="B000000002",
            user_id="USER-2",
            rating=1,
            verified_purchase=False,
        ),
    ]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output = tmp_path / "canonical.parquet"
    preprocess_reviews(source, output, config=make_config(deduplicate=False))

    profile = profile_canonical_reviews(output)

    assert profile["summary"]["review_count"] == 2
    assert profile["summary"]["parent_asin_count"] == 2
    assert profile["summary"]["verified_count"] == 1
    assert profile["rating_distribution"] == [
        {"rating": 1.0, "review_count": 1},
        {"rating": 5.0, "review_count": 1},
    ]
