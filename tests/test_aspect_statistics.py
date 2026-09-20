"""Tests for offline aspect statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics.aspect_statistics import build_aspect_statistics


def test_build_aspect_statistics_counts_sentiment_and_attention(tmp_path: Path) -> None:
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "taxonomy_version": "taxonomy_v1",
                "aspects": [
                    {
                        "aspect_id": "scent",
                        "canonical_name": "Scent",
                        "parent_group": "sensory",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "parent_asin": ["p1", "p2"],
            "rating": [2.0, 5.0],
            "helpful_vote": [3, 1],
            "review_timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        }
    ).to_parquet(reviews_path, index=False)
    sentiment_path = tmp_path / "sentiment.parquet"
    pd.DataFrame(
        {
            "extraction_version": ["extract_v1", "extract_v1"],
            "aspect_sentiment_version": ["sentiment_v1", "sentiment_v1"],
            "taxonomy_version": ["taxonomy_v1", "taxonomy_v1"],
            "review_id": ["r1", "r2"],
            "aspect_id": ["scent", "scent"],
            "aspect_sentiment": ["negative", "positive"],
            "aspect_sentiment_score": [0.9, 0.8],
            "aspect_probability": [0.7, 0.6],
            "prediction_sources_json": ['["exact_alias"]', '["char_tfidf"]'],
            "evidence_json": [
                '[{"text":"bad smell","start":0,"end":9}]',
                '[{"text":"nice scent","start":0,"end":10}]',
            ],
        }
    ).to_parquet(sentiment_path, index=False)
    output_path = tmp_path / "stats.parquet"
    csv_path = tmp_path / "stats.csv"
    report = build_aspect_statistics(
        taxonomy_path,
        reviews_path,
        sentiment_path,
        output_path,
        csv_path,
        statistics_version="stats_v1",
    )
    result = pd.read_parquet(output_path).iloc[0]
    assert report.review_aspect_count == 2
    assert result["review_share"] == 1.0
    assert result["negative_count"] == 1
    assert result["positive_count"] == 1
    assert result["needs_attention_share"] == 0.5
    assert json.loads(result["top_negative_examples_json"])[0]["review_id"] == "r1"
