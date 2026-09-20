"""Tests for product-aspect and month-aspect marts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics.aspect_marts import build_aspect_marts


def test_aspect_marts_reconcile_product_and_month_counts(tmp_path: Path) -> None:
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
                    },
                    {
                        "aspect_id": "softness",
                        "canonical_name": "Softness",
                        "parent_group": "hair_outcome",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "parent_asin": ["p1", "p1", "p2"],
            "rating": [2.0, 5.0, 4.0],
            "review_timestamp": pd.to_datetime(
                ["2023-01-01", "2023-02-01", "2023-02-02"], utc=True
            ),
            "helpful_vote": [2, 0, 1],
            "product_title": ["One", "One", "Two"],
            "store": ["Store", "Store", "Other"],
        }
    ).to_parquet(reviews_path, index=False)
    sentiment_path = tmp_path / "sentiment.parquet"
    pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "aspect_id": ["scent", "softness", "scent"],
            "taxonomy_version": ["taxonomy_v1"] * 3,
            "extraction_version": ["extract_v1"] * 3,
            "aspect_sentiment_version": ["sentiment_v1"] * 3,
            "aspect_sentiment": ["negative", "positive", "positive"],
            "aspect_sentiment_score": [0.9, 0.8, 0.7],
            "aspect_probability": [0.8, 0.7, 0.6],
            "prediction_sources_json": [
                '["exact_alias"]', '["char_tfidf"]', '["exact_alias","char_tfidf"]'
            ],
            "evidence_json": [
                '[{"text":"bad scent","start":0,"end":9}]',
                '[{"text":"soft","start":0,"end":4}]',
                '[{"text":"nice scent","start":0,"end":10}]',
            ],
        }
    ).to_parquet(sentiment_path, index=False)
    product_path = tmp_path / "product.parquet"
    month_path = tmp_path / "month.parquet"
    month_csv_path = tmp_path / "month.csv"
    report = build_aspect_marts(
        taxonomy_path,
        reviews_path,
        sentiment_path,
        product_path,
        month_path,
        month_csv_path,
        marts_version="marts_v1",
    )
    product = pd.read_parquet(product_path)
    month = pd.read_parquet(month_path)
    assert report.product_aspect_reconciled_count == 3
    assert report.month_aspect_reconciled_count == 3
    assert len(month) == 4
    assert product["needs_attention_count"].sum() == 1
    assert set(month["year_month"]) == {"2023-01", "2023-02"}
