"""Tests for diverse metadata-filtered review retrieval."""

from __future__ import annotations

import pandas as pd

from src.rag.evidence_retrieval import retrieve_review_evidence


def test_retrieve_review_evidence_filters_and_diversifies_products() -> None:
    reviews = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4"],
            "parent_asin": ["p1", "p1", "p2", "p3"],
            "rating": [1.0, 2.0, 2.0, 1.0],
            "review_timestamp": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
                utc=True,
            ),
            "review_text": [
                "The spray pump broke immediately",
                "The spray nozzle broke after a week",
                "Broken spray trigger on arrival",
                "The scent was unpleasant",
            ],
            "product_title": ["A", "A", "B", "C"],
            "store": ["A", "A", "B", "C"],
            "helpful_vote": [3, 1, 2, 10],
        }
    )
    observations = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4"],
            "aspect_id": ["dispenser_functionality"] * 3 + ["scent"],
            "aspect_probability": [0.9, 0.8, 0.7, 0.95],
            "aspect_sentiment": ["negative"] * 4,
            "evidence_json": [
                '[{"text":"spray pump broke"}]',
                '[{"text":"spray nozzle broke"}]',
                '[{"text":"Broken spray trigger"}]',
                '[{"text":"scent"}]',
            ],
        }
    )

    result = retrieve_review_evidence(
        reviews,
        observations,
        query="broken spray",
        parent_asins=["p1", "p2", "p3"],
        aspect_id="dispenser_functionality",
        sentiment="negative",
        top_k=3,
        max_per_product=1,
    )

    assert {item["parent_asin"] for item in result} == {"p1", "p2"}
    assert all(item["aspect_id"] == "dispenser_functionality" for item in result)
    assert all(item["retrieval_score"] > 0 for item in result)


def test_retrieve_review_evidence_returns_empty_when_query_has_no_match() -> None:
    reviews = pd.DataFrame(
        {
            "review_id": ["r1"],
            "parent_asin": ["p1"],
            "rating": [5.0],
            "review_timestamp": pd.to_datetime(["2022-01-01"], utc=True),
            "review_text": ["Soft shiny hair"],
            "product_title": ["A"],
            "store": ["A"],
            "helpful_vote": [0],
        }
    )
    observations = pd.DataFrame(
        {
            "review_id": ["r1"],
            "aspect_id": ["softness"],
            "aspect_probability": [0.9],
            "aspect_sentiment": ["positive"],
            "evidence_json": ['[{"text":"Soft"}]'],
        }
    )
    assert retrieve_review_evidence(
        reviews,
        observations,
        query="broken bottle",
        parent_asins=["p1"],
    ) == []

    dense_result = retrieve_review_evidence(
        reviews,
        observations,
        query="broken bottle",
        parent_asins=["p1"],
        dense_scores={"r1": 0.7},
    )
    assert dense_result[0]["review_id"] == "r1"
    assert dense_result[0]["retrieval_method"] == "hybrid_tfidf_lsa_v1"
