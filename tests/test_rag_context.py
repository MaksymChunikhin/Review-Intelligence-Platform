"""Tests for bounded citation-ready RAG context construction."""

from __future__ import annotations

from src.rag.context import build_rag_context


def test_build_rag_context_keeps_aggregate_scope_and_review_citations() -> None:
    report = {
        "workspace_version": "demo_v1",
        "review_date_start": "2021-01-01",
        "review_date_end": "2023-09-12",
        "products": [
            {
                "role": "seller",
                "store": "Seller",
                "parent_asin": "p1",
                "review_count": 100,
            },
            {"role": "competitor"},
        ],
        "strengths": [],
        "weaknesses": [
            {
                "aspect_name": "дозатор",
                "mention_count": 20,
                "seller_attention_share": 0.5,
                "competitor_attention_share": 0.2,
                "attention_gap_pp": 30.0,
                "seller_positive_share": 0.2,
                "competitor_positive_share": 0.6,
            }
        ],
        "complaints": [],
        "limitations": ["Historical"],
    }
    evidence = [
        {
            "review_id": "r1",
            "parent_asin": "p1",
            "store": "Seller",
            "rating": 1.0,
            "review_timestamp": "2022-01-01T00:00:00+00:00",
            "aspect_id": "dispenser_functionality",
            "aspect_sentiment": "negative",
            "review_text": "The sprayer broke.",
        }
    ]

    result = build_rag_context(
        question="Почему жалуются на дозатор?",
        report=report,
        evidence=evidence,
    )

    assert "seller_attention=0.5" in result["context"]
    assert "[R1]" in result["context"]
    assert result["citations"][0]["review_id"] == "r1"
    assert "Do not turn a single review" in result["instructions"]
