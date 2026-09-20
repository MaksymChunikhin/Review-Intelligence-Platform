"""API contract tests for the seller MVP."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pandas as pd

from src.api.app import create_app


def _build_artifacts(root: Path) -> None:
    workspace = {
        "workspace_id": "demo",
        "workspace_version": "demo_v1",
        "dataset_version": "dataset_v1",
        "niche": {
            "niche_id": "conditioners",
            "niche_version": "conditioners_v1",
            "dataset_version": "dataset_v1",
            "category_registry_schema_version": "registry_v1",
            "display_name": "Conditioners",
            "status": "approved",
            "category_path_ids": ["path-1"],
        },
        "seller_parent_asins": ["p1"],
        "competitor_parent_asins": ["p2"],
        "review_date_start": "2021-01-01",
        "review_date_end": "2023-09-12",
        "verified_purchase_only": False,
    }
    config_path = root / "config" / "workspaces" / "demo_v1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(workspace), encoding="utf-8")
    niche_path = root / "config" / "niches" / "hair_conditioners_v1.json"
    niche_path.parent.mkdir(parents=True)
    niche_path.write_text(json.dumps(workspace["niche"]), encoding="utf-8")

    data_root = root / "data" / "processed" / "dataset_v1"
    niche_root = data_root / "niches" / "conditioners_v1"
    workspace_root = data_root / "workspaces" / "demo_v1"
    niche_root.mkdir(parents=True)
    workspace_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "parent_asin": ["p1", "p2"],
            "category_path_id": ["path-1", "path-1"],
            "product_title": ["Seller leave-in", "Competitor leave-in"],
            "product_search_text": [
                "seller leave-in conditioner",
                "competitor leave-in conditioner",
            ],
            "store": ["Seller", "Competitor"],
            "price_at_collection": [8.0, 9.0],
            "average_rating": [4.5, 4.4],
            "rating_number": [500, 400],
            "review_count": [1, 1],
        }
    ).to_parquet(data_root / "product_catalog.parquet", index=False)
    pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "parent_asin": ["p1", "p2"],
            "rating": [5.0, 2.0],
            "review_timestamp": pd.to_datetime(
                ["2022-01-01", "2022-01-02"], utc=True
            ),
            "review_text": ["Very manageable hair", "Hard to manage"],
            "product_title": ["Seller leave-in", "Competitor leave-in"],
            "store": ["Seller", "Competitor"],
            "helpful_vote": [2, 1],
        }
    ).to_parquet(niche_root / "reviews.parquet", index=False)
    pd.DataFrame(
        {
            "parent_asin": ["p1", "p2"],
            "aspect_id": ["manageability", "manageability"],
            "aspect_review_count": [1, 1],
            "positive_count": [1, 0],
            "negative_count": [0, 1],
            "positive_share": [1.0, 0.0],
            "negative_share": [0.0, 1.0],
            "product_review_count": [1, 1],
            "product_title": ["Seller leave-in", "Competitor leave-in"],
            "store": ["Seller", "Competitor"],
            "canonical_name": ["Manageability", "Manageability"],
            "top_negative_review_id": [None, "r2"],
            "top_negative_evidence_json": [None, '[{"text":"Hard to manage"}]'],
            "top_positive_review_id": ["r1", None],
            "top_positive_evidence_json": [
                '[{"text":"Very manageable hair"}]',
                None,
            ],
        }
    ).to_parquet(niche_root / "product_aspect_mart_v1.parquet", index=False)
    pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "aspect_id": ["manageability", "manageability"],
            "aspect_probability": [0.9, 0.8],
            "aspect_sentiment": ["positive", "negative"],
            "evidence_json": [
                '[{"text":"manageable hair"}]',
                '[{"text":"Hard to manage"}]',
            ],
        }
    ).to_parquet(niche_root / "aspect_sentiment_v2.parquet", index=False)
    pd.DataFrame(
        {
            "aspect_id": ["manageability"],
            "canonical_name": ["Manageability"],
            "seller_aspect_review_count": [30],
            "seller_needs_attention_count": [3],
            "seller_positive_count": [27],
            "seller_needs_attention_share": [0.1],
            "competitor_needs_attention_share": [0.4],
            "attention_gap_pp": [-30.0],
            "seller_positive_share": [0.9],
            "competitor_positive_share": [0.6],
            "positive_gap_pp": [30.0],
            "advantage_score_pp": [60.0],
            "support_sufficient": [True],
        }
    ).to_parquet(workspace_root / "aspect_comparison.parquet", index=False)
    report_path = root / "reports" / "seller_demo" / "demo_v1.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "comparison_path": (
                    "data/processed/dataset_v1/workspaces/demo_v1/"
                    "aspect_comparison.parquet"
                )
            }
        ),
        encoding="utf-8",
    )


def test_api_serves_latest_report_and_evidence(tmp_path: Path) -> None:
    _build_artifacts(tmp_path)
    application = create_app(tmp_path)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            assert (await client.get("/api/health")).json()["status"] == "ok"
            workspaces = await client.get("/api/v1/workspaces")
            assert workspaces.status_code == 200
            assert workspaces.json()["items"][0]["workspace_version"] == "demo_v1"

            search = await client.get("/api/v1/products/search?q=seller")
            assert search.status_code == 200
            assert search.json()["items"][0]["parent_asin"] == "p1"

            asin_search = await client.get("/api/v1/products/search?q=p1")
            assert asin_search.status_code == 200
            assert asin_search.json()["items"][0]["parent_asin"] == "p1"

            punctuation_search = await client.get(
                "/api/v1/products/search", params={"q": ".."}
            )
            assert punctuation_search.status_code == 400

            existing_workspace = await client.post(
                "/api/v1/workspaces",
                json={
                    "seller_parent_asin": "AAAAAAAAAA",
                    "top_k": 1,
                    "minimum_review_count": 20,
                },
            )
            assert existing_workspace.status_code == 400

            response = await client.get("/api/v1/workspaces/demo_v1/report")
            assert response.status_code == 200
            payload = response.json()
            assert payload["products"][0]["review_count"] == 1
            assert payload["strengths"][0]["aspect_id"] == "manageability"
            assert payload["strengths"][0]["aspect_name"] == "Manageability"
            assert payload["limitations"][0].startswith("Historical")

            comparisons = await client.get(
                "/api/v1/workspaces/demo_v1/comparison-insights"
            )
            assert comparisons.status_code == 200
            assert comparisons.json()["seller_top_praise_reasons"][0][
                "aspect_id"
            ] == "manageability"

            evidence = await client.get(
                "/api/v1/workspaces/demo_v1/aspects/manageability/evidence"
            )
            assert evidence.status_code == 200
            assert evidence.json()["positive_examples"][0]["review_text"] == (
                "Very manageable hair"
            )

            search_evidence = await client.get(
                "/api/v1/workspaces/demo_v1/evidence/search",
                params={"q": "manageable hair", "scope": "seller"},
            )
            assert search_evidence.status_code == 200
            assert search_evidence.json()["items"][0]["review_id"] == "r1"

            negative_rating_evidence = await client.get(
                "/api/v1/workspaces/demo_v1/evidence/search",
                params={
                    "q": "manage",
                    "scope": "all",
                    "rating_band": "negative",
                },
            )
            assert negative_rating_evidence.status_code == 200
            assert all(
                item["rating"] <= 2
                for item in negative_rating_evidence.json()["items"]
            )

            ai_sentiment_evidence = await client.get(
                "/api/v1/workspaces/demo_v1/evidence/search",
                params={
                    "q": "manage",
                    "scope": "all",
                    "sentiment": "positive",
                },
            )
            assert ai_sentiment_evidence.status_code == 200
            assert ai_sentiment_evidence.json()["rating_band"] is None
            assert all(
                item["aspect_sentiment"] == "positive"
                for item in ai_sentiment_evidence.json()["items"]
            )

            trend = await client.get(
                "/api/v1/workspaces/demo_v1/aspects/manageability/trend"
            )
            assert trend.status_code == 200
            assert trend.json()["items"][0]["positive_share"] == 1.0

            context = await client.get(
                "/api/v1/workspaces/demo_v1/rag/context",
                params={
                    "question": "Почему волосы стали послушнее?",
                    "evidence_query": "manageable hair",
                    "scope": "seller",
                    "aspect_id": "manageability",
                },
            )
            assert context.status_code == 200
            assert context.json()["citations"][0]["citation_id"] == "R1"

            application.state.analytics_store.ask_analyst = lambda *args, **kwargs: {
                "workspace_version": "demo_v1",
                "question": kwargs["question"],
                "resolved_aspect_id": "manageability",
                "evidence_query": "hair manageable easy to style control",
                "evidence_review_count": 1,
                "answer_ru": "Волосы легче укладывать [R1].",
                "answer": "Волосы легче укладывать [R1].",
                "findings": [
                    {
                        "finding": "Есть положительный пример.",
                        "basis": "reviews",
                        "citations": ["R1"],
                    }
                ],
                "recommendations": [],
                "limitations": ["Исторические данные."],
                "sources": [
                    {
                        "citation_id": "R1",
                        "brand": "Seller",
                        "rating": 5.0,
                        "review_date": "2022-01-01",
                        "review_text": "Very manageable hair",
                        "aspect_id": "manageability",
                        "sentiment": "positive",
                    }
                ],
                "metadata": {
                    "analyst_version": "analyst_test_v1",
                    "model_version": "gemini-test",
                    "response_id": "response-1",
                    "usage_metadata": {"total_token_count": 100},
                    "available_citations": ["R1"],
                    "used_citations": ["R1"],
                },
            }
            analyst = await client.post(
                "/api/v1/workspaces/demo_v1/ask",
                json={
                    "question": "Почему волосы стали послушнее?",
                    "scope": "seller",
                    "external_processing_consent": True,
                },
            )
            assert analyst.status_code == 200
            assert analyst.json()["metadata"]["used_citations"] == ["R1"]

            missing_consent = await client.post(
                "/api/v1/workspaces/demo_v1/ask",
                json={"question": "Почему волосы стали послушнее?"},
            )
            assert missing_consent.status_code == 422

    asyncio.run(scenario())


def test_api_returns_404_for_unknown_workspace(tmp_path: Path) -> None:
    _build_artifacts(tmp_path)
    application = create_app(tmp_path)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/workspaces/missing/report")
            assert response.status_code == 404

    asyncio.run(scenario())
