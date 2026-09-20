"""Tests for evidence-context aspect sentiment scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

from src.ml.aspect_sentiment import build_aspect_context, score_aspect_sentiment
from src.ml.batched_aspect_sentiment import score_aspect_sentiment_batched


class FakeVectorizer:
    def transform(self, values: pd.Series) -> list[str]:
        return values.astype(str).tolist()


class FakeClassifier:
    classes_ = np.asarray(["negative", "neutral", "positive"])

    def predict_proba(self, values: list[str]) -> np.ndarray:
        rows = []
        for value in values:
            if "leaks" in value:
                rows.append([0.8, 0.1, 0.1])
            else:
                rows.append([0.1, 0.1, 0.8])
        return np.asarray(rows)


def evidence(text: str, phrase: str) -> str:
    start = text.index(phrase)
    return json.dumps(
        [
            {
                "text": phrase,
                "start": start,
                "end": start + len(phrase),
                "matched_alias": phrase,
            }
        ]
    )


def test_build_aspect_context_selects_evidence_sentence() -> None:
    text = "It works well. The bottle leaks badly! Shipping was quick."
    context = build_aspect_context(text, evidence(text, "bottle"))
    assert context == "The bottle leaks badly!"


def test_build_aspect_context_rejects_changed_evidence() -> None:
    text = "The scent is pleasant."
    payload = json.loads(evidence(text, "scent"))
    payload[0]["text"] = "smell"
    with pytest.raises(ValueError, match="does not match"):
        build_aspect_context(text, json.dumps(payload))


def test_score_aspect_sentiment_writes_lineage_and_scores(tmp_path: Path) -> None:
    review_text = "Lovely scent. The bottle leaks badly!"
    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {"review_id": ["r1"], "review_text": [review_text]}
    ).to_parquet(reviews_path, index=False)
    predictions_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "extraction_version": ["extract_v1", "extract_v1"],
            "taxonomy_version": ["taxonomy_v1", "taxonomy_v1"],
            "method": ["alias", "alias"],
            "review_id": ["r1", "r1"],
            "aspect_id": ["scent", "packaging"],
            "evidence_count": [1, 1],
            "matched_aliases_json": ['["scent"]', '["bottle"]'],
            "evidence_json": [
                evidence(review_text, "scent"),
                evidence(review_text, "bottle"),
            ],
        }
    ).to_parquet(predictions_path, index=False)
    output_path = tmp_path / "scored.parquet"
    report_path = tmp_path / "report.json"

    report = score_aspect_sentiment(
        predictions_path,
        reviews_path,
        tmp_path,
        output_path,
        aspect_sentiment_version="aspect_sentiment_test_v1",
        report_path=report_path,
        vectorizer=FakeVectorizer(),
        classifier=FakeClassifier(),
        sentiment_model_version="fake_sentiment_v1",
    )
    scored = pd.read_parquet(output_path)

    assert list(scored["aspect_sentiment"]) == ["positive", "negative"]
    assert list(scored["aspect_sentiment_score"]) == [0.8, 0.8]
    assert report.review_aspect_count == 2
    assert report.sentiment_counts == {"negative": 1, "positive": 1}
    assert json.loads(report_path.read_text())["output_sha256"]


def test_batched_scoring_matches_regular_scoring(tmp_path: Path) -> None:
    review_text = "Lovely scent. The bottle leaks badly!"
    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {"review_id": ["r1"], "review_text": [review_text], "unused": ["x"]}
    ).to_parquet(reviews_path, index=False)
    predictions_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "extraction_version": ["extract_v1", "extract_v1"],
            "taxonomy_version": ["taxonomy_v1", "taxonomy_v1"],
            "review_id": ["r1", "r1"],
            "aspect_id": ["scent", "packaging"],
            "evidence_json": [
                evidence(review_text, "scent"),
                evidence(review_text, "bottle"),
            ],
        }
    ).to_parquet(predictions_path, index=False)
    model_path = tmp_path / "model"
    model_path.mkdir()
    joblib.dump(FakeVectorizer(), model_path / "vectorizer.joblib")
    joblib.dump(FakeClassifier(), model_path / "classifier.joblib")
    (model_path / "experiment_config.json").write_text(
        json.dumps({"model_config": {"model_version": "fake_sentiment_v1"}}),
        encoding="utf-8",
    )

    output_path = tmp_path / "batched.parquet"
    report = score_aspect_sentiment_batched(
        predictions_path,
        reviews_path,
        model_path,
        output_path,
        aspect_sentiment_version="aspect_sentiment_test_v1",
        batch_size=1,
    )
    scored = pd.read_parquet(output_path)

    assert list(scored["aspect_sentiment"]) == ["positive", "negative"]
    assert report.review_aspect_count == 2
    assert report.review_count == 1
    assert report.sentiment_counts == {"negative": 1, "positive": 1}
