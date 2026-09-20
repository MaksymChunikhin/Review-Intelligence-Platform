"""Tests for reproducible sentiment sampling and evaluation."""

import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

import src.ml.sentiment as sentiment_module
from src.ingestion.dataset_manifest import sha256_file
from src.ml.sentiment import (
    SENTIMENT_MANIFEST_COLUMNS,
    SentimentSplitConfig,
    TfidfBaselineConfig,
    audit_sentiment_sample,
    balanced_evaluation_view,
    build_sentiment_sample,
    evaluate_predictions,
    load_sentiment_split_reviews,
    rating_to_sentiment,
    save_sentiment_experiment,
    write_sentiment_split_manifest,
)


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (1, "negative"),
        (2, "negative"),
        (3, "neutral"),
        (4, "positive"),
        (5, "positive"),
    ],
)
def test_rating_to_sentiment(rating: int, expected: str) -> None:
    assert rating_to_sentiment(rating) == expected


def test_rating_to_sentiment_rejects_invalid_rating() -> None:
    with pytest.raises(ValueError, match="1 to 5"):
        rating_to_sentiment(0)


def test_balanced_evaluation_view_and_metrics_are_deterministic() -> None:
    predictions = pd.DataFrame(
        {
            "review_id": [f"review-{index}" for index in range(12)],
            "parent_asin": [f"product-{index // 2}" for index in range(12)],
            "sentiment_label": ["negative"] * 3
            + ["neutral"] * 4
            + ["positive"] * 5,
            "predicted_label": ["negative"] * 2
            + ["neutral"]
            + ["neutral"] * 3
            + ["positive"]
            + ["positive"] * 4
            + ["negative"],
        }
    )

    first = balanced_evaluation_view(
        predictions, max_per_class=3, random_state=42
    )
    second = balanced_evaluation_view(
        predictions, max_per_class=3, random_state=42
    )
    metrics = evaluate_predictions(first, bootstrap_rounds=10, random_state=42)

    assert first["review_id"].tolist() == second["review_id"].tolist()
    assert first["sentiment_label"].value_counts().to_dict() == {
        "negative": 3,
        "neutral": 3,
        "positive": 3,
    }
    assert metrics["review_count"] == 9
    assert set(metrics["confidence_intervals_95"]) == {"accuracy", "macro_f1"}
    assert metrics["confidence_interval_method"]["name"] == (
        "percentile_cluster_bootstrap"
    )
    assert metrics["confidence_interval_method"]["cluster_column"] == (
        "parent_asin"
    )


def test_metrics_bootstrap_resamples_whole_parent_products() -> None:
    predictions = pd.DataFrame(
        {
            "parent_asin": ["correct-product"] * 10 + ["wrong-product"] * 10,
            "sentiment_label": ["negative"] * 10 + ["positive"] * 10,
            "predicted_label": ["negative"] * 20,
        }
    )

    metrics = evaluate_predictions(
        predictions,
        bootstrap_rounds=1_000,
        random_state=42,
    )

    assert metrics["confidence_intervals_95"]["accuracy"] == [0.0, 1.0]
    assert metrics["confidence_interval_method"]["cluster_count"] == 2


def test_optimized_cluster_bootstrap_matches_naive_resampling() -> None:
    actual = np.asarray(
        [
            "negative",
            "negative",
            "neutral",
            "positive",
            "positive",
            "positive",
            "neutral",
            "negative",
            "positive",
        ]
    )
    predicted = np.asarray(
        [
            "negative",
            "neutral",
            "neutral",
            "positive",
            "negative",
            "positive",
            "positive",
            "negative",
            "neutral",
        ]
    )
    clusters = np.asarray(
        ["a", "a", "b", "c", "c", "c", "d", "e", "e"]
    )
    rounds = 250
    random_state = 73

    optimized = sentiment_module._bootstrap_intervals(
        actual,
        predicted,
        clusters,
        rounds=rounds,
        random_state=random_state,
    )

    cluster_codes, unique_clusters = pd.factorize(clusters, sort=False)
    cluster_positions = [
        np.flatnonzero(cluster_codes == code)
        for code in range(len(unique_clusters))
    ]
    rng = np.random.default_rng(random_state)
    accuracy_values = []
    macro_f1_values = []
    for _ in range(rounds):
        sampled_clusters = rng.integers(
            0, len(cluster_positions), size=len(cluster_positions)
        )
        positions = np.concatenate(
            [cluster_positions[position] for position in sampled_clusters]
        )
        accuracy_values.append(
            accuracy_score(actual[positions], predicted[positions])
        )
        macro_f1_values.append(
            f1_score(
                actual[positions],
                predicted[positions],
                labels=list(sentiment_module.SENTIMENT_LABELS),
                average="macro",
                zero_division=0,
            )
        )
    expected = {
        "accuracy": np.quantile(accuracy_values, [0.025, 0.975]),
        "macro_f1": np.quantile(macro_f1_values, [0.025, 0.975]),
    }

    np.testing.assert_allclose(optimized["accuracy"], expected["accuracy"])
    np.testing.assert_allclose(optimized["macro_f1"], expected["macro_f1"])


def test_metrics_bootstrap_requires_product_identifier() -> None:
    predictions = pd.DataFrame(
        {
            "sentiment_label": ["negative", "neutral", "positive"],
            "predicted_label": ["negative", "neutral", "positive"],
        }
    )

    with pytest.raises(ValueError, match="parent_asin"):
        evaluate_predictions(predictions, bootstrap_rounds=1)


def test_build_sentiment_sample_prevents_product_and_text_leakage(
    tmp_path: Path,
) -> None:
    train_products = _products_for_bucket_range(0, 80, count=3)
    validation_products = _products_for_bucket_range(80, 90, count=3)
    test_products = _products_for_bucket_range(90, 100, count=6)
    all_products = train_products + validation_products + test_products
    niche_path = (
        "Beauty & Personal Care > Hair Care > Shampoo & Conditioner > "
        "Conditioners"
    )
    catalog = pd.DataFrame(
        {
            "parent_asin": all_products,
            "category_path_text": [
                niche_path if product in test_products[:3] else "Other"
                for product in all_products
            ],
        }
    )
    reviews = []
    review_number = 0
    labels = ((1.0, "negative"), (3.0, "neutral"), (5.0, "positive"))
    for product_group in (train_products, validation_products, test_products):
        for product in product_group:
            for rating, label in labels:
                reviews.append(
                    {
                        "review_id": f"review-{review_number}",
                        "parent_asin": product,
                        "asin": product,
                        "user_id": f"user-{review_number}",
                        "rating": rating,
                        "review_timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
                        "review_text": f"unique {label} text {review_number}",
                    }
                )
                review_number += 1
    for product in train_products:
        for rating, label in labels:
            reviews.append(
                {
                    "review_id": f"review-{review_number}",
                    "parent_asin": product,
                    "asin": product,
                    "user_id": f"user-{review_number}",
                    "rating": rating,
                    "review_timestamp": pd.Timestamp("2023-02-01", tz="UTC"),
                    "review_text": f"temporal {label} text {review_number}",
                }
            )
            review_number += 1
    reviews.extend(
        [
            {
                "review_id": "duplicate-train",
                "parent_asin": train_products[0],
                "asin": train_products[0],
                "user_id": "duplicate-user-a",
                "rating": 5.0,
                "review_timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
                "review_text": "Same repeated text",
            },
            {
                "review_id": "duplicate-test",
                "parent_asin": test_products[-1],
                "asin": test_products[-1],
                "user_id": "duplicate-user-b",
                "rating": 5.0,
                "review_timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
                "review_text": " same   repeated TEXT ",
            },
        ]
    )
    reviews_path = tmp_path / "reviews.parquet"
    catalog_path = tmp_path / "catalog.parquet"
    pd.DataFrame(reviews).to_parquet(reviews_path, index=False)
    catalog.to_parquet(catalog_path, index=False)
    config = SentimentSplitConfig(
        train_per_class=1,
        validation_per_class=1,
        product_test_size=3,
        temporal_test_size=3,
        niche_test_size=3,
    )

    sample = build_sentiment_sample(reviews_path, catalog_path, config=config)
    audit = audit_sentiment_sample(sample)
    train = sample.loc[sample["split_name"] == "train"]

    assert sample["review_id"].is_unique
    assert sample["text_fingerprint"].is_unique
    assert not sample["review_text"].str.contains("repeated", case=False).any()
    assert len(train) == 3
    assert audit["review_id_is_unique"]
    assert audit["text_fingerprint_is_unique"]
    audit_by_split = {row["split_name"]: row for row in audit["splits"]}
    assert audit_by_split["validation"]["train_product_overlap"] == 0
    assert audit_by_split["test_product"]["train_product_overlap"] == 0
    assert audit_by_split["test_conditioners"]["train_product_overlap"] == 0
    assert audit_by_split["test_temporal_seen"]["train_product_overlap"] > 0

    manifest_path = tmp_path / "split_manifest.parquet"
    write_sentiment_split_manifest(sample, manifest_path)
    reloaded = load_sentiment_split_reviews(
        reviews_path,
        manifest_path,
        split_names=("validation", "test_product"),
    )
    assert set(reloaded["split_name"]) == {"validation", "test_product"}
    assert reloaded["review_text"].notna().all()
    assert "review_text" not in pd.read_parquet(manifest_path).columns


def test_temporal_cutoff_is_midnight_utc_in_every_session_timezone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_product = _products_for_bucket_range(0, 80, count=1)[0]
    validation_product = _products_for_bucket_range(80, 90, count=1)[0]
    test_products = _products_for_bucket_range(90, 100, count=2)
    niche_path = (
        "Beauty & Personal Care > Hair Care > Shampoo & Conditioner > "
        "Conditioners"
    )
    catalog = pd.DataFrame(
        {
            "parent_asin": [train_product, validation_product, *test_products],
            "category_path_text": ["Other", "Other", niche_path, "Other"],
        }
    )
    reviews: list[dict[str, object]] = []
    for product, prefix in (
        (train_product, "train"),
        (validation_product, "validation"),
    ):
        for repetition in range(2):
            for rating, label in (
                (1.0, "negative"),
                (3.0, "neutral"),
                (5.0, "positive"),
            ):
                review_id = f"{prefix}-{repetition}-{label}"
                reviews.append(
                    {
                        "review_id": review_id,
                        "parent_asin": product,
                        "asin": product,
                        "user_id": f"user-{review_id}",
                        "rating": rating,
                        "review_timestamp": pd.Timestamp(
                            "2022-12-31T23:30:00Z"
                            if product == train_product and repetition == 1
                            else "2022-06-01T00:00:00Z"
                        ),
                        "review_text": f"unique text {review_id}",
                    }
                )
    for rating, label in (
        (1.0, "negative"),
        (3.0, "neutral"),
        (5.0, "positive"),
    ):
        review_id = f"temporal-after-{label}"
        reviews.append(
            {
                "review_id": review_id,
                "parent_asin": train_product,
                "asin": train_product,
                "user_id": f"user-{review_id}",
                "rating": rating,
                "review_timestamp": pd.Timestamp("2023-01-01T00:30:00Z"),
                "review_text": f"unique text {review_id}",
            }
        )
    for product, review_id in zip(
        test_products,
        ("niche-product-review", "general-product-review"),
        strict=True,
    ):
        reviews.append(
            {
                "review_id": review_id,
                "parent_asin": product,
                "asin": product,
                "user_id": f"user-{review_id}",
                "rating": 5.0,
                "review_timestamp": pd.Timestamp("2022-06-01T00:00:00Z"),
                "review_text": f"unique text {review_id}",
            }
        )

    reviews_path = tmp_path / "timezone_reviews.parquet"
    catalog_path = tmp_path / "timezone_catalog.parquet"
    pd.DataFrame(reviews).to_parquet(reviews_path, index=False)
    catalog.to_parquet(catalog_path, index=False)
    config = SentimentSplitConfig(
        train_per_class=2,
        validation_per_class=2,
        product_test_size=1,
        temporal_test_size=3,
        niche_test_size=1,
    )
    real_connect = duckdb.connect
    memberships: list[dict[str, str]] = []

    for session_timezone in ("Europe/Rome", "America/Los_Angeles"):
        def connect_in_timezone(
            *args: object,
            _session_timezone: str = session_timezone,
            **kwargs: object,
        ) -> duckdb.DuckDBPyConnection:
            connection = real_connect(*args, **kwargs)
            connection.execute(f"SET TimeZone = '{_session_timezone}'")
            return connection

        monkeypatch.setattr(
            sentiment_module.duckdb,
            "connect",
            connect_in_timezone,
        )
        sample = build_sentiment_sample(reviews_path, catalog_path, config=config)
        memberships.append(dict(zip(sample["review_id"], sample["split_name"])))
        assert str(sample["review_timestamp"].dt.tz) == "UTC"
        manifest_path = tmp_path / f"split-{session_timezone.replace('/', '-')}.parquet"
        write_sentiment_split_manifest(sample, manifest_path)
        loaded = load_sentiment_split_reviews(reviews_path, manifest_path)
        assert str(loaded["review_timestamp"].dt.tz) == "UTC"

    assert memberships[0] == memberships[1]
    for label in ("negative", "neutral", "positive"):
        assert memberships[0][f"train-1-{label}"] == "train"
        assert memberships[0][f"temporal-after-{label}"] == "test_temporal_seen"


def test_save_sentiment_experiment_replaces_predictions_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    sample_row = {
        "review_id": "review-1",
        "split_name": "validation",
        "sentiment_label": "positive",
        "rating": 5.0,
        "parent_asin": "product-1",
        "asin": "asin-1",
        "user_id": "user-1",
        "review_timestamp": pd.Timestamp("2023-01-01", tz="UTC"),
        "text_fingerprint": "fingerprint-1",
        "is_niche": False,
        "predicted_label": "positive",
        "model_score": 0.9,
        "score_negative": 0.05,
        "score_neutral": 0.05,
        "score_positive": 0.9,
    }
    predictions = pd.DataFrame([sample_row])
    written_paths: list[Path] = []

    def write_fake_parquet(
        self: pd.DataFrame,
        path: str | Path,
        *,
        index: bool,
    ) -> None:
        assert not index
        output_path = Path(path)
        written_paths.append(output_path)
        output_path.write_bytes(b"complete parquet bytes")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_fake_parquet)
    monkeypatch.setattr(sentiment_module, "_atomic_joblib_dump", lambda *args: None)
    monkeypatch.setattr(sentiment_module, "_atomic_json_dump", lambda *args: None)
    monkeypatch.setattr(
        sentiment_module,
        "write_sentiment_split_manifest",
        lambda sample, path: Path(path),
    )

    save_sentiment_experiment(
        vectorizer=SimpleNamespace(),
        classifier=SimpleNamespace(C=0.5),
        split_sample=predictions[list(SENTIMENT_MANIFEST_COLUMNS)],
        evaluation_predictions=predictions,
        metrics={},
        split_config=SentimentSplitConfig(),
        model_config=TfidfBaselineConfig(model_version="atomic-test"),
        dataset_version="dataset-v1",
        model_directory=tmp_path / "model",
        split_manifest_path=tmp_path / "split.parquet",
        predictions_path=prediction_path,
        report_path=tmp_path / "report.json",
    )

    assert written_paths == [tmp_path / "predictions.parquet.tmp"]
    assert prediction_path.read_bytes() == b"complete parquet bytes"
    assert not (tmp_path / "predictions.parquet.tmp").exists()


def test_save_experiment_keeps_identical_input_manifest_bytes_and_relative_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = _sentiment_prediction_frame()
    manifest_path = tmp_path / "splits" / "split.parquet"
    write_sentiment_split_manifest(predictions, manifest_path)
    original_hash = sha256_file(manifest_path)
    monkeypatch.setattr(sentiment_module, "find_project_root", lambda: tmp_path)

    report_path = tmp_path / "reports" / "model.json"
    save_sentiment_experiment(
        vectorizer=SimpleNamespace(vocabulary_={"example": 0}),
        classifier=SimpleNamespace(C=0.5),
        split_sample=predictions,
        evaluation_predictions=predictions,
        metrics={"test": {"accuracy": 1.0}},
        split_config=SentimentSplitConfig(split_version="split-v1"),
        model_config=TfidfBaselineConfig(model_version="model-v1"),
        dataset_version="dataset-v1",
        model_directory=tmp_path / "models" / "model-v1",
        split_manifest_path=manifest_path,
        predictions_path=tmp_path / "predictions" / "model-v1.parquet",
        report_path=report_path,
    )

    assert sha256_file(manifest_path) == original_hash
    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert str(tmp_path) not in report_text
    assert report["artifacts"] == {
        "vectorizer": "models/model-v1/vectorizer.joblib",
        "classifier": "models/model-v1/classifier.joblib",
        "split_manifest": "splits/split.parquet",
        "evaluation_predictions": "predictions/model-v1.parquet",
    }
    model_directory = tmp_path / "models" / "model-v1"
    vectorizer_hash = sha256_file(model_directory / "vectorizer.joblib")
    classifier_hash = sha256_file(model_directory / "classifier.joblib")
    created_at = report["created_at_utc"]

    save_sentiment_experiment(
        vectorizer=SimpleNamespace(vocabulary_={"different": 0}),
        classifier=SimpleNamespace(C=0.5),
        split_sample=predictions,
        evaluation_predictions=predictions,
        metrics={"test": {"accuracy": 1.0}},
        split_config=SentimentSplitConfig(split_version="split-v1"),
        model_config=TfidfBaselineConfig(model_version="model-v1"),
        dataset_version="dataset-v1",
        model_directory=model_directory,
        split_manifest_path=manifest_path,
        predictions_path=tmp_path / "predictions" / "model-v1.parquet",
        report_path=report_path,
    )

    assert sha256_file(model_directory / "vectorizer.joblib") == vectorizer_hash
    assert sha256_file(model_directory / "classifier.joblib") == classifier_hash
    assert json.loads(report_path.read_text())["created_at_utc"] == created_at


def test_save_experiment_rejects_existing_manifest_membership_change(
    tmp_path: Path,
) -> None:
    predictions = _sentiment_prediction_frame()
    manifest_path = tmp_path / "split.parquet"
    write_sentiment_split_manifest(predictions, manifest_path)
    changed = predictions.copy()
    changed.loc[0, "rating"] = 4.0
    model_directory = tmp_path / "model"

    with pytest.raises(ValueError, match="does not match supplied membership"):
        save_sentiment_experiment(
            vectorizer=SimpleNamespace(),
            classifier=SimpleNamespace(C=0.5),
            split_sample=changed,
            evaluation_predictions=changed,
            metrics={},
            split_config=SentimentSplitConfig(),
            model_config=TfidfBaselineConfig(),
            dataset_version="dataset-v1",
            model_directory=model_directory,
            split_manifest_path=manifest_path,
            predictions_path=tmp_path / "predictions.parquet",
            report_path=tmp_path / "report.json",
        )

    assert not model_directory.exists()


def test_sentiment_metadata_migration_removes_project_absolute_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sentiment_module, "find_project_root", lambda: tmp_path)
    report_path = tmp_path / "reports" / "model.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "created_at_utc": "2026-08-18T00:00:00+00:00",
                "metrics": {"accuracy": 0.8},
                "artifacts": {
                    "vectorizer": str(tmp_path / "models/model/vectorizer.joblib"),
                    "evaluation_predictions": str(
                        tmp_path / "predictions/model.parquet"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    sentiment_module.normalize_sentiment_artifact_metadata(report_path)

    report_text = report_path.read_text(encoding="utf-8")
    migrated = json.loads(report_text)
    assert str(tmp_path) not in report_text
    assert migrated["created_at_utc"] == "2026-08-18T00:00:00+00:00"
    assert migrated["metrics"] == {"accuracy": 0.8}
    assert not report_path.with_suffix(".json.tmp").exists()


def _sentiment_prediction_frame() -> pd.DataFrame:
    """Return one complete prediction row for persistence tests."""
    return pd.DataFrame(
        [
            {
                "review_id": "review-1",
                "split_name": "validation",
                "sentiment_label": "positive",
                "rating": 5.0,
                "parent_asin": "product-1",
                "asin": "asin-1",
                "user_id": "user-1",
                "review_timestamp": pd.Timestamp("2023-01-01", tz="UTC"),
                "text_fingerprint": "fingerprint-1",
                "is_niche": False,
                "predicted_label": "positive",
                "model_score": 0.9,
                "score_negative": 0.05,
                "score_neutral": 0.05,
                "score_positive": 0.9,
            }
        ]
    )


def _products_for_bucket_range(start: int, end: int, *, count: int) -> list[str]:
    """Find deterministic identifiers whose DuckDB hash falls in a range."""
    connection = duckdb.connect()
    products = []
    candidate = 0
    try:
        while len(products) < count:
            product = f"P{candidate:06d}"
            bucket = connection.execute(
                "SELECT md5_number_lower(?) % 100", [product]
            ).fetchone()[0]
            if start <= bucket < end:
                products.append(product)
            candidate += 1
    finally:
        connection.close()
    return products
