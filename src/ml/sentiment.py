"""Build and evaluate reproducible rating-derived sentiment baselines."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.common.project import find_project_root


SENTIMENT_LABELS = ("negative", "neutral", "positive")
SENTIMENT_MANIFEST_COLUMNS = (
    "review_id",
    "split_name",
    "sentiment_label",
    "rating",
    "parent_asin",
    "asin",
    "user_id",
    "review_timestamp",
    "text_fingerprint",
    "is_niche",
)


@dataclass(frozen=True)
class SentimentSplitConfig:
    """Configure deterministic product-aware sentiment samples."""

    split_version: str = "beauty_rating_sentiment_split_v1"
    temporal_cutoff: str = "2023-01-01"
    random_state: int = 42
    train_per_class: int = 100_000
    validation_per_class: int = 20_000
    product_test_size: int = 100_000
    temporal_test_size: int = 100_000
    niche_test_size: int = 50_000
    train_product_bucket_end: int = 80
    validation_product_bucket_end: int = 90
    niche_category_path: str = (
        "Beauty & Personal Care > Hair Care > Shampoo & Conditioner > "
        "Conditioners"
    )

    def __post_init__(self) -> None:
        """Reject invalid quotas and product-bucket boundaries."""
        _as_utc_datetime(self.temporal_cutoff)
        quotas = (
            self.train_per_class,
            self.validation_per_class,
            self.product_test_size,
            self.temporal_test_size,
            self.niche_test_size,
        )
        if any(quota <= 0 for quota in quotas):
            raise ValueError("All sentiment sample quotas must be positive")
        if not (
            0
            < self.train_product_bucket_end
            < self.validation_product_bucket_end
            < 100
        ):
            raise ValueError(
                "Product bucket boundaries must satisfy "
                "0 < train < validation < 100"
            )


def _as_utc_datetime(value: str) -> datetime:
    """Parse a split boundary, treating timezone-free values as UTC."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid temporal cutoff: {value!r}") from error
    if pd.isna(timestamp):
        raise ValueError(f"Invalid temporal cutoff: {value!r}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    return timestamp.to_pydatetime()


@dataclass(frozen=True)
class TfidfBaselineConfig:
    """Configure the classical text baseline."""

    model_version: str = "beauty_tfidf_rating_sentiment_1m_v3"
    max_features: int = 100_000
    ngram_min: int = 1
    ngram_max: int = 2
    min_document_frequency: int = 5
    max_document_frequency: float = 0.98
    regularization_c: float = 2.0
    regularization_candidates: tuple[float, ...] = (0.5, 2.0)
    max_iterations: int = 100
    tolerance: float = 1e-3
    random_state: int = 42


def rating_to_sentiment(rating: float) -> str:
    """Map an Amazon star rating to a three-class weak sentiment label."""
    if rating in (1, 2):
        return "negative"
    if rating == 3:
        return "neutral"
    if rating in (4, 5):
        return "positive"
    raise ValueError(f"Rating must be an integer-like value from 1 to 5: {rating}")


def build_sentiment_sample(
    reviews_path: str | Path,
    catalog_path: str | Path,
    *,
    config: SentimentSplitConfig,
) -> pd.DataFrame:
    """Return deterministic, text-deduplicated sentiment experiment rows.

    Historical reviews are grouped by parent product into train, validation,
    and unseen-product test pools. The temporal pool contains 2023 reviews for
    products represented in the historical training pool. Exact normalized
    review text that crosses a split boundary is purged, and only one copy of a
    repeated text is retained inside each split.
    """
    source_path = Path(reviews_path)
    product_path = Path(catalog_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Canonical reviews do not exist: {source_path}")
    if not product_path.is_file():
        raise FileNotFoundError(f"Product catalog does not exist: {product_path}")
    temporal_cutoff_utc = _as_utc_datetime(config.temporal_cutoff)

    query = """
        WITH catalog_scope AS (
            SELECT
                parent_asin,
                category_path_text = ? AS is_niche
            FROM read_parquet(?)
        ),
        labeled AS (
            SELECT
                reviews.review_id,
                reviews.parent_asin,
                reviews.asin,
                reviews.user_id,
                reviews.rating,
                reviews.review_timestamp,
                reviews.review_text,
                CASE
                    WHEN reviews.rating IN (1, 2) THEN 'negative'
                    WHEN reviews.rating = 3 THEN 'neutral'
                    ELSE 'positive'
                END AS sentiment_label,
                md5(
                    lower(
                        trim(regexp_replace(reviews.review_text, '\\s+', ' ', 'g'))
                    )
                ) AS text_fingerprint,
                md5_number_lower(reviews.parent_asin) % 100 AS product_bucket,
                coalesce(catalog_scope.is_niche, false) AS is_niche
            FROM read_parquet(?) AS reviews
            LEFT JOIN catalog_scope USING (parent_asin)
        ),
        historical_train_products AS (
            SELECT DISTINCT parent_asin
            FROM labeled
            WHERE review_timestamp < CAST(? AS TIMESTAMPTZ)
              AND product_bucket < ?
        ),
        raw_candidates AS (
            SELECT
                labeled.*,
                CASE
                    WHEN review_timestamp < CAST(? AS TIMESTAMPTZ)
                      AND product_bucket < ? THEN 'train'
                    WHEN review_timestamp < CAST(? AS TIMESTAMPTZ)
                      AND product_bucket < ? THEN 'validation'
                    WHEN review_timestamp < CAST(? AS TIMESTAMPTZ)
                      AND is_niche THEN 'test_conditioners'
                    WHEN review_timestamp < CAST(? AS TIMESTAMPTZ)
                      THEN 'test_product'
                    WHEN review_timestamp >= CAST(? AS TIMESTAMPTZ)
                      AND product_bucket < ?
                      AND parent_asin IN (
                          SELECT parent_asin FROM historical_train_products
                      ) THEN 'temporal_candidate'
                END AS split_name
            FROM labeled
        ),
        fingerprint_splits AS (
            SELECT
                text_fingerprint,
                count(DISTINCT split_name) AS split_count
            FROM raw_candidates
            WHERE split_name IS NOT NULL
            GROUP BY text_fingerprint
        ),
        unique_candidates AS (
            SELECT raw_candidates.* EXCLUDE (product_bucket)
            FROM raw_candidates
            JOIN fingerprint_splits USING (text_fingerprint)
            WHERE raw_candidates.split_name IS NOT NULL
              AND fingerprint_splits.split_count = 1
            QUALIFY row_number() OVER (
                PARTITION BY split_name, text_fingerprint
                ORDER BY md5(review_id || ':' || CAST(? AS VARCHAR))
            ) = 1
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY split_name, sentiment_label
                    ORDER BY md5(review_id || ':class:' || CAST(? AS VARCHAR))
                ) AS class_rank,
                row_number() OVER (
                    PARTITION BY split_name
                    ORDER BY md5(review_id || ':natural:' || CAST(? AS VARCHAR))
                ) AS natural_rank
            FROM unique_candidates
        ),
        selected_historical AS (
            SELECT *
            FROM ranked
            WHERE (split_name = 'train' AND class_rank <= ?)
               OR (split_name = 'validation' AND class_rank <= ?)
               OR (split_name = 'test_product' AND natural_rank <= ?)
               OR (split_name = 'test_conditioners' AND natural_rank <= ?)
        ),
        selected_train_products AS (
            SELECT DISTINCT parent_asin
            FROM selected_historical
            WHERE split_name = 'train'
        ),
        ranked_temporal AS (
            SELECT
                ranked.* REPLACE ('test_temporal_seen' AS split_name),
                row_number() OVER (
                    ORDER BY md5(review_id || ':temporal:' || CAST(? AS VARCHAR))
                ) AS temporal_rank
            FROM ranked
            JOIN selected_train_products USING (parent_asin)
            WHERE ranked.split_name = 'temporal_candidate'
        ),
        selected AS (
            SELECT * EXCLUDE (temporal_rank)
            FROM ranked_temporal
            WHERE temporal_rank <= ?
            UNION ALL BY NAME
            SELECT * FROM selected_historical
        )
        SELECT
            review_id,
            parent_asin,
            asin,
            user_id,
            rating,
            review_timestamp,
            review_text,
            sentiment_label,
            text_fingerprint,
            split_name,
            is_niche
        FROM selected
        ORDER BY split_name, review_id
    """
    parameters = [
        config.niche_category_path,
        str(product_path),
        str(source_path),
        temporal_cutoff_utc,
        config.train_product_bucket_end,
        temporal_cutoff_utc,
        config.train_product_bucket_end,
        temporal_cutoff_utc,
        config.validation_product_bucket_end,
        temporal_cutoff_utc,
        temporal_cutoff_utc,
        temporal_cutoff_utc,
        config.train_product_bucket_end,
        config.random_state,
        config.random_state,
        config.random_state,
        config.train_per_class,
        config.validation_per_class,
        config.product_test_size,
        config.niche_test_size,
        config.random_state,
        config.temporal_test_size,
    ]
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute("SET threads = 8")
        connection.execute("SET memory_limit = '12GB'")
        connection.execute("SET preserve_insertion_order = false")
        sample = connection.execute(query, parameters).fetchdf()
    finally:
        connection.close()

    _validate_sample_quotas(sample, config)
    if sample["review_id"].duplicated().any():
        raise ValueError("Sentiment split contains duplicate review_id values")
    if sample["text_fingerprint"].duplicated().any():
        raise ValueError("Sentiment split contains repeated normalized review text")
    return sample


def _validate_sample_quotas(
    sample: pd.DataFrame, config: SentimentSplitConfig
) -> None:
    """Require requested balanced quotas and non-empty evaluation pools."""
    counts = sample.groupby(["split_name", "sentiment_label"]).size()
    for split_name, quota in (
        ("train", config.train_per_class),
        ("validation", config.validation_per_class),
    ):
        for label in SENTIMENT_LABELS:
            actual = int(counts.get((split_name, label), 0))
            if actual != quota:
                raise ValueError(
                    f"Insufficient {split_name}/{label} rows: {actual} != {quota}"
                )
    for split_name in (
        "test_product",
        "test_temporal_seen",
        "test_conditioners",
    ):
        if not (sample["split_name"] == split_name).any():
            raise ValueError(f"Sentiment evaluation split is empty: {split_name}")


def audit_sentiment_sample(sample: pd.DataFrame) -> dict[str, Any]:
    """Summarize split sizes and train-overlap checks."""
    required = {
        "review_id",
        "parent_asin",
        "user_id",
        "review_timestamp",
        "sentiment_label",
        "text_fingerprint",
        "split_name",
    }
    missing = required.difference(sample.columns)
    if missing:
        raise ValueError(f"Sentiment sample is missing columns: {sorted(missing)}")

    train = sample.loc[sample["split_name"] == "train"]
    train_products = set(train["parent_asin"])
    train_users = set(train["user_id"].dropna())
    train_fingerprints = set(train["text_fingerprint"])
    split_rows = []
    for split_name, group in sample.groupby("split_name", sort=True):
        group_users = set(group["user_id"].dropna())
        split_rows.append(
            {
                "split_name": split_name,
                "review_count": int(len(group)),
                "product_count": int(group["parent_asin"].nunique()),
                "user_count": int(group["user_id"].nunique()),
                "train_product_overlap": int(
                    len(train_products.intersection(group["parent_asin"]))
                ),
                "train_user_overlap": int(len(train_users.intersection(group_users))),
                "train_text_overlap": int(
                    len(train_fingerprints.intersection(group["text_fingerprint"]))
                ),
                "min_timestamp": group["review_timestamp"].min().isoformat(),
                "max_timestamp": group["review_timestamp"].max().isoformat(),
            }
        )
    return {
        "review_id_is_unique": bool(sample["review_id"].is_unique),
        "text_fingerprint_is_unique": bool(sample["text_fingerprint"].is_unique),
        "splits": split_rows,
    }


def write_sentiment_split_manifest(
    sample: pd.DataFrame, path: str | Path
) -> Path:
    """Persist experiment membership without copying review text."""
    manifest_path = Path(path)
    missing = set(SENTIMENT_MANIFEST_COLUMNS).difference(sample.columns)
    if missing:
        raise ValueError(f"Sentiment sample is missing columns: {sorted(missing)}")
    if sample["review_id"].duplicated().any():
        raise ValueError("Sentiment split contains duplicate review_id values")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    sample[list(SENTIMENT_MANIFEST_COLUMNS)].to_parquet(
        temporary_path, index=False
    )
    os.replace(temporary_path, manifest_path)
    return manifest_path


def load_sentiment_split_reviews(
    reviews_path: str | Path,
    split_manifest_path: str | Path,
    *,
    split_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Join saved split identifiers to canonical review text."""
    source_path = Path(reviews_path)
    manifest_path = Path(split_manifest_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Canonical reviews do not exist: {source_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")

    where_clause = ""
    parameters: list[Any] = [str(manifest_path), str(source_path)]
    if split_names:
        where_clause = "WHERE split_manifest.split_name IN (SELECT * FROM unnest(?))"
        parameters.append(list(split_names))
    query = f"""
        SELECT
            split_manifest.*,
            reviews.review_text
        FROM read_parquet(?) AS split_manifest
        JOIN read_parquet(?) AS reviews USING (review_id)
        {where_clause}
        ORDER BY split_manifest.split_name, split_manifest.review_id
    """
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute("SET memory_limit = '12GB'")
        result = connection.execute(query, parameters).fetchdf()
    finally:
        connection.close()
    if result["review_id"].duplicated().any():
        raise ValueError("Joined sentiment reviews contain duplicate review IDs")
    return result


def train_tfidf_baseline(
    train_reviews: pd.DataFrame,
    *,
    config: TfidfBaselineConfig,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Fit one TF-IDF vectorizer and multinomial logistic classifier."""
    required = {"review_text", "sentiment_label"}
    missing = required.difference(train_reviews.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    if set(train_reviews["sentiment_label"].unique()) != set(SENTIMENT_LABELS):
        raise ValueError("Training data must contain all sentiment classes")

    vectorizer = TfidfVectorizer(
        dtype=np.float32,
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_document_frequency,
        max_df=config.max_document_frequency,
        sublinear_tf=True,
        max_features=config.max_features,
    )
    features = vectorizer.fit_transform(train_reviews["review_text"])
    classifier = LogisticRegression(
        C=config.regularization_c,
        max_iter=config.max_iterations,
        random_state=config.random_state,
        solver="saga",
        tol=config.tolerance,
    )
    classifier.fit(features, train_reviews["sentiment_label"])
    return vectorizer, classifier


def select_tfidf_baseline(
    train_reviews: pd.DataFrame,
    validation_reviews: pd.DataFrame,
    *,
    config: TfidfBaselineConfig,
) -> tuple[TfidfVectorizer, LogisticRegression, pd.DataFrame]:
    """Select logistic regularization using only the validation split."""
    if not config.regularization_candidates:
        raise ValueError("At least one regularization candidate is required")
    vectorizer = TfidfVectorizer(
        dtype=np.float32,
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_document_frequency,
        max_df=config.max_document_frequency,
        sublinear_tf=True,
        max_features=config.max_features,
    )
    train_features = vectorizer.fit_transform(train_reviews["review_text"])
    validation_features = vectorizer.transform(validation_reviews["review_text"])
    candidates: list[tuple[LogisticRegression, dict[str, float]]] = []
    for regularization_c in config.regularization_candidates:
        candidate_config = replace(config, regularization_c=regularization_c)
        classifier = LogisticRegression(
            C=candidate_config.regularization_c,
            max_iter=candidate_config.max_iterations,
            random_state=candidate_config.random_state,
            solver="saga",
            tol=candidate_config.tolerance,
        )
        classifier.fit(train_features, train_reviews["sentiment_label"])
        predicted = classifier.predict(validation_features)
        row = {
            "regularization_c": float(regularization_c),
            "accuracy": float(
                accuracy_score(validation_reviews["sentiment_label"], predicted)
            ),
            "macro_f1": float(
                f1_score(
                    validation_reviews["sentiment_label"],
                    predicted,
                    average="macro",
                )
            ),
            "weighted_f1": float(
                f1_score(
                    validation_reviews["sentiment_label"],
                    predicted,
                    average="weighted",
                )
            ),
        }
        candidates.append((classifier, row))
    best_classifier, _ = max(
        candidates,
        key=lambda candidate: (
            candidate[1]["macro_f1"],
            candidate[1]["weighted_f1"],
        ),
    )
    validation_results = pd.DataFrame([row for _, row in candidates]).sort_values(
        ["macro_f1", "weighted_f1"], ascending=False
    )
    return vectorizer, best_classifier, validation_results.reset_index(drop=True)


def predict_sentiment(
    reviews: pd.DataFrame,
    *,
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
) -> pd.DataFrame:
    """Return labels and uncalibrated model scores for review rows."""
    features = vectorizer.transform(reviews["review_text"])
    probabilities = classifier.predict_proba(features)
    predictions = classifier.classes_[np.argmax(probabilities, axis=1)]
    result = reviews.copy()
    result["predicted_label"] = predictions
    result["model_score"] = probabilities.max(axis=1)
    for index, label in enumerate(classifier.classes_):
        result[f"score_{label}"] = probabilities[:, index]
    return result


def balanced_evaluation_view(
    predictions: pd.DataFrame,
    *,
    max_per_class: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a deterministic equal-class view of one evaluation split."""
    available = predictions["sentiment_label"].value_counts()
    if set(available.index) != set(SENTIMENT_LABELS):
        raise ValueError("Evaluation data must contain all sentiment classes")
    per_class = min(int(available.min()), max_per_class)
    parts = [
        predictions.loc[predictions["sentiment_label"] == label].sample(
            n=per_class,
            random_state=random_state,
        )
        for label in SENTIMENT_LABELS
    ]
    return pd.concat(parts, ignore_index=True)


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    bootstrap_rounds: int = 200,
    random_state: int = 42,
    cluster_column: str = "parent_asin",
) -> dict[str, Any]:
    """Calculate metrics and product-cluster bootstrap intervals."""
    actual = predictions["sentiment_label"].to_numpy()
    predicted = predictions["predicted_label"].to_numpy()
    report = classification_report(
        actual,
        predicted,
        labels=list(SENTIMENT_LABELS),
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "review_count": int(len(predictions)),
        "class_distribution": {
            label: int((actual == label).sum()) for label in SENTIMENT_LABELS
        },
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted")),
        "per_class": {
            label: {
                metric: float(report[label][metric])
                for metric in ("precision", "recall", "f1-score")
            }
            for label in SENTIMENT_LABELS
        },
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(SENTIMENT_LABELS)
        ).tolist(),
    }
    if bootstrap_rounds > 0:
        if cluster_column not in predictions.columns:
            raise ValueError(
                "Cluster bootstrap requires prediction column "
                f"{cluster_column!r}"
            )
        clusters = predictions[cluster_column]
        if clusters.isna().any():
            raise ValueError(
                f"Cluster bootstrap column {cluster_column!r} contains null values"
            )
        result["confidence_intervals_95"] = _bootstrap_intervals(
            actual,
            predicted,
            clusters.to_numpy(),
            rounds=bootstrap_rounds,
            random_state=random_state,
        )
        result["confidence_interval_method"] = {
            "name": "percentile_cluster_bootstrap",
            "cluster_column": cluster_column,
            "cluster_count": int(clusters.nunique()),
            "rounds": bootstrap_rounds,
        }
    return result


def _bootstrap_intervals(
    actual: np.ndarray,
    predicted: np.ndarray,
    clusters: np.ndarray,
    *,
    rounds: int,
    random_state: int,
) -> dict[str, list[float]]:
    """Estimate percentile intervals by resampling whole product clusters."""
    rng = np.random.default_rng(random_state)
    cluster_codes, unique_clusters = pd.factorize(clusters, sort=False)
    if len(unique_clusters) == 0:
        raise ValueError("Cluster bootstrap requires at least one cluster")
    label_to_code = {label: code for code, label in enumerate(SENTIMENT_LABELS)}
    actual_codes = np.asarray([label_to_code.get(label, -1) for label in actual])
    predicted_codes = np.asarray(
        [label_to_code.get(label, -1) for label in predicted]
    )
    if (actual_codes < 0).any() or (predicted_codes < 0).any():
        raise ValueError("Cluster bootstrap received an unknown sentiment label")
    cluster_confusions = np.zeros(
        (len(unique_clusters), len(SENTIMENT_LABELS), len(SENTIMENT_LABELS)),
        dtype=np.int64,
    )
    np.add.at(
        cluster_confusions,
        (cluster_codes, actual_codes, predicted_codes),
        1,
    )
    accuracy_values = np.empty(rounds)
    macro_f1_values = np.empty(rounds)
    for index in range(rounds):
        sampled_clusters = rng.integers(
            0, len(unique_clusters), size=len(unique_clusters)
        )
        confusion = cluster_confusions[sampled_clusters].sum(axis=0)
        true_positives = np.diag(confusion).astype(float)
        false_positives = confusion.sum(axis=0) - true_positives
        false_negatives = confusion.sum(axis=1) - true_positives
        accuracy_values[index] = true_positives.sum() / confusion.sum()
        f1_denominators = (
            2 * true_positives + false_positives + false_negatives
        )
        class_f1 = np.divide(
            2 * true_positives,
            f1_denominators,
            out=np.zeros_like(true_positives),
            where=f1_denominators != 0,
        )
        macro_f1_values[index] = class_f1.mean()
    return {
        "accuracy": [
            float(value) for value in np.quantile(accuracy_values, [0.025, 0.975])
        ],
        "macro_f1": [
            float(value) for value in np.quantile(macro_f1_values, [0.025, 0.975])
        ],
    }


def save_sentiment_experiment(
    *,
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    split_sample: pd.DataFrame,
    evaluation_predictions: pd.DataFrame,
    metrics: dict[str, Any],
    split_config: SentimentSplitConfig,
    model_config: TfidfBaselineConfig,
    dataset_version: str,
    model_directory: str | Path,
    split_manifest_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
) -> dict[str, str]:
    """Persist model, split, predictions, configuration, and evaluation report."""
    model_path = Path(model_directory)
    manifest_path = Path(split_manifest_path)
    prediction_path = Path(predictions_path)
    metric_path = Path(report_path)
    if manifest_path.is_file():
        _validate_existing_sentiment_manifest(split_sample, manifest_path)
    else:
        write_sentiment_split_manifest(split_sample, manifest_path)

    model_path.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)

    vectorizer_path = model_path / "vectorizer.joblib"
    classifier_path = model_path / "classifier.joblib"
    experiment_config_path = model_path / "experiment_config.json"
    existing_model_parts = (
        vectorizer_path.is_file(),
        classifier_path.is_file(),
        experiment_config_path.is_file(),
    )
    if any(existing_model_parts) and not all(existing_model_parts):
        raise ValueError("Existing TF-IDF model version is incomplete")
    existing_metadata = None
    if all(existing_model_parts):
        existing_metadata = json.loads(
            experiment_config_path.read_text(encoding="utf-8")
        )
    else:
        _atomic_joblib_dump(vectorizer, vectorizer_path)
        _atomic_joblib_dump(classifier, classifier_path)
    prediction_columns = list(SENTIMENT_MANIFEST_COLUMNS) + [
        "predicted_label",
        "model_score",
        "score_negative",
        "score_neutral",
        "score_positive",
    ]
    temporary_prediction_path = prediction_path.with_suffix(
        prediction_path.suffix + ".tmp"
    )
    evaluation_predictions[prediction_columns].to_parquet(
        temporary_prediction_path, index=False
    )
    os.replace(temporary_prediction_path, prediction_path)

    created_at = (
        str(existing_metadata["created_at_utc"])
        if existing_metadata is not None
        else datetime.now(timezone.utc).isoformat()
    )
    experiment_metadata = {
        "dataset_version": dataset_version,
        "split_config": asdict(split_config),
        "model_config": {
            **asdict(model_config),
            "selected_regularization_c": float(classifier.C),
        },
        "label_definition": {
            "negative": [1, 2],
            "neutral": [3],
            "positive": [4, 5],
            "target_type": "rating-derived weak label",
        },
        "created_at_utc": created_at,
    }
    if existing_metadata is not None:
        normalized_expected = json.loads(json.dumps(experiment_metadata))
        if existing_metadata != normalized_expected:
            raise ValueError(
                "Existing TF-IDF model metadata does not match this experiment"
            )
    else:
        _atomic_json_dump(experiment_metadata, experiment_config_path)
    _atomic_json_dump(
        {
            **experiment_metadata,
            "metrics": metrics,
            "artifacts": {
                "vectorizer": _artifact_reference(vectorizer_path),
                "classifier": _artifact_reference(classifier_path),
                "split_manifest": _artifact_reference(manifest_path),
                "evaluation_predictions": _artifact_reference(prediction_path),
            },
        },
        metric_path,
    )
    return {
        "vectorizer": str(vectorizer_path),
        "classifier": str(classifier_path),
        "experiment_config": str(model_path / "experiment_config.json"),
        "split_manifest": str(manifest_path),
        "evaluation_predictions": str(prediction_path),
        "metrics_report": str(metric_path),
    }


def _validate_existing_sentiment_manifest(
    split_sample: pd.DataFrame,
    manifest_path: Path,
) -> None:
    """Require an existing input manifest to match without rewriting it."""
    expected_columns = list(SENTIMENT_MANIFEST_COLUMNS)
    existing = pd.read_parquet(manifest_path)
    if existing.columns.tolist() != expected_columns:
        raise ValueError(
            "Existing sentiment split manifest has unexpected columns or order"
        )
    missing = set(expected_columns).difference(split_sample.columns)
    if missing:
        raise ValueError(f"Sentiment sample is missing columns: {sorted(missing)}")
    incoming = split_sample[expected_columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            existing.reset_index(drop=True),
            incoming,
            check_exact=True,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as error:
        raise ValueError(
            "Existing sentiment split manifest does not match supplied membership"
        ) from error


def _artifact_reference(path: str | Path) -> str:
    """Return a portable repository-relative reference when possible."""
    artifact_path = Path(path).resolve()
    try:
        project_root = find_project_root()
        return artifact_path.relative_to(project_root).as_posix()
    except (FileNotFoundError, ValueError):
        return str(artifact_path)


def normalize_sentiment_artifact_metadata(
    report_path: str | Path,
) -> str:
    """Atomically migrate persisted TF-IDF artifact paths to portable refs."""
    metric_path = Path(report_path)
    if not metric_path.is_file():
        raise FileNotFoundError(f"Sentiment report does not exist: {metric_path}")
    report = json.loads(metric_path.read_text(encoding="utf-8"))
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Sentiment report has no artifact reference mapping")
    report["artifacts"] = {
        key: _artifact_reference(value)
        if isinstance(value, str) and value
        else value
        for key, value in artifacts.items()
    }
    _atomic_json_dump(report, metric_path)
    return str(metric_path)


def _atomic_joblib_dump(value: Any, path: Path) -> None:
    """Write a joblib artifact through a same-directory temporary file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary_path)
    os.replace(temporary_path, path)


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    """Write formatted JSON through a same-directory temporary file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)
