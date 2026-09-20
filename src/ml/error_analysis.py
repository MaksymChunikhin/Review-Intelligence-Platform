"""Explain recurring sentiment-model errors on saved prediction rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, log_loss

from src.analytics.seller_attention import add_seller_attention
from src.ml.sentiment import SENTIMENT_LABELS


PREDICTION_COLUMNS = {
    "review_id",
    "split_name",
    "sentiment_label",
    "rating",
    "parent_asin",
    "predicted_label",
    "model_score",
    "score_negative",
    "score_neutral",
    "score_positive",
}
PROBABILITY_COLUMNS = [f"score_{label}" for label in SENTIMENT_LABELS]
MANUAL_LABEL_COLUMNS = (
    "human_text_sentiment",
    "human_rating_matches_text",
    "human_error_cause",
    "human_notes",
)
HUMAN_TEXT_SENTIMENTS = (*SENTIMENT_LABELS, "mixed", "unclear")
HUMAN_RATING_MATCH_VALUES = ("yes", "no", "unclear")


def load_error_analysis_rows(
    predictions_path: str | Path,
    reviews_path: str | Path,
    catalog_path: str | Path,
    *,
    split_names: Sequence[str],
) -> pd.DataFrame:
    """Join saved predictions to review text and Amazon product information."""
    prediction_file = Path(predictions_path)
    review_file = Path(reviews_path)
    catalog_file = Path(catalog_path)
    for file_path in (prediction_file, review_file, catalog_file):
        if not file_path.is_file():
            raise FileNotFoundError(f"Required analysis file does not exist: {file_path}")
    if not split_names:
        raise ValueError("At least one evaluation split is required")

    connection = duckdb.connect()
    try:
        available_columns = {
            row[0]
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(prediction_file)]
            ).fetchall()
        }
        missing = PREDICTION_COLUMNS.difference(available_columns)
        if missing:
            raise ValueError(
                f"Prediction file is missing columns: {sorted(missing)}"
            )
        expected_count = connection.execute(
            """
            SELECT count(*)
            FROM read_parquet(?)
            WHERE split_name IN (SELECT unnest(?))
            """,
            [str(prediction_file), list(split_names)],
        ).fetchone()[0]
        rows = connection.execute(
            """
            SELECT
                predictions.*,
                reviews.review_text,
                reviews.review_title,
                reviews.verified_purchase,
                reviews.helpful_vote,
                catalog.product_title,
                catalog.category_path_text,
                catalog.leaf_category
            FROM read_parquet(?) AS predictions
            INNER JOIN read_parquet(?) AS reviews USING (review_id)
            LEFT JOIN read_parquet(?) AS catalog
                ON predictions.parent_asin = catalog.parent_asin
            WHERE predictions.split_name IN (SELECT unnest(?))
            ORDER BY predictions.split_name, predictions.review_id
            """,
            [
                str(prediction_file),
                str(review_file),
                str(catalog_file),
                list(split_names),
            ],
        ).fetchdf()
    finally:
        connection.close()

    if len(rows) != expected_count:
        raise ValueError(
            "Review or product join changed the number of prediction rows: "
            f"{len(rows):,} != {expected_count:,}"
        )
    if rows["review_text"].isna().any():
        raise ValueError("Some saved predictions could not be joined to review text")
    if rows["product_title"].isna().any():
        raise ValueError("Some saved predictions could not be joined to a product")
    return rows


def add_error_features(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add exact error fields and cautious text-based review indicators."""
    required = PREDICTION_COLUMNS.union(
        {"review_text", "token_count", "was_truncated"}
    )
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Error-analysis rows are missing columns: {sorted(missing)}")

    frame = add_seller_attention(predictions)
    frame["is_error"] = frame["predicted_label"].ne(frame["sentiment_label"])
    frame["error_pair"] = (
        frame["sentiment_label"] + " → " + frame["predicted_label"]
    ).where(frame["is_error"], "correct")
    frame["token_length_group"] = pd.cut(
        frame["token_count"],
        bins=[-1, 16, 32, 64, 128, 255, np.inf],
        labels=["0–16", "17–32", "33–64", "65–128", "129–255", "256+"],
    )

    text = frame["review_text"].fillna("").astype(str)
    lowercase_text = text.str.casefold()
    positive_pattern = (
        r"\b(?:love|loved|great|excellent|amazing|perfect|wonderful|awesome|"
        r"fantastic|recommend|works? well|five stars?)\b"
    )
    negative_pattern = (
        r"\b(?:hate|hated|terrible|awful|worst|useless|disappointed|"
        r"waste of money|does(?:n['’]t| not) work|did(?:n['’]t| not) work|"
        r"one star)\b"
    )
    negated_positive_pattern = (
        r"\b(?:not|never|isn['’]t|wasn['’]t|doesn['’]t|didn['’]t)\s+"
        r"(?:\w+\s+){0,2}(?:good|great|excellent|amazing|perfect|wonderful)\b"
    )
    contrast_pattern = r"\b(?:but|however|although|though|yet|on the other hand)\b"
    delivery_pattern = (
        r"\b(?:shipping|delivery|delivered|package|packaging|arrived|seller|"
        r"amazon|refund|returned?|replacement)\b"
    )

    frame["contains_positive_words"] = lowercase_text.str.contains(
        positive_pattern, regex=True
    ) & ~lowercase_text.str.contains(negated_positive_pattern, regex=True)
    frame["contains_negative_words"] = lowercase_text.str.contains(
        negative_pattern, regex=True
    )
    frame["contains_mixed_opinion_clue"] = (
        frame["contains_positive_words"] & frame["contains_negative_words"]
    ) | lowercase_text.str.contains(contrast_pattern, regex=True)
    frame["mentions_delivery_or_seller"] = lowercase_text.str.contains(
        delivery_pattern, regex=True
    )
    frame["possible_positive_text_low_rating"] = (
        frame["rating"].le(2)
        & frame["contains_positive_words"]
        & ~frame["contains_negative_words"]
    )
    frame["possible_negative_text_high_rating"] = (
        frame["rating"].ge(4)
        & frame["contains_negative_words"]
        & ~frame["contains_positive_words"]
    )
    frame["possible_text_rating_disagreement"] = frame[
        "possible_positive_text_low_rating"
    ] | frame["possible_negative_text_high_rating"]
    frame["no_clear_dictionary_sentiment"] = ~(
        frame["contains_positive_words"] | frame["contains_negative_words"]
    )
    frame["very_short_review"] = frame["token_count"].le(8)
    frame["strong_emotion_clue"] = text.str.count("!").ge(2) | text.str.contains(
        r"\b[A-Z]{4,}\b", regex=True
    )
    return frame


def error_rates_by_group(
    rows: pd.DataFrame, group_columns: str | Sequence[str]
) -> pd.DataFrame:
    """Count errors and model scores for one or more understandable groups."""
    columns = [group_columns] if isinstance(group_columns, str) else list(group_columns)
    required = set(columns).union({"is_error", "model_score"})
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Grouped error calculation is missing: {sorted(missing)}")
    summary = (
        rows.groupby(columns, observed=True, dropna=False)
        .agg(
            review_count=("is_error", "size"),
            error_count=("is_error", "sum"),
            mean_model_score=("model_score", "mean"),
        )
        .reset_index()
    )
    summary["accuracy"] = 1 - summary["error_count"] / summary["review_count"]
    summary["error_rate"] = summary["error_count"] / summary["review_count"]
    return summary.sort_values(columns).reset_index(drop=True)


def class_metric_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Return precision, recall, and F1 for each class in each test group."""
    required = {"split_name", "sentiment_label", "predicted_label"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Class metric calculation is missing: {sorted(missing)}")
    table_rows = []
    for split_name, split_rows in rows.groupby("split_name", sort=True):
        report = classification_report(
            split_rows["sentiment_label"],
            split_rows["predicted_label"],
            labels=SENTIMENT_LABELS,
            output_dict=True,
            zero_division=0,
        )
        for label in SENTIMENT_LABELS:
            table_rows.append(
                {
                    "split_name": split_name,
                    "sentiment_label": label,
                    "precision": float(report[label]["precision"]),
                    "recall": float(report[label]["recall"]),
                    "f1": float(report[label]["f1-score"]),
                    "review_count": int(report[label]["support"]),
                }
            )
    return pd.DataFrame(table_rows)


def calibration_report(
    rows: pd.DataFrame, *, bin_count: int = 10
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Measure whether the largest saved probability matches observed accuracy."""
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2")
    required = {"sentiment_label", "predicted_label", "model_score"}.union(
        PROBABILITY_COLUMNS
    )
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Calibration calculation is missing: {sorted(missing)}")
    probabilities = rows[PROBABILITY_COLUMNS].to_numpy(dtype="float64")
    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction probabilities contain non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Prediction probabilities must be between zero and one")
    probability_sums = probabilities.sum(axis=1)
    if not np.allclose(probability_sums, 1.0, atol=1e-4):
        raise ValueError("Prediction probabilities do not sum to one")
    # Float32 predictions can miss exactly 1.0 by a few billionths. Normalize
    # before log loss so scikit-learn does not treat harmless rounding as bad data.
    probabilities = probabilities / probability_sums[:, None]

    correct = rows["predicted_label"].eq(rows["sentiment_label"])
    bin_edges = np.linspace(0, 1, bin_count + 1)
    bin_ids = pd.cut(
        rows["model_score"], bin_edges, include_lowest=True, duplicates="drop"
    )
    table = (
        pd.DataFrame(
            {
                "score_range": bin_ids,
                "model_score": rows["model_score"],
                "correct": correct,
            }
        )
        .groupby("score_range", observed=True)
        .agg(
            review_count=("correct", "size"),
            mean_model_score=("model_score", "mean"),
            observed_accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    table["absolute_gap"] = (
        table["mean_model_score"] - table["observed_accuracy"]
    ).abs()
    weights = table["review_count"] / len(rows)
    expected_calibration_error = float((weights * table["absolute_gap"]).sum())

    actual_matrix = np.column_stack(
        [rows["sentiment_label"].eq(label).to_numpy() for label in SENTIMENT_LABELS]
    )
    brier_score = float(np.mean(np.sum((probabilities - actual_matrix) ** 2, axis=1)))
    summary: dict[str, float | int] = {
        "review_count": int(len(rows)),
        "accuracy": float(correct.mean()),
        "expected_calibration_error": expected_calibration_error,
        "multiclass_brier_score": brier_score,
        "log_loss": float(
            log_loss(rows["sentiment_label"], probabilities, labels=SENTIMENT_LABELS)
        ),
    }
    for threshold in (0.80, 0.90, 0.95):
        selected = rows["model_score"].ge(threshold)
        key = str(int(threshold * 100))
        summary[f"score_at_least_{key}_count"] = int(selected.sum())
        summary[f"score_at_least_{key}_error_rate"] = (
            float((~correct[selected]).mean()) if selected.any() else float("nan")
        )
    return table, summary


def compare_saved_models(
    primary_rows: pd.DataFrame, baseline_predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare two models only on review IDs present in both saved files."""
    required = {"review_id", "sentiment_label", "predicted_label"}
    for name, frame in (
        ("primary", primary_rows),
        ("baseline", baseline_predictions),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} model rows are missing: {sorted(missing)}")
    comparison = primary_rows[
        ["review_id", "split_name", "sentiment_label", "predicted_label"]
    ].merge(
        baseline_predictions[
            ["review_id", "sentiment_label", "predicted_label"]
        ].rename(
            columns={
                "sentiment_label": "baseline_sentiment_label",
                "predicted_label": "baseline_predicted_label",
            }
        ),
        on="review_id",
        how="inner",
        validate="one_to_one",
    )
    if not comparison["sentiment_label"].eq(
        comparison["baseline_sentiment_label"]
    ).all():
        raise ValueError("The two prediction files disagree on rating-derived labels")
    comparison["primary_correct"] = comparison["predicted_label"].eq(
        comparison["sentiment_label"]
    )
    comparison["baseline_correct"] = comparison["baseline_predicted_label"].eq(
        comparison["sentiment_label"]
    )
    comparison["comparison_result"] = np.select(
        [
            comparison["primary_correct"] & comparison["baseline_correct"],
            comparison["primary_correct"] & ~comparison["baseline_correct"],
            ~comparison["primary_correct"] & comparison["baseline_correct"],
        ],
        [
            "both correct",
            "DistilBERT only correct",
            "TF-IDF only correct",
        ],
        default="both wrong",
    )
    summary = (
        comparison.groupby(["split_name", "comparison_result"], observed=True)
        .size()
        .rename("review_count")
        .reset_index()
    )
    return comparison, summary


def build_manual_review_queue(
    error_rows: pd.DataFrame,
    *,
    rows_per_group: int = 25,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a balanced worksheet for a person to label model-error causes."""
    if rows_per_group <= 0:
        raise ValueError("rows_per_group must be positive")
    required = {
        "is_error",
        "possible_text_rating_disagreement",
        "contains_mixed_opinion_clue",
        "mentions_delivery_or_seller",
        "was_truncated",
        "sentiment_label",
        "model_score",
    }
    missing = required.difference(error_rows.columns)
    if missing:
        raise ValueError(f"Manual-review rows are missing: {sorted(missing)}")
    errors = error_rows.loc[error_rows["is_error"]].copy()
    if errors.empty:
        raise ValueError("Manual review queue cannot be built without model errors")

    errors["suggested_review_group"] = np.select(
        [
            errors["possible_text_rating_disagreement"],
            errors["contains_mixed_opinion_clue"],
            errors["mentions_delivery_or_seller"],
            errors["was_truncated"],
            errors["sentiment_label"].eq("neutral"),
            errors["model_score"].ge(0.95),
        ],
        [
            "possible text–rating disagreement",
            "possible mixed opinion",
            "delivery or seller mentioned",
            "review was shortened",
            "Neutral rating-derived label",
            "high-score model error",
        ],
        default="other model error",
    )
    sampled_parts = []
    for _, group in errors.groupby("suggested_review_group", sort=True):
        sampled_parts.append(
            group.sample(
                n=min(rows_per_group, len(group)),
                random_state=random_state,
            )
        )
    queue = pd.concat(sampled_parts, ignore_index=True).sort_values(
        ["suggested_review_group", "model_score"], ascending=[True, False]
    )
    for column in MANUAL_LABEL_COLUMNS:
        queue[column] = ""
    return queue.reset_index(drop=True)


def preserve_manual_review_labels(
    new_queue: pd.DataFrame,
    existing_queue: pd.DataFrame,
) -> pd.DataFrame:
    """Copy completed human labels into a regenerated review queue."""
    required = {"review_id", *MANUAL_LABEL_COLUMNS}
    for name, frame in (("new", new_queue), ("existing", existing_queue)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{name} manual-review queue is missing: {sorted(missing)}"
            )
        if frame["review_id"].duplicated().any():
            raise ValueError(f"{name} manual-review queue has duplicate review IDs")

    result = new_queue.copy()
    existing_by_review = existing_queue.set_index("review_id")
    for column in MANUAL_LABEL_COLUMNS:
        saved_values = result["review_id"].map(existing_by_review[column])
        result[column] = saved_values.fillna(result[column]).fillna("")
    return result


def summarize_manual_review(manual_review: pd.DataFrame) -> dict[str, Any]:
    """Validate completed human labels and summarize the diagnostic sample."""
    required = {
        "review_id",
        "sentiment_label",
        "predicted_label",
        "suggested_review_group",
        *MANUAL_LABEL_COLUMNS,
    }
    missing = required.difference(manual_review.columns)
    if missing:
        raise ValueError(f"Manual review is missing columns: {sorted(missing)}")
    if manual_review.empty:
        raise ValueError("Manual review cannot be empty")
    if manual_review["review_id"].duplicated().any():
        raise ValueError("Manual review has duplicate review IDs")

    frame = manual_review.copy()
    for column in MANUAL_LABEL_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    incomplete = frame[list(MANUAL_LABEL_COLUMNS)].eq("").any(axis=1)
    if incomplete.any():
        raise ValueError(
            "Manual review is incomplete: "
            f"{int(incomplete.sum())} rows have empty human labels"
        )

    invalid_sentiments = sorted(
        set(frame["human_text_sentiment"]).difference(HUMAN_TEXT_SENTIMENTS)
    )
    if invalid_sentiments:
        raise ValueError(
            f"Manual review has invalid human sentiments: {invalid_sentiments}"
        )
    invalid_rating_matches = sorted(
        set(frame["human_rating_matches_text"]).difference(
            HUMAN_RATING_MATCH_VALUES
        )
    )
    if invalid_rating_matches:
        raise ValueError(
            "Manual review has invalid rating-match answers: "
            f"{invalid_rating_matches}"
        )

    comparable = frame["human_text_sentiment"].isin(SENTIMENT_LABELS)
    comparable_rows = frame.loc[comparable].copy()
    model_matches_human = comparable_rows["predicted_label"].eq(
        comparable_rows["human_text_sentiment"]
    )
    rating_label_matches_human = comparable_rows["sentiment_label"].eq(
        comparable_rows["human_text_sentiment"]
    )
    rating_mismatch_rows = comparable_rows.loc[
        comparable_rows["human_rating_matches_text"].eq("no")
    ]
    model_matches_rating_mismatch = rating_mismatch_rows["predicted_label"].eq(
        rating_mismatch_rows["human_text_sentiment"]
    )

    def value_counts(column: str) -> dict[str, int]:
        return {
            str(label): int(count)
            for label, count in frame[column].value_counts().sort_index().items()
        }

    def agreement(count: int, total: int) -> dict[str, float | int | None]:
        return {
            "comparable_row_count": total,
            "agreement_count": count,
            "agreement_rate": count / total if total else None,
        }

    group_rows = []
    for group_name, group in frame.groupby("suggested_review_group", sort=True):
        group_comparable = group.loc[
            group["human_text_sentiment"].isin(SENTIMENT_LABELS)
        ]
        group_model_matches = group_comparable["predicted_label"].eq(
            group_comparable["human_text_sentiment"]
        )
        group_rating_matches = group_comparable["sentiment_label"].eq(
            group_comparable["human_text_sentiment"]
        )
        explicit_rating_matches = group["human_rating_matches_text"].eq("yes")
        group_rows.append(
            {
                "suggested_review_group": str(group_name),
                "row_count": int(len(group)),
                "model_human_agreement_count": int(group_model_matches.sum()),
                "model_human_agreement_rate": (
                    float(group_model_matches.mean())
                    if len(group_comparable)
                    else None
                ),
                "rating_label_human_agreement_count": int(
                    group_rating_matches.sum()
                ),
                "rating_label_human_agreement_rate": (
                    float(group_rating_matches.mean())
                    if len(group_comparable)
                    else None
                ),
                "human_rating_matches_text_yes_count": int(
                    explicit_rating_matches.sum()
                ),
                "human_rating_matches_text_yes_rate": float(
                    explicit_rating_matches.mean()
                ),
            }
        )

    model_confusion = (
        comparable_rows.groupby(
            ["human_text_sentiment", "predicted_label"], observed=True
        )
        .size()
        .rename("review_count")
        .reset_index()
        .to_dict(orient="records")
    )
    rating_label_confusion = (
        comparable_rows.groupby(
            ["human_text_sentiment", "sentiment_label"], observed=True
        )
        .size()
        .rename("review_count")
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "status": "completed",
        "row_count": int(len(frame)),
        "selection_scope": {
            "source": "stratified sample of model disagreements with rating-derived weak labels",
            "is_representative_accuracy_sample": False,
            "warning": (
                "Agreement rates diagnose apparent model errors and label noise; "
                "they are not unbiased estimates of corpus-wide model accuracy."
            ),
        },
        "human_sentiment_counts": value_counts("human_text_sentiment"),
        "human_rating_matches_text_counts": value_counts(
            "human_rating_matches_text"
        ),
        "human_error_cause_counts": value_counts("human_error_cause"),
        "model_human_agreement": agreement(
            int(model_matches_human.sum()), int(len(comparable_rows))
        ),
        "rating_label_human_agreement": agreement(
            int(rating_label_matches_human.sum()), int(len(comparable_rows))
        ),
        "model_human_agreement_when_rating_marked_mismatch": agreement(
            int(model_matches_rating_mismatch.sum()), int(len(rating_mismatch_rows))
        ),
        "by_suggested_review_group": group_rows,
        "model_human_confusion": model_confusion,
        "rating_label_human_confusion": rating_label_confusion,
    }


def json_ready(value: Any) -> Any:
    """Convert NumPy and pandas scalar values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Interval):
        return str(value)
    if pd.isna(value):
        return None
    return value
