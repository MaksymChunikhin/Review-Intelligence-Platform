"""Test reusable sentiment error-analysis calculations."""

import pandas as pd
import pytest

from src.ml.error_analysis import (
    add_error_features,
    build_manual_review_queue,
    calibration_report,
    class_metric_table,
    compare_saved_models,
    error_rates_by_group,
    preserve_manual_review_labels,
    summarize_manual_review,
)


def prediction_fixture() -> pd.DataFrame:
    """Return a small three-class prediction table with known errors."""
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4"],
            "split_name": ["test_product"] * 4,
            "sentiment_label": ["positive", "negative", "neutral", "positive"],
            "rating": [5.0, 1.0, 3.0, 5.0],
            "parent_asin": ["p1", "p2", "p3", "p4"],
            "predicted_label": ["positive", "positive", "negative", "negative"],
            "model_score": [0.8, 0.7, 0.6, 0.6],
            "score_negative": [0.1, 0.2, 0.6, 0.6],
            "score_neutral": [0.1, 0.1, 0.3, 0.2],
            "score_positive": [0.8, 0.7, 0.1, 0.2],
            "review_text": [
                "Works well and I love it",
                "Great product!",
                "It is fine, but the package arrived late.",
                "WORST purchase!!",
            ],
            "token_count": [6, 2, 10, 3],
            "was_truncated": [False, False, False, True],
        }
    )


def test_add_error_features_marks_exact_and_possible_causes() -> None:
    enriched = add_error_features(prediction_fixture())

    assert enriched["is_error"].tolist() == [False, True, True, True]
    assert enriched["seller_attention"].tolist() == [
        "satisfied",
        "needs_attention",
        "needs_attention",
        "satisfied",
    ]
    assert enriched.loc[1, "possible_positive_text_low_rating"]
    assert enriched.loc[2, "contains_mixed_opinion_clue"]
    assert enriched.loc[2, "mentions_delivery_or_seller"]
    assert enriched.loc[3, "possible_negative_text_high_rating"]
    assert enriched.loc[3, "strong_emotion_clue"]


def test_grouped_error_rates_have_expected_values() -> None:
    enriched = add_error_features(prediction_fixture())
    summary = error_rates_by_group(enriched, "split_name")
    class_metrics = class_metric_table(enriched)

    assert summary.loc[0, "review_count"] == 4
    assert summary.loc[0, "error_count"] == 3
    assert summary.loc[0, "accuracy"] == pytest.approx(0.25)
    assert set(class_metrics["sentiment_label"]) == {
        "negative",
        "neutral",
        "positive",
    }
    assert class_metrics["review_count"].sum() == 4


def test_calibration_report_measures_gap_and_probability_scores() -> None:
    enriched = add_error_features(prediction_fixture())
    table, summary = calibration_report(enriched, bin_count=5)

    assert table["review_count"].sum() == 4
    assert summary["accuracy"] == pytest.approx(0.25)
    assert summary["expected_calibration_error"] >= 0
    assert summary["multiclass_brier_score"] >= 0
    assert summary["log_loss"] >= 0


def test_calibration_rejects_probabilities_that_do_not_sum_to_one() -> None:
    invalid = add_error_features(prediction_fixture())
    invalid.loc[0, "score_positive"] = 0.7

    with pytest.raises(ValueError, match="do not sum to one"):
        calibration_report(invalid)


def test_saved_model_comparison_and_manual_queue() -> None:
    enriched = add_error_features(prediction_fixture())
    baseline = enriched[["review_id", "sentiment_label", "predicted_label"]].copy()
    baseline.loc[1, "predicted_label"] = "negative"
    comparison, summary = compare_saved_models(enriched, baseline)
    queue = build_manual_review_queue(enriched, rows_per_group=2)

    assert len(comparison) == 4
    assert summary["review_count"].sum() == 4
    assert "TF-IDF only correct" in set(comparison["comparison_result"])
    assert 0 < len(queue) <= 6
    assert queue["human_error_cause"].eq("").all()


def test_preserve_manual_review_labels_survives_queue_regeneration() -> None:
    enriched = add_error_features(prediction_fixture())
    queue = build_manual_review_queue(enriched, rows_per_group=2)
    existing = queue.copy()
    existing.loc[0, "human_text_sentiment"] = "negative"
    existing.loc[0, "human_rating_matches_text"] = "no"

    restored = preserve_manual_review_labels(queue, existing)

    assert restored.loc[0, "human_text_sentiment"] == "negative"
    assert restored.loc[0, "human_rating_matches_text"] == "no"


def test_manual_review_summary_compares_human_model_and_weak_labels() -> None:
    manual_review = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4"],
            "sentiment_label": ["positive", "negative", "neutral", "positive"],
            "predicted_label": ["neutral", "negative", "positive", "neutral"],
            "suggested_review_group": ["mixed", "mixed", "other", "other"],
            "human_text_sentiment": ["neutral", "negative", "negative", "mixed"],
            "human_rating_matches_text": ["no", "yes", "no", "unclear"],
            "human_error_cause": [
                "rating-text disagreement",
                "model error avoided",
                "model error",
                "ambiguous text",
            ],
            "human_notes": ["note 1", "note 2", "note 3", "note 4"],
        }
    )

    summary = summarize_manual_review(manual_review)

    assert summary["status"] == "completed"
    assert summary["row_count"] == 4
    assert summary["model_human_agreement"] == {
        "comparable_row_count": 3,
        "agreement_count": 2,
        "agreement_rate": pytest.approx(2 / 3),
    }
    assert summary["rating_label_human_agreement"]["agreement_count"] == 1
    assert summary["model_human_agreement_when_rating_marked_mismatch"] == {
        "comparable_row_count": 2,
        "agreement_count": 1,
        "agreement_rate": pytest.approx(0.5),
    }
    assert not summary["selection_scope"]["is_representative_accuracy_sample"]


def test_manual_review_summary_rejects_incomplete_or_invalid_labels() -> None:
    queue = build_manual_review_queue(
        add_error_features(prediction_fixture()), rows_per_group=2
    )

    with pytest.raises(ValueError, match="incomplete"):
        summarize_manual_review(queue)

    for column in (
        "human_text_sentiment",
        "human_rating_matches_text",
        "human_error_cause",
        "human_notes",
    ):
        queue[column] = "filled"
    queue["human_rating_matches_text"] = "yes"

    with pytest.raises(ValueError, match="invalid human sentiments"):
        summarize_manual_review(queue)
