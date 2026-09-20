"""Tests for diagnostics against a silver aspect reference."""

from __future__ import annotations

from src.analytics.aspect_silver_evaluation import _scores, _weighted_scores


def test_pair_scores() -> None:
    predicted = {("r1", "a"), ("r1", "b")}
    reference = {("r1", "a"), ("r2", "a")}
    result = _scores(predicted, reference)
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["f1"] == 0.5


def test_weighted_pair_scores_use_review_weights() -> None:
    predicted = {("r1", "a"), ("r2", "b")}
    reference = {("r1", "a"), ("r2", "a")}
    result = _weighted_scores(predicted, reference, {"r1": 2.0, "r2": 1.0})
    assert result["true_positive_weight"] == 2.0
    assert result["precision"] == 2 / 3
    assert result["recall"] == 2 / 3
