"""Tests for diverse disagreement selection."""

from __future__ import annotations

from src.analytics.aspect_error_audit import _diverse_take


def test_diverse_take_prefers_new_reviews_and_aspects() -> None:
    candidates = [
        {"review_id": "r1", "aspect_id": "a", "sort_key": (0, -0.9)},
        {"review_id": "r2", "aspect_id": "a", "sort_key": (0, -0.8)},
        {"review_id": "r3", "aspect_id": "b", "sort_key": (0, -0.7)},
    ]
    selected = _diverse_take(candidates, 2, set())
    assert {item["aspect_id"] for item in selected} == {"a", "b"}
    assert len({item["review_id"] for item in selected}) == 2
