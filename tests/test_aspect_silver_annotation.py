"""Tests for privacy-minimized Gemini silver annotation."""

from __future__ import annotations

import json

import pytest

from src.llm.aspect_silver_annotation import (
    SilverPlanBatch,
    SilverPlanItem,
    _ground_phrase,
    build_silver_prompt,
    validate_silver_result,
)
from src.schemas.aspects import AspectSilverBatchResult


def batch() -> SilverPlanBatch:
    return SilverPlanBatch(
        silver_version="silver_v1",
        evaluation_version="eval_v1",
        taxonomy_version="taxonomy_v1",
        niche_id="conditioners",
        niche_version="conditioners_v1",
        dataset_version="dataset_v1",
        batch_id="silver_v1_batch_0001",
        items=[
            SilverPlanItem(
                item_id="item_0001",
                review_id="internal-review-id",
                review_text="Leaves hair soft, but the scent is awful.",
            )
        ],
    )


def taxonomy() -> dict[str, object]:
    return {
        "aspects": [
            {
                "aspect_id": "softness",
                "canonical_name": "Softness",
                "definition": "How soft hair feels.",
            },
            {
                "aspect_id": "scent",
                "canonical_name": "Scent",
                "definition": "Product fragrance.",
            },
        ]
    }


def valid_result() -> AspectSilverBatchResult:
    return AspectSilverBatchResult.model_validate(
        {
            "batch_id": "silver_v1_batch_0001",
            "reviews": [
                {
                    "item_id": "item_0001",
                    "aspects": [
                        {
                            "aspect_id": "softness",
                            "sentiment": "positive",
                            "customer_phrases": ["Leaves hair soft"],
                        },
                        {
                            "aspect_id": "scent",
                            "sentiment": "negative",
                            "customer_phrases": ["scent is awful"],
                        },
                    ],
                    "no_supported_aspect": False,
                }
            ],
        }
    )


def test_prompt_exposes_only_opaque_item_id_and_text() -> None:
    prompt = build_silver_prompt(batch(), taxonomy())
    assert "item_0001" in prompt
    assert "Leaves hair soft" in prompt
    assert "internal-review-id" not in prompt
    assert "user_id" not in prompt
    assert "rating" not in prompt
    assert "product_title" not in prompt


def test_validation_accepts_complete_grounded_taxonomy_result() -> None:
    result = valid_result()
    validate_silver_result(batch(), result, {"softness", "scent"})
    assert result.reviews[0].aspects[0].customer_phrases == ["Leaves hair soft"]


def test_grounding_restores_exact_source_typography() -> None:
    source = "It doesn’t feel heavy. Once it ABSORBS….my hair feels normal."
    assert _ground_phrase(source, "It doesn't feel heavy.") == "It doesn’t feel heavy."
    assert (
        _ground_phrase(source, "Once it ABSORBS…my hair feels normal.")
        == "Once it ABSORBS….my hair feels normal."
    )


def test_grounding_repairs_only_close_contiguous_transcription() -> None:
    source = "The oils are likely messing it up and weighing it down."
    assert _ground_phrase(source, "weighing them down.") == "weighing it down."
    assert _ground_phrase(source, "wonderful floral fragrance") is None


def test_validation_rejects_unknown_aspect_or_paraphrased_evidence() -> None:
    result = valid_result()
    result.reviews[0].aspects[0].aspect_id = "unknown"
    with pytest.raises(ValueError, match="unknown aspect"):
        validate_silver_result(batch(), result, {"softness", "scent"})

    result = valid_result()
    result.reviews[0].aspects[0].customer_phrases = ["makes it silky"]
    with pytest.raises(ValueError, match="not verbatim"):
        validate_silver_result(batch(), result, {"softness", "scent"})


def test_result_requires_consistent_none_state() -> None:
    payload = json.loads(valid_result().model_dump_json())
    payload["reviews"][0]["no_supported_aspect"] = True
    with pytest.raises(ValueError, match="no_supported_aspect"):
        AspectSilverBatchResult.model_validate(payload)
