"""Contracts for aspect discovery observations and taxonomy proposals."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AspectRecord(BaseModel):
    """Forbid undeclared fields at aspect artifact boundaries."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AspectDiscoveryConfig(AspectRecord):
    """Configure one reproducible Gemini-assisted discovery run."""

    discovery_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    proposed_taxonomy_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    niche_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    niche_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    dataset_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    sample_schema_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    provider: Literal["google_genai"] = "google_genai"
    backend: Literal["vertex_ai", "gemini_api"] = "vertex_ai"
    model: str = Field(min_length=1)
    location: str | None = Field(default=None, min_length=1)
    batch_size: int = Field(ge=1, le=100)
    temperature: float = Field(ge=0, le=2)
    seed: int = Field(ge=0)
    thinking_level: Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"]
    max_output_tokens: int = Field(ge=1)
    max_retries: int = Field(ge=1, le=10)
    retry_base_seconds: float = Field(ge=0, le=60)
    max_validation_repairs: int = Field(ge=0, le=5)
    maximum_ungrounded_mention_share: float = Field(ge=0, le=0.25)
    minimum_candidate_sample_reviews: int = Field(ge=1)
    minimum_candidate_products: int = Field(ge=1)
    maximum_normalization_candidates: int = Field(ge=1, le=1000)

    @model_validator(mode="after")
    def validate_backend_location(self) -> "AspectDiscoveryConfig":
        """Require a location only for the Vertex AI backend."""
        if self.backend == "vertex_ai" and self.location is None:
            raise ValueError("location is required for the Vertex AI backend")
        return self


class DiscoveryReview(AspectRecord):
    """Represent one review supplied to a discovery request."""

    review_id: str = Field(min_length=1)
    parent_asin: str = Field(min_length=1)
    product_title: str | None = None
    rating: float = Field(ge=1, le=5)
    review_year: int = Field(ge=2000, le=2100)
    sampling_weight: float = Field(gt=0)
    review_text: str = Field(min_length=1)


class AspectDiscoveryBatch(AspectRecord):
    """Store one deterministic batch of sampled reviews."""

    discovery_version: str
    niche_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    niche_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    dataset_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    sample_schema_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    batch_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    reviews: list[DiscoveryReview] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_reviews(self) -> "AspectDiscoveryBatch":
        """Require every review to appear once within a request batch."""
        review_ids = [review.review_id for review in self.reviews]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("batch review IDs must be unique")
        return self


class AspectMention(AspectRecord):
    """Represent one grounded concept mention in a review."""

    aspect_label: str = Field(min_length=1, max_length=80)
    topic_label: str = Field(min_length=1, max_length=120)
    topic_type: Literal[
        "attribute",
        "benefit",
        "complaint",
        "use_case",
        "need",
    ]
    customer_phrase: str = Field(min_length=1, max_length=240)


class ReviewAspectObservation(AspectRecord):
    """Return all grounded concept mentions for one supplied review."""

    review_id: str = Field(min_length=1)
    mentions: list[AspectMention]


class AspectDiscoveryBatchResult(AspectRecord):
    """Structured Gemini response for one discovery batch."""

    batch_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    observations: list[ReviewAspectObservation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_observations(self) -> "AspectDiscoveryBatchResult":
        """Require exactly one observation object per returned review ID."""
        review_ids = [item.review_id for item in self.observations]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("result review IDs must be unique")
        return self


class AspectMerge(AspectRecord):
    """Describe one proposed canonical aspect and its source labels."""

    aspect_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    canonical_name: str = Field(min_length=1, max_length=80)
    definition: str = Field(min_length=1, max_length=400)
    parent_group: Literal[
        "performance",
        "hair_outcome",
        "sensory",
        "formulation",
        "packaging",
        "value",
        "suitability",
        "usage",
        "sustainability",
        "other",
    ]
    source_candidate_keys: list[str] = Field(min_length=1)


class ExcludedAspectCandidate(AspectRecord):
    """Record a discovered label rejected from the taxonomy proposal."""

    candidate_key: str = Field(min_length=1)
    reason: Literal[
        "not_product_aspect",
        "too_specific",
        "ambiguous",
        "duplicate_or_noise",
        "other",
    ]
    explanation: str = Field(min_length=1, max_length=300)


class AspectNormalizationResult(AspectRecord):
    """Structured Gemini proposal for merging discovered aspect labels."""

    proposed_taxonomy_version: str
    aspects: list[AspectMerge] = Field(min_length=1)
    excluded_candidates: list[ExcludedAspectCandidate]


class ApprovedAspectDefinition(AspectMerge):
    """Define one human-approved aspect and optional rare aliases."""

    additional_aliases: list[str] = Field(default_factory=list)


class ReviewedTaxonomyDefinition(AspectRecord):
    """Store the structured decisions used to materialize a taxonomy."""

    taxonomy_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    review_schema_version: Literal["aspect_taxonomy_review_v1"]
    reviewed_on: date
    reviewed_by: str = Field(min_length=1)
    aspects: list[ApprovedAspectDefinition] = Field(min_length=1)
    excluded_candidates: list[ExcludedAspectCandidate]

    @model_validator(mode="after")
    def validate_unique_assignments(self) -> "ReviewedTaxonomyDefinition":
        """Reject duplicate IDs, source assignments, and aliases."""
        aspect_ids = [aspect.aspect_id for aspect in self.aspects]
        if len(aspect_ids) != len(set(aspect_ids)):
            raise ValueError("approved aspect IDs must be unique")

        source_keys = [
            key
            for aspect in self.aspects
            for key in aspect.source_candidate_keys
        ]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("approved source candidate keys must be unique")

        excluded_keys = [
            candidate.candidate_key
            for candidate in self.excluded_candidates
        ]
        if len(excluded_keys) != len(set(excluded_keys)):
            raise ValueError("excluded candidate keys must be unique")
        if set(source_keys) & set(excluded_keys):
            raise ValueError("candidate keys cannot be assigned and excluded")

        aliases = [
            alias.casefold()
            for aspect in self.aspects
            for alias in aspect.additional_aliases
        ]
        if len(aliases) != len(set(aliases)):
            raise ValueError("additional aliases must be unique")
        return self


class AspectEvaluationSampleConfig(AspectRecord):
    """Configure a held-out gold sample for aspect-extraction evaluation."""

    evaluation_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    evaluation_schema_version: Literal["aspect_extraction_evaluation_v1"]
    taxonomy_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    niche_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    niche_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    dataset_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    discovery_sample_schema_version: str = Field(
        default="aspect_discovery_sample_v2",
        pattern=r"^[a-z0-9][a-z0-9_]*$",
    )
    representative_sample_size: int = Field(ge=1)
    targeted_sample_size: int = Field(ge=1)
    minimum_words: int = Field(ge=1)
    representative_minimum_per_stratum: int = Field(ge=1)
    targeted_minimum_per_aspect: int = Field(ge=1)
    maximum_reviews_per_product: int = Field(ge=1)
    deterministic_seed: int = Field(ge=0)


class AspectSilverAnnotationConfig(AspectRecord):
    """Configure resumable Gemini silver labels for an extraction evaluation."""

    silver_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    evaluation_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    taxonomy_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    niche_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    niche_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    dataset_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    provider: Literal["google_genai"] = "google_genai"
    backend: Literal["vertex_ai", "gemini_api"] = "gemini_api"
    model: str = Field(min_length=1)
    location: str | None = Field(default=None, min_length=1)
    batch_size: int = Field(ge=1, le=100)
    temperature: float = Field(ge=0, le=2)
    seed: int = Field(ge=0)
    thinking_level: Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"]
    max_output_tokens: int = Field(ge=1)
    max_retries: int = Field(ge=1, le=10)
    retry_base_seconds: float = Field(ge=0, le=60)
    max_validation_repairs: int = Field(ge=0, le=3)

    @model_validator(mode="after")
    def validate_backend_location(self) -> "AspectSilverAnnotationConfig":
        """Require a location only when using Vertex AI."""
        if self.backend == "vertex_ai" and self.location is None:
            raise ValueError("location is required for the Vertex AI backend")
        return self


class SilverAspectPrediction(AspectRecord):
    """Store one taxonomy-constrained aspect label with verbatim evidence."""

    aspect_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    sentiment: Literal["negative", "neutral", "positive", "mixed", "unclear"]
    customer_phrases: list[str] = Field(min_length=1, max_length=5)

    @field_validator("customer_phrases")
    @classmethod
    def validate_unique_phrases(cls, phrases: list[str]) -> list[str]:
        if len(phrases) != len(set(phrases)):
            raise ValueError("silver evidence phrases must be unique")
        return phrases


class SilverReviewPrediction(AspectRecord):
    """Store all constrained aspect labels for one opaque request item."""

    item_id: str = Field(pattern=r"^item_[0-9]{4}$")
    aspects: list[SilverAspectPrediction]
    no_supported_aspect: bool

    @model_validator(mode="before")
    @classmethod
    def merge_duplicate_aspects(cls, value: Any) -> Any:
        """Merge repeated model rows for the same taxonomy aspect safely."""
        if not isinstance(value, dict) or not isinstance(value.get("aspects"), list):
            return value
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for raw in value["aspects"]:
            if not isinstance(raw, dict) or not raw.get("aspect_id"):
                continue
            aspect_id = str(raw["aspect_id"])
            if aspect_id not in merged:
                merged[aspect_id] = dict(raw)
                order.append(aspect_id)
                continue
            current = merged[aspect_id]
            phrases = [
                *current.get("customer_phrases", []),
                *raw.get("customer_phrases", []),
            ]
            current["customer_phrases"] = list(dict.fromkeys(phrases))[:5]
            sentiments = {
                str(current.get("sentiment", "unclear")),
                str(raw.get("sentiment", "unclear")),
            }
            supported = sentiments - {"unclear"}
            current["sentiment"] = (
                next(iter(supported))
                if len(supported) == 1
                else "mixed"
                if len(supported) > 1
                else "unclear"
            )
        normalized = dict(value)
        normalized["aspects"] = [merged[aspect_id] for aspect_id in order]
        return normalized

    @model_validator(mode="after")
    def validate_none_state(self) -> "SilverReviewPrediction":
        aspect_ids = [aspect.aspect_id for aspect in self.aspects]
        if len(aspect_ids) != len(set(aspect_ids)):
            raise ValueError("silver aspect IDs must be unique within an item")
        if self.no_supported_aspect == bool(self.aspects):
            raise ValueError(
                "no_supported_aspect must be true exactly when aspects is empty"
            )
        return self


class AspectSilverBatchResult(AspectRecord):
    """Represent one complete structured silver-annotation response."""

    batch_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    reviews: list[SilverReviewPrediction] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_items(self) -> "AspectSilverBatchResult":
        item_ids = [item.item_id for item in self.reviews]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("silver result item IDs must be unique")
        return self


def load_aspect_evaluation_sample_config(
    path: str | Path,
) -> AspectEvaluationSampleConfig:
    """Load and validate an aspect-extraction evaluation configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return AspectEvaluationSampleConfig.model_validate(payload)


def load_aspect_silver_annotation_config(
    path: str | Path,
) -> AspectSilverAnnotationConfig:
    """Load and validate an aspect silver-annotation configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return AspectSilverAnnotationConfig.model_validate(payload)


def load_aspect_discovery_config(path: str | Path) -> AspectDiscoveryConfig:
    """Load and validate an aspect-discovery configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return AspectDiscoveryConfig.model_validate(payload)


def load_reviewed_taxonomy_definition(
    path: str | Path,
) -> ReviewedTaxonomyDefinition:
    """Load and validate a human-reviewed taxonomy definition."""
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return ReviewedTaxonomyDefinition.model_validate(payload)
