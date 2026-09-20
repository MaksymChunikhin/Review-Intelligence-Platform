"""Canonical Amazon review and product contracts."""

from datetime import datetime, timezone
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class StrictRecord(BaseModel):
    """Forbid undeclared fields at persisted data-contract boundaries."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AmazonReviewRecord(StrictRecord):
    """Represent one processed Amazon review with source provenance."""

    review_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_category: str = Field(min_length=1)
    source_record_index: int = Field(ge=0)
    asin: str = Field(min_length=1)
    parent_asin: str = Field(min_length=1)
    rating: float = Field(ge=1, le=5)
    review_title: str
    review_body: str
    review_text: str = Field(min_length=1)
    review_timestamp: datetime
    source_timestamp_ms: int = Field(ge=0)
    user_id: str | None = None
    verified_purchase: bool | None = None
    helpful_vote: int | None = Field(default=None, ge=0)

    @field_validator("review_timestamp")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        """Reject naive timestamps and normalize aware timestamps to UTC."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("review_timestamp must include a timezone")
        return value.astimezone(timezone.utc)


class AmazonProductRecord(StrictRecord):
    """Represent one Amazon parent product from item metadata."""

    dataset_version: str = Field(min_length=1)
    dataset_category: str = Field(min_length=1)
    source_record_index: int = Field(ge=0)
    parent_asin: str = Field(min_length=1)
    product_title: str | None = Field(default=None, min_length=1)
    main_category: str | None = None
    category_path: list[str] = Field(default_factory=list)
    store: str | None = None
    average_rating: float | None = Field(default=None, ge=0, le=5)
    rating_number: int | None = Field(default=None, ge=0)
    features: list[str] = Field(default_factory=list)
    description: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    price_at_collection: float | None = Field(default=None, ge=0)
    images: list[dict[str, Any]] = Field(default_factory=list)
    videos: list[dict[str, Any]] = Field(default_factory=list)
    bought_together: list[str] = Field(default_factory=list)
    metadata_quality_flags: list[str] = Field(default_factory=list)


class AmazonEnrichedReviewRecord(StrictRecord):
    """Join one canonical review to its parent product metadata."""

    review: AmazonReviewRecord
    product: AmazonProductRecord

    @model_validator(mode="after")
    def validate_join_identity(self) -> "AmazonEnrichedReviewRecord":
        """Require review and product records to share dataset and parent ID."""
        if self.review.dataset_version != self.product.dataset_version:
            raise ValueError("review and product dataset versions must match")
        if self.review.dataset_category != self.product.dataset_category:
            raise ValueError("review and product dataset categories must match")
        if self.review.parent_asin != self.product.parent_asin:
            raise ValueError("review and product parent_asin values must match")
        return self
