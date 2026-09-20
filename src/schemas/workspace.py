"""Versioned seller, competitor, and product-niche scope contracts."""

from datetime import date
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkspaceRecord(BaseModel):
    """Forbid undeclared fields in saved analytical scope configuration."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProductNicheSegmentDefinition(WorkspaceRecord):
    """Map one exact Amazon path to analytical and comparison scopes."""

    category_path_id: str = Field(min_length=1)
    analytical_family_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    analytical_family_name: str = Field(min_length=1)
    competitor_niche_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    competitor_niche_name: str = Field(min_length=1)
    comparison_status: Literal["eligible", "quarantined"] = "eligible"


class ProductNicheDefinition(WorkspaceRecord):
    """Define a reproducible niche as reviewed category path IDs."""

    niche_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    niche_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    dataset_version: str = Field(min_length=1)
    category_registry_schema_version: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    status: Literal["candidate", "approved"]
    category_path_ids: list[str] = Field(min_length=1)
    segments: list[ProductNicheSegmentDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_paths(self) -> "ProductNicheDefinition":
        """Reject duplicate category paths that would obscure scope counts."""
        if len(self.category_path_ids) != len(set(self.category_path_ids)):
            raise ValueError("category_path_ids must be unique")
        if self.segments:
            segment_paths = [segment.category_path_id for segment in self.segments]
            if len(segment_paths) != len(set(segment_paths)):
                raise ValueError("segment category_path_id values must be unique")
            configured_paths = set(self.category_path_ids)
            segmented_paths = set(segment_paths)
            if segmented_paths != configured_paths:
                missing = sorted(configured_paths - segmented_paths)
                unexpected = sorted(segmented_paths - configured_paths)
                raise ValueError(
                    "segments must cover every category_path_id exactly once: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        return self

    def segment_records(self) -> list[dict[str, str]]:
        """Return complete path mappings, including legacy single-scope configs."""
        if self.segments:
            return [segment.model_dump() for segment in self.segments]
        return [
            {
                "category_path_id": path_id,
                "analytical_family_id": self.niche_id,
                "analytical_family_name": self.display_name,
                "competitor_niche_id": self.niche_id,
                "competitor_niche_name": self.display_name,
                "comparison_status": "eligible",
            }
            for path_id in self.category_path_ids
        ]


def load_product_niche_definition(
    path: str | Path,
) -> ProductNicheDefinition:
    """Load and validate one versioned product-niche configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return ProductNicheDefinition.model_validate(payload)


class SellerWorkspaceScope(WorkspaceRecord):
    """Define explicit seller and competitor products for one niche analysis."""

    workspace_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    workspace_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    dataset_version: str = Field(min_length=1)
    niche: ProductNicheDefinition
    seller_parent_asins: list[str] = Field(min_length=1)
    competitor_parent_asins: list[str] = Field(min_length=1)
    review_date_start: date
    review_date_end: date
    verified_purchase_only: bool = False
    pipeline_version: str | None = None
    requested_top_k: int | None = Field(default=None, ge=1)
    requested_minimum_review_count: int | None = Field(default=None, ge=1)
    actual_minimum_review_count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_scope_identity(self) -> "SellerWorkspaceScope":
        """Require consistent dataset, dates, unique products, and disjoint sets."""
        if self.dataset_version != self.niche.dataset_version:
            raise ValueError("workspace and niche dataset versions must match")
        if self.review_date_start > self.review_date_end:
            raise ValueError("review_date_start must not be after review_date_end")
        seller_products = set(self.seller_parent_asins)
        competitor_products = set(self.competitor_parent_asins)
        if len(seller_products) != len(self.seller_parent_asins):
            raise ValueError("seller_parent_asins must be unique")
        if len(competitor_products) != len(self.competitor_parent_asins):
            raise ValueError("competitor_parent_asins must be unique")
        if seller_products & competitor_products:
            raise ValueError("seller and competitor product sets must be disjoint")
        return self


def load_seller_workspace_scope(
    path: str | Path,
) -> SellerWorkspaceScope:
    """Load and validate one seller-versus-competitor workspace."""
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return SellerWorkspaceScope.model_validate(payload)
