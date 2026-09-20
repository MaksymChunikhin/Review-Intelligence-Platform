"""Dataset manifest contracts."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ManifestRecord(BaseModel):
    """Forbid unknown fields in versioned dataset manifests."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DateWindow(ManifestRecord):
    """Define an inclusive analytical date window."""

    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> "DateWindow":
        """Require the window start to precede its end."""
        if self.start > self.end:
            raise ValueError("date window start must not be after end")
        return self


class DatasetSchemaVersions(ManifestRecord):
    """Register logical contract versions for persisted analytical records."""

    canonical_reviews: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    amazon_products: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    enriched_reviews: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    category_registry: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    product_catalog: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")


class DatasetFile(ManifestRecord):
    """Identify and fingerprint one dataset artifact."""

    role: Literal[
        "raw_reviews",
        "raw_product_metadata",
        "filtered_reviews",
        "canonical_reviews",
        "category_registry",
        "product_catalog",
    ]
    path: str = Field(min_length=1)
    format: Literal["jsonl", "parquet"]
    compression: Literal["gzip", "snappy", "zstd", "none"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int | None = Field(default=None, ge=0)


class DatasetManifest(ManifestRecord):
    """Describe one immutable, reproducible Amazon dataset version."""

    manifest_version: str = Field(min_length=1)
    dataset_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    source_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_snapshot: str = Field(min_length=1)
    dataset_category: str = Field(min_length=1)
    date_window: DateWindow
    schema_versions: DatasetSchemaVersions
    default_analysis_unit: Literal["parent_asin"] = "parent_asin"
    files: list[DatasetFile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_file_roles(self) -> "DatasetManifest":
        """Require at most one file for every declared artifact role."""
        roles = [file.role for file in self.files]
        if len(roles) != len(set(roles)):
            raise ValueError("dataset file roles must be unique")
        return self

    def file_by_role(self, role: str) -> DatasetFile:
        """Return the file registered for an artifact role."""
        for file in self.files:
            if file.role == role:
                return file
        raise KeyError(f"Manifest does not define file role: {role}")
