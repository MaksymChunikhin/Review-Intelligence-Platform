"""Tests for dataset manifest validation and file verification."""

import json
from pathlib import Path

from src.ingestion.dataset_manifest import (
    load_dataset_manifest,
    manifest_is_valid,
    sha256_file,
    verify_dataset_manifest,
)


def test_manifest_verifies_size_checksum_and_record_count(tmp_path: Path) -> None:
    source = tmp_path / "reviews.jsonl"
    source.write_text('{"rating": 5}\n{"rating": 1}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": "1.0",
                "dataset_version": "test_dataset_v1",
                "source_name": "Test source",
                "source_url": "https://example.com",
                "source_snapshot": "test",
                "dataset_category": "Beauty_and_Personal_Care",
                "date_window": {"start": "2023-01-01", "end": "2023-01-31"},
                "schema_versions": {
                    "canonical_reviews": "amazon_review_v1",
                    "amazon_products": "amazon_product_v1",
                    "enriched_reviews": "amazon_enriched_review_v1",
                    "category_registry": "amazon_category_registry_v1",
                    "product_catalog": "amazon_product_catalog_v1",
                },
                "default_analysis_unit": "parent_asin",
                "files": [
                    {
                        "role": "filtered_reviews",
                        "path": source.name,
                        "format": "jsonl",
                        "compression": "none",
                        "size_bytes": source.stat().st_size,
                        "sha256": sha256_file(source),
                        "record_count": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_dataset_manifest(manifest_path)
    results = verify_dataset_manifest(
        manifest,
        project_root=tmp_path,
        verify_checksums=True,
        verify_record_counts=True,
    )

    assert manifest_is_valid(results)
    assert results[0]["actual_record_count"] == 2
