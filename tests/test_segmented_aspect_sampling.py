"""Tests for family-aware aspect-discovery sampling."""

from pathlib import Path

import pandas as pd

from src.data.segmented_aspect_sampling import (
    build_segmented_aspect_discovery_sample,
)
from src.schemas.workspace import ProductNicheDefinition


def test_segmented_sample_preserves_requested_coverage_and_weights(
    tmp_path: Path,
) -> None:
    niche = ProductNicheDefinition(
        niche_id="hair_section",
        niche_version="hair_section_v1",
        dataset_version="dataset_v1",
        category_registry_schema_version="registry_v1",
        display_name="Hair Section",
        status="approved",
        category_path_ids=["path-a", "path-b"],
        segments=[
            {
                "category_path_id": "path-a",
                "analytical_family_id": "family_a",
                "analytical_family_name": "Family A",
                "competitor_niche_id": "segment_a",
                "competitor_niche_name": "Segment A",
            },
            {
                "category_path_id": "path-b",
                "analytical_family_id": "family_b",
                "analytical_family_name": "Family B",
                "competitor_niche_id": "segment_b",
                "competitor_niche_name": "Segment B",
                "comparison_status": "quarantined",
            },
        ],
    )
    rows = []
    for index in range(10):
        is_a = index < 6
        rows.append(
            {
                "review_id": f"r{index}",
                "dataset_version": "dataset_v1",
                "niche_id": "hair_section",
                "niche_version": "hair_section_v1",
                "parent_asin": f"p{index}",
                "asin": f"a{index}",
                "rating": 5.0,
                "review_timestamp": pd.Timestamp("2023-01-01", tz="UTC"),
                "verified_purchase": True,
                "helpful_vote": 0,
                "review_text": "five useful words for sample review",
                "product_title": "Product",
                "store": "Store",
                "category_path_id": "path-a" if is_a else "path-b",
                "category_path_text": "Path A" if is_a else "Path B",
                "analytical_family_id": "family_a" if is_a else "family_b",
                "analytical_family_name": "Family A" if is_a else "Family B",
                "competitor_niche_id": "segment_a" if is_a else "segment_b",
                "competitor_niche_name": "Segment A" if is_a else "Segment B",
                "comparison_status": "eligible" if is_a else "quarantined",
            }
        )
    source = tmp_path / "reviews.parquet"
    output = tmp_path / "sample.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)

    report = build_segmented_aspect_discovery_sample(
        niche,
        source,
        output,
        sampling_version="sample_v1",
        segment_sample_sizes={"segment_a": 3, "segment_b": 2},
        minimum_per_stratum=1,
        deterministic_seed=7,
    )
    sample = pd.read_parquet(output)

    assert len(sample) == 5
    assert sample["review_id"].is_unique
    assert sample.groupby("competitor_niche_id").size().to_dict() == {
        "segment_a": 3,
        "segment_b": 2,
    }
    assert sample["sampling_weight"].sum() == 10
    assert report.eligible_review_count == 10
    assert report.sample_product_count == 5
    assert report.sample_schema_version == "aspect_discovery_sample_v3"


def test_segmented_sample_excludes_multiple_artifacts(tmp_path: Path) -> None:
    niche = ProductNicheDefinition(
        niche_id="section",
        niche_version="section_v1",
        dataset_version="dataset_v1",
        category_registry_schema_version="registry_v1",
        display_name="Section",
        status="approved",
        category_path_ids=["path"],
    )
    frame = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4"],
            "dataset_version": ["dataset_v1"] * 4,
            "niche_id": ["section"] * 4,
            "niche_version": ["section_v1"] * 4,
            "parent_asin": ["p1", "p2", "p3", "p4"],
            "asin": ["a1", "a2", "a3", "a4"],
            "rating": [5.0] * 4,
            "review_timestamp": [pd.Timestamp("2023-01-01", tz="UTC")] * 4,
            "verified_purchase": [True] * 4,
            "helpful_vote": [0] * 4,
            "review_text": ["five useful words for sample review"] * 4,
            "product_title": ["Product"] * 4,
            "store": ["Store"] * 4,
            "category_path_id": ["path"] * 4,
            "category_path_text": ["Path"] * 4,
            "analytical_family_id": ["section"] * 4,
            "analytical_family_name": ["Section"] * 4,
            "competitor_niche_id": ["section"] * 4,
            "competitor_niche_name": ["Section"] * 4,
            "comparison_status": ["eligible"] * 4,
        }
    )
    source = tmp_path / "reviews.parquet"
    excluded_one = tmp_path / "excluded_one.parquet"
    excluded_two = tmp_path / "excluded_two.parquet"
    output = tmp_path / "sample.parquet"
    frame.to_parquet(source, index=False)
    pd.DataFrame({"review_id": ["r1"]}).to_parquet(excluded_one, index=False)
    pd.DataFrame({"review_id": ["r2"]}).to_parquet(excluded_two, index=False)

    report = build_segmented_aspect_discovery_sample(
        niche,
        source,
        output,
        sampling_version="sample_v1",
        segment_sample_sizes={"section": 2},
        minimum_per_stratum=1,
        exclude_review_ids_path=[excluded_one, excluded_two],
    )

    assert set(pd.read_parquet(output)["review_id"]) == {"r3", "r4"}
    assert report.excluded_review_count == 2
