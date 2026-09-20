"""Tests for combining nested aspect-discovery waves."""

from pathlib import Path

import pandas as pd

from src.analytics.discovery_combination import combine_discovery_components
from src.schemas.aspects import AspectDiscoveryConfig


def _config() -> AspectDiscoveryConfig:
    return AspectDiscoveryConfig(
        discovery_version="section_discovery_v2",
        proposed_taxonomy_version="section_taxonomy_v1",
        niche_id="section",
        niche_version="section_v1",
        dataset_version="dataset_v1",
        sample_schema_version="sample_v1",
        backend="vertex_ai",
        model="test-model",
        location="eu",
        batch_size=2,
        temperature=0,
        seed=1,
        thinking_level="LOW",
        max_output_tokens=100,
        max_retries=1,
        retry_base_seconds=0,
        max_validation_repairs=0,
        maximum_ungrounded_mention_share=0,
        minimum_candidate_sample_reviews=2,
        minimum_candidate_products=2,
        maximum_normalization_candidates=10,
    )


def _sample(review_ids: list[str], weights: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": review_ids,
            "dataset_version": ["dataset_v1"] * len(review_ids),
            "niche_version": ["section_v1"] * len(review_ids),
            "parent_asin": [f"p{value}" for value in review_ids],
            "rating": [5.0] * len(review_ids),
            "review_year": [2023] * len(review_ids),
            "sampling_weight": weights,
        }
    )


def _observations(review_ids: list[str], version: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "discovery_version": [version] * len(review_ids),
            "batch_id": [f"batch_{value}" for value in review_ids],
            "review_id": review_ids,
            "parent_asin": ["temporary"] * len(review_ids),
            "rating": [1.0] * len(review_ids),
            "review_year": [2021] * len(review_ids),
            "sampling_weight": [1.0] * len(review_ids),
            "aspect_label": ["Scent"] * len(review_ids),
            "candidate_key": ["scent"] * len(review_ids),
            "topic_label": ["smell"] * len(review_ids),
            "topic_type": ["attribute"] * len(review_ids),
            "customer_phrase": ["smells nice"] * len(review_ids),
        }
    )


def test_combination_uses_expanded_sample_weights(tmp_path: Path) -> None:
    base_sample = tmp_path / "base.parquet"
    extension_sample = tmp_path / "extension.parquet"
    expanded_sample = tmp_path / "expanded.parquet"
    base_observations = tmp_path / "base_observations.parquet"
    extension_observations = tmp_path / "extension_observations.parquet"
    _sample(["r1", "r2"], [10.0, 20.0]).to_parquet(base_sample, index=False)
    _sample(["r3"], [30.0]).to_parquet(extension_sample, index=False)
    _sample(["r1", "r2", "r3"], [10.0, 20.0, 30.0]).to_parquet(
        expanded_sample, index=False
    )
    _observations(["r1", "r2"], "v1").to_parquet(
        base_observations, index=False
    )
    _observations(["r3"], "extension").to_parquet(
        extension_observations, index=False
    )
    observations = tmp_path / "combined.parquet"
    candidates = tmp_path / "candidates.json"

    report = combine_discovery_components(
        _config(),
        expanded_sample,
        [
            (base_sample, base_observations),
            (extension_sample, extension_observations),
        ],
        observations,
        candidates,
    )

    combined = pd.read_parquet(observations)
    assert report.review_count == 3
    assert report.weighted_sample_population == 60.0
    assert set(combined["sampling_weight"]) == {10.0, 20.0, 30.0}
    assert set(combined["discovery_version"]) == {"section_discovery_v2"}
