"""Tests for deterministic aspect-discovery sampling."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.aspect_sampling import (
    allocate_stratum_quotas,
    build_aspect_discovery_sample,
)
from tests.test_niche_scope import make_niche, write_scope_fixture


def test_allocate_stratum_quotas_preserves_floor_and_total() -> None:
    quotas = allocate_stratum_quotas(
        {"large": 100, "medium": 10, "small": 2},
        sample_size=30,
        minimum_per_stratum=3,
    )

    assert sum(quotas.values()) == 30
    assert quotas["small"] == 2
    assert quotas["medium"] >= 3
    assert quotas["large"] > quotas["medium"]


def test_allocate_stratum_quotas_rejects_impossible_floor() -> None:
    with pytest.raises(ValueError, match="too small"):
        allocate_stratum_quotas(
            {"a": 10, "b": 10},
            sample_size=3,
            minimum_per_stratum=2,
        )


def test_aspect_sample_is_deterministic_stratified_and_traceable(
    tmp_path: Path,
) -> None:
    catalog, reviews, _ = write_scope_fixture(tmp_path)
    output_one = tmp_path / "sample_one.parquet"
    output_two = tmp_path / "sample_two.parquet"

    with pytest.warns(DeprecationWarning, match="deprecated and ignored"):
        report = build_aspect_discovery_sample(
            make_niche(),
            catalog,
            reviews,
            output_one,
            sample_size=4,
            minimum_words=5,
            minimum_per_stratum=1,
            maximum_reviews_per_product=1,
            deterministic_seed=7,
            memory_limit="512MB",
        )
    build_aspect_discovery_sample(
        make_niche(),
        catalog,
        reviews,
        output_two,
        sample_size=4,
        minimum_words=5,
        minimum_per_stratum=1,
        deterministic_seed=7,
        memory_limit="512MB",
    )
    sample_one = pd.read_parquet(output_one)
    sample_two = pd.read_parquet(output_two)

    assert sample_one["review_id"].tolist() == sample_two["review_id"].tolist()
    assert sample_one["review_id"].is_unique
    assert sample_one["word_count"].min() >= 5
    assert sample_one["stratum_id"].nunique() == 4
    assert report.population_review_count == 5
    assert report.eligible_review_count == 4
    assert report.excluded_short_review_count == 1
    assert report.sample_product_count == 2
    assert report.actual_sample_size == 4
    assert report.product_cap_applied is False
    assert report.maximum_sample_reviews_per_product == 3
    assert report.sampling_design == (
        "stratified_simple_random_sampling_without_replacement"
    )
    assert sample_one["sampling_weight"].sum() == report.eligible_review_count
    assert set(sample_one["sampling_weight_semantics"]) == {
        "inverse_review_inclusion_probability"
    }
    assert set(sample_one["dataset_version"]) == {"beauty_test_v1"}
    assert set(sample_one["niche_version"]) == {"hair_conditioners_v1"}
    assert set(sample_one["sample_schema_version"]) == {
        "aspect_discovery_sample_v2"
    }
    assert report.sample_schema_version == "aspect_discovery_sample_v2"
    assert report.output_sha256
