"""Tests for held-out aspect-extraction evaluation sampling."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.aspect_evaluation_sampling import (
    build_aspect_evaluation_sample,
    select_targeted_reviews,
)
from src.schemas.aspects import AspectEvaluationSampleConfig
from tests.test_niche_scope import make_niche


def make_evaluation_config() -> AspectEvaluationSampleConfig:
    """Return a compact evaluation-sampling configuration."""
    return AspectEvaluationSampleConfig(
        evaluation_version="conditioners_extraction_eval_test_v1",
        evaluation_schema_version="aspect_extraction_evaluation_v1",
        taxonomy_version="conditioners_taxonomy_test_v1",
        niche_id="hair_conditioners",
        niche_version="hair_conditioners_v1",
        dataset_version="beauty_test_v1",
        representative_sample_size=2,
        targeted_sample_size=2,
        minimum_words=5,
        representative_minimum_per_stratum=1,
        targeted_minimum_per_aspect=1,
        maximum_reviews_per_product=1,
        deterministic_seed=13,
    )


def write_evaluation_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Write identity-rich inputs with enough aliases for both components."""
    catalog_path = tmp_path / "catalog.parquet"
    reviews_path = tmp_path / "reviews.parquet"
    discovery_path = tmp_path / "discovery.parquet"
    taxonomy_path = tmp_path / "taxonomy.json"
    pd.DataFrame(
        {
            "dataset_version": ["beauty_test_v1"] * 7,
            "parent_asin": [f"P{number}" for number in range(7)],
            "product_title": [f"Product {number}" for number in range(7)],
            "store": ["Store"] * 7,
            "category_path_id": ["amazon-category:conditioners"] * 7,
            "category_path_text": ["Beauty > Conditioners"] * 7,
        }
    ).to_parquet(catalog_path, index=False)
    pd.DataFrame(
        {
            "review_id": [f"R{number}" for number in range(7)],
            "dataset_version": ["beauty_test_v1"] * 7,
            "parent_asin": [f"P{number}" for number in range(7)],
            "asin": [f"A{number}" for number in range(7)],
            "rating": [4.0] * 7,
            "review_timestamp": pd.to_datetime(
                ["2022-01-01"] * 7, utc=True
            ),
            "verified_purchase": [True] * 7,
            "helpful_vote": [0] * 7,
            "review_text": [
                "discovery review with smell and price today",
                *[
                    f"review {number} discusses smell and price today"
                    for number in range(1, 7)
                ],
            ],
        }
    ).to_parquet(reviews_path, index=False)
    pd.DataFrame(
        {
            "review_id": ["R0"],
            "dataset_version": ["beauty_test_v1"],
            "niche_id": ["hair_conditioners"],
            "niche_version": ["hair_conditioners_v1"],
            "sample_schema_version": ["aspect_discovery_sample_v2"],
        }
    ).to_parquet(discovery_path, index=False)
    taxonomy_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "taxonomy_version": "conditioners_taxonomy_test_v1",
                "niche_id": "hair_conditioners",
                "niche_version": "hair_conditioners_v1",
                "dataset_version": "beauty_test_v1",
                "aspects": [
                    {"aspect_id": "scent", "aliases": ["smell"]},
                    {"aspect_id": "price", "aliases": ["price"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return catalog_path, reviews_path, taxonomy_path, discovery_path


def test_targeted_selection_is_balanced_deterministic_and_product_capped() -> None:
    matches = pd.DataFrame(
        [
            ("R1", "P1", "scent"),
            ("R1", "P1", "softness"),
            ("R2", "P1", "scent"),
            ("R3", "P2", "softness"),
            ("R4", "P3", "price"),
            ("R5", "P4", "price"),
            ("R6", "P5", "scent"),
        ],
        columns=["review_id", "parent_asin", "aspect_id"],
    )

    selected, coverage = select_targeted_reviews(
        matches,
        target_size=4,
        minimum_per_aspect=1,
        maximum_reviews_per_product=1,
        deterministic_seed=11,
    )
    repeated, repeated_coverage = select_targeted_reviews(
        matches,
        target_size=4,
        minimum_per_aspect=1,
        maximum_reviews_per_product=1,
        deterministic_seed=11,
    )

    product_by_review = (
        matches.drop_duplicates("review_id")
        .set_index("review_id")["parent_asin"]
        .to_dict()
    )
    assert selected == repeated
    assert coverage == repeated_coverage
    assert min(coverage.values()) >= 1
    assert len({product_by_review[review_id] for review_id in selected}) == 4


def test_build_evaluation_sample_excludes_discovery_and_writes_blind_queue(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "catalog.parquet"
    pd.DataFrame(
        {
            "dataset_version": ["beauty_test_v1"] * 6,
            "parent_asin": [f"P{number}" for number in range(6)],
            "product_title": [f"=Product {number}" for number in range(6)],
            "store": ["Store"] * 6,
            "category_path_id": ["amazon-category:conditioners"] * 6,
            "category_path_text": ["Beauty > Conditioners"] * 6,
        }
    ).to_parquet(catalog_path, index=False)

    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {
            "review_id": [f"R{number}" for number in range(6)],
            "dataset_version": ["beauty_test_v1"] * 6,
            "parent_asin": [f"P{number}" for number in range(6)],
            "asin": [f"A{number}" for number in range(6)],
            "rating": [4.0] * 6,
            "review_timestamp": pd.to_datetime(
                ["2022-01-01"] * 6, utc=True
            ),
            "verified_purchase": [True] * 6,
            "helpful_vote": [0] * 6,
            "review_text": [
                "=the discovery review has a nice smell",
                "=the smell is very pleasant today",
                "=the price is much too high today",
                "=the smell remains pleasant for my hair",
                "=the price seems fair for this conditioner",
                "=the smell and price both work for me",
            ],
        }
    ).to_parquet(reviews_path, index=False)

    discovery_path = tmp_path / "discovery.parquet"
    pd.DataFrame(
        {
            "review_id": ["R0"],
            "dataset_version": ["beauty_test_v1"],
            "niche_id": ["hair_conditioners"],
            "niche_version": ["hair_conditioners_v1"],
            "sample_schema_version": ["aspect_discovery_sample_v2"],
        }
    ).to_parquet(discovery_path, index=False)
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "taxonomy_version": "conditioners_taxonomy_test_v1",
                "niche_id": "hair_conditioners",
                "niche_version": "hair_conditioners_v1",
                "dataset_version": "beauty_test_v1",
                "aspects": [
                    {"aspect_id": "scent", "aliases": ["smell"]},
                    {"aspect_id": "price", "aliases": ["price"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = make_evaluation_config()
    output_path = tmp_path / "evaluation.parquet"
    queue_path = tmp_path / "queue.csv"

    report = build_aspect_evaluation_sample(
        config,
        make_niche(),
        catalog_path,
        reviews_path,
        taxonomy_path,
        discovery_path,
        output_path,
        queue_path,
        memory_limit="512MB",
    )
    sample = pd.read_parquet(output_path)
    queue = pd.read_csv(queue_path, keep_default_na=False)

    assert report.total_review_count == 4
    assert report.discovery_overlap_count == 0
    assert report.component_overlap_count == 0
    assert report.maximum_reviews_per_product == 1
    assert set(sample["review_id"]) <= {"R1", "R2", "R3", "R4", "R5"}
    representative = sample[sample["sampling_component"] == "representative"]
    targeted = sample[sample["sampling_component"] == "targeted"]
    assert representative["sampling_weight"].sum() == 5.0
    assert representative["population_stratum_count"].eq(5).all()
    assert targeted["sampling_weight"].isna().all()
    assert set(targeted["sampling_weight_semantics"]) == {
        "not_applicable_targeted_nonprobability_sample"
    }
    assert report.representative_product_cap_applied is False
    assert report.representative_population_review_count == 5
    assert "must not estimate population prevalence" in (
        report.targeted_sampling_weight_semantics
    )
    assert report.discovery_sample_identity_fields_validated == [
        "dataset_version",
        "niche_id",
        "niche_version",
        "sample_schema_version",
    ]
    assert set(sample["review_text"].str[:1]) == {"="}
    assert set(queue["review_text"].str[:2]) == {"'="}
    assert set(queue["product_title"].str[:2]) == {"'="}
    assert "target_aspect_ids" not in queue.columns
    assert set(queue["annotation_status"]) == {"pending"}
    assert queue["human_aspect_ids"].eq("").all()

    queue.loc[0, "human_aspect_ids"] = "[]"
    queue.loc[0, "human_evidence_json"] = "[]"
    queue.loc[0, "no_supported_aspect"] = "true"
    queue.loc[0, "annotation_status"] = "complete"
    queue.to_csv(queue_path, index=False)
    build_aspect_evaluation_sample(
        config,
        make_niche(),
        catalog_path,
        reviews_path,
        taxonomy_path,
        discovery_path,
        output_path,
        queue_path,
        memory_limit="512MB",
    )
    preserved = pd.read_csv(queue_path, keep_default_na=False)

    assert preserved.loc[0, "annotation_status"] == "complete"
    assert preserved.loc[0, "no_supported_aspect"] == "true"


def test_evaluation_sampling_rejects_cross_artifact_identity_mismatches(
    tmp_path: Path,
) -> None:
    catalog, reviews, taxonomy, discovery = write_evaluation_fixture(tmp_path)
    output = tmp_path / "evaluation.parquet"
    queue = tmp_path / "queue.csv"
    config = make_evaluation_config()

    wrong_niche_config = config.model_copy(update={"niche_id": "other_niche"})
    with pytest.raises(ValueError, match="niche ID"):
        build_aspect_evaluation_sample(
            wrong_niche_config,
            make_niche(),
            catalog,
            reviews,
            taxonomy,
            discovery,
            output,
            queue,
            memory_limit="512MB",
        )

    taxonomy_payload = json.loads(taxonomy.read_text(encoding="utf-8"))
    taxonomy_payload["dataset_version"] = "other_dataset_v1"
    taxonomy.write_text(json.dumps(taxonomy_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="taxonomy dataset_version"):
        build_aspect_evaluation_sample(
            config,
            make_niche(),
            catalog,
            reviews,
            taxonomy,
            discovery,
            output,
            queue,
            memory_limit="512MB",
        )

    taxonomy_payload["dataset_version"] = "beauty_test_v1"
    taxonomy_payload.pop("niche_id")
    taxonomy.write_text(json.dumps(taxonomy_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="taxonomy is missing identity fields"):
        build_aspect_evaluation_sample(
            config,
            make_niche(),
            catalog,
            reviews,
            taxonomy,
            discovery,
            output,
            queue,
            memory_limit="512MB",
        )

    taxonomy_payload["niche_id"] = "hair_conditioners"
    taxonomy.write_text(json.dumps(taxonomy_payload), encoding="utf-8")
    discovery_frame = pd.read_parquet(discovery)
    discovery_frame["dataset_version"] = "other_dataset_v1"
    discovery_frame.to_parquet(discovery, index=False)
    with pytest.raises(ValueError, match="discovery sample dataset_version"):
        build_aspect_evaluation_sample(
            config,
            make_niche(),
            catalog,
            reviews,
            taxonomy,
            discovery,
            output,
            queue,
            memory_limit="512MB",
        )

    discovery_frame = discovery_frame.drop(columns="niche_version")
    discovery_frame.to_parquet(discovery, index=False)
    with pytest.raises(ValueError, match="missing identity columns"):
        build_aspect_evaluation_sample(
            config,
            make_niche(),
            catalog,
            reviews,
            taxonomy,
            discovery,
            output,
            queue,
            memory_limit="512MB",
        )

    assert not output.exists()
    assert not queue.exists()


def test_completed_queue_rejection_preserves_existing_sample_and_queue(
    tmp_path: Path,
) -> None:
    catalog, reviews, taxonomy, discovery = write_evaluation_fixture(tmp_path)
    output = tmp_path / "evaluation.parquet"
    queue = tmp_path / "queue.csv"
    output.write_bytes(b"existing evaluation sample")
    pd.DataFrame(
        [
            {
                "review_id": "FOREIGN",
                "annotation_id": "existing:annotation:0001",
                "human_aspect_ids": "[]",
                "human_evidence_json": "[]",
                "no_supported_aspect": "true",
                "annotation_status": "complete",
                "annotator_notes": "finished",
            }
        ]
    ).to_csv(queue, index=False)
    original_output = output.read_bytes()
    original_queue = queue.read_bytes()

    with pytest.raises(ValueError, match="started annotation queue with new reviews"):
        build_aspect_evaluation_sample(
            make_evaluation_config(),
            make_niche(),
            catalog,
            reviews,
            taxonomy,
            discovery,
            output,
            queue,
            memory_limit="512MB",
        )

    assert output.read_bytes() == original_output
    assert queue.read_bytes() == original_queue
    assert not (tmp_path / "evaluation.tmp.parquet").exists()
    assert not (tmp_path / "queue.tmp.csv").exists()
