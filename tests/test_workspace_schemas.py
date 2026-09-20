"""Tests for reproducible niche and seller-workspace contracts."""

from datetime import date

import pytest
from pydantic import ValidationError

from src.schemas.workspace import (
    ProductNicheDefinition,
    SellerWorkspaceScope,
    load_product_niche_definition,
)


def make_niche() -> ProductNicheDefinition:
    """Return one candidate niche for workspace tests."""
    return ProductNicheDefinition(
        niche_id="hair_conditioners",
        niche_version="hair_conditioners_candidate_v1",
        dataset_version="beauty_v1",
        category_registry_schema_version="amazon_category_registry_v1",
        display_name="Hair Conditioners",
        status="candidate",
        category_path_ids=["amazon-category:conditioners"],
    )


def test_workspace_accepts_disjoint_product_sets() -> None:
    scope = SellerWorkspaceScope(
        workspace_id="conditioner_demo",
        workspace_version="conditioner_demo_v1",
        dataset_version="beauty_v1",
        niche=make_niche(),
        seller_parent_asins=["P1"],
        competitor_parent_asins=["P2", "P3"],
        review_date_start=date(2021, 1, 1),
        review_date_end=date(2023, 9, 13),
    )

    assert scope.niche.status == "candidate"
    assert set(scope.seller_parent_asins).isdisjoint(scope.competitor_parent_asins)


def test_workspace_rejects_overlapping_seller_and_competitor_products() -> None:
    with pytest.raises(ValidationError, match="must be disjoint"):
        SellerWorkspaceScope(
            workspace_id="conditioner_demo",
            workspace_version="conditioner_demo_v1",
            dataset_version="beauty_v1",
            niche=make_niche(),
            seller_parent_asins=["P1"],
            competitor_parent_asins=["P1", "P2"],
            review_date_start=date(2021, 1, 1),
            review_date_end=date(2023, 9, 13),
        )


def test_load_product_niche_definition(tmp_path) -> None:
    config_path = tmp_path / "niche.json"
    config_path.write_text(make_niche().model_dump_json(), encoding="utf-8")

    loaded = load_product_niche_definition(config_path)

    assert loaded == make_niche()


def test_niche_segments_must_cover_configured_paths_exactly_once() -> None:
    with pytest.raises(ValidationError, match="segments must cover"):
        ProductNicheDefinition(
            niche_id="hair_care",
            niche_version="hair_care_v1",
            dataset_version="beauty_v1",
            category_registry_schema_version="amazon_category_registry_v1",
            display_name="Hair Care",
            status="approved",
            category_path_ids=["path-1", "path-2"],
            segments=[
                {
                    "category_path_id": "path-1",
                    "analytical_family_id": "cleansing",
                    "analytical_family_name": "Cleansing",
                    "competitor_niche_id": "shampoo",
                    "competitor_niche_name": "Shampoo",
                }
            ],
        )


def test_legacy_niche_gets_one_default_segment_per_path() -> None:
    niche = make_niche()

    assert niche.segment_records() == [
        {
            "category_path_id": "amazon-category:conditioners",
            "analytical_family_id": "hair_conditioners",
            "analytical_family_name": "Hair Conditioners",
            "competitor_niche_id": "hair_conditioners",
            "competitor_niche_name": "Hair Conditioners",
            "comparison_status": "eligible",
        }
    ]
