"""Adapt Amazon Reviews 2023 source records to canonical contracts."""

from collections.abc import Mapping
from typing import Any

from src.schemas.amazon import AmazonProductRecord


def _optional_number(
    value: Any,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float | None:
    """Return an in-range finite-like number or null for invalid metadata."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number < minimum or (maximum is not None and number > maximum):
        return None
    return number


def _string_list(value: Any) -> list[str]:
    """Normalize a source list to non-empty stripped strings."""
    if not isinstance(value, list):
        return []
    return [
        text
        for item in value
        if (text := str(item).strip())
    ]


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    """Keep structured list entries and discard malformed scalar entries."""
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def transform_amazon_product_record(
    source: Mapping[str, Any],
    *,
    dataset_version: str,
    dataset_category: str,
    source_record_index: int,
) -> AmazonProductRecord:
    """Transform one Amazon item-metadata row into the product contract."""
    category_path = _string_list(source.get("categories"))
    features = _string_list(source.get("features"))
    description = _string_list(source.get("description"))
    store = str(source.get("store") or "").strip() or None
    main_category = str(source.get("main_category") or "").strip() or None
    average_rating = _optional_number(
        source.get("average_rating"), minimum=0, maximum=5
    )
    rating_number_value = _optional_number(
        source.get("rating_number"), minimum=0
    )
    price = _optional_number(source.get("price"), minimum=0)

    quality_flags: list[str] = []
    if main_category is None:
        quality_flags.append("missing_main_category")
    if not category_path:
        quality_flags.append("missing_category_path")
    if store is None:
        quality_flags.append("missing_store")
    if source.get("average_rating") is not None and average_rating is None:
        quality_flags.append("invalid_average_rating")
    if source.get("rating_number") is not None and rating_number_value is None:
        quality_flags.append("invalid_rating_number")
    if source.get("price") is None:
        quality_flags.append("missing_price")
    elif price is None:
        quality_flags.append("invalid_price")

    details = source.get("details")
    return AmazonProductRecord(
        dataset_version=dataset_version,
        dataset_category=dataset_category,
        source_record_index=source_record_index,
        parent_asin=source.get("parent_asin"),
        product_title=source.get("title"),
        main_category=main_category,
        category_path=category_path,
        store=store,
        average_rating=average_rating,
        rating_number=(
            int(rating_number_value)
            if rating_number_value is not None
            else None
        ),
        features=features,
        description=description,
        details=dict(details) if isinstance(details, Mapping) else {},
        price_at_collection=price,
        images=_mapping_list(source.get("images")),
        videos=_mapping_list(source.get("videos")),
        bought_together=_string_list(source.get("bought_together")),
        metadata_quality_flags=quality_flags,
    )
