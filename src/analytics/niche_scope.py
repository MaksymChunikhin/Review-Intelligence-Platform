"""Validate a versioned product niche against catalog and review artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import load_dataset_manifest
from src.schemas.workspace import (
    ProductNicheDefinition,
    load_product_niche_definition,
)


@dataclass(frozen=True)
class NichePopulationReport:
    """Record the exact product and review population for one niche version."""

    niche_id: str
    niche_version: str
    niche_status: str
    display_name: str
    dataset_version: str
    category_registry_schema_version: str
    catalog_path: str
    reviews_path: str
    category_registry_path: str
    category_paths: list[dict[str, Any]]
    catalog_product_count: int
    reviewed_product_count: int
    catalog_product_review_coverage: float
    reviewed_variation_asin_count: int
    review_count: int
    distinct_review_parent_asin_count: int
    distinct_review_asin_count: int
    review_date_min: str | None
    review_date_max: str | None
    verified_purchase_count: int
    verified_purchase_share: float
    helpful_review_count: int
    helpful_review_share: float
    rating_distribution: list[dict[str, Any]]
    review_year_distribution: list[dict[str, Any]]
    review_length_distribution: list[dict[str, Any]]
    analytical_family_distribution: list[dict[str, Any]]
    competitor_niche_distribution: list[dict[str, Any]]


def _artifact_reference(path: Path) -> str:
    """Prefer a repository-relative path in persisted provenance."""
    try:
        root = find_project_root(path.parent)
        return path.resolve().relative_to(root).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _records(
    rows: list[tuple[Any, ...]], columns: list[str]
) -> list[dict[str, Any]]:
    """Convert DuckDB result rows to JSON-compatible dictionaries."""
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _distribution(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    """Execute one population distribution query."""
    rows = connection.execute(sql).fetchall()
    columns = [column[0] for column in connection.description]
    return _records(rows, columns)


def profile_niche_population(
    niche: ProductNicheDefinition,
    catalog_path: str | Path,
    reviews_path: str | Path,
    category_registry_path: str | Path,
    *,
    report_path: str | Path | None = None,
    require_approved: bool = True,
    memory_limit: str = "4GB",
) -> NichePopulationReport:
    """Validate and profile the full product/review population of a niche."""
    catalog = Path(catalog_path)
    reviews = Path(reviews_path)
    registry = Path(category_registry_path)
    for source in (catalog, reviews, registry):
        if not source.is_file():
            raise FileNotFoundError(source)
    if require_approved and niche.status != "approved":
        raise ValueError("niche must be approved before population validation")

    connection = duckdb.connect()
    escaped_memory_limit = memory_limit.replace("'", "''")
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute(f"SET memory_limit = '{escaped_memory_limit}'")
        registry_rows = connection.execute(
            """
            SELECT
                category_path_id,
                category_path_text,
                product_record_count,
                distinct_parent_asin_count,
                dataset_version,
                registry_schema_version
            FROM read_parquet(?)
            WHERE category_path_id IN (SELECT * FROM unnest(?))
            ORDER BY category_path_id
            """,
            [str(registry), niche.category_path_ids],
        ).fetchall()
        registry_columns = [column[0] for column in connection.description]
        category_paths = _records(registry_rows, registry_columns)
        resolved_path_ids = {row["category_path_id"] for row in category_paths}
        missing_path_ids = set(niche.category_path_ids) - resolved_path_ids
        if missing_path_ids:
            raise ValueError(
                "niche category paths are absent from registry: "
                + ", ".join(sorted(missing_path_ids))
            )
        if any(
            row["dataset_version"] != niche.dataset_version
            for row in category_paths
        ):
            raise ValueError("niche and category registry dataset versions differ")
        if any(
            row["registry_schema_version"]
            != niche.category_registry_schema_version
            for row in category_paths
        ):
            raise ValueError("niche and category registry schema versions differ")

        connection.register(
            "configured_niche_segments",
            pd.DataFrame(niche.segment_records()),
        )

        catalog_identity = connection.execute(
            """
            SELECT
                count(DISTINCT dataset_version),
                min(dataset_version),
                count(*) - count(DISTINCT parent_asin)
            FROM read_parquet(?)
            """,
            [str(catalog)],
        ).fetchone()
        if catalog_identity[0] != 1 or catalog_identity[1] != niche.dataset_version:
            raise ValueError("niche and product catalog dataset versions differ")
        if catalog_identity[2] != 0:
            raise ValueError("product catalog parent_asin values must be unique")

        review_identity = connection.execute(
            """
            SELECT count(DISTINCT dataset_version), min(dataset_version)
            FROM read_parquet(?)
            """,
            [str(reviews)],
        ).fetchone()
        if review_identity[0] != 1 or review_identity[1] != niche.dataset_version:
            raise ValueError("niche and canonical review dataset versions differ")

        connection.execute(
            """
            CREATE TEMP TABLE niche_products AS
            SELECT
                catalog.*,
                segments.analytical_family_id,
                segments.analytical_family_name,
                segments.competitor_niche_id,
                segments.competitor_niche_name,
                segments.comparison_status
            FROM read_parquet(?) AS catalog
            INNER JOIN configured_niche_segments AS segments
                USING (category_path_id)
            WHERE catalog.category_path_id IN (SELECT * FROM unnest(?))
            """,
            [str(catalog), niche.category_path_ids],
        )
        connection.execute(
            """
            CREATE TEMP TABLE niche_reviews AS
            SELECT reviews.*
            FROM read_parquet(?) AS reviews
            INNER JOIN niche_products USING (parent_asin)
            """,
            [str(reviews)],
        )

        product_summary = connection.execute(
            """
            SELECT
                count(*) AS catalog_product_count,
                count(*) FILTER (WHERE review_count > 0)
                    AS reviewed_product_count,
                coalesce(sum(reviewed_asin_count), 0)
                    AS reviewed_variation_asin_count,
                coalesce(sum(review_count), 0) AS catalog_review_count
            FROM niche_products
            """
        ).fetchone()
        review_summary = connection.execute(
            """
            SELECT
                count(*) AS review_count,
                count(DISTINCT parent_asin) AS distinct_review_parent_asin_count,
                count(DISTINCT asin) AS distinct_review_asin_count,
                min(review_timestamp) AS review_date_min,
                max(review_timestamp) AS review_date_max,
                count(*) FILTER (WHERE verified_purchase)
                    AS verified_purchase_count,
                count(*) FILTER (WHERE helpful_vote > 0) AS helpful_review_count
            FROM niche_reviews
            """
        ).fetchone()

        catalog_review_count = int(product_summary[3])
        review_count = int(review_summary[0])
        if catalog_review_count != review_count:
            raise RuntimeError(
                "catalog review counts do not reconcile to selected reviews"
            )
        if int(product_summary[1]) != int(review_summary[1]):
            raise RuntimeError(
                "reviewed catalog products do not reconcile to selected reviews"
            )

        rating_distribution = _distribution(
            connection,
            """
            SELECT rating, count(*) AS review_count
            FROM niche_reviews
            GROUP BY rating
            ORDER BY rating
            """,
        )
        review_year_distribution = _distribution(
            connection,
            """
            SELECT year(review_timestamp) AS review_year,
                   count(*) AS review_count
            FROM niche_reviews
            GROUP BY review_year
            ORDER BY review_year
            """,
        )
        review_length_distribution = _distribution(
            connection,
            """
            WITH lengths AS (
                SELECT array_length(
                    regexp_split_to_array(trim(review_text), '\\s+')
                ) AS word_count
                FROM niche_reviews
            )
            SELECT
                CASE
                    WHEN word_count < 5 THEN 'too_short_1_4'
                    WHEN word_count <= 40 THEN 'short_5_40'
                    WHEN word_count <= 120 THEN 'medium_41_120'
                    ELSE 'long_121_plus'
                END AS length_bucket,
                count(*) AS review_count
            FROM lengths
            GROUP BY length_bucket
            ORDER BY length_bucket
            """,
        )
        analytical_family_distribution = _distribution(
            connection,
            """
            SELECT
                products.analytical_family_id,
                min(products.analytical_family_name) AS analytical_family_name,
                min(products.comparison_status) AS comparison_status,
                count(DISTINCT products.parent_asin) AS catalog_product_count,
                count(DISTINCT products.parent_asin) FILTER (
                    WHERE reviews.parent_asin IS NOT NULL
                ) AS reviewed_product_count,
                count(reviews.review_id) AS review_count
            FROM niche_products AS products
            LEFT JOIN niche_reviews AS reviews USING (parent_asin)
            GROUP BY products.analytical_family_id
            ORDER BY products.analytical_family_id
            """,
        )
        competitor_niche_distribution = _distribution(
            connection,
            """
            SELECT
                products.competitor_niche_id,
                min(products.competitor_niche_name) AS competitor_niche_name,
                min(products.analytical_family_id) AS analytical_family_id,
                min(products.comparison_status) AS comparison_status,
                count(DISTINCT products.parent_asin) AS catalog_product_count,
                count(DISTINCT products.parent_asin) FILTER (
                    WHERE reviews.parent_asin IS NOT NULL
                ) AS reviewed_product_count,
                count(reviews.review_id) AS review_count
            FROM niche_products AS products
            LEFT JOIN niche_reviews AS reviews USING (parent_asin)
            GROUP BY products.competitor_niche_id
            ORDER BY products.competitor_niche_id
            """,
        )
    finally:
        connection.close()

    catalog_product_count = int(product_summary[0])
    reviewed_product_count = int(product_summary[1])
    verified_purchase_count = int(review_summary[5])
    helpful_review_count = int(review_summary[6])
    report = NichePopulationReport(
        niche_id=niche.niche_id,
        niche_version=niche.niche_version,
        niche_status=niche.status,
        display_name=niche.display_name,
        dataset_version=niche.dataset_version,
        category_registry_schema_version=niche.category_registry_schema_version,
        catalog_path=_artifact_reference(catalog),
        reviews_path=_artifact_reference(reviews),
        category_registry_path=_artifact_reference(registry),
        category_paths=category_paths,
        catalog_product_count=catalog_product_count,
        reviewed_product_count=reviewed_product_count,
        catalog_product_review_coverage=(
            reviewed_product_count / catalog_product_count
            if catalog_product_count
            else 0.0
        ),
        reviewed_variation_asin_count=int(product_summary[2]),
        review_count=review_count,
        distinct_review_parent_asin_count=int(review_summary[1]),
        distinct_review_asin_count=int(review_summary[2]),
        review_date_min=(
            review_summary[3].date().isoformat() if review_summary[3] else None
        ),
        review_date_max=(
            review_summary[4].date().isoformat() if review_summary[4] else None
        ),
        verified_purchase_count=verified_purchase_count,
        verified_purchase_share=(
            verified_purchase_count / review_count if review_count else 0.0
        ),
        helpful_review_count=helpful_review_count,
        helpful_review_share=(
            helpful_review_count / review_count if review_count else 0.0
        ),
        rating_distribution=rating_distribution,
        review_year_distribution=review_year_distribution,
        review_length_distribution=review_length_distribution,
        analytical_family_distribution=analytical_family_distribution,
        competitor_niche_distribution=competitor_niche_distribution,
    )
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.json")
        temporary.write_text(
            json.dumps(asdict(report), indent=2), encoding="utf-8"
        )
        temporary.replace(destination)
    return report


def main() -> None:
    """Profile one niche using paths declared by a dataset manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("niche_config_path", type=Path)
    parser.add_argument("report_path", type=Path)
    args = parser.parse_args()

    root = find_project_root(args.manifest_path.parent)
    manifest = load_dataset_manifest(args.manifest_path)
    niche = load_product_niche_definition(args.niche_config_path)
    report = profile_niche_population(
        niche,
        root / manifest.file_by_role("product_catalog").path,
        root / manifest.file_by_role("canonical_reviews").path,
        root / manifest.file_by_role("category_registry").path,
        report_path=args.report_path,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
