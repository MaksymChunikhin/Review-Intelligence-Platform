"""Build the complete Amazon product catalog and exact review coverage."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.analytics.category_registry import (
    category_path_key,
    normalize_category_display_label,
)
from src.common.project import find_project_root
from src.ingestion.dataset_manifest import load_dataset_manifest


PRODUCT_CATALOG_SCHEMA = pa.schema(
    [
        pa.field("dataset_version", pa.string(), nullable=False),
        pa.field("dataset_category", pa.string(), nullable=False),
        pa.field("catalog_schema_version", pa.string(), nullable=False),
        pa.field("source_record_index", pa.int64(), nullable=False),
        pa.field("parent_asin", pa.string(), nullable=False),
        pa.field("product_title", pa.string()),
        pa.field("main_category", pa.string()),
        # A partial or stale registry is reported as a quality issue, so the
        # foreign key must be nullable even though the local normalized path is
        # always present.
        pa.field("category_path_id", pa.string()),
        pa.field("category_path", pa.list_(pa.string()), nullable=False),
        pa.field("category_path_text", pa.string(), nullable=False),
        pa.field("leaf_category", pa.string(), nullable=False),
        pa.field("store", pa.string()),
        pa.field("average_rating", pa.float64()),
        pa.field("rating_number", pa.int64()),
        pa.field("price_at_collection", pa.float64()),
        pa.field("features", pa.list_(pa.string()), nullable=False),
        pa.field("description", pa.list_(pa.string()), nullable=False),
        pa.field("details_json", pa.string(), nullable=False),
        pa.field("product_search_text", pa.string(), nullable=False),
        pa.field("metadata_quality_flags", pa.list_(pa.string()), nullable=False),
        pa.field("review_count", pa.int64(), nullable=False),
        pa.field("reviewed_asin_count", pa.int64(), nullable=False),
        pa.field("min_review_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("max_review_timestamp", pa.timestamp("us", tz="UTC")),
    ]
)


@dataclass(frozen=True)
class ProductCatalogBuildReport:
    """Record catalog quality and exact review-to-metadata coverage."""

    dataset_version: str
    dataset_category: str
    catalog_schema_version: str
    metadata_input_path: str
    reviews_input_path: str
    category_registry_path: str
    output_path: str
    metadata_input_rows: int
    output_rows: int
    distinct_parent_asin_count: int
    missing_product_title_rows: int
    missing_main_category_rows: int
    missing_category_path_rows: int
    missing_store_rows: int
    missing_price_rows: int
    invalid_price_rows: int
    unmatched_category_path_rows: int
    canonical_review_rows: int
    matched_review_rows: int
    unmatched_review_rows: int
    review_join_coverage: float
    catalog_products_with_reviews: int
    catalog_product_review_coverage: float


def _sql_path(path: Path) -> str:
    """Escape a path for use in a DuckDB SQL string literal."""
    return str(path.resolve()).replace("'", "''")


def _sql_text(value: str) -> str:
    """Escape a controlled string value for a DuckDB SQL literal."""
    return value.replace("'", "''")


def _artifact_reference(path: Path) -> str:
    """Prefer a repository-relative reference in persisted provenance."""
    try:
        root = find_project_root(path.parent)
        return path.resolve().relative_to(root).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _write_catalog_parquet(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
) -> None:
    """Stream the catalog through Arrow to preserve nullability constraints."""
    batches = connection.execute(
        "SELECT * FROM catalog ORDER BY source_record_index"
    ).to_arrow_reader(batch_size=100_000)
    writer = pq.ParquetWriter(
        output_path,
        PRODUCT_CATALOG_SCHEMA,
        compression="zstd",
    )
    try:
        for batch in batches:
            table = pa.Table.from_batches([batch]).cast(
                PRODUCT_CATALOG_SCHEMA,
                safe=True,
            )
            writer.write_table(table, row_group_size=100_000)
    except Exception:
        writer.close()
        output_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()


def build_product_catalog(
    metadata_path: str | Path,
    reviews_path: str | Path,
    category_registry_path: str | Path,
    output_path: str | Path,
    *,
    dataset_version: str,
    dataset_category: str,
    catalog_schema_version: str,
    report_path: str | Path | None = None,
    memory_limit: str = "4GB",
) -> ProductCatalogBuildReport:
    """Build one row per parent ASIN and calculate exact review coverage."""
    metadata_source = Path(metadata_path)
    reviews_source = Path(reviews_path)
    category_source = Path(category_registry_path)
    destination = Path(output_path)
    for source in (metadata_source, reviews_source, category_source):
        if not source.is_file():
            raise FileNotFoundError(source)
    if not dataset_version or not dataset_category or not catalog_schema_version:
        raise ValueError("dataset and catalog identities must not be empty")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = destination.with_suffix(".tmp.parquet")
    temporary_output.unlink(missing_ok=True)
    metadata_sql = _sql_path(metadata_source)
    reviews_sql = _sql_path(reviews_source)
    category_sql = _sql_path(category_source)
    version_sql = _sql_text(dataset_version)
    category_name_sql = _sql_text(dataset_category)
    schema_sql = _sql_text(catalog_schema_version)
    escaped_memory_limit = memory_limit.replace("'", "''")

    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute(f"SET memory_limit = '{escaped_memory_limit}'")
        connection.create_function(
            "normalize_category_display_label",
            normalize_category_display_label,
            ["VARCHAR"],
            "VARCHAR",
            null_handling="special",
        )
        connection.create_function(
            "normalized_category_path_key",
            category_path_key,
            ["VARCHAR[]"],
            "VARCHAR",
            null_handling="special",
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE metadata_source AS
            WITH raw_metadata AS (
            SELECT
                row_number() OVER () - 1 AS source_record_index,
                nullif(trim(parent_asin), '') AS parent_asin,
                nullif(trim(title), '') AS product_title,
                nullif(trim(main_category), '') AS main_category,
                categories AS raw_category_path,
                nullif(trim(store), '') AS store,
                try_cast(average_rating AS DOUBLE) AS average_rating,
                try_cast(rating_number AS BIGINT) AS rating_number,
                try_cast(price AS DOUBLE) AS price_at_collection,
                price IS NOT NULL AND try_cast(price AS DOUBLE) IS NULL
                    AS invalid_price,
                coalesce(features, []::VARCHAR[]) AS features,
                coalesce(description, []::VARCHAR[]) AS description,
                coalesce(cast(details AS VARCHAR), '{{}}') AS details_json
            FROM read_json(
                '{metadata_sql}',
                format = 'newline_delimited',
                maximum_object_size = 10485760,
                columns = {{
                    main_category: 'VARCHAR',
                    title: 'VARCHAR',
                    average_rating: 'JSON',
                    rating_number: 'JSON',
                    features: 'VARCHAR[]',
                    description: 'VARCHAR[]',
                    price: 'JSON',
                    store: 'VARCHAR',
                    categories: 'VARCHAR[]',
                    details: 'JSON',
                    parent_asin: 'VARCHAR'
                }}
            )
            ), normalized_metadata AS (
                SELECT
                    * EXCLUDE (raw_category_path),
                    list_filter(
                        list_transform(
                            coalesce(raw_category_path, []::VARCHAR[]),
                            label -> normalize_category_display_label(label)
                        ),
                        label -> label IS NOT NULL
                    ) AS category_path
                FROM raw_metadata
            )
            SELECT
                *,
                normalized_category_path_key(category_path)
                    AS category_path_key
            FROM normalized_metadata
            """
        )
        parent_asin_summary = connection.execute(
            """
            SELECT
                count(*) FILTER (WHERE parent_asin IS NULL),
                count(*) - count(DISTINCT parent_asin)
                    - count(*) FILTER (WHERE parent_asin IS NULL)
            FROM metadata_source
            """
        ).fetchone()
        missing_parent_asin_rows = int(parent_asin_summary[0])
        duplicate_parent_asin_rows = int(parent_asin_summary[1])
        if missing_parent_asin_rows:
            raise ValueError(
                "Product metadata contains "
                f"{missing_parent_asin_rows} missing parent_asin row(s)"
            )
        if duplicate_parent_asin_rows:
            duplicate_example = connection.execute(
                """
                SELECT parent_asin, count(*) AS record_count
                FROM metadata_source
                GROUP BY parent_asin
                HAVING count(*) > 1
                ORDER BY parent_asin
                LIMIT 1
                """
            ).fetchone()
            raise ValueError(
                "Product metadata must contain exactly one row per parent_asin; "
                f"found {duplicate_parent_asin_rows} duplicate row(s) "
                f"(example {duplicate_example[0]!r}: {duplicate_example[1]} rows)"
            )
        duplicate_registry_keys = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM (
                    SELECT category_path_key
                    FROM read_parquet('{category_sql}')
                    GROUP BY category_path_key
                    HAVING count(*) > 1
                )
                """
            ).fetchone()[0]
        )
        if duplicate_registry_keys:
            raise ValueError(
                "Category registry contains duplicate category_path_key values"
            )
        connection.execute(
            f"""
            CREATE TEMP TABLE review_coverage AS
            SELECT
                parent_asin,
                count(*) AS review_count,
                count(DISTINCT asin) AS reviewed_asin_count,
                min(review_timestamp) AS min_review_timestamp,
                max(review_timestamp) AS max_review_timestamp
            FROM read_parquet('{reviews_sql}')
            GROUP BY parent_asin
            """
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE catalog AS
            SELECT
                '{version_sql}' AS dataset_version,
                '{category_name_sql}' AS dataset_category,
                '{schema_sql}' AS catalog_schema_version,
                metadata.source_record_index,
                metadata.parent_asin,
                metadata.product_title,
                metadata.main_category,
                registry.category_path_id,
                coalesce(
                    registry.category_path,
                    metadata.category_path,
                    []::VARCHAR[]
                ) AS category_path,
                coalesce(
                    registry.category_path_text,
                    array_to_string(metadata.category_path, ' > '),
                    ''
                ) AS category_path_text,
                coalesce(
                    registry.leaf_category,
                    list_extract(metadata.category_path, -1),
                    ''
                ) AS leaf_category,
                metadata.store,
                metadata.average_rating,
                metadata.rating_number,
                metadata.price_at_collection,
                metadata.features,
                metadata.description,
                metadata.details_json,
                trim(regexp_replace(
                    concat_ws(
                        ' ',
                        metadata.product_title,
                        metadata.store,
                        array_to_string(metadata.category_path, ' '),
                        array_to_string(metadata.features, ' '),
                        array_to_string(metadata.description, ' ')
                    ),
                    '\\s+',
                    ' ',
                    'g'
                )) AS product_search_text,
                list_filter(
                    [
                        CASE WHEN metadata.product_title IS NULL
                            THEN 'missing_product_title' END,
                        CASE WHEN metadata.main_category IS NULL
                            THEN 'missing_main_category' END,
                        CASE WHEN metadata.category_path IS NULL
                                OR len(metadata.category_path) = 0
                            THEN 'missing_category_path' END,
                        CASE WHEN registry.category_path_id IS NULL
                            THEN 'unmatched_category_path' END,
                        CASE WHEN metadata.store IS NULL
                            THEN 'missing_store' END,
                        CASE WHEN metadata.price_at_collection IS NULL
                                AND NOT metadata.invalid_price
                            THEN 'missing_price' END,
                        CASE WHEN metadata.invalid_price
                            THEN 'invalid_price' END
                    ],
                    flag -> flag IS NOT NULL
                ) AS metadata_quality_flags,
                coalesce(coverage.review_count, 0) AS review_count,
                coalesce(coverage.reviewed_asin_count, 0) AS reviewed_asin_count,
                coverage.min_review_timestamp,
                coverage.max_review_timestamp
            FROM metadata_source AS metadata
            LEFT JOIN read_parquet('{category_sql}') AS registry
                ON metadata.category_path_key = registry.category_path_key
            LEFT JOIN review_coverage AS coverage
                ON metadata.parent_asin = coverage.parent_asin
            ORDER BY metadata.source_record_index
            """
        )

        summary_row = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM metadata_source) AS metadata_input_rows,
                count(*) AS output_rows,
                count(DISTINCT parent_asin) AS distinct_parent_asin_count,
                count(*) FILTER (WHERE product_title IS NULL)
                    AS missing_product_title_rows,
                count(*) FILTER (WHERE main_category IS NULL)
                    AS missing_main_category_rows,
                count(*) FILTER (WHERE len(category_path) = 0)
                    AS missing_category_path_rows,
                count(*) FILTER (WHERE store IS NULL) AS missing_store_rows,
                count(*) FILTER (
                    WHERE list_contains(metadata_quality_flags, 'missing_price')
                ) AS missing_price_rows,
                count(*) FILTER (
                    WHERE list_contains(metadata_quality_flags, 'invalid_price')
                ) AS invalid_price_rows,
                count(*) FILTER (
                    WHERE list_contains(
                        metadata_quality_flags, 'unmatched_category_path'
                    )
                ) AS unmatched_category_path_rows,
                count(*) FILTER (WHERE review_count > 0)
                    AS catalog_products_with_reviews
            FROM catalog
            """
        ).fetchone()
        summary_columns = [column[0] for column in connection.description]
        summary = dict(zip(summary_columns, summary_row, strict=True))

        review_row = connection.execute(
            f"""
            SELECT
                count(*) AS canonical_review_rows,
                count(*) FILTER (WHERE catalog.parent_asin IS NOT NULL)
                    AS matched_review_rows
            FROM read_parquet('{reviews_sql}') AS reviews
            LEFT JOIN catalog USING (parent_asin)
            """
        ).fetchone()
        _write_catalog_parquet(connection, temporary_output)
    finally:
        connection.close()

    output_rows = int(summary["output_rows"])
    if int(summary["metadata_input_rows"]) != output_rows:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError("Metadata input and catalog output do not reconcile")
    written = pq.ParquetFile(temporary_output)
    if written.metadata.num_rows != output_rows:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError("Written catalog row count is invalid")
    actual_schema = written.schema_arrow
    if not actual_schema.equals(PRODUCT_CATALOG_SCHEMA, check_metadata=False):
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError("Written catalog schema does not match its contract")
    temporary_output.replace(destination)

    canonical_review_rows = int(review_row[0])
    matched_review_rows = int(review_row[1])
    report = ProductCatalogBuildReport(
        dataset_version=dataset_version,
        dataset_category=dataset_category,
        catalog_schema_version=catalog_schema_version,
        metadata_input_path=_artifact_reference(metadata_source),
        reviews_input_path=_artifact_reference(reviews_source),
        category_registry_path=_artifact_reference(category_source),
        output_path=_artifact_reference(destination),
        metadata_input_rows=int(summary["metadata_input_rows"]),
        output_rows=output_rows,
        distinct_parent_asin_count=int(summary["distinct_parent_asin_count"]),
        missing_product_title_rows=int(summary["missing_product_title_rows"]),
        missing_main_category_rows=int(summary["missing_main_category_rows"]),
        missing_category_path_rows=int(summary["missing_category_path_rows"]),
        missing_store_rows=int(summary["missing_store_rows"]),
        missing_price_rows=int(summary["missing_price_rows"]),
        invalid_price_rows=int(summary["invalid_price_rows"]),
        unmatched_category_path_rows=int(
            summary["unmatched_category_path_rows"]
        ),
        canonical_review_rows=canonical_review_rows,
        matched_review_rows=matched_review_rows,
        unmatched_review_rows=canonical_review_rows - matched_review_rows,
        review_join_coverage=(
            matched_review_rows / canonical_review_rows
            if canonical_review_rows
            else 0.0
        ),
        catalog_products_with_reviews=int(summary["catalog_products_with_reviews"]),
        catalog_product_review_coverage=(
            int(summary["catalog_products_with_reviews"]) / output_rows
            if output_rows
            else 0.0
        ),
    )
    if report_path is not None:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = report_destination.with_suffix(".tmp.json")
        temporary_report.write_text(
            json.dumps(asdict(report), indent=2), encoding="utf-8"
        )
        temporary_report.replace(report_destination)
    return report


def main() -> None:
    """Build the product catalog declared by a dataset manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()

    project_root = find_project_root(args.manifest_path.parent)
    manifest = load_dataset_manifest(args.manifest_path)
    report = build_product_catalog(
        project_root / manifest.file_by_role("raw_product_metadata").path,
        project_root / manifest.file_by_role("canonical_reviews").path,
        project_root / manifest.file_by_role("category_registry").path,
        args.output_path,
        dataset_version=manifest.dataset_version,
        dataset_category=manifest.dataset_category,
        catalog_schema_version=manifest.schema_versions.product_catalog,
        report_path=args.report_path,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
