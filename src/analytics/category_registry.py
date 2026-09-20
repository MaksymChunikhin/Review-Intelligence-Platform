"""Build a versioned Amazon category registry from full item metadata."""

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import load_dataset_manifest


CATEGORY_REGISTRY_SCHEMA = pa.schema(
    [
        pa.field("category_path_id", pa.string(), nullable=False),
        pa.field("dataset_version", pa.string(), nullable=False),
        pa.field("dataset_category", pa.string(), nullable=False),
        pa.field("registry_schema_version", pa.string(), nullable=False),
        pa.field("main_categories", pa.list_(pa.string()), nullable=False),
        pa.field("main_category_count", pa.int16(), nullable=False),
        pa.field("category_path", pa.list_(pa.string()), nullable=False),
        pa.field("category_path_text", pa.string(), nullable=False),
        pa.field("category_path_key", pa.string(), nullable=False),
        pa.field("category_depth", pa.int16(), nullable=False),
        pa.field("leaf_category", pa.string(), nullable=False),
        pa.field("leaf_category_key", pa.string(), nullable=False),
        pa.field("product_record_count", pa.int64(), nullable=False),
        pa.field("distinct_parent_asin_count", pa.int64(), nullable=False),
        pa.field("product_share", pa.float64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class CategoryRegistryBuildReport:
    """Record exact source-quality and registry build counts."""

    dataset_version: str
    dataset_category: str
    registry_schema_version: str
    input_path: str
    output_path: str
    input_rows: int
    distinct_parent_asin_count: int
    duplicate_parent_asin_rows: int
    missing_parent_asin_rows: int
    missing_main_category_rows: int
    missing_category_path_rows: int
    main_category_count: int
    registry_row_count: int
    distinct_category_path_count: int
    category_node_count: int
    min_category_depth: int | None
    max_category_depth: int | None
    main_category_distribution: list[dict[str, Any]]
    category_depth_distribution: list[dict[str, int]]


def normalize_category_label(value: str | None) -> str | None:
    """Create a stable matching key without changing the displayed label."""
    normalized = normalize_category_display_label(value)
    return normalized.casefold() if normalized is not None else None


def normalize_category_display_label(value: str | None) -> str | None:
    """Normalize Unicode and whitespace while preserving display casing."""
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def category_path_key(category_path: list[str] | None) -> str | None:
    """Return the shared normalized matching key for a category path."""
    if not category_path:
        return None
    labels = [
        normalized
        for label in category_path
        if (normalized := normalize_category_label(label)) is not None
    ]
    return " > ".join(labels) or None


def _category_path_id(
    dataset_category: str,
    category_path_key: str,
) -> str:
    """Create a stable namespaced identifier for one declared category path."""
    identity = "|".join([dataset_category, category_path_key])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"amazon-category:{digest}"


def _artifact_reference(path: Path) -> str:
    """Prefer a repository-relative reference in persisted provenance."""
    try:
        root = find_project_root(path.parent)
        return path.resolve().relative_to(root).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def build_category_registry(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dataset_version: str,
    dataset_category: str,
    registry_schema_version: str,
    report_path: str | Path | None = None,
    memory_limit: str = "4GB",
) -> CategoryRegistryBuildReport:
    """Scan complete Amazon metadata and atomically write its category registry."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not dataset_version or not dataset_category or not registry_schema_version:
        raise ValueError("dataset and registry identities must not be empty")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = destination.with_suffix(".tmp.parquet")
    temporary_output.unlink(missing_ok=True)

    connection = duckdb.connect()
    escaped_memory_limit = memory_limit.replace("'", "''")
    try:
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
            """
            CREATE TEMP TABLE raw_category_source AS
            SELECT
                row_number() OVER () - 1 AS source_record_index,
                nullif(trim(parent_asin), '') AS parent_asin,
                nullif(trim(main_category), '') AS main_category,
                categories AS raw_category_path
            FROM read_json_auto(
                ?,
                format = 'newline_delimited',
                maximum_object_size = 10485760,
                union_by_name = true
            )
            """,
            [str(source)],
        )
        connection.execute(
            """
            CREATE TEMP TABLE category_source AS
            WITH normalized_paths AS (
                SELECT
                    * EXCLUDE (raw_category_path),
                    list_filter(
                        list_transform(
                            coalesce(raw_category_path, []::VARCHAR[]),
                            label -> normalize_category_display_label(label)
                        ),
                        label -> label IS NOT NULL
                    ) AS category_path
                FROM raw_category_source
            )
            SELECT
                *,
                normalized_category_path_key(category_path)
                    AS category_path_key
            FROM normalized_paths
            """
        )
        summary = connection.execute(
            """
            SELECT
                count(*) AS input_rows,
                count(DISTINCT parent_asin) AS distinct_parent_asin_count,
                count(*) FILTER (WHERE parent_asin IS NULL)
                    AS missing_parent_asin_rows,
                count(*) FILTER (WHERE main_category IS NULL)
                    AS missing_main_category_rows,
                count(*) FILTER (
                    WHERE category_path IS NULL OR len(category_path) = 0
                ) AS missing_category_path_rows,
                count(DISTINCT main_category) AS main_category_count,
                min(len(category_path)) FILTER (
                    WHERE category_path IS NOT NULL AND len(category_path) > 0
                ) AS min_category_depth,
                max(len(category_path)) FILTER (
                    WHERE category_path IS NOT NULL AND len(category_path) > 0
                ) AS max_category_depth
            FROM category_source
            """
        ).fetchone()
        summary_columns = [column[0] for column in connection.description]
        summary_values = dict(zip(summary_columns, summary, strict=True))

        path_rows = connection.execute(
            """
            SELECT
                min(category_path) AS category_path,
                category_path_key,
                list(DISTINCT main_category ORDER BY main_category) FILTER (
                    WHERE main_category IS NOT NULL
                ) AS main_categories,
                count(*) AS product_record_count,
                count(DISTINCT parent_asin) AS distinct_parent_asin_count
            FROM category_source
            WHERE category_path IS NOT NULL AND len(category_path) > 0
            GROUP BY category_path_key
            ORDER BY product_record_count DESC, category_path_key
            """
        ).fetchall()
        main_category_rows = connection.execute(
            """
            SELECT main_category, count(*) AS product_record_count
            FROM category_source
            GROUP BY main_category
            ORDER BY product_record_count DESC, main_category
            """
        ).fetchall()
        depth_rows = connection.execute(
            """
            SELECT len(category_path) AS category_depth, count(*) AS product_record_count
            FROM category_source
            WHERE category_path IS NOT NULL AND len(category_path) > 0
            GROUP BY category_depth
            ORDER BY category_depth
            """
        ).fetchall()
    finally:
        connection.close()

    registry_records: list[dict[str, Any]] = []
    category_prefixes: set[tuple[str, ...]] = set()
    input_rows = int(summary_values["input_rows"])
    for (
        raw_path,
        normalized_path_key,
        main_categories,
        product_count,
        parent_count,
    ) in path_rows:
        category_path = [str(label) for label in raw_path]
        category_prefixes.update(
            tuple(
                normalize_category_label(label) or ""
                for label in category_path[:depth]
            )
            for depth in range(1, len(category_path) + 1)
        )
        registry_records.append(
            {
                "category_path_id": _category_path_id(
                    dataset_category, normalized_path_key
                ),
                "dataset_version": dataset_version,
                "dataset_category": dataset_category,
                "registry_schema_version": registry_schema_version,
                "main_categories": list(main_categories or []),
                "main_category_count": len(main_categories or []),
                "category_path": category_path,
                "category_path_text": " > ".join(category_path),
                "category_path_key": normalized_path_key,
                "category_depth": len(category_path),
                "leaf_category": category_path[-1],
                "leaf_category_key": normalize_category_label(category_path[-1]),
                "product_record_count": int(product_count),
                "distinct_parent_asin_count": int(parent_count),
                "product_share": float(product_count) / input_rows,
            }
        )

    registry_frame = pd.DataFrame.from_records(
        registry_records,
        columns=CATEGORY_REGISTRY_SCHEMA.names,
    )
    if registry_frame["category_path_id"].duplicated().any():
        raise RuntimeError("Category path IDs are not unique")
    if int(registry_frame["product_record_count"].sum()) != (
        input_rows - int(summary_values["missing_category_path_rows"])
    ):
        raise RuntimeError("Category path counts do not reconcile to source rows")

    table = pa.Table.from_pandas(
        registry_frame,
        schema=CATEGORY_REGISTRY_SCHEMA,
        preserve_index=False,
        safe=True,
    )
    pq.write_table(table, temporary_output, compression="zstd")
    written = pq.ParquetFile(temporary_output)
    if written.metadata.num_rows != len(registry_frame):
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError("Written category registry row count is invalid")
    actual_schema = written.schema_arrow
    if not actual_schema.equals(CATEGORY_REGISTRY_SCHEMA, check_metadata=False):
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(
            "Written category registry schema does not match its contract"
        )
    temporary_output.replace(destination)

    distinct_parent_asin_count = int(
        summary_values["distinct_parent_asin_count"]
    )
    missing_parent_asin_rows = int(summary_values["missing_parent_asin_rows"])
    report = CategoryRegistryBuildReport(
        dataset_version=dataset_version,
        dataset_category=dataset_category,
        registry_schema_version=registry_schema_version,
        input_path=_artifact_reference(source),
        output_path=_artifact_reference(destination),
        input_rows=input_rows,
        distinct_parent_asin_count=distinct_parent_asin_count,
        duplicate_parent_asin_rows=(
            input_rows - missing_parent_asin_rows - distinct_parent_asin_count
        ),
        missing_parent_asin_rows=missing_parent_asin_rows,
        missing_main_category_rows=int(
            summary_values["missing_main_category_rows"]
        ),
        missing_category_path_rows=int(
            summary_values["missing_category_path_rows"]
        ),
        main_category_count=int(summary_values["main_category_count"]),
        registry_row_count=len(registry_frame),
        distinct_category_path_count=int(
            registry_frame["category_path_key"].nunique()
        ),
        category_node_count=len(category_prefixes),
        min_category_depth=(
            int(summary_values["min_category_depth"])
            if summary_values["min_category_depth"] is not None
            else None
        ),
        max_category_depth=(
            int(summary_values["max_category_depth"])
            if summary_values["max_category_depth"] is not None
            else None
        ),
        main_category_distribution=[
            {
                "main_category": main_category,
                "product_record_count": int(product_count),
            }
            for main_category, product_count in main_category_rows
        ],
        category_depth_distribution=[
            {
                "category_depth": int(depth),
                "product_record_count": int(product_count),
            }
            for depth, product_count in depth_rows
        ],
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
    """Build the category registry declared by a dataset manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()

    project_root = find_project_root(args.manifest_path.parent)
    manifest = load_dataset_manifest(args.manifest_path)
    report = build_category_registry(
        project_root / manifest.file_by_role("raw_product_metadata").path,
        args.output_path,
        dataset_version=manifest.dataset_version,
        dataset_category=manifest.dataset_category,
        registry_schema_version=manifest.schema_versions.category_registry,
        report_path=args.report_path,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
