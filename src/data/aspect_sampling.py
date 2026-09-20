"""Build a deterministic stratified sample for aspect-taxonomy discovery."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import (
    load_dataset_manifest,
    sha256_file,
)
from src.schemas.workspace import (
    ProductNicheDefinition,
    load_product_niche_definition,
)


@dataclass(frozen=True)
class AspectDiscoverySampleReport:
    """Record sample lineage, eligibility rules, strata, and coverage."""

    sample_schema_version: str
    sampling_algorithm_version: str
    sampling_design: str
    sampling_weight_semantics: str
    population_estimand: str
    product_cap_applied: bool
    niche_id: str
    niche_version: str
    dataset_version: str
    catalog_path: str
    reviews_path: str
    output_path: str
    output_sha256: str
    requested_sample_size: int
    actual_sample_size: int
    deterministic_seed: int
    minimum_words: int
    minimum_per_stratum: int
    deprecated_maximum_reviews_per_product: int | None
    stratification_dimensions: list[str]
    rating_group_rules: dict[str, str]
    length_bucket_rules: dict[str, str]
    population_review_count: int
    eligible_review_count: int
    excluded_short_review_count: int
    population_product_count: int
    eligible_product_count: int
    sample_product_count: int
    sample_product_coverage: float
    maximum_sample_reviews_per_product: int
    strata: list[dict[str, Any]]
    sample_rating_distribution: list[dict[str, Any]]
    sample_year_distribution: list[dict[str, Any]]
    sample_length_distribution: list[dict[str, Any]]


def _artifact_reference(path: Path) -> str:
    """Prefer a repository-relative path in persisted provenance."""
    try:
        root = find_project_root(path.parent)
        return path.resolve().relative_to(root).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def allocate_stratum_quotas(
    stratum_counts: dict[str, int],
    *,
    sample_size: int,
    minimum_per_stratum: int,
) -> dict[str, int]:
    """Allocate coverage floors, then distribute remaining quota by size."""
    if not stratum_counts or any(count <= 0 for count in stratum_counts.values()):
        raise ValueError("stratum counts must be positive and non-empty")
    if sample_size <= 0 or minimum_per_stratum <= 0:
        raise ValueError("sample size and stratum minimum must be positive")
    population_size = sum(stratum_counts.values())
    if sample_size > population_size:
        raise ValueError("sample size cannot exceed the eligible population")

    quotas = {
        stratum: min(count, minimum_per_stratum)
        for stratum, count in stratum_counts.items()
    }
    if sum(quotas.values()) > sample_size:
        raise ValueError(
            "sample size is too small for the requested stratum minimum"
        )

    remaining = sample_size - sum(quotas.values())
    while remaining:
        available = {
            stratum: count - quotas[stratum]
            for stratum, count in stratum_counts.items()
            if count > quotas[stratum]
        }
        if not available:
            raise RuntimeError("unable to allocate the complete sample quota")
        weight_total = sum(stratum_counts[stratum] for stratum in available)
        ideal = {
            stratum: remaining * stratum_counts[stratum] / weight_total
            for stratum in available
        }
        additions = {
            stratum: min(available[stratum], math.floor(share))
            for stratum, share in ideal.items()
        }
        allocated = sum(additions.values())
        for stratum, addition in additions.items():
            quotas[stratum] += addition
        remaining -= allocated
        if remaining == 0:
            break

        candidates = sorted(
            (
                stratum
                for stratum in available
                if quotas[stratum] < stratum_counts[stratum]
            ),
            key=lambda stratum: (-(ideal[stratum] % 1), stratum),
        )
        if not candidates:
            continue
        for stratum in candidates:
            if remaining == 0:
                break
            quotas[stratum] += 1
            remaining -= 1
    return dict(sorted(quotas.items()))


def _query_records(
    connection: duckdb.DuckDBPyConnection, sql: str
) -> list[dict[str, Any]]:
    """Return a query as JSON-compatible row dictionaries."""
    rows = connection.execute(sql).fetchall()
    columns = [column[0] for column in connection.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _validate_dataset_artifact(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    expected_dataset_version: str,
    identity_column: str,
    artifact_name: str,
) -> None:
    """Require one matching dataset identity and unique row identifiers."""
    row = connection.execute(
        f"""
        SELECT
            count(*) AS row_count,
            count(DISTINCT dataset_version) AS dataset_version_count,
            min(dataset_version) AS dataset_version,
            count(*) FILTER (WHERE dataset_version IS NULL)
                AS missing_dataset_version_rows,
            count(*) - count(DISTINCT {identity_column}) AS duplicate_or_null_ids
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchone()
    if int(row[0]) == 0:
        raise ValueError(f"{artifact_name} must not be empty")
    if (
        int(row[1]) != 1
        or row[2] != expected_dataset_version
        or int(row[3]) != 0
    ):
        raise ValueError(
            f"{artifact_name} dataset version does not match the approved niche"
        )
    if int(row[4]) != 0:
        raise ValueError(
            f"{artifact_name} {identity_column} values must be unique and non-null"
        )


def build_aspect_discovery_sample(
    niche: ProductNicheDefinition,
    catalog_path: str | Path,
    reviews_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    sample_size: int = 1_500,
    minimum_words: int = 5,
    minimum_per_stratum: int = 20,
    maximum_reviews_per_product: int | None = None,
    deterministic_seed: int = 42,
    memory_limit: str = "4GB",
) -> AspectDiscoverySampleReport:
    """Sample reviews across rating, year, and text-length strata."""
    if niche.status != "approved":
        raise ValueError("niche must be approved before aspect sampling")
    if minimum_words < 1:
        raise ValueError("minimum words must be positive")
    if maximum_reviews_per_product is not None:
        if maximum_reviews_per_product < 1:
            raise ValueError("legacy per-product limit must be positive")
        warnings.warn(
            "maximum_reviews_per_product is deprecated and ignored; "
            "population-weighted sampling uses review-level stratified SRS",
            DeprecationWarning,
            stacklevel=2,
        )
    catalog = Path(catalog_path)
    reviews = Path(reviews_path)
    destination = Path(output_path)
    for source in (catalog, reviews):
        if not source.is_file():
            raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = destination.with_suffix(".tmp.parquet")
    temporary_output.unlink(missing_ok=True)

    connection = duckdb.connect()
    escaped_memory_limit = memory_limit.replace("'", "''")
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute(f"SET memory_limit = '{escaped_memory_limit}'")
        _validate_dataset_artifact(
            connection,
            catalog,
            expected_dataset_version=niche.dataset_version,
            identity_column="parent_asin",
            artifact_name="product catalog",
        )
        _validate_dataset_artifact(
            connection,
            reviews,
            expected_dataset_version=niche.dataset_version,
            identity_column="review_id",
            artifact_name="canonical reviews",
        )
        connection.execute(
            """
            CREATE TEMP TABLE niche_products AS
            SELECT
                parent_asin,
                product_title,
                store,
                category_path_id,
                category_path_text
            FROM read_parquet(?)
            WHERE category_path_id IN (SELECT * FROM unnest(?))
                AND dataset_version = ?
            """,
            [str(catalog), niche.category_path_ids, niche.dataset_version],
        )
        population_product_count = int(
            connection.execute("SELECT count(*) FROM niche_products").fetchone()[0]
        )
        if population_product_count == 0:
            raise ValueError("approved niche resolves to no catalog products")

        connection.execute(
            """
            CREATE TEMP TABLE niche_review_population AS
            WITH enriched AS (
                SELECT
                    reviews.review_id,
                    reviews.dataset_version,
                    reviews.parent_asin,
                    reviews.asin,
                    reviews.rating,
                    reviews.review_timestamp,
                    reviews.verified_purchase,
                    reviews.helpful_vote,
                    reviews.review_text,
                    products.product_title,
                    products.store,
                    products.category_path_id,
                    products.category_path_text,
                    array_length(
                        regexp_split_to_array(trim(reviews.review_text), '\\s+')
                    ) AS word_count
                FROM read_parquet(?) AS reviews
                INNER JOIN niche_products AS products USING (parent_asin)
                WHERE reviews.dataset_version = ?
            )
            SELECT
                *,
                year(review_timestamp) AS review_year,
                CASE
                    WHEN rating <= 2 THEN 'low_1_2'
                    WHEN rating = 3 THEN 'mid_3'
                    ELSE 'high_4_5'
                END AS rating_group,
                CASE
                    WHEN word_count <= 40 THEN 'short_5_40'
                    WHEN word_count <= 120 THEN 'medium_41_120'
                    ELSE 'long_121_plus'
                END AS length_bucket
            FROM enriched
            """,
            [str(reviews), niche.dataset_version],
        )
        population_summary = connection.execute(
            """
            SELECT
                count(*) AS population_review_count,
                count(DISTINCT parent_asin) AS population_product_count,
                count(*) FILTER (WHERE word_count >= ?)
                    AS eligible_review_count,
                count(DISTINCT parent_asin) FILTER (WHERE word_count >= ?)
                    AS eligible_product_count
            FROM niche_review_population
            """,
            [minimum_words, minimum_words],
        ).fetchone()
        population_review_count = int(population_summary[0])
        reviewed_population_product_count = int(population_summary[1])
        eligible_review_count = int(population_summary[2])
        eligible_product_count = int(population_summary[3])
        if population_review_count == 0:
            raise ValueError("approved niche resolves to no canonical reviews")

        stratum_rows = connection.execute(
            """
            SELECT
                concat_ws('|', rating_group, review_year, length_bucket)
                    AS stratum_id,
                rating_group,
                review_year,
                length_bucket,
                count(*) AS population_stratum_count
            FROM niche_review_population
            WHERE word_count >= ?
            GROUP BY rating_group, review_year, length_bucket
            ORDER BY stratum_id
            """,
            [minimum_words],
        ).fetchall()
        stratum_counts = {row[0]: int(row[4]) for row in stratum_rows}
        quotas = allocate_stratum_quotas(
            stratum_counts,
            sample_size=sample_size,
            minimum_per_stratum=minimum_per_stratum,
        )
        quota_frame = pd.DataFrame(
            [
                {
                    "stratum_id": stratum_id,
                    "population_stratum_count": stratum_counts[stratum_id],
                    "sample_stratum_count": quota,
                }
                for stratum_id, quota in quotas.items()
            ]
        )
        connection.register("sample_quotas", quota_frame)
        connection.execute(
            """
            CREATE TEMP TABLE aspect_sample AS
            WITH eligible AS (
                SELECT
                    *,
                    concat_ws('|', rating_group, review_year, length_bucket)
                        AS stratum_id,
                    md5(review_id || ':' || cast(? AS VARCHAR))
                        AS deterministic_key
                FROM niche_review_population
                WHERE word_count >= ?
            ),
            stratum_ranked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY stratum_id
                        ORDER BY deterministic_key, review_id
                    ) AS stratum_sample_rank
                FROM eligible
            )
            SELECT
                sampled.review_id,
                sampled.dataset_version,
                sampled.parent_asin,
                sampled.asin,
                sampled.rating,
                sampled.review_timestamp,
                sampled.verified_purchase,
                sampled.helpful_vote,
                sampled.review_text,
                sampled.product_title,
                sampled.store,
                sampled.category_path_id,
                sampled.category_path_text,
                sampled.review_year,
                sampled.rating_group,
                sampled.word_count,
                sampled.length_bucket,
                sampled.stratum_id,
                ?::VARCHAR AS niche_id,
                ?::VARCHAR AS niche_version,
                'aspect_discovery_sample_v2'::VARCHAR AS sample_schema_version,
                quotas.population_stratum_count,
                quotas.sample_stratum_count,
                quotas.population_stratum_count::DOUBLE
                    / quotas.sample_stratum_count AS sampling_weight,
                'inverse_review_inclusion_probability'::VARCHAR
                    AS sampling_weight_semantics
            FROM stratum_ranked AS sampled
            INNER JOIN sample_quotas AS quotas USING (stratum_id)
            WHERE sampled.stratum_sample_rank <= quotas.sample_stratum_count
            """,
            [
                deterministic_seed,
                minimum_words,
                niche.niche_id,
                niche.niche_version,
            ],
        )
        sample_summary = connection.execute(
            """
            SELECT
                count(*) AS sample_size,
                count(DISTINCT review_id) AS distinct_review_count,
                count(DISTINCT parent_asin) AS sample_product_count,
                max(product_sample_count) AS maximum_reviews_per_product,
                sum(sampling_weight) AS weighted_review_population
            FROM (
                SELECT *, count(*) OVER (PARTITION BY parent_asin)
                    AS product_sample_count
                FROM aspect_sample
            )
            """
        ).fetchone()
        actual_sample_size = int(sample_summary[0])
        if actual_sample_size != sample_size or int(sample_summary[1]) != sample_size:
            raise RuntimeError("sample row count or review uniqueness is invalid")
        if not math.isclose(
            float(sample_summary[4]),
            float(eligible_review_count),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "sampling weights do not reconcile to the eligible review population"
            )

        strata = _query_records(
            connection,
            """
            SELECT
                stratum_id,
                min(rating_group) AS rating_group,
                min(review_year) AS review_year,
                min(length_bucket) AS length_bucket,
                min(population_stratum_count) AS population_stratum_count,
                min(sample_stratum_count) AS sample_stratum_count,
                min(sampling_weight) AS sampling_weight
            FROM aspect_sample
            GROUP BY stratum_id
            ORDER BY stratum_id
            """,
        )
        sample_rating_distribution = _query_records(
            connection,
            """
            SELECT rating_group, count(*) AS sample_count
            FROM aspect_sample
            GROUP BY rating_group
            ORDER BY rating_group
            """,
        )
        sample_year_distribution = _query_records(
            connection,
            """
            SELECT review_year, count(*) AS sample_count
            FROM aspect_sample
            GROUP BY review_year
            ORDER BY review_year
            """,
        )
        sample_length_distribution = _query_records(
            connection,
            """
            SELECT length_bucket, count(*) AS sample_count
            FROM aspect_sample
            GROUP BY length_bucket
            ORDER BY length_bucket
            """,
        )
        connection.execute(
            """
            COPY (
                SELECT * FROM aspect_sample
                ORDER BY stratum_id, review_id
            ) TO ? (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 10000
            )
            """,
            [str(temporary_output)],
        )
    finally:
        connection.close()

    written = pq.ParquetFile(temporary_output)
    if written.metadata.num_rows != sample_size:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError("written sample row count is invalid")
    temporary_output.replace(destination)

    sample_product_count = int(sample_summary[2])
    report = AspectDiscoverySampleReport(
        sample_schema_version="aspect_discovery_sample_v2",
        sampling_algorithm_version="stratified_review_srs_v2",
        sampling_design=(
            "stratified_simple_random_sampling_without_replacement"
        ),
        sampling_weight_semantics=(
            "inverse review inclusion probability N_h/n_h for the eligible "
            "review population"
        ),
        population_estimand=(
            "all reviews in the approved niche meeting minimum_words"
        ),
        product_cap_applied=False,
        niche_id=niche.niche_id,
        niche_version=niche.niche_version,
        dataset_version=niche.dataset_version,
        catalog_path=_artifact_reference(catalog),
        reviews_path=_artifact_reference(reviews),
        output_path=_artifact_reference(destination),
        output_sha256=sha256_file(destination),
        requested_sample_size=sample_size,
        actual_sample_size=actual_sample_size,
        deterministic_seed=deterministic_seed,
        minimum_words=minimum_words,
        minimum_per_stratum=minimum_per_stratum,
        deprecated_maximum_reviews_per_product=maximum_reviews_per_product,
        stratification_dimensions=[
            "rating_group",
            "review_year",
            "length_bucket",
        ],
        rating_group_rules={
            "low_1_2": "rating <= 2",
            "mid_3": "rating = 3",
            "high_4_5": "rating >= 4",
        },
        length_bucket_rules={
            "short_5_40": "5-40 words",
            "medium_41_120": "41-120 words",
            "long_121_plus": "121+ words",
        },
        population_review_count=population_review_count,
        eligible_review_count=eligible_review_count,
        excluded_short_review_count=(
            population_review_count - eligible_review_count
        ),
        population_product_count=reviewed_population_product_count,
        eligible_product_count=eligible_product_count,
        sample_product_count=sample_product_count,
        sample_product_coverage=(
            sample_product_count / eligible_product_count
            if eligible_product_count
            else 0.0
        ),
        maximum_sample_reviews_per_product=int(sample_summary[3]),
        strata=strata,
        sample_rating_distribution=sample_rating_distribution,
        sample_year_distribution=sample_year_distribution,
        sample_length_distribution=sample_length_distribution,
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
    """Build a niche aspect-discovery sample from a dataset manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("niche_config_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--sample-size", type=int, default=1_500)
    parser.add_argument("--minimum-words", type=int, default=5)
    parser.add_argument("--minimum-per-stratum", type=int, default=20)
    parser.add_argument(
        "--maximum-reviews-per-product",
        type=int,
        help=(
            "deprecated compatibility option; ignored because weighted "
            "sampling is review-level stratified SRS"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = find_project_root(args.manifest_path.parent)
    manifest = load_dataset_manifest(args.manifest_path)
    niche = load_product_niche_definition(args.niche_config_path)
    report = build_aspect_discovery_sample(
        niche,
        root / manifest.file_by_role("product_catalog").path,
        root / manifest.file_by_role("canonical_reviews").path,
        args.output_path,
        report_path=args.report_path,
        sample_size=args.sample_size,
        minimum_words=args.minimum_words,
        minimum_per_stratum=args.minimum_per_stratum,
        maximum_reviews_per_product=args.maximum_reviews_per_product,
        deterministic_seed=args.seed,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
