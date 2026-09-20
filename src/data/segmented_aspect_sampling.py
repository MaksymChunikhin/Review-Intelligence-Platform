"""Build a deterministic family-aware sample for aspect discovery."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from src.common.project import find_project_root
from src.data.aspect_sampling import allocate_stratum_quotas
from src.ingestion.dataset_manifest import sha256_file
from src.schemas.workspace import (
    ProductNicheDefinition,
    load_product_niche_definition,
)


SAMPLE_SCHEMA_VERSION = "aspect_discovery_sample_v3"


@dataclass(frozen=True)
class SegmentedAspectSampleReport:
    """Record lineage and exact segment/stratum coverage of one sample."""

    sampling_version: str
    sample_schema_version: str
    sampling_algorithm_version: str
    niche_id: str
    niche_version: str
    dataset_version: str
    source_path: str
    output_path: str
    output_sha256: str
    requested_sample_size: int
    actual_sample_size: int
    minimum_words: int
    minimum_per_stratum: int
    deterministic_seed: int
    excluded_review_count: int
    population_review_count: int
    eligible_review_count: int
    excluded_short_review_count: int
    population_product_count: int
    eligible_product_count: int
    sample_product_count: int
    maximum_sample_reviews_per_product: int
    sampling_design: str
    sampling_weight_semantics: str
    population_estimand: str
    stratification_dimensions: list[str]
    requested_segment_sample_sizes: dict[str, int]
    segment_distribution: list[dict[str, Any]]
    strata: list[dict[str, Any]]


def _artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


def build_segmented_aspect_discovery_sample(
    niche: ProductNicheDefinition,
    niche_reviews_path: str | Path,
    output_path: str | Path,
    *,
    sampling_version: str,
    segment_sample_sizes: dict[str, int],
    report_path: str | Path | None = None,
    exclude_review_ids_path: str | Path | list[str | Path] | None = None,
    minimum_words: int = 5,
    minimum_per_stratum: int = 3,
    deterministic_seed: int = 142,
    memory_limit: str = "4GB",
) -> SegmentedAspectSampleReport:
    """Sample every configured competitor niche with valid population weights."""
    if niche.status != "approved":
        raise ValueError("niche must be approved before aspect sampling")
    if not sampling_version:
        raise ValueError("sampling_version must not be empty")
    if minimum_words < 1 or minimum_per_stratum < 1:
        raise ValueError("word and stratum minimums must be positive")
    if not segment_sample_sizes or any(
        size <= 0 for size in segment_sample_sizes.values()
    ):
        raise ValueError("segment sample sizes must be positive and non-empty")

    source = Path(niche_reviews_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)

    connection = duckdb.connect()
    try:
        escaped_limit = memory_limit.replace("'", "''")
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute(f"SET memory_limit = '{escaped_limit}'")
        required_columns = {
            "review_id",
            "dataset_version",
            "niche_id",
            "niche_version",
            "parent_asin",
            "asin",
            "rating",
            "review_timestamp",
            "verified_purchase",
            "helpful_vote",
            "review_text",
            "product_title",
            "store",
            "category_path_id",
            "category_path_text",
            "analytical_family_id",
            "analytical_family_name",
            "competitor_niche_id",
            "competitor_niche_name",
            "comparison_status",
        }
        available_columns = {
            row[0]
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]
            ).fetchall()
        }
        missing_columns = required_columns - available_columns
        if missing_columns:
            raise ValueError(
                "niche review artifact is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
        identity = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT review_id),
                count(DISTINCT dataset_version), min(dataset_version),
                count(DISTINCT niche_id), min(niche_id),
                count(DISTINCT niche_version), min(niche_version)
            FROM read_parquet(?)
            """,
            [str(source)],
        ).fetchone()
        if int(identity[0]) == 0 or int(identity[0]) != int(identity[1]):
            raise ValueError("niche review artifact must contain unique reviews")
        expected_identity = (
            niche.dataset_version,
            niche.niche_id,
            niche.niche_version,
        )
        actual_identity = (identity[3], identity[5], identity[7])
        if (identity[2], identity[4], identity[6]) != (1, 1, 1) or (
            actual_identity != expected_identity
        ):
            raise ValueError(
                "niche review artifact identity mismatch: "
                f"expected={expected_identity}, actual={actual_identity}"
            )

        excluded_review_count = 0
        if exclude_review_ids_path is not None:
            raw_sources = (
                exclude_review_ids_path
                if isinstance(exclude_review_ids_path, list)
                else [exclude_review_ids_path]
            )
            exclusion_frames = []
            for raw_source in raw_sources:
                exclusion_source = Path(raw_source)
                if not exclusion_source.is_file():
                    raise FileNotFoundError(exclusion_source)
                exclusion_columns = {
                    row[0]
                    for row in connection.execute(
                        "DESCRIBE SELECT * FROM read_parquet(?)",
                        [str(exclusion_source)],
                    ).fetchall()
                }
                if "review_id" not in exclusion_columns:
                    raise ValueError("exclusion artifact must contain review_id")
                exclusion_frames.append(
                    pd.read_parquet(exclusion_source, columns=["review_id"])
                )
            excluded_frame = pd.concat(
                exclusion_frames, ignore_index=True
            ).drop_duplicates("review_id")
            connection.register("configured_excluded_reviews", excluded_frame)
            connection.execute(
                """
                CREATE TEMP TABLE excluded_reviews AS
                SELECT review_id FROM configured_excluded_reviews
                """
            )
            excluded_review_count = int(
                connection.execute(
                    "SELECT count(*) FROM excluded_reviews"
                ).fetchone()[0]
            )
        else:
            connection.execute(
                "CREATE TEMP TABLE excluded_reviews(review_id VARCHAR)"
            )

        connection.execute(
            """
            CREATE TEMP TABLE eligible_population AS
            WITH enriched AS (
                SELECT
                    *,
                    array_length(
                        regexp_split_to_array(trim(review_text), '\\s+')
                    ) AS word_count
                FROM read_parquet(?) AS reviews
                LEFT JOIN excluded_reviews USING (review_id)
                WHERE excluded_reviews.review_id IS NULL
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
                END AS length_bucket,
                concat_ws(
                    '|', competitor_niche_id, rating_group,
                    year(review_timestamp), length_bucket
                ) AS stratum_id
            FROM enriched
            WHERE word_count >= ?
            """,
            [str(source), minimum_words],
        )
        population_summary = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM read_parquet(?)),
                count(*),
                (SELECT count(DISTINCT parent_asin) FROM read_parquet(?)),
                count(DISTINCT parent_asin)
            FROM eligible_population
            """,
            [str(source), str(source)],
        ).fetchone()
        population_review_count = int(population_summary[0])
        eligible_review_count = int(population_summary[1])
        population_product_count = int(population_summary[2])
        eligible_product_count = int(population_summary[3])
        excluded_short_review_count = int(
            connection.execute(
                """
                SELECT count(*)
                FROM read_parquet(?)
                WHERE array_length(
                    regexp_split_to_array(trim(review_text), '\\s+')
                ) < ?
                """,
                [str(source), minimum_words],
            ).fetchone()[0]
        )

        actual_segments = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT competitor_niche_id FROM eligible_population"
            ).fetchall()
        }
        requested_segments = set(segment_sample_sizes)
        if requested_segments != actual_segments:
            raise ValueError(
                "segment quotas must cover every eligible competitor niche: "
                f"missing={sorted(actual_segments - requested_segments)}, "
                f"unexpected={sorted(requested_segments - actual_segments)}"
            )

        stratum_frame = connection.execute(
            """
            SELECT
                competitor_niche_id,
                stratum_id,
                min(analytical_family_id) AS analytical_family_id,
                min(rating_group) AS rating_group,
                min(review_year) AS review_year,
                min(length_bucket) AS length_bucket,
                count(*) AS population_stratum_count
            FROM eligible_population
            GROUP BY competitor_niche_id, stratum_id
            ORDER BY competitor_niche_id, stratum_id
            """
        ).fetchdf()
        quota_rows: list[dict[str, Any]] = []
        for segment_id in sorted(actual_segments):
            segment_strata = stratum_frame[
                stratum_frame["competitor_niche_id"] == segment_id
            ]
            stratum_counts = {
                str(row.stratum_id): int(row.population_stratum_count)
                for row in segment_strata.itertuples(index=False)
            }
            quotas = allocate_stratum_quotas(
                stratum_counts,
                sample_size=segment_sample_sizes[segment_id],
                minimum_per_stratum=minimum_per_stratum,
            )
            quota_rows.extend(
                {
                    "stratum_id": stratum_id,
                    "population_stratum_count": stratum_counts[stratum_id],
                    "sample_stratum_count": quota,
                }
                for stratum_id, quota in quotas.items()
            )
        quota_frame = pd.DataFrame(quota_rows)
        connection.register("sample_quotas", quota_frame)
        connection.execute(
            """
            CREATE TEMP TABLE aspect_sample AS
            WITH ranked AS (
                SELECT
                    population.*,
                    row_number() OVER (
                        PARTITION BY stratum_id
                        ORDER BY md5(review_id || ':' || cast(? AS VARCHAR)),
                            review_id
                    ) AS stratum_sample_rank
                FROM eligible_population AS population
            )
            SELECT
                ranked.* EXCLUDE (stratum_sample_rank),
                ?::VARCHAR AS sample_schema_version,
                ?::VARCHAR AS sampling_version,
                quotas.population_stratum_count,
                quotas.sample_stratum_count,
                quotas.population_stratum_count::DOUBLE
                    / quotas.sample_stratum_count AS sampling_weight,
                'inverse_review_inclusion_probability'::VARCHAR
                    AS sampling_weight_semantics
            FROM ranked
            INNER JOIN sample_quotas AS quotas USING (stratum_id)
            WHERE ranked.stratum_sample_rank <= quotas.sample_stratum_count
            """,
            [deterministic_seed, SAMPLE_SCHEMA_VERSION, sampling_version],
        )
        sample_summary = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT review_id),
                count(DISTINCT parent_asin),
                max(product_sample_count),
                sum(sampling_weight)
            FROM (
                SELECT *, count(*) OVER (PARTITION BY parent_asin)
                    AS product_sample_count
                FROM aspect_sample
            )
            """
        ).fetchone()
        requested_sample_size = sum(segment_sample_sizes.values())
        if int(sample_summary[0]) != requested_sample_size or int(
            sample_summary[1]
        ) != requested_sample_size:
            raise RuntimeError("segmented sample size or uniqueness is invalid")
        if not math.isclose(
            float(sample_summary[4]),
            float(eligible_review_count),
            rel_tol=1e-12,
            abs_tol=1e-8,
        ):
            raise RuntimeError("sample weights do not reconcile to the population")

        segment_distribution = _records(
            connection.execute(
                """
                SELECT
                    competitor_niche_id,
                    min(competitor_niche_name) AS competitor_niche_name,
                    min(analytical_family_id) AS analytical_family_id,
                    min(comparison_status) AS comparison_status,
                    count(*) AS sample_review_count,
                    count(DISTINCT parent_asin) AS sample_product_count,
                    sum(sampling_weight) AS weighted_eligible_review_count
                FROM aspect_sample
                GROUP BY competitor_niche_id
                ORDER BY competitor_niche_id
                """
            ).fetchdf()
        )
        strata = _records(
            connection.execute(
                """
                SELECT
                    stratum_id,
                    min(competitor_niche_id) AS competitor_niche_id,
                    min(rating_group) AS rating_group,
                    min(review_year) AS review_year,
                    min(length_bucket) AS length_bucket,
                    min(population_stratum_count) AS population_stratum_count,
                    min(sample_stratum_count) AS sample_stratum_count,
                    min(sampling_weight) AS sampling_weight
                FROM aspect_sample
                GROUP BY stratum_id
                ORDER BY stratum_id
                """
            ).fetchdf()
        )
        connection.execute(
            """
            COPY (
                SELECT * FROM aspect_sample
                ORDER BY stratum_id, review_id
            ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 10000)
            """,
            [str(temporary)],
        )
    finally:
        connection.close()

    if pq.ParquetFile(temporary).metadata.num_rows != sum(
        segment_sample_sizes.values()
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("written segmented sample row count is invalid")
    output_sha256 = sha256_file(temporary)
    temporary.replace(destination)
    report = SegmentedAspectSampleReport(
        sampling_version=sampling_version,
        sample_schema_version=SAMPLE_SCHEMA_VERSION,
        sampling_algorithm_version="segmented_stratified_review_srs_v1",
        niche_id=niche.niche_id,
        niche_version=niche.niche_version,
        dataset_version=niche.dataset_version,
        source_path=_artifact_reference(source),
        output_path=_artifact_reference(destination),
        output_sha256=output_sha256,
        requested_sample_size=sum(segment_sample_sizes.values()),
        actual_sample_size=int(sample_summary[0]),
        minimum_words=minimum_words,
        minimum_per_stratum=minimum_per_stratum,
        deterministic_seed=deterministic_seed,
        excluded_review_count=excluded_review_count,
        population_review_count=population_review_count,
        eligible_review_count=eligible_review_count,
        excluded_short_review_count=excluded_short_review_count,
        population_product_count=population_product_count,
        eligible_product_count=eligible_product_count,
        sample_product_count=int(sample_summary[2]),
        maximum_sample_reviews_per_product=int(sample_summary[3]),
        sampling_design=(
            "competitor_niche_by_rating_year_length_stratified_srs_without_replacement"
        ),
        sampling_weight_semantics=(
            "inverse review inclusion probability N_h/n_h within each exact stratum"
        ),
        population_estimand=(
            "all section reviews meeting minimum_words, including quarantined "
            "generic-path reviews for taxonomy discovery only"
        ),
        stratification_dimensions=[
            "competitor_niche_id",
            "rating_group",
            "review_year",
            "length_bucket",
        ],
        requested_segment_sample_sizes=dict(sorted(segment_sample_sizes.items())),
        segment_distribution=segment_distribution,
        strata=strata,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("niche_config_path", type=Path)
    parser.add_argument("niche_reviews_path", type=Path)
    parser.add_argument("sampling_config_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument(
        "--exclude-review-ids-path", type=Path, action="append"
    )
    args = parser.parse_args()

    niche = load_product_niche_definition(args.niche_config_path)
    sampling_payload = json.loads(
        args.sampling_config_path.read_text(encoding="utf-8")
    )
    expected_identity = {
        "niche_id": niche.niche_id,
        "niche_version": niche.niche_version,
        "dataset_version": niche.dataset_version,
    }
    mismatches = {
        field: {"expected": expected, "actual": sampling_payload.get(field)}
        for field, expected in expected_identity.items()
        if sampling_payload.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"sampling configuration identity mismatch: {mismatches}")
    report = build_segmented_aspect_discovery_sample(
        niche,
        args.niche_reviews_path,
        args.output_path,
        sampling_version=sampling_payload["sampling_version"],
        segment_sample_sizes={
            str(key): int(value)
            for key, value in sampling_payload["segment_sample_sizes"].items()
        },
        report_path=args.report_path,
        exclude_review_ids_path=args.exclude_review_ids_path,
        minimum_words=int(sampling_payload.get("minimum_words", 5)),
        minimum_per_stratum=int(
            sampling_payload.get("minimum_per_stratum", 3)
        ),
        deterministic_seed=int(sampling_payload.get("deterministic_seed", 142)),
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
