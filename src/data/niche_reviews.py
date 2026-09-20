"""Materialize a compact local review table for one approved product niche."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file
from src.schemas.workspace import load_product_niche_definition


@dataclass(frozen=True)
class NicheReviewMaterializationReport:
    niche_id: str
    niche_version: str
    dataset_version: str
    review_count: int
    distinct_parent_asin_count: int
    analytical_family_count: int
    competitor_niche_count: int
    quarantined_review_count: int
    segment_distribution: list[dict[str, object]]
    output_path: str
    output_sha256: str
    excluded_columns: list[str]


def _artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def materialize_niche_reviews(
    niche_config_path: str | Path,
    reviews_path: str | Path,
    catalog_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> NicheReviewMaterializationReport:
    """Join canonical reviews to catalog scope without carrying user IDs."""
    niche = load_product_niche_definition(niche_config_path)
    if niche.status != "approved":
        raise ValueError("niche review materialization requires an approved niche")
    reviews = Path(reviews_path)
    catalog = Path(catalog_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    connection = duckdb.connect()
    try:
        segment_frame = pd.DataFrame(niche.segment_records())
        connection.register("configured_niche_segments", segment_frame)
        output_sql = str(temporary).replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT
                    reviews.review_id,
                    reviews.dataset_version,
                    ?::VARCHAR AS niche_id,
                    ?::VARCHAR AS niche_version,
                    reviews.parent_asin,
                    reviews.asin,
                    reviews.rating,
                    reviews.review_timestamp,
                    reviews.verified_purchase,
                    reviews.helpful_vote,
                    reviews.review_text,
                    catalog.product_title,
                    catalog.store,
                    catalog.category_path_id,
                    catalog.category_path_text,
                    segments.analytical_family_id,
                    segments.analytical_family_name,
                    segments.competitor_niche_id,
                    segments.competitor_niche_name,
                    segments.comparison_status
                FROM read_parquet(?) AS reviews
                INNER JOIN read_parquet(?) AS catalog USING (parent_asin)
                INNER JOIN configured_niche_segments AS segments
                    USING (category_path_id)
                WHERE list_contains(?, catalog.category_path_id)
                  AND reviews.dataset_version = ?
                ORDER BY reviews.review_id
            ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [
                niche.niche_id,
                niche.niche_version,
                str(reviews),
                str(catalog),
                niche.category_path_ids,
                niche.dataset_version,
            ],
        )
    finally:
        connection.close()
    frame = pq.read_table(temporary, columns=["review_id", "parent_asin"])
    review_ids = frame.column("review_id").to_pylist()
    if not review_ids or len(review_ids) != len(set(review_ids)):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("materialized niche reviews are empty or duplicated")
    parent_ids = set(frame.column("parent_asin").to_pylist())
    summary = duckdb.sql(
        """
        SELECT
            count(DISTINCT analytical_family_id) AS analytical_family_count,
            count(DISTINCT competitor_niche_id) AS competitor_niche_count,
            count(*) FILTER (WHERE comparison_status = 'quarantined')
                AS quarantined_review_count
        FROM read_parquet(?)
        """,
        params=[str(temporary)],
    ).fetchone()
    segment_distribution_frame = duckdb.sql(
        """
        SELECT
            analytical_family_id,
            analytical_family_name,
            competitor_niche_id,
            competitor_niche_name,
            comparison_status,
            count(DISTINCT parent_asin) AS reviewed_product_count,
            count(*) AS review_count
        FROM read_parquet(?)
        GROUP BY ALL
        ORDER BY analytical_family_id, competitor_niche_id
        """,
        params=[str(temporary)],
    ).fetchdf()
    temporary.replace(destination)
    report = NicheReviewMaterializationReport(
        niche_id=niche.niche_id,
        niche_version=niche.niche_version,
        dataset_version=niche.dataset_version,
        review_count=len(review_ids),
        distinct_parent_asin_count=len(parent_ids),
        analytical_family_count=int(summary[0]),
        competitor_niche_count=int(summary[1]),
        quarantined_review_count=int(summary[2]),
        segment_distribution=segment_distribution_frame.to_dict("records"),
        output_path=_artifact_reference(destination),
        output_sha256=sha256_file(destination),
        excluded_columns=["user_id", "review_title", "review_body"],
    )
    if report_path is not None:
        Path(report_path).write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("niche_config_path", type=Path)
    parser.add_argument("reviews_path", type=Path)
    parser.add_argument("catalog_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    report = materialize_niche_reviews(
        args.niche_config_path,
        args.reviews_path,
        args.catalog_path,
        args.output_path,
        report_path=args.report_path,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
