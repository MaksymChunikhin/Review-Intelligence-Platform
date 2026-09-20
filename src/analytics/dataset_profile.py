"""Deterministic profiles for canonical review datasets."""

import argparse
import json
from pathlib import Path
from typing import Any


def _sql_path(path: Path) -> str:
    """Escape a filesystem path for a DuckDB SQL string literal."""
    return str(path.resolve()).replace("'", "''")


def profile_canonical_reviews(path: str | Path) -> dict[str, Any]:
    """Calculate exact core statistics from a canonical review Parquet."""
    try:
        import duckdb
    except ImportError as error:
        raise RuntimeError("DuckDB is required for dataset profiling") from error

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = _sql_path(source_path)
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        summary_row = connection.execute(
            f"""
            SELECT
                count(*) AS review_count,
                count(DISTINCT asin) AS asin_count,
                count(DISTINCT parent_asin) AS parent_asin_count,
                count(DISTINCT user_id) AS user_count,
                min(review_timestamp) AS min_review_timestamp,
                max(review_timestamp) AS max_review_timestamp,
                count(*) FILTER (WHERE verified_purchase) AS verified_count,
                sum(coalesce(helpful_vote, 0)) AS helpful_vote_sum,
                count(*) FILTER (WHERE user_id IS NULL) AS missing_user_id,
                count(*) FILTER (WHERE verified_purchase IS NULL)
                    AS missing_verified_purchase,
                count(*) FILTER (WHERE helpful_vote IS NULL) AS missing_helpful_vote,
                avg(length(review_text)) AS mean_review_length,
                quantile_cont(length(review_text), [0.5, 0.9, 0.99])
                    AS review_length_quantiles
            FROM read_parquet('{source}')
            """
        ).fetchone()
        summary_columns = [column[0] for column in connection.description]
        summary = dict(zip(summary_columns, summary_row, strict=True))

        rating_rows = connection.execute(
            f"""
            SELECT rating, count(*) AS review_count
            FROM read_parquet('{source}')
            GROUP BY rating
            ORDER BY rating
            """
        ).fetchall()
        year_rows = connection.execute(
            f"""
            SELECT year(review_timestamp) AS year, count(*) AS review_count
            FROM read_parquet('{source}')
            GROUP BY year(review_timestamp)
            ORDER BY year
            """
        ).fetchall()
    finally:
        connection.close()

    for key in ("min_review_timestamp", "max_review_timestamp"):
        if summary[key] is not None:
            summary[key] = summary[key].isoformat()
    summary["mean_review_length"] = float(summary["mean_review_length"])
    summary["review_length_quantiles"] = [
        float(value) for value in summary["review_length_quantiles"]
    ]
    return {
        "profile_version": "1.0",
        "path": str(source_path),
        "summary": summary,
        "rating_distribution": [
            {"rating": float(rating), "review_count": int(count)}
            for rating, count in rating_rows
        ],
        "year_distribution": [
            {"year": int(year), "review_count": int(count)}
            for year, count in year_rows
        ],
    }


def main() -> None:
    """Calculate and optionally persist a canonical review profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews_path", type=Path)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    profile = profile_canonical_reviews(args.reviews_path)
    payload = json.dumps(profile, indent=2)
    print(payload)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = args.output_path.with_suffix(".tmp.json")
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(args.output_path)


if __name__ == "__main__":
    main()
