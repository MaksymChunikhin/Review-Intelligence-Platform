"""Build trustworthy product rankings inside exact competitor niches."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file


@dataclass(frozen=True)
class NicheRankingsReport:
    rankings_version: str
    marts_version: str
    source_path: str
    source_sha256: str
    output_path: str
    output_sha256: str
    row_count: int
    product_count: int
    competitor_niche_count: int
    ranked_aspect_count: int
    excluded_aspect_count: int
    high_rating_recurring_complaint_count: int
    minimum_product_reviews: int
    minimum_aspect_mentions: int
    minimum_sentiment_mentions: int
    prior_strength: float
    warning: str


def _artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _rank_supported(
    frame: pd.DataFrame,
    *,
    value_column: str,
    support_column: str,
    minimum_support: int,
) -> pd.Series:
    ranks = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    supported = frame[support_column] >= minimum_support
    ranks.loc[supported] = (
        frame.loc[supported]
        .groupby(["competitor_niche_id", "aspect_id"])[value_column]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    return ranks


def build_niche_rankings(
    product_aspect_mart_path: str | Path,
    output_path: str | Path,
    *,
    rankings_version: str,
    report_path: str | Path | None = None,
    minimum_product_reviews: int = 10,
    minimum_aspect_mentions: int = 3,
    minimum_sentiment_mentions: int = 2,
    prior_strength: float = 10.0,
) -> NicheRankingsReport:
    """Rank product-aspect performance without mixing competitor niches."""
    if minimum_product_reviews < 1 or minimum_aspect_mentions < 1:
        raise ValueError("minimum review and aspect support must be positive")
    if minimum_sentiment_mentions < 1 or prior_strength <= 0:
        raise ValueError("sentiment support and prior strength must be positive")
    source = Path(product_aspect_mart_path)
    destination = Path(output_path)
    frame = pd.read_parquet(source)
    required = {
        "marts_version",
        "parent_asin",
        "aspect_id",
        "canonical_name",
        "competitor_niche_id",
        "competitor_niche_name",
        "comparison_status",
        "product_review_count",
        "product_average_rating",
        "aspect_review_count",
        "positive_count",
        "negative_count",
        "comparative_analytics_allowed",
        "reliability_tier",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"product aspect mart is missing columns: {sorted(missing)}")
    versions = set(frame["marts_version"].astype(str))
    if len(versions) != 1:
        raise ValueError("rankings require one marts version")
    if frame.duplicated(["parent_asin", "aspect_id"]).any():
        raise ValueError("product-aspect rows must be unique")

    excluded_aspects = frame.loc[
        ~frame["comparative_analytics_allowed"].astype(bool), "aspect_id"
    ].nunique()
    ranked = frame[
        frame["comparative_analytics_allowed"].astype(bool)
        & frame["comparison_status"].eq("eligible")
        & frame["competitor_niche_id"].notna()
        & frame["product_review_count"].ge(minimum_product_reviews)
        & frame["aspect_review_count"].ge(minimum_aspect_mentions)
    ].copy()
    if ranked.empty:
        raise ValueError("no product-aspect rows meet comparison policy")

    group_columns = ["competitor_niche_id", "aspect_id"]
    for sentiment in ("positive", "negative"):
        count_column = f"{sentiment}_count"
        group_count = ranked.groupby(group_columns)[count_column].transform("sum")
        group_mentions = ranked.groupby(group_columns)["aspect_review_count"].transform(
            "sum"
        )
        prior = group_count / group_mentions
        score_column = f"{sentiment}_smoothed_share"
        ranked[score_column] = (
            ranked[count_column] + prior_strength * prior
        ) / (ranked["aspect_review_count"] + prior_strength)
        ranked[f"{sentiment}_frequency_rank"] = _rank_supported(
            ranked,
            value_column=count_column,
            support_column=count_column,
            minimum_support=minimum_sentiment_mentions,
        )
        ranked[f"{sentiment}_quality_rank"] = _rank_supported(
            ranked,
            value_column=score_column,
            support_column=count_column,
            minimum_support=minimum_sentiment_mentions,
        )
        reason_rank = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
        supported = ranked[count_column] >= minimum_sentiment_mentions
        reason_rank.loc[supported] = (
            ranked.loc[supported]
            .groupby("parent_asin")[count_column]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )
        ranked[f"{sentiment}_reason_rank"] = reason_rank

    ranked["is_high_rating_recurring_complaint"] = (
        ranked["product_average_rating"].ge(4.0)
        & ranked["negative_count"].ge(3)
    )
    ranked.insert(0, "rankings_version", rankings_version)
    ranked = ranked.sort_values(
        ["competitor_niche_id", "aspect_id", "parent_asin"]
    ).reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        ranked.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    report = NicheRankingsReport(
        rankings_version=rankings_version,
        marts_version=next(iter(versions)),
        source_path=_artifact_reference(source),
        source_sha256=sha256_file(source),
        output_path=_artifact_reference(destination),
        output_sha256=sha256_file(destination),
        row_count=len(ranked),
        product_count=int(ranked["parent_asin"].nunique()),
        competitor_niche_count=int(ranked["competitor_niche_id"].nunique()),
        ranked_aspect_count=int(ranked["aspect_id"].nunique()),
        excluded_aspect_count=int(excluded_aspects),
        high_rating_recurring_complaint_count=int(
            ranked["is_high_rating_recurring_complaint"].sum()
        ),
        minimum_product_reviews=minimum_product_reviews,
        minimum_aspect_mentions=minimum_aspect_mentions,
        minimum_sentiment_mentions=minimum_sentiment_mentions,
        prior_strength=prior_strength,
        warning=(
            "Rankings are limited to exact competitor niches and strong/usable "
            "aspects. Scores use diagnostic silver-trained extraction and "
            "weak-label aspect sentiment, not human-gold product claims."
        ),
    )
    if report_path is not None:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        report_destination.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("product_aspect_mart_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--rankings-version", required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--minimum-product-reviews", type=int, default=10)
    parser.add_argument("--minimum-aspect-mentions", type=int, default=3)
    parser.add_argument("--minimum-sentiment-mentions", type=int, default=2)
    parser.add_argument("--prior-strength", type=float, default=10.0)
    args = parser.parse_args()
    report = build_niche_rankings(
        args.product_aspect_mart_path,
        args.output_path,
        rankings_version=args.rankings_version,
        report_path=args.report_path,
        minimum_product_reviews=args.minimum_product_reviews,
        minimum_aspect_mentions=args.minimum_aspect_mentions,
        minimum_sentiment_mentions=args.minimum_sentiment_mentions,
        prior_strength=args.prior_strength,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
