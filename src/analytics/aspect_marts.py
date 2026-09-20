"""Build product-aspect and month-aspect marts for offline seller analytics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file


SELLER_ATTENTION_POLICY_VERSION = "amazon_rating_attention_v1"


@dataclass(frozen=True)
class AspectMartsReport:
    marts_version: str
    taxonomy_version: str
    extraction_version: str
    aspect_sentiment_version: str
    seller_attention_policy_version: str
    review_population_count: int
    reviewed_product_count: int
    product_aspect_product_count: int
    source_review_aspect_count: int
    product_aspect_row_count: int
    product_aspect_reconciled_count: int
    month_aspect_row_count: int
    month_aspect_reconciled_count: int
    month_count: int
    aspect_count: int
    product_aspect_path: str
    product_aspect_sha256: str
    month_aspect_path: str
    month_aspect_sha256: str
    month_aspect_csv_path: str
    month_aspect_csv_sha256: str
    reliability_path: str | None
    warning: str


def _artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _top_evidence(
    joined: pd.DataFrame,
    sentiment: str,
    prefix: str,
) -> pd.DataFrame:
    selected = (
        joined[joined["aspect_sentiment"] == sentiment]
        .sort_values(
            [
                "parent_asin",
                "aspect_id",
                "aspect_sentiment_score",
                "helpful_vote",
                "aspect_probability",
            ],
            ascending=[True, True, False, False, False],
        )
        .drop_duplicates(["parent_asin", "aspect_id"])
    )
    return selected[
        ["parent_asin", "aspect_id", "review_id", "evidence_json"]
    ].rename(
        columns={
            "review_id": f"top_{prefix}_review_id",
            "evidence_json": f"top_{prefix}_evidence_json",
        }
    )


def build_aspect_marts(
    taxonomy_path: str | Path,
    reviews_path: str | Path,
    aspect_sentiment_path: str | Path,
    product_aspect_path: str | Path,
    month_aspect_path: str | Path,
    month_aspect_csv_path: str | Path,
    *,
    marts_version: str,
    report_path: str | Path | None = None,
    reliability_path: str | Path | None = None,
) -> AspectMartsReport:
    """Create reconciled product and monthly analytical marts."""
    taxonomy_source = Path(taxonomy_path)
    reviews_source = Path(reviews_path)
    sentiment_source = Path(aspect_sentiment_path)
    product_destination = Path(product_aspect_path)
    month_destination = Path(month_aspect_path)
    month_csv_destination = Path(month_aspect_csv_path)
    taxonomy = json.loads(taxonomy_source.read_text(encoding="utf-8"))
    if taxonomy.get("status") != "approved":
        raise ValueError("aspect marts require an approved taxonomy")
    definitions = {
        str(item["aspect_id"]): item for item in taxonomy["aspects"]
    }
    aspect_ids = list(definitions)
    reliability_by_aspect: dict[str, dict[str, Any]] = {}
    if reliability_path is not None:
        reliability = json.loads(Path(reliability_path).read_text(encoding="utf-8"))
        if reliability.get("taxonomy_version") != taxonomy.get("taxonomy_version"):
            raise ValueError("reliability and taxonomy versions differ")
        reliability_by_aspect = {
            str(item["aspect_id"]): item for item in reliability["aspects"]
        }
        if set(reliability_by_aspect) != set(aspect_ids):
            raise ValueError("reliability policy must cover every taxonomy aspect")

    review_columns = pq.ParquetFile(reviews_source).schema_arrow.names
    scope_columns = [
        column
        for column in (
            "analytical_family_id",
            "analytical_family_name",
            "competitor_niche_id",
            "competitor_niche_name",
            "comparison_status",
        )
        if column in review_columns
    ]
    reviews = pd.read_parquet(
        reviews_source,
        columns=[
            "review_id",
            "parent_asin",
            "rating",
            "review_timestamp",
            "helpful_vote",
            "product_title",
            "store",
            *scope_columns,
        ],
    )
    predictions = pd.read_parquet(sentiment_source)
    if reviews["review_id"].duplicated().any():
        raise ValueError("review population IDs must be unique")
    if predictions.duplicated(["review_id", "aspect_id"]).any():
        raise ValueError("aspect predictions must have unique review-aspect pairs")
    unknown = set(predictions["aspect_id"].astype(str)) - set(aspect_ids)
    if unknown:
        raise ValueError(f"aspect predictions contain unknown IDs: {sorted(unknown)}")
    versions = {
        "taxonomy_version": set(predictions["taxonomy_version"].astype(str)),
        "extraction_version": set(predictions["extraction_version"].astype(str)),
        "aspect_sentiment_version": set(
            predictions["aspect_sentiment_version"].astype(str)
        ),
    }
    if any(len(value) != 1 for value in versions.values()):
        raise ValueError("aspect mart inputs must contain one version per boundary")
    taxonomy_version = next(iter(versions["taxonomy_version"]))
    if taxonomy_version != str(taxonomy["taxonomy_version"]):
        raise ValueError("prediction and approved taxonomy versions differ")

    joined = predictions.merge(reviews, on="review_id", validate="many_to_one")
    if len(joined) != len(predictions):
        raise ValueError("some aspect predictions have no source review")
    joined["needs_attention"] = joined["rating"] <= 3
    joined["satisfied"] = joined["rating"] >= 4
    for sentiment in ("negative", "neutral", "positive"):
        joined[f"is_{sentiment}"] = joined["aspect_sentiment"] == sentiment
    joined["has_exact_alias"] = joined["prediction_sources_json"].map(
        lambda value: "exact_alias" in json.loads(value)
    )
    joined["year_month"] = (
        pd.to_datetime(joined["review_timestamp"], utc=True)
        .dt.strftime("%Y-%m")
    )

    for column in scope_columns:
        conflicting = reviews.groupby("parent_asin")[column].nunique(dropna=False)
        if (conflicting > 1).any():
            raise ValueError(f"products map to multiple {column} values")
    product_aggregations: dict[str, tuple[str, str]] = {
        "product_review_count": ("review_id", "size"),
        "product_average_rating": ("rating", "mean"),
        "product_title": ("product_title", "first"),
        "store": ("store", "first"),
    }
    product_aggregations.update(
        {column: (column, "first") for column in scope_columns}
    )
    product_review_counts = reviews.groupby("parent_asin").agg(
        **product_aggregations
    )
    product = joined.groupby(["parent_asin", "aspect_id"], sort=True).agg(
        aspect_review_count=("review_id", "size"),
        mean_rating=("rating", "mean"),
        needs_attention_count=("needs_attention", "sum"),
        satisfied_count=("satisfied", "sum"),
        negative_count=("is_negative", "sum"),
        neutral_count=("is_neutral", "sum"),
        positive_count=("is_positive", "sum"),
        mean_aspect_probability=("aspect_probability", "mean"),
        exact_alias_evidence_count=("has_exact_alias", "sum"),
        first_review_timestamp=("review_timestamp", "min"),
        last_review_timestamp=("review_timestamp", "max"),
    ).reset_index()
    product = product.merge(
        product_review_counts.reset_index(), on="parent_asin", validate="many_to_one"
    )
    product["canonical_name"] = product["aspect_id"].map(
        {key: str(value["canonical_name"]) for key, value in definitions.items()}
    )
    product["parent_group"] = product["aspect_id"].map(
        {key: str(value["parent_group"]) for key, value in definitions.items()}
    )
    if reliability_by_aspect:
        product["reliability_tier"] = product["aspect_id"].map(
            {key: value["reliability_tier"] for key, value in reliability_by_aspect.items()}
        )
        product["diagnostic_f1"] = product["aspect_id"].map(
            {key: value["f1"] for key, value in reliability_by_aspect.items()}
        )
        product["comparative_analytics_allowed"] = product["aspect_id"].map(
            {
                key: value["comparative_analytics_allowed"]
                for key, value in reliability_by_aspect.items()
            }
        )
    product["aspect_review_share"] = (
        product["aspect_review_count"] / product["product_review_count"]
    )
    product["needs_attention_share"] = (
        product["needs_attention_count"] / product["aspect_review_count"]
    )
    product["satisfied_share"] = (
        product["satisfied_count"] / product["aspect_review_count"]
    )
    for sentiment in ("negative", "neutral", "positive"):
        product[f"{sentiment}_share"] = (
            product[f"{sentiment}_count"] / product["aspect_review_count"]
        )
    product["exact_alias_evidence_share"] = (
        product["exact_alias_evidence_count"] / product["aspect_review_count"]
    )
    product["aspect_frequency_rank"] = product.groupby("parent_asin")[
        "aspect_review_count"
    ].rank(method="dense", ascending=False).astype(int)
    product["complaint_rank"] = product.groupby("parent_asin")[
        "needs_attention_count"
    ].rank(method="dense", ascending=False).astype(int)
    product["strength_rank"] = product.groupby("parent_asin")[
        "positive_count"
    ].rank(method="dense", ascending=False).astype(int)
    product = product.merge(
        _top_evidence(joined, "negative", "negative"),
        on=["parent_asin", "aspect_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        _top_evidence(joined, "positive", "positive"),
        on=["parent_asin", "aspect_id"],
        how="left",
        validate="one_to_one",
    )
    product.insert(0, "seller_attention_policy_version", SELLER_ATTENTION_POLICY_VERSION)
    product.insert(0, "marts_version", marts_version)
    product = product.sort_values(
        ["parent_asin", "aspect_frequency_rank", "aspect_id"]
    ).reset_index(drop=True)

    reviews_for_month = reviews.copy()
    reviews_for_month["year_month"] = (
        pd.to_datetime(reviews_for_month["review_timestamp"], utc=True)
        .dt.strftime("%Y-%m")
    )
    monthly_population = reviews_for_month.groupby("year_month").agg(
        month_review_population_count=("review_id", "size"),
        month_product_population_count=("parent_asin", "nunique"),
    )
    month = joined.groupby(["year_month", "aspect_id"], sort=True).agg(
        aspect_review_count=("review_id", "size"),
        distinct_product_count=("parent_asin", "nunique"),
        mean_rating=("rating", "mean"),
        needs_attention_count=("needs_attention", "sum"),
        satisfied_count=("satisfied", "sum"),
        negative_count=("is_negative", "sum"),
        neutral_count=("is_neutral", "sum"),
        positive_count=("is_positive", "sum"),
        mean_aspect_probability=("aspect_probability", "mean"),
        exact_alias_evidence_count=("has_exact_alias", "sum"),
    )
    all_months = sorted(monthly_population.index.astype(str))
    full_index = pd.MultiIndex.from_product(
        [all_months, aspect_ids], names=["year_month", "aspect_id"]
    )
    month = month.reindex(full_index).reset_index()
    count_columns = [
        "aspect_review_count",
        "distinct_product_count",
        "needs_attention_count",
        "satisfied_count",
        "negative_count",
        "neutral_count",
        "positive_count",
        "exact_alias_evidence_count",
    ]
    month[count_columns] = month[count_columns].fillna(0).astype(int)
    month = month.merge(
        monthly_population.reset_index(), on="year_month", validate="many_to_one"
    )
    month["canonical_name"] = month["aspect_id"].map(
        {key: str(value["canonical_name"]) for key, value in definitions.items()}
    )
    month["parent_group"] = month["aspect_id"].map(
        {key: str(value["parent_group"]) for key, value in definitions.items()}
    )
    if reliability_by_aspect:
        month["reliability_tier"] = month["aspect_id"].map(
            {key: value["reliability_tier"] for key, value in reliability_by_aspect.items()}
        )
        month["diagnostic_f1"] = month["aspect_id"].map(
            {key: value["f1"] for key, value in reliability_by_aspect.items()}
        )
        month["comparative_analytics_allowed"] = month["aspect_id"].map(
            {
                key: value["comparative_analytics_allowed"]
                for key, value in reliability_by_aspect.items()
            }
        )
    month["aspect_review_share"] = (
        month["aspect_review_count"] / month["month_review_population_count"]
    )
    denominator = month["aspect_review_count"].replace(0, np.nan)
    month["needs_attention_share"] = month["needs_attention_count"] / denominator
    month["satisfied_share"] = month["satisfied_count"] / denominator
    for sentiment in ("negative", "neutral", "positive"):
        month[f"{sentiment}_share"] = month[f"{sentiment}_count"] / denominator
    month["exact_alias_evidence_share"] = (
        month["exact_alias_evidence_count"] / denominator
    )
    month = month.sort_values(["aspect_id", "year_month"]).reset_index(drop=True)
    month["previous_month_aspect_review_count"] = month.groupby("aspect_id")[
        "aspect_review_count"
    ].shift(1)
    month["month_over_month_count_change"] = (
        month["aspect_review_count"] - month["previous_month_aspect_review_count"]
    )
    previous_share = month.groupby("aspect_id")["aspect_review_share"].shift(1)
    month["month_over_month_share_change"] = month["aspect_review_share"] - previous_share
    month.insert(0, "seller_attention_policy_version", SELLER_ATTENTION_POLICY_VERSION)
    month.insert(0, "marts_version", marts_version)
    month = month.sort_values(["year_month", "aspect_id"]).reset_index(drop=True)

    product_reconciled = int(product["aspect_review_count"].sum())
    month_reconciled = int(month["aspect_review_count"].sum())
    if product_reconciled != len(joined) or month_reconciled != len(joined):
        raise RuntimeError("aspect marts do not reconcile to source observations")
    _write_parquet(product, product_destination)
    _write_parquet(month, month_destination)
    month_csv_destination.parent.mkdir(parents=True, exist_ok=True)
    month.to_csv(month_csv_destination, index=False)
    report = AspectMartsReport(
        marts_version=marts_version,
        taxonomy_version=taxonomy_version,
        extraction_version=next(iter(versions["extraction_version"])),
        aspect_sentiment_version=next(iter(versions["aspect_sentiment_version"])),
        seller_attention_policy_version=SELLER_ATTENTION_POLICY_VERSION,
        review_population_count=len(reviews),
        reviewed_product_count=int(reviews["parent_asin"].nunique()),
        product_aspect_product_count=int(product["parent_asin"].nunique()),
        source_review_aspect_count=len(joined),
        product_aspect_row_count=len(product),
        product_aspect_reconciled_count=product_reconciled,
        month_aspect_row_count=len(month),
        month_aspect_reconciled_count=month_reconciled,
        month_count=len(all_months),
        aspect_count=len(aspect_ids),
        product_aspect_path=_artifact_reference(product_destination),
        product_aspect_sha256=sha256_file(product_destination),
        month_aspect_path=_artifact_reference(month_destination),
        month_aspect_sha256=sha256_file(month_destination),
        month_aspect_csv_path=_artifact_reference(month_csv_destination),
        month_aspect_csv_sha256=sha256_file(month_csv_destination),
        reliability_path=(
            _artifact_reference(Path(reliability_path))
            if reliability_path is not None
            else None
        ),
        warning=(
            "Marts use candidate aspect extraction and weak-label aspect sentiment. "
            "seller_attention is deterministic from 1–3 versus 4–5 star ratings."
        ),
    )
    if report_path is not None:
        Path(report_path).write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("taxonomy_path", type=Path)
    parser.add_argument("reviews_path", type=Path)
    parser.add_argument("aspect_sentiment_path", type=Path)
    parser.add_argument("product_aspect_path", type=Path)
    parser.add_argument("month_aspect_path", type=Path)
    parser.add_argument("month_aspect_csv_path", type=Path)
    parser.add_argument("--marts-version", required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--reliability-path", type=Path)
    args = parser.parse_args()
    report = build_aspect_marts(
        args.taxonomy_path,
        args.reviews_path,
        args.aspect_sentiment_path,
        args.product_aspect_path,
        args.month_aspect_path,
        args.month_aspect_csv_path,
        marts_version=args.marts_version,
        report_path=args.report_path,
        reliability_path=args.reliability_path,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
