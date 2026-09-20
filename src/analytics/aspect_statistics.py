"""Build the first offline aspect-level business statistics table."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file


@dataclass(frozen=True)
class AspectStatisticsReport:
    statistics_version: str
    extraction_version: str
    aspect_sentiment_version: str
    taxonomy_version: str
    review_population_count: int
    matched_review_count: int
    review_aspect_count: int
    aspect_count: int
    output_path: str
    output_sha256: str
    csv_path: str
    csv_sha256: str
    warning: str


def _artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _examples(group: pd.DataFrame, sentiment: str, count: int = 3) -> str:
    selected = group[group["aspect_sentiment"] == sentiment].sort_values(
        ["aspect_sentiment_score", "helpful_vote", "aspect_probability"],
        ascending=False,
    ).head(count)
    values = []
    for row in selected.itertuples(index=False):
        evidence = json.loads(row.evidence_json)
        values.append(
            {
                "review_id": str(row.review_id),
                "rating": float(row.rating),
                "helpful_vote": int(row.helpful_vote),
                "evidence": [str(item["text"]) for item in evidence],
            }
        )
    return json.dumps(values, ensure_ascii=False)


def build_aspect_statistics(
    taxonomy_path: str | Path,
    reviews_path: str | Path,
    aspect_sentiment_path: str | Path,
    output_path: str | Path,
    csv_path: str | Path,
    *,
    statistics_version: str,
    report_path: str | Path | None = None,
) -> AspectStatisticsReport:
    """Aggregate review-aspect predictions into a compact analytics table."""
    taxonomy_source = Path(taxonomy_path)
    reviews_source = Path(reviews_path)
    sentiment_source = Path(aspect_sentiment_path)
    destination = Path(output_path)
    csv_destination = Path(csv_path)
    taxonomy = json.loads(taxonomy_source.read_text(encoding="utf-8"))
    if taxonomy.get("status") != "approved":
        raise ValueError("aspect statistics require an approved taxonomy")
    definitions = {
        str(item["aspect_id"]): item for item in taxonomy["aspects"]
    }
    reviews = pd.read_parquet(
        reviews_source,
        columns=[
            "review_id", "parent_asin", "rating", "helpful_vote",
            "review_timestamp",
        ],
    )
    predictions = pd.read_parquet(sentiment_source)
    if reviews["review_id"].duplicated().any():
        raise ValueError("review population IDs must be unique")
    if predictions.duplicated(["review_id", "aspect_id"]).any():
        raise ValueError("aspect predictions must have unique review-aspect pairs")
    unknown = set(predictions["aspect_id"].astype(str)) - set(definitions)
    if unknown:
        raise ValueError(f"predictions contain unknown aspects: {sorted(unknown)}")
    extraction_versions = set(predictions["extraction_version"].astype(str))
    sentiment_versions = set(predictions["aspect_sentiment_version"].astype(str))
    taxonomy_versions = set(predictions["taxonomy_version"].astype(str))
    if len(extraction_versions) != 1 or len(sentiment_versions) != 1:
        raise ValueError("statistics inputs must use one extraction/sentiment version")
    if taxonomy_versions != {str(taxonomy["taxonomy_version"])}:
        raise ValueError("prediction taxonomy version does not match taxonomy")
    joined = predictions.merge(reviews, on="review_id", validate="many_to_one")
    if len(joined) != len(predictions):
        raise ValueError("some aspect predictions have no source review")
    joined["needs_attention"] = joined["rating"] <= 3

    rows: list[dict[str, Any]] = []
    for aspect_id, definition in definitions.items():
        group = joined[joined["aspect_id"] == aspect_id]
        counts = group["aspect_sentiment"].value_counts()
        assignment_count = len(group)
        source_values = [
            set(json.loads(value)) for value in group["prediction_sources_json"]
        ]
        exact_alias_count = sum("exact_alias" in value for value in source_values)
        if assignment_count == 0:
            raise ValueError(f"approved aspect has no population predictions: {aspect_id}")
        rows.append(
            {
                "statistics_version": statistics_version,
                "extraction_version": next(iter(extraction_versions)),
                "aspect_sentiment_version": next(iter(sentiment_versions)),
                "taxonomy_version": str(taxonomy["taxonomy_version"]),
                "aspect_id": aspect_id,
                "canonical_name": str(definition["canonical_name"]),
                "parent_group": str(definition["parent_group"]),
                "review_count": assignment_count,
                "review_share": assignment_count / len(reviews),
                "distinct_product_count": int(group["parent_asin"].nunique()),
                "mean_rating": float(group["rating"].mean()),
                "needs_attention_count": int(group["needs_attention"].sum()),
                "needs_attention_share": float(group["needs_attention"].mean()),
                "negative_count": int(counts.get("negative", 0)),
                "neutral_count": int(counts.get("neutral", 0)),
                "positive_count": int(counts.get("positive", 0)),
                "negative_share": float(counts.get("negative", 0) / assignment_count),
                "neutral_share": float(counts.get("neutral", 0) / assignment_count),
                "positive_share": float(counts.get("positive", 0) / assignment_count),
                "mean_aspect_probability": float(group["aspect_probability"].mean()),
                "exact_alias_evidence_count": exact_alias_count,
                "exact_alias_evidence_share": exact_alias_count / assignment_count,
                "top_negative_examples_json": _examples(group, "negative"),
                "top_positive_examples_json": _examples(group, "positive"),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["review_count", "aspect_id"], ascending=[False, True]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    result.to_parquet(temporary, index=False)
    temporary.replace(destination)
    result.drop(
        columns=["top_negative_examples_json", "top_positive_examples_json"]
    ).to_csv(csv_destination, index=False)
    report = AspectStatisticsReport(
        statistics_version=statistics_version,
        extraction_version=next(iter(extraction_versions)),
        aspect_sentiment_version=next(iter(sentiment_versions)),
        taxonomy_version=str(taxonomy["taxonomy_version"]),
        review_population_count=len(reviews),
        matched_review_count=int(joined["review_id"].nunique()),
        review_aspect_count=len(joined),
        aspect_count=len(result),
        output_path=_artifact_reference(destination),
        output_sha256=sha256_file(destination),
        csv_path=_artifact_reference(csv_destination),
        csv_sha256=sha256_file(csv_destination),
        warning=(
            "Statistics use a silver-trained candidate extractor and weak-label "
            "aspect sentiment. They are diagnostic business analytics."
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
    parser.add_argument("output_path", type=Path)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--statistics-version", required=True)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    report = build_aspect_statistics(
        args.taxonomy_path,
        args.reviews_path,
        args.aspect_sentiment_path,
        args.output_path,
        args.csv_path,
        statistics_version=args.statistics_version,
        report_path=args.report_path,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
