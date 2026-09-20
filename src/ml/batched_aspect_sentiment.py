"""Score aspect evidence sentiment in bounded parquet batches."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.dataset_manifest import sha256_file
from src.ml.aspect_sentiment import (
    AspectSentimentReport,
    _artifact_reference,
    build_aspect_context,
)
from src.ml.sentiment import predict_sentiment


def score_aspect_sentiment_batched(
    predictions_path: str | Path,
    reviews_path: str | Path,
    model_directory: str | Path,
    output_path: str | Path,
    *,
    aspect_sentiment_version: str,
    report_path: str | Path | None = None,
    batch_size: int = 25_000,
    max_context_characters: int = 1_200,
) -> AspectSentimentReport:
    """Attach sentiment to large review-aspect artifacts without unbounded RAM."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    predictions_source = Path(predictions_path)
    reviews_source = Path(reviews_path)
    model_path = Path(model_directory)
    destination = Path(output_path)
    review_columns = pq.ParquetFile(reviews_source).schema_arrow.names
    columns = ["review_id", "review_text"]
    if "sampling_component" in review_columns:
        columns.append("sampling_component")
    reviews = pd.read_parquet(reviews_source, columns=columns)
    if reviews["review_id"].duplicated().any() or "review_text" not in reviews:
        raise ValueError("reviews require unique IDs and review_text")
    text_by_review = reviews.set_index("review_id")["review_text"].astype(str)
    available_review_ids = set(text_by_review.index)
    component_by_review = (
        reviews.set_index("review_id")["sampling_component"].astype(str)
        if "sampling_component" in reviews.columns
        else None
    )
    vectorizer = joblib.load(model_path / "vectorizer.joblib")
    classifier = joblib.load(model_path / "classifier.joblib")
    experiment = json.loads(
        (model_path / "experiment_config.json").read_text(encoding="utf-8")
    )
    sentiment_model_version = str(experiment["model_config"]["model_version"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    writer: pq.ParquetWriter | None = None
    extraction_versions: set[str] = set()
    taxonomy_versions: set[str] = set()
    review_ids: set[str] = set()
    aspect_ids: set[str] = set()
    sentiment_counts: dict[str, int] = {}
    component_counts: dict[str, dict[str, object]] = {}
    row_count = 0
    try:
        source = pq.ParquetFile(predictions_source)
        for record_batch in source.iter_batches(batch_size=batch_size):
            predictions = record_batch.to_pandas()
            required = {
                "extraction_version",
                "taxonomy_version",
                "review_id",
                "aspect_id",
                "evidence_json",
            }
            missing = required - set(predictions.columns)
            if missing:
                raise ValueError(
                    f"aspect predictions are missing columns: {sorted(missing)}"
                )
            extraction_versions.update(predictions["extraction_version"].astype(str))
            taxonomy_versions.update(predictions["taxonomy_version"].astype(str))
            missing_ids = set(predictions["review_id"]) - available_review_ids
            if missing_ids:
                raise ValueError("aspect predictions reference absent reviews")
            contexts = [
                build_aspect_context(
                    text_by_review.loc[row.review_id],
                    row.evidence_json,
                    max_characters=max_context_characters,
                )
                for row in predictions.itertuples(index=False)
            ]
            scored = predict_sentiment(
                pd.DataFrame({"review_text": contexts}),
                vectorizer=vectorizer,
                classifier=classifier,
            )
            result = predictions.copy()
            result["aspect_context"] = contexts
            result["aspect_sentiment"] = scored["predicted_label"].to_numpy()
            result["aspect_sentiment_score"] = scored["model_score"].to_numpy()
            for label in ("negative", "neutral", "positive"):
                column = f"score_{label}"
                result[f"aspect_{column}"] = scored[column].to_numpy()
            result["aspect_sentiment_version"] = aspect_sentiment_version
            result["sentiment_model_version"] = sentiment_model_version
            result["sentiment_method"] = (
                "tfidf_review_model_on_evidence_context_v1"
            )
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)

            row_count += len(result)
            review_ids.update(result["review_id"].astype(str))
            aspect_ids.update(result["aspect_id"].astype(str))
            for name, count in result["aspect_sentiment"].value_counts().items():
                sentiment_counts[str(name)] = sentiment_counts.get(str(name), 0) + int(count)
            if component_by_review is not None:
                components = result["review_id"].map(component_by_review)
                for component in sorted(components.unique()):
                    selected = result[components == component]
                    bucket = component_counts.setdefault(
                        str(component),
                        {"review_ids": set(), "review_aspect_count": 0, "sentiments": {}},
                    )
                    bucket["review_ids"].update(selected["review_id"].astype(str))
                    bucket["review_aspect_count"] += len(selected)
                    for name, count in selected["aspect_sentiment"].value_counts().items():
                        sentiments = bucket["sentiments"]
                        sentiments[str(name)] = sentiments.get(str(name), 0) + int(count)
        if writer is None:
            raise ValueError("aspect predictions must not be empty")
        writer.close()
        writer = None
        temporary.replace(destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    if len(extraction_versions) != 1 or len(taxonomy_versions) != 1:
        raise ValueError("aspect predictions must use one extraction and taxonomy version")

    component_rows = [
        {
            "sampling_component": component,
            "review_aspect_count": int(bucket["review_aspect_count"]),
            "review_count": len(bucket["review_ids"]),
            "sentiment_counts": dict(sorted(bucket["sentiments"].items())),
        }
        for component, bucket in sorted(component_counts.items())
    ]
    report = AspectSentimentReport(
        aspect_sentiment_version=aspect_sentiment_version,
        extraction_version=next(iter(extraction_versions)),
        taxonomy_version=next(iter(taxonomy_versions)),
        sentiment_model_version=sentiment_model_version,
        method="tfidf_review_model_on_evidence_context_v1",
        predictions_path=_artifact_reference(predictions_source),
        predictions_sha256=sha256_file(predictions_source),
        reviews_path=_artifact_reference(reviews_source),
        reviews_sha256=sha256_file(reviews_source),
        model_directory=_artifact_reference(model_path),
        output_path=_artifact_reference(destination),
        output_sha256=sha256_file(destination),
        review_aspect_count=row_count,
        review_count=len(review_ids),
        distinct_aspect_count=len(aspect_ids),
        sentiment_counts=dict(sorted(sentiment_counts.items())),
        human_gold_evaluated=False,
        warning=(
            "This reuses a weak-label review sentiment model on local evidence "
            "contexts. Counts are baseline predictions, not validated aspect "
            "sentiment metrics."
        ),
        by_sampling_component=component_rows,
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
    parser.add_argument("predictions_path", type=Path)
    parser.add_argument("reviews_path", type=Path)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--aspect-sentiment-version", required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=25_000)
    args = parser.parse_args()
    report = score_aspect_sentiment_batched(
        args.predictions_path,
        args.reviews_path,
        args.model_directory,
        args.output_path,
        aspect_sentiment_version=args.aspect_sentiment_version,
        report_path=args.report_path,
        batch_size=args.batch_size,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
