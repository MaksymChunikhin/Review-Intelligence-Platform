"""Score sentiment around exact aspect evidence with a local baseline model."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file
from src.ml.sentiment import predict_sentiment


@dataclass(frozen=True)
class AspectSentimentReport:
    """Describe one evidence-context sentiment baseline run."""

    aspect_sentiment_version: str
    extraction_version: str
    taxonomy_version: str
    sentiment_model_version: str
    method: str
    predictions_path: str
    predictions_sha256: str
    reviews_path: str
    reviews_sha256: str
    model_directory: str
    output_path: str
    output_sha256: str
    review_aspect_count: int
    review_count: int
    distinct_aspect_count: int
    sentiment_counts: dict[str, int]
    human_gold_evaluated: bool
    warning: str
    by_sampling_component: list[dict[str, Any]]


def _artifact_reference(path: Path) -> str:
    """Prefer a repository-relative path in persisted provenance."""
    try:
        root = find_project_root(path.parent)
        return path.resolve().relative_to(root).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return simple sentence-like spans without changing source text."""
    spans = [
        (match.start(), match.end())
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text)
        if match.group(0).strip()
    ]
    return spans or [(0, len(text))]


def build_aspect_context(
    review_text: str,
    evidence_json: str,
    *,
    max_characters: int = 1_200,
) -> str:
    """Collect exact source sentences containing an aspect's evidence spans."""
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    evidence = json.loads(evidence_json)
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence_json must contain a non-empty JSON list")
    sentence_spans = _sentence_spans(review_text)
    selected: list[str] = []
    for item in evidence:
        start = int(item["start"])
        end = int(item["end"])
        text = str(item["text"])
        if start < 0 or end <= start or end > len(review_text):
            raise ValueError("aspect evidence offsets are outside review text")
        if review_text[start:end] != text:
            raise ValueError("aspect evidence text does not match its offsets")
        containing = next(
            (
                review_text[left:right].strip()
                for left, right in sentence_spans
                if left <= start and end <= right
            ),
            review_text[max(0, start - 300) : min(len(review_text), end + 300)].strip(),
        )
        if containing and containing not in selected:
            selected.append(containing)
    context = " ".join(selected).strip()
    if not context:
        raise ValueError("aspect evidence did not produce a context")
    return context[:max_characters]


def score_aspect_sentiment(
    predictions_path: str | Path,
    reviews_path: str | Path,
    model_directory: str | Path,
    output_path: str | Path,
    *,
    aspect_sentiment_version: str,
    report_path: str | Path | None = None,
    vectorizer: Any | None = None,
    classifier: Any | None = None,
    sentiment_model_version: str | None = None,
    max_context_characters: int = 1_200,
) -> AspectSentimentReport:
    """Attach local sentiment predictions to extracted review-aspect rows."""
    predictions_source = Path(predictions_path)
    reviews_source = Path(reviews_path)
    model_path = Path(model_directory)
    destination = Path(output_path)
    predictions = pd.read_parquet(predictions_source)
    reviews = pd.read_parquet(reviews_source)
    required_predictions = {
        "extraction_version",
        "taxonomy_version",
        "review_id",
        "aspect_id",
        "evidence_json",
    }
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_predictions:
        raise ValueError(
            "aspect predictions are missing columns: "
            f"{sorted(missing_predictions)}"
        )
    if predictions.empty:
        raise ValueError("aspect predictions must not be empty")
    if reviews["review_id"].duplicated().any() or reviews["review_id"].isna().any():
        raise ValueError("review IDs must be unique and non-null")
    if "review_text" not in reviews:
        raise ValueError("reviews are missing review_text")
    extraction_versions = set(predictions["extraction_version"].astype(str))
    taxonomy_versions = set(predictions["taxonomy_version"].astype(str))
    if len(extraction_versions) != 1 or len(taxonomy_versions) != 1:
        raise ValueError("aspect predictions must use one extraction and taxonomy version")

    review_text_by_id = reviews.set_index("review_id")["review_text"].astype(str)
    missing_review_ids = set(predictions["review_id"]) - set(review_text_by_id.index)
    if missing_review_ids:
        raise ValueError("aspect predictions reference reviews absent from input")
    contexts = [
        build_aspect_context(
            review_text_by_id.loc[row.review_id],
            row.evidence_json,
            max_characters=max_context_characters,
        )
        for row in predictions.itertuples(index=False)
    ]

    experiment_path = model_path / "experiment_config.json"
    if vectorizer is None or classifier is None:
        vectorizer = joblib.load(model_path / "vectorizer.joblib")
        classifier = joblib.load(model_path / "classifier.joblib")
    if sentiment_model_version is None:
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        sentiment_model_version = str(
            experiment["model_config"]["model_version"]
        )
    context_predictions = predict_sentiment(
        pd.DataFrame({"review_text": contexts}),
        vectorizer=vectorizer,
        classifier=classifier,
    )
    result = predictions.copy()
    result["aspect_context"] = contexts
    result["aspect_sentiment"] = context_predictions["predicted_label"].to_numpy()
    result["aspect_sentiment_score"] = context_predictions["model_score"].to_numpy()
    for label in ("negative", "neutral", "positive"):
        column = f"score_{label}"
        result[f"aspect_{column}"] = context_predictions[column].to_numpy()
    result["aspect_sentiment_version"] = aspect_sentiment_version
    result["sentiment_model_version"] = sentiment_model_version
    result["sentiment_method"] = "tfidf_review_model_on_evidence_context_v1"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        result.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    counts = result["aspect_sentiment"].value_counts().sort_index()
    component_rows: list[dict[str, Any]] = []
    if "sampling_component" in reviews.columns:
        component_by_review = reviews.set_index("review_id")[
            "sampling_component"
        ].astype(str)
        result_components = result["review_id"].map(component_by_review)
        for component in sorted(result_components.unique()):
            component_result = result[result_components == component]
            component_counts = (
                component_result["aspect_sentiment"].value_counts().sort_index()
            )
            component_rows.append(
                {
                    "sampling_component": component,
                    "review_aspect_count": int(len(component_result)),
                    "review_count": int(component_result["review_id"].nunique()),
                    "sentiment_counts": {
                        str(name): int(value)
                        for name, value in component_counts.items()
                    },
                }
            )
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
        review_aspect_count=len(result),
        review_count=int(result["review_id"].nunique()),
        distinct_aspect_count=int(result["aspect_id"].nunique()),
        sentiment_counts={str(name): int(value) for name, value in counts.items()},
        human_gold_evaluated=False,
        warning=(
            "This reuses a weak-label review sentiment model on local evidence "
            "contexts. Counts are baseline predictions, not validated aspect "
            "sentiment metrics; targeted rows are non-probability diagnostics."
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
    """Score local aspect contexts with a saved TF-IDF sentiment model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_path", type=Path)
    parser.add_argument("reviews_path", type=Path)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--aspect-sentiment-version", required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--max-context-characters", type=int, default=1_200)
    args = parser.parse_args()
    report = score_aspect_sentiment(
        args.predictions_path,
        args.reviews_path,
        args.model_directory,
        args.output_path,
        aspect_sentiment_version=args.aspect_sentiment_version,
        report_path=args.report_path,
        max_context_characters=args.max_context_characters,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
