"""Run the current hybrid aspect extractor in bounded local batches."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.analytics.aspect_extraction import (
    build_alias_matcher,
    extract_review_aspects,
)
from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file
from src.ml.aspect_multilabel import AspectMultilabelConfig


@dataclass(frozen=True)
class AspectInferenceReport:
    model_version: str
    taxonomy_version: str
    method: str
    threshold: float
    input_path: str
    input_sha256: str
    input_review_count: int
    matched_review_count: int
    no_match_review_count: int
    review_aspect_count: int
    distinct_aspect_count: int
    source_counts: dict[str, int]
    batch_size: int
    output_path: str
    output_sha256: str
    human_gold_accepted: bool
    warning: str


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("extraction_version", pa.string(), nullable=False),
        pa.field("taxonomy_version", pa.string(), nullable=False),
        pa.field("method", pa.string(), nullable=False),
        pa.field("review_id", pa.string(), nullable=False),
        pa.field("aspect_id", pa.string(), nullable=False),
        pa.field("evidence_count", pa.int16(), nullable=False),
        pa.field("matched_aliases_json", pa.string(), nullable=False),
        pa.field("evidence_json", pa.string(), nullable=False),
        pa.field("aspect_probability", pa.float32(), nullable=False),
        pa.field("prediction_sources_json", pa.string(), nullable=False),
    ]
)


def _artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = [
        (match.start(), match.end())
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text)
        if match.group(0).strip()
    ]
    return spans or [(0, len(text))]


def _semantic_evidence_from_review_features(
    text: str,
    aspect_index: int,
    review_features: Any,
    classifier: Any,
    feature_names: np.ndarray,
) -> dict[str, Any]:
    """Select exact sentence evidence without re-vectorizing every sentence."""
    spans = _sentence_spans(text)
    selected_span = spans[0]
    estimator = classifier.estimators_[aspect_index]
    if hasattr(estimator, "coef_") and review_features.nnz:
        coefficients = estimator.coef_[0, review_features.indices]
        contributions = review_features.data * coefficients
        ordered = np.argsort(contributions)[::-1]
        for position in ordered[:20]:
            feature = str(feature_names[review_features.indices[position]]).strip()
            if len(feature) < 2:
                continue
            match = re.search(re.escape(feature), text, flags=re.IGNORECASE)
            if match is None:
                continue
            selected_span = next(
                (
                    (start, end)
                    for start, end in spans
                    if start <= match.start() and match.end() <= end
                ),
                selected_span,
            )
            break
    start, end = selected_span
    return {
        "text": text[start:end],
        "start": start,
        "end": end,
        "source": "semantic_feature_sentence_v2",
    }


def predict_aspects(
    config: AspectMultilabelConfig,
    taxonomy_path: str | Path,
    input_path: str | Path,
    model_directory: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    batch_size: int = 1_000,
) -> AspectInferenceReport:
    """Apply semantic and exact-alias extraction with exact sentence evidence."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    taxonomy_source = Path(taxonomy_path)
    input_source = Path(input_path)
    model_source = Path(model_directory)
    destination = Path(output_path)
    taxonomy = json.loads(taxonomy_source.read_text(encoding="utf-8"))
    matcher, rules = build_alias_matcher(taxonomy)
    aspect_ids = [str(item["aspect_id"]) for item in taxonomy["aspects"]]
    if taxonomy.get("taxonomy_version") != config.taxonomy_version:
        raise ValueError("taxonomy version does not match inference config")
    metadata = json.loads(
        (model_source / "model_metadata.json").read_text(encoding="utf-8")
    )
    expected = {
        "model_version": config.model_version,
        "taxonomy_version": config.taxonomy_version,
        "silver_version": config.silver_version,
        "method": config.method,
        "threshold": config.threshold,
        "classes": aspect_ids,
    }
    mismatch = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise ValueError(f"aspect model metadata mismatch: {mismatch}")
    vectorizer = joblib.load(model_source / "vectorizer.joblib")
    classifier = joblib.load(model_source / "classifier.joblib")
    feature_names = vectorizer.get_feature_names_out()
    reviews = pd.read_parquet(input_source, columns=["review_id", "review_text"])
    if reviews.empty or reviews["review_id"].isna().any():
        raise ValueError("inference reviews must be non-empty with defined IDs")
    if reviews["review_id"].duplicated().any() or reviews["review_text"].isna().any():
        raise ValueError("inference reviews and text must be unique/non-null")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    writer: pq.ParquetWriter | None = None
    matched_reviews: set[str] = set()
    aspect_counts: dict[str, int] = {}
    source_counts = {"exact_alias_only": 0, "char_tfidf_only": 0, "both": 0}
    review_aspect_count = 0
    try:
        writer = pq.ParquetWriter(temporary, OUTPUT_SCHEMA, compression="zstd")
        for start in range(0, len(reviews), batch_size):
            batch = reviews.iloc[start : start + batch_size]
            texts = batch["review_text"].astype(str).tolist()
            review_features = vectorizer.transform(texts)
            review_probabilities = classifier.predict_proba(review_features)

            output_rows: list[dict[str, Any]] = []
            for local_index, row in enumerate(batch.itertuples(index=False)):
                review_id = str(row.review_id)
                text = str(row.review_text)
                alias_values = {
                    value["aspect_id"]: value
                    for value in extract_review_aspects(
                        review_id, text, matcher=matcher, rules=rules
                    )
                }
                semantic_indices = set(
                    np.flatnonzero(
                        review_probabilities[local_index] >= config.threshold
                    ).tolist()
                )
                alias_indices = {aspect_ids.index(value) for value in alias_values}
                for aspect_index in sorted(semantic_indices | alias_indices):
                    aspect_id = aspect_ids[aspect_index]
                    alias = alias_values.get(aspect_id)
                    sources: list[str] = []
                    if alias is not None:
                        evidence_json = alias["evidence_json"]
                        matched_aliases_json = alias["matched_aliases_json"]
                        sources.append("exact_alias")
                    else:
                        evidence = _semantic_evidence_from_review_features(
                            text,
                            aspect_index,
                            review_features.getrow(local_index),
                            classifier,
                            feature_names,
                        )
                        evidence_json = json.dumps(
                            [evidence],
                            ensure_ascii=False,
                        )
                        matched_aliases_json = "[]"
                    if aspect_index in semantic_indices:
                        sources.append("char_tfidf")
                    source_key = (
                        "both"
                        if len(sources) == 2
                        else "exact_alias_only"
                        if sources == ["exact_alias"]
                        else "char_tfidf_only"
                    )
                    source_counts[source_key] += 1
                    matched_reviews.add(review_id)
                    aspect_counts[aspect_id] = aspect_counts.get(aspect_id, 0) + 1
                    output_rows.append(
                        {
                            "extraction_version": config.model_version,
                            "taxonomy_version": config.taxonomy_version,
                            "method": config.method,
                            "review_id": review_id,
                            "aspect_id": aspect_id,
                            "evidence_count": len(json.loads(evidence_json)),
                            "matched_aliases_json": matched_aliases_json,
                            "evidence_json": evidence_json,
                            "aspect_probability": float(
                                review_probabilities[local_index, aspect_index]
                            ),
                            "prediction_sources_json": json.dumps(sources),
                        }
                    )
            if output_rows:
                table = pa.Table.from_pylist(output_rows, schema=OUTPUT_SCHEMA)
                writer.write_table(table)
                review_aspect_count += len(output_rows)
        writer.close()
        writer = None
        temporary.replace(destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)

    report = AspectInferenceReport(
        model_version=config.model_version,
        taxonomy_version=config.taxonomy_version,
        method=config.method,
        threshold=config.threshold,
        input_path=_artifact_reference(input_source),
        input_sha256=sha256_file(input_source),
        input_review_count=len(reviews),
        matched_review_count=len(matched_reviews),
        no_match_review_count=len(reviews) - len(matched_reviews),
        review_aspect_count=review_aspect_count,
        distinct_aspect_count=len(aspect_counts),
        source_counts=source_counts,
        batch_size=batch_size,
        output_path=_artifact_reference(destination),
        output_sha256=sha256_file(destination),
        human_gold_accepted=False,
        warning=(
            "Population output from a Gemini-silver-trained candidate. It is "
            "suitable for diagnostic analytics, not final accuracy claims."
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
    parser.add_argument("config_path", type=Path)
    parser.add_argument("taxonomy_path", type=Path)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=1_000)
    args = parser.parse_args()
    config = AspectMultilabelConfig.model_validate_json(
        args.config_path.read_text(encoding="utf-8")
    )
    report = predict_aspects(
        config,
        args.taxonomy_path,
        args.input_path,
        args.model_directory,
        args.output_path,
        report_path=args.report_path,
        batch_size=args.batch_size,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
