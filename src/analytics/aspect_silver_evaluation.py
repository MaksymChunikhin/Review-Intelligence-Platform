"""Compare a local aspect baseline with a clearly labelled silver reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file


def _artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _scores(
    predicted: set[tuple[str, str]], reference: set[tuple[str, str]]
) -> dict[str, float | int]:
    true_positive = len(predicted & reference)
    false_positive = len(predicted - reference)
    false_negative = len(reference - predicted)
    precision = true_positive / (true_positive + false_positive) if predicted else 0.0
    recall = true_positive / (true_positive + false_negative) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _weighted_scores(
    predicted: set[tuple[str, str]],
    reference: set[tuple[str, str]],
    weights: dict[str, float],
) -> dict[str, float]:
    true_positive = sum(weights[item[0]] for item in predicted & reference)
    false_positive = sum(weights[item[0]] for item in predicted - reference)
    false_negative = sum(weights[item[0]] for item in reference - predicted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive_weight": true_positive,
        "false_positive_weight": false_positive,
        "false_negative_weight": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_against_silver(
    sample_path: str | Path,
    prediction_path: str | Path,
    sentiment_path: str | Path,
    silver_path: str | Path,
) -> dict[str, Any]:
    """Return extraction and overlapping-pair sentiment diagnostics."""
    sample_source = Path(sample_path)
    prediction_source = Path(prediction_path)
    sentiment_source = Path(sentiment_path)
    silver_source = Path(silver_path)
    sample = pd.read_parquet(sample_source)
    required_sample_columns = {
        "review_id",
        "sampling_component",
        "sampling_weight",
    }
    missing_sample_columns = required_sample_columns - set(sample.columns)
    if missing_sample_columns:
        raise ValueError(
            "evaluation sample is missing columns: "
            f"{sorted(missing_sample_columns)}"
        )
    predictions = pd.read_parquet(
        prediction_source, columns=["review_id", "aspect_id"]
    )
    sentiments = pd.read_parquet(
        sentiment_source, columns=["review_id", "aspect_id", "aspect_sentiment"]
    )
    silver = pd.read_parquet(
        silver_source, columns=["review_id", "aspect_ids_json", "aspects_json"]
    )
    sample_ids = set(sample["review_id"].astype(str))
    silver_ids = set(silver["review_id"].astype(str))
    if sample_ids != silver_ids or len(sample) != len(silver):
        raise ValueError("silver reference must cover the evaluation sample exactly")
    if not set(predictions["review_id"].astype(str)) <= sample_ids:
        raise ValueError("predictions contain reviews outside the evaluation sample")

    predicted_pairs = {
        (str(row.review_id), str(row.aspect_id))
        for row in predictions.itertuples(index=False)
    }
    reference_pairs = {
        (str(row.review_id), str(aspect_id))
        for row in silver.itertuples(index=False)
        for aspect_id in json.loads(row.aspect_ids_json)
    }
    components = {
        str(component): set(group["review_id"].astype(str))
        for component, group in sample.groupby("sampling_component")
    }
    extraction: dict[str, Any] = {"overall": _scores(predicted_pairs, reference_pairs)}
    for component, review_ids in sorted(components.items()):
        predicted = {pair for pair in predicted_pairs if pair[0] in review_ids}
        reference = {pair for pair in reference_pairs if pair[0] in review_ids}
        extraction[component] = {
            **_scores(predicted, reference),
            "review_count": len(review_ids),
        }
    predicted_by_review = {
        review_id: {aspect for rid, aspect in predicted_pairs if rid == review_id}
        for review_id in sample_ids
    }
    reference_by_review = {
        review_id: {aspect for rid, aspect in reference_pairs if rid == review_id}
        for review_id in sample_ids
    }
    extraction["overall"].update(
        {
            "review_count": len(sample_ids),
            "exact_review_set_match_count": sum(
                predicted_by_review[review_id] == reference_by_review[review_id]
                for review_id in sample_ids
            ),
            "prediction_pair_count": len(predicted_pairs),
            "silver_pair_count": len(reference_pairs),
        }
    )

    representative = sample[sample["sampling_component"] == "representative"]
    representative_ids = set(representative["review_id"].astype(str))
    weights = dict(
        zip(
            representative["review_id"].astype(str),
            representative["sampling_weight"].astype(float),
        )
    )
    extraction["representative_population_weighted"] = _weighted_scores(
        {pair for pair in predicted_pairs if pair[0] in representative_ids},
        {pair for pair in reference_pairs if pair[0] in representative_ids},
        weights,
    )

    scope_extraction: dict[str, Any] = {}
    for scope_column in ("analytical_family_id", "competitor_niche_id"):
        if scope_column not in sample.columns:
            continue
        scope_rows: dict[str, Any] = {}
        for scope_value, group in sample.groupby(scope_column, sort=True):
            review_ids = set(group["review_id"].astype(str))
            predicted = {
                pair for pair in predicted_pairs if pair[0] in review_ids
            }
            reference = {
                pair for pair in reference_pairs if pair[0] in review_ids
            }
            scope_rows[str(scope_value)] = {
                **_scores(predicted, reference),
                "review_count": len(review_ids),
                "prediction_pair_count": len(predicted),
                "silver_pair_count": len(reference),
            }
        scope_extraction[scope_column] = scope_rows

    aspect_ids = sorted({pair[1] for pair in predicted_pairs | reference_pairs})
    per_aspect = {}
    for aspect_id in aspect_ids:
        predicted = {pair for pair in predicted_pairs if pair[1] == aspect_id}
        reference = {pair for pair in reference_pairs if pair[1] == aspect_id}
        per_aspect[aspect_id] = {
            **_scores(predicted, reference),
            "prediction_count": len(predicted),
            "silver_count": len(reference),
        }

    predicted_sentiment = {
        (str(row.review_id), str(row.aspect_id)): str(row.aspect_sentiment)
        for row in sentiments.itertuples(index=False)
    }
    silver_sentiment = {
        (str(row.review_id), str(aspect["aspect_id"])): str(aspect["sentiment"])
        for row in silver.itertuples(index=False)
        for aspect in json.loads(row.aspects_json)
    }
    common = sorted(set(predicted_sentiment) & set(silver_sentiment))
    eligible = [
        pair
        for pair in common
        if silver_sentiment[pair] in {"negative", "neutral", "positive"}
    ]
    confusion: dict[str, int] = {}
    for pair in eligible:
        key = f"{silver_sentiment[pair]}->{predicted_sentiment[pair]}"
        confusion[key] = confusion.get(key, 0) + 1
    correct = sum(
        predicted_sentiment[pair] == silver_sentiment[pair] for pair in eligible
    )

    return {
        "reference_kind": "gemini_silver_not_human_gold",
        "interpretation": (
            "Diagnostic agreement with Gemini silver labels; not a final human-ground-truth score."
        ),
        "inputs": {
            "sample_path": _artifact_reference(sample_source),
            "sample_sha256": sha256_file(sample_source),
            "prediction_path": _artifact_reference(prediction_source),
            "prediction_sha256": sha256_file(prediction_source),
            "sentiment_path": _artifact_reference(sentiment_source),
            "sentiment_sha256": sha256_file(sentiment_source),
            "silver_path": _artifact_reference(silver_source),
            "silver_sha256": sha256_file(silver_source),
        },
        "extraction": extraction,
        "scope_extraction": scope_extraction,
        "per_aspect": per_aspect,
        "sentiment_on_common_pairs": {
            "common_pair_count": len(common),
            "eligible_three_class_pair_count": len(eligible),
            "excluded_silver_mixed_or_unclear_count": len(common) - len(eligible),
            "agreement_count": correct,
            "accuracy": correct / len(eligible) if eligible else 0.0,
            "confusion": dict(sorted(confusion.items())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_path", type=Path)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("sentiment_path", type=Path)
    parser.add_argument("silver_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = evaluate_against_silver(
        args.sample_path, args.prediction_path, args.sentiment_path, args.silver_path
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
