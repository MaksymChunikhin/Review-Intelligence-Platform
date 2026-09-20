"""Summarize cross-section coverage of a frozen aspect extractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ingestion.dataset_manifest import sha256_file


def build_section_baseline_coverage_report(
    evaluation_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Report prediction coverage without treating it as accuracy."""
    evaluation_source = Path(evaluation_path)
    predictions_source = Path(predictions_path)
    evaluation = pd.read_parquet(
        evaluation_source,
        columns=[
            "review_id",
            "sampling_component",
            "analytical_family_id",
            "competitor_niche_id",
        ],
    )
    predictions = pd.read_parquet(
        predictions_source, columns=["review_id", "aspect_id"]
    )
    counts = (
        predictions.groupby("review_id")
        .agg(predicted_aspect_count=("aspect_id", "size"))
        .reset_index()
    )
    joined = evaluation.merge(counts, on="review_id", how="left")
    joined["predicted_aspect_count"] = joined["predicted_aspect_count"].fillna(0)
    joined["matched"] = joined["predicted_aspect_count"] > 0

    def summarize(group: pd.DataFrame) -> dict[str, object]:
        review_count = len(group)
        matched_count = int(group["matched"].sum())
        return {
            "review_count": review_count,
            "matched_review_count": matched_count,
            "no_match_review_count": review_count - matched_count,
            "matched_review_share": matched_count / review_count,
            "mean_predicted_aspects_per_review": float(
                group["predicted_aspect_count"].mean()
            ),
        }

    segment_rows = []
    for keys, group in joined.groupby(
        ["sampling_component", "competitor_niche_id"], sort=True
    ):
        row = {
            "sampling_component": str(keys[0]),
            "competitor_niche_id": str(keys[1]),
        }
        row.update(summarize(group))
        segment_rows.append(row)
    family_rows = []
    for family_id, group in joined.groupby("analytical_family_id", sort=True):
        row = {"analytical_family_id": str(family_id)}
        row.update(summarize(group))
        family_rows.append(row)

    report = {
        "report_type": "frozen_extractor_coverage_not_accuracy",
        "evaluation_path": str(evaluation_source),
        "evaluation_sha256": sha256_file(evaluation_source),
        "predictions_path": str(predictions_source),
        "predictions_sha256": sha256_file(predictions_source),
        "overall": summarize(joined),
        "families": family_rows,
        "segments": segment_rows,
        "limitation": (
            "Coverage only measures whether the old Conditioner extractor "
            "emitted labels. Precision, recall, and missing new aspects require "
            "the independent evaluation reference."
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_path", type=Path)
    parser.add_argument("predictions_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = build_section_baseline_coverage_report(
        args.evaluation_path, args.predictions_path, args.output_path
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
