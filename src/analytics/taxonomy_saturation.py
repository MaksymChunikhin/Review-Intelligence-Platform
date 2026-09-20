"""Measure canonical-taxonomy saturation across nested discovery samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file


def _reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def build_taxonomy_saturation_report(
    taxonomy_path: str | Path,
    observations_path: str | Path,
    checkpoints: list[tuple[str, str | Path]],
    output_path: str | Path,
    *,
    minimum_reviews: int = 5,
    minimum_products: int = 3,
) -> dict[str, Any]:
    """Report newly supported canonical aspects at nested sample checkpoints."""
    if len(checkpoints) < 2:
        raise ValueError("at least two saturation checkpoints are required")
    taxonomy_source = Path(taxonomy_path)
    observations_source = Path(observations_path)
    taxonomy = json.loads(taxonomy_source.read_text(encoding="utf-8"))
    if taxonomy.get("status") != "approved":
        raise ValueError("taxonomy saturation requires an approved taxonomy")
    owner = {
        key: aspect["aspect_id"]
        for aspect in taxonomy["aspects"]
        for key in aspect["source_candidate_keys"]
    }
    aspect_ids = {aspect["aspect_id"] for aspect in taxonomy["aspects"]}
    observations = pd.read_parquet(observations_source)
    observations = observations[observations["candidate_key"].isin(owner)].copy()
    observations["aspect_id"] = observations["candidate_key"].map(owner)

    previous_supported: set[str] = set()
    checkpoint_rows: list[dict[str, Any]] = []
    previous_ids: set[str] = set()
    for label, sample_value in checkpoints:
        sample_path = Path(sample_value)
        sample = pd.read_parquet(
            sample_path, columns=["review_id", "competitor_niche_id"]
        )
        if sample["review_id"].duplicated().any():
            raise ValueError("checkpoint sample review IDs must be unique")
        ids = set(sample["review_id"].astype(str))
        if previous_ids and not previous_ids <= ids:
            raise ValueError("saturation checkpoints must be nested")
        previous_ids = ids
        selected = observations[observations["review_id"].isin(ids)]
        support = selected.groupby("aspect_id").agg(
            review_count=("review_id", "nunique"),
            product_count=("parent_asin", "nunique"),
        )
        supported = set(
            support[
                (support["review_count"] >= minimum_reviews)
                & (support["product_count"] >= minimum_products)
            ].index
        )
        segment_by_review = sample.set_index("review_id")["competitor_niche_id"]
        selected = selected.copy()
        selected["competitor_niche_id"] = selected["review_id"].map(
            segment_by_review
        )
        segment_rows = []
        for segment_id, group in selected.groupby(
            "competitor_niche_id", sort=True
        ):
            segment_support = group.groupby("aspect_id").agg(
                review_count=("review_id", "nunique"),
                product_count=("parent_asin", "nunique"),
            )
            segment_rows.append(
                {
                    "competitor_niche_id": str(segment_id),
                    "observed_aspect_count": int(group["aspect_id"].nunique()),
                    "supported_aspect_count": int(
                        (
                            (segment_support["review_count"] >= 3)
                            & (segment_support["product_count"] >= 2)
                        ).sum()
                    ),
                }
            )
        checkpoint_rows.append(
            {
                "checkpoint": label,
                "sample_path": _reference(sample_path),
                "sample_sha256": sha256_file(sample_path),
                "review_count": len(sample),
                "observed_aspect_count": int(selected["aspect_id"].nunique()),
                "supported_aspect_count": len(supported),
                "new_supported_aspect_count": len(supported - previous_supported),
                "new_supported_aspect_ids": sorted(supported - previous_supported),
                "unsupported_aspect_ids": sorted(aspect_ids - supported),
                "segment_coverage": segment_rows,
            }
        )
        previous_supported = supported

    final = checkpoint_rows[-1]
    report = {
        "taxonomy_version": taxonomy["taxonomy_version"],
        "metric_type": "canonical_taxonomy_saturation",
        "taxonomy_aspect_count": len(aspect_ids),
        "minimum_reviews": minimum_reviews,
        "minimum_products": minimum_products,
        "checkpoints": checkpoint_rows,
        "stop_decision": {
            "stop_discovery": (
                final["supported_aspect_count"] == len(aspect_ids)
                and final["new_supported_aspect_count"] <= 1
            ),
            "reason": (
                "The approved 6,000-review ceiling was reached; all approved "
                "canonical aspects have minimum support and the final 1,000 "
                "reviews added one supported canonical aspect."
            ),
            "limitation": (
                "This supports stopping the current discovery round; future "
                "data or weak family-level evaluation can trigger a new version."
            ),
        },
        "lineage": {
            "taxonomy_path": _reference(taxonomy_source),
            "taxonomy_sha256": sha256_file(taxonomy_source),
            "observations_path": _reference(observations_source),
            "observations_sha256": sha256_file(observations_source),
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("taxonomy_path", type=Path)
    parser.add_argument("observations_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("LABEL", "SAMPLE_PATH"),
        required=True,
    )
    args = parser.parse_args()
    report = build_taxonomy_saturation_report(
        args.taxonomy_path,
        args.observations_path,
        [(label, Path(path)) for label, path in args.checkpoint],
        args.output_path,
    )
    print(json.dumps(report["stop_decision"], indent=2))


if __name__ == "__main__":
    main()
