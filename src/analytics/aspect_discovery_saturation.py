"""Measure raw-candidate saturation across completed discovery batches."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.aspect_taxonomy import normalize_candidate_label
from src.llm.aspect_discovery import (
    iter_discovery_batches,
    validate_batch_result,
    validate_saved_response_envelope,
)
from src.schemas.aspects import (
    AspectDiscoveryBatchResult,
    load_aspect_discovery_config,
)


def build_discovery_saturation_report(
    config_path: str | Path,
    plan_path: str | Path,
    responses_directory: str | Path,
    sample_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint_batch_count: int = 5,
    low_increment_threshold: int = 5,
) -> dict[str, Any]:
    """Track cumulative supported candidates at fixed review checkpoints."""
    if checkpoint_batch_count < 1 or low_increment_threshold < 0:
        raise ValueError("checkpoint and increment thresholds are invalid")
    config = load_aspect_discovery_config(config_path)
    batches = list(iter_discovery_batches(plan_path))
    responses = Path(responses_directory)
    sample = pd.read_parquet(
        sample_path,
        columns=["review_id", "competitor_niche_id"],
    )
    segment_by_review = dict(
        zip(sample["review_id"].astype(str), sample["competitor_niche_id"], strict=True)
    )

    candidate_reviews: dict[str, set[str]] = defaultdict(set)
    candidate_products: dict[str, set[str]] = defaultdict(set)
    segment_candidate_reviews: dict[tuple[str, str], set[str]] = defaultdict(set)
    segment_candidate_products: dict[tuple[str, str], set[str]] = defaultdict(set)
    completed_batch_count = 0
    completed_review_count = 0
    checkpoints: list[dict[str, Any]] = []
    previous_eligible: set[str] = set()

    for batch in batches:
        response_path = responses / f"{batch.batch_id}.json"
        if not response_path.is_file():
            break
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        validate_saved_response_envelope(payload, batch, config)
        result = AspectDiscoveryBatchResult.model_validate(payload["result"])
        validate_batch_result(batch, result)
        review_by_id = {review.review_id: review for review in batch.reviews}
        for observation in result.observations:
            review = review_by_id[observation.review_id]
            segment_id = str(segment_by_review[review.review_id])
            for mention in observation.mentions:
                candidate_key = normalize_candidate_label(mention.aspect_label)
                if not candidate_key:
                    continue
                candidate_reviews[candidate_key].add(review.review_id)
                candidate_products[candidate_key].add(review.parent_asin)
                segment_key = (segment_id, candidate_key)
                segment_candidate_reviews[segment_key].add(review.review_id)
                segment_candidate_products[segment_key].add(review.parent_asin)
        completed_batch_count += 1
        completed_review_count += len(batch.reviews)
        if completed_batch_count % checkpoint_batch_count:
            continue

        eligible = {
            key
            for key in candidate_reviews
            if len(candidate_reviews[key]) >= config.minimum_candidate_sample_reviews
            and len(candidate_products[key]) >= config.minimum_candidate_products
        }
        new_eligible = eligible - previous_eligible
        segment_coverage = []
        for segment_id in sorted(set(segment_by_review.values())):
            supported = {
                candidate_key
                for (candidate_segment, candidate_key), review_ids in (
                    segment_candidate_reviews.items()
                )
                if candidate_segment == segment_id
                and len(review_ids) >= 3
                and len(
                    segment_candidate_products[(candidate_segment, candidate_key)]
                )
                >= 2
            }
            segment_coverage.append(
                {
                    "competitor_niche_id": str(segment_id),
                    "supported_candidate_count": len(supported),
                }
            )
        checkpoints.append(
            {
                "completed_batch_count": completed_batch_count,
                "completed_review_count": completed_review_count,
                "raw_candidate_count": len(candidate_reviews),
                "eligible_candidate_count": len(eligible),
                "new_eligible_candidate_count": len(new_eligible),
                "new_eligible_candidate_keys": sorted(new_eligible),
                "segment_coverage": segment_coverage,
            }
        )
        previous_eligible = eligible

    recent_increments = [
        checkpoint["new_eligible_candidate_count"] for checkpoint in checkpoints[-2:]
    ]
    preliminary_low_increment = (
        len(recent_increments) == 2
        and all(value <= low_increment_threshold for value in recent_increments)
    )
    report = {
        "discovery_version": config.discovery_version,
        "metric_type": "raw_candidate_saturation_proxy",
        "expected_batch_count": len(batches),
        "completed_batch_count": completed_batch_count,
        "completed_review_count": completed_review_count,
        "checkpoint_batch_count": checkpoint_batch_count,
        "checkpoint_review_count": checkpoint_batch_count * config.batch_size,
        "minimum_candidate_sample_reviews": config.minimum_candidate_sample_reviews,
        "minimum_candidate_products": config.minimum_candidate_products,
        "low_increment_threshold": low_increment_threshold,
        "preliminary_low_increment": preliminary_low_increment,
        "checkpoints": checkpoints,
        "limitation": (
            "This is a raw-label proxy. Final stopping requires normalized "
            "canonical-aspect comparison and adequate coverage of every segment."
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
    parser.add_argument("config_path", type=Path)
    parser.add_argument("plan_path", type=Path)
    parser.add_argument("responses_directory", type=Path)
    parser.add_argument("sample_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--checkpoint-batches", type=int, default=5)
    parser.add_argument("--low-increment-threshold", type=int, default=5)
    args = parser.parse_args()
    report = build_discovery_saturation_report(
        args.config_path,
        args.plan_path,
        args.responses_directory,
        args.sample_path,
        args.output_path,
        checkpoint_batch_count=args.checkpoint_batches,
        low_increment_threshold=args.low_increment_threshold,
    )
    print(
        json.dumps(
            {
                "completed_batch_count": report["completed_batch_count"],
                "completed_review_count": report["completed_review_count"],
                "preliminary_low_increment": report["preliminary_low_increment"],
                "checkpoint_count": len(report["checkpoints"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
