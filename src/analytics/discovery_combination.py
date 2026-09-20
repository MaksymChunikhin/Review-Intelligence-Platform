"""Combine nested aspect-discovery waves with final sampling weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.aspect_taxonomy import normalize_candidate_label
from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file
from src.schemas.aspects import AspectDiscoveryConfig, load_aspect_discovery_config


@dataclass(frozen=True)
class CombinedDiscoveryReport:
    """Record the lineage and coverage of a combined discovery artifact."""

    discovery_version: str
    expanded_sample_path: str
    observations_path: str
    candidates_path: str
    component_count: int
    review_count: int
    review_with_mentions_count: int
    mention_count: int
    distinct_product_count: int
    distinct_candidate_count: int
    normalization_candidate_count: int
    weighted_sample_population: float


def _artifact_reference(path: Path) -> str:
    try:
        root = find_project_root(path.parent)
        return path.resolve().relative_to(root).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)


def _load_sample(path: Path, config: AspectDiscoveryConfig) -> pd.DataFrame:
    columns = [
        "review_id",
        "dataset_version",
        "niche_version",
        "parent_asin",
        "rating",
        "review_year",
        "sampling_weight",
    ]
    sample = pd.read_parquet(path, columns=columns)
    if sample.empty or sample["review_id"].duplicated().any():
        raise ValueError(f"sample must contain unique reviews: {path}")
    identities = {
        "dataset_version": config.dataset_version,
        "niche_version": config.niche_version,
    }
    for column, expected in identities.items():
        actual = set(sample[column].dropna().astype(str))
        if actual != {expected}:
            raise ValueError(
                f"sample identity mismatch for {column}: "
                f"expected={expected!r}, actual={sorted(actual)!r}"
            )
    return sample


def _candidate_payload(
    config: AspectDiscoveryConfig,
    observations: pd.DataFrame,
    sample: pd.DataFrame,
) -> dict[str, Any]:
    weighted_population = float(sample["sampling_weight"].sum())
    candidates: list[dict[str, Any]] = []
    for candidate_key, group in observations.groupby("candidate_key", sort=True):
        review_support = group.drop_duplicates("review_id")
        raw_label_counts = group["aspect_label"].value_counts()
        topic_counts = group["topic_type"].value_counts().sort_index()
        weighted_support = float(review_support["sampling_weight"].sum())
        candidates.append(
            {
                "candidate_key": candidate_key,
                "display_label": str(raw_label_counts.index[0]),
                "source_labels": sorted(
                    {str(value) for value in group["aspect_label"]},
                    key=str.casefold,
                ),
                "mention_count": int(len(group)),
                "sample_review_count": int(review_support["review_id"].nunique()),
                "sample_review_share": float(
                    review_support["review_id"].nunique() / len(sample)
                ),
                "product_count": int(group["parent_asin"].nunique()),
                "weighted_review_support": weighted_support,
                "weighted_population_share": float(
                    weighted_support / weighted_population
                ),
                "topic_type_counts": {
                    str(name): int(count) for name, count in topic_counts.items()
                },
            }
        )
    candidates.sort(
        key=lambda item: (
            -item["weighted_review_support"],
            -item["sample_review_count"],
            item["candidate_key"],
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["support_rank"] = rank
        candidate["eligible_for_normalization"] = (
            candidate["sample_review_count"]
            >= config.minimum_candidate_sample_reviews
            and candidate["product_count"] >= config.minimum_candidate_products
        )
    eligible = [
        candidate
        for candidate in candidates
        if candidate["eligible_for_normalization"]
    ][: config.maximum_normalization_candidates]
    eligible_keys = {candidate["candidate_key"] for candidate in eligible}
    for candidate in candidates:
        candidate["selected_for_normalization"] = (
            candidate["candidate_key"] in eligible_keys
        )
    return {
        "discovery_version": config.discovery_version,
        "dataset_version": config.dataset_version,
        "niche_version": config.niche_version,
        "weighted_sample_population": weighted_population,
        "candidate_count": len(candidates),
        "normalization_candidate_count": len(eligible),
        "candidates": candidates,
    }


def combine_discovery_components(
    config: AspectDiscoveryConfig,
    expanded_sample_path: str | Path,
    components: list[tuple[str | Path, str | Path]],
    observations_path: str | Path,
    candidates_path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> CombinedDiscoveryReport:
    """Combine disjoint discovery samples and apply expanded-sample weights."""
    if len(components) < 2:
        raise ValueError("at least two discovery components are required")
    expanded_path = Path(expanded_sample_path)
    expanded = _load_sample(expanded_path, config)
    expanded_ids = set(expanded["review_id"].astype(str))
    final_by_review = expanded.set_index("review_id")

    component_ids: set[str] = set()
    observation_frames: list[pd.DataFrame] = []
    component_lineage: list[dict[str, Any]] = []
    required_observation_columns = {
        "review_id",
        "aspect_label",
        "candidate_key",
        "topic_label",
        "topic_type",
        "customer_phrase",
    }
    for sample_value, observations_value in components:
        sample_path = Path(sample_value)
        component_sample = _load_sample(sample_path, config)
        ids = set(component_sample["review_id"].astype(str))
        overlap = component_ids & ids
        if overlap:
            raise ValueError(
                f"component samples overlap on {len(overlap)} review IDs"
            )
        component_ids.update(ids)

        component_observations_path = Path(observations_value)
        observations = pd.read_parquet(component_observations_path)
        missing = required_observation_columns - set(observations.columns)
        if missing:
            raise ValueError(
                f"component observations are missing columns: {sorted(missing)}"
            )
        observation_ids = set(observations["review_id"].astype(str))
        if not observation_ids <= ids:
            raise ValueError("component observations contain unplanned reviews")
        normalized = observations["aspect_label"].map(normalize_candidate_label)
        if not normalized.equals(observations["candidate_key"].astype(str)):
            raise ValueError("component candidate keys do not match aspect labels")
        observation_frames.append(observations)
        component_lineage.append(
            {
                "sample_path": _artifact_reference(sample_path),
                "sample_sha256": sha256_file(sample_path),
                "review_count": len(component_sample),
                "observations_path": _artifact_reference(
                    component_observations_path
                ),
                "observations_sha256": sha256_file(
                    component_observations_path
                ),
                "mention_count": len(observations),
            }
        )

    if component_ids != expanded_ids:
        raise ValueError(
            "component samples must partition the expanded sample exactly: "
            f"missing={len(expanded_ids - component_ids)}, "
            f"unexpected={len(component_ids - expanded_ids)}"
        )

    combined = pd.concat(observation_frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=[
            "review_id",
            "candidate_key",
            "topic_label",
            "topic_type",
            "customer_phrase",
        ]
    )
    if combined.empty:
        raise ValueError("combined observations contain no mentions")
    metadata = final_by_review.loc[combined["review_id"]]
    combined["discovery_version"] = config.discovery_version
    combined["parent_asin"] = metadata["parent_asin"].to_numpy()
    combined["rating"] = metadata["rating"].to_numpy()
    combined["review_year"] = metadata["review_year"].to_numpy()
    combined["sampling_weight"] = metadata["sampling_weight"].to_numpy()
    output_columns = [
        "discovery_version",
        "batch_id",
        "review_id",
        "parent_asin",
        "rating",
        "review_year",
        "sampling_weight",
        "aspect_label",
        "candidate_key",
        "topic_label",
        "topic_type",
        "customer_phrase",
    ]
    combined = combined[output_columns]

    observations_destination = Path(observations_path)
    observations_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = observations_destination.with_suffix(".tmp.parquet")
    combined.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(observations_destination)

    candidates_payload = _candidate_payload(config, combined, expanded)
    candidates_payload["lineage"] = {
        "expanded_sample_path": _artifact_reference(expanded_path),
        "expanded_sample_sha256": sha256_file(expanded_path),
        "components": component_lineage,
    }
    candidates_destination = Path(candidates_path)
    _write_json(candidates_payload, candidates_destination)

    report = CombinedDiscoveryReport(
        discovery_version=config.discovery_version,
        expanded_sample_path=_artifact_reference(expanded_path),
        observations_path=_artifact_reference(observations_destination),
        candidates_path=_artifact_reference(candidates_destination),
        component_count=len(components),
        review_count=len(expanded),
        review_with_mentions_count=int(combined["review_id"].nunique()),
        mention_count=len(combined),
        distinct_product_count=int(expanded["parent_asin"].nunique()),
        distinct_candidate_count=candidates_payload["candidate_count"],
        normalization_candidate_count=candidates_payload[
            "normalization_candidate_count"
        ],
        weighted_sample_population=candidates_payload[
            "weighted_sample_population"
        ],
    )
    if report_path is not None:
        report_payload = asdict(report)
        report_payload["lineage"] = candidates_payload["lineage"]
        _write_json(report_payload, Path(report_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("expanded_sample_path", type=Path)
    parser.add_argument("observations_path", type=Path)
    parser.add_argument("candidates_path", type=Path)
    parser.add_argument(
        "--component",
        action="append",
        nargs=2,
        metavar=("SAMPLE_PATH", "OBSERVATIONS_PATH"),
        required=True,
    )
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    report = combine_discovery_components(
        load_aspect_discovery_config(args.config_path),
        args.expanded_sample_path,
        [(Path(sample), Path(observations)) for sample, observations in args.component],
        args.observations_path,
        args.candidates_path,
        report_path=args.report_path,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
