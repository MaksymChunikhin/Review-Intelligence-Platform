"""Materialize taxonomy labels from reviewed aspect-discovery observations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file


@dataclass(frozen=True)
class DiscoveryTrainingLabelReport:
    label_version: str
    taxonomy_version: str
    review_count: int
    review_with_aspects_count: int
    no_supported_aspect_count: int
    review_aspect_count: int
    distinct_aspect_count: int
    sample_path: str
    sample_sha256: str
    observations_path: str
    observations_sha256: str
    taxonomy_path: str
    taxonomy_sha256: str
    output_path: str
    output_sha256: str


def _reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def materialize_discovery_training_labels(
    sample_path: str | Path,
    observations_path: str | Path,
    taxonomy_path: str | Path,
    output_path: str | Path,
    *,
    label_version: str,
    report_path: str | Path | None = None,
) -> DiscoveryTrainingLabelReport:
    """Map reviewed candidate keys to canonical multi-label training rows."""
    sample_source = Path(sample_path)
    observations_source = Path(observations_path)
    taxonomy_source = Path(taxonomy_path)
    sample = pd.read_parquet(sample_source, columns=["review_id"])
    observations = pd.read_parquet(
        observations_source, columns=["review_id", "candidate_key"]
    )
    taxonomy = json.loads(taxonomy_source.read_text(encoding="utf-8"))
    if taxonomy.get("status") != "approved":
        raise ValueError("training labels require an approved taxonomy")
    if sample.empty or sample["review_id"].duplicated().any():
        raise ValueError("training sample must contain unique reviews")
    sample_ids = set(sample["review_id"].astype(str))
    if not set(observations["review_id"].astype(str)) <= sample_ids:
        raise ValueError("observations contain reviews outside the training sample")
    owner = {
        key: aspect["aspect_id"]
        for aspect in taxonomy["aspects"]
        for key in aspect["source_candidate_keys"]
    }
    mapped = observations[observations["candidate_key"].isin(owner)].copy()
    mapped["aspect_id"] = mapped["candidate_key"].map(owner)
    aspects_by_review = (
        mapped.groupby("review_id")["aspect_id"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    rows = []
    for review_id in sample["review_id"].astype(str):
        aspect_ids = aspects_by_review.get(review_id, [])
        rows.append(
            {
                "silver_version": label_version,
                "taxonomy_version": taxonomy["taxonomy_version"],
                "review_id": review_id,
                "aspect_ids_json": json.dumps(aspect_ids),
                "no_supported_aspect": not aspect_ids,
                "aspect_count": len(aspect_ids),
                "label_source": "reviewed_discovery_mapping_v1",
            }
        )
    frame = pd.DataFrame(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(destination)
    report = DiscoveryTrainingLabelReport(
        label_version=label_version,
        taxonomy_version=taxonomy["taxonomy_version"],
        review_count=len(frame),
        review_with_aspects_count=int((frame["aspect_count"] > 0).sum()),
        no_supported_aspect_count=int(frame["no_supported_aspect"].sum()),
        review_aspect_count=int(frame["aspect_count"].sum()),
        distinct_aspect_count=len(
            {aspect for values in aspects_by_review.values() for aspect in values}
        ),
        sample_path=_reference(sample_source),
        sample_sha256=sha256_file(sample_source),
        observations_path=_reference(observations_source),
        observations_sha256=sha256_file(observations_source),
        taxonomy_path=_reference(taxonomy_source),
        taxonomy_sha256=sha256_file(taxonomy_source),
        output_path=_reference(destination),
        output_sha256=sha256_file(destination),
    )
    if report_path is not None:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = report_destination.with_suffix(".tmp.json")
        temporary_report.write_text(
            json.dumps(asdict(report), indent=2), encoding="utf-8"
        )
        temporary_report.replace(report_destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_path", type=Path)
    parser.add_argument("observations_path", type=Path)
    parser.add_argument("taxonomy_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--label-version", required=True)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    report = materialize_discovery_training_labels(
        args.sample_path,
        args.observations_path,
        args.taxonomy_path,
        args.output_path,
        label_version=args.label_version,
        report_path=args.report_path,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
