"""Combine independent representative and targeted section evaluation rows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file


@dataclass(frozen=True)
class SectionEvaluationReport:
    evaluation_version: str
    evaluation_schema_version: str
    representative_path: str
    representative_sha256: str
    targeted_path: str
    targeted_sha256: str
    discovery_path: str
    discovery_sha256: str
    output_path: str
    output_sha256: str
    representative_review_count: int
    targeted_review_count: int
    total_review_count: int
    distinct_product_count: int
    discovery_overlap_count: int
    component_overlap_count: int
    segment_distribution: list[dict[str, object]]


def _artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def combine_section_evaluation_samples(
    representative_path: str | Path,
    targeted_path: str | Path,
    discovery_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_version: str,
    report_path: str | Path | None = None,
) -> SectionEvaluationReport:
    """Create a disjoint pre-taxonomy evaluation artifact."""
    representative_source = Path(representative_path)
    targeted_source = Path(targeted_path)
    discovery_source = Path(discovery_path)
    for source in (
        representative_source,
        targeted_source,
        discovery_source,
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
    representative = pd.read_parquet(representative_source)
    targeted = pd.read_parquet(targeted_source)
    discovery_ids = set(
        pd.read_parquet(discovery_source, columns=["review_id"])["review_id"]
    )
    representative_ids = set(representative["review_id"])
    targeted_ids = set(targeted["review_id"])
    component_overlap = representative_ids & targeted_ids
    discovery_overlap = (representative_ids | targeted_ids) & discovery_ids
    if component_overlap:
        raise ValueError("representative and targeted samples overlap")
    if discovery_overlap:
        raise ValueError("evaluation and discovery samples overlap")

    representative = representative.copy()
    targeted = targeted.copy()
    representative["sampling_component"] = "representative"
    representative["evaluation_weight"] = representative["sampling_weight"]
    representative["evaluation_weight_semantics"] = (
        "inverse_review_inclusion_probability"
    )
    targeted["sampling_component"] = "targeted"
    targeted["evaluation_weight"] = float("nan")
    targeted["evaluation_weight_semantics"] = (
        "not_applicable_targeted_diagnostic"
    )
    sample = pd.concat([representative, targeted], ignore_index=True)
    sample["evaluation_version"] = evaluation_version
    sample["evaluation_schema_version"] = "section_aspect_evaluation_v1"
    sample = sample.sort_values(
        ["sampling_component", "competitor_niche_id", "stratum_id", "review_id"]
    ).reset_index(drop=True)
    sample["evaluation_row_id"] = [
        f"{evaluation_version}:row:{index:04d}"
        for index in range(1, len(sample) + 1)
    ]
    if not sample["review_id"].is_unique:
        raise RuntimeError("combined evaluation review IDs must be unique")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    sample.to_parquet(temporary, index=False, compression="zstd")
    if pq.ParquetFile(temporary).metadata.num_rows != len(sample):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("written evaluation row count is invalid")
    output_sha256 = sha256_file(temporary)
    temporary.replace(destination)
    segment_distribution = (
        sample.groupby(
            ["sampling_component", "competitor_niche_id"], sort=True
        )
        .agg(
            review_count=("review_id", "size"),
            product_count=("parent_asin", "nunique"),
        )
        .reset_index()
        .to_dict("records")
    )
    report = SectionEvaluationReport(
        evaluation_version=evaluation_version,
        evaluation_schema_version="section_aspect_evaluation_v1",
        representative_path=_artifact_reference(representative_source),
        representative_sha256=sha256_file(representative_source),
        targeted_path=_artifact_reference(targeted_source),
        targeted_sha256=sha256_file(targeted_source),
        discovery_path=_artifact_reference(discovery_source),
        discovery_sha256=sha256_file(discovery_source),
        output_path=_artifact_reference(destination),
        output_sha256=output_sha256,
        representative_review_count=len(representative),
        targeted_review_count=len(targeted),
        total_review_count=len(sample),
        distinct_product_count=int(sample["parent_asin"].nunique()),
        discovery_overlap_count=0,
        component_overlap_count=0,
        segment_distribution=segment_distribution,
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
    parser.add_argument("representative_path", type=Path)
    parser.add_argument("targeted_path", type=Path)
    parser.add_argument("discovery_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--evaluation-version", required=True)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    report = combine_section_evaluation_samples(
        args.representative_path,
        args.targeted_path,
        args.discovery_path,
        args.output_path,
        evaluation_version=args.evaluation_version,
        report_path=args.report_path,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
