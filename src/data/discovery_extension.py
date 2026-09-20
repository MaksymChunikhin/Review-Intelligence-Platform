"""Materialize only the new rows in a nested discovery-sample extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.ingestion.dataset_manifest import sha256_file


def build_discovery_extension(
    base_sample_path: str | Path,
    expanded_sample_path: str | Path,
    output_path: str | Path,
    *,
    extension_sampling_version: str,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    """Extract nested new rows and give the extension a valid census design."""
    base_source = Path(base_sample_path)
    expanded_source = Path(expanded_sample_path)
    base_ids = set(
        pd.read_parquet(base_source, columns=["review_id"])["review_id"]
    )
    expanded = pd.read_parquet(expanded_source)
    expanded_ids = set(expanded["review_id"])
    if not base_ids <= expanded_ids:
        raise ValueError("base discovery sample is not nested in expanded sample")
    extension = expanded[~expanded["review_id"].isin(base_ids)].copy()
    if extension.empty or not extension["review_id"].is_unique:
        raise ValueError("discovery extension must be non-empty and unique")

    extension["sampling_version"] = extension_sampling_version
    extension_counts = extension.groupby("stratum_id")["review_id"].transform(
        "size"
    )
    extension["population_stratum_count"] = extension_counts
    extension["sample_stratum_count"] = extension_counts
    extension["sampling_weight"] = 1.0
    extension["sampling_weight_semantics"] = (
        "inverse_review_inclusion_probability"
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    extension.sort_values(["stratum_id", "review_id"]).to_parquet(
        temporary, index=False, compression="zstd"
    )
    if pq.ParquetFile(temporary).metadata.num_rows != len(extension):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("written discovery extension count is invalid")
    output_sha256 = sha256_file(temporary)
    temporary.replace(destination)
    report = {
        "extension_sampling_version": extension_sampling_version,
        "base_sample_path": str(base_source),
        "base_sample_sha256": sha256_file(base_source),
        "base_review_count": len(base_ids),
        "expanded_sample_path": str(expanded_source),
        "expanded_sample_sha256": sha256_file(expanded_source),
        "expanded_review_count": len(expanded),
        "nested_base_review_count": len(base_ids & expanded_ids),
        "extension_review_count": len(extension),
        "extension_product_count": int(extension["parent_asin"].nunique()),
        "output_path": str(destination),
        "output_sha256": output_sha256,
        "weight_note": (
            "Extension request weights are a census of extension rows. Final "
            "candidate aggregation must replace them with weights from the "
            "complete expanded sample."
        ),
    }
    if report_path is not None:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = report_destination.with_suffix(".tmp.json")
        temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary_report.replace(report_destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_sample_path", type=Path)
    parser.add_argument("expanded_sample_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--extension-sampling-version", required=True)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    report = build_discovery_extension(
        args.base_sample_path,
        args.expanded_sample_path,
        args.output_path,
        extension_sampling_version=args.extension_sampling_version,
        report_path=args.report_path,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
