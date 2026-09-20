"""Run aspect inference over local parquet partitions in parallel."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file
from src.ml.aspect_inference import (
    OUTPUT_SCHEMA,
    AspectInferenceReport,
    predict_aspects,
)
from src.ml.aspect_multilabel import AspectMultilabelConfig


def _reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _run_partition(
    config_path: str,
    taxonomy_path: str,
    input_path: str,
    model_directory: str,
    output_path: str,
    batch_size: int,
) -> AspectInferenceReport:
    config = AspectMultilabelConfig.model_validate_json(
        Path(config_path).read_text(encoding="utf-8")
    )
    return predict_aspects(
        config,
        taxonomy_path,
        input_path,
        model_directory,
        output_path,
        batch_size=batch_size,
    )


def predict_aspects_parallel(
    config_path: str | Path,
    taxonomy_path: str | Path,
    input_path: str | Path,
    model_directory: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    workers: int = 4,
    batch_size: int = 5_000,
) -> AspectInferenceReport:
    """Partition reviews, infer concurrently, and reconcile one final output."""
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch size must be positive")
    config_source = Path(config_path)
    taxonomy_source = Path(taxonomy_path)
    input_source = Path(input_path)
    model_source = Path(model_directory)
    destination = Path(output_path)
    config = AspectMultilabelConfig.model_validate_json(
        config_source.read_text(encoding="utf-8")
    )
    reviews = pd.read_parquet(input_source, columns=["review_id", "review_text"])
    if reviews.empty or reviews["review_id"].duplicated().any():
        raise ValueError("parallel inference requires unique non-empty reviews")
    worker_count = min(workers, len(reviews))

    with tempfile.TemporaryDirectory(
        prefix="aspect-inference-", dir="/tmp"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        input_parts: list[Path] = []
        output_parts: list[Path] = []
        index_parts = np.array_split(np.arange(len(reviews)), worker_count)
        for index, row_indexes in enumerate(index_parts, start=1):
            partition = reviews.iloc[row_indexes]
            input_part = temporary_root / f"reviews_{index:02d}.parquet"
            output_part = temporary_root / f"predictions_{index:02d}.parquet"
            partition.to_parquet(input_part, index=False, compression="zstd")
            input_parts.append(input_part)
            output_parts.append(output_part)
        review_count = len(reviews)
        del reviews
        gc.collect()

        arguments = [
            (
                str(config_source),
                str(taxonomy_source),
                str(input_part),
                str(model_source),
                str(output_part),
                batch_size,
            )
            for input_part, output_part in zip(
                input_parts, output_parts, strict=True
            )
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            reports = list(executor.map(_run_partition_star, arguments))

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = destination.with_suffix(".tmp.parquet")
        writer = pq.ParquetWriter(
            temporary_output, OUTPUT_SCHEMA, compression="zstd"
        )
        try:
            for output_part in output_parts:
                writer.write_table(pq.read_table(output_part))
        finally:
            writer.close()
        temporary_output.replace(destination)

        aspect_ids: set[str] = set()
        for output_part in output_parts:
            aspect_ids.update(
                pq.read_table(output_part, columns=["aspect_id"])
                .column("aspect_id")
                .to_pylist()
            )

    source_counts = {
        key: sum(report.source_counts[key] for report in reports)
        for key in ("exact_alias_only", "char_tfidf_only", "both")
    }
    matched = sum(report.matched_review_count for report in reports)
    review_aspects = sum(report.review_aspect_count for report in reports)
    report = AspectInferenceReport(
        model_version=config.model_version,
        taxonomy_version=config.taxonomy_version,
        method=config.method,
        threshold=config.threshold,
        input_path=_reference(input_source),
        input_sha256=sha256_file(input_source),
        input_review_count=review_count,
        matched_review_count=matched,
        no_match_review_count=review_count - matched,
        review_aspect_count=review_aspects,
        distinct_aspect_count=len(aspect_ids),
        source_counts=source_counts,
        batch_size=batch_size,
        output_path=_reference(destination),
        output_sha256=sha256_file(destination),
        human_gold_accepted=False,
        warning=(
            "Population output from a Gemini-silver-trained candidate. It is "
            "suitable for diagnostic analytics, not final accuracy claims."
        ),
    )
    if report_path is not None:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        report_destination.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _run_partition_star(
    arguments: tuple[str, str, str, str, str, int],
) -> AspectInferenceReport:
    return _run_partition(*arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("taxonomy_path", type=Path)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=5_000)
    args = parser.parse_args()
    report = predict_aspects_parallel(
        args.config_path,
        args.taxonomy_path,
        args.input_path,
        args.model_directory,
        args.output_path,
        report_path=args.report_path,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
