"""Build a canonical, versioned Amazon review dataset."""

import argparse
import bz2
import gzip
import json
import lzma
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import load_dataset_manifest


SOURCE_COLUMNS = {
    "rating",
    "title",
    "text",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
}

NULLABLE_OR_REJECTABLE_SOURCE_COLUMNS = {
    "user_id",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
}

CANONICAL_REVIEW_SCHEMA = pa.schema(
    [
        pa.field("review_id", pa.string(), nullable=False),
        pa.field("dataset_version", pa.string(), nullable=False),
        pa.field("dataset_category", pa.string(), nullable=False),
        pa.field("source_record_index", pa.int64(), nullable=False),
        pa.field("asin", pa.string(), nullable=False),
        pa.field("parent_asin", pa.string(), nullable=False),
        pa.field("rating", pa.float32(), nullable=False),
        pa.field("review_title", pa.string(), nullable=False),
        pa.field("review_body", pa.string(), nullable=False),
        pa.field("review_text", pa.string(), nullable=False),
        pa.field("review_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("source_timestamp_ms", pa.int64(), nullable=False),
        pa.field("user_id", pa.string()),
        pa.field("verified_purchase", pa.bool_()),
        pa.field("helpful_vote", pa.int32()),
    ]
)


@dataclass(frozen=True)
class ReviewPreprocessingConfig:
    """Configure one deterministic review preprocessing run."""

    dataset_version: str
    dataset_category: str
    start_date: date
    end_date: date
    chunk_size: int = 100_000
    compression: str = "zstd"
    deduplicate: bool = True
    deduplication_memory_limit: str = "4GB"

    def __post_init__(self) -> None:
        """Validate values that control artifact identity and execution."""
        if not self.dataset_version:
            raise ValueError("dataset_version must not be empty")
        if not self.dataset_category:
            raise ValueError("dataset_category must not be empty")
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not self.deduplication_memory_limit:
            raise ValueError("deduplication_memory_limit must not be empty")


@dataclass
class ReviewPreprocessingReport:
    """Collect reconciled counts and quality signals for one build."""

    dataset_version: str
    dataset_category: str
    input_path: str
    output_path: str
    start_date: str
    end_date: str
    input_rows: int = 0
    valid_rows_before_deduplication: int = 0
    output_rows: int = 0
    duplicate_rows_removed: int = 0
    invalid_optional_values: dict[str, int] = field(default_factory=dict)
    dropped_by_primary_reason: dict[str, int] = field(default_factory=dict)
    min_review_timestamp: str | None = None
    max_review_timestamp: str | None = None


@dataclass(frozen=True)
class CanonicalReviewArtifactValidation:
    """Describe physical-schema and required-null checks for one artifact."""

    row_count: int
    column_names_match: bool
    column_types_match: bool
    column_nullability_match: bool
    required_null_counts: dict[str, int | None]

    @property
    def is_valid(self) -> bool:
        """Return whether the artifact satisfies the canonical review contract."""
        return (
            self.column_names_match
            and self.column_types_match
            and self.column_nullability_match
            and all(count == 0 for count in self.required_null_counts.values())
        )


def clean_text(series: pd.Series) -> pd.Series:
    """Normalize whitespace and remove HTML tags while preserving wording."""
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def make_review_ids(
    dataset_version: str, source_record_indices: pd.Series
) -> pd.Series:
    """Create stable IDs tied to an immutable input file and record position."""
    width = max(9, len(str(int(source_record_indices.max()))))
    return source_record_indices.map(
        lambda index: f"{dataset_version}:review:{int(index):0{width}d}"
    )


def _coerce_bounded_integer(
    series: pd.Series,
    *,
    dtype: str,
    minimum: int,
    maximum: int,
) -> tuple[pd.Series, pd.Series]:
    """Coerce only finite, integral values inside the target integer range."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = (
        numeric.notna()
        & numeric.ge(minimum)
        & numeric.le(maximum)
        & numeric.eq(numeric.round())
    )
    return numeric.where(valid).astype(dtype), valid


def _primary_drop_reasons(
    *,
    rating: pd.Series,
    timestamp: pd.Series,
    in_date_window: pd.Series,
    asin: pd.Series,
    parent_asin: pd.Series,
    review_text: pd.Series,
) -> pd.Series:
    """Assign one deterministic reason to every rejected source row."""
    reasons = pd.Series(pd.NA, index=rating.index, dtype="string")
    checks = [
        (rating.isin([1, 2, 3, 4, 5]), "invalid_rating"),
        (timestamp.notna(), "invalid_timestamp"),
        (in_date_window, "outside_date_window"),
        (asin.ne(""), "missing_asin"),
        (parent_asin.ne(""), "missing_parent_asin"),
        (review_text.ne(""), "empty_review_text"),
    ]
    for valid, reason in checks:
        reasons = reasons.mask(reasons.isna() & ~valid, reason)
    return reasons


def transform_review_chunk(
    chunk: pd.DataFrame,
    *,
    config: ReviewPreprocessingConfig,
    source_record_offset: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and transform one source chunk into the canonical schema."""
    missing = SOURCE_COLUMNS - set(chunk.columns)
    missing_required = missing - NULLABLE_OR_REJECTABLE_SOURCE_COLUMNS
    if missing_required:
        raise ValueError(
            f"Missing source review columns: {sorted(missing_required)}"
        )
    frame = chunk.copy()
    for column in missing:
        frame[column] = pd.NA
    source_indices = pd.Series(
        range(source_record_offset, source_record_offset + len(frame)),
        index=frame.index,
        dtype="int64",
    )
    rating = pd.to_numeric(frame["rating"], errors="coerce")
    # Constrain milliseconds to pandas' nanosecond timestamp range.  Besides
    # keeping Arrow conversion safe, this makes fractional and absurdly large
    # source values deterministic data-quality failures rather than exceptions.
    timestamp_ms, _ = _coerce_bounded_integer(
        frame["timestamp"],
        dtype="Int64",
        minimum=-9_223_372_036_855,
        maximum=9_223_372_036_854,
    )
    review_timestamp = pd.to_datetime(
        timestamp_ms, unit="ms", utc=True, errors="coerce"
    )
    window_start = pd.Timestamp(config.start_date, tz="UTC")
    window_end = pd.Timestamp(config.end_date, tz="UTC") + pd.Timedelta(days=1)
    in_date_window = review_timestamp.ge(window_start) & review_timestamp.lt(
        window_end
    )
    asin = frame["asin"].fillna("").astype(str).str.strip()
    parent_asin = frame["parent_asin"].fillna("").astype(str).str.strip()
    review_title = clean_text(frame["title"])
    review_body = clean_text(frame["text"])
    review_text = (review_title + " " + review_body).str.strip()

    reasons = _primary_drop_reasons(
        rating=rating,
        timestamp=review_timestamp,
        in_date_window=in_date_window,
        asin=asin,
        parent_asin=parent_asin,
        review_text=review_text,
    )
    valid = reasons.isna()

    helpful_vote, valid_helpful = _coerce_bounded_integer(
        frame["helpful_vote"],
        dtype="Int32",
        minimum=0,
        maximum=2_147_483_647,
    )
    invalid_helpful = frame["helpful_vote"].notna() & ~valid_helpful

    verified_purchase = frame["verified_purchase"].where(
        frame["verified_purchase"].isin([True, False]), pd.NA
    )
    invalid_verified = frame["verified_purchase"].notna() & verified_purchase.isna()
    verified_purchase = verified_purchase.astype("boolean")

    output = pd.DataFrame(
        {
            "review_id": make_review_ids(config.dataset_version, source_indices),
            "dataset_version": config.dataset_version,
            "dataset_category": config.dataset_category,
            "source_record_index": source_indices,
            "asin": asin,
            "parent_asin": parent_asin,
            "rating": rating.astype("float32"),
            "review_title": review_title,
            "review_body": review_body,
            "review_text": review_text,
            "review_timestamp": review_timestamp,
            "source_timestamp_ms": timestamp_ms,
            "user_id": frame["user_id"].astype("string").str.strip(),
            "verified_purchase": verified_purchase,
            "helpful_vote": helpful_vote,
        }
    ).loc[valid]

    quality = {
        "input_rows": len(frame),
        "output_rows": len(output),
        "dropped_by_primary_reason": {
            str(reason): int(count)
            for reason, count in reasons.dropna().value_counts().items()
        },
        "invalid_optional_values": {
            "helpful_vote": int(invalid_helpful.sum()),
            "verified_purchase": int(invalid_verified.sum()),
        },
    }
    return output.reset_index(drop=True), quality


def _merge_counts(target: Counter[str], values: dict[str, int]) -> None:
    """Accumulate named quality counts across chunks."""
    target.update({key: int(value) for key, value in values.items()})


def _read_json_line_chunks(
    path: Path,
    *,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """Yield JSONL chunks without imposing a fixed-width numeric parser."""
    if path.suffix == ".gz":
        opener = gzip.open
    elif path.suffix == ".bz2":
        opener = bz2.open
    elif path.suffix in {".xz", ".lzma"}:
        opener = lzma.open
    else:
        opener = Path.open

    records: list[dict[str, Any]] = []
    with opener(path, mode="rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON review record on line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Review record on line {line_number} is not an object"
                )
            records.append(record)
            if len(records) == chunk_size:
                yield pd.DataFrame.from_records(records)
                records = []
    if records:
        yield pd.DataFrame.from_records(records)


def _sql_path(path: Path) -> str:
    """Escape a filesystem path for a DuckDB SQL string literal."""
    return str(path.resolve()).replace("'", "''")


def validate_canonical_review_parquet(
    path: str | Path,
) -> CanonicalReviewArtifactValidation:
    """Validate columns, Arrow types, and required values in canonical Parquet."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)

    parquet_file = pq.ParquetFile(artifact_path)
    actual_schema = parquet_file.schema_arrow
    expected_fields = {field.name: field for field in CANONICAL_REVIEW_SCHEMA}
    actual_fields = {field.name: field for field in actual_schema}
    column_names_match = actual_schema.names == CANONICAL_REVIEW_SCHEMA.names
    column_types_match = all(
        name in actual_fields and actual_fields[name].type.equals(expected.type)
        for name, expected in expected_fields.items()
    )
    column_nullability_match = all(
        name in actual_fields
        and actual_fields[name].nullable == expected.nullable
        for name, expected in expected_fields.items()
    )

    required_names = [
        field.name for field in CANONICAL_REVIEW_SCHEMA if not field.nullable
    ]
    required_null_counts: dict[str, int | None] = {
        name: None for name in required_names if name not in actual_fields
    }
    present_required_names = [
        name for name in required_names if name in actual_fields
    ]

    if present_required_names:
        try:
            import duckdb
        except ImportError as error:
            raise RuntimeError(
                "DuckDB is required for exact canonical null validation"
            ) from error

        expressions = ", ".join(
            f'count(*) FILTER (WHERE "{name}" IS NULL) AS "{name}"'
            for name in present_required_names
        )
        connection = duckdb.connect()
        try:
            values = connection.execute(
                f"SELECT {expressions} "
                f"FROM read_parquet('{_sql_path(artifact_path)}')"
            ).fetchone()
        finally:
            connection.close()
        required_null_counts.update(
            {
                name: int(value)
                for name, value in zip(
                    present_required_names, values, strict=True
                )
            }
        )

    return CanonicalReviewArtifactValidation(
        row_count=parquet_file.metadata.num_rows,
        column_names_match=column_names_match,
        column_types_match=column_types_match,
        column_nullability_match=column_nullability_match,
        required_null_counts=required_null_counts,
    )


def _artifact_reference(path: Path) -> str:
    """Prefer a repository-relative path in persisted provenance."""
    try:
        project_root = find_project_root(path.parent)
        return path.resolve().relative_to(project_root).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def deduplicate_review_parquet(
    input_path: Path,
    output_path: Path,
    *,
    memory_limit: str,
) -> tuple[int, int]:
    """Deduplicate a Parquet artifact with disk-backed DuckDB execution."""
    try:
        import duckdb
    except ImportError as error:
        raise RuntimeError(
            "DuckDB is required for memory-safe full-dataset deduplication"
        ) from error

    source = _sql_path(input_path)
    with TemporaryDirectory(
        prefix=".review_dedup_", dir=output_path.parent
    ) as temporary_directory:
        connection = duckdb.connect()
        temporary = _sql_path(Path(temporary_directory))
        escaped_memory_limit = memory_limit.replace("'", "''")
        try:
            connection.execute("SET TimeZone = 'UTC'")
            connection.execute(f"SET memory_limit = '{escaped_memory_limit}'")
            connection.execute(f"SET temp_directory = '{temporary}'")
            input_rows = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet('{source}')"
                ).fetchone()[0]
            )
            query = f"""
                SELECT * EXCLUDE (_duplicate_rank)
                FROM (
                    SELECT
                        *,
                        row_number() OVER (
                            PARTITION BY
                                user_id,
                                asin,
                                rating,
                                source_timestamp_ms,
                                review_text
                            ORDER BY source_record_index
                        ) AS _duplicate_rank
                    FROM read_parquet('{source}')
                )
                WHERE _duplicate_rank = 1
                ORDER BY source_record_index
            """
            batches = connection.execute(query).to_arrow_reader(
                batch_size=100_000
            )
            writer = pq.ParquetWriter(
                output_path,
                CANONICAL_REVIEW_SCHEMA,
                compression="zstd",
            )
            try:
                for batch in batches:
                    table = pa.Table.from_batches([batch]).cast(
                        CANONICAL_REVIEW_SCHEMA,
                        safe=True,
                    )
                    writer.write_table(table, row_group_size=100_000)
            except Exception:
                writer.close()
                output_path.unlink(missing_ok=True)
                raise
            else:
                writer.close()
            output_rows = pq.ParquetFile(output_path).metadata.num_rows
        finally:
            connection.close()
    return input_rows, output_rows


def preprocess_reviews(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: ReviewPreprocessingConfig,
    report_path: str | Path | None = None,
    max_rows: int | None = None,
) -> ReviewPreprocessingReport:
    """Create an atomic canonical review Parquet and reconciliation report."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_path = destination.with_name(f".{destination.stem}.stage.parquet")
    deduplicated_path = destination.with_name(
        f".{destination.stem}.deduplicated.parquet"
    )
    stage_path.unlink(missing_ok=True)
    deduplicated_path.unlink(missing_ok=True)

    report = ReviewPreprocessingReport(
        dataset_version=config.dataset_version,
        dataset_category=config.dataset_category,
        input_path=_artifact_reference(source),
        output_path=_artifact_reference(destination),
        start_date=config.start_date.isoformat(),
        end_date=config.end_date.isoformat(),
    )
    dropped: Counter[str] = Counter()
    invalid_optional: Counter[str] = Counter()
    writer: pq.ParquetWriter | None = None
    consumed_rows = 0
    min_timestamp: pd.Timestamp | None = None
    max_timestamp: pd.Timestamp | None = None

    try:
        reader = _read_json_line_chunks(source, chunk_size=config.chunk_size)
        for chunk in reader:
            if max_rows is not None:
                remaining = max_rows - consumed_rows
                if remaining <= 0:
                    break
                chunk = chunk.iloc[:remaining].copy()

            transformed, quality = transform_review_chunk(
                chunk,
                config=config,
                source_record_offset=consumed_rows,
            )
            consumed_rows += len(chunk)
            report.input_rows += quality["input_rows"]
            report.valid_rows_before_deduplication += quality["output_rows"]
            _merge_counts(dropped, quality["dropped_by_primary_reason"])
            _merge_counts(invalid_optional, quality["invalid_optional_values"])

            if transformed.empty:
                continue
            chunk_min = transformed["review_timestamp"].min()
            chunk_max = transformed["review_timestamp"].max()
            min_timestamp = (
                chunk_min if min_timestamp is None else min(min_timestamp, chunk_min)
            )
            max_timestamp = (
                chunk_max if max_timestamp is None else max(max_timestamp, chunk_max)
            )

            table = pa.Table.from_pandas(
                transformed,
                schema=CANONICAL_REVIEW_SCHEMA,
                preserve_index=False,
                safe=True,
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    stage_path,
                    CANONICAL_REVIEW_SCHEMA,
                    compression=config.compression,
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if report.valid_rows_before_deduplication == 0:
        stage_path.unlink(missing_ok=True)
        raise ValueError("No valid reviews were produced")

    if config.deduplicate:
        input_rows, output_rows = deduplicate_review_parquet(
            stage_path,
            deduplicated_path,
            memory_limit=config.deduplication_memory_limit,
        )
        if input_rows != report.valid_rows_before_deduplication:
            raise RuntimeError("Deduplication input count does not match build report")
        report.output_rows = output_rows
        report.duplicate_rows_removed = input_rows - output_rows
        deduplicated_path.replace(destination)
        stage_path.unlink(missing_ok=True)
    else:
        report.output_rows = report.valid_rows_before_deduplication
        stage_path.replace(destination)

    report.dropped_by_primary_reason = dict(sorted(dropped.items()))
    report.invalid_optional_values = dict(sorted(invalid_optional.items()))
    report.min_review_timestamp = (
        min_timestamp.isoformat() if min_timestamp is not None else None
    )
    report.max_review_timestamp = (
        max_timestamp.isoformat() if max_timestamp is not None else None
    )

    if report.input_rows != (
        report.valid_rows_before_deduplication + sum(dropped.values())
    ):
        raise RuntimeError("Input, accepted, and dropped counts do not reconcile")
    if report.valid_rows_before_deduplication != (
        report.output_rows + report.duplicate_rows_removed
    ):
        raise RuntimeError("Pre- and post-deduplication counts do not reconcile")

    if report_path is not None:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        report_temp = report_destination.with_suffix(".tmp.json")
        report_temp.write_text(
            json.dumps(asdict(report), indent=2), encoding="utf-8"
        )
        report_temp.replace(report_destination)
    return report


def main() -> None:
    """Build canonical reviews from a registered filtered source artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--no-deduplicate", action="store_true")
    args = parser.parse_args()

    project_root = find_project_root(args.manifest_path.parent)
    manifest = load_dataset_manifest(args.manifest_path)
    source_file = manifest.file_by_role("filtered_reviews")
    config = ReviewPreprocessingConfig(
        dataset_version=manifest.dataset_version,
        dataset_category=manifest.dataset_category,
        start_date=manifest.date_window.start,
        end_date=manifest.date_window.end,
        deduplicate=not args.no_deduplicate,
    )
    report = preprocess_reviews(
        project_root / source_file.path,
        args.output_path,
        config=config,
        report_path=args.report_path,
        max_rows=args.max_rows,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
