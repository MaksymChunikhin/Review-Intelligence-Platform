"""Load and verify immutable dataset manifests."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, TextIO

import pyarrow.parquet as pq

from src.common.project import find_project_root
from src.schemas.dataset import DatasetFile, DatasetManifest


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """Load and validate a JSON dataset manifest."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return DatasetManifest.model_validate(payload)


def sha256_file(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    """Calculate a file SHA-256 digest without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _open_jsonl(path: Path, compression: str) -> TextIO:
    """Open a manifest JSONL file using its declared compression."""
    if compression == "gzip":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def count_jsonl_records(path: str | Path, *, compression: str = "none") -> int:
    """Count non-empty records in a JSONL artifact."""
    with _open_jsonl(Path(path), compression) as stream:
        return sum(bool(line.strip()) for line in stream)


def count_artifact_records(file: DatasetFile, path: Path) -> int:
    """Count records using metadata where the artifact format supports it."""
    if file.format == "jsonl":
        return count_jsonl_records(path, compression=file.compression)
    if file.format == "parquet":
        return pq.ParquetFile(path).metadata.num_rows
    raise ValueError(f"Unsupported artifact format: {file.format}")


def verify_dataset_file(
    file: DatasetFile,
    *,
    project_root: str | Path,
    verify_checksum: bool = False,
    verify_record_count: bool = False,
) -> dict[str, Any]:
    """Verify one file against its manifest declaration."""
    root = Path(project_root)
    path = root / file.path
    result: dict[str, Any] = {
        "role": file.role,
        "path": file.path,
        "exists": path.is_file(),
        "size_matches": False,
        "checksum_matches": None,
        "record_count_matches": None,
    }
    if not result["exists"]:
        return result

    result["actual_size_bytes"] = path.stat().st_size
    result["size_matches"] = result["actual_size_bytes"] == file.size_bytes

    if verify_checksum:
        result["actual_sha256"] = sha256_file(path)
        result["checksum_matches"] = result["actual_sha256"] == file.sha256

    if verify_record_count and file.record_count is not None:
        result["actual_record_count"] = count_artifact_records(file, path)
        result["record_count_matches"] = (
            result["actual_record_count"] == file.record_count
        )
    return result


def verify_dataset_manifest(
    manifest: DatasetManifest,
    *,
    project_root: str | Path,
    verify_checksums: bool = False,
    verify_record_counts: bool = False,
) -> list[dict[str, Any]]:
    """Verify every file registered in a dataset manifest."""
    return [
        verify_dataset_file(
            file,
            project_root=project_root,
            verify_checksum=verify_checksums,
            verify_record_count=verify_record_counts,
        )
        for file in manifest.files
    ]


def manifest_is_valid(results: list[dict[str, Any]]) -> bool:
    """Return whether all requested manifest checks passed."""
    for result in results:
        if not result["exists"] or not result["size_matches"]:
            return False
        if result["checksum_matches"] is False:
            return False
        if result["record_count_matches"] is False:
            return False
    return True


def main() -> None:
    """Verify a dataset manifest from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("--checksums", action="store_true")
    parser.add_argument("--record-counts", action="store_true")
    args = parser.parse_args()

    project_root = find_project_root(args.manifest_path.parent)
    manifest = load_dataset_manifest(args.manifest_path)
    results = verify_dataset_manifest(
        manifest,
        project_root=project_root,
        verify_checksums=args.checksums,
        verify_record_counts=args.record_counts,
    )
    print(json.dumps(results, indent=2))
    if not manifest_is_valid(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
