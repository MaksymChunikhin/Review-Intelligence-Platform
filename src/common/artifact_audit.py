"""Validate artifact ledgers, including files intentionally moved to archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ingestion.dataset_manifest import sha256_file


@dataclass(frozen=True)
class ArtifactAuditResult:
    ledger_path: str
    checked_reference_count: int
    live_reference_count: int
    archived_reference_count: int
    intentionally_pruned_count: int
    errors: list[str]

    @property
    def valid(self) -> bool:
        return not self.errors


def _artifact_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(
            value.get("sha256"), str
        ):
            records.append(value)
        for child in value.values():
            records.extend(_artifact_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_artifact_records(child))
    return records


def audit_artifact_ledger(
    ledger_path: str | Path, project_root: str | Path
) -> ArtifactAuditResult:
    """Verify live files, archived members, and declared pruned artifacts."""
    ledger_source = Path(ledger_path)
    root = Path(project_root).resolve()
    payload = json.loads(ledger_source.read_text(encoding="utf-8"))
    archive_value = payload.get("historical_archive_path")
    archive = (root / archive_value).resolve() if archive_value else None
    archive_members: set[str] = set()
    errors: list[str] = []
    if archive is not None:
        if not archive.is_file():
            errors.append(f"historical archive is missing: {archive_value}")
        elif sha256_file(archive) != payload.get("historical_archive_sha256"):
            errors.append(f"historical archive hash mismatch: {archive_value}")
        else:
            with tarfile.open(archive, "r:gz") as bundle:
                archive_members = set(bundle.getnames())

    live = archived = pruned = 0
    records = _artifact_records(payload)
    for record in records:
        relative = str(record["path"])
        target = (root / relative).resolve()
        if root not in target.parents:
            errors.append(f"artifact escapes project root: {relative}")
            continue
        if target.is_file():
            live += 1
            if sha256_file(target) != str(record["sha256"]):
                errors.append(f"artifact hash mismatch: {relative}")
            continue
        if relative in archive_members:
            archived += 1
            with tarfile.open(archive, "r:gz") as bundle:
                stream = bundle.extractfile(relative)
                if stream is None:
                    errors.append(f"cannot read archived artifact: {relative}")
                elif hashlib.sha256(stream.read()).hexdigest() != str(
                    record["sha256"]
                ):
                    errors.append(f"archived artifact hash mismatch: {relative}")
            continue
        if record.get("present") is False and "pruned" in str(
            record.get("status", "")
        ):
            pruned += 1
            continue
        errors.append(f"artifact is missing: {relative}")

    return ArtifactAuditResult(
        ledger_path=str(ledger_source),
        checked_reference_count=len(records),
        live_reference_count=live,
        archived_reference_count=archived,
        intentionally_pruned_count=pruned,
        errors=errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("ledger_paths", nargs="+", type=Path)
    args = parser.parse_args()
    results = [
        audit_artifact_ledger(path, args.project_root) for path in args.ledger_paths
    ]
    print(json.dumps([asdict(item) | {"valid": item.valid} for item in results], indent=2))
    if any(not item.valid for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
