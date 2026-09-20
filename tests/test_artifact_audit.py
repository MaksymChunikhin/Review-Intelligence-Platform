"""Tests for live, archived, and deliberately pruned artifact lineage."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

from src.common.artifact_audit import audit_artifact_ledger


def test_artifact_audit_accepts_live_archive_and_pruned_records(tmp_path: Path) -> None:
    live = tmp_path / "live.txt"
    live.write_text("live", encoding="utf-8")
    archived_content = b"archived"
    archive = tmp_path / "history.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("old.txt")
        info.size = len(archived_content)
        bundle.addfile(info, io.BytesIO(archived_content))
    ledger = {
        "historical_archive_path": "history.tar.gz",
        "historical_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "records": [
            {"path": "live.txt", "sha256": hashlib.sha256(b"live").hexdigest()},
            {"path": "old.txt", "sha256": hashlib.sha256(archived_content).hexdigest()},
            {
                "path": "removed.bin",
                "sha256": "0" * 64,
                "present": False,
                "status": "historical_model_pruned_after_report_preservation",
            },
        ],
    }
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    result = audit_artifact_ledger(ledger_path, tmp_path)
    assert result.valid
    assert result.live_reference_count == 1
    assert result.archived_reference_count == 1
    assert result.intentionally_pruned_count == 1
