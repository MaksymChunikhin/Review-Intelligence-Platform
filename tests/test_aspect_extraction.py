"""Tests for deterministic multi-aspect lexical extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.aspect_extraction import (
    build_alias_matcher,
    extract_aspects,
    extract_review_aspects,
)


def taxonomy_payload() -> dict[str, object]:
    return {
        "taxonomy_version": "taxonomy_test_v1",
        "status": "approved",
        "aspects": [
            {
                "aspect_id": "scent",
                "canonical_name": "Scent",
                "aliases": ["scent", "fragrance"],
                "additional_aliases": ["smell"],
            },
            {
                "aspect_id": "packaging",
                "canonical_name": "Packaging",
                "aliases": ["bottle design"],
                "additional_aliases": [],
            },
        ],
    }


def test_extract_review_aspects_supports_multiple_aspects_and_exact_spans() -> None:
    matcher, rules = build_alias_matcher(taxonomy_payload())
    text = "Great fragrance, but the bottle design leaks."

    rows = extract_review_aspects(
        "review-1", text, matcher=matcher, rules=rules
    )

    assert [row["aspect_id"] for row in rows] == ["packaging", "scent"]
    for row in rows:
        for evidence in json.loads(row["evidence_json"]):
            assert text[evidence["start"] : evidence["end"]] == evidence["text"]


def test_alias_matcher_uses_boundaries_and_rejects_duplicate_owners() -> None:
    matcher, rules = build_alias_matcher(taxonomy_payload())
    rows = extract_review_aspects(
        "review-1", "The fragrances differ; no exact scent term otherwise.",
        matcher=matcher, rules=rules
    )
    assert [row["aspect_id"] for row in rows] == ["scent"]
    assert [
        value.casefold()
        for value in json.loads(rows[0]["matched_aliases_json"])
    ] == ["scent"]

    duplicate = taxonomy_payload()
    duplicate["aspects"][1]["aliases"] = ["fragrance"]
    with pytest.raises(ValueError, match="multiple aspects"):
        build_alias_matcher(duplicate)


def test_extract_aspects_writes_versioned_parquet_and_report(tmp_path: Path) -> None:
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(taxonomy_payload()), encoding="utf-8")
    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "review_text": [
                "Lovely scent and smart bottle design.",
                "No matching topic here.",
                "The smell is unpleasant.",
            ],
        }
    ).to_parquet(reviews_path, index=False)
    output_path = tmp_path / "extracted.parquet"
    report_path = tmp_path / "report.json"

    report = extract_aspects(
        taxonomy_path,
        reviews_path,
        output_path,
        extraction_version="extractor_test_v1",
        report_path=report_path,
    )
    extracted = pd.read_parquet(output_path)

    assert report.review_count == 3
    assert report.matched_review_count == 2
    assert report.no_match_review_count == 1
    assert report.review_aspect_count == 3
    assert report.evidence_span_count == 3
    assert set(extracted["aspect_id"]) == {"scent", "packaging"}
    assert set(extracted["extraction_version"]) == {"extractor_test_v1"}
    assert json.loads(report_path.read_text())["output_sha256"]


def test_extraction_rejects_nonapproved_taxonomy() -> None:
    payload = taxonomy_payload()
    payload["status"] = "candidate"
    with pytest.raises(ValueError, match="approved taxonomy"):
        build_alias_matcher(payload)
