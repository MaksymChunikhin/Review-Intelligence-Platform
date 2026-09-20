"""Tests for exact-evidence aspect annotation validation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.aspect_annotations import validate_aspect_annotation_queue


def write_taxonomy(path: Path) -> None:
    """Write a compact approved taxonomy fixture."""
    path.write_text(
        json.dumps(
            {
                "status": "approved",
                "aspects": [
                    {"aspect_id": "scent"},
                    {"aspect_id": "softness_smoothness"},
                ],
            }
        ),
        encoding="utf-8",
    )


def make_queue() -> pd.DataFrame:
    """Return one complete and one pending annotation row."""
    return pd.DataFrame(
        [
            {
                "annotation_id": "A1",
                "review_id": "R1",
                "review_text": "Smells great and makes my hair soft.",
                "human_aspect_ids": json.dumps(
                    ["scent", "softness_smoothness"]
                ),
                "human_evidence_json": json.dumps(
                    [
                        {
                            "aspect_id": "scent",
                            "customer_phrases": ["Smells great"],
                        },
                        {
                            "aspect_id": "softness_smoothness",
                            "customer_phrases": ["makes my hair soft"],
                        },
                    ]
                ),
                "no_supported_aspect": "false",
                "annotation_status": "complete",
                "annotator_notes": "",
            },
            {
                "annotation_id": "A2",
                "review_id": "R2",
                "review_text": "It arrived yesterday.",
                "human_aspect_ids": "",
                "human_evidence_json": "",
                "no_supported_aspect": "",
                "annotation_status": "pending",
                "annotator_notes": "",
            },
        ]
    )


def test_annotation_validation_accepts_exact_evidence_and_pending_rows(
    tmp_path: Path,
) -> None:
    taxonomy = tmp_path / "taxonomy.json"
    queue_path = tmp_path / "queue.csv"
    write_taxonomy(taxonomy)
    make_queue().to_csv(queue_path, index=False)

    report = validate_aspect_annotation_queue(queue_path, taxonomy)

    assert report.review_count == 2
    assert report.completed_review_count == 1
    assert report.pending_review_count == 1
    assert report.annotated_aspect_mention_count == 2


def test_annotation_validation_rejects_nonverbatim_evidence(
    tmp_path: Path,
) -> None:
    taxonomy = tmp_path / "taxonomy.json"
    queue_path = tmp_path / "queue.csv"
    write_taxonomy(taxonomy)
    queue = make_queue().iloc[:1].copy()
    evidence = json.loads(queue.loc[0, "human_evidence_json"])
    evidence[0]["customer_phrases"] = ["smells wonderful"]
    queue.loc[0, "human_evidence_json"] = json.dumps(evidence)
    queue.to_csv(queue_path, index=False)

    with pytest.raises(ValueError, match="exact source substring"):
        validate_aspect_annotation_queue(queue_path, taxonomy)


@pytest.mark.parametrize(
    ("phrases", "message"),
    [
        ([""], "must not be empty"),
        (["   "], "must not be empty"),
        (["Smells great", "Smells great"], "must be unique"),
    ],
)
def test_annotation_validation_rejects_empty_or_duplicate_evidence_phrases(
    tmp_path: Path,
    phrases: list[str],
    message: str,
) -> None:
    taxonomy = tmp_path / "taxonomy.json"
    queue_path = tmp_path / "queue.csv"
    write_taxonomy(taxonomy)
    queue = make_queue().iloc[:1].copy()
    evidence = json.loads(queue.loc[0, "human_evidence_json"])
    evidence[0]["customer_phrases"] = phrases
    queue.loc[0, "human_evidence_json"] = json.dumps(evidence)
    queue.to_csv(queue_path, index=False)

    with pytest.raises(ValueError, match=message):
        validate_aspect_annotation_queue(queue_path, taxonomy)
