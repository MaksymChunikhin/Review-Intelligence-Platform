"""Validate human gold annotations for multi-aspect extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GoldAspectEvidence(BaseModel):
    """Store exact source phrases supporting one annotated aspect."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    aspect_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    customer_phrases: list[str] = Field(min_length=1)

    @field_validator("customer_phrases")
    @classmethod
    def validate_customer_phrases(cls, phrases: list[str]) -> list[str]:
        """Reject evidence that cannot identify distinct source spans."""
        if any(not phrase for phrase in phrases):
            raise ValueError("gold evidence phrases must not be empty")
        if len(phrases) != len(set(phrases)):
            raise ValueError("gold evidence phrases must be unique within an aspect")
        return phrases


class GoldReviewAnnotation(BaseModel):
    """Represent the completed aspect annotation for one review."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    annotation_id: str = Field(min_length=1)
    review_id: str = Field(min_length=1)
    aspect_ids: list[str]
    evidence: list[GoldAspectEvidence]
    no_supported_aspect: bool
    annotation_status: Literal["complete"]
    annotator_notes: str = ""

    @model_validator(mode="after")
    def validate_aspect_evidence_alignment(self) -> "GoldReviewAnnotation":
        """Require unique aspects, complete evidence, and exclusive none state."""
        if len(self.aspect_ids) != len(set(self.aspect_ids)):
            raise ValueError("gold aspect IDs must be unique within a review")
        evidence_ids = [item.aspect_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("gold evidence aspect IDs must be unique")
        if self.no_supported_aspect:
            if self.aspect_ids or self.evidence:
                raise ValueError(
                    "no-supported-aspect rows cannot contain aspects or evidence"
                )
        elif not self.aspect_ids:
            raise ValueError("a completed review must contain an aspect or none")
        if set(self.aspect_ids) != set(evidence_ids):
            raise ValueError("every gold aspect must have exact evidence")
        return self


@dataclass(frozen=True)
class AspectAnnotationValidationReport:
    """Summarize pending and validated annotation rows."""

    review_count: int
    completed_review_count: int
    pending_review_count: int
    annotated_aspect_mention_count: int
    no_supported_aspect_review_count: int


def _parse_json_list(value: str, *, field_name: str) -> list[object]:
    """Parse a JSON list from one completed CSV field."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field_name} must contain valid JSON") from error
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must contain a JSON list")
    return parsed


def validate_aspect_annotation_queue(
    queue_path: str | Path,
    taxonomy_path: str | Path,
    *,
    require_complete: bool = False,
) -> AspectAnnotationValidationReport:
    """Validate IDs, exact phrases, and completion state in an annotation CSV."""
    queue = pd.read_csv(queue_path, keep_default_na=False)
    required_columns = {
        "annotation_id",
        "review_id",
        "review_text",
        "human_aspect_ids",
        "human_evidence_json",
        "no_supported_aspect",
        "annotation_status",
        "annotator_notes",
    }
    missing_columns = required_columns - set(queue.columns)
    if missing_columns:
        raise ValueError(
            f"annotation queue is missing columns: {sorted(missing_columns)}"
        )
    if queue["annotation_id"].duplicated().any():
        raise ValueError("annotation IDs must be unique")
    if queue["review_id"].duplicated().any():
        raise ValueError("annotation review IDs must be unique")

    taxonomy = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
    if taxonomy.get("status") != "approved":
        raise ValueError("annotation validation requires an approved taxonomy")
    allowed_aspects = {aspect["aspect_id"] for aspect in taxonomy["aspects"]}

    completed_count = 0
    mention_count = 0
    none_count = 0
    allowed_statuses = {"pending", "complete"}
    statuses = set(queue["annotation_status"])
    if not statuses <= allowed_statuses:
        raise ValueError("annotation status must be pending or complete")
    for row in queue.itertuples(index=False):
        if row.annotation_status == "pending":
            if any(
                [
                    row.human_aspect_ids,
                    row.human_evidence_json,
                    row.no_supported_aspect,
                ]
            ):
                raise ValueError(
                    f"pending annotation contains labels: {row.annotation_id}"
                )
            continue

        aspect_ids_raw = _parse_json_list(
            row.human_aspect_ids,
            field_name="human_aspect_ids",
        )
        if not all(isinstance(value, str) for value in aspect_ids_raw):
            raise ValueError("human_aspect_ids values must be strings")
        aspect_ids = [str(value) for value in aspect_ids_raw]
        unexpected = set(aspect_ids) - allowed_aspects
        if unexpected:
            raise ValueError(
                f"annotation uses unknown aspects: {sorted(unexpected)}"
            )
        evidence_raw = _parse_json_list(
            row.human_evidence_json,
            field_name="human_evidence_json",
        )
        no_supported = str(row.no_supported_aspect).casefold()
        if no_supported not in {"true", "false"}:
            raise ValueError("no_supported_aspect must be true or false")
        annotation = GoldReviewAnnotation.model_validate(
            {
                "annotation_id": row.annotation_id,
                "review_id": row.review_id,
                "aspect_ids": aspect_ids,
                "evidence": evidence_raw,
                "no_supported_aspect": no_supported == "true",
                "annotation_status": "complete",
                "annotator_notes": row.annotator_notes,
            }
        )
        for evidence in annotation.evidence:
            for phrase in evidence.customer_phrases:
                if phrase not in row.review_text:
                    raise ValueError(
                        f"evidence is not an exact source substring: "
                        f"{row.annotation_id} / {evidence.aspect_id} / {phrase!r}"
                    )
        completed_count += 1
        mention_count += len(annotation.aspect_ids)
        none_count += int(annotation.no_supported_aspect)

    pending_count = len(queue) - completed_count
    if require_complete and pending_count:
        raise ValueError(f"annotation queue has {pending_count} pending rows")
    return AspectAnnotationValidationReport(
        review_count=len(queue),
        completed_review_count=completed_count,
        pending_review_count=pending_count,
        annotated_aspect_mention_count=mention_count,
        no_supported_aspect_review_count=none_count,
    )
