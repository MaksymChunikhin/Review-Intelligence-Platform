"""Contracts for grounded Gemini analyst generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalystRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AnalystConfig(AnalystRecord):
    analyst_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    provider: Literal["google_genai"] = "google_genai"
    backend: Literal["vertex_ai", "gemini_api"] = "gemini_api"
    model: str = Field(min_length=1)
    location: str | None = Field(default=None, min_length=1)
    temperature: float = Field(ge=0, le=1)
    seed: int = Field(ge=0)
    thinking_level: Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"]
    max_output_tokens: int = Field(ge=256, le=16_384)
    max_retries: int = Field(ge=1, le=5)
    retry_base_seconds: float = Field(ge=0, le=30)
    request_timeout_seconds: float = Field(default=45, ge=5, le=300)
    maximum_evidence_reviews: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def validate_backend(self) -> "AnalystConfig":
        if self.backend == "vertex_ai" and self.location is None:
            raise ValueError("location is required for Vertex AI")
        return self


class AnalystFinding(AnalystRecord):
    finding: str = Field(min_length=1, max_length=800)
    basis: Literal["aggregate", "reviews", "both"]
    citations: list[
        Annotated[str, Field(pattern=r"^R[1-8]$")]
    ] = Field(max_length=8)

    @model_validator(mode="after")
    def require_review_citations(self) -> "AnalystFinding":
        if self.basis in {"reviews", "both"} and not self.citations:
            raise ValueError("review-based findings require citations")
        if self.basis == "aggregate" and self.citations:
            raise ValueError("aggregate-only findings cannot cite reviews")
        return self


class GroundedAnalystAnswer(AnalystRecord):
    answer_ru: str = Field(min_length=1, max_length=4_000)
    findings: list[AnalystFinding] = Field(min_length=1, max_length=8)
    recommendations: list[str] = Field(max_length=6)
    limitations: list[str] = Field(min_length=1, max_length=6)


def load_analyst_config(path: str | Path) -> AnalystConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        return AnalystConfig.model_validate(json.load(stream))
