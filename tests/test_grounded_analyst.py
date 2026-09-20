"""Tests for citation-validated Gemini analyst answers."""

from __future__ import annotations

import pytest

from src.llm.gemini_client import GeminiResponseMetadata
from src.llm.grounded_analyst import (
    GroundedAnswerValidationError,
    generate_grounded_answer,
)
from src.schemas.analyst import (
    AnalystConfig,
    AnalystFinding,
    GroundedAnalystAnswer,
)


class FakeStructuredClient:
    def __init__(self, answer: GroundedAnalystAnswer) -> None:
        self.answer = answer

    def generate(self, **_: object):
        return (
            self.answer,
            GeminiResponseMetadata(
                response_id="response-1",
                model_version="gemini-test",
                usage_metadata={"total_token_count": 100},
            ),
            "{}",
        )


def _config() -> AnalystConfig:
    return AnalystConfig(
        analyst_version="analyst_test_v1",
        backend="gemini_api",
        model="gemini-test",
        location=None,
        temperature=0,
        seed=42,
        thinking_level="LOW",
        max_output_tokens=512,
        max_retries=1,
        retry_base_seconds=0,
        maximum_evidence_reviews=8,
    )


def _context() -> dict[str, object]:
    return {
        "question": "Почему ломается дозатор?",
        "instructions": "Use only context.",
        "context": "[R1] brand=Seller | The sprayer broke.",
        "citations": [
            {"citation_id": "R1", "review_id": "private-1", "parent_asin": "p1"}
        ],
    }


def test_grounded_answer_accepts_supplied_citation() -> None:
    answer = GroundedAnalystAnswer(
        answer_ru="В одном найденном отзыве сломался распылитель [R1].",
        findings=[
            AnalystFinding(
                finding="Есть жалоба на поломку распылителя.",
                basis="reviews",
                citations=["R1"],
            )
        ],
        recommendations=["Проверить конструкцию распылителя."],
        limitations=["Это исторические отзывы."],
    )

    result, metadata = generate_grounded_answer(
        _context(), _config(), client=FakeStructuredClient(answer)
    )

    assert result.answer_ru.endswith("[R1].")
    assert metadata["used_citations"] == ["R1"]


def test_grounded_answer_rejects_unknown_citation() -> None:
    answer = GroundedAnalystAnswer(
        answer_ru="Вывод якобы подтверждён [R2].",
        findings=[
            AnalystFinding(
                finding="Неподтверждённый вывод.",
                basis="reviews",
                citations=["R2"],
            )
        ],
        recommendations=[],
        limitations=["Ограничение."],
    )

    with pytest.raises(GroundedAnswerValidationError, match="unknown citations"):
        generate_grounded_answer(
            _context(), _config(), client=FakeStructuredClient(answer)
        )


def test_grounded_answer_rejects_non_source_bracket_tag() -> None:
    answer = GroundedAnalystAnswer(
        answer_ru="Агрегированный вывод [AGGREGATE], пример [R1].",
        findings=[
            AnalystFinding(
                finding="Есть жалоба.",
                basis="reviews",
                citations=["R1"],
            )
        ],
        recommendations=[],
        limitations=["Ограничение."],
    )

    with pytest.raises(GroundedAnswerValidationError, match="AGGREGATE"):
        generate_grounded_answer(
            _context(), _config(), client=FakeStructuredClient(answer)
        )
