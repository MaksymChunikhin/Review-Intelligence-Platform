"""Build a bounded, citation-ready context for a grounded analyst answer."""

from __future__ import annotations

from typing import Any


GROUNDING_INSTRUCTIONS = """You are a customer-review analyst.
Answer in English using only the aggregate facts and reviews in the context.
Cite every review-based claim with an individual source such as [R1] or [R2].
Do not turn a single review into a general conclusion.
If the evidence is insufficient, say so explicitly.
Always state the historical data period and the model limitations."""


def build_rag_context(
    *,
    question: str,
    report: dict[str, Any],
    evidence: list[dict[str, Any]],
    analytics_facts: list[str] | None = None,
    max_characters: int = 12_000,
) -> dict[str, Any]:
    """Create compact aggregate facts plus individually citable review evidence."""
    if len(question.strip()) < 2:
        raise ValueError("question must contain at least two characters")
    if max_characters < 1_000:
        raise ValueError("max_characters must be at least 1000")
    products = report["products"]
    seller = next(item for item in products if item["role"] == "seller")
    lines = [
        "SCOPE",
        f"period={report['review_date_start']}..{report['review_date_end']}",
        f"seller_brand={seller['store']} | reviews={seller['review_count']}",
        f"competitors={len(products) - 1}",
        "",
        "AGGREGATE FACTS",
    ]
    for section_name in ("strengths", "weaknesses", "complaints"):
        for item in report[section_name]:
            lines.append(
                f"{section_name}: {item['aspect_name']}; mentions={item['mention_count']}; "
                f"seller_attention={item['seller_attention_share']}; "
                f"competitor_attention={item['competitor_attention_share']}; "
                f"attention_gap_pp={item['attention_gap_pp']}; "
                f"seller_positive={item['seller_positive_share']}; "
                f"competitor_positive={item['competitor_positive_share']}"
            )
    if analytics_facts:
        lines.extend(["", "DETERMINISTIC QUESTION FACTS", *analytics_facts])
    lines.extend(["", "REVIEW EVIDENCE"])
    citations: list[dict[str, Any]] = []
    for index, item in enumerate(evidence, start=1):
        citation_id = f"R{index}"
        entry = (
            f"[{citation_id}] brand={item['store']} | "
            f"rating={item['rating']} | date={item['review_timestamp'][:10]} | "
            f"aspect={item['aspect_id']} | sentiment={item['aspect_sentiment']}\n"
            f"{item['review_text']}"
        )
        candidate = "\n".join([*lines, entry])
        if len(candidate) > max_characters:
            break
        lines.append(entry)
        citations.append(
            {
                "citation_id": citation_id,
                "review_id": item["review_id"],
                "parent_asin": item["parent_asin"],
                "store": item["store"],
                "rating": item["rating"],
                "review_timestamp": item["review_timestamp"],
                "review_text": item["review_text"],
                "aspect_id": item["aspect_id"],
                "aspect_sentiment": item["aspect_sentiment"],
            }
        )
    return {
        "question": question,
        "instructions": GROUNDING_INSTRUCTIONS,
        "context": "\n".join(lines),
        "citations": citations,
        "limitations": report["limitations"],
    }
