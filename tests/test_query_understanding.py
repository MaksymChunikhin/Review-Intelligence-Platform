"""Tests for deterministic bilingual query routing."""

from src.rag.query_understanding import (
    infer_analytics_intent,
    infer_evidence_sentiment,
    understand_evidence_query,
)


def test_understand_russian_dispenser_question() -> None:
    aspect, query = understand_evidence_query("Почему покупатели жалуются на дозатор?")
    assert aspect == "dispenser_functionality"
    assert "sprayer" in query


def test_explicit_aspect_controls_default_query() -> None:
    aspect, query = understand_evidence_query(
        "Что происходит?", aspect_id="price_affordability"
    )
    assert aspect == "price_affordability"
    assert "price" in query


def test_generic_english_complaint_routes_to_negative_problem_search() -> None:
    aspect, query = understand_evidence_query(
        "Why are people complaining about the product?"
    )
    assert aspect is None
    assert "disappointed" in query
    assert infer_evidence_sentiment(
        "Why are people complaining about the product?"
    ) == "negative"


def test_customer_question_catalog_terms_route_to_expected_aspects() -> None:
    examples = {
        "For which hair types do customers recommend the product?": (
            "hair_type_compatibility"
        ),
        "Which products are praised most often for adding shine?": (
            "shine_appearance"
        ),
        "Which products are described as good value for money?": (
            "price_affordability"
        ),
        "What do customers with damaged or bleached hair praise?": (
            "damage_repair_hair_health"
        ),
    }
    for question, expected_aspect in examples.items():
        aspect, _ = understand_evidence_query(question)
        assert aspect == expected_aspect


def test_catalog_praise_and_criticism_infer_sentiment() -> None:
    assert infer_evidence_sentiment(
        "What do customers praise most often?"
    ) == "positive"
    assert infer_evidence_sentiment(
        "Which attributes occur most often in 1–2 star reviews?"
    ) == "negative"
    assert infer_evidence_sentiment(
        "What do customers criticize most often?"
    ) == "negative"


def test_comparison_questions_route_to_deterministic_analytics() -> None:
    assert infer_analytics_intent(
        "Which products are praised most often for hydration?"
    ) == "aspect_product_comparison"
    assert infer_analytics_intent(
        "Which products do customers most often say they would buy again?"
    ) == "repurchase_product_comparison"
    assert infer_analytics_intent(
        "Are there highly rated products with recurring complaints?"
    ) == "high_rating_recurring_complaints"
    assert infer_analytics_intent(
        "What do customers praise most often? Top 5 reasons."
    ) == "seller_top_reasons"


def test_cleaning_action_wins_over_oily_hair_context() -> None:
    aspect, _ = understand_evidence_query(
        "Does this shampoo cleanse oily hair well?"
    )
    assert aspect == "cleansing"
