"""Small deterministic Russian/English query router for supported hair care."""

from __future__ import annotations

import re
from typing import Literal


AnalyticsIntent = Literal[
    "seller_top_reasons",
    "aspect_product_comparison",
    "repurchase_product_comparison",
    "high_rating_recurring_complaints",
]


ASPECT_QUERY_TERMS = {
    "dispenser_functionality": {
        "triggers": ("дозатор", "распылител", "спре", "pump", "sprayer", "nozzle"),
        "query": "broken sprayer pump nozzle trigger does not work",
    },
    "packaging_integrity": {
        "triggers": ("упаков", "бутыл", "протек", "packaging", "bottle", "leak"),
        "query": "packaging bottle leaked damaged arrived broken",
    },
    "scent": {
        "triggers": ("запах", "аромат", "scent", "smell", "fragrance"),
        "query": "scent smell fragrance strong unpleasant pleasant",
    },
    "hair_type_compatibility": {
        "triggers": (
            "тип волос",
            "тонкими волос",
            "сухими волос",
            "hair type",
            "types of hair",
            "fine hair",
            "thin hair",
        ),
        "query": "hair type fine thin thick dry damaged compatibility suitable",
    },
    "shine_appearance": {
        "triggers": ("блеск", "сияни", "shine", "shiny", "glossy"),
        "query": "hair shine shiny glossy appearance",
    },
    "damage_repair_hair_health": {
        "triggers": (
            "поврежден",
            "осветлен",
            "damaged hair",
            "bleached hair",
            "damage repair",
        ),
        "query": "damaged bleached hair repair healthier restored",
    },
    "curl_definition": {
        "triggers": ("кудр", "локон", "curly hair", "curls", "waves"),
        "query": "curly hair curls waves curl pattern",
    },
    "residue_buildup": {
        "triggers": ("остат", "налет", "residue", "buildup"),
        "query": "residue buildup coating left on hair",
    },
    "hair_weight": {
        "triggers": ("утяж", "тяжел", "weigh", "heavy hair"),
        "query": "heavy weighed hair down flat",
    },
    "volume_body": {
        "triggers": ("объем", "объём", "пышн", "volume", "body", "fuller hair"),
        "query": "hair volume body fuller flat limp",
    },
    "price_affordability": {
        "triggers": (
            "соотношение цены",
            "стоит своих денег",
            "value for money",
            "worth the price",
            "worth it",
            "цен",
            "дорог",
            "дешев",
            "price",
            "expensive",
            "cheap",
        ),
        "query": "price expensive affordable cheap cost",
    },
    "manageability": {
        "triggers": ("послуш", "уклад", "manageable", "manageability"),
        "query": "hair manageable easy to style control",
    },
    "detangling": {
        "triggers": ("распут", "колтун", "tangle", "detang"),
        "query": "detangle tangles knots easy comb",
    },
    "frizz_control": {
        "triggers": ("пушист", "фриз", "frizz", "flyaway"),
        "query": "frizz control flyaways smooth hair",
    },
    "moisture_and_hydration": {
        "triggers": (
            "увлаж",
            "сух",
            "moisture",
            "hydrate",
            "hydration",
            "dry hair",
            "hair feel dry",
        ),
        "query": "moisture hydration dry hair moisturized",
    },
    "softness": {
        "triggers": ("мягк", "soft", "softness"),
        "query": "soft softness silky hair",
    },
    "hair_strength": {
        "triggers": ("ломк", "укреп", "breakage", "strength"),
        "query": "hair breakage stronger weak brittle",
    },
    "hair_growth_loss": {
        "triggers": ("выпад", "рост волос", "hair loss", "hair growth", "shedding"),
        "query": "hair growth hair loss shedding hair fall",
    },
    "greasy_oily_finish": {
        "triggers": ("жирн", "маслян", "greasy", "oily"),
        "query": "greasy oily hair finish",
    },
    "lather_foam": {
        "triggers": ("пен", "lather", "foam", "suds"),
        "query": "lather foam suds rich weak",
    },
    "cleansing": {
        "triggers": ("очища", "промыва", "cleanse", "cleansing", "washes hair"),
        "query": "cleanse clean hair scalp wash removes oil dirt buildup",
    },
    "conditioning_effect": {
        "triggers": (
            "кондициониру",
            "conditioning effect",
            "feel conditioned",
            "leave hair conditioned",
            "leaves hair conditioned",
        ),
        "query": "conditioning effect leaves hair conditioned smooth manageable",
    },
    "oil_control_absorption": {
        "triggers": (
            "впитывает жир",
            "контроль жирности",
            "absorb oil",
            "absorbs oil",
            "oil control",
            "between washes",
        ),
        "query": "absorbs oil oil control fresh between washes",
    },
    "overall_performance": {
        "triggers": (
            "эффектив",
            "работает",
            "результат",
            "performance",
            "works",
            "results",
        ),
        "query": "product works results effective ineffective",
    },
}

COMPLAINT_TRIGGERS = (
    "complain",
    "complaint",
    "unhappy",
    "dislike",
    "what is wrong",
    "criticize",
    "criticism",
    "negative feedback",
    "lower product ratings",
    "lower rating",
    "1-star",
    "1 star",
    "1–2 star",
    "1-2 star",
    "unmet",
    "not repurchase",
    "not to repurchase",
    "жалоб",
    "жалуются",
    "недоволь",
    "не нравится",
    "проблем",
)
PRAISE_TRIGGERS = (
    "praise",
    "positive review",
    "high rating",
    "5-star",
    "5 star",
    "recommend the product",
    "customers like",
    "customers love",
    "buy again",
    "repurchase",
    "хвал",
    "положительн",
    "5 звезд",
    "рекомендуют продукт",
    "покупатели любят",
    "купят снова",
    "повторн",
)
COMPLAINT_QUERY = (
    "bad problem disappointed does not work dry greasy heavy sticky "
    "unpleasant smell worse"
)


def understand_evidence_query(
    question: str,
    *,
    aspect_id: str | None = None,
    evidence_query: str | None = None,
) -> tuple[str | None, str]:
    """Resolve an aspect and English evidence query without an external call."""
    if evidence_query is not None and len(evidence_query.strip()) >= 2:
        return aspect_id, evidence_query.strip()
    if aspect_id is not None and aspect_id in ASPECT_QUERY_TERMS:
        return aspect_id, str(ASPECT_QUERY_TERMS[aspect_id]["query"])
    normalized = re.sub(r"\s+", " ", question.lower()).strip()
    # Choose the most specific matching phrase. Dictionary order previously
    # routed questions such as "cleanse oily hair" to oily finish merely
    # because that aspect appeared first.
    matches: list[tuple[int, int, str, dict[str, object]]] = []
    for candidate_aspect, definition in ASPECT_QUERY_TERMS.items():
        matched = [
            str(trigger)
            for trigger in definition["triggers"]
            if str(trigger) in normalized
        ]
        if matched:
            best = max(matched, key=lambda value: (len(value.split()), len(value)))
            matches.append(
                (len(best.split()), len(best), candidate_aspect, definition)
            )
    if matches:
        _, _, candidate_aspect, definition = max(matches)
        return candidate_aspect, str(definition["query"])
    if any(trigger in normalized for trigger in COMPLAINT_TRIGGERS):
        return aspect_id, COMPLAINT_QUERY
    if re.search(r"[a-z]{2,}", normalized):
        return aspect_id, normalized
    raise ValueError(
        "Could not determine the question topic; select an aspect or provide "
        "an English evidence-search phrase."
    )


def infer_evidence_sentiment(question: str) -> str | None:
    """Infer only high-confidence sentiment intent from a user question."""
    normalized = re.sub(r"\s+", " ", question.lower()).strip()
    if any(trigger in normalized for trigger in COMPLAINT_TRIGGERS):
        return "negative"
    if any(trigger in normalized for trigger in PRAISE_TRIGGERS):
        return "positive"
    return None


def infer_analytics_intent(question: str) -> AnalyticsIntent | None:
    """Route questions that require deterministic aggregate calculations."""
    normalized = re.sub(r"\s+", " ", question.lower()).strip()
    product_comparison = (
        any(
            trigger in normalized
            for trigger in (
            "which product",
            "which brand",
            "which item",
            "what product",
            "what brand",
            "какие продукт",
            "какие товар",
            "какие бренд",
            "какой продукт",
            "какой товар",
            )
        )
        or bool(re.search(r"\bproducts\b", normalized))
    ) and not any(
        phrase in normalized
        for phrase in (
            "which product attribute",
            "what product attribute",
            "какие характеристики продукт",
        )
    )
    if (
        product_comparison
        and any(
            term in normalized
            for term in ("high rating", "highly rated", "высоким рейтинг")
        )
        and any(term in normalized for term in ("recurring complaint", "повторяющихся жалоб"))
    ):
        return "high_rating_recurring_complaints"
    if product_comparison and any(
        term in normalized
        for term in ("buy again", "repurchase", "купят снова", "повторн")
    ):
        return "repurchase_product_comparison"
    if product_comparison:
        return "aspect_product_comparison"
    if any(
        term in normalized
        for term in (
            "top 5",
            "most often",
            "чаще всего",
            "топ-5",
        )
    ) and any(
        term in normalized
        for term in (
            "praise",
            "criticize",
            "issue",
            "attribute",
            "хвал",
            "критик",
            "проблем",
            "характеристик",
        )
    ):
        return "seller_top_reasons"
    return None
