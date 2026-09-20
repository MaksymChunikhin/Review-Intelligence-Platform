"""Typed public contracts for the seller analytics API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiRecord):
    status: Literal["ok", "degraded"]
    service: str
    active_niche: str | None
    workspace_count: int


class WorkspaceSummary(ApiRecord):
    workspace_id: str
    workspace_version: str
    niche_id: str
    niche_name: str
    seller_parent_asins: list[str]
    competitor_parent_asins: list[str]
    review_date_start: str
    review_date_end: str


class WorkspaceListResponse(ApiRecord):
    items: list[WorkspaceSummary]


class CreateWorkspaceRequest(ApiRecord):
    seller_parent_asin: str = Field(pattern=r"^[A-Z0-9]{10}$")
    top_k: int = Field(default=4, ge=1, le=8)
    minimum_review_count: int = Field(default=250, ge=20, le=10_000)


class ProductSearchItem(ApiRecord):
    parent_asin: str
    product_title: str
    store: str | None
    price_at_collection: float | None
    average_rating: float | None
    rating_number: int | None
    review_count: int
    analytical_family_name: str | None = None
    competitor_niche_name: str | None = None


class ProductSearchResponse(ApiRecord):
    query: str
    niche_id: str
    items: list[ProductSearchItem]


class ProductSummary(ApiRecord):
    parent_asin: str
    product_title: str
    store: str | None
    review_count: int
    average_rating: float
    needs_attention_share: float
    role: Literal["seller", "competitor"]


class AspectInsight(ApiRecord):
    aspect_id: str
    aspect_name: str
    mention_count: int
    seller_attention_share: float | None
    competitor_attention_share: float | None
    attention_gap_pp: float | None
    seller_positive_share: float | None
    competitor_positive_share: float | None


class ProductAspectRankingItem(ApiRecord):
    parent_asin: str
    store: str | None
    product_title: str
    product_review_count: int
    average_rating: float
    mention_count: int
    sentiment_count: int
    sentiment_share: float
    support_sufficient: bool


class AspectProductRanking(ApiRecord):
    aspect_id: str
    aspect_name: str
    positive_products: list[ProductAspectRankingItem]
    negative_products: list[ProductAspectRankingItem]


class RepurchaseProductRanking(ApiRecord):
    parent_asin: str
    store: str | None
    product_title: str
    review_count: int
    average_rating: float
    signal_count: int
    signal_share: float
    support_sufficient: bool


class RecurringComplaintRanking(ApiRecord):
    parent_asin: str
    store: str | None
    product_title: str
    average_rating: float
    aspect_id: str
    aspect_name: str
    mention_count: int
    negative_count: int
    negative_share: float


class AspectReasonRanking(ApiRecord):
    aspect_id: str
    aspect_name: str
    mention_count: int
    sentiment_count: int
    sentiment_share: float


class ProductComparisonInsightsResponse(ApiRecord):
    workspace_version: str
    comparison_product_count: int
    comparison_niche_id: str | None
    aspect_rankings: list[AspectProductRanking]
    repurchase_products: list[RepurchaseProductRanking]
    high_rating_recurring_complaints: list[RecurringComplaintRanking]
    seller_top_praise_reasons: list[AspectReasonRanking]
    seller_top_criticism_reasons: list[AspectReasonRanking]


class SellerReportResponse(ApiRecord):
    workspace_id: str
    workspace_version: str
    dataset_version: str
    niche_id: str
    niche_name: str
    review_date_start: str
    review_date_end: str
    products: list[ProductSummary]
    complaints: list[AspectInsight]
    weaknesses: list[AspectInsight]
    strengths: list[AspectInsight]
    limitations: list[str]


class EvidenceExample(ApiRecord):
    review_id: str
    rating: float | None
    review_timestamp: str | None
    review_text: str | None
    evidence_phrases: list[str]


class AspectEvidenceResponse(ApiRecord):
    workspace_version: str
    aspect_id: str
    aspect_name: str
    negative_examples: list[EvidenceExample]
    positive_examples: list[EvidenceExample]


class EvidenceSearchItem(ApiRecord):
    review_id: str
    parent_asin: str
    product_title: str
    store: str | None
    rating: float
    review_timestamp: str
    review_text: str
    aspect_id: str
    aspect_sentiment: Literal["negative", "neutral", "positive"]
    aspect_probability: float
    evidence_phrases: list[str]
    helpful_vote: int
    lexical_score: float
    dense_score: float
    retrieval_score: float
    retrieval_method: Literal["lexical_tfidf_v1", "hybrid_tfidf_lsa_v1"]


class EvidenceSearchResponse(ApiRecord):
    workspace_version: str
    query: str
    scope: Literal["seller", "all"]
    aspect_id: str | None
    sentiment: Literal["negative", "neutral", "positive"] | None
    rating_band: Literal["negative", "neutral", "positive"] | None
    items: list[EvidenceSearchItem]


class AspectTrendPoint(ApiRecord):
    year_month: str
    mention_count: int
    needs_attention_share: float
    positive_share: float
    negative_share: float


class AspectTrendResponse(ApiRecord):
    workspace_version: str
    aspect_id: str
    aspect_name: str
    items: list[AspectTrendPoint]


class RagCitation(ApiRecord):
    citation_id: str
    review_id: str
    parent_asin: str
    store: str | None
    rating: float
    review_timestamp: str
    review_text: str
    aspect_id: str
    aspect_sentiment: Literal["negative", "neutral", "positive"]


class RagContextResponse(ApiRecord):
    question: str
    instructions: str
    context: str
    citations: list[RagCitation]
    limitations: list[str]


class AskAnalystRequest(ApiRecord):
    question: str = Field(min_length=2, max_length=500)
    aspect_id: str | None = Field(default=None, max_length=80)
    evidence_query: str | None = Field(default=None, min_length=2, max_length=200)
    sentiment: Literal["negative", "neutral", "positive"] | None = None
    scope: Literal["seller", "all"] = "all"
    maximum_evidence_reviews: int = Field(default=8, ge=1, le=8)
    external_processing_consent: Literal[True]


class AnalystFindingResponse(ApiRecord):
    finding: str
    basis: Literal["aggregate", "reviews", "both"]
    citations: list[str]


class AnalystMetadata(ApiRecord):
    analyst_version: str
    model_version: str | None
    response_id: str | None
    usage_metadata: dict[str, object]
    available_citations: list[str]
    used_citations: list[str]


class AnalystSourceResponse(ApiRecord):
    citation_id: str
    brand: str | None
    rating: float
    review_date: str
    review_text: str
    aspect_id: str
    sentiment: Literal["negative", "neutral", "positive"]


class AskAnalystResponse(ApiRecord):
    workspace_version: str
    question: str
    resolved_aspect_id: str | None
    evidence_query: str
    evidence_review_count: int
    answer: str
    answer_ru: str = Field(
        description="Deprecated compatibility field; use answer."
    )
    findings: list[AnalystFindingResponse]
    recommendations: list[str]
    limitations: list[str]
    sources: list[AnalystSourceResponse]
    metadata: AnalystMetadata
