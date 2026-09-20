"""FastAPI entry point for the Review Intelligence seller MVP."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

from src.api.models import (
    AskAnalystRequest,
    AskAnalystResponse,
    AspectEvidenceResponse,
    AspectTrendResponse,
    CreateWorkspaceRequest,
    EvidenceSearchResponse,
    HealthResponse,
    ProductComparisonInsightsResponse,
    ProductSearchResponse,
    RagContextResponse,
    SellerReportResponse,
    WorkspaceListResponse,
)
from src.api.service import SellerAnalyticsStore
from src.common.project import find_project_root
from src.llm.gemini_client import GeminiConfigurationError


def create_app(project_root: Path | None = None) -> FastAPI:
    """Create an API bound to one immutable local artifact root."""
    root = (project_root or find_project_root(Path(__file__).parent)).resolve()
    store = SellerAnalyticsStore(root)
    application = FastAPI(
        title="Review Intelligence API",
        version="0.1.0",
        description="Evidence-backed historical seller analytics.",
    )
    application.state.analytics_store = store
    analyst_slots = threading.BoundedSemaphore(value=2)

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        page = root / "src" / "api" / "dashboard.html"
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @application.get("/api/health", response_model=HealthResponse)
    async def health() -> dict[str, object]:
        try:
            niche = store._analysis_niche()
            workspace_count = len(store.list_workspaces())
            return {
                "status": "ok",
                "service": "review-intelligence",
                "active_niche": niche.niche_version,
                "workspace_count": workspace_count,
            }
        except (FileNotFoundError, ValueError):
            return {
                "status": "degraded",
                "service": "review-intelligence",
                "active_niche": None,
                "workspace_count": 0,
            }

    @application.get("/api/v1/workspaces", response_model=WorkspaceListResponse)
    async def workspaces(
        request: Request,
        include_history: bool = Query(default=False),
    ) -> dict[str, object]:
        analytics: SellerAnalyticsStore = request.app.state.analytics_store
        return {"items": analytics.list_workspaces(include_history=include_history)}

    @application.post(
        "/api/v1/workspaces",
        response_model=SellerReportResponse,
        status_code=201,
    )
    async def create_workspace(
        payload: CreateWorkspaceRequest,
        request: Request,
    ) -> dict[str, object]:
        analytics: SellerAnalyticsStore = request.app.state.analytics_store
        try:
            return analytics.create_workspace(**payload.model_dump())
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get("/api/v1/products/search", response_model=ProductSearchResponse)
    async def product_search(
        request: Request,
        q: str = Query(min_length=2, max_length=120),
        limit: int = Query(default=20, ge=1, le=50),
    ) -> dict[str, object]:
        analytics: SellerAnalyticsStore = request.app.state.analytics_store
        try:
            return analytics.search_products(q, limit=limit)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get(
        "/api/v1/workspaces/{workspace_version}/report",
        response_model=SellerReportResponse,
    )
    async def report(
        workspace_version: str, request: Request
    ) -> dict[str, object]:
        analytics: SellerAnalyticsStore = request.app.state.analytics_store
        try:
            return analytics.report(workspace_version)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.get(
        "/api/v1/workspaces/{workspace_version}/comparison-insights",
        response_model=ProductComparisonInsightsResponse,
    )
    async def comparison_insights(
        workspace_version: str, request: Request
    ) -> dict[str, object]:
        analytics: SellerAnalyticsStore = request.app.state.analytics_store
        try:
            return analytics.comparison_insights(workspace_version)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.get(
        "/api/v1/workspaces/{workspace_version}/aspects/{aspect_id}/evidence",
        response_model=AspectEvidenceResponse,
    )
    async def evidence(
        workspace_version: str,
        aspect_id: str,
        request: Request,
    ) -> dict[str, object]:
        analytics: SellerAnalyticsStore = request.app.state.analytics_store
        try:
            return analytics.evidence(workspace_version, aspect_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.get(
        "/api/v1/workspaces/{workspace_version}/evidence/search",
        response_model=EvidenceSearchResponse,
    )
    async def evidence_search(
        workspace_version: str,
        request: Request,
        q: str = Query(min_length=2, max_length=200),
        scope: Literal["seller", "all"] = Query(default="all"),
        aspect_id: str | None = Query(default=None),
        sentiment: Literal["negative", "neutral", "positive"] | None = Query(
            default=None
        ),
        rating_band: Literal["negative", "neutral", "positive"] | None = Query(
            default=None
        ),
        limit: int = Query(default=8, ge=1, le=25),
    ) -> dict[str, object]:
        analytics: SellerAnalyticsStore = request.app.state.analytics_store
        try:
            return analytics.search_evidence(
                workspace_version,
                query=q,
                scope=scope,
                aspect_id=aspect_id,
                sentiment=sentiment,
                rating_band=rating_band,
                limit=limit,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.get(
        "/api/v1/workspaces/{workspace_version}/aspects/{aspect_id}/trend",
        response_model=AspectTrendResponse,
    )
    async def aspect_trend(
        workspace_version: str,
        aspect_id: str,
        request: Request,
    ) -> dict[str, object]:
        analytics: SellerAnalyticsStore = request.app.state.analytics_store
        try:
            return analytics.aspect_trend(workspace_version, aspect_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.get(
        "/api/v1/workspaces/{workspace_version}/rag/context",
        response_model=RagContextResponse,
    )
    async def rag_context(
        workspace_version: str,
        request: Request,
        question: str = Query(min_length=2, max_length=500),
        evidence_query: str = Query(min_length=2, max_length=200),
        scope: Literal["seller", "all"] = Query(default="all"),
        aspect_id: str | None = Query(default=None),
        sentiment: Literal["negative", "neutral", "positive"] | None = Query(
            default=None
        ),
        limit: int = Query(default=8, ge=1, le=25),
    ) -> dict[str, object]:
        analytics: SellerAnalyticsStore = request.app.state.analytics_store
        try:
            return analytics.rag_context(
                workspace_version,
                question=question,
                evidence_query=evidence_query,
                aspect_id=aspect_id,
                sentiment=sentiment,
                scope=scope,
                limit=limit,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.post(
        "/api/v1/workspaces/{workspace_version}/ask",
        response_model=AskAnalystResponse,
    )
    async def ask_analyst(
        workspace_version: str,
        payload: AskAnalystRequest,
        request: Request,
    ) -> dict[str, object]:
        analytics: SellerAnalyticsStore = request.app.state.analytics_store
        if not analyst_slots.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="AI analyst is busy; try again in a moment",
            )
        try:
            return analytics.ask_analyst(
                workspace_version,
                **payload.model_dump(),
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except GeminiConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=502,
                detail="Gemini temporarily failed to return a valid answer",
            ) from error
        finally:
            analyst_slots.release()

    @application.get("/api/v1/workspaces/{workspace_version}/export.xlsx")
    async def export_workbook(
        workspace_version: str,
        request: Request,
    ) -> Response:
        analytics: SellerAnalyticsStore = request.app.state.analytics_store
        try:
            workbook = analytics.workbook_path(workspace_version)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return Response(
            content=workbook.read_bytes(),
            media_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{workspace_version}.xlsx"'
                )
            },
        )

    return application


app = create_app()
