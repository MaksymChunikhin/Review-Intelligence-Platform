"""Read versioned seller analytics artifacts for API delivery."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import duckdb
import pyarrow.parquet as pq

from src.analytics.competitor_selection import select_direct_competitors
from src.analytics.product_comparison_insights import (
    build_product_comparison_insights,
    build_global_niche_comparison_insights,
)
from src.analytics.seller_comparison import build_seller_comparison
from src.embeddings.lsa_index import LsaEvidenceIndex
from src.llm.grounded_analyst import generate_grounded_answer
from src.ingestion.dataset_manifest import sha256_file
from src.rag.evidence_retrieval import retrieve_review_evidence
from src.rag.context import build_rag_context
from src.rag.query_understanding import (
    infer_analytics_intent,
    infer_evidence_sentiment,
    understand_evidence_query,
)
from src.schemas.analyst import load_analyst_config
from src.schemas.workspace import (
    ProductNicheDefinition,
    SellerWorkspaceScope,
    load_product_niche_definition,
    load_seller_workspace_scope,
)


InsightKind = Literal["complaints", "weaknesses", "strengths"]
WORKSPACE_PIPELINE_VERSION = "seller_workspace_v4"
WORKSPACE_REVISION = 4
MINIMUM_COMPARISON_GAP_PP = 5.0

LEGACY_CONDITIONER_ASPECT_IDS = {
    "price_affordability": "value_for_money",
    "packaging_integrity": "packaging",
    "overall_performance": "performance",
    "volume_body": "volume",
    "curl_definition": "curls",
    "lather_foam": "lather",
    "cleansing": "cleaning",
    "result_longevity": "longevity",
    "rinsability": "rinsing",
    "product_size_quantity": "product_size",
    "application_usability": "application_and_usability",
    "hair_strength": "hair_strength_breakage",
    "hair_growth_loss": "hair_growth_loss_shedding",
    "color_toning_deposit": "color_toning",
    "scalp_reactions": "skin_sensitivity_reactions",
}


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def _read_evidence(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [str(item["text"]) for item in json.loads(str(value))]


def _read_parquet_rows(
    path: Path,
    *,
    columns: list[str],
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    """Project and filter large Parquet files with bounded DuckDB memory."""
    names = [*columns, *(item[0] for item in (filters or []))]
    if any(not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name) for name in names):
        raise ValueError("invalid Parquet column name")
    projection = ", ".join(f'"{name}"' for name in columns)
    clauses: list[str] = []
    parameters: list[Any] = [str(path)]
    for column, operator, value in filters or []:
        if operator == "in":
            if not value:
                clauses.append("FALSE")
                continue
            clauses.append(f'"{column}" IN (SELECT UNNEST(?))')
        elif operator == "=":
            clauses.append(f'"{column}" = ?')
        else:
            raise ValueError(f"unsupported Parquet filter operator: {operator}")
        parameters.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit='256MB'")
        connection.execute("SET threads=2")
        return connection.execute(
            f"SELECT {projection} FROM read_parquet(?){where}", parameters
        ).fetchdf()
    finally:
        connection.close()


@dataclass(frozen=True)
class WorkspaceArtifacts:
    scope: SellerWorkspaceScope
    scope_path: Path
    report_metadata: dict[str, Any]
    comparison_path: Path
    reviews_path: Path
    product_mart_path: Path
    aspect_sentiment_path: Path


class SellerAnalyticsStore:
    """Resolve approved local artifacts without recomputing model output."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._retrieval_indexes: dict[Path, LsaEvidenceIndex] = {}
        self._workspace_creation_lock = threading.Lock()
        self._aspect_names: dict[str, str] = {}
        for taxonomy_path in (
            self.project_root
            / "config"
            / "aspects"
            / "hair_conditioners_taxonomy_v2_reviewed.json",
            self.project_root
            / "reports"
            / "aspects"
            / "shampoo_and_conditioner_taxonomy_v1.json",
        ):
            if taxonomy_path.is_file():
                taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
                self._aspect_names.update(
                    {
                        str(item["aspect_id"]): str(item["canonical_name"])
                        for item in taxonomy["aspects"]
                    }
                )

    def _analysis_niche(self) -> ProductNicheDefinition:
        """Prefer the complete section when its population artifacts exist."""
        for name in ("shampoo_and_conditioner_v1.json", "hair_conditioners_v1.json"):
            path = self.project_root / "config" / "niches" / name
            if not path.is_file():
                continue
            niche = load_product_niche_definition(path)
            niche_root = (
                self.project_root
                / "data"
                / "processed"
                / niche.dataset_version
                / "niches"
                / niche.niche_version
            )
            required = [
                niche_root / "reviews.parquet",
                niche_root / "product_aspect_mart_v1.parquet",
            ]
            if niche.segments:
                required.append(niche_root / "niche_product_rankings_v1.parquet")
            sentiment_files = list(niche_root.glob("aspect_sentiment_*.parquet"))
            if all(path.is_file() for path in required) and len(sentiment_files) == 1:
                return niche
        raise FileNotFoundError("no materialized analysis niche is available")

    def _aspect_name(self, aspect_id: str, fallback: str | None = None) -> str:
        return self._aspect_names.get(aspect_id, fallback or aspect_id)

    @staticmethod
    def _artifact_aspect_id(
        artifacts: WorkspaceArtifacts, aspect_id: str | None
    ) -> str | None:
        if aspect_id is None:
            return None
        if artifacts.scope.niche.niche_version == "shampoo_and_conditioner_v1":
            return aspect_id
        return LEGACY_CONDITIONER_ASPECT_IDS.get(aspect_id, aspect_id)

    def _relative_path(self, value: str) -> Path:
        resolved = (self.project_root / value).resolve()
        if self.project_root not in resolved.parents:
            raise ValueError("artifact path escapes project root")
        return resolved

    def _workspace_files(self) -> list[Path]:
        return sorted((self.project_root / "config" / "workspaces").glob("*.json"))

    def list_workspaces(self, *, include_history: bool = False) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self._workspace_files():
            scope = load_seller_workspace_scope(path)
            report_path = (
                self.project_root
                / "reports"
                / "seller_demo"
                / f"{scope.workspace_version}.json"
            )
            if not report_path.is_file():
                continue
            records.append(
                {
                    "workspace_id": scope.workspace_id,
                    "workspace_version": scope.workspace_version,
                    "niche_id": scope.niche.niche_id,
                    "niche_name": scope.niche.display_name,
                    "seller_parent_asins": scope.seller_parent_asins,
                    "competitor_parent_asins": scope.competitor_parent_asins,
                    "review_date_start": scope.review_date_start.isoformat(),
                    "review_date_end": scope.review_date_end.isoformat(),
                }
            )
        records.sort(
            key=lambda item: (
                item["niche_id"] != "shampoo_and_conditioner",
                item["workspace_id"],
                item["workspace_version"],
            )
        )
        if include_history:
            return records
        latest: dict[str, dict[str, Any]] = {}
        for record in records:
            latest[record["workspace_id"]] = record
        return list(latest.values())

    def resolve(self, workspace_version: str) -> WorkspaceArtifacts:
        matches: list[tuple[Path, SellerWorkspaceScope]] = []
        for path in self._workspace_files():
            scope = load_seller_workspace_scope(path)
            if scope.workspace_version == workspace_version:
                matches.append((path, scope))
        if len(matches) != 1:
            raise FileNotFoundError(f"unknown workspace version: {workspace_version}")
        scope_path, scope = matches[0]
        metadata_path = (
            self.project_root
            / "reports"
            / "seller_demo"
            / f"{workspace_version}.json"
        )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"report is absent for workspace: {workspace_version}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparison_path = self._relative_path(str(metadata["comparison_path"]))
        expected_comparison_hash = metadata.get("comparison_sha256")
        if expected_comparison_hash and sha256_file(comparison_path) != str(
            expected_comparison_hash
        ):
            raise ValueError(f"artifact integrity check failed: {comparison_path}")
        base = self.project_root / "data" / "processed" / scope.dataset_version
        reviews_path = base / "niches" / scope.niche.niche_version / "reviews.parquet"
        product_mart_path = (
            base
            / "niches"
            / scope.niche.niche_version
            / "product_aspect_mart_v1.parquet"
        )
        sentiment_files = sorted(
            (base / "niches" / scope.niche.niche_version).glob(
                "aspect_sentiment_*.parquet"
            )
        )
        if len(sentiment_files) != 1:
            raise FileNotFoundError(
                "expected exactly one versioned aspect-sentiment artifact for "
                f"{scope.niche.niche_version}"
            )
        aspect_sentiment_path = sentiment_files[0]
        for required in (
            comparison_path,
            reviews_path,
            product_mart_path,
            aspect_sentiment_path,
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        return WorkspaceArtifacts(
            scope=scope,
            scope_path=scope_path,
            report_metadata=metadata,
            comparison_path=comparison_path,
            reviews_path=reviews_path,
            product_mart_path=product_mart_path,
            aspect_sentiment_path=aspect_sentiment_path,
        )

    def search_products(self, query: str, *, limit: int = 20) -> dict[str, Any]:
        """Search reviewed products in the broadest materialized approved scope."""
        normalized_query = " ".join(query.lower().split())
        searchable_query = " ".join(
            re.findall(r"[a-z0-9]+", normalized_query, flags=re.IGNORECASE)
        )
        if len(searchable_query) < 2:
            raise ValueError("query must contain at least two letters or digits")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        niche = self._analysis_niche()
        catalog_path = (
            self.project_root
            / "data"
            / "processed"
            / niche.dataset_version
            / "product_catalog.parquet"
        )
        terms = [term for term in searchable_query.split(" ") if len(term) >= 2][:8]
        where_terms = " AND ".join(
            ["contains(lower(product_search_text), ?)" for _ in terms]
        )
        sql = f"""
            SELECT
                parent_asin,
                category_path_id,
                product_title,
                store,
                price_at_collection,
                average_rating,
                rating_number,
                review_count
            FROM read_parquet(?)
            WHERE category_path_id IN (SELECT UNNEST(?))
              AND review_count > 0
              AND (
                  lower(parent_asin) = ?
                  OR ({where_terms})
              )
            ORDER BY
                lower(parent_asin) = ? DESC,
                contains(lower(product_title), ?) DESC,
                review_count DESC,
                parent_asin
            LIMIT ?
        """
        parameters: list[Any] = [
            str(catalog_path),
            niche.category_path_ids,
            searchable_query,
            *terms,
            searchable_query,
            searchable_query,
            limit,
        ]
        connection = duckdb.connect()
        try:
            connection.execute("SET memory_limit='256MB'")
            connection.execute("SET threads=2")
            rows = connection.execute(sql, parameters).fetchdf()
        finally:
            connection.close()
        segment_by_path = {
            item["category_path_id"]: item for item in niche.segment_records()
        }
        items = []
        for record in rows.to_dict("records"):
            cleaned = {key: _clean_scalar(value) for key, value in record.items()}
            segment = segment_by_path.get(str(cleaned.pop("category_path_id")), {})
            cleaned.update(
                {
                    "analytical_family_name": segment.get("analytical_family_name"),
                    "competitor_niche_name": segment.get("competitor_niche_name"),
                }
            )
            items.append(cleaned)
        return {
            "query": query,
            "niche_id": niche.niche_id,
            "items": items,
        }

    def create_workspace(
        self,
        *,
        seller_parent_asin: str,
        top_k: int = 4,
        minimum_review_count: int = 250,
    ) -> dict[str, Any]:
        """Create one idempotent exact-niche product analysis."""
        if not re.fullmatch(r"[A-Z0-9]{10}", seller_parent_asin):
            raise ValueError("seller_parent_asin must be a 10-character Amazon ID")
        niche = self._analysis_niche()
        existing = [
            (path, load_seller_workspace_scope(path))
            for path in self._workspace_files()
        ]
        current = [
            scope
            for _, scope in existing
            if scope.seller_parent_asins == [seller_parent_asin]
            and scope.niche.niche_version == niche.niche_version
            and scope.pipeline_version == WORKSPACE_PIPELINE_VERSION
            and scope.requested_top_k == top_k
            and scope.requested_minimum_review_count == minimum_review_count
        ]
        if current:
            return self.report(current[-1].workspace_version)

        with self._workspace_creation_lock:
            return self._create_workspace_locked(
                niche=niche,
                seller_parent_asin=seller_parent_asin,
                top_k=top_k,
                minimum_review_count=minimum_review_count,
            )

    def _create_workspace_locked(
        self,
        *,
        niche: ProductNicheDefinition,
        seller_parent_asin: str,
        top_k: int,
        minimum_review_count: int,
    ) -> dict[str, Any]:
        """Build a workspace while serializing artifact publication."""
        # Recheck after acquiring the lock in case another request created it.
        for path in self._workspace_files():
            scope = load_seller_workspace_scope(path)
            if (
                scope.seller_parent_asins == [seller_parent_asin]
                and scope.niche.niche_version == niche.niche_version
                and scope.pipeline_version == WORKSPACE_PIPELINE_VERSION
                and scope.requested_top_k == top_k
                and scope.requested_minimum_review_count == minimum_review_count
            ):
                return self.report(scope.workspace_version)
        base = self.project_root / "data" / "processed" / niche.dataset_version
        catalog_path = base / "product_catalog.parquet"
        seller_scope = _read_parquet_rows(
            catalog_path,
            columns=["parent_asin", "category_path_id"],
            filters=[("parent_asin", "=", seller_parent_asin)],
        )
        seller_scope = seller_scope[
            seller_scope["category_path_id"].isin(niche.category_path_ids)
        ]
        if len(seller_scope) != 1:
            raise ValueError("product is not in the supported Shampoo & Conditioner section")
        category_path_id = str(seller_scope.iloc[0]["category_path_id"])
        segment = next(
            item
            for item in niche.segment_records()
            if item["category_path_id"] == category_path_id
        )
        if segment["comparison_status"] != "eligible":
            raise ValueError("product type is not yet eligible for competitor comparison")
        workspace_id = f"{niche.niche_id}_{seller_parent_asin.lower()}"
        parameter_suffix = (
            "" if (top_k, minimum_review_count) == (4, 250)
            else f"_k{top_k}_r{minimum_review_count}"
        )
        workspace_version = f"{workspace_id}_v{WORKSPACE_REVISION}{parameter_suffix}"
        workspace_data = base / "workspaces" / workspace_version
        selection_report_path = (
            self.project_root
            / "reports"
            / "seller_demo"
            / f"{workspace_version}_competitor_selection.json"
        )
        selection_parameters = {
            "seller_parent_asin": seller_parent_asin,
            "category_path_ids": [category_path_id],
            "selection_version": f"{workspace_version}_direct_competitors_v1",
            "top_k": top_k,
            "selection_profile": (
                "catalog_path" if niche.segments else "conditioner"
            ),
            "report_path": selection_report_path,
        }
        actual_minimum_review_count = minimum_review_count
        try:
            selection = select_direct_competitors(
                catalog_path,
                workspace_data / "competitor_ranking.parquet",
                minimum_review_count=minimum_review_count,
                **selection_parameters,
            )
        except ValueError as error:
            if "eligible competitors" not in str(error) or minimum_review_count <= 20:
                raise
            actual_minimum_review_count = 20
            selection = select_direct_competitors(
                catalog_path,
                workspace_data / "competitor_ranking.parquet",
                minimum_review_count=20,
                **selection_parameters,
            )
        workspace_payload = {
            "workspace_id": workspace_id,
            "workspace_version": workspace_version,
            "dataset_version": niche.dataset_version,
            "niche": niche.model_dump(mode="json"),
            "seller_parent_asins": [seller_parent_asin],
            "competitor_parent_asins": selection.selected_parent_asins,
            "review_date_start": "2021-01-01",
            "review_date_end": "2023-09-12",
            "verified_purchase_only": False,
            "pipeline_version": WORKSPACE_PIPELINE_VERSION,
            "requested_top_k": top_k,
            "requested_minimum_review_count": minimum_review_count,
            "actual_minimum_review_count": actual_minimum_review_count,
        }
        config_path = (
            self.project_root / "config" / "workspaces" / f"{workspace_version}.json"
        )
        temporary_config = config_path.with_name(f".{config_path.name}.tmp")
        temporary_config.write_text(
            json.dumps(workspace_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_base = self.project_root / "reports" / "seller_demo" / workspace_version
        try:
            build_seller_comparison(
                temporary_config,
                base / "niches" / niche.niche_version / "reviews.parquet",
                base
                / "niches"
                / niche.niche_version
                / "product_aspect_mart_v1.parquet",
                workspace_data / "aspect_comparison.parquet",
                report_base.with_suffix(".xlsx"),
                report_base.with_suffix(".md"),
                report_version="seller_comparison_v4",
                minimum_aspect_support=20,
                minimum_comparison_gap_pp=MINIMUM_COMPARISON_GAP_PP,
                report_path=report_base.with_suffix(".json"),
            )
            temporary_config.replace(config_path)
        finally:
            temporary_config.unlink(missing_ok=True)
        return self.report(workspace_version)

    @staticmethod
    def _insights(comparison: pd.DataFrame, kind: InsightKind) -> pd.DataFrame:
        supported = comparison[comparison["support_sufficient"]].copy()
        if kind == "complaints":
            return supported.sort_values(
                ["seller_needs_attention_count", "attention_gap_pp"],
                ascending=False,
            ).head(5)
        if kind == "weaknesses":
            return supported[
                supported["attention_gap_pp"] >= MINIMUM_COMPARISON_GAP_PP
            ].sort_values(
                ["attention_gap_pp", "seller_needs_attention_count"],
                ascending=False,
            ).head(5)
        return supported[
            (supported["seller_positive_share"] >= 0.55)
            & (supported["positive_gap_pp"] >= MINIMUM_COMPARISON_GAP_PP)
            & (supported["attention_gap_pp"] <= -MINIMUM_COMPARISON_GAP_PP)
        ].sort_values(
            ["advantage_score_pp", "seller_positive_count"], ascending=False
        ).head(5)

    def _product_overview(self, artifacts: WorkspaceArtifacts) -> list[dict[str, Any]]:
        scope = artifacts.scope
        selected_ids = scope.seller_parent_asins + scope.competitor_parent_asins
        reviews = _read_parquet_rows(
            artifacts.reviews_path,
            columns=[
                "review_id",
                "parent_asin",
                "rating",
                "product_title",
                "store",
            ],
            filters=[("parent_asin", "in", selected_ids)],
        )
        reviews["needs_attention"] = reviews["rating"] <= 3
        overview = reviews.groupby("parent_asin", sort=False).agg(
            product_title=("product_title", "first"),
            store=("store", "first"),
            review_count=("review_id", "size"),
            average_rating=("rating", "mean"),
            needs_attention_share=("needs_attention", "mean"),
        ).reset_index()
        overview["role"] = overview["parent_asin"].map(
            lambda value: "seller" if value in scope.seller_parent_asins else "competitor"
        )
        order = {value: index for index, value in enumerate(selected_ids)}
        overview["order"] = overview["parent_asin"].map(order)
        overview = overview.sort_values("order").drop(columns="order")
        return [
            {key: _clean_scalar(value) for key, value in record.items()}
            for record in overview.to_dict("records")
        ]

    def report(self, workspace_version: str) -> dict[str, Any]:
        artifacts = self.resolve(workspace_version)
        comparison = pd.read_parquet(artifacts.comparison_path)
        mart_columns = set(pq.read_schema(artifacts.product_mart_path).names)
        if "comparative_analytics_allowed" in mart_columns:
            mart_schema = _read_parquet_rows(
                artifacts.product_mart_path,
                columns=["aspect_id", "comparative_analytics_allowed"],
                filters=[
                    ("parent_asin", "in", artifacts.scope.seller_parent_asins)
                ],
            )
            allowed_aspects = set(
                mart_schema.loc[
                    mart_schema["comparative_analytics_allowed"].astype(bool),
                    "aspect_id",
                ].astype(str)
            )
            comparison = comparison[
                comparison["aspect_id"].astype(str).isin(allowed_aspects)
            ].copy()
        insight_payload: dict[str, list[dict[str, Any]]] = {}
        for kind in ("complaints", "weaknesses", "strengths"):
            rows = self._insights(comparison, kind)
            payload: list[dict[str, Any]] = []
            for row in rows.itertuples(index=False):
                aspect_id = str(row.aspect_id)
                name = self._aspect_name(aspect_id, str(row.canonical_name))
                payload.append(
                    {
                        "aspect_id": aspect_id,
                        "aspect_name": name,
                        "mention_count": int(row.seller_aspect_review_count),
                        "seller_attention_share": _clean_scalar(
                            row.seller_needs_attention_share
                        ),
                        "competitor_attention_share": _clean_scalar(
                            row.competitor_needs_attention_share
                        ),
                        "attention_gap_pp": _clean_scalar(row.attention_gap_pp),
                        "seller_positive_share": _clean_scalar(
                            row.seller_positive_share
                        ),
                        "competitor_positive_share": _clean_scalar(
                            row.competitor_positive_share
                        ),
                    }
                )
            insight_payload[kind] = payload
        scope = artifacts.scope
        return {
            "workspace_id": scope.workspace_id,
            "workspace_version": scope.workspace_version,
            "dataset_version": scope.dataset_version,
            "niche_id": scope.niche.niche_id,
            "niche_name": scope.niche.display_name,
            "review_date_start": scope.review_date_start.isoformat(),
            "review_date_end": scope.review_date_end.isoformat(),
            "products": self._product_overview(artifacts),
            **insight_payload,
            "limitations": [
                "Historical Amazon snapshot covering 2021–2023.",
                "Aspect detection and aspect sentiment are diagnostic, not final.",
                "Interpret findings together with the supporting review evidence.",
            ],
        }

    def comparison_insights(self, workspace_version: str) -> dict[str, Any]:
        """Return workspace facts plus whole-niche precomputed rankings."""
        artifacts = self.resolve(workspace_version)
        selected_ids = (
            artifacts.scope.seller_parent_asins
            + artifacts.scope.competitor_parent_asins
        )
        product_mart = _read_parquet_rows(
            artifacts.product_mart_path,
            columns=list(pq.read_schema(artifacts.product_mart_path).names),
            filters=[("parent_asin", "in", selected_ids)],
        )
        reviews = _read_parquet_rows(
            artifacts.reviews_path,
            columns=[
                "review_id",
                "parent_asin",
                "rating",
                "review_text",
                "product_title",
                "store",
            ],
            filters=[("parent_asin", "in", selected_ids)],
        )
        result = {
            "workspace_version": workspace_version,
            **build_product_comparison_insights(
                product_mart,
                reviews,
                seller_parent_asins=artifacts.scope.seller_parent_asins,
                competitor_parent_asins=artifacts.scope.competitor_parent_asins,
                aspect_names=self._aspect_names,
            ),
        }
        rankings_path = (
            artifacts.product_mart_path.parent / "niche_product_rankings_v1.parquet"
        )
        if rankings_path.is_file() and "competitor_niche_id" in product_mart.columns:
            seller_rows = product_mart[
                product_mart["parent_asin"].isin(
                    artifacts.scope.seller_parent_asins
                )
            ]
            niche_ids = seller_rows["competitor_niche_id"].dropna().astype(str).unique()
            if len(niche_ids) == 1:
                global_rankings = _read_parquet_rows(
                    rankings_path,
                    columns=list(pq.read_schema(rankings_path).names),
                    filters=[("competitor_niche_id", "=", niche_ids[0])],
                )
                global_result = build_global_niche_comparison_insights(
                    global_rankings,
                    competitor_niche_id=str(niche_ids[0]),
                    aspect_names=self._aspect_names,
                )
                result.update(global_result)
        result.setdefault("comparison_product_count", len(selected_ids))
        result.setdefault("comparison_niche_id", None)
        return result

    @staticmethod
    def _question_analytics_facts(
        insights: dict[str, Any],
        *,
        intent: str,
        question: str,
        aspect_id: str | None,
        sentiment: str | None,
    ) -> list[str]:
        """Select compact deterministic facts relevant to one question."""
        if intent == "repurchase_product_comparison":
            return [
                " | ".join(
                    [
                        f"rank={rank}",
                        f"brand={item['store']}",
                        f"product={item['product_title'][:160]}",
                        f"repurchase_review_count={item['signal_count']}",
                        f"all_product_reviews={item['review_count']}",
                        f"repurchase_share={item['signal_share']:.6f}",
                        f"average_rating={item['average_rating']:.3f}",
                        f"support_sufficient={item['support_sufficient']}",
                    ]
                )
                for rank, item in enumerate(
                    insights["repurchase_products"], start=1
                )
            ]
        if intent == "high_rating_recurring_complaints":
            return [
                " | ".join(
                    [
                        f"rank={rank}",
                        f"brand={item['store']}",
                        f"product={item['product_title'][:160]}",
                        f"average_rating={item['average_rating']:.3f}",
                        f"complaint_aspect={item['aspect_name']}",
                        f"negative_mentions={item['negative_count']}",
                        f"all_aspect_mentions={item['mention_count']}",
                        f"negative_share={item['negative_share']:.6f}",
                    ]
                )
                for rank, item in enumerate(
                    insights["high_rating_recurring_complaints"], start=1
                )
            ]
        if intent == "seller_top_reasons":
            key = (
                "seller_top_criticism_reasons"
                if sentiment == "negative"
                else "seller_top_praise_reasons"
            )
            label = "negative" if sentiment == "negative" else "positive"
            return [
                " | ".join(
                    [
                        f"rank={rank}",
                        f"aspect={item['aspect_name']}",
                        f"{label}_mentions={item['sentiment_count']}",
                        f"all_aspect_mentions={item['mention_count']}",
                        f"{label}_share={item['sentiment_share']:.6f}",
                    ]
                )
                for rank, item in enumerate(insights[key], start=1)
            ]

        aspect_ids = [aspect_id] if aspect_id else []
        normalized = question.lower()
        if any(term in normalized for term in ("greasy", "oily", "жирн")):
            aspect_ids.append("greasy_oily_finish")
        if any(term in normalized for term in ("heavy", "weigh", "утяж")):
            aspect_ids.append("hair_weight")
        aspect_ids = list(dict.fromkeys(value for value in aspect_ids if value))
        tone = "negative" if sentiment == "negative" else "positive"
        facts: list[str] = []
        for ranking in insights["aspect_rankings"]:
            if ranking["aspect_id"] not in aspect_ids:
                continue
            for rank, item in enumerate(
                ranking[f"{tone}_products"], start=1
            ):
                facts.append(
                    " | ".join(
                        [
                            f"aspect={ranking['aspect_name']}",
                            f"rank={rank}",
                            f"brand={item['store']}",
                            f"product={item['product_title'][:160]}",
                            f"{tone}_mentions={item['sentiment_count']}",
                            f"all_aspect_mentions={item['mention_count']}",
                            f"{tone}_share={item['sentiment_share']:.6f}",
                            f"all_product_reviews={item['product_review_count']}",
                            f"average_rating={item['average_rating']:.3f}",
                            f"support_sufficient={item['support_sufficient']}",
                        ]
                    )
                )
        return facts

    def evidence(self, workspace_version: str, aspect_id: str) -> dict[str, Any]:
        artifacts = self.resolve(workspace_version)
        aspect_id = self._artifact_aspect_id(artifacts, aspect_id) or aspect_id
        seller_ids = artifacts.scope.seller_parent_asins
        mart = _read_parquet_rows(
            artifacts.product_mart_path,
            columns=list(pq.read_schema(artifacts.product_mart_path).names),
            filters=[
                ("parent_asin", "in", seller_ids),
                ("aspect_id", "=", aspect_id),
            ],
        )
        if mart.empty:
            raise FileNotFoundError(f"aspect not found in seller scope: {aspect_id}")
        negative_ids = mart["top_negative_review_id"].dropna().astype(str).tolist()
        positive_ids = mart["top_positive_review_id"].dropna().astype(str).tolist()
        review_ids = list(dict.fromkeys([*negative_ids, *positive_ids]))
        reviews = _read_parquet_rows(
            artifacts.reviews_path,
            columns=["review_id", "rating", "review_timestamp", "review_text"],
            filters=[("review_id", "in", review_ids)],
        )
        review_by_id = reviews.set_index("review_id").to_dict("index")

        def examples(kind: Literal["negative", "positive"]) -> list[dict[str, Any]]:
            id_column = f"top_{kind}_review_id"
            evidence_column = f"top_{kind}_evidence_json"
            output: list[dict[str, Any]] = []
            for row in mart.itertuples(index=False):
                review_id = getattr(row, id_column)
                if pd.isna(review_id):
                    continue
                review = review_by_id.get(str(review_id), {})
                output.append(
                    {
                        "review_id": str(review_id),
                        "rating": _clean_scalar(review.get("rating")),
                        "review_timestamp": _clean_scalar(
                            review.get("review_timestamp")
                        ),
                        "review_text": _clean_scalar(review.get("review_text")),
                        "evidence_phrases": _read_evidence(
                            getattr(row, evidence_column)
                        ),
                    }
                )
            return output

        generic_name = self._aspect_name(aspect_id)
        return {
            "workspace_version": workspace_version,
            "aspect_id": aspect_id,
            "aspect_name": generic_name,
            "negative_examples": examples("negative"),
            "positive_examples": examples("positive"),
        }

    def search_evidence(
        self,
        workspace_version: str,
        *,
        query: str,
        scope: Literal["seller", "all"] = "all",
        aspect_id: str | None = None,
        sentiment: str | None = None,
        rating_band: Literal["negative", "neutral", "positive"] | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        """Search review evidence inside a fixed seller/competitor scope."""
        artifacts = self.resolve(workspace_version)
        aspect_id = self._artifact_aspect_id(artifacts, aspect_id)
        parent_asins = list(artifacts.scope.seller_parent_asins)
        if scope == "all":
            parent_asins.extend(artifacts.scope.competitor_parent_asins)
        reviews = _read_parquet_rows(
            artifacts.reviews_path,
            columns=[
                "review_id",
                "parent_asin",
                "rating",
                "review_timestamp",
                "review_text",
                "product_title",
                "store",
                "helpful_vote",
            ],
            filters=[("parent_asin", "in", parent_asins)],
        )
        if rating_band == "negative":
            reviews = reviews[reviews["rating"] <= 2].copy()
        elif rating_band == "neutral":
            reviews = reviews[reviews["rating"] == 3].copy()
        elif rating_band == "positive":
            reviews = reviews[reviews["rating"] >= 4].copy()
        if reviews.empty:
            return {
                "workspace_version": workspace_version,
                "query": query,
                "scope": scope,
                "aspect_id": aspect_id,
                "sentiment": sentiment,
                "rating_band": rating_band,
                "items": [],
            }
        review_ids = reviews["review_id"].astype(str).tolist()
        observation_filters: list[tuple[str, str, object]] = [
            ("review_id", "in", review_ids)
        ]
        if aspect_id is not None:
            observation_filters.append(("aspect_id", "=", aspect_id))
        if sentiment is not None:
            observation_filters.append(("aspect_sentiment", "=", sentiment))
        observations = _read_parquet_rows(
            artifacts.aspect_sentiment_path,
            columns=[
                "review_id",
                "aspect_id",
                "aspect_probability",
                "aspect_sentiment",
                "evidence_json",
            ],
            filters=observation_filters,
        )
        index_path = (
            artifacts.reviews_path.parent / "retrieval" / "tfidf_lsa_v1"
        )
        dense_scores: dict[str, float] | None = None
        if (index_path / "manifest.json").is_file():
            if index_path not in self._retrieval_indexes:
                self._retrieval_indexes[index_path] = LsaEvidenceIndex(index_path)
            dense_scores = self._retrieval_indexes[index_path].search_scores(
                query,
                allowed_review_ids=set(reviews["review_id"].astype(str)),
                top_k=max(200, limit * 20),
            )
        items = retrieve_review_evidence(
            reviews,
            observations,
            query=query,
            parent_asins=parent_asins,
            aspect_id=aspect_id,
            sentiment=sentiment,
            top_k=limit,
            max_per_product=limit if len(parent_asins) == 1 else 2,
            dense_scores=dense_scores,
        )
        return {
            "workspace_version": workspace_version,
            "query": query,
            "scope": scope,
            "aspect_id": aspect_id,
            "sentiment": sentiment,
            "rating_band": rating_band,
            "items": items,
        }

    def aspect_trend(
        self, workspace_version: str, aspect_id: str
    ) -> dict[str, Any]:
        """Calculate a seller-only monthly trend from review-level artifacts."""
        artifacts = self.resolve(workspace_version)
        aspect_id = self._artifact_aspect_id(artifacts, aspect_id) or aspect_id
        seller_ids = artifacts.scope.seller_parent_asins
        reviews = _read_parquet_rows(
            artifacts.reviews_path,
            columns=["review_id", "parent_asin", "rating", "review_timestamp"],
            filters=[("parent_asin", "in", seller_ids)],
        )
        observations = _read_parquet_rows(
            artifacts.aspect_sentiment_path,
            columns=["review_id", "aspect_id", "aspect_sentiment"],
            filters=[("aspect_id", "=", aspect_id)],
        )
        joined = observations.merge(
            reviews,
            on="review_id",
            how="inner",
            validate="many_to_one",
        )
        if joined.empty:
            raise FileNotFoundError(f"aspect not found in seller scope: {aspect_id}")
        joined["year_month"] = pd.to_datetime(
            joined["review_timestamp"], utc=True
        ).dt.strftime("%Y-%m")
        joined["needs_attention"] = joined["rating"] <= 3
        joined["positive"] = joined["aspect_sentiment"] == "positive"
        joined["negative"] = joined["aspect_sentiment"] == "negative"
        trend = joined.groupby("year_month", sort=True).agg(
            mention_count=("review_id", "size"),
            needs_attention_share=("needs_attention", "mean"),
            positive_share=("positive", "mean"),
            negative_share=("negative", "mean"),
        ).reset_index()
        return {
            "workspace_version": workspace_version,
            "aspect_id": aspect_id,
            "aspect_name": self._aspect_name(aspect_id),
            "items": [
                {key: _clean_scalar(value) for key, value in record.items()}
                for record in trend.to_dict("records")
            ],
        }

    def rag_context(
        self,
        workspace_version: str,
        *,
        question: str,
        evidence_query: str,
        aspect_id: str | None = None,
        sentiment: str | None = None,
        scope: Literal["seller", "all"] = "all",
        limit: int = 8,
    ) -> dict[str, Any]:
        """Prepare a grounded context without sending it to an external model."""
        report = self.report(workspace_version)
        analytics_intent = infer_analytics_intent(question)
        effective_scope = (
            "all"
            if analytics_intent
            in {
                "aspect_product_comparison",
                "repurchase_product_comparison",
                "high_rating_recurring_complaints",
            }
            else scope
        )
        retrieval = self.search_evidence(
            workspace_version,
            query=evidence_query,
            scope=effective_scope,
            aspect_id=aspect_id,
            sentiment=sentiment,
            limit=limit,
        )
        analytics_facts: list[str] = []
        if analytics_intent is not None:
            insights = self.comparison_insights(workspace_version)
            analytics_facts = self._question_analytics_facts(
                insights,
                intent=analytics_intent,
                question=question,
                aspect_id=aspect_id,
                sentiment=sentiment,
            )
        return build_rag_context(
            question=question,
            report=report,
            evidence=retrieval["items"],
            analytics_facts=analytics_facts,
        )

    def ask_analyst(
        self,
        workspace_version: str,
        *,
        question: str,
        aspect_id: str | None = None,
        evidence_query: str | None = None,
        sentiment: str | None = None,
        scope: Literal["seller", "all"] = "all",
        maximum_evidence_reviews: int = 8,
        external_processing_consent: bool,
        analyst_client: Any | None = None,
    ) -> dict[str, Any]:
        """Answer one question with bounded, consented Gemini processing."""
        if external_processing_consent is not True:
            raise ValueError("Consent is required before sending reviews to Gemini")
        config = load_analyst_config(
            self.project_root
            / "config"
            / "analyst"
            / "gemini_grounded_analyst_v1.json"
        )
        resolved_aspect_id, resolved_query = understand_evidence_query(
            question,
            aspect_id=aspect_id,
            evidence_query=evidence_query,
        )
        resolved_aspect_id = self._artifact_aspect_id(
            self.resolve(workspace_version), resolved_aspect_id
        )
        resolved_sentiment = sentiment or infer_evidence_sentiment(question)
        evidence_limit = min(
            maximum_evidence_reviews,
            config.maximum_evidence_reviews,
            8,
        )
        context_packet = self.rag_context(
            workspace_version,
            question=question,
            evidence_query=resolved_query,
            aspect_id=resolved_aspect_id,
            sentiment=resolved_sentiment,
            scope=scope,
            limit=evidence_limit,
        )
        answer, metadata = generate_grounded_answer(
            context_packet,
            config,
            client=analyst_client,
        )
        answer_payload = answer.model_dump(mode="json")
        return {
            "workspace_version": workspace_version,
            "question": question,
            "resolved_aspect_id": resolved_aspect_id,
            "evidence_query": resolved_query,
            "evidence_review_count": len(context_packet["citations"]),
            "answer": answer_payload["answer_ru"],
            **answer_payload,
            "sources": [
                {
                    "citation_id": item["citation_id"],
                    "brand": item["store"],
                    "rating": item["rating"],
                    "review_date": str(item["review_timestamp"])[:10],
                    "review_text": item["review_text"],
                    "aspect_id": item["aspect_id"],
                    "sentiment": item["aspect_sentiment"],
                }
                for item in context_packet["citations"]
            ],
            "metadata": metadata,
        }

    def workbook_path(self, workspace_version: str) -> Path:
        """Return the verified workbook artifact registered by a report."""
        artifacts = self.resolve(workspace_version)
        workbook = self._relative_path(
            str(artifacts.report_metadata["workbook_path"])
        )
        if not workbook.is_file():
            raise FileNotFoundError(workbook)
        expected = artifacts.report_metadata.get("workbook_sha256")
        if expected and sha256_file(workbook) != str(expected):
            raise ValueError(f"artifact integrity check failed: {workbook}")
        return workbook
