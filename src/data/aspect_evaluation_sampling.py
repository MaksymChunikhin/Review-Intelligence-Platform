"""Build a held-out review sample for aspect-extraction annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from src.common.csv_safety import escape_dataframe_for_spreadsheet
from src.common.project import find_project_root
from src.data.aspect_sampling import allocate_stratum_quotas
from src.ingestion.dataset_manifest import load_dataset_manifest, sha256_file
from src.schemas.aspects import (
    AspectEvaluationSampleConfig,
    load_aspect_evaluation_sample_config,
)
from src.schemas.workspace import (
    ProductNicheDefinition,
    load_product_niche_definition,
)


@dataclass(frozen=True)
class AspectEvaluationSampleReport:
    """Record lineage, holdout rules, and component coverage."""

    evaluation_version: str
    evaluation_schema_version: str
    taxonomy_version: str
    niche_id: str
    niche_version: str
    dataset_version: str
    representative_sampling_design: str
    representative_sampling_weight_semantics: str
    representative_population_review_count: int
    representative_product_cap_applied: bool
    targeted_sampling_design: str
    targeted_sampling_weight_semantics: str
    targeted_maximum_reviews_per_product: int
    taxonomy_identity_fields_validated: list[str]
    discovery_sample_identity_fields_validated: list[str]
    discovery_sample_membership_validated: bool
    taxonomy_path: str
    discovery_sample_path: str
    output_path: str
    output_sha256: str
    annotation_queue_path: str
    annotation_queue_sha256: str
    eligible_holdout_review_count: int
    excluded_discovery_review_count: int
    representative_review_count: int
    targeted_review_count: int
    total_review_count: int
    distinct_product_count: int
    maximum_reviews_per_product: int
    discovery_overlap_count: int
    component_overlap_count: int
    deterministic_seed: int
    minimum_words: int
    representative_strata: list[dict[str, Any]]
    targeted_aspect_coverage: list[dict[str, Any]]


def _artifact_reference(path: Path) -> str:
    """Prefer a repository-relative path in persisted provenance."""
    try:
        root = find_project_root(path.parent)
        return path.resolve().relative_to(root).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _validate_dataset_artifact(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    expected_dataset_version: str,
    identity_column: str,
    artifact_name: str,
) -> None:
    """Require one matching dataset identity and unique row identifiers."""
    row = connection.execute(
        f"""
        SELECT
            count(*) AS row_count,
            count(DISTINCT dataset_version) AS dataset_version_count,
            min(dataset_version) AS dataset_version,
            count(*) FILTER (WHERE dataset_version IS NULL)
                AS missing_dataset_version_rows,
            count(*) - count(DISTINCT {identity_column}) AS duplicate_or_null_ids
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchone()
    if int(row[0]) == 0:
        raise ValueError(f"{artifact_name} must not be empty")
    if (
        int(row[1]) != 1
        or row[2] != expected_dataset_version
        or int(row[3]) != 0
    ):
        raise ValueError(
            f"{artifact_name} dataset version does not match evaluation config"
        )
    if int(row[4]) != 0:
        raise ValueError(
            f"{artifact_name} {identity_column} values must be unique and non-null"
        )


def _validate_discovery_sample_identity(
    connection: duckdb.DuckDBPyConnection,
    discovery_source: Path,
    *,
    config: AspectEvaluationSampleConfig,
) -> list[str]:
    """Require and validate embedded discovery-sample identity."""
    columns = set(pq.ParquetFile(discovery_source).schema_arrow.names)
    if "review_id" not in columns:
        raise ValueError("discovery sample is missing review_id")
    expected = {
        "dataset_version": config.dataset_version,
        "niche_id": config.niche_id,
        "niche_version": config.niche_version,
        "sample_schema_version": config.discovery_sample_schema_version,
    }
    present = set(expected) & columns
    if present != set(expected):
        missing = sorted(set(expected) - columns)
        raise ValueError(
            "discovery sample is missing identity columns: "
            f"{missing}"
        )
    for field_name, expected_value in expected.items():
        row = connection.execute(
            f"""
            SELECT
                count(DISTINCT {field_name}),
                min({field_name}),
                count(*) FILTER (WHERE {field_name} IS NULL)
            FROM read_parquet(?)
            """,
            [str(discovery_source)],
        ).fetchone()
        if int(row[0]) != 1 or row[1] != expected_value or int(row[2]) != 0:
            raise ValueError(
                f"discovery sample {field_name} does not match evaluation config"
            )
    return sorted(expected)


def _stable_key(review_id: str, seed: int) -> str:
    """Return a stable ordering key for one review and seed."""
    value = f"{seed}:{review_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _alias_pattern(aliases: list[str]) -> str:
    """Build a case-insensitive phrase pattern with alphanumeric boundaries."""
    alternatives = []
    for alias in aliases:
        words = alias.casefold().split()
        if not words:
            continue
        alternatives.append(r"\s+".join(re.escape(word) for word in words))
    if not alternatives:
        raise ValueError("an aspect must have at least one non-empty alias")
    alternatives.sort(key=lambda value: (-len(value), value))
    return r"(?:^|[^a-z0-9])(?:" + "|".join(alternatives) + r")(?:$|[^a-z0-9])"


def select_targeted_reviews(
    matches: pd.DataFrame,
    *,
    target_size: int,
    minimum_per_aspect: int,
    maximum_reviews_per_product: int,
    deterministic_seed: int,
) -> tuple[list[str], dict[str, int]]:
    """Greedily balance heuristic aspect coverage under a product cap."""
    required = {"review_id", "parent_asin", "aspect_id"}
    if required - set(matches.columns):
        raise ValueError("target matches are missing required columns")
    if matches.empty:
        raise ValueError("target matches cannot be empty")
    matches = matches.drop_duplicates(list(required)).copy()
    review_aspects = (
        matches.groupby("review_id")["aspect_id"].agg(lambda values: set(values))
    ).to_dict()
    review_products = (
        matches.drop_duplicates("review_id")
        .set_index("review_id")["parent_asin"]
        .to_dict()
    )
    aspect_reviews = (
        matches.groupby("aspect_id")["review_id"].agg(lambda values: set(values))
    ).to_dict()
    if target_size > len(review_aspects):
        raise ValueError("target size exceeds reviews with alias matches")

    coverage = {aspect_id: 0 for aspect_id in sorted(aspect_reviews)}
    selected: list[str] = []
    selected_set: set[str] = set()
    product_counts: Counter[str] = Counter()

    def available(review_id: str) -> bool:
        product_id = str(review_products[review_id])
        return (
            review_id not in selected_set
            and product_counts[product_id] < maximum_reviews_per_product
        )

    while len(selected) < target_size:
        deficient = [
            aspect_id
            for aspect_id, count in coverage.items()
            if count < minimum_per_aspect
            and any(available(review_id) for review_id in aspect_reviews[aspect_id])
        ]
        if deficient:
            focus = min(
                deficient,
                key=lambda aspect_id: (
                    len(aspect_reviews[aspect_id]),
                    coverage[aspect_id],
                    aspect_id,
                ),
            )
            candidate_ids = [
                review_id
                for review_id in aspect_reviews[focus]
                if available(review_id)
            ]
        else:
            candidate_ids = [
                review_id for review_id in review_aspects if available(review_id)
            ]
        if not candidate_ids:
            raise RuntimeError("unable to fill targeted sample under product cap")

        def candidate_rank(review_id: str) -> tuple[float | str, ...]:
            aspects = review_aspects[review_id]
            unmet = sum(
                coverage[aspect_id] < minimum_per_aspect
                for aspect_id in aspects
            )
            balance = sum(1.0 / (coverage[aspect_id] + 1) for aspect_id in aspects)
            return (
                -float(unmet),
                -balance,
                -float(len(aspects)),
                _stable_key(review_id, deterministic_seed),
            )

        chosen = min(candidate_ids, key=candidate_rank)
        selected.append(chosen)
        selected_set.add(chosen)
        product_id = str(review_products[chosen])
        product_counts[product_id] += 1
        for aspect_id in review_aspects[chosen]:
            coverage[aspect_id] += 1

    undercovered = {
        aspect_id: count
        for aspect_id, count in coverage.items()
        if count < minimum_per_aspect
    }
    if undercovered:
        raise RuntimeError(
            f"targeted sample misses minimum aspect coverage: {undercovered}"
        )
    return selected, coverage


def _prepare_annotation_queue(
    sample: pd.DataFrame,
    destination: Path,
) -> pd.DataFrame:
    """Build a safe blind queue while preserving compatible human annotations."""
    queue = sample[
        [
            "annotation_order",
            "annotation_id",
            "review_id",
            "sampling_component",
            "product_title",
            "rating",
            "review_year",
            "review_text",
        ]
    ].copy()
    queue["human_aspect_ids"] = ""
    queue["human_evidence_json"] = ""
    queue["no_supported_aspect"] = ""
    queue["annotation_status"] = "pending"
    queue["annotator_notes"] = ""
    annotation_columns = [
        "human_aspect_ids",
        "human_evidence_json",
        "no_supported_aspect",
        "annotation_status",
        "annotator_notes",
    ]
    if destination.is_file():
        existing = pd.read_csv(destination, keep_default_na=False)
        completed = existing.get(
            "annotation_status", pd.Series(dtype=str)
        ).eq("complete")
        if completed.any():
            required_existing = {
                "review_id",
                "annotation_id",
                *annotation_columns,
            }
            missing_existing = required_existing - set(existing.columns)
            if missing_existing:
                raise ValueError(
                    "started annotation queue is missing columns: "
                    f"{sorted(missing_existing)}"
                )
            if set(existing["review_id"]) != set(queue["review_id"]):
                raise ValueError(
                    "cannot replace a started annotation queue with new reviews"
                )
            existing_by_review = existing.set_index("review_id")
            queue_by_review = queue.set_index("review_id")
            if not queue_by_review["annotation_id"].equals(
                existing_by_review.loc[
                    queue_by_review.index, "annotation_id"
                ]
            ):
                raise ValueError(
                    "cannot replace a started queue with new annotation IDs"
                )
            queue_by_review.loc[:, annotation_columns] = existing_by_review.loc[
                queue_by_review.index, annotation_columns
            ]
            queue = queue_by_review.reset_index()
    safe_source = escape_dataframe_for_spreadsheet(
        queue[["product_title", "review_text"]]
    )
    queue.loc[:, ["product_title", "review_text"]] = safe_source
    return queue


def _commit_staged_artifacts(
    staged_artifacts: list[tuple[Path, Path]],
) -> None:
    """Replace related artifacts together and restore originals on failure."""
    token = uuid4().hex
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for staged, destination in staged_artifacts:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = destination.with_name(
                    f".{destination.name}.{token}.backup"
                )
                destination.replace(backup)
                backups[destination] = backup
            staged.replace(destination)
            committed.append(destination)
    except BaseException:
        for destination in reversed(committed):
            destination.unlink(missing_ok=True)
        for destination, backup in reversed(list(backups.items())):
            if backup.exists():
                backup.replace(destination)
        for staged, _ in staged_artifacts:
            staged.unlink(missing_ok=True)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def build_aspect_evaluation_sample(
    config: AspectEvaluationSampleConfig,
    niche: ProductNicheDefinition,
    catalog_path: str | Path,
    reviews_path: str | Path,
    taxonomy_path: str | Path,
    discovery_sample_path: str | Path,
    output_path: str | Path,
    annotation_queue_path: str | Path,
    *,
    report_path: str | Path | None = None,
    memory_limit: str = "4GB",
) -> AspectEvaluationSampleReport:
    """Build representative and rare-aspect holdout components."""
    if niche.status != "approved":
        raise ValueError("niche must be approved before evaluation sampling")
    if niche.niche_id != config.niche_id:
        raise ValueError("niche ID does not match evaluation config")
    if niche.niche_version != config.niche_version:
        raise ValueError("niche version does not match evaluation config")
    if niche.dataset_version != config.dataset_version:
        raise ValueError("niche dataset version does not match evaluation config")
    catalog = Path(catalog_path)
    reviews = Path(reviews_path)
    taxonomy_source = Path(taxonomy_path)
    discovery_source = Path(discovery_sample_path)
    destination = Path(output_path)
    queue_destination = Path(annotation_queue_path)
    for source in (catalog, reviews, taxonomy_source, discovery_source):
        if not source.is_file():
            raise FileNotFoundError(source)

    taxonomy = json.loads(taxonomy_source.read_text(encoding="utf-8"))
    if taxonomy.get("status") != "approved":
        raise ValueError("evaluation sampling requires an approved taxonomy")
    taxonomy_identity = {
        "taxonomy_version": config.taxonomy_version,
        "niche_id": config.niche_id,
        "niche_version": config.niche_version,
        "dataset_version": config.dataset_version,
    }
    missing_taxonomy_identity = set(taxonomy_identity) - set(taxonomy)
    if missing_taxonomy_identity:
        raise ValueError(
            "taxonomy is missing identity fields: "
            f"{sorted(missing_taxonomy_identity)}"
        )
    for field_name, expected_value in taxonomy_identity.items():
        if taxonomy[field_name] != expected_value:
            raise ValueError(
                f"taxonomy {field_name} does not match evaluation config"
            )
    taxonomy_identity_fields_validated = sorted(taxonomy_identity)

    connection = duckdb.connect()
    try:
        escaped_limit = memory_limit.replace("'", "''")
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute(f"SET memory_limit = '{escaped_limit}'")
        _validate_dataset_artifact(
            connection,
            catalog,
            expected_dataset_version=config.dataset_version,
            identity_column="parent_asin",
            artifact_name="product catalog",
        )
        _validate_dataset_artifact(
            connection,
            reviews,
            expected_dataset_version=config.dataset_version,
            identity_column="review_id",
            artifact_name="canonical reviews",
        )
        discovery_identity_fields_validated = _validate_discovery_sample_identity(
            connection,
            discovery_source,
            config=config,
        )
        connection.execute(
            """
            CREATE TEMP TABLE niche_products AS
            SELECT parent_asin, product_title, store, category_path_id,
                category_path_text
            FROM read_parquet(?)
            WHERE category_path_id IN (SELECT * FROM unnest(?))
                AND dataset_version = ?
            """,
            [str(catalog), niche.category_path_ids, config.dataset_version],
        )
        connection.execute(
            """
            CREATE TEMP TABLE discovery_ids AS
            SELECT DISTINCT review_id FROM read_parquet(?)
            """,
            [str(discovery_source)],
        )
        discovery_count_row = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM read_parquet(?)) AS row_count,
                count(*) AS distinct_review_count
            FROM discovery_ids
            """,
            [str(discovery_source)],
        ).fetchone()
        if int(discovery_count_row[0]) == 0:
            raise ValueError("discovery sample must not be empty")
        if int(discovery_count_row[0]) != int(discovery_count_row[1]):
            raise ValueError("discovery sample review IDs must be unique")
        excluded_discovery_count = int(
            connection.execute("SELECT count(*) FROM discovery_ids").fetchone()[0]
        )
        invalid_discovery_membership = int(
            connection.execute(
                """
                SELECT count(*)
                FROM discovery_ids AS discovery
                LEFT JOIN read_parquet(?) AS reviews
                    ON discovery.review_id = reviews.review_id
                    AND reviews.dataset_version = ?
                LEFT JOIN niche_products AS products USING (parent_asin)
                WHERE reviews.review_id IS NULL OR products.parent_asin IS NULL
                """,
                [str(reviews), config.dataset_version],
            ).fetchone()[0]
        )
        if invalid_discovery_membership:
            raise ValueError(
                "discovery sample contains reviews outside the configured "
                "dataset or niche"
            )
        connection.execute(
            """
            CREATE TEMP TABLE eligible_holdout AS
            WITH enriched AS (
                SELECT
                    reviews.review_id,
                    reviews.parent_asin,
                    reviews.asin,
                    reviews.rating,
                    reviews.review_timestamp,
                    reviews.verified_purchase,
                    reviews.helpful_vote,
                    reviews.review_text,
                    products.product_title,
                    products.store,
                    products.category_path_id,
                    products.category_path_text,
                    array_length(
                        regexp_split_to_array(trim(reviews.review_text), '\\s+')
                    ) AS word_count
                FROM read_parquet(?) AS reviews
                INNER JOIN niche_products AS products USING (parent_asin)
                LEFT JOIN discovery_ids USING (review_id)
                WHERE reviews.dataset_version = ?
                    AND discovery_ids.review_id IS NULL
            )
            SELECT
                *,
                year(review_timestamp) AS review_year,
                CASE
                    WHEN rating <= 2 THEN 'low_1_2'
                    WHEN rating = 3 THEN 'mid_3'
                    ELSE 'high_4_5'
                END AS rating_group,
                CASE
                    WHEN word_count <= 40 THEN 'short_5_40'
                    WHEN word_count <= 120 THEN 'medium_41_120'
                    ELSE 'long_121_plus'
                END AS length_bucket,
                concat_ws(
                    '|', rating_group, review_year, length_bucket
                ) AS stratum_id
            FROM enriched
            WHERE word_count >= ?
            """,
            [str(reviews), config.dataset_version, config.minimum_words],
        )
        eligible_count = int(
            connection.execute("SELECT count(*) FROM eligible_holdout").fetchone()[0]
        )

        stratum_rows = connection.execute(
            """
            SELECT stratum_id, count(*) AS population_count
            FROM eligible_holdout
            GROUP BY stratum_id
            ORDER BY stratum_id
            """
        ).fetchall()
        stratum_counts = {str(row[0]): int(row[1]) for row in stratum_rows}
        quotas = allocate_stratum_quotas(
            stratum_counts,
            sample_size=config.representative_sample_size,
            minimum_per_stratum=config.representative_minimum_per_stratum,
        )
        quota_frame = pd.DataFrame(
            [
                {
                    "stratum_id": stratum_id,
                    "population_stratum_count": stratum_counts[stratum_id],
                    "sample_stratum_count": quota,
                }
                for stratum_id, quota in quotas.items()
            ]
        )
        connection.register("evaluation_quotas", quota_frame)
        representative = connection.execute(
            """
            WITH candidates AS (
                SELECT
                    holdout.*,
                    md5(review_id || ':' || cast(? AS VARCHAR))
                        AS deterministic_key
                FROM eligible_holdout AS holdout
            ),
            ranked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY stratum_id
                        ORDER BY deterministic_key, review_id
                    ) AS stratum_sample_rank
                FROM candidates
            )
            SELECT
                ranked.* EXCLUDE (
                    deterministic_key,
                    stratum_sample_rank
                ),
                quotas.population_stratum_count,
                quotas.sample_stratum_count,
                quotas.population_stratum_count::DOUBLE
                    / quotas.sample_stratum_count AS sampling_weight
            FROM ranked
            INNER JOIN evaluation_quotas AS quotas USING (stratum_id)
            WHERE ranked.stratum_sample_rank <= quotas.sample_stratum_count
            """,
            [config.deterministic_seed],
        ).fetchdf()
        if len(representative) != config.representative_sample_size:
            raise RuntimeError("representative sample size is incomplete")
        if not math.isclose(
            float(representative["sampling_weight"].sum()),
            float(eligible_count),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "representative weights do not reconcile to the eligible "
                "holdout population"
            )

        representative_ids_frame = representative[["review_id"]].copy()
        connection.register(
            "selected_representative_ids", representative_ids_frame
        )
        match_frames = []
        alias_candidate_counts: dict[str, int] = {}
        for aspect in taxonomy["aspects"]:
            aspect_id = aspect["aspect_id"]
            pattern = _alias_pattern(aspect["aliases"])
            frame = connection.execute(
                """
                SELECT holdout.review_id, holdout.parent_asin
                FROM eligible_holdout AS holdout
                LEFT JOIN selected_representative_ids AS representative
                    USING (review_id)
                WHERE representative.review_id IS NULL
                    AND regexp_matches(lower(holdout.review_text), ?)
                """,
                [pattern],
            ).fetchdf()
            frame["aspect_id"] = aspect_id
            match_frames.append(frame)
            alias_candidate_counts[aspect_id] = int(frame["review_id"].nunique())
        missing_target_aspects = sorted(
            aspect_id
            for aspect_id, count in alias_candidate_counts.items()
            if count == 0
        )
        if missing_target_aspects:
            raise ValueError(
                "taxonomy aspects have no non-representative alias candidates: "
                f"{missing_target_aspects}"
            )
        matches = pd.concat(match_frames, ignore_index=True)
        selected_target_ids, target_coverage = select_targeted_reviews(
            matches,
            target_size=config.targeted_sample_size,
            minimum_per_aspect=config.targeted_minimum_per_aspect,
            maximum_reviews_per_product=config.maximum_reviews_per_product,
            deterministic_seed=config.deterministic_seed,
        )
        target_aliases = (
            matches[matches["review_id"].isin(selected_target_ids)]
            .groupby("review_id")["aspect_id"]
            .agg(lambda values: sorted(set(values)))
            .to_dict()
        )
        target_ids_frame = pd.DataFrame({"review_id": selected_target_ids})
        connection.register("selected_target_ids", target_ids_frame)

        target_details = connection.execute(
            """
            SELECT holdout.*
            FROM eligible_holdout AS holdout
            INNER JOIN selected_target_ids USING (review_id)
            """
        ).fetchdf()
    finally:
        connection.close()

    representative["sampling_component"] = "representative"
    representative["sampling_weight_semantics"] = (
        "inverse_review_inclusion_probability"
    )
    representative["target_aspect_ids"] = [
        [] for _ in range(len(representative))
    ]
    target_details["population_stratum_count"] = pd.NA
    target_details["sample_stratum_count"] = pd.NA
    target_details["sampling_weight"] = float("nan")
    target_details["sampling_component"] = "targeted"
    target_details["sampling_weight_semantics"] = (
        "not_applicable_targeted_nonprobability_sample"
    )
    target_details["target_aspect_ids"] = target_details["review_id"].map(
        target_aliases
    )
    sample = pd.concat([representative, target_details], ignore_index=True)
    expected_total = (
        config.representative_sample_size + config.targeted_sample_size
    )
    if len(sample) != expected_total or not sample["review_id"].is_unique:
        raise RuntimeError("combined evaluation sample is incomplete or duplicated")
    sample["annotation_sort_key"] = sample["review_id"].map(
        lambda review_id: _stable_key(
            str(review_id), config.deterministic_seed + 1
        )
    )
    sample = sample.sort_values(
        ["annotation_sort_key", "review_id"]
    ).reset_index(drop=True)
    sample["annotation_order"] = sample.index + 1
    sample["annotation_id"] = sample["annotation_order"].map(
        lambda order: f"{config.evaluation_version}:annotation:{order:04d}"
    )
    sample = sample.drop(columns="annotation_sort_key")
    sample["evaluation_version"] = config.evaluation_version
    sample["evaluation_schema_version"] = config.evaluation_schema_version
    sample["taxonomy_version"] = config.taxonomy_version
    sample["niche_id"] = config.niche_id
    sample["niche_version"] = config.niche_version
    sample["dataset_version"] = config.dataset_version

    annotation_queue = _prepare_annotation_queue(sample, queue_destination)

    product_counts = sample["parent_asin"].value_counts()
    discovery_ids = set(
        pd.read_parquet(discovery_source, columns=["review_id"])["review_id"]
    )
    representative_ids = set(
        sample.loc[
            sample["sampling_component"] == "representative", "review_id"
        ]
    )
    targeted_ids = set(
        sample.loc[sample["sampling_component"] == "targeted", "review_id"]
    )
    representative_strata = [
        {
            "stratum_id": stratum_id,
            "population_stratum_count": stratum_counts[stratum_id],
            "sample_stratum_count": quotas[stratum_id],
            "sampling_weight": (
                stratum_counts[stratum_id] / quotas[stratum_id]
            ),
        }
        for stratum_id in sorted(quotas)
    ]
    targeted_coverage = [
        {
            "aspect_id": aspect["aspect_id"],
            "alias_candidate_review_count": alias_candidate_counts[
                aspect["aspect_id"]
            ],
            "selected_target_review_count": target_coverage[aspect["aspect_id"]],
        }
        for aspect in taxonomy["aspects"]
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    queue_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = destination.with_suffix(".tmp.parquet")
    temporary_queue = queue_destination.with_suffix(".tmp.csv")
    temporary_paths = [temporary_output, temporary_queue]
    for temporary_path in temporary_paths:
        temporary_path.unlink(missing_ok=True)
    try:
        sample.to_parquet(temporary_output, index=False)
        if pq.ParquetFile(temporary_output).metadata.num_rows != expected_total:
            raise RuntimeError("written evaluation sample row count is invalid")
        escape_dataframe_for_spreadsheet(annotation_queue).to_csv(
            temporary_queue, index=False
        )

        report = AspectEvaluationSampleReport(
            evaluation_version=config.evaluation_version,
            evaluation_schema_version=config.evaluation_schema_version,
            taxonomy_version=config.taxonomy_version,
            niche_id=config.niche_id,
            niche_version=config.niche_version,
            dataset_version=config.dataset_version,
            representative_sampling_design=(
                "stratified_simple_random_sampling_without_replacement"
            ),
            representative_sampling_weight_semantics=(
                "inverse review inclusion probability N_h/n_h for the full "
                "eligible holdout review population"
            ),
            representative_population_review_count=eligible_count,
            representative_product_cap_applied=False,
            targeted_sampling_design=(
                "deterministic_alias_coverage_nonprobability_sample"
            ),
            targeted_sampling_weight_semantics=(
                "none; targeted rows are diagnostic and must not estimate "
                "population prevalence"
            ),
            targeted_maximum_reviews_per_product=(
                config.maximum_reviews_per_product
            ),
            taxonomy_identity_fields_validated=sorted(
                taxonomy_identity_fields_validated
            ),
            discovery_sample_identity_fields_validated=(
                discovery_identity_fields_validated
            ),
            discovery_sample_membership_validated=True,
            taxonomy_path=_artifact_reference(taxonomy_source),
            discovery_sample_path=_artifact_reference(discovery_source),
            output_path=_artifact_reference(destination),
            output_sha256=sha256_file(temporary_output),
            annotation_queue_path=_artifact_reference(queue_destination),
            annotation_queue_sha256=sha256_file(temporary_queue),
            eligible_holdout_review_count=eligible_count,
            excluded_discovery_review_count=excluded_discovery_count,
            representative_review_count=len(representative_ids),
            targeted_review_count=len(targeted_ids),
            total_review_count=len(sample),
            distinct_product_count=int(sample["parent_asin"].nunique()),
            maximum_reviews_per_product=int(product_counts.max()),
            discovery_overlap_count=len(set(sample["review_id"]) & discovery_ids),
            component_overlap_count=len(representative_ids & targeted_ids),
            deterministic_seed=config.deterministic_seed,
            minimum_words=config.minimum_words,
            representative_strata=representative_strata,
            targeted_aspect_coverage=targeted_coverage,
        )
        staged_artifacts = [
            (temporary_output, destination),
            (temporary_queue, queue_destination),
        ]
        if report_path is not None:
            report_destination = Path(report_path)
            report_destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_report = report_destination.with_suffix(".tmp.json")
            temporary_report.unlink(missing_ok=True)
            temporary_paths.append(temporary_report)
            temporary_report.write_text(
                json.dumps(asdict(report), indent=2), encoding="utf-8"
            )
            staged_artifacts.append((temporary_report, report_destination))
        _commit_staged_artifacts(staged_artifacts)
    except BaseException:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise
    return report


def main() -> None:
    """Build a held-out aspect-extraction annotation sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("niche_config_path", type=Path)
    parser.add_argument("evaluation_config_path", type=Path)
    parser.add_argument("taxonomy_path", type=Path)
    parser.add_argument("discovery_sample_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("annotation_queue_path", type=Path)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()

    root = find_project_root(args.manifest_path.parent)
    manifest = load_dataset_manifest(args.manifest_path)
    niche = load_product_niche_definition(args.niche_config_path)
    config = load_aspect_evaluation_sample_config(args.evaluation_config_path)
    report = build_aspect_evaluation_sample(
        config,
        niche,
        root / manifest.file_by_role("product_catalog").path,
        root / manifest.file_by_role("canonical_reviews").path,
        args.taxonomy_path,
        args.discovery_sample_path,
        args.output_path,
        args.annotation_queue_path,
        report_path=args.report_path,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
