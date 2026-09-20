"""Deterministic multi-aspect lexical extraction with exact evidence spans."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file


@dataclass(frozen=True)
class AliasRule:
    """Map one compiled regex branch to one approved taxonomy alias."""

    group_name: str
    aspect_id: str
    alias: str


@dataclass(frozen=True)
class AspectExtractionReport:
    """Describe one versioned lexical extraction run."""

    extraction_version: str
    taxonomy_version: str
    method: str
    taxonomy_path: str
    taxonomy_sha256: str
    input_path: str
    input_sha256: str
    output_path: str
    output_sha256: str
    review_count: int
    matched_review_count: int
    no_match_review_count: int
    review_aspect_count: int
    evidence_span_count: int
    distinct_aspect_count: int
    human_gold_evaluated: bool
    warning: str
    by_sampling_component: list[dict[str, Any]]


def _artifact_reference(path: Path) -> str:
    """Prefer a repository-relative path in persisted provenance."""
    try:
        root = find_project_root(path.parent)
        return path.resolve().relative_to(root).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _alias_expression(alias: str) -> str:
    """Match a literal alias with flexible spaces and safe word boundaries."""
    tokens = re.split(r"\s+", alias.strip())
    expression = r"\s+".join(re.escape(token) for token in tokens if token)
    if not expression:
        raise ValueError("taxonomy aliases must not be empty")
    return rf"(?<![A-Za-z0-9])(?:{expression})(?![A-Za-z0-9])"


def build_alias_matcher(
    taxonomy: dict[str, Any],
) -> tuple[re.Pattern[str], dict[str, AliasRule]]:
    """Compile all unique approved aliases into one deterministic matcher."""
    if taxonomy.get("status") != "approved":
        raise ValueError("aspect extraction requires an approved taxonomy")
    aspects = taxonomy.get("aspects")
    if not isinstance(aspects, list) or not aspects:
        raise ValueError("approved taxonomy must contain aspects")

    alias_owner: dict[str, str] = {}
    aliases: list[tuple[str, str]] = []
    aspect_ids: set[str] = set()
    for aspect in aspects:
        aspect_id = str(aspect["aspect_id"])
        if aspect_id in aspect_ids:
            raise ValueError("taxonomy aspect IDs must be unique")
        aspect_ids.add(aspect_id)
        values = {
            str(aspect["canonical_name"]),
            *(str(value) for value in aspect.get("aliases", [])),
            *(str(value) for value in aspect.get("additional_aliases", [])),
        }
        for alias in values:
            normalized = " ".join(alias.casefold().split())
            if not normalized:
                raise ValueError("taxonomy aliases must not be empty")
            owner = alias_owner.setdefault(normalized, aspect_id)
            if owner != aspect_id:
                raise ValueError(
                    f"taxonomy alias {alias!r} belongs to multiple aspects"
                )
            aliases.append((alias, aspect_id))

    unique_aliases = {
        (" ".join(alias.casefold().split()), aspect_id): alias
        for alias, aspect_id in aliases
    }
    ordered = sorted(
        (
            (display_alias, aspect_id, normalized)
            for (normalized, aspect_id), display_alias in unique_aliases.items()
        ),
        key=lambda item: (-len(item[2]), item[2], item[1]),
    )
    rules: dict[str, AliasRule] = {}
    branches: list[str] = []
    for index, (alias, aspect_id, _) in enumerate(ordered):
        group_name = f"a{index}"
        rules[group_name] = AliasRule(
            group_name=group_name,
            aspect_id=aspect_id,
            alias=alias,
        )
        branches.append(f"(?P<{group_name}>{_alias_expression(alias)})")
    return re.compile("|".join(branches), flags=re.IGNORECASE), rules


def extract_review_aspects(
    review_id: str,
    review_text: str,
    *,
    matcher: re.Pattern[str],
    rules: dict[str, AliasRule],
    max_evidence_per_aspect: int = 5,
) -> list[dict[str, Any]]:
    """Return one row per matched aspect with exact source-text evidence."""
    if max_evidence_per_aspect < 1:
        raise ValueError("max_evidence_per_aspect must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for match in matcher.finditer(review_text):
        rule = rules[match.lastgroup or ""]
        evidence = {
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "matched_alias": rule.alias,
        }
        values = grouped.setdefault(rule.aspect_id, [])
        if evidence not in values and len(values) < max_evidence_per_aspect:
            values.append(evidence)

    rows = []
    for aspect_id in sorted(grouped):
        evidence = grouped[aspect_id]
        rows.append(
            {
                "review_id": review_id,
                "aspect_id": aspect_id,
                "evidence_count": len(evidence),
                "matched_aliases_json": json.dumps(
                    sorted({item["matched_alias"] for item in evidence}),
                    ensure_ascii=False,
                ),
                "evidence_json": json.dumps(evidence, ensure_ascii=False),
            }
        )
    return rows


def _load_reviews(path: Path) -> pd.DataFrame:
    """Load a supported tabular review artifact."""
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    raise ValueError("review input must be Parquet or CSV")


def extract_aspects(
    taxonomy_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    extraction_version: str,
    report_path: str | Path | None = None,
    review_id_column: str = "review_id",
    text_column: str = "review_text",
    max_evidence_per_aspect: int = 5,
) -> AspectExtractionReport:
    """Extract approved taxonomy aliases from every supplied review."""
    taxonomy_source = Path(taxonomy_path)
    input_source = Path(input_path)
    destination = Path(output_path)
    taxonomy = json.loads(taxonomy_source.read_text(encoding="utf-8"))
    matcher, rules = build_alias_matcher(taxonomy)
    reviews = _load_reviews(input_source)
    required = {review_id_column, text_column}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"review input is missing columns: {sorted(missing)}")
    if reviews.empty:
        raise ValueError("review input must not be empty")
    if reviews[review_id_column].isna().any() or reviews[review_id_column].duplicated().any():
        raise ValueError("review IDs must be unique and non-null")
    if reviews[text_column].isna().any():
        raise ValueError("review text must be non-null")

    rows: list[dict[str, Any]] = []
    for review in reviews[[review_id_column, text_column]].itertuples(index=False):
        review_id = str(review[0])
        review_text = str(review[1])
        rows.extend(
            extract_review_aspects(
                review_id,
                review_text,
                matcher=matcher,
                rules=rules,
                max_evidence_per_aspect=max_evidence_per_aspect,
            )
        )
    columns = [
        "extraction_version",
        "taxonomy_version",
        "method",
        "review_id",
        "aspect_id",
        "evidence_count",
        "matched_aliases_json",
        "evidence_json",
    ]
    extracted = pd.DataFrame(rows)
    if extracted.empty:
        extracted = pd.DataFrame(columns=columns[3:])
    extracted.insert(0, "method", "exact_taxonomy_alias_v1")
    extracted.insert(0, "taxonomy_version", taxonomy["taxonomy_version"])
    extracted.insert(0, "extraction_version", extraction_version)
    extracted = extracted[columns]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        extracted.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    matched_reviews = int(extracted["review_id"].nunique())
    component_rows: list[dict[str, Any]] = []
    if "sampling_component" in reviews.columns:
        for component, component_reviews in reviews.groupby(
            "sampling_component", sort=True
        ):
            component_ids = set(component_reviews[review_id_column].astype(str))
            component_predictions = extracted[
                extracted["review_id"].isin(component_ids)
            ]
            component_matched = int(
                component_predictions["review_id"].nunique()
            )
            component_rows.append(
                {
                    "sampling_component": str(component),
                    "review_count": int(len(component_reviews)),
                    "matched_review_count": component_matched,
                    "coverage": component_matched / len(component_reviews),
                    "review_aspect_count": int(len(component_predictions)),
                }
            )
    report = AspectExtractionReport(
        extraction_version=extraction_version,
        taxonomy_version=str(taxonomy["taxonomy_version"]),
        method="exact_taxonomy_alias_v1",
        taxonomy_path=_artifact_reference(taxonomy_source),
        taxonomy_sha256=sha256_file(taxonomy_source),
        input_path=_artifact_reference(input_source),
        input_sha256=sha256_file(input_source),
        output_path=_artifact_reference(destination),
        output_sha256=sha256_file(destination),
        review_count=len(reviews),
        matched_review_count=matched_reviews,
        no_match_review_count=len(reviews) - matched_reviews,
        review_aspect_count=len(extracted),
        evidence_span_count=int(extracted["evidence_count"].sum()),
        distinct_aspect_count=int(extracted["aspect_id"].nunique()),
        human_gold_evaluated=False,
        warning=(
            "Lexical baseline coverage is not precision, recall, or population "
            "prevalence; targeted rows are non-probability diagnostics."
        ),
        by_sampling_component=component_rows,
    )
    if report_path is not None:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        report_destination.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    """Run deterministic aspect extraction from a tabular review artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("taxonomy_path", type=Path)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--extraction-version", required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--review-id-column", default="review_id")
    parser.add_argument("--text-column", default="review_text")
    parser.add_argument("--max-evidence-per-aspect", type=int, default=5)
    args = parser.parse_args()
    report = extract_aspects(
        args.taxonomy_path,
        args.input_path,
        args.output_path,
        extraction_version=args.extraction_version,
        report_path=args.report_path,
        review_id_column=args.review_id_column,
        text_column=args.text_column,
        max_evidence_per_aspect=args.max_evidence_per_aspect,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
