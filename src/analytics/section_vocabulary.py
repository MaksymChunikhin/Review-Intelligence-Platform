"""Profile frequent review phrases locally for each competitor niche."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
from sklearn.feature_extraction.text import CountVectorizer

from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file


@dataclass(frozen=True)
class SectionVocabularyReport:
    vocabulary_version: str
    source_path: str
    source_sha256: str
    review_count: int
    segment_count: int
    ngram_range: list[int]
    maximum_features_per_segment: int
    top_phrase_limit: int
    segments: list[dict[str, Any]]


def _artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def build_section_vocabulary_report(
    reviews_path: str | Path,
    output_path: str | Path,
    *,
    vocabulary_version: str,
    maximum_features_per_segment: int = 5_000,
    top_phrase_limit: int = 150,
) -> SectionVocabularyReport:
    """Count document-level unigrams, bigrams, and trigrams in every segment."""
    source = Path(reviews_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if maximum_features_per_segment < top_phrase_limit or top_phrase_limit < 1:
        raise ValueError("feature and phrase limits are inconsistent")

    connection = duckdb.connect()
    try:
        segment_rows = connection.execute(
            """
            SELECT
                competitor_niche_id,
                min(competitor_niche_name) AS competitor_niche_name,
                min(analytical_family_id) AS analytical_family_id,
                min(comparison_status) AS comparison_status,
                count(*) AS review_count
            FROM read_parquet(?)
            GROUP BY competitor_niche_id
            ORDER BY competitor_niche_id
            """,
            [str(source)],
        ).fetchall()
        total_review_count = int(
            connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(source)]
            ).fetchone()[0]
        )
        segments: list[dict[str, Any]] = []
        for (
            segment_id,
            segment_name,
            family_id,
            comparison_status,
            review_count,
        ) in segment_rows:
            texts = connection.execute(
                """
                SELECT review_text
                FROM read_parquet(?)
                WHERE competitor_niche_id = ?
                ORDER BY review_id
                """,
                [str(source), segment_id],
            ).fetchnumpy()["review_text"].tolist()
            minimum_document_frequency = max(5, int(len(texts) * 0.0005))
            vectorizer = CountVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 3),
                binary=True,
                min_df=minimum_document_frequency,
                max_df=0.95,
                max_features=maximum_features_per_segment,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
            )
            matrix = vectorizer.fit_transform(str(text) for text in texts)
            document_frequencies = matrix.sum(axis=0).A1
            phrases = vectorizer.get_feature_names_out()
            ranking = sorted(
                zip(phrases, document_frequencies, strict=True),
                key=lambda item: (-int(item[1]), str(item[0])),
            )[:top_phrase_limit]
            segments.append(
                {
                    "competitor_niche_id": str(segment_id),
                    "competitor_niche_name": str(segment_name),
                    "analytical_family_id": str(family_id),
                    "comparison_status": str(comparison_status),
                    "review_count": int(review_count),
                    "minimum_document_frequency": minimum_document_frequency,
                    "vocabulary_feature_count": int(len(phrases)),
                    "top_phrases": [
                        {
                            "phrase": str(phrase),
                            "review_count": int(count),
                            "review_share": int(count) / int(review_count),
                        }
                        for phrase, count in ranking
                    ],
                }
            )
    finally:
        connection.close()

    report = SectionVocabularyReport(
        vocabulary_version=vocabulary_version,
        source_path=_artifact_reference(source),
        source_sha256=sha256_file(source),
        review_count=total_review_count,
        segment_count=len(segments),
        ngram_range=[1, 3],
        maximum_features_per_segment=maximum_features_per_segment,
        top_phrase_limit=top_phrase_limit,
        segments=segments,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    temporary.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--vocabulary-version", required=True)
    parser.add_argument("--maximum-features", type=int, default=5_000)
    parser.add_argument("--top-phrases", type=int, default=150)
    args = parser.parse_args()
    report = build_section_vocabulary_report(
        args.reviews_path,
        args.output_path,
        vocabulary_version=args.vocabulary_version,
        maximum_features_per_segment=args.maximum_features,
        top_phrase_limit=args.top_phrases,
    )
    print(
        json.dumps(
            {
                "vocabulary_version": report.vocabulary_version,
                "review_count": report.review_count,
                "segment_count": report.segment_count,
                "output_path": str(args.output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
