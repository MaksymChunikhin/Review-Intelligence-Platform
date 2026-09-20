"""Train a product-separated multi-label aspect extractor from silver labels."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.analytics.aspect_extraction import build_alias_matcher
from src.analytics.aspect_silver_evaluation import _scores
from src.common.project import find_project_root
from src.ingestion.dataset_manifest import sha256_file


class AspectMultilabelConfig(BaseModel):
    """Configuration for the compact current Hair Conditioners extractor."""

    model_config = ConfigDict(extra="forbid")

    model_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    taxonomy_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    silver_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    method: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    fold_count: int = Field(ge=2, le=10)
    threshold: float = Field(gt=0, lt=1)
    random_state: int = Field(ge=0)
    ngram_min: int = Field(ge=1, le=10)
    ngram_max: int = Field(ge=1, le=10)
    minimum_document_frequency: int = Field(ge=1)
    maximum_features: int = Field(ge=100)
    regularization_c: float = Field(gt=0)
    maximum_iterations: int = Field(ge=10)


@dataclass(frozen=True)
class AspectMultilabelReport:
    model_version: str
    taxonomy_version: str
    silver_version: str
    method: str
    reference_kind: str
    review_count: int
    aspect_count: int
    fold_count: int
    product_overlap_across_folds: int
    threshold: float
    oof_review_aspect_count: int
    oof_metrics: dict[str, float | int]
    alias_baseline_metrics: dict[str, float | int]
    f1_delta_vs_alias_baseline: float
    sample_path: str
    sample_sha256: str
    silver_path: str
    silver_sha256: str
    taxonomy_path: str
    taxonomy_sha256: str
    alias_predictions_path: str
    alias_predictions_sha256: str
    oof_predictions_path: str
    oof_predictions_sha256: str
    model_directory: str
    vectorizer_sha256: str
    classifier_sha256: str
    evaluation_warning: str


def load_config(path: str | Path) -> AspectMultilabelConfig:
    return AspectMultilabelConfig.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def _artifact_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(find_project_root(path.parent)).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = [
        (match.start(), match.end())
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text)
        if match.group(0).strip()
    ]
    return spans or [(0, len(text))]


def _fit(
    texts: pd.Series,
    labels: np.ndarray,
    config: AspectMultilabelConfig,
) -> tuple[TfidfVectorizer, OneVsRestClassifier]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.minimum_document_frequency,
        sublinear_tf=True,
        max_features=config.maximum_features,
    )
    features = vectorizer.fit_transform(texts)
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=config.regularization_c,
            class_weight="balanced",
            max_iter=config.maximum_iterations,
            random_state=config.random_state,
            solver="liblinear",
        )
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        classifier.fit(features, labels)
    return vectorizer, classifier


def _semantic_evidence(
    text: str,
    aspect_index: int,
    vectorizer: TfidfVectorizer,
    classifier: OneVsRestClassifier,
) -> dict[str, Any]:
    spans = _sentence_spans(text)
    snippets = [text[start:end] for start, end in spans]
    probabilities = classifier.predict_proba(vectorizer.transform(snippets))
    best = int(np.argmax(probabilities[:, aspect_index]))
    start, end = spans[best]
    return {
        "text": text[start:end],
        "start": start,
        "end": end,
        "source": "semantic_sentence_v1",
    }


def _write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def train_aspect_multilabel(
    config: AspectMultilabelConfig,
    taxonomy_path: str | Path,
    sample_path: str | Path,
    silver_path: str | Path,
    alias_predictions_path: str | Path,
    oof_predictions_path: str | Path,
    model_directory: str | Path,
    *,
    report_path: str | Path | None = None,
) -> AspectMultilabelReport:
    """Create leakage-aware OOF predictions and one full-data current model."""
    taxonomy_source = Path(taxonomy_path)
    sample_source = Path(sample_path)
    silver_source = Path(silver_path)
    alias_source = Path(alias_predictions_path)
    oof_destination = Path(oof_predictions_path)
    model_destination = Path(model_directory)
    taxonomy = json.loads(taxonomy_source.read_text(encoding="utf-8"))
    build_alias_matcher(taxonomy)
    if taxonomy.get("taxonomy_version") != config.taxonomy_version:
        raise ValueError("taxonomy version does not match aspect model config")
    aspect_ids = [str(item["aspect_id"]) for item in taxonomy["aspects"]]
    if len(aspect_ids) != len(set(aspect_ids)):
        raise ValueError("taxonomy aspect IDs must be unique")

    sample = pd.read_parquet(
        sample_source,
        columns=["review_id", "parent_asin", "review_text"],
    )
    silver = pd.read_parquet(
        silver_source,
        columns=["review_id", "silver_version", "taxonomy_version", "aspect_ids_json"],
    )
    aliases = pd.read_parquet(alias_source)
    if sample["review_id"].duplicated().any() or sample["parent_asin"].isna().any():
        raise ValueError("sample must have unique reviews and defined parent products")
    if set(silver["silver_version"].astype(str)) != {config.silver_version}:
        raise ValueError("silver version does not match aspect model config")
    if set(silver["taxonomy_version"].astype(str)) != {config.taxonomy_version}:
        raise ValueError("silver taxonomy version does not match config")
    data = sample.merge(silver, on="review_id", validate="one_to_one")
    if len(data) != len(sample):
        raise ValueError("silver reference must cover every sample review")
    if data["parent_asin"].nunique() < config.fold_count:
        raise ValueError("not enough parent products for configured folds")

    labeler = MultiLabelBinarizer(classes=aspect_ids)
    labels = labeler.fit_transform(
        [json.loads(value) for value in data["aspect_ids_json"]]
    )
    unknown = {
        aspect
        for value in data["aspect_ids_json"]
        for aspect in json.loads(value)
        if aspect not in set(aspect_ids)
    }
    if unknown:
        raise ValueError(f"silver reference contains unknown aspects: {sorted(unknown)}")

    required_alias_columns = {
        "review_id", "aspect_id", "evidence_json", "matched_aliases_json"
    }
    if required_alias_columns - set(aliases.columns):
        raise ValueError("alias predictions are missing required evidence columns")
    alias_rows = {
        (str(row.review_id), str(row.aspect_id)): row
        for row in aliases.itertuples(index=False)
    }
    if not set(alias_rows) <= {
        (str(review_id), aspect_id)
        for review_id in data["review_id"]
        for aspect_id in aspect_ids
    }:
        raise ValueError("alias predictions do not align with sample and taxonomy")

    probabilities = np.zeros(labels.shape, dtype=np.float32)
    folds = np.zeros(len(data), dtype=np.int16)
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    split_products: list[set[str]] = []
    splitter = GroupKFold(config.fold_count)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(data["review_text"], labels, data["parent_asin"]), start=1
    ):
        vectorizer, classifier = _fit(
            data.iloc[train_index]["review_text"], labels[train_index], config
        )
        fold_probabilities = classifier.predict_proba(
            vectorizer.transform(data.iloc[test_index]["review_text"])
        )
        probabilities[test_index] = fold_probabilities
        folds[test_index] = fold
        split_products.append(set(data.iloc[test_index]["parent_asin"].astype(str)))
        for local_index, row_index in enumerate(test_index):
            review_id = str(data.iloc[row_index]["review_id"])
            text = str(data.iloc[row_index]["review_text"])
            selected = np.flatnonzero(fold_probabilities[local_index] >= config.threshold)
            for aspect_index in selected:
                aspect_id = aspect_ids[int(aspect_index)]
                if (review_id, aspect_id) not in alias_rows:
                    evidence[(review_id, aspect_id)] = _semantic_evidence(
                        text, int(aspect_index), vectorizer, classifier
                    )

    overlap = sum(
        len(left & right)
        for index, left in enumerate(split_products)
        for right in split_products[index + 1 :]
    )
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(data.itertuples(index=False)):
        review_id = str(row.review_id)
        semantic_indices = set(np.flatnonzero(probabilities[row_index] >= config.threshold))
        alias_indices = {
            aspect_ids.index(aspect_id)
            for rid, aspect_id in alias_rows
            if rid == review_id
        }
        for aspect_index in sorted(semantic_indices | alias_indices):
            aspect_id = aspect_ids[aspect_index]
            alias = alias_rows.get((review_id, aspect_id))
            sources = []
            if alias is not None:
                sources.append("exact_alias")
                evidence_json = str(alias.evidence_json)
                matched_aliases_json = str(alias.matched_aliases_json)
            else:
                evidence_json = json.dumps(
                    [evidence[(review_id, aspect_id)]], ensure_ascii=False
                )
                matched_aliases_json = "[]"
            if aspect_index in semantic_indices:
                sources.append("char_tfidf")
            rows.append(
                {
                    "extraction_version": config.model_version,
                    "taxonomy_version": config.taxonomy_version,
                    "method": config.method,
                    "review_id": review_id,
                    "aspect_id": aspect_id,
                    "evidence_count": len(json.loads(evidence_json)),
                    "matched_aliases_json": matched_aliases_json,
                    "evidence_json": evidence_json,
                    "aspect_probability": float(probabilities[row_index, aspect_index]),
                    "prediction_sources_json": json.dumps(sources),
                    "oof_fold": int(folds[row_index]),
                }
            )
    predictions = pd.DataFrame(rows).sort_values(["review_id", "aspect_id"])
    _write_parquet(predictions, oof_destination)

    vectorizer, classifier = _fit(data["review_text"], labels, config)
    model_destination.mkdir(parents=True, exist_ok=True)
    vectorizer_path = model_destination / "vectorizer.joblib"
    classifier_path = model_destination / "classifier.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, classifier_path)
    metadata = {
        "model_version": config.model_version,
        "taxonomy_version": config.taxonomy_version,
        "silver_version": config.silver_version,
        "method": config.method,
        "classes": aspect_ids,
        "threshold": config.threshold,
        "training_review_count": len(data),
        "training_reference_kind": "gemini_silver_not_human_gold",
        "sample_sha256": sha256_file(sample_source),
        "silver_sha256": sha256_file(silver_source),
        "taxonomy_sha256": sha256_file(taxonomy_source),
    }
    metadata_path = model_destination / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    predicted_pairs = set(zip(predictions["review_id"], predictions["aspect_id"]))
    reference_pairs = {
        (str(row.review_id), str(aspect))
        for row in data.itertuples(index=False)
        for aspect in json.loads(row.aspect_ids_json)
    }
    baseline_pairs = set(alias_rows)
    oof_metrics = _scores(predicted_pairs, reference_pairs)
    baseline_metrics = _scores(baseline_pairs, reference_pairs)
    report = AspectMultilabelReport(
        model_version=config.model_version,
        taxonomy_version=config.taxonomy_version,
        silver_version=config.silver_version,
        method=config.method,
        reference_kind="gemini_silver_not_human_gold",
        review_count=len(data),
        aspect_count=len(aspect_ids),
        fold_count=config.fold_count,
        product_overlap_across_folds=overlap,
        threshold=config.threshold,
        oof_review_aspect_count=len(predictions),
        oof_metrics=oof_metrics,
        alias_baseline_metrics=baseline_metrics,
        f1_delta_vs_alias_baseline=float(oof_metrics["f1"] - baseline_metrics["f1"]),
        sample_path=_artifact_reference(sample_source),
        sample_sha256=sha256_file(sample_source),
        silver_path=_artifact_reference(silver_source),
        silver_sha256=sha256_file(silver_source),
        taxonomy_path=_artifact_reference(taxonomy_source),
        taxonomy_sha256=sha256_file(taxonomy_source),
        alias_predictions_path=_artifact_reference(alias_source),
        alias_predictions_sha256=sha256_file(alias_source),
        oof_predictions_path=_artifact_reference(oof_destination),
        oof_predictions_sha256=sha256_file(oof_destination),
        model_directory=_artifact_reference(model_destination),
        vectorizer_sha256=sha256_file(vectorizer_path),
        classifier_sha256=sha256_file(classifier_path),
        evaluation_warning=(
            "OOF folds separate parent products, but the threshold was selected "
            "during silver-label experimentation. Metrics are diagnostic agreement "
            "with Gemini silver, not unbiased human-gold accuracy."
        ),
    )
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("taxonomy_path", type=Path)
    parser.add_argument("sample_path", type=Path)
    parser.add_argument("silver_path", type=Path)
    parser.add_argument("alias_predictions_path", type=Path)
    parser.add_argument("oof_predictions_path", type=Path)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    report = train_aspect_multilabel(
        load_config(args.config_path),
        args.taxonomy_path,
        args.sample_path,
        args.silver_path,
        args.alias_predictions_path,
        args.oof_predictions_path,
        args.model_directory,
        report_path=args.report_path,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
