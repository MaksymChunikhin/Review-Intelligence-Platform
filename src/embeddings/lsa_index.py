"""Build and query a compact local TF-IDF/SVD evidence index."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from src.ingestion.dataset_manifest import sha256_file


@dataclass(frozen=True)
class LsaIndexReport:
    index_version: str
    representation: str
    source_path: str
    source_sha256: str
    input_review_count: int
    indexed_review_count: int
    feature_count: int
    component_count: int
    vectorizer_path: str
    vectorizer_sha256: str
    svd_path: str
    svd_sha256: str
    normalizer_path: str
    normalizer_sha256: str
    embeddings_path: str
    embeddings_sha256: str
    metadata_path: str
    metadata_sha256: str
    warning: str


def _atomic_joblib(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        joblib.dump(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_numpy(value: np.ndarray, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(value: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        value.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_lsa_index(
    reviews_path: str | Path,
    output_dir: str | Path,
    *,
    index_version: str,
    max_features: int = 30_000,
    component_count: int = 128,
    minimum_document_frequency: int = 3,
) -> LsaIndexReport:
    """Fit a deterministic local latent-semantic baseline over review text."""
    if max_features < 100:
        raise ValueError("max_features must be at least 100")
    if component_count < 2:
        raise ValueError("component_count must be at least 2")
    source = Path(reviews_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    reviews = pd.read_parquet(
        source,
        columns=["review_id", "parent_asin", "review_text"],
    )
    reviews["review_text"] = reviews["review_text"].fillna("").astype(str)
    indexed = reviews[reviews["review_text"].str.strip().str.len() >= 2].copy()
    if indexed["review_id"].duplicated().any():
        raise ValueError("review_id must be unique")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=minimum_document_frequency,
        max_df=0.995,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    sparse = vectorizer.fit_transform(indexed["review_text"])
    actual_components = min(
        component_count,
        max(2, sparse.shape[0] - 1),
        max(2, sparse.shape[1] - 1),
    )
    svd = TruncatedSVD(n_components=actual_components, random_state=42)
    normalizer = Normalizer(copy=False)
    embeddings = normalizer.fit_transform(svd.fit_transform(sparse)).astype(
        np.float32,
        copy=False,
    )

    vectorizer_path = destination / "vectorizer.joblib"
    svd_path = destination / "svd.joblib"
    normalizer_path = destination / "normalizer.joblib"
    embeddings_path = destination / "embeddings.npy"
    metadata_path = destination / "reviews.parquet"
    _atomic_joblib(vectorizer, vectorizer_path)
    _atomic_joblib(svd, svd_path)
    _atomic_joblib(normalizer, normalizer_path)
    _atomic_numpy(embeddings, embeddings_path)
    _atomic_parquet(indexed[["review_id", "parent_asin"]], metadata_path)
    report = LsaIndexReport(
        index_version=index_version,
        representation="word_1_2_tfidf_truncated_svd_l2",
        source_path=str(source),
        source_sha256=sha256_file(source),
        input_review_count=int(len(reviews)),
        indexed_review_count=int(len(indexed)),
        feature_count=int(sparse.shape[1]),
        component_count=int(actual_components),
        vectorizer_path=str(vectorizer_path),
        vectorizer_sha256=sha256_file(vectorizer_path),
        svd_path=str(svd_path),
        svd_sha256=sha256_file(svd_path),
        normalizer_path=str(normalizer_path),
        normalizer_sha256=sha256_file(normalizer_path),
        embeddings_path=str(embeddings_path),
        embeddings_sha256=sha256_file(embeddings_path),
        metadata_path=str(metadata_path),
        metadata_sha256=sha256_file(metadata_path),
        warning=(
            "Local LSA retrieval baseline; evaluate against human relevance "
            "judgments before treating it as production semantic search."
        ),
    )
    manifest_path = destination / "manifest.json"
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return report


class LsaEvidenceIndex:
    """Memory-map one immutable LSA index and return cosine scores."""

    def __init__(self, index_dir: str | Path) -> None:
        root = Path(index_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        artifact_names = {
            "vectorizer": "vectorizer.joblib",
            "svd": "svd.joblib",
            "normalizer": "normalizer.joblib",
            "embeddings": "embeddings.npy",
            "metadata": "reviews.parquet",
        }
        for key, filename in artifact_names.items():
            artifact = root / filename
            expected = manifest.get(f"{key}_sha256")
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            if not expected or sha256_file(artifact) != str(expected):
                raise ValueError(f"LSA artifact integrity check failed: {artifact}")
        self.index_version = str(manifest["index_version"])
        self.vectorizer: TfidfVectorizer = joblib.load(root / "vectorizer.joblib")
        self.svd: TruncatedSVD = joblib.load(root / "svd.joblib")
        self.normalizer: Normalizer = joblib.load(root / "normalizer.joblib")
        self.embeddings = np.load(root / "embeddings.npy", mmap_mode="r")
        self.metadata = pd.read_parquet(root / "reviews.parquet")
        if len(self.metadata) != len(self.embeddings):
            raise ValueError("embedding and metadata row counts differ")

    def search_scores(
        self,
        query: str,
        *,
        allowed_review_ids: set[str] | None = None,
        top_k: int = 200,
    ) -> dict[str, float]:
        """Return positive dense similarity scores keyed by review ID."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        sparse_query = self.vectorizer.transform([query])
        dense_query = self.normalizer.transform(self.svd.transform(sparse_query))[0]
        if not np.any(dense_query):
            return {}
        if allowed_review_ids is None:
            indexes = np.arange(len(self.metadata))
        else:
            mask = self.metadata["review_id"].astype(str).isin(allowed_review_ids).to_numpy()
            indexes = np.flatnonzero(mask)
        if indexes.size == 0:
            return {}
        scores = np.asarray(self.embeddings[indexes] @ dense_query).reshape(-1)
        positive = np.flatnonzero(scores > 0)
        if positive.size == 0:
            return {}
        take = min(top_k, positive.size)
        local_top = positive[np.argpartition(scores[positive], -take)[-take:]]
        local_top = local_top[np.argsort(scores[local_top])[::-1]]
        result: dict[str, float] = {}
        for local_index in local_top:
            index = int(indexes[local_index])
            result[str(self.metadata.iloc[index]["review_id"])] = float(
                scores[local_index]
            )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--index-version", required=True)
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--component-count", type=int, default=128)
    parser.add_argument("--minimum-document-frequency", type=int, default=3)
    args = parser.parse_args()
    report = build_lsa_index(
        args.reviews_path,
        args.output_dir,
        index_version=args.index_version,
        max_features=args.max_features,
        component_count=args.component_count,
        minimum_document_frequency=args.minimum_document_frequency,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
