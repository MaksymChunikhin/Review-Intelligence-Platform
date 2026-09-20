"""Tests for the local latent-semantic evidence index."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.embeddings.lsa_index import LsaEvidenceIndex, build_lsa_index


def test_build_and_query_lsa_index_with_product_filter(tmp_path: Path) -> None:
    source = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4", "r5"],
            "parent_asin": ["p1", "p1", "p2", "p2", "p3"],
            # Repeated topic terms keep the tiny SVD fixture stable across
            # supported scikit-learn versions.
            "review_text": [
                "The spray pump broke",
                "The spray nozzle pump broke",
                "Hair feels soft and smooth",
                "Lovely soft hair",
                "Strong pleasant scent",
            ],
        }
    ).to_parquet(source, index=False)
    output = tmp_path / "index"
    report = build_lsa_index(
        source,
        output,
        index_version="lsa_v1",
        max_features=100,
        component_count=2,
        minimum_document_frequency=1,
    )

    index = LsaEvidenceIndex(output)
    scores = index.search_scores(
        "broken spray pump",
        allowed_review_ids={"r1", "r3"},
        top_k=2,
    )

    assert report.indexed_review_count == 5
    assert list(scores)[0] == "r1"
    assert set(scores) <= {"r1", "r3"}
    assert (output / "manifest.json").is_file()
