"""Tests for the compact hybrid aspect extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.aspect_multilabel import AspectMultilabelConfig, train_aspect_multilabel
from src.ml.aspect_inference import predict_aspects


def test_train_aspect_multilabel_writes_oof_and_current_model(tmp_path: Path) -> None:
    taxonomy = {
        "status": "approved",
        "taxonomy_version": "taxonomy_v1",
        "aspects": [
            {"aspect_id": "scent", "canonical_name": "Scent", "aliases": ["scent"]},
            {"aspect_id": "softness", "canonical_name": "Softness", "aliases": ["soft"]},
        ],
    }
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(taxonomy), encoding="utf-8")
    sample = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(6)],
            "parent_asin": [f"p{i}" for i in range(6)],
            "review_text": [
                "lovely floral scent", "hair feels very soft", "bad chemical scent",
                "soft and smooth hair", "pleasant perfume", "silky gentle finish",
            ],
        }
    )
    sample_path = tmp_path / "sample.parquet"
    sample.to_parquet(sample_path, index=False)
    silver = pd.DataFrame(
        {
            "review_id": sample["review_id"],
            "silver_version": "silver_v1",
            "taxonomy_version": "taxonomy_v1",
            "aspect_ids_json": [
                '["scent"]', '["softness"]', '["scent"]',
                '["softness"]', '["scent"]', '["softness"]',
            ],
        }
    )
    silver_path = tmp_path / "silver.parquet"
    silver.to_parquet(silver_path, index=False)
    aliases = pd.DataFrame(
        {
            "review_id": ["r0", "r1"],
            "aspect_id": ["scent", "softness"],
            "evidence_json": [
                '[{"text":"scent","start":14,"end":19}]',
                '[{"text":"soft","start":16,"end":20}]',
            ],
            "matched_aliases_json": ['["scent"]', '["soft"]'],
        }
    )
    aliases_path = tmp_path / "aliases.parquet"
    aliases.to_parquet(aliases_path, index=False)
    output_path = tmp_path / "oof.parquet"
    model_path = tmp_path / "model"
    report = train_aspect_multilabel(
        AspectMultilabelConfig(
            model_version="extractor_v2",
            taxonomy_version="taxonomy_v1",
            silver_version="silver_v1",
            method="hybrid_v1",
            fold_count=3,
            threshold=0.45,
            random_state=1,
            ngram_min=2,
            ngram_max=3,
            minimum_document_frequency=1,
            maximum_features=500,
            regularization_c=1.0,
            maximum_iterations=100,
        ),
        taxonomy_path,
        sample_path,
        silver_path,
        aliases_path,
        output_path,
        model_path,
    )
    output = pd.read_parquet(output_path)
    assert report.product_overlap_across_folds == 0
    assert output["oof_fold"].nunique() == 3
    assert set(output["extraction_version"]) == {"extractor_v2"}
    assert (model_path / "vectorizer.joblib").is_file()
    assert (model_path / "classifier.joblib").is_file()
    texts = sample.set_index("review_id")["review_text"]
    for row in output.itertuples(index=False):
        for item in json.loads(row.evidence_json):
            assert texts[row.review_id][item["start"] : item["end"]] == item["text"]

    inference_path = tmp_path / "inference.parquet"
    inference_report = predict_aspects(
        AspectMultilabelConfig(
            model_version="extractor_v2",
            taxonomy_version="taxonomy_v1",
            silver_version="silver_v1",
            method="hybrid_v1",
            fold_count=3,
            threshold=0.45,
            random_state=1,
            ngram_min=2,
            ngram_max=3,
            minimum_document_frequency=1,
            maximum_features=500,
            regularization_c=1.0,
            maximum_iterations=100,
        ),
        taxonomy_path,
        sample_path,
        model_path,
        inference_path,
        batch_size=2,
    )
    inference = pd.read_parquet(inference_path)
    assert inference_report.input_review_count == 6
    assert set(inference["review_id"]) <= set(sample["review_id"])
    for row in inference.itertuples(index=False):
        for item in json.loads(row.evidence_json):
            assert texts[row.review_id][item["start"] : item["end"]] == item["text"]
