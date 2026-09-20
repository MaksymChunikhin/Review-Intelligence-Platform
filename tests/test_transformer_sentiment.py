"""Tests for Transformer sentiment configuration and metric helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.ml.transformer_sentiment as transformer_module
from src.ml.sentiment import SENTIMENT_MANIFEST_COLUMNS
from src.ml.transformer_sentiment import (
    TransformerSentimentConfig,
    normalize_transformer_artifact_metadata,
    save_transformer_evaluation,
    transformer_training_metrics,
)


def test_transformer_config_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch sizes"):
        TransformerSentimentConfig(train_batch_size=0)


def test_checkpoint_cleanup_preserves_unrelated_files(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "checkpoints"
    first = checkpoint_directory / "checkpoint-10"
    second = checkpoint_directory / "checkpoint-20"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "model.safetensors").write_bytes(b"checkpoint")
    unrelated = checkpoint_directory / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    removed = transformer_module._remove_transformer_checkpoints(
        checkpoint_directory
    )

    assert removed == 2
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_transformer_training_metrics_use_all_three_classes() -> None:
    prediction = SimpleNamespace(
        predictions=np.array(
            [
                [3.0, 1.0, 0.0],
                [0.0, 2.0, 1.0],
                [0.0, 1.0, 3.0],
                [0.0, 2.0, 1.0],
            ]
        ),
        label_ids=np.array([0, 1, 2, 2]),
    )

    metrics = transformer_training_metrics(prediction)

    assert metrics["accuracy"] == 0.75
    assert 0 < metrics["macro_f1"] < 1
    assert 0 < metrics["weighted_f1"] < 1


def test_existing_model_mode_supplies_validation_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    (model_directory / "model.safetensors").write_bytes(b"existing")
    (model_directory / "training_summary.json").write_text(
        '{"training_completed": true, "best_validation_macro_f1": 0.8}',
        encoding="utf-8",
    )
    tokenizer = object()
    model = object()
    validation_dataset = object()
    trainer = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        transformer_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        transformer_module.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        transformer_module,
        "prepare_transformer_dataset",
        lambda *args, **kwargs: validation_dataset,
    )

    def fake_trainer(**kwargs: object) -> object:
        captured.update(kwargs)
        return trainer

    monkeypatch.setattr(transformer_module, "Trainer", fake_trainer)
    reviews = pd.DataFrame(
        {"review_text": ["example"], "sentiment_label": ["neutral"]}
    )

    loaded_model, loaded_tokenizer, loaded_trainer, summary = (
        transformer_module.train_or_load_transformer(
            reviews,
            reviews,
            config=TransformerSentimentConfig(),
            model_directory=model_directory,
            checkpoint_directory=tmp_path / "checkpoints",
        )
    )

    assert captured["eval_dataset"] is validation_dataset
    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert loaded_trainer is trainer
    assert summary["training_completed"]
    assert summary["best_validation_macro_f1"] == 0.8
    assert not summary["training_performed_in_current_run"]


def test_transformer_report_uses_relative_artifact_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transformer_module, "find_project_root", lambda: tmp_path)
    predictions = _transformer_prediction_frame()
    report_path = tmp_path / "reports" / "transformer.json"
    prediction_path = tmp_path / "predictions" / "transformer.parquet"
    model_path = tmp_path / "models" / "transformer-v1"
    summary = {
        "training_completed": True,
        "best_checkpoint": str(model_path.parent / "checkpoints/checkpoint-10"),
        "selected_model_directory": str(model_path),
    }

    save_transformer_evaluation(
        predictions=predictions,
        metrics={"test": {"accuracy": 1.0}},
        training_summary=summary,
        config=TransformerSentimentConfig(model_version="transformer-v1"),
        dataset_version="dataset-v1",
        split_version="split-v1",
        predictions_path=prediction_path,
        report_path=report_path,
    )

    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert str(tmp_path) not in report_text
    assert report["predictions_path"] == "predictions/transformer.parquet"
    assert report["model_directory"] == "models/transformer-v1"
    assert report["training_summary"]["best_checkpoint"] == (
        "models/checkpoints/checkpoint-10"
    )


def test_transformer_metadata_migration_is_atomic_and_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transformer_module, "find_project_root", lambda: tmp_path)
    model_path = tmp_path / "models" / "transformer-v1"
    model_path.mkdir(parents=True)
    report_path = tmp_path / "reports" / "transformer.json"
    report_path.parent.mkdir(parents=True)
    checkpoint = tmp_path / "models" / "checkpoints" / "checkpoint-10"
    prediction_path = tmp_path / "predictions" / "transformer.parquet"
    (model_path / "training_summary.json").write_text(
        json.dumps(
            {
                "training_completed": True,
                "best_checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(
            {
                "dataset_version": "dataset-v1",
                "training_summary": {"best_checkpoint": str(checkpoint)},
                "predictions_path": str(prediction_path),
            }
        ),
        encoding="utf-8",
    )

    normalize_transformer_artifact_metadata(
        model_directory=model_path,
        report_path=report_path,
    )

    summary_text = (model_path / "training_summary.json").read_text()
    report_text = report_path.read_text()
    assert str(tmp_path) not in summary_text
    assert str(tmp_path) not in report_text
    assert not (model_path / "training_summary.json.tmp").exists()
    assert not report_path.with_suffix(".json.tmp").exists()


def _transformer_prediction_frame() -> pd.DataFrame:
    """Return one complete Transformer prediction row."""
    row = {
        "review_id": "review-1",
        "split_name": "validation",
        "sentiment_label": "positive",
        "rating": 5.0,
        "parent_asin": "product-1",
        "asin": "asin-1",
        "user_id": "user-1",
        "review_timestamp": pd.Timestamp("2023-01-01", tz="UTC"),
        "text_fingerprint": "fingerprint-1",
        "is_niche": False,
        "token_count": 3,
        "was_truncated": False,
        "predicted_label": "positive",
        "model_score": 0.9,
        "score_negative": 0.05,
        "score_neutral": 0.05,
        "score_positive": 0.9,
    }
    assert set(SENTIMENT_MANIFEST_COLUMNS).issubset(row)
    return pd.DataFrame([row])
