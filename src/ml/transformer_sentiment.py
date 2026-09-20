"""Train and run versioned Transformer sentiment classifiers."""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.common.project import find_project_root
from src.ml.sentiment import SENTIMENT_LABELS, SENTIMENT_MANIFEST_COLUMNS


LABEL_TO_ID = {label: index for index, label in enumerate(SENTIMENT_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class TransformerSentimentConfig:
    """Configure one reproducible DistilBERT sentiment experiment."""

    model_version: str = "beauty_distilbert_rating_sentiment_1m_v2"
    pretrained_model: str = "distilbert-base-uncased"
    pretrained_revision: str = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    max_length: int = 256
    num_train_epochs: float = 1.0
    train_batch_size: int = 32
    evaluation_batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_fraction: float = 0.05
    weight_decay: float = 0.01
    evaluation_steps: int = 5_000
    logging_steps: int = 500
    random_state: int = 42
    keep_checkpoints_after_training: bool = False

    def __post_init__(self) -> None:
        """Reject settings that cannot form a valid training run."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.train_batch_size <= 0 or self.evaluation_batch_size <= 0:
            raise ValueError("Training and evaluation batch sizes must be positive")
        if self.evaluation_steps <= 0 or self.logging_steps <= 0:
            raise ValueError("Evaluation and logging steps must be positive")


def set_transformer_seed(random_state: int) -> None:
    """Seed Python, NumPy, PyTorch, and Transformers."""
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    set_seed(random_state)


def prepare_transformer_dataset(
    reviews: pd.DataFrame,
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """Tokenize review text and attach integer class labels."""
    required = {"review_text", "sentiment_label"}
    missing = required.difference(reviews.columns)
    if missing:
        raise ValueError(f"Transformer data is missing columns: {sorted(missing)}")
    frame = reviews[["review_text", "sentiment_label"]].copy()
    frame["labels"] = frame["sentiment_label"].map(LABEL_TO_ID)
    if frame["labels"].isna().any():
        raise ValueError("Transformer data contains an unknown sentiment label")
    frame["labels"] = frame["labels"].astype("int64")
    dataset = Dataset.from_pandas(
        frame[["review_text", "labels"]], preserve_index=False
    )

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["review_text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1_000,
        remove_columns=["review_text"],
        desc="Tokenizing reviews",
    )


def transformer_training_metrics(
    evaluation_prediction: Any,
) -> dict[str, float]:
    """Calculate validation metrics used for checkpoint selection."""
    logits = evaluation_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted = np.argmax(logits, axis=1)
    actual = evaluation_prediction.label_ids
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted")),
    }


def train_or_load_transformer(
    train_reviews: pd.DataFrame,
    validation_reviews: pd.DataFrame,
    *,
    config: TransformerSentimentConfig,
    model_directory: str | Path,
    checkpoint_directory: str | Path,
) -> tuple[
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    dict[str, Any],
]:
    """Train a missing model or load an existing versioned model."""
    model_path = Path(model_directory)
    checkpoint_path = Path(checkpoint_directory)
    set_transformer_seed(config.random_state)
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)
    use_tf32 = bool(torch.cuda.is_available())
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    existing_model = (model_path / "model.safetensors").is_file()
    training_summary_path = model_path / "training_summary.json"
    tokenizer_source = str(model_path) if existing_model else config.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if existing_model else config.pretrained_revision,
    )
    training_steps = int(
        np.ceil(len(train_reviews) / config.train_batch_size)
        * config.num_train_epochs
    )
    warmup_steps = int(training_steps * config.warmup_fraction)
    training_arguments = TrainingArguments(
        output_dir=str(checkpoint_path),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.evaluation_batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        eval_strategy="steps",
        eval_steps=config.evaluation_steps,
        save_strategy="steps",
        save_steps=config.evaluation_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=config.logging_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_tf32,
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        dataloader_pin_memory=torch.cuda.is_available(),
        eval_accumulation_steps=32,
        report_to="none",
        seed=config.random_state,
        data_seed=config.random_state,
    )

    if existing_model:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        validation_dataset = prepare_transformer_dataset(
            validation_reviews,
            tokenizer=tokenizer,
            max_length=config.max_length,
        )
        trainer = Trainer(
            model=model,
            args=training_arguments,
            eval_dataset=validation_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=transformer_training_metrics,
        )
        saved_training_summary = (
            json.loads(training_summary_path.read_text(encoding="utf-8"))
            if training_summary_path.is_file()
            else {"training_completed": True, "training_history_missing": True}
        )
        return model, tokenizer, trainer, {
            **saved_training_summary,
            "training_performed_in_current_run": False,
            "loaded_model": _artifact_reference(model_path),
            "selected_model_directory": _artifact_reference(model_path),
        }

    train_dataset = prepare_transformer_dataset(
        train_reviews,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    validation_dataset = prepare_transformer_dataset(
        validation_reviews,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model,
        revision=config.pretrained_revision,
        num_labels=len(SENTIMENT_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=transformer_training_metrics,
    )
    last_checkpoint = (
        get_last_checkpoint(str(checkpoint_path))
        if checkpoint_path.is_dir()
        else None
    )
    training_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    model_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    saved_training_summary = {
        "training_completed": True,
        "training_review_count": int(len(train_reviews)),
        "validation_review_count": int(len(validation_reviews)),
        "resumed_from_checkpoint": (
            _artifact_reference(last_checkpoint) if last_checkpoint else None
        ),
        "training_metrics": {
            key: float(value)
            for key, value in training_result.metrics.items()
            if isinstance(value, (int, float))
        },
        "best_checkpoint": (
            _artifact_reference(trainer.state.best_model_checkpoint)
            if trainer.state.best_model_checkpoint
            else None
        ),
        "best_validation_macro_f1": trainer.state.best_metric,
        "selected_model_directory": _artifact_reference(model_path),
    }
    if config.keep_checkpoints_after_training:
        saved_training_summary["checkpoint_status"] = "preserved"
    else:
        removed_checkpoints = _remove_transformer_checkpoints(checkpoint_path)
        saved_training_summary["checkpoint_status"] = (
            f"removed_after_final_model_save:{removed_checkpoints}"
        )
    _atomic_json_dump(saved_training_summary, training_summary_path)
    return model, tokenizer, trainer, {
        **saved_training_summary,
        "training_performed_in_current_run": True,
    }


def predict_with_transformer(
    reviews: pd.DataFrame,
    *,
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> pd.DataFrame:
    """Attach Transformer labels and uncalibrated model scores."""
    dataset = prepare_transformer_dataset(
        reviews,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    prediction = trainer.predict(dataset)
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    predicted_ids = np.argmax(probabilities, axis=1)
    result = reviews.copy()
    result["token_count"] = count_review_tokens(
        reviews["review_text"], tokenizer=tokenizer
    )
    result["was_truncated"] = result["token_count"] > max_length
    result["predicted_label"] = [ID_TO_LABEL[index] for index in predicted_ids]
    result["model_score"] = probabilities.max(axis=1)
    for index, label in ID_TO_LABEL.items():
        result[f"score_{label}"] = probabilities[:, index]
    return result


def count_review_tokens(
    review_text: pd.Series,
    *,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 1_000,
) -> np.ndarray:
    """Count original tokens before model truncation in bounded batches."""
    counts: list[int] = []
    values = review_text.astype(str).tolist()
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_length=True,
            verbose=False,
        )
        counts.extend(int(length) for length in encoded["length"])
    return np.asarray(counts, dtype=np.int32)


def save_transformer_evaluation(
    *,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    training_summary: dict[str, Any],
    config: TransformerSentimentConfig,
    dataset_version: str,
    split_version: str,
    predictions_path: str | Path,
    report_path: str | Path,
) -> dict[str, str]:
    """Persist Transformer predictions and an auditable JSON report."""
    prediction_path = Path(predictions_path)
    metric_path = Path(report_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_columns = list(SENTIMENT_MANIFEST_COLUMNS) + [
        "token_count",
        "was_truncated",
        "predicted_label",
        "model_score",
        "score_negative",
        "score_neutral",
        "score_positive",
    ]
    temporary_predictions = prediction_path.with_suffix(
        prediction_path.suffix + ".tmp"
    )
    predictions[prediction_columns].to_parquet(
        temporary_predictions, index=False
    )
    os.replace(temporary_predictions, prediction_path)
    normalized_training_summary = _normalize_training_summary(training_summary)
    existing_report = (
        json.loads(metric_path.read_text(encoding="utf-8"))
        if metric_path.is_file()
        else None
    )
    if existing_report is not None:
        expected_identity = {
            "dataset_version": dataset_version,
            "split_version": split_version,
            "model_version": config.model_version,
        }
        actual_identity = {
            "dataset_version": existing_report.get("dataset_version"),
            "split_version": existing_report.get("split_version"),
            "model_version": existing_report.get("model_config", {}).get(
                "model_version"
            ),
        }
        if actual_identity != expected_identity:
            raise ValueError("Existing Transformer report identity does not match")
    created_at = (
        str(existing_report["created_at_utc"])
        if existing_report is not None
        else datetime.now(timezone.utc).isoformat()
    )
    report = {
        "dataset_version": dataset_version,
        "split_version": split_version,
        "model_config": asdict(config),
        "label_definition": {
            "negative": [1, 2],
            "neutral": [3],
            "positive": [4, 5],
            "target_type": "rating-derived weak label",
        },
        "training_summary": normalized_training_summary,
        "metrics": metrics,
        "created_at_utc": created_at,
        "predictions_path": _artifact_reference(prediction_path),
    }
    selected_model = normalized_training_summary.get("selected_model_directory")
    if selected_model is not None:
        report["model_directory"] = selected_model
    temporary_report = metric_path.with_suffix(metric_path.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, metric_path)
    return {
        "predictions": str(prediction_path),
        "metrics_report": str(metric_path),
    }


def normalize_transformer_artifact_metadata(
    *,
    model_directory: str | Path,
    report_path: str | Path,
) -> dict[str, str]:
    """Atomically migrate persisted Transformer paths to portable references."""
    model_path = Path(model_directory)
    metric_path = Path(report_path)
    training_summary_path = model_path / "training_summary.json"
    if not training_summary_path.is_file() or not metric_path.is_file():
        raise FileNotFoundError("Transformer metadata artifacts are incomplete")

    training_summary = _normalize_training_summary(
        json.loads(training_summary_path.read_text(encoding="utf-8"))
    )
    training_summary["selected_model_directory"] = _artifact_reference(model_path)
    _atomic_json_dump(training_summary, training_summary_path)

    report = json.loads(metric_path.read_text(encoding="utf-8"))
    report["training_summary"] = _normalize_training_summary(
        report.get("training_summary", {})
    )
    report["training_summary"]["selected_model_directory"] = (
        _artifact_reference(model_path)
    )
    report["model_directory"] = _artifact_reference(model_path)
    if "predictions_path" in report:
        report["predictions_path"] = _artifact_reference(
            report["predictions_path"]
        )
    _atomic_json_dump(report, metric_path)
    return {
        "training_summary": str(training_summary_path),
        "metrics_report": str(metric_path),
    }


def _normalize_training_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Normalize only fields that are defined as filesystem references."""
    normalized = dict(summary)
    for key in (
        "resumed_from_checkpoint",
        "best_checkpoint",
        "best_checkpoint_during_training",
        "selected_model_directory",
        "loaded_model",
    ):
        value = normalized.get(key)
        if isinstance(value, str) and value:
            normalized[key] = _artifact_reference(value)
    return normalized


def _artifact_reference(path: str | Path) -> str:
    """Return a portable repository-relative reference when possible."""
    artifact_path = Path(path).resolve()
    try:
        project_root = find_project_root()
        return artifact_path.relative_to(project_root).as_posix()
    except (FileNotFoundError, ValueError):
        return str(artifact_path)


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    """Write formatted JSON through a same-directory temporary file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _remove_transformer_checkpoints(checkpoint_directory: Path) -> int:
    """Remove resumable checkpoints after the final model is safely saved."""
    if not checkpoint_directory.is_dir():
        return 0
    removed = 0
    for checkpoint in checkpoint_directory.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
            removed += 1
    try:
        checkpoint_directory.rmdir()
    except OSError:
        pass
    return removed
