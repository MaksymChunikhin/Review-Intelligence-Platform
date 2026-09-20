"""Memory-bounded sampling helpers for large review datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sample_balanced_reviews(
    file_path: str | Path,
    *,
    sample_per_class: int | dict[Any, int],
    class_masks: dict[Any, Callable[[pd.Series], pd.Series]],
    columns: tuple[str, ...] = ("review_text", "rating"),
    class_column: str = "rating",
    batch_size: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return an equal-size reservoir sample for each logical class.

    Only one Parquet batch and the requested number of rows per class are kept
    in memory. ``sample_per_class`` can be one integer for equal quotas or a
    mapping with a quota for each class. ``class_masks`` maps class names to
    predicates over ``class_column``. The function deliberately does not load
    the complete dataset just to draw a sample.
    """
    if not class_masks:
        raise ValueError("class_masks must not be empty")
    if class_column not in columns:
        raise ValueError("class_column must be included in columns")

    if isinstance(sample_per_class, int):
        sample_sizes = {label: sample_per_class for label in class_masks}
    else:
        if set(sample_per_class) != set(class_masks):
            raise ValueError("sample_per_class mapping must match class_masks")
        sample_sizes = dict(sample_per_class)
    if any(size <= 0 for size in sample_sizes.values()):
        raise ValueError("sample sizes must be positive")

    reservoirs: dict[Any, list[tuple[Any, ...]]] = {
        label: [] for label in class_masks
    }
    seen: dict[Any, int] = {label: 0 for label in class_masks}
    rng = np.random.default_rng(random_state)

    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size, columns=list(columns)
    ):
        chunk = batch.to_pandas()
        class_values = chunk[class_column]

        for label, predicate in class_masks.items():
            selected = chunk.loc[predicate(class_values), list(columns)]
            for row in selected.itertuples(index=False, name=None):
                seen[label] += 1
                reservoir = reservoirs[label]
                target_size = sample_sizes[label]
                if len(reservoir) < target_size:
                    reservoir.append(row)
                    continue

                replacement = int(rng.integers(0, seen[label]))
                if replacement < target_size:
                    reservoir[replacement] = row

    insufficient = {
        label: count
        for label, count in seen.items()
        if count < sample_sizes[label]
    }
    if insufficient:
        raise ValueError(
            "Not enough rows for requested class sample: "
            + ", ".join(f"{label}={count}" for label, count in insufficient.items())
        )

    frames = [
        pd.DataFrame(reservoirs[label], columns=list(columns)).assign(sentiment=label)
        for label in class_masks
    ]
    return pd.concat(frames, ignore_index=True)
