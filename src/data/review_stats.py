"""Streaming statistics for review JSONL datasets.

The project datasets are too large to load into one pandas DataFrame.  This
module keeps the expensive JSONL scan in one place and uses bounded samples for
quantiles instead of retaining every text length in memory.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "rating",
    "title",
    "text",
    "parent_asin",
    "user_id",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
}


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from either the repo or notebooks directory."""
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "data").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not find the project root containing data/ and notebooks/")


def _update_reservoir(
    sample: list[int],
    values: list[int],
    seen: int,
    limit: int,
    rng: np.random.Generator,
) -> int:
    """Update a reservoir sample and return the new number of seen values."""
    for value in values:
        seen += 1
        if len(sample) < limit:
            sample.append(int(value))
        else:
            replacement = int(rng.integers(0, seen))
            if replacement < limit:
                sample[replacement] = int(value)
    return seen


def scan_reviews(
    file_path: str | Path,
    *,
    chunksize: int = 250_000,
    sample_size: int = 200_000,
    progress_every: int = 1_000_000,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Scan a review JSONL file once and return memory-bounded EDA statistics.

    ``max_rows`` is useful for a quick notebook smoke test before starting a
    full scan.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    stats = {
        "reviews": 0,
        "users": set(),
        "products": set(),
        "verified": 0,
        "missing_title": 0,
        "missing_text": 0,
        "helpful_votes": 0,
        "invalid_timestamps": 0,
    }
    rating_counts: Counter = Counter()
    year_counts: Counter = Counter()
    product_counts: Counter = Counter()
    rating_length: dict[Any, dict[str, Any]] = {}
    helpful_by_rating: dict[Any, dict[str, int]] = {}
    verified_by_rating: dict[Any, dict[str, int]] = {}
    rating_by_year: dict[Any, dict[str, float]] = {}
    rating_year_counts: Counter = Counter()
    sentiment_year: Counter = Counter()
    text_lengths: list[int] = []
    rating_length_seen: Counter = Counter()
    rng = np.random.default_rng(42)

    reader = pd.read_json(path, lines=True, chunksize=chunksize)
    for chunk in reader:
        if max_rows is not None:
            remaining = max_rows - stats["reviews"]
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
        missing = REQUIRED_COLUMNS - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        stats["reviews"] += len(chunk)
        stats["users"].update(chunk["user_id"].dropna())
        stats["products"].update(chunk["parent_asin"].dropna())
        stats["verified"] += int(chunk["verified_purchase"].fillna(False).astype(bool).sum())
        stats["missing_title"] += int(chunk["title"].isna().sum())
        stats["missing_text"] += int(chunk["text"].isna().sum())
        stats["helpful_votes"] += int(chunk["helpful_vote"].fillna(0).sum())

        ratings = chunk["rating"]
        rating_counts.update(ratings.dropna())
        product_counts.update(chunk["parent_asin"].dropna())

        dates = pd.to_datetime(chunk["timestamp"], unit="ms", utc=True, errors="coerce")
        stats["invalid_timestamps"] += int(dates.isna().sum())
        years = dates.dt.year
        year_counts.update(years.dropna().astype(int))

        lengths = chunk["text"].fillna("").astype(str).str.len()
        text_lengths_seen = _update_reservoir(
            text_lengths, lengths.tolist(), stats["reviews"] - len(chunk), sample_size, rng
        )

        for rating, group in chunk.assign(_text_length=lengths).groupby("rating", dropna=True):
            aggregate = rating_length.setdefault(rating, {"sum": 0, "count": 0, "sample": []})
            aggregate["sum"] += int(group["_text_length"].sum())
            aggregate["count"] += int(group["_text_length"].count())
            rating_length_seen[rating] = _update_reservoir(
                aggregate["sample"],
                group["_text_length"].tolist(),
                rating_length_seen[rating],
                max(10_000, sample_size // 5),
                rng,
            )

        for rating, group in chunk.groupby("rating", dropna=True):
            helpful = helpful_by_rating.setdefault(rating, {"sum": 0, "count": 0})
            helpful["sum"] += int(group["helpful_vote"].fillna(0).sum())
            helpful["count"] += int(group["helpful_vote"].notna().sum())

            verified = verified_by_rating.setdefault(rating, {"verified": 0, "count": 0})
            verified["verified"] += int(group["verified_purchase"].fillna(False).astype(bool).sum())
            verified["count"] += int(group["verified_purchase"].notna().sum())

        valid = chunk.loc[dates.notna(), ["rating"]].copy()
        valid["year"] = years.loc[dates.notna()].astype(int).to_numpy()
        for year, group in valid.groupby("year"):
            aggregate = rating_by_year.setdefault(year, {"sum": 0, "count": 0})
            aggregate["sum"] += float(group["rating"].sum())
            aggregate["count"] += int(group["rating"].count())
            for rating, count in group["rating"].value_counts().items():
                rating_year_counts[(year, rating)] += int(count)

        sentiment = pd.cut(
            chunk["rating"],
            bins=[0, 2, 3, 5],
            labels=["Negative", "Neutral", "Positive"],
        )
        for (year, label), count in (
            pd.DataFrame({"year": years, "sentiment": sentiment})
            .dropna()
            .groupby(["year", "sentiment"], observed=False)
            .size()
            .items()
        ):
            sentiment_year[(int(year), label)] += int(count)

        if progress_every and stats["reviews"] % progress_every == 0:
            print(f"Processed: {stats['reviews']:,} reviews", flush=True)

        if max_rows is not None and stats["reviews"] >= max_rows:
            break

    stats["text_lengths"] = text_lengths
    stats["rating_counts"] = rating_counts
    stats["year_counts"] = year_counts
    stats["product_counts"] = product_counts
    stats["rating_length"] = rating_length
    stats["helpful_by_rating"] = helpful_by_rating
    stats["verified_by_rating"] = verified_by_rating
    stats["rating_by_year"] = rating_by_year
    stats["rating_year_counts"] = rating_year_counts
    stats["sentiment_year"] = sentiment_year
    return stats
