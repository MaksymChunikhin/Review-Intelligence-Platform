"""Filter a review JSONL.GZ dataset by UTC calendar year."""

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path


def filter_reviews(input_path: str | Path, output_path: str | Path, start_year: int, end_year: int) -> tuple[int, int]:
    """Stream input to output and return (total_reviews, selected_reviews)."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_reviews = 0
    selected_reviews = 0

    with gzip.open(input_path, "rt", encoding="utf-8") as input_file, output_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                review = json.loads(line)
                timestamp = int(review["timestamp"])
                year = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).year
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid review at line {line_number}: {exc}") from exc

            total_reviews += 1
            if start_year <= year <= end_year:
                output_file.write(json.dumps(review, ensure_ascii=False) + "\n")
                selected_reviews += 1

    return total_reviews, selected_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2023)
    args = parser.parse_args()

    total, selected = filter_reviews(args.input_path, args.output_path, args.start_year, args.end_year)
    print(f"Total reviews: {total:,}")
    print(f"Selected reviews: {selected:,}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
