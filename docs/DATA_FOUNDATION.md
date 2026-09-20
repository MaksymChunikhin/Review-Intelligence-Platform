# Amazon Beauty Data Foundation

## Verified Snapshot — 2026-08-18

The source window is unchanged: Amazon Reviews 2023,
`Beauty_and_Personal_Care`, from 2021-01-01 through 2023-09-13. The artifacts
were rebuilt and revalidated on 2026-08-18.

| Artifact | Verified result |
|---|---:|
| Canonical reviews | 8,986,368 rows |
| Required canonical fields | physically non-nullable; 0 nulls |
| Full category paths in the registry | 724 |
| Catalog rows / unique `parent_asin` values | 1,028,914 / 1,028,914 |
| Canonical reviews joined to catalog | 8,986,368 / 8,986,368 (100%) |

All five notebooks in `notebooks/01_data_foundation/` have been executed from
top to bottom against this snapshot. Every non-empty code cell has an execution
count and no saved error output.

## Current Dataset Version

```text
amazon_reviews_2023_beauty_2021_2023_v1
```

Source:

```text
Amazon Reviews 2023
Beauty_and_Personal_Care
2021-01-01 through 2023-09-13
```

The dataset manifest is stored at:

```text
config/datasets/amazon_reviews_2023_beauty_2021_2023_v1.json
```

It records logical identity, source paths, file sizes, checksums, formats,
known row counts, and the logical schema versions for canonical reviews,
Amazon products, and enriched reviews. Data and model binaries remain excluded
from Git.

Current schema contracts:

```text
amazon_review_v1
amazon_product_v1
amazon_enriched_review_v1
amazon_category_registry_v1
amazon_product_catalog_v1
```

## Build the Amazon Category Registry

```bash
python -m src.analytics.category_registry \
  config/datasets/amazon_reviews_2023_beauty_2021_2023_v1.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/category_registry.parquet \
  --report-path \
  reports/data_quality/amazon_reviews_2023_beauty_2021_2023_v1_category_registry.json
```

The builder scans every item-metadata row, preserves Amazon display labels,
creates shared NFKC + collapsed-whitespace + case-folded lookup keys, aggregates
exact product counts by complete category path, and writes the registry
atomically. The product-catalog join uses the same normalization function; a
display-path variant can no longer resolve differently in the two artifacts.

Current Beauty classification profile:

```text
product metadata rows:       1,028,914
distinct parent ASINs:       1,028,914
missing main_category:         102,896
missing category paths:              0
distinct full paths:               724
hierarchical path nodes:            756
path depth range:                   2–6
```

## Build the Complete Product Catalog

```bash
python -m src.analytics.product_catalog \
  config/datasets/amazon_reviews_2023_beauty_2021_2023_v1.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/product_catalog.parquet \
  --report-path \
  reports/data_quality/amazon_reviews_2023_beauty_2021_2023_v1_product_catalog.json
```

Current catalog and join results:

```text
catalog products:                1,028,914
catalog products with reviews:    550,351
canonical reviews:              8,986,368
reviews matched to metadata:    8,986,368
review join coverage:                100%
missing product titles:                 54
missing stores:                     50,713
missing prices:                    648,234
invalid price values:                  83
```

The catalog retains products with incomplete optional metadata and explains
limitations through `metadata_quality_flags`. It does not silently parse price
strings such as `from 28.00` as exact prices.

## Verify Registered Artifacts

Fast existence and size verification:

```bash
python -m src.ingestion.dataset_manifest \
  config/datasets/amazon_reviews_2023_beauty_2021_2023_v1.json
```

Strict checksum and record-count verification:

```bash
python -m src.ingestion.dataset_manifest \
  config/datasets/amazon_reviews_2023_beauty_2021_2023_v1.json \
  --checksums \
  --record-counts
```

Strict verification scans large source files and is not intended for every
interactive notebook run.

## Build Canonical Reviews

Smoke test without replacing the versioned full artifact:

```bash
python -m src.preprocessing.reviews \
  config/datasets/amazon_reviews_2023_beauty_2021_2023_v1.json \
  /tmp/beauty_canonical_smoke.parquet \
  --report-path /tmp/beauty_canonical_smoke_report.json \
  --max-rows 10000
```

Full versioned build:

```bash
python -m src.preprocessing.reviews \
  config/datasets/amazon_reviews_2023_beauty_2021_2023_v1.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/reviews.parquet \
  --report-path \
  reports/data_quality/amazon_reviews_2023_beauty_2021_2023_v1.json
```

The build:

- reads JSONL in chunks;
- validates rating, timestamp, date window, ASINs, and review text;
- parses optional integers with explicit bounds, so fractional or oversized
  `helpful_vote` values are reported and nulled instead of crashing a build;
- preserves `asin` and `parent_asin`;
- creates deterministic source-position review IDs;
- writes the required columns as physically non-nullable Arrow fields;
- removes exact duplicate records with a bounded-memory DuckDB operation,
  including otherwise identical anonymous reviews;
- atomically replaces the final artifact only after a successful build;
- reconciles input, rejected, duplicate, and output counts.

The physically required, non-nullable fields are `review_id`,
`dataset_version`, `dataset_category`, `source_record_index`, `asin`,
`parent_asin`, `rating`, `review_title`, `review_body`, `review_text`,
`review_timestamp`, and `source_timestamp_ms`. The rebuilt Parquet schema marks
all 12 fields as non-nullable and the exact full-table validation reports zero
null values in each. `review_timestamp` is stored as
`timestamp[us, tz=UTC]`.

## Generate Exact Dataset Profile

```bash
python -m src.analytics.dataset_profile \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/reviews.parquet \
  --output-path \
  reports/data_quality/amazon_reviews_2023_beauty_2021_2023_v1_profile.json
```

The profile calculates exact counts with the session timezone fixed to UTC.

## Validate the Notebooks

```bash
jupyter nbconvert \
  --execute \
  --to notebook \
  --inplace \
  notebooks/01_data_foundation/01_beauty_source_audit.ipynb \
  --ExecutePreprocessor.timeout=600
```

Run the same command for `02_cleaning_validation.ipynb`,
`03_amazon_contract_validation.ipynb`, `04_amazon_category_registry.ipynb`, and
`05_catalog_and_scope_validation.ipynb`, in that order. The saved 2026-08-18
executions are clean for all five notebooks. Notebooks are the human inspection
and explanation layer; reusable build and profiling logic remains in `src/`.

## Current Reconciliation

```text
filtered input rows:              9,084,957
valid rows before deduplication:  9,084,957
exact duplicate rows removed:       98,589
canonical output rows:            8,986,368
```

Current canonical identifiers:

```text
ASINs:          830,332
parent ASINs:   550,351
users:        5,180,541
```

These values are generated by the current versioned pipeline and reports. Any
future dataset or preprocessing change must produce a new dataset version or an
explicitly reviewed manifest update.
