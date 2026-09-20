# Amazon Category Onboarding Notebooks

This directory defines the contract for future parameterized validation
notebooks used to add an Amazon Reviews 2023 product category. As of 2026-08-18
it contains this specification only; the four templates listed below have not
been implemented. Do not treat their names as executable current notebooks.

The user is expected to inspect the data visually. Automated checks cannot
replace examination of category paths, products, review examples, discovered
aspects, and model errors.

Each template must explain the purpose of its checks and the interpretation of
important outputs in English and Russian. This makes category onboarding both
a reproducible pipeline review and a guided learning workflow.

Planned templates:

```text
01_category_profile.ipynb
02_taxonomy_review.ipynb
03_model_evaluation.ipynb
04_launch_readiness.ipynb
```

## Verified Beauty Reference Baseline — 2026-08-18

The five clean-run notebooks in `../01_data_foundation/` establish the reference
contract that a new category must reproduce or explicitly amend:

| Check | Beauty reference result |
|---|---:|
| Canonical reviews | 8,986,368 |
| Required canonical fields | physically non-nullable; 0 nulls |
| Complete category paths in registry | 724 |
| Catalog rows / unique `parent_asin` values | 1,028,914 / 1,028,914 |
| Review-to-catalog join | 8,986,368 / 8,986,368 (100%) |

These are reference values for the versioned Beauty dataset, not universal
thresholds for another category. A new category must record its own counts,
hashes, and accepted deviations.

## Required Parameters

Each template receives configuration rather than hard-coded paths:

```text
category_config_path
dataset_version
pipeline_version
output_report_directory
```

## 01 — Category Profile

Visual and statistical review of:

- source and processed row counts;
- date range;
- rating and review-volume distributions;
- `asin` and `parent_asin` cardinality;
- review-to-metadata join coverage;
- `main_category` and category-path coverage;
- products and review counts by category node;
- missing metadata and suspicious values;
- representative reviews and products;
- language and text-length distributions;
- duplicate and near-duplicate indicators.

## 02 — Taxonomy Review

Human review of:

- discovered aspects, benefits, complaints, needs, and use cases;
- canonical aspect names and aliases;
- empirical support and product coverage;
- overlapping or overly broad aspects;
- missing important category concepts;
- evidence examples and ambiguous cases.

Any probability claim must also document the sample frame and inclusion
probability. The corrected Beauty reference uses review-level stratified SRSWOR
without a product cap: 1,500 discovery reviews from 904 observed products, with
weights summing to the 114,025-review eligible population. A product-diversity
cap may be used for a targeted diagnostic sample only when those rows are
explicitly unweighted.

## 03 — Model Evaluation

Category-specific checks for:

- overall sentiment;
- aspect extraction;
- aspect sentiment;
- complaint and intent classification;
- error slices by product, time, rating, and review length;
- comparison with currently approved models;
- decision to reuse, calibrate, fine-tune, or replace a model.

For aspect extraction, keep probability and diagnostic components separate. In
the current Beauty reference, the evaluation sample has 160 representative
rows whose weights sum to the 112,525-review holdout population and 80 targeted,
unweighted rows. All 240 annotations are pending, so no extraction-quality
metric is accepted yet.

## 04 — Launch Readiness

Final review of:

- complete offline population and aggregate reconciliation;
- seller and competitor scope behavior;
- analytics and trend tables;
- evidence retrieval quality;
- Gemini tool answers and unsupported-claim behavior;
- data, taxonomy, model, embedding, and index versions;
- documented limitations.

External work must be visible as a gate, not implied by a prepared local plan.
For the current Beauty aspect refresh, the V3 plan contains 15 batches of 100
reviews but has 0 saved Gemini responses; external Gemini execution and the new
human taxonomy review remain pending. Taxonomy V1 labels and aliases are usable
qualitatively, while its legacy weighted prevalence fields are invalid.

Generated executions should be saved as versioned reports outside the notebook
source directory, for example:

```text
reports/categories/beauty_and_personal_care/<dataset_version>/
reports/categories/electronics/<dataset_version>/
```

The templates will remain shared once implemented. Reports are
category-specific and must record dataset, taxonomy, model, sampling, and
lineage identities.

See the repository [notebook organization](../README.md),
[data-foundation contract](../../docs/DATA_FOUNDATION.md), and
[aspect-discovery status](../../docs/ASPECT_DISCOVERY.md) for the current
reference workflow.
