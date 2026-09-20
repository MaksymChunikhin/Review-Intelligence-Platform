# Notebook Organization

Notebooks in this repository are grouped by research responsibility. Their
numbering is local to each directory; it is not a global project-stage counter.

```text
notebooks/
├── 01_data_foundation/
├── 02_sentiment/
├── 03_aspects/
└── category_onboarding/
```

`04_analytics/` and `05_retrieval/` remain planned; they are not present in the
current notebook tree.

## Current Notebooks

```text
01_data_foundation/
├── 01_beauty_source_audit.ipynb
├── 02_cleaning_validation.ipynb
├── 03_amazon_contract_validation.ipynb
├── 04_amazon_category_registry.ipynb
└── 05_catalog_and_scope_validation.ipynb

02_sentiment/
├── 01_tfidf_baseline.ipynb
├── 02_distilbert_training.ipynb
└── 03_error_analysis.ipynb

03_aspects/
├── 01_taxonomy_review.ipynb
└── 02_extraction_evaluation_sample.ipynb
```

These files were moved from the original flat `01–08` sequence after the
notebook audit. Acceptance is recorded per notebook after its rewrite and clean
execution; model-specific details remain in the
[notebook audit](../docs/NOTEBOOK_AUDIT.md).

## Verified Run State — 2026-08-18

| Notebook | Status |
|---|---|
| `01_data_foundation/01_beauty_source_audit.ipynb` | Clean execution; 8,986,368-row canonical source reconciled |
| `01_data_foundation/02_cleaning_validation.ipynb` | Clean execution; physical required-field non-nullability validated |
| `01_data_foundation/03_amazon_contract_validation.ipynb` | Clean execution; schema and contract checks passed |
| `01_data_foundation/04_amazon_category_registry.ipynb` | Clean execution; 724 complete category paths validated |
| `01_data_foundation/05_catalog_and_scope_validation.ipynb` | Clean execution; 1,028,914 unique catalog parents and 100% review join validated |
| `02_sentiment/01_tfidf_baseline.ipynb` | Current model-refresh state is recorded in `docs/NOTEBOOK_AUDIT.md` |
| `02_sentiment/02_distilbert_training.ipynb` | Current model-refresh state is recorded in `docs/NOTEBOOK_AUDIT.md` |
| `02_sentiment/03_error_analysis.ipynb` | Current model-refresh state is recorded in `docs/NOTEBOOK_AUDIT.md` |
| `03_aspects/01_taxonomy_review.ipynb` | Clean execution; Taxonomy V1 retained only as a qualitative dictionary; legacy weighted prevalence invalid |
| `03_aspects/02_extraction_evaluation_sample.ipynb` | Clean execution; corrected V2 sample validated; all 240 gold annotations pending |

All five data-foundation notebooks and both aspect notebooks have execution
counts for every non-empty code cell and contain no saved error output. The
single row for each notebook above is intentional: historical V1 aspect output
is described inside the refreshed notebooks instead of being listed as a
second current status.

The obsolete generic domain-detection notebook was removed after its audit and
replaced by `04_amazon_category_registry.ipynb`. Generic domain inference is
not part of the Amazon-focused architecture.

## Notebook Responsibilities

A notebook may:

- inspect representative source records;
- display row counts, schemas, null rates, distributions, and examples;
- visualize analytical results;
- compare experimental methods;
- interpret results and document limitations;
- validate versioned artifacts produced by reusable code.

Notebook narrative is educational and bilingual: English comes first for
professional terminology, followed by a detailed Russian explanation. Code
comments follow the same order when they explain a non-obvious decision. We do
not translate self-evident syntax line by line.

A notebook must not be the only implementation of:

- ingestion or cleaning pipelines;
- schemas and validation contracts;
- catalog construction;
- dataset splitting;
- model inference;
- analytical functions;
- API or LLM clients.

Reusable implementation belongs in `src/`. Small deterministic checks belong
in `tests/`.

## Dependency Order

Notebook execution and acceptance follow this dependency order:

```text
01_data_foundation
        ↓
Amazon category registry and full product catalog
        ↓
02_sentiment reevaluation
        ↓
03_aspects
        ↓
04_analytics
        ↓
05_retrieval and Gemini evaluation
```

The data-foundation and aspect notebooks in the current tree have been refreshed
and clean-run. Later analytics and retrieval notebooks should not be created
until their input artifacts and evaluation gates are ready.

## Adding an Amazon Category

Do not copy the complete notebook tree for a new category.

A new category normally adds:

```text
category configuration
dataset manifest
category taxonomy
evaluation artifacts
optional category-specific models
generated onboarding reports
```

The shared pipeline remains in `src/`. The planned parameterized notebooks are
described in the
[category-onboarding README](category_onboarding/README.md); generated category
reports must remain versioned outside the shared notebook sources.

All notebooks must follow the repository
[code style](../docs/CODE_STYLE.md).
