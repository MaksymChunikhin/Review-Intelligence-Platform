# Notebook Audit and Refresh Record — 10 Current Notebooks

## 1. Audit Status

Initial audit date: 2026-08-12

Full refresh date: 2026-08-18

Scope:

- the original notebooks `01–08`, now organized under
  `notebooks/01_data_foundation/` and `notebooks/02_sentiment/`;
- the two current notebooks under `notebooks/03_aspects/`;
- reusable code called by those notebooks;
- generated datasets and model artifacts referenced by them;
- compatibility with `PLAN.md` and `docs/CODE_STYLE.md`.

The 2026-08-12 audit itself was read-only. The 2026-08-18 refresh subsequently
rewrote the current notebooks, rematerialized their versioned inputs, and began
a dependency-ordered clean execution of all ten notebooks. Sections describing
the original defects are retained as historical rationale; the table below and
each rewritten notebook section state current status explicitly.

Current refresh state:

| Notebook | 2026-08-18 result |
|---|---|
| `01_data_foundation/01_beauty_source_audit.ipynb` | clean-run verified |
| `01_data_foundation/02_cleaning_validation.ipynb` | clean-run verified |
| `01_data_foundation/03_amazon_contract_validation.ipynb` | clean-run verified |
| `01_data_foundation/04_amazon_category_registry.ipynb` | clean-run verified |
| `01_data_foundation/05_catalog_and_scope_validation.ipynb` | clean-run verified |
| `02_sentiment/01_tfidf_baseline.ipynb` | clean-run verified; corrected V2/V3 reports materialized |
| `02_sentiment/02_distilbert_training.ipynb` | clean run in progress; `candidate_pending_human_review`; exact metrics not yet materialized |
| `02_sentiment/03_error_analysis.ipynb` | pending upstream V2 predictions and clean run |
| `03_aspects/01_taxonomy_review.ipynb` | clean-run verified; Taxonomy V1 is qualitative only |
| `03_aspects/02_extraction_evaluation_sample.ipynb` | clean-run verified; all 240 human annotations pending |

The foundation has 8,986,368 canonical reviews, 724 registry paths, and
1,028,914 unique catalog parents with 100% review-join coverage. The corrected
aspect sample contains 1,500 probability-sampled reviews with weights summing
to 114,025; the V2 extraction queue contains 160 weighted representative and 80
unweighted targeted rows, all pending human annotation. Eight notebooks have a
completed clean run; the DistilBERT and refreshed Error Analysis notebooks are
the two remaining execution gates.

## 2. Original Executive Verdict — 2026-08-12

The original work was a useful research prototype, but none of the eight
notebooks was accepted as production-ready at that audit boundary.

```text
01  REFACTOR
02  REPLACE PIPELINE; KEEP VALIDATED CLEANING DECISIONS
03  REPLACE EVALUATION PROTOCOL
04  REPLACE TRAINING NOTEBOOK
05  REFACTOR AFTER 03/04 ARE REBUILT
06  REPLACE WITH AMAZON DATA CONTRACTS
07  REMOVE DOMAIN DETECTION; REPLACE WITH AMAZON CATEGORY REGISTRY
08  REPLACE WITH FULL PRODUCT CATALOG AND WORKSPACE SCOPE
```

The main problem is not the choice of libraries or models. It is artifact
lineage and evaluation validity: several notebooks depend on samples or split
indices that are not independently identifiable, some cannot run from a fresh
state, and the product-selection result uses only a 5,000-review sample.

Existing files must be preserved until their replacements have been validated.
They should be treated as candidate or quarantined artifacts, not deleted.

## 3. Dataset Facts and 2026-08-18 Correction

Current V1 working source:

```text
Amazon Reviews 2023
dataset category: Beauty_and_Personal_Care
working window: 2021-01-01 through 2023-09-13
```

Observed current pipeline counts:

```text
filtered JSONL reviews:              9,084,957
cleaned Parquet reviews:             8,986,368
rows removed by current dedup step:     98,589
unique parent ASINs in clean data:     550,351
item-metadata rows:                  1,028,914
review parent ASINs matched to meta:   550,351
current product-level join coverage:    100.00%
```

The join coverage is strong for the current review population, but it must be
recomputed by the future versioned pipeline rather than copied as a permanent
assumption.

The pre-refactor clean Parquet stored:

```text
timestamp   → naive timestamp[ns]
review_date → UTC timestamp[ns]
```

The rebuilt canonical artifact now keeps one unambiguous
`timestamp[us, tz=UTC]` field named `review_timestamp` plus
`source_timestamp_ms`. All 12 mandatory Arrow fields are physically
non-nullable and contain zero nulls.

## 4. Critical Cross-Notebook Findings

### C-01 — The sentiment split is not a durable data artifact

`sentiment_1m_split.npz` stores only integer row positions. It does not store:

- review IDs;
- sampled review IDs;
- dataset checksum/version;
- sample checksum;
- split strategy version.

The indices remain meaningful only if reservoir sampling recreates exactly the
same sample from the same file in the same order with the same implementation.
The notebook does not validate that assumption before reusing the split.

Resolution (2026-08-18): both current split manifests are Parquet tables keyed
by deterministic `review_id` and carry product, ASIN, user, UTC timestamp,
weak-label, text-fingerprint, and niche membership fields. Versioned reports
record their exact hashes and membership deltas from the legacy splits.

### C-02 — Notebook 04 cannot reliably run from a fresh state

Two concrete execution defects exist:

1. The sample column `sentiment` is renamed to `label_name`, but the split-
   creation branch later stratifies on `reviews["sentiment"]`. That branch will
   fail when the saved split file is absent.
2. When a saved model exists, the notebook loads `model` but does not create
   `trainer`. The evaluation cell later calls `trainer.predict(...)` and
   `trainer.save_model(...)`.

The currently committed outputs were produced through a state in which those
branches happened not to fail. This is not clean-run reproducibility.

Resolution status (2026-08-18): reusable train/load/predict paths now construct
a valid Trainer in either branch and support checkpoint resume. Their clean
RAPIDS-kernel execution under `beauty_distilbert_rating_sentiment_1m_v2` is in
progress; this finding closes only after Notebook 02 materializes and validates
the final model, predictions, and report.

### C-03 — Original sentiment metrics had leakage and representativeness risks

The original protocol used a random row-level split. It did not prevent overlap
between train and test by:

- repeated review text;
- user;
- `asin`;
- `parent_asin`;
- time period.

Original Notebook 02 explicitly retained repeated generic review texts and said
they would be handled during splitting, but Notebooks 03 and 04 did not
implement that handling.

The test set is also class-balanced. It is useful for per-class comparison but
does not represent the natural rating distribution. The rebuilt evaluation must
include both:

```text
balanced diagnostic test set
natural-distribution product-aware or temporal test set
```

All metrics reported before the UTC-stable rebuild remain historical and cannot
be promoted to the corrected experiment.

Current correction (2026-08-18): both product-aware manifests were rebuilt
with an explicit UTC cutoff as `beauty_rating_sentiment_split_v2` and
`beauty_rating_sentiment_split_1m_v3`. The newly versioned TF-IDF models and
predictions have been evaluated on those memberships with natural and balanced
views and percentile bootstrap intervals over whole `parent_asin` clusters.
The 1M results are in
`reports/model_evaluation/beauty_tfidf_rating_sentiment_1m_v3.json`.
`beauty_distilbert_rating_sentiment_1m_v2` is
`candidate_pending_human_review`; its exact automated metrics and refreshed
error analysis are pending clean runs of sentiment Notebooks 02 and 03. Legacy
models, metrics, and the completed 158-row diagnostic remain historical only.

### C-04 — The Product Catalog result is sample-only

Notebook 08 reads `reviews_universal_sample.parquet`, which contains only the
first 5,000 cleaned reviews and 4,602 products. It then builds a catalog only
for those products.

The saved hair-conditioner result contains:

```text
matched products: 25
selected reviews: 27
```

This is not a niche review population and cannot support sentiment, aspect, or
competitor analytics. The full metadata catalog must be built first, and all
eligible reviews must be selected by product scope from the full processed
population or queried from precomputed analytical tables.

Resolution (2026-08-18): the catalog now contains all 1,028,914 unique
`parent_asin` values, with zero duplicate parents and 100% reconciliation of
8,986,368 canonical reviews. Hair Conditioners resolves by an approved category
path to 13,985 catalog products and 118,804 reviews.

### C-05 — Generic domain detection conflicts with the new product scope

Notebook 07 detects a domain from hard-coded keyword lists. The dataset category
is already known from the Amazon Reviews 2023 source and product metadata.

The notebook's confidence score is an ad hoc keyword margin, not a calibrated
probability. Its synthetic tests reuse the same keywords that define the
detector. This stage should not remain in the pipeline.

Resolution: the generic domain notebook was removed and replaced by the full
Amazon category registry and explicit versioned niche definitions.

### C-06 — Artifact lineage is incomplete

Generated artifacts do not consistently record:

- dataset version and file checksums;
- code/pipeline version;
- schema version;
- category and time window;
- row counts before and after each rule;
- parent artifact identifiers;
- taxonomy/model/index versions.

Absolute local filesystem paths are stored in JSON metadata. They should be
replaced by repository-relative logical artifact URIs or manifest references.

Resolution (2026-08-18): the dataset manifest, model/aspect validity ledgers,
versioned configs, reports, and boundary identity checks now bind current
artifacts by logical version and SHA-256. Historical invalidity remains
explicit instead of being overwritten.

### C-07 — Several notebook conclusions are detached from executable output

Notebooks 02, 03, and 07 have cleared code-cell execution counts and outputs,
while their conclusions state concrete results. Those claims may be historically
correct, but the committed notebooks do not currently demonstrate them.

All completed notebooks must run from a clean kernel and derive conclusions
from the current run or a versioned metrics artifact.

Resolution status (2026-08-18): eight current notebooks are clean-executed in
dependency order with sequential execution counts and no saved error outputs.
The DistilBERT notebook is running and refreshed Error Analysis remains pending
its predictions; the all-ten-notebook gate is therefore still open.

## 5. Notebook-by-Notebook Audit

## 01 — Data Exploration

Verdict: `REFACTOR` and `MOVE REUSABLE CODE TO src/`

Current path:
`notebooks/01_data_foundation/01_beauty_source_audit.ipynb`

Refactoring status: `COMPLETED FOR SOURCE/CANONICAL REVIEW SCOPE`

The notebook now reads the versioned manifest, inspects bounded source samples,
validates preprocessing reconciliation, calculates exact full-population
statistics through reusable code, contains one imports cell, and executes from
a clean kernel. Product category-path and join analysis remains assigned to the
full catalog validation notebook rather than being duplicated here.

Useful work to retain:

- correct Beauty review and metadata files;
- explicit `parent_asin` metadata join demonstration;
- full working-window rating and year counts;
- verified-purchase and helpful-vote exploration;
- bounded sampling for text-length quantiles;
- product concentration analysis;
- clear distinction between rating-derived groups and model sentiment.

Required changes:

- read paths and the time window from a dataset manifest;
- report raw, filtered, cleaned, and joined populations separately;
- add schema, null, duplicate, identifier, category, language, and metadata-
  coverage analysis;
- inspect `asin` and `parent_asin` cardinality and variation structure;
- add Amazon category-path coverage and product/niche distributions;
- remove duplicated verified-purchase cells;
- rewrite explanatory Markdown and meaningful comments in the agreed
  bilingual style: English first, then a detailed Russian explanation;
- make every chart interpretation explicit in Markdown;
- save a versioned EDA/data-quality report rather than relying on notebook
  output only.

Reusable utility issue:

`src/data/review_stats.py` calls itself memory-bounded but keeps full user sets,
product sets, and product counters. With millions of users/products this state
is not strictly bounded. Exact cardinalities should use a deliberate scalable
strategy, and sampled statistics should be labelled as estimates.

Target role:

```text
Demonstrate and explain the versioned Beauty snapshot and data-quality report.
Do not implement the ingestion pipeline inside the notebook.
```

## 02 — Data Cleaning and Preparation

Verdict: `VERIFIED UNDER CURRENT PLAN` (rewritten 2026-08-12).

Current path:
`notebooks/01_data_foundation/02_cleaning_validation.ipynb`

Resolution:

- production cleaning, validation, reconciliation, and disk-backed exact
  deduplication now live in `src.preprocessing.reviews`;
- the notebook calls that reusable pipeline on a readable temporary fixture
  rather than implementing a second cleaning path;
- the saved full Beauty artifact is checked against the canonical names and
  Arrow types, with exact null counts for mandatory fields;
- deterministic source-position review IDs, UTC conversion, text handling,
  optional-field behavior, primary drop reasons, and row reconciliation are
  covered by executable assertions and regression tests;
- the notebook has one imports cell, follows the bilingual learning style, and
  executes from a clean kernel.

Useful decisions retained from the original notebook:

- chunked source processing;
- atomic replacement idea;
- UTC review-date creation;
- title/body combination into `review_text`;
- explicit rating and empty-text validation;
- preservation of short and long reviews;
- distinction between repeated text and exact duplicate records.

Original required changes, now resolved:

- move cleaning, validation, and deduplication to tested `src/` modules;
- define canonical Amazon review schema before writing output;
- add `review_id`, `dataset_version`, `dataset_category`, `asin`, and
  `parent_asin` contracts;
- make every drop rule explicit in a data-quality report;
- replace repeated full-file rewrites and repeated analytical scans;
- avoid Python sets containing every review text or complete record key;
- define stable dedup keys before filling missing identifiers with empty
  strings;
- preserve deterministic input order or document the new order;
- validate the final artifact after deduplication, not only before it;
- remove hard-coded conclusion counts;
- add small fixture-based regression tests.

Original inconsistencies removed by the rewrite:

- `review_text` is already created in the main write pass and then recreated in
  a second full rewrite;
- one validation counts empty `text_clean`, even though records with an empty
  body but non-empty title are intentionally valid through `review_text`;
- the `MAX_ROWS` condition counts retained rows rather than consumed source
  rows, which makes smoke-test scope ambiguous when rows are dropped.

Target role:

```text
Notebook validates and explains a reusable cleaning pipeline run; production
logic lives outside the notebook.
```

## 03 — TF-IDF Sentiment Baseline

Verdict: `UTC-STABLE REFRESH VERIFIED` (clean-run 2026-08-18)

Current path: `notebooks/02_sentiment/01_tfidf_baseline.ipynb`

Current implementation:

- TF-IDF + Logistic Regression as an interpretable baseline;
- rating-derived three-class weak-label definition;
- deterministic review-ID split manifest with product, user, ASIN, date, and
  normalized-text fingerprints;
- disjoint pre-cutoff train, validation, held-out Beauty-parent, and held-out
  Conditioner-parent groups, plus a later temporal test for sampled training
  products;
- no repeated normalized text within or across accepted experiment splits;
- balanced training and validation together with natural-distribution and
  balanced evaluation views;
- validation-only regularization selection;
- accuracy, macro F1, weighted F1, per-class metrics, confusion matrices, and
  95% percentile bootstrap intervals that resample complete `parent_asin`
  clusters;
- persisted vectorizer, classifier, experiment configuration, predictions,
  split manifest, and JSON metrics report;
- reusable implementation and fixture-based regression tests in
  `src.ml.sentiment` and `tests/test_sentiment.py`;
- two new models: `beauty_tfidf_rating_sentiment_v2` on the corrected 300k
  reference and `beauty_tfidf_rating_sentiment_1m_v3` on 999,999 train rows;
- clean execution in the RAPIDS Python 3.11.15 environment with no saved error
  output.

Current 1M results:

```text
Held-out Beauty parents, natural:  accuracy 0.84731; macro F1 0.72757
Held-out Beauty parents, balanced: accuracy 0.78644; macro F1 0.78717
Later 2023 reviews, natural:        accuracy 0.84722; macro F1 0.72717
Later 2023 reviews, balanced:       accuracy 0.78382; macro F1 0.78435
Held-out Conditioner parents, natural:  accuracy 0.86288; macro F1 0.72051
Held-out Conditioner parents, balanced: accuracy 0.77778; macro F1 0.77808
```

Natural accuracy is higher because positive reviews dominate the real
distribution. It must not be compared with balanced macro F1 as though they
were the same measurement. The target remains a rating-derived weak label, and
model scores remain uncalibrated. Exact confidence intervals, cluster counts,
per-class metrics, and confusion matrices are in
`reports/model_evaluation/beauty_tfidf_rating_sentiment_1m_v3.json`. The
Conditioner test holds out 658 parent products, not the whole niche.

Target role:

```text
Establish a trustworthy classical baseline on the same corrected evaluation
sets used by Transformer experiments.
```

## 04 — Transformer Sentiment

Verdict: `CLEAN RUN IN PROGRESS; CANDIDATE_PENDING_HUMAN_REVIEW`

Current path: `notebooks/02_sentiment/02_distilbert_training.ipynb`

Current refresh target:

- model version `beauty_distilbert_rating_sentiment_1m_v2`;
- corrected split `beauty_rating_sentiment_split_1m_v3` with an explicit UTC
  cutoff;
- 999,999 balanced training reviews with 333,333 rows per rating-derived class;
- the same validation, held-out Beauty-parent, held-out Conditioner-parent, and
  later-review IDs as the verified 1M TF-IDF experiment;
- a separate later-2023 test for products represented in training;
- validation every 5,000 steps and selection by equal-class macro F1;
- independent train, load, and inference paths with resume checkpoints;
- saved model, tokenizer, training summary, predictions, token counts,
  truncation flags, and evaluation report;
- natural and balanced metrics, per-class results, and 95% percentile bootstrap
  intervals over complete `parent_asin` clusters;
- reusable training and inference code in `src.ml.transformer_sentiment` with
  regression tests.

The GPU clean run is still in progress. Consequently this document asserts no
new DistilBERT accuracy, F1, calibration, or error-analysis value. Exact
automated results become current only when Notebook 02 completes and validates
the versioned model, predictions, and JSON report, and Notebook 03 clean-runs on
those exact predictions.

The candidate status is `candidate_pending_human_review`, not accepted. The
historical V1 model and its reported metrics remain in the artifact-validity
ledger for audit only. The completed 158-row manual review is bound to legacy V1
predictions; it must not be copied or treated as human evidence for V2. The
Conditioner evaluation holds out a defined set of parent products from the
niche, not the whole Hair Conditioners niche; other Conditioner parent products
remain in training.

Target role:

```text
Train one newly versioned candidate on the corrected manifest, evaluate it on
immutable tests, obtain identity-bound human review, and publish an acceptance
decision only through the model registry workflow.
```

## 05 — Error Analysis

Verdict: `PENDING UPSTREAM V2 PREDICTIONS AND CLEAN RUN`

Current path: `notebooks/02_sentiment/03_error_analysis.ipynb`

Current refresh contract:

- removed model inference and every cross-notebook variable dependency;
- load only predictions for
  `beauty_distilbert_rating_sentiment_1m_v2` and
  `beauty_rating_sentiment_split_1m_v3`, then restore text and product data
  through protected review/product joins;
- keep held-out Beauty parents, later reviews, and held-out Conditioner parents
  separate;
- measure error rates by rating-derived class, product, category, month,
  token length, and actual truncation status after the clean run;
- measure saved-probability calibration overall, by test group, and by model
  answer before interpreting `model_score`;
- retain cautious search indicators for possible text-rating disagreement,
  mixed opinions, delivery/seller content, unclear dictionary sentiment,
  short text, and emotional writing;
- document that these indicators are not human labels and cannot prove a
  buyer or rating error;
- compare DistilBERT V2 and TF-IDF 1M V3 only on matching `review_id` values;
- save newly versioned error rows, an identity-bound manual-review queue, and
  compact JSON reports without review text;
- moved reusable joins, calibration, indicators, grouping, model comparison,
  manual-sample selection, validation, and summary calculations to
  `src.ml.error_analysis` with unit tests.

Notebook 03 has not yet clean-run against the new V2 predictions, so no current
automatic error count, calibration value, review-length comparison, or human
agreement rate is asserted here. Those exact automated results will be
materialized only after Notebook 02 finishes and Notebook 03 completes from a
clean kernel.

The completed 158-row V1 manual table and its agreement rates are historical
diagnostics tied to legacy prediction identity. They cannot accept
`beauty_distilbert_rating_sentiment_1m_v2` and must not be relabeled as current
gold. The new candidate remains `candidate_pending_human_review` until its own
queue is reviewed. Dictionary indicators remain search aids rather than proof
that a buyer chose the wrong star rating.

Target role:

```text
Explain current-model failure modes, bind every conclusion to exact prediction
identity, and turn the findings into data/model requirements.
```

## 06 — Amazon Data Contracts

Verdict: `VERIFIED UNDER CURRENT PLAN` (rewritten 2026-08-12).

Current path:
`notebooks/01_data_foundation/03_amazon_contract_validation.ipynb`

Resolution:

- removed the arbitrary-platform adapter and obsolete universal sample output;
- registered separate schema versions for canonical reviews, Amazon products,
  and enriched reviews in the dataset manifest;
- kept strict Pydantic review/product contracts and added a protected enriched
  join that rejects mismatched dataset identity, category, or `parent_asin`;
- moved real Amazon item-metadata adaptation to
  `src.ingestion.amazon_records` with tested metadata quality flags;
- validated names, Arrow types, and mandatory-field null counts over the full
  canonical review artifact;
- demonstrated structured logical errors and a real review-to-product join;
- retained both `asin` and `parent_asin`, created no sample artifact, used one
  imports cell, and completed a clean RAPIDS-kernel run.

Useful work to retain:

- typed Pydantic boundary models;
- explicit required/optional fields;
- structured validation errors;
- deterministic review-ID intention;
- data-quality report intention.

Required changes:

- remove the arbitrary-platform and generic-column-mapping objective;
- preserve both `asin` and `parent_asin`;
- add dataset category/version, schema version, source row identity, and UTC
  timestamp;
- use the real product catalog for product title and category data;
- do not use `parent_asin` as both product ID and fallback product name;
- define a stable, namespaced review ID independent of pandas implementation
  details;
- validate the full pipeline in batches rather than only the first 5,000 rows;
- move schemas and adapters to `src/schemas/` and `src/ingestion/`;
- replace the sample artifact with tested canonical data products.

Target role: explain and validate Amazon review, product, and enriched-review
contracts using the renamed notebook after its implementation is moved to
tested modules.

## 07 — Domain Detection

Verdict: `REMOVED AND REPLACED` (verified 2026-08-12).

Removed legacy path:
the former `notebooks/01_data_foundation/04_legacy_domain_detection.ipynb`

Do not retain:

- generic domain inference;
- hard-coded cross-domain keyword lists;
- ad hoc confidence score;
- synthetic keyword tests;
- `domain_detection.json` as a downstream dependency.

Replacement path:
`notebooks/01_data_foundation/04_amazon_category_registry.ipynb`

Resolution:

- removed keyword-based generic domain inference and its output artifact;
- built a full-source registry from all 1,028,914 item-metadata rows;
- registered 724 distinct Amazon category paths and 756 hierarchical nodes;
- preserved source display labels while creating normalized matching keys and
  stable namespaced path IDs;
- measured missing `main_category` and category-path coverage exactly;
- documented reproducible niche selection as an approved set of path IDs;
- inspected conditioner-related candidates without silently locking the MVP
  niche;
- saved a versioned registry and compact quality report, used one imports cell,
  and completed a clean RAPIDS-kernel run.

It should inspect and validate:

- dataset category;
- `main_category`;
- hierarchical `categories` paths;
- missing/empty category metadata;
- category-path normalization without changing source labels;
- niche definition rules;
- category configuration and taxonomy version references.

The Amazon source category is trusted as declared input. Product metadata is
validated, not guessed from review keywords.

## 08 — Product Catalog and Review Selection

Verdict: `VERIFIED UNDER CURRENT PLAN` (rewritten 2026-08-12).

Current path:
`notebooks/01_data_foundation/05_catalog_and_scope_validation.ipynb`

Resolution:

- replaced the 5,000-review/sample-catalog workflow with a versioned catalog
  containing all 1,028,914 product-metadata records;
- preserved `parent_asin`, structured category paths, features, descriptions,
  details JSON, catalog-search text, metadata quality flags, and exact review
  coverage;
- safely converted mixed-type prices and distinguished 648,234 missing prices
  from 83 invalid price values;
- reconciled every canonical review to product metadata: 8,986,368 of
  8,986,368 reviews matched;
- resolved the candidate standard-conditioner niche through a stable category
  path ID and inspected its complete product/review population;
- added tested contracts for versioned niche definitions and disjoint seller
  and competitor product sets;
- kept the example workspace in memory and clearly marked it as fictional
  because seller ownership is absent from the public dataset;
- created no sample review output, used one imports cell, and completed a clean
  RAPIDS-kernel run.

Useful work to retain:

- metadata-first product resolution;
- `parent_asin` join concept;
- typed catalog and selection result intention;
- product text composed from metadata;
- explicit selection context artifact.

Required changes:

- build a versioned catalog from all 1,028,914 metadata rows;
- preserve structured category paths and `details` instead of flattening all
  metadata into one string only;
- keep both ASIN levels and variation counts;
- add metadata quality flags and review coverage counts;
- resolve official Amazon niche/category filters before fuzzy text search;
- support explicit seller and competitor ASIN sets;
- separate catalog search from analytical population selection;
- evaluate product/niche matching with labelled queries;
- move Russian aliases out of core logic; later query understanding can use
  validated translation/LLM routing;
- query the complete eligible review population or precomputed facts;
- remove sample-only catalog and selected-review artifacts.

Target role:

```text
Demonstrate full Beauty catalog construction, category/niche scope, and a
seller-versus-competitor workspace definition.
```

## 6. Artifact Disposition

| Artifact | Current status | Action |
|---|---|---|
| `reviews_2021_2023.jsonl` | registered derived source | Preserve; manifest and checksum verified |
| `reviews_clean.parquet` | superseded prototype | Deleted after canonical replacement was verified |
| `sentiment_1m_split.npz` | invalid durable split | Deleted; rebuild with review IDs |
| `distilbert_sentiment_1m/` | frozen historical model | Preserve for audit; never promote to corrected V3 |
| `sentiment-baseline/` (`beauty_tfidf_rating_sentiment_1m_v3`) | verified UTC-stable baseline | Preserve with split, predictions, and metric report |
| `sentiment-transformer/` (`beauty_distilbert_rating_sentiment_1m_v2`) | `candidate_pending_human_review` | Finish clean runs 02/03; do not accept before its own human review |
| `error_analysis.csv` | inherited old split | Deleted; regenerate from accepted evaluation data |
| `reviews_full_train.parquet` | lineage not found | Deleted; unexplained data is not durable input |
| `reviews_full_test.parquet` | lineage not found | Deleted; unexplained data is not durable input |
| `reviews_universal_sample.parquet` | obsolete schema/sample | Deleted after canonical replacement |
| `domain_detection.json` | obsolete product concept | Deleted with the legacy notebook |
| `product_catalog_sample.parquet` | incomplete sample catalog | Deleted; replace with full versioned catalog |
| `selected_reviews_hair_conditioners.parquet` | only 27 sample reviews | Deleted; never use analytically |
| `product_selection.json` | sample-only scope | Deleted; replace with versioned workspace scope |

Deletion was performed on 2026-08-12 after the canonical review replacement and
the first two data-foundation notebooks were verified. Raw sources, the
registered filtered source, the canonical dataset, versioned reports, and the
historical DistilBERT candidate were intentionally retained.

## 7. Required Reusable Modules

The notebook refactor should create or consolidate these responsibilities:

```text
src/ingestion/amazon_reviews.py
src/ingestion/amazon_metadata.py
src/schemas/review.py
src/schemas/product.py
src/schemas/artifact.py
src/preprocessing/reviews.py
src/preprocessing/deduplication.py
src/catalog/build.py
src/catalog/categories.py
src/catalog/scope.py
src/ml/sampling.py
src/ml/splits.py
src/ml/sentiment/train.py
src/ml/sentiment/inference.py
src/ml/evaluation/sentiment.py
src/analytics/data_quality.py
```

Exact module names may change, but these responsibilities must not remain only
inside notebook cells.

## 8. Refactoring Sequence and 2026-08-18 Status

### Step 1 — Freeze and Manifest the Existing Beauty Snapshot

Status: `COMPLETED`.

The fixed source is registered as:

```text
amazon_reviews_2023_beauty_2021_2023_v1
```

The manifest now records source files, checksums, exact time range, row counts,
schemas, filtering rules, and artifact relationships; strict verification
passes against the materialized files.

### Step 2 — Rebuild 01, 02, and 06 as the Data Foundation

Status: `COMPLETED`. Canonical Amazon contracts and the tested
ingestion/cleaning pipeline are materialized, and all five data-foundation
notebooks have current clean runs.

### Step 3 — Replace 07 and 08

Status: `FOUNDATION COMPLETED; PRODUCT-QUERY UX REMAINS`. The complete Beauty
catalog, category registry, first niche, and full review-population selection
are verified. User-facing product resolution and a real saved
seller-versus-competitor workspace remain product work.

### Step 4 — Rebuild 03, 04, and 05

Status: `PARTIAL`. Immutable UTC-stable review-ID splits and both new TF-IDF
baselines are verified. DistilBERT V2 clean execution, refreshed Error Analysis,
and an identity-bound human review remain open.

### Step 5 — Start Aspect Intelligence

Status: `STARTED`. Corrected local V3 discovery inputs and the V2 extraction
evaluation queue are ready. External Gemini discovery, Taxonomy V2 human review,
and all 240 extraction annotations remain open.

## 9. Acceptance Gates

The all-ten-notebook refresh is accepted only when:

- [x] the fixed Beauty dataset has a manifest and checksums;
- [ ] every processed artifact has lineage and schema version;
- [x] the canonical model preserves `asin` and `parent_asin`;
- [x] cleaning and dedup rules have tests and reconciliation counts;
- [x] the complete product catalog is versioned;
- [x] category/niche selection is metadata-first and population-validated;
- [ ] seller and competitor scopes are explicit;
- [x] evaluation splits use review IDs and prevent defined leakage types;
- [x] TF-IDF V2/V3 metrics are reproduced on corrected tests with
  `parent_asin`-cluster intervals;
- [ ] DistilBERT V2 metrics and predictions are materialized by a clean run;
- [ ] Error Analysis is regenerated from those exact V2 predictions;
- [ ] the V2 candidate receives its own identity-bound human review;
- [x] five data-foundation, one TF-IDF, and two aspect notebooks run
  top-to-bottom from clean kernels;
- [ ] the remaining DistilBERT and Error Analysis notebooks run top-to-bottom
  from clean kernels;
- [ ] all ten notebooks pass the final execution-count and saved-error audit;
- [x] each notebook has one imports cell and follows `CODE_STYLE.md`;
- [x] no production dependency relies on a sample notebook artifact.

## 10. Aspect Notebook Refresh

`notebooks/03_aspects/01_taxonomy_review.ipynb` clean-runs as a qualitative
historical review of the complete `hair_conditioners_discovery_v2` proposal. It
validates the 23 candidate rows and exact evidence, but explicitly refuses to
present the legacy weighted support/share fields as population estimates. The
completed human decisions and 33 Taxonomy V1 labels/aliases remain approved
qualitative provenance only. Corrected quantitative discovery starts at
`aspect_discovery_sample_v2` / `hair_conditioners_discovery_v3` and requires
new Gemini responses, normalization, and human Taxonomy V2 review.

`notebooks/03_aspects/02_extraction_evaluation_sample.ipynb` clean-runs against
`hair_conditioners_extraction_eval_v2`. It verifies 160 probability-sampled
representative rows whose weights sum to 112,525 and 80 separate unweighted
targeted diagnostics. Its formula-safe blind queue contains no heuristic hints;
all 240 rows remain pending human annotation.

## 11. Immediate Next Work Item

The data foundation, full catalog, corrected sentiment manifests, both TF-IDF
refreshes, and the first niche decision are complete. Hair Conditioners V1
resolves through one approved Amazon category-path ID to 13,985 catalog products
and 118,804 reviews. Its corrected 1,500-review probability sample covers rating
group, year, review length, and 904 distinct products.

The immediate sentiment gates are to finish the clean run of
`02_distilbert_training.ipynb`, materialize exact automated V2 metrics and
predictions, and then clean-run `03_error_analysis.ipynb` on those identities.
The resulting model remains `candidate_pending_human_review` until its own new
queue is labeled; the legacy 158 labels cannot be reused.

For aspects, the V3 sample and 15-batch plan are ready, but external Gemini
extraction, normalization, and a new human Taxonomy V2 review remain open. The
corrected V2 blind extraction-evaluation queue is also ready, with all 240 rows
pending annotation against the qualitative Taxonomy V1 labels.

First implementation slice:

```text
finish clean-run 02 and materialize exact DistilBERT V2 evidence
    ↓
clean-run 03 and generate the identity-bound human-review queue
    ↓
complete new human review and decide candidate promotion
    ↓
run all 15 external V3 Gemini batches and human-review Taxonomy V2
    ↓
annotate all 240 V2 extraction-evaluation rows
    ↓
compare extraction approaches and measure precision, recall, F1, and coverage
```
