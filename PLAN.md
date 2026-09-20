# Amazon Seller Review Intelligence — Product and Development Plan

## Document Status

This document is the primary source of truth for the product scope and the
development sequence.

The repository contains work completed under an earlier plan. Existing
notebooks, source files, processed datasets, and model artifacts are not
considered verified until they are audited against this document.

Status labels used below:

- `VERIFIED` — inspected and accepted under the current plan;
- `IN PROGRESS` — current work has verified outputs but still has open gates;
- `AUDIT REQUIRED` — implemented earlier but not yet accepted;
- `PLANNED` — not implemented under the current plan;
- `DEFERRED` — intentionally outside the current milestone.

Current refresh state (2026-09-09): the UTC-stable sentiment manifests and the
probability-sampled aspect discovery/evaluation artifacts are prepared. The
newly versioned TF-IDF baselines have been trained and evaluated; the current
1M result is `beauty_tfidf_rating_sentiment_1m_v3`, with exact metrics and
`parent_asin`-cluster intervals in its versioned report. The new DistilBERT
version `beauty_distilbert_rating_sentiment_1m_v2` is
`candidate_pending_human_review`, not an accepted model. Its exact automated
metrics, refreshed error analysis, and identity-bound 156-row blind queue have
been materialized by clean runs of sentiment Notebooks 02 and 03. A
stratified 20-row subset has completed human review: model/human agreement is
14/20 and rating-label/human agreement is 4/20. This is a limited diagnostic,
not a representative accuracy estimate or formal model acceptance; the
remaining 136 rows are intentionally left open and do not block downstream
project work. Legacy sentiment models, metrics, and the
completed 158-row review remain historical diagnostics only. Taxonomy V1
labels and aliases remain valid qualitative provenance, but its population
shares are invalid. All 15 V3 Gemini discovery batches, deterministic
candidate aggregation, semantic normalization, and project-owner review are
complete; the quantitative 39-aspect Taxonomy V2 is approved. Legacy and corrected-input validity is recorded in
`reports/model_evaluation/artifact_validity_v1.json` and
`reports/aspects/artifact_validity_v1.json`; newly materialized model outputs
must be added to the sentiment ledger after the final clean runs.

---

# 1. Product Definition

## 1.1 Product Goal

Build an Amazon-focused Review Intelligence platform that transforms large
collections of customer reviews into actionable analytics for Amazon sellers.

The platform helps a seller understand:

- their own products;
- competing products;
- a selected Amazon product niche;
- what customers like and dislike;
- why products receive positive or negative feedback;
- which problems, expectations, and use cases recur;
- how customer opinion changes over time;
- how the seller's products differ from competitors;
- which customer phrases may inform listing copy and advertising research.

The final product is not a sentiment classifier. Sentiment is one signal inside
a larger seller intelligence system.

## 1.2 Target Users

Primary users are Amazon sellers, brand owners, product managers, and marketplace
analysts who need review-based intelligence for product and competitor research.

Typical user goals:

```text
Improve an existing product
Find recurring defects and complaints
Understand why customers prefer a competitor
Discover unmet customer needs
Identify strengths worth emphasizing in a listing
Find customer vocabulary for keyword and advertising research
Track new complaints and changes in sentiment
Compare products inside one Amazon niche
```

## 1.3 Core Product Promise

Perform expensive processing offline, then let the user explore precomputed and
auditable analytics interactively through dashboards and a Vertex AI / Gemini
analyst.

```text
Large Amazon review corpus
          ↓
Heavy offline processing
          ↓
Versioned analytical data products
          ↓
Fast deterministic analytics + evidence retrieval
          ↓
Vertex AI / Gemini conversational analyst
```

## 1.4 Development Context and Success Modes

The repository is developed in three compatible modes:

1. **Learning project.** The owner is learning data engineering, machine
   learning, analytics, cloud architecture, and product development while
   building the system. Decisions must therefore explain not only *what* is
   done, but also *why*, which alternatives exist, and how the result is
   verified. Questions and repeated explanations are an expected part of the
   workflow, not a project problem.
2. **Portfolio project.** The repository must demonstrate professional habits:
   reproducible datasets, explicit contracts, honest evaluation, tested
   reusable code, readable notebooks, documented limitations, and traceable
   architectural decisions. A polished interface cannot substitute for valid
   data and defensible analytics.
3. **Potential commercial product.** If the analytical value is validated and
   sellers show demand, the same foundation may evolve into a commercial
   service. We preserve clear data/model versions, category configuration,
   seller isolation boundaries, and replaceable pipeline components so that
   this growth is possible without promising it prematurely.

These modes do not have equal priority at every moment. Correctness,
reproducibility, and understanding come first. Portfolio presentation is built
on that foundation. Commercial scalability is introduced when evidence and
real usage justify the additional complexity.

The assistant also acts as a technical teacher and collaborator: handoffs
should explain unfamiliar terms, assumptions, trade-offs, and verification in
clear Russian while preserving standard English technical names.

---

# 2. Product Scope

## 2.1 In Scope

- Amazon Reviews 2023 as the initial research dataset;
- Amazon dataset categories and product category metadata;
- Amazon review and item metadata joined by product identifiers;
- Beauty and Personal Care as the first working category;
- analysis by product, competitor set, and product niche;
- overall sentiment and rating analytics;
- aspect discovery, extraction, normalization, and aspect sentiment;
- strengths, complaints, customer needs, use cases, and trend analytics;
- customer-language and keyword-phrase discovery from reviews;
- offline processing over large review populations;
- semantic retrieval of representative review evidence;
- deterministic Python analytics;
- Vertex AI / Gemini for question understanding, tool use, synthesis, and
  explanation;
- FastAPI backend and a user-facing analytical application;
- category-specific configuration, taxonomies, and model artifacts within one
  shared Amazon pipeline.

## 2.2 Not a Goal for the Current Product

- supporting arbitrary review datasets from unrelated platforms;
- automatically detecting an unknown business domain from raw text;
- creating one universal model that is assumed to work equally well in every
  Amazon category;
- processing every review with Gemini in production;
- replacing deterministic analytics with LLM calculations;
- live Amazon scraping;
- automatic ownership detection for a seller's ASINs;
- direct access to Seller Central, Amazon Ads, or Brand Analytics in the first
  version;
- automatic advertising-budget optimization;
- claiming search volume, conversion, or PPC performance from review text.

The platform can discover customer language and candidate phrases from reviews.
Those phrases are inputs to listing and advertising research, not verified
Amazon search-demand metrics.

---

# 3. Data Source and Amazon Classification

## 3.1 Primary Dataset

Official source:

```text
Amazon Reviews 2023
https://amazon-reviews-2023.github.io/
```

The dataset contains:

- user reviews;
- product/item metadata;
- Amazon product identifiers;
- dataset files grouped by Amazon category;
- product category metadata and product attributes.

The historical snapshot is fixed and versioned for reproducibility. It is not a
live view of the current Amazon marketplace.

The accepted V1 working snapshot is the data already prepared in the repository:

```text
dataset category: Beauty_and_Personal_Care
time window:      2021-01-01 through 2023-09-13
filtered reviews: 9,084,957 before the current cleaning/deduplication step
logical version:  amazon_reviews_2023_beauty_2021_2023_v1
```

The versioned manifest is
`config/datasets/amazon_reviews_2023_beauty_2021_2023_v1.json`. Its registered
sizes, SHA-256 values, and row counts have been strictly verified against the
materialized files; the current cleaning reconciliation is stored in
`reports/data_quality/amazon_reviews_2023_beauty_2021_2023_v1.json`.

## 3.2 First Working Category

The first working dataset is:

```text
Beauty_and_Personal_Care
```

The earlier reference to Home and Kitchen is obsolete. The project began with
another category idea and then moved to cosmetics / Beauty and Personal Care.

The first product niche inside Beauty and Personal Care is the complete Amazon
leaf `Beauty & Personal Care > Hair Care > Shampoo & Conditioner >
Conditioners`, registered as `hair_conditioners_v1`. Related Amazon leaves such
as deep conditioners, 2-in-1 products, beard conditioners, and color
conditioners are outside V1.

## 3.3 Amazon Classification Strategy

The platform uses the classification and product metadata available in Amazon
Reviews 2023. It does not infer a completely unrelated domain from review text.

Classification signals, in priority order:

```text
dataset_category
    ↓
main_category
    ↓
categories (Amazon category hierarchy, when available)
    ↓
product metadata: title, features, description, details, store
```

Product metadata may be incomplete. Missing category paths must be recorded as
a data-quality limitation. Metadata-based fallback matching may be used, but it
must not silently pretend to be an official Amazon category label.

## 3.4 Product Identity

The source distinguishes:

```text
asin        → a purchasable product or variation
parent_asin → the parent product/family used to join product metadata
```

Default analytical unit:

```text
parent_asin
```

Variation-level analysis may use `asin` when the data supports a meaningful
comparison between sizes, colors, styles, or other variants.

## 3.5 Seller and Competitor Sets

The public dataset does not identify which products belong to the current user.
The product must therefore accept explicit user configuration:

```text
seller_asins
competitor_asins
selected_niche
```

These selections define the analytical comparison scope. They must be stored as
versioned workspace configuration rather than hard-coded in notebooks.

## 3.6 Known Data Limitations

- The dataset is a historical 2023 snapshot.
- Some products do not have complete item metadata.
- Amazon category paths may be missing or empty for some products.
- Product price and rating metadata reflect collection time, not current values.
- Reviews do not contain Amazon search volume, ad impressions, clicks, spend,
  conversion, or keyword-rank data.
- Review rating is an imperfect proxy for textual sentiment.
- Seller ownership and the intended competitor set are not present in the
  public dataset.

Every user-facing answer must respect these limitations.

---

# 4. Seller Intelligence Questions

The analytical system should answer questions in seven groups.

## 4.1 Product Health

```text
What is the rating and sentiment distribution?
Which aspects receive the most positive or negative feedback?
What are the most frequent complaints?
Which complaints are severe but uncommon?
Which positive qualities are consistently mentioned?
```

## 4.2 Competitive Intelligence

```text
Where does my product outperform competitors?
Where do competitors receive better customer feedback?
Which complaints are unique to my product?
Which competitor strengths should influence product development?
What unmet needs appear across the whole niche?
```

## 4.3 Product Improvement

```text
Which defects or usability problems recur?
Which product attributes should be improved first?
Which complaints have high volume, high negativity, and recent growth?
What trade-offs do customers describe?
Which requested features are missing?
```

## 4.4 Voice of Customer

```text
How do customers describe the problem they are trying to solve?
Which benefits matter most to them?
Which phrases do they repeatedly use?
What expectations cause disappointment?
What customer segments or use cases appear in the reviews?
```

## 4.5 Listing and Advertising Research

```text
Which customer phrases describe important benefits?
Which use cases and pain points should be reflected in listing copy?
Which aspect-language combinations may become candidate keyword themes?
Which claims are supported by review evidence?
Which claims would be contradicted by frequent complaints?
```

This layer produces review-derived keyword candidates. It does not claim to
measure Amazon query popularity or advertising performance.

## 4.6 Trends and Emerging Issues

```text
How do ratings, sentiment, and aspect sentiment change over time?
Which complaints are growing unusually quickly?
Did a product improve after a particular period?
Are new packaging, quality, or reliability problems appearing?
```

## 4.7 Evidence and Explanation

```text
Which reviews support this conclusion?
Are the examples representative or exceptional?
How many reviews and products are included?
What filters and time range were used?
How confident is the system in the conclusion?
```

---

# 5. Core Design Principles

## Principle 1 — Seller Decisions Come First

Every model, metric, and dashboard must support a concrete seller decision:
product improvement, competitive positioning, listing communication, or market
monitoring.

## Principle 2 — Amazon-Focused, Not Artificially Universal

The pipeline is reusable across Amazon categories, but it is allowed to use
Amazon-specific identifiers, metadata, and category structure.

## Principle 3 — Product Catalog Before Analytical Comparison

Products and niches are resolved from Amazon metadata before review analytics
are filtered or compared. Review text alone must not decide product identity.

## Principle 4 — Process the Relevant Population Offline

Aggregate analytics should use the complete eligible review population in the
chosen dataset snapshot, subject to explicit quality filters. Top-K retrieval is
not an acceptable source for aggregate percentages.

## Principle 5 — Aspect Intelligence Is Central

Overall sentiment says whether customers are satisfied. Aspect intelligence
explains what creates satisfaction or dissatisfaction.

## Principle 6 — Python Calculates Facts

Counts, percentages, rankings, trends, comparisons, confidence intervals, and
scores are calculated by deterministic and tested Python code.

## Principle 7 — Gemini Explains and Orchestrates

Gemini interprets a question, selects approved tools, synthesizes structured
results, and explains retrieved evidence. It does not invent statistics.

## Principle 8 — RAG Provides Evidence

Retrieval supplies representative reviews for explanation and traceability. It
does not replace the analytical warehouse.

## Principle 9 — Shared Pipeline, Category-Aware Artifacts

Core code is shared across Amazon categories. Category-specific taxonomies,
evaluation sets, configurations, and ML models are versioned separately when
the evidence shows they are needed.

## Principle 10 — Every Result Has Scope and Provenance

An analytical result must identify its dataset version, category, products,
time window, filters, review count, method, and model version.

---

# 6. Target Architecture

## 6.1 Offline Pipeline

Heavy processing is performed before user interaction.

```text
Amazon Reviews 2023 category files
                  │
                  ▼
         Ingestion and validation
                  │
                  ▼
       Canonical Amazon data model
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Product catalog       Clean reviews
        │                   │
        └─────────┬─────────┘
                  ▼
       Product/category enrichment
                  │
       ┌──────────┴───────────┐
       ▼                      ▼
Overall sentiment       Aspect intelligence
                              │
                Discovery → Normalization
                    → Extraction → Sentiment
       │                      │
       └──────────┬───────────┘
                  ▼
       Analytical fact tables
                  │
       ┌──────────┴───────────┐
       ▼                      ▼
Aggregated marts       Evidence embeddings
       │                      │
       └──────────┬───────────┘
                  ▼
       Versioned data products
```

## 6.2 Online Pipeline

The online path works mainly with precomputed results.

```text
Seller question
      │
      ▼
Workspace scope
(own ASINs, competitors, niche, dates)
      │
      ▼
Gemini query understanding
      │
      ▼
Approved analytical tools
      │
      ├───────────────┐
      ▼               ▼
Deterministic      Evidence retrieval
analytics          with metadata filters
      │               │
      └───────┬───────┘
              ▼
      Structured result
              │
              ▼
      Gemini explanation
              │
              ▼
Dashboard / chat / export
```

## 6.3 Workload Separation

Offline workloads:

- dataset scanning and cleaning;
- product metadata processing;
- product/category indexing;
- model inference over large review populations;
- aspect extraction and aspect sentiment;
- embeddings;
- aggregations and trend tables;
- data-quality reports;
- artifact and model versioning.

Online workloads:

- workspace and product-scope selection;
- question understanding;
- analytical queries over precomputed tables;
- filtered evidence retrieval;
- comparison assembly;
- natural-language explanation.

---

# 7. Canonical Data Contracts

The internal representation is Amazon-focused. It normalizes the public source
without pretending to be universal across unrelated platforms.

## 7.1 Review Record

Required fields:

```text
review_id
dataset_version
dataset_category
asin
parent_asin
rating
review_title
review_text
review_date
```

Optional source fields:

```text
user_id
verified_purchase
helpful_vote
review_images
```

The source does not provide a dedicated review ID. The project must define and
test a deterministic ID strategy during the data audit.

## 7.2 Product Record

```text
parent_asin
main_category
category_path
product_title
store
average_rating
rating_number
features
description
details
price_at_collection
images
bought_together
metadata_quality_flags
```

## 7.3 Enriched Review Record

```text
review_id
asin
parent_asin
dataset_category
main_category
category_path
product_title
store
rating
review_text
review_date
verified_purchase
helpful_vote
overall_sentiment
model_score
seller_attention
seller_attention_policy_version
pipeline_version
```

`seller_attention` is a deterministic business field, not another sentiment
class and not another ML model. Policy `amazon_rating_attention_v1` maps 1–3
stars to `needs_attention` and 4–5 stars to `satisfied`. The original rating
and the independent text-sentiment result are always preserved.

## 7.4 Aspect Observation

One review may produce several rows.

```text
review_id
parent_asin
canonical_aspect
aspect_expression
aspect_sentiment
confidence
evidence_span
taxonomy_version
model_version
```

## 7.5 Analytical Result Contract

Every aggregate returned to the application must include:

```text
metric_name
metric_value
review_count
product_count
product_scope
category_scope
time_window
filters
dataset_version
pipeline_version
model_version
calculation_timestamp
```

---

# 8. Analytical Intelligence Layers

## 8.1 Rating, Seller Attention, and Overall Sentiment

Outputs:

- rating distribution;
- `needs_attention` count and share using a versioned rating policy;
- sentiment distribution;
- mean and median rating;
- review volume;
- sentiment and rating trends;
- product and competitor comparisons;
- verified-purchase breakdown;
- helpfulness-weighted views where analytically justified.

The verified current quantitative reference is the TF-IDF + Logistic Regression
baseline `beauty_tfidf_rating_sentiment_1m_v3`, trained on the corrected
UTC-stable manifest. On natural-distribution tests it reaches accuracy / macro
F1 of 0.84731 / 0.72757 for held-out Beauty parent products, 0.84722 / 0.72717
for later reviews of training products, and 0.86288 / 0.72051 for held-out Hair
Conditioner parent products. The corresponding balanced macro F1 values are
0.78717, 0.78435, and 0.77808. All 95% intervals resample complete
`parent_asin` clusters; exact intervals and per-class results are in
`reports/model_evaluation/beauty_tfidf_rating_sentiment_1m_v3.json`.

`beauty_distilbert_rating_sentiment_1m_v2` is currently
`candidate_pending_human_review`, not accepted. Its exact automated metrics and
refreshed error-analysis results will be published only by clean runs of
sentiment Notebooks 02 and 03. The completed 158-row review belongs to the
legacy V1 predictions and is historical diagnostic evidence only; its labels
must not be transferred to the V2 candidate. All sentiment targets remain
rating-derived weak labels rather than human ground truth, and a model score is
not presented as guaranteed human confidence, especially for Neutral answers.

The Hair Conditioner evaluation excludes a defined set of held-out
`parent_asin` values, not the whole niche. Reviews for other Conditioner parent
products remain in training.

Seller attention must not be presented as textual negativity. For example, a
three-star review may be `overall_sentiment=neutral` and simultaneously
`seller_attention=needs_attention`. Aspect extraction explains what may need
improvement.

## 8.2 Aspect Discovery

Use representative, stratified review samples to discover recurring product
attributes, benefits, complaints, and use cases within an Amazon category or
niche.

Gemini may assist discovery, but the resulting candidates must be counted,
reviewed, versioned, and evaluated before entering the taxonomy.

## 8.3 Aspect Normalization

Map related customer expressions to a canonical aspect.

```text
smell
scent
fragrance
strong odor
    ↓
fragrance
```

Normalization may use embeddings and Gemini, with deterministic storage of the
accepted mapping and taxonomy version.

## 8.4 Aspect Extraction

Identify all relevant aspects in each review. The extractor must support
multi-aspect and mixed-sentiment reviews.

## 8.5 Aspect Sentiment

Assign sentiment to each extracted aspect and preserve the supporting text span.

```text
"Works well, but the bottle leaks."

effectiveness → positive
packaging     → negative
```

## 8.6 Complaint and Strength Mining

Generate ranked, evidence-backed signals for:

- frequent strengths;
- frequent complaints;
- high-severity complaints;
- product-specific problems;
- niche-wide problems;
- competitor advantages;
- potential unmet needs.

Ranking must consider support, sentiment strength, recency, confidence, and
product coverage. A single extreme review must not become a market conclusion.

## 8.7 Use Cases and Customer Needs

Extract recurring situations, intended outcomes, and customer constraints.

Examples:

```text
hair type or skin type
travel use
sensitive skin
gift use
professional use
ease-of-use expectations
durability expectations
```

## 8.8 Customer Language and Keyword Candidates

Extract recurring n-grams, noun phrases, aspect phrases, benefits, problems,
and use-case language from reviews.

Outputs may include:

- phrase frequency;
- product and niche coverage;
- sentiment association;
- aspect association;
- recent growth;
- representative evidence;
- candidate listing or advertising themes.

These are review-language signals, not Amazon keyword-volume metrics.

## 8.9 Trend and Emerging-Issue Detection

Track:

- review volume;
- rating and sentiment;
- aspect frequency;
- aspect sentiment;
- complaint share;
- new or rapidly growing complaint phrases.

Trend alerts require minimum support and must distinguish an absolute count
increase from a share increase.

## 8.10 Competitive Gap Analysis

For a seller product and a defined competitor set, calculate:

```text
aspect importance
seller sentiment by aspect
competitor sentiment by aspect
sentiment gap
mention gap
complaint gap
evidence coverage
```

This becomes the basis for product-development and positioning insights.

---

# 9. Analytics Engine

The analytics engine exposes tested deterministic functions or services.

Initial tool surface:

```text
get_workspace_overview()
get_product_summary()
get_niche_summary()
get_rating_distribution()
get_sentiment_distribution()
get_aspect_statistics()
get_strengths()
get_complaints()
get_unmet_needs()
get_trends()
get_emerging_issues()
compare_products()
get_competitive_gaps()
get_customer_phrases()
search_review_evidence()
```

Each tool must:

- require an explicit product/category/time scope;
- validate filters;
- return structured data;
- include population sizes and provenance;
- avoid LLM-generated calculations;
- have unit and integration tests.

---

# 10. Embeddings, Retrieval, and RAG

## 10.1 Purpose

Embeddings support:

- semantic review search;
- representative evidence retrieval;
- similar complaint discovery;
- aspect normalization;
- clustering and phrase exploration.

## 10.2 Retrieval Requirements

Evidence search must support metadata filters for:

```text
dataset_category
main_category
category_path
asin
parent_asin
aspect
sentiment
rating
verified_purchase
date
```

Hybrid retrieval should be evaluated against pure vector retrieval because
exact product terms, ASINs, brands, and defect phrases may benefit from lexical
matching.

## 10.3 Evidence Quality

Retrieved evidence should be:

- relevant to the question;
- inside the requested product scope;
- diverse rather than near-duplicate;
- representative of the analytical result;
- traceable to review identifiers;
- safe to display under the project's data-use policy.

RAG answers must distinguish representative examples from aggregate facts.

---

# 11. Vertex AI / Gemini Analyst

## 11.1 Responsibilities

Gemini is used for:

- question and intent understanding;
- choosing analytical tools;
- multi-step comparison workflows;
- aspect discovery and ambiguous taxonomy normalization;
- synthesis of structured statistics;
- explanation of trends and competitive gaps;
- summarization of retrieved evidence;
- producing clear seller-oriented answers.

## 11.2 Restrictions

Gemini must not:

- calculate aggregate statistics from raw review text;
- invent product scope or competitor membership;
- claim current marketplace facts from a historical dataset;
- claim advertising performance without advertising data;
- present a few retrieved reviews as population statistics;
- omit material uncertainty or data limitations.

## 11.3 Answer Contract

A high-quality answer contains:

```text
Direct answer
Key metrics
Comparison scope
Interpretation for the seller
Representative review evidence
Data/model limitations
```

## 11.4 Initial Function Calling Tools

```text
resolve_product_scope()
get_product_summary()
get_aspect_statistics()
get_complaints()
get_trends()
compare_products()
get_competitive_gaps()
get_customer_phrases()
search_review_evidence()
```

The LLM can select tools, but tool inputs and outputs are validated with typed
schemas.

---

# 12. User Experience

## 12.1 Seller Workspace

The user creates a workspace containing:

```text
Amazon dataset category
selected niche
seller products / parent ASINs
competitor products / parent ASINs
default time window
default quality filters
```

## 12.2 Dashboard

Initial dashboard sections:

- workspace overview;
- own product performance;
- competitor comparison;
- aspect strengths and weaknesses;
- top complaints;
- unmet needs;
- rating, sentiment, and aspect trends;
- emerging issues;
- customer language and candidate phrase themes;
- representative evidence;
- AI Analyst chat.

## 12.3 User Flow

```text
Select Amazon category
        ↓
Select niche or category node
        ↓
Select own and competitor products
        ↓
Open precomputed dashboard
        ↓
Ask follow-up questions
        ↓
Inspect metrics and review evidence
```

Arbitrary dataset upload is not part of the first product flow.

---

# 13. Repository Architecture

Target structure:

```text
config/
├── datasets/
├── categories/
├── taxonomies/
└── workspaces/

src/
├── ingestion/
├── schemas/
├── catalog/
├── preprocessing/
├── ml/
│   ├── sentiment/
│   ├── aspects/
│   └── evaluation/
├── analytics/
├── embeddings/
├── retrieval/
├── llm/
├── agents/
├── api/
└── common/

models/
└── beauty_and_personal_care/
    ├── sentiment/
    └── aspects/

data/
├── raw/
├── interim/
├── processed/
└── analytics/

notebooks/
tests/
docs/
app/
```

Notebooks explain exploration and experiments. Reusable schemas, pipeline code,
metrics, and analytics belong in `src/` and are covered by tests.

---

# 14. Development Roadmap

## Phase 00 — Product Plan and Documentation

Status: `VERIFIED`

- [x] Define Amazon sellers as the target users.
- [x] Limit the product to Amazon data and Amazon category classification.
- [x] Correct the first dataset to Beauty and Personal Care.
- [x] Define offline and online responsibilities.
- [x] Separate review-derived phrase insights from Amazon Ads metrics.
- [x] Align `README.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, and
  `docs/ROADMAP.md` with this plan.
- [x] Confirm Hair Conditioners V1 as the first Beauty product niche.
- [x] Confirm the V1 analytical time window.

Exit criterion:

```text
One consistent product definition and one consistent architecture across all
project documents.
```

## Phase 01 — Audit Existing Work

Status: `VERIFIED`

Audit the original notebooks `01–08` (now organized under
`notebooks/01_data_foundation/` and `notebooks/02_sentiment/`), then include the
two notebooks under `notebooks/03_aspects/` in the 2026-08-18 refresh. The
current execution scope is all ten notebooks, their reusable source code,
processed artifacts, and model metadata.

For every notebook determine:

```text
KEEP
REFACTOR
MOVE REUSABLE CODE TO src/
REPLACE
REMOVE
```

Audit dimensions:

- compatibility with the Amazon-focused scope;
- correct Beauty and Personal Care dataset references;
- dataset date window and row counts;
- schema and identifier correctness;
- use of `asin` versus `parent_asin`;
- leakage and split correctness;
- reproducibility and random seeds;
- memory safety for large datasets;
- artifact inputs and outputs;
- metrics and conclusions;
- compliance with `docs/CODE_STYLE.md`;
- one imports cell per notebook;
- reusable code placement;
- stale outputs and execution order.

The initial audit is recorded in `docs/NOTEBOOK_AUDIT.md`. It found that the
historical notebooks required refactoring or replacement before acceptance.
Each of the ten current notebooks receives its own refresh status: five
data-foundation notebooks, the TF-IDF notebook, and two aspect notebooks have
clean runs; the DistilBERT and refreshed Error Analysis clean runs remain open.
No notebook inherits acceptance from the original audit or from another
notebook's saved output.

Exit criterion:

```text
Written audit result for every existing notebook and artifact, with an approved
refactoring sequence.
```

## Phase 02 — Amazon Data Foundation

Status: `VERIFIED`

- [x] Create versioned dataset configuration.
- [x] Implement streaming review ingestion.
- [x] Implement item-metadata ingestion and a full product catalog build.
- [x] Define canonical review and product schemas.
- [x] Register review, product, and enriched-review schema versions.
- [x] Define deterministic `review_id` generation.
- [x] Validate ratings, timestamps, text, and identifiers.
- [x] Preserve `asin` and `parent_asin` correctly.
- [x] Produce review data-quality reports.
- [x] Create the versioned canonical review dataset.
- [x] Add initial unit and integration tests.
- [x] Add product-metadata quality reports and integration tests.
- [x] Rematerialize the full canonical dataset with physically non-nullable
  required Arrow fields and a bounded optional-integer parser.
- [x] Cover anonymous exact deduplication and malformed timestamp/helpful-vote
  cases with executable notebook fixtures and regression tests.
- [x] Strictly verify every manifest size, SHA-256 value, and declared row count.

## Phase 03 — Product Catalog and Amazon Category Index

Status: `IN PROGRESS`

- [x] Build the product catalog from item metadata.
- [x] Build a versioned full-source Amazon category registry.
- [x] Use one NFKC, whitespace-collapse, and case-fold normalization contract in
  both registry and catalog paths.
- [x] Store dataset category, `main_category`, and category paths in contracts
  and registry outputs.
- [x] Track missing and inconsistent category metadata.
- [x] Build normalized product text for catalog search.
- [x] Enforce exactly one catalog row per `parent_asin`; verify 1,028,914 rows,
  no duplicate parents, and 100% reconciliation of 8,986,368 reviews.
- [ ] Resolve product and parent-ASIN queries.
- [x] Resolve the approved Hair Conditioners V1 path to full catalog products.
- [x] Define validated explicit seller and competitor product-set contracts.
- [ ] Evaluate product/niche matching quality.

This phase replaces the old generic Domain Detection stage.

## Phase 04 — Overall Sentiment Baseline Audit and Productionization

Status: `IN PROGRESS`

- [x] Verify rating-derived sentiment labels.
- [x] Rebuild and verify the 300k V2 and 1M V3 manifests with an explicit UTC
  cutoff.
- [x] Check product/user/time leakage.
- [x] Train `beauty_tfidf_rating_sentiment_v2` and
  `beauty_tfidf_rating_sentiment_1m_v3` on corrected membership.
- [x] Publish reproducible TF-IDF metrics with percentile bootstrap intervals
  over complete `parent_asin` clusters.
- [x] Complete the clean run of `02_distilbert_training.ipynb` and materialize
  exact `beauty_distilbert_rating_sentiment_1m_v2` metrics and predictions.
- [x] Complete the clean run of `03_error_analysis.ipynb` against those new
  predictions.
- [x] Evaluate the current Transformer candidate by parent-product holdout,
  later-review holdout, review length, and calibration.
- [x] Preserve the completed legacy 158-row review as historical diagnostic
  evidence only.
- [x] Complete and summarize a limited stratified 20-row V2 human diagnostic.
- [ ] Complete an identity-bound human review for the V2 candidate.
- [ ] Accept or reject the V2 candidate after automated and human gates.
- [x] Move TF-IDF and DistilBERT inference and evaluation into reusable modules.
- [x] Create TF-IDF and DistilBERT model metadata and versioning.
- [x] Define versioned seller attention as `1–3 needs_attention`,
  `4–5 satisfied`, without training a redundant model.

Current verified 1M TF-IDF results:

```text
Held-out Beauty parents, natural:       accuracy 0.84731; macro F1 0.72757
Held-out Beauty parents, balanced:      accuracy 0.78644; macro F1 0.78717
Later 2023 reviews, natural:             accuracy 0.84722; macro F1 0.72717
Later 2023 reviews, balanced:            accuracy 0.78382; macro F1 0.78435
Held-out Conditioner parents, natural:  accuracy 0.86288; macro F1 0.72051
Held-out Conditioner parents, balanced: accuracy 0.77778; macro F1 0.77808
```

The Conditioner test holds out 658 parent products from the niche, not the
whole Hair Conditioners niche; other Conditioner parents remain in training.
The exact confidence intervals, cluster counts, per-class metrics, and confusion
matrices are in
`reports/model_evaluation/beauty_tfidf_rating_sentiment_1m_v3.json`.

`beauty_distilbert_rating_sentiment_1m_v2` is
`candidate_pending_human_review`. Its balanced macro F1 is 0.81815 on held-out
Beauty parent products, 0.81687 on later 2023 reviews, and 0.81461 on held-out
Conditioner parent products. The refreshed error analysis covers 209,021
reviews, records 26,782 errors, and materializes a new 156-row identity-bound
blind queue. A stratified 20-row subset is now human-labeled: the model agrees
with the human text label on 14/20 rows, while the rating-derived weak label
agrees on 4/20. The subset diagnoses likely weak-label noise but is not an
unbiased corpus-wide accuracy sample, so the model remains a candidate while
the rest of the project proceeds. The old V1 metrics and completed 158-row label review are retained
only as historical diagnostics and cannot accept the new candidate. Seller
attention therefore remains separate from textual sentiment.

## Phase 05 — Aspect Taxonomy Discovery

Status: `IN PROGRESS`

- [x] Preserve the completed historical V2 discovery and human-reviewed
  Taxonomy V1 as qualitative provenance only.
- [x] Mark every legacy V1 weighted support/share field invalid for population
  inference.
- [x] Build the corrected 1,500-review SRS sample, stratified by rating, year,
  and length without a product cap; verify that weights sum to 114,025.
- [x] Implement the reusable Gemini structured-output client with Gemini API
  and Vertex AI backends.
- [x] Prepare the V3 plan as 15 resumable batches of 100 reviews with strict
  sample/plan/response identities.
- [x] Verify local Gemini client readiness without making an external request.
- [x] Run all 15 corrected V3 Gemini batches.
- [x] Normalize and validate the corrected V3 candidates: 89 measured labels
  are covered exactly once by 39 proposed aspects.
- [x] Human-review candidate merges, definitions, evidence, and coverage for
  Taxonomy V2: 22 approvals, 9 edits, 7 splits, and 1 exclusion.
- [x] Materialize the first quantitatively valid Beauty taxonomy support
  report.

## Phase 06 — Aspect Extraction and Aspect Sentiment

Status: `IN PROGRESS`

- [x] Define the V1 gold-annotation contract and exact-evidence validation.
- [x] Build `hair_conditioners_extraction_eval_v2`: 160 probability-sampled
  representative rows whose weights sum to 112,525, plus 80 unweighted
  rare-aspect-targeted diagnostics.
- [x] Build a blind, formula-safe annotation queue with no heuristic hints.
- [x] Rebuild the 240-row evaluation as `extraction_eval_v3`, identity-bound to
  the approved Taxonomy V2.
- [ ] Complete human gold annotation only if formal acceptance is required.
- [x] Build privacy-minimized Gemini silver labels for all 240 rows and compare
  the exact-alias baseline against them.
- [x] Compare exact-alias and character-TF-IDF hybrid extraction approaches.
- [x] Implement a deterministic exact-alias extraction baseline.
- [x] Support multiple aspects per review.
- [x] Preserve exact evidence spans and character offsets.
- [x] Build a Gemini-silver training set and product-separated out-of-fold
  evaluation predictions; human gold remains open for formal acceptance.
- [x] Implement aspect normalization.
- [x] Implement a local TF-IDF evidence-context aspect-sentiment baseline.
- [x] Evaluate diagnostic precision, recall, F1, and coverage against Gemini
  silver; reserve final accuracy claims for human gold.
- [x] Analyze silver-reference errors by aspect and representative/targeted
  review type; prepare a diverse 24-row human disagreement audit.
- [x] Implement bounded local batch inference with exact sentence evidence.
- [x] Store 236,316 versioned aspect observations for all 118,804 niche reviews.

## Phase 07 — Offline Seller Analytics

Status: `IN PROGRESS`

- [x] Build the first 39-row niche aspect-statistics table from versioned
  review-aspect observations.
- [ ] Add `seller_attention` and its policy version to review-level analytics.
- [x] Calculate `needs_attention` counts and rates by product, aspect, rating,
  and time; competitor scope remains open.
- [x] Build the product-aspect aggregate mart.
- [x] Build the first seller-versus-competitor aspect-comparison mart.
- [x] Build initial niche aspect benchmarks.
- [x] Calculate strengths and complaint rankings for the demo workspace.
- [x] Calculate support-filtered competitive gaps.
- [ ] Extract use cases and customer needs.
- [ ] Extract customer-language and keyword candidates.
- [x] Build the month-aspect mart with normalized share and month-over-month
  changes; formal emerging-issue alert thresholds remain open.
- [x] Add workspace, model, taxonomy, policy, date-scope, and evidence
  provenance to the demo outputs.
- [x] Reconcile product and monthly marts against all 236,316 source
  review-aspect observations.

## Phase 08 — Evidence Embeddings and Retrieval

Status: `IN PROGRESS`

- [x] Select the local TF-IDF/SVD baseline and NumPy-backed vector storage.
- [x] Define the first review/evidence representation.
- [x] Generate 128-dimensional embeddings for all 118,804 niche reviews.
- [x] Track representation, source, index version, and artifact hashes.
- [x] Implement metadata filtering.
- [x] Implement lexical retrieval.
- [x] Add local dense retrieval and hybrid score fusion.
- [x] Remove exact normalized-text duplicates and diversify products.
- [ ] Evaluate relevance, scope correctness, and evidence diversity.

## Phase 09 — Analytics API and Vertex AI Analyst

Status: `IN PROGRESS`

- [x] Implement typed analytics API contracts.
- [ ] Configure Vertex AI authentication and reusable client.
- [x] Configure Gemini API authentication and reusable client.
- [x] Define structured analyst response schemas.
- [x] Implement initial deterministic query understanding.
- [ ] Implement multi-step product comparisons.
- [x] Connect evidence retrieval.
- [x] Require scope, metrics, evidence, and limitations in answers.
- [ ] Track token use, latency, and cost.
- [ ] Build analyst evaluation cases.
- [x] Reject absent and malformed review citations.
- [ ] Test broader hallucination and unsupported-claim behavior.

## Phase 10 — FastAPI and Seller Application

Status: `IN PROGRESS`

Backend capabilities:

```text
workspaces
category and product search
seller/competitor scope
product analytics
niche analytics
comparisons
trends
review evidence
AI Analyst questions
```

Application capabilities:

- [x] Create and save a seller workspace.
- [ ] Select niche and products (product search is complete; niche selection is open).
- [x] View product and competitor dashboards.
- [x] Explore strengths, complaints, and gaps.
- [x] Explore customer phrases and evidence.
- [x] Ask grounded follow-up questions.
- [x] Display data scope and limitations.
- [x] Export selected analytical results.

## Phase 10A — Expand from Conditioners to Shampoo & Conditioner

Status: `ANALYTICS COMPLETE — HARDENING IN PROGRESS; DOCKER DEFERRED`

The next catalog expansion is the complete Amazon section:

```text
Beauty & Personal Care
└── Hair Care
    └── Shampoo & Conditioner
```

It contains 40,402 catalog products, 21,651 reviewed parent products, and
358,925 reviews in the accepted historical snapshot. The existing
`hair_conditioners_v1` population and artifacts remain versioned and unchanged
while the broader section is built and evaluated.

### Catalog grouping rules

The scalable hierarchy is:

```text
dataset category
    → Amazon category path
    → analytical section
    → model family
    → exact competitor niche
    → parent_asin
    → asin variation, when analytically meaningful
```

- An **analytical/model family** groups products that customers discuss using
  mostly the same criteria.
- An **exact competitor niche** contains products that a buyer could plausibly
  choose instead of one another. Product comparisons and rankings stay within
  this scope by default.
- Amazon paths define source membership; analytical families do not overwrite
  Amazon's catalog classification.
- Taxonomies are hierarchical: shared Hair Care aspects, family-specific
  aspects, and optional niche-specific aspects.
- One universal Hair Care model will not be assumed to work equally well for
  every product type. Shared models are allowed only after fixed-set
  evaluation shows that quality is adequate.

### Initial Shampoo & Conditioner families

| Model family | Amazon leaves | Reviewed products | Reviews |
|---|---|---:|---:|
| Cleansing products | Shampoos, 2-in-1, 3-in-1 | 9,522 | 132,731 |
| Conditioning products | Conditioners, Deep Conditioners | 7,684 | 122,891 |
| Dry shampoos | Dry Shampoos | 686 | 17,684 |
| Product sets | Shampoo & Conditioner Sets | 3,653 | 85,049 |
| Unresolved generic rows | Generic Shampoo & Conditioner path | 106 | 570 |

The 570 unresolved generic-path reviews are quarantined until their product
metadata can place them safely; they do not silently enter a competitor pool.
The family split is an initial design hypothesis and must be validated by the
evaluation below.

### Execution plan

- [x] Register a versioned `shampoo_and_conditioner_v1` section configuration
  and its exact Amazon path membership.
- [x] Materialize and reconcile all 358,925 eligible reviews and 21,651
  reviewed parent products without changing `hair_conditioners_v1`.
- [x] Persist the family and exact-competitor-niche fields in product scope
  metadata; report unclassified and ambiguous products explicitly.
- [x] Build a fixed evaluation set of 300 representative reviews plus 200
  targeted diagnostics from smaller or failure-prone leaves; verify zero
  discovery overlap and zero component overlap.
- [x] Run the current Conditioner extractor on that set as a frozen coverage
  baseline and report matches separately for every family and leaf.
- [x] Measure precision, recall, and F1 on the independent 500-review Gemini
  silver set separately for every family. The selected shared extractor has
  F1 0.640 overall and 0.632–0.663 across the four resolved families; these
  are diagnostic silver-agreement metrics, not human-gold accuracy.
- [x] Scan all 358,925 reviews locally for vocabulary and frequency signals and
  build an adaptive stratified discovery sample, expanded from 5,000 to 6,000
  reviews across every family and leaf, and use privacy-minimized Gemini labels
  for iteration.
- [x] Run discovery in waves: begin with 3,000 reviews, add 1,000, and then add
  batches of up to 500–1,000 only while they continue to produce meaningful
  new canonical aspects or close family/leaf coverage gaps.
- [x] Measure taxonomy saturation after every wave and stop when an additional
  500 reviews yield no material aspect and every family/leaf has adequate
  coverage; record the stopping evidence in the discovery report. At 5,000
  reviews, 45 of the final 46 aspects met support thresholds; the final 1,000
  added only Product Concentration & Yield, and all 46 then met the thresholds.
- [x] Audit and extend the aspect taxonomy for cleaning/lather, oil absorption,
  white residue, aerosol/application, and multi-product attribution where the
  evidence supports them. Taxonomy V1 contains 46 reviewed aspects; 17 generic,
  catalog, or fulfillment labels are explicitly excluded.
- [x] Train the shared section extractor only after fixed-set evaluation. The
  close family results do not currently justify separate family models; the
  one physical model artifact is stored as `shampoo-conditioner-aspect-extractor`.
- [x] Process the complete 358,925-review section into 855,465 versioned
  review-aspect observations, product/month marts, exact-niche rankings, and a
  358,923-review local evidence index.
- [x] Restrict product comparisons to exact competitor niches while permitting
  higher-level section benchmarks where definitions remain comparable. Only
  strong/usable aspects enter product rankings.
- [x] Add section-wide product search and automatic family/niche selection to
  the application without exposing internal workspace or artifact names.
- [x] Re-run a fixed 20-question AI Analyst benchmark spanning shampoo, dry
  shampoo, 2-in-1 products, and sets: 20/20 routes, 20/20 retrieval cases, and
  20/20 structurally validated Gemini answers with valid review citations.
- [ ] Proceed to Docker and the demonstration deployment only after data,
  model, analytics, retrieval, API, and UI reconciliation gates pass.

Provisional engineering thresholds, to be validated rather than treated as
product truth:

- about 3,000–5,000 reviews can justify a separately measured competitor
  niche;
- about 20,000 reviews can justify testing a separate family-level model;
- a smaller niche may reuse its parent-family model, but keeps its own
  competitor pool and evaluation slice.

More data improves coverage and support, but does not automatically improve
analytical quality. Quality is accepted only from per-family and per-leaf
evaluation; the project will not report an estimated gain or loss before that
benchmark exists.

The fixed 300 representative plus 200 targeted evaluation rows
are independent of taxonomy discovery: they must not be used to invent, merge,
or tune aspects. If coverage diagnostics require it, the independent set may
grow to at most approximately 600 rows while preserving that separation.

## Phase 11 — Testing, Deployment, and Operations

Status: `IN PROGRESS`

- [x] Unit tests.
- [x] Data-contract tests.
- [x] ML evaluation tests.
- [x] Analytics reconciliation tests.
- [x] Retrieval and RAG evaluation (20/20 section benchmark cases routed,
  retrieved, generated, and citation-validated).
- [x] API tests.
- [ ] Agent/tool tests.
- [x] End-to-end seller scenarios.
- [ ] Docker images.
- [ ] GCP deployment.
- [ ] Secret management.
- [ ] Logging and monitoring.
- [ ] Offline job monitoring.
- [x] Dataset, pipeline, model, taxonomy, and index versioning.

## Phase 12 — Second Amazon Category

Status: `DEFERRED`

After the Beauty MVP is stable, run the same pipeline on a substantially
different Amazon Reviews 2023 category.

Success does not mean zero category-specific artifacts. It means:

```text
same ingestion contracts
same pipeline code
same analytics interfaces
same application
different category configuration
different taxonomy and, when justified, different models
```

---

# 15. Notebook Strategy

Notebook numbering is local to a research area rather than one global sequence:

```text
notebooks/
├── 01_data_foundation/
├── 02_sentiment/
├── 03_aspects/
├── 04_analytics/
├── 05_retrieval/
└── category_onboarding/
```

The original `01–08` notebooks have been moved and renamed according to their
responsibility. Their contents remain subject to the audit in
`docs/NOTEBOOK_AUDIT.md` and will be refactored in dependency order, not all at
once.

Notebooks are appropriate for:

- exploratory data analysis;
- experimental comparison;
- model training and evaluation;
- error analysis;
- taxonomy discovery analysis;
- retrieval and RAG evaluation;
- communicating conclusions and limitations.
- visually inspecting category data, products, review examples, distributions,
  taxonomy candidates, and model errors during onboarding.

Notebooks are not the production implementation of:

- schemas;
- adapters;
- ingestion pipelines;
- analytics functions;
- model inference services;
- API logic;
- reusable LLM clients.

## Category Growth Strategy

Adding a category does not create a copied set of production notebooks. It
creates category-specific configuration and artifacts while reusing the shared
pipeline:

```text
category configuration
+ dataset manifest
+ taxonomy
+ evaluation artifacts
+ optional category-specific models
        ↓
shared pipeline in src/
        ↓
versioned category analytics and reports
```

Parameterized notebooks under `notebooks/category_onboarding/` provide a human
inspection layer for every category:

```text
01_category_profile.ipynb
02_taxonomy_review.ipynb
03_model_evaluation.ipynb
04_launch_readiness.ipynb
```

The notebook templates remain shared. Executed reports are stored by category
and dataset version. A new notebook is created only for a genuinely new research
question or category-specific failure mode, not merely to rerun existing code.

## Refactoring Order

```text
01_data_foundation
        ↓
Amazon category registry and full catalog
        ↓
02_sentiment reevaluation
        ↓
03_aspects
        ↓
04_analytics
        ↓
05_retrieval and Gemini evaluation
```

Candidate future experimental sequence:

```text
Amazon category and catalog EDA
        ↓
Sentiment audit and error analysis
        ↓
Aspect taxonomy discovery
        ↓
Aspect extraction evaluation
        ↓
Aspect sentiment evaluation
        ↓
Seller analytics validation
        ↓
Evidence retrieval evaluation
        ↓
Vertex AI Analyst evaluation
```

All notebooks must follow `docs/CODE_STYLE.md`.
The current layout and category-onboarding rules are documented in
`notebooks/README.md`.

---

# 16. Evaluation and Success Criteria

## 16.1 Data Quality

- reproducible row counts for every processing stage;
- valid review/product identifiers;
- measured join coverage between reviews and metadata;
- explicit missing-metadata rates;
- no silent record loss;
- versioned input and output schemas.

## 16.2 ML Quality

- sentiment metrics on a fixed test set;
- aspect extraction precision, recall, F1, and coverage;
- aspect sentiment metrics;
- error analysis by product, aspect, rating, time, and review length;
- model confidence interpreted or calibrated;
- no unmeasured cross-category deployment assumptions.

Final metric thresholds will be set after evaluation datasets are audited or
created. The plan does not invent target numbers before a trustworthy benchmark
exists.

## 16.3 Analytics Quality

- all aggregate metrics reconcile with eligible source rows;
- product and competitor scopes are correct;
- rankings enforce minimum support;
- trends distinguish counts from shares;
- every insight contains population size and provenance;
- examples do not contradict the aggregate conclusion.

## 16.4 Retrieval and AI Quality

- retrieved reviews match requested products and filters;
- evidence is relevant and diverse;
- answers use tool-provided statistics without changing them;
- claims are supported by metrics or review evidence;
- historical-data limitations are stated when relevant;
- the analyst refuses unsupported Amazon Ads or live-marketplace claims.

## 16.5 MVP Product Success

The MVP succeeds when an Amazon seller can:

1. select a Beauty niche;
2. define one or more own products and competitors;
3. inspect reliable product, niche, aspect, complaint, trend, and gap analytics;
4. ask natural-language questions through Vertex AI / Gemini;
5. receive answers grounded in deterministic metrics and traceable review
   evidence;
6. discover actionable product-improvement and customer-language opportunities
   without the system overstating what the dataset proves.

---

# 17. Locked Decisions

The following decisions are accepted for the current product:

```text
Primary users: Amazon sellers and brand/product analysts
Primary source: Amazon Reviews 2023
Initial dataset category: Beauty_and_Personal_Care
Historical first niche: hair_conditioners_v1
Active MVP section: shampoo_and_conditioner_v1
Comparison scope: exact Amazon competitor niche inside the active section
V1 snapshot: Beauty_and_Personal_Care, 2021-01-01 through 2023-09-13
Classification: Amazon dataset/category metadata
Default product join and analysis unit: parent_asin
Core analytical layer: aspect-based review intelligence
Heavy computation: offline
Numerical facts: deterministic Python analytics
LLM platform: Vertex AI / Gemini
LLM role: reasoning, orchestration, discovery, and explanation
RAG role: review evidence, not aggregate calculation
Core-code strategy: shared across Amazon categories
Category strategy: category-aware taxonomies/configuration/models
Project modes: learning + portfolio + potential commercial product
Educational style: English technical terms with detailed Russian explanations
Notebook/comment style: meaningful explanations in English, then Russian
Artifact hygiene: freeze lineage-bearing historical evidence; remove only
explicitly temporary, reproducible outputs after verification
```

---

# 18. Open Product and Engineering Decisions

These product questions remain open after the Shampoo & Conditioner expansion:

1. Should users be able to override automatically selected competitors?
2. Which Amazon category field is sufficiently complete for niche selection,
   and what is the explicit fallback when `categories` is missing?
3. Which review filters are defaults: verified purchase, language, minimum text
   length, helpfulness, or no additional filter?
4. Should variation-level `asin` comparisons be part of the MVP or deferred?
5. Which aspect extraction and aspect sentiment approaches will meet both
   quality and offline processing cost requirements?
6. Which vector store and GCP services fit the expected corpus size and budget?
7. Which customer-phrase outputs are useful without implying search-volume or
   advertising-performance data?
8. What evidence-display and review-text usage rules apply to the final UI?

---

# 19. Current Repository Status

Observed artifacts:

```text
Beauty_and_Personal_Care review data
Beauty_and_Personal_Care item metadata
verified canonical-review, category-registry, and product-catalog artifacts
UTC-stable 300k V2 and 1M V3 sentiment manifests
verified TF-IDF V2 and 1M V3 predictions and metric reports
sentiment-baseline and sentiment-transformer model directories
beauty_distilbert_rating_sentiment_1m_v2 predictions, error analysis, and blind queue
corrected V3 aspect-discovery sample and batch plan
corrected V2 aspect-extraction evaluation sample and blind queue
ten current notebooks across data foundation, sentiment, and aspects
tested reusable data, catalog, sampling, ML, and annotation utilities
```

Acceptance status:

```text
Product direction                           VERIFIED
Amazon Reviews 2023 source                 VERIFIED
Canonical Beauty review dataset            VERIFIED; STRICTLY REMATERIALIZED
Category registry and Product Catalog      VERIFIED; FULL VERSIONED ARTIFACTS
Five data-foundation notebooks             CLEAN-RUN VERIFIED
TF-IDF notebook and 1M V3 baseline         CLEAN-RUN VERIFIED
DistilBERT V2 notebook/model                CLEAN-RUN VERIFIED; LIMITED HUMAN DIAGNOSTIC COMPLETE
Refreshed Error Analysis notebook           CLEAN-RUN VERIFIED; 20/156 HUMAN LABELS COMPLETE
Two aspect notebooks                        CLEAN-RUN VERIFIED; 240 SILVER LABELS COMPLETE
All ten notebooks                           10 CLEAN-RUN VERIFIED
Legacy sentiment metrics and 158 labels     HISTORICAL DIAGNOSTICS ONLY
Aspect discovery V3                         COMPLETE; TAXONOMY V2 APPROVED (39 ASPECTS)
Extraction evaluation V3                    TAXONOMY V2 ALIGNED; 240 SILVER LABELS COMPLETE
Aspect extraction/sentiment baselines       HYBRID V2; FULL NICHE DIAGNOSTIC COMPLETE
Offline seller analytics                    FIRST END-TO-END SELLER DEMO COMPLETE
Evidence retrieval and RAG                  IN PROGRESS; GROUNDED GEMINI PATH COMPLETE
Gemini AI Analyst                           20/20 FIXED BENCHMARK ANSWERS GENERATED AND CITATION-VALIDATED
Seller application                          IN PROGRESS; AI ANALYST UI/API COMPLETE
```

Immediate next action:

```text
finish reliability, lineage, performance, and semantic regression checks for
the completed Shampoo & Conditioner section; Docker remains deferred by the
project owner
```
