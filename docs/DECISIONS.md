# Review Intelligence Platform --- Architecture Decisions

This document records important architectural decisions so that the
project does not lose its design rationale during development.

------------------------------------------------------------------------

# ADR-001 --- Use Amazon Reviews 2023 as the baseline dataset

## Decision

Use Amazon Reviews 2023 as the main reproducible research dataset.

Primary analytical window:

``` text
2021-01-01 → 2023-09-13
```

## Reason

The project needs a fixed dataset for reproducible experiments, model
evaluation and portfolio demonstration.

## Consequence

Results can be reproduced without depending on continuously changing
external data.

------------------------------------------------------------------------

# ADR-002 --- Use category-specific ML models

## Decision

ML models may be trained separately for each Amazon product category.

Example:

``` text
Beauty model
Electronics model
Automotive model
Home & Kitchen model
```

## Reason

Sentiment language, vocabulary and data distributions can differ
substantially between categories.

A Beauty-trained model should not automatically be assumed to perform
optimally on Electronics.

## Consequence

The project uses one reusable training pipeline but stores
category-specific model artifacts.

------------------------------------------------------------------------

# ADR-003 --- Use DistilBERT as the Beauty V1 overall-text-sentiment model

> Status: superseded on 2026-08-18. The model, metrics, and 158-row manual
> review cited below are historical artifacts tied to timezone-dependent split
> membership. They are not acceptance evidence for the UTC-stable refresh.

## Decision

Accept the verified DistilBERT artifact as the first Beauty Overall Sentiment
Model, version `beauty_distilbert_rating_sentiment_1m_v1`.

Current Beauty result:

``` text
Held-out Beauty parent products, balanced macro F1: 0.8187
Later 2023 reviews, balanced macro F1:               0.8172
Held-out Conditioner parent products, macro F1:      0.8146
```

## Reason

Overall sentiment is still useful for:

-   sentiment distribution;
-   product comparison;
-   trends;
-   filtering;
-   negative-review detection.

In that historical evaluation, DistilBERT outperformed the TF-IDF reference on
every test split. In a
completed 158-row diagnostic sample of model disagreements with rating-derived
weak labels, human labels matched DistilBERT on 108 rows and the weak labels on
46 rows. The sample is intentionally enriched for disagreements, so these
counts support error interpretation but are not an unbiased accuracy estimate.

## Consequence

The model remains one signal rather than the complete Review Intelligence
solution. Rating-derived targets are documented as weak labels, Neutral scores
are not presented as guaranteed human confidence, and seller attention remains
a separate deterministic rating signal under ADR-019. The current model
decision is recorded by the later UTC-refresh ADR and its versioned report.

------------------------------------------------------------------------

# ADR-004 --- Make Aspect-Based Analysis the core analytical layer

## Decision

The central analytical task is:

``` text
Aspect Discovery
→ Aspect Extraction
→ Aspect Sentiment
→ Aggregation
```

## Reason

Business users need to know not only whether customers are positive or
negative, but:

``` text
What do customers like?

What do customers dislike?

Why?

Which product characteristics generate complaints?

Which characteristics generate satisfaction?
```

## Consequence

Future ML development prioritizes aspect intelligence over further
optimization of overall sentiment alone.

------------------------------------------------------------------------

# ADR-005 --- Use the Product Catalog before review analysis

## Decision

Resolve a user's product/category query against product metadata before
selecting reviews.

Example:

``` text
"hair conditioners"
      ↓
Product Catalog
      ↓
product_id
      ↓
reviews
```

## Reason

Review text does not always explicitly identify the product type.

## Consequence

Product metadata becomes a key component of query routing and review
selection.

------------------------------------------------------------------------

# ADR-006 --- Process all relevant reviews for aggregate analytics

## Decision

When the user asks for aggregate information about a product group,
analyze the full relevant review population available in the dataset
rather than retrieving only a small semantic subset.

## Reason

Using only top-K semantic retrieval for aggregate statistics can
introduce selection bias.

## Consequence

Large-scale ML inference and analytics are performed offline.

RAG is used later for evidence and explanation.

------------------------------------------------------------------------

# ADR-007 --- Python calculates deterministic facts

## Decision

Use Python/analytics code for numerical calculations.

Examples:

``` text
review counts
sentiment percentages
aspect frequencies
rating averages
time trends
product comparisons
```

## Reason

These calculations must be deterministic, reproducible and auditable.

## Consequence

Gemini receives structured analytical results instead of being asked to
calculate raw statistics from millions of reviews.

------------------------------------------------------------------------

# ADR-008 --- Gemini is not the per-review production classifier

## Decision

Do not send every review to Gemini by default.

## Reason

Processing millions of reviews with an LLM would be:

-   expensive;
-   slower;
-   harder to reproduce;
-   unnecessarily dependent on LLM inference.

## Consequence

Use ML models for large-scale structured processing.

Use Gemini for discovery, normalization, synthesis, reasoning and
explanation.

------------------------------------------------------------------------

# ADR-009 --- Use Gemini for aspect discovery on representative samples

## Decision

Use a representative sample of reviews to discover candidate aspects.

``` text
Large Dataset
      ↓
Representative Sample
      ↓
Gemini
      ↓
Candidate Aspect Taxonomy
```

## Reason

The full dataset may contain millions of reviews, while a representative
sample can reveal the dominant topics much more efficiently.

## Consequence

The discovered taxonomy becomes structured category configuration/data
and is used by the large-scale extraction pipeline.

------------------------------------------------------------------------

# ADR-010 --- Normalize aspect vocabulary

## Decision

Different expressions referring to the same concept should map to one
canonical aspect.

Example:

``` text
smell
scent
fragrance
```

→

``` text
fragrance
```

## Reason

Without normalization, aggregation would fragment the same customer
concern across multiple labels.

## Consequence

Embeddings and/or Gemini can support normalization.

------------------------------------------------------------------------

# ADR-011 --- RAG is for evidence, not aggregate analytics

## Decision

Use RAG when users need examples, explanations or evidence from actual
reviews.

## Reason

Analytics and retrieval solve different problems.

Analytics:

``` text
What is happening?
```

RAG:

``` text
Why?
Show me examples.
What do customers actually say?
```

## Consequence

The system can provide both numerical evidence and representative review
evidence.

------------------------------------------------------------------------

# ADR-012 --- Use a shared pipeline for category changes

## Decision

Changing the product category must not require rewriting the core code.

Example:

``` text
Beauty config
      ↓
Beauty dataset
      ↓
Beauty models
      ↓
Beauty analytics
```

The same architecture is used for:

``` text
Electronics
Automotive
Home & Kitchen
...
```

## Reason

The project should demonstrate reusable engineering rather than
duplicated category-specific code.

## Consequence

Category-specific behavior is controlled through configuration, model
artifacts and taxonomy.

------------------------------------------------------------------------

# ADR-013 --- Version datasets and models

## Decision

Every model is associated with a dataset version.

Example:

``` text
amazon_reviews_2023
      ↓
beauty_sentiment_v1
```

## Reason

Model results must be reproducible and traceable.

## Consequence

Metadata must record:

-   dataset version;
-   training date;
-   training window;
-   category;
-   model version;
-   evaluation metrics.

------------------------------------------------------------------------

# ADR-014 --- Do not automatically retrain when new data appears

## Decision

New datasets are first used for evaluation.

``` text
Existing Model
      ↓
New Data
      ↓
Evaluation
```

Retraining is performed only when justified.

## Triggers

``` text
performance degradation
data drift
vocabulary changes
new product types
```

## Consequence

Model lifecycle becomes:

``` text
Train
 ↓
Evaluate
 ↓
Deploy
 ↓
Monitor
 ↓
Retrain when necessary
 ↓
New version
```

------------------------------------------------------------------------

# ADR-015 --- Future datasets can extend the fixed 2023 baseline

## Decision

Future datasets such as Amazon Reviews 2024, 2025 and 2026 are treated
as new dataset versions rather than silently replacing the baseline.

## Reason

The 2023 dataset provides a stable reproducible benchmark.

New data can be evaluated against the existing models.

## Consequence

The project can demonstrate model drift and controlled retraining.

------------------------------------------------------------------------

# ADR-016 --- Separate offline and online workloads

## Decision

Large-scale ML processing is performed offline.

User requests operate mainly on precomputed analytical data.

## Reason

Running inference over hundreds of thousands or millions of reviews for
every user query would be inefficient.

## Consequence

The architecture becomes:

``` text
OFFLINE

Dataset
 ↓
ML
 ↓
Aspect Analysis
 ↓
Embeddings
 ↓
Analytics
 ↓
Persist


ONLINE

User
 ↓
Product Selection
 ↓
Analytics
 ↓
RAG
 ↓
Gemini
 ↓
Answer
```

------------------------------------------------------------------------

# ADR-017 --- Business Intelligence is the final product

## Decision

The final product is not a sentiment classifier.

The final product is a business intelligence system built on review
data.

## Target outputs

``` text
Customer strengths
Customer complaints
Aspect popularity
Aspect sentiment
Pain points
Product comparisons
Trends
Emerging issues
Representative evidence
AI-generated explanations
```

## Consequence

All future architecture decisions should be evaluated against this
objective.

------------------------------------------------------------------------

# ADR-018 --- Develop as a learning, portfolio, and potential commercial project

## Decision

Develop the repository simultaneously as:

``` text
Learning project
Portfolio project
Potential commercial product, if seller value is validated
```

Keep technical identifiers and public code contracts in English. Write
educational notebook explanations and meaningful non-obvious comments in
English followed by detailed Russian.

## Reason

The owner is learning the technologies while building the system and needs to
understand the reasons behind implementation decisions. The repository must
also remain professional enough to demonstrate engineering skill and provide a
credible foundation for a future seller-facing product.

## Consequence

Notebooks and work handoffs explain terminology, assumptions, trade-offs, and
verification rather than presenting unexplained code. Reusable modules remain
clean, tested, and conventional; bilingual detail is concentrated around
meaningful decisions instead of obvious syntax. Commercial-scale complexity is
added only when validation or real users justify it.

------------------------------------------------------------------------

# ADR-019 --- Separate textual sentiment from seller attention

## Decision

Keep textual sentiment and seller attention as separate analytical fields.

``` text
Text sentiment:
negative / neutral / positive

Seller attention:
1–3 stars → needs_attention
4–5 stars → satisfied
```

The first seller-attention policy is versioned as
`amazon_rating_attention_v1`. It is calculated directly from the available
Amazon rating and does not require another ML model.

## Reason

A three-star review can be neutral in wording but still matter commercially:
Amazon sellers generally want to understand what prevents a product from
receiving four or five stars. Relabeling every Neutral text as Negative would
mix the meaning of the language with the seller's business priority and make
both measurements less clear.

## Consequence

Review-level analytical data preserves all of the following independently:

``` text
rating
seller_attention
overall_sentiment
aspect
aspect_sentiment
```

Seller dashboards and Gemini may use `needs_attention` to select improvement
opportunities. They must not describe it as proof that the text is negative.
If the business threshold changes later, a new policy version is created
instead of silently changing historical results.

------------------------------------------------------------------------

# ADR-020 --- Approve Hair Conditioners V1 and its historical discovery sample

> Historical decision: the niche boundary remains approved, but ADR-024
> supersedes the V1 sampling design for all quantitative inference.

## Decision

Use the complete Amazon leaf below as the first MVP niche:

``` text
Beauty & Personal Care
  > Hair Care
  > Shampoo & Conditioner
  > Conditioners
```

The scope is versioned as `hair_conditioners_v1` and resolved by the stable
category-path ID `amazon-category:9efc15d3fc9a2cbf3b001316`. Related Amazon
leaves for deep conditioners, 2-in-1 products, beard conditioners, and color
conditioners are excluded.

The initial Aspect Taxonomy Discovery used a deterministic 1,500-review sample,
stratified by rating group, review year, and review length, with a five-word
minimum and a cap of five selected reviews per parent product. This historical
design is retained for qualitative provenance only; ADR-024 replaces it for
every population estimate.

## Reason

The exact leaf provides a reproducible metadata-first boundary and a substantial
population: 13,985 catalog products, 7,382 reviewed products, 8,921 reviewed
variation ASINs, and 118,804 canonical reviews. Narrowing the scope with title
keywords before discovering the actual product/aspect structure would introduce
an unmeasured classification rule.

The coverage floor exposed rare negative, three-star, recent, and long-review
strata to discovery, and the cap improved qualitative product diversity.
However, the recorded `N_h / n_h` weights omitted the probability of passing
that cap. They therefore do not identify review-population prevalence.

## Consequence

Products such as leave-in conditioners, detanglers, children's products, bars,
and sets remain included when Amazon places them in the approved leaf. They may
be represented as use cases or product subtypes in the taxonomy. A narrower
niche requires a new explicit, evaluated niche version rather than a silent
keyword filter.

The saved discovery sample is suitable for qualitative candidate generation,
not aggregate prevalence claims. Corrected quantitative discovery must use the
review-level probability design in ADR-024 and undergo a new model run and
human review.

------------------------------------------------------------------------

# ADR-021 --- Accept the V2 Free Tier discovery run as the taxonomy input

> Historical decision: V2 remains the qualitative source of Taxonomy V1, but
> ADR-024 invalidates its weighted support as a review-population estimate.

## Decision

Use `hair_conditioners_discovery_v2` as the sole model-generated input to the
first Hair Conditioners taxonomy review. It processed the fixed 1,500-review
sample in 15 batches of 100 with `gemini-3.5-flash-lite` and produced a
lineage-tracked 23-aspect candidate proposal.

Keep the incomplete `hair_conditioners_discovery_v1` responses as an audit
artifact, but do not combine them with V2. The V1 pilot stopped after 11 of 60
requests when its model-specific Free Tier daily limit was reached.

## Reason

One homogeneous complete run is easier to reproduce and audit than a mixture
of outputs from different models and batch strategies. Larger V2 batches
completed the fixed sample within the available experimental request allowance.
All retained evidence remains grounded in exact review substrings; 43
ungrounded phrases were rejected rather than repaired semantically.

## Consequence

V2 is retained only as qualitative provenance for Taxonomy V1. Its 23 proposed
aspects, source-label merges, definitions, exclusions, and exact evidence were
human-reviewed, but its weighted support/share fields are invalid. Corrected
quantitative discovery requires fresh V3 responses and a new human review; it
still does not replace full-corpus extraction.

------------------------------------------------------------------------

# ADR-022 --- Approve Hair Conditioners Taxonomy V1

## Decision

Approve `hair_conditioners_taxonomy_v1` after human review of every row in the
23-aspect Gemini proposal. Materialize 33 narrower canonical aspects from 3
unchanged approvals, 7 edits, 11 splits, and 2 exclusions.

Assign each of the 80 normalized discovery keys exactly once: 74 keys belong to
approved aspects and 6 are explicitly excluded. Require aliases to be unique
across aspects and retain labels, aliases, exclusions, exact evidence, and
lineage hashes. Preserve the historical weighted fields only with an explicit
invalidity marker.

## Reason

The candidate proposal combined several related but operationally different
dimensions, including product texture versus hair feel, detangling versus slip,
packaging integrity versus dispenser function, color protection versus toning,
and price versus value for money. Broad generic labels would hide the action a
seller should take and create semantic overlap in later analytics.

## Consequence

`reports/aspects/hair_conditioners_taxonomy_v1.json` is the approved qualitative
taxonomy input for extraction experiments. It is not quantitative prevalence
evidence. The corrected V2 extraction queue must be human-reviewed for
multi-aspect labels and exact evidence spans.

------------------------------------------------------------------------

# ADR-023 --- Separate representative and rare-aspect extraction evaluation

> Superseded by ADR-024 for representative sampling and weight semantics.
> This section records the historical V1 decision; V1 is frozen and
> unannotated.

## Decision

Use `hair_conditioners_extraction_eval_v1` as a 240-review human gold sample:
160 reviews stratified by rating group, year, and length, plus 80 reviews
targeted by taxonomy-alias matches. Exclude every review used by
`hair_conditioners_discovery_v2`, prevent overlap between components, and cap
the combined sample at two reviews per parent product.

Keep target-aspect hints in the audit-only Parquet artifact and remove them from
the human CSV queue. Require exact case-sensitive evidence substrings for every
annotated aspect.

## Reason

A representative-only sample may contain too few examples to diagnose rare
aspects, while a targeted sample cannot estimate natural aggregate performance.
Keeping the components separate supports both realistic headline metrics and
per-aspect stress testing. Hiding heuristic target labels reduces anchoring
bias during human annotation.

## Consequence

Representative and targeted metrics must be reported separately. Alias matches
are sampling aids, never gold labels. The 240 rows cannot be used both to tune
an extractor and to claim final evaluation performance.

The current artifact is `hair_conditioners_extraction_eval_v2`: its 160
representative rows are probability sampled without a product cap and their
weights reconcile to 112,525 eligible holdout reviews; its 80 targeted rows are
unweighted diagnostics. All 240 human labels are pending.

The Taxonomy-V2-aligned successor is `hair_conditioners_extraction_eval_v3`.
All 240 rows now have Gemini silver labels; human gold remains an optional gate
for formal model acceptance rather than a blocker for iterative development.

------------------------------------------------------------------------

# ADR-024 --- Use review-level probability samples for population estimates

## Decision

Use stratified simple random sampling without replacement for discovery and
the representative evaluation component. Within stratum `h`, select `n_h`
reviews uniformly from `N_h` eligible reviews and record weight `N_h / n_h`.
Do not apply a per-product cap to a component used for population estimates.

Keep rare-aspect targeting as a separate non-probability component. A targeted
row has no sampling weight and can support only diagnostic, per-aspect results.
Require dataset, niche, sample-schema, discovery/taxonomy version, provider,
backend, and requested-model identities at every artifact boundary.

## Reason

The legacy V1 discovery and evaluation samplers capped reviews per product
before assigning `N_h / n_h` weights. Those weights omitted the probability of
surviving the cap and could severely bias prevalence estimates. A review-level
stratified design has a known inclusion probability and its weights reconcile
exactly to the eligible population.

## Consequence

The V1 sample, V2 discovery responses, and Taxonomy V1 labels remain frozen
historical evidence. Their weighted support and share fields are explicitly
invalid for review-population inference. The corrected chain starts with
`aspect_discovery_sample_v2` and `hair_conditioners_discovery_v3`; it requires
new Gemini responses and a new human taxonomy review rather than relabeling old
outputs. Representative V2 evaluation metrics may be population-weighted;
targeted metrics must always be reported separately and unweighted.

------------------------------------------------------------------------

# ADR-025 --- Rematerialize the fixed snapshot under its declared physical contracts

## Decision

Treat the 2026-08-18 rebuild as an explicitly reviewed rematerialization of
`amazon_reviews_2023_beauty_2021_2023_v1`, not as a new source dataset. Keep the
logical dataset version because the source files, date window, accepted review
IDs, and row values are unchanged. Update the manifest sizes and SHA-256 values
for canonical reviews, the category registry, and the product catalog.

Require the rebuilt artifacts to satisfy all of the following before any
downstream run:

- mandatory Arrow fields are physically non-nullable;
- optional bounded integers cannot crash ingestion;
- anonymous exact duplicates are removed by content identity;
- registry and catalog keys share NFKC, whitespace, and case normalization;
- `parent_asin` is unique in the full catalog;
- manifest size, checksum, and record-count validation passes for every
  registered artifact.

## Reason

The previous Parquet bytes did not fully implement the already-declared schema:
mandatory fields were physically nullable. The parser and key normalization
also needed correctness hardening. Those are materialization defects, not a new
Amazon snapshot or a new analytical population. Silently replacing bytes would
still break provenance, so the manifest update and rebuild are recorded as one
reviewed migration.

## Consequence

The former binary hashes are historical and are not valid inputs for new
quantitative claims. Downstream samples that reproduce the same membership may
retain their own version IDs only when their content hashes and identities
remain unchanged. Models whose split membership changed for an independent UTC
boundary correction must use new model versions and fresh evaluation reports.

------------------------------------------------------------------------

# ADR-026 --- Refresh sentiment lineage without inheriting model acceptance

## Decision

Use `beauty_rating_sentiment_split_v2` and
`beauty_rating_sentiment_split_1m_v3` as the only current sentiment manifests.
Interpret the temporal cutoff as midnight UTC in both split construction and
loading, and compute percentile confidence intervals by resampling whole
`parent_asin` clusters.

Store refreshed artifacts under new identities:

- `beauty_tfidf_rating_sentiment_v2` for the 300k reference;
- `beauty_tfidf_rating_sentiment_1m_v3` for the direct 1M comparison;
- `beauty_distilbert_rating_sentiment_1m_v2` for the Transformer candidate.

Treat the Transformer as `candidate_pending_human_review` until a blind queue
generated from its own predictions is labeled. Never transfer the completed
158-row legacy review or its acceptance decision to the refreshed model.

The Conditioner evaluation holds out a defined set of Hair Conditioners
`parent_asin` values. It does not exclude the whole niche; other Conditioner
products remain in the training population.

## Reason

The legacy timezone-free cutoff was interpreted in the DuckDB session
timezone, changing split membership across environments. Its row bootstrap
also understated product-level dependence. Because both the training lineage
and uncertainty method changed, reusing the former model identity or human
decision would make the new result irreproducible.

The refreshed 1M TF-IDF baseline reaches balanced macro F1 0.78717 on held-out
Beauty parent products, 0.78435 on later 2023 reviews, and 0.77808 on held-out
Conditioner parent products. These values and their cluster intervals come
from its versioned report, not from notebook prose.

## Consequence

Historical V1/V2 splits, models, predictions, metrics, error analysis, and the
158 labels remain immutable audit evidence but are not current acceptance
evidence. The new Transformer report and automatic error analysis may support
technical comparison; promotion still requires the refreshed human gate. All
persisted lineage references use repository-relative paths and input split
artifacts are immutable once created.

------------------------------------------------------------------------

# ADR-027 --- Use silver labels for iteration, not final acceptance

## Decision

Train `hair_conditioners_aspect_extractor_v2` from the 240-row Gemini silver
reference using character TF-IDF, one-vs-rest logistic regression, and the
existing exact-alias rules. Evaluate with five out-of-fold splits separated by
`parent_asin`. Store the current fitted model in one minimal directory,
`models/hair-conditioners-aspect-extractor`, without checkpoint folders.

Treat the 0.45 decision threshold and all resulting metrics as development
choices tied to Gemini silver. Require a diverse 24-row human disagreement
audit before population-scale inference, and human gold before any formal
accuracy or acceptance claim.

## Reason

The exact-alias baseline is precise but misses paraphrases. The hybrid raises
overall silver recall from 0.4634 to 0.7079 and F1 from 0.5273 to 0.5909. On the
representative component, F1 rises from 0.3836 to 0.5641. Separating parent
products prevents direct product leakage between out-of-fold train and test
rows, while the small audit targets the remaining ambiguity efficiently.

## Consequence

The V2 artifact is the current aspect-extraction candidate, not a production-
accepted model. The assistant completed the 24-row disagreement review and
kept the 0.45 threshold, allowing diagnostic inference on all 118,804 niche
reviews. Production acceptance still requires independent human evidence.
