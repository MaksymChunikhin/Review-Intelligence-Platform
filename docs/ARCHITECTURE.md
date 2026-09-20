# Review Intelligence Platform --- Architecture

## 1. Purpose

The platform transforms large collections of customer reviews into
structured business intelligence.

The system is designed for Amazon review datasets grouped by product
category.

Examples:

``` text
Beauty
Electronics
Automotive
Home & Kitchen
Video Games
Books
...
```

The architecture separates:

1.  data ingestion;
2.  product selection;
3.  ML processing;
4.  aspect intelligence;
5.  deterministic analytics;
6.  semantic retrieval;
7.  Gemini reasoning and explanation;
8.  user-facing applications.

------------------------------------------------------------------------

# 2. Core Design Principles

## Principle 1 --- Reviews are the raw analytical source

The platform should process large volumes of reviews rather than relying
on a small manually selected subset.

## Principle 2 --- Product selection happens before review analysis

The product catalog is used to identify the relevant product group.

Example:

``` text
User:
"What do customers think about hair conditioners?"

        ↓

Product Catalog

        ↓

Matching products

        ↓

product_id values

        ↓

All reviews for those products
```

The system should not infer the product group only from review text.

## Principle 3 --- Sentiment is an analytical signal, not the final product

Overall sentiment provides useful information but does not explain why
customers are satisfied or dissatisfied.

Text sentiment is separate from the seller-oriented attention signal. The
first attention policy maps 1–3 stars to `needs_attention` and 4–5 stars to
`satisfied`. A three-star review may remain Neutral in text while still
requiring the seller's attention.

## Principle 4 --- Aspect-Based Analysis is central

The platform must identify what customers discuss and what they think
about each aspect.

## Principle 5 --- Python calculates facts

Deterministic statistics should be calculated by Python/analytics code.

Gemini should not be responsible for basic numerical aggregation.

## Principle 6 --- Gemini explains and synthesizes

Gemini receives structured statistics and relevant evidence and converts
them into human-readable business insights.

## Principle 7 --- RAG provides evidence

Vector retrieval is used when the user needs representative reviews,
explanations, or supporting evidence.

## Principle 8 --- Shared pipeline, category-specific models

The processing architecture is reusable, while ML models may be trained
separately for each product category.

------------------------------------------------------------------------

# 3. Offline vs Online Architecture

## Offline Pipeline

Heavy computation is performed before user interaction.

``` text
Dataset
   ↓
Cleaning
   ↓
Product Catalog
   ↓
Model Inference
   ↓
Aspect Processing
   ↓
Embeddings
   ↓
Analytics
   ↓
Persisted Results
```

The objective is to avoid running expensive ML inference over millions
of reviews for every user question.

## Online Pipeline

User interaction works mainly with precomputed analytical data.

``` text
User Question
      ↓
Product Selection
      ↓
Analytics Query
      ↓
Statistics
      ↓
Optional RAG
      ↓
Gemini
      ↓
Answer
```

------------------------------------------------------------------------

# 4. Data Layer

## Amazon Dataset

Baseline:

``` text
Amazon Reviews 2023
```

Primary working window:

``` text
2021-01-01 → 2023-09-13
```

The snapshot is kept fixed for reproducibility.

Older records may be retained when useful for model training or category
coverage.

------------------------------------------------------------------------

# 5. Product Catalog

The product catalog contains relatively stable product metadata.

The catalog is used to map natural-language user requests to actual
products.

Example:

``` text
"hair conditioners"
        ↓
product catalog
        ↓
matching products
        ↓
product_id
        ↓
reviews
```

This prevents the system from relying exclusively on keywords found in
review text.

------------------------------------------------------------------------

# 6. Universal Review Representation

Core fields:

``` text
review_id
product_id
product_name
rating
review_text
review_date
```

Optional fields:

``` text
title
verified_purchase
helpful_votes
user_id
category
```

------------------------------------------------------------------------

# 7. Category-Specific ML Architecture

The platform uses a shared training/inference pipeline.

Each category can have its own model artifacts:

``` text
models/
├── beauty/
│   ├── sentiment/
│   └── aspects/
│
├── electronics/
│   ├── sentiment/
│   └── aspects/
│
├── automotive/
│   ├── sentiment/
│   └── aspects/
│
└── ...
```

The code is shared.

Configuration and model artifacts determine which category is processed.

------------------------------------------------------------------------

# 8. Overall Sentiment

Current training and evaluation are anchored to the UTC-stable
`beauty_rating_sentiment_split_v2` and
`beauty_rating_sentiment_split_1m_v3` manifests. The refreshed 1M TF-IDF
baseline is `beauty_tfidf_rating_sentiment_1m_v3`; on balanced views its macro
F1 is 0.78717 for held-out Beauty parent products, 0.78435 for later 2023
reviews, and 0.77808 for held-out Conditioner parent products. Every interval
is a percentile bootstrap interval over whole `parent_asin` clusters.

The refreshed Transformer is
`beauty_distilbert_rating_sentiment_1m_v2`. Its balanced macro F1 is 0.81815
for held-out Beauty parent products, 0.81687 for later 2023 reviews, and 0.81461
for held-out Conditioner parent products, with 95% product-cluster intervals
[0.81318, 0.82296], [0.81186, 0.82180], and [0.79729, 0.82858]. It has an
independent promotion state: automated metrics do not transfer the legacy
human decision, so until a new blind review queue is labeled the only
permissible state is `candidate_pending_human_review`.

The frozen Beauty V1 DistilBERT artifact remains historical. Its temporal
boundary depended on the DuckDB session timezone and its confidence intervals
resampled rows, so its metrics cannot support the corrected chain.

Historical Beauty V1 metrics (not corrected-chain acceptance evidence):

``` text
Training dataset: 999,999 balanced reviews
Held-out Beauty parent products, balanced macro F1: 0.8187
Later 2023 reviews, balanced macro F1:             0.8172
Held-out Conditioner parent products, macro F1:    0.8146
```

The Conditioner score is a parent-product holdout, not a holdout of the entire
Conditioners niche; other Conditioner products remain in training.

Output:

``` text
review_id
overall_sentiment
model_score
seller_attention
seller_attention_policy_version
```

Example:

``` text
review_id = 001
overall_sentiment = positive
model_score = 0.94
seller_attention = satisfied
seller_attention_policy_version = amazon_rating_attention_v1
```

`seller_attention` is not a sentiment-model answer. It is calculated from the
Amazon rating. This distinction finds commercially important three-star
reviews without falsely claiming that their language is Negative.

`model_score` is retained for ranking and error analysis but is not displayed
as guaranteed human confidence. The completed 158-row diagnostic review belongs
to the legacy V1 predictions. It was not relabeled for the UTC-stable refresh,
cannot support a refreshed model decision, and—because it is enriched for
model-rating disagreements—cannot estimate unbiased corpus-wide accuracy.

This signal supports:

-   overall sentiment distribution;
-   product comparison;
-   sentiment trends;
-   negative-review filtering;
-   quality monitoring.

The Beauty model should not automatically be deployed as the Electronics
model. Electronics should have its own evaluation and, where
appropriate, its own fine-tuned model.

------------------------------------------------------------------------

# 9. Aspect Intelligence

This is the core analytical layer.

For a review such as:

``` text
"The conditioner makes my hair very soft,
but the smell is terrible and the bottle leaks."
```

the target representation is:

``` text
softness → positive
smell → negative
packaging → negative
```

The pipeline consists of:

``` text
Aspect Discovery
       ↓
Aspect Normalization
       ↓
Aspect Extraction
       ↓
Aspect Sentiment
       ↓
Aggregation
```

------------------------------------------------------------------------

# 10. Aspect Discovery

A representative sample of reviews is used to discover recurring topics.

The current corrected Hair Conditioners materialization is
`aspect_discovery_sample_v2` / `hair_conditioners_discovery_v3`. It selects
1,500 reviews by stratified review-level SRS without a product cap, covers 904
parent products, and has inverse-inclusion weights that reconcile to all
114,025 eligible reviews. The 15 local batches are prepared, but no V3 Gemini
responses have been saved; extraction, normalization, and a new human Taxonomy
V2 review remain external gates.

Taxonomy V1 is retained only as a human-reviewed qualitative dictionary of 33
labels and aliases. Its legacy weighted support/share fields cannot be used for
population inference. Extraction evaluation V2 therefore keeps 160
probability-sampled representative rows (weights sum to the 112,525-review
holdout population) separate from 80 unweighted targeted diagnostics. All 240
gold labels are pending.

Gemini can be used during the discovery phase.

Example:

``` text
Sample Reviews
      ↓
Gemini
      ↓
Candidate aspects
```

For Hair Care, candidates might include:

``` text
moisturizing
softness
smell
texture
price
packaging
ingredients
frizz control
hair repair
```

The resulting taxonomy is stored as structured configuration/data.

The goal is not to hard-code a universal list of aspects for every
category.

------------------------------------------------------------------------

# 11. Aspect Normalization

Different expressions should map to the same normalized aspect.

Example:

``` text
"smells great"
"nice scent"
"pleasant fragrance"
```

should map to:

``` text
fragrance
```

Similarly:

``` text
"battery lasts all day"
"battery life is poor"
"battery drains quickly"
```

should map to:

``` text
battery_life
```

Embeddings and/or Gemini can support normalization and taxonomy
maintenance.

------------------------------------------------------------------------

# 12. Aspect Extraction

The production pipeline must identify which aspects occur in each
review.

Example:

``` text
Review
  ↓
[softness, smell, packaging]
```

The extraction layer should support multi-aspect reviews.

One review may contain several aspects with different sentiments.

------------------------------------------------------------------------

# 13. Aspect Sentiment

Overall sentiment is insufficient for business intelligence.

Example:

``` text
Review:
"Great conditioner, but the smell is awful."
```

Overall:

``` text
positive / mixed
```

Aspect-level:

``` text
effectiveness → positive
fragrance → negative
```

This allows the analytics engine to answer:

``` text
What do customers like?

What do customers dislike?

Which aspects generate the most complaints?
```

------------------------------------------------------------------------

# 14. Analytical Data Model

A useful analytical representation is:

``` text
review_id
product_id
aspect
aspect_sentiment
confidence
evidence
```

Example:

``` text
001 | A123 | softness | positive | 0.91 | "hair very soft"
001 | A123 | smell    | negative | 0.96 | "smell is terrible"
002 | A123 | price    | negative | 0.89 | "too expensive"
```

This structure enables aggregation over hundreds of thousands or
millions of reviews.

------------------------------------------------------------------------

# 15. Analytics Engine

Python performs deterministic calculations.

Examples:

``` python
get_statistics()
get_sentiment_distribution()
get_aspect_statistics()
get_rating_trends()
find_common_complaints()
compare_products()
get_aspect_trends()
```

Example output:

``` text
Aspect          Mentions    Positive    Negative
-------------------------------------------------
softness        84,231      91%         9%
moisturizing    71,432      89%         11%
smell           62,341      59%         41%
price           48,221      32%         68%
packaging       31,921      44%         56%
```

These numbers are facts generated by the analytics engine.

------------------------------------------------------------------------

# 16. Business Intelligence

The analytics layer should produce:

-   overall sentiment;
-   seller-attention volume and rate;
-   aspect frequency;
-   positive aspects;
-   negative aspects;
-   customer pain points;
-   customer strengths;
-   product comparisons;
-   rating trends;
-   sentiment trends;
-   aspect trends;
-   emerging complaints;
-   representative evidence.

The goal is to transform millions of raw reviews into actionable
information.

------------------------------------------------------------------------

# 17. Embeddings

Embeddings provide semantic representations of reviews and user
questions.

``` text
Review
   ↓
Embedding
   ↓
Vector
```

Embeddings support:

-   semantic search;
-   similar review retrieval;
-   RAG;
-   evidence retrieval;
-   clustering;
-   aspect normalization.

------------------------------------------------------------------------

# 18. Vector Search

Vector search retrieves reviews relevant to a question.

Example:

``` text
Question:
"Why do customers complain about battery life?"

        ↓

Query embedding

        ↓

Vector search

        ↓

Relevant reviews
```

Metadata filters should support:

``` text
product_id
category
aspect
sentiment
date
```

------------------------------------------------------------------------

# 19. RAG

RAG is used to retrieve evidence from real reviews.

``` text
User Question
      ↓
Query Embedding
      ↓
Vector Search
      ↓
Relevant Reviews
      ↓
Context
      ↓
Gemini
      ↓
Grounded Answer
```

RAG is not a replacement for the analytics engine.

Analytics answers:

``` text
What is happening?
```

RAG helps answer:

``` text
Why?
Show me evidence.
What do customers actually say?
```

------------------------------------------------------------------------

# 20. Vertex AI / Gemini

Gemini is the reasoning and natural-language layer.

Gemini responsibilities:

-   aspect discovery;
-   aspect normalization;
-   analytical synthesis;
-   explanation of statistics;
-   answering natural-language questions;
-   reasoning over structured tool results;
-   summarizing retrieved evidence.

Gemini should not be used to individually classify every review in the
production pipeline unless there is a specific reason.

The preferred architecture is:

``` text
Millions of reviews
       ↓
ML + deterministic analytics
       ↓
structured facts
       ↓
Gemini
       ↓
business explanation
```

------------------------------------------------------------------------

# 21. AI Analyst

The AI Analyst combines:

``` text
Statistics
+
Sentiment
+
Aspect Analysis
+
RAG
+
Gemini
```

Example questions:

``` text
What do customers like most?

What are the biggest complaints?

Why are customers unhappy?

Which product performs better?

How has sentiment changed over time?

What are the emerging problems?
```

------------------------------------------------------------------------

# 22. Model Versioning

Models are versioned by category and release.

Example:

``` text
models/
├── beauty/
│   ├── v1/
│   └── v2/
├── electronics/
│   └── v1/
└── automotive/
    └── v1/
```

Each model version stores metadata such as:

``` text
model architecture
category
dataset version
training date
training window
metrics
configuration
```

------------------------------------------------------------------------

# 23. Dataset Versioning

Example:

``` text
amazon_reviews_2023
    date_range: 2021-01-01 → 2023-09-13
```

Future:

``` text
amazon_reviews_2024
amazon_reviews_2025
amazon_reviews_2026
```

The application should be able to identify which model was trained on
which dataset version.

------------------------------------------------------------------------

# 24. Future Data Updates

When a new dataset appears:

``` text
Existing Model
      ↓
New Dataset
      ↓
Evaluation
```

Retraining is not automatic.

Retraining is considered when:

``` text
performance degradation
OR
data drift
OR
vocabulary changes
OR
new product types
```

If retraining is required:

``` text
Old Model v1
      ↓
New Data
      ↓
Retraining
      ↓
Model v2
      ↓
Evaluation
      ↓
Model Registry
```

------------------------------------------------------------------------

# 25. Category Change Workflow

Changing the product category should not require rewriting the core
pipeline.

Example:

``` text
Beauty
   ↓
beauty.yaml
   ↓
Beauty dataset
   ↓
Beauty models
   ↓
Beauty analytics
```

Change to:

``` text
Electronics
   ↓
electronics.yaml
   ↓
Electronics dataset
   ↓
Electronics models
   ↓
Electronics analytics
```

The pipeline remains the same.

------------------------------------------------------------------------

# 26. Offline / Online Separation

## Offline

``` text
Dataset
 ↓
Preprocessing
 ↓
Model inference
 ↓
Aspect extraction
 ↓
Aspect sentiment
 ↓
Embeddings
 ↓
Analytics
 ↓
Persist
```

## Online

``` text
User
 ↓
Question understanding
 ↓
Product selection
 ↓
Analytics query
 ↓
Optional RAG
 ↓
Gemini
 ↓
Answer
```

This separation is important for performance and cost.

------------------------------------------------------------------------

# 27. Final Target Architecture

``` text
                         USER
                           │
                           ▼
                    Gemini / Agent
                           │
                    Query Understanding
                           │
                           ▼
                    Product Catalog
                           │
                           ▼
                    Product Selection
                           │
                           ▼
                      ALL REVIEWS
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
     Overall Sentiment              Aspect Pipeline
   Versioned category model              │
            │                    ┌────────┴────────┐
            │                    ▼                 ▼
            │              Aspect Extraction  Aspect Sentiment
            │                    │                 │
            └────────────────────┴────────┬────────┘
                                          ▼
                                   Analytics Engine
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                         Statistics                 RAG
                              │                       │
                              └───────────┬───────────┘
                                          ▼
                                        Gemini
                                          │
                                          ▼
                                    AI Analyst
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                           FastAPI                Streamlit
```
