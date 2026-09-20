# Review Intelligence Platform --- Roadmap

## 01 --- Dataset Exploration

-   [x] Load Amazon review dataset
-   [x] Inspect file format, size and columns
-   [x] Inspect rating distribution
-   [x] Inspect review text
-   [x] Analyze missing values
-   [x] Analyze duplicates
-   [x] Analyze review statistics
-   [x] Initial visualizations

------------------------------------------------------------------------

## 02 --- Data Cleaning & Preprocessing

-   [x] Clean review text
-   [x] Remove invalid records
-   [x] Handle duplicates
-   [x] Validate ratings
-   [x] Normalize data types
-   [x] Create processed dataset
-   [x] Validate processed dataset
-   [x] Rematerialize all 8,986,368 canonical rows with physically non-nullable
    required Arrow fields
-   [x] Reject or null malformed, fractional, and overflowing optional values
    without crashing the streaming build
-   [x] Deduplicate otherwise identical anonymous reviews correctly
-   [x] Strictly verify manifest sizes, SHA-256 values, and record counts

The full 2026-08-18 rebuild retains the same logical source population: 9,084,957
filtered rows, 98,589 exact duplicates removed, and 8,986,368 canonical rows.
The category registry has 724 paths; the full catalog has 1,028,914 unique
`parent_asin` rows and matches every canonical review.

------------------------------------------------------------------------

## 03 --- Sentiment Baseline (UTC-stable refresh complete)

-   [x] Create sentiment labels
-   [x] Create a balanced 300k-review training sample
-   [x] Create versioned product-aware and temporal splits
-   [x] Prevent repeated-text leakage
-   [x] Add natural and balanced evaluation views
-   [x] Evaluate the first candidate `Conditioners` niche
-   [x] TF-IDF
-   [x] Logistic Regression
-   [x] Accuracy
-   [x] Weighted F1
-   [x] Macro F1
-   [x] Classification report
-   [x] Confusion matrix
-   [x] Persist baseline results
-   [x] Train `beauty_tfidf_rating_sentiment_v2` on the corrected 300k split
-   [x] Train `beauty_tfidf_rating_sentiment_1m_v3` on the corrected 999,999-row
    split for direct Transformer comparison
-   [x] Replace row bootstrap with percentile bootstrap over whole
    `parent_asin` clusters

Current 1M TF-IDF results from
`reports/model_evaluation/beauty_tfidf_rating_sentiment_1m_v3.json`:

``` text
Unseen Beauty parent products, natural:
  accuracy 0.84731; macro F1 0.72757; cluster CI [0.72391, 0.73087]
Unseen Beauty parent products, balanced:
  accuracy 0.78644; macro F1 0.78717; cluster CI [0.78248, 0.79201]
Later 2023 reviews, natural:
  accuracy 0.84722; macro F1 0.72717; cluster CI [0.72398, 0.73029]
Later 2023 reviews, balanced:
  accuracy 0.78382; macro F1 0.78435; cluster CI [0.77967, 0.78947]
Held-out Conditioner parent products, natural:
  accuracy 0.86288; macro F1 0.72051; cluster CI [0.70915, 0.73246]
Held-out Conditioner parent products, balanced:
  accuracy 0.77778; macro F1 0.77808; cluster CI [0.76053, 0.79374]
```

The target is a three-class rating-derived weak label. Natural and balanced
results answer different questions and must not be collapsed into one score.
The Conditioner test holds out 658 defined `parent_asin` values; it does not
exclude the entire niche. On exact shared tests, the 1M baseline improves macro
F1 over the refreshed 300k reference by 0.00957 (products), 0.00999 (temporal),
and 0.01100 (Conditioner parents).

------------------------------------------------------------------------

## 04 --- Transformer Sentiment (UTC-stable refresh)

-   [x] Prepare balanced 1M-review dataset
-   [x] Train / validation / test split
-   [x] Tokenization
-   [x] Preserve the legacy Beauty V1 run as historical evidence
-   [x] Rebuild the 1M split with an explicit UTC cutoff
-   [x] Retrain the TF-IDF comparison model on corrected V3 membership
-   [x] Finish the newly versioned DistilBERT V2 clean run on corrected V3
    membership
-   [x] Publish classification reports, confusion matrices, and direct TF-IDF
    comparisons from the V2 report
-   [x] Recompute every interval with percentile bootstrap over whole
    `parent_asin` clusters
-   [ ] Keep V2 at `candidate_pending_human_review` until its own blind queue is
    labeled; do not inherit the legacy acceptance decision
-   [x] Complete a limited stratified V2 diagnostic on 20/156 rows; use it to
    continue downstream work, not as formal model acceptance or a
    corpus-wide accuracy estimate

Historical model metrics (superseded; not current acceptance evidence):

``` text
DistilBERT
Beauty
999,999 balanced training reviews

Held-out Beauty parent products, balanced macro F1:    0.8187
Later 2023 reviews, balanced macro F1:                  0.8172
Held-out Conditioner parent products, balanced macro F1: 0.8146
```

Historical classical-reference metrics:

``` text
TF-IDF + Logistic Regression
Beauty
999,999 identical balanced training reviews

Held-out Beauty parent products, balanced macro F1:    0.7900
Later 2023 reviews, balanced macro F1:                  0.7873
Held-out Conditioner parent products, balanced macro F1: 0.7798
```

The corrected UTC-stable 1M V3 manifest and refreshed TF-IDF comparison are
complete. The Conditioner evaluation holds out 658 defined parent products;
other Conditioner products remain in training, so this is not a whole-niche
zero-shot test.

The completed 158-row diagnostic remains attached only to the legacy V1
predictions. It was not relabeled for V2 and cannot promote the refreshed
model. Current automated metrics and the V2 promotion decision are read only
from the newly versioned V2 evaluation and error-analysis reports.

------------------------------------------------------------------------

## 05 --- Error Analysis (limited human diagnostic complete)

-   [x] Load the newly versioned V2 predictions
-   [x] Find possible disagreement between review text and star rating
-   [x] Analyze mixed opinions and Neutral errors
-   [x] Measure probability calibration before using the word confidence
-   [x] Analyze errors by product, niche, date, and review length
-   [x] Compare shortened and complete review texts
-   [x] Regenerate the blind manual-review queue from V2 predictions
-   [x] Label and summarize a stratified 20-row subset of the new queue
-   [ ] Complete the remaining 136 rows only if formal model acceptance is
    required
-   [ ] Make a V2 promotion decision from V2 evidence only

The checkmarks above for automatic analyses describe implemented checks. Their
current outputs must be regenerated from V2 predictions before they are treated
as current evidence. The historical 158 labels remain useful for auditing the
legacy run but are never copied into the refreshed queue.

Seller-facing interpretation:

``` text
Text sentiment remains: negative / neutral / positive
Seller attention is:    1–3 stars needs_attention / 4–5 stars satisfied
Policy version:         amazon_rating_attention_v1
```

This business signal is calculated from ratings and does not require a new ML
model. Aspect analysis will explain what needs attention.

------------------------------------------------------------------------

## 06 --- Amazon Data Contracts (rewritten and verified)

-   [x] Define canonical Amazon review and product schemas
-   [x] Preserve `asin` and `parent_asin`
-   [x] Define deterministic `review_id`
-   [x] Validate ratings, timestamps, text, and identifiers
-   [x] Register schema and dataset versions
-   [x] Produce reconciliation and data-quality reports

The current product is Amazon-focused. Supporting arbitrary unrelated review
datasets through a universal schema is not a V1 goal.

------------------------------------------------------------------------

## 07 --- Amazon Category Registry (rewritten and verified)

-   [x] Build the full-source Amazon category registry
-   [x] Preserve ordered category paths and stable path IDs
-   [x] Measure missing and inconsistent metadata
-   [x] Define metadata-first niche-selection rules
-   [x] Inspect conditioner-related path candidates
-   [x] Approve Hair Conditioners V1 as the first Beauty MVP niche

Generic LLM domain detection was removed. Amazon metadata and versioned
category-path IDs determine the analytical scope.

------------------------------------------------------------------------

## 08 --- Product Catalog & Review Selection (niche scope verified)

-   [x] Build the complete versioned product catalog
-   [x] Reconcile all metadata rows
-   [x] Join canonical reviews with 100% product coverage
-   [x] Create normalized product text for catalog search
-   [x] Define versioned niche and seller/competitor workspace contracts
-   [x] Inspect the candidate standard-conditioner population
-   [ ] Resolve user ASIN and parent-ASIN queries
-   [x] Approve the exact category-path filter for Hair Conditioners V1
-   [ ] Evaluate product/niche matching quality
-   [ ] Save a real seller-versus-competitor workspace

Target:

``` text
Seller Workspace
      ↓
Versioned Product Catalog
      ↓
Own Products + Competitors + Approved Niche
      ↓
Full Relevant Review Population
```

------------------------------------------------------------------------

# 09 --- Aspect Discovery (complete for Hair Conditioners V1)

This is the active unfinished analytical stage.

Goal:

Discover recurring aspects/topics from a representative sample without
maintaining a universal hard-coded aspect list.

Example:

``` text
Beauty:
moisturizing
softness
smell
price
packaging
```

Tasks:

-   [x] Build representative sampling strategy
-   [x] Replace the biased product-capped design with review-level stratified SRS
-   [x] Prepare a lineage-checked 1,500-review V3 sample covering 904 products;
    verify inverse-inclusion weights sum to 114,025
-   [x] Prepare 15 request batches of 100 and verify local client readiness
-   [x] Send all corrected V3 sampled reviews to Gemini
-   [x] Extract V3 candidate aspects
-   [x] Measure corrected weighted aspect support
-   [x] Create and human-review taxonomy V2
-   [x] Preserve V1 taxonomy labels while marking its population shares invalid

------------------------------------------------------------------------

# 10 --- Aspect Normalization

Goal:

Map different expressions to one canonical aspect.

Example:

``` text
smell
scent
fragrance
nice smell
strong fragrance
```

→

``` text
fragrance
```

Tasks:

-   [x] Define canonical aspect schema
-   [ ] Generate embeddings for aspect labels
-   [ ] Cluster similar aspect expressions
-   [x] Preserve historical V2 Gemini normalization as qualitative Taxonomy V1
    provenance
-   [x] Validate historical assignments, exclusions, aliases, and lineage while
    marking weighted support/share invalid
-   [x] Normalize the 89 corrected V3 candidates into 39 proposed aspects
-   [x] Human-review Taxonomy V2 merges and exclusions
-   [x] Materialize the approved 39-aspect quantitative Taxonomy V2
-   [x] Store a quantitatively valid current taxonomy

------------------------------------------------------------------------

# 11 --- Aspect Extraction

Goal:

Identify all relevant aspects in each review.

Example:

``` text
"Great conditioner but the smell is awful."

→

effectiveness
fragrance
```

Tasks:

-   [x] Select exact-alias + character-TF-IDF hybrid V2 candidate
-   [x] Build `hair_conditioners_extraction_eval_v2`: 160 probability-sampled
    representative rows (weights sum to 112,525) and 80 unweighted targeted rows
-   [x] Define blind human annotation and exact-evidence validation
-   [x] Complete privacy-minimized Gemini silver annotation for all 240 rows
-   [ ] Complete human gold annotation only if formal acceptance is required
-   [x] Implement an exact-alias multi-aspect extraction baseline with evidence
    offsets
-   [x] Run the baseline on Taxonomy-V2-aligned `extraction_eval_v3`
-   [x] Evaluate diagnostic precision / recall / F1 against Gemini silver
-   [x] Build five product-separated out-of-fold predictions
-   [x] Prepare a diverse 24-row disagreement audit
-   [x] Complete a non-independent assistant review of the 24 disagreements
-   [x] Optimize bounded local batch inference
-   [x] Persist 236,316 extracted aspect observations for 118,804 niche reviews

------------------------------------------------------------------------

# 12 --- Aspect Sentiment

Goal:

Determine sentiment for each extracted aspect.

Example:

``` text
softness → positive
smell → negative
price → negative
```

Tasks:

-   [x] Define the initial aspect sentiment prediction schema
-   [x] Select the local TF-IDF evidence-context baseline architecture
-   [x] Build Gemini silver training labels
-   [x] Train the Hair Conditioners hybrid aspect extractor
-   [x] Evaluate agreement against Gemini silver on overlapping aspect pairs
-   [x] Persist local TF-IDF evidence-context baseline predictions

Target analytical table:

``` text
review_id
product_id
aspect
aspect_sentiment
confidence
evidence
```

------------------------------------------------------------------------

# 13 --- Large-Scale Analytics Engine

This is the core business-intelligence layer.

Tasks:

-   [ ] Overall sentiment statistics
-   [x] Aspect frequency
-   [x] Initial positive aspect statistics
-   [x] Initial negative aspect statistics
-   [x] Common complaints for the first demo workspace
-   [x] Customer strengths for the first demo workspace
-   [x] Product comparison for the first demo workspace
-   [x] Initial rating-based seller-attention trends
-   [ ] Sentiment trends
-   [x] Initial month-over-month aspect trends
-   [ ] Emerging complaints
-   [ ] Representative reviews

Example:

``` text
Aspect          Mentions    Positive    Negative
-------------------------------------------------
softness        84,231      91%         9%
moisturizing    71,432      89%         11%
smell           62,341      59%         41%
price           48,221      32%         68%
packaging       31,921      44%         56%
```

------------------------------------------------------------------------

# 14 --- Vertex AI / Gemini

Connect the platform to Google Vertex AI.

Tasks:

-   [x] Configure a Free Tier Gemini API discovery backend
-   [ ] Configure GCP project
-   [ ] Enable Vertex AI API
-   [ ] Configure authentication
-   [x] Install required SDK
-   [x] Make first Gemini request
-   [x] Create reusable LLM client
-   [x] Create system prompts
-   [x] Structured output
-   [x] Pydantic validation
-   [x] Error handling and resumable batches
-   [x] Token/cost tracking contract

Gemini responsibilities:

``` text
Aspect Discovery
Aspect Normalization
Analytical Synthesis
Explanation
Natural-language Answers
```

Gemini should not be the primary processor of millions of individual
reviews.

------------------------------------------------------------------------

# 15 --- Embeddings

-   [x] Select local TF-IDF/SVD baseline
-   [x] Generate embeddings
-   [x] Build embedding pipeline
-   [x] Store embeddings
-   [x] Track model/version and hashes
-   [ ] Measure cost
-   [ ] Validate embedding quality

------------------------------------------------------------------------

# 16 --- Vector Search

-   [x] Select NumPy-backed local vector storage
-   [x] Create vector index
-   [x] Index all 118,804 niche reviews
-   [x] Latent semantic similarity search
-   [x] Top-K retrieval (lexical stage)
-   [x] Metadata filtering (lexical stage)
-   [x] Product filtering
-   [ ] Category filtering
-   [x] Aspect filtering
-   [x] Sentiment filtering
-   [ ] Date filtering

------------------------------------------------------------------------

# 17 --- RAG

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

Tasks:

-   [x] Retrieval (lexical stage)
-   [x] Context construction
-   [x] Prompt template
-   [x] Gemini generation
-   [x] Review/source references
-   [x] Citation hallucination checks
-   [ ] RAG evaluation

------------------------------------------------------------------------

# 18 --- AI Analyst

Combine:

``` text
Statistics
+
Sentiment
+
Aspects
+
RAG
+
Gemini
```

Tasks:

-   [x] Build analyst prompt
-   [x] Connect aggregate analytics
-   [x] Connect sentiment
-   [x] Connect aspects
-   [x] Connect RAG
-   [x] Generate grounded answers
-   [x] Add source references
-   [ ] Evaluate answers

Example questions:

``` text
What do customers like most?

What are the biggest complaints?

Why are customers unhappy?

Which product performs better?

How has sentiment changed?

What are emerging problems?
```

------------------------------------------------------------------------

# 19 --- Agent / Function Calling

Tools:

``` text
search_reviews()
get_statistics()
get_sentiment()
get_aspects()
get_trends()
compare_products()
```

Tasks:

-   [ ] Define tools
-   [ ] Define function schemas
-   [ ] Implement tool calling
-   [ ] Validate tool results
-   [ ] Implement multi-step tool usage
-   [ ] Error handling

------------------------------------------------------------------------

# 20 --- FastAPI

Endpoints:

``` text
POST /upload
POST /analyze
POST /ask

GET /dataset/{id}
GET /analytics/{id}
GET /products/{id}
```

Tasks:

-   [x] Create FastAPI application
-   [x] Request models
-   [x] Response models
-   [ ] Dataset upload
-   [x] Analysis endpoint
-   [x] Ask/RAG endpoint
-   [x] Analytics endpoints
-   [x] Error handling
-   [x] API documentation

------------------------------------------------------------------------

# 21 --- Seller Web Dashboard

Main flow:

``` text
Upload Dataset
      ↓
Detect Schema
      ↓
Select Category
      ↓
Analyze Dataset
      ↓
Dashboard
      ↓
Ask AI
```

Dashboard:

-   [x] Dataset overview
-   [x] Ratings
-   [ ] Sentiment
-   [x] Aspects
-   [x] Trends
-   [x] Pain points
-   [x] Product comparison
-   [ ] AI Analyst chat

------------------------------------------------------------------------

# 22 --- Testing

-   [x] Unit tests for data loading
-   [x] Schema validation tests
-   [x] Preprocessing tests
-   [x] Sentiment tests
-   [x] Aspect extraction tests
-   [x] Aspect sentiment tests
-   [ ] Embedding tests
-   [x] Retrieval tests
-   [ ] RAG evaluation
-   [x] API tests
-   [ ] Agent/tool tests
-   [ ] UI tests
-   [ ] Edge cases
-   [ ] Large-scale processing tests

------------------------------------------------------------------------

# 23 --- Docker / GCP

-   [ ] Dockerfile
-   [ ] Containerize backend
-   [ ] Containerize application
-   [ ] Environment variables
-   [ ] Secrets management
-   [ ] GCP deployment
-   [ ] Connect production application to Vertex AI
-   [ ] Logging
-   [ ] Monitoring

------------------------------------------------------------------------

# 24 --- Model & Dataset Versioning

Tasks:

-   [ ] Dataset versioning
-   [ ] Model versioning
-   [ ] Store training metadata
-   [ ] Store evaluation metrics
-   [ ] Link model versions to dataset versions
-   [ ] Create model registry structure

Example:

``` text
Beauty
  dataset: amazon_reviews_2023
  model: sentiment_v1

Electronics
  dataset: amazon_reviews_2023
  model: sentiment_v1
```

------------------------------------------------------------------------

# 25 --- Future Dataset Updates

When 2024/2025/2026 datasets become available:

``` text
Existing Model
      ↓
New Dataset
      ↓
Evaluation
      ↓
Drift Monitoring
```

Do not retrain automatically.

Retrain when:

``` text
performance degradation
OR
data drift
OR
vocabulary changes
OR
new product types
```

Then:

``` text
v1
 ↓
new data
 ↓
retraining
 ↓
v2
 ↓
evaluation
 ↓
model registry
```

------------------------------------------------------------------------

# 26 --- Category Generalization

Final test:

``` text
Beauty
```

and then:

``` text
Electronics
```

The same codebase should work with:

``` text
different dataset
different category configuration
different model artifacts
different aspect taxonomy
```

without rewriting the core pipeline.

Success criterion:

``` text
New Category
      ↓
Configuration
      ↓
Dataset
      ↓
Category-specific models
      ↓
Aspect taxonomy
      ↓
Processing
      ↓
Analytics
      ↓
RAG
      ↓
Gemini
      ↓
AI Analyst
```

------------------------------------------------------------------------

# 27 --- Next Milestone: Shampoo & Conditioner Expansion

Keep the completed Hair Conditioners pipeline intact and expand next to the
complete `Beauty & Personal Care > Hair Care > Shampoo & Conditioner` section:
40,402 catalog products, 21,651 reviewed parent products, and 358,925 reviews.

Initial analytical/model families:

``` text
Cleansing products     → Shampoos, 2-in-1, 3-in-1
Conditioning products  → Conditioners, Deep Conditioners
Dry shampoos           → separate family
Product sets           → Shampoo & Conditioner Sets
Generic unresolved     → quarantine until metadata resolves the product type
```

Rules:

- model families contain products evaluated by similar customer criteria;
- competitor niches contain plausible purchase substitutes;
- comparisons default to exact competitor niches;
- aspects have shared, family-specific, and optional niche-specific levels;
- broader data is never assumed to improve quality without fixed-set
  evaluation by family and leaf.

Tasks:

-   [x] Register `shampoo_and_conditioner_v1` and exact source paths.
-   [x] Materialize and reconcile the complete 358,925-review population.
-   [x] Assign versioned family and competitor-niche scope metadata.
-   [x] Build 300 representative plus 200 targeted evaluation rows with no
    overlap with discovery.
-   [x] Run the frozen Conditioner extractor as the expansion coverage
    baseline; accuracy remains pending independent reference labels.
-   [x] Scan all 358,925 reviews locally and build the first 5,000-review
    family/leaf-stratified discovery sample.
-   [x] Complete adaptive Gemini discovery and saturation analysis over the
    planned 4,000–6,000 range.
-   [x] Run discovery in waves and stop only after documented taxonomy
    saturation and adequate coverage of every family and leaf.
-   [x] Keep the independent 450–600-row evaluation set completely separate
    from taxonomy creation and tuning.
-   [x] Approve the expanded hierarchical taxonomy (46 aspects; 17 excluded
    noise/catalog/fulfillment labels).
-   [x] Train one shared extractor after independent silver evaluation showed
    similar family-level performance; keep family splitting as a future gate.
-   [x] Run full-section extraction, aspect sentiment, marts, exact-niche
    rankings, and local evidence indexing.
-   [x] Add section-wide search plus automatic family/niche selection to API
    and UI without exposing technical workspace selectors.
-   [x] Re-run the 20-question analyst benchmark with shampoo, dry-shampoo,
    combination-product, and set cases; all 20 generated answers validated.
-   [ ] Then complete Docker and the demonstration deployment.

The detailed grouping rationale, counts, gates, and provisional volume
thresholds are maintained in `PLAN.md`, Phase 10A.

------------------------------------------------------------------------

# Final Target

The project should demonstrate:

``` text
Large-scale NLP
+
Transformer ML
+
Aspect-Based Sentiment Analysis
+
Data Engineering
+
Analytics
+
Embeddings
+
Vector Search
+
RAG
+
Vertex AI / Gemini
+
Agentic Tool Calling
+
FastAPI
+
Streamlit
+
Docker
+
GCP
+
ML Model Lifecycle
```
