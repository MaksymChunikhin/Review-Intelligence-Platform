# Aspect Extraction Baselines

## Verified Snapshot — 2026-09-09

The approved `hair_conditioners_taxonomy_v2` is connected to a new evaluation
artifact, `hair_conditioners_extraction_eval_v3`. It contains 160 probability-
sampled representative reviews and 80 non-probability reviews targeted for
rare-aspect coverage. Neither component overlaps the 1,500-review discovery
sample.

The first extraction baseline performs case-insensitive exact matching of
approved taxonomy aliases. It supports multiple aspects per review and stores
the exact source phrase plus start/end character offsets. On the 240-row
evaluation artifact it produced 765 review-aspect assignments and 1,097 exact
evidence spans across all 39 aspects.

Coverage must be read by component:

- representative: 86/160 reviews (53.75%), 169 review-aspect assignments;
- targeted: 80/80 reviews (100%), 596 assignments.

The targeted component was selected using taxonomy aliases, so its 100%
coverage is expected and is not a quality or population statistic.

The aspect-sentiment baseline applies the saved
`beauty_tfidf_rating_sentiment_1m_v3` weak-label review classifier to the local
sentence context around each exact evidence span. It produced 82 negative, 234
neutral, and 449 positive predictions. These are diagnostic predictions, not
validated aspect-sentiment metrics.

## Gemini Silver Diagnostic

With explicit project-owner authorization, the 240 public review texts were
sent to Gemini without `user_id`, rating, product title, ASIN, or internal
`review_id`. Requests contained only an opaque `item_id` and `review_text` and
were processed in 12 resumable batches of 20. Every returned aspect is limited
to Taxonomy V2 and every accepted evidence phrase is restored to an exact
contiguous span of the source review. The accepted batch records account for
202,446 tokens; earlier rejected attempts made before re-batching are not
included in that figure.

The silver reference contains 1,010 review-aspect assignments across all 39
aspects. Against it, the exact-alias baseline has overall micro precision
0.6118, recall 0.4634, and F1 0.5273. On the 160 natural representative rows,
precision is 0.6923 and recall is 0.2653; population-weighted precision is
0.7564 and recall is 0.2334. The baseline is therefore conservative and misses
many paraphrases. Aspect-sentiment agreement on 462 overlapping three-class
pairs is 0.6494.

These are agreement scores against one Gemini silver pass, not human-ground-
truth accuracy. They are suitable for prioritizing extractor improvements and
a small disagreement audit, but not for final model acceptance.

## Hybrid Extractor V2

`hair_conditioners_aspect_extractor_v2` combines the exact-alias rules with a
character 3–5 gram TF-IDF one-vs-rest logistic model. Five out-of-fold splits
are separated by `parent_asin`; product overlap between folds is zero. The
conditioner-only artifact was the production-shaped prototype. It was retired
after the project expanded to the full Shampoo & Conditioner section; the
active replacement is `models/shampoo-conditioner-aspect-extractor`.

Against Gemini silver, its out-of-fold overall precision is 0.5071, recall is
0.7079, and F1 is 0.5909, versus 0.6118 / 0.4634 / 0.5273 for exact aliases.
On the natural representative component, F1 rises from 0.3836 to 0.5641; the
population-weighted estimate rises from 0.3567 to 0.5950. Aspect-sentiment
agreement on overlapping three-class pairs rises from 0.6494 to 0.6624.

The selected 0.45 threshold was chosen during silver-label experimentation, so
these scores are useful model-development diagnostics but are not an unbiased
final test. A 24-row assistant audit checked eight likely false positives,
eight false negatives, and eight sentiment disagreements. It supported keeping
the 0.45 threshold, but it is explicitly non-independent and is not human gold.

## Full Niche Diagnostic

The commands below document the historical conditioner-only experiment. Its
reports are retained in `reports/archive/legacy_reports_2026-09-13.tar.gz`; the
retired model directory is intentionally not kept in the active tree.

The approved Hair Conditioners V1 niche contains 118,804 reviews across 7,382
reviewed parent products. Local batch inference matched 104,170 reviews and
produced 236,316 review-aspect observations across all 39 aspects. No external
API was used. The first offline statistics table contains frequency, rating-
based seller attention, weak-label aspect sentiment, model evidence source, and
representative positive/negative evidence.

The most frequent candidate aspects are scent (49,659 reviews; 41.8% of the
niche), overall performance (39,783; 33.5%), and softness (24,269; 20.4%).
Packaging has the strongest high-volume seller-attention signal: 62.2% of its
4,851 matched reviews have a 1–3 star rating. Frequency and seller attention
are useful diagnostic aggregates; aspect-sentiment shares remain model outputs.

Two reconciled seller-analytics marts are now available:

- `product_aspect_mart_v1`: 54,590 product-aspect rows covering 7,040 products;
- `month_aspect_mart_v1`: 1,287 rows covering 33 months and all 39 aspects.

Both marts sum exactly to the 236,316 source review-aspect observations. The
product mart includes frequency, complaint/strength ranks, rating-based seller
attention, and top evidence references. The monthly mart includes normalized
aspect share and month-over-month changes. September 2023 contains only 47
source reviews and must not be compared as a complete month.

## Reproduction

```bash
python -m src.analytics.aspect_extraction \
  reports/aspects/hair_conditioners_taxonomy_v2.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/sample.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/alias_baseline_predictions.parquet \
  --extraction-version hair_conditioners_alias_extractor_v1 \
  --report-path reports/aspects/hair_conditioners_alias_extractor_v1_eval_v3.json

python -m src.ml.aspect_sentiment \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/alias_baseline_predictions.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/sample.parquet \
  models/sentiment-baseline \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/alias_baseline_aspect_sentiment.parquet \
  --aspect-sentiment-version hair_conditioners_tfidf_aspect_sentiment_v1 \
  --report-path reports/aspects/hair_conditioners_tfidf_aspect_sentiment_v1_eval_v3.json

python -m src.analytics.aspect_silver_evaluation \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/sample.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/alias_baseline_predictions.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/alias_baseline_aspect_sentiment.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/silver/silver_annotations.parquet \
  reports/aspects/hair_conditioners_alias_vs_gemini_silver_v1.json

python -m src.ml.aspect_multilabel \
  config/aspects/hair_conditioners_aspect_extractor_v2.json \
  reports/aspects/hair_conditioners_taxonomy_v2.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/sample.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/silver/silver_annotations.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/alias_baseline_predictions.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_evaluation/hair_conditioners_extraction_eval_v3/hybrid_v2_oof_predictions.parquet \
  models/hair-conditioners-aspect-extractor \
  --report-path reports/aspects/hair_conditioners_aspect_extractor_v2_training.json

python -m src.ml.aspect_inference \
  config/aspects/hair_conditioners_aspect_extractor_v2.json \
  reports/aspects/hair_conditioners_taxonomy_v2.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/niches/hair_conditioners_v1/reviews.parquet \
  models/hair-conditioners-aspect-extractor \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/niches/hair_conditioners_v1/aspect_predictions_v2.parquet \
  --report-path reports/aspects/hair_conditioners_aspect_extractor_v2_population.json
```
