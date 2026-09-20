# Hair Conditioners Aspect Discovery

## Verified Snapshot — 2026-09-09

The current reproducible path is `hair_conditioners_discovery_v3`, built on
`aspect_discovery_sample_v2`. All external discovery batches and deterministic
candidate aggregation, semantic normalization, and project-owner review are
complete. The resulting 39-aspect Taxonomy V2 is approved.

| Stage | Artifact state |
|---|---|
| Discovery sample V2 | 1,500 reviews from 904 products |
| Sampling design | review-level stratified SRSWOR; no product cap |
| Eligible review population | 114,025 reviews |
| Sum of discovery weights | 114,025 |
| Discovery V3 plan | 15 batches × 100 reviews |
| Saved V3 Gemini responses | 15 of 15 |
| Grounded V3 mentions | 2,373 across 1,353 reviews |
| V3 candidate labels | 202 measured; 89 selected for normalization |
| V3 semantic normalization | 89 labels covered exactly once; 39 proposed aspects |
| Taxonomy V2 | approved; 39 aspects, 88 assigned labels, 1 exclusion |
| Taxonomy V1 | labels and aliases are qualitatively usable only |
| Extraction evaluation V2 | 160 representative + 80 targeted reviews |
| Sum of representative evaluation weights | 112,525 |
| Human gold annotations | 0 complete; 240 pending |

The saved executions of both notebooks in `notebooks/03_aspects/` are clean:
all non-empty code cells were executed and neither notebook contains an error
output. The taxonomy notebook presents V1 only as a qualitative historical
artifact; the evaluation notebook validates the corrected V2 sample and queue.

The machine-readable audit is
[`artifact_validity_v1.json`](../reports/aspects/artifact_validity_v1.json).

## Statistical Contract

The V2 discovery sample uses stratified simple random sampling without
replacement at review level. No product cap is applied before sampling. Within
each stratum, a selected review receives the inverse-inclusion weight
`N_h / n_h`; consequently the 1,500 weights sum to the full eligible population
of 114,025 reviews. The 904-product count is an observed property of the sample,
not a quota or cap.

All current discovery artifacts are identity-checked by dataset, niche, sample
schema, discovery version, backend, and model. Hashes bind the sample to the
batch plan. V2 Gemini responses were produced from different review membership
and cannot be copied, relabeled, or aggregated as V3 output.

For extraction evaluation, only the 160-row representative component is a
probability sample. It also uses review-level stratified SRSWOR without a
product cap, and its weights sum to the 112,525-review eligible holdout
population. The 80-row targeted component is a deterministic alias-coverage
diagnostic sample, uses at most two reviews per product, and has no population
weight. Never combine targeted rows with representative rows to estimate
prevalence.

## Current Artifacts

```text
config/aspects/hair_conditioners_discovery_v3.json
config/aspects/hair_conditioners_extraction_eval_v2.json
data/processed/.../aspect_discovery/hair_conditioners_v2_sample.parquet
data/processed/.../aspect_discovery/hair_conditioners_discovery_v3/batch_plan.jsonl
data/processed/.../aspect_discovery/hair_conditioners_discovery_v3/responses/
data/processed/.../aspect_discovery/hair_conditioners_discovery_v3/observations.parquet
data/processed/.../aspect_discovery/hair_conditioners_discovery_v3/candidates.json
data/processed/.../aspect_discovery/hair_conditioners_discovery_v3/normalization.json
data/processed/.../aspect_evaluation/hair_conditioners_extraction_eval_v2/sample.parquet
notebooks/03_aspects/hair_conditioners_extraction_eval_v2_review_queue.csv
reports/aspects/hair_conditioners_v2_sample_report.json
reports/aspects/hair_conditioners_discovery_v3_batch_plan.json
reports/aspects/hair_conditioners_discovery_v3_gemini_readiness.json
reports/aspects/hair_conditioners_discovery_v3_run.json
reports/aspects/hair_conditioners_discovery_v3_candidates.json
reports/aspects/hair_conditioners_taxonomy_v2_proposal.json
reports/aspects/hair_conditioners_taxonomy_v2_review_queue.csv
config/aspects/hair_conditioners_taxonomy_v2_reviewed.json
reports/aspects/hair_conditioners_taxonomy_v2.json
reports/aspects/hair_conditioners_extraction_eval_v2_sample_report.json
reports/aspects/artifact_validity_v1.json
```

The JSONL plan and Parquet samples contain review text and remain under the
ignored processed-data directory. Compact reports contain counts, identities,
and hashes without duplicating the review corpus.

V3 normalization and Taxonomy V2 review are complete. The 39 approved aspects
have quantitatively valid support estimates derived from the corrected
probability sample.

## Rebuild the Local V3 Inputs

Build the probability sample:

```bash
python -m src.data.aspect_sampling \
  config/datasets/amazon_reviews_2023_beauty_2021_2023_v1.json \
  config/niches/hair_conditioners_v1.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_v2_sample.parquet \
  --report-path reports/aspects/hair_conditioners_v2_sample_report.json \
  --sample-size 1500 --minimum-words 5 --minimum-per-stratum 20 --seed 42
```

Build the 15-by-100 V3 plan:

```bash
python -m src.llm.aspect_discovery prepare \
  config/aspects/hair_conditioners_discovery_v3.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_v2_sample.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v3/batch_plan.jsonl \
  --report-path reports/aspects/hair_conditioners_discovery_v3_batch_plan.json
```

Both local artifacts already exist and are hash-verified. Rebuilding them with
the recorded configuration must reproduce the same membership and identities.

## External Gemini Gate

Authentication is external to repository configuration. Store the key only as
`GEMINI_API_KEY` in a local ignored `.env`; never place keys in notebooks,
reports, source files, Git, or chat. Free Tier prompts and responses may be used
by Google to improve its products. This workflow sends public Amazon review
text but not source `user_id` values.

Readiness can be checked without sending review batches:

```bash
python -m src.llm.gemini_client \
  config/aspects/hair_conditioners_discovery_v3.json
```

The following command makes external Gemini API requests. Run one batch,
inspect the saved response, and then resume only after approval:

```bash
python -m src.llm.aspect_discovery run \
  config/aspects/hair_conditioners_discovery_v3.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v3/batch_plan.jsonl \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v3/responses \
  --report-path reports/aspects/hair_conditioners_discovery_v3_run.json \
  --limit 1
```

Each completed batch is validated and saved before the next request. A resumed
run revalidates and skips completed batches. After all 15 responses exist, run
`aggregate`, `normalize`, `proposal`, create a new review queue, and perform a
new human review under the same V3 identity. Do not copy V1 human decisions onto
the changed V3 candidate set.

## Safety and Validation Contract

Reviews are untrusted quoted data. The system instruction requires the model to
ignore instructions inside reviews and extract only explicitly supported
concepts. Every response must satisfy all of these checks:

- exactly one observation object for each supplied `review_id`;
- no missing, duplicate, or unexpected review IDs;
- every evidence phrase is a verbatim contiguous substring of its review;
- response JSON validates against the versioned Pydantic contract;
- every completed batch is persisted before another request is sent;
- resumed runs validate saved batches before skipping them.

Gemini proposes labels and semantic merges. Python computes support, product
coverage, valid probability weights, and lineage hashes. Targeted diagnostic
rows never receive population weights.

## Taxonomy V1: Qualitative Use Only

`hair_conditioners_taxonomy_v1` remains useful as a human-reviewed dictionary:
its 33 aspect labels, semantic boundaries, and 165 normalized aliases may be
used for annotation guidance and qualitative inspection. It was derived from
the historical V1 discovery sample, which capped reviews per product before
sampling and omitted the first-stage inclusion probability.

Therefore `weighted_review_support`, `weighted_population_share`, and every
prevalence or ranking derived from those fields are invalid. The fields may
remain in frozen JSON for lineage, but current analysis must ignore them. A new
quantitative taxonomy requires complete V3 responses, normalization, and human
review.

## Historical V2 Execution Record — Not a Current Reproduction Path

`hair_conditioners_discovery_v2` is frozen audit history. It completed 15 of 15
batches over 1,500 legacy-sample reviews and recorded:

- 1,332 reviews with at least one retained mention;
- 2,213 grounded mentions;
- 43 discarded ungrounded phrases (1.91% of raw mentions);
- 189 measured raw candidate labels;
- 80 labels sent to semantic normalization;
- 23 proposed canonical aspects;
- 462,334 Gemini tokens.

The historical review recorded 3 approvals, 7 edits, 11 splits, and 2
exclusions, producing the 33-label Taxonomy V1 dictionary. Those qualitative
decisions remain auditable, but all V2 weighted support and population-share
claims are invalid.

The command transcript below is preserved only to explain lineage. Do not run
it as a current workflow: the strict runtime rejects its legacy sample/plan
schema, and none of these artifacts may be mixed with V3.

```bash
# HISTORICAL / NON-REPRODUCIBLE WITH THE CURRENT STRICT RUNTIME
python -m src.llm.aspect_discovery run \
  config/aspects/hair_conditioners_discovery_v2.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/batch_plan.jsonl \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/responses \
  --report-path reports/aspects/hair_conditioners_discovery_v2_run.json \
  --limit 1

python -m src.analytics.aspect_taxonomy aggregate \
  config/aspects/hair_conditioners_discovery_v2.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/batch_plan.jsonl \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/responses \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/observations.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/candidates.json \
  --report-path reports/aspects/hair_conditioners_discovery_v2_candidates.json

python -m src.analytics.aspect_taxonomy normalize \
  config/aspects/hair_conditioners_discovery_v2.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/candidates.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/normalization.json

python -m src.analytics.aspect_taxonomy proposal \
  config/aspects/hair_conditioners_discovery_v2.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/observations.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/candidates.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/normalization.json \
  reports/aspects/hair_conditioners_taxonomy_v1_proposal.json

python -m src.analytics.aspect_taxonomy review-queue \
  reports/aspects/hair_conditioners_taxonomy_v1_proposal.json \
  reports/aspects/hair_conditioners_taxonomy_v1_review_queue.csv

python -m src.analytics.aspect_taxonomy materialize-approved \
  reports/aspects/hair_conditioners_taxonomy_v1_proposal.json \
  reports/aspects/hair_conditioners_taxonomy_v1_reviewed.csv \
  config/aspects/hair_conditioners_taxonomy_v1_reviewed.json \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/observations.parquet \
  data/processed/amazon_reviews_2023_beauty_2021_2023_v1/aspect_discovery/hair_conditioners_discovery_v2/candidates.json \
  reports/aspects/hair_conditioners_taxonomy_v1.json
```

The historical normalization response omitted one of 80 candidates. Its
invalid payload is retained as `normalization.invalid.json`; deterministic
completion kept `hair feel` separate rather than inventing a merge. This detail
explains the frozen V1 lineage but does not validate its weighted statistics.

The still earlier `hair_conditioners_discovery_v1` pilot stopped after 11 of 60
batches when the model's Free Tier daily request allowance was reached. Its
partial artifacts are audit history only and are not part of V2 or V3 results.

Model and structured-output behavior is documented in the official
[Gemini 3.5 Flash-Lite documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-flash-lite)
and the
[Google Gen AI Python SDK documentation](https://googleapis.github.io/python-genai/).
