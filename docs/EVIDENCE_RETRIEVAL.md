# Evidence Retrieval and RAG Context

## Current Status

The first local hybrid retrieval layer and grounded Gemini answer path are
complete for versioned seller workspaces. The system searches only public
review text already stored in the approved historical snapshot.

The index uses a deterministic 30,000-feature word/phrase TF-IDF matrix reduced
to 128 normalized latent dimensions. All 118,804 Hair Conditioners reviews are
indexed in a NumPy-backed artifact with source and artifact hashes. It is a
compact local LSA baseline, not a claim that a modern embedding model has been
fully evaluated.

The retriever:

- filters by seller/competitor product scope;
- optionally filters by aspect and aspect sentiment;
- combines exact word/phrase TF-IDF relevance with latent-vector similarity;
- includes aspect probability and helpful votes as small secondary signals;
- removes exact normalized-text duplicates;
- limits repeated evidence from one product;
- returns full review text, rating, date, product, aspect, and stable review ID.

The RAG context builder combines aggregate report facts with at most eight
review texts and assigns temporary citations such as `[R1]`. With explicit
consent, the answer endpoint sends Gemini the question, aggregate facts, public
review text, rating/date/aspect/sentiment, and brand. It does not send `user_id`,
ASIN, stable review IDs, or workspace IDs. Its prompt treats review text as
untrusted input, forbids turning one review into a general claim, and requires
limitations in the answer. Responses containing absent or malformed review
citations are rejected locally (with one corrective regeneration attempt).

A deterministic Russian/English router covers the current aspect vocabulary.
A stronger multilingual embedding candidate and formal human relevance and
answer-quality evaluation remain open.

## API

```text
GET /api/v1/products/search
POST /api/v1/workspaces
GET /api/v1/workspaces/{version}/evidence/search
GET /api/v1/workspaces/{version}/rag/context
POST /api/v1/workspaces/{version}/ask
GET /api/v1/workspaces/{version}/aspects/{aspect_id}/trend
GET /api/v1/workspaces/{version}/export.xlsx
```

The interactive documentation is available at `/docs` while the FastAPI
service is running.
