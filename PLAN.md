# Review Intelligence Platform — Project Roadmap

## Project Goal

Build a domain-agnostic AI platform that can analyze review datasets from different domains
(e-commerce, electronics, hotels, cars, games, etc.) without being tied to one product category.

The first development dataset will be the existing Amazon Home & Kitchen reviews from V1.
Later, the same pipeline must be tested on a different domain to prove generalization.

---

# 0. Repository & Local Environment

- [ ] Create GitHub repository: `Review-Intelligence-Platform`
- [ ] Create a new local project folder with the same name
- [ ] Open the folder in VS Code
- [ ] Initialize Git
- [ ] Connect local repository to GitHub
- [ ] Create Python virtual environment
- [ ] Create `requirements.txt`
- [ ] Create `.gitignore`
- [ ] Create initial README
- [ ] Make first commit

Initial structure:

```text
Review-Intelligence-Platform/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── analytics/
│   ├── ml/
│   ├── embeddings/
│   ├── rag/
│   └── llm/
│
├── app/
├── tests/
│
├── requirements.txt
├── .gitignore
├── README.md
└── PLAN.md
```

---

# 1. Dataset

## First dataset

Use the existing Amazon Home & Kitchen dataset from V1.

- [ ] Locate/download the dataset
- [ ] Put raw data in `data/raw/`
- [ ] Do NOT commit the raw dataset to GitHub
- [ ] Inspect file format, size and columns
- [ ] Document the original schema

Goal: use the old Amazon dataset as the first test case, but do not make the new system Amazon-specific.

---

# 2. Universal Review Schema

Create a standard internal format that every review dataset must be converted into.

### Required fields

```text
review_id
product_id
product_name
rating
review_text
review_date
```

### Optional fields

```text
title
verified_purchase
helpful_votes
user_id
category
```

- [ ] Define schema
- [ ] Create validation rules
- [ ] Create conversion/adapter for Amazon dataset
- [ ] Verify resulting DataFrame

Goal:

```text
Any dataset
    ↓
Dataset adapter
    ↓
Universal Review Schema
```

---

# 3. Data Ingestion

Create reusable loaders.

Example:

```python
load_reviews("reviews.csv")
load_reviews("reviews.jsonl")
```

- [ ] CSV loader
- [ ] JSON/JSONL loader
- [ ] Schema detection
- [ ] Column mapping
- [ ] Data type normalization
- [ ] Error handling

Later, Gemini can help detect semantic column mappings such as:

```text
stars   → rating
comment → review_text
date    → review_date
product → product_name
```

---

# 4. Data Quality & Validation

After loading a dataset:

- [ ] Missing values
- [ ] Duplicate reviews
- [ ] Empty review text
- [ ] Invalid ratings
- [ ] Invalid dates
- [ ] Wrong data types
- [ ] Dataset statistics
- [ ] Automatic quality report

Example output:

```text
Reviews loaded: 124,532
Valid reviews: 122,840
Removed: 1,692
Rating scale: 1–5
Missing review text: 0.4%
```

---

# 5. Exploratory Data Analysis

Automatically calculate:

- [ ] Number of reviews
- [ ] Number of products
- [ ] Average rating
- [ ] Rating distribution
- [ ] Review volume over time
- [ ] Verified purchase distribution
- [ ] Helpful vote statistics
- [ ] Missing-value report

This part must work for ANY supported review dataset.

---

# 6. Baseline NLP

Before using an LLM, create a classical baseline.

- [ ] Text preprocessing
- [ ] Sentiment classification
- [ ] Rating-based weak labels when rating exists
- [ ] Pretrained sentiment model when rating is unavailable
- [ ] Basic metrics
- [ ] Error analysis

Important principle:

Python/ML should calculate objective facts and statistics.
Gemini should explain and synthesize those facts.

---

# 7. Domain Detection

Use Gemini to automatically identify the dataset domain.

Examples:

```text
Smartphones → Consumer Electronics
Cars → Automotive
Hotels → Hospitality
Games → Gaming
Kitchen appliances → Home & Kitchen
```

- [ ] Define domain-detection prompt
- [ ] Structured JSON output
- [ ] Validate output with Pydantic
- [ ] Store detected domain

---

# 8. Dynamic Aspect Extraction

The system must NOT have hard-coded aspects.

For example:

### Smartphones

```text
Battery
Camera
Display
Performance
Price
```

### Cars

```text
Fuel consumption
Comfort
Reliability
Engine
Maintenance
```

### Hotels

```text
Location
Room
Cleanliness
Staff
Breakfast
```

- [ ] Extract domain-specific aspects with Gemini
- [ ] Normalize aspect names
- [ ] Store aspects
- [ ] Link reviews to aspects

---

# 9. Vertex AI / Gemini

Create Google Cloud project and connect Vertex AI.

- [ ] Create GCP project
- [ ] Enable Vertex AI API
- [ ] Configure authentication
- [ ] Install Google Cloud / Vertex AI SDK
- [ ] Make first Gemini request
- [ ] Create reusable LLM client
- [ ] System prompts
- [ ] Structured output
- [ ] Error handling
- [ ] Token/cost tracking

First milestone:

```text
Python → Vertex AI → Gemini → structured response
```

---

# 10. Embeddings

Use Vertex AI embeddings for semantic review search.

```text
Review
   ↓
Vertex AI Embedding
   ↓
Vector
```

- [ ] Generate embeddings
- [ ] Batch embedding pipeline
- [ ] Store embeddings
- [ ] Track embedding model/version

---

# 11. Vector Search

- [ ] Select vector storage/search solution
- [ ] Index review embeddings
- [ ] Semantic similarity search
- [ ] Top-K retrieval
- [ ] Metadata filtering
- [ ] Product/category filtering

Example:

```text
Question:
"What are customers complaining about battery life?"

        ↓

Semantic retrieval

        ↓

Relevant reviews
```

---

# 12. RAG

Build the core RAG pipeline:

```text
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

- [ ] Retrieval
- [ ] Context construction
- [ ] Prompt template
- [ ] Gemini generation
- [ ] Source review references
- [ ] Hallucination checks
- [ ] RAG evaluation

---

# 13. Analytics Engine

Create deterministic Python functions:

```text
get_statistics()
get_sentiment_distribution()
get_aspect_statistics()
get_rating_trends()
find_common_complaints()
compare_products()
```

These functions calculate facts.

Gemini explains the results.

---

# 14. AI Analyst

Combine:

```text
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

Example questions:

```text
What are customers complaining about?
What are the strongest advantages?
What are the biggest pain points?
How has sentiment changed over time?
Which product performs better?
Should I buy this product?
```

---

# 15. Agent / Function Calling

Give Gemini access to tools:

```text
search_reviews()
get_statistics()
get_sentiment()
get_aspects()
get_trends()
compare_products()
```

Architecture:

```text
User
 ↓
Gemini
 ↓
Tool selection
 ↓
Python function
 ↓
Tool result
 ↓
Gemini
 ↓
Final answer
```

---

# 16. FastAPI

Create backend API:

```text
POST /upload
POST /analyze
POST /ask
GET  /dataset/{id}
GET  /analytics/{id}
GET  /products/{id}
```

---

# 17. Streamlit UI

Main flow:

```text
Upload Dataset
      ↓
Detect Schema
      ↓
Analyze Dataset
      ↓
Dashboard
      ↓
Ask AI
```

Dashboard should show:

- dataset overview
- ratings
- sentiment
- aspects
- trends
- pain points
- AI Analyst chat

---

# 18. Testing

- [ ] Unit tests for data loading
- [ ] Schema validation tests
- [ ] NLP tests
- [ ] Retrieval tests
- [ ] RAG evaluation
- [ ] API tests
- [ ] Edge cases

---

# 19. Docker & GCP Deployment

- [ ] Dockerfile
- [ ] Containerize API
- [ ] Containerize application
- [ ] Environment variables
- [ ] Deploy to Google Cloud
- [ ] Connect production app to Vertex AI

---

# 20. Generalization Test

This is a critical milestone.

### Dataset A

Amazon Home & Kitchen

### Dataset B

Different domain, e.g. Electronics

The same application must work without rewriting the core pipeline.

Success criterion:

```text
New review dataset
       ↓
Automatic schema mapping
       ↓
Domain detection
       ↓
Dynamic aspects
       ↓
Sentiment
       ↓
Embeddings
       ↓
RAG
       ↓
AI Analyst
```

---

# Final Architecture

```text
                    ANY REVIEW DATASET
                           │
                           ▼
                   Data Ingestion
                           │
                           ▼
                 Schema Detection
                           │
                           ▼
                  Data Validation
                           │
                           ▼
                 Universal Schema
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          NLP/ML       Analytics      Metadata
             │             │
             └──────┬──────┘
                    ▼
              Domain Detection
                    │
                    ▼
            Dynamic Aspects
                    │
                    ▼
          Vertex AI Embeddings
                    │
                    ▼
              Vector Search
                    │
                    ▼
                  RAG
                    │
                    ▼
            Vertex AI / Gemini
                    │
              ┌─────┴─────┐
              ▼           ▼
          AI Analyst    AI Agent
              │           │
              └─────┬─────┘
                    ▼
                 FastAPI
                    │
                    ▼
               Streamlit UI
                    │
                    ▼
              Google Cloud
```

---

# Development order

The project should be developed in this exact order:

**1. GitHub + local project**

**2. Dataset**

**3. Universal Schema**

**4. Data Loader**

**5. Validation**

**6. Baseline NLP**

**7. Domain Detection**

**8. Vertex AI / Gemini**

**9. Dynamic Aspects**

**10. Embeddings**

**11. Vector Search**

**12. RAG**

**13. AI Analyst**

**14. Agent / Tools**

**15. FastAPI**

**16. Streamlit**

**17. Docker**

**18. GCP Deployment**

**19. Test on a second domain**

---

## Current milestone

We are currently at:

> **Milestone 1 — Create the local project + GitHub repository.**

Do not start Vertex AI yet. First make the data foundation universal.
