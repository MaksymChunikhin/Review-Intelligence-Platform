# Review Intelligence Platform

Domain-agnostic AI platform for analyzing customer reviews using NLP, ML, Vertex AI, Gemini, RAG and AI agents.

## Project Goal

Build a reusable system that can analyze review datasets from different domains:

- Consumer Electronics
- Home & Kitchen
- Automotive
- Hotels
- Games
- and other review-based domains

The system should automatically detect the dataset structure, normalize it into a universal review schema, analyze sentiment and customer aspects, generate embeddings, perform semantic search, and provide grounded insights through Gemini.

## Architecture

```text
Any Review Dataset
        ↓
Data Ingestion
        ↓
Schema Detection
        ↓
Universal Review Schema
        ↓
NLP / ML
        ↓
Domain Detection
        ↓
Dynamic Aspects
        ↓
Vertex AI Embeddings
        ↓
Vector Search
        ↓
RAG
        ↓
Gemini AI Analyst
        ↓
FastAPI / Streamlit


Project Structure
Review-Intelligence-Platform/
│
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── analytics/
│   ├── ml/
│   ├── embeddings/
│   ├── rag/
│   └── llm/
├── app/
├── tests/
├── PLAN.md
├── README.md
├── requirements.txt
└── .gitignore



Development Status

🚧 Project initialization

See PLAN.md for the complete roadmap