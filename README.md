<p align="center">
  <img src="docs/assets/veridoc-banner.svg" alt="Advanced-RAG-Document-Intelligence — Ask your documents. Get answers with sources." width="100%" />
</p>

# Advanced-RAG-Document-Intelligence

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Node.js](https://img.shields.io/badge/Node.js-20%2B-339933?logo=node.js)](https://nodejs.org)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17%2Bpgvector-4169E1?logo=postgresql)](https://github.com/pgvector/pgvector)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek%20%7C%20Gemini%20%7C%20OpenAI-purple)](https://www.deepseek.com)

> **Enterprise-Ready Advanced RAG & Document Intelligence Platform (VeriDoc)** — Self-hostable, production-grade document intelligence featuring **PostgreSQL 17 + pgvector**, **Hybrid Search (Dense + FTS RRF)**, **Parent-Child Context Window Expansion**, **Multi-Query Decomposition**, **Durable Async Worker Ingestion**, and **Streaming Cited Answers** powered by **DeepSeek (V3 / R1)**, **Google Gemini 2.5 Flash**, and **OpenAI**.

---

## Table of Contents

- [Overview](#overview)
- [Key Differentiators](#key-differentiators)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
  - [Option A — Docker Compose (Recommended)](#option-a--docker-compose-recommended)
  - [Option B — Local Development](#option-b--local-development)
- [Essential Environment Variables](#essential-environment-variables)
- [API Highlights & Interactive Docs](#api-highlights--interactive-docs)
- [Database Schema](#database-schema)
- [Testing & Quality Gates](#testing--quality-gates)
- [Security & Performance](#security--performance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

### What Problem It Solves

Teams accumulate mission-critical knowledge across PDFs, spreadsheets, Word documents, and technical wikis. Standard LLMs hallucinate or cannot answer without exact grounding. This platform bridges that gap: it ingests documents through a resilient background pipeline, indexes them with vector embeddings + full-text search, expands contextual boundaries, and delivers real-time streaming answers with verifiable, click-through page citations.

---

## Key Differentiators

| Differentiator | Implementation Details |
|----------------|------------------------|
| **Multi-Provider BYOK** | Provider keys (DeepSeek, Gemini, OpenAI) stay in browser sessions via `X-Provider-Api-Key` and are used temporarily during ingestion jobs without SaaS lock-in. |
| **Durable Ingestion Worker** | Upload returns `202 Accepted` immediately. A dedicated background worker (`backend/worker.py`) consumes jobs with automatic font/NUL byte sanitization. |
| **Production pgvector Engine** | PostgreSQL 17 with `pgvector` HNSW index and `tsvector` full-text search fused via Reciprocal Rank Fusion (RRF). |
| **Context Window Expansion** | Generic parent-child expansion (+2 forward, -1 backward chunks) eliminates boundary information loss (`backend/services/window_expansion.py`). |
| **Sub-Query Decomposition** | Complex comparative queries are broken down into sub-queries, routed per document scope, and synthesized into cohesive answers. |
| **Multi-Tenant Workspaces & RBAC** | Organization workspaces with fine-grained role-based access control (`owner`, `admin`, `editor`, `viewer`). |
| **Split-Pane PDF Notebook** | Real-time SSE streaming with inline bracketed citations (`[1]`, `[2]`) that jump directly to source pages in an interactive PDF viewer. |

---

## Features

### Core Capabilities
- **Multi-LLM Support:** DeepSeek (`deepseek-chat`, `deepseek-reasoner` / R1), Google Gemini (`gemini-2.5-flash`), and OpenAI (`gpt-4o`, `gpt-4o-mini`).
- **Comprehensive Document Parsing:** PDF, DOCX, XLSX, CSV, PPTX, TXT, Markdown, and Web URLs with checksum deduplication.
- **Robust Ingestion Pipeline:** Automatic text/font artifact cleanup, retry queue with exponential backoff, and heartbeat health checks.
- **Streaming Cited Chat:** Real-time Server-Sent Events (SSE) streaming answers with confidence scores and source excerpts.
- **Analysis Modes:** `ask` (standard Q&A), `compare` (multi-document decomposition), `extract` (structured fields), and `brief` (executive summary).
- **Notebook Experience:** Split-pane PDF viewer (`/documents/{id}/notebook`) with click-to-page navigation.

### Advanced RAG Techniques (Feature-Flagged)
- **Hybrid Search:** Dense vector retrieval + PostgreSQL full-text search fused via Reciprocal Rank Fusion (RRF).
- **Parent-Child Context Expansion:** Dynamic neighboring chunk expansion around top hits.
- **Cross-Encoder Reranking:** Optional Cohere, Jina, or local BGE rerankers.
- **Query Transformations:** HyDE (Hypothetical Document Embeddings), step-back prompting, and sub-question decomposition.
- **Active Grounding & Guardrails:** Verification module to detect and mitigate hallucinations against source chunks.

---

## Architecture Overview

The system operates as a four-process decoupled architecture in production:

```mermaid
flowchart TB
  subgraph Client["Frontend (Next.js 16)"]
    UI["Web Dashboard & PDF Notebook<br/>(React 19 / TypeScript)"]
  end

  subgraph API["Backend API (FastAPI)"]
    MW["Security & Rate Limit Middleware"]
    R["API Routers (Chat, Ingest, Workspaces)"]
  end

  subgraph Worker["Background Engine (Python Worker)"]
    JQ["Job Queue Consumer (document_jobs)"]
    PIPE["Pipeline: Extract -> Chunk -> Embed -> Index"]
  end

  subgraph Storage["Data & Vector Store"]
    PG[("PostgreSQL 17 + pgvector<br/>(HNSW Index + Full-Text Search)")]
  end

  UI -->|HTTP / SSE Stream| MW
  MW --> R
  R --> PG
  JQ --> PIPE
  PIPE --> PG
  JQ -.->|Heartbeat /ready| PG
```

### Ingestion & Chat Execution Flow

```mermaid
sequenceDiagram
  autonumber
  actor User
  participant API as FastAPI Gateway
  participant Worker as Background Worker
  participant DB as PostgreSQL + pgvector
  participant LLM as LLM Provider

  User->>API: Upload File (POST /api/ingest)
  API->>DB: Record job (status = queued)
  API-->>User: 202 Accepted {job_id}

  Worker->>DB: Claim next queued job
  Worker->>Worker: Parse text, chunk, embed
  Worker->>DB: Store chunks with pgvector HNSW index
  Worker->>DB: Update job (status = ready)

  User->>API: Question (POST /api/chat/stream)
  API->>DB: Hybrid Search (pgvector + FTS RRF)
  API->>API: Expand Parent-Child Context Windows
  API->>LLM: Stream prompt with grounded contexts
  LLM-->>API: Stream token chunks
  API-->>User: Real-time SSE with citations [1], [2]
```

---

## Tech Stack

| Domain | Technologies |
|--------|--------------|
| **Frontend** | [Next.js 16](https://nextjs.org) (App Router), [React 19](https://react.dev), [TypeScript 5](https://www.typescriptlang.org), [Tailwind CSS 4](https://tailwindcss.com), [Framer Motion](https://www.framer.com/motion/), [react-pdf](https://github.com/wojtekmaj/react-pdf) |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com) 0.115, [Uvicorn](https://www.uvicorn.org), [Pydantic v2](https://docs.pydantic.dev), [psycopg 3](https://www.psycopg.org) (async), `sse-starlette`, `orjson` |
| **Vector & DB** | [PostgreSQL 17](https://www.postgresql.org/) with [pgvector](https://github.com/pgvector/pgvector) (HNSW index + tsvector), SQLite fallback |
| **AI & Models** | DeepSeek (V3 / R1), Google Gemini (2.5 Flash), OpenAI (GPT-4o), FastEmbed ONNX, Cohere / Jina rerankers |
| **DevOps & Infra** | Docker, Docker Compose, GitHub Actions CI/CD, Devcontainer |

---

## Quick Start

### Option A — Docker Compose (Recommended)

Run the entire production stack (Postgres + pgvector, API, Worker, Next.js Web) with a single command:

```bash
# 1. Clone repository
git clone https://github.com/khoiabc2020/Advanced-RAG-Document-Intelligence.git
cd Advanced-RAG-Document-Intelligence

# 2. Configure environment
cp .env.example .env

# 3. Start all services
docker compose up -d --build
```

Access the services:
* **Web Dashboard:** [http://localhost:3000/dashboard](http://localhost:3000/dashboard)
* **Interactive API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check:** `http://localhost:8000/health` (Liveness) & `http://localhost:8000/ready` (Readiness)

---

### Option B — Local Development

#### 1. Start PostgreSQL (pgvector)
```bash
docker compose up -d postgres
```

#### 2. Install & Run Frontend
```bash
npm ci
npm run dev
# Running on http://localhost:3000
```

#### 3. Install & Run Backend (Terminal 1)
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements.txt

# Start API Server
python run_server.py
# Running on http://127.0.0.1:8000
```

#### 4. Run Ingestion Worker (Terminal 2)
```bash
python run_worker.py
```

---

## Essential Environment Variables

Configure your `.env` file based on [`.env.example`](./.env.example):

| Variable | Default / Example | Purpose |
|----------|-------------------|---------|
| `DEFAULT_SUPERUSER_PASSWORD` | `dev-admin-secret` | **Required.** Initial admin user password (`admin@example.com`). |
| `DATABASE_URL` | `postgresql://document_rag:document_rag@localhost:5433/document_rag` | PostgreSQL DSN for pgvector and full-text search. |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Target URL for Next.js API proxying. |
| `MAX_DOCUMENT_BYTES` | `10485760` (10MB) | Maximum file upload size cap. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Text chunking parameters. |
| `RAG_TOP_K` | `5` | Default number of chunks retrieved per query. |

> Complete list of feature flags (`RETRIEVAL_HYBRID_ENABLED`, `RETRIEVAL_RERANKER_ENABLED`, etc.) is documented in [`.env.example`](./.env.example).

---

## API Highlights & Interactive Docs

FastAPI provides full interactive Swagger documentation at **`http://127.0.0.1:8000/docs`**.

### 1. Ingest a Document
```bash
curl -X POST http://127.0.0.1:8000/api/ingest \
  -H "X-Client-Session: demo-session" \
  -H "X-Provider-Api-Key: $OPENAI_API_KEY" \
  -F "provider=openai" \
  -F "embedding_model=text-embedding-3-small" \
  -F "file=@./contract.pdf"
```

### 2. Stream a Grounded Answer (SSE)
```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-Client-Session: demo-session" \
  -H "X-Provider-Api-Key: $OPENAI_API_KEY" \
  -d '{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "document_id": "<DOCUMENT_UUID>",
    "question": "What are the termination conditions?",
    "mode": "ask"
  }'
```

---

## Database Schema

Managed via automated migrations (`backend/migrations.py` Schema v15):

```mermaid
erDiagram
  users ||--o{ auth_sessions : has
  users ||--o{ documents : owns
  documents ||--o{ document_chunks : contains
  documents ||--o{ document_jobs : processes
  documents ||--o{ conversations : scopes
  conversations ||--o{ messages : contains
  workspaces ||--o{ workspace_sources : lists
  workspaces ||--o{ workspace_members : grants
  workspaces ||--o{ workspace_artifacts : stores
  workspaces ||--o{ documents : contains
```

* **`documents` & `document_chunks`:** Store raw metadata, chunk texts, and HNSW vector embeddings.
* **`document_jobs`:** Durable queue tracking asynchronous extraction, chunking, and embedding stages.
* **`conversations` & `messages`:** Persisted chat history with structured citations (`chunk_id`, `score`, `page`).
* **`workspaces` & `workspace_members`:** Multi-tenant access boundaries with RBAC enforcement.

---

## Testing & Quality Gates

```bash
# Run backend test suite
pytest -q backend/tests

# Run retrieval quality benchmark against golden set
pytest -m eval backend/tests/eval

# Validate frontend types and ESLint
npm run check
```

Automated GitHub Actions workflows run CI tests, linting, and scheduled retrieval evals on every commit.

---

## Security & Performance

- **BYOK (Bring Your Own Key):** API keys stay client-side in browser session storage and are only used ephemerally during requests.
- **Authentication:** PBKDF2-SHA256 (480k iterations) password hashing with secure HttpOnly cookies.
- **Performance:** `orjson` serialization on SSE streams, HNSW vector indexing for sub-millisecond nearest neighbor search, and SHA-256 deduplication to prevent duplicate embeddings.
- **Tenant Isolation:** All database queries are strictly scoped by `owner_id` and workspace RBAC policies.

---

## Contributing

Contributions are welcome! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards, and [SECURITY.md](SECURITY.md) to report vulnerabilities.

---

## License

Distributed under the [Apache License 2.0](LICENSE).
