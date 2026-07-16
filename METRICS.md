# BioFeed AI — System Metrics: Current State & Targets

> Last measured: 2026-07-16 (local Docker stack, branch `v0.2-content-platform`)
> Companion docs: [`PROJECT_STATUS.md`](./PROJECT_STATUS.md) · [`BLUEPRINT.md`](./BLUEPRINT.md)

This file answers "where are we now, and what are we building toward" for every
system-level metric. **Current** values are measured on the running local stack;
**Target** values come from the roadmap (v0.5–v1.0) and are goals, not claims.

---

## 1. Content & Ingestion

| Metric | Current (measured 2026-07-16) | Building toward |
|---|---|---|
| Sources ingested | **4 RSS feeds** — FiercePharma, STAT News, GEN, BioPharma Dive | 10+ sources: + PubMed (E-utilities), bioRxiv, FDA announcements (v0.2–v0.5) |
| Articles in DB | **120** (FiercePharma 41, STAT News 40, GEN 20, BioPharma Dive 19) | Thousands+, growing continuously once scheduled ingestion lands |
| Ingestion cadence | Manual trigger only (`POST /ingest/run`) | Scheduled runs via Celery + Redis or APScheduler (v0.2 remaining work) |
| Dedup effectiveness | 3-tier dedup (URL → DOI → normalized-title hash); skipped ~65 duplicates on latest run | Same layers, exercised across PubMed/bioRxiv where DOI matching matters most |
| Run observability | `IngestionRun` table + `GET /ingest/runs` (timestamps, totals, per-source detail) | Prometheus/Grafana dashboards on top (v1.0) |

## 2. Users & Personalization

| Metric | Current | Building toward |
|---|---|---|
| Users | **0** — no user model exists | Auth in v0.3 (Sign in with Apple, Google OAuth, JWT); real users via the iOS app (v0.4) |
| Interaction signals | None captured | Likes, bookmarks, reading time, scroll depth, hides, searches (v0.6) |

## 3. ML / NLP Pipeline

| Metric | Current | Building toward |
|---|---|---|
| Embedding model | **PubMedBERT** (`NeuML/pubmedbert-base-embeddings`, 768-dim) in Docker; hashing fallback in tests/CI via `EMBEDDING_BACKEND` | Compare BioBERT/other biomedical encoders against a real eval set (v0.5 remaining) |
| Number of embeddings | One per article, written at ingestion; **not re-measured since the corpus was last re-ingested** | Same, + per-user embeddings (v0.6) |
| Retrieval | **pgvector HNSW (cosine)** behind `GET /search?q=` and `GET /articles/{id}/related`; verified end-to-end on Postgres | FAISS only if pgvector latency becomes the bottleneck (v0.7); sub-100 ms target |
| Retrieval quality | **Not yet measured** — no labeled eval set; correctness verified by inspection only | NDCG@k / recall@k against a hand-labeled query → article set (v0.5 remaining) |
| Ranking model | **None** | Two-tower retrieval + LightGBM/XGBoost reranker; cross-encoder experiments (v0.7) |
| Recommendation metrics | **N/A — nothing to evaluate yet** | NDCG@k / recall@k offline; CTR, save-rate, dwell time online (v0.6–v0.7) |
| Training dataset | **None** | Interaction logs from real usage; features: user/article embeddings, freshness, publisher, popularity, topic, history (v0.7) |
| ML inference latency | **N/A** | Sub-100 ms feed-ranking budget end-to-end (retrieval + rerank) |
| Explainability | None | "Recommended because…" per feed item (v0.9) |
| Knowledge graph | None | Entity graph grounded in UMLS / MONDO / ChEMBL (v0.8) |

## 4. API & Serving (measured)

| Metric | Current (measured 2026-07-16) | Building toward |
|---|---|---|
| API latency | **~4–5 ms** per `GET /articles?limit=20` (warm, local Docker) | Keep p95 low as feed becomes personalized; sub-100 ms personalized-feed budget |
| API throughput | **~190 req/s** at concurrency 10 (conservative floor — test spawned one curl process per request) | Proper load testing (Locust/k6) with published p50/p95/p99 once deployed |
| Pagination | Bounded server-side (`limit` capped) | Cursor pagination if feed depth demands it |

## 5. Infrastructure & Operations

| Metric | Current | Building toward |
|---|---|---|
| Deployment | Local **Docker Compose** (FastAPI + Postgres 16) | Dockerized AWS deployment with CI/CD (v1.0) |
| Cloud | Not deployed | AWS (v1.0) |
| CI | GitHub Actions: `alembic upgrade head` against a pgvector service + pytest (API, RSS, CRUD, dedup, retrieval) on Python 3.12 | + tests for new sources, Redis service in CI when scheduling lands |
| Monitoring | `IngestionRun` history only | Prometheus + Grafana, MLflow for experiment tracking (v1.0) |
| Database | PostgreSQL 16 (**pgvector image**), SQLAlchemy 2.0 typed ORM, Alembic migrations (0001–0003) | Decision made: pgvector in-database, no FAISS sidecar until latency demands it |

## 6. Hardening Done So Far (not perf numbers, but real)

- **Race-condition-safe inserts** — DB-level unique constraint on URL with
  `IntegrityError` recovery, so concurrent ingestion runs can't duplicate rows.
- **Per-source error isolation** — one dead feed never aborts a run; errors are
  captured per source and surfaced in the API response and run history.
- **Bounded pagination** — clients can't request unbounded result sets.
- **Layered cross-source dedup** — URL → DOI → sha256 of normalized title.

---

## Measurement notes

- Latency: 5 sequential `curl` timings against the local containerized API;
  first request ~42 ms (cold), then 4–5 ms steady-state.
- Throughput: 200 requests at concurrency 10 via parallel curl processes;
  process-spawn overhead means the real capacity is higher. Re-measure with a
  proper load tool before quoting externally.
- Article counts: `SELECT count(*) FROM articles` on the live Postgres volume.
  These numbers go stale — re-run ingestion and re-measure before updating this file.
