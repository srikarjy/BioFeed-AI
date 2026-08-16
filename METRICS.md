# BioFeed AI — System Metrics: Current State & Targets

> Last measured: 2026-08-15 (local Docker stack unless noted, branch `main`)
> Companion docs: [`PROJECT_STATUS.md`](./PROJECT_STATUS.md) · [`BLUEPRINT.md`](./BLUEPRINT.md)

This file answers "where are we now, and what are we building toward" for every
system-level metric. **Current** values are measured; **Target** values come
from the roadmap (v0.8–v1.0) and are goals, not claims. Numbers not
re-measured in this pass are carried forward from 2026-07-16 and flagged.

---

## 1. Content & Ingestion

| Metric | Current | Building toward |
|---|---|---|
| Sources ingested | **RSS (4 feeds: FiercePharma, STAT News, GEN, BioPharma Dive) + PubMed E-utilities + bioRxiv + medRxiv** — all four source types are implemented and registered (`app/ingestion/registry.py`) | FDA announcements; broader RSS list |
| Articles in DB | *(carried forward, 2026-07-16)* 65, one run's worth of the RSS sources | Thousands+, continuous — scheduler now runs this automatically; re-measure after a live deployment accumulates runs |
| Ingestion cadence | **Scheduled** — APScheduler `AsyncIOScheduler` runs every `INGESTION_INTERVAL_MINUTES` (default 60) via the FastAPI lifespan, plus manual trigger (`POST /ingest/run`) | Same; Celery/Redis only if a single-process scheduler becomes the bottleneck |
| Dedup effectiveness | *(carried forward)* 3-tier dedup (URL → DOI → normalized-title hash); skipped ~65 duplicates on the last measured RSS-only run | Re-measure now that PubMed/bioRxiv DOIs exercise the DOI tier, which RSS-only runs didn't |
| Run observability | `IngestionRun` table + `GET /ingest/runs` (timestamps, totals, per-source detail) | Prometheus/Grafana dashboards on top (exist today for the LLM path; not yet extended to core ingestion) |

## 2. Users & Personalization

| Metric | Current | Building toward |
|---|---|---|
| Users | **0 real users** — `User`/`UserInteraction`/`UserEmbedding` models and full CRUD exist and are tested (`tests/test_recommendations.py`), but nothing has signed up because auth (v0.3) and the mobile app (v0.4) don't exist yet | Real users via the iOS app once v0.3/v0.4 ship |
| Interaction signals | **Captured, schema-complete, zero real volume** — read time, bookmark, like, hide, search all modeled and weighted into the user embedding | Real interaction logs feed v0.7 training |

## 3. ML / NLP Pipeline

| Metric | Current | Building toward |
|---|---|---|
| Embedding model | **PubMedBERT** (`NeuML/pubmedbert-base-embeddings`, 768-dim) in Docker; hashing fallback in tests/CI via `EMBEDDING_BACKEND` | Compare against BioBERT/other biomedical encoders |
| Number of embeddings | *(carried forward)* 65/65 articles from the last measured run — full coverage; `embed_missing` backfills, ingestion embeds inline | Continuous, at ingestion scale |
| Retrieval | **pgvector HNSW (cosine)** behind `GET /search?q=` and `GET /articles/{id}/related`; verified end-to-end on Postgres | FAISS only if pgvector latency becomes the bottleneck; not the case yet |
| **Retrieval quality — now measured, both backends** | `scripts/eval_retrieval.py` against `tests/fixtures/retrieval_eval.json` (26 hand-labeled queries, 30-article multi-topic corpus). **Hashing embedder**: Recall@5 = 0.962 (25/26), Recall@10 = 1.000 (26/26), NDCG@10 = 0.959. **Real PubMedBERT** (`EMBEDDING_BACKEND=sentence-transformers`, installed and run this pass): **Recall@5 = 1.000 (26/26), Recall@10 = 1.000 (26/26), NDCG@10 = 0.989** — beats the lexical-overlap baseline on every metric, including the one query the hashing embedder missed at k=5 ("new antibiotics for drug-resistant bacteria"). Recall@10 ≥ 0.9 enforced as a CI floor in `tests/test_retrieval_eval.py` (CI still runs the hash backend, to stay torch-free). | Expand corpus/query count; compare against BioBERT/other biomedical encoders on the same set |
| Ranking model | **Two-tower + LightGBM reranker, trained and verified end-to-end on a synthetic corpus** this pass (`scripts/seed_v07_synthetic.py` → `scripts/train_v07.py`): 139 synthetic articles / 10 topic clusters / 25 synthetic users → 344 positive pairs → two-tower (15 epochs) + reranker (688 samples, **validation AUC 0.911**, `user_item_cosine` the dominant feature by gain). Running this surfaced and fixed 4 real bugs (CI-breaking unconditional torch/lightgbm import, a hardcoded personal `MODEL_DIR` path, a pgvector-array truthiness bug, a lightgbm 4.x API break) plus an environment issue (torch+lightgbm together segfault on macOS without `OMP_NUM_THREADS=1`) — see `PROJECT_STATUS.md` §1 v0.7 and decisions 22–25. **Not trained on real usage** — no checkpoint ships in the repo (`.gitignore`s `*.pt`/`models/*.txt`), so `GET /users/{id}/feed` (v0.7 route) correctly falls back to the v0.6 heuristic feed by default; this only proves the pipeline runs, not that the model is good on real behavior | Train via `scripts/train_v07.py` once v0.3/v0.4 produce real interaction logs |
| Recommendation metrics | **N/A — no real users to evaluate against** | NDCG@k / recall@k offline once trained; CTR, save-rate, dwell time online after launch |
| Training dataset | **None (real)**; synthetic interaction fixtures exist for tests only | Interaction logs from real usage |
| ML inference latency | *(not re-measured this pass)* | Sub-100 ms feed-ranking budget end-to-end (retrieval + rerank) |
| Explainability | **Heuristic "recommended because…" string** per feed item (`crud._generate_reason`) — real but simple (nearest bookmarked/liked article, or a generic fallback), not a modeled explanation | Richer explanation modeling (v0.9) |
| Knowledge graph | None | Entity graph grounded in UMLS / MONDO / ChEMBL (v0.8) |
| Anomaly detection | **Real, implemented and tested** — cross-source burst detector (`app/anomaly/detector.py`) flags an article when other sources cover the same story within 48h, reusing pgvector similarity | Tune thresholds against real multi-source volume |
| LLM-generated explanations | **Route real and tested against a local CPU stand-in** (`scripts/stub_vllm_server.py`); **vLLM verified on a real GPU this pass** — `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` served via `vllm/vllm-openai` on a Hugging Face Jobs T4 (`--flavor t4-small`), same engine args as `llm_serving/serve.sh`. Real TTFT p50 105ms / p95 2.1s at concurrency 10 (degrading to p50 5.2s at concurrency 30 — one T4's `--max-num-seqs 16` limit, not a wiring problem). **Still not measured**: the same request through the actual FastAPI/SSE relay (this run hit vLLM directly, not through `app/llm/client.py`) — see `benchmarks/report.md` for exactly what's covered. | Benchmark through the full FastAPI/SSE relay path with the backend + Postgres also running on a GPU host |

## 4. API & Serving

| Metric | Current | Building toward |
|---|---|---|
| API latency | *(carried forward, 2026-07-16)* ~4–5 ms per `GET /articles?limit=20` (warm, local Docker) | Keep p95 low as feed becomes personalized; sub-100 ms personalized-feed budget |
| API throughput | *(carried forward)* ~190 req/s at concurrency 10 (conservative floor — curl-process-spawn overhead) | Proper load testing (Locust/k6) with published p50/p95/p99 once deployed |
| Anomaly-explain, CPU stand-in | concurrency 10: 8.1 req/s, p50 1228 ms; concurrency 50: 33.9 req/s, p50 1457 ms — **wiring validation only, not real model serving** (`benchmarks/report.md`) | Superseded by real vLLM/GPU numbers below for serving performance; still the only measurement of the FastAPI/SSE relay itself |
| Anomaly-explain, real vLLM on GPU (`vllm-t4-direct`) | **Measured 2026-08-15** on a Hugging Face Jobs T4 (`hf jobs run --flavor t4-small`, `TheBloke/Mistral-7B-Instruct-v0.2-AWQ`): concurrency 10 → 1.8 req/s, 254 tokens/s, TTFT p50 105ms/p95 2.1s, total-latency p50 4.9s; concurrency 30 → 2.7 req/s, 385 tokens/s, TTFT p50 5.2s, total-latency p50 10.5s. Hits vLLM directly, not through the FastAPI SSE relay — see `benchmarks/report.md` for the exact scope. Job cost ~$0.06 (525s at $0.40/hr). | Same benchmark through the FastAPI/SSE relay with Postgres running alongside; A10G/L4-class numbers (the originally targeted GPU class) for comparison against this T4 baseline |
| Pagination | Bounded server-side (`limit` capped) | Cursor pagination if feed depth demands it |

## 5. Infrastructure & Operations

| Metric | Current | Building toward |
|---|---|---|
| Deployment | Local **Docker Compose** (FastAPI + Postgres 16/pgvector) | Dockerized AWS deployment with CI/CD (v1.0) |
| Cloud | Not deployed | AWS (v1.0) |
| CI | **Fixed this pass** — `backend-ci` had been failing on every push since the v0.6/v0.7 commit (2026-08-04) because `app.main` unconditionally imported torch/lightgbm, which the light `requirements.txt` CI installs doesn't include; confirmed by `gh run list` before the fix and by building a clean venv from `requirements.txt` alone after. GitHub Actions: `alembic upgrade head` against a pgvector service + pytest (35 tests: API, RSS, CRUD, dedup, retrieval, retrieval-eval, recommendations incl. the new v0.7-fallback regression test, anomaly) on Python 3.12; backend image built on PRs, published to GHCR on `main` | Extend to the anomaly/LLM path once GPU serving is verified |
| Monitoring | `IngestionRun` history for the core pipeline; Prometheus + Grafana already stood up for the LLM path (`observability/`), scraping vLLM's native `/metrics` — **still not screenshotted against a live vLLM server**: the 2026-08-15 GPU run (see above) didn't wire up `observability/`'s Prometheus scrape against the ephemeral HF Job (port wasn't exposed for external scraping, out of scope for this pass), so this remains open even though vLLM itself is now verified | Extend Prometheus/Grafana to core ingestion/API; verify the LLM dashboard against a real GPU run with `/metrics` actually scraped |
| Database | PostgreSQL 16 (**pgvector image**), SQLAlchemy 2.0 typed ORM, Alembic migrations 0001–0005 (articles → dedup/runs → embeddings → anomaly → recommendation engine) | Decision made: pgvector in-database, no FAISS sidecar until latency demands it |

## 6. Known Data-Quality Issues

- *(Resolved 2026-07-xx, v0.5)* Raw HTML leaking into RSS titles/summaries —
  fixed in `RSSSource.fetch` before it could dilute the embedder or the eval
  set.

## 7. Hardening Done So Far

- **Race-condition-safe inserts** — DB-level unique constraint on URL with
  `IntegrityError` recovery.
- **Per-source error isolation** — one dead source never aborts a run.
- **Bounded pagination** — clients can't request unbounded result sets.
- **Layered cross-source dedup** — URL → DOI → sha256 of normalized title.
- **Scheduler overlap protection** — `max_instances=1` + `coalesce=True` so a
  slow run can't stack concurrent ingestion jobs.
- **v0.7 safe degrade** — the intelligent-ranking route never 500s for lack
  of a trained model; it falls back to the v0.6 heuristic feed.
- **LLM kill switch** — `LLM_ENABLED` disables the anomaly-explain route
  cleanly without touching any other endpoint.

---

## Measurement notes

- Retrieval eval: `cd backend && EMBEDDING_BACKEND=hash python scripts/eval_retrieval.py`,
  measured 2026-08-15 against `tests/fixtures/retrieval_eval.json` (26
  queries, 30-article corpus). Re-run with `EMBEDDING_BACKEND=sentence-transformers`
  for the PubMedBERT number.
- API latency/throughput and article counts are carried forward from
  2026-07-16 (local Docker/curl measurements) and marked as such above — they
  go stale fast; re-run before quoting externally.
- Anomaly-explain benchmark: see `benchmarks/report.md` for full methodology
  and the `stub-*` vs. `vllm-*` labeling convention.
