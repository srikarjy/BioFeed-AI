# BioFeed AI — Project Status & Architecture Decisions

> Last updated: 2026-08-15 · Current branch: `main`
> Companion docs: [`BLUEPRINT.md`](./BLUEPRINT.md) (full roadmap & positioning), [`README.md`](./README.md), [`METRICS.md`](./METRICS.md)

---

## 1. What's Done So Far

### v0.1 — Foundation ✅

FastAPI + PostgreSQL + SQLAlchemy 2.0 (typed `Mapped` columns) + Alembic
migrations. Docker + docker-compose. RSS ingestion via `feedparser` with
per-feed error isolation. Articles API with bounded pagination. Pytest suite
+ CI. Race-condition-safe inserts (DB-level unique constraint on URL).

### v0.2 — Content Platform ✅

- Cross-source dedup: URL → shared DOI → sha256 of normalized title.
- `IngestionRun` model recording every pipeline execution (totals, per-source
  detail) — `GET /ingest/runs`.
- **`Source` ABC + registry + generic runner** (`app/ingestion/base.py`,
  `registry.py`, `runner.py`) — every content source implements
  `fetch() -> Iterable[ArticleCreate]`; adding a source never touches the
  runner.
- **`PubMedSource`** (`app/ingestion/pubmed.py`) — NCBI E-utilities, rate-limit
  aware (3 req/s without an API key, higher with `NCBI_API_KEY`), populates
  `doi`/`external_id`/`authors`.
- **`BioRxivSource`** (`app/ingestion/biorxiv.py`) — bioRxiv and medRxiv both,
  configurable lookback window.
- **Scheduled ingestion** (`app/scheduler.py`) — APScheduler `AsyncIOScheduler`
  wired into the FastAPI lifespan, runs `run_and_record` on an interval
  (`INGESTION_INTERVAL_MINUTES`, default 60), `max_instances=1` +
  `coalesce=True` so overlapping runs can't stack up.

### v0.5 — Semantic Retrieval ✅ (reordered ahead of the original NLP-pipeline plan)

Reordered because embedding quality is only judgeable through retrieval
results — the retrieval loop had to exist before summaries/entity extraction
were worth building.

- `app/ml/embeddings.py` — `Embedder` protocol: PubMedBERT via
  sentence-transformers (Docker/prod) or a dependency-free hashing embedder
  (tests/local dev), selected by `EMBEDDING_BACKEND`. Both emit 768-dim
  vectors.
- **pgvector** storage (`Article.embedding`) + HNSW cosine index.
- `crud.find_similar` — pgvector `<=>` on Postgres, Python scan on SQLite
  for tests.
- `GET /search?q=` (semantic search), `GET /articles/{id}/related`.
- Ingestion embeds new articles after each run; failures are isolated like
  source failures so a model hiccup never loses an article or a run record.
- HTML stripped from RSS titles/summaries before embedding (raw markup was
  diluting vectors with tag/URL tokens).
- **Retrieval eval set** (`tests/fixtures/retrieval_eval.json`,
  `scripts/eval_retrieval.py`) — 26 hand-labeled queries against a
  30-article, multi-topic corpus, deliberately spanning distinct biotech
  subtopics (gene therapy, immunotherapy, GLP-1s, biosimilars, microbiome,
  AI drug discovery, M&A/funding news) so lexical-overlap alone can't
  trivially pass. Reports Recall@5, Recall@10, NDCG@10; a floor test
  (`tests/test_retrieval_eval.py`, Recall@10 ≥ 0.9) runs in CI. See
  `METRICS.md` for current numbers and the honest caveat about which
  embedding backend they're measured against.

### v0.6 — Recommendation Engine (MVP) ✅

- `User`, `UserInteraction`, `UserEmbedding` models; `POST /users`,
  `POST /users/{id}/interactions`, `GET /users/{id}/feed`.
- Interaction-weighted user embedding: read time (capped, linear up to 5 min),
  bookmark, like all contribute; hides are excluded from the embedding and
  filtered out of the feed.
- `crud.get_personalized_feed` — warm start via cosine similarity against the
  user embedding; cold start (no interactions yet) falls back to recent
  articles, which is the permanent day-one experience, not throwaway
  scaffolding.
- Per-item "recommended because…" reason string (`crud._generate_reason`).

### v0.7 — Intelligent Ranking ✅ (code + training pipeline; not yet trained on real users)

- `app/ml/two_tower.py` — two-tower retrieval model (user tower, item tower,
  contrastive training on interaction pairs).
- `app/ml/reranker.py` — LightGBM reranker over user/article/affinity
  features (topic affinity, source affinity, freshness, user stats).
- `scripts/train_v07.py` — training pipeline for both models from real
  interaction data.
- `app/ml/recommender_v07.py` / `GET /users/{id}/feed` (v0.7 route,
  `recommendations_v07.py`) — loads trained models from `MODEL_DIR` if
  present, retrieval → rerank → reasons; **falls back cleanly to v0.6 logic**
  when no trained checkpoint exists, which is the honest current state:
  there are no real users yet, so nothing has been trained. The pipeline is
  real and tested against synthetic interaction data
  (`tests/test_recommendations.py`); training against real usage is gated on
  v0.3/v0.4 (auth + mobile app) actually producing interaction logs.

### Anomaly Detection + Self-Hosted LLM Explanations ✅ (additive, backend-only, not on the original roadmap)

Built to demonstrate self-hosted LLM serving (vLLM, continuous batching,
SSE streaming, Prometheus/Grafana observability) without waiting for v2.0's
market-signal module. Fully isolated in `app/anomaly/`, `app/llm/`,
`llm_serving/`, `scripts/`, `observability/` — touches no existing route.

- `app/anomaly/detector.py` — cross-source burst detector: reuses the v0.5
  pgvector embeddings and `crud.find_similar` to flag an article when other
  sources publish closely matching coverage within a 48h window.
- `app/llm/` — vLLM OpenAI-compatible client, prompt builder,
  `GET /internal/anomaly-explain/{event_id}` (SSE), `LLM_ENABLED` kill switch.
- `llm_serving/` — GPU-instance deployment scripts (download, serve,
  startup) for a self-hosted quantized model behind vLLM's OpenAI-compatible
  server.
- **Honest caveat** (see [`ANOMALY_EXPLAIN_LLM.md`](./ANOMALY_EXPLAIN_LLM.md)):
  the route and SSE wiring are tested against a local CPU stand-in server
  (`scripts/stub_vllm_server.py`); the vLLM scripts and Grafana dashboard
  have **not** been run against a real GPU — no rented instance was
  available while building this. Benchmark numbers in `benchmarks/report.md`
  validate the wiring, not serving performance, and are labeled as such.

---

## 2. Future Tasks

### v0.3 — Authentication
- [ ] Sign in with Apple, Google OAuth, JWT access + refresh tokens, user
      profiles, Keychain storage on iOS.

### v0.4 — Mobile Application
- [ ] SwiftUI feed, article page, search, bookmarking, reading history,
      offline cache (`/ios` directory — doesn't exist yet).

### v0.5 remaining (deferred, not blocking)
- [ ] Summaries and entity extraction (orgs, diseases, genes, drugs, funding
      events) — serves v0.8's knowledge graph and UX, not retrieval; still
      deferred on purpose.
- [ ] Run `scripts/eval_retrieval.py` with `EMBEDDING_BACKEND=sentence-transformers`
      (real PubMedBERT) and record the numbers next to the hashing-backend
      baseline in `METRICS.md` — not yet done in this environment because
      `sentence-transformers`/torch aren't installed outside Docker.

### v0.7 remaining
- [ ] Run `scripts/train_v07.py` against real interaction data once v0.3/v0.4
      produce actual users; until then the v0.7 route always falls back to
      v0.6 logic in practice (correctly — there's nothing to score against).
- [ ] Cross-encoder reranking experiment (mentioned in `BLUEPRINT.md`, not
      started).

### Anomaly / LLM remaining
- [ ] Run `llm_serving/` against a real GPU instance and re-verify the
      Grafana dashboard and `benchmarks/report.md` numbers against it.

### v0.8 — Knowledge Graph
- [ ] Entity relationship graph (company → disease → trial → drug → paper),
      grounded in real ontologies (UMLS, MONDO, ChEMBL), not just NLP
      co-occurrence.

### v0.9 — Explainable AI
- [ ] Richer "recommended because…" reasoning surfaced per feed item (the
      v0.6 version is a single heuristic string, not a modeled explanation).

### v1.0 — Production Release
- [ ] Full ingestion + auth + personalized feed + search + bookmarks +
      notifications, monitoring (Prometheus/Grafana already exist for the
      LLM path — extend to the core service), MLflow, CI/CD, Dockerized AWS
      deployment.

### Post-1.0
- v1.1 push notifications / daily digest · v1.2 follow
  companies/researchers/journals · v1.3 audio mode.
- **v2.0 — Biotech Market Signal Module**: ticker tagging via NER + static
  ticker map, event windows around FDA approvals / PDUFA dates / trial
  readouts, event-study with cumulative abnormal return vs. XBI benchmark,
  fed back as a ranking signal.
- v2.1 multi-agent intelligence · v3.0 enterprise team feeds.

---

## 3. Architecture Decisions Taken

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **FastAPI + PostgreSQL + SQLAlchemy 2.0 (typed ORM) + Alembic** | Standard, production-grade Python stack; typed `Mapped` columns catch schema mistakes early; migrations from day one so the schema can evolve safely. |
| 2 | **Docker + docker-compose from v0.1** | Reproducible dev environment and a straight path to cloud deployment later. |
| 3 | **`Source` abstraction (ABC) + registry + generic runner** for ingestion | Adding PubMed/bioRxiv/FDA never touches the runner — each source owns its fetching/parsing and yields normalized `ArticleCreate` items. The registry is the single place declaring what BioFeed pulls from. |
| 4 | **Per-source error isolation in the runner** | One dead or unreachable feed must not abort the whole ingestion run; errors are captured per source and surfaced in the API response and run history. |
| 5 | **Layered cross-source dedup: URL → DOI → normalized-title hash** | The same paper surfaces from multiple sources under different URLs. Matching goes strongest-to-weakest signal; the sha256-of-normalized-title is the last resort when no DOI is shared. |
| 6 | **DB-level unique constraint on URL as a race-condition safety net** | Two concurrent runs can both pass the existence check before either commits; the loser's `IntegrityError` is rolled back and the winner's row is returned, keeping the session usable. Application-level checks alone aren't enough. |
| 7 | **`IngestionRun` table for pipeline observability** | Both scheduled and manual runs persist a row (totals + per-source JSON detail), exposed via `GET /ingest/runs` — ingestion freshness and failures are queryable, not just logged. |
| 8 | **Authors stored as free text; `doi`/`external_id` nullable** | Author-list shape varies wildly across PubMed/bioRxiv/RSS, and most RSS news items carry no DOI — keep provenance fields flexible rather than over-normalizing early. |
| 9 | **Bounded pagination on the articles API** | `limit` is capped server-side so a client can't request unbounded result sets. |
| 10 | **SQLite-backed test fixtures, real Postgres in runtime** | Fast, dependency-free test suite in CI while keeping Postgres semantics (unique constraints, timezone-aware datetimes) in production code paths. |
| 11 | **Backward-compatible refactors** (`ingest_feed` kept as a thin wrapper) | Existing tests and callers keep working while the ingestion layer is generalized. |
| 12 | **Biotech as the vertical, general recsys as the architecture** | Positioning decision (see `BLUEPRINT.md`): the system is a general personalized content-ranking platform applied to a data-rich niche, extended with a text-to-market signal — differentiating it from generic recsys clones. |
| 13 | **Retrieval loop before personalization** (v0.5 reordered) | Recommendations need user signals, which need the mobile app and real users. Article→article similarity and semantic search need neither, are evaluable today by inspection, and *are* the recommender's retrieval half — v0.6 swaps a user vector in for the query vector. |
| 14 | **pgvector over FAISS** | Postgres, Alembic, and the per-article row already exist; at this corpus size pgvector is fast enough, keeps vectors transactionally consistent with dedup, and removes a service from the architecture. FAISS becomes a justified optimization in v0.7 if ANN latency actually hurts — not before (it doesn't yet). |
| 15 | **HNSW index over IVFFlat** | IVFFlat needs training data to build a useful index; HNSW builds fine on an empty/small table and stays accurate as the corpus grows — right for a system whose corpus starts at zero. |
| 16 | **Pluggable embedder with a hashing fallback** | Tests and CI stay torch-free and fast (`EMBEDDING_BACKEND=hash`) while prod runs real PubMedBERT. The `Embedder` protocol also makes swapping models a config change. |
| 17 | **Embeddings nullable + backfilled, embedder failures isolated** | Ingestion must never lose articles or a run record because a model download failed. `embed_missing` backfills on the next run, and `/related` embeds on demand. |
| 18 | **Interaction-weighted user embedding over a learned user model (v0.6)** | A weighted average of interacted-article embeddings is enough to validate the retrieval→personalization loop end to end with zero training data required; the two-tower model (v0.7) is the upgrade path once there's enough interaction volume to train one. |
| 19 | **v0.7 falls back to v0.6, not to an error** | `recommender_v07.py` checks for trained checkpoints and degrades to the v0.6 heuristic feed when none exist, so the "intelligent ranking" route is always safe to call even with zero trained models — honest about the current state (no real users yet) instead of hiding it behind a required model file. |
| 20 | **Anomaly/LLM feature built additive and isolated from the core feed** | Demonstrates self-hosted LLM serving skills (vLLM, SSE, observability) on its own timeline, without coupling the resume-critical recommendation path to GPU availability. `LLM_ENABLED` is a real kill switch, not decoration. |
| 21 | **Retrieval eval set is a fixed synthetic corpus, not the live DB** | The production DB is one ingestion run's worth of articles (see `METRICS.md`) — too small and too RSS-skewed to be a stable eval target. A fixed, deliberately multi-topic 30-article corpus with hand-labeled queries gives a reproducible, CI-enforceable signal independent of what's currently ingested. |

---

## 4. Current Repo Layout

```
BLUEPRINT.md              roadmap, positioning, resume framing
PROJECT_STATUS.md         this file
METRICS.md                measured vs. target numbers
ANOMALY_EXPLAIN_LLM.md    anomaly detection + self-hosted vLLM explanation feature
docker-compose.yml        backend + Postgres
backend/
  alembic/                migrations 0001 (articles) → 0005+ (users, interactions, embeddings, anomaly)
  app/
    main.py, config.py, database.py, scheduler.py
    models.py             Article, IngestionRun, User, UserInteraction, UserEmbedding
    schemas.py            Pydantic models
    crud.py               dedup + get-or-create + run recording + find_similar + recommendation logic
    routers/              articles (+ /related), ingestion, search, recommendations (v0.6), recommendations_v07
    ingestion/
      base.py, registry.py, runner.py    Source ABC + registry + generic runner
      rss.py, pubmed.py, biorxiv.py      RSS, PubMed E-utilities, bioRxiv/medRxiv sources
      feeds.py            RSS feed list
    ml/
      embeddings.py       Embedder backends (hash / PubMedBERT)
      service.py          embed lifecycle
      two_tower.py        v0.7 two-tower retrieval model
      reranker.py         v0.7 LightGBM reranker + feature extraction
      recommender_v07.py  v0.7 pipeline with v0.6 fallback
    anomaly/              AnomalyEvent model, cross-source burst detector, internal routes
    llm/                  vLLM client, prompt builder, SSE route
  scripts/
    eval_retrieval.py     v0.5 retrieval quality eval (Recall@k, NDCG@k)
    train_v07.py          two-tower + reranker training pipeline
  requirements-ml.txt     heavy ML deps (sentence-transformers/torch), split out to keep CI light
  tests/                  API, RSS, CRUD, dedup, retrieval, retrieval-eval, recommendations, anomaly tests
llm_serving/              GPU-instance vLLM deployment scripts (see ANOMALY_EXPLAIN_LLM.md)
observability/            Prometheus + Grafana, scrapes a remote vLLM /metrics
benchmarks/               Load-test reports for the anomaly-explain endpoint
.github/workflows/        backend CI
```
