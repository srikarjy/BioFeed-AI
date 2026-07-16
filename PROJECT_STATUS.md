# BioFeed AI — Project Status & Architecture Decisions

> Last updated: 2026-07-16 · Current branch: `v0.2-content-platform`
> Companion docs: [`BLUEPRINT.md`](./BLUEPRINT.md) (full roadmap & positioning), [`README.md`](./README.md)

---

## 1. What's Done So Far

### v0.1 — Foundation ✅ (commit `dac7862`, hardened in `30aca52`)

- **FastAPI backend** with a clean app layout: `app/main.py`, routers (`articles`, `ingestion`), `config.py`, `database.py`.
- **PostgreSQL + SQLAlchemy 2.0** (typed `Mapped`/`mapped_column` style) with the initial `articles` table.
- **Alembic migrations** (`0001_create_articles_table.py`).
- **Docker + docker-compose** for the backend and Postgres.
- **RSS ingestion** via `feedparser` with per-feed error isolation (one dead feed doesn't abort the run) and a manual trigger endpoint `POST /ingest/run`.
- **Articles API**: list with source filter, bounded pagination (`limit`/`offset` capped), get by id.
- **Test suite** (pytest): API tests, RSS-parsing tests against a fixture feed, CRUD tests. SQLite-backed test fixtures in `conftest.py`.
- **CI**: GitHub Actions workflow running the backend test suite.
- **Hardening pass**: fixed an ingestion race condition (concurrent inserts of the same URL) and bounded article pagination.

### v0.2 — Content Platform 🔨 (in progress on this branch)

Committed (`fd84ea7`):
- **Cross-source dedup fields** on `Article`: `doi`, `external_id`, `authors`, `content_hash` (+ migration `0002_content_platform.py`).
- **Dedup logic** in `crud.create_article`: strongest-to-weakest matching — exact URL → shared DOI → sha256 of normalized title. Title normalization lowercases, strips punctuation, collapses whitespace.
- **`IngestionRun` model** recording every pipeline execution (started/finished timestamps, totals, per-source `detail` JSON) for observability.
- **Dedup tests** (`tests/test_dedup.py`).

Uncommitted working-tree changes (source-abstraction refactor):
- `app/ingestion/base.py` — new `Source` ABC: every content source (RSS, and soon PubMed/bioRxiv) implements `fetch() -> Iterable[ArticleCreate]`.
- `app/ingestion/registry.py` — single declaration point for configured sources; the runner is source-agnostic.
- `app/ingestion/runner.py` — generic `ingest_source` / `run_ingestion` / `run_and_record` (persists an `IngestionRun` row).
- `app/ingestion/rss.py` — refactored into `RSSSource(Source)`; old `ingest_feed` kept as a thin compatibility wrapper.
- `app/routers/ingestion.py` — `POST /ingest/run` now records runs; new `GET /ingest/runs` exposes ingestion history.

---

## 2. Future Tasks

### Finish v0.2 — Content Platform (immediate)
- [ ] Commit the source-abstraction refactor (base/registry/runner) currently in the working tree.
- [ ] **PubMed source** (`PubMedSource(Source)`) — E-utilities API, populate `doi`/`external_id`/`authors`.
- [ ] **bioRxiv source** — bioRxiv API integration.
- [ ] **Scheduled ingestion** — Celery + Redis (or APScheduler as a lighter first step) so runs happen on a cadence, not just via manual trigger.
- [ ] Tests for the new sources and the runner; update CI if new services (Redis) are needed.

### v0.3 — Authentication
- [ ] Sign in with Apple, Google OAuth, JWT access + refresh tokens, user profiles, Keychain storage on iOS.

### v0.4 — Mobile Application
- [ ] SwiftUI feed, article page, search, bookmarking, reading history, offline cache (`/ios` directory).

### v0.5 — Semantic Retrieval 🔨 (reordered; core slice built, uncommitted)

Reordered from "NLP Pipeline". The original plan built embeddings, summaries, and
entity extraction as a milestone that nothing consumed — embedding quality is only
judgeable through retrieval results, so the loop comes first and the rest follows
once there's something to evaluate against.

Built (working tree):
- `app/ml/embeddings.py` — `Embedder` protocol with two backends: PubMedBERT via
  sentence-transformers (Docker/prod) and a dependency-free hashing embedder
  (tests/local dev). Selected by `EMBEDDING_BACKEND`; both emit 768-dim vectors.
- `app/ml/service.py` — `embed_missing` / `embed_articles`; articles embedded from
  title + summary, one vector each.
- **pgvector** storage: `Article.embedding` + HNSW cosine index (migration `0003`).
- `crud.find_similar` — pgvector `<=>` on Postgres, Python scan on SQLite for tests.
- `GET /search?q=` (semantic search) and `GET /articles/{id}/related`.
- Ingestion embeds new articles after each run; `IngestionRun.detail` records the
  embedded count. Embedder failures are isolated like source failures.
- `tests/test_retrieval.py`; CI now runs `alembic upgrade head` on a pgvector image.

Remaining:
- [ ] Summaries and entity extraction (orgs, diseases, genes, drugs, funding events)
      — deferred; they serve v0.8's knowledge graph and UX, not retrieval.
- [ ] Retrieval eval set: hand-labeled query → expected-article pairs to compare
      embedding models against something other than intuition.

### v0.6 — Recommendation Engine (MVP) ⭐ first resume-ready milestone
- [ ] Interaction-signal capture (likes, bookmarks, reading time, scroll, hides, searches).
- [ ] User embedding (from interaction history) → reuse the v0.5 retrieval path → ranking → personalized feed.
- [ ] Content-similarity + search as the cold-start fallback — the permanent day-one
      experience for every new user, not throwaway scaffolding.

### v0.7 — Intelligent Ranking
- [ ] Two-tower retrieval + LightGBM/XGBoost reranker; cross-encoder experiments. Features: user/article embeddings, freshness, publisher, popularity, topic, history.

### v0.8 — Knowledge Graph
- [ ] Entity relationship graph (company → disease → trial → drug → paper), grounded in real ontologies (UMLS, MONDO, ChEMBL), not just NLP co-occurrence.

### v0.9 — Explainable AI
- [ ] "Recommended because…" reasoning surfaced per feed item.

### v1.0 — Production Release
- [ ] Full ingestion + auth + personalized feed + search + bookmarks + notifications, monitoring (Prometheus/Grafana), MLflow, CI/CD, Dockerized AWS deployment.

### Post-1.0
- v1.1 push notifications / daily digest · v1.2 follow companies/researchers/journals · v1.3 audio mode.
- **v2.0 — Biotech Market Signal Module**: ticker tagging via NER + static ticker map, event windows around FDA approvals / PDUFA dates / trial readouts, event-study with cumulative abnormal return vs. XBI benchmark, fed back as a ranking signal.
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
| 13 | **Retrieval loop before personalization** (v0.5 reordered) | Recommendations need user signals, which need the mobile app and real users. Article→article similarity and semantic search need neither, are evaluable today by inspection, and *are* the recommender's retrieval half — v0.6 swaps a user vector in for the query vector. Building the loop first means embedding choices get validated against real neighbor results instead of guessed at. |
| 14 | **pgvector over FAISS** | Postgres, Alembic, and the per-article row already exist; at this corpus size pgvector is fast enough, keeps vectors transactionally consistent with dedup, and removes a service from the architecture. FAISS becomes a justified optimization in v0.7 if ANN latency actually hurts — not before. |
| 15 | **HNSW index over IVFFlat** | IVFFlat needs training data to build a useful index; HNSW builds fine on an empty/small table and stays accurate as the corpus grows — right for a system whose corpus starts at zero. |
| 16 | **Pluggable embedder with a hashing fallback** | Tests and CI stay torch-free and fast (`EMBEDDING_BACKEND=hash`) while prod runs real PubMedBERT. The `Embedder` protocol also makes swapping models a config change, which the v0.5 eval set will need. |
| 17 | **Embeddings nullable + backfilled, embedder failures isolated** | Ingestion must never lose articles or a run record because a model download failed. `embed_missing` backfills on the next run, and `/related` embeds on demand, so the endpoints work regardless. |

---

## 4. Current Repo Layout

```
BLUEPRINT.md              roadmap, positioning, resume framing
PROJECT_STATUS.md         this file
docker-compose.yml        backend + Postgres
backend/
  alembic/                migrations 0001 (articles), 0002 (dedup + runs)
  app/
    main.py, config.py, database.py
    models.py             Article, IngestionRun
    schemas.py            Pydantic models
    crud.py               dedup + get-or-create + run recording + find_similar
    routers/              articles (+ /related), ingestion, search
    ingestion/
      base.py             Source ABC          (uncommitted)
      registry.py         configured sources  (uncommitted)
      runner.py            generic runner      (uncommitted)
      rss.py              RSSSource
      feeds.py            RSS feed list
    ml/
      embeddings.py       Embedder backends   (uncommitted)
      service.py          embed lifecycle     (uncommitted)
  requirements-ml.txt     heavy ML deps, split out to keep CI light
  tests/                  API, RSS, CRUD, dedup, retrieval tests
.github/workflows/        backend CI
```
