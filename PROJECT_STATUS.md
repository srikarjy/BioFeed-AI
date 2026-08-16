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

### v0.7 — Intelligent Ranking ✅ (pipeline built, fixed, and verified end-to-end on synthetic data; not yet trained on real users)

- `app/ml/two_tower.py` — two-tower retrieval model (user tower, item tower,
  contrastive training on interaction pairs).
- `app/ml/reranker.py` — LightGBM reranker over user/article/affinity
  features (topic affinity, source affinity, freshness, user stats).
- `scripts/train_v07.py` — training pipeline for both models from real
  interaction data.
- `app/ml/recommender_v07.py` / `GET /users/{id}/feed` (v0.7 route,
  `recommendations_v07.py`) — loads trained models from `MODEL_DIR` if
  present, retrieval → rerank → reasons; **falls back cleanly to v0.6 logic**
  when no trained checkpoint exists.

**This pipeline was written but had never actually been run before this
pass — running it end to end surfaced four real bugs, now fixed:**

1. **CI was silently broken.** `app.main` imports `recommendations_v07`,
   which imported `torch` and `lightgbm` unconditionally at module level.
   `requirements.txt` (the light file CI installs, deliberately excluding
   `requirements-ml.txt`'s heavy deps) has neither — so `backend-ci` has
   been red on every push since the v0.6/v0.7 commit landed. Fixed by making
   those imports lazy (inside `V07Recommender._load_models` /
   `get_recommendations`, not at module import time), with an `ImportError`
   fallback to the v0.6 feed. Verified by building a clean venv from
   `requirements.txt` alone and confirming `from app.main import app` and
   the full test suite both succeed without torch/lightgbm installed.
2. **`MODEL_DIR` defaulted to a hardcoded personal absolute path**
   (`/Users/srikarjy/...`) — broken on any other machine. Now resolved
   relative to the module file.
3. **A pgvector truthiness bug** in `two_tower.create_training_data`:
   `if article and article.embedding:` raises `ValueError: truth value of
   an array...` because a loaded `Vector` column comes back as a numpy
   array, not a plain list. Fixed to `is not None` checks.
4. **`lightgbm` 4.x API break**: `lgb.train(..., early_stopping_rounds=...,
   verbose_eval=...)` — both kwargs were removed in favor of `callbacks=
   [lgb.early_stopping(...), lgb.log_evaluation(...)]`. Fixed.

**Environment note (not a code bug, but real and worth knowing):** on the
macOS dev machine, torch and lightgbm both bundle their own OpenMP runtime;
loading both in one process and then calling into lightgbm segfaults
(SIGSEGV) unless `OMP_NUM_THREADS=1`. Reproduced directly (`python -c
"import torch, lightgbm as lgb; <train>"` crashes;
`OMP_NUM_THREADS=1 python -c "..."` doesn't). Set defensively in the
Dockerfile; not yet confirmed whether the Linux/slim image needs it too (no
GPU/Docker runtime available to verify here — see Future Tasks).

**Verified end to end** (`scripts/seed_v07_synthetic.py` — 139 synthetic
articles across 10 topic clusters, 25 synthetic users with topic-affinity
interactions, since there are zero real users; the corpus is disjoint from
the retrieval eval corpus and exists only for this verification):
`train_v07.py` trained both models (two-tower: 344 positive pairs, 15
epochs; LightGBM reranker: 688 samples, validation AUC 0.91, with
`user_item_cosine` dominating feature importance as expected). Loading the
resulting checkpoints via `MODEL_DIR` and calling `get_enhanced_feed`
returns `cold_start=False` with real two-tower + reranker scores and
on-topic recommendations — the full pipeline runs, not just its unit tests.
Training against *real* usage is still gated on v0.3/v0.4 (auth + mobile
app) actually producing interaction logs; synthetic verification confirms
the pipeline works, not that the model is good on real behavior.

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
- **Honest caveat, updated 2026-08-16**: the route and SSE wiring were
  first tested against a local CPU stand-in server
  (`scripts/stub_vllm_server.py`); `stub-cpu-local` benchmark numbers
  validate that wiring, not serving performance.
  **Now verified for real, twice**: (1) vLLM itself on a real GPU
  (`vllm-t4-direct` — `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` via
  `vllm/vllm-openai` on a Hugging Face Jobs T4, 90/90 real streaming
  requests, TTFT p50 105ms at concurrency 10, ~$0.06); (2) **the actual
  product path end to end** (`vllm-t4-full-relay` — real FastAPI backend
  and real vLLM in the same GPU job, `GET /internal/anomaly-explain/{id}`
  → `app/llm/client.py` → real vLLM → SSE back to the caller, no
  stand-ins anywhere; 15/15 requests, TTFT p50 147ms; ~$0.04). See
  `benchmarks/report.md` for full numbers and methodology on both.
  Getting the full relay running on GPU required a real fix first: a bare
  SQLite `DATABASE_URL` crashes a real `uvicorn` process (threadpool +
  single-threaded sqlite3 connection) unless the engine is configured with
  `check_same_thread=False` + `StaticPool` — `app/database.py` now does
  this automatically for any `sqlite://` URL; production still always uses
  Postgres. **Grafana dashboard**: checked against real `/metrics` output —
  7 of 10 panel metric names matched immediately, 3 didn't
  (`gpu_cache_usage_perc`, `cpu_cache_usage_perc`,
  `time_per_output_token_seconds`), a real, previously-unknown gap. **Fixed
  the same day**: `kv_cache_usage_perc` (renamed), prefix-cache hit rate
  (replaces `cpu_cache_usage_perc`, which no longer exists in this vLLM
  version — no direct replacement), `inter_token_latency_seconds`
  (renamed) — all confirmed present in real `/metrics` output. Still not
  screenshotted against a live Grafana instance.
  **Also verified the originally-targeted GPU class**: a matched
  Hugging Face Jobs `l4x1` (1x L4, 24GB) run at concurrency 10 measured
  TTFT p50 57.3ms direct / 89.1ms full-relay vs. the T4's 104.8ms/147.4ms
  — roughly halves TTFT and isolates a real ~32ms FastAPI-hop cost, both
  numbers `llm_serving/serve.sh`'s comments assumed but hadn't measured.
  ~$0.19 total GPU spend across all three jobs this pass.

### v0.3 — Authentication ✅ (backend; no iOS client to actually run it against)

`app/auth/`, additive and isolated the same way the anomaly/LLM feature is.

- JWT access (30 min) + refresh (30 day) tokens (`app/auth/jwt.py`), HS256,
  typed claims so a refresh token can't be replayed as an access token.
- Pluggable identity verification (`app/auth/providers.py`):
  `AppleIdentityVerifier`/`GoogleIdentityVerifier` fetch the real JWKS
  (`appleid.apple.com`/`googleapis.com`) and verify RS256 signature +
  iss/aud — real code, but **unverified beyond that**: exercising it needs
  a registered Apple Services ID / Google OAuth client and a real identity
  token minted by their SDKs, neither of which exist without v0.4's iOS
  app. `FakeIdentityVerifier` (default, `AUTH_PROVIDER=fake`) trusts a JSON
  blob instead, making the rest of the flow — issuance, refresh, route
  enforcement — fully testable without either. Nothing defaults to `real`,
  so a misconfigured deployment fails closed.
- `POST /auth/apple`, `POST /auth/google`, `POST /auth/refresh`, `GET /auth/me`.
- `require_self` dependency actually wired into every `/users/{id}/...`
  route that touches a specific user's private data (interactions,
  embedding, feed, the new v0.9 explain endpoint) — 401 unauthenticated,
  403 authenticated-as-someone-else. `POST /users` (raw create/lookup) and
  `GET /users/{id}` (profile) stay open by design; see the comment in
  `app/routers/recommendations.py`.
- 17 tests (`tests/test_auth.py`) covering issuance, refresh, wrong-token-type
  rejection, and enforcement on the protected routes.

### v0.8 — Knowledge Graph ✅

`app/kg/`, wired into the ingestion pipeline.

- **Entities grounded in real ontology identifiers** — each looked up live
  against the source ontology's public API during development: MONDO/HPO
  for disease, ChEMBL for drug, HGNC for gene (`app/kg/gazetteer.json`,
  see its `_provenance` field for the exact lookup commands to
  re-verify). Substituted HGNC/HPO for UMLS since UMLS needs a license/API
  key this project doesn't have — same "real, public, no key" bar, not a
  downgrade in kind. Company entities are self-grounding (real names +
  public tickers).
- **Dictionary/gazetteer extraction** (`app/kg/extractor.py`) — word-boundary,
  case-insensitive matching over the gazetteer, longest-alias-first so
  overlapping surface forms don't double-count. Deliberately not a trained
  NER model (e.g. scispaCy): that would catch entities outside the
  gazetteer, but "matches a small curated real-ontology list" is a more
  honest and inspectable v0.8 slice than a model whose precision/recall on
  this corpus was never measured — see the module docstring.
- **Co-occurrence relations** (`app/kg/service.py`) — two entities
  co-mentioned in one article get a directed edge, predicate chosen by
  entity-type pair (company+drug → `develops`, drug+disease → `targets`,
  etc.), falling back to `co_mentioned_with`. Explicitly a same-article
  co-occurrence heuristic, not a verified causal or biological claim — the
  model docstring says so directly.
- **Trial entities grounded dynamically** (`app/kg/trials.py`, added
  2026-08-16) — unlike the static gazetteer, NCT ids are extracted via
  regex from article text and looked up live against ClinicalTrials.gov's
  public API per article, only for NCT ids not already known locally (no
  repeat network call for a trial already seen). A lookup failure skips
  that NCT id rather than fabricating a placeholder entity or failing the
  whole article's extraction — same isolation discipline as source/embedder
  failures elsewhere. `fetch_trial_title` is injectable so tests don't
  depend on network access; separately verified against the real API
  outside pytest (`fetch_trial_title("NCT04173585")` → a real title).
- `GET /kg/entities`, `GET /kg/entities/{id}`, `GET /kg/entities/{id}/relations`,
  `GET /kg/articles/{id}/entities`, `POST /kg/extract` (manual backfill).
- Wired into `run_and_record` (same isolation pattern as embedding: an
  extraction failure can't lose the ingestion run record) and recorded in
  `IngestionRun.detail.entities_extracted`.
- `Article.kg_extracted_at` tracks processed state so an article matching
  zero gazetteer entities doesn't get rescanned by every backfill forever
  — a real gap the first draft of `article_ids_missing_extraction` had
  (using "no EntityMention row" as the signal), fixed before it shipped.
- Gazetteer expanded 30 → 54 entities (2026-08-16: +8 disease, +4 gene, +7
  company, +5 drug — the drug entries added once ChEMBL's API, briefly
  down earlier the same day, recovered; see `gazetteer.json`'s
  `_provenance`).
- 19 tests (`tests/test_kg.py`, was 13).

### v0.9 signals fix (2026-08-16)

Beyond the initial v0.9 explanation feature, closed the documented
`compute_user_affinities` gap this same pass: `topic_affinity` is now
keyed by real KG entity id instead of proxying article source, and a new
`topic_affinity_score` sums it per candidate article — see "v0.9
remaining" for the detail and the test that would have caught the old
behavior.

### v0.9 — Explainable AI ✅

`app/ml/explain.py` + `GET /users/{id}/articles/{id}/explain`.

- Multi-signal structured explanation (`ExplanationSignal(label, detail,
  weight)`, sorted by weight) instead of the v0.6 feed's single heuristic
  string: nearest-interaction cosine similarity, topic/source affinity
  (the *same* signal the v0.7 reranker trains on — this explains the
  model, it isn't a separate story invented for display), freshness,
  popularity, source quality, and — the one signal that couldn't exist
  before this pass — v0.8 knowledge-graph entities shared between the
  candidate article and the user's recently-interacted articles.
  `FeedItem.reason` (every `/feed` item) stays the cheap one-liner for list
  rendering; `/explain` is the tap-through detail view.
- **Refactor along the way**: pulled `compute_source_quality`,
  `compute_freshness_days`, `compute_item_popularity`,
  `compute_user_affinities`, `get_user_stats` out of `reranker.py` into a
  new dependency-free `app/ml/signals.py`, so `explain.py` (imported by the
  always-loaded `/users` router) doesn't pull in numpy/lightgbm — the exact
  import-discipline that fixed the v0.7 CI bug (decision #22), now applied
  proactively instead of after breaking something.
- 8 tests (`tests/test_explain.py`).

**77 tests passing overall (was 35 before this pass)**, including a clean
venv built from `requirements.txt` alone (no torch/lightgbm) to confirm
none of this reintroduced the unconditional-heavy-import bug.

---

## 2. Future Tasks

### v0.4 — Mobile Application
- [ ] SwiftUI feed, article page, search, bookmarking, reading history,
      offline cache (`/ios` directory — doesn't exist yet). Also the only
      way to actually exercise the real Apple/Google identity verifiers
      end to end (needs a real device + registered app).

### v0.5 remaining (deferred, not blocking)
- [ ] Summaries and entity extraction (orgs, diseases, genes, drugs, funding
      events) — serves v0.8's knowledge graph and UX, not retrieval; still
      deferred on purpose.
- [x] ~~Run `scripts/eval_retrieval.py` with `EMBEDDING_BACKEND=sentence-transformers`~~
      — done: Recall@5 1.000, Recall@10 1.000, NDCG@10 0.989 (vs. 0.962/1.000/0.959
      for the hashing baseline). See `METRICS.md`.

### v0.7 remaining
- [x] ~~Verify the training pipeline actually runs~~ — done this pass on
      synthetic data; surfaced and fixed 4 real bugs (see above).
- [ ] Run `scripts/train_v07.py` against real interaction data once v0.3/v0.4
      produce actual users — synthetic verification proves the pipeline
      works, not that the model is good on real behavior.
- [ ] Confirm `OMP_NUM_THREADS=1` is actually needed on the Linux/Docker
      image (only reproduced on macOS so far; set defensively either way).
- [ ] Cross-encoder reranking experiment (mentioned in `BLUEPRINT.md`, not
      started).

### Anomaly / LLM remaining
- [x] ~~Run vLLM against a real GPU~~ — done 2026-08-15 (`vllm-t4-direct`).
- [x] ~~Benchmark through the actual FastAPI/SSE relay on GPU~~ — done
      2026-08-16 (`vllm-t4-full-relay`): real backend + real vLLM, same GPU
      job, no stand-ins. See `benchmarks/report.md`.
- [x] ~~Check the Grafana dashboard's metric names against real vLLM
      `/metrics`~~ — done: 3 of 10 don't match this vLLM version
      (`gpu_cache_usage_perc`, `cpu_cache_usage_perc`,
      `time_per_output_token_seconds`), now documented.
- [x] ~~Fix the 3 stale metric names~~ — done 2026-08-16: captured real
      `/metrics` output on a live vLLM server and grepped the actual names.
      `gpu_cache_usage_perc` → `kv_cache_usage_perc` (renamed);
      `cpu_cache_usage_perc` → **removed, no replacement** (this vLLM
      version dropped CPU/swap KV cache tracking; the panel now shows
      prefix-cache hit rate instead, the closest real signal);
      `time_per_output_token_seconds` → `inter_token_latency_seconds`
      (renamed). Applied to
      `observability/grafana/dashboards/vllm-anomaly-explain.json`.
      **Still not done**: pointing the dashboard at a live Prometheus+vLLM
      pair and visually confirming it renders — this fix is confirmed by
      metric-name presence in real `/metrics` output, not a screenshot.
- [x] ~~A10G/L4-class numbers~~ — done 2026-08-16 on `l4x1` (1x L4, 24GB):
      TTFT p50 57.3ms vs. the T4's 104.8ms at the same concurrency (10) —
      roughly halves TTFT and cuts p95 tail ~5x, the number
      `llm_serving/serve.sh`'s engine-arg comments assumed but never
      measured. See `benchmarks/report.md` (`vllm-l4-direct`).
- [x] ~~Matched-concurrency direct vs. full-relay comparison~~ — done in
      the same L4 job, both at concurrency 10: TTFT p50 57.3ms (direct) →
      89.1ms (full relay), isolating a real ~32ms FastAPI-hop cost
      (`vllm-l4-full-relay`).
- [x] ~~A from-scratch GPU-host run with real Postgres in the loop~~ —
      done 2026-08-16 (`vllm-t4-postgres-full-relay`): installed
      PostgreSQL + built pgvector from source in a Hugging Face Jobs T4
      container, ran the real `alembic upgrade head` (migrations 0001–0006,
      the same chain CI runs, not `Base.metadata.create_all`), then the
      real pipeline end to end — ingestion → `embed_missing` → KG
      extraction → anomaly detection, all against real Postgres — through
      to real vLLM. Also separately confirmed `GET /search` against the
      real pgvector HNSW index (not the SQLite Python-scan fallback)
      returned correct results with real cosine similarity. TTFT p50
      186ms — higher than the SQLite full-relay run's 147ms, a real and
      now-quantified Postgres overhead rather than an assumed one. Job
      cost ~$0.05 (412s). See `benchmarks/report.md`.

### v0.8 remaining
- [x] ~~Expand the gazetteer~~ — 30 → 54 entities across two passes
      (2026-08-16): +8 disease (MONDO), +4 gene (HGNC), +7 company
      first (ChEMBL's API was returning HTTP 500 for both
      `/molecule/search` and `/molecule.json` at the time, confirmed
      transient/upstream — previously-working queries also failed that
      day); +5 drug entries (lecanemab, donanemab, elranatamab,
      trastuzumab, adalimumab) once ChEMBL recovered later the same day.
- [x] ~~Ground `trial` entities (NCT ids)~~ — done: `app/kg/trials.py`
      extracts NCT ids via regex and looks up the real title from
      ClinicalTrials.gov's public API live, per-article, only for NCT ids
      not already known locally (no repeat network calls for a trial
      already seen). Unlike the static gazetteer, this is dynamic — new
      trials get grounded automatically as they're mentioned. A lookup
      failure skips that NCT id rather than creating a placeholder entity.
      6 new tests (`tests/test_kg.py`), all against an injected fake
      fetcher so CI doesn't depend on network access; the real fetcher was
      separately verified against the live API
      (`fetch_trial_title("NCT04173585")` → a real trial title).
- [ ] Still only ~54 hand-curated + dynamically-grounded trial entities —
      a trained NER model as a second extraction pass would catch entities
      outside the gazetteer entirely; still deferred, see extractor.py.

### v0.9 remaining
- [ ] Surface the structured explanation in a UI once v0.4 exists — right
      now it's a real, tested JSON endpoint with no client consuming it.
- [x] ~~Replace the source-affinity-as-topic-affinity proxy with a real
      per-article topic derived from v0.8 KG entities~~ — done 2026-08-16:
      `compute_user_affinities` now returns `topic_affinity` keyed by KG
      entity id (weighted by how often each entity appears across the
      user's positively-interacted articles), and a new
      `topic_affinity_score(db, article, topic_affinity)` sums those
      weights for a candidate article's actual entities. Wired into both
      the v0.7 reranker's `extract_features` (replacing
      `topic_affinity.get(article.source, 0.0)`) and the v0.9 explanation
      builder (new `topic_affinity` signal). Degrades to `{}`/`0.0`
      cleanly when KG extraction hasn't run on an article yet — no crash,
      just no signal. 4 new tests (`tests/test_signals.py`), including one
      that would have failed under the old source-proxy behavior (two
      articles from *different* sources both mentioning Moderna now score
      real affinity overlap; the old signal could never see that since
      different source strings never overlapped).

### v1.0 — Production Release
- [ ] Full ingestion + auth + personalized feed + search + bookmarks +
      notifications, monitoring (Prometheus/Grafana already exist for the
      LLM path — extend to the core service), MLflow, CI/CD, Dockerized AWS
      deployment. Needs real AWS credentials/spend — out of scope until
      explicitly requested.

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
| 22 | **Heavy ML imports (torch, lightgbm) are lazy in `recommender_v07.py`, not module-level** | Discovered this pass: `app.main` imported them unconditionally through the v0.7 router, so `backend-ci` (which only installs the light `requirements.txt`) had been failing on every push since v0.6/v0.7 landed. Lazy imports inside `_load_models`/`get_recommendations`, with an `ImportError`-caught fallback to the v0.6 feed, keep `app.main` importable — and the v0.7 route serving a real (if simpler) feed — with or without the ML deps installed. |
| 23 | **`lightgbm` imported before `torch` in the v0.7 load path** | Both bundle their own OpenMP runtime; on macOS, importing torch first and then calling into lightgbm segfaults. `_load_models` imports `lightgbm` first (even though it's only used later in the reranker branch) to establish a safe load order for the rest of the process's lifetime; `OMP_NUM_THREADS=1` is still required on top of this (see Dockerfile) — the reorder alone wasn't sufficient. |
| 24 | **Synthetic seed script (`scripts/seed_v07_synthetic.py`) to verify the v0.7 pipeline, not to fake a trained model** | With zero real users, `train_v07.py` had literally never been executed — it was untested code. A disjoint, deliberately non-real, topic-clustered synthetic corpus + users lets the training/inference pipeline be run and verified end to end (it trained, it loaded, it scored). Explicitly not committed as a shipped model or represented as trained on real behavior — see the caveat in `METRICS.md`. |
| 25 | **Model checkpoints (`*.pt`, `models/*.txt`) stay out of git (`.gitignore`)** | Trained artifacts are reproducible from `train_v07.py`/`seed_v07_synthetic.py` and change size/content on every run; committing them would bloat the repo for no benefit over documenting how to regenerate them. |
| 26 | **Pluggable identity verification, `fake` as the default** | Real Apple/Google verification needs a registered app and real device-minted tokens that don't exist without v0.4. A fake provider that trusts a JSON blob keeps the entire auth flow — issuance, refresh, route enforcement — testable now; nothing defaults to `real`, so a misconfigured deployment fails closed instead of silently trusting anything. |
| 27 | **`require_self` enforced on existing v0.6/v0.7 routes immediately, not deferred to v0.4** | Auth infrastructure that issues tokens nothing checks is the same "built but never wired up" pattern that caused the v0.7 CI bug — worth avoiding proactively. `POST /users` and `GET /users/{id}` stay open (predates auth / public profile) so the change is additive enforcement, not a breaking redesign. |
| 28 | **HGNC/HPO in place of UMLS for gene/phenotype grounding** | UMLS requires a license and API key this project doesn't have. HGNC (genes) and HPO (phenotypes) are real, public, no-key ontologies with the same "genuinely grounded" property BLUEPRINT.md asks for — a substitution within the spirit of the requirement, not a downgrade to NLP co-occurrence. |
| 29 | **Dictionary/gazetteer entity extraction over a trained NER model** | A trained model (e.g. scispaCy) would generalize beyond the gazetteer, but adds a dependency/download and its precision/recall on this corpus would be unmeasured. A small, hand-verified gazetteer is fully inspectable — every entity's grounding can be checked with one curl command (see gazetteer.json's `_provenance`) — which is a more honest v0.8 slice to ship first. |
| 30 | **`Article.kg_extracted_at` instead of "has an EntityMention row" as the backfill signal** | An article can legitimately match zero gazetteer entities. Keying the backfill query off mention-row existence would rescan every zero-match article on every run forever; an explicit processed-timestamp (mirroring how embeddings use a nullable column, not a side table) fixes it. Caught before shipping, not after. |
| 31 | **Relations are same-article co-occurrence with type-informed predicates, labeled as a heuristic** | No relation-extraction model exists in this stack. Labeling edges `develops`/`targets`/`co_mentioned_with` by entity-type pair is a legible, cheap heuristic — but it's still just co-occurrence, and the model/service docstrings say so directly rather than implying a verified biological claim. |
| 32 | **`app/ml/signals.py` split out of `reranker.py`** | `explain.py` needed the same user/item signal functions reranker.py already had, but importing reranker.py would pull numpy+lightgbm into the always-loaded `/users` router's import path — the exact bug class fixed for v0.7 (decision #22). Splitting the pure-Python signal functions into their own module lets both consume them without either paying that cost. |
| 33 | **`app/database.py` auto-detects `sqlite://` and configures `check_same_thread=False` + `StaticPool`** | Discovered running the full FastAPI/SSE relay on a GPU job without Postgres available: a real `uvicorn` process runs sync path operations in a threadpool, and a bare SQLite connection can't cross threads, so the app crashed on the first request. Production always used Postgres so this never surfaced. Fixing it in `app/database.py` (rather than requiring every ad-hoc script to know the workaround, as `tests/conftest.py` and `scripts/seed_v07_synthetic.py` each independently did/needed) makes any future SQLite-backed run — tests, scripts, or a quick verification job — correct by default. |
| 34 | **Trial entities grounded dynamically (live API), not added to the static gazetteer** | Diseases/drugs/genes/companies are a bounded, slowly-changing set — a curated list is a reasonable fit. Clinical trials are the opposite: new NCT ids appear continuously in real news. A live per-mention ClinicalTrials.gov lookup (cached locally after the first sighting via `external_id`) fits that shape better than trying to pre-populate a list that would always be stale. |
| 35 | **A lookup failure (trial or ontology) never fabricates a placeholder entity** | `_extract_trial_entities` skips an NCT id outright if the API call fails, rather than creating an entity with a synthetic name like "Trial NCT04173585." A knowledge graph with a wrong or made-up node is worse than a knowledge graph missing a node — the gap is visible and fixable; a plausible-looking fabrication isn't. |

---

## 4. Current Repo Layout

```
BLUEPRINT.md              roadmap, positioning, resume framing
PROJECT_STATUS.md         this file
METRICS.md                measured vs. target numbers
ANOMALY_EXPLAIN_LLM.md    anomaly detection + self-hosted vLLM explanation feature
docker-compose.yml        backend + Postgres
backend/
  alembic/                migrations 0001 (articles) → 0006 (kg entities/mentions/relations)
  app/
    main.py, config.py, database.py, scheduler.py
    models.py             Article (+kg_extracted_at), IngestionRun, User, UserInteraction, UserEmbedding
    schemas.py            Pydantic models (+ ExplanationRead for v0.9)
    crud.py               dedup + get-or-create + run recording + find_similar + recommendation logic
    routers/              articles (+ /related), ingestion, search, recommendations (v0.6, +/explain), recommendations_v07
    auth/                 JWT issuance/refresh, pluggable Apple/Google/fake identity verification, require_self
    kg/                   Entity/EntityMention/EntityRelation, gazetteer extractor, /kg routes
    ingestion/
      base.py, registry.py, runner.py    Source ABC + registry + generic runner (embeds + extracts entities)
      rss.py, pubmed.py, biorxiv.py      RSS, PubMed E-utilities, bioRxiv/medRxiv sources
      feeds.py            RSS feed list
    ml/
      embeddings.py       Embedder backends (hash / PubMedBERT)
      service.py          embed lifecycle
      signals.py          dependency-free user/item signals shared by reranker + explain
      two_tower.py        v0.7 two-tower retrieval model
      reranker.py         v0.7 LightGBM reranker + feature extraction
      recommender_v07.py  v0.7 pipeline with v0.6 fallback
      explain.py          v0.9 structured multi-signal explanations
    anomaly/              AnomalyEvent model, cross-source burst detector, internal routes
    llm/                  vLLM client, prompt builder, SSE route
  scripts/
    eval_retrieval.py     v0.5 retrieval quality eval (Recall@k, NDCG@k)
    train_v07.py          two-tower + reranker training pipeline
    seed_v07_synthetic.py synthetic corpus/users to exercise train_v07.py without real users
  requirements-ml.txt     heavy ML deps (sentence-transformers/torch/lightgbm/numpy), split out to keep CI light
  tests/                  API, RSS, CRUD, dedup, retrieval, retrieval-eval, recommendations, auth, kg, explain, anomaly tests
llm_serving/              GPU-instance vLLM deployment scripts (see ANOMALY_EXPLAIN_LLM.md)
observability/            Prometheus + Grafana, scrapes a remote vLLM /metrics
benchmarks/               Load-test reports for the anomaly-explain endpoint
.github/workflows/        backend CI
```
