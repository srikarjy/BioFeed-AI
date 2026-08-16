# BioFeed AI

**🔗 Live demo: [biofeed-ai.onrender.com/demo](https://biofeed-ai.onrender.com/demo)**
([API docs](https://biofeed-ai.onrender.com/docs)) — a free-tier deployment of
the core service (search, recommendations, KG, auth). First request after
idling can take 30–60s to wake the instance up. Runs with dependency-free hash
embeddings, not the PubMedBERT quality in `METRICS.md` — see the demo page's
banner and the [Live deployment](#live-deployment) section below for exactly
what's traded off and why.

A personalized biotech intelligence feed. A FastAPI service ingests biomedical
literature and biotech news, embeds it with biomedical transformer models,
and ranks it per-user with a two-tower retrieval model and a LightGBM
reranker — the same architecture family as feed ranking at TikTok/Spotify/
LinkedIn, applied to a data-rich niche vertical instead of another generic
"X for Y" recsys clone.

See [`BLUEPRINT.md`](./BLUEPRINT.md) for the full roadmap and resume framing,
[`PROJECT_STATUS.md`](./PROJECT_STATUS.md) for what's built and the
architecture decisions behind it, and [`METRICS.md`](./METRICS.md) for
measured vs. target numbers.

---

## Try it in 60 seconds

Against the live deployment, no clone required:

```bash
curl https://biofeed-ai.onrender.com/health
curl -X POST https://biofeed-ai.onrender.com/ingest/run   # seed it with fresh articles if empty
curl "https://biofeed-ai.onrender.com/search?q=CRISPR%20sickle%20cell"
```

Or locally, with the real PubMedBERT embedder and the full stack:

```bash
docker compose up --build
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ingest/run
curl "http://localhost:8000/search?q=CRISPR%20sickle%20cell"
```

---

## What this is

A FastAPI backend (native iOS client planned, not yet built) that:

1. **Ingests** biotech news (RSS), biomedical literature (PubMed E-utilities),
   and preprints (bioRxiv/medRxiv) on a scheduled cadence, with per-source
   error isolation and three-tier cross-source dedup (URL → DOI → normalized
   title hash).
2. **Embeds** every article with PubMedBERT (768-dim, pgvector/HNSW) so
   `GET /search?q=` and `GET /articles/{id}/related` do real semantic
   retrieval, not keyword matching.
3. **Learns a personalized ranking** from interaction signals (reads,
   bookmarks, likes, hides, searches): a user embedding built from weighted
   interaction history, retrieved via a two-tower model, reranked by
   LightGBM on user/article/affinity features — falling back cleanly to
   content-similarity for cold-start users with no history yet.
4. **Explains anomalies via a self-hosted LLM**: a cross-source burst
   detector flags when multiple outlets cover the same story within 48h,
   and a vLLM-served model streams a plain-language explanation, verified
   on a real GPU (see [`ANOMALY_EXPLAIN_LLM.md`](./ANOMALY_EXPLAIN_LLM.md)).
5. **Authenticates users** via Sign in with Apple/Google — JWT access +
   refresh tokens, pluggable identity verification (a fake provider for
   dev/test, real Apple/Google JWKS verification for prod), and every
   user-scoped route actually enforces "you can only touch your own data."
6. **Builds a knowledge graph** of entities grounded in real ontology
   identifiers (MONDO/HPO, ChEMBL, HGNC) extracted from every ingested
   article, with co-occurrence relations (`company → develops → drug`,
   `drug → targets → disease`) queryable via `GET /kg/...`.
7. **Explains recommendations with structured, multi-signal reasoning** —
   nearest-interaction similarity, topic/source affinity, freshness,
   popularity, and shared knowledge-graph entities, each surfaced with its
   own weight via `GET /users/{id}/articles/{id}/explain`.

The mobile client and the v2.0 market-signal module are roadmap, not
shipped — see [`BLUEPRINT.md`](./BLUEPRINT.md) §4 for what's next and why
the order was chosen. Both need real AWS spend / an iOS device+Apple
Developer account to go further, so they're gated on an explicit decision
to invest in them, not on remaining engineering effort.

## Quick links

- [Full blueprint & roadmap](./BLUEPRINT.md)
- [Project status & architecture decisions](./PROJECT_STATUS.md)
- [Measured metrics](./METRICS.md)
- [Anomaly detection + self-hosted LLM explanations](./ANOMALY_EXPLAIN_LLM.md)
- Tech stack: Python/FastAPI/PostgreSQL(pgvector)/SQLAlchemy/Alembic/
  APScheduler · PyTorch/Sentence-Transformers/PubMedBERT/LightGBM ·
  Docker/GitHub Actions/Prometheus/Grafana/vLLM · Swift/SwiftUI (planned)

## Status

**v0.1–v0.9 shipped**: FastAPI + PostgreSQL foundation, multi-source
ingestion (RSS + PubMed + bioRxiv/medRxiv) on a scheduler, semantic search
and retrieval over pgvector, a two-tower + LightGBM recommendation pipeline
with a labeled retrieval eval set, JWT auth with pluggable Apple/Google
verification, a knowledge graph grounded in real ontology identifiers, and
structured multi-signal recommendation explanations. Plus an additive
anomaly detection + self-hosted LLM explanation feature, verified on a real
GPU. 77 backend tests passing. See `PROJECT_STATUS.md` for the full
breakdown and what's next (mobile app, market-signal module — both gated
on real spend/an Apple Developer account, not remaining engineering work).

## Running it

```bash
docker compose up --build          # FastAPI + Postgres(pgvector)
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ingest/run   # manual trigger; scheduler also runs it every INGESTION_INTERVAL_MINUTES
curl "http://localhost:8000/search?q=CRISPR%20sickle%20cell"
```

Backend tests (no Docker/torch needed — uses the dependency-free hashing
embedder and SQLite):

```bash
cd backend
pip install -r requirements.txt
EMBEDDING_BACKEND=hash pytest -q
```

Retrieval-quality eval against the hand-labeled query set
(`tests/fixtures/retrieval_eval.json`), enforced as a CI floor and runnable
standalone for a full per-query report:

```bash
cd backend
EMBEDDING_BACKEND=hash python scripts/eval_retrieval.py
```

## Live deployment

The demo at [biofeed-ai.onrender.com](https://biofeed-ai.onrender.com/demo)
runs the exact image CI publishes to GHCR (`.github/workflows/docker.yml`),
deployed via [`render.yaml`](./render.yaml) — a Render Blueprint (free web
service + free Postgres/pgvector). CI pings Render's deploy hook after every
push to `main`, so the live demo tracks the repo automatically; nothing about
it is deployed or updated by hand.

What's different from a full local/prod run, and why:

- **`EMBEDDING_BACKEND=hash`**, not `sentence-transformers` — the real
  PubMedBERT embedder needs more RAM than a free-tier instance has. The
  hashing embedder (`app/ml/embeddings.py`) is deterministic and
  dependency-free but not semantically meaningful, so search/related-article
  results on the live demo are topically rough, not the retrieval quality
  reported in `METRICS.md`. Locally/in production this flips back to real
  embeddings with one env var.
- **`LLM_ENABLED=false`** — the self-hosted vLLM anomaly-explanation feature
  needs a real GPU instance running continuously, which is real ongoing
  spend, not a one-off cost. It's verified separately on a real GPU (see
  [`ANOMALY_EXPLAIN_LLM.md`](./ANOMALY_EXPLAIN_LLM.md)) rather than kept live
  here.
- **Auth stays on `AUTH_PROVIDER=fake`** (the default — see
  `app/auth/config.py` and `app/auth/providers.py`). Real Sign in with
  Apple/Google needs a registered app and device-minted tokens that don't
  exist without the (roadmap, spend-gated) mobile client. The fake provider
  isn't a shortcut around auth — it exercises the *real* token
  issuance/refresh/`require_self`-enforcement code path end to end, just
  against a self-issued identity claim instead of Apple/Google's. The
  `/demo` page uses this to sign in a fresh demo user
  (`POST /auth/apple` with a `{"sub": "..."}` identity token) and get a real
  JWT before calling the auth-protected `/users/{id}/feed` and
  `/users/{id}/articles/{id}/explain` routes. A production deploy sets
  `AUTH_PROVIDER=real` plus `AUTH_APPLE_CLIENT_ID`/`AUTH_GOOGLE_CLIENT_ID`
  once those app registrations exist — nothing defaults to `real`, so a
  misconfigured deployment fails closed rather than silently trusting
  anything.

## Repo structure

```
backend/            FastAPI service
  app/
    ingestion/       Source abstraction: RSS, PubMed, bioRxiv/medRxiv
    ml/               Embeddings, two-tower model, LightGBM reranker, explanations
    auth/             JWT + pluggable Apple/Google/fake identity verification
    kg/               Knowledge graph: gazetteer + live trial extraction, entities, relations
    anomaly/          Cross-source burst detector
    llm/              vLLM client + SSE route for anomaly explanations
    routers/          articles, search, ingestion, recommendations
  scripts/            eval_retrieval.py, train_v07.py, seed_v07_synthetic.py
  tests/              pytest suite (API, dedup, retrieval, recommendations, auth, kg, explain, anomaly)
llm_serving/         GPU-instance vLLM deployment scripts
observability/       Prometheus + Grafana (scrapes vLLM /metrics)
benchmarks/          Load-test reports for the anomaly-explain endpoint
ios/                 (planned, v0.4)
```

## License

TBD
