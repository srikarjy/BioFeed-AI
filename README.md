# BioFeed AI

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
   and a vLLM-served model streams a plain-language explanation — backend
   infrastructure work (SSE streaming, Prometheus/Grafana, load-testing)
   kept isolated from the core feed, documented honestly as unverified
   against a real GPU (see [`ANOMALY_EXPLAIN_LLM.md`](./ANOMALY_EXPLAIN_LLM.md)).

The mobile client, auth, knowledge graph, and the v2.0 market-signal module
are roadmap, not shipped — see [`BLUEPRINT.md`](./BLUEPRINT.md) §4 for what's
next and why the order was chosen.

## Quick links

- [Full blueprint & roadmap](./BLUEPRINT.md)
- [Project status & architecture decisions](./PROJECT_STATUS.md)
- [Measured metrics](./METRICS.md)
- [Anomaly detection + self-hosted LLM explanations](./ANOMALY_EXPLAIN_LLM.md)
- Tech stack: Python/FastAPI/PostgreSQL(pgvector)/SQLAlchemy/Alembic/
  APScheduler · PyTorch/Sentence-Transformers/PubMedBERT/LightGBM ·
  Docker/GitHub Actions/Prometheus/Grafana/vLLM · Swift/SwiftUI (planned)

## Status

**v0.1–v0.7 shipped**: FastAPI + PostgreSQL foundation, multi-source
ingestion (RSS + PubMed + bioRxiv/medRxiv) on a scheduler, semantic search
and retrieval over pgvector, and a two-tower + LightGBM recommendation
pipeline with a labeled retrieval eval set. Plus an additive anomaly
detection + self-hosted LLM explanation feature. See `PROJECT_STATUS.md`
for the full breakdown and what's next (auth, mobile app, knowledge graph,
market-signal module).

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

## Repo structure

```
backend/            FastAPI service
  app/
    ingestion/       Source abstraction: RSS, PubMed, bioRxiv/medRxiv
    ml/               Embeddings, two-tower model, LightGBM reranker
    anomaly/          Cross-source burst detector
    llm/              vLLM client + SSE route for anomaly explanations
    routers/          articles, search, ingestion, recommendations
  scripts/            eval_retrieval.py, train_v07.py
  tests/              pytest suite (API, dedup, retrieval, recommendations)
llm_serving/         GPU-instance vLLM deployment scripts
observability/       Prometheus + Grafana (scrapes vLLM /metrics)
benchmarks/          Load-test reports for the anomaly-explain endpoint
ios/                 (planned, v0.4)
```

## License

TBD
