# BioFeed AI

Personalized biotech intelligence feed for iOS, with a recommendation engine built on biomedical text embeddings, learning-to-rank, and (from v2.0) a biotech stock event-correlation module.

See [`BLUEPRINT.md`](./BLUEPRINT.md) for the full architecture, tech stack, versioned roadmap, and resume framing.

---

## What this is

A native iOS app backed by a FastAPI service that:

1. Ingests biotech literature, news, FDA updates, and company announcements
2. Embeds and understands content using biomedical transformer models (PubMedBERT, BioBERT)
3. Learns a personalized ranking from user interaction signals (reading time, bookmarks, scroll, search)
4. Serves an explainable, personalized feed ("recommended because...")
5. (v2.0+) Correlates biotech scientific/regulatory events with stock price movement using event-study methodology

Architecturally, this is a general-purpose personalized content ranking system (the same family as consumer feed ranking at TikTok/Spotify/LinkedIn), applied to a biotech-specific vertical to avoid duplicating the thousand generic "X for Y" recsys clones, and extended with a market-signal module that most content-feed projects don't have.

## Quick links

- [Full blueprint & roadmap](./BLUEPRINT.md)
- Tech stack: Swift/SwiftUI (mobile) · Python/FastAPI/PostgreSQL/Redis (backend) · PyTorch/Sentence-Transformers/FAISS (ML) · Docker/AWS/MLflow/Prometheus (infra)

## Status

Currently scoping toward v0.6 (first resume-ready milestone: end-to-end personalized recommender). See the roadmap in `BLUEPRINT.md` for what's shipped vs. planned.

## Repo structure (planned)

```
/ios              SwiftUI application
/backend          FastAPI service, ingestion, ranking
/ml               training scripts, embedding pipelines, ranking models
/market           (v2.0+) event-study pipeline, ticker mapping, OHLCV ingestion
/infra            Docker, CI/CD, monitoring configs
```

## License

TBD
