# BioFeed AI

## Personalized Biotech Intelligence & Market Signal Platform

**Project Codename:** BioFeed AI

**Positioning:** A vertical-specific personalized content ranking system (same architecture family as TikTok/Spotify/LinkedIn feed ranking), applied to biotech literature and news, extended with a biotech-stock event-correlation module. Biotech is the chosen niche vertical specifically to differentiate from generic recsys clone projects, and to open a secondary door into quant/fintech-adjacent roles.

---

## 1. Project Goal

Instead of manually tracking dozens of biotech sources, BioFeed AI:

* Continuously collects biotech literature, news, FDA updates, and company announcements
* Understands content using transformer-based embeddings
* Learns each user's interests from interaction signals
* Delivers a personalized, explainable native iOS feed
* Correlates scientific/regulatory events with biotech stock price movement (v2.0+)

---

## 2. High-Level Architecture

```
                    Data Sources
   PubMed · bioRxiv · medRxiv · FDA · NIH
   Nature · Science · Cell · Company Blogs
   Press Releases · RSS · SEC Filings
   Market Data (OHLCV) · PDUFA Calendar · Trial Readouts
                        │
                        ▼
              Content Ingestion Service
                        │
                 Cleaning & Parsing
                        │
                Duplicate Detection
                        │
           NLP + Embedding Generation
                        │
                Metadata Extraction
              (entities, tickers, event types)
                        │
              Event-to-Price Correlation
                 (v2.0 module, biotech tickers only)
                        │
                Recommendation Engine
                        │
                  FastAPI Backend
                        │
                    PostgreSQL
                        │
                    Redis Cache
                        │
              Authentication Service
                        │
                 SwiftUI iOS App
```

---

## 3. Technology Stack

### Mobile
Swift · SwiftUI · Combine · URLSession · Keychain

### Backend
Python · FastAPI · PostgreSQL · Redis · SQLAlchemy · Celery · Docker

### Machine Learning
PyTorch · Sentence Transformers · PubMedBERT · BioBERT · FAISS · LightGBM/XGBoost (ranking)

### Market Data (v2.0+)
yfinance / Polygon.io (OHLCV) · FDA PDUFA calendar (public) · ClinicalTrials.gov readouts · static biotech ticker map (~600–700 tickers) · XBI (biotech sector ETF) as market benchmark for event-study

### Infrastructure
Docker · GitHub Actions · AWS · MLflow · Prometheus · Grafana

---

## 4. Development Roadmap

### v0.1 — Foundation
FastAPI, PostgreSQL, Docker, basic article API, RSS ingestion, initial schema.
**Skills:** backend engineering, REST APIs, Docker, DB design.
**Resume:** ❌ No

### v0.2 — Content Platform
RSS, PubMed, bioRxiv integration, parsing, dedup, scheduled jobs.
**Skills:** data engineering, ETL, API integrations.
**Resume:** ❌ No

### v0.3 — Authentication
Sign in with Apple, Google OAuth, JWT, refresh tokens, profiles, Keychain.
**Skills:** OAuth, auth, security, iOS.
**Resume:** ❌ No — expected baseline, not differentiating.

### v0.4 — Mobile Application
SwiftUI feed, article page, search, bookmarking, reading history, offline cache.
**Skills:** native iOS, API integration, state management.
**Resume:** ⚠ Optional — only for mobile/backend-leaning applications.

### v0.5 — NLP Pipeline
Embeddings, summaries, entity extraction (orgs, diseases, genes, drugs, funding events).
**Models:** PubMedBERT, BioBERT, Sentence Transformers.
**Skills:** NLP, transformers, biomedical text processing.
**Resume:** ⚠ Optional

### v0.6 — Recommendation Engine (MVP)
Learns from likes, bookmarks, reading time, scroll behavior, hidden articles, searches.

```
User Embedding → Similarity Search → Ranking Model → Top Feed
```

**Skills:** recommendation systems, vector search, personalized ranking.
**Resume:** ✅ Yes — first end-to-end personalized recommender.

### v0.7 — Intelligent Ranking
Two-tower retrieval + LightGBM/XGBoost reranker + cross-encoder experiments.
Input: user embedding, article embedding, freshness, publisher, popularity, topic, history.
Output: click probability.
**Skills:** learning-to-rank, recsys engineering.
**Resume:** ✅ Yes — core recsys engineering signal, transfers directly to consumer feed / search-relevance roles.

### v0.8 — Knowledge Graph
Entity relationship graph (company → disease → trial → drug → paper).
**Skills:** graph traversal, relationship modeling.
**Resume:** ✅ Yes — note: for genuine biomedical KG depth, ground entities in real ontologies (UMLS, MONDO, ChEMBL) rather than pure NLP co-occurrence.

### v0.9 — Explainable AI
Surfaces "recommended because..." reasoning per feed item.
**Skills:** explainable AI, recommendation transparency.
**Resume:** ✅ Yes — genuinely underused differentiator; most consumer feeds don't expose this at all.

### v1.0 — Production Release
Full ingestion, auth, personalized feed, search, bookmarks, notifications, recommendation + explanation engine, monitoring, CI/CD, Docker deployment.
**Resume:** ✅ Absolutely — production-quality portfolio project.

---

## 5. Future Versions

### v1.1
Push notifications · daily digest · weekly biotech summary

### v1.2
Follow companies · follow researchers · follow journals

### v1.3
Voice summaries · podcasts · audio mode

### v2.0 — Biotech Market Signal Module ⭐ (new)

**Goal:** Correlate scientific/regulatory events with biotech stock price movement.

**Weak version (avoid):** a ticker widget next to articles. Cosmetic, near-zero hiring signal on its own.

**Strong version (target this):**
- Tag articles/press releases with company tickers via NER + static ticker lookup table.
- Compute price movement in an event window (e.g. ±5 days) around FDA approvals, PDUFA dates, trial readouts, patent grants.
- Run a classic event-study: cumulative abnormal return (CAR) relative to XBI (biotech sector ETF) as market benchmark — not raw price change.
- Feed this as a ranking signal ("this company's stock moved 12% after its last trial readout") or as a labeled dataset for a lightweight predictive model (does extracted sentiment/entity data correlate with subsequent price movement).

**Data sources:** yfinance/Polygon.io (OHLCV), FDA PDUFA calendar, ClinicalTrials.gov readout dates, static biotech/pharma ticker map.

**Skills demonstrated:** event-study methodology, financial time-series analysis, feature engineering across unstructured + market data, cross-domain signal fusion.

**Resume:** ✅ Strong — opens doors to biotech-focused funds' internal tooling teams and quant/fintech-adjacent roles, in addition to recsys/ML roles. Does **not** substitute for biological modeling depth (still NLP + finance, not biology) — keep this framing honest in interviews.

### v2.1
Multi-agent intelligence: Investment Agent, Research Agent, Clinical Trial Agent, Patent Agent, FDA Agent.

### v3.0
Enterprise platform — team-specific feeds (Drug Discovery, Clinical Development, AI Research, Business Development, Competitive Intelligence).

---

## 6. Resume Timeline

| Version | Resume Ready | Reason |
|---|---|---|
| 0.1 | ❌ | Infrastructure only |
| 0.2 | ❌ | Data ingestion only |
| 0.3 | ❌ | Auth is expected, not differentiating |
| 0.4 | ⚠ | Mobile showcase, little ML |
| 0.5 | ⚠ | NLP pipeline useful but incomplete |
| 0.6 | ✅ | End-to-end personalized recommender |
| 0.7 | ✅ | Strong ML/recsys engineering |
| 0.8 | ✅ | Adds graph-based intelligence |
| 0.9 | ✅ | Adds explainability |
| 1.0 | ⭐ | Production-quality portfolio project |
| 2.0 | ⭐ | Adds cross-domain (text + market) signal fusion — differentiates from generic recsys clones |

---

## 7. Suggested Resume Bullets

**Core (v0.6+):**
- Built a full-stack iOS and FastAPI application that ingests biomedical literature, biotech news, FDA announcements, and company updates into a unified personalized feed.
- Developed a transformer-based recommendation engine using biomedical text embeddings and vector search to personalize content based on user interactions (reading time, bookmarks, search behavior).
- Implemented secure authentication (Sign in with Apple, JWT), PostgreSQL, Redis caching, and Dockerized deployment for scalable multi-user access.

**Add after v2.0:**
- Built an event-study pipeline correlating extracted biotech news/trial/FDA events with abnormal stock returns (vs. XBI benchmark), turning unstructured biomedical text into a quantitative market-reaction signal used as a ranking feature.

**Framing note:** to a generalist recsys/ML recruiter, describe this as a "personalized content ranking system with domain-specific embeddings and cross-domain (text + market) signal fusion" — not "biotech intelligence platform." Biotech is the vertical, not the whole pitch.

---

## 8. Final Goal

By v2.0, BioFeed AI should demonstrate:

* Native iOS development
* Backend API engineering
* Authentication & security
* Biomedical NLP / transformer embeddings
* Recommendation systems & learning-to-rank
* Vector search
* Knowledge graphs (ontology-grounded)
* Explainable AI
* Event-study / financial time-series methodology
* Data engineering & MLOps
* Cloud deployment

The project should read as a focused, production-grade product — general-purpose recsys architecture, applied to a data-rich niche vertical, extended with a genuine cross-domain (text-to-market) signal — not a biotech-branded tutorial clone.
