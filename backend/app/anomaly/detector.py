"""Minimal candidate early-signal detector.

Heuristic: cross-source burst. If an article's semantic neighbors (via the
existing pgvector embedding / find_similar) include articles from other
sources published within a short window of it, independent outlets are
covering the same story close together in time -- a plausible early signal
worth surfacing to a human, and a reasonable precursor for the LLM
explanation feature downstream.

This is intentionally simple and not a claim of production-grade anomaly
detection: it exists so /internal/anomaly-explain has a real event to
consume instead of a mocked schema. It only reads Article rows and the
existing crud.find_similar helper; it never touches ingestion or dedup.
"""

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import crud
from app.anomaly import crud as anomaly_crud
from app.anomaly.models import AnomalyEvent
from app.models import Article

WINDOW_HOURS = 48
SIMILARITY_THRESHOLD = 0.82
MIN_OTHER_SOURCES = 2
NEIGHBOR_LIMIT = 20
KIND = "cross_source_burst"


def detect_for_article(db: Session, article: Article) -> AnomalyEvent | None:
    """Check one article for a cross-source burst and persist it if found.

    Idempotent: re-running against an article that already raised this kind
    of event returns the existing row instead of duplicating it.
    """
    existing = anomaly_crud.get_existing(db, article.id, KIND)
    if existing:
        return existing

    if article.embedding is None or article.published_at is None:
        return None

    neighbors = crud.find_similar(
        db, article.embedding, limit=NEIGHBOR_LIMIT, exclude_id=article.id
    )

    window = timedelta(hours=WINDOW_HOURS)
    related: list[tuple[Article, float]] = []
    for neighbor, similarity in neighbors:
        # crud.find_similar's SQLite branch computes similarity over
        # numpy-backed embedding rows, yielding numpy.float32 -- normalize to
        # a plain float so the JSON `detail` column can serialize it.
        similarity = float(similarity)
        if similarity < SIMILARITY_THRESHOLD:
            continue
        if neighbor.source == article.source:
            continue
        if neighbor.published_at is None:
            continue
        if abs(neighbor.published_at - article.published_at) > window:
            continue
        related.append((neighbor, similarity))

    related_sources = sorted({neighbor.source for neighbor, _ in related})
    if len(related_sources) < MIN_OTHER_SOURCES:
        return None

    mean_similarity = sum(similarity for _, similarity in related) / len(related)
    event = AnomalyEvent(
        article_id=article.id,
        kind=KIND,
        score=round(mean_similarity * len(related_sources), 4),
        detail={
            "related_article_ids": [neighbor.id for neighbor, _ in related],
            "related_sources": related_sources,
            "window_hours": WINDOW_HOURS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "mean_similarity": round(mean_similarity, 4),
        },
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def detect_recent(db: Session, limit: int = 100) -> list[AnomalyEvent]:
    """Scan the most recently fetched, embedded articles for candidate events."""
    articles = (
        db.execute(
            select(Article)
            .where(Article.embedding.is_not(None))
            .order_by(Article.fetched_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    events = []
    for article in articles:
        event = detect_for_article(db, article)
        if event:
            events.append(event)
    return events
