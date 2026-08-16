"""User/item signal computation shared by the v0.7 reranker's feature
extraction and the v0.9 explanation builder.

Deliberately no numpy/torch/lightgbm imports: this module is on the import
path of app/ml/explain.py, which the always-loaded recommendations router
imports at module level. Keeping it dependency-free is what lets that
router stay importable without requirements-ml.txt installed -- the exact
bug class fixed in app/ml/recommender_v07.py (see PROJECT_STATUS.md
decision #22). reranker.py imports these rather than redefining them.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.models import Article, UserInteraction

TOP_JOURNALS = {"Nature", "Science", "Cell", "NEJM", "Lancet", "Nature Medicine"}
HIGH_IMPACT_JOURNALS = {
    "Nature Biotechnology", "Nature Methods", "Nature Genetics",
    "Cell Reports", "Molecular Cell", "Immunity", "Cancer Cell",
    "Science Translational Medicine", "Science Immunology",
}
MAJOR_NEWS = {"STAT News", "FiercePharma", "BioPharma Dive", "GEN", "Endpoints News"}


def compute_source_quality(source: str) -> float:
    """Heuristic source quality score (0-1)."""
    if source in TOP_JOURNALS:
        return 1.0
    if source in HIGH_IMPACT_JOURNALS:
        return 0.9
    if source in MAJOR_NEWS:
        return 0.7
    if "bioRxiv" in source or "medRxiv" in source:
        return 0.6
    if source == "PubMed":
        return 0.5
    return 0.3


def compute_freshness_days(published_at: Optional[datetime]) -> float:
    """Days since publication (capped at 365)."""
    if published_at is None:
        return 365.0
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    days = (now - published_at).days
    return min(float(days), 365.0)


def compute_item_popularity(db: Session, article_id: int) -> int:
    """Total interactions for this article across all users."""
    return db.query(UserInteraction).filter(UserInteraction.article_id == article_id).count()


def compute_user_affinities(db: Session, user_id: int) -> tuple[dict[int, float], dict[str, float]]:
    """User's topic and source affinities from interaction history.

    `topic_affinity` is keyed by KG entity id (see app.kg), weighted by how
    often each entity appears across the user's positively-interacted
    articles -- real per-article topics, not the source-as-proxy this used
    to be before v0.8's knowledge graph existed to tag articles with
    anything finer-grained than their source. An article with no KG
    entities (nothing in the gazetteer matched, or extraction hasn't run
    yet) contributes nothing to `topic_affinity` but still counts toward
    `source_affinity`. Use `topic_affinity_score` to turn this dict into a
    per-candidate-article score.
    """
    from app import crud
    from app.kg import crud as kg_crud

    interactions = crud.get_user_positive_interactions(db, user_id, limit=500)
    if not interactions:
        return {}, {}

    article_ids = [i.article_id for i in interactions]
    articles = db.query(Article).filter(Article.id.in_(article_ids)).all()

    source_counts: dict[str, int] = {}
    entity_counts: dict[int, int] = {}
    for art in articles:
        source_counts[art.source] = source_counts.get(art.source, 0) + 1
        for entity in kg_crud.get_article_entities(db, art.id):
            entity_counts[entity.id] = entity_counts.get(entity.id, 0) + 1

    source_total = sum(source_counts.values())
    source_affinity = {k: v / source_total for k, v in source_counts.items()}

    entity_total = sum(entity_counts.values())
    topic_affinity = {k: v / entity_total for k, v in entity_counts.items()} if entity_total else {}

    return topic_affinity, source_affinity


def topic_affinity_score(db: Session, article: Article, topic_affinity: dict[int, float]) -> float:
    """Sum of the user's topic-affinity weights for KG entities mentioned
    in `article` -- how much this article overlaps with what the user has
    shown interest in, by real entity rather than by source. 0 if the
    article has no KG entities extracted yet, or none overlap.
    """
    if not topic_affinity:
        return 0.0
    from app.kg import crud as kg_crud

    entities = kg_crud.get_article_entities(db, article.id)
    return sum(topic_affinity.get(e.id, 0.0) for e in entities)


def get_user_stats(db: Session, user_id: int) -> dict:
    """Aggregate user statistics used as reranker/explanation features."""
    from app import crud

    interactions = crud.get_user_positive_interactions(db, user_id, limit=1000)

    if not interactions:
        return {"interaction_count": 0, "avg_read_time": 0.0, "bookmark_rate": 0.0, "like_rate": 0.0}

    read_times = [i.read_time_seconds for i in interactions if i.read_time_seconds and i.interaction_type == "read"]
    bookmarks = sum(1 for i in interactions if i.interaction_type == "bookmark")
    likes = sum(1 for i in interactions if i.interaction_type == "like")

    return {
        "interaction_count": len(interactions),
        "avg_read_time": sum(read_times) / len(read_times) if read_times else 0.0,
        "bookmark_rate": bookmarks / len(interactions),
        "like_rate": likes / len(interactions),
    }
