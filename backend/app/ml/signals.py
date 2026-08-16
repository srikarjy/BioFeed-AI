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


def compute_user_affinities(db: Session, user_id: int) -> tuple[dict[str, float], dict[str, float]]:
    """User's topic and source affinities from interaction history.

    Topic affinity is currently source-as-proxy (same values as source
    affinity) -- a real topic model needs entity/topic tagging per article,
    which v0.8's knowledge graph now provides the entities for but nothing
    here consumes yet. See PROJECT_STATUS.md "v0.9 remaining."
    """
    from app import crud

    interactions = crud.get_user_positive_interactions(db, user_id, limit=500)
    if not interactions:
        return {}, {}

    article_ids = [i.article_id for i in interactions]
    articles = db.query(Article).filter(Article.id.in_(article_ids)).all()

    source_counts: dict[str, int] = {}
    for art in articles:
        source_counts[art.source] = source_counts.get(art.source, 0) + 1

    total = sum(source_counts.values())
    source_affinity = {k: v / total for k, v in source_counts.items()}
    topic_affinity = source_affinity.copy()

    return topic_affinity, source_affinity


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
