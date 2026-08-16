"""Tests for app/ml/signals.py, in particular the v0.9 fix replacing
source-as-topic-proxy with real KG-entity-based topic affinity.
"""

from app import crud
from app.kg import service as kg_service
from app.ml.signals import compute_user_affinities, topic_affinity_score
from app.schemas import ArticleCreate, InteractionType


def _make_article(db, title, source="STAT News", summary=""):
    article, _ = crud.create_article(
        db, ArticleCreate(title=title, url=f"http://signals-test/{title}", source=source, summary=summary)
    )
    return article


def test_topic_affinity_is_keyed_by_entity_not_source(db_session):
    user = crud.create_user(db_session, email="signals@example.com")

    read = _make_article(db_session, "Moderna advances mRNA vaccine research", source="STAT News")
    kg_service.extract_for_article(db_session, read)
    crud.create_interaction(db_session, user.id, read.id, InteractionType.READ, read_time_seconds=180)

    topic_affinity, source_affinity = compute_user_affinities(db_session, user.id)

    # Keyed by entity id, not by article source string.
    assert "STAT News" not in topic_affinity
    assert all(isinstance(k, int) for k in topic_affinity)
    assert "STAT News" in source_affinity


def test_topic_affinity_score_rewards_shared_entities_across_different_sources(db_session):
    user = crud.create_user(db_session, email="signals2@example.com")

    read = _make_article(db_session, "Moderna reports mRNA vaccine trial update", source="STAT News")
    kg_service.extract_for_article(db_session, read)
    crud.create_interaction(db_session, user.id, read.id, InteractionType.BOOKMARK)

    # Different source, but mentions the same company -> should score > 0
    # under the new entity-based signal, which the old source-proxy signal
    # could never do (different source strings never overlapped).
    same_topic_diff_source = _make_article(
        db_session, "Moderna vaccine platform expands pipeline", source="FiercePharma"
    )
    kg_service.extract_for_article(db_session, same_topic_diff_source)

    unrelated = _make_article(db_session, "Stock market closes higher on tech earnings", source="FiercePharma")
    kg_service.extract_for_article(db_session, unrelated)

    topic_affinity, _ = compute_user_affinities(db_session, user.id)

    same_topic_score = topic_affinity_score(db_session, same_topic_diff_source, topic_affinity)
    unrelated_score = topic_affinity_score(db_session, unrelated, topic_affinity)

    assert same_topic_score > 0
    assert same_topic_score > unrelated_score


def test_topic_affinity_score_zero_without_kg_entities(db_session):
    user = crud.create_user(db_session, email="signals3@example.com")
    article = _make_article(db_session, "Some article")
    topic_affinity, _ = compute_user_affinities(db_session, user.id)
    assert topic_affinity_score(db_session, article, topic_affinity) == 0.0


def test_compute_user_affinities_empty_for_user_with_no_interactions(db_session):
    user = crud.create_user(db_session, email="signals4@example.com")
    topic_affinity, source_affinity = compute_user_affinities(db_session, user.id)
    assert topic_affinity == {}
    assert source_affinity == {}
