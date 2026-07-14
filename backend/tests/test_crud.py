from app import crud
from app.schemas import ArticleCreate
from tests.conftest import TestingSessionLocal


def test_create_article_survives_concurrent_insert(db_session):
    """Two sessions both pass the existence check before either commits.

    This deterministically reproduces the race two overlapping /ingest/run
    requests could hit: the loser's commit must not raise, and the session
    must stay usable afterwards instead of sitting in an aborted transaction.
    """
    article = ArticleCreate(title="Race", url="https://example.com/race", source="TestSource")

    other_session = TestingSessionLocal()
    try:
        assert crud.get_article_by_url(db_session, article.url) is None
        assert crud.get_article_by_url(other_session, article.url) is None

        winner, winner_created = crud.create_article(other_session, article)
        assert winner_created is True

        loser, loser_created = crud.create_article(db_session, article)
        assert loser_created is False
        assert loser.id == winner.id

        # the session must still be usable after recovering from the collision
        assert crud.get_article(db_session, loser.id) is not None
    finally:
        other_session.close()
