from pathlib import Path

from app import crud
from app.ingestion.rss import ingest_feed

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_feed.xml"


def test_ingest_feed_adds_articles(db_session):
    added = ingest_feed(db_session, "TestSource", str(FIXTURE_PATH))

    assert added == 3
    articles = crud.get_articles(db_session)
    assert {a.url for a in articles} == {
        "https://example.com/articles/1",
        "https://example.com/articles/2",
        "https://example.com/articles/3",
    }
    assert all(a.source == "TestSource" for a in articles)


def test_ingest_feed_is_idempotent(db_session):
    first_run = ingest_feed(db_session, "TestSource", str(FIXTURE_PATH))
    second_run = ingest_feed(db_session, "TestSource", str(FIXTURE_PATH))

    assert first_run == 3
    assert second_run == 0
    assert len(crud.get_articles(db_session)) == 3


def test_ingest_feed_strips_html(db_session):
    ingest_feed(db_session, "TestSource", str(FIXTURE_PATH))
    article = crud.get_article_by_url(db_session, "https://example.com/articles/3")

    # Tags dropped, entities decoded, whitespace collapsed — no angle brackets
    # or tag names survive into the text the embedder sees.
    assert article.title == "Gene Therapy Wins Approval"
    assert article.summary == "Summary with bold markup & an entity."
    assert "<" not in article.title and "<" not in article.summary
    assert "href" not in article.title


def test_ingest_feed_bad_source_raises(db_session):
    missing_path = Path(__file__).parent / "fixtures" / "does_not_exist.xml"
    try:
        ingest_feed(db_session, "BadSource", str(missing_path))
    except Exception:
        pass
    else:
        raise AssertionError("Expected an exception for an unparseable feed")
