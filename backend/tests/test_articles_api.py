from app import crud
from app.schemas import ArticleCreate


def _make_article(db_session, **overrides):
    defaults = dict(
        title="Test Article",
        url="https://example.com/a1",
        source="TestSource",
        summary="A summary",
        published_at=None,
    )
    defaults.update(overrides)
    article, _ = crud.create_article(db_session, ArticleCreate(**defaults))
    return article


def test_list_articles_empty(client):
    response = client.get("/articles")
    assert response.status_code == 200
    assert response.json() == []


def test_list_articles_returns_created(client, db_session):
    _make_article(db_session)

    response = client.get("/articles")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["title"] == "Test Article"
    assert body[0]["url"] == "https://example.com/a1"


def test_list_articles_filters_by_source(client, db_session):
    _make_article(db_session, url="https://example.com/a1", source="SourceA")
    _make_article(db_session, url="https://example.com/a2", source="SourceB")

    response = client.get("/articles", params={"source": "SourceA"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["source"] == "SourceA"


def test_get_article_by_id(client, db_session):
    article = _make_article(db_session)

    response = client.get(f"/articles/{article.id}")
    assert response.status_code == 200
    assert response.json()["id"] == article.id


def test_get_article_404(client):
    response = client.get("/articles/9999")
    assert response.status_code == 404
