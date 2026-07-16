"""Retrieval loop: embedding on ingest, /search, and /related.

These run against the SQLite fixture with EMBEDDING_BACKEND=hash, so similarity
is token-overlap rather than true semantics — enough to assert the wiring and
ranking behavior end to end.
"""

from app import crud
from app.ml import service as ml_service
from app.schemas import ArticleCreate


def _make(db, title, summary=""):
    article, _ = crud.create_article(
        db, ArticleCreate(title=title, url=f"http://x/{title}", source="test", summary=summary)
    )
    return article


def test_embed_missing_fills_and_is_idempotent(db_session):
    _make(db_session, "CRISPR gene editing in sickle cell disease")
    _make(db_session, "Quarterly biotech venture funding report")

    assert ml_service.embed_missing(db_session) == 2
    # Second call finds nothing left to do.
    assert ml_service.embed_missing(db_session) == 0


def test_search_ranks_by_similarity(client, db_session):
    _make(db_session, "CRISPR gene editing therapy for sickle cell disease")
    _make(db_session, "Stock market closes higher on tech earnings")
    ml_service.embed_missing(db_session)

    resp = client.get("/search", params={"q": "gene editing sickle cell", "limit": 2})
    assert resp.status_code == 200
    results = resp.json()
    assert results[0]["title"].startswith("CRISPR gene editing")
    # Similarity is descending and bounded to [-1, 1].
    scores = [r["similarity"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_related_excludes_self_and_ranks(client, db_session):
    anchor = _make(db_session, "mRNA vaccine platform for influenza")
    _make(db_session, "mRNA vaccine manufacturing scale-up")
    _make(db_session, "Agricultural commodity prices this quarter")
    ml_service.embed_missing(db_session)

    resp = client.get(f"/articles/{anchor.id}/related", params={"limit": 5})
    assert resp.status_code == 200
    results = resp.json()
    assert all(r["id"] != anchor.id for r in results)
    assert results[0]["title"] == "mRNA vaccine manufacturing scale-up"


def test_related_404_for_missing_article(client):
    assert client.get("/articles/9999/related").status_code == 404


def test_related_embeds_on_demand_when_missing(client, db_session):
    anchor = _make(db_session, "Novel antibody therapy for lymphoma")
    _make(db_session, "Antibody drug conjugate for lymphoma treatment")
    # Note: no embed_missing() — anchor has no embedding yet.
    resp = client.get(f"/articles/{anchor.id}/related")
    assert resp.status_code == 200
