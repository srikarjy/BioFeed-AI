"""Tests for v0.9 structured explanations (app/ml/explain.py + the
/users/{id}/articles/{id}/explain endpoint).
"""

from app import crud
from app.auth.jwt import create_access_token
from app.kg import service as kg_service
from app.ml import service as ml_service
from app.ml.explain import build_explanation
from app.schemas import ArticleCreate, InteractionType


def _make_article(db, title, source="STAT News", summary=""):
    article, _ = crud.create_article(
        db, ArticleCreate(title=title, url=f"http://explain-test/{title}", source=source, summary=summary)
    )
    return article


def test_explanation_falls_back_when_no_signals(db_session):
    user = crud.create_user(db_session, email="explain@example.com")
    article = _make_article(db_session, "Unrelated article", source="Obscure Blog")
    explanation = build_explanation(db_session, user.id, article)
    assert explanation.summary
    assert explanation.signals == []


def test_explanation_surfaces_nearest_interaction_signal(db_session):
    user = crud.create_user(db_session, email="explain2@example.com")

    bookmarked = _make_article(db_session, "CRISPR gene editing breakthrough")
    candidate = _make_article(db_session, "New CRISPR therapy shows promise")
    for art in (bookmarked, candidate):
        art.embedding = [0.2] * 768
        art.embedding[0] = 0.9
    db_session.commit()

    crud.create_interaction(db_session, user.id, bookmarked.id, InteractionType.BOOKMARK)

    explanation = build_explanation(db_session, user.id, candidate)
    labels = {s.label for s in explanation.signals}
    assert "similar_to_interaction" in labels


def test_explanation_surfaces_shared_entity_signal(db_session):
    user = crud.create_user(db_session, email="explain3@example.com")

    read_article = _make_article(db_session, "Moderna advances mRNA vaccine research")
    candidate = _make_article(db_session, "Moderna reports new vaccine trial data")

    kg_service.extract_for_article(db_session, read_article)
    kg_service.extract_for_article(db_session, candidate)

    crud.create_interaction(db_session, user.id, read_article.id, InteractionType.READ, read_time_seconds=120)

    explanation = build_explanation(db_session, user.id, candidate)
    labels = {s.label for s in explanation.signals}
    assert "shared_entities" in labels
    shared = next(s for s in explanation.signals if s.label == "shared_entities")
    assert "Moderna" in shared.detail


def test_explanation_signals_sorted_by_weight_descending(db_session):
    user = crud.create_user(db_session, email="explain4@example.com")
    article = _make_article(db_session, "STAT News investigative piece", source="STAT News")
    # Popularity signal: another user's interaction with this article.
    other = crud.create_user(db_session, email="other@example.com")
    crud.create_interaction(db_session, other.id, article.id, InteractionType.READ, read_time_seconds=60)

    explanation = build_explanation(db_session, user.id, article)
    weights = [s.weight for s in explanation.signals]
    assert weights == sorted(weights, reverse=True)


def test_explain_endpoint_requires_self(client, db_session):
    import json

    me = client.post("/auth/apple", json={"identity_token": json.dumps({"sub": "explain-self-1"})})
    other = client.post("/auth/apple", json={"identity_token": json.dumps({"sub": "explain-self-2"})})

    article = _make_article(db_session, "Some article")
    other_user_id = other.json()["user"]["id"]
    my_token = me.json()["access_token"]

    resp = client.get(
        f"/users/{other_user_id}/articles/{article.id}/explain",
        headers={"Authorization": f"Bearer {my_token}"},
    )
    assert resp.status_code == 403

    resp = client.get(f"/users/{other_user_id}/articles/{article.id}/explain")
    assert resp.status_code == 401


def test_explain_endpoint_returns_structured_payload(client, db_session):
    import json

    signup = client.post("/auth/apple", json={"identity_token": json.dumps({"sub": "explain-user-1"})})
    token = signup.json()["access_token"]
    user_id = signup.json()["user"]["id"]
    headers = {"Authorization": f"Bearer {token}"}

    article = _make_article(db_session, "CRISPR therapy update")
    ml_service.embed_missing(db_session)

    resp = client.get(f"/users/{user_id}/articles/{article.id}/explain", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "summary" in body
    assert isinstance(body["signals"], list)


def test_explain_endpoint_404_for_missing_article(client):
    import json

    signup = client.post("/auth/apple", json={"identity_token": json.dumps({"sub": "explain-user-2"})})
    token = signup.json()["access_token"]
    user_id = signup.json()["user"]["id"]

    resp = client.get(
        f"/users/{user_id}/articles/999999/explain",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404
