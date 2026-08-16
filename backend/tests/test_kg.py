"""Tests for v0.8 knowledge graph: gazetteer extraction, mentions,
co-occurrence relations, and the manual backfill endpoint.
"""

from app import crud
from app.kg import service
from app.kg.extractor import extract_entities
from app.schemas import ArticleCreate


def _make_article(db, title, summary=""):
    article, _ = crud.create_article(
        db, ArticleCreate(title=title, url=f"http://kg-test/{title}", source="test", summary=summary)
    )
    return article


def test_extract_entities_matches_gazetteer_terms():
    matches = extract_entities(
        "Moderna reports new mRNA vaccine data for semaglutide-adjacent obesity research"
    )
    names = {m.entity.name for m in matches}
    assert "Moderna" in names
    assert "semaglutide" in names
    assert "obesity" in names


def test_extract_entities_prefers_longer_alias_over_substring():
    matches = extract_entities("New non-Hodgkin lymphoma trial shows durable responses")
    names = [m.entity.name for m in matches]
    assert names.count("non-Hodgkin lymphoma") == 1


def test_extract_entities_empty_text():
    assert extract_entities("") == []
    assert extract_entities("nothing biotech-related here at all") == []


def test_extract_for_article_persists_mentions_and_marks_processed(db_session):
    article = _make_article(
        db_session,
        "Vertex and Moderna partner on sickle cell disease gene therapy",
        "The companies will co-develop a CRISPR-based approach targeting BCL11A.",
    )
    entities = service.extract_for_article(db_session, article)
    names = {e.name for e in entities}
    assert {"Vertex Pharmaceuticals", "Moderna", "sickle cell disease", "BCL11A"} <= names

    db_session.refresh(article)
    assert article.kg_extracted_at is not None


def test_extract_for_article_is_idempotent(db_session):
    article = _make_article(db_session, "Pfizer studies pembrolizumab combination in melanoma")
    first = service.extract_for_article(db_session, article)
    second = service.extract_for_article(db_session, article)
    assert {e.id for e in first} == {e.id for e in second}


def test_co_occurrence_creates_typed_relation(db_session):
    article = _make_article(
        db_session, "Pfizer advances pembrolizumab program", "New data on the PD-1 inhibitor from Pfizer."
    )
    entities = service.extract_for_article(db_session, article)
    by_type = {e.entity_type: e for e in entities}
    assert "company" in by_type and "drug" in by_type

    from app.kg import crud as kg_crud

    relations = kg_crud.get_entity_relations(db_session, by_type["company"].id)
    predicates = {r.predicate for r in relations}
    assert "develops" in predicates


def test_extract_missing_processes_only_unprocessed_articles(db_session):
    a1 = _make_article(db_session, "Moderna mRNA update")
    a2 = _make_article(db_session, "Unrelated local weather report")

    processed_first = service.extract_missing(db_session)
    assert processed_first == 2

    # Nothing left to process.
    processed_second = service.extract_missing(db_session)
    assert processed_second == 0


def test_kg_router_entities_and_extract_endpoint(client, db_session):
    _make_article(db_session, "Moderna and Pfizer both study mRNA vaccine platforms")

    resp = client.post("/kg/extract")
    assert resp.status_code == 200
    assert resp.json()["articles_processed"] == 1

    resp = client.get("/kg/entities", params={"entity_type": "company"})
    assert resp.status_code == 200
    names = {e["name"] for e in resp.json()}
    assert "Moderna" in names
    assert "Pfizer" in names


def test_kg_router_article_entities_and_relations(client, db_session):
    article = _make_article(db_session, "Vertex researches sickle cell disease gene therapy")
    service.extract_for_article(db_session, article)

    resp = client.get(f"/kg/articles/{article.id}/entities")
    assert resp.status_code == 200
    names = {e["name"] for e in resp.json()}
    assert "Vertex Pharmaceuticals" in names
    assert "sickle cell disease" in names

    company = next(e for e in resp.json() if e["entity_type"] == "company")
    resp = client.get(f"/kg/entities/{company['id']}/relations")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_kg_router_404s(client):
    assert client.get("/kg/entities/999999").status_code == 404
    assert client.get("/kg/entities/999999/relations").status_code == 404
    assert client.get("/kg/articles/999999/entities").status_code == 404
