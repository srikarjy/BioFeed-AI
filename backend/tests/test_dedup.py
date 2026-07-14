from app import crud
from app.schemas import ArticleCreate


def test_normalize_title_collapses_cosmetic_differences():
    a = crud.normalize_title("CRISPR-Cas9: A New Era!")
    b = crud.normalize_title("crispr cas9   a new era")
    assert a == b


def test_dedup_by_url(db_session):
    art = ArticleCreate(title="First", url="https://example.com/x", source="A")
    _, created_first = crud.create_article(db_session, art)
    _, created_second = crud.create_article(db_session, art)

    assert created_first is True
    assert created_second is False
    assert len(crud.get_articles(db_session)) == 1


def test_dedup_by_doi_across_sources(db_session):
    """Same paper from two sources under different URLs, sharing a DOI."""
    pubmed = ArticleCreate(
        title="A Trial Readout",
        url="https://pubmed.ncbi.nlm.nih.gov/123/",
        source="PubMed",
        doi="10.1101/2024.01.01.573000",
    )
    biorxiv = ArticleCreate(
        title="A completely different headline",
        url="https://doi.org/10.1101/2024.01.01.573000",
        source="bioRxiv",
        doi="10.1101/2024.01.01.573000",
    )
    _, first = crud.create_article(db_session, pubmed)
    existing, second = crud.create_article(db_session, biorxiv)

    assert first is True
    assert second is False
    assert existing.source == "PubMed"
    assert len(crud.get_articles(db_session)) == 1


def test_dedup_by_title_hash_when_no_doi(db_session):
    a = ArticleCreate(
        title="Novel Antibody Shows Promise",
        url="https://news-a.example.com/story",
        source="SourceA",
    )
    b = ArticleCreate(
        title="novel antibody shows promise!",
        url="https://news-b.example.com/other",
        source="SourceB",
    )
    _, first = crud.create_article(db_session, a)
    _, second = crud.create_article(db_session, b)

    assert first is True
    assert second is False
    assert len(crud.get_articles(db_session)) == 1


def test_distinct_articles_are_not_deduped(db_session):
    a = ArticleCreate(title="Paper One", url="https://example.com/1", source="A")
    b = ArticleCreate(title="Paper Two", url="https://example.com/2", source="A")
    crud.create_article(db_session, a)
    crud.create_article(db_session, b)

    assert len(crud.get_articles(db_session)) == 2
