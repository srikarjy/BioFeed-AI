"""Embedding lifecycle: what gets embedded and when.

Articles are embedded from title + summary (one vector per article — at this
granularity chunking buys nothing). Ingestion calls ``embed_missing`` after
each run so new articles become searchable immediately, and the same function
doubles as a backfill for articles that predate the embedding column.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.embeddings import get_embedder
from app.models import Article

_BATCH_SIZE = 32
# Generous cap; sentence-transformers truncates to the model's max sequence
# length anyway, this just bounds the work of building the string.
_MAX_CHARS = 2000


def embedding_text(article: Article) -> str:
    parts = [article.title]
    if article.summary:
        parts.append(article.summary)
    return ". ".join(parts)[:_MAX_CHARS]


def embed_articles(db: Session, articles: list[Article]) -> int:
    """Embed the given articles in batches and commit. Returns count embedded."""
    embedder = get_embedder()
    embedded = 0
    for start in range(0, len(articles), _BATCH_SIZE):
        batch = articles[start : start + _BATCH_SIZE]
        vectors = embedder.embed_texts([embedding_text(a) for a in batch])
        for article, vector in zip(batch, vectors):
            article.embedding = vector
        db.commit()
        embedded += len(batch)
    return embedded


def embed_missing(db: Session, limit: int | None = None) -> int:
    """Embed articles that have no embedding yet. Returns count embedded."""
    query = select(Article).where(Article.embedding.is_(None))
    if limit is not None:
        query = query.limit(limit)
    articles = list(db.execute(query).scalars().all())
    if not articles:
        return 0
    return embed_articles(db, articles)
