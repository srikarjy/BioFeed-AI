from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Article
from app.schemas import ArticleCreate


def get_article(db: Session, article_id: int) -> Article | None:
    return db.get(Article, article_id)


def get_article_by_url(db: Session, url: str) -> Article | None:
    return db.execute(select(Article).where(Article.url == url)).scalar_one_or_none()


def get_articles(
    db: Session, source: str | None = None, limit: int = 50, offset: int = 0
) -> list[Article]:
    query = select(Article).order_by(Article.published_at.desc().nulls_last())
    if source:
        query = query.where(Article.source == source)
    query = query.limit(limit).offset(offset)
    return list(db.execute(query).scalars().all())


def create_article(db: Session, article: ArticleCreate) -> tuple[Article, bool]:
    """Get-or-create by URL. Returns (article, created)."""
    existing = get_article_by_url(db, article.url)
    if existing:
        return existing, False

    db_article = Article(**article.model_dump())
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return db_article, True
