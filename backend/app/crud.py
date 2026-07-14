import hashlib
import re
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Article, IngestionRun
from app.schemas import ArticleCreate

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalize_title(title: str) -> str:
    """Collapse a title to a canonical form for cross-source matching.

    Lowercases, strips punctuation, and collapses whitespace so that the same
    paper surfaced by two sources with cosmetic differences hashes identically.
    """
    lowered = title.lower()
    stripped = _NON_ALNUM.sub(" ", lowered)
    return _WHITESPACE.sub(" ", stripped).strip()


def content_hash_for(title: str) -> str:
    return hashlib.sha256(normalize_title(title).encode("utf-8")).hexdigest()


def get_article(db: Session, article_id: int) -> Article | None:
    return db.get(Article, article_id)


def get_article_by_url(db: Session, url: str) -> Article | None:
    return db.execute(select(Article).where(Article.url == url)).scalar_one_or_none()


def get_article_by_doi(db: Session, doi: str) -> Article | None:
    return db.execute(select(Article).where(Article.doi == doi)).scalars().first()


def get_article_by_content_hash(db: Session, content_hash: str) -> Article | None:
    return (
        db.execute(select(Article).where(Article.content_hash == content_hash))
        .scalars()
        .first()
    )


def find_duplicate(db: Session, article: ArticleCreate, content_hash: str) -> Article | None:
    """Locate an existing article that represents the same content.

    Matching is tried from strongest to weakest signal: exact URL, then shared
    DOI (same paper, different landing page), then an identical normalized
    title (same paper from a source that exposes no DOI).
    """
    existing = get_article_by_url(db, article.url)
    if existing:
        return existing
    if article.doi:
        existing = get_article_by_doi(db, article.doi)
        if existing:
            return existing
    return get_article_by_content_hash(db, content_hash)


def get_articles(
    db: Session, source: str | None = None, limit: int = 50, offset: int = 0
) -> list[Article]:
    query = select(Article).order_by(Article.published_at.desc().nulls_last())
    if source:
        query = query.where(Article.source == source)
    query = query.limit(limit).offset(offset)
    return list(db.execute(query).scalars().all())


def create_article(db: Session, article: ArticleCreate) -> tuple[Article, bool]:
    """Get-or-create with cross-source dedup. Returns (article, created).

    Deduplicates on URL, DOI, and normalized-title hash (see find_duplicate).
    The URL column also carries a DB-level unique constraint as a safety net:
    two concurrent ingestion runs can both pass the existence check for the
    same URL before either commits. If that happens, the loser's commit hits
    the constraint; roll back and return the winner's row instead of leaving
    the session in an aborted-transaction state (which would break every
    subsequent query on this session).
    """
    content_hash = content_hash_for(article.title)
    existing = find_duplicate(db, article, content_hash)
    if existing:
        return existing, False

    db_article = Article(**article.model_dump(), content_hash=content_hash)
    db.add(db_article)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return get_article_by_url(db, article.url), False
    db.refresh(db_article)
    return db_article, True


def record_ingestion_run(
    db: Session,
    started_at: datetime,
    added: dict[str, int],
    errors: dict[str, str],
) -> IngestionRun:
    run = IngestionRun(
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        added_total=sum(added.values()),
        error_count=len(errors),
        detail={"added": added, "errors": errors},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_ingestion_runs(db: Session, limit: int = 20) -> list[IngestionRun]:
    return list(
        db.execute(
            select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(limit)
        )
        .scalars()
        .all()
    )
