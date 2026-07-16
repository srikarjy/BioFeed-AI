from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import crud
from app.database import get_db
from app.ml import service as ml_service
from app.schemas import ArticleRead, ScoredArticleRead

router = APIRouter(prefix="/articles", tags=["articles"])


@router.get("", response_model=list[ArticleRead])
def list_articles(
    source: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    return crud.get_articles(db, source=source, limit=limit, offset=offset)


@router.get("/{article_id}", response_model=ArticleRead)
def read_article(article_id: int, db: Session = Depends(get_db)):
    article = crud.get_article(db, article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@router.get("/{article_id}/related", response_model=list[ScoredArticleRead])
def related_articles(
    article_id: int,
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    article = crud.get_article(db, article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    if article.embedding is None:
        # Article predates the embedder or slipped through a run; embed now so
        # the endpoint always works rather than returning nothing.
        ml_service.embed_articles(db, [article])
    results = crud.find_similar(db, article.embedding, limit=limit, exclude_id=article_id)
    return [
        ScoredArticleRead(
            **ArticleRead.model_validate(match).model_dump(), similarity=round(score, 4)
        )
        for match, score in results
    ]
