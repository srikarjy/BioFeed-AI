from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import crud
from app.database import get_db
from app.schemas import ArticleRead

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
