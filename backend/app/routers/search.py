from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app import crud
from app.database import get_db
from app.ml.embeddings import get_embedder
from app.schemas import ArticleRead, ScoredArticleRead

router = APIRouter(tags=["search"])


@router.get("/search", response_model=list[ScoredArticleRead])
def semantic_search(
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Embed the query and return the nearest articles by cosine similarity."""
    query_embedding = get_embedder().embed_texts([q])[0]
    results = crud.find_similar(db, query_embedding, limit=limit)
    return [
        ScoredArticleRead(
            **ArticleRead.model_validate(article).model_dump(), similarity=round(score, 4)
        )
        for article, score in results
    ]
