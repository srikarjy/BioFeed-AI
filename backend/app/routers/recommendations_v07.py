"""v0.7 Enhanced recommendation endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app import crud
from app.auth.dependencies import require_self
from app.database import get_db
from app.ml.recommender_v07 import get_enhanced_feed
from app.models import User
from app.schemas import FeedResponse, FeedItem

router = APIRouter(prefix="/v0.7", tags=["v0.7 recommendations"])


@router.get("/users/{user_id}/feed", response_model=FeedResponse)
def get_enhanced_feed_endpoint(
    user_id: int,
    limit: int = Query(default=20, ge=1, le=50),
    db: Session = Depends(get_db),
    _: User = Depends(require_self),
):
    """Get enhanced feed using two-tower + LightGBM reranker (v0.7).

    Falls back to v0.6 logic if models not trained.
    """
    items, cold_start = get_enhanced_feed(db, user_id, limit)
    
    feed_items = [
        FeedItem(
            article=item[0],
            similarity=round(item[1], 4),
            reason=item[2],
        )
        for item in items
    ]
    
    return FeedResponse(
        items=feed_items,
        next_cursor=None,
        cold_start=cold_start,
    )


@router.post("/users/{user_id}/retrain-embedding")
def retrain_user_embedding(
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_self),
):
    """Trigger user embedding retrain (alias for v0.6 endpoint)."""
    embedding = crud.refresh_user_embedding(db, user_id)
    if not embedding:
        raise HTTPException(status_code=400, detail="Not enough interactions to compute embedding")
    
    return {
        "status": "updated", 
        "interaction_count": embedding.interaction_count,
        "version": "v0.7"
    }