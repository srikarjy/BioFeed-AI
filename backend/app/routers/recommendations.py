from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import crud
from app.database import get_db
from app.schemas import (
    FeedResponse, FeedItem, InteractionCreate, InteractionRead, 
    UserCreate, UserRead, InteractionType
)

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserRead)
def create_or_get_user(user: UserCreate, db: Session = Depends(get_db)):
    """Get existing user or create new one (by email/Apple/Google ID)."""
    db_user = crud.get_or_create_user(
        db, 
        email=user.email,
        apple_user_id=user.apple_user_id,
        google_user_id=user.google_user_id,
        display_name=user.display_name,
    )
    return db_user


@router.get("/{user_id}", response_model=UserRead)
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("/{user_id}/interactions", response_model=InteractionRead)
def record_interaction(
    user_id: int, 
    interaction: InteractionCreate, 
    db: Session = Depends(get_db)
):
    """Record a user interaction (read, bookmark, like, hide, search)."""
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    article = crud.get_article(db, interaction.article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    db_interaction = crud.create_interaction(
        db,
        user_id=user_id,
        article_id=interaction.article_id,
        interaction_type=interaction.interaction_type,
        read_time_seconds=interaction.read_time_seconds,
        search_query=interaction.search_query,
    )
    
    # Trigger embedding refresh for positive signals
    if interaction.interaction_type in (InteractionType.READ, InteractionType.BOOKMARK, InteractionType.LIKE):
        crud.refresh_user_embedding(db, user_id)
    
    return db_interaction


@router.get("/{user_id}/interactions", response_model=list[InteractionRead])
def list_interactions(
    user_id: int,
    interaction_type: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return crud.get_user_interactions(db, user_id, interaction_type, limit)


@router.post("/{user_id}/refresh-embedding")
def refresh_embedding(user_id: int, db: Session = Depends(get_db)):
    """Manually trigger user embedding recomputation."""
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    embedding = crud.refresh_user_embedding(db, user_id)
    if not embedding:
        raise HTTPException(status_code=400, detail="Not enough interactions to compute embedding")
    
    return {"status": "updated", "interaction_count": embedding.interaction_count}


@router.get("/{user_id}/feed", response_model=FeedResponse)
def get_feed(
    user_id: int,
    limit: int = Query(default=20, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Get personalized feed for user.
    
    Uses user embedding if available (warm start), falls back to recent articles (cold start).
    """
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    items, cold_start = crud.get_personalized_feed(db, user_id, limit, offset)
    
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
        next_cursor=offset + len(feed_items) if len(feed_items) == limit else None,
        cold_start=cold_start,
    )