from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import crud as article_crud
from app.database import get_db
from app.kg import crud, service
from app.kg.schemas import EntityRead, ExtractResult, RelationRead

router = APIRouter(prefix="/kg", tags=["knowledge graph"])


@router.get("/entities", response_model=list[EntityRead])
def list_entities(
    entity_type: str | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return crud.list_entities(db, entity_type=entity_type, q=q, limit=limit)


@router.get("/entities/{entity_id}", response_model=EntityRead)
def get_entity(entity_id: int, db: Session = Depends(get_db)):
    entity = crud.get_entity(db, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity


@router.get("/entities/{entity_id}/relations", response_model=list[RelationRead])
def get_entity_relations(
    entity_id: int,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    if not crud.get_entity(db, entity_id):
        raise HTTPException(status_code=404, detail="Entity not found")
    return crud.get_entity_relations(db, entity_id, limit=limit)


@router.get("/articles/{article_id}/entities", response_model=list[EntityRead])
def get_article_entities(article_id: int, db: Session = Depends(get_db)):
    if not article_crud.get_article(db, article_id):
        raise HTTPException(status_code=404, detail="Article not found")
    return crud.get_article_entities(db, article_id)


@router.post("/extract", response_model=ExtractResult)
def extract(limit: int = Query(default=500, ge=1, le=2000), db: Session = Depends(get_db)):
    """Manually trigger entity extraction for articles not yet processed.
    Also runs automatically at the end of every ingestion run (see
    app/ingestion/runner.run_and_record) -- this endpoint exists for
    backfilling articles ingested before v0.8, or re-running after a
    gazetteer change.
    """
    processed = service.extract_missing(db, limit=limit)
    return ExtractResult(articles_processed=processed)
