from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app import crud
from app.database import get_db
from app.ingestion.runner import run_and_record
from app.schemas import IngestionRunRead, IngestResult

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/run", response_model=IngestResult)
def trigger_ingestion(db: Session = Depends(get_db)):
    run = run_and_record(db)
    return IngestResult(added=run.detail["added"], errors=run.detail["errors"])


@router.get("/runs", response_model=list[IngestionRunRead])
def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return crud.get_ingestion_runs(db, limit=limit)
