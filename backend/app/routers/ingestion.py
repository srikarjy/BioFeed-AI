from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.ingestion.rss import run_ingestion
from app.schemas import IngestResult

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/run", response_model=IngestResult)
def trigger_ingestion(db: Session = Depends(get_db)):
    added, errors = run_ingestion(db)
    return IngestResult(added=added, errors=errors)
