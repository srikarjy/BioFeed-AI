from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.anomaly import crud as anomaly_crud
from app.anomaly import detector
from app.anomaly.schemas import AnomalyEventRead
from app.database import get_db

router = APIRouter(prefix="/internal/anomaly", tags=["anomaly"])


@router.post("/detect", response_model=list[AnomalyEventRead])
def trigger_detection(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Scan recent articles and persist any new candidate events found."""
    return detector.detect_recent(db, limit=limit)


@router.get("/events", response_model=list[AnomalyEventRead])
def list_events(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    return anomaly_crud.list_events(db, limit=limit)
