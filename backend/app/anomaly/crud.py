from sqlalchemy import select
from sqlalchemy.orm import Session

from app.anomaly.models import AnomalyEvent


def get_event(db: Session, event_id: int) -> AnomalyEvent | None:
    return db.get(AnomalyEvent, event_id)


def get_existing(db: Session, article_id: int, kind: str) -> AnomalyEvent | None:
    return db.execute(
        select(AnomalyEvent).where(
            AnomalyEvent.article_id == article_id, AnomalyEvent.kind == kind
        )
    ).scalars().first()


def list_events(db: Session, limit: int = 50) -> list[AnomalyEvent]:
    return list(
        db.execute(
            select(AnomalyEvent).order_by(AnomalyEvent.detected_at.desc()).limit(limit)
        )
        .scalars()
        .all()
    )
