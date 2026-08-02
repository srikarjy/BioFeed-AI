from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AnomalyEvent(Base):
    """A candidate early-signal event surfaced by the anomaly pipeline.

    Kept in its own table/module rather than folded into Article: an article
    can accumulate multiple events over time (different kinds, re-detected as
    more corroborating coverage arrives), and this stays a downstream,
    removable layer rather than a core ingestion concept.
    """

    __tablename__ = "anomaly_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), nullable=False, index=True)
    # Detector that raised this event, e.g. "cross_source_burst". Free text so
    # new detector kinds don't require a migration.
    kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    # Detector-specific evidence (related article ids/sources, thresholds
    # used, etc.) so an explanation consumer has something to ground on.
    detail: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
