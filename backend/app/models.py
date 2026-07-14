from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True, index=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Semicolon-joined author names; kept as free text since author lists vary
    # wildly in shape across PubMed, bioRxiv, and RSS.
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Digital Object Identifier, normalized to bare form (e.g. "10.1101/2024.01.01.573000").
    # Nullable because most RSS news items don't carry one.
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    # Source-native identifier (PubMed PMID, bioRxiv id, ...) for provenance.
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # sha256 of the normalized title, used as a last-resort dedup key when the
    # same paper surfaces from two sources under different URLs and no shared DOI.
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class IngestionRun(Base):
    """One execution of the ingestion pipeline, recorded for observability.

    Scheduled runs (Celery beat) and manual /ingest/run calls both write a row
    here so the API can expose ingestion history and freshness.
    """

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    added_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Per-source breakdown: {"added": {...}, "errors": {...}}
    detail: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
