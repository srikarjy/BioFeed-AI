from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.ml.embeddings import EMBEDDING_DIM


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
    # Semantic embedding of title + summary (see app.ml.service). Nullable so
    # ingestion never blocks on the embedder; embed_missing backfills gaps.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
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


class User(Base):
    """Minimal user model for v0.6. Extensible for OAuth (Apple/Google) in v0.3."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True, index=True)
    apple_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True, index=True)
    google_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    interactions: Mapped[list["UserInteraction"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    embedding: Mapped["UserEmbedding | None"] = relationship(back_populates="user", cascade="all, delete-orphan", uselist=False)


class UserInteraction(Base):
    """User interaction signals for recommendation engine.
    
    Types: read, bookmark, like, hide, search
    """

    __tablename__ = "user_interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True)
    interaction_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    read_time_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    search_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    user: Mapped[User] = relationship(back_populates="interactions")
    article: Mapped[Article] = relationship()

    __table_args__ = (
        Index("ix_user_interactions_user_article_type", "user_id", "article_id", "interaction_type"),
    )


class UserEmbedding(Base):
    """Per-user embedding computed from interaction history.
    
    Updated periodically or on-demand from positive signals (reads, bookmarks, likes).
    """

    __tablename__ = "user_embeddings"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    interaction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Relationships
    user: Mapped[User] = relationship(back_populates="embedding")

    __table_args__ = (
        Index(
            "ix_user_embeddings_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
