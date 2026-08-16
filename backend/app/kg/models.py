"""v0.8 knowledge graph: entities grounded in real ontology identifiers
(MONDO/HPO for disease, ChEMBL for drugs, HGNC for genes -- see
app/kg/gazetteer.json's _provenance field), mentions linking them to
articles, and relations inferred from same-article co-occurrence.
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Entity(Base):
    __tablename__ = "kg_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    external_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    external_id: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True, index=True)
    aliases: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class EntityMention(Base):
    """One row per (article, entity) pair -- repeated mentions of the same
    entity within one article collapse into a single mention, since what
    matters for the graph is "this article discusses this entity," not a
    count.
    """

    __tablename__ = "kg_entity_mentions"
    __table_args__ = (UniqueConstraint("article_id", "entity_id", name="uq_kg_mention_article_entity"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False, index=True)
    mention_text: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    entity: Mapped["Entity"] = relationship()


class EntityRelation(Base):
    """A directed edge between two entities, evidenced by one article where
    both were mentioned. `predicate` is chosen by entity-type pair (see
    app/kg/service.PREDICATE_BY_TYPE_PAIR) -- a same-article co-occurrence
    heuristic, not a verified causal claim. Multiple articles co-mentioning
    the same pair each add their own evidence row rather than
    deduplicating, so relation strength (row count) is queryable.
    """

    __tablename__ = "kg_entity_relations"
    __table_args__ = (
        UniqueConstraint(
            "subject_entity_id", "predicate", "object_entity_id", "evidence_article_id",
            name="uq_kg_relation",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_entity_id: Mapped[int] = mapped_column(
        ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    predicate: Mapped[str] = mapped_column(String(32), nullable=False)
    object_entity_id: Mapped[int] = mapped_column(
        ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    evidence_article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    subject: Mapped["Entity"] = relationship(foreign_keys=[subject_entity_id])
    object: Mapped["Entity"] = relationship(foreign_keys=[object_entity_id])
