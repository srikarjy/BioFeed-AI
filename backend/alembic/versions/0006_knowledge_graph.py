"""v0.8: knowledge graph - entities, mentions, relations; kg_extracted_at on articles

Revision ID: 0006
Revises: 8c2f51308361
Create Date: 2026-08-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0006"
down_revision: Union[str, None] = "8c2f51308361"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("articles", sa.Column("kg_extracted_at", sa.DateTime(timezone=True), nullable=True))

    op.create_table(
        "kg_entities",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, index=True),
        sa.Column("entity_type", sa.String(32), nullable=False, index=True),
        sa.Column("external_source", sa.String(32), nullable=True),
        sa.Column("external_id", sa.String(64), nullable=True, unique=True, index=True),
        sa.Column("aliases", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "kg_entity_mentions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("article_id", sa.Integer, sa.ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("entity_id", sa.Integer, sa.ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("mention_text", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("article_id", "entity_id", name="uq_kg_mention_article_entity"),
    )

    op.create_table(
        "kg_entity_relations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("subject_entity_id", sa.Integer, sa.ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("predicate", sa.String(32), nullable=False),
        sa.Column("object_entity_id", sa.Integer, sa.ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("evidence_article_id", sa.Integer, sa.ForeignKey("articles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint(
            "subject_entity_id", "predicate", "object_entity_id", "evidence_article_id", name="uq_kg_relation"
        ),
    )


def downgrade() -> None:
    op.drop_table("kg_entity_relations")
    op.drop_table("kg_entity_mentions")
    op.drop_table("kg_entities")
    op.drop_column("articles", "kg_extracted_at")
