"""v0.6: recommendation engine - users, interactions, user embeddings

Revision ID: 0005
Revises: 0003
Create Date: 2026-08-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "0005"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    # Users table - minimal for v0.6, can extend with OAuth later
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String(255), nullable=True, unique=True, index=True),
        sa.Column("apple_user_id", sa.String(255), nullable=True, unique=True, index=True),
        sa.Column("google_user_id", sa.String(255), nullable=True, unique=True, index=True),
        sa.Column("display_name", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Interaction types: read, bookmark, like, hide, search
    op.create_table(
        "user_interactions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("article_id", sa.Integer, sa.ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("interaction_type", sa.String(32), nullable=False, index=True),  # read, bookmark, like, hide, search
        sa.Column("read_time_seconds", sa.Integer, nullable=True),  # for 'read' type
        sa.Column("search_query", sa.Text, nullable=True),  # for 'search' type
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    # Composite index for common queries
    op.create_index(
        "ix_user_interactions_user_article_type",
        "user_interactions",
        ["user_id", "article_id", "interaction_type"],
    )

    # User embeddings - one per user, updated from interaction history
    op.create_table(
        "user_embeddings",
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("interaction_count", sa.Integer, nullable=False, default=0),
    )
    op.create_index(
        "ix_user_embeddings_embedding_hnsw",
        "user_embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_user_embeddings_embedding_hnsw", table_name="user_embeddings")
    op.drop_table("user_embeddings")
    op.drop_index("ix_user_interactions_user_article_type", table_name="user_interactions")
    op.drop_table("user_interactions")
    op.drop_table("users")