"""semantic retrieval: pgvector extension + article embedding column

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("articles", sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True))
    # HNSW over IVFFlat: builds fine on an empty/small table (IVFFlat needs
    # training data) and stays accurate as the corpus grows.
    op.create_index(
        "ix_articles_embedding_hnsw",
        "articles",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_articles_embedding_hnsw", table_name="articles")
    op.drop_column("articles", "embedding")
