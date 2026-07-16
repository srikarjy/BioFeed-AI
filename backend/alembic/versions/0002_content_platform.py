"""content platform: article dedup fields + ingestion runs

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("articles", sa.Column("authors", sa.Text(), nullable=True))
    op.add_column("articles", sa.Column("doi", sa.String(length=255), nullable=True))
    op.add_column("articles", sa.Column("external_id", sa.String(length=255), nullable=True))
    op.add_column("articles", sa.Column("content_hash", sa.String(length=64), nullable=True))
    op.create_index("ix_articles_doi", "articles", ["doi"], unique=False)
    op.create_index("ix_articles_content_hash", "articles", ["content_hash"], unique=False)

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("added_total", sa.Integer(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("ingestion_runs")
    op.drop_index("ix_articles_content_hash", table_name="articles")
    op.drop_index("ix_articles_doi", table_name="articles")
    op.drop_column("articles", "content_hash")
    op.drop_column("articles", "external_id")
    op.drop_column("articles", "doi")
    op.drop_column("articles", "authors")
