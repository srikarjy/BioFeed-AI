"""anomaly detection: anomaly_events table

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "anomaly_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "article_id",
            sa.Integer(),
            sa.ForeignKey("articles.id"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_anomaly_events_article_id", "anomaly_events", ["article_id"])
    op.create_index("ix_anomaly_events_kind", "anomaly_events", ["kind"])


def downgrade() -> None:
    op.drop_index("ix_anomaly_events_kind", table_name="anomaly_events")
    op.drop_index("ix_anomaly_events_article_id", table_name="anomaly_events")
    op.drop_table("anomaly_events")
