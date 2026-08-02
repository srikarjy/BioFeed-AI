"""merge 0004 and 0005

Revision ID: 8c2f51308361
Revises: 0004, 0005
Create Date: 2026-08-01 19:09:29.128496

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '8c2f51308361'
down_revision: Union[str, None] = ('0004', '0005')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
