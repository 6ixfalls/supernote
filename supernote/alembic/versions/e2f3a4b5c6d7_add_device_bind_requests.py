"""add device bind requests

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
Create Date: 2026-09-04 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e2f3a4b5c6d7"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "device_bind_requests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("equipment_no", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("total_capacity", sa.String(), nullable=False),
        sa.Column("create_time", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "equipment_no",
            name="uq_device_bind_requests_user_equipment",
        ),
    )
    op.create_index(
        op.f("ix_device_bind_requests_user_id"),
        "device_bind_requests",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_device_bind_requests_equipment_no"),
        "device_bind_requests",
        ["equipment_no"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_device_bind_requests_equipment_no"),
        table_name="device_bind_requests",
    )
    op.drop_index(
        op.f("ix_device_bind_requests_user_id"),
        table_name="device_bind_requests",
    )
    op.drop_table("device_bind_requests")
