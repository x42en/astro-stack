"""Create followed_objects table.

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-28
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the followed_objects table, its index, and unique constraint."""
    op.create_table(
        "followed_objects",
        sa.Column("id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("owner_user_id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("catalog_id", sa.String(64), nullable=False),
        sa.Column("note", sa.String(500), nullable=True),
        sa.Column(
            "notify_when_visible",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "owner_user_id", "catalog_id", name="uq_followed_owner_catalog"
        ),
    )
    op.create_index(
        "ix_followed_objects_owner_user_id",
        "followed_objects",
        ["owner_user_id"],
    )


def downgrade() -> None:
    """Drop the followed_objects table, its index, and unique constraint."""
    op.drop_index("ix_followed_objects_owner_user_id", table_name="followed_objects")
    op.drop_constraint(
        "uq_followed_owner_catalog", "followed_objects", type_="unique"
    )
    op.drop_table("followed_objects")
