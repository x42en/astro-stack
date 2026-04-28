"""Create observation_sites table.

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-28
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the observation_sites table and its index."""
    op.create_table(
        "observation_sites",
        sa.Column("id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("owner_user_id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("elevation_m", sa.Float(), nullable=False, server_default="0"),
        sa.Column("timezone", sa.String(64), nullable=False, server_default="UTC"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_observation_sites_owner_user_id",
        "observation_sites",
        ["owner_user_id"],
    )


def downgrade() -> None:
    """Drop the observation_sites table and its index."""
    op.drop_index("ix_observation_sites_owner_user_id", table_name="observation_sites")
    op.drop_table("observation_sites")
