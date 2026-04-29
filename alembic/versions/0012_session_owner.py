"""Add owner_id to astro_sessions for per-user scoping.

Introduces a nullable ``owner_id`` column referencing the user that
created the session.  Nullable so legacy sessions created before this
migration keep working (they simply remain unscoped).  All sessions
created from this point forward must carry an owner_id when the caller
is identifiable (auth enabled or X-Mock-User header present).

Revision ID: 0012
Revises: 0011
Create Date: 2026-04-29
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add ``owner_id`` (nullable UUID) and a btree index for fast filtering."""
    op.add_column(
        "astro_sessions",
        sa.Column(
            "owner_id",
            PG_UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_astro_sessions_owner_id",
        "astro_sessions",
        ["owner_id"],
    )


def downgrade() -> None:
    """Drop the owner column and its index."""
    op.drop_index("ix_astro_sessions_owner_id", table_name="astro_sessions")
    op.drop_column("astro_sessions", "owner_id")
