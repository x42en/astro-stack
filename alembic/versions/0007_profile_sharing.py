"""Add sharing flags to processing_profiles.

Adds:
* ``is_shared`` — boolean (NOT NULL, default false) marking profiles that
  the owner has explicitly published for use by other authenticated users.
* ``shared_at`` — timestamptz set the first time the owner toggles
  ``is_shared`` on; never cleared automatically.

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "processing_profiles",
        sa.Column(
            "is_shared",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "processing_profiles",
        sa.Column("shared_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_processing_profiles_is_shared",
        "processing_profiles",
        ["is_shared"],
    )


def downgrade() -> None:
    op.drop_index("ix_processing_profiles_is_shared", table_name="processing_profiles")
    op.drop_column("processing_profiles", "shared_at")
    op.drop_column("processing_profiles", "is_shared")
