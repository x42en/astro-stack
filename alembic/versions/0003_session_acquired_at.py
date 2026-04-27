"""Add acquired_at timestamp to astro_sessions.

Captures the actual acquisition time of the imaging session, extracted from
EXIF ``DateTimeOriginal`` for camera RAW files (NEF/CR2/ARW/DNG) or from
the ``DATE-OBS`` / ``DATE`` FITS header card.  When several light frames
are present, the earliest valid timestamp is used so the session is dated
to the start of the actual sky session, not to whenever it was uploaded.

The column is nullable: legacy sessions and corrupt files keep ``NULL``
and the UI falls back to ``created_at`` for display.

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add acquired_at column to astro_sessions."""
    op.add_column(
        "astro_sessions",
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Drop acquired_at column from astro_sessions."""
    op.drop_column("astro_sessions", "acquired_at")
