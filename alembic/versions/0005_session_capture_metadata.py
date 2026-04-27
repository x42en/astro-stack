"""Add aggregated capture metadata to astro_sessions.

Adds a single nullable ``capture_metadata`` JSONB column on
:class:`AstroSession` that stores per-session aggregated EXIF / FITS
header values (ISO, exposure, focal length, camera, …) extracted at
ingestion time by :func:`app.pipeline.utils.exif.extract_capture_metadata`.

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers
revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "astro_sessions",
        sa.Column("capture_metadata", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("astro_sessions", "capture_metadata")
