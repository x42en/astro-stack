"""Add frame_count_dark_flats to astro_sessions.

Dark-flats are calibration frames acquired with the same ISO/exposure as
the flats but with the optical path covered, used to subtract the
sensor's dark current from the flats master.  We track them as a first-
class frame type alongside darks/flats/bias.

Revision ID: 0013
Revises: 0012
Create Date: 2026-04-29
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add ``frame_count_dark_flats`` column with default 0."""
    op.add_column(
        "astro_sessions",
        sa.Column(
            "frame_count_dark_flats",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    """Drop the dark-flats column."""
    op.drop_column("astro_sessions", "frame_count_dark_flats")
