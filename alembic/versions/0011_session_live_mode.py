"""Add session mode (batch / live) and live frame counter.

Introduces the ``mode`` column on ``astro_sessions`` to discriminate
between batch sessions (frames uploaded then processed) and live
sessions (frames pushed one at a time during acquisition with a
live-stacking preview).  ``live_frame_count`` tracks how many frames
have been ingested by the live stacker so far.

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-29
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add ``mode`` (default 'batch') and ``live_frame_count`` columns."""
    op.add_column(
        "astro_sessions",
        sa.Column(
            "mode",
            sa.String(20),
            nullable=False,
            server_default="batch",
        ),
    )
    op.add_column(
        "astro_sessions",
        sa.Column(
            "live_frame_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.create_index(
        "ix_astro_sessions_mode",
        "astro_sessions",
        ["mode"],
    )


def downgrade() -> None:
    """Drop the live-mode columns and their index."""
    op.drop_index("ix_astro_sessions_mode", table_name="astro_sessions")
    op.drop_column("astro_sessions", "live_frame_count")
    op.drop_column("astro_sessions", "mode")
