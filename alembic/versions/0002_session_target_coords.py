"""Add user-supplied target_ra / target_dec to astro_sessions.

These columns store the J2000 coordinates of the imaging target as supplied
by the user at session creation time. They are passed to ASTAP as a hint to
turn an expensive blind solve (180° radius) into a fast targeted solve
(≈5° radius), which both speeds up plate solving and dramatically improves
its success rate on noisy DSLR stacks.

Distinct from the existing ``ra`` / ``dec`` columns, which are populated by
ASTAP **after** a successful plate-solve and represent the actual centre of
the stacked frame.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add target_ra and target_dec columns to astro_sessions."""
    op.add_column(
        "astro_sessions",
        sa.Column("target_ra", sa.Float, nullable=True),
    )
    op.add_column(
        "astro_sessions",
        sa.Column("target_dec", sa.Float, nullable=True),
    )


def downgrade() -> None:
    """Drop target_ra and target_dec columns from astro_sessions."""
    op.drop_column("astro_sessions", "target_dec")
    op.drop_column("astro_sessions", "target_ra")
