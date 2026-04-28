"""Add denoise engine selection fields to existing profiles.

The new fields live inside the JSONB ``config`` column of
``processing_profiles`` (no schema change needed, only a data backfill so
existing user profiles surface sane defaults in the UI):

* ``denoise_engine`` — ``"cosmic_clarity"`` (default) or ``"graxpert"``.
* ``denoise_graxpert_ai_model`` — semver string, default ``"3.0.2"``.
* ``denoise_graxpert_batch_size`` — integer 1–32, default ``4``.

Existing profiles keep ``cosmic_clarity`` as the engine so behaviour is
unchanged after the migration.

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-28
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers
revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Merge the new keys into every existing profile config that doesn't
    # already define them. ``jsonb || jsonb`` keeps the right-hand value on
    # collisions, so we use ``NOT (config ? 'key')`` guards to avoid
    # clobbering a user override.
    op.execute(
        """
        UPDATE processing_profiles
        SET config = config || jsonb_build_object('denoise_engine', 'cosmic_clarity')
        WHERE NOT (config ? 'denoise_engine')
        """
    )
    op.execute(
        """
        UPDATE processing_profiles
        SET config = config || jsonb_build_object('denoise_graxpert_ai_model', '3.0.2')
        WHERE NOT (config ? 'denoise_graxpert_ai_model')
        """
    )
    op.execute(
        """
        UPDATE processing_profiles
        SET config = config || jsonb_build_object('denoise_graxpert_batch_size', 4)
        WHERE NOT (config ? 'denoise_graxpert_batch_size')
        """
    )


def downgrade() -> None:
    # Strip the keys back out of the JSONB config.
    op.execute(
        """
        UPDATE processing_profiles
        SET config = config
            - 'denoise_engine'
            - 'denoise_graxpert_ai_model'
            - 'denoise_graxpert_batch_size'
        """
    )
