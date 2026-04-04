"""Initial database schema.

Creates all three core tables:
- astro_sessions
- pipeline_jobs
- job_steps
- processing_profiles

Revision ID: 0001
Revises:
Create Date: 2026-04-04
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all application tables."""
    # ── astro_sessions ────────────────────────────────────────────────────────
    op.create_table(
        "astro_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False, index=True),
        sa.Column("inbox_path", sa.Text, nullable=False),
        sa.Column("status", sa.String(50), nullable=False, default="pending", index=True),
        sa.Column("input_format", sa.String(20), nullable=True),
        sa.Column("frame_count_lights", sa.Integer, nullable=False, default=0),
        sa.Column("frame_count_darks", sa.Integer, nullable=False, default=0),
        sa.Column("frame_count_flats", sa.Integer, nullable=False, default=0),
        sa.Column("frame_count_bias", sa.Integer, nullable=False, default=0),
        sa.Column("object_name", sa.String(255), nullable=True),
        sa.Column("ra", sa.Float, nullable=True),
        sa.Column("dec", sa.Float, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # ── processing_profiles ───────────────────────────────────────────────────
    op.create_table(
        "processing_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("owner_user_id", postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column("name", sa.String(255), nullable=False, index=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("config", postgresql.JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # ── pipeline_jobs ─────────────────────────────────────────────────────────
    op.create_table(
        "pipeline_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("astro_sessions.id"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "profile_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("processing_profiles.id"),
            nullable=True,
        ),
        sa.Column("profile_preset", sa.String(50), nullable=False, default="standard"),
        sa.Column("status", sa.String(50), nullable=False, default="pending", index=True),
        sa.Column("current_step", sa.String(100), nullable=True),
        sa.Column("arq_job_id", sa.String(255), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(100), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("output_fits_path", sa.Text, nullable=True),
        sa.Column("output_tiff_path", sa.Text, nullable=True),
        sa.Column("output_preview_path", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # ── job_steps ─────────────────────────────────────────────────────────────
    op.create_table(
        "job_steps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pipeline_jobs.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("step_name", sa.String(100), nullable=False, index=True),
        sa.Column("step_index", sa.Integer, nullable=False),
        sa.Column("status", sa.String(50), nullable=False, default="pending", index=True),
        sa.Column("attempt_count", sa.Integer, nullable=False, default=0),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(100), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("output_metadata", postgresql.JSONB, nullable=True),
    )


def downgrade() -> None:
    """Drop all application tables."""
    op.drop_table("job_steps")
    op.drop_table("pipeline_jobs")
    op.drop_table("processing_profiles")
    op.drop_table("astro_sessions")
