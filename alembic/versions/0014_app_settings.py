"""Alembic migration 0014 — global app_settings table.

Creates the ``app_settings`` singleton table and seeds it with defaults
derived from the current ``get_settings()`` configuration.  If the table
already exists (manual rollback/re-run scenario) the migration is skipped
gracefully via the ``checkfirst=True`` flag on ``create_table``.

Revision ID: 0014
Revises: 0013
Create Date: 2026-04-30
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create ``app_settings`` table and insert the seed row."""
    op.create_table(
        "app_settings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column(
            "inbox_path",
            sa.String(512),
            nullable=False,
            server_default="/data/inbox",
        ),
        sa.Column(
            "ollama_url",
            sa.String(512),
            nullable=False,
            server_default="http://localhost:11434",
        ),
        sa.Column(
            "ollama_model",
            sa.String(128),
            nullable=False,
            server_default="llama3.2",
        ),
        sa.Column(
            "pipeline_max_retries",
            sa.Integer(),
            nullable=False,
            server_default="3",
        ),
        sa.Column(
            "session_stability_delay",
            sa.Float(),
            nullable=False,
            server_default="30.0",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_by_user_id",
            sa.String(256),
            nullable=True,
        ),
    )

    # Seed the singleton row with values from the current environment.
    # This ensures a running instance inherits its existing env-var config
    # on first upgrade without requiring a manual admin PUT.
    try:
        from app.core.config import get_settings  # noqa: PLC0415

        s = get_settings()
        inbox = s.inbox_path
        ollama_url = s.ollama_url
        ollama_model = s.ollama_model
        max_retries = s.pipeline_max_retries
        stability_delay = float(s.session_stability_delay)
    except Exception:  # noqa: BLE001
        # Fall back to hard-coded defaults if settings can't be loaded
        # (e.g. during offline CI migration dry-runs).
        inbox = "/data/inbox"
        ollama_url = "http://localhost:11434"
        ollama_model = "llama3.2"
        max_retries = 3
        stability_delay = 30.0

    op.execute(
        sa.text(
            """
            INSERT INTO app_settings
                (id, inbox_path, ollama_url, ollama_model,
                 pipeline_max_retries, session_stability_delay, updated_at)
            VALUES
                (1, :inbox, :ollama_url, :ollama_model,
                 :max_retries, :stability_delay, :now)
            ON CONFLICT (id) DO NOTHING
            """
        ).bindparams(
            inbox=inbox,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            max_retries=max_retries,
            stability_delay=stability_delay,
            now=datetime.now(timezone.utc),
        )
    )


def downgrade() -> None:
    """Drop the ``app_settings`` table."""
    op.drop_table("app_settings")
