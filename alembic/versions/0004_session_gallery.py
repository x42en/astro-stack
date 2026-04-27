"""Add public gallery to astro_sessions and create download audit table.

Adds four new nullable / defaulted columns to :class:`AstroSession`:

* ``is_in_gallery`` — owner toggled the session as publicly visible.
* ``gallery_published_at`` — when it was first toggled on.
* ``gallery_author_name`` — display credit for the gallery card.
* ``gallery_download_count`` — bumped each time a download is served.

Creates the :class:`GalleryDownload` table that records each public
download request (email + format + IP + timestamp).  The table is purely
an audit trail: it does *not* gate downloads.  Downloads are authorised
via short-lived HMAC-signed URLs returned by the request endpoint.

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

# revision identifiers
revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add gallery columns and create the download audit table."""
    op.add_column(
        "astro_sessions",
        sa.Column(
            "is_in_gallery",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "astro_sessions",
        sa.Column("gallery_published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "astro_sessions",
        sa.Column("gallery_author_name", sa.String(length=120), nullable=True),
    )
    op.add_column(
        "astro_sessions",
        sa.Column(
            "gallery_download_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.create_index(
        "ix_astro_sessions_is_in_gallery",
        "astro_sessions",
        ["is_in_gallery"],
    )

    op.create_table(
        "astro_gallery_downloads",
        sa.Column("id", PG_UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            PG_UUID(as_uuid=True),
            sa.ForeignKey("astro_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("format", sa.String(length=10), nullable=False),
        sa.Column("requester_ip", sa.String(length=45), nullable=True),
        sa.Column(
            "requested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_astro_gallery_downloads_session_id",
        "astro_gallery_downloads",
        ["session_id"],
    )


def downgrade() -> None:
    """Drop gallery columns and the download audit table."""
    op.drop_index(
        "ix_astro_gallery_downloads_session_id",
        table_name="astro_gallery_downloads",
    )
    op.drop_table("astro_gallery_downloads")
    op.drop_index("ix_astro_sessions_is_in_gallery", table_name="astro_sessions")
    op.drop_column("astro_sessions", "gallery_download_count")
    op.drop_column("astro_sessions", "gallery_author_name")
    op.drop_column("astro_sessions", "gallery_published_at")
    op.drop_column("astro_sessions", "is_in_gallery")
