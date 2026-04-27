"""Add profile_snapshot to pipeline_jobs.

Adds a single nullable ``profile_snapshot`` JSONB column on
:class:`PipelineJob` so we persist the full resolved
:class:`ProcessingProfileConfig` at job creation time.  This guarantees
that the UI can always show the parameters that produced a render even
if the preset definition or the saved advanced profile is edited later.

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers
revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "pipeline_jobs",
        sa.Column("profile_snapshot", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("pipeline_jobs", "profile_snapshot")
