"""Alembic migration 0015 — LLM provider profiles in app_settings.

Adds the operator-configurable active LLM provider plus per-provider URL /
model overrides to the ``app_settings`` singleton. Secrets stay in env vars
only and are never persisted here. Also migrates the legacy default Ollama
model (``llama3.2``, text-only) to a vision-capable default (``qwen3-vl:8b``)
so the vision critic works out of the box on an Ollama stack.

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add LLM provider columns and fix the legacy Ollama model default."""
    op.add_column(
        "app_settings",
        sa.Column("llm_active_provider", sa.String(32), nullable=False, server_default="vllm"),
    )
    op.add_column(
        "app_settings",
        sa.Column("llm_ollama_url", sa.String(512), nullable=False, server_default=""),
    )
    op.add_column(
        "app_settings",
        sa.Column("llm_ollama_model", sa.String(128), nullable=False, server_default=""),
    )
    op.add_column(
        "app_settings",
        sa.Column("llm_vllm_base_url", sa.String(512), nullable=False, server_default=""),
    )
    op.add_column(
        "app_settings",
        sa.Column("llm_vllm_model", sa.String(128), nullable=False, server_default=""),
    )
    op.add_column(
        "app_settings",
        sa.Column("llm_kilo_model", sa.String(128), nullable=False, server_default=""),
    )
    # Legacy seed used a text-only model that cannot serve the vision critic.
    op.execute(
        sa.text(
            "UPDATE app_settings SET ollama_model = 'qwen3-vl:8b' WHERE ollama_model = 'llama3.2'"
        )
    )


def downgrade() -> None:
    """Drop the LLM provider columns."""
    op.drop_column("app_settings", "llm_kilo_model")
    op.drop_column("app_settings", "llm_vllm_model")
    op.drop_column("app_settings", "llm_vllm_base_url")
    op.drop_column("app_settings", "llm_ollama_model")
    op.drop_column("app_settings", "llm_ollama_url")
    op.drop_column("app_settings", "llm_active_provider")
