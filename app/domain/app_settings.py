"""Global application settings domain model.

A single row (``id = 1``) stores operator-configurable settings that should
be shared across all API and worker containers rather than duplicated in each
instance's environment.  The row is seeded during the Alembic migration and
can be updated at runtime by administrators via ``PUT /api/v1/settings``.

Connection-level config (database URL, Redis URL, auth secrets, …) intentionally
stays in environment variables — these must be known before the app can reach the
database at all.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlmodel import Column, Field, SQLModel


class AppSettings(SQLModel, table=True):
    """Single-row table holding global operational settings.

    Attributes:
        id: Always 1 — enforces the singleton pattern at DB level.
        inbox_path: Host path watched by the file watcher for new sessions.
        ollama_url: Base URL of the Ollama REST API (local LLM stack).
        ollama_model: Ollama model name for vision-critic inference.
        llm_active_provider: Active LLM provider (``ollama`` | ``vllm`` | ``kilo``).
        llm_ollama_url: Operator-configured Ollama base URL override.
        llm_ollama_model: Operator-configured Ollama model override.
        llm_vllm_base_url: Operator-configured vLLM endpoint override.
        llm_vllm_model: Operator-configured vLLM model override.
        llm_kilo_model: Operator-configured Kilo model id (``provider/model``).
        pipeline_max_retries: Default maximum retry count for failed pipeline steps.
        session_stability_delay: Seconds to wait after the last file write before
            considering a new session stable and ready for auto-processing.
        updated_at: Timestamp of the last write — displayed to admins in the UI.
        updated_by_user_id: Opaque user identifier of the last writer (from JWT sub
            or mock username).  Nullable so the seed row has no attribution.
    """

    __tablename__ = "app_settings"

    id: int = Field(
        default=1,
        sa_column=Column(Integer, primary_key=True, autoincrement=False),
    )

    inbox_path: str = Field(
        default="/data/inbox",
        sa_column=Column(String(512), nullable=False, server_default="/data/inbox"),
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        sa_column=Column(
            String(512), nullable=False, server_default="http://localhost:11434"
        ),
    )
    ollama_model: str = Field(
        default="qwen3-vl:8b",
        sa_column=Column(String(128), nullable=False, server_default="qwen3-vl:8b"),
    )
    llm_active_provider: str = Field(
        default="vllm",
        sa_column=Column(String(32), nullable=False, server_default="vllm"),
    )
    llm_ollama_url: str = Field(
        default="",
        sa_column=Column(String(512), nullable=False, server_default=""),
    )
    llm_ollama_model: str = Field(
        default="",
        sa_column=Column(String(128), nullable=False, server_default=""),
    )
    llm_vllm_base_url: str = Field(
        default="",
        sa_column=Column(String(512), nullable=False, server_default=""),
    )
    llm_vllm_model: str = Field(
        default="",
        sa_column=Column(String(128), nullable=False, server_default=""),
    )
    llm_kilo_model: str = Field(
        default="",
        sa_column=Column(String(128), nullable=False, server_default=""),
    )
    pipeline_max_retries: int = Field(
        default=3,
        sa_column=Column(Integer, nullable=False, server_default="3"),
    )
    session_stability_delay: float = Field(
        default=30.0,
        sa_column=Column(Float, nullable=False, server_default="30.0"),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
        ),
    )
    updated_by_user_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String(256), nullable=True),
    )


# ── Read / update schemas ─────────────────────────────────────────────────────


class AppSettingsRead(SQLModel):
    """Public response schema — returned by GET and PUT /api/v1/settings."""

    inbox_path: str
    ollama_url: str
    ollama_model: str
    llm_active_provider: str
    llm_ollama_url: str
    llm_ollama_model: str
    llm_vllm_base_url: str
    llm_vllm_model: str
    llm_kilo_model: str
    pipeline_max_retries: int
    session_stability_delay: float
    updated_at: datetime
    updated_by_user_id: Optional[str] = None


class AppSettingsUpdate(SQLModel):
    """Partial update body for PUT /api/v1/settings — all fields optional."""

    inbox_path: Optional[str] = None
    ollama_url: Optional[str] = None
    ollama_model: Optional[str] = None
    llm_active_provider: Optional[str] = None
    llm_ollama_url: Optional[str] = None
    llm_ollama_model: Optional[str] = None
    llm_vllm_base_url: Optional[str] = None
    llm_vllm_model: Optional[str] = None
    llm_kilo_model: Optional[str] = None
    pipeline_max_retries: Optional[int] = Field(default=None, ge=0, le=10)
    session_stability_delay: Optional[float] = Field(default=None, ge=1.0, le=300.0)


class LlmProfileInfo(SQLModel):
    """Public descriptor for one LLM provider — keys are never exposed."""

    provider: str
    display_name: str
    base_url: str
    model: str
    has_api_key: bool
    is_active: bool


class LlmSettingsRead(SQLModel):
    """LLM provider overview — returned by GET /api/v1/settings/llm."""

    active_provider: str
    profiles: list[LlmProfileInfo]


class LlmModelEntry(SQLModel):
    """One model id advertised by a provider's ``/models`` endpoint."""

    id: str
    name: Optional[str] = None
    vision: Optional[bool] = None
    free: Optional[bool] = None
