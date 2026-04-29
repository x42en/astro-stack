"""Application configuration loaded from environment variables.

All settings are validated by Pydantic at startup. Missing required values
will cause an explicit startup failure with a descriptive error message.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings resolved from environment variables or `.env`.

    Attributes:
        app_name: Human-readable name shown in OpenAPI docs.
        app_version: Semver string injected into health-check response.
        log_level: Logging verbosity for structlog.
        auth_enabled: When False, JWT middleware is bypassed entirely.
        jwt_secret: HMAC-SHA256 signing secret for JWT tokens.
        jwt_algorithm: Signing algorithm (default HS256).
        jwt_expire_minutes: Token lifetime in minutes.
        database_url: Async PostgreSQL DSN (asyncpg driver).
        redis_url: Redis connection URL consumed by both ARQ and pub/sub.
        ollama_url: Base URL of the Ollama REST API.
        inbox_path: Host path watched for incoming astrophotography sessions.
        sessions_path: Working directory for in-progress session data.
        output_path: Destination for processed output files.
        models_path: Directory containing AI model weight files (.pth).
        siril_binary: Absolute path (or bare name) of the siril-cli executable.
        astap_binary: Absolute path (or bare name) of the astap executable.
        astap_star_db_path: Directory containing ASTAP star catalogue files.
        gpu_devices: Comma-separated CUDA device indices, e.g. "0,1".
        worker_concurrency: Number of parallel ARQ worker tasks per process.
        session_stability_delay: Seconds to wait after last file write before
            considering a session ready for processing.
        pipeline_max_retries: Default maximum retry count for failed steps.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "Astro-Stack"
    app_version: str = "0.1.0"
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    # ── Authentication ────────────────────────────────────────────────────────
    # auth_enabled: legacy toggle kept for backwards compatibility.
    # Prefer AUTH_MODE which takes priority if explicitly set.
    #   disabled → all checks bypassed (local dev, no auth server needed)
    #   mock     → HS256 JWT + X-Mock-User header (staging / integration tests)
    #   oidc     → OIDC/OAuth 2.1 via external provider (production)
    auth_enabled: bool = False
    auth_mode: Optional[Literal["disabled", "mock", "oidc"]] = None
    jwt_secret: str = Field(default="change-me-in-production", min_length=16)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = Field(default=60, gt=0)

    # ── OIDC (used when effective_auth_mode="oidc") ───────────────────────────
    oidc_issuer: Optional[str] = None
    oidc_audience: Optional[str] = None
    # Derived automatically from oidc_issuer if not set:
    # https://<issuer>/api/auth/jwks
    oidc_jwks_url: Optional[str] = None
    oidc_jwks_cache_ttl_seconds: int = Field(default=3600, ge=60)
    # Role name in the OIDC token that grants admin access (Settings page, etc.)
    oidc_admin_role: str = "admin"
    # Username that acts as admin in mock mode (X-Mock-User header value)
    mock_admin_user: str = "admin"

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins. Use "*" (default) during dev;
    # restrict to known origins in production (e.g. "https://app.astromote.com").
    cors_allowed_origins: str = "*"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(default="postgresql+asyncpg://astro:astro@localhost:5432/astrostack")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2"

    # ── Storage paths ─────────────────────────────────────────────────────────
    inbox_path: str = "/inbox"
    sessions_path: str = "/sessions"
    output_path: str = "/output"
    models_path: str = "/models"

    # ── External tool binaries ────────────────────────────────────────────────
    siril_binary: str = "siril-cli"
    # astap_cli is the headless command-line solver (no GTK / no display
    # required).  The standard "astap" GUI binary fails in Docker with
    # "Gtk-WARNING: cannot open display".  Both accept identical CLI flags.
    astap_binary: str = "astap_cli"
    astap_star_db_path: str = "/opt/astap/stars"

    # ── GPU ───────────────────────────────────────────────────────────────────
    gpu_devices: str = "0"  # "0" or "0,1"

    # ── Pipeline behaviour ────────────────────────────────────────────────────
    worker_concurrency: int = Field(default=2, ge=1, le=8)
    session_stability_delay: float = Field(default=30.0, ge=5.0)
    pipeline_max_retries: int = Field(default=3, ge=0, le=10)

    # ── GraXpert ──────────────────────────────────────────────────────────────
    graxpert_source_path: str = "/opt/graxpert"
    # AI model version (must match GraXpert's regex ^\d+\.\d+\.\d+$).
    graxpert_ai_model: str = "1.0.1"

    # ── Cosmic Clarity ────────────────────────────────────────────────────────
    cosmic_clarity_source_path: str = "/opt/cosmic-clarity"

    # ── Mock auth bridge (used only when auth_enabled is False) ──────────────
    # Deterministic UUID namespace turning a mock username into a stable user id
    # so the data persisted today can be remapped to a real user when auth lands.
    mock_user_namespace: str = "f5b3b1a4-9b1e-4c41-9f3d-9a3a5d1c0001"
    mock_user_header: str = "X-Mock-User"

    # ── Planning & weather (open-meteo + skyfield) ───────────────────────────
    openmeteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    # NOTE: Open-Meteo only exposes *forward* geocoding (`/v1/search`); there
    # is no reverse endpoint. The setting is kept for forward-search use; the
    # reverse path is served by Nominatim (see `nominatim_reverse_url`).
    openmeteo_geocode_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    nominatim_reverse_url: str = "https://nominatim.openstreetmap.org/reverse"
    weather_cache_ttl_s: int = Field(default=3600, ge=60)
    geocode_cache_ttl_s: int = Field(default=86400, ge=60)
    # Skyfield ephemeris file (DE440s, ~32 MB, 1849-2150, bundled in image).
    # DE440s supersedes DE421 with improved planetary positions; the small ("s")
    # variant trims the time span to keep the file size manageable while still
    # covering well beyond any plausible planning horizon.
    ephemeris_path: str = "/opt/ephemerides/de440s.bsp"
    # Default minimum altitude (degrees) above the horizon to consider an
    # object "visible" during the planning night window.
    planner_min_altitude_deg: float = Field(default=30.0, ge=5.0, le=85.0)
    planner_max_results: int = Field(default=50, ge=1, le=200)

    @property
    def effective_auth_mode(self) -> Literal["disabled", "mock", "oidc"]:
        """Resolve the active authentication mode from env vars.

        Priority: AUTH_MODE > AUTH_ENABLED + OIDC_ISSUER heuristic.
        """
        if self.auth_mode is not None:
            return self.auth_mode
        if not self.auth_enabled:
            return "disabled"
        if self.oidc_issuer:
            return "oidc"
        return "mock"

    @property
    def oidc_jwks_endpoint(self) -> Optional[str]:
        """JWKS URL for OIDC token verification.

        Falls back to ``<oidc_issuer>/api/auth/jwks`` if not explicitly set.
        """
        if self.oidc_jwks_url:
            return self.oidc_jwks_url
        if self.oidc_issuer:
            return f"{self.oidc_issuer.rstrip('/')}/api/auth/jwks"
        return None

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse ``cors_allowed_origins`` into a list of origin strings."""
        parts = [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]
        return parts or ["*"]

    @field_validator("gpu_devices")
    @classmethod
    def validate_gpu_devices(cls, value: str) -> str:
        """Ensure gpu_devices is a comma-separated list of non-negative integers.

        Args:
            value: Raw string value from environment.

        Returns:
            The validated, stripped string.

        Raises:
            ValueError: If any device index is not a non-negative integer.
        """
        parts = [p.strip() for p in value.split(",")]
        for part in parts:
            if not part.isdigit():
                raise ValueError(
                    f"Invalid GPU device index '{part}'. Must be a non-negative integer."
                )
        return ",".join(parts)

    @property
    def gpu_device_list(self) -> list[int]:
        """Return parsed GPU device indices as a list of ints.

        Returns:
            List of CUDA device indices.
        """
        return [int(d) for d in self.gpu_devices.split(",")]

    @property
    def database_url_str(self) -> str:
        """Return the database URL as a plain string.

        Returns:
            Async PostgreSQL connection string.
        """
        return self.database_url

    @property
    def redis_url_str(self) -> str:
        """Return the Redis URL as a plain string.

        Returns:
            Redis connection string.
        """
        return self.redis_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Uses ``lru_cache`` so the `.env` file is read only once per process.

    Returns:
        The validated :class:`Settings` instance.
    """
    return Settings()
