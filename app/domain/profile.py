"""Processing profile domain model.

A :class:`ProcessingProfile` stores the full parameter set for an ``advanced``
pipeline run. Preset profiles (quick/standard/quality) are built-in and not
stored in the database; only user-defined advanced profiles are persisted.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlmodel import Column, Field, SQLModel

from app.domain.job import ProfilePreset


class ProcessingProfileConfig(SQLModel):
    """Full parameter object for an advanced processing profile.

    All fields are optional; defaults mirror the ``standard`` preset values
    so that users only need to supply the parameters they wish to override.

    Attributes:
        rejection_algorithm: Statistical rejection method for stacking.
        rejection_low: Lower sigma/kappa threshold for frame rejection.
        rejection_high: Upper sigma/kappa threshold for frame rejection.
        normalization: Additive or multiplicative normalization mode.
        drizzle_enabled: Whether to apply drizzle super-sampling.
        drizzle_scale: Output scale multiplier (typically 2).
        drizzle_pixfrac: Pixel fraction parameter (0.0–1.0).
        debayer_pattern: Override auto-detected Bayer matrix pattern.
        plate_solving_enabled: Whether to run ASTAP plate solving.
        plate_solving_radius_deg: Search radius in degrees.
        plate_solving_speed: ASTAP speed mode.
        gradient_removal_enabled: Whether to run GraXpert.
        gradient_removal_method: ``ai`` or ``polynomial`` background model.
        gradient_removal_ai_model: GraXpert model name.
        stretch_method: Stretch algorithm applied after stacking.
        stretch_strength: Strength parameter for asinh stretch.
        color_calibration_enabled: Whether to run photometric color calibration.
        photometric_calibration_enabled: Whether to run Siril ``pcc`` (requires
            plate-solve).  Recommended only for defiltered/OSC astro cameras;
            on stock DSLR it tends to neutralise residual H\u03b1.
        denoise_enabled: Whether to run Cosmic Clarity Denoise.
        denoise_strength: Denoise strength (0.0–1.0).
        denoise_luminance_only: Apply denoise to luminance channel only.
        sharpen_enabled: Whether to run Cosmic Clarity Sharpen.
        sharpen_stellar_amount: Stellar sharpening amount (0.0–1.0).
        sharpen_nonstellar_amount: Non-stellar (nebula) sharpening amount (0.0–1.0).
        sharpen_radius: PSF radius hint for the sharpening model.
        super_resolution_enabled: Whether to run 2×AI upscaling.
        super_resolution_scale: Upscaling factor (typically 2).
        star_separation_enabled: Whether to run Dark Star star removal.
        star_separation_recombine: Recombine nebula and star layers after processing.
        star_separation_nebula_weight: Blend weight for the nebula layer.
        star_separation_star_weight: Blend weight for the star layer.
        max_retries: Override the global maximum retry count for this profile.
    """

    # ── Stacking ──────────────────────────────────────────────────────────────
    rejection_algorithm: str = "winsorized"  # sigma|winsorized|linear|none
    rejection_low: float = 3.0
    rejection_high: float = 3.0
    normalization: str = "addscale"  # addscale|mulscale|none
    drizzle_enabled: bool = False
    drizzle_scale: int = 2
    drizzle_pixfrac: float = 0.7
    debayer_pattern: str = "auto"  # auto|RGGB|BGGR|GRBG|GBRG

    # ── Plate solving ─────────────────────────────────────────────────────────
    plate_solving_enabled: bool = True
    plate_solving_radius_deg: float = 180.0
    plate_solving_speed: str = "auto"  # auto|slow|fast

    # ── Gradient removal ──────────────────────────────────────────────────────
    gradient_removal_enabled: bool = True
    gradient_removal_method: str = "ai"  # ai|polynomial
    gradient_removal_ai_model: str = "1.0.1"

    # ── Stretch & colour ─────────────────────────────────────────────
    stretch_method: str = "asinh"  # asinh|auto|linear
    stretch_strength: float = 150.0
    color_calibration_enabled: bool = True
    # Photometric Colour Calibration (Siril `pcc`) rescales each channel to
    # match a stellar B-V catalogue.  This is appropriate for cameras with
    # broadband response across all three channels (defiltered DSLR, dedicated
    # OSC astro camera).  On a stock DSLR the IR-cut filter strongly
    # attenuates Hα so the red channel carries proportionally more noise than
    # signal; PCC then equalises noise rather than signal and washes out the
    # nebular colour.  Default OFF for camera-agnostic safety; opt-in via the
    # QUALITY preset or per-profile.
    photometric_calibration_enabled: bool = False

    # ── Denoise ───────────────────────────────────────────────────────────────
    denoise_enabled: bool = True
    denoise_strength: float = 0.8
    denoise_luminance_only: bool = False

    # ── Sharpen ───────────────────────────────────────────────────────────────
    sharpen_enabled: bool = True
    sharpen_stellar_amount: float = 0.3
    sharpen_nonstellar_amount: float = 0.4
    sharpen_radius: int = 2

    # ── Super-resolution ──────────────────────────────────────────────────────
    super_resolution_enabled: bool = False
    super_resolution_scale: int = 2

    # ── Star separation ───────────────────────────────────────────────────────
    star_separation_enabled: bool = False
    star_separation_recombine: bool = True
    star_separation_nebula_weight: float = 0.8
    star_separation_star_weight: float = 0.5

    # ── Retry ─────────────────────────────────────────────────────────────────
    max_retries: int = 3


# Built-in preset configurations ──────────────────────────────────────────────

PRESET_QUICK = ProcessingProfileConfig(
    rejection_algorithm="sigma",
    drizzle_enabled=False,
    plate_solving_enabled=False,
    gradient_removal_enabled=False,
    stretch_method="auto",
    color_calibration_enabled=False,
    photometric_calibration_enabled=False,
    denoise_enabled=True,
    denoise_strength=0.5,
    sharpen_enabled=False,
    super_resolution_enabled=False,
    star_separation_enabled=False,
)

PRESET_STANDARD = ProcessingProfileConfig(
    rejection_algorithm="sigma",
    drizzle_enabled=False,
    plate_solving_enabled=True,
    gradient_removal_enabled=True,
    gradient_removal_method="ai",
    stretch_method="asinh",
    stretch_strength=150.0,
    color_calibration_enabled=True,
    photometric_calibration_enabled=False,
    denoise_enabled=True,
    denoise_strength=0.8,
    sharpen_enabled=True,
    sharpen_stellar_amount=0.3,
    sharpen_nonstellar_amount=0.4,
    super_resolution_enabled=False,
    star_separation_enabled=False,
)

PRESET_QUALITY = ProcessingProfileConfig(
    rejection_algorithm="winsorized",
    drizzle_enabled=True,
    drizzle_scale=2,
    drizzle_pixfrac=0.7,
    plate_solving_enabled=True,
    gradient_removal_enabled=True,
    gradient_removal_method="ai",
    stretch_method="asinh",
    stretch_strength=200.0,
    color_calibration_enabled=True,
    # OFF by default even on QUALITY: PCC tends to neutralise residual Hα on
    # stock DSLR.  Users with defiltered/OSC astro cameras can opt-in per profile.
    photometric_calibration_enabled=False,
    denoise_enabled=True,
    denoise_strength=0.9,
    sharpen_enabled=True,
    sharpen_stellar_amount=0.6,
    sharpen_nonstellar_amount=0.8,
    sharpen_radius=2,
    super_resolution_enabled=True,
    star_separation_enabled=True,
    star_separation_recombine=True,
)

PRESET_MAP: dict[ProfilePreset, ProcessingProfileConfig] = {
    ProfilePreset.QUICK: PRESET_QUICK,
    ProfilePreset.STANDARD: PRESET_STANDARD,
    ProfilePreset.QUALITY: PRESET_QUALITY,
}


def get_preset_config(preset: ProfilePreset) -> ProcessingProfileConfig:
    """Return the built-in configuration for the given preset.

    Args:
        preset: One of the built-in :class:`ProfilePreset` values.
            Use :attr:`ProfilePreset.ADVANCED` to load from the database instead.

    Returns:
        The :class:`ProcessingProfileConfig` for the requested preset.

    Raises:
        ValueError: If ``preset`` is ``ADVANCED`` (must be loaded from DB).
    """
    if preset == ProfilePreset.ADVANCED:
        raise ValueError(
            "ADVANCED profile config must be loaded from the database, not from presets."
        )
    return PRESET_MAP[preset]


class ProcessingProfile(SQLModel, table=True):
    """ORM model for user-saved advanced processing profiles.

    Maps to the ``processing_profiles`` table in PostgreSQL.

    Attributes:
        id: UUID primary key.
        owner_user_id: ID of the owning user (NULL = shared/system profile).
        name: Display name shown in the UI.
        description: Optional longer description.
        config: Full JSON configuration (serialised :class:`ProcessingProfileConfig`).
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "processing_profiles"  # type: ignore[assignment]

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    )
    owner_user_id: Optional[uuid.UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), nullable=True, index=True),
    )
    name: str = Field(
        max_length=255,
        sa_column=Column(String(255), nullable=False, index=True),
    )
    description: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    config: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
    )

    # Sharing: when ``is_shared`` is True the profile is visible (read-only)
    # to every authenticated user via ``GET /profiles``.  Other users may
    # duplicate it to obtain an editable private copy.  ``shared_at`` is
    # stamped on the first toggle and never cleared automatically.
    is_shared: bool = Field(
        default=False,
        sa_column=Column(
            "is_shared",
            Boolean(),
            nullable=False,
            index=True,
            server_default="false",
        ),
    )
    shared_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False,
        ),
    )


# ── Read/write schemas ────────────────────────────────────────────────────────


class ProcessingProfileCreate(SQLModel):
    """Schema for creating a new advanced profile via the REST API.

    Attributes:
        name: Display name (must be unique per user).
        description: Optional longer description.
        config: Full parameter set (see :class:`ProcessingProfileConfig`).
    """

    name: str = Field(max_length=255)
    description: Optional[str] = None
    config: ProcessingProfileConfig = Field(default_factory=ProcessingProfileConfig)


class ProcessingProfileRead(SQLModel):
    """Read schema returned by the REST API for profile resources.

    Attributes:
        id: Profile UUID.
        owner_user_id: ID of the owner; ``None`` for anonymous-mode or system
            profiles.
        is_shared: Whether the profile is publicly visible to other
            authenticated users.
        shared_at: Timestamp of the first share.
        is_owner: True when the requesting user owns this profile (or when
            running in anonymous/no-auth mode).  Computed server-side.
        name: Display name.
        description: Optional description.
        config: Parameter set as dict.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
    """

    id: uuid.UUID
    owner_user_id: Optional[uuid.UUID] = None
    is_shared: bool = False
    shared_at: Optional[datetime] = None
    is_owner: bool = True
    name: str
    description: Optional[str]
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ProcessingProfileUpdate(SQLModel):
    """Schema for partial profile updates.

    Attributes:
        name: New display name.
        description: New description.
        config: Replacement configuration (full overwrite, not patch).
    """

    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = None
    config: Optional[ProcessingProfileConfig] = None
