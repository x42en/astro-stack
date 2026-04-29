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
        gradient_removal_ai_model: GraXpert AI selector.  Acts as a combined
            ``mode + version`` field so the existing UI dropdown can expose
            every GraXpert AI capability through one control:

            * ``"1.0.1"`` — Background Extraction (default, recommended for
              nebulae).
            * ``"deconv-obj-1.0.1"`` — Deconvolution on the object layer
              only (recovers faint detail in nebula / galaxy disks).
            * ``"deconv-stars-1.0.0"`` — Deconvolution on the stellar layer
              only (tightens star PSFs).
            * ``"deconv-both-1.0.1"`` — Object then stars deconvolution
              chained.  Default for galaxies and clusters via the
              object-type catalogue (``app/pipeline/utils/object_type.py``).
        stretch_method: Stretch algorithm applied after stacking.
        stretch_strength: Strength parameter for asinh stretch.
        color_calibration_enabled: Whether to run photometric color calibration.
        camera_defiltered: Acquisition hardware hint.  ``True`` (default) for
            defiltered DSLR / dedicated OSC astro cameras (the norm in modern
            astrophotography).  Set ``False`` for a stock DSLR with full
            IR-cut filter; the display pipeline then preserves the faint
            residual H\u03b1 with a softened red black-point and a mild red
            saturation boost.
        photometric_calibration_enabled: Whether to run Siril ``pcc`` (requires
            plate-solve).  Independent from ``camera_defiltered``: kept
            opt-in because catalogue lookup may fail on small FOV / sparse
            star fields.
        denoise_enabled: Whether to run the AI denoise step.
        denoise_engine: ``"cosmic_clarity"`` (default) or ``"graxpert"``.
        denoise_strength: Denoise strength (0.0–1.0). Shared across engines.
        denoise_luminance_only: Apply denoise to luminance channel only.
            Cosmic Clarity-only flag (silently ignored by GraXpert).
        denoise_graxpert_ai_model: GraXpert denoise model version (e.g. ``"3.0.2"``).
        denoise_graxpert_batch_size: GraXpert tile batch size (1–32, default 4).
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

    # ── Star detection (Siril ``findstar``) ───────────────────────────────────
    # ``findstar`` powers Siril's ``register`` (frame alignment before stack)
    # and is configured session-wide via ``setfindstar``.  When enabled, our
    # script emits ``setfindstar reset`` then applies the values below before
    # the registration pass.
    #
    # **Stacking-chrominance impact**: relaxed values (low sigma, low
    # roundness, large radius) cause Siril to accept non-stellar structures
    # (nebula contours, hot pixels) as alignment anchors.  The resulting
    # micro-jitter between frames smears fine chrominance details (visible
    # on bright nebula cores like M42).  Defaults preserve colour fidelity;
    # only relax for genuinely faint/wide-field rigs that fail to find
    # enough true stars to align.
    findstar_override_enabled: bool = False
    findstar_radius: int = 10        # Siril default
    findstar_sigma: float = 1.0      # Siril default
    findstar_roundness: float = 0.5  # Siril default
    findstar_relax: bool = False     # Siril default

    # ── Plate solving ─────────────────────────────────────────────────────────
    plate_solving_enabled: bool = True
    plate_solving_radius_deg: float = 180.0
    plate_solving_speed: str = "auto"  # auto|slow|fast

    # ── Gradient removal ──────────────────────────────────────────────────────
    gradient_removal_enabled: bool = True
    gradient_removal_method: str = "ai"  # ai|polynomial
    gradient_removal_ai_model: str = "1.0.1"
    # GraXpert ``-correction`` flag.  ``"Subtraction"`` (default) removes the
    # absolute background level and is appropriate for high-SNR data where
    # the per-channel sky background is well measured.  ``"Division"``
    # preserves the per-channel **ratios** of signal to background, which
    # protects faint chromatic signal (e.g. residual Hα on a stock DSLR) from
    # being clipped to zero when the red sky background dominates the
    # recorded red signal.  Only honoured when ``gradient_removal_method`` is
    # ``"ai"`` (polynomial mode is always Subtraction).
    gradient_removal_correction: str = "Subtraction"  # Subtraction|Division
    # GraXpert ``-smoothing`` flag, ``[0.0, 1.0]``.  Controls how smoothly the
    # background model interpolates between sample tiles: ``1.0`` (GraXpert
    # default) builds a very smooth large-scale model that may absorb diffuse
    # nebulosity as if it were gradient; lower values (~0.3) produce a more
    # locally-detailed model that follows sky variations without flattening
    # extended emission targets.  Reduce on rich nebula fields with stock DSLR.
    gradient_removal_smoothing: float = 1.0
    # GraXpert deconvolution tunables — only honoured when
    # ``gradient_removal_ai_model`` selects a ``deconv-*`` mode.  Defaults
    # match the GraXpert CLI defaults so behaviour is predictable on any
    # imported profile that omits them.
    gradient_removal_deconv_strength: float = 0.5  # [0.0, 1.0]
    gradient_removal_deconv_psfsize: float = 0.3   # [0.0, 5.0]
    gradient_removal_deconv_batch_size: int = 4    # [1, 32]

    # ── Stretch & colour ─────────────────────────────────────────────
    stretch_method: str = "asinh"  # asinh|auto|linear
    stretch_strength: float = 150.0
    color_calibration_enabled: bool = True
    # ``camera_defiltered`` describes the **acquisition hardware**, not a
    # processing toggle.  When ``True`` (default — the norm in modern
    # astrophotography) the imager is assumed to have a broad spectral
    # response across R/G/B (defiltered DSLR, dedicated OSC astro camera).
    # The display pipeline then keeps its standard split black-point /
    # global white-point policy and a neutral saturation.
    # When ``False`` (stock DSLR with full IR-cut filter) the red channel is
    # heavily attenuated; the display pipeline applies a softened red
    # black-point and a mild red/saturation boost so the residual Hα signal
    # of emission targets is not crushed.
    camera_defiltered: bool = True
    # Photometric Colour Calibration (Siril `pcc`) rescales each channel to
    # match a stellar B-V catalogue.  Independent from ``camera_defiltered``:
    # PCC requires a successful plate-solve and an internet catalogue lookup,
    # which can fail silently on small FOV or with sparse star fields.  Kept
    # strictly opt-in to avoid surprising the user.  Recommended on
    # defiltered/OSC cameras when plate-solving is reliable.
    photometric_calibration_enabled: bool = False

    # ── Denoise ───────────────────────────────────────────────────────────────
    denoise_enabled: bool = True
    # ``denoise_engine`` selects which AI backend runs the denoise step.
    # ``cosmic_clarity`` (default) is the historical engine and is well-tuned
    # for emission nebulae.  ``graxpert`` uses GraXpert's 3.x ``-cmd denoising``
    # mode which is generally more aggressive and better at preserving fine
    # stellar detail; useful as an alternative on noisy galaxy frames.
    denoise_engine: str = "cosmic_clarity"
    # Strength is shared across engines (sémantique 0–1 compatible). Cosmic
    # Clarity treats it as a blend factor; GraXpert maps it to the ``-strength``
    # CLI flag (also clamped 0–1 server-side).
    denoise_strength: float = 0.8
    # When ``True`` Cosmic Clarity processes only the luminance channel which
    # preserves chrominance — important for emission-line targets where the
    # red channel carries the bulk of the Hα signal.  Ignored by GraXpert
    # (which has no equivalent flag).
    denoise_luminance_only: bool = False
    # GraXpert-specific knobs (only used when ``denoise_engine == 'graxpert'``).
    # The model version must match a folder under ``GraXpert/denoise-ai-models/``
    # in the models volume; ``3.0.2`` is the latest at the time of writing.
    denoise_graxpert_ai_model: str = "3.0.2"
    # Number of tiles processed in parallel by GraXpert (1–32).  Higher values
    # are faster but may cause GPU OOM on large frames.  Clamped server-side.
    denoise_graxpert_batch_size: int = 4

    # ── Sharpen ───────────────────────────────────────────────────────────────
    sharpen_enabled: bool = True
    sharpen_stellar_amount: float = 0.3
    sharpen_nonstellar_amount: float = 0.4
    sharpen_radius: int = 2

    # ── Super-resolution ──────────────────────────────────────────────────────
    super_resolution_enabled: bool = False
    super_resolution_scale: int = 2
    # Tri-state policy controlling how the object-type catalogue interacts
    # with ``super_resolution_enabled`` at job start:
    #   * ``"auto"`` (default) — honour ``super_resolution_enabled`` from
    #     the profile and, in addition, auto-skip on object types listed in
    #     :data:`SKIP_SUPER_RESOLUTION_TYPES` (e.g. bright nebulae where the
    #     model amplifies clipped cores).
    #   * ``"on"`` — force the step ON regardless of the catalogue
    #     (advanced override; can degrade the result).
    #   * ``"off"`` — force the step OFF regardless of the catalogue.
    super_resolution_mode: str = "auto"  # auto|on|off

    # ── Star separation ───────────────────────────────────────────────────────
    star_separation_enabled: bool = False
    star_separation_recombine: bool = True
    star_separation_nebula_weight: float = 0.8
    star_separation_star_weight: float = 0.5
    # Same tri-state semantics as ``super_resolution_mode``; auto-skips on
    # types listed in :data:`SKIP_STAR_SEPARATION_TYPES` (galaxies, clusters)
    # when the mode is ``"auto"``.
    star_separation_mode: str = "auto"  # auto|on|off

    # ── Retry ─────────────────────────────────────────────────────────────────
    max_retries: int = 3


# Built-in preset configurations ──────────────────────────────────────────────

PRESET_QUICK = ProcessingProfileConfig(
    rejection_algorithm="sigma",
    drizzle_enabled=False,
    plate_solving_enabled=False,
    gradient_removal_enabled=False,
    gradient_removal_ai_model="auto",
    stretch_method="auto",
    color_calibration_enabled=False,
    camera_defiltered=True,
    photometric_calibration_enabled=False,
    denoise_enabled=True,
    denoise_engine="cosmic_clarity",
    denoise_strength=0.5,
    sharpen_enabled=False,
    super_resolution_enabled=False,
    super_resolution_mode="auto",
    star_separation_enabled=False,
    star_separation_mode="auto",
)

# ``stretch_strength=150`` is tuned for emission nebulae (Hα-rich, high
# surface brightness). The ``stretch_color`` step adapts this value
# automatically for galaxies / clusters via the bundled catalogue lookup —
# see ``app/pipeline/steps/stretch_color.py`` and
# ``app/pipeline/utils/object_type.py``.
PRESET_STANDARD = ProcessingProfileConfig(
    rejection_algorithm="sigma",
    drizzle_enabled=False,
    plate_solving_enabled=True,
    gradient_removal_enabled=True,
    gradient_removal_method="ai",
    gradient_removal_ai_model="auto",
    stretch_method="asinh",
    stretch_strength=150.0,
    color_calibration_enabled=True,
    camera_defiltered=True,
    photometric_calibration_enabled=False,
    denoise_enabled=True,
    denoise_engine="cosmic_clarity",
    # Reduced from 0.8 → 0.55: Cosmic Clarity at strength ≥0.7 erases faint
    # nebular filaments on emission targets (M42 wings, IFN).
    denoise_strength=0.55,
    # Luminance-only denoise preserves chrominance; critical to keep the
    # red Hα signal which is otherwise smoothed away with full RGB denoise.
    denoise_luminance_only=True,
    sharpen_enabled=True,
    # Reduced from 0.3/0.4 → 0.25/0.30 to avoid amplifying denoised noise floor.
    sharpen_stellar_amount=0.25,
    sharpen_nonstellar_amount=0.30,
    super_resolution_enabled=False,
    super_resolution_mode="auto",
    star_separation_enabled=False,
    star_separation_mode="auto",
)

PRESET_QUALITY = ProcessingProfileConfig(
    rejection_algorithm="winsorized",
    drizzle_enabled=True,
    drizzle_scale=2,
    drizzle_pixfrac=0.7,
    plate_solving_enabled=True,
    gradient_removal_enabled=True,
    gradient_removal_method="ai",
    gradient_removal_ai_model="auto",
    stretch_method="asinh",
    # Reduced from 200 → 180: combined with the lower display highlight
    # rolloff (display.py) this preserves star cores without losing midtones.
    stretch_strength=180.0,
    color_calibration_enabled=True,
    camera_defiltered=True,
    # PCC kept opt-in even on QUALITY: requires plate-solve + catalogue
    # lookup, which fail silently on small FOV / sparse fields.
    photometric_calibration_enabled=False,
    denoise_enabled=True,
    denoise_engine="cosmic_clarity",
    # Reduced from 0.9 → 0.65 (was destroying faint signal in tests).
    denoise_strength=0.65,
    denoise_luminance_only=True,
    sharpen_enabled=True,
    # Reduced from 0.6/0.8 → 0.45/0.55.
    sharpen_stellar_amount=0.45,
    sharpen_nonstellar_amount=0.55,
    sharpen_radius=2,
    # Super-resolution and star separation are ON for the full Quality
    # experience.  Object-type adaptation auto-skips them on targets where
    # they would damage the image (super-res off on bright nebulae, star
    # separation off on galaxies / clusters); see
    # ``app/pipeline/utils/object_type.py``.
    super_resolution_enabled=True,
    super_resolution_mode="auto",
    star_separation_enabled=True,
    star_separation_mode="auto",
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
