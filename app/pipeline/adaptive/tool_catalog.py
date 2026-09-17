"""Structured capability catalog for the external image-processing tools.

Describes every tunable field of
:class:`~app.domain.profile.ProcessingProfileConfig` in terms an automated
critic (or a human) can reason about: which external tool implements it,
what it visually/numerically affects, and what can go wrong if it is pushed
too far. This is deliberately *not* a duplicate of the field docstrings in
``profile.py`` — it is a compact, structured index over them, meant to be
sliced and injected into an LLM prompt or rendered as reference
documentation.

Grounding policy (anti-hallucination): every ``field_name`` below is
validated by ``tests/unit/test_tool_catalog.py`` against the real
``ProcessingProfileConfig.model_fields`` — a typo or a field that no longer
exists fails the test suite immediately rather than silently drifting from
the actual, tested pipeline configuration. Tool/flag attributions were
cross-checked against the adapters that actually issue the corresponding
CLI/API calls (``cosmic_adapter.py``, ``graxpert_adapter.py``,
``siril_adapter.py`` / ``siril_script_builder.py``, ``astap_adapter.py``),
themselves verified against each tool's real ``--help`` output or official
documentation (see repository memory notes for the verification log).

Example:
    >>> from app.pipeline.adaptive.tool_catalog import get_capability
    >>> get_capability("denoise_strength").tool
    <ToolName.COSMIC_CLARITY: 'cosmic_clarity'>
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict


class ToolName(str, Enum):
    """External processing tool that implements a given capability.

    Attributes:
        SIRIL: Calibration, stacking, registration, plate-solve orchestration,
            stretch and colour calibration (PCC/SPCC).
        ASTAP: Standalone astrometric plate solver.
        GRAXPERT: Background gradient removal and its denoise/deconvolution engines.
        COSMIC_CLARITY: SASpro's ``cosmicclarity`` CLI (AI denoise, sharpen,
            super-resolution, star separation, satellite trail removal).
    """

    SIRIL = "siril"
    ASTAP = "astap"
    GRAXPERT = "graxpert"
    COSMIC_CLARITY = "cosmic_clarity"


class RiskLevel(str, Enum):
    """Coarse severity of pushing a capability's value too far.

    Attributes:
        LOW: Safe to explore broadly; unlikely to visibly damage the image.
        MEDIUM: Can produce clearly wrong results at extreme values, but
            failure modes are easy to spot in a preview.
        HIGH: Can silently destroy faint signal or introduce artefacts that
            are easy to miss in a quick visual check.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ValueType(str, Enum):
    """Python/JSON type of a capability's value, matching its Pydantic field."""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"


class ToolCapability(BaseModel):
    """Describes one tunable field of ``ProcessingProfileConfig``.

    Attributes:
        field_name: Exact attribute name on ``ProcessingProfileConfig``.
        tool: External tool that implements this capability, or ``None`` for
            pipeline-infrastructure fields (e.g. retry policy) that are not
            owned by a specific external tool.
        role: One-line description of what the field controls.
        effect: One-line description of what changing the value does to the
            processed image.
        risk: One-line description of the failure mode at extreme values.
        risk_level: Coarse severity, see :class:`RiskLevel`.
        value_type: See :class:`ValueType`.
        choices: Allowed string values, for string fields with a closed set
            of options (``None`` for free-form strings, e.g. version numbers).
        min_value: Inclusive lower bound, for numeric fields (``None`` if unbounded).
        max_value: Inclusive upper bound, for numeric fields (``None`` if unbounded).
    """

    model_config = ConfigDict(frozen=True)

    field_name: str
    tool: Optional[ToolName]
    role: str
    effect: str
    risk: str
    risk_level: RiskLevel
    value_type: ValueType
    choices: Optional[tuple[str, ...]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


def _cap(
    field_name: str,
    tool: Optional[ToolName],
    role: str,
    effect: str,
    risk: str,
    risk_level: RiskLevel,
    value_type: ValueType,
    choices: Optional[tuple[str, ...]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> ToolCapability:
    """Construct a :class:`ToolCapability`; a thin positional-args convenience wrapper."""
    return ToolCapability(
        field_name=field_name,
        tool=tool,
        role=role,
        effect=effect,
        risk=risk,
        risk_level=risk_level,
        value_type=value_type,
        choices=choices,
        min_value=min_value,
        max_value=max_value,
    )


_LOW = RiskLevel.LOW
_MED = RiskLevel.MEDIUM
_HIGH = RiskLevel.HIGH
_BOOL = ValueType.BOOL
_INT = ValueType.INT
_FLOAT = ValueType.FLOAT
_STR = ValueType.STR
_SIRIL = ToolName.SIRIL
_ASTAP = ToolName.ASTAP
_GRAXPERT = ToolName.GRAXPERT
_COSMIC = ToolName.COSMIC_CLARITY

# ── Catalog data ───────────────────────────────────────────────────────────────
# Ordered to mirror app/domain/profile.py so the two stay easy to diff by eye.
TOOL_CAPABILITIES: tuple[ToolCapability, ...] = (
    # ── Stacking (Siril) ────────────────────────────────────────────────────
    _cap(
        "rejection_algorithm", _SIRIL,
        "Statistical method used to reject outlier pixels (satellite trails, "
        "cosmic rays, tracking glitches) when combining frames.",
        "'winsorized'/'sigma' remove outliers before averaging; 'linear' fits a "
        "trend; 'none' keeps every pixel from every frame.",
        "'none' lets a single bad frame (cloud, satellite, guiding glitch) "
        "permanently pollute the stack.",
        _MED, _STR, choices=("sigma", "winsorized", "linear", "none"),
    ),
    _cap(
        "rejection_low", _SIRIL,
        "Lower (dark outlier) sigma threshold for the rejection algorithm.",
        "Lower values reject more pixels as too-dark outliers.",
        "Too low on few-frame stacks rejects real signal, increasing apparent noise.",
        _MED, _FLOAT, min_value=0.0,
    ),
    _cap(
        "rejection_high", _SIRIL,
        "Upper (bright outlier) sigma threshold for the rejection algorithm.",
        "Lower values reject more pixels as too-bright outliers (satellite trails, hot pixels).",
        "Too low clips genuinely bright structures (star cores, bright nebula knots).",
        _MED, _FLOAT, min_value=0.0,
    ),
    _cap(
        "normalization", _SIRIL,
        "How per-frame background level/scale is matched before combination.",
        "'addscale'/'mulscale' correct both offset and gain drift between frames; "
        "'none' assumes frames are already matched.",
        "'none' on frames with varying sky brightness (moon rise, transparency "
        "change) produces visible tiling/banding in the stack.",
        _MED, _STR, choices=("addscale", "mulscale", "none"),
    ),
    _cap(
        "drizzle_enabled", _SIRIL,
        "Enables the HST drizzle algorithm to reconstruct sub-pixel resolution "
        "from a dithered frame set.",
        "Increases effective resolution/detail when frames were dithered.",
        "Amplifies noise and produces no benefit (or artefacts) without genuine "
        "sub-pixel dithering between frames.",
        _MED, _BOOL,
    ),
    _cap(
        "drizzle_scale", _SIRIL,
        "Output upscale factor applied by drizzle.",
        "Higher values increase output pixel dimensions and (if enough dithered "
        "frames exist) recovered detail.",
        "Scaling beyond what the dither pattern supports mostly upscales noise.",
        _LOW, _INT, min_value=1, max_value=3,
    ),
    _cap(
        "drizzle_pixfrac", _SIRIL,
        "Drizzle 'pixel fraction' — how much each input pixel is shrunk before "
        "being redistributed onto the output grid.",
        "Lower values sharpen the drizzle result at the cost of more noise between samples.",
        "Very low values leave visible gaps/noise if the frame count is low.",
        _MED, _FLOAT, min_value=0.0, max_value=1.0,
    ),
    _cap(
        "debayer_pattern", _SIRIL,
        "Overrides the auto-detected Bayer/CFA demosaic pattern for OSC/DSLR frames.",
        "Selects which raw pixel layout is used to reconstruct RGB from a mono sensor.",
        "A wrong explicit pattern produces channel-swapped or checkerboard colour artefacts.",
        _HIGH, _STR, choices=("auto", "RGGB", "BGGR", "GRBG", "GBRG"),
    ),
    # ── Star detection for registration (Siril findstar) ─────────────────────
    _cap(
        "findstar_override_enabled", _SIRIL,
        "Enables the explicit findstar_* overrides below instead of Siril defaults.",
        "Lets the profile relax/tighten star-detection criteria used for frame "
        "alignment before stacking.",
        "Relaxed criteria let non-stellar structures act as alignment anchors, "
        "smearing fine chrominance detail on bright nebula cores.",
        _MED, _BOOL,
    ),
    _cap(
        "findstar_radius", _SIRIL,
        "Initial search box radius (pixels) for star candidate detection.",
        "Larger radius finds stars over a wider local area per candidate.",
        "Too large merges close double stars or picks up nebula contours as stars.",
        _LOW, _INT, min_value=3, max_value=50,
    ),
    _cap(
        "findstar_sigma", _SIRIL,
        "Detection threshold above background noise, in sigma units.",
        "Lower values detect fainter stars; higher values keep only bright, unambiguous ones.",
        "Too low on noisy/wide-field data lets hot pixels and noise peaks register as stars.",
        _MED, _FLOAT, min_value=0.05,
    ),
    _cap(
        "findstar_roundness", _SIRIL,
        "Minimum star roundness accepted as a valid detection.",
        "Higher values reject elongated/trailed blobs, keeping only well-focused round stars.",
        "Too high on a field with tracking drift or coma can reject most real stars, "
        "starving registration of anchors.",
        _MED, _FLOAT, min_value=0.0, max_value=0.95,
    ),
    _cap(
        "findstar_relax", _SIRIL,
        "Relaxes star-candidate acceptance checks to allow non-star-shaped objects.",
        "Helps registration succeed on genuinely faint/wide-field data with few true stars.",
        "Directly smears chrominance detail on bright nebula cores by aligning on "
        "non-stellar structures — only use when registration otherwise fails.",
        _HIGH, _BOOL,
    ),
    # ── Plate solving (ASTAP) ─────────────────────────────────────────────────
    _cap(
        "plate_solving_enabled", _ASTAP,
        "Runs ASTAP to attach an astrometric (WCS) solution to the stacked image.",
        "Required for photometric colour calibration (PCC/SPCC) and catalogue-based "
        "object-type adaptation later in the pipeline.",
        "Disabling silently disables every downstream feature that depends on a WCS "
        "solution, with no visible error.",
        _MED, _BOOL,
    ),
    _cap(
        "plate_solving_radius_deg", _ASTAP,
        "Search radius (degrees) ASTAP spirals through around the initial position guess.",
        "Larger radius tolerates a less accurate initial position/FOV guess at the "
        "cost of solve time.",
        "Too small fails to solve when the initial guess (mount coordinates, FOV) is off.",
        _LOW, _FLOAT, min_value=0.0, max_value=180.0,
    ),
    _cap(
        "plate_solving_speed", _ASTAP,
        "ASTAP search thoroughness mode.",
        "'fast' trades search area overlap for speed; 'slow' forces 50% search-field "
        "overlap, helping on images with few detectable stars.",
        "'fast' can fail to solve sparse star fields that 'slow' would have found.",
        _LOW, _STR, choices=("auto", "slow", "fast"),
    ),
    # ── Gradient removal (GraXpert) ───────────────────────────────────────────
    _cap(
        "gradient_removal_enabled", _GRAXPERT,
        "Runs background gradient/light-pollution removal on the stacked image.",
        "Flattens sky-glow and vignetting gradients before stretching.",
        "Disabling on a gradient-heavy frame leaves colour casts that later steps "
        "cannot fully correct.",
        _MED, _BOOL,
    ),
    _cap(
        "gradient_removal_method", _GRAXPERT,
        "Background-model algorithm: AI (neural background estimate) or classic "
        "polynomial fitting.",
        "'ai' generally isolates true background better on complex nebula fields; "
        "'polynomial' is a simpler, more predictable classic fit.",
        "'polynomial' on a rich nebula field can flatten extended faint emission, "
        "mistaking it for gradient.",
        _MED, _STR, choices=("ai", "polynomial"),
    ),
    _cap(
        "gradient_removal_ai_model", _GRAXPERT,
        "Selects GraXpert's AI mode/version: background extraction ('1.0.1') or one "
        "of the 'deconv-obj-*'/'deconv-stars-*'/'deconv-both-*' deconvolution modes.",
        "Background-extraction modes flatten gradient; deconvolution modes instead "
        "sharpen the object and/or star layer.",
        "Selecting a deconvolution mode when only gradient removal was intended "
        "changes step behaviour entirely (sharpening, not flattening).",
        _MED, _STR,
    ),
    _cap(
        "gradient_removal_correction", _GRAXPERT,
        "AI background subtraction mode: absolute 'Subtraction' vs ratio-preserving "
        "'Division'. Only honoured in 'ai' method.",
        "'Division' preserves per-channel signal/background ratios, protecting faint "
        "chromatic signal (e.g. residual Hα on stock DSLR) from being clipped.",
        "'Subtraction' on a stock (non-defiltered) DSLR can clip faint red Hα signal "
        "to zero where sky background dominates the recorded red channel.",
        _MED, _STR, choices=("Subtraction", "Division"),
    ),
    _cap(
        "gradient_removal_smoothing", _GRAXPERT,
        "How smoothly the AI background model interpolates between sample tiles.",
        "Lower values (~0.3) produce a locally-detailed model that follows sky "
        "variation without flattening extended emission.",
        "The GraXpert default (1.0) can absorb diffuse nebulosity as if it were "
        "gradient on rich fields with a stock DSLR.",
        _MED, _FLOAT, min_value=0.0, max_value=1.0,
    ),
    _cap(
        "gradient_removal_deconv_strength", _GRAXPERT,
        "Deconvolution strength, only honoured when 'gradient_removal_ai_model' "
        "selects a deconv-* mode.",
        "Higher values sharpen more aggressively.",
        "Too high amplifies noise and produces ringing artefacts around edges/stars.",
        _MED, _FLOAT, min_value=0.0, max_value=1.0,
    ),
    _cap(
        "gradient_removal_deconv_psfsize", _GRAXPERT,
        "Assumed PSF size for the deconvolution model.",
        "Should roughly match the actual star FWHM for a physically correct correction.",
        "A PSF size far from reality produces halo or ringing artefacts.",
        _MED, _FLOAT, min_value=0.0, max_value=5.0,
    ),
    _cap(
        "gradient_removal_deconv_batch_size", _GRAXPERT,
        "Number of tiles GraXpert processes in parallel during deconvolution.",
        "Higher values are faster.",
        "Too high can exhaust GPU memory (OOM) on large frames.",
        _LOW, _INT, min_value=1, max_value=32,
    ),
    # ── Stretch & colour (Siril) ──────────────────────────────────────────────
    _cap(
        "stretch_method", _SIRIL,
        "Non-linear stretch algorithm applied after stacking/gradient removal.",
        "'asinh' preserves star cores better on high-dynamic-range nebulae; 'auto' "
        "picks a generic histogram stretch; 'linear' applies none.",
        "'linear' output looks almost black — only useful as a debug/passthrough setting.",
        _MED, _STR, choices=("asinh", "auto", "linear"),
    ),
    _cap(
        "stretch_strength", _SIRIL,
        "Strength of the asinh stretch curve.",
        "Higher values pull more faint signal out of the background at the cost of "
        "more visible noise and star bloat.",
        "Too high on bright nebulae/galaxies blows out cores and crushes midtone "
        "colour information.",
        _HIGH, _FLOAT, min_value=0.0,
    ),
    _cap(
        "color_calibration_enabled", _SIRIL,
        "Runs background colour neutralisation before/around the stretch.",
        "Removes a global colour cast from the sky background.",
        "Disabling on data with a strong colour cast (light pollution, uncalibrated "
        "white balance) leaves an obvious tint across the whole image.",
        _MED, _BOOL,
    ),
    _cap(
        "camera_defiltered", _SIRIL,
        "Acquisition-hardware hint: True for a defiltered/dedicated astro camera, "
        "False for a stock DSLR with full IR-cut filter.",
        "When False, the display pipeline softens the red black-point and adds a "
        "mild red/saturation boost to preserve residual Hα signal.",
        "Setting True on a stock DSLR crushes the faint residual Hα signal that a "
        "stock IR-cut filter still lets through.",
        _MED, _BOOL,
    ),
    _cap(
        "photometric_calibration_enabled", _SIRIL,
        "Runs Siril's Photometric Colour Calibration (PCC) against a star catalogue. "
        "Requires a successful plate-solve.",
        "Produces a more physically accurate colour balance than neutralisation alone.",
        "Catalogue lookup can fail silently on small FOV or sparse star fields, "
        "leaving colour calibration effectively skipped.",
        _MED, _BOOL,
    ),
    # ── Denoise (Cosmic Clarity or GraXpert) ──────────────────────────────────
    _cap(
        "denoise_enabled", _COSMIC,
        "Runs AI-based noise reduction.",
        "Smooths sensor/read/shot noise while attempting to preserve fine detail.",
        "Disabling on noisy data leaves grain that later sharpening will amplify.",
        _LOW, _BOOL,
    ),
    _cap(
        "denoise_engine", _COSMIC,
        "Selects the denoise backend: 'cosmic_clarity' (default, tuned for emission "
        "nebulae) or 'graxpert' (more aggressive, better fine stellar detail).",
        "Changes which external tool and model family perform this step.",
        "Switching engines mid-experiment makes strength/luminance-only settings "
        "non-comparable, since the two engines respond differently to them.",
        _LOW, _STR, choices=("cosmic_clarity", "graxpert"),
    ),
    _cap(
        "denoise_strength", _COSMIC,
        "Denoise blend factor (Cosmic Clarity) or '-strength' (GraXpert).",
        "Higher values smooth more aggressively.",
        "At >=0.7 (Cosmic Clarity) faint nebular filaments and IFN are visibly erased.",
        _HIGH, _FLOAT, min_value=0.0, max_value=1.0,
    ),
    _cap(
        "denoise_luminance_only", _COSMIC,
        "Restricts denoising to the luminance channel (Cosmic Clarity only; ignored "
        "by GraXpert).",
        "Preserves chrominance, which matters for emission targets where the red "
        "channel carries most of the Hα signal.",
        "Full-channel denoise on emission targets can smooth away real colour signal, "
        "not just noise.",
        _MED, _BOOL,
    ),
    _cap(
        "denoise_graxpert_ai_model", _GRAXPERT,
        "GraXpert denoise model version. Must match a folder under "
        "'GraXpert/denoise-ai-models/' in the models volume.",
        "Different versions can trade off smoothing strength vs detail preservation.",
        "An unavailable/unfetched version falls back to a runtime download or step failure.",
        _LOW, _STR,
    ),
    _cap(
        "denoise_graxpert_batch_size", _GRAXPERT,
        "Number of tiles GraXpert processes in parallel during denoising.",
        "Higher values are faster.",
        "Too high can exhaust GPU memory (OOM) on large frames.",
        _LOW, _INT, min_value=1, max_value=32,
    ),
    _cap(
        "denoise_aberration_first", _COSMIC,
        "Runs SASpro's standalone Aberration Remover ('cc correct') before denoising.",
        "Corrects colour fringing from chromatic aberration at the optics level.",
        "Chaining an extra AI pass adds processing time; redundant if sharpen's own "
        "'sharpen_aberration_first' already ran on the same image.",
        _LOW, _BOOL,
    ),
    # ── Sharpen (Cosmic Clarity) ───────────────────────────────────────────────
    _cap(
        "sharpen_enabled", _COSMIC,
        "Runs AI-based sharpening/deconvolution.",
        "Recovers detail lost to seeing, focus, and tracking imperfections.",
        "Disabling leaves stars/structures softer than the data would otherwise support.",
        _LOW, _BOOL,
    ),
    _cap(
        "sharpen_stellar_amount", _COSMIC,
        "Sharpening amount applied to point sources (stars).",
        "Higher values tighten star cores more aggressively.",
        "Too high produces ringing halos around bright stars.",
        _MED, _FLOAT, min_value=0.0, max_value=1.0,
    ),
    _cap(
        "sharpen_nonstellar_amount", _COSMIC,
        "Sharpening amount applied to extended objects (nebulae, galaxies).",
        "Higher values crisp up extended structure detail.",
        "Too high amplifies the noise floor left by denoising.",
        _MED, _FLOAT, min_value=0.0, max_value=1.0,
    ),
    _cap(
        "sharpen_radius", _COSMIC,
        "PSF radius hint for the non-stellar sharpening model.",
        "Should roughly match the actual star/PSF size in the image.",
        "A radius far from reality produces a visibly wrong sharpening scale (either "
        "no effect or harsh ringing).",
        _MED, _INT, min_value=1, max_value=8,
    ),
    _cap(
        "sharpen_aberration_first", _COSMIC,
        "Runs the Aberration Remover before sharpening via the inline "
        "'--stellar-correct-mode correct_sharpen' flag.",
        "Corrects colour fringing immediately before sharpening in a single pass.",
        "Redundant (wasted processing time) if 'denoise_aberration_first' already "
        "corrected the same image earlier in the pipeline.",
        _LOW, _BOOL,
    ),
    # ── Super-resolution (Cosmic Clarity) ─────────────────────────────────────
    _cap(
        "super_resolution_enabled", _COSMIC,
        "Runs AI 2x/3x/4x upscaling.",
        "Increases output resolution, hallucinating plausible high-frequency detail.",
        "On already-oversampled or bright, saturated nebula cores it can amplify "
        "clipping artefacts instead of adding real detail.",
        _HIGH, _BOOL,
    ),
    _cap(
        "super_resolution_scale", _COSMIC,
        "Upscaling factor.",
        "Higher factors increase output pixel dimensions further.",
        "Higher factors take proportionally longer and amplify any upstream artefact.",
        _LOW, _INT, choices=None, min_value=2, max_value=4,
    ),
    _cap(
        "super_resolution_mode", None,
        "Tri-state policy controlling how the object-type catalogue interacts with "
        "'super_resolution_enabled': 'auto' auto-skips on catalogue-flagged object "
        "types (e.g. bright nebulae), 'on'/'off' force the step regardless.",
        "Changes whether the catalogue is allowed to override the enabled flag.",
        "'on' overrides the catalogue's protection against amplifying clipped cores "
        "on bright nebulae.",
        _MED, _STR, choices=("auto", "on", "off"),
    ),
    # ── Star separation (Cosmic Clarity Dark Star) ────────────────────────────
    _cap(
        "star_separation_enabled", _COSMIC,
        "Runs Dark Star to split the image into starless and stars-only layers.",
        "Enables independent processing of nebula/galaxy structure vs stars.",
        "Running on galaxies/clusters (where stars themselves, or HII regions on "
        "galaxies, are the subject) can remove the intended subject.",
        _MED, _BOOL,
    ),
    _cap(
        "star_separation_recombine", _COSMIC,
        "Recombines the nebula and star layers after independent processing.",
        "Produces a single final image instead of leaving separate layers.",
        "Disabling leaves only the starless layer as the pipeline's final image.",
        _LOW, _BOOL,
    ),
    _cap(
        "star_separation_nebula_weight", _COSMIC,
        "Blend weight of the nebula (starless) layer during recombination.",
        "Higher values favour the nebula layer's contribution.",
        "Extreme values unbalance the recombination, making one layer dominate unnaturally.",
        _MED, _FLOAT, min_value=0.0, max_value=1.0,
    ),
    _cap(
        "star_separation_star_weight", _COSMIC,
        "Blend weight of the stars-only layer during recombination.",
        "Higher values favour the star layer's contribution.",
        "Extreme values unbalance the recombination, making one layer dominate unnaturally.",
        _MED, _FLOAT, min_value=0.0, max_value=1.0,
    ),
    _cap(
        "star_separation_mode", None,
        "Tri-state policy controlling how the object-type catalogue interacts with "
        "'star_separation_enabled': 'auto' auto-skips on catalogue-flagged types "
        "(galaxies, clusters), 'on'/'off' force the step regardless.",
        "Changes whether the catalogue is allowed to override the enabled flag.",
        "'on' overrides the catalogue's protection against removing the intended "
        "subject on galaxies/clusters.",
        _MED, _STR, choices=("auto", "on", "off"),
    ),
    # ── Satellite trail removal (Cosmic Clarity) ──────────────────────────────
    _cap(
        "satellite_removal_enabled", _COSMIC,
        "Runs AI satellite/aircraft trail detection and removal on the final image.",
        "Removes linear trail artefacts common in real-world wide-field/long-exposure "
        "sessions.",
        "An extra AI pass adds processing time on every job even when no trail is present.",
        _LOW, _BOOL,
    ),
    _cap(
        "satellite_removal_mode", _COSMIC,
        "Processes all channels ('full') or luminance only (faster, preserves colour).",
        "'luminance' is faster and better preserves chrominance.",
        "'full' takes longer for a usually marginal quality difference.",
        _LOW, _STR, choices=("full", "luminance"),
    ),
    _cap(
        "satellite_removal_sensitivity", _COSMIC,
        "Detection sensitivity threshold (lower = more aggressive detection).",
        "Lower values catch fainter/thinner trails.",
        "Too low produces false positives on genuine linear structures (diffraction "
        "spikes, bright nebula edges).",
        _MED, _FLOAT, min_value=0.0, max_value=1.0,
    ),
    _cap(
        "satellite_removal_clip_trail", _COSMIC,
        "Hard-clips detected trail pixels instead of inpainting/blending them.",
        "More reliable removal of the trail itself.",
        "Can leave a visible seam on wide trails; disable for a softer, less complete "
        "correction.",
        _MED, _BOOL,
    ),
    # ── Pipeline infrastructure (not owned by an external tool) ──────────────
    _cap(
        "max_retries", None,
        "Maximum automatic retry count for a failed pipeline step.",
        "Higher values tolerate more transient failures (network timeouts, GPU "
        "contention) before the job is marked failed.",
        "Too high delays surfacing a genuinely broken configuration to the user.",
        _LOW, _INT, min_value=0, max_value=10,
    ),
    # ── Adaptive vision critic (Phase 2, not owned by an external tool) ──────
    _cap(
        "adaptive_critic_enabled", None,
        "Enables the Phase 2 vision-critic loop, which re-runs an eligible "
        "step with a vision-LLM-adjusted config until satisfied.",
        "Adds an automated refinement pass on top of the fixed pipeline; fully "
        "unattended, no pause, unless a human-approval gate is also enabled.",
        "Adds latency (extra vision-model calls and step re-runs) and depends "
        "on the configured vLLM endpoint being reachable.",
        _LOW, _BOOL,
    ),
    _cap(
        "adaptive_critic_max_iterations", None,
        "Hard cap on critic iterations for the adaptive loop.",
        "Higher values allow more refinement passes before giving up.",
        "Too high adds latency for marginal gains once the critic has converged.",
        _LOW, _INT, min_value=1, max_value=10,
    ),
    _cap(
        "adaptive_critic_require_human_approval", None,
        "Gates critic-proposed patches behind external approval instead of "
        "applying them automatically.",
        "When True, a human reviewer must approve each patch before it is "
        "applied and the step re-run.",
        "Defaults to False (fully autonomous) to keep the pipeline hands-off "
        "for novices; enabling it without a reviewer configured auto-approves "
        "with a logged warning instead of stalling the job.",
        _MED, _BOOL,
    ),
)

_BY_FIELD: dict[str, ToolCapability] = {cap.field_name: cap for cap in TOOL_CAPABILITIES}

# Pipeline steps eligible for the Phase 2 adaptive vision-critic loop, mapped
# to the profile field names the critic is allowed to adjust for that step.
# Currently only ``stretch_color`` is wired up in the orchestrator (first
# target per the phased rollout plan); extending coverage only requires
# adding another step name here plus wiring it in
# ``PipelineOrchestrator._run_adaptive_loop_for_step``.
ADAPTIVE_STEP_FIELDS: dict[str, tuple[str, ...]] = {
    "stretch_color": (
        "stretch_method",
        "stretch_strength",
        "color_calibration_enabled",
        "camera_defiltered",
        "photometric_calibration_enabled",
    ),
}


def get_capability(field_name: str) -> ToolCapability:
    """Look up the catalog entry for a single profile field.

    Args:
        field_name: Attribute name on ``ProcessingProfileConfig``.

    Returns:
        The matching :class:`ToolCapability`.

    Raises:
        KeyError: If ``field_name`` is not present in the catalog.
    """
    try:
        return _BY_FIELD[field_name]
    except KeyError as exc:
        raise KeyError(f"No tool capability registered for field {field_name!r}") from exc


def capabilities_for_tool(tool: ToolName) -> tuple[ToolCapability, ...]:
    """Return every capability implemented by a given external tool.

    Args:
        tool: The tool to filter by.

    Returns:
        Matching capabilities, in catalog declaration order.
    """
    return tuple(cap for cap in TOOL_CAPABILITIES if cap.tool is tool)


def render_capabilities_for_prompt(field_names: Sequence[str]) -> str:
    """Render a subset of the catalog as compact text for an LLM prompt.

    Args:
        field_names: Profile field names to include, in the given order.

    Returns:
        One line per field: ``field_name (tool): role | effect | risk``.

    Raises:
        KeyError: If any name in ``field_names`` is not in the catalog.
    """
    lines = []
    for name in field_names:
        cap = get_capability(name)
        tool_label = cap.tool.value if cap.tool is not None else "pipeline"
        lines.append(f"{cap.field_name} ({tool_label}): {cap.role} | {cap.effect} | {cap.risk}")
    return "\n".join(lines)
