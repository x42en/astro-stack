"""Shared FITS → display-ready image conversion.

A single function used by per-step previews, the orchestrator preview helper,
and the final export step so the visible JPEG matches the in-pipeline previews.

The pipeline historically used three different percentile stretches
(``0.01/99.99``, ``0.1/99.9`` and ``1.0/99.5``), which made the final export
look much darker than the intermediate previews. This module unifies them.

Colour-stretch strategy — **split black-point / global white-point**:

The naive options both fail on uncalibrated DSLR data:

* **Pure global** (single percentile shared across R/G/B) keeps the natural
  channel ratios but never neutralises the sky background → yellow/green cast.
* **Pure per-channel** (independent percentile per R/G/B) removes the cast but
  also normalises the *signal*, so an emission-line target like M42 (strongly
  Hα-dominated) loses its red signature and becomes uniformly grey/blue.

The accepted astrophoto convention — and what we apply here — is to combine
the two:

* The **black point** (low percentile) is computed **per channel**: this
  subtracts the per-channel sky background, which is what causes the cast.
* The **white point** (high percentile) is computed **globally** across all
  three channels: this preserves the natural emission-line dominance because
  a brighter channel keeps its higher post-clip values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Default percentile clip — leaves a hint of background (not pure black) and
# preserves star cores (does not blow them to pure white).
_DEFAULT_LOW_PCT = 0.5
_DEFAULT_HIGH_PCT = 99.7

# asinh midtone strength applied AFTER the percentile clip to lift faint
# nebulosity. ~30-100 gives a strong "stretch" without crushing star cores;
# 0 disables the asinh and keeps a pure linear percentile stretch.
_DEFAULT_ASINH_STRENGTH = 50.0


def load_fits_display_rgb(
    fits_path: Path,
    *,
    low_pct: float = _DEFAULT_LOW_PCT,
    high_pct: float = _DEFAULT_HIGH_PCT,
    asinh_strength: float = _DEFAULT_ASINH_STRENGTH,
    per_channel: bool = True,
    camera_defiltered: bool = True,
) -> np.ndarray:
    """Load a FITS image and return a display-ready float array in ``[0, 1]``.

    The returned array is either ``(H, W)`` for monochrome data or
    ``(H, W, 3)`` in **RGB** order (channels already swapped from Siril's
    internal BGR layout).

    Args:
        fits_path: Source FITS file.
        low_pct: Lower percentile for the linear clip (default 0.5).
        high_pct: Upper percentile for the linear clip (default 99.7).
        asinh_strength: Strength of the post-clip asinh stretch.
            ``0.0`` disables it (pure linear percentile stretch).
        per_channel: When ``True`` and the image is RGB, compute the
            percentile clip independently per channel. This neutralises
            colour casts from an uncalibrated sky background.
        camera_defiltered: ``True`` (default, modern astrophoto norm) keeps
            the standard split BP/WP policy.  ``False`` (stock DSLR with
            full IR-cut filter) softens the per-channel **red** black-point
            so the faint residual Hα signal is not crushed when neutralising
            the sky background.

    Returns:
        ``float32`` ndarray in ``[0, 1]``, RGB order if 3-channel.

    Raises:
        ValueError: If the FITS file contains no image data.
    """
    from astropy.io import fits as _fits  # noqa: PLC0415

    with _fits.open(str(fits_path)) as hdul:
        data: np.ndarray | None = None
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                data = np.array(hdu.data, dtype=np.float32)
                break

    if data is None:
        raise ValueError(f"FITS file {fits_path} contains no image data.")

    # Normalise axis layout: (H, W) or (H, W, C) with C in {1, 3}.
    if data.ndim == 3:
        if data.shape[0] in (1, 3, 4):
            # FITS stores as (C, H, W) — move channels to last axis
            data = np.moveaxis(data, 0, -1)
        if data.shape[-1] == 1:
            data = data[..., 0]
        elif data.shape[-1] == 3:
            # Siril stores colour FITS planes in B, G, R order; PIL and our
            # downstream code expect R, G, B.
            data = data[..., ::-1]

    # Sanitise non-finite values before any percentile / arithmetic.
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return _stretch_array(
        data,
        low_pct=low_pct,
        high_pct=high_pct,
        asinh_strength=asinh_strength,
        per_channel=per_channel,
        camera_defiltered=camera_defiltered,
    )


def _stretch_array(
    arr: np.ndarray,
    *,
    low_pct: float,
    high_pct: float,
    asinh_strength: float,
    per_channel: bool,
    camera_defiltered: bool = True,
) -> np.ndarray:
    """Apply a percentile clip + optional asinh midtone stretch.

    See :func:`load_fits_display_rgb` for argument semantics.

    For 3-channel RGB data with ``per_channel=True`` this implements the
    *split BP/WP* policy described in the module docstring: the low percentile
    (black point) is computed **per channel** but the high percentile
    (white point) is computed **globally** across all channels so the natural
    colour balance of the signal is preserved.

    When ``camera_defiltered=False`` the per-channel red black-point is
    further attenuated (multiplied by 0.55) so the faint residual Hα of a
    stock DSLR survives the sky-background subtraction.

    Safeguard: if the input is already saturated (median > 0.95 — typically
    the result of an over-aggressive Siril ``asinh`` on a low-surface-
    brightness target), a soft compression is applied before the percentile
    clip so the preview still shows usable structure instead of pure white.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    # ── Over-stretch safeguard ────────────────────────────────────────────
    # When ``arr`` is mostly clipped to the white point (median > 0.95) the
    # downstream percentile clip would compute lo ≈ hi ≈ 1.0 and fall back
    # to the 1e-12 denom guard, producing a near-uniform image (typically
    # all-black after the asinh).  Re-expand the upper range with a fourth
    # power so the brightest highlights are pulled back into the displayable
    # range while preserving relative ordering of pixel values.
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.median(finite)) > 0.95:
        arr = np.power(arr, 4.0, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[-1] == 3 and per_channel:
        # Per-channel black point — neutralises sky background cast.
        lo = np.array(
            [np.percentile(arr[..., c], low_pct) for c in range(3)],
            dtype=np.float32,
        )
        # Stock DSLR: the IR-cut filter slashes the red signal; the per-channel
        # red BP would then subtract most of what the sensor still captured of
        # Hα.  Soften it (factor 0.55) to keep the warm tone of emission targets.
        if not camera_defiltered:
            lo[0] = lo[0] * 0.55
        # Global white point — preserves emission-line channel dominance
        # (e.g. Hα-rich M42 stays red; Oxygen-III-rich M27 stays teal).
        hi = float(np.percentile(arr, high_pct))

        out = np.empty_like(arr)
        for c in range(3):
            denom = max(hi - float(lo[c]), 1e-12)
            out[..., c] = np.clip((arr[..., c] - lo[c]) / denom, 0.0, 1.0)

        if asinh_strength > 0.0:
            out = np.arcsinh(asinh_strength * out) / np.arcsinh(asinh_strength)
        return out.astype(np.float32, copy=False)

    return _stretch_2d(
        arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1),
        low_pct=low_pct,
        high_pct=high_pct,
        asinh_strength=asinh_strength,
    ).reshape(arr.shape)


def _stretch_2d(
    plane: np.ndarray,
    *,
    low_pct: float,
    high_pct: float,
    asinh_strength: float,
) -> np.ndarray:
    """Percentile-clip a single plane to ``[0, 1]`` and optionally asinh-stretch."""
    lo, hi = np.percentile(plane, (low_pct, high_pct))
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-12:
        hi = lo + 1.0
    out = np.clip((plane - lo) / float(hi - lo), 0.0, 1.0)

    if asinh_strength > 0.0:
        # arcsinh midtone stretch: lifts faint signal while compressing
        # highlights, mimicking Siril's "asinh -human" curve on display data.
        out = np.arcsinh(asinh_strength * out) / np.arcsinh(asinh_strength)

    return out.astype(np.float32, copy=False)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a ``[0, 1]`` float array to uint8 ``[0, 255]``."""
    return np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)


def to_uint16(arr: np.ndarray) -> np.ndarray:
    """Convert a ``[0, 1]`` float array to uint16 ``[0, 65535]``."""
    return np.clip(arr * 65535.0, 0.0, 65535.0).astype(np.uint16)


def apply_hdr_polish(
    arr: np.ndarray,
    *,
    saturation: float = 1.18,
    midtone_contrast: float = 0.18,
    highlight_rolloff: float = 0.85,
    camera_defiltered: bool = True,
) -> np.ndarray:
    """Apply a gentle HDR-style polish to a normalised ``[0, 1]`` RGB array.

    Three sequential passes, each conservative enough to preserve calibrated
    star colours and the Hα signature on emission targets:

    1. **S-curve on luminance** — a soft sigmoid centred on 0.5 lifts the
       midtones while leaving deep sky and shadows untouched. Strength is
       controlled by ``midtone_contrast`` (0 = identity, 0.3 = pronounced).
    2. **Highlight rolloff** — values above ``highlight_rolloff`` are
       compressed via ``tanh`` so bright stars never clip to pure white.
    3. **Saturation boost** — chrominance is scaled in HSV space by
       ``saturation`` (1.0 = identity). Hue is preserved exactly.

    Args:
        arr: Float array shaped ``(H, W)`` or ``(H, W, 3)`` with values in
            ``[0, 1]``. Mono inputs skip the saturation pass.
        saturation: Multiplicative gain on HSV S channel (≥ 0).
        midtone_contrast: Strength of the midtone S-curve (0 disables).
        highlight_rolloff: Threshold above which highlights are softly
            compressed (set to 1.0 to disable).  Default 0.85 keeps bright
            star cores from clipping to pure white — a generic,
            object-agnostic recovery of any over-exposed region.
        camera_defiltered: When ``False`` (stock DSLR) apply a small extra
            saturation gain (+0.10) and a +5% red-channel boost to compensate
            for the IR-cut filter attenuation on Hα.

    Returns:
        New array with the same shape and dtype as ``arr``, polished and
        clipped to ``[0, 1]``.
    """
    out = np.clip(arr.astype(np.float32, copy=True), 0.0, 1.0)

    # 1) Midtone S-curve. Acts as `out = out + k * sin(2π·out) / (2π)` which
    #    is a smooth, monotonic, midtone-only contrast lift (no clipping risk).
    if midtone_contrast > 0:
        bump = np.sin(2.0 * np.pi * out) / (2.0 * np.pi)
        out = np.clip(out + midtone_contrast * bump, 0.0, 1.0)

    # 2) Highlight rolloff — soft tanh compression above the threshold.
    if 0.0 < highlight_rolloff < 1.0:
        excess = np.maximum(out - highlight_rolloff, 0.0)
        # Compress the excess into the remaining [t, 1] range
        compressed = np.tanh(excess / (1.0 - highlight_rolloff)) * (1.0 - highlight_rolloff)
        out = np.where(out > highlight_rolloff, highlight_rolloff + compressed, out)

    # 3) Stock-DSLR red compensation — small +5% gain on the R channel before
    #    the saturation pass so the residual Hα signal regains visibility on
    #    emission targets.  Object-agnostic: also slightly warms star cores,
    #    which is acceptable for an IR-cut sensor that under-records red.
    effective_saturation = saturation
    if not camera_defiltered:
        if out.ndim == 3 and out.shape[2] == 3:
            out[..., 0] = np.clip(out[..., 0] * 1.05, 0.0, 1.0)
        effective_saturation = saturation + 0.10

    # 4) Saturation boost in HSV — only meaningful for RGB inputs.
    if out.ndim == 3 and out.shape[2] == 3 and effective_saturation != 1.0:
        out = _boost_saturation_rgb(out, effective_saturation)

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _boost_saturation_rgb(rgb: np.ndarray, gain: float) -> np.ndarray:
    """Multiply the HSV S channel of an RGB array by ``gain``.

    Implemented in vectorised numpy without external dependencies. Hue and
    Value are preserved bit-exact; only saturation changes.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    # New saturation: grey pixels (cmax==0 or delta==0) stay grey by construction.
    safe_cmax = np.where(cmax > 0, cmax, 1.0)
    s = delta / safe_cmax
    s_new = np.clip(s * gain, 0.0, 1.0)

    # Reconstruct chroma component then re-add to the value channel.
    # For each pixel: new_min = cmax * (1 - s_new); the relative position of
    # each channel inside [cmin, cmax] is preserved.
    new_min = cmax * (1.0 - s_new)
    safe_delta = np.where(delta > 0, delta, 1.0)

    def _rescale(c: np.ndarray) -> np.ndarray:
        # Map c from [cmin, cmax] to [new_min, cmax]
        return new_min + (c - cmin) * (cmax - new_min) / safe_delta

    out = np.stack([_rescale(r), _rescale(g), _rescale(b)], axis=-1)
    # Pixels with delta==0 are achromatic — keep them as-is.
    achromatic = delta == 0
    if np.any(achromatic):
        out[achromatic] = rgb[achromatic]
    return out


# ── Output metadata helpers ──────────────────────────────────────────────────

ASTROSTACK_BRAND = "Generated by AstroStack"
ASTROSTACK_URL = "https://astromote.com"


def summarize_profile_config(config: dict | None) -> list[tuple[str, str]]:
    """Return an ordered list of ``(label, value)`` tuples summarising the
    profile parameters that meaningfully shape the rendered image.

    Used both for FITS HISTORY cards and for the JPEG badge. Falls back to
    an empty list when ``config`` is missing.
    """
    if not config:
        return []

    def _on_off(v: object) -> str:
        return "on" if bool(v) else "off"

    pairs: list[tuple[str, str]] = []

    rej = config.get("rejection_algorithm")
    if rej:
        pairs.append(("Stacking", str(rej)))

    if config.get("drizzle_enabled"):
        pairs.append(("Drizzle", f"x{config.get('drizzle_scale', 2)}"))

    if config.get("plate_solving_enabled"):
        pairs.append(("Plate solve", _on_off(True)))

    if config.get("gradient_removal_enabled"):
        method = config.get("gradient_removal_method", "ai")
        pairs.append(("Gradient", str(method)))

    method = config.get("stretch_method")
    strength = config.get("stretch_strength")
    if method:
        if strength is not None:
            pairs.append(("Stretch", f"{method} {strength:g}"))
        else:
            pairs.append(("Stretch", str(method)))

    if config.get("color_calibration_enabled"):
        pairs.append(("Color cal.", _on_off(True)))
    if config.get("photometric_calibration_enabled"):
        pairs.append(("PCC", _on_off(True)))
    # camera_defiltered is "True" by default; only mention when False so the
    # badge stays terse for the common case.
    if config.get("camera_defiltered") is False:
        pairs.append(("Camera", "stock DSLR"))

    if config.get("denoise_enabled"):
        engine = str(config.get("denoise_engine", "cosmic_clarity")).lower()
        engine_label = "GraXpert" if engine == "graxpert" else "Cosmic"
        pairs.append(
            ("Denoise", f"{engine_label} {config.get('denoise_strength', 0):.2f}")
        )

    if config.get("sharpen_enabled"):
        pairs.append(("Sharpen", _on_off(True)))

    if config.get("super_resolution_enabled"):
        pairs.append(("Super-res", f"x{config.get('super_resolution_scale', 2)}"))

    if config.get("star_separation_enabled"):
        pairs.append(("Star sep.", _on_off(True)))

    return pairs


def _load_badge_font(size: int):
    """Load a TrueType font for the badge, with fallbacks.

    Tries DejaVuSans then PIL's bitmap default. Returns a PIL font instance.
    """
    from PIL import ImageFont  # noqa: PLC0415

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_metadata_badge(
    img,
    profile_summary: list[tuple[str, str]] | None = None,
    *,
    brand: str = ASTROSTACK_BRAND,
    url: str = ASTROSTACK_URL,
):
    """Composite a clean info bar at the bottom of a PIL RGB image.

    Renders a semi-transparent dark band (~9% of image height) containing:
    - left: brand line + URL
    - right: up to 6 ``key: value`` pairs from ``profile_summary``

    Args:
        img: PIL.Image (RGB or convertible).
        profile_summary: Output of :func:`summarize_profile_config`.

    Returns:
        A new PIL.Image with the badge composited in. The original image is
        not modified.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    base = img.convert("RGBA") if img.mode != "RGBA" else img.copy()
    w, h = base.size

    # Scale band height proportionally; clamp for very small/large images.
    band_h = max(38, min(int(h * 0.09), 110))
    pad_x = max(12, int(band_h * 0.4))
    pad_y = max(6, int(band_h * 0.18))

    # ── Background: vertical fade transparent → deep black ──
    # The band is built from a per-row alpha gradient (0 at the top edge,
    # ~235 at the bottom) so the image content fades smoothly into the
    # watermark instead of being cut by a hard horizontal line.
    band = Image.new("RGBA", (w, band_h), (0, 0, 0, 0))
    gradient = Image.new("L", (1, band_h))
    for y in range(band_h):
        # Cubic ease-in keeps the top barely visible and ramps quickly so
        # the text area stays well-contrasted.
        t = y / max(band_h - 1, 1)
        gradient.putpixel((0, y), int(235 * (t ** 1.6)))
    gradient = gradient.resize((w, band_h))
    fill = Image.new("RGBA", (w, band_h), (0, 0, 0, 255))
    fill.putalpha(gradient)
    band = Image.alpha_composite(band, fill)
    draw = ImageDraw.Draw(band)

    # Font sizes derived from band height.
    title_size = max(11, int(band_h * 0.30))
    sub_size = max(9, int(band_h * 0.22))
    val_size = max(9, int(band_h * 0.22))

    f_title = _load_badge_font(title_size)
    f_sub = _load_badge_font(sub_size)
    f_val = _load_badge_font(val_size)

    # ── Left block: brand + url ──
    draw.text((pad_x, pad_y), brand, fill=(245, 247, 252, 235), font=f_title)
    draw.text(
        (pad_x, pad_y + title_size + 2),
        url,
        fill=(160, 175, 200, 200),
        font=f_sub,
    )

    # ── Right block: profile summary key/value chips ──
    summary = (profile_summary or [])[:6]
    if summary:
        # Compute total width from right edge.
        chip_pad_x = max(7, int(band_h * 0.14))
        chip_pad_y = max(3, int(band_h * 0.07))
        chip_gap = max(8, int(band_h * 0.12))

        # Pre-measure each chip.
        chips = []
        for label, value in summary:
            text = f"{label}: {value}"
            bbox = draw.textbbox((0, 0), text, font=f_val)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            chip_w = text_w + 2 * chip_pad_x
            chip_h = text_h + 2 * chip_pad_y
            chips.append((text, chip_w, chip_h))

        # Right-align: lay out from rightmost chip.
        x = w - pad_x
        y_center = band_h // 2
        for text, chip_w, chip_h in reversed(chips):
            x -= chip_w
            y0 = y_center - chip_h // 2
            # Modern rounded-square chip (radius ≈ h/3, not full pill) with
            # subtle 1px border that picks up the UI's aesthetic.
            draw.rounded_rectangle(
                [(x, y0), (x + chip_w, y0 + chip_h)],
                radius=max(4, chip_h // 3),
                fill=(255, 255, 255, 14),
                outline=(255, 255, 255, 55),
                width=1,
            )
            draw.text(
                (x + chip_pad_x, y0 + chip_pad_y - 1),
                text,
                fill=(225, 233, 245, 240),
                font=f_val,
            )
            x -= chip_gap
            if x < pad_x + 240:  # leave room for brand block
                break

    # Composite the band over the bottom of the image.
    base.alpha_composite(band, dest=(0, h - band_h))
    return base.convert("RGB")


