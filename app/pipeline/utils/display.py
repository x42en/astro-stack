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
    )


def _stretch_array(
    arr: np.ndarray,
    *,
    low_pct: float,
    high_pct: float,
    asinh_strength: float,
    per_channel: bool,
) -> np.ndarray:
    """Apply a percentile clip + optional asinh midtone stretch.

    See :func:`load_fits_display_rgb` for argument semantics.

    For 3-channel RGB data with ``per_channel=True`` this implements the
    *split BP/WP* policy described in the module docstring: the low percentile
    (black point) is computed **per channel** but the high percentile
    (white point) is computed **globally** across all channels so the natural
    colour balance of the signal is preserved.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[-1] == 3 and per_channel:
        # Per-channel black point — neutralises sky background cast.
        lo = np.array(
            [np.percentile(arr[..., c], low_pct) for c in range(3)],
            dtype=np.float32,
        )
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
