"""Auto-stretching utilities for the live-stacking preview pipeline.

This module implements the *Midtones Transfer Function* (MTF)
auto-stretch algorithm popularised by PixInsight's
``HistogramTransformation`` tool.  The implementation is a direct port
of the ``Stretch`` class from the AstroLiveStacker (ALS) project
(``src/contrib/stretch.py``), kept compatible licence-wise as both
projects ship under GPLv3.

Original credits
----------------
- Algorithm: PixInsight (Pleiades Astrophoto), public reference at
  https://pixinsight.com/doc/tools/HistogramTransformation/HistogramTransformation.html
- Python implementation: AstroLiveStacker contributors, GPLv3.

The functions below operate on ``numpy`` arrays normalised to
``float32``/``float64`` in ``[0, 1]`` and return arrays of the same
shape.  Higher-level helpers are provided for the live-stack service
that needs uint8 output ready to encode as JPEG.
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "StretchMethod",
    "compute_stretch_parameters",
    "apply_mtf",
    "apply_mtf_autostretch",
    "stretch_to_uint8",
]


class StretchMethod(str, Enum):
    """Supported stretch algorithms for the live preview.

    Attributes:
        MTF: PixInsight-style Midtones Transfer Function autostretch.
            Currently the only supported method; placeholder for future
            additions (e.g. asinh, STF link channels, ...).
    """

    MTF = "mtf"


# ── Core MTF primitives ────────────────────────────────────────────────────
#
# Ported from the Stretch class (PixInsight reference algorithm) used by
# AstroLiveStacker.  Logic preserved verbatim; only the ergonomics
# (function-based API, type hints, no decorator) differ.


def _avg_dev(data: npt.NDArray[np.floating], median: float) -> float:
    """Return the average absolute deviation from ``median``.

    Args:
        data: Flat or N-D array of pixel values normalised in ``[0, 1]``.
        median: Pre-computed median of ``data`` (avoids recomputation).

    Returns:
        Average of ``|x - median|`` over all elements of ``data``.
    """
    return float(np.mean(np.abs(data - median)))


def _mtf(m: float, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Vectorised PixInsight Midtones Transfer Function.

    For each input value ``x``:

        MTF(m, 0) = 0
        MTF(m, m) = 0.5
        MTF(m, 1) = 1
        MTF(m, x) = (m - 1) * x / ((2m - 1) * x - m)   otherwise

    Args:
        m: Midtones balance parameter in ``(0, 1)``. Below 0.5 darkens
            the midtones, above 0.5 lightens them.
        x: Array of pixel values, in-place safe (a copy is made).

    Returns:
        Transformed array, same shape as ``x``.
    """
    out = x.astype(np.float32, copy=True)
    flat = out.reshape(-1)

    zeros = flat == 0
    halfs = flat == m
    ones = flat == 1
    others = ~(zeros | halfs | ones)

    flat[zeros] = 0.0
    flat[halfs] = 0.5
    flat[ones] = 1.0
    fo = flat[others]
    flat[others] = (m - 1.0) * fo / (((2.0 * m) - 1.0) * fo - m)
    return out


def compute_stretch_parameters(
    data: npt.NDArray[np.floating],
    *,
    target_bkg: float = 0.25,
    shadows_clip: float = -2.8,
) -> Tuple[float, float, float]:
    """Compute ``(shadows, midtones, highlights)`` for an MTF autostretch.

    Args:
        data: Image data normalised in ``[0, 1]`` (single channel or
            multi-channel — global statistics are used).
        target_bkg: Desired background luminance after stretch
            (PixInsight default: 0.25). Higher = brighter background.
        shadows_clip: Multiplier applied to the average absolute
            deviation to derive the shadows clipping point. Negative
            values keep faint signal; ``-2.8`` is a strong but safe
            default that matches PixInsight's STF "boosted" preset.

    Returns:
        Tuple ``(c0, m, c1)`` where ``c0`` is the shadows clip,
        ``m`` is the midtones balance and ``c1`` is the highlights
        clip (always ``1`` for the autostretch flavour).
    """
    median = float(np.median(data))
    avg_dev = _avg_dev(data, median)

    c0 = float(np.clip(median + (shadows_clip * avg_dev), 0.0, 1.0))
    # Single-element MTF call to find the midtones balance that maps the
    # background level (median - c0) to ``target_bkg``.
    m = float(_mtf(target_bkg, np.array([median - c0], dtype=np.float32))[0])
    return c0, m, 1.0


def apply_mtf(
    data: npt.NDArray[np.floating],
    *,
    shadows: float,
    midtones: float,
    highlights: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Apply an MTF stretch with explicit parameters.

    Performs shadows clipping, midtones transfer and (implicit)
    highlights clipping.  Output is in ``[0, 1]``.

    Args:
        data: Input array in ``[0, 1]``.
        shadows: Shadows clip ``c0`` — pixels below are forced to 0.
        midtones: Midtones balance ``m`` in ``(0, 1)``.
        highlights: Highlights clip ``c1`` — currently unused beyond the
            normalisation by ``(1 - shadows)``; kept for API symmetry
            with PixInsight's HistogramTransformation.

    Returns:
        Stretched float32 array, same shape as ``data``.
    """
    del highlights  # reserved for future custom highlights clipping
    d = data.astype(np.float32, copy=True)

    below = d < shadows
    above = ~below

    d[below] = 0.0
    if shadows < 1.0:
        d[above] = _mtf(midtones, (d[above] - shadows) / (1.0 - shadows))
    else:
        d[above] = 1.0
    return d


def apply_mtf_autostretch(
    data: npt.NDArray[np.floating],
    *,
    target_bkg: float = 0.25,
    shadows_clip: float = -2.8,
) -> npt.NDArray[np.float32]:
    """One-shot autostretch: compute parameters then apply them.

    Multi-channel images (HxWxC) are stretched per channel — this
    matches ALS behaviour and avoids colour casts when a single channel
    is much brighter than the others.

    Args:
        data: Input array, single-channel (HxW) or RGB (HxWx3),
            normalised in ``[0, 1]``.
        target_bkg: See :func:`compute_stretch_parameters`.
        shadows_clip: See :func:`compute_stretch_parameters`.

    Returns:
        Stretched float32 array in ``[0, 1]``, same shape as ``data``.
    """
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        out = np.empty_like(data, dtype=np.float32)
        for c in range(data.shape[-1]):
            ch = data[..., c]
            c0, m, c1 = compute_stretch_parameters(
                ch, target_bkg=target_bkg, shadows_clip=shadows_clip
            )
            out[..., c] = apply_mtf(ch, shadows=c0, midtones=m, highlights=c1)
        return out

    c0, m, c1 = compute_stretch_parameters(
        data, target_bkg=target_bkg, shadows_clip=shadows_clip
    )
    return apply_mtf(data, shadows=c0, midtones=m, highlights=c1)


def stretch_to_uint8(
    data: npt.NDArray[np.floating],
    *,
    target_bkg: float = 0.25,
    shadows_clip: float = -2.8,
) -> npt.NDArray[np.uint8]:
    """Run the autostretch and return a JPEG-ready uint8 array.

    Args:
        data: Float input in any range — it will be normalised by its
            maximum before stretching.
        target_bkg: See :func:`compute_stretch_parameters`.
        shadows_clip: See :func:`compute_stretch_parameters`.

    Returns:
        uint8 array with the same shape as ``data``, ready for
        :func:`PIL.Image.fromarray`.
    """
    peak = float(np.max(data)) if data.size else 1.0
    if peak <= 0.0:
        return np.zeros(data.shape, dtype=np.uint8)
    normalised = data.astype(np.float32) / peak
    stretched = apply_mtf_autostretch(
        normalised, target_bkg=target_bkg, shadows_clip=shadows_clip
    )
    return np.clip(stretched * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
