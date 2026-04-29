"""Frame I/O and pre-processing helpers for the live-stacking pipeline.

This module centralises the operations applied to every incoming frame
**before** it is merged into the running stack:

* :func:`read_frame` decodes FITS or RAW DSLR files into a normalised
  ``float32`` array in ``[0, 1]``;
* :func:`remove_hot_pixels` performs a fast median-based hot-pixel
  rejection to avoid star-like artefacts polluting the autostretch
  statistics;
* :func:`align_to_reference` uses :mod:`astroalign` to compute and
  apply an affine transform aligning the new frame to the session
  reference frame;
* :func:`accumulate_running_mean` updates an on-disk ``np.memmap``
  accumulator with a numerically stable running mean.

All functions are pure (no I/O side-effects beyond what's documented)
and synchronous; the service layer is responsible for dispatching them
to a worker thread when called from async code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "FrameReadError",
    "AlignmentError",
    "read_frame",
    "remove_hot_pixels",
    "align_to_reference",
    "accumulate_running_mean",
    "open_or_create_accumulator",
]

# Extensions handled natively by astropy (case-insensitive).
_FITS_EXTENSIONS = {".fit", ".fits", ".fts"}

# Extensions delegated to rawpy (DSLR / mirrorless RAW formats).
_RAW_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".dng", ".raf", ".rw2", ".orf", ".pef", ".rwl",
}


class FrameReadError(RuntimeError):
    """Raised when a frame cannot be decoded from disk."""


class AlignmentError(RuntimeError):
    """Raised when astroalign cannot register the frame."""


# ── Frame decoding ─────────────────────────────────────────────────────────


def _read_fits(path: Path) -> npt.NDArray[np.float32]:
    """Read a FITS file and return the primary HDU as float32.

    Multi-channel FITS (3xHxW) is transposed to HxWx3 to match the
    convention used elsewhere in the pipeline (HWC).
    """
    # Local import: astropy is heavy to import and only needed for FITS.
    from astropy.io import fits  # type: ignore[import-not-found]

    with fits.open(str(path), memmap=False) as hdul:
        data = None
        for hdu in hdul:
            if hdu.data is not None:
                data = np.asarray(hdu.data)
                break
        if data is None:
            raise FrameReadError(f"FITS file {path} contains no image data")

    arr = data.astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        # CHW → HWC
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _read_raw(path: Path) -> npt.NDArray[np.float32]:
    """Read a camera RAW file and return a debayered HxWx3 float32 array."""
    import rawpy  # type: ignore[import-not-found]

    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            output_bps=16,
            no_auto_bright=True,
            use_camera_wb=True,
            gamma=(1.0, 1.0),  # linear output for downstream stretching
            output_color=rawpy.ColorSpace.sRGB,
        )
    return rgb16.astype(np.float32, copy=False)


def read_frame(path: str | os.PathLike[str]) -> npt.NDArray[np.float32]:
    """Decode a frame from disk and return it normalised in ``[0, 1]``.

    Args:
        path: Filesystem path to a FITS or RAW DSLR file.

    Returns:
        Float32 array of shape ``(H, W)`` (mono) or ``(H, W, 3)`` (RGB),
        with values in ``[0, 1]``. Saturation is computed from the
        per-frame maximum and falls back to common bit-depth peaks when
        the frame is significantly under-exposed.

    Raises:
        FrameReadError: When the extension is unsupported or decoding
            fails.
    """
    p = Path(path)
    ext = p.suffix.lower()

    try:
        if ext in _FITS_EXTENSIONS:
            arr = _read_fits(p)
        elif ext in _RAW_EXTENSIONS:
            arr = _read_raw(p)
        else:
            raise FrameReadError(f"Unsupported frame extension: {ext}")
    except FrameReadError:
        raise
    except Exception as exc:  # pragma: no cover - delegated to libraries
        raise FrameReadError(f"Failed to decode {path}: {exc}") from exc

    # Normalise to [0, 1]. Use the most plausible saturation reference:
    # the per-frame max for floats already in range, else a power-of-two
    # bit-depth ceiling.
    peak = float(arr.max()) if arr.size else 1.0
    if peak <= 0.0:
        return arr.astype(np.float32, copy=False)
    if peak <= 1.0:
        return arr
    # Pick the smallest standard bit-depth ceiling that contains ``peak``.
    for ceiling in (255.0, 1023.0, 4095.0, 16383.0, 65535.0):
        if peak <= ceiling:
            return (arr / ceiling).astype(np.float32, copy=False)
    return (arr / peak).astype(np.float32, copy=False)


# ── Hot-pixel rejection ────────────────────────────────────────────────────


def remove_hot_pixels(
    image: npt.NDArray[np.floating],
    *,
    threshold: float = 4.0,
) -> npt.NDArray[np.float32]:
    """Replace strongly isolated bright pixels by a local 3x3 median.

    Used as a cheap, in-place sanitiser before alignment so that single
    hot pixels do not get mistaken for stars by ``astroalign``.

    Args:
        image: Single-channel or multi-channel float array.
        threshold: A pixel is flagged when its value exceeds the local
            median by ``threshold`` × the local median absolute
            deviation. Lower = more aggressive rejection.

    Returns:
        Cleaned float32 array, same shape as ``image``.
    """
    # Local import — scipy is only needed for live-stacking.
    from scipy.ndimage import median_filter  # type: ignore[import-not-found]

    arr = image.astype(np.float32, copy=True)

    def _clean(plane: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        med = median_filter(plane, size=3, mode="reflect")
        diff = plane - med
        # Robust scale ~ MAD of the local difference.
        mad = float(np.median(np.abs(diff))) or 1e-6
        hot = diff > (threshold * mad)
        plane[hot] = med[hot]
        return plane

    if arr.ndim == 3:
        for c in range(arr.shape[-1]):
            arr[..., c] = _clean(arr[..., c])
    else:
        arr = _clean(arr)
    return arr


# ── Alignment ──────────────────────────────────────────────────────────────


def _to_alignment_view(image: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Return a single-channel projection used for star-pattern matching."""
    if image.ndim == 3:
        # Luminance approximation — channel weights match Rec. 709.
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
    return image.astype(np.float32, copy=False)


def align_to_reference(
    frame: npt.NDArray[np.floating],
    reference: npt.NDArray[np.floating],
) -> Tuple[npt.NDArray[np.float32], float]:
    """Register ``frame`` onto ``reference`` using astroalign.

    Args:
        frame: New frame (HxW or HxWxC) to align.
        reference: Reference frame chosen at session start, same shape
            family as ``frame``.

    Returns:
        Tuple ``(aligned, fwhm)`` where ``aligned`` is the warped frame
        with the same shape as ``reference``, and ``fwhm`` is a coarse
        seeing estimate (mean star FWHM in pixels) — ``nan`` when the
        estimation was not possible.

    Raises:
        AlignmentError: When astroalign cannot find a matching star
            triangulation between the two frames.
    """
    import astroalign as aa  # type: ignore[import-not-found]

    src = _to_alignment_view(frame)
    dst = _to_alignment_view(reference)

    try:
        transform, (src_pts, dst_pts) = aa.find_transform(src, dst)
    except Exception as exc:  # astroalign raises a variety of errors
        raise AlignmentError(f"astroalign failed: {exc}") from exc

    if frame.ndim == 3:
        out = np.empty_like(reference, dtype=np.float32)
        for c in range(frame.shape[-1]):
            warped, _ = aa.apply_transform(transform, frame[..., c], reference[..., c])
            out[..., c] = warped
    else:
        out, _ = aa.apply_transform(transform, frame, reference)

    # Crude FWHM estimate: median pairwise nearest-neighbour distance is
    # not strictly FWHM, but provides a useful relative seeing metric.
    try:
        if len(src_pts) >= 3:
            diffs = src_pts[:, None, :] - src_pts[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            np.fill_diagonal(dists, np.inf)
            fwhm = float(np.median(dists.min(axis=1)))
        else:
            fwhm = float("nan")
    except Exception:  # pragma: no cover
        fwhm = float("nan")

    return out.astype(np.float32, copy=False), fwhm


# ── Accumulator (running mean on disk) ─────────────────────────────────────


def open_or_create_accumulator(
    path: str | os.PathLike[str],
    shape: Tuple[int, ...],
    dtype: np.dtype = np.dtype(np.float32),
) -> np.memmap:
    """Open a memmap accumulator, creating it (zero-filled) when missing.

    Args:
        path: Path to the memmap-backed file.
        shape: Expected array shape.
        dtype: Element dtype (default float32).

    Returns:
        A writable :class:`numpy.memmap` instance bound to ``path``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+" if p.exists() and p.stat().st_size == int(np.prod(shape)) * dtype.itemsize else "w+"
    mm = np.memmap(p, dtype=dtype, mode=mode, shape=shape)
    if mode == "w+":
        mm[:] = 0
        mm.flush()
    return mm


def accumulate_running_mean(
    accumulator: np.memmap,
    new_frame: npt.NDArray[np.floating],
    n_existing: int,
) -> None:
    """Update ``accumulator`` in place with a numerically stable mean.

    Implements ``mean_{n+1} = mean_n + (x_{n+1} - mean_n) / (n + 1)``,
    which avoids the precision loss of summing then dividing.

    Args:
        accumulator: Memmap holding the running mean. Modified in place
            and flushed to disk before returning.
        new_frame: Frame to fold in. Must match ``accumulator.shape``.
        n_existing: Number of frames already merged before this one.
    """
    if new_frame.shape != accumulator.shape:
        raise ValueError(
            f"frame shape {new_frame.shape} != accumulator shape {accumulator.shape}"
        )
    n_next = n_existing + 1
    np.add(
        accumulator,
        (new_frame.astype(np.float32) - accumulator) / np.float32(n_next),
        out=accumulator,
    )
    accumulator.flush()
