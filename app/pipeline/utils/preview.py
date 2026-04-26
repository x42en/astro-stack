"""FITS → JPEG preview generation for per-step visual debugging.

Provides :func:`save_step_preview` which converts a FITS image to a
compressed JPEG suitable for display in the UI. The conversion runs in a
thread pool so it never blocks the async event loop.

Example:
    >>> await save_step_preview(
    ...     fits_path=Path("/output/session/process/stack_result.fit"),
    ...     output_path=Path("/output/session/previews/preprocessing.jpg"),
    ... )
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="preview_gen")

_MAX_DIM = 1200
_JPEG_QUALITY = 85
_STRETCH_LOW = 0.1
_STRETCH_HIGH = 99.9


async def save_step_preview(fits_path: Path, output_path: Path) -> None:
    """Generate a JPEG preview from a FITS image, running in a thread pool.

    Args:
        fits_path: Source FITS file (any extension: .fit, .fits, .fts).
        output_path: Destination JPEG file path (parent dir is created automatically).
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_EXECUTOR, _generate_preview, fits_path, output_path)


def _generate_preview(fits_path: Path, output_path: Path) -> None:
    """Synchronous FITS → JPEG conversion with percentile stretch.

    Handles both 2-D (grayscale) and 3-D (C × H × W RGB) FITS arrays.
    NaN/inf values are zeroed. The image is resized to at most
    :data:`_MAX_DIM` pixels on its longest side.

    Args:
        fits_path: Source FITS file.
        output_path: Destination JPEG path.
    """
    import numpy as np  # noqa: PLC0415
    from astropy.io import fits as astrofits  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with astrofits.open(str(fits_path)) as hdul:
        data = hdul[0].data

    if data is None:
        logger.warning("preview_fits_empty", path=str(fits_path))
        return

    data = data.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalise axis layout to (H, W) or (H, W, 3)
    if data.ndim == 3:
        if data.shape[0] in (1, 3):
            # FITS stores as (C, H, W) — move channels to last axis
            data = np.moveaxis(data, 0, -1)
            if data.shape[2] == 1:
                data = data[:, :, 0]  # collapse single-channel to 2-D
        # else: already (H, W, C) — unlikely from FITS but keep as-is

    # Percentile stretch to [0, 1]
    lo, hi = np.percentile(data, [_STRETCH_LOW, _STRETCH_HIGH])
    if hi <= lo:
        hi = lo + 1.0
    data = np.clip((data - lo) / (hi - lo), 0.0, 1.0)

    arr8 = (data * 255).astype(np.uint8)

    if arr8.ndim == 2:
        img = Image.fromarray(arr8, mode="L").convert("RGB")
    else:
        img = Image.fromarray(arr8, mode="RGB")

    # Resize to at most _MAX_DIM on longest side
    w, h = img.size
    if max(w, h) > _MAX_DIM:
        if w >= h:
            img = img.resize((_MAX_DIM, int(h * _MAX_DIM / w)), Image.LANCZOS)
        else:
            img = img.resize((int(w * _MAX_DIM / h), _MAX_DIM), Image.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), format="JPEG", quality=_JPEG_QUALITY)
    logger.debug("preview_saved", path=str(output_path))
