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
from app.pipeline.utils.display import load_fits_display_rgb, to_uint8

logger = get_logger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="preview_gen")

_MAX_DIM = 1200
_JPEG_QUALITY = 85


async def save_step_preview(
    fits_path: Path,
    output_path: Path,
    *,
    camera_defiltered: bool = True,
) -> None:
    """Generate a JPEG preview from a FITS image, running in a thread pool.

    Args:
        fits_path: Source FITS file (any extension: .fit, .fits, .fts).
        output_path: Destination JPEG file path (parent dir is created automatically).
        camera_defiltered: When ``False`` (stock DSLR) softens the per-channel
            red BP in the display stretch so the residual Hα signal survives
            sky-background subtraction.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _EXECUTOR, _generate_preview, fits_path, output_path, camera_defiltered
    )


def _generate_preview(
    fits_path: Path,
    output_path: Path,
    camera_defiltered: bool = True,
) -> None:
    """Synchronous FITS → JPEG conversion using the shared display stretch.

    Handles both 2-D (grayscale) and 3-D (C × H × W RGB) FITS arrays via
    :func:`app.pipeline.utils.display.load_fits_display_rgb`. The image is
    resized to at most :data:`_MAX_DIM` pixels on its longest side.

    Args:
        fits_path: Source FITS file.
        output_path: Destination JPEG path.
    """
    from PIL import Image  # noqa: PLC0415

    try:
        data = load_fits_display_rgb(fits_path, camera_defiltered=camera_defiltered)
    except ValueError:
        logger.warning("preview_fits_empty", path=str(fits_path))
        return

    arr8 = to_uint8(data)
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
