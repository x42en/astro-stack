"""JPEG preview encoding for the live-stack pipeline.

The preview is the user-facing artefact: a stretched, JPEG-encoded
snapshot of the running stack served over HTTP and refreshed via a
WebSocket event.  We always write through a temporary file and rename
into place so concurrent readers never observe a half-written JPEG.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["encode_preview_jpeg"]


def encode_preview_jpeg(
    image_uint8: npt.NDArray[np.uint8],
    output_path: str | os.PathLike[str],
    *,
    quality: int = 90,
) -> None:
    """Atomically write an 8-bit image as a JPEG preview.

    Args:
        image_uint8: 2-D (mono) or 3-D (HxWx3 RGB) uint8 array.
        output_path: Destination JPEG path. Parent directory is created
            if missing. The write goes through ``<path>.tmp`` then is
            ``os.replace``-d into place to guarantee atomicity.
        quality: JPEG quality, 1-95. Default 90 (visually lossless on
            stretched astro data while keeping previews under ~1 MB).
    """
    from PIL import Image  # type: ignore[import-not-found]

    if image_uint8.dtype != np.uint8:
        raise TypeError(f"expected uint8 array, got {image_uint8.dtype}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if image_uint8.ndim == 2:
        img = Image.fromarray(image_uint8, mode="L")
    elif image_uint8.ndim == 3 and image_uint8.shape[-1] == 3:
        img = Image.fromarray(image_uint8, mode="RGB")
    elif image_uint8.ndim == 3 and image_uint8.shape[-1] == 4:
        # Drop alpha for JPEG (no transparency support).
        img = Image.fromarray(image_uint8[..., :3], mode="RGB")
    else:
        raise ValueError(f"unsupported preview shape: {image_uint8.shape}")

    tmp = out.with_suffix(out.suffix + ".tmp")
    img.save(tmp, format="JPEG", quality=quality, optimize=True, progressive=False)
    os.replace(tmp, out)
