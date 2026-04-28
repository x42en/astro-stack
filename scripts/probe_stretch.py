"""Diagnostic probe: dump statistics from the stretch_color FITS pipeline.

Usage (inside the worker container):

    python scripts/probe_stretch.py /sessions/<session_id>

It looks at every FITS produced along the pipeline and prints the
min / median / mean / 99.7 percentile / max of each channel so we can
see exactly where the data goes south.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def _stats(arr: np.ndarray) -> dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"empty": True}  # type: ignore[dict-item]
    return {
        "min": float(np.min(finite)),
        "p005": float(np.percentile(finite, 0.5)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p997": float(np.percentile(finite, 99.7)),
        "max": float(np.max(finite)),
    }


def _inspect(path: Path) -> None:
    print(f"\n── {path} ─────────────────────────────────────────────")
    if not path.exists():
        print("  (missing)")
        return
    try:
        with fits.open(str(path)) as hdul:
            for idx, hdu in enumerate(hdul):
                if hdu.data is None:
                    continue
                data = np.asarray(hdu.data, dtype=np.float32)
                print(f"  hdu={idx} shape={data.shape} dtype={data.dtype}")
                if data.ndim == 3 and data.shape[0] in (1, 3, 4):
                    for c in range(data.shape[0]):
                        print(f"    chan {c}: {_stats(data[c])}")
                elif data.ndim == 3 and data.shape[-1] in (1, 3, 4):
                    for c in range(data.shape[-1]):
                        print(f"    chan {c}: {_stats(data[..., c])}")
                else:
                    print(f"    {_stats(data)}")
                break
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {exc!r}")


def main(session_dir: Path) -> None:
    candidates = [
        session_dir / "process" / "stack_result.fit",
        session_dir / "process" / "stack_result.fits",
        session_dir / "output" / "background_removed.fits",
        session_dir / "output" / "for_stretch.fits",
        session_dir / "output" / "for_stretch.fit",
        session_dir / "output" / "denoised.fits",
        session_dir / "output" / "sharpened.fits",
    ]
    for path in candidates:
        _inspect(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(Path(sys.argv[1]))
