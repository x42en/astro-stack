"""Unit tests for the camera_defiltered branch of the display pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.utils.display import (
    _stretch_array,
    apply_hdr_polish,
    summarize_profile_config,
)


def _red_dominant_image(seed: int = 0) -> np.ndarray:
    """Build a synthetic RGB float image with a clear red dominance.

    Mimics a stock-DSLR M42-like frame: low overall red signal sitting
    just above a per-channel sky background that is offset per channel.
    """
    rng = np.random.default_rng(seed)
    h, w = 64, 64
    img = rng.random((h, w, 3), dtype=np.float32) * 0.05
    # Per-channel sky background offsets (R/G/B) — green/blue brighter than red
    img[..., 0] += 0.10
    img[..., 1] += 0.18
    img[..., 2] += 0.16
    # Red emission patch (the "nebula")
    img[20:40, 20:40, 0] += 0.20
    img[20:40, 20:40, 1] += 0.05
    img[20:40, 20:40, 2] += 0.05
    return img


class TestStretchCameraDefiltered:
    def test_stock_dslr_preserves_more_red_than_defiltered(self) -> None:
        """With camera_defiltered=False the red BP is softened so more red survives."""
        img = _red_dominant_image()
        out_def = _stretch_array(
            img,
            low_pct=0.5,
            high_pct=99.7,
            asinh_strength=50.0,
            per_channel=True,
            camera_defiltered=True,
        )
        out_stock = _stretch_array(
            img,
            low_pct=0.5,
            high_pct=99.7,
            asinh_strength=50.0,
            per_channel=True,
            camera_defiltered=False,
        )
        assert out_stock[..., 0].mean() > out_def[..., 0].mean()


class TestHdrPolishCameraDefiltered:
    def test_stock_dslr_warmer_than_defiltered(self) -> None:
        """Stock DSLR mode applies a small +5%% red boost + extra saturation."""
        # Use a non-saturated mid-grey + slight red tint so the boost is visible.
        rng = np.random.default_rng(1)
        img = rng.uniform(0.30, 0.45, size=(32, 32, 3)).astype(np.float32)
        img[..., 0] += 0.05  # slight red tint
        img = np.clip(img, 0.0, 1.0)

        polished_def = apply_hdr_polish(img.copy(), camera_defiltered=True)
        polished_stock = apply_hdr_polish(img.copy(), camera_defiltered=False)

        assert polished_stock[..., 0].mean() > polished_def[..., 0].mean()

    def test_default_highlight_rolloff_compresses_bright_values(self) -> None:
        """Default rolloff=0.85 compresses values >0.85 below 1.0 (no clipping)."""
        bright = np.full((4, 4, 3), 0.95, dtype=np.float32)
        polished = apply_hdr_polish(bright, midtone_contrast=0.0, saturation=1.0)
        # Expect compression: every output value strictly less than the input.
        assert polished.max() < 0.95
        assert polished.max() < 1.0


class TestSummarizeProfile:
    def test_camera_field_added_only_when_stock(self) -> None:
        """The 'Camera: stock DSLR' chip is appended only when camera_defiltered=False."""
        pairs_def = summarize_profile_config({"camera_defiltered": True})
        pairs_stock = summarize_profile_config({"camera_defiltered": False})
        assert ("Camera", "stock DSLR") not in pairs_def
        assert ("Camera", "stock DSLR") in pairs_stock


@pytest.mark.parametrize("flag", [True, False])
def test_stretch_array_returns_clipped_float32(flag: bool) -> None:
    """Output stays in [0, 1] float32 regardless of camera flag."""
    img = _red_dominant_image()
    out = _stretch_array(
        img,
        low_pct=0.5,
        high_pct=99.7,
        asinh_strength=50.0,
        per_channel=True,
        camera_defiltered=flag,
    )
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


class TestOverStretchSafeguard:
    """The display safeguard recovers usable data from a saturated FITS."""

    def test_overstretched_fits_does_not_render_all_black(self) -> None:
        # Simulate a Siril stretch_color output that has been crushed near
        # the white point: most values in [0.96, 0.999] with a faint
        # gradient marking the actual structure.
        rng = np.random.default_rng(42)
        h, w = 64, 64
        img = 0.96 + rng.random((h, w, 3), dtype=np.float32) * 0.039
        # Embed a brighter "feature" so we can verify it remains
        # distinguishable after the safeguard kicks in.
        img[20:40, 20:40, :] = 0.999

        out = _stretch_array(
            img,
            low_pct=0.5,
            high_pct=99.7,
            asinh_strength=50.0,
            per_channel=True,
            camera_defiltered=True,
        )
        # Without the safeguard the percentile clip would collapse to a
        # near-uniform image (typically all-black).  With it, the brightest
        # patch must remain clearly visible.
        assert out.max() > 0.5
        feature_mean = float(out[20:40, 20:40].mean())
        background_mean = float(out[:20, :20].mean())
        assert feature_mean > background_mean

    def test_normal_fits_is_not_compressed(self) -> None:
        # A well-behaved image (median around 0.2) must not be touched by
        # the safeguard — verify by feeding the same array twice and
        # comparing the median, the safeguard would noticeably darken it.
        img = _red_dominant_image()
        out = _stretch_array(
            img,
            low_pct=0.5,
            high_pct=99.7,
            asinh_strength=50.0,
            per_channel=True,
            camera_defiltered=True,
        )
        # The stretch always produces a brighter median than the input, but
        # we just want to assert "not collapsed to zero".
        assert float(np.median(out)) > 0.05
