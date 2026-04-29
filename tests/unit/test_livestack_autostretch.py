"""Unit tests for the live-stacking auto-stretch (MTF) module."""

from __future__ import annotations

import numpy as np
import pytest

from app.livestack.autostretch import (
    apply_mtf,
    apply_mtf_autostretch,
    compute_stretch_parameters,
    stretch_to_uint8,
)


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """Synthetic faint-signal image with a few hot pixels."""
    rng = np.random.default_rng(seed=2026)
    base = rng.normal(loc=0.05, scale=0.01, size=(128, 192)).clip(0.0, 1.0).astype(np.float32)
    # Sprinkle a handful of bright pixels so the autostretch has a non-trivial
    # high tail to clip.
    bright = rng.random(base.shape) > 0.999
    base[bright] = 0.85
    return base


def test_compute_stretch_parameters_returns_normalised_triplet(synthetic_image):
    c0, m, c1 = compute_stretch_parameters(synthetic_image)
    assert 0.0 <= c0 < 1.0
    assert 0.0 < m < 1.0
    assert c1 == 1.0


def test_apply_mtf_autostretch_targets_background(synthetic_image):
    out = apply_mtf_autostretch(synthetic_image, target_bkg=0.25)
    # The whole point of the stretch: the post-stretch median should sit
    # close to the requested background.
    assert abs(float(np.median(out)) - 0.25) < 0.05
    assert out.dtype == np.float32
    assert out.shape == synthetic_image.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_apply_mtf_explicit_params_clips_below_shadows():
    arr = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    out = apply_mtf(arr, shadows=0.3, midtones=0.5)
    # Anything < 0.3 must be clipped to zero.
    assert np.all(out[arr < 0.3] == 0.0)
    # Identity check at midtones=0.5 (MTF reduces to (x-c0)/(1-c0)).
    assert out[-1] == pytest.approx(1.0)


def test_stretch_to_uint8_handles_zero_image():
    zero = np.zeros((10, 10), dtype=np.float32)
    out = stretch_to_uint8(zero)
    assert out.dtype == np.uint8
    assert out.shape == zero.shape
    assert out.max() == 0


def test_apply_mtf_autostretch_per_channel(synthetic_image):
    rgb = np.stack([synthetic_image, synthetic_image * 0.8, synthetic_image * 1.1], axis=-1).clip(0, 1)
    out = apply_mtf_autostretch(rgb)
    assert out.shape == rgb.shape
    assert out.dtype == np.float32
    # Each channel should still target ~0.25 median.
    for c in range(3):
        assert abs(float(np.median(out[..., c])) - 0.25) < 0.07
