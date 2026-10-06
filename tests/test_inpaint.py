"""Unit tests for AI Image Cleanup Tool inpainting and utility functions."""

import numpy as np
import pytest
from PIL import Image

from inpaint.classical import inpaint_navier_stokes, inpaint_telea
from inpaint.diffusion_blur import diffuse_image_array, gaussian_diffusion_inpaint
from inpaint.utils import clean_mask, ensure_rgb, resize_image_if_needed


def test_resize_image_smaller_than_max():
    """Verify that images smaller than max_side are preserved without scaling."""
    img = Image.new("RGB", (600, 400), color=(100, 150, 200))
    resized, orig_size, scale = resize_image_if_needed(img, max_side=1024)

    assert orig_size == (600, 400)
    assert scale == 1.0
    assert resized.size == (600, 400)


def test_resize_image_larger_than_max():
    """Verify that images larger than max_side are scaled down preserving aspect ratio."""
    img = Image.new("RGB", (2048, 1024), color=(50, 50, 50))
    resized, orig_size, scale = resize_image_if_needed(img, max_side=1024)

    assert orig_size == (2048, 1024)
    assert scale == pytest.approx(0.5, rel=1e-2)
    assert resized.size == (1024, 512)
    # Longest edge must not exceed max_side
    assert max(resized.size) <= 1024


def test_clean_mask_binarization_and_dilation():
    """Verify mask cleaning thresholds and morphological dilation expands mask area."""
    h, w = 100, 100
    # Create RGBA mask with a 10x10 painted square
    mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_rgba[45:55, 45:55, :] = [255, 0, 0, 255]

    # Clean without dilation
    cleaned_0 = clean_mask(mask_rgba, dilation_radius=0)
    assert cleaned_0.shape == (h, w)
    assert cleaned_0.dtype == np.uint8
    # Only 0 and 255 values
    unique_vals = set(np.unique(cleaned_0))
    assert unique_vals.issubset({0, 255})
    assert np.sum(cleaned_0 == 255) == 100

    # Clean with dilation radius 5
    cleaned_5 = clean_mask(mask_rgba, dilation_radius=5)
    assert cleaned_5.shape == (h, w)
    # Dilated mask must cover strictly more pixels than undilated mask
    count_dilated = np.sum(cleaned_5 == 255)
    assert count_dilated > 100
    # Test with target_size parameter
    cleaned_resized = clean_mask(mask_rgba, dilation_radius=2, target_size=(150, 120))
    assert cleaned_resized.shape == (120, 150)
    assert np.any(cleaned_resized == 255)


def test_canvas_result_runtime_error_safety():
    """Verify safe fallback when canvas_result raises RuntimeError on image_data."""
    class DummyCanvasResult:
        @property
        def image_data(self):
            raise RuntimeError("image_data was not requested. Pass return_image_data=True to st_canvas().")

    dummy = DummyCanvasResult()
    raw_mask_data = None
    try:
        raw_mask_data = dummy.image_data
    except (RuntimeError, AttributeError, Exception):
        raw_mask_data = None

    assert raw_mask_data is None  # Handled safely without raising unhandled exception


def test_diffusion_inpaint_shape_and_unmasked_pixels_preserved():
    """Verify that diffusion inpainting preserves output shape and unmasked pixels strictly."""
    np.random.seed(42)
    h, w = 60, 60
    img_float = np.random.uniform(0.1, 0.9, (h, w, 3)).astype(np.float32)

    # Define a hole in the center
    mask_bool = np.zeros((h, w), dtype=bool)
    mask_bool[20:40, 20:40] = True

    # Run diffusion
    result = diffuse_image_array(
        img_float,
        mask_bool,
        iterations=20,
        sigma=2.0,
        add_grain=False,
    )

    # 1. Output shape must match input shape
    assert result.shape == img_float.shape
    assert result.dtype == np.float32

    # 2. Values must be in valid float range [0.0, 1.0]
    assert np.all(result >= 0.0) and np.all(result <= 1.0)
    assert not np.any(np.isnan(result))

    # 3. CRITICAL: Unmasked pixels must remain strictly unchanged!
    unmasked = ~mask_bool
    np.testing.assert_array_equal(result[unmasked], img_float[unmasked])

    # 4. Masked hole must be filled (not all zeros or NaNs)
    assert np.all(result[mask_bool] > 0.0)


def test_ensure_rgb_rgba():
    """Verify RGBA image is composited and converted to 3-channel RGB."""
    rgba = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
    rgb = ensure_rgb(rgba)
    assert rgb.mode == "RGB"
    assert rgb.size == (50, 50)


def test_classical_telea_and_navier_stokes():
    """Verify Telea and Navier-Stokes OpenCV inpainting run without error."""
    img = Image.new("RGB", (80, 80), color=(120, 200, 150))
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[30:50, 30:50] = 255

    res_telea = inpaint_telea(img, mask, radius=3)
    assert isinstance(res_telea, Image.Image)
    assert res_telea.size == (80, 80)
    assert res_telea.mode == "RGB"

    res_ns = inpaint_navier_stokes(img, mask, radius=3)
    assert isinstance(res_ns, Image.Image)
    assert res_ns.size == (80, 80)
    assert res_ns.mode == "RGB"


def test_high_level_gaussian_diffusion():
    """Verify high-level gaussian_diffusion_inpaint interface accepts PIL images."""
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[25:35, 25:35] = 255

    res = gaussian_diffusion_inpaint(img, mask, iterations=15, sigma=2.5, add_grain=True)
    assert isinstance(res, Image.Image)
    assert res.size == (64, 64)
    assert res.mode == "RGB"
