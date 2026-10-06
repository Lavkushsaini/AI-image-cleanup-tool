r"""Custom Gaussian Diffusion Inpainting Algorithm using NumPy and SciPy.

=============================================================================
INTERVIEW & ALGORITHM EXPLANATION:
-----------------------------------------------------------------------------
1. Physical / Mathematical Formulation:
   Inpainting can be modeled as solving the classical Heat Equation (or Laplace's
   equation in steady state):
       du/dt = c * \Delta u = c * (\partial^2 u / \partial x^2 + \partial^2 u / \partial y^2)
   subject to Dirichlet boundary conditions, where the known (unmasked) pixels
   act as fixed heat sources / boundary conditions.

2. Gaussian Convolution as Heat Kernel:
   The fundamental solution (Green's function) of the 2D heat equation in free space
   is a 2D Gaussian distribution:
       G_sigma(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
   Applying a Gaussian blur is mathematically equivalent to evolving the heat equation
   for time t = sigma^2 / 2.

3. Iterative Relaxation with Boundary Clamping:
   In each iteration:
   - The entire image is filtered with a 2D Gaussian kernel across RGB channels.
   - The blurred pixels are copied ONLY into the masked region (the "hole").
   - The unmasked region is strictly clamped back to its original values.
   This progressively diffuses color gradients inward from the boundary.

4. Coarse-to-Fine Annealing:
   If a fixed small sigma is used, diffusion into large masked areas takes thousands
   of iterations and can leave the center flat or disconnected. By scheduling
   sigma from large (coarse) to small (fine):
   - Early iterations (high sigma) propagate global ambient illumination and broad
     color tones deep into the center of the masked hole.
   - Late iterations (low sigma) preserve sharp local gradients and seamless color
     transitions right along the boundary edges.

5. Ambient Noise / Grain Injection:
   Harmonic functions (the steady state of heat diffusion) are infinitely smooth (C^infinity).
   In real camera photographs, completely smooth regions look unnaturally plastic
   and airbrushed. By measuring the variance (standard deviation) of the surrounding
   pixels and injecting subtle matched Gaussian grain, the filled area seamlessly blends
   into natural sensor noise and optical texture.
=============================================================================
"""

from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from inpaint.utils import (
    clean_mask,
    ensure_rgb,
    numpy_float_to_pil,
    pil_to_numpy_float,
)


def diffuse_image_array(
    image: np.ndarray,
    mask: np.ndarray,
    iterations: int = 150,
    sigma: float = 3.0,
    add_grain: bool = True,
    grain_strength: float = 0.25,
) -> np.ndarray:
    """Run iterative Gaussian diffusion inpainting on a float32 image array.

    Args:
        image: (H, W, 3) float array with color values normalized to [0.0, 1.0].
        mask: (H, W) boolean array where True indicates pixels to fill/inpaint,
              and False indicates known pixels to strictly preserve.
        iterations: Number of diffusion steps (default 150).
        sigma: Target fine-scale Gaussian standard deviation (default 3.0).
        add_grain: Whether to inject subtle texture noise matching the surroundings.
        grain_strength: Noise injection factor relative to surrounding variance (0.0 to 1.0).

    Returns:
        (H, W, 3) float32 array in [0.0, 1.0] with masked region inpainted.
    """
    # Defensive copies and shape checks
    img_float = np.asarray(image, dtype=np.float32).copy()
    mask_bool = np.asarray(mask, dtype=bool)

    if img_float.ndim != 3 or img_float.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image array, got {img_float.shape}")

    if mask_bool.shape[:2] != img_float.shape[:2]:
        raise ValueError(
            f"Image shape {img_float.shape[:2]} does not match mask shape {mask_bool.shape[:2]}"
        )

    # Edge case 1: Nothing to inpaint
    if not np.any(mask_bool):
        return img_float

    # Edge case 2: Everything is masked (no known pixels)
    if np.all(mask_bool):
        return np.full_like(img_float, 0.5)

    # -------------------------------------------------------------------------
    # STEP 1: Analyze surrounding boundary region for mean color and texture
    # -------------------------------------------------------------------------
    # Create a boundary ring around the mask (dilate mask slightly and subtract mask)
    # This isolates known pixels directly bordering the object to remove.
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dilated_mask = cv2.dilate(mask_bool.astype(np.uint8), dilation_kernel) > 0
    nearby_unmasked = dilated_mask & (~mask_bool)

    # Compute mean color and standard deviation of surrounding known pixels
    if np.any(nearby_unmasked):
        surrounding_mean = img_float[nearby_unmasked].mean(axis=0)
        surrounding_std = img_float[nearby_unmasked].std(axis=0)
    else:
        # Fallback to whole known image if mask covers entire border
        surrounding_mean = img_float[~mask_bool].mean(axis=0)
        surrounding_std = img_float[~mask_bool].std(axis=0)

    # -------------------------------------------------------------------------
    # STEP 2: Initialize masked pixels with ambient mean color
    # -------------------------------------------------------------------------
    working = img_float.copy()
    working[mask_bool] = surrounding_mean

    # -------------------------------------------------------------------------
    # STEP 3: Coarse-to-fine Gaussian diffusion schedule
    # -------------------------------------------------------------------------
    # Start with a larger sigma to diffuse broad color fields across large distances,
    # then anneal down to the target sigma for local boundary precision.
    start_sigma = max(float(sigma) * 2.5, 6.0)
    end_sigma = max(float(sigma), 0.8)
    sigma_schedule = np.linspace(start_sigma, end_sigma, max(1, int(iterations)))

    # ROI optimization: Diffusion only impacts the mask and pixels within a few sigmas
    # Computing only on the bounding box gives a massive 10x-30x speedup for typical brush strokes!
    ys, xs = np.where(mask_bool)
    margin = int(np.ceil(start_sigma * 4.0)) + 20
    y1 = max(0, int(np.min(ys)) - margin)
    y2 = min(img_float.shape[0], int(np.max(ys)) + margin + 1)
    x1 = max(0, int(np.min(xs)) - margin)
    x2 = min(img_float.shape[1], int(np.max(xs)) + margin + 1)

    sub_working = working[y1:y2, x1:x2].copy()
    sub_orig = img_float[y1:y2, x1:x2]
    sub_mask = mask_bool[y1:y2, x1:x2]
    sub_blurred = np.empty((y2 - y1, x2 - x1), dtype=np.float32)

    for step_idx in range(int(iterations)):
        cur_sigma = float(sigma_schedule[step_idx])

        # Gaussian blur across each color channel independently
        # mode='reflect' provides smooth boundary reflection without dark edge vignetting
        for c in range(3):
            gaussian_filter(sub_working[:, :, c], sigma=cur_sigma, mode="reflect", output=sub_blurred)
            # STEP 4: Copy blurred values ONLY into the masked region
            sub_working[sub_mask, c] = sub_blurred[sub_mask]

        # STEP 5: Strictly preserve original known pixels within the patch
        sub_working[~sub_mask] = sub_orig[~sub_mask]

    # -------------------------------------------------------------------------
    # STEP 6: Optional texture / grain synthesis
    # -------------------------------------------------------------------------
    if add_grain and grain_strength > 0.0:
        # Generate Gaussian noise scaled by the surrounding area's variance
        # This breaks up synthetic flatness and mimics camera sensor ISO grain
        noise_scale = surrounding_std * (float(grain_strength) * 0.35)
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=sub_working.shape).astype(np.float32)

        # Apply noise only inside the inpainted mask
        sub_working[sub_mask] = np.clip(sub_working[sub_mask] + noise[sub_mask], 0.0, 1.0)

        # Gentle post-smoothing step (sigma 0.6) so the noise feels like film grain
        # rather than harsh independent pixel noise
        for c in range(3):
            gaussian_filter(sub_working[:, :, c], sigma=0.6, mode="reflect", output=sub_blurred)
            sub_working[sub_mask, c] = sub_blurred[sub_mask]

        sub_working[~sub_mask] = sub_orig[~sub_mask]

    # Write inpainted patch back into the full canvas
    working[y1:y2, x1:x2] = sub_working

    # Final guarantee: Unmasked pixels remain bit-for-bit identical to the input
    working[~mask_bool] = img_float[~mask_bool]
    return np.clip(working, 0.0, 1.0)


def gaussian_diffusion_inpaint(
    image: Union[Image.Image, np.ndarray],
    mask: Union[Image.Image, np.ndarray],
    iterations: int = 150,
    sigma: float = 3.0,
    add_grain: bool = True,
    grain_strength: float = 0.25,
) -> Image.Image:
    """High-level inpainting interface for Custom Gaussian Diffusion.

    Accepts PIL Images or NumPy arrays and returns an inpainted PIL RGB Image.

    Args:
        image: PIL Image or (H, W, 3) NumPy array.
        mask: Mask PIL Image or NumPy array (non-zero = fill, zero = keep).
        iterations: Number of diffusion iterations (default 150).
        sigma: Target fine-scale Gaussian sigma (default 3.0).
        add_grain: Whether to inject subtle matched noise to mimic photograph grain.
        grain_strength: Strength multiplier for grain noise (default 0.25).

    Returns:
        Inpainted PIL Image (RGB).
    """
    if isinstance(image, Image.Image):
        pil_img = ensure_rgb(image)
        img_float = pil_to_numpy_float(pil_img)
    else:
        img_float = np.asarray(image, dtype=np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

    binary_mask_uint8 = clean_mask(mask, dilation_radius=0)

    # Match spatial dimensions if needed
    if binary_mask_uint8.shape[:2] != img_float.shape[:2]:
        binary_mask_uint8 = cv2.resize(
            binary_mask_uint8,
            (img_float.shape[1], img_float.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_bool = binary_mask_uint8 > 0

    result_float = diffuse_image_array(
        image=img_float,
        mask=mask_bool,
        iterations=iterations,
        sigma=sigma,
        add_grain=add_grain,
        grain_strength=grain_strength,
    )

    return numpy_float_to_pil(result_float)
