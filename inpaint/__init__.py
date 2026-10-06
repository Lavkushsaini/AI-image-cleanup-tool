"""AI Image Cleanup Tool - Inpainting Package.

Provides multiple inpainting algorithms:
- Classical OpenCV (Telea and Navier-Stokes)
- Custom Iterative Gaussian Diffusion
- Stable Diffusion (Optional, GPU-accelerated)
"""

from inpaint.classical import inpaint_navier_stokes, inpaint_telea
from inpaint.diffusion_blur import gaussian_diffusion_inpaint
from inpaint.sd_inpaint import inpaint_stable_diffusion, is_sd_available
from inpaint.utils import (
    clean_mask,
    cv2_to_pil,
    ensure_rgb,
    numpy_float_to_pil,
    pil_to_cv2,
    pil_to_numpy_float,
    resize_image_if_needed,
)

__all__ = [
    "inpaint_telea",
    "inpaint_navier_stokes",
    "gaussian_diffusion_inpaint",
    "inpaint_stable_diffusion",
    "is_sd_available",
    "clean_mask",
    "ensure_rgb",
    "resize_image_if_needed",
    "pil_to_cv2",
    "cv2_to_pil",
    "pil_to_numpy_float",
    "numpy_float_to_pil",
]
