"""Classical OpenCV inpainting methods: Telea and Navier-Stokes.

These methods reconstruct damaged or unwanted image regions using partial differential
equations and fast marching methods based on boundary pixel gradients.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image

from inpaint.utils import clean_mask, cv2_to_pil, pil_to_cv2


def inpaint_telea(
    image: Image.Image,
    mask: Union[np.ndarray, Image.Image],
    radius: int = 3,
) -> Image.Image:
    """Inpaint masked regions using Alexandru Telea's Fast Marching Method.

    Telea's algorithm (2004) propagates image information from the boundary inward
    along the gradient direction. It estimates pixel values as a weighted sum of
    known neighboring pixels, weighted by distance, level set direction, and
    gradient coherence. It is computationally fast and well-suited for narrow
    blemishes, scratches, text removal, and small objects.

    Args:
        image: Source PIL Image (RGB).
        mask: Inpainting mask where non-zero pixels represent the area to remove.
        radius: Neighborhood radius in pixels around each point considered for inpainting (default 3).

    Returns:
        Inpainted PIL Image (RGB).
    """
    img_bgr = pil_to_cv2(image)
    binary_mask = clean_mask(mask, dilation_radius=0)

    # Ensure mask has matching spatial dimensions (H, W)
    if binary_mask.shape[:2] != img_bgr.shape[:2]:
        binary_mask = cv2.resize(
            binary_mask,
            (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # cv2.INPAINT_TELEA requires 8-bit 1-channel mask
    result_bgr = cv2.inpaint(img_bgr, binary_mask, inpaintRadius=max(1, int(radius)), flags=cv2.INPAINT_TELEA)
    return cv2_to_pil(result_bgr)


def inpaint_navier_stokes(
    image: Image.Image,
    mask: Union[np.ndarray, Image.Image],
    radius: int = 3,
) -> Image.Image:
    """Inpaint masked regions using the Navier-Stokes (Fluid Dynamics) method.

    Based on the method by Bertalmio, Sapiro, Caselles, and Ballester (2001), this
    approach treats pixel intensity as a stream function in 2D fluid dynamics.
    It propagates isophotes (lines of constant brightness/intensity) smoothly
    from the boundary into the interior while matching gradient directions.
    It produces smoother transitions than Telea, making it ideal for smooth gradients
    and organic surfaces.

    Args:
        image: Source PIL Image (RGB).
        mask: Inpainting mask where non-zero pixels represent the area to remove.
        radius: Neighborhood radius in pixels around each point considered for inpainting (default 3).

    Returns:
        Inpainted PIL Image (RGB).
    """
    img_bgr = pil_to_cv2(image)
    binary_mask = clean_mask(mask, dilation_radius=0)

    # Ensure mask has matching spatial dimensions (H, W)
    if binary_mask.shape[:2] != img_bgr.shape[:2]:
        binary_mask = cv2.resize(
            binary_mask,
            (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # cv2.INPAINT_NS requires 8-bit 1-channel mask
    result_bgr = cv2.inpaint(img_bgr, binary_mask, inpaintRadius=max(1, int(radius)), flags=cv2.INPAINT_NS)
    return cv2_to_pil(result_bgr)
