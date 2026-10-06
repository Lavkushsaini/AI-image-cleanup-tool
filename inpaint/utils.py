"""Image and mask utility functions for image cleanup and inpainting.

Provides utilities for:
- Image format normalization (handling RGBA transparency, palette modes).
- Intelligent resizing with aspect ratio preservation while tracking original dimensions.
- Mask binarization, cleanup, and morphological dilation.
- Color space and array conversion helpers (PIL, OpenCV BGR, float32 NumPy).
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Ensure the image is in 3-channel RGB mode, compositing alpha channels cleanly.

    Transparent areas in PNG/RGBA or palette images are composited over a solid white
    background rather than converting transparent pixels to black.

    Args:
        image: Input PIL Image in any mode (RGBA, LA, P, L, RGB, etc.).

    Returns:
        A PIL Image strictly in 'RGB' mode.
    """
    if image.mode == "RGB":
        return image

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        # Create solid white canvas to composite against
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(background, rgba)
        return composite.convert("RGB")

    return image.convert("RGB")


def resize_image_if_needed(
    image: Image.Image,
    max_side: int = 1024,
) -> Tuple[Image.Image, Tuple[int, int], float]:
    """Resize an image so its longest edge does not exceed `max_side`.

    Preserves the original aspect ratio using high-quality Lanczos resampling.
    Tracks original dimensions so the result can later be restored or referenced.

    Args:
        image: Input PIL Image.
        max_side: Maximum allowed dimension for width or height (default 1024).

    Returns:
        A tuple of:
        - resized_image: PIL Image (either scaled down or unchanged original).
        - original_size: (width, height) of the original unscaled image.
        - scale_factor: float scale multiplier applied (<= 1.0).
    """
    orig_w, orig_h = image.size
    longest = max(orig_w, orig_h)

    if longest <= max_side:
        return image.copy(), (orig_w, orig_h), 1.0

    scale = max_side / float(longest)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return resized, (orig_w, orig_h), scale


def clean_mask(
    mask_input: Union[np.ndarray, Image.Image],
    dilation_radius: int = 0,
    threshold: int = 10,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Normalize, binarize, and optionally dilate an inpainting mask.

    Handles streamlit-drawable-canvas mask output (which is typically RGBA uint8),
    grayscale arrays, or PIL images. Non-zero or alpha-drawn areas become 255 (masked
    to fill), while untouched pixels become 0 (preserved).

    Args:
        mask_input: Mask as a NumPy array (2D grayscale, 3D RGB/BGR, or 4D RGBA)
                    or a PIL Image.
        dilation_radius: Morphological dilation radius in pixels (0 to disable).
                         Expands the mask outward to cover blurred edges and halos.
        threshold: Intensity threshold above which a pixel is considered masked (0-255).
        target_size: Optional (width, height) tuple to ensure mask matches image dimensions.

    Returns:
        2D uint8 NumPy array of shape (H, W) with values 0 (background) or 255 (hole).
    """
    if isinstance(mask_input, Image.Image):
        mask_arr = np.array(mask_input)
    else:
        mask_arr = np.asarray(mask_input)

    if mask_arr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    # If mask is 4-channel RGBA (standard output from streamlit-drawable-canvas)
    if mask_arr.ndim == 3 and mask_arr.shape[2] == 4:
        # Check both alpha channel and RGB channel luminosity
        alpha = mask_arr[:, :, 3]
        rgb_max = mask_arr[:, :, :3].max(axis=2)
        # Any drawn pixel where either alpha > threshold or any color > threshold
        binary_mask = np.where((alpha > threshold) | (rgb_max > threshold), 255, 0).astype(np.uint8)
    elif mask_arr.ndim == 3 and mask_arr.shape[2] == 3:
        # 3-channel RGB/BGR: check max over channels
        rgb_max = mask_arr.max(axis=2)
        binary_mask = np.where(rgb_max > threshold, 255, 0).astype(np.uint8)
    elif mask_arr.ndim == 2:
        # Grayscale single-channel
        binary_mask = np.where(mask_arr > threshold, 255, 0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported mask shape: {mask_arr.shape}")

    # Morphological dilation using an elliptical kernel for smooth circular brush expansion
    if dilation_radius > 0:
        kernel_size = 2 * int(dilation_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Ensure mask strictly matches expected target dimensions (W, H)
    if target_size is not None:
        target_w, target_h = target_size
        if binary_mask.shape[:2] != (target_h, target_w):
            binary_mask = cv2.resize(
                binary_mask,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

    return binary_mask


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB Image to an OpenCV BGR uint8 NumPy array.

    Args:
        image: PIL RGB image.

    Returns:
        (H, W, 3) uint8 NumPy array in BGR channel order.
    """
    rgb_arr = np.array(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR uint8 NumPy array to a PIL RGB Image.

    Args:
        image_bgr: (H, W, 3) uint8 NumPy array in BGR channel order.

    Returns:
        PIL Image in RGB mode.
    """
    rgb_arr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_arr)


def pil_to_numpy_float(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a float32 NumPy array normalized to [0.0, 1.0].

    Args:
        image: PIL RGB image.

    Returns:
        (H, W, 3) float32 NumPy array with values in [0.0, 1.0].
    """
    rgb = image.convert("RGB")
    return np.asarray(rgb, dtype=np.float32) / 255.0


def numpy_float_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a float32 NumPy array with values in [0.0, 1.0] to a PIL RGB Image.

    Args:
        arr: (H, W, 3) float NumPy array.

    Returns:
        PIL Image in RGB mode.
    """
    clipped = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")
