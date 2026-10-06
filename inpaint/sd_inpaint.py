"""Stable Diffusion Inpainting Module (Lazy-loaded, GPU-only).

Provides integration with Hugging Face diffusers StableDiffusionInpaintPipeline.
Model weights are loaded on-demand only when requested. If GPU/CUDA is unavailable,
or if packages are missing, clear status messages and fallbacks are provided.
"""

from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image

from inpaint.utils import clean_mask, ensure_rgb

# Cache pipeline globally in memory to avoid reloading across Streamlit interactions
_CACHED_PIPE = None


def is_sd_available() -> Tuple[bool, str]:
    """Check whether PyTorch, CUDA, and Diffusers are available on this system.

    Returns:
        (available: bool, status_message: str)
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return (
            False,
            "PyTorch is not installed. Install requirements-ai.txt to enable AI mode.",
        )

    try:
        if not torch.cuda.is_available():
            return (
                False,
                "No CUDA-capable GPU detected. Stable Diffusion requires an NVIDIA GPU.",
            )
        device_name = torch.cuda.get_device_name(0)
    except Exception as e:
        return False, f"CUDA check failed: {e}"

    try:
        import diffusers  # type: ignore # noqa: F401
        import transformers  # type: ignore # noqa: F401
    except ImportError:
        return (
            False,
            "Diffusers or Transformers not installed. Run 'pip install -r requirements-ai.txt'.",
        )

    return True, f"NVIDIA GPU detected: {device_name}"


def get_sd_pipeline():
    """Lazily load and cache the Stable Diffusion inpainting pipeline on GPU.

    Returns:
        StableDiffusionInpaintPipeline loaded on 'cuda' in float16 precision.

    Raises:
        RuntimeError: If dependencies are missing, CUDA is absent, or download fails.
    """
    global _CACHED_PIPE
    if _CACHED_PIPE is not None:
        return _CACHED_PIPE

    available, msg = is_sd_available()
    if not available:
        raise RuntimeError(f"Stable Diffusion unavailable: {msg}")

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    model_ids = [
        "runwayml/stable-diffusion-inpainting",
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
    ]

    last_error: Optional[Exception] = None
    for model_id in model_ids:
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipe = pipe.to("cuda")
            # Enable memory optimization for laptop GPUs (e.g., RTX 3050 4GB/6GB)
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            _CACHED_PIPE = pipe
            return _CACHED_PIPE
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Failed to load Stable Diffusion inpainting models. Last error: {last_error}"
    )


def inpaint_stable_diffusion(
    image: Image.Image,
    mask: Union[Image.Image, np.ndarray],
    prompt: str = "clean background, natural scene, no objects",
    negative_prompt: str = "person, object, text, watermark, blur, artifacts, distorted",
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
) -> Image.Image:
    """Inpaint masked regions using Stable Diffusion.

    Args:
        image: Source PIL Image (RGB).
        mask: Mask PIL Image or NumPy array (white/non-zero = area to replace).
        prompt: Text prompt guiding the background generation.
        negative_prompt: Unwanted attributes in generated background.
        num_inference_steps: Diffusion denoising steps (default 25).
        guidance_scale: Classifier-free guidance scale (default 7.5).

    Returns:
        Inpainted PIL Image (RGB).

    Raises:
        RuntimeError: If pipeline fails to load or execution errors.
    """
    pipe = get_sd_pipeline()

    pil_img = ensure_rgb(image)

    # Clean mask and convert to PIL grayscale image for diffusers pipeline
    mask_uint8 = clean_mask(mask, dilation_radius=0)
    if mask_uint8.shape[:2] != (pil_img.height, pil_img.width):
        import cv2
        mask_uint8 = cv2.resize(
            mask_uint8,
            (pil_img.width, pil_img.height),
            interpolation=cv2.INTER_NEAREST,
        )
    pil_mask = Image.fromarray(mask_uint8, mode="L")

    # Run inference
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=max(10, int(num_inference_steps)),
        guidance_scale=float(guidance_scale),
    ).images[0]

    return result
