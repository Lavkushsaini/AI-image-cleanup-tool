"""AI Image Cleanup Tool - Streamlit Web Application.

Provides an interactive web interface for removing unwanted objects and people
from photos using:
1. Fast OpenCV Telea Inpainting
2. Smooth OpenCV Navier-Stokes Inpainting
3. Custom Coarse-to-Fine Gaussian Diffusion Inpainting
4. Optional Stable Diffusion AI Inpainting (GPU-enabled)
"""

import io
import os
import time
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from inpaint.classical import inpaint_navier_stokes, inpaint_telea
from inpaint.diffusion_blur import gaussian_diffusion_inpaint
from inpaint.sd_inpaint import inpaint_stable_diffusion, is_sd_available
from inpaint.utils import clean_mask, ensure_rgb, resize_image_if_needed

# Set page configuration
st.set_page_config(
    page_title="AI Image Cleanup Tool",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern CSS styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-gpu-on {
        background-color: #DCFCE7;
        color: #166534;
    }
    .badge-gpu-off {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initialize necessary Streamlit session state variables."""
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "original_size" not in st.session_state:
        st.session_state.original_size = None
    if "result_image" not in st.session_state:
        st.session_state.result_image = None
    if "last_stats" not in st.session_state:
        st.session_state.last_stats = None
    if "image_source_name" not in st.session_state:
        st.session_state.image_source_name = "None"


def load_image_from_file_or_path(file_or_path) -> None:
    """Load an image, convert to RGB, resize if over 1024px, and update session state."""
    try:
        raw_img = Image.open(file_or_path)
        rgb_img = ensure_rgb(raw_img)
        resized_img, orig_dims, scale = resize_image_if_needed(rgb_img, max_side=1024)

        st.session_state.current_image = resized_img
        st.session_state.original_size = orig_dims
        st.session_state.result_image = None
        st.session_state.last_stats = None
        st.session_state.canvas_key += 1
    except Exception as e:
        st.error(f"Failed to load image: {e}")


def reset_canvas() -> None:
    """Reset canvas strokes and clear previous results."""
    st.session_state.canvas_key += 1
    st.session_state.result_image = None
    st.session_state.last_stats = None


def main() -> None:
    """Main application routine."""
    init_session_state()

    # Header section
    st.markdown('<div class="main-header">✨ AI Image Cleanup Tool</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Paint over unwanted objects, people, or imperfections to seamlessly erase them from your photos.</div>',
        unsafe_allow_html=True,
    )

    # Check GPU / Stable Diffusion availability
    sd_available, sd_status_msg = is_sd_available()

    # =========================================================================
    # SIDEBAR: CONTROLS & SETTINGS
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Inpainting Controls")

        # 1. Method Selector
        method_options = [
            "Fast (OpenCV Telea)",
            "Smooth (OpenCV Navier-Stokes)",
            "Custom Gaussian Diffusion",
        ]
        if sd_available:
            method_options.append("AI (Stable Diffusion)")
            st.markdown(
                f'<span class="status-badge badge-gpu-on">⚡ {sd_status_msg}</span>',
                unsafe_allow_html=True,
            )
        else:
            method_options.append("AI (Stable Diffusion) [Requires GPU]")
            st.markdown(
                '<span class="status-badge badge-gpu-off">ℹ️ GPU Inactive: CPU Classical & Custom Diffusion Active</span>',
                unsafe_allow_html=True,
            )

        selected_method = st.selectbox(
            "Select Inpainting Method",
            options=method_options,
            index=2,  # Default to Custom Gaussian Diffusion
            help="Choose between ultra-fast classical boundary methods or custom coarse-to-fine diffusion.",
        )

        is_disabled_ai = selected_method == "AI (Stable Diffusion) [Requires GPU]"
        if is_disabled_ai:
            st.info(f"💡 Stable Diffusion is unavailable: {sd_status_msg}")

        # 2. Brush Configuration
        st.subheader("🖌️ Brush & Mask Tools")
        brush_size = st.slider(
            "Brush Size (px)",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            help="Adjust the thickness of the canvas drawing brush.",
        )

        mask_expansion = st.slider(
            "Mask Expansion (Dilate px)",
            min_value=0,
            max_value=20,
            value=4,
            step=1,
            help="Expands the painted mask outwards to completely cover blurred fringes and object borders.",
        )

        # 3. Method-specific Hyperparameters
        if selected_method == "Custom Gaussian Diffusion":
            st.subheader("🌊 Diffusion Parameters")
            diff_iterations = st.slider(
                "Iterations",
                min_value=20,
                max_value=300,
                value=150,
                step=10,
                help="More iterations allow color gradients to propagate deeper into large holes.",
            )
            diff_sigma = st.slider(
                "Target Sigma",
                min_value=1.0,
                max_value=8.0,
                value=3.0,
                step=0.5,
                help="Base Gaussian blur radius for boundary blending.",
            )
            add_grain = st.checkbox(
                "Synthesize Sensor Grain / Noise",
                value=True,
                help="Matches ambient texture variance so the inpainted patch does not look unnaturally flat.",
            )
            grain_strength = 0.25
            if add_grain:
                grain_strength = st.slider("Grain Intensity", 0.05, 0.60, 0.25, 0.05)

        elif selected_method in ("Fast (OpenCV Telea)", "Smooth (OpenCV Navier-Stokes)"):
            st.subheader("🎯 OpenCV Parameters")
            cv_radius = st.slider(
                "Inpaint Radius (px)",
                min_value=1,
                max_value=15,
                value=3,
                step=1,
                help="Radius of circular neighborhood around each point considered for interpolation.",
            )

        elif selected_method == "AI (Stable Diffusion)":
            st.subheader("🤖 Stable Diffusion Parameters")
            sd_prompt = st.text_input(
                "Prompt",
                value="clean background, natural scene, no objects",
                help="Guidance prompt for the generative fill.",
            )
            sd_neg_prompt = st.text_input(
                "Negative Prompt",
                value="person, object, text, watermark, blur, artifacts",
            )
            sd_steps = st.slider("Inference Steps", 15, 50, 25, 5)

        st.markdown("---")
        # Undo / Start Over button
        if st.button("🔄 Reset Canvas / Start Over", use_container_width=True):
            reset_canvas()
            st.rerun()

    # =========================================================================
    # MAIN AREA: IMAGE SELECTION & SAMPLES
    # =========================================================================
    col_upload, col_samples = st.columns([3, 2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload an Image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Images larger than 1024px are automatically downscaled for fast, responsive inpainting.",
        )
        if uploaded_file is not None:
            # Check if this is a newly uploaded file
            if st.session_state.image_source_name != uploaded_file.name:
                st.session_state.image_source_name = uploaded_file.name
                load_image_from_file_or_path(uploaded_file)

    with col_samples:
        st.markdown("##### Or Try A Built-in Sample:")
        col_s1, col_s2 = st.columns(2)
        sample1_path = os.path.join("samples", "sample1_landscape.jpg")
        sample2_path = os.path.join("samples", "sample2_tabletop.png")

        with col_s1:
            if st.button("🏞️ Landscape Cone", use_container_width=True):
                if os.path.exists(sample1_path):
                    st.session_state.image_source_name = "sample1_landscape.jpg"
                    load_image_from_file_or_path(sample1_path)
                    st.rerun()

        with col_s2:
            if st.button("☕ Table Sticky Note", use_container_width=True):
                if os.path.exists(sample2_path):
                    st.session_state.image_source_name = "sample2_tabletop.png"
                    load_image_from_file_or_path(sample2_path)
                    st.rerun()

    # Ensure an image is loaded before displaying canvas
    current_img = st.session_state.current_image
    if current_img is None:
        st.info("👆 Please upload an image or choose one of the sample images above to begin.")
        # Algorithm explanation cards for empty state
        st.markdown("### 🔍 How Inpainting Algorithms Work")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                **⚡ Fast (OpenCV Telea)**
                - Fast Marching Method (FMM).
                - Propagates image values from boundary inward along gradient normal vectors.
                - Best for: small blemishes, lines, wires, scratches.
                """
            )
        with c2:
            st.markdown(
                """
                **🌊 Smooth (OpenCV Navier-Stokes)**
                - Fluid dynamics formulation.
                - Propagates continuous isophote contours into the interior hole.
                - Best for: smooth gradients, skies, and organic skin tones.
                """
            )
        with c3:
            st.markdown(
                """
                **🔬 Custom Gaussian Diffusion**
                - Solves the Heat / Laplace equation iteratively with coarse-to-fine annealing.
                - Injects ambient variance to match photograph sensor grain.
                - Best for: balanced gradient fills without hard edge seam artifacts.
                """
            )
        return

    # Image info badge
    orig_w, orig_h = st.session_state.original_size
    curr_w, curr_h = current_img.size
    resize_note = f" (Resized from {orig_w}×{orig_h})" if (orig_w, orig_h) != (curr_w, curr_h) else ""
    st.caption(f"Active Image: **{st.session_state.image_source_name}** | Size: **{curr_w}×{curr_h} px**{resize_note}")

    # =========================================================================
    # DRAWING CANVAS & INTERACTION
    # =========================================================================
    st.markdown("#### 1. Paint over the object you want to remove:")
    st.caption("Use your mouse or touchpad to brush over the object. Make sure the stroke covers the object edges.")

    # Calculate canvas display dimensions
    # Streamlit canvas displays at native image dimensions
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.4)",
        stroke_width=brush_size,
        stroke_color="rgba(255, 30, 30, 0.8)",
        background_image=current_img,
        update_streamlit=True,
        height=current_img.height,
        width=current_img.width,
        drawing_mode="freedraw",
        point_display_radius=0,
        return_image_data=True,
        key=f"canvas_{st.session_state.canvas_key}",
    )

    # Action buttons
    col_action1, col_action2 = st.columns([2, 5])
    with col_action1:
        process_clicked = st.button("🚀 Remove Object", type="primary", use_container_width=True)

    with col_action2:
        preview_mask = st.checkbox("Preview Processed Mask (with dilation)", value=False)

    # Processed mask generation
    processed_mask: Optional[np.ndarray] = None
    mask_drawn = False
    raw_mask_data = None

    if canvas_result is not None:
        try:
            raw_mask_data = canvas_result.image_data
        except (RuntimeError, AttributeError, Exception):
            raw_mask_data = None

    if raw_mask_data is not None:
        # Clean, dilate, and ensure mask matches displayed image size exactly
        processed_mask = clean_mask(
            raw_mask_data,
            dilation_radius=mask_expansion,
            target_size=(current_img.width, current_img.height),
        )
        mask_drawn = bool(np.any(processed_mask > 0))

    if preview_mask and processed_mask is not None and mask_drawn:
        st.image(
            processed_mask,
            caption=f"Processed Binary Mask (Dilated +{mask_expansion}px)",
            width=min(400, current_img.width),
        )

    # =========================================================================
    # INPAINTING EXECUTION
    # =========================================================================
    if process_clicked:
        if not mask_drawn or processed_mask is None:
            st.warning("⚠️ Please paint over an unwanted object on the image before clicking 'Remove Object'.")
        elif is_disabled_ai:
            st.error(f"Cannot run AI mode: {sd_status_msg}. Please select one of the non-AI methods.")
        else:
            with st.spinner(f"Inpainting with {selected_method}... Please wait."):
                start_time = time.perf_counter()
                result_pil: Optional[Image.Image] = None
                method_name_used = selected_method

                try:
                    if selected_method == "Fast (OpenCV Telea)":
                        result_pil = inpaint_telea(current_img, processed_mask, radius=cv_radius)

                    elif selected_method == "Smooth (OpenCV Navier-Stokes)":
                        result_pil = inpaint_navier_stokes(current_img, processed_mask, radius=cv_radius)

                    elif selected_method == "Custom Gaussian Diffusion":
                        result_pil = gaussian_diffusion_inpaint(
                            image=current_img,
                            mask=processed_mask,
                            iterations=diff_iterations,
                            sigma=diff_sigma,
                            add_grain=add_grain,
                            grain_strength=grain_strength,
                        )

                    elif selected_method == "AI (Stable Diffusion)":
                        try:
                            result_pil = inpaint_stable_diffusion(
                                image=current_img,
                                mask=processed_mask,
                                prompt=sd_prompt,
                                negative_prompt=sd_neg_prompt,
                                num_inference_steps=sd_steps,
                            )
                        except Exception as ai_err:
                            st.warning(
                                f"⚠️ Stable Diffusion encountered an error ({ai_err}). Falling back to OpenCV Navier-Stokes."
                            )
                            result_pil = inpaint_navier_stokes(current_img, processed_mask, radius=3)
                            method_name_used = "OpenCV Navier-Stokes (Fallback)"

                    elapsed = time.perf_counter() - start_time
                    st.session_state.result_image = result_pil
                    st.session_state.last_stats = {
                        "elapsed": elapsed,
                        "method": method_name_used,
                    }

                except Exception as err:
                    st.error(f"An unexpected error occurred during processing: {err}")

    # =========================================================================
    # BEFORE / AFTER RESULTS DISPLAY & DOWNLOAD
    # =========================================================================
    if st.session_state.result_image is not None and st.session_state.last_stats is not None:
        st.markdown("---")
        st.markdown("### 🖼️ Results")

        stats = st.session_state.last_stats
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Method Used", stats["method"])
        with col_m2:
            st.metric("Processing Time", f"{stats['elapsed']:.3f} s")
        with col_m3:
            st.metric("Resolution", f"{current_img.width} × {current_img.height} px")

        # Side-by-side Before / After comparison
        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("##### ⬅️ Before (Original)")
            st.image(current_img, use_container_width=True)

        with col_after:
            st.markdown(f"##### ➡️ After ({stats['method']})")
            st.image(st.session_state.result_image, use_container_width=True)

        # Download button
        buf = io.BytesIO()
        st.session_state.result_image.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="💾 Download Cleaned Image (PNG)",
            data=buf.getvalue(),
            file_name="cleaned_result.png",
            mime="image/png",
            type="primary",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
