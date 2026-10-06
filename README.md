# ✨ AI Image Cleanup Tool

An interactive Python web application for seamlessly removing unwanted objects, people, blemishes, or text from photos and naturally reconstructing the background.

Built with **Streamlit**, **OpenCV**, **NumPy**, **SciPy**, and optional GPU-accelerated **Stable Diffusion**.

---

## 🚀 Key Features

1. **Interactive Brush Canvas**: Paint freely over any unwanted object or defect directly in the browser with customizable brush radius.
2. **Multiple Inpainting Algorithms**:
   - **Fast (OpenCV Telea)**: Alexandru Telea’s Fast Marching Method for rapid scratch and blemish removal.
   - **Smooth (OpenCV Navier-Stokes)**: Partial Differential Equation fluid mechanics method that smoothly propagates continuous isophotes into the interior.
   - **Custom Gaussian Diffusion**: A custom iterative coarse-to-fine Laplace/Heat diffusion solver with localized ROI optimization and ambient sensor grain synthesis.
   - **AI (Stable Diffusion)**: Optional generative diffusion model (`diffusers` pipeline) automatically enabled when a CUDA-capable NVIDIA GPU is detected.
3. **Mask Expansion (Dilation)**: Slider (0–20px) to dilate painted masks, ensuring soft edges, anti-aliased object boundaries, and halos are completely eliminated.
4. **Instant Sample Gallery**: Pre-bundled sample images (Landscape with Traffic Cone, Tabletop Workspace with Sticky Note) ready to test with a single click.
5. **High-Performance Image Handling**: Automatically resizes oversized images (>1024px) for interactive responsiveness while maintaining aspect ratios and color profiles.
6. **Side-by-Side Comparison & Export**: Real-time Before/After visualization with execution metrics (duration in seconds, algorithm used, resolution) and 1-click lossless PNG export.
7. **Clean Reset**: Canvas key rotation to undo brush strokes or start fresh without reloading the page.

---

## 🧠 How the Inpainting Algorithms Work

### 1. OpenCV Telea (`inpaint/classical.py`)
- **Principle**: Fast Marching Method (FMM) based on Telea's 2004 algorithm.
- **Mechanism**: Iterates along the boundary of the hole moving inward along image gradient directions. For each boundary pixel, it computes a weighted average of known neighboring pixels based on Euclidean distance, level-set distance, and directional gradient coherence.
- **Best For**: Thin objects, power lines, wires, scratches, small blemishes, and text.

### 2. OpenCV Navier-Stokes (`inpaint/classical.py`)
- **Principle**: Fluid Dynamics / Partial Differential Equations (Bertalmio et al., 2001).
- **Mechanism**: Treats image intensity as the stream function of a 2D incompressible fluid. It transports the curl/vorticity of the image inward, ensuring that isophotes (lines of equal brightness) enter smoothly and continuously without kink angles at the boundary.
- **Best For**: Smooth continuous gradients, sky patches, water reflections, and soft skin tones.

### 3. Custom Gaussian Diffusion (`inpaint/diffusion_blur.py`)
- **Principle**: Iterative Heat Equation / Laplace Solver with Dirichlet Boundary Conditions:
  $$\frac{\partial u}{\partial t} = c \nabla^2 u = c \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$
- **Coarse-to-Fine Annealing**: Employs a scheduled Gaussian blur radius ($\sigma$) that begins wide ($\sigma \approx 2.5 \times \text{target}$) and anneals down to the target fine resolution. Wide sigmas rapidly transport broad ambient tones into the deep interior of the hole; smaller sigmas refine sharp boundary transitions.
- **Sensor Grain Matching**: Pure harmonic functions are mathematically smooth ($C^\infty$), which can look unnaturally airbrushed in real camera photographs. The algorithm measures the color variance ($\sigma_{\text{ambient}}$) of the surrounding boundary ring and injects subtle matched Gaussian noise to replicate natural camera sensor ISO grain.
- **ROI Acceleration**: Automatically calculates an expanded bounding box around the painted mask, filtering only the affected region for a **10x–20x execution speedup**.

### 4. AI Generative Inpainting (`inpaint/sd_inpaint.py`)
- **Principle**: Latent Diffusion Models (Stable Diffusion Inpainting).
- **Mechanism**: Encodes the masked image into latent space and conditions a U-Net denoising process on text prompts (e.g. `"clean background, natural scene, no objects"`) and negative prompts.
- **Resource Management**: Lazy-loaded on demand only when selected, with half-precision (`float16`) and attention slicing for consumer laptop GPUs. Disabled gracefully on CPU-only machines.

---

## 📁 Project Structure

```text
image-cleanup-tool/
│
├── app.py                      # Main Streamlit web application & UI
├── inpaint/
│   ├── __init__.py             # Package exports
│   ├── classical.py            # OpenCV Telea and Navier-Stokes implementations
│   ├── diffusion_blur.py       # Custom coarse-to-fine Gaussian diffusion algorithm
│   ├── sd_inpaint.py           # Optional Stable Diffusion inpainting (lazy-loaded)
│   └── utils.py                # Image resizing, RGBA handling, mask dilation & cleanup
│
├── samples/                    # Ready-to-test sample images
│   ├── sample1_landscape.jpg   # Landscape with unwanted traffic cone
│   └── sample2_tabletop.png    # Workspace desk with unwanted sticky note
│
├── tests/
│   └── test_inpaint.py         # Pytest test suite for resizing, dilation & diffusion
│
├── requirements.txt            # Core dependencies (Streamlit, OpenCV, SciPy, Pillow, NumPy)
├── requirements-ai.txt         # Optional GPU dependencies (PyTorch, Diffusers, Accelerate)
├── pytest.ini                  # Pytest configuration
├── README.md                   # Documentation & guide
└── .gitignore                  # Git ignore rules
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.10, 3.11, 3.12, 3.13)
- Windows, macOS, or Linux

### 1. Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Core Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Install GPU AI Dependencies
If you have an NVIDIA GPU with CUDA:
```bash
pip install -r requirements-ai.txt
```

### 4. Run the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Run the test suite with `pytest`:
```bash
pytest
```
All tests verify:
- Aspect-ratio preserving downscaling for images larger than 1024px.
- Binary mask thresholding and morphological dilation expansion.
- Diffusion inpainting preservation of unmasked pixels and output shape invariance.
- RGBA transparency compositing over neutral white background.
- Classical Telea and Navier-Stokes execution integrity.

---

## 🖼️ Before / After Examples

### Example 1: Landscape Object Removal
```
Before:                                 After (Custom Gaussian Diffusion):
+-----------------------------------+   +-----------------------------------+
|  🌤️ Sky & Clouds                 |   |  🌤️ Sky & Clouds                 |
|  ⛰️ Rolling Hills                 |   |  ⛰️ Rolling Hills                 |
|            [ ⚠️ Red Cone ]        |   |            (Natural Green Grass)  |
+-----------------------------------+   +-----------------------------------+
```

### Example 2: Tabletop Workspace Cleanup
```
Before:                                 After (Navier-Stokes / Telea):
+-----------------------------------+   +-----------------------------------+
|  💻 Laptop     ☕ Coffee Mug      |   |  💻 Laptop     ☕ Coffee Mug      |
|         [ 📝 Yellow Note ]        |   |         (Clean Wood Surface)      |
+-----------------------------------+   +-----------------------------------+
```

---

## ⚠️ Current Limitations

- **Large & Complex Semantic Foreground Removals**: Classical interpolation and blur-based diffusion infer color and texture exclusively from the immediate boundary. When removing large complex subjects (e.g. an entire person blocking a structured building or brick wall), generative AI mode is recommended for synthesizing novel background geometry.
- **Single Canvas Undo**: The drawable canvas resets strokes via the "Reset Canvas" button rather than multi-level stroke history.

---

## 🔮 Future Improvements

1. **One-Click Object Selection (SAM / YOLO)**: Integrate Segment Anything Model (SAM) or YOLO-world to let users click once on an object to segment its exact boundary without manual brush strokes.
2. **Video Inpainting**: Extend diffusion inpainting across sequential video frames with optical flow tracking for object removal in short video clips.
3. **Texture Synthesis PatchMatch**: Implement randomized patch-based texture replacement (similar to Photoshop Content-Aware Fill) for structured repeating patterns.
