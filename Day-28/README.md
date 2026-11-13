# 🎨 Awais Studio: Computer Vision Image Processing App

Awais Studio brings the image-processing techniques from the course into one application. You upload an image, pick an operation from a dropdown, tune it with live preview, compare before and after, chain several filters into a pipeline, and download the result.

The project ships with **two front-ends that share one processing engine**:

| File | Framework | Purpose |
|---|---|---|
| `gradio_app.py` | Gradio | The assignment's Gradio interface |
| `streamlit_app.py` | Streamlit | Deployed to **Streamlit Community Cloud** from GitHub |
| `core/` | OpenCV + NumPy | All image processing, shared by both apps |

> **Why two apps?** Streamlit Community Cloud can only run Streamlit apps, so a Gradio app can't be deployed there directly. Both apps import the same `core/` package, so every feature works the same way in both.

---

## ✨ Features

### Assignment requirements
- ✅ Upload an image (file, webcam or clipboard), or click a built-in sample
- ✅ Choose an operation from a dropdown
- ✅ Process the image (a **Live preview** option re-processes as soon as a setting changes)
- ✅ Original and processed shown **side by side**, plus a **before/after compare slider**
- ✅ Download the result as **PNG, JPEG or WEBP**, and save it to the `outputs/` folder
- ✅ A clean, guided interface (① Upload → ② Operation → ③ Fine-tune → ④ Save)

### 27 operations in 5 categories

| Category | Operations |
|---|---|
| **Basics** | 📘 Grayscale (3 methods) · 📘 Blur (Gaussian, Median, Bilateral, Box, Motion) · 📘 Edge Detection (Canny with auto thresholds, Sobel, Scharr, Laplacian; overlay mode) |
| **Transform** | 📘 Rotate (expand canvas, lossless 90°/180°, border modes) · ✨ Flip / Mirror · ✨ Resize · ✨ Center Crop (1:1, 4:3, 16:9, 9:16…) |
| **Enhance** | 📘 Image Enhancement (Auto Enhance, CLAHE, Histogram Equalization, White Balance, Detail Enhance) · ✨ Brightness & Contrast · ✨ Sharpen (Unsharp Mask, Kernel, High-pass) · ✨ Gamma · ✨ Color & Temperature · ✨ Denoise |
| **Detection & Analysis** | 📘 Contour Detection (count, area, numbering, boxes) · 📘 Shape Detection (triangle, square, rectangle, pentagon, hexagon, circle, ellipse, star) · ✨ Thresholding (Binary, Otsu, Triangle, Adaptive Mean/Gaussian…) · ✨ Morphology · ✨ Face Detection with privacy blur/pixelate · ✨ Color Splash |
| **Artistic** | ✨ Pencil Sketch · ✨ Cartoon · ✨ Watercolor · ✨ Sepia · ✨ Emboss · ✨ Vignette · ✨ Pixelate · ✨ Negative |

📘 = covered in class · ✨ = custom feature (20 in total)

### Custom features beyond the class material
1. **Multi-filter pipelines.** Chain any operations in sequence and see each step in a gallery. In Streamlit you can also reorder and remove steps.
2. **6 one-click presets:** Document Scanner, Vintage Photo, Vivid & Sharp, Comic Book, Object Outlines, Night Vision.
3. **Brightness & Contrast, Sharpening, Flip, Thresholding**, all the ideas suggested in the brief.
4. **Face detection with privacy mode** that blurs or pixelates every face before you share a photo.
5. **Colour-aware segmentation** for contours and shapes. It measures each pixel's colour distance from the background in CIE-Lab, so a yellow shape on a white background is still detected.
6. **Insights tab** with RGB and luminance histograms and image statistics (brightness, contrast, colourfulness, sharpness) for the original and the processed image.
7. **Smart input handling:** phone-photo EXIF rotation, transparent PNGs, 16-bit images, and automatic downscaling of very large photos for speed.

---

## 🗂️ Project structure

```
Day-28/
├── gradio_app.py            # Gradio interface
├── streamlit_app.py         # Streamlit interface (deployed)
├── core/
│   ├── __init__.py
│   ├── operations.py        # 27 operations + declarative registry
│   ├── pipeline.py          # multi-step pipelines + presets
│   └── utils.py             # I/O, export, histogram, stats, compare
├── assets/
│   ├── samples/             # demo images (shapes, document, cat, coffee, rocket)
│   └── models/              # Haar cascades for face/eye detection
├── .streamlit/config.toml   # theme + upload limit
├── requirements.txt         # Streamlit app (used by Streamlit Cloud)
├── requirements-gradio.txt  # adds Gradio
└── README.md
```

**How it is designed:** each operation is declared once in `core/operations.py` with its name, category, description and parameters. Both UIs **generate their controls automatically** from that registry. To add a new filter, you write one function with a `@register(...)` decorator, and it shows up in both apps, in the pipeline and in the presets.

---

## 🚀 Run locally (Windows)

Open a terminal in the project folder (`C:\Users\HP\Desktop\Day-28`):

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# 2. Install everything (Streamlit + Gradio)
pip install -r requirements-gradio.txt

# 3a. Run the Gradio app  → http://127.0.0.1:7860
python gradio_app.py

# 3b. Or run the Streamlit app  → http://localhost:8501
streamlit run streamlit_app.py
```

Processed images saved with the **Save** buttons go to the `outputs/` folder.

---

## ☁️ Deploy to Streamlit Community Cloud (via GitHub)

### Step 1: Push the project to GitHub
1. Create a new **public** repository on [github.com/new](https://github.com/new), for example `Awais-studio`. Don't add a README, because the project already has one.
2. In the project folder, run:

```bash
git init
git add .
git commit -m "Awais Studio: computer vision image processing app"
git branch -M main
git remote add origin https://github.com/<your-username>/Awais-studio.git
git push -u origin main
```

### Step 2: Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/Awais-studio`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** choose a name, e.g. `Awais-studio`
4. *(Optional)* Under **Advanced settings**, choose Python **3.12**.
5. Click **Deploy**. The first build takes 2–4 minutes while it installs `requirements.txt`.

Your app will be live at `https://<your-app-name>.streamlit.app`. Every `git push` to `main` redeploys it automatically.

### Troubleshooting
| Problem | Fix |
|---|---|
| `ImportError: libGL.so.1` | Make sure `requirements.txt` uses **`opencv-python-headless`**, not `opencv-python`. |
| `ModuleNotFoundError: core` | The `core/` folder (with `__init__.py`) must be pushed next to `streamlit_app.py`. |
| App is slow on huge photos | Lower **Working resolution** in the sidebar. The free tier has about 1 GB of RAM. |
| Faces aren't detected | Use a clear, front-facing photo, or lower *Min neighbours*. |

### Bonus: hosting the Gradio app
Gradio apps can be hosted for free on **Hugging Face Spaces**. Create a Space with the *Gradio* SDK, upload the project, rename `gradio_app.py` to `app.py`, and rename `requirements-gradio.txt` to `requirements.txt` with the `-r` line replaced by the packages it lists.

---

## 🛠️ Tech stack
Python · OpenCV · NumPy · Pillow · Gradio · Streamlit · pandas

Sample photos come from the scikit-image data set (CC0 / public domain). The shapes and document samples were generated for this project. The Haar cascades are from OpenCV (Intel License).
