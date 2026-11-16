"""
Simple OCR Document Reader - web version (Streamlit)
====================================================

Same OCR engine as the desktop app (ocr_reader/), with a browser interface,
so it can be deployed to Streamlit Community Cloud.

Run locally:
    py -3.11 -m streamlit run streamlit_app.py
"""

import io
import sys
from pathlib import Path

import streamlit as st

# Streamlit Cloud starts the app from the repository root, so make sure
# the ocr_reader package next to this file can always be imported.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ocr_reader.config import DATASET_DIR, SUPPORTED_EXTENSIONS  # noqa: E402
from ocr_reader.engine import OCREngine  # noqa: E402
from ocr_reader.output import build_preview, draw_boxes, to_pil  # noqa: E402

# Free cloud servers have a small CPU, so very large photos are shrunk first.
# EasyOCR's own detector works at 2560 px anyway, so no detail is lost.
MAX_IMAGE_SIDE = 2560
UPLOAD_TYPES = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]

st.set_page_config(page_title="OCR Document Reader", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# OCR (cached so the model loads once and each image is read only once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_engine():
    engine = OCREngine(languages=["en"])
    engine.load()
    return engine


@st.cache_data(show_spinner=False, max_entries=40)
def read_image(image_bytes, name, enhance):
    return get_engine().read(image_bytes, enhance=enhance, name=name, max_side=MAX_IMAGE_SIDE)


def preview_png(result):
    buffer = io.BytesIO()
    build_preview(result).save(buffer, format="PNG")
    return buffer.getvalue()


def sample_images():
    if not DATASET_DIR.exists():
        return []
    return sorted(p for p in DATASET_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def pretty_name(path):
    """'09_receipt_phone_photo.jpg' -> '09 · Receipt phone photo'"""
    number, _, rest = path.stem.partition("_")
    return f"{number} · {rest.replace('_', ' ').capitalize()}"


# ---------------------------------------------------------------------------
# Sidebar: input + settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📥 Input")
    source = st.radio("Image source", ["Upload an image", "Take a photo", "Try a sample"],
                      label_visibility="collapsed")

    image_bytes, image_name = None, None
    if source == "Upload an image":
        upload = st.file_uploader("Choose an image", type=UPLOAD_TYPES)
        if upload is not None:
            image_bytes, image_name = upload.getvalue(), upload.name
    elif source == "Take a photo":
        photo = st.camera_input("Take a photo of a document")
        if photo is not None:
            image_bytes, image_name = photo.getvalue(), "camera_photo.jpg"
    else:
        samples = sample_images()
        if samples:
            choice = st.selectbox("Sample image (20 real documents)", samples, format_func=pretty_name, index=8)
            image_bytes, image_name = choice.read_bytes(), choice.name
        else:
            st.warning("The dataset folder was not found.")

    st.header("⚙️ Settings")
    enhance = st.checkbox(
        "Enhance image before OCR", value=False,
        help="Grayscale, upscaling of small images, shadow removal, denoising and contrast boost. "
             "Helps with shadows, faded print and tiny text; not needed for clean images.")
    show_boxes = st.checkbox("Show detected text boxes", value=True)

    with st.expander("ℹ️ About"):
        st.markdown(
            "Text is extracted with **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** "
            "(CRAFT text detector + CRNN recognizer), English model.\n\n"
            "Reading order is rebuilt into lines, paragraphs and columns, so "
            "receipts read row by row.\n\n"
            "Internship project by **Muhammad Awais**. "
            "[Source code on GitHub](https://github.com/MuhammadAwais005/MLB-Internship/tree/main/Day-29)")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("📄 OCR Document Reader")
st.caption("Upload an image of a document, receipt, sign or note → EasyOCR extracts the text → "
           "check it, edit it and download it as a .txt file.")

if image_bytes is None:
    st.info("👈 Upload an image, take a photo, or pick one of the 20 sample documents in the sidebar.")
    st.stop()

with st.spinner("Reading text... (the very first run also downloads the OCR model, about 1 minute)"):
    try:
        result = read_image(image_bytes, image_name, enhance)
    except Exception as error:
        st.error(f"Could not read this image: {error}")
        st.stop()

# summary numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lines", result.line_count)
c2.metric("Words", result.word_count)
c3.metric("Avg. confidence", f"{result.average_confidence * 100:.0f}%")
c4.metric("OCR time", f"{result.seconds:.1f} s")

if result.detections and result.average_confidence < 0.65:
    st.warning("Low average confidence: check the text carefully. "
               "For shadowed, faded or very small images, try **Enhance image** in the sidebar.")

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.subheader("Original image")
    original = to_pil(result.image)
    shown = draw_boxes(original, result.detections) if show_boxes else original
    st.image(shown, caption=image_name, width="stretch")

with right:
    st.subheader("Extracted text")
    if not result.detections:
        st.info("No text was found in this image.")
    # the key changes with the image/settings, so the box resets for a new image
    text_key = f"text_{hash(image_bytes)}_{enhance}"
    edited_text = st.text_area("You can correct mistakes here before downloading",
                               value=result.text, height=480, key=text_key)

    txt_name = f"{Path(image_name).stem}.txt"
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download .txt", data=edited_text + "\n", file_name=txt_name,
                       mime="text/plain", width="stretch", type="primary")
    d2.download_button("🖼️ Download preview", data=preview_png(result),
                       file_name=f"{Path(image_name).stem}_preview.png", mime="image/png",
                       width="stretch")

with st.expander(f"🔍 Detected text boxes ({len(result.detections)})"):
    rows = [{"Text": d.text, "Confidence": f"{d.confidence * 100:.1f}%"} for d in result.detections]
    st.dataframe(rows, width="stretch", hide_index=True)
