"""
Document & Object Segmentation Tool - web app (Streamlit)
=========================================================

Run locally:
    py -3.11 -m streamlit run streamlit_app.py
"""

import sys
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

# Streamlit Cloud starts the app from the repository root - make the package importable.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from segmenter import METHODS, segment  # noqa: E402
from segmenter.config import DATASET_DIR, IMAGE_TYPES, OUTPUT_DIR  # noqa: E402
from segmenter.document import find_document  # noqa: E402
from segmenter.io_utils import encode_png, load_image  # noqa: E402
from segmenter.output import cutout, on_checkerboard, overlay, save_best  # noqa: E402

st.set_page_config(page_title="Document & Object Segmentation", page_icon="✂️", layout="wide")


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


@st.cache_data(show_spinner=False, max_entries=30)
def run_segmentation(image_bytes, keep_all):
    return segment(load_image(image_bytes), keep_all=keep_all)


def sample_images():
    if not DATASET_DIR.exists():
        return []
    return sorted(p for p in DATASET_DIR.iterdir() if p.suffix.lower() in IMAGE_TYPES)


def pretty(path):
    number, _, rest = path.stem.partition("_")
    return f"{number} · {rest.replace('_', ' ').capitalize()}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📥 Input image")
    source = st.radio("Source", ["Try a sample", "Upload an image", "Take a photo"], label_visibility="collapsed")

    image_bytes, image_name = None, None
    if source == "Upload an image":
        upload = st.file_uploader("Image", type=[e.lstrip(".") for e in IMAGE_TYPES])
        if upload:
            image_bytes, image_name = upload.getvalue(), upload.name
    elif source == "Take a photo":
        photo = st.camera_input("Photo of a document or object")
        if photo:
            image_bytes, image_name = photo.getvalue(), "camera_photo.jpg"
    else:
        samples = sample_images()
        if samples:
            pick = st.selectbox("Sample (16 images)", samples, format_func=pretty)
            image_bytes, image_name = pick.read_bytes(), pick.name

    st.header("⚙️ Options")
    keep_all = st.checkbox("Keep every object", value=False,
                           help="Off: keep the main object(s) only - the largest region plus anything at least "
                                "a quarter of its size. On: keep every region that survives the cleanup.")
    view = st.radio("Show each result as", ["Overlay", "Cut-out", "Mask"], horizontal=True)

    with st.expander("ℹ️ How the best result is chosen"):
        st.markdown(
            "Every method gets a **0–100 quality score** without needing the right answer:\n\n"
            "- **Edge**: does the outline sit on real edges in the photo?\n"
            "- **Contrast**: do inside and outside differ in colour?\n"
            "- **Solidity**: is the object a solid shape, not lace?\n"
            "- **Border**: does it stay off the image border?\n\n"
            "Masks covering < 10% or > 95% of the image are penalised. "
            "You can always override the choice below.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("✂️ Document & Object Segmentation Tool")
st.caption("Seven thresholding techniques separate the main object or document from the background. "
           "Every result is scored and the best one is picked automatically. Download it, or choose another.")

if image_bytes is None:
    st.info("👈 Pick a sample, upload an image or take a photo in the sidebar.")
    st.stop()

with st.spinner("Running 7 segmentation methods..."):
    try:
        seg = run_segmentation(image_bytes, keep_all)
    except Exception as error:
        st.error(f"Could not process this image: {error}")
        st.stop()

stem = Path(image_name).stem
best = seg.best
h, w = seg.image.shape[:2]

top_left, top_right = st.columns([1, 2], gap="large")
with top_left:
    st.subheader("Input")
    st.image(rgb(seg.image), caption=f"{image_name} · {w}×{h}", width="stretch")
with top_right:
    st.subheader("Automatic choice")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best method", best.label)
    m2.metric("Score", f"{best.score.total:.0f} / 100")
    m3.metric("Foreground", f"{best.score.coverage:.0f}%")
    m4.metric("Document page", "found ✅" if seg.document is not None else "no")
    st.image(rgb(overlay(seg.image, best.mask, seg.document)), width="stretch",
             caption="Background dimmed, object outlined in red, page corners in yellow")

# --- every method ----------------------------------------------------------
st.subheader("Output of each segmentation method")
st.caption("Sorted from best to worst score. The same cleanup is applied to every method, so only the "
           "thresholding technique differs.")

columns = st.columns(4)
for i, (key, r) in enumerate(seg.results.items()):
    if view == "Overlay":
        shown = overlay(seg.image, r.mask)
    elif view == "Cut-out":
        shown = on_checkerboard(cutout(seg.image, r.mask))
    else:
        shown = r.mask
    star = "⭐ " if key == seg.best_key else f"#{i + 1} "
    with columns[i % 4]:
        st.image(rgb(shown), width="stretch")
        st.markdown(f"**{star}{r.label}** · score **{r.score.total:.0f}**  \n"
                    f"<small>{r.description} · {r.score.coverage:.0f}% foreground · "
                    f"{r.objects} object(s)</small>", unsafe_allow_html=True)

# --- chosen result + saving --------------------------------------------------
st.divider()
st.subheader("💾 Save the result")
keys = list(seg.results)
choice = st.selectbox(
    "Result to save (the best is pre-selected; pick another if you prefer it)", keys,
    format_func=lambda k: f"{'⭐ ' if k == seg.best_key else ''}{seg.results[k].label}  ·  score {seg.results[k].score.total:.0f}")
chosen = seg.results[choice]
document = seg.document if choice == seg.best_key else find_document(seg.image, chosen.mask)

c1, c2, c3 = st.columns(3)
c1.image(rgb(on_checkerboard(cutout(seg.image, chosen.mask))), caption="Cut-out (transparent background)",
         width="stretch")
c2.image(chosen.mask, caption="Mask (white = foreground)", width="stretch")
c3.image(rgb(overlay(seg.image, chosen.mask, document)), caption="Overlay", width="stretch")

if document is not None:
    st.markdown("**📄 Document detected: straightened page**")
    d1, d2 = st.columns(2)
    d1.image(rgb(document.color), caption="Straightened colour scan", width="stretch")
    d2.image(document.clean, caption="Black & white scan", width="stretch")

b1, b2, b3, b4 = st.columns(4)
b1.download_button("⬇️ Cut-out (PNG)", encode_png(cutout(seg.image, chosen.mask)),
                   f"{stem}_best_cutout.png", "image/png", type="primary", width="stretch")
b2.download_button("⬇️ Mask (PNG)", encode_png(chosen.mask), f"{stem}_best_mask.png", "image/png", width="stretch")
if document is not None:
    b3.download_button("⬇️ Document scan", encode_png(document.color), f"{stem}_document_scan.png",
                       "image/png", width="stretch")
if b4.button("💾 Save to outputs/ folder", width="stretch",
             help="Writes the files next to the app (useful when running on your own PC)."):
    saved = save_best(seg, OUTPUT_DIR / stem, stem, key=choice)
    st.success(f"Saved {len(saved)} files to outputs/{stem}/ at {datetime.now():%H:%M:%S}")

# --- numbers -------------------------------------------------------------------
with st.expander("📊 Score breakdown for every method"):
    rows = [{"Method": r.label, "Score": round(r.score.total, 1), "Edge": round(r.score.edge, 2),
             "Contrast": round(r.score.contrast, 2), "Solidity": round(r.score.solidity, 2),
             "Border": round(r.score.border, 2), "Foreground %": round(r.score.coverage, 1),
             "Objects": r.objects, "Time (ms)": round(r.seconds * 1000, 1)} for r in seg.results.values()]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(f"{len(METHODS)} methods run on a copy at most 1000 px wide; masks are scaled back to "
               f"the full {w}×{h} resolution before saving.")
