"""
Awais Studio: Streamlit front-end (the one deployed to Streamlit Community Cloud)
=====================================================================================

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

It shares the exact same processing engine (``core/``) as ``gradio_app.py``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from core import (
    CATEGORIES,
    EXPORT_FORMATS,
    OPERATIONS,
    PRESETS,
    ProcessingError,
    Step,
    apply_operation,
    encode_image,
    histogram_image,
    image_stats,
    limit_size,
    list_samples,
    load_image,
    operations_by_category,
    preset_steps,
    run_pipeline,
    save_image,
    wipe_compare,
)

APP_TITLE = "Awais Studio"

st.set_page_config(page_title=APP_TITLE, page_icon="🎨", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1500px;}
.hero {background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 45%,#db2777 100%); border-radius: 18px;
       padding: 24px 28px; color: #fff; margin-bottom: 1.1rem; box-shadow: 0 10px 30px rgba(79,70,229,.25);}
.hero h1 {margin: 0; font-size: 2.05rem; font-weight: 800; color: #fff; padding: 0; letter-spacing: -.5px;}
.hero p {margin: 6px 0 0; opacity: .92; font-size: 1.02rem;}
.chips {margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;}
.chip {background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3); padding: 3px 12px;
       border-radius: 999px; font-size: .82rem;}
.op-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 10px 12px; margin: 4px 0 10px;
          background: rgba(99,102,241,.06); font-size: .88rem; line-height: 1.45;}
.badge {font-size: .7rem; padding: 2px 8px; border-radius: 999px; font-weight: 600; margin-right: 4px;}
.b-core {background: rgba(14,165,233,.16); color: #0284c7;}
.b-custom {background: rgba(219,39,119,.16); color: #db2777;}
.feature {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 16px 18px; height: 100%;
          background: rgba(128,128,128,.04);}
.feature h4 {margin: 0 0 6px; font-size: 1.02rem;}
.feature p {margin: 0; font-size: .9rem; opacity: .8;}
.step-row {font-size: .88rem;}
[data-testid="stMetricValue"] {font-size: 1.35rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="hero">
  <h1>🎨 {APP_TITLE}</h1>
  <p>A computer-vision lab in your browser: upload an image, pick an operation, compare before/after and download the result.</p>
  <div class="chips">
    <span class="chip">{len(OPERATIONS)} operations</span>
    <span class="chip">{sum(o.custom for o in OPERATIONS.values())} custom features</span>
    <span class="chip">Multi-filter pipelines</span>
    <span class="chip">{len(PRESETS)} presets</span>
    <span class="chip">OpenCV powered</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# State & cached helpers
# --------------------------------------------------------------------------- #
st.session_state.setdefault("steps", [])  # pipeline: list of {"op_key", "params"}


@st.cache_data(show_spinner=False, max_entries=16)
def decode_image(data: bytes, max_side: int | None) -> tuple[np.ndarray, bool, tuple[int, int]]:
    img = load_image(data)
    original_size = (img.shape[1], img.shape[0])
    img, scaled = limit_size(img, max_side)
    return img, scaled, original_size


@st.cache_data(show_spinner=False, max_entries=64)
def cached_apply(img: np.ndarray, key: str, params: tuple) -> tuple[np.ndarray, str, float]:
    t0 = time.perf_counter()
    result = apply_operation(img, key, dict(params))
    return result.image, result.info, (time.perf_counter() - t0) * 1000


@st.cache_data(show_spinner=False, max_entries=16)
def cached_pipeline(img: np.ndarray, steps: tuple) -> tuple[np.ndarray, list, float]:
    t0 = time.perf_counter()
    final, reports = run_pipeline(img, [Step(k, dict(p)) for k, p in steps])
    rows = [(r.index, r.description, r.info, r.millis, r.image) for r in reports]
    return final, rows, (time.perf_counter() - t0) * 1000


@st.cache_data(show_spinner=False, max_entries=16)
def cached_encode(img: np.ndarray, fmt: str, quality: int) -> tuple[bytes, str, str]:
    return encode_image(img, fmt, quality)


def freeze(params: Dict[str, Any]) -> tuple:
    return tuple(sorted(params.items()))


def widget_key(op_key: str, name: str) -> str:
    return f"p__{op_key}__{name}"


def render_params(op) -> Dict[str, Any]:
    """Build sidebar widgets from the registry; values persist per operation."""
    values = {}
    if not op.params:
        st.caption("This operation has no settings.")
    for p in op.params:
        k = widget_key(op.key, p.name)
        if p.kind == "slider":
            cast = int if p.is_int else float  # Streamlit needs consistent numeric types
            values[p.name] = st.slider(p.label, cast(p.minimum), cast(p.maximum), cast(p.default), cast(p.step),
                                       key=k, help=p.help or None)
        elif p.kind == "checkbox":
            values[p.name] = st.checkbox(p.label, p.default, key=k, help=p.help or None)
        elif len(p.choices) <= 3:
            values[p.name] = st.radio(p.label, p.choices, index=p.choices.index(p.default), key=k,
                                      horizontal=True, help=p.help or None)
        else:
            values[p.name] = st.selectbox(p.label, p.choices, index=p.choices.index(p.default), key=k,
                                          help=p.help or None)
    return values


def reset_params(op_key: str) -> None:
    for p in OPERATIONS[op_key].params:
        st.session_state.pop(widget_key(op_key, p.name), None)


def add_step(op_key: str, params: Dict[str, Any]) -> None:
    st.session_state.steps.append({"op_key": op_key, "params": dict(params)})
    st.toast(f"Added step {len(st.session_state.steps)}: {OPERATIONS[op_key].name}", icon="➕")


def move_step(i: int, delta: int) -> None:
    steps = st.session_state.steps
    j = i + delta
    if 0 <= j < len(steps):
        steps[i], steps[j] = steps[j], steps[i]


def remove_step(i: int) -> None:
    st.session_state.steps.pop(i)


def load_preset(name: str) -> None:
    st.session_state.steps = [s.to_dict() for s in preset_steps(name)]
    st.toast(f"Loaded preset {name}", icon="📥")


# --------------------------------------------------------------------------- #
# Sidebar: image source, mode, operation & parameters
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("① Image")
    source = st.radio("Source", ["Upload", "Sample", "Camera"], horizontal=True, label_visibility="collapsed")
    raw: bytes | None = None
    image_name = "image"
    if source == "Upload":
        up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])
        if up is not None:
            raw, image_name = up.getvalue(), up.name.rsplit(".", 1)[0]
    elif source == "Sample":
        samples = list_samples()
        if samples:
            choice = st.selectbox("Sample image", samples, format_func=lambda p: p.stem.title(), index=len(samples) - 1)
            raw, image_name = choice.read_bytes(), choice.stem
        else:
            st.warning("No samples found in assets/samples.")
    else:
        shot = st.camera_input("Take a photo")
        if shot is not None:
            raw, image_name = shot.getvalue(), "camera"

    max_side = st.select_slider(
        "Working resolution (longest side)",
        options=[800, 1200, 1600, 2048, 3000, 0],
        value=1600,
        format_func=lambda v: "Original" if v == 0 else f"{v}px",
        help="Large photos are downscaled for speed. Choose 'Original' for full resolution.",
    )

    st.divider()
    st.header("② Mode")
    mode = st.radio("Mode", ["🎛️ Single operation", "🧪 Pipeline"], horizontal=True, label_visibility="collapsed")

    st.divider()
    st.header("③ Operation")
    grouped = operations_by_category()
    category = st.selectbox("Category", [c for c in CATEGORIES if grouped.get(c)])
    op = st.selectbox("Operation", grouped[category], format_func=lambda o: o.label)
    badge = "<span class='badge b-custom'>✨ Custom feature</span>" if op.custom else "<span class='badge b-core'>📘 Covered in class</span>"
    st.markdown(f"<div class='op-card'>{badge}<br>{op.description}</div>", unsafe_allow_html=True)

    params = render_params(op)
    c1, c2 = st.columns(2)
    c1.button("↺ Reset", on_click=reset_params, args=(op.key,), width="stretch")
    c2.button("➕ Add to pipeline", on_click=add_step, args=(op.key, params), width="stretch",
              type="primary" if mode.endswith("Pipeline") else "secondary")

    if mode.endswith("Pipeline"):
        st.divider()
        st.header("④ Pipeline")
        preset = st.selectbox("Presets", list(PRESETS), index=None, placeholder="Choose a preset…")
        st.button("📥 Load preset", on_click=load_preset, args=(preset,), disabled=preset is None, width="stretch")
        steps = st.session_state.steps
        if not steps:
            st.info("Pipeline is empty. Tune an operation above and press **➕ Add to pipeline**, or load a preset.")
        for i, s in enumerate(steps):
            step = Step.from_dict(s)
            row = st.columns([6, 1, 1, 1], vertical_alignment="center")
            row[0].markdown(f"<div class='step-row'><b>{i + 1}.</b> {step.op.icon} {step.describe()}</div>",
                            unsafe_allow_html=True)
            row[1].button("↑", key=f"up{i}", on_click=move_step, args=(i, -1), disabled=i == 0)
            row[2].button("↓", key=f"dn{i}", on_click=move_step, args=(i, 1), disabled=i == len(steps) - 1)
            row[3].button("✕", key=f"rm{i}", on_click=remove_step, args=(i,))
        if steps:
            st.button("🗑 Clear pipeline", on_click=lambda: st.session_state.update(steps=[]), width="stretch")

    st.divider()
    st.caption("Built with OpenCV, NumPy & Streamlit · same engine as the Gradio app.")


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
if raw is None:
    st.subheader("Get started")
    st.write("Upload an image in the sidebar, choose a sample, or take a photo with your camera.")
    cols = st.columns(4)
    features = [
        ("🎛️ 27 operations", "Grayscale, blur, edges, rotation, enhancement, contours, shapes, faces and many more."),
        ("🧪 Pipelines", "Chain filters in sequence, reorder steps and start from one-click presets."),
        ("↔️ Compare", "Side-by-side view, a before/after wipe slider, histograms and image statistics."),
        ("💾 Export", "Download as PNG, JPEG or WEBP with adjustable quality."),
    ]
    for col, (title, text) in zip(cols, features):
        col.markdown(f"<div class='feature'><h4>{title}</h4><p>{text}</p></div>", unsafe_allow_html=True)
    st.stop()

try:
    original, scaled, original_size = decode_image(raw, max_side or None)
except Exception as exc:  # corrupt / unsupported file
    st.error(f"Could not read this image: {exc}")
    st.stop()

pipeline_mode = mode.endswith("Pipeline")
reports: List[tuple] = []
try:
    with st.spinner("Processing…"):
        if pipeline_mode:
            steps = st.session_state.steps
            if not steps:
                processed, info, millis = original, "Pipeline is empty: showing the original image.", 0.0
            else:
                frozen = tuple((s["op_key"], freeze(s["params"])) for s in steps)
                processed, reports, millis = cached_pipeline(original, frozen)
                info = f"Pipeline of {len(reports)} step(s) finished. Last step: {reports[-1][2]}"
            stem = "pipeline"
        else:
            processed, info, millis = cached_apply(original, op.key, freeze(params))
            stem = op.key
except ProcessingError as exc:
    st.error(str(exc))
    st.stop()

# --- headline metrics ----------------------------------------------------- #
m = st.columns(4)
m[0].metric("Input", f"{original.shape[1]}×{original.shape[0]}",
            help=f"Uploaded at {original_size[0]}×{original_size[1]}" if scaled else None,
            border=True)
m[1].metric("Output", f"{processed.shape[1]}×{processed.shape[0]}", border=True)
m[2].metric("Operation" if not pipeline_mode else "Pipeline steps",
            op.name if not pipeline_mode else str(len(st.session_state.steps)), border=True)
m[3].metric("Processing time", f"{millis:.0f} ms", border=True)

st.success(info, icon="✅")
if scaled:
    st.caption(f"ℹ️ Working on a copy downscaled from {original_size[0]}×{original_size[1]} "
               f"(change 'Working resolution' in the sidebar).")

# --- views --------------------------------------------------------------- #
tab_names = ["🖼️ Side by side", "↔️ Compare slider", "📊 Insights"]
if pipeline_mode:
    tab_names.insert(2, "🧪 Steps")
tabs = st.tabs(tab_names)

with tabs[0]:
    left, right = st.columns(2, gap="medium")
    left.image(original, caption="Original", width="stretch")
    right.image(processed, caption="Processed", width="stretch")

with tabs[1]:
    pos = st.slider("Drag to compare: original ⟵ | ⟶ processed", 0, 100, 50, key="wipe")
    _, mid, _ = st.columns([1, 6, 1])
    mid.image(wipe_compare(original, processed, pos / 100), width="stretch")

if pipeline_mode:
    with tabs[2]:
        if not reports:
            st.info("Add steps or load a preset in the sidebar to see each stage here.")
        else:
            thumbs = [(0, "Original", "", 0.0, original)] + reports
            per_row = 4
            for start in range(0, len(thumbs), per_row):
                cols = st.columns(per_row)
                for col, (idx, desc, step_info, ms, im) in zip(cols, thumbs[start:start + per_row]):
                    col.image(im, width="stretch")
                    col.markdown(f"**{idx}.** {desc}" + (f"  \n<small>{ms:.0f} ms · {step_info}</small>" if idx else ""),
                                 unsafe_allow_html=True)

with tabs[-1]:
    h1, h2 = st.columns(2)
    h1.image(histogram_image(original), caption="Histogram: original (R, G, B + luminance)", width="stretch")
    h2.image(histogram_image(processed), caption="Histogram: processed", width="stretch")
    a, b = image_stats(original), image_stats(processed)
    st.dataframe(pd.DataFrame({"Original": a, "Processed": b}), width="stretch")

# --- export -------------------------------------------------------------- #
st.divider()
st.subheader("💾 Save & download")
e1, e2, e3, e4 = st.columns([2, 2, 2, 2], vertical_alignment="bottom")
fmt = e1.selectbox("Format", list(EXPORT_FORMATS))
quality = e2.slider("Quality (JPEG/WEBP)", 50, 100, 92, disabled=fmt.startswith("PNG"))
data, ext, mime = cached_encode(processed, fmt, quality)
file_name = f"Awais_{image_name}_{stem}.{ext}"
e3.download_button(f"⬇️ Download ({len(data) / 1024:,.0f} KB)", data, file_name=file_name, mime=mime,
                   type="primary", width="stretch")
if e4.button("💾 Save to outputs/ folder", width="stretch"):
    path = save_image(processed, stem=f"Awais_{image_name}_{stem}", fmt_label=fmt, quality=quality)
    st.toast(f"Saved to outputs/{path.name}", icon="💾")
