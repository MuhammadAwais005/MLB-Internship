"""
PPE Detection - Streamlit app
=============================

Upload a photo or a video of a work site. The custom-trained YOLO11n model finds
helmets, vests, boots and workers, shows every detection with its confidence,
checks who is missing equipment, and lets you download the processed result.

Run locally:
    streamlit run app.py

Deploy (Streamlit Community Cloud): main file = Day-31/Mini_Project/app.py,
Python 3.11. Dependencies come from requirements.txt in this folder.
"""

import hashlib
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from ppe_detector.compliance import check_workers, count_by_class
from ppe_detector.detector import PPEDetector
from ppe_detector.settings import CLASS_NAMES
from ppe_detector.video import process_video, video_info
from ppe_detector.visualize import draw_detections

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "ppe_yolo11n.onnx"
MODEL_INFO = APP_DIR / "models" / "model_info.json"
SAMPLE_IMAGES = APP_DIR / "samples" / "images"
SAMPLE_VIDEO = APP_DIR / "samples" / "ppe_demo.mp4"

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]

st.set_page_config(page_title="PPE Detection | YOLO11", page_icon="🦺", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading the model ...")
def load_detector():
    return PPEDetector(MODEL_PATH)


@st.cache_data
def load_model_info():
    return json.loads(MODEL_INFO.read_text(encoding="utf-8")) if MODEL_INFO.exists() else {}


def decode_image(data):
    """bytes -> BGR image. cv2 also applies the EXIF rotation of phone photos."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("This file could not be read as an image.")
    return image


def to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def encode(image_bgr, ext=".png"):
    ok, buffer = cv2.imencode(ext, image_bgr)
    return buffer.tobytes() if ok else b""


def fingerprint(data, *settings):
    """Key for remembering results, so a download click does not re-run the model."""
    return hashlib.md5(data + repr(settings).encode()).hexdigest()


# ---------------------------------------------------------------- sidebar
def sidebar(info):
    st.sidebar.title("🦺 PPE Detection")
    st.sidebar.caption("Custom YOLO11n trained on the Roboflow PPE dataset")

    st.sidebar.subheader("Detection settings")
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05,
                             help="Boxes below this confidence are hidden. Lower = more boxes, more false alarms.")
    iou = st.sidebar.slider("Overlap (IoU) threshold", 0.10, 0.90, 0.45, 0.05,
                            help="Two boxes of the same class overlapping more than this are merged into one.")
    names = list(CLASS_NAMES.values())
    chosen = st.sidebar.multiselect("Classes to show", names, default=names)
    classes = [i for i, n in CLASS_NAMES.items() if n in chosen]
    show_conf = st.sidebar.checkbox("Show confidence on boxes", value=True)

    if info:
        st.sidebar.subheader("Model")
        test = info.get("test") or info.get("validation_best", {})
        c1, c2 = st.sidebar.columns(2)
        c1.metric("mAP@50 (test)", f"{test.get('mAP50', 0):.1%}")
        c2.metric("mAP@50-95", f"{test.get('mAP50-95', 0):.1%}")
        t = info.get("training", {})
        st.sidebar.caption(f"YOLO11n · {t.get('imgsz', '?')} px · {t.get('epochs', '?')} epochs · "
                           f"ONNX Runtime on CPU")
    return conf, iou, classes, show_conf


# ---------------------------------------------------------------- image tab
def show_image_results(image, detections, elapsed_ms, show_conf, base_name):
    annotated = draw_detections(image, detections, show_confidence=show_conf)
    workers = check_workers(detections)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objects detected", len(detections))
    c2.metric("Workers", len(workers))
    c3.metric("Fully equipped", sum(w["status"] == "OK" for w in workers))
    c4.metric("Inference time", f"{elapsed_ms:.0f} ms")

    left, right = st.columns(2)
    left.image(to_rgb(image), caption="Original", width="stretch")
    right.image(to_rgb(annotated), caption="Detections", width="stretch")

    if not detections:
        st.info("No objects found at this confidence. Try lowering the confidence threshold.")
        return

    table = pd.DataFrame([d.to_dict() for d in detections])
    table = table.rename(columns={"name": "object"})[["object", "confidence", "x1", "y1", "x2", "y2"]]
    counts = count_by_class(detections)

    t1, t2 = st.columns([3, 2])
    with t1:
        st.markdown("**Detected objects**")
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"confidence": st.column_config.ProgressColumn(
                         "confidence", min_value=0.0, max_value=1.0, format="%.2f")})
    with t2:
        st.markdown("**Count per class**")
        st.dataframe(pd.DataFrame({"object": list(counts), "count": list(counts.values())}),
                     hide_index=True, width="stretch")
        if workers:
            st.markdown("**PPE check per worker**")
            st.dataframe(pd.DataFrame(workers).drop(columns="confidence"), hide_index=True,
                         width="stretch")

    st.markdown("**Download results**")
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Processed image (PNG)", encode(annotated, ".png"),
                       f"{base_name}_detected.png", "image/png", width="stretch", on_click="ignore")
    d2.download_button("⬇ Detections (CSV)", table.to_csv(index=False).encode(),
                       f"{base_name}_detections.csv", "text/csv", width="stretch", on_click="ignore")
    report = {"image": base_name, "detections": [d.to_dict() for d in detections], "workers": workers}
    d3.download_button("⬇ Full report (JSON)", json.dumps(report, indent=2).encode(),
                       f"{base_name}_report.json", "application/json", width="stretch", on_click="ignore")


def image_tab(detector, conf, iou, classes, show_conf):
    source = st.radio("Image source", ["Upload an image", "Use a sample image"], horizontal=True,
                      label_visibility="collapsed")
    if source == "Upload an image":
        upload = st.file_uploader("Upload a photo of a work site", type=IMAGE_TYPES)
        if upload is None:
            st.caption("Supported: " + ", ".join(IMAGE_TYPES))
            return
        data, base_name = upload.getvalue(), Path(upload.name).stem
    else:
        samples = sorted(SAMPLE_IMAGES.glob("*.jpg"))
        if not samples:
            st.warning("No sample images found in samples/images.")
            return
        choice = st.selectbox("Sample image", samples, format_func=lambda p: p.stem)
        data, base_name = choice.read_bytes(), choice.stem

    try:
        image = decode_image(data)
    except ValueError as error:
        st.error(str(error))
        return

    start = time.perf_counter()
    detections = detector.detect(image, conf=conf, iou=iou, classes=classes)
    elapsed_ms = (time.perf_counter() - start) * 1000
    show_image_results(image, detections, elapsed_ms, show_conf, base_name)


# ---------------------------------------------------------------- video tab
def video_tab(detector, conf, iou, classes):
    source = st.radio("Video source", ["Upload a video", "Use the sample video"], horizontal=True,
                      label_visibility="collapsed")
    if source == "Upload a video":
        upload = st.file_uploader("Upload a video", type=VIDEO_TYPES)
        if upload is None:
            st.caption("Supported: " + ", ".join(VIDEO_TYPES) + ". Short clips work best on the free cloud CPU.")
            return
        data, base_name, suffix = upload.getvalue(), Path(upload.name).stem, Path(upload.name).suffix
    else:
        if not SAMPLE_VIDEO.exists():
            st.warning("Sample video not found.")
            return
        data, base_name, suffix = SAMPLE_VIDEO.read_bytes(), SAMPLE_VIDEO.stem, ".mp4"

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / f"input{suffix}"
        input_path.write_bytes(data)
        try:
            info = video_info(input_path)
        except ValueError as error:
            st.error(str(error))
            return

        st.caption(f"{info['width']}×{info['height']} · {info['fps']:.0f} fps · {info['seconds']:.1f} s")
        c1, c2 = st.columns(2)
        longest = min(120, max(1, int(np.ceil(info["seconds"])))) if info["seconds"] else 30
        if longest > 1:
            max_seconds = c1.slider("Seconds to process", 1, longest, min(longest, 20),
                                    help="Processing runs on a CPU, so long videos are cut to this length.")
        else:
            max_seconds = longest
        detect_every = c2.select_slider("Run the model on every N-th frame", [1, 2, 3, 4, 5], value=2,
                                        help="2 = twice as fast. Boxes are kept on the frames in between.")

        key = fingerprint(data, conf, iou, tuple(classes), max_seconds, detect_every)
        if st.button("▶ Run detection on the video", type="primary"):
            bar = st.progress(0.0, text="Processing frames ...")
            output_path = Path(tmp) / "output.mp4"
            start = time.perf_counter()
            rows, summary = process_video(
                input_path, output_path, detector, conf=conf, iou=iou, classes=classes,
                detect_every=detect_every, max_seconds=max_seconds,
                progress=lambda f: bar.progress(f, text=f"Processing frames ... {f:.0%}"))
            summary["processing_s"] = round(time.perf_counter() - start, 1)
            bar.empty()
            st.session_state["video_result"] = {"key": key, "video": output_path.read_bytes(),
                                                "rows": rows, "summary": summary, "name": base_name}

    result = st.session_state.get("video_result")
    if not result or result["key"] != key:
        return

    s = result["summary"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames", s["frames_written"])
    m2.metric("Frames analysed", s["frames_detected"])
    m3.metric("Most workers in one frame", s["peak_counts"].get("person", 0))
    m4.metric("Processing time", f"{s['processing_s']} s")

    frames = pd.DataFrame(result["rows"])
    player, chart = st.columns([3, 2])
    player.video(result["video"])
    with chart:
        if not frames.empty:
            st.markdown("**Objects per analysed frame**")
            st.line_chart(frames.set_index("time_s")[list(CLASS_NAMES.values())], x_label="seconds", height=260)
        st.markdown("**Most objects seen in one frame**")
        st.dataframe(pd.DataFrame({"object": list(s["peak_counts"]), "max in a frame": list(s["peak_counts"].values())}),
                     hide_index=True, width="stretch")

    d1, d2 = st.columns(2)
    d1.download_button("⬇ Processed video (MP4)", result["video"], f"{result['name']}_detected.mp4",
                       "video/mp4", width="stretch", on_click="ignore")
    d2.download_button("⬇ Counts per frame (CSV)", frames.to_csv(index=False).encode(),
                       f"{result['name']}_frames.csv", "text/csv", width="stretch", on_click="ignore")


# ---------------------------------------------------------------- about tab
def about_tab(info):
    st.markdown("""
**What it does.** A YOLO11n detector, fine-tuned on a construction-site PPE dataset,
finds four classes: **helmet**, **vest**, **boots** and **person**. Each item is matched
to the worker whose box contains it, so the app can flag workers who are missing equipment.

**How it runs.** The trained PyTorch weights were exported to ONNX and run with ONNX
Runtime on the CPU. That gives the same boxes as Ultralytics without installing PyTorch,
so the app starts quickly on Streamlit Community Cloud.

**Limits.** The model learned from 551 training photos, mostly from a few sites. Expect
weaker results on very different scenes, at night, or when workers are far away and tiny.
The PPE check is a simple rule (item inside the worker's box), not a safety certificate.
""")
    if info:
        st.markdown("**Model card**")
        st.json(info, expanded=False)
    st.caption("Dataset: Roboflow Universe · anjrh81/ppe-asmsk v2 · CC BY 4.0")


# ---------------------------------------------------------------- main
def main():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.relative_to(APP_DIR)}. "
                 "Run training/train.py and training/export_onnx.py first.")
        st.stop()

    detector = load_detector()
    info = load_model_info()
    conf, iou, classes, show_conf = sidebar(info)

    st.title("Construction Site PPE Detection")
    st.write("Upload an image or a video. The model marks every **helmet**, **vest**, **boots** "
             "and **person** with its confidence, and you can download the processed result.")

    image, video, about = st.tabs(["🖼 Image", "🎬 Video", "ℹ About"])
    with image:
        image_tab(detector, conf, iou, classes, show_conf)
    with video:
        video_tab(detector, conf, iou, classes)
    with about:
        about_tab(info)


main()
