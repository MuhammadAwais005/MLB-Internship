"""
Awais Studio: Gradio front-end
===================================

Run locally:
    pip install -r requirements-gradio.txt
    python gradio_app.py

Then open http://127.0.0.1:7860 in your browser.

Every control on the page is generated from the operation registry in
``core/operations.py``, so this file only deals with layout and wiring.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import gradio as gr
import numpy as np

from core import (
    EXPORT_FORMATS,
    OPERATIONS,
    PRESETS,
    ProcessingError,
    Step,
    apply_operation,
    ensure_rgb,
    format_steps,
    histogram_image,
    image_stats,
    limit_size,
    list_samples,
    operations_by_category,
    preset_steps,
    run_pipeline,
    save_image,
)

APP_TITLE = "Awais Studio"
MAX_SIDE = 2400  # longest side processed, keeps the UI snappy on huge photos
DEFAULT_OP = "edges"

# Dropdown choices grouped by category order: ("⚫  Grayscale", "grayscale")
OP_CHOICES = [(op.label, op.key) for ops in operations_by_category().values() for op in ops]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def op_card(key: str) -> str:
    op = OPERATIONS[key]
    badge = ("<span class='badge custom'>✨ Custom feature</span>" if op.custom
             else "<span class='badge core'>📘 Covered in class</span>")
    return (f"<div class='op-card'><div class='op-head'><span class='op-title'>{op.icon} {op.name}</span>"
            f"<span class='badge cat'>{op.category}</span>{badge}</div>"
            f"<div class='op-desc'>{op.description}</div></div>")


def status_html(text: str, kind: str = "ok", millis: float | None = None) -> str:
    timing = f"<span class='timing'>{millis:.0f} ms</span>" if millis is not None else ""
    icon = {"ok": "✅", "warn": "⚠️", "idle": "💡"}.get(kind, "ℹ️")
    return f"<div class='status {kind}'>{icon} <span>{text}</span>{timing}</div>"


def stats_table(original: np.ndarray, processed: np.ndarray) -> str:
    a, b = image_stats(original), image_stats(processed)
    rows = "".join(f"<tr><td>{k}</td><td>{a[k]}</td><td>{b[k]}</td></tr>" for k in a)
    return (f"<table class='stats'><thead><tr><th>Metric</th><th>Original</th><th>Processed</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def collect_params(key: str, values: tuple) -> Dict[str, Any]:
    start, end = SLICES[key]
    op = OPERATIONS[key]
    return {p.name: v for p, v in zip(op.params, values[start:end])}


def prepare_input(img) -> tuple[np.ndarray, str]:
    if img is None:
        raise gr.Error("Please upload an image (or pick a sample) first.")
    img = ensure_rgb(img)
    img, scaled = limit_size(img, MAX_SIDE)
    note = f" Working copy downscaled to {img.shape[1]}×{img.shape[0]} for speed." if scaled else ""
    return img, note


def _result_outputs(original, processed, info, millis, name, kind="ok"):
    return (
        original,                                   # side-by-side: original
        processed,                                  # side-by-side: processed
        (original, processed),                      # compare slider
        status_html(info, kind, millis),            # status
        processed,                                  # result_state
        histogram_image(original),                  # hist original
        histogram_image(processed),                 # hist processed
        stats_table(original, processed),           # stats
        name,                                       # name_state (for file name)
    )


N_RESULT_OUTPUTS = 9


# --------------------------------------------------------------------------- #
# Event handlers
# --------------------------------------------------------------------------- #
def process(img, key, *values):
    original, note = prepare_input(img)
    params = collect_params(key, values)
    t0 = time.perf_counter()
    try:
        result = apply_operation(original, key, params)
    except ProcessingError as exc:
        raise gr.Error(str(exc))
    millis = (time.perf_counter() - t0) * 1000
    return _result_outputs(original, result.image, result.info + note, millis, key)


def live_process(live, skip, img, key, *values):
    """Auto-apply when settings change (if Live preview is on)."""
    if skip:
        return (gr.skip(),) * N_RESULT_OUTPUTS + (False,)
    if not live or img is None:
        return (gr.skip(),) * N_RESULT_OUTPUTS + (False,)
    try:
        return process(img, key, *values) + (False,)
    except gr.Error:
        return (gr.skip(),) * N_RESULT_OUTPUTS + (False,)


def on_op_change(key):
    return [gr.update(visible=(k == key)) for k in OPERATIONS] + [op_card(key)]


def reset_params(key):
    updates = []
    for k, op in OPERATIONS.items():
        for p in op.params:
            updates.append(gr.update(value=p.default) if k == key else gr.skip())
    return updates


def add_step(steps: List[dict], key, *values):
    params = collect_params(key, values)
    steps = list(steps or []) + [Step(key, params).to_dict()]
    gr.Info(f"Added step {len(steps)}: {OPERATIONS[key].name}")
    return steps, format_steps([Step.from_dict(s) for s in steps])


def undo_step(steps):
    steps = list(steps or [])[:-1]
    return steps, format_steps([Step.from_dict(s) for s in steps])


def clear_steps():
    return [], format_steps([])


def load_preset(name):
    if not name:
        raise gr.Error("Choose a preset first.")
    steps = [s.to_dict() for s in preset_steps(name)]
    gr.Info(f"Loaded preset {name} ({len(steps)} steps). Press ▶ Run pipeline.")
    return steps, format_steps([Step.from_dict(s) for s in steps])


def run_pipe(img, steps):
    original, note = prepare_input(img)
    if not steps:
        raise gr.Error("The pipeline is empty. Add a step or load a preset.")
    t0 = time.perf_counter()
    try:
        final, reports = run_pipeline(original, [Step.from_dict(s) for s in steps])
    except ProcessingError as exc:
        raise gr.Error(str(exc))
    millis = (time.perf_counter() - t0) * 1000
    info = f"Pipeline of {len(reports)} step(s) finished. Last step: {reports[-1].info}{note}"
    gallery = [(original, "0 · Original")] + [(r.image, f"{r.index} · {r.description}") for r in reports]
    return _result_outputs(original, final, info, millis, "pipeline") + (gallery,)


def export(result, fmt, quality, name):
    if result is None:
        raise gr.Error("Nothing to save yet. Process an image first.")
    path = save_image(result, stem=f"Awais_{name or 'result'}", fmt_label=fmt, quality=quality)
    gr.Info(f"Saved to outputs/{path.name}")
    return gr.update(value=str(path), visible=True)


def use_as_input(result):
    if result is None:
        raise gr.Error("Nothing to reuse yet. Process an image first.")
    gr.Info("Result is now the input. Apply another operation to keep chaining.")
    return result, True  # True = skip the automatic re-processing once


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
.gradio-container {max-width: 1480px !important; margin: auto;}
#hero {background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 45%,#db2777 100%); border-radius: 18px;
       padding: 26px 30px; color: white; margin-bottom: 6px; box-shadow: 0 10px 30px rgba(79,70,229,.25);}
#hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.5px; color: white;}
#hero p {margin: 6px 0 0; opacity: .92; font-size: 1.02rem; color: white;}
#hero .chips {margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap;}
#hero .chip {background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3); padding: 4px 12px;
             border-radius: 999px; font-size: .85rem; backdrop-filter: blur(4px);}
.op-card {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px;
          background: var(--background-fill-secondary);}
.op-head {display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 6px;}
.op-title {font-weight: 700; font-size: 1.05rem;}
.op-desc {font-size: .92rem; opacity: .85; line-height: 1.45;}
.badge {font-size: .72rem; padding: 2px 9px; border-radius: 999px; font-weight: 600;}
.badge.cat {background: rgba(99,102,241,.15); color: #6366f1;}
.badge.core {background: rgba(14,165,233,.15); color: #0284c7;}
.badge.custom {background: rgba(219,39,119,.15); color: #db2777;}
.status {display: flex; align-items: center; gap: 10px; padding: 11px 14px; border-radius: 12px;
         font-size: .95rem; border: 1px solid var(--border-color-primary);}
.status.ok {background: rgba(16,185,129,.09); border-color: rgba(16,185,129,.35);}
.status.idle {background: rgba(99,102,241,.07); border-color: rgba(99,102,241,.3);}
.status span:first-of-type {flex: 1;}
.status .timing {font-family: ui-monospace, monospace; font-size: .8rem; opacity: .7; white-space: nowrap;}
table.stats {width: 100%; border-collapse: collapse; font-size: .9rem;}
table.stats th, table.stats td {padding: 7px 10px; border-bottom: 1px solid var(--border-color-primary); text-align: left;}
table.stats th {opacity: .7; font-weight: 600;}
#apply-btn {font-size: 1.05rem; font-weight: 700;}
footer {opacity: .6;}
"""

HERO = f"""
<div id="hero">
  <h1>🎨 {APP_TITLE}</h1>
  <p>A computer-vision lab in your browser: upload an image, pick an operation, compare before/after and download the result.</p>
  <div class="chips">
    <span class="chip">{len(OPERATIONS)} operations</span>
    <span class="chip">{sum(o.custom for o in OPERATIONS.values())} custom features</span>
    <span class="chip">Multi-filter pipelines</span>
    <span class="chip">{len(PRESETS)} one-click presets</span>
    <span class="chip">Live preview</span>
    <span class="chip">OpenCV powered</span>
  </div>
</div>
"""


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
SLICES: Dict[str, tuple] = {}

with gr.Blocks(title=APP_TITLE) as demo:
    result_state = gr.State(None)
    name_state = gr.State("result")
    steps_state = gr.State([])
    skip_state = gr.State(False)

    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        # ------------------------------ LEFT: controls ---------------------- #
        with gr.Column(scale=5, min_width=360):
            input_image = gr.Image(label="① Upload an image", type="numpy", height=300,
                                   sources=["upload", "webcam", "clipboard"])
            samples = [str(p) for p in list_samples()]
            if samples:
                gr.Examples(samples, inputs=input_image, label="…or try a sample", examples_per_page=6)

            op_dropdown = gr.Dropdown(OP_CHOICES, value=DEFAULT_OP, label="② Choose an operation",
                                      info="Grouped: Basics → Transform → Enhance → Detection → Artistic",
                                      interactive=True)
            op_info = gr.HTML(op_card(DEFAULT_OP))
            with gr.Row():
                apply_btn = gr.Button("▶ Apply", variant="primary", elem_id="apply-btn", scale=2)
                add_btn = gr.Button("➕ Add to pipeline", variant="secondary", scale=2)
            with gr.Row(equal_height=True):
                live = gr.Checkbox(value=True, label="⚡ Live preview", scale=3,
                                   info="Re-process automatically when a setting changes")
                reset_btn = gr.Button("↺ Reset settings", size="sm", scale=1)

            gr.Markdown("**③ Fine-tune**")
            groups, all_params = {}, []
            for key, op in OPERATIONS.items():
                with gr.Group(visible=(key == DEFAULT_OP)) as grp:
                    start = len(all_params)
                    if not op.params:
                        gr.Markdown("&nbsp;&nbsp;This operation has no settings. Just press **Apply**.")
                    for p in op.params:
                        if p.kind == "slider":
                            comp = gr.Slider(p.minimum, p.maximum, value=p.default, step=p.step,
                                             label=p.label, info=p.help or None)
                        elif p.kind == "checkbox":
                            comp = gr.Checkbox(value=p.default, label=p.label, info=p.help or None)
                        elif len(p.choices) <= 4:
                            comp = gr.Radio(list(p.choices), value=p.default, label=p.label, info=p.help or None)
                        else:
                            comp = gr.Dropdown(list(p.choices), value=p.default, label=p.label, info=p.help or None)
                        all_params.append(comp)
                    SLICES[key] = (start, len(all_params))
                groups[key] = grp


        # ------------------------------ RIGHT: results ---------------------- #
        with gr.Column(scale=7, min_width=420):
            status = gr.HTML(status_html("Upload an image or click a sample to get started.", "idle"))
            with gr.Tabs():
                with gr.Tab("🖼️ Side by side"):
                    with gr.Row():
                        orig_view = gr.Image(label="Original", interactive=False, height=430,
                                             buttons=["fullscreen"])
                        proc_view = gr.Image(label="Processed", interactive=False, height=430,
                                             buttons=["download", "fullscreen"])
                with gr.Tab("↔️ Compare slider"):
                    compare = gr.ImageSlider(label="Drag to compare (original | processed)", height=480,
                                             interactive=False)
                with gr.Tab("🧪 Pipeline"):
                    gr.Markdown("Chain several filters. Tune an operation on the left and press "
                                "**➕ Add to pipeline**, or start from a preset.")
                    with gr.Row():
                        preset_dd = gr.Dropdown(list(PRESETS), label="Presets", scale=3)
                        load_btn = gr.Button("📥 Load preset", scale=1)
                    steps_md = gr.Markdown(format_steps([]))
                    with gr.Row():
                        run_btn = gr.Button("▶ Run pipeline", variant="primary")
                        undo_btn = gr.Button("⤺ Undo last step")
                        clear_btn = gr.Button("🗑 Clear")
                    gallery = gr.Gallery(label="Step by step", columns=4, height=260, object_fit="contain")
                with gr.Tab("📊 Insights"):
                    with gr.Row():
                        hist_orig = gr.Image(label="Histogram: original", interactive=False, height=210)
                        hist_proc = gr.Image(label="Histogram: processed", interactive=False, height=210)
                    stats = gr.HTML()

            gr.Markdown("**④ Save & download**")
            with gr.Row(equal_height=True):
                fmt = gr.Dropdown(list(EXPORT_FORMATS), value="PNG (lossless)", label="Format", scale=1)
                quality = gr.Slider(50, 100, value=92, step=1, label="Quality (JPEG/WEBP)", scale=1)
            with gr.Row():
                save_btn = gr.Button("💾 Save & download", variant="primary")
                reuse_btn = gr.Button("↩️ Use result as new input")
            download = gr.File(label="Your file (also saved in the outputs/ folder)", visible=False)

    with gr.Accordion("ℹ️ About this app", open=False):
        gr.Markdown(
            f"""
**{APP_TITLE}** combines the image-processing techniques from the course into one tool.

* **Covered in class:** grayscale, blur, edge detection, rotation, enhancement, contour detection, shape detection.
* **Custom features:** brightness & contrast, sharpening, flip, thresholding (incl. Otsu & adaptive),
  multi-filter pipelines with presets, gamma, colour & temperature, denoising, morphology, face detection with
  privacy blur, colour splash, crop, resize and artistic filters (sketch, cartoon, watercolour, sepia, emboss,
  vignette, pixelate, negative).
* Built with **OpenCV**, **NumPy** and **Gradio**. All processing runs locally on your machine.
"""
        )

    # ------------------------------ Wiring ---------------------------------- #
    result_outputs = [orig_view, proc_view, compare, status, result_state, hist_orig, hist_proc, stats, name_state]
    proc_inputs = [input_image, op_dropdown, *all_params]
    live_inputs = [live, skip_state, *proc_inputs]

    op_dropdown.change(on_op_change, op_dropdown, [*groups.values(), op_info]).then(
        live_process, live_inputs, result_outputs + [skip_state])
    apply_btn.click(process, proc_inputs, result_outputs)
    input_image.change(live_process, live_inputs, result_outputs + [skip_state])

    for comp in all_params:
        event = comp.release if isinstance(comp, gr.Slider) else comp.input
        event(live_process, live_inputs, result_outputs + [skip_state], show_progress="hidden")

    reset_btn.click(reset_params, op_dropdown, all_params).then(
        live_process, live_inputs, result_outputs + [skip_state])

    add_btn.click(add_step, [steps_state, op_dropdown, *all_params], [steps_state, steps_md])
    undo_btn.click(undo_step, steps_state, [steps_state, steps_md])
    clear_btn.click(clear_steps, None, [steps_state, steps_md])
    load_btn.click(load_preset, preset_dd, [steps_state, steps_md])
    run_btn.click(run_pipe, [input_image, steps_state], result_outputs + [gallery])

    save_btn.click(export, [result_state, fmt, quality, name_state], download)
    reuse_btn.click(use_as_input, result_state, [input_image, skip_state])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="violet",
                             font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]),
        css=CSS,
    )
