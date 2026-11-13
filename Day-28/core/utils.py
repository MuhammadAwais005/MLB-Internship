"""
Image I/O and helper utilities shared by both front-ends.

Conventions used across the whole project
-----------------------------------------
* Every image travelling between functions is a ``numpy.ndarray`` of dtype
  ``uint8`` with shape ``(H, W, 3)`` in **RGB** channel order.
* OpenCV natively works in BGR, so conversion only happens at the very edges
  (reading/writing files). Inside the app everything stays RGB, which is what
  Gradio, Streamlit and PIL all expect.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"
MODELS_DIR = ASSETS_DIR / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EXPORT_FORMATS = {
    # label: (PIL format, file extension, mime type)
    "PNG (lossless)": ("PNG", "png", "image/png"),
    "JPEG": ("JPEG", "jpg", "image/jpeg"),
    "WEBP": ("WEBP", "webp", "image/webp"),
}


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def ensure_rgb(img) -> np.ndarray:
    """Coerce almost anything image-like into a contiguous uint8 RGB array.

    Handles PIL images, grayscale arrays, RGBA (alpha is composited on white),
    16-bit images and float images in the 0-1 range.
    """
    if img is None:
        raise ValueError("No image provided.")

    if isinstance(img, Image.Image):
        img = _pil_to_rgb(img)
        return np.ascontiguousarray(np.asarray(img))

    arr = np.asarray(img)

    # --- dtype -------------------------------------------------------------
    if arr.dtype == np.uint16:
        arr = (arr / 257.0).round().astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if arr.max(initial=0) <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # --- channels ----------------------------------------------------------
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        arr = (rgb * alpha + 255.0 * (1.0 - alpha)).round().astype(np.uint8)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    return np.ascontiguousarray(arr)


def _pil_to_rgb(pil: Image.Image) -> Image.Image:
    """Respect EXIF orientation (phone photos!) and flatten transparency."""
    pil = ImageOps.exif_transpose(pil)
    if pil.mode in ("RGBA", "LA") or (pil.mode == "P" and "transparency" in pil.info):
        pil = pil.convert("RGBA")
        background = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(background, pil)
    return pil.convert("RGB")


def load_image(source) -> np.ndarray:
    """Load an image from a path, raw bytes or a file-like object."""
    if isinstance(source, (str, os.PathLike)):
        pil = Image.open(source)
    elif isinstance(source, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(source))
    else:  # file-like (e.g. Streamlit UploadedFile)
        pil = Image.open(source)
    return ensure_rgb(pil)


def limit_size(img: np.ndarray, max_side: int | None) -> Tuple[np.ndarray, bool]:
    """Downscale so the longest side is at most ``max_side`` pixels.

    Returns the (possibly) resized image and a flag telling if it was resized.
    """
    if not max_side:
        return img, False
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, False
    scale = max_side / float(longest)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), True


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def encode_image(img: np.ndarray, fmt_label: str = "PNG (lossless)", quality: int = 92) -> Tuple[bytes, str, str]:
    """Encode an RGB array to bytes. Returns ``(data, extension, mime)``."""
    pil_fmt, ext, mime = EXPORT_FORMATS.get(fmt_label, EXPORT_FORMATS["PNG (lossless)"])
    buffer = io.BytesIO()
    kwargs = {}
    if pil_fmt in ("JPEG", "WEBP"):
        kwargs["quality"] = int(quality)
    if pil_fmt == "JPEG":
        kwargs["optimize"] = True
    if pil_fmt == "PNG":
        kwargs["optimize"] = True
    Image.fromarray(ensure_rgb(img)).save(buffer, format=pil_fmt, **kwargs)
    return buffer.getvalue(), ext, mime


def build_filename(stem: str, ext: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem.lower()).strip("_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe or 'processed'}_{stamp}.{ext}"


def save_image(img: np.ndarray, stem: str = "processed", fmt_label: str = "PNG (lossless)",
               quality: int = 92, out_dir: Path | str = OUTPUT_DIR) -> Path:
    """Save to ``outputs/`` (created on demand) and return the file path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data, ext, _ = encode_image(img, fmt_label, quality)
    path = out_dir / build_filename(stem, ext)
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
def image_stats(img: np.ndarray) -> dict:
    """Quick, human-friendly statistics about an image."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    r, g, b = [c.astype(np.float32) for c in cv2.split(img)]
    rg, yb = np.abs(r - g), np.abs(0.5 * (r + g) - b)
    # Hasler & Süsstrunk colourfulness metric
    colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {
        "Resolution": f"{w} × {h}",
        "Megapixels": f"{w * h / 1e6:.2f} MP",
        "Brightness": f"{gray.mean():.0f} / 255",
        "Contrast (σ)": f"{gray.std():.1f}",
        "Colourfulness": f"{colorfulness:.1f}",
        "Sharpness": f"{sharpness:.0f}",
    }


def histogram_image(img: np.ndarray, width: int = 520, height: int = 200) -> np.ndarray:
    """Render an RGB + luminance histogram as an image (no matplotlib needed)."""
    pad = 10
    canvas = np.full((height, width, 3), (24, 26, 33), np.uint8)
    plot_w, plot_h = width - 2 * pad, height - 2 * pad

    # faint grid
    for i in range(1, 4):
        x = pad + i * plot_w // 4
        cv2.line(canvas, (x, pad), (x, height - pad), (45, 48, 58), 1)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    channels = [
        (img[:, :, 0], (239, 83, 80)),
        (img[:, :, 1], (102, 187, 106)),
        (img[:, :, 2], (66, 165, 245)),
        (gray, (230, 230, 230)),
    ]
    # sqrt scaling keeps small populations visible next to big spikes
    hists = [np.sqrt(cv2.calcHist([c], [0], None, [256], [0, 256]).ravel()) for c, _ in channels]
    peak = max(h[1:-1].max(initial=1) for h in hists) or 1  # ignore clipped extremes

    xs = pad + np.arange(256) * plot_w / 255.0
    overlay = canvas.copy()
    for (_, color), hist in zip(channels, hists):
        ys = height - pad - np.clip(hist / peak, 0, 1) * plot_h
        pts = np.column_stack([xs, ys]).astype(np.int32)
        if color != (230, 230, 230):
            poly = np.vstack([[[pad, height - pad]], pts, [[width - pad, height - pad]]])
            cv2.fillPoly(overlay, [poly], color, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
    for (_, color), hist in zip(channels, hists):
        ys = height - pad - np.clip(hist / peak, 0, 1) * plot_h
        pts = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)
    return canvas


def wipe_compare(original: np.ndarray, processed: np.ndarray, position: float = 0.5) -> np.ndarray:
    """Before/after "wipe": left of the divider is original, right is processed."""
    h, w = original.shape[:2]
    if processed.shape[:2] != (h, w):
        processed = cv2.resize(processed, (w, h), interpolation=cv2.INTER_AREA)
    x = int(np.clip(position, 0.0, 1.0) * w)
    out = processed.copy()
    out[:, :x] = original[:, :x]
    thickness = max(2, w // 300)
    cv2.line(out, (x, 0), (x, h), (255, 255, 255), thickness, cv2.LINE_AA)
    r = max(10, w // 60)
    cv2.circle(out, (x, h // 2), r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(out, (x, h // 2), r, (40, 40, 40), max(1, thickness // 2), cv2.LINE_AA)
    return out


def list_samples() -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in exts)
