"""
Image-processing operations + a declarative registry.

Every operation is described once (name, category, parameters, function) and
both the Gradio and the Streamlit front-ends build their controls *from this
registry*. Adding a new filter means writing one function and one
``register(...)`` call: the UIs, the pipeline and the presets pick it up.

All functions take and return ``uint8`` RGB arrays of shape ``(H, W, 3)``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import MODELS_DIR, ensure_rgb


# =========================================================================== #
# Registry model
# =========================================================================== #
@dataclass(frozen=True)
class Param:
    """Declarative description of one operation parameter."""

    name: str
    label: str
    kind: str  # "slider" | "choice" | "checkbox"
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: Tuple[str, ...] = ()
    help: str = ""

    @property
    def is_int(self) -> bool:
        return self.kind == "slider" and all(
            float(v).is_integer() for v in (self.minimum, self.maximum, self.step or 1, self.default)
        )

    def coerce(self, value: Any) -> Any:
        """Clamp / cast incoming UI values so operations never see bad input."""
        if value is None:
            return self.default
        if self.kind == "slider":
            v = float(value)
            v = min(max(v, float(self.minimum)), float(self.maximum))
            return int(round(v)) if self.is_int else v
        if self.kind == "checkbox":
            return bool(value)
        if self.kind == "choice":
            return value if value in self.choices else self.default
        return value


@dataclass
class OpResult:
    image: np.ndarray
    info: str = ""


@dataclass
class Operation:
    key: str
    name: str
    category: str
    icon: str
    description: str
    func: Callable[..., OpResult]
    params: List[Param] = field(default_factory=list)
    custom: bool = False  # True = extra feature beyond what was covered in class

    @property
    def label(self) -> str:
        return f"{self.icon}  {self.name}"

    def defaults(self) -> Dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def clean_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        params = params or {}
        return {p.name: p.coerce(params.get(p.name, p.default)) for p in self.params}

    def summarize(self, params: Optional[Dict[str, Any]]) -> str:
        """Short human description, e.g. ``Blur (Gaussian, kernel 15)``."""
        clean = self.clean_params(params)
        parts = []
        for p in self.params:
            v = clean[p.name]
            if p.kind == "checkbox":
                if v:
                    parts.append(p.label.split("(")[0].strip().lower())
            elif p.kind == "choice":
                parts.append(str(v))
            else:
                parts.append(f"{p.label.split('(')[0].strip().lower()} {v:g}" if isinstance(v, float) else f"{p.label.split('(')[0].strip().lower()} {v}")
        return f"{self.name} ({', '.join(parts)})" if parts else self.name


CATEGORIES = ["Basics", "Transform", "Enhance", "Detection & Analysis", "Artistic"]
OPERATIONS: Dict[str, Operation] = {}


def register(key: str, name: str, category: str, icon: str, description: str,
             params: List[Param] | None = None, custom: bool = False):
    def decorator(func: Callable[..., OpResult]):
        OPERATIONS[key] = Operation(key, name, category, icon, description, func, params or [], custom)
        return func
    return decorator


class ProcessingError(RuntimeError):
    """Raised with a user-friendly message when an operation fails."""


def apply_operation(img: np.ndarray, key: str, params: Optional[Dict[str, Any]] = None) -> OpResult:
    if key not in OPERATIONS:
        raise ProcessingError(f"Unknown operation: {key!r}")
    op = OPERATIONS[key]
    img = ensure_rgb(img)
    try:
        result = op.func(img, **op.clean_params(params))
    except Exception as exc:  # pragma: no cover - defensive
        raise ProcessingError(f"{op.name} failed: {exc}") from exc
    result.image = ensure_rgb(result.image)
    return result


def operations_by_category() -> Dict[str, List[Operation]]:
    grouped: Dict[str, List[Operation]] = {c: [] for c in CATEGORIES}
    for op in OPERATIONS.values():
        grouped.setdefault(op.category, []).append(op)
    return grouped


# =========================================================================== #
# Small helpers
# =========================================================================== #
def _odd(k: int, minimum: int = 1) -> int:
    k = max(int(k), minimum)
    return k if k % 2 == 1 else k + 1


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _font_scale(img: np.ndarray) -> float:
    return float(np.clip(min(img.shape[:2]) / 620.0, 0.5, 3.0))


def _put_label(img: np.ndarray, text: str, org: Tuple[int, int], color=(255, 255, 255), scale: float | None = None,
               center: bool = False) -> None:
    """Readable text: dark outline + coloured fill."""
    scale = scale or _font_scale(img)
    thick = max(1, int(round(scale * 2)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org
    if center:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x, y = int(x - tw / 2), int(y + th / 2)
    cv2.putText(img, text, (x, y), font, scale, (15, 15, 15), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


DRAW_COLORS = {
    "Lime": (57, 255, 20),
    "Red": (255, 59, 48),
    "Cyan": (0, 229, 255),
    "Yellow": (255, 214, 10),
    "Magenta": (255, 45, 220),
    "White": (255, 255, 255),
}


def _palette(i: int) -> Tuple[int, int, int]:
    """Distinct, bright colours via golden-ratio hue stepping."""
    hue = int((i * 0.61803398875 % 1.0) * 179)
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def _foreground_mask(gray: np.ndarray, method: str) -> np.ndarray:
    """Binary mask where objects are white (255) on black, whatever the background."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if method == "Canny edges":
        v = float(np.median(blurred))
        edges = cv2.Canny(blurred, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(cv2.dilate(edges, kernel, iterations=1), cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask
    if method == "Adaptive":
        block = _odd(max(11, min(gray.shape[:2]) // 25))
        mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 5)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Otsu (auto) – flip so that the minority (objects) is white
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:  # background (image border) came out white -> invert
        mask = cv2.bitwise_not(mask)
    return mask


def _object_mask(img: np.ndarray, method: str) -> np.ndarray:
    """Colour-aware segmentation for contour/shape detection.

    For "Otsu (auto)" we measure how far each pixel's *colour* (CIE Lab) is from
    the background colour estimated on the image border, then Otsu-threshold
    that distance. Unlike brightness-only Otsu this also separates a yellow
    object from a white background, or a dark object from a dark-but-coloured one.
    """
    if method != "Otsu (auto)":
        return _foreground_mask(_gray(img), method)
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    bw = max(2, min(img.shape[:2]) // 50)
    border = np.concatenate([lab[:bw].reshape(-1, 3), lab[-bw:].reshape(-1, 3),
                             lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3)])
    background = np.median(border, axis=0)
    dist = np.linalg.norm(lab - background, axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


# =========================================================================== #
# BASICS
# =========================================================================== #
@register("grayscale", "Grayscale", "Basics", "⚫",
          "Remove colour information. Luminosity matches human perception; average and lightness are alternatives.",
          [Param("method", "Method", "choice", "Luminosity", choices=("Luminosity", "Average", "Lightness"))])
def op_grayscale(img, method):
    if method == "Average":
        gray = img.mean(axis=2).round().astype(np.uint8)
    elif method == "Lightness":
        gray = ((img.max(axis=2).astype(np.uint16) + img.min(axis=2)) // 2).astype(np.uint8)
    else:
        gray = _gray(img)
    return OpResult(_rgb(gray), f"Converted to grayscale using the {method.lower()} method.")


@register("blur", "Blur", "Basics", "💧",
          "Smooth the image. Gaussian is natural, Median removes salt-and-pepper noise, Bilateral keeps edges sharp.",
          [Param("method", "Blur type", "choice", "Gaussian", choices=("Gaussian", "Median", "Bilateral", "Box", "Motion")),
           Param("kernel", "Strength (kernel size)", "slider", 15, 1, 61, 2, help="Larger = blurrier. Always odd.")])
def op_blur(img, method, kernel):
    k = _odd(kernel)
    if method == "Median":
        out = cv2.medianBlur(img, k)
    elif method == "Bilateral":
        d = min(k, 25)
        out = cv2.bilateralFilter(img, d, sigmaColor=k * 6, sigmaSpace=k * 6)
    elif method == "Box":
        out = cv2.blur(img, (k, k))
    elif method == "Motion":
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        out = cv2.filter2D(img, -1, kern)
    else:
        out = cv2.GaussianBlur(img, (k, k), 0)
    return OpResult(out, f"{method} blur with a {k}×{k} kernel.")


@register("edges", "Edge Detection", "Basics", "📐",
          "Find boundaries where brightness changes sharply. Canny gives thin clean lines; Sobel/Scharr/Laplacian show gradient strength.",
          [Param("method", "Detector", "choice", "Canny", choices=("Canny", "Sobel", "Scharr", "Laplacian")),
           Param("low", "Low threshold (Canny)", "slider", 60, 0, 255, 1),
           Param("high", "High threshold (Canny)", "slider", 160, 0, 255, 1),
           Param("auto", "Auto thresholds (Canny)", "checkbox", False, help="Pick thresholds from the median brightness."),
           Param("style", "Display", "choice", "White on black", choices=("White on black", "Black on white", "Overlay on image"))])
def op_edges(img, method, low, high, auto, style):
    gray = cv2.GaussianBlur(_gray(img), (5, 5), 0)
    note = ""
    if method == "Canny":
        if auto:
            v = float(np.median(gray))
            low, high = int(max(0, 0.66 * v)), int(min(255, 1.33 * v))
        low, high = min(low, high), max(low, high)
        edges = cv2.Canny(gray, low, high)
        note = f" (thresholds {low}/{high})"
    elif method in ("Sobel", "Scharr"):
        if method == "Sobel":
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        else:
            gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        edges = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        edges = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_16S, ksize=3))
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)

    strong = edges > 60
    pct = 100.0 * strong.mean()
    if style == "Overlay on image":
        out = (img * 0.55).astype(np.uint8)
        out[strong] = DRAW_COLORS["Lime"]
    elif style == "Black on white":
        out = _rgb(255 - edges)
    else:
        out = _rgb(edges)
    return OpResult(out, f"{method} edges{note}: {pct:.1f}% of pixels are edges.")


# =========================================================================== #
# TRANSFORM
# =========================================================================== #
_BORDERS = {
    "Black": (cv2.BORDER_CONSTANT, (0, 0, 0)),
    "White": (cv2.BORDER_CONSTANT, (255, 255, 255)),
    "Replicate edge": (cv2.BORDER_REPLICATE, None),
    "Mirror": (cv2.BORDER_REFLECT_101, None),
}


@register("rotate", "Rotate", "Transform", "🔄",
          "Rotate around the centre. 'Fit whole image' grows the canvas so no corner gets cut off.",
          [Param("angle", "Angle (° counter-clockwise)", "slider", 45, -180, 180, 1),
           Param("expand", "Fit whole image (expand canvas)", "checkbox", True),
           Param("border", "Fill empty area with", "choice", "Black", choices=tuple(_BORDERS))])
def op_rotate(img, angle, expand, border):
    angle = int(angle)
    # exact, lossless fast path for right angles
    exact = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, -90: cv2.ROTATE_90_CLOCKWISE,
             180: cv2.ROTATE_180, -180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
    if angle == 0:
        return OpResult(img.copy(), "Angle is 0°, image unchanged.")
    if angle in exact and expand:
        return OpResult(cv2.rotate(img, exact[angle]), f"Rotated {angle}° (lossless).")

    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    out_w, out_h = w, h
    if expand:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        out_w, out_h = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
        M[0, 2] += out_w / 2.0 - cx
        M[1, 2] += out_h / 2.0 - cy
    mode, value = _BORDERS[border]
    out = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=mode,
                         borderValue=value if value is not None else 0)
    return OpResult(out, f"Rotated {angle}° → output {out_w}×{out_h}.")


@register("flip", "Flip / Mirror", "Transform", "🪞",
          "Mirror the image horizontally, vertically, or both.",
          [Param("direction", "Direction", "choice", "Horizontal", choices=("Horizontal", "Vertical", "Both"))],
          custom=True)
def op_flip(img, direction):
    code = {"Horizontal": 1, "Vertical": 0, "Both": -1}[direction]
    return OpResult(cv2.flip(img, code), f"Flipped {direction.lower()}.")


@register("resize", "Resize", "Transform", "📏",
          "Scale the image up or down. 'Auto' uses INTER_AREA when shrinking and bicubic when enlarging.",
          [Param("scale", "Scale (%)", "slider", 50, 10, 300, 5),
           Param("interpolation", "Interpolation", "choice", "Auto",
                 choices=("Auto", "Nearest", "Bilinear", "Bicubic", "Lanczos"))],
          custom=True)
def op_resize(img, scale, interpolation):
    h, w = img.shape[:2]
    f = scale / 100.0
    new_w, new_h = max(1, round(w * f)), max(1, round(h * f))
    if new_w * new_h > 40_000_000:
        raise ProcessingError("Result would exceed 40 megapixels. Choose a smaller scale.")
    interp = {
        "Nearest": cv2.INTER_NEAREST, "Bilinear": cv2.INTER_LINEAR,
        "Bicubic": cv2.INTER_CUBIC, "Lanczos": cv2.INTER_LANCZOS4,
    }.get(interpolation, cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    return OpResult(cv2.resize(img, (new_w, new_h), interpolation=interp), f"Resized {w}×{h} → {new_w}×{new_h}.")


@register("crop", "Center Crop", "Transform", "✂️",
          "Crop around the centre to a classic aspect ratio, or keep a percentage of the frame.",
          [Param("aspect", "Aspect ratio", "choice", "1:1 (square)",
                 choices=("1:1 (square)", "4:3", "3:2", "16:9", "9:16 (story)", "Free")),
           Param("zoom", "Keep (% of frame)", "slider", 100, 20, 100, 1)],
          custom=True)
def op_crop(img, aspect, zoom):
    h, w = img.shape[:2]
    ratios = {"1:1 (square)": 1.0, "4:3": 4 / 3, "3:2": 3 / 2, "16:9": 16 / 9, "9:16 (story)": 9 / 16}
    if aspect in ratios:
        r = ratios[aspect]
        cw, ch = (w, int(w / r)) if w / h < r else (int(h * r), h)
    else:
        cw, ch = w, h
    cw, ch = max(1, int(cw * zoom / 100)), max(1, int(ch * zoom / 100))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return OpResult(img[y0:y0 + ch, x0:x0 + cw].copy(), f"Cropped to {cw}×{ch} ({aspect}).")


# =========================================================================== #
# ENHANCE
# =========================================================================== #
@register("brightness_contrast", "Brightness & Contrast", "Enhance", "☀️",
          "Shift brightness and stretch contrast around mid-grey, like the sliders in a photo editor.",
          [Param("brightness", "Brightness", "slider", 20, -100, 100, 1),
           Param("contrast", "Contrast", "slider", 20, -100, 100, 1)],
          custom=True)
def op_brightness_contrast(img, brightness, contrast):
    alpha = 1.0 + contrast / 100.0
    beta = brightness * 1.27
    out = _clip((img.astype(np.float32) - 128.0) * alpha + 128.0 + beta)
    return OpResult(out, f"Brightness {brightness:+d}, contrast {contrast:+d} (α={alpha:.2f}).")


@register("enhance", "Image Enhancement", "Enhance", "✨",
          "Improve visibility. CLAHE boosts local contrast without blowing out highlights; Auto Enhance chains white balance, CLAHE, vibrance and sharpening.",
          [Param("method", "Method", "choice", "Auto Enhance",
                 choices=("Auto Enhance", "CLAHE (adaptive)", "Histogram Equalization", "Auto White Balance", "Detail Enhance")),
           Param("clip_limit", "Strength / clip limit (CLAHE)", "slider", 2.0, 0.5, 8.0, 0.1),
           Param("tiles", "Tile grid size (CLAHE)", "slider", 8, 2, 16, 1)])
def op_enhance(img, method, clip_limit, tiles):
    def clahe(src, clip):
        lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    def white_balance(src):
        f = src.astype(np.float32)
        # Gray-world with clamped gains and partial strength: corrects colour casts
        # without tinting images that are legitimately dominated by one colour.
        means = f.reshape(-1, 3).mean(axis=0)
        gains = np.clip(means.mean() / np.maximum(means, 1e-6), 0.85, 1.18)
        gains = 1.0 + 0.7 * (gains - 1.0)
        return _clip(f * gains)

    if method == "CLAHE (adaptive)":
        out = clahe(img, clip_limit)
    elif method == "Histogram Equalization":
        ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    elif method == "Auto White Balance":
        out = white_balance(img)
    elif method == "Detail Enhance":
        out = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    else:  # Auto Enhance
        out = clahe(white_balance(img), clip_limit)
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        blurred = cv2.GaussianBlur(out, (0, 0), 1.5)
        out = cv2.addWeighted(out, 1.4, blurred, -0.4, 0)
    before, after = _gray(img).std(), _gray(out).std()
    return OpResult(out, f"{method}: contrast σ {before:.1f} → {after:.1f}.")


@register("sharpen", "Sharpen", "Enhance", "🔪",
          "Make details crisper. Unsharp mask is what professional photo tools use.",
          [Param("method", "Method", "choice", "Unsharp Mask", choices=("Unsharp Mask", "Laplacian Kernel", "High-pass")),
           Param("amount", "Amount", "slider", 1.0, 0.1, 3.0, 0.1),
           Param("radius", "Radius (unsharp / high-pass)", "slider", 2.0, 0.5, 10.0, 0.5)],
          custom=True)
def op_sharpen(img, method, amount, radius):
    f = img.astype(np.float32)
    if method == "Laplacian Kernel":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        sharp = cv2.filter2D(f, -1, kernel)
        out = f + amount * (sharp - f)
    elif method == "High-pass":
        high = f - cv2.GaussianBlur(f, (0, 0), radius)
        out = f + amount * high * 1.5
    else:
        blurred = cv2.GaussianBlur(f, (0, 0), radius)
        out = f * (1 + amount) - blurred * amount
    out = _clip(out)
    b, a = cv2.Laplacian(_gray(img), cv2.CV_64F).var(), cv2.Laplacian(_gray(out), cv2.CV_64F).var()
    return OpResult(out, f"{method}: sharpness score {b:.0f} → {a:.0f}.")


@register("gamma", "Gamma Correction", "Enhance", "🌗",
          "Brighten shadows (gamma > 1) or deepen them (gamma < 1) without clipping highlights.",
          [Param("gamma", "Gamma (>1 brightens)", "slider", 1.5, 0.1, 4.0, 0.05)],
          custom=True)
def op_gamma(img, gamma):
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).clip(0, 255).astype(np.uint8)
    return OpResult(cv2.LUT(img, lut), f"Gamma {gamma:.2f} applied via lookup table.")


@register("color_adjust", "Color & Temperature", "Enhance", "🎨",
          "Adjust saturation, rotate hues and warm up / cool down the image.",
          [Param("saturation", "Saturation (%)", "slider", 130, 0, 300, 5),
           Param("hue", "Hue shift (°)", "slider", 0, -180, 180, 5),
           Param("warmth", "Temperature (cool ← → warm)", "slider", 0, -100, 100, 5)],
          custom=True)
def op_color_adjust(img, saturation, hue, warmth):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue / 2.0) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation / 100.0, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out[:, :, 0] += warmth * 0.35
    out[:, :, 2] -= warmth * 0.35
    return OpResult(_clip(out), f"Saturation {saturation}%, hue {hue:+d}°, temperature {warmth:+d}.")


@register("denoise", "Denoise", "Enhance", "🧹",
          "Remove grain and sensor noise. Non-local means is slower but much higher quality.",
          [Param("method", "Method", "choice", "Non-local Means", choices=("Non-local Means", "Median (fast)", "Bilateral")),
           Param("strength", "Strength", "slider", 10, 1, 30, 1)],
          custom=True)
def op_denoise(img, method, strength):
    if method == "Median (fast)":
        out = cv2.medianBlur(img, _odd(strength // 3 + 1, 3))
    elif method == "Bilateral":
        out = cv2.bilateralFilter(img, 9, strength * 8, strength * 8)
    else:
        out = cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
    return OpResult(out, f"{method} denoising (strength {strength}).")


# =========================================================================== #
# DETECTION & ANALYSIS
# =========================================================================== #
@register("threshold", "Thresholding", "Detection & Analysis", "◐",
          "Turn the image into pure black and white. Otsu picks the threshold automatically; adaptive methods handle uneven lighting (try the document sample).",
          [Param("method", "Method", "choice", "Otsu (auto)",
                 choices=("Binary", "Binary Inverted", "Otsu (auto)", "Triangle (auto)",
                          "Adaptive Mean", "Adaptive Gaussian", "Truncate", "To Zero")),
           Param("thresh", "Threshold (manual methods)", "slider", 127, 0, 255, 1),
           Param("block", "Block size (adaptive)", "slider", 31, 3, 101, 2),
           Param("c", "Constant C (adaptive)", "slider", 10, -20, 30, 1)],
          custom=True)
def op_threshold(img, method, thresh, block, c):
    gray = _gray(img)
    if method.startswith("Adaptive"):
        algo = cv2.ADAPTIVE_THRESH_MEAN_C if "Mean" in method else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        out = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, algo, cv2.THRESH_BINARY, _odd(block, 3), c)
        info = f"{method} threshold (block {_odd(block, 3)}, C={c})."
    elif method in ("Otsu (auto)", "Triangle (auto)"):
        flag = cv2.THRESH_OTSU if method.startswith("Otsu") else cv2.THRESH_TRIANGLE
        t, out = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + flag)
        info = f"{method.split()[0]} automatically chose T = {t:.0f}."
    else:
        mode = {"Binary": cv2.THRESH_BINARY, "Binary Inverted": cv2.THRESH_BINARY_INV,
                "Truncate": cv2.THRESH_TRUNC, "To Zero": cv2.THRESH_TOZERO}[method]
        _, out = cv2.threshold(gray, thresh, 255, mode)
        info = f"{method} threshold at T = {thresh}."
    white = 100.0 * (out > 127).mean()
    return OpResult(_rgb(out), f"{info} {white:.1f}% white pixels.")


@register("morphology", "Morphology", "Detection & Analysis", "🧱",
          "Grow, shrink or clean shapes. Opening removes specks, Closing fills small holes, Gradient outlines objects.",
          [Param("operation", "Operation", "choice", "Opening",
                 choices=("Erode", "Dilate", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat")),
           Param("shape", "Kernel shape", "choice", "Ellipse", choices=("Rectangle", "Ellipse", "Cross")),
           Param("kernel", "Kernel size", "slider", 5, 1, 31, 2),
           Param("iterations", "Iterations", "slider", 1, 1, 10, 1),
           Param("binarize", "Binarize first (Otsu)", "checkbox", False)],
          custom=True)
def op_morphology(img, operation, shape, kernel, iterations, binarize):
    src = img
    if binarize:
        _, bw = cv2.threshold(_gray(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        src = _rgb(bw)
    shapes = {"Rectangle": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}
    k = cv2.getStructuringElement(shapes[shape], (_odd(kernel), _odd(kernel)))
    ops = {"Erode": cv2.MORPH_ERODE, "Dilate": cv2.MORPH_DILATE, "Opening": cv2.MORPH_OPEN,
           "Closing": cv2.MORPH_CLOSE, "Gradient": cv2.MORPH_GRADIENT,
           "Top Hat": cv2.MORPH_TOPHAT, "Black Hat": cv2.MORPH_BLACKHAT}
    out = cv2.morphologyEx(src, ops[operation], k, iterations=iterations)
    return OpResult(out, f"{operation} with {_odd(kernel)}×{_odd(kernel)} {shape.lower()} kernel × {iterations}.")


@register("contours", "Contour Detection", "Detection & Analysis", "〰️",
          "Find object outlines, count them and measure their area. Works best on objects that stand out from the background.",
          [Param("detection", "Segmentation", "choice", "Otsu (auto)", choices=("Otsu (auto)", "Canny edges", "Adaptive")),
           Param("mode", "Contours to find", "choice", "Outer only", choices=("Outer only", "All (incl. holes)")),
           Param("min_area", "Minimum area (px²)", "slider", 300, 0, 20000, 50),
           Param("color", "Colour", "choice", "Rainbow", choices=("Rainbow",) + tuple(DRAW_COLORS)),
           Param("thickness", "Line thickness", "slider", 3, 1, 15, 1),
           Param("fill", "Fill shapes", "checkbox", False),
           Param("boxes", "Bounding boxes", "checkbox", False),
           Param("numbers", "Number each contour", "checkbox", True),
           Param("background", "Background", "choice", "Original", choices=("Original", "Black", "Binary mask"))])
def op_contours(img, detection, mode, min_area, color, thickness, fill, boxes, numbers, background):
    mask = _object_mask(img, detection)
    retrieval = cv2.RETR_EXTERNAL if mode == "Outer only" else cv2.RETR_LIST
    contours, _ = cv2.findContours(mask, retrieval, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted((c for c in contours if cv2.contourArea(c) >= min_area), key=cv2.contourArea, reverse=True)

    if background == "Black":
        canvas = np.zeros_like(img)
    elif background == "Binary mask":
        canvas = _rgb(mask)
    else:
        canvas = img.copy()

    if fill and contours:
        overlay = canvas.copy()
        for i, c in enumerate(contours):
            col = _palette(i) if color == "Rainbow" else DRAW_COLORS[color]
            cv2.drawContours(overlay, [c], -1, col, cv2.FILLED, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

    for i, c in enumerate(contours):
        col = _palette(i) if color == "Rainbow" else DRAW_COLORS[color]
        cv2.drawContours(canvas, [c], -1, col, thickness, cv2.LINE_AA)
        if boxes:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), DRAW_COLORS["Yellow"], max(1, thickness // 2), cv2.LINE_AA)
        if numbers:
            m = cv2.moments(c)
            if m["m00"]:
                _put_label(canvas, str(i + 1), (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])), center=True)

    if not contours:
        return OpResult(canvas, "No contours found. Lower the minimum area or try another segmentation method.")
    areas = [cv2.contourArea(c) for c in contours]
    return OpResult(canvas, f"Found {len(contours)} contour(s). Largest {areas[0]:,.0f} px², "
                            f"total area {sum(areas):,.0f} px² ({100 * sum(areas) / mask.size:.1f}% of image).")


_SHAPE_COLORS = {
    "Triangle": (255, 82, 82), "Square": (64, 156, 255), "Rectangle": (0, 200, 120),
    "Pentagon": (180, 100, 255), "Hexagon": (255, 196, 0), "Circle": (0, 220, 220),
    "Ellipse": (255, 120, 200), "Star": (255, 140, 0), "Polygon": (200, 200, 200),
}


def classify_shape(contour: np.ndarray, epsilon_pct: float) -> Tuple[str, np.ndarray]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_pct / 100.0 * peri, True)
    v = len(approx)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (peri * peri) if peri else 0
    hull_area = cv2.contourArea(cv2.convexHull(contour)) or 1
    solidity = area / hull_area

    if v == 3:
        return "Triangle", approx
    if v == 4 and solidity > 0.9:
        (_, _), (w, h), _ = cv2.minAreaRect(approx)
        ratio = w / h if h else 0
        return ("Square" if 0.92 <= ratio <= 1.08 else "Rectangle"), approx
    if solidity < 0.75 and 8 <= v <= 12:
        return "Star", approx
    if v == 5 and solidity > 0.9:
        return "Pentagon", approx
    if v == 6 and solidity > 0.9:
        return "Hexagon", approx
    if circularity > 0.78 and len(contour) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(contour)
        ratio = max(ma, mi) / max(min(ma, mi), 1e-6)
        return ("Circle" if ratio < 1.15 else "Ellipse"), approx
    if circularity > 0.65 and v > 6:
        return "Ellipse", approx
    return "Polygon", approx


@register("shapes", "Shape Detection", "Detection & Analysis", "🔺",
          "Recognise geometric shapes (triangle, square, rectangle, pentagon, hexagon, circle, ellipse, star) using polygon approximation, circularity and solidity. Try the 'shapes' sample.",
          [Param("detection", "Segmentation", "choice", "Otsu (auto)", choices=("Otsu (auto)", "Canny edges", "Adaptive")),
           Param("min_area", "Minimum area (px²)", "slider", 800, 50, 30000, 50),
           Param("epsilon", "Approximation precision (%)", "slider", 3.0, 0.5, 8.0, 0.5,
                 help="Lower = follows the outline more closely (more vertices)."),
           Param("thickness", "Line thickness", "slider", 4, 1, 15, 1),
           Param("labels", "Show labels", "checkbox", True),
           Param("vertices", "Mark vertices", "checkbox", True)])
def op_shapes(img, detection, min_area, epsilon, thickness, labels, vertices):
    mask = _object_mask(img, detection)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = img.copy()
    found = Counter()
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area:
            continue
        name, approx = classify_shape(c, epsilon)
        found[name] += 1
        col = _SHAPE_COLORS[name]
        cv2.drawContours(canvas, [c], -1, col, thickness, cv2.LINE_AA)
        if vertices and name not in ("Circle", "Ellipse"):
            for p in approx.reshape(-1, 2):
                cv2.circle(canvas, tuple(int(v) for v in p), thickness + 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(int(v) for v in p), thickness + 3, col, 2, cv2.LINE_AA)
        if labels:
            m = cv2.moments(c)
            if m["m00"]:
                cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                _put_label(canvas, name, (cx, cy), center=True)
    if not found:
        return OpResult(canvas, "No shapes found. Lower the minimum area or change segmentation.")
    summary = ", ".join(f"{n} {s}{'s' if n > 1 else ''}" for s, n in found.most_common())
    return OpResult(canvas, f"Detected {sum(found.values())} shape(s): {summary}.")


@lru_cache(maxsize=4)
def _cascade(name: str) -> Optional[cv2.CascadeClassifier]:
    """Find a Haar cascade: OpenCV's bundled data first, then assets/models."""
    candidates = [MODELS_DIR / name]
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data_dir:
        candidates.insert(0, Path(data_dir) / name)
    for path in candidates:
        if path.exists():
            clf = cv2.CascadeClassifier(str(path))
            if not clf.empty():
                return clf
    return None


@register("faces", "Face Detection", "Detection & Analysis", "🙂",
          "Detect faces with a Haar cascade. 'Privacy' modes blur or pixelate every face, handy before sharing photos.",
          [Param("style", "Mode", "choice", "Boxes", choices=("Boxes", "Privacy blur", "Privacy pixelate")),
           Param("scale_factor", "Scale factor", "slider", 1.1, 1.02, 1.5, 0.01, help="Smaller = slower but finds more faces."),
           Param("min_neighbors", "Min neighbours", "slider", 5, 1, 15, 1, help="Higher = fewer false positives."),
           Param("eyes", "Also detect eyes", "checkbox", False)],
          custom=True)
def op_faces(img, style, scale_factor, min_neighbors, eyes):
    face_clf = _cascade("haarcascade_frontalface_default.xml")
    if face_clf is None:
        return OpResult(img.copy(), "Face model not found (assets/models/haarcascade_frontalface_default.xml).")
    gray = cv2.equalizeHist(_gray(img))
    min_size = max(24, min(img.shape[:2]) // 25)
    faces = face_clf.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_size, min_size))
    out = img.copy()
    eye_clf = _cascade("haarcascade_eye.xml") if eyes else None
    eye_count = 0
    for i, (x, y, w, h) in enumerate(faces):
        roi = out[y:y + h, x:x + w]
        if style == "Privacy blur":
            k = _odd(max(w, h) // 3)
            out[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)
        elif style == "Privacy pixelate":
            small = cv2.resize(roi, (max(1, w // 12), max(1, h // 12)), interpolation=cv2.INTER_LINEAR)
            out[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            t = max(2, w // 60)
            cv2.rectangle(out, (x, y), (x + w, y + h), DRAW_COLORS["Lime"], t, cv2.LINE_AA)
            _put_label(out, f"Face {i + 1}", (x, max(20, y - 10)), DRAW_COLORS["Lime"])
            if eye_clf is not None:
                for (ex, ey, ew, eh) in eye_clf.detectMultiScale(gray[y:y + h // 2 + h // 6, x:x + w], 1.1, 8):
                    eye_count += 1
                    cv2.ellipse(out, (x + ex + ew // 2, y + ey + eh // 2), (ew // 2, eh // 2), 0, 0, 360,
                                DRAW_COLORS["Cyan"], max(1, t // 2), cv2.LINE_AA)
    info = f"Detected {len(faces)} face(s)" + (f" and {eye_count} eye(s)." if eyes and style == "Boxes" else ".")
    return OpResult(out, info)


_HUES = {"Red": 0, "Orange": 14, "Yellow": 28, "Green": 60, "Cyan": 90, "Blue": 115, "Purple": 140, "Pink": 165}


@register("color_splash", "Color Splash", "Detection & Analysis", "🌈",
          "Keep one colour and turn everything else grey. Uses an HSV hue mask; also a simple colour-segmentation demo.",
          [Param("color", "Colour to keep", "choice", "Red", choices=tuple(_HUES)),
           Param("tolerance", "Hue tolerance", "slider", 12, 2, 40, 1),
           Param("min_saturation", "Minimum saturation", "slider", 70, 0, 255, 5),
           Param("show_mask", "Show mask only", "checkbox", False)],
          custom=True)
def op_color_splash(img, color, tolerance, min_saturation, show_mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    diff = np.abs(h.astype(np.int16) - _HUES[color])
    diff = np.minimum(diff, 180 - diff)  # hue is circular
    mask = ((diff <= tolerance) & (s >= min_saturation) & (v >= 40)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pct = 100.0 * (mask > 0).mean()
    if show_mask:
        return OpResult(_rgb(mask), f"{color} mask covers {pct:.1f}% of the image.")
    soft = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32)[..., None] / 255.0
    out = img * soft + _rgb(_gray(img)) * (1 - soft)
    return OpResult(_clip(out), f"Kept {color.lower()} tones ({pct:.1f}% of pixels).")


# =========================================================================== #
# ARTISTIC
# =========================================================================== #
@register("invert", "Negative", "Artistic", "🔁", "Invert every colour, like a photo negative.", custom=True)
def op_invert(img):
    return OpResult(255 - img, "Colours inverted.")


@register("sepia", "Sepia / Vintage", "Artistic", "📜",
          "Warm brown tones of old photographs.",
          [Param("intensity", "Intensity (%)", "slider", 100, 0, 100, 5)], custom=True)
def op_sepia(img, intensity):
    kernel = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], np.float32)
    sep = cv2.transform(img.astype(np.float32), kernel)
    a = intensity / 100.0
    return OpResult(_clip(sep * a + img * (1 - a)), f"Sepia at {intensity}%.")


@register("sketch", "Pencil Sketch", "Artistic", "✏️",
          "Turn a photo into a hand-drawn pencil sketch using the colour-dodge technique.",
          [Param("style", "Style", "choice", "Graphite", choices=("Graphite", "Coloured pencil")),
           Param("detail", "Stroke softness", "slider", 21, 3, 81, 2),
           Param("darkness", "Darkness", "slider", 1.4, 1.0, 3.0, 0.1)], custom=True)
def op_sketch(img, style, detail, darkness):
    gray = _gray(img)
    blurred = cv2.GaussianBlur(255 - gray, (_odd(detail), _odd(detail)), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    sketch = (255 * (sketch / 255.0) ** darkness).astype(np.uint8)
    if style == "Coloured pencil":
        out = _clip(img.astype(np.float32) * (sketch[..., None] / 255.0) * 0.6 + _rgb(sketch) * 0.4)
    else:
        out = _rgb(sketch)
    return OpResult(out, f"{style} sketch.")


@register("cartoon", "Cartoon", "Artistic", "🖍️",
          "Flat colours plus bold ink outlines: bilateral smoothing, colour quantisation and adaptive-threshold edges.",
          [Param("levels", "Colour levels", "slider", 8, 2, 24, 1),
           Param("smoothness", "Smoothness", "slider", 5, 1, 10, 1),
           Param("line_size", "Outline thickness", "slider", 9, 3, 25, 2)], custom=True)
def op_cartoon(img, levels, smoothness, line_size):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    for _ in range(smoothness):
        small = cv2.bilateralFilter(small, 9, 40, 9)
    color = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    step = 256 // levels
    color = _clip((color // step) * step + step // 2)
    gray = cv2.medianBlur(_gray(img), 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, _odd(line_size, 3), 7)
    return OpResult(cv2.bitwise_and(color, color, mask=edges), f"Cartoon with {levels} colour levels.")


@register("emboss", "Emboss", "Artistic", "🪨", "Raised, metal-relief look.",
          [Param("strength", "Strength", "slider", 1.0, 0.2, 3.0, 0.1),
           Param("keep_color", "Keep colour", "checkbox", False)], custom=True)
def op_emboss(img, strength, keep_color):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32) * strength
    kernel[1, 1] = 1
    if keep_color:
        out = cv2.filter2D(img, -1, kernel)
    else:
        out = _rgb(_clip(cv2.filter2D(_gray(img).astype(np.float32), -1, kernel) - _gray(img) + 128))
    return OpResult(out, f"Emboss strength {strength:.1f}.")


@register("vignette", "Vignette", "Artistic", "🌑", "Darken the corners to pull the eye to the centre.",
          [Param("strength", "Strength (%)", "slider", 60, 0, 100, 5),
           Param("radius", "Clear centre (%)", "slider", 40, 0, 90, 5)], custom=True)
def op_vignette(img, strength, radius):
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    r0 = radius / 100.0 * 0.9
    falloff = np.clip((d - r0) / max(1e-6, 1 - r0), 0, 1) ** 1.6
    mask = 1 - (strength / 100.0) * falloff
    return OpResult(_clip(img * mask[..., None]), f"Vignette {strength}%.")


@register("pixelate", "Pixelate", "Artistic", "👾", "Retro mosaic / 8-bit look.",
          [Param("block", "Block size (px)", "slider", 12, 2, 80, 1)], custom=True)
def op_pixelate(img, block):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    return OpResult(cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST), f"Pixelated with {block}px blocks.")


@register("stylize", "Watercolor", "Artistic", "🖌️", "Painterly watercolour effect (edge-preserving stylisation).",
          [Param("sigma_s", "Brush size", "slider", 60, 10, 200, 5),
           Param("sigma_r", "Colour blending", "slider", 0.45, 0.05, 1.0, 0.05)], custom=True)
def op_stylize(img, sigma_s, sigma_r):
    return OpResult(cv2.stylization(img, sigma_s=float(sigma_s), sigma_r=float(sigma_r)), "Watercolour stylisation.")
