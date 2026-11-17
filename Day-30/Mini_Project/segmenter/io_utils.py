"""Reading and writing images safely (any format, phone rotation, Windows paths)."""

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(source):
    """
    Returns an OpenCV BGR image from a file path or raw bytes (an upload).
    Pillow handles EXIF rotation (phone photos taken in portrait) and formats
    such as TIFF/WEBP; it also has no trouble with non-English Windows paths.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    elif isinstance(source, (str, Path)):
        source = Path(source)
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_longest(image, max_side):
    """Shrinks an image so its longest side is max_side. Returns (image, scale)."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image, 1.0
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return small, scale


def save_image(path, image):
    """cv2.imwrite fails on non-English paths on Windows; imencode + tofile does not."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 92]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    else:
        params = []
    ok, buffer = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise IOError(f"Could not encode {path.name}")
    buffer.tofile(str(path))
    return path


def encode_png(image):
    """Image -> PNG bytes (for download buttons)."""
    ok, buffer = cv2.imencode(".png", image)
    return buffer.tobytes() if ok else b""
