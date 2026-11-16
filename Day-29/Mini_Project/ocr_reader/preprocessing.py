"""
Image loading and optional enhancement before OCR.

The enhancement pipeline is the one that worked best in Task 1:
grayscale -> upscale small images -> make text dark-on-light ->
shadow removal -> denoise -> CLAHE contrast.
It is OFF by default: EasyOCR already does well on most clean images,
and enhancement mainly pays off on shadowed, faded or tiny images.
"""

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import MIN_IMAGE_SIZE


def load_image(source):
    """
    Opens any common image format and returns it as an OpenCV BGR array.
    `source` can be a file path or the raw bytes of an image (e.g. an upload).

    Pillow is used instead of cv2.imread because:
      - it fixes phone photos that are stored sideways (EXIF rotation),
      - it reads TIFF / WEBP / palette images reliably,
      - it has no trouble with special characters in Windows paths.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    elif isinstance(source, (str, Path)):
        source = Path(source)

    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def limit_size(image, max_side):
    """Shrinks very large photos (e.g. 4000 px phone pictures) to speed up OCR."""
    longest = max(image.shape[:2])
    if not max_side or longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _has_light_text(gray):
    """True when the text is lighter than its background (white text on a dark sign)."""
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    background = cv2.medianBlur(smooth, 51)
    diff = smooth.astype(np.int16) - background.astype(np.int16)
    return np.count_nonzero(diff > 25) > np.count_nonzero(diff < -25)


def _remove_shadow(gray):
    """Divides the image by its estimated empty-paper background to flatten the lighting."""
    background = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    background = cv2.medianBlur(background, 41)
    return cv2.divide(gray, background, scale=255)


def enhance_for_ocr(image):
    """
    Returns (enhanced_gray_image, scale).

    `scale` tells how much the image was enlarged, so text box coordinates
    can be mapped back onto the original image afterwards.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    scale = 1.0
    longest = max(gray.shape[:2])
    if longest < MIN_IMAGE_SIZE:
        scale = min(4.0, MIN_IMAGE_SIZE / longest)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if _has_light_text(gray):
        gray = cv2.bitwise_not(gray)

    gray = _remove_shadow(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray, scale
