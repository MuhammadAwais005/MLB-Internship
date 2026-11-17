"""
The thresholding techniques.

Every method takes the same prepared Context and returns a raw 0/255 map.
It does not matter which colour ends up as the object: the pipeline decides
that afterwards with the border rule. That keeps every method short and
lets all of them share the same cleanup, so the comparison is fair.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import FIXED_THRESHOLD


@dataclass
class Context:
    """The image in the forms the methods need, computed once."""
    bgr: np.ndarray        # colour image (working size)
    gray: np.ndarray       # grayscale, lightly blurred
    lab: np.ndarray        # CIE Lab colour space (float32)
    hsv: np.ndarray        # HSV colour space

    @classmethod
    def from_image(cls, bgr):
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return cls(bgr=bgr, gray=gray, lab=lab, hsv=hsv)


def border_pixels(image):
    """All pixels along the four image edges - in most photos that's background."""
    return np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])


def odd(value, minimum=3):
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


# ---------------------------------------------------------------------------
# 1-3: classic global thresholds on the grayscale image
# ---------------------------------------------------------------------------

def binary_fixed(ctx):
    """Global threshold at a fixed value (127). The baseline everyone starts with."""
    _, mask = cv2.threshold(ctx.gray, FIXED_THRESHOLD, 255, cv2.THRESH_BINARY)
    return mask


def otsu(ctx):
    """Global threshold chosen automatically: maximises the separation of the two pixel groups."""
    _, mask = cv2.threshold(ctx.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def triangle(ctx):
    """
    Global threshold from the triangle method: draws a line from the histogram
    peak to its far end and cuts where the histogram is furthest from that line.
    Works better than Otsu when the object is small and the background dominates.
    """
    _, mask = cv2.threshold(ctx.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return mask


# ---------------------------------------------------------------------------
# 4: local threshold
# ---------------------------------------------------------------------------

def adaptive_filled(ctx):
    """
    Adaptive (local) threshold. On its own it only marks edges and fine detail,
    so the object's outline is closed with a morphological closing and then
    filled in to become a solid region. Insensitive to lighting changes.
    """
    h, w = ctx.gray.shape
    block = odd(min(h, w) / 20, minimum=11)
    edges = cv2.adaptiveThreshold(ctx.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, block, 7)
    k = odd(min(h, w) / 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    filled = edges.copy()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


# ---------------------------------------------------------------------------
# 5-7: thresholds on a better input than plain grayscale
# ---------------------------------------------------------------------------

def otsu_light_corrected(ctx):
    """
    Otsu after removing uneven lighting: the image is divided by a very blurred
    copy of itself (an estimate of the light falling on the scene), so a dark
    corner and a bright corner of the same surface end up the same grey.
    """
    image = ctx.gray.astype(np.float32)
    lighting = cv2.GaussianBlur(image, (0, 0), sigmaX=min(image.shape) / 4)
    flat = np.clip(image / (lighting + 1.0) * lighting.mean(), 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def saturation_otsu(ctx):
    """
    Otsu on the colour saturation channel (HSV). White, grey and black
    backgrounds - and shadows on them - have almost no saturation, while most
    coloured objects have a lot. So shadows are ignored automatically.
    """
    saturation = cv2.GaussianBlur(ctx.hsv[:, :, 1], (5, 5), 0)
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def background_distance(ctx):
    """
    Colour distance from the background, then Otsu. The background colour is
    the median of the border pixels; every pixel gets its distance to that
    colour in Lab space (close to how humans see colour difference).
    Brightness (L) counts half, so a shadow - same colour, just darker - stays
    close to the background while a real object is far away.
    """
    background = np.median(border_pixels(ctx.lab), axis=0)
    diff = ctx.lab - background
    distance = np.sqrt((0.5 * diff[:, :, 0]) ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)
    distance = cv2.GaussianBlur(distance, (5, 5), 0)
    distance = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# name -> (label shown to users, function, one-line description)
METHODS = {
    "binary_fixed": ("Binary (T=127)", binary_fixed, "Global threshold at a fixed value"),
    "otsu": ("Otsu", otsu, "Global threshold picked automatically from the histogram"),
    "triangle": ("Triangle", triangle, "Global threshold for histograms dominated by background"),
    "adaptive": ("Adaptive + fill", adaptive_filled, "Local threshold, outline closed and filled"),
    "otsu_light": ("Otsu + light correction", otsu_light_corrected, "Uneven lighting removed before Otsu"),
    "saturation": ("Saturation (HSV) Otsu", saturation_otsu, "Colourful objects on grey/white, ignores shadows"),
    "bg_distance": ("Background colour distance", background_distance, "Distance from the border colour, shadow tolerant"),
}
