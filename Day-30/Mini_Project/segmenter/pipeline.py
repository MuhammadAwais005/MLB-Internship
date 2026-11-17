"""
Runs every thresholding method, cleans each result the same way, scores it,
and picks the best one.
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import WORK_SIZE
from .document import find_document
from .io_utils import resize_longest
from .methods import METHODS, Context, border_pixels
from .scoring import Score, gradient_magnitude, score_mask


@dataclass
class MethodResult:
    key: str
    label: str
    description: str
    mask: np.ndarray            # final mask at ORIGINAL image size (0/255)
    score: Score
    objects: int
    seconds: float


@dataclass
class Segmentation:
    image: np.ndarray                           # original image (BGR)
    results: dict = field(default_factory=dict)  # key -> MethodResult, sorted best first
    best_key: str = ""
    document: object = None                     # DocumentScan when a document page was found

    @property
    def best(self):
        return self.results[self.best_key]


# ---------------------------------------------------------------------------
# Shared cleanup
# ---------------------------------------------------------------------------

def choose_foreground(mask):
    """Border rule: whatever colour covers most of the image border is background."""
    border = border_pixels(mask)
    return cv2.bitwise_not(mask) if np.count_nonzero(border) > border.size / 2 else mask


def fill_holes(mask):
    """Fills regions that are fully enclosed by foreground (holes, text inside a page, etc.)."""
    background = cv2.bitwise_not(mask)
    count, labels = cv2.connectedComponents(background, connectivity=4)
    outside = set(np.unique(border_pixels(labels)))
    holes = ~np.isin(labels, list(outside)) & (labels > 0)
    filled = mask.copy()
    filled[holes] = 255
    return filled


def clean_mask(raw, keep_all=False):
    """
    Same cleanup for every method so the comparison is fair:
        pick foreground (border rule) -> opening removes specks -> closing
        bridges small gaps -> fill holes -> keep the main object(s).
    Main objects = the largest region plus any region at least 25% of its size
    (so a single document stays alone, but all 24 coins are kept).
    """
    mask = choose_foreground(raw)
    h, w = mask.shape
    k = max(3, min(h, w) // 150) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = fill_holes(mask)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(mask.size * 0.001, 0 if keep_all else 0.25 * areas.max())
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_area]
    return np.isin(labels, keep).astype(np.uint8) * 255, len(keep)


def to_full_size(mask, shape):
    """Scales a working-size mask back to the original resolution with smooth edges."""
    h, w = shape[:2]
    if mask.shape[:2] == (h, w):
        return mask
    big = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.where(big >= 128, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def segment(image, methods=None, keep_all=False):
    """
    Runs the chosen thresholding methods (default: all) on one image.
    Returns a Segmentation with every result, sorted best first.
    """
    small, _ = resize_longest(image, WORK_SIZE)
    ctx = Context.from_image(small)
    gradient = gradient_magnitude(ctx.lab)

    results = []
    for key in methods or METHODS:
        label, func, description = METHODS[key]
        start = time.perf_counter()
        raw = func(ctx)
        mask, objects = clean_mask(raw, keep_all=keep_all)
        seconds = time.perf_counter() - start
        score = score_mask(mask, ctx.lab, gradient)
        results.append(MethodResult(key, label, description, to_full_size(mask, image.shape),
                                    score, objects, seconds))

    results.sort(key=lambda r: r.score.total, reverse=True)
    seg = Segmentation(image=image, results={r.key: r for r in results}, best_key=results[0].key)
    seg.document = find_document(image, seg.best.mask)
    return seg
