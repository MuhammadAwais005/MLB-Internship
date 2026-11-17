"""
Judging a segmentation without knowing the right answer.

There is no ground-truth mask for a user's photo, so the "best" method is
picked with four checks that a good object mask passes:

  edge      the outline should sit on real edges in the photo
            (mean gradient on the mask boundary vs. the image average)
  contrast  inside and outside should look different in colour
            (Fisher separation in Lab space)
  solidity  objects and documents are solid shapes, not lace
            (area / convex hull area)
  border    the main object should not spill into the image border
            (that usually means background was taken as foreground)

Each check gives 0..1; the weighted sum becomes a 0-100 score. Masks that
cover almost nothing or almost everything are heavily penalised, and masks
under 10% of the image are scaled down a little (the main object is rarely tiny).
"""

from dataclasses import dataclass

import cv2
import numpy as np

WEIGHTS = {"edge": 0.40, "contrast": 0.25, "solidity": 0.20, "border": 0.15}


@dataclass
class Score:
    total: float
    edge: float
    contrast: float
    solidity: float
    border: float
    coverage: float        # % of the image that is foreground

    def as_dict(self):
        return {k: round(v, 3) for k, v in self.__dict__.items()}


def gradient_magnitude(lab):
    """Colour edge strength: Sobel gradient combined over the L, a and b channels."""
    total = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        channel = cv2.GaussianBlur(lab[:, :, c], (3, 3), 0)
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        total += gx * gx + gy * gy
    return np.sqrt(total)


def edge_score(mask, gradient):
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if not boundary.any():
        return 0.0
    ratio = gradient[boundary].mean() / (gradient.mean() + 1e-6)
    # ratio 1 = the outline is no better than a random line; on the test images
    # good outlines reach 5-15x. A log scale keeps 5x and 12x distinguishable.
    return float(np.clip(np.log(max(ratio, 1e-6)) / np.log(12.0), 0.0, 1.0))


def contrast_score(mask, lab):
    inside, outside = lab[mask > 0], lab[mask == 0]
    if len(inside) < 50 or len(outside) < 50:
        return 0.0
    separation = ((inside.mean(0) - outside.mean(0)) ** 2 / (inside.var(0) + outside.var(0) + 1e-6)).sum()
    return float(separation / (separation + 1.0))


def solidity_score(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas, solid = [], []
    for contour in contours:
        area = cv2.contourArea(contour)
        hull = cv2.contourArea(cv2.convexHull(contour))
        if hull > 0:
            areas.append(area)
            solid.append(area / hull)
    if not areas:
        return 0.0
    return float(np.average(solid, weights=areas))


def border_score(mask):
    edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    touching = np.count_nonzero(edge) / edge.size
    return float(np.clip(1.0 - 2.0 * touching, 0.0, 1.0))


def score_mask(mask, lab, gradient):
    coverage = np.count_nonzero(mask) / mask.size
    if coverage == 0:
        return Score(0, 0, 0, 0, 0, 0)

    parts = {
        "edge": edge_score(mask, gradient),
        "contrast": contrast_score(mask, lab),
        "solidity": solidity_score(mask),
        "border": border_score(mask),
    }
    total = sum(WEIGHTS[k] * v for k, v in parts.items()) * 100

    # a "segmentation" that is almost empty or almost the whole picture is not useful
    if coverage < 0.005 or coverage > 0.95:
        total *= 0.2
    # the MAIN object of a photo is rarely tiny: below 10% of the image the score
    # is scaled down gently (so e.g. just the printed text doesn't beat the page)
    elif coverage < 0.10:
        total *= 0.4 + 0.6 * coverage / 0.10
    return Score(total=total, coverage=coverage * 100, **parts)
