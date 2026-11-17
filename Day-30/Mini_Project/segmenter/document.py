"""
Document mode: if the segmented object is a four-cornered page, straighten it
into a flat, top-down "scan" (like a phone scanner app does).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DocumentScan:
    corners: np.ndarray     # 4 x 2 float32, ordered: top-left, top-right, bottom-right, bottom-left
    color: np.ndarray       # straightened colour page
    clean: np.ndarray       # black & white "scanned" version (adaptive threshold)


def order_corners(points):
    """Sorts 4 points into top-left, top-right, bottom-right, bottom-left."""
    points = points.reshape(4, 2).astype(np.float32)
    s, d = points.sum(axis=1), np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(s)], points[np.argmin(d)],
                     points[np.argmax(s)], points[np.argmax(d)]], dtype=np.float32)


def find_quadrilateral(mask):
    """
    Looks for a page-like shape. The largest region must:
      - cover at least 10% of the image,
      - be solid (area / convex hull >= 0.95) - pages have straight edges,
        while a pile of cards or a group of blocks has a ragged outline,
      - simplify to 4 corners that describe its outline closely
        (hull area at most 10% bigger than the 4-corner shape).
    On the test images every real page scored >= 0.96 solidity; the card pile 0.89.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if area < 0.10 * mask.size or hull_area == 0 or area / hull_area < 0.95:
        return None

    perimeter = cv2.arcLength(hull, True)
    for tolerance in (0.01, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(hull, tolerance * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if hull_area / cv2.contourArea(approx) <= 1.10:
                return order_corners(approx)
            return None
    return None


def find_document(image, mask):
    corners = find_quadrilateral(mask)
    if corners is None:
        return None

    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return None

    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners, target)
    color = cv2.warpPerspective(image, matrix, (width, height))

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    block = max(15, (min(width, height) // 40) | 1)
    clean = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)
    return DocumentScan(corners=corners, color=color, clean=clean)
