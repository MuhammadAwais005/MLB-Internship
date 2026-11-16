"""
The OCR engine: wraps EasyOCR and turns its raw output into clean text lines.
"""

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DEFAULT_LANGUAGES
from .preprocessing import enhance_for_ocr, limit_size, load_image

# PyTorch prints two harmless warnings every time EasyOCR runs on a CPU
# ("pin_memory ... no accelerator" and a quantization deprecation notice).
# They don't affect the result, so we keep the console clean.
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*quantize.*deprecated.*")
warnings.filterwarnings("ignore", message=".*quantized tensor.*")


@dataclass
class Detection:
    """One piece of text found by EasyOCR."""
    text: str
    confidence: float
    box: list          # 4 corner points [[x, y], ...] in ORIGINAL image coordinates


@dataclass
class OCRResult:
    image_path: Path
    image: np.ndarray                      # original image (BGR), used for display
    detections: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    seconds: float = 0.0
    enhanced: bool = False

    @property
    def text(self):
        return "\n".join(self.lines)

    @property
    def line_count(self):
        return sum(1 for line in self.lines if line.strip())

    @property
    def word_count(self):
        return len(self.text.split())

    @property
    def average_confidence(self):
        if not self.detections:
            return 0.0
        return float(np.mean([d.confidence for d in self.detections]))


def _to_items(detections):
    """Flattens EasyOCR boxes into simple rectangles that are easy to sort and split."""
    items = []
    for d in detections:
        xs, ys = [p[0] for p in d.box], [p[1] for p in d.box]
        items.append({"text": d.text, "left": min(xs), "right": max(xs), "top": min(ys),
                      "bottom": max(ys), "center": sum(ys) / 4, "height": max(ys) - min(ys)})
    return items


def _split(items, start_key, end_key, min_gap):
    """
    Splits items into groups wherever there is an empty band wider than
    min_gap along one axis (top/bottom for rows, left/right for columns).
    """
    items = sorted(items, key=lambda i: i[start_key])
    groups, current, edge = [], [items[0]], items[0][end_key]
    for item in items[1:]:
        if item[start_key] - edge > min_gap:
            groups.append(current)
            current = []
        current.append(item)
        edge = max(edge, item[end_key])
    groups.append(current)
    return groups


def _looks_like_text_column(group, text_height):
    """
    A real text column has at least a few long lines of text. A column of
    prices on a receipt or a table cell column only has short boxes, so it
    stays on the same line as its row instead of being split off.
    """
    long_lines = sum(1 for i in group if i["right"] - i["left"] > 6 * text_height)
    return long_lines >= 3


def _xy_cut(items, text_height):
    """
    Recursive XY-cut, a classic page layout algorithm:
      1. cut the page at wide horizontal gaps (between paragraphs / sections),
      2. inside each part, cut at wide vertical gaps (between text columns),
      3. repeat until nothing can be cut. Each leaf is one text block.
    Blocks come out in reading order: top to bottom, left column before right.
    """
    rows = _split(items, "top", "bottom", min_gap=0.8 * text_height)
    if len(rows) > 1:
        return [block for row in rows for block in _xy_cut(row, text_height)]

    columns = _split(items, "left", "right", min_gap=1.2 * text_height)
    if len(columns) > 1 and all(_looks_like_text_column(c, text_height) for c in columns):
        return [block for column in columns for block in _xy_cut(column, text_height)]

    return [items]


def _block_to_lines(items, text_height):
    """Joins boxes that sit on the same row into one line (left to right)."""
    items = sorted(items, key=lambda i: i["center"])
    lines, current = [], [items[0]]
    for item in items[1:]:
        line_center = sum(i["center"] for i in current) / len(current)
        if abs(item["center"] - line_center) < text_height * 0.5:
            current.append(item)
        else:
            lines.append(current)
            current = [item]
    lines.append(current)
    return [" ".join(i["text"] for i in sorted(line, key=lambda i: i["left"])) for line in lines]


def group_into_lines(detections):
    """
    EasyOCR returns boxes in detection order, not reading order. On a receipt
    you often get all the prices first and then the item names, and on a
    two-column page the columns get mixed. This rebuilds the reading order:
    blocks via XY-cut, then lines inside each block. Blocks are separated
    by an empty line, like paragraphs.
    """
    if not detections:
        return []

    items = _to_items(detections)
    text_height = float(np.median([i["height"] for i in items])) or 1.0

    lines = []
    for block in _xy_cut(items, text_height):
        if lines:
            lines.append("")
        lines += _block_to_lines(block, text_height)
    return lines


class OCREngine:
    """
    Loads the EasyOCR model once and reuses it for every image.
    Loading takes a few seconds, reading an image is much faster after that.
    """

    def __init__(self, languages=None, use_gpu=None):
        self.languages = languages or list(DEFAULT_LANGUAGES)
        self.use_gpu = use_gpu
        self._reader = None

    @property
    def is_loaded(self):
        return self._reader is not None

    def load(self):
        if self._reader is not None:
            return
        # imported here so the GUI window can open instantly while this loads
        import easyocr
        import torch

        if self.use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)

    def read(self, source, enhance=False, name=None, max_side=None):
        """
        source   : path to an image, or the image's raw bytes (uploads)
        enhance  : clean the image before OCR
        name     : file name to use for bytes input (used when saving the .txt)
        max_side : shrink images whose longest side is bigger than this
        """
        self.load()
        if isinstance(source, (bytes, bytearray)):
            image_path = Path(name or "uploaded_image.png")
        else:
            image_path = Path(source)
        image = limit_size(load_image(source), max_side)

        if enhance:
            to_read, scale = enhance_for_ocr(image)
        else:
            to_read, scale = image, 1.0

        start = time.perf_counter()
        raw_results = self._reader.readtext(to_read)
        seconds = time.perf_counter() - start

        detections = []
        for box, text, confidence in raw_results:
            original_box = [[float(x) / scale, float(y) / scale] for x, y in box]
            detections.append(Detection(text=text, confidence=float(confidence), box=original_box))

        return OCRResult(
            image_path=image_path,
            image=image,
            detections=detections,
            lines=group_into_lines(detections),
            seconds=seconds,
            enhanced=enhance,
        )
