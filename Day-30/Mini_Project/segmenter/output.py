"""
Turning masks into things people can look at and use: cut-outs, overlays,
the comparison sheet of all methods, and the saved best result.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from .io_utils import save_image

GREEN, RED, GOLD, WHITE, DARK = (60, 200, 60), (40, 40, 230), (0, 190, 255), (255, 255, 255), (35, 35, 35)


def cutout(image, mask):
    """Foreground on a transparent background (BGRA, for PNG)."""
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    result[mask == 0] = 0       # hidden pixels set to zero -> PNG files several times smaller
    return result


def on_checkerboard(rgba, square=16):
    """Shows transparency as a grey checkerboard (for displaying cut-outs)."""
    h, w = rgba.shape[:2]
    board = (((np.indices((h, w)) // square).sum(axis=0) % 2) * 45 + 195).astype(np.uint8)
    board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return (rgba[:, :, :3] * alpha + board * (1 - alpha)).astype(np.uint8)


def overlay(image, mask, document=None):
    """Background dimmed, foreground outlined in red (and page corners if a document was found)."""
    dimmed = (image * 0.35).astype(np.uint8)
    result = np.where(mask[:, :, None] > 0, image, dimmed)
    thickness = max(2, min(image.shape[:2]) // 300)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, RED, thickness)
    if document is not None:
        corners = document.corners.astype(np.int32)
        cv2.polylines(result, [corners], True, GOLD, thickness)
        for x, y in corners:
            cv2.circle(result, (int(x), int(y)), thickness * 4, GOLD, -1)
    return result


def _tile(image, size):
    """Fits an image into a size x size square tile (letterboxed)."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    scale = size / max(h, w)
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    tile = np.full((size, size, 3), 245, np.uint8)
    y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    tile[y:y + small.shape[0], x:x + small.shape[1]] = small
    return tile


def _caption(tile, lines, highlight=False):
    bar = np.full((52, tile.shape[1], 3), (0, 140, 30) if highlight else DARK, np.uint8)
    cv2.putText(bar, lines[0], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(bar, lines[1], (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return np.vstack([tile, bar])


def comparison_sheet(name, seg, tile=300, columns=4):
    """
    One image with: the original, then every method's result (dimmed background
    + red outline), labelled with its score. The best method is shown in green.
    """
    tiles = [_caption(_tile(seg.image, tile), [name[:34], "original"])]
    for rank, (key, r) in enumerate(seg.results.items(), start=1):
        label = f"{'BEST  ' if key == seg.best_key else f'#{rank}  '}{r.label}"
        info = f"score {r.score.total:.0f}  |  {r.score.coverage:.0f}% fg  |  {r.objects} obj"
        tiles.append(_caption(_tile(overlay(seg.image, r.mask), tile), [label, info], key == seg.best_key))

    blank = np.full_like(tiles[0], 255)
    while len(tiles) % columns:
        tiles.append(blank)
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


def save_best(seg, folder, name, key=None):
    """
    Saves the best segmentation - or the method given by `key` if you want to
    override the automatic choice - plus the document scan if a page was found.
    Returns a dict of the files written.
    """
    from .document import find_document

    folder = Path(folder)
    chosen = seg.results[key] if key else seg.best
    document = seg.document if chosen.key == seg.best_key else find_document(seg.image, chosen.mask)
    saved = {
        "cutout": save_image(folder / f"{name}_best_cutout.png", cutout(seg.image, chosen.mask)),
        "mask": save_image(folder / f"{name}_best_mask.png", chosen.mask),
        "overlay": save_image(folder / f"{name}_best_overlay.jpg", overlay(seg.image, chosen.mask, document)),
    }
    if document is not None:
        saved["scan"] = save_image(folder / f"{name}_document_scan.jpg", document.color)
        saved["scan_bw"] = save_image(folder / f"{name}_document_scan_bw.png", document.clean)
    return saved


def save_all_methods(seg, folder):
    """Saves every method's mask and cut-out (optional, for detailed comparison)."""
    folder = Path(folder) / "methods"
    for key, r in seg.results.items():
        save_image(folder / f"{key}_mask.png", r.mask)
        save_image(folder / f"{key}_cutout.png", cutout(seg.image, r.mask))
    return folder


def write_summary(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
