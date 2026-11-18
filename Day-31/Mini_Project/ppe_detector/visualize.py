"""Drawing boxes and labels on images."""

import cv2

from .settings import CLASS_COLORS

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(canvas, text, x, y_baseline, color, font_scale, weight, placed):
    (tw, th), base = cv2.getTextSize(text, FONT, font_scale, weight)
    x = max(0, min(x, canvas.shape[1] - tw - 4))

    def overlaps(y):
        top, bottom = y - th - 3, y + base
        return any(x < r[2] and x + tw + 4 > r[0] and top < r[3] and bottom > r[1] for r in placed)

    # two boots side by side would print their labels on top of each other:
    # move a label up one line at a time until it is free (a few tries at most)
    for _ in range(4):
        if not overlaps(y_baseline) or y_baseline - 2 * th - 6 < 0:
            break
        y_baseline -= th + base + 4
    placed.append((x, y_baseline - th - 3, x + tw + 4, y_baseline + base))
    cv2.rectangle(canvas, (x, y_baseline - th - 3), (x + tw + 4, y_baseline + base), color, -1)
    cv2.putText(canvas, text, (x + 2, y_baseline - 1), FONT, font_scale, (0, 0, 0), weight, cv2.LINE_AA)


def draw_detections(image_bgr, detections, show_confidence=True):
    """
    Returns a copy of the image with one coloured box per detection and a
    label such as 'helmet 0.93'. Line width and font size follow the image
    size, so labels stay readable on a 640 px photo and on a 4K frame.

    A helmet sits at the top of its worker's box, so both labels would land
    on the same spot. Person labels therefore go under the box, all other
    labels above theirs.
    """
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    thickness = max(2, round(min(h, w) / 320))
    font_scale = max(0.45, min(h, w) / 1100)
    weight = max(1, thickness - 1)
    text_h = cv2.getTextSize("Ag", FONT, font_scale, weight)[0][1]

    # boxes first, labels afterwards, so no box line runs through a label
    for det in detections:
        color = CLASS_COLORS.get(det.class_id, (255, 255, 255))
        cv2.rectangle(canvas, (det.x1, det.y1), (det.x2, det.y2), color, thickness, cv2.LINE_AA)

    placed = []
    for det in sorted(detections, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1), reverse=True):
        color = CLASS_COLORS.get(det.class_id, (255, 255, 255))
        text = f"{det.name} {det.confidence:.2f}" if show_confidence else det.name
        if det.name == "person":
            below = det.y2 + text_h + 6
            y = below if below < h else det.y2 - 6              # under the box, or just inside it
        else:
            above = det.y1 - 6
            y = above if above - text_h > 0 else det.y1 + text_h + 6   # above the box, or just inside it
        _label(canvas, text, det.x1, y, color, font_scale, weight, placed)
    return canvas
