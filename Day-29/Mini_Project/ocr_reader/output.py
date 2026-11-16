"""
Everything that leaves the program: .txt files, preview images and the CSV summary.
"""

import csv
import textwrap
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

# Fonts are tried in order: Windows first, then Linux/macOS, then Pillow's built-in.
MONO_FONTS = ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Menlo.ttc"]
BOLD_FONTS = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"]

BOX_COLOR = (0, 170, 80)


def _font(candidates, size):
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def to_pil(image_bgr):
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def draw_boxes(pil_image, detections, scale=1.0, width=None):
    """Draws a green outline around every detected piece of text."""
    image = pil_image.copy()
    draw = ImageDraw.Draw(image)
    line_width = width or max(2, round(max(image.size) / 500))
    for d in detections:
        points = [(x * scale, y * scale) for x, y in d.box]
        draw.line(points + [points[0]], fill=BOX_COLOR, width=line_width)
    return image


def save_text(result, folder):
    """Saves the extracted text as <image name>.txt and returns the file path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{result.image_path.stem}.txt"
    path.write_text(result.text + "\n", encoding="utf-8")
    return path


def build_preview(result, height=1100, text_width=900):
    """
    One picture that shows the original image (with text boxes) on the left
    and the extracted text on the right - handy for reports and quick checks.
    """
    original = to_pil(result.image)

    # fit the photo into the left half without making tiny images huge
    scale = min(height / original.height, 900 / original.width, 2.0)
    shown = original.resize((max(1, int(original.width * scale)), max(1, int(original.height * scale))),
                            Image.LANCZOS)
    shown = draw_boxes(shown, result.detections, scale=scale)

    canvas_height = max(shown.height, 700)
    canvas = Image.new("RGB", (shown.width + text_width + 30, canvas_height), "white")
    canvas.paste(shown, (0, (canvas_height - shown.height) // 2))

    draw = ImageDraw.Draw(canvas)
    left = shown.width + 30
    draw.line([(shown.width + 12, 20), (shown.width + 12, canvas_height - 20)], fill=(210, 210, 210), width=2)

    title_font, info_font, text_font = _font(BOLD_FONTS, 30), _font(MONO_FONTS, 18), _font(MONO_FONTS, 20)
    draw.text((left, 20), "Extracted text", font=title_font, fill=(30, 30, 30))
    info = (f"{result.line_count} lines | {result.word_count} words | "
            f"avg confidence {result.average_confidence * 100:.0f}% | {result.seconds:.1f}s"
            f"{' | enhanced' if result.enhanced else ''}")
    draw.text((left, 62), info, font=info_font, fill=(110, 110, 110))

    # wrap long lines so they fit the panel
    char_width = max(1, draw.textlength("M", font=text_font))
    max_chars = int((text_width - 30) // char_width)
    wrapped = []
    for line in result.lines or ["(no text found)"]:
        wrapped += textwrap.wrap(line, max_chars) or [""]

    line_height = 27
    y = 105
    max_lines = (canvas_height - y - 20) // line_height
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines - 1] + [f"... ({len(wrapped) - max_lines + 1} more lines in the .txt file)"]
    for line in wrapped:
        draw.text((left, y), line, font=text_font, fill=(20, 20, 20))
        y += line_height

    return canvas


def save_preview(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{result.image_path.stem}_preview.jpg"
    build_preview(result).save(path, quality=88)
    return path


def write_summary(results, path):
    """One row per image: how much text was found, how confident, how fast."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "width", "height", "text_boxes", "lines", "words",
                         "avg_confidence_%", "seconds", "enhanced", "processed_at"])
        for r in results:
            h, w = r.image.shape[:2]
            writer.writerow([r.image_path.name, w, h, len(r.detections), r.line_count, r.word_count,
                             round(r.average_confidence * 100, 1), round(r.seconds, 2),
                             "yes" if r.enhanced else "no", datetime.now().strftime("%Y-%m-%d %H:%M")])
    return path
