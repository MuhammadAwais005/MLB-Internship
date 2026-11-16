"""
easyocr_task.py
---------------
Internship Task: OCR with EasyOCR

What this script does:
    1. Loads EasyOCR (English model).
    2. Reads text from every image in the images/ folder.
    3. Runs each image through 4 versions:
           original   -> the photo as it is
           grayscale  -> converted to gray only
           enhanced   -> gray + upscale + shadow removal + denoise + contrast (CLAHE)
           binarized  -> enhanced image turned into pure black & white
    4. Compares the results: confidence, words found, time taken, and accuracy
       (accuracy only when a ground truth file exists in images/ground_truth/).
    5. Saves everything to the outputs/ folder: CSV table, text files,
       preprocessed images, images with boxes drawn, a chart and a short report.

Run:
    python easyocr_task.py

First run downloads the EasyOCR models (~100 MB), so it needs internet once.
"""

import glob
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from urllib.error import ContentTooShortError

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        "This project is verified on Python 3.11.x.\n"
        "Create or activate a Python 3.11 environment, then run:\n"
        "  py -3.11 -m pip install -r requirements.txt\n"
        "  py -3.11 easyocr_task.py"
    )

try:
    import cv2
    import easyocr
    import matplotlib
    matplotlib.use("Agg")          # save charts to file, no window needed
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing required Python dependencies. Install them with:\n"
        "  py -3.11 -m pip install -r requirements.txt\n"
        f"Original error: {exc}"
    ) from exc

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
TRUTH_DIR = os.path.join(IMAGE_DIR, "ground_truth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

LANGUAGES = ["en"]
USE_GPU = torch.cuda.is_available()      # uses GPU automatically if you have one
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# Images smaller than this (longest side, in pixels) get upscaled.
# EasyOCR struggles when letters are only a few pixels tall.
MIN_IMAGE_SIZE = 1000


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def to_grayscale(image):
    if len(image.shape) == 2:          # already gray
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def upscale_if_small(gray):
    height, width = gray.shape
    longest = max(height, width)
    if longest >= MIN_IMAGE_SIZE:
        return gray
    scale = min(4.0, MIN_IMAGE_SIZE / longest)       # never more than 4x
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def has_light_text(gray):
    """
    True if the text is lighter than its background (white text on a dark sign).

    Idea: compare every pixel with the local background around it (a big median
    blur). Text pixels stand out from the background; we simply count whether
    more of them are much brighter or much darker than their surroundings.
    This works even when one side of the photo is in shadow.
    """
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    background = cv2.medianBlur(smooth, 51)
    diff = smooth.astype(np.int16) - background.astype(np.int16)
    brighter = np.count_nonzero(diff > 25)
    darker = np.count_nonzero(diff < -25)
    return brighter > darker


def remove_shadow(gray):
    """
    Evens out the lighting. We estimate the 'empty paper' brightness by
    removing the text (dilate + big median blur), then divide the image by it.
    A shadowed area and a bright area both end up as the same white paper.
    """
    kernel = np.ones((15, 15), np.uint8)
    background = cv2.dilate(gray, kernel)
    background = cv2.medianBlur(background, 41)
    return cv2.divide(gray, background, scale=255)


def enhance_image(image):
    """Full cleanup: grayscale -> upscale -> fix polarity -> shadow removal -> denoise -> contrast."""
    gray = to_grayscale(image)
    gray = upscale_if_small(gray)

    # Shadow removal expects dark text on light paper, so flip light-on-dark signs.
    # EasyOCR itself does not care which way round the text is.
    if has_light_text(gray):
        gray = cv2.bitwise_not(gray)

    gray = remove_shadow(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)

    # CLAHE = contrast boost that works on small tiles, so a faded corner
    # gets boosted without blowing out the parts that were already fine.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize_image(image):
    """Enhanced image converted to pure black and white (adaptive threshold)."""
    enhanced = enhance_image(image)
    return cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)


# The order here is the order you will see in the results
METHODS = {
    "original": lambda image: image,
    "grayscale": to_grayscale,
    "enhanced": enhance_image,
    "binarized": binarize_image,
}


# ---------------------------------------------------------------------------
# Comparing OCR output with the correct text
# ---------------------------------------------------------------------------

def normalize(text):
    """Lowercase, remove accents and punctuation. 'Musée,' and 'musee' count as equal."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return text.split()


def word_accuracy(predicted, truth):
    """Percentage of the correct words that OCR found (order does not matter)."""
    truth_words = Counter(normalize(truth))
    found_words = Counter(normalize(predicted))
    if not truth_words:
        return None
    matched = sum((truth_words & found_words).values())
    return 100 * matched / sum(truth_words.values())


def edit_distance(a, b):
    """Classic Levenshtein distance: number of insert/delete/replace operations."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def character_accuracy(predicted, truth):
    """100 - Character Error Rate. Catches small mistakes like 'Tornatoes' vs 'Tomatoes'."""
    truth_text = " ".join(normalize(truth))
    predicted_text = " ".join(normalize(predicted))
    if not truth_text:
        return None
    cer = edit_distance(predicted_text, truth_text) / len(truth_text)
    return max(0.0, 100 * (1 - cer))


def load_ground_truth(image_name):
    path = os.path.join(TRUTH_DIR, image_name + ".txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# OCR + saving results
# ---------------------------------------------------------------------------

def run_ocr(reader, image):
    """Returns the raw EasyOCR results and how long it took."""
    start = time.time()
    results = reader.readtext(image)        # list of (box, text, confidence)
    return results, time.time() - start


def results_to_lines(results):
    """
    EasyOCR does not return boxes in reading order. On a receipt you often get
    all the prices first and then the item names. Here we group boxes that sit
    on the same row into one line, then sort top-to-bottom and left-to-right.
    """
    if not results:
        return []

    boxes = []
    for box, text, _ in results:
        ys = [point[1] for point in box]
        boxes.append({"text": text, "x": min(p[0] for p in box),
                      "top": min(ys), "bottom": max(ys), "center": sum(ys) / len(ys)})

    # two boxes are on the same line if their centers are closer than half a text height
    typical_height = np.median([b["bottom"] - b["top"] for b in boxes])
    boxes.sort(key=lambda b: b["center"])

    lines, current = [], [boxes[0]]
    for b in boxes[1:]:
        if abs(b["center"] - current[-1]["center"]) < typical_height * 0.5:
            current.append(b)
        else:
            lines.append(current)
            current = [b]
    lines.append(current)

    return [" ".join(b["text"] for b in sorted(line, key=lambda b: b["x"])) for line in lines]


def draw_boxes(image, results, save_path):
    """Draws a green box and the detected text on top of the image."""
    canvas = image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    thickness = max(2, canvas.shape[1] // 600)
    font_scale = 0.45 * thickness
    for box, text, confidence in results:
        points = np.array(box, dtype=np.int32)
        cv2.polylines(canvas, [points], isClosed=True, color=(0, 200, 0), thickness=thickness)

        # label sits just above the box, on a white strip so it stays readable
        label = f"{text} ({confidence:.2f})"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        x, y = int(points[0][0]), max(label_h + 4, int(points[0][1]) - 4)
        cv2.rectangle(canvas, (x, y - label_h - 4), (x + label_w + 4, y + 2), (255, 255, 255), -1)
        cv2.putText(canvas, label, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 220), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, canvas)


def make_output_folders():
    for sub in ("preprocessed", "annotated", "text"):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def process_image(reader, image_path):
    """Runs every preprocessing method on one image and returns one row per method."""
    file_name = os.path.basename(image_path)
    image_name = os.path.splitext(file_name)[0]
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ! could not read {file_name}, skipping")
        return []

    truth = load_ground_truth(image_name)
    rows = []
    text_report = [f"IMAGE: {file_name}", "=" * 60]

    for method, prepare in METHODS.items():
        processed = prepare(image)
        results, seconds = run_ocr(reader, processed)

        lines = results_to_lines(results)
        confidences = [conf for _, _, conf in results]
        full_text = "\n".join(lines)

        rows.append({
            "image": file_name,
            "method": method,
            "text_boxes": len(results),
            "words": len(full_text.split()),
            "avg_confidence": round(float(np.mean(confidences)) * 100, 1) if confidences else 0.0,
            "word_accuracy": None if truth is None else round(word_accuracy(full_text, truth), 1),
            "char_accuracy": None if truth is None else round(character_accuracy(full_text, truth), 1),
            "seconds": round(seconds, 2),
            "text": " | ".join(lines),
        })

        # keep the preprocessed image so you can see what EasyOCR actually got
        if method != "original":
            cv2.imwrite(os.path.join(OUTPUT_DIR, "preprocessed", f"{image_name}_{method}.png"), processed)

        # boxes drawn on original vs enhanced is enough to compare visually
        if method in ("original", "enhanced"):
            draw_boxes(processed, results, os.path.join(OUTPUT_DIR, "annotated", f"{image_name}_{method}.jpg"))

        text_report += [f"\n--- {method.upper()} ---", full_text or "(no text found)"]

        row = rows[-1]
        accuracy = f"{row['word_accuracy']:5.1f}%" if truth else "  n/a "
        print(f"    {method:<10} boxes={row['text_boxes']:<3} conf={row['avg_confidence']:5.1f}%  "
              f"word accuracy={accuracy}  time={seconds:.1f}s")

    if truth:
        text_report += ["\n--- CORRECT TEXT (ground truth) ---", truth]
    with open(os.path.join(OUTPUT_DIR, "text", f"{image_name}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text_report))

    return rows


# ---------------------------------------------------------------------------
# Comparison: chart + report
# ---------------------------------------------------------------------------

def save_chart(df):
    """Grouped bar chart: word accuracy per image for each method."""
    scored = df.dropna(subset=["word_accuracy"])
    if scored.empty:
        return
    table = scored.pivot(index="image", columns="method", values="word_accuracy")[list(METHODS)]
    table.index = [name.rsplit(".", 1)[0] for name in table.index]

    colors = ["#9e9e9e", "#5c6bc0", "#2e7d32", "#ef6c00"]
    ax = table.plot(kind="bar", figsize=(15, 6), width=0.8, color=colors)
    ax.set_title("EasyOCR word accuracy per image and preprocessing method")
    ax.set_ylabel("Word accuracy (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method", loc="lower left")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_chart.png"), dpi=130)
    plt.close()


def save_report(df):
    """Writes a short markdown report with the comparison tables."""
    summary = (df.groupby("method", sort=False)
                 .agg(avg_word_accuracy=("word_accuracy", "mean"),
                      avg_char_accuracy=("char_accuracy", "mean"),
                      avg_confidence=("avg_confidence", "mean"),
                      avg_seconds=("seconds", "mean"))
                 .round(1))

    accuracy_table = df.pivot(index="image", columns="method", values="word_accuracy")[list(METHODS)]
    char_table = df.pivot(index="image", columns="method", values="char_accuracy")[list(METHODS)]
    best = []
    for image, row in accuracy_table.iterrows():
        if row.isna().all():
            continue
        top = row.max()
        winners = [m for m in METHODS if row[m] == top]
        change = top - row["original"]
        best.append((image, ", ".join(winners), f"{row['original']:.1f}", f"{top:.1f}", f"{change:+.1f}"))

    lines = [
        "# EasyOCR Results Report",
        "",
        f"- Images tested: {df['image'].nunique()}",
        f"- Methods compared: {', '.join(METHODS)}",
        f"- Device: {'GPU' if USE_GPU else 'CPU'}",
        "",
        "## Average per method",
        "",
        summary.to_markdown(),
        "",
        "## Word accuracy (%) per image",
        "",
        "Share of the correct words that EasyOCR found. Main metric, because it",
        "does not depend on the order in which boxes come back.",
        "",
        accuracy_table.round(1).to_markdown(),
        "",
        "## Character accuracy (%) per image",
        "",
        "100 minus Character Error Rate. Stricter: it also catches single wrong",
        "letters, but it drops when lines are rebuilt in a different order",
        "(e.g. the tilted receipt), so read it together with word accuracy.",
        "",
        char_table.round(1).to_markdown(),
        "",
        "## Best method per image (word accuracy)",
        "",
        "| Image | Best method | Original | Best | Gain |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in best]
    lines += ["", "Full extracted text for every image and method is in `outputs/text/`.", ""]

    with open(os.path.join(OUTPUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_reader_with_retry(languages, use_gpu, max_attempts=3):
    """EasyOCR sometimes fails on a flaky network during first model download.
    Retry a few times and clear the partial temp archive between attempts.
    """
    model_dir = os.path.join(os.path.expanduser("~"), ".EasyOCR", "model")
    for attempt in range(1, max_attempts + 1):
        try:
            return easyocr.Reader(languages, gpu=use_gpu, verbose=False)
        except (ContentTooShortError, OSError, TimeoutError, ConnectionError) as exc:
            for partial in glob.glob(os.path.join(model_dir, "temp.zip")):
                try:
                    os.remove(partial)
                except OSError:
                    pass
            if attempt == max_attempts:
                raise SystemExit(
                    "EasyOCR model download failed repeatedly. Check your internet connection "
                    "and retry.\n"
                    f"Last error: {exc}"
                ) from exc
            print(f"  Download interrupted; retrying EasyOCR model load ({attempt}/{max_attempts})...")
            time.sleep(5)


def main():
    image_files = sorted(
        os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(IMAGE_TYPES)
    )
    if not image_files:
        print(f"No images found in {IMAGE_DIR}. Run make_test_images.py or add your own photos.")
        return

    make_output_folders()
    print(f"Loading EasyOCR ({', '.join(LANGUAGES)}) on {'GPU' if USE_GPU else 'CPU'}...")
    reader = create_reader_with_retry(LANGUAGES, USE_GPU)

    all_rows = []
    for number, path in enumerate(image_files, start=1):
        print(f"\n[{number}/{len(image_files)}] {os.path.basename(path)}")
        all_rows += process_image(reader, path)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "ocr_results.csv"), index=False, encoding="utf-8")
    save_chart(df)
    summary = save_report(df)

    print("\n" + "=" * 70)
    print("AVERAGE RESULTS PER METHOD")
    print("=" * 70)
    print(summary.to_string())
    print(f"\nAll results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
