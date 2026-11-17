"""
thresholding.py
---------------
Day 30 - Coding Practice: Image Thresholding & Segmentation (OpenCV)

For every image in images/ the script:
    1. reads it and converts it to grayscale
    2. applies Binary (global, fixed) thresholding
    3. applies Adaptive (local) thresholding
    4. applies Otsu (automatic global) thresholding
    5. compares the three methods: side-by-side figure, histogram with the
       chosen thresholds, and numbers (threshold, foreground %, blobs, time)
    6. performs simple foreground/background segmentation
       (Otsu mask -> morphological cleanup -> cut-out of the foreground)
    7. saves every output image to outputs/<image name>/ for comparison

Run:
    python thresholding.py                      # all images in images/
    python thresholding.py --image photo.jpg    # a single image
    python thresholding.py --threshold 100 --block-size 31 --c 12

Thresholding in one sentence each:
    Binary   - one fixed number for the whole image (pixel > T -> white).
    Adaptive - a different threshold for every pixel, from its neighbourhood.
    Otsu     - one number for the whole image, chosen automatically from the
               histogram so that the two classes are as separated as possible.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                      # write figures to files, no window
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

DEFAULT_THRESHOLD = 127      # the classic "middle grey" starting point
DEFAULT_C = 10               # how much darker than its neighbourhood a pixel must be (adaptive)


# ---------------------------------------------------------------------------
# 1. Reading and grayscale
# ---------------------------------------------------------------------------

def read_image(path):
    """
    Reads an image from disk. np.fromfile + imdecode instead of cv2.imread,
    because cv2.imread silently returns None for paths with non-English
    characters on Windows.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Not a readable image: {path}")
    return image


def to_grayscale(image):
    """BGR -> one channel. Uses the standard luminance weights (0.299 R + 0.587 G + 0.114 B)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def smooth(gray):
    """
    A light 5x5 Gaussian blur removes sensor noise and JPEG speckles, which
    would otherwise become thousands of tiny black/white dots after
    thresholding. All three methods get the SAME smoothed image, so the
    comparison is about the thresholding rule only.
    """
    return cv2.GaussianBlur(gray, (5, 5), 0)


# ---------------------------------------------------------------------------
# 2-4. The three thresholding methods
# ---------------------------------------------------------------------------

def binary_threshold(gray, threshold=DEFAULT_THRESHOLD):
    """Global, fixed threshold: pixel > threshold -> 255 (white), else 0 (black)."""
    _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return result, float(threshold)


def adaptive_block_size(gray):
    """
    The neighbourhood must be bigger than a letter / object stroke, but small
    enough that the lighting is roughly constant inside it. About 1/15 of the
    shorter image side works well; it must be an odd number >= 3.
    """
    size = max(11, min(gray.shape[:2]) // 15)
    return size if size % 2 == 1 else size + 1


def adaptive_threshold(gray, block_size=None, c=DEFAULT_C):
    """
    Local threshold: for every pixel, T = (Gaussian-weighted mean of its
    block_size x block_size neighbourhood) - c. Handles shadows and uneven
    light because each area is compared only with its own surroundings.
    """
    block_size = block_size or adaptive_block_size(gray)
    result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, c)
    return result, block_size


def otsu_threshold(gray):
    """
    Otsu's method tries every threshold 0..255 and keeps the one that gives the
    largest between-class variance (the two groups of pixels are as far apart
    as possible). OpenCV returns the chosen value, so we can report it.
    """
    value, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result, float(value)


def otsu_separability(gray, threshold):
    """
    How well a single threshold splits this image, from 0 to 1
    (between-class variance / total variance). Close to 1 = clearly two groups
    (ideal for any global method). Low = no clean split exists.
    """
    pixels = gray.ravel().astype(np.float64)
    total_var = pixels.var()
    if total_var == 0:
        return 0.0
    dark, bright = pixels[pixels <= threshold], pixels[pixels > threshold]
    if len(dark) == 0 or len(bright) == 0:
        return 0.0
    w0, w1 = len(dark) / len(pixels), len(bright) / len(pixels)
    between = w0 * w1 * (dark.mean() - bright.mean()) ** 2
    return float(between / total_var)


# ---------------------------------------------------------------------------
# 5. Comparison numbers
# ---------------------------------------------------------------------------

def count_blobs(binary, min_area):
    """Counts connected black regions (text, objects, noise) bigger than min_area pixels."""
    dark = cv2.bitwise_not(binary)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    return int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= min_area))


def describe(binary, seconds, min_area):
    total = binary.size
    return {
        "white_%": round(100 * np.count_nonzero(binary) / total, 1),
        "black_%": round(100 * (total - np.count_nonzero(binary)) / total, 1),
        "dark_blobs": count_blobs(binary, min_area),
        "time_ms": round(seconds * 1000, 2),
    }


# ---------------------------------------------------------------------------
# 6. Foreground / background segmentation
# ---------------------------------------------------------------------------

def foreground_from_border(mask):
    """
    After thresholding we still have to decide WHICH colour is the object.
    Simple, robust rule: the background is whatever touches the image border
    most (objects are usually inside the frame, the background surrounds them).
    Returns the mask with the foreground in white.
    """
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    background_is_white = np.count_nonzero(border) > border.size / 2
    return cv2.bitwise_not(mask) if background_is_white else mask


def fill_holes(mask):
    """
    Fills holes inside objects (e.g. a reflection in the middle of a coin).
    A hole is a background region that does NOT touch the image border, so
    it must be enclosed by an object. (A single flood fill from the corner
    is the usual shortcut, but it breaks when the corner itself is foreground.)
    """
    background = cv2.bitwise_not(mask)
    count, labels = cv2.connectedComponents(background, connectivity=4)
    touching_border = set(np.unique(np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])))
    filled = mask.copy()
    for label in range(1, count):
        if label not in touching_border:
            filled[labels == label] = 255
    return filled


def flatten_illumination(gray):
    """
    Removes slow brightness changes (a lamp on one side, a shadow, a vignette)
    before a global threshold. The 'lighting' is estimated with a very large
    Gaussian blur - so large that objects and text disappear into it - and the
    image is divided by it. A dark corner and a bright corner of the same
    paper then end up at the same grey level.
    """
    image = gray.astype(np.float32)
    lighting = cv2.GaussianBlur(image, (0, 0), sigmaX=min(gray.shape) / 4)
    flat = image / (lighting + 1.0) * lighting.mean()
    return np.clip(flat, 0, 255).astype(np.uint8)


def segment_foreground(gray):
    """
    Simple foreground/background segmentation:
        illumination correction -> Otsu threshold -> pick foreground by the
        border rule -> opening (removes specks) -> closing (joins broken
        edges) -> fill holes -> drop specks smaller than 0.005% of the image.
    Returns (mask, number_of_objects).
    """
    mask, _ = otsu_threshold(smooth(flatten_illumination(gray)))
    mask = foreground_from_border(mask)

    # small fixed opening, so thin strokes (text) survive; closing scales with the image
    speck_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k = max(3, min(gray.shape) // 150) | 1
    join_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, speck_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, join_kernel)
    mask = fill_holes(mask)

    min_area = max(15, int(mask.size * 0.00005))       # ~ a small printed letter
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    objects = 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label] = 255
            objects += 1
    return clean, objects


def cut_out(image, mask):
    """Foreground on a transparent background (PNG with alpha) and background with the object removed."""
    foreground = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    foreground[:, :, 3] = mask
    background = image.copy()
    background[mask > 0] = (255, 255, 255)
    return foreground, background


def overlay(image, mask):
    """Original image with the foreground tinted green and outlined, for a quick visual check."""
    result = image.copy()
    tint = np.zeros_like(image)
    tint[:] = (60, 200, 60)
    result[mask > 0] = cv2.addWeighted(image, 0.55, tint, 0.45, 0)[mask > 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 0, 255), max(1, min(image.shape[:2]) // 250))
    return result


# ---------------------------------------------------------------------------
# 7. Saving and comparison figures
# ---------------------------------------------------------------------------

def save(path, image):
    """cv2.imwrite cannot handle non-English paths on Windows; imencode + tofile can."""
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise IOError(f"Could not write {path}")
    buffer.tofile(str(path))


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


def comparison_figure(name, image, gray, results, save_path):
    """
    One figure per image:
        top row    : original | grayscale | histogram with both global thresholds
        bottom row : binary | adaptive | otsu
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Thresholding comparison - {name}", fontsize=15, fontweight="bold")

    axes[0, 0].imshow(rgb(image))
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Grayscale")

    hist = axes[0, 2]
    hist.hist(gray.ravel(), bins=256, range=(0, 256), color="#607d8b")
    hist.axvline(results["binary"]["threshold"], color="#e53935", linewidth=2,
                 label=f"Binary T = {results['binary']['threshold']:.0f}")
    hist.axvline(results["otsu"]["threshold"], color="#2e7d32", linewidth=2, linestyle="--",
                 label=f"Otsu T = {results['otsu']['threshold']:.0f}")
    hist.set_title("Grayscale histogram")
    hist.set_xlabel("Pixel value (0 = black, 255 = white)")
    hist.set_ylabel("Pixel count")
    hist.set_xlim(0, 255)
    hist.legend()

    titles = {
        "binary": f"Binary (T = {results['binary']['threshold']:.0f})",
        "adaptive": f"Adaptive Gaussian (block = {results['adaptive']['threshold']:.0f}, C = {DEFAULT_C})",
        "otsu": f"Otsu (auto T = {results['otsu']['threshold']:.0f})",
    }
    for ax, method in zip(axes[1], ("binary", "adaptive", "otsu")):
        ax.imshow(results[method]["image"], cmap="gray", vmin=0, vmax=255)
        ax.set_title(titles[method])

    for ax in list(axes[0, :2]) + list(axes[1]):
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close(fig)


def segmentation_figure(name, image, mask, foreground, objects, save_path):
    """Original | mask | foreground cut-out (on a checkerboard so transparency is visible) | overlay."""
    h, w = mask.shape
    checker = (((np.indices((h, w)) // 16).sum(axis=0) % 2) * 40 + 200).astype(np.uint8)
    checker = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    alpha = (foreground[:, :, 3:4] / 255.0)
    on_checker = (foreground[:, :, :3] * alpha + checker * (1 - alpha)).astype(np.uint8)

    panels = [("Original", rgb(image), None), ("Foreground mask", mask, "gray"),
              ("Foreground only", rgb(on_checker), None), (f"Overlay ({objects} objects)", rgb(overlay(image, mask)), None)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"Foreground / background segmentation - {name}", fontsize=14, fontweight="bold")
    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close(fig)


def overview_figure(all_results, save_path):
    """All images x all methods in one grid - the quickest way to compare everything."""
    rows = len(all_results)
    fig, axes = plt.subplots(rows, 5, figsize=(20, 3.8 * rows), squeeze=False)
    headers = ["Grayscale", "Binary (T=127)", "Adaptive", "Otsu", "Segmentation"]
    for r, item in enumerate(all_results):
        panels = [item["gray"], item["binary"], item["adaptive"], item["otsu"], rgb(item["overlay"])]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None, vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(headers[c], fontsize=14, fontweight="bold")
            if c == 0:
                ax.set_ylabel(item["name"], fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_image(path, threshold, block_size, c):
    name = path.stem
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    image = read_image(path)
    gray = to_grayscale(image)
    prepared = smooth(gray)
    min_blob = max(5, int(gray.size * 0.00002))

    # run the three methods on the same prepared image, timing each one
    results = {}
    for method, func in (("binary", lambda g: binary_threshold(g, threshold)),
                         ("adaptive", lambda g: adaptive_threshold(g, block_size, c)),
                         ("otsu", otsu_threshold)):
        start = time.perf_counter()
        output, value = func(prepared)
        seconds = time.perf_counter() - start
        results[method] = {"image": output, "threshold": value, **describe(output, seconds, min_blob)}

    mask, objects = segment_foreground(gray)
    foreground, background = cut_out(image, mask)

    save(out_dir / "01_grayscale.png", gray)
    save(out_dir / "02_binary.png", results["binary"]["image"])
    save(out_dir / "03_adaptive.png", results["adaptive"]["image"])
    save(out_dir / "04_otsu.png", results["otsu"]["image"])
    save(out_dir / "05_segmentation_mask.png", mask)
    save(out_dir / "06_foreground.png", foreground)          # transparent background
    save(out_dir / "07_background.png", background)
    save(out_dir / "08_segmentation_overlay.png", overlay(image, mask))
    comparison_figure(name, image, gray, results, out_dir / "comparison.png")
    segmentation_figure(name, image, mask, foreground, objects, out_dir / "segmentation.png")

    separability = otsu_separability(prepared, results["otsu"]["threshold"])
    print(f"\n{path.name}  ({gray.shape[1]}x{gray.shape[0]})  "
          f"Otsu T = {results['otsu']['threshold']:.0f}, separability = {separability:.2f}")
    print(f"  {'method':<9} {'threshold':>9} {'white %':>8} {'dark blobs':>11} {'time ms':>8}")
    for method in ("binary", "adaptive", "otsu"):
        r = results[method]
        label = f"block {r['threshold']:.0f}" if method == "adaptive" else f"{r['threshold']:.0f}"
        print(f"  {method:<9} {label:>9} {r['white_%']:>8} {r['dark_blobs']:>11} {r['time_ms']:>8}")
    print(f"  segmentation: {objects} foreground object(s), "
          f"{100 * np.count_nonzero(mask) / mask.size:.1f}% of the image")

    rows = [{"image": path.name, "method": m, "threshold": results[m]["threshold"],
             "white_%": results[m]["white_%"], "black_%": results[m]["black_%"],
             "dark_blobs": results[m]["dark_blobs"], "time_ms": results[m]["time_ms"],
             "otsu_separability": round(separability, 3)} for m in ("binary", "adaptive", "otsu")]
    summary = {"name": name, "gray": gray, "binary": results["binary"]["image"],
               "adaptive": results["adaptive"]["image"], "otsu": results["otsu"]["image"],
               "overlay": overlay(image, mask), "objects": objects}
    return rows, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Binary vs Adaptive vs Otsu thresholding + segmentation")
    parser.add_argument("--image", type=Path, help="process a single image instead of the images/ folder")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="binary threshold (default 127)")
    parser.add_argument("--block-size", type=int, default=None,
                        help="adaptive neighbourhood size, odd number (default: automatic)")
    parser.add_argument("--c", type=int, default=DEFAULT_C, help="adaptive constant C (default 10)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.block_size is not None and (args.block_size < 3 or args.block_size % 2 == 0):
        sys.exit("--block-size must be an odd number >= 3")

    if args.image:
        if not args.image.is_file():
            sys.exit(f"Image not found: {args.image}")
        paths = [args.image]
    else:
        paths = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_TYPES)
        if not paths:
            sys.exit(f"No images found in {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    all_rows, summaries = [], []
    for path in paths:
        try:
            rows, summary = process_image(path, args.threshold, args.block_size, args.c)
        except ValueError as error:
            print(f"Skipping {path.name}: {error}")
            continue
        all_rows += rows
        summaries.append(summary)

    if not all_rows:
        sys.exit("Nothing was processed.")

    with open(OUTPUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    overview_figure(summaries, OUTPUT_DIR / "all_methods_overview.png")

    print(f"\nSaved everything to {OUTPUT_DIR}")
    print("  outputs/<image>/comparison.png     - the three methods side by side + histogram")
    print("  outputs/<image>/segmentation.png   - mask, foreground cut-out, overlay")
    print("  outputs/all_methods_overview.png   - every image x every method")
    print("  outputs/summary.csv                - numbers for the comparison")


if __name__ == "__main__":
    main()
