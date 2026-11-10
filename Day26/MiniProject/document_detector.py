"""Detect and draw document boundaries using OpenCV (no model training).

Run all supplied images: python document_detector.py
Run one image: python document_detector.py --image "my_document.jpg"
"""

from pathlib import Path
import argparse
import csv
import json
import time

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def save_image(path, image):
    """imencode + tofile supports Windows paths containing non-ASCII names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Cannot encode output: {path}")
    encoded.tofile(str(path))


def load_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv.imdecode(data, cv.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"Image is empty, corrupt or unsupported: {path}")
    return image


def remove_small_components(binary, min_pixels=25):
    """Remove isolated edge noise without eroding away one-pixel contours."""
    count, labels, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (stats[1:, cv.CC_STAT_AREA] >= min_pixels).astype(np.uint8) * 255
    return keep[labels]


def choose_boundary(contours, shape, min_area_ratio=0.12, selection="quadrilateral"):
    """Largest-area first. Optional quadrilateral checks reject text/noise.

    'largest' draws the actual largest contour, as in the basic assignment.
    'quadrilateral' selects the largest plausible convex four-corner document.
    Neither mode can prove that a rectangle is a document.
    """
    h, w = shape
    image_area = h * w
    ranked = sorted(contours, key=cv.contourArea, reverse=True)
    if selection == "largest":
        if not ranked or cv.contourArea(ranked[0]) < min_area_ratio * image_area:
            return None, None, "No sufficiently large contour"
        return ranked[0], 1, "Largest contour; verify visually"

    for rank, contour in enumerate(ranked, start=1):
        area = cv.contourArea(contour)
        if area < min_area_ratio * image_area:
            break
        if area > 0.98 * image_area:
            continue
        hull = cv.convexHull(contour)
        # Very concave blobs are often joined text or background clutter.
        if area / max(cv.contourArea(hull), 1) < 0.8:
            continue
        perimeter = cv.arcLength(contour, True)
        for epsilon_fraction in (0.02, 0.03, 0.015):
            polygon = cv.approxPolyDP(contour, epsilon_fraction * perimeter, True)
            if len(polygon) != 4 or not cv.isContourConvex(polygon):
                continue
            points = polygon.reshape(4, 2)
            sides = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
            if sides.min() < 0.07 * min(h, w):
                continue
            if cv.contourArea(polygon) < min_area_ratio * image_area:
                continue
            return polygon, rank, "Largest plausible four-corner contour"
    return None, None, "No sufficiently large, closed four-corner boundary"


def detect_document(image, low=40, high=120, kernel_size=5,
                    min_area_ratio=0.12, selection="quadrilateral", max_size=1000, retry=True):
    """Return full-resolution overlay, status metadata and processing stages."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a nonempty BGR color image")
    if image.size == 0 or min(image.shape[:2]) < 8:
        raise ValueError("Image is too small")
    if not np.isfinite(low) or not np.isfinite(high) or not 0 <= low < high:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer")
    if not 0 < min_area_ratio < 1:
        raise ValueError("Minimum area ratio must lie between 0 and 1")
    if selection not in {"largest", "quadrilateral"}:
        raise ValueError("Selection must be largest or quadrilateral")
    if max_size < 32:
        raise ValueError("Maximum processing size must be at least 32")

    original_h, original_w = image.shape[:2]
    scale = min(1.0, max_size / max(original_h, original_w))
    work = cv.resize(image, (max(8, round(original_w * scale)),
                             max(8, round(original_h * scale))), interpolation=cv.INTER_AREA)

    # 1. Convert the loaded document image to grayscale.
    gray = cv.cvtColor(work, cv.COLOR_BGR2GRAY)
    # 2. Reduce small intensity fluctuations with Gaussian blur.
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    # 3. Canny: two gradient thresholds, with explicit preprocessing above.
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    threshold_pairs = [(low, high)]
    if retry and selection == "quadrilateral":
        # One slightly lower-threshold attempt can recover a faint outer edge
        # instead of selecting a stronger internal rectangle. Keep the larger
        # candidate. The strict largest-contour mode remains a single pass.
        threshold_pairs.append((low * 0.75, high * 0.75))
    attempts = []
    best = None
    for attempt_low, attempt_high in threshold_pairs:
        edges = cv.Canny(blurred, attempt_low, attempt_high, apertureSize=3, L2gradient=True)
        # 4. Closing reconnects short gaps in the document outline.
        closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=1)
        # Opening can erase thin edges; filter small components instead.
        cleaned = remove_small_components(closed, min_pixels=25)
        # 5. Sort contours by area and select the largest document candidate.
        contours, _ = cv.findContours(cleaned, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        boundary, rank, reason = choose_boundary(contours, gray.shape, min_area_ratio, selection)
        area = cv.contourArea(boundary) if boundary is not None else 0
        attempts.append({"thresholds": [attempt_low, attempt_high], "candidate_area": round(area, 1)})
        if best is None or area > best[0]:
            best = (area, boundary, rank, reason, edges, closed, cleaned,
                    len(contours), [attempt_low, attempt_high])
    _, boundary, rank, reason, edges, closed, cleaned, contour_count, chosen_thresholds = best

    overlay = image.copy()
    metadata = {
        "status": "no_boundary", "reason": reason, "selection": selection,
        "contour_count": contour_count, "contour_rank": rank,
        "area_ratio": None, "boundary_points": None,
        "original_size": [original_w, original_h],
        "processing_size": [work.shape[1], work.shape[0]],
        "canny_thresholds": chosen_thresholds, "morphology_kernel": kernel_size,
        "attempts": attempts,
    }
    if boundary is not None:
        # Use separate actual x/y scales to undo integer resize rounding.
        factor = np.array([original_w / work.shape[1], original_h / work.shape[0]])
        full_boundary = np.rint(boundary.reshape(-1, 2) * factor).astype(np.int32)
        full_boundary[:, 0] = np.clip(full_boundary[:, 0], 0, original_w - 1)
        full_boundary[:, 1] = np.clip(full_boundary[:, 1], 0, original_h - 1)
        # 6. Draw on a copy of the original, full-resolution image.
        thickness = max(2, round(max(original_w, original_h) / 350))
        cv.polylines(overlay, [full_boundary], True, (0, 255, 0), thickness, cv.LINE_AA)
        metadata.update(status="boundary_candidate", boundary_points=full_boundary.tolist(),
                        area_ratio=round(float(cv.contourArea(boundary) / gray.size), 4))
    stages = {"01_original_resized": work, "02_gray": gray, "03_gaussian_blur": blurred,
              "04_canny_edges": edges, "05_morphological_closing": closed,
              "06_cleaned_edges": cleaned}
    return overlay, metadata, stages


def process_image(path, output_dir, **options):
    started = time.perf_counter()
    image = load_image(path)
    overlay, metadata, stages = detect_document(image, **options)
    output_dir = Path(output_dir)
    # 7. Save the final image even when no boundary was found; status.json
    # then explains the failure. No invented boundary is drawn in that case.
    save_image(output_dir / "document_boundary.png", overlay)
    for name, stage in stages.items():
        save_image(output_dir / f"{name}.png", stage)
    metadata["file"] = Path(path).name
    metadata["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "status.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Process one image")
    source.add_argument("--input-dir", type=Path, help="Process every supported image in this folder")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--low", type=float, default=40)
    parser.add_argument("--high", type=float, default=120)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--min-area", type=float, default=0.12)
    parser.add_argument("--max-size", type=int, default=1000)
    parser.add_argument("--selection", choices=["quadrilateral", "largest"], default="quadrilateral")
    parser.add_argument("--single-pass", action="store_true", help="Disable the lower-threshold comparison attempt")
    args = parser.parse_args()
    folder = args.input_dir if args.input_dir is not None else ROOT / "dataset" / "images"
    if args.image is not None:
        files = [args.image]
    elif folder.is_dir():
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        parser.exit(1, f"Input folder does not exist: {folder}\n")
    if not files:
        parser.exit(1, "No supported input images found\n")
    rows = []
    failed = False
    for path in files:
        # Include the extension in folder names to avoid foo.jpg/foo.png collisions.
        destination = args.output / path.name.replace(".", "_")
        try:
            metadata = process_image(path, destination, low=args.low, high=args.high,
                                     kernel_size=args.kernel, min_area_ratio=args.min_area,
                                     selection=args.selection, max_size=args.max_size, retry=not args.single_pass)
            row = {k: metadata[k] for k in ("file", "status", "area_ratio", "contour_rank", "reason")}
        except (ValueError, OSError, cv.error) as error:
            failed = True
            row = {"file": path.name, "status": "error", "area_ratio": None,
                   "contour_rank": None, "reason": str(error)}
        rows.append(row)
        print(f"{row['file']}: {row['status']} — {row['reason']}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidates = sum(row["status"] == "boundary_candidate" for row in rows)
    print(f"\nProcessed {len(files)} image(s); proposed {candidates} boundaries.")
    print("These are detection candidates, not measured accuracy. Inspect the overlays.")
    print(f"Saved results to: {args.output.resolve()}")
    if failed:
        parser.exit(1, "One or more images could not be processed; see summary.csv\n")


if __name__ == "__main__":
    main()
