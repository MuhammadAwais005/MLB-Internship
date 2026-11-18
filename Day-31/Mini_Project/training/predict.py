"""
predict.py - Run the model on test images and save the results
==============================================================

Uses the same ONNX detector as the web app, so what you see here is exactly
what the app does. For every image it prints the detected objects with their
confidence scores and saves:

    results/predictions/images/<name>.jpg   image with boxes and 'class 0.93' labels
    results/predictions/detections.csv      one row per box (image, class, confidence, x1, y1, x2, y2)
    results/predictions/detections.json     the same, grouped by image
    results/predictions/ppe_report.csv      per worker: helmet / vest / boots, status
    results/predictions/summary.json        totals, average confidence, speed
    results/predictions/predictions_grid.jpg  first 12 results on one sheet

Run:
    python training/predict.py                         # all 78 test images
    python training/predict.py --num 12 --conf 0.4
    python training/predict.py --source path/to/photos_or_one_image.jpg
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from common import FINAL_ONNX, PREDICTIONS_DIR, find_dataset
from ppe_detector.compliance import check_workers
from ppe_detector.detector import PPEDetector
from ppe_detector.visualize import draw_detections

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Predict PPE on images and save the results")
    p.add_argument("--source", default=None, help="image or folder (default: the test split)")
    p.add_argument("--model", default=str(FINAL_ONNX))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--num", type=int, default=0, help="only the first N images (0 = all)")
    p.add_argument("--dataset", default=None)
    return p.parse_args()


def list_images(source, dataset_arg):
    if source is None:
        folder = find_dataset(dataset_arg) / "test" / "images"
    else:
        folder = Path(source)
        if folder.is_file():
            return [folder]
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES)


def short_name(path):
    """Roboflow names are long: 'image_190_jpg.rf.0595f7e6...jpg' -> 'image_190'"""
    stem = path.stem.split(".rf.")[0]
    return stem[:-4] if stem.endswith("_jpg") else stem


def make_grid(paths, columns=4, tile=360):
    tiles = []
    for p in paths:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        s = tile / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        canvas = np.full((tile, tile, 3), 30, np.uint8)
        y, x = (tile - img.shape[0]) // 2, (tile - img.shape[1]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        tiles.append(canvas)
    while len(tiles) % columns:
        tiles.append(np.full((tile, tile, 3), 30, np.uint8))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


def main():
    args = parse_args()
    detector = PPEDetector(args.model)
    images = list_images(args.source, args.dataset)
    if args.num:
        images = images[:args.num]
    if not images:
        raise SystemExit("No images found.")

    out_images = PREDICTIONS_DIR / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    print(f"Model {Path(args.model).name} | {len(images)} images | conf {args.conf} | iou {args.iou}\n")

    box_rows, per_image, worker_rows, saved = [], {}, [], []
    used_names = Counter()      # augmented copies share a short name -> number them
    totals, timings = Counter(), []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  skipped (unreadable): {path.name}")
            continue
        start = time.perf_counter()
        detections = detector.detect(image, conf=args.conf, iou=args.iou)
        timings.append((time.perf_counter() - start) * 1000)

        base = short_name(path)
        used_names[base] += 1
        suffix = f"_copy{used_names[base]}" if used_names[base] > 1 else ""
        out_path = out_images / f"{base}{suffix}.jpg"
        cv2.imwrite(str(out_path), draw_detections(image, detections))
        saved.append((base, out_path))

        totals.update(d.name for d in detections)
        per_image[path.name] = [d.to_dict() for d in detections]
        box_rows += [{"image": path.name, **d.to_dict()} for d in detections]
        worker_rows += [{"image": path.name, **w} for w in check_workers(detections)]

        found = ", ".join(f"{d.name} {d.confidence:.2f}" for d in detections) or "nothing"
        print(f"{short_name(path):<32} {len(detections):>2} objects: {found}")

    fields = ["image", "class_id", "name", "confidence", "x1", "y1", "x2", "y2"]
    with (PREDICTIONS_DIR / "detections.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(box_rows)
    (PREDICTIONS_DIR / "detections.json").write_text(json.dumps(per_image, indent=2), encoding="utf-8")

    if worker_rows:
        with (PREDICTIONS_DIR / "ppe_report.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(worker_rows[0]))
            writer.writeheader()
            writer.writerows(worker_rows)

    # Roboflow keeps several augmented copies of some photos; show each photo once
    unique, seen_photos = [], set()
    for base, out_path in saved:
        if base not in seen_photos:
            seen_photos.add(base)
            unique.append(out_path)
    cv2.imwrite(str(PREDICTIONS_DIR / "predictions_grid.jpg"), make_grid(unique[:12]))

    confidences = [r["confidence"] for r in box_rows]
    compliant = sum(r["status"] == "OK" for r in worker_rows)
    summary = {
        "images": len(saved), "distinct_photos": len(seen_photos), "detections": len(box_rows), "per_class": dict(totals),
        "mean_confidence": round(float(np.mean(confidences)), 3) if confidences else None,
        "workers": len(worker_rows), "workers_fully_equipped": compliant,
        "ms_per_image": round(float(np.mean(timings)), 1),
        "conf_threshold": args.conf, "iou_threshold": args.iou, "model": Path(args.model).name,
    }
    (PREDICTIONS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{summary['detections']} objects in {summary['images']} images: "
          + ", ".join(f"{k} {v}" for k, v in totals.most_common()))
    print(f"Mean confidence {summary['mean_confidence']}, {summary['ms_per_image']} ms per image (CPU, ONNX Runtime)")
    print(f"Workers: {len(worker_rows)}, fully equipped: {compliant}")
    print(f"Saved to {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
