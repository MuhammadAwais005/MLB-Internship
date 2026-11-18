"""
05_inference.py - Run the trained model on new images
=====================================================

Default: predicts on the 12 test images chosen below (at least 10 are required)
and saves:
    outputs/inference/predictions/<image>.jpg   image with predicted boxes
    outputs/inference/ground_truth_vs_prediction.jpg   labels (left) vs model (right)
    outputs/inference/inference_grid.jpg         all predictions in one picture
    outputs/inference/detections.csv             every box: class, confidence, coordinates
    outputs/inference/ppe_check.csv              per worker: helmet / vest / boots found?

Run:
    python 05_inference.py                          # 12 test images
    python 05_inference.py --num 30                 # 30 test images
    python 05_inference.py --source my_photo.jpg    # any image or folder
    python 05_inference.py --conf 0.4               # stricter confidence threshold
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from config import BEST_MODEL, CLASS_COLORS, CLASS_NAMES, DATASET_DIR, INFER_DIR, RUN_NAME, TRAIN_DIR

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PERSON, HELMET, BOOTS, VEST = 2, 0, 1, 3


def training_imgsz():
    args_file = TRAIN_DIR / RUN_NAME / "args.yaml"
    if args_file.exists():
        return int(yaml.safe_load(args_file.read_text(encoding="utf-8"))["imgsz"])
    return 640


def pick_images(source, num):
    if source:
        source = Path(source)
        if source.is_file():
            return [source]
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_TYPES)[:num or None]
    images = sorted((DATASET_DIR / "test" / "images").iterdir())
    random.Random(31).shuffle(images)        # fixed seed -> same images every run
    return images[:num]


def draw(image, boxes, confidences=None):
    """Draws boxes with class name (and confidence) labels."""
    out = image.copy()
    thickness = max(2, round(min(out.shape[:2]) / 320))
    for i, (cls, x1, y1, x2, y2) in enumerate(boxes):
        color = CLASS_COLORS[cls]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = CLASS_NAMES[cls] + (f" {confidences[i]:.2f}" if confidences is not None else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(th + 4, y1)
        cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 4, y_text), color, -1)
        cv2.putText(out, label, (x1 + 2, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def ground_truth(image_path, shape):
    """Reads the YOLO label file next to a dataset image (if there is one)."""
    label = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    if not label.exists():
        return None
    h, w = shape[:2]
    boxes = []
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, x, y, bw, bh = int(parts[0]), *map(float, parts[1:])
        boxes.append((c, int((x - bw / 2) * w), int((y - bh / 2) * h), int((x + bw / 2) * w), int((y + bh / 2) * h)))
    return boxes


def ppe_check(boxes):
    """
    Simple safety check: a piece of equipment belongs to a worker when its
    centre lies inside the worker's (person) box.
    Returns one dict per detected person.
    """
    people = [b for b in boxes if b[0] == PERSON]
    report = []
    for _, px1, py1, px2, py2 in people:
        inside = lambda b: px1 <= (b[1] + b[3]) / 2 <= px2 and py1 <= (b[2] + b[4]) / 2 <= py2
        found = {name: any(b[0] == cls and inside(b) for b in boxes)
                 for cls, name in ((HELMET, "helmet"), (VEST, "vest"), (BOOTS, "boots"))}
        report.append(found)
    return report


def caption(tile, text, color=(40, 40, 40)):
    bar = np.full((28, tile.shape[1], 3), color, np.uint8)
    cv2.putText(bar, text[:48], (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([tile, bar])


def main():
    parser = argparse.ArgumentParser(description="PPE detection inference")
    parser.add_argument("--weights", default=str(BEST_MODEL))
    parser.add_argument("--source", help="image file or folder (default: test split)")
    parser.add_argument("--num", type=int, default=12, help="number of test images (default 12)")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum confidence to keep a box")
    args = parser.parse_args()

    images = pick_images(args.source, args.num)
    if not images:
        raise SystemExit("No images found.")
    model = YOLO(args.weights)
    imgsz = training_imgsz()
    pred_dir = INFER_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    detection_rows, ppe_rows, grid_tiles, gt_tiles = [], [], [], []
    total_ms = 0.0
    print(f"Running {Path(args.weights).name} on {len(images)} image(s), imgsz {imgsz}, conf >= {args.conf}\n")

    for result in model.predict([str(p) for p in images], imgsz=imgsz, conf=args.conf, stream=True, verbose=False):
        path = Path(result.path)
        image = result.orig_img
        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        boxes = [(int(c), *map(int, b)) for c, b in zip(classes, xyxy)]
        total_ms += sum(result.speed.values())

        annotated = draw(image, boxes, confs)
        cv2.imwrite(str(pred_dir / f"{path.stem[:60]}.jpg"), annotated)

        for (c, x1, y1, x2, y2), conf in zip(boxes, confs):
            detection_rows.append({"image": path.name, "class": CLASS_NAMES[c], "confidence": round(float(conf), 3),
                                   "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        workers = ppe_check(boxes)
        for i, w in enumerate(workers, start=1):
            missing = [k for k, v in w.items() if not v]
            ppe_rows.append({"image": path.name, "worker": i, **w,
                             "status": "OK" if not missing else "missing " + ", ".join(missing)})

        counts = {name: int(np.sum(classes == c)) for c, name in CLASS_NAMES.items()}
        summary = " ".join(f"{n[0].upper()}{v}" for n, v in counts.items() if v) or "nothing"
        no_helmet = sum(1 for w in workers if not w["helmet"])
        print(f"  {path.name[:46]:<46} {len(boxes):>2} boxes  " +
              ", ".join(f"{v} {n}" for n, v in counts.items() if v) +
              (f"  | {no_helmet} worker(s) without helmet" if no_helmet else ""))

        tile = cv2.resize(annotated, (320, 320))
        grid_tiles.append(caption(tile, f"{len(boxes)} boxes: {summary}"))

        truth = ground_truth(path, image.shape)
        if truth is not None and len(gt_tiles) < 6:
            left = caption(cv2.resize(draw(image, truth), (320, 320)), f"labels ({len(truth)})", (90, 90, 90))
            right = caption(tile, f"model ({len(boxes)})", (0, 110, 40))
            gt_tiles.append(np.hstack([left, np.full((348, 6, 3), 255, np.uint8), right]))

    # grid of all predictions
    while len(grid_tiles) % 4:
        grid_tiles.append(np.full_like(grid_tiles[0], 255))
    grid = np.vstack([np.hstack(grid_tiles[i:i + 4]) for i in range(0, len(grid_tiles), 4)])
    cv2.imwrite(str(INFER_DIR / "inference_grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if gt_tiles:
        pairs = [np.hstack([gt_tiles[i], np.full((348, 16, 3), 255, np.uint8),
                            gt_tiles[i + 1] if i + 1 < len(gt_tiles) else np.full_like(gt_tiles[i], 255)])
                 for i in range(0, len(gt_tiles), 2)]
        cv2.imwrite(str(INFER_DIR / "ground_truth_vs_prediction.jpg"),
                    np.vstack([np.vstack([p, np.full((12, p.shape[1], 3), 255, np.uint8)]) for p in pairs]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    for name, rows in (("detections.csv", detection_rows), ("ppe_check.csv", ppe_rows)):
        if rows:
            with open(INFER_DIR / name, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    per_class = {n: sum(1 for r in detection_rows if r["class"] == n) for n in CLASS_NAMES.values()}
    ok = sum(1 for r in ppe_rows if r["status"] == "OK")
    print(f"\n{len(detection_rows)} boxes in {len(images)} images: " + ", ".join(f"{v} {k}" for k, v in per_class.items()))
    print(f"Workers detected: {len(ppe_rows)}, fully equipped (helmet + vest + boots): {ok}")
    print(f"Average time per image: {total_ms / len(images):.1f} ms")
    print(f"Saved to {INFER_DIR}")


if __name__ == "__main__":
    main()
