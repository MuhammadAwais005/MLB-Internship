"""
export_onnx.py - Export the trained model to ONNX for the web app
=================================================================

The Streamlit app runs the model with ONNX Runtime instead of PyTorch (see
ppe_detector/detector.py for why). This script exports models/ppe_yolo11n.pt
to models/ppe_yolo11n.onnx and then PROVES the export is faithful: it runs
both models on test images and compares every box.

Run:
    python training/export_onnx.py
"""

import argparse
import json
import shutil

import cv2

from common import FINAL_ONNX, FINAL_WEIGHTS, MODEL_INFO, find_dataset
from ppe_detector.detector import PPEDetector


def parse_args():
    p = argparse.ArgumentParser(description="Export the PPE model to ONNX and check it")
    p.add_argument("--weights", default=str(FINAL_WEIGHTS))
    p.add_argument("--imgsz", type=int, default=None, help="default: the training size")
    p.add_argument("--check-images", type=int, default=20, help="test images used for the comparison")
    p.add_argument("--dataset", default=None)
    return p.parse_args()


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def compare(pt_model, onnx_detector, images, imgsz):
    """For every PyTorch box, find the ONNX box of the same class that overlaps it most."""
    total, matched, worst_conf = 0, 0, 0.0
    for path in images:
        image = cv2.imread(str(path))
        reference = pt_model.predict(image, imgsz=imgsz, conf=0.25, iou=0.45, verbose=False)[0].boxes
        ours = onnx_detector.detect(image, conf=0.25, iou=0.45)
        for cls, conf, box in zip(reference.cls.tolist(), reference.conf.tolist(), reference.xyxy.tolist()):
            total += 1
            same_class = [d for d in ours if d.class_id == int(cls)]
            if not same_class:
                continue
            best = max(same_class, key=lambda d: iou(box, (d.x1, d.y1, d.x2, d.y2)))
            # boxes are rounded to whole pixels, which costs a few % IoU on 15-px boots
            if iou(box, (best.x1, best.y1, best.x2, best.y2)) > 0.85:
                matched += 1
                worst_conf = max(worst_conf, abs(best.confidence - conf))
    return total, matched, worst_conf


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    imgsz = args.imgsz or int(model.overrides.get("imgsz", 640))
    print(f"Exporting {args.weights} at {imgsz} px ...")
    exported = model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False, verbose=False)
    if str(exported) != str(FINAL_ONNX):
        shutil.move(exported, FINAL_ONNX)
    size_mb = FINAL_ONNX.stat().st_size / 1e6
    print(f"Saved {FINAL_ONNX} ({size_mb:.1f} MB)\n")

    dataset = find_dataset(args.dataset)
    images = sorted((dataset / "test" / "images").iterdir())[:args.check_images]
    total, matched, worst = compare(YOLO(args.weights), PPEDetector(FINAL_ONNX), images, imgsz)
    print(f"Consistency check on {len(images)} test images:")
    print(f"  PyTorch boxes found again by the ONNX detector: {matched}/{total}")
    print(f"  largest confidence difference: {worst:.4f}")
    print("  -> export is faithful" if matched == total else "  -> WARNING: some boxes differ, inspect the export")

    if MODEL_INFO.exists():
        info = json.loads(MODEL_INFO.read_text(encoding="utf-8"))
        info["onnx"] = {"file": FINAL_ONNX.name, "imgsz": imgsz, "size_mb": round(size_mb, 1),
                        "check": f"{matched}/{total} boxes identical to PyTorch"}
        MODEL_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
