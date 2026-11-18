"""
evaluate.py - Evaluate the trained model
========================================

Scores models/ppe_yolo11n.pt on three image sets:
    valid       157 images, used during training to pick the best epoch
    test        78 images, never used for any training decision
    clean test  test images whose original photo has NO augmented copy in
                train/valid. Roboflow put copies of 40 test photos into train,
                so this smaller set is the strictest, most honest number.

Reports precision, recall, mAP50, mAP50-95 (overall and per class) and the
speed, checks the mAP50 >= 0.80 target, and saves the confusion matrices and
PR / F1 curves.

Run:
    python training/evaluate.py
    python training/evaluate.py --weights models/ppe_yolo11n.onnx   # score the exported model

Output: results/evaluation/  (metrics.json, per_class_map.png, one folder of plots per set)
"""

import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from common import (CLASS_NAMES, DATASET_YAML, EVALUATION_DIR, FINAL_WEIGHTS, MODEL_INFO,
                    find_dataset, write_dataset_yaml)

TARGET_MAP50 = 0.80


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the PPE model")
    p.add_argument("--weights", default=str(FINAL_WEIGHTS))
    p.add_argument("--imgsz", type=int, default=None, help="default: the size the model was trained at")
    p.add_argument("--dataset", default=None)
    return p.parse_args()


def original_photo(filename):
    """'image_190_jpg.rf.0595f7e6....jpg' -> 'image_190_jpg' (all augmented copies share it)"""
    return re.sub(r"\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", "", filename)


def clean_test_yaml(dataset):
    seen = {original_photo(p.name) for split in ("train", "valid") for p in (dataset / split / "images").iterdir()}
    test_images = sorted((dataset / "test" / "images").iterdir())
    clean = [p for p in test_images if original_photo(p.name) not in seen]

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    list_file = EVALUATION_DIR / "clean_test_images.txt"
    list_file.write_text("\n".join(str(p) for p in clean) + "\n", encoding="utf-8")

    config = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    config["val"] = str(list_file)       # Ultralytics accepts a .txt list of image paths
    path = EVALUATION_DIR / "clean_test.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path, len(clean)


def model_imgsz(model, requested):
    if requested:
        return requested
    args = getattr(model, "overrides", {}) or {}
    if args.get("imgsz"):
        size = args["imgsz"]
        return size[0] if isinstance(size, (list, tuple)) else int(size)
    if MODEL_INFO.exists():
        return json.loads(MODEL_INFO.read_text(encoding="utf-8"))["training"]["imgsz"]
    return 640


def score(model, data, split, name, imgsz):
    # conf 0.001 is the standard for mAP: the whole precision-recall curve is needed.
    # (For an ONNX file with a fixed batch of 1, Ultralytics lowers the batch itself.)
    m = model.val(data=str(data), split=split, imgsz=imgsz, conf=0.001, iou=0.7, batch=16,
                  project=str(EVALUATION_DIR), name=name, exist_ok=True, plots=True, verbose=False)
    box = m.box
    result = {
        "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4), "mAP50-95": round(float(box.map), 4),
        "ms_per_image": round(sum(m.speed.values()), 1),
        "per_class": {},
    }
    for i, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        result["per_class"][CLASS_NAMES[int(cls)]] = {
            "precision": round(float(p), 4), "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4)}
    return result


def plot_per_class(results):
    sets = list(results)
    classes = list(CLASS_NAMES.values())
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.8 / len(sets)
    for k, name in enumerate(sets):
        values = [results[name]["per_class"].get(c, {}).get("mAP50", 0) for c in classes]
        xs = [i + (k - (len(sets) - 1) / 2) * width for i in range(len(classes))]
        bars = ax.bar(xs, values, width, label=f"{name} (all: {results[name]['mAP50']:.3f})")
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.axhline(TARGET_MAP50, color="red", ls="--", lw=1, label=f"target {TARGET_MAP50:.2f}")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mAP50")
    ax.set_title("mAP50 per class")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(EVALUATION_DIR / "per_class_map.png", dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    from ultralytics import YOLO

    dataset = find_dataset(args.dataset)
    data_yaml = write_dataset_yaml(dataset)
    clean_yaml, clean_count = clean_test_yaml(dataset)

    model = YOLO(args.weights, task="detect")
    imgsz = model_imgsz(model, args.imgsz)
    suffix = "_onnx" if args.weights.endswith(".onnx") else ""
    print(f"Weights: {args.weights}  |  imgsz {imgsz}\n")

    results = {
        "valid": score(model, data_yaml, "val", "valid" + suffix, imgsz),
        "test": score(model, data_yaml, "test", "test" + suffix, imgsz),
        "clean test": score(model, clean_yaml, "val", "clean_test" + suffix, imgsz),
    }
    counts = {"valid": 157, "test": 78, "clean test": clean_count}

    print(f"{'set':<11} {'images':>6} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9} {'ms/img':>7}")
    for name, r in results.items():
        print(f"{name:<11} {counts[name]:>6} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['mAP50']:>7.3f} {r['mAP50-95']:>9.3f} {r['ms_per_image']:>7.1f}")

    print("\nPer class (test):")
    for cls, r in results["test"]["per_class"].items():
        print(f"  {cls:<7} P {r['precision']:.3f}  R {r['recall']:.3f}  "
              f"mAP50 {r['mAP50']:.3f}  mAP50-95 {r['mAP50-95']:.3f}")

    reached = results["test"]["mAP50"] >= TARGET_MAP50
    print(f"\nTarget mAP50 >= {TARGET_MAP50:.2f} on the test set: "
          f"{'REACHED' if reached else 'NOT reached'} ({results['test']['mAP50']:.3f})")

    report = {"weights": args.weights.replace("\\", "/").split("Mini_Project/")[-1], "imgsz": imgsz,
              "images": counts, "target_mAP50": TARGET_MAP50, "target_reached": reached, "results": results}
    (EVALUATION_DIR / f"metrics{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not suffix:
        plot_per_class(results)
        if MODEL_INFO.exists():         # the app shows these numbers in its sidebar
            info = json.loads(MODEL_INFO.read_text(encoding="utf-8"))
            info["test"] = {k: results["test"][k] for k in ("precision", "recall", "mAP50", "mAP50-95")}
            info["clean_test"] = {k: results["clean test"][k] for k in ("precision", "recall", "mAP50", "mAP50-95")}
            MODEL_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Saved to {EVALUATION_DIR}")


if __name__ == "__main__":
    main()
