"""
04_evaluate.py - Evaluate the trained model
===========================================

Measures the best weights on three image sets:
    valid        the validation split (used during training to pick the best epoch)
    test         the full test split (never used for training decisions)
    clean test   only the test photos whose original photo is NOT in train/valid
                 (Roboflow put augmented copies of 40 test photos into train, so
                 the full test score is optimistic - this one is the honest number)

For each set it reports precision, recall, mAP50 and mAP50-95, overall and per
class, plus inference speed, and saves Ultralytics' plots (confusion matrix,
PR / F1 curves). Everything ends up in outputs/evaluation/.

Run:
    python 04_evaluate.py
    python 04_evaluate.py --weights models/ppe_yolo11n_best.pt --conf 0.001
"""

import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

from config import BEST_MODEL, CLASS_NAMES, DATASET_DIR, EVAL_DIR, FIXED_YAML, RUN_NAME, TRAIN_DIR, write_dataset_yaml


def original_name(filename):
    return re.sub(r"\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", "", filename)


def make_clean_test_yaml():
    """
    Builds a dataset file whose 'val' is a list of only the unseen test images.
    Ultralytics accepts a .txt file with one image path per line.
    """
    seen = {original_name(p.name) for split in ("train", "valid")
            for p in (DATASET_DIR / split / "images").iterdir()}
    test_images = sorted((DATASET_DIR / "test" / "images").iterdir())
    clean = [p for p in test_images if original_name(p.name) not in seen]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    list_file = EVAL_DIR / "clean_test_images.txt"
    list_file.write_text("\n".join(str(p) for p in clean) + "\n", encoding="utf-8")

    config = yaml.safe_load(FIXED_YAML.read_text(encoding="utf-8"))
    config["val"] = str(list_file)
    clean_yaml = EVAL_DIR / "clean_test.yaml"
    clean_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return clean_yaml, len(clean), len(test_images)


def training_imgsz():
    args_file = TRAIN_DIR / RUN_NAME / "args.yaml"
    if args_file.exists():
        return int(yaml.safe_load(args_file.read_text(encoding="utf-8"))["imgsz"])
    return 640


def evaluate(model, data, split, name, imgsz, conf):
    metrics = model.val(data=str(data), split=split, imgsz=imgsz, conf=conf, batch=16,
                        project=str(EVAL_DIR), name=name, exist_ok=True, plots=True, verbose=False)
    box = metrics.box
    result = {
        "precision": float(box.mp), "recall": float(box.mr),
        "mAP50": float(box.map50), "mAP50-95": float(box.map),
        "speed_ms_per_image": {k: round(v, 1) for k, v in metrics.speed.items()},
        "per_class": {},
    }
    # box.ap_class_index lists the classes that appear in this set
    for i, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        result["per_class"][CLASS_NAMES[int(cls)]] = {
            "precision": float(p), "recall": float(r), "mAP50": float(ap50), "mAP50-95": float(ap)}
    return result


def print_result(title, result, images):
    print(f"\n{title}  ({images} images)")
    print(f"  {'class':<8}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}")
    for cls, m in result["per_class"].items():
        print(f"  {cls:<8}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['mAP50']:>8.3f}{m['mAP50-95']:>10.3f}")
    print(f"  {'ALL':<8}{result['precision']:>7.3f}{result['recall']:>7.3f}{result['mAP50']:>8.3f}{result['mAP50-95']:>10.3f}")
    s = result["speed_ms_per_image"]
    print(f"  speed: {s.get('preprocess', 0)} ms preprocess + {s.get('inference', 0)} ms inference + "
          f"{s.get('postprocess', 0)} ms NMS per image")


def comparison_plot(results):
    """Per-class mAP50 for valid / test / clean test side by side."""
    names = list(CLASS_NAMES.values()) + ["ALL"]
    x = np.arange(len(names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, res) in enumerate(results.items()):
        values = [res["per_class"].get(n, {}).get("mAP50", 0) for n in names[:-1]] + [res["mAP50"]]
        bars = ax.bar(x + (i - 1) * width, values, width, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mAP50")
    ax.set_title("mAP50 per class on each evaluation set")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "map50_comparison.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained PPE model")
    parser.add_argument("--weights", default=str(BEST_MODEL))
    parser.add_argument("--conf", type=float, default=0.001,
                        help="confidence threshold for mAP (keep it low - mAP integrates over all thresholds)")
    args = parser.parse_args()

    write_dataset_yaml()
    model = YOLO(args.weights)
    imgsz = training_imgsz()
    clean_yaml, n_clean, n_test = make_clean_test_yaml()
    n_valid = len(list((DATASET_DIR / "valid" / "images").iterdir()))

    print(f"Weights: {args.weights}   image size: {imgsz}")
    results = {
        "valid": evaluate(model, FIXED_YAML, "val", "valid", imgsz, args.conf),
        "test": evaluate(model, FIXED_YAML, "test", "test", imgsz, args.conf),
        "clean test": evaluate(model, clean_yaml, "val", "clean_test", imgsz, args.conf),
    }

    print("\n" + "=" * 52)
    print_result("VALIDATION", results["valid"], n_valid)
    print_result("TEST (all)", results["test"], n_test)
    print_result("CLEAN TEST (never-seen photos only)", results["clean test"], n_clean)
    gap = results["test"]["mAP50-95"] - results["clean test"]["mAP50-95"]
    print(f"\nLeakage effect: full test mAP50-95 is {gap:+.3f} vs. the clean subset.")

    comparison_plot(results)
    summary = {"weights": args.weights, "imgsz": imgsz, "images": {"valid": n_valid, "test": n_test, "clean test": n_clean},
               "results": results}
    (EVAL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {EVAL_DIR / 'metrics.json'}, map50_comparison.png and the plots in valid/, test/, clean_test/")


if __name__ == "__main__":
    main()
