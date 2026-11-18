"""
01_explore_dataset.py - Explore the Roboflow YOLO dataset
=========================================================

Answers the questions you should ask before training anything:
    - How is the dataset organised, and what does data.yaml say?
    - How many images / labels / boxes per split and per class?
    - Are labels valid (5 numbers per line, coordinates between 0 and 1)?
    - How big are the objects? (tiny objects are hard for detectors)
    - Is the same photo in more than one split? (data leakage)

Outputs (outputs/exploration/):
    class_distribution.png     boxes per class for train / valid / test
    box_sizes.png              object width x height per class
    sample_annotations.jpg     12 training images with their labelled boxes
    class_examples.jpg         crops of each class (how the class names were identified)
    dataset_summary.json       all numbers in one file

Run:
    python 01_explore_dataset.py
"""

import json
import random
import re
from collections import Counter, defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from config import CLASS_COLORS, CLASS_NAMES, DATASET_DIR, EXPLORE_DIR, ORIGINAL_YAML, SPLITS, write_dataset_yaml

random.seed(7)


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------

def explain_data_yaml():
    """Prints the Roboflow data.yaml with an explanation of every key."""
    raw = ORIGINAL_YAML.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    print("=" * 72)
    print("data.yaml (as exported by Roboflow)")
    print("=" * 72)
    print(raw.strip())
    print("\nWhat each key means:")
    print(f"  train / val / test : folders with the images of each split ({data['train']}, ...).")
    print("                       YOLO finds each label by swapping 'images' for 'labels' in the path")
    print("                       and '.jpg' for '.txt'.")
    print(f"  nc                 : number of classes = {data['nc']}")
    print(f"  names              : class names, index = class id in the label files -> {data['names']}")
    print("  roboflow           : where the dataset came from (workspace, project, version, licence)")
    print("\nTwo problems, fixed automatically in dataset.yaml (see config.py):")
    print("  1. '../train/images' is relative to the wrong folder for Ultralytics -> 'images not found'.")
    print("  2. Names are just '0'-'3'. From the labelled boxes they are:", CLASS_NAMES)
    fixed = write_dataset_yaml()
    print(f"\nWritten: {fixed.name}")
    print(fixed.read_text(encoding="utf-8").strip())
    return data


# ---------------------------------------------------------------------------
# Reading labels
# ---------------------------------------------------------------------------

def read_split(split):
    """
    Reads every label file of a split. YOLO format, one object per line:
        class_id  x_center  y_center  width  height      (all relative 0..1)
    """
    image_dir, label_dir = DATASET_DIR / split / "images", DATASET_DIR / split / "labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    info = {"images": len(images), "labels": len(list(label_dir.glob("*.txt"))),
            "missing_labels": 0, "empty_labels": 0, "bad_lines": 0, "duplicate_boxes": 0,
            "boxes": [], "boxes_per_image": [], "files": images, "image_sizes": Counter()}

    for image in images:
        label = label_dir / (image.stem + ".txt")
        if not label.exists():
            info["missing_labels"] += 1
            continue
        lines = [l.split() for l in label.read_text().splitlines() if l.strip()]
        if not lines:
            info["empty_labels"] += 1
        seen = set()
        count = 0
        for parts in lines:
            if len(parts) != 5:
                info["bad_lines"] += 1
                continue
            cls, *xywh = int(parts[0]), *map(float, parts[1:])
            if not all(0 <= v <= 1 for v in xywh) or cls not in CLASS_NAMES:
                info["bad_lines"] += 1
                continue
            key = (cls, *[round(v, 4) for v in xywh])
            if key in seen:
                info["duplicate_boxes"] += 1
                continue
            seen.add(key)
            info["boxes"].append((cls, *xywh, image.name))
            count += 1
        info["boxes_per_image"].append(count)

    for image in images[:: max(1, len(images) // 40)]:      # sample image sizes
        h, w = cv2.imread(str(image)).shape[:2]
        info["image_sizes"][f"{w}x{h}"] += 1
    return info


def original_name(filename):
    """'image_150_jpg.rf.3426ad87....jpg' -> 'image_150_jpg' (the photo before Roboflow's export)."""
    return re.sub(r"\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", "", filename)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_class_distribution(stats):
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.27
    x = np.arange(len(CLASS_NAMES))
    for i, split in enumerate(SPLITS):
        counts = Counter(b[0] for b in stats[split]["boxes"])
        values = [counts.get(c, 0) for c in CLASS_NAMES]
        bars = ax.bar(x + (i - 1) * width, values, width, label=f"{split} ({stats[split]['images']} images)")
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, [f"{c}: {n}" for c, n in CLASS_NAMES.items()])
    ax.set_ylabel("Number of boxes")
    ax.set_title("Boxes per class and split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(EXPLORE_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)


def plot_box_sizes(stats):
    boxes = stats["train"]["boxes"] + stats["valid"]["boxes"] + stats["test"]["boxes"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for c, name in CLASS_NAMES.items():
        w = [b[3] for b in boxes if b[0] == c]
        h = [b[4] for b in boxes if b[0] == c]
        color = np.array(CLASS_COLORS[c][::-1]) / 255
        axes[0].scatter(w, h, s=6, alpha=0.45, color=color, label=name)
        axes[1].hist(np.sqrt(np.array(w) * np.array(h)) * 640, bins=40, alpha=0.55, color=color, label=name)
    axes[0].set_xlabel("box width (fraction of image)")
    axes[0].set_ylabel("box height (fraction of image)")
    axes[0].set_title("Object shapes")
    axes[0].legend(markerscale=3)
    axes[1].set_xlabel("object size in pixels at 640 px (sqrt of area)")
    axes[1].set_ylabel("number of boxes")
    axes[1].set_title("Object sizes")
    axes[1].axvline(32, color="red", linestyle="--", linewidth=1)
    axes[1].text(34, axes[1].get_ylim()[1] * 0.9, "small (<32 px)", color="red", fontsize=9)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(EXPLORE_DIR / "box_sizes.png", dpi=120)
    plt.close(fig)


def draw_boxes(image, rows):
    h, w = image.shape[:2]
    for cls, x, y, bw, bh in rows:
        x1, y1, x2, y2 = int((x - bw / 2) * w), int((y - bh / 2) * h), int((x + bw / 2) * w), int((y + bh / 2) * h)
        color = CLASS_COLORS[cls]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES[cls]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
        cv2.putText(image, label, (x1 + 2, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def save_sample_grid(stats):
    by_image = defaultdict(list)
    for cls, x, y, w, h, name in stats["train"]["boxes"]:
        by_image[name].append((cls, x, y, w, h))
    names = random.sample(sorted(by_image), 12)
    tiles = []
    for name in names:
        image = cv2.imread(str(DATASET_DIR / "train" / "images" / name))
        tiles.append(cv2.resize(draw_boxes(image, by_image[name]), (320, 320)))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]
    cv2.imwrite(str(EXPLORE_DIR / "sample_annotations.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def save_class_examples(stats, per_class=10):
    """Crops of each class - this is how '0'..'3' were identified as helmet/boots/person/vest."""
    crops = defaultdict(list)
    boxes = stats["train"]["boxes"][:]
    random.shuffle(boxes)
    cache = {}
    for cls, x, y, w, h, name in boxes:
        if len(crops[cls]) >= per_class:
            continue
        if name not in cache:
            cache[name] = cv2.imread(str(DATASET_DIR / "train" / "images" / name))
        image = cache[name]
        H, W = image.shape[:2]
        x1, y1, x2, y2 = int((x - w / 2) * W), int((y - h / 2) * H), int((x + w / 2) * W), int((y + h / 2) * H)
        crop = image[max(0, y1):y2, max(0, x1):x2]
        if crop.size and min(crop.shape[:2]) > 12:
            crops[cls].append(cv2.resize(crop, (110, 110)))
    rows = []
    for cls, name in CLASS_NAMES.items():
        label = np.full((110, 150, 3), 255, np.uint8)
        cv2.putText(label, f"{cls} = {name}", (6, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        tiles = crops[cls] + [np.full((110, 110, 3), 255, np.uint8)] * (per_class - len(crops[cls]))
        rows.append(np.hstack([label] + tiles))
    cv2.imwrite(str(EXPLORE_DIR / "class_examples.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EXPLORE_DIR.mkdir(parents=True, exist_ok=True)
    explain_data_yaml()

    print("\n" + "=" * 72)
    print("Folder structure")
    print("=" * 72)
    print("Dataset/\n├── data.yaml\n├── README.dataset.txt, README.roboflow.txt")
    for split in SPLITS:
        print(f"├── {split}/\n│   ├── images/   (.jpg)\n│   └── labels/   (.txt, same file name, one line per object)")

    stats = {split: read_split(split) for split in SPLITS}

    print("\n" + "=" * 72)
    print("Images, labels and boxes")
    print("=" * 72)
    header = f"{'split':<7}{'images':>8}{'labels':>8}{'boxes':>7}{'box/img':>9}{'max':>5}  " + \
             "  ".join(f"{n:>7}" for n in CLASS_NAMES.values())
    print(header)
    summary = {}
    for split in SPLITS:
        s = stats[split]
        counts = Counter(b[0] for b in s["boxes"])
        bpi = s["boxes_per_image"]
        print(f"{split:<7}{s['images']:>8}{s['labels']:>8}{len(s['boxes']):>7}{np.mean(bpi):>9.1f}{max(bpi):>5}  " +
              "  ".join(f"{counts.get(c, 0):>7}" for c in CLASS_NAMES))
        summary[split] = {"images": s["images"], "labels": s["labels"], "boxes": len(s["boxes"]),
                          "boxes_per_image_mean": round(float(np.mean(bpi)), 2), "boxes_per_image_max": max(bpi),
                          "per_class": {CLASS_NAMES[c]: counts.get(c, 0) for c in CLASS_NAMES},
                          "missing_labels": s["missing_labels"], "empty_labels": s["empty_labels"],
                          "bad_lines": s["bad_lines"], "duplicate_boxes": s["duplicate_boxes"],
                          "image_sizes_sampled": dict(s["image_sizes"])}

    total = sum(len(stats[s]["files"]) for s in SPLITS)
    print(f"\nSplit ratio: " + ", ".join(f"{s} {100 * stats[s]['images'] / total:.0f}%" for s in SPLITS))
    print("Image sizes (sampled):", dict(stats["train"]["image_sizes"]))

    print("\nLabel quality checks:")
    for split in SPLITS:
        s = stats[split]
        print(f"  {split:<6} missing labels: {s['missing_labels']}, empty: {s['empty_labels']}, "
              f"invalid lines: {s['bad_lines']}, duplicate boxes: {s['duplicate_boxes']}")

    all_boxes = [b for s in SPLITS for b in stats[s]["boxes"]]
    sizes = {c: np.sqrt(np.array([b[3] * b[4] for b in all_boxes if b[0] == c])) * 640 for c in CLASS_NAMES}
    print("\nObject size at 640 px (median, and share smaller than 32 px):")
    for c, values in sizes.items():
        print(f"  {CLASS_NAMES[c]:<7} median {np.median(values):5.0f} px   small: {100 * np.mean(values < 32):4.1f}%")
        summary.setdefault("object_size_px", {})[CLASS_NAMES[c]] = {
            "median": round(float(np.median(values)), 1), "small_pct": round(float(100 * np.mean(values < 32)), 1)}

    # data leakage: the same original photo in more than one split
    originals = {s: {original_name(p.name) for p in stats[s]["files"]} for s in SPLITS}
    leak_test = originals["test"] & originals["train"]
    leak_valid = originals["valid"] & originals["train"]
    print("\nData leakage check (same original photo in several splits):")
    for s in SPLITS:
        print(f"  {s:<6} {stats[s]['images']} files from {len(originals[s])} original photos")
    print(f"  test photos also in train : {len(leak_test)} of {len(originals['test'])}")
    print(f"  valid photos also in train: {len(leak_valid)} of {len(originals['valid'])}")
    print("  -> Roboflow exported some photos several times (augmented copies) and they ended up in")
    print("     different splits. Scores on these images are optimistic; 04_evaluate.py also reports")
    print("     the test score on only the photos the model has never seen in any form.")
    summary["leakage"] = {"test_in_train": len(leak_test), "test_originals": len(originals["test"]),
                          "valid_in_train": len(leak_valid), "valid_originals": len(originals["valid"])}

    plot_class_distribution(stats)
    plot_box_sizes(stats)
    save_sample_grid(stats)
    save_class_examples(stats)
    (EXPLORE_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved plots and summary to {EXPLORE_DIR}")


if __name__ == "__main__":
    main()
