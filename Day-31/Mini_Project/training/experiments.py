"""
experiments.py - Controlled training experiments
================================================

The task asks what happens when you change the number of epochs, the batch
size, the image size and the data augmentation. This script answers that with
short, comparable runs. Every run starts from the same COCO weights with the
same seed and changes ONE thing compared with the baseline:

    A_baseline    imgsz 416, batch 16, Ultralytics default augmentation
    B_imgsz320    smaller images   (faster, but small helmets/boots lose pixels)
    C_imgsz512    larger images    (slower, more detail)
    D_batch8      smaller batch    (noisier gradients, more updates per epoch)
    E_no_aug      augmentation off (no mosaic, flips, colour, scale, translate, erasing)

All runs use the same short budget (10 epochs by default). The effect of training
LONGER is read from the final run (train.py), whose per-epoch curve is compared
with the 10-epoch run in the README.

Note on mosaic: Ultralytics switches mosaic augmentation off for the last 10
epochs (close_mosaic=10). With a 10-epoch budget that means mosaic is off for
the WHOLE run, so "default augmentation" here = flips, colour, scale, translate
and random erasing, without mosaic. E_no_aug switches those off as well.

Results:
    results/experiments/<name>/results.csv      per-epoch numbers of each run
    results/experiments/summary.csv / .json     one row per run
    results/experiments/experiment_curves.png   mAP per epoch, all runs
    results/experiments/experiment_summary.png  final mAP and time per epoch

Run:
    python training/experiments.py                    # all runs, 10 epochs each
    python training/experiments.py --only A_baseline,E_no_aug --epochs 5

Finished runs are skipped, so the script can be stopped and started again.
"""

import argparse
import csv
import json
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EXPERIMENTS_DIR, device_label, epoch_logger, find_dataset, pick_device, write_dataset_yaml

BASELINE = {"imgsz": 416, "batch": 16}

NO_AUGMENTATION = {
    "mosaic": 0.0, "mixup": 0.0, "fliplr": 0.0, "scale": 0.0, "translate": 0.0,
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "erasing": 0.0,
}

EXPERIMENTS = {
    "A_baseline": ("baseline (416 px, batch 16, default augmentation)", {}),
    "B_imgsz320": ("image size 320", {"imgsz": 320}),
    "C_imgsz512": ("image size 512", {"imgsz": 512}),
    "D_batch8":   ("batch size 8", {"batch": 8}),
    "E_no_aug":   ("no data augmentation", NO_AUGMENTATION),
}

SUMMARY_FIELDS = ["name", "change", "epochs", "imgsz", "batch", "augmentation",
                  "precision", "recall", "mAP50", "mAP50-95", "best_epoch", "min_per_epoch"]


def parse_args():
    p = argparse.ArgumentParser(description="Run the PPE training experiments")
    p.add_argument("--epochs", type=int, default=10, help="epochs per experiment")
    p.add_argument("--only", default="", help="comma-separated experiment names")
    p.add_argument("--dataset", default=None, help="Roboflow dataset folder")
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--keep-weights", action="store_true", help="keep each run's .pt files (5 MB each)")
    return p.parse_args()


def read_results(csv_path):
    """results.csv -> list of dicts with float values (column names are space-padded in some versions)."""
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): float(v) for k, v in row.items() if v not in (None, "")} for row in rows]


def summarise(name, change, settings, epochs, rows):
    best = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])   # Ultralytics picks best.pt the same way (mostly mAP50-95)
    return {
        "name": name,
        "change": change,
        "epochs": epochs,
        "imgsz": settings["imgsz"],
        "batch": settings["batch"],
        # close_mosaic=10 removes mosaic from the last 10 epochs, i.e. from every epoch of a short run
        "augmentation": "off" if settings.get("mosaic") == 0.0 else
                        ("default, no mosaic" if epochs <= 10 else "default"),
        "precision": round(best["metrics/precision(B)"], 4),
        "recall": round(best["metrics/recall(B)"], 4),
        "mAP50": round(best["metrics/mAP50(B)"], 4),
        "mAP50-95": round(best["metrics/mAP50-95(B)"], 4),
        "best_epoch": int(best["epoch"]),
        "min_per_epoch": round(rows[-1]["time"] / len(rows) / 60, 2),
    }


def save_summary(summary):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENTS_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    (EXPERIMENTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_curves(names):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name in names:
        csv_path = EXPERIMENTS_DIR / name / "results.csv"
        if not csv_path.exists():
            continue
        rows = read_results(csv_path)
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["metrics/mAP50(B)"] for r in rows], marker="o", ms=3, label=name)
        axes[1].plot(epochs, [r["metrics/mAP50-95(B)"] for r in rows], marker="o", ms=3, label=name)
    axes[0].axhline(0.80, color="red", ls="--", lw=1, label="target mAP50 = 0.80")
    for ax, title in zip(axes, ("mAP50 (validation)", "mAP50-95 (validation)")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Training experiments - one change at a time")
    fig.tight_layout()
    fig.savefig(EXPERIMENTS_DIR / "experiment_curves.png", dpi=130)
    plt.close(fig)


def plot_summary(summary):
    names = [s["name"] for s in summary]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = range(len(names))
    axes[0].bar([i - 0.2 for i in x], [s["mAP50"] for s in summary], width=0.4, label="mAP50")
    axes[0].bar([i + 0.2 for i in x], [s["mAP50-95"] for s in summary], width=0.4, label="mAP50-95")
    axes[0].axhline(0.80, color="red", ls="--", lw=1, label="target 0.80")
    for i, s in enumerate(summary):
        axes[0].text(i - 0.2, s["mAP50"] + 0.01, f"{s['mAP50']:.3f}", ha="center", fontsize=8)
        axes[0].text(i + 0.2, s["mAP50-95"] + 0.01, f"{s['mAP50-95']:.3f}", ha="center", fontsize=8)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title("Best validation score")
    axes[1].bar(x, [s["min_per_epoch"] for s in summary], color="gray")
    for i, s in enumerate(summary):
        axes[1].text(i, s["min_per_epoch"] + 0.03, f"{s['min_per_epoch']:.2f}", ha="center", fontsize=8)
    axes[1].set_title("Minutes per epoch (CPU)")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=15, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(EXPERIMENTS_DIR / "experiment_summary.png", dpi=130)
    plt.close(fig)


def run_experiment(name, overrides, args, data_yaml, device):
    from ultralytics import YOLO

    settings = {**BASELINE, **overrides}
    model = YOLO("yolo11n.pt")
    model.add_callback("on_fit_epoch_end", epoch_logger(args.epochs, prefix=f"{name:<11} "))
    model.train(
        data=str(data_yaml), epochs=args.epochs, patience=0,   # patience 0 = never stop early, so runs stay comparable
        workers=args.workers, device=device, seed=0, deterministic=True, cache=True,
        project=str(EXPERIMENTS_DIR), name=name, exist_ok=True, plots=False, verbose=False,
        **settings,
    )
    if not args.keep_weights:
        shutil.rmtree(EXPERIMENTS_DIR / name / "weights", ignore_errors=True)
    return settings


def main():
    args = parse_args()
    chosen = [n.strip() for n in args.only.split(",") if n.strip()] or list(EXPERIMENTS)
    unknown = set(chosen) - set(EXPERIMENTS)
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {', '.join(sorted(unknown))}. Choose from {', '.join(EXPERIMENTS)}")

    data_yaml = write_dataset_yaml(find_dataset(args.dataset))
    device = pick_device(args.device)
    print(f"Device: {device_label(device)} | {len(chosen)} experiment(s), {args.epochs} epochs each\n")

    previous = {}
    if (EXPERIMENTS_DIR / "summary.json").exists():
        previous = {s["name"]: s for s in json.loads((EXPERIMENTS_DIR / "summary.json").read_text(encoding="utf-8"))}

    summary = []
    for name in EXPERIMENTS:          # keep the A..E order in the summary
        if name not in chosen:
            if name in previous:
                summary.append(previous[name])
            continue
        change, overrides = EXPERIMENTS[name]
        csv_path = EXPERIMENTS_DIR / name / "results.csv"
        if name in previous and previous[name]["epochs"] == args.epochs and csv_path.exists():
            print(f"{name}: already done, skipping")
            summary.append(previous[name])
            continue

        print(f"=== {name}: {change} ===")
        start = time.time()
        settings = run_experiment(name, overrides, args, data_yaml, device)
        row = summarise(name, change, settings, args.epochs, read_results(csv_path))
        summary.append(row)
        save_summary(summary)          # save after every run, so nothing is lost if we stop
        print(f"{name}: mAP50 {row['mAP50']:.3f}, mAP50-95 {row['mAP50-95']:.3f} "
              f"({(time.time() - start) / 60:.1f} min)\n")

    save_summary(summary)
    plot_curves([s["name"] for s in summary])
    plot_summary(summary)

    print(f"{'experiment':<12} {'imgsz':>5} {'batch':>5} {'aug':>18} {'mAP50':>7} {'mAP50-95':>9} {'min/ep':>7}")
    for s in summary:
        print(f"{s['name']:<12} {s['imgsz']:>5} {s['batch']:>5} {s['augmentation']:>18} "
              f"{s['mAP50']:>7.3f} {s['mAP50-95']:>9.3f} {s['min_per_epoch']:>7.2f}")
    print(f"\nSaved to {EXPERIMENTS_DIR}")


if __name__ == "__main__":
    main()
