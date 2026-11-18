"""
03_monitor_training.py - Watch the training while it runs (or review it afterwards)
===================================================================================

Ultralytics writes one row per epoch to outputs/training/ppe_yolo11n/results.csv.
This script reads that file and shows:
    - a table of every epoch (losses, precision, recall, mAP), best epoch marked
    - training_progress.png: loss curves (train vs. validation) and metric curves
    - a short diagnosis: still improving? over-fitting? when was the best epoch?

Run in a second terminal while 02_train.py is running:
    python 03_monitor_training.py --watch          # refreshes every 30 s until training ends
    python 03_monitor_training.py                  # one-off report

TensorBoard also works:  tensorboard --logdir outputs/training
"""

import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from config import RUN_NAME, TRAIN_DIR

RUN_DIR = TRAIN_DIR / RUN_NAME
RESULTS = RUN_DIR / "results.csv"
PLOT = RUN_DIR / "training_progress.png"


def load_results():
    if not RESULTS.exists():
        return []
    with RESULTS.open(encoding="utf-8") as f:
        rows = [{k.strip(): float(v) for k, v in row.items() if v.strip()} for row in csv.DictReader(f)]
    return rows


def planned_epochs():
    args_file = RUN_DIR / "args.yaml"
    if args_file.exists():
        return int(yaml.safe_load(args_file.read_text(encoding="utf-8")).get("epochs", 0))
    return 0


def print_table(rows, total):
    best = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
    print(f"\n{'epoch':>5} {'box':>6} {'cls':>6} {'dfl':>6} | {'val box':>7} {'val cls':>7} | "
          f"{'P':>6} {'R':>6} {'mAP50':>6} {'mAP50-95':>8} | {'time':>6}")
    for r in rows:
        mark = "  <- best" if r is best else ""
        print(f"{int(r['epoch']):>5} {r['train/box_loss']:>6.3f} {r['train/cls_loss']:>6.3f} {r['train/dfl_loss']:>6.3f} | "
              f"{r['val/box_loss']:>7.3f} {r['val/cls_loss']:>7.3f} | "
              f"{r['metrics/precision(B)']:>6.3f} {r['metrics/recall(B)']:>6.3f} {r['metrics/mAP50(B)']:>6.3f} "
              f"{r['metrics/mAP50-95(B)']:>8.3f} | {r['time'] / 60:>5.1f}m{mark}")
    done = int(rows[-1]["epoch"])
    print(f"\nProgress: {done}/{total or '?'} epochs, {rows[-1]['time'] / 60:.1f} min elapsed")
    return best


def diagnose(rows, best):
    """Plain-language reading of the curves."""
    last = rows[-1]
    notes = [f"Best so far: epoch {int(best['epoch'])} with mAP50-95 {best['metrics/mAP50-95(B)']:.3f} "
             f"(mAP50 {best['metrics/mAP50(B)']:.3f})."]
    if len(rows) >= 6:
        recent = [r["metrics/mAP50-95(B)"] for r in rows[-5:]]
        if max(recent) - min(recent) < 0.01:
            notes.append("mAP has been flat for 5 epochs -> the model is close to converged.")
        elif recent[-1] > recent[0]:
            notes.append("mAP is still rising -> more epochs could help.")
        train_trend = rows[-1]["train/box_loss"] - rows[-6]["train/box_loss"]
        val_trend = rows[-1]["val/box_loss"] - rows[-6]["val/box_loss"]
        if train_trend < 0 < val_trend:
            notes.append("Training loss falls while validation loss rises -> early signs of over-fitting.")
        else:
            notes.append("Training and validation losses move together -> no over-fitting.")
    epochs_since_best = int(last["epoch"] - best["epoch"])
    if epochs_since_best:
        notes.append(f"{epochs_since_best} epoch(s) since the best result (early stopping patience counts these).")
    print("\nDiagnosis:")
    for note in notes:
        print("  -", note)


def plot(rows):
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, name in zip(axes[0], ("box_loss", "cls_loss", "dfl_loss")):
        ax.plot(epochs, [r[f"train/{name}"] for r in rows], "o-", ms=3, label="train")
        ax.plot(epochs, [r[f"val/{name}"] for r in rows], "s-", ms=3, label="validation")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    pairs = [("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall")]
    for key, label in pairs:
        axes[1, 0].plot(epochs, [r[key] for r in rows], "o-", ms=3, label=label)
    axes[1, 0].set_title("precision & recall (validation)")
    for key, label in (("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")):
        axes[1, 1].plot(epochs, [r[key] for r in rows], "o-", ms=3, label=label)
    axes[1, 1].set_title("mAP (validation)")
    axes[1, 2].plot(epochs, [r["lr/pg0"] for r in rows], "o-", ms=3, color="gray")
    axes[1, 2].set_title("learning rate")
    for ax in axes[1]:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    for ax in axes[1, :2]:
        ax.set_ylim(0, 1)
        ax.legend()
    fig.suptitle(f"Training progress - {RUN_NAME}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT, dpi=110)
    plt.close(fig)


def report():
    rows = load_results()
    if not rows:
        print(f"No results yet ({RESULTS}). Has 02_train.py finished its first epoch?")
        return False
    total = planned_epochs()
    best = print_table(rows, total)
    diagnose(rows, best)
    plot(rows)
    print(f"\nCurves saved to {PLOT}")
    finished = (RUN_DIR / "weights" / "best.pt").exists() and (int(rows[-1]["epoch"]) >= total or
                                                                 time.time() - RESULTS.stat().st_mtime > 900)
    return finished


def main():
    parser = argparse.ArgumentParser(description="Monitor YOLO training")
    parser.add_argument("--watch", action="store_true", help="refresh until training ends")
    parser.add_argument("--every", type=int, default=30, help="seconds between refreshes")
    args = parser.parse_args()

    if not args.watch:
        report()
        return
    while True:
        print("\033[2J\033[H", end="")          # clear the terminal
        if report():
            print("\nTraining finished.")
            break
        print(f"\nRefreshing every {args.every}s - Ctrl+C to stop watching (training keeps running).")
        time.sleep(args.every)


if __name__ == "__main__":
    main()
