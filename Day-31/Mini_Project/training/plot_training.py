"""
plot_training.py - Monitor / review the final training run
==========================================================

Reads results/training/ppe_final/results.csv (written by Ultralytics after
every epoch) and draws the losses and validation scores per epoch. The short
10-epoch experiment at the same image size (C_imgsz512) is drawn on top, which
shows directly what training for longer bought us.

Works while train.py is still running, so it doubles as a progress monitor:
    python training/plot_training.py
    python training/plot_training.py --watch 60     # redraw every 60 s

Output: results/training/training_curves.png
"""

import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EXPERIMENTS_DIR, FINAL_RUN_NAME, TRAINING_RUNS_DIR

RESULTS = TRAINING_RUNS_DIR / FINAL_RUN_NAME / "results.csv"
SHORT_RUN = EXPERIMENTS_DIR / "C_imgsz512" / "results.csv"
OUTPUT = TRAINING_RUNS_DIR / "training_curves.png"


def read(csv_path):
    with csv_path.open(encoding="utf-8") as f:
        return [{k.strip(): float(v) for k, v in row.items() if v not in (None, "")} for row in csv.DictReader(f)]


def column(rows, key):
    return [r["epoch"] for r in rows], [r[key] for r in rows]


def draw(rows, short):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for key, style, label in (("train/box_loss", "-", "train box"), ("val/box_loss", "--", "val box"),
                              ("train/cls_loss", "-", "train cls"), ("val/cls_loss", "--", "val cls")):
        axes[0].plot(*column(rows, key), style, label=label)
    axes[0].set_title("Losses (lower is better)")

    for ax, key, title in ((axes[1], "metrics/mAP50(B)", "mAP50 (validation)"),
                           (axes[2], "metrics/mAP50-95(B)", "mAP50-95 (validation)")):
        ax.plot(*column(rows, key), marker="o", ms=3, label="final run")
        if short:
            ax.plot(*column(short, key), marker="o", ms=3, alpha=0.7, label="10-epoch experiment")
        best = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
        ax.axvline(best["epoch"], color="gray", ls=":", lw=1, label=f"best epoch {int(best['epoch'])}")
        ax.set_title(title)
    axes[1].axhline(0.80, color="red", ls="--", lw=1, label="target 0.80")

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Final training run - {len(rows)} epochs (512 px, batch 16, default augmentation)")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=130)
    plt.close(fig)


def report(rows):
    last, best = rows[-1], max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
    print(f"epochs done : {len(rows)}")
    print(f"last epoch  : mAP50 {last['metrics/mAP50(B)']:.3f}  mAP50-95 {last['metrics/mAP50-95(B)']:.3f}")
    print(f"best epoch  : {int(best['epoch'])}  mAP50 {best['metrics/mAP50(B)']:.3f}  "
          f"mAP50-95 {best['metrics/mAP50-95(B)']:.3f}")
    # simple health checks a human would do by eye
    if len(rows) >= 6:
        recent = [r["val/box_loss"] + r["val/cls_loss"] for r in rows[-5:]]
        early_best = min(r["val/box_loss"] + r["val/cls_loss"] for r in rows[:-5])
        if min(recent) > early_best * 1.05:
            print("warning     : validation loss is rising again -> overfitting, best.pt is still safe")
        elif rows[-1]["epoch"] - best["epoch"] >= 5:
            print("note        : no improvement for 5+ epochs -> the model is close to converged")
        else:
            print("status      : still improving")
    print(f"plot        : {OUTPUT}")


def main():
    p = argparse.ArgumentParser(description="Plot the final training run")
    p.add_argument("--watch", type=int, default=0, help="redraw every N seconds (0 = once)")
    args = p.parse_args()

    while True:
        if RESULTS.exists():
            rows = read(RESULTS)
            if rows:
                draw(rows, read(SHORT_RUN) if SHORT_RUN.exists() else None)
                report(rows)
        else:
            print(f"Waiting for {RESULTS} ...")
        if not args.watch:
            break
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    main()
