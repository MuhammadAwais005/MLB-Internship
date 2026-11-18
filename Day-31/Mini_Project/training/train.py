"""
train.py - Train the final PPE detector
=======================================

Fine-tunes COCO-pretrained YOLO11n on the PPE dataset with the settings chosen
from the experiments (experiments.py, results in the README):

    image size 512   - best mAP50 of the five runs; helmets and boots are small,
                       so extra pixels help (cost: ~1.5x longer epochs than 416)
    batch 16         - batch 8 scored lower after the same number of epochs
    default augmentation - with augmentation off the model memorised the training
                       images (training loss halved) and stopped improving
    more epochs      - every run with augmentation was still improving at epoch 10

Run:
    python training/train.py                         # 512 px, 50 epochs (GPU if available)
    python training/train.py --epochs 40 --patience 10   # the CPU run used for the results
    python training/train.py --epochs 80 --imgsz 640 # bigger budget on a GPU

Output:
    results/training/ppe_final/   Ultralytics run: results.csv, curves, confusion matrix
    models/ppe_yolo11n.pt         best weights
    models/model_info.json        settings and validation score, shown in the app
"""

import argparse
import csv
import json
import shutil
import time

from common import (FINAL_RUN_NAME, FINAL_WEIGHTS, MODEL_INFO, MODELS_DIR, TRAINING_RUNS_DIR,
                    device_label, epoch_logger, find_dataset, pick_device, write_dataset_yaml)


def parse_args():
    p = argparse.ArgumentParser(description="Train the final PPE model")
    p.add_argument("--model", default="yolo11n.pt", help="pretrained weights to start from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dataset", default=None, help="Roboflow dataset folder")
    p.add_argument("--device", default=None, help="'0' = first GPU, 'cpu', empty = automatic")
    return p.parse_args()


def best_epoch(results_csv):
    with results_csv.open(encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    return {
        "epoch": int(float(best["epoch"])),
        "epochs_run": len(rows),
        "precision": round(float(best["metrics/precision(B)"]), 4),
        "recall": round(float(best["metrics/recall(B)"]), 4),
        "mAP50": round(float(best["metrics/mAP50(B)"]), 4),
        "mAP50-95": round(float(best["metrics/mAP50-95(B)"]), 4),
    }


def main():
    args = parse_args()
    from ultralytics import YOLO

    dataset = find_dataset(args.dataset)
    data_yaml = write_dataset_yaml(dataset)
    device = pick_device(args.device)

    print(f"Dataset  : {dataset}")
    print(f"Device   : {device_label(device)}")
    print(f"Settings : {args.model}, {args.epochs} epochs, imgsz {args.imgsz}, batch {args.batch}, "
          f"patience {args.patience}\n")

    model = YOLO(args.model)
    model.add_callback("on_fit_epoch_end", epoch_logger(args.epochs))
    start = time.time()
    model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, workers=args.workers, device=device, seed=0, cache=True,
        project=str(TRAINING_RUNS_DIR), name=FINAL_RUN_NAME, exist_ok=True, plots=True, verbose=False,
    )
    minutes = (time.time() - start) / 60

    run_dir = TRAINING_RUNS_DIR / FINAL_RUN_NAME
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(run_dir / "weights" / "best.pt", FINAL_WEIGHTS)

    info = {
        "model": "YOLO11n fine-tuned from COCO weights",
        "dataset": "Roboflow Universe anjrh81/ppe-asmsk v2 (CC BY 4.0)",
        "classes": ["helmet", "boots", "person", "vest"],
        "training": {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
                     "patience": args.patience, "device": device_label(device),
                     "minutes": round(minutes, 1)},
        "validation_best": best_epoch(run_dir / "results.csv"),
    }
    MODEL_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")

    v = info["validation_best"]
    print(f"\nFinished in {minutes:.0f} min - best epoch {v['epoch']}/{v['epochs_run']}: "
          f"mAP50 {v['mAP50']:.3f}, mAP50-95 {v['mAP50-95']:.3f}")
    print(f"Weights saved to {FINAL_WEIGHTS}")
    print("Next: python training/evaluate.py")


if __name__ == "__main__":
    main()
