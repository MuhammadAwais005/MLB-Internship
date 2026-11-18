"""
02_train.py - Train a YOLO11 model on the PPE dataset
=====================================================

Fine-tunes a COCO-pretrained YOLO11n (nano) detector on the 4 PPE classes.
Transfer learning: the network already knows edges, shapes and people from
COCO, so 551 training images are enough to teach it helmets, boots and vests.

Run:
    python 02_train.py                          # GPU if available, 640 px, 50 epochs
    python 02_train.py --epochs 35 --imgsz 416  # the CPU-friendly run used for the results

Everything is saved to outputs/training/ppe_yolo11n/ and the best weights are
copied to models/ppe_yolo11n_best.pt. While it runs, watch progress with:
    python 03_monitor_training.py
"""

import argparse
import shutil
import time

import torch
from ultralytics import YOLO

from config import BEST_MODEL, MODEL_DIR, RUN_NAME, TRAIN_DIR, write_dataset_yaml


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11 on the PPE dataset")
    p.add_argument("--model", default="yolo11n.pt", help="pretrained weights to start from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640, help="training image size")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=15, help="stop early after N epochs without improvement")
    p.add_argument("--workers", type=int, default=2, help="data-loading processes")
    p.add_argument("--device", default=None, help="'0' for first GPU, 'cpu', or empty for automatic")
    return p.parse_args()


def make_epoch_logger(total_epochs):
    """
    A callback that prints one clear line per epoch: losses, validation
    metrics, time per epoch and an estimate of the time left.
    """
    state = {"start": time.time()}

    def training_losses(trainer):
        """box / cls / dfl loss, whether Ultralytics stores them as a tensor or a dict."""
        loss = trainer.tloss
        if loss is None:
            return [0.0, 0.0, 0.0]
        if isinstance(loss, dict):
            return [float(v) for v in loss.values()][:3]
        values = loss.tolist() if hasattr(loss, "tolist") else list(loss)
        return [float(v) for v in values][:3]

    def on_fit_epoch_end(trainer):
        # a logging problem must never stop a long training run
        try:
            print_epoch(trainer)
        except Exception as error:
            print(f"[epoch log skipped: {error}]", flush=True)

    def print_epoch(trainer):
        epoch = trainer.epoch + 1
        m = trainer.metrics
        loss = (training_losses(trainer) + [0.0, 0.0, 0.0])[:3]
        elapsed = time.time() - state["start"]
        eta = elapsed / epoch * max(0, total_epochs - epoch)
        # after the last epoch Ultralytics re-validates best.pt, which calls this hook once more
        tag = "[final check, best.pt]" if epoch > total_epochs else f"[epoch {epoch:>3}/{total_epochs}]"
        print(f"{tag} "
              f"loss box {loss[0]:.3f} cls {loss[1]:.3f} dfl {loss[2]:.3f} | "
              f"P {m.get('metrics/precision(B)', 0):.3f} R {m.get('metrics/recall(B)', 0):.3f} "
              f"mAP50 {m.get('metrics/mAP50(B)', 0):.3f} mAP50-95 {m.get('metrics/mAP50-95(B)', 0):.3f} | "
              f"{elapsed / 60:5.1f} min, ~{eta / 60:4.1f} min left", flush=True)

    return on_fit_epoch_end


def main():
    args = parse_args()
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    data_yaml = write_dataset_yaml()

    print(f"Dataset config : {data_yaml}")
    print(f"Model          : {args.model}  (COCO-pretrained, fine-tuned)")
    print(f"Device         : {'GPU ' + torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}")
    print(f"Settings       : {args.epochs} epochs, imgsz {args.imgsz}, batch {args.batch}, patience {args.patience}\n")

    model = YOLO(args.model)
    model.add_callback("on_fit_epoch_end", make_epoch_logger(args.epochs))

    start = time.time()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=device,
        project=str(TRAIN_DIR),
        name=RUN_NAME,
        exist_ok=True,
        seed=0,
        cache=True,          # keep resized images in RAM -> faster epochs
        plots=True,          # results.png, PR curves, confusion matrix, label stats
        verbose=False,
    )
    minutes = (time.time() - start) / 60

    run_dir = TRAIN_DIR / RUN_NAME
    best = run_dir / "weights" / "best.pt"
    MODEL_DIR.mkdir(exist_ok=True)
    shutil.copy(best, BEST_MODEL)

    print(f"\nTraining finished in {minutes:.0f} min")
    print(f"Run folder  : {run_dir}")
    print(f"Best weights: {BEST_MODEL}")
    print("Next: python 04_evaluate.py")


if __name__ == "__main__":
    main()
