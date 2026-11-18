"""
Shared pieces for the training scripts: where things live, the dataset file,
and a small console logger that prints one readable line per epoch.

The Mini Project trains on the same Roboflow PPE dataset as the Day-31 coding
practice (anjrh81/ppe-asmsk v2, CC BY 4.0). It is NOT copied here, so the repo
does not carry the same 49 MB twice. By default the scripts look for it in
../Coding_Practice/Dataset; pass --dataset or set PPE_DATASET to use another copy.
"""

import os
import sys
import time
from pathlib import Path

import yaml

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAINING_DIR.parent

# make "import ppe_detector" work when a script in training/ is run directly
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ppe_detector.settings import CLASS_NAMES  # noqa: E402  (needs the path fix above)

DEFAULT_DATASET = PROJECT_DIR.parent / "Coding_Practice" / "Dataset"
DATASET_YAML = PROJECT_DIR / "dataset.yaml"

RESULTS_DIR = PROJECT_DIR / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
TRAINING_RUNS_DIR = RESULTS_DIR / "training"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

MODELS_DIR = PROJECT_DIR / "models"
FINAL_RUN_NAME = "ppe_final"
FINAL_WEIGHTS = MODELS_DIR / "ppe_yolo11n.pt"
FINAL_ONNX = MODELS_DIR / "ppe_yolo11n.onnx"
MODEL_INFO = MODELS_DIR / "model_info.json"


def find_dataset(cli_value=None):
    """--dataset argument  >  PPE_DATASET environment variable  >  ../Coding_Practice/Dataset"""
    candidate = Path(cli_value or os.environ.get("PPE_DATASET") or DEFAULT_DATASET).expanduser().resolve()
    if not (candidate / "train" / "images").is_dir():
        sys.exit(f"Dataset not found at {candidate}\n"
                 f"Expected the Roboflow YOLO export (train/valid/test folders). "
                 f"Use --dataset <folder> or set PPE_DATASET.")
    return candidate


def write_dataset_yaml(dataset_dir):
    """
    Roboflow's data.yaml says 'train: ../train/images' and names the classes
    '0'..'3'. Ultralytics resolves those paths from the dataset root, so they
    point one level too high. We write our own file with an absolute path and
    readable class names, and leave the downloaded file untouched.
    """
    config = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with DATASET_YAML.open("w", encoding="utf-8") as f:
        f.write("# Written automatically by training/common.py - do not edit by hand\n")
        yaml.safe_dump(config, f, sort_keys=False)
    return DATASET_YAML


def pick_device(requested=None):
    import torch
    if requested not in (None, ""):
        return requested
    return 0 if torch.cuda.is_available() else "cpu"


def device_label(device):
    import torch
    return "CPU" if device == "cpu" else f"GPU {torch.cuda.get_device_name(int(device))}"


def epoch_logger(total_epochs, prefix=""):
    """
    Callback for model.add_callback("on_fit_epoch_end", ...).
    Prints losses, validation metrics and time left after every epoch.
    """
    start = time.time()

    def losses(trainer):
        # Ultralytics keeps the running loss as a tensor in some versions, a dict in others
        loss = trainer.tloss
        if loss is None:
            return [0.0, 0.0, 0.0]
        if isinstance(loss, dict):
            values = list(loss.values())
        else:
            values = loss.tolist() if hasattr(loss, "tolist") else list(loss)
        return ([float(v) for v in values] + [0.0, 0.0, 0.0])[:3]

    def report(trainer):
        epoch = trainer.epoch + 1
        m = trainer.metrics
        box, cls, dfl = losses(trainer)
        elapsed = time.time() - start
        left = elapsed / epoch * max(0, total_epochs - epoch)
        # Ultralytics re-validates best.pt after the last epoch and calls this hook again
        tag = "[final check, best.pt]" if epoch > total_epochs else f"[epoch {epoch:>3}/{total_epochs}]"
        print(f"{prefix}{tag} loss box {box:.3f} cls {cls:.3f} dfl {dfl:.3f} | "
              f"P {m.get('metrics/precision(B)', 0):.3f} R {m.get('metrics/recall(B)', 0):.3f} "
              f"mAP50 {m.get('metrics/mAP50(B)', 0):.3f} mAP50-95 {m.get('metrics/mAP50-95(B)', 0):.3f} | "
              f"{elapsed / 60:5.1f} min, ~{left / 60:4.1f} min left", flush=True)

    def on_fit_epoch_end(trainer):
        try:                      # a logging hiccup must never stop a long run
            report(trainer)
        except Exception as error:
            print(f"[epoch log skipped: {error}]", flush=True)

    return on_fit_epoch_end
