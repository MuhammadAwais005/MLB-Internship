# PPE Detection with YOLO

This project trains a YOLO object detector on the Roboflow PPE dataset in this
folder. It includes training, held-out test evaluation, and image prediction.

## Dataset

`data.yaml` defines seven classes and the existing splits:

- `train/images` and `train/labels`
- `valid/images` and `valid/labels`
- `test/images` and `test/labels`

Keep the image and label filenames aligned when adding or replacing data. Labels
must use YOLO detection format: `class_id x_center y_center width height`, with
normalized coordinates.

## Setup (Windows)

From this folder, create and activate a virtual environment, then install the
project dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The scripts use the Ultralytics package. Its required runtime packages,
including PyTorch and OpenCV, are installed as dependencies. On systems without
a supported GPU, training runs on CPU and can be much slower.

## Train

```powershell
python train.py
```

The default starts from `yolo11n.pt` (downloaded by Ultralytics if needed), runs
100 epochs at 640 pixels, and saves checkpoints and logs under
`runs/ppe-detection/`. Common options:

```powershell
python train.py --epochs 50 --batch 8 --device 0
python train.py --model path\to\weights.pt --name experiment-2
```

Use `--device cpu` to force CPU training or omit `--device` for automatic
selection.

## Evaluate

After training, evaluate the held-out test split:

```powershell
python evaluate.py
```

Metrics and plots are written under `runs/test-evaluation/`. To evaluate a
different checkpoint, pass `--weights path\to\best.pt`.

## Predict

```powershell
python predict.py --source path\to\image.jpg
python predict.py --source path\to\images
```

Annotated predictions are saved under `runs/predictions/`. Pass `--weights` to
use a checkpoint other than the default `runs/ppe-detection/weights/best.pt`.
