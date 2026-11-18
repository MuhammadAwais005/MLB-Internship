# Day 31 – Training a YOLO Model on a Roboflow Dataset (PPE Detection)

A complete, end-to-end object-detection workflow: take a dataset from Roboflow
Universe in YOLO format, understand it, fine-tune **YOLO11n** on it, watch the
training, evaluate it honestly and run it on unseen test images.

The dataset is a **Personal Protective Equipment (PPE)** set: construction workers
with helmets, safety vests and safety boots. The trained model reaches
**mAP50 0.962 / mAP50-95 0.769 on the test split** and runs at about
**40 ms per image on a plain CPU**.

| Task | Where it is done |
|---|---|
| 1. Download a dataset from Roboflow Universe (YOLO format) | `Dataset/` (Roboflow export, see below) |
| 2. Explore the dataset structure | `01_explore_dataset.py` |
| 3. Understand the `data.yaml` file | `01_explore_dataset.py` + section 3 of this README |
| 4. Train a YOLO model | `02_train.py` |
| 5. Monitor the training process | live epoch log in `02_train.py` + `03_monitor_training.py` |
| 6. Evaluate the trained model | `04_evaluate.py` |
| 7. Run inference on at least 10 test images | `05_inference.py` (12 images by default) |

---

## 1. Dataset

| | |
|---|---|
| Source | Roboflow Universe – workspace `anjrh81`, project `ppe-asmsk`, version 2 |
| URL | https://universe.roboflow.com/anjrh81/ppe-asmsk/dataset/2 |
| Licence | CC BY 4.0 |
| Format | YOLO (one `.txt` label file per image) |
| Images | 786 in total – 551 train / 157 valid / 78 test (70 / 20 / 10 %) |
| Image size | 640 × 640 (resized by Roboflow) |
| Objects | 3,565 bounding boxes |

### Class names

Roboflow exported the classes only as `'0', '1', '2', '3'`, and the project page
gives no description. I identified each class by cropping and looking at its boxes
(`outputs/exploration/class_examples.jpg`):

| ID | Class | Train | Valid | Test | Median box size |
|---|---|---|---|---|---|
| 0 | helmet | 509 | 122 | 58 | 62 px (9.7 % smaller than 32 px) |
| 1 | boots  | 832 | 216 | 112 | 53 px (7.3 % smaller than 32 px) |
| 2 | person | 528 | 143 | 65 | 321 px |
| 3 | vest   | 689 | 205 | 86 | 147 px |

The readable names live in one place – `CLASS_NAMES` in `config.py` – and are
written into `dataset.yaml`, so every plot and prediction shows real names instead
of numbers.

---

## 2. Folder structure

```
Coding_Practice/
├── Dataset/                    # Roboflow export (unchanged)
│   ├── data.yaml
│   ├── train/ images/ labels/
│   ├── valid/ images/ labels/
│   └── test/  images/ labels/
├── config.py                   # all paths, class names, colours; writes dataset.yaml
├── dataset.yaml                # corrected copy of data.yaml (generated automatically)
├── 01_explore_dataset.py       # structure, data.yaml, counts, label checks, leakage check
├── 02_train.py                 # fine-tunes YOLO11n, prints one clean line per epoch
├── 03_monitor_training.py      # reads results.csv, diagnoses and plots the training
├── 04_evaluate.py              # valid / test / clean-test metrics, confusion matrices
├── 05_inference.py             # predictions on test images + simple PPE compliance check
├── requirements.txt
├── models/
│   └── ppe_yolo11n_best.pt     # the trained model (5.3 MB)
└── outputs/
    ├── exploration/            # charts and sample images of the dataset
    ├── training/               # training_log.txt + Ultralytics run folder (curves, results.csv)
    ├── evaluation/             # metrics.json, plots per split, mAP comparison
    └── inference/              # annotated predictions, grid, CSV reports
```

---

## 3. Understanding `data.yaml`

This is the file Roboflow ships with the dataset:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 4
names: ['0', '1', '2', '3']

roboflow:
  workspace: anjrh81
  project: ppe-asmsk
  version: 2
  license: CC BY 4.0
  url: https://universe.roboflow.com/anjrh81/ppe-asmsk/dataset/2
```

| Key | Meaning |
|---|---|
| `train` | folder of training images. YOLO finds each label by swapping `images` for `labels` in the path and `.jpg` for `.txt`. |
| `val` | images checked after every epoch. They decide which epoch becomes `best.pt`. |
| `test` | images held back until the very end, for the final honest score. |
| `nc` | number of classes. It must match the number of entries in `names`. |
| `names` | class names in ID order: line `2 0.51 0.43 0.20 0.66` in a label file means "class 2". |
| `roboflow` | where the data came from. YOLO ignores this block. |

Each label line is `class x_center y_center width height`, all normalised to
0–1 relative to the image size.

**Two problems with the original file, and how they are fixed:**

1. **Paths.** `../train/images` assumes the file sits one folder below the data.
   Ultralytics resolves relative paths from the dataset root, so it looks for
   `Coding_Practice/train/images` and cannot find the images.
2. **Names.** `'0'…'3'` make every plot and prediction unreadable.

Rather than edit the downloaded file, `config.write_dataset_yaml()` writes a
corrected `dataset.yaml` next to the scripts, with an absolute `path:` and real
class names. Every script regenerates it on start, so the project still works if
you move the folder.

---

## 4. Exploring the dataset – findings

Run `python 01_explore_dataset.py`. Results are in `outputs/exploration/`.

- **Labels are clean.** Every image has a label file, there are no empty or
  malformed lines and no coordinates outside 0–1. There is one duplicate box in
  train and one in test, which is harmless.
- **Classes are reasonably balanced.** Boots are the most frequent class (two per
  worker). No class is rare enough to need re-weighting.
- **Helmets and boots are small.** Their median size is about 55–60 px on a 640 px
  image, and close to 10 % are under 32 px. These are the hardest objects for a
  nano model, especially at a lower training resolution.
- **There is data leakage between splits.** Roboflow names augmented copies
  `<original>_jpg.rf.<hash>.jpg`. The 551 training files come from only 419
  original photos, and **40 of the 73 original test photos also appear (augmented)
  in train**. Valid has the same problem (57 of 145). A model can score well on
  those test images partly from memory, so the evaluation also reports a
  **clean test set**: the 32 test images whose original photo never appears in train.

---

## 5. Training

```
python 02_train.py --epochs 35 --imgsz 416 --patience 10
```

| Setting | Value | Why |
|---|---|---|
| Base model | `yolo11n.pt` (COCO pre-trained) | smallest YOLO11 model. Transfer learning needs far less data. |
| Epochs | 35 (early stopping patience 10) | enough to converge on a CPU in about an hour |
| Image size | 416 | 640 costs about 5.5 min/epoch on a 2-core CPU, 416 about 1.8 min |
| Batch | 16 | |
| Optimizer / LR | `auto` → Ultralytics picked AdamW, lr 0.00125 | chosen automatically for a short run |
| Augmentation | Ultralytics defaults. Mosaic is switched off for the last 10 epochs. | |
| Seed | 0 | reproducible run |
| Hardware | 2 CPU cores, no GPU | total time **62 minutes** |

**Why epoch 1 already scored mAP50 0.74:** our class `person` has the same name
as COCO class `person`. Ultralytics noticed this and copied that pre-trained
output row into our head ("Remapped 1/4 cls head rows"), so people were detected
from the first epoch. The other three classes were learned from scratch.

The best weights are copied to `models/ppe_yolo11n_best.pt`.

**Defaults on your own PC.** Without arguments, `02_train.py` uses 50 epochs at
640 px and selects the GPU automatically if one exists. On an NVIDIA GPU that takes
a few minutes and should give better scores than this CPU run.

---

## 6. Monitoring the training

- **During training**, a callback prints one line per epoch:
  ```
  [epoch  12/35] loss box 1.014 cls 0.704 dfl 1.095 | P 0.912 R 0.858 mAP50 0.926 mAP50-95 0.677 |  21.5 min, ~41.2 min left
  ```
  The full cleaned log is in `outputs/training/training_log.txt`.
- **During or after training**, `python 03_monitor_training.py` reads
  `results.csv`. It prints a summary table, flags problems (losses flat, validation
  loss rising, overfitting) and saves `training_progress.png`. With `--watch` it
  refreshes every 30 seconds while a run is in progress.
- Ultralytics also writes TensorBoard logs. Run `tensorboard --logdir outputs/training`
  to view them.

**What the curves show** (`outputs/training/ppe_yolo11n/training_progress.png`):

| | Epoch 1 | Epoch 35 |
|---|---|---|
| train box loss | 1.309 | 0.732 |
| train cls loss | 1.819 | 0.419 |
| val box loss | 1.149 | 0.879 |
| val cls loss | 2.608 | 0.547 |
| mAP50 | 0.739 | 0.954 |
| mAP50-95 | 0.497 | 0.745 |

The training and validation losses fall together, so there is **no overfitting**.
mAP50-95 was still climbing at the last epoch, which means the best epoch was the
final one. The model had not finished learning: more epochs would help.

---

## 7. Evaluation

Run `python 04_evaluate.py`. The image size is read from the training run, and the
confidence threshold is 0.001 (the standard for mAP). Results are in
`outputs/evaluation/`.

### Overall

| Split | Images | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Validation | 157 | 0.931 | 0.908 | 0.954 | 0.745 |
| **Test** | 78 | **0.965** | **0.895** | **0.962** | **0.769** |
| Clean test (no leakage) | 32 | 0.954 | 0.945 | 0.964 | 0.790 |

### Per class – test split

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| helmet | 0.974 | 0.914 | 0.957 | 0.726 |
| boots  | 0.962 | 0.920 | 0.979 | 0.729 |
| person | 0.939 | 0.877 | 0.939 | 0.845 |
| vest   | 0.987 | 0.869 | 0.973 | 0.776 |

### Reading the results

- **Leakage did not inflate the score.** The clean test images score the same as
  the full test set (mAP50-95 0.790 vs 0.769). With only 32 images that difference
  is noise, but there is no sign that the model just memorised the duplicated photos.
- **The model rarely confuses the classes.** In the normalised confusion matrix the
  diagonal is helmet 0.97, boots 0.99, person 0.98 and vest 1.00, and it never
  calls a helmet a vest or similar. Its errors are of two kinds: a few **missed
  objects** (3 % of helmets, 2 % of persons) and **false alarms on background**,
  mostly boots (39 % of false positives) and vests (25 %).
- **mAP50 vs mAP50-95.** Boxes are almost always in the right place (mAP50 ≈ 0.96)
  but less tight on small objects. Helmets and boots have the lowest mAP50-95. That
  is the cost of training at 416 px.
- **Speed:** about 41 ms per image on CPU (1 ms pre-processing, 0.3 ms post-processing).

---

## 8. Inference on test images

```
python 05_inference.py                     # 12 random test images (fixed seed)
python 05_inference.py --num 20 --conf 0.4
python 05_inference.py --source path/to/your/photos
```

Outputs in `outputs/inference/`:

| File | Content |
|---|---|
| `predictions/*.jpg` | each image with coloured, labelled boxes |
| `inference_grid.jpg` | all predictions on one sheet |
| `ground_truth_vs_prediction.jpg` | side-by-side comparison with the real labels |
| `detections.csv` | every box: image, class, confidence, coordinates |
| `ppe_check.csv` | per worker: helmet / vest / boots present, status |

**Results on 12 test images (conf 0.25):** 54 detections (12 person, 12 vest,
23 boots, 7 helmet) at an average of **45.5 ms per image**.

The PPE check counts a helmet, vest or boot as belonging to a worker when its
centre falls inside that worker's box. Of the 12 workers, **7 are fully equipped
and 5 have no helmet**. I checked all 5 against the images and they really are
bare-headed. One rotated image was handled correctly. Most of the sampled images
come from the same rooftop scene, so these 12 images are a demonstration. The
evaluation tables above are the real measure of quality.

---

## 9. How to run it on your PC

Python **3.11.9**.

```bash
cd E:\MLB-Internship\Day-31\Coding_Practice
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# NVIDIA GPU: install the CUDA build of torch first, from https://pytorch.org

python 01_explore_dataset.py
python 02_train.py                 # GPU: 50 epochs @ 640 | CPU: add --epochs 35 --imgsz 416
python 03_monitor_training.py      # in a second terminal: add --watch
python 04_evaluate.py
python 05_inference.py
```

The trained model is already in `models/`, so you can run `04_evaluate.py` and
`05_inference.py` straight away without retraining. `yolo11n.pt` (the COCO starting
weights) downloads automatically the first time you train.

---

## 10. What I would improve next

1. **Train at 640 px** on a GPU. This matters most for the small helmets and boots.
2. **Train for longer** (80–100 epochs). mAP was still rising at epoch 35.
3. **Try `yolo11s.pt`**, about 3× larger, if the speed budget allows.
4. **Re-split the data by original photo** so that no augmented copy of a test
   image is in train. This gives a fully honest test score.
5. **Add more varied background images** (sites with no workers, piles of
   material) to reduce false-alarm boots and vests, and fix the few images where a
   worker is not labelled as `person`.
