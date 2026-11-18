# Day 31 – Mini Project: Custom Object Detection System (PPE)

A complete custom object-detection system for **construction-site safety**. A YOLO11n
model is trained on a Roboflow Universe PPE dataset to find **helmets, vests, boots and
workers**. It is evaluated honestly, run on test images, and served through a
**Streamlit web app**. In the app you can upload an image or a video, see every
detection with its confidence score, check which workers are missing equipment, and
download the processed result.

| | Result |
|---|---|
| **mAP@50, test set** | **0.972** (target ≥ 0.80 ✅) |
| mAP@50-95, test set | 0.776 |
| mAP@50, test photos never seen in training (clean test) | 0.976 |
| Speed of the deployed ONNX model | ~50 ms per image on a CPU |
| Model size | 10.6 MB (ONNX) |

| Requirement | Where |
|---|---|
| Train a custom YOLO model | `training/train.py` (+ `training/experiments.py`) |
| Evaluate the model | `training/evaluate.py` |
| Run inference on test images | `training/predict.py` |
| Display detected objects with confidence scores | `predict.py` console output and saved images, and the app |
| Save the prediction results | `results/predictions/` (images, CSV, JSON), plus downloads in the app |
| mAP@50 ≥ 80 % | 0.972 on test, see section 5 |
| Experiments with epochs, batch size, image size, augmentation | section 4 |
| Streamlit app: upload image/video, run model, view, download | `app.py`, section 7 |

---

## 1. Dataset

The same PPE dataset as the Day-31 coding practice:

| | |
|---|---|
| Source | Roboflow Universe – `anjrh81/ppe-asmsk`, version 2 – https://universe.roboflow.com/anjrh81/ppe-asmsk/dataset/2 |
| Licence | CC BY 4.0 |
| Format | YOLO (one `.txt` label file per image) |
| Images | 786 – 551 train / 157 valid / 78 test, all 640 × 640 |
| Classes | 0 helmet · 1 boots · 2 person · 3 vest (Roboflow only exports the numbers 0–3. The meaning of each was identified from the labelled boxes during the coding practice.) |

The dataset is **not copied** into this folder, so the repository doesn't carry the same
49 MB twice. The training scripts read it from `../Coding_Practice/Dataset`. To use
another copy, pass `--dataset <folder>` or set the `PPE_DATASET` environment variable.
The app does not need the dataset at all.

Two problems in Roboflow's `data.yaml` are fixed automatically. Every training script
writes its own `dataset.yaml` with an absolute path (Roboflow's `../train/images` points
one folder too high for Ultralytics) and readable class names.

**Data leakage:** Roboflow put augmented copies of 40 of the 73 test photos into the
training set. That's why every score is also reported on a **clean test** set: the 32
test images whose original photo never appears in train or valid.

---

## 2. Project structure

```
Mini_Project/
├── app.py                    # Streamlit web app
├── requirements.txt          # app dependencies - what Streamlit Cloud installs (no PyTorch)
├── requirements-train.txt    # + Ultralytics / PyTorch for training and evaluation
├── .streamlit/config.toml    # upload limit, theme (used when you run the app locally)
├── ppe_detector/             # the inference package used by the app and predict.py
│   ├── detector.py           #   ONNX Runtime YOLO detector: letterbox, decode, NMS
│   ├── visualize.py          #   boxes + "class 0.93" labels
│   ├── compliance.py         #   PPE check: which worker wears what
│   ├── video.py              #   frame-by-frame video processing -> H.264 MP4
│   └── settings.py           #   class names and colours
├── training/
│   ├── common.py             # paths, dataset.yaml, epoch logger
│   ├── experiments.py        # 5 controlled experiments (image size, batch, augmentation)
│   ├── train.py              # final training run
│   ├── plot_training.py      # training curves / live monitor
│   ├── evaluate.py           # valid / test / clean-test metrics, target check
│   ├── export_onnx.py        # export to ONNX + box-by-box consistency check
│   └── predict.py            # inference on test images, saves all results
├── models/
│   ├── ppe_yolo11n.pt        # trained weights (PyTorch, 5.5 MB)
│   ├── ppe_yolo11n.onnx      # same model for the app (10.6 MB)
│   └── model_info.json       # settings + scores, shown in the app
├── samples/                  # 8 sample photos + a 12 s demo video for the app
└── results/
    ├── experiments/          # summary.csv/json, experiment_curves.png, experiment_summary.png
    ├── training/             # training_curves.png + Ultralytics run (results.csv, plots)
    ├── evaluation/           # metrics.json, per_class_map.png, plots per set (+ *_onnx)
    └── predictions/          # 78 annotated images, detections.csv/json, ppe_report.csv, grid
```

---

## 3. How to run

Python **3.11.9**.

```bash
cd E:\MLB-Internship\Day-31\Mini_Project
python -m venv .venv
.venv\Scripts\activate

# just the app (the trained model is included)
pip install -r requirements.txt
streamlit run app.py

# the full training pipeline
pip install -r requirements-train.txt     # NVIDIA GPU: install CUDA torch first (pytorch.org)
python training/experiments.py            # optional, ~90 min on CPU
python training/train.py                  # GPU: 50 epochs @ 512 | CPU run used: --epochs 40 --patience 10
python training/plot_training.py          # curves (add --watch 60 while training runs)
python training/evaluate.py
python training/export_onnx.py
python training/predict.py
```

---

## 4. Experiments and observations

Every experiment starts from the same COCO weights with the same seed and changes **one
thing** compared with the baseline. Each gets the same 10-epoch budget, on a 2-core CPU.

| Run | Change | Image size | Batch | Augmentation | mAP@50 | mAP@50-95 | min / epoch |
|---|---|---|---|---|---|---|---|
| A_baseline | – | 416 | 16 | default* | 0.950 | **0.712** | 1.86 |
| B_imgsz320 | smaller images | 320 | 16 | default* | 0.940 | 0.694 | **0.97** |
| C_imgsz512 | larger images | 512 | 16 | default* | **0.957** | 0.710 | 2.79 |
| D_batch8 | smaller batch | 416 | 8 | default* | 0.942 | 0.697 | 1.61 |
| E_no_aug | augmentation off | 416 | 16 | none | 0.932 | 0.709 | 1.79 |

\* *Ultralytics switches **mosaic** off for the last 10 epochs (`close_mosaic=10`). With a
10-epoch budget, mosaic was therefore off for the whole run. "default" here means flips,
colour (HSV), scale, translate and random erasing, without mosaic. I only noticed this
when the final run started much lower at epoch 1 (0.46 vs 0.81). That difference came
from mosaic, not from a bug.*

Charts: `results/experiments/experiment_curves.png` and `experiment_summary.png`.

**What each change did**

- **Image size: the biggest lever.** 320 → 416 → 512 gave mAP@50 0.940 → 0.950 → 0.957.
  Helmets and boots are small (median ~55–60 px on a 640 px photo), and shrinking the
  image takes pixels away from exactly those objects. The cost is time: 512 is about
  1.5× slower per epoch than 416, and 320 is twice as fast.
- **Batch size: 16 beat 8.** Batch 8 makes twice as many weight updates per epoch, so it
  learned faster in the first epochs (0.87 vs 0.80 at epoch 2). By epoch 10 the noisier
  gradients left it behind: 0.942 vs 0.950.
- **Augmentation: needed for generalisation.** Without augmentation the model learned
  fastest at first (0.92 at epoch 4), then **stayed at about 0.933 from epoch 7 on**. Its
  training box loss fell to **0.42**, less than half the baseline's 0.89, while the
  validation loss stayed the same. That is memorisation, not learning. Every run with
  augmentation was still improving at epoch 10.
- **Epochs: more helped, mostly in box quality.** The final run (512 px, 40 epochs,
  mosaic on for epochs 1–30) is compared with the 10-epoch run at the same size in
  `results/training/training_curves.png`:

  | Epoch | 1 | 5 | 10 | 20 | 30 | 40 |
  |---|---|---|---|---|---|---|
  | mAP@50 | 0.462 | 0.811 | 0.890 | 0.939 | 0.954 | **0.958** |
  | mAP@50-95 | 0.293 | 0.556 | 0.642 | 0.706 | 0.725 | **0.751** |

  The 0.80 target was passed at **epoch 4**. Mosaic makes the start slower: at epoch 10
  the final run was behind the 10-epoch experiment. It catches up on mAP@50 by epoch 30
  (0.954 vs 0.957), and its **mAP@50-95 ends clearly higher: 0.751 vs 0.710**, meaning
  tighter boxes. Training and validation losses fell together for all 40 epochs (val
  box loss 1.25 → 0.87, val cls loss 2.71 → 0.53), so there was no overfitting. The best
  epoch was the last one, so longer training would still help a little.

**Final settings:** 512 px, batch 16, default augmentation (mosaic included), 40 epochs,
early stopping patience 10. Training took 125 minutes on CPU.

---

## 5. Evaluation

`python training/evaluate.py` uses conf 0.001 and IoU 0.7 (the standard settings for
mAP). Results are in `results/evaluation/`.

| Set | Images | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Validation | 157 | 0.940 | 0.919 | 0.959 | 0.752 |
| **Test** | **78** | **0.975** | **0.931** | **0.972** | **0.776** |
| Clean test (no leakage) | 32 | 0.975 | 0.960 | 0.976 | 0.795 |

**Per class – test set**

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| helmet | 0.965 | 0.944 | 0.960 | 0.719 |
| boots  | 0.999 | 0.946 | 0.987 | 0.741 |
| person | 0.984 | 0.937 | 0.969 | 0.865 |
| vest   | 0.952 | 0.895 | 0.972 | 0.779 |

**Observations**

- **The target is met on every set**, including the clean test images the model has never
  seen in any form (0.976).
- **The leakage did not inflate the score.** The clean test set scores slightly *higher*
  than the full test set. With 32 images that is within noise, but it shows the model
  isn't relying on memorised duplicates.
- **The model rarely confuses classes.** In the normalised test confusion matrix
  (`results/evaluation/test/confusion_matrix_normalized.png`) the diagonal is helmet
  0.97, boots 1.00, person 0.98, vest 0.99. The only cross-class error is 1 % of vests
  predicted as helmets. Misses are rare (3 % of helmets, 2 % of persons). The main error
  is **false alarms on background**, mostly boots (37 % of all false positives), then
  vests (25 %) and helmets (20 %). These are low-confidence boxes. The app's default
  confidence of 0.35 hides most of them.
- **Helmets have the lowest mAP@50-95 (0.72).** They are found reliably, but small boxes
  are hard to fit tightly. Persons are the opposite: large and easy to fit (0.87).
- **Compared with the Day-31 coding-practice model** (416 px, 35 epochs: test mAP@50
  0.962, mAP@50-95 0.769), the experiment-driven settings improved both numbers.
- **The deployed ONNX file was scored separately** (`metrics_onnx.json`): test mAP@50
  0.968, mAP@50-95 0.778. The ±0.004 difference comes from the very low-confidence boxes
  (conf 0.001) that mAP counts. At the thresholds the app uses, the boxes are identical
  (next section).

---

## 6. Inference on test images

```
python training/predict.py              # all 78 test images
python training/predict.py --num 12 --conf 0.4
python training/predict.py --source my_photos/
```

For each image the script prints the objects and their confidence, for example
`image_190   5 objects: vest 0.89, person 0.85, boots 0.84, helmet 0.83, boots 0.81`.
It saves the following to `results/predictions/`:

| File | Content |
|---|---|
| `images/*.jpg` | 78 annotated images ("class 0.93" labels) |
| `predictions_grid.jpg` | 12 different photos on one sheet |
| `detections.csv` / `.json` | every box: image, class, confidence, x1 y1 x2 y2 |
| `ppe_report.csv` | per worker: helmet / vest / boots, status |
| `summary.json` | totals and speed |

**Results on the 78 test images (conf 0.25):** 334 detections, mean confidence 0.81, about
50 ms per image. Predicted counts are close to the labels (ground truth in brackets):
boots 115 (112), vest 88 (86), person 70 (65), helmet 61 (58). The PPE check found 70
workers, 43 of them fully equipped.

**What the images show:** the model works in daylight, indoors, at night, on rotated
photos and in crowded scenes. Typical mistakes are very small, far-away people, a glove
taken for a helmet at low confidence (0.62), and a white helmet missed in an advertising
photo.

`predict.py` uses the **same ONNX detector as the app**, so these results are exactly what
users get. `export_onnx.py` checks this: on 20 test images all **74/74** PyTorch boxes
come out of the ONNX detector with the same class, the same position and identical
confidence (largest difference 0.0000).

---

## 7. Streamlit app

```
streamlit run app.py
```

**Image tab:** upload an image (jpg, png, bmp, webp) or pick one of 8 sample photos. The
app shows:

- the original next to the detections
- the object count, number of workers, how many are fully equipped, and inference time
- a table of every detected object with a confidence bar
- the count per class and a PPE check per worker
- **downloads:** processed image (PNG), detections (CSV), full report (JSON)

**Video tab:** upload a video (mp4, mov, avi, mkv, webm) or use the demo video, choose how
many seconds to process and how often the model runs, then click **Run detection**. You
get:

- the processed video in the page
- a chart of objects per frame and the most objects seen in one frame
- **downloads:** processed video (MP4), counts per frame (CSV)

**Sidebar:** confidence threshold, IoU threshold, class filter, show/hide confidence, and
the model's test scores.

**Design choices**

- **ONNX Runtime instead of PyTorch.** Ultralytics pulls in PyTorch (over 1 GB). The app
  runs the exported ONNX model with `onnxruntime` instead, so the cloud install contains
  no PyTorch at all. The detector (`ppe_detector/detector.py`) re-implements YOLO's
  pre- and post-processing: letterbox, box decoding and class-aware NMS. It was verified
  box by box against Ultralytics.
- **Browser-playable video.** Browsers only play H.264. OpenCV's pip wheels can't write
  it, so frames are piped to the ffmpeg binary bundled with `imageio-ffmpeg` (libx264,
  yuv420p, faststart). No system ffmpeg is needed.
- **Fast enough on a free CPU.** "Run the model on every N-th frame" keeps the boxes on
  the frames in between, so the video keeps its length and frame rate at 2–5× the speed.
  Frames wider than 1280 px are scaled down. A 12 s clip takes about 6 s with N = 2.
- **Downloads don't re-run anything.** Results are kept in the session, and download
  buttons don't rerun the page.

**Tested:** with Streamlit's test harness and in a real browser, in a clean Python 3.11.9
environment built only from `requirements.txt` (no PyTorch, no Ultralytics). That covered
all 8 sample images, a real image upload, the demo video, and an odd-sized portrait `.mov`
at 30 fps (721 × 1283). Also checked: transparent PNG, grayscale, very wide images, and
empty or black frames, which produce no detections.

---

## 8. Deploying on Streamlit Community Cloud

1. Push the whole `Day-31/Mini_Project` folder. It must include `models/ppe_yolo11n.onnx`
   (10.6 MB) and `samples/`. `results/**/weights/`, `yolo11n.pt` and `dataset.yaml` are
   in `.gitignore`, and the app doesn't need them.
2. On https://share.streamlit.io choose **Create app → Deploy a public app from GitHub**,
   then set:
   - Repository: `MuhammadAwais005/MLB-Internship`, Branch: `main`
   - Main file path: `Day-31/Mini_Project/app.py`
   - Advanced settings → **Python 3.11**
3. Deploy. Community Cloud uses the `requirements.txt` **next to `app.py`**. It looks in
   the entrypoint's folder before the repository root, so another `requirements.txt` at
   the root won't interfere.

Notes:

- Community Cloud runs every app from the repository root. All file paths in `app.py`
  are built from the app's own folder, so the model and samples are found either way.
- Cloud reads `.streamlit/config.toml` only from the **repository root**. On Cloud, the
  orange theme from `Mini_Project/.streamlit/` is not applied and the default theme and
  200 MB upload limit are used. The app works the same. Copy the file to the root only
  if you want the theme for every app in the repo.

---

## 9. Limitations and next steps

- **Data.** 551 training photos from a limited number of sites, many with speckle-noise
  augmentation. Some photos have workers who aren't labelled as `person`. Expect weaker
  results on very different scenes, at night, or with tiny distant workers.
- **The PPE check is a rule, not a guarantee.** An item counts for a worker when its centre
  is inside that worker's box. When a person isn't detected, their items are counted but
  not assigned to anyone.
- **Next steps:**
  1. Train at 640 px for 80–100 epochs on a GPU. Accuracy was still rising.
  2. Try YOLO11s.
  3. Re-split the dataset by original photo to remove the leakage.
  4. Add background-only images to cut false boots.
  5. Add tracking (ByteTrack) so the video tab counts unique workers instead of workers
     per frame.
