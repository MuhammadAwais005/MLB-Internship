# Day-35 – Coding Practice: AI Inference API (FastAPI + YOLO)

An API that serves my **custom YOLO11n PPE detection model** (trained on Day-31).
Instead of running the model from a notebook or a script, any app can now send an image over HTTP and get the detections back as JSON.

**Model:** `models/ppe_yolo11n.pt`, which detects 4 classes: `helmet`, `boots`, `person` and `vest`.

## Project structure

```
Coding_Practice/
├── app/
│   ├── main.py          # FastAPI app: routes, model loading at startup, error handlers
│   ├── detector.py      # YOLODetector: load() once, predict() per request, draw()
│   ├── validation.py    # upload checks (type, size, real image, dimensions)
│   ├── schemas.py       # Pydantic response models
│   └── config.py        # settings (model path, thresholds, size limits)
├── models/ppe_yolo11n.pt
├── samples/             # test images
├── tests/test_api.py    # 17 automated tests
├── YOLO_Inference_API.postman_collection.json
├── requirements.txt
└── README.md
```

## Endpoints

| Method | Endpoint          | What it does                                               |
|--------|-------------------|------------------------------------------------------------|
| GET    | `/health`         | Confirms the API is running **and** the model is loaded    |
| POST   | `/predict`        | Upload an image and get the detections as JSON              |
| POST   | `/predict/image`  | Upload an image and get it back with the boxes drawn (JPEG) |
| GET    | `/`               | Welcome message                                            |

Both predict endpoints take the image as **form-data** with the key **`file`**, plus two optional query parameters:
- `conf` – minimum confidence, 0.01–1 (default 0.25)
- `iou` – NMS overlap threshold, 0.1–0.95 (default 0.45)

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "ppe_yolo11n.pt",
  "classes": {"0": "helmet", "1": "boots", "2": "person", "3": "vest"},
  "model_load_time_seconds": 1.404
}
```
If the model fails to load, this returns **503** with `"model_loaded": false`.

### `POST /predict`

```json
{
  "detections": [
    {"class": "helmet", "class_id": 0, "confidence": 0.89, "bbox": [375, 56, 459, 156]},
    {"class": "helmet", "class_id": 0, "confidence": 0.29, "bbox": [69, 84, 114, 137]}
  ],
  "total": 2,
  "counts": {"helmet": 2},
  "filename": "group_handover.jpg",
  "image_size": {"width": 640, "height": 640},
  "inference_time_ms": 90.84,
  "model": "ppe_yolo11n.pt"
}
```
`bbox` is `[x1, y1, x2, y2]` in pixels of the original image (top-left and bottom-right corners).

## Key concepts

**Model loading vs inference**
- *Loading* reads the `.pt` weights and builds the network. It takes about 1.4 s, so it happens **once at startup**, in FastAPI's `lifespan` event. A warm-up prediction runs at the same time, so the first real request is fast too.
- *Inference* runs one image through the loaded model, which takes about 90 ms on CPU. This happens on **every** request.
- Loading the model inside `/predict` would make every request about 15 times slower.

**Handling an upload** (`UploadFile`)
1. Check the extension: `.jpg .jpeg .png .bmp .webp`. Anything else gets **415**.
2. Check the content type. Anything that isn't an image type gets **415**.
3. Check the size: an empty file gets **400**, and anything over 10 MB gets **413**.
4. Decode the bytes with OpenCV. Fake or corrupted images get **400**. A `.txt` renamed to `.jpg` is caught here.
5. Check the dimensions: smaller than 32 px or larger than 8000 px gets **400**.
6. Save the image to a **temporary file**, run YOLO, and always delete the file in `finally`.

**Error format** – every error looks the same:
```json
{"success": false, "status_code": 415, "error": "Unsupported file type '.txt'. Allowed: .bmp, .jpeg, .jpg, .png, .webp"}
```

| Code | When |
|------|------|
| 400 | Empty, corrupted, or too small/large image |
| 413 | File over 10 MB |
| 415 | Not a supported image type |
| 422 | No file sent, or `conf`/`iou` out of range |
| 500 | Model crashed during inference |
| 503 | Model not loaded |

## Setup & run

```bash
cd Day-35/Coding_Practice
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The logs show `Model loaded: ppe_yolo11n.pt in 1.40s`, and the API runs at http://127.0.0.1:8000.

## Testing

### Swagger UI
1. Open http://127.0.0.1:8000/docs.
2. Open **GET /health** → **Try it out** → **Execute**.
3. Open **POST /predict** → **Try it out** → choose an image from `samples/` → **Execute**.
4. Try **POST /predict/image** to see the image with the boxes drawn.

### Postman / Thunder Client
Import `YOLO_Inference_API.postman_collection.json`. It has 7 requests: health, predict, predict with confidence, annotated image, and 3 invalid-input cases.
For the file requests, re-select the image under **Body → form-data → file**, because Postman stores file paths per computer.

Manual setup, if you prefer: **POST** `http://127.0.0.1:8000/predict` → **Body** → **form-data** → key `file` (type **File**) → choose an image → **Send**.

### curl
```bash
curl -X POST "http://127.0.0.1:8000/predict?conf=0.4" -F "file=@samples/factory_worker.jpg"
curl -X POST "http://127.0.0.1:8000/predict/image" -F "file=@samples/rooftop_red_vest.jpg" --output result.jpg
```

### Automated tests
```bash
pytest -v
```
