# Day-35 – Mini Project: Custom YOLO Prediction API

A REST API that serves my **custom-trained YOLO model**. Another application sends an image and gets back the model's predictions as JSON, or the image with the bounding boxes drawn on it.

**Model:** `models/best.pt`, the YOLO11n PPE detector I trained myself (Day-31). It detects 4 classes: `helmet`, `boots`, `person` and `vest`.
The API reads the class names from the model, so any other YOLO `.pt` file works too. Replace `best.pt`, or set the `MODEL_PATH` environment variable.

## Project structure

```
Mini_Project/
├── app/
│   ├── main.py                  # creates the app, loads the model at startup, /health, error handlers
│   ├── config.py                # settings (model path, thresholds, upload limits)
│   ├── routes/
│   │   └── prediction.py        # HTTP layer only: /predict and /predict/image
│   ├── services/
│   │   ├── detector.py          # model + inference logic (no FastAPI code)
│   │   └── image_validator.py   # upload checks
│   └── schemas/
│       └── response.py          # Pydantic response models
├── models/
│   └── best.pt
├── samples/                     # test images + test_files/ (empty, .txt, fake .jpg)
├── tests/test_api.py            # 22 automated tests
├── YOLO_Prediction_API.postman_collection.json
├── requirements.txt
└── README.md
```

**Separation of concerns:** `services/detector.py` only knows about images and YOLO. It never imports FastAPI. `routes/prediction.py` only deals with HTTP: inputs, outputs and status codes. It hands every piece of model work to the detector service. That means the model could be reused in a script or a Streamlit app without touching the API.

## Endpoints

| Method | Endpoint          | Returns                                         |
|--------|-------------------|-------------------------------------------------|
| GET    | `/`               | Welcome message                                 |
| GET    | `/health`         | API status and whether the model is loaded      |
| POST   | `/predict`        | JSON predictions                                |
| POST   | `/predict/image`  | JPEG with bounding boxes and labels drawn       |

**Request:** send the image as **form-data** with the key **`file`**.

**Optional query parameters:**

| Parameter | Default | Range      | Meaning                                        |
|-----------|---------|------------|------------------------------------------------|
| `conf`    | 0.25    | 0.01 – 1   | Confidence threshold                           |
| `iou`     | 0.45    | 0.1 – 0.95 | Overlap threshold for removing duplicate boxes |
| `classes` | all     | names      | Keep only these classes, e.g. `helmet,vest`    |

### `POST /predict` – example response

```json
{
  "success": true,
  "message": "5 objects detected",
  "count": 5,
  "counts_by_class": {"person": 1, "boots": 2, "vest": 1, "helmet": 1},
  "detections": [
    {
      "class": "person",
      "class_id": 2,
      "confidence": 0.9562,
      "bbox": {"x1": 250, "y1": 87, "x2": 492, "y2": 484, "width": 242, "height": 397}
    }
  ],
  "image": {"filename": "rooftop_red_vest.jpg", "width": 640, "height": 640},
  "confidence_threshold": 0.5,
  "iou_threshold": 0.45,
  "inference_time_ms": 98.4,
  "model": "best.pt"
}
```
`bbox` is in pixels of the original image. `(x1, y1)` is the top-left corner and `(x2, y2)` is the bottom-right.
When nothing is found, you get `"count": 0`, `"detections": []` and `"message": "No objects detected at confidence >= 0.25"`.

### `POST /predict/image`

Returns `image/jpeg` with the boxes, labels and a "Detections: N" banner drawn on it. The counts are also in the response headers: `X-Detection-Count`, `X-Counts-By-Class` and `X-Inference-Time-Ms`.

## Error messages

Every error has the same shape:

```json
{"success": false, "status_code": 415, "error": "Unsupported file type '.txt'. Allowed types: .jpg, .jpeg, .png, .bmp, .webp", "details": null}
```

| Code | Cause |
|------|-------|
| 400 | Empty file, corrupted or fake image, image too small or large, unknown class name |
| 413 | File larger than 10 MB |
| 415 | Unsupported file type or content type |
| 422 | No file, file sent as text, or `conf`/`iou` out of range (see `details`) |
| 500 | Inference failed |
| 503 | Model not loaded (`/health` shows the reason) |

## Setup & run

```bash
cd Day-35/Mini_Project
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The log shows `Model best.pt loaded in ~1.4s`. The model is loaded **once** at startup, and each prediction then takes about 100 ms on CPU.

## API testing

### Test results

| Test case                      | File                                   | Result |
|--------------------------------|----------------------------------------|--------|
| Valid image                    | `samples/rooftop_red_vest.jpg`         | 200, 5 detections: person, helmet, vest, 2× boots |
| Unsupported file type          | `samples/test_files/not_an_image.txt`  | 415, "Unsupported file type '.txt'" |
| Fake image (text renamed .jpg) | `samples/test_files/fake_image.jpg`    | 400, "could not be read as an image" |
| Empty file                     | `samples/test_files/empty.jpg`         | 400, "The uploaded file is empty" |
| Image with no objects          | `samples/no_objects.jpg`               | 200, `count: 0`, "No objects detected" |
| conf = 0.1                     | `samples/group_handover.jpg`           | 5 detections (2 helmet, 1 person, 2 boots) |
| conf = 0.25                    | same                                   | 2 detections (2 helmet) |
| conf = 0.5                     | same                                   | 1 detection (1 helmet) |
| conf = 0.9                     | same                                   | 0 detections |
| conf = 2                       | same                                   | 422, out of range |

A higher confidence threshold returns fewer, but more certain, detections.

### Swagger UI
1. Open http://127.0.0.1:8000/docs.
2. **POST /predict** → **Try it out** → choose a file → set `conf` → **Execute**.
3. **POST /predict/image** → the processed image appears right in the browser.

### Postman / Thunder Client
Import `YOLO_Prediction_API.postman_collection.json`. It has 17 requests, organised into folders that match the test cases above.
After importing, re-select each file under **Body → form-data → file**, because Postman stores file paths per computer.

### Automated tests
```bash
pytest -v
```
22 tests cover every case in the table above, plus PNG upload, the class filter, the processed image, a missing file and the file-size limit.
