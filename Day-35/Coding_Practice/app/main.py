"""
Day-35 - AI Inference API
FastAPI + custom YOLO model (PPE detector from Day-31).

Endpoints
    GET  /                -> welcome + endpoint list
    GET  /health          -> API running + model loaded?
    POST /predict         -> upload an image, get detections as JSON
    POST /predict/image   -> upload an image, get the image back with boxes drawn

Run (from the Coding_Practice folder):
    uvicorn app.main:app --reload
Swagger UI:
    http://127.0.0.1:8000/docs
"""

import logging
import os
import tempfile
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import config
from app.detector import YOLODetector
from app.schemas import ErrorResponse, HealthResponse, PredictionResponse
from app.validation import read_and_validate_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("inference-api")

detector = YOLODetector(config.MODEL_PATH, image_size=config.IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Model loading happens ONCE, when the server starts (lifespan event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        detector.load()
        log.info("Model loaded: %s in %.2fs, classes=%s",
                 config.MODEL_PATH.name, detector.load_time_seconds, detector.class_names)
    except Exception as exc:  # keep the API up so /health can report the problem
        log.error("Could not load the model: %s", exc)
    yield


app = FastAPI(
    title="YOLO PPE Detection API",
    description=(
        "Day-35: AI inference through an API. Upload an image and the custom "
        "YOLO11n PPE model (helmet, boots, person, vest) returns what it detects."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

ERRORS = {
    400: {"model": ErrorResponse, "description": "Empty, corrupted or wrong-sized image"},
    413: {"model": ErrorResponse, "description": "File too large"},
    415: {"model": ErrorResponse, "description": "Not a supported image type"},
    422: {"model": ErrorResponse, "description": "No file sent or bad query parameter"},
    503: {"model": ErrorResponse, "description": "Model not loaded"},
}


# ---------------------------------------------------------------------------
# Consistent error format: {"success": false, "status_code": ..., "error": "..."}
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "status_code": exc.status_code, "error": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    messages = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"][1:]) or str(err["loc"][0])
        if err["type"] == "missing" and field == "file":
            messages.append("No image uploaded. Send it as form-data with the key 'file'")
        else:
            messages.append(f"{field}: {err['msg']}")
    return JSONResponse(
        status_code=422,
        content={"success": False, "status_code": 422, "error": "; ".join(messages)},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_model():
    if not detector.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check /health and the server logs",
        )


async def _run_inference(file: UploadFile, conf: float, iou: float):
    """Validate the upload, save it to a temporary file, run the model, clean up."""
    _require_model()
    data, _ = await read_and_validate_image(file)

    # Save to a temporary file -> run YOLO on it -> always delete it afterwards
    suffix = Path(file.filename).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        detections, result, inference_ms = detector.predict(tmp.name, conf=conf, iou=iou)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    finally:
        tmp.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        await file.close()

    return detections, result, inference_ms


CONF_QUERY = Query(config.DEFAULT_CONFIDENCE, ge=0.01, le=1.0, description="Minimum confidence (0.01–1)")
IOU_QUERY = Query(config.DEFAULT_IOU, ge=0.1, le=0.95, description="NMS IoU threshold (0.1–0.95)")
FILE_PARAM = File(..., description="Image file (jpg, jpeg, png, bmp, webp), max 10 MB")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to the YOLO PPE Detection API",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict  (form-data key: file)",
            "predict_image": "POST /predict/image  (returns annotated JPEG)",
        },
    }


@app.get("/health", response_model=HealthResponse, responses={503: ERRORS[503]}, tags=["General"])
def health():
    """Confirms the API is running and whether the model is loaded."""
    body = {
        "status": "healthy" if detector.is_loaded else "unhealthy",
        "model_loaded": detector.is_loaded,
        "model_name": config.MODEL_PATH.name,
        "classes": detector.class_names,
        "model_load_time_seconds": detector.load_time_seconds,
    }
    if not detector.is_loaded:
        return JSONResponse(status_code=503, content=body)
    return body


@app.post("/predict", response_model=PredictionResponse, responses=ERRORS, tags=["Inference"])
async def predict(file: UploadFile = FILE_PARAM, conf: float = CONF_QUERY, iou: float = IOU_QUERY):
    """
    Upload an image → the model detects PPE → JSON with class, confidence,
    bounding box [x1, y1, x2, y2] and the total number of detections.
    """
    detections, result, inference_ms = await _run_inference(file, conf, iou)
    height, width = result.orig_shape
    return {
        "detections": detections,
        "total": len(detections),
        "counts": dict(Counter(d["class"] for d in detections)),
        "filename": file.filename,
        "image_size": {"width": width, "height": height},
        "inference_time_ms": inference_ms,
        "model": config.MODEL_PATH.name,
    }


@app.post(
    "/predict/image",
    responses={**ERRORS, 200: {"content": {"image/jpeg": {}}, "description": "Annotated image"}},
    response_class=Response,
    tags=["Inference"],
)
async def predict_image(file: UploadFile = FILE_PARAM, conf: float = CONF_QUERY, iou: float = IOU_QUERY):
    """Same as /predict, but returns the processed image with boxes drawn on it."""
    detections, result, _ = await _run_inference(file, conf, iou)
    ok, jpeg = cv2.imencode(".jpg", detector.draw(result), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the output image")
    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Total-Detections": str(len(detections)),
            "Content-Disposition": f'inline; filename="detected_{Path(file.filename).stem}.jpg"',
        },
    )
