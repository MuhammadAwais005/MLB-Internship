"""
Prediction routes. They only handle HTTP concerns (inputs, outputs, status codes);
all model work is delegated to app.services.detector.
"""

import json
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app import config
from app.schemas.response import ErrorResponse, PredictionResponse
from app.services.detector import PredictionResult, YOLODetector, get_detector
from app.services.image_validator import load_image_from_upload

router = APIRouter(tags=["Prediction"])

ERRORS = {
    400: {"model": ErrorResponse, "description": "Empty, corrupted or wrongly sized image, or unknown class"},
    413: {"model": ErrorResponse, "description": "File too large"},
    415: {"model": ErrorResponse, "description": "Unsupported file type"},
    422: {"model": ErrorResponse, "description": "Missing file or invalid parameter"},
    503: {"model": ErrorResponse, "description": "Model not loaded"},
}

# Shared parameters for both endpoints
FILE = File(..., description="Image to analyse (jpg, jpeg, png, bmp, webp, max 10 MB)")
CONF = Query(config.DEFAULT_CONFIDENCE, ge=0.01, le=1.0,
             description="Confidence threshold: only detections at or above this score are returned")
IOU = Query(config.DEFAULT_IOU, ge=0.1, le=0.95,
            description="IoU threshold for non-maximum suppression (removes duplicate boxes)")
CLASSES = Query(None, description="Optional comma-separated class names to keep, e.g. helmet,vest",
                examples=["helmet,vest"])


def _parse_classes(classes: Optional[str], det: YOLODetector) -> Optional[list[int]]:
    if not classes:
        return None
    name_to_id = {name.lower(): cid for cid, name in det.class_names.items()}
    wanted = [c.strip().lower() for c in classes.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in name_to_id]
    if unknown:
        raise HTTPException(
            400, f"Unknown class name(s): {', '.join(unknown)}. "
                 f"Available: {', '.join(det.class_names.values())}"
        )
    return [name_to_id[c] for c in wanted] or None


async def _detect(file: UploadFile, conf: float, iou: float, classes: Optional[str], det: YOLODetector):
    """Validate upload -> run model in a worker thread -> return (image, result)."""
    if not det.is_loaded:
        raise HTTPException(503, "Model is not loaded. Check GET /health for details")
    class_ids = _parse_classes(classes, det)
    image = await load_image_from_upload(file)
    try:
        # YOLO inference is CPU-heavy and blocking; run it off the event loop
        result = await run_in_threadpool(
            det.predict, image, conf, iou, class_ids, config.MAX_DETECTIONS
        )
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}")
    return image, result


def _message(result: PredictionResult, conf: float) -> str:
    if result.count == 0:
        return f"No objects detected at confidence >= {conf}"
    return f"{result.count} object{'s' if result.count != 1 else ''} detected"


@router.post("/predict", response_model=PredictionResponse, responses=ERRORS,
             summary="Detect objects and return JSON")
async def predict(
    file: UploadFile = FILE,
    conf: float = CONF,
    iou: float = IOU,
    classes: Optional[str] = CLASSES,
    det: YOLODetector = Depends(get_detector),
):
    """
    Upload an image and receive the model's predictions: class, confidence and
    bounding box for every object, plus the total count and a count per class.
    """
    image, result = await _detect(file, conf, iou, classes, det)
    return {
        "success": True,
        "message": _message(result, conf),
        "count": result.count,
        "counts_by_class": result.counts,
        "detections": [d.to_dict() for d in result.detections],
        "image": {"filename": file.filename, "width": result.image_width, "height": result.image_height},
        "confidence_threshold": conf,
        "iou_threshold": iou,
        "inference_time_ms": result.inference_ms,
        "model": det.model_path.name,
    }


@router.post(
    "/predict/image",
    response_class=Response,
    responses={**ERRORS, 200: {"content": {"image/jpeg": {}}, "description": "Image with boxes drawn"}},
    summary="Detect objects and return the processed image",
)
async def predict_image(
    file: UploadFile = FILE,
    conf: float = CONF,
    iou: float = IOU,
    classes: Optional[str] = CLASSES,
    det: YOLODetector = Depends(get_detector),
):
    """
    Same as /predict, but returns a JPEG with bounding boxes and labels drawn.
    The detection count is also sent in the `X-Detection-Count` response header.
    """
    image, result = await _detect(file, conf, iou, classes, det)
    annotated = det.annotate(image, result)
    ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "Could not encode the processed image")
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Detection-Count": str(result.count),
            "X-Counts-By-Class": json.dumps(result.counts),
            "X-Inference-Time-Ms": str(result.inference_ms),
            "Content-Disposition": f'inline; filename="detected_{Path(file.filename).stem}.jpg"',
        },
    )
