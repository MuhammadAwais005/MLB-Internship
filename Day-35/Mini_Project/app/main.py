"""
Custom YOLO Prediction API - Day-35 Mini Project

Run from the Mini_Project folder:
    uvicorn app.main:app --reload
Swagger UI:
    http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.routes import prediction
from app.schemas.response import HealthResponse
from app.services.detector import YOLODetector, detector, get_detector

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("yolo-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup, not on every request
    try:
        detector.load()
        log.info("Model %s loaded in %.2fs | classes: %s",
                 detector.model_path.name, detector.load_time_seconds, detector.class_names)
    except Exception as exc:
        detector.load_error = str(exc)
        log.error("Model failed to load: %s", exc)  # API stays up; /health reports it
    yield


app = FastAPI(
    title="Custom YOLO Prediction API",
    description=(
        "Send an image, get the predictions of a custom-trained YOLO model.\n\n"
        "Model: `best.pt`, a YOLO11n trained on a PPE dataset "
        "(classes: helmet, boots, person, vest)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(prediction.router)


# ---------------------------------------------------------------------------
# One error format for everything:
#   {"success": false, "status_code": 415, "error": "...", "details": [...]}
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "status_code": exc.status_code, "error": exc.detail, "details": None},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"][1:]) or str(err["loc"][0])
        if field == "file" and err["type"] == "missing":
            details.append("file: no image was uploaded. Send it as form-data with the key 'file'")
        elif field == "file":
            details.append("file: must be an uploaded file (form-data type 'File'), not text")
        else:
            details.append(f"{field}: {err['msg']}")
    return JSONResponse(
        status_code=422,
        content={"success": False, "status_code": 422, "error": "Invalid request", "details": details},
    )


# ---------------------------------------------------------------------------
# General routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Custom YOLO Prediction API is running",
        "docs": "/docs",
        "endpoints": {
            "GET /health": "API and model status",
            "POST /predict": "upload an image -> JSON predictions",
            "POST /predict/image": "upload an image -> JPEG with bounding boxes",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"],
         responses={503: {"model": HealthResponse, "description": "Model not loaded"}})
def health(det: YOLODetector = Depends(get_detector)):
    body = {
        "status": "healthy" if det.is_loaded else "unhealthy",
        "model_loaded": det.is_loaded,
        "model_name": det.model_path.name,
        "classes": det.class_names,
        "model_load_time_seconds": det.load_time_seconds,
        "error": det.load_error,
    }
    return body if det.is_loaded else JSONResponse(status_code=503, content=body)
