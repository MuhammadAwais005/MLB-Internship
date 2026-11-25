"""Pydantic response models: they define the JSON shape and document Swagger UI."""

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    x1: int = Field(..., description="Left edge (px)", examples=[120])
    y1: int = Field(..., description="Top edge (px)", examples=[80])
    x2: int = Field(..., description="Right edge (px)", examples=[340])
    y2: int = Field(..., description="Bottom edge (px)", examples=[420])
    width: int = Field(..., examples=[220])
    height: int = Field(..., examples=[340])


class DetectionOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(..., alias="class", examples=["helmet"])
    class_id: int = Field(..., examples=[0])
    confidence: float = Field(..., ge=0, le=1, examples=[0.91])
    bbox: BoundingBox


class ImageInfo(BaseModel):
    filename: str
    width: int
    height: int


class PredictionResponse(BaseModel):
    success: bool = True
    message: str = Field(..., examples=["3 objects detected"])
    count: int = Field(..., description="Total number of detections")
    counts_by_class: dict[str, int] = Field(..., examples=[{"helmet": 1, "vest": 2}])
    detections: list[DetectionOut]
    image: ImageInfo
    confidence_threshold: float
    iou_threshold: float
    inference_time_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    classes: dict[int, str]
    model_load_time_seconds: float | None = None
    error: str | None = None


class ErrorResponse(BaseModel):
    success: bool = False
    status_code: int
    error: str
    details: list[str] | None = None
