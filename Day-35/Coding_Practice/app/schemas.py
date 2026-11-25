"""Pydantic models describing the API responses (they also document Swagger UI)."""

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # "class" is a Python keyword, so the field is class_name and the JSON key is "class"
    class_name: str = Field(..., alias="class", examples=["helmet"])
    class_id: int = Field(..., examples=[0])
    confidence: float = Field(..., ge=0, le=1, examples=[0.91])
    bbox: list[int] = Field(
        ..., min_length=4, max_length=4,
        description="[x1, y1, x2, y2] in pixels of the original image",
        examples=[[120, 80, 340, 420]],
    )


class ImageSize(BaseModel):
    width: int
    height: int


class PredictionResponse(BaseModel):
    detections: list[Detection]
    total: int
    counts: dict[str, int] = Field(..., description="Number of detections per class")
    filename: str
    image_size: ImageSize
    inference_time_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    classes: dict[int, str]
    model_load_time_seconds: float | None


class ErrorResponse(BaseModel):
    success: bool = False
    status_code: int
    error: str
