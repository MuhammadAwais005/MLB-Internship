"""
Model / inference logic. This file knows nothing about FastAPI or HTTP:
it takes a numpy image and returns plain Python data. The routes call it.

    load()      -> read best.pt once (slow, done at startup)
    predict()   -> run one image through the loaded model (fast, every request)
    annotate()  -> draw the boxes on the image
"""

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from app import config

# One colour per class id (BGR), cycled if the model has more classes
PALETTE = [
    (0, 165, 255), (180, 105, 255), (255, 160, 0), (0, 220, 120),
    (60, 60, 255), (255, 255, 0), (200, 0, 200), (0, 255, 255),
]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "width": self.x2 - self.x1, "height": self.y2 - self.y1,
            },
        }


@dataclass
class PredictionResult:
    detections: list[Detection]
    image_width: int
    image_height: int
    inference_ms: float
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.detections)


class YOLODetector:
    def __init__(self, model_path: Path, image_size: int = 640):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.model: YOLO | None = None
        self.class_names: dict[int, str] = {}
        self.load_time_seconds: float | None = None
        self.load_error: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load the weights once and warm the model up."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        start = time.perf_counter()
        self.model = YOLO(str(self.model_path))
        self.class_names = {int(k): str(v) for k, v in self.model.names.items()}
        warmup = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.model.predict(warmup, imgsz=self.image_size, verbose=False)
        self.load_time_seconds = round(time.perf_counter() - start, 3)
        self.load_error = None

    # ---------------------------------------------------------------- inference
    def predict(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: list[int] | None = None,
        max_det: int = 300,
    ) -> PredictionResult:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()
        with self._lock:  # the model object is shared between requests
            result = self.model.predict(
                image_bgr, conf=conf, iou=iou, classes=classes, max_det=max_det,
                imgsz=self.image_size, verbose=False,
            )[0]
        inference_ms = round((time.perf_counter() - start) * 1000, 2)

        detections = []
        for box in result.boxes:
            cid = int(box.cls)
            x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(cid, self.class_names.get(cid, str(cid)), float(box.conf), x1, y1, x2, y2)
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)

        h, w = image_bgr.shape[:2]
        return PredictionResult(
            detections=detections,
            image_width=w,
            image_height=h,
            inference_ms=inference_ms,
            counts=dict(Counter(d.class_name for d in detections)),
        )

    # --------------------------------------------------------------- drawing
    @staticmethod
    def annotate(image_bgr: np.ndarray, result: PredictionResult) -> np.ndarray:
        """Draw boxes, labels and a detection count on a copy of the image."""
        out = image_bgr.copy()
        h, w = out.shape[:2]
        thickness = max(2, round(min(h, w) / 300))
        font_scale = max(0.5, min(h, w) / 1000)

        for d in result.detections:
            color = PALETTE[d.class_id % len(PALETTE)]
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, thickness)
            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            ty = d.y1 - 4 if d.y1 - th - base - 4 > 0 else d.y1 + th + base + 4
            cv2.rectangle(out, (d.x1, ty - th - base), (d.x1 + tw + 4, ty + base // 2), color, -1)
            cv2.putText(out, label, (d.x1 + 2, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

        banner = f"Detections: {result.count}"
        (tw, th), base = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, thickness)
        cv2.rectangle(out, (0, 0), (tw + 20, th + base + 16), (0, 0, 0), -1)
        cv2.putText(out, banner, (10, th + 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1,
                    (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)
        return out


# A single shared detector for the whole app
detector = YOLODetector(model_path=config.MODEL_PATH, image_size=config.IMAGE_SIZE)


def get_detector() -> YOLODetector:
    """FastAPI dependency: routes ask for the detector instead of importing a global."""
    return detector
