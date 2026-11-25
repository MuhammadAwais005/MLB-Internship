"""
YOLO model wrapper.

Model LOADING vs model INFERENCE
--------------------------------
* Loading = reading the .pt weights from disk and building the network.
  It is slow (about a second) and uses memory, so we do it ONCE when the API
  starts, not on every request.
* Inference = running an image through the already-loaded network.
  It is fast (tens of milliseconds) and happens on every /predict call.
"""

import threading
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path: Path, image_size: int = 640):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.model = None
        self.class_names: dict[int, str] = {}
        self.load_time_seconds: float | None = None
        self._lock = threading.Lock()  # one inference at a time on the shared model

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load the weights and run one warm-up prediction."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        start = time.perf_counter()
        self.model = YOLO(str(self.model_path))
        self.class_names = {int(k): str(v) for k, v in self.model.names.items()}

        # Warm-up: the very first prediction is slow (memory allocation, setup).
        # Doing it now means the first real user request is fast.
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.image_size, verbose=False)

        self.load_time_seconds = round(time.perf_counter() - start, 3)

    def predict(self, source, conf: float, iou: float):
        """
        Run inference.
        `source` can be a file path or a numpy BGR image.
        Returns (detections, ultralytics_result, inference_ms).
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()
        with self._lock:
            result = self.model.predict(
                source, conf=conf, iou=iou, imgsz=self.image_size, verbose=False
            )[0]
        inference_ms = round((time.perf_counter() - start) * 1000, 2)

        detections = []
        for box in result.boxes:
            class_id = int(box.cls)
            x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
            detections.append(
                {
                    "class": self.class_names.get(class_id, str(class_id)),
                    "class_id": class_id,
                    "confidence": round(float(box.conf), 2),
                    "bbox": [x1, y1, x2, y2],
                }
            )
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections, result, inference_ms

    @staticmethod
    def draw(result) -> np.ndarray:
        """Return the image with boxes and labels drawn on it (BGR numpy array)."""
        return result.plot(line_width=2)
