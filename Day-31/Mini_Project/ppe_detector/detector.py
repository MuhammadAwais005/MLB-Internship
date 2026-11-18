"""
PPEDetector - runs the trained YOLO model through ONNX Runtime.

Why ONNX for the app and not Ultralytics?
    Ultralytics pulls in PyTorch (about 1 GB, more with CUDA). On Streamlit
    Community Cloud that means slow builds, a slow cold start and a real risk of
    running out of memory. The same trained weights, exported to ONNX, need only
    onnxruntime (~15 MB) and give the same boxes (checked in export_onnx.py).

What YOLO11's ONNX output looks like:
    one array of shape (1, 4 + number_of_classes, N), N candidate boxes
    rows 0-3 : box centre x, centre y, width, height, in pixels of the
               (letterboxed) network input
    rows 4.. : one score per class, already passed through a sigmoid
    No non-maximum suppression is done inside the model, so we do it here.
"""

import ast
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .settings import CLASS_NAMES


@dataclass
class Detection:
    class_id: int
    name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def contains(self, point):
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def to_dict(self):
        row = asdict(self)
        row["confidence"] = round(self.confidence, 4)
        return row


class PPEDetector:
    def __init__(self, model_path, threads=None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        # Ultralytics stores the image size and class names inside the ONNX file
        meta = self.session.get_modelmeta().custom_metadata_map
        height, width = self.session.get_inputs()[0].shape[2:4]
        self.input_size = (int(height), int(width))
        if "imgsz" in meta:
            self.input_size = tuple(ast.literal_eval(meta["imgsz"]))
        self.names = ast.literal_eval(meta["names"]) if "names" in meta else dict(CLASS_NAMES)

    # ------------------------------------------------------------------ steps
    def letterbox(self, image):
        """
        Resize keeping the aspect ratio, then pad to the network size with grey
        (114), exactly like Ultralytics does. Returns the padded image plus the
        scale and padding needed to map boxes back to the original image.
        """
        h, w = image.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_h / h, target_w / w)
        new_w, new_h = round(w * scale), round(h * scale)
        pad_w, pad_h = (target_w - new_w) / 2, (target_h - new_h) / 2

        if (new_w, new_h) != (w, h):
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
        left, right = round(pad_w - 0.1), round(pad_w + 0.1)
        padded = cv2.copyMakeBorder(image, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, scale, (left, top)

    def preprocess(self, image_bgr):
        padded, scale, pad = self.letterbox(image_bgr)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)   # HWC -> CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32)[None] / 255.0
        return blob, scale, pad

    @staticmethod
    def nms(boxes, scores, class_ids, iou_threshold):
        """
        Class-aware non-maximum suppression: boxes of different classes never
        suppress each other (a vest inside a person box must survive). The usual
        trick is to shift every class into its own coordinate range.
        """
        if len(boxes) == 0:
            return np.empty(0, dtype=int)
        shifted = boxes + class_ids[:, None] * 10_000.0
        x1, y1, x2, y2 = shifted.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_threshold]
        return np.array(keep, dtype=int)

    # ------------------------------------------------------------------ main
    def detect(self, image_bgr, conf=0.25, iou=0.45, classes=None, max_det=300):
        """
        image_bgr : numpy image as read by cv2.imread (BGR)
        conf      : minimum confidence to keep a box
        iou       : overlap above which a weaker box of the same class is dropped
        classes   : optional list of class ids to keep
        Returns a list of Detection, highest confidence first.
        """
        blob, scale, (pad_x, pad_y) = self.preprocess(image_bgr)
        output = self.session.run(None, {self.input_name: blob})[0][0].T    # (N, 4 + classes)

        class_scores = output[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]
        mask = scores >= conf
        if classes is not None:
            mask &= np.isin(class_ids, list(classes))
        if not mask.any():
            return []

        cx, cy, bw, bh = output[mask, :4].T
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        scores, class_ids = scores[mask], class_ids[mask]

        keep = self.nms(boxes, scores, class_ids, iou)[:max_det]

        # undo the letterbox: remove padding, then scale back to the original size
        h, w = image_bgr.shape[:2]
        boxes = boxes[keep]
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / scale).clip(0, w)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / scale).clip(0, h)

        return [
            Detection(int(c), self.names.get(int(c), str(c)), float(s), *(int(round(v)) for v in box))
            for box, s, c in zip(boxes, scores[keep], class_ids[keep])
        ]
