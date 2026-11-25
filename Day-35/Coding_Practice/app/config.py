"""
All settings in one place.
Any value can be overridden with an environment variable of the same name.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Model ----------------------------------------------------------------------
# Custom PPE detector trained on Day-31 (YOLO11n, classes: helmet, boots, person, vest)
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "ppe_yolo11n.pt"))
DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE", "0.25"))
DEFAULT_IOU = float(os.getenv("DEFAULT_IOU", "0.45"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

# Upload validation ----------------------------------------------------------
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE = int(MAX_FILE_SIZE_MB * 1024 * 1024)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/x-ms-bmp"}
MIN_IMAGE_SIDE = 32       # pixels
MAX_IMAGE_SIDE = 8000     # pixels
