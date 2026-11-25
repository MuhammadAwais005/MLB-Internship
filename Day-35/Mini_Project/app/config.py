"""
Settings for the API. Every value can be overridden with an environment
variable of the same name, e.g.  set MODEL_PATH=models\\other.pt
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Model
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "best.pt"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE", "0.25"))
DEFAULT_IOU = float(os.getenv("DEFAULT_IOU", "0.45"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "300"))

# Upload rules
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE = int(MAX_FILE_SIZE_MB * 1024 * 1024)
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ALLOWED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/bmp", "image/x-ms-bmp", "image/webp")
MIN_IMAGE_SIDE = 32
MAX_IMAGE_SIDE = 8000
