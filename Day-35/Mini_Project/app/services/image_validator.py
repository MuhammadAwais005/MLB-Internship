"""
Upload validation: turns an UploadFile into a decoded image, or raises a clear error.

Order of checks
    1. a file name with an allowed extension          -> 415
    2. an image content type (if the client sent one) -> 415
    3. not empty                                      -> 400
    4. not bigger than MAX_FILE_SIZE                  -> 413
    5. bytes really decode to an image                -> 400
    6. sensible dimensions                            -> 400
"""

from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from app import config


async def load_image_from_upload(file: UploadFile) -> np.ndarray:
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    allowed = ", ".join(config.ALLOWED_EXTENSIONS)

    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext or 'no extension'}'. Allowed types: {allowed}")

    ctype = (file.content_type or "").lower()
    if ctype and ctype != "application/octet-stream" and ctype not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported content type '{ctype}'. Allowed types: {allowed}")

    data = await file.read(config.MAX_FILE_SIZE + 1)
    await file.close()
    if not data:
        raise HTTPException(400, "The uploaded file is empty")
    if len(data) > config.MAX_FILE_SIZE:
        raise HTTPException(413, f"File is too large. Maximum size is {config.MAX_FILE_SIZE_MB:g} MB")

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "The file could not be read as an image. It may be corrupted or not a real image")

    h, w = image.shape[:2]
    if min(h, w) < config.MIN_IMAGE_SIDE:
        raise HTTPException(400, f"Image is too small ({w}x{h}). Minimum side is {config.MIN_IMAGE_SIDE} px")
    if max(h, w) > config.MAX_IMAGE_SIDE:
        raise HTTPException(400, f"Image is too large ({w}x{h}). Maximum side is {config.MAX_IMAGE_SIDE} px")

    return image
