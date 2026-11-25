"""
Upload validation.

Checks, in order:
1. A file name with an allowed extension
2. An allowed content type (if the client sent one)
3. Not empty, not larger than MAX_FILE_SIZE
4. The bytes really are an image (a renamed .txt fails here)
5. Sensible image dimensions
"""

from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app import config


def _error(code: int, message: str) -> HTTPException:
    return HTTPException(status_code=code, detail=message)


async def read_and_validate_image(file: UploadFile) -> tuple[bytes, np.ndarray]:
    """Return the raw bytes and the decoded image (BGR numpy array), or raise an HTTPException."""
    # 1. extension
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise _error(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{ext or 'none'}'. "
            f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
        )

    # 2. content type (some clients send application/octet-stream, so allow that)
    ctype = (file.content_type or "").lower()
    if ctype and ctype != "application/octet-stream" and ctype not in config.ALLOWED_CONTENT_TYPES:
        raise _error(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported content type '{ctype}'. Please upload a JPG, PNG, BMP or WEBP image",
        )

    # 3. size (read one byte more than the limit so we can tell if it is too big)
    data = await file.read(config.MAX_FILE_SIZE + 1)
    if not data:
        raise _error(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty")
    if len(data) > config.MAX_FILE_SIZE:
        raise _error(
            413,
            f"File is too large. Maximum size is {config.MAX_FILE_SIZE_MB:g} MB",
        )

    # 4. real image check: OpenCV returns None if the bytes are not a decodable image
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise _error(status.HTTP_400_BAD_REQUEST, "The file is not a valid image or is corrupted")

    # 5. dimensions
    h, w = image.shape[:2]
    if min(w, h) < config.MIN_IMAGE_SIDE:
        raise _error(
            status.HTTP_400_BAD_REQUEST,
            f"Image is too small ({w}x{h}). Minimum side is {config.MIN_IMAGE_SIDE} px",
        )
    if max(w, h) > config.MAX_IMAGE_SIDE:
        raise _error(
            status.HTTP_400_BAD_REQUEST,
            f"Image is too large ({w}x{h}). Maximum side is {config.MAX_IMAGE_SIDE} px",
        )

    return data, image
